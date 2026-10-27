//! Validated shared witness kernels and bounded native arithmetic chains.
use std::borrow::Cow;
use std::collections::HashMap;
#[allow(unused_imports)]
use std::sync::Arc;
use std::sync::OnceLock;

use anyhow::{ensure, Result};
use plonky2_maybe_rayon::*;

use super::cpu::{chunk_end, Executor};
use super::{Group, Kind, RegionProgram};

mod generated {
    include!("native_kernels.rs");
}

#[derive(Clone, Debug)]
pub(super) struct FastLayout {
    pub single: Cow<'static, [u8]>,
    pub sparse_ready: Option<Vec<u32>>,
    pub poseidon: OnceLock<HashMap<u32, Vec<u32>>>,
    pub adaptive: OnceLock<HashMap<u32, Vec<u8>>>,
    pub random_spans: OnceLock<HashMap<u32, (u32, u32)>>,
    /// Column-major cell to state slot table on the device.
    #[cfg(feature = "gpu")]
    pub wire_slots: crate::gpu::DeviceCache,
    #[cfg(feature = "gpu")]
    pub gpu_tail: OnceLock<Arc<super::gpu_tail::Tail>>,
}

#[cfg(feature = "witness-benchmarks")]
pub(crate) fn layout() -> u8 {
    super::super::WitnessConfig::from_env()
        .unwrap_or_default()
        .compilation_key() as u8
}

#[cfg(feature = "witness-benchmarks")]
pub(crate) fn file_stem(digest: &[u8]) -> String {
    let hex = digest
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    format!("circuit_{}_{hex}", layout())
}

pub(super) fn fast_enabled(p: &RegionProgram) -> bool {
    p.runtime.config.policy.execution != super::super::policy::WitnessExecution::Checked
}

pub(super) fn execute(executor: &Executor<'_>) -> Result<bool> {
    if !fast_enabled(executor.program) {
        return Ok(false);
    }
    generated::run(executor)?;
    Ok(true)
}

pub(super) fn run_fixed<const K: u8>(executor: &Executor<'_>, group: &Group) -> Result<()> {
    #[cfg(all(feature = "gpu", feature = "parallel"))]
    if let Some(tail) = executor.gpu_tail {
        return tail.cpu_ranges(group, 0, group.count, |a, b| {
            run_range_inner(executor, group, a, b)
        });
    }

    let kind = Kind::from_code(K as usize)?;
    let use_simd = K == 0 && poseidon_mode() != 0 && group.count >= poseidon_min_group();
    let chunk = if use_simd {
        poseidon_chunk()
    } else {
        super::cpu::lane_chunk(kind)
    };
    if K == 17 && group.count >= 512 {
        if let Some(&(source, destination)) =
            random_spans(executor.program).get(&group.targets_offset)
        {
            return (0..group.count.div_ceil(chunk))
                .into_par_iter()
                .try_for_each(|c| {
                    let start = c * chunk;
                    let end = chunk_end(c * chunk, chunk, group.count);
                    let _profile = executor.profile.kind(K, end - start);
                    executor.step_random_span(source, destination, start, end)
                });
        }
    }
    if K == 1 {
        return (0..group.count.div_ceil(chunk))
            .into_par_iter()
            .try_for_each(|c| {
                let _profile = executor
                    .profile
                    .kind(K, (chunk_end(c * chunk, chunk, group.count)) - c * chunk);
                arithmetic_simplified(
                    executor,
                    group,
                    c * chunk,
                    chunk_end(c * chunk, chunk, group.count),
                )
            });
    }
    if K == 4 {
        return (0..group.count.div_ceil(chunk))
            .into_par_iter()
            .try_for_each(|c| {
                let _profile = executor
                    .profile
                    .kind(K, (chunk_end(c * chunk, chunk, group.count)) - c * chunk);
                equality_batch(
                    executor,
                    group,
                    c * chunk,
                    chunk_end(c * chunk, chunk, group.count),
                )
            });
    }
    if K == 0 {
        let bases = poseidon_bases(executor.program);
        let starts = &bases[&group.targets_offset];
        return (0..group.count.div_ceil(chunk))
            .into_par_iter()
            .try_for_each(|c| {
                let begin = c * chunk;
                let end = chunk_end(c * chunk, chunk, group.count);
                let _profile = executor.profile.kind(K, end - begin);
                let mode = if use_simd { poseidon_mode() } else { 0 };
                if mode != 0 {
                    return executor.step_poseidon_simd(group, begin, end, starts, mode);
                }
                for k in begin..end {
                    if starts[k as usize] != u32::MAX {
                        executor.step_poseidon_bulk(
                            k,
                            group.count,
                            group.targets_offset,
                            starts[k as usize],
                        )?;
                    } else {
                        executor.step_fixed::<K>(group, k)?;
                    }
                }
                Ok(())
            });
    }
    (0..group.count.div_ceil(chunk))
        .into_par_iter()
        .try_for_each(|c| {
            let _profile = executor
                .profile
                .kind(K, (chunk_end(c * chunk, chunk, group.count)) - c * chunk);
            for k in c * chunk..chunk_end(c * chunk, chunk, group.count) {
                executor.step_fixed::<K>(group, k)?;
            }
            Ok(())
        })
}

fn random_spans(p: &RegionProgram) -> &HashMap<u32, (u32, u32)> {
    let fast = p.fast.get().expect("certified plan");
    fast.random_spans.get_or_init(|| {
        let mut spans = HashMap::new();
        let mut eligible = 0u64;
        for group in p.groups.iter().flatten().filter(|g| g.kind == Kind::RandomValue && g.count >= 512) {
            let co = group.coefficients_offset().expect("random coefficient storage") as usize;
            let Some(coefficients) = p.coefficients.get(co..).and_then(|s| s.get(..group.count as usize)) else {
                continue;
            };
            let targets = &p.targets[group.targets_offset as usize..][..group.count as usize];
            let source = coefficients[0];
            let destination = targets[0];
            if source + group.count as u64 <= p.random_count as u64
                && targets.iter().enumerate().all(|(i, &slot)| {
                    slot as u64 == destination as u64 + i as u64
                        && fast.single[slot as usize] != 0
                        && coefficients[i] == source + i as u64
                })
            {
                spans.insert(group.targets_offset, (source as u32, destination));
                eligible += group.count as u64;
            }
        }
        log::debug!(target: super::COMPILE_LOG, "contiguous random copies: {eligible}/{}", p.random_count);
        spans
    })
}

fn arithmetic_simplified(e: &Executor<'_>, g: &Group, start: u32, end: u32) -> Result<()> {
    use crate::field::goldilocks_field::GoldilocksField as G;
    use crate::field::types::Field;
    let co = g.coefficients_offset()?;
    let scale = |x: G, c: G| {
        if c == G::ONE {
            x
        } else if c == G::NEG_ONE {
            -x
        } else if c == G::TWO {
            x + x
        } else {
            x * c
        }
    };
    for k in start..end {
        let c0 = G::from_noncanonical_u64(e.coefficient(co, 0, g.count, k)?);
        let c1 = G::from_noncanonical_u64(e.coefficient(co, 1, g.count, k)?);
        let product = if c0 == G::ZERO {
            G::ZERO
        } else {
            scale(
                e.input(g.targets_offset, 0, g.count, k)?
                    * e.input(g.targets_offset, 1, g.count, k)?,
                c0,
            )
        };
        let addend = if c1 == G::ZERO {
            G::ZERO
        } else {
            scale(e.input(g.targets_offset, 2, g.count, k)?, c1)
        };
        e.output(g.targets_offset, 3, g.count, k, product + addend)?;
    }
    Ok(())
}

fn equality_batch(e: &Executor<'_>, g: &Group, start: u32, end: u32) -> Result<()> {
    use crate::field::goldilocks_field::GoldilocksField as G;
    use crate::field::types::Field;
    let mut differences = Vec::with_capacity((end - start) as usize);
    let mut nonzero = Vec::with_capacity((end - start) as usize);
    for k in start..end {
        let d =
            e.input(g.targets_offset, 0, g.count, k)? - e.input(g.targets_offset, 1, g.count, k)?;
        differences.push(d);
        if d != G::ZERO {
            nonzero.push(d);
        }
    }
    let inverses = G::batch_multiplicative_inverse(&nonzero);
    let mut index = 0;
    for (k, d) in (start..end).zip(differences) {
        let (flag, inverse) = if d == G::ZERO {
            (G::ONE, G::ZERO)
        } else {
            let inverse = inverses[index];
            index += 1;
            (G::ZERO, inverse)
        };
        e.output(g.targets_offset, 2, g.count, k, flag)?;
        e.output(g.targets_offset, 3, g.count, k, inverse)?;
    }
    Ok(())
}

fn poseidon_bases(p: &RegionProgram) -> &HashMap<u32, Vec<u32>> {
    p.fast
        .get()
        .expect("validated fast plan")
        .poseidon
        .get_or_init(|| {
            let mut groups = HashMap::new();
            let mut eligible = 0;
            let mut total = 0;
            for g in p
                .groups
                .iter()
                .flatten()
                .filter(|g| g.kind == Kind::Poseidon)
            {
                let mut starts = Vec::with_capacity(g.count as usize);
                for k in 0..g.count {
                    let base = p.targets[(g.targets_offset + 13 * g.count + k) as usize];
                    let contiguous = (0..122).all(|j| {
                        let slot = p.targets[(g.targets_offset + (13 + j) * g.count + k) as usize];
                        slot == base + j && p.fast.get().unwrap().single[slot as usize] != 0
                    });
                    starts.push(if contiguous { base } else { u32::MAX });
                    eligible += u64::from(contiguous);
                    total += 1;
                }
                groups.insert(g.targets_offset, starts);
            }
            log::debug!(target: super::COMPILE_LOG,"Poseidon contiguous outputs: {eligible}/{total}");
            groups
        })
}

/// Confirm that the topology certificate covers the executor's actual operands.
/// Arity metadata alone is insufficient when reading a damaged serialized plan.
fn validate_shape(p: &RegionProgram, g: &Group) -> Result<()> {
    let coefficient = |pos: u64, lane: u32| -> Result<u64> {
        let index = (g.coefficients_offset()? as u64)
            .checked_add(
                pos.checked_mul(g.count as u64)
                    .ok_or_else(|| anyhow::anyhow!("coefficient shape overflow"))?,
            )
            .and_then(|i| i.checked_add(lane as u64))
            .ok_or_else(|| anyhow::anyhow!("coefficient shape overflow"))?;
        p.coefficients
            .get(index as usize)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("invalid coefficient shape"))
    };
    let declared = (g.inputs as u64, g.outputs as u64);
    let expected = match g.kind {
        Kind::Poseidon => (13, 122),
        Kind::Arithmetic => (3, 1),
        Kind::Copy => (1, 1),
        Kind::Constant | Kind::RandomValue => (0, 1),
        Kind::Equality => (2, 2),
        Kind::ExtensionArithmetic => (6, 2),
        Kind::ExtensionMultiply | Kind::ExtensionQuotient => (4, 2),
        Kind::EcAddXu => (26, 90),
        Kind::PoseidonMds => (24, 24),
        Kind::ConditionalZero => (2, 1),
        Kind::QuinticMultiply | Kind::QuinticQuotient => (10, 5),
        Kind::EcAddHomogOffset => (20, 20),
        Kind::EcScalarMul => (10, 10),
        Kind::LowHigh => (1, 2),
        Kind::BitSum => {
            ensure!(g.inputs <= 63, "invalid bit sum width");
            (g.inputs as u64, 1)
        }
        Kind::BaseSplit | Kind::WireSplit => {
            let parameter = g.parameter()?;
            if g.kind == Kind::BaseSplit {
                ensure!(parameter & 255 < 64, "invalid base split width");
            }
            (1, (parameter >> 8) as u64)
        }
        Kind::RandomAccess => {
            let size = g.parameter()?;
            ensure!(size.is_power_of_two(), "invalid random access size");
            (1 + size as u64, 1 + size.ilog2() as u64)
        }
        Kind::Reducing | Kind::ReducingExtension => {
            let n = g.parameter()? as u64;
            (
                4 + n * if g.kind == Kind::Reducing { 1 } else { 2 },
                2 * n + if g.kind == Kind::Reducing { 2 } else { 0 },
            )
        }
        Kind::LtMask => (1, g.parameter()? as u64),
        Kind::TableLookup | Kind::CosetInterpolation | Kind::Comparison => {
            for lane in 0..g.count {
                let shape = match g.kind {
                    Kind::TableLookup => {
                        let rows = coefficient(0, lane)?;
                        (
                            2 + coefficient(
                                rows.checked_add(2)
                                    .ok_or_else(|| anyhow::anyhow!("table shape overflow"))?,
                                lane,
                            )?,
                            coefficient(1, lane)?,
                        )
                    }
                    Kind::CosetInterpolation => {
                        let degree = coefficient(0, lane)?;
                        let n = coefficient(1, lane)?;
                        ensure!(
                            degree > 1 && n >= 2 && n <= u32::MAX as u64,
                            "invalid interpolation shape"
                        );
                        (3 + 2 * n, 4 + 4 * ((n - 2) / (degree - 1)))
                    }
                    _ => {
                        let n = coefficient(0, lane)?;
                        let width = coefficient(1, lane)?;
                        ensure!(
                            n > 0 && n <= u32::MAX as u64 && width > 0 && width < 64,
                            "invalid comparison shape"
                        );
                        (2, 3 + 5 * n + width)
                    }
                };
                ensure!(declared == shape, "operand arity does not match operation");
            }
            return Ok(());
        }
        Kind::ArithmeticChain => {
            ensure!(
                g.outputs > 0 && g.outputs <= 8 && g.inputs <= 3 * g.outputs,
                "invalid arithmetic chain shape"
            );
            for lane in 0..g.count {
                for op in 0..g.outputs as u64 {
                    for pos in 0..3 {
                        let reference = coefficient(5 * op + pos, lane)?;
                        ensure!(
                            if reference >> 63 != 0 {
                                reference & !(1u64 << 63) < op
                            } else {
                                reference < g.inputs as u64
                            },
                            "invalid chain input reference"
                        );
                    }
                }
            }
            return Ok(());
        }
    };
    ensure!(
        declared == expected,
        "operand arity does not match operation"
    );
    Ok(())
}

/// Validates read-before-write, complete coverage, and exclusive ownership.
/// Multiple writes are permitted only within one lane or to a seeded root;
/// these retain the original runtime agreement checks.
pub(super) fn certify(p: &RegionProgram) -> Result<FastLayout> {
    let n = p.state_slots as usize;
    let mut available = vec![false; n];
    available[..p.roots.len()].fill(true);
    let mut counts = vec![0u32; n];
    let mut writers = vec![u64::MAX; n];
    let mut owner = 0u64;
    for layer in &p.groups {
        let mut next = Vec::new();
        for g in layer {
            validate_shape(p, g)?;
            let start = g.targets_offset as usize;
            let count = g.count as usize;
            let width = (g.inputs + g.outputs) as usize;
            ensure!(
                start
                    .checked_add(width * count)
                    .is_some_and(|end| end <= p.targets.len()),
                "invalid target block"
            );
            for k in 0..count {
                for pos in 0..g.inputs as usize {
                    let slot = p.targets[start + pos * count + k] as usize;
                    ensure!(
                        slot < n && available[slot],
                        "input not produced by an earlier layer"
                    );
                }
                for pos in g.inputs as usize..width {
                    let slot = p.targets[start + pos * count + k] as usize;
                    ensure!(slot < n, "invalid output slot");
                    if slot >= p.roots.len() {
                        ensure!(
                            writers[slot] == u64::MAX || writers[slot] == owner,
                            "output has multiple independent writers"
                        );
                    }
                    writers[slot] = owner;
                    counts[slot] += 1;
                    next.push(slot);
                }
                owner += 1;
            }
        }
        for slot in next {
            available[slot] = true;
        }
    }
    ensure!(
        available.iter().all(|&v| v),
        "program leaves an unwritten slot"
    );
    let single = counts
        .into_iter()
        .enumerate()
        .map(|(i, n)| u8::from(i >= p.roots.len() && n == 1))
        .collect::<Vec<_>>();
    // Keep sparse indexing small. Dense/repeated-output circuits retain the
    // ordinary per-slot flags; the common case stores only a few exceptions.
    let limit = n / 64;
    let repeated = single
        .iter()
        .enumerate()
        .skip(p.roots.len())
        .filter_map(|(slot, &flag)| (flag == 0).then_some(slot as u32))
        .take(limit + 1)
        .collect::<Vec<_>>();
    let sparse_ready = (repeated.len() <= limit).then_some(repeated);
    Ok(FastLayout {
        sparse_ready,
        single: single.into(),
        poseidon: Default::default(),
        adaptive: Default::default(),
        random_spans: Default::default(),
        #[cfg(feature = "gpu")]
        wire_slots: Default::default(),
        #[cfg(feature = "gpu")]
        gpu_tail: Default::default(),
    })
}

/// Places each operation's outputs together while retaining operand and
/// group ordering.
pub(super) fn reorder(p: &mut RegionProgram) -> Result<()> {
    let canonical = p.representatives.len();
    let mut order = (0..p.roots.len() as u32).collect::<Vec<_>>();
    let mut seen = vec![false; canonical];
    seen[..p.roots.len()].fill(true);
    for layer in &p.groups {
        for g in layer {
            for k in 0..g.count {
                for pos in g.inputs..g.inputs + g.outputs {
                    let slot = p.targets[(g.targets_offset + pos * g.count + k) as usize] as usize;
                    if slot < canonical && !seen[slot] {
                        order.push(slot as u32);
                        seen[slot] = true;
                    }
                }
            }
        }
    }
    order.extend(
        (p.roots.len()..canonical)
            .filter(|&i| !seen[i])
            .map(|i| i as u32),
    );
    ensure!(order.len() == canonical, "incomplete layout");
    order.extend(canonical as u32..p.state_slots);
    let mut remap = vec![0u32; p.state_slots as usize];
    for (new, &old) in order.iter().enumerate() {
        remap[old as usize] = new as u32;
    }
    p.representatives = order[..canonical]
        .iter()
        .map(|&old| p.representatives[old as usize])
        .collect::<Vec<_>>()
        .into();
    for target in std::sync::Arc::make_mut(&mut p.targets) {
        *target = remap[*target as usize];
    }
    Ok(())
}

fn run_dynamic(e: &Executor<'_>, g: &Group) -> Result<()> {
    match g.kind {
        Kind::Poseidon => run_fixed::<0>(e, g),
        Kind::Arithmetic => run_fixed::<1>(e, g),
        Kind::Copy => run_fixed::<2>(e, g),
        Kind::Constant => run_fixed::<3>(e, g),
        Kind::Equality => run_fixed::<4>(e, g),
        Kind::ExtensionArithmetic => run_fixed::<5>(e, g),
        Kind::ExtensionMultiply => run_fixed::<6>(e, g),
        Kind::RandomAccess => run_fixed::<7>(e, g),
        Kind::BaseSplit => run_fixed::<8>(e, g),
        Kind::WireSplit => run_fixed::<9>(e, g),
        Kind::LowHigh => run_fixed::<10>(e, g),
        Kind::Reducing => run_fixed::<11>(e, g),
        Kind::ReducingExtension => run_fixed::<12>(e, g),
        Kind::TableLookup => run_fixed::<13>(e, g),
        Kind::ExtensionQuotient => run_fixed::<14>(e, g),
        Kind::EcAddXu => run_fixed::<15>(e, g),
        Kind::PoseidonMds => run_fixed::<16>(e, g),
        Kind::RandomValue => run_fixed::<17>(e, g),
        Kind::ConditionalZero => run_fixed::<18>(e, g),
        Kind::LtMask => run_fixed::<19>(e, g),
        Kind::QuinticMultiply => run_fixed::<20>(e, g),
        Kind::QuinticQuotient => run_fixed::<21>(e, g),
        Kind::EcAddHomogOffset => run_fixed::<22>(e, g),
        Kind::EcScalarMul => run_fixed::<23>(e, g),
        Kind::CosetInterpolation => run_fixed::<24>(e, g),
        Kind::Comparison => run_fixed::<25>(e, g),
        Kind::ArithmeticChain => run_fixed::<26>(e, g),
        Kind::BitSum => run_fixed::<27>(e, g),
    }
}

pub(super) type PatternFn = fn(&Executor<'_>, &Group, u32) -> Result<()>;

pub(super) fn pattern_fallback(e: &Executor<'_>, g: &Group, k: u32) -> Result<()> {
    e.step_fixed::<26>(g, k)
}

pub(super) fn run_pattern_group(
    e: &Executor<'_>,
    g: &Group,
    ids: &[u8],
    kernels: &[PatternFn],
) -> Result<()> {
    ensure!(
        ids.len() == g.count as usize,
        "native pattern lane count mismatch"
    );
    let chunk = super::cpu::lane_chunk(Kind::ArithmeticChain) as usize;
    let run = |k: u32| -> Result<()> {
        let kernel = kernels
            .get(ids[k as usize] as usize)
            .ok_or_else(|| anyhow::anyhow!("unknown native pattern"))?;
        kernel(e, g, k)
    };
    ids.par_chunks(chunk)
        .enumerate()
        .try_for_each(|(c, lanes)| {
            let _profile = e.profile.kind(26, lanes.len() as u32);
            let start = (c * chunk) as u32;
            let end = start + lanes.len() as u32;
            // Lanes deferred to the device are not run here.
            #[cfg(feature = "gpu")]
            if let Some(tail) = e.gpu_tail {
                return tail.cpu_ranges(g, start, end, |a, b| (a..b).try_for_each(run));
            }
            (start..end).try_for_each(run)
        })
}

/// Classifies each arithmetic chain once against a small bank of native kernels.
/// Descriptor keys are compared in full; unfamiliar shapes retain the generic
/// chain executor. This works across circuits and state layouts.
fn adaptive_tags<'a>(p: &'a RegionProgram, descriptors: &[&[u64]]) -> &'a HashMap<u32, Vec<u8>> {
    let fast = p.fast.get().expect("validated adaptive plan");
    fast.adaptive.get_or_init(|| {
        let known = descriptors
            .iter()
            .enumerate()
            .map(|(i, &key)| (key, (i + 1) as u8))
            .collect::<HashMap<_, _>>();
        let mut groups = HashMap::new();
        let mut covered = 0u64;
        let mut total = 0u64;
        for g in p
            .groups
            .iter()
            .flatten()
            .filter(|g| g.kind == Kind::ArithmeticChain)
        {
            let base = g.coefficients_offset().expect("chain coefficient storage") as usize;
            let mut key = Vec::with_capacity(2 + 5 * g.outputs as usize);
            let mut ids = Vec::with_capacity(g.count as usize);
            for k in 0..g.count as usize {
                key.clear();
                key.extend([g.inputs as u64, g.outputs as u64]);
                for position in 0..5 * g.outputs as usize {
                    key.push(p.coefficients[base + position * g.count as usize + k]);
                }
                let tag = known.get(key.as_slice()).copied().unwrap_or(0);
                ids.push(tag);
                total += g.outputs as u64;
                if tag != 0 {
                    covered += g.outputs as u64;
                }
            }
            groups.insert(g.targets_offset, ids);
        }
        log::debug!(target: super::COMPILE_LOG,"native arithmetic coverage: {covered}/{total}");
        groups
    })
}

pub(super) fn prepare(p: &RegionProgram) -> Result<()> {
    if p.fast.get().is_none() {
        let certificate = certify(p)?;
        let _ = p.fast.set(certificate);
    }
    if fast_enabled(p) {
        let _ = poseidon_bases(p);
        let _ = random_spans(p);
        generated::prepare(p);
        #[cfg(feature = "parallel")]
        if p.runtime.config.policy.schedule != super::super::policy::WitnessSchedule::Layers {
            let _ = super::schedule::prepare(p)?;
        }
    }
    Ok(())
}

pub(super) fn run_adaptive(
    e: &Executor<'_>,
    descriptors: &[&[u64]],
    kernels: &[PatternFn],
) -> Result<()> {
    #[cfg(feature = "parallel")]
    if super::schedule::run(e)? {
        return Ok(());
    }
    let tags = adaptive_tags(e.program, descriptors);
    for (index, layer) in e.program.groups.iter().enumerate() {
        let _layer = e
            .profile
            .layer(index, layer.iter().map(|g| g.count as u64).sum());
        layer.par_iter().try_for_each(|g| {
            if g.kind == Kind::ArithmeticChain {
                run_pattern_group(e, g, &tags[&g.targets_offset], kernels)
            } else {
                run_dynamic(e, g)
            }
        })?;
    }
    Ok(())
}

/// Uploads the executor's compact state for kernels that read wires through
/// the cell-to-slot table, which is uploaded once per program. With a device
/// tail only the CPU-produced slots are uploaded and scattered on the device.
#[cfg(feature = "gpu")]
pub(super) fn compact_on_gpu(
    e: &Executor<'_>,
    ctx: &crate::gpu::GpuContext,
    slots: &[u32],
) -> Result<(crate::gpu::DeviceBuffer, crate::gpu::DeviceBuffer)> {
    let cache = &e
        .program
        .fast
        .get()
        .expect("validated device witness plan")
        .wire_slots;
    let indices = cache.get().unwrap_or_else(|| {
        let words = slots
            .chunks(2)
            .map(|x| x[0] as u64 | ((x.get(1).copied().unwrap_or(u32::MAX) as u64) << 32))
            .collect::<Vec<_>>();
        let indices = ctx.upload("witness wire slots", &words);
        cache.set_if_within_limit(ctx, indices.clone());
        indices
    });
    if let Some(tail) = e.gpu_tail {
        return Ok((tail.upload_state(e, ctx), indices));
    }
    let state_bytes = e.state.len() * 8;
    let staging = ctx.upload_staging(state_bytes as u64);
    e.program.runtime.run(|| {
        #[cfg(test)]
        e.program.runtime.assert_active_pool();
        e.state
            .par_chunks(16384)
            .enumerate()
            .for_each(|(block, state)| {
                let begin = block * 16384 * 8;
                let mut view = staging
                    .buffer
                    .slice(begin as u64..(begin + state.len() * 8) as u64)
                    .get_mapped_range_mut()
                    .expect("state upload mapping");
                let (words, _) = view.slice(..).into_chunks::<8>();
                for (word, value) in words.into_iter().zip(state) {
                    word.write(
                        value
                            .load(std::sync::atomic::Ordering::Relaxed)
                            .to_le_bytes(),
                    );
                }
            });
    });
    let state = ctx.storage("compact witness state", state_bytes as u64);
    ctx.finish_upload(staging, &state, state_bytes as u64);
    Ok((state, indices))
}

fn poseidon_mode() -> u8 {
    if crate::hash::poseidon_trace_simd::native_available() {
        5
    } else {
        0
    }
}
fn poseidon_min_group() -> u32 {
    64
}
fn poseidon_chunk() -> u32 {
    8
}

#[cfg(feature = "parallel")]
pub(super) fn run_range(e: &Executor<'_>, g: &Group, start: u32, end: u32) -> Result<()> {
    #[cfg(feature = "gpu")]
    if let Some(tail) = e.gpu_tail {
        return tail.cpu_ranges(g, start, end, |a, b| run_range_inner(e, g, a, b));
    }
    run_range_inner(e, g, start, end)
}
#[cfg(feature = "parallel")]
fn run_range_inner(e: &Executor<'_>, g: &Group, start: u32, end: u32) -> Result<()> {
    match g.kind {
        Kind::Arithmetic => return arithmetic_simplified(e, g, start, end),
        Kind::Equality => return equality_batch(e, g, start, end),
        Kind::Poseidon => {
            let starts = &poseidon_bases(e.program)[&g.targets_offset];
            let mode = if g.count >= poseidon_min_group() {
                poseidon_mode()
            } else {
                0
            };
            if mode != 0 {
                return e.step_poseidon_simd(g, start, end, starts, mode);
            }
            for k in start..end {
                if starts[k as usize] != u32::MAX {
                    e.step_poseidon_bulk(k, g.count, g.targets_offset, starts[k as usize])?
                } else {
                    e.step_fixed::<0>(g, k)?
                }
            }
            return Ok(());
        }
        Kind::RandomValue if g.count >= 512 => {
            if let Some(&(src, dst)) = random_spans(e.program).get(&g.targets_offset) {
                return e.step_random_span(src, dst, start, end);
            }
        }
        Kind::ArithmeticChain => return generated::sequential(e, g, start..end),
        _ => {}
    }
    macro_rules! run {($($k:literal),*)=>{match g.kind.code(){$($k=>{for k in start..end{e.step_fixed::<$k>(g,k)?;}},)*_=>unreachable!()}}}
    run!(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn copy(offset: u32, count: u32) -> Group {
        Group {
            kind: Kind::Copy,
            count,
            targets_offset: offset,
            argument: Kind::Copy.argument(0),
            inputs: 1,
            outputs: 1,
        }
    }
    fn plan(groups: Vec<Vec<Group>>, targets: Vec<u32>, slots: u32) -> RegionProgram {
        RegionProgram {
            groups,
            targets: targets.into(),
            coefficients: Vec::new().into(),
            representatives: (0..slots as usize).collect::<Vec<_>>().into(),
            roots: vec![0].into(),
            random_count: 0,
            alias_count: 0,
            alias_pair_offset: 0,
            state_slots: slots,
            runtime: Default::default(),
            fast: Default::default(),
            rep_slots: Default::default(),
            cell_slots: Default::default(),
        }
    }
    #[test]
    fn certificate_rejects_reads_from_the_current_layer() {
        let mut p = plan(vec![vec![copy(0, 1), copy(2, 1)]], vec![0, 1, 1, 2], 3);
        assert!(certify(&p)
            .unwrap_err()
            .to_string()
            .contains("earlier layer"));
        let dependent = p.groups[0].pop().unwrap();
        p.groups.push(vec![dependent]);
        assert_eq!(certify(&p).unwrap().single.as_ref(), &[0, 1, 1]);
    }
    #[test]
    fn certificate_rejects_independent_writers_and_missing_slots() {
        let p = plan(vec![vec![copy(0, 2)]], vec![0, 0, 1, 1], 2);
        assert!(certify(&p)
            .unwrap_err()
            .to_string()
            .contains("independent writers"));
        let p = plan(vec![vec![copy(0, 1)]], vec![0, 1], 3);
        assert!(certify(&p).unwrap_err().to_string().contains("unwritten"));
    }
    #[test]
    fn certificate_does_not_trust_declared_operand_arities() {
        let mut p = plan(vec![vec![copy(0, 1)]], vec![0, 1], 2);
        p.groups[0][0].inputs = 0;
        assert!(certify(&p).unwrap_err().to_string().contains("arity"));
    }
    #[test]
    fn certificate_preserves_checks_for_roots_and_repeated_lane_outputs() {
        let roots = plan(vec![vec![copy(0, 2)]], vec![0, 0, 0, 0], 1);
        let roots_cert = certify(&roots).unwrap();
        assert_eq!(roots_cert.single.as_ref(), &[0]);
        assert_eq!(roots_cert.sparse_ready.as_deref(), Some(&[][..]));
        let split = Group {
            kind: Kind::BaseSplit,
            count: 1,
            targets_offset: 0,
            argument: Kind::BaseSplit.argument((2 << 8) | 1),
            inputs: 1,
            outputs: 2,
        };
        let repeated = plan(vec![vec![split]], vec![0, 1, 1], 2);
        let repeated_cert = certify(&repeated).unwrap();
        assert_eq!(repeated_cert.single.as_ref(), &[0, 0]);
        assert!(repeated_cert.sparse_ready.is_none());
    }
}
