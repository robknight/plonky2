//! Validated shared witness kernels and bounded native arithmetic chains.
use std::borrow::Cow;
use std::collections::HashMap;
#[allow(unused_imports)]
use std::sync::Arc;
use std::sync::OnceLock;

use anyhow::{ensure, Result};
use plonky2_maybe_rayon::*;

use super::cpu::Executor;
use super::{Group, Kind, RegionProgram};

mod generated {
    include!("native_kernels.rs");
}

#[derive(Clone, Debug)]
pub(super) struct FastLayout {
    pub single: Cow<'static, [u8]>,
    pub poseidon: OnceLock<HashMap<u32, Vec<u32>>>,
    pub adaptive: OnceLock<HashMap<u32, Vec<u8>>>,
    #[cfg(feature = "gpu")]
    pub gpu_gather: OnceLock<Arc<GpuGather>>,
}

pub(crate) fn layout() -> u8 {
    let placement = match std::env::var("P2_AOT_LAYOUT").as_deref() {
        Ok("aos") => 1,
        Ok("representative") => 2,
        Ok("block4") => 3,
        Ok("block8") => 4,
        _ => 0,
    };
    let chain = std::env::var("P2_AOT_CHAIN_LEN")
        .ok()
        .and_then(|x| x.parse::<u8>().ok())
        .unwrap_or(8)
        .clamp(2, 8);
    placement | if fusion_enabled() { chain * 8 } else { 0 }
}

pub(crate) fn file_stem(digest: &[u8]) -> String {
    let hex = digest
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    format!("circuit_{}_{hex}", layout())
}

pub(super) fn fast_enabled() -> bool {
    matches!(
        std::env::var("P2_AOT_MODE").as_deref(),
        Ok("fast") | Ok("fast-batch") | Ok("bulk") | Ok("adaptive")
    )
}

pub(super) fn execute(executor: &Executor<'_>) -> Result<bool> {
    if !fast_enabled() {
        return Ok(false);
    }
    if std::env::var("P2_AOT_MODE").as_deref() == Ok("adaptive") {
        generated::run(executor)?;
    } else {
        for (index, layer) in executor.program.groups.iter().enumerate() {
            let _layer = executor
                .profile
                .layer(index, layer.iter().map(|g| g.count as u64).sum());
            layer
                .par_iter()
                .try_for_each(|g| run_dynamic(executor, g))?;
        }
    }
    Ok(true)
}

pub(super) fn run_fixed<const K: u8>(executor: &Executor<'_>, group: &Group) -> Result<()> {
    let kind = Kind::from_code(K as usize)?;
    let use_simd = K == 0 && poseidon_mode() != 0 && group.count >= poseidon_min_group();
    let chunk = if use_simd {
        poseidon_chunk()
    } else {
        tuned_chunk(kind)
    };
    if K == 1 && simplify_enabled() {
        return (0..group.count.div_ceil(chunk))
            .into_par_iter()
            .try_for_each(|c| {
                let _profile = executor
                    .profile
                    .kind(K, (((c + 1) * chunk).min(group.count)) - c * chunk);
                arithmetic_simplified(
                    executor,
                    group,
                    c * chunk,
                    ((c + 1) * chunk).min(group.count),
                )
            });
    }
    if K == 4 && batch_enabled() {
        return (0..group.count.div_ceil(chunk))
            .into_par_iter()
            .try_for_each(|c| {
                let _profile = executor
                    .profile
                    .kind(K, (((c + 1) * chunk).min(group.count)) - c * chunk);
                equality_batch(
                    executor,
                    group,
                    c * chunk,
                    ((c + 1) * chunk).min(group.count),
                )
            });
    }
    if K == 0 && bulk_enabled() {
        let bases = poseidon_bases(executor.program);
        let starts = &bases[&group.targets_offset];
        return (0..group.count.div_ceil(chunk))
            .into_par_iter()
            .try_for_each(|c| {
                let begin = c * chunk;
                let end = ((c + 1) * chunk).min(group.count);
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
                .kind(K, (((c + 1) * chunk).min(group.count)) - c * chunk);
            for k in c * chunk..((c + 1) * chunk).min(group.count) {
                executor.step_fixed::<K>(group, k)?;
            }
            Ok(())
        })
}

fn tuned_chunk(kind: Kind) -> u32 {
    static SCALE: OnceLock<u32> = OnceLock::new();
    let scale = *SCALE.get_or_init(|| {
        std::env::var("P2_AOT_CHUNK_SCALE")
            .ok()
            .and_then(|x| x.parse::<u32>().ok())
            .unwrap_or(1)
            .max(1)
    });
    super::cpu::lane_chunk(kind).saturating_mul(scale).max(1)
}

fn batch_enabled() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| {
        matches!(
            std::env::var("P2_AOT_MODE").as_deref(),
            Ok("fast-batch") | Ok("bulk") | Ok("adaptive")
        )
    })
}

fn simplify_enabled() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| std::env::var("P2_AOT_SIMPLIFY_ARITH").as_deref() == Ok("1"))
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

fn bulk_enabled() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| {
        matches!(
            std::env::var("P2_AOT_MODE").as_deref(),
            Ok("bulk") | Ok("adaptive")
        )
    })
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
            eprintln!("AOT_POSEIDON_BLOCKS eligible={eligible} total={total}");
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
    Ok(FastLayout {
        single: single.into(),
        poseidon: Default::default(),
        adaptive: Default::default(),
        #[cfg(feature = "gpu")]
        gpu_gather: Default::default(),
    })
}

/// Changes value placement while retaining operand and group ordering.
pub(super) fn reorder(p: &mut RegionProgram) -> Result<()> {
    let mode = layout() & 7;
    if mode == 0 {
        return Ok(());
    }
    let canonical = p.representatives.len();
    let mut order = (0..p.roots.len() as u32).collect::<Vec<_>>();
    let mut seen = vec![false; canonical];
    seen[..p.roots.len()].fill(true);
    if mode == 1 || mode >= 3 {
        let block = match mode {
            3 => 4,
            4 => 8,
            _ => 1,
        };
        for layer in &p.groups {
            for g in layer {
                for start in (0..g.count).step_by(block as usize) {
                    for pos in g.inputs..g.inputs + g.outputs {
                        for k in start..(start + block).min(g.count) {
                            let slot =
                                p.targets[(g.targets_offset + pos * g.count + k) as usize] as usize;
                            if slot < canonical && !seen[slot] {
                                order.push(slot as u32);
                                seen[slot] = true;
                            }
                        }
                    }
                }
            }
        }
    }
    let mut rest = (p.roots.len()..canonical)
        .filter(|&i| !seen[i])
        .collect::<Vec<_>>();
    if mode == 2 {
        rest.sort_unstable_by_key(|&i| p.representatives[i]);
    }
    order.extend(rest.into_iter().map(|i| i as u32));
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
    for target in p.targets.to_mut() {
        *target = remap[*target as usize];
    }
    Ok(())
}

pub(crate) fn fusion_enabled() -> bool {
    std::env::var("P2_AOT_FUSE").as_deref() == Ok("1")
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
    let chunk = tuned_chunk(Kind::ArithmeticChain) as usize;
    ids.par_chunks(chunk).enumerate().try_for_each(|(c, ids)| {
        let _profile = e.profile.kind(26, ids.len() as u32);
        for (i, &id) in ids.iter().enumerate() {
            let kernel = kernels
                .get(id as usize)
                .ok_or_else(|| anyhow::anyhow!("unknown native pattern"))?;
            kernel(e, g, (c * chunk + i) as u32)?;
        }
        Ok(())
    })
}

/// Classifies each arithmetic chain once against a small bank of native kernels.
/// Descriptor keys are compared in full; unfamiliar shapes retain the generic
/// chain executor. This works across circuits and state layouts.
pub(super) fn run_adaptive(
    e: &Executor<'_>,
    descriptors: &[&[u64]],
    kernels: &[PatternFn],
) -> Result<()> {
    let fast = e.program.fast.get().expect("validated adaptive plan");
    let tags = fast.adaptive.get_or_init(|| {
        let known = descriptors
            .iter()
            .enumerate()
            .map(|(i, &key)| (key, (i + 1) as u8))
            .collect::<HashMap<_, _>>();
        let mut groups = HashMap::new();
        let mut covered = 0u64;
        let mut total = 0u64;
        for g in e
            .program
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
                    key.push(e.program.coefficients[base + position * g.count as usize + k]);
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
        eprintln!("AOT_ADAPTIVE_MATCH covered_operations={covered} total_operations={total}");
        groups
    });
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
        assert_eq!(certify(&roots).unwrap().single.as_ref(), &[0]);
        let split = Group {
            kind: Kind::BaseSplit,
            count: 1,
            targets_offset: 0,
            argument: Kind::BaseSplit.argument((2 << 8) | 1),
            inputs: 1,
            outputs: 2,
        };
        let repeated = plan(vec![vec![split]], vec![0, 1, 1], 2);
        assert_eq!(certify(&repeated).unwrap().single.as_ref(), &[0, 0]);
    }
}

#[cfg(feature = "gpu")]
#[derive(Debug)]
pub(super) struct GpuGather {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    slots: crate::gpu::DeviceCache,
}

/// Upload compact contiguous state; resolve the fixed wire permutation on GPU.
/// Synchronous submission makes this experiment's gather timer include GPU work.
#[cfg(feature = "gpu")]
pub(super) fn gather_on_gpu(
    e: &Executor<'_>,
    ctx: &crate::gpu::GpuContext,
    slots: &[u32],
) -> Result<crate::gpu::DeviceBuffer> {
    let cache = e.program.fast.get().expect("validated GPU gather plan").gpu_gather.get_or_init(|| {
        let layout = crate::gpu::storage_layout(&ctx.device, "witness gather", &[(false,true),(false,true),(false,false)]);
        let module = crate::gpu::module(&ctx.device, "witness gather", r#"
@group(0) @binding(0) var<storage, read> state: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> slots: array<u32>;
@group(0) @binding(2) var<storage, read_write> wires: array<vec2<u32>>;
@compute @workgroup_size(256)
fn gather(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = id.x + id.y * groups.x * 256u;
    if (i >= arrayLength(&slots) || i >= arrayLength(&wires)) { return; }
    let slot = slots[i];
    var value = vec2<u32>(0u);
    if (slot != 0xffffffffu && slot < arrayLength(&state)) { value = state[slot]; }
    wires[i] = value;
}
"#);
        let pipeline = crate::gpu::pipeline(&ctx.device, &layout, &module, "gather");
        Arc::new(GpuGather { layout, pipeline, slots: Default::default() })
    });
    let indices = cache.slots.get().unwrap_or_else(|| {
        let words = slots
            .chunks(2)
            .map(|x| x[0] as u64 | ((x.get(1).copied().unwrap_or(u32::MAX) as u64) << 32))
            .collect::<Vec<_>>();
        let indices = ctx.upload("witness wire slots", &words);
        cache.slots.set_if_within_limit(ctx, indices.clone());
        indices
    });
    let state_bytes = e.state.len() * 8;
    let staging = ctx.upload_staging(state_bytes as u64);
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
    let state = ctx.storage("compact witness state", state_bytes as u64);
    ctx.finish_upload(staging, &state, state_bytes as u64);
    let out = ctx.storage("GPU gathered proving wires", (slots.len() * 8) as u64);
    let bind = ctx.bind(&cache.layout, &[&state, &indices, &out]);
    let mut encoder = ctx.encoder("witness wire gather");
    {
        let mut pass = ctx.begin_pass(&mut encoder, "witness wire gather", None);
        pass.set_pipeline(&cache.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        let groups = (slots.len() as u32).div_ceil(256);
        let width = groups.min(65535);
        pass.dispatch_workgroups(width, groups.div_ceil(width), 1);
    }
    ctx.submit(encoder);
    Ok(out)
}

fn poseidon_mode() -> u8 {
    let override_mode =
        crate::hash::poseidon_trace_simd::BENCH_MODE.load(std::sync::atomic::Ordering::Relaxed);
    if override_mode != 255 {
        return override_mode;
    }
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("P2_POSEIDON_MODE").as_deref() {
        Ok("batch8-fft") => {
            if crate::hash::poseidon_trace_simd::NATIVE {
                5
            } else {
                0
            }
        }
        _ => 0,
    })
}
fn poseidon_min_group() -> u32 {
    64
}
fn poseidon_chunk() -> u32 {
    8
}
