//! Runs a prepared typed-operation graph on canonical atomic witness state.
//!
//! Certification establishes dependencies and writer ownership before proving.
//! The fast path retains agreement checks for seeded/repeated destinations and
//! private alias slots; the checked interpreter remains available as a control.
//! Parallel automatic execution advances ready chunks without global barriers.
//! Layered and single-thread execution retain the original dependency layers.
//! Sparse readiness metadata avoids one flag per value on eligible programs.
//!
//! Gathering writes directly into column-major wires or GPU upload staging,
//! avoiding construction and merging of a partition witness. CPU sampling,
//! execution, gathering and wiping honor the configured execution pool; GPU
//! orchestration stays on the caller. State and readiness buffers are wiped
//! before release or bounded reuse.
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{ensure, Result};
use plonky2_maybe_rayon::*;

#[cfg(target_arch = "x86_64")]
use super::profile::Poseidon as PoseidonPart;
use super::profile::{self, Stage};
use super::{Kind, RegionProgram};
use crate::field::extension::quintic::QuinticExtension;
use crate::field::extension::Extendable;
use crate::field::goldilocks_field::GoldilocksField as G;
use crate::field::types::{Field, Field64, PrimeField64};
use crate::hash::hash_types::RichField;
use crate::hash::poseidon::{self, Poseidon, SPONGE_WIDTH};
use crate::iop::generator::WitnessOperation as Op;
use crate::iop::target::Target;
pub(crate) use crate::iop::witness::ProverWires as Wires;
use crate::iop::witness::{MatrixWitness, PartialWitness, PartitionWitness, Witness};
use crate::plonk::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
use crate::plonk::config::GenericConfig;

type Q = QuinticExtension<G>;

#[cfg(feature = "gpu")]
fn gather_tiling(_config: &crate::plonk::witness_program::WitnessConfig) -> (usize, usize) {
    #[cfg(test)]
    return _config.policy.gather_tiles;
    #[cfg(not(test))]
    (16, 256)
}

impl RegionProgram {
    /// State slot written by each random value, indexed by the random index
    /// the program assigns. Slots below the representative count are
    /// canonical; higher slots belong to alias writers.
    pub fn random_slots(&self) -> Vec<u32> {
        let mut out = vec![u32::MAX; self.random_count as usize];
        for group in self.groups.iter().flatten() {
            if group.kind != Kind::RandomValue {
                continue;
            }
            for k in 0..group.count {
                let index = self.coefficients[(group
                    .coefficients_offset()
                    .expect("random group coefficient storage")
                    + k) as usize] as usize;
                out[index] = self.targets[(group.targets_offset + k) as usize];
            }
        }
        out
    }

    /// Representative written by each random value, indexed by the random
    /// index the program assigns. `None` when the writer is an alias.
    pub fn random_representatives(&self) -> Vec<Option<usize>> {
        self.random_slots()
            .into_iter()
            .map(|slot| self.representatives.get(slot as usize).copied())
            .collect()
    }

    /// Runs the compiled program on the CPU, parallelizing independent lanes
    /// when the `parallel` feature is enabled. `random_values`
    /// supplies one canonical field element per random index. Boundary
    /// roots come from `entry`; every other slot is written exactly once by
    /// its owner, additional writers go to alias slots and are compared at
    /// the end. Returns one value per entry of `self.representatives`.
    pub fn execute_cpu<F: PrimeField64>(
        &self,
        entry: &PartitionWitness<F>,
        random_values: &[u64],
    ) -> Result<Vec<F>> {
        ensure!(Op::field_supported::<F>(), "unsupported execution field");
        let root_values = self
            .roots
            .iter()
            .map(|&r| {
                entry
                    .values
                    .get(r)
                    .and_then(|v| *v)
                    .map(|v| v.to_canonical_u64())
                    .ok_or_else(|| anyhow::anyhow!("missing boundary input {r}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let executor = self.execute_state(&root_values, random_values)?;
        Ok(executor.state[..self.representatives.len()]
            .iter()
            .map(|v| F::from_canonical_u64(v.load(Ordering::Relaxed)))
            .collect())
    }

    /// Representative index to canonical slot over `len` representatives,
    /// `u32::MAX` where the program has no slot. Built once per program.
    fn rep_slots(&self, len: usize) -> &[u32] {
        self.rep_slots.get_or_init(|| {
            let mut slots = vec![u32::MAX; len];
            for (slot, &representative) in self.representatives.iter().enumerate() {
                slots[representative] = slot as u32;
            }
            slots
        })
    }

    /// Canonical slot of every wire cell, column-major (`column * degree +
    /// row`), `u32::MAX` where the program has no slot. Built once per
    /// program and used by both the host gather and the device upload.
    fn cell_slots(&self, map: &[usize], num_wires: usize, degree: usize) -> &[u32] {
        self.cell_slots.get_or_init(|| {
            let rep_slots = self.rep_slots(map.len());
            let mut cells = vec![u32::MAX; num_wires * degree];
            cells
                .par_chunks_mut(degree)
                .enumerate()
                .for_each(|(column, cells)| {
                    for (row, cell) in cells.iter_mut().enumerate() {
                        *cell = rep_slots[map[row * num_wires + column]];
                    }
                });
            cells
        })
    }

    pub(crate) fn prepare(&self, map: &[usize], num_wires: usize, degree: usize) -> Result<()> {
        super::aot::prepare(self)?;
        let _ = self.rep_slots(map.len());
        let _ = self.cell_slots(map, num_wires, degree);
        Ok(())
    }

    pub(crate) fn is_prepared(&self) -> bool {
        self.fast.get().is_some()
            && self.rep_slots.get().is_some()
            && self.cell_slots.get().is_some()
            && {
                #[cfg(feature = "parallel")]
                {
                    self.runtime.config.policy.execution
                        == super::super::policy::WitnessExecution::Checked
                        || self.runtime.config.policy.schedule
                            == super::super::policy::WitnessSchedule::Layers
                        || matches!(self.runtime.schedule.get(), Some(Ok(_)))
                }
                #[cfg(not(feature = "parallel"))]
                {
                    true
                }
            }
            && (!super::aot::fast_enabled(self)
                || self.fast.get().is_some_and(|f| {
                    f.poseidon.get().is_some()
                        && f.adaptive.get().is_some()
                        && f.random_spans.get().is_some()
                }))
    }

    /// Runs the program over `root_values`, one canonical value per boundary
    /// root in `roots` order, and returns the final slot state.
    fn execute_state<'a>(
        &'a self,
        root_values: &[u64],
        random_values: &'a [u64],
    ) -> Result<Executor<'a>> {
        self.runtime
            .run(|| self.execute_state_inner(root_values, random_values, false))
    }

    fn execute_state_inner<'a>(
        &'a self,
        root_values: &[u64],
        random_values: &'a [u64],
        reference: bool,
    ) -> Result<Executor<'a>> {
        self.execute_state_inner_with_tail(
            root_values,
            random_values,
            reference,
            #[cfg(feature = "gpu")]
            None,
        )
    }
    fn execute_state_inner_with_tail<'a>(
        &'a self,
        root_values: &[u64],
        random_values: &'a [u64],
        reference: bool,
        #[cfg(feature = "gpu")] gpu_tail: Option<&'a super::gpu_tail::Tail>,
    ) -> Result<Executor<'a>> {
        ensure!(
            random_values.len() == self.random_count as usize,
            "expected {} random values, got {}",
            self.random_count,
            random_values.len()
        );
        ensure!(
            root_values.len() == self.roots.len(),
            "expected {} boundary roots, got {}",
            self.roots.len(),
            root_values.len()
        );
        let fast = if !reference && super::aot::fast_enabled(self) {
            if self.fast.get().is_none() {
                let certificate = super::aot::certify(self)?;
                let _ = self.fast.set(certificate);
            }
            self.fast.get()
        } else {
            None
        };
        let profile = profile::Execution::new(self.state_slots);
        let allocation = profile.stage(Stage::Allocate);
        let compact_readiness = fast.is_some_and(|f| f.sparse_ready.is_some());
        let (state, ready) = self.runtime.buffers.acquire(
            self.state_slots as usize,
            if compact_readiness {
                fast.unwrap().sparse_ready.as_ref().unwrap().len()
            } else {
                self.state_slots as usize
            },
        );
        drop(allocation);
        let executor = Executor {
            profile,
            program: self,
            random: random_values,
            #[cfg(feature = "gpu")]
            gpu_tail,
            compact_readiness,
            fast,
            state,
            ready,
        };
        let seeding = executor.profile.stage(Stage::Seed);
        // The compiler places boundary roots at the start of canonical state.
        for (slot, &value) in root_values.iter().enumerate() {
            ensure!(value < G::ORDER, "boundary input {value} is not canonical");
            if compact_readiness {
                executor.state[slot].store(value, Ordering::Relaxed);
            } else {
                executor.store(slot as u32, G::from_canonical_u64(value))?;
            }
        }
        drop(seeding);
        let layers = executor.profile.stage(Stage::Layers);
        // Operations in one layer are independent: every unseeded slot has
        // one writer, additional writers use private alias slots, and
        // inputs were produced by earlier layers. The Rayon join at each
        // layer boundary orders the layers; relaxed atomics suffice inside.
        let run_group = |group: &super::Group| {
            let chunk = lane_chunk(group.kind);
            let chunks = group.count.div_ceil(chunk);
            (0..chunks).into_par_iter().try_for_each(|c| {
                for k in c * chunk..chunk_end(c * chunk, chunk, group.count) {
                    executor.step(group, k)?;
                }
                Ok::<(), anyhow::Error>(())
            })
        };
        if cfg!(feature = "gpu-profiling")
            && crate::util::switches::flag("PLONKY2_GPU_WITNESS_CPU_EXECUTOR_PROFILE")
        {
            // Diagnostic: groups run one at a time so each kind's wall time
            // is attributable; lanes stay parallel.
            let mut by_kind = vec![(Duration::ZERO, 0u64, 0u64); Kind::COUNT];
            let started = Instant::now();
            for layer in &self.groups {
                for group in layer {
                    let group_started = Instant::now();
                    run_group(group)?;
                    let entry = &mut by_kind[group.kind.code()];
                    entry.0 += group_started.elapsed();
                    entry.1 += 1;
                    entry.2 += group.count as u64;
                }
            }
            let total = started.elapsed();
            let mut ranked = by_kind.iter().enumerate().collect::<Vec<_>>();
            ranked.sort_by_key(|(_, (time, _, _))| std::cmp::Reverse(*time));
            let summary = ranked
                .iter()
                .filter(|(_, (_, groups, _))| *groups > 0)
                .map(|(kind, (time, groups, lanes))| {
                    format!(
                        "{kind}:{:.2}ms/{groups}g/{lanes}l",
                        time.as_secs_f64() * 1e3
                    )
                })
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!(
                "GPU_WITNESS_CPU_EXECUTOR_PROFILE layers={} total={total:?} {summary}",
                self.groups.len()
            );
        } else if reference || !super::aot::execute(&executor)? {
            for layer in &self.groups {
                layer.par_iter().try_for_each(run_group)?;
            }
        }
        drop(layers);
        let aliases = executor.profile.stage(Stage::Aliases);
        #[cfg(feature = "gpu")]
        let device_checks = executor.gpu_tail.is_some();
        #[cfg(not(feature = "gpu"))]
        let device_checks = false;
        if !device_checks {
            for i in 0..self.alias_count {
                let canonical = self.targets[(self.alias_pair_offset + 2 * i) as usize];
                let duplicate = self.targets[(self.alias_pair_offset + 2 * i + 1) as usize];
                ensure!(
                    executor.load(canonical)? == executor.load(duplicate)?,
                    "alias slot {duplicate} disagrees with its canonical slot {canonical}"
                );
            }
        }
        drop(aliases);
        let readiness = executor.profile.stage(Stage::Ready);
        // Certification covers all non-roots, roots were seeded above, and
        // sparse flags cover every repeated non-root writer. Dense/checked
        // execution retains its original complete slot check.
        let complete = if compact_readiness {
            executor.ready.par_iter().all(|r| r.load(Ordering::Relaxed))
        } else {
            executor.ready.par_iter().enumerate().all(|(i, r)| {
                executor.fast.is_some_and(|f| f.single[i] != 0) || r.load(Ordering::Relaxed)
            })
        };
        ensure!(complete, "a state slot was never written by the plan");
        drop(readiness);
        Ok(executor)
    }
}

pub(super) struct Executor<'a> {
    pub(super) profile: profile::Execution,
    pub(super) program: &'a RegionProgram,
    pub(super) random: &'a [u64],
    #[cfg(feature = "gpu")]
    pub(super) gpu_tail: Option<&'a super::gpu_tail::Tail>,
    pub(super) state: Vec<AtomicU64>,
    ready: Vec<AtomicBool>,
    compact_readiness: bool,
    fast: Option<&'a super::aot::FastLayout>,
}

impl Drop for Executor<'_> {
    /// Witness values do not outlive the executor in host memory. The wipe
    /// goes through `zeroize`, whose volatile writes the compiler may not
    /// elide, rather than relaxed atomic stores.
    fn drop(&mut self) {
        let wipe = self.profile.stage(Stage::Wipe);
        let state = core::mem::take(&mut self.state);
        let ready = core::mem::take(&mut self.ready);
        let runtime = &self.program.runtime;
        runtime.run(|| {
            #[cfg(test)]
            runtime.assert_active_pool();
            runtime.buffers.release(state, ready);
        });
        drop(wipe);
        self.profile.report();
    }
}

/// Lanes per Rayon task. Expensive kinds split to single lanes; cheap
/// kinds are batched so task overhead stays below the arithmetic.
#[inline]
pub(super) fn chunk_end(start: u32, chunk: u32, count: u32) -> u32 {
    start.saturating_add(chunk).min(count)
}

pub(super) fn lane_chunk(kind: Kind) -> u32 {
    match kind {
        Kind::Poseidon
        | Kind::EcAddXu
        | Kind::PoseidonMds
        | Kind::EcAddHomogOffset
        | Kind::EcScalarMul
        | Kind::CosetInterpolation => 1,
        Kind::ArithmeticChain => 64,
        Kind::TableLookup | Kind::QuinticMultiply | Kind::QuinticQuotient | Kind::Comparison => 16,
        _ => 512,
    }
}

fn canonical(value: G) -> G {
    G::from_canonical_u64(value.to_canonical_u64())
}

fn field(value: u64) -> G {
    G::from_noncanonical_u64(value)
}

/// Quadratic extension product with W = 7.
fn ext_mul(a: [G; 2], b: [G; 2]) -> [G; 2] {
    let w = G::from_canonical_u64(7);
    [a[0] * b[0] + w * a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

fn inverse(value: G) -> Result<G> {
    value
        .try_inverse()
        .ok_or_else(|| anyhow::anyhow!("invalid arithmetic: inverse of zero"))
}

fn bit(value: u64, index: u64) -> u64 {
    if index >= 64 {
        0
    } else {
        (value >> index) & 1
    }
}

/// Multiplication by `factor * X`, using X^5 = 3.
fn q_mul_generator(a: Q, factor: G) -> Q {
    let mut result = Q::ZERO;
    result.0[0] = a.0[4] * factor * G::from_canonical_u64(3);
    for i in 1..5 {
        result.0[i] = a.0[i - 1] * factor;
    }
    result
}

fn q_inverse(a: Q) -> Result<Q> {
    a.try_inverse()
        .ok_or_else(|| anyhow::anyhow!("quintic inverse of zero"))
}

/// Shared inversion for the two affine coordinates. A zero denominator in
/// either coordinate is an error.
fn q_div_pair(a: Q, b: Q, c: Q, d: Q) -> Result<[Q; 2]> {
    let inverse = q_inverse(b * d)?;
    Ok([a * d * inverse, c * b * inverse])
}

/// Homogeneous (X:Z:U:T) sum of two affine points on the quintic curve.
fn q_add_homog(x1: Q, u1: Q, x2: Q, u2: Q, b1: G) -> [Q; 4] {
    let t1 = x1 * x2;
    let t3 = u1 * u2;
    let t5 = x1 + x2;
    let t6 = u1 + u2;
    let mut t7 = t1;
    t7.0[1] += b1;
    let t9 = t3 * (q_mul_generator(t5, b1 + b1) + (t7 + t7));
    let mut t3_double_plus_one = t3 + t3;
    t3_double_plus_one.0[0] += G::ONE;
    let t10 = t3_double_plus_one * (t5 + t7);
    let x = q_mul_generator(t10 - t7, b1);
    let z = t7 - t9;
    let mut neg_t1 = Q::ZERO - t1;
    neg_t1.0[1] += b1;
    let u = t6 * neg_t1;
    let t = t7 + t9;
    [x, z, u, t]
}

fn q_add_xu(x1: Q, u1: Q, x2: Q, u2: Q, b1: G) -> Result<[Q; 2]> {
    let h = q_add_homog(x1, u1, x2, u2, b1);
    q_div_pair(h[0], h[1], h[2], h[3])
}

/// Fractional (X:Z:U:T) addition from precomputed products, used by scalar
/// multiplication. Zero Z or T at any step is an error.
fn q_add_fractional_products(xx: Q, zz: Q, uu: Q, tt: Q, xz: Q, ut: Q, b1: G) -> Result<[Q; 4]> {
    let bzz = q_mul_generator(zz, b1);
    let t7 = xx + bzz;
    let t8 = tt * t7;
    let t9 = uu * (q_mul_generator(xz, b1 + b1) + (t7 + t7));
    let t10 = (tt + (uu + uu)) * (xz + t7);
    let result = [
        q_mul_generator(t10 - t8, b1),
        t8 - t9,
        ut * (bzz - xx),
        t8 + t9,
    ];
    ensure!(
        result[1] != Q::ZERO && result[3] != Q::ZERO,
        "quintic inverse of zero in scalar multiplication"
    );
    Ok(result)
}

fn q_double_fractional(p: [Q; 4], b1: G) -> Result<[Q; 4]> {
    let xz = p[0] * p[1];
    let ut = p[2] * p[3];
    q_add_fractional_products(
        p[0] * p[0],
        p[1] * p[1],
        p[2] * p[2],
        p[3] * p[3],
        xz + xz,
        ut + ut,
        b1,
    )
}

fn q_add_mixed_fractional(p: [Q; 4], x: Q, u: Q, b1: G) -> Result<[Q; 4]> {
    q_add_fractional_products(
        p[0] * x,
        p[1],
        p[2] * u,
        p[3],
        p[0] + p[1] * x,
        p[2] + p[3] * u,
        b1,
    )
}

impl Executor<'_> {
    pub(super) fn load(&self, slot: u32) -> Result<G> {
        let slot = slot as usize;
        ensure!(slot < self.state.len(), "state slot {slot} out of bounds");
        ensure!(
            self.fast.is_some() || self.ready[slot].load(Ordering::Relaxed),
            "state slot {slot} read before it was written"
        );
        Ok(G::from_canonical_u64(
            self.state[slot].load(Ordering::Relaxed),
        ))
    }

    pub(super) fn store(&self, slot: u32, value: G) -> Result<()> {
        let slot = slot as usize;
        ensure!(slot < self.state.len(), "state slot {slot} out of bounds");
        let value = value.to_canonical_u64();
        if self.fast.is_some_and(|f| f.single[slot] != 0) {
            self.state[slot].store(value, Ordering::Relaxed);
            return Ok(());
        }
        if self.compact_readiness && slot < self.program.roots.len() {
            ensure!(
                self.state[slot].load(Ordering::Relaxed) == value,
                "conflicting write to state slot {slot}"
            );
            return Ok(());
        }
        let flag = if self.compact_readiness {
            self.fast
                .unwrap()
                .sparse_ready
                .as_ref()
                .unwrap()
                .binary_search(&(slot as u32))
                .map_err(|_| anyhow::anyhow!("uncertified repeated write to slot {slot}"))?
        } else {
            slot
        };
        if self.ready[flag].load(Ordering::Relaxed) {
            ensure!(
                self.state[slot].load(Ordering::Relaxed) == value,
                "conflicting write to state slot {slot}"
            );
            return Ok(());
        }
        self.state[slot].store(value, Ordering::Relaxed);
        self.ready[flag].store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Slot index of lane `k`'s `position`-th target in a group. Every plan
    /// access is bounds-checked so a malformed plan fails instead of panicking.
    fn target(&self, offset: u32, position: u32, count: u32, k: u32) -> Result<u32> {
        let index = offset as u64 + position as u64 * count as u64 + k as u64;
        self.program
            .targets
            .get(index as usize)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("plan target index {index} out of range"))
    }

    pub(super) fn input(&self, offset: u32, position: u32, count: u32, k: u32) -> Result<G> {
        self.load(self.target(offset, position, count, k)?)
    }

    pub(super) fn output(
        &self,
        offset: u32,
        position: u32,
        count: u32,
        k: u32,
        value: G,
    ) -> Result<()> {
        let slot = self.target(offset, position, count, k)?;
        self.store(slot, value)
    }

    /// Preparation certifies contiguous, exclusive, non-root destinations.
    /// Keep random-word validation even for benchmark-supplied streams.
    pub(super) fn step_random_span(
        &self,
        source: u32,
        destination: u32,
        start: u32,
        end: u32,
    ) -> Result<()> {
        let source = self
            .random
            .get(source as usize + start as usize..source as usize + end as usize)
            .ok_or_else(|| anyhow::anyhow!("random span outside supplied values"))?;
        let destination = self
            .state
            .get(destination as usize + start as usize..destination as usize + end as usize)
            .ok_or_else(|| anyhow::anyhow!("random span outside state"))?;
        ensure!(
            source.iter().all(|&v| v < G::ORDER),
            "random value is not canonical"
        );
        for (cell, &value) in destination.iter().zip(source) {
            cell.store(value, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Coefficient of lane `k` at `position` in a group's coefficient block,
    /// with checked arithmetic and a bounds check.
    pub(super) fn coefficient(&self, base: u32, position: u64, count: u32, k: u32) -> Result<u64> {
        let index = position
            .checked_mul(count as u64)
            .and_then(|v| v.checked_add(base as u64))
            .and_then(|v| v.checked_add(k as u64))
            .ok_or_else(|| anyhow::anyhow!("plan coefficient index overflows"))?;
        self.program
            .coefficients
            .get(index as usize)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("plan coefficient index {index} out of range"))
    }

    fn q_input(&self, offset: u32, start: u32, count: u32, k: u32) -> Result<Q> {
        let mut result = Q::ZERO;
        for i in 0..5 {
            result.0[i as usize] = self.input(offset, start + i, count, k)?;
        }
        Ok(result)
    }

    fn q_output(&self, offset: u32, start: u32, count: u32, k: u32, value: Q) -> Result<()> {
        for i in 0..5 {
            self.output(offset, start + i, count, k, value.0[i as usize])?;
        }
        Ok(())
    }

    fn step(&self, group: &super::Group, k: u32) -> Result<()> {
        let kind = group.kind;
        let count = group.count;
        let offset = group.targets_offset;
        match kind {
            Kind::Poseidon => self.step_poseidon(k, count, offset),
            Kind::Arithmetic => {
                let co = group.coefficients_offset()?;
                let x = self.input(offset, 0, count, k)?;
                let y = self.input(offset, 1, count, k)?;
                let z = self.input(offset, 2, count, k)?;
                let c0 = field(self.coefficient(co, 0, count, k)?);
                let c1 = field(self.coefficient(co, 1, count, k)?);
                self.output(offset, 3, count, k, x * y * c0 + z * c1)
            }
            Kind::ArithmeticChain => {
                let co = group.coefficients_offset()?;
                let n = group.outputs as usize;
                ensure!(n > 0 && n <= 8, "invalid arithmetic chain length");
                let mut locals = [G::ZERO; 8];
                for j in 0..n {
                    let read = |pos: u64| -> Result<G> {
                        let reference = self.coefficient(co, (5 * j) as u64 + pos, count, k)?;
                        if reference >> 63 != 0 {
                            let index = (reference & !(1u64 << 63)) as usize;
                            ensure!(index < j, "chain reads an unproduced temporary");
                            Ok(locals[index])
                        } else {
                            ensure!(
                                reference < group.inputs as u64,
                                "chain input outside its boundary"
                            );
                            self.input(offset, reference as u32, count, k)
                        }
                    };
                    let c0 = field(self.coefficient(co, (5 * j + 3) as u64, count, k)?);
                    let c1 = field(self.coefficient(co, (5 * j + 4) as u64, count, k)?);
                    let product = if c0 == G::ZERO {
                        G::ZERO
                    } else {
                        let product = read(0)? * read(1)?;
                        if c0 == G::ONE {
                            product
                        } else {
                            product * c0
                        }
                    };
                    let addend = if c1 == G::ZERO {
                        G::ZERO
                    } else {
                        let addend = read(2)?;
                        if c1 == G::ONE {
                            addend
                        } else {
                            addend * c1
                        }
                    };
                    locals[j] = product + addend;
                    self.output(offset, group.inputs + j as u32, count, k, locals[j])?;
                }
                Ok(())
            }
            Kind::Copy => {
                let x = self.input(offset, 0, count, k)?;
                self.output(offset, 1, count, k, x)
            }
            Kind::Constant => {
                let co = group.coefficients_offset()?;
                let value = field(self.coefficient(co, 0, count, k)?);
                self.output(offset, 0, count, k, value)
            }
            Kind::Equality => {
                let x = canonical(self.input(offset, 0, count, k)?);
                let y = canonical(self.input(offset, 1, count, k)?);
                let same = x == y;
                let flag = if same { G::ONE } else { G::ZERO };
                let inverse = if same { G::ZERO } else { inverse(x - y)? };
                self.output(offset, 2, count, k, flag)?;
                self.output(offset, 3, count, k, inverse)
            }
            Kind::ExtensionArithmetic => {
                let co = group.coefficients_offset()?;
                let a = [
                    self.input(offset, 0, count, k)?,
                    self.input(offset, 1, count, k)?,
                ];
                let b = [
                    self.input(offset, 2, count, k)?,
                    self.input(offset, 3, count, k)?,
                ];
                let z = [
                    self.input(offset, 4, count, k)?,
                    self.input(offset, 5, count, k)?,
                ];
                let product = ext_mul(a, b);
                let c0 = field(self.coefficient(co, 0, count, k)?);
                let c1 = field(self.coefficient(co, 1, count, k)?);
                self.output(offset, 6, count, k, product[0] * c0 + z[0] * c1)?;
                self.output(offset, 7, count, k, product[1] * c0 + z[1] * c1)
            }
            Kind::ExtensionMultiply => {
                let co = group.coefficients_offset()?;
                let a = [
                    self.input(offset, 0, count, k)?,
                    self.input(offset, 1, count, k)?,
                ];
                let b = [
                    self.input(offset, 2, count, k)?,
                    self.input(offset, 3, count, k)?,
                ];
                let product = ext_mul(a, b);
                let c = field(self.coefficient(co, 0, count, k)?);
                self.output(offset, 4, count, k, product[0] * c)?;
                self.output(offset, 5, count, k, product[1] * c)
            }
            Kind::RandomAccess => {
                let co = group.parameter()?;
                let vec_size = co;
                let access = self.input(offset, 0, count, k)?.to_canonical_u64();
                ensure!(
                    access < vec_size as u64,
                    "invalid index: random access {access} of {vec_size}"
                );
                let access = access as u32;
                let selected = self.input(offset, 1 + access, count, k)?;
                self.output(offset, 1 + vec_size, count, k, selected)?;
                let mut bits = 0;
                let mut size = vec_size;
                while size > 1 {
                    size >>= 1;
                    bits += 1;
                }
                for i in 0..bits {
                    let bit = G::from_canonical_u64(((access >> i) & 1) as u64);
                    self.output(offset, 2 + vec_size + i, count, k, bit)?;
                }
                Ok(())
            }
            Kind::BitSum => {
                ensure!(group.inputs <= 63, "invalid bit sum width");
                let mut word = 0u64;
                for i in (0..group.inputs).rev() {
                    let bit = self.input(offset, i, count, k)?.to_canonical_u64();
                    ensure!(bit <= 1, "invalid arithmetic: bit sum input is not Boolean");
                    word = (word << 1) | bit;
                }
                self.output(offset, group.inputs, count, k, G::from_canonical_u64(word))
            }
            Kind::BaseSplit => {
                let co = group.parameter()?;
                let limb_bits = co & 255;
                let limbs = co >> 8;
                let mask = (1u64 << limb_bits) - 1;
                let mut word = self.input(offset, 0, count, k)?.to_canonical_u64();
                for i in 0..limbs {
                    let limb = word & mask;
                    self.output(offset, 1 + i, count, k, G::from_canonical_u64(limb))?;
                    word >>= limb_bits;
                }
                ensure!(word == 0, "invalid arithmetic: base split overflow");
                Ok(())
            }
            Kind::WireSplit => {
                let co = group.parameter()?;
                let width = co & 255;
                let chunks = co >> 8;
                let mut word = self.input(offset, 0, count, k)?.to_canonical_u64();
                for i in 0..chunks {
                    let chunk = if width == 0 {
                        0
                    } else if width < 64 {
                        let chunk = word & ((1u64 << width) - 1);
                        word >>= width;
                        chunk
                    } else {
                        let chunk = word;
                        word = 0;
                        chunk
                    };
                    self.output(offset, 1 + i, count, k, G::from_canonical_u64(chunk))?;
                }
                ensure!(word == 0, "invalid arithmetic: wire split overflow");
                Ok(())
            }
            Kind::LowHigh => {
                let co = group.parameter()?;
                let width = co;
                let word = self.input(offset, 0, count, k)?.to_canonical_u64();
                let (low, high) = if width == 0 {
                    (0, word)
                } else if width < 64 {
                    (word & ((1u64 << width) - 1), word >> width)
                } else {
                    (word, 0)
                };
                self.output(offset, 1, count, k, G::from_canonical_u64(low))?;
                self.output(offset, 2, count, k, G::from_canonical_u64(high))
            }
            Kind::Reducing | Kind::ReducingExtension => {
                let co = group.parameter()?;
                let extension_coeffs = kind == Kind::ReducingExtension;
                let num_coeffs = co;
                let alpha = [
                    self.input(offset, 0, count, k)?,
                    self.input(offset, 1, count, k)?,
                ];
                let mut acc = [
                    self.input(offset, 2, count, k)?,
                    self.input(offset, 3, count, k)?,
                ];
                let input_len = if extension_coeffs {
                    4 + 2 * num_coeffs
                } else {
                    4 + num_coeffs
                };
                for i in 0..num_coeffs {
                    let product = ext_mul(acc, alpha);
                    if extension_coeffs {
                        let c0 = self.input(offset, 4 + 2 * i, count, k)?;
                        let c1 = self.input(offset, 5 + 2 * i, count, k)?;
                        acc = [product[0] + c0, product[1] + c1];
                    } else {
                        let c = self.input(offset, 4 + i, count, k)?;
                        acc = [product[0] + c, product[1]];
                    }
                    self.output(offset, input_len + 2 * i, count, k, acc[0])?;
                    self.output(offset, input_len + 2 * i + 1, count, k, acc[1])?;
                }
                if !extension_coeffs {
                    // The CPU base-coefficient generator records the final output twice.
                    self.output(offset, input_len + 2 * num_coeffs, count, k, acc[0])?;
                    self.output(offset, input_len + 2 * num_coeffs + 1, count, k, acc[1])?;
                }
                Ok(())
            }
            Kind::TableLookup => {
                let co = group.coefficients_offset()?;
                let metadata = co;
                let low = self.input(offset, 0, count, k)?;
                let high = self.input(offset, 1, count, k)?;
                let index = (low + high * G::from_canonical_u64(64)).to_canonical_u64();
                let rows = self.coefficient(metadata, 0, count, k)?;
                ensure!(
                    index < rows,
                    "invalid index: table lookup {index} of {rows}"
                );
                let index = index as u32;
                let output_len = self.coefficient(metadata, 1, count, k)? as u32;
                let start = self.coefficient(metadata, index as u64 + 2, count, k)? as u32;
                let end = self.coefficient(metadata, index as u64 + 3, count, k)? as u32;
                let total = self.coefficient(metadata, rows + 2, count, k)? as u32;
                for j in 0..output_len {
                    let value = if start + j < end {
                        self.input(offset, 2 + start + j, count, k)?
                    } else {
                        G::ZERO
                    };
                    self.output(offset, 2 + total + j, count, k, value)?;
                }
                Ok(())
            }
            Kind::ExtensionQuotient => {
                let n0 = self.input(offset, 0, count, k)?;
                let n1 = self.input(offset, 1, count, k)?;
                let d0 = self.input(offset, 2, count, k)?;
                let d1 = self.input(offset, 3, count, k)?;
                let w = G::from_canonical_u64(7);
                let norm = d0 * d0 - w * d1 * d1;
                let inverse = inverse(norm)?;
                let q0 = (n0 * d0 - w * n1 * d1) * inverse;
                let q1 = (n1 * d0 - n0 * d1) * inverse;
                self.output(offset, 4, count, k, q0)?;
                self.output(offset, 5, count, k, q1)
            }
            Kind::EcAddXu => {
                let co = group.coefficients_offset()?;
                let yx = self.q_input(offset, 6, count, k)?;
                let yu = self.q_input(offset, 11, count, k)?;
                let mut px = self.q_input(offset, 16, count, k)?;
                let mut pu = self.q_input(offset, 21, count, k)?;
                let b1 = field(self.coefficient(co, 0, count, k)?);
                let mut gx = Q::ZERO;
                let mut gu = Q::ZERO;
                for j in 0..5u32 {
                    gx.0[j as usize] = field(self.coefficient(co, 1 + j as u64, count, k)?);
                    gu.0[j as usize] = field(self.coefficient(co, 6 + j as u64, count, k)?);
                }
                for i in 0..3u32 {
                    [px, pu] = q_add_xu(px, pu, px, pu, b1)?;
                    self.q_output(offset, 106 - 30 * i, count, k, px)?;
                    self.q_output(offset, 111 - 30 * i, count, k, pu)?;
                    if canonical(self.input(offset, i, count, k)?) == G::ONE {
                        [px, pu] = q_add_xu(px, pu, gx, gu, b1)?;
                    }
                    self.q_output(offset, 96 - 30 * i, count, k, px)?;
                    self.q_output(offset, 101 - 30 * i, count, k, pu)?;
                    if canonical(self.input(offset, 3 + i, count, k)?) == G::ONE {
                        [px, pu] = q_add_xu(px, pu, yx, yu, b1)?;
                    }
                    self.q_output(offset, 86 - 30 * i, count, k, px)?;
                    self.q_output(offset, 91 - 30 * i, count, k, pu)?;
                }
                Ok(())
            }
            Kind::PoseidonMds => {
                let mut first = [G::ZERO; SPONGE_WIDTH];
                let mut second = [G::ZERO; SPONGE_WIDTH];
                for i in 0..SPONGE_WIDTH as u32 {
                    first[i as usize] = self.input(offset, 2 * i, count, k)?;
                    second[i as usize] = self.input(offset, 2 * i + 1, count, k)?;
                }
                let out0 = <G as Poseidon>::mds_layer(&first);
                let out1 = <G as Poseidon>::mds_layer(&second);
                for i in 0..SPONGE_WIDTH as u32 {
                    self.output(offset, 24 + 2 * i, count, k, out0[i as usize])?;
                    self.output(offset, 25 + 2 * i, count, k, out1[i as usize])?;
                }
                Ok(())
            }
            Kind::RandomValue => {
                let co = group.coefficients_offset()?;
                let index = self.coefficient(co, 0, count, k)?;
                let value = *self
                    .random
                    .get(usize::try_from(index).unwrap_or(usize::MAX))
                    .ok_or_else(|| anyhow::anyhow!("plan random index {index} out of range"))?;
                ensure!(value < G::ORDER, "random value {value} is not canonical");
                self.output(offset, 0, count, k, G::from_canonical_u64(value))
            }
            Kind::ConditionalZero => {
                let denominator = canonical(self.input(offset, 0, count, k)?);
                let numerator = self.input(offset, 1, count, k)?;
                let quotient = if denominator != G::ZERO {
                    numerator * inverse(denominator)?
                } else {
                    G::ZERO
                };
                self.output(offset, 2, count, k, quotient)
            }
            Kind::LtMask => {
                let co = group.parameter()?;
                let length = co;
                let n = self.input(offset, 0, count, k)?.to_canonical_u64();
                for j in 0..length {
                    let bit = if (j as u64) < n { G::ONE } else { G::ZERO };
                    self.output(offset, j + 1, count, k, bit)?;
                }
                Ok(())
            }
            Kind::QuinticMultiply => {
                let lhs = self.q_input(offset, 0, count, k)?;
                let rhs = self.q_input(offset, 5, count, k)?;
                self.q_output(offset, 10, count, k, lhs * rhs)
            }
            Kind::QuinticQuotient => {
                let numerator = self.q_input(offset, 0, count, k)?;
                let denominator = self.q_input(offset, 5, count, k)?;
                let quotient = numerator * q_inverse(denominator)?;
                self.q_output(offset, 10, count, k, quotient)
            }
            Kind::EcAddHomogOffset => {
                let co = group.coefficients_offset()?;
                let b1 = field(self.coefficient(co, 0, count, k)?);
                let mut h = q_add_homog(
                    self.q_input(offset, 0, count, k)?,
                    self.q_input(offset, 5, count, k)?,
                    self.q_input(offset, 10, count, k)?,
                    self.q_input(offset, 15, count, k)?,
                    b1,
                );
                h[1].0[1] -= b1;
                h[3].0[1] -= b1;
                for (i, value) in h.into_iter().enumerate() {
                    self.q_output(offset, 20 + 5 * i as u32, count, k, value)?;
                }
                Ok(())
            }
            Kind::EcScalarMul => {
                let co = group.coefficients_offset()?;
                let px = self.q_input(offset, 0, count, k)?;
                let pu = self.q_input(offset, 5, count, k)?;
                let words = self.coefficient(co, 0, count, k)? as u32;
                let b1 = field(self.coefficient(co, 1, count, k)?);
                let mut point = [Q::ZERO, Q::ONE, Q::ZERO, Q::ONE];
                for bit in (0..words * 32).rev() {
                    point = q_double_fractional(point, b1)?;
                    let word = self.coefficient(co, 2 + (bit / 32) as u64, count, k)?;
                    if word & (1 << (bit % 32)) != 0 {
                        point = q_add_mixed_fractional(point, px, pu, b1)?;
                    }
                }
                let result = q_div_pair(point[0], point[1], point[2], point[3])?;
                self.q_output(offset, 10, count, k, result[0])?;
                self.q_output(offset, 15, count, k, result[1])
            }
            Kind::CosetInterpolation => {
                let co = group.coefficients_offset()?;
                let degree = self.coefficient(co, 0, count, k)? as u32;
                let n = self.coefficient(co, 1, count, k)? as u32;
                let shift = self.input(offset, 0, count, k)?;
                let inv_shift = inverse(shift)?;
                let z = [
                    self.input(offset, 1, count, k)? * inv_shift,
                    self.input(offset, 2, count, k)? * inv_shift,
                ];
                let output_start = 3 + 2 * n;
                self.output(offset, output_start, count, k, z[0])?;
                self.output(offset, output_start + 1, count, k, z[1])?;
                let mut e = [G::ZERO; 2];
                let mut p = [G::ONE, G::ZERO];
                let mut boundary = degree;
                let mut intermediate = 0;
                let num_intermediates = (n - 2) / (degree - 1);
                for j in 0..n {
                    let x = field(self.coefficient(co, 2 + j as u64, count, k)?);
                    let weight = field(self.coefficient(co, 2 + n as u64 + j as u64, count, k)?);
                    let t = [z[0] - x, z[1]];
                    let v = [
                        self.input(offset, 3 + 2 * j, count, k)?,
                        self.input(offset, 4 + 2 * j, count, k)?,
                    ];
                    let et = ext_mul(e, t);
                    let vp = ext_mul(v, p);
                    e = [et[0] + vp[0] * weight, et[1] + vp[1] * weight];
                    p = ext_mul(p, t);
                    if j + 1 == boundary && intermediate < num_intermediates {
                        let out = output_start + 2 + 4 * intermediate;
                        self.output(offset, out, count, k, e[0])?;
                        self.output(offset, out + 1, count, k, e[1])?;
                        self.output(offset, out + 2, count, k, p[0])?;
                        self.output(offset, out + 3, count, k, p[1])?;
                        intermediate += 1;
                        boundary += degree - 1;
                    }
                }
                let final_out = output_start + 2 + 4 * num_intermediates;
                self.output(offset, final_out, count, k, e[0])?;
                self.output(offset, final_out + 1, count, k, e[1])
            }
            Kind::Comparison => {
                let co = group.coefficients_offset()?;
                let n = self.coefficient(co, 0, count, k)? as u32;
                let width = self.coefficient(co, 1, count, k)? as u32;
                ensure!(
                    n != 0 && width != 0 && width < 64,
                    "invalid arithmetic: comparison shape"
                );
                let first = self.input(offset, 0, count, k)?.to_canonical_u64();
                let second = self.input(offset, 1, count, k)?.to_canonical_u64();
                let out = 2;
                let less_or_equal = if first <= second { G::ONE } else { G::ZERO };
                self.output(offset, out, count, k, less_or_equal)?;
                let mut msd = G::ZERO;
                for i in 0..n {
                    let mut fchunk = G::ZERO;
                    let mut schunk = G::ZERO;
                    let mut power = G::ONE;
                    for j in 0..width {
                        let index = (i * width + j) as u64;
                        if bit(first, index) != 0 {
                            fchunk += power;
                        }
                        if bit(second, index) != 0 {
                            schunk += power;
                        }
                        power += power;
                    }
                    self.output(offset, out + 2 + i, count, k, fchunk)?;
                    self.output(offset, out + 2 + n + i, count, k, schunk)?;
                    let equal = fchunk == schunk;
                    let dummy = if equal {
                        G::ONE
                    } else {
                        inverse(schunk - fchunk)?
                    };
                    self.output(offset, out + 2 + 2 * n + i, count, k, dummy)?;
                    let equal_flag = if equal { G::ONE } else { G::ZERO };
                    self.output(offset, out + 2 + 3 * n + i, count, k, equal_flag)?;
                    let carried = if equal { msd } else { G::ZERO };
                    self.output(offset, out + 2 + 4 * n + i, count, k, carried)?;
                    if !equal {
                        msd = schunk - fchunk;
                    }
                }
                self.output(offset, out + 1, count, k, msd)?;
                let mut two_n = G::ONE;
                for _ in 0..width {
                    two_n += two_n;
                }
                let sum = (two_n + msd).to_canonical_u64();
                for j in 0..=width {
                    let value = G::from_canonical_u64(bit(sum, j as u64));
                    self.output(offset, out + 2 + 5 * n + j, count, k, value)?;
                }
                Ok(())
            }
        }
    }

    pub(super) fn step_fixed<const KIND: u8>(&self, group: &super::Group, k: u32) -> Result<()> {
        let kind = Kind::from_code(KIND as usize)?;
        let count = group.count;
        let offset = group.targets_offset;
        match kind {
            Kind::Poseidon => self.step_poseidon(k, count, offset),
            Kind::Arithmetic => {
                let co = group.coefficients_offset()?;
                let x = self.input(offset, 0, count, k)?;
                let y = self.input(offset, 1, count, k)?;
                let z = self.input(offset, 2, count, k)?;
                let c0 = field(self.coefficient(co, 0, count, k)?);
                let c1 = field(self.coefficient(co, 1, count, k)?);
                self.output(offset, 3, count, k, x * y * c0 + z * c1)
            }
            Kind::ArithmeticChain => {
                let co = group.coefficients_offset()?;
                let n = group.outputs as usize;
                ensure!(n > 0 && n <= 8, "invalid arithmetic chain length");
                let mut locals = [G::ZERO; 8];
                for j in 0..n {
                    let read = |pos: u64| -> Result<G> {
                        let reference = self.coefficient(co, (5 * j) as u64 + pos, count, k)?;
                        if reference >> 63 != 0 {
                            let index = (reference & !(1u64 << 63)) as usize;
                            ensure!(index < j, "chain reads an unproduced temporary");
                            Ok(locals[index])
                        } else {
                            ensure!(
                                reference < group.inputs as u64,
                                "chain input outside its boundary"
                            );
                            self.input(offset, reference as u32, count, k)
                        }
                    };
                    let c0 = field(self.coefficient(co, (5 * j + 3) as u64, count, k)?);
                    let c1 = field(self.coefficient(co, (5 * j + 4) as u64, count, k)?);
                    let product = if c0 == G::ZERO {
                        G::ZERO
                    } else {
                        let product = read(0)? * read(1)?;
                        if c0 == G::ONE {
                            product
                        } else {
                            product * c0
                        }
                    };
                    let addend = if c1 == G::ZERO {
                        G::ZERO
                    } else {
                        let addend = read(2)?;
                        if c1 == G::ONE {
                            addend
                        } else {
                            addend * c1
                        }
                    };
                    locals[j] = product + addend;
                    self.output(offset, group.inputs + j as u32, count, k, locals[j])?;
                }
                Ok(())
            }
            Kind::Copy => {
                let x = self.input(offset, 0, count, k)?;
                self.output(offset, 1, count, k, x)
            }
            Kind::Constant => {
                let co = group.coefficients_offset()?;
                let value = field(self.coefficient(co, 0, count, k)?);
                self.output(offset, 0, count, k, value)
            }
            Kind::Equality => {
                let x = canonical(self.input(offset, 0, count, k)?);
                let y = canonical(self.input(offset, 1, count, k)?);
                let same = x == y;
                let flag = if same { G::ONE } else { G::ZERO };
                let inverse = if same { G::ZERO } else { inverse(x - y)? };
                self.output(offset, 2, count, k, flag)?;
                self.output(offset, 3, count, k, inverse)
            }
            Kind::ExtensionArithmetic => {
                let co = group.coefficients_offset()?;
                let a = [
                    self.input(offset, 0, count, k)?,
                    self.input(offset, 1, count, k)?,
                ];
                let b = [
                    self.input(offset, 2, count, k)?,
                    self.input(offset, 3, count, k)?,
                ];
                let z = [
                    self.input(offset, 4, count, k)?,
                    self.input(offset, 5, count, k)?,
                ];
                let product = ext_mul(a, b);
                let c0 = field(self.coefficient(co, 0, count, k)?);
                let c1 = field(self.coefficient(co, 1, count, k)?);
                self.output(offset, 6, count, k, product[0] * c0 + z[0] * c1)?;
                self.output(offset, 7, count, k, product[1] * c0 + z[1] * c1)
            }
            Kind::ExtensionMultiply => {
                let co = group.coefficients_offset()?;
                let a = [
                    self.input(offset, 0, count, k)?,
                    self.input(offset, 1, count, k)?,
                ];
                let b = [
                    self.input(offset, 2, count, k)?,
                    self.input(offset, 3, count, k)?,
                ];
                let product = ext_mul(a, b);
                let c = field(self.coefficient(co, 0, count, k)?);
                self.output(offset, 4, count, k, product[0] * c)?;
                self.output(offset, 5, count, k, product[1] * c)
            }
            Kind::RandomAccess => {
                let co = group.parameter()?;
                let vec_size = co;
                let access = self.input(offset, 0, count, k)?.to_canonical_u64();
                ensure!(
                    access < vec_size as u64,
                    "invalid index: random access {access} of {vec_size}"
                );
                let access = access as u32;
                let selected = self.input(offset, 1 + access, count, k)?;
                self.output(offset, 1 + vec_size, count, k, selected)?;
                let mut bits = 0;
                let mut size = vec_size;
                while size > 1 {
                    size >>= 1;
                    bits += 1;
                }
                for i in 0..bits {
                    let bit = G::from_canonical_u64(((access >> i) & 1) as u64);
                    self.output(offset, 2 + vec_size + i, count, k, bit)?;
                }
                Ok(())
            }
            Kind::BitSum => {
                ensure!(group.inputs <= 63, "invalid bit sum width");
                let mut word = 0u64;
                for i in (0..group.inputs).rev() {
                    let bit = self.input(offset, i, count, k)?.to_canonical_u64();
                    ensure!(bit <= 1, "invalid arithmetic: bit sum input is not Boolean");
                    word = (word << 1) | bit;
                }
                self.output(offset, group.inputs, count, k, G::from_canonical_u64(word))
            }
            Kind::BaseSplit => {
                let co = group.parameter()?;
                let limb_bits = co & 255;
                let limbs = co >> 8;
                let mask = (1u64 << limb_bits) - 1;
                let mut word = self.input(offset, 0, count, k)?.to_canonical_u64();
                for i in 0..limbs {
                    let limb = word & mask;
                    self.output(offset, 1 + i, count, k, G::from_canonical_u64(limb))?;
                    word >>= limb_bits;
                }
                ensure!(word == 0, "invalid arithmetic: base split overflow");
                Ok(())
            }
            Kind::WireSplit => {
                let co = group.parameter()?;
                let width = co & 255;
                let chunks = co >> 8;
                let mut word = self.input(offset, 0, count, k)?.to_canonical_u64();
                for i in 0..chunks {
                    let chunk = if width == 0 {
                        0
                    } else if width < 64 {
                        let chunk = word & ((1u64 << width) - 1);
                        word >>= width;
                        chunk
                    } else {
                        let chunk = word;
                        word = 0;
                        chunk
                    };
                    self.output(offset, 1 + i, count, k, G::from_canonical_u64(chunk))?;
                }
                ensure!(word == 0, "invalid arithmetic: wire split overflow");
                Ok(())
            }
            Kind::LowHigh => {
                let co = group.parameter()?;
                let width = co;
                let word = self.input(offset, 0, count, k)?.to_canonical_u64();
                let (low, high) = if width == 0 {
                    (0, word)
                } else if width < 64 {
                    (word & ((1u64 << width) - 1), word >> width)
                } else {
                    (word, 0)
                };
                self.output(offset, 1, count, k, G::from_canonical_u64(low))?;
                self.output(offset, 2, count, k, G::from_canonical_u64(high))
            }
            Kind::Reducing | Kind::ReducingExtension => {
                let co = group.parameter()?;
                let extension_coeffs = kind == Kind::ReducingExtension;
                let num_coeffs = co;
                let alpha = [
                    self.input(offset, 0, count, k)?,
                    self.input(offset, 1, count, k)?,
                ];
                let mut acc = [
                    self.input(offset, 2, count, k)?,
                    self.input(offset, 3, count, k)?,
                ];
                let input_len = if extension_coeffs {
                    4 + 2 * num_coeffs
                } else {
                    4 + num_coeffs
                };
                for i in 0..num_coeffs {
                    let product = ext_mul(acc, alpha);
                    if extension_coeffs {
                        let c0 = self.input(offset, 4 + 2 * i, count, k)?;
                        let c1 = self.input(offset, 5 + 2 * i, count, k)?;
                        acc = [product[0] + c0, product[1] + c1];
                    } else {
                        let c = self.input(offset, 4 + i, count, k)?;
                        acc = [product[0] + c, product[1]];
                    }
                    self.output(offset, input_len + 2 * i, count, k, acc[0])?;
                    self.output(offset, input_len + 2 * i + 1, count, k, acc[1])?;
                }
                if !extension_coeffs {
                    // The CPU base-coefficient generator records the final output twice.
                    self.output(offset, input_len + 2 * num_coeffs, count, k, acc[0])?;
                    self.output(offset, input_len + 2 * num_coeffs + 1, count, k, acc[1])?;
                }
                Ok(())
            }
            Kind::TableLookup => {
                let co = group.coefficients_offset()?;
                let metadata = co;
                let low = self.input(offset, 0, count, k)?;
                let high = self.input(offset, 1, count, k)?;
                let index = (low + high * G::from_canonical_u64(64)).to_canonical_u64();
                let rows = self.coefficient(metadata, 0, count, k)?;
                ensure!(
                    index < rows,
                    "invalid index: table lookup {index} of {rows}"
                );
                let index = index as u32;
                let output_len = self.coefficient(metadata, 1, count, k)? as u32;
                let start = self.coefficient(metadata, index as u64 + 2, count, k)? as u32;
                let end = self.coefficient(metadata, index as u64 + 3, count, k)? as u32;
                let total = self.coefficient(metadata, rows + 2, count, k)? as u32;
                for j in 0..output_len {
                    let value = if start + j < end {
                        self.input(offset, 2 + start + j, count, k)?
                    } else {
                        G::ZERO
                    };
                    self.output(offset, 2 + total + j, count, k, value)?;
                }
                Ok(())
            }
            Kind::ExtensionQuotient => {
                let n0 = self.input(offset, 0, count, k)?;
                let n1 = self.input(offset, 1, count, k)?;
                let d0 = self.input(offset, 2, count, k)?;
                let d1 = self.input(offset, 3, count, k)?;
                let w = G::from_canonical_u64(7);
                let norm = d0 * d0 - w * d1 * d1;
                let inverse = inverse(norm)?;
                let q0 = (n0 * d0 - w * n1 * d1) * inverse;
                let q1 = (n1 * d0 - n0 * d1) * inverse;
                self.output(offset, 4, count, k, q0)?;
                self.output(offset, 5, count, k, q1)
            }
            Kind::EcAddXu => {
                let co = group.coefficients_offset()?;
                let yx = self.q_input(offset, 6, count, k)?;
                let yu = self.q_input(offset, 11, count, k)?;
                let mut px = self.q_input(offset, 16, count, k)?;
                let mut pu = self.q_input(offset, 21, count, k)?;
                let b1 = field(self.coefficient(co, 0, count, k)?);
                let mut gx = Q::ZERO;
                let mut gu = Q::ZERO;
                for j in 0..5u32 {
                    gx.0[j as usize] = field(self.coefficient(co, 1 + j as u64, count, k)?);
                    gu.0[j as usize] = field(self.coefficient(co, 6 + j as u64, count, k)?);
                }
                for i in 0..3u32 {
                    [px, pu] = q_add_xu(px, pu, px, pu, b1)?;
                    self.q_output(offset, 106 - 30 * i, count, k, px)?;
                    self.q_output(offset, 111 - 30 * i, count, k, pu)?;
                    if canonical(self.input(offset, i, count, k)?) == G::ONE {
                        [px, pu] = q_add_xu(px, pu, gx, gu, b1)?;
                    }
                    self.q_output(offset, 96 - 30 * i, count, k, px)?;
                    self.q_output(offset, 101 - 30 * i, count, k, pu)?;
                    if canonical(self.input(offset, 3 + i, count, k)?) == G::ONE {
                        [px, pu] = q_add_xu(px, pu, yx, yu, b1)?;
                    }
                    self.q_output(offset, 86 - 30 * i, count, k, px)?;
                    self.q_output(offset, 91 - 30 * i, count, k, pu)?;
                }
                Ok(())
            }
            Kind::PoseidonMds => {
                let mut first = [G::ZERO; SPONGE_WIDTH];
                let mut second = [G::ZERO; SPONGE_WIDTH];
                for i in 0..SPONGE_WIDTH as u32 {
                    first[i as usize] = self.input(offset, 2 * i, count, k)?;
                    second[i as usize] = self.input(offset, 2 * i + 1, count, k)?;
                }
                let out0 = <G as Poseidon>::mds_layer(&first);
                let out1 = <G as Poseidon>::mds_layer(&second);
                for i in 0..SPONGE_WIDTH as u32 {
                    self.output(offset, 24 + 2 * i, count, k, out0[i as usize])?;
                    self.output(offset, 25 + 2 * i, count, k, out1[i as usize])?;
                }
                Ok(())
            }
            Kind::RandomValue => {
                let co = group.coefficients_offset()?;
                let index = self.coefficient(co, 0, count, k)?;
                let value = *self
                    .random
                    .get(usize::try_from(index).unwrap_or(usize::MAX))
                    .ok_or_else(|| anyhow::anyhow!("plan random index {index} out of range"))?;
                ensure!(value < G::ORDER, "random value {value} is not canonical");
                self.output(offset, 0, count, k, G::from_canonical_u64(value))
            }
            Kind::ConditionalZero => {
                let denominator = canonical(self.input(offset, 0, count, k)?);
                let numerator = self.input(offset, 1, count, k)?;
                let quotient = if denominator != G::ZERO {
                    numerator * inverse(denominator)?
                } else {
                    G::ZERO
                };
                self.output(offset, 2, count, k, quotient)
            }
            Kind::LtMask => {
                let co = group.parameter()?;
                let length = co;
                let n = self.input(offset, 0, count, k)?.to_canonical_u64();
                for j in 0..length {
                    let bit = if (j as u64) < n { G::ONE } else { G::ZERO };
                    self.output(offset, j + 1, count, k, bit)?;
                }
                Ok(())
            }
            Kind::QuinticMultiply => {
                let lhs = self.q_input(offset, 0, count, k)?;
                let rhs = self.q_input(offset, 5, count, k)?;
                self.q_output(offset, 10, count, k, lhs * rhs)
            }
            Kind::QuinticQuotient => {
                let numerator = self.q_input(offset, 0, count, k)?;
                let denominator = self.q_input(offset, 5, count, k)?;
                let quotient = numerator * q_inverse(denominator)?;
                self.q_output(offset, 10, count, k, quotient)
            }
            Kind::EcAddHomogOffset => {
                let co = group.coefficients_offset()?;
                let b1 = field(self.coefficient(co, 0, count, k)?);
                let mut h = q_add_homog(
                    self.q_input(offset, 0, count, k)?,
                    self.q_input(offset, 5, count, k)?,
                    self.q_input(offset, 10, count, k)?,
                    self.q_input(offset, 15, count, k)?,
                    b1,
                );
                h[1].0[1] -= b1;
                h[3].0[1] -= b1;
                for (i, value) in h.into_iter().enumerate() {
                    self.q_output(offset, 20 + 5 * i as u32, count, k, value)?;
                }
                Ok(())
            }
            Kind::EcScalarMul => {
                let co = group.coefficients_offset()?;
                let px = self.q_input(offset, 0, count, k)?;
                let pu = self.q_input(offset, 5, count, k)?;
                let words = self.coefficient(co, 0, count, k)? as u32;
                let b1 = field(self.coefficient(co, 1, count, k)?);
                let mut point = [Q::ZERO, Q::ONE, Q::ZERO, Q::ONE];
                for bit in (0..words * 32).rev() {
                    point = q_double_fractional(point, b1)?;
                    let word = self.coefficient(co, 2 + (bit / 32) as u64, count, k)?;
                    if word & (1 << (bit % 32)) != 0 {
                        point = q_add_mixed_fractional(point, px, pu, b1)?;
                    }
                }
                let result = q_div_pair(point[0], point[1], point[2], point[3])?;
                self.q_output(offset, 10, count, k, result[0])?;
                self.q_output(offset, 15, count, k, result[1])
            }
            Kind::CosetInterpolation => {
                let co = group.coefficients_offset()?;
                let degree = self.coefficient(co, 0, count, k)? as u32;
                let n = self.coefficient(co, 1, count, k)? as u32;
                let shift = self.input(offset, 0, count, k)?;
                let inv_shift = inverse(shift)?;
                let z = [
                    self.input(offset, 1, count, k)? * inv_shift,
                    self.input(offset, 2, count, k)? * inv_shift,
                ];
                let output_start = 3 + 2 * n;
                self.output(offset, output_start, count, k, z[0])?;
                self.output(offset, output_start + 1, count, k, z[1])?;
                let mut e = [G::ZERO; 2];
                let mut p = [G::ONE, G::ZERO];
                let mut boundary = degree;
                let mut intermediate = 0;
                let num_intermediates = (n - 2) / (degree - 1);
                for j in 0..n {
                    let x = field(self.coefficient(co, 2 + j as u64, count, k)?);
                    let weight = field(self.coefficient(co, 2 + n as u64 + j as u64, count, k)?);
                    let t = [z[0] - x, z[1]];
                    let v = [
                        self.input(offset, 3 + 2 * j, count, k)?,
                        self.input(offset, 4 + 2 * j, count, k)?,
                    ];
                    let et = ext_mul(e, t);
                    let vp = ext_mul(v, p);
                    e = [et[0] + vp[0] * weight, et[1] + vp[1] * weight];
                    p = ext_mul(p, t);
                    if j + 1 == boundary && intermediate < num_intermediates {
                        let out = output_start + 2 + 4 * intermediate;
                        self.output(offset, out, count, k, e[0])?;
                        self.output(offset, out + 1, count, k, e[1])?;
                        self.output(offset, out + 2, count, k, p[0])?;
                        self.output(offset, out + 3, count, k, p[1])?;
                        intermediate += 1;
                        boundary += degree - 1;
                    }
                }
                let final_out = output_start + 2 + 4 * num_intermediates;
                self.output(offset, final_out, count, k, e[0])?;
                self.output(offset, final_out + 1, count, k, e[1])
            }
            Kind::Comparison => {
                let co = group.coefficients_offset()?;
                let n = self.coefficient(co, 0, count, k)? as u32;
                let width = self.coefficient(co, 1, count, k)? as u32;
                ensure!(
                    n != 0 && width != 0 && width < 64,
                    "invalid arithmetic: comparison shape"
                );
                let first = self.input(offset, 0, count, k)?.to_canonical_u64();
                let second = self.input(offset, 1, count, k)?.to_canonical_u64();
                let out = 2;
                let less_or_equal = if first <= second { G::ONE } else { G::ZERO };
                self.output(offset, out, count, k, less_or_equal)?;
                let mut msd = G::ZERO;
                for i in 0..n {
                    let mut fchunk = G::ZERO;
                    let mut schunk = G::ZERO;
                    let mut power = G::ONE;
                    for j in 0..width {
                        let index = (i * width + j) as u64;
                        if bit(first, index) != 0 {
                            fchunk += power;
                        }
                        if bit(second, index) != 0 {
                            schunk += power;
                        }
                        power += power;
                    }
                    self.output(offset, out + 2 + i, count, k, fchunk)?;
                    self.output(offset, out + 2 + n + i, count, k, schunk)?;
                    let equal = fchunk == schunk;
                    let dummy = if equal {
                        G::ONE
                    } else {
                        inverse(schunk - fchunk)?
                    };
                    self.output(offset, out + 2 + 2 * n + i, count, k, dummy)?;
                    let equal_flag = if equal { G::ONE } else { G::ZERO };
                    self.output(offset, out + 2 + 3 * n + i, count, k, equal_flag)?;
                    let carried = if equal { msd } else { G::ZERO };
                    self.output(offset, out + 2 + 4 * n + i, count, k, carried)?;
                    if !equal {
                        msd = schunk - fchunk;
                    }
                }
                self.output(offset, out + 1, count, k, msd)?;
                let mut two_n = G::ONE;
                for _ in 0..width {
                    two_n += two_n;
                }
                let sum = (two_n + msd).to_canonical_u64();
                for j in 0..=width {
                    let value = G::from_canonical_u64(bit(sum, j as u64));
                    self.output(offset, out + 2 + 5 * n + j, count, k, value)?;
                }
                Ok(())
            }
        }
    }

    /// The standard Poseidon gate generator, emitting the same 122 values in
    /// the same order: deltas, S-box inputs, final outputs.
    fn step_poseidon(&self, k: u32, count: u32, offset: u32) -> Result<()> {
        let mut state = [G::ZERO; SPONGE_WIDTH];
        for (i, value) in state.iter_mut().enumerate() {
            *value = self.input(offset, i as u32, count, k)?;
        }
        let swap = canonical(self.input(offset, SPONGE_WIDTH as u32, count, k)?);
        ensure!(
            swap == G::ZERO || swap == G::ONE,
            "invalid Poseidon swap flag"
        );
        let mut slot = SPONGE_WIDTH as u32 + 1;
        let mut emit = |executor: &Self, value: G| -> Result<()> {
            executor.output(offset, slot, count, k, value)?;
            slot += 1;
            Ok(())
        };
        for i in 0..4 {
            emit(self, swap * (state[i + 4] - state[i]))?;
        }
        if swap == G::ONE {
            for i in 0..4 {
                state.swap(i, 4 + i);
            }
        }
        let mut round_ctr = 0;
        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            <G as Poseidon>::constant_layer_field(&mut state, round_ctr);
            if r != 0 {
                for value in state {
                    emit(self, value)?;
                }
            }
            <G as Poseidon>::sbox_layer(&mut state);
            state = <G as Poseidon>::mds_layer(&state);
            round_ctr += 1;
        }
        <G as Poseidon>::partial_first_constant_layer(&mut state);
        state = <G as Poseidon>::mds_partial_layer_init(&state);
        for r in 0..(poseidon::N_PARTIAL_ROUNDS - 1) {
            emit(self, state[0])?;
            state[0] = <G as Poseidon>::sbox_monomial(state[0]);
            state[0] += G::from_canonical_u64(<G as Poseidon>::FAST_PARTIAL_ROUND_CONSTANTS[r]);
            state = <G as Poseidon>::mds_partial_layer_fast(&state, r);
        }
        emit(self, state[0])?;
        state[0] = <G as Poseidon>::sbox_monomial(state[0]);
        state = <G as Poseidon>::mds_partial_layer_fast(&state, poseidon::N_PARTIAL_ROUNDS - 1);
        round_ctr += poseidon::N_PARTIAL_ROUNDS;
        for _ in 0..poseidon::HALF_N_FULL_ROUNDS {
            <G as Poseidon>::constant_layer_field(&mut state, round_ctr);
            for value in state {
                emit(self, value)?;
            }
            <G as Poseidon>::sbox_layer(&mut state);
            state = <G as Poseidon>::mds_layer(&state);
            round_ctr += 1;
        }
        for value in state {
            emit(self, value)?;
        }
        Ok(())
    }

    pub(super) fn step_poseidon_simd(
        &self,
        g: &super::Group,
        start: u32,
        end: u32,
        bases: &[u32],
        _: u8,
    ) -> Result<()> {
        let k = {
            #[cfg(target_arch = "x86_64")]
            {
                let mut k = start;
                while crate::hash::poseidon_trace_simd::native_available() && k + 8 <= end {
                    self.poseidon_batch8(g, k, bases)?;
                    k += 8;
                }
                k
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                start
            }
        };
        for k in k..end {
            if bases[k as usize] != u32::MAX {
                self.step_poseidon_bulk(k, g.count, g.targets_offset, bases[k as usize])?;
            } else {
                self.step_fixed::<0>(g, k)?;
            }
        }
        Ok(())
    }
    #[cfg(target_arch = "x86_64")]
    fn poseidon_batch8(&self, g: &super::Group, k: u32, bases: &[u32]) -> Result<()> {
        use crate::hash::poseidon_trace_simd::{trace8, Lanes};
        let mut input = [Lanes([G::ZERO; 8]); 12];
        let mut swap = Lanes([G::ZERO; 8]);
        let pack = self.profile.poseidon(PoseidonPart::Input);
        for lane in 0..8 {
            for i in 0..12 {
                input[i].0[lane] =
                    self.input(g.targets_offset, i as u32, g.count, k + lane as u32)?;
            }
            swap.0[lane] = canonical(self.input(g.targets_offset, 12, g.count, k + lane as u32)?);
            ensure!(
                swap.0[lane] == G::ZERO || swap.0[lane] == G::ONE,
                "invalid Poseidon swap flag"
            );
        }
        drop(pack);
        let arithmetic = self.profile.poseidon(PoseidonPart::Arithmetic);
        let out = trace8(input, swap);
        drop(arithmetic);
        let _output = self.profile.poseidon(PoseidonPart::Output);
        for lane in 0..8 {
            let values = core::array::from_fn(|i| out[i].0[lane]);
            self.store_poseidon(g, k + lane as u32, bases[k as usize + lane], &values)?;
        }
        Ok(())
    }
    #[cfg(target_arch = "x86_64")]
    fn store_poseidon(&self, g: &super::Group, k: u32, base: u32, values: &[G; 122]) -> Result<()> {
        if base == u32::MAX {
            for (i, &v) in values.iter().enumerate() {
                self.output(g.targets_offset, 13 + i as u32, g.count, k, v)?;
            }
        } else {
            let destination = self
                .state
                .get(base as usize..base as usize + 122)
                .ok_or_else(|| anyhow::anyhow!("Poseidon block outside state"))?;
            for (cell, value) in destination.iter().zip(values) {
                let value = value.to_canonical_u64();
                cell.store(value, Ordering::Relaxed);
            }
        }
        Ok(())
    }
    pub(super) fn step_poseidon_bulk(
        &self,
        k: u32,
        count: u32,
        offset: u32,
        base: u32,
    ) -> Result<()> {
        let mut state = [G::ZERO; SPONGE_WIDTH];
        for (i, value) in state.iter_mut().enumerate() {
            *value = self.input(offset, i as u32, count, k)?;
        }
        let swap = canonical(self.input(offset, SPONGE_WIDTH as u32, count, k)?);
        ensure!(
            swap == G::ZERO || swap == G::ONE,
            "invalid Poseidon swap flag"
        );
        let mut values = [G::ZERO; 122];
        let mut written = 0;
        let mut emit = |_: &Self, value: G| -> Result<()> {
            values[written] = value;
            written += 1;
            Ok(())
        };
        for i in 0..4 {
            emit(self, swap * (state[i + 4] - state[i]))?;
        }
        if swap == G::ONE {
            for i in 0..4 {
                state.swap(i, 4 + i);
            }
        }
        let mut round_ctr = 0;
        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            <G as Poseidon>::constant_layer_field(&mut state, round_ctr);
            if r != 0 {
                for value in state {
                    emit(self, value)?;
                }
            }
            <G as Poseidon>::sbox_layer(&mut state);
            state = <G as Poseidon>::mds_layer(&state);
            round_ctr += 1;
        }
        <G as Poseidon>::partial_first_constant_layer(&mut state);
        state = <G as Poseidon>::mds_partial_layer_init(&state);
        for r in 0..(poseidon::N_PARTIAL_ROUNDS - 1) {
            emit(self, state[0])?;
            state[0] = <G as Poseidon>::sbox_monomial(state[0]);
            state[0] += G::from_canonical_u64(<G as Poseidon>::FAST_PARTIAL_ROUND_CONSTANTS[r]);
            state = <G as Poseidon>::mds_partial_layer_fast(&state, r);
        }
        emit(self, state[0])?;
        state[0] = <G as Poseidon>::sbox_monomial(state[0]);
        state = <G as Poseidon>::mds_partial_layer_fast(&state, poseidon::N_PARTIAL_ROUNDS - 1);
        round_ctr += poseidon::N_PARTIAL_ROUNDS;
        for _ in 0..poseidon::HALF_N_FULL_ROUNDS {
            <G as Poseidon>::constant_layer_field(&mut state, round_ctr);
            for value in state {
                emit(self, value)?;
            }
            <G as Poseidon>::sbox_layer(&mut state);
            state = <G as Poseidon>::mds_layer(&state);
            round_ctr += 1;
        }
        for value in state {
            emit(self, value)?;
        }
        ensure!(written == 122, "wrong Poseidon output width");
        let end = base as usize + 122;
        let destination = self
            .state
            .get(base as usize..end)
            .ok_or_else(|| anyhow::anyhow!("Poseidon block outside state"))?;
        for (cell, value) in destination.iter().zip(values) {
            let value = value.to_canonical_u64();
            cell.store(value, Ordering::Relaxed);
        }
        Ok(())
    }
}

/// Random values per independently seeded work chunk.
const SAMPLE_CHUNK: usize = 4096;

/// Samples canonical random field values with the existing chunked,
/// OS-seeded CPU sampler used by ordinary witness generation.
fn sample_random_values<F: PrimeField64>(count: usize) -> zeroize::Zeroizing<Vec<u64>> {
    let mut words = zeroize::Zeroizing::new(vec![0u64; count]);
    words.par_chunks_mut(SAMPLE_CHUNK).for_each(|chunk| {
        crate::util::blinding_rng::fill_field_words::<F>(chunk);
    });
    words
}

/// Timings and counts from one `generate_wires` or `check_against_worklist` call.
#[derive(Clone, Debug, Default)]
pub struct CpuExecutorReport {
    pub generators: usize,
    pub representatives: usize,
    pub random_values: usize,
    pub layers: usize,
    pub compile_cached: bool,
    pub initial_witness: Duration,
    pub sample: Duration,
    pub compile: Duration,
    pub execute: Duration,
    pub merge: Duration,
    pub gather: Duration,
    /// "installed" for a program the caller installed; diagnostics that
    /// compile their own program report "cached".
    pub plan: &'static str,
    /// How the wires were filled: "reused" (the pooled upload buffer) for
    /// device wires, an empty string for a host matrix, "none" for the
    /// partition-witness test helper.
    pub gather_mode: &'static str,
    /// "resident" when the wires were uploaded through a mapped staging buffer,
    /// "host" for a host matrix, "partition" for the test helper that merges
    /// values into a partition witness.
    pub mode: &'static str,
}

impl CpuExecutorReport {
    /// Logs the report at debug level under `plonky2::witness_program::executor`.
    pub fn log(&self) {
        log::debug!(
            target: "plonky2::witness_program::executor",
            "mode={} plan={} gather_mode={} generators={} representatives={} random={} layers={} initial={:?} compile={:?} cached={} sample={:?} execute={:?} merge={:?} gather={:?}",
            self.mode,
            self.plan,
            self.gather_mode,
            self.generators,
            self.representatives,
            self.random_values,
            self.layers,
            self.initial_witness,
            self.compile,
            self.compile_cached,
            self.sample,
            self.execute,
            self.merge,
            self.gather
        );
    }
}

#[cfg(test)]
thread_local! {
    /// Executor runs on this thread, so tests can tell which path a proof took.
    pub(crate) static EXECUTOR_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Public inputs, wires and timings of one executor run.
pub(crate) type ExecutorOutput<F> = (Vec<F>, Wires<F>, CpuExecutorReport);

/// Witness generation with the installed program: execute over the
/// compact slot state, read the public inputs from their slots and gather the
/// column-major wire matrix into a mapped staging buffer for upload when the
/// GPU commitment applies and `allow_device` is set, else a host matrix.
/// No partition witness is built. `None` when no program is
/// installed for these inputs; nothing is compiled, loaded or saved here.
pub(crate) fn generate_wires<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    inputs: &PartialWitness<F>,
    prover: &ProverOnlyCircuitData<F, C, D>,
    common: &CommonCircuitData<F, D>,
    allow_device: bool,
) -> Result<Option<ExecutorOutput<F>>> {
    generate_wires_with(inputs, prover, common, allow_device, None)
}

/// Bind and validate an input mask in one pass through the supplied targets.
/// The canonical slot map is already cached for final wire gathering. Roots
/// occupy its prefix, so membership and position need no additional index.
fn bind_roots_direct<F: PrimeField64>(
    inputs: &PartialWitness<F>,
    program: &RegionProgram,
    map: &[usize],
    num_wires: usize,
    degree: usize,
) -> Result<Option<Vec<u64>>> {
    let slots = program.rep_slots(map.len());
    let mut values = vec![0; program.roots.len()];
    let mut seen = vec![false; values.len()];
    for (&target, &value) in &inputs.target_values {
        let representative = target
            .checked_index(num_wires, degree)
            .and_then(|index| map.get(index))
            .copied()
            .ok_or_else(|| anyhow::anyhow!("input target {target:?} is outside the circuit"))?;
        let Some(position) = slots
            .get(representative)
            .copied()
            .filter(|&s| (s as usize) < values.len())
            .map(|s| s as usize)
        else {
            return Ok(None);
        };
        let value = value.to_canonical_u64();
        if seen[position] {
            ensure!(
                values[position] == value,
                "input {target:?} conflicts with another input of its partition"
            );
        } else {
            values[position] = value;
            seen[position] = true;
        }
    }
    Ok(seen.iter().all(|&x| x).then_some(values))
}

fn report_input_mismatch<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    inputs: &PartialWitness<F>,
    prover: &ProverOnlyCircuitData<F, C, D>,
    common: &CommonCircuitData<F, D>,
) -> Result<()> {
    let mut initial = inputs
        .target_values
        .keys()
        .map(|t| {
            t.checked_index(common.config.num_wires, common.degree())
                .and_then(|index| prover.representative_map.get(index))
                .copied()
                .ok_or_else(|| anyhow::anyhow!("input outside circuit"))
        })
        .collect::<Result<Vec<_>>>()?;
    initial.sort_unstable();
    initial.dedup();
    let _ = prover
        .witness_program
        .lookup_all(prover.generators.len(), &initial);
    ensure!(
        !prover.witness_program.strict_inputs(),
        "proof inputs do not match the prepared witness program"
    );
    Ok(())
}

pub(crate) fn generate_wires_with<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    inputs: &PartialWitness<F>,
    prover: &ProverOnlyCircuitData<F, C, D>,
    common: &CommonCircuitData<F, D>,
    allow_device: bool,
    random_values: Option<Vec<u64>>,
) -> Result<Option<ExecutorOutput<F>>> {
    // Without an installed program, one lock checks the slot and returns
    // before any executor preparation.
    if !prover.witness_program.has_all(prover.generators.len()) {
        return Ok(None);
    }
    ensure!(Op::field_supported::<F>(), "unsupported execution field");
    ensure!(
        common.luts.is_empty() && common.num_lookup_polys == 0,
        "compiled executor requires CPU lookup finalization to be absent"
    );
    let config = &common.config;
    let num_wires = config.num_wires;
    let degree = common.degree();
    let map = &prover.representative_map;
    let rep = |target: Target| -> Result<usize> {
        target
            .checked_index(num_wires, degree)
            .and_then(|index| map.get(index))
            .copied()
            .ok_or_else(|| anyhow::anyhow!("input target {target:?} is outside the circuit"))
    };
    let mut report = CpuExecutorReport {
        generators: prover.generators.len(),
        ..Default::default()
    };

    let mut direct = true;
    let started = Instant::now();
    let input_discovery = profile::Pipeline::new("input-discovery", prover.generators.len());
    let lookup = || -> Result<Option<std::sync::Arc<RegionProgram>>> {
        let mut initial = inputs
            .target_values
            .keys()
            .map(|&t| rep(t))
            .collect::<Result<Vec<_>>>()?;
        initial.sort_unstable();
        initial.dedup();
        Ok(
            match prover
                .witness_program
                .lookup_all(prover.generators.len(), &initial)
            {
                super::Installed::Program(p) => Some(p),
                super::Installed::OtherInputs => {
                    ensure!(
                        !prover.witness_program.strict_inputs(),
                        "proof inputs do not match the prepared witness program"
                    );
                    None
                }
                super::Installed::None => None,
            },
        )
    };
    // Bind directly when the installed program's inputs are exactly its
    // boundary roots; otherwise look the program up by the inputs' targets.
    let program = match prover
        .witness_program
        .candidate_all(prover.generators.len())
    {
        Some(program) => program,
        None => {
            direct = false;
            match lookup()? {
                Some(program) => program,
                None => return Ok(None),
            }
        }
    };
    report.compile_cached = true;
    report.plan = "installed";
    report.compile = started.elapsed();
    drop(input_discovery);
    report.representatives = program.representatives.len();
    report.random_values = program.random_count as usize;
    report.layers = program.layer_count();
    let started = Instant::now();
    let root_binding = profile::Pipeline::new("root-binding", prover.generators.len());
    let root_values = if direct {
        match bind_roots_direct(inputs, &program, map, num_wires, degree)? {
            Some(values) => values,
            None => {
                report_input_mismatch(inputs, prover, common)?;
                return Ok(None);
            }
        }
    } else {
        let mut values = vec![0u64; program.roots.len()];
        let mut seen = vec![false; program.roots.len()];
        for (&target, &value) in &inputs.target_values {
            let representative = rep(target)?;
            let position = program
                .roots
                .binary_search(&representative)
                .map_err(|_| anyhow::anyhow!("input {target:?} is not a boundary root"))?;
            let value = value.to_canonical_u64();
            if seen[position] {
                ensure!(
                    values[position] == value,
                    "input {target:?} conflicts with another input of its partition"
                );
            } else {
                values[position] = value;
                seen[position] = true;
            }
        }
        ensure!(
            seen.iter().all(|&set| set),
            "a boundary root has no input value"
        );
        values
    };
    drop(root_binding);
    // A resident destination, held through execution and upload: routing may
    // change concurrently, and deferred values must be produced on this device.
    #[cfg(feature = "gpu")]
    let ctx = if allow_device && crate::gpu::permutation::supported_circuit(prover, common) {
        crate::gpu::lde::applicable::<F, C, D>(
            num_wires,
            degree,
            config.fri_config.rate_bits,
            config.zero_knowledge && crate::plonk::plonk_common::PlonkOracle::WIRES.blinding,
        )
    } else {
        None
    };
    #[cfg(feature = "gpu")]
    let on_device = ctx.as_ref().is_some_and(|ctx| {
        #[cfg(test)]
        if let Some(forced) = program.runtime.config.policy.device_witness {
            return forced;
        }
        ctx.options.witness_on_device
    });
    // Deferring operations needs the parallel executor's certified plan; the
    // serial and checked executors upload their complete compact state.
    #[cfg(feature = "gpu")]
    let tail = if on_device
        && cfg!(feature = "parallel")
        && program.runtime.config.policy.execution
            != super::super::policy::WitnessExecution::Checked
    {
        let public_slots = prover
            .public_inputs
            .iter()
            .map(|&t| rep(t).map(|r| program.rep_slots(map.len())[r]))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            public_slots.iter().all(|&s| s != u32::MAX),
            "missing public input in GPU witness plan"
        );
        let fast = program.fast.get().expect("prepared tail certificate");
        let tail = fast
            .gpu_tail
            .get_or_init(|| {
                std::sync::Arc::new(super::gpu_tail::Tail::compile(&program, &public_slots))
            })
            .as_ref();
        ensure!(
            tail.matches_public_inputs(&public_slots),
            "GPU witness plan public inputs changed"
        );
        Some(tail)
    } else {
        None
    };
    // Deferred random values come from a fresh device seed unless the caller
    // supplied the values.
    #[cfg(feature = "gpu")]
    let device_seed = if tail.is_some() && random_values.is_none() {
        Some(crate::gpu::random::fresh_seed())
    } else {
        None
    };
    let random_sampling = profile::Pipeline::new("random-sampling", prover.generators.len());
    let random = match random_values {
        Some(values) => zeroize::Zeroizing::new(values),
        None => program.runtime.run(|| {
            #[cfg(test)]
            program.runtime.assert_active_pool();
            #[cfg(feature = "gpu")]
            if device_seed.is_some() {
                let mask = &tail.expect("device random tail").gpu_random;
                let mut words = zeroize::Zeroizing::new(vec![0u64; program.random_count as usize]);
                words
                    .par_chunks_mut(SAMPLE_CHUNK)
                    .enumerate()
                    .for_each(|(block, chunk)| {
                        let begin = block * SAMPLE_CHUNK;
                        if !mask[begin..begin + chunk.len()].iter().all(|&x| x) {
                            crate::util::blinding_rng::fill_field_words::<F>(chunk);
                        }
                    });
                return words;
            }
            sample_random_values::<F>(program.random_count as usize)
        }),
    };
    report.sample = started.elapsed();
    drop(random_sampling);
    #[cfg(feature = "witness-benchmarks")]
    if let Some(directory) = std::env::var_os("P2_AOT_SNAPSHOT_DIR") {
        static SAVED: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> =
            std::sync::OnceLock::new();
        use crate::plonk::config::GenericHashOut;
        let stem = super::aot::file_stem(&prover.circuit_digest.to_bytes());
        if SAVED
            .get_or_init(Default::default)
            .lock()
            .unwrap()
            .insert(stem.clone())
        {
            let directory = std::path::Path::new(&directory);
            std::fs::create_dir_all(directory)?;
            let bytes = crate::plonk::witness_program::installed(prover)
                .ok_or_else(|| anyhow::anyhow!("missing installed snapshot program"))?
                .to_bytes();
            std::fs::write(directory.join(format!("{stem}.p2w")), bytes)?;
            std::fs::write(
                directory.join(format!("{stem}.roots")),
                root_values
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            std::fs::write(
                directory.join(format!("{stem}.random")),
                random
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            eprintln!(
                "AOT_SNAPSHOT stem={stem} generators={} layers={}",
                prover.generators.len(),
                program.layer_count()
            );
        }
    }
    let started = Instant::now();
    #[cfg(feature = "gpu")]
    let executor = program
        .runtime
        .run(|| program.execute_state_inner_with_tail(&root_values, &random, false, tail))?;
    #[cfg(not(feature = "gpu"))]
    let executor = program.execute_state(&root_values, &random)?;
    report.execute = started.elapsed();

    let started = Instant::now();
    let rep_slots = program.rep_slots(map.len());
    let slot_value = |slot: u32| -> u64 {
        match slot {
            u32::MAX => 0,
            slot => executor.state[slot as usize].load(Ordering::Relaxed),
        }
    };
    let value_of = |representative: usize| -> u64 { slot_value(rep_slots[representative]) };
    let public_inputs = prover
        .public_inputs
        .iter()
        .map(|&target| {
            let representative = rep(target)?;
            ensure!(
                rep_slots[representative] != u32::MAX,
                "public input {target:?} is not produced by the program"
            );
            Ok(F::from_canonical_u64(value_of(representative)))
        })
        .collect::<Result<Vec<_>>>()?;
    // A device destination exists only with the `gpu` feature.
    #[cfg(not(feature = "gpu"))]
    let ctx: Option<()> = {
        let _ = (allow_device, config);
        None
    };
    let wires = match ctx {
        #[cfg(feature = "gpu")]
        Some(ctx) => {
            let cell_slots = program.cell_slots(map, num_wires, degree);
            let mut wire_slots = None;
            let mut error = None;
            let wire_values = if on_device {
                report.gather_mode = "compact";
                let (state, slots) = super::aot::compact_on_gpu(&executor, &ctx, cell_slots)?;
                wire_slots = Some(slots);
                if let Some(tail) = tail {
                    error = Some(tail.submit(&executor, &ctx, &state, device_seed.as_deref())?);
                    report.gather_mode = "tail";
                }
                state
            } else {
                let bytes = num_wires * degree * 8;
                // A reused mapped upload buffer, one mapped range per column
                // filled in parallel, then copied into a pooled device buffer.
                report.gather_mode = "reused";
                let staging = ctx.upload_staging(bytes as u64);
                let (columns, rows) = gather_tiling(&program.runtime.config);
                let columns = columns.clamp(1, num_wires);
                let rows = rows.clamp(1, degree);
                let fill = |columns: usize, rows: usize| {
                    cell_slots
                        .par_chunks(degree * columns)
                        .enumerate()
                        .for_each(|(block, slots)| {
                            let start = (block * columns * degree * 8) as u64;
                            let mut view = staging
                                .buffer
                                .slice(start..start + (slots.len() * 8) as u64)
                                .get_mapped_range_mut()
                                .expect("mapped upload buffer");
                            for row in (0..degree).step_by(rows) {
                                let end = (row + rows).min(degree);
                                for (column, slots) in slots.chunks(degree).enumerate() {
                                    let begin = (column * degree + row) * 8;
                                    let finish = (column * degree + end) * 8;
                                    let (words, _) = view.slice(begin..finish).into_chunks::<8>();
                                    for (cell, &slot) in words.into_iter().zip(&slots[row..end]) {
                                        cell.write(slot_value(slot).to_le_bytes());
                                    }
                                }
                            }
                        });
                };
                program.runtime.run(|| {
                    #[cfg(test)]
                    program.runtime.assert_active_pool();
                    fill(columns, rows);
                });
                let wire_values = ctx.storage("resident proving wires", bytes as u64);
                ctx.finish_upload(staging, &wire_values, bytes as u64);
                wire_values
            };
            report.mode = "resident";
            Wires::Resident(crate::gpu::resident_witness::ResidentWitness {
                ctx,
                wire_values,
                wire_slots,
                error,
            })
        }
        _ => {
            let cell_slots = program.cell_slots(map, num_wires, degree);
            let columns = program.runtime.run(|| {
                #[cfg(test)]
                program.runtime.assert_active_pool();
                cell_slots
                    .par_chunks(degree)
                    .map(|slots| {
                        slots
                            .iter()
                            .map(|&slot| F::from_canonical_u64(slot_value(slot)))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            });
            report.mode = "host";
            Wires::Host(MatrixWitness {
                wire_values: columns,
            })
        }
    };
    report.gather = started.elapsed();
    #[cfg(test)]
    EXECUTOR_RUNS.with(|runs| runs.set(runs.get() + 1));
    Ok(Some((public_inputs, wires, report)))
}

/// Test helper: compile, execute and merge the representatives into a
/// partition witness.
#[cfg(test)]
pub(crate) fn generate_with_report<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    inputs: &PartialWitness<F>,
    prover: &'a ProverOnlyCircuitData<F, C, D>,
    common: &'a CommonCircuitData<F, D>,
) -> Result<(PartitionWitness<'a, F>, CpuExecutorReport)> {
    ensure!(
        common.luts.is_empty() && common.num_lookup_polys == 0,
        "compiled executor requires CPU lookup finalization to be absent"
    );
    let started = Instant::now();
    let mut entry = crate::iop::generator::initial_witness(inputs, prover, common)?;
    let initial_witness = started.elapsed();
    let started = Instant::now();
    let (program, cached) =
        prover
            .witness_program
            .get_or_compile_all(&prover.generators, &entry, inputs)?;
    let compile = started.elapsed();
    let started = Instant::now();
    let random = sample_random_values::<F>(program.random_count as usize);
    let sample = started.elapsed();
    let started = Instant::now();
    let values = program.execute_cpu(&entry, &random)?;
    let execute = started.elapsed();
    let started = Instant::now();
    merge(&mut entry, &program, values)?;
    Ok((
        entry,
        CpuExecutorReport {
            generators: prover.generators.len(),
            representatives: program.representatives.len(),
            random_values: program.random_count as usize,
            layers: program.layer_count(),
            compile_cached: cached,
            initial_witness,
            sample,
            compile,
            execute,
            merge: started.elapsed(),
            gather: Duration::ZERO,
            mode: "partition",
            plan: "cached",
            gather_mode: "none",
        },
    ))
}

#[cfg(test)]
fn merge<F: Field>(
    entry: &mut PartitionWitness<F>,
    program: &RegionProgram,
    values: Vec<F>,
) -> Result<()> {
    ensure!(
        values.len() == program.representatives.len(),
        "executor returned the wrong number of representatives"
    );
    for (&representative, value) in program.representatives.iter().zip(values) {
        if let Some(initial) = entry.values[representative] {
            ensure!(initial == value, "executor conflicts with an initial value");
        } else {
            entry.values[representative] = Some(value);
        }
    }
    Ok(())
}

/// Correctness gate: run the worklist witness, feed its random values to the
/// compiled executor and require every representative to agree exactly. A
/// diagnostic: it compiles a program for `inputs` and leaves it installed.
pub fn check_against_worklist<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    inputs: &PartialWitness<F>,
    prover: &ProverOnlyCircuitData<F, C, D>,
    common: &CommonCircuitData<F, D>,
) -> Result<CpuExecutorReport> {
    let worklist = crate::iop::generator::generate_partial_witness(inputs.clone(), prover, common)?;
    let entry = crate::iop::generator::initial_witness(inputs, prover, common)?;
    let started = Instant::now();
    let (program, cached) =
        prover
            .witness_program
            .get_or_compile_all(&prover.generators, &entry, inputs)?;
    let compile = started.elapsed();
    let random = program
        .random_representatives()
        .iter()
        .enumerate()
        .map(|(index, representative)| {
            let representative = representative
                .ok_or_else(|| anyhow::anyhow!("random value {index} is written by an alias"))?;
            worklist.values[representative]
                .map(|v| v.to_canonical_u64())
                .ok_or_else(|| anyhow::anyhow!("worklist left random value {index} unset"))
        })
        .collect::<Result<Vec<_>>>()?;
    let started = Instant::now();
    let values = program.execute_cpu(&entry, &random)?;
    let execute = started.elapsed();
    let mut mismatches = 0usize;
    let mut first = None;
    for (&representative, value) in program.representatives.iter().zip(&values) {
        if worklist.values[representative] != Some(*value) {
            mismatches += 1;
            first.get_or_insert((representative, worklist.values[representative], *value));
        }
    }
    ensure!(
        mismatches == 0,
        "{mismatches} of {} representatives differ from the worklist witness; first: {first:?}",
        values.len()
    );
    let set = worklist.values.iter().filter(|v| v.is_some()).count();
    ensure!(
        set == program.representatives.len(),
        "worklist sets {set} representatives, the program covers {}",
        program.representatives.len()
    );
    // The delivered wire matrix and public inputs must equal the
    // worklist's full witness exactly, read back through the device buffer
    // when the GPU commitment applies.
    let expected_public = worklist.get_targets(&prover.public_inputs);
    let expected = worklist.full_witness();
    let (public_inputs, wires, wires_report) =
        generate_wires_with(inputs, prover, common, true, Some(random.clone()))?
            .ok_or_else(|| anyhow::anyhow!("the compiled program was not installed"))?;
    ensure!(
        public_inputs == expected_public,
        "public inputs differ from the worklist witness"
    );
    let num_wires = common.config.num_wires;
    let degree = common.degree();
    let actual = match wires {
        Wires::Host(witness) => witness.wire_values,
        #[cfg(feature = "gpu")]
        Wires::Resident(resident) => crate::gpu::lde::read_columns::<F>(
            &resident.ctx,
            &resident.wire_values,
            num_wires,
            degree,
        ),
    };
    ensure!(actual.len() == num_wires, "wire matrix has the wrong width");
    let mut wire_mismatches = 0usize;
    for (column, (actual, expected)) in actual.iter().zip(&expected.wire_values).enumerate() {
        ensure!(
            actual.len() == degree,
            "column {column} has the wrong height"
        );
        wire_mismatches += actual.iter().zip(expected).filter(|(a, e)| a != e).count();
    }
    ensure!(
        wire_mismatches == 0,
        "{wire_mismatches} wire cells differ from the worklist full witness ({} mode)",
        wires_report.mode
    );
    Ok(CpuExecutorReport {
        generators: prover.generators.len(),
        representatives: program.representatives.len(),
        random_values: program.random_count as usize,
        layers: program.layer_count(),
        compile_cached: cached,
        initial_witness: Duration::ZERO,
        sample: Duration::ZERO,
        compile,
        execute,
        merge: Duration::ZERO,
        gather: wires_report.gather,
        mode: wires_report.mode,
        plan: wires_report.plan,
        gather_mode: wires_report.gather_mode,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::field::extension::quadratic::QuadraticExtension;
    use crate::field::types::Sample;
    use crate::gates::coset_interpolation::CosetInterpolationGate;
    use crate::hash::poseidon::PoseidonHash;
    use crate::iop::ext_target::ExtensionTarget;
    use crate::iop::target::Target;
    use crate::iop::witness::{Witness, WitnessWrite};
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::{CircuitConfig, CircuitData};
    use crate::plonk::config::PoseidonGoldilocksConfig as C;
    use crate::util::reducing::ReducingFactorTarget;
    use crate::util::timing::TimingTree;

    #[test]
    fn chunk_end_handles_the_last_u32_range_without_wrapping() {
        for count in [1, 512, u32::MAX / 2, u32::MAX - 1, u32::MAX] {
            for chunk in [1, 2, 511, 1 << 31, u32::MAX] {
                let last = count.div_ceil(chunk) - 1;
                for c in [0, last] {
                    let start = c * chunk;
                    assert_eq!(
                        chunk_end(start, chunk, count) as u64,
                        (start as u64 + chunk as u64).min(count as u64)
                    );
                }
            }
        }
    }

    #[test]
    fn direct_binding_validates_aliases_missing_and_additional_roots() {
        let mut builder = CircuitBuilder::<G, 2>::new(CircuitConfig::standard_recursion_config());
        let a = builder.add_virtual_target();
        let alias = builder.add_virtual_target();
        builder.connect(a, alias);
        let b = builder.add_virtual_target();
        let unconstrained = builder.add_virtual_target();
        let sum = builder.add(a, b);
        builder.register_public_input(sum);
        let data = builder.build::<C>();
        let mut inputs = PartialWitness::new();
        inputs.set_target(a, G::from_canonical_u64(9)).unwrap();
        inputs.set_target(b, G::from_canonical_u64(5)).unwrap();
        inputs
            .set_target(unconstrained, G::from_canonical_u64(17))
            .unwrap();
        let program = data.compile_witness_program_like(&inputs).unwrap();
        let bind = |inputs: &PartialWitness<G>| {
            bind_roots_direct(
                inputs,
                &program.program,
                &data.prover_only.representative_map,
                data.common.config.num_wires,
                data.common.degree(),
            )
        };
        let expected = bind(&inputs).unwrap().unwrap();
        assert_eq!(expected.len(), 3);
        // Agreeing aliases may use distinct noncanonical field encodings.
        inputs.set_target(alias, G(G::ORDER + 9)).unwrap();
        assert_eq!(bind(&inputs).unwrap(), Some(expected));
        let mut conflict = inputs.clone();
        conflict
            .target_values
            .insert(alias, G::from_canonical_u64(10));
        assert!(bind(&conflict)
            .unwrap_err()
            .to_string()
            .contains("conflicts"));
        let mut missing = inputs.clone();
        missing.target_values.remove(&unconstrained);
        assert!(bind(&missing).unwrap().is_none());
        let mut extra = inputs.clone();
        extra.set_target(sum, G::from_canonical_u64(14)).unwrap();
        assert!(bind(&extra).unwrap().is_none());
    }

    pub(crate) struct Fixture {
        pub(crate) data: CircuitData<G, C, 2>,
        a: Target,
        b: Target,
        pub(crate) index: Target,
        alpha: ExtensionTarget<2>,
        shift: Target,
        point: ExtensionTarget<2>,
        values: Vec<ExtensionTarget<2>>,
    }

    /// A small circuit touching the standard typed kinds: arithmetic,
    /// constants, copies, equality, extension arithmetic, multiply and
    /// quotient, random access, base split, low/high split, reducing with
    /// base and extension coefficients, coset interpolation and Poseidon.
    pub(crate) fn circuit(blinded: bool) -> Fixture {
        let mut config = CircuitConfig::standard_recursion_config();
        config.zero_knowledge = blinded;
        let mut builder = CircuitBuilder::<G, 2>::new(config);
        let a = builder.add_virtual_target();
        let b = builder.add_virtual_target();
        let index = builder.add_virtual_target();
        let c = builder.mul(a, b);
        let d = builder.add(c, a);
        let e = builder.mul_many([d, d, d, a]);
        let equal = builder.is_equal(a, b);
        let bits = builder.split_le(d, 64);
        let (low, high) = builder.split_low_high(a, 16, 32);
        let selected = builder.random_access(index, vec![a, b, c, d]);
        let hash = builder.hash_n_to_hash_no_pad::<PoseidonHash>(vec![a, b, c, d, e]);
        let ae = builder.convert_to_ext(a);
        let be = builder.convert_to_ext(b);
        let product = builder.mul_extension(ae, be);
        let quotient = builder.div_extension(ae, be);
        let sum = builder.add_extension(product, quotient);
        let alpha = builder.add_virtual_extension_target();
        let mut factor = ReducingFactorTarget::new(alpha);
        let reduced_base = factor.reduce_base(&[a, b, c, d, e], &mut builder);
        let reduced_ext = factor.reduce(&[ae, be, product, sum], &mut builder);
        let shift = builder.add_virtual_target();
        let point = builder.add_virtual_extension_target();
        let values = builder.add_virtual_extension_targets(8);
        let interpolated =
            builder.interpolate_coset(CosetInterpolationGate::new(3), shift, &values, point);
        builder.register_public_inputs(&hash.elements);
        builder.register_public_inputs(&[selected, equal.target, low, high, bits[3].target]);
        builder.register_public_inputs(&reduced_base.0);
        builder.register_public_inputs(&reduced_ext.0);
        builder.register_public_inputs(&interpolated.0);
        Fixture {
            data: builder.build::<C>(),
            a,
            b,
            index,
            alpha,
            shift,
            point,
            values,
        }
    }

    pub(crate) fn inputs(fixture: &Fixture) -> PartialWitness<G> {
        let mut pw = PartialWitness::new();
        pw.set_target(fixture.a, G::from_canonical_u64(123456789))
            .unwrap();
        pw.set_target(fixture.b, G::from_canonical_u64(987654321))
            .unwrap();
        pw.set_target(fixture.index, G::from_canonical_u64(2))
            .unwrap();
        pw.set_extension_target(fixture.alpha, QuadraticExtension::<G>::rand())
            .unwrap();
        pw.set_target(fixture.shift, G::from_canonical_u64(5))
            .unwrap();
        pw.set_extension_target(fixture.point, QuadraticExtension::<G>::rand())
            .unwrap();
        for &value in &fixture.values {
            pw.set_extension_target(value, QuadraticExtension::<G>::rand())
                .unwrap();
        }
        pw
    }

    #[test]
    fn executor_matches_worklist_and_proves() {
        for blinded in [false, true] {
            let fixture = circuit(blinded);
            let pw = inputs(&fixture);
            let data = &fixture.data;
            let report = check_against_worklist(&pw, &data.prover_only, &data.common).unwrap();
            assert!(report.representatives > 0);
            if blinded {
                assert!(report.random_values > 0);
            }
            let (witness, _) = generate_with_report(&pw, &data.prover_only, &data.common).unwrap();
            let proof = crate::plonk::prover::prove_with_partition_witness(
                &data.prover_only,
                &data.common,
                witness,
                &mut TimingTree::default(),
            )
            .unwrap();
            data.verify(proof).unwrap();
        }
    }

    #[test]
    fn executor_wires_match_full_witness_and_prove() {
        for blinded in [false, true] {
            let fixture = circuit(blinded);
            let pw = inputs(&fixture);
            let data = &fixture.data;
            let worklist = crate::iop::generator::generate_partial_witness(
                pw.clone(),
                &data.prover_only,
                &data.common,
            )
            .unwrap();
            let entry =
                crate::iop::generator::initial_witness(&pw, &data.prover_only, &data.common)
                    .unwrap();
            let (program, _) = data
                .prover_only
                .witness_program
                .get_or_compile_all(&data.prover_only.generators, &entry, &pw)
                .unwrap();
            let random = program
                .random_representatives()
                .into_iter()
                .map(|r| worklist.values[r.unwrap()].unwrap().to_canonical_u64())
                .collect::<Vec<_>>();
            let expected_public = worklist.get_targets(&data.prover_only.public_inputs);
            let expected = worklist.full_witness();
            let (public_inputs, wires, report) =
                generate_wires_with(&pw, &data.prover_only, &data.common, false, Some(random))
                    .unwrap()
                    .expect("installed program");
            assert_eq!(report.mode, "host");
            assert_eq!(public_inputs, expected_public);
            #[allow(irrefutable_let_patterns)]
            let Wires::Host(witness) = wires
            else {
                panic!("host wires expected")
            };
            assert_eq!(witness.wire_values, expected.wire_values);
            let proof = crate::plonk::prover::prove_from_wires(
                &data.prover_only,
                &data.common,
                public_inputs,
                Wires::Host(witness),
                &mut TimingTree::default(),
            )
            .unwrap();
            data.verify(proof).unwrap();
        }
    }

    #[test]
    fn executor_rejects_a_plan_that_leaves_a_slot_unwritten() {
        let fixture = circuit(true);
        let pw = inputs(&fixture);
        let data = &fixture.data;
        let entry =
            crate::iop::generator::initial_witness(&pw, &data.prover_only, &data.common).unwrap();
        let (program, _) = data
            .prover_only
            .witness_program
            .get_or_compile_all(&data.prover_only.generators, &entry, &pw)
            .unwrap();
        let random = vec![1u64; program.random_count as usize];
        program.execute_cpu(&entry, &random).unwrap();
        // Redirect one random writer to a fresh slot: its real slot is never
        // written, which must be an error rather than a zero in the witness.
        let mut tampered = (*program).clone();
        let group = tampered
            .groups
            .iter()
            .flatten()
            .find(|g| g.kind == Kind::RandomValue)
            .cloned()
            .expect("blinded circuit has random writers");
        std::sync::Arc::make_mut(&mut tampered.targets)[group.targets_offset as usize] =
            tampered.state_slots;
        tampered.state_slots += 1;
        tampered.fast = Default::default();
        let error = tampered
            .execute_cpu(&entry, &random)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("never written")
                || error.contains("unwritten")
                || error.contains("earlier layer"),
            "{error}"
        );
        // A plan whose target offsets run past the table is an error, not a panic.
        let mut broken = (*program).clone();
        broken.groups[0][0].targets_offset = u32::MAX;
        broken.fast = Default::default();
        let error = broken.execute_cpu(&entry, &random).unwrap_err().to_string();
        assert!(
            error.contains("out of range") || error.contains("invalid target block"),
            "{error}"
        );
    }

    #[test]
    #[cfg(feature = "gpu")]
    #[ignore = "requires a native GPU; run alone with --ignored"]
    fn gpu_cpu_gather_tiles_match_every_worklist_wire() {
        use crate::hash::poseidon::PoseidonHash;
        use crate::plonk::witness_program::policy::WitnessExecution;
        use crate::plonk::witness_program::WitnessConfig;
        for hashes in [512, 1024] {
            let mut builder =
                CircuitBuilder::<G, 2>::new(CircuitConfig::standard_recursion_zk_config());
            let input = builder.add_virtual_target();
            // Private independent hash regions exercise device deferral while
            // the public chain preserves a CPU boundary through the graph.
            for i in 0..128 {
                let c = builder.constant(G::from_canonical_u64(i));
                let v = builder.add(input, c);
                let _ = builder.hash_n_to_hash_no_pad::<PoseidonHash>(vec![v; 4]);
            }
            let mut state = vec![input; 4];
            for _ in 0..hashes {
                state = builder
                    .hash_n_to_hash_no_pad::<PoseidonHash>(state)
                    .elements
                    .to_vec();
            }
            builder.register_public_inputs(&state);
            let data = builder.build::<C>();
            let mut pw = PartialWitness::new();
            pw.set_target(input, G::from_canonical_u64(13)).unwrap();
            let worklist = crate::iop::generator::generate_partial_witness(
                pw.clone(),
                &data.prover_only,
                &data.common,
            )
            .unwrap();
            let program = data
                .compile_witness_program_with_config(&[input], WitnessConfig::default())
                .unwrap();
            let random = program
                .program
                .random_representatives()
                .iter()
                .map(|r| {
                    worklist.values[r.expect("exclusive random writer")]
                        .unwrap()
                        .to_canonical_u64()
                })
                .collect::<Vec<_>>();
            let expected = worklist
                .full_witness()
                .wire_values
                .into_iter()
                .flatten()
                .map(|v| v.to_canonical_u64())
                .collect::<Vec<_>>();
            // (device witness, checked execution, CPU gather tile)
            for (device, checked, columns, rows) in [
                (false, false, 2, 1024),
                (false, false, 16, 256),
                (false, false, 7, 123),
                (true, false, 16, 256),
                (true, true, 16, 256),
            ] {
                let mut config = program.config().clone();
                config.policy.device_witness = Some(device);
                if checked {
                    config.policy.execution = WitnessExecution::Checked;
                }
                config.policy.gather_tiles = (columns, rows);
                if columns == 7 && cfg!(feature = "parallel") {
                    config.threads = 2;
                }
                let tail = device && !checked && cfg!(feature = "parallel");
                data.install_witness_program(program.with_execution_config(config).unwrap())
                    .unwrap();
                let (public_inputs, wires, report) = generate_wires_with(
                    &pw,
                    &data.prover_only,
                    &data.common,
                    true,
                    Some(random.clone()),
                )
                .unwrap()
                .unwrap();
                assert_eq!(
                    report.gather_mode,
                    if tail {
                        "tail"
                    } else if device {
                        "compact"
                    } else {
                        "reused"
                    }
                );
                let Wires::Resident(resident) = &wires else {
                    panic!("native resident wires required")
                };
                resident.validate().unwrap();
                let actual = resident
                    .read_columns::<G>(data.common.config.num_wires, data.common.degree())
                    .into_iter()
                    .flatten()
                    .map(|x| x.to_canonical_u64())
                    .collect::<Vec<_>>();
                assert_eq!(actual.len(), expected.len());
                for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(a, b, "hashes={hashes} tile={columns}x{rows} cell={i}");
                }
                let proof = crate::plonk::prover::prove_from_wires(
                    &data.prover_only,
                    &data.common,
                    public_inputs,
                    wires,
                    &mut TimingTree::default(),
                )
                .unwrap();
                data.verify(proof).unwrap();
                if tail {
                    let mut invalid = random.clone();
                    invalid[0] = G::ORDER;
                    match generate_wires_with(
                        &pw,
                        &data.prover_only,
                        &data.common,
                        true,
                        Some(invalid),
                    ) {
                        Err(_) => {}
                        Ok(Some((_, Wires::Resident(bad), _))) => assert!(bad.validate().is_err()),
                        _ => panic!("invalid random values must be rejected"),
                    }
                    let (_, valid, _) = generate_wires_with(
                        &pw,
                        &data.prover_only,
                        &data.common,
                        true,
                        Some(random.clone()),
                    )
                    .unwrap()
                    .unwrap();
                    let Wires::Resident(valid) = valid else {
                        panic!("resident tail")
                    };
                    valid.validate().unwrap();
                }
            }
        }
    }

    #[test]
    #[cfg(all(feature = "gpu", feature = "parallel"))]
    #[ignore = "requires a native GPU; run alone"]
    fn gpu_tail_random_proves_fresh_inputs() {
        use crate::hash::poseidon::PoseidonHash;
        use crate::plonk::witness_program::{WitnessConfig, WitnessFallback};
        let mut builder =
            CircuitBuilder::<G, 2>::new(CircuitConfig::standard_recursion_zk_config());
        let input = builder.add_virtual_target();
        for i in 0..128 {
            let c = builder.constant(G::from_canonical_u64(i));
            let v = builder.add(input, c);
            let mut state = vec![v; 4];
            for _ in 0..4 {
                state = builder
                    .hash_n_to_hash_no_pad::<PoseidonHash>(state)
                    .elements
                    .to_vec();
            }
        }
        builder.register_public_input(input);
        let data = builder.build::<C>();
        let mut config = WitnessConfig::default();
        config.policy.device_witness = Some(true);
        config.fallback = WitnessFallback::Error;
        data.prepare_witness_program_with_config(&[input], config)
            .unwrap();
        let mut previous = None;
        for value in [13, 13, G::ORDER - 1] {
            let mut pw = PartialWitness::new();
            pw.set_target(input, G::from_canonical_u64(value)).unwrap();
            let (public, wires, report) =
                generate_wires(&pw, &data.prover_only, &data.common, true)
                    .unwrap()
                    .unwrap();
            assert_eq!(report.gather_mode, "tail");
            assert_eq!(public, vec![G::from_canonical_u64(value)]);
            let Wires::Resident(resident) = &wires else {
                panic!("resident tail")
            };
            resident.validate().unwrap();
            let actual =
                resident.read_columns::<G>(data.common.config.num_wires, data.common.degree());
            if let Some(prior) = previous {
                assert_ne!(prior, actual, "fresh witness randomness");
            }
            previous = Some(actual);
            let proof = crate::plonk::prover::prove_from_wires(
                &data.prover_only,
                &data.common,
                public,
                wires,
                &mut TimingTree::default(),
            )
            .unwrap();
            data.verify(proof).unwrap();
        }
    }

    #[test]
    #[cfg(all(feature = "gpu", feature = "parallel"))]
    #[ignore = "requires a native GPU; run alone"]
    fn gpu_tail_preserves_shared_random_sources() {
        let mut builder =
            CircuitBuilder::<G, 2>::new(CircuitConfig::standard_recursion_zk_config());
        let input = builder.add_virtual_target();
        builder.register_public_input(input);
        let data = builder.build::<C>();
        let installed = data.compile_witness_program(&[input]).unwrap();
        let mut p = (*installed.program).clone();
        let g = p
            .groups
            .iter()
            .flatten()
            .find(|g| g.kind == Kind::RandomValue && g.count > 1)
            .unwrap()
            .clone();
        let co = g.coefficients_offset().unwrap() as usize;
        let source = p.coefficients[co] as usize;
        std::sync::Arc::make_mut(&mut p.coefficients)[co + 1] = source as u64;
        p.fast = Default::default();
        p.prepare(
            &data.prover_only.representative_map,
            data.common.config.num_wires,
            data.common.degree(),
        )
        .unwrap();
        let first = p.targets[g.targets_offset as usize];
        let second = p.targets[g.targets_offset as usize + 1];
        let tail = super::super::gpu_tail::Tail::compile(&p, &[first]);
        assert!(!tail.gpu_random[source]);
        let roots = vec![13; p.roots.len()];
        let random = vec![17; p.random_count as usize];
        let executor = p
            .runtime
            .run(|| p.execute_state_inner_with_tail(&roots, &random, false, Some(&tail)))
            .unwrap();
        let ctx = crate::gpu::context().expect("native GPU");
        let state = tail.upload_state(&executor, &ctx);
        let error = tail
            .submit(&executor, &ctx, &state, Some(&[42; 32]))
            .unwrap();
        assert!(ctx.read_u64(&error, 2).iter().all(|&x| x == 0));
        let actual = ctx.read_u64(&state, p.state_slots as usize);
        assert_eq!(actual[first as usize], 17);
        assert_eq!(actual[second as usize], 17);
    }

    #[test]
    #[cfg(all(feature = "gpu", feature = "parallel"))]
    #[ignore = "requires native GPU; run alone"]
    fn gpu_tail_checks_seeded_roots_and_aliases() {
        use crate::hash::poseidon::PoseidonHash;
        use crate::plonk::config::Hasher;
        use crate::plonk::witness_program::{WitnessConfig, WitnessFallback};
        let mut builder =
            CircuitBuilder::<G, 2>::new(CircuitConfig::standard_recursion_zk_config());
        let a = builder.add_virtual_target();
        let b = builder.add_virtual_target();
        let claim = builder.add_virtual_target();
        let ha = builder.hash_n_to_hash_no_pad::<PoseidonHash>(vec![a; 4]);
        let hb = builder.hash_n_to_hash_no_pad::<PoseidonHash>(vec![b; 4]);
        builder.connect(ha.elements[0], claim);
        builder.connect(hb.elements[0], claim);
        builder.register_public_input(claim);
        let data = builder.build::<C>();
        let input = G::from_canonical_u64(13);
        let expected = PoseidonHash::hash_no_pad(&[input; 4]).elements[0];
        let mut pw = PartialWitness::new();
        pw.set_target(a, input).unwrap();
        pw.set_target(b, input).unwrap();
        pw.set_target(claim, expected).unwrap();
        let worklist = crate::iop::generator::generate_partial_witness(
            pw.clone(),
            &data.prover_only,
            &data.common,
        )
        .unwrap();
        let mut config = WitnessConfig::default();
        config.policy.device_witness = Some(true);
        config.fallback = WitnessFallback::Error;
        let program = data
            .compile_witness_program_with_config(&[a, b, claim], config)
            .unwrap();
        let random = program
            .program
            .random_representatives()
            .iter()
            .map(|r| worklist.values[r.unwrap()].unwrap().to_canonical_u64())
            .collect::<Vec<_>>();
        let expected_wires = worklist.full_witness().wire_values;
        data.install_witness_program(program).unwrap();
        for (right, claimed, valid) in [
            (13, expected, true),
            (14, expected, false),
            (13, G::ZERO, false),
            (13, expected, true),
        ] {
            let mut pw = pw.clone();
            pw.target_values.insert(b, G::from_canonical_u64(right));
            pw.target_values.insert(claim, claimed);
            let (public, wires, report) = generate_wires_with(
                &pw,
                &data.prover_only,
                &data.common,
                true,
                Some(random.clone()),
            )
            .unwrap()
            .unwrap();
            assert_eq!(report.gather_mode, "tail");
            let Wires::Resident(resident) = &wires else {
                panic!("resident checked regions")
            };
            if valid {
                resident.validate().unwrap();
                assert_eq!(
                    resident.read_columns::<G>(data.common.config.num_wires, data.common.degree()),
                    expected_wires
                );
                let proof = crate::plonk::prover::prove_from_wires(
                    &data.prover_only,
                    &data.common,
                    public,
                    wires,
                    &mut TimingTree::default(),
                )
                .unwrap();
                data.verify(proof).unwrap();
            } else {
                assert!(
                    resident.validate().is_err(),
                    "conflicting root/alias must be rejected"
                );
            }
        }
    }

    #[test]
    fn sparse_flags_preserve_repeated_output_agreement_and_failure_cleanup() {
        let config = crate::plonk::witness_program::WitnessConfig {
            buffers: crate::plonk::witness_program::WitnessBuffers::Reuse { max_bytes: 4096 },
            ..Default::default()
        };
        let mut targets = vec![0, 1, 1];
        targets.extend(2..128);
        let program = RegionProgram {
            groups: vec![vec![
                super::super::Group {
                    kind: Kind::BaseSplit,
                    count: 1,
                    targets_offset: 0,
                    argument: Kind::BaseSplit.argument((2 << 8) | 1),
                    inputs: 1,
                    outputs: 2,
                },
                super::super::Group {
                    kind: Kind::Constant,
                    count: 126,
                    targets_offset: 3,
                    argument: Kind::Constant.argument(0),
                    inputs: 0,
                    outputs: 1,
                },
            ]],
            targets: targets.into(),
            coefficients: vec![0; 126].into(),
            representatives: (0..128).collect::<Vec<_>>().into(),
            roots: vec![0].into(),
            random_count: 0,
            alias_count: 0,
            alias_pair_offset: 0,
            state_slots: 128,
            runtime: std::sync::Arc::new(
                crate::plonk::witness_program::config::Runtime::new(config).unwrap(),
            ),
            fast: Default::default(),
            rep_slots: Default::default(),
            cell_slots: Default::default(),
        };
        for _ in 0..3 {
            for checked in [false, true] {
                assert!(program
                    .execute_state_inner(&[1], &[], checked)
                    .err()
                    .expect("conflicting split outputs")
                    .to_string()
                    .contains("conflicting write"));
                for root in [0, 3] {
                    let actual = program.execute_state_inner(&[root], &[], checked).unwrap();
                    assert_eq!(actual.state[1].load(Ordering::Relaxed), root & 1);
                    assert_eq!(actual.ready.len(), if checked { 128 } else { 1 });
                }
            }
        }
        assert_eq!(
            program.fast.get().unwrap().sparse_ready.as_deref(),
            Some(&[1][..])
        );
    }

    #[test]
    fn flagless_execution_checks_seeded_outputs_and_reuses_both_buffer_shapes() {
        let config = crate::plonk::witness_program::WitnessConfig {
            buffers: crate::plonk::witness_program::WitnessBuffers::Reuse { max_bytes: 64 },
            ..Default::default()
        };
        let program = RegionProgram {
            groups: vec![vec![super::super::Group {
                kind: Kind::Constant,
                count: 1,
                targets_offset: 0,
                argument: Kind::Constant.argument(0),
                inputs: 0,
                outputs: 1,
            }]],
            targets: vec![0].into(),
            coefficients: vec![5].into(),
            representatives: vec![0].into(),
            roots: vec![0].into(),
            random_count: 0,
            alias_count: 0,
            alias_pair_offset: 0,
            state_slots: 1,
            runtime: std::sync::Arc::new(
                crate::plonk::witness_program::config::Runtime::new(config).unwrap(),
            ),
            fast: Default::default(),
            rep_slots: Default::default(),
            cell_slots: Default::default(),
        };
        for _ in 0..3 {
            for checked in [false, true] {
                assert!(program
                    .execute_state_inner(&[7], &[], checked)
                    .err()
                    .expect("conflicting seeded output")
                    .to_string()
                    .contains("conflicting write"));
                let actual = program.execute_state_inner(&[5], &[], checked).unwrap();
                assert_eq!(actual.state[0].load(Ordering::Relaxed), 5);
                assert_eq!(actual.ready.len(), usize::from(checked));
            }
        }
    }

    #[test]
    fn contiguous_random_copies_match_checked_execution_with_extreme_words() {
        let mut builder = CircuitBuilder::<G, 2>::new(CircuitConfig::standard_recursion_config());
        for _ in 0..1024 {
            let target = builder.add_virtual_target();
            builder.add_simple_generator(crate::iop::generator::RandomValueGenerator { target });
        }
        let data = builder.build::<C>();
        let pw = PartialWitness::new();
        let entry =
            crate::iop::generator::initial_witness(&pw, &data.prover_only, &data.common).unwrap();
        let program = data.compile_witness_program_like(&pw).unwrap();
        let program = &*program.program;
        let roots = program
            .roots
            .iter()
            .map(|&r| entry.values[r].unwrap().to_canonical_u64())
            .collect::<Vec<_>>();
        let random = (0..program.random_count as usize)
            .map(|i| [0, 1, G::ORDER - 1, G::ORDER - 2][i % 4])
            .collect::<Vec<_>>();
        for irregular in [false, true] {
            let mut program = program.clone();
            if irregular {
                let group = program
                    .groups
                    .iter()
                    .flatten()
                    .find(|g| g.kind == Kind::RandomValue && g.count >= 2)
                    .unwrap();
                let co = group.coefficients_offset().unwrap() as usize;
                std::sync::Arc::make_mut(&mut program.coefficients).swap(co, co + 1);
            }
            program.fast = Default::default();
            let expected = program.execute_state_inner(&roots, &random, true).unwrap();
            let actual = program.execute_state(&roots, &random).unwrap();
            for (a, b) in actual.state.iter().zip(&expected.state) {
                assert_eq!(a.load(Ordering::Relaxed), b.load(Ordering::Relaxed));
            }
            for invalid in [G::ORDER, u64::MAX] {
                let mut bad = random.clone();
                bad[0] = invalid;
                assert!(program.execute_state_inner(&roots, &bad, true).is_err());
                assert!(program.execute_state(&roots, &bad).is_err());
            }
            let spans = program.fast.get().unwrap().random_spans.get().unwrap();
            if !irregular {
                assert!(!spans.is_empty());
            }
        }
    }

    #[test]
    fn plan_bytes_reject_malformed_programs() {
        use super::super::plan_bytes::{self, PlanKey};
        let fixture = circuit(true);
        let pw = inputs(&fixture);
        let data = &fixture.data;
        let entry =
            crate::iop::generator::initial_witness(&pw, &data.prover_only, &data.common).unwrap();
        let (program, _) = data
            .prover_only
            .witness_program
            .get_or_compile_all(&data.prover_only.generators, &entry, &pw)
            .unwrap();
        let key = PlanKey {
            digest: vec![1, 2, 3],
            generators: data.prover_only.generators.len(),
            map_len: data.prover_only.representative_map.len(),
            mask: vec![],
            format: 7,
            operations: 9 << 64 | 3,
            compilation: program.runtime.config.compilation_key(),
        };
        let bytes = plan_bytes::encode(&program, &key);
        let (decoded_key, decoded) = plan_bytes::decode(&bytes).unwrap();
        assert_eq!(decoded_key, key);
        assert_eq!(plan_bytes::encode(&decoded, &key), bytes);
        // Digest length of u64::MAX with a valid checksum: rejected, no panic.
        let mut huge = bytes.clone();
        huge[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(plan_bytes::decode(&plan_bytes::reseal(&huge)).is_err());
        // An unsorted or out-of-range input mask: rejected.
        for mask in [vec![2, 1], vec![key.map_len]] {
            let odd = PlanKey {
                mask,
                ..key.clone()
            };
            assert!(plan_bytes::decode(&plan_bytes::encode(&program, &odd)).is_err());
        }
        // Duplicate random index: rejected.
        let mut duplicate = (*program).clone();
        let group = duplicate
            .groups
            .iter()
            .flatten()
            .find(|g| g.kind == Kind::RandomValue && g.count >= 2)
            .cloned()
            .expect("a random group with two lanes");
        let first = duplicate.coefficients[group
            .coefficients_offset()
            .expect("random group coefficient storage")
            as usize];
        std::sync::Arc::make_mut(&mut duplicate.coefficients)[group
            .coefficients_offset()
            .expect("random group coefficient storage")
            as usize
            + 1] = first;
        let error = plan_bytes::decode(&plan_bytes::encode(&duplicate, &key))
            .unwrap_err()
            .to_string();
        assert!(error.contains("random"), "{error}");
        // Roots must occupy the prefix of canonical state: reject a changed root.
        let mut roots = (*program).clone();
        std::sync::Arc::make_mut(&mut roots.roots)[0] = roots.representatives[0].wrapping_add(1);
        assert!(plan_bytes::decode(&plan_bytes::encode(&roots, &key)).is_err());
        // A flipped byte: rejected by the checksum.
        let mut damaged = bytes.clone();
        damaged[bytes.len() / 2] ^= 1;
        assert!(plan_bytes::decode(&damaged).is_err());
    }
}

/// Replays locally generated benchmark fixtures, excluding decode and input I/O.
/// Both executors use the identical slot topology, inputs, and randomness.
#[cfg(feature = "witness-benchmarks")]
pub(crate) fn replay_snapshot(stem: &std::path::Path, iterations: usize) -> Result<()> {
    let started = Instant::now();
    let (key, mut program) =
        super::plan_bytes::decode(&std::fs::read(stem.with_extension("p2w"))?)?;
    ensure!(
        key.format == super::plan_bytes::WITNESS_PROGRAM_FORMAT,
        "snapshot executor format changed"
    );
    // Replay opts into development controls explicitly; the decoder itself
    // stays deterministic and never reads process settings.
    program.runtime = std::sync::Arc::new(super::super::config::Runtime::new(
        super::super::WitnessConfig::from_env()?.with_compilation_key(key.compilation)?,
    )?);
    let read_words = |extension: &str| -> Result<Vec<u64>> {
        let bytes = std::fs::read(stem.with_extension(extension))?;
        ensure!(bytes.len() % 8 == 0, "invalid word file");
        Ok(bytes
            .chunks_exact(8)
            .map(|x| u64::from_le_bytes(x.try_into().unwrap()))
            .collect())
    };
    let mut buckets = [(0u32, 0u32); 8];
    for g in program
        .groups
        .iter()
        .flatten()
        .filter(|g| g.kind == Kind::Poseidon)
    {
        let i = if g.count < 8 {
            0
        } else if g.count < 16 {
            1
        } else if g.count < 32 {
            2
        } else if g.count < 64 {
            3
        } else if g.count < 128 {
            4
        } else if g.count < 256 {
            5
        } else if g.count < 512 {
            6
        } else {
            7
        };
        buckets[i].0 += 1;
        buckets[i].1 += g.count;
    }
    eprintln!("POSEIDON_GROUP_HIST buckets_lt_8_16_32_64_128_256_512_inf={buckets:?}");
    let roots = read_words("roots")?;
    let random = zeroize::Zeroizing::new(read_words("random")?);
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let reference = program.execute_state_inner(&roots, &random, true)?;
    eprintln!(
        "AOT_REPLAY generators={} layers={} slots={} load_ms={load_ms:.6}",
        key.generators,
        program.layer_count(),
        program.state_slots
    );
    for iteration in 0..iterations {
        let started = Instant::now();
        let actual = program.execute_state(&roots, &random)?;
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        for (a, b) in actual.state.iter().zip(&reference.state) {
            ensure!(
                a.load(Ordering::Relaxed) == b.load(Ordering::Relaxed),
                "snapshot execution mismatch"
            );
        }
        eprintln!(
            "AOT_EXECUTE generators={} layers={} slots={} iteration={} ms={ms:.6} equal=true",
            key.generators,
            program.layer_count(),
            program.state_slots,
            iteration
        );
    }
    Ok(())
}
