//! The compiled witness executor: runs a `RegionProgram` on the CPU.
//!
//! A program is typed operations with resolved slot indices, grouped by kind
//! inside dependency layers, plus alias slots for additional writers. The
//! executor keeps one atomic value and one ready flag per slot, seeds the
//! roots from the proof's inputs and runs the layers in order: the lanes of
//! every group in a layer can run in parallel with the `parallel` feature,
//! one Rust loop body per operation kind. Each layer finishes before the next.
//! Readiness and conflict guards check every read and write, and each alias
//! slot is compared with its canonical slot at the end to reject missing
//! dependencies and conflicting writes. Values follow the generators' typed
//! operation descriptions. The gather then writes the slots
//! straight into the column-major wire matrix, or into a mapped upload buffer
//! that is copied to device memory when the GPU prover applies. This avoids
//! building and merging a partition witness before gathering the wires.
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Result};
use plonky2_maybe_rayon::*;

use super::RegionProgram;
use crate::field::extension::quintic::QuinticExtension;
use crate::field::extension::Extendable;
use crate::field::goldilocks_field::GoldilocksField as G;
use crate::field::types::{Field, Field64, PrimeField64};
use crate::hash::hash_types::RichField;
use crate::hash::poseidon::{self, Poseidon, SPONGE_WIDTH};
use crate::iop::generator::WitnessOperation as Op;
use crate::iop::target::Target;
use crate::iop::witness::{MatrixWitness, PartialWitness, PartitionWitness, Witness};
use crate::plonk::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
use crate::plonk::config::GenericConfig;

type Q = QuinticExtension<G>;

impl RegionProgram {
    /// State slot written by each random value, indexed by the random index
    /// the program assigns. Slots below the representative count are
    /// canonical; higher slots belong to alias writers.
    pub fn random_slots(&self) -> Vec<u32> {
        let mut out = vec![u32::MAX; self.random_count as usize];
        for group in self.groups.iter().flatten() {
            if group.kind != 17 {
                continue;
            }
            for k in 0..group.count {
                let index = self.coefficients[(group.coefficients_offset + k) as usize] as usize;
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

    /// Runs the program over `root_values`, one canonical value per boundary
    /// root in `roots` order, and returns the final slot state.
    fn execute_state<'a>(
        &'a self,
        root_values: &[u64],
        random_values: &'a [u64],
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
        let executor = Executor {
            program: self,
            random: random_values,
            state: (0..self.state_slots).map(|_| AtomicU64::new(0)).collect(),
            ready: (0..self.state_slots)
                .map(|_| AtomicBool::new(false))
                .collect(),
        };
        for (&value, &slot) in root_values.iter().zip(&self.root_slots) {
            ensure!(value < G::ORDER, "boundary input {value} is not canonical");
            executor.store(slot, G::from_canonical_u64(value))?;
        }
        // Operations in one layer are independent: every unseeded slot has
        // one writer, additional writers use private alias slots, and
        // inputs were produced by earlier layers. The Rayon join at each
        // layer boundary orders the layers; relaxed atomics suffice inside.
        let run_group = |group: &super::Group| {
            let chunk = lane_chunk(group.kind);
            let chunks = group.count.div_ceil(chunk);
            (0..chunks).into_par_iter().try_for_each(|c| {
                for k in c * chunk..((c + 1) * chunk).min(group.count) {
                    executor.step(
                        group.kind,
                        k,
                        group.count,
                        group.targets_offset,
                        group.coefficients_offset,
                    )?;
                }
                Ok::<(), anyhow::Error>(())
            })
        };
        if cfg!(feature = "gpu-profiling")
            && crate::util::switches::flag("PLONKY2_GPU_WITNESS_CPU_EXECUTOR_PROFILE")
        {
            // Diagnostic: groups run one at a time so each kind's wall time
            // is attributable; lanes stay parallel.
            let mut by_kind = vec![(Duration::ZERO, 0u64, 0u64); 26];
            let started = Instant::now();
            for layer in &self.groups {
                for group in layer {
                    let group_started = Instant::now();
                    run_group(group)?;
                    let entry = &mut by_kind[group.kind];
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
        } else {
            for layer in &self.groups {
                layer.par_iter().try_for_each(run_group)?;
            }
        }
        for i in 0..self.alias_count {
            let canonical = self.targets[(self.alias_pair_offset + 2 * i) as usize];
            let duplicate = self.targets[(self.alias_pair_offset + 2 * i + 1) as usize];
            ensure!(
                executor.load(canonical)? == executor.load(duplicate)?,
                "alias slot {duplicate} disagrees with its canonical slot {canonical}"
            );
        }
        // Every slot, canonical or alias, must have been written: a plan
        // that omits a writer (for example a blinding value) is rejected
        // here rather than gathered as zero.
        ensure!(
            executor.ready.par_iter().all(|r| r.load(Ordering::Relaxed)),
            "a state slot was never written by the plan"
        );
        Ok(executor)
    }
}

struct Executor<'a> {
    program: &'a RegionProgram,
    random: &'a [u64],
    state: Vec<AtomicU64>,
    ready: Vec<AtomicBool>,
}

impl Drop for Executor<'_> {
    /// Witness values do not outlive the executor in host memory. The wipe
    /// goes through `zeroize`, whose volatile writes the compiler may not
    /// elide, rather than relaxed atomic stores.
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.state
            .par_iter_mut()
            .for_each(|v| v.get_mut().zeroize());
    }
}

/// Lanes per Rayon task. Expensive kinds split to single lanes; cheap
/// kinds are batched so task overhead stays below the arithmetic.
fn lane_chunk(kind: usize) -> u32 {
    match kind {
        0 | 15 | 16 | 22 | 23 | 24 => 1,
        13 | 20 | 21 | 25 => 16,
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
    fn load(&self, slot: u32) -> Result<G> {
        let slot = slot as usize;
        ensure!(slot < self.state.len(), "state slot {slot} out of bounds");
        ensure!(
            self.ready[slot].load(Ordering::Relaxed),
            "state slot {slot} read before it was written"
        );
        Ok(G::from_canonical_u64(
            self.state[slot].load(Ordering::Relaxed),
        ))
    }

    fn store(&self, slot: u32, value: G) -> Result<()> {
        let slot = slot as usize;
        ensure!(slot < self.state.len(), "state slot {slot} out of bounds");
        let value = value.to_canonical_u64();
        if self.ready[slot].load(Ordering::Relaxed) {
            ensure!(
                self.state[slot].load(Ordering::Relaxed) == value,
                "conflicting write to state slot {slot}"
            );
            return Ok(());
        }
        self.state[slot].store(value, Ordering::Relaxed);
        self.ready[slot].store(true, Ordering::Relaxed);
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

    fn input(&self, offset: u32, position: u32, count: u32, k: u32) -> Result<G> {
        self.load(self.target(offset, position, count, k)?)
    }

    fn output(&self, offset: u32, position: u32, count: u32, k: u32, value: G) -> Result<()> {
        let slot = self.target(offset, position, count, k)?;
        self.store(slot, value)
    }

    /// Coefficient of lane `k` at `position` in a group's coefficient block,
    /// with checked arithmetic and a bounds check.
    fn coefficient(&self, base: u32, position: u64, count: u32, k: u32) -> Result<u64> {
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

    fn step(&self, kind: usize, k: u32, count: u32, offset: u32, co: u32) -> Result<()> {
        match kind {
            0 => self.step_poseidon(k, count, offset),
            1 => {
                let x = self.input(offset, 0, count, k)?;
                let y = self.input(offset, 1, count, k)?;
                let z = self.input(offset, 2, count, k)?;
                let c0 = field(self.coefficient(co, 0, count, k)?);
                let c1 = field(self.coefficient(co, 1, count, k)?);
                self.output(offset, 3, count, k, x * y * c0 + z * c1)
            }
            2 => {
                let x = self.input(offset, 0, count, k)?;
                self.output(offset, 1, count, k, x)
            }
            3 => {
                let value = field(self.coefficient(co, 0, count, k)?);
                self.output(offset, 0, count, k, value)
            }
            4 => {
                let x = canonical(self.input(offset, 0, count, k)?);
                let y = canonical(self.input(offset, 1, count, k)?);
                let same = x == y;
                let flag = if same { G::ONE } else { G::ZERO };
                let inverse = if same { G::ZERO } else { inverse(x - y)? };
                self.output(offset, 2, count, k, flag)?;
                self.output(offset, 3, count, k, inverse)
            }
            5 => {
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
            6 => {
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
            7 => {
                // The coefficient word is the list size.
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
            8 => {
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
            9 => {
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
            10 => {
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
            11 | 12 => {
                let extension_coeffs = kind == 12;
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
            13 => {
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
            14 => {
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
            15 => {
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
            16 => {
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
            17 => {
                let index = self.coefficient(co, 0, count, k)?;
                let value = *self
                    .random
                    .get(usize::try_from(index).unwrap_or(usize::MAX))
                    .ok_or_else(|| anyhow::anyhow!("plan random index {index} out of range"))?;
                ensure!(value < G::ORDER, "random value {value} is not canonical");
                self.output(offset, 0, count, k, G::from_canonical_u64(value))
            }
            18 => {
                let denominator = canonical(self.input(offset, 0, count, k)?);
                let numerator = self.input(offset, 1, count, k)?;
                let quotient = if denominator != G::ZERO {
                    numerator * inverse(denominator)?
                } else {
                    G::ZERO
                };
                self.output(offset, 2, count, k, quotient)
            }
            19 => {
                let length = co;
                let n = self.input(offset, 0, count, k)?.to_canonical_u64();
                for j in 0..length {
                    let bit = if (j as u64) < n { G::ONE } else { G::ZERO };
                    self.output(offset, j + 1, count, k, bit)?;
                }
                Ok(())
            }
            20 => {
                let lhs = self.q_input(offset, 0, count, k)?;
                let rhs = self.q_input(offset, 5, count, k)?;
                self.q_output(offset, 10, count, k, lhs * rhs)
            }
            21 => {
                let numerator = self.q_input(offset, 0, count, k)?;
                let denominator = self.q_input(offset, 5, count, k)?;
                let quotient = numerator * q_inverse(denominator)?;
                self.q_output(offset, 10, count, k, quotient)
            }
            22 => {
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
            23 => {
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
            24 => {
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
            25 => {
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
            _ => bail!("unknown operation kind {kind}"),
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
}

/// Random values per independently seeded work chunk.
const SAMPLE_CHUNK: usize = 4096;

/// Samples canonical random field values with the existing chunked,
/// OS-seeded CPU sampler used by ordinary witness generation.
fn sample_random_values<F: PrimeField64>(count: usize) -> zeroize::Zeroizing<Vec<u64>> {
    let mut words = zeroize::Zeroizing::new(vec![0u64; count]);
    words.par_chunks_mut(SAMPLE_CHUNK).for_each(|chunk| {
        crate::iop::generator::with_randomness(|| {
            for word in chunk {
                *word = crate::iop::generator::random_value::<F>().to_canonical_u64();
            }
        });
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
    pub worklist: Duration,
    /// "installed" for a program the caller installed; diagnostics that
    /// compile their own program report "cached".
    pub plan: &'static str,
    /// Program I/O time; zero because these executor paths perform no I/O.
    pub plan_io: Duration,
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
            "mode={} plan={} plan_io={:?} gather_mode={} generators={} representatives={} random={} layers={} initial={:?} compile={:?} cached={} sample={:?} execute={:?} merge={:?} gather={:?}",
            self.mode,
            self.plan,
            self.plan_io,
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

/// Where the executor delivered the wire matrix.
pub(crate) enum Wires<F: Field> {
    Host(MatrixWitness<F>),
    #[cfg(feature = "gpu")]
    Resident(crate::gpu::resident_witness::ResidentWitness),
}

impl<F: Field> std::fmt::Debug for Wires<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Host(witness) => write!(f, "Wires::Host({} columns)", witness.wire_values.len()),
            #[cfg(feature = "gpu")]
            Self::Resident(_) => write!(f, "Wires::Resident"),
        }
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
        map.get(target.index(num_wires, degree))
            .copied()
            .ok_or_else(|| anyhow::anyhow!("input target {target:?} is outside the circuit"))
    };
    let mut report = CpuExecutorReport {
        generators: prover.generators.len(),
        ..Default::default()
    };

    let started = Instant::now();
    let mut initial = inputs
        .target_values
        .keys()
        .map(|&t| rep(t))
        .collect::<Result<Vec<_>>>()?;
    initial.sort_unstable();
    initial.dedup();
    let program = match prover
        .witness_program
        .lookup_all(prover.generators.len(), &initial)
    {
        super::Installed::Program(program) => program,
        super::Installed::OtherInputs | super::Installed::None => return Ok(None),
    };
    report.compile_cached = true;
    report.plan = "installed";
    report.compile = started.elapsed();
    report.representatives = program.representatives.len();
    report.random_values = program.random_count as usize;
    report.layers = program.layer_count();

    // Boundary roots straight from the inputs; duplicates must agree.
    let started = Instant::now();
    let mut root_values = vec![0u64; program.roots.len()];
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
                root_values[position] == value,
                "input {target:?} conflicts with another input of its partition"
            );
        } else {
            root_values[position] = value;
            seen[position] = true;
        }
    }
    ensure!(
        seen.iter().all(|&set| set),
        "a boundary root has no input value"
    );
    let random = match random_values {
        Some(values) => zeroize::Zeroizing::new(values),
        None => sample_random_values::<F>(program.random_count as usize),
    };
    report.sample = started.elapsed();
    let started = Instant::now();
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
    #[cfg(feature = "gpu")]
    let ctx = if allow_device {
        crate::gpu::lde::applicable::<F, C, D>(
            num_wires,
            degree,
            config.fri_config.rate_bits,
            config.zero_knowledge && crate::plonk::plonk_common::PlonkOracle::WIRES.blinding,
        )
    } else {
        None
    };
    #[cfg(not(feature = "gpu"))]
    let ctx: Option<()> = {
        let _ = (allow_device, config);
        None
    };
    let wires = match ctx {
        #[cfg(feature = "gpu")]
        Some(ctx) => {
            let cell_slots = program.cell_slots(map, num_wires, degree);
            let bytes = num_wires * degree * 8;
            // A reused mapped upload buffer, one mapped range per column
            // filled in parallel, then copied into a pooled device buffer.
            report.gather_mode = "reused";
            let staging = ctx.upload_staging(bytes as u64);
            cell_slots
                .par_chunks(degree)
                .enumerate()
                .for_each(|(column, slots)| {
                    let start = (column * degree * 8) as u64;
                    let mut view = staging
                        .buffer
                        .slice(start..start + (degree * 8) as u64)
                        .get_mapped_range_mut()
                        .expect("mapped upload buffer");
                    let (words, _) = view.slice(..).into_chunks::<8>();
                    for (cell, &slot) in words.into_iter().zip(slots) {
                        cell.write(slot_value(slot).to_le_bytes());
                    }
                });
            let wire_values = ctx.storage("resident proving wires", bytes as u64);
            ctx.finish_upload(staging, &wire_values, bytes as u64);
            report.mode = "resident";
            Wires::Resident(crate::gpu::resident_witness::ResidentWitness { ctx, wire_values })
        }
        _ => {
            let cell_slots = program.cell_slots(map, num_wires, degree);
            let columns = cell_slots
                .par_chunks(degree)
                .map(|slots| {
                    slots
                        .iter()
                        .map(|&slot| F::from_canonical_u64(slot_value(slot)))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
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
            worklist: Duration::ZERO,
            mode: "partition",
            plan: "cached",
            plan_io: Duration::ZERO,
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
    let started = Instant::now();
    let worklist = crate::iop::generator::generate_partial_witness(inputs.clone(), prover, common)?;
    let worklist_time = started.elapsed();
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
        worklist: worklist_time,
        mode: wires_report.mode,
        plan: wires_report.plan,
        plan_io: wires_report.plan_io,
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
                Some(witness),
                &mut TimingTree::default(),
                #[cfg(feature = "gpu")]
                None,
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
            .find(|g| g.kind == 17)
            .cloned()
            .expect("blinded circuit has random writers");
        tampered.targets[group.targets_offset as usize] = tampered.state_slots;
        tampered.state_slots += 1;
        let error = tampered
            .execute_cpu(&entry, &random)
            .unwrap_err()
            .to_string();
        assert!(error.contains("never written"), "{error}");
        // A plan whose target offsets run past the table is an error, not a panic.
        let mut broken = (*program).clone();
        broken.groups[0][0].targets_offset = u32::MAX;
        let error = broken.execute_cpu(&entry, &random).unwrap_err().to_string();
        assert!(error.contains("out of range"), "{error}");
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
            .find(|g| g.kind == 17 && g.count >= 2)
            .cloned()
            .expect("a random group with two lanes");
        let first = duplicate.coefficients[group.coefficients_offset as usize];
        duplicate.coefficients[group.coefficients_offset as usize + 1] = first;
        let error = plan_bytes::decode(&plan_bytes::encode(&duplicate, &key))
            .unwrap_err()
            .to_string();
        assert!(error.contains("random"), "{error}");
        // Root slot out of range: rejected.
        let mut roots = (*program).clone();
        roots.root_slots[0] = roots.state_slots;
        assert!(plan_bytes::decode(&plan_bytes::encode(&roots, &key)).is_err());
        // A flipped byte: rejected by the checksum.
        let mut damaged = bytes.clone();
        damaged[bytes.len() / 2] ^= 1;
        assert!(plan_bytes::decode(&damaged).is_err());
    }
}
