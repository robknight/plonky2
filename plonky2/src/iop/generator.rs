#[cfg(not(feature = "std"))]
use alloc::{
    boxed::Box,
    collections::BTreeMap,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::fmt::Debug;
use core::marker::PhantomData;
#[cfg(feature = "std")]
use std::collections::BTreeMap;

use anyhow::{anyhow, Result};
use dyn_clone::DynClone;
use plonky2_maybe_rayon::*;
#[cfg(feature = "timing")]
use web_time::Instant;

use crate::field::extension::Extendable;
use crate::field::types::Field;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::Target;
use crate::iop::wire::Wire;
use crate::iop::witness::{PartialWitness, PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
use crate::plonk::config::GenericConfig;
use crate::util::serialization::{Buffer, IoResult, Read, Write};

pub(crate) fn initial_witness<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    inputs: &PartialWitness<F>,
    prover_data: &'a ProverOnlyCircuitData<F, C, D>,
    common_data: &'a CommonCircuitData<F, D>,
) -> Result<PartitionWitness<'a, F>> {
    let config = &common_data.config;
    #[cfg(feature = "std")]
    let reused = prover_data
        .witness_buffers
        .take()
        .filter(|values| values.len() == prover_data.representative_map.len());
    #[cfg(not(feature = "std"))]
    let reused: Option<Vec<Option<F>>> = None;
    let values = match reused {
        Some(mut values) => {
            values
                .par_chunks_mut(4096)
                .for_each(|chunk| chunk.fill(None));
            values
        }
        None => vec![None; prover_data.representative_map.len()],
    };
    let mut witness = PartitionWitness {
        values,
        num_wires: config.num_wires,
        degree: common_data.degree(),
        representative_map: &prover_data.representative_map,
    };
    for (&t, &v) in &inputs.target_values {
        witness.set_target(t, v)?;
    }
    Ok(witness)
}

/// Given a `PartitionWitness` that has only inputs set, populates the rest of the witness using the
/// given set of generators.
///
/// Generators run in rounds. Initially every generator is pending; a round
/// runs its pending generators in parallel against the witness as it was at
/// the start of the round, then merges what they produced and queues the
/// unfinished generators watching a newly populated target. A generator that
/// needs a value produced in the same round simply reports itself unfinished
/// and is queued for another round when a watched target is populated.
///
/// With `std`, the first proof of a circuit records the rounds in which its
/// generators finish, and later proofs try to run the recorded
/// rounds directly, without the watch table or the runs that find their
/// dependencies missing. Input-dependent generators can change readiness;
/// the replay checks that every scheduled generator
/// finishes and that each round populates as many targets as recorded;
/// otherwise the witness is generated from scratch with the worklist, which
/// records a new schedule. Rounds are split across the generator pool's
/// threads in chunks; the smallest rounds run on the calling thread.
///
/// Witness buffers are reused across proofs, and large replay merges are
/// partitioned by representative range. Both merge paths check conflicting
/// values and populated counts.
pub fn generate_partial_witness<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    inputs: PartialWitness<F>,
    prover_data: &'a ProverOnlyCircuitData<F, C, D>,
    common_data: &'a CommonCircuitData<F, D>,
) -> Result<PartitionWitness<'a, F>> {
    let generators = &prover_data.generators;
    let new_witness =
        |inputs: &PartialWitness<F>| initial_witness(inputs, prover_data, common_data);

    #[cfg(feature = "std")]
    if let Some(schedule) = prover_data.generator_schedule.get() {
        #[cfg(feature = "timing")]
        let started = Instant::now();
        let mut witness = new_witness(&inputs)?;
        match replay_schedule(&mut witness, generators, &schedule)? {
            Ok(()) => {
                #[cfg(feature = "timing")]
                log::debug!(
                    "witness: {} generators in {} recorded rounds, {:?}",
                    generators.len(),
                    schedule.len(),
                    started.elapsed()
                );
                return Ok(witness);
            }
            Err(round) => {
                log::info!(
                    "witness: the recorded schedule failed in round {round}; running the worklist"
                );
                prover_data.generator_schedule.clear();
            }
        }
    }

    #[cfg(feature = "std")]
    let watch_table = prover_data.watch_table.0.get_or_init(|| {
        WatchTable::new(
            &prover_data.generator_indices_by_watches,
            prover_data.representative_map.len(),
        )
    });
    #[cfg(not(feature = "std"))]
    let watch_table = &WatchTable::new(
        &prover_data.generator_indices_by_watches,
        prover_data.representative_map.len(),
    );

    let mut witness = new_witness(&inputs)?;
    drop(inputs);

    let mut pending: Vec<u32> = (0..generators.len() as u32).collect();
    let mut expired = vec![false; generators.len()];
    let mut queued = vec![false; generators.len()];
    let mut remaining_generators = generators.len();
    let mut recording: Vec<(Vec<u32>, u32)> = Vec::new();

    let mut rounds = 0usize;
    let mut runs = 0usize;
    #[cfg(feature = "timing")]
    let mut run_time = core::time::Duration::ZERO;
    #[cfg(feature = "timing")]
    let started = Instant::now();
    #[cfg(feature = "timing")]
    let mut round_log: Vec<(usize, core::time::Duration)> = Vec::new();
    while !pending.is_empty() {
        rounds += 1;
        runs += pending.len();
        #[cfg(feature = "timing")]
        let round_started = Instant::now();
        let results = run_round(generators, &witness, &pending, Some(&expired));
        #[cfg(feature = "timing")]
        {
            run_time += round_started.elapsed();
            round_log.push((pending.len(), round_started.elapsed()));
        }

        // Finished generators first, so that a value produced this round does
        // not queue a generator that also finished this round.
        let mut finished_this_round = Vec::new();
        for (finished, _) in &results {
            for &generator_idx in finished {
                if !expired[generator_idx as usize] {
                    expired[generator_idx as usize] = true;
                    remaining_generators -= 1;
                    finished_this_round.push(generator_idx);
                }
            }
        }
        let mut populated = 0u32;
        let mut next_pending = Vec::new();
        for (_, values) in results {
            for (t, v) in values {
                if let Some(rep) = witness.set_target_returning_rep(t, v)? {
                    populated += 1;
                    for &watcher in watch_table.watchers(rep) {
                        if !expired[watcher as usize] && !queued[watcher as usize] {
                            queued[watcher as usize] = true;
                            next_pending.push(watcher);
                        }
                    }
                }
            }
        }
        for &generator_idx in &next_pending {
            queued[generator_idx as usize] = false;
        }
        recording.push((finished_this_round, populated));
        pending = next_pending;
    }
    #[cfg(feature = "timing")]
    {
        let small: Vec<_> = round_log.iter().filter(|(n, _)| *n < 10_000).collect();
        let small_time: core::time::Duration = small.iter().map(|(_, t)| *t).sum();
        let mut big: Vec<_> = round_log.iter().filter(|(n, _)| *n >= 10_000).collect();
        big.sort_by_key(|(n, _)| core::cmp::Reverse(*n));
        log::debug!(
            "witness: {} generators, {rounds} rounds, {runs} runs, {run_time:?} running generators, {:?} in all; {} rounds below 10k generators took {small_time:?}; big rounds {:?}",
            generators.len(),
            started.elapsed(),
            small.len(),
            big
        );
    }
    #[cfg(not(feature = "timing"))]
    log::debug!(
        "witness: {} generators, {rounds} rounds, {runs} runs",
        generators.len()
    );

    if remaining_generators != 0 {
        return Err(anyhow!("{} generators weren't run", remaining_generators));
    }
    #[cfg(feature = "std")]
    prover_data.generator_schedule.set(recording);
    #[cfg(not(feature = "std"))]
    drop(recording);
    Ok(witness)
}

/// Runs the generators `indices` (those not `expired`) against `witness`,
/// in parallel chunks on the generator pool unless the round is small.
fn run_round<F: RichField + Extendable<D>, const D: usize>(
    generators: &[WitnessGeneratorRef<F, D>],
    witness: &PartitionWitness<F>,
    indices: &[u32],
    expired: Option<&[bool]>,
) -> Vec<RoundResults<F>> {
    let task = |chunk: &[u32]| -> RoundResults<F> {
        with_randomness(|| {
            let mut finished = Vec::new();
            let mut buffer = GeneratedValues::empty();
            let mut values = Vec::new();
            for &generator_idx in chunk {
                if expired.is_some_and(|expired| expired[generator_idx as usize]) {
                    continue;
                }
                if generators[generator_idx as usize]
                    .0
                    .run(witness, &mut buffer)
                {
                    finished.push(generator_idx);
                }
                values.append(&mut buffer.target_values);
            }
            (finished, values)
        })
    };
    // Small rounds still split across the pool's threads, in chunks of at
    // least GENERATOR_CHUNK_MIN; the smallest run on the calling thread.
    if indices.len() <= GENERATOR_CHUNK_MIN {
        return vec![task(indices)];
    }
    let chunk = (indices.len() / 8).clamp(GENERATOR_CHUNK_MIN, GENERATOR_CHUNK);
    let run = || indices.par_chunks(chunk).map(task).collect();
    #[cfg(all(feature = "parallel", feature = "std"))]
    {
        generator_pool().install(run)
    }
    #[cfg(not(all(feature = "parallel", feature = "std")))]
    {
        run()
    }
}

/// Runs the recorded rounds. `Ok(Err(round))` reports the first round whose
/// generators did not all finish or that populated a different number of
/// targets than recorded; the witness is then incomplete.
#[cfg(feature = "std")]
fn replay_schedule<F: RichField + Extendable<D>, const D: usize>(
    witness: &mut PartitionWitness<F>,
    generators: &[WitnessGeneratorRef<F, D>],
    schedule: &[(Vec<u32>, u32)],
) -> Result<core::result::Result<(), usize>> {
    #[cfg(feature = "timing")]
    let (mut run_time, mut merge_time, mut small_time, mut values_merged) = (
        core::time::Duration::ZERO,
        core::time::Duration::ZERO,
        core::time::Duration::ZERO,
        0usize,
    );
    #[cfg(feature = "timing")]
    let mut big: Vec<(usize, core::time::Duration, core::time::Duration)> = Vec::new();
    for (round, (indices, populated)) in schedule.iter().enumerate() {
        #[cfg(feature = "timing")]
        let started = Instant::now();
        let results = run_round(generators, witness, indices, None);
        #[cfg(feature = "timing")]
        let ran = started.elapsed();
        let finished: usize = results.iter().map(|(finished, _)| finished.len()).sum();
        if finished != indices.len() {
            return Ok(Err(round));
        }
        #[cfg(feature = "timing")]
        {
            values_merged += results
                .iter()
                .map(|(_, values)| values.len())
                .sum::<usize>();
        }
        let count = merge_replay(witness, results, true)?;
        if count != *populated {
            return Ok(Err(round));
        }
        #[cfg(feature = "timing")]
        {
            let merged = started.elapsed() - ran;
            run_time += ran;
            merge_time += merged;
            if indices.len() < 10_000 {
                small_time += ran + merged;
            } else {
                big.push((indices.len(), ran, merged));
            }
        }
    }
    #[cfg(feature = "timing")]
    log::debug!(
        "witness replay: {run_time:?} running, {merge_time:?} merging {values_merged} values; small rounds {small_time:?}; big rounds (size, run, merge) {big:?}"
    );
    Ok(Ok(()))
}

/// Keep conflict checks on replay too: a schedule records readiness, not values.
#[cfg(feature = "std")]
fn merge_replay<F: RichField>(
    witness: &mut PartitionWitness<F>,
    results: Vec<RoundResults<F>>,
    _parallel: bool,
) -> Result<u32> {
    #[cfg(feature = "parallel")]
    if _parallel
        && results
            .iter()
            .map(|(_, values)| values.len())
            .sum::<usize>()
            >= 10_000
    {
        let num_wires = witness.num_wires;
        let degree = witness.degree;
        let map = witness.representative_map;
        let chunk_size = witness
            .values
            .len()
            .div_ceil(generator_pool().current_num_threads())
            .max(1);
        let parts = witness.values.len().div_ceil(chunk_size);
        // Assign each update to its owner once. Per-task buckets keep this
        // scatter parallel; merging buckets never scans unrelated updates.
        let buckets = generator_pool().install(|| {
            results
                .into_par_iter()
                .fold(
                    || vec![Vec::new(); parts],
                    |mut buckets, (_, values)| {
                        for (target, value) in values {
                            let rep = map[target.index(num_wires, degree)];
                            buckets[rep / chunk_size].push((rep, value));
                        }
                        buckets
                    },
                )
                .reduce(
                    || vec![Vec::new(); parts],
                    |mut left, right| {
                        for (dest, mut source) in left.iter_mut().zip(right) {
                            dest.append(&mut source);
                        }
                        left
                    },
                )
        });
        return generator_pool().install(|| {
            let merge = |(part, (chunk, updates)): (usize, (&mut [Option<F>], Vec<(usize, F)>))| -> Result<u32> {
                let base = part * chunk_size;
                let mut count = 0;
                for (rep, value) in updates {
                    let slot = &mut chunk[rep - base];
                    if let Some(old) = *slot {
                        if old != value {
                            return Err(anyhow!(
                                "Representative {rep} was set twice with different values: {old} != {value}"
                            ));
                        }
                    } else {
                        *slot = Some(value);
                        count += 1;
                    }
                }
                Ok(count)
            };
            witness.values.par_chunks_mut(chunk_size).zip(buckets.into_par_iter()).enumerate().map(merge)
                .try_reduce(|| 0, |a, b| Ok(a + b))
        });
    }
    let mut count = 0;
    for (_, values) in results {
        for (t, v) in values {
            count += u32::from(witness.set_target_returning_rep(t, v)?.is_some());
        }
    }
    Ok(count)
}

/// The rounds in which a circuit's generators finished on its first proof:
/// per round, the generators that finished and the number of targets the
/// round populated. Replayed by later proofs of the circuit; cleared when a
/// replay does not reproduce the recording. Ignored by equality.
#[cfg(feature = "std")]
#[derive(Default)]
pub(crate) struct GeneratorSchedule(std::sync::Mutex<Option<std::sync::Arc<Vec<(Vec<u32>, u32)>>>>);

#[cfg(feature = "std")]
impl GeneratorSchedule {
    /// The recorded rounds, if any.
    pub fn get(&self) -> Option<std::sync::Arc<Vec<(Vec<u32>, u32)>>> {
        self.0.lock().unwrap().clone()
    }

    fn set(&self, rounds: Vec<(Vec<u32>, u32)>) {
        *self.0.lock().unwrap() = Some(std::sync::Arc::new(rounds));
    }

    /// Discards the recording; the next proof records again.
    pub fn clear(&self) {
        *self.0.lock().unwrap() = None;
    }
}

#[cfg(feature = "std")]
impl Clone for GeneratorSchedule {
    fn clone(&self) -> Self {
        Self(std::sync::Mutex::new(self.get()))
    }
}

#[cfg(feature = "std")]
impl Debug for GeneratorSchedule {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.get() {
            Some(rounds) => write!(f, "GeneratorSchedule({} rounds)", rounds.len()),
            None => write!(f, "GeneratorSchedule(none)"),
        }
    }
}

#[cfg(feature = "std")]
impl PartialEq for GeneratorSchedule {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

#[cfg(feature = "std")]
impl Eq for GeneratorSchedule {}

/// Generators per task of a large round.
const GENERATOR_CHUNK: usize = 2048;
/// Generators per task of a small round, and the largest round run on the calling thread.
const GENERATOR_CHUNK_MIN: usize = 128;

/// The finished generators and the values produced by one task of a round.
type RoundResults<F> = (Vec<u32>, Vec<(Target, F)>);

/// The thread pool the rounds run on. Generator runs are bound by memory
/// latency (random reads of the witness and the representative map), and on
/// Apple silicon adding the efficiency cores to the four performance cores
/// made a round 2.5x slower, so the pool has one thread per performance core
/// there and one per available core elsewhere (`available_parallelism`, which
/// follows Linux CPU quotas). On Linux, 4 to 32 threads measured within 12% of
/// each other.
#[cfg(all(feature = "parallel", feature = "std"))]
fn generator_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let threads = performance_cores()
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("plonky2-generators-{i}"))
            .build()
            .expect("building the generator thread pool")
    })
}

/// The number of performance cores, where the system distinguishes them.
#[cfg(all(feature = "parallel", feature = "std", target_os = "macos"))]
fn performance_cores() -> Option<usize> {
    let name = c"hw.perflevel0.logicalcpu";
    let mut value: u32 = 0;
    let mut size = core::mem::size_of::<u32>();
    // SAFETY: the buffer and its size describe a u32 that sysctlbyname fills.
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&mut value as *mut u32).cast(),
            &mut size,
            core::ptr::null_mut(),
            0,
        )
    };
    (status == 0 && value > 0).then_some(value as usize)
}

#[cfg(all(feature = "parallel", feature = "std", not(target_os = "macos")))]
fn performance_cores() -> Option<usize> {
    None
}

/// The generators watching each representative, as one flat table: the
/// watchers of representative `r` are `generators[offsets[r]..offsets[r + 1]]`.
/// Built once per circuit from `generator_indices_by_watches`.
/// The watch table, built on a circuit's first proof. A cache derived from the
/// generators, which circuit equality already compares: it compares equal
/// whether built or not, so circuit data equals its serialized round trip
/// after proving, as in upstream plonky2.
#[cfg(feature = "std")]
#[derive(Debug, Clone, Default)]
pub(crate) struct WatchTableCache(pub(crate) std::sync::OnceLock<WatchTable>);

#[cfg(feature = "std")]
impl PartialEq for WatchTableCache {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

#[cfg(feature = "std")]
impl Eq for WatchTableCache {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WatchTable {
    offsets: Vec<u32>,
    generators: Vec<u32>,
}

impl WatchTable {
    pub fn new(by_watches: &BTreeMap<usize, Vec<usize>>, num_representatives: usize) -> Self {
        let mut offsets = vec![0u32; num_representatives + 1];
        for (&rep, watchers) in by_watches {
            offsets[rep + 1] = watchers.len() as u32;
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }
        let mut generators = vec![0u32; *offsets.last().unwrap() as usize];
        for (&rep, watchers) in by_watches {
            let start = offsets[rep] as usize;
            for (i, &watcher) in watchers.iter().enumerate() {
                generators[start + i] = watcher as u32;
            }
        }
        Self {
            offsets,
            generators,
        }
    }

    pub fn watchers(&self, representative: usize) -> &[u32] {
        &self.generators
            [self.offsets[representative] as usize..self.offsets[representative + 1] as usize]
    }
}

/// The Poseidon gate's inputs at `row`: the 12 state words and the swap wire.
fn poseidon_inputs(row: usize) -> [Target; 13] {
    core::array::from_fn(|i| Target::wire(row, if i < 12 { i } else { 24 }))
}

/// The Poseidon gate's outputs at `row`, in the CPU generator's emission
/// order: deltas, S-box inputs, final outputs.
fn poseidon_outputs(row: usize) -> [Target; 122] {
    core::array::from_fn(|i| Target::wire(row, if i < 110 { 25 + i } else { 12 + i - 110 }))
}

/// A generator's witness computation as a typed operation. Available in every
/// build; with `std` and `rand_chacha`, the compiled witness executor in
/// `crate::plonk::witness_program` can run it in place of the generator's code.
/// Parameters describe the operation, independently of circuit identity or
/// debug IDs. A generator that returns one must compute exactly what the
/// operation describes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum WitnessOperation {
    Poseidon {
        row: usize,
    },
    Arithmetic {
        inputs: [Target; 3],
        output: Target,
        coefficients: [u64; 2],
    },
    Copy {
        source: Target,
        destination: Target,
    },
    Constant {
        output: Target,
        value: u64,
    },
    Equality {
        inputs: [Target; 2],
        outputs: [Target; 2],
    },
    ExtensionArithmetic {
        inputs: [Target; 6],
        outputs: [Target; 2],
        coefficients: [u64; 2],
    },
    ExtensionMultiply {
        inputs: [Target; 4],
        outputs: [Target; 2],
        coefficient: u64,
    },
    RandomAccess {
        row: usize,
        copy: usize,
        bits: usize,
        num_copies: usize,
        num_extra_constants: usize,
    },
    BaseSplit {
        row: usize,
        num_limbs: usize,
        limb_bits: u32,
    },
    WireSplit {
        integer: Target,
        gates: Vec<usize>,
        num_limbs: usize,
    },
    LowHigh {
        integer: Target,
        low: Target,
        high: Target,
        low_bits: u32,
    },
    Reducing {
        row: usize,
        num_coeffs: usize,
        extension_coeffs: bool,
    },
    TableLookup {
        index: [Target; 2],
        entries: Vec<Vec<Target>>,
        outputs: Vec<Target>,
    },
    ExtensionQuotient {
        inputs: [Target; 4],
        outputs: [Target; 2],
    },
    EcAddXu {
        row: usize,
        curve_b1: u64,
        generator_x: [u64; 5],
        generator_u: [u64; 5],
    },
    PoseidonMds {
        row: usize,
    },
    RandomValue {
        target: Target,
    },
    ConditionalZero {
        if_zero: Target,
        then_zero: Target,
        quotient: Target,
    },
    LtMask {
        count: Target,
        mask: Vec<Target>,
    },
    QuinticMultiply {
        inputs: [Target; 10],
        outputs: [Target; 5],
    },
    QuinticQuotient {
        inputs: [Target; 10],
        outputs: [Target; 5],
    },
    EcAddHomogOffset {
        inputs: [Target; 20],
        outputs: [Target; 20],
        curve_b1: u64,
    },
    EcScalarMul {
        inputs: [Target; 10],
        outputs: [Target; 10],
        curve_b1: u64,
        scalar_words: Vec<u32>,
    },
    CosetInterpolation {
        inputs: Vec<Target>,
        outputs: Vec<Target>,
        degree: usize,
        domain: Vec<u64>,
        weights: Vec<u64>,
    },
    Comparison {
        inputs: [Target; 2],
        outputs: Vec<Target>,
        num_chunks: usize,
        chunk_bits: usize,
    },
}

impl WitnessOperation {
    pub fn field_supported<F: Field>() -> bool {
        core::any::TypeId::of::<F>()
            == core::any::TypeId::of::<crate::field::goldilocks_field::GoldilocksField>()
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::Poseidon { .. } => "poseidon",
            Self::Arithmetic { .. } => "arithmetic",
            Self::Copy { .. } => "copy",
            Self::Constant { .. } => "constant",
            Self::Equality { .. } => "equality",
            Self::ExtensionArithmetic { .. } => "extension_arithmetic",
            Self::ExtensionMultiply { .. } => "extension_multiply",
            Self::RandomAccess { .. } => "random_access",
            Self::BaseSplit { .. } => "base_split",
            Self::WireSplit { .. } => "wire_split",
            Self::LowHigh { .. } => "low_high",
            Self::Reducing {
                extension_coeffs: false,
                ..
            } => "reducing",
            Self::Reducing {
                extension_coeffs: true,
                ..
            } => "reducing_extension",
            Self::TableLookup { .. } => "table_lookup",
            Self::ExtensionQuotient { .. } => "extension_quotient",
            Self::EcAddXu { .. } => "ec_add_xu",
            Self::PoseidonMds { .. } => "poseidon_mds",
            Self::RandomValue { .. } => "random_value",
            Self::ConditionalZero { .. } => "conditional_zero",
            Self::LtMask { .. } => "lt_mask",
            Self::QuinticMultiply { .. } => "quintic_multiply",
            Self::QuinticQuotient { .. } => "quintic_quotient",
            Self::EcAddHomogOffset { .. } => "ec_add_homog_offset",
            Self::EcScalarMul { .. } => "ec_scalar_mul",
            Self::CosetInterpolation { .. } => "coset_interpolation",
            Self::Comparison { .. } => "comparison",
        }
    }

    pub fn inputs(&self) -> Vec<Target> {
        match self {
            Self::Poseidon { row } => poseidon_inputs(*row).to_vec(),
            Self::Arithmetic { inputs, .. } => inputs.to_vec(),
            Self::Copy { source, .. } => vec![*source],
            Self::Constant { .. } => vec![],
            Self::Equality { inputs, .. } => inputs.to_vec(),
            Self::ExtensionArithmetic { inputs, .. } => inputs.to_vec(),
            Self::ExtensionMultiply { inputs, .. } => inputs.to_vec(),
            Self::RandomAccess {
                row, copy, bits, ..
            } => {
                let size = 1usize << *bits;
                let start = (2 + size) * *copy;
                core::iter::once(Target::wire(*row, start))
                    .chain((0..size).map(|i| Target::wire(*row, start + 2 + i)))
                    .collect()
            }
            Self::BaseSplit { row, .. } => vec![Target::wire(*row, 0)],
            Self::WireSplit { integer, .. } => vec![*integer],
            Self::LowHigh { integer, .. } => vec![*integer],
            Self::Reducing {
                row,
                num_coeffs,
                extension_coeffs,
            } => {
                let coeff_words = *num_coeffs * if *extension_coeffs { 2 } else { 1 };
                (2..6 + coeff_words)
                    .map(|column| Target::wire(*row, column))
                    .collect()
            }
            Self::TableLookup { index, entries, .. } => index
                .iter()
                .copied()
                .chain(entries.iter().flatten().copied())
                .collect(),
            Self::ExtensionQuotient { inputs, .. } => inputs.to_vec(),
            Self::EcAddXu { row, .. } => (0..26).map(|column| Target::wire(*row, column)).collect(),
            Self::PoseidonMds { row } => (0..24).map(|column| Target::wire(*row, column)).collect(),
            Self::RandomValue { .. } => vec![],
            Self::ConditionalZero {
                if_zero, then_zero, ..
            } => vec![*if_zero, *then_zero],
            Self::LtMask { count, .. } => vec![*count],
            Self::QuinticMultiply { inputs, .. } | Self::QuinticQuotient { inputs, .. } => {
                inputs.to_vec()
            }
            Self::EcAddHomogOffset { inputs, .. } => inputs.to_vec(),
            Self::EcScalarMul { inputs, .. } => inputs.to_vec(),
            Self::CosetInterpolation { inputs, .. } => inputs.clone(),
            Self::Comparison { inputs, .. } => inputs.to_vec(),
        }
    }

    pub fn outputs(&self) -> Vec<Target> {
        match self {
            Self::Poseidon { row } => poseidon_outputs(*row).to_vec(),
            Self::Arithmetic { output, .. } | Self::Constant { output, .. } => vec![*output],
            Self::Copy { destination, .. } => vec![*destination],
            Self::Equality { outputs, .. } => outputs.to_vec(),
            Self::ExtensionArithmetic { outputs, .. } => outputs.to_vec(),
            Self::ExtensionMultiply { outputs, .. } => outputs.to_vec(),
            Self::RandomAccess {
                row,
                copy,
                bits,
                num_copies,
                num_extra_constants,
            } => {
                let size = 1usize << *bits;
                let start = (2 + size) * *copy;
                let bit_start = (2 + size) * *num_copies + *num_extra_constants + *copy * *bits;
                core::iter::once(Target::wire(*row, start + 1))
                    .chain((0..*bits).map(|i| Target::wire(*row, bit_start + i)))
                    .collect()
            }
            Self::BaseSplit { row, num_limbs, .. } => (1..=*num_limbs)
                .map(|column| Target::wire(*row, column))
                .collect(),
            Self::WireSplit { gates, .. } => {
                gates.iter().map(|&row| Target::wire(row, 0)).collect()
            }
            Self::LowHigh { low, high, .. } => vec![*low, *high],
            Self::Reducing {
                row,
                num_coeffs,
                extension_coeffs,
            } => {
                let start_accs = 6 + *num_coeffs * if *extension_coeffs { 2 } else { 1 };
                let mut outputs = Vec::with_capacity(2 * *num_coeffs + 2);
                for i in 0..*num_coeffs {
                    let column = if i + 1 == *num_coeffs {
                        0
                    } else {
                        start_accs + 2 * i
                    };
                    outputs.extend([Target::wire(*row, column), Target::wire(*row, column + 1)]);
                }
                if !*extension_coeffs {
                    outputs.extend([Target::wire(*row, 0), Target::wire(*row, 1)]);
                }
                outputs
            }
            Self::TableLookup { outputs, .. } => outputs.clone(),
            Self::ExtensionQuotient { outputs, .. } => outputs.to_vec(),
            Self::EcAddXu { row, .. } => {
                (26..116).map(|column| Target::wire(*row, column)).collect()
            }
            Self::PoseidonMds { row } => {
                (24..48).map(|column| Target::wire(*row, column)).collect()
            }
            Self::RandomValue { target } => vec![*target],
            Self::ConditionalZero { quotient, .. } => vec![*quotient],
            Self::LtMask { mask, .. } => mask.clone(),
            Self::QuinticMultiply { outputs, .. } | Self::QuinticQuotient { outputs, .. } => {
                outputs.to_vec()
            }
            Self::EcAddHomogOffset { outputs, .. } => outputs.to_vec(),
            Self::EcScalarMul { outputs, .. } => outputs.to_vec(),
            Self::CosetInterpolation { outputs, .. } => outputs.clone(),
            Self::Comparison { outputs, .. } => outputs.clone(),
        }
    }
}

/// A generator participates in the generation of the witness.
pub trait WitnessGenerator<F: RichField + Extendable<D>, const D: usize>:
    'static + Send + Sync + Debug + DynClone
{
    fn id(&self) -> String;

    /// Targets to be "watched" by this generator. Whenever a target in the watch list is populated,
    /// the generator will be queued to run.
    fn watch_list(&self) -> Vec<Target>;

    /// This generator's witness computation as a typed operation, so the
    /// compiled witness executor can run it (`crate::plonk::witness_program`,
    /// available with `std` and `rand_chacha`).
    /// `None` by default; a circuit with any generator returning `None` keeps
    /// the worklist.
    fn witness_operation(&self) -> Option<WitnessOperation> {
        None
    }

    /// Run this generator, returning a flag indicating whether the generator is finished. If the
    /// flag is true, the generator will never be run again, otherwise it will be queued for another
    /// run next time a target in its watch list is populated.
    fn run(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) -> bool;

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()>;

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self>
    where
        Self: Sized;
}

impl<F: RichField + Extendable<D>, const D: usize> Clone for Box<dyn WitnessGenerator<F, D>> {
    fn clone(&self) -> Self {
        dyn_clone::clone_box(&**self)
    }
}

/// A wrapper around an `Box<WitnessGenerator>` which implements `PartialEq`
/// and `Eq` based on generator IDs.
#[derive(Clone)]
pub struct WitnessGeneratorRef<F: RichField + Extendable<D>, const D: usize>(
    pub Box<dyn WitnessGenerator<F, D>>,
);

impl<F: RichField + Extendable<D>, const D: usize> WitnessGeneratorRef<F, D> {
    pub fn new<G: WitnessGenerator<F, D>>(generator: G) -> WitnessGeneratorRef<F, D> {
        WitnessGeneratorRef(Box::new(generator))
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PartialEq for WitnessGeneratorRef<F, D> {
    fn eq(&self, other: &Self) -> bool {
        self.0.id() == other.0.id()
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Eq for WitnessGeneratorRef<F, D> {}

impl<F: RichField + Extendable<D>, const D: usize> Debug for WitnessGeneratorRef<F, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0.id())
    }
}

/// Values generated by a generator invocation.
#[derive(Debug)]
pub struct GeneratedValues<F: Field> {
    pub target_values: Vec<(Target, F)>,
}

impl<F: Field> From<Vec<(Target, F)>> for GeneratedValues<F> {
    fn from(target_values: Vec<(Target, F)>) -> Self {
        Self { target_values }
    }
}

impl<F: Field> WitnessWrite<F> for GeneratedValues<F> {
    fn set_target(&mut self, target: Target, value: F) -> Result<()> {
        self.target_values.push((target, value));

        Ok(())
    }
}

impl<F: Field> GeneratedValues<F> {
    pub fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity).into()
    }

    pub fn empty() -> Self {
        Vec::new().into()
    }

    pub fn singleton_wire(wire: Wire, value: F) -> Self {
        Self::singleton_target(Target::Wire(wire), value)
    }

    pub fn singleton_target(target: Target, value: F) -> Self {
        vec![(target, value)].into()
    }

    pub fn singleton_extension_target<const D: usize>(
        et: ExtensionTarget<D>,
        value: F::Extension,
    ) -> Result<Self>
    where
        F: RichField + Extendable<D>,
    {
        let mut witness = Self::with_capacity(D);
        witness.set_extension_target(et, value)?;

        Ok(witness)
    }
}

/// A generator which runs once after a list of dependencies is present in the witness.
pub trait SimpleGenerator<F: RichField + Extendable<D>, const D: usize>:
    'static + Send + Sync + Debug + Clone
{
    fn id(&self) -> String;

    fn dependencies(&self) -> Vec<Target>;

    fn witness_operation(&self) -> Option<WitnessOperation> {
        None
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()>;

    fn adapter(self) -> SimpleGeneratorAdapter<F, Self, D>
    where
        Self: Sized,
    {
        SimpleGeneratorAdapter {
            inner: self,
            _phantom: PhantomData,
        }
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()>;

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self>
    where
        Self: Sized;
}

#[derive(Debug, Clone)]
pub struct SimpleGeneratorAdapter<
    F: RichField + Extendable<D>,
    SG: SimpleGenerator<F, D>,
    const D: usize,
> {
    _phantom: PhantomData<F>,
    inner: SG,
}

impl<F: RichField + Extendable<D>, SG: SimpleGenerator<F, D>, const D: usize> WitnessGenerator<F, D>
    for SimpleGeneratorAdapter<F, SG, D>
{
    fn id(&self) -> String {
        self.inner.id()
    }

    fn watch_list(&self) -> Vec<Target> {
        self.inner.dependencies()
    }

    fn witness_operation(&self) -> Option<WitnessOperation> {
        self.inner.witness_operation()
    }

    fn run(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) -> bool {
        if witness.contains_all(&self.inner.dependencies()) {
            self.inner.run_once(witness, out_buffer).is_ok()
        } else {
            false
        }
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        self.inner.serialize(dst, common_data)
    }

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Self {
            inner: SG::deserialize(src, common_data)?,
            _phantom: PhantomData,
        })
    }
}

/// A generator which copies one wire to another.
#[derive(Debug, Default, Clone)]
pub struct CopyGenerator {
    pub(crate) src: Target,
    pub(crate) dst: Target,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for CopyGenerator {
    fn id(&self) -> String {
        "CopyGenerator".to_string()
    }

    fn witness_operation(&self) -> Option<WitnessOperation> {
        WitnessOperation::field_supported::<F>().then_some(WitnessOperation::Copy {
            source: self.src,
            destination: self.dst,
        })
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![self.src]
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let value = witness.get_target(self.src);
        out_buffer.set_target(self.dst, value)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target(self.src)?;
        dst.write_target(self.dst)
    }

    fn deserialize(source: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let src = source.read_target()?;
        let dst = source.read_target()?;
        Ok(Self { src, dst })
    }
}

/// A generator for including a random value
#[derive(Debug, Default, Clone)]
pub struct RandomValueGenerator {
    pub(crate) target: Target,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for RandomValueGenerator {
    fn id(&self) -> String {
        "RandomValueGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        Vec::new()
    }

    fn witness_operation(&self) -> Option<WitnessOperation> {
        WitnessOperation::field_supported::<F>().then_some(WitnessOperation::RandomValue {
            target: self.target,
        })
    }

    fn run_once(
        &self,
        _witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        out_buffer.set_target(self.target, random_value())
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target(self.target)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let target = src.read_target()?;
        Ok(Self { target })
    }
}

/// Sample with a short-lived, OS-seeded CSPRNG inside a generator work chunk.
/// Direct calls outside the scheduler use upstream's OS sampler.
#[cfg(all(feature = "std", feature = "rand_chacha"))]
pub(crate) fn random_value<F: Field>() -> F {
    crate::util::blinding_rng::sample()
}

#[cfg(not(all(feature = "std", feature = "rand_chacha")))]
pub(crate) fn random_value<F: Field>() -> F {
    F::rand()
}

/// A scope owns lazily seeded randomness only until this work chunk finishes.
/// This also covers recorded schedule replay.
pub(crate) fn with_randomness<T>(work: impl FnOnce() -> T) -> T {
    #[cfg(all(feature = "std", feature = "rand_chacha"))]
    {
        crate::util::blinding_rng::with_scope(work)
    }
    #[cfg(not(all(feature = "std", feature = "rand_chacha")))]
    {
        work()
    }
}

/// A generator for testing if a value equals zero
#[derive(Debug, Default, Clone)]
pub struct NonzeroTestGenerator {
    pub(crate) to_test: Target,
    pub(crate) dummy: Target,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for NonzeroTestGenerator {
    fn id(&self) -> String {
        "NonzeroTestGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![self.to_test]
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let to_test_value = witness.get_target(self.to_test);

        let dummy_value = if to_test_value == F::ZERO {
            F::ONE
        } else {
            to_test_value.inverse()
        };

        out_buffer.set_target(self.dummy, dummy_value)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target(self.to_test)?;
        dst.write_target(self.dummy)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let to_test = src.read_target()?;
        let dummy = src.read_target()?;
        Ok(Self { to_test, dummy })
    }
}

/// Generator used to fill an extra constant.
#[derive(Debug, Clone, Default)]
pub struct ConstantGenerator<F: Field> {
    pub row: usize,
    pub constant_index: usize,
    pub wire_index: usize,
    pub constant: F,
}

impl<F: Field> ConstantGenerator<F> {
    pub fn set_constant(&mut self, c: F) {
        self.constant = c;
    }
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for ConstantGenerator<F> {
    fn id(&self) -> String {
        "ConstantGenerator".to_string()
    }

    fn witness_operation(&self) -> Option<WitnessOperation> {
        WitnessOperation::field_supported::<F>().then(|| WitnessOperation::Constant {
            output: Target::wire(self.row, self.wire_index),
            value: self.constant.to_canonical_u64(),
        })
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![]
    }

    fn run_once(
        &self,
        _witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        out_buffer.set_target(Target::wire(self.row, self.wire_index), self.constant)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_usize(self.constant_index)?;
        dst.write_usize(self.wire_index)?;
        dst.write_field(self.constant)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let constant_index = src.read_usize()?;
        let wire_index = src.read_usize()?;
        let constant = src.read_field()?;
        Ok(Self {
            row,
            constant_index,
            wire_index,
            constant,
        })
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::Field;
    use crate::iop::witness::WitnessWrite;
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::{CircuitConfig, CircuitData};
    use crate::plonk::config::PoseidonGoldilocksConfig;

    type F = GoldilocksField;
    type C = PoseidonGoldilocksConfig;

    /// A chain of hashes and arithmetic, so that generators finish over many rounds.
    fn chain_circuit() -> (CircuitData<F, C, 2>, Target) {
        let mut builder = CircuitBuilder::<F, 2>::new(CircuitConfig::standard_recursion_config());
        let input = builder.add_virtual_target();
        let mut state = vec![input; 4];
        for i in 0..20u64 {
            let hash =
                builder.hash_n_to_hash_no_pad::<crate::hash::poseidon::PoseidonHash>(state.clone());
            let c = builder.constant(F::from_canonical_u64(i + 1));
            let x = builder.mul_add(hash.elements[0], c, hash.elements[1]);
            state = vec![x, hash.elements[1], hash.elements[2], hash.elements[3]];
        }
        builder.register_public_inputs(&state);
        (builder.build::<C>(), input)
    }

    #[test]
    fn random_chunks_refresh_across_recorded_schedule_replay() {
        let mut builder = CircuitBuilder::<F, 2>::new(CircuitConfig::standard_recursion_config());
        let targets = builder.add_virtual_targets(8201);
        for &target in &targets {
            builder.add_simple_generator(RandomValueGenerator { target });
        }
        let data = builder.build::<C>();
        let first =
            generate_partial_witness(PartialWitness::new(), &data.prover_only, &data.common)
                .unwrap()
                .get_targets(&targets);
        assert!(data.prover_only.generator_schedule.get().is_some());
        let replayed =
            generate_partial_witness(PartialWitness::new(), &data.prover_only, &data.common)
                .unwrap()
                .get_targets(&targets);
        assert!(first
            .chunks(4096)
            .zip(replayed.chunks(4096))
            .all(|(a, b)| a != b));
        data.prover_only.generator_schedule.clear();
        let worklist =
            generate_partial_witness(PartialWitness::new(), &data.prover_only, &data.common)
                .unwrap()
                .get_targets(&targets);
        assert!(replayed
            .chunks(4096)
            .zip(worklist.chunks(4096))
            .all(|(a, b)| a != b));
    }

    #[test]
    fn recorded_schedule_reproduces_the_worklist() {
        let (data, input) = chain_circuit();
        let mut pw = PartialWitness::new();
        pw.set_target(input, F::from_canonical_u64(7)).unwrap();
        assert!(data.prover_only.generator_schedule.get().is_none());
        let first = generate_partial_witness(pw.clone(), &data.prover_only, &data.common).unwrap();
        let schedule = data
            .prover_only
            .generator_schedule
            .get()
            .expect("the first proof records the schedule");
        assert!(schedule.len() > 1);
        assert_eq!(
            schedule.iter().map(|(round, _)| round.len()).sum::<usize>(),
            data.prover_only.generators.len()
        );
        // Unused wires get random values, so the witnesses are compared on
        // the public inputs (the end of the chain) and on which targets are set.
        let same = |a: &PartitionWitness<F>, b: &PartitionWitness<F>| {
            assert_eq!(
                a.get_targets(&data.prover_only.public_inputs),
                b.get_targets(&data.prover_only.public_inputs)
            );
            assert!(a
                .values
                .iter()
                .zip(&b.values)
                .all(|(x, y)| x.is_some() == y.is_some()));
        };
        let replayed =
            generate_partial_witness(pw.clone(), &data.prover_only, &data.common).unwrap();
        same(&replayed, &first);
        data.prover_only.generator_schedule.clear();
        let worklist = generate_partial_witness(pw, &data.prover_only, &data.common).unwrap();
        same(&worklist, &first);
        assert!(data.prover_only.generator_schedule.get().is_some());
        // The schedule survives a full proof.
        let mut pw = PartialWitness::new();
        pw.set_target(input, F::from_canonical_u64(8)).unwrap();
        let proof = data.prove(pw).unwrap();
        data.verify(proof).unwrap();
    }

    #[test]
    fn replay_merge_checks_aliases_and_conflicts() {
        // Cross several partitions, including an uneven final partition. Aliases
        // must count only once, whether they occur in one task or two tasks.
        let map: Vec<_> = (0..20_002).map(|i| i / 2).collect();
        let results = || {
            vec![
                (
                    vec![],
                    (0..20_002)
                        .step_by(2)
                        .map(|i| {
                            (
                                Target::VirtualTarget { index: i },
                                F::from_canonical_usize(i / 2),
                            )
                        })
                        .collect(),
                ),
                (
                    vec![],
                    (1..20_002)
                        .step_by(2)
                        .map(|i| {
                            (
                                Target::VirtualTarget { index: i },
                                F::from_canonical_usize(i / 2),
                            )
                        })
                        .collect(),
                ),
            ]
        };
        let mut serial = PartitionWitness::new(0, 0, &map);
        let mut parallel = serial.clone();
        assert_eq!(merge_replay(&mut serial, results(), false).unwrap(), 10_001);
        assert_eq!(
            merge_replay(&mut parallel, results(), true).unwrap(),
            10_001
        );
        assert_eq!(parallel.values, serial.values);
        assert_eq!(merge_replay(&mut parallel, results(), true).unwrap(), 0);
        for enabled in [false, true] {
            let mut conflicting = results();
            conflicting[1].1.last_mut().unwrap().1 = F::NEG_ONE;
            let mut witness = PartitionWitness::new(0, 0, &map);
            assert!(merge_replay(&mut witness, conflicting, enabled).is_err());
            assert!(merge_replay(
                &mut serial.clone(),
                vec![(vec![], vec![(Target::VirtualTarget { index: 0 }, F::ONE)])],
                enabled
            )
            .is_err());
        }
    }

    #[test]
    fn reusable_witness_is_reset_between_distinct_proofs() {
        let (data, input) = chain_circuit();
        for value in [7, 42, 9] {
            let mut pw = PartialWitness::new();
            pw.set_target(input, F::from_canonical_u64(value)).unwrap();
            let witness =
                generate_partial_witness(pw.clone(), &data.prover_only, &data.common).unwrap();
            assert_eq!(witness.get_target(input), F::from_canonical_u64(value));
            let (_, buffer) = witness.full_witness_and_buffer();
            data.prover_only.witness_buffers.recycle(buffer);
            let proof = data.prove(pw).unwrap();
            data.verify(proof).unwrap();
        }
    }

    #[test]
    fn shared_circuit_proves_distinct_inputs_concurrently() {
        let (data, input) = chain_circuit();
        let prove = |value| {
            let mut pw = PartialWitness::new();
            pw.set_target(input, F::from_canonical_u64(value)).unwrap();
            let proof = data.prove(pw).unwrap();
            data.verify(proof).unwrap();
        };
        prove(0); // Record once before workers share the circuit.
        std::thread::scope(|scope| {
            for worker in 1..=4 {
                let prove = &prove;
                scope.spawn(move || {
                    for value in 0..3 {
                        prove(worker * 10 + value);
                    }
                });
            }
        });
    }

    /// Finishes as soon as `a` is set if `a` is zero, and otherwise only once
    /// `b` is set too, so that the round it finishes in depends on a value.
    #[derive(Debug, Clone)]
    struct ValueDependentGenerator {
        a: Target,
        b: Target,
        out: Target,
    }

    impl WitnessGenerator<F, 2> for ValueDependentGenerator {
        fn id(&self) -> String {
            "ValueDependentGenerator".to_string()
        }

        fn watch_list(&self) -> Vec<Target> {
            vec![self.a, self.b]
        }

        fn run(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) -> bool {
            let Some(a) = witness.try_get_target(self.a) else {
                return false;
            };
            if a.is_zero() {
                out_buffer.set_target(self.out, F::ZERO).unwrap();
                return true;
            }
            match witness.try_get_target(self.b) {
                Some(b) => {
                    out_buffer.set_target(self.out, b).unwrap();
                    true
                }
                None => false,
            }
        }

        fn serialize(
            &self,
            _dst: &mut Vec<u8>,
            _common_data: &CommonCircuitData<F, 2>,
        ) -> IoResult<()> {
            unimplemented!()
        }

        fn deserialize(
            _src: &mut Buffer,
            _common_data: &CommonCircuitData<F, 2>,
        ) -> IoResult<Self> {
            unimplemented!()
        }
    }

    #[test]
    fn schedule_falls_back_when_a_generator_is_not_ready() {
        let mut builder = CircuitBuilder::<F, 2>::new(CircuitConfig::standard_recursion_config());
        let a = builder.add_virtual_target();
        // `b` is produced a round after `a` is known.
        let b = builder.add_const(a, F::ONE);
        let out = builder.add_virtual_target();
        builder.add_generators(vec![WitnessGeneratorRef::new(ValueDependentGenerator {
            a,
            b,
            out,
        })]);
        builder.register_public_input(b);
        let data = builder.build::<C>();
        let witness_for = |value: u64| {
            let mut pw = PartialWitness::new();
            pw.set_target(a, F::from_canonical_u64(value)).unwrap();
            generate_partial_witness(pw, &data.prover_only, &data.common).unwrap()
        };
        // With `a = 0` the generator finishes in the first round.
        let zero = witness_for(0);
        assert_eq!(zero.try_get_target(out), Some(F::ZERO));
        let recorded = data.prover_only.generator_schedule.get().unwrap();
        // With `a = 1` it is scheduled too early: the replay fails, the
        // worklist takes over and records a new schedule.
        let one = witness_for(1);
        assert_eq!(one.try_get_target(out), Some(F::TWO));
        let rerecorded = data.prover_only.generator_schedule.get().unwrap();
        assert_ne!(recorded.as_slice(), rerecorded.as_slice());
        // The new schedule replays for `a = 1`.
        let again = witness_for(1);
        assert_eq!(again.try_get_target(out), Some(F::TWO));
        assert!(std::sync::Arc::ptr_eq(
            &rerecorded,
            &data.prover_only.generator_schedule.get().unwrap()
        ));
    }
}
