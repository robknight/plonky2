//! Compiled typed region program: the whole-circuit witness plan that the CPU
//! executor (`cpu`) runs when a caller installs one (see
//! [`crate::plonk::witness_program`]), its per-circuit slot and its byte
//! encoding. Built with the default `std` and `rand_chacha` features; the
//! `gpu` feature adds wire uploads from a mapped staging buffer. Every layer shares a
//! compact slot state; coalesced output slots replace per-generator host
//! records.
//!
//! Internal: the stable interface is [`crate::plonk::witness_program`].
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{ensure, Result};
use plonky2_maybe_rayon::*;

use crate::field::extension::Extendable;
use crate::hash::hash_types::RichField;
use crate::iop::generator::{WitnessGeneratorRef, WitnessOperation as Op};
use crate::iop::target::Target;
use crate::iop::witness::{PartialWitness, PartitionWitness};

pub(crate) mod aot;
pub(super) mod buffers;
pub mod cpu;
pub(crate) mod plan_bytes;
mod profile;

/// Log target of the compiler's phase timings (debug level).
pub(crate) const COMPILE_LOG: &str = "plonky2::witness_program::compile";

/// No owner or slot yet, in the compiler's dense per-representative arrays.
const NONE: u32 = u32::MAX;

// One declaration defines both the operation IDs and their checked decoder.
macro_rules! operation_kinds {
    ($($kind:ident = $code:literal),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
        #[repr(u8)]
        enum OperationKind { $($kind = $code),+ }
        impl OperationKind {
            const COUNT: usize = [$(Self::$kind),+].len();
            fn code(self) -> usize { self as usize }
            fn from_code(code: usize) -> Result<Self> {
                match code { $($code => Ok(Self::$kind),)+
                    _ => anyhow::bail!("plan has an unknown operation kind {code}"),
                }
            }
        }
    };
}

operation_kinds! {
    Poseidon = 0,
    Arithmetic = 1,
    Copy = 2,
    Constant = 3,
    Equality = 4,
    ExtensionArithmetic = 5,
    ExtensionMultiply = 6,
    RandomAccess = 7,
    BaseSplit = 8,
    WireSplit = 9,
    LowHigh = 10,
    Reducing = 11,
    ReducingExtension = 12,
    TableLookup = 13,
    ExtensionQuotient = 14,
    EcAddXu = 15,
    PoseidonMds = 16,
    RandomValue = 17,
    ConditionalZero = 18,
    LtMask = 19,
    QuinticMultiply = 20,
    QuinticQuotient = 21,
    EcAddHomogOffset = 22,
    EcScalarMul = 23,
    CosetInterpolation = 24,
    Comparison = 25,
    ArithmeticChain = 26,
}
use OperationKind as Kind;

#[derive(Clone, Debug)]
enum GroupArgument {
    Coefficients(u32),
    Parameter(u32),
}

impl GroupArgument {
    fn word(&self) -> u32 {
        match *self {
            Self::Coefficients(word) | Self::Parameter(word) => word,
        }
    }
}

impl OperationKind {
    fn argument(self, word: u32) -> GroupArgument {
        match self {
            Self::RandomAccess
            | Self::BaseSplit
            | Self::WireSplit
            | Self::LowHigh
            | Self::Reducing
            | Self::ReducingExtension
            | Self::LtMask => GroupArgument::Parameter(word),
            _ => GroupArgument::Coefficients(word),
        }
    }
}

#[derive(Clone, Debug)]
struct Group {
    inputs: u32,
    outputs: u32,
    kind: Kind,
    count: u32,
    targets_offset: u32,
    argument: GroupArgument,
}

impl Group {
    fn coefficients_offset(&self) -> Result<u32> {
        match self.argument {
            GroupArgument::Coefficients(offset) => Ok(offset),
            GroupArgument::Parameter(_) => anyhow::bail!("operation has an immediate parameter"),
        }
    }

    fn parameter(&self) -> Result<u32> {
        match self.argument {
            GroupArgument::Parameter(value) => Ok(value),
            GroupArgument::Coefficients(_) => anyhow::bail!("operation has coefficient storage"),
        }
    }
}

/// Immutable compiled topology, reused across executions.
#[derive(Clone, Debug)]
pub struct RegionProgram {
    /// Operation groups in dependency layers: every group in a layer depends
    /// only on earlier layers.
    groups: Vec<Vec<Group>>,
    targets: Arc<[u32]>,
    coefficients: Arc<[u64]>,
    representatives: Arc<[usize]>,
    roots: Arc<[usize]>,
    random_count: u32,
    alias_count: u32,
    alias_pair_offset: u32,
    state_slots: u32,
    pub(super) runtime: Arc<super::config::Runtime>,
    fast: Arc<std::sync::OnceLock<aot::FastLayout>>,
    /// Representative index to canonical slot, `u32::MAX` when absent; built
    /// on first use by the CPU executor's gather.
    rep_slots: Arc<std::sync::OnceLock<Vec<u32>>>,
    /// Canonical slot of every wire cell in column-major order, `u32::MAX`
    /// where the program has no slot; built on first use by the gather.
    cell_slots: Arc<std::sync::OnceLock<Vec<u32>>>,
}

/// The program slot of one prover circuit: the whole-circuit program a caller
/// installed or compiled through it. The initial-target mask is part of the
/// key because it determines boundary roots and writer ownership. Clones
/// share the slot.
#[derive(Clone, Debug)]
pub struct ProgramCache(Arc<Mutex<CacheState>>, Arc<Mutex<()>>);

#[derive(Debug)]
struct CachedProgram {
    initial: Vec<usize>,
    roots_match_inputs: bool,
    generators: usize,
    program: Arc<RegionProgram>,
    /// Diagnostics that compile their own program leave this unset.
    operations: Option<u128>,
    key: Option<Arc<super::region::plan_bytes::PlanKey>>,
}

#[derive(Debug, Default)]
struct CacheState {
    // Only complete circuit programs are installed in this cache.
    program: Option<CachedProgram>,
    warned_mismatch: bool,
    unsupported: Option<super::WitnessProgramError>,
}

/// What a proof's inputs find in the program slot.
pub(crate) enum Installed {
    /// A whole-circuit program for exactly these input representatives.
    Program(Arc<RegionProgram>),
    /// A whole-circuit program for other input representatives.
    OtherInputs,
    /// No whole-circuit program.
    None,
}

impl Default for ProgramCache {
    fn default() -> Self {
        Self(
            Arc::new(Mutex::new(CacheState::default())),
            Arc::new(Mutex::new(())),
        )
    }
}

impl PartialEq for ProgramCache {
    fn eq(&self, _other: &Self) -> bool {
        true // A cached plan is not part of circuit identity.
    }
}
impl Eq for ProgramCache {}

impl ProgramCache {
    pub(crate) fn preparation_guard(&self) -> std::sync::MutexGuard<'_, ()> {
        self.1.lock().unwrap_or_else(|e| e.into_inner())
    }
    pub(crate) fn unsupported(&self) -> Option<super::WitnessProgramError> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .unsupported
            .clone()
    }
    pub(crate) fn record_unsupported(&self, error: &super::WitnessProgramError) {
        if matches!(
            error,
            super::WitnessProgramError::UnsupportedField
                | super::WitnessProgramError::Lookups
                | super::WitnessProgramError::UnsupportedGenerator { .. }
        ) {
            self.0.lock().unwrap_or_else(|e| e.into_inner()).unsupported = Some(error.clone());
            log::info!("compiled witness execution unavailable: {error}");
        }
    }
    pub(crate) fn status(&self, generators: usize) -> super::WitnessProgramStatus {
        let c = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if c.program
            .as_ref()
            .is_some_and(|p| p.generators == generators)
        {
            super::WitnessProgramStatus::Ready
        } else if c.unsupported.is_some() {
            super::WitnessProgramStatus::Unsupported
        } else {
            super::WitnessProgramStatus::Unprepared
        }
    }

    /// Installs a whole-circuit program for proofs whose inputs set exactly
    /// the representatives in `initial`, replacing any program in the slot.
    pub(crate) fn install_all(
        &self,
        generators: usize,
        initial: Vec<usize>,
        program: Arc<RegionProgram>,
        operations: u128,
        key: Arc<super::region::plan_bytes::PlanKey>,
    ) {
        let mut cached = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let roots_match_inputs = initial.as_slice() == program.roots.as_ref();
        cached.program = Some(CachedProgram {
            roots_match_inputs,
            initial,
            generators,
            program,
            operations: Some(operations),
            key: Some(key),
        });
        cached.warned_mismatch = false;
        cached.unsupported = None;
    }

    /// Empties the slot.
    pub(crate) fn clear(&self) {
        let mut cached = self.0.lock().unwrap_or_else(|e| e.into_inner());
        cached.program = None;
        cached.warned_mismatch = false;
        cached.unsupported = None;
    }

    /// Whether the slot holds a whole-circuit program, without computing a
    /// proof's mask.
    pub(crate) fn has_all(&self, generators: usize) -> bool {
        let cached = self.0.lock().unwrap_or_else(|e| e.into_inner());
        cached
            .program
            .as_ref()
            .is_some_and(|program| program.generators == generators)
    }

    /// Candidate program; callers must validate that every supplied input
    /// maps to a root and that every root is supplied before executing it.
    pub(crate) fn candidate_all(&self, generators: usize) -> Option<Arc<RegionProgram>> {
        let cached = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let program = cached.program.as_ref()?;
        (program.generators == generators && program.roots_match_inputs)
            .then(|| program.program.clone())
    }

    pub(crate) fn strict_inputs(&self) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .program
            .as_ref()
            .is_some_and(|p| p.program.runtime.config.fallback == super::WitnessFallback::Error)
    }

    pub(crate) fn installed_program(&self, generators: usize) -> Option<super::WitnessProgram> {
        let cached = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let p = cached.program.as_ref()?;
        (p.generators == generators).then_some(())?;
        Some(super::WitnessProgram {
            program: p.program.clone(),
            key: p.key.as_ref()?.clone(),
        })
    }

    /// The whole-circuit program in the slot, its mask and its operation
    /// digest when known.
    pub(crate) fn installed_all(
        &self,
        generators: usize,
    ) -> Option<(Vec<usize>, Arc<RegionProgram>, Option<u128>)> {
        let cached = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let program = cached.program.as_ref()?;
        (program.generators == generators).then(|| {
            (
                program.initial.clone(),
                program.program.clone(),
                program.operations,
            )
        })
    }

    /// The whole-circuit program for this initial-target mask. A program for
    /// another mask is reported once with a warning; the caller then uses
    /// the worklist, which is correct for any inputs.
    pub(crate) fn lookup_all(&self, generators: usize, initial: &[usize]) -> Installed {
        let mut cached = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let Some(program) = cached.program.as_ref() else {
            return Installed::None;
        };
        if program.generators != generators {
            return Installed::None;
        }
        if program.initial == initial {
            return Installed::Program(program.program.clone());
        }
        let compiled_for = program.initial.len();
        if !cached.warned_mismatch {
            cached.warned_mismatch = true;
            log::warn!(
                "plonky2: the installed witness program was compiled for {compiled_for} input \
                 classes, but this proof sets {}; using the worklist",
                initial.len()
            );
        }
        Installed::OtherInputs
    }

    pub(crate) fn get_or_compile_all<F: RichField + Extendable<D>, const D: usize>(
        &self,
        generators: &[WitnessGeneratorRef<F, D>],
        entry: &PartitionWitness<F>,
        inputs: &PartialWitness<F>,
    ) -> Result<(Arc<RegionProgram>, bool)> {
        let mut initial = inputs
            .target_values
            .keys()
            .map(|t| entry.representative_map[t.index(entry.num_wires, entry.degree)])
            .collect::<Vec<_>>();
        initial.sort_unstable();
        initial.dedup();
        let mut cached = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(program) = cached.program.as_ref() {
            if program.initial == initial && program.generators == generators.len() {
                return Ok((program.program.clone(), true));
            }
        }
        let indices = (0..generators.len())
            .map(u32::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        let program = Arc::new(RegionProgram::compile(generators, entry, &indices)?);
        let roots_match_inputs = initial.as_slice() == program.roots.as_ref();
        cached.program = Some(CachedProgram {
            roots_match_inputs,
            initial,
            generators: generators.len(),
            program: program.clone(),
            operations: None,
            key: None,
        });
        cached.warned_mismatch = false;
        Ok((program, false))
    }
}

/// What the layout needs from one generator's operation, without keeping
/// the operation, keeping compiler memory use bounded for large circuits.
struct Lane {
    index: u32,
    /// Executor kind number.
    kind: Kind,
    /// Shape parameter, part of the grouping key.
    parameter: u32,
    /// Operation name, for errors.
    name: &'static str,
    /// This lane's value at each position of its group's coefficient block.
    coefficients: Vec<u64>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
}

impl Lane {
    fn new(index: u32, op: &Op, inputs: Vec<usize>, outputs: Vec<usize>) -> Self {
        let kind = match op {
            Op::Poseidon { .. } => Kind::Poseidon,
            Op::Arithmetic { .. } => Kind::Arithmetic,
            Op::Copy { .. } => Kind::Copy,
            Op::Constant { .. } => Kind::Constant,
            Op::Equality { .. } => Kind::Equality,
            Op::ExtensionArithmetic { .. } => Kind::ExtensionArithmetic,
            Op::ExtensionMultiply { .. } => Kind::ExtensionMultiply,
            Op::RandomAccess { .. } => Kind::RandomAccess,
            Op::BaseSplit { .. } => Kind::BaseSplit,
            Op::WireSplit { .. } => Kind::WireSplit,
            Op::LowHigh { .. } => Kind::LowHigh,
            Op::Reducing {
                extension_coeffs: false,
                ..
            } => Kind::Reducing,
            Op::Reducing {
                extension_coeffs: true,
                ..
            } => Kind::ReducingExtension,
            Op::TableLookup { .. } => Kind::TableLookup,
            Op::ExtensionQuotient { .. } => Kind::ExtensionQuotient,
            Op::EcAddXu { .. } => Kind::EcAddXu,
            Op::PoseidonMds { .. } => Kind::PoseidonMds,
            Op::RandomValue { .. } => Kind::RandomValue,
            Op::ConditionalZero { .. } => Kind::ConditionalZero,
            Op::LtMask { .. } => Kind::LtMask,
            Op::QuinticMultiply { .. } => Kind::QuinticMultiply,
            Op::QuinticQuotient { .. } => Kind::QuinticQuotient,
            Op::EcAddHomogOffset { .. } => Kind::EcAddHomogOffset,
            Op::EcScalarMul { .. } => Kind::EcScalarMul,
            Op::CosetInterpolation { .. } => Kind::CosetInterpolation,
            Op::Comparison { .. } => Kind::Comparison,
        };
        let parameter = match op {
            Op::BaseSplit { limb_bits, .. } => *limb_bits,
            Op::WireSplit { num_limbs, .. } => *num_limbs as u32,
            Op::LowHigh { low_bits, .. } => *low_bits,
            Op::Reducing { num_coeffs, .. } => *num_coeffs as u32,
            Op::TableLookup { entries, .. } => entries.len() as u32,
            Op::EcScalarMul { scalar_words, .. } => scalar_words.len() as u32,
            Op::CosetInterpolation { degree, .. } => *degree as u32,
            _ => 0,
        };
        // Random values take their index from a counter in layout order.
        let coefficients = match op {
            Op::Arithmetic { coefficients, .. } | Op::ExtensionArithmetic { coefficients, .. } => {
                coefficients.to_vec()
            }
            Op::Constant { value, .. } => vec![*value],
            Op::ExtensionMultiply { coefficient, .. } => vec![*coefficient],
            // Each lane carries its own table-row boundaries. Entry lengths
            // may differ even when flattened input sizes match.
            Op::TableLookup { entries, .. } => {
                let mut values = vec![entries.len() as u64, outputs.len() as u64];
                let mut start = 0;
                values.push(0);
                for entry in entries {
                    start += entry.len();
                    values.push(start as u64);
                }
                values
            }
            Op::EcAddXu {
                curve_b1,
                generator_x,
                generator_u,
                ..
            } => [*curve_b1]
                .into_iter()
                .chain(generator_x.iter().copied())
                .chain(generator_u.iter().copied())
                .collect(),
            Op::EcAddHomogOffset { curve_b1, .. } => vec![*curve_b1],
            Op::EcScalarMul {
                curve_b1,
                scalar_words,
                ..
            } => [scalar_words.len() as u64, *curve_b1]
                .into_iter()
                .chain(scalar_words.iter().map(|&w| w as u64))
                .collect(),
            Op::CosetInterpolation {
                degree,
                domain,
                weights,
                ..
            } => (0..inputs.len().saturating_sub(1))
                .map(|pos| match pos {
                    0 => *degree as u64,
                    1 => domain.len() as u64,
                    _ if pos < 2 + domain.len() => domain[pos - 2],
                    _ => weights[pos - 2 - domain.len()],
                })
                .collect(),
            Op::Comparison {
                num_chunks,
                chunk_bits,
                ..
            } => vec![*num_chunks as u64, *chunk_bits as u64],
            _ => Vec::new(),
        };
        Self {
            index,
            kind,
            parameter,
            name: op.kind(),
            coefficients,
            inputs,
            outputs,
        }
    }
}

/// Fuses only single-consumer arithmetic chains. Intermediate outputs remain
/// materialized for the proof, but their next arithmetic consumer uses a local
/// temporary. Choosing only single-consumer edges makes contraction acyclic.
fn fuse_arithmetic<F: RichField>(
    nodes: Vec<Lane>,
    entry: &PartitionWitness<F>,
    config: &super::WitnessConfig,
) -> Result<Vec<Lane>> {
    if config.chain_length == 0 {
        return Ok(nodes);
    }
    let limit = config.chain_length as usize;
    const MANY: u32 = u32::MAX - 1;
    let mut producer = vec![NONE; entry.values.len()];
    let mut consumer = vec![NONE; entry.values.len()];
    for (i, lane) in nodes.iter().enumerate() {
        for &r in &lane.outputs {
            producer[r] = if producer[r] == NONE { i as u32 } else { MANY };
        }
        for &r in &lane.inputs {
            if consumer[r] == NONE {
                consumer[r] = i as u32;
            } else if consumer[r] != i as u32 {
                consumer[r] = MANY;
            }
        }
    }
    let mut next = vec![NONE; nodes.len()];
    let mut previous = vec![NONE; nodes.len()];
    for (i, lane) in nodes.iter().enumerate() {
        if lane.kind != Kind::Arithmetic || lane.outputs.len() != 1 {
            continue;
        }
        let r = lane.outputs[0];
        let j = consumer[r];
        if entry.values[r].is_some() || producer[r] != i as u32 || j >= MANY || j == i as u32 {
            continue;
        }
        if nodes[j as usize].kind != Kind::Arithmetic {
            continue;
        }
        if previous[j as usize] == NONE {
            previous[j as usize] = i as u32;
            next[i] = j;
        }
    }
    drop(producer);
    drop(consumer);
    let heads = (0..nodes.len())
        .filter(|&i| previous[i] == NONE)
        .collect::<Vec<_>>();
    let before = nodes.len();
    let mut remaining = nodes.into_iter().map(Some).collect::<Vec<_>>();
    let mut result = Vec::with_capacity(before);
    let mut chain_count = 0;
    let mut fused_nodes = 0;
    for head in heads {
        let mut current = head as u32;
        while current != NONE && remaining[current as usize].is_some() {
            let mut chain = Vec::with_capacity(limit);
            while current != NONE && remaining[current as usize].is_some() && chain.len() < limit {
                chain.push(remaining[current as usize].take().unwrap());
                current = next[current as usize];
            }
            if chain.len() == 1 {
                result.push(chain.pop().unwrap());
                continue;
            }
            chain_count += 1;
            fused_nodes += chain.len();
            let index = chain[0].index;
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();
            let mut coefficients = Vec::new();
            for lane in chain {
                ensure!(
                    lane.inputs.len() == 3 && lane.coefficients.len() == 2,
                    "invalid arithmetic shape"
                );
                for r in lane.inputs {
                    if let Some(j) = outputs.iter().position(|&out| out == r) {
                        coefficients.push((1u64 << 63) | j as u64);
                    } else {
                        let position =
                            inputs
                                .iter()
                                .position(|&input| input == r)
                                .unwrap_or_else(|| {
                                    let position = inputs.len();
                                    inputs.push(r);
                                    position
                                });
                        coefficients.push(position as u64);
                    }
                }
                coefficients.extend(lane.coefficients);
                outputs.push(lane.outputs[0]);
            }
            result.push(Lane {
                index,
                kind: Kind::ArithmeticChain,
                parameter: outputs.len() as u32,
                name: "arithmetic_chain",
                coefficients,
                inputs,
                outputs,
            });
        }
    }
    // Any cyclic component remains intact for the existing cycle diagnostic.
    result.extend(remaining.into_iter().flatten());
    log::debug!(target:COMPILE_LOG,"arithmetic fusion: {before} -> {}, chains={chain_count}, operations={fused_nodes}, limit={limit}", result.len());
    Ok(result)
}

impl RegionProgram {
    /// Number of dependency layers.
    pub fn layer_count(&self) -> usize {
        self.groups.len()
    }

    /// Compile explicitly selected standard generators. External dependencies
    /// are boundary roots; their values must be supplied at execution time.
    /// Additional writers receive separate slots, which the executor checks
    /// against the canonical value.
    /// Cycles are rejected.
    /// Caller supplies the witness at region entry, not a completed reference.
    pub fn compile<F: RichField + Extendable<D>, const D: usize>(
        generators: &[WitnessGeneratorRef<F, D>],
        entry: &PartitionWitness<F>,
        indices: &[u32],
    ) -> Result<Self> {
        let runtime = Arc::new(super::config::Runtime::new(
            super::WitnessConfig::from_env()?,
        )?);
        Self::compile_with_runtime(generators, entry, indices, runtime)
    }
    pub(super) fn compile_with_runtime<F: RichField + Extendable<D>, const D: usize>(
        generators: &[WitnessGeneratorRef<F, D>],
        entry: &PartitionWitness<F>,
        indices: &[u32],
        runtime: Arc<super::config::Runtime>,
    ) -> Result<Self> {
        ensure!(Op::field_supported::<F>(), "unsupported witness field");
        // Phase timings at debug level under COMPILE_LOG.
        let profile = log::log_enabled!(target: COMPILE_LOG, log::Level::Debug);
        let mut phase = Instant::now();
        let rep = |t: Target| -> Result<usize> {
            entry
                .representative_map
                .get(t.index(entry.num_wires, entry.degree))
                .copied()
                .ok_or_else(|| anyhow::anyhow!("invalid descriptor target"))
        };
        // Representatives index dense arrays below in place of hash maps.
        let reps = entry.values.len();
        let mut seen = vec![false; generators.len()];
        for &index in indices {
            let selected = seen
                .get_mut(index as usize)
                .ok_or_else(|| anyhow::anyhow!("invalid generator index"))?;
            ensure!(!*selected, "duplicate generator selection");
            *selected = true;
        }
        drop(seen);
        // Each generator's operation and representatives, independently.
        let nodes = indices
            .par_iter()
            .map(|&index| {
                let g = &generators[index as usize];
                let op =
                    g.0.witness_operation()
                        .ok_or_else(|| anyhow::anyhow!("unsupported generator {}", g.0.id()))?;
                let input = op
                    .inputs()
                    .into_iter()
                    .map(rep)
                    .collect::<Result<Vec<_>>>()?;
                let output = op
                    .outputs()
                    .into_iter()
                    .map(rep)
                    .collect::<Result<Vec<_>>>()?;
                Ok(Lane::new(index, &op, input, output))
            })
            .collect::<Result<Vec<_>>>()?;
        let nodes = fuse_arithmetic(nodes, entry, &runtime.config)?;
        let mut produced = vec![false; reps];
        for lane in &nodes {
            for &r in &lane.outputs {
                produced[r] = true;
            }
        }
        if profile {
            log::debug!(target: COMPILE_LOG,"compile phase nodes {:?}", phase.elapsed());
            phase = Instant::now();
        }
        // Select each representative's first *ready* producer. Circuit
        // connections can make a generator read and redundantly write the
        // same representative, so the first generator in source order need
        // not be a valid owner. Availability depends on representatives,
        // while every later writer becomes an independently checked alias.
        ensure!(
            nodes.len() < NONE as usize,
            "region exceeds the u32 index range"
        );
        // Each representative's waiting generators, in generator order, as
        // one flat list indexed by `watch_start`.
        let mut unavailable = vec![0u32; nodes.len()];
        let mut watch_start = vec![0u32; reps + 1];
        // A lane stamp deduplicates dependencies in linear time, without
        // scanning each operand's prefix or changing operand order.
        let mut seen_input = vec![NONE; reps];
        let waits = |r: usize, lane: u32, seen: &mut [u32]| {
            let first = seen[r] != lane;
            seen[r] = lane;
            first && entry.values[r].is_none() && produced[r]
        };
        let mut watches = 0usize;
        for (i, lane) in nodes.iter().enumerate() {
            let inputs = &lane.inputs;
            for position in 0..inputs.len() {
                if waits(inputs[position], i as u32, &mut seen_input) {
                    unavailable[i] += 1;
                    watch_start[inputs[position]] += 1;
                    watches += 1;
                }
            }
        }
        ensure!(
            watches < NONE as usize,
            "region exceeds the u32 index range"
        );
        // Running totals give each list's end; filling backwards from the
        // ends leaves each list in generator order and `watch_start[r]` at
        // its start.
        for r in 1..=reps {
            watch_start[r] += watch_start[r - 1];
        }
        let mut watchers = vec![0u32; watches];
        seen_input.fill(NONE);
        for (i, lane) in nodes.iter().enumerate().rev() {
            let inputs = &lane.inputs;
            for position in (0..inputs.len()).rev() {
                if waits(inputs[position], i as u32, &mut seen_input) {
                    let r = inputs[position];
                    watch_start[r] -= 1;
                    watchers[watch_start[r] as usize] = i as u32;
                }
            }
        }
        drop(seen_input);
        let mut owners = vec![NONE; reps];
        let mut ready_owners = unavailable
            .iter()
            .enumerate()
            .filter_map(|(i, &missing)| (missing == 0).then_some(i))
            .collect::<VecDeque<_>>();
        let mut scheduled = 0usize;
        while let Some(i) = ready_owners.pop_front() {
            scheduled += 1;
            for &r in &nodes[i].outputs {
                if entry.values[r].is_some() || owners[r] != NONE {
                    continue;
                }
                owners[r] = i as u32;
                for &next in &watchers[watch_start[r] as usize..watch_start[r + 1] as usize] {
                    let next = next as usize;
                    unavailable[next] -= 1;
                    if unavailable[next] == 0 {
                        ready_owners.push_back(next);
                    }
                }
            }
        }
        if scheduled != nodes.len() {
            let blocked = unavailable
                .iter()
                .enumerate()
                .filter(|(_, count)| **count > 0)
                .take(8)
                .map(|(i, &count)| {
                    (
                        nodes[i].index,
                        nodes[i].name,
                        count,
                        &nodes[i].inputs,
                        &nodes[i].outputs,
                    )
                })
                .collect::<Vec<_>>();
            anyhow::bail!(
                "typed witness graph has {} unready generators: {blocked:?}",
                nodes.len() - scheduled
            );
        }
        if profile {
            log::debug!(target: COMPILE_LOG,"compile phase ready_owners {:?}", phase.elapsed());
            phase = Instant::now();
        }
        let mut is_root = vec![false; reps];
        if indices.len() == generators.len() {
            // A complete witness must preserve supplied targets even when no
            // generator watches them (e.g. an unconstrained public input).
            for (r, value) in entry.values.iter().enumerate() {
                is_root[r] |= value.is_some();
            }
        }
        // Each generator's distinct producers in ascending order, and each
        // producer's dependants in generator order, as flat lists.
        let mut dependency_start = Vec::with_capacity(nodes.len() + 1);
        dependency_start.push(0usize);
        let mut dependencies = Vec::<u32>::new();
        let mut successor_start = vec![0usize; nodes.len() + 1];
        for lane in &nodes {
            let first = dependencies.len();
            for &r in &lane.inputs {
                if entry.values[r].is_some() || owners[r] == NONE {
                    is_root[r] = true;
                } else {
                    dependencies.push(owners[r]);
                }
            }
            for &r in &lane.outputs {
                is_root[r] |= entry.values[r].is_some();
            }
            dependencies[first..].sort_unstable();
            let mut kept = first;
            for k in first..dependencies.len() {
                if k == first || dependencies[k] != dependencies[kept - 1] {
                    dependencies[kept] = dependencies[k];
                    kept += 1;
                }
            }
            dependencies.truncate(kept);
            for &p in &dependencies[first..] {
                successor_start[p as usize + 1] += 1;
            }
            dependency_start.push(dependencies.len());
        }
        for i in 0..nodes.len() {
            successor_start[i + 1] += successor_start[i];
        }
        let mut successors = vec![0u32; successor_start[nodes.len()]];
        let mut fill = successor_start.clone();
        for i in 0..nodes.len() {
            for &p in &dependencies[dependency_start[i]..dependency_start[i + 1]] {
                successors[fill[p as usize]] = i as u32;
                fill[p as usize] += 1;
            }
        }
        drop(fill);
        let mut remaining = (0..nodes.len())
            .map(|i| dependency_start[i + 1] - dependency_start[i])
            .collect::<Vec<_>>();
        let mut ready = remaining
            .iter()
            .enumerate()
            .filter_map(|(i, &n)| (n == 0).then_some(i))
            .collect::<VecDeque<_>>();
        let mut depth = vec![0usize; nodes.len()];
        let mut ordered = Vec::with_capacity(nodes.len());
        while let Some(i) = ready.pop_front() {
            ordered.push(i);
            for &next in &successors[successor_start[i]..successor_start[i + 1]] {
                let next = next as usize;
                depth[next] = depth[next].max(depth[i] + 1);
                remaining[next] -= 1;
                if remaining[next] == 0 {
                    ready.push_back(next);
                }
            }
        }
        if ordered.len() != nodes.len() {
            let blocked = remaining
                .iter()
                .enumerate()
                .filter(|(_, count)| **count > 0)
                .take(8)
                .map(|(i, _)| {
                    (
                        nodes[i].index,
                        nodes[i].name,
                        &nodes[i].inputs,
                        &nodes[i].outputs,
                        dependencies[dependency_start[i]..dependency_start[i + 1]]
                            .iter()
                            .map(|&p| nodes[p as usize].index)
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>();
            anyhow::bail!("cyclic region or missing self dependency: {blocked:?}");
        }
        if profile {
            log::debug!(
                target: COMPILE_LOG,
                "compile phase dependencies_and_depth {:?}",
                phase.elapsed()
            );
            phase = Instant::now();
        }
        let roots = (0..reps).filter(|&r| is_root[r]).collect::<Vec<_>>();
        let mut representatives = roots.clone();
        let mut slots = vec![NONE; reps];
        for (i, &r) in roots.iter().enumerate() {
            slots[r] = i as u32;
        }
        let mut targets = Vec::new();
        let mut alias_outputs = Vec::<(usize, usize, usize)>::new();
        let mut coefficients = Vec::new();
        let mut random_count = 0u32;
        let mut groups = Vec::new();
        let max_depth = depth.iter().copied().max().map_or(0, |d| d + 1);
        // Each layer's generators in topological order.
        let mut by_depth = vec![Vec::new(); max_depth];
        for &i in &ordered {
            by_depth[depth[i]].push(i);
        }
        for selected in by_depth {
            let mut by_kind = BTreeMap::<(Kind, usize, usize, u32), Vec<usize>>::new();
            for &i in &selected {
                let lane = &nodes[i];
                by_kind
                    .entry((
                        lane.kind,
                        lane.inputs.len(),
                        lane.outputs.len(),
                        lane.parameter,
                    ))
                    .or_default()
                    .push(i);
            }
            let mut layer_groups = Vec::new();
            for ((kind, input_len, output_len, parameter), indices) in by_kind {
                let count = indices.len();
                let target_offset = targets.len();
                for position in 0..input_len {
                    for &i in &indices {
                        let slot = slots[nodes[i].inputs[position]];
                        ensure!(slot != NONE, "input read before its slot exists");
                        targets.push(slot);
                    }
                }
                // Allocate output position across lanes first, preserving coalesced stores.
                for position in 0..output_len {
                    for &i in &indices {
                        let r = nodes[i].outputs[position];
                        if slots[r] == NONE {
                            slots[r] = representatives.len() as u32;
                            representatives.push(r);
                        }
                        let slot = slots[r];
                        if entry.values[r].is_none() && owners[r] != i as u32 {
                            alias_outputs.push((targets.len(), i, r));
                        }
                        targets.push(slot);
                    }
                }
                // Immediate shape parameters are distinct from coefficient offsets.
                let coefficient_offset = match kind {
                    Kind::RandomAccess => input_len - 1,
                    Kind::BaseSplit => (output_len << 8) | parameter as usize,
                    Kind::WireSplit => (output_len << 8) | parameter as usize,
                    Kind::LowHigh => parameter as usize,
                    Kind::Reducing | Kind::ReducingExtension => parameter as usize,
                    Kind::LtMask => output_len,
                    _ => coefficients.len(),
                };
                if kind == Kind::RandomValue {
                    for _ in &indices {
                        coefficients.push(random_count as u64);
                        random_count = random_count
                            .checked_add(1)
                            .ok_or_else(|| anyhow::anyhow!("too many random witness values"))?;
                    }
                } else {
                    let width = nodes[indices[0]].coefficients.len();
                    for pos in 0..width {
                        for &i in &indices {
                            let lane = &nodes[i].coefficients;
                            ensure!(lane.len() == width, "lanes of one group differ in shape");
                            coefficients.push(lane[pos]);
                        }
                    }
                }
                layer_groups.push(Group {
                    inputs: input_len as u32,
                    outputs: output_len as u32,
                    kind,
                    count: u32::try_from(count)?,
                    targets_offset: u32::try_from(target_offset)?,
                    argument: kind.argument(u32::try_from(coefficient_offset)?),
                });
            }
            groups.push(layer_groups);
        }
        if profile {
            log::debug!(
                target: COMPILE_LOG,
                "compile phase layers_and_layout {:?}",
                phase.elapsed()
            );
            phase = Instant::now();
        }
        // Canonical slots are stable for all consumers. Each additional
        // generator writes an independent slot, even if it runs in parallel
        // with the owner. Compare every alias after executing all layers.
        let mut alias_slots = HashMap::<(usize, usize), u32>::new();
        let mut alias_pairs = Vec::<(u32, u32)>::new();
        for (target_position, generator, representative) in alias_outputs {
            let alias = if let Some(&slot) = alias_slots.get(&(generator, representative)) {
                slot
            } else {
                let slot = u32::try_from(representatives.len() + alias_pairs.len())?;
                alias_slots.insert((generator, representative), slot);
                alias_pairs.push((slots[representative], slot));
                slot
            };
            targets[target_position] = alias;
        }
        let state_slots = u32::try_from(representatives.len() + alias_pairs.len())?;
        let alias_pair_offset = u32::try_from(targets.len())?;
        for &(canonical, alias) in &alias_pairs {
            targets.extend([canonical, alias]);
        }
        let alias_count = u32::try_from(alias_pairs.len())?;
        ensure!(
            targets.len() <= u32::MAX as usize,
            "region exceeds the u32 index range"
        );
        if profile {
            log::debug!(target: COMPILE_LOG,"compile phase aliases {:?}", phase.elapsed());
        }
        let mut program = Self {
            groups,
            targets: targets.into(),
            coefficients: coefficients.into(),
            representatives: representatives.into(),
            roots: roots.into(),
            random_count,
            alias_count,
            alias_pair_offset,
            state_slots,
            runtime,
            fast: Default::default(),
            rep_slots: Default::default(),
            cell_slots: Default::default(),
        };
        aot::reorder(&mut program)?;
        Ok(program)
    }
}

#[cfg(feature = "witness-benchmarks")]
pub(super) fn benchmark_profile(active: bool) {
    profile::benchmark_override(active);
}
