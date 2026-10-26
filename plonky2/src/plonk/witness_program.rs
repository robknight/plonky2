//! Compiled witness programs: an explicit, caller-managed optimisation of
//! witness generation.
//!
//! By default a proof generates its witness with the worklist, which runs
//! every generator and needs no preparation. A caller that proves the same
//! circuit many times can instead compile the circuit's generators once into
//! a [`WitnessProgram`] and install it; later proofs whose inputs set the
//! same input classes then run the prepared executor. Performance depends on
//! circuit structure, hardware and the rest of the proving pipeline. Measure
//! preparation, first proof and steady-state witness/proof time separately;
//! [`WitnessPreparation`] reports the preparation breakdown.
//!
//! Available with the default `std` and `rand_chacha` features, with or
//! without `gpu`; with `gpu` the executor gathers wires into a mapped staging
//! buffer and uploads them when the GPU prover applies.
//!
//! Preparation is explicit: [`CircuitData::prove`] never compiles or loads a
//! program. [`CircuitData::prepare_witness_program`] loads a compatible cached
//! executor or compiles one, prepares its host tables, and installs it. Set
//! `PLONKY2_EXECUTOR_CACHE_DIR` to enable persistent executor caching; without
//! it preparation is in memory. Cache read/write failure is reported in
//! [`WitnessPreparation`] and does not prevent in-memory execution.
//!
//! Advanced callers can separately compile, encode, decode and install a
//! [`WitnessProgram`]. Stored programs are trusted input like serialized
//! circuit data: their checksum detects damage, not deliberate tampering.
//!
//! A program applies to proofs whose
//! [`PartialWitness`](crate::iop::witness::PartialWitness) sets targets in
//! exactly the copy-constraint classes it was compiled for; other proofs use
//! the worklist by default, and the first such proof logs a warning.
//! [`WitnessFallback::Error`] requires compiled execution instead. A circuit with
//! lookup tables, or with a generator that does not describe its operation
//! (see [`WitnessGenerator::witness_operation`]), cannot be compiled; check with
//! [`CircuitData::witness_program_support`].
//!
//! [`CircuitData::prepare_witness_program`]: crate::plonk::circuit_data::CircuitData::prepare_witness_program
//! [`CircuitData::prove`]: crate::plonk::circuit_data::CircuitData::prove
//! [`CircuitData::compile_witness_program`]: crate::plonk::circuit_data::CircuitData::compile_witness_program
//! [`CircuitData::install_witness_program`]: crate::plonk::circuit_data::CircuitData::install_witness_program
//! [`CircuitData::witness_program_from_bytes`]: crate::plonk::circuit_data::CircuitData::witness_program_from_bytes
//! [`CircuitData::witness_program_support`]: crate::plonk::circuit_data::CircuitData::witness_program_support
//! [`WitnessGenerator::witness_operation`]: crate::iop::generator::WitnessGenerator::witness_operation

mod cache;
pub use cache::{WitnessPreparation, WitnessProgramSource};
mod config;
pub use config::{WitnessBuffers, WitnessCache, WitnessConfig, WitnessFallback};
mod policy;

#[doc(hidden)]
pub(crate) mod region;

use std::fmt;
use std::sync::Arc;

use self::region::plan_bytes::{self, PlanKey};
use self::region::{RegionProgram, COMPILE_LOG};
use crate::field::extension::Extendable;
use crate::hash::hash_types::RichField;
use crate::iop::generator::WitnessOperation;
use crate::iop::target::Target;
use crate::iop::witness::{PartitionWitness, WitnessWrite};
use crate::plonk::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
use crate::plonk::config::{GenericConfig, GenericHashOut};

/// Fingerprint of this build's witness compiler, executor and encoding.
/// [`CircuitData::witness_program_from_bytes`] rejects programs from builds
/// with another value; callers that store programs can include it in their
/// cache keys, so a changed executor gets a fresh entry instead of a rejected
/// one.
///
/// [`CircuitData::witness_program_from_bytes`]: crate::plonk::circuit_data::CircuitData::witness_program_from_bytes
pub const FORMAT: u64 = plan_bytes::WITNESS_PROGRAM_FORMAT;

/// Cheap preparation state for integrations that prepare on first use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum WitnessProgramStatus {
    /// No preparation has completed.
    Unprepared,
    /// A program is installed. Individual input shapes are still checked.
    Ready,
    /// This circuit has an unsupported field, lookup table or generator.
    /// Removing the program clears this memoized result if a circuit is changed.
    Unsupported,
}

/// Validation belongs to an immutable circuit and its current preparation epoch.
/// A weak handle avoids keeping circuit resources alive through program handles.
#[derive(Clone)]
pub(crate) struct ProgramOrigin {
    slot: std::sync::Weak<std::sync::Mutex<()>>,
    generation: u64,
}

/// A compiled witness program for one circuit and one set of input targets.
/// Cheap to clone; clones share the compiled program.
#[derive(Clone)]
pub struct WitnessProgram {
    origin: ProgramOrigin,
    program: Arc<RegionProgram>,
    key: Arc<PlanKey>,
}

impl fmt::Debug for WitnessProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WitnessProgram")
            .field("generators", &self.key.generators)
            .field("input_classes", &self.key.mask.len())
            .field("layers", &self.program.layer_count())
            .finish()
    }
}

/// Why a witness program could not be compiled, decoded or installed.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum WitnessProgramError {
    /// The executor runs over the Goldilocks field only.
    UnsupportedField,
    /// A supplied configuration or environment override is invalid.
    InvalidConfiguration(String),
    /// The circuit uses lookup tables, which the executor does not support.
    Lookups,
    /// A generator does not describe its operation; the first one found.
    UnsupportedGenerator { index: usize, id: String },
    /// An input target lies outside the circuit.
    InputOutsideCircuit(Target),
    /// The program was compiled for another circuit.
    WrongCircuit,
    /// The program was compiled by another version of the witness compiler,
    /// executor or encoding.
    WrongFormat,
    /// The circuit's generators now describe different operations than when
    /// the program was compiled (a generator's code changed).
    ChangedOperations,
    /// The compiler rejected the circuit, or the bytes are not a valid
    /// program.
    Invalid(String),
}

impl fmt::Display for WitnessProgramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(reason) => {
                write!(f, "invalid witness configuration: {reason}")
            }
            Self::UnsupportedField => write!(f, "the witness executor supports Goldilocks only"),
            Self::Lookups => write!(f, "the circuit uses lookup tables"),
            Self::UnsupportedGenerator { index, id } => {
                write!(
                    f,
                    "generator {index} ({id}) does not describe its operation"
                )
            }
            Self::InputOutsideCircuit(target) => {
                write!(f, "input target {target:?} is outside the circuit")
            }
            Self::WrongCircuit => write!(f, "the witness program is for another circuit"),
            Self::WrongFormat => {
                write!(
                    f,
                    "the witness program is from another version of the executor"
                )
            }
            Self::ChangedOperations => write!(
                f,
                "the circuit's generators changed since the witness program was compiled"
            ),
            Self::Invalid(reason) => write!(f, "invalid witness program: {reason}"),
        }
    }
}

impl std::error::Error for WitnessProgramError {}

/// Checks that every generator of the circuit describes its operation, without
/// compiling. Cheap: one pass over the generators.
pub(crate) fn support<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    prover: &ProverOnlyCircuitData<F, C, D>,
    common: &CommonCircuitData<F, D>,
) -> Result<(), WitnessProgramError> {
    if !WitnessOperation::field_supported::<F>() {
        return Err(WitnessProgramError::UnsupportedField);
    }
    if !common.luts.is_empty() || common.num_lookup_polys != 0 {
        return Err(WitnessProgramError::Lookups);
    }
    match prover
        .generators
        .iter()
        .position(|g| g.0.witness_operation().is_none())
    {
        Some(index) => Err(WitnessProgramError::UnsupportedGenerator {
            index,
            id: prover.generators[index].0.id(),
        }),
        None => Ok(()),
    }
}

/// The key a program for these input representatives and operations carries.
fn key<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    prover: &ProverOnlyCircuitData<F, C, D>,
    mask: Vec<usize>,
    operations: u128,
) -> PlanKey {
    PlanKey {
        digest: prover.circuit_digest.to_bytes(),
        generators: prover.generators.len(),
        map_len: prover.representative_map.len(),
        mask,
        format: plan_bytes::WITNESS_PROGRAM_FORMAT,
        operations,
        compilation: 0,
    }
}

/// Whether `key` names this circuit.
fn same_circuit<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    key: &PlanKey,
    prover: &ProverOnlyCircuitData<F, C, D>,
) -> bool {
    key.digest == prover.circuit_digest.to_bytes()
        && key.generators == prover.generators.len()
        && key.map_len == prover.representative_map.len()
}

/// Compiles the circuit for proofs whose inputs set exactly `input_targets`
/// (their values do not matter; duplicates and targets in the same
/// copy-constraint class are allowed).
pub(crate) fn compile<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    prover: &ProverOnlyCircuitData<F, C, D>,
    common: &CommonCircuitData<F, D>,
    input_targets: &[Target],
) -> Result<WitnessProgram, WitnessProgramError> {
    compile_with_config(prover, common, input_targets, WitnessConfig::from_env()?)
}

pub(crate) fn compile_with_config<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    prover: &ProverOnlyCircuitData<F, C, D>,
    common: &CommonCircuitData<F, D>,
    input_targets: &[Target],
    config: WitnessConfig,
) -> Result<WitnessProgram, WitnessProgramError> {
    compile_with_identity(prover, common, input_targets, config, None)
}
fn compile_with_identity<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    prover: &ProverOnlyCircuitData<F, C, D>,
    common: &CommonCircuitData<F, D>,
    input_targets: &[Target],
    config: WitnessConfig,
    operations: Option<u128>,
) -> Result<WitnessProgram, WitnessProgramError> {
    let runtime = Arc::new(config::Runtime::new(config)?);
    let started = std::time::Instant::now();
    support(prover, common)?;
    log::debug!(target: COMPILE_LOG, "compile phase support {:?}", started.elapsed());
    let started = std::time::Instant::now();
    let num_wires = common.config.num_wires;
    let degree = common.degree();
    let map = &prover.representative_map;
    // Compilation reads which classes are set, never their values.
    let mut entry = PartitionWitness::new(num_wires, degree, map);
    for &target in input_targets {
        if !target
            .checked_index(num_wires, degree)
            .is_some_and(|i| i < map.len())
        {
            return Err(WitnessProgramError::InputOutsideCircuit(target));
        }
        entry
            .set_target(target, F::ZERO)
            .map_err(|e| WitnessProgramError::Invalid(e.to_string()))?;
    }
    let mask = entry
        .values
        .iter()
        .enumerate()
        .filter_map(|(r, value)| value.is_some().then_some(r))
        .collect::<Vec<_>>();
    let indices = (0..prover.generators.len() as u32).collect::<Vec<_>>();
    log::debug!(target: COMPILE_LOG, "compile phase entry {:?}", started.elapsed());
    let program =
        RegionProgram::compile_with_runtime(&prover.generators, &entry, &indices, runtime.clone())
            .map_err(|e| WitnessProgramError::Invalid(format!("{e:#}")))?;
    let started = std::time::Instant::now();
    let operations = operations.unwrap_or_else(|| plan_bytes::operation_digest(&prover.generators));
    log::debug!(target: COMPILE_LOG, "compile phase operation_digest {:?}", started.elapsed());
    let mut key = key(prover, mask, operations);
    key.compilation = runtime.config.compilation_key();
    #[cfg(feature = "witness-benchmarks")]
    if let Some(directory) = std::env::var_os("P2_AOT_EXPORT_DIR") {
        std::fs::create_dir_all(&directory)
            .and_then(|_| {
                std::fs::write(
                    std::path::Path::new(&directory).join(format!(
                        "{}.p2w",
                        region::aot::file_stem(&prover.circuit_digest.to_bytes())
                    )),
                    plan_bytes::encode(&program, &key),
                )
            })
            .map_err(|e| WitnessProgramError::Invalid(format!("export: {e:#}")))?;
    }
    Ok(WitnessProgram {
        origin: prover.witness_program.origin(),
        program: Arc::new(program),
        key: Arc::new(key),
    })
}

/// Installs `program` on the circuit, replacing any installed program.
pub(crate) fn install<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    prover: &ProverOnlyCircuitData<F, C, D>,
    common: &CommonCircuitData<F, D>,
    program: WitnessProgram,
) -> Result<(), WitnessProgramError> {
    let _preparation = prover.witness_program.preparation_guard();
    install_inner(prover, common, program)
}

fn install_inner<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    prover: &ProverOnlyCircuitData<F, C, D>,
    common: &CommonCircuitData<F, D>,
    program: WitnessProgram,
) -> Result<(), WitnessProgramError> {
    if !WitnessOperation::field_supported::<F>() {
        return Err(WitnessProgramError::UnsupportedField);
    }
    if !same_circuit(&program.key, prover) {
        return Err(WitnessProgramError::WrongCircuit);
    }
    if !prover.witness_program.has_origin(&program.origin)
        && plan_bytes::operation_digest(&prover.generators) != program.key.operations
    {
        return Err(WitnessProgramError::ChangedOperations);
    }
    program
        .program
        .prepare(
            &prover.representative_map,
            common.config.num_wires,
            common.degree(),
        )
        .map_err(|e| WitnessProgramError::Invalid(format!("preparing execution: {e:#}")))?;
    prover.witness_program.install_all(
        prover.generators.len(),
        program.key.mask.clone(),
        program.program,
        program.key.operations,
        program.key,
    );
    Ok(())
}

/// The installed program, if any.
pub(crate) fn installed<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    prover: &ProverOnlyCircuitData<F, C, D>,
) -> Option<WitnessProgram> {
    if let Some(program) = prover
        .witness_program
        .installed_program(prover.generators.len())
    {
        return Some(program);
    }
    let (mask, program, operations) = prover
        .witness_program
        .installed_all(prover.generators.len())?;
    let operations = operations.unwrap_or_else(|| plan_bytes::operation_digest(&prover.generators));
    let compilation = program.runtime.config.compilation_key();
    Some(WitnessProgram {
        origin: prover.witness_program.origin(),
        program,
        key: Arc::new(PlanKey {
            compilation,
            ..key(prover, mask, operations)
        }),
    })
}

/// Removes the installed program; later proofs use the worklist.
pub(crate) fn remove<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    prover: &ProverOnlyCircuitData<F, C, D>,
) {
    let _preparation = prover.witness_program.preparation_guard();
    prover.witness_program.clear();
}

/// Decodes a program for this circuit, checking the circuit, the format and
/// the generators' operations (one pass over the generators).
pub(crate) fn from_bytes<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    bytes: &[u8],
    prover: &ProverOnlyCircuitData<F, C, D>,
) -> Result<WitnessProgram, WitnessProgramError> {
    from_bytes_with_config(bytes, prover, WitnessConfig::from_env()?)
}
pub(crate) fn from_bytes_with_config<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    bytes: &[u8],
    prover: &ProverOnlyCircuitData<F, C, D>,
    config: WitnessConfig,
) -> Result<WitnessProgram, WitnessProgramError> {
    from_bytes_with_identity(bytes, prover, config, None)
}
fn from_bytes_with_identity<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    bytes: &[u8],
    prover: &ProverOnlyCircuitData<F, C, D>,
    config: WitnessConfig,
    operations: Option<u128>,
) -> Result<WitnessProgram, WitnessProgramError> {
    if !WitnessOperation::field_supported::<F>() {
        return Err(WitnessProgramError::UnsupportedField);
    }
    let (key, mut program) =
        plan_bytes::decode(bytes).map_err(|e| WitnessProgramError::Invalid(e.to_string()))?;
    if !same_circuit(&key, prover) {
        return Err(WitnessProgramError::WrongCircuit);
    }
    if key.format != plan_bytes::WITNESS_PROGRAM_FORMAT {
        return Err(WitnessProgramError::WrongFormat);
    }
    if key.operations
        != operations.unwrap_or_else(|| plan_bytes::operation_digest(&prover.generators))
    {
        return Err(WitnessProgramError::ChangedOperations);
    }
    program.runtime = Arc::new(config::Runtime::new(
        config.with_compilation_key(key.compilation)?,
    )?);
    Ok(WitnessProgram {
        origin: prover.witness_program.origin(),
        program: Arc::new(program),
        key: Arc::new(key),
    })
}

pub(crate) fn prepare<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    prover: &ProverOnlyCircuitData<F, C, D>,
    common: &CommonCircuitData<F, D>,
    input_targets: &[Target],
    mut config: WitnessConfig,
) -> Result<WitnessPreparation, WitnessProgramError> {
    let start = std::time::Instant::now();
    config.validate()?;
    let directory = cache::directory(&mut config);
    let _preparation = prover.witness_program.preparation_guard();
    if let Some(error) = prover.witness_program.unsupported() {
        return Err(error);
    }
    let unsupported = if !WitnessOperation::field_supported::<F>() {
        Some(WitnessProgramError::UnsupportedField)
    } else if !common.luts.is_empty() || common.num_lookup_polys != 0 {
        Some(WitnessProgramError::Lookups)
    } else {
        None
    };
    if let Some(error) = unsupported {
        prover.witness_program.record_unsupported(&error);
        return Err(error);
    }
    let mut mask = input_targets
        .iter()
        .map(|&t| {
            t.checked_index(common.config.num_wires, common.degree())
                .and_then(|index| prover.representative_map.get(index))
                .copied()
                .ok_or(WitnessProgramError::InputOutsideCircuit(t))
        })
        .collect::<Result<Vec<_>, _>>()?;
    mask.sort_unstable();
    mask.dedup();
    let known = prover
        .witness_program
        .installed_program(prover.generators.len())
        .filter(|p| p.key.mask == mask && p.key.compilation == config.compilation_key());
    if let Some(program) = &known {
        if program.config().same_settings(&config) {
            return Ok(WitnessPreparation {
                source: WitnessProgramSource::Installed,
                elapsed: start.elapsed(),
                cache_read: std::time::Duration::ZERO,
                compile: std::time::Duration::ZERO,
                cache_write: std::time::Duration::ZERO,
                host_prepare: std::time::Duration::ZERO,
                cache_warning: None,
                program: program.clone(),
            });
        }
        if program.config().cache == config.cache {
            let host = std::time::Instant::now();
            let configured = program.with_execution_config(config)?;
            install_inner(prover, common, configured.clone())?;
            return Ok(WitnessPreparation {
                source: WitnessProgramSource::Installed,
                elapsed: start.elapsed(),
                cache_read: std::time::Duration::ZERO,
                compile: std::time::Duration::ZERO,
                cache_write: std::time::Duration::ZERO,
                host_prepare: host.elapsed(),
                cache_warning: None,
                program: configured,
            });
        }
    }
    let operations = known.as_ref().map_or_else(
        || plan_bytes::operation_digest(&prover.generators),
        |p| p.key.operations,
    );
    let mut expected = key(prover, mask, operations);
    expected.compilation = config.compilation_key();
    let reused = known.is_some();
    let outcome = cache::obtain(
        directory.as_deref(),
        &expected,
        |bytes| from_bytes_with_identity(bytes, prover, config.clone(), Some(operations)),
        || match known {
            Some(program) => program.with_execution_config(config.clone()),
            None => compile_with_identity(
                prover,
                common,
                input_targets,
                config.clone(),
                Some(operations),
            ),
        },
    );
    let mut outcome = match outcome {
        Ok(o) => o,
        Err(e) => {
            prover.witness_program.record_unsupported(&e);
            return Err(e);
        }
    };
    if reused && outcome.source == WitnessProgramSource::Compiled {
        outcome.source = WitnessProgramSource::Installed;
    }
    let host = std::time::Instant::now();
    install_inner(prover, common, outcome.program.clone())?;
    let host_prepare = host.elapsed();
    if let Some(warning) = &outcome.warning {
        log::warn!("{warning}; witness preparation succeeded in memory");
    }
    Ok(WitnessPreparation {
        source: outcome.source,
        elapsed: start.elapsed(),
        cache_read: outcome.read,
        compile: outcome.compile,
        cache_write: outcome.cache_write,
        host_prepare,
        cache_warning: outcome.warning,
        program: outcome.program,
    })
}

/// Checks the compiled executor against the worklist for proofs with inputs
/// `inputs`: generates the witness both ways, giving the executor the
/// worklist's random values, and fails unless every value, public input and
/// wire cell agrees exactly. For testing a library's
/// [`WitnessGenerator::witness_operation`] implementations on real circuits;
/// a wrong description otherwise surfaces only as a proof that fails to
/// verify. Compiles a program for `inputs` and leaves it installed.
///
/// [`WitnessGenerator::witness_operation`]: crate::iop::generator::WitnessGenerator::witness_operation
pub fn check_against_worklist<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    inputs: &crate::iop::witness::PartialWitness<F>,
    prover: &ProverOnlyCircuitData<F, C, D>,
    common: &CommonCircuitData<F, D>,
) -> anyhow::Result<()> {
    region::cpu::check_against_worklist(inputs, prover, common).map(|_| ())
}

impl WitnessProgram {
    /// Effective configuration resolved when this program was compiled or loaded.
    pub fn config(&self) -> &WitnessConfig {
        &self.program.runtime.config
    }

    /// Whether all host execution tables have been initialized.
    pub fn is_prepared(&self) -> bool {
        self.program.is_prepared()
    }

    /// Reuses this program and its prepared tables with different resource or
    /// execution settings. The stored layout and fusion policy are preserved.
    pub fn with_execution_config(
        &self,
        config: WitnessConfig,
    ) -> Result<Self, WitnessProgramError> {
        let config = config.with_compilation_key(self.key.compilation)?;
        let mut program = (*self.program).clone();
        program.runtime = Arc::new(config::Runtime::new(config)?);
        Ok(Self {
            origin: self.origin.clone(),
            program: Arc::new(program),
            key: self.key.clone(),
        })
    }

    /// Encodes the program for storage. Decode it with
    /// [`CircuitData::witness_program_from_bytes`] (or the same method on
    /// [`ProverCircuitData`]), which checks that it belongs to the circuit,
    /// that the running build's witness executor is the same version, and that
    /// the circuit's generators still describe the same operations.
    ///
    /// [`CircuitData::witness_program_from_bytes`]: crate::plonk::circuit_data::CircuitData::witness_program_from_bytes
    /// [`ProverCircuitData`]: crate::plonk::circuit_data::ProverCircuitData
    pub fn to_bytes(&self) -> Vec<u8> {
        plan_bytes::encode(&self.program, &self.key)
    }

    /// The number of copy-constraint classes the inputs set.
    pub fn num_input_classes(&self) -> usize {
        self.key.mask.len()
    }
}

#[cfg(test)]
mod tests {
    use super::region::cpu::tests::{circuit, inputs};
    use super::region::cpu::EXECUTOR_RUNS;
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField as G;
    use crate::field::types::Field;
    use crate::iop::witness::PartialWitness;
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::{CircuitConfig, CircuitData};
    use crate::plonk::config::PoseidonGoldilocksConfig as C;

    /// Proves and verifies; returns whether the compiled executor ran.
    fn prove_and_verify(data: &CircuitData<G, C, 2>, pw: PartialWitness<G>) -> bool {
        let before = EXECUTOR_RUNS.with(|runs| runs.get());
        let proof = data.prove(pw).unwrap();
        data.verify(proof).unwrap();
        EXECUTOR_RUNS.with(|runs| runs.get()) > before
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn dependency_schedules_support_concurrent_fork_join_witnesses_and_reconfiguration() {
        use crate::hash::poseidon::PoseidonHash;
        use crate::iop::witness::WitnessWrite;
        let mut builder = CircuitBuilder::<G, 2>::new(CircuitConfig::standard_recursion_config());
        let a = builder.add_virtual_target();
        let b = builder.add_virtual_target();
        let mut branches = Vec::new();
        for i in 0..257 {
            let constant = builder.constant(G::from_canonical_usize(i + 1));
            let mut x = a;
            for _ in 0..(i % 16 + 1) {
                x = builder.mul_add(x, b, constant);
            }
            branches.push(x);
        }
        let sum = builder.add_many(branches);
        let hash = builder.hash_n_to_hash_no_pad::<PoseidonHash>(vec![sum, a, b]);
        builder.register_public_inputs(&hash.elements);
        let data = builder.build::<C>();
        let mut config = WitnessConfig {
            policy: policy::Policy {
                schedule: policy::WitnessSchedule::Dependencies,
                ..Default::default()
            },
            threads: 2,
            buffers: WitnessBuffers::Reuse {
                max_bytes: 8 * 1024 * 1024,
            },
            ..WitnessConfig::default()
        };
        data.prepare_witness_program_with_config(&[a, b], config.clone())
            .unwrap();
        std::thread::scope(|scope| {
            for worker in 0..8 {
                let data = &data;
                scope.spawn(move || {
                    for case in 0..3 {
                        let mut pw = PartialWitness::new();
                        pw.set_target(a, G::from_canonical_usize(worker + case))
                            .unwrap();
                        pw.set_target(b, G::from_canonical_usize(case + 1)).unwrap();
                        check_against_worklist(&pw, &data.prover_only, &data.common).unwrap();
                    }
                });
            }
        });
        config.policy.schedule = policy::WitnessSchedule::Layers;
        data.prepare_witness_program_with_config(&[a, b], config)
            .unwrap();
        assert_eq!(
            data.witness_program().unwrap().config().policy.schedule,
            policy::WitnessSchedule::Layers
        );
        let mut pw = PartialWitness::new();
        pw.set_target(a, G::ONE).unwrap();
        pw.set_target(b, G::TWO).unwrap();
        assert!(prove_and_verify(&data, pw));
    }

    #[test]
    fn external_target_coordinates_are_rejected_without_wrapping_or_aliasing() {
        let mut builder = CircuitBuilder::<G, 2>::new(CircuitConfig::standard_recursion_config());
        let a = builder.add_virtual_target();
        let b = builder.add_virtual_target();
        let product = builder.mul(a, b);
        builder.register_public_input(product);
        let data = builder.build::<C>();
        let mut inputs = PartialWitness::new();
        inputs.set_target(a, G::from_canonical_u64(7)).unwrap();
        inputs.set_target(b, G::from_canonical_u64(9)).unwrap();
        let invalids = [
            Target::wire(0, data.common.config.num_wires),
            Target::wire(data.common.degree(), 0),
            Target::wire(usize::MAX, 0),
            Target::VirtualTarget { index: usize::MAX },
        ];
        for target in invalids {
            assert_eq!(
                data.compile_witness_program(&[target]).unwrap_err(),
                WitnessProgramError::InputOutsideCircuit(target)
            );
            assert_eq!(
                data.prepare_witness_program(&[target]).unwrap_err(),
                WitnessProgramError::InputOutsideCircuit(target)
            );
            for mode in 0..3 {
                data.remove_witness_program();
                if mode != 0 {
                    let mut config = WitnessConfig::default();
                    if mode == 2 {
                        config.fallback = WitnessFallback::Error;
                    }
                    data.prepare_witness_program_with_config(&[a, b], config)
                        .unwrap();
                }
                let mut bad = inputs.clone();
                bad.set_target(target, G::from_canonical_u64(7)).unwrap();
                assert!(data.prove(bad).is_err());
            }
        }
    }

    #[test]
    fn worklist_by_default_and_nothing_compiled_implicitly() {
        let fixture = circuit(true);
        let data = &fixture.data;
        assert!(!prove_and_verify(data, inputs(&fixture)));
        assert!(!prove_and_verify(data, inputs(&fixture)));
        assert!(data.witness_program().is_none());
    }

    #[test]
    fn installed_program_runs_until_removed() {
        for blinded in [false, true] {
            let fixture = circuit(blinded);
            let data = &fixture.data;
            data.witness_program_support().unwrap();
            let program = data
                .compile_witness_program_like(&inputs(&fixture))
                .unwrap();
            assert!(program.num_input_classes() > 0);
            data.install_witness_program(program).unwrap();
            assert!(data.witness_program().is_some());
            assert!(prove_and_verify(data, inputs(&fixture)));
            // Clones share the installed program.
            assert!(prove_and_verify(&data.clone(), inputs(&fixture)));
            data.remove_witness_program();
            assert!(data.witness_program().is_none());
            assert!(!prove_and_verify(data, inputs(&fixture)));
        }
    }

    #[test]
    fn proofs_with_other_inputs_use_the_worklist() {
        let fixture = circuit(true);
        let data = &fixture.data;
        let pw = inputs(&fixture);
        // Compiled for every input except `index`, so a full proof's inputs
        // set one more class than the program expects.
        let targets = pw
            .target_values
            .keys()
            .copied()
            .filter(|&t| t != fixture.index)
            .collect::<Vec<_>>();
        let program = data.compile_witness_program(&targets).unwrap();
        data.install_witness_program(program).unwrap();
        assert!(!prove_and_verify(data, pw));
        assert!(data.witness_program().is_some());
    }

    #[test]
    fn bytes_round_trip_and_are_checked() {
        let fixture = circuit(true);
        let other = circuit(false);
        let data = &fixture.data;
        let program = data
            .compile_witness_program_like(&inputs(&fixture))
            .unwrap();
        let bytes = program.to_bytes();

        let decoded = data.witness_program_from_bytes(&bytes).unwrap();
        assert_eq!(decoded.to_bytes(), bytes);
        data.install_witness_program(decoded).unwrap();
        assert!(prove_and_verify(data, inputs(&fixture)));

        assert_eq!(
            other.data.witness_program_from_bytes(&bytes).unwrap_err(),
            WitnessProgramError::WrongCircuit
        );
        assert_eq!(
            other
                .data
                .install_witness_program(program.clone())
                .unwrap_err(),
            WitnessProgramError::WrongCircuit
        );
        let mut damaged = bytes.clone();
        damaged[bytes.len() / 2] ^= 1;
        assert!(matches!(
            data.witness_program_from_bytes(&damaged),
            Err(WitnessProgramError::Invalid(_))
        ));
        let other_format = PlanKey {
            format: program.key.format.wrapping_add(1),
            ..(*program.key).clone()
        };
        assert_eq!(
            data.witness_program_from_bytes(&plan_bytes::encode(&program.program, &other_format))
                .unwrap_err(),
            WitnessProgramError::WrongFormat
        );
        let other_operations = PlanKey {
            operations: program.key.operations ^ 1,
            ..(*program.key).clone()
        };
        assert_eq!(
            data.witness_program_from_bytes(&plan_bytes::encode(
                &program.program,
                &other_operations
            ))
            .unwrap_err(),
            WitnessProgramError::ChangedOperations
        );
    }

    #[test]
    fn programs_carry_over_to_an_independent_build_of_the_circuit() {
        let first = circuit(true);
        let again = circuit(true);
        let other = circuit(false);
        let digest = |data: &CircuitData<G, C, 2>| {
            plan_bytes::operation_digest(&data.prover_only.generators)
        };
        assert_eq!(digest(&first.data), digest(&again.data));
        assert_ne!(digest(&first.data), digest(&other.data));
        // As if stored by one process and loaded by the next.
        let bytes = first
            .data
            .compile_witness_program_like(&inputs(&first))
            .unwrap()
            .to_bytes();
        let loaded = again.data.witness_program_from_bytes(&bytes).unwrap();
        again.data.install_witness_program(loaded).unwrap();
        assert!(prove_and_verify(&again.data, inputs(&again)));
        assert_eq!(again.data.witness_program().unwrap().to_bytes(), bytes);
    }

    #[test]
    fn explicit_configuration_is_per_program_and_survives_storage() {
        let fast = circuit(true);
        let checked = circuit(true);
        let targets = inputs(&fast)
            .target_values
            .keys()
            .copied()
            .collect::<Vec<_>>();
        let mut config = WitnessConfig {
            buffers: WitnessBuffers::Reuse {
                max_bytes: 4 * 1024 * 1024,
            },
            ..WitnessConfig::default()
        };
        #[cfg(feature = "parallel")]
        {
            config.thread_pool = Some(Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(2)
                    .build()
                    .unwrap(),
            ));
        }
        let program = fast
            .data
            .compile_witness_program_with_config(&targets, config.clone())
            .unwrap();
        assert_eq!(
            program.config().policy.execution,
            policy::WitnessExecution::Auto
        );
        let bytes = program.to_bytes();
        fast.data.install_witness_program(program).unwrap();
        config.policy.execution = policy::WitnessExecution::Checked;
        let loaded = checked
            .data
            .witness_program_from_bytes_with_config(&bytes, config)
            .unwrap();
        assert_eq!(
            loaded.config().policy.execution,
            policy::WitnessExecution::Checked
        );
        assert_eq!(loaded.config().policy.chain_length, 4);
        checked.data.install_witness_program(loaded).unwrap();
        assert!(prove_and_verify(&fast.data, inputs(&fast)));
        assert!(prove_and_verify(&checked.data, inputs(&checked)));
        assert!(fast.data.has_witness_program());
        assert_eq!(
            fast.data
                .witness_program()
                .unwrap()
                .config()
                .policy
                .execution,
            policy::WitnessExecution::Auto
        );
        let invalid = WitnessConfig {
            buffers: WitnessBuffers::Reuse { max_bytes: 0 },
            ..WitnessConfig::default()
        };
        assert!(matches!(
            fast.data
                .compile_witness_program_with_config(&targets, invalid),
            Err(WitnessProgramError::InvalidConfiguration(_))
        ));
    }
    #[test]
    fn installation_prepares_tables_and_program_clones_share_metadata() {
        let fixture = circuit(false);
        let data = &fixture.data;
        let program = data
            .compile_witness_program_like(&inputs(&fixture))
            .unwrap();
        let clone = program.clone();
        assert!(Arc::ptr_eq(&program.key, &clone.key));
        assert!(Arc::ptr_eq(&program.program, &clone.program));
        let config = WitnessConfig {
            policy: policy::Policy {
                execution: policy::WitnessExecution::Checked,
                ..Default::default()
            },
            ..WitnessConfig::default()
        };
        let checked = program.with_execution_config(config).unwrap();
        data.install_witness_program(clone).unwrap();
        assert!(program.program.is_prepared());
        assert!(checked.program.is_prepared());
        let installed = data.witness_program().unwrap();
        assert!(Arc::ptr_eq(&program.key, &installed.key));
        assert!(Arc::ptr_eq(&program.program, &installed.program));
        assert!(prove_and_verify(data, inputs(&fixture)));
    }

    #[test]
    fn strict_input_shape_reports_unexpected_worklist_fallback() {
        let fixture = circuit(true);
        let pw = inputs(&fixture);
        let targets = pw
            .target_values
            .keys()
            .copied()
            .filter(|&t| t != fixture.index)
            .collect::<Vec<_>>();
        let config = WitnessConfig {
            fallback: WitnessFallback::Error,
            ..WitnessConfig::default()
        };
        let program = fixture
            .data
            .compile_witness_program_with_config(&targets, config)
            .unwrap();
        fixture.data.install_witness_program(program).unwrap();
        let Err(error) = fixture.data.prove(pw) else {
            panic!("strict execution must reject a mismatched input shape")
        };
        assert!(format!("{error:#}").contains("inputs do not match"));
    }

    #[test]
    fn unsupported_generator_is_named() {
        let mut builder = CircuitBuilder::<G, 2>::new(CircuitConfig::standard_recursion_config());
        // Summing more bits than one arithmetic gate holds adds a
        // `BaseSumGenerator`, which does not describe its operation.
        let bits = (0..32)
            .map(|_| builder.add_virtual_bool_target_safe())
            .collect::<Vec<_>>();
        let sum = builder.le_sum(bits.iter());
        builder.register_public_input(sum);
        let data = builder.build::<C>();
        let error = data.witness_program_support().unwrap_err();
        let WitnessProgramError::UnsupportedGenerator { id, .. } = &error else {
            panic!("unexpected {error}");
        };
        assert!(id.contains("BaseSumGenerator"), "{id}");
        assert_eq!(data.compile_witness_program(&[]).unwrap_err(), error);
        assert_eq!(
            data.witness_program_status(),
            WitnessProgramStatus::Unprepared
        );
        assert_eq!(data.prepare_witness_program(&[]).unwrap_err(), error);
        assert_eq!(
            data.witness_program_status(),
            WitnessProgramStatus::Unsupported
        );
        assert_eq!(data.prepare_witness_program(&[]).unwrap_err(), error);
        data.remove_witness_program();
        assert_eq!(
            data.witness_program_status(),
            WitnessProgramStatus::Unprepared
        );
        // The worklist can prove a circuit with an unsupported generator.
        let mut pw = PartialWitness::new();
        for (i, bit) in bits.iter().enumerate() {
            pw.set_bool_target(*bit, i % 3 == 0).unwrap();
        }
        assert!(!prove_and_verify(&data, pw));
    }
}

#[doc(hidden)]
#[cfg(feature = "witness-benchmarks")]
pub fn replay_snapshot(stem: &std::path::Path, iterations: usize) -> anyhow::Result<()> {
    region::cpu::replay_snapshot(stem, iterations)
}

/// Benchmark-only control. Change only between proofs, with no executions active.
#[doc(hidden)]
#[cfg(feature = "witness-benchmarks")]
pub fn benchmark_profile(active: bool) {
    region::benchmark_profile(active);
}

/// Benchmark-only input binding control. Change only between serial proofs.
#[doc(hidden)]
#[cfg(feature = "witness-benchmarks")]
pub fn benchmark_input_binding(direct: bool) {
    region::cpu::benchmark_input_binding(direct);
}

/// Benchmark-only buffer reuse control. Change only between serial proofs.
#[doc(hidden)]
#[cfg(feature = "witness-benchmarks")]
pub fn benchmark_buffer_reuse(enabled: bool) {
    region::buffers::benchmark_override(enabled);
}

#[doc(hidden)]
#[cfg(feature = "witness-benchmarks")]
pub fn benchmark_random_spans(enabled: bool) {
    region::aot::benchmark_random_spans(enabled);
}

#[doc(hidden)]
#[cfg(feature = "witness-benchmarks")]
pub fn benchmark_compact_readiness(enabled: bool) {
    region::cpu::benchmark_compact_readiness(enabled);
}

#[doc(hidden)]
#[cfg(all(feature = "gpu", feature = "witness-benchmarks"))]
pub fn benchmark_gather_tiling(current: bool) {
    region::cpu::benchmark_gather_tiling(current);
}

#[doc(hidden)]
#[cfg(all(feature = "parallel", feature = "witness-benchmarks"))]
pub fn benchmark_dependency_schedule(enabled: bool) {
    region::schedule::benchmark_enabled(enabled);
}
