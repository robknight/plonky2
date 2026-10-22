//! Compiled witness programs: an explicit, caller-managed optimisation of
//! witness generation.
//!
//! By default a proof generates its witness with the worklist, which runs
//! every generator and needs no preparation. A caller that proves the same
//! circuit many times can instead compile the circuit's generators once into
//! a [`WitnessProgram`] and install it; later proofs whose inputs set the
//! same targets then run the compiled executor, which is two to three times
//! faster on large circuits. Compiling takes about 0.3 s per million
//! generators on an Apple M4 (several times longer on slower server CPUs),
//! so it pays off only for circuits that are proved repeatedly, and only the
//! caller knows whether that is the case.
//!
//! Available with the default `std` and `rand_chacha` features, with or
//! without `gpu`; with `gpu` the executor writes the wires straight into
//! device memory when the GPU prover applies.
//!
//! Nothing here happens implicitly: [`CircuitData::prove`] never compiles,
//! stores or loads a program. The caller compiles with
//! [`CircuitData::compile_witness_program`], installs with
//! [`CircuitData::install_witness_program`], and may keep the program across
//! processes with [`WitnessProgram::to_bytes`] and
//! [`CircuitData::witness_program_from_bytes`]. Stored programs are a
//! trusted input, like serialized circuit data: a checksum detects damage,
//! not tampering.
//!
//! A program applies to proofs whose
//! [`PartialWitness`](crate::iop::witness::PartialWitness) sets targets in
//! exactly the copy-constraint classes it was compiled for; other proofs use
//! the worklist, and the first such proof logs a warning. A circuit with
//! lookup tables, or with a generator that does not describe its operation
//! (see [`WitnessGenerator::witness_operation`]), cannot be compiled; check with
//! [`CircuitData::witness_program_support`].
//!
//! [`CircuitData::prove`]: crate::plonk::circuit_data::CircuitData::prove
//! [`CircuitData::compile_witness_program`]: crate::plonk::circuit_data::CircuitData::compile_witness_program
//! [`CircuitData::install_witness_program`]: crate::plonk::circuit_data::CircuitData::install_witness_program
//! [`CircuitData::witness_program_from_bytes`]: crate::plonk::circuit_data::CircuitData::witness_program_from_bytes
//! [`CircuitData::witness_program_support`]: crate::plonk::circuit_data::CircuitData::witness_program_support
//! [`WitnessGenerator::witness_operation`]: crate::iop::generator::WitnessGenerator::witness_operation

#[doc(hidden)]
pub mod region;

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

/// A compiled witness program for one circuit and one set of input targets.
/// Cheap to clone; clones share the compiled program.
#[derive(Clone)]
pub struct WitnessProgram {
    program: Arc<RegionProgram>,
    key: PlanKey,
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
        if target.index(num_wires, degree) >= map.len() {
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
    let program = RegionProgram::compile(&prover.generators, &entry, &indices)
        .map_err(|e| WitnessProgramError::Invalid(format!("{e:#}")))?;
    let started = std::time::Instant::now();
    let operations = plan_bytes::operation_digest(&prover.generators);
    log::debug!(target: COMPILE_LOG, "compile phase operation_digest {:?}", started.elapsed());
    Ok(WitnessProgram {
        program: Arc::new(program),
        key: key(prover, mask, operations),
    })
}

/// Installs `program` on the circuit, replacing any installed program.
pub(crate) fn install<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    prover: &ProverOnlyCircuitData<F, C, D>,
    program: WitnessProgram,
) -> Result<(), WitnessProgramError> {
    if !same_circuit(&program.key, prover) {
        return Err(WitnessProgramError::WrongCircuit);
    }
    prover.witness_program.install_all(
        prover.generators.len(),
        program.key.mask,
        program.program,
        program.key.operations,
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
    let (mask, program, operations) = prover
        .witness_program
        .installed_all(prover.generators.len())?;
    let operations = operations.unwrap_or_else(|| plan_bytes::operation_digest(&prover.generators));
    Some(WitnessProgram {
        program,
        key: key(prover, mask, operations),
    })
}

/// Removes the installed program; later proofs use the worklist.
pub(crate) fn remove<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    prover: &ProverOnlyCircuitData<F, C, D>,
) {
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
    let (key, program) =
        plan_bytes::decode(bytes).map_err(|e| WitnessProgramError::Invalid(e.to_string()))?;
    if !same_circuit(&key, prover) {
        return Err(WitnessProgramError::WrongCircuit);
    }
    if key.format != plan_bytes::WITNESS_PROGRAM_FORMAT {
        return Err(WitnessProgramError::WrongFormat);
    }
    if key.operations != plan_bytes::operation_digest(&prover.generators) {
        return Err(WitnessProgramError::ChangedOperations);
    }
    Ok(WitnessProgram {
        program: Arc::new(program),
        key,
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
            ..program.key.clone()
        };
        assert_eq!(
            data.witness_program_from_bytes(&plan_bytes::encode(&program.program, &other_format))
                .unwrap_err(),
            WitnessProgramError::WrongFormat
        );
        let other_operations = PlanKey {
            operations: program.key.operations ^ 1,
            ..program.key.clone()
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
        // The worklist proves it as before.
        let mut pw = PartialWitness::new();
        for (i, bit) in bits.iter().enumerate() {
            pw.set_bool_target(*bit, i % 3 == 0).unwrap();
        }
        assert!(!prove_and_verify(&data, pw));
    }
}
