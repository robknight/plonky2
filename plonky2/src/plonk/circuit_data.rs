//! Circuit data specific to the prover and the verifier.
//!
//! This module also defines a [`CircuitConfig`] to be customized
//! when building circuits for arbitrary statements.
//!
//! After building a circuit, one obtains an instance of [`CircuitData`].
//! This contains both prover and verifier data, allowing to generate
//! proofs for the given circuit and verify them.
//!
//! Most of the [`CircuitData`] is actually prover-specific, and can be
//! extracted by calling [`CircuitData::prover_data`] method.
//! The verifier data can similarly be extracted by calling [`CircuitData::verifier_data`].
//! This is useful to allow even small devices to verify plonky2 proofs.

#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeMap, vec, vec::Vec};
use core::ops::{Range, RangeFrom};
#[cfg(feature = "std")]
use std::collections::BTreeMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::circuit_builder::LookupWire;
use crate::field::extension::Extendable;
use crate::field::fft::FftRootTable;
use crate::field::types::Field;
use crate::fri::oracle::PolynomialBatch;
use crate::fri::reduction_strategies::FriReductionStrategy;
use crate::fri::structure::{
    FriBatchInfo, FriBatchInfoTarget, FriInstanceInfo, FriInstanceInfoTarget, FriOracleInfo,
    FriPolynomialInfo,
};
use crate::fri::{FriConfig, FriParams};
use crate::gates::gate::GateRef;
use crate::gates::lookup::Lookup;
use crate::gates::lookup_table::LookupTable;
use crate::gates::selectors::SelectorsInfo;
use crate::hash::hash_types::{HashOutTarget, MerkleCapTarget, RichField};
use crate::hash::merkle_tree::MerkleCap;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{generate_partial_witness, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::witness::{PartialWitness, PartitionWitness};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::config::{GenericConfig, Hasher};
use crate::plonk::plonk_common::PlonkOracle;
use crate::plonk::proof::{CompressedProofWithPublicInputs, ProofWithPublicInputs};
use crate::plonk::prover::prove;
use crate::plonk::verifier::verify;
use crate::util::serialization::{
    Buffer, GateSerializer, IoResult, Read, WitnessGeneratorSerializer, Write,
};
use crate::util::timing::TimingTree;

/// Configuration to be used when building a circuit. This defines the shape of the circuit
/// as well as its targeted security level and sub-protocol (e.g. FRI) parameters.
///
/// It supports a [`Default`] implementation tailored for recursion with Poseidon hash (of width 12)
/// as internal hash function and FRI rate of 1/8.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CircuitConfig {
    /// The number of wires available at each row. This corresponds to the "width" of the circuit,
    /// and consists in the sum of routed wires and advice wires.
    pub num_wires: usize,
    /// The number of routed wires, i.e. wires that will be involved in Plonk's permutation argument.
    /// This allows copy constraints, i.e. enforcing that two distant values in a circuit are equal.
    /// Non-routed wires are called advice wires.
    pub num_routed_wires: usize,
    /// The number of constants that can be used per gate. If a gate requires more constants than the config
    /// allows, the [`CircuitBuilder`] will complain when trying to add this gate to its set of gates.
    pub num_constants: usize,
    /// Whether to use a dedicated gate for base field arithmetic, rather than using a single gate
    /// for both base field and extension field arithmetic.
    pub use_base_arithmetic_gate: bool,
    pub security_bits: usize,
    /// The number of challenge points to generate, for IOPs that have soundness errors of (roughly)
    /// `degree / |F|`.
    pub num_challenges: usize,
    /// A boolean to activate the zero-knowledge property. When this is set to `false`, proofs *may*
    /// leak additional information.
    pub zero_knowledge: bool,
    /// A cap on the quotient polynomial's degree factor. The actual degree factor is derived
    /// systematically, but will never exceed this value.
    pub max_quotient_degree_factor: usize,
    pub fri_config: FriConfig,
}

impl Default for CircuitConfig {
    fn default() -> Self {
        Self::standard_recursion_config()
    }
}

impl CircuitConfig {
    pub const fn num_advice_wires(&self) -> usize {
        self.num_wires - self.num_routed_wires
    }

    /// A typical recursion config, without zero-knowledge, targeting ~100 bit security.
    pub const fn standard_recursion_config() -> Self {
        Self {
            num_wires: 135,
            num_routed_wires: 80,
            num_constants: 2,
            use_base_arithmetic_gate: true,
            security_bits: 100,
            num_challenges: 2,
            zero_knowledge: false,
            max_quotient_degree_factor: 8,
            fri_config: FriConfig {
                rate_bits: 3,
                cap_height: 4,
                proof_of_work_bits: 16,
                reduction_strategy: FriReductionStrategy::ConstantArityBits(4, 5),
                num_query_rounds: 28,
            },
        }
    }

    pub fn standard_ecc_config() -> Self {
        Self {
            num_wires: 136,
            ..Self::standard_recursion_config()
        }
    }

    pub fn wide_ecc_config() -> Self {
        Self {
            num_wires: 234,
            ..Self::standard_recursion_config()
        }
    }

    pub fn standard_recursion_zk_config() -> Self {
        CircuitConfig {
            zero_knowledge: true,
            ..Self::standard_recursion_config()
        }
    }
}

/// Mock circuit data to only do witness generation without generating a proof.
#[derive(Eq, PartialEq, Debug)]
pub struct MockCircuitData<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    pub prover_only: ProverOnlyCircuitData<F, C, D>,
    pub common: CommonCircuitData<F, D>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    MockCircuitData<F, C, D>
{
    pub fn generate_witness(&self, inputs: PartialWitness<F>) -> PartitionWitness<'_, F> {
        generate_partial_witness::<F, C, D>(inputs, &self.prover_only, &self.common).unwrap()
    }
}

/// Circuit data required by the prover or the verifier.
#[derive(Eq, PartialEq, Debug, Clone)]
pub struct CircuitData<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> {
    pub prover_only: ProverOnlyCircuitData<F, C, D>,
    pub verifier_only: VerifierOnlyCircuitData<C, D>,
    pub common: CommonCircuitData<F, D>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    CircuitData<F, C, D>
{
    pub fn to_bytes(
        &self,
        gate_serializer: &dyn GateSerializer<F, D>,
        generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
    ) -> IoResult<Vec<u8>> {
        let mut buffer = Vec::new();
        buffer.write_circuit_data(self, gate_serializer, generator_serializer)?;
        Ok(buffer)
    }

    pub fn from_bytes(
        bytes: &[u8],
        gate_serializer: &dyn GateSerializer<F, D>,
        generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
    ) -> IoResult<Self> {
        let mut buffer = Buffer::new(bytes);
        buffer.read_circuit_data(gate_serializer, generator_serializer)
    }

    pub fn prove(&self, inputs: PartialWitness<F>) -> Result<ProofWithPublicInputs<F, C, D>> {
        prove::<F, C, D>(
            &self.prover_only,
            &self.common,
            inputs,
            &mut TimingTree::default(),
        )
    }

    pub fn verify(&self, proof_with_pis: ProofWithPublicInputs<F, C, D>) -> Result<()> {
        verify::<F, C, D>(proof_with_pis, &self.verifier_only, &self.common)
    }

    /// Checks, without compiling, whether the circuit can have a witness
    /// program: every generator must describe its operation and the circuit
    /// must not use lookup tables. See [`crate::plonk::witness_program`].
    #[cfg(all(feature = "std", feature = "rand_chacha"))]
    pub fn witness_program_support(
        &self,
    ) -> core::result::Result<(), crate::plonk::witness_program::WitnessProgramError> {
        crate::plonk::witness_program::support(&self.prover_only, &self.common)
    }

    /// Compiles a witness program for proofs whose inputs set exactly
    /// `input_targets`; their values do not matter. Takes about 0.3 s per
    /// million generators on an Apple M4; does not install the program.
    #[cfg(all(feature = "std", feature = "rand_chacha"))]
    pub fn compile_witness_program(
        &self,
        input_targets: &[crate::iop::target::Target],
    ) -> core::result::Result<
        crate::plonk::witness_program::WitnessProgram,
        crate::plonk::witness_program::WitnessProgramError,
    > {
        crate::plonk::witness_program::compile(&self.prover_only, &self.common, input_targets)
    }

    /// Compiles a witness program for proofs that set the same targets as
    /// `inputs`; only the targets are read, not their values.
    #[cfg(all(feature = "std", feature = "rand_chacha"))]
    pub fn compile_witness_program_like(
        &self,
        inputs: &PartialWitness<F>,
    ) -> core::result::Result<
        crate::plonk::witness_program::WitnessProgram,
        crate::plonk::witness_program::WitnessProgramError,
    > {
        let targets = inputs.target_values.keys().copied().collect::<Vec<_>>();
        self.compile_witness_program(&targets)
    }

    /// Installs `program`, replacing any installed program. Later proofs whose
    /// inputs match it use the compiled executor; others use the worklist.
    /// Clones of this circuit data share the installed program.
    #[cfg(all(feature = "std", feature = "rand_chacha"))]
    pub fn install_witness_program(
        &self,
        program: crate::plonk::witness_program::WitnessProgram,
    ) -> core::result::Result<(), crate::plonk::witness_program::WitnessProgramError> {
        crate::plonk::witness_program::install(&self.prover_only, program)
    }

    /// Removes the installed witness program; later proofs use the worklist.
    #[cfg(all(feature = "std", feature = "rand_chacha"))]
    pub fn remove_witness_program(&self) {
        crate::plonk::witness_program::remove(&self.prover_only)
    }

    /// The installed witness program, if any.
    #[cfg(all(feature = "std", feature = "rand_chacha"))]
    pub fn witness_program(&self) -> Option<crate::plonk::witness_program::WitnessProgram> {
        crate::plonk::witness_program::installed(&self.prover_only)
    }

    /// Decodes a witness program stored with
    /// [`WitnessProgram::to_bytes`](crate::plonk::witness_program::WitnessProgram::to_bytes),
    /// checking that it belongs to this circuit and to the running build. Does
    /// not install it.
    #[cfg(all(feature = "std", feature = "rand_chacha"))]
    pub fn witness_program_from_bytes(
        &self,
        bytes: &[u8],
    ) -> core::result::Result<
        crate::plonk::witness_program::WitnessProgram,
        crate::plonk::witness_program::WitnessProgramError,
    > {
        crate::plonk::witness_program::from_bytes(bytes, &self.prover_only)
    }

    pub fn verify_compressed(
        &self,
        compressed_proof_with_pis: CompressedProofWithPublicInputs<F, C, D>,
    ) -> Result<()> {
        compressed_proof_with_pis.verify(&self.verifier_only, &self.common)
    }

    pub fn compress(
        &self,
        proof: ProofWithPublicInputs<F, C, D>,
    ) -> Result<CompressedProofWithPublicInputs<F, C, D>> {
        proof.compress(&self.verifier_only.circuit_digest, &self.common)
    }

    pub fn decompress(
        &self,
        proof: CompressedProofWithPublicInputs<F, C, D>,
    ) -> Result<ProofWithPublicInputs<F, C, D>> {
        proof.decompress(&self.verifier_only.circuit_digest, &self.common)
    }

    pub fn verifier_data(&self) -> VerifierCircuitData<F, C, D> {
        let CircuitData {
            verifier_only,
            common,
            ..
        } = self;
        VerifierCircuitData {
            verifier_only: verifier_only.clone(),
            common: common.clone(),
        }
    }

    pub fn prover_data(self) -> ProverCircuitData<F, C, D> {
        let CircuitData {
            prover_only,
            common,
            ..
        } = self;
        ProverCircuitData {
            prover_only,
            common,
        }
    }
}

/// Circuit data required by the prover. This may be thought of as a proving key, although it
/// includes code for witness generation.
///
/// The goal here is to make proof generation as fast as we can, rather than making this prover
/// structure as succinct as we can. Thus we include various precomputed data which isn't strictly
/// required, like LDEs of preprocessed polynomials. If more succinctness was desired, we could
/// construct a more minimal prover structure and convert back and forth.
#[derive(Debug)]
pub struct ProverCircuitData<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    pub prover_only: ProverOnlyCircuitData<F, C, D>,
    pub common: CommonCircuitData<F, D>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    ProverCircuitData<F, C, D>
{
    pub fn to_bytes(
        &self,
        gate_serializer: &dyn GateSerializer<F, D>,
        generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
    ) -> IoResult<Vec<u8>> {
        let mut buffer = Vec::new();
        buffer.write_prover_circuit_data(self, gate_serializer, generator_serializer)?;
        Ok(buffer)
    }

    pub fn from_bytes(
        bytes: &[u8],
        gate_serializer: &dyn GateSerializer<F, D>,
        generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
    ) -> IoResult<Self> {
        let mut buffer = Buffer::new(bytes);
        buffer.read_prover_circuit_data(gate_serializer, generator_serializer)
    }

    pub fn prove(&self, inputs: PartialWitness<F>) -> Result<ProofWithPublicInputs<F, C, D>> {
        prove::<F, C, D>(
            &self.prover_only,
            &self.common,
            inputs,
            &mut TimingTree::default(),
        )
    }

    /// Checks, without compiling, whether the circuit can have a witness
    /// program: every generator must describe its operation and the circuit
    /// must not use lookup tables. See [`crate::plonk::witness_program`].
    #[cfg(all(feature = "std", feature = "rand_chacha"))]
    pub fn witness_program_support(
        &self,
    ) -> core::result::Result<(), crate::plonk::witness_program::WitnessProgramError> {
        crate::plonk::witness_program::support(&self.prover_only, &self.common)
    }

    /// Compiles a witness program for proofs whose inputs set exactly
    /// `input_targets`; their values do not matter. Takes about 0.3 s per
    /// million generators on an Apple M4; does not install the program.
    #[cfg(all(feature = "std", feature = "rand_chacha"))]
    pub fn compile_witness_program(
        &self,
        input_targets: &[crate::iop::target::Target],
    ) -> core::result::Result<
        crate::plonk::witness_program::WitnessProgram,
        crate::plonk::witness_program::WitnessProgramError,
    > {
        crate::plonk::witness_program::compile(&self.prover_only, &self.common, input_targets)
    }

    /// Compiles a witness program for proofs that set the same targets as
    /// `inputs`; only the targets are read, not their values.
    #[cfg(all(feature = "std", feature = "rand_chacha"))]
    pub fn compile_witness_program_like(
        &self,
        inputs: &PartialWitness<F>,
    ) -> core::result::Result<
        crate::plonk::witness_program::WitnessProgram,
        crate::plonk::witness_program::WitnessProgramError,
    > {
        let targets = inputs.target_values.keys().copied().collect::<Vec<_>>();
        self.compile_witness_program(&targets)
    }

    /// Installs `program`, replacing any installed program. Later proofs whose
    /// inputs match it use the compiled executor; others use the worklist.
    /// Clones of this circuit data share the installed program.
    #[cfg(all(feature = "std", feature = "rand_chacha"))]
    pub fn install_witness_program(
        &self,
        program: crate::plonk::witness_program::WitnessProgram,
    ) -> core::result::Result<(), crate::plonk::witness_program::WitnessProgramError> {
        crate::plonk::witness_program::install(&self.prover_only, program)
    }

    /// Removes the installed witness program; later proofs use the worklist.
    #[cfg(all(feature = "std", feature = "rand_chacha"))]
    pub fn remove_witness_program(&self) {
        crate::plonk::witness_program::remove(&self.prover_only)
    }

    /// The installed witness program, if any.
    #[cfg(all(feature = "std", feature = "rand_chacha"))]
    pub fn witness_program(&self) -> Option<crate::plonk::witness_program::WitnessProgram> {
        crate::plonk::witness_program::installed(&self.prover_only)
    }

    /// Decodes a witness program stored with
    /// [`WitnessProgram::to_bytes`](crate::plonk::witness_program::WitnessProgram::to_bytes),
    /// checking that it belongs to this circuit and to the running build. Does
    /// not install it.
    #[cfg(all(feature = "std", feature = "rand_chacha"))]
    pub fn witness_program_from_bytes(
        &self,
        bytes: &[u8],
    ) -> core::result::Result<
        crate::plonk::witness_program::WitnessProgram,
        crate::plonk::witness_program::WitnessProgramError,
    > {
        crate::plonk::witness_program::from_bytes(bytes, &self.prover_only)
    }
}

/// Circuit data required by the prover.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierCircuitData<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    pub verifier_only: VerifierOnlyCircuitData<C, D>,
    pub common: CommonCircuitData<F, D>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    VerifierCircuitData<F, C, D>
{
    pub fn to_bytes(&self, gate_serializer: &dyn GateSerializer<F, D>) -> IoResult<Vec<u8>> {
        let mut buffer = Vec::new();
        buffer.write_verifier_circuit_data(self, gate_serializer)?;
        Ok(buffer)
    }

    pub fn from_bytes(bytes: &[u8], gate_serializer: &dyn GateSerializer<F, D>) -> IoResult<Self> {
        let mut buffer = Buffer::new(bytes);
        buffer.read_verifier_circuit_data(gate_serializer)
    }

    pub fn verify(&self, proof_with_pis: ProofWithPublicInputs<F, C, D>) -> Result<()> {
        verify::<F, C, D>(proof_with_pis, &self.verifier_only, &self.common)
    }

    pub fn verify_compressed(
        &self,
        compressed_proof_with_pis: CompressedProofWithPublicInputs<F, C, D>,
    ) -> Result<()> {
        compressed_proof_with_pis.verify(&self.verifier_only, &self.common)
    }
}

/// Circuit data required by the prover, but not the verifier.
#[derive(Eq, PartialEq, Debug, Clone)]
pub struct ProverOnlyCircuitData<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    pub generators: Vec<WitnessGeneratorRef<F, D>>,
    /// Generator indices (within the `Vec` above), indexed by the representative of each target
    /// they watch.
    pub generator_indices_by_watches: BTreeMap<usize, Vec<usize>>,
    /// Commitments to the constants polynomials and sigma polynomials.
    pub constants_sigmas_commitment: PolynomialBatch<F, C, D>,
    /// The transpose of the list of sigma polynomials.
    pub sigmas: Vec<Vec<F>>,
    /// Subgroup of order `degree`.
    pub subgroup: Vec<F>,
    /// Targets to be made public.
    pub public_inputs: Vec<Target>,
    /// A map from each `Target`'s index to the index of its representative in the disjoint-set
    /// forest.
    pub representative_map: Vec<usize>,
    /// Pre-computed roots for faster FFT.
    pub fft_root_table: Option<FftRootTable<F>>,
    /// A digest of the "circuit" (i.e. the instance, minus public inputs), which can be used to
    /// seed Fiat-Shamir.
    pub circuit_digest: <<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash,
    ///The concrete placement of the lookup gates for each lookup table index.
    pub lookup_rows: Vec<LookupWire>,
    /// A vector of (looking_in, looking_out) pairs for each lookup table index.
    pub lut_to_lookups: Vec<Lookup>,
    /// The generators watching each target, built on the first proof.
    #[cfg(feature = "std")]
    pub(crate) watch_table: crate::iop::generator::WatchTableCache,
    /// The rounds in which the generators finished, recorded on the first
    /// proof and replayed by later ones.
    #[cfg(feature = "std")]
    pub(crate) generator_schedule: crate::iop::generator::GeneratorSchedule,
    /// Reusable witness allocations, excluded from serialization and equality.
    #[cfg(feature = "std")]
    pub(crate) witness_buffers: crate::iop::witness::WitnessBuffers<F>,
    /// The device copy of the sigma values (column-major), uploaded by the
    /// first proof that computes the partial products on the GPU.
    #[cfg(feature = "gpu")]
    pub(crate) device_sigmas: crate::gpu::DeviceCache,
    /// The compiled witness program the caller installed, if any (see
    /// [`crate::plonk::witness_program`]); omitted from serialization and
    /// equality.
    #[cfg(all(feature = "std", feature = "rand_chacha"))]
    pub(crate) witness_program: crate::plonk::witness_program::region::ProgramCache,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    ProverOnlyCircuitData<F, C, D>
{
    pub fn to_bytes(
        &self,
        generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
        common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<Vec<u8>> {
        let mut buffer = Vec::new();
        buffer.write_prover_only_circuit_data(self, generator_serializer, common_data)?;
        Ok(buffer)
    }

    pub fn from_bytes(
        bytes: &[u8],
        generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
        common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<Self> {
        let mut buffer = Buffer::new(bytes);
        buffer.read_prover_only_circuit_data(generator_serializer, common_data)
    }
}

/// Circuit data required by the verifier, but not the prover.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct VerifierOnlyCircuitData<C: GenericConfig<D>, const D: usize> {
    /// A commitment to each constant polynomial and each permutation polynomial.
    pub constants_sigmas_cap: MerkleCap<C::F, C::Hasher>,
    /// A digest of the "circuit" (i.e. the instance, minus public inputs), which can be used to
    /// seed Fiat-Shamir.
    pub circuit_digest: <<C as GenericConfig<D>>::Hasher as Hasher<C::F>>::Hash,
}

impl<C: GenericConfig<D>, const D: usize> VerifierOnlyCircuitData<C, D> {
    pub fn to_bytes(&self) -> IoResult<Vec<u8>> {
        let mut buffer = Vec::new();
        buffer.write_verifier_only_circuit_data(self)?;
        Ok(buffer)
    }

    pub fn from_bytes(bytes: &[u8]) -> IoResult<Self> {
        let mut buffer = Buffer::new(bytes);
        buffer.read_verifier_only_circuit_data()
    }
}

/// Circuit data required by both the prover and the verifier.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct CommonCircuitData<F: RichField + Extendable<D>, const D: usize> {
    pub config: CircuitConfig,

    pub fri_params: FriParams,

    /// The types of gates used in this circuit, along with their prefixes.
    pub gates: Vec<GateRef<F, D>>,

    /// Information on the circuit's selector polynomials.
    pub selectors_info: SelectorsInfo,

    /// The degree of the PLONK quotient polynomial.
    pub quotient_degree_factor: usize,

    /// The largest number of constraints imposed by any gate.
    pub num_gate_constraints: usize,

    /// The number of constant wires.
    pub num_constants: usize,

    pub num_public_inputs: usize,

    /// The `{k_i}` valued used in `S_ID_i` in Plonk's permutation argument.
    pub k_is: Vec<F>,

    /// The number of partial products needed to compute the `Z` polynomials.
    pub num_partial_products: usize,

    /// The number of lookup polynomials.
    pub num_lookup_polys: usize,

    /// The number of lookup selectors.
    pub num_lookup_selectors: usize,

    /// The stored lookup tables.
    pub luts: Vec<LookupTable>,
}

impl<F: RichField + Extendable<D>, const D: usize> CommonCircuitData<F, D> {
    pub fn to_bytes(&self, gate_serializer: &dyn GateSerializer<F, D>) -> IoResult<Vec<u8>> {
        let mut buffer = Vec::new();
        buffer.write_common_circuit_data(self, gate_serializer)?;
        Ok(buffer)
    }

    pub fn from_bytes(bytes: &[u8], gate_serializer: &dyn GateSerializer<F, D>) -> IoResult<Self> {
        let mut buffer = Buffer::new(bytes);
        buffer.read_common_circuit_data(gate_serializer)
    }

    pub const fn degree_bits(&self) -> usize {
        self.fri_params.degree_bits
    }

    pub const fn degree(&self) -> usize {
        1 << self.degree_bits()
    }

    pub const fn lde_size(&self) -> usize {
        self.fri_params.lde_size()
    }

    pub fn lde_generator(&self) -> F {
        F::primitive_root_of_unity(self.degree_bits() + self.config.fri_config.rate_bits)
    }

    pub fn constraint_degree(&self) -> usize {
        self.gates
            .iter()
            .map(|g| g.0.degree())
            .max()
            .expect("No gates?")
    }

    pub const fn quotient_degree(&self) -> usize {
        self.quotient_degree_factor * self.degree()
    }

    /// Range of the constants polynomials in the `constants_sigmas_commitment`.
    pub const fn constants_range(&self) -> Range<usize> {
        0..self.num_constants
    }

    /// Range of the sigma polynomials in the `constants_sigmas_commitment`.
    pub const fn sigmas_range(&self) -> Range<usize> {
        self.num_constants..self.num_constants + self.config.num_routed_wires
    }

    /// Range of the `z`s polynomials in the `zs_partial_products_commitment`.
    pub const fn zs_range(&self) -> Range<usize> {
        0..self.config.num_challenges
    }

    /// Range of the partial products polynomials in the `zs_partial_products_lookup_commitment`.
    pub const fn partial_products_range(&self) -> Range<usize> {
        self.config.num_challenges..(self.num_partial_products + 1) * self.config.num_challenges
    }

    /// Range of lookup polynomials in the `zs_partial_products_lookup_commitment`.
    pub const fn lookup_range(&self) -> RangeFrom<usize> {
        self.num_zs_partial_products_polys()..
    }

    /// Range of lookup polynomials needed for evaluation at `g * zeta`.
    pub const fn next_lookup_range(&self, i: usize) -> Range<usize> {
        self.num_zs_partial_products_polys() + i * self.num_lookup_polys
            ..self.num_zs_partial_products_polys() + i * self.num_lookup_polys + 2
    }

    pub(crate) fn get_fri_instance(&self, zeta: F::Extension) -> FriInstanceInfo<F, D> {
        // All polynomials are opened at zeta.
        let zeta_batch = FriBatchInfo {
            point: zeta,
            polynomials: self.fri_all_polys(),
        };

        // The Z polynomials are also opened at g * zeta.
        let g = F::Extension::primitive_root_of_unity(self.degree_bits());
        let zeta_next = g * zeta;
        let zeta_next_batch = FriBatchInfo {
            point: zeta_next,
            polynomials: self.fri_next_batch_polys(),
        };

        let openings = vec![zeta_batch, zeta_next_batch];
        FriInstanceInfo {
            oracles: self.fri_oracles(),
            batches: openings,
        }
    }

    pub(crate) fn get_fri_instance_target(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        zeta: ExtensionTarget<D>,
    ) -> FriInstanceInfoTarget<D> {
        // All polynomials are opened at zeta.
        let zeta_batch = FriBatchInfoTarget {
            point: zeta,
            polynomials: self.fri_all_polys(),
        };

        // The Z polynomials are also opened at g * zeta.
        let g = F::primitive_root_of_unity(self.degree_bits());
        let zeta_next = builder.mul_const_extension(g, zeta);
        let zeta_next_batch = FriBatchInfoTarget {
            point: zeta_next,
            polynomials: self.fri_next_batch_polys(),
        };

        let openings = vec![zeta_batch, zeta_next_batch];
        FriInstanceInfoTarget {
            oracles: self.fri_oracles(),
            batches: openings,
        }
    }

    fn fri_oracles(&self) -> Vec<FriOracleInfo> {
        vec![
            FriOracleInfo {
                num_polys: self.num_preprocessed_polys(),
                blinding: PlonkOracle::CONSTANTS_SIGMAS.blinding,
            },
            FriOracleInfo {
                num_polys: self.config.num_wires,
                blinding: PlonkOracle::WIRES.blinding,
            },
            FriOracleInfo {
                num_polys: self.num_zs_partial_products_polys() + self.num_all_lookup_polys(),
                blinding: PlonkOracle::ZS_PARTIAL_PRODUCTS.blinding,
            },
            FriOracleInfo {
                num_polys: self.num_quotient_polys(),
                blinding: PlonkOracle::QUOTIENT.blinding,
            },
        ]
    }

    fn fri_preprocessed_polys(&self) -> Vec<FriPolynomialInfo> {
        FriPolynomialInfo::from_range(
            PlonkOracle::CONSTANTS_SIGMAS.index,
            0..self.num_preprocessed_polys(),
        )
    }

    pub(crate) const fn num_preprocessed_polys(&self) -> usize {
        self.sigmas_range().end
    }

    fn fri_wire_polys(&self) -> Vec<FriPolynomialInfo> {
        let num_wire_polys = self.config.num_wires;
        FriPolynomialInfo::from_range(PlonkOracle::WIRES.index, 0..num_wire_polys)
    }

    fn fri_zs_partial_products_polys(&self) -> Vec<FriPolynomialInfo> {
        FriPolynomialInfo::from_range(
            PlonkOracle::ZS_PARTIAL_PRODUCTS.index,
            0..self.num_zs_partial_products_polys(),
        )
    }

    pub(crate) const fn num_zs_partial_products_polys(&self) -> usize {
        self.config.num_challenges * (1 + self.num_partial_products)
    }

    /// Returns the total number of lookup polynomials.
    pub(crate) const fn num_all_lookup_polys(&self) -> usize {
        self.config.num_challenges * self.num_lookup_polys
    }
    fn fri_zs_polys(&self) -> Vec<FriPolynomialInfo> {
        FriPolynomialInfo::from_range(PlonkOracle::ZS_PARTIAL_PRODUCTS.index, self.zs_range())
    }

    /// Returns polynomials that require evaluation at `zeta` and `g * zeta`.
    fn fri_next_batch_polys(&self) -> Vec<FriPolynomialInfo> {
        [self.fri_zs_polys(), self.fri_lookup_polys()].concat()
    }

    fn fri_quotient_polys(&self) -> Vec<FriPolynomialInfo> {
        FriPolynomialInfo::from_range(PlonkOracle::QUOTIENT.index, 0..self.num_quotient_polys())
    }

    /// Returns the information for lookup polynomials, i.e. the index within the oracle and the indices of the polynomials within the commitment.
    fn fri_lookup_polys(&self) -> Vec<FriPolynomialInfo> {
        FriPolynomialInfo::from_range(
            PlonkOracle::ZS_PARTIAL_PRODUCTS.index,
            self.num_zs_partial_products_polys()
                ..self.num_zs_partial_products_polys() + self.num_all_lookup_polys(),
        )
    }
    pub(crate) const fn num_quotient_polys(&self) -> usize {
        self.config.num_challenges * self.quotient_degree_factor
    }

    fn fri_all_polys(&self) -> Vec<FriPolynomialInfo> {
        [
            self.fri_preprocessed_polys(),
            self.fri_wire_polys(),
            self.fri_zs_partial_products_polys(),
            self.fri_quotient_polys(),
            self.fri_lookup_polys(),
        ]
        .concat()
    }
}

/// The `Target` version of `VerifierCircuitData`, for use inside recursive circuits. Note that this
/// is intentionally missing certain fields, such as `CircuitConfig`, because we support only a
/// limited form of dynamic inner circuits. We can't practically make things like the wire count
/// dynamic, at least not without setting a maximum wire count and paying for the worst case.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VerifierCircuitTarget {
    /// A commitment to each constant polynomial and each permutation polynomial.
    pub constants_sigmas_cap: MerkleCapTarget,
    /// A digest of the "circuit" (i.e. the instance, minus public inputs), which can be used to
    /// seed Fiat-Shamir.
    pub circuit_digest: HashOutTarget,
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::field::types::Field;
    use crate::iop::witness::WitnessWrite;
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::config::PoseidonGoldilocksConfig;
    use crate::util::serialization::{DefaultGateSerializer, DefaultGeneratorSerializer};

    /// Caches filled by proving (the watch table, the recorded generator
    /// schedule, retained buffers) are not part of a circuit's value, so a
    /// proved circuit still equals its serialized round trip.
    #[test]
    fn proved_circuit_equals_its_round_trip() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let x = builder.add_virtual_target();
        let mut y = x;
        for _ in 0..64 {
            y = builder.mul_add(y, y, x);
        }
        let hash = builder.hash_n_to_hash_no_pad::<crate::hash::poseidon::PoseidonHash>(vec![y]);
        builder.register_public_inputs(&hash.elements);
        let data = builder.build::<C>();
        for _ in 0..2 {
            let mut pw = PartialWitness::new();
            pw.set_target(x, F::from_canonical_u64(3)).unwrap();
            data.verify(data.prove(pw).unwrap()).unwrap();
        }
        let gates = DefaultGateSerializer;
        let generators = DefaultGeneratorSerializer::<C, D>::default();
        let bytes = data.to_bytes(&gates, &generators).unwrap();
        let round_trip = CircuitData::<F, C, D>::from_bytes(&bytes, &gates, &generators).unwrap();
        assert!(round_trip == data);
    }
}
