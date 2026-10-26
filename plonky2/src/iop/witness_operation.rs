//! Declarative witness semantics. Kept separate so the program-format
//! fingerprint covers target expansion without unrelated scheduler changes.
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::field::types::Field;
use crate::iop::target::Target;

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
