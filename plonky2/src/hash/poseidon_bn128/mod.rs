//! Poseidon over the BN254 scalar field as a [`Hasher`] for Goldilocks
//! values, for proofs that are verified inside a BN254 SNARK (a Groth16 or
//! PLONK wrapper, for example, verified on Ethereum).
//!
//! The permutation is iden3's Poseidon of width 4 (rate 3, 8 full and 56
//! partial rounds, `x^5` S-box). The sponge packs three canonical Goldilocks
//! elements into the low 64-bit limbs of each rate element, so a permutation
//! absorbs nine Goldilocks elements; the digest is the first state element.
//! [`Hasher::Permutation`] stays the Goldilocks [`PoseidonPermutation`], so
//! the challenger and the proof-of-work are those of [`PoseidonHash`].
//!
//! Adapted from succinctx's plonky2x wrapper
//! (<https://github.com/succinctlabs/succinctx/tree/main/plonky2x/core/src/backend/wrapper>).
//!
//! [`PoseidonHash`]: crate::hash::poseidon::PoseidonHash

mod constants;
pub mod permutation;

use core::fmt;
use core::marker::PhantomData;

use ark_ff::{BigInt, PrimeField};
use num::BigUint;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::field::types::Field;
use crate::hash::hash_types::RichField;
use crate::hash::poseidon::PoseidonPermutation;
use crate::plonk::config::{GenericHashOut, Hasher};

/// The BN254 scalar field. arkworks' Montgomery arithmetic uses x86_64
/// assembly when built with the `bmi2` and `adx` target features.
pub type Fr = ark_bn254::Fr;

/// Goldilocks elements packed into one field element.
pub const GOLDILOCKS_ELEMENTS: usize = 3;

/// The field element with these little-endian 64-bit limbs, if it is below
/// the modulus.
fn from_limbs(limbs: [u64; 4]) -> Option<Fr> {
    Fr::from_bigint(BigInt::new(limbs))
}

/// The field element with this canonical little-endian encoding, if it is
/// below the modulus.
fn from_le_bytes(bytes: &[u8; 32]) -> Option<Fr> {
    let limbs = bytes.as_chunks::<8>().0;
    from_limbs(core::array::from_fn(|i| u64::from_le_bytes(limbs[i])))
}

/// The canonical little-endian encoding.
fn to_le_bytes(x: &Fr) -> [u8; 32] {
    let mut bytes = [0; 32];
    for (chunk, limb) in bytes
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.into_bigint().0)
    {
        *chunk = limb.to_le_bytes();
    }
    bytes
}

/// A digest: one BN254 scalar. Its bytes are the canonical little-endian
/// encoding; serde writes it as a decimal string, as gnark-plonky2-verifier
/// reads it.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PoseidonBN128HashOut<F: Field> {
    pub value: Fr,
    _phantom: PhantomData<F>,
}

impl<F: Field> PoseidonBN128HashOut<F> {
    pub const fn new(value: Fr) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }
}

impl<F: RichField> GenericHashOut<F> for PoseidonBN128HashOut<F> {
    fn to_bytes(&self) -> Vec<u8> {
        to_le_bytes(&self.value).to_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let bytes: &[u8; 32] = bytes.try_into().unwrap();
        Self::new(from_le_bytes(bytes).unwrap())
    }

    fn to_vec(&self) -> Vec<F> {
        to_le_bytes(&self.value)
            // Chunks of 7 bytes since 8 bytes would allow collisions.
            .chunks(7)
            .map(|bytes| {
                let mut arr = [0; 8];
                arr[..bytes.len()].copy_from_slice(bytes);
                F::from_canonical_u64(u64::from_le_bytes(arr))
            })
            .collect()
    }
}

impl<F: Field> Serialize for PoseidonBN128HashOut<F> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let value = BigUint::from_bytes_le(&to_le_bytes(&self.value));
        serializer.serialize_str(&value.to_str_radix(10))
    }
}

impl<'de, F: Field> Deserialize<'de> for PoseidonBN128HashOut<F> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct DecimalVisitor;

        impl serde::de::Visitor<'_> for DecimalVisitor {
            type Value = Fr;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a decimal string of an element of the BN254 scalar field")
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Fr, E> {
                let value = BigUint::parse_bytes(v.as_bytes(), 10)
                    .ok_or_else(|| E::custom("not a decimal integer"))?;
                let mut bytes = value.to_bytes_le();
                if bytes.len() > 32 {
                    return Err(E::custom("not below the BN254 scalar field modulus"));
                }
                bytes.resize(32, 0);
                from_le_bytes(bytes.as_slice().try_into().unwrap())
                    .ok_or_else(|| E::custom("not below the BN254 scalar field modulus"))
            }
        }

        deserializer.deserialize_str(DecimalVisitor).map(Self::new)
    }
}

/// Poseidon over BN254 for Merkle trees of Goldilocks values.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PoseidonBN128Hash;

impl<F: RichField> Hasher<F> for PoseidonBN128Hash {
    const HASH_SIZE: usize = 32;
    type Hash = PoseidonBN128HashOut<F>;
    type Permutation = PoseidonPermutation<F>;

    fn hash_no_pad(input: &[F]) -> Self::Hash {
        let mut state = [Fr::from(0u64); permutation::WIDTH];
        // Overwrite mode: a partial final chunk leaves the remaining rate
        // elements as they were.
        for rate_chunk in input.chunks(permutation::RATE * GOLDILOCKS_ELEMENTS) {
            for (j, chunk) in rate_chunk.chunks(GOLDILOCKS_ELEMENTS).enumerate() {
                // Up to three canonical Goldilocks elements as the low limbs;
                // below 2^192.
                let mut limbs = [0; 4];
                for (limb, x) in limbs.iter_mut().zip(chunk) {
                    *limb = x.to_canonical_u64();
                }
                state[j + 1] = from_limbs(limbs).unwrap();
            }
            permutation::permute(&mut state);
        }
        PoseidonBN128HashOut::new(state[0])
    }

    fn hash_pad(input: &[F]) -> Self::Hash {
        let mut padded_input = input.to_vec();
        padded_input.push(F::ONE);
        while !(padded_input.len() + 1).is_multiple_of(permutation::RATE * GOLDILOCKS_ELEMENTS) {
            padded_input.push(F::ZERO);
        }
        padded_input.push(F::ONE);
        Self::hash_no_pad(&padded_input)
    }

    fn hash_or_noop(inputs: &[F]) -> Self::Hash {
        if inputs.len() <= GOLDILOCKS_ELEMENTS {
            let mut bytes = [0u8; 32];
            for (chunk, x) in bytes.as_chunks_mut::<8>().0.iter_mut().zip(inputs) {
                *chunk = x.to_canonical_u64().to_le_bytes();
            }
            PoseidonBN128HashOut::new(from_le_bytes(&bytes).unwrap())
        } else {
            Self::hash_no_pad(inputs)
        }
    }

    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
        let zero = Fr::from(0u64);
        let mut state = [zero, zero, left.value, right.value];
        permutation::permute(&mut state);
        PoseidonBN128HashOut::new(state[0])
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::hash::merkle_proofs::verify_merkle_proof_to_cap;
    use crate::hash::merkle_tree::MerkleTree;
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, PoseidonBN128GoldilocksConfig};

    type F = GoldilocksField;
    type H = PoseidonBN128Hash;

    /// `n` pseudo-random elements (splitmix64), noncanonical inputs included.
    fn elements(seed: u64, n: usize) -> Vec<F> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x.wrapping_add(0x9e3779b97f4a7c15);
                let mut z = x;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                F::from_noncanonical_u64(z ^ (z >> 31))
            })
            .collect()
    }

    fn dec(h: PoseidonBN128HashOut<F>) -> String {
        serde_json::to_string(&h).unwrap()
    }

    /// Hash outputs over fixed inputs, covering every `Hasher` entry point and
    /// a Merkle tree with the leaf width of pod2-onchain's wrap (135 wires).
    fn vectors() -> Vec<(String, String)> {
        let mut v = vec![];
        for n in [0, 1, 2, 3, 4, 8, 9, 10, 20, 135] {
            v.push((
                format!("hash_no_pad/{n}"),
                dec(H::hash_no_pad(&elements(n as u64, n))),
            ));
            v.push((
                format!("hash_pad/{n}"),
                dec(H::hash_pad(&elements(n as u64, n))),
            ));
            v.push((
                format!("hash_or_noop/{n}"),
                dec(H::hash_or_noop(&elements(n as u64, n))),
            ));
        }
        let a = H::hash_no_pad(&elements(100, 9));
        let b = H::hash_no_pad(&elements(101, 9));
        v.push(("two_to_one".into(), dec(H::two_to_one(a, b))));
        let leaves: Vec<Vec<F>> = (0..64).map(|i| elements(1000 + i, 135)).collect();
        let tree = MerkleTree::<F, H>::new(leaves, 2);
        for (i, c) in tree.cap.0.iter().enumerate() {
            v.push((format!("merkle_cap/{i}"), dec(*c)));
        }
        v
    }

    /// Outputs of the original succinctx-derived implementation.
    const GOLDEN: &[(&str, &str)] = &[
        ("hash_no_pad/0", "0"),
        (
            "hash_pad/0",
            "7088736580417989987124950536618348279576591433150501589226077525804660367082",
        ),
        ("hash_or_noop/0", "0"),
        (
            "hash_no_pad/1",
            "3703233135996724890779182454950263444598597611238841167188852493430971240817",
        ),
        (
            "hash_pad/1",
            "19274843143559544788113981955289003389138651291064227186783406318288069556917",
        ),
        ("hash_or_noop/1", "10451216379200822465"),
        (
            "hash_no_pad/2",
            "20764591326141118096869830754740281893875016672372700732251896820888062709887",
        ),
        (
            "hash_pad/2",
            "14310404417822983731949141736960644143539778957254694649240928915358590946543",
        ),
        ("hash_or_noop/2", "254922427606657880409455128571624773326"),
        (
            "hash_no_pad/3",
            "14388808131642276644996753961658578594564418003646700700264923771533118354544",
        ),
        (
            "hash_pad/3",
            "8834461814263468374312047504295098936336228422330063038458947002172812387141",
        ),
        (
            "hash_or_noop/3",
            "3847704443561515879326586667961746175955476545095883395053",
        ),
        (
            "hash_no_pad/4",
            "5412760131885201200094236781764806057490409457839695281779881073553600211015",
        ),
        (
            "hash_pad/4",
            "3216793234310421447666590927292375522167674046731221372913764676064532075693",
        ),
        (
            "hash_or_noop/4",
            "5412760131885201200094236781764806057490409457839695281779881073553600211015",
        ),
        (
            "hash_no_pad/8",
            "7592464771819052107709830378007477125030241743003341734234243468744389318397",
        ),
        (
            "hash_pad/8",
            "15176545645817954520167355552224322658994077441691757561490836029971371542297",
        ),
        (
            "hash_or_noop/8",
            "7592464771819052107709830378007477125030241743003341734234243468744389318397",
        ),
        (
            "hash_no_pad/9",
            "16644072240539715981429925156970839105826605616129121239064731376421269244614",
        ),
        (
            "hash_pad/9",
            "10500109315557341425799007752368722011279243113454804977320236925762719950869",
        ),
        (
            "hash_or_noop/9",
            "16644072240539715981429925156970839105826605616129121239064731376421269244614",
        ),
        (
            "hash_no_pad/10",
            "9622684325003536055532174195279717073962993138010978332930834260146488906912",
        ),
        (
            "hash_pad/10",
            "17807852705356247327622956914958074236262092335866858569122905365302068720478",
        ),
        (
            "hash_or_noop/10",
            "9622684325003536055532174195279717073962993138010978332930834260146488906912",
        ),
        (
            "hash_no_pad/20",
            "10668638861053567169722477228106982799705718778589413472216968864653148341219",
        ),
        (
            "hash_pad/20",
            "21131668499977879462129866404726778969878401958869591706857150088594587193909",
        ),
        (
            "hash_or_noop/20",
            "10668638861053567169722477228106982799705718778589413472216968864653148341219",
        ),
        (
            "hash_no_pad/135",
            "531443502252665886826194023491211015202046062876928284149521898540761406319",
        ),
        (
            "hash_pad/135",
            "2872766500848876662150261043516529673972202380421880849991131544468402160070",
        ),
        (
            "hash_or_noop/135",
            "531443502252665886826194023491211015202046062876928284149521898540761406319",
        ),
        (
            "two_to_one",
            "20660007035825148122810889377972755179365198909974219802059056074710323060651",
        ),
        (
            "merkle_cap/0",
            "6977713388666859931039231015563642232257250851003434983887285609439772760042",
        ),
        (
            "merkle_cap/1",
            "3525787546931182109049652867067237886313426778229132204478289005238331728632",
        ),
        (
            "merkle_cap/2",
            "2320624901615231013384481038241311877512163341378141384755349846615845498249",
        ),
        (
            "merkle_cap/3",
            "18440043361982931388036710836148135504538769166339506241526770549120013267963",
        ),
    ];

    #[test]
    fn hash_vectors() {
        let got = vectors();
        assert_eq!(got.len(), GOLDEN.len());
        for ((k, h), (gk, gh)) in got.iter().zip(GOLDEN) {
            assert_eq!(k, gk);
            assert_eq!(h, &format!("\"{gh}\""), "{k}");
        }
    }

    #[test]
    fn hash_out_round_trips() {
        let h = H::hash_no_pad(&elements(7, 20));
        assert_eq!(
            PoseidonBN128HashOut::<F>::from_bytes(&GenericHashOut::<F>::to_bytes(&h)),
            h
        );
        let json = dec(h);
        assert_eq!(
            serde_json::from_str::<PoseidonBN128HashOut<F>>(&json).unwrap(),
            h
        );
        // The modulus and anything larger are rejected.
        let modulus =
            "\"21888242871839275222246405745257275088548364400416034343698204186575808495617\"";
        assert!(serde_json::from_str::<PoseidonBN128HashOut<F>>(modulus).is_err());
        assert!(
            serde_json::from_str::<PoseidonBN128HashOut<F>>(&format!("\"1{}", &modulus[1..]))
                .is_err()
        );
        assert!(serde_json::from_str::<PoseidonBN128HashOut<F>>("\"0x1\"").is_err());
    }

    #[test]
    fn merkle_proofs_verify() -> Result<()> {
        for (log_n, cap_height) in [(8, 1), (8, 8)] {
            let leaves: Vec<Vec<F>> = (0..1u64 << log_n).map(|i| elements(i, 7)).collect();
            let tree = MerkleTree::<F, H>::new(leaves.clone(), cap_height);
            for (i, leaf) in leaves.into_iter().enumerate() {
                verify_merkle_proof_to_cap(leaf, i, &tree.cap, &tree.prove(i))?;
            }
        }
        Ok(())
    }

    #[test]
    #[should_panic]
    fn cap_height_too_big() {
        let leaves: Vec<Vec<F>> = (0..1 << 8).map(|i| elements(i, 7)).collect();
        let _ = MerkleTree::<F, H>::new(leaves, 9);
    }

    #[test]
    fn prove_and_verify() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonBN128GoldilocksConfig;
        let mut builder = CircuitBuilder::<<C as GenericConfig<D>>::F, D>::new(
            CircuitConfig::standard_recursion_config(),
        );
        let x = builder.add_virtual_target();
        let mut y = x;
        for _ in 0..1000 {
            y = builder.mul_add(y, y, x);
        }
        builder.register_public_input(y);
        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();
        pw.set_target(x, F::from_canonical_u64(3))?;
        let proof = data.prove(pw)?;
        data.verify(proof)
    }
}
