//! The `PoseidonBN128Hash` kernels against the CPU: the BN254 arithmetic on
//! a grid of boundary values (including the lazily reduced range `[p, 2p)`,
//! and sums of four products at their largest)
//! and the permutation on both flavours, then trees and whole proofs.
use ark_ff::{BigInt, BigInteger, Field as _, PrimeField};
use num::BigUint;

use super::inversion::flavors;
use super::*;
use crate::gpu::poseidon_bn128 as kernels;
use crate::hash::poseidon_bn128::permutation::permute;
use crate::hash::poseidon_bn128::{Fr, PoseidonBN128Hash};
use crate::plonk::config::{GpuHash, PoseidonBN128GoldilocksConfig};

type Hb = PoseidonBN128Hash;
type Cb = PoseidonBN128GoldilocksConfig;

fn modulus() -> BigUint {
    BigUint::from_bytes_le(&Fr::MODULUS.to_bytes_le())
}

fn limbs(x: &BigUint) -> [u64; 4] {
    let mut limbs = [0; 4];
    for (limb, digit) in limbs.iter_mut().zip(x.iter_u64_digits()) {
        *limb = digit;
    }
    assert!(x.bits() <= 256);
    limbs
}

fn words(limbs: [u64; 4]) -> [u32; 8] {
    core::array::from_fn(|i| (limbs[i / 2] >> (32 * (i % 2))) as u32)
}

/// Montgomery words below 2p: integers near 0, p and powers of two below p,
/// and some of them plus p.
fn grid() -> Vec<BigUint> {
    let p = modulus();
    let one = BigUint::from(1u32);
    let mut values = (0..4u32).map(BigUint::from).collect::<Vec<_>>();
    values.extend((1..4u32).map(|k| &p - k));
    values.extend((0..254).map(|k| &one << k));
    values.push((&p - 1u32) >> 1);
    values.push((&p + 1u32) >> 1);
    values.push((&one << 256) % &p);
    let lazy = [
        BigUint::from(0u32),
        one.clone(),
        &p - 1u32,
        &one << 253,
        (&p - 1u32) >> 1,
    ];
    values.extend(lazy.iter().map(|v| v + &p));
    values.retain(|v| *v < &p * 2u32);
    values
}

/// The field element whose Montgomery form is `v` (or `v - p`).
fn element(v: &BigUint) -> Fr {
    let p = modulus();
    let v = if *v >= p { v - &p } else { v.clone() };
    Fr::new_unchecked(BigInt::new(limbs(&v)))
}

fn canonical_words(x: &Fr) -> [u32; 8] {
    words(x.into_bigint().0)
}

const TEST_KERNELS: &str = r#"
struct Params { count: u32, n: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputs: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputs: array<u32>;

fn load(k: u32) -> Fr {
    var x: Fr;
    for (var j = 0u; j < 8u; j++) { x[j] = inputs[8u * k + j]; }
    return x;
}

fn store(k: u32, x: Fr) {
    for (var j = 0u; j < 8u; j++) { outputs[8u * k + j] = x[j]; }
}

@compute @workgroup_size(64)
fn grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    if (k >= params.count) { return; }
    let a = load(k / params.n);
    let b = load(k % params.n);
    store(5u * k, from_mont(fr_mul(a, b)));
    store(5u * k + 1u, from_mont(fr_add(a, b)));
    store(5u * k + 2u, from_mont(exp5(a)));
    // Matrix entries are below p, state elements below 2p.
    let ca = sub_p_if_ge(a);
    let cb = sub_p_if_ge(b);
    store(5u * k + 3u, from_mont(fr_dot4(ca, cb, ca, cb, b, a, b, a)));
    store(5u * k + 4u, from_mont(fr_sqr(b)));
}

// Canonical integers in, canonical integers out.
@compute @workgroup_size(64)
fn permute_states(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    if (k >= params.count) { return; }
    let b = 4u * k;
    let s = permute(array<Fr, 4>(to_mont(load(b)), to_mont(load(b + 1u)), to_mont(load(b + 2u)), to_mont(load(b + 3u))));
    store(b, from_mont(s[0]));
    store(b + 1u, from_mont(s[1]));
    store(b + 2u, from_mont(s[2]));
    store(b + 3u, from_mont(s[3]));
}
"#;

/// Runs `entry` of the test kernels over `count` invocations of `input`
/// words, returning `out_words` words.
fn run(
    ctx: &GpuContext,
    flavor: Flavor,
    entry: &str,
    input: &[u32],
    count: usize,
    n: usize,
    out_words: usize,
) -> Vec<u32> {
    let layout = storage_layout(
        &ctx.device,
        "bn128 test",
        &[
            (true, true),
            (false, true),
            (false, false),
            (false, false),
            (false, false),
            (true, true),
        ],
    );
    let source = kernels::permutation_source(flavor) + TEST_KERNELS;
    let module = module(&ctx.device, "bn128 test", &source);
    let pipeline = pipeline(&ctx.device, &layout, &module, entry);
    let input = ctx.upload(
        "bn128 test inputs",
        &input
            .chunks(2)
            .map(|w| u64::from(w[0]) | (u64::from(w[1]) << 32))
            .collect::<Vec<_>>(),
    );
    let output = ctx.storage("bn128 test outputs", (out_words * 4) as u64);
    let params = ctx.uniform("bn128 test params", &[count as u32, n as u32, 0, 0]);
    let constants = ctx.uniform("bn128 constants", &kernels::constant_words());
    let bindings = ctx.bind(
        &layout,
        &[
            &params,
            &input,
            &output,
            &ctx.dummy_rw,
            &ctx.dummy_rw,
            &constants,
        ],
    );
    let mut encoder = ctx.encoder("bn128 test");
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
    }
    ctx.submit(encoder);
    ctx.read(&output, (out_words * 4) as u64, |bytes| {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect()
    })
}

#[test]
fn bn128_shaders_validate() {
    for flavor in [Flavor::Native, Flavor::Portable] {
        let capabilities = match flavor {
            Flavor::Native => naga::valid::Capabilities::SHADER_INT64,
            Flavor::Portable => naga::valid::Capabilities::empty(),
        };
        let source = kernels::hash_module(flavor);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
            .validate(&module)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
    }
}

#[test]
fn bn128_arithmetic_matches_cpu_on_boundary_grid() {
    let Some(ctx) = ctx() else { return };
    let values = grid();
    let n = values.len();
    let count = n * n;
    let input = values
        .iter()
        .flat_map(|v| words(limbs(v)))
        .collect::<Vec<_>>();
    let elements = values.iter().map(element).collect::<Vec<_>>();
    let expected = (0..count)
        .flat_map(|k| {
            let (a, b) = (elements[k / n], elements[k % n]);
            [a * b, a + b, a.pow([5]), a * b * Fr::from(4u64), b.square()]
                .map(|x| canonical_words(&x))
        })
        .collect::<Vec<_>>();
    for flavor in flavors(&ctx) {
        let actual = run(&ctx, flavor, "grid", &input, count, n, count * 5 * 8);
        for (i, (got, want)) in actual.as_chunks::<8>().0.iter().zip(&expected).enumerate() {
            let (k, op) = (i / 5, i % 5);
            assert_eq!(
                got,
                want,
                "{flavor:?} op {op} on {:#x}, {:#x}",
                values[k / n],
                values[k % n]
            );
        }
    }
}

#[test]
fn bn128_permutation_matches_cpu() {
    let Some(ctx) = ctx() else { return };
    let p = modulus();
    let mut states: Vec<[Fr; 4]> = vec![
        [Fr::from(0u64); 4],
        core::array::from_fn(|i| Fr::from(i as u64)),
        [element(&(&p - 1u32)); 4],
    ];
    let mut rng = StdRng::seed_from_u64(0x424e313238);
    for _ in 0..256 {
        states.push(core::array::from_fn(|_| {
            let mut bytes = [0u8; 32];
            rng.fill_bytes(&mut bytes);
            Fr::from_le_bytes_mod_order(&bytes)
        }));
    }
    let input = states
        .iter()
        .flatten()
        .flat_map(canonical_words)
        .collect::<Vec<_>>();
    let expected = states
        .iter()
        .flat_map(|s| {
            let mut s = *s;
            permute(&mut s);
            s.map(|x| canonical_words(&x))
        })
        .collect::<Vec<_>>();
    for flavor in flavors(&ctx) {
        let actual = run(
            &ctx,
            flavor,
            "permute_states",
            &input,
            states.len(),
            0,
            input.len(),
        );
        for (i, (got, want)) in actual.as_chunks::<8>().0.iter().zip(&expected).enumerate() {
            assert_eq!(got, want, "{flavor:?} state {}", i / 4);
        }
    }
}

#[test]
fn bn128_merkle_trees_match_cpu() {
    let Some(ctx) = ctx() else { return };
    let ctx = &*ctx;
    assert!(merkle::supported_on::<F, Hb>(ctx));
    let mut rng = StdRng::seed_from_u64(0x424e3132384d);
    for log_n in [0u32, 1, 2, 4, 7, 10] {
        let n = 1usize << log_n;
        for leaf_len in [1usize, 2, 3, 4, 5, 8, 9, 10, 12, 16, 17, 18, 19, 135] {
            if log_n >= 10 && leaf_len > 17 {
                continue;
            }
            let mut leaves: Vec<Vec<F>> =
                (0..n).map(|_| random_fields(&mut rng, leaf_len)).collect();
            // Noncanonical and boundary words in the first leaves.
            for (leaf, &w) in leaves.iter_mut().zip(boundary_words().iter()) {
                leaf[0] = F::from_noncanonical_u64(w);
                if leaf_len > 1 {
                    leaf[leaf_len - 1] = F::from_noncanonical_u64(w ^ 0x5555);
                }
            }
            for cap_height in [0, 1, log_n.saturating_sub(1), log_n] {
                let cap_height = cap_height.min(log_n);
                let flat = MerkleLeaves::from_rows(leaves.clone());
                let expected = MerkleTree::<F, Hb>::new_cpu(flat.clone(), cap_height as usize);
                let (digests, cap) =
                    merkle::build_from_rows::<F, Hb>(ctx, &flat, log_n, cap_height);
                let what = format!("log_n={log_n} leaf_len={leaf_len} cap_height={cap_height}");
                assert_eq!(digests.as_slice(), &*expected.digests, "{what}");
                assert_eq!(cap, expected.cap.0, "{what}");
            }
        }
    }
}

/// A circuit that verifies a Poseidon proof, proved with the BN128 config, as
/// a wrapper for a BN254 SNARK would be.
fn wrap_circuit() -> (
    crate::plonk::circuit_data::CircuitData<F, Cb, 2>,
    crate::iop::witness::PartialWitness<F>,
) {
    use crate::iop::witness::PartialWitness;
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::PoseidonGoldilocksConfig;
    let (inner, pw) =
        hash_chain_circuit_with_config(1 << 8, CircuitConfig::standard_recursion_config());
    let inner_proof = inner.prove(pw).unwrap();
    let mut builder = CircuitBuilder::<F, 2>::new(CircuitConfig::standard_recursion_config());
    let proof = builder.add_virtual_proof_with_pis(&inner.common);
    let verifier = builder.constant_verifier_data(&inner.verifier_only);
    builder.verify_proof::<PoseidonGoldilocksConfig>(&proof, &verifier, &inner.common);
    builder.register_public_inputs(&proof.public_inputs);
    let mut pw = PartialWitness::new();
    crate::iop::witness::WitnessWrite::set_proof_with_pis_target(&mut pw, &proof, &inner_proof)
        .unwrap();
    (builder.build::<Cb>(), pw)
}

/// The verifier recomputes every opened leaf's hash and Merkle path on the
/// CPU, so a GPU proof that verifies has matching digests along every query.
/// Proofs themselves are not compared: the unused wires take random values.
#[test]
fn bn128_wrap_proofs_verify_on_cpu() {
    let _guard = ENABLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(ctx) = ctx() else { return };
    assert!(merkle::supported_on::<F, Hb>(&ctx));
    assert_eq!(
        <Hb as crate::plonk::config::Hasher<F>>::GPU_HASH,
        GpuHash::PoseidonBn128
    );
    // Its sponge is the Goldilocks Poseidon one, so the proof-of-work search
    // runs on the device too.
    assert_eq!(
        crate::gpu::pow::permutation::<F, Hb>(),
        crate::hash::hashing::GpuPermutation::Poseidon
    );
    let (gpu_data, pw) = wrap_circuit();
    assert!(
        gpu_data.common.degree_bits() + gpu_data.common.config.fri_config.rate_bits
            >= crate::gpu::MIN_LOG_LEAVES as usize
    );
    let gpu_proof = gpu_data.prove(pw.clone()).unwrap();
    set_enabled(false);
    let (cpu_data, _) = wrap_circuit();
    let cpu_proof = cpu_data.prove(pw).unwrap();
    set_enabled(true);
    // The constants and sigmas commitment was built on each side.
    assert_eq!(gpu_data.verifier_only, cpu_data.verifier_only);
    assert_eq!(gpu_data.common, cpu_data.common);
    cpu_data.verify(gpu_proof.clone()).unwrap();
    gpu_data.verify(cpu_proof).unwrap();
    let mut corrupted = gpu_proof;
    corrupted.proof.wires_cap.0[0] = corrupted.proof.plonk_zs_partial_products_cap.0[0];
    assert!(cpu_data.verify(corrupted).is_err());
}
