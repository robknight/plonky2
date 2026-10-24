//! Merkle trees hashed with [`PoseidonBN128Hash`]: the leaf and node kernels
//! of [`hash_module`], with the same parameters, bindings and digest layout as
//! the Goldilocks Poseidon kernels plus a uniform buffer of round constants.
//!
//! BN254 scalars are eight little-endian 32-bit words in Montgomery form with
//! `R = 2^256`, the representation arkworks uses (so the constants are copied
//! word for word). Multiplication is the CIOS Montgomery product without the
//! final subtraction: since `4p < R`, operands below `2p` give a product
//! below `2p`, and sums are brought back below `2p` with one conditional
//! subtraction. Digests are made canonical before they are written.
//!
//! The rounds are loops over constants read from the uniform buffer, so the
//! kernel holds one copy of each round's code rather than all 64 rounds
//! unrolled.
//!
//! [`PoseidonBN128Hash`]: crate::hash::poseidon_bn128::PoseidonBN128Hash

use core::fmt::Write;

use ark_ff::{BigInteger, PrimeField};

use crate::gpu::shader::{self, Flavor};
use crate::gpu::{module, pipeline, storage_layout, GpuContext};
use crate::hash::poseidon_bn128::constants::CONSTANTS;
use crate::hash::poseidon_bn128::permutation::{FULL_ROUNDS, PARTIAL_ROUNDS, WIDTH};
use crate::hash::poseidon_bn128::Fr;

/// Words of a field element.
const WORDS: usize = 8;

/// The compiled leaf and node kernels and their round constants.
pub(crate) struct TreeKernels {
    pub layout: wgpu::BindGroupLayout,
    pub leaves: wgpu::ComputePipeline,
    pub nodes: wgpu::ComputePipeline,
    /// Binding 5: [`constant_words`].
    pub constants: wgpu::Buffer,
}

impl core::fmt::Debug for TreeKernels {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("TreeKernels")
    }
}

/// Compiles the kernels, or `None` (with a warning) if the device rejects
/// them; the trees are then built on the CPU.
pub(crate) fn compile(ctx: &GpuContext) -> Option<TreeKernels> {
    let start = std::time::Instant::now();
    let validation = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let layout = storage_layout(
        &ctx.device,
        "poseidon bn128 hash",
        &[
            (true, true),
            (false, true),
            (false, false),
            (false, false),
            (false, false),
            (true, true),
        ],
    );
    let module = module(&ctx.device, "poseidon bn128 hash", &hash_module(ctx.flavor));
    let leaves = pipeline(&ctx.device, &layout, &module, "hash_leaves");
    let nodes = pipeline(&ctx.device, &layout, &module, "hash_nodes");
    let constants = ctx.uniform("poseidon bn128 constants", &constant_words());
    if let Some(error) = pollster::block_on(validation.pop()) {
        log::warn!("gpu: Poseidon BN128 kernels rejected, hashing on the CPU: {error}");
        return None;
    }
    log::debug!(
        "gpu: Poseidon BN128 kernels compiled in {:?}",
        start.elapsed()
    );
    Some(TreeKernels {
        layout,
        leaves,
        nodes,
        constants,
    })
}

/// Field elements in the uniform buffer: the round constants, the sparse
/// partial-round matrices, then the transposed MDS and pre-partial matrices.
const C_AT: usize = 0;
const S_AT: usize = C_AT + 88;
const M_AT: usize = S_AT + 392;
const P_AT: usize = M_AT + WIDTH * WIDTH;
const CONSTANTS_LEN: usize = P_AT + WIDTH * WIDTH;

/// The Montgomery words of a field element.
fn montgomery_words(x: &Fr) -> [u32; WORDS] {
    let limbs = x.0 .0;
    core::array::from_fn(|i| (limbs[i / 2] >> (32 * (i % 2))) as u32)
}

/// The words of an integer below 2^256.
fn integer_words(limbs: [u64; 4]) -> [u32; WORDS] {
    core::array::from_fn(|i| (limbs[i / 2] >> (32 * (i % 2))) as u32)
}

/// The uniform buffer's contents, `CONSTANTS_LEN` elements in Montgomery form.
pub(crate) fn constant_words() -> Vec<u32> {
    let k = &*CONSTANTS;
    let words =
        k.c.iter()
            .chain(&k.s)
            .chain(k.m_t.iter().flatten())
            .chain(k.p_t.iter().flatten())
            .flat_map(montgomery_words)
            .collect::<Vec<_>>();
    assert_eq!(words.len(), CONSTANTS_LEN * WORDS);
    words
}

/// `-p^-1 mod 2^32`.
fn neg_inverse_word(p0: u32) -> u32 {
    // Newton's iteration doubles the correct low bits from 1 (p0 is odd).
    let mut inv = 1u32;
    for _ in 0..5 {
        inv = inv.wrapping_mul(2u32.wrapping_sub(p0.wrapping_mul(inv)));
    }
    assert_eq!(p0.wrapping_mul(inv), 1);
    inv.wrapping_neg()
}

fn fr_literal(words: [u32; WORDS]) -> String {
    let words = words.map(|w| format!("{w:#010x}u"));
    format!("Fr({})", words.join(", "))
}

/// `t + a * b + c` as a low and a high word.
const NATIVE_MAC: &str = r#"
fn mac(t: u32, a: u32, b: u32, c: u32) -> vec2<u32> {
    let r = u64(a) * u64(b) + u64(t) + u64(c);
    return vec2<u32>(u32(r), u32(r >> 32u));
}
"#;

const PORTABLE_MAC: &str = r#"
fn mac(t: u32, a: u32, b: u32, c: u32) -> vec2<u32> {
    let p = mulwide32(a, b);
    let s = p.x + t;
    let c1 = u32(s < t);
    let lo = s + c;
    let c2 = u32(lo < c);
    return vec2<u32>(lo, p.y + c1 + c2);
}
"#;

/// The field arithmetic: `fr_mul`, `fr_add`, `to_mont`, `from_mont`.
fn arithmetic(out: &mut String, flavor: Flavor) {
    let p = integer_words(Fr::MODULUS.0);
    let mut two_p = Fr::MODULUS;
    two_p.mul2();
    let two_p = integer_words(two_p.0);
    let ninv = neg_inverse_word(p[0]);
    // The integer R^2 mod p is the Montgomery form of R mod p.
    let r2 = montgomery_words(&Fr::from_bigint(Fr::from(1u64).0).unwrap());

    out.push_str(
        "\n// A BN254 scalar: eight little-endian words, Montgomery form, below 2p.\nalias Fr = array<u32, 8>;\n",
    );
    out.push_str(match flavor {
        Flavor::Native => NATIVE_MAC,
        Flavor::Portable => PORTABLE_MAC,
    });

    // CIOS without the final subtraction; the top word of p is below 2^31 - 1,
    // so the running sum needs no ninth word.
    out.push_str("\nfn fr_mul(a: Fr, b: Fr) -> Fr {\n");
    for j in 0..WORDS {
        writeln!(out, "    var t{j} = 0u;").unwrap();
    }
    out.push_str("    var x: vec2<u32>;\n    var y: vec2<u32>;\n    var m: u32;\n");
    for i in 0..WORDS {
        writeln!(out, "    x = mac(t0, a[0], b[{i}], 0u); t0 = x.x;").unwrap();
        writeln!(out, "    m = t0 * {ninv:#010x}u;").unwrap();
        writeln!(out, "    y = mac(t0, m, {:#010x}u, 0u);", p[0]).unwrap();
        for j in 1..WORDS {
            writeln!(out, "    x = mac(t{j}, a[{j}], b[{i}], x.y); t{j} = x.x;").unwrap();
            writeln!(
                out,
                "    y = mac(t{j}, m, {:#010x}u, y.y); t{} = y.x;",
                p[j],
                j - 1
            )
            .unwrap();
        }
        writeln!(out, "    t{} = y.y + x.y;", WORDS - 1).unwrap();
    }
    out.push_str("    return Fr(t0, t1, t2, t3, t4, t5, t6, t7);\n}\n");

    // a - q if a >= q, for a below 2^256.
    let sub_if_ge = |out: &mut String, name: &str, q: [u32; WORDS]| {
        writeln!(out, "\nfn {name}(a: Fr) -> Fr {{").unwrap();
        out.push_str("    var d: Fr;\n    var borrow = 0u;\n");
        for j in 0..WORDS {
            writeln!(
                out,
                "    {{ let s = a[{j}] - {:#010x}u; let b1 = u32(a[{j}] < {:#010x}u); d[{j}] = s - borrow; borrow = b1 | u32(s < borrow); }}",
                q[j], q[j]
            )
            .unwrap();
        }
        out.push_str("    if (borrow != 0u) { return a; }\n    return d;\n}\n");
    };
    sub_if_ge(out, "sub_2p_if_ge", two_p);
    sub_if_ge(out, "sub_p_if_ge", p);

    // a + b for a, b below 2p: the sum is below 4p < 2^256.
    out.push_str("\nfn fr_add(a: Fr, b: Fr) -> Fr {\n    var s: Fr;\n    var carry = 0u;\n");
    for j in 0..WORDS {
        writeln!(
            out,
            "    {{ let w = a[{j}] + b[{j}]; let c1 = u32(w < a[{j}]); s[{j}] = w + carry; carry = c1 | u32(s[{j}] < w); }}"
        )
        .unwrap();
    }
    out.push_str("    return sub_2p_if_ge(s);\n}\n");

    writeln!(
        out,
        "\n// Montgomery form of an integer below 2^256.\nfn to_mont(a: Fr) -> Fr {{ return fr_mul(a, {}); }}",
        fr_literal(r2)
    )
    .unwrap();
    out.push_str(
        "\n// The canonical integer: below 2p * 1 / R + p, so at most p.\nfn from_mont(a: Fr) -> Fr { return sub_p_if_ge(fr_mul(a, Fr(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u))); }\n",
    );
}

/// The permutation over constants from the uniform buffer, as
/// `hash::poseidon_bn128::permutation::permute`.
fn permutation(out: &mut String) {
    assert_eq!((WIDTH, FULL_ROUNDS, PARTIAL_ROUNDS), (4, 8, 56));
    writeln!(
        out,
        r#"
@group(0) @binding(5) var<uniform> rc: array<vec4<u32>, {vec4s}>;

fn rc_fr(k: u32) -> Fr {{
    let a = rc[2u * k];
    let b = rc[2u * k + 1u];
    return Fr(a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w);
}}

fn exp5(x: Fr) -> Fr {{
    let x2 = fr_mul(x, x);
    return fr_mul(fr_mul(x2, x2), x);
}}

// The row of four matrix entries at `at` times the state.
fn dot4(at: u32, s0: Fr, s1: Fr, s2: Fr, s3: Fr) -> Fr {{
    return fr_add(
        fr_add(fr_mul(rc_fr(at), s0), fr_mul(rc_fr(at + 1u), s1)),
        fr_add(fr_mul(rc_fr(at + 2u), s2), fr_mul(rc_fr(at + 3u), s3)));
}}

fn permute(input: array<Fr, 4>) -> array<Fr, 4> {{
    var s0 = fr_add(input[0], rc_fr({c}u));
    var s1 = fr_add(input[1], rc_fr({c}u + 1u));
    var s2 = fr_add(input[2], rc_fr({c}u + 2u));
    var s3 = fr_add(input[3], rc_fr({c}u + 3u));
    for (var r = 0u; r < 8u; r++) {{
        if (r == 4u) {{
            // Partial rounds, with the sparse matrices.
            for (var i = 0u; i < 56u; i++) {{
                let x = fr_add(exp5(s0), rc_fr({c}u + 20u + i));
                let k = {s}u + 7u * i;
                let n0 = dot4(k, x, s1, s2, s3);
                s1 = fr_add(s1, fr_mul(x, rc_fr(k + 4u)));
                s2 = fr_add(s2, fr_mul(x, rc_fr(k + 5u)));
                s3 = fr_add(s3, fr_mul(x, rc_fr(k + 6u)));
                s0 = n0;
            }}
        }}
        s0 = exp5(s0);
        s1 = exp5(s1);
        s2 = exp5(s2);
        s3 = exp5(s3);
        if (r < 7u) {{
            // Rounds 0 to 3 take constants 4..20, rounds 4 to 6 76..88.
            let c = {c}u + select(60u + 4u * r, 4u + 4u * r, r < 4u);
            s0 = fr_add(s0, rc_fr(c));
            s1 = fr_add(s1, rc_fr(c + 1u));
            s2 = fr_add(s2, rc_fr(c + 2u));
            s3 = fr_add(s3, rc_fr(c + 3u));
        }}
        // The matrix before the partial rounds follows round 3.
        let mat = select({m}u, {p}u, r == 3u);
        let n0 = dot4(mat, s0, s1, s2, s3);
        let n1 = dot4(mat + 4u, s0, s1, s2, s3);
        let n2 = dot4(mat + 8u, s0, s1, s2, s3);
        let n3 = dot4(mat + 12u, s0, s1, s2, s3);
        s0 = n0;
        s1 = n1;
        s2 = n2;
        s3 = n3;
    }}
    return array<Fr, 4>(s0, s1, s2, s3);
}}
"#,
        vec4s = CONSTANTS_LEN * 2,
        c = C_AT,
        s = S_AT,
        m = M_AT,
        p = P_AT,
    )
    .unwrap();
}

/// Leaves (`hash_no_pad`, or `hash_or_noop` for up to three elements) and
/// two-to-one nodes, as `PoseidonBN128Hash`.
const TREE_KERNELS: &str = r#"
const FR_ZERO = Fr(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);

// Elements e, e + 1, e + 2 of leaf i (those below leaf_len) as the low limbs
// of an integer, in Montgomery form. Each element is read once.
fn absorb(i: u32, e: u32) -> Fr {
    var w = FR_ZERO;
    let a = canonical(leaf_elem(i, e));
    w[0] = a.x; w[1] = a.y;
    if (e + 1u < hp.leaf_len) { let b = canonical(leaf_elem(i, e + 1u)); w[2] = b.x; w[3] = b.y; }
    if (e + 2u < hp.leaf_len) { let c = canonical(leaf_elem(i, e + 2u)); w[4] = c.x; w[5] = c.y; }
    return to_mont(w);
}

fn write_fr(layer: u32, k: u32, d: Fr) {
    write_digest(layer, k, Field(d[0], d[1]), Field(d[2], d[3]), Field(d[4], d[5]), Field(d[6], d[7]));
}

@compute @workgroup_size(64)
fn hash_leaves(@builtin(global_invocation_id) id: vec3<u32>) {
    let t = id.x;
    if (t >= hp.count) { return; }
    var i = t;
    if (hp.leaf_layout != 0u) { i = rev_bits(t, hp.log_leaves); }
    if (hp.leaf_len <= 3u) {
        // hash_or_noop: the canonical elements' bytes, zero padded.
        var d = FR_ZERO;
        if (hp.leaf_len > 0u) { let a = canonical(leaf_elem(i, 0u)); d[0] = a.x; d[1] = a.y; }
        if (hp.leaf_len > 1u) { let b = canonical(leaf_elem(i, 1u)); d[2] = b.x; d[3] = b.y; }
        if (hp.leaf_len > 2u) { let c = canonical(leaf_elem(i, 2u)); d[4] = c.x; d[5] = c.y; }
        write_fr(0u, i, d);
        return;
    }
    var s = array<Fr, 4>(FR_ZERO, FR_ZERO, FR_ZERO, FR_ZERO);
    let chunks = (hp.leaf_len + 8u) / 9u;
    for (var c = 0u; c < chunks; c++) {
        // Overwrite mode: a partial final chunk leaves the rate elements it
        // does not reach untouched.
        let b = 9u * c;
        s[1] = absorb(i, b);
        if (b + 3u < hp.leaf_len) { s[2] = absorb(i, b + 3u); }
        if (b + 6u < hp.leaf_len) { s[3] = absorb(i, b + 6u); }
        s = permute(s);
    }
    write_fr(0u, i, from_mont(s[0]));
}

fn read_fr(p: u32) -> Fr {
    let a = digests[p]; let b = digests[p + 1u]; let c = digests[p + 2u]; let d = digests[p + 3u];
    return to_mont(Fr(a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y));
}

@compute @workgroup_size(64)
fn hash_nodes(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    if (k >= hp.count) { return; }
    let l = 4u * digest_pos(hp.layer - 1u, 2u * k);
    let out = permute(array<Fr, 4>(FR_ZERO, FR_ZERO, read_fr(l), read_fr(l + 4u)));
    write_fr(hp.layer, k, from_mont(out[0]));
}
"#;

/// The Goldilocks prelude (for leaf elements), the BN254 arithmetic and the
/// permutation, whose constants are uniform binding 5.
pub(crate) fn permutation_source(flavor: Flavor) -> String {
    let mut out = shader::prelude(flavor);
    arithmetic(&mut out, flavor);
    permutation(&mut out);
    out
}

/// The hashing module: `permutation_source` and the tree kernels.
pub fn hash_module(flavor: Flavor) -> String {
    let mut out = permutation_source(flavor);
    out.push_str(shader::HASH_BINDINGS);
    out.push_str(TREE_KERNELS);
    out
}
