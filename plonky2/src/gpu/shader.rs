//! WGSL kernel generation.
//!
//! All kernels are generated from Rust so that the Poseidon constants are
//! immediates and every loop over the sponge state is unrolled. The permutation
//! matches this crate's CPU implementation. Goldilocks elements are
//! pairs of `u32` words with explicit carries, noncanonical in `[0, 2^64)`
//! during arithmetic, as in the CPU code. Both flavors implement the same field
//! operations; the native flavor uses `u64` products and the portable flavor
//! assembles wide products from 16-bit pieces.
//!
//! Seven modules are generated, each with its own bind group 0:
//!
//! - [`hash_module`]: `hash_leaves` (which can also write the gathered leaves
//!   out row by row, replacing a separate transpose), `hash_nodes` and
//!   `poseidon_permute`.
//! - [`fft_module`]: `fft_group`, one group of radix-2 stages of a batched
//!   Goldilocks NTT through workgroup memory.
//! - [`fold_module`]: `fri_fold`, the arity-`2^k` coefficient fold of FRI over
//!   the quadratic extension.
//! - [`open_module`]: `eval_openings`, the committed polynomials at the
//!   opening points.
//! - [`combine_module`]: the alpha-weighted sum of each opening batch's
//!   polynomials, its division by `X - z`, and the combination of the
//!   batches, in coefficient form.
//! - [`permutation_module`]: the partial products and Z of the permutation
//!   argument.
//! - [`pow_module`]: `pow_search`, the proof-of-work nonce search.

use core::fmt::Write;

use crate::field::goldilocks_field::GoldilocksField;
use crate::hash::poseidon::{Poseidon, ALL_ROUND_CONSTANTS};

type F = GoldilocksField;

/// Which 32x32 to 64-bit multiplication the kernels use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flavor {
    /// Products through `u64`; requires `wgpu::Features::SHADER_INT64`.
    Native,
    /// Core WGSL only: the product is assembled from 16-bit pieces.
    Portable,
}

/// Workgroup size of the hashing and folding kernels.
pub const HASH_WORKGROUP: u32 = 64;
/// Workgroup size of the NTT kernel.
pub const FFT_WORKGROUP: u32 = 256;
/// Log2 of the FFT workgroup tile (elements held in workgroup memory, 8 bytes
/// each). The tile size balances workgroup memory use and dispatch count.
pub const FFT_TILE_BITS: u32 = 10;

/// Later stage groups keep runs of at least `2^FFT_MIN_LOW_BITS` contiguous
/// elements, so global memory is read in useful lengths.
const FFT_MIN_LOW_BITS: u32 = 3;

// Flags of `FftParams.flags`.
/// Load the tile from `input` in bit-reversed order, zero beyond `in_len`.
pub const FFT_LOAD_BITREV: u32 = 1;
/// Multiply loaded coefficients by `shift_pows[index]` (coset FFT).
pub const FFT_SHIFT_IN: u32 = 2;
/// Store element `i` at `(n - i) mod n` scaled by `n_inv` (inverse FFT).
pub const FFT_INVERSE: u32 = 4;
/// Multiply stored values by `shift_pows[output index]` (coset inverse FFT).
pub const FFT_SHIFT_OUT: u32 = 8;
/// Store into `output` instead of `data`.
pub const FFT_TO_OUTPUT: u32 = 16;

const NATIVE_MUL: &str = r#"
fn mulwide32(a: u32, b: u32) -> vec2<u32> {
    let p = u64(a) * u64(b);
    return vec2<u32>(u32(p), u32(p >> 32u));
}

// Exact 64x64 -> 128 as four words. Each partial product takes the previous
// one's word as an addend (a 32x32 + 32 + 32 -> 64 sum still fits), so the
// multiplier does the assembly and no carry chain is left on the simple ALU.
fn wide(a: Field, b: Field) -> vec4<u32> {
    let p00 = u64(a.x) * u64(b.x);
    let p01 = u64(a.x) * u64(b.y) + u64(u32(p00 >> 32u));
    let p10 = u64(a.y) * u64(b.x) + u64(u32(p01));
    let p11 = u64(a.y) * u64(b.y) + u64(u32(p01 >> 32u)) + u64(u32(p10 >> 32u));
    return vec4<u32>(u32(p00), u32(p10), u32(p11), u32(p11 >> 32u));
}

fn wide_square(a: Field) -> vec4<u32> {
    return wide(a, a);
}

// lo + hi * 2^64 for a 32-bit hi, as in Plonky2 reduce96: hi * 2^64 ==
// hi * (2^32 - 1), whose sum with the low word is one multiply-add; the carry
// out of the high word is one more EPSILON, as in add_wrap.
fn reduce96(lo: Field, hi: u32) -> Field {
    let q = u64(hi) * u64(0xffffffffu) + u64(lo.x);
    let qlo = u32(q);
    let h = u32(q >> 32u) + lo.y;
    let c = u32(h < lo.y);
    return Field(qlo - c, h + (c & u32(qlo != 0u)));
}
"#;

const PORTABLE_MUL: &str = r#"
// Exact 32x32 -> 64 multiplication, using 16-bit pieces to retain the high word.
fn mulwide32(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xffffu;
    let a1 = a >> 16u;
    let b0 = b & 0xffffu;
    let b1 = b >> 16u;
    let w0 = a0 * b0;
    let t = a1 * b0 + (w0 >> 16u);
    let w1 = a0 * b1 + (t & 0xffffu);
    return vec2<u32>((w1 << 16u) | (w0 & 0xffffu), a1 * b1 + (t >> 16u) + (w1 >> 16u));
}

// Exact 64x64 -> 128 as four words.
fn wide(a: Field, b: Field) -> vec4<u32> {
    let p00 = mulwide32(a.x, b.x);
    let p01 = mulwide32(a.x, b.y);
    let p10 = mulwide32(a.y, b.x);
    let p11 = mulwide32(a.y, b.y);
    let t = p00.y + p01.x;
    let c1 = u32(t < p00.y);
    let w1 = t + p10.x;
    let c2 = u32(w1 < t);
    let u = p01.y + p10.y;
    let c3 = u32(u < p01.y);
    let v = u + p11.x;
    let c4 = u32(v < u);
    let w2 = v + c1 + c2;
    let c5 = u32(w2 < v);
    return vec4<u32>(p00.x, w1, w2, p11.y + c3 + c4 + c5);
}

fn wide_square(a: Field) -> vec4<u32> {
    let p00 = mulwide32(a.x, a.x);
    let p01 = mulwide32(a.x, a.y);
    let p11 = mulwide32(a.y, a.y);
    // 2 * p01 spread over three words.
    let d0 = p01.x << 1u;
    let d1 = (p01.y << 1u) | (p01.x >> 31u);
    let d2 = p01.y >> 31u;
    let w1 = p00.y + d0;
    let c1 = u32(w1 < p00.y);
    let v = p11.x + d1;
    let c2 = u32(v < p11.x);
    let w2 = v + c1;
    let c3 = u32(w2 < v);
    return vec4<u32>(p00.x, w1, w2, p11.y + d2 + c2 + c3);
}


// lo + hi * 2^64 for a 32-bit hi, as in Plonky2 reduce96.
fn reduce96(lo: Field, hi: u32) -> Field {
    return add_wrap(lo, Field(0u - hi, hi - u32(hi != 0u)));
}
"#;

/// Field arithmetic shared by every module.
const FIELD: &str = r#"
// Goldilocks p = 2^64 - 2^32 + 1. A Field is a low word and a high word, and
// stays noncanonical in [0, 2^64) until output, like Plonky2's CPU code.
alias Field = vec2<u32>;

fn canonical(a: Field) -> Field {
    return select(a, Field(a.x - 1u, 0u), a.y == 0xffffffffu && a.x != 0u);
}

// a + b, adding EPSILON = 2^32 - 1 back on carry out. Correct whenever the
// wrapped sum plus EPSILON fits, as in Plonky2 add_no_canonicalize_trashing_input.
fn add_wrap(a: Field, b: Field) -> Field {
    let lo = a.x + b.x;
    let c = u32(lo < a.x);
    let s = a.y + b.y;
    let c1 = u32(s < a.y);
    let hi = s + c;
    let carry = c1 | u32(hi < s);
    return Field(lo - carry, hi + (carry & u32(lo != 0u)));
}

// x + EPSILON mod 2^64, adding EPSILON once more if that overflows (it can
// only overflow into a value below 2^32, so a second pass always fits).
fn add_eps(x: Field) -> Field {
    let nz = u32(x.x != 0u);
    let lo = x.x - 1u;
    let hi = x.y + nz;
    let over = (nz != 0u) && (hi == 0u);
    return select(Field(lo, hi), Field(lo - 1u, u32(lo != 0u)), over);
}

// a + b for any two words in [0, 2^64), as Plonky2's Add for GoldilocksField.
fn add_field(a: Field, b: Field) -> Field {
    let lo = a.x + b.x;
    let c0 = u32(lo < a.x);
    let s = a.y + b.y;
    let c1 = u32(s < a.y);
    let hi = s + c0;
    let carry = c1 | u32(hi < s);
    let r = Field(lo, hi);
    return select(r, add_eps(r), carry != 0u);
}

// x - EPSILON mod 2^64, subtracting EPSILON once more on borrow.
fn sub_eps(x: Field) -> Field {
    let lo = x.x + 1u;
    let c = u32(lo == 0u);
    let hi = x.y - 1u + c;
    let under = (c == 0u) && (x.y == 0u);
    let lo2 = lo + 1u;
    return select(Field(lo, hi), Field(lo2, 0xfffffffeu + u32(lo2 == 0u)), under);
}

// a - b for any two words in [0, 2^64), as Plonky2's Sub for GoldilocksField.
fn sub_field(a: Field, b: Field) -> Field {
    let lo = a.x - b.x;
    let b0 = u32(a.x < b.x);
    let d = a.y - b.y;
    let b1 = u32(a.y < b.y);
    let hi = d - b0;
    let borrow = b1 | u32(d < b0);
    let r = Field(lo, hi);
    return select(r, sub_eps(r), borrow != 0u);
}

// Plonky2 reduce128, with 2^64 == EPSILON and 2^96 == -1 (mod p):
// w == (w.x - w.z - w.w) + (w.y + w.z) 2^32. The low word's borrows and the
// high word's carry net to -1, 0 or 1 units of 2^64, folded back as that many
// EPSILONs: the low word moves by -net and the high word by net unless the
// low word wrapped. The result is any word in [0, 2^64), like the input words.
fn reduce128(w: vec4<u32>) -> Field {
    let t = w.x - w.z;
    let b1 = u32(w.x < w.z);
    let lo = t - w.w;
    let b2 = u32(t < w.w);
    let s = w.y + w.z;
    let c = u32(s < w.y);
    let bb = b1 + b2;
    let hi = s - bb;
    let u = u32(s < bb);
    let net = c - u;
    let lo2 = lo - net;
    let d = u32(lo2 < lo) - u32(lo2 > lo);
    let adj = u32(i32(net + d) >> 1u);
    return Field(lo2, hi + adj);
}

fn mul_field(a: Field, b: Field) -> Field {
    return reduce128(wide(a, b));
}

// a * c for a 32-bit c: a 96-bit product reduced once.
fn mul_small(a: Field, c: u32) -> Field {
    let p0 = mulwide32(a.x, c);
    let p1 = mulwide32(a.y, c);
    let w1 = p0.y + p1.x;
    let carry = u32(w1 < p0.y);
    return reduce96(Field(p0.x, w1), p1.y + carry);
}

fn square_field(a: Field) -> Field {
    return reduce128(wide_square(a));
}

fn square_repeated(a: Field, count: u32) -> Field {
    var result = a;
    for (var i = 0u; i < count; i++) { result = square_field(result); }
    return result;
}

// a^(p - 2), using the CPU Goldilocks addition chain: 63 squares + 9 multiplies.
// A zero field element still maps to zero; callers retain their error checks.
fn inv_field(a: Field) -> Field {
    let t2 = mul_field(square_field(a), a);
    let t3 = mul_field(square_field(t2), a);
    let t6 = mul_field(square_repeated(t3, 3u), t3);
    let t12 = mul_field(square_repeated(t6, 6u), t6);
    let t24 = mul_field(square_repeated(t12, 12u), t12);
    let t30 = mul_field(square_repeated(t24, 6u), t6);
    let t31 = mul_field(square_field(t30), a);
    let t63 = mul_field(square_repeated(t31, 32u), t31);
    return mul_field(square_field(t63), a);
}

// a + c for a canonical constant c, as in Plonky2 add_canonical_u64.
fn add_const(a: Field, c: Field) -> Field {
    return add_wrap(a, c);
}

// acc + a * b, reduced once (Plonky2 multiply_accumulate).
fn mul_add(acc: Field, a: Field, b: Field) -> Field {
    let w = wide(a, b);
    let w0 = w.x + acc.x;
    let c0 = u32(w0 < w.x);
    let s = w.y + acc.y;
    let c1 = u32(s < w.y);
    let w1 = s + c0;
    let w2 = w.z + (c1 | u32(w1 < s));
    return reduce128(vec4<u32>(w0, w1, w2, w.w + u32(w2 < w.z)));
}

// acc - a * b and a * b - acc.
fn mul_sub(acc: Field, a: Field, b: Field) -> Field {
    return sub_field(acc, mul_field(a, b));
}

fn mul_rsub(acc: Field, a: Field, b: Field) -> Field {
    return sub_field(mul_field(a, b), acc);
}

fn sbox(a: Field) -> Field {
    let a2 = reduce128(wide_square(a));
    let a4 = reduce128(wide_square(a2));
    let a3 = reduce128(wide(a, a2));
    return reduce128(wide(a3, a4));
}

// Quadratic extension F[X]/(X^2 - 7): (a0 + a1 X)(b0 + b1 X).
fn ext_mul(a0: Field, a1: Field, b0: Field, b1: Field) -> array<Field, 2> {
    let c0 = add_field(mul_field(a0, b0), mul_field(mul_field(a1, b1), Field(7u, 0u)));
    let c1 = add_field(mul_field(a0, b1), mul_field(a1, b0));
    return array<Field, 2>(c0, c1);
}

// The low `bits` bits of x in reverse order.
fn rev_bits(x: u32, bits: u32) -> u32 {
    if (bits == 0u) { return 0u; }
    return reverseBits(x) >> (32u - bits);
}
"#;

/// The Poseidon MDS routines, transcribed from `poseidon12_mds::mds_multiply_freq`.
const MDS: &str = r#"
// Recombine three 22-bit limb products (each below 2^31) into one field value.
fn mds_combine(o0: u32, o1: u32, o2: u32) -> Field {
    let lo = o0 + (o1 << 22u);
    let m = (o1 >> 10u) + u32(lo < o0);
    let mid = m + (o2 << 12u);
    return reduce96(Field(lo, mid), (o2 >> 20u) + u32(mid < m));
}

// Carry-normalize three limb sums (each below 2^32) into limbs below 2^23
// representing the same field element: the carry out of the top limb, t, is
// folded as t * 2^64 == t * 2^32 - t == (2^22 - t) + (t * 2^10 - 1) * 2^22
// for t > 0. Cheaper than recombining and splitting when no S-box follows.
fn mds_norm(o0: u32, o1: u32, o2: u32) -> vec3<u32> {
    let a1 = o1 + (o0 >> 22u);
    let a2 = o2 + (a1 >> 22u);
    let t = a2 >> 20u;
    let nz = u32(t != 0u);
    return vec3<u32>(
        (o0 & 0x3fffffu) + select(0u, (1u << 22u) - t, t != 0u),
        (a1 & 0x3fffffu) + (t << 10u) - nz,
        a2 & 0xfffffu);
}

// Split 3 x 4 FFT-based multiplication by the circulant part of the Poseidon
// MDS matrix, transcribed from Plonky2's poseidon12_mds::mds_multiply_freq and
// applied to one 22-bit limb of every state word. Every constant is a power of
// two or a small signed integer, so no general multiplications remain. All
// operations are ring operations modulo 2^32 (unsigned, so the wrapping is
// defined) and each true result is below 2^31 (the matrix's row sum is 264),
// so wrapping intermediates cannot change the answer.
struct Row12 {
    f0: u32, f1: u32, f2: u32, f3: u32, f4: u32, f5: u32,
    f6: u32, f7: u32, f8: u32, f9: u32, f10: u32, f11: u32,
}

struct Complex3 {
    z0r: u32, z0i: u32, z1r: u32, z1i: u32, z2r: u32, z2i: u32,
}

// MDS_FREQ_BLOCK_TWO = [(2, -1), (-4, 1), (16, 1)].
const Y0R: u32 = 2u;
const Y0I: u32 = 0xffffffffu; // -1
const Y1R: u32 = 0xfffffffcu; // -4
const Y1I: u32 = 1u;
const Y2R: u32 = 16u;
const Y2I: u32 = 1u;

fn block2(x0r: u32, x0i: u32, x1r: u32, x1i: u32, x2r: u32, x2i: u32) -> Complex3 {
    let x0s = x0r + x0i;
    let x1s = x1r + x1i;
    let x2s = x2r + x2i;
    let y0s = 1u; // Y0R + Y0I
    let y1s = 0xfffffffdu; // Y1R + Y1I = -3
    let y2s = 17u; // Y2R + Y2I
    // x0 y0 - i x1 y2 - i x2 y1
    var m0r = x0r * Y0R; var m0i = x0i * Y0I;
    var m1r = x1r * Y2R; var m1i = x1i * Y2I;
    var m2r = x2r * Y1R; var m2i = x2i * Y1I;
    let z0r = (m0r - m0i) + (x1s * y2s - m1r - m1i) + (x2s * y1s - m2r - m2i);
    let z0i = (x0s * y0s - m0r - m0i) + (m1i - m1r) + (m2i - m2r);
    // x0 y1 + x1 y0 - i x2 y2
    m0r = x0r * Y1R; m0i = x0i * Y1I;
    m1r = x1r * Y0R; m1i = x1i * Y0I;
    m2r = x2r * Y2R; m2i = x2i * Y2I;
    let z1r = (m0r - m0i) + (m1r - m1i) + (x2s * y2s - m2r - m2i);
    let z1i = (x0s * y1s - m0r - m0i) + (x1s * y0s - m1r - m1i) + (m2i - m2r);
    // x0 y2 + x1 y1 + x2 y0
    m0r = x0r * Y2R; m0i = x0i * Y2I;
    m1r = x1r * Y1R; m1i = x1i * Y1I;
    m2r = x2r * Y0R; m2i = x2i * Y0I;
    let z2r = (m0r - m0i) + (m1r - m1i) + (m2r - m2i);
    let z2i = (x0s * y2s - m0r - m0i) + (x1s * y1s - m1r - m1i) + (x2s * y0s - m2r - m2i);
    return Complex3(z0r, z0i, z1r, z1i, z2r, z2i);
}

fn mds_freq(s0: u32, s1: u32, s2: u32, s3: u32, s4: u32, s5: u32,
            s6: u32, s7: u32, s8: u32, s9: u32, s10: u32, s11: u32) -> Row12 {
    // Real 4-point FFTs of (s0, s3, s6, s9), (s1, s4, s7, s10), (s2, s5, s8, s11).
    let a0 = s0 + s6; let a2 = s0 - s6; let a1 = s3 + s9; let a3 = s3 - s9;
    let u0 = a0 + a1; let u1r = a2; let u1i = 0u - a3; let u2 = a0 - a1;
    let b0 = s1 + s7; let b2 = s1 - s7; let b1 = s4 + s10; let b3 = s4 - s10;
    let u4 = b0 + b1; let u5r = b2; let u5i = 0u - b3; let u6 = b0 - b1;
    let c0 = s2 + s8; let c2 = s2 - s8; let c1 = s5 + s11; let c3 = s5 - s11;
    let u8 = c0 + c1; let u9r = c2; let u9i = 0u - c3; let u10 = c0 - c1;
    // block1 with MDS_FREQ_BLOCK_ONE = [16, 32, 16].
    let v0 = 16u * u0 + 16u * u4 + 32u * u8;
    let v4 = 32u * u0 + 16u * u4 + 16u * u8;
    let v8 = 16u * u0 + 32u * u4 + 16u * u8;
    let v = block2(u1r, u1i, u5r, u5i, u9r, u9i);
    // block3 with MDS_FREQ_BLOCK_THREE = [-1, -8, 2].
    let v2 = 8u * u10 - u2 - 2u * u6;
    let v6 = 0u - 8u * u2 - u6 - 2u * u10;
    let v10 = 2u * u2 - 8u * u6 - u10;
    // Real 4-point inverse FFTs (unreduced, as in poseidon12_mds).
    let d0 = v0 + v2; let d1 = v0 - v2; let d2 = v.z0r; let d3 = 0u - v.z0i;
    let e0 = v4 + v6; let e1 = v4 - v6; let e2 = v.z1r; let e3 = 0u - v.z1i;
    let g0 = v8 + v10; let g1 = v8 - v10; let g2 = v.z2r; let g3 = 0u - v.z2i;
    return Row12(
        d0 + d2, e0 + e2, g0 + g2,
        d1 + d3, e1 + e3, g1 + g3,
        d0 - d2, e0 - e2, g0 - g2,
        d1 - d3, e1 - e3, g1 - g3);
}

struct State {
    s0: Field, s1: Field, s2: Field, s3: Field, s4: Field, s5: Field,
    s6: Field, s7: Field, s8: Field, s9: Field, s10: Field, s11: Field,
}
"#;

/// A WGSL `Field` literal.
pub fn literal(value: u64) -> String {
    format!("Field({:#x}u, {:#x}u)", value as u32, value >> 32)
}

fn round_constants(round: usize) -> [u64; 12] {
    core::array::from_fn(|i| ALL_ROUND_CONSTANTS[round * 12 + i])
}

/// Dense MDS layer: three 22-bit limbs per word, each multiplied by the
/// circulant matrix with the FFT-based routine, plus the diagonal term on row
/// 0, then each row is recombined and reduced once. With `next`, the next
/// round's constants are added limb-wise before the recombination (each limb
/// product is below 2^31.1, so a 22-bit addend keeps it below 2^32), which
/// costs three adds per word instead of a field addition. With `limbs_in`,
/// words 1 to 11 enter as the limb variables `t0_i, t1_i, t2_i` left by the
/// previous layer instead of being split from `s_i`; with `limbs_out` they
/// leave that way, normalized by `mds_norm` instead of recombined.
fn mds_layer(out: &mut String, next: Option<[u64; 12]>, limbs_in: bool, limbs_out: bool) {
    for i in 0..12 {
        if limbs_in && i > 0 {
            continue;
        }
        writeln!(
            out,
            "    let l0_{i} = s{i}.x & 0x3fffffu; let l1_{i} = ((s{i}.x >> 22u) | (s{i}.y << 10u)) & 0x3fffffu; let l2_{i} = s{i}.y >> 12u;"
        )
        .unwrap();
    }
    for limb in 0..3 {
        let inputs = (0..12)
            .map(|i| {
                if limbs_in && i > 0 {
                    format!("t{limb}_{i}")
                } else {
                    format!("l{limb}_{i}")
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(out, "    let m{limb} = mds_freq({inputs});").unwrap();
    }
    assert_eq!(<F as Poseidon>::MDS_MATRIX_DIAG[1..], [0; 11]);
    let diag = <F as Poseidon>::MDS_MATRIX_DIAG[0];
    assert!(<F as Poseidon>::MDS_MATRIX_CIRC.iter().sum::<u64>() + diag < 1 << 9);
    for row in 0..12 {
        let limbs = (0..3)
            .map(|limb| {
                let addend = match next {
                    Some(c) => format!(" + {}u", (c[row] >> (22 * limb)) & 0x3fffff),
                    None => String::new(),
                };
                if row == 0 {
                    format!("m{limb}.f0 + {diag}u * l{limb}_0{addend}")
                } else {
                    format!("m{limb}.f{row}{addend}")
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        if limbs_out && row > 0 {
            writeln!(
                out,
                "    {{ let n = mds_norm({limbs}); t0_{row} = n.x; t1_{row} = n.y; t2_{row} = n.z; }}"
            )
            .unwrap();
        } else {
            writeln!(out, "    s{row} = mds_combine({limbs});").unwrap();
        }
    }
}

/// The unrolled `permute(State) -> State` function: round 0's constants are
/// added to the input, every later round's constants ride on the previous
/// round's MDS recombination. The 22 partial rounds (4..26) are in their plain
/// form (S-box on word 0, then the dense layer): on the GPU the dense layer
/// (22-bit limbs, small constant multiplies) is cheaper than Plonky2's sparse
/// "fast" variant, whose dot products with 64-bit coefficients need a full
/// 64x64 product and a 160-bit carry chain per term.
fn permutation(out: &mut String) {
    out.push_str("\nfn permute(input: State) -> State {\n");
    for i in 0..12 {
        writeln!(out, "    var s{i} = input.s{i};").unwrap();
    }
    // Words 1 to 11 as limbs between the partial rounds' layers.
    for i in 1..12 {
        writeln!(
            out,
            "    var t0_{i}: u32; var t1_{i}: u32; var t2_{i}: u32;"
        )
        .unwrap();
    }
    for (i, c) in round_constants(0).iter().enumerate() {
        writeln!(out, "    s{i} = add_const(s{i}, {});", literal(*c)).unwrap();
    }
    let partial = 4..26;
    for round in 0..30 {
        let full = !partial.contains(&round);
        writeln!(out, "    {{ // round {round}").unwrap();
        let words = if full { 12 } else { 1 };
        for i in 0..words {
            writeln!(out, "    s{i} = sbox(s{i});").unwrap();
        }
        let next = (round + 1 < 30).then(|| round_constants(round + 1));
        // The layer before every partial round leaves words 1 to 11 as limbs.
        let limbs_in = !full;
        let limbs_out = partial.contains(&(round + 1));
        mds_layer(out, next, limbs_in, limbs_out);
        out.push_str("    }\n");
    }
    out.push_str("    return State(");
    out.push_str(
        &(0..12)
            .map(|i| format!("s{i}"))
            .collect::<Vec<_>>()
            .join(", "),
    );
    out.push_str(");\n}\n");
}

/// The field arithmetic every module starts with.
pub fn prelude(flavor: Flavor) -> String {
    let mut out = String::new();
    out.push_str(match flavor {
        Flavor::Native => NATIVE_MUL,
        Flavor::Portable => PORTABLE_MUL,
    });
    out.push_str(FIELD);
    out
}

/// Parameters of the hashing kernels (uniform binding 0 of [`hash_module`]).
const HASH_KERNELS: &str = r#"
struct HashParams {
    // Leaves (hash_leaves), nodes of the output layer (hash_nodes) or states
    // (poseidon_permute) in this dispatch.
    count: u32,
    // Elements per leaf.
    leaf_len: u32,
    // 0: src holds leaves row by row. 1: src holds `columns` columns of 2^log_n
    // elements and leaf i gathers rows rev(i * 2^arity_bits + e / columns).
    leaf_layout: u32,
    columns: u32,
    log_n: u32,
    arity_bits: u32,
    // log2 of the number of leaves.
    log_leaves: u32,
    // hash_nodes: the layer being produced (1..=num_layers).
    layer: u32,
    // log2(leaves) - cap_height.
    num_layers: u32,
    // Digests per cap subtree.
    tree_len: u32,
    // hash_leaves: also write each gathered leaf to rows[i * leaf_len ..].
    write_rows: u32,
    pad: u32,
}
@group(0) @binding(0) var<uniform> hp: HashParams;
@group(0) @binding(1) var<storage, read> src: array<Field>;
@group(0) @binding(2) var<storage, read_write> digests: array<Field>;
@group(0) @binding(3) var<storage, read_write> cap: array<Field>;
@group(0) @binding(4) var<storage, read_write> rows: array<Field>;

// Position (in digests) of node k of layer `layer`, in Plonky2's layout:
// left_subtree || left_digest || right_digest || right_subtree, per cap subtree.
fn digest_pos(layer: u32, k: u32) -> u32 {
    let shift = hp.num_layers - layer;
    let subtree = k >> shift;
    let local = k & ((1u << shift) - 1u);
    return subtree * hp.tree_len + ((local >> 1u) << (layer + 2u)) + (1u << (layer + 1u)) - 2u + (local & 1u);
}

fn write_digest(layer: u32, k: u32, d0: Field, d1: Field, d2: Field, d3: Field) {
    if (layer == hp.num_layers) {
        let p = 4u * k;
        cap[p] = d0; cap[p + 1u] = d1; cap[p + 2u] = d2; cap[p + 3u] = d3;
    } else {
        let p = 4u * digest_pos(layer, k);
        digests[p] = d0; digests[p + 1u] = d1; digests[p + 2u] = d2; digests[p + 3u] = d3;
    }
}

fn leaf_elem(i: u32, e: u32) -> Field {
    if (hp.leaf_layout == 0u) {
        return src[i * hp.leaf_len + e];
    }
    let column = e % hp.columns;
    let row = rev_bits((i << hp.arity_bits) + e / hp.columns, hp.log_n);
    let v = src[column * (1u << hp.log_n) + row];
    if (hp.write_rows != 0u) { rows[i * hp.leaf_len + e] = v; }
    return v;
}

@compute @workgroup_size(64)
fn hash_leaves(@builtin(global_invocation_id) id: vec3<u32>) {
    let t = id.x;
    if (t >= hp.count) { return; }
    // Gathered leaves are visited in bit-reversed order so that neighbouring
    // threads read neighbouring rows.
    var i = t;
    if (hp.leaf_layout != 0u) { i = rev_bits(t, hp.log_leaves); }
    var s0 = Field(0u, 0u); var s1 = Field(0u, 0u); var s2 = Field(0u, 0u); var s3 = Field(0u, 0u);
    var s4 = Field(0u, 0u); var s5 = Field(0u, 0u); var s6 = Field(0u, 0u); var s7 = Field(0u, 0u);
    var s8 = Field(0u, 0u); var s9 = Field(0u, 0u); var s10 = Field(0u, 0u); var s11 = Field(0u, 0u);
    if (hp.leaf_len <= 4u) {
        // hash_or_noop: the canonical elements, zero padded.
        if (hp.leaf_len > 0u) { s0 = canonical(leaf_elem(i, 0u)); }
        if (hp.leaf_len > 1u) { s1 = canonical(leaf_elem(i, 1u)); }
        if (hp.leaf_len > 2u) { s2 = canonical(leaf_elem(i, 2u)); }
        if (hp.leaf_len > 3u) { s3 = canonical(leaf_elem(i, 3u)); }
        write_digest(0u, i, s0, s1, s2, s3);
        return;
    }
    let chunks = (hp.leaf_len + 7u) / 8u;
    for (var c = 0u; c < chunks; c++) {
        let b = c * 8u;
        // Overwrite mode: a partial final chunk leaves the remaining rate
        // elements untouched, as PlonkyPermutation::set_from_slice does.
        if (b < hp.leaf_len) { s0 = leaf_elem(i, b); }
        if (b + 1u < hp.leaf_len) { s1 = leaf_elem(i, b + 1u); }
        if (b + 2u < hp.leaf_len) { s2 = leaf_elem(i, b + 2u); }
        if (b + 3u < hp.leaf_len) { s3 = leaf_elem(i, b + 3u); }
        if (b + 4u < hp.leaf_len) { s4 = leaf_elem(i, b + 4u); }
        if (b + 5u < hp.leaf_len) { s5 = leaf_elem(i, b + 5u); }
        if (b + 6u < hp.leaf_len) { s6 = leaf_elem(i, b + 6u); }
        if (b + 7u < hp.leaf_len) { s7 = leaf_elem(i, b + 7u); }
        let r = permute(State(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11));
        s0 = r.s0; s1 = r.s1; s2 = r.s2; s3 = r.s3; s4 = r.s4; s5 = r.s5;
        s6 = r.s6; s7 = r.s7; s8 = r.s8; s9 = r.s9; s10 = r.s10; s11 = r.s11;
    }
    write_digest(0u, i, canonical(s0), canonical(s1), canonical(s2), canonical(s3));
}

@compute @workgroup_size(64)
fn hash_nodes(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    if (k >= hp.count) { return; }
    let l = 4u * digest_pos(hp.layer - 1u, 2u * k);
    let r = l + 4u;
    let zero = Field(0u, 0u);
    let out = permute(State(
        digests[l], digests[l + 1u], digests[l + 2u], digests[l + 3u],
        digests[r], digests[r + 1u], digests[r + 2u], digests[r + 3u],
        zero, zero, zero, zero));
    write_digest(hp.layer, k, canonical(out.s0), canonical(out.s1), canonical(out.s2), canonical(out.s3));
}

// Applies the permutation to `count` states of twelve elements in src,
// writing the canonical results to digests.
@compute @workgroup_size(64)
fn poseidon_permute(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= hp.count) { return; }
    let b = 12u * i;
    let r = permute(State(
        src[b], src[b + 1u], src[b + 2u], src[b + 3u], src[b + 4u], src[b + 5u],
        src[b + 6u], src[b + 7u], src[b + 8u], src[b + 9u], src[b + 10u], src[b + 11u]));
    digests[b] = canonical(r.s0); digests[b + 1u] = canonical(r.s1);
    digests[b + 2u] = canonical(r.s2); digests[b + 3u] = canonical(r.s3);
    digests[b + 4u] = canonical(r.s4); digests[b + 5u] = canonical(r.s5);
    digests[b + 6u] = canonical(r.s6); digests[b + 7u] = canonical(r.s7);
    digests[b + 8u] = canonical(r.s8); digests[b + 9u] = canonical(r.s9);
    digests[b + 10u] = canonical(r.s10); digests[b + 11u] = canonical(r.s11);
}
"#;

/// `fn mds12(p: array<Field, 12>) -> array<Field, 12>`: the Poseidon MDS
/// layer as one function, for the quotient kernels.
pub fn mds_function() -> String {
    let mut out = String::from("\nfn mds12(p: array<Field, 12>) -> array<Field, 12> {\n");
    for i in 0..12 {
        writeln!(out, "    var s{i} = p[{i}];").unwrap();
    }
    mds_layer(&mut out, None, false, false);
    out.push_str("    return array<Field, 12>(");
    out.push_str(
        &(0..12)
            .map(|i| format!("s{i}"))
            .collect::<Vec<_>>()
            .join(", "),
    );
    out.push_str(");\n}\n");
    out
}

/// The prelude plus the MDS routines and `mds12`.
pub fn prelude_with_mds(flavor: Flavor) -> String {
    let mut out = prelude(flavor);
    out.push_str(MDS);
    out.push_str(&mds_function());
    out
}

/// The product of two elements of the quintic extension `F[X] / (X^5 - 3)`,
/// coefficient by coefficient: the products that wrap
/// past X^5 take the second operand's coefficient scaled by 3 first.
pub(crate) const QMUL: &str = r#"
fn qmul(x: array<Field, 5>, y: array<Field, 5>) -> array<Field, 5> {
    let t1 = mul_small(y[1], 3u);
    let t2 = mul_small(y[2], 3u);
    let t3 = mul_small(y[3], 3u);
    let t4 = mul_small(y[4], 3u);
    var r: array<Field, 5>;
    r[0] = mul_add(mul_add(mul_add(mul_add(mul_field(x[0], y[0]), x[1], t4), x[2], t3), x[3], t2), x[4], t1);
    r[1] = mul_add(mul_add(mul_add(mul_add(mul_field(x[0], y[1]), x[1], y[0]), x[2], t4), x[3], t3), x[4], t2);
    r[2] = mul_add(mul_add(mul_add(mul_add(mul_field(x[0], y[2]), x[1], y[1]), x[2], y[0]), x[3], t4), x[4], t3);
    r[3] = mul_add(mul_add(mul_add(mul_add(mul_field(x[0], y[3]), x[1], y[2]), x[2], y[1]), x[3], y[0]), x[4], t4);
    r[4] = mul_add(mul_add(mul_add(mul_add(mul_field(x[0], y[4]), x[1], y[3]), x[2], y[2]), x[3], y[1]), x[4], y[0]);
    return r;
}
"#;

/// The hashing module: sponge over leaves, two-to-one nodes and a batch permutation.
pub fn hash_module(flavor: Flavor) -> String {
    let mut out = prelude(flavor);
    out.push_str(MDS);
    permutation(&mut out);
    out.push_str(HASH_KERNELS);
    out
}

/// Proof-of-work search: each invocation tries candidate
/// `base + id` in word `position` of the prepared sponge state, applies the
/// exact permutation and keeps the smallest index whose response (word 7, the
/// last rate element) has at least `min_zeros` leading zero bits.
const POW_KERNEL: &str = r#"
struct PowParams {
    position: u32,
    base_lo: u32,
    base_hi: u32,
    count: u32,
    min_zeros: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}
@group(0) @binding(0) var<uniform> pp: PowParams;
@group(0) @binding(1) var<storage, read> pow_state: array<Field, 12>;
@group(0) @binding(2) var<storage, read_write> pow_found: array<atomic<u32>, 4>;

fn pow_word(k: u32, c: Field) -> Field {
    return select(pow_state[k], c, pp.position == k);
}

@compute @workgroup_size(64)
fn pow_search(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= pp.count) { return; }
    let lo = pp.base_lo + i;
    let c = Field(lo, pp.base_hi + u32(lo < i));
    let r = permute(State(
        pow_word(0u, c), pow_word(1u, c), pow_word(2u, c), pow_word(3u, c),
        pow_word(4u, c), pow_word(5u, c), pow_word(6u, c), pow_word(7u, c),
        pow_word(8u, c), pow_word(9u, c), pow_word(10u, c), pow_word(11u, c)));
    let o = canonical(r.s7);
    let zeros = select(countLeadingZeros(o.y), 32u + countLeadingZeros(o.x), o.y == 0u);
    if (zeros >= pp.min_zeros) {
        atomicMin(&pow_found[0], i);
    }
}
"#;

/// The proof-of-work search kernel.
pub fn pow_module(flavor: Flavor) -> String {
    let mut out = prelude(flavor);
    out.push_str(MDS);
    permutation(&mut out);
    out.push_str(POW_KERNEL);
    out
}

/// One group of radix-2 decimation-in-time stages of a batched NTT.
///
/// The input is in bit-reversed order (or loaded that way when
/// `FFT_LOAD_BITREV` is set), as in Plonky2's `fft_classic`. A workgroup holds
/// `2^(stages + low_bits)` elements: the `2^stages` elements that the stages
/// `s0..s0 + stages` connect, times `2^low_bits` neighbouring elements so that
/// global memory is read in contiguous runs. The first group has `s0 = 0` and
/// no low bits; later groups need `low_bits <= s0`.
const FFT_KERNEL: &str = r#"
struct FftParams {
    log_n: u32,
    s0: u32,
    stages: u32,
    low_bits: u32,
    columns: u32,
    in_len: u32,
    in_stride: u32,
    flags: u32,
    n_inv: Field,
    // Stages of the group that are replication of the loaded coefficients
    // (first group of a zero-padded forward transform).
    skip: u32,
    pad: u32,
}
const LOAD_BITREV: u32 = 1u;
const SHIFT_IN: u32 = 2u;
const INVERSE: u32 = 4u;
const SHIFT_OUT: u32 = 8u;
const TO_OUTPUT: u32 = 16u;

@group(0) @binding(0) var<uniform> fp: FftParams;
@group(0) @binding(1) var<storage, read> input: array<Field>;
@group(0) @binding(2) var<storage, read_write> data: array<Field>;
@group(0) @binding(3) var<storage, read_write> output: array<Field>;
@group(0) @binding(4) var<storage, read> twiddles: array<Field>;
@group(0) @binding(5) var<storage, read> shift_pows: array<Field>;

var<workgroup> tile: array<Field, FFT_TILE_SIZE>;

@compute @workgroup_size(256)
fn fft_group(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    let w = wid.x;
    let col = wid.y;
    let n = 1u << fp.log_n;
    let s0 = fp.s0;
    let bl = fp.low_bits;
    let tile_size = 1u << (fp.stages + bl);
    let low_mask = (1u << bl) - 1u;
    let base = ((w & ((1u << (s0 - bl)) - 1u)) << bl) | ((w >> (s0 - bl)) << (s0 + fp.stages));
    let col_off = col * n;
    if (fp.skip != 0u) {
        // Zero-padded input in bit-reversed order: the coefficients that can be
        // nonzero sit at multiples of 2^skip, and the first `skip` stages only
        // copy each of them into the 2^skip - 1 slots after it (their partners
        // are zero and their twiddles irrelevant). Load and shift each source
        // once and replicate it, then start the butterflies at stage `skip`.
        let rep = 1u << fp.skip;
        for (var g = tid; g < (tile_size >> fp.skip); g += 256u) {
            let l = g << fp.skip;
            let i = base | ((l >> bl) << s0) | (l & low_mask);
            let x = rev_bits(i, fp.log_n);
            var v = Field(0u, 0u);
            if (x < fp.in_len) {
                v = input[col * fp.in_stride + x];
                if ((fp.flags & SHIFT_IN) != 0u) { v = mul_field(v, shift_pows[x]); }
            }
            for (var k = 0u; k < rep; k++) { tile[l | k] = v; }
        }
    } else {
        for (var l = tid; l < tile_size; l += 256u) {
            let i = base | ((l >> bl) << s0) | (l & low_mask);
            var v = Field(0u, 0u);
            if ((fp.flags & LOAD_BITREV) != 0u) {
                let x = rev_bits(i, fp.log_n);
                if (x < fp.in_len) {
                    v = input[col * fp.in_stride + x];
                    if ((fp.flags & SHIFT_IN) != 0u) { v = mul_field(v, shift_pows[x]); }
                }
            } else {
                v = data[col_off + i];
            }
            tile[l] = v;
        }
    }
    workgroupBarrier();
    let half = tile_size >> 1u;
    for (var st = fp.skip; st < fp.stages; st++) {
        let s = s0 + st;
        for (var b = tid; b < half; b += 256u) {
            let low = b & low_mask;
            let jp = b >> bl;
            let j0 = ((jp >> st) << (st + 1u)) | (jp & ((1u << st) - 1u));
            let j1 = j0 | (1u << st);
            let l0 = (j0 << bl) | low;
            let l1 = (j1 << bl) | low;
            let i0 = base | (j0 << s0) | low;
            let omega = twiddles[(i0 & ((1u << s) - 1u)) << (fp.log_n - s - 1u)];
            let u = tile[l0];
            let t = mul_field(omega, tile[l1]);
            tile[l0] = add_field(u, t);
            tile[l1] = sub_field(u, t);
        }
        workgroupBarrier();
    }
    for (var l = tid; l < tile_size; l += 256u) {
        let i = base | ((l >> bl) << s0) | (l & low_mask);
        var v = tile[l];
        var o = i;
        if ((fp.flags & INVERSE) != 0u) {
            o = (n - i) & (n - 1u);
            v = mul_field(v, fp.n_inv);
        }
        if ((fp.flags & SHIFT_OUT) != 0u) { v = mul_field(v, shift_pows[o]); }
        if ((fp.flags & TO_OUTPUT) != 0u) {
            output[col_off + o] = v;
        } else {
            data[col_off + o] = v;
        }
    }
}
"#;

/// The NTT module.
pub fn fft_module(flavor: Flavor) -> String {
    let mut out = prelude(flavor);
    let tile = 1u32 << FFT_TILE_BITS;
    writeln!(out, "const FFT_TILE_SIZE: u32 = {tile}u;").unwrap();
    out.push_str(FFT_KERNEL);
    out
}

const FOLD_KERNEL: &str = r#"
struct FoldParams {
    // Output coefficients.
    count: u32,
    arity: u32,
    // Input coefficients (count * arity); the imaginary parts follow at this offset.
    in_len: u32,
    pad: u32,
}
@group(0) @binding(0) var<uniform> fo: FoldParams;
@group(0) @binding(1) var<storage, read> fin: array<Field>;
@group(0) @binding(2) var<storage, read_write> fout: array<Field>;
// beta^i for i < arity, real part at 2i and imaginary part at 2i + 1.
@group(0) @binding(3) var<storage, read> betas: array<Field>;

// fout[k] = sum_{i < arity} fin[k * arity + i] * beta^i over the quadratic
// extension, as Plonky2's reduce_with_powers over each chunk.
@compute @workgroup_size(64)
fn fri_fold(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    if (k >= fo.count) { return; }
    var acc0 = Field(0u, 0u);
    var acc1 = Field(0u, 0u);
    for (var i = 0u; i < fo.arity; i++) {
        let idx = k * fo.arity + i;
        let m = ext_mul(fin[idx], fin[fo.in_len + idx], betas[2u * i], betas[2u * i + 1u]);
        acc0 = add_field(acc0, m[0]);
        acc1 = add_field(acc1, m[1]);
    }
    fout[k] = acc0;
    fout[fo.count + k] = acc1;
}
"#;

/// The FRI folding module.
pub fn fold_module(flavor: Flavor) -> String {
    let mut out = prelude(flavor);
    out.push_str(FOLD_KERNEL);
    out
}

/// The radix-2 stage groups of an NTT of `2^log_n` elements: `(s0, stages, low_bits)`.
pub fn fft_plan(log_n: u32) -> Vec<(u32, u32, u32)> {
    let bits = FFT_TILE_BITS;
    let first = log_n.min(bits);
    let mut plan = vec![(0, first, 0)];
    let mut s0 = first;
    while s0 < log_n {
        let stages = (log_n - s0).min(bits - FFT_MIN_LOW_BITS);
        let low_bits = (bits - stages).min(s0);
        plan.push((s0, stages, low_bits));
        s0 += stages;
    }
    plan
}

const EXT_POW: &str = r#"
// z^k over the quadratic extension, by squaring.
fn ext_pow(z0: Field, z1: Field, k: u32) -> array<Field, 2> {
    var r0 = Field(1u, 0u);
    var r1 = Field(0u, 0u);
    var b0 = z0;
    var b1 = z1;
    var e = k;
    while (e != 0u) {
        if ((e & 1u) != 0u) {
            let m = ext_mul(r0, r1, b0, b1);
            r0 = m[0];
            r1 = m[1];
        }
        let s = ext_mul(b0, b1, b0, b1);
        b0 = s[0];
        b1 = s[1];
        e = e >> 1u;
    }
    return array<Field, 2>(r0, r1);
}
"#;

const OPENING_KERNELS: &str = r#"
struct OpenParams {
    // Coefficients per polynomial.
    degree: u32,
    // Polynomials evaluated by this dispatch.
    count: u32,
    pad: vec2<u32>,
    // The opening points, each as four words (real, then imaginary).
    points: array<vec4<u32>, 4>,
}
@group(0) @binding(0) var<uniform> op: OpenParams;
// The coefficients of up to four oracles, column-major, `degree` per polynomial.
@group(0) @binding(1) var<storage, read> c0: array<Field>;
@group(0) @binding(2) var<storage, read> c1: array<Field>;
@group(0) @binding(3) var<storage, read> c2: array<Field>;
@group(0) @binding(4) var<storage, read> c3: array<Field>;
// Per polynomial: its oracle in the low half and its point in the high
// half of x, and its column in y.
@group(0) @binding(5) var<storage, read> terms: array<vec2<u32>>;
// Per point: z^k for k < degree, the real parts then the imaginary parts.
@group(0) @binding(6) var<storage, read_write> zpows: array<Field>;
// Per polynomial: its value at its point, the real parts then the imaginary parts.
@group(0) @binding(7) var<storage, read_write> evals: array<Field>;

var<workgroup> red0: array<Field, 256>;
var<workgroup> red1: array<Field, 256>;

fn coefficient(oracle: u32, index: u32) -> Field {
    switch (oracle) {
        case 0u: { return c0[index]; }
        case 1u: { return c1[index]; }
        case 2u: { return c2[index]; }
        default: { return c3[index]; }
    }
}

// zpows for point id.y.
@compute @workgroup_size(64)
fn point_powers(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    let p = id.y;
    if (k >= op.degree) { return; }
    let z = ext_pow(op.points[p].xy, op.points[p].zw, k);
    let o = p * 2u * op.degree;
    zpows[o + k] = z[0];
    zpows[o + op.degree + k] = z[1];
}

// One workgroup per polynomial: sum_k c_k z^k, every thread taking each
// 256th coefficient, the partial sums reduced through workgroup memory.
@compute @workgroup_size(256)
fn eval_openings(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let t = wid.x;
    let tid = lid.x;
    let term = terms[t];
    let oracle = term.x & 0xffffu;
    let z = (term.x >> 16u) * 2u * op.degree;
    let base = term.y * op.degree;
    var a0 = Field(0u, 0u);
    var a1 = Field(0u, 0u);
    for (var k = tid; k < op.degree; k += 256u) {
        let c = coefficient(oracle, base + k);
        a0 = mul_add(a0, c, zpows[z + k]);
        a1 = mul_add(a1, c, zpows[z + op.degree + k]);
    }
    red0[tid] = a0;
    red1[tid] = a1;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            red0[tid] = add_field(red0[tid], red0[tid + s]);
            red1[tid] = add_field(red1[tid], red1[tid + s]);
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        evals[t] = red0[0];
        evals[op.count + t] = red1[0];
    }
}
"#;

const COMBINE_KERNELS: &str = r#"
struct CombineParams {
    // Coefficients per polynomial.
    degree: u32,
    // Coefficients of the output (degree << rate_bits); those beyond degree are zero.
    n: u32,
    num_batches: u32,
    // Blocks of BLOCK coefficients per polynomial.
    num_blocks: u32,
    // Per batch: its first term and its number of terms.
    batches: array<vec4<u32>, 4>,
    // Per batch: its opening point, as four words (real, then imaginary).
    points: array<vec4<u32>, 4>,
    // Per batch: the power of alpha scaling its quotient, as four words.
    scales: array<vec4<u32>, 4>,
}
@group(0) @binding(0) var<uniform> cp: CombineParams;
// The coefficients of up to four oracles, column-major, `degree` per polynomial.
@group(0) @binding(1) var<storage, read> c0: array<Field>;
@group(0) @binding(2) var<storage, read> c1: array<Field>;
@group(0) @binding(3) var<storage, read> c2: array<Field>;
@group(0) @binding(4) var<storage, read> c3: array<Field>;
// Per term: its oracle (in the low half of x) and its column, batch by batch.
@group(0) @binding(5) var<storage, read> terms: array<vec2<u32>>;
// Per term: the power of alpha weighting it (real, imaginary).
@group(0) @binding(6) var<storage, read> alphas: array<Field>;
// Per batch, each as the real parts then the imaginary parts: z^k for
// k <= BLOCK; the reduced coefficients C_b; the within-block quotients;
// then per batch the block sums T_B and the block suffixes R_B.
@group(0) @binding(7) var<storage, read_write> scratch: array<Field>;
// The combined quotient's coefficients: the real parts, then the imaginary parts.
@group(0) @binding(8) var<storage, read_write> out: array<Field>;

const BLOCK: u32 = 256u;

fn coefficient(oracle: u32, index: u32) -> Field {
    switch (oracle) {
        case 0u: { return c0[index]; }
        case 1u: { return c1[index]; }
        case 2u: { return c2[index]; }
        default: { return c3[index]; }
    }
}

fn zp_off(b: u32) -> u32 { return b * 2u * (BLOCK + 1u); }
fn c_off(b: u32) -> u32 { return zp_off(cp.num_batches) + b * 2u * cp.degree; }
fn l_off(b: u32) -> u32 { return c_off(cp.num_batches + b); }
fn t_off(b: u32) -> u32 { return l_off(cp.num_batches) + b * 2u * cp.num_blocks; }
fn r_off(b: u32) -> u32 { return t_off(cp.num_batches + b); }

// z_b^k for batch id.y.
@compute @workgroup_size(64)
fn batch_powers(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    let b = id.y;
    if (k > BLOCK || k >= cp.degree) { return; }
    let z = ext_pow(cp.points[b].xy, cp.points[b].zw, k);
    let o = zp_off(b);
    scratch[o + k] = z[0];
    scratch[o + BLOCK + 1u + k] = z[1];
}

// C_b = sum_j alpha^j f_j over the batch's polynomials: one thread per
// coefficient of batch id.y.
@compute @workgroup_size(64)
fn reduce_batch(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    let b = id.y;
    if (k >= cp.degree) { return; }
    let first = cp.batches[b].x;
    let end = first + cp.batches[b].y;
    var s0 = Field(0u, 0u);
    var s1 = Field(0u, 0u);
    for (var t = first; t < end; t++) {
        let term = terms[t];
        let c = coefficient(term.x & 0xffffu, term.y * cp.degree + k);
        s0 = mul_add(s0, c, alphas[2u * t]);
        s1 = mul_add(s1, c, alphas[2u * t + 1u]);
    }
    let o = c_off(b);
    scratch[o + k] = s0;
    scratch[o + cp.degree + k] = s1;
}

// The quotient (C_b(X) - C_b(z)) / (X - z) has q_k = c_{k+1} + z q_{k+1}.
// Within block id.x of batch id.y, Horner from the block's end gives the
// quotient of the block's own coefficients, l_k, with l at the block's last
// coefficient zero; the block's sum T_B = c_{BS} + z l_{BS} is kept for the
// suffixes.
@compute @workgroup_size(64)
fn divide_blocks(@builtin(global_invocation_id) id: vec3<u32>) {
    let blk = id.x;
    let b = id.y;
    if (blk >= cp.num_blocks) { return; }
    let z0 = cp.points[b].xy;
    let z1 = cp.points[b].zw;
    let c = c_off(b);
    let l = l_off(b);
    let start = blk * BLOCK;
    var a0 = Field(0u, 0u);
    var a1 = Field(0u, 0u);
    var k = start + BLOCK - 1u;
    loop {
        scratch[l + k] = a0;
        scratch[l + cp.degree + k] = a1;
        let m = ext_mul(a0, a1, z0, z1);
        a0 = add_field(m[0], scratch[c + k]);
        a1 = add_field(m[1], scratch[c + cp.degree + k]);
        if (k == start) { break; }
        k--;
    }
    let t = t_off(b);
    scratch[t + blk] = a0;
    scratch[t + cp.num_blocks + blk] = a1;
}

// R_B = T_{B+1} + z^BLOCK R_{B+1}, the quotient's value just below block B
// due to the later blocks, with R of the last block zero. One thread per batch.
@compute @workgroup_size(64)
fn scan_blocks(@builtin(global_invocation_id) id: vec3<u32>) {
    let b = id.x;
    if (b >= cp.num_batches) { return; }
    let t = t_off(b);
    let r = r_off(b);
    var a0 = Field(0u, 0u);
    var a1 = Field(0u, 0u);
    var blk = cp.num_blocks - 1u;
    loop {
        scratch[r + blk] = a0;
        scratch[r + cp.num_blocks + blk] = a1;
        if (blk == 0u) { break; }
        let zs0 = scratch[zp_off(b) + BLOCK];
        let zs1 = scratch[zp_off(b) + BLOCK + 1u + BLOCK];
        let m = ext_mul(a0, a1, zs0, zs1);
        a0 = add_field(m[0], scratch[t + blk]);
        a1 = add_field(m[1], scratch[t + cp.num_blocks + blk]);
        blk--;
    }
}

// q_k = l_k + z^(E_B - 1 - k) R_B per batch, with E_B the end of k's block,
// scaled and summed over the batches; zero beyond the degree.
@compute @workgroup_size(64)
fn combine_quotients(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    if (k >= cp.n) { return; }
    var f0 = Field(0u, 0u);
    var f1 = Field(0u, 0u);
    if (k < cp.degree) {
        let blk = k / BLOCK;
        let e = (blk + 1u) * BLOCK - 1u - k;
        for (var b = 0u; b < cp.num_batches; b++) {
            let l = l_off(b);
            let r = r_off(b);
            let z = zp_off(b);
            let m = ext_mul(scratch[z + e], scratch[z + BLOCK + 1u + e],
                            scratch[r + blk], scratch[r + cp.num_blocks + blk]);
            let q0 = add_field(scratch[l + k], m[0]);
            let q1 = add_field(scratch[l + cp.degree + k], m[1]);
            let s = ext_mul(q0, q1, cp.scales[b].xy, cp.scales[b].zw);
            f0 = add_field(f0, s[0]);
            f1 = add_field(f1, s[1]);
        }
    }
    out[k] = f0;
    out[cp.n + k] = f1;
}
"#;

/// The opening module: `point_powers` and `eval_openings`, the committed
/// polynomials at the opening points.
pub fn open_module(flavor: Flavor) -> String {
    let mut out = prelude(flavor);
    out.push_str(EXT_POW);
    out.push_str(OPENING_KERNELS);
    out
}

/// Coefficients per block of the combination module's synthetic division.
pub const COMBINE_BLOCK: u32 = 256;

/// The combination module: `batch_powers`, `reduce_batch`, `divide_blocks`,
/// `scan_blocks` and `combine_quotients`, the coefficients of FRI's initial
/// polynomial from the committed coefficients.
pub fn combine_module(flavor: Flavor) -> String {
    let mut out = prelude(flavor);
    out.push_str(EXT_POW);
    out.push_str(COMBINE_KERNELS);
    out
}

const PERMUTATION_KERNELS: &str = r#"
struct PermParams {
    // Rows (the degree).
    degree: u32,
    num_routed: u32,
    // Chunks of max_degree routed wires per row; the last chunk's product
    // belongs to Z(gx), the others to the partial products.
    num_chunks: u32,
    max_degree: u32,
    num_challenges: u32,
    // Blocks of BLOCK rows of the Z scan.
    num_blocks: u32,
    // Offset in consts of beta_c * k_j (per challenge, num_routed each).
    bk_offset: u32,
    pad: u32,
}
@group(0) @binding(0) var<uniform> pp: PermParams;
// The wire values, column-major, degree per wire.
@group(0) @binding(1) var<storage, read> wires: array<Field>;
// The sigma values, column-major, degree per routed wire.
@group(0) @binding(2) var<storage, read> sigmas: array<Field>;
// Per challenge beta then gamma; then from bk_offset per challenge beta * k_j.
@group(0) @binding(3) var<storage, read> consts: array<Field>;
// g^k for k < degree / 2, g the primitive degree-th root of unity.
@group(0) @binding(4) var<storage, read> twiddles: array<Field>;
// Per challenge: the chunk products, row-major (num_chunks per row); then
// per challenge the block products, then the block prefixes.
@group(0) @binding(5) var<storage, read_write> scratch: array<Field>;
// The committed columns: Z per challenge, then the partial products of
// every challenge (num_chunks - 1 per challenge), degree per column.
@group(0) @binding(6) var<storage, read_write> out: array<Field>;

const BLOCK: u32 = 256u;
const MAX_ROUTED: u32 = 128u;

fn chunks_off(c: u32) -> u32 { return c * pp.degree * pp.num_chunks; }
fn blocks_off(c: u32) -> u32 { return pp.num_challenges * pp.degree * pp.num_chunks + c * pp.num_blocks; }
fn prefix_off(c: u32) -> u32 { return pp.num_challenges * (pp.degree * pp.num_chunks + pp.num_blocks) + c * pp.num_blocks; }


// Row id.x, challenge id.y: the products over chunks of the quotients
// (w_j + beta k_j x + gamma) / (w_j + beta sigma_j + gamma), with the
// denominators inverted together.
@compute @workgroup_size(64)
fn row_products(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let c = id.y;
    if (i >= pp.degree) { return; }
    let half = pp.degree >> 1u;
    var x = twiddles[i & (half - 1u)];
    if (i >= half) { x = sub_field(Field(0u, 0u), x); }
    let beta = consts[2u * c];
    let gamma = consts[2u * c + 1u];
    var den: array<Field, MAX_ROUTED>;
    var acc = Field(1u, 0u);
    for (var j = 0u; j < pp.num_routed; j++) {
        let w = wires[j * pp.degree + i];
        let d = add_field(add_field(w, mul_field(beta, sigmas[j * pp.degree + i])), gamma);
        den[j] = acc;
        acc = mul_field(acc, d);
    }
    var inv = inv_field(acc);
    let base = chunks_off(c) + i * pp.num_chunks;
    var chunk = Field(1u, 0u);
    var k = pp.num_routed;
    loop {
        k--;
        let w = wires[k * pp.degree + i];
        let d = add_field(add_field(w, mul_field(beta, sigmas[k * pp.degree + i])), gamma);
        let n = add_field(add_field(w, mul_field(consts[pp.bk_offset + c * pp.num_routed + k], x)), gamma);
        // 1 / d_k = inv * prod_{j < k} d_j, then drop d_k from inv.
        let q = mul_field(n, mul_field(inv, den[k]));
        inv = mul_field(inv, d);
        chunk = mul_field(chunk, q);
        if (k % pp.max_degree == 0u) {
            scratch[base + k / pp.max_degree] = chunk;
            chunk = Field(1u, 0u);
        }
        if (k == 0u) { break; }
    }
}

// Block id.x of challenge id.y: the product of its rows' chunk products.
@compute @workgroup_size(64)
fn block_products(@builtin(global_invocation_id) id: vec3<u32>) {
    let b = id.x;
    let c = id.y;
    if (b >= pp.num_blocks) { return; }
    var acc = Field(1u, 0u);
    let base = chunks_off(c) + b * BLOCK * pp.num_chunks;
    for (var e = 0u; e < BLOCK * pp.num_chunks; e++) {
        acc = mul_field(acc, scratch[base + e]);
    }
    scratch[blocks_off(c) + b] = acc;
}

// Challenge id.x: Z at the start of every block, as a prefix product.
@compute @workgroup_size(64)
fn block_prefix(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = id.x;
    if (c >= pp.num_challenges) { return; }
    var acc = Field(1u, 0u);
    for (var b = 0u; b < pp.num_blocks; b++) {
        scratch[prefix_off(c) + b] = acc;
        acc = mul_field(acc, scratch[blocks_off(c) + b]);
    }
}

// Block id.x of challenge id.y: Z and the partial products of its rows.
@compute @workgroup_size(64)
fn fill_columns(@builtin(global_invocation_id) id: vec3<u32>) {
    let b = id.x;
    let c = id.y;
    if (b >= pp.num_blocks) { return; }
    var z = scratch[prefix_off(c) + b];
    let num_prods = pp.num_chunks - 1u;
    for (var r = 0u; r < BLOCK; r++) {
        let i = b * BLOCK + r;
        out[c * pp.degree + i] = z;
        var acc = z;
        let base = chunks_off(c) + i * pp.num_chunks;
        for (var k = 0u; k < num_prods; k++) {
            acc = mul_field(acc, scratch[base + k]);
            out[(pp.num_challenges + c * num_prods + k) * pp.degree + i] = acc;
        }
        z = mul_field(acc, scratch[base + num_prods]);
    }
}
"#;

/// Rows per block of the permutation module's Z scan.
pub const PERMUTATION_BLOCK: u32 = 256;

/// The permutation module: `row_products`, `block_products`, `block_prefix`
/// and `fill_columns`, the partial products and Z of the permutation
/// argument from the wire and sigma values.
pub fn permutation_module(flavor: Flavor) -> String {
    let mut out = prelude(flavor);
    out.push_str(PERMUTATION_KERNELS);
    out
}
