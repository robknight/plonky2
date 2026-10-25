//! Trace-preserving SIMD across independent Poseidon permutations.
use core::ops::{Add, Mul, Sub};

use super::poseidon::{Poseidon, ALL_ROUND_CONSTANTS};
use crate::field::goldilocks_field::GoldilocksField as G;
use crate::field::ops::Square;
use crate::field::types::Field;

pub trait Word: Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> {
    fn c(x: u64) -> Self;
    fn sq(self) -> Self {
        self * self
    }
}
impl Word for G {
    #[inline(always)]
    fn c(x: u64) -> Self {
        G::from_canonical_u64(x)
    }
    #[inline(always)]
    fn sq(self) -> Self {
        self.square()
    }
}

/// Each lane is one independent permutation. Scalar emulation is used on ARM.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Lanes<const N: usize>(pub [G; N]);
macro_rules! lane_op { ($tr:ident,$fn:ident,$op:tt) => {
impl<const N:usize> $tr for Lanes<N> { type Output=Self;
    #[inline(always)] fn $fn(self, rhs:Self)->Self {

        #[cfg(all(target_arch="x86_64",target_feature="avx512bw",target_feature="avx512cd",target_feature="avx512dq",target_feature="avx512f",target_feature="avx512vl"))]
        if N==8 { use crate::field::WitnessAvx512 as V;
            let a=V(core::array::from_fn(|i|self.0[i])); let b=V(core::array::from_fn(|i|rhs.0[i]));
            let r=(a $op b).0; return Self(core::array::from_fn(|i|r[i]));
        }
        Self(core::array::from_fn(|i|self.0[i] $op rhs.0[i]))
    }
} }; }
lane_op!(Add,add,+);
lane_op!(Sub,sub,-);
lane_op!(Mul,mul,*);
impl<const N: usize> Word for Lanes<N> {
    #[inline(always)]
    fn c(x: u64) -> Self {
        Self([G::from_canonical_u64(x); N])
    }
    #[inline(always)]
    fn sq(self) -> Self {
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512f",
            target_feature = "avx512vl"
        ))]
        if N == 8 {
            use crate::field::WitnessAvx512 as V;
            let r = V(core::array::from_fn(|i| self.0[i])).square().0;
            return Self(core::array::from_fn(|i| r[i]));
        }
        Self(core::array::from_fn(|i| self.0[i].square()))
    }
}
#[inline(always)]
fn pow7<W: Word>(x: W) -> W {
    let x2 = x.sq();
    let x4 = x2.sq();
    (x * x2) * x4
}
#[inline(always)]
fn scale<W: Word>(x: W, n: i64) -> W {
    let x2 = x + x;
    let x4 = x2 + x2;
    let x8 = x4 + x4;
    let x16 = x8 + x8;
    let x32 = x16 + x16;
    match n {
        0 => W::c(0),
        1 => x,
        -1 => W::c(0) - x,
        2 => x2,
        -2 => W::c(0) - x2,
        4 => x4,
        -4 => W::c(0) - x4,
        8 => x8,
        -8 => W::c(0) - x8,
        16 => x16,
        32 => x32,
        _ => x * W::c(n as u64),
    }
}
#[inline(always)]
#[unroll::unroll_for_loops]
fn sbox_all<W: Word, const N: usize>(s: &mut [W; N]) {
    for i in 0..12 {
        if i < N {
            s[i] = pow7(s[i]);
        }
    }
}

/// Field version of the existing integer FFT MDS network. All constants are small.
#[inline(always)]
pub fn mds_fft<W: Word>(s: [W; 12]) -> [W; 12] {
    #[inline(always)]
    fn fft<W: Word>(x: [W; 4]) -> (W, (W, W), W) {
        let a = x[0] + x[2];
        let b = x[0] - x[2];
        let c = x[1] + x[3];
        let d = x[1] - x[3];
        (a + c, (b, W::c(0) - d), a - c)
    }
    #[inline(always)]
    fn ifft<W: Word>(x: (W, (W, W), W)) -> [W; 4] {
        let a = x.0 + x.2;
        let b = x.0 - x.2;
        [a + x.1 .0, b - x.1 .1, a - x.1 .0, b + x.1 .1]
    }
    #[inline(always)]
    fn cm<W: Word>(a: (W, W), b: (i64, i64)) -> (W, W) {
        (
            scale(a.0, b.0) - scale(a.1, b.1),
            scale(a.0, b.1) + scale(a.1, b.0),
        )
    }
    #[inline(always)]
    fn ca<W: Word>(a: (W, W), b: (W, W)) -> (W, W) {
        (a.0 + b.0, a.1 + b.1)
    }
    #[inline(always)]
    fn mi<W: Word>(a: (W, W)) -> (W, W) {
        (a.1, W::c(0) - a.0)
    }
    let a = fft([s[0], s[3], s[6], s[9]]);
    let b = fft([s[1], s[4], s[7], s[10]]);
    let c = fft([s[2], s[5], s[8], s[11]]);
    let d = [
        scale(a.0, 16) + scale(b.0, 16) + scale(c.0, 32),
        scale(a.0, 32) + scale(b.0, 16) + scale(c.0, 16),
        scale(a.0, 16) + scale(b.0, 32) + scale(c.0, 16),
    ];
    let e = [
        ca(
            ca(cm(a.1, (2, -1)), mi(cm(b.1, (16, 1)))),
            mi(cm(c.1, (-4, 1))),
        ),
        ca(ca(cm(a.1, (-4, 1)), cm(b.1, (2, -1))), mi(cm(c.1, (16, 1)))),
        ca(ca(cm(a.1, (16, 1)), cm(b.1, (-4, 1))), cm(c.1, (2, -1))),
    ];
    let f = [
        scale(a.2, -1) - scale(b.2, 2) - scale(c.2, -8),
        scale(a.2, -8) + scale(b.2, -1) - scale(c.2, 2),
        scale(a.2, 2) + scale(b.2, -8) + scale(c.2, -1),
    ];
    let a = ifft((d[0], e[0], f[0]));
    let b = ifft((d[1], e[1], f[1]));
    let c = ifft((d[2], e[2], f[2]));
    [
        a[0] + scale(s[0], 8),
        b[0],
        c[0],
        a[1],
        b[1],
        c[1],
        a[2],
        b[2],
        c[2],
        a[3],
        b[3],
        c[3],
    ]
}
#[inline(always)]
#[unroll::unroll_for_loops]
fn partial_init<W: Word>(s: [W; 12]) -> [W; 12] {
    core::array::from_fn(|j| {
        if j == 0 {
            s[0]
        } else {
            let mut v = W::c(0);
            for i in 1..12 {
                v = v + s[i] * W::c(G::FAST_PARTIAL_ROUND_INITIAL_MATRIX[i - 1][j - 1]);
            }
            v
        }
    })
}
#[inline(always)]
#[unroll::unroll_for_loops]
fn partial<W: Word>(s: [W; 12], r: usize) -> [W; 12] {
    let mut d = s[0] * W::c(G::MDS_MATRIX_CIRC[0] + G::MDS_MATRIX_DIAG[0]);
    for i in 1..12 {
        d = d + s[i] * W::c(G::FAST_PARTIAL_ROUND_W_HATS[r][i - 1]);
    }
    core::array::from_fn(|i| {
        if i == 0 {
            d
        } else {
            s[i] + s[0] * W::c(G::FAST_PARTIAL_ROUND_VS[r][i - 1])
        }
    })
}
/// Emits exactly the 122 ordinary Poseidon gate witness values. Swap must be 0/1.
#[inline(always)]
pub fn trace<W: Word>(s: [W; 12], swap: W) -> [W; 122] {
    trace_core(s, swap)
}
#[inline(never)]
fn trace_core<W: Word>(mut s: [W; 12], swap: W) -> [W; 122] {
    let mut out = [W::c(0); 122];
    let mut pos = 0;
    for i in 0..4 {
        let d = swap * (s[i + 4] - s[i]);
        out[pos] = d;
        pos += 1;
        s[i] = s[i] + d;
        s[i + 4] = s[i + 4] - d;
    }
    let mut round = 0;
    for r in 0..4 {
        for i in 0..12 {
            s[i] = s[i] + W::c(ALL_ROUND_CONSTANTS[round * 12 + i]);
        }
        if r != 0 {
            out[pos..pos + 12].copy_from_slice(&s);
            pos += 12;
        }
        sbox_all(&mut s);
        s = mds_fft(s);
        round += 1;
    }
    for i in 0..12 {
        s[i] = s[i] + W::c(G::FAST_PARTIAL_FIRST_ROUND_CONSTANT[i]);
    }
    s = partial_init(s);
    for r in 0..22 {
        out[pos] = s[0];
        pos += 1;
        s[0] = pow7(s[0]);
        if r != 21 {
            s[0] = s[0] + W::c(G::FAST_PARTIAL_ROUND_CONSTANTS[r]);
        }
        s = partial(s, r);
    }
    round += 22;
    for _ in 0..4 {
        for i in 0..12 {
            s[i] = s[i] + W::c(ALL_ROUND_CONSTANTS[round * 12 + i]);
        }
        out[pos..pos + 12].copy_from_slice(&s);
        pos += 12;
        sbox_all(&mut s);
        s = mds_fft(s);
        round += 1;
    }
    out[pos..pos + 12].copy_from_slice(&s);
    debug_assert_eq!(pos + 12, 122);
    out
}

/// Independent scalar reference using the production scalar/NEON primitives.
pub fn reference(mut s: [G; 12], swap: G) -> [G; 122] {
    let mut out = [G::ZERO; 122];
    let mut p = 0;
    for i in 0..4 {
        out[p] = swap * (s[i + 4] - s[i]);
        p += 1;
    }
    if swap == G::ONE {
        for i in 0..4 {
            s.swap(i, i + 4);
        }
    }
    let mut round = 0;
    for r in 0..4 {
        G::constant_layer_field(&mut s, round);
        if r != 0 {
            out[p..p + 12].copy_from_slice(&s);
            p += 12;
        }
        G::sbox_layer(&mut s);
        s = G::mds_layer(&s);
        round += 1;
    }
    G::partial_first_constant_layer(&mut s);
    s = G::mds_partial_layer_init(&s);
    for r in 0..22 {
        out[p] = s[0];
        p += 1;
        s[0] = G::sbox_monomial(s[0]);
        if r != 21 {
            s[0] += G::from_canonical_u64(G::FAST_PARTIAL_ROUND_CONSTANTS[r]);
        }
        s = G::mds_partial_layer_fast(&s, r);
    }
    round += 22;
    for _ in 0..4 {
        G::constant_layer_field(&mut s, round);
        out[p..p + 12].copy_from_slice(&s);
        p += 12;
        G::sbox_layer(&mut s);
        s = G::mds_layer(&s);
        round += 1;
    }
    out[p..p + 12].copy_from_slice(&s);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_trace_values() {
        let mut seed = 42u64;
        let edges = [
            0,
            1,
            0xffffffff00000000,
            0xffffffff00000001,
            0xffffffff00000002,
            u64::MAX,
        ];
        for n in 0..260 {
            let input: [Lanes<8>; 12] = core::array::from_fn(|i| {
                Lanes(core::array::from_fn(|j| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    G(if n < edges.len() {
                        edges[(n + i + j) % edges.len()]
                    } else {
                        seed
                    })
                }))
            });
            let swap = Lanes(core::array::from_fn(|i| {
                G::from_canonical_usize((n + i) % 2)
            }));
            let output = trace(input, swap);
            for lane in 0..8 {
                let s = input.map(|x| x.0[lane]);
                let expected = reference(s, swap.0[lane]);
                assert_eq!(mds_fft(s), G::mds_layer(&s));
                for i in 0..122 {
                    assert_eq!(
                        output[i].0[lane], expected[i],
                        "trial={n} lane={lane} value={i}"
                    );
                }
                let mut swapped = s;
                if swap.0[lane] == G::ONE {
                    for i in 0..4 {
                        swapped.swap(i, i + 4);
                    }
                }
                assert_eq!(expected[110..], G::poseidon(swapped));
            }
        }
    }
}
pub(crate) const NATIVE: bool = cfg!(all(
    target_arch = "x86_64",
    target_feature = "avx512bw",
    target_feature = "avx512cd",
    target_feature = "avx512dq",
    target_feature = "avx512f",
    target_feature = "avx512vl"
));
// Benchmark-only per-proof control. Call only while no witness execution is active.
#[doc(hidden)]
#[cfg(feature = "witness-benchmarks")]
pub static BENCH_MODE: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(255);
#[doc(hidden)]
#[cfg(feature = "witness-benchmarks")]
pub fn benchmark_override(mode: u8, _minimum: u32) {
    assert!(matches!(mode, 0 | 5 | 255));
    BENCH_MODE.store(mode, core::sync::atomic::Ordering::Relaxed);
}
