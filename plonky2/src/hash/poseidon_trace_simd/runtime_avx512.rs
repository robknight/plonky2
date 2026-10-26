//! Private runtime-dispatched Poseidon arithmetic. No vector type escapes this module.
use core::arch::x86_64::*;
use core::mem::transmute;
use core::ops::{Add, Mul, Sub};

use super::{Lanes, Word, G};
use crate::field::types::{Field, Field64};
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
struct Vector([G; 8]);
impl Vector {
    #[inline(always)]
    fn get(self) -> __m512i {
        unsafe { transmute(self) }
    }
    #[inline(always)]
    fn new(x: __m512i) -> Self {
        unsafe { transmute(x) }
    }
}
// Vector is private and constructed only inside the feature-checked entry.
// Inlining trace_impl and these adapters exposes the existing arithmetic to
// the feature-enabled root while keeping the rest of the binary portable.
macro_rules! operation {
    ($trait:ident,$method:ident) => {
        impl $trait for Vector {
            type Output = Self;
            #[inline(always)]
            fn $method(self, rhs: Self) -> Self {
                Self::new(unsafe { $method(self.get(), rhs.get()) })
            }
        }
    };
}
operation!(Add, add);
operation!(Sub, sub);
operation!(Mul, mul);
impl Word for Vector {
    #[inline(always)]
    fn c(x: u64) -> Self {
        Self([G::from_canonical_u64(x); 8])
    }
    #[inline(always)]
    fn sq(self) -> Self {
        Self::new(unsafe { square(self.get()) })
    }
}
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512cd")]
pub(super) unsafe fn trace(s: [Lanes<8>; 12], swap: Lanes<8>) -> [Lanes<8>; 122] {
    // Both Copy wrappers are repr(transparent) over [G; 8], with identical
    // alignment and valid bit patterns. These are representation conversions.
    let input: [Vector; 12] = transmute(s);
    let output = super::trace_impl::<Vector, true>(input, Vector(swap.0));
    transmute(output)
}
// Arithmetic follows the existing field AVX-512 backend, including its
// noncanonical intermediate representation and specialized squaring.
const FIELD_ORDER: __m512i = unsafe { transmute([G::ORDER; 8]) };
const EPSILON: __m512i = unsafe { transmute([G::ORDER.wrapping_neg(); 8]) };

#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512cd")]
unsafe fn canonicalize(x: __m512i) -> __m512i {
    let mask = _mm512_cmpge_epu64_mask(x, FIELD_ORDER);
    _mm512_mask_sub_epi64(x, mask, x, FIELD_ORDER)
}

#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512cd")]
unsafe fn add_no_double_overflow_64_64(x: __m512i, y: __m512i) -> __m512i {
    let res_wrapped = _mm512_add_epi64(x, y);
    let mask = _mm512_cmplt_epu64_mask(res_wrapped, y); // mask set if add overflowed
    _mm512_mask_sub_epi64(res_wrapped, mask, res_wrapped, FIELD_ORDER)
}

#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512cd")]
unsafe fn sub_no_double_overflow_64_64(x: __m512i, y: __m512i) -> __m512i {
    let mask = _mm512_cmplt_epu64_mask(x, y); // mask set if sub will underflow (x < y)
    let res_wrapped = _mm512_sub_epi64(x, y);
    _mm512_mask_add_epi64(res_wrapped, mask, res_wrapped, FIELD_ORDER)
}

#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512cd")]
unsafe fn add(x: __m512i, y: __m512i) -> __m512i {
    add_no_double_overflow_64_64(x, canonicalize(y))
}

#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512cd")]
unsafe fn sub(x: __m512i, y: __m512i) -> __m512i {
    sub_no_double_overflow_64_64(x, canonicalize(y))
}

const LO_32_BITS_MASK: __mmask16 = 0b0101010101010101u16;

#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512cd")]
unsafe fn mul64_64(x: __m512i, y: __m512i) -> (__m512i, __m512i) {
    // We want to move the high 32 bits to the low position. The multiplication instruction ignores
    // the high 32 bits, so it's ok to just duplicate it into the low position. This duplication can
    // be done on port 5; bitshifts run on port 0, competing with multiplication.
    //   This instruction is only provided for 32-bit floats, not integers. Idk why Intel makes the
    // distinction; the casts are free and it guarantees that the exact bit pattern is preserved.
    // Using a swizzle instruction of the wrong domain (float vs int) does not increase latency
    // since Haswell.
    let x_hi = _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(x)));
    let y_hi = _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(y)));

    // All four pairwise multiplications
    let mul_ll = _mm512_mul_epu32(x, y);
    let mul_lh = _mm512_mul_epu32(x, y_hi);
    let mul_hl = _mm512_mul_epu32(x_hi, y);
    let mul_hh = _mm512_mul_epu32(x_hi, y_hi);

    // Bignum addition
    // Extract high 32 bits of mul_ll and add to mul_hl. This cannot overflow.
    let mul_ll_hi = _mm512_srli_epi64::<32>(mul_ll);
    let t0 = _mm512_add_epi64(mul_hl, mul_ll_hi);
    // Extract low 32 bits of t0 and add to mul_lh. Again, this cannot overflow.
    // Also, extract high 32 bits of t0 and add to mul_hh.
    let t0_lo = _mm512_and_si512(t0, EPSILON);
    let t0_hi = _mm512_srli_epi64::<32>(t0);
    let t1 = _mm512_add_epi64(mul_lh, t0_lo);
    let t2 = _mm512_add_epi64(mul_hh, t0_hi);
    // Lastly, extract the high 32 bits of t1 and add to t2.
    let t1_hi = _mm512_srli_epi64::<32>(t1);
    let res_hi = _mm512_add_epi64(t2, t1_hi);

    // Form res_lo by combining the low half of mul_ll with the low half of t1 (shifted into high
    // position).
    let t1_lo = _mm512_castps_si512(_mm512_moveldup_ps(_mm512_castsi512_ps(t1)));
    let res_lo = _mm512_mask_blend_epi32(LO_32_BITS_MASK, t1_lo, mul_ll);

    (res_hi, res_lo)
}

#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512cd")]
unsafe fn square64(x: __m512i) -> (__m512i, __m512i) {
    // Get high 32 bits of x. See comment in mul64_64_s.
    let x_hi = _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(x)));

    // All pairwise multiplications.
    let mul_ll = _mm512_mul_epu32(x, x);
    let mul_lh = _mm512_mul_epu32(x, x_hi);
    let mul_hh = _mm512_mul_epu32(x_hi, x_hi);

    // Bignum addition, but mul_lh is shifted by 33 bits (not 32).
    let mul_ll_hi = _mm512_srli_epi64::<33>(mul_ll);
    let t0 = _mm512_add_epi64(mul_lh, mul_ll_hi);
    let t0_hi = _mm512_srli_epi64::<31>(t0);
    let res_hi = _mm512_add_epi64(mul_hh, t0_hi);

    // Form low result by adding the mul_ll and the low 31 bits of mul_lh (shifted to the high
    // position).
    let mul_lh_lo = _mm512_slli_epi64::<33>(mul_lh);
    let res_lo = _mm512_add_epi64(mul_ll, mul_lh_lo);

    (res_hi, res_lo)
}

#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512cd")]
unsafe fn reduce128(x: (__m512i, __m512i)) -> __m512i {
    let (hi0, lo0) = x;
    let hi_hi0 = _mm512_srli_epi64::<32>(hi0);
    let lo1 = sub_no_double_overflow_64_64(lo0, hi_hi0);
    let t1 = _mm512_mul_epu32(hi0, EPSILON);
    add_no_double_overflow_64_64(lo1, t1)
}

#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512cd")]
unsafe fn mul(x: __m512i, y: __m512i) -> __m512i {
    reduce128(mul64_64(x, y))
}

#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512cd")]
unsafe fn square(x: __m512i) -> __m512i {
    reduce128(square64(x))
}
