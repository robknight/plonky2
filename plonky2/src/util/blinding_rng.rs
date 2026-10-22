//! Short-lived CPU randomness for witness padding and polynomial salts.
//!
//! Each work unit has an independent OS seed. No seed is cached across jobs
//! or proofs, and no deterministic seed or cloning API is exposed.
use core::cell::RefCell;

use chacha20::rand_core::{Rng, SeedableRng};
use rand::rngs::OsRng;
use rand::{CryptoRng, RngCore};
use zeroize::Zeroizing;

use crate::field::types::Field;

/// Bridges rand 0.8's field sampler to RustCrypto's zeroizing ChaCha20 RNG.
/// The dependency clears both the key/counter state and buffered output on
/// drop; Zeroizing clears our explicit OS seed, including on unwind.
pub(crate) struct BlindingRng(chacha20::ChaCha20Rng);

impl BlindingRng {
    pub(crate) fn new() -> Self {
        let mut seed = Zeroizing::new([0u8; 32]);
        // OS failure must stop proving, never produce predictable padding.
        OsRng.fill_bytes(&mut *seed);
        Self(chacha20::ChaCha20Rng::from_seed(*seed))
    }
}

impl RngCore for BlindingRng {
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }
    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.fill_bytes(dest);
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}
impl CryptoRng for BlindingRng {}

#[derive(Default)]
struct State {
    active: bool,
    // Keep the live key at a stable heap address while scope guards move.
    rng: Option<Box<BlindingRng>>,
}
thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

struct Scope(State);
impl Drop for Scope {
    fn drop(&mut self) {
        // Restores an enclosing job when custom generators nest proving.
        // The displaced RNG is dropped and zeroized on success or unwind.
        STATE.with(|state| {
            state.replace(core::mem::take(&mut self.0));
        });
    }
}

pub(crate) fn with_scope<T>(work: impl FnOnce() -> T) -> T {
    let previous = STATE.with(|state| {
        state.replace(State {
            active: true,
            rng: None,
        })
    });
    let _scope = Scope(previous);
    work()
}

pub(crate) fn sample<F: Field>() -> F {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        if state.active {
            F::sample(
                state
                    .rng
                    .get_or_insert_with(|| Box::new(BlindingRng::new()))
                    .as_mut(),
            )
        } else {
            // Direct generator callers must not leave a long-lived TLS RNG.
            F::rand()
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField as F;
    use crate::field::types::Sample;

    #[test]
    fn matches_reference_stream_and_field_sampler() {
        let seed = [42; 32];
        let mut new = BlindingRng(chacha20::ChaCha20Rng::from_seed(seed));
        let mut reference = <rand_chacha::ChaCha20Rng as rand::SeedableRng>::from_seed(seed);
        for _ in 0..257 {
            assert_eq!(new.next_u32(), reference.next_u32());
            assert_eq!(new.next_u64(), reference.next_u64());
            let mut a = [0; 71];
            let mut b = [0; 71];
            new.fill_bytes(&mut a);
            reference.fill_bytes(&mut b);
            assert_eq!(a, b);
            assert_eq!(F::sample(&mut new), F::sample(&mut reference));
        }
        fn zeroizes<T: zeroize::ZeroizeOnDrop>() {}
        zeroizes::<chacha20::ChaCha20Rng>();
    }

    fn idle() {
        STATE.with(|state| {
            let state = state.borrow();
            assert!(!state.active);
            assert!(state.rng.is_none());
        });
    }

    #[test]
    fn scope_is_lazy_and_cleans_up_on_success_error_and_panic() {
        idle();
        with_scope(|| STATE.with(|s| assert!(s.borrow().rng.is_none())));
        idle();
        let result: Result<(), ()> = with_scope(|| {
            let _: F = sample();
            Err(())
        });
        assert!(result.is_err());
        idle();
        assert!(std::panic::catch_unwind(|| with_scope(|| {
            let _: F = sample();
            panic!("intentional unwind");
        }))
        .is_err());
        idle();
        let _: F = sample();
        idle();
    }

    #[test]
    fn independent_jobs_threads_and_nested_scopes() {
        let draw = || with_scope(|| (0..64).map(|_| sample::<F>()).collect::<Vec<_>>());
        let first = draw();
        assert_ne!(first, draw());
        assert_ne!(first, std::thread::spawn(draw).join().unwrap());
        with_scope(|| {
            let _: F = sample();
            let pos = STATE.with(|s| s.borrow().rng.as_ref().unwrap().0.get_word_pos());
            assert_ne!(first, draw());
            STATE.with(|s| assert_eq!(s.borrow().rng.as_ref().unwrap().0.get_word_pos(), pos));
            let _: F = sample();
        });
        idle();
    }
}
