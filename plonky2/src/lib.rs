#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::cloned_ref_to_slice_refs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
#![cfg_attr(not(feature = "std"), no_std)]
// The auto-trait proofs for the wgpu types held by the GPU context exceed the default depth.
#![cfg_attr(feature = "gpu", recursion_limit = "512")]

#[cfg(not(feature = "std"))]
pub extern crate alloc;

/// Re-export of `plonky2_field`.
#[doc(inline)]
pub use plonky2_field as field;

pub mod batch_fri;
pub mod fri;
pub mod gadgets;
pub mod gates;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod hash;
pub mod iop;
pub mod plonk;
pub mod recursion;
pub mod util;

#[cfg(test)]
mod lookup_test;
