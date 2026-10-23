//! Proof-of-work nonce search on the device. Every invocation tries
//! one candidate with the same exact permutation the Merkle kernels use, and
//! the smallest matching index of a batch wins, so the result is the smallest
//! valid candidate and does not depend on scheduling. Batches start at zero
//! and grow until a candidate is found; the caller rechecks the nonce with the
//! CPU permutation before the transcript observes it.
use std::sync::Arc;

use crate::gpu::{context, field_supported, module, pipeline, shader, storage_layout, GpuContext};
use crate::hash::hash_types::RichField;

pub(crate) struct PowKernel {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl core::fmt::Debug for PowKernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("PowKernel")
    }
}

/// Candidates in the first batch; later batches double up to `MAX_BATCH`.
const FIRST_BATCH: u32 = 1 << 17;
const MAX_BATCH: u32 = 1 << 20;
/// The search covers up to `2^(min_leading_zeros + BUDGET_BITS)` candidates,
/// so it fails to find a nonce with probability about `exp(-2^BUDGET_BITS)`.
const BUDGET_BITS: u32 = 4;

/// The smallest candidate `c` such that the permutation of `state` with word
/// `position` replaced by `c` has a response with at least
/// `min_leading_zeros` leading zero bits, searched on the device. `None` when
/// the device or field is unsupported, or no candidate was found within the
/// bounded number of batches.
pub(crate) fn search<F: RichField>(
    state: &[F],
    position: usize,
    min_leading_zeros: u32,
) -> Option<u64> {
    if !field_supported::<F>() || state.len() != 12 || position >= 12 {
        return None;
    }
    if min_leading_zeros == 0 {
        return Some(0);
    }
    let ctx = context()?;
    search_on(&ctx, state, position, min_leading_zeros)
}

/// The compiled kernel, or `None` when compilation failed (the search then
/// falls back to the CPU instead of failing the proof).
fn kernel(ctx: &GpuContext) -> Option<&PowKernel> {
    ctx.pow_kernel
        .get_or_init(|| {
            let validation = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let source = shader::pow_module(ctx.flavor);
            let module = module(&ctx.device, "proof of work", &source);
            let layout = storage_layout(
                &ctx.device,
                "proof of work",
                &[(true, true), (false, true), (false, false)],
            );
            let pipeline = pipeline(&ctx.device, &layout, &module, "pow_search");
            if let Some(error) = pollster::block_on(validation.pop()) {
                log::warn!("gpu: proof-of-work kernel rejected, using the CPU search: {error}");
                return None;
            }
            Some(PowKernel { layout, pipeline })
        })
        .as_ref()
}

/// Joins the warm-up thread when dropped, so the thread's context reference
/// never outlives the proof that started it, even if the proof fails early.
pub(crate) struct WarmGuard(Option<std::thread::JoinHandle<()>>);

impl Drop for WarmGuard {
    fn drop(&mut self) {
        if let Some(handle) = self.0.take() {
            let _ = handle.join();
        }
    }
}

/// Compiles the search kernel on a background thread while a proof runs, so
/// its first use (about 0.2 s on the M4) is off the critical path; a later
/// search waits for the same one-time initialisation. Hold the returned guard
/// for the whole proof: dropping it joins the thread.
pub(crate) fn warm(ctx: &Arc<GpuContext>) -> WarmGuard {
    if ctx.pow_kernel.get().is_some() {
        return WarmGuard(None);
    }
    let ctx = ctx.clone();
    let handle = std::thread::Builder::new()
        .name("plonky2-gpu-pow-compile".into())
        .spawn(move || {
            kernel(&ctx);
        })
        .ok();
    WarmGuard(handle)
}

pub(crate) fn search_on<F: RichField>(
    ctx: &Arc<GpuContext>,
    state: &[F],
    position: usize,
    min_leading_zeros: u32,
) -> Option<u64> {
    if min_leading_zeros == 0 {
        return Some(0);
    }
    let capacity = ctx.max_workgroups.saturating_mul(64).min(MAX_BATCH);
    if capacity == 0 {
        return None;
    }
    let kernel = kernel(ctx)?;
    let words = state
        .iter()
        .map(|x| x.to_canonical_u64())
        .collect::<Vec<_>>();
    let state_buffer = ctx.upload("proof of work state", &words);
    let budget = 1u64 << min_leading_zeros.saturating_add(BUDGET_BITS).min(63);
    let mut base = 0u64;
    let mut count = FIRST_BATCH;
    while base < budget {
        let batch_count = (budget - base).min(u64::from(count.min(capacity))) as u32;
        // Candidates stay canonical field elements.
        if base.checked_add(batch_count as u64)? > F::ORDER {
            return None;
        }
        let found = ctx.upload("proof of work result", &[u64::MAX, u64::MAX]);
        let params = ctx.uniform(
            "proof of work params",
            &[
                position as u32,
                base as u32,
                (base >> 32) as u32,
                batch_count,
                min_leading_zeros,
                0,
                0,
                0,
            ],
        );
        let bindings = ctx.bind(&kernel.layout, &[&params, &state_buffer, &found]);
        let mut encoder = ctx.encoder("proof of work search");
        {
            let mut pass = ctx.begin_pass(&mut encoder, "proof of work search", None);
            pass.set_pipeline(&kernel.pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(batch_count.div_ceil(64), 1, 1);
        }
        let done = ctx.submit_async(encoder);
        let index = ctx.read_after(&found, 16, Some(&done), |bytes| {
            u32::from_le_bytes(bytes[..4].try_into().unwrap())
        });
        if index != u32::MAX {
            return Some(base + index as u64);
        }
        base += batch_count as u64;
        count = (count * 2).min(MAX_BATCH);
    }
    None
}
