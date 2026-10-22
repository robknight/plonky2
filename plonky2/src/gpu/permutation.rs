//! The partial products and Z of the permutation argument on the device.
//!
//! For every row and challenge, the quotients `(w_j + beta k_j x + gamma) /
//! (w_j + beta sigma_j(x) + gamma)` over the routed wires are multiplied in
//! chunks of the quotient degree, the denominators inverted together; Z is
//! the prefix product of the rows' chunk products, computed as a blocked
//! scan; and the committed columns (Z per challenge, then each challenge's
//! partial products) are written in the layout `from_values` expects, so
//! that the commitment proceeds from the device buffer. The wire values are
//! the ones the wires commitment kept on the device; the sigma values are
//! uploaded once per circuit.

use crate::field::extension::Extendable;
use crate::fri::oracle::PolynomialBatch;
use crate::gpu::lde::{from_device_values, upload_columns};
use crate::gpu::merkle::supported;
use crate::gpu::shader::{HASH_WORKGROUP, PERMUTATION_BLOCK};
use crate::gpu::{context, DeviceBuffer, GpuContext};
use crate::hash::hash_types::RichField;
use crate::plonk::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
use crate::plonk::config::GenericConfig;
use crate::timed;
use crate::util::timing::TimingTree;

/// The kernels take at most this many routed wires.
const MAX_ROUTED: usize = 128;

/// The device copy of the circuit's sigma values, column-major.
fn device_sigmas<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    ctx: &GpuContext,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    num_routed: usize,
) -> DeviceBuffer {
    if let Some(buffer) = prover_data.device_sigmas.get() {
        return buffer;
    }
    let degree = prover_data.sigmas.len();
    let columns = (0..num_routed)
        .map(|j| {
            (0..degree)
                .map(|i| prover_data.sigmas[i][j])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let slices = columns.iter().map(|c| c.as_slice()).collect::<Vec<_>>();
    let buffer = upload_columns(ctx, "sigma values", &slices);
    prover_data
        .device_sigmas
        .set_if_within_limit(ctx, buffer.clone());
    buffer
}

/// The commitment to the Z's and partial products, computed on the device
/// from the wires commitment's values, or `None` if the GPU does not take
/// it (lookups, no device values, or limits).
pub(crate) fn try_partial_products_commitment<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    wires_commitment: &PolynomialBatch<F, C, D>,
    betas: &[F],
    gammas: &[F],
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    timing: &mut TimingTree,
    plan: Option<&crate::gpu::policy::ExecutionPlan>,
) -> Option<PolynomialBatch<F, C, D>> {
    let config = &common_data.config;
    if !supported::<F, C::Hasher>()
        || !common_data.luts.is_empty()
        || common_data.num_lookup_polys != 0
        || config.num_routed_wires > MAX_ROUTED
        || config.num_routed_wires == 0
    {
        return None;
    }
    let ctx = context()?;
    let degree = common_data.degree();
    let num_challenges = config.num_challenges;
    let num_prods = common_data.num_partial_products;
    let max_degree = common_data.quotient_degree_factor;
    let num_chunks = config.num_routed_wires.div_ceil(max_degree);
    if num_chunks != num_prods + 1
        || degree % PERMUTATION_BLOCK as usize != 0
        || prover_data.sigmas.len() != degree
        || wires_commitment.degree_log != crate::util::log2_strict(degree)
        || betas.len() != num_challenges
        || gammas.len() != num_challenges
    {
        return None;
    }
    let wires = wires_commitment.device_values.get()?;
    let num_blocks = degree / PERMUTATION_BLOCK as usize;
    let num_columns = num_challenges * (1 + num_prods);
    let out = timed!(timing, "gpu: partial products and Z's", {
        let sigmas = device_sigmas(&ctx, prover_data, config.num_routed_wires);
        let mut consts = Vec::new();
        for (&beta, &gamma) in betas.iter().zip(gammas) {
            consts.push(beta.to_noncanonical_u64());
            consts.push(gamma.to_noncanonical_u64());
        }
        let bk_offset = consts.len();
        for &beta in betas {
            consts.extend(
                common_data
                    .k_is
                    .iter()
                    .map(|&k| (beta * k).to_noncanonical_u64()),
            );
        }
        let consts = ctx.upload("permutation constants", &consts);
        let params = ctx.uniform(
            "permutation params",
            &[
                degree as u32,
                config.num_routed_wires as u32,
                num_chunks as u32,
                max_degree as u32,
                num_challenges as u32,
                num_blocks as u32,
                bk_offset as u32,
                0,
            ],
        );
        let twiddles = ctx.twiddles(crate::util::log2_strict(degree) as u32);
        let scratch = ctx.storage(
            "permutation scratch",
            (num_challenges * (degree * num_chunks + 2 * num_blocks) * 8) as u64,
        );
        let out = ctx.storage("partial products", (num_columns * degree * 8) as u64);
        let bindings =
            ctx.bind_permutation(&[&params, &wires, &sigmas, &consts, &twiddles, &scratch, &out]);
        let mut encoder = ctx.encoder("partial products");
        {
            let mut pass = ctx.begin_pass(&mut encoder, "partial products", None);
            pass.set_bind_group(0, &bindings, &[]);
            let [rows, blocks, prefix, fill] = &ctx.permutation;
            pass.set_pipeline(rows);
            pass.dispatch_workgroups(
                (degree as u32).div_ceil(HASH_WORKGROUP),
                num_challenges as u32,
                1,
            );
            pass.set_pipeline(blocks);
            pass.dispatch_workgroups(
                (num_blocks as u32).div_ceil(HASH_WORKGROUP),
                num_challenges as u32,
                1,
            );
            pass.set_pipeline(prefix);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(fill);
            pass.dispatch_workgroups(
                (num_blocks as u32).div_ceil(HASH_WORKGROUP),
                num_challenges as u32,
                1,
            );
        }
        ctx.submit(encoder);
        out
    });
    Some(from_device_values(
        &ctx,
        out,
        num_columns,
        degree,
        config.fri_config.rate_bits,
        config.zero_knowledge
            && crate::plonk::plonk_common::PlonkOracle::ZS_PARTIAL_PRODUCTS.blinding,
        config.fri_config.cap_height,
        timing,
        plan.map_or_else(
            || ctx.options.retain_leaves(ctx.mappable),
            |p| p.retain_leaves(),
        ),
    ))
}
