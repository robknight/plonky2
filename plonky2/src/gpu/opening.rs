//! The opening phase on the device.
//!
//! The openings are dot products of each committed polynomial's coefficients
//! with the powers of the opening point, one workgroup per polynomial, read
//! from the coefficient buffers the GPU commitments keep (the circuit's fixed
//! commitment is uploaded and cached within the retained-buffer budget).
//! FRI's initial polynomial is
//! then built from the same coefficients as `prove_openings` builds it on the
//! CPU: for each opening batch the alpha-weighted sum of its polynomials, the
//! synthetic division by `X - z` (a blocked scan: within each block of
//! coefficients, then across the blocks), and the alpha-shifted sum of the
//! batches' quotients, all in coefficient form on the degree of the batch;
//! one coset FFT gives the codeword that the commit phase folds. The trees,
//! the final polynomial and the challenger state are the ones the CPU
//! produces. Only the openings, caps, sparse query data and final coefficients
//! are read back; the initial codeword and folded layers stay on the device.

use crate::field::extension::{Extendable, FieldExtension};
use crate::field::types::Field;
use crate::fri::oracle::PolynomialBatch;
use crate::fri::proof::FriProof;
use crate::fri::prover::{fri_proof_of_work, fri_prover_query_rounds};
use crate::fri::structure::FriInstanceInfo;
use crate::fri::FriParams;
use crate::gpu::fri::{commit_phase_fits, committed_trees, read_extension};
use crate::gpu::lde::{device_coefficients, fft, FftJob};
use crate::gpu::merkle::supported;
use crate::gpu::shader::{fft_plan, COMBINE_BLOCK, HASH_WORKGROUP};
use crate::gpu::{context, field_supported, DeviceBuffer, GpuContext};
use crate::hash::hash_types::RichField;
use crate::iop::challenger::Challenger;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::config::GenericConfig;
use crate::plonk::proof::OpeningSet;
use crate::timed;
use crate::util::timing::TimingTree;

/// The kernels take at most this many oracles, batches and points.
const MAX_ORACLES: usize = 4;

/// A polynomial as `(oracle, column)`, packed as the kernels' `vec2<u32>`,
/// with the index of the point it is evaluated at in the high half of the
/// first word.
fn term(oracle: usize, column: usize, point: usize) -> u64 {
    oracle as u64 | ((point as u64) << 16) | ((column as u64) << 32)
}

/// Whether the batch has `2^degree_log` coefficients per polynomial.
fn uniform_degree<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    batch: &PolynomialBatch<F, C, D>,
    degree_log: usize,
) -> bool {
    !batch.polynomials.is_empty()
        && batch.degree_log == degree_log
        && batch.polynomials.iter().all(|p| p.len() == 1 << degree_log)
}

/// The components of an extension element as kernel words.
fn words<F: RichField + Extendable<D>, const D: usize>(x: F::Extension) -> [u64; D] {
    x.to_basefield_array()
        .map(|component| component.to_noncanonical_u64())
}

/// An extension element as the four words of a uniform `vec4<u32>`.
fn uniform_words<F: RichField + Extendable<D>, const D: usize>(x: F::Extension) -> [u32; 4] {
    let mut out = [0; 4];
    for (i, word) in words::<F, D>(x).iter().enumerate().take(2) {
        out[2 * i] = *word as u32;
        out[2 * i + 1] = (*word >> 32) as u32;
    }
    out
}

/// `coeffs[oracle][column]` at `points[point]` for every term.
fn evaluate<F: RichField + Extendable<D>, const D: usize>(
    ctx: &GpuContext,
    coeffs: &[DeviceBuffer],
    degree: usize,
    terms: &[u64],
    points: &[F::Extension],
) -> Vec<F::Extension> {
    assert!(coeffs.len() <= MAX_ORACLES && points.len() <= MAX_ORACLES);
    let count = terms.len();
    let mut params = vec![degree as u32, count as u32, 0, 0];
    for p in 0..MAX_ORACLES {
        params.extend(
            points
                .get(p)
                .map_or([0; 4], |&point| uniform_words::<F, D>(point)),
        );
    }
    let params = ctx.uniform("open params", &params);
    let zpows = ctx.storage("point powers", (points.len() * D * degree * 8) as u64);
    let terms = ctx.upload("opening terms", terms);
    let evals = ctx.storage("openings", (D * count * 8) as u64);
    let mut buffers = vec![&params];
    buffers.extend(coeffs.iter().map(|b| &**b));
    buffers.resize(1 + MAX_ORACLES, &ctx.dummy);
    buffers.extend([&*terms, &*zpows, &*evals]);
    let bindings = ctx.bind_open(&buffers);
    let mut encoder = ctx.encoder("openings");
    {
        let mut pass = ctx.begin_pass(&mut encoder, "openings", None);
        pass.set_bind_group(0, &bindings, &[]);
        pass.set_pipeline(&ctx.point_powers);
        pass.dispatch_workgroups(
            (degree as u32).div_ceil(HASH_WORKGROUP),
            points.len() as u32,
            1,
        );
        pass.set_pipeline(&ctx.eval_openings);
        pass.dispatch_workgroups(count as u32, 1, 1);
    }
    ctx.submit(encoder);
    read_extension::<F, D>(ctx, &evals, count)
}

/// `OpeningSet::new` on the device, or `None` if the GPU does not take it.
pub(crate) fn try_opening_set<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    zeta: F::Extension,
    g: F::Extension,
    constants_sigmas_commitment: &PolynomialBatch<F, C, D>,
    wires_commitment: &PolynomialBatch<F, C, D>,
    zs_partial_products_lookup_commitment: &PolynomialBatch<F, C, D>,
    quotient_polys_commitment: &PolynomialBatch<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Option<OpeningSet<F, D>> {
    if D != 2 || !field_supported::<F>() {
        return None;
    }
    let batches = [
        constants_sigmas_commitment,
        wires_commitment,
        zs_partial_products_lookup_commitment,
        quotient_polys_commitment,
    ];
    let degree_log = wires_commitment.degree_log;
    let degree = 1usize << degree_log;
    let columns = batches
        .iter()
        .map(|b| b.polynomials.len())
        .max()
        .unwrap_or(0);
    if (degree_log + wires_commitment.rate_bits) < crate::gpu::MIN_LOG_LEAVES as usize
        || batches.iter().any(|b| !uniform_degree(b, degree_log))
    {
        return None;
    }
    let ctx = context()?;
    let ctx = &*ctx;
    if !ctx.fits((columns * degree * 8) as u64)
        || columns > ctx.max_workgroups as usize / MAX_ORACLES
    {
        return None;
    }
    let coeffs = batches
        .iter()
        .map(|b| device_coefficients(ctx, b, "opening"))
        .collect::<Vec<_>>();
    // Every polynomial at zeta; only Z and lookup polynomials need g * zeta.
    let num_zs = zs_partial_products_lookup_commitment.polynomials.len();
    let next_columns = common_data
        .zs_range()
        .chain(common_data.lookup_range().start..num_zs);
    let terms = batches
        .iter()
        .enumerate()
        .flat_map(|(o, b)| (0..b.polynomials.len()).map(move |c| term(o, c, 0)))
        .chain(next_columns.map(|c| term(2, c, 1)))
        .collect::<Vec<_>>();
    let evals = evaluate::<F, D>(ctx, &coeffs, degree, &terms, &[zeta, g * zeta]);
    let mut evals = evals.as_slice();
    let mut take = |len: usize| {
        let (head, tail) = evals.split_at(len);
        evals = tail;
        head
    };
    let constants_sigmas = take(constants_sigmas_commitment.polynomials.len());
    let wires = take(wires_commitment.polynomials.len());
    let zs = take(num_zs);
    let quotient_polys = take(quotient_polys_commitment.polynomials.len());
    let zs_next = take(common_data.zs_range().len());
    let lookup_zs_next = take(common_data.num_all_lookup_polys());
    Some(OpeningSet {
        constants: constants_sigmas[common_data.constants_range()].to_vec(),
        plonk_sigmas: constants_sigmas[common_data.sigmas_range()].to_vec(),
        wires: wires.to_vec(),
        plonk_zs: zs[common_data.zs_range()].to_vec(),
        plonk_zs_next: zs_next.to_vec(),
        partial_products: zs[common_data.partial_products_range()].to_vec(),
        quotient_polys: quotient_polys.to_vec(),
        lookup_zs: zs[common_data.lookup_range()].to_vec(),
        lookup_zs_next: lookup_zs_next.to_vec(),
    })
}

/// `PolynomialBatch::prove_openings` with the initial polynomial combined and
/// folded on the device, or `None` if the GPU does not take this instance.
/// The challenger is only touched once the GPU has been chosen.
pub(crate) fn try_prove_openings<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    instance: &FriInstanceInfo<F, D>,
    oracles: &[&PolynomialBatch<F, C, D>],
    challenger: &mut Challenger<F, C::Hasher>,
    fri_params: &FriParams,
    final_poly_coeff_len: Option<usize>,
    max_num_query_steps: Option<usize>,
    timing: &mut TimingTree,
) -> Option<FriProof<F, C::Hasher, D>> {
    if D != 2
        || !supported::<F, C::Hasher>()
        || oracles.is_empty()
        || oracles.len() > MAX_ORACLES
        || instance.batches.is_empty()
        || instance.batches.len() > MAX_ORACLES
    {
        return None;
    }
    let degree_log = oracles[0].degree_log;
    let rate_bits = fri_params.config.rate_bits;
    let degree = 1usize << degree_log;
    let log_n = (degree_log + rate_bits) as u32;
    let n = 1usize << log_n;
    let num_batches = instance.batches.len();
    let num_blocks = degree / COMBINE_BLOCK as usize;
    let columns = oracles
        .iter()
        .map(|o| o.polynomials.len())
        .max()
        .unwrap_or(0);
    if log_n < crate::gpu::MIN_LOG_LEAVES
        || degree < COMBINE_BLOCK as usize
        || oracles
            .iter()
            .any(|o| o.rate_bits != rate_bits || !uniform_degree(o, degree_log))
        || instance.batches.iter().any(|b| {
            b.polynomials.iter().any(|p| {
                p.oracle_index >= oracles.len()
                    || p.polynomial_index >= oracles[p.oracle_index].polynomials.len()
            })
        })
    {
        return None;
    }
    let ctx = context()?;
    let ctx = &ctx;
    if !commit_phase_fits(ctx, log_n, fri_params)
        || !ctx.fits((columns * degree * 8) as u64)
        || fft_plan(log_n)
            .iter()
            .any(|&(_, stages, low_bits)| (n >> (stages + low_bits)) > ctx.max_workgroups as usize)
        || columns > ctx.max_workgroups as usize / MAX_ORACLES
    {
        return None;
    }

    let alpha = challenger.get_extension_challenge::<D>();
    let (values, coeffs) = timed!(timing, "gpu: submit the initial polynomial", {
        // The terms batch by batch, each weighted by the power of alpha
        // `reduce_polys_base` gives the j-th polynomial of its batch; each
        // batch's quotient is scaled by alpha to the number of later terms,
        // as `shift_poly` does.
        let mut terms = Vec::new();
        let mut alphas = Vec::new();
        let mut params = vec![
            degree as u32,
            n as u32,
            num_batches as u32,
            num_blocks as u32,
        ];
        let mut points = Vec::new();
        let mut scales = Vec::new();
        for (b, batch) in instance.batches.iter().enumerate() {
            params.extend([terms.len() as u32, batch.polynomials.len() as u32, 0, 0]);
            terms.extend(
                batch
                    .polynomials
                    .iter()
                    .map(|p| term(p.oracle_index, p.polynomial_index, b)),
            );
            for power in alpha.powers().take(batch.polynomials.len()) {
                alphas.extend(words::<F, D>(power));
            }
            let later = instance.batches[b + 1..]
                .iter()
                .map(|batch| batch.polynomials.len() as u64)
                .sum::<u64>();
            points.extend(uniform_words::<F, D>(batch.point));
            scales.extend(uniform_words::<F, D>(alpha.exp_u64(later)));
        }
        params.resize(4 + 4 * MAX_ORACLES, 0);
        points.resize(4 * MAX_ORACLES, 0);
        scales.resize(4 * MAX_ORACLES, 0);
        params.extend(points);
        params.extend(scales);
        let params = ctx.uniform("combine params", &params);
        let coeff_buffers = oracles
            .iter()
            .map(|o| device_coefficients(ctx, o, "opening"))
            .collect::<Vec<_>>();
        let terms = ctx.upload("combination terms", &terms);
        let alphas = ctx.upload("alpha powers", &alphas);
        let scratch = ctx.storage(
            "combination scratch",
            (num_batches * (3 * D * degree + 2 * D * num_blocks) * 8) as u64,
        );
        let coeffs = ctx.storage("initial polynomial", (D * n * 8) as u64);
        let values = ctx.storage("initial codeword", (D * n * 8) as u64);
        let mut buffers = vec![&params];
        buffers.extend(coeff_buffers.iter().map(|b| &**b));
        buffers.resize(1 + MAX_ORACLES, &ctx.dummy);
        buffers.extend([&*terms, &*alphas, &*scratch, &*coeffs]);
        let bindings = ctx.bind_combine(&buffers);
        let mut encoder = ctx.encoder("initial polynomial");
        {
            let mut pass = ctx.begin_pass(&mut encoder, "initial polynomial", None);
            pass.set_bind_group(0, &bindings, &[]);
            let [powers, reduce, divide, scan, combine] = &ctx.combine;
            pass.set_pipeline(powers);
            pass.dispatch_workgroups(
                (degree as u32).div_ceil(HASH_WORKGROUP),
                num_batches as u32,
                1,
            );
            pass.set_pipeline(reduce);
            pass.dispatch_workgroups(
                (degree as u32).div_ceil(HASH_WORKGROUP),
                num_batches as u32,
                1,
            );
            pass.set_pipeline(divide);
            pass.dispatch_workgroups(
                (num_blocks as u32).div_ceil(HASH_WORKGROUP),
                num_batches as u32,
                1,
            );
            pass.set_pipeline(scan);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(combine);
            pass.dispatch_workgroups((n as u32).div_ceil(HASH_WORKGROUP), 1, 1);
        }
        fft(
            ctx,
            &mut encoder,
            FftJob {
                log_n,
                columns: D as u32,
                input: &coeffs,
                in_len: degree as u32,
                in_stride: n as u32,
                data: &values,
                output: None,
                shift_in: Some(crate::field::goldilocks_field::GoldilocksField::coset_shift()),
                inverse: false,
                shift_out: None,
            },
        );
        ctx.submit_async(encoder);
        (values, coeffs)
    });
    let (trees, final_coeffs) = timed!(
        timing,
        "fold codewords in the commitment phase",
        committed_trees::<F, C, D>(
            ctx,
            values,
            coeffs,
            n,
            challenger,
            fri_params,
            final_poly_coeff_len,
            max_num_query_steps,
        )
    );
    let pow_witness = timed!(
        timing,
        "find proof-of-work witness",
        fri_proof_of_work::<F, C, D>(challenger, &fri_params.config)
    );
    let initial_merkle_trees = oracles.iter().map(|o| &o.merkle_tree).collect::<Vec<_>>();
    let query_round_proofs = timed!(
        timing,
        "query rounds",
        fri_prover_query_rounds::<F, C, D>(
            &initial_merkle_trees,
            &trees,
            challenger,
            n,
            fri_params
        )
    );
    Some(FriProof {
        commit_phase_merkle_caps: trees.iter().map(|t| t.cap.clone()).collect(),
        query_round_proofs,
        final_poly: final_coeffs,
        pow_witness,
    })
}
