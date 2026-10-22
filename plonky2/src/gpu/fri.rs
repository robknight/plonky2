//! FRI's commit phase on the device: each layer's leaves (bit-reversed chunks
//! of `arity` extension elements) are hashed from the column-major values,
//! the coefficients are folded with the challenge, and the next layer's
//! values are a coset FFT of the folded coefficients. The first layer's
//! values and coefficients come from the host, or from the device when the
//! opening phase combined the codeword there (see [`super::opening`]).

use plonky2_maybe_rayon::*;

use crate::field::extension::{Extendable, FieldExtension};
use crate::field::goldilocks_field::GoldilocksField;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::field::types::Field;
use crate::fri::prover::FriCommitedTrees;
use crate::fri::FriParams;
use crate::gpu::lde::{fft, FftJob};
use crate::gpu::merkle::{build_tree, supported, LeafSource};
use crate::gpu::shader::HASH_WORKGROUP;
use crate::gpu::{context, DeviceBuffer, GpuContext};
use crate::hash::hash_types::{RichField, NUM_HASH_OUT_ELTS};
use crate::hash::merkle_tree::{MerkleCap, MerkleLeaves, MerkleTree};
use crate::iop::challenger::Challenger;
use crate::plonk::config::GenericConfig;
use crate::util::{log2_strict, reverse_bits};

/// Uploads extension elements as `D` columns of their base field components.
fn upload_extension<F: RichField + Extendable<D>, const D: usize>(
    ctx: &GpuContext,
    label: &str,
    elements: &[F::Extension],
) -> DeviceBuffer {
    let n = elements.len();
    let mut words = vec![0u64; D * n];
    for (i, element) in elements.iter().enumerate() {
        for (d, component) in element.to_basefield_array().iter().enumerate() {
            words[d * n + i] = component.to_noncanonical_u64();
        }
    }
    ctx.upload(label, &words)
}

/// Reads `D` columns of `n` base field components back as extension elements.
pub(crate) fn read_extension<F: RichField + Extendable<D>, const D: usize>(
    ctx: &GpuContext,
    buffer: &wgpu::Buffer,
    n: usize,
) -> Vec<F::Extension> {
    let words = ctx.read_u64(buffer, D * n);
    (0..n)
        .map(|i| {
            let mut components = [F::ZERO; D];
            for (d, component) in components.iter_mut().enumerate() {
                *component = F::from_noncanonical_u64(words[d * n + i]);
            }
            F::Extension::from_basefield_array(components)
        })
        .collect()
}

/// The leaves of the layer's tree: leaf `i` is the flattening of the
/// values at the bit-reversed positions `i * arity + j`.
fn layer_leaves<F: RichField + Extendable<D>, const D: usize>(
    values: &[F::Extension],
    arity_bits: usize,
) -> MerkleLeaves<F> {
    let n = values.len();
    let log_n = log2_strict(n);
    let arity = 1 << arity_bits;
    let num_leaves = n >> arity_bits;
    let leaf_len = D * arity;
    let mut flat = vec![F::ZERO; num_leaves * leaf_len];
    flat.par_chunks_exact_mut(leaf_len)
        .enumerate()
        .for_each(|(i, leaf)| {
            for j in 0..arity {
                let element = values[reverse_bits(i * arity + j, log_n)].to_basefield_array();
                leaf[j * D..(j + 1) * D].copy_from_slice(&element);
            }
        });
    MerkleLeaves::new(flat, leaf_len, num_leaves)
}

/// `fri_committed_trees` on the device, or `None` if the GPU does not take
/// this instance. The challenger is only touched once the GPU has been chosen.
pub(crate) fn try_committed_trees<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    coeffs: &PolynomialCoeffs<F::Extension>,
    values: &PolynomialValues<F::Extension>,
    challenger: &mut Challenger<F, C::Hasher>,
    fri_params: &FriParams,
    final_poly_coeff_len: Option<usize>,
    max_num_query_steps: Option<usize>,
) -> Option<FriCommitedTrees<F, C, D>> {
    if D != 2 || !supported::<F, C::Hasher>() {
        return None;
    }
    let ctx = context()?;
    let ctx = &*ctx;
    let n = values.len();
    if n != coeffs.len() || !n.is_power_of_two() {
        return None;
    }
    let log_n = log2_strict(n) as u32;
    if !commit_phase_fits(ctx, log_n, fri_params) {
        return None;
    }
    let vals = upload_extension::<F, D>(ctx, "fri values", &values.values);
    let coeffs_buffer = upload_extension::<F, D>(ctx, "fri coeffs", &coeffs.coeffs);
    Some(committed_trees::<F, C, D>(
        ctx,
        vals,
        coeffs_buffer,
        values.values.clone(),
        challenger,
        fri_params,
        final_poly_coeff_len,
        max_num_query_steps,
    ))
}

/// Whether the commit phase of a codeword of `2^log_n` points can run on the device.
pub(crate) fn commit_phase_fits(ctx: &GpuContext, log_n: u32, fri_params: &FriParams) -> bool {
    let n = 1u64 << log_n;
    log_n >= crate::gpu::MIN_LOG_LEAVES
        && n.div_ceil(HASH_WORKGROUP as u64) <= ctx.max_workgroups as u64
        && !fri_params.reduction_arity_bits.contains(&0)
        && fri_params.reduction_arity_bits.iter().sum::<usize>() <= log_n as usize
}

/// The commit phase from the device copies of the first layer: `vals` and
/// `coeffs_buffer` hold `D` columns of `host_values.len()` components, and
/// `host_values` is the same codeword on the host (for the layer's leaves).
pub(crate) fn committed_trees<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    ctx: &GpuContext,
    mut vals: DeviceBuffer,
    mut coeffs_buffer: DeviceBuffer,
    mut host_values: Vec<F::Extension>,
    challenger: &mut Challenger<F, C::Hasher>,
    fri_params: &FriParams,
    final_poly_coeff_len: Option<usize>,
    max_num_query_steps: Option<usize>,
) -> FriCommitedTrees<F, C, D> {
    let mut trees = Vec::with_capacity(fri_params.reduction_arity_bits.len());
    let mut n = host_values.len();
    let mut log_n = log2_strict(n) as u32;
    let mut shift = F::MULTIPLICATIVE_GROUP_GENERATOR;
    for &arity_bits in &fri_params.reduction_arity_bits {
        let arity = 1usize << arity_bits;
        let tree = build_tree(
            ctx,
            &vals,
            LeafSource::Columns {
                columns: D as u32,
                log_n,
                arity_bits: arity_bits as u32,
            },
            log_n - arity_bits as u32,
            fri_params.config.cap_height as u32,
            None,
        );
        let leaves = layer_leaves::<F, D>(&host_values, arity_bits);
        let (digests, cap) = tree.read::<F, C::Hasher>(ctx);
        let tree = MerkleTree {
            leaves,
            digests: digests.into(),
            cap: MerkleCap(cap),
        };
        challenger.observe_cap(&tree.cap);
        trees.push(tree);

        let beta = challenger.get_extension_challenge::<D>();
        // P(x) = sum_{i<r} x^i * P_i(x^r) becomes sum_{i<r} beta^i * P_i(x).
        let betas = beta
            .powers()
            .take(arity)
            .flat_map(|power| {
                power
                    .to_basefield_array()
                    .into_iter()
                    .map(|component| component.to_noncanonical_u64())
            })
            .collect::<Vec<_>>();
        let betas = ctx.upload("fri betas", &betas);
        let new_n = n >> arity_bits;
        let folded = ctx.storage("fri folded coeffs", (D * new_n * 8) as u64);
        let params = ctx.uniform("fold params", &[new_n as u32, arity as u32, n as u32, 0]);
        let bindings = ctx.bind_fold(&params, &coeffs_buffer, &folded, &betas);
        let mut encoder = ctx.encoder("fri fold");
        {
            let mut pass = ctx.begin_pass(&mut encoder, "fri fold", None);
            pass.set_pipeline(&ctx.fold);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups((new_n as u32).div_ceil(HASH_WORKGROUP), 1, 1);
        }
        shift = shift.exp_u64(arity as u64);
        n = new_n;
        log_n -= arity_bits as u32;
        let new_vals = ctx.storage("fri values", (D * n * 8) as u64);
        fft(
            ctx,
            &mut encoder,
            FftJob {
                log_n,
                columns: D as u32,
                input: &folded,
                in_len: n as u32,
                in_stride: n as u32,
                data: &new_vals,
                output: None,
                shift_in: Some(GoldilocksField::from_noncanonical_u64(
                    shift.to_noncanonical_u64(),
                )),
                inverse: false,
                shift_out: None,
            },
        );
        ctx.submit(encoder);
        coeffs_buffer = folded;
        vals = new_vals;
        host_values = read_extension::<F, D>(ctx, &vals, n);
    }

    // When verifying this proof in a circuit with a different number of query steps,
    // we need the challenger to stay in sync with the verifier. Therefore, the challenger
    // must observe the additional hash caps and generate dummy challenges.
    if let Some(step_count) = max_num_query_steps {
        let cap_len = (1 << fri_params.config.cap_height) * NUM_HASH_OUT_ELTS;
        let zero_cap = vec![F::ZERO; cap_len];
        for _ in fri_params.reduction_arity_bits.len()..step_count {
            challenger.observe_elements(&zero_cap);
            challenger.get_extension_challenge::<D>();
        }
    }

    // The coefficients being removed here should always be zero.
    let mut coeffs = PolynomialCoeffs::new(read_extension::<F, D>(ctx, &coeffs_buffer, n));
    coeffs
        .coeffs
        .truncate(coeffs.len() >> fri_params.config.rate_bits);

    challenger.observe_extension_elements(&coeffs.coeffs);
    // When verifying this proof in a circuit with a different final polynomial length,
    // the challenger needs to observe the full length of the final polynomial.
    if let Some(len) = final_poly_coeff_len {
        let current_len = coeffs.coeffs.len();
        for _ in current_len..len {
            challenger.observe_extension_element(&F::Extension::ZERO);
        }
    }

    (trees, coeffs)
}

/// Opening FFTs below `2^OPENING_FFT_MIN_LOG_N` points stay on the CPU, where
/// submission and transfer overhead would dominate.
pub(crate) const OPENING_FFT_MIN_LOG_N: u32 = 16;

/// Final opening FFT, including upload/readback, from `2^OPENING_FFT_MIN_LOG_N`
/// points.
pub(crate) fn try_opening_fft<F: RichField + Extendable<D>, const D: usize>(
    coeffs: &PolynomialCoeffs<F::Extension>,
) -> Option<PolynomialValues<F::Extension>> {
    try_opening_fft_from::<F, D>(coeffs, OPENING_FFT_MIN_LOG_N)
}

/// `try_opening_fft` with the size floor `2^min_log_n` (at least 2).
pub(crate) fn try_opening_fft_from<F: RichField + Extendable<D>, const D: usize>(
    coeffs: &PolynomialCoeffs<F::Extension>,
    min_log_n: u32,
) -> Option<PolynomialValues<F::Extension>> {
    if D != 2 || !crate::gpu::field_supported::<F>() {
        return None;
    }
    let ctx = context()?;
    let ctx = &*ctx;
    let n = coeffs.len();
    if !n.is_power_of_two()
        || n.ilog2() < min_log_n.max(1)
        || n > u32::MAX as usize / D
        || !ctx.fits((D * n * 8) as u64)
        || crate::gpu::shader::fft_plan(n.ilog2())
            .iter()
            .any(|&(_, stages, low_bits)| (n >> (stages + low_bits)) > ctx.max_workgroups as usize)
    {
        return None;
    }
    let input = upload_extension::<F, D>(ctx, "opening FFT input", &coeffs.coeffs);
    let data = ctx.storage("opening FFT values", (D * n * 8) as u64);
    let mut encoder = ctx.encoder("opening FFT");
    fft(
        ctx,
        &mut encoder,
        FftJob {
            log_n: log2_strict(n) as u32,
            columns: D as u32,
            input: &input,
            in_len: n as u32,
            in_stride: n as u32,
            data: &data,
            output: None,
            shift_in: Some(GoldilocksField::coset_shift()),
            inverse: false,
            shift_out: None,
        },
    );
    ctx.submit(encoder);
    Some(PolynomialValues::new(read_extension::<F, D>(ctx, &data, n)))
}
