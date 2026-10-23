#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};

use itertools::Itertools;
use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;
use serde::{Deserialize, Serialize};

use crate::field::extension::Extendable;
use crate::field::fft::FftRootTable;
use crate::field::packed::PackedField;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::proof::FriProof;
use crate::fri::prover::fri_proof;
use crate::fri::structure::{FriBatchInfo, FriInstanceInfo};
use crate::fri::FriParams;
use crate::hash::hash_types::RichField;
use crate::hash::merkle_tree::{MerkleLeaves, MerkleTree};
use crate::iop::challenger::Challenger;
use crate::plonk::config::GenericConfig;
use crate::timed;
use crate::util::reducing::ReducingFactor;
use crate::util::timing::TimingTree;
use crate::util::{log2_strict, reverse_bits};

mod prover;
pub(crate) use prover::{ProverOracle, ProverPolynomialBatch};

/// Four (~64 bit) field elements gives ~128 bit security.
pub const SALT_SIZE: usize = 4;

/// The rows of the column-major `columns`, as Merkle leaves in bit-reversed
/// row order: leaf `i` is row `reverse_bits(i)`.
pub(crate) fn transpose_to_leaves<F: Field>(columns: &[Vec<F>]) -> MerkleLeaves<F> {
    let num_columns = columns.len();
    let num_rows = columns[0].len();
    let log_rows = log2_strict(num_rows);
    let mut values = vec![F::ZERO; num_columns * num_rows];
    values
        .par_chunks_exact_mut(num_columns)
        .enumerate()
        .for_each(|(i, leaf)| {
            let row = reverse_bits(i, log_rows);
            for (value, column) in leaf.iter_mut().zip(columns) {
                *value = column[row];
            }
        });
    MerkleLeaves::new(values, num_columns, num_rows)
}

/// Represents a FRI oracle, i.e. a batch of polynomials which have been Merklized.
#[derive(Eq, PartialEq, Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct PolynomialBatch<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    pub polynomials: Vec<PolynomialCoeffs<F>>,
    pub merkle_tree: MerkleTree<F, C::Hasher>,
    pub degree_log: usize,
    pub rate_bits: usize,
    pub blinding: bool,
    /// The device copy of the extended evaluations, when committed on the GPU.
    #[cfg(feature = "gpu")]
    #[serde(skip)]
    pub(crate) device: crate::gpu::DeviceCache,
    /// The device copy of the coefficients (column-major), when committed on
    /// the GPU or uploaded by the opening phase.
    #[cfg(feature = "gpu")]
    #[serde(skip)]
    pub(crate) device_coeffs: crate::gpu::DeviceCache,
    /// The device copy of the values on the subgroup (column-major), when
    /// committed from values on the GPU.
    #[cfg(feature = "gpu")]
    #[serde(skip)]
    pub(crate) device_values: crate::gpu::DeviceCache,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> Default
    for PolynomialBatch<F, C, D>
{
    fn default() -> Self {
        PolynomialBatch {
            polynomials: Vec::new(),
            merkle_tree: MerkleTree::default(),
            degree_log: 0,
            rate_bits: 0,
            blinding: false,
            #[cfg(feature = "gpu")]
            device: Default::default(),
            #[cfg(feature = "gpu")]
            device_coeffs: Default::default(),
            #[cfg(feature = "gpu")]
            device_values: Default::default(),
        }
    }
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    PolynomialBatch<F, C, D>
{
    /// Creates a list polynomial commitment for the polynomials interpolating the values in `values`.
    ///
    /// With the `gpu` feature and a suitable adapter, the IFFT, LDE, transpose and Merkle tree are
    /// computed on the device, using the same commitment format as the CPU.
    /// With blinding enabled, each call samples fresh salt.
    pub fn from_values(
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        Self::from_values_with_plan(
            values,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
            #[cfg(feature = "gpu")]
            None,
            #[cfg(feature = "gpu")]
            false,
        )
        .into_host()
    }

    pub(crate) fn from_values_with_plan(
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
        #[cfg(feature = "gpu")] plan: Option<&crate::gpu::policy::ExecutionPlan>,
        #[cfg(feature = "gpu")] retain_values: bool,
    ) -> ProverPolynomialBatch<F, C, D> {
        #[cfg(feature = "gpu")]
        if let Some(ctx) = crate::gpu::lde::applicable::<F, C, D>(
            values.len(),
            values.first().map_or(0, |v| v.len()),
            rate_bits,
            blinding,
        ) {
            let batch = crate::gpu::lde::from_values_with_residency(
                &ctx,
                values,
                rate_bits,
                blinding,
                cap_height,
                timing,
                plan.map_or_else(
                    || ctx.options.retain_leaves(ctx.mappable),
                    |p| p.retain_leaves(),
                ),
                retain_values,
            );
            if plan.is_none() {
                batch.batch.device.retain_within_limit(&ctx);
                batch.batch.device_coeffs.retain_within_limit(&ctx);
            }
            return batch;
        }
        Self::from_values_cpu(
            values,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
        )
        .into()
    }

    /// `from_values` on the CPU.
    pub(crate) fn from_values_cpu(
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        let coeffs = timed!(
            timing,
            "IFFT",
            values.into_par_iter().map(|v| v.ifft()).collect::<Vec<_>>()
        );

        Self::from_coeffs_cpu(
            coeffs,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
        )
    }

    /// Creates a list polynomial commitment for the polynomials `polynomials`.
    ///
    /// With the `gpu` feature and a suitable adapter, the LDE, transpose and Merkle tree are
    /// computed on the device, using the same commitment format as the CPU.
    /// With blinding enabled, each call samples fresh salt.
    pub fn from_coeffs(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        Self::from_coeffs_with_plan(
            polynomials,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
            #[cfg(feature = "gpu")]
            None,
        )
        .into_host()
    }

    pub(crate) fn from_coeffs_with_plan(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
        #[cfg(feature = "gpu")] plan: Option<&crate::gpu::policy::ExecutionPlan>,
    ) -> ProverPolynomialBatch<F, C, D> {
        #[cfg(feature = "gpu")]
        if let Some(ctx) = crate::gpu::lde::applicable::<F, C, D>(
            polynomials.len(),
            polynomials.first().map_or(0, |p| p.len()),
            rate_bits,
            blinding,
        ) {
            let batch = crate::gpu::lde::from_coeffs_with_residency(
                &ctx,
                polynomials,
                rate_bits,
                blinding,
                cap_height,
                timing,
                plan.map_or_else(
                    || ctx.options.retain_leaves(ctx.mappable),
                    |p| p.retain_leaves(),
                ),
            );
            if plan.is_none() {
                batch.batch.device.retain_within_limit(&ctx);
                batch.batch.device_coeffs.retain_within_limit(&ctx);
            }
            return batch;
        }
        Self::from_coeffs_cpu(
            polynomials,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
        )
        .into()
    }

    /// `from_coeffs` on the CPU.
    pub(crate) fn from_coeffs_cpu(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        let degree = polynomials[0].len();
        let lde_values = timed!(
            timing,
            "FFT + blinding",
            Self::lde_values(&polynomials, rate_bits, blinding, fft_root_table)
        );

        let leaves = timed!(timing, "transpose LDEs", transpose_to_leaves(&lde_values));
        let merkle_tree = timed!(
            timing,
            "build Merkle tree",
            MerkleTree::from_leaves(leaves, cap_height)
        );

        Self {
            polynomials,
            merkle_tree,
            degree_log: log2_strict(degree),
            rate_bits,
            blinding,
            #[cfg(feature = "gpu")]
            device: Default::default(),
            #[cfg(feature = "gpu")]
            device_coeffs: Default::default(),
            #[cfg(feature = "gpu")]
            device_values: Default::default(),
        }
    }

    pub(crate) fn lde_values(
        polynomials: &[PolynomialCoeffs<F>],
        rate_bits: usize,
        blinding: bool,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Vec<Vec<F>> {
        let degree = polynomials[0].len();

        // If blinding, salt with two random elements to each leaf vector.
        let salt_size = if blinding { SALT_SIZE } else { 0 };

        polynomials
            .par_iter()
            .map(|p| {
                assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
                p.lde(rate_bits)
                    .coset_fft_with_options(F::coset_shift(), Some(rate_bits), fft_root_table)
                    .values
            })
            .chain(
                (0..salt_size)
                    .into_par_iter()
                    .map(|_| F::rand_vec(degree << rate_bits)),
            )
            .collect()
    }

    /// Fetches LDE values at the `index * step`th point.
    pub fn get_lde_values(&self, index: usize, step: usize) -> &[F] {
        let index = index * step;
        let index = reverse_bits(index, self.degree_log + self.rate_bits);
        let slice = self.merkle_tree.leaves.row(index);
        &slice[..slice.len() - if self.blinding { SALT_SIZE } else { 0 }]
    }

    /// Prepares the rows that `get_lde_values(index, step)` will read for
    /// these indices, from the device buffer where the host maps it directly.
    #[cfg(feature = "gpu")]
    pub(crate) fn prepare_lde_values(&self, indices: impl IntoIterator<Item = usize>, step: usize) {
        let bits = self.degree_log + self.rate_bits;
        let rows = indices
            .into_iter()
            .map(|index| reverse_bits(index * step, bits))
            .collect::<Vec<_>>();
        self.merkle_tree.leaves.prepare_rows(&rows);
    }

    /// Like `get_lde_values`, but fetches LDE values from a batch of `P::WIDTH` points, and returns
    /// packed values.
    pub fn get_lde_values_packed<P>(&self, index_start: usize, step: usize) -> Vec<P>
    where
        P: PackedField<Scalar = F>,
    {
        let row_wise = (0..P::WIDTH)
            .map(|i| self.get_lde_values(index_start + i, step))
            .collect_vec();

        // This is essentially a transpose, but we will not use the generic transpose method as we
        // want inner lists to be of type P, not Vecs which would involve allocation.
        let leaf_size = row_wise[0].len();
        (0..leaf_size)
            .map(|j| {
                let mut packed = P::ZEROS;
                packed
                    .as_slice_mut()
                    .iter_mut()
                    .zip(&row_wise)
                    .for_each(|(packed_i, row_i)| *packed_i = row_i[j]);
                packed
            })
            .collect_vec()
    }

    /// Produces a batch opening proof.
    ///
    /// With the `gpu` feature and a suitable adapter, the initial FRI
    /// polynomial is combined from the device copies of the oracles'
    /// coefficients and folded on the device. Both paths use the same proof
    /// format; the proof-of-work nonce, and hence query indices, may differ.
    pub fn prove_openings(
        instance: &FriInstanceInfo<F, D>,
        oracles: &[&Self],
        challenger: &mut Challenger<F, C::Hasher>,
        fri_params: &FriParams,
        final_poly_coeff_len: Option<usize>,
        max_num_query_steps: Option<usize>,
        timing: &mut TimingTree,
    ) -> FriProof<F, C::Hasher, D> {
        let oracles = oracles
            .iter()
            .map(|&o| o as &dyn ProverOracle<F, C, D>)
            .collect::<Vec<_>>();
        Self::prove_openings_with_batches(
            instance,
            &oracles,
            challenger,
            fri_params,
            final_poly_coeff_len,
            max_num_query_steps,
            timing,
        )
    }

    pub(crate) fn prove_openings_with_batches(
        instance: &FriInstanceInfo<F, D>,
        oracles: &[&dyn ProverOracle<F, C, D>],
        challenger: &mut Challenger<F, C::Hasher>,
        fri_params: &FriParams,
        final_poly_coeff_len: Option<usize>,
        max_num_query_steps: Option<usize>,
        timing: &mut TimingTree,
    ) -> FriProof<F, C::Hasher, D> {
        #[cfg(feature = "gpu")]
        if let Some(proof) = crate::gpu::opening::try_prove_openings::<F, C, D>(
            instance,
            oracles,
            challenger,
            fri_params,
            final_poly_coeff_len,
            max_num_query_steps,
            timing,
        ) {
            return proof;
        }
        Self::prove_openings_cpu_with_batches(
            instance,
            oracles,
            challenger,
            fri_params,
            final_poly_coeff_len,
            max_num_query_steps,
            timing,
        )
    }

    /// Combines the opening polynomial on the CPU. Its final FFT and the FRI
    /// commit phase may still run on the GPU.
    #[cfg(test)]
    pub(crate) fn prove_openings_cpu(
        instance: &FriInstanceInfo<F, D>,
        oracles: &[&Self],
        challenger: &mut Challenger<F, C::Hasher>,
        fri_params: &FriParams,
        final_poly_coeff_len: Option<usize>,
        max_num_query_steps: Option<usize>,
        timing: &mut TimingTree,
    ) -> FriProof<F, C::Hasher, D> {
        let oracles = oracles
            .iter()
            .map(|&o| o as &dyn ProverOracle<F, C, D>)
            .collect::<Vec<_>>();
        Self::prove_openings_cpu_with_batches(
            instance,
            &oracles,
            challenger,
            fri_params,
            final_poly_coeff_len,
            max_num_query_steps,
            timing,
        )
    }

    pub(crate) fn prove_openings_cpu_with_batches(
        instance: &FriInstanceInfo<F, D>,
        oracles: &[&dyn ProverOracle<F, C, D>],
        challenger: &mut Challenger<F, C::Hasher>,
        fri_params: &FriParams,
        final_poly_coeff_len: Option<usize>,
        max_num_query_steps: Option<usize>,
        timing: &mut TimingTree,
    ) -> FriProof<F, C::Hasher, D> {
        assert!(D > 1, "Not implemented for D=1.");
        let alpha = challenger.get_extension_challenge::<D>();
        let mut alpha = ReducingFactor::new(alpha);

        // Final low-degree polynomial that goes into FRI.
        let mut final_poly = PolynomialCoeffs::empty();

        // Each batch `i` consists of an opening point `z_i` and polynomials `{f_ij}_j` to be opened at that point.
        // For each batch, we compute the composition polynomial `F_i = sum alpha^j f_ij`,
        // where `alpha` is a random challenge in the extension field.
        // The final polynomial is then computed as `final_poly = sum_i alpha^(k_i) (F_i(X) - F_i(z_i))/(X-z_i)`
        // where the `k_i`s are chosen such that each power of `alpha` appears only once in the final sum.
        // There are usually two batches for the openings at `zeta` and `g * zeta`.
        // `PlonkOracle` in `plonky2/src/plonk/plonk_common.rs` identifies the PLONK oracles.
        for FriBatchInfo { point, polynomials } in &instance.batches {
            // Collect the coefficients of all the polynomials in `polynomials`.
            let polys_coeff = polynomials.iter().map(|fri_poly| {
                &oracles[fri_poly.oracle_index].polynomials()[fri_poly.polynomial_index]
            });
            let composition_poly = timed!(
                timing,
                &format!("reduce batch of {} polynomials", polynomials.len()),
                alpha.reduce_polys_base(polys_coeff)
            );
            let mut quotient = composition_poly.divide_by_linear(*point);
            quotient.coeffs.push(F::Extension::ZERO); // pad back to power of two
            alpha.shift_poly(&mut final_poly);
            final_poly += quotient;
        }

        let lde_final_poly = final_poly.lde(fri_params.config.rate_bits);
        let lde_final_values = timed!(
            timing,
            &format!("perform final FFT {}", lde_final_poly.len()),
            {
                #[cfg(feature = "gpu")]
                let gpu_values = crate::gpu::fri::try_opening_fft::<F, D>(&lde_final_poly);
                #[cfg(not(feature = "gpu"))]
                let gpu_values: Option<PolynomialValues<F::Extension>> = None;
                gpu_values.unwrap_or_else(|| lde_final_poly.coset_fft(F::coset_shift().into()))
            }
        );

        let fri_proof = fri_proof::<F, C, D>(
            &oracles
                .par_iter()
                .map(|c| &c.batch().merkle_tree)
                .collect::<Vec<_>>(),
            lde_final_poly,
            lde_final_values,
            challenger,
            fri_params,
            final_poly_coeff_len,
            max_num_query_steps,
            timing,
        );

        fri_proof
    }
}
