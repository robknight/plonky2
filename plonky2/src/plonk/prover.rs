//! plonky2 prover implementation.

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};
use core::cmp::min;
use core::mem::swap;

use anyhow::{ensure, Result};
use hashbrown::HashMap;
use plonky2_maybe_rayon::*;

use super::circuit_builder::{LookupChallenges, LookupWire};
use crate::field::extension::Extendable;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::field::types::Field;
use crate::field::zero_poly_coset::ZeroPolyOnCoset;
#[cfg(not(feature = "gpu"))]
use crate::fri::oracle::ProverPolynomialBatch;
use crate::fri::oracle::{PolynomialBatch, ProverOracle};
use crate::gates::lookup::LookupGate;
use crate::gates::lookup_table::LookupTableGate;
use crate::gates::selectors::LookupSelectors;
use crate::hash::hash_types::RichField;
use crate::iop::challenger::Challenger;
use crate::iop::generator::generate_partial_witness;
use crate::iop::target::Target;
use crate::iop::witness::{MatrixWitness, PartialWitness, PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::NUM_COINS_LOOKUP;
use crate::plonk::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
use crate::plonk::config::{GenericConfig, Hasher};
use crate::plonk::plonk_common::PlonkOracle;
use crate::plonk::proof::{OpeningSet, Proof, ProofWithPublicInputs};
use crate::plonk::vanishing_poly::{eval_vanishing_poly_base_batch, get_lut_poly};
use crate::plonk::vars::EvaluationVarsBaseBatch;
use crate::timed;
use crate::util::partial_products::{partial_products_and_z_gx, quotient_chunk_products};
use crate::util::timing::TimingTree;
use crate::util::{log2_ceil, transpose};

/// Set all the lookup gate wires (including multiplicities) and pad unused LU slots.
/// Warning: rows are in descending order: the first gate to appear is the last LU gate, and
/// the last gate to appear is the first LUT gate.
pub fn set_lookup_wires<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    pw: &mut PartitionWitness<F>,
) -> Result<()> {
    for (
        lut_index,
        &LookupWire {
            last_lu_gate: _,
            last_lut_gate,
            first_lut_gate,
        },
    ) in prover_data.lookup_rows.iter().enumerate()
    {
        let lut_len = common_data.luts[lut_index].len();
        let num_entries = LookupGate::num_slots(&common_data.config);
        let num_lut_entries = LookupTableGate::num_slots(&common_data.config);

        // Compute multiplicities.
        let mut multiplicities = vec![0; lut_len];

        let table_value_to_idx: HashMap<u16, usize> = common_data.luts[lut_index]
            .iter()
            .enumerate()
            .map(|(i, (inp_target, _))| (*inp_target, i))
            .collect();

        for (inp_target, _) in prover_data.lut_to_lookups[lut_index].iter() {
            let inp_value = pw.get_target(*inp_target);
            let idx = table_value_to_idx
                .get(&u16::try_from(inp_value.to_canonical_u64()).unwrap())
                .unwrap();

            multiplicities[*idx] += 1;
        }

        // Pad the last `LookupGate` with the first entry from the LUT.
        let remaining_slots = (num_entries
            - (prover_data.lut_to_lookups[lut_index].len() % num_entries))
            % num_entries;
        let (first_inp_value, first_out_value) = common_data.luts[lut_index][0];
        for slot in (num_entries - remaining_slots)..num_entries {
            let inp_target =
                Target::wire(last_lut_gate - 1, LookupGate::wire_ith_looking_inp(slot));
            let out_target =
                Target::wire(last_lut_gate - 1, LookupGate::wire_ith_looking_out(slot));
            pw.set_target(inp_target, F::from_canonical_u16(first_inp_value))?;
            pw.set_target(out_target, F::from_canonical_u16(first_out_value))?;

            multiplicities[0] += 1;
        }

        for lut_entry in 0..lut_len {
            let row = first_lut_gate - lut_entry / num_lut_entries;
            let col = lut_entry % num_lut_entries;

            let mul_target = Target::wire(row, LookupTableGate::wire_ith_multiplicity(col));

            pw.set_target(
                mul_target,
                F::from_canonical_usize(multiplicities[lut_entry]),
            )?;
        }
    }

    Ok(())
}

pub fn prove<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    inputs: PartialWitness<F>,
    timing: &mut TimingTree,
) -> Result<ProofWithPublicInputs<F, C, D>>
where
    C::Hasher: Hasher<F>,
    C::InnerHasher: Hasher<F>,
{
    // The compiled executor runs only with a witness program the caller
    // installed for these inputs (see `crate::plonk::witness_program`).
    #[cfg(all(feature = "std", feature = "rand_chacha"))]
    if prover_data
        .witness_program
        .has_all(prover_data.generators.len())
    {
        use crate::plonk::witness_program::region::cpu::{self, Wires};
        // Same label as the worklist, so timing trees compare across paths.
        match timed!(
            timing,
            &format!("run {} generators", prover_data.generators.len()),
            cpu::generate_wires(&inputs, prover_data, common_data, true)
        ) {
            Ok(None) => {}
            Ok(Some((public_inputs, wires, report))) => {
                report.log();
                #[cfg(feature = "gpu")]
                let (witness, resident) = match wires {
                    Wires::Host(witness) => (Some(witness), None),
                    Wires::Resident(resident) => (None, Some(resident)),
                };
                #[cfg(not(feature = "gpu"))]
                let Wires::Host(witness) = wires;
                #[cfg(not(feature = "gpu"))]
                let witness = Some(witness);
                return prove_from_wires(
                    prover_data,
                    common_data,
                    public_inputs,
                    witness,
                    timing,
                    #[cfg(feature = "gpu")]
                    resident,
                );
            }
            // An installed program that cannot run these inputs; the worklist
            // is correct for any inputs.
            Err(error) => {
                log::warn!("plonky2: witness program failed, using the worklist: {error:#}");
            }
        }
    }
    let partition_witness = timed!(
        timing,
        &format!("run {} generators", prover_data.generators.len()),
        generate_partial_witness(inputs, prover_data, common_data)?
    );
    prove_with_partition_witness(prover_data, common_data, partition_witness, timing)
}

pub fn prove_with_partition_witness<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    mut partition_witness: PartitionWitness<F>,
    timing: &mut TimingTree,
) -> Result<ProofWithPublicInputs<F, C, D>>
where
    C::Hasher: Hasher<F>,
    C::InnerHasher: Hasher<F>,
{
    timed!(
        timing,
        "set lookup wires",
        set_lookup_wires(prover_data, common_data, &mut partition_witness)
    )?;
    let public_inputs = timed!(
        timing,
        "collect public inputs",
        partition_witness.get_targets(&prover_data.public_inputs)
    );
    let witness = timed!(timing, "compute full witness", {
        let (witness, buffer) = partition_witness.full_witness_and_buffer();
        #[cfg(feature = "std")]
        prover_data.witness_buffers.recycle(buffer);
        #[cfg(not(feature = "std"))]
        drop(buffer);
        witness
    });
    prove_from_wires(
        prover_data,
        common_data,
        public_inputs,
        Some(witness),
        timing,
        #[cfg(feature = "gpu")]
        None,
    )
}

/// Proves from the wire values: a host matrix, or with the `gpu` feature a
/// resident device buffer in column-major order. The public inputs are
/// supplied by the caller.
pub(crate) fn prove_from_wires<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    public_inputs: Vec<F>,
    witness: Option<MatrixWitness<F>>,
    timing: &mut TimingTree,
    #[cfg(feature = "gpu")] resident: Option<crate::gpu::resident_witness::ResidentWitness>,
) -> Result<ProofWithPublicInputs<F, C, D>>
where
    C::Hasher: Hasher<F>,
    C::InnerHasher: Hasher<F>,
{
    #[cfg(feature = "gpu")]
    let gpu_plan = timed!(
        timing,
        "plan GPU execution",
        crate::gpu::policy::ExecutionPlan::new(common_data, prover_data)
    );
    // Compile the GPU proof-of-work search in the background (Poseidon only);
    // the guard joins the thread before this proof returns.
    #[cfg(feature = "gpu")]
    let _pow_warm = match &gpu_plan.prepared {
        Some((ctx, _))
            if <C::Hasher as crate::plonk::config::Hasher<F>>::GPU_POSEIDON
                && common_data.config.fri_config.proof_of_work_bits > 0 =>
        {
            Some(crate::gpu::pow::warm(ctx))
        }
        _ => None,
    };

    let has_lookup = !common_data.luts.is_empty();
    let config = &common_data.config;
    let num_challenges = config.num_challenges;
    let quotient_degree = common_data.quotient_degree();
    let degree = common_data.degree();

    let public_inputs_hash = timed!(
        timing,
        "hash public inputs",
        C::InnerHasher::hash_no_pad(&public_inputs)
    );
    #[cfg(feature = "gpu")]
    let resident_mode = resident.is_some();
    #[cfg(not(feature = "gpu"))]
    let resident_mode = false;
    #[cfg(not(feature = "gpu"))]
    debug_assert!(witness.is_some());
    #[cfg(feature = "gpu")]
    let mut witness = witness;
    let wires_values = if resident_mode {
        None
    } else {
        Some(timed!(
            timing,
            "compute wire polynomials",
            witness
                .as_ref()
                .unwrap()
                .wire_values
                .par_iter()
                .map(|column| PolynomialValues::new(column.clone()))
                .collect::<Vec<_>>()
        ))
    };
    // The device that holds resident wires, kept for the CPU permutation
    // fallback so it never depends on the global context mid-proof.
    #[cfg(feature = "gpu")]
    let resident_ctx = resident.as_ref().map(|resident| resident.ctx.clone());
    #[cfg(feature = "gpu")]
    let retain_values = crate::gpu::permutation::supported_circuit(prover_data, common_data);
    let mut wires_commitment = {
        #[cfg(feature = "gpu-profiling")]
        let _capture = crate::gpu::capture::Capture::wires();
        #[cfg(feature = "gpu")]
        let resident_commitment = resident.map(|resident| {
            timed!(
                timing,
                "compute resident wires commitment",
                crate::gpu::lde::from_device_values::<F, C, D>(
                    &resident.ctx,
                    resident.wire_values,
                    config.num_wires,
                    degree,
                    config.fri_config.rate_bits,
                    config.zero_knowledge && PlonkOracle::WIRES.blinding,
                    config.fri_config.cap_height,
                    timing,
                    gpu_plan.retain_leaves(),
                    retain_values
                )
            )
        });
        #[cfg(not(feature = "gpu"))]
        let resident_commitment: Option<ProverPolynomialBatch<F, C, D>> = None;
        match resident_commitment {
            Some(commitment) => commitment,
            None => timed!(
                timing,
                "compute wires commitment",
                PolynomialBatch::<F, C, D>::from_values_with_plan(
                    wires_values.unwrap(),
                    config.fri_config.rate_bits,
                    config.zero_knowledge && PlonkOracle::WIRES.blinding,
                    config.fri_config.cap_height,
                    timing,
                    prover_data.fft_root_table.as_ref(),
                    #[cfg(feature = "gpu")]
                    Some(&gpu_plan),
                    #[cfg(feature = "gpu")]
                    retain_values
                )
            ),
        }
    };

    let mut challenger = Challenger::<F, C::Hasher>::new();

    // Observe the FRI config
    common_data.fri_params.observe(&mut challenger);

    // Observe the instance.
    challenger.observe_hash::<C::Hasher>(prover_data.circuit_digest);
    challenger.observe_hash::<C::InnerHasher>(public_inputs_hash);

    challenger.observe_cap::<C::Hasher>(&wires_commitment.batch.merkle_tree.cap);

    // We need 4 values per challenge: 2 for the combos, 1 for (X-combo) in the accumulators and 1 to prove that the lookup table was computed correctly.
    // We can reuse betas and gammas for two of them.
    let num_lookup_challenges = NUM_COINS_LOOKUP * num_challenges;

    let betas = challenger.get_n_challenges(num_challenges);
    let gammas = challenger.get_n_challenges(num_challenges);

    let deltas = if has_lookup {
        let mut delts = Vec::with_capacity(2 * num_challenges);
        let num_additional_challenges = num_lookup_challenges - 2 * num_challenges;
        let additional = challenger.get_n_challenges(num_additional_challenges);
        delts.extend(&betas);
        delts.extend(&gammas);
        delts.extend(additional);
        delts
    } else {
        vec![]
    };

    assert!(
        common_data.quotient_degree_factor < common_data.config.num_routed_wires,
        "When the number of routed wires is smaller that the degree, we should change the logic to avoid computing partial products."
    );
    // With the `gpu` feature the partial products and Z's are computed from
    // the wires commitment's device values and committed from the device.
    #[cfg(feature = "gpu")]
    let gpu_commitment = timed!(
        timing,
        "compute and commit to partial products and Z's",
        crate::gpu::permutation::try_partial_products_commitment(
            &wires_commitment,
            &betas,
            &gammas,
            prover_data,
            common_data,
            timing,
            Some(&gpu_plan),
        )
    );
    #[cfg(not(feature = "gpu"))]
    let gpu_commitment: Option<ProverPolynomialBatch<F, C, D>> = None;
    let mut partial_products_zs_and_lookup_commitment = match gpu_commitment {
        Some(commitment) => commitment,
        None => {
            #[cfg(feature = "gpu")]
            if witness.is_none() {
                let ctx = resident_ctx
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("GPU witness context unavailable"))?;
                let values = wires_commitment.batch.device_values.take().ok_or_else(|| {
                    anyhow::anyhow!("GPU witness values unavailable for CPU permutation fallback")
                })?;
                witness = Some(MatrixWitness {
                    wire_values: crate::gpu::lde::read_columns::<F>(
                        &ctx,
                        &values,
                        config.num_wires,
                        degree,
                    ),
                });
                eprintln!("GPU_WITNESS_RESIDENT fallback_materialization=true");
            }
            let witness = witness
                .as_ref()
                .expect("CPU permutation requires host wires");
            let mut partial_products_and_zs = timed!(
                timing,
                "compute partial products",
                all_wires_permutation_partial_products(
                    &witness,
                    &betas,
                    &gammas,
                    prover_data,
                    common_data
                )
            );

            // Z is expected at the front of our batch; see `zs_range` and `partial_products_range`.
            let plonk_z_vecs = partial_products_and_zs
                .iter_mut()
                .map(|partial_products_and_z| partial_products_and_z.pop().unwrap())
                .collect();
            let zs_partial_products = [plonk_z_vecs, partial_products_and_zs.concat()].concat();

            // All lookup polys: RE and partial SLDCs.
            let lookup_polys =
                compute_all_lookup_polys(&witness, &deltas, prover_data, common_data, has_lookup);

            let zs_partial_products_lookups = if has_lookup {
                [zs_partial_products, lookup_polys].concat()
            } else {
                zs_partial_products
            };

            timed!(
                timing,
                "commit to partial products, Z's and, if any, lookup polynomials",
                PolynomialBatch::from_values_with_plan(
                    zs_partial_products_lookups,
                    config.fri_config.rate_bits,
                    config.zero_knowledge && PlonkOracle::ZS_PARTIAL_PRODUCTS.blinding,
                    config.fri_config.cap_height,
                    timing,
                    prover_data.fft_root_table.as_ref(),
                    #[cfg(feature = "gpu")]
                    Some(&gpu_plan),
                    #[cfg(feature = "gpu")]
                    false,
                )
            )
        }
    };

    challenger.observe_cap::<C::Hasher>(
        &partial_products_zs_and_lookup_commitment
            .batch
            .merkle_tree
            .cap,
    );

    let alphas = challenger.get_n_challenges(num_challenges);

    let quotient_polys = timed!(
        timing,
        "compute quotient polys",
        compute_quotient_polys::<F, C, D>(
            common_data,
            prover_data,
            &public_inputs_hash,
            &wires_commitment,
            &partial_products_zs_and_lookup_commitment,
            &betas,
            &gammas,
            &deltas,
            &alphas,
            #[cfg(feature = "gpu")]
            Some(&gpu_plan),
        )
    );

    let mut quotient_polys_commitment = match quotient_polys {
        #[cfg(feature = "gpu")]
        QuotientCoefficients::Device(quotient) => timed!(
            timing,
            "commit to quotient polys",
            quotient.commit::<F, C, D>(common_data, timing, gpu_plan.retain_leaves())
        ),
        QuotientCoefficients::Host(polynomials) => {
            let all_quotient_poly_chunks: Vec<PolynomialCoeffs<F>> = timed!(
                timing,
                "split up quotient polys",
                polynomials
                    .into_par_iter()
                    .flat_map(|mut quotient_poly| {
                        quotient_poly.trim_to_len(quotient_degree).expect(
                            "Quotient has failed, the vanishing polynomial is not divisible by Z_H",
                        );
                        // Split quotient into degree-n chunks.
                        quotient_poly.chunks(degree)
                    })
                    .collect()
            );

            timed!(
                timing,
                "commit to quotient polys",
                PolynomialBatch::<F, C, D>::from_coeffs_with_plan(
                    all_quotient_poly_chunks,
                    config.fri_config.rate_bits,
                    config.zero_knowledge && PlonkOracle::QUOTIENT.blinding,
                    config.fri_config.cap_height,
                    timing,
                    prover_data.fft_root_table.as_ref(),
                    #[cfg(feature = "gpu")]
                    Some(&gpu_plan),
                )
            )
        }
    };

    challenger.observe_cap::<C::Hasher>(&quotient_polys_commitment.batch.merkle_tree.cap);

    let zeta = challenger.get_extension_challenge::<D>();
    // To avoid leaking witness data, we want to ensure that our opening locations, `zeta` and
    // `g * zeta`, are not in our subgroup `H`. It suffices to check `zeta` only, since
    // `(g * zeta)^n = zeta^n`, where `n` is the order of `g`.
    let g = F::Extension::primitive_root_of_unity(common_data.degree_bits());
    ensure!(
        zeta.exp_power_of_2(common_data.degree_bits()) != F::Extension::ONE,
        "Opening point is in the subgroup."
    );

    let openings = timed!(
        timing,
        "construct the opening set, including lookups",
        OpeningSet::new_with_batches(
            zeta,
            g,
            &prover_data.constants_sigmas_commitment,
            &wires_commitment,
            &partial_products_zs_and_lookup_commitment,
            &quotient_polys_commitment,
            common_data
        )
    );
    challenger.observe_openings(&openings.to_fri_openings());
    let instance = common_data.get_fri_instance(zeta);

    let opening_proof = timed!(
        timing,
        "compute opening proofs",
        PolynomialBatch::<F, C, D>::prove_openings_with_batches(
            &instance,
            &[
                &prover_data.constants_sigmas_commitment,
                &wires_commitment,
                &partial_products_zs_and_lookup_commitment,
                &quotient_polys_commitment,
            ],
            &mut challenger,
            &common_data.fri_params,
            None,
            None,
            timing,
        )
    );

    let proof = Proof::<F, C, D> {
        wires_cap: core::mem::take(&mut wires_commitment.batch.merkle_tree.cap),
        plonk_zs_partial_products_cap: core::mem::take(
            &mut partial_products_zs_and_lookup_commitment
                .batch
                .merkle_tree
                .cap,
        ),
        quotient_polys_cap: core::mem::take(&mut quotient_polys_commitment.batch.merkle_tree.cap),
        openings,
        opening_proof,
    };
    // Time the release of large buffers, including
    // host deallocation and the return/destruction of GPU buffers.
    timed!(timing, "release prover buffers", {
        timed!(
            timing,
            "release quotient commitment",
            drop(quotient_polys_commitment)
        );
        timed!(
            timing,
            "release permutation commitment",
            drop(partial_products_zs_and_lookup_commitment)
        );
        drop(challenger);
        timed!(timing, "release wires commitment", drop(wires_commitment));
        timed!(timing, "release witness", drop(witness));
    });
    // Diagnostic: log this proof's device pass timeline, after its timing.
    #[cfg(feature = "gpu-profiling")]
    if let Some(ctx) = crate::gpu::context() {
        ctx.flush_pass_timeline();
    }
    Ok(ProofWithPublicInputs::<F, C, D> {
        proof,
        public_inputs,
    })
}

/// Compute the partial products used in the `Z` polynomials.
fn all_wires_permutation_partial_products<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>,
    betas: &[F],
    gammas: &[F],
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Vec<Vec<PolynomialValues<F>>> {
    (0..common_data.config.num_challenges)
        .map(|i| {
            wires_permutation_partial_products_and_zs(
                witness,
                betas[i],
                gammas[i],
                prover_data,
                common_data,
            )
        })
        .collect()
}

/// Compute the partial products used in the `Z` polynomial.
/// Returns the polynomials interpolating `partial_products(f / g)`
/// where `f, g` are the products in the definition of `Z`: `Z(g^i) = f / g`.
fn wires_permutation_partial_products_and_zs<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>,
    beta: F,
    gamma: F,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Vec<PolynomialValues<F>> {
    let degree = common_data.quotient_degree_factor;
    let subgroup = &prover_data.subgroup;
    let k_is = &common_data.k_is;
    let num_prods = common_data.num_partial_products;
    let all_quotient_chunk_products = subgroup
        .par_iter()
        .enumerate()
        .map(|(i, &x)| {
            let s_sigmas = &prover_data.sigmas[i];
            let numerators = (0..common_data.config.num_routed_wires).map(|j| {
                let wire_value = witness.get_wire(i, j);
                let k_i = k_is[j];
                let s_id = k_i * x;
                wire_value + beta * s_id + gamma
            });
            let denominators = (0..common_data.config.num_routed_wires)
                .map(|j| {
                    let wire_value = witness.get_wire(i, j);
                    let s_sigma = s_sigmas[j];
                    wire_value + beta * s_sigma + gamma
                })
                .collect::<Vec<_>>();
            let denominator_invs = F::batch_multiplicative_inverse(&denominators);
            let quotient_values = numerators
                .zip(denominator_invs)
                .map(|(num, den_inv)| num * den_inv)
                .collect::<Vec<_>>();

            quotient_chunk_products(&quotient_values, degree)
        })
        .collect::<Vec<_>>();

    let mut z_x = F::ONE;
    let mut all_partial_products_and_zs = Vec::with_capacity(all_quotient_chunk_products.len());
    for quotient_chunk_products in all_quotient_chunk_products {
        let mut partial_products_and_z_gx =
            partial_products_and_z_gx(z_x, &quotient_chunk_products);
        // The last term is Z(gx), but we replace it with Z(x), otherwise Z would end up shifted.
        swap(&mut z_x, &mut partial_products_and_z_gx[num_prods]);
        all_partial_products_and_zs.push(partial_products_and_z_gx);
    }

    transpose(&all_partial_products_and_zs)
        .into_par_iter()
        .map(PolynomialValues::new)
        .collect()
}

/// Computes lookup polynomials for a given challenge.
/// The polynomials hold the value of RE, Sum and Ldc of the Tip5 paper (<https://eprint.iacr.org/2023/107.pdf>). To reduce their
/// numbers, we batch multiple slots in a single polynomial. Since RE only involves degree one constraints, we can batch
/// all the slots of a row. For Sum and Ldc, batching increases the constraint degree, so we bound the number of
/// partial polynomials according to `max_quotient_degree_factor`.
/// As another optimization, Sum and LDC polynomials are shared (in so called partial SLDC polynomials), and the last value
/// of the last partial polynomial is Sum(end) - LDC(end). If the lookup argument is valid, then it must be equal to 0.
fn compute_lookup_polys<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>,
    deltas: &[F; 4],
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Vec<PolynomialValues<F>> {
    let degree = common_data.degree();
    let num_lu_slots = LookupGate::num_slots(&common_data.config);
    let max_lookup_degree = common_data.config.max_quotient_degree_factor - 1;
    let num_partial_lookups = num_lu_slots.div_ceil(max_lookup_degree);
    let num_lut_slots = LookupTableGate::num_slots(&common_data.config);
    let max_lookup_table_degree = num_lut_slots.div_ceil(num_partial_lookups);

    // First poly is RE, the rest are partial SLDCs.
    let mut final_poly_vecs = Vec::with_capacity(num_partial_lookups + 1);
    for _ in 0..num_partial_lookups + 1 {
        final_poly_vecs.push(PolynomialValues::<F>::new(vec![F::ZERO; degree]));
    }

    for LookupWire {
        last_lu_gate: last_lu_row,
        last_lut_gate: last_lut_row,
        first_lut_gate: first_lut_row,
    } in prover_data.lookup_rows.clone()
    {
        // Set values for partial Sums and RE.
        for row in (last_lut_row..(first_lut_row + 1)).rev() {
            // Get combos for Sum.
            let looked_combos: Vec<F> = (0..num_lut_slots)
                .map(|s| {
                    let looked_inp = witness.get_wire(row, LookupTableGate::wire_ith_looked_inp(s));
                    let looked_out = witness.get_wire(row, LookupTableGate::wire_ith_looked_out(s));

                    looked_inp + deltas[LookupChallenges::ChallengeA as usize] * looked_out
                })
                .collect();
            // Get (alpha - combo).
            let minus_looked_combos: Vec<F> = (0..num_lut_slots)
                .map(|s| deltas[LookupChallenges::ChallengeAlpha as usize] - looked_combos[s])
                .collect();
            // Get 1/(alpha - combo).
            let looked_combo_inverses = F::batch_multiplicative_inverse(&minus_looked_combos);

            // Get lookup combos, used to check the well formation of the LUT.
            let lookup_combos: Vec<F> = (0..num_lut_slots)
                .map(|s| {
                    let looked_inp = witness.get_wire(row, LookupTableGate::wire_ith_looked_inp(s));
                    let looked_out = witness.get_wire(row, LookupTableGate::wire_ith_looked_out(s));

                    looked_inp + deltas[LookupChallenges::ChallengeB as usize] * looked_out
                })
                .collect();

            // Compute next row's first value of RE.
            // If `row == first_lut_row`, then `final_poly_vecs[0].values[row + 1] == 0`.
            let mut new_re = final_poly_vecs[0].values[row + 1];
            for elt in &lookup_combos {
                new_re = new_re * deltas[LookupChallenges::ChallengeDelta as usize] + *elt
            }
            final_poly_vecs[0].values[row] = new_re;

            for slot in 0..num_partial_lookups {
                let prev = if slot != 0 {
                    final_poly_vecs[slot].values[row]
                } else {
                    // If `row == first_lut_row`, then `final_poly_vecs[num_partial_lookups].values[row + 1] == 0`.
                    final_poly_vecs[num_partial_lookups].values[row + 1]
                };
                let sum = (slot * max_lookup_table_degree
                    ..min((slot + 1) * max_lookup_table_degree, num_lut_slots))
                    .fold(prev, |acc, s| {
                        acc + witness.get_wire(row, LookupTableGate::wire_ith_multiplicity(s))
                            * looked_combo_inverses[s]
                    });
                final_poly_vecs[slot + 1].values[row] = sum;
            }
        }

        // Set values for partial LDCs.
        for row in (last_lu_row..last_lut_row).rev() {
            // Get looking combos.
            let looking_combos: Vec<F> = (0..num_lu_slots)
                .map(|s| {
                    let looking_in = witness.get_wire(row, LookupGate::wire_ith_looking_inp(s));
                    let looking_out = witness.get_wire(row, LookupGate::wire_ith_looking_out(s));

                    looking_in + deltas[LookupChallenges::ChallengeA as usize] * looking_out
                })
                .collect();
            // Get (alpha - combo).
            let minus_looking_combos: Vec<F> = (0..num_lu_slots)
                .map(|s| deltas[LookupChallenges::ChallengeAlpha as usize] - looking_combos[s])
                .collect();
            // Get 1 / (alpha - combo).
            let looking_combo_inverses = F::batch_multiplicative_inverse(&minus_looking_combos);

            for slot in 0..num_partial_lookups {
                let prev = if slot == 0 {
                    // Valid at _any_ row, even `first_lu_row`.
                    final_poly_vecs[num_partial_lookups].values[row + 1]
                } else {
                    final_poly_vecs[slot].values[row]
                };
                let sum = (slot * max_lookup_degree
                    ..min((slot + 1) * max_lookup_degree, num_lu_slots))
                    .fold(F::ZERO, |acc, s| acc + looking_combo_inverses[s]);
                final_poly_vecs[slot + 1].values[row] = prev - sum;
            }
        }
    }

    final_poly_vecs
}

/// Computes lookup polynomials for all challenges.
fn compute_all_lookup_polys<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>,
    deltas: &[F],
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    lookup: bool,
) -> Vec<PolynomialValues<F>> {
    if lookup {
        let polys: Vec<Vec<PolynomialValues<F>>> = (0..common_data.config.num_challenges)
            .map(|c| {
                compute_lookup_polys(
                    witness,
                    &deltas[c * NUM_COINS_LOOKUP..(c + 1) * NUM_COINS_LOOKUP]
                        .try_into()
                        .unwrap(),
                    prover_data,
                    common_data,
                )
            })
            .collect();
        polys.concat()
    } else {
        vec![]
    }
}

const BATCH_SIZE: usize = 32;

enum QuotientCoefficients<F: RichField> {
    Host(Vec<PolynomialCoeffs<F>>),
    #[cfg(feature = "gpu")]
    Device(crate::gpu::quotient::DeviceQuotient),
}

#[cfg(all(test, feature = "gpu"))]
impl<F: RichField> QuotientCoefficients<F> {
    fn into_host(self) -> Vec<PolynomialCoeffs<F>> {
        match self {
            Self::Host(polynomials) => polynomials,
            Self::Device(device) => device.read(),
        }
    }
}

/// The quotient polynomials, one per challenge, as coefficients. With the `gpu` feature and a
/// suitable adapter, supported circuits without lookups use the GPU kernel,
/// optionally sharing rows with the CPU. The result is the same as
/// [`compute_quotient_polys_cpu`].
fn compute_quotient_polys<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    common_data: &CommonCircuitData<F, D>,
    prover_data: &'a ProverOnlyCircuitData<F, C, D>,
    public_inputs_hash: &<<C as GenericConfig<D>>::InnerHasher as Hasher<F>>::Hash,
    wires_commitment: &'a dyn ProverOracle<F, C, D>,
    zs_partial_products_and_lookup_commitment: &'a dyn ProverOracle<F, C, D>,
    betas: &[F],
    gammas: &[F],
    deltas: &[F],
    alphas: &[F],
    #[cfg(feature = "gpu")] plan: Option<&crate::gpu::policy::ExecutionPlan>,
) -> QuotientCoefficients<F> {
    #[cfg(feature = "gpu")]
    if let Some(result) = crate::gpu::quotient::try_compute_quotient::<F, C, D>(
        common_data,
        prover_data,
        public_inputs_hash,
        wires_commitment,
        zs_partial_products_and_lookup_commitment,
        betas,
        gammas,
        alphas,
        plan,
    ) {
        // Preserve commitment fallback for unsupported hashers or a quotient
        // batch whose extended chunks exceed the adapter's limits.
        if crate::gpu::lde::applicable::<F, C, D>(
            common_data.config.num_challenges * common_data.quotient_degree_factor,
            common_data.degree(),
            common_data.config.fri_config.rate_bits,
            common_data.config.zero_knowledge && PlonkOracle::QUOTIENT.blinding,
        )
        .is_some()
        {
            return QuotientCoefficients::Device(result);
        }
        return QuotientCoefficients::Host(result.read());
    }
    QuotientCoefficients::Host(compute_quotient_polys_cpu::<F, C, D>(
        common_data,
        prover_data,
        public_inputs_hash,
        wires_commitment.batch(),
        zs_partial_products_and_lookup_commitment.batch(),
        betas,
        gammas,
        deltas,
        alphas,
    ))
}

/// `compute_quotient_polys` on the CPU.
pub(crate) fn compute_quotient_polys_cpu<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    common_data: &CommonCircuitData<F, D>,
    prover_data: &'a ProverOnlyCircuitData<F, C, D>,
    public_inputs_hash: &<<C as GenericConfig<D>>::InnerHasher as Hasher<F>>::Hash,
    wires_commitment: &'a PolynomialBatch<F, C, D>,
    zs_partial_products_and_lookup_commitment: &'a PolynomialBatch<F, C, D>,
    betas: &[F],
    gammas: &[F],
    deltas: &[F],
    alphas: &[F],
) -> Vec<PolynomialCoeffs<F>> {
    let quotient_degree_bits = log2_ceil(common_data.quotient_degree_factor);
    let lde_size = common_data.degree() << quotient_degree_bits;
    let quotient_values = quotient_values_cpu::<F, C, D>(
        common_data,
        prover_data,
        public_inputs_hash,
        wires_commitment,
        zs_partial_products_and_lookup_commitment,
        betas,
        gammas,
        deltas,
        alphas,
        0..lde_size,
    );
    transpose(&quotient_values)
        .into_par_iter()
        .map(PolynomialValues::new)
        .map(|values| values.coset_ifft(F::coset_shift()))
        .collect()
}

/// The quotient's values at the points `rows` of the extended domain, one
/// vector of `num_challenges` values per point.
pub(crate) fn quotient_values_cpu<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    common_data: &CommonCircuitData<F, D>,
    prover_data: &'a ProverOnlyCircuitData<F, C, D>,
    public_inputs_hash: &<<C as GenericConfig<D>>::InnerHasher as Hasher<F>>::Hash,
    wires_commitment: &'a PolynomialBatch<F, C, D>,
    zs_partial_products_and_lookup_commitment: &'a PolynomialBatch<F, C, D>,
    betas: &[F],
    gammas: &[F],
    deltas: &[F],
    alphas: &[F],
    rows: core::ops::Range<usize>,
) -> Vec<Vec<F>> {
    let num_challenges = common_data.config.num_challenges;

    let has_lookup = common_data.num_lookup_polys != 0;

    let quotient_degree_bits = log2_ceil(common_data.quotient_degree_factor);
    assert!(
        quotient_degree_bits <= common_data.config.fri_config.rate_bits,
        "Having constraints of degree higher than the rate is not supported yet. \
        If we need this in the future, we can precompute the larger LDE before computing the `PolynomialBatch`s."
    );

    // We reuse the LDE computed in `PolynomialBatch` and extract every `step` points to get
    // an LDE matching `max_filtered_constraint_degree`.
    let step = 1 << (common_data.config.fri_config.rate_bits - quotient_degree_bits);
    // When opening the `Z`s polys at the "next" point in Plonk, need to look at the point `next_step`
    // steps away since we work on an LDE of degree `max_filtered_constraint_degree`.
    let next_step = 1 << quotient_degree_bits;

    let points = F::two_adic_subgroup(common_data.degree_bits() + quotient_degree_bits);
    let lde_size = points.len();

    let z_h_on_coset = ZeroPolyOnCoset::new(common_data.degree_bits(), quotient_degree_bits);

    // Precompute the lookup table evals on the challenges in delta
    // These values are used to produce the final RE constraints for each lut,
    // and are the same each time in check_lookup_constraints_batched.
    // lut_poly_evals[i][j] gives the eval for the i'th challenge and the j'th lookup table
    let lut_re_poly_evals: Vec<Vec<F>> = if has_lookup {
        let num_lut_slots = LookupTableGate::num_slots(&common_data.config);
        (0..num_challenges)
            .map(move |i| {
                let cur_deltas = &deltas[NUM_COINS_LOOKUP * i..NUM_COINS_LOOKUP * (i + 1)];
                let cur_challenge_delta = cur_deltas[LookupChallenges::ChallengeDelta as usize];

                (LookupSelectors::StartEnd as usize..common_data.num_lookup_selectors)
                    .map(|r| {
                        let lut_row_number = common_data.luts
                            [r - LookupSelectors::StartEnd as usize]
                            .len()
                            .div_ceil(num_lut_slots);

                        get_lut_poly(
                            common_data,
                            r - LookupSelectors::StartEnd as usize,
                            cur_deltas,
                            num_lut_slots * lut_row_number,
                        )
                        .eval(cur_challenge_delta)
                    })
                    .collect()
            })
            .collect()
    } else {
        vec![]
    };

    let lut_re_poly_evals_refs: Vec<&[F]> =
        lut_re_poly_evals.iter().map(|v| v.as_slice()).collect();

    let points_batches = points[rows.clone()].par_chunks(BATCH_SIZE);
    let num_batches = rows.len().div_ceil(BATCH_SIZE);

    let quotient_values: Vec<Vec<F>> = points_batches
        .enumerate()
        .flat_map(|(batch_i, xs_batch)| {
            // Each batch must be the same size, except the last one, which may be smaller.
            debug_assert!(
                xs_batch.len() == BATCH_SIZE
                    || (batch_i == num_batches - 1 && xs_batch.len() <= BATCH_SIZE)
            );

            let first = rows.start + BATCH_SIZE * batch_i;
            let indices_batch: Vec<usize> = (first..first + xs_batch.len()).collect();

            let mut shifted_xs_batch = Vec::with_capacity(xs_batch.len());
            let mut local_zs_batch = Vec::with_capacity(xs_batch.len());
            let mut next_zs_batch = Vec::with_capacity(xs_batch.len());

            let mut local_lookup_batch = Vec::with_capacity(xs_batch.len());
            let mut next_lookup_batch = Vec::with_capacity(xs_batch.len());

            let mut partial_products_batch = Vec::with_capacity(xs_batch.len());
            let mut s_sigmas_batch = Vec::with_capacity(xs_batch.len());

            let mut local_constants_batch_refs = Vec::with_capacity(xs_batch.len());
            let mut local_wires_batch_refs = Vec::with_capacity(xs_batch.len());

            for (&i, &x) in indices_batch.iter().zip(xs_batch) {
                let shifted_x = F::coset_shift() * x;
                let i_next = (i + next_step) % lde_size;
                let local_constants_sigmas = prover_data
                    .constants_sigmas_commitment
                    .get_lde_values(i, step);
                let local_constants = &local_constants_sigmas[common_data.constants_range()];
                let s_sigmas = &local_constants_sigmas[common_data.sigmas_range()];
                let local_wires = wires_commitment.get_lde_values(i, step);
                let local_zs_partial_and_lookup =
                    zs_partial_products_and_lookup_commitment.get_lde_values(i, step);
                let next_zs_partial_and_lookup =
                    zs_partial_products_and_lookup_commitment.get_lde_values(i_next, step);

                let local_zs = &local_zs_partial_and_lookup[common_data.zs_range()];

                let next_zs = &next_zs_partial_and_lookup[common_data.zs_range()];

                let partial_products =
                    &local_zs_partial_and_lookup[common_data.partial_products_range()];

                if has_lookup {
                    let local_lookup_zs = &local_zs_partial_and_lookup[common_data.lookup_range()];

                    let next_lookup_zs = &next_zs_partial_and_lookup[common_data.lookup_range()];
                    debug_assert_eq!(local_lookup_zs.len(), common_data.num_all_lookup_polys());

                    local_lookup_batch.push(local_lookup_zs);
                    next_lookup_batch.push(next_lookup_zs);
                }

                debug_assert_eq!(local_wires.len(), common_data.config.num_wires);
                debug_assert_eq!(local_zs.len(), num_challenges);

                local_constants_batch_refs.push(local_constants);
                local_wires_batch_refs.push(local_wires);

                shifted_xs_batch.push(shifted_x);
                local_zs_batch.push(local_zs);
                next_zs_batch.push(next_zs);
                partial_products_batch.push(partial_products);
                s_sigmas_batch.push(s_sigmas);
            }

            // NB (JN): I'm not sure how (in)efficient the below is. It needs measuring.
            let mut local_constants_batch =
                vec![F::ZERO; xs_batch.len() * local_constants_batch_refs[0].len()];
            for i in 0..local_constants_batch_refs[0].len() {
                for (j, constants) in local_constants_batch_refs.iter().enumerate() {
                    local_constants_batch[i * xs_batch.len() + j] = constants[i];
                }
            }

            let mut local_wires_batch =
                vec![F::ZERO; xs_batch.len() * local_wires_batch_refs[0].len()];
            for i in 0..local_wires_batch_refs[0].len() {
                for (j, wires) in local_wires_batch_refs.iter().enumerate() {
                    local_wires_batch[i * xs_batch.len() + j] = wires[i];
                }
            }

            let vars_batch = EvaluationVarsBaseBatch::new(
                xs_batch.len(),
                &local_constants_batch,
                &local_wires_batch,
                public_inputs_hash,
            );

            let mut quotient_values_batch = eval_vanishing_poly_base_batch::<F, D>(
                common_data,
                &indices_batch,
                &shifted_xs_batch,
                vars_batch,
                &local_zs_batch,
                &next_zs_batch,
                &local_lookup_batch,
                &next_lookup_batch,
                &partial_products_batch,
                &s_sigmas_batch,
                betas,
                gammas,
                deltas,
                alphas,
                &z_h_on_coset,
                &lut_re_poly_evals_refs,
            );

            for (&i, quotient_values) in indices_batch.iter().zip(quotient_values_batch.iter_mut())
            {
                let denominator_inv = z_h_on_coset.eval_inverse(i);
                quotient_values
                    .iter_mut()
                    .for_each(|v| *v *= denominator_inv);
            }
            quotient_values_batch
        })
        .collect();
    quotient_values
}

#[cfg(all(test, feature = "gpu"))]
mod gpu_tests {
    use super::*;
    use crate::gpu::tests::{hash_chain_circuit, mixed_circuit};
    use crate::hash::hash_types::HashOut;
    use crate::plonk::circuit_data::CircuitData;
    use crate::plonk::config::PoseidonGoldilocksConfig;

    type F = crate::field::goldilocks_field::GoldilocksField;
    type C = PoseidonGoldilocksConfig;

    /// The prover's state after the commitment of the Z's.
    struct Commitments {
        public_inputs_hash: HashOut<F>,
        wires_commitment: PolynomialBatch<F, C, 2>,
        zs_commitment: PolynomialBatch<F, C, 2>,
        challenger: Challenger<F, <C as GenericConfig<2>>::Hasher>,
        betas: Vec<F>,
        gammas: Vec<F>,
        alphas: Vec<F>,
    }

    /// Runs the prover up to the commitment of the Z's.
    fn commitments(data: &CircuitData<F, C, 2>, inputs: PartialWitness<F>) -> Commitments {
        let prover_data = &data.prover_only;
        let common_data = &data.common;
        let config = &common_data.config;
        let partition_witness = generate_partial_witness(inputs, prover_data, common_data).unwrap();
        let public_inputs = partition_witness.get_targets(&prover_data.public_inputs);
        let public_inputs_hash = <C as GenericConfig<2>>::InnerHasher::hash_no_pad(&public_inputs);
        let witness = partition_witness.full_witness();
        let wires_values: Vec<PolynomialValues<F>> = witness
            .wire_values
            .iter()
            .map(|column| PolynomialValues::new(column.clone()))
            .collect();
        let mut timing = TimingTree::default();
        let wires_commitment = PolynomialBatch::<F, C, 2>::from_values_with_plan(
            wires_values,
            config.fri_config.rate_bits,
            config.zero_knowledge && PlonkOracle::WIRES.blinding,
            config.fri_config.cap_height,
            &mut timing,
            prover_data.fft_root_table.as_ref(),
            None,
            true,
        )
        .into_host();
        let mut challenger = Challenger::<F, <C as GenericConfig<2>>::Hasher>::new();
        common_data.fri_params.observe(&mut challenger);
        challenger.observe_hash::<<C as GenericConfig<2>>::Hasher>(prover_data.circuit_digest);
        challenger.observe_hash::<<C as GenericConfig<2>>::InnerHasher>(public_inputs_hash);
        challenger
            .observe_cap::<<C as GenericConfig<2>>::Hasher>(&wires_commitment.merkle_tree.cap);
        let num_challenges = config.num_challenges;
        let betas = challenger.get_n_challenges(num_challenges);
        let gammas = challenger.get_n_challenges(num_challenges);
        let mut partial_products_and_zs = all_wires_permutation_partial_products(
            &witness,
            &betas,
            &gammas,
            prover_data,
            common_data,
        );
        let plonk_z_vecs = partial_products_and_zs
            .iter_mut()
            .map(|partial_products_and_z| partial_products_and_z.pop().unwrap())
            .collect();
        let zs_partial_products = [plonk_z_vecs, partial_products_and_zs.concat()].concat();
        let zs_commitment = PolynomialBatch::<F, C, 2>::from_values(
            zs_partial_products,
            config.fri_config.rate_bits,
            config.zero_knowledge && PlonkOracle::ZS_PARTIAL_PRODUCTS.blinding,
            config.fri_config.cap_height,
            &mut timing,
            prover_data.fft_root_table.as_ref(),
        );
        challenger.observe_cap::<<C as GenericConfig<2>>::Hasher>(&zs_commitment.merkle_tree.cap);
        let alphas = challenger.get_n_challenges(num_challenges);
        Commitments {
            public_inputs_hash,
            wires_commitment,
            zs_commitment,
            challenger,
            betas,
            gammas,
            alphas,
        }
    }

    /// Runs the prover up to the quotient and compares the GPU and CPU quotients.
    fn compare_quotients(data: &CircuitData<F, C, 2>, inputs: PartialWitness<F>) {
        let prover_data = &data.prover_only;
        let common_data = &data.common;
        let Commitments {
            public_inputs_hash,
            wires_commitment,
            zs_commitment,
            betas,
            gammas,
            alphas,
            ..
        } = commitments(data, inputs);
        let cpu = compute_quotient_polys_cpu::<F, C, 2>(
            common_data,
            prover_data,
            &public_inputs_hash,
            &wires_commitment,
            &zs_commitment,
            &betas,
            &gammas,
            &[],
            &alphas,
        );
        let gpu = crate::gpu::quotient::try_compute_quotient_polys::<F, C, 2>(
            common_data,
            prover_data,
            &public_inputs_hash,
            &wires_commitment,
            &zs_commitment,
            &betas,
            &gammas,
            &alphas,
            None,
        )
        .expect("the GPU takes this circuit");
        assert_eq!(gpu.len(), cpu.len());
        for (c, (g, e)) in gpu.iter().zip(&cpu).enumerate() {
            assert_eq!(g.coeffs.len(), e.coeffs.len());
            let mismatches = g
                .coeffs
                .iter()
                .zip(&e.coeffs)
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                mismatches, 0,
                "challenge {c}: {mismatches} coefficients differ"
            );
        }
    }

    /// Runs the prover through the quotient commitment and compares the GPU
    /// and CPU opening sets and opening proofs for the same challenges.
    fn compare_opening_phase(data: &CircuitData<F, C, 2>, inputs: PartialWitness<F>) {
        let prover_data = &data.prover_only;
        let common_data = &data.common;
        let config = &common_data.config;
        let Commitments {
            public_inputs_hash,
            wires_commitment,
            zs_commitment,
            mut challenger,
            betas,
            gammas,
            alphas,
        } = commitments(data, inputs);
        let mut timing = TimingTree::default();
        let quotient_polys = compute_quotient_polys::<F, C, 2>(
            common_data,
            prover_data,
            &public_inputs_hash,
            &wires_commitment,
            &zs_commitment,
            &betas,
            &gammas,
            &[],
            &alphas,
            #[cfg(feature = "gpu")]
            None,
        )
        .into_host();
        let degree = common_data.degree();
        let quotient_degree = common_data.quotient_degree();
        let chunks: Vec<PolynomialCoeffs<F>> = quotient_polys
            .into_iter()
            .flat_map(|mut quotient_poly| {
                quotient_poly.trim_to_len(quotient_degree).unwrap();
                quotient_poly.chunks(degree)
            })
            .collect();
        let quotient_commitment = PolynomialBatch::<F, C, 2>::from_coeffs(
            chunks,
            config.fri_config.rate_bits,
            config.zero_knowledge && PlonkOracle::QUOTIENT.blinding,
            config.fri_config.cap_height,
            &mut timing,
            prover_data.fft_root_table.as_ref(),
        );
        challenger
            .observe_cap::<<C as GenericConfig<2>>::Hasher>(&quotient_commitment.merkle_tree.cap);
        let zeta = challenger.get_extension_challenge::<2>();
        let g = <F as Extendable<2>>::Extension::primitive_root_of_unity(common_data.degree_bits());
        let cpu_openings = OpeningSet::new_cpu(
            zeta,
            g,
            &prover_data.constants_sigmas_commitment,
            &wires_commitment,
            &zs_commitment,
            &quotient_commitment,
            common_data,
        );
        let gpu_openings = crate::gpu::opening::try_opening_set::<F, C, 2>(
            zeta,
            g,
            &prover_data.constants_sigmas_commitment,
            &wires_commitment,
            &zs_commitment,
            &quotient_commitment,
            common_data,
        )
        .expect("the GPU takes the opening set");
        assert_eq!(gpu_openings, cpu_openings, "opening set");
        challenger.observe_openings(&cpu_openings.to_fri_openings());
        let instance = common_data.get_fri_instance(zeta);
        let oracles = [
            &prover_data.constants_sigmas_commitment,
            &wires_commitment,
            &zs_commitment,
            &quotient_commitment,
        ];
        let mut cpu_challenger = challenger.clone();
        let cpu_proof = PolynomialBatch::<F, C, 2>::prove_openings_cpu(
            &instance,
            &oracles,
            &mut cpu_challenger,
            &common_data.fri_params,
            None,
            None,
            &mut timing,
        );
        let gpu_proof = crate::gpu::opening::try_prove_openings::<F, C, 2>(
            &instance,
            &oracles.map(|o| o as &dyn ProverOracle<F, C, 2>),
            &mut challenger,
            &common_data.fri_params,
            None,
            None,
            &mut timing,
        )
        .expect("the GPU takes the opening proof");
        assert_eq!(
            gpu_proof.commit_phase_merkle_caps, cpu_proof.commit_phase_merkle_caps,
            "commit phase caps"
        );
        assert_eq!(
            gpu_proof.final_poly, cpu_proof.final_poly,
            "final polynomial"
        );
        // The proof-of-work search takes any witness (`find_any`), so the
        // query phase over the same committed trees can
        // legitimately differ from here on.
        assert_eq!(
            gpu_proof.query_round_proofs.len(),
            cpu_proof.query_round_proofs.len()
        );
    }

    #[test]
    fn gpu_opening_phase_matches_cpu() {
        let _guard = crate::gpu::tests::ENABLE_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Some(_ctx) = crate::gpu::context() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let rows = 1usize << crate::gpu::MIN_LOG_LEAVES.saturating_sub(2).max(7);
        let (data, pw) = hash_chain_circuit(rows);
        compare_opening_phase(&data, pw);
        let (data, pw) = mixed_circuit();
        compare_opening_phase(&data, pw);
    }

    /// Compares the GPU's partial products and Z's (as the committed
    /// polynomials) with the CPU's.
    fn compare_partial_products(data: &CircuitData<F, C, 2>, inputs: PartialWitness<F>) {
        let prover_data = &data.prover_only;
        let common_data = &data.common;
        let Commitments {
            wires_commitment,
            zs_commitment,
            betas,
            gammas,
            ..
        } = commitments(data, inputs);
        let mut timing = TimingTree::default();
        let gpu = crate::gpu::permutation::try_partial_products_commitment(
            &wires_commitment,
            &betas,
            &gammas,
            prover_data,
            common_data,
            &mut timing,
            None,
        )
        .expect("the GPU takes the partial products")
        .into_host();
        assert!(wires_commitment.device_values.get().is_none());
        assert!(gpu.device_values.get().is_none());
        assert_eq!(gpu.polynomials.len(), zs_commitment.polynomials.len());
        for (k, (g, e)) in gpu
            .polynomials
            .iter()
            .zip(&zs_commitment.polynomials)
            .enumerate()
        {
            assert_eq!(g, e, "polynomial {k}");
        }
        assert_eq!(
            gpu.merkle_tree.leaves.len(),
            zs_commitment.merkle_tree.leaves.len()
        );
    }

    #[test]
    fn gpu_partial_products_match_cpu() {
        let _guard = crate::gpu::tests::ENABLE_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Some(_ctx) = crate::gpu::context() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let rows = 1usize << crate::gpu::MIN_LOG_LEAVES.saturating_sub(2).max(8);
        let (data, pw) = hash_chain_circuit(rows);
        compare_partial_products(&data, pw);
        let (data, pw) = mixed_circuit();
        compare_partial_products(&data, pw);
    }

    #[test]
    fn gpu_quotient_matches_cpu() {
        let _guard = crate::gpu::tests::ENABLE_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Some(_ctx) = crate::gpu::context() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let rows = 1usize << crate::gpu::MIN_LOG_LEAVES.saturating_sub(2).max(7);
        let (data, pw) = hash_chain_circuit(rows);
        compare_quotients(&data, pw);
        let (data, pw) = mixed_circuit();
        compare_quotients(&data, pw);
    }
}
