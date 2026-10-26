//! Run twice with PLONKY2_EXECUTOR_CACHE_DIR set to demonstrate a cache reload.
use anyhow::Result;
use plonky2::field::goldilocks_field::GoldilocksField as F;
use plonky2::field::types::Field;
use plonky2::hash::poseidon::PoseidonHash;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::PoseidonGoldilocksConfig as C;

fn main() -> Result<()> {
    let _ = env_logger::try_init();
    let mut builder = CircuitBuilder::<F, 2>::new(CircuitConfig::standard_recursion_config());
    let a = builder.add_virtual_target();
    let b = builder.add_virtual_target();
    let product = builder.mul(a, b);
    let hash = builder.hash_n_to_hash_no_pad::<PoseidonHash>(vec![a, b, product]);
    builder.register_public_inputs(&hash.elements);
    let circuit = builder.build::<C>();

    // The schema comes from circuit construction; no dummy witness is needed.
    let prepared = circuit.prepare_witness_program(&[a, b])?;
    println!(
        "executor {:?}, ready in {:?}",
        prepared.source, prepared.elapsed
    );
    if let Some(warning) = &prepared.cache_warning {
        eprintln!("executor cache: {warning}");
    }
    for (a_value, b_value) in [(7, 11), (13, 17)] {
        let mut inputs = PartialWitness::new();
        inputs.set_target(a, F::from_canonical_u64(a_value))?;
        inputs.set_target(b, F::from_canonical_u64(b_value))?;
        let proof = circuit.prove(inputs)?;
        circuit.verify(proof)?;
    }
    println!("two proofs verified");
    Ok(())
}
