//! Opt-in whole-proof benchmarks. Run each configuration in its own process.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use super::*;
use crate::iop::witness::{PartialWitness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::{CircuitConfig, CircuitData};
use crate::plonk::config::PoseidonGoldilocksConfig as C;

fn circuit(shape: &str) -> (CircuitData<F, C, 2>, PartialWitness<F>) {
    let mut config = CircuitConfig::standard_recursion_zk_config();
    config.zero_knowledge = shape != "hash-unblinded";
    let mut builder = CircuitBuilder::<F, 2>::new(config);
    let input = builder.add_virtual_target();
    builder.register_public_input(input);
    let mut pw = PartialWitness::new();
    pw.set_target(input, F::from_canonical_u64(7)).unwrap();
    match shape {
        "hash-small" | "hash-large" | "hash-unblinded" => {
            let rows = match shape {
                "hash-small" => 256,
                "hash-large" => 16384,
                _ => 4096,
            };
            let mut state = vec![input; 4];
            for _ in 0..rows {
                state = builder.hash_n_to_hash_no_pad::<H>(state).elements.to_vec();
            }
            builder.register_public_inputs(&state);
        }
        "arithmetic" => {
            let mut value = input;
            // Fill real arithmetic rows instead of padding with no-op gates.
            while builder.num_gates() < 4096 {
                value = builder.mul_add(value, input, input);
            }
            builder.register_public_input(value);
        }
        "lookup" => {
            let table =
                builder.add_lookup_table_from_fn(|x| (x + 1) % 8, &[0, 1, 2, 3, 4, 5, 6, 7]);
            let mut value = input;
            for _ in 0..8192 {
                value = builder.add_lookup_from_index(value, table);
            }
            builder.register_public_input(value);
        }
        _ => panic!("unknown residency benchmark shape: {shape}"),
    }
    (builder.build::<C>(), pw)
}

#[test]
#[ignore = "isolated unified-memory GPU benchmark; configure PLONKY2_GPU_BENCH_* variables"]
fn unified_residency_matrix() {
    let _ = env_logger::try_init();
    let ctx = context().expect("benchmark requires a GPU");
    assert!(ctx.mappable, "benchmark requires unified memory");
    let shape = std::env::var("PLONKY2_GPU_BENCH_SHAPE").unwrap_or("hash-small".into());
    let number = |key: &str, default| {
        std::env::var(key)
            .map(|x| x.parse::<usize>().unwrap())
            .unwrap_or(default)
    };
    let threads = number("PLONKY2_GPU_BENCH_THREADS", 1);
    let repeats = number("PLONKY2_GPU_BENCH_REPEAT", 13);
    let verify = std::env::var("PLONKY2_GPU_BENCH_VERIFY").as_deref() == Ok("1");
    assert!(threads > 0 && repeats > threads);
    let (data, pw) = circuit(&shape);
    let blinded = data.common.config.zero_knowledge;
    let plan = policy::ExecutionPlan::new(&data.common, &data.prover_only);
    let resident = plan.retain_leaves();
    // Check the actual commitment representation with the circuit's degree/rate.
    let probe = lde::from_values_with_residency::<F, C, 2>(
        &ctx,
        vec![PolynomialValues::new(vec![F::ONE; data.common.degree()])],
        data.common.config.fri_config.rate_bits,
        blinded,
        data.common.config.fri_config.cap_height,
        &mut TimingTree::default(),
        resident,
        false,
    );
    assert_eq!(
        probe.batch.merkle_tree.leaves.is_device_resident(),
        resident
    );
    drop(probe);
    println!("matrix shape={shape} degree_bits={} wires={} lookups={} blinded={blinded} resident={resident} threads={threads} verify={verify}",
        data.common.degree_bits(), data.common.config.num_wires, data.common.num_lookup_polys);
    drop(ctx);
    drop(plan);
    let prove_one = |i| {
        let start = Instant::now();
        let proof = data.prove(pw.clone()).unwrap();
        let elapsed = start.elapsed().as_secs_f64();
        if verify {
            data.verify(proof).unwrap();
        }
        println!("matrix-proof index={i} seconds={elapsed:.9} verified={verify}");
    };
    // Warm the shared circuit and adaptive share before starting workers.
    for i in 0..threads {
        prove_one(i);
    }
    let next = AtomicUsize::new(threads);
    let start = Instant::now();
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= repeats {
                        break;
                    }
                    prove_one(i);
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    let seconds = start.elapsed().as_secs_f64();
    let proofs = repeats - threads;
    println!(
        "matrix-throughput proofs={proofs} seconds={seconds:.9} proofs_per_second={:.9}",
        proofs as f64 / seconds
    );
    shutdown().unwrap();
    assert!(!is_available());
    println!("matrix shutdown completed");
}
