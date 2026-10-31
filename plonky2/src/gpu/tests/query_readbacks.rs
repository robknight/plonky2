//! Nested query readbacks must work even when the prover exhausts its pool.

use plonky2_maybe_rayon::*;

use super::*;

#[test]
fn resident_proofs_share_a_circuit_in_small_worker_pools() {
    let _guard = ENABLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(_ctx) = ctx() else { return };
    let rows = 1usize << crate::gpu::MIN_LOG_LEAVES.saturating_sub(2).max(7);
    let (data, pw) = hash_chain_circuit(rows);
    if !policy::ExecutionPlan::new(&data.common, &data.prover_only).retain_leaves() {
        eprintln!("resident query readbacks are not selected; skipping");
        return;
    }
    // A single worker exercises the sequential path; more workers exercise
    // nested query preparation while multiple proofs share the fixed oracle.
    for workers in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        pool.install(|| {
            (0..4usize).into_par_iter().for_each(|_| {
                let proof = data.prove(pw.clone()).unwrap();
                data.verify(proof).unwrap();
            });
        });
    }
}
