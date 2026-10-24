# Witness execution profiling

Build with `witness-profiling` and set `P2_WITNESS_PROFILE=1` to attribute live
witness generation without serializing its dependency layers. The feature is
off by default. Normal builds contain no diagnostic clocks or counters.

`WITNESS_PIPELINE` separates input discovery, root binding, and CSPRNG sampling.
`WITNESS_PROFILE` reports allocation, seeding, layers, alias checks, readiness
checks, and wiping as `stage_ns`, in that order. `kind_cpu_ns`, `kind_tasks`, and
`kind_lanes` use `OperationKind` codes. `poseidon_cpu_ns` contains input packing,
arithmetic, and output handling for native batches. These CPU totals overlap
with the Poseidon kind total; do not add them together. CPU clocks use the
current thread clock on Unix. `WITNESS_LAYERS` reports the twelve longest layers
as `(index, wall_ns, lanes)` tuples.

Use uninstrumented controls for speed comparisons. Diagnostic instrumentation
added about 18% to MainPod execution in paired Xeon measurements. Worker CPU
time measures work rather than the critical path; concurrent CPU totals can
exceed elapsed time.

For exact-state replay of snapshots exported by the same source version:

```sh
RAYON_NUM_THREADS=8 P2_AOT_MODE=adaptive P2_AOT_SIMPLIFY_ARITH=1 \
P2_POSEIDON_MODE=batch8-fft P2_WITNESS_PROFILE=1 \
CARGO_TARGET_DIR=/tmp/witness-profile-target \
cargo +nightly run --release --manifest-path tools/witness-profile/Cargo.toml \
  -- /path/to/circuit_snapshot 24
```

The stem identifies `.p2w`, `.roots`, and `.random` files. A prover using an
installed program can export them with `P2_AOT_SNAPSHOT_DIR=/path/to/output`.
Replay checks every state slot against the checked executor after each run.
Its execute timer excludes decode, input I/O, comparison, and state wiping;
wiping has a separate diagnostic span. Replay does not measure live input
binding, sampling, or GPU proving.

For same-process instrumented controls, `benchmark_profile(bool)` changes
profiling between proofs. Call it only while no witness execution is active.
It has no effect when the profiling feature is absent.
