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
added about 18% to MainPod execution in earlier paired Xeon measurements and
about 25% with the final dependency scheduler. Worker CPU
time measures work rather than the critical path; concurrent CPU totals can
exceed elapsed time.

For exact-state replay of snapshots exported by the same source version:

```sh
RAYON_NUM_THREADS=8 P2_WITNESS_PROFILE=1 \
CARGO_TARGET_DIR=/tmp/witness-profile-target \
cargo +nightly run --release --manifest-path tools/witness-profile/Cargo.toml \
  -- /path/to/circuit_snapshot 24
```

The stem identifies `.p2w`, `.roots`, and `.random` files. Enable the development-only `witness-benchmarks` feature in the prover to
export an installed program with `P2_AOT_SNAPSHOT_DIR=/path/to/output`.
Replay checks every state slot against the checked executor after each run.
Its execute timer excludes decode, input I/O, comparison, and state wiping;
wiping has a separate diagnostic span. Replay does not measure live input
binding, sampling, or GPU proving.

Snapshot/export environment variables and process-wide benchmark overrides are
compiled only with `witness-benchmarks`; ordinary applications ignore them.
The replay CLI enables this feature itself.

For same-process instrumented controls, `benchmark_profile(bool)` changes
profiling between proofs. Call it only while no witness execution is active.
It has no effect when the profiling feature is absent.

Installed programs bind inputs through the existing canonical slot map when
their input mask matches the full boundary-root set; other masks sort and
binary-search their inputs.

`PLONKY2_EXECUTOR_BUFFER_BYTES=67108864` enables bounded reuse of wiped witness buffers. The
default uses fresh allocations. The pool retains at most 64 MiB of buffer
capacity; MainPod uses about 31.5 MiB for one state/sparse-readiness pair. Values are
wiped and readiness reset before retention. Profile allocation and wiping
together with full witness calls: execution savings can be larger than the
end-to-end improvement. `benchmark_buffer_reuse(bool)` is for serial paired
controls under the same exclusion rule as the profiling control.


With dependency scheduling, kind/task CPU totals remain available while global
layer-wall attribution is absent. Use `PLONKY2_WITNESS_BENCH_SCHEDULE=layers` when
measuring layer barriers as a control. The `layers` execution stage covers the
operation phase under either scheduler; CPU totals are work, not critical-path
wall time. Scalar Poseidon currently has whole-kind attribution; the nested
input/arithmetic/output counters describe the SIMD path.


With `witness-benchmarks`, comparisons can be selected at preparation/replay:

| Development variable | Values |
| --- | --- |
| `PLONKY2_WITNESS_BENCH_MODE` | `auto`, `checked` |
| `PLONKY2_WITNESS_BENCH_SCHEDULE` | `auto`, `layers`, `dependencies` |

Normal builds ignore these. The retired `PLONKY2_EXECUTOR_*` tuning names and
`P2_*` configuration aliases are not controls for either build. Layout, fusion,
chunk sizes, arithmetic simplification and gather tiles use private measured
implementation defaults; there is no public tuning config. Replay resolves its
resource/diagnostic settings explicitly after deterministic program decoding.
