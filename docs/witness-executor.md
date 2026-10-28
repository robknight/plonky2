# Preparing fast witness generation

Build or load your circuit, record its input targets, and call `prepare_witness_program` once before accepting proof requests. Preparation loads or compiles an executor, validates it, initializes host lookup tables, and installs it on the circuit. Subsequent ordinary `prove` calls use it when their input shape matches. Unprepared circuits keep the ordinary worklist; `prove` never starts compilation or disk I/O on its own.

```rust,ignore
let prepared = circuit.prepare_witness_program(&input_targets)?;
println!("{:?} in {:?}", prepared.source, prepared.elapsed);
let proof = circuit.prove(inputs)?;
```

The input schema is a slice of targets from circuit construction. Values are irrelevant to preparation. Duplicates and aliases in the same copy-constraint class are accepted. `prepare_witness_program_like(&inputs)` is a convenience when an input witness is already available. `examples/witness_program.rs` is a complete example that proves and verifies twice.

## Persistent caching

Set `PLONKY2_EXECUTOR_CACHE_DIR` to let plonky2 manage compiled-executor storage. Without it, preparation is in memory. POD2 does not need to encode executors or maintain a registry to use this cache.

```sh
PLONKY2_EXECUTOR_CACHE_DIR=/var/cache/my-app/executors cargo run --release -p plonky2 --example witness_program
```

Run the example twice: the first process reports `Compiled`, and the second reports `DiskCache`. Repeated preparation of the same circuit object reports `Installed`. The cache stores circuit programs and constants, never witness values or random seeds. Treat it as trusted serialized circuit data; its checksum detects damage rather than deliberate tampering.

Cache identity includes the circuit digest, generator count, representative-map size, canonical input shape, witness-operation digest, compiler/encoding fingerprint, and layout/fusion policy. A different execution thread count, memory policy or Poseidon policy does not change the portable graph identity. Format compatibility is conservative: changes to the witness compiler/executor sources invalidate entries. Callers should not interpret cache filenames or edit entries.

Missing entries are compiled. Incompatible or damaged entries are rebuilt and replaced atomically. Per-entry file locks prevent duplicate compilation by cooperating processes. Valid entries can be read without write access. If the directory is unavailable or cannot be written, preparation can still succeed in memory and returns a `cache_warning`. POD2 circuit-data/proof caching is independent of executor caching.

## Configuration and resource ownership

The [runtime configuration audit](runtime-configuration.md) identifies deployment settings that can be controlled through environment defaults, alongside resource/lifecycle APIs that remain explicit.

`WitnessConfig` exposes only cache policy, CPU resources, retained-buffer budget and fallback. `WitnessConfig::default()` selects deterministic recommended settings. `WitnessConfig::from_env()` applies process defaults; the ordinary preparation methods use it. To override process settings, pass a configuration to `prepare_witness_program_with_config`.

```rust,ignore
let mut config = WitnessConfig::default();
config.cache = WitnessCache::Directory(cache_directory);
config.buffers = WitnessBuffers::Reuse { max_bytes: 64 * 1024 * 1024 };
config.fallback = WitnessFallback::Error;
let prepared = circuit.prepare_witness_program_with_config(&input_targets, config)?;
```

Libraries supplying suggested policy should use `WitnessConfig::from_env_with_defaults(defaults)` so application environment variables override those suggestions. The example above is an explicit application override. POD2 resolves its defaults on the plonky2 side and exposes only an optional caller-owned witness pool, rather than forwarding these policy fields to applications.

The default scheduler lets independent branches advance as soon as their inputs are ready. It retains global layers on single-thread pools and builds without `parallel`. Layered and checked comparisons are available only in development tooling. They are not application configuration options. Preparation builds dependency metadata once per execution-resource configuration; no generated Rust or external compiler is involved.

Settings are resolved during preparation. These are the only four executor environment settings supported by normal applications:

| Variable | Meaning | Default |
| --- | --- | --- |
| `PLONKY2_EXECUTOR_CACHE_DIR` | Persistent executor directory | In memory |
| `PLONKY2_EXECUTOR_THREADS` | Program-owned execution pool size; zero uses the current Rayon pool | `0` |
| `PLONKY2_EXECUTOR_BUFFER_BYTES` | Retained capacity for wiped buffers; zero uses fresh allocations | `0` |
| `PLONKY2_EXECUTOR_FALLBACK` | Input-shape/failure policy: `worklist` or `error` | `worklist` |

Invalid settings return `WitnessProgramError::InvalidConfiguration`. `WitnessCache::Disabled` overrides an environment cache directory. With the `parallel` feature, `config.thread_pool` accepts an `Arc<rayon::ThreadPool>` to share a caller-owned pool across prepared programs. A positive `threads` count creates an owned pool instead. These pools govern witness sampling, execution, gathering and wiping. Circuit compilation, host preparation and cache I/O use the caller context. The worklist, including fallback, runs generators on the pool sized by `PLONKY2_GENERATOR_THREADS` (see `docs/runtime-configuration.md`). Avoid creating separate large pools for every circuit when a shared pool suits the application.

Clones of a circuit share its installed program slot. Program handles share immutable topology, input metadata and prepared tables. Wiped reusable state/readiness allocations belong to their execution resources; the last owner releases retained allocations. A reuse budget limits retained buffer payload, not total process memory or allocations for proofs currently in flight. `with_execution_config` reuses a program's topology and tables with different resource/fallback settings while preserving its compilation policy.

## Hardware selection and portable binaries

Automatic hardware selection uses eight independent permutations together when x86 CPU and operating-system support for AVX-512 is available. Portable x86 binaries use an isolated, runtime-dispatched kernel; the rest of the binary does not require AVX-512. Builds already targeting all required AVX-512 features retain their existing native specialization. Non-AVX-512 x86 and ARM use the scalar/production field path. Small groups and tails also use individual permutations. Individual-permutation comparisons are available in benchmark builds.

This dispatch accelerates witness execution; it does not change hardware selection in the rest of the prover. Whole-binary `-C target-cpu=native` remains an application build choice for a known deployment CPU. Portable applications do not need that flag to get this witness kernel. Scheduling and batching are automatic; no MainPod identifier or circuit-specific generated crate is required.

## Failures and capability checks

By default, a mismatched input shape uses the worklist and logs the mismatch once. `WitnessFallback::Error` returns an error when the prepared executor cannot run the proof. Input conflicts remain errors; falling back does not make conflicting assignments valid.

The executor supports Goldilocks circuits with described generators and no lookup-table finalization. `witness_program_support()` reports the first unsupported generator. Cache-aware preparation memoizes unsupported capabilities on that circuit object; `witness_program_status()` returns `Unprepared`, `Ready` or `Unsupported` without cloning program metadata. `has_witness_program()` is a cheap presence query. Integrations can skip repeated preparation attempts for a known unsupported circuit and keep using the worklist.

Compilation and caching assume the circuit and its generator descriptions remain unchanged. Call `remove_witness_program()` before changing them; this also clears memoized unsupported capability. Removing a program invalidates prior validation handles. Build a new circuit object when creating a different circuit rather than changing one clone of shared circuit data. Removing a program atomically affects later lookups; an in-flight proof may finish with its existing program handle. Preparing and installing are synchronized operations, and concurrent proofs own distinct execution buffers.

For custom generators, implement `WitnessGenerator::witness_operation` only when the description exactly matches the generator's semantics. Use `check_against_worklist` on representative input cases, including edge values and aliases, before relying on compiled execution. Unsupported descriptions retain the ordinary generator path by leaving the executor uninstalled.

## Advanced lifecycle and measurement

The explicit `compile_witness_program`, `WitnessProgram::to_bytes`, `witness_program_from_bytes`, and `install_witness_program` methods remain available. Compilation does not install or persist a program. Installation checks generator semantics when a program comes from an independent circuit or a cleared lifecycle, then prepares its host execution tables. Handles already validated for the current circuit avoid another generator traversal. Explicit compile/load variants accept `WitnessConfig`; loaded programs retain their stored layout/fusion policy. `WitnessProgram::is_prepared()` and `config()` expose readiness and effective settings.

`WitnessPreparation` reports source, total elapsed time, cache reading, compilation, writing, host preparation and any nonfatal cache warning. Total elapsed time also includes identity calculation and synchronization. GPU device/context creation and prover-wide shader warm-up belong to the GPU prover lifecycle and should be measured separately. Report preparation, first proof, warm witness generation and total proof time separately; moving work into preparation does not make it disappear.

`tools/witness-profile` documents optional diagnostic attribution. Use ordinary builds for performance claims. The experiment history and retained/rejected candidates are in `docs/witness-executor-experiments.md`; the completion measurements and scope are in `docs/witness-executor-acceptance.md`.

Development tooling uses a separate `witness-benchmarks` feature. Snapshot
export, replay and process-wide comparison overrides are unavailable in normal
application builds; see `tools/witness-profile/README.md`. Consumer code uses
`WitnessConfig` and preparation methods rather than internal region structures.


The measured implementation fixes output-local placement, four-operation chain
fusion, arithmetic simplification, standard operation chunks and 16×256 gather
tiles internally. Scheduling and Poseidon batching are selected automatically.
The former tuning fields and enums have been removed from the public config;
old `PLONKY2_EXECUTOR_*` tuning names and `P2_*` configuration aliases are ignored.
Development comparisons use `PLONKY2_WITNESS_BENCH_*` only with the
`witness-benchmarks` feature; see `tools/witness-profile/README.md`.
