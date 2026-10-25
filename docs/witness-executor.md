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

`WitnessConfig::default()` selects deterministic recommended settings. `WitnessConfig::from_env()` applies process defaults; the ordinary preparation methods use it. To override process settings, pass a configuration to `prepare_witness_program_with_config`.

```rust,ignore
let mut config = WitnessConfig::default();
config.cache = WitnessCache::Directory(cache_directory);
config.buffers = WitnessBuffers::Reuse { max_bytes: 64 * 1024 * 1024 };
config.fallback = WitnessFallback::Error;
let prepared = circuit.prepare_witness_program_with_config(&input_targets, config)?;
```

Settings are resolved during preparation. The following environment settings are supported; historical `P2_AOT_*` tuning names remain migration aliases, with the new names taking precedence.

| Variable | Meaning | Default |
| --- | --- | --- |
| `PLONKY2_EXECUTOR_CACHE_DIR` | Persistent executor directory | In memory |
| `PLONKY2_EXECUTOR_THREADS` | Program-owned execution pool size; zero uses the current Rayon pool | `0` |
| `PLONKY2_EXECUTOR_BUFFER_BYTES` | Retained capacity for wiped buffers; zero uses fresh allocations | `0` |
| `PLONKY2_EXECUTOR_FALLBACK` | Input-shape/failure policy: `worklist` or `error` | `worklist` |
| `PLONKY2_EXECUTOR_MODE` | Validated adaptive execution or checked control: `auto` or `checked` | `auto` |
| `PLONKY2_EXECUTOR_POSEIDON` | Hardware-eligible batching or individual permutations: `auto` or `scalar` | `auto` |
| `PLONKY2_EXECUTOR_LAYOUT` | Value placement: `outputs`, `canonical`, `representative`, `block4`, `block8` | `outputs` |
| `PLONKY2_EXECUTOR_CHAIN_LENGTH` | Fused arithmetic-chain limit; zero disables, otherwise 2–8 | `4` |
| `PLONKY2_EXECUTOR_CHUNK_SCALE` | Advanced operation chunk-size multiplier | `1` |
| `PLONKY2_EXECUTOR_SIMPLIFY_ARITHMETIC` | Simplify arithmetic coefficients | `true` |
| `PLONKY2_EXECUTOR_GATHER` | Final gather policy: `auto`, `cpu`, `gpu` | `auto` |
| `PLONKY2_EXECUTOR_GATHER_COLUMNS` | Advanced GPU upload staging tile width | `16` |
| `PLONKY2_EXECUTOR_GATHER_ROWS` | Advanced GPU upload staging tile height | `256` |

Invalid settings return `WitnessProgramError::InvalidConfiguration`. `WitnessCache::Disabled` overrides an environment cache directory. With the `parallel` feature, `config.thread_pool` accepts an `Arc<rayon::ThreadPool>` to share a caller-owned pool across prepared programs. A positive `threads` count creates an owned pool instead. These pools govern witness sampling, execution, gathering and wiping. Circuit compilation, host preparation, cache I/O and worklist fallback use the caller context. Avoid creating separate large pools for every circuit when a shared pool suits the application.

Clones of a circuit share its installed program slot. Program handles share immutable topology, input metadata and prepared tables. Wiped reusable state/readiness allocations belong to their execution resources; the last owner releases retained allocations. A reuse budget limits retained buffer payload, not total process memory or allocations for proofs currently in flight. `with_execution_config` reuses a program's topology and tables with different resource/execution settings while preserving its compilation policy.

## Failures and capability checks

By default, a mismatched input shape uses the worklist and logs the mismatch once. `WitnessFallback::Error` returns an error when the prepared executor cannot run the proof. Input conflicts remain errors; falling back does not make conflicting assignments valid.

The executor supports Goldilocks circuits with described generators and no lookup-table finalization. `witness_program_support()` reports the first unsupported generator. Cache-aware preparation memoizes unsupported capabilities on that circuit object; `witness_program_status()` returns `Unprepared`, `Ready` or `Unsupported` without cloning program metadata. `has_witness_program()` is a cheap presence query. Integrations can skip repeated preparation attempts for a known unsupported circuit and keep using the worklist.

Compilation and caching assume the circuit and its generator descriptions remain unchanged. Call `remove_witness_program()` before changing them; this also clears memoized unsupported capability. Removing a program invalidates prior validation handles. Build a new circuit object when creating a different circuit rather than changing one clone of shared circuit data. Removing a program atomically affects later lookups; an in-flight proof may finish with its existing program handle. Preparing and installing are synchronized operations, and concurrent proofs own distinct execution buffers.

For custom generators, implement `WitnessGenerator::witness_operation` only when the description exactly matches the generator's semantics. Use `check_against_worklist` on representative input cases, including edge values and aliases, before relying on compiled execution. Unsupported descriptions retain the ordinary generator path by leaving the executor uninstalled.

## Advanced lifecycle and measurement

The explicit `compile_witness_program`, `WitnessProgram::to_bytes`, `witness_program_from_bytes`, and `install_witness_program` methods remain available. Compilation does not install or persist a program. Installation checks generator semantics when a program comes from an independent circuit or a cleared lifecycle, then prepares its host execution tables. Handles already validated for the current circuit avoid another generator traversal. Explicit compile/load variants accept `WitnessConfig`; loaded programs retain their stored layout/fusion policy. `WitnessProgram::is_prepared()` and `config()` expose readiness and effective settings.

`WitnessPreparation` reports source, total elapsed time, cache reading, compilation, writing, host preparation and any nonfatal cache warning. Total elapsed time also includes identity calculation and synchronization. GPU device/context creation and prover-wide shader warm-up belong to the GPU prover lifecycle and should be measured separately. Report preparation, first proof, warm witness generation and total proof time separately; moving work into preparation does not make it disappear.

`tools/witness-profile` documents optional diagnostic attribution. Use ordinary builds for performance claims. The experiment history and retained/rejected candidates are in `docs/witness-executor-experiments.md`.

Development tooling uses a separate `witness-benchmarks` feature. Snapshot
export, replay and process-wide comparison overrides are unavailable in normal
application builds; see `tools/witness-profile/README.md`. Consumer code uses
`WitnessConfig` and preparation methods rather than internal region structures.
