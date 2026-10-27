# GPU proving and compiled witness programs

This fork adds two optional speed-ups to plonky2's prover. Neither changes proofs, circuit data or the verifier.

- **GPU proving** (the `gpu` feature): the prover's parallel phases run on a GPU through [wgpu](https://wgpu.rs).
- **Compiled witness programs** (with `std` and `rand_chacha`, both enabled by default): a caller can compile a circuit's witness generators once and have later proofs run a compiled executor instead of the generic worklist.

The API reference is in the rustdoc of `plonky2::gpu` and `plonky2::plonk::witness_program`; this page is an overview.

## GPU proving

### What runs on the GPU

Merkle hashing, the batched IFFT, coset low-degree extensions and leaf layout of polynomial commitments, the permutation argument's partial products and Z polynomials, FRI folding and commitment, the quotient polynomial, the openings and the proof-of-work search. The quotient kernel interprets a program generated per circuit from the gates' own constraint evaluations; the interpreter itself is compiled per program, with one switch case per opcode the program uses, numbered by frequency, since the per-op dispatch is the largest part of the kernel's time. A constraint computed by the op before its weighting is weighted by that op, and the program's registers are reassigned by a linear scan after compilation. Custom gates need no GPU code when their recursive constraint evaluations use operations the tracer supports; unsupported circuits use the CPU quotient path. Witness generation stays on the CPU (see below).

Merkle trees are built on the GPU for `PoseidonHash` and, with the `poseidon_bn128` feature, for `PoseidonBN128Hash` (Poseidon over BN254, used by `PoseidonBN128GoldilocksConfig` for proofs that are verified inside a BN254 SNARK); with any other hasher the commitments, FRI and openings stay on the CPU. The proof-of-work search runs on the GPU whenever the hasher's sponge uses the Goldilocks Poseidon permutation, which `PoseidonBN128Hash` does.

Every kernel reproduces the CPU implementation bit for bit; the tests compare digests, caps and coefficients exactly. A proof made on the GPU is identical in form to one made on the CPU and verifies with the unchanged verifier.

Resident proofs keep commitment leaves in the same column-major buffers used by the quotient, and gather the queried rows on demand. This avoids a full duplicate row layout (556 MiB for 139 columns of 2^19 elements). Hybrid proofs on unified memory keep the row layout for their CPU quotient share. Resident FRI layers also reuse their column buffers; hybrid and CPU-quotient proofs retain the existing FRI row layout to avoid extra query dispatches. Full host access, equality and serialization still materialize leaves in their original row order.

The prover releases column LDE caches when no later computation consumes them: after commitments for a CPU quotient, and after committing the quotient itself. Resident leaves can still share those buffers. Freed storage may remain in the bounded reuse pool; live buffer savings and allocated device memory are different measurements.

### Platforms

wgpu's Metal, Vulkan and DirectX 12 backends. Tested on Apple M4 (Metal) and NVIDIA GPUs under Linux (Vulkan). Adapters without 64-bit shader integers use a portable arithmetic flavour.

Buffers are bound whole, so the adapter must allow storage bindings as large as an extended polynomial batch (for example, 139 columns of 2^19 field elements take about 583 MB). Work that exceeds the adapter's limits runs on the CPU.

On Linux containers the NVIDIA Vulkan driver needs the Vulkan loader and `libegl1`; without `libegl1`, `vulkaninfo` fails with `ERROR_INCOMPATIBLE_DRIVER` and no GPU is found.

### Use

```toml
plonky2 = { git = "...", features = ["gpu"] }
```

Nothing else is required: on the first proof the prover creates a process-wide GPU context on the best available adapter, and proves on the CPU if there is none. The context is shut down automatically at normal process exit; `plonky2::gpu::shutdown()` releases it earlier, after proving workers have been joined.

Deployment settings are read automatically from typed `PLONKY2_GPU_*` environment defaults. For example, `PLONKY2_GPU_PROVING_WORKERS=2` sizes retained resources for two concurrent proofs; `PLONKY2_GPU_RETAINED_BYTES=0` disables retained caches, and `PLONKY2_GPU_ENABLED=false` avoids GPU initialization. POD2 does not need to forward a config structure. The complete variable table and API audit are in [runtime configuration](runtime-configuration.md).

The environment selects backend/adapter, execution mode, worker hint, memory budgets, hybrid CPU share and exit cleanup. Invalid values return an error from `GpuConfig::from_env()` or the fallible `plonky2::gpu::config()`, without creating a device. Automatic GPU routing warns and falls back to the CPU on invalid settings. Process defaults are captured once on first configuration access or context use.

Libraries should leave platform defaults alone unless they have useful additional knowledge. They can offer suggestions with `gpu::configure_defaults(defaults)`, which applies the application's environment afterward and never overwrites explicit application configuration. `GpuConfig::from_env_with_defaults(defaults)` provides the same precedence for configuration resolution without global registration. POD2 does not require a GPU configuration wrapper.

For an explicit application override, preserve environment defaults and change only the fields the application knows:

```rust,ignore
let mut config = plonky2::gpu::GpuConfig::from_env()?;
config.proving_workers = actual_workers;
plonky2::gpu::configure(config)?;
```

`GpuConfig::default()` remains deterministic and intentionally bypasses environment settings. Explicit application `configure` remains available. Independently owned `GpuContext::with_config` contexts serve low-level operations; creating one does not route normal proofs or POD2 through it. `plonky2::gpu::set_enabled(false)` temporarily routes work to the CPU; orderly shutdown remains an explicit lifecycle API as well as an exit default.

## Compiled witness programs

Witness generation computes every wire value from a proof's inputs by running the circuit's generators, and it runs on the CPU in every build. plonky2's default worklist runs each generator through its trait object and merges its outputs back into the witness. A **witness program** is compiled once per circuit: every generator becomes a typed operation over fixed slots, grouped into layers of independent operations, which the executor runs as tight parallel loops. It computes the same witness as the worklist for the same inputs and random values, reducing scheduling and merge overhead.

Nothing happens implicitly: `prove` never compiles, stores or loads a program. The caller decides:

```rust
// Compile for proofs that set the same targets as `pw` (values do not matter).
let program = circuit_data.compile_witness_program_like(&pw)?;
// Optionally keep it across processes.
let bytes = program.to_bytes();
let program = circuit_data.witness_program_from_bytes(&bytes)?;
// Later proofs whose inputs set the same targets use the executor.
circuit_data.install_witness_program(program)?;
```

Compilation time and stored program size depend on the circuit. Loading checks the circuit, the build (`witness_program::FORMAT`, which callers can put in cache keys) and the generators' operations. Stored programs are a trusted input, like serialized circuit data: a checksum detects damage, not tampering.

A circuit can be compiled only over the Goldilocks field, if every generator describes its computation through `WitnessGenerator::witness_operation` and it has no lookup tables. Most of plonky2's own generators do (not those of lookups, the dummy-proof generator or nonbinary `BaseSumGenerator`s); `witness_program_support` checks a circuit without compiling and names the first generator that does not. Libraries with custom generators implement `witness_operation` and can test their implementations on real circuits with `witness_program::check_against_worklist`, which fails on any value that differs from the worklist.

## Diagnostics and tools

- Debug logs need no switch: `RUST_LOG=plonky2::util::timing=debug` prints the prover's timing tree, `plonky2::witness_program::compile=debug` a program's compile phases, `plonky2::witness_program::executor=debug` the executor's report per proof.
- The `gpu-profiling` feature adds measurement switches (device timestamps, a per-pass timeline, Metal captures, program dumps), listed in the `plonky2::gpu` module documentation. They change what is measured, never the proof.
- `tools/gpu/gpu-timeline`: summarises a per-pass GPU timeline or an Instruments trace, and measures which prover phases are on the critical path.
- `tools/gpu/nvidia-check`: packages this repository and runs its GPU library suites and recursive proof example on a disposable NVIDIA machine.
- `cargo test --release --features gpu-profiling bench_fft_tile_plans -- --ignored --nocapture --test-threads=1` compares warmed 1024- and 2048-element FFT tiles on identical buffers. Larger tiles remain experimental: the M4 measurements regressed with both 256- and 512-thread workgroups, so production retains the 1024-element plan. `bench_fft_passes` times each stage group of a wires-sized LDE and IFFT with one tile per workgroup and with the pipelined kernel, whose workgroups each loop over several tiles and load the next one while transforming the current one (`PLONKY2_GPU_FFT_TILES_PER_WORKGROUP`; the default is 8 on unified-memory adapters, where the M4's wires LDE takes about 50 ms instead of 60 and its IFFT 7 ms instead of 11, and one tile per workgroup on discrete adapters until measured there).

## Testing

Run the tests in release mode: `cargo test --release --features gpu --lib` runs the differential tests of every kernel against the CPU on the available adapter, `--features gpu,poseidon_bn128` adds the BN254 Poseidon kernels, and `--features gpu-profiling` adds the diagnostic paths.
