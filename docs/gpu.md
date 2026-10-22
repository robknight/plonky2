# GPU proving and compiled witness programs

This fork adds two optional speed-ups to plonky2's prover. Neither changes proofs, circuit data or the verifier.

- **GPU proving** (the `gpu` feature): the prover's parallel phases run on a GPU through [wgpu](https://wgpu.rs).
- **Compiled witness programs** (every `std` build): a caller can compile a circuit's witness generators once and have later proofs run a compiled executor instead of the generic worklist.

The API reference is in the rustdoc of `plonky2::gpu` and `plonky2::plonk::witness_program`; this page is an overview.

## GPU proving

### What runs on the GPU

Merkle hashing, the batched IFFT, coset low-degree extensions and leaf layout of polynomial commitments, FRI folding and commitment, the quotient polynomial and the openings. The quotient kernel interprets a program generated per circuit from the gates' own constraint evaluations, so custom gates need no GPU code. Witness generation stays on the CPU (see below).

Every kernel reproduces the CPU implementation bit for bit; the tests compare digests, caps and coefficients exactly. A proof made on the GPU is identical in form to one made on the CPU and verifies with the unchanged verifier.

### Platforms

wgpu's Metal, Vulkan and DirectX 12 backends. Tested on Apple M4 (Metal) and NVIDIA GPUs under Linux (Vulkan). Adapters without 64-bit shader integers use a portable arithmetic flavour.

Buffers are bound whole, so the adapter must allow storage bindings as large as an extended polynomial batch (about 600 MB for pod2's MainPod circuit). Native Metal and Vulkan adapters do; WebGPU's default limits do not. Work that exceeds the adapter's limits runs on the CPU.

On Linux containers the NVIDIA Vulkan driver needs the Vulkan loader and `libegl1`; without `libegl1`, `vulkaninfo` fails with `ERROR_INCOMPATIBLE_DRIVER` and no GPU is found.

### Use

```toml
plonky2 = { git = "...", features = ["gpu"] }
```

Nothing else is required: on the first proof the prover creates a process-wide GPU context on the best available adapter, and proves on the CPU if there is none. The context is shut down automatically at normal process exit; `plonky2::gpu::shutdown()` releases it earlier, after proving workers have been joined.

To change settings, call `plonky2::gpu::configure` before the first proof:

```rust
let mut config = plonky2::gpu::GpuConfig::default();
config.proving_workers = 2; // proofs that run concurrently
plonky2::gpu::configure(config)?;
```

`GpuConfig` selects the backend and adapter, the execution mode (whether commitment leaves stay on the device and how the quotient is split between CPU and GPU), the number of concurrent proving workers, device memory budgets and the automatic shutdown. `plonky2::gpu::set_enabled(false)` keeps proving on the CPU. The library reads no settings from environment variables.

## Compiled witness programs

Witness generation computes every wire value from a proof's inputs by running the circuit's generators, and it runs on the CPU in every build. plonky2's default worklist runs each generator through its trait object and merges its outputs back into the witness, and that overhead outweighs the arithmetic. A **witness program** is compiled once per circuit: every generator becomes a typed operation over fixed slots, grouped into layers of independent operations, which the executor runs as tight parallel loops. It computes exactly the worklist's witness, two to three times faster on large circuits (on pod2's MainPod circuit, about 24 ms against 60 ms on an M4).

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

Compiling takes about 0.3 s per million generators on an Apple M4, several times longer on slower server CPUs. A stored program for pod2's MainPod circuit (1.6 million generators) is 20 MB, 1.9 MB with zstd. Loading checks the circuit, the build (`witness_program::FORMAT`, which callers can put in cache keys) and the generators' operations. Stored programs are a trusted input, like serialized circuit data: a checksum detects damage, not tampering.

A circuit can be compiled only if every generator describes its computation through `WitnessGenerator::witness_operation` and it has no lookup tables. Most of plonky2's own generators do (not those of lookups, the dummy-proof generator or one of the split gadgets); `witness_program_support` checks a circuit without compiling and names the first generator that does not. Libraries with custom generators implement `witness_operation` and can test their implementations on real circuits with `witness_program::check_against_worklist`, which fails on any value that differs from the worklist.

## Diagnostics and tools

- Debug logs need no switch: `RUST_LOG=plonky2::util::timing=debug` prints the prover's timing tree, `plonky2::witness_program::compile=debug` a program's compile phases, `plonky2::witness_program::executor=debug` the executor's report per proof.
- The `gpu-profiling` feature adds measurement switches (device timestamps, a per-pass timeline, Metal captures, program dumps), listed in the `plonky2::gpu` module documentation. They change what is measured, never the proof.
- `tools/gpu/gpu-timeline`: summarises a per-pass GPU timeline or an Instruments trace, and measures which prover phases are on the critical path.
- `tools/gpu/nvidia-check`: runs the GPU test suites and pod2 MainPod proofs on a disposable NVIDIA cloud machine.

## Testing

Run the tests in release mode: `cargo test --release --features gpu --lib` runs the differential tests of every kernel against the CPU on the available adapter, and `--features gpu-profiling` adds the diagnostic paths.
