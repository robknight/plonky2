# GPU timeline and critical path

Diagnostics for locating GPU work and host delays within a proof. Both need a `gpu-profiling` build; Instruments traces also need macOS and Xcode's `xctrace`. The Python scripts use only the standard library.

## Timeline

Every compute pass on the proving path is named (for example `quotient`, `merkle leaves 524288 x 139`, `fft 135 x 2^19 inverse`). Record a Metal System Trace of the repository's recursive proof example and summarise it:

```sh
cargo build -p plonky2 --release --features gpu-profiling --example bench_recursion
xcrun xctrace record --template 'Metal System Trace' --output run.trace \
  --target-stdout run.log --launch -- ./target/release/examples/bench_recursion --size 14 --threads 1
python3 tools/gpu/gpu-timeline/timeline.py run.trace bench_recursion --skip 1 --sequence
```

The script joins each GPU interval with its pass label and identifies proof windows by inverse FFTs, separated by at least `--min-proof-ms`. It keeps windows containing a GPU quotient pass and reports GPU span, busy time, idle time, gaps by neighbouring passes and time by pass. Adjust `--start-label` and `--min-proof-ms` to the workload and check the resulting windows. `--skip` excludes leading windows; `--sequence` prints the last window's passes in order. The example proves different circuits, so its windows are not repeated measurements of one circuit.

On any platform with timestamp queries, `PLONKY2_GPU_PASS_TIMELINE=1` gives each named compute pass without dedicated timestamps a pair of queries. At the end of each proof the prover resolves them and logs `GPU_PASS` lines (`RUST_LOG=plonky2::gpu::timestamps=info`). Leave `PLONKY2_GPU_TIMESTAMPS` unset to include Merkle and quotient passes in this timeline. `timeline.py --log RUN.log --skip 1` summarises the log, one proof per flush; the first flush may include circuit setup. Copies and queue writes count as idle time. The recorder waits for the device at each flush, after the proof's timing.

An idle gap names its neighbours, not its cause; read it beside the host timing tree (`RUST_LOG=plonky2::util::timing=debug`) from the same process.

## Critical path

`PLONKY2_DELAY_SCOPE=<timing scope name>` with `PLONKY2_DELAY_MS=<ms>` makes the calling thread sleep at the start of every matching timing scope. See `plonky2/src/util/timing.rs`.

`critical_path.py BINARY OUT [--scopes FILE] [--delay MS] [--sleep S] [--proof-scope NAME] [--skip N] [--binary-arg ARG ...] [--verified-pattern REGEX]` runs a caller-supplied proving executable in fresh processes. The executable must print `TimingTree` logs with the root name selected by `--proof-scope` (default `prove`) and should generate several comparable proofs per process. Pass its arguments with repeated `--binary-arg` options; use `--binary-arg=--flag` for flags. `--verified-pattern` optionally counts one verification message per proof; without it, verification is unreported and remains the caller's responsibility.

Each scope is delayed twice, with undelayed processes interleaved. The default delay is 25 ms, and the first proof in each process is excluded from the median (`--skip`). Default scopes refer to this crate's prover; use `--scopes` for a file with one exact scope name per line, including any circuit-specific generator counts. The script preserves thread settings and writes `OUT/results.json`; `report_critical_path.py OUT/results.json` reports the share of each injected delay that reached the proof and flags failed or missing measurements.

A scope whose delay lengthens the proof by the full injected amount per occurrence is on the critical path; a shortfall indicates slack, such as GPU work already in flight. Sleeping can also affect CPU clocks and subsequent work, so compare measurements without treating small differences as exact.
