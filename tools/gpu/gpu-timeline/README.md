# GPU timeline and critical path

Two diagnostics for finding where a proof's wall time goes beyond its GPU work. Both need a `gpu-profiling` build; on macOS the timeline needs Xcode's `xctrace`.

## Timeline

Every compute pass on the proving path is named (for example `quotient`, `merkle leaves 524288 x 139`, `fft 135 x 2^19 inverse`), so Instruments, Xcode captures and this script can tell passes and commitments apart. Record a Metal System Trace of a proving process and summarise it:

```sh
xcrun xctrace record --template 'Metal System Trace' --output run.trace \
  --target-stdout run.log --launch -- ./mainpod_prove --proofs 6
python3 timeline.py run.trace mainpod_prove --skip 4 --sequence
```

The script joins each GPU interval with its pass label, splits the trace into proofs at the wires IFFT (an inverse FFT after an idle gap, at least `--min-proof-ms` apart, keeping windows that contain a quotient pass), and prints per proof the GPU span, busy time and idle time, the idle gaps grouped by the passes either side, and GPU time by pass. `--skip` excludes leading windows; early windows include circuit setup and the cold proof, so check the busy column before choosing it. `--sequence` prints the last proof's passes in order with the gap before each.

On any platform, `PLONKY2_GPU_PASS_TIMELINE=1` (profiling builds) gives every named compute pass a pair of device timestamp queries; at the end of each proof the prover resolves them and logs one `GPU_PASS` line per pass (`RUST_LOG=plonky2::gpu::timestamps=info`). `timeline.py --log RUN.log --skip 2` summarises such a log, one proof per flush (the first flush includes circuit setup and the EmptyPod proof). Copies and queue writes are not passes, so they count as idle time; the recorder waits for the device at each flush, after the proof's own timing. On the M4 it agrees with the Instruments trace (GPU busy about 600 ms per warm proof).

An idle gap names its neighbours, not its cause; read it beside the host timing tree (`RUST_LOG=plonky2::util::timing=debug`) of the same process.

## Critical path

`PLONKY2_DELAY_SCOPE=<timing scope name>` with `PLONKY2_DELAY_MS=<ms>` makes the calling thread sleep at the start of every timing scope with exactly that name (see `plonky2/src/util/timing.rs`). `critical_path.py BINARY OUT [--scopes FILE] [--delay MS] [--sleep S]` (`BINARY` is pod2's `mainpod_prove`, built with `witness_executor`, and `disk_cache` so that later processes load the compiled witness programs) delays each of a list of MainPod scopes (25 ms by default) in fresh processes, two per scope, with undelayed processes interleaved, and records whole-proof medians in `OUT/results.json`; `report_critical_path.py OUT/results.json` prints each scope's share of its injected delay that reached the proof. It keeps the caller's pool and thread settings. A scope whose delay lengthens the proof by the full amount times its occurrences per proof is on the critical path; a shortfall is the slack at that point, for example GPU work already in flight. Sleeping lets the CPU and the worker threads go idle, which slows the work that follows, so a fully critical scope can read above 100%; compare scopes with each other rather than reading small differences.
