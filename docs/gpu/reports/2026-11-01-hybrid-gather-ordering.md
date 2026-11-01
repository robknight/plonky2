# Hybrid quotient gather ordering on Apple Metal

The hybrid quotient can delay CPU row gathers behind the large GPU quotient dispatch, causing repeated proof latency spikes. Waiting for the gather submissions before launching the quotient removes this dependency stall while preserving overlap between host row decoding and GPU evaluation. The arithmetic, circuit and proof format stay unchanged.

Fable identified the scheduling problem in the [follow-up deep dive](/Users/robknight/Projects/0xPARC/do/.benchmark-artifacts/gpu-deep-dive-2026-10-31/README.md#correction-and-follow-up-1-november-2026). The saved timeline independently confirms the three delayed gathers: 186–189 ms, followed by 167–175 ms gaps between quotient evaluation and its inverse FFT. The source previously submitted both gathers first but finished them in the CPU branch after queuing the quotient. Submission order alone did not guarantee completion order for these independent buffers on Metal.

## Implementation

A private `PendingLeafRows` handle separates device completion from host decoding. Both gathers are submitted, then their exact submission indices are waited on before the GPU quotient is submitted. The existing CPU branch still decodes the rows, installs the compact row cache, evaluates its quotient share and writes its output while the GPU share runs.

The wait is conditional on an actual pending device gather. Device-resident proofs with no CPU share, host-backed leaves, already prepared rows and non-mappable adapters do not acquire a new gather wait. This addresses a hybrid unified-memory dependency rather than identifying a particular circuit or chip. No public setting or experiment flag is retained. Independent concurrent proofs can still contend for device resources; the dependency prevents this proof's own quotient from starving its gathers.

## Measurement method

Apple M4, 10 GPU cores, 32 GiB, macOS 27.2 (26B123), Xcode 26.3. Release Rust nightly-2026-03-14, original probe manifest/source path, same stock CSP sources, compiled u32 witness executor and default empty MainPod probe. Hardware Metal is required, software adapters are disabled, witness fallback is Error, and every measured proof is verified.

Each comparison uses baseline/ordered/ordered/baseline process order. Discard the first two proofs per process. POD2 uses 31 proofs per process; small hashes use 11. Means and percentiles below pool all retained proofs with equal counts per process. The companion JSON also records every process mean and median. No build, test or other profiling experiment ran concurrently with timings. Performance comparisons have timestamps disabled; diagnostic timelines are separate runs.

The first prototype finished both complete row preparations before launching the quotient. It reduced POD2 mean time from 600.886 to 564.205 ms, but also serialized 7–11 ms of host row preparation. The refined implementation waits only for device completion. Its same-binary comparison follows.

## Refined implementation results

| Case | Baseline mean ms | Ordered mean ms | Mean time reduction |
| --- | ---: | ---: | ---: |
| POD2 | 600.644 | 566.345 | 5.71% |
| Poseidon16 | 104.802 | 102.070 | 2.61% |
| SHA128 | 221.095 | 219.290 | 0.82% |
| Keccak128 | 204.572 | 202.523 | 1.00% |

POD2 p95 fell from 720.382 to 587.775 ms. Of 58 retained proofs per variant, 12 baseline proofs exceeded the predeclared 650 ms threshold, versus zero ordered proofs. Median of process medians was 577.738 versus 567.333 ms. Across all four cases, 256 proofs verified.

Hash mean changes are small and should be described as roughly flat to modestly better. SHA's median of process medians moved slightly the other way, 216.844 to 218.446 ms. Between-process drift was substantial, so the precise percentage improvements are estimates. The result establishes a lower mean and reduced tail for these POD2 samples, not elimination of every possible latency spike or a universal 10% improvement.

## Diagnostic timelines

The full-preparation prototype's separate 11-proof timelines show two baseline gathers overlapping quotient evaluation, each accompanied by an approximately 160 ms gap before the inverse FFT. Ordered gathers completed before quotient evaluation in all 11 proofs and never exceeded 1.877 ms. One ordered proof still had a 58 ms CPU completion gap; other CPU scheduling or adaptive-share delays remain possible. These timeline timings include profiling overhead and are not pooled into the clean performance comparison.

`fable-timeline-check.json`, `timeline-baseline.json` and `timeline-ordered.json` preserve the pass-level analysis. Each timeline joins the last N quotient-containing flushes to N probe measurements, excluding circuit preparation's internal proofs.

## Reproduction and receipts

The source baseline is `4ffd4c6e` with report-only HEAD `5c3e4376`. The isolated worktree was restored at `/private/tmp/csp-opt-20261031/plonky2`; the user's original branches and CSP dependency pins are unchanged.

`prototype.patch` and `binaries/prototype` contain the full-preparation prototype. `wait-only.patch` and `binaries/wait-only` contain the refined candidate with a profiling-only comparison switch. Both archives compare the same binary with only `PLONKY2_GPU_ORDER_HYBRID_GATHERS=1` toggled. Their metadata files record hashes and source state. Final source removes this switch.

Build with the original `../csp-profile-2026-10-31/Cargo.toml` and run the probe as `BINARY CASE warm COUNT`. Clear inherited PLONKY2/RAYON/METAL_CAPTURE variables, select backend Metal, enable GPU, disallow software adapters and set executor fallback to Error. Select the experiment flag only for archived candidates. For separate diagnostics, set `PLONKY2_GPU_PASS_TIMELINE=1` and enable the timestamp logger, leaving dedicated per-kernel timestamp spans disabled.

`analyse_runs.py` validates proof records and computes process/pooled statistics; `analyse_timeline.py` extracts gathers, quotient duration and inverse FFT gaps. Raw proof logs, patches, build logs and summaries remain alongside this report.

## Final source acceptance

The final source always enforces the gather dependency, retains host decoding in the CPU branch, and has no experimental flag. A second comparison uses the final executable against the archived refined binary with its flag off, in ordered/baseline/baseline/ordered process order. The original source/manifest path and release debug level 1 are preserved. Every proof is verified. Counts are 31 proofs per POD2 process, nine per fresh Poseidon process and 11 per SHA512 process; discard the first two as before.

| Case | Baseline mean ms | Final mean ms | Mean time reduction |
| --- | ---: | ---: | ---: |
| POD2 warm | 597.013 | 579.707 | +2.90% |
| Poseidon16 fresh | 114.109 | 112.267 | +1.61% |
| SHA512 warm | 460.080 | 462.483 | -0.52% |

POD2 p95 was 715.534 versus 625.010 ms. There were again 12 of 58 retained baseline proofs above 650 ms, versus zero of 58 final proofs. Median of process medians was 573.228 versus 577.259 ms; a typical-latency improvement is not established in this repeat. POD2 ordered process medians ranged from 541.900 to 612.618 ms, making between-process drift material.

A background-load snapshot after the measurements showed macOS `mediaanalysisd` at 138% CPU and `photolibraryd` at 55.4% CPU; `pmset -g therm` reported no recorded thermal/performance warning. Background tasks were left running. This snapshot cannot establish their load throughout the runs, but it reinforces the need to treat the exact percentages cautiously. SHA512's final mean was 0.52% slower and its process-median estimate 1.57% slower in these noisy samples; this is not evidence of a uniform improvement across workloads. Fresh Poseidon was modestly faster.

Across the refined and final comparisons, POD2 mean time decreased by approximately 3–6%, with the repeatable large gather-stall tail removed in both samples. The final timeline verifies that both gathers finish before quotient evaluation in all 11 measured proofs; the longest gather was 1.585 ms. Its largest remaining quotient-to-IFFT gap was 40.859 ms, so the change does not eliminate every CPU completion delay.

## Validation

The final GPU suite passed: **73 passed, 25 ignored**. Separate processes passed eight verified concurrent hybrid proofs, a single-Rayon-thread hybrid proof, unified resident proof/shutdown and emulated discrete-memory policy proof/shutdown. The latter exercises the policy on M4; it is not an NVIDIA hardware measurement. Release CPU-only and serial GPU compilation checks passed. Clippy reported the same seven existing warnings, all outside the changed files.

**617 measured/profiled proofs verified** in this follow-up: 124 full-preparation prototype proofs, 22 prototype timeline proofs, 256 refined-candidate proofs, 204 final acceptance proofs and 11 final timeline proofs. Correctness-suite internal proofs are additional and are not counted as benchmark measurements.

`final.patch`, `binaries/final`, `final-metadata.json`, `final-summary.json`, `timeline-final.json` and `run_final.py` preserve the final implementation, binary, measurements, timeline and reproduction procedure. The final performance repeat uses no experiment flags. No PR has been submitted and dependency pins are unchanged.

Raw receipts: `/Users/robknight/Projects/0xPARC/do/.benchmark-artifacts/gpu-hybrid-gather-ordering-2026-11-01/`.
