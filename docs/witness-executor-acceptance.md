# Witness executor completion: acceptance report

25 October 2026. This closes the bounded completion plan in [the experiment log](witness-executor-experiments.md#completion-plan-agreed-on-25-october-2026). The consumer contract is documented in [Preparing fast witness generation](witness-executor.md).

## Comparison and scope

The completion baseline is plonky2 `224296db`, after typed configuration, eager preparation and managed caching were implemented, and before the final execution/startup optimizations. It already contains the earlier SIMD, fused arithmetic, direct binding and optional buffer-pool work. These measurements are incremental to that baseline; earlier campaign percentages must not be added to them.

Ordinary live measurements use native GPU proving, eight Rayon threads, recommended default executor settings, explicit preparation and no diagnostic or benchmark feature. They measure full witness generation and a verified MainPod proof, rather than summing isolated kernel estimates. Process medians discard eight warm-ups. Exact-state replays are separate execution-only diagnostics and check every state slot against the checked executor.

## Retained changes

| Area | Decision |
| --- | --- |
| Consumer lifecycle | Typed per-program configuration, explicit preparation and readiness/capability reporting. POD2 delegates executor storage and preparation to plonky2. |
| Startup | Descriptor-only semantic hashing, direct compact decoding, parallel checksums, shared metadata and validated installation tokens. |
| Sampling | Chunk-local CSPRNG work and certified contiguous random-output copies. |
| State | Sparse readiness on eligible graphs; opt-in bounded reuse of wiped buffers. |
| Gathering | Tiled CPU gather remains the measured default. Explicit GPU gather honors CPU resource ownership. |
| Scheduling | Prepared chunk dependencies with inline ready-successor continuations; layer scheduling remains available. |
| Hardware | Runtime AVX-512 witness dispatch for portable x86 binaries, existing native specialization for AVX-512-targeted builds, scalar fallback elsewhere. |

No circuit-specific generated Rust, external compiler or undocumented MainPod tuning is required. Shared pools, memory budgets, fallback policy and comparison controls have typed APIs. Environment defaults, including `PLONKY2_EXECUTOR_CACHE_DIR`, work through POD2 without a wrapper field per option.

## M4 startup and execution

The final M4 live comparison verified 160 MainPod proofs in four alternating processes. Full witness medians were 21.60/24.25 ms for the baseline and 14.90/15.65 ms for the retained implementation, roughly 30–35% lower. Operation execution was 12.36/13.86 ms versus 7.78/8.17 ms. Total proof time drifted substantially across the run (about 0.7–1.0 seconds); this comparison does not establish a reliable M4 total-proof improvement.

A separate public-API startup comparison loaded MainPod's 18,763,951-byte program 20 times per process, compiled it six times, and verified a GPU proof afterward. These are warm-filesystem measurements, not cold-storage/application startup claims.

| MainPod preparation span | Baseline process medians | Retained process medians |
| --- | --- | --- |
| Direct validated decoding | 108.19 / 109.05 ms | 52.68 / 52.76 ms |
| Decode through host readiness | 163.18 / 163.71 ms | 116.26 / 117.96 ms |
| Managed disk-cache preparation | 174.34 / 181.22 ms | 120.69 / 120.97 ms |
| Compile through host readiness | 678.17 / 627.00 ms | 684.67 / 616.73 ms |

Dependency metadata adds about 8–10 ms of eager host work on MainPod, included above. Faster decoding outweighs it on the load path; fresh compilation is broadly neutral. Cache identity calculation, synchronization and filesystem work remain included in managed preparation. First proof still includes GPU/prover warm-up, which is not executor preparation.

With warm dependencies, eight build jobs, native targeting and incremental compilation disabled, the final M4 plonky2 library-unit builds were 7.30/7.08 seconds for the baseline and 7.18/7.24 seconds for the selected implementation, effectively neutral; wrapper wall time remained about 7.6–7.9 seconds. Compiler CPU time increased about 8%. An earlier matched pass measured about 2% more library-unit time. These are library rebuilds, not clean dependency/workspace builds.

## Xeon / NVIDIA final acceptance

The final machine was a Secure Cloud Xeon Gold 6342 with an NVIDIA A40, driver 595.91.07 and Vulkan 1.4.329. Eight physical cores (0–7 on NUMA node 0) were pinned; the container CPU quota was 7.65 cores. Rust was nightly-2026-10-08, rustc 1.101.0-nightly, LLVM 23.1.3. The four ordinary-build processes verified 160 MainPod proofs in baseline/selected/selected/baseline order.

| Phase | Baseline process medians | Selected process medians | Reduction in mean process median |
| --- | --- | --- | --- |
| Sampling | 7.12 / 7.08 ms | 5.16 / 4.56 ms | 32% |
| Operation execution | 43.05 / 43.16 ms | 26.72 / 25.30 ms | 40% |
| Gathering | 9.58 / 9.68 ms | 7.63 / 6.75 ms | 25% |
| Full witness generation | 62.25 / 62.10 ms | 41.55 / 38.30 ms | 36% |
| Verified total proof | 303.70 / 288.22 ms | 254.55 / 255.89 ms | 14% |

The total-proof baseline drifted by about 5% between the opening and closing processes. The gain was positive against both baselines, but 14% is this cohort's result rather than a guaranteed application speedup. Earlier NVIDIA candidate cohorts measured about 31% full-witness and 8% proof improvement; those used a different runtime-kernel selection and must not be treated as an additional gain. Use the final source and four-process table above for the retained implementation.

The runtime-dispatch investigation separately found a roughly 26% MainPod execution improvement in portable x86 builds after correcting outlined helper lowering. Native builds keep their existing specialization because replacing it with that isolated implementation regressed the bounded replay comparison. Both final build modes passed all eight fixture replays (512 exact executions), both trace tests and all 33 witness tests. Final selected GPU checks verified 13 additional MainPod proofs, including concurrent workers, and the every-wire gather test verified eight proofs. After optional-feature guards and lint-equivalent binding cleanup, x86 `no_std`/std-only checks, trace tests, lint checks and two further portable NVIDIA MainPod proofs passed. The timed and final source forms are preserved separately.

Native x86 library-unit build times were 14.28/12.76 seconds for the baseline and 13.34/13.13 seconds for the selected implementation; there is no material wall-time regression. Compiler CPU time increased about 6%. These matched nonincremental builds use warm dependencies and do not measure fresh dependency installation or a complete application link.

## Preparation and first proof remain separate costs

The ordinary live harness also records complete `mainpod::prepare` and its first verified proof, before discarding warm-ups for the tables above:

| Machine / span | Baseline processes | Selected processes |
| --- | --- | --- |
| M4 complete POD2 preparation | 9.38 / 8.65 s | 9.31 / 9.10 s |
| M4 first proof after preparation | 845 / 810 ms | 826 / 852 ms |
| Xeon/A40 complete POD2 preparation | 30.02 / 23.01 s | 25.01 / 21.19 s |
| Xeon/A40 first proof after preparation | 429 / 286 ms | 247 / 256 ms |

Complete POD2 preparation includes building/loading circuit data, padding proofs and GPU initialization; it is much broader than executor preparation. These live processes reused their task cache directories and do not constitute a controlled cold-application startup comparison. They establish that preparation and first-proof costs were observed separately, not that the executor removes GPU warm-up or guarantees a particular startup percentage. The repeated executor-only compile/load matrix above is the startup optimization comparison.

## Where the remaining time goes

An optional diagnostic build profiles the selected native x86 replay under the dependency scheduler. Across two processes after warm-up, Poseidon accounts for about 63% of attributed operation CPU time, ordinary plus fused arithmetic about 14%, and elliptic-curve operations about 12%. SIMD input packing/arithmetic/output handling account for roughly 3.8/41.3/14.5 CPU milliseconds; these nested totals cover the SIMD portion and are already included in the whole Poseidon total. Small scalar groups/tails and surrounding kernel work explain part of the difference.

Instrumentation increased execution time by about 25% versus the same diagnostic binary with counters disabled (18.1–19.9 versus 15.1–15.3 ms). This is useful bottleneck attribution, not an uninstrumented speed estimate. CPU totals overlap across workers and do not identify the critical path. The ordinary live phase timings above separately include sampling and gathering. Further substantial gains most plausibly require better hashing or different hardware execution, rather than more arbitrary native arithmetic-region code.

## Rejected candidates

- General mixed-operation regions supported irregular slots and multiple arithmetic descriptor kinds. The compact interpreter regressed MainPod. Generating fixed-address Rust from the same partitions produced 19.2 MB across 12,850 functions, took 172 seconds to build with warm dependencies, and was slower again. No implementation is retained.
- Mixed-region dependency scheduling lost to simpler raw chunk dependencies. The latter uses 8,125 MainPod tasks without a new instruction interpreter.
- A per-worker completion counter did not show a repeatable material improvement after reverse-order M4 and NVIDIA comparisons. Its allocation and thread-index complexity are excluded.
- Poseidon scheduling grains 1, 4, 8, 16 and 32 were screened. Four's small advantage was insufficient to justify another policy; larger grains regressed. Keep eight.
- Fixed-width storage was larger and slower to read/decode than the improved compact format. Alternative layouts and GPU gather did not justify changing the MainPod defaults.
- Static propagation covered only 8 of MainPod's 18,368 Poseidon permutations; no identity copies remained. These are not a substantial remaining inexpensive opportunity.

## Library acceptance and limitations

The final M4 parallel suite passes 33 witness tests; the serial suite passes 31. Coverage includes semantic/cache mismatches, malformed programs, input aliases and conflicts, strict fallback, lifecycle invalidation, concurrent preparation/proofs, independently configured pools, failed-execution cleanup, buffer reuse and maximum chunk bounds. POD2's adapter tests and actual MainPod preparation pass. The real Points example verifies recursive MainPod proofs as well as signed PODs. Native GPU tests compare every uploaded wire with the worklist under CPU/GPU gather and explicit resource pools.

The graph fixture set includes MainPod, mixed operations, empty, deep/wide arithmetic, deep/wide hashes and equality. Portable x86 fallback is also tested under a QEMU Haswell CPU model with AVX2 available and AVX-512 absent: the public example verifies two proofs and the wide-hash fixture matches exactly. This is an emulated fallback check, not a physical AVX2-machine performance result.

The bounded plan does not claim globally optimal witness generation. Full GPU witness execution, an unrestricted JIT, new ARM SIMD arithmetic and universal custom-generator support remain separate projects. Custom generators still need accurate operation descriptions and worklist comparisons. No broadly profitable unimplemented cheap improvement was established by the screened candidates.

## Evidence

Local recipes, source snapshots and raw measurements are under `.benchmark-artifacts/witness-completion-2026-10-25/`. Important files include `final-m4-summary.json`, `final-startup-summary.json`, `final-lib-build-summary.json`, `final-lib-unit-timings.json`, `runtime-lib-build-summary.json`, `runtime-lib-unit-timings.json`, `points-final-live.log` and the final CPU/consumer test logs. The `nvidia/` subdirectory contains upload manifests, source deltas, remote recipes and retrieved hardware/results. Benchmark-owned snapshot rekeying is confined to these recipes; production cache validation is unchanged.


The retrieved archive is `nvidia/witness-final-results.tar.gz`, SHA-256 `7aca30712ae810a896ec3204f59a1133f477ef6f52af773ce46ffabff050d015`; it was extracted into `nvidia/witness-final-results/` and all 216 file checksums verified. `final-native-proof-summary.json`, `final-analysis.json`, `final-replay-summary.json`, `final-lib-build-summary.json`, source deltas and recipes there support the NVIDIA tables. The temporary pod was deleted with a fresh 404 read-back; no persistent volume was created. Local retrieval and cleanup receipts are beside the archive.
