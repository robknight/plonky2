# MULC ADD quotient fusion experiment

**Decision: do not enable by default yet.** The fused opcode is correct and makes the native MainPod quotient kernel about 2.1% faster, but complete-proof acceptance is mixed. The original process comparison was roughly flat for POD2, a closer interleaved comparison suggested about 0.5% improvement, and the final separate-process repeat was slower. The source is restored to the accepted gather-ordering change `2ec41d36`; the complete candidate is archived for further investigation.

## Compiler transformation

The new opcode 28 computes `dst = mul_small(y, immediate) + z`, using the same field routines and arithmetic order as the original two instructions. A backwards liveness pass permits fusion only if the multiply temporary is dead after the add or overwritten by that add. It accounts for scalar reads, wire masks, in-place MDS, packed MDS and packed quintic products. Adds consuming the temporary twice are left alone.

The pass runs after literal preloads are resolved and before register reallocation and MDS table offsets are assigned. The final candidate selects it automatically only for native Apple Metal; other backends and arithmetic flavors retain their previous bytecode. This selection is an adapter property, not a benchmark or circuit-name choice. The final candidate removes the experimental production override.

MainPod has 314 adjacent MULC/ADD pairs, of which **299 can be fused safely**. The program shrinks from 4,911 to 4,612 executable instructions while retaining its 128-register capacity. The final native dump contains 299 opcode-28 instructions; the final portable dump contains none and is byte-identical to the archived original MainPod dump (`legacy-bytecode-check.json`).

## Correctness validation

Three new tests cover 4,608 combinations of field boundary values, constant multipliers, operand masks, add order and destination aliases; preservation of live temporaries including packed reads; and fusion through register reallocation with the optimization enabled and disabled. Both native and portable full-kernel outputs match the original across all 1,048,576 output field values.

The initial GPU suite passed 75 tests with 25 ignored. The final suite, including the additional register-reallocation test, passed **76 tests with 25 ignored**, separately under native and forced-portable arithmetic. Eight concurrent hybrid proofs also verified. Release serial GPU compilation passed. Clippy reported the same seven existing warnings, all outside the changed code. The initial test build reported one unused test import, removed before the final clean test build.

## Full kernel measurements

The benchmark executes the real archived MainPod program on 2^19 points with real field arithmetic, seeded canonical inputs and the original buffer shapes. The fused variant keeps the same register capacity and rebases packed MDS constant-table addresses after removing instructions. Every output is compared bit-for-bit with the original before timing. Six samples per variant alternate in ABBA order; each sample is the best of three dispatches. Kernels and pipelines are warmed before comparison.

| Arithmetic on M4 | Original median ms | Fused median ms | Time reduction |
| --- | ---: | ---: | ---: |
| Native | 199.294 | 195.100 | 2.10% |
| Portable | 287.129 | 283.478 | 1.27% |

Native samples were tightly grouped. Portable samples had substantial variation, so its point estimate does not establish a reliable gain. Portable performance here is not an NVIDIA measurement. Raw samples are in `full-kernel-native.log` and `full-kernel-portable.log`.

## Initial complete proof comparison

Hardware: Apple M4 with 10 GPU cores and 32 GiB, macOS 27.2 (26B123), Xcode 26.3, Rust nightly-2026-03-14. The original probe manifest/source path, stock CSP circuits, compiled u32 witness executor and default empty MainPod are retained. Every proof requires hardware Metal, disallows software adapters and uses witness fallback Error. All inherited PLONKY2/RAYON/METAL_CAPTURE overrides are cleared.

The initial comparison uses the same binary with only `PLONKY2_GPU_MULC_ADD_FUSION=1` toggled, in baseline/fused/fused/baseline process order. POD2 uses 21 proofs/process, hashes 15; discard the first two in each process. Means below average equal-sized retained samples. Every measured proof is verified. No build, other test or profiling experiment ran concurrently with comparisons.

| Case | Baseline mean ms | Fused mean ms | Time reduction |
| --- | ---: | ---: | ---: |
| keccak128 | 202.573 | 206.203 | -1.79% |
| pod2 | 552.731 | 552.251 | +0.09% |
| poseidon16 | 106.376 | 105.629 | +0.70% |
| sha128 | 226.109 | 220.107 | +2.65% |

These samples show substantial between-process drift. In particular, the POD2 means were effectively flat and the Keccak estimate moved the wrong way. The full kernel's 4.2 ms saving is a small fraction of complete proof time.

## Interleaved complete proofs

A diagnostic version of the probe prepares two independent processes, then blocks each on stdin before its proof timer starts. One holds the original program, the other the fused program. The parent requests and verifies one proof at a time; no two measured proofs overlap. Four warmups/process are followed by 16 blocks of baseline/fused/fused/baseline individual proofs, giving 32 retained proofs per variant per case. Both contexts remain resident, so this is a complementary diagnostic rather than an exact replacement for the single-process production workload.

| Case | Baseline mean ms | Fused mean ms | Time reduction |
| --- | ---: | ---: | ---: |
| pod2 | 549.964 | 547.130 | +0.52% |
| sha128 | 217.946 | 216.651 | +0.59% |
| keccak128 | 197.982 | 197.356 | +0.32% |

The paired POD2 block difference averaged 2.834 ms with a standard error of 1.306 ms across 16 blocks; 11 of the 16 blocks favored fusion. SHA and Keccak block differences were smaller relative to their variability. These are modest point estimates, not evidence of a broad multi-percent proof improvement.

`interleave.py`, `probe-stepped.rs` and the stdout/stderr logs preserve the controller and measurements. The original probe source was backed up and restored byte-for-byte after this diagnostic.

## Final automatic selection acceptance

The final source has native Apple Metal automatic selection and no production experiment flag. It was rebuilt with the original probe source, then compared against the archived prototype binary with fusion disabled, in fused/baseline/baseline/fused process order. POD2 uses 31 proofs/process, fresh Poseidon nine and larger SHA512 eleven; discard the first two.

| Case | Baseline mean ms | Fused mean ms | Time reduction |
| --- | ---: | ---: | ---: |
| pod2 | 589.293 | 620.342 | -5.27% |
| poseidon16 | 109.242 | 113.015 | -3.45% |
| sha512 | 571.301 | 572.307 | -0.18% |

These negative results are retained rather than dismissed. The final POD2 process means were 549.542 ms fused, 575.586 and 603.000 ms baseline, then 691.143 ms fused. Within that last fused process, individual proof times moved from roughly 542–582 ms early to peaks above 850 ms. SHA512 process means ranged from 530.694 to 611.909 ms. Such variation limits attribution, but the acceptance check still fails to demonstrate a consistent end-to-end improvement, especially for fresh setup and larger workloads.

A CPU-load snapshot after the final run showed OrbStack Helper at 13% CPU and no recorded thermal/performance warning from `pmset`; it does not identify the cause of the preceding drift. Background work was left running. A future quiet comparison could resolve the small effect more clearly.

Given the requirement to avoid workload tradeoffs, the accepted source stays unchanged. The kernel result is promising, but it is insufficient on its own to enable this candidate.

## Artifacts and reproduction

**686 measured/profiled proofs verified**: 264 initial comparisons, 216 interleaved proofs including warmups, 204 final comparisons and two final program-dump proofs. Correctness-suite internal proofs and the eight concurrency proofs are additional and are not counted as benchmark measurements.

- `candidate-complete.patch` and `final.patch` contain the tested automatic-selection implementation and tests against `2ec41d36`.
- `prototype.patch` and `binaries/prototype` contain the opt-in version used for same-binary controls.
- `binaries/final` and `binaries/final-tests` contain the final automatic-selection proof probe and GPU tests.
- `binaries/stepped`, `probe-stepped.rs` and `interleave.py` contain the interleaved diagnostic.
- `warm-summary.json`, `interleaved-summary.json`, and `final-summary.json` retain process and proof measurements, including negative results.
- `final-native.ops`, `final-portable.ops` and `legacy-bytecode-check.json` verify automatic selection and legacy preservation.
- Build/check/test logs, binary hashes and environment metadata remain alongside the raw logs.

For kernel reproduction, apply the candidate patch, build release lib tests with `gpu-profiling`, then run the ignored exact test `gpu::tests::benchmarks::bench_quotient_program`, setting `PLONKY2_GPU_PROGRAM` to the archived MainPod dump and `PLONKY2_GPU_COMPARE_MULC_ADD=1`. For the opt-in archived proof binary, toggle only `PLONKY2_GPU_MULC_ADD_FUSION=1`; the final archived binary selects native Apple Metal automatically. `run_final.py` repeats the final comparisons and program checks.

The user's working branches, dependency pins and held PR are unchanged. The original benchmark probe source is restored. No rejected default or new environment setting remains in production source.

Raw receipts: `/Users/robknight/Projects/0xPARC/do/.benchmark-artifacts/gpu-mulc-add-fusion-2026-11-01/`.
