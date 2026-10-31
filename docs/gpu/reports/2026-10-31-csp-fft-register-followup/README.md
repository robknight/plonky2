# Register-blocked forward FFT and reading follow-up — 31 October 2026

Retained a general forward-NTT improvement for native Apple Metal: 128-thread
workgroups keep four Goldilocks values in registers across two radix-2 stages.
On this M4 the final committed build reduces hash proving time by about 1–3%
and POD2 by about 1–1.5% incrementally. Its main forward FFT is about 16% faster.
Inverse transforms retain their previous kernel. All candidate knobs and
rejected experiments were removed from the production source.

Source commit: `4ffd4c6e21d7b718cde941c3504c73ff4d94e489`, isolated branch
`codex/csp-shared-improvements`. Incremental baseline: `b9bc25d2`, whose code is
`0c6ff330`. The original GPU fork baseline is `85a5624e`. The user's primary
branches, CSP source pins and PR submission were left unchanged.

## Implementation and scope

The four-value packet executes the same two stages, with the same field
multiplications, additions and twiddles, while removing their intermediate
shared-memory reads/writes and barrier. An odd remaining stage uses radix-2.
The tile stays at 1024 elements and the existing prefetch count stays at eight
on this unified-memory adapter. Zero-padding skips, coset shifts, inverse
normalization and proof/circuit formats are unchanged.

Automatic selection depends on adapter/backend/arithmetic and transform
direction. Native Apple Metal selects the new kernel for pipelined forward
transforms. Inverse, ordinary nonpipelined, compact-witness-input and other
adapter/arithmetic paths retain their previous kernels. The additional pipeline
is compiled lazily and cached in a OnceLock. There is no benchmark/circuit-name
selection, autotuning or new public setting.

The shader generator now shares stage templates between its two entry points.
A byte comparison during extraction verified that the legacy FFT kernel source
is reconstructed exactly; `legacy-fft-source-check.json` records its hash.
The committed cleanup also avoids constructing an unused alternate pipeline
for compact witness input. That path's actual kernel was always unchanged.

## Method

Apple M4, 10 CPU / 10 GPU cores, 32 GiB, macOS 15.8.1. Rust
nightly-2026-03-14; release with debug information level 1. The probe imports
stock CSP circuit definitions at `b68fd117`, uses u32 `80b832b5` and POD2
`52e14a99`, and patches all core dependencies to one worktree. The original
probe manifest/source path was retained throughout. POD2 uses its default
MainPod parameters and an empty builder.

Every measurement requires hardware Metal and a compiled witness program with
fallback set to Error. Every measured proof is verified. Inherited PLONKY2 and
RAYON overrides are cleared. No build/test/GPU profiling ran concurrently with
comparison timings. Timestamp queries are disabled for timing comparisons;
profiles are separate diagnostic runs.

Comparisons use balanced process order, normally ABBA. Discard the first two
proofs per process, take each process's median, then the median of process
medians. Final committed warm runs use nine proofs/process for hashes and 21
for POD2. The longer POD2 acceptance repeat uses 31 proofs/process. Fresh
Poseidon acceptance uses eight balanced processes of nine proofs each.

Warm mode reuses a prepared circuit. Fresh mode constructs and installs an
executor for each proof, recording preparation and cleanup separately; the
table measures proving only. Fresh circuits are not first-ever driver-cache
or process-start measurements. Initial shader compilation can cost additional
time; no claim is made that setup is free.

## Incremental results

The rebuilt committed source, with automatic selection and no experimental
flags, compared with the previous accepted GPU code:

| Case | Baseline ms | Candidate ms | Time reduction |
| --- | ---: | ---: | ---: |
| poseidon16 | 102.399 | 100.150 | +2.20% |
| sha128 | 216.866 | 213.148 | +1.71% |
| keccak128 | 201.262 | 195.509 | +2.86% |
| pod2 | 558.095 | 551.939 | +1.10% |

The initial automatic build independently gave +2.79% Poseidon, +0.93% SHA-128,
+1.29% Keccak-128 and -0.43% POD2 in a short run. Keep that negative POD2 point:
recurring query-gather stalls made short-process medians unstable. Longer
balanced acceptance checks gave:

| Case | Baseline ms | Candidate ms | Time reduction |
| --- | ---: | ---: | ---: |
| poseidon16 | 119.501 | 118.931 | +0.48% |
| pod2 | 562.126 | 553.397 | +1.55% |

The longer POD2 repeat's mean, including every retained spike, was
594.811 → 591.393 ms, a 0.57% reduction. Median improvement is about 1.55%.
Fresh Poseidon was -2.89% in the initial short automatic comparison and +0.48%
in the longer repeat; describe it as roughly flat, not a demonstrated gain.
The prototype's +7.21% fresh Poseidon point is also too noisy to generalize.

Larger/fresh cases from the initial automatic build (same emitted FFT shaders
and measured benchmark paths as the committed implementation):

| Case | Baseline ms | Candidate ms | Time reduction |
| --- | ---: | ---: | ---: |
| sha512 | 425.399 | 419.406 | +1.41% |
| keccak512 | 392.472 | 384.729 | +1.97% |
| sha2048 | 1735.621 | 1698.286 | +2.15% |
| pod2 | 554.071 | 551.112 | +0.53% |
| poseidon16 | 115.394 | 118.730 | -2.89% |
| keccak128 | 222.339 | 218.128 | +1.89% |

Here the Poseidon and Keccak-128 rows are fresh; all other rows are warm.
The initial executable is archived as `binaries/final-initial`; `binaries/final`
was rebuilt from the committed source. Each raw run records its binary hash.
The last source cleanup only removes unused pipeline construction on compact
witness input; none of these cases uses that input path.

## Direct total gain across all optimization rounds

This is a new, contemporary comparison against the archived original GPU fork
build `85a5624e`, rather than multiplying earlier percentage estimates. All
other probe dependencies and settings match. The candidate here is the
initial automatic executable described above; the committed executable was
separately retimed in the incremental table.

| Case | Baseline ms | Candidate ms | Time reduction |
| --- | ---: | ---: | ---: |
| poseidon16 | 112.456 | 100.747 | +10.41% |
| sha128 | 231.883 | 212.208 | +8.48% |
| keccak128 | 214.542 | 196.260 | +8.52% |
| pod2 | 577.503 | 544.255 | +5.76% |

These are time reductions against our earlier GPU fork, not speedup factors
against stock CPU Plonky2, Ethproofs, M1 hardware or other provers. Small
percentages and absolute timings remain sensitive to host/device activity.

## Device evidence and latency spikes

Separate profiles show the median main wires forward FFT at 2^19 × 135
changing from 48.278 to 40.525 ms, about 16.1% less. Its inverse stays about
6.5 ms. Poseidon's 2^17 × 135 forward FFT changes from 8.687 to 7.231 ms.
The main FFT is a limited fraction of whole proving time, consistent with
about 1–2% end-to-end POD2 improvement.

Both the original fork and the prior/modified builds show occasional roughly
150–170 ms proof-time spikes. Device profiles place the extra time in
`gather column leaves`: examples add 140 or 184 ms there, while quotient and
FFT times remain close. These stalls predate this round's FFT change.
A read-only system check showed 63% memory free and no obvious shortage;
it did not measure GPU utilization or establish a cause. No other workloads
were stopped. All raw samples, including spikes, remain in the results.
This is a useful further profiling target; it is not attributed to external
contention or to parallel query preparation without evidence.

## Experiments rejected or screened out

- **2048-point tiles:** the user's recollection was correct; `docs/gpu.md`
  already records failure. Retesting the existing plain/nonprefetched tile
  benchmark at 2^19 × 135 gave 52.730 ms for 1024/256, 490.514 for 2048/256,
  and 498.407 for 2048/512. The 2048/256 CPU differential passed; 512-thread
  output was not separately checked in this timing screen. These are not the
  production prefetch-eight FFT timings. Raw results are in the sibling
  `gpu-reading-2026-10-31` artifact directory.
- **Radix-4 everywhere with 256 threads:** modest warm gains, but fresh Poseidon
  regressed 9.2% in its extended run. Not retained. The accepted design changes
  workgroup size and restricts selection to forward pipelined transforms;
  its longer fresh repeat was roughly flat.
- **Coalescing guard:** fusing only after five contiguous address bits did not
  improve the existing configuration reliably. The accepted packets use all
  eligible pairs.
- **Reduced-multiplication radix-4:** correct three-general-product formulation
  with multiplication by Goldilocks' fourth root 2^48 implemented as a shift
  and reduction. Native/portable CPU comparisons and all GPU tests passed,
  but POD2 was -0.33% and Poseidon approximately flat in its sweep. Rejected.
- **Radix-8 packets:** CPU exactness passed in native and portable arithmetic,
  including 64/128/256-thread configurations, plus the full GPU suite at 128.
  In microbenchmarks, pipelined forward LDE was 48.51/45.08/49.84 ms at
  256/128/64 threads, versus controls 48.38 and 47.24. Inverse FFTs worsened.
  Radix-4/128 forward was stronger at 42.18 ms. Radix-8 was screened out;
  no end-to-end radix-8 improvement is claimed.
- **128-thread radix-2 forward:** useful control. In the same executable it
  gave +1.15% warm Poseidon, +0.22% SHA-128, +1.57% POD2, and -1.16% fresh
  Poseidon. Radix-4 forward gave +2.10%, +0.55%, +2.04%, and +7.21%
  respectively in that sweep. The fresh points are noisy; final acceptance
  uses longer independent comparisons.
- **Quotient workgroups 32/128/256:** all correctness suites passed. Apparent
  sub-percent POD2 gains mostly matched the disabled same-binary control;
  Poseidon was slightly worse. Retained 64 threads.
- **Partial-round unroll factors 2 and 11:** correct but slower than the
  existing one-round loop. Poseidon regressed 1.3% and 4.5%, POD2 1.8% and
  3.2%, respectively. Retained the previously accepted loop.

## What the supplied links contribute

| Source | Application here |
| --- | --- |
| [Apple Silicon FFT paper](https://arxiv.org/html/2603.27569v1) | Register blocking and memory layout are actionable; they motivated this round. Its FP32 complex FFT, vDSP comparisons and reported GFLOPS do not predict exact Goldilocks NTT or proof speedups. Its own shuffle variant lost, so fewer barriers alone are insufficient. |
| [Apple ML kernels](https://medium.com/@srivarshan02/writing-fast-ml-kernels-on-apple-silicon-123152624078) | Useful occupancy, coalescing and roofline guidance. Neighbouring scalar loads can already coalesce; floating-point reductions/matrix units are not drop-in prime-field arithmetic. |
| [Blackwell CUDA lessons](https://jared-hpc.com/posts/cuda-optimization-lessons-blackwell/) | Register-blocked NTTs, spill measurement, device residency and overlap are relevant principles. Most residency/queue overlap is already present here; PTX-specific operations are unavailable directly through our Metal/WGSL path. |
| [GPU Halo2](https://tribuste.com/assets/gpu-halo2-acc.pdf) | Gate evaluation, slicing and lifetime reuse are relevant. Our quotient already has DAG/liveness allocation and CPU/GPU overlap. Slicing mainly helps larger/discrete-memory cases; bounded compilation of hot quotient blocks remains a substantial future candidate. |
| [zkSecurity cryptography optimization](https://blog.zksecurity.xyz/posts/optimizing-cryptography-with-ai/) | Keep exact reference tests and transcript/commitment anchors alongside proof verification. SIMD/cache/pipeline work applies. Hardware SHA does not remove SHA-circuit witness work or permit changing Poseidon commitments without changing the protocol. |
| [NVIDIA carryless multiplication](https://developer.nvidia.com/blog/building-faster-cryptography-with-carryless-multiplication-in-nvidia-cuda-13-3/) | CLMAD targets binary-field GF(2^n) arithmetic. This prover uses a Goldilocks prime field and Metal, so it offers no direct multiplication shortcut here. It is relevant to a future CUDA binary-field backend. |

No author implementation was copied. The retained kernel uses standard exact
radix-2 butterflies grouped into register packets.

## Validation and handoff

Across these comparison and profiling series, 1,948 measured proofs
were verified; unit-test proofs are separate from that count.

The committed build passes all 57 GPU tests (24 diagnostic tests ignored),
including the new native/portable exact CPU comparisons over small and large
plain/coset padded transforms, odd stage counts and multiple tile-loop counts.
Existing inverse, proof/circuit compatibility and concurrent-query tests pass.
Unified-resident and emulated discrete-memory proof/shutdown tests pass in
isolated processes. The portable legacy FFT differential also passes.

`cargo check --no-default-features --features std,gpu`, library Clippy with
`gpu`, changed-file rustfmt checks and `git diff --check` pass. Clippy retains
seven pre-existing warnings outside the modified FFT code. No new NVIDIA/M1
hardware test was run; other paths retain their legacy FFT source.

Raw scripts, logs, comparison JSON, profiles and executable hashes are under
workspace `.benchmark-artifacts/gpu-fft-register-2026-10-31/`.
`final-metadata.json` records revisions, binary hashes and verification counts.
`accepted-changes.patch` contains this round; `accepted-stack.patch` contains
the whole accepted stack against `85a5624e`. Rejected prototypes are archived
separately. `REVIEW.zip` packages the report, patches, scripts and raw receipts;
the larger executable binaries remain alongside it.
