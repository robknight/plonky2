# Follow-up GPU hash optimization — 31 October 2026

Retained one further improvement: loop the 22 Poseidon partial rounds in native
Apple Metal kernels, with their constants in a small read-only table. The eight
full rounds remain unrolled. This gives a few percent less proving time for
hash circuits and about 1–1.5% on POD2, incrementally over the previous accepted
optimization round. Larger hash cases improve by 1.7–3.0%. Fresh Poseidon and
Keccak repeats also improve by about 1.8% and 2.5% respectively.

Measured source: `0c6ff330cbf08c65170bdfc5c3e5d55fd78e5e87`, branch
`codex/csp-shared-improvements`. Baseline: `47c0e287`, whose code is the same
as the previously measured `f676e3ef`. See the [previous round](../2026-10-31-csp-shared-improvements/README.md)
for the earlier changes. These are incremental M4 comparisons; no new M1
speedup or comparison against a different prover was measured.

## Implementation

The native Apple Metal context selects the looped permutation for Merkle leaf
hashing, internal nodes, and proof-of-work. Other hardware/compiler/arithmetic
paths retain their existing generated kernels. The choice applies to every
workload using that context; there is no circuit or benchmark-name selection.

The preceding full MDS layer already absorbs round 4's constants. Each of the
22 loop iterations applies the partial S-box and MDS, then absorbs the next
constants, rounds 5 through 26 inclusive. Words 1–11 remain in normalized limbs
through the loop and are recombined before the following full round. Typed
constant emission supports both immediate and table-based constants; the
prototype's string substitution and all unsuccessful experiment controls were
removed. Circuit definitions, hash outputs, transcript, proof parameters, and
witness execution retain their existing semantics.

Generated native Metal source for the hashing module shrinks from **170,555 to
96,391 bytes (43.5%)**. This is generated source size, not a measurement of
machine-code size or instruction-cache misses. Reducing the instruction working
set motivated the experiment; direct device timings establish the speedup.

## Method

Apple M4, 10 CPU / 10 GPU cores, 32 GiB, macOS 15.8.1. Compiler
nightly-2026-03-14, release with debug information level 1. `gpu-profiling` is
compiled but timestamp capture is disabled for end-to-end timing runs. Hardware
Metal and installed compiled witness programs are required; every measured
proof is verified separately. Inherited PLONKY2/RAYON overrides are cleared.
No build, test or other GPU experiment ran concurrently with the comparisons.

The probe directly imports stock CSP circuit definitions at `b68fd117`, with
u32 at `80b832b5`. POD2 is at `52e14a99`, using its default MainPod circuit and
an empty builder. Cargo patches unify all core dependencies to the tested
worktree. Complete revisions, binary hashes, and proof counts are recorded in
`final-metadata.json` in the workspace artifact directory.

Normal comparisons use ABBA process order. Nine proofs/process, first two
discarded, give 14 retained samples/variant; results are medians of the two
process medians. Proof preparation and verification are outside the proving
time. Warm mode reuses a prepared circuit; fresh mode constructs and installs
an executor for each proof, recording preparation and cleanup separately.

## Results

The first looped-kernel prototype gave these incremental time reductions:
Poseidon-16 **2.61%**, SHA-128 **2.99%**, Keccak-128 **2.68%**, POD2 **1.43%**.
The cleaned implementation, with automatic selection and no experimental flags,
was then repeated independently:

| Case | Baseline ms | Candidate ms | Time reduction |
| --- | ---: | ---: | ---: |
| poseidon16 | 108.14 | 101.86 | 5.81% |
| sha128 | 225.66 | 216.12 | 4.23% |
| keccak128 | 205.13 | 201.91 | 1.57% |
| pod2 | 559.18 | 553.42 | 1.03% |

Small differences are sensitive to host activity and proof-of-work randomness.
The direction repeats across all four warm cases; the 5.8% Poseidon point in
this particular run should not be treated as a precise expected improvement.
The first prototype and device traces support describing the hash gain as a
few percent and the POD2 gain as about 1–1.5%.

Larger reused circuits, plus another independent POD2 comparison:

| Case | Baseline ms | Candidate ms | Time reduction |
| --- | ---: | ---: | ---: |
| sha512 | 445.93 | 434.91 | 2.47% |
| keccak512 | 399.95 | 393.24 | 1.68% |
| sha2048 | 1781.12 | 1727.16 | 3.03% |
| pod2 | 564.62 | 556.88 | 1.37% |

SHA/Keccak-512 use seven proofs/process, SHA-2048 five, and POD2 nine, always
discarding the first two warmups. The independent POD2 repeats gave 1.43%,
1.03% and 1.37% incremental reductions.

### Fresh circuits

The initial short five-proof ABBA run was noisy: Poseidon was -4.45%, SHA +1.89%
and Keccak -1.14%. The negative results were retained and prompted longer
repeats of Poseidon and Keccak rather than being omitted or accepted as a
confirmed regression.

The repeats used balanced ABBA/BAAB ordering, four processes per variant and
nine fresh proofs/process. Discarding two warmups leaves 28 samples/variant.
Each value below is the median of four process medians:

| Fresh case / metric | Baseline ms | Candidate ms | Time reduction |
| --- | ---: | ---: | ---: |
| poseidon16 proving | 121.13 | 118.92 | 1.82% |
| poseidon16 proving + cleanup | 135.47 | 132.79 | 1.98% |
| keccak128 proving | 228.20 | 222.43 | 2.53% |
| keccak128 proving + cleanup | 253.93 | 248.62 | 2.09% |

Fresh-circuit allocation and cache effects still limit precision. These probe
measurements do not replace the prior full CSP harness results. Preparation,
verification and cleanup are explicitly separate; the cleanup-inclusive metric
approximates the lifecycle cost included in the CSP proving closure.

### Device-stage evidence

A separate timestamp run records medians over the last three main proofs:

| Stage | Baseline ms | Candidate ms |
| --- | ---: | ---: |
| Poseidon-16: wires leaf hashing | 43.61 | 41.81 |
| Poseidon-16: all internal Merkle levels | 11.20 | 10.04 |
| POD2: wires leaf hashing | 169.53 | 164.61 |
| POD2: all internal Merkle levels | 33.25 | 31.29 |

The internal-level row sums all trees' level passes within each proof before
taking the median. GPU stages confirm that hashing itself improved; changes
in unrelated adaptive quotient timing are not attributed to this optimization.

## Candidates rejected or left unchanged

- **Fuse all Merkle levels in groups of three:** correct, but slower. Internal
  levels rose from 11.23 to 14.25 ms on Poseidon and 34.77 to 43.88 ms on POD2;
  full proofs regressed about 0.8–3.1%. Reduced parallelism outweighed dispatch
  savings.
- **Fuse only small upper levels:** a six-setting sweep initially suggested
  +2.2% on Poseidon for three levels below 1024 nodes. Full four-case ABBA did
  not repeat it: roughly -0.3% to -0.6%. All fusion code was discarded.
- **32/128-thread leaf/node groups:** no reliable shared gain over 64 threads.
  Defaults remain unchanged.
- **Finer quotient register buckets:** POD2 needs 66 allocated registers but
  uses a 128-entry bucket. Correct 80/96-entry variants gave only +0.33% and
  -0.04% on POD2, respectively. The unused space alone did not establish a
  performance cost. Existing allocation remains unchanged.
- **Further MDS common-sum simplification:** initially about +1% in the sweep;
  confirmation gave +0.60% Poseidon, +0.60% SHA, -0.26% Keccak and -0.12% POD2.
  Inconclusive; discarded.
- **FFT prefetch settings 0/4/16 versus default 8:** disabling prefetch regressed
  about 1–2%; alternatives were within about 0.5%. Defaults remain unchanged.
  This tested existing scheduling controls, not a new subgroup FFT algorithm.

Deeper quotient compiler work and new FFT butterfly implementations remain
possible research directions. This round does not establish that every such
optimization has been exhausted.

## Validation and artifacts

The final implementation passed 56 GPU regression tests (24 intentionally
ignored), including CPU permutation/tree differentials, FRI transcript checks,
mixed-gate proofs and concurrent shared-circuit proofs. Both isolated memory
policy/proof/shutdown tests passed. The preserved portable permutation path
passed its CPU differential check; the rolled portable prototype also passed
before limiting automatic selection to the measured native Apple Metal path.
Naga validation covers both rolled and unrolled hash/PoW modules for both
arithmetic flavors. Serial `std,gpu` compilation passed. Clippy completed with
existing warnings outside the change. Shader/PoW rustfmt and diff checks passed.
The profiling dump tool supports `hash-rolled` and `pow-rolled` explicitly.

**1,631 measured/profiled proofs verified across this round**, including controls
and rejected experiments. Of those, **476** belong to final acceptance campaigns
(warm, larger/fresh, fresh repeats, and device profiling), all with the final
implementation and its baseline. Raw per-proof records are preserved.

Workspace artifact directory:
`.benchmark-artifacts/csp-profile-round2-2026-10-31/`. It contains immutable
binaries, source/lockfile snapshots, rejected patches, accepted source patch,
logs, per-proof JSON, metadata, and the investigation's `NOTES.txt`.

Repeat from that artifact directory with Metal access:

```sh
python3 compare.py final
python3 compare-extended.py final
python3 compare-fresh-repeat.py final
python3 profile-pair.py final
```

These commands overwrite their named result directories; preserve existing
results before repeating. Scripts use archived binaries, so temporary build
directory cleanup does not invalidate the recorded executable. Source worktree:
`/private/tmp/csp-opt-20261031/plonky2`; committed history remains in the original
repository. For generated native MSL, run the ignored `dump_hash_msl` test with
`PLONKY2_GPU_MSL_MODULE=hash-rolled` or `pow-rolled` and an explicit
`PLONKY2_GPU_MSL_OUT` path.

The CSP submission branch, dependency pins, and public forks are unchanged.
No PR was published. Integration and a new full submission-harness sweep remain
a separate step once these core changes are adopted.
