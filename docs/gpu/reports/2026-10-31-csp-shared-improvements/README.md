# Shared CSP / POD2 prover improvements — 31 October 2026

The accepted candidate gives a modest shared improvement: approximately 3–5%
less proving time on most reused hash circuits and 2.5% on POD2. The larger
SHA-2048 case improves by 2.6%. A longer fresh-SHA repeat improves by 3.3%
for proving and 3.7% when prepared-state cleanup is included. Fresh Keccak
was close to the noise level. These are same-device incremental comparisons.

The source is saved on `codex/csp-shared-improvements`, based on core
`85a5624e8fe4e0315fa01b8b8af78a93fc17ebe1`. The measured candidate is
`f676e3eff277cea3e84cb56e3a5cb9bdc926df8e`. The source worktree is
`/private/tmp/csp-opt-20261031/plonky2`; commits remain in the original
plonky2 repository. The CSP submission branch and dependency pins remain at
the previously validated versions; this round has not published a PR.

## Changes

- `3d7ac3e1`: expand the existing small constants in Poseidon's complex MDS
  block, cancelling terms from the general complex-product formula. The
  expressions are equivalent in the wrapping u32 ring. The same routine is
  used by Merkle hashing, proof-of-work, and the GPU quotient's MDS operations.
- `c75c57dd`: prepare independent FRI trees' sparse queries on the existing
  Rayon pool, allowing one device wait to overlap another tree's gather.
  No new pool or thread budget is introduced. Includes shared-circuit proof
  coverage with one-, two-, and four-worker pools.
- `f676e3ef`: submit FRI folds and opening evaluations asynchronously; the
  following cap/coefficient/opening read waits on the dependent submission.
  FRI folds can execute while the host encodes the next tree.

The permutation, circuit definitions, transcript, and proof parameters retain
their existing semantics. No circuit-size or benchmark-name dispatch is added.

## Measurements

Apple M4, 10 CPU / 10 GPU cores, 32 GiB, macOS 15.8.1 (24H32), native Metal
arithmetic. Compiler: nightly-2026-03-14, rustc 1.96.0-nightly
`1d8897a4e`, LLVM 22.1.0. Release build, debug information level 1, with
`gpu-profiling` compiled but timestamp recording disabled for timings.
Hardware GPU and the installed witness executor were required. No inherited
PLONKY2 or RAYON overrides were used apart from the recorded Metal/executor
settings. Every measured proof was verified separately.

The probe directly imports the stock CSP circuit sources. Sources:

- CSP: `b68fd117f4e6de20e5931c29795b91f3a2c98be3`.
- u32: `80b832b5ebb9d2cb6fa259c5a1f99d8028f38683`.
- POD2: `52e14a99be6b2ec82b102b057095ca259a57312a`, default MainPod circuit,
  empty builder, prepared once. Core dependencies were unified to the tested
  worktree through Cargo patches.

**Reused circuits.** ABBA process order, nine proofs/process, discard two
warmups, median of the two process medians per variant (14 retained samples
per variant per case). Preparation, verification and destruction of prepared
state are outside these proving times.

| Case | Baseline ms | Candidate ms | Time reduction |
| --- | ---: | ---: | ---: |
| poseidon16 | 111.83 | 105.75 | 5.4% |
| sha128 | 233.94 | 222.90 | 4.7% |
| keccak128 | 217.44 | 207.30 | 4.7% |
| pod2 | 581.22 | 566.79 | 2.5% |

**Larger reused circuits and an independent POD2 repeat.** Same ABBA order;
seven proofs/process for SHA/Keccak-512, five for SHA-2048 and nine for POD2,
always discard the first two.

| Case | Baseline ms | Candidate ms | Time reduction |
| --- | ---: | ---: | ---: |
| sha512 | 451.30 | 435.23 | 3.6% |
| keccak512 | 419.69 | 401.61 | 4.3% |
| sha2048 | 1815.21 | 1767.45 | 2.6% |
| pod2 | 578.44 | 564.08 | 2.5% |

**Fresh circuits.** A new circuit and installed executor are prepared before
each proof; preparation and verification are measured separately. The first
five-proof ABBA run gave Poseidon-16 127.03 → 119.15 ms (6.2%) and
Keccak-128 233.73 → 232.17 ms (0.7%, inconclusive). SHA-128 was noisy and
slightly slower (247.74 → 253.33 ms, -2.3%), so it was repeated rather than
accepted as a regression or omitted.

The SHA repeat compared baseline, MDS-only, and the combined candidate in a
balanced 12-process order. Each variant had four processes of nine fresh
proofs; two warmups were discarded, giving 28 retained samples/variant.
The following values are medians of the four process medians:

| SHA-128 fresh | Baseline ms | Combined ms | Time reduction |
| --- | ---: | ---: | ---: |
| Proving | 252.16 | 243.75 | 3.3% |
| Proving + prepared-state cleanup | 286.67 | 276.00 | 3.7% |

MDS-only in this repeat was +0.4% for proving and +1.2% with cleanup. The
repeat supports keeping the combined candidate; fresh-circuit allocator and
host-cache variation still limit precision. These probe measurements do not
replace the prior complete CSP harness results or establish a new M1 speedup.
Only M4 performance was measured.

**Device evidence.** A separate timestamp run confirms less GPU work and
shorter idle gaps. Medians over the last three main proofs (initialization
and first proof excluded):

| Workload | Wires leaf hashing ms, old → new | GPU idle ms, old → new |
| --- | ---: | ---: |
| Poseidon-16 | 44.72 → 42.88 | 8.24 → 5.48 |
| SHA-128 | 88.77 → 85.07 | 10.89 → 7.50 |
| POD2 | 176.76 → 169.35 | 16.34 → 11.11 |

## Validation

- 448 measured/profiled proofs verified across the final comparison campaigns,
  including the MDS-only control in the fresh-SHA repeat.
- GPU suite: 56 passed, 24 intentionally ignored, no failures, using the
  expected default configuration on the physical Apple M4.
- Native and portable Poseidon permutation CPU differential tests passed.
- Isolated unified-resident and emulated-discrete-memory proof/shutdown tests
  passed. The latter is functional policy coverage on M4, not a discrete-GPU
  performance measurement.
- The new concurrent shared-circuit proof test passed with Rayon pools of
  one, two and four workers. This checks correctness and completion, not
  concurrent POD2 throughput.
- Serial build (`--no-default-features --features std,gpu`) passed.
- Clippy completed; existing warnings remain outside the changed code.
  New-test rustfmt and `git diff --check` passed.

An initial suite run forced the Metal backend in the environment, which
violated an existing test's expectation of default settings and poisoned its
shared test mutex. The default-config rerun above passed. The failed log is
retained for transparency.

## Experiments discarded

A three-product native squaring formula passed arithmetic checks but regressed
most complete proofs by roughly 2–5%. Splitting leaf absorption into full
blocks and a tail did not improve timings. Both patches were discarded.
An early overlapping-query comparison had compilation activity during its
last POD2 processes; its POD2 numbers were excluded from acceptance. The
clean combined runs above supersede that diagnostic result.

## Artifacts and repeat commands

Workspace artifact directory: `.benchmark-artifacts/csp-profile-2026-10-31/`.
It contains the standalone Cargo probe, source/lockfile, immutable binaries,
raw logs, per-proof JSON, binary hashes and source revisions, rejected patches,
and `accepted-changes.patch`. `NOTES.txt` records the investigation history.

Run from the artifact directory, with Metal access:

```sh
python3 compare.py combined
python3 compare-extended.py combined
python3 compare-fresh-sha.py combined
python3 profile-pair.py
```

Scripts use the archived `binaries/` and verify every proof. They overwrite
their output directories; preserve previous results before repeating. The
first two use ABBA; the fresh-SHA script additionally compares MDS-only.
`final-metadata.json` identifies the measured binaries and source revisions.
Timestamp details are in `profile-combined/stages.json`. Raw result directories
are `compare-combined-vs-baseline`, `extended-combined-vs-baseline`, and
`fresh-sha-repeat-combined-vs-baseline`.
