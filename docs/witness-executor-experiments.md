# Witness executor optimization experiments and priorities

As of 24 October 2026. The objective is to beat the existing witness executor during execution, with MainPod as the primary workload and a small set of other circuits checking whether the approach generalizes. Program generation should stay quick, and Rust build cost must remain practical. Larger files on disk are acceptable when they buy speed without expensive generation or compilation.

The first batch retains the validated SIMD and arithmetic kernels, direct input binding, and optional wiped-buffer reuse. Pre-resolved arithmetic slots, affine fixed-address Rust, scheduling over existing groups, and safe Poseidon I/O changes did not establish consistent MainPod gains and are excluded from the implementation. The next architectural priority is a general region compiler supporting irregular operands and mixed descriptors, with partitioning and scheduling designed together. The aim remains an automatic compiler with a small set of optional policies.

# What the experiments have established

The original campaign improved placement, bulk output stores, shared kernels, and short native arithmetic chains. On M4, MainPod execution fell from about 12.1 to 8.8 ms and the full witness call from 21.1 to 18.2 ms. Static program embedding and group specialization produced little additional speed for their build and binary cost. These experiments did not test larger native regions with fixed slot addresses.

The subsequent x86 experiment demonstrated a further gain by processing eight independent Poseidon permutations together using AVX-512. The final comparison used a Xeon Gold 6342 with an A40, eight Rayon threads pinned to physical cores, and the same prepared circuit with alternating implementations. Witness execution remained on the CPU; proving used the existing GPU path.

| Implementation | Execute ms | Full witness ms | Prove and verify ms |
| --- | ---: | ---: | ---: |
| Previous optimized scalar kernel bank | 43.929 | 71.650 | 286.485 |
| Eight-way Poseidon with field FFT | 36.887 | 64.500 | 284.477 |
| Eight-way Poseidon with raw integer FFT | 36.849 | 64.950 | 271.102 |

Values are medians of three process medians, with 18 warm observations per implementation. The field FFT version reduced execution time by roughly 16% and full witness time by roughly 10% in this cohort. Its full-proof improvement was much smaller and varied between processes. The raw integer variant had similar execution time and a promising roughly 5% proof reduction in the paired processes; that difference needs longer confirmation before selecting it on proof performance.

These are comparisons against the previous optimized implementation. Results from earlier original-library cohorts should not be combined with these timings to manufacture a single cumulative speedup.

The field FFT implementation is the simplest current integration candidate. It introduces no per-circuit Rust generation. The tested policy uses eight lanes for Poseidon groups of at least 64, with scalar handling for smaller groups and tails. It is a measured policy for this machine, not a finished portable automatic policy.

# Which earlier opportunities remain open

| Opportunity | Status | Consequence |
| --- | --- | --- |
| SIMD across independent Poseidon permutations | Demonstrated on AVX-512 | Keep the gain and use it as the next baseline |
| Larger native regions with fixed slot addresses | Affine prototype tested; broader regions remain open | Support irregular operands and mixed descriptors to improve MainPod coverage |
| Graph regions scheduled with fewer global barriers | Existing-group schedulers tested without a consistent MainPod win | Design coarser partitioning and scheduling together |
| Trustworthy breakdown of the final optimized executor | Stage and worker CPU attribution measured | Poseidon dominates worker CPU work; use uninstrumented controls because profiling adds overhead |

The 97.6% native coverage figure described fused arithmetic operations, not the share of execution time accelerated. Native kernels still obtain operands through runtime target tables and executor access methods. High coverage therefore does not establish that indexing, checks, materialization, or scheduling are already cheap.

Earlier per-kind CPU-time profiling predates the final SIMD compiler-lowering fix. It also does not identify the critical path of live GPU proving. CPU work and wall time must be measured separately.

# Priorities and experiment sequence

| Order | Experiment | Main question | Advance when |
| --- | --- | --- | --- |
| 1 | Profile the final baseline | What consumes CPU work and determines elapsed time? | Measurements explain the remaining execution cost |
| 2 | Fixed-address kernels within existing regions | How much does executor indirection cost? | Direct addressing yields a repeatable gain at acceptable build cost |
| 3 | Larger regions and a region dependency graph | Can locality and fewer barriers outweigh lost parallelism? | Whole execution improves across relevant graph shapes |
| 4 | SIMD input and output layout | How much of Poseidon cost is data movement? | Live witness time improves after gather and store costs |
| 5 | Input preparation, allocation, randomness, and final gather | Can the surrounding witness pipeline become cheaper? | Full witness calls improve without hiding costs elsewhere |
| 6 | Startup and deployment strategy | Which representation best amortizes preparation? | Generation, load, and build costs fit intended reuse |

Orders 2 and 3 are substantial architectural investigations. Their priority should not be displaced merely because incremental kernel tuning is easier. Results from the first profile can change the order of the contained work in orders 4 and 5.

# Experiment 1 Profile the final implementation

Use the final eight-way field FFT candidate as the reference, retaining the previous scalar bank as a control. Measure both isolated replay and live MainPod proving, and report each separately.

Collect coarse spans for input binding, random-value generation, state allocation and initialization, dependency-layer execution, alias and readiness validation, output gather, and witness-state wiping. Within execution, attribute CPU samples to Poseidon arithmetic, operand lookup and input packing, output canonicalization and stores, other generators, and scheduler activity. Record layer widths, task counts, chunk sizes, and the distribution of per-layer completion times to locate stragglers and idle workers.

Prefer sampling and aggregate counters over timing every primitive. Measure instrumentation overhead against uninstrumented execution. Where available, use hardware counters for cache misses, memory traffic, branches, and vector behavior; keep static assembly observations separate from hardware measurements.

The current timing labels need clarification: the warm “compile” bucket includes input representative discovery and installed-program lookup. The “sample” bucket includes root binding and duplicate checks as well as CSPRNG sampling. It must not be treated as pure random-number cost.

Deliver a CPU-work breakdown and an elapsed-time breakdown, with explicit timer boundaries. Use those measurements to estimate upper bounds for each proposed optimization. A fast isolated Poseidon benchmark alone cannot predict a full witness or proof speedup.

# Experiment 2 Remove runtime addressing from native kernels

Hold the existing partitioning and scheduling policy constant at first. Compile each selected region's input and output slots once, then compare three execution forms:

1. Current native kernels using target tables and executor access methods.
2. The same kernels using validated, pre-resolved slot arrays or contiguous ranges.
3. Bounded generated Rust kernels with slot offsets embedded as constants.

This separates the benefit of resolving addresses early from the additional benefit of exposing constants and larger expressions to LLVM. Start with common arithmetic regions and Poseidon boundary handling. Preserve the eight-way Poseidon arithmetic implementation so that a change in hashing does not obscure the addressing comparison.

Validate slot ownership, aliasing, bounds, and dependencies during program preparation or loading. Preserve duplicate-write agreement where required and provide a fallback for unsupported generators. Only use unchecked or non-atomic access where the validation and scheduler make it sound.

Generated regions should retain intermediate values in registers when useful. Every value required by the prover still has to be materialized; generator temporaries and prover-visible witness cells need separate treatment.

Sweep a small range of region sizes and code budgets. Measure generated source bytes, machine-code size, compiler memory, Rust build and link time, per-circuit generation time, load and validation time, and live execution. Inspect representative assembly for residual table loads, calls, spills, and redundant canonicalization.

If fixed-address Rust wins but compilation scales badly, test a separately compiled circuit crate or bounded library of repeated region shapes. If pre-resolved descriptors capture most of the gain, prefer their simpler deployment. A small targeted JIT becomes a follow-up only if these results justify its additional complexity.

Deliver an ablation showing how much comes from address resolution, constant specialization, and region size. Static program embedding alone is not a substitute for this experiment.

# Experiment 3 Compile and schedule larger graph regions

Construct a dependency graph whose nodes contain multiple generators, allowing a ready region to run when its own inputs are available. Compare it with the current execution through global dependency layers.

Begin with deterministic automatic partitioning based on estimated work, fan-out, locality, live values, and available parallelism. Compare three policies: larger regions under the existing layer barriers; statically coarsened region schedules; and dependency-driven ready regions with work stealing or an equivalent task runtime. The first comparison isolates fusion and locality; the others measure the value and cost of removing barriers.

Sweep a bounded set of region work budgets and target ready-task counts. Include schedules that preserve wide Poseidon batches. Over-fusion can reduce SIMD occupancy, serialize independent work, increase register pressure, or delay a critical successor. Fine-grained dependency counters can recreate the original worklist's overhead, so bookkeeping should be paid per coarse region.

Measure total execution, critical-path estimates, task queue and dependency-counter overhead, worker idle time, tail latency, and cache behavior. Compare schedules first with the same kernels and layout, then combine the best scheduler with fixed-address kernels.

Deliver an automatic policy, a small number of useful overrides if necessary, and a fallback for graphs where coarsening loses. MainPod-specific generator identifiers should not determine region boundaries.

# Experiment 4 Reduce Poseidon data movement

Keep the winning arithmetic kernel fixed. Compare pre-resolved input slots, direct packing into vector lanes, and layouts that make independent permutations contiguous. Test output paths that reduce temporary transposes, canonicalization overhead, and individual stores.

MainPod executes 18,368 Poseidon permutations. The gate trace includes 122 values per permutation, so output placement matters as well as the final hash. Optimize stores around certified ownership and contiguous ranges while preserving all required trace values.

Compare array-of-structures, blocked layouts, and SIMD-oriented layouts together with the final prover gather. Earlier experiments showed that a layout which helps execution can make column-oriented output gathering more expensive. Report execution, gather, and the whole witness call for every candidate.

Sweep the minimum group size for batching and compare four- and eight-lane policies on supporting CPUs. AVX2-targeted code on the tested Ice Lake machine is useful diagnostic evidence, but it is not a measurement on a separate AVX2-only CPU.

Deliver a hardware-aware policy driven by CPU features and graph shape, with a scalar fallback and optional overrides. Avoid selecting a layout from a microbenchmark alone.

# Experiment 5 Reduce the surrounding witness cost

Roughly 28 ms remained outside the execute timer in the final field FFT cohort's headline full witness call. Stage medians do not necessarily add to the median full call, but the difference is large enough to investigate.

| Area | Proposed experiment | Required condition |
| --- | --- | --- |
| Input binding | Cache mappings for repeated input shapes and replace repeated sorting or binary searches with validated direct mappings | Handle changing root sets and conflicting duplicate assignments |
| State storage | Reuse allocations and initialize only what correctness requires | Preserve readiness, alias validation, concurrency, and witness wiping |
| Randomness | Separate binding from sampling, then test chunking and CSPRNG batching | Preserve randomness semantics and isolation between proofs |
| Final gather | Compare tiled CPU gather, precomputed ranges, and GPU gather | Include transfers, staging, synchronization, and full proof time |

These changes can proceed independently once profiling identifies worthwhile costs. Buffer reuse must be checked with concurrent proofs. Timing must include destruction or explicitly report wiping outside the execution span.

# Experiment 6 Keep generation and builds practical

Distinguish circuit construction, witness-program generation, native source generation, Rust compilation and linking, loading and validation, and steady-state execution. Report cold and warm cases with the filesystem cache state recorded.

Compare a portable validated program, a pre-resolved descriptor representation, a bounded generated circuit crate, and any larger native variant that survives the earlier experiments. A larger artifact is acceptable; expensive preparation for a small execution gain needs a reuse argument.

For each candidate, report the number of executions needed to recover extra preparation cost: extra one-time cost divided by time saved per execution. Include native compilation for workflows that regenerate circuit binaries. Report long-lived and short-lived prover scenarios separately.

The prior native bank contained 32 kernels and about 27.6 KB of generated Rust. Full static embedding and group specialization offered little extra MainPod benefit while increasing build and binary cost. The SIMD experiment's roughly 20.7 s library build versus 13.4 s original-library build included the earlier bank, failed variants, and diagnostics; it is not a clean marginal SIMD cost. Measure a pruned candidate before making integration claims.

# Workloads and measurement rules

Use MainPod as the main decision workload, plus recursive or nonempty Points circuits for realistic generalization. Retain arithmetic-wide, arithmetic-deep, hash-wide, hash-deep, equality, and mixed synthetic graphs to explain why a policy wins or loses. Include smaller and larger instances where practical.

Run fast correctness checks and broad screening locally. Confirm surviving candidates on x86 with NVIDIA proving, with M4 as a portability and regression check. Use consistent thread counts, CPU affinity, NUMA placement, CPU quota, GPU residency, compiler version, and build flags. Record the actual hardware rather than relying only on the advertised pod shape.

Alternate controls and candidates within a process where feasible, use multiple independent processes, and discard warm-up according to a rule fixed before inspecting results. Report per-process results and dispersion. Keep isolated replay, live witness execution, full witness calls, and prove-and-verify timings in separate columns.

For final candidates, compare witness states exactly against the existing executor and verify resulting proofs. Include aliases, conflicting assignments, edge field values, unsupported generators, snapshot reloads, and concurrent proofs where relevant. Use the default worklist as a correctness reference and contextual baseline; the performance target is the already optimized executor.

Every retained result should have source identity, compiler flags, configuration, machine details, raw timings, correctness output, and commands sufficient to reproduce it. Force rebuilds when extracting archived sources could otherwise leave misleading Cargo timestamps.

# Decision criteria and next deliverables

The first milestone is a refreshed profile and a controlled fixed-address comparison. The second is a region-scheduling comparison, followed by the best combination of addressing, scheduling, and SIMD layout.

As a working screening rule, pursue a repeatable improvement of around 5% or more in MainPod execution, or a smaller gain with very low complexity and broad usefulness. This is a proposed research threshold, not a measured prediction. Larger compiler changes should justify themselves with material live-witness savings and acceptable generation and build costs.

Retain improvements that survive repeated live measurements and remain correct across the workload set. Make a policy configurable when the tradeoff follows hardware or graph structure. Stop expanding an approach when its best result depends on expensive manual tuning, or when gains disappear after full witness and proof costs are included.

The final output should include a recommended default, useful configuration options, negative results, a before-and-after timing table, preparation and build costs, and a compact patch that excludes unsuccessful experimental machinery. No numerical speedup is forecast for fixed-address regions or region scheduling until the controlled experiments run.

# Evidence and source locations

The retained local artifacts are under `.benchmark-artifacts/` in the shared workspace:

- `witness-aot-campaign-2026-10-23/report.html` and `results-summary.json`: original compiler and kernel experiments, M4 measurements, and build costs.
- `witness-aot-campaign-2026-10-23/nvidia-measurement/report.html`: earlier x86 and A40 measurements.
- `poseidon-simd-experiment-2026-10-24/report.html` and `results-summary.json`: final SIMD comparisons, per-process results, and correctness evidence.
- `poseidon-simd-experiment-2026-10-24/poseidon-simd-experiments.patch` and `reproduce.txt`: experimental implementation and reproduction instructions.
- `poseidon-simd-experiment-2026-10-24/plonky2/plonky2/src/plonk/witness_program/region/cpu.rs`, the adjacent `aot.rs`, and `plonky2/plonky2/src/hash/poseidon_trace_simd.rs` in the same experiment directory: execution, dispatch, and SIMD code within the experimental source tree.

The historical measurements describe experiments, not a production merge. Temporary Runpod resources from those measurements have been deleted.

# Experiment log

## Baseline preparation and profiling on 24 October 2026

Work is on `codex/witness-executor-optimization`, in the `plonky2-witness-optimization` worktree. The original `plonky2-integrated` and `pod2-gpu-integrated` checkouts remain unchanged. Code candidates stay uncommitted until their correctness, performance, and maintenance cost justify retaining them.

The working baseline removes unsuccessful within-permutation SIMD, dense MDS and fused-accumulator variants, static embedding, and non-atomic access experiments. It retains the shared validated kernels, native arithmetic bank, and eight-way field FFT Poseidon candidate. Both ordinary and optimized paths pass the 14 witness-program tests on M4. The pruned SIMD code also passes all-trace field-edge tests. Initial x86 testing completed 243 verified live proofs, including worklist comparisons, plus 2,026 exact snapshot executions across eight workloads.

Five alternating replay pairs against the archived optimized implementation put MainPod about 2.2% faster at the median paired comparison, with individual pairs ranging from 1.3% slower to 3.1% faster. The live process comparisons varied substantially. Equality replay regressed relative to that unmerged archive, with a median paired slowdown of about 5.9%. These results do not establish an additional general speedup from cleanup.

The retained implementation is a substantial improvement over the checked executor on the original repository branch: same-machine replay controls measured about 28.26 ms for MainPod, 3.43 ms for equality, and 1.50 ms for arithmetic-deep, compared with roughly 14.41 ms, 1.16 ms, and 0.56 ms for the retained optimized paths. These are replay measurements, not live proving. Two live checked-executor controls measured execution at 48.30 and 45.04 ms. The optimized baseline remains explicitly selectable; it does not change the default worklist or implicitly install a witness program. An isolated checkout of the exact staged implementation passed all 14 witness-program tests and the SIMD trace test before commit.

The optional profiler preserves the normal parallel schedule. It distinguishes input discovery, root binding, random sampling, allocation, seeding, execution layers, alias checks, readiness checks, gather, and wiping. Per-kind CPU totals measure work across threads and overlap with nested Poseidon measurements. Paired instrumentation controls added roughly 18% to MainPod execution, so instrumented times are diagnostic and uninstrumented controls determine performance claims.

In one instrumented live cohort, input discovery took about 3.7 ms, root binding 7.5 ms, and random sampling 3.5 ms. Poseidon was the largest category of worker CPU work. Within batched Poseidon, measured CPU work was approximately 50 ms arithmetic, 19 ms output handling, and 5 ms input handling across threads. These measurements support investigating both output movement and input binding; they are not estimates of wall time recoverable by eliminating either operation.

## Resolving arithmetic slots in advance

The prototype caches lane-major input and output slot arrays for arithmetic chains and calls equivalent native kernels without repeated target-table address calculations. It preserves the existing partitioning, layers, arithmetic expressions, and checked output semantics.

Three same-process alternating MainPod proof comparisons found execution reductions of about 3.9%, minus 4.6%, and 3.2%. Full witness improvements were smaller and inconsistent. One x86 replay comparison improved arithmetic-deep by about 22% but regressed equality by about 8%. Three local replay passes checked 1,152 executions exactly; the x86 prototype completed another 384 exact executions and 204 verified proofs, including worklist comparisons.

**Decision:** do not commit this implementation as it stands. MainPod does not show a sufficiently consistent gain to justify duplicating the native kernel bank, retaining extra slot arrays, and accepting the equality regression. Keep the results as evidence for a more compact implementation or for larger regions. This result does not rule out fixed-address generated Rust or region scheduling.

## Bounded generated Rust with affine slot addresses

A separate prototype generates Rust for regions whose arithmetic descriptor is uniform and whose input and output slots follow constant strides. It uses fixed state-array lengths and constant offsets, preserves the existing scheduling, and matches the complete compiled topology before dispatch. Other regions retain the existing executor.

For MainPod, automatic detection finds 98 eligible groups covering 6,584 arithmetic operations, only about 2.3% of fused arithmetic operations. The generated source is about 92.5 KB and the Python prototype takes about 4.3 seconds to parse and generate it. For arithmetic-deep, a roughly 262 KB body budget covers 55,296 of 65,536 arithmetic operations across 216 groups, with about 0.16 seconds of generation. These generation times describe the Python prototype, rather than a lower bound for a Rust implementation. Exact-state replay passed; x86 results follow below.

**Decision:** reject this affine-only prototype, based on the x86 results below. More general fixed-address regions supporting irregular operands and mixed arithmetic descriptors remain open.

Raw results and source snapshots for this series are retained in `.benchmark-artifacts/witness-optimization-2026-10-24/` in the shared workspace.

## Fixed regions and dependency scheduling results

The affine fixed-address Rust prototype improved arithmetic-deep execution by 24–32% in three x86 replay comparisons. MainPod live execution changed by approximately minus 0.8%, plus 2.3%, and plus 3.7%, with small full-witness changes. Its limited MainPod coverage and additional circuit-specific generation and compilation do not justify retaining this prototype. More general regions with irregular operands and mixed descriptors remain open.

Two scheduling prototypes preserved the existing arithmetic kernels and group boundaries. One used scoped ready-group tasks with dependency counters; the other used persistent workers consuming serial chunks from a priority queue. Both passed exact-state checks, worklist comparisons, and concurrent proofs. MainPod live results were inconsistent across three paired processes. The scoped scheduler substantially regressed arithmetic-deep; the chunk queue also regressed deep and wide hash workloads. Neither implementation is retained. These results test scheduling over existing groups, rather than a compiler that fuses larger regions and retains values across their boundaries.

## Direct input binding retained

Input binding now uses the cached canonical slot map already needed for final gathering. It checks membership, duplicate agreement, and complete root coverage while producing root values, removing repeated input sorting and per-input binary searches on the common path. The fast path is eligible only when the installed input mask equals the program's boundary roots. Programs with other masks retain the previous lookup and worklist fallback behavior.

Six same-process alternating comparisons reduced the full MainPod witness call by 10.8–16.6%, saving approximately 6–10.8 ms per proof. Execution arithmetic is unchanged; this is a witness-preparation improvement. The final input-mask eligibility check was confirmed in two of those processes. MainPod, recursive/nonempty Points, worklist comparisons, and concurrent proofs passed. The exact staged implementation passed all 15 witness-program tests, including agreeing noncanonical aliases, conflicting aliases, unconstrained roots, missing inputs, and extra inputs.

The six paired processes measured the following full-witness medians. Each process alternated implementations over 40 proofs, discarded the first eight observations, and contributed 16 warm observations per implementation. Times from different processes show substantial machine variation; each row is its own comparison. The logged `compile` span includes input-shape discovery and cache lookup, rather than Rust compilation.

| Process | Legacy binding ms | Direct binding ms | Reduction |
| --- | ---: | ---: | ---: |
| 1 | 38.85 | 32.65 | 16.0% |
| 2 | 55.05 | 49.10 | 10.8% |
| 3 | 50.25 | 42.90 | 14.6% |
| 4 | 58.50 | 50.00 | 14.5% |
| 5 | 64.65 | 53.90 | 16.6% |
| 6 | 60.40 | 52.00 | 13.9% |

**Decision:** retain direct binding as the default for eligible installed programs. `P2_WITNESS_INPUT_BINDING=legacy` remains available for comparison. It introduces no additional per-circuit generated code or mapping allocation.

## Poseidon input and output experiments rejected

The I/O prototype loaded input slots in contiguous batches and stored canonical outputs in value order, avoiding temporary per-permutation trace arrays. A second path recognized contiguous blocks in the block8 layout and stored interleaved traces directly. Both used safe Rust and atomic stores. Exact replay, worklist comparisons, concurrent proofs, and Points passed.

Hash-wide and hash-deep replay improved by about 9–10% in some configurations. MainPod's three same-process comparisons showed approximately 1.9% faster, 3.8% slower, and 1.0% faster execution. Block8 did not establish an overall MainPod advantage after gathering and full proving. Neither I/O implementation is retained. Vector stores with separately proved ownership and layouts designed jointly with region partitioning remain different experiments.

## Wiped buffer reuse retained as an option

The executor can reuse state and readiness allocations after wiping witness values with the existing volatile `zeroize` operations and clearing readiness flags. The cache owns each buffer exclusively, matches allocations by length, and retains at most 64 MiB of buffer capacity across circuits. Recent sizes can replace older ones. MainPod's state and readiness payload is about 35.5 MiB.

Six same-process paired runs reduced MainPod execution by 1.4–14.1%, with a median paired reduction of 5.4%. Full-witness results ranged from 0.8% slower to 8.4% faster, with a median paired reduction of 2.1%. All eight x86 replay workloads improved. The final two processes showed smaller execution gains of 2.5% and 1.9%, so the larger earlier gains should not be treated as typical. Buffer ownership, wiping, readiness reset, budget enforcement, exact-state replay, worklist checks, concurrent proofs, and Points were checked.

The buffer comparisons use direct binding in both modes, the same 40-proof alternating protocol, and the same eight-observation warm-up. They must not be added to the binding percentages as a measured combined speedup.

| Process | Fresh execute ms | Reused execute ms | Fresh full witness ms | Reused full witness ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 32.85 | 28.22 | 46.50 | 42.85 |
| 2 | 33.44 | 32.97 | 47.35 | 47.75 |
| 3 | 39.68 | 34.73 | 54.80 | 50.20 |
| 4 | 38.68 | 35.49 | 53.25 | 51.70 |
| 5 | 36.28 | 35.37 | 50.15 | 50.35 |
| 6 | 35.33 | 34.67 | 50.30 | 49.65 |

**Decision:** retain reuse as an opt-in execution optimization with `P2_WITNESS_BUFFERS=pool`. Fresh allocation remains the default. This makes the memory and latency tradeoff explicit without requiring circuit-specific tuning. It adds no circuit-specific Rust generation.

## Configuration and validation at the end of this batch

Use the `witness-executor` integration of pod2 with a Cargo override pointing at this branch's `plonky2` crate. The retained kernel configuration measured on the Xeon/A40 is `P2_AOT_MODE=adaptive`, `P2_AOT_LAYOUT=aos`, `P2_AOT_FUSE=1`, `P2_AOT_CHAIN_LEN=4`, `P2_AOT_SIMPLIFY_ARITH=1`, and `P2_POSEIDON_MODE=batch8-fft`, with eight Rayon threads and a native CPU build. Eight-way execution falls back on builds without the required AVX-512 features. Input binding is automatic for eligible installed programs. Buffer reuse is optional.

This batch completed 1,515 verified MainPod proofs on the temporary Xeon/A40, plus recursive/nonempty Points checks and thousands of exact-state replay executions. The final local implementation passes 17 witness-program tests. The original repositories were left unchanged. Rejected prototypes remain in the benchmark artifacts and are absent from committed implementation code.

The temporary Runpod pod `samwnq8xu2z3r3` was deleted; read-back returned 404. Final results and the measured binary were retrieved first, with archive SHA256 `ce6f7b155afbecc31f3dfe85824c1cccd80a2a55d2d16308010434ba200e2511`.

The next architectural investigation is a more general region compiler supporting irregular operands and mixed descriptors, followed by partitioning and scheduling designed together. Affine fixed regions and scheduling over existing groups did not establish a general MainPod win. The surrounding witness pipeline now has a useful retained improvement, but these experiments do not establish that execution is close to optimal.


## Build cost and unused SIMD code generation

CPU-only M4 release-library rebuilds used eight build jobs, native CPU targeting, incremental compilation disabled, and warm dependencies. The original repository library took 4.00 and 4.10 seconds; the retained kernel baseline took 8.82 and 8.95 seconds; adding direct binding and buffer reuse took 9.08 and 8.69 seconds. Thus the latter changes introduced no measurable marginal library build cost in this comparison. These are Cargo library-unit times, rather than clean workspace build or link times.

A separate alternating comparison then guarded the eight-lane executor call with the existing compile-time native-AVX-512 capability constant. Production dispatch already selects scalar execution on other builds. The guard prevents LLVM from generating unused scalar-emulated eight-lane executor code. Unguarded rebuilds took 8.57 and 8.52 seconds; guarded rebuilds took 6.91 and 6.88 seconds, a roughly 19% reduction in the median library-unit time. This reduces the retained implementation's build overhead on M4 while preserving the native AVX-512 path. The final guarded implementation passes all 17 witness tests with both fresh and pooled buffers, plus the 122-value SIMD trace comparison including field-edge inputs.

**Decision:** retain the compile-time guard. It introduces no per-circuit generation. This final build-only change was made after the Runpod machine was deleted; the native-enabled branch remains the one measured there. A matched final NVIDIA build-cost comparison remains open: the attempted shared-target comparison reused an incompatible cached field dependency after archive extraction, so those partial numbers are excluded. Future archive-based comparisons must touch every local Rust source or use isolated targets.

The final candidates do not add circuit-specific generated Rust. Fresh witness-program generation and disk loading were not re-benchmarked as a complete cold/warm matrix in this batch. The retained input eligibility check runs at installation and reuses the existing canonical slot map. Future region-compiler comparisons must include generation, validation, loading, native compilation, and the number of executions required to repay preparation cost.

## Remaining priorities after this batch

1. Build a bounded region compiler that accepts irregular slot operands and mixed arithmetic descriptors. Measure MainPod coverage before generating large Rust bodies; compare compact resolved descriptors with fixed-address code using the same partitions. Keep prover-visible cells materialized and distinguish them from temporary values that can remain in registers.
2. Design partitioning and scheduling together once useful regions exist. Preserve wide Poseidon batches and enough independent work, then compare larger regions under the current barriers against coarsened or dependency-driven schedules. The rejected schedulers retained the old group boundaries.
3. Revisit Poseidon output layout and stores with region ownership information. The safe I/O prototypes show potential on hash-only graphs but do not justify a MainPod default. Any non-atomic vector-store design needs a sound ownership argument and whole-witness validation.
4. Measure CSPRNG sampling and final gathering separately before changing them. Compare tiled CPU gathering and GPU gathering with transfers and full proof time included. Reuse the feature-gated profiler for attribution and ordinary builds for headline comparisons.
5. Complete the generation/load and build-cost matrix for any surviving architectural candidate, including native NVIDIA builds, an AVX2-only machine, and M4. Keep the current configuration explicit until the CPU and graph-shape policy has broader evidence.

## Reproduction and evidence for this batch

The campaign directory is `.benchmark-artifacts/witness-optimization-2026-10-24/`. `remote/final/results/` contains the cumulative machine details, source/build logs, paired measurements, exact-state output, and proof checks retrieved from the Xeon/A40. `remote/*_measure.py` and the adjacent shell scripts retain the exact remote commands. `input-summary.json`, `io-summary.json`, `pool-summary.json`, and `pool-confirm-summary.json` retain the live samples and process medians. `build-cost-summary.json` and `native-gate-build-comparison.json` retain the local Cargo-unit measurements.

For the measured live configuration, also set `P2_AOT_WITNESS_THREADS=0`, `P2_AOT_GATHER_COLUMNS=2`, `P2_AOT_GATHER_ROWS=1024`, and `P2_AOT_GPU_GATHER=0`. This uses the normal eight-thread Rayon pool and CPU gather; GPU proving remains enabled in pod2. The pod measurements pinned CPUs 24–31 on NUMA node 1. Compiler flags, CPU quota, and complete hardware details are in the retrieved logs. The experimental pod2 harness uses its `witness-executor` branch and a Cargo path override; it is preserved in the source bundles rather than committed into the original pod2 checkout.

Local correctness commands, run from the new worktree, are `CARGO_TARGET_DIR=/tmp/witness-optimization-target RUSTFLAGS="-C target-cpu=native" cargo +nightly test --offline --release -p plonky2 --lib --features witness-profiling witness_program`, repeated with `P2_WITNESS_BUFFERS=pool`, and the same Cargo command with the test filter `poseidon_trace_simd::tests::all_trace_values`. The profiling replay CLI and its configuration are documented in `tools/witness-profile/README.md`.


# Completion plan agreed on 25 October 2026

The deliverable is a supported library feature with measured preparation, first-proof and steady-state costs, plus a bounded final optimization pass. M4 is the development and screening machine; non-M4 testing is deferred until the end at the user's request. Retain each meaningful change in a separate commit after validation. Record rejected prototypes and exclude their implementation from the final branch.

1. Provide typed configuration with measured fast defaults and documented environment overrides. Explicit instance settings take precedence. Resolve settings during preparation, expose effective settings, and keep benchmark overrides out of the normal consumer API. Preserve the explicit compile/load/install lifecycle and normal worklist behavior for circuits without a prepared executor.
2. Let plonky2 own persistent compiled-executor caching through `PLONKY2_EXECUTOR_CACHE_DIR`. Unset means memory-only preparation. POD2 supplies the circuit and input targets, while plonky2 handles identity, validation, cache misses, replacement, atomic writes and concurrent preparation. Cache failure may be reported without preventing in-memory preparation. POD2's circuit-data cache remains separate.
3. Make preparation initialize host certification, slot maps and kernel classification before the first proof. Measure compilation, decoding, validation, preparation, first proof and warm proof separately. Compare compact and fixed-width storage, and remove avoidable generator scans and allocations while preserving compatibility checks.
4. Finish the inexpensive execution pass: metadata copies and locks, input binding, state/readiness allocation and wiping, CSPRNG batching, CPU/GPU gathering including transfer costs, and bounded chunk/batching policies. Validate concurrency, failure cleanup and resource limits.
5. Give general mixed-operation regions with irregular slots a bounded experiment and code budget. Compare compact region execution and generated constant-address code under the same partitions, then test larger regions and scheduling together. Retain only repeatable live improvements that justify preparation, build and maintenance costs. A negative result closes an experiment.
6. Establish the portable hardware-selection story and safe fallbacks. Investigate isolated runtime-dispatched SIMD rather than requiring whole-binary native targeting. Validate M4 first; finish native NVIDIA and non-AVX-512 checks at the end, including a matched build-cost comparison.
7. Integrate POD2 through the supported preparation API, remove its executor-specific caching and registry, and add consumer examples, migration guidance and an acceptance report. Check MainPod, Points, graph-shape fixtures, input aliases/mismatches, cache rejection and recovery, failed executions, concurrent preparation/proofs and multiple circuits.

Environment variables are a supported way for applications to tune plonky2 through POD2 without wrapper fields for every option. Configuration APIs remain necessary for instance-specific policies, resource ownership and lifecycle guarantees. The completion bar is a reproducible fast default without undocumented settings; a complete startup story; and a recorded keep/reject decision for the bounded performance candidates. Full GPU witness generation, an unrestricted JIT and support for every custom generator are follow-up projects unless these measurements establish a concrete need.


## Per-program configuration and resource ownership

The first completion change replaces execution-loop environment reads and global tuning singletons with `WitnessConfig`, resolved when a program is compiled or loaded. Defaults select output-local placement, four-operation chain fusion, validated adaptive arithmetic, bulk Poseidon and native Poseidon batching where compiled support exists. Public compile/load methods accept explicit configuration, while ordinary methods apply documented environment defaults. The new `PLONKY2_EXECUTOR_*` names take precedence over migration aliases. `prove` still requires an installed program before using the executor.

Caller-supplied Rayon pools and program-owned pools are supported. Wiped buffer reuse has a per-program retained-capacity budget and is released with its final owner. `has_witness_program` avoids cloning input-mask metadata. `WitnessFallback::Error` is honored by the prover rather than being swallowed by its worklist fallback. Stored programs now carry their layout and fusion policy; loading can apply different execution/resource settings without changing the stored graph.

Validation: all 19 witness tests passed, including explicit independent configurations, caller-owned pools, stored-program reload and strict input-shape failure. Three alternating replay pairs across eight graph fixtures checked 1,152 executions exactly. With matched non-incremental build settings, median paired MainPod execution changed by about 0.3% slower, with the last two pairs within 0.4%; hash-wide and hash-deep medians were within 0.6%. This is an API/resource-ownership change rather than a claimed speedup. A preliminary comparison used mismatched Cargo incremental settings and is excluded. Raw results are in `.benchmark-artifacts/witness-completion-2026-10-25/`.

## Complete host preparation and shared metadata

Installing a program now completes host certification, representative/cell slot maps, Poseidon output metadata and native arithmetic classification. No witness values or random seeds are sampled during preparation. Immutable topology arrays, tables and the input/circuit key are shared by program handles. Retrieving an installed program no longer copies its input mask. `with_execution_config` can reuse the same topology and prepared tables with different execution resources while retaining its compilation policy.

All 20 witness tests passed, including prepared-table checks and sharing across cloned/reconfigured handles. Another three alternating replay pairs across all eight fixtures checked 1,152 executions exactly. MainPod's median paired difference versus the retained baseline was about 1% slower, within the process variation seen in these local runs; hash-wide/deep medians were within 1.1%. This change is retained for predictable first-proof work and inexpensive handle sharing, rather than a steady-state speedup claim. The final live acceptance measurements will assess the combined public-API configuration.

## Executor-owned caching and preparation API

`prepare_witness_program` and its explicit-configuration/input-witness variants now perform the supported load-or-compile, prepare and install lifecycle. `PLONKY2_EXECUTOR_CACHE_DIR` enables persistent caching independently of POD2's circuit-data cache. Identity includes the circuit, canonical input shape, operation digest, format and compilation policy. Cache misses use per-entry OS file locks, recheck after locking, and atomically replace complete entries. Invalid entries are rebuilt; unavailable storage is a reported, nonfatal memory-only path. The preparation report exposes source and timing spans. Unsupported generator/field/lookup capability is memoized on the circuit and observable through a cheap status API.

All 23 witness tests passed, including cache reload, alias-equivalent schemas, additional input classes, corrupt-entry repair, unavailable cache, concurrent same-object preparation/proofs and independent-object disk misses. Same-object and independent-object races each compiled once. The complete public example verified two proofs with memory-only preparation and with a real environment-configured cold and warm cache. GPU-feature compilation also passed locally on M4. No non-M4 measurements were run in this phase.

Consumer lifecycle, environment precedence, cache compatibility, fallback policy, resource ownership, custom generators and a complete example are documented in `docs/witness-executor.md` and `plonky2/examples/witness_program.rs`.

## Application configuration validation

`WitnessConfig::validate` is public so application/POD2 configuration can fail before threads or circuit preparation are started. Empty explicit cache directories and unsupported serial-build thread requests are rejected. `from_env` now resolves an absent cache directory to a definite disabled policy, allowing a consumer to capture process defaults once. Two focused tests passed, including precedence checks in a subprocess that does not mutate the parent test environment.


## Direct chunk-local randomness sampling

Witness padding now samples directly from one independently OS-seeded, zeroizing ChaCha20 RNG per work chunk. It uses the same field sampler and seed lifetime as before, removing the per-value thread-local/RefCell lookup and deferred boxed RNG. Reference-stream, nested-scope, cleanup and independent-job tests passed. No seed or deterministic-seeding interface is exposed.

An alternating M4 microbenchmark over MainPod's 741,181 random values measured roughly 1.05 ms for the TLS path and 0.73 ms for direct field sampling. A raw bulk/rejection variant measured about 0.70 ms; its extra sampling implementation is not retained for that small additional gain. Three pairs of 16-proof Metal processes completed 96 verified MainPod proofs. After four warm-up proofs per process, the combined root-binding/sampling span improved in all three pairs. Execution and total-proof results varied substantially, so this experiment establishes reduced sampling overhead rather than a reliable total-proof percentage. The simple direct sampler is retained.

The first sandboxed proof measurement could not enumerate Metal and used CPU proving; it was interrupted and retained under `sandbox-cpu-fallback/` as correctness evidence only. Headline M4 GPU comparisons run with native Metal access. POD2's adapter tests and its MainPod preparation/worklist-comparison test passed through the new public API. Local test dependency setup used the actual macOS SCIP binary after identifying Linux binaries in an existing build cache.


## Avoid duplicate expansion during operation validation

Operation identity now hashes the complete typed descriptor once instead of also allocating and hashing its expanded input/output vectors. Descriptor expansion and target/wire mapping live in fingerprinted sources, so changes to those rules still invalidate stored programs. External generators remain covered by their complete descriptors. A new application-generator test uses identical circuit digests but different generated public values: loading rejects the changed operations, cache preparation creates separate entries, and both proofs verify with the correct values.

All 26 witness-program tests passed. MainPod's 18.76 MB compact program was loaded 20 times per process and followed by a verified Metal proof. Discarding the first four loads, the first comparison reduced median validated decoding from 104.3 ms to 68.8 ms and decoding plus host preparation from 152.2 ms to 119.3 ms. The reversed-order candidate measured 67.1 ms and 117.2 ms respectively; the reversed-order control measured 103.7 ms and 152.6 ms. This removes preparation work without changing the execution schedule or introducing circuit-specific code.
