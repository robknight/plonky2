# Witness executor optimization experiments and priorities

As of 24 October 2026. The objective is to beat the existing witness executor during execution, with MainPod as the primary workload and a small set of other circuits checking whether the approach generalizes. Program generation should stay quick, and Rust build cost must remain practical. Larger files on disk are acceptable when they buy speed without expensive generation or compilation.

The next sequence is: profile the final SIMD implementation; isolate the benefit of fixed slot addresses within the current execution structure; then test larger regions and scheduling with fewer global barriers. Poseidon data movement and witness preparation remain useful complementary experiments. The aim is an automatic compiler with a small set of optional policies, rather than hand-written MainPod code.

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
| Larger native regions with fixed slot addresses | Substantially unexplored | First major compiler experiment |
| Graph regions scheduled with fewer global barriers | Substantially unexplored | First major scheduling experiment |
| Trustworthy breakdown of the final optimized executor | Incomplete | Refresh before predicting remaining headroom |

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

For MainPod, automatic detection finds 98 eligible groups covering 6,584 arithmetic operations, only about 2.3% of fused arithmetic operations. The generated source is about 92.5 KB and the Python prototype takes about 4.3 seconds to parse and generate it. For arithmetic-deep, a roughly 262 KB body budget covers 55,296 of 65,536 arithmetic operations across 216 groups, with about 0.16 seconds of generation. These generation times describe the Python prototype, rather than a lower bound for a Rust implementation. Exact-state replay passes locally for the MainPod variant; x86 performance measurements are in progress.

**Decision pending:** this affine-only strategy has limited MainPod coverage. Its results must be distinguished from more general fixed-address regions that support irregular operands and mixed arithmetic descriptors.

Raw results and source snapshots for this series are retained in `.benchmark-artifacts/witness-optimization-2026-10-24/` in the shared workspace.

