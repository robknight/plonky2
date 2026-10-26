# Runtime configuration audit

26 October 2026. Prefer typed environment defaults for deployment-wide performance and resource policy. Applications using POD2 can set these without adding plonky2-specific fields to every intervening library. plonky2 owns extraction, validation and the effective typed settings.

## Audit decisions

| API / settings | Decision | Reason |
| --- | --- | --- |
| `GpuConfig`: backend, adapter, mode, workers, budgets, hybrid share, software adapters, exit cleanup | Environment defaults, applied automatically | Deployment policy; no POD2 wrapper is necessary. |
| `gpu::set_enabled` startup use | Add `PLONKY2_GPU_ENABLED` | CPU-only startup needs no direct plonky2 API call. Keep `set_enabled` for temporary runtime routing. |
| `WitnessConfig`: scheduling, layout, fusion, gather policies and budgets | Keep environment-first preparation; share strict typed extraction | Already supports `PLONKY2_EXECUTOR_*`; remove duplicated parsing and preserve documented migration aliases. |
| `WitnessCache` / cache directory | Keep `PLONKY2_EXECUTOR_CACHE_DIR` and explicit overrides | Plonky2 manages programs independently of POD2 circuit-data caching. Paths support native OS strings. |
| `WitnessConfig::thread_pool` | Keep explicit `Arc<rayon::ThreadPool>` | A live resource cannot be encoded in an environment variable. Thread count already has an environment setting. |
| `prepare_*_with_config`, `with_execution_config`, `GpuContext::with_config` | Keep low-level explicit APIs | Support independently configured programs and caller-owned low-level contexts. Private GPU contexts do not select the device for ordinary plonky2/POD2 proofs. Ordinary calls need no configuration argument. |
| POD2 `set_witness_thread_pool` | Expose only resource injection | Accept an application-owned Rayon pool. Deployment policy stays in environment variables; POD2 no longer forwards plonky2 config structs or re-exports their enums. |
| MainPod proving example | Start GPU overrides from `GpuConfig::from_env()` | Preserve deployment settings while setting its known actual worker count from the CLI. |
| `CircuitConfig`, `FriConfig`, `StarkConfig`, `GenericConfig` / hash choice, POD2 `Params` and container/Merkle limits | Keep explicit | Circuit shape, security targets, commitments and verifier/proof parameters must remain reproducible and visible to circuit construction. |
| `TimingTree`, witness inputs, circuit input targets, serializers, shutdown | Keep explicit | These are execution data, observation or lifecycle operations rather than deployment defaults. |
| Profiling/export/benchmark controls | Keep their separate diagnostic parsers and feature gates | Loose diagnostic flags are not the contract for production configuration. This audit does not expose experimental controls in normal builds. |
| Cargo features and Rayon global pool | Keep existing mechanisms | Environment variables cannot enable code omitted at compilation. Rayon owns `RAYON_NUM_THREADS`; no duplicate global CPU-thread setting is introduced. |

## GPU environment contract

These variables work in ordinary `gpu` builds, through POD2 or any other consumer. No call to `configure` is required.

| Variable | Accepted value | Default |
| --- | --- | --- |
| `PLONKY2_GPU_ENABLED` | Boolean | `true` |
| `PLONKY2_GPU_BACKEND` | `auto`, `metal`, `vulkan`, `dx12` | `auto` |
| `PLONKY2_GPU_ADAPTER_NAME` | Case-insensitive name substring; empty matches any | Any hardware adapter |
| `PLONKY2_GPU_ALLOW_SOFTWARE` | Boolean | `false` |
| `PLONKY2_GPU_MODE` | `auto`, `resident`, `hybrid` | `auto` |
| `PLONKY2_GPU_PROVING_WORKERS` | Integer at least 1 | `1` |
| `PLONKY2_GPU_RETAINED_BYTES` | `auto` or non-negative integer bytes; zero disables retention | 2 GiB per proving worker |
| `PLONKY2_GPU_POOL_BYTES` | `auto` or non-negative integer bytes; zero disables pooling | 4 GiB on unified memory, zero on discrete |
| `PLONKY2_GPU_QUOTIENT_CPU_SHARE` | Finite fraction in `[0, 1]` | `0.10` |
| `PLONKY2_GPU_SHUTDOWN_AT_EXIT` | Boolean | `true` |

Booleans accept exactly `true`, `false`, `1` and `0`. Numeric settings use Rust's integer/float parsing, followed by range validation. Empty numeric/boolean/enum settings are errors. Overflow, negative byte budgets, NaN and out-of-range fractions are rejected. Unknown enum values and non-Unicode textual settings are errors naming the actual variable. No environment value is printed in these errors. Cache paths preserve non-Unicode OS paths; absent/empty cache paths disable persistent executor storage.

```sh
PLONKY2_GPU_PROVING_WORKERS=4 \
PLONKY2_GPU_RETAINED_BYTES=2147483648 \
PLONKY2_EXECUTOR_THREADS=2 \
PLONKY2_EXECUTOR_CACHE_DIR=/var/cache/my-app/executors \
./my-app
```

The worker hint sizes GPU resources; it does not spawn application proof requests. Witness `THREADS=2` creates an execution pool, whereas zero inherits the caller's Rayon pool. Applications with multiple circuits can instead share a caller-owned pool explicitly.

## Extraction, precedence and lifetime

`GpuConfig::from_env()` and `WitnessConfig::from_env()` return typed, validated settings without creating a device, execution pool or disk cache. They use a shared internal extractor with numeric, boolean, enum and native-path readers. `EnvironmentError` preserves the selected variable name and expected type/value. Tests inject a lookup to avoid mutating process-global environment in parallel; lifecycle tests run in isolated child processes.

The precedence is plonky2 defaults, then library suggestions, then application environment variables, then optional explicit application overrides. `GpuConfig::from_env_with_defaults(defaults)` and `WitnessConfig::from_env_with_defaults(defaults)` implement the library-default layer. Unset variables preserve suggested values. Empty adapter text clears a filter; GPU budget `auto` clears a suggested fixed budget; an empty executor cache directory disables a suggested cache; a nonempty directory can enable caching even if the library suggested disabling it. An explicit witness thread-count variable selects an owned/inherited pool instead of a library-suggested pool handle.

Libraries offering process-wide GPU suggestions can call `gpu::configure_defaults(defaults)`. This never initializes a device or overrides application settings. It returns true only for the first offer before settings have been resolved; later offers, application overrides and already-initialized contexts are left unchanged. Environment variables are applied when global settings are first resolved. `gpu::configure(config)` is the distinct explicit application-override API. POD2 currently leaves global GPU settings alone and uses plonky2's platform defaults plus the application's environment.

GPU global settings are captured once by `gpu::config()` or the first automatic context request. Both values and decoding errors are captured. Set variables before either operation. `gpu::config()` now returns `Result<GpuConfig, ConfigureError>` so applications can validate deployment settings at startup. Automatic GPU routing reports invalid configuration with a warning and uses the CPU, consistent with its optional-device fallback; it does not create a GPU with silently substituted defaults. Independently creating `GpuContext::new()` propagates configuration errors.

`gpu::configure(config)` overrides the complete global configuration until the context is initialized, including an earlier failed `config()` read. `GpuContext::with_config(&config)` uses exactly its argument for a caller-owned low-level context. It does not replace the global context or route POD2 proofs through that private context. An explicit `GpuConfig::default()` intentionally ignores environment settings. An application can override one field while retaining process defaults:

```rust,ignore
let mut settings = plonky2::gpu::GpuConfig::from_env()?;
settings.proving_workers = actual_application_workers;
plonky2::gpu::configure(settings)?;
```

`enabled=false` avoids device creation and is fixed with the startup configuration. `set_enabled(false)` is a separate temporary routing gate; setting it back to true does not undo startup disablement or permanent shutdown.

Witness settings are captured during preparation and remain in the installed program. Current `PLONKY2_EXECUTOR_*` names override supported `P2_*` aliases; both override library defaults when using `from_env_with_defaults`. Standalone applications can deliberately bypass environment settings with the explicit `_with_config` APIs. Shadowed aliases are not parsed. `WitnessCache::Environment` is an explicit exception in otherwise passed-in configuration: it still resolves the cache-directory hint during preparation; choose `Disabled` or `Directory` for an explicit cache policy. POD2 captures its defaults through `WitnessConfig::from_env_with_defaults` on first preparation/proving because circuit objects are shared. It exposes no scalar plonky2 policy wrapper. Applications may call `set_witness_thread_pool(Arc<rayon::ThreadPool>)` before that point to inject a resource; this explicit application handle takes precedence over an environment thread-count hint while other environment policies remain effective. The pool is retained with process-wide circuit caches; this is not an independently disposable per-proof runtime.

No environment lookup is added to kernel loops. Execution remains independent of later environment changes. The shared extractor is included in the conservative executor source fingerprint, so this change invalidates old on-disk executor entries once; managed preparation rebuilds them.

## Migration

- Existing code using `gpu::config()` should handle its new `Result`, usually with `?`.
- Prefer environment variables for deployment knobs. Library authors should use `from_env_with_defaults` or `gpu::configure_defaults`; full `configure` is reserved for explicit application overrides. Standalone plonky2 explicit APIs remain supported.
- Change partial overrides built from `GpuConfig::default()` to `GpuConfig::from_env()?` when they should preserve deployment settings.
- Witness migration aliases retain priority below the new names. Legacy fuse/gather/buffer switches now reject unknown values rather than silently accepting them; use documented booleans or `pool`/`fresh` for `P2_WITNESS_BUFFERS`.
- POD2 removes `configure_witness_executor(WitnessConfig)` and its configuration-type re-exports. Use environment variables for policy and `set_witness_thread_pool` for a caller-owned pool.
- Security/circuit configuration, custom-generator descriptions and application proof concurrency stay explicit.


## Validation

Initial audit logs remain in `.benchmark-artifacts/configuration-audit-2026-10-26/`. The library-default refinement is recorded in `.benchmark-artifacts/library-configuration-precedence-2026-10-26/`:

- Seven environment tests, including six isolated GPU-global lifecycle scenarios, validate preserved defaults, environment precedence, explicit application precedence, rejected late suggestions and reset values.
- All 36 witness tests passed, including library cache/buffer/fallback suggestions overridden by application variables and replacement of a library-suggested pool by an environment thread count.
- POD2's resource tests validate shared caller-owned pool identity, environment-controlled policy and setup freezing. A standalone application with direct dependencies only on POD2, Rayon, logging and error handling verified two MainPod proofs on Apple M4/Metal. It used a two-thread application pool alongside a one-thread environment hint, strict compiled execution, environment GPU policy and automatic cleanup; its source imports no plonky2 API.
- Default CPU, `no_std`, std-only, GPU rustdoc, formatting and lint checks passed. Lint output contains existing unrelated warnings. This changes startup configuration and public integration, without adding kernel-loop environment reads or making a new performance claim.
