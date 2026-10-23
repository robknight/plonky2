//! GPU acceleration of the prover's parallel phases through wgpu.
//!
//! Behind the `gpu` feature, Merkle hashing (`MerkleTree::new`), the batched
//! IFFT, coset LDE and leaf layout of `PolynomialBatch::from_values` and
//! `from_coeffs`, the folding and commitment of FRI layers, the
//! evaluation of the quotient polynomial (a program interpreted per row,
//! traced from the gates' own circuit evaluations; see [`quotient`] and
//! [`trace`]) and the opening phase (the openings and FRI's initial
//! polynomial from the device copies of the committed coefficients; see
//! [`opening`]) run on a GPU when a suitable adapter is found at runtime.
//! Proofs are unchanged: every
//! kernel reproduces the CPU implementation bit for bit (digests, caps and
//! coefficients are compared exactly in the tests), so `CommonCircuitData`,
//! `VerifierCircuitData` and the verifier do not know which path produced a
//! proof.
//!
//! The kernels are generated in [`shader`] from this crate's Poseidon constants.
//! The field arithmetic and permutation run on Metal, Vulkan and DirectX 12,
//! with a portable flavor for adapters without 64-bit shader integers.
//!
//! Selection is by feature plus a runtime check. A process-wide
//! [`GpuContext`] holds the device, queue and compiled pipelines and is
//! created on first use. It is shut down automatically at normal process exit
//! when nothing still holds it; applications may call [`shutdown`] earlier,
//! after joining proving workers, to release driver resources sooner.
//!
//! The context reads no settings from the environment; only the
//! `gpu-profiling` diagnostics below do. Applications choose its
//! settings with [`configure`] before the first proof (see [`GpuConfig`]), and
//! [`set_enabled`] sends proofs to the CPU; without either, the context uses
//! [`GpuConfig::default`] on the best available adapter.
//!
//! # Diagnostics
//!
//! Debug logs need no switch: `RUST_LOG=plonky2::witness_program::compile=debug`
//! logs a witness program's compile phases, `plonky2::witness_program::executor`
//! the compiled executor's report per proof, and `plonky2::util::timing` the
//! prover's timing tree.
//!
//! The switches below are read only in `gpu-profiling` builds (and, for
//! `PLONKY2_GPU_FLAVOR`, in tests). A flag is on when set to anything but `0`;
//! a value that does not parse is ignored with a warning. They change what is
//! measured or recorded, never the proof.
//!
//! | Variable | Value | Effect |
//! | --- | --- | --- |
//! | `PLONKY2_GPU_TIMESTAMPS` | flag | Device timestamps for Merkle leaf and level passes and the quotient evaluation (debug logs). Adapters without timestamp queries prove without them. |
//! | `PLONKY2_GPU_PASS_TIMELINE` | flag | Logs a device timeline of every named compute pass after each proof (`plonky2::gpu::timestamps`, info); see `tools/gpu/gpu-timeline`. |
//! | `PLONKY2_DELAY_SCOPE`, `PLONKY2_DELAY_MS` | scope name, milliseconds | Sleeps at the start of every timing scope with that name, to probe the critical path. |
//! | `PLONKY2_GPU_CAPTURE`, `PLONKY2_GPU_CAPTURE_QUOTIENT` | `.gputrace` path | Metal capture of the wires commitment or quotient evaluation; `PLONKY2_GPU_CAPTURE_INDEX` selects the zero-based invocation of each phase (default 1). Needs `METAL_CAPTURE_ENABLED=1`. |
//! | `PLONKY2_GPU_PROGRAM_DUMP` | path | Writes the quotient program, for `bench_quotient_occupancy`. |
//! | `PLONKY2_GPU_QMUL_DEBUG`, `PLONKY2_GPU_QMUL_DUMP` | flag, path prefix | Logs quintic fusion counts; dumps the gate DAG and its fused groups. |
//! | `PLONKY2_GPU_QUOTIENT_NONCE` | integer | Renames the quotient entry point, forcing a driver shader cache miss. |
//! | `PLONKY2_GPU_COEFFICIENT_AUDIT` | flag | Prints each coefficient transfer's direction, stage, size and time to stderr. |
//! | `PLONKY2_GPU_FLAVOR` | `portable` | Portable multiplication even where 64-bit shader integers are available. |
//! | `PLONKY2_GPU_WITNESS_CPU_EXECUTOR_PROFILE` | flag | Runs the compiled executor's groups one at a time and prints the time per operation kind. |
//!
//! Buffers are bound whole, so the adapter must allow storage bindings of the
//! size of an extended polynomial batch (about 583 MB for 139 columns of
//! 2^19 field elements). Batches that exceed the adapter's limits fall back
//! to the CPU.

use core::sync::atomic::{AtomicBool, Ordering};
use std::borrow::Cow;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use hashbrown::HashMap;
use wgpu::util::DeviceExt;

use crate::field::goldilocks_field::GoldilocksField;
use crate::field::types::{Field, Field64, PrimeField64};
use crate::gpu::shader::Flavor;

#[cfg(feature = "gpu-profiling")]
pub(crate) mod capture;
mod config;
#[doc(hidden)]
pub mod fri;
#[doc(hidden)]
pub mod lde;
#[doc(hidden)]
pub mod merkle;
#[doc(hidden)]
pub mod opening;
#[doc(hidden)]
pub mod permutation;
pub(crate) mod policy;
pub use config::{Backend, ConfigureError, GpuConfig};
pub use policy::ExecutionMode;
pub(crate) mod pow;
#[doc(hidden)]
pub mod quotient;
pub(crate) mod resident_witness;
mod retained;
#[doc(hidden)]
pub mod shader;
#[cfg(test)]
pub(crate) mod tests;
pub(crate) mod timestamps;
#[doc(hidden)]
pub mod trace;

/// The order of the field the kernels implement.
#[doc(hidden)]
pub const GOLDILOCKS_ORDER: u64 = GoldilocksField::ORDER;

enum ContextState {
    Uninitialized,
    Available(Option<Arc<GpuContext>>),
    Shutdown,
}

static CONTEXT: Mutex<ContextState> = Mutex::new(ContextState::Uninitialized);

/// The settings `configure` stored; `None` until it is called.
static CONFIG: Mutex<Option<GpuConfig>> = Mutex::new(None);

thread_local! {
    /// The last submission made from this thread, which its readbacks wait for.
    static LAST_SUBMISSION: core::cell::RefCell<Option<wgpu::SubmissionIndex>> =
        const { core::cell::RefCell::new(None) };
}
static ENABLED: AtomicBool = AtomicBool::new(true);

/// The process-wide GPU context, created on first use, or `None` if GPU
/// paths are disabled, unavailable, or permanently shut down.
///
/// Keep this handle alive for the entire GPU operation. Outstanding handles
/// prevent [`shutdown`] from releasing resources that are still in use.
#[doc(hidden)]
pub fn context() -> Option<Arc<GpuContext>> {
    if !ENABLED.load(Ordering::Relaxed) {
        return None;
    }
    let mut state = CONTEXT.lock().unwrap();
    if matches!(*state, ContextState::Uninitialized) {
        let config = config();
        let context = match GpuContext::with_config(&config) {
            Ok(context) => {
                log::info!(
                    "plonky2 gpu: using {} ({:?}, {:?}, {:?} multiplication)",
                    context.info.name,
                    context.info.backend,
                    context.info.device_type,
                    context.flavor
                );
                if config.shutdown_at_exit {
                    register_exit_shutdown();
                }
                Some(Arc::new(context))
            }
            Err(error) => {
                log::info!("plonky2 gpu: unavailable, using the CPU ({error})");
                None
            }
        };
        *state = ContextState::Available(context);
    }
    match &*state {
        ContextState::Available(context) => context.clone(),
        _ => None,
    }
}

/// Sets the global context's settings. Call it before the first proof: once
/// the context exists, or after [`shutdown`], its settings are fixed and this
/// returns [`ConfigureError::AlreadyInitialized`]. Without a call the context
/// uses [`GpuConfig::default`].
pub fn configure(config: GpuConfig) -> Result<(), ConfigureError> {
    config.validate()?;
    let state = CONTEXT.lock().unwrap();
    if !matches!(*state, ContextState::Uninitialized) {
        return Err(ConfigureError::AlreadyInitialized);
    }
    *CONFIG.lock().unwrap() = Some(config);
    Ok(())
}

/// The settings the global context uses, or will use when it is created.
pub fn config() -> GpuConfig {
    CONFIG.lock().unwrap().clone().unwrap_or_default()
}

/// Why an orderly GPU shutdown could not complete.
#[derive(Debug)]
pub enum ShutdownError {
    /// A caller or GPU-resident leaf storage still holds a context handle.
    InUse,
    /// Waiting for submitted GPU work failed. The context is kept for retry.
    Poll(wgpu::PollError),
}

impl core::fmt::Display for ShutdownError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InUse => f.write_str("GPU context is still in use"),
            Self::Poll(error) => write!(f, "waiting for GPU shutdown: {error}"),
        }
    }
}

impl std::error::Error for ShutdownError {}

/// Waits for GPU work, clears retained polynomial buffers (including buffers
/// in application circuit caches), and drops the global context and pipelines.
///
/// The same shutdown runs automatically at normal process exit when
/// [`GpuConfig::shutdown_at_exit`] is enabled. Call it to
/// release the GPU earlier, after stopping/joining proving workers and
/// releasing all [`context`] handles and GPU-resident leaves. Fully
/// materializing leaves on the host releases their device/context handles.
/// Releasing GPU resources before the graphics driver's process-exit cleanup
/// avoids crashes seen on NVIDIA/Vulkan. An error leaves the context
/// available for retry. Success is permanent and idempotent: subsequent
/// proving uses the CPU, even after `set_enabled(true)`. Does not initialize
/// a device if the GPU was never used.
///
/// Independently created [`GpuContext::new`] contexts are caller-owned and
/// must be dropped separately.
pub fn shutdown() -> Result<(), ShutdownError> {
    let mut state = CONTEXT.lock().unwrap();
    shutdown_locked(&mut state, false)
}

/// Registers [`shutdown_at_exit`] with the C runtime's `atexit`, once, when
/// the first context is created. Registered after the graphics driver has
/// loaded, so it runs before the driver's own exit handlers. Skipped when
/// [`GpuConfig::shutdown_at_exit`] is false.
fn register_exit_shutdown() {
    static REGISTERED: std::sync::Once = std::sync::Once::new();
    #[cfg(not(target_arch = "wasm32"))]
    REGISTERED.call_once(|| {
        extern "C" {
            fn atexit(callback: extern "C" fn()) -> core::ffi::c_int;
        }
        // SAFETY: atexit only stores the function pointer. The callback has
        // the C ABI and catches panics before they cross that boundary.
        if unsafe { atexit(shutdown_at_exit) } != 0 {
            log::debug!("plonky2 gpu: could not register the exit shutdown");
        }
    });
    #[cfg(target_arch = "wasm32")]
    let _ = &REGISTERED;
}

/// The exit-time shutdown: tries the context and cache-registry locks without
/// waiting and catches panics. It can still wait for queued GPU work. If proving is in progress on
/// another thread, or a handle is still held, it leaves the context alive
/// and lets the process exit.
pub(crate) extern "C" fn shutdown_at_exit() {
    let _ = std::panic::catch_unwind(|| {
        let Ok(mut state) = CONTEXT.try_lock() else {
            return;
        };
        match shutdown_locked(&mut state, true) {
            Ok(()) => log::debug!("plonky2 gpu: shut down at process exit"),
            Err(error) => log::debug!("plonky2 gpu: no shutdown at exit: {error}"),
        }
    });
}

fn shutdown_locked(state: &mut ContextState, at_exit: bool) -> Result<(), ShutdownError> {
    if let ContextState::Available(Some(context)) = &mut *state {
        // get_mut also excludes Weak handles that could be upgraded while
        // shutdown is running; merely checking strong_count would race them.
        let context = Arc::get_mut(context).ok_or(ShutdownError::InUse)?;
        context
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(ShutdownError::Poll)?;
        // Clone and cache registration share this lock, so a concurrent
        // circuit clone cannot retain a buffer after this pass clears it.
        let mut caches = if at_exit {
            DEVICE_CACHES.try_lock().map_err(|_| ShutdownError::InUse)?
        } else {
            DEVICE_CACHES.lock().unwrap()
        };
        for cache in caches.drain(..).filter_map(|cache| cache.upgrade()) {
            cache.take();
        }
    }
    *state = ContextState::Shutdown;
    Ok(())
}

/// Enables or disables the GPU paths for the whole process. The context, once
/// created, is kept until [`shutdown`]; disabling only routes calls to the CPU.
pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether a GPU context is available and enabled.
pub fn is_available() -> bool {
    context().is_some()
}

/// Whether `F` is the field the kernels implement.
#[doc(hidden)]
pub fn field_supported<F: PrimeField64>() -> bool {
    F::ORDER == GOLDILOCKS_ORDER
}

/// A device, its queue, the compiled pipelines and cached constant tables.
#[doc(hidden)]
pub struct GpuContext {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub flavor: Flavor,
    /// Storage buffers can be mapped directly (unified memory).
    pub mappable: bool,
    readback_pool: Mutex<Vec<wgpu::Buffer>>,
    /// Host-mappable upload buffers reused across proofs (`upload_staging`).
    upload_pool: Mutex<Vec<UploadStaging>>,
    /// Free storage buffers for reuse by `storage`.
    pub(crate) storage_pool: Arc<StoragePool>,
    pub(crate) options: GpuOptions,
    retained: Arc<retained::RetainedBudget>,
    pub max_binding: u64,
    pub max_workgroups: u32,
    hash_layout: wgpu::BindGroupLayout,
    pub(crate) hash_leaves: wgpu::ComputePipeline,
    pub(crate) hash_nodes: wgpu::ComputePipeline,
    pub(crate) permute: wgpu::ComputePipeline,
    fft_layout: wgpu::BindGroupLayout,
    pub(crate) fft: wgpu::ComputePipeline,
    fold_layout: wgpu::BindGroupLayout,
    pub(crate) fold: wgpu::ComputePipeline,
    /// The proof-of-work search, compiled on first use.
    pub(crate) pow_kernel: OnceLock<Option<pow::PowKernel>>,
    open_layout: wgpu::BindGroupLayout,
    pub(crate) point_powers: wgpu::ComputePipeline,
    pub(crate) eval_openings: wgpu::ComputePipeline,
    combine_layout: wgpu::BindGroupLayout,
    /// `batch_powers`, `reduce_batch`, `divide_blocks`, `scan_blocks` and
    /// `combine_quotients`, dispatched in this order.
    pub(crate) combine: [wgpu::ComputePipeline; 5],
    permutation_layout: wgpu::BindGroupLayout,
    /// `row_products`, `block_products`, `block_prefix` and `fill_columns`,
    /// dispatched in this order.
    pub(crate) permutation: [wgpu::ComputePipeline; 4],
    /// A small buffer bound where a kernel does not use a read-only binding.
    pub(crate) dummy: DeviceBuffer,
    /// As `dummy`, for read-write bindings (a buffer cannot be bound both ways in one dispatch).
    pub(crate) dummy_rw: DeviceBuffer,
    /// `g^k` for `k < 2^(log_n - 1)`, with `g` the primitive `2^log_n`-th root of unity.
    twiddles: Mutex<HashMap<u32, DeviceBuffer>>,
    /// Powers of a coset shift.
    shift_pows: Mutex<HashMap<(u64, u32), DeviceBuffer>>,
    /// Quotient programs by circuit key (`None` when a circuit cannot be traced).
    pub(crate) quotient_programs:
        Mutex<HashMap<u64, Option<std::sync::Arc<quotient::QuotientProgram>>>>,
    /// Quotient kernels by register count, challenge count and quintic usage.
    pub(crate) quotient_kernels:
        Mutex<HashMap<(u32, u32, bool), std::sync::Arc<quotient::QuotientKernel>>>,
    pub(crate) row_tables: Mutex<HashMap<(u32, u32), DeviceBuffer>>,
    /// The whole-proof pass timeline (`PLONKY2_GPU_PASS_TIMELINE=1`).
    #[cfg(feature = "gpu-profiling")]
    pub(crate) pass_timeline: Option<timestamps::PassTimeline>,
    /// The share of the quotient's points the CPU evaluates alongside the
    /// kernel, adapted from the measured rates of completed proofs.
    pub(crate) quotient_cpu_share: Mutex<f64>,
}

/// A device buffer handle. Buffers from `GpuContext::storage` return to the
/// context's pool when the last handle is dropped and pooling is enabled.
/// Reuse avoids allocation, zero-initialization and first-use page faults
/// for large commitment buffers.
/// Reused buffers hold their previous contents.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct DeviceBuffer(Arc<DeviceBufferInner>);

#[derive(Debug)]
struct DeviceBufferInner {
    buffer: wgpu::Buffer,
    pool: Option<Arc<StoragePool>>,
}

impl Drop for DeviceBufferInner {
    fn drop(&mut self) {
        if let Some(pool) = &self.pool {
            pool.put(self.buffer.clone());
        }
    }
}

impl core::ops::Deref for DeviceBuffer {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &wgpu::Buffer {
        &self.0.buffer
    }
}

impl DeviceBuffer {
    /// A handle that does not return `buffer` to any pool.
    pub(crate) fn unpooled(buffer: wgpu::Buffer) -> Self {
        Self(Arc::new(DeviceBufferInner { buffer, pool: None }))
    }
}

/// Free storage buffers by exact size, at most `limit` bytes in all (the
/// oldest leave first), [`GpuConfig::pool_bytes`].
#[derive(Debug)]
pub(crate) struct StoragePool {
    free: Mutex<Vec<wgpu::Buffer>>,
    limit: u64,
}

impl StoragePool {
    fn take(&self, size: u64) -> Option<wgpu::Buffer> {
        let mut free = self.free.lock().unwrap();
        let i = free.iter().rposition(|b| b.size() == size)?;
        Some(free.swap_remove(i))
    }

    fn put(&self, buffer: wgpu::Buffer) {
        if buffer.size() > self.limit {
            return;
        }
        let mut free = self.free.lock().unwrap();
        free.push(buffer);
        while free.iter().map(|b| b.size()).sum::<u64>() > self.limit {
            free.remove(0);
        }
    }

    /// Bytes of free buffers held.
    pub(crate) fn bytes(&self) -> u64 {
        self.free.lock().unwrap().iter().map(|b| b.size()).sum()
    }
}

const UPLOAD_PENDING: u8 = 0;
const UPLOAD_MAPPED: u8 = 1;
const UPLOAD_FAILED: u8 = 2;

/// A reusable host-mappable upload buffer, its mapping state, and the
/// submission of its last copy (which its re-mapping waits for).
pub(crate) struct UploadStaging {
    pub(crate) buffer: wgpu::Buffer,
    state: std::sync::Arc<std::sync::atomic::AtomicU8>,
    copied: Option<wgpu::SubmissionIndex>,
}

impl Drop for UploadStaging {
    /// A buffer leaving the pool while mapped (eviction, shutdown) is zeroed
    /// first, so the last upload's plaintext does not outlive it in mapped
    /// host memory. Buffers that are not mapped cannot be cleared here.
    fn drop(&mut self) {
        if self.state.load(std::sync::atomic::Ordering::Acquire) == UPLOAD_MAPPED {
            if let Ok(mut view) = self.buffer.slice(..).get_mapped_range_mut() {
                let (words, _) = view.slice(..).into_chunks::<8>();
                for word in words {
                    word.write([0; 8]);
                }
            }
        }
    }
}

/// Trees and polynomial batches below `2^MIN_LOG_LEAVES` rows stay on the CPU,
/// where dispatch latency dominates.
pub(crate) const MIN_LOG_LEAVES: u32 = 10;

/// Immutable per-context policy, resolved from a [`GpuConfig`].
pub(crate) struct GpuOptions {
    pub pool_bytes: u64,
    pub mode: ExecutionMode,
    pub proving_workers: usize,
    pub retained_bytes: u64,
}

impl GpuOptions {
    fn new(config: &GpuConfig, mappable: bool) -> Self {
        Self {
            pool_bytes: config
                .pool_bytes
                .unwrap_or_else(|| Self::default_pool_bytes(mappable)),
            mode: config.mode,
            proving_workers: config.proving_workers,
            retained_bytes: config.retained_bytes_limit(),
        }
    }

    fn default_pool_bytes(mappable: bool) -> u64 {
        if mappable {
            4 << 30
        } else {
            0
        }
    }

    /// Whether a commitment made outside a proof keeps its leaf rows on the
    /// device. Proofs decide through `ExecutionPlan::retain_leaves`, which
    /// also keeps them for hybrid proofs on unified memory.
    pub(crate) fn retain_leaves(&self, mappable: bool) -> bool {
        self.mode.path(mappable, true) == policy::ExecutionPath::Resident
    }
}

impl core::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GpuContext")
            .field("adapter", &self.info.name)
            .field("backend", &self.info.backend)
            .field("flavor", &self.flavor)
            .field("mappable", &self.mappable)
            .field("storage_pool_bytes", &self.storage_pool.bytes())
            .finish()
    }
}

pub(crate) fn storage_layout(
    device: &wgpu::Device,
    label: &str,
    entries: &[(bool, bool)],
) -> wgpu::BindGroupLayout {
    let entries = entries
        .iter()
        .enumerate()
        .map(
            |(binding, &(uniform, read_only))| wgpu::BindGroupLayoutEntry {
                binding: binding as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if uniform {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        )
        .collect::<Vec<_>>();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

pub(crate) fn module(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ShaderModule {
    // Every buffer access in the kernels is guarded by the parameters the host
    // derives from the buffer sizes, so wgpu's injected bounds clamps would
    // only add per-access cost.
    unsafe {
        device.create_shader_module_trusted(
            wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
            },
            wgpu::ShaderRuntimeChecks::unchecked(),
        )
    }
}

pub(crate) fn pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    module: &wgpu::ShaderModule,
    entry: &str,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry),
        bind_group_layouts: &[Some(layout)],
        ..Default::default()
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Optional host wall-clock attribution; untimed callers keep their existing path.
fn timed_optional<T>(
    timing: Option<&mut crate::util::timing::TimingTree>,
    name: &str,
    f: impl FnOnce() -> T,
) -> T {
    match timing {
        Some(timing) => crate::timed!(timing, name, f()),
        None => f(),
    }
}

impl GpuContext {
    /// Creates a context on the best available adapter, with the global
    /// context's settings ([`config`]).
    pub fn new() -> anyhow::Result<Self> {
        Self::with_config(&config())
    }

    /// Creates a context on the best available adapter with `config`.
    pub fn with_config(config: &GpuConfig) -> anyhow::Result<Self> {
        config.validate()?;
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        if let Some(backend) = config.backend {
            descriptor.backends = backend.backends();
        }
        let backends = descriptor.backends;
        let instance = wgpu::Instance::new(descriptor);
        let allow_software = config.allow_software;
        let name = config
            .adapter_name
            .as_deref()
            .unwrap_or_default()
            .to_lowercase();
        let mut adapters = pollster::block_on(instance.enumerate_adapters(backends));
        adapters.retain(|adapter| {
            let info = adapter.get_info();
            (allow_software || info.device_type != wgpu::DeviceType::Cpu)
                && info.name.to_lowercase().contains(&name)
        });
        adapters.sort_by_key(|adapter| match adapter.get_info().device_type {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            wgpu::DeviceType::VirtualGpu => 2,
            wgpu::DeviceType::Other => 3,
            wgpu::DeviceType::Cpu => 4,
        });
        let adapter = adapters.into_iter().next().ok_or_else(|| {
            anyhow::anyhow!(
                "no compute adapter found (software adapters need GpuConfig::allow_software)"
            )
        })?;
        let info = adapter.get_info();
        // The device takes the adapter's own limits. The kernels need its
        // large storage bindings, buffers and dispatches (checked against the
        // device by `fits` and `max_workgroups`), and requesting exactly what
        // the adapter reports never fails, where a wgpu default above an
        // adapter's limit would (iOS Metal reports 15 inter-stage shader
        // variables, below the default of 16).
        let limits = adapter.limits();
        let adapter_features = adapter.features();
        // Direct mapping suits shared memory; discrete GPUs use staging copies.
        let mappable = adapter_features.contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS)
            && info.device_type == wgpu::DeviceType::IntegratedGpu;
        // Tests and profiling builds can exercise the portable arithmetic on
        // adapters that have 64-bit shader integers.
        let portable = cfg!(any(test, feature = "gpu-profiling"))
            && crate::util::switches::text("PLONKY2_GPU_FLAVOR").as_deref() == Some("portable");
        let flavor = if adapter_features.contains(wgpu::Features::SHADER_INT64) && !portable {
            Flavor::Native
        } else {
            Flavor::Portable
        };
        let mut features = wgpu::Features::empty();
        if mappable {
            features |= wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
        }
        if flavor == Flavor::Native {
            features |= wgpu::Features::SHADER_INT64;
        }
        let pass_timeline = cfg!(feature = "gpu-profiling")
            && crate::util::switches::flag("PLONKY2_GPU_PASS_TIMELINE");
        let timestamps = cfg!(feature = "gpu-profiling")
            && crate::util::switches::flag("PLONKY2_GPU_TIMESTAMPS");
        if timestamps || pass_timeline {
            if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY) {
                features |= wgpu::Features::TIMESTAMP_QUERY;
            } else {
                log::warn!("plonky2 gpu: timestamp queries unavailable on this adapter");
            }
        }
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("plonky2"),
                required_features: features,
                required_limits: limits,
                ..Default::default()
            }))?;
        #[cfg(feature = "gpu-profiling")]
        let pass_timeline = pass_timeline
            .then(|| timestamps::PassTimeline::new(&device))
            .flatten();
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);

        let hash_layout = storage_layout(
            &device,
            "hash",
            &[
                (true, true),
                (false, true),
                (false, false),
                (false, false),
                (false, false),
            ],
        );
        let hash_module = module(&device, "plonky2 hash", &shader::hash_module(flavor));
        let hash_leaves = pipeline(&device, &hash_layout, &hash_module, "hash_leaves");
        let hash_nodes = pipeline(&device, &hash_layout, &hash_module, "hash_nodes");
        let permute = pipeline(&device, &hash_layout, &hash_module, "poseidon_permute");

        let fft_layout = storage_layout(
            &device,
            "fft",
            &[
                (true, true),
                (false, true),
                (false, false),
                (false, false),
                (false, true),
                (false, true),
            ],
        );
        let fft_module = module(&device, "plonky2 fft", &shader::fft_module(flavor));
        let fft = pipeline(&device, &fft_layout, &fft_module, "fft_group");

        let fold_layout = storage_layout(
            &device,
            "fold",
            &[(true, true), (false, true), (false, false), (false, true)],
        );
        let fold_module = module(&device, "plonky2 fold", &shader::fold_module(flavor));
        let fold = pipeline(&device, &fold_layout, &fold_module, "fri_fold");

        let open_layout = storage_layout(
            &device,
            "open",
            &[
                (true, true),
                (false, true),
                (false, true),
                (false, true),
                (false, true),
                (false, true),
                (false, false),
                (false, false),
            ],
        );
        let combine_layout = storage_layout(
            &device,
            "combine",
            &[
                (true, true),
                (false, true),
                (false, true),
                (false, true),
                (false, true),
                (false, true),
                (false, true),
                (false, false),
                (false, false),
            ],
        );
        let open_module = module(&device, "plonky2 open", &shader::open_module(flavor));
        let point_powers = pipeline(&device, &open_layout, &open_module, "point_powers");
        let eval_openings = pipeline(&device, &open_layout, &open_module, "eval_openings");
        let combine_module = module(&device, "plonky2 combine", &shader::combine_module(flavor));
        let combine = [
            "batch_powers",
            "reduce_batch",
            "divide_blocks",
            "scan_blocks",
            "combine_quotients",
        ]
        .map(|entry| pipeline(&device, &combine_layout, &combine_module, entry));
        let permutation_layout = storage_layout(
            &device,
            "permutation",
            &[
                (true, true),
                (false, true),
                (false, true),
                (false, true),
                (false, true),
                (false, false),
                (false, false),
            ],
        );
        let permutation_module = module(
            &device,
            "plonky2 permutation",
            &shader::permutation_module(flavor),
        );
        let permutation = [
            "row_products",
            "block_products",
            "block_prefix",
            "fill_columns",
        ]
        .map(|entry| pipeline(&device, &permutation_layout, &permutation_module, entry));

        if let Some(error) = pollster::block_on(validation.pop()) {
            anyhow::bail!("shader or pipeline validation failed: {error}");
        }
        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let dummy_rw = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy rw"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let options = GpuOptions::new(config, mappable);
        Ok(Self {
            max_binding: device
                .limits()
                .max_storage_buffer_binding_size
                .min(device.limits().max_buffer_size),
            max_workgroups: device.limits().max_compute_workgroups_per_dimension,
            device,
            queue,
            info,
            flavor,
            mappable,
            readback_pool: Mutex::new(Vec::new()),
            upload_pool: Mutex::new(Vec::new()),
            storage_pool: Arc::new(StoragePool {
                free: Mutex::new(Vec::new()),
                limit: options.pool_bytes,
            }),
            retained: Arc::new(retained::RetainedBudget::new(options.retained_bytes)),
            options,
            hash_layout,
            hash_leaves,
            hash_nodes,
            permute,
            fft_layout,
            fft,
            fold_layout,
            fold,
            pow_kernel: OnceLock::new(),
            open_layout,
            point_powers,
            eval_openings,
            combine_layout,
            combine,
            permutation_layout,
            permutation,
            dummy: DeviceBuffer::unpooled(dummy),
            dummy_rw: DeviceBuffer::unpooled(dummy_rw),
            twiddles: Mutex::new(HashMap::new()),
            shift_pows: Mutex::new(HashMap::new()),
            quotient_programs: Mutex::new(HashMap::new()),
            quotient_kernels: Mutex::new(HashMap::new()),
            row_tables: Mutex::new(HashMap::new()),
            #[cfg(feature = "gpu-profiling")]
            pass_timeline,
            quotient_cpu_share: Mutex::new(config.quotient_cpu_share),
        })
    }

    /// Whether a buffer of `bytes` can be bound as one storage binding.
    pub(crate) fn fits(&self, bytes: u64) -> bool {
        bytes <= self.max_binding
    }

    pub(crate) fn storage_usage(&self) -> wgpu::BufferUsages {
        let mut usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        if self.mappable {
            usage |= wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE;
        }
        usage
    }

    /// A storage buffer of `bytes` (at least 16), from the pool when one of
    /// that size is free. Pooled buffers retain their contents; callers must
    /// write every element they will read.
    pub(crate) fn storage(&self, label: &str, bytes: u64) -> DeviceBuffer {
        let size = bytes.max(16).next_multiple_of(16);
        let buffer = self.storage_pool.take(size).unwrap_or_else(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: self.storage_usage(),
                mapped_at_creation: false,
            })
        });
        DeviceBuffer(Arc::new(DeviceBufferInner {
            buffer,
            pool: (self.options.pool_bytes > 0).then(|| self.storage_pool.clone()),
        }))
    }

    /// A storage buffer initialized with `words` (not pooled: it is mapped
    /// at creation).
    pub(crate) fn upload(&self, label: &str, words: &[u64]) -> DeviceBuffer {
        let bytes = (words.len() as u64 * 8).max(16).next_multiple_of(16);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: self.storage_usage(),
            mapped_at_creation: true,
        });
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .expect("mapped at creation")
            .slice(..words.len() * 8)
            .copy_from_slice(u64_bytes(words));
        buffer.unmap();
        DeviceBuffer::unpooled(buffer)
    }

    /// Writes `words` at element offset `offset` of `buffer`.
    pub(crate) fn write(&self, buffer: &wgpu::Buffer, offset: u64, words: &[u64]) {
        if words.is_empty() {
            return;
        }
        self.queue
            .write_buffer(buffer, offset * 8, u64_bytes(words));
    }

    /// A uniform buffer holding `words`.
    pub(crate) fn uniform(&self, label: &str, words: &[u32]) -> wgpu::Buffer {
        let mut bytes = words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>();
        bytes.resize(bytes.len().next_multiple_of(16), 0);
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    pub(crate) fn bind(
        &self,
        layout: &wgpu::BindGroupLayout,
        buffers: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let entries = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &entries,
        })
    }

    pub(crate) fn bind_hash(
        &self,
        params: &wgpu::Buffer,
        src: &wgpu::Buffer,
        digests: &wgpu::Buffer,
        cap: &wgpu::Buffer,
        rows: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.bind(&self.hash_layout, &[params, src, digests, cap, rows])
    }

    pub(crate) fn bind_fft(
        &self,
        params: &wgpu::Buffer,
        input: &wgpu::Buffer,
        data: &wgpu::Buffer,
        output: &wgpu::Buffer,
        twiddles: &wgpu::Buffer,
        shift_pows: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.bind(
            &self.fft_layout,
            &[params, input, data, output, twiddles, shift_pows],
        )
    }

    pub(crate) fn bind_open(&self, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
        self.bind(&self.open_layout, buffers)
    }

    pub(crate) fn bind_combine(&self, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
        self.bind(&self.combine_layout, buffers)
    }

    pub(crate) fn bind_permutation(&self, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
        self.bind(&self.permutation_layout, buffers)
    }

    pub(crate) fn bind_fold(
        &self,
        params: &wgpu::Buffer,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        betas: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.bind(&self.fold_layout, &[params, input, output, betas])
    }

    pub(crate) fn encoder(&self, label: &str) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
    }

    /// Begins a compute pass named `label`, with `writes` if the pass has its
    /// own timestamps, otherwise with a pass-timeline pair when recording.
    pub(crate) fn begin_pass<'e>(
        &self,
        encoder: &'e mut wgpu::CommandEncoder,
        label: &str,
        writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) -> wgpu::ComputePass<'e> {
        #[cfg(feature = "gpu-profiling")]
        let slot = match (&writes, &self.pass_timeline) {
            (None, Some(timeline)) => Some(timeline.slot(&self.device, label)),
            _ => None,
        };
        #[cfg(feature = "gpu-profiling")]
        let writes = writes.or(slot
            .as_ref()
            .map(|(query, i)| wgpu::ComputePassTimestampWrites {
                query_set: query,
                beginning_of_pass_write_index: Some(*i),
                end_of_pass_write_index: Some(*i + 1),
            }));
        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: writes,
        })
    }

    /// Logs and clears the pass timeline, if one is recording.
    #[cfg(feature = "gpu-profiling")]
    pub(crate) fn flush_pass_timeline(&self) {
        if let Some(timeline) = &self.pass_timeline {
            timeline.flush(self);
        }
    }

    /// Submits `encoder` and blocks until the device has executed it.
    pub(crate) fn submit(&self, encoder: wgpu::CommandEncoder) {
        let index = self.submit_async(encoder);
        self.wait(&index);
    }

    /// Submits `encoder` without waiting; `wait` or `read_after` take the index.
    pub(crate) fn submit_async(&self, encoder: wgpu::CommandEncoder) -> wgpu::SubmissionIndex {
        let index = self.queue.submit([encoder.finish()]);
        LAST_SUBMISSION.with(|last| *last.borrow_mut() = Some(index.clone()));
        index
    }

    /// Blocks until the device has executed submission `index`.
    pub(crate) fn wait(&self, index: &wgpu::SubmissionIndex) {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index.clone()),
                timeout: None,
            })
            .expect("waiting for the GPU");
    }

    /// Calls `f` with the first `bytes` of `buffer`, after this thread's last submission.
    pub(crate) fn read<R>(
        &self,
        buffer: &wgpu::Buffer,
        bytes: u64,
        f: impl FnOnce(&[u8]) -> R,
    ) -> R {
        self.read_with_timing(buffer, bytes, None, f)
    }

    pub(crate) fn read_with_timing<R>(
        &self,
        buffer: &wgpu::Buffer,
        bytes: u64,
        timing: Option<&mut crate::util::timing::TimingTree>,
        f: impl FnOnce(&[u8]) -> R,
    ) -> R {
        let index = LAST_SUBMISSION.with(|last| last.borrow().clone());
        self.read_after_with_timing(buffer, bytes, index.as_ref(), timing, f)
    }

    /// Calls `f` with the first `bytes` of `buffer` once submission `index`
    /// has executed; later submissions may still be running.
    pub(crate) fn read_after<R>(
        &self,
        buffer: &wgpu::Buffer,
        bytes: u64,
        index: Option<&wgpu::SubmissionIndex>,
        f: impl FnOnce(&[u8]) -> R,
    ) -> R {
        self.read_after_with_timing(buffer, bytes, index, None, f)
    }

    fn read_after_with_timing<R>(
        &self,
        buffer: &wgpu::Buffer,
        bytes: u64,
        index: Option<&wgpu::SubmissionIndex>,
        mut timing: Option<&mut crate::util::timing::TimingTree>,
        f: impl FnOnce(&[u8]) -> R,
    ) -> R {
        if bytes == 0 {
            return f(&[]);
        }
        if self.mappable {
            return self.map_and_read(buffer, bytes, index, timing, f);
        }
        let (staging, copied) = timed_optional(
            timing.as_deref_mut(),
            "gpu: readback staging setup and submit",
            || {
                let staging = self.readback_buffer(bytes);
                let mut encoder = self.encoder("readback");
                encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes.next_multiple_of(16));
                let copied = self.submit_async(encoder);
                (staging, copied)
            },
        );
        let result = self.map_and_read(&staging, bytes, Some(&copied), timing.as_deref_mut(), f);
        timed_optional(timing, "gpu: recycle readback", || {
            self.recycle_readback(staging)
        });
        result
    }

    /// A host-mappable buffer of at least `bytes`, from the readback pool when
    /// one of that size is idle, as the target of a copy read later.
    pub(crate) fn readback_buffer(&self, bytes: u64) -> wgpu::Buffer {
        let size = bytes.next_multiple_of(16);
        let cached = {
            let mut pool = self.readback_pool.lock().unwrap();
            pool.iter()
                .position(|b| b.size() == size)
                .map(|i| pool.swap_remove(i))
        };
        cached.unwrap_or_else(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
    }

    /// A host-mappable upload buffer of at least `bytes`, mapped for writing,
    /// reused across proofs: filling freshly mapped pages costs several times
    /// more than filling a reused mapping. `finish_upload` copies it
    /// into a device buffer and returns it to the pool. Its contents (for the
    /// witness, plaintext wire values) stay in host-visible memory until the
    /// next upload of the same size overwrites them, as with the readback pool.
    pub(crate) fn upload_staging(&self, bytes: u64) -> UploadStaging {
        let size = bytes.max(16).next_multiple_of(16);
        let mapped =
            |s: &UploadStaging| s.state.load(std::sync::atomic::Ordering::Acquire) == UPLOAD_MAPPED;
        loop {
            // Prefer a buffer of this size that is already mapped again.
            let cached = {
                let mut pool = self.upload_pool.lock().unwrap();
                pool.iter()
                    .position(|s| s.buffer.size() == size && mapped(s))
                    .or_else(|| pool.iter().position(|s| s.buffer.size() == size))
                    .map(|i| pool.remove(i))
            };
            let Some(staging) = cached else {
                let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("upload staging"),
                    size,
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: true,
                });
                return UploadStaging {
                    buffer,
                    state: std::sync::Arc::new(std::sync::atomic::AtomicU8::new(UPLOAD_MAPPED)),
                    copied: None,
                };
            };
            // Re-mapped once its last copy has executed: wait for that copy
            // only, not for other work on the queue.
            loop {
                match staging.state.load(std::sync::atomic::Ordering::Acquire) {
                    UPLOAD_MAPPED => return staging,
                    UPLOAD_PENDING => {
                        self.device
                            .poll(wgpu::PollType::Wait {
                                submission_index: staging.copied.clone(),
                                timeout: None,
                            })
                            .expect("waiting for the GPU");
                    }
                    _ => break, // The mapping failed: drop it and take another.
                }
            }
        }
    }

    /// Copies the first `bytes` of `staging` into `dst` in a new submission,
    /// then returns `staging` to the pool, mapping it again for the next
    /// upload once the copy has executed.
    pub(crate) fn finish_upload(
        &self,
        staging: UploadStaging,
        dst: &wgpu::Buffer,
        bytes: u64,
    ) -> wgpu::SubmissionIndex {
        let mut staging = staging;
        staging.buffer.unmap();
        staging
            .state
            .store(UPLOAD_PENDING, std::sync::atomic::Ordering::Release);
        let mut encoder = self.encoder("upload staging copy");
        encoder.copy_buffer_to_buffer(&staging.buffer, 0, dst, 0, bytes);
        let index = self.submit_async(encoder);
        staging.copied = Some(index.clone());
        let state = staging.state.clone();
        staging
            .buffer
            .slice(..)
            .map_async(wgpu::MapMode::Write, move |result| {
                let next = if result.is_ok() {
                    UPLOAD_MAPPED
                } else {
                    UPLOAD_FAILED
                };
                state.store(next, std::sync::atomic::Ordering::Release);
            });
        // Bounded like the readback pool: evict the oldest buffers (zeroed
        // on drop if mapped) to stay within 4 buffers and 1 GiB.
        if staging.buffer.size() <= 1 << 30 {
            let mut pool = self.upload_pool.lock().unwrap();
            while !pool.is_empty()
                && (pool.len() >= 4
                    || pool.iter().map(|b| b.buffer.size()).sum::<u64>() + staging.buffer.size()
                        > 1 << 30)
            {
                pool.remove(0);
            }
            pool.push(staging);
        }
        index
    }

    /// Returns a readback buffer to the pool, evicting the oldest buffers to
    /// stay within 16 buffers and 1 GiB.
    fn recycle_readback(&self, buffer: wgpu::Buffer) {
        if buffer.size() <= 1 << 30 {
            let mut pool = self.readback_pool.lock().unwrap();
            while !pool.is_empty()
                && (pool.len() >= 16
                    || pool.iter().map(|b| b.size()).sum::<u64>() + buffer.size() > 1 << 30)
            {
                pool.remove(0);
            }
            pool.push(buffer);
        }
    }

    /// Calls `f` with the first `bytes` of `staging`, a `readback_buffer`
    /// filled by a copy in submission `index`, then recycles it. Later
    /// submissions may still be running: the mapping waits only for `index`.
    pub(crate) fn read_staged_with_timing<R>(
        &self,
        staging: wgpu::Buffer,
        bytes: u64,
        index: &wgpu::SubmissionIndex,
        mut timing: Option<&mut crate::util::timing::TimingTree>,
        f: impl FnOnce(&[u8]) -> R,
    ) -> R {
        let result = self.map_and_read(&staging, bytes, Some(index), timing.as_deref_mut(), f);
        timed_optional(timing, "gpu: recycle readback", || {
            self.recycle_readback(staging)
        });
        result
    }

    /// Maps `source` for reading once submission `index` has executed, calls
    /// `f` with its first `bytes`, and unmaps it.
    fn map_and_read<R>(
        &self,
        source: &wgpu::Buffer,
        bytes: u64,
        index: Option<&wgpu::SubmissionIndex>,
        mut timing: Option<&mut crate::util::timing::TimingTree>,
        f: impl FnOnce(&[u8]) -> R,
    ) -> R {
        if bytes == 0 {
            return f(&[]);
        }
        let receive = timed_optional(
            timing.as_deref_mut(),
            "gpu: request readback mapping",
            || {
                let (send, receive) = std::sync::mpsc::channel();
                source
                    .slice(..bytes)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        let _ = send.send(result);
                    });
                receive
            },
        );
        // Wait for the given submission rather than for an idle queue, which
        // other threads' submissions could postpone indefinitely.
        timed_optional(
            timing.as_deref_mut(),
            "gpu: wait for readback mapping",
            || {
                self.device
                    .poll(wgpu::PollType::Wait {
                        submission_index: index.cloned(),
                        timeout: None,
                    })
                    .expect("waiting for the GPU");
                receive
                    .recv()
                    .expect("map callback")
                    .expect("mapping a GPU buffer");
            },
        );
        let result = timed_optional(timing.as_deref_mut(), "gpu: decode mapped readback", || {
            let view = source
                .slice(..bytes)
                .get_mapped_range()
                .expect("mapped range");
            f(&view)
        });
        timed_optional(timing, "gpu: unmap and recycle readback", || source.unmap());
        result
    }

    /// Reads `count` words from `buffer`.
    pub(crate) fn read_u64(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<u64> {
        self.read(buffer, count as u64 * 8, |bytes| {
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|chunk| u64::from_le_bytes(*chunk))
                .collect()
        })
    }

    /// The twiddle table of a `2^log_n`-point NTT.
    pub(crate) fn twiddles(&self, log_n: u32) -> DeviceBuffer {
        let mut cache = self.twiddles.lock().unwrap();
        cache
            .entry(log_n)
            .or_insert_with(|| {
                let half = 1usize << log_n.saturating_sub(1);
                let g = GoldilocksField::primitive_root_of_unity(log_n as usize);
                let table = g
                    .powers()
                    .take(half)
                    .map(|x| x.to_canonical_u64())
                    .collect::<Vec<_>>();
                self.upload("twiddles", &table)
            })
            .clone()
    }

    /// `shift^k` for `k < len`.
    pub(crate) fn shift_pows(&self, shift: GoldilocksField, len: u32) -> DeviceBuffer {
        let mut cache = self.shift_pows.lock().unwrap();
        cache
            .entry((shift.to_canonical_u64(), len))
            .or_insert_with(|| {
                let table = shift
                    .powers()
                    .take(len as usize)
                    .map(|x| x.to_canonical_u64())
                    .collect::<Vec<_>>();
                self.upload("shift powers", &table)
            })
            .clone()
    }

    /// Applies the Poseidon permutation to each state.
    pub fn permute_states(&self, states: &[[GoldilocksField; 12]]) -> Vec<[GoldilocksField; 12]> {
        let words = states
            .iter()
            .flatten()
            .map(|x| x.to_noncanonical_u64())
            .collect::<Vec<_>>();
        let src = self.upload("states", &words);
        let out = self.storage("permuted", words.len() as u64 * 8);
        let params = self.uniform(
            "permute params",
            &[states.len() as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        );
        let bindings = self.bind_hash(&params, &src, &out, &self.dummy_rw, &self.dummy_rw);
        let mut encoder = self.encoder("permute");
        {
            let mut pass = self.begin_pass(&mut encoder, "permute", None);
            pass.set_pipeline(&self.permute);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups((states.len() as u32).div_ceil(shader::HASH_WORKGROUP), 1, 1);
        }
        self.submit(encoder);
        self.read_u64(&out, words.len())
            .as_chunks::<12>()
            .0
            .iter()
            .map(|chunk| chunk.map(GoldilocksField::from_canonical_u64))
            .collect()
    }
}

/// A cached device buffer for polynomial coefficients, extended evaluations
/// or sigma values. Proof-local buffers are consumed by subsequent GPU work;
/// fixed circuit data can retain buffers across proofs, subject to
/// [`GpuConfig::retained_bytes`]. Ignored by equality and serialization, and
/// registered so shutdown can release buffers held by application caches.
#[doc(hidden)]
pub struct DeviceCache(Arc<CachedBuffer>);

#[derive(Default)]
struct CachedBuffer {
    entry: Mutex<Option<CacheEntry>>,
}

#[derive(Clone)]
struct CacheEntry {
    buffer: DeviceBuffer,
    // Cloned cache slots share this reservation, counting the allocation once.
    reservation: Option<Arc<retained::Reservation>>,
}

impl CachedBuffer {
    fn take(&self) -> Option<DeviceBuffer> {
        self.entry.lock().unwrap().take().map(|entry| entry.buffer)
    }
}

// Weak entries let shutdown clear buffers even when their circuit is held in
// an application static. Dead entries are pruned on registration.
static DEVICE_CACHES: Mutex<Vec<Weak<CachedBuffer>>> = Mutex::new(Vec::new());

impl DeviceCache {
    fn register(entry: Option<CacheEntry>, caches: &mut Vec<Weak<CachedBuffer>>) -> Self {
        caches.retain(|cache| cache.strong_count() != 0);
        let cache = Arc::new(CachedBuffer {
            entry: Mutex::new(entry),
        });
        caches.push(Arc::downgrade(&cache));
        Self(cache)
    }

    /// Stores a proof-local buffer, outside the persistent circuit-cache budget.
    pub(crate) fn set(&self, buffer: DeviceBuffer) {
        *self.0.entry.lock().unwrap() = Some(CacheEntry {
            buffer,
            reservation: None,
        });
    }

    /// Promotes an outside-proof commitment's buffer to persistent storage,
    /// or clears its cache when the budget cannot accommodate it.
    pub(crate) fn retain_within_limit(&self, ctx: &GpuContext) {
        let mut slot = self.0.entry.lock().unwrap();
        if let Some(entry) = slot.as_mut() {
            if entry.reservation.is_none() {
                entry.reservation = ctx.retained.reserve(entry.buffer.size());
                if entry.reservation.is_none() {
                    *slot = None;
                }
            }
        }
    }

    /// Fills an empty persistent cache if this allocation fits the budget.
    pub(crate) fn set_if_within_limit(&self, ctx: &GpuContext, buffer: DeviceBuffer) {
        let mut slot = self.0.entry.lock().unwrap();
        if slot.is_none() {
            if let Some(reservation) = ctx.retained.reserve(buffer.size()) {
                *slot = Some(CacheEntry {
                    buffer,
                    reservation: Some(reservation),
                });
            }
        }
    }

    /// Removes and returns the buffer.
    pub(crate) fn take(&self) -> Option<DeviceBuffer> {
        self.0.take()
    }

    /// A handle to the buffer, which stays cached.
    pub(crate) fn get(&self) -> Option<DeviceBuffer> {
        self.0
            .entry
            .lock()
            .unwrap()
            .as_ref()
            .map(|entry| entry.buffer.clone())
    }
}

impl Default for DeviceCache {
    fn default() -> Self {
        Self::register(None, &mut DEVICE_CACHES.lock().unwrap())
    }
}

impl Clone for DeviceCache {
    fn clone(&self) -> Self {
        let mut caches = DEVICE_CACHES.lock().unwrap();
        Self::register(self.0.entry.lock().unwrap().clone(), &mut caches)
    }
}

impl core::fmt::Debug for DeviceCache {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "DeviceCache({})", self.0.entry.lock().unwrap().is_some())
    }
}

impl PartialEq for DeviceCache {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for DeviceCache {}

/// The little-endian bytes of `words`.
pub(crate) fn u64_bytes(words: &[u64]) -> &[u8] {
    // SAFETY: u64 has no padding and any byte pattern is a valid u8; the
    // slice covers exactly the words' memory and shares their lifetime.
    unsafe { core::slice::from_raw_parts(words.as_ptr().cast::<u8>(), words.len() * 8) }
}

/// The words of `fields`, in their noncanonical representation.
pub(crate) fn field_words<F: PrimeField64>(fields: &[F]) -> Vec<u64> {
    fields.iter().map(|x| x.to_noncanonical_u64()).collect()
}
