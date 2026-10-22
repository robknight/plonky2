//! Settings of the GPU context, chosen by the application before the first
//! proof (see [`configure`]).

use super::policy::ExecutionMode;

/// A graphics API the context may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Backend {
    Metal,
    Vulkan,
    Dx12,
}

impl Backend {
    pub(crate) fn backends(self) -> wgpu::Backends {
        match self {
            Self::Metal => wgpu::Backends::METAL,
            Self::Vulkan => wgpu::Backends::VULKAN,
            Self::Dx12 => wgpu::Backends::DX12,
        }
    }
}

/// Settings of a GPU context, fixed when the context is created. Start from
/// [`GpuConfig::default`] and change the fields that matter:
///
/// ```ignore
/// let mut config = plonky2::gpu::GpuConfig::default();
/// config.proving_workers = 2;
/// plonky2::gpu::configure(config)?;
/// ```
///
/// [`gpu::set_enabled(false)`](super::set_enabled) before the first proof
/// keeps proving on the CPU without creating a context.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct GpuConfig {
    /// The graphics API; `None` lets wgpu consider every one it supports.
    pub backend: Option<Backend>,
    /// Use only adapters whose name contains this text, ignoring case.
    pub adapter_name: Option<String>,
    /// Accept software adapters (such as Mesa lavapipe), which are useful
    /// only for running the differential tests without a GPU.
    pub allow_software: bool,
    /// Where proofs keep commitment leaves and evaluate the quotient.
    pub mode: ExecutionMode,
    /// Proofs expected to run concurrently, at least 1. Scales the default
    /// `retained_bytes` and damps the hybrid quotient's CPU share adaptation.
    pub proving_workers: usize,
    /// Device memory that circuits' fixed-commitment extended evaluations may
    /// occupy across proofs; beyond it they are recomputed per proof. `None`
    /// is 2 GiB per proving worker.
    pub retained_bytes: Option<u64>,
    /// Idle storage buffers kept for reuse. `None` is 4 GiB on unified-memory
    /// adapters and 0 (no pooling) on discrete ones.
    pub pool_bytes: Option<u64>,
    /// The share of quotient rows the CPU evaluates in a hybrid proof's first
    /// plan, from 0 (GPU only) to 1. With one proving worker each proof then
    /// moves it to its measured balance.
    pub quotient_cpu_share: f64,
    /// Release the global context automatically at normal process exit,
    /// before graphics drivers run their own exit handlers.
    pub shutdown_at_exit: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backend: None,
            adapter_name: None,
            allow_software: false,
            mode: ExecutionMode::Auto,
            proving_workers: 1,
            retained_bytes: None,
            pool_bytes: None,
            quotient_cpu_share: 0.10,
            shutdown_at_exit: true,
        }
    }
}

impl GpuConfig {
    pub(crate) fn validate(&self) -> Result<(), ConfigureError> {
        if self.proving_workers == 0 {
            return Err(ConfigureError::Invalid(
                "proving_workers must be at least 1",
            ));
        }
        if !(0.0..=1.0).contains(&self.quotient_cpu_share) {
            return Err(ConfigureError::Invalid(
                "quotient_cpu_share must be between 0 and 1",
            ));
        }
        Ok(())
    }

    /// The retained-buffer budget this configuration resolves to.
    pub(crate) fn retained_bytes_limit(&self) -> u64 {
        self.retained_bytes
            .unwrap_or_else(|| (2u64 << 30).saturating_mul(self.proving_workers as u64))
    }
}

/// Why [`configure`](super::configure) refused a configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigureError {
    /// The global context already exists or was shut down; its settings are
    /// fixed for the rest of the process.
    AlreadyInitialized,
    /// A setting is out of range.
    Invalid(&'static str),
}

impl core::fmt::Display for ConfigureError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AlreadyInitialized => {
                f.write_str("the GPU context already exists or was shut down")
            }
            Self::Invalid(reason) => write!(f, "invalid GPU configuration: {reason}"),
        }
    }
}

impl std::error::Error for ConfigureError {}
