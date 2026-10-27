//! Typed GPU runtime settings, read from the environment on first use unless
//! the application supplies explicit settings through [`configure`](super::configure).

use super::policy::{ExecutionMode, WitnessPlacement};
use crate::util::environment::{Environment, EnvironmentError};

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
/// [`GpuConfig::from_env`] and override fields that matter to the application:
///
/// ```ignore
/// let mut config = plonky2::gpu::GpuConfig::from_env()?;
/// config.proving_workers = 2;
/// plonky2::gpu::configure(config)?;
/// ```
///
/// [`gpu::set_enabled(false)`](super::set_enabled) before the first proof
/// keeps proving on the CPU without creating a context.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct GpuConfig {
    /// Enable GPU routing at startup. False avoids creating a context.
    pub enabled: bool,
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
    /// Shared budget for fixed commitments, sigma values, quotient bytecode
    /// and constant tables retained across proofs. Context caches evict their
    /// least recently used entries; buffers that do not fit are recomputed or
    /// uploaded per proof. Each context cache also holds at most 128 entries.
    /// `None` is 2 GiB per proving worker.
    pub retained_bytes: Option<u64>,
    /// Idle storage buffers kept for reuse. `None` is 4 GiB on unified-memory
    /// adapters and 0 (no pooling) on discrete ones.
    pub pool_bytes: Option<u64>,
    /// Idle storage buffers are released once they have stayed unused while
    /// this many later proofs started. `None` keeps them until `pool_bytes`
    /// evicts them.
    pub pool_max_idle_proofs: Option<u64>,
    /// Proofs of circuits with fewer than `2^pool_min_degree_bits` rows
    /// neither take buffers from the pool nor return them. The most recently
    /// started proof decides for all proofs running at that time. `None` is 0.
    pub pool_min_degree_bits: Option<u32>,
    /// The share of quotient rows the CPU evaluates in a hybrid proof's first
    /// plan, from 0 (GPU only) to 1. With one proving worker each proof then
    /// moves it to its measured balance.
    pub quotient_cpu_share: f64,
    /// Release the global context automatically at normal process exit,
    /// before graphics drivers run their own exit handlers.
    pub shutdown_at_exit: bool,
    /// Tiles of 1024 elements each NTT workgroup loops over, prefetching the
    /// next tile while it transforms the current one. Zero dispatches one
    /// workgroup per tile without prefetching. `None` is 8 on unified-memory
    /// adapters, where it was measured, and 0 on discrete ones.
    pub fft_tiles_per_workgroup: Option<u32>,
    /// Whether witness work after the CPU executor runs on the device.
    pub witness: WitnessPlacement,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            backend: None,
            adapter_name: None,
            allow_software: false,
            mode: ExecutionMode::Auto,
            proving_workers: 1,
            retained_bytes: None,
            pool_bytes: None,
            pool_max_idle_proofs: None,
            pool_min_degree_bits: None,
            quotient_cpu_share: 0.10,
            shutdown_at_exit: true,
            fft_tiles_per_workgroup: None,
            witness: WitnessPlacement::Auto,
        }
    }
}

impl GpuConfig {
    /// Read and validate documented `PLONKY2_GPU_*` process defaults. No device
    /// is initialized. Explicit `configure`/`with_config` calls do not read the
    /// environment; start from this method to override only selected fields.
    pub fn from_env() -> Result<Self, ConfigureError> {
        Self::from_env_with_defaults(Self::default())
    }

    /// Apply application environment overrides to library-supplied defaults.
    /// Unset variables preserve those defaults; explicit empty adapter text
    /// clears a filter, and `auto` restores automatic budget sizing.
    pub fn from_env_with_defaults(defaults: Self) -> Result<Self, ConfigureError> {
        Self::from_environment(defaults, &Environment::process())
    }

    fn from_environment<F: Fn(&str) -> Option<std::ffi::OsString>>(
        mut c: Self,
        env: &Environment<F>,
    ) -> Result<Self, ConfigureError> {
        if let Some(v) = env.boolean("PLONKY2_GPU_ENABLED", "PLONKY2_GPU_ENABLED")? {
            c.enabled = v;
        }
        if let Some(v) = env.choice(
            "PLONKY2_GPU_BACKEND",
            "PLONKY2_GPU_BACKEND",
            &[
                ("auto", None),
                ("metal", Some(Backend::Metal)),
                ("vulkan", Some(Backend::Vulkan)),
                ("dx12", Some(Backend::Dx12)),
            ],
        )? {
            c.backend = v;
        }
        if let Some(name) = env.text("PLONKY2_GPU_ADAPTER_NAME", "PLONKY2_GPU_ADAPTER_NAME")? {
            c.adapter_name = (!name.is_empty()).then_some(name);
        }
        if let Some(v) = env.boolean("PLONKY2_GPU_ALLOW_SOFTWARE", "PLONKY2_GPU_ALLOW_SOFTWARE")? {
            c.allow_software = v;
        }
        if let Some(v) = env.choice(
            "PLONKY2_GPU_MODE",
            "PLONKY2_GPU_MODE",
            &[
                ("auto", ExecutionMode::Auto),
                ("resident", ExecutionMode::Resident),
                ("hybrid", ExecutionMode::Hybrid),
            ],
        )? {
            c.mode = v;
        }
        if let Some(v) = env.number("PLONKY2_GPU_PROVING_WORKERS", "PLONKY2_GPU_PROVING_WORKERS")? {
            if v == 0 {
                return Err(EnvironmentError {
                    variable: "PLONKY2_GPU_PROVING_WORKERS",
                    expected: "an integer of at least 1".into(),
                }
                .into());
            }
            c.proving_workers = v;
        }
        if let Some(value) =
            env.optional_number("PLONKY2_GPU_RETAINED_BYTES", "PLONKY2_GPU_RETAINED_BYTES")?
        {
            c.retained_bytes = value;
        }
        if let Some(value) =
            env.optional_number("PLONKY2_GPU_POOL_BYTES", "PLONKY2_GPU_POOL_BYTES")?
        {
            c.pool_bytes = value;
        }
        if let Some(value) = env.optional_number(
            "PLONKY2_GPU_POOL_MAX_IDLE_PROOFS",
            "PLONKY2_GPU_POOL_MAX_IDLE_PROOFS",
        )? {
            c.pool_max_idle_proofs = value;
        }
        if let Some(value) = env.optional_number(
            "PLONKY2_GPU_POOL_MIN_DEGREE_BITS",
            "PLONKY2_GPU_POOL_MIN_DEGREE_BITS",
        )? {
            c.pool_min_degree_bits = value;
        }
        if let Some(v) = env.number(
            "PLONKY2_GPU_QUOTIENT_CPU_SHARE",
            "PLONKY2_GPU_QUOTIENT_CPU_SHARE",
        )? {
            if !(0.0..=1.0).contains(&v) {
                return Err(EnvironmentError {
                    variable: "PLONKY2_GPU_QUOTIENT_CPU_SHARE",
                    expected: "a finite fraction between 0 and 1".into(),
                }
                .into());
            }
            c.quotient_cpu_share = v;
        }
        if let Some(v) = env.boolean(
            "PLONKY2_GPU_SHUTDOWN_AT_EXIT",
            "PLONKY2_GPU_SHUTDOWN_AT_EXIT",
        )? {
            c.shutdown_at_exit = v;
        }
        if let Some(value) = env.optional_number(
            "PLONKY2_GPU_FFT_TILES_PER_WORKGROUP",
            "PLONKY2_GPU_FFT_TILES_PER_WORKGROUP",
        )? {
            c.fft_tiles_per_workgroup = value;
        }
        if let Some(v) = env.choice(
            "PLONKY2_GPU_WITNESS",
            "PLONKY2_GPU_WITNESS",
            &[
                ("auto", WitnessPlacement::Auto),
                ("host", WitnessPlacement::Host),
                ("device", WitnessPlacement::Device),
            ],
        )? {
            c.witness = v;
        }
        c.validate()?;
        Ok(c)
    }
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
    /// A process default could not be decoded or is out of range.
    Environment(EnvironmentError),
    /// The global context already exists or was shut down; its settings are
    /// fixed for the rest of the process.
    AlreadyInitialized,
    /// A setting is out of range.
    Invalid(&'static str),
}

impl core::fmt::Display for ConfigureError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Environment(error) => write!(f, "invalid GPU environment: {error}"),
            Self::AlreadyInitialized => {
                f.write_str("the GPU context already exists or was shut down")
            }
            Self::Invalid(reason) => write!(f, "invalid GPU configuration: {reason}"),
        }
    }
}

impl From<EnvironmentError> for ConfigureError {
    fn from(error: EnvironmentError) -> Self {
        Self::Environment(error)
    }
}
impl std::error::Error for ConfigureError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Environment(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::ffi::OsString;

    use super::*;
    fn read(values: &[(&str, &str)]) -> Result<GpuConfig, ConfigureError> {
        read_defaults(GpuConfig::default(), values)
    }
    fn read_defaults(
        defaults: GpuConfig,
        values: &[(&str, &str)],
    ) -> Result<GpuConfig, ConfigureError> {
        let values: BTreeMap<_, _> = values
            .iter()
            .map(|(k, v)| (*k, OsString::from(v)))
            .collect();
        GpuConfig::from_environment(
            defaults,
            &Environment::with_lookup(|key: &str| values.get(key).cloned()),
        )
    }
    #[test]
    fn application_environment_overrides_library_defaults_without_erasing_unset_fields() {
        let defaults = GpuConfig {
            enabled: false,
            backend: Some(Backend::Vulkan),
            adapter_name: Some("suggested".into()),
            allow_software: true,
            mode: ExecutionMode::Hybrid,
            proving_workers: 7,
            retained_bytes: Some(1024),
            pool_bytes: Some(2048),
            pool_max_idle_proofs: Some(3),
            pool_min_degree_bits: Some(12),
            quotient_cpu_share: 0.3,
            shutdown_at_exit: false,
            fft_tiles_per_workgroup: Some(5),
            witness: WitnessPlacement::Host,
        };
        assert_eq!(read_defaults(defaults.clone(), &[]).unwrap(), defaults);
        let c = read_defaults(
            defaults.clone(),
            &[
                ("PLONKY2_GPU_ENABLED", "true"),
                ("PLONKY2_GPU_BACKEND", "auto"),
                ("PLONKY2_GPU_ADAPTER_NAME", ""),
                ("PLONKY2_GPU_PROVING_WORKERS", "2"),
                ("PLONKY2_GPU_RETAINED_BYTES", "auto"),
                ("PLONKY2_GPU_POOL_BYTES", "0"),
                ("PLONKY2_GPU_POOL_MAX_IDLE_PROOFS", "1"),
                ("PLONKY2_GPU_POOL_MIN_DEGREE_BITS", "auto"),
            ],
        )
        .unwrap();
        assert!(c.enabled);
        assert_eq!(c.backend, None);
        assert_eq!(c.adapter_name, None);
        assert_eq!(c.proving_workers, 2);
        assert_eq!(c.retained_bytes, None);
        assert_eq!(c.pool_bytes, Some(0));
        assert_eq!(c.pool_max_idle_proofs, Some(1));
        assert_eq!(c.pool_min_degree_bits, None);
        assert_eq!(c.retained_bytes_limit(), 4 << 30);
        assert_eq!(c.mode, defaults.mode);
        assert_eq!(c.quotient_cpu_share, defaults.quotient_cpu_share);
        assert_eq!(c.allow_software, defaults.allow_software);
        assert_eq!(c.shutdown_at_exit, defaults.shutdown_at_exit);
        assert_eq!(c.fft_tiles_per_workgroup, Some(5));
        assert_eq!(c.witness, WitnessPlacement::Host);
        assert_eq!(
            read_defaults(
                defaults.clone(),
                &[("PLONKY2_GPU_FFT_TILES_PER_WORKGROUP", "auto")]
            )
            .unwrap()
            .fft_tiles_per_workgroup,
            None
        );
        assert_eq!(
            read_defaults(defaults.clone(), &[("PLONKY2_GPU_POOL_BYTES", "auto")])
                .unwrap()
                .pool_bytes,
            None
        );
        assert!(read_defaults(defaults, &[("PLONKY2_GPU_PROVING_WORKERS", "bad")]).is_err());
    }
    #[test]
    fn typed_environment_controls_context_selection_and_budgets() {
        assert_eq!(read(&[]).unwrap(), GpuConfig::default());
        let c = read(&[
            ("PLONKY2_GPU_ENABLED", "false"),
            ("PLONKY2_GPU_BACKEND", "vulkan"),
            ("PLONKY2_GPU_ADAPTER_NAME", "A40"),
            ("PLONKY2_GPU_ALLOW_SOFTWARE", "1"),
            ("PLONKY2_GPU_MODE", "resident"),
            ("PLONKY2_GPU_PROVING_WORKERS", "3"),
            ("PLONKY2_GPU_RETAINED_BYTES", "0"),
            ("PLONKY2_GPU_POOL_BYTES", "4096"),
            ("PLONKY2_GPU_QUOTIENT_CPU_SHARE", "0.25"),
            ("PLONKY2_GPU_SHUTDOWN_AT_EXIT", "0"),
            ("PLONKY2_GPU_FFT_TILES_PER_WORKGROUP", "0"),
            ("PLONKY2_GPU_WITNESS", "device"),
        ])
        .unwrap();
        assert!(!c.enabled && c.allow_software && !c.shutdown_at_exit);
        assert_eq!(c.backend, Some(Backend::Vulkan));
        assert_eq!(c.mode, ExecutionMode::Resident);
        assert_eq!(c.adapter_name.as_deref(), Some("A40"));
        assert_eq!(c.proving_workers, 3);
        assert_eq!(c.retained_bytes, Some(0));
        assert_eq!(c.pool_bytes, Some(4096));
        assert_eq!(c.quotient_cpu_share, 0.25);
        assert_eq!(c.fft_tiles_per_workgroup, Some(0));
        assert_eq!(c.witness, WitnessPlacement::Device);
        assert!(read(&[("PLONKY2_GPU_FFT_TILES_PER_WORKGROUP", "-1")]).is_err());
        assert!(read(&[("PLONKY2_GPU_WITNESS", "gpu")]).is_err());
        let c = read(&[("PLONKY2_GPU_PROVING_WORKERS", "3")]).unwrap();
        assert_eq!(c.retained_bytes_limit(), 6 << 30);
    }
    #[test]
    fn witness_placement_follows_memory_type() {
        // (setting, on unified memory, on discrete memory)
        for (placement, unified, discrete) in [
            (WitnessPlacement::Auto, false, true),
            (WitnessPlacement::Host, false, false),
            (WitnessPlacement::Device, true, true),
        ] {
            assert_eq!(placement.on_device(true), unified, "{placement:?}");
            assert_eq!(placement.on_device(false), discrete, "{placement:?}");
        }
    }
    /// Isolated children exercise real process defaults without racing GPU tests.
    #[test]
    fn global_environment_is_captured_and_explicit_settings_override_it() {
        use crate::gpu::{config, configure, configure_defaults, context};
        const CHILD: &str = "PLONKY2_GPU_ENV_TEST_CHILD";
        if let Ok(case) = std::env::var(CHILD) {
            match case.as_str() {
                "defaults" => {
                    assert!(configure_defaults(GpuConfig {
                        proving_workers: 7,
                        pool_bytes: Some(99),
                        ..Default::default()
                    })
                    .unwrap());
                    assert!(!configure_defaults(GpuConfig::default()).unwrap());
                    let c = config().unwrap();
                    assert_eq!(c.proving_workers, 3);
                    assert_eq!(c.pool_bytes, Some(0));
                    assert!(!c.enabled);
                }
                "application" => {
                    configure(GpuConfig {
                        enabled: false,
                        proving_workers: 2,
                        pool_bytes: Some(77),
                        ..Default::default()
                    })
                    .unwrap();
                    assert!(!configure_defaults(GpuConfig::default()).unwrap());
                    assert_eq!(config().unwrap().proving_workers, 2);
                    assert_eq!(config().unwrap().pool_bytes, Some(77));
                }
                "late_defaults" => {
                    let c = config().unwrap();
                    assert!(!configure_defaults(GpuConfig::default()).unwrap());
                    assert_eq!(config().unwrap(), c);
                }
                "capture" => {
                    let c = config().unwrap();
                    assert_eq!(c.proving_workers, 3);
                    assert!(!c.enabled);
                    std::env::set_var("PLONKY2_GPU_PROVING_WORKERS", "7");
                    assert_eq!(config().unwrap(), c);
                }
                "override" => {
                    assert!(config().is_err());
                    configure(GpuConfig {
                        enabled: false,
                        proving_workers: 2,
                        ..Default::default()
                    })
                    .unwrap();
                    assert_eq!(config().unwrap().proving_workers, 2);
                }
                "invalid" => {
                    assert!(context().is_none());
                    assert!(matches!(config(), Err(ConfigureError::Environment(_))));
                    std::env::set_var("PLONKY2_GPU_PROVING_WORKERS", "2");
                    assert!(config().is_err());
                    assert_eq!(
                        configure(GpuConfig::default()),
                        Err(ConfigureError::AlreadyInitialized)
                    );
                    return;
                }
                _ => panic!("unexpected test case"),
            }
            assert!(context().is_none());
            assert_eq!(
                configure(GpuConfig::default()),
                Err(ConfigureError::AlreadyInitialized)
            );
            return;
        }
        for case in [
            "capture",
            "override",
            "invalid",
            "defaults",
            "application",
            "late_defaults",
        ] {
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command.args(["--exact", "gpu::config::tests::global_environment_is_captured_and_explicit_settings_override_it"])
                .env(CHILD, case).env("PLONKY2_GPU_ENABLED", "false")
                .env("PLONKY2_GPU_BACKEND", "auto").env("PLONKY2_GPU_ADAPTER_NAME", "")
                .env("PLONKY2_GPU_ALLOW_SOFTWARE", "false").env("PLONKY2_GPU_MODE", "auto")
                .env("PLONKY2_GPU_PROVING_WORKERS", if matches!(case, "override" | "invalid") { "invalid" } else { "3" })
                .env("PLONKY2_GPU_RETAINED_BYTES", "0").env("PLONKY2_GPU_POOL_BYTES", "0")
                .env("PLONKY2_GPU_QUOTIENT_CPU_SHARE", "0.1").env("PLONKY2_GPU_SHUTDOWN_AT_EXIT", "true");
            assert!(command.status().unwrap().success(), "{case}");
        }
    }
    #[test]
    fn malformed_environment_is_an_error_with_the_variable_name() {
        for (key, value) in [
            ("PLONKY2_GPU_BACKEND", "cuda"),
            ("PLONKY2_GPU_MODE", "cpu"),
            ("PLONKY2_GPU_ENABLED", "yes"),
            ("PLONKY2_GPU_ENABLED", ""),
            ("PLONKY2_GPU_PROVING_WORKERS", "0"),
            ("PLONKY2_GPU_POOL_BYTES", "-1"),
            ("PLONKY2_GPU_RETAINED_BYTES", "18446744073709551616"),
            ("PLONKY2_GPU_QUOTIENT_CPU_SHARE", "NaN"),
            ("PLONKY2_GPU_QUOTIENT_CPU_SHARE", "1.1"),
        ] {
            let error = read(&[(key, value)]).unwrap_err();
            assert!(matches!(error, ConfigureError::Environment(_)));
            assert!(error.to_string().contains(key), "{error}");
        }
    }
}
