//! Configuration is resolved once per program, before execution.
use std::path::PathBuf;
use std::sync::Arc;

use super::policy::Policy;
use super::region::buffers::Buffers;
use super::WitnessProgramError;
use crate::util::environment::{Environment, EnvironmentError};

/// Policy when a proof sets a different input shape from the installed program.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum WitnessFallback {
    /// Use the ordinary worklist and report the mismatch once.
    #[default]
    Worklist,
    /// Return an error instead of allowing unexpected worklist latency.
    Error,
}

/// Retention of already-wiped execution buffers. Owned by the prepared program.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum WitnessBuffers {
    /// Release allocations after each execution.
    #[default]
    Fresh,
    /// Reuse allocations up to this payload-capacity budget, across concurrent proofs.
    Reuse { max_bytes: usize },
}

/// Persistent executor-cache policy. Only program data is stored, never witnesses.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum WitnessCache {
    /// Use `PLONKY2_EXECUTOR_CACHE_DIR` when set; otherwise prepare in memory.
    #[default]
    Environment,
    /// Prepare in memory without persistent caching.
    Disabled,
    /// Use this directory independently of the process environment.
    Directory(PathBuf),
}

/// Per-program settings. `Default` is deterministic; [`Self::from_env`] applies
/// process defaults. Explicit configuration passed to a circuit takes precedence.
/// Programs and circuit clones share resources; independently prepared programs
/// can use different settings. Compilation never occurs implicitly in `prove`.
#[derive(Clone)]
#[non_exhaustive]
pub struct WitnessConfig {
    pub fallback: WitnessFallback,
    pub buffers: WitnessBuffers,
    /// CPU sampling, execution, gathering and wiping use this pool size.
    /// Zero inherits the caller's Rayon pool; positive values create an owned pool.
    pub threads: usize,
    /// Application-owned CPU witness pool; takes precedence over `threads`.
    #[cfg(feature = "parallel")]
    pub thread_pool: Option<Arc<rayon::ThreadPool>>,
    pub cache: WitnessCache,
    pub(crate) policy: Policy,
}
impl core::fmt::Debug for WitnessConfig {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut debug = f.debug_struct("WitnessConfig");
        debug
            .field("fallback", &self.fallback)
            .field("buffers", &self.buffers)
            .field("threads", &self.threads)
            .field("cache", &self.cache);
        #[cfg(feature = "parallel")]
        debug.field("thread_pool", &self.thread_pool);
        debug.finish_non_exhaustive()
    }
}
impl Default for WitnessConfig {
    fn default() -> Self {
        Self {
            fallback: WitnessFallback::Worklist,
            buffers: WitnessBuffers::Fresh,
            threads: 0,
            #[cfg(feature = "parallel")]
            thread_pool: None,
            cache: WitnessCache::Environment,
            policy: Policy::default(),
        }
    }
}
pub(super) fn invalid(message: impl Into<String>) -> WitnessProgramError {
    WitnessProgramError::InvalidConfiguration(message.into())
}
impl From<EnvironmentError> for WitnessProgramError {
    fn from(error: EnvironmentError) -> Self {
        invalid(error.to_string())
    }
}
pub(super) fn cache_from_env() -> WitnessCache {
    cache_from_environment(&Environment::process())
}
fn cache_from_environment<F: Fn(&str) -> Option<std::ffi::OsString>>(
    env: &Environment<F>,
) -> WitnessCache {
    env.path("PLONKY2_EXECUTOR_CACHE_DIR")
        .map_or(WitnessCache::Disabled, WitnessCache::Directory)
}
impl WitnessConfig {
    /// Resolve the four operational `PLONKY2_EXECUTOR_*` settings: cache
    /// directory, threads, retained-buffer bytes and fallback. Algorithm tuning
    /// and legacy aliases are not part of the application configuration API.
    pub fn from_env() -> Result<Self, WitnessProgramError> {
        Self::from_env_with_defaults(Self::default())
    }
    /// Apply application environment settings after a library's defaults.
    /// Unset variables preserve defaults. An explicit cache path overrides
    /// the suggested policy (empty disables caching). A thread-count override
    /// selects an owned/inherited pool instead of a suggested pool handle.
    pub fn from_env_with_defaults(defaults: Self) -> Result<Self, WitnessProgramError> {
        Self::from_environment(defaults, &Environment::process())
    }
    fn from_environment<F: Fn(&str) -> Option<std::ffi::OsString>>(
        mut c: Self,
        env: &Environment<F>,
    ) -> Result<Self, WitnessProgramError> {
        if let Some(v) = env.choice(
            "PLONKY2_EXECUTOR_FALLBACK",
            "PLONKY2_EXECUTOR_FALLBACK",
            &[
                ("worklist", WitnessFallback::Worklist),
                ("error", WitnessFallback::Error),
            ],
        )? {
            c.fallback = v;
        }
        if let Some(v) = env.number("PLONKY2_EXECUTOR_THREADS", "PLONKY2_EXECUTOR_THREADS")? {
            c.threads = v;
            #[cfg(feature = "parallel")]
            {
                c.thread_pool = None;
            }
        }
        if let Some(v) = env.number(
            "PLONKY2_EXECUTOR_BUFFER_BYTES",
            "PLONKY2_EXECUTOR_BUFFER_BYTES",
        )? {
            c.buffers = if v == 0 {
                WitnessBuffers::Fresh
            } else {
                WitnessBuffers::Reuse { max_bytes: v }
            };
        }
        if env.present("PLONKY2_EXECUTOR_CACHE_DIR") || c.cache == WitnessCache::Environment {
            c.cache = cache_from_environment(env);
        }
        #[cfg(feature = "witness-benchmarks")]
        c.policy.apply_diagnostics(env)?;
        c.validate()?;
        Ok(c)
    }
    /// Checks options without allocating execution buffers or creating threads.
    /// Preparation and compilation also validate configuration automatically.
    pub fn validate(&self) -> Result<(), WitnessProgramError> {
        if matches!(&self.cache,WitnessCache::Directory(path) if path.as_os_str().is_empty()) {
            return Err(invalid("executor cache directory must not be empty"));
        }
        #[cfg(not(feature = "parallel"))]
        if self.threads != 0 {
            return Err(invalid("executor threads require the parallel feature"));
        }
        self.policy.validate()?;
        if matches!(self.buffers, WitnessBuffers::Reuse { max_bytes: 0 }) {
            return Err(invalid("buffer reuse requires a positive memory budget"));
        }
        Ok(())
    }
    pub(crate) fn same_settings(&self, other: &Self) -> bool {
        #[cfg(feature = "parallel")]
        let pool = self.thread_pool.as_ref().map(Arc::as_ptr)
            == other.thread_pool.as_ref().map(Arc::as_ptr);
        #[cfg(not(feature = "parallel"))]
        let pool = true;
        self.policy == other.policy
            && self.fallback == other.fallback
            && self.buffers == other.buffers
            && self.threads == other.threads
            && pool
            && self.cache == other.cache
    }

    pub(crate) fn compilation_key(&self) -> u32 {
        self.policy.compilation_key()
    }
    pub(crate) fn with_compilation_key(mut self, key: u32) -> Result<Self, WitnessProgramError> {
        self.policy = self.policy.with_compilation_key(key)?;
        self.validate()?;
        Ok(self)
    }
}

#[derive(Debug)]
pub(crate) struct Runtime {
    #[cfg(feature = "parallel")]
    pub schedule: super::region::schedule::PreparedSchedule,
    pub config: WitnessConfig,
    pub buffers: Buffers,
    #[cfg(feature = "parallel")]
    pub pool: Option<Arc<rayon::ThreadPool>>,
}
impl Runtime {
    #[inline]
    pub(crate) fn run<R: Send>(&self, task: impl FnOnce() -> R + Send) -> R {
        #[cfg(feature = "parallel")]
        if let Some(pool) = &self.pool {
            return pool.install(task);
        }
        task()
    }
    #[cfg(test)]
    pub(crate) fn assert_active_pool(&self) {
        #[cfg(feature = "parallel")]
        if let Some(pool) = &self.pool {
            assert!(
                pool.current_thread_index().is_some(),
                "witness CPU work escaped its configured pool"
            );
        }
    }
    pub fn new(config: WitnessConfig) -> Result<Self, WitnessProgramError> {
        config.validate()?;
        #[cfg(feature = "parallel")]
        let pool = match &config.thread_pool {
            Some(pool) => Some(pool.clone()),
            None if config.threads > 0 => Some(Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(config.threads)
                    .build()
                    .map_err(|e| invalid(format!("witness thread pool: {e}")))?,
            )),
            None => None,
        };
        Ok(Self {
            #[cfg(feature = "parallel")]
            schedule: Default::default(),
            buffers: Buffers::new(config.buffers),
            config,
            #[cfg(feature = "parallel")]
            pool,
        })
    }
}
impl Default for Runtime {
    fn default() -> Self {
        Self::new(WitnessConfig::default()).expect("valid default witness configuration")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn read(
        defaults: WitnessConfig,
        values: &[(&str, &str)],
    ) -> Result<WitnessConfig, WitnessProgramError> {
        let values: std::collections::BTreeMap<_, std::ffi::OsString> =
            values.iter().map(|(k, v)| (*k, (*v).into())).collect();
        WitnessConfig::from_environment(
            defaults,
            &Environment::with_lookup(|key: &str| values.get(key).cloned()),
        )
    }
    #[test]
    fn environment_overrides_library_resource_defaults() {
        let defaults = WitnessConfig {
            fallback: WitnessFallback::Error,
            cache: WitnessCache::Directory("library-cache".into()),
            buffers: WitnessBuffers::Reuse { max_bytes: 8192 },
            ..Default::default()
        };
        assert!(read(defaults.clone(), &[])
            .unwrap()
            .same_settings(&defaults));
        let c = read(
            defaults,
            &[
                ("PLONKY2_EXECUTOR_FALLBACK", "worklist"),
                ("PLONKY2_EXECUTOR_CACHE_DIR", ""),
                ("PLONKY2_EXECUTOR_BUFFER_BYTES", "0"),
            ],
        )
        .unwrap();
        assert_eq!(c.fallback, WitnessFallback::Worklist);
        assert_eq!(c.cache, WitnessCache::Disabled);
        assert_eq!(c.buffers, WitnessBuffers::Fresh);
        assert_eq!(
            read(
                WitnessConfig {
                    cache: WitnessCache::Disabled,
                    ..Default::default()
                },
                &[("PLONKY2_EXECUTOR_CACHE_DIR", "app-cache")]
            )
            .unwrap()
            .cache,
            WitnessCache::Directory("app-cache".into())
        );
        for (key, value) in [
            ("PLONKY2_EXECUTOR_FALLBACK", "bad"),
            ("PLONKY2_EXECUTOR_THREADS", "bad"),
            ("PLONKY2_EXECUTOR_BUFFER_BYTES", "-1"),
        ] {
            assert!(read(WitnessConfig::default(), &[(key, value)])
                .unwrap_err()
                .to_string()
                .contains(key));
        }
    }
    #[test]
    #[cfg(feature = "parallel")]
    fn environment_thread_count_replaces_library_suggested_pool() {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap(),
        );
        let defaults = WitnessConfig {
            thread_pool: Some(pool.clone()),
            ..Default::default()
        };
        assert!(Arc::ptr_eq(
            read(defaults.clone(), &[])
                .unwrap()
                .thread_pool
                .as_ref()
                .unwrap(),
            &pool
        ));
        for count in ["0", "1"] {
            let c = read(defaults.clone(), &[("PLONKY2_EXECUTOR_THREADS", count)]).unwrap();
            assert!(c.thread_pool.is_none());
            assert_eq!(c.threads.to_string(), count);
        }
    }
    #[test]
    fn retired_tuning_and_legacy_names_have_no_effect() {
        let names = [
            "PLONKY2_EXECUTOR_LAYOUT",
            "PLONKY2_EXECUTOR_CHAIN_LENGTH",
            "PLONKY2_EXECUTOR_CHUNK_SCALE",
            "PLONKY2_EXECUTOR_SIMPLIFY_ARITHMETIC",
            "PLONKY2_EXECUTOR_GATHER_COLUMNS",
            "PLONKY2_EXECUTOR_GATHER_ROWS",
            "PLONKY2_EXECUTOR_MODE",
            "PLONKY2_EXECUTOR_SCHEDULE",
            "PLONKY2_EXECUTOR_POSEIDON",
            "PLONKY2_EXECUTOR_GATHER",
            "P2_AOT_LAYOUT",
            "P2_AOT_CHAIN_LEN",
            "P2_AOT_FUSE",
            "P2_AOT_MODE",
            "P2_POSEIDON_MODE",
            "P2_AOT_GPU_GATHER",
            "P2_AOT_WITNESS_THREADS",
            "P2_WITNESS_BUFFERS",
            "P2_AOT_SIMPLIFY_ARITH",
        ];
        let values: Vec<_> = names.iter().map(|&name| (name, "bad")).collect();
        let defaults = WitnessConfig {
            cache: WitnessCache::Disabled,
            ..Default::default()
        };
        let c = read(defaults.clone(), &values).unwrap();
        assert!(c.same_settings(&defaults));
        let public = format!("{c:?}");
        for private in [
            "policy",
            "layout",
            "chain_length",
            "schedule",
            "poseidon",
            "gather",
        ] {
            assert!(!public.contains(private), "{public}");
        }
    }
    #[test]
    fn diagnostic_environment_is_feature_gated() {
        let values = [
            ("PLONKY2_WITNESS_BENCH_MODE", "checked"),
            ("PLONKY2_WITNESS_BENCH_SCHEDULE", "layers"),
            ("PLONKY2_WITNESS_BENCH_POSEIDON", "scalar"),
            ("PLONKY2_WITNESS_BENCH_GATHER", "gpu"),
        ];
        let c = read(WitnessConfig::default(), &values).unwrap();
        #[cfg(feature = "witness-benchmarks")]
        {
            assert_eq!(
                c.policy.execution,
                super::super::policy::WitnessExecution::Checked
            );
            assert_eq!(
                c.policy.schedule,
                super::super::policy::WitnessSchedule::Layers
            );
            assert_eq!(
                c.policy.poseidon,
                super::super::policy::WitnessPoseidon::Scalar
            );
            assert_eq!(c.policy.gather, super::super::policy::WitnessGather::Gpu);
            assert!(read(
                WitnessConfig::default(),
                &[("PLONKY2_WITNESS_BENCH_MODE", "bad")]
            )
            .is_err());
        }
        #[cfg(not(feature = "witness-benchmarks"))]
        {
            assert_eq!(c.policy, Policy::default());
            assert!(read(
                WitnessConfig::default(),
                &[("PLONKY2_WITNESS_BENCH_MODE", "bad")]
            )
            .is_ok());
        }
    }
    #[test]
    fn validates_application_resources_before_allocation() {
        let options = WitnessConfig::default();
        options.validate().unwrap();
        assert!(WitnessConfig {
            cache: WitnessCache::Directory(PathBuf::new()),
            ..options.clone()
        }
        .validate()
        .is_err());
        assert!(WitnessConfig {
            buffers: WitnessBuffers::Reuse { max_bytes: 0 },
            ..options
        }
        .validate()
        .is_err());
    }
    #[test]
    #[cfg(feature = "parallel")]
    fn environment_is_captured_without_mutating_the_test_process() {
        const CHILD: &str = "PLONKY2_EXECUTOR_CONFIG_TEST_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let config = WitnessConfig::from_env().unwrap();
            assert_eq!(config.threads, 2);
            assert_eq!(config.buffers, WitnessBuffers::Reuse { max_bytes: 4096 });
            assert_eq!(config.fallback, WitnessFallback::Error);
            assert_eq!(config.cache, WitnessCache::Disabled);
            return;
        }
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "plonk::witness_program::config::tests::environment_is_captured_without_mutating_the_test_process"])
            .env(CHILD,"1").env("PLONKY2_EXECUTOR_THREADS","2").env("PLONKY2_EXECUTOR_BUFFER_BYTES","4096")
            .env("PLONKY2_EXECUTOR_FALLBACK","error").env_remove("PLONKY2_EXECUTOR_CACHE_DIR").status().unwrap();
        assert!(status.success());
    }
}
