//! Configuration is resolved once per program, before execution.
use std::path::PathBuf;
use std::sync::Arc;

use super::region::buffers::Buffers;
use super::WitnessProgramError;

/// Placement of witness values. `Outputs` is the measured general default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum WitnessLayout {
    /// Keep the compiler's original canonical ordering.
    Canonical,
    /// Place each operation's outputs together.
    #[default]
    Outputs,
    /// Sort non-root values by copy-constraint representative.
    Representatives,
    /// Interleave outputs in blocks of four operations.
    Block4,
    /// Interleave outputs in blocks of eight operations.
    Block8,
}
impl WitnessLayout {
    pub(crate) fn code(self) -> u32 {
        match self {
            Self::Canonical => 0,
            Self::Outputs => 1,
            Self::Representatives => 2,
            Self::Block4 => 3,
            Self::Block8 => 4,
        }
    }
    fn from_code(code: u32) -> Result<Self, WitnessProgramError> {
        match code {
            0 => Ok(Self::Canonical),
            1 => Ok(Self::Outputs),
            2 => Ok(Self::Representatives),
            3 => Ok(Self::Block4),
            4 => Ok(Self::Block8),
            _ => Err(invalid("invalid stored witness layout")),
        }
    }
}

/// Execution policy. Automatic execution retains checks for aliases and inputs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum WitnessExecution {
    /// Use validated shared/native kernels, with generic operation fallbacks.
    #[default]
    Auto,
    /// Use the checked interpreter, useful as a control or for diagnosis.
    Checked,
}

/// Poseidon selection, independent of the serialized circuit program.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum WitnessPoseidon {
    /// Batch independent permutations when the compiled backend supports it.
    #[default]
    Auto,
    /// Use individual permutations.
    Scalar,
}

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

/// Destination for the final witness gather.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum WitnessGather {
    /// Use the recommended CPU gather, including mapped GPU staging when applicable.
    #[default]
    Auto,
    /// Gather on the CPU.
    Cpu,
    /// Request GPU gathering when GPU proving and a validated plan are available.
    Gpu,
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
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct WitnessConfig {
    pub layout: WitnessLayout,
    /// Maximum fused arithmetic-chain length: zero disables fusion; otherwise 2..=8.
    pub chain_length: u8,
    pub execution: WitnessExecution,
    pub poseidon: WitnessPoseidon,
    pub fallback: WitnessFallback,
    pub buffers: WitnessBuffers,
    pub gather: WitnessGather,
    /// CPU witness sampling, execution, gathering and wiping use this pool.
    /// Zero inherits the current Rayon pool; positive values create an owned pool.
    /// Circuit compilation/preparation and disk I/O use the caller's context.
    pub threads: usize,
    /// A caller-owned pool, shared across programs. Takes precedence over `threads`.
    #[cfg(feature = "parallel")]
    pub thread_pool: Option<Arc<rayon::ThreadPool>>,
    pub cache: WitnessCache,
    /// Advanced execution tuning. One uses the standard operation chunk sizes.
    pub chunk_scale: u32,
    pub simplify_arithmetic: bool,
    pub gather_columns: usize,
    pub gather_rows: usize,
}
impl Default for WitnessConfig {
    fn default() -> Self {
        Self {
            layout: WitnessLayout::Outputs,
            chain_length: 4,
            execution: WitnessExecution::Auto,
            poseidon: WitnessPoseidon::Auto,
            fallback: WitnessFallback::Worklist,
            buffers: WitnessBuffers::Fresh,
            gather: WitnessGather::Auto,
            threads: 0,
            #[cfg(feature = "parallel")]
            thread_pool: None,
            cache: WitnessCache::Environment,
            chunk_scale: 1,
            simplify_arithmetic: true,
            gather_columns: 16,
            gather_rows: 256,
        }
    }
}
fn invalid(message: impl Into<String>) -> WitnessProgramError {
    WitnessProgramError::InvalidConfiguration(message.into())
}
fn setting(name: &str, legacy: &str) -> Result<Option<String>, WitnessProgramError> {
    let key = if std::env::var_os(name).is_some() {
        name
    } else {
        legacy
    };
    match std::env::var(key) {
        Ok(v) => Ok(Some(v)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(_) => Err(invalid(format!("{key} must be Unicode"))),
    }
}
fn number<T: std::str::FromStr>(
    name: &str,
    legacy: &str,
) -> Result<Option<T>, WitnessProgramError> {
    setting(name, legacy)?
        .map(|v| {
            v.parse()
                .map_err(|_| invalid(format!("{name}: invalid number {v:?}")))
        })
        .transpose()
}
impl WitnessConfig {
    /// Resolves documented `PLONKY2_EXECUTOR_*` settings. Historical `P2_AOT_*`
    /// controls are accepted as migration aliases; the new names take precedence.
    pub fn from_env() -> Result<Self, WitnessProgramError> {
        let mut c = Self::default();
        if let Some(v) = setting("PLONKY2_EXECUTOR_LAYOUT", "P2_AOT_LAYOUT")? {
            c.layout = match v.as_str() {
                "canonical" => WitnessLayout::Canonical,
                "outputs" | "aos" => WitnessLayout::Outputs,
                "representative" => WitnessLayout::Representatives,
                "block4" => WitnessLayout::Block4,
                "block8" => WitnessLayout::Block8,
                _ => return Err(invalid(format!("unknown witness layout {v:?}"))),
            };
        }
        if let Some(v) = number("PLONKY2_EXECUTOR_CHAIN_LENGTH", "P2_AOT_CHAIN_LEN")? {
            c.chain_length = v;
        }
        if setting("P2_AOT_FUSE", "P2_AOT_FUSE")?.as_deref() == Some("0")
            && std::env::var_os("PLONKY2_EXECUTOR_CHAIN_LENGTH").is_none()
        {
            c.chain_length = 0;
        }
        if let Some(v) = setting("PLONKY2_EXECUTOR_MODE", "P2_AOT_MODE")? {
            c.execution = match v.as_str() {
                "auto" | "adaptive" | "fast" | "fast-batch" | "bulk" => WitnessExecution::Auto,
                "checked" => WitnessExecution::Checked,
                _ => return Err(invalid(format!("unknown witness execution mode {v:?}"))),
            };
        }
        if let Some(v) = setting("PLONKY2_EXECUTOR_POSEIDON", "P2_POSEIDON_MODE")? {
            c.poseidon = match v.as_str() {
                "auto" | "batch8-fft" => WitnessPoseidon::Auto,
                "scalar" => WitnessPoseidon::Scalar,
                _ => return Err(invalid(format!("unknown Poseidon mode {v:?}"))),
            };
        }
        if let Some(v) = setting("PLONKY2_EXECUTOR_FALLBACK", "PLONKY2_EXECUTOR_FALLBACK")? {
            c.fallback = match v.as_str() {
                "worklist" => WitnessFallback::Worklist,
                "error" => WitnessFallback::Error,
                _ => return Err(invalid(format!("unknown witness fallback {v:?}"))),
            };
        }
        if let Some(v) = number("PLONKY2_EXECUTOR_THREADS", "P2_AOT_WITNESS_THREADS")? {
            c.threads = v;
        }
        if let Some(v) = number(
            "PLONKY2_EXECUTOR_BUFFER_BYTES",
            "PLONKY2_EXECUTOR_BUFFER_BYTES",
        )? {
            c.buffers = if v == 0 {
                WitnessBuffers::Fresh
            } else {
                WitnessBuffers::Reuse { max_bytes: v }
            };
        } else if setting("P2_WITNESS_BUFFERS", "P2_WITNESS_BUFFERS")?.as_deref() == Some("pool") {
            c.buffers = WitnessBuffers::Reuse {
                max_bytes: 64 * 1024 * 1024,
            };
        }
        if let Some(v) = number("PLONKY2_EXECUTOR_CHUNK_SCALE", "P2_AOT_CHUNK_SCALE")? {
            c.chunk_scale = v;
        }
        if let Some(v) = number("PLONKY2_EXECUTOR_GATHER_COLUMNS", "P2_AOT_GATHER_COLUMNS")? {
            c.gather_columns = v;
        }
        if let Some(v) = number("PLONKY2_EXECUTOR_GATHER_ROWS", "P2_AOT_GATHER_ROWS")? {
            c.gather_rows = v;
        }
        if let Some(v) = setting("PLONKY2_EXECUTOR_GATHER", "PLONKY2_EXECUTOR_GATHER")? {
            c.gather = match v.as_str() {
                "auto" => WitnessGather::Auto,
                "cpu" => WitnessGather::Cpu,
                "gpu" => WitnessGather::Gpu,
                _ => return Err(invalid(format!("unknown witness gather {v:?}"))),
            };
        } else if setting("P2_AOT_GPU_GATHER", "P2_AOT_GPU_GATHER")?.as_deref() == Some("1") {
            c.gather = WitnessGather::Gpu;
        }
        if let Some(v) = setting(
            "PLONKY2_EXECUTOR_SIMPLIFY_ARITHMETIC",
            "P2_AOT_SIMPLIFY_ARITH",
        )? {
            c.simplify_arithmetic = match v.as_str() {
                "1" | "true" => true,
                "0" | "false" => false,
                _ => return Err(invalid("arithmetic simplification must be true or false")),
            };
        }
        c.cache = std::env::var_os("PLONKY2_EXECUTOR_CACHE_DIR")
            .filter(|dir| !dir.is_empty())
            .map_or(WitnessCache::Disabled, |dir| {
                WitnessCache::Directory(dir.into())
            });
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
        if self.chain_length == 1 || self.chain_length > 8 {
            return Err(invalid("chain_length must be zero or 2..=8"));
        }
        if self.chunk_scale == 0 || self.gather_columns == 0 || self.gather_rows == 0 {
            return Err(invalid(
                "chunk_scale and gather dimensions must be positive",
            ));
        }
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
        self.compilation_key() == other.compilation_key()
            && self.execution == other.execution
            && self.poseidon == other.poseidon
            && self.fallback == other.fallback
            && self.buffers == other.buffers
            && self.gather == other.gather
            && self.threads == other.threads
            && pool
            && self.cache == other.cache
            && self.chunk_scale == other.chunk_scale
            && self.simplify_arithmetic == other.simplify_arithmetic
            && self.gather_columns == other.gather_columns
            && self.gather_rows == other.gather_rows
    }

    pub(crate) fn compilation_key(&self) -> u32 {
        self.layout.code() | (u32::from(self.chain_length) << 3)
    }
    pub(crate) fn with_compilation_key(mut self, key: u32) -> Result<Self, WitnessProgramError> {
        self.layout = WitnessLayout::from_code(key & 7)?;
        self.chain_length =
            u8::try_from(key >> 3).map_err(|_| invalid("invalid stored chain length"))?;
        self.validate()?;
        Ok(self)
    }
}

#[derive(Debug)]
pub(crate) struct Runtime {
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
    #[test]
    fn validates_application_options_before_resource_creation() {
        let options = WitnessConfig::default();
        options.validate().unwrap();
        let invalid = WitnessConfig {
            cache: WitnessCache::Directory(PathBuf::new()),
            ..options.clone()
        };
        assert!(invalid.validate().is_err());
        let invalid = WitnessConfig {
            chain_length: 1,
            ..options.clone()
        };
        assert!(invalid.validate().is_err());
        let invalid = WitnessConfig {
            gather_columns: 0,
            ..options.clone()
        };
        assert!(invalid.validate().is_err());
        let invalid = WitnessConfig {
            buffers: WitnessBuffers::Reuse { max_bytes: 0 },
            ..options
        };
        assert!(invalid.validate().is_err());
    }
    #[test]
    #[cfg(feature = "parallel")]
    fn environment_precedence_is_resolved_without_mutating_the_test_process() {
        const CHILD: &str = "PLONKY2_EXECUTOR_CONFIG_TEST_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let config = WitnessConfig::from_env().unwrap();
            assert_eq!(config.threads, 2);
            assert_eq!(config.chain_length, 0);
            assert_eq!(config.cache, WitnessCache::Disabled);
            return;
        }
        let status=std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact","plonk::witness_program::config::tests::environment_precedence_is_resolved_without_mutating_the_test_process"])
            .env(CHILD,"1").env("P2_AOT_WITNESS_THREADS","7").env("PLONKY2_EXECUTOR_THREADS","2")
            .env("P2_AOT_CHAIN_LEN","8").env("PLONKY2_EXECUTOR_CHAIN_LENGTH","0")
            .env_remove("PLONKY2_EXECUTOR_CACHE_DIR").status().unwrap();
        assert!(status.success());
    }
}
