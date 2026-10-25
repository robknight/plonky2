//! The executor owns its persistent representation; callers only request preparation.
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::region::plan_bytes::PlanKey;
use super::{WitnessCache, WitnessConfig, WitnessProgram, WitnessProgramError};

/// Where this preparation obtained the immutable compiled program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum WitnessProgramSource {
    /// A matching program already belongs to this circuit (possibly reconfigured).
    Installed,
    /// A compatible program was read and validated from disk.
    DiskCache,
    /// A new program was compiled.
    Compiled,
}

/// Preparation completes host certification and lookup tables before returning.
/// Durations include all work, including storing a new cache entry. GPU context
/// creation and prover-wide shader warm-up remain the GPU prover's lifecycle.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct WitnessPreparation {
    pub source: WitnessProgramSource,
    pub elapsed: Duration,
    pub cache_read: Duration,
    pub compile: Duration,
    pub cache_write: Duration,
    pub host_prepare: Duration,
    /// Nonfatal cache I/O or rejection information. Execution remains available.
    pub cache_warning: Option<String>,
    pub(super) program: WitnessProgram,
}
impl WitnessPreparation {
    /// The installed program. Cloning this handle shares topology and tables.
    pub fn program(&self) -> &WitnessProgram {
        &self.program
    }
}

pub(super) struct Outcome {
    pub program: WitnessProgram,
    pub source: WitnessProgramSource,
    pub read: Duration,
    pub compile: Duration,
    pub cache_write: Duration,
    pub warning: Option<String>,
}

pub(super) fn directory(config: &mut WitnessConfig) -> Option<PathBuf> {
    if config.cache == WitnessCache::Environment {
        config.cache = std::env::var_os("PLONKY2_EXECUTOR_CACHE_DIR")
            .filter(|v| !v.is_empty())
            .map_or(WitnessCache::Disabled, |v| {
                WitnessCache::Directory(v.into())
            });
    }
    match &config.cache {
        WitnessCache::Directory(p) => Some(p.clone()),
        _ => None,
    }
}
fn filename(key: &PlanKey) -> String {
    let mut mask = 0xcbf29ce484222325u64;
    for word in &key.mask {
        for byte in (*word as u64).to_le_bytes() {
            mask = (mask ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
    let digest = key
        .digest
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    format!(
        "{digest}-{:016x}-{:032x}-{mask:016x}-{:08x}.p2w",
        key.format, key.operations, key.compilation
    )
}
fn read(
    path: &Path,
    key: &PlanKey,
    decode: &impl Fn(&[u8]) -> Result<WitnessProgram, WitnessProgramError>,
) -> Result<Option<WitnessProgram>, String> {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("reading {}: {e}", path.display())),
    };
    let program = decode(&bytes).map_err(|e| format!("rejected {}: {e}", path.display()))?;
    if program.key.as_ref() != key {
        return Err(format!(
            "rejected {}: executor identity differs",
            path.display()
        ));
    }
    Ok(Some(program))
}
fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Temporary(PathBuf);
    impl Drop for Temporary {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }
    let name = path
        .file_name()
        .expect("generated executor cache filename")
        .to_string_lossy();
    for _ in 0..32 {
        let temp = Temporary(path.with_file_name(format!(
            ".{name}.{}.{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )));
        let mut file = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp.0)
        {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        };
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temp.0, path)?;
        return Ok(());
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "executor cache temporary files exhausted",
    ))
}

pub(super) fn obtain(
    directory: Option<&Path>,
    key: &PlanKey,
    decode: impl Fn(&[u8]) -> Result<WitnessProgram, WitnessProgramError>,
    compile: impl FnOnce() -> Result<WitnessProgram, WitnessProgramError>,
) -> Result<Outcome, WitnessProgramError> {
    let mut read_time = Duration::ZERO;
    let mut warning = None;
    let mut lock: Option<File> = None;
    let path = directory.map(|dir| dir.join("witness-executors-v2").join(filename(key)));
    if let Some(path) = &path {
        let start = Instant::now();
        match read(path, key, &decode) {
            Ok(Some(program)) => {
                return Ok(Outcome {
                    program,
                    source: WitnessProgramSource::DiskCache,
                    read: start.elapsed(),
                    compile: Duration::ZERO,
                    cache_write: Duration::ZERO,
                    warning: None,
                })
            }
            Err(e) => warning = Some(e),
            Ok(None) => {}
        }
        // Reads need no lock: writers replace complete entries atomically. On a
        // miss, serialize compilation and recheck after acquiring the file lock.
        let acquire = || -> std::io::Result<File> {
            fs::create_dir_all(path.parent().expect("cache namespace"))?;
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(path.with_extension("lock"))?;
            file.lock()?;
            Ok(file)
        };
        match acquire() {
            Ok(file) => {
                lock = Some(file);
                match read(path, key, &decode) {
                    Ok(Some(program)) => {
                        return Ok(Outcome {
                            program,
                            source: WitnessProgramSource::DiskCache,
                            read: start.elapsed(),
                            compile: Duration::ZERO,
                            cache_write: Duration::ZERO,
                            warning,
                        })
                    }
                    Err(e) => warning = Some(e),
                    Ok(None) => {}
                }
            }
            Err(e) => {
                warning = Some(format!(
                    "executor cache {} unavailable: {e}",
                    path.display()
                ))
            }
        }
        read_time = start.elapsed();
    }
    let start = Instant::now();
    let program = compile()?;
    let compile_time = start.elapsed();
    let write_start = Instant::now();
    if lock.is_some() {
        if let Some(path) = &path {
            if let Err(e) = write_atomic(path, &program.to_bytes()) {
                warning = Some(format!("writing executor cache {}: {e}", path.display()));
            }
        }
    }
    Ok(Outcome {
        program,
        source: WitnessProgramSource::Compiled,
        read: read_time,
        compile: compile_time,
        cache_write: if lock.is_some() {
            write_start.elapsed()
        } else {
            Duration::ZERO
        },
        warning,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField as F;
    use crate::field::types::Field;
    use crate::hash::poseidon::PoseidonHash;
    use crate::iop::target::Target;
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::{CircuitConfig, CircuitData};
    use crate::plonk::config::PoseidonGoldilocksConfig as C;
    use crate::plonk::witness_program::WitnessProgramStatus;

    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "plonky2-executor-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn config(&self) -> WitnessConfig {
            WitnessConfig {
                cache: WitnessCache::Directory(self.0.clone()),
                ..WitnessConfig::default()
            }
        }
        fn entries(&self) -> Vec<PathBuf> {
            fs::read_dir(self.0.join("witness-executors-v2"))
                .unwrap()
                .map(|p| p.unwrap().path())
                .filter(|p| p.extension().is_some_and(|e| e == "p2w"))
                .collect()
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn circuit() -> (CircuitData<F, C, 2>, [Target; 3], Target) {
        let mut builder = CircuitBuilder::<F, 2>::new(CircuitConfig::standard_recursion_config());
        let a = builder.add_virtual_target();
        let alias = builder.add_virtual_target();
        builder.connect(a, alias);
        let b = builder.add_virtual_target();
        let extra = builder.add_virtual_target();
        let product = builder.mul(a, b);
        let hash = builder.hash_n_to_hash_no_pad::<PoseidonHash>(vec![a, b, product]);
        builder.register_public_inputs(&hash.elements);
        (builder.build::<C>(), [a, b, extra], alias)
    }
    fn prove(data: &CircuitData<F, C, 2>, targets: [Target; 3], extra: bool) {
        let mut pw = PartialWitness::new();
        pw.set_target(targets[0], F::from_canonical_u64(7)).unwrap();
        pw.set_target(targets[1], F::from_canonical_u64(11))
            .unwrap();
        if extra {
            pw.set_target(targets[2], F::from_canonical_u64(19))
                .unwrap();
        }
        data.verify(data.prove(pw).unwrap()).unwrap();
    }
    #[test]
    fn cache_reloads_repairs_damage_and_keys_canonical_input_shapes() {
        let dir = Directory::new();
        let (first, targets, alias) = circuit();
        let built = first
            .prepare_witness_program_with_config(&targets[..2], dir.config())
            .unwrap();
        assert_eq!(built.source, WitnessProgramSource::Compiled);
        assert!(built.program().is_prepared());
        prove(&first, targets, false);
        let entries = dir.entries();
        assert_eq!(entries.len(), 1);
        let (second, other_targets, _) = circuit();
        let loaded = second
            .prepare_witness_program_with_config(&[alias, other_targets[1]], dir.config())
            .unwrap();
        assert_eq!(loaded.source, WitnessProgramSource::DiskCache);
        assert_eq!(built.program().to_bytes(), loaded.program().to_bytes());
        prove(&second, other_targets, false);
        // Additional unconstrained input classes change the executor's boundary contract.
        let different = second
            .prepare_witness_program_with_config(&other_targets, dir.config())
            .unwrap();
        assert_eq!(different.source, WitnessProgramSource::Compiled);
        assert_eq!(dir.entries().len(), 2);
        prove(&second, other_targets, true);
        fs::write(&entries[0], b"interrupted or damaged cached entry").unwrap();
        let (third, third_targets, _) = circuit();
        let repaired = third
            .prepare_witness_program_with_config(&third_targets[..2], dir.config())
            .unwrap();
        assert_eq!(repaired.source, WitnessProgramSource::Compiled);
        assert!(repaired
            .cache_warning
            .as_deref()
            .unwrap()
            .contains("rejected"));
        prove(&third, third_targets, false);
        let (fourth, fourth_targets, _) = circuit();
        let loaded = fourth
            .prepare_witness_program_with_config(&fourth_targets[..2], dir.config())
            .unwrap();
        assert_eq!(loaded.source, WitnessProgramSource::DiskCache);
        assert!(loaded.cache_warning.is_none());
    }
    #[test]
    fn concurrent_preparation_shares_an_installed_program_and_serializes_disk_misses() {
        let dir = Directory::new();
        let (data, targets, _) = circuit();
        let data = Arc::new(data);
        let barrier = Arc::new(Barrier::new(4));
        let results = std::thread::scope(|scope| {
            let tasks = (0..4)
                .map(|_| {
                    let data = data.clone();
                    let barrier = barrier.clone();
                    let config = dir.config();
                    scope.spawn(move || {
                        barrier.wait();
                        let prepared = data
                            .prepare_witness_program_with_config(&targets[..2], config)
                            .unwrap();
                        prove(&data, targets, false);
                        prepared
                    })
                })
                .collect::<Vec<_>>();
            tasks
                .into_iter()
                .map(|t| t.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(
            results
                .iter()
                .filter(|r| r.source == WitnessProgramSource::Compiled)
                .count(),
            1
        );
        assert_eq!(
            results
                .iter()
                .filter(|r| r.source == WitnessProgramSource::Installed)
                .count(),
            3
        );
        assert_eq!(data.witness_program_status(), WitnessProgramStatus::Ready);
        // Independent circuit objects share only the disk entry/lock, not a process registry.
        let second_dir = Directory::new();
        let barrier = Arc::new(Barrier::new(4));
        let sources = std::thread::scope(|scope| {
            let tasks = (0..4)
                .map(|_| {
                    let config = second_dir.config();
                    let barrier = barrier.clone();
                    scope.spawn(move || {
                        let (data, targets, _) = circuit();
                        barrier.wait();
                        data.prepare_witness_program_with_config(&targets[..2], config)
                            .unwrap()
                            .source
                    })
                })
                .collect::<Vec<_>>();
            tasks
                .into_iter()
                .map(|t| t.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(
            sources
                .iter()
                .filter(|s| **s == WitnessProgramSource::Compiled)
                .count(),
            1
        );
        assert_eq!(
            sources
                .iter()
                .filter(|s| **s == WitnessProgramSource::DiskCache)
                .count(),
            3
        );
    }
    #[test]
    fn unavailable_cache_does_not_prevent_memory_preparation() {
        let dir = Directory::new();
        let file = dir.0.join("not-a-directory");
        fs::write(&file, b"owned test file").unwrap();
        let (data, targets, _) = circuit();
        let config = WitnessConfig {
            cache: WitnessCache::Directory(file),
            ..WitnessConfig::default()
        };
        let prepared = data
            .prepare_witness_program_with_config(&targets[..2], config)
            .unwrap();
        assert_eq!(prepared.source, WitnessProgramSource::Compiled);
        assert!(prepared.cache_warning.is_some());
        assert!(prepared.program().is_prepared());
        prove(&data, targets, false);
    }
}
