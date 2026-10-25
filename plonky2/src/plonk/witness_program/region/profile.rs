//! Opt-in attribution that preserves the ordinary parallel schedule.
//!
//! CPU totals measure work, not critical-path wall time. Nested Poseidon
//! attribution is a subset of kind totals and must not be added to them.

#[derive(Clone, Copy)]
pub(super) enum Stage {
    Allocate,
    Seed,
    Layers,
    Aliases,
    Ready,
    Wipe,
}
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[derive(Clone, Copy)]
pub(super) enum Poseidon {
    Input,
    Arithmetic,
    Output,
}

#[cfg(feature = "witness-profiling")]
mod enabled {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    use super::{Poseidon, Stage};

    #[cfg(feature = "witness-benchmarks")]
    static OVERRIDE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);
    #[cfg(feature = "witness-benchmarks")]
    pub(crate) fn benchmark_override(active: bool) {
        OVERRIDE.store(i8::from(active), Ordering::Relaxed);
    }

    pub(super) fn enabled() -> bool {
        #[cfg(feature = "witness-benchmarks")]
        {
            let mode = OVERRIDE.load(Ordering::Relaxed);
            if mode >= 0 {
                return mode != 0;
            }
        }
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var("P2_WITNESS_PROFILE").as_deref() == Ok("1"))
    }
    fn cpu_ns() -> u64 {
        #[cfg(unix)]
        unsafe {
            let mut t: libc::timespec = core::mem::zeroed();
            if libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut t) == 0 {
                return t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64;
            }
        }
        0
    }
    pub(crate) struct Execution {
        active: bool,
        slots: u32,
        stage: [AtomicU64; 6],
        cpu: [AtomicU64; 27],
        tasks: [AtomicU64; 27],
        lanes: [AtomicU64; 27],
        poseidon: [AtomicU64; 3],
        layers: Mutex<Vec<(usize, u64, u64)>>,
    }
    impl Execution {
        pub(crate) fn new(slots: u32) -> Self {
            Self {
                active: enabled(),
                slots,
                stage: std::array::from_fn(|_| AtomicU64::new(0)),
                cpu: std::array::from_fn(|_| AtomicU64::new(0)),
                tasks: std::array::from_fn(|_| AtomicU64::new(0)),
                lanes: std::array::from_fn(|_| AtomicU64::new(0)),
                poseidon: std::array::from_fn(|_| AtomicU64::new(0)),
                layers: Mutex::new(Vec::new()),
            }
        }
        pub(crate) fn stage(&self, stage: Stage) -> Span<'_> {
            self.span(Target::Stage(stage))
        }
        pub(crate) fn kind(&self, kind: u8, lanes: u32) -> Span<'_> {
            self.span(Target::Kind(kind as usize, lanes))
        }
        pub(crate) fn poseidon(&self, part: Poseidon) -> Span<'_> {
            self.span(Target::Poseidon(part))
        }
        pub(crate) fn layer(&self, index: usize, lanes: u64) -> Span<'_> {
            self.span(Target::Layer(index, lanes))
        }
        fn span(&self, target: Target) -> Span<'_> {
            Span {
                profile: self,
                target,
                started: self.active.then(Instant::now),
                cpu: if self.active { cpu_ns() } else { 0 },
            }
        }
        pub(crate) fn report(&self) {
            if !self.active {
                return;
            }
            let load = |x: &AtomicU64| x.load(Ordering::Relaxed);
            eprintln!("WITNESS_PROFILE slots={} stage_ns={:?} kind_cpu_ns={:?} kind_tasks={:?} kind_lanes={:?} poseidon_cpu_ns={:?}",
                self.slots, self.stage.each_ref().map(load), self.cpu.each_ref().map(load),
                self.tasks.each_ref().map(load), self.lanes.each_ref().map(load), self.poseidon.each_ref().map(load));
            let mut layers = self.layers.lock().unwrap();
            layers.sort_by_key(|x| core::cmp::Reverse(x.1));
            eprintln!(
                "WITNESS_LAYERS slots={} count={} top_wall_ns={:?}",
                self.slots,
                layers.len(),
                &layers[..layers.len().min(12)]
            );
        }
    }
    enum Target {
        Stage(Stage),
        Kind(usize, u32),
        Poseidon(Poseidon),
        Layer(usize, u64),
    }
    pub(crate) struct Span<'a> {
        profile: &'a Execution,
        target: Target,
        started: Option<Instant>,
        cpu: u64,
    }
    impl Drop for Span<'_> {
        fn drop(&mut self) {
            if let Some(started) = self.started {
                let wall = started.elapsed().as_nanos() as u64;
                let cpu = cpu_ns().saturating_sub(self.cpu);
                match self.target {
                    Target::Stage(s) => {
                        self.profile.stage[s as usize].fetch_add(wall, Ordering::Relaxed);
                    }
                    Target::Kind(k, n) => {
                        self.profile.cpu[k].fetch_add(cpu, Ordering::Relaxed);
                        self.profile.tasks[k].fetch_add(1, Ordering::Relaxed);
                        self.profile.lanes[k].fetch_add(n as u64, Ordering::Relaxed);
                    }
                    Target::Poseidon(p) => {
                        self.profile.poseidon[p as usize].fetch_add(cpu, Ordering::Relaxed);
                    }
                    Target::Layer(i, n) => {
                        self.profile.layers.lock().unwrap().push((i, wall, n));
                    }
                }
            }
        }
    }
    pub(crate) struct Pipeline {
        label: &'static str,
        generators: usize,
        started: Option<Instant>,
    }
    impl Pipeline {
        pub(crate) fn new(label: &'static str, generators: usize) -> Self {
            Self {
                label,
                generators,
                started: enabled().then(Instant::now),
            }
        }
    }
    impl Drop for Pipeline {
        fn drop(&mut self) {
            if let Some(started) = self.started {
                eprintln!(
                    "WITNESS_PIPELINE generators={} stage={} ns={}",
                    self.generators,
                    self.label,
                    started.elapsed().as_nanos()
                );
            }
        }
    }
}

#[cfg(all(feature = "witness-profiling", feature = "witness-benchmarks"))]
pub(super) use enabled::benchmark_override;
#[cfg(feature = "witness-profiling")]
pub(super) use enabled::{Execution, Pipeline};

#[cfg(not(feature = "witness-profiling"))]
pub(super) struct Execution;
#[cfg(not(feature = "witness-profiling"))]
pub(super) struct Span;
#[cfg(not(feature = "witness-profiling"))]
impl Execution {
    #[inline(always)]
    pub fn new(_: u32) -> Self {
        Self
    }
    #[inline(always)]
    pub fn stage(&self, _: Stage) -> Span {
        Span
    }
    #[inline(always)]
    pub fn kind(&self, _: u8, _: u32) -> Span {
        Span
    }
    #[inline(always)]
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub fn poseidon(&self, _: Poseidon) -> Span {
        Span
    }
    #[inline(always)]
    pub fn layer(&self, _: usize, _: u64) -> Span {
        Span
    }
    #[inline(always)]
    pub fn report(&self) {}
}
#[cfg(not(feature = "witness-profiling"))]
pub(super) struct Pipeline;
#[cfg(not(feature = "witness-profiling"))]
impl Pipeline {
    #[inline(always)]
    pub fn new(_: &'static str, _: usize) -> Self {
        Self
    }
}

#[cfg(all(not(feature = "witness-profiling"), feature = "witness-benchmarks"))]
pub(super) fn benchmark_override(_: bool) {}

// Match the enabled guards' explicit-drop API while compiling to no work.
#[cfg(not(feature = "witness-profiling"))]
impl Drop for Span {
    fn drop(&mut self) {}
}
#[cfg(not(feature = "witness-profiling"))]
impl Drop for Pipeline {
    fn drop(&mut self) {}
}
