//! Chunk dependencies let independent branches advance without layer barriers.
//! This is prepared metadata; executions own their counters and error state.
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use anyhow::{ensure, Result};

use super::cpu::Executor;
use super::{Group, Kind, RegionProgram};
use crate::plonk::witness_program::policy::WitnessSchedule;

#[derive(Debug)]
struct Task {
    group: Group,
    start: u32,
    end: u32,
}
#[derive(Debug)]
pub(crate) struct Schedule {
    tasks: Vec<Task>,
    offsets: Vec<usize>,
    successors: Vec<u32>,
    indegrees: Vec<u32>,
    entries: Vec<u32>,
}

pub(crate) type PreparedSchedule = OnceLock<Result<Schedule, String>>;

pub(super) fn prepare(p: &RegionProgram) -> Result<&Schedule> {
    p.runtime
        .schedule
        .get_or_init(|| Schedule::compile(p).map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|error| anyhow::anyhow!("witness scheduling: {error}"))
}

pub(super) fn run(e: &Executor<'_>) -> Result<bool> {
    #[cfg(feature = "witness-benchmarks")]
    if !BENCH_ENABLED.load(Ordering::Relaxed) {
        return Ok(false);
    }
    let requested = e.program.runtime.config.policy.schedule;
    if requested == WitnessSchedule::Layers
        || (requested == WitnessSchedule::Auto && rayon::current_num_threads() == 1)
    {
        return Ok(false);
    }
    prepare(e.program)?.execute(e)?;
    Ok(true)
}

#[cfg(feature = "witness-benchmarks")]
static BENCH_ENABLED: AtomicBool = AtomicBool::new(true);
#[cfg(feature = "witness-benchmarks")]
pub(crate) fn benchmark_enabled(enabled: bool) {
    BENCH_ENABLED.store(enabled, Ordering::Relaxed);
}

impl Schedule {
    fn compile(p: &RegionProgram) -> Result<Self> {
        // Certification has already established a single writer lane per
        // non-root slot. Repeated outputs within that lane stay in one task.
        ensure!(p.fast.get().is_some(), "uncertified scheduling input");
        let started = std::time::Instant::now();
        let mut tasks = Vec::new();
        let mut layers = Vec::new();
        for (layer, groups) in p.groups.iter().enumerate() {
            for group in groups {
                let base = match group.kind {
                    Kind::Poseidon if group.count >= 64 => 8,
                    Kind::RandomValue => 4096,
                    _ => super::cpu::lane_chunk(group.kind),
                };
                let chunk = base;
                for c in 0..group.count.div_ceil(chunk) {
                    let start = c * chunk;
                    tasks.push(Task {
                        group: group.clone(),
                        start,
                        end: super::cpu::chunk_end(start, chunk, group.count),
                    });
                    layers.push(layer);
                }
            }
        }
        ensure!(tasks.len() < u32::MAX as usize, "too many witness tasks");
        let mut owner = vec![u32::MAX; p.state_slots as usize];
        let target = |g: &Group, pos: u32, lane: u32| {
            p.targets[g.targets_offset as usize + pos as usize * g.count as usize + lane as usize]
        };
        for (id, task) in tasks.iter().enumerate() {
            let g = &task.group;
            for pos in g.inputs..g.inputs + g.outputs {
                for lane in task.start..task.end {
                    owner[target(g, pos, lane) as usize] = id as u32;
                }
            }
        }
        let mut successors = vec![Vec::<u32>::new(); tasks.len()];
        let mut indegrees = Vec::with_capacity(tasks.len());
        let mut entries = Vec::new();
        let mut parents = Vec::new();
        for (id, task) in tasks.iter().enumerate() {
            parents.clear();
            let g = &task.group;
            for pos in 0..g.inputs {
                for lane in task.start..task.end {
                    let slot = target(g, pos, lane) as usize;
                    if slot < p.roots.len() {
                        continue;
                    }
                    let from = owner[slot];
                    ensure!(from != u32::MAX, "unowned scheduling input");
                    ensure!(
                        layers[from as usize] < layers[id],
                        "non-forward scheduling edge"
                    );
                    parents.push(from);
                }
            }
            parents.sort_unstable();
            parents.dedup();
            indegrees.push(u32::try_from(parents.len())?);
            if parents.is_empty() {
                entries.push(id as u32);
            }
            for &from in &parents {
                successors[from as usize].push(id as u32);
            }
        }
        let mut offsets = Vec::with_capacity(tasks.len() + 1);
        let mut edges = Vec::new();
        for next in successors {
            offsets.push(edges.len());
            edges.extend(next);
        }
        offsets.push(edges.len());
        log::debug!(target: super::COMPILE_LOG,"dependency schedule: {} tasks, {} edges in {:?}",tasks.len(),edges.len(),started.elapsed());
        Ok(Self {
            tasks,
            offsets,
            successors: edges,
            indegrees,
            entries,
        })
    }

    fn execute(&self, executor: &Executor<'_>) -> Result<()> {
        let run = Run {
            executor,
            graph: self,
            remaining: self.indegrees.iter().map(|&n| AtomicU32::new(n)).collect(),
            failed: AtomicBool::new(false),
            error: Mutex::new(None),
            completed: AtomicUsize::new(0),
        };
        rayon::scope(|scope| {
            for &id in &self.entries {
                run.spawn(scope, id as usize);
            }
        });
        if let Some(error) = run.error.into_inner().unwrap() {
            return Err(error);
        }
        ensure!(
            run.completed.load(Ordering::Relaxed) == self.tasks.len(),
            "unfinished witness dependency graph"
        );
        Ok(())
    }
}

struct Run<'a, 'state> {
    executor: &'a Executor<'state>,
    graph: &'a Schedule,
    remaining: Vec<AtomicU32>,
    failed: AtomicBool,
    error: Mutex<Option<anyhow::Error>>,
    completed: AtomicUsize,
}
impl<'a, 'state> Run<'a, 'state> {
    fn spawn<'scope>(&'scope self, scope: &rayon::Scope<'scope>, mut id: usize)
    where
        'a: 'scope,
        'state: 'scope,
    {
        scope.spawn(move |scope| loop {
            if self.failed.load(Ordering::Acquire) {
                return;
            }
            let task = &self.graph.tasks[id];
            let result = {
                let _profile = self
                    .executor
                    .profile
                    .kind(task.group.kind.code() as u8, task.end - task.start);
                super::aot::run_range(self.executor, &task.group, task.start, task.end)
            };
            if let Err(error) = result {
                self.failed.store(true, Ordering::Release);
                let mut first = self.error.lock().unwrap();
                if first.is_none() {
                    *first = Some(error);
                }
                return;
            }
            self.completed.fetch_add(1, Ordering::Relaxed);
            let mut continuation = None;
            for &next in &self.graph.successors[self.graph.offsets[id]..self.graph.offsets[id + 1]]
            {
                // AcqRel transitively publishes every parent's relaxed state
                // stores before the last parent starts this child.
                if self.remaining[next as usize].fetch_sub(1, Ordering::AcqRel) == 1 {
                    if continuation.is_none() {
                        continuation = Some(next as usize);
                    } else {
                        self.spawn(scope, next as usize);
                    }
                }
            }
            match continuation {
                Some(next) => id = next,
                None => return,
            }
        });
    }
}
