//! Bounded reuse of already-wiped witness buffers. No values or readiness
//! flags survive release, including executions that return an error.
#[cfg(any(test, feature = "witness-benchmarks"))]
use std::sync::atomic::Ordering;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::Mutex;

use plonky2_maybe_rayon::*;
use zeroize::Zeroize;

use super::super::WitnessBuffers;

const LIMIT: usize = 64 * 1024 * 1024;
#[cfg(feature = "witness-benchmarks")]
static OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(255);
#[cfg(feature = "witness-benchmarks")]
pub(crate) fn benchmark_override(enabled: bool) {
    OVERRIDE.store(u8::from(enabled), Ordering::Relaxed);
}
#[derive(Debug)]
struct Storage {
    state: Vec<AtomicU64>,
    ready: Vec<AtomicBool>,
}
impl Storage {
    fn fresh(len: usize) -> Self {
        Self {
            state: (0..len).map(|_| AtomicU64::new(0)).collect(),
            ready: (0..len).map(|_| AtomicBool::new(false)).collect(),
        }
    }
    fn bytes(&self) -> usize {
        self.state.capacity() * core::mem::size_of::<AtomicU64>()
            + self.ready.capacity() * core::mem::size_of::<AtomicBool>()
    }
    fn scrub(mut self, reset_ready: bool) -> Self {
        self.state
            .par_iter_mut()
            .for_each(|v| v.get_mut().zeroize());
        if reset_ready {
            self.ready.par_iter_mut().for_each(|r| *r.get_mut() = false);
        }
        self
    }
}
#[derive(Debug)]
struct Pool {
    entries: Vec<Storage>,
    bytes: usize,
    limit: usize,
}
impl Pool {
    fn new(limit: usize) -> Self {
        Self {
            entries: Vec::new(),
            bytes: 0,
            limit,
        }
    }
    fn take(&mut self, len: usize) -> Option<Storage> {
        let index = self
            .entries
            .iter()
            .position(|x| x.state.len() == len && x.ready.len() == len)?;
        let entry = self.entries.remove(index);
        self.bytes -= entry.bytes();
        Some(entry)
    }
    /// Callers supply scrubbed buffers. Prefer recent sizes when the budget
    /// fills, so an earlier circuit cannot permanently displace the hot one.
    fn put(&mut self, entry: Storage) -> bool {
        let bytes = entry.bytes();
        if bytes > self.limit {
            return false;
        }
        while self.bytes + bytes > self.limit {
            let old = self.entries.remove(0);
            self.bytes -= old.bytes();
        }
        self.bytes += bytes;
        self.entries.push(entry);
        true
    }
}
/// Shared only by clones of this program; memory is released with its last owner.
#[derive(Debug)]
pub(crate) struct Buffers {
    reuse: bool,
    pool: Mutex<Pool>,
}
impl Buffers {
    pub fn new(policy: WitnessBuffers) -> Self {
        let (reuse, limit) = match policy {
            WitnessBuffers::Fresh => (false, LIMIT),
            WitnessBuffers::Reuse { max_bytes } => (true, max_bytes),
        };
        Self {
            reuse,
            pool: Mutex::new(Pool::new(limit)),
        }
    }
    fn enabled(&self) -> bool {
        #[cfg(feature = "witness-benchmarks")]
        match OVERRIDE.load(Ordering::Relaxed) {
            0 => return false,
            255 => {}
            _ => return true,
        }
        self.reuse
    }

    pub(super) fn acquire(&self, len: usize) -> (Vec<AtomicU64>, Vec<AtomicBool>) {
        if self.enabled() {
            if let Some(entry) = self
                .pool
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take(len)
            {
                return (entry.state, entry.ready);
            }
        }
        let entry = Storage::fresh(len);
        (entry.state, entry.ready)
    }
    pub(super) fn release(&self, state: Vec<AtomicU64>, ready: Vec<AtomicBool>) {
        let entry = Storage { state, ready };
        let retain = self.enabled()
            && entry.bytes() <= self.pool.lock().unwrap_or_else(|e| e.into_inner()).limit;
        let entry = entry.scrub(retain);
        if retain {
            self.pool
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .put(entry);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reused_buffers_have_no_values_or_readiness_and_have_one_owner() {
        let entry = Storage::fresh(8);
        let pointer = entry.state.as_ptr();
        for v in &entry.state {
            v.store(0xdeadbeef, Ordering::Relaxed);
        }
        for r in &entry.ready {
            r.store(true, Ordering::Relaxed);
        }
        let mut pool = Pool::new(1024);
        assert!(pool.put(entry.scrub(true)));
        assert!(pool.take(7).is_none());
        let entry = pool.take(8).unwrap();
        assert_eq!(entry.state.as_ptr(), pointer);
        assert!(pool.take(8).is_none());
        assert!(entry.state.iter().all(|v| v.load(Ordering::Relaxed) == 0));
        assert!(entry.ready.iter().all(|r| !r.load(Ordering::Relaxed)));
    }
    #[test]
    fn retention_is_bounded_and_recent_sizes_can_replace_old_sizes() {
        let mut pool = Pool::new(27);
        assert!(pool.put(Storage::fresh(3).scrub(true)));
        assert!(!pool.put(Storage::fresh(4).scrub(true)));
        assert_eq!(pool.bytes, 27);
        assert!(pool.put(Storage::fresh(2).scrub(true)));
        assert!(pool.take(3).is_none());
        assert_eq!(pool.bytes, 18);
        assert!(pool.take(2).is_some());
        assert_eq!(pool.bytes, 0);
    }
}
