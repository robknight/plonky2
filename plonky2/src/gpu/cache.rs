//! Per-key initialization and bounded retention for GPU programs and tables.

use core::hash::Hash;
use std::sync::{Arc, Mutex, OnceLock};

use hashbrown::HashMap;

use super::retained::{Reservation, RetainedBudget};

// Also bounds negative program entries and driver pipelines whose allocation
// sizes are opaque. Active callers keep their own handles after eviction.
const MAX_ENTRIES: usize = 128;

pub(super) trait CacheValue: Clone {
    fn bytes(&self) -> u64;
}

impl<T: CacheValue> CacheValue for Option<T> {
    fn bytes(&self) -> u64 {
        self.as_ref().map_or(0, CacheValue::bytes)
    }
}

pub(super) struct GpuCache<K, V> {
    state: Mutex<CacheState<K, V>>,
    budget: Arc<RetainedBudget>,
}

struct CacheState<K, V> {
    entries: HashMap<K, Entry<V>>,
    clock: u64,
}

struct Entry<V> {
    value: Arc<OnceLock<V>>,
    touched: u64,
    reservation: Option<Arc<Reservation>>,
}

impl<K: Eq + Hash + Clone, V> CacheState<K, V> {
    fn evict_oldest(&mut self, except: Option<&K>) -> bool {
        // Do not evict in-flight initializations: concurrent requests for
        // their key must continue to share the same builder.
        let key = self
            .entries
            .iter()
            .filter(|(k, e)| Some(*k) != except && e.value.get().is_some())
            .min_by_key(|(_, e)| e.touched)
            .map(|(k, _)| k.clone());
        key.is_some_and(|k| self.entries.remove(&k).is_some())
    }
}

impl<K, V> GpuCache<K, V> {
    pub(super) fn new(budget: Arc<RetainedBudget>) -> Self {
        Self {
            state: Mutex::new(CacheState {
                entries: HashMap::new(),
                clock: 0,
            }),
            budget,
        }
    }
}

impl<K: Eq + Hash + Clone, V: CacheValue> GpuCache<K, V> {
    pub(super) fn get_or_init(&self, key: K, build: impl FnOnce() -> V) -> V {
        let slot = {
            let mut state = self.state.lock().unwrap();
            state.clock = state.clock.wrapping_add(1);
            let touched = state.clock;
            if let Some(entry) = state.entries.get_mut(&key) {
                entry.touched = touched;
                entry.value.clone()
            } else {
                if state.entries.len() >= MAX_ENTRIES {
                    state.evict_oldest(None);
                }
                let value = Arc::new(OnceLock::new());
                // If all slots are building, serve this request uncached.
                if state.entries.len() < MAX_ENTRIES {
                    state.entries.insert(
                        key.clone(),
                        Entry {
                            value: value.clone(),
                            touched,
                            reservation: None,
                        },
                    );
                }
                value
            }
        };
        let value = slot.get_or_init(build);
        let bytes = value.bytes();
        if bytes > 0 {
            let mut state = self.state.lock().unwrap();
            if state
                .entries
                .get(&key)
                .is_some_and(|e| Arc::ptr_eq(&e.value, &slot) && e.reservation.is_none())
            {
                let mut reservation = self.budget.reserve(bytes);
                if bytes <= self.budget.limit() {
                    while reservation.is_none() && state.evict_oldest(Some(&key)) {
                        reservation = self.budget.reserve(bytes);
                    }
                }
                if let Some(reservation) = reservation {
                    state.entries.get_mut(&key).unwrap().reservation = Some(reservation);
                } else {
                    state.entries.remove(&key);
                }
            }
        }
        value.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{mpsc, Barrier};
    use std::time::Duration;

    use super::*;

    impl CacheValue for usize {
        fn bytes(&self) -> u64 {
            8
        }
    }
    impl CacheValue for Arc<usize> {
        fn bytes(&self) -> u64 {
            8
        }
    }

    #[test]
    fn a_blocked_builder_does_not_block_unrelated_hits() {
        let cache = GpuCache::new(Arc::new(RetainedBudget::new(1024)));
        cache.get_or_init(2usize, || 2usize);
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (hit_tx, hit_rx) = mpsc::channel();
        std::thread::scope(|scope| {
            let cache = &cache;
            scope.spawn(move || {
                cache.get_or_init(1, || {
                    started_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    1
                })
            });
            started_rx.recv().unwrap();
            scope.spawn(move || {
                hit_tx
                    .send(cache.get_or_init(2, || panic!("cached hit rebuilt")))
                    .unwrap()
            });
            let hit = hit_rx.recv_timeout(Duration::from_secs(5));
            release_tx.send(()).unwrap();
            assert_eq!(hit.unwrap(), 2);
        });
    }

    #[test]
    fn concurrent_requests_initialize_one_key_once() {
        let cache = GpuCache::new(Arc::new(RetainedBudget::new(1024)));
        let calls = AtomicUsize::new(0);
        let barrier = Barrier::new(8);
        std::thread::scope(|scope| {
            let handles = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        cache.get_or_init(1, || {
                            calls.fetch_add(1, Ordering::Relaxed);
                            Arc::new(7usize)
                        })
                    })
                })
                .collect::<Vec<_>>();
            let values = handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(calls.load(Ordering::Relaxed), 1);
            assert!(values.iter().all(|v| Arc::ptr_eq(v, &values[0])));
        });
    }
    #[test]
    fn eviction_obeys_the_shared_budget_and_preserves_active_handles() {
        let budget = Arc::new(RetainedBudget::new(16));
        let cache = GpuCache::new(budget.clone());
        let first = cache.get_or_init(1, || Arc::new(1usize));
        cache.get_or_init(2, || Arc::new(2usize));
        cache.get_or_init(1, || panic!("first entry was lost"));
        cache.get_or_init(3, || Arc::new(3usize));
        assert_eq!(budget.bytes(), 16);
        assert_eq!(*first, 1, "eviction must not invalidate a proof's handle");
        assert!(!cache.state.lock().unwrap().entries.contains_key(&2));
        let other = GpuCache::new(budget.clone());
        other.get_or_init(1, || Arc::new(4usize));
        assert!(
            other.state.lock().unwrap().entries.is_empty(),
            "other caches share the budget"
        );
        drop(cache);
        assert_eq!(budget.bytes(), 0);
        assert_eq!(*first, 1);
    }

    #[test]
    fn oversized_values_are_uncached_and_negative_entries_are_bounded() {
        let budget = Arc::new(RetainedBudget::new(4));
        let cache = GpuCache::new(budget.clone());
        for _ in 0..2 {
            cache.get_or_init(0, || Some(9usize));
        }
        assert!(cache.state.lock().unwrap().entries.is_empty());
        assert_eq!(budget.bytes(), 0);
        for key in 0..MAX_ENTRIES + 10 {
            cache.get_or_init(key, || None);
        }
        assert_eq!(cache.state.lock().unwrap().entries.len(), MAX_ENTRIES);
    }
}
