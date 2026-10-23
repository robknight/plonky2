//! Per-key initialization keeps expensive compilation outside the map lock.

use core::hash::Hash;
use std::sync::{Arc, Mutex, OnceLock};

use hashbrown::HashMap;

pub(super) struct GpuCache<K, V> {
    slots: Mutex<HashMap<K, Arc<OnceLock<V>>>>,
}

impl<K, V> Default for GpuCache<K, V> {
    fn default() -> Self {
        Self {
            slots: Mutex::new(HashMap::new()),
        }
    }
}

impl<K: Eq + Hash, V: Clone> GpuCache<K, V> {
    pub(super) fn get_or_init(&self, key: K, build: impl FnOnce() -> V) -> V {
        let slot = self.slots.lock().unwrap().entry(key).or_default().clone();
        slot.get_or_init(build).clone()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{mpsc, Barrier};
    use std::time::Duration;

    use super::*;

    #[test]
    fn a_blocked_builder_does_not_block_unrelated_hits() {
        let cache = GpuCache::default();
        cache.get_or_init(2, || 2);
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
        let cache = GpuCache::default();
        let calls = AtomicUsize::new(0);
        let barrier = Barrier::new(8);
        std::thread::scope(|scope| {
            let handles = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        cache.get_or_init(1, || {
                            calls.fetch_add(1, Ordering::Relaxed);
                            Arc::new(7)
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
}
