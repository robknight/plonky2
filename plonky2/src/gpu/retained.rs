//! Per-context budget for allocations retained across proofs.
use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub(super) struct RetainedBudget {
    used: AtomicU64,
    limit: u64,
}

impl RetainedBudget {
    pub(super) fn new(limit: u64) -> Self {
        Self {
            used: AtomicU64::new(0),
            limit,
        }
    }

    pub(super) fn reserve(self: &Arc<Self>, bytes: u64) -> Option<Arc<Reservation>> {
        self.used
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                used.checked_add(bytes).filter(|&total| total <= self.limit)
            })
            .ok()?;
        Some(Arc::new(Reservation {
            budget: self.clone(),
            bytes,
        }))
    }

    #[cfg(test)]
    pub(super) fn bytes(&self) -> u64 {
        self.used.load(Ordering::Relaxed)
    }
}

pub(super) struct Reservation {
    budget: Arc<RetainedBudget>,
    bytes: u64,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clones_share_one_reservation_until_the_last_owner_drops() {
        let budget = Arc::new(RetainedBudget::new(100));
        let allocation = budget.reserve(80).unwrap();
        let clone = allocation.clone();
        assert_eq!(budget.bytes(), 80);
        assert!(budget.reserve(21).is_none());
        drop(allocation);
        assert_eq!(budget.bytes(), 80);
        drop(clone);
        assert_eq!(budget.bytes(), 0);
        assert!(budget.reserve(100).is_some());
    }

    #[test]
    fn concurrent_admission_cannot_exceed_the_budget() {
        let budget = Arc::new(RetainedBudget::new(100));
        let barrier = std::sync::Barrier::new(8);
        std::thread::scope(|scope| {
            let handles = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        let reservation = budget.reserve(30);
                        barrier.wait();
                        assert!(budget.bytes() <= 100);
                        reservation
                    })
                })
                .collect::<Vec<_>>();
            let reservations = handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(reservations.iter().filter(|r| r.is_some()).count(), 3);
            assert_eq!(budget.bytes(), 90);
        });
        assert_eq!(budget.bytes(), 0);
    }
}
