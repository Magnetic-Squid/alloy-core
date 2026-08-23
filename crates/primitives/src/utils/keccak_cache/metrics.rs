use core::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) const INPUT_SIZE_BUCKETS: usize = 4;
const FLUSH_INTERVAL: u16 = 1_024;

/// Cumulative global Keccak cache diagnostic counters.
///
/// Cacheable inputs are split into `1..=16`, `17..=32`, `33..=64`, and
/// `65..=MAX_INPUT_LEN` byte buckets. Empty and oversized inputs bypass the cache and have
/// separate counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeccakCacheMetricsSnapshot {
    /// Cache hits by cacheable input-size bucket.
    pub hits_by_input_size: [u64; INPUT_SIZE_BUCKETS],
    /// Cache misses by cacheable input-size bucket.
    pub misses_by_input_size: [u64; INPUT_SIZE_BUCKETS],
    /// Empty inputs that returned the constant empty hash without consulting the cache.
    pub empty_bypasses: u64,
    /// Inputs larger than the maximum cacheable length that were hashed without the cache.
    pub oversized_bypasses: u64,
}

#[derive(Clone, Copy, Default)]
struct LocalMetrics {
    hits_by_input_size: [u64; INPUT_SIZE_BUCKETS],
    misses_by_input_size: [u64; INPUT_SIZE_BUCKETS],
    empty_bypasses: u64,
    oversized_bypasses: u64,
    pending: u16,
}

static HITS_BY_INPUT_SIZE: [AtomicU64; INPUT_SIZE_BUCKETS] =
    [const { AtomicU64::new(0) }; INPUT_SIZE_BUCKETS];
static MISSES_BY_INPUT_SIZE: [AtomicU64; INPUT_SIZE_BUCKETS] =
    [const { AtomicU64::new(0) }; INPUT_SIZE_BUCKETS];
static EMPTY_BYPASSES: AtomicU64 = AtomicU64::new(0);
static OVERSIZED_BYPASSES: AtomicU64 = AtomicU64::new(0);

std::thread_local! {
    static LOCAL_METRICS: Cell<LocalMetrics> = const { Cell::new(LocalMetrics {
        hits_by_input_size: [0; INPUT_SIZE_BUCKETS],
        misses_by_input_size: [0; INPUT_SIZE_BUCKETS],
        empty_bypasses: 0,
        oversized_bypasses: 0,
        pending: 0,
    }) };
}

#[inline]
pub(super) fn record_cacheable(input_len: usize, missed: bool) {
    record(|metrics| {
        let bucket = input_size_bucket(input_len);
        if missed {
            metrics.misses_by_input_size[bucket] += 1;
        } else {
            metrics.hits_by_input_size[bucket] += 1;
        }
    });
}

#[cold]
pub(super) fn record_bypass(input_len: usize) {
    record(|metrics| {
        if input_len == 0 {
            metrics.empty_bypasses += 1;
        } else {
            metrics.oversized_bypasses += 1;
        }
    });
}

#[inline]
fn record(update: impl FnOnce(&mut LocalMetrics)) {
    LOCAL_METRICS.with(|local| {
        let mut metrics = local.get();
        update(&mut metrics);
        metrics.pending += 1;
        if metrics.pending == FLUSH_INTERVAL {
            flush(&mut metrics);
        }
        local.set(metrics);
    });
}

#[inline]
pub(super) const fn input_size_bucket(input_len: usize) -> usize {
    match input_len {
        0..=16 => 0,
        17..=32 => 1,
        33..=64 => 2,
        _ => 3,
    }
}

pub(super) fn snapshot() -> KeccakCacheMetricsSnapshot {
    KeccakCacheMetricsSnapshot {
        hits_by_input_size: core::array::from_fn(|index| {
            HITS_BY_INPUT_SIZE[index].load(Ordering::Relaxed)
        }),
        misses_by_input_size: core::array::from_fn(|index| {
            MISSES_BY_INPUT_SIZE[index].load(Ordering::Relaxed)
        }),
        empty_bypasses: EMPTY_BYPASSES.load(Ordering::Relaxed),
        oversized_bypasses: OVERSIZED_BYPASSES.load(Ordering::Relaxed),
    }
}

#[cfg(test)]
pub(super) fn flush_current_thread() {
    LOCAL_METRICS.with(|local| {
        let mut metrics = local.get();
        flush(&mut metrics);
        local.set(metrics);
    });
}

fn flush(metrics: &mut LocalMetrics) {
    for (global, local) in HITS_BY_INPUT_SIZE.iter().zip(metrics.hits_by_input_size) {
        global.fetch_add(local, Ordering::Relaxed);
    }
    for (global, local) in MISSES_BY_INPUT_SIZE.iter().zip(metrics.misses_by_input_size) {
        global.fetch_add(local, Ordering::Relaxed);
    }
    EMPTY_BYPASSES.fetch_add(metrics.empty_bypasses, Ordering::Relaxed);
    OVERSIZED_BYPASSES.fetch_add(metrics.oversized_bypasses, Ordering::Relaxed);
    *metrics = LocalMetrics::default();
}
