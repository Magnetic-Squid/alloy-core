use core::cell::{Cell, RefCell};
use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

pub(super) const INPUT_SIZE_BUCKETS: usize = 4;
const FLUSH_INTERVAL: u16 = 1_024;
pub(super) const TIMING_SAMPLE_INTERVAL: u16 = 1_024;
const TIMING_FLUSH_INTERVAL: u8 = 16;

#[cfg(all(test, feature = "keccak-cache-local"))]
pub(super) const LOCAL_LOOKUP_STAGE: usize = 0;
#[cfg(all(test, not(feature = "keccak-cache-local")))]
pub(super) const GLOBAL_LOOKUP_STAGE: usize = 1;
#[cfg(test)]
pub(super) const HASH_COMPUTE_STAGE: usize = 2;
const TIMING_STAGES: usize = 3;

/// Inclusive upper bounds for sampled cache-stage durations, in nanoseconds.
pub const KECCAK_CACHE_TIMING_BUCKET_UPPER_BOUNDS_NS: [u64; 17] = [
    32, 64, 128, 256, 512, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000, 512_000,
    2_000_000, 8_000_000, 32_000_000,
];
const TIMING_BUCKETS: usize = KECCAK_CACHE_TIMING_BUCKET_UPPER_BOUNDS_NS.len();

/// Cumulative global Keccak cache diagnostic counters.
///
/// Cacheable inputs are split into `1..=16`, `17..=32`, `33..=64`, and
/// `65..=MAX_INPUT_LEN` byte buckets. Empty and oversized inputs bypass the cache and have
/// separate counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeccakCacheMetricsSnapshot {
    /// Hits served by the calling thread's local cache, by cacheable input-size bucket.
    pub local_hits_by_input_size: [u64; INPUT_SIZE_BUCKETS],
    /// Local misses served by the shared process cache, by cacheable input-size bucket.
    pub global_hits_by_input_size: [u64; INPUT_SIZE_BUCKETS],
    /// Misses in both the local and shared caches, by cacheable input-size bucket.
    pub misses_by_input_size: [u64; INPUT_SIZE_BUCKETS],
    /// Empty inputs that returned the constant empty hash without consulting the cache.
    pub empty_bypasses: u64,
    /// Inputs larger than the maximum cacheable length that were hashed without the cache.
    pub oversized_bypasses: u64,
    /// Sample counts by cache stage and cacheable input-size bucket.
    pub sampled_timing_counts: [[u64; INPUT_SIZE_BUCKETS]; TIMING_STAGES],
    /// Cumulative sampled duration in nanoseconds by cache stage and input-size bucket.
    pub sampled_timing_total_ns: [[u64; INPUT_SIZE_BUCKETS]; TIMING_STAGES],
    /// Non-cumulative sampled duration buckets by cache stage and input-size bucket.
    pub sampled_timing_buckets: [[[u64; TIMING_BUCKETS]; INPUT_SIZE_BUCKETS]; TIMING_STAGES],
}

#[derive(Clone, Copy, Default)]
struct LocalMetrics {
    local_hits_by_input_size: [u64; INPUT_SIZE_BUCKETS],
    global_hits_by_input_size: [u64; INPUT_SIZE_BUCKETS],
    misses_by_input_size: [u64; INPUT_SIZE_BUCKETS],
    empty_bypasses: u64,
    oversized_bypasses: u64,
    pending: u16,
}

#[derive(Default)]
struct LocalTimingMetrics {
    counts: [[u64; INPUT_SIZE_BUCKETS]; TIMING_STAGES],
    total_ns: [[u64; INPUT_SIZE_BUCKETS]; TIMING_STAGES],
    buckets: [[[u64; TIMING_BUCKETS]; INPUT_SIZE_BUCKETS]; TIMING_STAGES],
    pending: u8,
}

static LOCAL_HITS_BY_INPUT_SIZE: [AtomicU64; INPUT_SIZE_BUCKETS] =
    [const { AtomicU64::new(0) }; INPUT_SIZE_BUCKETS];
static GLOBAL_HITS_BY_INPUT_SIZE: [AtomicU64; INPUT_SIZE_BUCKETS] =
    [const { AtomicU64::new(0) }; INPUT_SIZE_BUCKETS];
static MISSES_BY_INPUT_SIZE: [AtomicU64; INPUT_SIZE_BUCKETS] =
    [const { AtomicU64::new(0) }; INPUT_SIZE_BUCKETS];
static EMPTY_BYPASSES: AtomicU64 = AtomicU64::new(0);
static OVERSIZED_BYPASSES: AtomicU64 = AtomicU64::new(0);
static SAMPLED_TIMING_COUNTS: [[AtomicU64; INPUT_SIZE_BUCKETS]; TIMING_STAGES] =
    [const { [const { AtomicU64::new(0) }; INPUT_SIZE_BUCKETS] }; TIMING_STAGES];
static SAMPLED_TIMING_TOTAL_NS: [[AtomicU64; INPUT_SIZE_BUCKETS]; TIMING_STAGES] =
    [const { [const { AtomicU64::new(0) }; INPUT_SIZE_BUCKETS] }; TIMING_STAGES];
static SAMPLED_TIMING_BUCKETS: [[[AtomicU64; TIMING_BUCKETS]; INPUT_SIZE_BUCKETS]; TIMING_STAGES] =
    [const { [const { [const { AtomicU64::new(0) }; TIMING_BUCKETS] }; INPUT_SIZE_BUCKETS] };
        TIMING_STAGES];

std::thread_local! {
    static LOCAL_METRICS: Cell<LocalMetrics> = const { Cell::new(LocalMetrics {
        local_hits_by_input_size: [0; INPUT_SIZE_BUCKETS],
        global_hits_by_input_size: [0; INPUT_SIZE_BUCKETS],
        misses_by_input_size: [0; INPUT_SIZE_BUCKETS],
        empty_bypasses: 0,
        oversized_bypasses: 0,
        pending: 0,
    }) };
    static TIMING_SAMPLE_CURSOR: Cell<u16> = const { Cell::new(0) };
    static LOCAL_TIMING_METRICS: RefCell<LocalTimingMetrics> =
        RefCell::new(LocalTimingMetrics::default());
}

#[inline]
pub(super) fn should_sample_timing() -> bool {
    TIMING_SAMPLE_CURSOR.with(|cursor| {
        let next = cursor.get().wrapping_add(1);
        cursor.set(next);
        next & (TIMING_SAMPLE_INTERVAL - 1) == 0
    })
}

#[inline]
pub(super) fn record_cacheable(input_len: usize, local_missed: bool, global_missed: bool) {
    record(|metrics| {
        let bucket = input_size_bucket(input_len);
        if !local_missed {
            metrics.local_hits_by_input_size[bucket] += 1;
        } else if !global_missed {
            metrics.global_hits_by_input_size[bucket] += 1;
        } else {
            metrics.misses_by_input_size[bucket] += 1;
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
pub(super) fn record_timing(input_len: usize, durations: [Option<Duration>; TIMING_STAGES]) {
    LOCAL_TIMING_METRICS.with(|local| {
        let mut metrics = local.borrow_mut();
        let input_bucket = input_size_bucket(input_len);
        for (stage, duration) in durations.into_iter().enumerate() {
            let Some(duration) = duration else { continue };
            let duration_ns = duration.as_nanos().min(u64::MAX as u128) as u64;
            metrics.counts[stage][input_bucket] += 1;
            metrics.total_ns[stage][input_bucket] =
                metrics.total_ns[stage][input_bucket].saturating_add(duration_ns);
            if let Some(bucket) = KECCAK_CACHE_TIMING_BUCKET_UPPER_BOUNDS_NS
                .iter()
                .position(|upper_bound| duration_ns <= *upper_bound)
            {
                metrics.buckets[stage][input_bucket][bucket] += 1;
            }
        }
        metrics.pending += 1;
        if metrics.pending == TIMING_FLUSH_INTERVAL {
            flush_timing(&mut metrics);
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
        local_hits_by_input_size: core::array::from_fn(|index| {
            LOCAL_HITS_BY_INPUT_SIZE[index].load(Ordering::Relaxed)
        }),
        global_hits_by_input_size: core::array::from_fn(|index| {
            GLOBAL_HITS_BY_INPUT_SIZE[index].load(Ordering::Relaxed)
        }),
        misses_by_input_size: core::array::from_fn(|index| {
            MISSES_BY_INPUT_SIZE[index].load(Ordering::Relaxed)
        }),
        empty_bypasses: EMPTY_BYPASSES.load(Ordering::Relaxed),
        oversized_bypasses: OVERSIZED_BYPASSES.load(Ordering::Relaxed),
        sampled_timing_counts: core::array::from_fn(|stage| {
            core::array::from_fn(|input| {
                SAMPLED_TIMING_COUNTS[stage][input].load(Ordering::Relaxed)
            })
        }),
        sampled_timing_total_ns: core::array::from_fn(|stage| {
            core::array::from_fn(|input| {
                SAMPLED_TIMING_TOTAL_NS[stage][input].load(Ordering::Relaxed)
            })
        }),
        sampled_timing_buckets: core::array::from_fn(|stage| {
            core::array::from_fn(|input| {
                core::array::from_fn(|bucket| {
                    SAMPLED_TIMING_BUCKETS[stage][input][bucket].load(Ordering::Relaxed)
                })
            })
        }),
    }
}

#[cfg(test)]
pub(super) fn flush_current_thread() {
    LOCAL_METRICS.with(|local| {
        let mut metrics = local.get();
        flush(&mut metrics);
        local.set(metrics);
    });
    LOCAL_TIMING_METRICS.with(|local| flush_timing(&mut local.borrow_mut()));
}

fn flush(metrics: &mut LocalMetrics) {
    for (global, local) in LOCAL_HITS_BY_INPUT_SIZE.iter().zip(metrics.local_hits_by_input_size) {
        global.fetch_add(local, Ordering::Relaxed);
    }
    for (global, local) in GLOBAL_HITS_BY_INPUT_SIZE.iter().zip(metrics.global_hits_by_input_size) {
        global.fetch_add(local, Ordering::Relaxed);
    }
    for (global, local) in MISSES_BY_INPUT_SIZE.iter().zip(metrics.misses_by_input_size) {
        global.fetch_add(local, Ordering::Relaxed);
    }
    EMPTY_BYPASSES.fetch_add(metrics.empty_bypasses, Ordering::Relaxed);
    OVERSIZED_BYPASSES.fetch_add(metrics.oversized_bypasses, Ordering::Relaxed);
    *metrics = LocalMetrics::default();
}

fn flush_timing(metrics: &mut LocalTimingMetrics) {
    for stage in 0..TIMING_STAGES {
        for input in 0..INPUT_SIZE_BUCKETS {
            let count = metrics.counts[stage][input];
            if count != 0 {
                SAMPLED_TIMING_COUNTS[stage][input].fetch_add(count, Ordering::Relaxed);
            }
            let total_ns = metrics.total_ns[stage][input];
            if total_ns != 0 {
                SAMPLED_TIMING_TOTAL_NS[stage][input].fetch_add(total_ns, Ordering::Relaxed);
            }
            for bucket in 0..TIMING_BUCKETS {
                let count = metrics.buckets[stage][input][bucket];
                if count != 0 {
                    SAMPLED_TIMING_BUCKETS[stage][input][bucket]
                        .fetch_add(count, Ordering::Relaxed);
                }
            }
        }
    }
    *metrics = LocalTimingMetrics::default();
}
