//! A minimalistic one-way set associative cache for Keccak256 values.
//!
//! This cache has a fixed size to allow fast access and minimize per-call overhead.

use super::{hint::unlikely, keccak256_impl as keccak256};
use crate::{B256, KECCAK256_EMPTY};
use std::mem::MaybeUninit;
#[cfg(feature = "keccak-cache-metrics")]
use std::time::{Duration, Instant};

#[cfg(feature = "keccak-cache-metrics")]
mod metrics;
#[cfg(feature = "keccak-cache-metrics")]
pub use metrics::{KECCAK_CACHE_TIMING_BUCKET_UPPER_BOUNDS_NS, KeccakCacheMetricsSnapshot};

/// Maximum input length that can be cached.
pub(super) const MAX_INPUT_LEN: usize =
    128 - size_of::<B256>() - size_of::<u8>() - size_of::<usize>();

const COUNT: usize = 1 << 17; // ~131k entries * 128 bytes = 16MiB
type Cache = fixed_cache::Cache<Key, B256, BuildHasher, CacheConfig>;

#[cfg(not(feature = "keccak-cache-local"))]
static GLOBAL_CACHE: Cache = fixed_cache::static_cache!(Key, B256, COUNT, BuildHasher::new());

#[cfg(feature = "keccak-cache-local")]
std::thread_local! {
    static LOCAL_CACHE: Cache = Cache::new(COUNT, BuildHasher::new());
}

struct CacheConfig {}
impl fixed_cache::CacheConfig for CacheConfig {
    const STATS: bool = false;
    const EPOCHS: bool = false;
}

pub(super) fn compute(input: &[u8], imp: impl FnOnce(&[u8]) -> B256) -> B256 {
    if unlikely(input.is_empty() | (input.len() > MAX_INPUT_LEN)) {
        #[cfg(feature = "keccak-cache-metrics")]
        metrics::record_bypass(input.len());
        return if input.is_empty() { KECCAK256_EMPTY } else { keccak256(input) };
    }

    #[cfg(all(not(feature = "keccak-cache-local"), not(feature = "keccak-cache-metrics")))]
    return GLOBAL_CACHE.get_or_insert_with_ref(input, imp, make_key);

    #[cfg(all(feature = "keccak-cache-local", not(feature = "keccak-cache-metrics")))]
    return LOCAL_CACHE
        .with(|local_cache| local_cache.get_or_insert_with_ref(input, imp, make_key));

    #[cfg(all(not(feature = "keccak-cache-local"), feature = "keccak-cache-metrics"))]
    {
        if unlikely(metrics::should_sample_timing()) {
            return compute_sampled_global(input, imp);
        }
        let mut missed = false;
        let output = GLOBAL_CACHE.get_or_insert_with_ref(
            input,
            |input| {
                missed = true;
                imp(input)
            },
            make_key,
        );
        metrics::record_cacheable(input.len(), true, missed);
        output
    }

    #[cfg(all(feature = "keccak-cache-local", feature = "keccak-cache-metrics"))]
    {
        if unlikely(metrics::should_sample_timing()) {
            return compute_sampled_local(input, imp);
        }
        let mut local_missed = false;
        let output = LOCAL_CACHE.with(|local_cache| {
            local_cache.get_or_insert_with_ref(
                input,
                |input| {
                    local_missed = true;
                    imp(input)
                },
                make_key,
            )
        });
        metrics::record_cacheable(input.len(), local_missed, true);
        output
    }
}

#[cfg(all(not(feature = "keccak-cache-local"), feature = "keccak-cache-metrics"))]
#[cold]
fn compute_sampled_global(input: &[u8], imp: impl FnOnce(&[u8]) -> B256) -> B256 {
    let mut global_missed = false;
    let mut hash_duration = None;
    let global_start = Instant::now();
    let output = GLOBAL_CACHE.get_or_insert_with_ref(
        input,
        |input| {
            global_missed = true;
            let hash_start = Instant::now();
            let output = imp(input);
            hash_duration = Some(hash_start.elapsed());
            output
        },
        make_key,
    );
    let global_duration = global_start.elapsed();
    let global_lookup_duration =
        global_duration.saturating_sub(hash_duration.unwrap_or(Duration::ZERO));
    metrics::record_cacheable(input.len(), true, global_missed);
    metrics::record_timing(input.len(), [None, Some(global_lookup_duration), hash_duration]);
    output
}

#[cfg(all(feature = "keccak-cache-local", feature = "keccak-cache-metrics"))]
#[cold]
fn compute_sampled_local(input: &[u8], imp: impl FnOnce(&[u8]) -> B256) -> B256 {
    let mut local_missed = false;
    let mut hash_duration = None;
    let local_start = Instant::now();
    let output = LOCAL_CACHE.with(|local_cache| {
        local_cache.get_or_insert_with_ref(
            input,
            |input| {
                local_missed = true;
                let hash_start = Instant::now();
                let output = imp(input);
                hash_duration = Some(hash_start.elapsed());
                output
            },
            make_key,
        )
    });
    let local_duration = local_start.elapsed();
    let local_lookup_duration =
        local_duration.saturating_sub(hash_duration.unwrap_or(Duration::ZERO));
    metrics::record_cacheable(input.len(), local_missed, true);
    metrics::record_timing(input.len(), [Some(local_lookup_duration), None, hash_duration]);
    output
}

#[cfg(feature = "keccak-cache-local")]
pub(super) fn initialize_local_cache() {
    LOCAL_CACHE.with(|_| {});
}

#[cfg(feature = "keccak-cache-metrics")]
pub(super) fn metrics_snapshot() -> KeccakCacheMetricsSnapshot {
    metrics::snapshot()
}

type BuildHasher = std::hash::BuildHasherDefault<Hasher>;
#[derive(Default)]
struct Hasher(u64);

impl std::hash::Hasher for Hasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // This is tricky because our most common inputs are medium length: 16..=88
        // `foldhash` and `rapidhash` have a fast-path for ..16 bytes and outline the rest,
        // but really we want the opposite, or at least the 16.. path to be inlined.

        // SAFETY: `bytes.len()` is checked to be within the bounds of `MAX_INPUT_LEN` by caller.
        unsafe { core::hint::assert_unchecked(bytes.len() <= MAX_INPUT_LEN) };
        if bytes.len() <= 16 {
            super::hint::cold_path();
        }
        self.0 = rapidhash::v3::rapidhash_v3_micro_inline::<false, false>(
            bytes,
            const { &rapidhash::v3::RapidSecrets::seed(0) },
        );
    }

    // We can just skip hashing the length prefix entirely since we know it's always
    // `<=MAX_INPUT_LEN`, and the hash is good enough.

    // `write_length_prefix` calls `write_usize` by default.
    #[inline]
    fn write_usize(&mut self, i: usize) {
        debug_assert!(i <= MAX_INPUT_LEN, "{i} > {MAX_INPUT_LEN}")
    }

    #[cfg(feature = "nightly")]
    #[inline]
    fn write_length_prefix(&mut self, len: usize) {
        debug_assert!(len <= MAX_INPUT_LEN, "{len} > {MAX_INPUT_LEN}")
    }
}

#[derive(Clone, Copy)]
struct Key {
    len: u8,
    data: [MaybeUninit<u8>; MAX_INPUT_LEN],
}

#[inline]
fn make_key(input: &[u8]) -> Key {
    let mut data = [MaybeUninit::uninit(); MAX_INPUT_LEN];
    unsafe { std::ptr::copy_nonoverlapping(input.as_ptr(), data.as_mut_ptr().cast(), input.len()) };
    Key { len: input.len() as u8, data }
}

impl PartialEq for Key {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}
impl Eq for Key {}

impl std::borrow::Borrow<[u8]> for Key {
    #[inline]
    fn borrow(&self) -> &[u8] {
        self.get()
    }
}

impl std::hash::Hash for Key {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write(self.get());
    }
}

impl Key {
    #[inline]
    const fn get(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr().cast(), self.len as usize) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(size_of::<Key>(), MAX_INPUT_LEN + 1);
        assert_eq!(size_of::<fixed_cache::Bucket<(Key, B256)>>(), 128);
    }

    #[test]
    fn caching() {
        let mut count: usize = 0;
        let mut compute = |input| {
            compute(input, |x| {
                count += 1;
                keccak256(x)
            })
        };

        let input = b"Hello World!";
        let input2 = b"Hello World! 2";

        let a = compute(input);
        let b = compute(input);
        let c = compute(input);
        assert_eq!(a, b);
        assert_eq!(a, c);

        let d = compute(input2);
        let e = compute(input2);
        assert_ne!(a, d);
        assert_eq!(d, e);

        assert_eq!(count, 2);
    }

    #[cfg(feature = "keccak-cache-metrics")]
    #[test]
    fn metrics_count_hits_misses_and_bypasses() {
        let before = metrics::snapshot();
        let input = b"alloy-keccak-cache-metrics-unique-input";

        let first = compute(input, keccak256);
        let second = compute(input, keccak256);
        assert_eq!(first, second);
        assert_eq!(compute(&[], keccak256), KECCAK256_EMPTY);
        let oversized = [0x42; MAX_INPUT_LEN + 1];
        assert_eq!(compute(&oversized, keccak256), keccak256(&oversized));
        metrics::flush_current_thread();

        let after = metrics::snapshot();
        let bucket = metrics::input_size_bucket(input.len());
        assert!(after.misses_by_input_size[bucket] >= before.misses_by_input_size[bucket] + 1);
        #[cfg(feature = "keccak-cache-local")]
        assert!(
            after.local_hits_by_input_size[bucket] >= before.local_hits_by_input_size[bucket] + 1
        );
        #[cfg(not(feature = "keccak-cache-local"))]
        assert!(
            after.global_hits_by_input_size[bucket] >= before.global_hits_by_input_size[bucket] + 1
        );
        assert!(after.empty_bypasses >= before.empty_bypasses + 1);
        assert!(after.oversized_bypasses >= before.oversized_bypasses + 1);
    }

    #[cfg(all(feature = "keccak-cache-local", feature = "keccak-cache-metrics"))]
    #[test]
    fn a_local_miss_is_recomputed_on_another_thread() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };

        let input = b"alloy-keccak-local-cache-cross-thread-unique-input";
        let before = metrics::snapshot();
        let computations = Arc::new(AtomicUsize::new(0));

        let first_computations = Arc::clone(&computations);
        let first = std::thread::spawn(move || {
            let output = compute(input, |input| {
                first_computations.fetch_add(1, Ordering::Relaxed);
                keccak256(input)
            });
            metrics::flush_current_thread();
            output
        })
        .join()
        .unwrap();
        let after_first = metrics::snapshot();
        let bucket = metrics::input_size_bucket(input.len());
        assert!(
            after_first.misses_by_input_size[bucket] >= before.misses_by_input_size[bucket] + 1
        );

        let second_computations = Arc::clone(&computations);
        let second = std::thread::spawn(move || {
            let output = compute(input, |input| {
                second_computations.fetch_add(1, Ordering::Relaxed);
                keccak256(input)
            });
            metrics::flush_current_thread();
            output
        })
        .join()
        .unwrap();
        let after_second = metrics::snapshot();

        assert_eq!(first, second);
        assert_eq!(computations.load(Ordering::Relaxed), 2);
        assert!(
            after_second.misses_by_input_size[bucket]
                >= after_first.misses_by_input_size[bucket] + 1
        );
    }

    #[cfg(feature = "keccak-cache-metrics")]
    #[test]
    fn sampled_timings_cover_cache_and_hash_stages() {
        let before = metrics::snapshot();
        for nonce in 0..u64::from(metrics::TIMING_SAMPLE_INTERVAL) * 2 {
            let mut input = [0xa7; 64];
            input[..8].copy_from_slice(&nonce.to_ne_bytes());
            compute(&input, keccak256);
        }
        metrics::flush_current_thread();

        let after = metrics::snapshot();
        let bucket = metrics::input_size_bucket(64);
        #[cfg(feature = "keccak-cache-local")]
        assert!(
            after.sampled_timing_counts[metrics::LOCAL_LOOKUP_STAGE][bucket]
                > before.sampled_timing_counts[metrics::LOCAL_LOOKUP_STAGE][bucket]
        );
        #[cfg(not(feature = "keccak-cache-local"))]
        assert!(
            after.sampled_timing_counts[metrics::GLOBAL_LOOKUP_STAGE][bucket]
                > before.sampled_timing_counts[metrics::GLOBAL_LOOKUP_STAGE][bucket]
        );
        assert!(
            after.sampled_timing_counts[metrics::HASH_COMPUTE_STAGE][bucket]
                > before.sampled_timing_counts[metrics::HASH_COMPUTE_STAGE][bucket]
        );
    }
}
