//! A minimalistic one-way set associative cache for Keccak256 values.
//!
//! This cache has a fixed size to allow fast access and minimize per-call overhead.
//!
//! With the `keccak-cache-stats` feature, hit/miss/insert/collision counts are
//! tracked in the global [`KECCAK_CACHE_STATS`] static.

use super::{hint::unlikely, keccak256_impl as keccak256};
use crate::{B256, KECCAK256_EMPTY};
use std::mem::MaybeUninit;

#[cfg(not(feature = "keccak-cache-local"))]
use std::sync::OnceLock;
#[cfg(feature = "keccak-cache-local")]
use std::{
    alloc::{Layout, alloc_zeroed, handle_alloc_error},
    cell::{Cell, UnsafeCell},
    ptr,
};

#[cfg(all(feature = "keccak-cache-stats", not(feature = "keccak-cache-local")))]
pub use stats::{KECCAK_CACHE_STATS, KeccakCacheStats};

/// Maximum input length that can be cached.
pub(super) const MAX_INPUT_LEN: usize =
    128 - size_of::<B256>() - size_of::<u8>() - size_of::<usize>();

const DEFAULT_COUNT: usize = 1 << 17; // ~131k entries * 128 bytes = 16MiB
#[cfg(feature = "keccak-cache-local")]
const INDEX_MASK: usize = DEFAULT_COUNT - 1;

#[cfg(not(feature = "keccak-cache-local"))]
static CACHE: OnceLock<Cache> = OnceLock::new();

#[cfg(not(feature = "keccak-cache-local"))]
type Cache = fixed_cache::Cache<Key, B256, BuildHasher, CacheConfig>;

#[cfg(not(feature = "keccak-cache-local"))]
struct CacheConfig {}
#[cfg(not(feature = "keccak-cache-local"))]
impl fixed_cache::CacheConfig for CacheConfig {
    const STATS: bool = cfg!(feature = "keccak-cache-stats");
    const EPOCHS: bool = false;
}

#[cfg(feature = "keccak-cache-local")]
std::thread_local! {
    static LOCAL_CACHE: LocalCache = LocalCache::new();
}

#[cfg(feature = "keccak-cache-local")]
pub(super) fn initialize_local_cache() {
    LOCAL_CACHE.with(|_| {});
}

/// Initializes the process-global keccak cache with `entries` buckets.
///
/// Returns `true` if this call initialized the cache, and `false` if the cache was already
/// initialized by an earlier call or by the first cached hash computation. If this is never called,
/// the cache is initialized lazily with the default size on first use.
///
/// # Panics
///
/// Panics if `entries` is not a power of two or is less than 4.
#[must_use]
#[cfg(not(feature = "keccak-cache-local"))]
pub fn init_keccak_cache(entries: usize) -> bool {
    init_cache(&CACHE, entries)
}

pub(super) fn compute(input: &[u8], imp: impl FnOnce(&[u8]) -> B256) -> B256 {
    if unlikely(input.is_empty() | (input.len() > MAX_INPUT_LEN)) {
        return if input.is_empty() { KECCAK256_EMPTY } else { keccak256(input) };
    }

    #[cfg(feature = "keccak-cache-local")]
    return LOCAL_CACHE.with(|cache| cache.get_or_insert(input, imp));

    #[cfg(not(feature = "keccak-cache-local"))]
    {
        let cache = CACHE.get_or_init(default_cache);
        cache.get_or_insert_with_ref(input, imp, make_key)
    }
}

#[cfg(not(feature = "keccak-cache-local"))]
fn init_cache(cache: &OnceLock<Cache>, entries: usize) -> bool {
    if cache.get().is_some() {
        return false;
    }
    cache.set(new_cache(entries)).is_ok()
}

#[cfg(not(feature = "keccak-cache-local"))]
fn default_cache() -> Cache {
    new_cache(DEFAULT_COUNT)
}

#[cfg(not(feature = "keccak-cache-local"))]
fn new_cache(entries: usize) -> Cache {
    let cache = fixed_cache::Cache::new(entries, BuildHasher::new());
    #[cfg(feature = "keccak-cache-stats")]
    let cache = cache.with_stats(Some(fixed_cache::Stats::new(&stats::KECCAK_CACHE_STATS)));
    cache
}

#[cfg(not(feature = "keccak-cache-local"))]
type BuildHasher = std::hash::BuildHasherDefault<Hasher>;
#[cfg(not(feature = "keccak-cache-local"))]
#[derive(Default)]
struct Hasher(u64);

#[cfg(not(feature = "keccak-cache-local"))]
impl std::hash::Hasher for Hasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.0 = rapid_hash(bytes);
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

#[cfg(feature = "keccak-cache-local")]
struct LocalCache {
    buckets: Box<[LocalBucket]>,
}

#[cfg(feature = "keccak-cache-local")]
impl LocalCache {
    fn new() -> Self {
        let layout = Layout::array::<LocalBucket>(DEFAULT_COUNT).unwrap();
        let buckets = unsafe { alloc_zeroed(layout).cast::<LocalBucket>() };
        if buckets.is_null() {
            handle_alloc_error(layout);
        }
        let buckets = ptr::slice_from_raw_parts_mut(buckets, DEFAULT_COUNT);

        // Zero is an empty tag; entries remain uninitialized until their tag is published.
        Self { buckets: unsafe { Box::from_raw(buckets) } }
    }

    #[inline]
    fn get_or_insert(&self, input: &[u8], compute: impl FnOnce(&[u8]) -> B256) -> B256 {
        let hash = hash_input(input);
        let bucket = unsafe { self.buckets.get_unchecked(hash & INDEX_MASK) };
        let tag = (hash & !INDEX_MASK) | 1;

        if let Some(output) = bucket.get(input, tag) {
            return output;
        }

        let output = compute(input);
        bucket.insert(input, output, tag);
        output
    }
}

#[cfg(feature = "keccak-cache-local")]
#[repr(C, align(128))]
struct LocalBucket {
    tag: Cell<usize>,
    entry: UnsafeCell<MaybeUninit<(Key, B256)>>,
}

#[cfg(feature = "keccak-cache-local")]
impl LocalBucket {
    #[inline]
    fn get(&self, input: &[u8], tag: usize) -> Option<B256> {
        if self.tag.get() != tag {
            return None;
        }

        // The nonzero tag is written after initialization in this thread-confined cache.
        let (key, output) = unsafe { (*self.entry.get()).assume_init_ref() };
        (key.get() == input).then_some(*output)
    }

    #[inline]
    fn insert(&self, input: &[u8], output: B256, tag: usize) {
        // The cache is thread-local and the entry has no drop glue.
        unsafe { ptr::write(self.entry.get().cast(), (make_key(input), output)) };
        self.tag.set(tag);
    }
}

#[inline]
fn rapid_hash(bytes: &[u8]) -> u64 {
    // SAFETY: the caller rejects inputs larger than `MAX_INPUT_LEN`.
    unsafe { core::hint::assert_unchecked(bytes.len() <= MAX_INPUT_LEN) };
    if bytes.len() <= 16 {
        super::hint::cold_path();
    }

    rapidhash::v3::rapidhash_v3_micro_inline::<false, false>(
        bytes,
        const { &rapidhash::v3::RapidSecrets::seed(0) },
    )
}

#[cfg(feature = "keccak-cache-local")]
#[inline]
fn hash_input(input: &[u8]) -> usize {
    let hash = rapid_hash(input);
    if cfg!(target_pointer_width = "32") {
        ((hash >> 32) as usize) ^ hash as usize
    } else {
        hash as usize
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

#[cfg(all(feature = "keccak-cache-stats", not(feature = "keccak-cache-local")))]
mod stats {
    use super::Key;
    use crate::B256;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Counters for the global keccak cache.
    ///
    /// Accessed via the [`KECCAK_CACHE_STATS`] static. All counters use relaxed atomics.
    ///
    /// Available only with the `keccak-cache-stats` feature.
    #[derive(Debug, Default)]
    pub struct KeccakCacheStats {
        hits: AtomicU64,
        misses: AtomicU64,
        inserts: AtomicU64,
        collisions: AtomicU64,
    }

    impl KeccakCacheStats {
        const fn new() -> Self {
            Self {
                hits: AtomicU64::new(0),
                misses: AtomicU64::new(0),
                inserts: AtomicU64::new(0),
                collisions: AtomicU64::new(0),
            }
        }

        /// Returns the number of cache hits.
        #[inline]
        pub fn hits(&self) -> u64 {
            self.hits.load(Ordering::Relaxed)
        }

        /// Returns the number of cache misses.
        #[inline]
        pub fn misses(&self) -> u64 {
            self.misses.load(Ordering::Relaxed)
        }

        /// Returns the number of inserted entries.
        ///
        /// Includes inserts that evicted a different key on hash collision (see
        /// [`collisions`](Self::collisions)).
        #[inline]
        pub fn inserts(&self) -> u64 {
            self.inserts.load(Ordering::Relaxed)
        }

        /// Returns the number of collisions (a different key was evicted on insert).
        #[inline]
        pub fn collisions(&self) -> u64 {
            self.collisions.load(Ordering::Relaxed)
        }

        /// Resets all counters to zero.
        pub fn reset(&self) {
            self.hits.store(0, Ordering::Relaxed);
            self.misses.store(0, Ordering::Relaxed);
            self.inserts.store(0, Ordering::Relaxed);
            self.collisions.store(0, Ordering::Relaxed);
        }
    }

    impl fixed_cache::StatsHandler<Key, B256> for &'static KeccakCacheStats {
        #[inline]
        fn on_hit(&self, _key: &Key, _value: &B256) {
            self.hits.fetch_add(1, Ordering::Relaxed);
        }

        #[inline]
        fn on_miss(&self, _key: fixed_cache::AnyRef<'_>) {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }

        #[inline]
        fn on_insert(&self, key: &Key, _value: &B256, evicted: Option<(&Key, &B256)>) {
            match evicted {
                // Race: another thread inserted the same key concurrently. Same input
                // always produces the same hash, so this is a no-op redundant write.
                Some((old, _)) if old == key => {}
                Some(_) => {
                    self.inserts.fetch_add(1, Ordering::Relaxed);
                    self.collisions.fetch_add(1, Ordering::Relaxed);
                }
                None => {
                    self.inserts.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    /// Global counters for the keccak cache.
    ///
    /// Available only with the `keccak-cache-stats` feature.
    pub static KECCAK_CACHE_STATS: KeccakCacheStats = KeccakCacheStats::new();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(size_of::<Key>(), MAX_INPUT_LEN + 1);
        #[cfg(not(feature = "keccak-cache-local"))]
        assert_eq!(size_of::<fixed_cache::Bucket<(Key, B256)>>(), 128);
        #[cfg(feature = "keccak-cache-local")]
        assert_eq!(size_of::<LocalBucket>(), 128);
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
}
