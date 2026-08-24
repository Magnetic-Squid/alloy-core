#![allow(unknown_lints, clippy::incompatible_msrv, missing_docs)]

use alloy_primitives::{Address, B256, keccak256};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn primitives(c: &mut Criterion) {
    let mut g = c.benchmark_group("primitives");
    g.bench_function("address/checksum", |b| {
        let address = Address::random();
        let out = &mut [0u8; 42];
        b.iter(|| {
            let x = address.to_checksum_raw(black_box(out), None);
            black_box(x);
        })
    });
    g.bench_function("keccak256/32", |b| {
        let mut out = B256::random();
        b.iter(|| {
            out = keccak256(out.as_slice());
            black_box(&out);
        });
    });
    g.finish();
}

fn keccak_cache(c: &mut Criterion) {
    #[cfg(feature = "keccak-cache-local")]
    {
        use alloy_primitives::{keccak256_uncached, utils::initialize_local_keccak_cache};

        initialize_local_keccak_cache();
        let mut g = c.benchmark_group("keccak_cache");

        let hit_input = [0x42; 64];
        black_box(keccak256(hit_input));
        g.bench_function("hit/64", |b| b.iter(|| black_box(keccak256(black_box(&hit_input)))));

        let mut miss_input = [0xa7; 64];
        let mut nonce = 0u64;
        g.bench_function("miss/64", |b| {
            b.iter(|| {
                nonce = nonce.wrapping_add(1);
                miss_input[..8].copy_from_slice(&nonce.to_ne_bytes());
                black_box(keccak256(black_box(&miss_input)))
            })
        });

        let mut uncached_input = [0xa7; 64];
        let mut nonce = 0u64;
        g.bench_function("uncached/64", |b| {
            b.iter(|| {
                nonce = nonce.wrapping_add(1);
                uncached_input[..8].copy_from_slice(&nonce.to_ne_bytes());
                black_box(keccak256_uncached(black_box(&uncached_input)))
            })
        });

        g.finish();
    }

    #[cfg(not(feature = "keccak-cache-local"))]
    let _ = c;
}

criterion_group!(benches, primitives, keccak_cache);
criterion_main!(benches);
