use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use reed_solomon::codec::{encode, reconstruct_hot, reconstruct_scalar, split};

const K: usize = 8; // data shards - fixed
const M: usize = 2; // parity shards - fixed

fn make_shards_with_erasures(shard_len: usize) -> Vec<Option<Vec<u8>>> {
    // Create original data
    let data: Vec<u8> = (0..K * shard_len)
        .map(|i| ((i * 31 + 17) % 256) as u8)
        .collect();

    // Split into data shards
    let data_shards = split(&data, K);

    // Encode to get parity shards
    let parity_shards = encode(&data_shards, M).unwrap();

    // Combine into full shard set with Option wrapper
    let mut shards: Vec<Option<Vec<u8>>> = Vec::with_capacity(K + M);
    for shard in data_shards {
        shards.push(Some(shard));
    }
    for shard in parity_shards {
        shards.push(Some(shard));
    }

    // Erase 2 shards (simulating data loss)
    shards[0] = None;
    shards[1] = None;

    shards
}

fn bench_reconstruct(c: &mut Criterion) {
    let shard_sizes: Vec<usize> = vec![
        1 << 10,   // 1 KB
        4 << 10,   // 4 KB
        16 << 10,  // 16 KB
        64 << 10,  // 64 KB
        256 << 10, // 256 KB
        1 << 20,   // 1 MB
        4 << 20,   // 4 MB
    ];

    let mut group = c.benchmark_group("reconstruct_throughput");

    for &shard_size in &shard_sizes {
        let shards = make_shards_with_erasures(shard_size);

        // Throughput = data bytes processed per iteration
        let bytes_per_iter = (K * shard_size) as u64;
        group.throughput(Throughput::Bytes(bytes_per_iter));

        // --- Scalar ---
        group.bench_with_input(
            BenchmarkId::new("scalar", format!("{}KB", shard_size / 1024)),
            &shard_size,
            |b, _| {
                b.iter(|| {
                    reconstruct_scalar(black_box(&shards), black_box(K), black_box(M)).unwrap()
                });
            },
        );

        // --- SIMD (reconstruct_hot) ---
        group.bench_with_input(
            BenchmarkId::new("simd", format!("{}KB", shard_size / 1024)),
            &shard_size,
            |b, _| {
                b.iter(|| unsafe {
                    reconstruct_hot(black_box(&shards), black_box(K), black_box(M)).unwrap()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_reconstruct);
criterion_main!(benches);
