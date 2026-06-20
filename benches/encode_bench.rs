use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use reed_solomon::codec::encode_hot;
use reed_solomon::matrix::Matrix;
use reed_solomon::simd;

const K: usize = 8; // data shards - fixed
const M: usize = 2; // parity shards - fixed

fn make_data_shards(k: usize, shard_len: usize) -> Vec<Vec<u8>> {
    (0..k)
        .map(|i| {
            (0..shard_len)
                .map(|j| ((i * 17 + j * 31) % 256) as u8)
                .collect()
        })
        .collect()
}

fn bench_roofline(c: &mut Criterion) {
    // Shard sizes spanning L1 (128-192KB) through L2 (4MB) on M-series
    // Total working set = k * shard_size, so:
    //   1KB shards  ->   8KB total (fits L1)
    //   16KB shards -> 128KB total (L1 boundary)
    //   64KB shards -> 512KB total (L2)
    //   512KB shards -> 4MB total (L2 boundary)
    //   1MB shards  ->  8MB total (main memory)
    let shard_sizes: Vec<usize> = vec![
        1 << 10,   // 1 KB
        4 << 10,   // 4 KB
        16 << 10,  // 16 KB
        64 << 10,  // 64 KB
        256 << 10, // 256 KB
        1 << 20,   // 1 MB
        4 << 20,   // 4 MB
    ];

    // Precompute coefficients (the parity rows of the Vandermonde matrix)
    let f = Matrix::vand_fix(M, K).unwrap();
    let coeffs: Vec<u8> = f.elements[K * K..(K + M) * K].to_vec();

    let mut group = c.benchmark_group("encode_throughput");

    for &shard_size in &shard_sizes {
        let data_shards = make_data_shards(K, shard_size);

        // Throughput = data bytes read per iteration
        // Each iteration reads k shards of shard_size bytes
        let bytes_per_iter = (K * shard_size) as u64;
        group.throughput(Throughput::Bytes(bytes_per_iter));

        // --- Scalar (encode_hot) ---
        let mut parity_scalar: Vec<Vec<u8>> = vec![vec![0u8; shard_size]; M];
        group.bench_with_input(
            BenchmarkId::new("scalar", format!("{}KB", shard_size / 1024)),
            &shard_size,
            |b, _| {
                b.iter(|| {
                    encode_hot(
                        black_box(&coeffs),
                        black_box(&data_shards),
                        black_box(&mut parity_scalar),
                        K,
                        M,
                    );
                });
            },
        );

        // --- SIMD (simd::encode, called M times for M parity shards) ---
        group.bench_with_input(
            BenchmarkId::new("simd", format!("{}KB", shard_size / 1024)),
            &shard_size,
            |b, _| {
                b.iter(|| {
                    // Produce all M parity shards
                    for p in 0..M {
                        let row_coeffs = &coeffs[p * K..(p + 1) * K];
                        let parity =
                            unsafe { simd::encode(black_box(&data_shards), black_box(row_coeffs)) };
                        black_box(parity);
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_roofline);
criterion_main!(benches);
