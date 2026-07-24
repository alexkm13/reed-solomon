use reed_solomon::matrix::Matrix;
use reed_solomon::simd;
use std::time::Instant;

const K: usize = 8;
const M: usize = 2;

fn main() {
    let shard_sizes = [
        1 << 10,   // 1 KB
        4 << 10,   // 4 KB
        16 << 10,  // 16 KB
        64 << 10,  // 64 KB
        256 << 10, // 256 KB
        1 << 20,   // 1 MB
    ];
    let thread_counts = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24];

    let f = Matrix::vand_fix(M, K).unwrap();
    let coeffs: Vec<u8> = f.elements[K * K..(K + M) * K].to_vec();

    // === CORRECTNESS CHECK ===
    println!("=== CORRECTNESS CHECK ===\n");

    let mut all_correct = true;
    for &shard_size in &shard_sizes {
        let data_shards: Vec<Vec<u8>> = (0..K)
            .map(|i| (0..shard_size).map(|j| ((i * 17 + j * 31) % 256) as u8).collect())
            .collect();

        // Reference: single-threaded encode
        let mut expected = vec![0u8; shard_size];
        let row_coeffs = &coeffs[0..K];
        unsafe { simd::encode(&data_shards, row_coeffs, &mut expected) };

        for &num_threads in &thread_counts {
            let mut got = vec![0u8; shard_size];
            unsafe { simd::encode_threaded(&data_shards, row_coeffs, &mut got, num_threads) };

            if expected == got {
                println!("✓ shard_size={:>7}, threads={}: MATCH", shard_size, num_threads);
            } else {
                println!("✗ shard_size={:>7}, threads={}: MISMATCH", shard_size, num_threads);
                // Find first mismatch
                for i in 0..shard_size {
                    if expected[i] != got[i] {
                        println!("  First diff at byte {}: expected={}, got={}", i, expected[i], got[i]);
                        break;
                    }
                }
                all_correct = false;
            }
        }
    }

    if !all_correct {
        println!("\n❌ CORRECTNESS FAILED - stopping here");
        return;
    }
    println!("\n✓ All correctness checks passed!\n");

    // === THROUGHPUT BENCHMARK ===
    println!("=== THROUGHPUT BENCHMARK ===\n");

    let shard_size = 4 * 1024 * 1024; // 4MB for benchmark
    let data_shards: Vec<Vec<u8>> = (0..K)
        .map(|i| (0..shard_size).map(|j| ((i * 17 + j * 31) % 256) as u8).collect())
        .collect();
    let mut parity = vec![0u8; shard_size];
    let row_coeffs = &coeffs[0..K];

    println!("Shard size: {}KB, K={}, warming up...\n", shard_size / 1024, K);

    for &num_threads in &thread_counts {
        // Warmup
        for _ in 0..100 {
            unsafe { simd::encode_threaded(&data_shards, row_coeffs, &mut parity, num_threads) };
        }

        // Benchmark
        let iterations = 500;
        let start = Instant::now();
        for _ in 0..iterations {
            unsafe { simd::encode_threaded(&data_shards, row_coeffs, &mut parity, num_threads) };
        }
        let elapsed = start.elapsed();

        let bytes_processed = iterations as u64 * (K * shard_size) as u64;
        let throughput_gib = bytes_processed as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);

        println!("threads={}: {:>6.2} GiB/s", num_threads, throughput_gib);
    }

    // Also benchmark single-threaded encode for reference
    println!();
    for _ in 0..100 {
        unsafe { simd::encode(&data_shards, row_coeffs, &mut parity) };
    }
    let iterations = 500;
    let start = Instant::now();
    for _ in 0..iterations {
        unsafe { simd::encode(&data_shards, row_coeffs, &mut parity) };
    }
    let elapsed = start.elapsed();
    let bytes_processed = iterations as u64 * (K * shard_size) as u64;
    let throughput_gib = bytes_processed as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
    println!("encode (no threading): {:>6.2} GiB/s", throughput_gib);
}
