use reed_solomon::matrix::Matrix;
use reed_solomon::simd;
use std::hint::black_box;
use std::time::{Duration, Instant};

const K: usize = 8;
const M: usize = 2;
const RUN_DURATION: Duration = Duration::from_secs(10);

fn main() {
    let shard_size: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16 * 1024);

    let data_shards: Vec<Vec<u8>> = (0..K)
        .map(|i| {
            (0..shard_size)
                .map(|j| ((i * 17 + j * 31) % 256) as u8)
                .collect()
        })
        .collect();

    let f = Matrix::vand_fix(M, K).unwrap();
    let coeffs: Vec<u8> = f.elements[K * K..(K + M) * K].to_vec();

    // Pre-allocate parity buffers
    let mut parity: Vec<Vec<u8>> = vec![vec![0u8; shard_size]; M];

    let start = Instant::now();
    let mut iterations: u64 = 0;

    while start.elapsed() < RUN_DURATION {
        for p in 0..M {
            let row_coeffs = &coeffs[p * K..(p + 1) * K];
            unsafe {
                simd::encode(
                    black_box(&data_shards),
                    black_box(row_coeffs),
                    black_box(&mut parity[p]),
                );
            }
        }
        iterations += 1;
    }

    black_box(&parity);

    let elapsed = start.elapsed();
    let bytes_processed = iterations * (K * shard_size) as u64;
    let throughput_gib = bytes_processed as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);

    eprintln!(
        "{} iterations in {:.2}s ({:.2} GiB/s)",
        iterations,
        elapsed.as_secs_f64(),
        throughput_gib
    );
}
