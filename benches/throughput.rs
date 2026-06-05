use reed_solomon::codec::{split, encode, reconstruct};
use std::time::Instant;

const DATA_SIZE: usize = 1024 * 1024; // 1 MB
const ITERATIONS: usize = 10;
const K: usize = 4;
const M: usize = 2;

fn bench_encode() -> f64 {
    let data: Vec<u8> = (0..DATA_SIZE).map(|i| i as u8).collect();
    let data_shards = split(&data, K);

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = encode(&data_shards, M).unwrap();
    }
    let elapsed = start.elapsed();

    let total_bytes = DATA_SIZE * ITERATIONS;
    let mb_per_sec = (total_bytes as f64 / 1_000_000.0) / elapsed.as_secs_f64();
    mb_per_sec
}

fn bench_reconstruct() -> f64 {
    let data: Vec<u8> = (0..DATA_SIZE).map(|i| i as u8).collect();
    let data_shards = split(&data, K);
    let parity_shards = encode(&data_shards, M).unwrap();

    let mut all_shards: Vec<Option<Vec<u8>>> = Vec::new();
    for shard in &data_shards {
        all_shards.push(Some(shard.clone()));
    }
    for shard in &parity_shards {
        all_shards.push(Some(shard.clone()));
    }

    // Lose some shards
    all_shards[0] = None;
    all_shards[K + 1] = None;

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let shards = all_shards.clone();
        let _ = reconstruct(&shards, K, M).unwrap();
    }
    let elapsed = start.elapsed();

    let total_bytes = DATA_SIZE * ITERATIONS;
    let mb_per_sec = (total_bytes as f64 / 1_000_000.0) / elapsed.as_secs_f64();
    mb_per_sec
}

fn main() {
    println!("Benchmark baseline:");
    println!("  Data size: {} MB, K={}, M={}, Iterations={}",
             DATA_SIZE / 1_000_000, K, M, ITERATIONS);
    println!();

    let encode_throughput = bench_encode();
    println!("encode throughput:      {:.2} MB/s", encode_throughput);

    let reconstruct_throughput = bench_reconstruct();
    println!("reconstruct throughput: {:.2} MB/s", reconstruct_throughput);
}
