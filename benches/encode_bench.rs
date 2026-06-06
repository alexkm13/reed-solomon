use criterion::{black_box, criterion_group, criterion_main, Criterion};
use reed_solomon::codec::{encode, encode_hot, encode_hot_unsafe};
use reed_solomon::matrix::Matrix;

fn make_data_shards(k: usize, shard_len: usize) -> Vec<Vec<u8>> {
    (0..k)
        .map(|i| (0..shard_len).map(|j| ((i * 17 + j * 31) % 256) as u8).collect())
        .collect()
}

fn bench_encode(c: &mut Criterion) {
    let k = 8;
    let m = 2;
    let shard_len = 1 << 20; // 1 MiB shards

    let data_shards = make_data_shards(k, shard_len);
    let f = Matrix::vand_fix(m, k).unwrap();
    let coeffs: Vec<u8> = f.elements[k * k..(k + m) * k].to_vec();

    c.bench_function("encode", |b| {
        b.iter(|| {
            let result = encode(black_box(&data_shards), m).unwrap();
            black_box(result);
        });
    });

    c.bench_function("encode_safe", |b| {
        let mut parity: Vec<Vec<u8>> = vec![vec![0u8; shard_len]; m];
        b.iter(|| {
            encode_hot(black_box(&coeffs), black_box(&data_shards), black_box(&mut parity), k, m);
            black_box(&parity);
        });
    });

    c.bench_function("encode_ptr", |b| {
        let mut parity: Vec<Vec<u8>> = vec![vec![0u8; shard_len]; m];
        b.iter(|| {
            encode_hot_unsafe(black_box(&coeffs), black_box(&data_shards), black_box(&mut parity), k, m);
            black_box(&parity);
        });
    });
}

criterion_group!(benches, bench_encode);
criterion_main!(benches);
