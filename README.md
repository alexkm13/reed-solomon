# Reed-Solomon Erasure Coding Engine

A high-performance Reed-Solomon erasure coding engine in Rust. Hand-built from GF(2^8) finite field arithmetic through AVX2-vectorized kernels.

## What It Does

Splits data into `k` data shards, generates `m` parity shards, and reconstructs any missing shards from survivors, as long as at least `k` shards remain.

## Status

Under construction.

## Build

```bash
cargo build
cargo test
cargo bench
```
