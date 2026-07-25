# Reed-Solomon

A high-performance Reed-Solomon erasure coding library in Rust with SIMD acceleration.

Splits data into `k` data shards, generates `m` parity shards, and reconstructs the original from any `k` surviving shards.

## Quick Demo

```bash
# Create a test file
echo "Hello, Reed-Solomon!" > testfile.txt

# Encode into 8 data + 2 parity shards
cargo run --release -- encode testfile.txt 8 2

# Delete 2 shards (simulating data loss)
rm shard_0 shard_3

# Reconstruct from survivors
cargo run --release -- reconstruct 8 2 recovered.txt

# Verify
diff testfile.txt recovered.txt && echo "Perfect recovery!"
```

## Performance

Benchmarked on Apple M3 Pro with 4MB shards (k=8, m=2):

| Implementation | Throughput | Speedup |
|----------------|------------|---------|
| Scalar         | 0.6 GiB/s  | 1x      |
| SIMD (NEON)    | 18 GiB/s   | 30x     |
| SIMD + 8 threads | 113 GiB/s | 188x   |

Threading scales linearly up to ~8 threads before hitting memory bandwidth limits.

## The Optimization Story

**Stage 1: Scalar baseline** — Straightforward GF(256) arithmetic using log/exp tables. Correct but slow (~0.6 GiB/s).

**Stage 2: SIMD table lookups** — Split each byte into 4-bit nibbles. Precompute `c * 0x00..0x0F` and `c * 0x00..0xF0` tables (16 bytes each). Use NEON `vqtbl1q_u8` to do 16 lookups in parallel, XOR results. Throughput jumps to ~18 GiB/s.

**Stage 3: Allocation elimination** — Changed `encode` to accept a pre-allocated buffer instead of returning a `Vec`. Removes malloc/free overhead from the hot loop. ~9% improvement.

**Stage 4: Multi-threading** — Partition output buffer across threads with `thread::scope`. Each thread processes its chunk independently. Scales to ~113 GiB/s at 8 threads before saturating memory bandwidth.

## Usage

### As a CLI

```bash
# Encode
reed-solomon encode <file> <k> <m>

# Reconstruct
reed-solomon reconstruct <k> <m> <output>
```

### As a Library

```rust
use reed_solomon::matrix::Matrix;
use reed_solomon::simd::encode;
use reed_solomon::codec::reconstruct_hot;

// Create encoding matrix
let f = Matrix::vand_fix(m, k).unwrap();
let coeffs = f.elements[k * k..(k + m) * k].to_vec();

// Encode (SIMD-accelerated)
let mut parity = vec![0u8; shard_len];
unsafe { encode(&data_shards, &coeffs[i * k..(i + 1) * k], &mut parity); }

// Reconstruct from survivors
let recovered = unsafe { reconstruct_hot(&shards, k, m).unwrap() };
```

The math uses GF(2^8) with the AES polynomial (x^8 + x^4 + x^3 + x + 1). Multiplication uses log/exp tables; the SIMD path uses nibble-decomposition for vectorized table lookups.

## Build

```bash
cargo build --release
cargo test
cargo bench
```

## License

MIT
