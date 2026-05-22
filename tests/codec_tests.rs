#[cfg(test)]
mod tests {
    use reed_solomon::codec::{split, encode};

    #[test]
    fn evenly_divisible() {
        let data: Vec<u8> = (0..16).collect();
        let shards = split(&data, 4);

        assert_eq!(shards.len(), 4);
        for shard in &shards {
            assert_eq!(shard.len(), 4);
        }
        let reconstructed: Vec<u8> = shards.into_iter().flatten().collect();
        assert_eq!(reconstructed, data);
    }

    #[test]
    fn not_divisible_needs_padding() {
        let data: Vec<u8> = (0..10).collect();
        let shards = split(&data, 4);

        assert_eq!(shards.len(), 4);
        // Each shard should be ceil(10/4) = 3 bytes
        for shard in &shards {
            assert_eq!(shard.len(), 3);
        }
        // First 10 bytes should match input, remaining 2 bytes are padding zeros
        let reconstructed: Vec<u8> = shards.into_iter().flatten().collect();
        assert_eq!(&reconstructed[..10], &data[..]);
        assert_eq!(&reconstructed[10..], &[0, 0]);
    }

    #[test]
    fn empty_input() {
        let data: Vec<u8> = vec![];
        let shards = split(&data, 4);

        assert_eq!(shards.len(), 4);
        // Empty input with k=4 results in 4 empty shards
        for shard in &shards {
            assert!(shard.is_empty());
        }
    }

    #[test]
    fn single_byte_input() {
        let data: Vec<u8> = vec![42];
        let shards = split(&data, 4);

        assert_eq!(shards.len(), 4);
        // First shard contains the byte
        assert_eq!(shards[0][0], 42);
    }

    #[test]
    fn equal_shard_sizes() {
        // Test various input sizes to ensure all shards have equal length
        for data_len in [1, 5, 10, 16, 17, 100] {
            for k in [2, 3, 4, 5] {
                let data: Vec<u8> = (0..data_len as u8).collect();
                let shards = split(&data, k);

                let first_len = shards[0].len();
                for shard in &shards {
                    assert_eq!(
                        shard.len(),
                        first_len,
                        "Shard length mismatch for data_len={}, k={}",
                        data_len,
                        k
                    );
                }
            }
        }
    }

    // --- encode tests ---

    #[test]
    fn encode_output_dimensions() {
        let k = 4;
        let m = 2;
        let shard_len = 8;
        let data_shards: Vec<Vec<u8>> = (0..k)
            .map(|i| (0..shard_len).map(|j| (i * shard_len + j) as u8).collect())
            .collect();

        let parity = encode(&data_shards, m).unwrap();

        assert_eq!(parity.len(), m, "Should produce m parity shards");
        for (i, shard) in parity.iter().enumerate() {
            assert_eq!(
                shard.len(),
                shard_len,
                "Parity shard {} should have same length as data shards",
                i
            );
        }
    }

    #[test]
    fn encode_deterministic() {
        let m = 2;
        let data_shards: Vec<Vec<u8>> = vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
            vec![13, 14, 15, 16],
        ];

        let parity1 = encode(&data_shards, m).unwrap();
        let parity2 = encode(&data_shards, m).unwrap();

        assert_eq!(parity1, parity2, "Encoding same data twice should produce identical parity");
    }

    #[test]
    fn encode_single_byte_shards() {
        let m = 2;
        let data_shards: Vec<Vec<u8>> = vec![vec![10], vec![20], vec![30]];

        let parity = encode(&data_shards, m).unwrap();

        assert_eq!(parity.len(), m);
        for shard in &parity {
            assert_eq!(shard.len(), 1, "Parity shards should be single byte");
        }
    }

    #[test]
    fn encode_zero_data() {
        let k = 4;
        let m = 2;
        let shard_len = 5;
        let data_shards: Vec<Vec<u8>> = vec![vec![0u8; shard_len]; k];

        let parity = encode(&data_shards, m).unwrap();

        for (i, shard) in parity.iter().enumerate() {
            for (j, &byte) in shard.iter().enumerate() {
                assert_eq!(
                    byte, 0,
                    "Parity shard {} byte {} should be zero (F * 0 = 0)",
                    i, j
                );
            }
        }
    }

    #[test]
    fn encode_known_small_case() {
        // k=2, m=1: minimal case
        // The fixed Vandermonde matrix F for k=2, m=1 has rows:
        //   Row 0: [1, 0] (identity)
        //   Row 1: [0, 1] (identity)
        //   Row 2: parity row from vand_fix
        // We only use the parity portion (row 2) for encoding.
        //
        // For a 2x2 Vandermonde with generator [0,1,2]:
        //   V = [[1,1], [1,2]] for the top k=2 rows
        // After V * V^-1, the parity row gives specific coefficients.
        //
        // With data shards d0=[a], d1=[b], parity p0 = f[0]*a + f[1]*b in GF(2^8).
        // We verify consistency: encode then check structure.

        let m = 1;
        let data_shards: Vec<Vec<u8>> = vec![vec![1], vec![1]];

        let parity = encode(&data_shards, m).unwrap();

        assert_eq!(parity.len(), 1);
        assert_eq!(parity[0].len(), 1);

        // Re-encode with different data to verify it changes
        let data_shards2: Vec<Vec<u8>> = vec![vec![1], vec![2]];
        let parity2 = encode(&data_shards2, m).unwrap();

        assert_ne!(
            parity[0][0], parity2[0][0],
            "Different input data should produce different parity"
        );
    }
}
