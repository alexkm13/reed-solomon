#[cfg(test)]
mod tests {
    use reed_solomon::codec::{split, encode, encode_hot, reconstruct, verify, ShardError};
    use reed_solomon::matrix::Matrix;

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

    // --- reconstruct tests ---

    #[test]
    fn reconstruct_no_shards_lost() {
        let k = 4;
        let m = 2;
        let data_shards: Vec<Vec<u8>> = vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
            vec![13, 14, 15, 16],
        ];
        let parity_shards = encode(&data_shards, m).unwrap();

        let mut all_shards: Vec<Option<Vec<u8>>> = Vec::new();
        for shard in &data_shards {
            all_shards.push(Some(shard.clone()));
        }
        for shard in &parity_shards {
            all_shards.push(Some(shard.clone()));
        }

        let recovered = reconstruct(&all_shards, k, m).unwrap();

        for i in 0..k {
            assert_eq!(recovered[i], data_shards[i], "Data shard {} mismatch", i);
        }
        for i in 0..m {
            assert_eq!(recovered[k + i], parity_shards[i], "Parity shard {} mismatch", i);
        }
    }

    #[test]
    fn reconstruct_one_data_shard_lost() {
        let k = 4;
        let m = 2;
        let data_shards: Vec<Vec<u8>> = vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
            vec![13, 14, 15, 16],
        ];
        let parity_shards = encode(&data_shards, m).unwrap();

        let mut all_shards: Vec<Option<Vec<u8>>> = Vec::new();
        for shard in &data_shards {
            all_shards.push(Some(shard.clone()));
        }
        for shard in &parity_shards {
            all_shards.push(Some(shard.clone()));
        }

        // Lose data shard 1
        all_shards[1] = None;

        let recovered = reconstruct(&all_shards, k, m).unwrap();

        for i in 0..k {
            assert_eq!(recovered[i], data_shards[i], "Data shard {} not recovered correctly", i);
        }
    }

    #[test]
    fn reconstruct_one_parity_shard_lost() {
        let k = 4;
        let m = 2;
        let data_shards: Vec<Vec<u8>> = vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
            vec![13, 14, 15, 16],
        ];
        let parity_shards = encode(&data_shards, m).unwrap();

        let mut all_shards: Vec<Option<Vec<u8>>> = Vec::new();
        for shard in &data_shards {
            all_shards.push(Some(shard.clone()));
        }
        for shard in &parity_shards {
            all_shards.push(Some(shard.clone()));
        }

        // Lose parity shard 0 (index k)
        all_shards[k] = None;

        let recovered = reconstruct(&all_shards, k, m).unwrap();

        for i in 0..k {
            assert_eq!(recovered[i], data_shards[i], "Data shard {} mismatch", i);
        }
        for i in 0..m {
            assert_eq!(recovered[k + i], parity_shards[i], "Parity shard {} not recovered correctly", i);
        }
    }

    #[test]
    fn reconstruct_multiple_shards_lost_up_to_m() {
        let k = 4;
        let m = 3;
        let data_shards: Vec<Vec<u8>> = vec![
            vec![10, 20, 30],
            vec![40, 50, 60],
            vec![70, 80, 90],
            vec![100, 110, 120],
        ];
        let parity_shards = encode(&data_shards, m).unwrap();

        let mut all_shards: Vec<Option<Vec<u8>>> = Vec::new();
        for shard in &data_shards {
            all_shards.push(Some(shard.clone()));
        }
        for shard in &parity_shards {
            all_shards.push(Some(shard.clone()));
        }

        // Lose m shards (maximum recoverable)
        all_shards[0] = None;  // data shard 0
        all_shards[2] = None;  // data shard 2
        all_shards[k + 1] = None;  // parity shard 1

        let recovered = reconstruct(&all_shards, k, m).unwrap();

        for i in 0..k {
            assert_eq!(recovered[i], data_shards[i], "Data shard {} not recovered correctly", i);
        }
    }

    #[test]
    fn reconstruct_too_many_shards_lost() {
        let k = 4;
        let m = 2;
        let data_shards: Vec<Vec<u8>> = vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
            vec![13, 14, 15, 16],
        ];
        let parity_shards = encode(&data_shards, m).unwrap();

        let mut all_shards: Vec<Option<Vec<u8>>> = Vec::new();
        for shard in &data_shards {
            all_shards.push(Some(shard.clone()));
        }
        for shard in &parity_shards {
            all_shards.push(Some(shard.clone()));
        }

        // Lose m+1 shards (one more than recoverable)
        all_shards[0] = None;
        all_shards[1] = None;
        all_shards[2] = None;

        let result = reconstruct(&all_shards, k, m);

        assert!(matches!(result, Err(ShardError::UnrecoverableError)));
    }

    #[test]
    fn reconstruct_end_to_end() {
        let k = 4;
        let m = 2;
        let original_data: Vec<u8> = b"Hello, Reed-Solomon! This is a test of erasure coding.".to_vec();

        // Split into data shards
        let data_shards = split(&original_data, k);

        // Encode to get parity shards
        let parity_shards = encode(&data_shards, m).unwrap();

        // Combine into all shards
        let mut all_shards: Vec<Option<Vec<u8>>> = Vec::new();
        for shard in &data_shards {
            all_shards.push(Some(shard.clone()));
        }
        for shard in &parity_shards {
            all_shards.push(Some(shard.clone()));
        }

        // Simulate losing some shards
        all_shards[1] = None;  // lose data shard 1
        all_shards[k] = None;  // lose parity shard 0

        // Reconstruct
        let recovered = reconstruct(&all_shards, k, m).unwrap();

        // Reassemble and verify
        let mut reconstructed_data: Vec<u8> = Vec::new();
        for i in 0..k {
            reconstructed_data.extend(&recovered[i]);
        }

        // Trim padding and compare
        let trimmed: Vec<u8> = reconstructed_data[..original_data.len()].to_vec();
        assert_eq!(trimmed, original_data, "End-to-end reconstruction failed");
    }

    // --- verify tests ---

    #[test]
    fn verify_identical_input() {
        let original: Vec<u8> = vec![1, 2, 3];
        let recovered: Vec<Vec<u8>> = vec![vec![1], vec![2], vec![3]];
        assert!(verify(&original, &recovered, 3));
    }

    #[test]
    fn verify_identical_with_padding() {
        let original: Vec<u8> = vec![1, 2, 3];
        let recovered: Vec<Vec<u8>> = vec![vec![1], vec![2], vec![3], vec![0]];
        // k=3 means only first 3 shards are concatenated
        assert!(verify(&original, &recovered, 3));
    }

    #[test]
    fn verify_different_content() {
        let original: Vec<u8> = vec![1, 2, 3];
        let recovered: Vec<Vec<u8>> = vec![vec![1], vec![2], vec![4]];
        assert!(!verify(&original, &recovered, 3));
    }

    #[test]
    fn verify_different_length_original_longer() {
        let original: Vec<u8> = vec![1, 2, 3, 4, 5];
        let recovered: Vec<Vec<u8>> = vec![vec![1], vec![2], vec![3]];
        // recovered_concat = 3 bytes, original = 5 bytes
        // Returns false because recovered can't hold all of original
        assert!(!verify(&original, &recovered, 3));
    }

    #[test]
    fn verify_end_to_end_happy_path() {
        let k = 4;
        let m = 2;
        let original_data: Vec<u8> = b"Hello, Reed-Solomon!".to_vec();

        let data_shards = split(&original_data, k);
        let parity_shards = encode(&data_shards, m).unwrap();

        let mut all_shards: Vec<Option<Vec<u8>>> = Vec::new();
        for shard in &data_shards {
            all_shards.push(Some(shard.clone()));
        }
        for shard in &parity_shards {
            all_shards.push(Some(shard.clone()));
        }

        // Reconstruct with no losses
        let recovered = reconstruct(&all_shards, k, m).unwrap();

        assert!(verify(&original_data, &recovered, k));
    }

    #[test]
    fn verify_end_to_end_with_losses() {
        let k = 4;
        let m = 2;
        let original_data: Vec<u8> = b"Testing erasure coding recovery!".to_vec();

        let data_shards = split(&original_data, k);
        let parity_shards = encode(&data_shards, m).unwrap();

        let mut all_shards: Vec<Option<Vec<u8>>> = Vec::new();
        for shard in &data_shards {
            all_shards.push(Some(shard.clone()));
        }
        for shard in &parity_shards {
            all_shards.push(Some(shard.clone()));
        }

        // Lose some shards
        all_shards[0] = None;
        all_shards[k + 1] = None;

        let recovered = reconstruct(&all_shards, k, m).unwrap();

        assert!(verify(&original_data, &recovered, k));
    }

    #[test]
    fn verify_end_to_end_with_corruption() {
        let k = 4;
        let m = 2;
        let original_data: Vec<u8> = b"Corruption detection test!".to_vec();

        let data_shards = split(&original_data, k);
        let parity_shards = encode(&data_shards, m).unwrap();

        let mut all_shards: Vec<Option<Vec<u8>>> = Vec::new();
        for shard in &data_shards {
            all_shards.push(Some(shard.clone()));
        }
        for shard in &parity_shards {
            all_shards.push(Some(shard.clone()));
        }

        // Corrupt a shard (flip a bit) - this simulates undetected corruption
        if let Some(ref mut shard) = all_shards[0] {
            shard[0] ^= 0x01;
        }

        let recovered = reconstruct(&all_shards, k, m).unwrap();

        // The corrupted data won't match original
        assert!(!verify(&original_data, &recovered, k));
    }

    // --- encode_hot tests ---

    #[test]
    fn encode_hot_matches_encode() {
        let k = 4;
        let m = 2;
        let shard_len = 16;

        // Create data shards
        let data_shards: Vec<Vec<u8>> = (0..k)
            .map(|i| (0..shard_len).map(|j| ((i * shard_len + j) * 7 + 13) as u8).collect())
            .collect();

        // Get expected result from encode
        let expected = encode(&data_shards, m).unwrap();

        // Manually extract coefficients from vand_fix (bottom m rows)
        let f = Matrix::vand_fix(m, k).unwrap();
        let coeffs = &f.elements[k * k..(k + m) * k];

        // Allocate parity buffer
        let mut parity: Vec<Vec<u8>> = vec![vec![0u8; shard_len]; m];

        // Call encode_hot
        encode_hot(coeffs, &data_shards, &mut parity, k, m);

        // Verify it matches encode
        assert_eq!(parity, expected, "encode_hot should produce same result as encode");
    }

    #[test]
    fn encode_hot_matches_encode_various_sizes() {
        for (k, m, shard_len) in [(2, 1, 8), (3, 2, 10), (4, 3, 32), (5, 4, 64)] {
            // Create data shards with varied content
            let data_shards: Vec<Vec<u8>> = (0..k)
                .map(|i| (0..shard_len).map(|j| (i ^ j) as u8).collect())
                .collect();

            // Get expected from encode
            let expected = encode(&data_shards, m).unwrap();

            // Manual setup: extract bottom m rows of vand_fix
            let f = Matrix::vand_fix(m, k).unwrap();
            let coeffs = &f.elements[k * k..(k + m) * k];

            // Allocate parity buffer and call encode_hot
            let mut parity: Vec<Vec<u8>> = vec![vec![0u8; shard_len]; m];
            encode_hot(coeffs, &data_shards, &mut parity, k, m);

            assert_eq!(
                parity, expected,
                "encode_hot mismatch for k={}, m={}, shard_len={}",
                k, m, shard_len
            );
        }
    }
}
