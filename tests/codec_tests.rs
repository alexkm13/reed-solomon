#[cfg(test)]
mod tests {
    use reed_solomon::codec::split;

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
}
