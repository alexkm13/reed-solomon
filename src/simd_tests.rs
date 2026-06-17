#[cfg(test)]
mod tests {
    use crate::field::{mult, setup_tables};
    use crate::simd::encode;

    const SETUP: ([u8; 256], [u8; 512]) = setup_tables();
    const LOG_TABLE: [u8; 256] = SETUP.0;
    const EXP_TABLE: [u8; 512] = SETUP.1;

    /// Scalar reference implementation for computing parity:
    /// parity[i] = fold over shards s with XOR of mult(coeffs[s], data_shards[s][i])
    fn encode_scalar(data_shards: &[Vec<u8>], coeffs: &[u8]) -> Vec<u8> {
        assert!(!data_shards.is_empty());
        assert_eq!(data_shards.len(), coeffs.len());

        let parity_len = data_shards[0].len();
        let mut parity = vec![0u8; parity_len];

        for i in 0..parity_len {
            parity[i] = data_shards
                .iter()
                .zip(coeffs.iter())
                .fold(0u8, |acc, (shard, &coeff)| {
                    acc ^ mult(coeff, shard[i], &LOG_TABLE, &EXP_TABLE)
                });
        }

        parity
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_encode_against_scalar() {
        // Create test data shards
        let shard_len = 64; // multiple of 16 for SIMD alignment
        let num_shards = 4;

        let data_shards: Vec<Vec<u8>> = (0..num_shards)
            .map(|s| (0..shard_len).map(|i| ((s * 17 + i * 13) % 256) as u8).collect())
            .collect();

        let coeffs: Vec<u8> = vec![0x01, 0x02, 0x04, 0x08];

        // Compute expected result using scalar implementation
        let expected = encode_scalar(&data_shards, &coeffs);

        // Compute result using SIMD implementation
        let actual = unsafe { encode(&data_shards, &coeffs, shard_len) };

        assert_eq!(
            actual, expected,
            "SIMD encode result differs from scalar reference"
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_encode_against_scalar_with_tail() {
        // Test with length that has a tail (not multiple of 16)
        let shard_len = 50;
        let num_shards = 3;

        let data_shards: Vec<Vec<u8>> = (0..num_shards)
            .map(|s| (0..shard_len).map(|i| ((s * 31 + i * 7) % 256) as u8).collect())
            .collect();

        let coeffs: Vec<u8> = vec![0x03, 0x05, 0x0F];

        let expected = encode_scalar(&data_shards, &coeffs);
        let actual = unsafe { encode(&data_shards, &coeffs, shard_len) };

        assert_eq!(
            actual, expected,
            "SIMD encode with tail differs from scalar reference"
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_encode_single_shard() {
        // Edge case: single shard
        let shard_len = 32;
        let data_shards: Vec<Vec<u8>> = vec![vec![0xAB; shard_len]];
        let coeffs: Vec<u8> = vec![0x02];

        let expected = encode_scalar(&data_shards, &coeffs);
        let actual = unsafe { encode(&data_shards, &coeffs, shard_len) };

        assert_eq!(
            actual, expected,
            "SIMD encode for single shard differs from scalar reference"
        );
    }
}
