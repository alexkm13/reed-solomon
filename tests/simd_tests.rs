#[cfg(test)]
mod tests {
    use reed_solomon::field::{mult, setup_tables};
    use reed_solomon::simd::create_tables;
    #[cfg(target_arch = "aarch64")]
    use reed_solomon::simd::encode;
    #[cfg(target_arch = "aarch64")]
    use reed_solomon::simd::mult_into;

    const SETUP: ([u8; 256], [u8; 512]) = setup_tables();
    const LOG_TABLE: [u8; 256] = SETUP.0;
    const EXP_TABLE: [u8; 512] = SETUP.1;

    #[test]
    fn create_tables_basic() {
        for c in 1..=255u8 {
            let (lo, hi) = create_tables(c);
            assert_eq!(lo[0], 0); // c * 0 = 0
            assert_eq!(lo[1], c); // c * 1 = c
            assert_eq!(hi[1], mult(c, 16, &LOG_TABLE, &EXP_TABLE)); // c * (1<<4)
        }
    }
    #[test]
    fn decomposition_matches_scalar() {
        for c in 0..=255u8 {
            let (lo, hi) = create_tables(c);
            for b in 0..=255u8 {
                let decomposed = hi[(b >> 4) as usize] ^ lo[(b & 0x0F) as usize];
                let direct = mult(c, b, &LOG_TABLE, &EXP_TABLE);
                assert_eq!(decomposed, direct, "mismatch at c={c}, b={b}");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_mult_into() {
        for &c in &[0x02, 0x1D, 0xFF] {
            for len in [0, 1, 15, 16, 17, 31, 100] {
                // build input data
                let data: Vec<u8> = (0..len).map(|i| i as u8).collect();
                let mut out = vec![0u8; len];

                // run mult_into
                unsafe { mult_into(&data, c, &mut out) };

                // compare against scalar mult for each byte
                for i in 0..len {
                    let expected = mult(c, data[i], &LOG_TABLE, &EXP_TABLE);
                    assert_eq!(
                        out[i], expected,
                        "mismatch at c={:#x}, len={}, i={}",
                        c, len, i
                    );
                }
            }
        }
    }

    /// Scalar oracle: dead simple, obviously correct.
    /// parity[p] = XOR over all shards s of: gf_mul(coeffs[s], data_shards[s][p])
    fn encode_scalar(data_shards: &[Vec<u8>], coeffs: &[u8]) -> Vec<u8> {
        if data_shards.is_empty() {
            return vec![];
        }
        let n = data_shards[0].len();
        let mut parity = vec![0u8; n];
        for p in 0..n {
            for s in 0..data_shards.len() {
                parity[p] ^= mult(coeffs[s], data_shards[s][p], &LOG_TABLE, &EXP_TABLE);
            }
        }
        parity
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_encode_vs_scalar() {
        for num_shards in [1, 2, 3, 4, 8] {
            for len in [0, 1, 15, 16, 17, 31, 32, 100, 1000] {
                // Build deterministic data shards
                let data_shards: Vec<Vec<u8>> = (0..num_shards)
                    .map(|s| (0..len).map(|i| ((s * 17 + i * 13) & 0xFF) as u8).collect())
                    .collect();

                // Build coefficients
                let coeffs: Vec<u8> = (0..num_shards).map(|s| (s as u8).wrapping_add(1)).collect();

                let expected = encode_scalar(&data_shards, &coeffs);
                let mut got = vec![0u8; len];
                unsafe { encode(&data_shards, &coeffs, &mut got) };

                assert_eq!(expected, got, "mismatch: shards={num_shards}, len={len}");
            }
        }
    }
}
