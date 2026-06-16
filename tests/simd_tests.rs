#[cfg(test)]
mod tests {
    use reed_solomon::simd::build_tables;
    #[cfg(target_arch = "aarch64")]
    use reed_solomon::simd::mult_into;
    #[cfg(target_arch = "aarch64")]
    use reed_solomon::simd::sum_bytes;
    use reed_solomon::field::{setup_tables, mult};

    const SETUP: ([u8; 256], [u8; 512]) = setup_tables();
    const LOG_TABLE: [u8; 256] = SETUP.0;
    const EXP_TABLE: [u8; 512] = SETUP.1;

    #[test]
    fn build_tables_basic() {
        for c in 1..=255u8 {
            let (hi, lo) = build_tables(c);
            assert_eq!(lo[0], 0);              // c * 0 = 0
            assert_eq!(lo[1], c);              // c * 1 = c
            assert_eq!(hi[1], mult(c, 16, &LOG_TABLE, &EXP_TABLE));  // c * (1<<4)
        }
    }
    #[test]
    fn decomposition_matches_scalar() {
        for c in 0..=255u8 {
            let (hi, lo) = build_tables(c);
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
                    assert_eq!(out[i], expected, "mismatch at c={:#x}, len={}, i={}", c, len, i);
                }
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_sum_bytes() {
        // small lengths: basic operation and tail handling
        // large lengths: 1000+ triggers one flush, 5000 triggers several
        for len in [0, 1, 15, 16, 17, 31, 100, 1000, 1024, 1025, 2000, 5000, 10000] {
            let bytes: Vec<u8> = (0..len).map(|i| (i % 256) as u8).collect();

            let expected: u32 = bytes.iter().map(|&b| b as u32).sum();
            let result = unsafe { sum_bytes(&bytes) };

            assert_eq!(result, expected, "mismatch at len={}", len);
        }
    }
}
