#[cfg(test)]
mod tests {
    use reed_solomon::simd::build_tables;
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
}
