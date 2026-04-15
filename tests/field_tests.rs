#[cfg(test)]
mod tests {
    use reed_solomon::field::*;
    #[test]
    fn test_tables_are_inverses() {
        let (log_table, exp_table) = setup_tables();
        
        // log and exp should be inverses for all nonzero elements
        for i in 1..=255u8 {
            let log_i = log_table[i as usize];
            let roundtrip = exp_table[log_i as usize];
            assert_eq!(roundtrip, i, "roundtrip failed for {}", i);
        }
    }
    #[test]
    fn add_is_xor() {
        assert_eq!(add(0, 0), 0);
        assert_eq!(add(5, 5), 0); // a + a = 0 in characteristic 2
    }

    #[test]
    fn mul_identity() {
        for a in 0..=255u8 {
            assert_eq!(mult(a, 1), a);
            assert_eq!(mult(1, a), a);
        }
    }

    #[test]
    fn mul_by_zero() {
        for a in 0..=255u8 {
            assert_eq!(mult(a, 0), 0);
            assert_eq!(mult(0, a), 0);
        }
}

    #[test]
    fn inverse_property() {
        for a in 1..=255u8 {
            assert_eq!(mult(a, div(1, a)), 1);
        }
    }
}
