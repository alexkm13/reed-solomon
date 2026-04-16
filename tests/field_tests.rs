#[cfg(test)]
mod tests {
    use reed_solomon::field::*;

    #[test]
    fn test_tables_are_inverses() {
        let (log, exp) = setup_tables();
        for i in 1..=255u8 {
            let log_i = log[i as usize];
            let roundtrip = exp[log_i as usize];
            assert_eq!(roundtrip, i, "roundtrip failed for {}", i);
        }
    }

    #[test]
    fn add_is_xor() {
        assert_eq!(add(0, 0), 0);
        assert_eq!(add(5, 5), 0);
    }

    #[test]
    fn mul_identity() {
        let (log, exp) = setup_tables();
        for a in 0..=255u8 {
            assert_eq!(mult(a, 1, &log, &exp), a);
            assert_eq!(mult(1, a, &log, &exp), a);
        }
    }

    #[test]
    fn mul_by_zero() {
        let (log, exp) = setup_tables();
        for a in 0..=255u8 {
            assert_eq!(mult(a, 0, &log, &exp), 0);
            assert_eq!(mult(0, a, &log, &exp), 0);
        }
    }

    #[test]
    fn inverse_property() {
        let (log, exp) = setup_tables();
        for a in 1..=255u8 {
            assert_eq!(mult(a, inv(a, &log, &exp), &log, &exp), 1);
        }
    }
}
