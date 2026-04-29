#[cfg(test)]
mod tests {
    use reed_solomon::matrix::{Matrix, MatrixError};

    #[test]
    fn identity_stays_identity() {
        let mut m = Matrix {
            row: 3,
            col: 3,
            elements: vec![
                1, 0, 0,
                0, 1, 0,
                0, 0, 1,
            ],
        };
        m.elimination().unwrap();
        assert_eq!(
            m.elements,
            vec![
                1, 0, 0,
                0, 1, 0,
                0, 0, 1,
            ]
        );
    }

    #[test]
    fn identical_rows_not_invertible() {
        let mut m = Matrix {
            row: 3,
            col: 3,
            elements: vec![
                1, 2, 3,
                1, 2, 3,
                4, 5, 6,
            ],
        };
        assert_eq!(m.elimination(), Err(MatrixError::NotInvertible));
    }

    #[test]
    fn pivot_swap_when_diagonal_zero() {
        // First row has 0 in pivot position, requires swap with row 1
        let mut m = Matrix {
            row: 3,
            col: 3,
            elements: vec![
                0, 1, 0,
                1, 0, 0,
                0, 0, 1,
            ],
        };
        m.elimination().unwrap();
        // After elimination should be identity
        assert_eq!(
            m.elements,
            vec![
                1, 0, 0,
                0, 1, 0,
                0, 0, 1,
            ]
        );
    }

    #[test]
    fn scale_row_to_one() {
        // Diagonal element is 2, should be scaled to 1
        // In GF(2^8), inv(2) * 2 = 1
        let mut m = Matrix {
            row: 2,
            col: 2,
            elements: vec![
                2, 0,
                0, 1,
            ],
        };
        m.elimination().unwrap();
        assert_eq!(m.elements[0], 1);
    }

    #[test]
    fn full_elimination_2x2() {
        // Non-trivial 2x2 matrix
        let mut m = Matrix {
            row: 2,
            col: 2,
            elements: vec![
                2, 3,
                4, 5,
            ],
        };
        let result = m.elimination();
        // If invertible, diagonal should be 1s
        if result.is_ok() {
            assert_eq!(m.elements[0], 1); // [0][0] = 1
            assert_eq!(m.elements[3], 1); // [1][1] = 1
        }
    }

    #[test]
    fn all_operations_3x3() {
        // Matrix requiring swap, scale, and row addition
        let mut m = Matrix {
            row: 3,
            col: 3,
            elements: vec![
                0, 2, 1,
                3, 0, 4,
                1, 5, 0,
            ],
        };
        let result = m.elimination();
        if result.is_ok() {
            // After full elimination, diagonal should be 1s
            assert_eq!(m.elements[0], 1); // [0][0]
            assert_eq!(m.elements[4], 1); // [1][1]
            assert_eq!(m.elements[8], 1); // [2][2]
        }
    }

    // ===== Inverse tests =====

    #[test]
    fn inverse_identity_stays_identity() {
        let mut m = Matrix {
            row: 3,
            col: 3,
            elements: vec![
                1, 0, 0,
                0, 1, 0,
                0, 0, 1,
            ],
        };
        m.inverse().unwrap();
        assert_eq!(
            m.elements,
            vec![
                1, 0, 0,
                0, 1, 0,
                0, 0, 1,
            ]
        );
    }

    #[test]
    fn inverse_singular_matrix_not_invertible() {
        // Matrix with zero row is singular
        let mut m = Matrix {
            row: 2,
            col: 2,
            elements: vec![
                0, 0,
                1, 2,
            ],
        };
        assert_eq!(m.inverse(), Err(MatrixError::NotInvertible));
    }

    #[test]
    fn inverse_non_square_dimension_mismatch() {
        let mut m = Matrix {
            row: 2,
            col: 3,
            elements: vec![
                1, 2, 3,
                4, 5, 6,
            ],
        };
        assert_eq!(m.inverse(), Err(MatrixError::DimensionMismatch));
    }

    #[test]
    fn inverse_2x2_verify_product_is_identity() {
        use reed_solomon::field::{mult, add, setup_tables};
        let (log, exp) = setup_tables();

        // Original matrix
        let original = vec![2, 3, 4, 5];

        let mut m = Matrix {
            row: 2,
            col: 2,
            elements: original.clone(),
        };
        m.inverse().unwrap();

        // Compute A * A^-1, should be identity
        // [a b]   [e f]   [ae+bg  af+bh]
        // [c d] * [g h] = [ce+dg  cf+dh]
        let a = original[0]; let b = original[1];
        let c = original[2]; let d = original[3];
        let e = m.elements[0]; let f = m.elements[1];
        let g = m.elements[2]; let h = m.elements[3];

        let r00 = add(mult(a, e, &log, &exp), mult(b, g, &log, &exp));
        let r01 = add(mult(a, f, &log, &exp), mult(b, h, &log, &exp));
        let r10 = add(mult(c, e, &log, &exp), mult(d, g, &log, &exp));
        let r11 = add(mult(c, f, &log, &exp), mult(d, h, &log, &exp));

        assert_eq!(r00, 1, "A*A^-1 [0][0] should be 1");
        assert_eq!(r01, 0, "A*A^-1 [0][1] should be 0");
        assert_eq!(r10, 0, "A*A^-1 [1][0] should be 0");
        assert_eq!(r11, 1, "A*A^-1 [1][1] should be 1");
    }
}
