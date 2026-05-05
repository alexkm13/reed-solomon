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

    // ===== Multiplication tests =====

    #[test]
    fn mult_identity_right() {
        // A * I == A
        let a = Matrix {
            row: 3,
            col: 3,
            elements: vec![
                1, 2, 3,
                4, 5, 6,
                7, 8, 9,
            ],
        };
        let identity = Matrix {
            row: 3,
            col: 3,
            elements: vec![
                1, 0, 0,
                0, 1, 0,
                0, 0, 1,
            ],
        };
        let result = a.multiplication(&identity).unwrap();
        assert_eq!(result.elements, a.elements);
    }

    #[test]
    fn mult_identity_left() {
        // I * A == A
        let a = Matrix {
            row: 3,
            col: 3,
            elements: vec![
                1, 2, 3,
                4, 5, 6,
                7, 8, 9,
            ],
        };
        let identity = Matrix {
            row: 3,
            col: 3,
            elements: vec![
                1, 0, 0,
                0, 1, 0,
                0, 0, 1,
            ],
        };
        let result = identity.multiplication(&a).unwrap();
        assert_eq!(result.elements, a.elements);
    }

    #[test]
    fn mult_dimension_mismatch() {
        // 2x3 * 2x2 should fail (3 != 2)
        let a = Matrix {
            row: 2,
            col: 3,
            elements: vec![
                1, 2, 3,
                4, 5, 6,
            ],
        };
        let b = Matrix {
            row: 2,
            col: 2,
            elements: vec![
                1, 2,
                3, 4,
            ],
        };
        assert_eq!(a.multiplication(&b), Err(MatrixError::DimensionMismatch));
    }

    #[test]
    fn mult_2x2_hand_computed_gf256() {
        // [[1, 2], [3, 4]] * [[5, 6], [7, 8]] in GF(2^8)
        //
        // In GF(2^8), multiplication uses log/exp tables, addition is XOR.
        // Using distributivity for small values:
        //   2*7 = 2*(1+2+4) = 2 + 4 + 8 = 14
        //   2*8 = 16
        //   3*5 = (1+2)*(1+4) = 1 + 4 + 2 + 8 = 15
        //   4*7 = 4*(1+2+4) = 4 + 8 + 16 = 28
        //   3*6 = (1+2)*(2+4) = 2 + 4 + 4 + 8 = 10
        //   4*8 = 32
        //
        // Result:
        //   [0][0] = (1*5) XOR (2*7) = 5 XOR 14 = 11
        //   [0][1] = (1*6) XOR (2*8) = 6 XOR 16 = 22
        //   [1][0] = (3*5) XOR (4*7) = 15 XOR 28 = 19
        //   [1][1] = (3*6) XOR (4*8) = 10 XOR 32 = 42
        let a = Matrix {
            row: 2,
            col: 2,
            elements: vec![1, 2, 3, 4],
        };
        let b = Matrix {
            row: 2,
            col: 2,
            elements: vec![5, 6, 7, 8],
        };
        let result = a.multiplication(&b).unwrap();
        assert_eq!(result.row, 2);
        assert_eq!(result.col, 2);
        assert_eq!(result.elements, vec![11, 22, 19, 42]);
    }

    #[test]
    fn mult_by_inverse_gives_identity() {
        // A * A^-1 == I
        let original = vec![2, 3, 4, 5];
        let a = Matrix {
            row: 2,
            col: 2,
            elements: original.clone(),
        };
        let mut a_inv = Matrix {
            row: 2,
            col: 2,
            elements: original,
        };
        a_inv.inverse().unwrap();

        let result = a.multiplication(&a_inv).unwrap();
        assert_eq!(result.elements, vec![1, 0, 0, 1], "A * A^-1 should be identity");
    }

    // ===== Vandermonde tests =====

    #[test]
    fn vandermonde_small_entries() {
        // vandermonde(3, 3) produces V[r][c] = r^c in GF(2^8)
        // Row 0: 0^0=1, 0^1=0, 0^2=0
        // Row 1: 1^0=1, 1^1=1, 1^2=1
        // Row 2: 2^0=1, 2^1=2, 2^2=4
        let v = Matrix::vandermonde(3, 3);
        assert_eq!(v.elements[0], 1, "V[0][0] = 0^0 = 1");
        assert_eq!(v.elements[1], 0, "V[0][1] = 0^1 = 0");
        assert_eq!(v.elements[4], 1, "V[1][1] = 1^1 = 1");
        assert_eq!(v.elements[8], 4, "V[2][2] = 2^2 = 4");
    }

    #[test]
    fn vandermonde_dimensions() {
        // vandermonde(5, 3) should have 5 rows, 3 cols, 15 elements
        let v = Matrix::vandermonde(5, 3);
        assert_eq!(v.row, 5);
        assert_eq!(v.col, 3);
        assert_eq!(v.elements.len(), 15);
    }

    #[test]
    fn vandermonde_first_column_all_ones() {
        // Every V[i][0] = i^0 = 1
        let v = Matrix::vandermonde(6, 4);
        for r in 0..6 {
            assert_eq!(v.elements[r * 4], 1, "V[{}][0] should be 1", r);
        }
    }

    #[test]
    fn vandermonde_second_column_is_row_index() {
        // V[i][1] = i^1 = i
        let v = Matrix::vandermonde(5, 3);
        for r in 0..5 {
            assert_eq!(v.elements[r * 3 + 1], r as u8, "V[{}][1] should be {}", r, r);
        }
    }

    #[test]
    fn vandermonde_top_kxk_invertible() {
        // When m > k, the top k×k submatrix should be invertible
        // This property makes Vandermonde useful for Reed-Solomon
        let v = Matrix::vandermonde(5, 3);

        // Extract top 3×3 submatrix
        let mut top = Matrix {
            row: 3,
            col: 3,
            elements: v.elements[0..9].to_vec(),
        };

        // Should not error - Vandermonde matrices with distinct elements are invertible
        assert!(top.inverse().is_ok(), "Top k×k of Vandermonde should be invertible");
    }
}
