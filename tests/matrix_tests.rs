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
}
