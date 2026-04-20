struct Matrix {
    row: u8,
    col: u8,
    elements: Vec<u8>,
};
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixError {
    DimensionMismatch,
    NotInvertible,
}

impl Matrix {
    pub fn multiply(&self, m: &Matrix) -> Option<Matrix,   {
        // TODO
    }
} 



