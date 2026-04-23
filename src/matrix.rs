use crate::field::{mult, add, setup_tables};

const SETUP: ([u8; 256], [u8; 512]) = setup_tables();
const LOG_TABLE: [u8; 256] = SETUP.0;
const EXP_TABLE: [u8; 512] = SETUP.1;


struct Matrix {
    row: usize,
    col: usize,
    elements: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MatrixError {
    DimensionMismatch,
    NotInvertible,
}

impl Matrix {
    pub fn elimination(&self, m: &Matrix) -> Result<Matrix, MatrixError> {
        // TODO
    }

    pub fn scale_row(scalar: u8, row: usize, m: &mut Matrix) -> () {
        for e in &mut m.elements[row * m.col..(row * m.col + m.col)] {
            *e = mult(*e, scalar, &LOG_TABLE, &EXP_TABLE);
        }
    }

    pub fn add_rows(base_row: usize, target_row: usize, m: &mut Matrix) -> () {
        for i in 0..m.col {
            let val = add(m.elements[base_row * m.col + i], m.elements[target_row * m.col + i]);
            m.elements[(target_row * m.col) + i] = val;
        }
    }

    pub fn swap_rows(base_row: usize, target_row: usize, m: &mut Matrix) -> () {
        for i in 0..m.col {
            m.elements.swap((base_row * m.col) + i, (target_row * m.col) + i);
        }
    }
} 



