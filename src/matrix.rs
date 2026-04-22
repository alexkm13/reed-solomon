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
    pub fn multiply(&self, m: &Matrix) -> Result<Matrix, MatrixError> {
        // TODO
    }

    pub fn scale_row(scalar: u8, row: u16, m: &mut Matrix) -> () {
        for e in &mut m.elements[row * m.col..((row * m.col) as usize + m.col as usize)] {
            *e = mult(*e, scalar, &LOG_TABLE, &EXP_TABLE);
        }
    }

    pub fn add_rows(base_row: usize, target_row: usize, m: &mut Matrix) -> () {
        for i in &m.elements[&m.row * &m.col..((&m.row * &m.col) + &m.col)] {
            *m.elements[target_row][i] = add(&m.elements[base_row][i], &m.elements[target_row][i]);
        }
    }

    pub fn swap_rows(base_row: u8, target_row: u8, m: &mut Matrix) -> () {
        //TODO
    }
} 



