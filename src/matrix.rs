use crate::field::{mult, add, inv, setup_tables};

const SETUP: ([u8; 256], [u8; 512]) = setup_tables();
const LOG_TABLE: [u8; 256] = SETUP.0;
const EXP_TABLE: [u8; 512] = SETUP.1;


pub struct Matrix {
    pub row: usize,
    pub col: usize,
    pub elements: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MatrixError {
    DimensionMismatch,
    NotInvertible,
}

impl Matrix {
    pub fn elimination(&mut self) -> Result<(), MatrixError> {
        let mut found: bool = false;
        for k in 0..self.col {
            if self.elements[k * self.col + k] == 0 {
                for i in (k+1)..self.row {
                    if self.elements[i * self.col + k] != 0 {
                        self.swap_rows(k, i);
                        found = true;
                        break
                    }
                }
                if !found {
                    return Err(MatrixError::NotInvertible);
                } else {
                    found = false;
                }
            }
            if self.elements[k * self.col + k] != 1 {
                self.scale_row(inv(self.elements[k * self.col + k], &LOG_TABLE, &EXP_TABLE), k);
            }
            for n in 0..self.row {
                if n == k { continue; }
                let factor = self.elements[n * self.col + k];
                if factor == 0 {continue;}
                self.add_scaled_row(factor, k, n);
            }
        }
        Ok(())
    }

    pub fn add_scaled_row(&mut self, scalar: u8, source_row: usize, target_row: usize) -> () {
       for i in 0..self.col {
           let val = mult(self.elements[source_row * self.col + i], scalar, &LOG_TABLE, &EXP_TABLE);
           self.elements[target_row * self.col + i] = add(val, self.elements[target_row * self.col + i]);
       } 
    }

    pub fn scale_row(&mut self, scalar: u8, row: usize) -> () {
        for e in &mut self.elements[row * self.col..(row * self.col + self.col)] {
            *e = mult(*e, scalar, &LOG_TABLE, &EXP_TABLE);
        }
    }

    pub fn add_rows(&mut self, base_row: usize, target_row: usize) -> () {
        for i in 0..self.col {
            let val = add(self.elements[base_row * self.col + i], self.elements[target_row * self.col + i]);
            self.elements[(target_row * self.col) + i] = val;
        }
    }

    pub fn swap_rows(&mut self, base_row: usize, target_row: usize) -> () {
        for i in 0..self.col {
            self.elements.swap((base_row * self.col) + i, (target_row * self.col) + i);
        }
    }
} 



