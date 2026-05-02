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
        for k in 0..std::cmp::min(self.row, self.col) {
            if self.elements[k * self.col + k] == 0 {
                for i in (k+1)..self.row {
                    if self.elements[i * self.col + k] != 0 {
                        self.swap_rows(k, i);
                        found = true;
                        break;
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

    pub fn inverse(&mut self) -> Result<(), MatrixError> {
        if self.row != self.col {return Err(MatrixError::DimensionMismatch)};

        let mut aug_vec: Vec<u8> = Vec::new();
        let mut identity: Vec<u8> = Vec::new();

        for i in 0..self.row {
            aug_vec.extend(&self.elements[i * self.col .. (i + 1) * self.col]);
            for n in 0..self.col {
                if n != i {
                    identity.push(0);
                } else {
                    identity.push(1);
                }
            }
            aug_vec.extend(&identity);
            identity.clear();
        }

        let mut m = Matrix{row: self.row, col: self.col * 2, elements: aug_vec};
        m.elimination()?;
        
        for r in 0..self.row {
           self.elements[r * self.col .. (r + 1) * self.col].copy_from_slice(&m.elements[r * 2 * self.col + self.col .. (r + 1) * 2 * self.col]);        
        }

        Ok(())
    }

    pub fn multiplication(&mut self, other: Matrix) -> Result<Matrix, MatrixError> {
        if self.col != other.row {return Err(MatrixError::DimensionMismatch);};

        let mut result: Vec<u8> = Vec::new();
        let mut sum: u8 = 0;
        let mut product: u8;
        for r in 0..self.row {
            for c in 0..other.col {
                for i in 0..self.row {
                    product = mult(self.elements[r * self.col + i], other.elements[i * other.col + c], &LOG_TABLE, &EXP_TABLE); 
                    sum ^= product;
                }

                result[r * other.col + c] = sum;
            }
        }
       let new_res: Matrix = Matrix{row: self.row, col: other.col, elements: result};
       return Ok(new_res);
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



