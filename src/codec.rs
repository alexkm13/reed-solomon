use crate::matrix::{Matrix, MatrixError};
use crate::field::{self, setup_tables};

const SETUP: ([u8; 256], [u8; 512]) = setup_tables();
const LOG_TABLE: [u8; 256] = SETUP.0;
const EXP_TABLE: [u8; 512] = SETUP.1;


#[derive(Debug, Clone, PartialEq)]
pub enum ShardError {
    UnrecoverableError,
    MatrixError(MatrixError),
}

impl From<MatrixError> for ShardError {
    fn from(e: MatrixError) -> Self {
        ShardError::MatrixError(e)
    }
}

pub fn split(data: &[u8], k: usize) -> Vec<Vec<u8>> {
    let shard_size: usize = (data.len() + k - 1) / k;
    let mut shards: Vec<Vec<u8>> = Vec::new();
    let mut new_data: Vec<u8> = data.to_vec();
    
    while new_data.len() < shard_size * k {
        new_data.push(0);
    }
    
    for i in 0..k {
        shards.push(new_data[i * shard_size..shard_size * (i + 1)].to_vec());
    }
    shards
}

pub fn encode(data_shards: &[Vec<u8>], m: usize) -> Result<Vec<Vec<u8>>, MatrixError> {
    let shard_len = data_shards[0].len();
    let mut parity_shards: Vec<Vec<u8>> = vec![vec![0u8; shard_len]; m];
    let f: Matrix = Matrix::vand_fix(m, data_shards.len())?;
    let k = data_shards.len();
    let f_trimmed = f.elements[k * k .. (k + m) * k].to_vec();
    let fixed_f: Matrix = Matrix{col: k, row: m, elements: f_trimmed};

    for byte_pos in 0..data_shards[0].len() {
       let mut d = Vec::with_capacity(k);
       for shard in data_shards {
           d.push(shard[byte_pos]);
       }; 

       let d_mat: Matrix = Matrix{row: data_shards.len(), col: 1, elements: d};
       let c: Matrix = fixed_f.multiplication(&d_mat)?;
       for i in 0..m {
           parity_shards[i][byte_pos] = c.elements[i];
       }
    } 
    Ok(parity_shards)
}

pub fn reconstruct(shards: &[Option<Vec<u8>>], k: usize, m: usize) -> Result<Vec<Vec<u8>>, ShardError> {
    let mut shard_len = 0;
    for s in shards {
        if let Some(v) = s {
            shard_len = v.len();
            break;
        }
    }
    if shard_len == 0 {
        return Err(ShardError::UnrecoverableError);
    }

    let a: Matrix = Matrix::vand_fix(m, k)?;
    let mut survivor_indices: Vec<usize> = Vec::new();
    for i in 0..k + m {
        if shards[i].is_some() {
            survivor_indices.push(i);
        }
    }

    if survivor_indices.len() < k {
        return Err(ShardError::UnrecoverableError);
    } else {
        survivor_indices.truncate(k);
    }

    let mut a_prime_elements = Vec::new();
    // loop through and read new rows for first k rows
    for r in 0..survivor_indices.len() {
        let i = survivor_indices[r];
        a_prime_elements.extend(&a.elements[i * k..(i + 1) * k]);
    }
    let a_prime: Matrix = Matrix{col: k, row: k, elements: a_prime_elements};
    let mut a_inv = a_prime.clone();
    a_inv.inverse()?;

    let mut output: Vec<Vec<u8>> = vec![vec![0u8; shard_len]; k + m];
    let mut e_prime: Vec<u8> = vec![0u8; k];
    for byte in 0..shard_len {
        for i in 0..k {
            e_prime[i] = shards[survivor_indices[i]].as_ref().unwrap()[byte];
        }
        let e: Matrix = Matrix{col: 1, row: k, elements: e_prime.clone()};
        let d: Matrix = a_inv.multiplication(&e)?;
        for i in 0..k + m {
            if let Some(shard_data) = &shards[i] {
                output[i][byte] = shard_data[byte];
            } else {
                if i < k {
                    output[i][byte] = d.elements[i];
                } else {
                    let mut sum: u8 = 0;
                    for j in 0..k {
                        sum = field::add(sum, field::mult(a.elements[i * k + j] as u8, d.elements[j] as u8, &LOG_TABLE, &EXP_TABLE));
                    }
                    output[i][byte] = sum;
                }
            }
        }
    }
    Ok(output)
}
