use crate::field::{self, mult, setup_tables};
use crate::matrix::{Matrix, MatrixError};
use crate::simd::create_tables;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

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
    let shard_len: usize = data_shards[0].len();
    let k: usize = data_shards.len();
    let f: Matrix = Matrix::vand_fix(m, k)?;
    let f_trimmed = f.elements[k * k..(k + m) * k].to_vec();
    let fixed_f: Matrix = Matrix {
        col: k,
        row: m,
        elements: f_trimmed,
    };
    let mut parity_shards: Vec<Vec<u8>> = vec![vec![0u8; shard_len]; m];
    for byte_pos in 0..data_shards[0].len() {
        let mut d: Vec<u8> = Vec::with_capacity(k);
        for shard in data_shards {
            d.push(shard[byte_pos]);
        }
        let d_mat: Matrix = Matrix {
            row: k,
            col: 1,
            elements: d,
        };
        let c: Matrix = fixed_f.multiplication(&d_mat)?;
        for i in 0..m {
            parity_shards[i][byte_pos] = c.elements[i];
        }
    }
    Ok(parity_shards)
}

/// Scalar encode for benchmarking - writes parity shards in place
pub fn encode_hot(
    coeffs: &[u8],
    data_shards: &[Vec<u8>],
    parity: &mut [Vec<u8>],
    k: usize,
    m: usize,
) {
    let shard_len = data_shards[0].len();

    // For each parity shard
    for p in 0..m {
        // For each byte position
        for byte_pos in 0..shard_len {
            let mut sum: u8 = 0;
            // Sum over all data shards: coeff[p,j] * data_shards[j][byte_pos]
            for j in 0..k {
                sum = field::add(
                    sum,
                    mult(coeffs[p * k + j], data_shards[j][byte_pos], &LOG_TABLE, &EXP_TABLE),
                );
            }
            parity[p][byte_pos] = sum;
        }
    }
}

pub unsafe fn reconstruct_hot(
    shards: &[Option<Vec<u8>>],
    k: usize,
    m: usize,
) -> Result<Vec<Vec<u8>>, ShardError> {
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

    let a_prime: Matrix = Matrix {
        col: k,
        row: k,
        elements: a_prime_elements,
    };
    let mut a_inv = a_prime.clone();
    a_inv.inverse()?;

    // create chunks count and n, along with parity vector
    let n: usize = shard_len;
    let chunks = n / 16;
    let mut output: Vec<Vec<u8>> = vec![vec![0u8; shard_len]; k];

    unsafe {
        // masking variable
        let mask = vdupq_n_u8(0x0F);

        // generates tables for all shards
        let mut tables = Vec::new();
        for (_idx, i) in a_inv.elements.iter().enumerate() {
            // create tables for each shard
            let (t_lo_arr, t_hi_arr) = create_tables(*i);
            let t_lo = vld1q_u8(t_lo_arr.as_ptr()); // low nibble table
            let t_hi = vld1q_u8(t_hi_arr.as_ptr()); // high nibble table
            tables.push((t_lo, t_hi)); // loads into two registers and pushes into table
        }

        for i in 0..k {
            // loop through the chunks to load into each chunk
            for c in 0..chunks {
                let off = c * 16;
                // create a accumulator register
                let mut output_acc = vdupq_n_u8(0);
                // iterate between both shards and shard index simultaneously
                // index for tables, shards to operate on
                for (j, s_idx) in survivor_indices.iter().enumerate() {
                    // load shard s
                    let v = vld1q_u8(shards[*s_idx].as_ref().unwrap().as_ptr().add(off));
                    // create tables for each shard
                    let (t_lo, t_hi) = tables[i * k + j];
                    // create low and high nibbles
                    let lo = vandq_u8(v, mask);
                    let hi = vshrq_n_u8::<4>(v);
                    // access table values at chunk location
                    let value_lo = vqtbl1q_u8(t_lo, lo);
                    let value_hi = vqtbl1q_u8(t_hi, hi);
                    // xor these values (multiply)
                    let value_xor = veorq_u8(value_lo, value_hi);
                    // sum into parity accumulator
                    output_acc = veorq_u8(output_acc, value_xor);
                }
                // load parity_accumulator into parity vector
                vst1q_u8(output[i].as_mut_ptr().add(off), output_acc);
            }
        }
    }
    for idx in chunks * 16..n {
        for t_idx in 0..k {
            for (sur, s_idx) in survivor_indices.iter().enumerate() {
                output[t_idx][idx] ^= mult(
                    shards[*s_idx].as_ref().unwrap()[idx],
                    a_inv.elements[t_idx * k + sur],
                    &LOG_TABLE,
                    &EXP_TABLE,
                );
            }
        }
    }
    Ok(output)
}

pub fn reconstruct_scalar(
    shards: &[Option<Vec<u8>>],
    k: usize,
    m: usize,
) -> Result<Vec<Vec<u8>>, ShardError> {
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

    let a_prime: Matrix = Matrix {
        col: k,
        row: k,
        elements: a_prime_elements,
    };
    let mut a_inv = a_prime.clone();
    a_inv.inverse()?;
    let mut output: Vec<Vec<u8>> = vec![vec![0u8; shard_len]; k + m];
    let mut e_prime: Vec<u8> = vec![0u8; k];
    for byte in 0..shard_len {
        for i in 0..k {
            e_prime[i] = shards[survivor_indices[i]].as_ref().unwrap()[byte];
        }
        let e: Matrix = Matrix {
            col: 1,
            row: k,
            elements: e_prime.clone(),
        };
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
                        sum = field::add(
                            sum,
                            field::mult(
                                a.elements[i * k + j] as u8,
                                d.elements[j] as u8,
                                &LOG_TABLE,
                                &EXP_TABLE,
                            ),
                        );
                    }
                    output[i][byte] = sum;
                }
            }
        }
    }
    Ok(output)
}

pub fn verify(original: &[u8], recovered: &[Vec<u8>]) -> bool {
    let mut recovered_concat: Vec<u8> = Vec::with_capacity(recovered.len() * recovered[0].len());
    for shard in recovered {
        for s in shard {
            recovered_concat.push(*s);
        }
    }
    if recovered_concat.len() < original.len() {
        return false;
    }
    for i in 0..original.len() {
        if recovered_concat[i] != original[i] {
            return false;
        }
    }
    true
}
