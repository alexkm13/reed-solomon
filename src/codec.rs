use crate::matrix::{Matrix, MatrixError};

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
