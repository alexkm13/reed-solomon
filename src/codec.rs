use crate::matrix::{Matrix, MatrixError};
use crate::field::{mult, add, inv, setup_tables, pow};

pub fn split(data: &[u8], k: usize) -> Vec<Vec<u8>> {
    let mut shard_size: usize = (data.len() + k - 1) / k;
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
