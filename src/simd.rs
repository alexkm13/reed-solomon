use crate::field::{mult, setup_tables};
use std::thread;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

const SETUP: ([u8; 256], [u8; 512]) = setup_tables();
const LOG_TABLE: [u8; 256] = SETUP.0;
const EXP_TABLE: [u8; 512] = SETUP.1;

pub fn build_tables(c: u8) -> ([u8; 16], [u8; 16]) {
    let mut mul_hi: [u8; 16] = [0u8; 16];
    let mut mul_lo: [u8; 16] = [0u8; 16];

    // create tables for nibbles to access
    for i in 0..16u8 {
        mul_hi[i as usize] = mult(c, i << 4, &LOG_TABLE, &EXP_TABLE);
        mul_lo[i as usize] = mult(c, i, &LOG_TABLE, &EXP_TABLE);
    }

    (mul_hi, mul_lo)
}

pub fn xor_scalar(a: &[u8], b: &[u8], out: &mut [u8]) {
    for i in 0..a.len() {
        out[i] = a[i] ^ b[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn xor_neon(a: &[u8], b: &[u8], out: &mut [u8]) {
    unsafe {
        let n = a.len();
        let chunks = n / 16;

        // iterate through the chunks, load, load, xor, store, keep going
        for c in 0..chunks {
            let off = c * 16;

            let a_vec = vld1q_u8(a.as_ptr().add(off));
            let b_vec = vld1q_u8(b.as_ptr().add(off));

            let v_xor = veorq_u8(a_vec, b_vec);

            vst1q_u8(out.as_mut_ptr().add(off), v_xor);
        }

        // XOR the leftover tail elements outside the chunks
        for i in (chunks * 16)..n {
            out[i] = a[i] ^ b[i];
        }
    }
}

pub fn create_tables(c: u8) -> ([u8; 16], [u8; 16]) {
    let mut t_lo: [u8; 16] = [0u8; 16];
    let mut t_hi: [u8; 16] = [0u8; 16];

    for n in 0u8..16 {
        t_lo[n as usize] = mult(c, n, &LOG_TABLE, &EXP_TABLE);
        t_hi[n as usize] = mult(c, n << 4, &LOG_TABLE, &EXP_TABLE);
    }

    (t_lo, t_hi)
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn mult_into(data: &[u8], c: u8, out: &mut [u8]) {
    unsafe {
        // create tables for low/high nibbles
        let (t_lo_arr, t_hi_arr) = create_tables(c);
        let t_lo = vld1q_u8(t_lo_arr.as_ptr());
        let t_hi = vld1q_u8(t_hi_arr.as_ptr());

        // create variables for byte count and chunks
        let n = data.len();
        let chunks = n / 16;

        // variable for masking
        let mask = vdupq_n_u8(0x0F);

        // iterate through chunks and operate on each chunk
        for i in 0..chunks {
            let off = i * 16;
            let v = vld1q_u8(data.as_ptr().add(off));
            // create our low and high nibbles
            let lo = vandq_u8(v, mask);
            let hi = vshrq_n_u8::<4>(v);
            // access table values at chunk location
            let value_lo = vqtbl1q_u8(t_lo, lo);
            let value_hi = vqtbl1q_u8(t_hi, hi);
            // xor these values (multiply)
            let value_xor = veorq_u8(value_lo, value_hi);
            // store into out register
            vst1q_u8(out.as_mut_ptr().add(off), value_xor);
        }

        // tail loop, scalar multiplication
        for t in (chunks * 16)..n {
            out[t] = mult(c, data[t], &LOG_TABLE, &EXP_TABLE);
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn encode(data_shards: &[Vec<u8>], coeffs: &[u8], parity: &mut [u8]) {
    if data_shards.is_empty() {
        return;
    }

    let n: usize = data_shards[0].len();
    let chunks = n / 16;

    unsafe {
        let mask = vdupq_n_u8(0x0F);

        let mut tables = Vec::new();
        for (idx, _) in data_shards.iter().enumerate() {
            let (t_lo_arr, t_hi_arr) = create_tables(coeffs[idx]);
            let t_lo = vld1q_u8(t_lo_arr.as_ptr());
            let t_hi = vld1q_u8(t_hi_arr.as_ptr());
            tables.push((t_lo, t_hi));
        }

        for c in 0..chunks {
            let off = c * 16;
            let mut parity_acc = vdupq_n_u8(0);
            for (s_idx, s) in data_shards.iter().enumerate() {
                let v = vld1q_u8(s.as_ptr().add(off));
                let (t_lo, t_hi) = tables[s_idx];
                let lo = vandq_u8(v, mask);
                let hi = vshrq_n_u8::<4>(v);
                let value_lo = vqtbl1q_u8(t_lo, lo);
                let value_hi = vqtbl1q_u8(t_hi, hi);
                let value_xor = veorq_u8(value_lo, value_hi);
                parity_acc = veorq_u8(parity_acc, value_xor);
            }
            vst1q_u8(parity.as_mut_ptr().add(off), parity_acc);
        }
    }

    for i in chunks * 16..n {
        parity[i] = 0;
        for j in 0..data_shards.len() {
            parity[i] ^= mult(data_shards[j][i], coeffs[j], &LOG_TABLE, &EXP_TABLE);
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn encode_threaded(data_shards: &[Vec<u8>], coeffs: &[u8], parity: &mut [u8], num_threads: usize) {
    if data_shards.is_empty() {
        return;
    }

    let n: usize = data_shards[0].len();
    let chunk_byte = ((n / num_threads) + 127) / 128 * 128;

    unsafe {
        let mask = vdupq_n_u8(0x0F);

        let mut tables = Vec::new();
        for (idx, _) in data_shards.iter().enumerate() {
            let (t_lo_arr, t_hi_arr) = create_tables(coeffs[idx]);
            let t_lo = vld1q_u8(t_lo_arr.as_ptr());
            let t_hi = vld1q_u8(t_hi_arr.as_ptr());
            tables.push((t_lo, t_hi));
        }
        
        let tables = &tables;
        thread::scope(|s| {
            for (t, parity_slice) in parity.chunks_mut(chunk_byte).enumerate() {
                let global_base = t * chunk_byte;
                s.spawn(move || {
                    for c in 0..(parity_slice.len() / 16) {
                        let mut parity_acc = vdupq_n_u8(0);
                        let local_off = c * 16;
                        let global_off = global_base + local_off;

                        for (s_idx, shard) in data_shards.iter().enumerate() {
                            let v = vld1q_u8(shard.as_ptr().add(global_off));
                            let (t_lo, t_hi) = tables[s_idx];
                            let lo = vandq_u8(v, mask);
                            let hi = vshrq_n_u8::<4>(v);
                            let value_lo = vqtbl1q_u8(t_lo, lo);
                            let value_hi = vqtbl1q_u8(t_hi, hi);
                            let value_xor = veorq_u8(value_lo, value_hi);
                            parity_acc = veorq_u8(parity_acc, value_xor);
                        }
                        vst1q_u8(parity_slice.as_mut_ptr().add(local_off), parity_acc);
                    }
                });
            }
        });
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_tables() {
        let (t_lo, t_hi) = create_tables(0x02);
        assert_eq!(t_hi[8], 0x1D);
    }
}
