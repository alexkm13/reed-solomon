use crate::field::{setup_tables, mult};

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

fn create_tables(c: u8) -> ([u8; 16], [u8; 16]) {
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
    for t in (chunks*16)..n {
        out[t] = mult(c, data[t], &LOG_TABLE, &EXP_TABLE);
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn sum_bytes(bytes: &[u8]) -> u32 {
    let n = bytes.len();
    let chunks = n / 16;
    let mut short: uint16x8_t = vdupq_n_u16(0); // for u16 lanes, quick storage
    let mut long: uint32x4_t = vdupq_n_u32(0); // to flush the u16 lanes when there's a risk of
                                               // wrapping

    for i in 0..chunks {
        let off = i * 16;
        let v = vld1q_u8(bytes.as_ptr().add(off));
        
        short = vpadalq_u8(short, v); // pairwise add the u8 lanes into u16 lanes

        // chunks overflow around 128, so we check around halfway to flush
        if i % 64 == 63 { 
            // flush into u32 lanes register
            // then, refresh u16 lanes
            long = vpadalq_u16(long, short);
            short = vdupq_n_u16(0);
        }
    }
    
    // flush the remaining data into long so all sums are in one area
    long = vpadalq_u16(long, short);
    
    // make it a scalar of u32 size
    let mut total_sum = vaddvq_u32(long);
    
    // tail loop
    for t in chunks * 16..n {
        total_sum += bytes[t] as u32;
    }

    total_sum
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn encode(data_shards: &[Vec<u8>], coeffs: &[u8], parity_len: usize) -> Vec<u8> {
// produce ONE parity shard from all data shards
// parity[byte] = XOR over all shards s of: gf_mul(coeffs[s], data_shards[s][byte])
    // create variables for parity and chunks
    let mut parity: Vec<u8> = vec![0u8; parity_len];
    let mut temp: Vec<u8> = vec![0u8; parity_len];

    for shard in 0..data_shards.len() {
        mult_into(&data_shards[shard], coeffs[shard], &mut temp);
        for i in 0..parity_len {
            parity[i] ^= temp[i]
        }
    }

    parity
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
