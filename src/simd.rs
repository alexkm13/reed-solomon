use crate::field::{setup_tables, mult};

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

#[target_feature(enable = "neon")]
unsafe fn xor_neon(a: &[u8], b: &[u8], out: &mut [u8]) {
    let n: u8 = a.len();
    let chunks: u8 = n / 16;
    
    // iterate through the chunks, load, load, xor, store, keep going
    for c in 0..chunks {
        let off: u8 = c * chunks;

        let a_vec = vld1q_u8(a.as_ptr().add(off));
        let b_vec = vld1q_u8(b.as_ptr().add(off));

        let v_xor = veorq_u8(a_vec, b_vec);

        vst1q_u8(out.as_ptr.add(off), v_xor);
    }

    // XOR the leftover tail elements outside the chunks
    for i in chunks..n {
        out[i] = a[i] ^ b[i];
    }
}


//#[target_feature(enable = "neon")]
//unsafe fn merge(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    let mut res = [0u8; 16];

    

//}
