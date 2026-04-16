// galois field (GF^8) operations
const W: u16 = 8; // w in 2^w -- hardcode to 8 to represent bytes
const LOG_ARR: [u8; 256] = [0u8; 256]; // log table for quick access
const EXP_ARR: [u8; 512] = [0u8; 512]; // exponential table for inverse log table
const NW_MINUS_1: u16 = 255;
// adds/subtracts (XOR) between two elements in a field
pub fn add(a: u8, b: u8) -> u8 {
    a ^ b
}

// multiplies between two elements in a field
pub fn mult(a: u8, b: u8, log: &[u8; 256], exp: &[u8; 512]) -> u8 {
    let mut sum_log: u16;
    if a == 0 || b == 0 {
        return 0;
    }

    sum_log = log[a as usize] as u16 + log[b as usize] as u16;
    if sum_log >= NW_MINUS_1 {
        sum_log -= NW_MINUS_1;
    }

    exp[sum_log as usize] 
}

// inverse of an element a (1/a)
pub fn inv(a: u8, log: &[u8; 256], exp: &[u8; 512]) -> u8 {
    div(1, a, log, exp)
}

// divides between two elements in a field
pub fn div(a: u8, b: u8, log: &[u8; 256], exp: &[u8; 512]) -> u8 {
    let mut diff_log: i16; 
    if a == 0 {
        return 0_u8;
    }

    if b == 0 { panic!("Cannot divide by zero.") }

    diff_log = log[a as usize] as i16 - log[b as usize] as i16;
    if diff_log < 0 {
        diff_log += (NW_MINUS_1) as i16;
    }

    exp[diff_log as usize] 
}

// sets up the log and exponential tables 
pub fn setup_tables() -> ([u8; 256], [u8; 512]) {
    let mut b: u16 = 1;
    let mut exp = [0u8; 512];
    let mut log = [0u8; 256];
    for i in 0..255 {
        exp[i] = b as u8;
        log[b as usize] = i as u8;
        b <<= 1; 
        if b & 0x100 != 0 { 
            b ^= 0x11D;
        }
    }

    for i in 255..512 {
        exp[i] = exp[i - 255];
    }
    (log, exp)
}
