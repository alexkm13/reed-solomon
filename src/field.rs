// galois field (GF^8) operations
const w: u16 = 8; // w in 2^w -- hardcode to 8 to represent bytes
const log_arr: [u8; 256] = [0u8; 256]; // log table for quick access
const exp_arr: [u8; 512] = [0u8; 512]; // exponential table for inverse log table
const nw_minus_1: u16 = 255;
// adds/subtracts (XOR) between two elements in a field
pub fn add(a: u8, b: u8) -> u8 {
    a ^ b
}

// multiplies between two elements in a field
pub fn mult(a: u8, b: u8) -> u8 {
    let mut sum_log: u16;
    if a == 0 || b == 0 {
        return 0;
    }

    sum_log = log_arr[a as usize] as u16 + log_arr[b as usize] as u16;
    if sum_log >= nw_minus_1 {
        sum_log -= nw_minus_1;
    }

    exp_arr[sum_log as usize] 
}

// divides between two elements in a field
pub fn div(a: u8, b: u8) -> u8 {
    let mut diff_log: i16;
    
    if a == 0 {
        return 0;
    }
    
    match b {
        0 => panic!("Cannot divide by zero."),
        _ => {},
    }

    diff_log = log_arr[a as usize] as i16 - log_arr[b as usize] as i16;
    if diff_log < 0 {
        diff_log += (nw_minus_1) as i16;
    }

    exp_arr[diff_log as usize] 
}

// sets up the log and exponential tables 
pub fn setup_tables() -> ([u8; 256], [u8; 512]) {
    let mut b: u16 = 1;
    let mut exp = [0u8; 512];
    let mut log = [0u8; 256];
    for i in 0..255 {
        exp[i] = b as u8;
        log[b as usize] = i as u8;
        b <<= 1; // left shift 1 byte (multiples by 2)
        if b & 0x100 != 0 { 
            b ^= 0x11D;
        }
    }

    for i in 255..512 {
        exp[i] = exp[i - 255];
    }
    (log, exp)
}
