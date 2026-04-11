// galois field (GF^8) operations

const W: u16 = 8; // w in 2^w -- hardcode to 8 to represent bytes
const LOG_ARR: [u8; 256]; // log table for quick access
const EXP_ARR: [u8; 512]; // exponential table for inverse log table
const NW_MINUS_1: u16 = 255;

fn add(a: u8, b: u8) -> u8 {
    (a ^ b)
}

fn subtract(a: u8, b: u8) -> u8 {
    (a ^ b)
}

fn mult(a: u8, b: u8) -> u8 {
    let mut sum_log: u16;
    if a == 0 || b == 0 {
        return 0;
    }

    sum_log = LOG_ARR[a as usize] as u16 + LOG_ARR[b as usize] as u16;
    if sum_log >= NW_MINUS_1 {
        sum_log -= NW_MINUS_1;
    }

    EXP_ARR[sum_log as usize] 
}

fn div(a: u8, b: u8) -> u8 {
    let mut diff_log: i16;
    
    if a == 0 {
        return 0;
    }
    
    match b {
        0 => panic!("Cannot divide by zero."),
        _ => {},
    }

    diff_log = LOG_ARR[a as usize] as i16 - LOG_ARR[b as usize] as i16;
    if diff_log < 0 {
        diff_log += (NW_MINUS_1) as i16;
    }

    EXP_ARR[diff_log as usize] 
}
