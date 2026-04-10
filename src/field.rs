// galois field (GF^8) operations

const W: u16 = 8; // w in 2^w -- hardcode to 8 to represent bytes
const LOG_ARR: [u8; 256] // log table for quick access
const EXP_ARR: [u8; 512] // exponential table for inverse log table

fn add(a: u8, b: u8) -> u8 {
    (a ^ b)
}

fn subtract(a: u8, b: u8) -> u8 {
    (a ^ b)
}

fn mult(a: u8, b: u8) -> usize {
    let mut sum_log: u16;
    if a == 0 || b == 0 {
        0
    };

    sum_log = LOG_ARR[a as usize] + LOG_ARR[b as usize];
    if sum_log >= 2u16.pow(W) - 1 {
        sum_log -= 2u16.pow(W) - 1;
    }

    EXP_ARR[sum_log as usize]
}
