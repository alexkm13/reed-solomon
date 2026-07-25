use std::fs;
use std::process::ExitCode;
use reed_solomon::matrix::Matrix;
use reed_solomon::simd::encode;
use reed_solomon::codec::reconstruct_hot;

fn usage() {
    eprintln!("Usage:");
    eprintln!("  reed-solomon encode <file> <k> <m>");
    eprintln!("  reed-solomon reconstruct <k> <m> <output>");
    eprintln!();
    eprintln!("  k = number of data shards");
    eprintln!("  m = number of parity shards");
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        usage();
        return ExitCode::FAILURE;
    }

    let command = &args[1];

    match command.as_str() {
        "encode" => {
            if args.len() < 5 {
                eprintln!("Error: encode requires <file> <k> <m>");
                usage();
                return ExitCode::FAILURE;
            }

            let file = &args[2];
            let k: usize = match args[3].parse() {
                Ok(v) if v > 0 => v,
                Ok(_) => {
                    eprintln!("Error: k must be a positive integer");
                    return ExitCode::FAILURE;
                }
                Err(_) => {
                    eprintln!("Error: invalid k value '{}'", args[3]);
                    return ExitCode::FAILURE;
                }
            };
            let m: usize = match args[4].parse() {
                Ok(v) if v > 0 => v,
                Ok(_) => {
                    eprintln!("Error: m must be a positive integer");
                    return ExitCode::FAILURE;
                }
                Err(_) => {
                    eprintln!("Error: invalid m value '{}'", args[4]);
                    return ExitCode::FAILURE;
                }
            };

            let file_bytes = match fs::read(file) {
                Ok(bytes) => bytes,
                Err(e) => {
                    eprintln!("Error: failed to read '{}': {}", file, e);
                    return ExitCode::FAILURE;
                }
            };

            if file_bytes.is_empty() {
                eprintln!("Error: file '{}' is empty", file);
                return ExitCode::FAILURE;
            }

            let shard_len = (file_bytes.len() + k - 1) / k; // Round up
            let mut padded = file_bytes.clone();
            while padded.len() < shard_len * k {
                padded.push(0);
            }

            let data_shards: Vec<Vec<u8>> = padded
                .chunks(shard_len)
                .map(|s| s.to_vec())
                .collect();

            let f = match Matrix::vand_fix(m, k) {
                Ok(matrix) => matrix,
                Err(e) => {
                    eprintln!("Error: failed to create encoding matrix: {:?}", e);
                    return ExitCode::FAILURE;
                }
            };

            let coeffs = f.elements[k * k..(k + m) * k].to_vec();
            let mut parity_shards: Vec<Vec<u8>> = Vec::new();

            for i in 0..m {
                let raw_coeffs = &coeffs[i * k..(i + 1) * k];
                let mut parity = vec![0u8; shard_len];
                unsafe {
                    encode(&data_shards, raw_coeffs, &mut parity);
                }
                parity_shards.push(parity);
            }

            for (i, shard) in data_shards.iter().chain(parity_shards.iter()).enumerate() {
                if let Err(e) = fs::write(format!("shard_{i}"), shard) {
                    eprintln!("Error: failed to write shard_{}: {}", i, e);
                    return ExitCode::FAILURE;
                }
            }

            // Store original file length for reconstruction
            if let Err(e) = fs::write("shard_meta", file_bytes.len().to_string()) {
                eprintln!("Error: failed to write shard_meta: {}", e);
                return ExitCode::FAILURE;
            }

            println!("Encoded '{}' into {} data + {} parity shards", file, k, m);
        }

        "reconstruct" => {
            if args.len() < 5 {
                eprintln!("Error: reconstruct requires <k> <m> <output>");
                usage();
                return ExitCode::FAILURE;
            }

            let k: usize = match args[2].parse() {
                Ok(v) if v > 0 => v,
                Ok(_) => {
                    eprintln!("Error: k must be a positive integer");
                    return ExitCode::FAILURE;
                }
                Err(_) => {
                    eprintln!("Error: invalid k value '{}'", args[2]);
                    return ExitCode::FAILURE;
                }
            };
            let m: usize = match args[3].parse() {
                Ok(v) if v > 0 => v,
                Ok(_) => {
                    eprintln!("Error: m must be a positive integer");
                    return ExitCode::FAILURE;
                }
                Err(_) => {
                    eprintln!("Error: invalid m value '{}'", args[3]);
                    return ExitCode::FAILURE;
                }
            };
            let output = &args[4];

            let mut shards: Vec<Option<Vec<u8>>> = Vec::new();
            let mut present = 0;
            for i in 0..(k + m) {
                match fs::read(format!("shard_{i}")) {
                    Ok(data) => {
                        shards.push(Some(data));
                        present += 1;
                    }
                    Err(_) => shards.push(None),
                }
            }

            if present < k {
                eprintln!(
                    "Error: not enough shards to reconstruct (have {}, need {})",
                    present, k
                );
                return ExitCode::FAILURE;
            }

            // Read original file length from metadata
            let original_len: usize = match fs::read_to_string("shard_meta") {
                Ok(s) => match s.trim().parse() {
                    Ok(len) => len,
                    Err(_) => {
                        eprintln!("Error: invalid shard_meta format");
                        return ExitCode::FAILURE;
                    }
                },
                Err(e) => {
                    eprintln!("Error: failed to read shard_meta: {}", e);
                    return ExitCode::FAILURE;
                }
            };

            let recovered = match unsafe { reconstruct_hot(&shards, k, m) } {
                Ok(data) => data,
                Err(e) => {
                    eprintln!("Error: reconstruction failed: {:?}", e);
                    return ExitCode::FAILURE;
                }
            };

            let output_data: Vec<u8> = recovered.into_iter().take(k).flatten().take(original_len).collect();

            if let Err(e) = fs::write(output, &output_data) {
                eprintln!("Error: failed to write '{}': {}", output, e);
                return ExitCode::FAILURE;
            }

            println!(
                "Reconstructed from {}/{} shards to '{}'",
                present,
                k + m,
                output
            );
        }

        "help" | "--help" | "-h" => {
            usage();
        }

        _ => {
            eprintln!("Error: unknown command '{}'", command);
            usage();
            return ExitCode::FAILURE;
        }
    }

    ExitCode::SUCCESS
}
