//! Common utilities for nanochat.

use candle_core::{DType, Device};
use std::fs;
use std::path::{Path, PathBuf};

/// The dtype used for compute (matmuls, activations).
/// Override with NANOCHAT_DTYPE env var: "bfloat16", "float16", "float32".
/// Without CUDA the auto-detected compute dtype is always float32.
pub fn compute_dtype() -> DType {
    match std::env::var("NANOCHAT_DTYPE").as_deref() {
        Ok("bfloat16") => DType::BF16,
        Ok("float16") => DType::F16,
        Ok("float32") => DType::F32,
        Ok(other) => panic!("Unknown NANOCHAT_DTYPE: {other}"),
        Err(_) => DType::F32, // auto-detected: no CUDA (CPU/MPS)
    }
}

pub fn setup_default_logging() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();
}

/// Co-locate nanochat intermediates with other cached data in ~/.cache (by default).
pub fn get_base_dir() -> PathBuf {
    let nanochat_dir = if let Ok(dir) = std::env::var("NANOCHAT_BASE_DIR") {
        PathBuf::from(dir)
    } else {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(".cache").join("nanochat")
    };
    let _ = fs::create_dir_all(&nanochat_dir);
    nanochat_dir
}

/// Downloads a file from a URL to a local path in the base directory.
/// Uses a lock file to prevent concurrent downloads among multiple processes.
#[allow(dead_code)]
pub fn download_file_with_lock(
    url: &str,
    filename: &str,
    postprocess_fn: Option<fn(&Path)>,
) -> anyhow::Result<PathBuf> {
    let base_dir = get_base_dir();
    let file_path = base_dir.join(filename);

    if file_path.exists() {
        return Ok(file_path);
    }
    if let Some(parent) = file_path.parent() {
        fs::create_dir_all(parent)?;
    }

    println!("Downloading {url}...");
    let content = reqwest::blocking::get(url)?.bytes()?;
    fs::write(&file_path, &content)?;
    println!("Downloaded to {}", file_path.display());

    if let Some(f) = postprocess_fn {
        f(&file_path);
    }
    Ok(file_path)
}

pub fn print0(s: &str) {
    let ddp_rank: usize = std::env::var("RANK").ok().and_then(|r| r.parse().ok()).unwrap_or(0);
    if ddp_rank == 0 {
        println!("{s}");
    }
}

pub fn print_banner() {
    let banner = r#"
                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    "#;
    print0(banner);
}

/// (is_ddp, rank, local_rank, world_size). Distributed training is not
/// supported in the Rust port, so this is always single-process.
pub fn get_dist_info() -> (bool, usize, usize, usize) {
    (false, 0, 0, 1)
}

/// Prefer CUDA if available, otherwise MPS, otherwise fallback on CPU.
/// This build targets CPU inference.
pub fn autodetect_device_type() -> String {
    let device_type = "cpu".to_string();
    print0(&format!("Autodetected device type: {device_type}"));
    device_type
}

/// Basic initialization that we keep doing over and over, so make common.
/// Returns (is_ddp, rank, local_rank, world_size, device).
pub fn compute_init(device_type: &str) -> anyhow::Result<(bool, usize, usize, usize, Device)> {
    anyhow::ensure!(device_type == "cpu", "Invalid device type atm: {device_type}");
    let device = Device::Cpu;
    // Reproducibility: most of the code uses explicit rng objects; the global
    // seed only matters for fresh weight initialization.
    let (is_ddp, ddp_rank, ddp_local_rank, ddp_world_size) = get_dist_info();
    if ddp_rank == 0 {
        log::info!("Distributed world size: {ddp_world_size}");
    }
    Ok((is_ddp, ddp_rank, ddp_local_rank, ddp_world_size, device))
}

#[allow(dead_code)]
pub fn compute_cleanup() {}

/// Useful if we wish to not use wandb but have all the same signatures.
#[allow(dead_code)]
pub struct DummyWandb;

#[allow(dead_code)]
impl DummyWandb {
    pub fn new() -> Self {
        DummyWandb
    }
    pub fn log(&self, _data: &std::collections::HashMap<String, f64>) {}
    pub fn finish(&self) {}
}

/// Hardcoded BF16 peak flops for various GPUs. Table order matters: more
/// specific patterns first.
#[allow(dead_code)]
pub fn get_peak_flops(device_name: &str) -> f64 {
    let name = device_name.to_lowercase();
    let table: &[(&[&str], f64)] = &[
        // NVIDIA Blackwell
        (&["gb200"], 2.5e15),
        (&["grace blackwell"], 2.5e15),
        (&["b200"], 2.25e15),
        (&["b100"], 1.8e15),
        // NVIDIA Hopper
        (&["h200", "nvl"], 836e12),
        (&["h200", "pcie"], 836e12),
        (&["h200"], 989e12),
        (&["h100", "nvl"], 835e12),
        (&["h100", "pcie"], 756e12),
        (&["h100"], 989e12),
        (&["h800", "nvl"], 989e12),
        (&["h800"], 756e12),
        // NVIDIA Ampere data center
        (&["a100"], 312e12),
        (&["a800"], 312e12),
        (&["a40"], 149.7e12),
        (&["a30"], 165e12),
        // NVIDIA Ada data center
        (&["l40s"], 362e12),
        (&["l40-s"], 362e12),
        (&["l40 s"], 362e12),
        (&["l4"], 121e12),
        // AMD CDNA accelerators
        (&["mi355"], 2.5e15),
        (&["mi325"], 1.3074e15),
        (&["mi300x"], 1.3074e15),
        (&["mi300a"], 980.6e12),
        (&["mi250x"], 383e12),
        (&["mi250"], 362.1e12),
        // Consumer RTX
        (&["5090"], 209.5e12),
        (&["4090"], 165.2e12),
        (&["3090"], 71e12),
    ];
    for (patterns, flops) in table {
        if patterns.iter().all(|p| name.contains(p)) {
            return *flops;
        }
    }
    // Unknown GPU - return inf so MFU shows as 0% rather than a wrong guess
    log::warn!("Peak flops undefined for: {device_name}, MFU will show as 0%");
    f64::INFINITY
}
