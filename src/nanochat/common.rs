use std::path::{Path, PathBuf};
use std::fs;

#[allow(dead_code)]
pub const SPECIAL_TOKENS: &[&str] = &[
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
];

/// Get the base directory for nanochat data
pub fn get_base_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("NANOCHAT_BASE_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache").join("nanochat")
}

/// Setup logging with sane defaults
pub fn setup_default_logging() {
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();
}

/// Autodetect the best available device
pub fn autodetect_device_type() -> String {
    "cpu".to_string() // Hardcoded as requested to skip cuda
}

/// Initialize compute environment
pub fn compute_init(device_type: &str) -> anyhow::Result<(bool, usize, usize, usize, candle_core::Device)> {
    let device = if device_type == "cpu" {
        candle_core::Device::Cpu
    } else {
        anyhow::bail!("Unsupported device type: {}", device_type);
    };
    Ok((false, 0, 1, 0, device))
}

#[allow(dead_code)]
pub fn print_banner() {
    println!("NANOCHAT RUST PORT");
}

#[allow(dead_code)]
pub fn download_file_with_lock(url: &str, filename: &str, _postprocess_fn: Option<fn(&Path)>) -> anyhow::Result<PathBuf> {
    let base_dir = get_base_dir();
    let dest = base_dir.join(filename);
    if dest.exists() {
        return Ok(dest);
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    log::info!("Downloading {}...", url);
    let bytes = reqwest::blocking::get(url)?.bytes()?;
    fs::write(&dest, &bytes)?;
    Ok(dest)
}

#[allow(dead_code)]
pub fn is_ddp_requested() -> bool { false }
#[allow(dead_code)]
pub fn is_ddp_initialized() -> bool { false }
#[allow(dead_code)]
pub fn get_dist_info() -> (bool, usize, usize, usize) { (false, 0, 1, 0) }
#[allow(dead_code)]
pub fn compute_cleanup() {}

#[allow(dead_code)]
pub struct DummyWandb;
#[allow(dead_code)]
impl DummyWandb {
    pub fn new() -> Self { DummyWandb }
    pub fn log(&self, _data: &std::collections::HashMap<String, f64>) {}
    pub fn finish(&self) {}
}

#[allow(dead_code)]
pub fn get_peak_flops(_device_name: &str) -> f64 {
    0.0
}
