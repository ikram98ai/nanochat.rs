use candle_core::{DType, Device, Result};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use crate::nanochat::gpt::{GPT, GPTConfig};
use crate::nanochat::tokenizer::Tokenizer;
use crate::nanochat::common::get_base_dir;

#[allow(dead_code)]
pub fn find_largest_model(base_dir: &Path) -> Option<PathBuf> {
    if !base_dir.exists() { return None; }
    let mut best = None;
    if let Ok(entries) = std::fs::read_dir(base_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                best = Some(path);
            }
        }
    }
    best
}

pub fn load_model(
    source: &str,
    device: &Device,
) -> Result<(GPT, Tokenizer, HashMap<String, serde_json::Value>)> {
    load_checkpoint(source, None, None, device)
}

pub fn load_checkpoint(
    source: &str,
    step: Option<usize>,
    _model_tag: Option<&str>,
    device: &Device,
) -> Result<(GPT, Tokenizer, HashMap<String, serde_json::Value>)> {
    let base_dir = get_base_dir();
    let source_dir = base_dir.join(source);
    
    let step_dir = if let Some(s) = step {
        source_dir.join(format!("step_{:06}", s))
    } else {
        // Find latest step or use base
        source_dir
    };

    // Load config
    let config_path = step_dir.join("config.json");
    log::info!("Loading config from: {}", config_path.display());
    let config_data = std::fs::read_to_string(&config_path)
        .map_err(|e| candle_core::Error::Msg(format!("Error reading {}: {}", config_path.display(), e)))?;
    let config: GPTConfig = serde_json::from_str(&config_data)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    // Load weights
    let weights_path = step_dir.join("model.safetensors");
    log::info!("Loading weights from: {}", weights_path.display());
    let vb = if weights_path.exists() {
        unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)? }
    } else {
        return Err(candle_core::Error::Msg(format!("No model weights found in {}", step_dir.display())));
    };

    let model = GPT::new(config, vb)?;

    // Load tokenizer
    let tokenizer_path = step_dir.join("tokenizer.json");
    log::info!("Loading tokenizer from: {}", tokenizer_path.display());
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;

    Ok((model, tokenizer, HashMap::new()))
}

#[allow(dead_code)]
pub fn save_checkpoint(
    _model: &GPT,
    _tokenizer: &Tokenizer,
    _metadata: &HashMap<String, serde_json::Value>,
    _target_dir: &Path,
) -> Result<()> {
    Ok(())
}
