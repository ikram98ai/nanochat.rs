//! Utilities for saving and loading model/state checkpoints.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use serde_json::Value;

use super::common::get_base_dir;
use super::gpt::{GPT, GPTConfig};
use super::tokenizer::{get_tokenizer, Tokenizer};

fn msg(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

fn log0(message: &str) {
    if std::env::var("RANK").ok().and_then(|r| r.parse::<usize>().ok()).unwrap_or(0) == 0 {
        log::info!("{message}");
    }
}

/// Add default values for new config keys missing in old checkpoints.
fn patch_missing_config_keys(model_config_kwargs: &mut Value) {
    // Old models were trained with full context (no sliding window)
    if model_config_kwargs.get("window_pattern").is_none() {
        model_config_kwargs["window_pattern"] = Value::String("L".to_string());
        log0("Patching missing window_pattern in model config to 'L'");
    }
}

/// Add default values for new parameters that may be missing in old checkpoints.
fn patch_missing_keys(model_data: &mut HashMap<String, Tensor>, model_config: &GPTConfig, device: &Device) -> Result<()> {
    let n_layer = model_config.n_layer;
    // resid_lambdas defaults to 1.0 (identity scaling)
    if !model_data.contains_key("resid_lambdas") {
        model_data.insert("resid_lambdas".into(), Tensor::ones(n_layer, DType::F32, device)?);
        log0("Patching missing resid_lambdas in model data to 1.0");
    }
    // x0_lambdas defaults to 0.0 (disabled)
    if !model_data.contains_key("x0_lambdas") {
        model_data.insert("x0_lambdas".into(), Tensor::zeros(n_layer, DType::F32, device)?);
        log0("Patching missing x0_lambdas in model data to 0.0");
    }
    Ok(())
}

/// Save model tensors (safetensors) and metadata (json) for a given step.
pub fn save_checkpoint(
    checkpoint_dir: &Path,
    step: usize,
    model_data: &HashMap<String, Tensor>,
    meta_data: &Value,
) -> Result<()> {
    std::fs::create_dir_all(checkpoint_dir).map_err(msg)?;
    // Save the model state parameters
    let model_path = checkpoint_dir.join(format!("model_{step:06}.safetensors"));
    candle_core::safetensors::save(model_data, &model_path)?;
    log::info!("Saved model parameters to: {}", model_path.display());
    // Save the metadata dict as json
    let meta_path = checkpoint_dir.join(format!("meta_{step:06}.json"));
    std::fs::write(&meta_path, serde_json::to_string_pretty(meta_data).map_err(msg)?).map_err(msg)?;
    log::info!("Saved metadata to: {}", meta_path.display());
    Ok(())
}

/// Load the model state and metadata for a given step. Reads either the
/// safetensors export or the original PyTorch .pt checkpoint.
pub fn load_checkpoint(
    checkpoint_dir: &Path,
    step: usize,
    device: &Device,
) -> Result<(HashMap<String, Tensor>, Value)> {
    let safetensors_path = checkpoint_dir.join(format!("model_{step:06}.safetensors"));
    let pt_path = checkpoint_dir.join(format!("model_{step:06}.pt"));
    let model_data: HashMap<String, Tensor> = if safetensors_path.exists() {
        candle_core::safetensors::load(&safetensors_path, device)?
    } else {
        candle_core::pickle::read_all(&pt_path)?.into_iter().collect()
    };
    // Load the metadata
    let meta_path = checkpoint_dir.join(format!("meta_{step:06}.json"));
    let meta_data = std::fs::read_to_string(&meta_path)
        .map_err(|e| msg(format!("Error reading {}: {e}", meta_path.display())))?;
    let meta_data: Value = serde_json::from_str(&meta_data).map_err(msg)?;
    Ok((model_data, meta_data))
}

/// A bunch of repetitive code to build a model from a given checkpoint.
/// Returns the model, the tokenizer, and the meta data saved during training.
pub fn build_model(
    checkpoint_dir: &Path,
    step: usize,
    device: &Device,
    phase: &str,
) -> Result<(GPT, Tokenizer, Value)> {
    assert!(phase == "train" || phase == "eval", "Invalid phase: {phase}");
    let (model_data, meta_data) = load_checkpoint(checkpoint_dir, step, device)?;
    // Convert reduced-precision tensors to float for CPU inference,
    // and fix torch compile checkpoints which prepend all keys with _orig_mod.
    let mut model_data: HashMap<String, Tensor> = model_data
        .into_iter()
        .map(|(k, v)| {
            let k = k.strip_prefix("_orig_mod.").unwrap_or(&k).to_string();
            let v = if v.dtype() != DType::F32 { v.to_dtype(DType::F32)? } else { v };
            Ok((k, v))
        })
        .collect::<Result<_>>()?;
    let mut model_config_value = meta_data["model_config"].clone();
    patch_missing_config_keys(&mut model_config_value);
    log0(&format!("Building model with config: {model_config_value}"));
    let model_config: GPTConfig = serde_json::from_value(model_config_value).map_err(msg)?;
    patch_missing_keys(&mut model_data, &model_config, device)?;
    let vb = VarBuilder::from_tensors(model_data, DType::F32, device);
    let model = GPT::new(model_config.clone(), vb)?;
    // Load the Tokenizer
    let tokenizer = get_tokenizer()?;
    // Sanity check: compatibility between model and tokenizer
    assert!(
        tokenizer.get_vocab_size() == model_config.vocab_size,
        "Tokenizer vocab size {} does not match model config vocab size {}",
        tokenizer.get_vocab_size(),
        model_config.vocab_size
    );
    Ok((model, tokenizer, meta_data))
}

/// Attempt to guess the model tag: take the biggest model available.
pub fn find_largest_model(checkpoints_dir: &Path) -> Result<String> {
    let model_tags: Vec<String> = std::fs::read_dir(checkpoints_dir)
        .map_err(|e| msg(format!("No checkpoints found in {}: {e}", checkpoints_dir.display())))?
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    if model_tags.is_empty() {
        return Err(msg(format!("No checkpoints found in {}", checkpoints_dir.display())));
    }
    // 1) normally all model tags are of the form d<number>, try that first:
    let mut candidates: Vec<(usize, &String)> = model_tags
        .iter()
        .filter_map(|tag| {
            tag.strip_prefix('d')
                .and_then(|rest| {
                    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                    digits.parse::<usize>().ok()
                })
                .map(|depth| (depth, tag))
        })
        .collect();
    if !candidates.is_empty() {
        candidates.sort_by(|a, b| b.0.cmp(&a.0));
        return Ok(candidates[0].1.clone());
    }
    // 2) if that failed, take the most recently updated model:
    let mut with_mtime: Vec<(std::time::SystemTime, &String)> = model_tags
        .iter()
        .map(|tag| {
            let mtime = std::fs::metadata(checkpoints_dir.join(tag))
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            (mtime, tag)
        })
        .collect();
    with_mtime.sort_by(|a, b| b.0.cmp(&a.0));
    Ok(with_mtime[0].1.clone())
}

/// Look into checkpoint_dir and find the model checkpoint (model_<step>.pt or
/// model_<step>.safetensors) with the highest step.
pub fn find_last_step(checkpoint_dir: &Path) -> Result<usize> {
    let mut last_step: Option<usize> = None;
    for entry in std::fs::read_dir(checkpoint_dir)
        .map_err(|e| msg(format!("No checkpoints found in {}: {e}", checkpoint_dir.display())))?
        .flatten()
    {
        let name = entry.file_name().to_string_lossy().into_owned();
        let step = name
            .strip_prefix("model_")
            .and_then(|rest| rest.strip_suffix(".pt").or_else(|| rest.strip_suffix(".safetensors")))
            .filter(|step_str| !step_str.is_empty() && step_str.chars().all(|c| c.is_ascii_digit()))
            .and_then(|step_str| step_str.parse::<usize>().ok());
        if let Some(step) = step {
            last_step = Some(last_step.map_or(step, |prev| prev.max(step)));
        }
    }
    last_step.ok_or_else(|| msg(format!("No checkpoints found in {}", checkpoint_dir.display())))
}

/// Build a trainable model (VarMap-backed parameters) from a checkpoint.
/// Returns the model, its VarMap, the tokenizer, and the training meta data.
pub fn build_model_for_training(
    checkpoint_dir: &Path,
    step: usize,
    device: &Device,
) -> Result<(GPT, candle_nn::VarMap, Tokenizer, Value)> {
    let (model_data, meta_data) = load_checkpoint(checkpoint_dir, step, device)?;
    let mut model_data: HashMap<String, Tensor> = model_data
        .into_iter()
        .map(|(k, v)| {
            let k = k.strip_prefix("_orig_mod.").unwrap_or(&k).to_string();
            let v = if v.dtype() != DType::F32 { v.to_dtype(DType::F32)? } else { v };
            Ok((k, v))
        })
        .collect::<Result<_>>()?;
    let mut model_config_value = meta_data["model_config"].clone();
    patch_missing_config_keys(&mut model_config_value);
    let model_config: GPTConfig = serde_json::from_value(model_config_value).map_err(msg)?;
    patch_missing_keys(&mut model_data, &model_config, device)?;
    let mut varmap = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = GPT::new(model_config.clone(), vb)?;
    varmap.set(model_data.iter().map(|(k, v)| (k, v)))?;
    let tokenizer = get_tokenizer()?;
    assert!(
        tokenizer.get_vocab_size() == model_config.vocab_size,
        "Tokenizer vocab size {} does not match model config vocab size {}",
        tokenizer.get_vocab_size(),
        model_config.vocab_size
    );
    Ok((model, varmap, tokenizer, meta_data))
}

/// Load just the optimizer state for a given step, if present.
pub fn load_optimizer_state(
    checkpoint_dir: &Path,
    step: usize,
    device: &Device,
) -> Result<Option<HashMap<String, Tensor>>> {
    let optimizer_path = checkpoint_dir.join(format!("optim_{step:06}_rank0.safetensors"));
    if !optimizer_path.exists() {
        log0(&format!("Optimizer checkpoint not found: {}", optimizer_path.display()));
        return Ok(None);
    }
    log0(&format!("Loading optimizer state from {}", optimizer_path.display()));
    Ok(Some(candle_core::safetensors::load(&optimizer_path, device)?))
}

// -----------------------------------------------------------------------------
// convenience functions that take into account nanochat's directory structure

pub fn load_model_from_dir(
    checkpoints_dir: &Path,
    device: &Device,
    phase: &str,
    model_tag: Option<&str>,
    step: Option<usize>,
) -> Result<(GPT, Tokenizer, Value)> {
    let model_tag = match model_tag {
        Some(tag) => tag.to_string(),
        None => {
            // guess the model tag by defaulting to the largest model
            let tag = find_largest_model(checkpoints_dir)?;
            log0(&format!("No model tag provided, guessing model tag: {tag}"));
            tag
        }
    };
    let checkpoint_dir = checkpoints_dir.join(&model_tag);
    let step = match step {
        Some(step) => step,
        // guess the step by defaulting to the last step
        None => find_last_step(&checkpoint_dir)?,
    };
    // build the model
    log0(&format!("Loading model from {} with step {step}", checkpoint_dir.display()));
    build_model(&checkpoint_dir, step, device, phase)
}

pub fn load_model(
    source: &str,
    device: &Device,
    phase: &str,
    model_tag: Option<&str>,
    step: Option<usize>,
) -> Result<(GPT, Tokenizer, Value)> {
    let model_dir = match source {
        "base" => "base_checkpoints",
        "sft" => "chatsft_checkpoints",
        "rl" => "chatrl_checkpoints",
        other => return Err(msg(format!("Unknown model source: {other}"))),
    };
    let checkpoints_dir = get_base_dir().join(model_dir);
    load_model_from_dir(&checkpoints_dir, device, phase, model_tag, step)
}

/// Helper for tests and tools: directory of the d<depth> naming convention.
#[allow(dead_code)]
pub fn model_tag_dir(source: &str, tag: &str) -> PathBuf {
    let model_dir = match source {
        "base" => "base_checkpoints",
        "sft" => "chatsft_checkpoints",
        "rl" => "chatrl_checkpoints",
        _ => source,
    };
    get_base_dir().join(model_dir).join(tag)
}
