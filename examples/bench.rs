//! Benchmark the Rust engine on CPU for comparison with the Python version.
//!
//! cargo run --release --example bench

use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use nanochat::engine::KVCache;
use nanochat::gpt::{GPTConfig, GPT};

fn bench(name: &str, model: &GPT, prompt_len: usize, decode_steps: usize) -> anyhow::Result<()> {
    let config = &model.config;
    let device = Device::Cpu;
    let mut rng = StdRng::seed_from_u64(11);
    let prompt: Vec<i64> = (0..prompt_len)
        .map(|_| rng.random_range(0..config.vocab_size as i64))
        .collect();
    let prompt = Tensor::from_vec(prompt, (1, prompt_len), &device)?;
    let kv = KVCache::new(
        1,
        config.n_kv_head,
        prompt_len + decode_steps,
        config.n_embd / config.n_head,
        config.n_layer,
        &device,
        DType::F32,
    )?;
    let t0 = Instant::now();
    let mut logits = model.forward(&prompt, None, Some(&kv), "mean")?;
    let t1 = Instant::now();
    for _ in 0..decode_steps {
        let t = logits.dim(1)?;
        let next_tok = logits
            .narrow(1, t - 1, 1)?
            .squeeze(1)?
            .argmax(1)?
            .to_vec1::<u32>()?[0];
        let ids = Tensor::from_vec(vec![next_tok as i64], (1, 1), &device)?;
        logits = model.forward(&ids, None, Some(&kv), "mean")?;
    }
    let t2 = Instant::now();
    let prefill_s = (t1 - t0).as_secs_f64();
    let decode_s = (t2 - t1).as_secs_f64();
    println!(
        "{name}: prefill {prompt_len} tok in {prefill_s:.3}s ({:.0} tok/s), \
         decode {decode_steps} tok in {decode_s:.3}s ({:.1} tok/s)",
        prompt_len as f64 / prefill_s,
        decode_steps as f64 / decode_s,
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;

    // tiny model: identical weights to the Python parity fixtures
    let data_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(data_dir.join("meta.json"))?)?;
    let config: GPTConfig = serde_json::from_value(meta["config"].clone())?;
    let weights = candle_core::safetensors::load(data_dir.join("model.safetensors"), &device)?;
    let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
    let model = GPT::new(config, vb)?;
    bench("tiny  (d5/64emb)", &model, 150, 100)?;

    // medium model: realistic-ish depth/width, random weights (speed only)
    let config = GPTConfig {
        sequence_len: 1024,
        vocab_size: 32768,
        n_layer: 8,
        n_head: 8,
        n_kv_head: 4,
        n_embd: 512,
        window_pattern: "SSSL".to_string(),
    };
    let weights = GPT::init_weights(&config, 64, &device, 42)?;
    let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
    let model = GPT::new(config, vb)?;
    bench("medium(d8/512emb)", &model, 200, 100)?;

    Ok(())
}
