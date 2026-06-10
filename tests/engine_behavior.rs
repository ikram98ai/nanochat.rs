//! Engine behavior tests, mirroring the Python tests/test_engine.py:
//! KV cache mechanics, sampling diversity/reproducibility, and the
//! calculator tool, using a mock model and a byte-level tokenizer.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device, Result, Tensor};

use nanochat::engine::{use_calculator, Engine, EngineModel, EngineTokenizer, KVCache};
use nanochat::gpt::GPTConfig;

/// Minimal config for Engine tests.
fn mock_config() -> GPTConfig {
    GPTConfig {
        sequence_len: 128,
        vocab_size: 262, // 256 bytes + 6 special tokens
        n_layer: 2,
        n_head: 4,
        n_kv_head: 4,
        n_embd: 64,
        window_pattern: "SSSL".to_string(),
    }
}

/// Mock model that returns uniform logits over the vocab. This ensures that
/// with temperature > 0, different samples should (with very high probability)
/// produce different tokens.
struct MockModel {
    config: GPTConfig,
    device: Device,
}

impl MockModel {
    fn new() -> Self {
        Self { config: mock_config(), device: Device::Cpu }
    }
}

impl EngineModel for MockModel {
    fn config(&self) -> &GPTConfig {
        &self.config
    }
    fn get_device(&self) -> &Device {
        &self.device
    }
    fn forward(&self, ids: &Tensor, kv_cache: Option<&KVCache>) -> Result<Tensor> {
        let (b, t) = ids.dims2()?;
        // The attention layer updates the cache in-place and advances the position
        if let Some(kv_cache) = kv_cache {
            kv_cache.advance(t);
        }
        // Uniform logits -> equal probability for all tokens
        Tensor::zeros((b, t, self.config.vocab_size), DType::F32, &self.device)
    }
}

/// Simple byte-level tokenizer for testing.
/// Tokens 0-255 are raw bytes, 256+ are special tokens.
struct ByteTokenizer {
    special_tokens: HashMap<&'static str, u32>,
}

impl ByteTokenizer {
    fn new() -> Self {
        let special_tokens = HashMap::from([
            ("<|python_start|>", 256),
            ("<|python_end|>", 257),
            ("<|output_start|>", 258),
            ("<|output_end|>", 259),
            ("<|assistant_end|>", 260),
            ("<|bos|>", 261),
        ]);
        Self { special_tokens }
    }
}

impl EngineTokenizer for ByteTokenizer {
    fn encode_special(&self, s: &str) -> u32 {
        self.special_tokens[s]
    }
    fn get_bos_token_id(&self) -> u32 {
        261
    }
    fn encode(&self, s: &str) -> Vec<u32> {
        s.bytes().map(|b| b as u32).collect()
    }
    fn decode(&self, tokens: &[u32]) -> String {
        let bytes: Vec<u8> = tokens.iter().filter(|&&t| t < 256).map(|&t| t as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

const PROMPT: [u32; 6] = [261, 72, 101, 108, 108, 111]; // <bos> + "Hello"

#[test]
fn test_kv_cache_basic() {
    let (batch_size, num_heads, seq_len, head_dim, num_layers) = (2, 3, 64, 5, 6);
    let kv_cache = KVCache::new(
        batch_size, num_heads, seq_len, head_dim, num_layers, &Device::Cpu, DType::F32,
    ).unwrap();

    // Check initial state
    assert_eq!(kv_cache.get_pos(), 0);
    let (k0, v0) = kv_cache.get_layer_cache(0);
    assert_eq!(k0.dims(), [batch_size, seq_len, num_heads, head_dim]);
    assert_eq!(v0.dims(), [batch_size, seq_len, num_heads, head_dim]);

    // Test advance
    kv_cache.advance(10);
    assert_eq!(kv_cache.get_pos(), 10);
    kv_cache.advance(5);
    assert_eq!(kv_cache.get_pos(), 15);

    // Test reset
    kv_cache.reset();
    assert_eq!(kv_cache.get_pos(), 0);
}

#[test]
fn test_kv_cache_prefill() {
    let (batch_size, num_heads, head_dim, num_layers) = (1, 4, 8, 2);
    let src_cache = KVCache::new(
        batch_size, num_heads, 32, head_dim, num_layers, &Device::Cpu, DType::F32,
    ).unwrap();
    // Write some data to source cache
    let (k0, v0) = src_cache.get_layer_cache(0);
    k0.slice_set(&Tensor::ones((1, 16, num_heads, head_dim), DType::F32, &Device::Cpu).unwrap(), 1, 0).unwrap();
    v0.slice_set(&(Tensor::ones((1, 16, num_heads, head_dim), DType::F32, &Device::Cpu).unwrap() * 2.0).unwrap(), 1, 0).unwrap();
    src_cache.advance(16);

    // Create destination cache with larger seq_len
    let dst_cache = KVCache::new(
        batch_size, num_heads, 64, head_dim, num_layers, &Device::Cpu, DType::F32,
    ).unwrap();
    dst_cache.prefill(&src_cache).unwrap();

    // Check position was copied
    assert_eq!(dst_cache.get_pos(), 16);

    // Check data was copied
    let (k0, v0) = dst_cache.get_layer_cache(0);
    let k_sum = k0.narrow(1, 0, 16).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
    let v_sum = v0.narrow(1, 0, 16).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
    assert_eq!(k_sum, (16 * num_heads * head_dim) as f32);
    assert_eq!(v_sum, (2 * 16 * num_heads * head_dim) as f32);
}

/// When generating multiple samples, each sample must get an independently
/// sampled first token (not a broadcast of the same token to all rows).
#[test]
fn test_multi_sample_first_token_diversity() {
    let model = MockModel::new();
    let tokenizer = ByteTokenizer::new();
    let engine = Engine::new(&model, &tokenizer);

    let num_samples = 16;
    let mut first_tokens = Vec::new();
    let stream = engine.generate(&PROMPT, num_samples, Some(1), 1.0, None, 42).unwrap();
    for step in stream {
        let (token_column, _) = step.unwrap();
        first_tokens = token_column; // first (and only) yield
    }
    let unique: std::collections::HashSet<u32> = first_tokens.iter().copied().collect();
    assert!(
        unique.len() > 1,
        "All {num_samples} samples got the same first token ({}). With uniform logits \
         this is statistically impossible unless tokens are broadcast instead of sampled.",
        first_tokens[0]
    );
}

/// Same seed must produce identical output.
#[test]
fn test_seed_reproducibility() {
    let model = MockModel::new();
    let tokenizer = ByteTokenizer::new();
    let engine = Engine::new(&model, &tokenizer);

    for seed in [1, 42, 123, 999] {
        let (r1, _) = engine.generate_batch(&PROMPT, 1, Some(5), 1.0, None, seed).unwrap();
        let (r2, _) = engine.generate_batch(&PROMPT, 1, Some(5), 1.0, None, seed).unwrap();
        let (r3, _) = engine.generate_batch(&PROMPT, 1, Some(5), 1.0, None, seed).unwrap();
        assert!(r1 == r2 && r2 == r3, "Same seed must produce identical output for the same prompt.");
    }
}

/// Temperature=0 is deterministic regardless of seed.
#[test]
fn test_temperature_zero_determinism() {
    let model = MockModel::new();
    let tokenizer = ByteTokenizer::new();
    let engine = Engine::new(&model, &tokenizer);

    let (r1, _) = engine.generate_batch(&PROMPT, 1, Some(5), 0.0, None, 1).unwrap();
    let (r2, _) = engine.generate_batch(&PROMPT, 1, Some(5), 0.0, None, 42).unwrap();
    let (r3, _) = engine.generate_batch(&PROMPT, 1, Some(5), 0.0, None, 123).unwrap();
    assert!(r1 == r2 && r2 == r3, "Temperature=0 must give the same output regardless of seed.");
}

/// Generation stops at max_tokens limit.
#[test]
fn test_max_tokens_respected() {
    let model = MockModel::new();
    let tokenizer = ByteTokenizer::new();
    let engine = Engine::new(&model, &tokenizer);

    for max_tokens in [1, 4, 16, 64] {
        let (results, _) = engine.generate_batch(&PROMPT, 1, Some(max_tokens), 1.0, None, 42).unwrap();
        let num_generated = results[0].len() - PROMPT.len();
        assert!(
            num_generated <= max_tokens,
            "Generated {num_generated} tokens, expected max_tokens={max_tokens} or less."
        );
    }
}

/// num_samples=N produces exactly N sequences.
#[test]
fn test_num_samples_count() {
    let model = MockModel::new();
    let tokenizer = ByteTokenizer::new();
    let engine = Engine::new(&model, &tokenizer);

    for num_samples in [1, 4, 16, 64] {
        let (results, _) = engine.generate_batch(&PROMPT, num_samples, Some(3), 1.0, None, 42).unwrap();
        assert_eq!(results.len(), num_samples);
    }
}

/// With temperature > 0, different seeds should introduce sampling variation.
#[test]
fn test_different_seeds_introduce_variation() {
    let model = MockModel::new();
    let tokenizer = ByteTokenizer::new();
    let engine = Engine::new(&model, &tokenizer);

    let mut outputs = std::collections::HashSet::new();
    for seed in [1, 42, 123, 999, 1000, 1001, 1002, 1003, 1004, 1005] {
        let (results, _) = engine.generate_batch(&PROMPT, 1, Some(5), 1.0, None, seed).unwrap();
        outputs.insert(results[0].clone());
    }
    assert!(outputs.len() > 1, "All seeds produced the same output which is highly improbable.");
}

/// The calculator tool must produce exactly the Python eval results
/// (cases and expected strings generated by the Python implementation).
#[test]
fn test_calculator_parity() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/meta.json");
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for case in meta["calculator_cases"].as_array().unwrap() {
        let expr = case["expr"].as_str().unwrap();
        let expected = case["result"].as_str().map(|s| s.to_string());
        assert_eq!(use_calculator(expr), expected, "use_calculator({expr:?})");
    }
}

/// Forced tool-output tokens: sampling <|python_start|>expr<|python_end|>
/// queues <|output_start|>result<|output_end|> as forced tokens with mask 0.
#[test]
fn test_tool_use_forcing() {
    /// Model that deterministically emits "<|python_start|>1+2<|python_end|>"
    /// then keeps emitting 'x'. Argmax sampling (temperature=0) follows the script.
    struct ScriptedModel {
        config: GPTConfig,
        device: Device,
        script: Vec<u32>,
        step: std::cell::Cell<usize>,
    }
    impl EngineModel for ScriptedModel {
        fn config(&self) -> &GPTConfig {
            &self.config
        }
        fn get_device(&self) -> &Device {
            &self.device
        }
        fn forward(&self, ids: &Tensor, kv_cache: Option<&KVCache>) -> Result<Tensor> {
            let (b, t) = ids.dims2()?;
            if let Some(kv_cache) = kv_cache {
                kv_cache.advance(t);
            }
            let vocab = self.config.vocab_size;
            let next = *self.script.get(self.step.get()).unwrap_or(&(b'x' as u32));
            self.step.set(self.step.get() + 1);
            let mut logits = vec![0f32; b * t * vocab];
            // set the scripted token as argmax at the last position of each row
            for row in 0..b {
                logits[row * t * vocab + (t - 1) * vocab + next as usize] = 1.0;
            }
            Tensor::from_vec(logits, (b, t, vocab), &self.device)
        }
    }

    let tokenizer = ByteTokenizer::new();
    let python_start = 256;
    let python_end = 257;
    let output_start = 258;
    let output_end = 259;
    let script = vec![python_start, b'1' as u32, b'+' as u32, b'2' as u32, python_end];
    let model = ScriptedModel {
        config: mock_config(),
        device: Device::Cpu,
        script,
        step: std::cell::Cell::new(0),
    };
    let engine = Engine::new(&model, &tokenizer);
    let stream = engine.generate(&PROMPT, 1, Some(10), 0.0, None, 42).unwrap();
    let mut tokens = Vec::new();
    let mut masks = Vec::new();
    for step in stream {
        let (column, mask) = step.unwrap();
        tokens.push(column[0]);
        masks.push(mask[0]);
    }
    // sampled: python block; forced: <|output_start|>"3"<|output_end|>
    let expected_prefix = vec![
        python_start, b'1' as u32, b'+' as u32, b'2' as u32, python_end,
        output_start, b'3' as u32, output_end,
    ];
    assert_eq!(&tokens[..8], &expected_prefix[..]);
    assert_eq!(&masks[..8], &[1, 1, 1, 1, 1, 0, 0, 0]);
}
