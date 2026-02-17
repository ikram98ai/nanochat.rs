use candle_core::{Tensor, Result, DType, Device};
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use crate::nanochat::gpt::GPT;
use crate::nanochat::tokenizer::Tokenizer;
use crate::nanochat::flash_attention;

/// KV Cache for efficiently generating tokens one by one.
pub struct KVCache {
    pub entries: Vec<flash_attention::KVCacheEntry>,
}

impl KVCache {
    /// Create a new KV cache with zeros.
    #[allow(dead_code)]
    pub fn new(n_layer: usize, b: usize, max_seq_len: usize, n_kv_head: usize, head_dim: usize, device: &Device) -> Result<Self> {
        let mut entries = Vec::with_capacity(n_layer);
        for _ in 0..n_layer {
            let k = Tensor::zeros((b, max_seq_len, n_kv_head, head_dim), DType::F32, device)?;
            let v = Tensor::zeros((b, max_seq_len, n_kv_head, head_dim), DType::F32, device)?;
            entries.push(flash_attention::KVCacheEntry { k, v });
        }
        Ok(Self { entries })
    }
}

pub struct TokenGenerator<'a> {
    engine: &'a mut Engine,
    tokens: Tensor,
    remaining: usize,
    temperature: f64,
    top_k: usize,
    kv_cache: Option<KVCache>,
    pos: usize,
}

pub fn sample_next_token(logits: &Tensor, temperature: f64, top_k: usize) -> Result<u32> {
    let logits = logits.to_dtype(DType::F32)?;
    let b_logits = logits.to_vec1()?;
    
    if temperature == 0.0 {
        let mut best_idx = 0;
        let mut best_val = b_logits[0];
        for (i, &v) in b_logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        return Ok(best_idx as u32);
    }

    let mut logits_with_idx: Vec<(usize, f32)> = b_logits.into_iter().enumerate().collect();
    logits_with_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    let k = top_k.min(logits_with_idx.len());
    let mut top_k_logits = Vec::with_capacity(k);
    let mut top_k_indices = Vec::with_capacity(k);
    for i in 0..k {
        top_k_indices.push(logits_with_idx[i].0);
        top_k_logits.push(logits_with_idx[i].1 / temperature as f32);
    }

    let max_logit = top_k_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_logits = Vec::with_capacity(k);
    let mut sum_exp = 0.0;
    for &l in &top_k_logits {
        let e = (l - max_logit).exp();
        exp_logits.push(e);
        sum_exp += e;
    }
    
    let probs: Vec<f64> = exp_logits.iter().map(|&e| (e / sum_exp) as f64).collect();
    let dist = WeightedIndex::new(&probs).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let mut rng = rand::rng();
    let idx = dist.sample(&mut rng);
    Ok(top_k_indices[idx] as u32)
}

pub struct Engine {
    pub model: GPT,
    pub tokenizer: Tokenizer,
}

impl Engine {
    pub fn new(model: GPT, tokenizer: Tokenizer) -> Self {
        Self { model, tokenizer }
    }

    pub fn generate(
        &mut self,
        input_ids: &[u32],
        max_new_tokens: usize,
        temperature: f64,
        top_k: usize,
    ) -> Result<TokenGenerator<'_>> {
        let device = self.model.device();
        let tokens = Tensor::new(input_ids, device)?.unsqueeze(0)?;
        
        // Initialize KV cache for faster generation
        // Rough estimate for max_seq_len (window/config should be used in full impl)
        let _max_seq_len = 2048; 
        let _n_layer = self.model.h.len(); 
        // In gpt.rs I'll make h public or add an accessor. 
        // For now, let's assume I modify gpt.rs too.
        
        Ok(TokenGenerator {
            engine: self,
            tokens,
            remaining: max_new_tokens,
            temperature,
            top_k,
            kv_cache: None, // Keep None for now until h is accessible and dims are certain
            pos: input_ids.len(),
        })
    }
}

impl<'a> Iterator for TokenGenerator<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if self.remaining == 0 {
            return None;
        }

        let logits = self.engine.model.forward(&self.tokens, self.kv_cache.as_mut().map(|c| &mut c.entries)).ok()?;
        let (_b, t, _v) = logits.dims3().ok()?;
        let last_logits = logits.narrow(1, t - 1, 1).ok()?.squeeze(1).ok()?.squeeze(0).ok()?;
        
        let next_token = sample_next_token(&last_logits, self.temperature, self.top_k).ok()?;
        
        let assistant_end = self.engine.tokenizer.token_to_id("<|assistant_end|>").unwrap_or(0);
        let eot = self.engine.tokenizer.token_to_id("<|endoftext|>").unwrap_or(0);
        
        if next_token == assistant_end || next_token == eot {
            self.remaining = 0;
            return None;
        }

        let device = self.engine.model.device();
        let next_tokens = Tensor::new(&[next_token], device).ok()?.unsqueeze(0).ok()?;
        self.tokens = Tensor::cat(&[&self.tokens, &next_tokens], 1).ok()?;
        self.remaining -= 1;
        self.pos += 1;
        
        Some(next_token)
    }
}
