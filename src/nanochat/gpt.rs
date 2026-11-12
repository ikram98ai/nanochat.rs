//! GPT model (rewrite, a lot simpler)
//! Notable features:
//! - rotary embeddings (and no positional embeddings)
//! - QK norm
//! - untied weights for token embedding and lm_head
//! - relu^2 activation in MLP
//! - norm after token embedding
//! - no learnable params in rmsnorm
//! - no bias in linear layers
//! - Group-Query Attention (GQA) support for more efficient inference
//! - sliding window attention with a per-layer long/short pattern

use candle_core::{DType, IndexOp, Result, Tensor, D};
use candle_nn::ops::{log_softmax, sigmoid};
use candle_nn::{embedding, Embedding, Linear, Module, VarBuilder};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use super::common::print0;
use super::engine::KVCache;
use super::flash_attention::{flash_attn_func, flash_attn_with_kvcache};

fn default_window_pattern() -> String {
    "SSSL".to_string()
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct GPTConfig {
    pub sequence_len: usize,
    pub vocab_size: usize,
    pub n_layer: usize,
    /// number of query heads
    pub n_head: usize,
    /// number of key/value heads (GQA)
    pub n_kv_head: usize,
    pub n_embd: usize,
    /// Sliding window attention pattern string, tiled across layers. Final layer always L.
    /// Characters: L=long (full context), S=short (quarter context)
    /// Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    #[serde(default = "default_window_pattern")]
    pub window_pattern: String,
}

impl Default for GPTConfig {
    fn default() -> Self {
        Self {
            sequence_len: 2048,
            vocab_size: 32768,
            n_layer: 12,
            n_head: 6,
            n_kv_head: 6,
            n_embd: 768,
            window_pattern: "SSSL".to_string(),
        }
    }
}

/// RMS norm without learnable params, eps matching torch (finfo(f32).eps).
pub fn norm(x: &Tensor) -> Result<Tensor> {
    let eps = f32::EPSILON as f64;
    let mean_sq = x.sqr()?.mean_keepdim(D::Minus1)?;
    x.broadcast_mul(&(mean_sq + eps)?.sqrt()?.recip()?)
}

/// Returns true if GPT layer should have Value Embedding (alternating, last layer always included).
pub fn has_ve(layer_idx: usize, n_layer: usize) -> bool {
    layer_idx % 2 == (n_layer - 1) % 2
}

pub fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _t, _h, dim) = x.dims4()?; // multihead attention
    let d = dim / 2;
    let x1 = x.narrow(D::Minus1, 0, d)?; // split up last dim into two halves
    let x2 = x.narrow(D::Minus1, d, d)?;
    let y1 = (x1.broadcast_mul(cos)? + x2.broadcast_mul(sin)?)?; // rotate pairs of dims
    let y2 = (x1.broadcast_mul(&sin.neg()?)? + x2.broadcast_mul(cos)?)?;
    Tensor::cat(&[y1, y2], 3)
}

pub struct CausalSelfAttention {
    layer_idx: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    c_q: Linear,
    c_k: Linear,
    c_v: Linear,
    c_proj: Linear,
    ve_gate_channels: usize,
    ve_gate: Option<Linear>,
}

impl CausalSelfAttention {
    fn new(config: &GPTConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let n_head = config.n_head;
        let n_kv_head = config.n_kv_head;
        let n_embd = config.n_embd;
        let head_dim = n_embd / n_head;
        assert!(n_embd % n_head == 0);
        assert!(n_kv_head <= n_head && n_head % n_kv_head == 0);
        let c_q = candle_nn::linear_no_bias(n_embd, n_head * head_dim, vb.pp("c_q"))?;
        let c_k = candle_nn::linear_no_bias(n_embd, n_kv_head * head_dim, vb.pp("c_k"))?;
        let c_v = candle_nn::linear_no_bias(n_embd, n_kv_head * head_dim, vb.pp("c_v"))?;
        let c_proj = candle_nn::linear_no_bias(n_embd, n_embd, vb.pp("c_proj"))?;
        let ve_gate_channels = 12;
        let ve_gate = if has_ve(layer_idx, config.n_layer) {
            Some(candle_nn::linear_no_bias(
                ve_gate_channels,
                n_kv_head,
                vb.pp("ve_gate"),
            )?)
        } else {
            None
        };
        Ok(Self {
            layer_idx,
            n_head,
            n_kv_head,
            head_dim,
            c_q,
            c_k,
            c_v,
            c_proj,
            ve_gate_channels,
            ve_gate,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        ve: Option<&Tensor>,
        cos_sin: (&Tensor, &Tensor),
        window_size: (i64, i64),
        kv_cache: Option<&KVCache>,
    ) -> Result<Tensor> {
        let (b, t, _c) = x.dims3()?;

        // Project the input to get queries, keys, and values
        // Shape: (B, T, H, D)
        let q = self
            .c_q
            .forward(x)?
            .reshape((b, t, self.n_head, self.head_dim))?;
        let k = self
            .c_k
            .forward(x)?
            .reshape((b, t, self.n_kv_head, self.head_dim))?;
        let mut v = self
            .c_v
            .forward(x)?
            .reshape((b, t, self.n_kv_head, self.head_dim))?;

        // Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if let (Some(ve), Some(ve_gate)) = (ve, &self.ve_gate) {
            let ve = ve.reshape((b, t, self.n_kv_head, self.head_dim))?;
            // (B, T, n_kv_head), range (0, 3)
            let gate_in = x
                .narrow(D::Minus1, 0, self.ve_gate_channels)?
                .contiguous()?;
            let gate = (sigmoid(&ve_gate.forward(&gate_in)?)? * 3.0)?;
            v = (v + gate.unsqueeze(D::Minus1)?.broadcast_mul(&ve)?)?;
        }

        // Apply Rotary Embeddings to queries and keys to get relative positional encoding
        let (cos, sin) = cos_sin;
        let q = apply_rotary_emb(&q, cos, sin)?;
        let k = apply_rotary_emb(&k, cos, sin)?;
        let (q, k) = (norm(&q)?, norm(&k)?); // QK norm
        let q = (q * 1.2)?; // sharper attention (split scale between Q and K)
        let k = (k * 1.2)?;

        // window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        let y = match kv_cache {
            None => {
                // Training: causal attention with optional sliding window
                flash_attn_func(&q, &k, &v, true, window_size)?
            }
            Some(kv_cache) => {
                // Inference: attention against the KV cache, which handles cache management
                let y = flash_attn_with_kvcache(
                    &q,
                    kv_cache,
                    self.layer_idx,
                    &k,
                    &v,
                    true,
                    window_size,
                )?;
                // Advance position after last layer processes
                if self.layer_idx == kv_cache.n_layers - 1 {
                    kv_cache.advance(t);
                }
                y
            }
        };

        // Re-assemble the heads and project back to residual stream
        let y = y.contiguous()?.reshape((b, t, ()))?;
        self.c_proj.forward(&y)
    }
}

pub struct MLP {
    c_fc: Linear,
    c_proj: Linear,
}

impl MLP {
    fn new(config: &GPTConfig, vb: VarBuilder) -> Result<Self> {
        let c_fc = candle_nn::linear_no_bias(config.n_embd, 4 * config.n_embd, vb.pp("c_fc"))?;
        let c_proj = candle_nn::linear_no_bias(4 * config.n_embd, config.n_embd, vb.pp("c_proj"))?;
        Ok(Self { c_fc, c_proj })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.c_fc.forward(x)?;
        let x = x.relu()?.sqr()?;
        self.c_proj.forward(&x)
    }
}

pub struct Block {
    attn: CausalSelfAttention,
    mlp: MLP,
}

impl Block {
    fn new(config: &GPTConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let attn = CausalSelfAttention::new(config, layer_idx, vb.pp("attn"))?;
        let mlp = MLP::new(config, vb.pp("mlp"))?;
        Ok(Self { attn, mlp })
    }

    fn forward(
        &self,
        x: &Tensor,
        ve: Option<&Tensor>,
        cos_sin: (&Tensor, &Tensor),
        window_size: (i64, i64),
        kv_cache: Option<&KVCache>,
    ) -> Result<Tensor> {
        let x = (x + self
            .attn
            .forward(&norm(x)?, ve, cos_sin, window_size, kv_cache)?)?;
        let x = (&x + self.mlp.forward(&norm(&x)?)?)?;
        Ok(x)
    }
}

pub struct GPT {
    pub config: GPTConfig,
    wte: Embedding,
    h: Vec<Block>,
    lm_head: Linear,
    /// Per-layer learnable scalars:
    /// resid_lambdas scales the residual stream at each layer,
    /// x0_lambdas blends the initial embedding back in at each layer.
    resid_lambdas: Tensor,
    x0_lambdas: Tensor,
    /// Smear: mix previous token's embedding into current token (cheap bigram-like info)
    smear_gate: Linear,
    smear_lambda: Tensor,
    /// Backout: subtract cached mid-layer residual before final norm to remove low-level features
    backout_lambda: Tensor,
    /// Value embeddings (ResFormer-style): alternating layers, last layer always included
    value_embeds: Vec<Option<Embedding>>,
    #[allow(dead_code)]
    rotary_seq_len: usize,
    cos: Tensor,
    sin: Tensor,
    pub window_sizes: Vec<(i64, i64)>,
}

impl GPT {
    pub fn new(config: GPTConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_padded(config, 64, vb)
    }

    pub fn new_padded(config: GPTConfig, pad_vocab_size_to: usize, vb: VarBuilder) -> Result<Self> {
        // Compute per-layer window sizes for sliding window attention
        let window_sizes = Self::compute_window_sizes(&config);
        // Pad vocab for efficiency. This is just an optimization - outputs are cropped in forward().
        let padded_vocab_size =
            ((config.vocab_size + pad_vocab_size_to - 1) / pad_vocab_size_to) * pad_vocab_size_to;
        if padded_vocab_size != config.vocab_size {
            print0(&format!(
                "Padding vocab_size from {} to {} for efficiency",
                config.vocab_size, padded_vocab_size
            ));
        }
        let vb_t = vb.pp("transformer");
        let wte = embedding(padded_vocab_size, config.n_embd, vb_t.pp("wte"))?;
        let mut h = Vec::with_capacity(config.n_layer);
        for layer_idx in 0..config.n_layer {
            h.push(Block::new(&config, layer_idx, vb_t.pp("h").pp(layer_idx))?);
        }
        let lm_head =
            candle_nn::linear_no_bias(config.n_embd, padded_vocab_size, vb.pp("lm_head"))?;
        let resid_lambdas = vb.get(config.n_layer, "resid_lambdas")?;
        let x0_lambdas = vb.get(config.n_layer, "x0_lambdas")?;
        let smear_gate = candle_nn::linear_no_bias(24, 1, vb.pp("smear_gate"))?;
        let smear_lambda = vb.get(1, "smear_lambda")?;
        let backout_lambda = vb.get(1, "backout_lambda")?;
        let head_dim = config.n_embd / config.n_head;
        let kv_dim = config.n_kv_head * head_dim;
        let mut value_embeds = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            if has_ve(i, config.n_layer) {
                value_embeds.push(Some(embedding(
                    padded_vocab_size,
                    kv_dim,
                    vb.pp("value_embeds").pp(i),
                )?));
            } else {
                value_embeds.push(None);
            }
        }
        // The rotary embeddings are pretty small/cheap in memory, so over-compute
        // them by 10X of sequence_len.
        let rotary_seq_len = config.sequence_len * 10;
        let (cos, sin) = Self::precompute_rotary_embeddings(rotary_seq_len, head_dim, vb.device())?;
        Ok(Self {
            config,
            wte,
            h,
            lm_head,
            resid_lambdas,
            x0_lambdas,
            smear_gate,
            smear_lambda,
            backout_lambda,
            value_embeds,
            rotary_seq_len,
            cos,
            sin,
            window_sizes,
        })
    }

    /// Initialize fresh model weights with the reference distributions:
    ///
    /// wte (embedding):     normal, std=0.8
    /// lm_head:             normal, std=0.001
    /// for each block:
    ///     attn.c_q/c_k/c_v: uniform, std=1/sqrt(n_embd)
    ///     attn.c_proj:      zeros
    ///     mlp.c_fc:         uniform, std=0.4/sqrt(n_embd)
    ///     mlp.c_proj:       zeros
    /// resid_lambdas decay 1.15 -> 1.05, x0_lambdas decay 0.20 -> 0.05,
    /// smear_lambda zero, backout_lambda 0.2, gates uniform [0, 0.02).
    ///
    /// Returns a name -> tensor map to construct the model from.
    pub fn init_weights(
        config: &GPTConfig,
        pad_vocab_size_to: usize,
        device: &candle_core::Device,
        seed: u64,
    ) -> Result<std::collections::HashMap<String, Tensor>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let n_embd = config.n_embd;
        let n_layer = config.n_layer;
        let padded_vocab_size =
            ((config.vocab_size + pad_vocab_size_to - 1) / pad_vocab_size_to) * pad_vocab_size_to;
        let head_dim = n_embd / config.n_head;
        let kv_dim = config.n_kv_head * head_dim;
        // sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        let s = 3f64.sqrt() * (n_embd as f64).powf(-0.5);

        let mut tensors = std::collections::HashMap::new();
        let normal = |shape: (usize, usize), std: f64, rng: &mut StdRng| -> Result<Tensor> {
            let dist = rand_distr::Normal::new(0.0f32, std as f32).unwrap();
            let data: Vec<f32> = (0..shape.0 * shape.1).map(|_| rng.sample(dist)).collect();
            Tensor::from_vec(data, shape, device)
        };
        let uniform =
            |shape: (usize, usize), lo: f64, hi: f64, rng: &mut StdRng| -> Result<Tensor> {
                let data: Vec<f32> = (0..shape.0 * shape.1)
                    .map(|_| rng.random_range(lo as f32..hi as f32))
                    .collect();
                Tensor::from_vec(data, shape, device)
            };

        tensors.insert(
            "transformer.wte.weight".to_string(),
            normal((padded_vocab_size, n_embd), 0.8, &mut rng)?,
        );
        tensors.insert(
            "lm_head.weight".to_string(),
            normal((padded_vocab_size, n_embd), 0.001, &mut rng)?,
        );
        for i in 0..n_layer {
            let p = format!("transformer.h.{i}");
            tensors.insert(
                format!("{p}.attn.c_q.weight"),
                uniform((n_embd, n_embd), -s, s, &mut rng)?,
            );
            tensors.insert(
                format!("{p}.attn.c_k.weight"),
                uniform((kv_dim, n_embd), -s, s, &mut rng)?,
            );
            tensors.insert(
                format!("{p}.attn.c_v.weight"),
                uniform((kv_dim, n_embd), -s, s, &mut rng)?,
            );
            tensors.insert(
                format!("{p}.attn.c_proj.weight"),
                Tensor::zeros((n_embd, n_embd), DType::F32, device)?,
            );
            tensors.insert(
                format!("{p}.mlp.c_fc.weight"),
                uniform((4 * n_embd, n_embd), -s * 0.4, s * 0.4, &mut rng)?,
            );
            tensors.insert(
                format!("{p}.mlp.c_proj.weight"),
                Tensor::zeros((n_embd, 4 * n_embd), DType::F32, device)?,
            );
            if has_ve(i, n_layer) {
                tensors.insert(
                    format!("value_embeds.{i}.weight"),
                    uniform((padded_vocab_size, kv_dim), -s, s, &mut rng)?,
                );
                tensors.insert(
                    format!("{p}.attn.ve_gate.weight"),
                    uniform((config.n_kv_head, 12), 0.0, 0.02, &mut rng)?,
                );
            }
        }
        // Per-layer resid init: stronger residual at early layers, weaker at deep layers
        let denom = (n_layer - 1).max(1) as f32;
        let resid: Vec<f32> = (0..n_layer)
            .map(|i| 1.15 - 0.10 * i as f32 / denom)
            .collect();
        // Decaying x0 init: earlier layers get more input embedding blending
        let x0: Vec<f32> = (0..n_layer)
            .map(|i| 0.20 - 0.15 * i as f32 / denom)
            .collect();
        tensors.insert(
            "resid_lambdas".to_string(),
            Tensor::from_vec(resid, n_layer, device)?,
        );
        tensors.insert(
            "x0_lambdas".to_string(),
            Tensor::from_vec(x0, n_layer, device)?,
        );
        tensors.insert(
            "smear_lambda".to_string(),
            Tensor::zeros(1, DType::F32, device)?,
        );
        tensors.insert(
            "backout_lambda".to_string(),
            (Tensor::ones(1, DType::F32, device)? * 0.2)?,
        );
        tensors.insert(
            "smear_gate.weight".to_string(),
            uniform((1, 24), 0.0, 0.02, &mut rng)?,
        );
        Ok(tensors)
    }

    fn precompute_rotary_embeddings(
        seq_len: usize,
        head_dim: usize,
        device: &candle_core::Device,
    ) -> Result<(Tensor, Tensor)> {
        let base = 100000f64;
        // stride the channels
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| (1.0 / base.powf(i as f64 / head_dim as f64)) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, head_dim / 2, device)?;
        // stride the time steps
        let t = Tensor::arange(0f32, seq_len as f32, device)?;
        // calculate the rotation frequencies at each (time, channel) pair
        let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let (cos, sin) = (freqs.cos()?, freqs.sin()?);
        // add batch and head dims for later broadcasting
        let cos = cos.reshape((1, seq_len, 1, head_dim / 2))?;
        let sin = sin.reshape((1, seq_len, 1, head_dim / 2))?;
        Ok((cos, sin))
    }

    /// Compute per-layer window sizes for sliding window attention.
    ///
    /// Returns a list of (left, right) tuples:
    /// - left: how many tokens before current position to attend to (-1 = unlimited)
    /// - right: how many tokens after current position to attend to (0 for causal)
    ///
    /// Pattern string is tiled across layers. Final layer always gets L (full context).
    fn compute_window_sizes(config: &GPTConfig) -> Vec<(i64, i64)> {
        let pattern: Vec<char> = config.window_pattern.to_uppercase().chars().collect();
        assert!(
            pattern.iter().all(|c| *c == 'S' || *c == 'L'),
            "Invalid window_pattern: {}. Use only S and L.",
            config.window_pattern
        );
        let long_window = config.sequence_len as i64;
        // ceil to attention tile size
        let short_window = ((long_window + 3) / 4 + 127) / 128 * 128;
        let mut window_sizes: Vec<(i64, i64)> = (0..config.n_layer)
            .map(|layer_idx| match pattern[layer_idx % pattern.len()] {
                'L' => (long_window, 0),
                _ => (short_window, 0),
            })
            .collect();
        // Final layer always gets full context
        *window_sizes.last_mut().unwrap() = (long_window, 0);
        window_sizes
    }

    pub fn get_device(&self) -> &candle_core::Device {
        self.wte.embeddings().device()
    }

    pub fn rotary_cos_sin(&self) -> (&Tensor, &Tensor) {
        (&self.cos, &self.sin)
    }

    fn param_counts(&self) -> (usize, usize, usize, usize, usize) {
        let wte = self.wte.embeddings().elem_count();
        let value_embeds: usize = self
            .value_embeds
            .iter()
            .flatten()
            .map(|ve| ve.embeddings().elem_count())
            .sum();
        let lm_head = self.lm_head.weight().elem_count();
        let transformer_matrices: usize = self
            .h
            .iter()
            .map(|b| {
                b.attn.c_q.weight().elem_count()
                    + b.attn.c_k.weight().elem_count()
                    + b.attn.c_v.weight().elem_count()
                    + b.attn.c_proj.weight().elem_count()
                    + b.attn
                        .ve_gate
                        .as_ref()
                        .map_or(0, |g| g.weight().elem_count())
                    + b.mlp.c_fc.weight().elem_count()
                    + b.mlp.c_proj.weight().elem_count()
            })
            .sum();
        let scalars = self.resid_lambdas.elem_count()
            + self.x0_lambdas.elem_count()
            + self.smear_gate.weight().elem_count()
            + self.smear_lambda.elem_count()
            + self.backout_lambda.elem_count();
        (wte, value_embeds, lm_head, transformer_matrices, scalars)
    }

    /// Return the estimated FLOPs per token for the model (forward + backward).
    /// Each matmul weight parameter contributes 2 FLOPs (multiply, accumulate) in
    /// forward, and 2X that in backward => 2+4=6. On top of that,
    /// 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside
    /// attention; with sliding windows, effective_seq_len varies per layer.
    pub fn estimate_flops(&self) -> usize {
        let h = self.config.n_head;
        let q = self.config.n_embd / self.config.n_head;
        let t = self.config.sequence_len;
        // Sum attention FLOPs per layer, accounting for sliding window
        let mut attn_flops = 0usize;
        for window_size in &self.window_sizes {
            let window = window_size.0; // (left, right) tuple, we use left
            let effective_seq = if window < 0 {
                t
            } else {
                (window as usize).min(t)
            };
            attn_flops += 12 * h * q * effective_seq;
        }
        6 * self.num_matmul_params() + attn_flops
    }

    /// The number of parameters that participate in matmuls with the token stream,
    /// i.e. contribute 2 FLOPs/param to the forward pass: every Linear layer
    /// (transformer matrices incl. the ve gates, lm_head, smear_gate). Embeddings
    /// are lookups and the per-layer scalars are not matmuls.
    pub fn num_matmul_params(&self) -> usize {
        let (_wte, _value_embeds, lm_head, transformer_matrices, _scalars) = self.param_counts();
        transformer_matrices + lm_head + self.smear_gate.weight().elem_count()
    }

    /// Left attention window of each layer, in tokens.
    fn windows(&self) -> impl Iterator<Item = usize> + '_ {
        let t = self.config.sequence_len;
        self.window_sizes
            .iter()
            .map(move |w| if w.0 < 0 { t } else { w.0 as usize })
    }

    /// Forward FLOPs to decode one token at a given context length during inference:
    /// 2 FLOPs per matmul param, plus attention over min(context, window) per layer.
    pub fn estimate_decode_flops(&self, context_len: usize) -> usize {
        let h = self.config.n_head;
        let q = self.config.n_embd / self.config.n_head;
        let attn_flops: usize = self
            .windows()
            .map(|window| 4 * h * q * context_len.min(window))
            .sum();
        2 * self.num_matmul_params() + attn_flops
    }

    /// Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window).
    pub fn estimate_prefill_flops(&self, num_tokens: usize) -> usize {
        let h = self.config.n_head;
        let q = self.config.n_embd / self.config.n_head;
        let mut attn_flops = 0usize;
        for window in self.windows() {
            let w = window.min(num_tokens);
            let attended_tokens = w * (w + 1) / 2 + (num_tokens - w) * w; // ramp up to w, then flat
            attn_flops += 4 * h * q * attended_tokens;
        }
        2 * self.num_matmul_params() * num_tokens + attn_flops
    }

    /// Bytes to *store* one token of KV cache during inference, per row (all layers).
    pub fn kv_bytes_per_token(&self) -> usize {
        let head_dim = self.config.n_embd / self.config.n_head;
        let kv_dtype_bytes = DType::F32.size_in_bytes(); // the KV cache is kept in the compute dtype
        self.config.n_layer * 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes
    }

    /// Bytes of KV cache *read* by one decode step at a given context length, per row.
    /// Sliding window layers only attend to (and read) the last `window` tokens.
    pub fn kv_read_bytes(&self, context_len: usize) -> usize {
        let head_dim = self.config.n_embd / self.config.n_head;
        let kv_dtype_bytes = DType::F32.size_in_bytes();
        self.windows()
            .map(|window| {
                2 * self.config.n_kv_head * head_dim * kv_dtype_bytes * context_len.min(window)
            })
            .sum()
    }

    /// Return detailed parameter counts for scaling law analysis.
    pub fn num_scaling_params(&self) -> std::collections::HashMap<&'static str, usize> {
        let (wte, value_embeds, lm_head, transformer_matrices, scalars) = self.param_counts();
        let total = wte + value_embeds + lm_head + transformer_matrices + scalars;
        std::collections::HashMap::from([
            ("wte", wte),
            ("value_embeds", value_embeds),
            ("lm_head", lm_head),
            ("transformer_matrices", transformer_matrices),
            ("scalars", scalars),
            ("total", total),
        ])
    }

    /// Group the parameters of a VarMap-backed model and build the combined
    /// MuonAdamW optimizer: AdamW for embeddings/lm_head/scalars, Muon for the
    /// transformer matrices (grouped by shape for stacking).
    pub fn setup_optimizer(
        &self,
        varmap: &candle_nn::VarMap,
        unembedding_lr: f64,
        embedding_lr: f64,
        matrix_lr: f64,
        weight_decay: f64,
        scalar_lr: f64,
    ) -> Result<super::optim::MuonAdamW> {
        use super::optim::ParamGroup;
        let model_dim = self.config.n_embd;
        let n_layer = self.config.n_layer;
        let vars = varmap.data().lock().unwrap();
        let get = |name: &str| -> Result<candle_core::Var> {
            vars.get(name)
                .cloned()
                .ok_or_else(|| candle_core::Error::Msg(format!("missing var: {name}")))
        };

        // Separate out all parameters into groups; matrix params keep module order
        let mut matrix_params = Vec::new();
        for i in 0..n_layer {
            let p = format!("transformer.h.{i}");
            matrix_params.push(get(&format!("{p}.attn.c_q.weight"))?);
            matrix_params.push(get(&format!("{p}.attn.c_k.weight"))?);
            matrix_params.push(get(&format!("{p}.attn.c_v.weight"))?);
            matrix_params.push(get(&format!("{p}.attn.c_proj.weight"))?);
            if has_ve(i, n_layer) {
                matrix_params.push(get(&format!("{p}.attn.ve_gate.weight"))?);
            }
            matrix_params.push(get(&format!("{p}.mlp.c_fc.weight"))?);
            matrix_params.push(get(&format!("{p}.mlp.c_proj.weight"))?);
        }
        let mut value_embeds_params = Vec::new();
        for i in 0..n_layer {
            if has_ve(i, n_layer) {
                value_embeds_params.push(get(&format!("value_embeds.{i}.weight"))?);
            }
        }
        let embedding_params = vec![get("transformer.wte.weight")?];
        let lm_head_params = vec![get("lm_head.weight")?];
        let resid_params = vec![get("resid_lambdas")?];
        let x0_params = vec![get("x0_lambdas")?];
        let smear_params = vec![
            get("smear_gate.weight")?,
            get("smear_lambda")?,
            get("backout_lambda")?,
        ];

        // Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        let dmodel_lr_scale = (model_dim as f64 / 768.0).powf(-0.5);
        print0(&format!(
            "Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6}"
        ));

        // Build param_groups with all required fields explicit
        let mut param_groups = vec![
            // AdamW groups (embeddings, lm_head, scalars)
            ParamGroup::adamw(
                lm_head_params,
                unembedding_lr * dmodel_lr_scale,
                (0.8, 0.96),
                1e-10,
                0.01,
            ),
            ParamGroup::adamw(
                embedding_params,
                embedding_lr * dmodel_lr_scale,
                (0.8, 0.995),
                1e-10,
                0.001,
            ),
            ParamGroup::adamw(
                value_embeds_params,
                embedding_lr * dmodel_lr_scale * 0.5,
                (0.8, 0.995),
                1e-10,
                0.01,
            ),
            ParamGroup::adamw(resid_params, scalar_lr * 0.01, (0.8, 0.95), 1e-10, 0.05),
            ParamGroup::adamw(x0_params, scalar_lr, (0.96, 0.95), 1e-10, 0.0), // higher beta1 for x0
            ParamGroup::adamw(smear_params, 0.2, (0.8, 0.95), 1e-10, 0.0),
        ];
        // Muon groups (matrix params, grouped by shape for stacking)
        let mut shapes: Vec<Vec<usize>> = matrix_params.iter().map(|p| p.dims().to_vec()).collect();
        shapes.sort();
        shapes.dedup();
        for shape in shapes {
            let group_params: Vec<_> = matrix_params
                .iter()
                .filter(|p| p.dims() == shape.as_slice())
                .cloned()
                .collect();
            param_groups.push(ParamGroup::muon(
                group_params,
                matrix_lr,
                0.95,
                5,
                0.9,
                weight_decay,
            ));
        }
        Ok(super::optim::MuonAdamW::new(param_groups))
    }

    /// Forward pass. With targets, returns the scalar cross-entropy loss;
    /// otherwise returns the logits (B, T, vocab_size).
    pub fn forward(
        &self,
        idx: &Tensor,
        targets: Option<&Tensor>,
        kv_cache: Option<&KVCache>,
        loss_reduction: &str,
    ) -> Result<Tensor> {
        let (_b, t) = idx.dims2()?;

        // Grab the rotary embeddings for the current sequence length (shape (1, seq_len, 1, head_dim/2))
        assert!(
            t <= self.cos.dim(1)?,
            "Sequence length grew beyond the rotary embeddings cache: {} > {}",
            t,
            self.cos.dim(1)?
        );
        // if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        let t0 = kv_cache.map_or(0, |kv| kv.get_pos());
        let cos = self.cos.narrow(1, t0, t)?; // truncate cache to current sequence length
        let sin = self.sin.narrow(1, t0, t)?;

        // Embed the tokens
        let x = self.wte.forward(idx)?; // embed current token
        let mut x = norm(&x)?;

        // Smear: mix previous token's embedding into current position (cheap bigram info)
        let smear = |x: &Tensor| -> Result<Tensor> {
            // gate = smear_lambda * sigmoid(smear_gate(x[:, 1:, :24])), applied to positions 1+
            let tail = x.narrow(1, 1, t - 1)?;
            let gate = sigmoid(
                &self
                    .smear_gate
                    .forward(&tail.narrow(2, 0, 24)?.contiguous()?)?,
            )?
            .broadcast_mul(&self.smear_lambda)?;
            let smeared = (&tail + gate.broadcast_mul(&x.narrow(1, 0, t - 1)?)?)?;
            Tensor::cat(&[x.narrow(1, 0, 1)?, smeared], 1)
        };
        match kv_cache {
            None => {
                // Training / naive generate: full sequence available, use fast slice
                assert!(t > 1, "Training forward pass should have T > 1");
                x = smear(&x)?;
            }
            Some(kv) => {
                // KV cache inference: read prev embedding from cache, store current for next step
                let x_pre_smear = kv.prev_embedding.borrow().clone();
                *kv.prev_embedding.borrow_mut() = Some(x.narrow(1, t - 1, 1)?);
                if t > 1 {
                    // Prefill: apply smear to positions 1+, same as training
                    x = smear(&x)?;
                } else if let Some(x_pre_smear) = x_pre_smear {
                    // Decode: single token, use cached prev embedding
                    let gate = sigmoid(
                        &self
                            .smear_gate
                            .forward(&x.narrow(2, 0, 24)?.contiguous()?)?,
                    )?
                    .broadcast_mul(&self.smear_lambda)?;
                    x = (&x + gate.broadcast_mul(&x_pre_smear)?)?;
                }
            }
        }

        // Forward the trunk of the Transformer
        let x0 = x.clone(); // save initial normalized embedding for x0 residual
        let n_layer = self.config.n_layer;
        let backout_layer = n_layer / 2; // cache at halfway point
        let mut x_backout = None;
        for (i, block) in self.h.iter().enumerate() {
            x = (x.broadcast_mul(&self.resid_lambdas.get(i)?)?
                + x0.broadcast_mul(&self.x0_lambdas.get(i)?)?)?;
            let ve = match &self.value_embeds[i] {
                Some(ve) => Some(ve.forward(idx)?),
                None => None,
            };
            x = block.forward(
                &x,
                ve.as_ref(),
                (&cos, &sin),
                self.window_sizes[i],
                kv_cache,
            )?;
            if i == backout_layer {
                x_backout = Some(x.clone());
            }
        }
        // Subtract mid-layer residual to remove low-level features before logit projection
        if let Some(x_backout) = x_backout {
            x = (x - x_backout.broadcast_mul(&self.backout_lambda)?)?;
        }
        x = norm(&x)?;

        // Forward the lm_head (compute logits)
        let softcap = 15f64; // smoothly cap the logits to the range [-softcap, softcap]
        let logits = self.lm_head.forward(&x)?; // (B, T, padded_vocab_size)
        let logits = logits.narrow(D::Minus1, 0, self.config.vocab_size)?; // slice to remove padding
        let logits = logits.to_dtype(DType::F32)?; // fp32 for logit softcap and loss computation
        let logits = ((logits / softcap)?.tanh()? * softcap)?; // squash the logits

        match targets {
            Some(targets) => {
                // training: given the targets, compute and return the loss
                cross_entropy(&logits, targets, -1, loss_reduction)
            }
            None => {
                // inference: just return the logits directly
                Ok(logits)
            }
        }
    }

    /// Naive autoregressive streaming inference (batch size 1, plain token lists).
    /// Calls yield_token with each sampled token; generation stops early if it returns false.
    pub fn generate(
        &self,
        tokens: &[u32],
        max_tokens: usize,
        temperature: f64,
        top_k: Option<usize>,
        seed: u64,
        mut yield_token: impl FnMut(u32) -> bool,
    ) -> Result<()> {
        let device = self.get_device();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut ids: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
        for _ in 0..max_tokens {
            let idx = Tensor::from_slice(&ids, (1, ids.len()), device)?; // add batch dim
            let logits = self.forward(&idx, None, None, "mean")?; // (B, T, vocab_size)
            let logits = logits.i((0, ids.len() - 1))?; // (vocab_size,)
            let mut logits: Vec<f32> = logits.to_vec1()?;
            if let Some(top_k) = top_k {
                if top_k > 0 {
                    // mask out everything below the k-th largest logit
                    let k = top_k.min(logits.len());
                    let mut sorted = logits.clone();
                    let (_, kth, _) = sorted.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
                    let kth = *kth;
                    for v in logits.iter_mut() {
                        if *v < kth {
                            *v = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            let token = if temperature > 0.0 {
                let scaled: Vec<f32> = logits.iter().map(|v| v / temperature as f32).collect();
                multinomial(&softmax_1d(&scaled), &mut rng)
            } else {
                argmax_1d(&logits)
            };
            ids.push(token as i64);
            if !yield_token(token) {
                break;
            }
        }
        Ok(())
    }
}

/// Cross-entropy loss over flattened logits, matching
/// F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index, reduction).
pub fn cross_entropy(
    logits: &Tensor,
    targets: &Tensor,
    ignore_index: i64,
    reduction: &str,
) -> Result<Tensor> {
    let vocab_size = logits.dim(D::Minus1)?;
    let logits = logits.reshape(((), vocab_size))?;
    let targets = targets.reshape(((),))?.to_dtype(DType::I64)?;
    let log_probs = log_softmax(&logits, D::Minus1)?;
    let valid = targets.ne(ignore_index)?.to_dtype(DType::F32)?;
    // clamp ignored targets to a valid row for the gather, then mask them out
    let gather_idx = targets.clamp(0i64, (vocab_size - 1) as i64)?.unsqueeze(1)?;
    let picked = log_probs.gather(&gather_idx, 1)?.squeeze(1)?;
    if reduction == "none" {
        // per-position loss, 0 at ignored targets, flattened like the reference
        return (picked * &valid)?.neg();
    }
    let loss_sum = (picked * &valid)?.sum_all()?.neg()?;
    match reduction {
        "sum" => Ok(loss_sum),
        "mean" => loss_sum.broadcast_div(&valid.sum_all()?),
        other => candle_core::bail!("unsupported loss reduction: {other}"),
    }
}

pub fn argmax_1d(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

pub fn softmax_1d(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|e| e / sum).collect()
}

pub fn multinomial(probs: &[f32], rng: &mut StdRng) -> u32 {
    let r: f32 = rng.random();
    let mut cdf = 0f32;
    for (i, &p) in probs.iter().enumerate() {
        cdf += p;
        if r < cdf {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}
