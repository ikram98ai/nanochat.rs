use candle_core::{Tensor, Result, DType, D, Device};
use candle_nn::{Linear, Embedding, Module, VarBuilder, embedding};
use candle_nn::ops::{sigmoid};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct GPTConfig {
    pub sequence_len: usize,
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub n_embd: usize,
    pub window_pattern: String,
}

fn norm(x: &Tensor) -> Result<Tensor> {
    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d
    };
    let x_f32 = x.to_dtype(internal_dtype)?;
    let hidden_size = x_f32.dim(D::Minus1)?;
    let norm_x = (x_f32.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let x_normed = x_f32.broadcast_div(&(norm_x + 1e-6)?.sqrt()?)?;
    x_normed.to_dtype(x_dtype)
}

fn has_ve(layer_idx: usize, n_layer: usize) -> bool {
    layer_idx % 2 == (n_layer - 1) % 2
}

fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _t, _h, d) = x.dims4()?;
    let x1 = x.narrow(D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
    let y1 = (x1.broadcast_mul(cos)? + x2.broadcast_mul(sin)?)?;
    let y2 = (x1.broadcast_mul(&sin.neg()?)? + x2.broadcast_mul(cos)?)?;
    Tensor::cat(&[y1, y2], D::Minus1)
}

pub struct CausalSelfAttention {
    pub(crate) n_head: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
    pub(crate) c_q: Linear,
    pub(crate) c_k: Linear,
    pub(crate) c_v: Linear,
    pub(crate) c_proj: Linear,
    pub(crate) ve_gate_channels: usize,
    pub(crate) ve_gate: Option<Linear>,
}

impl CausalSelfAttention {
    fn new(config: &GPTConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let n_head = config.n_head;
        let n_kv_head = config.n_kv_head;
        let n_embd = config.n_embd;
        let head_dim = n_embd / n_head;

        let c_q = candle_nn::linear_no_bias(n_embd, n_head * head_dim, vb.pp("c_q"))?;
        let c_k = candle_nn::linear_no_bias(n_embd, n_kv_head * head_dim, vb.pp("c_k"))?;
        let c_v = candle_nn::linear_no_bias(n_embd, n_kv_head * head_dim, vb.pp("c_v"))?;
        let c_proj = candle_nn::linear_no_bias(n_embd, n_embd, vb.pp("c_proj"))?;

        let ve_gate_channels = 32;
        let ve_gate = if has_ve(layer_idx, config.n_layer) {
            Some(candle_nn::linear_no_bias(ve_gate_channels, n_kv_head, vb.pp("ve_gate"))?)
        } else {
            None
        };

        Ok(Self {
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
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: Option<&mut super::flash_attention::KVCacheEntry>,
        window_size: (usize, usize),
    ) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let q = self.c_q.forward(x)?.reshape((b, t, self.n_head, self.head_dim))?;
        let k = self.c_k.forward(x)?.reshape((b, t, self.n_kv_head, self.head_dim))?;
        let mut v = self.c_v.forward(x)?.reshape((b, t, self.n_kv_head, self.head_dim))?;

        if let (Some(ve), Some(ref gate_lin)) = (ve, &self.ve_gate) {
            let ve = ve.reshape((b, t, self.n_kv_head, self.head_dim))?;
            let gate = (sigmoid(&gate_lin.forward(&x.narrow(2, 0, self.ve_gate_channels)?)?)? * 2.0)?;
            v = (v + gate.unsqueeze(D::Minus1)?.broadcast_mul(&ve)?)?;
        }

        let q = apply_rotary_emb(&q, cos, sin)?;
        let k = apply_rotary_emb(&k, cos, sin)?;

        let q = norm(&q)?;
        let k = norm(&k)?;

        let y = match kv_cache {
            Some(entry) => {
                let cache_len = entry.k.dim(1)?;
                super::flash_attention::flash_attn_with_kvcache(
                    &q, &mut entry.k, &mut entry.v, &k, &v, cache_len, None, true, Some(window_size)
                )?
            }
            None => {
                super::flash_attention::flash_attn_func(&q, &k, &v, None, true, Some(window_size))?
            }
        };

        self.c_proj.forward(&y.reshape((b, t, c))?)
    }
}

pub(crate) struct MLP {
    c_fc: Linear,
    c_proj: Linear,
}

impl MLP {
    fn new(config: &GPTConfig, vb: VarBuilder) -> Result<Self> {
        let n_embd = config.n_embd;
        let n_inner = n_embd * 4;
        let c_fc = candle_nn::linear_no_bias(n_embd, n_inner, vb.pp("c_fc"))?;
        let c_proj = candle_nn::linear_no_bias(n_inner, n_embd, vb.pp("c_proj"))?;
        Ok(Self { c_fc, c_proj })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.c_fc.forward(x)?;
        let x = x.relu()?.sqr()?;
        self.c_proj.forward(&x)
    }
}

pub struct Block {
    pub(crate) attn: CausalSelfAttention,
    pub(crate) mlp: MLP,
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
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: Option<&mut super::flash_attention::KVCacheEntry>,
        window_size: (usize, usize),
    ) -> Result<Tensor> {
        let resid = self.attn.forward(&norm(x)?, ve, cos, sin, kv_cache, window_size)?;
        let x = (x + resid)?;
        let resid = self.mlp.forward(&norm(&x)?)?;
        x + resid
    }
}

pub struct GPT {
    pub(crate) config: GPTConfig,
    pub(crate) wte: Embedding,
    pub(crate) h: Vec<Block>,
    pub(crate) lm_head: Linear,
    pub(crate) resid_lambdas: Tensor,
    pub(crate) x0_lambdas: Tensor,
    pub(crate) value_embeds: Vec<Option<Embedding>>,
    pub(crate) cos: Tensor,
    pub(crate) sin: Tensor,
    pub(crate) window_sizes: Vec<(usize, usize)>,
}

impl GPT {
    pub fn new(config: GPTConfig, vb: VarBuilder) -> Result<Self> {
        let vb_t = vb.pp("transformer");
        let wte = embedding(config.vocab_size, config.n_embd, vb_t.pp("wte"))?;
        let mut h = Vec::new();
        for i in 0..config.n_layer {
            h.push(Block::new(&config, i, vb_t.pp("h").pp(i))?);
        }
        let lm_head = candle_nn::linear_no_bias(config.n_embd, config.vocab_size, vb.pp("lm_head"))?;
        
        let resid_lambdas = vb.get(config.n_layer, "resid_lambdas")?;
        let x0_lambdas = vb.get(config.n_layer, "x0_lambdas")?;
        
        let mut value_embeds = Vec::new();
        for i in 0..config.n_layer {
            if has_ve(i, config.n_layer) {
                value_embeds.push(Some(embedding(config.vocab_size, config.n_embd, vb.pp("value_embeds").pp(i))?));
            } else {
                value_embeds.push(None);
            }
        }

        let head_dim = config.n_embd / config.n_head;
        let (cos, sin) = precompute_rotary_embeddings(config.sequence_len, head_dim, vb.device())?;
        let window_sizes = compute_window_sizes(&config);

        Ok(Self {
            config,
            wte,
            h,
            lm_head,
            resid_lambdas,
            x0_lambdas,
            value_embeds,
            cos,
            sin,
            window_sizes,
        })
    }

    pub fn device(&self) -> &Device {
        self.wte.embeddings().device()
    }

    pub fn forward(
        &self,
        idx: &Tensor,
        mut kv_cache: Option<&mut Vec<super::flash_attention::KVCacheEntry>>,
    ) -> Result<Tensor> {
        let (_b, t) = idx.dims2()?;
        let mut x = self.wte.forward(idx)?;
        x = norm(&x)?;
        let x0 = x.clone();

        let cos = self.cos.narrow(1, 0, t)?;
        let sin = self.sin.narrow(1, 0, t)?;

        for i in 0..self.config.n_layer {
            let resid_l = self.resid_lambdas.get(i)?;
            let x0_l = self.x0_lambdas.get(i)?;
            x = (x.broadcast_mul(&resid_l)? + x0.broadcast_mul(&x0_l)?)?;

            let ve = if let Some(ref ve_emb) = self.value_embeds[i] {
                Some(ve_emb.forward(idx)?)
            } else {
                None
            };

            let layer_kv = kv_cache.as_mut().map(|c| &mut c[i]);
            x = self.h[i].forward(&x, ve.as_ref(), &cos, &sin, layer_kv, self.window_sizes[i])?;
        }

        x = norm(&x)?;
        let logits = self.lm_head.forward(&x)?;
        Ok(logits)
    }

    pub fn num_scaling_params(&self) -> usize {
        let mut n = 0;
        n += self.resid_lambdas.elem_count();
        n += self.x0_lambdas.elem_count();
        n
    }

    pub fn estimate_flops(&self, seq_len: usize) -> f64 {
        let n_params = self.config.n_layer * self.config.n_embd * self.config.n_embd * 12;
        6.0 * (n_params as f64) * (seq_len as f64)
    }
}

fn precompute_rotary_embeddings(seq_len: usize, head_dim: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let inv_freq: Vec<_> = (0..head_dim).step_by(2)
        .map(|i| 1.0 / 10000.0f32.powf(i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (head_dim / 2,), device)?;
    let t = Tensor::arange(0u32, seq_len as u32, device)?.to_dtype(DType::F32)?;
    let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
    let cos = freqs.cos()?.reshape((1, seq_len, 1, head_dim / 2))?;
    let sin = freqs.sin()?.reshape((1, seq_len, 1, head_dim / 2))?;
    Ok((cos, sin))
}

fn compute_window_sizes(config: &GPTConfig) -> Vec<(usize, usize)> {
    let mut sizes = Vec::new();
    let pattern = config.window_pattern.as_str();
    for i in 0..config.n_layer {
        let char = pattern.chars().nth(i % pattern.len()).unwrap_or('L');
        let size = if char == 'S' { config.sequence_len / 2 } else { config.sequence_len };
        sizes.push((size, 0));
    }
    if let Some(last) = sizes.last_mut() {
        *last = (config.sequence_len, 0);
    }
    sizes
}
