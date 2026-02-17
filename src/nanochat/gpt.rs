use candle_core::{Tensor, Result, DType, D};
use candle_nn::{Linear, Embedding, Module, VarBuilder, embedding, linear_b as linear, cpu_flash_attention};
use candle_nn::ops::{sigmoid};

#[derive(serde::Deserialize, Debug, Clone)]
struct GPTConfig{
    sequence_len: usize,
    vocab_size: usize,
    n_layer: usize,
    n_head: usize,
    n_kv_head: usize,
    n_embd: usize,
    // Sliding window attention pattern string, tiled across layers. Final layer always L.
    // Characters: L=long (full context), S=short (half context)
    // Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattren: String
}

fn norm(x: &Tensor) -> Result<Tensor> {
    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d
    };
    let x = x.to_dtype(internal_dtype)?;
    let hidden_size = x.dim(D::Minus1)?;
    let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let x_normed = x.broadcast_div(&(norm_x + 1e-6)?.sqrt()?)?;

    Ok(x_normed)
}

fn has_ve(layer_idx: usize, n_layer: usize) -> bool {
        layer_idx % 2 == (n_layer - 1) % 2
}

fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    assert_eq!(x.dims().len(), 4);
    let d = x.dims()[3];
    let x1 = x.narrow(3, 0, d/2)?;
    let x2 = x.narrow(3, d/2, d/2)?;
    let y1 = x1.broadcast_mul(cos).unwrap() + x2.broadcast_mul(sin).unwrap();
    let y2 = x1.broadcast_mul(&(sin.neg()?)).unwrap() + x2.broadcast_mul(cos).unwrap();
    Tensor::cat(&[y1.unwrap(), y2.unwrap()], 3)
}

struct CausalSelfAttention{
    layer_idx: usize,
    n_head: usize,
    n_kv_head: usize,
    n_embd: usize,
    head_dim: usize,
    c_q: Linear,
    c_k: Linear,
    c_v: Linear,
    c_proj: Linear,
    ve_gate_channels: usize,
    ve_gate: Option<Linear>
}

impl CausalSelfAttention{
    fn new(config: &GPTConfig, layer_idx: usize, vb: VarBuilder)-> Result<Self> {
        let layer_idx = layer_idx;
        let n_head = config.n_head;
        let n_kv_head = config.n_kv_head;
        let n_embd = config.n_embd;
        let head_dim = n_embd / (n_head as usize);
        
        assert_eq!(n_embd % n_head as usize, 0);
        assert!(n_kv_head <= n_head && n_head % n_kv_head == 0);
        
        let c_q = linear(n_embd, n_head as usize * head_dim as usize, false, vb.pp("c_q"))?;
        let c_k = linear(n_embd, n_kv_head as usize * head_dim as usize, false, vb.pp("c_k"))?;
        let c_v = linear(n_embd, n_kv_head as usize * head_dim as usize, false, vb.pp("c_v"))?;

        let c_proj = linear(n_embd, n_embd, false, vb.pp("c_proj"))?;

        let ve_gate_channels = 32;
        let ve_gate = if has_ve(layer_idx, config.n_layer as usize) {
            Some(linear(ve_gate_channels, n_kv_head as usize, false, vb.pp("ve_gate"))?)
        } else {
            None
        };
        Ok(CausalSelfAttention{
            layer_idx,
            n_head,
            n_kv_head,
            n_embd,
            head_dim,
            c_q,
            c_k,
            c_v,
            c_proj,
            ve_gate_channels,
            ve_gate
        })
    }

    fn forward(&mut self, x: &Tensor, ve: Option<&Tensor>, cos_sin: (&Tensor, &Tensor)) -> Result<Tensor> {
        let (B, T, C) = x.dims3()?;

        // Project the input to get queries, keys, and values
        // Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
       
        let q = self.c_q.forward(x)?.reshape((B, T, self.n_head, self.head_dim))?;
        let k = self.c_k.forward(x)?.reshape((B, T, self.n_kv_head, self.head_dim))?;
        let mut v = self.c_v.forward(x)?.reshape((B, T, self.n_kv_head, self.head_dim))?;

        // Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        v = if let Some(ve) = ve {
            let ve = ve.reshape((B, T, self.n_kv_head, self.head_dim))?;
            let gate = 2.0 * sigmoid(&self.ve_gate.as_ref().unwrap().forward(&x.narrow(2, 0, self.ve_gate_channels)?)?)?;
            (v + gate.unwrap().unsqueeze(D::Minus1)?.broadcast_mul(&ve)?)?
        }else{
            v
        };

        let (cos, sin) = cos_sin;

        // Apply Rotary Embeddings to queries and keys to get relative positional encoding
        let q = apply_rotary_emb(&q, cos, sin)?;
        let k = apply_rotary_emb(&k, cos, sin)?;
        let q = norm(&q)?;
        let k = norm(&k)?;

        let scale = 1f32 / (self.head_dim as f32).sqrt();
        let y =  cpu_flash_attention::run_flash_attn_cpu::<f32>(&q , &k, &v, None , scale, None, None)?;
        

        // Re-assemble the heads and project back to residual stream
        y.contiguous()?.reshape((B,T,()))?.apply(&self.c_proj)
    }

}

struct MLP{
    c_fc: Linear,
    c_proj: Linear
}

impl MLP{
    fn new(config: &GPTConfig, vb: VarBuilder)-> Result<Self> {
        let n_embd = config.n_embd;
        let n_inner = config.n_embd*4;

        let c_fc = linear(n_embd, n_inner, false, vb.pp("c_fc"))?;
        let c_proj = linear(n_inner, n_embd, false, vb.pp("c_proj"))?;
        Ok(MLP{
            c_fc,
            c_proj
        })
    }

    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        self.c_fc.forward(x)?.relu()?.sqr()?.apply(&self.c_proj)
    }
}

struct Block{
    attn: CausalSelfAttention,
    mlp: MLP
}

impl Block{
    fn new(config: &GPTConfig, layer_idx: usize,vb: VarBuilder) -> Result<Self> {
        let attn = CausalSelfAttention::new(config, layer_idx, vb.pp("attn"))?;
        let mlp = MLP::new(config, vb.pp("mlp"))?;
        Ok(Block { attn, mlp })
    }

    fn forward(&mut self, x: &Tensor, ve: Option<&Tensor>, cos_sin: (&Tensor, &Tensor)) -> Result<Tensor> {
        let x = self.attn.forward(&norm(&x)?, ve, cos_sin)?;
        self.mlp.forward(&norm(&x)?)
    }
}

struct GPT{
    config:GPTConfig,
    lm_head:Linear,
    wte:Embedding,
    h: Vec<Block>
}

impl GPT {
    fn new(config: GPTConfig, vb: VarBuilder) -> Result<Self> {
        
        // self.transformer = nn.ModuleDict({
        //     "wte": nn.Embedding(padded_vocab_size, config.n_embd),
        //     "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        // })

        let vb_tranformer = vb.pp("transformer");
        let wte = embedding(config.vocab_size, config.n_embd, vb_tranformer.pp("wte"))?;

        let mut layers = Vec::with_capacity(config.n_layer);
        let vb_h = vb_tranformer.pp("h");

        for layer_idx in 0..config.n_layer {
            layers.push(Block::new(&config, layer_idx, vb_h.pp(&format!("{}_", layer_idx)))?);
        }
        
        let lm_head = linear(config.n_embd, config.vocab_size, false, vb.pp("lm_head"))?;

        Ok(GPT {
            config,
            lm_head,
            wte,
            h: layers
        })
    }
}