use candle_core::{Tensor, Result, DType};
use candle_nn::ops;

/// Flash attention function using SDPA as backend.
pub fn flash_attn_func(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: Option<f32>,
    causal: bool,
    _window_size: Option<(usize, usize)>,
) -> Result<Tensor> {
    let head_dim = q.dim(candle_core::D::Minus1)?;
    let scale = softmax_scale.unwrap_or(1.0 / (head_dim as f32).sqrt());
    
    let q_t = q.transpose(1, 2)?;
    let k_t = k.transpose(1, 2)?;
    let v_t = v.transpose(1, 2)?;

    let mask = if causal {
        let seq_len = q.dim(1)?;
        let arange = Tensor::arange(0u32, seq_len as u32, q.device())?;
        let mask = arange.unsqueeze(0)?.broadcast_as((seq_len, seq_len))?
            .gt(&arange.unsqueeze(1)?.broadcast_as((seq_len, seq_len))?)?
            .to_dtype(DType::U8)?;
        Some(mask)
    } else {
        None
    };

    let att = ops::sdpa(&q_t, &k_t, &v_t, mask.as_ref(), causal, scale, 0.0)?;
    att.transpose(1, 2)
}

/// KV cache entry for a single layer
#[derive(Clone)]
pub struct KVCacheEntry {
    pub k: Tensor,
    pub v: Tensor,
}

/// Flash attention with KV cache for inference.
pub fn flash_attn_with_kvcache(
    q: &Tensor,
    k_cache: &mut Tensor,
    v_cache: &mut Tensor,
    k_new: &Tensor,
    v_new: &Tensor,
    cache_seqlen: usize,
    softmax_scale: Option<f32>,
    causal: bool,
    _window_size: Option<(usize, usize)>,
) -> Result<Tensor> {
    let head_dim = q.dim(candle_core::D::Minus1)?;
    let scale = softmax_scale.unwrap_or(1.0 / (head_dim as f32).sqrt());
    let new_seqlen = k_new.dim(1)?;

    if new_seqlen > 0 {
        k_cache.slice_set(k_new, 1, cache_seqlen)?;
        v_cache.slice_set(v_new, 1, cache_seqlen)?;
    }

    let total_seqlen = cache_seqlen + new_seqlen;
    let k_used = k_cache.narrow(1, 0, total_seqlen)?;
    let v_used = v_cache.narrow(1, 0, total_seqlen)?;

    let q_t = q.transpose(1, 2)?;
    let k_t = k_used.transpose(1, 2)?;
    let v_t = v_used.transpose(1, 2)?;

    let n_heads_q = q_t.dim(1)?;
    let n_heads_kv = k_t.dim(1)?;
    let (k_t, v_t) = if n_heads_q != n_heads_kv {
        let n_rep = n_heads_q / n_heads_kv;
        let k_rep = k_t.unsqueeze(2)?.expand(&[k_t.dim(0)?, n_heads_kv, n_rep, k_t.dim(2)?, k_t.dim(3)?])?
            .reshape(&[k_t.dim(0)?, n_heads_q, k_t.dim(2)?, k_t.dim(3)?])?;
        let v_rep = v_t.unsqueeze(2)?.expand(&[v_t.dim(0)?, n_heads_kv, n_rep, v_t.dim(2)?, v_t.dim(3)?])?
            .reshape(&[v_t.dim(0)?, n_heads_q, v_t.dim(2)?, v_t.dim(3)?])?;
        (k_rep, v_rep)
    } else {
        (k_t, v_t)
    };

    let att = ops::sdpa(&q_t, &k_t, &v_t, None, causal, scale, 0.0)?;
    att.transpose(1, 2)
}
