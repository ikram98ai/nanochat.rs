//! Unified attention interface mirroring the FA3 API, implemented with a
//! scaled-dot-product-attention fallback (the CPU code path of the Python
//! version). Tensors are (B, T, H, D); window_size is a (left, right) tuple
//! where -1 means unlimited.

use candle_core::{DType, Result, Tensor, D};
use candle_nn::ops::softmax;

use super::engine::KVCache;

/// Scaled dot product attention with a bool mask (true = keep) and GQA support.
/// q, k, v are (B, H, T, D) format.
fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attn_mask: Option<&Tensor>,
    is_causal: bool,
    enable_gqa: bool,
) -> Result<Tensor> {
    let (b, h_q, t_q, d) = q.dims4()?;
    let (_, h_kv, t_k, _) = k.dims4()?;

    // GQA: repeat each kv head to match the number of query heads
    let (k, v) = if enable_gqa {
        let n_rep = h_q / h_kv;
        let repeat = |t: &Tensor| -> Result<Tensor> {
            t.unsqueeze(2)?
                .expand((b, h_kv, n_rep, t_k, d))?
                .reshape((b, h_q, t_k, d))
        };
        (repeat(k)?, repeat(v)?)
    } else {
        (k.clone(), v.clone())
    };

    let scale = 1.0 / (d as f64).sqrt();
    let mut scores = (q.contiguous()?.matmul(&k.contiguous()?.t()?)? * scale)?;

    let mask = if let Some(m) = attn_mask {
        Some(m.clone())
    } else if is_causal {
        // standard causal mask: row i attends to columns j <= i
        let row_idx = Tensor::arange(0u32, t_q as u32, q.device())?.unsqueeze(1)?;
        let col_idx = Tensor::arange(0u32, t_k as u32, q.device())?.unsqueeze(0)?;
        Some(col_idx.broadcast_le(&row_idx)?)
    } else {
        None
    };
    if let Some(mask) = mask {
        // additive (Tq, Tk) mask broadcast over batch and heads: 0 keeps, -inf drops
        let zeros = Tensor::zeros((t_q, t_k), scores.dtype(), scores.device())?;
        let neg_inf = Tensor::full(f32::NEG_INFINITY, (t_q, t_k), scores.device())?
            .to_dtype(scores.dtype())?;
        let additive = mask.where_cond(&zeros, &neg_inf)?;
        scores = scores.broadcast_add(&additive)?;
    }

    let weights = softmax(&scores, D::Minus1)?;
    weights.matmul(&v.contiguous()?)
}

/// SDPA attention with sliding window support. q, k, v are (B, H, T, D).
fn sdpa_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    window_size: (i64, i64),
    enable_gqa: bool,
) -> Result<Tensor> {
    let t_q = q.dim(2)?;
    let t_k = k.dim(2)?;
    let window = window_size.0;

    // Full context, same length
    if (window < 0 || window >= t_q as i64) && t_q == t_k {
        return scaled_dot_product_attention(q, k, v, None, true, enable_gqa);
    }

    // Single token generation
    if t_q == 1 {
        let (k, v) = if window >= 0 && (window as usize) < t_k {
            // window is "left" tokens: we need to include (window + 1) keys total
            let start = t_k.saturating_sub(window as usize + 1);
            (
                k.narrow(2, start, t_k - start)?,
                v.narrow(2, start, t_k - start)?,
            )
        } else {
            (k.clone(), v.clone())
        };
        return scaled_dot_product_attention(q, &k, &v, None, false, enable_gqa);
    }

    // Need explicit mask for sliding window/chunk inference
    // For chunk inference (Tq != Tk), is_causal is not aligned to cache position
    // => build an explicit bool mask
    let device = q.device();
    let row_idx = Tensor::arange((t_k - t_q) as i64, t_k as i64, device)?.unsqueeze(1)?;
    let col_idx = Tensor::arange(0i64, t_k as i64, device)?.unsqueeze(0)?;
    let mut mask = col_idx.broadcast_le(&row_idx)?;

    // sliding window (left)
    if window >= 0 && (window as usize) < t_k {
        let dist = row_idx.broadcast_sub(&col_idx)?;
        let window_t = Tensor::full(window, dist.shape(), device)?;
        mask = mask.mul(&dist.le(&window_t)?.to_dtype(DType::U8)?)?;
    }

    scaled_dot_product_attention(q, k, v, Some(&mask), false, enable_gqa)
}

/// Flash Attention for training (no KV cache).
///
/// q, k, v: (B, T, H, D); window_size: (left, right) sliding window, -1 unlimited.
/// Returns (B, T, H, D).
pub fn flash_attn_func(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    _causal: bool,
    window_size: (i64, i64),
) -> Result<Tensor> {
    // SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    let q = q.transpose(1, 2)?;
    let k = k.transpose(1, 2)?;
    let v = v.transpose(1, 2)?;
    let enable_gqa = q.dim(1)? != k.dim(1)?;
    let y = sdpa_attention(&q, &k, &v, window_size, enable_gqa)?;
    y.transpose(1, 2) // back to (B, T, H, D)
}

/// Flash Attention with KV cache for inference. Updates k_cache/v_cache of
/// the given layer in-place, matching the FA3 behavior.
///
/// q: (B, T_new, H, D); k, v: new keys/values (B, T_new, H_kv, D).
/// The current position is read from kv_cache.cache_seqlens.
pub fn flash_attn_with_kvcache(
    q: &Tensor,
    kv_cache: &KVCache,
    layer_idx: usize,
    k: &Tensor,
    v: &Tensor,
    _causal: bool,
    window_size: (i64, i64),
) -> Result<Tensor> {
    let t_new = q.dim(1)?;
    let pos = kv_cache.get_pos(); // assume uniform position across batch
    let (k_cache, v_cache) = kv_cache.get_layer_cache(layer_idx);

    // Insert new k, v into cache (in-place, matching FA3 behavior)
    k_cache.slice_set(k, 1, pos)?;
    v_cache.slice_set(v, 1, pos)?;

    // Get full cache up to current position + new tokens
    let end_pos = pos + t_new;
    let k_full = k_cache.narrow(1, 0, end_pos)?;
    let v_full = v_cache.narrow(1, 0, end_pos)?;

    // Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    let q_sdpa = q.transpose(1, 2)?;
    let k_sdpa = k_full.transpose(1, 2)?;
    let v_sdpa = v_full.transpose(1, 2)?;

    let enable_gqa = q_sdpa.dim(1)? != k_sdpa.dim(1)?;
    let y_sdpa = sdpa_attention(&q_sdpa, &k_sdpa, &v_sdpa, window_size, enable_gqa)?;

    y_sdpa.transpose(1, 2) // back to (B, T_new, H, D)
}
