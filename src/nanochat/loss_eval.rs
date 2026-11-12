//! A number of functions that help with evaluating a base model.

use candle_core::{DType, Device, Result, Tensor};

use super::gpt::GPT;

/// Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
/// which is a tokenization vocab size-independent metric: the sum loss is divided
/// by the total byte length of the target tokens.
///
/// 1) All "normal" tokens are normalized by the length of the token in bytes
/// 2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out
/// 3) No actively masked tokens (ignore_index of e.g. -1) are included in the metric
///
/// token_bytes is a 1D tensor of shape (vocab_size,) with the byte length of each
/// token id, or 0 if the token is to not be counted (e.g. special tokens).
pub fn evaluate_bpb(
    model: &GPT,
    batches: &mut dyn Iterator<Item = Result<(Tensor, Tensor)>>,
    steps: usize,
    token_bytes: &Tensor,
) -> Result<f64> {
    // byte length of each token id, pulled to the host once
    let token_bytes: Vec<i64> = token_bytes
        .to_device(&Device::Cpu)?
        .to_dtype(DType::I64)?
        .to_vec1()?;
    let mut total_nats = 0f64;
    let mut total_bytes = 0i64;
    for _ in 0..steps {
        let (x, y) = batches
            .next()
            .ok_or_else(|| candle_core::Error::Msg("batch iterator exhausted".into()))??;
        let loss2d = model.forward(&x, Some(&y), None, "none")?; // flat (B*T,)
        let y = y.flatten_all()?.to_dtype(DType::I64)?.to_vec1::<i64>()?;
        // map valid targets to their byte length; any target token < 0 is to be
        // ignored (do NOT index token_bytes with negatives) and contributes 0 bytes
        let num_bytes: Vec<i64> = y
            .iter()
            .map(|&t| if t < 0 { 0 } else { token_bytes[t as usize] })
            .collect();
        // tokens with 0 bytes (special or ignored) contribute zero loss
        let byte_mask: Vec<f32> = num_bytes.iter().map(|&b| (b > 0) as u8 as f32).collect();
        let byte_mask = Tensor::from_vec(byte_mask, num_bytes.len(), loss2d.device())?;
        total_nats += loss2d.mul(&byte_mask)?.sum_all()?.to_scalar::<f32>()? as f64;
        total_bytes += num_bytes.iter().sum::<i64>();
    }
    if total_bytes == 0 {
        return Ok(f64::INFINITY);
    }
    Ok(total_nats / (std::f64::consts::LN_2 * total_bytes as f64))
}
