//! Functions for evaluating the CORE metric, as described in the DCLM paper.
//! https://arxiv.org/abs/2406.11794

use candle_core::{D, IndexOp, Result, Tensor};
use serde_json::Value;

use super::gpt::{cross_entropy, GPT};
use super::pyrandom::PyRandom;
use super::tokenizer::Tokenizer;

// -----------------------------------------------------------------------------
// Prompt rendering utilities (each fewshot example followed by a blank line,
// then the final item; one prompt per option)

fn s(v: &Value) -> &str {
    v.as_str().unwrap_or_default()
}

/// Render complete prompts for a multiple choice question
pub fn render_prompts_mc(item: &Value, continuation_delimiter: &str, fewshot_examples: &[Value]) -> Vec<String> {
    let mut prefix = String::new();
    for example in fewshot_examples {
        let gold = example["gold"].as_u64().unwrap() as usize;
        prefix.push_str(&format!(
            "{}{}{}\n\n",
            s(&example["query"]),
            continuation_delimiter,
            s(&example["choices"][gold])
        ));
    }
    item["choices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|choice| format!("{prefix}{}{}{}", s(&item["query"]), continuation_delimiter, s(choice)))
        .collect()
}

/// Render complete prompts for a schema question
pub fn render_prompts_schema(item: &Value, continuation_delimiter: &str, fewshot_examples: &[Value]) -> Vec<String> {
    let mut prefix = String::new();
    for example in fewshot_examples {
        let gold = example["gold"].as_u64().unwrap() as usize;
        prefix.push_str(&format!(
            "{}{}{}\n\n",
            s(&example["context_options"][gold]),
            continuation_delimiter,
            s(&example["continuation"])
        ));
    }
    item["context_options"]
        .as_array()
        .unwrap()
        .iter()
        .map(|context_option| {
            format!("{prefix}{}{}{}", s(context_option), continuation_delimiter, s(&item["continuation"]))
        })
        .collect()
}

/// Render complete prompts for a language modeling task: [without, with] continuation.
/// The contexts are trimmed (some datasets have trailing whitespace), and the
/// without-prompt is stripped so the token-space prefix property holds.
pub fn render_prompts_lm(item: &Value, continuation_delimiter: &str, fewshot_examples: &[Value]) -> Vec<String> {
    let mut prefix = String::new();
    for example in fewshot_examples {
        prefix.push_str(&format!(
            "{}{}{}\n\n",
            s(&example["context"]).trim(),
            continuation_delimiter,
            s(&example["continuation"])
        ));
    }
    let base = format!("{prefix}{}{}", s(&item["context"]).trim(), continuation_delimiter);
    let prompt_without = base.trim().to_string();
    let prompt_with = format!("{base}{}", s(&item["continuation"]));
    vec![prompt_without, prompt_with]
}

/// Find the length of the common prefix ('left') or suffix ('right') across token sequences.
pub fn find_common_length(token_sequences: &[Vec<u32>], direction: &str) -> usize {
    let min_len = token_sequences.iter().map(|s| s.len()).min().unwrap();
    for i in 0..min_len {
        let token = match direction {
            "left" => token_sequences[0][i],
            _ => token_sequences[0][token_sequences[0].len() - 1 - i],
        };
        let all_match = token_sequences.iter().all(|seq| match direction {
            "left" => seq[i] == token,
            _ => seq[seq.len() - 1 - i] == token,
        });
        if !all_match {
            return i;
        }
    }
    min_len
}

/// Stack up a list of token sequences, pad to longest on the right.
pub fn stack_sequences(tokens: &[Vec<u32>], pad_token_id: u32, device: &candle_core::Device) -> Result<Tensor> {
    let bsz = tokens.len();
    let seq_len = tokens.iter().map(|x| x.len()).max().unwrap();
    let mut data = vec![pad_token_id as i64; bsz * seq_len];
    for (i, x) in tokens.iter().enumerate() {
        for (j, &tok) in x.iter().enumerate() {
            data[i * seq_len + j] = tok as i64;
        }
    }
    Tensor::from_vec(data, (bsz, seq_len), device)
}

fn encode_with_bos(tokenizer: &Tokenizer, prompts: &[String]) -> Vec<Vec<u32>> {
    prompts
        .iter()
        .map(|p| {
            let mut ids = vec![tokenizer.get_bos_token_id()];
            ids.extend(tokenizer.encode_ordinary(p));
            ids
        })
        .collect()
}

/// In multiple choice, contexts are the same but the continuation is different (common prefix)
pub fn batch_sequences_mc(tokenizer: &Tokenizer, prompts: &[String]) -> (Vec<Vec<u32>>, Vec<usize>, Vec<usize>) {
    let tokens = encode_with_bos(tokenizer, prompts);
    let answer_start_idx = find_common_length(&tokens, "left");
    let start_indices = vec![answer_start_idx; prompts.len()];
    let end_indices = tokens.iter().map(|x| x.len()).collect();
    (tokens, start_indices, end_indices)
}

/// In schema tasks, contexts vary but continuation is the same (common suffix)
pub fn batch_sequences_schema(tokenizer: &Tokenizer, prompts: &[String]) -> (Vec<Vec<u32>>, Vec<usize>, Vec<usize>) {
    let tokens = encode_with_bos(tokenizer, prompts);
    let suffix_length = find_common_length(&tokens, "right");
    let end_indices: Vec<usize> = tokens.iter().map(|x| x.len()).collect();
    let start_indices = end_indices.iter().map(|ei| ei - suffix_length).collect();
    (tokens, start_indices, end_indices)
}

/// In LM tasks, we have two prompts: without and with continuation (batch size 1)
pub fn batch_sequences_lm(tokenizer: &Tokenizer, prompts: &[String]) -> (Vec<Vec<u32>>, Vec<usize>, Vec<usize>) {
    let tokens = encode_with_bos(tokenizer, prompts);
    let (tokens_without, tokens_with) = (&tokens[0], &tokens[1]);
    let (start_idx, end_idx) = (tokens_without.len(), tokens_with.len());
    assert!(start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with");
    assert!(
        tokens_without[..] == tokens_with[..start_idx],
        "prompt without is supposed to be a prefix of prompt with"
    );
    (vec![tokens_with.clone()], vec![start_idx], vec![end_idx])
}

/// Take a BxT tensor of token ids, return BxT losses and argmax predictions.
/// The autoregressive targets at the last column wrap around; the reference
/// marks that column NaN, and downstream slices never read it.
pub fn forward_model(model: &GPT, input_ids: &Tensor) -> Result<(Tensor, Tensor)> {
    let (batch_size, seq_len) = input_ids.dims2()?;
    let outputs = model.forward(input_ids, None, None, "mean")?;
    // Roll the tensor to the left by one position to get the (autoregressive) target ids
    let target_ids = Tensor::cat(
        &[input_ids.narrow(1, 1, seq_len - 1)?, input_ids.narrow(1, 0, 1)?],
        1,
    )?;
    // Cross entropy at all positions (no ignore_index; ids are always valid)
    let losses = cross_entropy(&outputs, &target_ids, -1, "none")?.reshape((batch_size, seq_len))?;
    // Get the argmax predictions at each position
    let predictions = outputs.argmax(D::Minus1)?;
    Ok((losses, predictions))
}

/// Evaluate a single example, return true if correct.
pub fn evaluate_example(
    idx: usize,
    model: &GPT,
    tokenizer: &Tokenizer,
    data: &[Value],
    device: &candle_core::Device,
    task_meta: &Value,
) -> Result<bool> {
    let item = &data[idx];
    let task_type = s(&task_meta["task_type"]).to_string();
    let num_fewshot = task_meta["num_fewshot"].as_u64().unwrap_or(0) as usize;
    let continuation_delimiter = s(&task_meta["continuation_delimiter"]).to_string();

    // Sample few-shot examples (excluding current item)
    let mut fewshot_examples: Vec<Value> = Vec::new();
    if num_fewshot > 0 {
        let mut rng = PyRandom::new(1234 + idx as u64);
        let available_indices: Vec<usize> = (0..data.len()).filter(|&i| i != idx).collect();
        let fewshot_indices = rng.sample(&available_indices, num_fewshot);
        fewshot_examples = fewshot_indices.iter().map(|&i| data[i].clone()).collect();
    }

    // Render prompts and batch sequences based on task type
    let (tokens, start_idxs, end_idxs) = match task_type.as_str() {
        "multiple_choice" => {
            let prompts = render_prompts_mc(item, &continuation_delimiter, &fewshot_examples);
            batch_sequences_mc(tokenizer, &prompts)
        }
        "schema" => {
            let prompts = render_prompts_schema(item, &continuation_delimiter, &fewshot_examples);
            batch_sequences_schema(tokenizer, &prompts)
        }
        "language_modeling" => {
            let prompts = render_prompts_lm(item, &continuation_delimiter, &fewshot_examples);
            batch_sequences_lm(tokenizer, &prompts)
        }
        other => candle_core::bail!("Unsupported task type: {other}"),
    };
    // NOTE: the reference truncates to model.max_seq_len when set; the GPT
    // here has no such limit, so sequences are forwarded as-is.

    // Stack up all the sequences into a batch (BOS as pad token is ok)
    let pad_token_id = tokenizer.get_bos_token_id();
    let input_ids = stack_sequences(&tokens, pad_token_id, device)?;

    // Forward the model, get the autoregressive loss and argmax prediction at each token
    let (losses, predictions) = forward_model(model, &input_ids)?;

    // See if the losses/predictions come out correctly
    let is_correct = match task_type.as_str() {
        "language_modeling" => {
            // language modeling task is currently always batch size 1
            let (si, ei) = (start_idxs[0], end_idxs[0]);
            // predictions[i] predict input_ids[i+1] autoregressively
            let predicted_tokens = predictions.i((0, si - 1..ei - 1))?.to_vec1::<u32>()?;
            let actual_tokens = input_ids.i((0, si..ei))?.to_vec1::<i64>()?;
            predicted_tokens
                .iter()
                .zip(actual_tokens.iter())
                .all(|(&p, &a)| p as i64 == a)
        }
        _ => {
            // For MC/schema: find the option with lowest average loss
            let mut mean_losses = Vec::with_capacity(start_idxs.len());
            for (i, (&si, &ei)) in start_idxs.iter().zip(end_idxs.iter()).enumerate() {
                let mean = losses.i((i, si - 1..ei - 1))?.mean_all()?.to_scalar::<f32>()?;
                mean_losses.push(mean);
            }
            // index of the minimum, ties keep the first
            let mut pred_idx = 0;
            for (i, &l) in mean_losses.iter().enumerate() {
                if l < mean_losses[pred_idx] {
                    pred_idx = i;
                }
            }
            pred_idx == item["gold"].as_u64().unwrap() as usize
        }
    };
    Ok(is_correct)
}

/// Evaluate one task across many examples (single process).
pub fn evaluate_task(
    model: &GPT,
    tokenizer: &Tokenizer,
    data: &[Value],
    device: &candle_core::Device,
    task_meta: &Value,
) -> Result<f64> {
    let mut correct = vec![0f64; data.len()];
    for (idx, c) in correct.iter_mut().enumerate() {
        *c = evaluate_example(idx, model, tokenizer, data, device, task_meta)? as u8 as f64;
    }
    let mean_correct = correct.iter().sum::<f64>() / correct.len() as f64;
    Ok(mean_correct)
}
