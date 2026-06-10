//! Evaluate the Chat model: generic evaluation loops over tasks.
//!
//! Generative tasks go one problem at a time (sample, evaluate); categorical
//! tasks batch problems and just check the logits of the answer letters.

use std::collections::HashMap;

use candle_core::{IndexOp, Result, Tensor};

use super::common::{get_dist_info, print0};
use super::engine::Engine;
use super::gpt::GPT;
use super::tasks::arc::ARC;
use super::tasks::common::{Slice, Task};
use super::tasks::gsm8k::GSM8K;
use super::tasks::humaneval::HumanEval;
use super::tasks::mmlu::MMLU;
use super::tasks::spellingbee::SpellingBee;
use super::tokenizer::Tokenizer;

/// Generative evaluation loop (we go one problem at a time, sample, evaluate).
#[allow(clippy::too_many_arguments)]
pub fn run_generative_eval(
    task_object: &dyn Task,
    tokenizer: &Tokenizer,
    _model: &GPT,
    engine: &Engine<GPT, Tokenizer>,
    num_samples: usize,
    max_new_tokens: usize,
    temperature: f64,
    top_k: usize,
    max_problems: Option<usize>,
) -> Result<f64> {
    let (_ddp, ddp_rank, _local, ddp_world_size) = get_dist_info();
    let num_problems = match max_problems {
        Some(max) => task_object.len().min(max),
        None => task_object.len(),
    };

    // Run the evaluation
    let (mut num_passed, mut total) = (0usize, 0usize);
    let mut i = ddp_rank;
    while i < num_problems {
        let conversation = task_object.get(i)?;

        // Tokenize the prompt
        let encoded_prompt = tokenizer.render_for_completion(&conversation)?;
        // Get the completions
        let (results, _) = engine.generate_batch(
            &encoded_prompt,
            num_samples,
            Some(max_new_tokens),
            temperature,
            Some(top_k),
            42,
        )?;
        // Decode the completions as text
        let prefix_length = encoded_prompt.len();
        let completions: Vec<String> =
            results.iter().map(|tokens| tokenizer.decode(&tokens[prefix_length..])).collect();
        // Evaluate success criteria
        let mut passed = false;
        for completion in &completions {
            passed |= task_object.evaluate(&conversation, completion)?;
        }

        // Keep stats
        total += 1;
        num_passed += passed as usize;

        // Logging (overwrite the same line in the console)
        print!(
            "\r\x1b[KRank {ddp_rank} | {num_passed}/{total} ({:.2}%)",
            100.0 * num_passed as f64 / total as f64
        );
        use std::io::Write;
        std::io::stdout().flush().ok();

        i += ddp_world_size;
    }
    println!();

    print0(&"=".repeat(50));
    print0(&format!(
        "Final: {num_passed}/{total} ({:.2}%)",
        100.0 * num_passed as f64 / total as f64
    ));
    Ok(num_passed as f64 / total as f64)
}

/// Categorical evaluation loop. A lot easier because there is no sampling:
/// batches of problems at a time, just check the logits of the answer choices.
pub fn run_categorical_eval(
    task_object: &dyn Task,
    tokenizer: &Tokenizer,
    model: &GPT,
    batch_size: usize,
    max_problems: Option<usize>,
) -> Result<f64> {
    let (_ddp, ddp_rank, _local, ddp_world_size) = get_dist_info();
    let device = model.get_device();
    let bos = tokenizer.get_bos_token_id(); // pad with BOS, these positions are ignored

    let num_problems = match max_problems {
        Some(max) => task_object.len().min(max),
        None => task_object.len(),
    };
    let num_batches = (num_problems + batch_size - 1) / batch_size;

    // Run the evaluation
    let mut letter_to_id_cache: HashMap<String, u32> = HashMap::new(); // letters repeat often
    let (mut num_passed, mut total) = (0usize, 0usize);
    let mut i = ddp_rank;
    while i < num_batches {
        let (i0, i1) = (i * batch_size, ((i + 1) * batch_size).min(num_problems));

        // Prepare the batch of problems; pad/collate to the longest prompt.
        let conversations: Vec<_> = (i0..i1).map(|ii| task_object.get(ii)).collect::<Result<_>>()?;
        let prompt_ids: Vec<Vec<u32>> = conversations
            .iter()
            .map(|c| tokenizer.render_for_completion(c))
            .collect::<Result<_>>()?;
        let max_length = prompt_ids.iter().map(|ids| ids.len()).max().unwrap();
        // where the last token is (and the predicted answer)
        let answer_time_positions: Vec<usize> = prompt_ids.iter().map(|ids| ids.len() - 1).collect();
        let mut padded: Vec<i64> = Vec::with_capacity(prompt_ids.len() * max_length);
        for ids in &prompt_ids {
            padded.extend(ids.iter().map(|&t| t as i64));
            padded.extend(std::iter::repeat(bos as i64).take(max_length - ids.len()));
        }
        let input_ids = Tensor::from_vec(padded, (prompt_ids.len(), max_length), device)?;

        // Get the logits for the whole batch of conversations in parallel
        let logits = model.forward(&input_ids, None, None, "mean")?; // (B, T, V)

        // Focus the available answer on just the letters corresponding to choices.
        // This narrows the focus to only the available letters, which helps a lot.
        for (idx, conversation) in conversations.iter().enumerate() {
            // get the token ids of all the available letters of this problem
            let letters: Vec<String> =
                serde_json::from_value(conversation["letters"].clone())
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let mut letter_ids = Vec::with_capacity(letters.len());
            for letter in &letters {
                let id = match letter_to_id_cache.get(letter) {
                    Some(&id) => id,
                    None => {
                        let encoded_letter = tokenizer.encode(letter, None, None);
                        assert!(encoded_letter.len() == 1, "Each letter must be a single token");
                        letter_to_id_cache.insert(letter.clone(), encoded_letter[0]);
                        encoded_letter[0]
                    }
                };
                letter_ids.push(id);
            }
            // focus logits down to the answer position and the available letters
            let answer_pos = answer_time_positions[idx];
            let row: Vec<f32> = logits.i((idx, answer_pos))?.to_vec1()?;
            // argmax letter (first on ties), the predicted answer
            let mut argmax_letter_id = 0;
            for (j, &id) in letter_ids.iter().enumerate() {
                if row[id as usize] > row[letter_ids[argmax_letter_id] as usize] {
                    argmax_letter_id = j;
                }
            }
            let predicted_letter = &letters[argmax_letter_id];
            // evaluate the outcome
            let outcome = task_object.evaluate(conversation, predicted_letter)?;
            num_passed += outcome as usize;
            total += 1;
        }
        i += ddp_world_size;
    }

    let average = num_passed as f64 / total as f64;
    print0(&format!("Final: {num_passed}/{total} ({:.2}%)", 100.0 * average));
    Ok(average)
}

/// Build the named evaluation task.
pub fn build_eval_task(task_name: &str) -> Result<Box<dyn Task>> {
    Ok(match task_name {
        "HumanEval" => Box::new(HumanEval::new(Slice::default())?),
        "MMLU" => Box::new(MMLU::new("all", "test", Slice::default())?),
        "ARC-Easy" => Box::new(ARC::new("ARC-Easy", "test", Slice::default())?),
        "ARC-Challenge" => Box::new(ARC::new("ARC-Challenge", "test", Slice::default())?),
        "GSM8K" => Box::new(GSM8K::new("main", "test", Slice::default())?),
        "SpellingBee" => Box::new(SpellingBee::new(256, "test", Slice::default())?),
        other => candle_core::bail!("Unknown task name: {other}"),
    })
}

/// Run one named evaluation, dispatching on the task's eval type.
#[allow(clippy::too_many_arguments)]
pub fn run_chat_eval(
    task_name: &str,
    model: &GPT,
    tokenizer: &Tokenizer,
    engine: &Engine<GPT, Tokenizer>,
    batch_size: usize,
    num_samples: usize,
    max_new_tokens: usize,
    temperature: f64,
    top_k: usize,
    max_problems: Option<usize>,
) -> Result<f64> {
    let task_object = build_eval_task(task_name)?;
    match task_object.eval_type() {
        "generative" => run_generative_eval(
            task_object.as_ref(),
            tokenizer,
            model,
            engine,
            num_samples,
            max_new_tokens,
            temperature,
            top_k,
            max_problems,
        ),
        "categorical" => {
            run_categorical_eval(task_object.as_ref(), tokenizer, model, batch_size, max_problems)
        }
        other => candle_core::bail!("Unsupported task evaluation type: {other}"),
    }
}
