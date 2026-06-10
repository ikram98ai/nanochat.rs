//! GSM8K evaluation.
//! https://huggingface.co/datasets/openai/gsm8k
//!
//! GSM8K solutions use calculator tool calls inside << >> tags, which are
//! parsed into python/python_output conversation parts.

use candle_core::Result;
use serde_json::{json, Value};

use super::common::{hf_data_dir, load_jsonl_rows, Slice, Task};

/// Extract the numerical answer after the #### marker, with the official
/// normalization (strip + remove commas).
pub fn extract_answer(completion: &str) -> Option<String> {
    let re = regex::Regex::new(r"#### (\-?[0-9\.,]+)").unwrap();
    re.captures(completion)
        .map(|caps| caps.get(1).unwrap().as_str().trim().replace(',', ""))
}

/// Split an answer string into conversation parts: text segments interleaved
/// with <<expr=result>> calculator calls (python + python_output parts).
pub fn parse_answer_parts(answer: &str) -> Vec<Value> {
    let re = regex::Regex::new(r"<<[^>]+>>").unwrap();
    let mut parts = Vec::new();
    let mut prev_end = 0;
    for m in re.find_iter(answer) {
        // regular text in between tool calls (empty segments included)
        parts.push(json!({"type": "text", "text": &answer[prev_end..m.start()]}));
        let inner = &answer[m.start() + 2..m.end() - 2]; // remove << >>
        // split on the last = to get expression and result
        let (expr, result) = match inner.rfind('=') {
            Some(pos) => (&inner[..pos], &inner[pos + 1..]),
            None => (inner, ""),
        };
        parts.push(json!({"type": "python", "text": expr}));
        parts.push(json!({"type": "python_output", "text": result}));
        prev_end = m.end();
    }
    parts.push(json!({"type": "text", "text": &answer[prev_end..]}));
    parts
}

pub struct GSM8K {
    slice: Slice,
    rows: Vec<Value>,
}

impl GSM8K {
    pub fn new(subset: &str, split: &str, slice: Slice) -> Result<Self> {
        assert!(subset == "main" || subset == "socratic", "GSM8K subset must be main|socratic");
        assert!(split == "train" || split == "test", "GSM8K split must be train|test");
        let path = hf_data_dir().join(format!("gsm8k_{subset}_{split}.jsonl"));
        let rows = load_jsonl_rows(&path)?;
        Ok(Self::from_rows(rows, slice))
    }

    /// Rows in the seed-42 shuffled dataset order.
    pub fn from_rows(rows: Vec<Value>, slice: Slice) -> Self {
        Self { slice, rows }
    }
}

impl Task for GSM8K {
    fn slice(&self) -> &Slice {
        &self.slice
    }

    fn eval_type(&self) -> &'static str {
        "generative"
    }

    fn num_examples(&self) -> usize {
        self.rows.len()
    }

    /// Get a single problem from the dataset.
    fn get_example(&self, index: usize) -> Result<Value> {
        let row = &self.rows[index];
        let question = row["question"].as_str().unwrap(); // the question prompt
        let answer = row["answer"].as_str().unwrap(); // full solution, answer after ####
        let assistant_message_parts = parse_answer_parts(answer);
        Ok(json!({
            "messages": [
                {"role": "user", "content": question}, // note: simple string
                {"role": "assistant", "content": assistant_message_parts}, // list of parts
            ],
        }))
    }

    /// Return whether the sampled response reaches the ground-truth answer.
    fn evaluate(&self, conversation: &Value, assistant_response: &str) -> Result<bool> {
        // First extract the ground truth answer
        let messages = conversation["messages"].as_array().unwrap();
        let assistant_message = messages.last().unwrap();
        assert!(
            assistant_message["role"] == "assistant",
            "Last message must be from the Assistant"
        );
        let parts = assistant_message["content"]
            .as_array()
            .expect("This is expected to be a list of parts");
        // the last text part contains the final answer in GSM8K
        let last_text_part = parts.last().unwrap()["text"].as_str().unwrap();
        // Extract both the ground truth answer and the predicted answer
        let ref_num = extract_answer(last_text_part);
        let pred_num = extract_answer(assistant_response);
        Ok(pred_num == ref_num)
    }
}
