//! The MMLU dataset.
//! https://huggingface.co/datasets/cais/mmlu

use candle_core::Result;
use serde_json::{json, Value};

use super::common::{hf_data_dir, load_jsonl_rows, render_mc, Slice, Task};

pub const LETTERS: [&str; 4] = ["A", "B", "C", "D"];

pub struct MMLU {
    slice: Slice,
    rows: Vec<Value>,
}

impl MMLU {
    pub fn new(subset: &str, split: &str, slice: Slice) -> Result<Self> {
        assert!(subset == "all", "subset {subset} must be all");
        assert!(
            ["auxiliary_train", "validation", "dev", "test"].contains(&split),
            "split {split} must be auxiliary_train|validation|dev|test"
        );
        let path = hf_data_dir().join(format!("mmlu_{subset}_{split}.jsonl"));
        let rows = load_jsonl_rows(&path)?;
        Ok(Self::from_rows(rows, slice))
    }

    /// Rows in the seed-42 shuffled dataset order.
    pub fn from_rows(rows: Vec<Value>, slice: Slice) -> Self {
        Self { slice, rows }
    }
}

impl Task for MMLU {
    fn slice(&self) -> &Slice {
        &self.slice
    }

    fn eval_type(&self) -> &'static str {
        "categorical"
    }

    fn num_examples(&self) -> usize {
        self.rows.len()
    }

    fn get_example(&self, index: usize) -> Result<Value> {
        let row = &self.rows[index];
        let question = row["question"].as_str().unwrap(); // the question text
        let choices: Vec<String> = serde_json::from_value(row["choices"].clone())
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let answer = row["answer"].as_u64().unwrap() as usize; // index of the answer (A,B,C,D)
        let subject = row["subject"].as_str().unwrap_or_default();
        assert!(choices.len() == 4, "MMLU should have 4 choices");
        // create and return the Conversation object
        let letters: Vec<String> = LETTERS.iter().map(|l| l.to_string()).collect();
        let user_message = render_mc(question, &letters, &choices);
        let assistant_message = LETTERS[answer];
        Ok(json!({
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ],
            "subject": subject, // might be useful later for grouping metrics by subject
            "letters": letters,
        }))
    }

    fn evaluate(&self, conversation: &Value, assistant_response: &str) -> Result<bool> {
        assert!(
            LETTERS.contains(&assistant_response),
            "MMLU answer {assistant_response} is expected to be one of {LETTERS:?}"
        );
        let messages = conversation["messages"].as_array().unwrap();
        let assistant_message = messages.last().unwrap()["content"].as_str().unwrap(); // e.g. "A"
        Ok(assistant_response == assistant_message)
    }
}
