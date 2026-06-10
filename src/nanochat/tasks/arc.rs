//! The ARC dataset from Allen AI.
//! https://huggingface.co/datasets/allenai/ai2_arc

use candle_core::Result;
use serde_json::{json, Value};

use super::common::{hf_data_dir, load_jsonl_rows, render_mc, Slice, Task};

pub struct ARC {
    slice: Slice,
    rows: Vec<Value>,
}

impl ARC {
    pub fn new(subset: &str, split: &str, slice: Slice) -> Result<Self> {
        assert!(
            subset == "ARC-Easy" || subset == "ARC-Challenge",
            "ARC subset must be ARC-Easy or ARC-Challenge"
        );
        assert!(
            ["train", "validation", "test"].contains(&split),
            "ARC split must be train|validation|test"
        );
        let path = hf_data_dir().join(format!("arc_{subset}_{split}.jsonl"));
        let rows = load_jsonl_rows(&path)?;
        Ok(Self::from_rows(rows, slice))
    }

    /// Rows in the seed-42 shuffled dataset order.
    pub fn from_rows(rows: Vec<Value>, slice: Slice) -> Self {
        Self { slice, rows }
    }
}

impl Task for ARC {
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
        let choices: Vec<String> = serde_json::from_value(row["choices"]["text"].clone())
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let answer_string = row["answerKey"].as_str().unwrap(); // e.g. "A", "B", "C", "D"
        let letters: Vec<String> = serde_json::from_value(row["choices"]["label"].clone())
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        assert!(
            letters.iter().any(|l| l == answer_string),
            "ARC answer {answer_string} must be one of {letters:?}"
        );
        // create and return the Conversation object
        let user_message = render_mc(question, &letters, &choices);
        Ok(json!({
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": answer_string},
            ],
            // useful during evaluation, to narrow the assistant prediction to one of the letters
            "letters": letters,
        }))
    }

    fn evaluate(&self, conversation: &Value, assistant_response: &str) -> Result<bool> {
        let letters: Vec<String> = serde_json::from_value(conversation["letters"].clone())
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        assert!(
            letters.iter().any(|l| l == assistant_response),
            "ARC answer {assistant_response} is expected to be one of {letters:?}"
        );
        let messages = conversation["messages"].as_array().unwrap();
        let assistant_message = messages.last().unwrap()["content"].as_str().unwrap(); // e.g. "A"
        Ok(assistant_response == assistant_message)
    }
}
