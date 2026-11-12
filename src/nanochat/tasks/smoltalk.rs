//! SmolTalk by HuggingFace. Good "general" conversational dataset.
//! https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
//! We use the "smol" version, which is more appropriate for smaller models.

use candle_core::Result;
use serde_json::{json, Value};

use super::common::{Slice, Task};
use super::hub::load_shuffled;

/// smol-smoltalk dataset. train is 460K rows, test is 24K rows.
pub struct SmolTalk {
    slice: Slice,
    rows: Vec<Value>,
}

impl SmolTalk {
    pub fn new(split: &str, slice: Slice) -> Result<Self> {
        assert!(split == "train" || split == "test", "SmolTalk split must be train|test");
        let rows = load_shuffled("HuggingFaceTB/smol-smoltalk", "default", split)?;
        Ok(Self::from_rows(rows, slice))
    }

    /// Rows in the seed-42 shuffled dataset order.
    pub fn from_rows(rows: Vec<Value>, slice: Slice) -> Self {
        Self { slice, rows }
    }
}

impl Task for SmolTalk {
    fn slice(&self) -> &Slice {
        &self.slice
    }

    fn eval_type(&self) -> &'static str {
        unimplemented!("SmolTalk is a training dataset")
    }

    fn num_examples(&self) -> usize {
        self.rows.len()
    }

    fn get_example(&self, index: usize) -> Result<Value> {
        let row = &self.rows[index];
        let messages = row["messages"].as_array().expect("messages must be a list");
        // sanity checking: an optional system message, then alternating user/assistant
        assert!(!messages.is_empty());
        let rest = if messages[0]["role"] == "system" { &messages[1..] } else { &messages[..] };
        assert!(rest.len() >= 2, "SmolTalk messages must have at least 2 messages");
        for (i, message) in rest.iter().enumerate() {
            let expected_role = if i % 2 == 0 { "user" } else { "assistant" };
            assert!(
                message["role"] == expected_role,
                "Message {i} has role {} but should be {expected_role}",
                message["role"]
            );
            assert!(message["content"].is_string(), "Content must be a string");
        }
        // create and return the Conversation object (ok to emit the system message too)
        Ok(json!({ "messages": messages }))
    }
}
