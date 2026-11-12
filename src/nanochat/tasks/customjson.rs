//! CustomJSON task for loading conversations from JSONL files.
//! Each line in the JSONL file should be a JSON array of message objects with
//! 'role' and 'content' fields, alternating user/assistant.

use candle_core::Result;
use serde_json::{json, Value};

use super::common::{Slice, Task};

pub struct CustomJSON {
    slice: Slice,
    conversations: Vec<Value>,
}

impl CustomJSON {
    pub fn new(filepath: &std::path::Path, slice: Slice) -> Result<Self> {
        let mut conversations = Vec::new();
        if !filepath.exists() {
            println!("{}", "-".repeat(80));
            println!("Warning: File {} does not exist", filepath.display());
            println!("{}", "-".repeat(80));
        } else {
            let data = std::fs::read_to_string(filepath)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            for line in data.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue; // skip empty lines
                }
                let messages: Value = serde_json::from_str(line)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                // Validate the conversation structure
                let list = messages.as_array().expect("Expected list of messages");
                assert!(
                    list.len() >= 2,
                    "Conversation must have at least 2 messages"
                );
                // Validate message structure and alternating roles
                for (i, message) in list.iter().enumerate() {
                    assert!(
                        message.get("role").is_some(),
                        "Message {i} missing 'role' field"
                    );
                    assert!(
                        message.get("content").is_some(),
                        "Message {i} missing 'content' field"
                    );
                    let expected_role = if i % 2 == 0 { "user" } else { "assistant" };
                    assert!(
                        message["role"] == expected_role,
                        "Message {i} has role {} but should be {expected_role}",
                        message["role"]
                    );
                    assert!(
                        message["content"].is_string(),
                        "Message {i} content must be a string"
                    );
                }
                conversations.push(messages);
            }
        }
        Ok(Self {
            slice,
            conversations,
        })
    }
}

impl Task for CustomJSON {
    fn slice(&self) -> &Slice {
        &self.slice
    }

    fn eval_type(&self) -> &'static str {
        unimplemented!("CustomJSON is a training dataset")
    }

    fn num_examples(&self) -> usize {
        self.conversations.len()
    }

    fn get_example(&self, index: usize) -> Result<Value> {
        Ok(json!({ "messages": self.conversations[index] }))
    }
}
