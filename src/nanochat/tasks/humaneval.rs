//! Evaluate the Chat model on the HumanEval dataset.
//! Btw this dataset is a misnomer and has nothing to do with humans.
//! It is a coding benchmark.

use candle_core::Result;
use serde_json::{json, Value};

use super::common::{Slice, Task};
use super::hub::load_shuffled;
use crate::nanochat::execution::execute_code;

/// Extract import statements from the beginning of a code block.
pub fn extract_imports(prompt: &str) -> String {
    let mut imports = Vec::new();
    for line in prompt.split('\n') {
        let stripped = line.trim();
        if stripped.starts_with("import ") || stripped.starts_with("from ") {
            imports.push(stripped);
        } else if !stripped.is_empty() && !stripped.starts_with('#') {
            // Stop at first non-import, non-comment line
            break;
        }
    }
    imports.join("\n")
}

/// Extract Python code from an LLM completion: the first ```python ... ``` or
/// ``` ... ``` block if present, otherwise the whole completion.
pub fn extract_program(completion: &str) -> String {
    let re = regex::Regex::new(r"(?s)```(?:python)?\s*\n(.*?)\n```").unwrap();
    if let Some(caps) = re.captures(completion) {
        return caps.get(1).unwrap().as_str().trim().to_string();
    }
    completion.trim().to_string()
}

pub struct HumanEval {
    slice: Slice,
    rows: Vec<Value>,
}

impl HumanEval {
    pub fn new(slice: Slice) -> Result<Self> {
        // note: this dataset has no named subsets, its parquet config on the hub is "openai_humaneval"
        let rows = load_shuffled("openai/openai_humaneval", "openai_humaneval", "test")?;
        Ok(Self::from_rows(rows, slice))
    }

    /// Rows in the seed-42 shuffled dataset order.
    pub fn from_rows(rows: Vec<Value>, slice: Slice) -> Self {
        Self { slice, rows }
    }
}

impl Task for HumanEval {
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
        let prompt = row["prompt"].as_str().unwrap(); // the beginning of the program
        let solution = row["canonical_solution"].as_str().unwrap(); // the correct continuation
        let entry_point = row["entry_point"].as_str().unwrap(); // the function to check
        let test = row["test"].as_str().unwrap(); // the test cases
        let complete_solution = format!("{prompt}\n{solution}");
        Ok(json!({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": complete_solution},
            ],
            "entry_point": entry_point, // needed during evaluation
            "test": test, // needed during evaluation
        }))
    }

    /// Run the completion against the test cases and return success.
    fn evaluate(&self, conversation: &Value, completion: &str) -> Result<bool> {
        // the prompt will contain the imports and the function signature
        let messages = conversation["messages"].as_array().unwrap();
        let imports = extract_imports(messages[0]["content"].as_str().unwrap());
        // the completion will usually contain the whole function
        // but not always with the needed imports, so we manually append them
        let completion_code = extract_program(completion);
        let program = format!(
            "{imports}\n\n{completion_code}\n\n{}\ncheck({})",
            conversation["test"].as_str().unwrap(),
            conversation["entry_point"].as_str().unwrap(),
        );
        let result = execute_code(&program, 5.0);
        Ok(result.success)
    }
}
