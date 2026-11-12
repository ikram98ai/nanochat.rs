//! Base class for all Tasks.
//! A Task is basically a dataset of conversations, together with some
//! metadata and often also evaluation criteria.
//! Example tasks: MMLU, ARC-Easy, ARC-Challenge, GSM8K, HumanEval, SmolTalk.

use candle_core::Result;
use serde_json::Value;

use crate::nanochat::pyrandom::PyRandom;

/// Lightweight logical view (start/stop/step) over a dataset.
#[derive(Clone, Debug)]
pub struct Slice {
    pub start: usize,
    pub stop: Option<usize>,
    pub step: usize,
}

impl Default for Slice {
    fn default() -> Self {
        Self {
            start: 0,
            stop: None,
            step: 1,
        }
    }
}

impl Slice {
    pub fn new(start: usize, stop: Option<usize>, step: usize) -> Self {
        if let Some(stop) = stop {
            assert!(
                stop >= start,
                "Stop should be greater than or equal to start"
            );
        }
        assert!(step >= 1, "Step must be strictly positive");
        Self { start, stop, step }
    }
}

pub trait Task {
    fn slice(&self) -> &Slice;

    /// one of 'generative' | 'categorical'
    fn eval_type(&self) -> &'static str;

    fn num_examples(&self) -> usize;

    fn get_example(&self, index: usize) -> Result<Value>;

    fn len(&self) -> usize {
        let start = self.slice().start;
        let stop = self.slice().stop.unwrap_or_else(|| self.num_examples());
        let step = self.slice().step;
        let span = stop - start;
        (span + step - 1) / step // ceil_div(span, step)
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get(&self, index: usize) -> Result<Value> {
        let physical_index = self.slice().start + index * self.slice().step;
        self.get_example(physical_index)
    }

    /// Given (conversation, completion), return evaluation outcome.
    fn evaluate(&self, _conversation: &Value, _completion: &str) -> Result<bool> {
        candle_core::bail!("evaluate is not implemented for this task")
    }

    /// Used during RL. To keep things simple, just re-use the evaluation.
    fn reward(&self, conversation: &Value, completion: &str) -> Result<f64> {
        Ok(self.evaluate(conversation, completion)? as u8 as f64)
    }
}

/// For SFT Training it becomes useful to train on a mixture of datasets.
/// Fun trick: if you wish to oversample any task, just pass it in multiple times.
pub struct TaskMixture {
    slice: Slice,
    tasks: Vec<Box<dyn Task>>,
    num_conversations: usize,
    /// (task_idx, local_idx) pairs, deterministically shuffled to mix tasks
    index_map: Vec<(usize, usize)>,
}

impl TaskMixture {
    pub fn new(tasks: Vec<Box<dyn Task>>) -> Self {
        let lengths: Vec<usize> = tasks.iter().map(|t| t.len()).collect();
        let num_conversations = lengths.iter().sum();
        let mut index_map = Vec::with_capacity(num_conversations);
        for (task_idx, task_length) in lengths.iter().enumerate() {
            for local_idx in 0..*task_length {
                index_map.push((task_idx, local_idx));
            }
        }
        // Deterministically shuffle to mix tasks throughout training
        let mut rng = PyRandom::new(42);
        rng.shuffle(&mut index_map);
        Self {
            slice: Slice::default(),
            tasks,
            num_conversations,
            index_map,
        }
    }
}

impl Task for TaskMixture {
    fn slice(&self) -> &Slice {
        &self.slice
    }
    fn eval_type(&self) -> &'static str {
        unimplemented!("mixtures are for training, not evaluation")
    }
    fn num_examples(&self) -> usize {
        self.num_conversations
    }
    /// Access conversations according to a deterministic shuffle of all examples.
    fn get_example(&self, index: usize) -> Result<Value> {
        assert!(
            index < self.num_conversations,
            "Index {index} out of range for mixture"
        );
        let (task_idx, local_idx) = self.index_map[index];
        self.tasks[task_idx].get(local_idx)
    }
}

/// For SFT Training sometimes we want to sequentially train on a list of tasks
/// (useful for cases that require a training curriculum).
pub struct TaskSequence {
    slice: Slice,
    tasks: Vec<Box<dyn Task>>,
    lengths: Vec<usize>,
    num_conversations: usize,
}

impl TaskSequence {
    pub fn new(tasks: Vec<Box<dyn Task>>) -> Self {
        let lengths: Vec<usize> = tasks.iter().map(|t| t.len()).collect();
        let num_conversations = lengths.iter().sum();
        Self {
            slice: Slice::default(),
            tasks,
            lengths,
            num_conversations,
        }
    }
}

impl Task for TaskSequence {
    fn slice(&self) -> &Slice {
        &self.slice
    }
    fn eval_type(&self) -> &'static str {
        unimplemented!("sequences are for training, not evaluation")
    }
    fn num_examples(&self) -> usize {
        self.num_conversations
    }
    fn get_example(&self, index: usize) -> Result<Value> {
        assert!(
            index < self.num_conversations,
            "Index {index} out of range for sequence"
        );
        let mut index = index;
        for (task_idx, task_length) in self.lengths.iter().enumerate() {
            if index < *task_length {
                return self.tasks[task_idx].get(index);
            }
            index -= task_length;
        }
        unreachable!()
    }
}

/// The common multiple choice rendering format we will use.
///
/// Note two important design decisions:
/// 1) Smaller models prefer to have the letter *after* the choice, which
///    results in better binding.
/// 2) There is no whitespace between the delimiter (=) and the letter, so the
///    letter is the exact same token as in the assistant response.
pub fn render_mc(question: &str, letters: &[String], choices: &[String]) -> String {
    let mut query = format!("Multiple Choice question: {question}\n");
    for (letter, choice) in letters.iter().zip(choices.iter()) {
        query.push_str(&format!("- {choice}={letter}\n"));
    }
    query.push_str("\nRespond only with the letter of the correct answer.");
    query
}

/// Rows of a JSONL file (one JSON object per line), e.g. to build a task from
/// local rows with its from_rows constructor.
pub fn load_jsonl_rows(path: &std::path::Path) -> Result<Vec<Value>> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| candle_core::Error::Msg(format!("Error reading {}: {e}", path.display())))?;
    data.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).map_err(|e| candle_core::Error::Msg(e.to_string())))
        .collect()
}
