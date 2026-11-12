//! Task intended to make nanochat better in spelling and counting, for example:
//!
//! "How many r are in strawberry?" -> 3
//!
//! The assistant solves the problem using a combination of manual counting
//! and Python. Two tasks live here:
//! 1. SpellingBee: Counting the number of occurrences of a letter in a word
//! 2. SimpleSpelling: Simply spelling words

use candle_core::Result;
use serde_json::{json, Value};

use super::common::{Slice, Task};
use crate::nanochat::common::download_file_with_lock;
use crate::nanochat::pyrandom::PyRandom;

/// Letters of the alphabet
const LETTERS: &str = "abcdefghijklmnopqrstuvwxyz";
/// A list of 370K English words of large variety
const WORD_LIST_URL: &str =
    "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt";
/// A number bigger than 370K to separate train and test random seeds
const TEST_RANDOM_SEED_OFFSET: u64 = 10_000_000;

/// Extract the numerical answer after the #### marker (identical to gsm8k).
pub fn extract_answer(completion: &str) -> Option<String> {
    super::gsm8k::extract_answer(completion)
}

/// User message templates for data augmentation
const USER_MSG_TEMPLATES: [&str; 51] = [
    "How many {letter} are in the word {word}",
    "How many {letter} are in {word}",
    "Count the number of {letter} in {word}",
    "How many times does {letter} appear in {word}",
    "What's the count of {letter} in {word}",
    "In the word {word}, how many {letter} are there",
    "How many letter {letter} are in the word {word}",
    "Count how many {letter} appear in {word}",
    "Tell me the number of {letter} in {word}",
    "How many occurrences of {letter} are in {word}",
    "Find the count of {letter} in {word}",
    "Can you count the {letter} letters in {word}",
    "What is the frequency of {letter} in {word}",
    "How many {letter}s are in {word}",
    "How many {letter}'s are in {word}",
    "Count all the {letter} in {word}",
    "How many times is {letter} in {word}",
    "Number of {letter} in {word}",
    "Total count of {letter} in {word}",
    "How many {letter} does {word} have",
    "How many {letter} does {word} contain",
    "What's the number of {letter} in {word}",
    "{word} has how many {letter}",
    "In {word}, count the {letter}",
    "How many {letter} appear in {word}",
    "Count the {letter} in {word}",
    "Give me the count of {letter} in {word}",
    "How many instances of {letter} in {word}",
    "Show me how many {letter} are in {word}",
    "Calculate the number of {letter} in {word}",
    // Spanish
    "¿Cuántas {letter} hay en {word}?",
    "¿Cuántas veces aparece {letter} en {word}?",
    "Cuenta las {letter} en {word}",
    "¿Cuántas letras {letter} tiene {word}?",
    // Chinese (Simplified)
    "{word}中有多少个{letter}",
    "{word}里有几个{letter}",
    "数一下{word}中的{letter}",
    "{word}这个词里有多少{letter}",
    // Korean
    "{word}에 {letter}가 몇 개 있나요",
    "{word}에서 {letter}의 개수는",
    "{word}에 {letter}가 몇 번 나오나요",
    "{word}라는 단어에 {letter}가 몇 개",
    // French
    "Combien de {letter} dans {word}",
    "Combien de fois {letter} apparaît dans {word}",
    "Compte les {letter} dans {word}",
    // German
    "Wie viele {letter} sind in {word}",
    "Wie oft kommt {letter} in {word} vor",
    "Zähle die {letter} in {word}",
    // Japanese
    "{word}に{letter}は何個ありますか",
    "{word}の中に{letter}がいくつ",
    "{word}に{letter}が何回出てくる",
];

fn load_word_list() -> Result<Vec<String>> {
    let filename = WORD_LIST_URL.rsplit('/').next().unwrap();
    let path = download_file_with_lock(WORD_LIST_URL, filename, None)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let data =
        std::fs::read_to_string(&path).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    Ok(data.lines().map(|line| line.trim().to_string()).collect())
}

pub struct SpellingBee {
    slice: Slice,
    size: usize,
    split: String,
    words: Vec<String>,
}

impl SpellingBee {
    pub fn new(size: usize, split: &str, slice: Slice) -> Result<Self> {
        assert!(
            split == "train" || split == "test",
            "SpellingBee split must be train|test"
        );
        let words = load_word_list()?;
        Ok(Self {
            slice,
            size,
            split: split.to_string(),
            words,
        })
    }
}

impl Task for SpellingBee {
    fn slice(&self) -> &Slice {
        &self.slice
    }

    fn eval_type(&self) -> &'static str {
        "generative"
    }

    fn num_examples(&self) -> usize {
        self.size
    }

    fn get_example(&self, index: usize) -> Result<Value> {
        let seed = if self.split == "train" {
            index as u64
        } else {
            TEST_RANDOM_SEED_OFFSET + index as u64
        };
        let mut rng = PyRandom::new(seed);

        // pick a random word
        let word = rng.choice(&self.words).clone();
        let word_chars: Vec<char> = word.chars().collect();
        // pick a letter from it (90%) or a random letter (10%)
        let alphabet: Vec<char> = LETTERS.chars().collect();
        let letter = if rng.random() < 0.9 {
            *rng.choice(&word_chars)
        } else {
            *rng.choice(&alphabet)
        };

        // get the correct answer by simply counting
        let count = word_chars.iter().filter(|&&c| c == letter).count();

        // create a user message, with a bunch of variations as data augmentation
        let mut template = rng.choice(&USER_MSG_TEMPLATES).to_string();
        // 30% chance to lowercase the template (lazy people don't use shift)
        if rng.random() < 0.3 {
            template = template.to_lowercase();
        }
        let quote_options = ["", "'", "\""];
        let letter_quote = *rng.choice(&quote_options); // is the letter quoted?
        let word_quote = *rng.choice(&quote_options); // is the word quoted?
        let letter_wrapped = format!("{letter_quote}{letter}{letter_quote}");
        let word_wrapped = format!("{word_quote}{word}{word_quote}");
        let mut user_msg = template
            .replace("{letter}", &letter_wrapped)
            .replace("{word}", &word_wrapped);
        if rng.random() < 0.5 {
            // 50% of people don't even use question marks
            user_msg.push('?');
        }

        // Now create the ideal assistant response - build as parts (text + tool calls)
        let word_letters: Vec<String> = word_chars.iter().map(|c| c.to_string()).collect();
        let word_letters = word_letters.join(",");
        let mut manual_text = format!(
            "We are asked to find the number '{letter}' in the word '{word}'. Let me try a manual approach first.\n\n\
             First spell the word out:\n{word}:{word_letters}\n\n\
             Then count the occurrences of '{letter}':\n"
        );
        // Little simulated loop of the solution process
        let mut running_count = 0;
        for (i, char) in word_chars.iter().enumerate() {
            let i = i + 1;
            if *char == letter {
                running_count += 1;
                // note: deliberately no space between i and char (it would be a different token)
                manual_text.push_str(&format!("{i}:{char} hit! count={running_count}\n"));
            } else {
                manual_text.push_str(&format!("{i}:{char}\n"));
            }
        }
        manual_text.push_str(&format!("\nThis gives us {running_count}."));

        let python_expr = format!("'{word}'.count('{letter}')");
        let assistant_parts = json!([
            {"type": "text", "text": manual_text},
            {"type": "text", "text": "\n\nLet me double check this using Python:\n\n"},
            {"type": "python", "text": python_expr},
            {"type": "python_output", "text": count.to_string()},
            {"type": "text", "text": format!("\n\nPython gives us {count}.\n\nMy final answer is:\n\n#### {count}")},
        ]);

        Ok(json!({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_parts},
            ],
        }))
    }

    /// Identical to gsm8k's evaluation.
    fn evaluate(&self, conversation: &Value, assistant_response: &str) -> Result<bool> {
        let messages = conversation["messages"].as_array().unwrap();
        let assistant_message = messages.last().unwrap();
        assert!(
            assistant_message["role"] == "assistant",
            "Last message must be from the Assistant"
        );
        let parts = assistant_message["content"]
            .as_array()
            .expect("This is expected to be a list of parts");
        // The last text part contains the final answer with ####
        let last_text_part = parts.last().unwrap()["text"].as_str().unwrap();
        let ref_num = extract_answer(last_text_part);
        let pred_num = extract_answer(assistant_response);
        Ok(pred_num == ref_num)
    }
}

/// Much simpler task designed to get the model to just practice spelling words.
pub struct SimpleSpelling {
    slice: Slice,
    size: usize,
    split: String,
    words: Vec<String>,
}

impl SimpleSpelling {
    pub fn new(size: usize, split: &str, slice: Slice) -> Result<Self> {
        assert!(
            split == "train" || split == "test",
            "SpellingBee split must be train|test"
        );
        let mut words = load_word_list()?;
        // use a different word order than the SpellingBee task
        let mut rng = PyRandom::new(42);
        rng.shuffle(&mut words);
        Ok(Self {
            slice,
            size,
            split: split.to_string(),
            words,
        })
    }
}

impl Task for SimpleSpelling {
    fn slice(&self) -> &Slice {
        &self.slice
    }

    fn eval_type(&self) -> &'static str {
        "generative"
    }

    fn num_examples(&self) -> usize {
        self.size
    }

    fn get_example(&self, index: usize) -> Result<Value> {
        let seed = if self.split == "train" {
            index as u64
        } else {
            TEST_RANDOM_SEED_OFFSET + index as u64
        };
        let mut rng = PyRandom::new(seed);
        // pick a random word
        let word = rng.choice(&self.words);
        let word_letters: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        let word_letters = word_letters.join(",");
        Ok(json!({
            "messages": [
                {"role": "user", "content": format!("Spell the word: {word}")},
                {"role": "assistant", "content": format!("{word}:{word_letters}")},
            ],
        }))
    }
}
