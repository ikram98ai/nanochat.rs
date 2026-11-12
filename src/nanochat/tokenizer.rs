//! BPE Tokenizer in the style of GPT-4.
//!
//! Byte-pair encoding over a learned merge table with a GPT-4-style split
//! pattern, equivalent to the tiktoken encoding used for inference by the
//! Python version. The tokenizer is stored as {pattern, mergeable_ranks,
//! special_tokens} JSON (written by tok_train); the Python reference's
//! tokenizer.pkl and HuggingFace-format tokenizer.json files are read directly
//! as well (see tokenizer_import).

use std::collections::HashMap;
use std::path::Path;

use base64::Engine as _;
use candle_core::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::common::get_base_dir;
use super::tokenizer_import;

pub const SPECIAL_TOKENS: [&str; 9] = [
    // every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    // tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", // user messages
    "<|user_end|>",
    "<|assistant_start|>", // assistant messages
    "<|assistant_end|>",
    "<|python_start|>", // assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", // python REPL outputs back to assistant
    "<|output_end|>",
];

/// NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
/// to not "waste" too many tokens on numbers for smaller vocab sizes.
pub const SPLIT_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

fn msg(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

// -----------------------------------------------------------------------------
// Core byte-pair encoding (the tiktoken algorithm)

const RANK_MAX: u32 = u32::MAX;

/// Iteratively merge the lowest-rank adjacent pair until no merge applies.
/// Returns the boundaries of the resulting parts as (start, rank) entries.
fn byte_pair_merge(ranks: &HashMap<Vec<u8>, u32>, piece: &[u8]) -> Vec<(usize, u32)> {
    // parts[i] = (start index, rank of the pair starting at that index)
    let mut parts = Vec::with_capacity(piece.len() + 1);
    let mut min_rank: (u32, usize) = (RANK_MAX, usize::MAX);
    for i in 0..piece.len() - 1 {
        let rank = *ranks.get(&piece[i..i + 2]).unwrap_or(&RANK_MAX);
        if rank < min_rank.0 {
            min_rank = (rank, i);
        }
        parts.push((i, rank));
    }
    parts.push((piece.len() - 1, RANK_MAX));
    parts.push((piece.len(), RANK_MAX));

    let get_rank = |parts: &Vec<(usize, u32)>, i: usize| {
        if (i + 3) < parts.len() {
            *ranks
                .get(&piece[parts[i].0..parts[i + 3].0])
                .unwrap_or(&RANK_MAX)
        } else {
            RANK_MAX
        }
    };

    while min_rank.0 != RANK_MAX {
        let i = min_rank.1;
        // Update parts[i] and parts[i - 1] before removing parts[i + 1]
        if i > 0 {
            parts[i - 1].1 = get_rank(&parts, i - 1);
        }
        parts[i].1 = get_rank(&parts, i);
        parts.remove(i + 1);

        min_rank = (RANK_MAX, usize::MAX);
        for (j, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, j);
            }
        }
    }
    parts
}

fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, u32>) -> Vec<u32> {
    if piece.len() == 1 {
        return vec![ranks[piece]];
    }
    byte_pair_merge(ranks, piece)
        .windows(2)
        .map(|w| ranks[&piece[w[0].0..w[1].0]])
        .collect()
}

// -----------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct RankEntry {
    bytes: String, // base64
    rank: u32,
}

#[derive(Serialize, Deserialize)]
struct TokenizerFile {
    pattern: String,
    mergeable_ranks: Vec<RankEntry>,
    /// (sorted, so that saving the same tokenizer always writes the same file)
    special_tokens: std::collections::BTreeMap<String, u32>,
}

pub struct Tokenizer {
    pattern: fancy_regex::Regex,
    pattern_str: String,
    encoder: HashMap<Vec<u8>, u32>,
    decoder: HashMap<u32, Vec<u8>>,
    special_tokens: HashMap<String, u32>,
    special_tokens_decoder: HashMap<u32, String>,
    bos_token_id: u32,
}

impl Tokenizer {
    pub fn new(
        pattern: &str,
        mergeable_ranks: HashMap<Vec<u8>, u32>,
        special_tokens: HashMap<String, u32>,
        bos_token: &str,
    ) -> Result<Self> {
        let regex = fancy_regex::Regex::new(pattern).map_err(msg)?;
        let decoder: HashMap<u32, Vec<u8>> = mergeable_ranks
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();
        let special_tokens_decoder: HashMap<u32, String> = special_tokens
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();
        let bos_token_id = *special_tokens
            .get(bos_token)
            .ok_or_else(|| msg(format!("Failed to find BOS token {bos_token} in tokenizer")))?;
        Ok(Self {
            pattern: regex,
            pattern_str: pattern.to_string(),
            encoder: mergeable_ranks,
            decoder,
            special_tokens,
            special_tokens_decoder,
            bos_token_id,
        })
    }

    /// Load the tokenizer of a checkpoint directory. Reads, in order of preference:
    /// - tokenizer.json in this port's format (as written by tok_train),
    /// - tokenizer.json in the HuggingFace tokenizers format (HF-format nanochat repos),
    /// - tokenizer.pkl, the Python reference's pickled tiktoken Encoding.
    pub fn from_directory(tokenizer_dir: &Path) -> Result<Self> {
        let json_path = tokenizer_dir.join("tokenizer.json");
        let pickle_path = tokenizer_dir.join("tokenizer.pkl");
        let (pattern, mergeable_ranks, special_tokens) = if json_path.exists() {
            let data = std::fs::read_to_string(&json_path)
                .map_err(|e| msg(format!("Error reading {}: {e}", json_path.display())))?;
            let value: Value = serde_json::from_str(&data).map_err(msg)?;
            if value.get("mergeable_ranks").is_some() {
                let file: TokenizerFile = serde_json::from_value(value).map_err(msg)?;
                let mut mergeable_ranks = HashMap::new();
                for entry in &file.mergeable_ranks {
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(&entry.bytes)
                        .map_err(msg)?;
                    mergeable_ranks.insert(bytes, entry.rank);
                }
                (
                    file.pattern,
                    mergeable_ranks,
                    file.special_tokens.into_iter().collect(),
                )
            } else {
                let parts = tokenizer_import::from_hf_json(&value)?;
                tokenizer_import::validate(&parts)?;
                parts
            }
        } else if pickle_path.exists() {
            let data = std::fs::read(&pickle_path)
                .map_err(|e| msg(format!("Error reading {}: {e}", pickle_path.display())))?;
            let parts = tokenizer_import::from_tiktoken_pickle(&data)?;
            tokenizer_import::validate(&parts)?;
            parts
        } else {
            return Err(msg(format!(
                "No tokenizer.json or tokenizer.pkl found in {}",
                tokenizer_dir.display()
            )));
        };
        Self::new(&pattern, mergeable_ranks, special_tokens, "<|bos|>")
    }

    pub fn save(&self, tokenizer_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(tokenizer_dir).map_err(msg)?;
        let path = tokenizer_dir.join("tokenizer.json");
        let mut ranks: Vec<(&Vec<u8>, &u32)> = self.encoder.iter().collect();
        ranks.sort_by_key(|(_, rank)| **rank);
        let file = TokenizerFile {
            pattern: self.pattern_str.clone(),
            mergeable_ranks: ranks
                .into_iter()
                .map(|(bytes, rank)| RankEntry {
                    bytes: base64::engine::general_purpose::STANDARD.encode(bytes),
                    rank: *rank,
                })
                .collect(),
            special_tokens: self
                .special_tokens
                .iter()
                .map(|(name, id)| (name.clone(), *id))
                .collect(),
        };
        std::fs::write(&path, serde_json::to_string(&file).map_err(msg)?).map_err(msg)?;
        println!("Saved tokenizer to {}", path.display());
        Ok(())
    }

    /// n_vocab = max token id + 1
    pub fn get_vocab_size(&self) -> usize {
        let max_rank = self.encoder.values().max().copied().unwrap_or(0);
        let max_special = self.special_tokens.values().max().copied().unwrap_or(0);
        max_rank.max(max_special) as usize + 1
    }

    pub fn get_special_tokens(&self) -> Vec<String> {
        self.special_tokens.keys().cloned().collect()
    }

    pub fn id_to_token(&self, id: u32) -> String {
        self.decode(&[id])
    }

    /// encode a single special token via exact match
    pub fn encode_special(&self, text: &str) -> u32 {
        self.special_tokens[text]
    }

    pub fn get_bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    /// Encode ordinary text (no special token handling).
    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for piece in self.pattern.find_iter(text) {
            let piece = piece.expect("regex find failure").as_str().as_bytes();
            match self.encoder.get(piece) {
                Some(&id) => ids.push(id),
                None => ids.extend(byte_pair_encode(piece, &self.encoder)),
            }
        }
        ids
    }

    pub fn encode(&self, text: &str, prepend: Option<&str>, append: Option<&str>) -> Vec<u32> {
        let mut ids = self.encode_ordinary(text);
        if let Some(prepend) = prepend {
            ids.insert(0, self.encode_special(prepend));
        }
        if let Some(append) = append {
            ids.push(self.encode_special(append));
        }
        ids
    }

    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        texts.iter().map(|t| self.encode_ordinary(t)).collect()
    }

    /// Encode a batch of texts on up to num_threads threads (the equivalent of
    /// tiktoken's encode_ordinary_batch). The output order matches the input.
    pub fn encode_batch_threaded<S: AsRef<str> + Sync>(
        &self,
        texts: &[S],
        num_threads: usize,
    ) -> Vec<Vec<u32>> {
        let num_threads = num_threads.clamp(1, texts.len().max(1));
        if num_threads == 1 {
            return texts
                .iter()
                .map(|t| self.encode_ordinary(t.as_ref()))
                .collect();
        }
        let chunk_size = texts.len().div_ceil(num_threads);
        std::thread::scope(|scope| {
            let handles: Vec<_> = texts
                .chunks(chunk_size)
                .map(|chunk| {
                    scope.spawn(move || {
                        chunk
                            .iter()
                            .map(|t| self.encode_ordinary(t.as_ref()))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("tokenizer thread panicked"))
                .collect()
        })
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            if let Some(b) = self.decoder.get(id) {
                bytes.extend_from_slice(b);
            } else if let Some(s) = self.special_tokens_decoder.get(id) {
                bytes.extend_from_slice(s.as_bytes());
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Tokenize a single Chat conversation (a JSON object with a "messages" list).
    /// Returns:
    /// - ids: token ids of this rendered conversation
    /// - mask: same length, 1 for tokens that the Assistant is expected to train on
    pub fn render_conversation(
        &self,
        conversation: &Value,
        max_tokens: usize,
    ) -> Result<(Vec<u32>, Vec<u8>)> {
        let mut ids: Vec<u32> = Vec::new();
        let mut mask: Vec<u8> = Vec::new();
        let add_tokens =
            |ids: &mut Vec<u32>, mask: &mut Vec<u8>, token_ids: &[u32], mask_val: u8| {
                ids.extend_from_slice(token_ids);
                mask.extend(std::iter::repeat(mask_val).take(token_ids.len()));
            };

        let messages = conversation["messages"]
            .as_array()
            .ok_or_else(|| msg("Conversation must have a messages list"))?;
        // sometimes the first message is a system message...
        // => just merge it with the second (user) message
        let mut messages: Vec<Value> = messages.clone();
        if messages.first().and_then(|m| m["role"].as_str()) == Some("system") {
            assert!(
                messages.get(1).and_then(|m| m["role"].as_str()) == Some("user"),
                "System message must be followed by a user message"
            );
            let system_content = messages[0]["content"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            let user_content = messages[1]["content"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            messages[1]["content"] = Value::String(format!("{system_content}\n\n{user_content}"));
            messages.remove(0);
        }
        assert!(!messages.is_empty(), "Conversation has less than 1 message");

        // fetch all the special tokens we need
        let bos = self.get_bos_token_id();
        let user_start = self.encode_special("<|user_start|>");
        let user_end = self.encode_special("<|user_end|>");
        let assistant_start = self.encode_special("<|assistant_start|>");
        let assistant_end = self.encode_special("<|assistant_end|>");
        let python_start = self.encode_special("<|python_start|>");
        let python_end = self.encode_special("<|python_end|>");
        let output_start = self.encode_special("<|output_start|>");
        let output_end = self.encode_special("<|output_end|>");

        // now we can tokenize the conversation
        add_tokens(&mut ids, &mut mask, &[bos], 0);
        for (i, message) in messages.iter().enumerate() {
            // some sanity checking here around assumptions, to prevent footguns
            let must_be_from = if i % 2 == 0 { "user" } else { "assistant" };
            let role = message["role"].as_str().unwrap_or_default();
            assert!(
                role == must_be_from,
                "Message {i} is from {role} but should be from {must_be_from}"
            );

            // content can be either a simple string or a list of parts (e.g. containing tool calls)
            let content = &message["content"];

            if role == "user" {
                let content = content
                    .as_str()
                    .expect("User messages are simply expected to be strings");
                let value_ids = self.encode_ordinary(content);
                add_tokens(&mut ids, &mut mask, &[user_start], 0);
                add_tokens(&mut ids, &mut mask, &value_ids, 0);
                add_tokens(&mut ids, &mut mask, &[user_end], 0);
            } else {
                add_tokens(&mut ids, &mut mask, &[assistant_start], 0);
                if let Some(content) = content.as_str() {
                    // simple string => simply add the tokens
                    let value_ids = self.encode_ordinary(content);
                    add_tokens(&mut ids, &mut mask, &value_ids, 1);
                } else if let Some(parts) = content.as_array() {
                    for part in parts {
                        let text = part["text"].as_str().unwrap_or_default();
                        let value_ids = self.encode_ordinary(text);
                        match part["type"].as_str().unwrap_or_default() {
                            // string part => simply add the tokens
                            "text" => add_tokens(&mut ids, &mut mask, &value_ids, 1),
                            // python tool call => tokens inside <|python_start|> and <|python_end|>
                            "python" => {
                                add_tokens(&mut ids, &mut mask, &[python_start], 1);
                                add_tokens(&mut ids, &mut mask, &value_ids, 1);
                                add_tokens(&mut ids, &mut mask, &[python_end], 1);
                            }
                            // python output => tokens inside <|output_start|> and <|output_end|>
                            // none of these tokens are supervised because they come from Python at test time
                            "python_output" => {
                                add_tokens(&mut ids, &mut mask, &[output_start], 0);
                                add_tokens(&mut ids, &mut mask, &value_ids, 0);
                                add_tokens(&mut ids, &mut mask, &[output_end], 0);
                            }
                            other => panic!("Unknown part type: {other}"),
                        }
                    }
                } else {
                    panic!("Unknown content type");
                }
                add_tokens(&mut ids, &mut mask, &[assistant_end], 1);
            }
        }

        // truncate to max_tokens tokens MAX (helps prevent OOMs)
        ids.truncate(max_tokens);
        mask.truncate(max_tokens);
        Ok((ids, mask))
    }

    /// Small helper function useful in debugging: visualize the tokenization of render_conversation
    pub fn visualize_tokenization(&self, ids: &[u32], mask: &[u8]) -> String {
        const RED: &str = "\x1b[91m";
        const GREEN: &str = "\x1b[92m";
        const RESET: &str = "\x1b[0m";
        let tokens: Vec<String> = ids
            .iter()
            .zip(mask.iter())
            .map(|(&id, &m)| {
                let color = if m == 1 { GREEN } else { RED };
                format!("{color}{}{RESET}", self.decode(&[id]))
            })
            .collect();
        tokens.join("|")
    }

    /// Used during Reinforcement Learning: render the conversation priming the
    /// Assistant for a completion. Unlike the Chat SFT case, no mask is needed.
    pub fn render_for_completion(&self, conversation: &Value) -> Result<Vec<u32>> {
        // We have some surgery to do: we need to pop the last message (of the Assistant)
        let mut conversation = conversation.clone();
        let messages = conversation["messages"]
            .as_array_mut()
            .ok_or_else(|| msg("Conversation must have a messages list"))?;
        let last_role = messages.last().and_then(|m| m["role"].as_str());
        assert!(
            last_role == Some("assistant"),
            "Last message must be from the Assistant"
        );
        messages.pop(); // remove the last message (of the Assistant)

        // Now tokenize the conversation
        let (mut ids, _mask) = self.render_conversation(&conversation, 2048)?;

        // Finally, to prime the Assistant for a completion, append the Assistant start token
        let assistant_start = self.encode_special("<|assistant_start|>");
        ids.push(assistant_start);
        Ok(ids)
    }
}

impl Tokenizer {
    /// Train a BPE tokenizer from an iterator of text in the GPT-4 style:
    /// pretokenize with the split pattern, then iteratively merge the most
    /// frequent adjacent token pair (ties broken by the smallest pair).
    /// The special tokens are appended at the end of the vocabulary.
    pub fn train_from_iterator(
        texts: impl Iterator<Item = String>,
        vocab_size: usize,
    ) -> Result<Tokenizer> {
        let vocab_size_no_special = vocab_size - SPECIAL_TOKENS.len();
        assert!(
            vocab_size_no_special >= 256,
            "vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        );
        let pattern = fancy_regex::Regex::new(SPLIT_PATTERN).map_err(msg)?;

        // Pretokenize: count the frequency of each text chunk
        let mut chunk_counts: HashMap<Vec<u8>, u64> = HashMap::new();
        for text in texts {
            for piece in pattern.find_iter(&text) {
                let piece = piece.map_err(msg)?.as_str().as_bytes().to_vec();
                *chunk_counts.entry(piece).or_insert(0) += 1;
            }
        }
        // words as token-id sequences (starting as raw bytes), with counts
        let mut words: Vec<(Vec<u32>, u64)> = chunk_counts
            .into_iter()
            .map(|(bytes, count)| (bytes.iter().map(|&b| b as u32).collect(), count))
            .collect();
        words.sort(); // deterministic processing order

        // count all adjacent pairs, tracking which words contain each pair
        let mut pair_counts: HashMap<(u32, u32), i64> = HashMap::new();
        let mut pair_words: HashMap<(u32, u32), std::collections::HashSet<usize>> = HashMap::new();
        for (w_idx, (word, count)) in words.iter().enumerate() {
            for pair in word.windows(2) {
                let pair = (pair[0], pair[1]);
                *pair_counts.entry(pair).or_insert(0) += *count as i64;
                pair_words.entry(pair).or_default().insert(w_idx);
            }
        }

        // base vocabulary: the 256 bytes
        let mut token_bytes: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        let num_merges = vocab_size_no_special - 256;
        for merge_idx in 0..num_merges {
            // most frequent pair; ties broken by the smallest pair
            let best = pair_counts
                .iter()
                .filter(|(_, &c)| c > 0)
                .max_by(|(p1, c1), (p2, c2)| c1.cmp(c2).then(p2.cmp(p1)))
                .map(|(&p, _)| p);
            let Some(pair) = best else {
                break; // no pairs left to merge
            };
            let new_id = (256 + merge_idx) as u32;
            token_bytes.push(
                [
                    token_bytes[pair.0 as usize].clone(),
                    token_bytes[pair.1 as usize].clone(),
                ]
                .concat(),
            );

            // merge the pair in every word that contains it, updating pair counts
            let affected: Vec<usize> = pair_words
                .remove(&pair)
                .map(|s| s.into_iter().collect())
                .unwrap_or_default();
            for w_idx in affected {
                let (word, count) = &words[w_idx];
                let count = *count as i64;
                // remove this word's contribution to the pair counts
                for p in word.windows(2) {
                    let p = (p[0], p[1]);
                    *pair_counts.entry(p).or_insert(0) -= count;
                    if p != pair {
                        if let Some(set) = pair_words.get_mut(&p) {
                            set.remove(&w_idx);
                        }
                    }
                }
                // apply the merge
                let mut merged = Vec::with_capacity(word.len());
                let mut i = 0;
                while i < word.len() {
                    if i + 1 < word.len() && word[i] == pair.0 && word[i + 1] == pair.1 {
                        merged.push(new_id);
                        i += 2;
                    } else {
                        merged.push(word[i]);
                        i += 1;
                    }
                }
                // add the new contribution back
                for p in merged.windows(2) {
                    let p = (p[0], p[1]);
                    *pair_counts.entry(p).or_insert(0) += count;
                    pair_words.entry(p).or_default().insert(w_idx);
                }
                words[w_idx].0 = merged;
            }
            pair_counts.remove(&pair);
        }

        // construct the tokenizer: mergeable ranks then special tokens at the end
        let mergeable_ranks: HashMap<Vec<u8>, u32> = token_bytes
            .into_iter()
            .enumerate()
            .map(|(rank, bytes)| (bytes, rank as u32))
            .collect();
        let tokens_offset = mergeable_ranks.len() as u32;
        let special_tokens: HashMap<String, u32> = SPECIAL_TOKENS
            .iter()
            .enumerate()
            .map(|(i, name)| (name.to_string(), tokens_offset + i as u32))
            .collect();
        Tokenizer::new(SPLIT_PATTERN, mergeable_ranks, special_tokens, "<|bos|>")
    }

    /// Raw bytes of a single (non-special) token.
    pub fn decode_single_token_bytes(&self, id: u32) -> Option<&[u8]> {
        self.decoder.get(&id).map(|b| b.as_slice())
    }

    /// Number of bytes of each token id (0 for special tokens), as used by the
    /// bits-per-byte evaluation. Uses the raw bytes of the token: decoding to a
    /// string first corrupts tokens that are not valid standalone UTF-8
    /// (e.g. the raw bytes >= 0x80).
    pub fn token_bytes(&self) -> Vec<i32> {
        (0..self.get_vocab_size() as u32)
            .map(|id| {
                if self.special_tokens_decoder.contains_key(&id) {
                    0 // special tokens are not counted
                } else {
                    self.decode_single_token_bytes(id)
                        .map_or(0, |b| b.len() as i32)
                }
            })
            .collect()
    }
}

impl super::engine::EngineTokenizer for Tokenizer {
    fn encode_special(&self, text: &str) -> u32 {
        Tokenizer::encode_special(self, text)
    }
    fn get_bos_token_id(&self) -> u32 {
        Tokenizer::get_bos_token_id(self)
    }
    fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_ordinary(text)
    }
    fn decode(&self, tokens: &[u32]) -> String {
        Tokenizer::decode(self, tokens)
    }
}

// -----------------------------------------------------------------------------
// nanochat-specific convenience functions

pub fn get_tokenizer() -> Result<Tokenizer> {
    let base_dir = get_base_dir();
    let tokenizer_dir = base_dir.join("tokenizer");
    Tokenizer::from_directory(&tokenizer_dir)
}
