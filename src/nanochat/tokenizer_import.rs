//! Readers for the tokenizer files that nanochat checkpoints ship with, so they
//! load directly without a conversion step:
//!
//! - `tokenizer.pkl`: the reference's native format, a pickled tiktoken Encoding
//! - a HuggingFace `tokenizers` `tokenizer.json` exported from a nanochat
//!   tokenizer (BPE model + ByteLevel pre-tokenizer), as found in HF-format repos
//!
//! Both reduce to the same three things: the split pattern, the mergeable ranks
//! (token bytes -> token id) and the special tokens.

use std::collections::HashMap;

use candle_core::Result;
use serde_json::Value;

use super::tokenizer::SPLIT_PATTERN;

/// (pattern, mergeable_ranks, special_tokens)
pub type TokenizerParts = (String, HashMap<Vec<u8>, u32>, HashMap<String, u32>);

fn msg(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

// -----------------------------------------------------------------------------
// tokenizer.pkl: a minimal pickle reader. It only builds plain data (strings,
// bytes, ints, dicts, tuples) and records which class the object claims to be;
// nothing in the file is ever executed.

#[derive(Clone, Debug)]
enum Pickled {
    None,
    Mark,
    Int(i64),
    Str(String),
    Bytes(Vec<u8>),
    /// (only ever the constructor arguments of the object, which are not needed)
    Tuple,
    Dict(Vec<(Pickled, Pickled)>),
    Global(String, String),
    Object { class: Box<Pickled>, state: Option<Box<Pickled>> },
}

struct PickleReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> PickleReader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&end| end <= self.data.len());
        let end = end.ok_or_else(|| msg("truncated tokenizer pickle"))?;
        let bytes = &self.data[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    fn uint(&mut self, n: usize) -> Result<usize> {
        let bytes = self.take(n)?;
        Ok(bytes.iter().rev().fold(0usize, |acc, &b| (acc << 8) | b as usize))
    }

    fn line(&mut self) -> Result<String> {
        let rest = &self.data[self.pos..];
        let len = rest.iter().position(|&b| b == b'\n').ok_or_else(|| msg("truncated tokenizer pickle"))?;
        let line = String::from_utf8(rest[..len].to_vec()).map_err(msg)?;
        self.pos += len + 1;
        Ok(line)
    }

    fn string(&mut self, len: usize) -> Result<Pickled> {
        Ok(Pickled::Str(String::from_utf8(self.take(len)?.to_vec()).map_err(msg)?))
    }
}

fn pop(stack: &mut Vec<Pickled>) -> Result<Pickled> {
    stack.pop().ok_or_else(|| msg("malformed tokenizer pickle: stack underflow"))
}

/// Pop everything above the topmost MARK.
fn pop_mark(stack: &mut Vec<Pickled>) -> Result<Vec<Pickled>> {
    let mark = stack
        .iter()
        .rposition(|v| matches!(v, Pickled::Mark))
        .ok_or_else(|| msg("malformed tokenizer pickle: no mark"))?;
    let items = stack.split_off(mark + 1);
    stack.pop(); // the mark itself
    Ok(items)
}

fn set_items(stack: &mut [Pickled], items: Vec<Pickled>) -> Result<()> {
    match stack.last_mut() {
        Some(Pickled::Dict(entries)) => {
            let mut items = items.into_iter();
            while let (Some(key), Some(value)) = (items.next(), items.next()) {
                entries.push((key, value));
            }
            Ok(())
        }
        _ => Err(msg("malformed tokenizer pickle: SETITEMS without a dict")),
    }
}

fn unpickle(data: &[u8]) -> Result<Pickled> {
    let mut r = PickleReader { data, pos: 0 };
    let mut stack: Vec<Pickled> = Vec::new();
    let mut memo: Vec<Pickled> = Vec::new();
    loop {
        let opcode = r.take(1)?[0];
        match opcode {
            0x80 => drop(r.take(1)?),  // PROTO
            0x95 => drop(r.take(8)?),  // FRAME
            b'.' => return pop(&mut stack), // STOP
            b'N' => stack.push(Pickled::None),
            b'(' => stack.push(Pickled::Mark),
            b'K' => stack.push(Pickled::Int(r.uint(1)? as i64)), // BININT1
            b'M' => stack.push(Pickled::Int(r.uint(2)? as i64)), // BININT2
            b'J' => stack.push(Pickled::Int(r.uint(4)? as u32 as i32 as i64)), // BININT
            0x8a => {
                // LONG1: little-endian two's complement
                let len = r.uint(1)?;
                let bytes = r.take(len)?;
                if len > 8 {
                    return Err(msg("unsupported tokenizer pickle: integer too large"));
                }
                let mut value = bytes.iter().rev().fold(0i64, |acc, &b| (acc << 8) | b as i64);
                if len > 0 && len < 8 && bytes[len - 1] & 0x80 != 0 {
                    value -= 1i64 << (8 * len);
                }
                stack.push(Pickled::Int(value));
            }
            0x8c => { let n = r.uint(1)?; stack.push(r.string(n)?) } // SHORT_BINUNICODE
            b'X' => { let n = r.uint(4)?; stack.push(r.string(n)?) } // BINUNICODE
            0x8d => { let n = r.uint(8)?; stack.push(r.string(n)?) } // BINUNICODE8
            b'C' => { let n = r.uint(1)?; stack.push(Pickled::Bytes(r.take(n)?.to_vec())) } // SHORT_BINBYTES
            b'B' => { let n = r.uint(4)?; stack.push(Pickled::Bytes(r.take(n)?.to_vec())) } // BINBYTES
            0x8e => { let n = r.uint(8)?; stack.push(Pickled::Bytes(r.take(n)?.to_vec())) } // BINBYTES8
            b')' => stack.push(Pickled::Tuple),
            0x85 | 0x86 | 0x87 => {
                // TUPLE1, TUPLE2, TUPLE3
                let n = (opcode - 0x84) as usize;
                if stack.len() < n {
                    return Err(msg("malformed tokenizer pickle: stack underflow"));
                }
                stack.truncate(stack.len() - n);
                stack.push(Pickled::Tuple);
            }
            b't' => { pop_mark(&mut stack)?; stack.push(Pickled::Tuple) }
            b'}' => stack.push(Pickled::Dict(Vec::new())),
            b'u' => { let items = pop_mark(&mut stack)?; set_items(&mut stack, items)? } // SETITEMS
            b's' => {
                // SETITEM
                let value = pop(&mut stack)?;
                let key = pop(&mut stack)?;
                set_items(&mut stack, vec![key, value])?;
            }
            b'c' => { let module = r.line()?; let name = r.line()?; stack.push(Pickled::Global(module, name)) }
            0x93 => {
                // STACK_GLOBAL
                match (pop(&mut stack)?, pop(&mut stack)?) {
                    (Pickled::Str(name), Pickled::Str(module)) => stack.push(Pickled::Global(module, name)),
                    _ => return Err(msg("malformed tokenizer pickle: STACK_GLOBAL")),
                }
            }
            0x81 => {
                // NEWOBJ: an instance of the class, without running anything
                let _args = pop(&mut stack)?;
                let class = pop(&mut stack)?;
                stack.push(Pickled::Object { class: Box::new(class), state: None });
            }
            b'b' => {
                // BUILD: attach the state to the object
                let new_state = pop(&mut stack)?;
                match stack.last_mut() {
                    Some(Pickled::Object { state, .. }) => *state = Some(Box::new(new_state)),
                    _ => return Err(msg("malformed tokenizer pickle: BUILD without an object")),
                }
            }
            0x94 => memo.push(stack.last().cloned().ok_or_else(|| msg("malformed tokenizer pickle"))?), // MEMOIZE
            b'h' | b'j' => {
                // BINGET, LONG_BINGET: only plain values are safe to copy out of the memo
                let index = r.uint(if opcode == b'h' { 1 } else { 4 })?;
                match memo.get(index) {
                    Some(value @ (Pickled::Int(_) | Pickled::Str(_) | Pickled::Bytes(_) | Pickled::None)) => {
                        stack.push(value.clone())
                    }
                    _ => return Err(msg("unsupported tokenizer pickle: shared container")),
                }
            }
            other => return Err(msg(format!("unsupported opcode 0x{other:02x} in tokenizer pickle"))),
        }
    }
}

/// Read the reference's tokenizer.pkl (a pickled tiktoken Encoding).
pub fn from_tiktoken_pickle(data: &[u8]) -> Result<TokenizerParts> {
    let state = match unpickle(data)? {
        Pickled::Object { class, state: Some(state) }
            if matches!(&*class, Pickled::Global(module, name) if module == "tiktoken.core" && name == "Encoding") =>
        {
            *state
        }
        _ => return Err(msg("tokenizer.pkl does not hold a tiktoken Encoding")),
    };
    let Pickled::Dict(entries) = state else {
        return Err(msg("tokenizer.pkl holds a registered tiktoken encoding, not a custom one"));
    };
    let mut pattern = None;
    let mut mergeable_ranks = HashMap::new();
    let mut special_tokens = HashMap::new();
    let rank = |v: Pickled| -> Result<u32> {
        match v {
            Pickled::Int(i) => u32::try_from(i).map_err(msg),
            _ => Err(msg("malformed tokenizer.pkl: token id is not an integer")),
        }
    };
    for (key, value) in entries {
        match (key, value) {
            (Pickled::Str(key), Pickled::Str(value)) if key == "pat_str" => pattern = Some(value),
            (Pickled::Str(key), Pickled::Dict(items)) if key == "mergeable_ranks" => {
                for (token, id) in items {
                    let Pickled::Bytes(token) = token else {
                        return Err(msg("malformed tokenizer.pkl: token is not bytes"));
                    };
                    mergeable_ranks.insert(token, rank(id)?);
                }
            }
            (Pickled::Str(key), Pickled::Dict(items)) if key == "special_tokens" => {
                for (token, id) in items {
                    let Pickled::Str(token) = token else {
                        return Err(msg("malformed tokenizer.pkl: special token is not a string"));
                    };
                    special_tokens.insert(token, rank(id)?);
                }
            }
            _ => {} // name etc.
        }
    }
    let pattern = pattern.ok_or_else(|| msg("malformed tokenizer.pkl: no pat_str"))?;
    Ok((pattern, mergeable_ranks, special_tokens))
}

// -----------------------------------------------------------------------------
// HuggingFace tokenizers format

/// Inverse of the GPT-2 bytes_to_unicode table used by the ByteLevel pre-tokenizer:
/// printable latin-1 bytes map to themselves, the rest to U+0100 onwards.
fn byte_level_char_to_byte() -> HashMap<char, u8> {
    let printable = |b: u32| (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
    let mut table = HashMap::with_capacity(256);
    let mut n = 0;
    for b in 0..256u32 {
        let c = if printable(b) {
            b
        } else {
            n += 1;
            256 + n - 1
        };
        table.insert(char::from_u32(c).unwrap(), b as u8);
    }
    table
}

/// Read a HuggingFace tokenizer.json holding a byte-level BPE tokenizer whose
/// token ids are the tiktoken ranks (as exported from nanochat tokenizers).
pub fn from_hf_json(tok: &Value) -> Result<TokenizerParts> {
    let model = &tok["model"];
    if model["type"] != "BPE" {
        return Err(msg(format!("expected a BPE tokenizer model, got {}", model["type"])));
    }
    let pre = &tok["pre_tokenizer"];
    let steps: Vec<&Value> = match pre["pretokenizers"].as_array() {
        Some(steps) => steps.iter().collect(),
        None => vec![pre],
    };
    if !steps.iter().any(|s| s["type"] == "ByteLevel") {
        return Err(msg("expected a ByteLevel pre-tokenizer in tokenizer.json"));
    }
    let pattern = steps
        .iter()
        .find(|s| s["type"] == "Split")
        .and_then(|s| s["pattern"]["Regex"].as_str())
        .unwrap_or(SPLIT_PATTERN)
        .to_string();

    let char_to_byte = byte_level_char_to_byte();
    let vocab = model["vocab"].as_object().ok_or_else(|| msg("tokenizer.json has no vocab"))?;
    let mut mergeable_ranks = HashMap::with_capacity(vocab.len());
    for (token, id) in vocab {
        let bytes: Option<Vec<u8>> = token.chars().map(|c| char_to_byte.get(&c).copied()).collect();
        let bytes = bytes.ok_or_else(|| msg(format!("token {token:?} is not byte-level encoded")))?;
        let id = id.as_u64().ok_or_else(|| msg("token id is not an integer"))? as u32;
        if mergeable_ranks.insert(bytes, id).is_some() {
            return Err(msg(format!("two vocab entries map to the same bytes as {token:?}")));
        }
    }
    let mut special_tokens = HashMap::new();
    for added in tok["added_tokens"].as_array().into_iter().flatten() {
        let content = added["content"].as_str().ok_or_else(|| msg("added token without content"))?;
        let id = added["id"].as_u64().ok_or_else(|| msg("added token without id"))? as u32;
        special_tokens.insert(content.to_string(), id);
    }
    Ok((pattern, mergeable_ranks, special_tokens))
}

/// Sanity checks shared by both imports: ranks are the token ids, the 256 raw
/// bytes are all present, and the special tokens follow the mergeable ranks.
pub fn validate(parts: &TokenizerParts) -> Result<()> {
    let (_, mergeable_ranks, special_tokens) = parts;
    let n = mergeable_ranks.len() as u32;
    let mut seen = vec![false; n as usize];
    for &rank in mergeable_ranks.values() {
        if rank >= n || std::mem::replace(&mut seen[rank as usize], true) {
            return Err(msg("tokenizer ranks must be contiguous from 0"));
        }
    }
    if !(0..=255u8).all(|b| mergeable_ranks.contains_key(&[b][..])) {
        return Err(msg("tokenizer is missing raw byte tokens"));
    }
    if special_tokens.values().min().is_some_and(|&min| min != n) {
        return Err(msg("special tokens must follow the mergeable ranks"));
    }
    Ok(())
}
