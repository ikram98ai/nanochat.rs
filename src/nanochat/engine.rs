//! Engine for efficient inference of our models.
//!
//! Everything works around token sequences:
//! - The user can send token sequences to the engine
//! - The engine returns the next token
//!
//! Notes:
//! - The engine knows nothing about tokenization, it's purely token id sequences.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;

use candle_core::{DType, Device, Result, Tensor};
use rand::rngs::StdRng;
use rand::SeedableRng;

use super::gpt::{argmax_1d, multinomial, softmax_1d, GPTConfig, GPT};

// -----------------------------------------------------------------------------
// Calculator tool helpers

/// Python-like numeric value: int ops stay int, division yields float.
/// Ints are i128: wide enough for calculator-style math; anything beyond
/// (Python would promote to bignum) evaluates to None instead.
#[derive(Clone, Copy, Debug)]
enum PyNum {
    Int(i128),
    Float(f64),
}

impl PyNum {
    fn as_f64(self) -> f64 {
        match self {
            PyNum::Int(i) => i as f64,
            PyNum::Float(f) => f,
        }
    }

    fn add(self, rhs: Self) -> Option<Self> {
        match (self, rhs) {
            (PyNum::Int(a), PyNum::Int(b)) => a.checked_add(b).map(PyNum::Int),
            (a, b) => Some(PyNum::Float(a.as_f64() + b.as_f64())),
        }
    }

    fn sub(self, rhs: Self) -> Option<Self> {
        match (self, rhs) {
            (PyNum::Int(a), PyNum::Int(b)) => a.checked_sub(b).map(PyNum::Int),
            (a, b) => Some(PyNum::Float(a.as_f64() - b.as_f64())),
        }
    }

    fn mul(self, rhs: Self) -> Option<Self> {
        match (self, rhs) {
            (PyNum::Int(a), PyNum::Int(b)) => a.checked_mul(b).map(PyNum::Int),
            (a, b) => Some(PyNum::Float(a.as_f64() * b.as_f64())),
        }
    }

    /// True division: always float
    fn div(self, rhs: Self) -> Option<Self> {
        let b = rhs.as_f64();
        if b == 0.0 {
            return None; // ZeroDivisionError
        }
        Some(PyNum::Float(self.as_f64() / b))
    }

    /// Floor division: int // int stays int; floors toward negative infinity
    fn floordiv(self, rhs: Self) -> Option<Self> {
        match (self, rhs) {
            (PyNum::Int(a), PyNum::Int(b)) => {
                if b == 0 {
                    return None;
                }
                let q = a.checked_div(b)?;
                let r = a.checked_rem(b)?;
                let q = if r != 0 && (r < 0) != (b < 0) { q.checked_sub(1)? } else { q };
                Some(PyNum::Int(q))
            }
            (a, b) => {
                let b = b.as_f64();
                if b == 0.0 {
                    return None;
                }
                Some(PyNum::Float((a.as_f64() / b).floor()))
            }
        }
    }

    fn neg(self) -> Option<Self> {
        match self {
            PyNum::Int(i) => i.checked_neg().map(PyNum::Int),
            PyNum::Float(f) => Some(PyNum::Float(-f)),
        }
    }

    /// Format like Python str(): ints plain; floats with a decimal point, or in
    /// scientific notation when the decimal exponent is < -4 or >= 16.
    fn to_python_str(self) -> String {
        match self {
            PyNum::Int(i) => i.to_string(),
            PyNum::Float(f) => {
                if f.is_nan() {
                    return "nan".into();
                }
                if f.is_infinite() {
                    return if f > 0.0 { "inf".into() } else { "-inf".into() };
                }
                if f == 0.0 {
                    return "0.0".into();
                }
                // shortest scientific form, e.g. "1.5e16", "-1e-5"
                let sci = format!("{f:e}");
                let (mantissa, exp) = sci.split_once('e').unwrap();
                let exp: i32 = exp.parse().unwrap();
                if !(-4..16).contains(&exp) {
                    let sign = if exp < 0 { '-' } else { '+' };
                    format!("{mantissa}e{sign}{:02}", exp.abs())
                } else if f == f.trunc() {
                    format!("{f:.1}")
                } else {
                    format!("{f}")
                }
            }
        }
    }
}

/// Result of evaluating a calculator expression: a number or a string.
#[derive(Clone, Debug)]
enum PyValue {
    Num(PyNum),
    Str(String),
}

impl PyValue {
    fn as_num(&self) -> Option<PyNum> {
        match self {
            PyValue::Num(n) => Some(*n),
            PyValue::Str(_) => None, // TypeError in Python
        }
    }

    /// Format like Python str()
    fn to_python_str(&self) -> String {
        match self {
            PyValue::Num(n) => n.to_python_str(),
            PyValue::Str(s) => s.clone(),
        }
    }
}

/// Recursive-descent evaluator covering the expressions the calculator allows:
/// + - * / // ( ) on int/float literals, plus string literals (with implicit
/// concatenation) and chained string methods, of which .count() and the no-arg
/// case-changing/stripping ones are supported.
struct Evaluator<'a> {
    chars: &'a [char],
    pos: usize,
}

impl<'a> Evaluator<'a> {
    fn eval(expr: &str) -> Option<PyValue> {
        let chars: Vec<char> = expr.chars().collect();
        let mut e = Evaluator { chars: &chars, pos: 0 };
        let v = e.expression()?;
        e.skip_ws();
        if e.pos != e.chars.len() {
            return None; // trailing garbage => SyntaxError
        }
        Some(v)
    }

    fn skip_ws(&mut self) {
        while self.chars.get(self.pos).is_some_and(|c| c.is_whitespace()) {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.skip_ws();
        self.chars.get(self.pos).copied()
    }

    fn expression(&mut self) -> Option<PyValue> {
        let mut v = self.term()?;
        loop {
            match self.peek() {
                Some('+') => {
                    self.pos += 1;
                    v = PyValue::Num(v.as_num()?.add(self.term()?.as_num()?)?);
                }
                Some('-') => {
                    self.pos += 1;
                    v = PyValue::Num(v.as_num()?.sub(self.term()?.as_num()?)?);
                }
                _ => return Some(v),
            }
        }
    }

    fn term(&mut self) -> Option<PyValue> {
        let mut v = self.factor()?;
        loop {
            match self.peek() {
                Some('*') => {
                    self.pos += 1;
                    v = PyValue::Num(v.as_num()?.mul(self.factor()?.as_num()?)?);
                }
                Some('/') => {
                    self.pos += 1;
                    if self.chars.get(self.pos) == Some(&'/') {
                        self.pos += 1;
                        v = PyValue::Num(v.as_num()?.floordiv(self.factor()?.as_num()?)?);
                    } else {
                        v = PyValue::Num(v.as_num()?.div(self.factor()?.as_num()?)?);
                    }
                }
                _ => return Some(v),
            }
        }
    }

    fn factor(&mut self) -> Option<PyValue> {
        match self.peek() {
            Some('-') => {
                self.pos += 1;
                Some(PyValue::Num(self.factor()?.as_num()?.neg()?))
            }
            Some('+') => {
                self.pos += 1;
                Some(PyValue::Num(self.factor()?.as_num()?))
            }
            _ => self.primary(),
        }
    }

    fn primary(&mut self) -> Option<PyValue> {
        let v = match self.peek()? {
            '(' => {
                self.pos += 1;
                let v = self.expression()?;
                if self.peek() != Some(')') {
                    return None;
                }
                self.pos += 1;
                v
            }
            '\'' | '"' => {
                // adjacent string literals concatenate into a single atom
                let mut s = self.string_literal()?;
                while matches!(self.peek(), Some('\'') | Some('"')) {
                    s.push_str(&self.string_literal()?);
                }
                PyValue::Str(s)
            }
            c if c.is_ascii_digit() || c == '.' => return self.number(),
            _ => return None,
        };
        self.postfix(v)
    }

    /// Chained method calls on string values.
    fn postfix(&mut self, mut v: PyValue) -> Option<PyValue> {
        while self.peek() == Some('.') {
            self.pos += 1;
            let s = match &v {
                PyValue::Str(s) => s.clone(),
                PyValue::Num(_) => return None, // AttributeError in Python
            };
            let start = self.pos;
            while self.chars.get(self.pos).is_some_and(|c| c.is_ascii_alphabetic()) {
                self.pos += 1;
            }
            let method: String = self.chars[start..self.pos].iter().collect();
            if self.chars.get(self.pos) != Some(&'(') {
                return None;
            }
            self.pos += 1;
            v = match method.as_str() {
                "count" => {
                    let needle = match self.expression()? {
                        PyValue::Str(needle) => needle,
                        PyValue::Num(_) => return None, // TypeError in Python
                    };
                    // non-overlapping occurrences; empty needle counts gaps
                    let count = if needle.is_empty() {
                        s.chars().count() + 1
                    } else {
                        s.matches(&needle).count()
                    };
                    PyValue::Num(PyNum::Int(count as i128))
                }
                // no-arg string methods (other str methods => None)
                "upper" => PyValue::Str(s.to_uppercase()),
                "lower" => PyValue::Str(s.to_lowercase()),
                "strip" => PyValue::Str(s.trim().to_string()),
                "lstrip" => PyValue::Str(s.trim_start().to_string()),
                "rstrip" => PyValue::Str(s.trim_end().to_string()),
                _ => return None,
            };
            if self.peek() != Some(')') {
                return None;
            }
            self.pos += 1;
        }
        Some(v)
    }

    fn string_literal(&mut self) -> Option<String> {
        let quote = self.peek()?;
        if quote != '\'' && quote != '"' {
            return None;
        }
        self.pos += 1;
        let start = self.pos;
        while self.chars.get(self.pos).is_some_and(|&c| c != quote) {
            self.pos += 1;
        }
        if self.pos >= self.chars.len() {
            return None; // unterminated string
        }
        let s: String = self.chars[start..self.pos].iter().collect();
        self.pos += 1;
        Some(s)
    }

    fn number(&mut self) -> Option<PyValue> {
        self.skip_ws();
        let start = self.pos;
        let mut has_dot = false;
        while let Some(&c) = self.chars.get(self.pos) {
            if c.is_ascii_digit() {
                self.pos += 1;
            } else if c == '.' && !has_dot {
                has_dot = true;
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return None;
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        if has_dot {
            text.parse().ok().map(|f| PyValue::Num(PyNum::Float(f)))
        } else {
            // leading zeros in decimal integer literals are not permitted
            if text.len() > 1 && text.starts_with('0') && !text.chars().all(|c| c == '0') {
                return None;
            }
            text.parse().ok().map(|i| PyValue::Num(PyNum::Int(i)))
        }
    }
}

/// Evaluate a Python expression safely.
/// Supports both math expressions and string operations like .count().
/// Returns the result formatted with Python str() semantics, or None.
pub fn use_calculator(expr: &str) -> Option<String> {
    // Remove commas from numbers
    let expr = expr.replace(',', "");

    // Check if it's a pure math expression
    if expr.chars().all(|x| "0123456789*+-/.() ".contains(x)) {
        if expr.contains("**") {
            // disallow power operator
            return None;
        }
        return Evaluator::eval(&expr).map(|v| v.to_python_str());
    }

    // Check if it's a string operation we support
    // Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    let allowed_chars =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ ";
    if !expr.chars().all(|x| allowed_chars.contains(x)) {
        return None;
    }

    // Disallow dangerous patterns
    let dangerous_patterns = [
        "__", "import", "exec", "eval", "compile", "open", "file", "input", "raw_input",
        "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr", "hasattr",
    ];
    let expr_lower = expr.to_lowercase();
    if dangerous_patterns.iter().any(|p| expr_lower.contains(p)) {
        return None;
    }

    // Only allow .count() method for now (can expand later)
    if !expr.contains(".count(") {
        return None;
    }

    Evaluator::eval(&expr).map(|v| v.to_python_str())
}

// -----------------------------------------------------------------------------

/// KV Cache for inference.
///
/// - Tensors are (B, T, H, D), one per layer
/// - The attention call updates the cache in-place
/// - Position is tracked uniformly across batch elements
pub struct KVCache {
    pub batch_size: usize,
    pub max_seq_len: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    k_cache: Vec<Tensor>,
    v_cache: Vec<Tensor>,
    cache_seqlens: Cell<usize>,
    /// Previous token's normalized embedding for smear (set by model forward pass)
    pub prev_embedding: RefCell<Option<Tensor>>,
}

impl KVCache {
    pub fn new(
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        num_layers: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        // Pre-allocate cache tensors
        let mut k_cache = Vec::with_capacity(num_layers);
        let mut v_cache = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k_cache.push(Tensor::zeros((batch_size, seq_len, num_heads, head_dim), dtype, device)?);
            v_cache.push(Tensor::zeros((batch_size, seq_len, num_heads, head_dim), dtype, device)?);
        }
        Ok(Self {
            batch_size,
            max_seq_len: seq_len,
            n_layers: num_layers,
            n_heads: num_heads,
            head_dim,
            k_cache,
            v_cache,
            cache_seqlens: Cell::new(0),
            prev_embedding: RefCell::new(None),
        })
    }

    /// Reset cache to empty state.
    pub fn reset(&self) {
        self.cache_seqlens.set(0);
        *self.prev_embedding.borrow_mut() = None;
    }

    /// Get current position (uniform across batch elements).
    pub fn get_pos(&self) -> usize {
        self.cache_seqlens.get()
    }

    /// Return (k_cache, v_cache) tensors for a specific layer.
    pub fn get_layer_cache(&self, layer_idx: usize) -> (&Tensor, &Tensor) {
        (&self.k_cache[layer_idx], &self.v_cache[layer_idx])
    }

    /// Advance the cache position by num_tokens.
    pub fn advance(&self, num_tokens: usize) {
        self.cache_seqlens.set(self.cache_seqlens.get() + num_tokens);
    }

    /// Copy cached KV from another cache into this one.
    /// Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
    pub fn prefill(&self, other: &KVCache) -> Result<()> {
        assert!(self.get_pos() == 0, "Cannot prefill a non-empty KV cache");
        assert!(
            self.n_layers == other.n_layers
                && self.n_heads == other.n_heads
                && self.head_dim == other.head_dim
        );
        assert!(self.max_seq_len >= other.max_seq_len);
        let other_pos = other.get_pos();
        for layer in 0..self.n_layers {
            let src_k = other.k_cache[layer].narrow(1, 0, other_pos)?;
            let src_v = other.v_cache[layer].narrow(1, 0, other_pos)?;
            // broadcast over the batch dim (prefill is usually batch=1)
            let shape = (self.batch_size, other_pos, self.n_heads, self.head_dim);
            self.k_cache[layer].slice_set(&src_k.broadcast_as(shape)?.contiguous()?, 1, 0)?;
            self.v_cache[layer].slice_set(&src_v.broadcast_as(shape)?.contiguous()?, 1, 0)?;
        }
        self.cache_seqlens.set(other_pos);
        // Copy smear state: expand batch=1 prev_embedding to num_samples
        if let Some(prev) = other.prev_embedding.borrow().as_ref() {
            let (_b, t, c) = prev.dims3()?;
            let expanded = prev.broadcast_as((self.batch_size, t, c))?.contiguous()?;
            *self.prev_embedding.borrow_mut() = Some(expanded);
        }
        Ok(())
    }
}

// -----------------------------------------------------------------------------

/// Sample a single next token per row from logits of shape (B, vocab_size).
pub fn sample_next_token(
    logits: &Tensor,
    rng: &mut StdRng,
    temperature: f64,
    top_k: Option<usize>,
) -> Result<Vec<u32>> {
    assert!(temperature >= 0.0, "temperature must be non-negative");
    let rows: Vec<Vec<f32>> = logits.to_dtype(DType::F32)?.to_vec2()?;
    let mut out = Vec::with_capacity(rows.len());
    for row in &rows {
        if temperature == 0.0 {
            out.push(argmax_1d(row));
            continue;
        }
        match top_k {
            Some(top_k) if top_k > 0 => {
                let k = top_k.min(row.len());
                // top-k indices, ties resolved to the lower index (stable sort)
                let mut idx: Vec<usize> = (0..row.len()).collect();
                idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
                let vals: Vec<f32> = idx[..k].iter().map(|&i| row[i] / temperature as f32).collect();
                let probs = softmax_1d(&vals);
                let choice = multinomial(&probs, rng) as usize;
                out.push(idx[choice] as u32);
            }
            _ => {
                let scaled: Vec<f32> = row.iter().map(|v| v / temperature as f32).collect();
                let probs = softmax_1d(&scaled);
                out.push(multinomial(&probs, rng));
            }
        }
    }
    Ok(out)
}

// -----------------------------------------------------------------------------

/// Per-row state tracking during generation
struct RowState {
    /// Current token sequence for this row
    current_tokens: Vec<u32>,
    /// Queue of tokens to force inject
    forced_tokens: VecDeque<u32>,
    /// Whether we are inside a python block
    in_python_block: bool,
    /// Tokens of the current python expression
    python_expr_tokens: Vec<u32>,
    /// Whether this row has completed generation
    completed: bool,
}

impl RowState {
    fn new(current_tokens: Vec<u32>) -> Self {
        Self {
            current_tokens,
            forced_tokens: VecDeque::new(),
            in_python_block: false,
            python_expr_tokens: Vec::new(),
            completed: false,
        }
    }
}

/// Model interface the engine needs (lets tests substitute a mock model).
pub trait EngineModel {
    fn config(&self) -> &GPTConfig;
    fn get_device(&self) -> &Device;
    fn forward(&self, idx: &Tensor, kv_cache: Option<&KVCache>) -> Result<Tensor>;
}

impl EngineModel for GPT {
    fn config(&self) -> &GPTConfig {
        &self.config
    }
    fn get_device(&self) -> &Device {
        GPT::get_device(self)
    }
    fn forward(&self, idx: &Tensor, kv_cache: Option<&KVCache>) -> Result<Tensor> {
        GPT::forward(self, idx, None, kv_cache, "mean")
    }
}

/// Tokenizer interface the engine needs for tool use.
pub trait EngineTokenizer {
    fn encode_special(&self, text: &str) -> u32;
    fn get_bos_token_id(&self) -> u32;
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, tokens: &[u32]) -> String;
}

pub struct Engine<'a, M: EngineModel, T: EngineTokenizer> {
    model: &'a M,
    tokenizer: &'a T, // needed for tool use
}

/// Streaming generator over (token_column, token_masks) pairs, one per step.
/// Masks are 1 for sampled tokens and 0 for forced (tool output) tokens.
pub struct Generate<'a, M: EngineModel, T: EngineTokenizer> {
    engine: &'a Engine<'a, M, T>,
    rng: StdRng,
    temperature: f64,
    top_k: Option<usize>,
    max_tokens: Option<usize>,
    num_samples: usize,
    kv_cache_decode: KVCache,
    row_states: Vec<RowState>,
    logits: Option<Tensor>,
    /// token column from the previous step, fed through the model lazily
    pending_ids: Option<Vec<u32>>,
    num_generated: usize,
    // special tokens coordinating the tool use state machine
    python_start: u32,
    python_end: u32,
    output_start: u32,
    output_end: u32,
    assistant_end: u32,
    bos: u32,
}

impl<'a, M: EngineModel, T: EngineTokenizer> Engine<'a, M, T> {
    pub fn new(model: &'a M, tokenizer: &'a T) -> Self {
        Self { model, tokenizer }
    }

    /// Single prefill of the prompt, then clone the KV cache across num_samples
    /// rows and stream sampled token columns.
    pub fn generate(
        &'a self,
        tokens: &[u32],
        num_samples: usize,
        max_tokens: Option<usize>,
        temperature: f64,
        top_k: Option<usize>,
        seed: u64,
    ) -> Result<Generate<'a, M, T>> {
        assert!(!tokens.is_empty(), "expecting non-empty list of tokens");
        let device = self.model.get_device();
        let dtype = DType::F32; // cuda would use bf16; everything else float32
        let rng = StdRng::seed_from_u64(seed);

        // Get the special tokens we need to coordinate the tool use state machine
        let python_start = self.tokenizer.encode_special("<|python_start|>");
        let python_end = self.tokenizer.encode_special("<|python_end|>");
        let output_start = self.tokenizer.encode_special("<|output_start|>");
        let output_end = self.tokenizer.encode_special("<|output_end|>");
        let assistant_end = self.tokenizer.encode_special("<|assistant_end|>"); // if sampled, ends row
        let bos = self.tokenizer.get_bos_token_id(); // if sampled, ends row

        // 1) Run a batch 1 prefill of the prompt tokens
        let m = self.model.config();
        let (num_heads, head_dim, num_layers) = (m.n_kv_head, m.n_embd / m.n_head, m.n_layer);
        let kv_cache_prefill =
            KVCache::new(1, num_heads, tokens.len(), head_dim, num_layers, device, dtype)?;
        let ids: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
        let ids = Tensor::from_vec(ids, (1, tokens.len()), device)?;
        let logits = self.model.forward(&ids, Some(&kv_cache_prefill))?;
        let (_, t, v) = logits.dims3()?;
        // (num_samples, vocab_size)
        let logits = logits
            .narrow(1, t - 1, 1)?
            .squeeze(1)?
            .broadcast_as((num_samples, v))?
            .contiguous()?;

        // 2) Replicate the KV cache for each sample/row
        let kv_length_hint = match max_tokens {
            Some(max_tokens) => tokens.len() + max_tokens,
            None => m.sequence_len,
        };
        let kv_cache_decode =
            KVCache::new(num_samples, num_heads, kv_length_hint, head_dim, num_layers, device, dtype)?;
        kv_cache_decode.prefill(&kv_cache_prefill)?;
        drop(kv_cache_prefill); // no need to keep this memory around

        // 3) Initialize states for each sample
        let row_states = (0..num_samples).map(|_| RowState::new(tokens.to_vec())).collect();

        // 4) The main generation loop runs in Generate::next
        Ok(Generate {
            engine: self,
            rng,
            temperature,
            top_k,
            max_tokens,
            num_samples,
            kv_cache_decode,
            row_states,
            logits: Some(logits),
            pending_ids: None,
            num_generated: 0,
            python_start,
            python_end,
            output_start,
            output_end,
            assistant_end,
            bos,
        })
    }

    /// Non-streaming batch generation that just returns the final token sequences.
    /// Returns (token sequences, masks). Terminal tokens (assistant_end, bos) are
    /// not included in the results.
    pub fn generate_batch(
        &'a self,
        tokens: &[u32],
        num_samples: usize,
        max_tokens: Option<usize>,
        temperature: f64,
        top_k: Option<usize>,
        seed: u64,
    ) -> Result<(Vec<Vec<u32>>, Vec<Vec<u8>>)> {
        let assistant_end = self.tokenizer.encode_special("<|assistant_end|>");
        let bos = self.tokenizer.get_bos_token_id();
        let mut results: Vec<Vec<u32>> = (0..num_samples).map(|_| tokens.to_vec()).collect();
        let mut masks: Vec<Vec<u8>> = (0..num_samples).map(|_| vec![0; tokens.len()]).collect();
        let mut completed = vec![false; num_samples];
        let stream = self.generate(tokens, num_samples, max_tokens, temperature, top_k, seed)?;
        for step in stream {
            let (token_column, token_masks) = step?;
            for (i, (&token, &mask)) in token_column.iter().zip(token_masks.iter()).enumerate() {
                if !completed[i] {
                    if token == assistant_end || token == bos {
                        completed[i] = true;
                    } else {
                        results[i].push(token);
                        masks[i].push(mask);
                    }
                }
            }
            // Stop if all rows are completed
            if completed.iter().all(|&c| c) {
                break;
            }
        }
        Ok((results, masks))
    }
}

impl<'a, M: EngineModel, T: EngineTokenizer> Iterator for Generate<'a, M, T> {
    type Item = Result<(Vec<u32>, Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        // Stop condition: we've reached max tokens
        if let Some(max_tokens) = self.max_tokens {
            if self.num_generated >= max_tokens {
                return None;
            }
        }
        // Stop condition: all rows are completed
        if self.row_states.iter().all(|s| s.completed) {
            return None;
        }
        match self.step() {
            Ok(column) => Some(Ok(column)),
            Err(e) => Some(Err(e)),
        }
    }
}

impl<'a, M: EngineModel, T: EngineTokenizer> Generate<'a, M, T> {
    fn step(&mut self) -> Result<(Vec<u32>, Vec<u8>)> {
        let device = self.engine.model.get_device();

        // Compute logits for the previous token column (deferred from the last step)
        if let Some(prev_column) = self.pending_ids.take() {
            let ids: Vec<i64> = prev_column.iter().map(|&t| t as i64).collect();
            let ids = Tensor::from_vec(ids, (self.num_samples, 1), device)?;
            let logits = self.engine.model.forward(&ids, Some(&self.kv_cache_decode))?;
            let t = logits.dim(1)?;
            self.logits = Some(logits.narrow(1, t - 1, 1)?.squeeze(1)?); // (B, vocab_size)
        }

        // Sample the next token for each row
        let logits = self.logits.as_ref().unwrap();
        let sampled_tokens = sample_next_token(logits, &mut self.rng, self.temperature, self.top_k)?;

        // Process each row: choose the next token, update state, optional tool use
        let mut token_column = Vec::with_capacity(self.num_samples); // next token id along each row
        let mut token_masks = Vec::with_capacity(self.num_samples); // sampled (1) or forced (0)?
        for (i, state) in self.row_states.iter_mut().enumerate() {
            // Select the next token in this row
            let forced = state.forced_tokens.pop_front(); // tokens waiting to be forced?
            token_masks.push(if forced.is_some() { 0u8 } else { 1u8 });
            let next_token = forced.unwrap_or(sampled_tokens[i]);
            token_column.push(next_token);
            // Update the state of this row to include the next token
            state.current_tokens.push(next_token);
            // On <|assistant_end|> or <|bos|>, mark the row as completed
            if next_token == self.assistant_end || next_token == self.bos {
                state.completed = true;
            }
            // Handle tool logic
            if next_token == self.python_start {
                state.in_python_block = true;
                state.python_expr_tokens.clear();
            } else if next_token == self.python_end && state.in_python_block {
                state.in_python_block = false;
                if !state.python_expr_tokens.is_empty() {
                    let expr = self.engine.tokenizer.decode(&state.python_expr_tokens);
                    if let Some(result) = use_calculator(&expr) {
                        let result_tokens = self.engine.tokenizer.encode(&result);
                        state.forced_tokens.push_back(self.output_start);
                        state.forced_tokens.extend(result_tokens);
                        state.forced_tokens.push_back(self.output_end);
                    }
                }
                state.python_expr_tokens.clear();
            } else if state.in_python_block {
                state.python_expr_tokens.push(next_token);
            }
        }

        self.num_generated += 1;
        // Logits for the next iteration are computed lazily at the start of the next step
        self.pending_ids = Some(token_column.clone());
        Ok((token_column, token_masks))
    }
}
