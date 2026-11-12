//! Regenerate the test data in tests/data from this implementation.
//!
//!     cargo run --release --example gen_test_data [-- OUT_DIR]
//!
//! tests/data holds the inputs (tiny seeded models, random tensors, synthetic
//! datasets, parquet shards, tokenizer files) and the expected outputs that the
//! tests in tests/parity_*.rs compare against. Everything is derived from fixed
//! seeds, so the directory can be rebuilt at any time and is not checked in.
//!
//! The expected outputs come from the code in this repo. The tests therefore
//! tell you whether behaviour changed since the data was generated: regenerate
//! on a known-good commit, change the code, run the tests. After an intended
//! change (e.g. to the architecture), regenerate. To check against the Python
//! reference instead, its fixture scripts write this same directory layout.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use parquet::data_type::{ByteArray, ByteArrayType, Int64Type};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{json, Map, Value};

use nanochat::chat_eval::{run_categorical_eval, run_generative_eval};
use nanochat::core_eval;
use nanochat::dataloader::{DataLoader, LoaderState, SftDataLoader};
use nanochat::engine::{use_calculator, Engine, EngineTokenizer, KVCache};
use nanochat::flash_attention::{flash_attn_func, flash_attn_with_kvcache};
use nanochat::gpt::{apply_rotary_emb, norm, GPTConfig, GPT};
use nanochat::loss_eval::evaluate_bpb;
use nanochat::optim::{MuonAdamW, ParamGroup};
use nanochat::pyrandom::PyRandom;
use nanochat::tasks::arc::ARC;
use nanochat::tasks::common::{render_mc, Slice, Task, TaskMixture};
use nanochat::tasks::gsm8k::{extract_answer, GSM8K};
use nanochat::tasks::hub;
use nanochat::tasks::humaneval::{extract_imports, extract_program, HumanEval};
use nanochat::tasks::mmlu::MMLU;
use nanochat::tasks::smoltalk::SmolTalk;
use nanochat::tasks::spellingbee::{SimpleSpelling, SpellingBee};
use nanochat::tokenizer::{Tokenizer, SPECIAL_TOKENS, SPLIT_PATTERN};

type Tensors = HashMap<String, Tensor>;

const CPU: &Device = &Device::Cpu;

// -----------------------------------------------------------------------------
// seeded random tensors

fn randn(rng: &mut StdRng, shape: &[usize], std: f32) -> Result<Tensor> {
    let dist = rand_distr::Normal::new(0f32, std).unwrap();
    let data: Vec<f32> = (0..shape.iter().product()).map(|_| rng.sample(dist)).collect();
    Ok(Tensor::from_vec(data, shape, CPU)?)
}

fn uniform(rng: &mut StdRng, shape: &[usize], lo: f32, hi: f32) -> Result<Tensor> {
    let data: Vec<f32> = (0..shape.iter().product()).map(|_| rng.random_range(lo..hi)).collect();
    Ok(Tensor::from_vec(data, shape, CPU)?)
}

/// Random token ids in [0, vocab_size)
fn randint(rng: &mut StdRng, shape: &[usize], vocab_size: usize) -> Result<Tensor> {
    let data: Vec<i64> =
        (0..shape.iter().product()).map(|_| rng.random_range(0..vocab_size as i64)).collect();
    Ok(Tensor::from_vec(data, shape, CPU)?)
}

/// Replace a random fraction p of the targets with -1 (ignore_index)
fn mask_targets(rng: &mut StdRng, targets: &Tensor, p: f32) -> Result<Tensor> {
    let mut data = targets.flatten_all()?.to_vec1::<i64>()?;
    for t in data.iter_mut() {
        if rng.random::<f32>() < p {
            *t = -1;
        }
    }
    Ok(Tensor::from_vec(data, targets.dims(), CPU)?)
}

/// Random weights for every parameter of a model, in sorted name order.
fn random_weights(
    config: &GPTConfig,
    seed: u64,
    rule: impl Fn(&str, &[usize], &mut StdRng) -> Result<Tensor>,
) -> Result<Tensors> {
    // init_weights provides the parameter names and shapes
    let template = GPT::init_weights(config, 64, CPU, 0)?;
    let mut names: Vec<&String> = template.keys().collect();
    names.sort();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut weights = HashMap::new();
    for name in names {
        weights.insert(name.clone(), rule(name, template[name].dims(), &mut rng)?);
    }
    Ok(weights)
}

fn build_model(config: &GPTConfig, weights: &Tensors) -> Result<GPT> {
    Ok(GPT::new(config.clone(), VarBuilder::from_tensors(weights.clone(), DType::F32, CPU))?)
}

/// A trainable copy of the model: parameters live in the returned VarMap.
fn build_train_model(config: &GPTConfig, weights: &Tensors) -> Result<(GPT, VarMap)> {
    let mut varmap = VarMap::new();
    let model = GPT::new(config.clone(), VarBuilder::from_varmap(&varmap, DType::F32, CPU))?;
    varmap.set(weights.iter())?;
    Ok((model, varmap))
}

fn named_vars(varmap: &VarMap) -> Vec<(String, Var)> {
    let vars = varmap.data().lock().unwrap();
    let mut named: Vec<(String, Var)> = vars.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    named.sort_by(|a, b| a.0.cmp(&b.0));
    named
}

// -----------------------------------------------------------------------------
// file helpers

fn write_json(path: &Path, value: &Value) -> Result<()> {
    std::fs::create_dir_all(path.parent().unwrap())?;
    std::fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}

fn write_jsonl(path: &Path, rows: &[Value]) -> Result<()> {
    std::fs::create_dir_all(path.parent().unwrap())?;
    let lines: Vec<String> = rows.iter().map(|r| r.to_string()).collect();
    std::fs::write(path, lines.join("\n") + "\n")?;
    Ok(())
}

/// One leaf column of a parquet row group.
enum Leaf {
    Str(Vec<String>),
    Int(Vec<i64>),
    /// one list of strings per row (a LIST of required elements, never empty)
    StrList(Vec<Vec<String>>),
}

fn byte_arrays<'a>(values: impl Iterator<Item = &'a String>) -> Vec<ByteArray> {
    values.map(|v| ByteArray::from(v.as_bytes().to_vec())).collect()
}

/// Write a parquet file with the given schema; each row group lists its leaf
/// columns in schema order.
fn write_parquet(path: &Path, schema: &str, row_groups: Vec<Vec<Leaf>>) -> Result<()> {
    std::fs::create_dir_all(path.parent().unwrap())?;
    let schema = Arc::new(parse_message_type(schema)?);
    let props = Arc::new(WriterProperties::builder().build());
    let mut writer = SerializedFileWriter::new(File::create(path)?, schema, props)?;
    for leaves in row_groups {
        let mut row_group = writer.next_row_group()?;
        for leaf in leaves {
            let mut column = row_group.next_column()?.expect("more leaves than schema columns");
            match leaf {
                Leaf::Str(values) => {
                    column.typed::<ByteArrayType>().write_batch(&byte_arrays(values.iter()), None, None)?;
                }
                Leaf::Int(values) => {
                    column.typed::<Int64Type>().write_batch(&values, None, None)?;
                }
                Leaf::StrList(lists) => {
                    // every element is defined; a repetition level of 0 starts a new row
                    let total: usize = lists.iter().map(|l| l.len()).sum();
                    let rep_levels: Vec<i16> = lists
                        .iter()
                        .flat_map(|l| (0..l.len()).map(|i| if i == 0 { 0 } else { 1 }))
                        .collect();
                    column.typed::<ByteArrayType>().write_batch(
                        &byte_arrays(lists.iter().flatten()),
                        Some(&vec![1i16; total]),
                        Some(&rep_levels),
                    )?;
                }
            }
            column.close()?;
        }
        row_group.close()?;
    }
    writer.close()?;
    Ok(())
}

fn strs(values: &Value) -> Vec<String> {
    values.as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect()
}

fn str_column(rows: &[Value], key: &str) -> Leaf {
    Leaf::Str(rows.iter().map(|r| r[key].as_str().unwrap().to_string()).collect())
}

/// Place parquet shards in the hub cache layout ({out}/task_data/...), so the
/// hub loader reads them without the network.
fn write_hub_cache(
    out: &Path,
    repo_id: &str,
    subset: &str,
    split: &str,
    schema: &str,
    shards: &[&[Value]],
    to_leaves: impl Fn(&[Value]) -> Vec<Leaf>,
) -> Result<()> {
    let shards_dir = out.join("task_data").join(repo_id.replace('/', "--")).join(subset).join(split);
    let mut filenames = Vec::new();
    for (shard_index, rows) in shards.iter().enumerate() {
        let filename = format!("{shard_index:05}.parquet");
        write_parquet(&shards_dir.join(&filename), schema, vec![to_leaves(rows)])?;
        filenames.push(filename);
    }
    std::fs::write(shards_dir.join("manifest.json"), serde_json::to_string(&filenames)?)?;
    Ok(())
}

// -----------------------------------------------------------------------------
// tokenizer files in the two foreign formats

fn pickle_str(out: &mut Vec<u8>, s: &str) {
    if s.len() < 256 {
        out.push(0x8c); // SHORT_BINUNICODE
        out.push(s.len() as u8);
    } else {
        out.push(b'X'); // BINUNICODE
        out.extend((s.len() as u32).to_le_bytes());
    }
    out.extend(s.as_bytes());
}

fn pickle_int(out: &mut Vec<u8>, i: u32) {
    if i < 256 {
        out.push(b'K'); // BININT1
        out.push(i as u8);
    } else if i < 65536 {
        out.push(b'M'); // BININT2
        out.extend((i as u16).to_le_bytes());
    } else {
        out.push(b'J'); // BININT
        out.extend((i as i32).to_le_bytes());
    }
}

/// The reference's tokenizer.pkl: a pickled tiktoken Encoding (protocol 4).
fn tiktoken_pickle(pattern: &str, ranks: &[Vec<u8>], specials: &[(String, u32)]) -> Vec<u8> {
    let mut out = vec![0x80, 4]; // PROTO 4
    pickle_str(&mut out, "tiktoken.core");
    pickle_str(&mut out, "Encoding");
    out.extend([0x93, b')', 0x81]); // STACK_GLOBAL, EMPTY_TUPLE, NEWOBJ
    out.extend([b'}', b'(']); // state dict: EMPTY_DICT, MARK
    pickle_str(&mut out, "name");
    pickle_str(&mut out, "rustbpe");
    pickle_str(&mut out, "pat_str");
    pickle_str(&mut out, pattern);
    pickle_str(&mut out, "mergeable_ranks");
    out.extend([b'}', b'(']);
    for (rank, token) in ranks.iter().enumerate() {
        assert!(token.len() < 256);
        out.extend([b'C', token.len() as u8]); // SHORT_BINBYTES
        out.extend(token);
        pickle_int(&mut out, rank as u32);
    }
    out.push(b'u'); // SETITEMS
    pickle_str(&mut out, "special_tokens");
    out.extend([b'}', b'(']);
    for (name, id) in specials {
        pickle_str(&mut out, name);
        pickle_int(&mut out, *id);
    }
    out.push(b'u'); // SETITEMS
    out.extend([b'u', b'b', b'.']); // SETITEMS (state), BUILD, STOP
    out
}

/// A HuggingFace tokenizers file, as HF-format nanochat repos ship: byte-level
/// BPE whose vocab maps each token (bytes spelled with the GPT-2
/// byte->unicode table) to its rank.
fn hf_tokenizer_json(pattern: &str, ranks: &[Vec<u8>], specials: &[(String, u32)]) -> Value {
    let printable =
        |b: u32| (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
    let mut byte_to_char = Vec::with_capacity(256);
    let mut n = 0;
    for b in 0..256u32 {
        byte_to_char.push(char::from_u32(if printable(b) { b } else { n += 1; 255 + n }).unwrap());
    }
    let vocab: Map<String, Value> = ranks
        .iter()
        .enumerate()
        .map(|(rank, token)| (token.iter().map(|&b| byte_to_char[b as usize]).collect(), json!(rank)))
        .collect();
    let added_tokens: Vec<Value> = specials
        .iter()
        .map(|(name, id)| json!({"id": id, "content": name, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}))
        .collect();
    json!({
        "version": "1.0",
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false},
        ]},
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "ignore_merges": true, "vocab": vocab, "merges": []},
    })
}

// -----------------------------------------------------------------------------
// model, engine and tokenizer data (model.safetensors, fixtures.safetensors,
// meta.json, tokenizer files)

/// The tiny model config exercising every path: GQA (4q/2kv heads), vocab
/// padding (199 -> 256), sliding windows (short=128 < seq), value embeddings
/// on layers 0/2/4, backout at layer 2.
fn tiny_config() -> GPTConfig {
    GPTConfig {
        sequence_len: 512,
        vocab_size: 199,
        n_layer: 5,
        n_head: 4,
        n_kv_head: 2,
        n_embd: 64,
        window_pattern: "SSL".to_string(),
    }
}

/// init_weights zeros all projections (c_proj, smear_lambda, ...) which would
/// make most code paths no-ops, so every parameter is randomized instead and
/// each feature contributes to the output.
fn tiny_weights(config: &GPTConfig) -> Result<Tensors> {
    let s = 3f32.sqrt() / (config.n_embd as f32).sqrt();
    random_weights(config, 1234, |name, shape, rng| {
        if name.contains("wte") {
            randn(rng, shape, 0.8)
        } else if name.contains("value_embeds") {
            randn(rng, shape, 0.5)
        } else if name.contains("lm_head") {
            randn(rng, shape, 0.02)
        } else if name.contains("resid_lambdas") {
            uniform(rng, shape, 0.9, 1.2)
        } else if name.contains("x0_lambdas") {
            uniform(rng, shape, 0.0, 0.3)
        } else if name.contains("smear_lambda") {
            Ok(Tensor::full(0.7f32, shape, CPU)?)
        } else if name.contains("backout_lambda") {
            Ok(Tensor::full(0.25f32, shape, CPU)?)
        } else if name.contains("smear_gate") || name.contains("ve_gate") {
            uniform(rng, shape, 0.0, 0.5)
        } else if name.contains("c_proj") {
            uniform(rng, shape, -0.5 * s, 0.5 * s)
        } else {
            uniform(rng, shape, -s, s) // c_q, c_k, c_v, c_fc
        }
    })
}

/// KV-cache inference: prefill + argmax decode steps
fn kv_decode_case(
    model: &GPT,
    fx: &mut Tensors,
    meta: &mut Map<String, Value>,
    prefix: &str,
    prompt_len: usize,
    cache_len: usize,
    n_steps: usize,
) -> Result<()> {
    let m = &model.config;
    let mut rng = StdRng::seed_from_u64(11);
    let prompt = randint(&mut rng, &[1, prompt_len], m.vocab_size)?;
    let kv = KVCache::new(1, m.n_kv_head, cache_len, m.n_embd / m.n_head, m.n_layer, CPU, DType::F32)?;
    let logits = model.forward(&prompt, None, Some(&kv), "mean")?;
    let mut last = logits.narrow(1, prompt_len - 1, 1)?.squeeze(1)?;
    fx.insert(format!("{prefix}_prompt"), prompt);
    fx.insert(format!("{prefix}_prefill_logits"), last.clone());
    let mut tokens = Vec::new();
    for _ in 0..n_steps {
        let next_tok = last.argmax(1)?.to_vec1::<u32>()?[0];
        tokens.push(next_tok);
        let ids = Tensor::from_vec(vec![next_tok as i64], (1, 1), CPU)?;
        last = model.forward(&ids, None, Some(&kv), "mean")?.narrow(1, 0, 1)?.squeeze(1)?;
    }
    fx.insert(format!("{prefix}_final_logits"), last);
    meta.insert(format!("{prefix}_tokens"), json!(tokens));
    Ok(())
}

/// The engine only needs special-token ids; map them above the model vocab.
struct SpecialOnlyTokenizer;

impl EngineTokenizer for SpecialOnlyTokenizer {
    fn encode_special(&self, text: &str) -> u32 {
        500 + SPECIAL_TOKENS.iter().position(|s| *s == text).unwrap() as u32
    }
    fn get_bos_token_id(&self) -> u32 {
        self.encode_special("<|bos|>")
    }
    fn encode(&self, _text: &str) -> Vec<u32> {
        unreachable!("not used in this fixture")
    }
    fn decode(&self, _tokens: &[u32]) -> String {
        unreachable!("not used in this fixture")
    }
}

const TOKENIZER_CORPUS: [&str; 5] = [
    "Hello world! This is a tiny training corpus for the nanochat tokenizer. ",
    "The quick brown fox jumps over the lazy dog. 0123456789 42 100 2048. ",
    "I'd've thought it'll be fine, isn't it? We're sure they've gone. ",
    "def main():\n    print('hello')\n    return 0\n",
    "Numbers: 1, 22, 333, 4444. Punctuation?! (parens) [brackets] {braces}. ",
];

fn gen_model_data(out: &Path) -> Result<(GPTConfig, Tensors, Tokenizer)> {
    let config = tiny_config();
    let weights = tiny_weights(&config)?;
    candle_core::safetensors::save(&weights, out.join("model.safetensors"))?;
    let model = build_model(&config, &weights)?;

    let mut fx: Tensors = HashMap::new();
    let mut meta = Map::new();
    meta.insert("config".into(), serde_json::to_value(&config)?);
    meta.insert("padded_vocab_size".into(), json!(weights["transformer.wte.weight"].dim(0)?));
    let window_sizes: Vec<[i64; 2]> = model.window_sizes.iter().map(|w| [w.0, w.1]).collect();
    meta.insert("window_sizes".into(), json!(window_sizes));
    meta.insert("num_flops_per_token".into(), json!(model.estimate_flops()));
    meta.insert("num_scaling_params".into(), json!(model.num_scaling_params()));
    meta.insert("rms_norm_eps".into(), json!(f32::EPSILON as f64));

    // Micro fixtures: rmsnorm, rotary, attention with/without windows and cache
    let mut g = StdRng::seed_from_u64(99);
    let x = randn(&mut g, &[2, 3, 32], 1.0)?;
    fx.insert("rmsnorm_out".into(), norm(&x)?);
    fx.insert("rmsnorm_in".into(), x);

    let (cos, sin) = model.rotary_cos_sin();
    let (cos, sin) = (cos.narrow(1, 0, 8)?, sin.narrow(1, 0, 8)?);
    let xr = randn(&mut g, &[1, 8, 2, 16], 1.0)?;
    fx.insert("rotary_out".into(), apply_rotary_emb(&xr, &cos, &sin)?);
    fx.insert("rotary_in".into(), xr);
    fx.insert("rotary_cos".into(), cos);
    fx.insert("rotary_sin".into(), sin);

    let q = randn(&mut g, &[1, 6, 4, 16], 1.0)?;
    let k = randn(&mut g, &[1, 6, 2, 16], 1.0)?;
    let v = randn(&mut g, &[1, 6, 2, 16], 1.0)?;
    fx.insert("attn_full_out".into(), flash_attn_func(&q, &k, &v, true, (512, 0))?);
    fx.insert("attn_win2_out".into(), flash_attn_func(&q, &k, &v, true, (2, 0))?);
    fx.insert("attn_q".into(), q);
    fx.insert("attn_k".into(), k);
    fx.insert("attn_v".into(), v);

    // KV-cache attention: prefill 12, chunk of 5, then single-token decode
    let cache = KVCache::new(1, 2, 20, 16, 1, CPU, DType::F32)?;
    for (name, t_new) in [("prefill", 12), ("chunk", 5), ("decode", 1)] {
        let q = randn(&mut g, &[1, t_new, 4, 16], 1.0)?;
        let k = randn(&mut g, &[1, t_new, 2, 16], 1.0)?;
        let v = randn(&mut g, &[1, t_new, 2, 16], 1.0)?;
        let y = flash_attn_with_kvcache(&q, &cache, 0, &k, &v, true, (4, 0))?;
        cache.advance(t_new);
        fx.insert(format!("kvattn_{name}_out"), y);
        fx.insert(format!("kvattn_q_{name}"), q);
        fx.insert(format!("kvattn_k_{name}"), k);
        fx.insert(format!("kvattn_v_{name}"), v);
    }
    let (k_cache, v_cache) = cache.get_layer_cache(0);
    fx.insert("kvattn_k_cache_final".into(), k_cache.copy()?);
    fx.insert("kvattn_v_cache_final".into(), v_cache.copy()?);

    // Full model forward, training mode (no KV cache)
    let mut g = StdRng::seed_from_u64(7);
    let ids1 = randint(&mut g, &[2, 16], config.vocab_size)?;
    fx.insert("fw1_logits".into(), model.forward(&ids1, None, None, "mean")?);
    let ids2 = randint(&mut g, &[1, 200], config.vocab_size)?; // windows bind: 128 < 200
    fx.insert("fw2_logits".into(), model.forward(&ids2, None, None, "mean")?);
    fx.insert("fw2_ids".into(), ids2);
    let targets = randint(&mut g, &[2, 16], config.vocab_size)?;
    let targets = mask_targets(&mut g, &targets, 0.15)?;
    fx.insert("loss_mean".into(), model.forward(&ids1, Some(&targets), None, "mean")?.reshape(1)?);
    fx.insert("loss_sum".into(), model.forward(&ids1, Some(&targets), None, "sum")?.reshape(1)?);
    fx.insert("loss_targets".into(), targets);
    fx.insert("fw1_ids".into(), ids1);

    kv_decode_case(&model, &mut fx, &mut meta, "kv1", 40, 64, 6)?;
    kv_decode_case(&model, &mut fx, &mut meta, "kv2", 150, 170, 6)?; // window binds: 128 < 150

    // Naive model.generate (recomputes the full forward each step), temperature=0
    let mut g = StdRng::seed_from_u64(13);
    let gen_prompt: Vec<u32> = (0..10).map(|_| g.random_range(0..config.vocab_size as u32)).collect();
    let mut generated = Vec::new();
    model.generate(&gen_prompt, 8, 0.0, None, 42, |t| {
        generated.push(t);
        true
    })?;
    meta.insert("naive_gen_prompt".into(), json!(gen_prompt));
    meta.insert("naive_gen_tokens".into(), json!(generated));

    // Engine.generate / generate_batch, temperature=0
    let special_tokenizer = SpecialOnlyTokenizer;
    let engine = Engine::new(&model, &special_tokenizer);
    let mut g = StdRng::seed_from_u64(17);
    let eng_prompt: Vec<u32> = (0..12).map(|_| g.random_range(0..config.vocab_size as u32)).collect();
    let (mut columns, mut masks) = (Vec::new(), Vec::new());
    for step in engine.generate(&eng_prompt, 2, Some(10), 0.0, None, 42)? {
        let (token_column, token_masks) = step?;
        columns.push(token_column);
        masks.push(token_masks);
    }
    let (results, result_masks) = engine.generate_batch(&eng_prompt, 2, Some(10), 0.0, None, 42)?;
    meta.insert("engine_prompt".into(), json!(eng_prompt));
    meta.insert("engine_token_columns".into(), json!(columns));
    meta.insert("engine_token_masks".into(), json!(masks));
    meta.insert("engine_batch_results".into(), json!(results));
    meta.insert("engine_batch_masks".into(), json!(result_masks));

    // Calculator tool
    let calc_cases = [
        "1+2", "3*4+5", "12/4", "1/3", "(8-3)*2", "10//3", "3.5*2", "2.5+0.1",
        "1,234+1", "  7 * 7 ", "2**3", "1+; ",
        "'strawberry'.count('r')", "\"hello world\".count(\"o\")", "'aaa'.count('aa')",
        "'abc'.count('d')", "import os", "'x'.upper()", "__class__",
        // floored division, signs
        "7//-2", "-7//-2", "2//-3", "-7//2", "-3.5//2",
        // literal edge cases
        "007+1", "00", "000", "00.5+1", "5.",
        // float str() formatting incl. scientific notation
        "1/100000", "1.5*10000000000000000", "1/3+0", "0.1+0.2", "10/2-5",
        // string atoms, concatenation, parens, chained methods
        "'.count('", "'a' 'b'.count('b')", "('ab').count('a')", "'ab'.count(('a'))",
        "'abc'.upper().count('A')", "'  x  '.strip().count('x')", "''.count('')",
        "'ab'.count('')",
    ];
    let calculator_cases: Vec<Value> =
        calc_cases.iter().map(|e| json!({"expr": e, "result": use_calculator(e)})).collect();
    meta.insert("calculator_cases".into(), json!(calculator_cases));

    // Tokenizer: train a tiny BPE, export it in all three formats
    let corpus = TOKENIZER_CORPUS.iter().cycle().take(TOKENIZER_CORPUS.len() * 20).map(|s| s.to_string());
    let tok = Tokenizer::train_from_iterator(corpus, 300)?;
    tok.save(out)?;
    let num_ranks = tok.get_vocab_size() - SPECIAL_TOKENS.len();
    let ranks: Vec<Vec<u8>> =
        (0..num_ranks as u32).map(|id| tok.decode_single_token_bytes(id).unwrap().to_vec()).collect();
    let specials: Vec<(String, u32)> =
        SPECIAL_TOKENS.iter().map(|name| (name.to_string(), tok.encode_special(name))).collect();
    let formats = out.join("tokenizer_formats");
    std::fs::create_dir_all(formats.join("pkl"))?;
    std::fs::write(formats.join("pkl").join("tokenizer.pkl"), tiktoken_pickle(SPLIT_PATTERN, &ranks, &specials))?;
    std::fs::create_dir_all(formats.join("hf"))?;
    std::fs::write(
        formats.join("hf").join("tokenizer.json"),
        serde_json::to_string(&hf_tokenizer_json(SPLIT_PATTERN, &ranks, &specials))?,
    )?;

    let test_strings = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "12345 + 67890 = ?",
        "I'd've THOUGHT so!",
        "héllo wörld 你好 \u{1f600}",
        "  leading spaces\n\nnewlines\t\ttabs   ",
        "def f(x):\n    return x*2  # comment\n",
        "",
        "a",
        "\n",
    ];
    let conversation = json!({
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 12 + 34?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me compute."},
                {"type": "python", "text": "12+34"},
                {"type": "python_output", "text": "46"},
                {"type": "text", "text": "The answer is 46."},
            ]},
            {"role": "user", "content": "Thanks!"},
            {"role": "assistant", "content": "You're welcome."},
        ]
    });
    let (conv_ids, conv_mask) = tok.render_conversation(&conversation, 2048)?;
    let (trunc_ids, trunc_mask) = tok.render_conversation(&conversation, 20)?;
    let encode_cases: Vec<Value> = test_strings
        .iter()
        .map(|t| {
            let ids = tok.encode(t, None, None);
            json!({"text": t, "decoded": tok.decode(&ids), "ids": ids})
        })
        .collect();
    let special_tokens: Map<String, Value> = specials.iter().map(|(name, id)| (name.clone(), json!(id))).collect();
    let id_to_token_samples: Map<String, Value> =
        [0u32, 1, 50, 255, 260, 290].iter().map(|&i| (i.to_string(), json!(tok.id_to_token(i)))).collect();
    let tok_fixtures = json!({
        "vocab_size": tok.get_vocab_size(),
        "bos_token_id": tok.get_bos_token_id(),
        "special_tokens": special_tokens,
        "encode_cases": encode_cases,
        "encode_prepend_append": {
            "text": "Hello world",
            "ids": tok.encode("Hello world", Some("<|bos|>"), Some("<|assistant_end|>")),
        },
        "encode_batch": tok.encode_batch(&["Hello world", "foo bar"]),
        "render_conversation": {"conversation": conversation, "ids": conv_ids, "mask": conv_mask},
        "render_conversation_truncated": {"max_tokens": 20, "ids": trunc_ids, "mask": trunc_mask},
        "render_for_completion": {
            "conversation": conversation,
            "ids": tok.render_for_completion(&conversation)?,
        },
        "id_to_token_samples": id_to_token_samples,
    });
    write_json(&out.join("tokenizer_fixtures.json"), &tok_fixtures)?;

    candle_core::safetensors::save(&fx, out.join("fixtures.safetensors"))?;
    write_json(&out.join("meta.json"), &Value::Object(meta))?;
    Ok((config, weights, tok))
}

// -----------------------------------------------------------------------------
// training stack data (train_fixtures.safetensors, train_meta.json, shards,
// core_model.safetensors)

const WORDS: [&str; 16] = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "hello", "world", "numbers", "42", "100", "fine", "sure", "gone",
];

fn make_text(rng: &mut StdRng, words: &[&str], n: usize) -> String {
    let picked: Vec<&str> = (0..n).map(|_| words[rng.random_range(0..words.len())]).collect();
    picked.join(" ")
}

fn gen_train_data(out: &Path, config: &GPTConfig, weights: &Tensors, tok: &Tokenizer) -> Result<(GPTConfig, Tensors)> {
    let mut fx: Tensors = HashMap::new();
    let mut meta = Map::new();

    // Optimizer: AdamW group + Muon groups (wide and tall), 5 steps
    let mut g = StdRng::seed_from_u64(42);
    let shapes: [(&[usize], f32); 7] = [
        (&[16, 8], 0.5), (&[5], 0.5), // AdamW
        (&[8, 16], 0.1), (&[8, 16], 0.1), (&[8, 16], 0.1), // Muon wide
        (&[16, 8], 0.1), (&[16, 8], 0.1), // Muon tall
    ];
    let mut vars = Vec::new();
    for (i, (shape, std)) in shapes.iter().enumerate() {
        let init = randn(&mut g, shape, *std)?;
        vars.push(Var::from_tensor(&init)?);
        fx.insert(format!("opt_p{i}_init"), init);
    }
    let mut opt = MuonAdamW::new(vec![
        ParamGroup::adamw(vars[0..2].to_vec(), 0.02, (0.8, 0.95), 1e-10, 0.1),
        ParamGroup::muon(vars[2..5].to_vec(), 0.02, 0.95, 5, 0.9, 0.1),
        ParamGroup::muon(vars[5..7].to_vec(), 0.02, 0.95, 5, 0.9, 0.1),
    ]);
    let n_steps = 5;
    for step in 0..n_steps {
        let mut grads = Vec::new();
        for (i, (shape, _)) in shapes.iter().enumerate() {
            let grad = randn(&mut g, shape, 0.1)?;
            fx.insert(format!("opt_p{i}_grad{step}"), grad.clone());
            grads.push(grad);
        }
        let ids: Vec<_> = vars.iter().map(|v| v.id()).collect();
        opt.step_with(&|v: &Var| {
            let idx = ids.iter().position(|&id| id == v.id())?;
            Some(grads[idx].clone())
        })?;
    }
    for (i, var) in vars.iter().enumerate() {
        fx.insert(format!("opt_p{i}_final"), var.as_tensor().copy()?);
    }
    meta.insert("opt_n_steps".into(), json!(n_steps));

    // Tiny GPT (same weights as the inference data): gradients + training steps
    let (model, varmap) = build_train_model(config, weights)?;
    let named = named_vars(&varmap);

    // 1) single backward pass: every parameter gradient
    let mut g = StdRng::seed_from_u64(21);
    let xb = randint(&mut g, &[2, 16], config.vocab_size)?;
    let yb = randint(&mut g, &[2, 16], config.vocab_size)?;
    let yb = mask_targets(&mut g, &yb, 0.1)?;
    let loss = model.forward(&xb, Some(&yb), None, "mean")?;
    let grads = loss.backward()?;
    for (name, var) in &named {
        let grad = grads.get(var).ok_or_else(|| anyhow::anyhow!("missing gradient for {name}"))?;
        fx.insert(format!("grad::{name}"), grad.clone());
    }
    fx.insert("grad_loss".into(), loss.reshape(1)?);
    fx.insert("grad_x".into(), xb);
    fx.insert("grad_y".into(), yb);

    // 2) three full training steps with the combined optimizer
    let mut optimizer = model.setup_optimizer(&varmap, 0.004, 0.2, 0.02, 0.1, 0.5)?;
    let mut losses = Vec::new();
    for step in 0..3 {
        let x = randint(&mut g, &[2, 16], config.vocab_size)?;
        let y = randint(&mut g, &[2, 16], config.vocab_size)?;
        let loss = model.forward(&x, Some(&y), None, "mean")?;
        losses.push(loss.to_scalar::<f32>()? as f64);
        optimizer.step(&loss.backward()?)?;
        fx.insert(format!("train_x{step}"), x);
        fx.insert(format!("train_y{step}"), y);
    }
    meta.insert("train_losses".into(), json!(losses));
    for (name, var) in &named {
        fx.insert(format!("trained::{name}"), var.as_tensor().copy()?);
    }

    // token_bytes as written by tokenizer training: raw bytes per token, 0 for specials
    let token_bytes: Vec<i64> = tok.token_bytes().iter().map(|&b| b as i64).collect();
    fx.insert("token_bytes_tok".into(), Tensor::from_vec(token_bytes, tok.get_vocab_size(), CPU)?);

    // Dataloader: tiny parquet shards (3 row groups of 4 documents each) + best-fit packing
    let shards = out.join("shards");
    let mut g = StdRng::seed_from_u64(7);
    for shard_idx in 0..2 {
        let docs: Vec<String> = (0..12)
            .map(|_| {
                let n = g.random_range(2..=30);
                make_text(&mut g, &WORDS, n) + "."
            })
            .collect();
        write_parquet(
            &shards.join(format!("shard_{shard_idx:05}.parquet")),
            "message schema { REQUIRED BYTE_ARRAY text (UTF8); }",
            docs.chunks(4).map(|group| vec![Leaf::Str(group.to_vec())]).collect(),
        )?;
    }
    let mut loader = DataLoader::new(tok, 2, 32, "train", Some(&shards), 4, CPU, None, 8)?;
    let mut states: Vec<LoaderState> = Vec::new();
    for i in 0..4 {
        let (inputs, targets, state) = loader.next_batch()?;
        fx.insert(format!("dl_inputs{i}"), inputs);
        fx.insert(format!("dl_targets{i}"), targets);
        states.push(state);
    }
    // resume from the state after batch 1
    let mut loader = DataLoader::new(tok, 2, 32, "train", Some(&shards), 4, CPU, Some(&states[1]), 8)?;
    let (inputs, targets, state) = loader.next_batch()?;
    fx.insert("dl_resume_inputs".into(), inputs);
    fx.insert("dl_resume_targets".into(), targets);
    meta.insert("dl_resume_state".into(), serde_json::to_value(state)?);
    // val split reads only the last shard
    let mut loader = DataLoader::new(tok, 2, 32, "val", Some(&shards), 4, CPU, None, 8)?;
    let (inputs, targets, state) = loader.next_batch()?;
    fx.insert("dl_val_inputs".into(), inputs);
    fx.insert("dl_val_targets".into(), targets);
    meta.insert("dl_val_state".into(), serde_json::to_value(state)?);
    meta.insert("dl_states".into(), serde_json::to_value(states)?);

    // evaluate_bpb (synthetic token_bytes over the model vocab) on the untrained weights
    let model = build_model(config, weights)?;
    let token_bytes: Vec<i64> = (0..config.vocab_size as i64).map(|i| if i < 3 { 0 } else { i % 7 }).collect();
    let token_bytes = Tensor::from_vec(token_bytes, config.vocab_size, CPU)?;
    let mut g = StdRng::seed_from_u64(33);
    let mut batches = Vec::new();
    for i in 0..2 {
        let x = randint(&mut g, &[2, 16], config.vocab_size)?;
        let mut y = randint(&mut g, &[2, 16], config.vocab_size)?;
        if i == 1 {
            y = mask_targets(&mut g, &y, 0.2)?; // exercise the masked path
        }
        fx.insert(format!("bpb_x{i}"), x.clone());
        fx.insert(format!("bpb_y{i}"), y.clone());
        batches.push((x, y));
    }
    // per-position loss (loss_reduction='none')
    fx.insert("loss_none".into(), model.forward(&batches[1].0, Some(&batches[1].1), None, "none")?);
    let mut batch_iter = batches.into_iter().map(Ok);
    meta.insert("bpb".into(), json!(evaluate_bpb(&model, &mut batch_iter, 2, &token_bytes)?));
    fx.insert("bpb_token_bytes".into(), token_bytes);

    // core_eval: synthetic tasks, rendered prompts, fewshot sampling, task accuracies
    let mut g = StdRng::seed_from_u64(99);
    let mut mc_data = Vec::new();
    for _ in 0..8 {
        let query = make_text(&mut g, &WORDS, 6) + "?";
        let choices = [make_text(&mut g, &WORDS, 2), make_text(&mut g, &WORDS, 3), make_text(&mut g, &WORDS, 2)];
        mc_data.push(json!({"query": query, "choices": choices, "gold": g.random_range(0..3)}));
    }
    let mut schema_data = Vec::new();
    for _ in 0..8 {
        let options = [make_text(&mut g, &WORDS, 4), make_text(&mut g, &WORDS, 5), make_text(&mut g, &WORDS, 4)];
        let continuation = make_text(&mut g, &WORDS, 3);
        schema_data.push(json!({"context_options": options, "continuation": continuation, "gold": g.random_range(0..3)}));
    }
    let mut lm_data = Vec::new();
    for _ in 0..8 {
        let context = make_text(&mut g, &WORDS, 6) + "  ";
        let continuation = format!(" {}", make_text(&mut g, &WORDS, 1));
        lm_data.push(json!({"context": context, "continuation": continuation}));
    }
    meta.insert("core_rendered".into(), json!({
        "mc_0shot": core_eval::render_prompts_mc(&mc_data[0], ": ", &[]),
        "mc_2shot": core_eval::render_prompts_mc(&mc_data[0], ": ", &[mc_data[1].clone(), mc_data[2].clone()]),
        "schema_0shot": core_eval::render_prompts_schema(&schema_data[0], " ", &[]),
        "schema_2shot": core_eval::render_prompts_schema(&schema_data[0], " ", &[schema_data[3].clone(), schema_data[4].clone()]),
        "lm_0shot": core_eval::render_prompts_lm(&lm_data[0], "", &[]),
        "lm_2shot": core_eval::render_prompts_lm(&lm_data[0], "", &[lm_data[5].clone(), lm_data[6].clone()]),
    }));

    // fewshot picks: random.Random(1234 + idx).sample(available, k)
    let mut sample_cases = Vec::new();
    for (n, k) in [(8usize, 2usize), (8, 5), (50, 3), (200, 8)] {
        for idx in 0..4usize {
            let mut rng = PyRandom::new(1234 + idx as u64);
            let available: Vec<usize> = (0..n).filter(|&i| i != idx).collect();
            sample_cases.push(json!({"n": n, "k": k, "idx": idx, "picked": rng.sample(&available, k)}));
        }
    }
    meta.insert("core_sample_cases".into(), json!(sample_cases));

    // full task accuracies need a model whose vocab covers the test tokenizer
    let core_config = GPTConfig {
        sequence_len: 512,
        vocab_size: tok.get_vocab_size(),
        n_layer: 2,
        n_head: 2,
        n_kv_head: 2,
        n_embd: 32,
        window_pattern: "L".to_string(),
    };
    let s = 3f32.sqrt() / (core_config.n_embd as f32).sqrt();
    let core_weights = random_weights(&core_config, 555, |name, shape, rng| {
        if name.contains("wte") || name.contains("value_embeds") {
            randn(rng, shape, 0.5)
        } else if name.contains("lm_head") {
            randn(rng, shape, 0.05)
        } else if name.contains("smear_lambda") {
            Ok(Tensor::full(0.5f32, shape, CPU)?)
        } else if name.contains("backout_lambda") {
            Ok(Tensor::full(0.25f32, shape, CPU)?)
        } else if name.contains("lambda") {
            uniform(rng, shape, 0.5, 1.0) // resid_lambdas, x0_lambdas
        } else {
            uniform(rng, shape, -s, s)
        }
    })?;
    candle_core::safetensors::save(&core_weights, out.join("core_model.safetensors"))?;
    meta.insert("core_model_config".into(), serde_json::to_value(&core_config)?);
    let core_model = build_model(&core_config, &core_weights)?;
    let mut task_accs = Map::new();
    for (name, data, task_meta) in [
        ("mc", &mc_data, json!({"task_type": "multiple_choice", "num_fewshot": 2, "continuation_delimiter": ": "})),
        ("schema", &schema_data, json!({"task_type": "schema", "num_fewshot": 0, "continuation_delimiter": " "})),
        ("lm", &lm_data, json!({"task_type": "language_modeling", "num_fewshot": 1, "continuation_delimiter": ""})),
    ] {
        task_accs.insert(name.into(), json!(core_eval::evaluate_task(&core_model, tok, data, CPU, &task_meta)?));
    }
    meta.insert("core_task_accs".into(), Value::Object(task_accs));
    meta.insert("core_mc_data".into(), json!(mc_data));
    meta.insert("core_schema_data".into(), json!(schema_data));
    meta.insert("core_lm_data".into(), json!(lm_data));

    candle_core::safetensors::save(&fx, out.join("train_fixtures.safetensors"))?;
    write_json(&out.join("train_meta.json"), &Value::Object(meta))?;
    Ok((core_config, core_weights))
}

// -----------------------------------------------------------------------------
// tasks and chat eval data (tasks_meta.json, hf_data, task_data, eval_bundle,
// words_alpha.txt)

fn conversations(task: &dyn Task) -> Result<Vec<Value>> {
    Ok((0..task.len()).map(|i| task.get(i)).collect::<candle_core::Result<_>>()?)
}

fn gen_tasks_data(out: &Path, core_config: &GPTConfig, core_weights: &Tensors, tok: &Tokenizer) -> Result<()> {
    let mut meta = Map::new();

    // Synthetic dataset rows, standing in for the seed-42 shuffled hub datasets
    let mut arc_rows: Vec<Value> = (0..6)
        .map(|i| json!({
            "question": format!("What is {i} + {i}?"),
            "choices": {
                "text": [(i * 2).to_string(), (i * 2 + 1).to_string(), i.to_string(), "none".to_string()],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "A",
        }))
        .collect();
    arc_rows[3]["choices"]["label"] = json!(["1", "2", "3", "4"]); // ARC sometimes uses numbers
    arc_rows[3]["answerKey"] = json!("1");
    let mmlu_rows: Vec<Value> = (0..6)
        .map(|i| json!({
            "question": format!("Question number {i}?"),
            "choices": ["opt0", "opt1", "opt2", "opt3"],
            "answer": i % 4,
            "subject": "anatomy",
        }))
        .collect();
    let gsm8k_rows = vec![
        json!({"question": "Weng earns $12 an hour. She did 50 minutes. How much?",
               "answer": "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10"}),
        json!({"question": "Two plus two?", "answer": "<<2+2=4>>4 indeed <<4*1=4>><<1+1=2>>\n#### 4"}),
        json!({"question": "No tools here?", "answer": "Just thinking.\n#### 1,234"}),
        json!({"question": "No equals sign?", "answer": "Look: <<weird>> ok\n#### -5.5"}),
    ];
    let humaneval_rows = vec![json!({
        "prompt": "from typing import List\n\n\ndef add(a: int, b: int) -> int:\n    \"\"\"Add two numbers.\"\"\"\n",
        "canonical_solution": "    return a + b\n",
        "entry_point": "add",
        "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n",
    })];
    let smoltalk_rows = vec![
        json!({"messages": [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]}),
        json!({"messages": [
            {"role": "user", "content": "What is up?"},
            {"role": "assistant", "content": "The sky."},
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": "Anytime."},
        ]}),
    ];

    // the rows as JSONL, which the tests feed to the from_rows constructors
    let hf_dir = out.join("hf_data");
    write_jsonl(&hf_dir.join("arc_ARC-Easy_test.jsonl"), &arc_rows)?;
    write_jsonl(&hf_dir.join("mmlu_all_test.jsonl"), &mmlu_rows)?;
    write_jsonl(&hf_dir.join("gsm8k_main_test.jsonl"), &gsm8k_rows)?;
    write_jsonl(&hf_dir.join("humaneval_test.jsonl"), &humaneval_rows)?;
    write_jsonl(&hf_dir.join("smoltalk_test.jsonl"), &smoltalk_rows)?;

    // Hub loader: the same rows as parquet shards in the hub cache layout; the
    // expected rows are the originals in the seed-42 shuffled order
    const LIST_OF_STR: &str = "(LIST) { REPEATED group list { REQUIRED BYTE_ARRAY element (UTF8); } }";
    write_hub_cache(
        out, "allenai/ai2_arc", "ARC-Easy", "test",
        &format!(
            "message schema {{ REQUIRED BYTE_ARRAY question (UTF8); REQUIRED group choices {{ \
             REQUIRED group text {LIST_OF_STR} REQUIRED group label {LIST_OF_STR} }} \
             REQUIRED BYTE_ARRAY answerKey (UTF8); }}"
        ),
        &[&arc_rows[..4], &arc_rows[4..]], // two shards
        |rows| vec![
            str_column(rows, "question"),
            Leaf::StrList(rows.iter().map(|r| strs(&r["choices"]["text"])).collect()),
            Leaf::StrList(rows.iter().map(|r| strs(&r["choices"]["label"])).collect()),
            str_column(rows, "answerKey"),
        ],
    )?;
    write_hub_cache(
        out, "cais/mmlu", "all", "test",
        &format!(
            "message schema {{ REQUIRED BYTE_ARRAY question (UTF8); REQUIRED group choices {LIST_OF_STR} \
             REQUIRED INT64 answer; REQUIRED BYTE_ARRAY subject (UTF8); }}"
        ),
        &[&mmlu_rows[..]],
        |rows| vec![
            str_column(rows, "question"),
            Leaf::StrList(rows.iter().map(|r| strs(&r["choices"])).collect()),
            Leaf::Int(rows.iter().map(|r| r["answer"].as_i64().unwrap()).collect()),
            str_column(rows, "subject"),
        ],
    )?;
    let message_fields = |rows: &[Value], key: &str| -> Leaf {
        Leaf::StrList(
            rows.iter()
                .map(|r| r["messages"].as_array().unwrap().iter().map(|m| m[key].as_str().unwrap().to_string()).collect())
                .collect(),
        )
    };
    write_hub_cache(
        out, "HuggingFaceTB/smol-smoltalk", "default", "test",
        "message schema { REQUIRED group messages (LIST) { REPEATED group list { REQUIRED group element { \
         REQUIRED BYTE_ARRAY role (UTF8); REQUIRED BYTE_ARRAY content (UTF8); } } } }",
        &[&smoltalk_rows[..]],
        |rows| vec![message_fields(rows, "role"), message_fields(rows, "content")],
    )?;
    let hub_arc = ARC::from_rows(hub::shuffle(arc_rows.clone(), 42), Slice::default());
    meta.insert("hub_arc_conversations".into(), json!(conversations(&hub_arc)?));
    meta.insert("hub_shuffled".into(), json!({
        "arc": hub::shuffle(arc_rows.clone(), 42),
        "mmlu": hub::shuffle(mmlu_rows.clone(), 42),
        "smoltalk": hub::shuffle(smoltalk_rows.clone(), 42),
    }));

    // Conversations, evaluation outcomes, slicing
    let arc = ARC::from_rows(arc_rows.clone(), Slice::default());
    let mmlu = MMLU::from_rows(mmlu_rows.clone(), Slice::default());
    let gsm8k = GSM8K::from_rows(gsm8k_rows.clone(), Slice::default());
    let humaneval = HumanEval::from_rows(humaneval_rows.clone(), Slice::default());
    let smoltalk = SmolTalk::from_rows(smoltalk_rows.clone(), Slice::default());

    let letters = ["A".to_string(), "B".to_string()];
    let choices = ["because".to_string(), "why not".to_string()];
    meta.insert("render_mc".into(), json!(render_mc("Why?", &letters, &choices)));
    meta.insert("arc_conversations".into(), json!(conversations(&arc)?));
    meta.insert("mmlu_conversations".into(), json!(conversations(&mmlu)?));
    meta.insert("gsm8k_conversations".into(), json!(conversations(&gsm8k)?));
    meta.insert("humaneval_conversations".into(), json!(conversations(&humaneval)?));
    meta.insert("smoltalk_conversations".into(), json!(conversations(&smoltalk)?));

    let mut arc_eval = Vec::new();
    for (idx, response) in [(0, "A"), (0, "B"), (3, "1")] {
        let outcome = arc.evaluate(&arc.get(idx)?, response)?;
        arc_eval.push(json!({"idx": idx, "response": response, "outcome": outcome}));
    }
    meta.insert("arc_eval".into(), json!(arc_eval));
    let mut gsm8k_eval = Vec::new();
    for (idx, response) in [
        (0, "Lots of math.\n#### 10"), (0, "#### 1,0"), (0, "no answer at all"),
        (2, "#### 1234"), (3, "#### -5.5"),
    ] {
        let outcome = gsm8k.evaluate(&gsm8k.get(idx)?, response)?;
        gsm8k_eval.push(json!({"idx": idx, "response": response, "outcome": outcome as i64}));
    }
    meta.insert("gsm8k_eval".into(), json!(gsm8k_eval));
    let conv0 = gsm8k.get(0)?;
    meta.insert("gsm8k_rewards".into(), json!([gsm8k.reward(&conv0, "#### 10")?, gsm8k.reward(&conv0, "#### 9")?]));
    let extract_answer_cases: Vec<Value> = ["#### 42", "x #### 1,234.5 y", "#### -7", "nothing", "a #### 1 b #### 2"]
        .iter()
        .map(|t| json!({"text": t, "answer": extract_answer(t)}))
        .collect();
    meta.insert("extract_answer_cases".into(), json!(extract_answer_cases));

    // humaneval: correct and broken completions through the real executor
    let he_conv = humaneval.get(0)?;
    let he_prompt = he_conv["messages"][0]["content"].as_str().unwrap().to_string();
    let correct_completion = he_conv["messages"][1]["content"].as_str().unwrap().to_string();
    let broken_completion = "```python\ndef add(a, b):\n    return a - b\n```".to_string();
    let mut humaneval_eval = Vec::new();
    for response in [correct_completion, broken_completion] {
        let outcome = humaneval.evaluate(&he_conv, &response)?;
        humaneval_eval.push(json!({"response": response, "outcome": outcome}));
    }
    meta.insert("humaneval_eval".into(), json!(humaneval_eval));
    let extract_imports_cases: Vec<Value> =
        [he_prompt.as_str(), "import os\nfrom sys import path\n# comment\nimport re\nx = 1\nimport late"]
            .iter()
            .map(|t| json!({"text": t, "imports": extract_imports(t)}))
            .collect();
    meta.insert("extract_imports_cases".into(), json!(extract_imports_cases));
    let extract_program_cases: Vec<Value> = [
        "```python\nx = 1\n```",
        "text before\n```\ny = 2\nz = 3\n```\ntext after",
        "plain code\nno blocks",
        "```python\nfirst\n```\n```python\nsecond\n```",
    ]
    .iter()
    .map(|t| json!({"text": t, "program": extract_program(t)}))
    .collect();
    meta.insert("extract_program_cases".into(), json!(extract_program_cases));

    // slicing + mixture
    let sliced = MMLU::from_rows(mmlu_rows.clone(), Slice::new(1, Some(6), 2));
    meta.insert("sliced_len".into(), json!(sliced.len()));
    meta.insert("sliced_conversations".into(), json!(conversations(&sliced)?));
    let mixture = TaskMixture::new(vec![
        Box::new(ARC::from_rows(arc_rows.clone(), Slice::default())),
        Box::new(MMLU::from_rows(mmlu_rows.clone(), Slice::default())),
    ]);
    meta.insert("mixture_len".into(), json!(mixture.len()));
    meta.insert("mixture_conversations".into(), json!(conversations(&mixture)?));

    // SpellingBee / SimpleSpelling with a synthetic word list
    let words = [
        "strawberry", "banana", "letter", "mississippi", "queue", "rhythm",
        "bookkeeper", "committee", "balloon", "coffee", "pepper", "success",
        "assess", "footloose", "deed", "noon", "level", "radar", "kayak", "stats",
    ];
    std::fs::write(out.join("words_alpha.txt"), words.join("\n") + "\n")?;
    let bee_train = SpellingBee::new(1000, "train", Slice::default())?;
    let bee_test = SpellingBee::new(1000, "test", Slice::default())?;
    let first = |task: &dyn Task, n: usize| -> Result<Vec<Value>> {
        Ok((0..n).map(|i| task.get(i)).collect::<candle_core::Result<_>>()?)
    };
    meta.insert("spellingbee_train".into(), json!(first(&bee_train, 8)?));
    meta.insert("spellingbee_test".into(), json!(first(&bee_test, 4)?));
    let bee_conv = bee_train.get(0)?;
    let mut spellingbee_eval = Vec::new();
    for response in ["#### 3", "#### 2"] {
        let outcome = bee_train.evaluate(&bee_conv, response)?;
        spellingbee_eval.push(json!({"idx": 0, "response": response, "outcome": outcome as i64}));
    }
    meta.insert("spellingbee_eval".into(), json!(spellingbee_eval));
    let simple = SimpleSpelling::new(1000, "train", Slice::default())?;
    meta.insert("simplespelling_train".into(), json!(first(&simple, 6)?));

    // chat_eval loops on the tiny core model + test tokenizer
    let model = build_model(core_config, core_weights)?;
    meta.insert("chat_eval_categorical_acc".into(), json!(run_categorical_eval(&arc, tok, &model, 2, None)?));
    let engine = Engine::new(&model, tok);
    meta.insert(
        "chat_eval_generative_acc".into(),
        json!(run_generative_eval(&gsm8k, tok, &model, &engine, 1, 8, 0.0, 50, None)?),
    );

    // base_eval.evaluate_core over a synthetic eval bundle
    let bundle_dir = out.join("eval_bundle");
    let mut g = StdRng::seed_from_u64(5);
    let mut core_mc_items = Vec::new();
    for _ in 0..10 {
        let query = make_text(&mut g, &words, 5) + "?";
        let choices = [make_text(&mut g, &words, 1), make_text(&mut g, &words, 1), make_text(&mut g, &words, 1)];
        core_mc_items.push(json!({"query": query, "choices": choices, "gold": g.random_range(0..3)}));
    }
    let mut core_lm_items = Vec::new();
    for _ in 0..10 {
        let context = make_text(&mut g, &words, 5) + " ";
        let continuation = format!(" {}", make_text(&mut g, &words, 1));
        core_lm_items.push(json!({"context": context, "continuation": continuation}));
    }
    write_jsonl(&bundle_dir.join("eval_data").join("tiny_mc.jsonl"), &core_mc_items)?;
    write_jsonl(&bundle_dir.join("eval_data").join("tiny_lm.jsonl"), &core_lm_items)?;
    std::fs::write(
        bundle_dir.join("core.yaml"),
        "icl_tasks:\n\
         - label: tiny_mc\n  dataset_uri: tiny_mc.jsonl\n  num_fewshot: [2]\n  icl_task_type: multiple_choice\n  continuation_delimiter: ': '\n\
         - label: tiny_lm\n  dataset_uri: tiny_lm.jsonl\n  num_fewshot: [0]\n  icl_task_type: language_modeling\n",
    )?;
    std::fs::write(
        bundle_dir.join("eval_meta_data.csv"),
        "Eval Task,Random baseline\ntiny_mc,33.3\ntiny_lm,0.0\n",
    )?;
    meta.insert("evaluate_core".into(), nanochat::base_eval::evaluate_core(&model, tok, CPU, 8)?);

    // SFT bestfit-pad batches (the chat_sft packing logic)
    let sft_mixture = TaskMixture::new(vec![
        Box::new(SmolTalk::from_rows(smoltalk_rows.clone(), Slice::default())),
        Box::new(MMLU::from_rows(mmlu_rows.clone(), Slice::default())),
    ]);
    let mut sft_loader = SftDataLoader::new(&sft_mixture, tok, 2, 64, true, -1, CPU, 6);
    let mut sft_batches = Vec::new();
    for _ in 0..3 {
        let (inputs, targets) = sft_loader.next_batch()?;
        sft_batches.push(json!({"inputs": inputs.to_vec2::<i64>()?, "targets": targets.to_vec2::<i64>()?}));
    }
    meta.insert("sft_batches".into(), json!(sft_batches));

    write_json(&out.join("tasks_meta.json"), &Value::Object(meta))?;
    Ok(())
}

fn main() -> Result<()> {
    let out = match std::env::args().nth(1) {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("data"),
    };
    std::fs::create_dir_all(&out)?;
    let out = out.canonicalize()?;
    // the word list, hub cache and eval bundle are found via the base dir
    std::env::set_var("NANOCHAT_BASE_DIR", &out);

    println!("Generating test data in {}", out.display());
    let (config, weights, tok) = gen_model_data(&out)?;
    println!("  model, engine and tokenizer data");
    let (core_config, core_weights) = gen_train_data(&out, &config, &weights, &tok)?;
    println!("  training stack data");
    gen_tasks_data(&out, &core_config, &core_weights, &tok)?;
    println!("  tasks and chat eval data");
    println!("Done. Run the tests with: cargo test --release");
    Ok(())
}
