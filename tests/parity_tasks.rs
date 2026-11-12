//! Tests for the tasks and chat_eval loops against the inputs and expected
//! outputs in tests/data (tasks_meta.json, hf_data, task_data, eval_bundle).
//!
//! tests/data is generated, not checked in: `cargo run --release --example
//! gen_test_data` rebuilds it from this implementation, so these tests detect
//! behaviour changes since it was generated. The Python reference's fixture
//! scripts write the same layout, which turns them into parity tests.

use std::path::PathBuf;
use std::sync::Once;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use serde_json::Value;

use nanochat::chat_eval::{run_categorical_eval, run_generative_eval};
use nanochat::engine::Engine;
use nanochat::gpt::{GPTConfig, GPT};
use nanochat::tasks::arc::ARC;
use nanochat::tasks::common::{load_jsonl_rows, render_mc, Slice, Task, TaskMixture};
use nanochat::tasks::gsm8k::{extract_answer, GSM8K};
use nanochat::tasks::hub::load_shuffled;
use nanochat::tasks::humaneval::{extract_imports, extract_program, HumanEval};
use nanochat::tasks::mmlu::MMLU;
use nanochat::tasks::smoltalk::SmolTalk;
use nanochat::tasks::spellingbee::{SimpleSpelling, SpellingBee};
use nanochat::tokenizer::Tokenizer;

fn data_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data");
    assert!(dir.join("meta.json").exists(), "tests/data is generated, not checked in: run `cargo run --release --example gen_test_data` first");
    dir
}

/// Synthetic dataset rows standing in for the (seed-42 shuffled) hub datasets.
fn rows(name: &str) -> Vec<Value> {
    load_jsonl_rows(&data_dir().join("hf_data").join(format!("{name}.jsonl"))).unwrap()
}

/// SpellingBee and the eval bundle resolve paths via the nanochat base dir;
/// point it at the fixtures (idempotent across tests).
fn set_base_dir() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| std::env::set_var("NANOCHAT_BASE_DIR", data_dir()));
}

fn load_meta() -> Value {
    let data = std::fs::read_to_string(data_dir().join("tasks_meta.json")).unwrap();
    serde_json::from_str(&data).unwrap()
}

fn conversations(task: &dyn Task) -> Vec<Value> {
    (0..task.len()).map(|i| task.get(i).unwrap()).collect()
}

#[test]
fn test_render_mc() {
    let meta = load_meta();
    let letters = vec!["A".to_string(), "B".to_string()];
    let choices = vec!["because".to_string(), "why not".to_string()];
    assert_eq!(
        render_mc("Why?", &letters, &choices),
        meta["render_mc"].as_str().unwrap()
    );
}

/// The hub loader reads parquet shards from the cache (same layout as the
/// reference's) and shuffles them with numpy's seed-42 permutation.
#[test]
fn test_hub_loader() {
    set_base_dir();
    let meta = load_meta();
    for (key, repo_id, subset) in [
        ("arc", "allenai/ai2_arc", "ARC-Easy"), // two shards, struct of lists
        ("mmlu", "cais/mmlu", "all"),
        ("smoltalk", "HuggingFaceTB/smol-smoltalk", "default"), // list of structs
    ] {
        let rows = load_shuffled(repo_id, subset, "test").unwrap();
        assert_eq!(Value::Array(rows), meta["hub_shuffled"][key], "{key}");
    }
    // the real task constructor, end to end
    let arc = ARC::new("ARC-Easy", "test", Slice::default()).unwrap();
    assert_eq!(
        Value::Array(conversations(&arc)),
        meta["hub_arc_conversations"]
    );
}

#[test]
fn test_task_conversations() {
    set_base_dir();
    let meta = load_meta();
    let arc = ARC::from_rows(rows("arc_ARC-Easy_test"), Slice::default());
    assert_eq!(
        serde_json::json!(conversations(&arc)),
        meta["arc_conversations"],
        "arc"
    );
    let mmlu = MMLU::from_rows(rows("mmlu_all_test"), Slice::default());
    assert_eq!(
        serde_json::json!(conversations(&mmlu)),
        meta["mmlu_conversations"],
        "mmlu"
    );
    let gsm8k = GSM8K::from_rows(rows("gsm8k_main_test"), Slice::default());
    assert_eq!(
        serde_json::json!(conversations(&gsm8k)),
        meta["gsm8k_conversations"],
        "gsm8k"
    );
    let humaneval = HumanEval::from_rows(rows("humaneval_test"), Slice::default());
    assert_eq!(
        serde_json::json!(conversations(&humaneval)),
        meta["humaneval_conversations"],
        "humaneval"
    );
    let smoltalk = SmolTalk::from_rows(rows("smoltalk_test"), Slice::default());
    assert_eq!(
        serde_json::json!(conversations(&smoltalk)),
        meta["smoltalk_conversations"],
        "smoltalk"
    );
}

#[test]
fn test_task_evaluate() {
    set_base_dir();
    let meta = load_meta();
    let arc = ARC::from_rows(rows("arc_ARC-Easy_test"), Slice::default());
    for case in meta["arc_eval"].as_array().unwrap() {
        let idx = case["idx"].as_u64().unwrap() as usize;
        let conv = arc.get(idx).unwrap();
        let outcome = arc
            .evaluate(&conv, case["response"].as_str().unwrap())
            .unwrap();
        assert_eq!(
            outcome,
            case["outcome"].as_bool().unwrap(),
            "arc_eval {case}"
        );
    }
    let gsm8k = GSM8K::from_rows(rows("gsm8k_main_test"), Slice::default());
    for case in meta["gsm8k_eval"].as_array().unwrap() {
        let idx = case["idx"].as_u64().unwrap() as usize;
        let conv = gsm8k.get(idx).unwrap();
        let outcome = gsm8k
            .evaluate(&conv, case["response"].as_str().unwrap())
            .unwrap();
        assert_eq!(
            outcome as i64,
            case["outcome"].as_i64().unwrap(),
            "gsm8k_eval {case}"
        );
    }
    let conv = gsm8k.get(0).unwrap();
    let expected: Vec<f64> = serde_json::from_value(meta["gsm8k_rewards"].clone()).unwrap();
    assert_eq!(gsm8k.reward(&conv, "#### 10").unwrap(), expected[0]);
    assert_eq!(gsm8k.reward(&conv, "#### 9").unwrap(), expected[1]);
}

#[test]
fn test_extract_helpers() {
    let meta = load_meta();
    for case in meta["extract_answer_cases"].as_array().unwrap() {
        let expected = case["answer"].as_str().map(|s| s.to_string());
        assert_eq!(extract_answer(case["text"].as_str().unwrap()), expected);
    }
    for case in meta["extract_imports_cases"].as_array().unwrap() {
        assert_eq!(
            extract_imports(case["text"].as_str().unwrap()),
            case["imports"].as_str().unwrap()
        );
    }
    for case in meta["extract_program_cases"].as_array().unwrap() {
        assert_eq!(
            extract_program(case["text"].as_str().unwrap()),
            case["program"].as_str().unwrap()
        );
    }
}

#[test]
fn test_humaneval_execution() {
    set_base_dir();
    let meta = load_meta();
    let humaneval = HumanEval::from_rows(rows("humaneval_test"), Slice::default());
    let conv = humaneval.get(0).unwrap();
    for case in meta["humaneval_eval"].as_array().unwrap() {
        let outcome = humaneval
            .evaluate(&conv, case["response"].as_str().unwrap())
            .unwrap();
        assert_eq!(
            outcome,
            case["outcome"].as_bool().unwrap(),
            "humaneval response"
        );
    }
}

#[test]
fn test_slicing_and_mixture() {
    set_base_dir();
    let meta = load_meta();
    let sliced = MMLU::from_rows(rows("mmlu_all_test"), Slice::new(1, Some(6), 2));
    assert_eq!(sliced.len() as u64, meta["sliced_len"].as_u64().unwrap());
    assert_eq!(
        serde_json::json!(conversations(&sliced)),
        meta["sliced_conversations"]
    );

    let mixture = TaskMixture::new(vec![
        Box::new(ARC::from_rows(rows("arc_ARC-Easy_test"), Slice::default())),
        Box::new(MMLU::from_rows(rows("mmlu_all_test"), Slice::default())),
    ]);
    assert_eq!(mixture.len() as u64, meta["mixture_len"].as_u64().unwrap());
    assert_eq!(
        serde_json::json!(conversations(&mixture)),
        meta["mixture_conversations"]
    );
}

#[test]
fn test_spellingbee() {
    set_base_dir();
    let meta = load_meta();
    let bee_train = SpellingBee::new(1000, "train", Slice::default()).unwrap();
    for (i, expected) in meta["spellingbee_train"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        assert_eq!(
            &bee_train.get(i).unwrap(),
            expected,
            "spellingbee train {i}"
        );
    }
    let bee_test = SpellingBee::new(1000, "test", Slice::default()).unwrap();
    for (i, expected) in meta["spellingbee_test"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        assert_eq!(&bee_test.get(i).unwrap(), expected, "spellingbee test {i}");
    }
    let conv = bee_train.get(0).unwrap();
    for case in meta["spellingbee_eval"].as_array().unwrap() {
        let outcome = bee_train
            .evaluate(&conv, case["response"].as_str().unwrap())
            .unwrap();
        assert_eq!(outcome as i64, case["outcome"].as_i64().unwrap());
    }
    let simple = SimpleSpelling::new(1000, "train", Slice::default()).unwrap();
    for (i, expected) in meta["simplespelling_train"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        assert_eq!(
            &simple.get(i).unwrap(),
            expected,
            "simplespelling train {i}"
        );
    }
}

/// The SFT bestfit-pad packing must produce the exact reference batches,
/// including the loss masks and padding ignore_index positions.
#[test]
fn test_sft_dataloader() {
    set_base_dir();
    let meta = load_meta();
    let tokenizer = Tokenizer::from_directory(&data_dir()).unwrap();
    let mixture = TaskMixture::new(vec![
        Box::new(SmolTalk::from_rows(rows("smoltalk_test"), Slice::default())),
        Box::new(MMLU::from_rows(rows("mmlu_all_test"), Slice::default())),
    ]);
    let mut loader = nanochat::dataloader::SftDataLoader::new(
        &mixture,
        &tokenizer,
        2,
        64,
        true,
        -1,
        &Device::Cpu,
        6,
    );
    for (i, expected) in meta["sft_batches"].as_array().unwrap().iter().enumerate() {
        let (inputs, targets) = loader.next_batch().unwrap();
        let expected_inputs: Vec<Vec<i64>> =
            serde_json::from_value(expected["inputs"].clone()).unwrap();
        let expected_targets: Vec<Vec<i64>> =
            serde_json::from_value(expected["targets"].clone()).unwrap();
        assert_eq!(
            inputs.to_vec2::<i64>().unwrap(),
            expected_inputs,
            "sft inputs batch {i}"
        );
        assert_eq!(
            targets.to_vec2::<i64>().unwrap(),
            expected_targets,
            "sft targets batch {i}"
        );
    }
}

#[test]
fn test_evaluate_core_bundle() {
    set_base_dir();
    let meta = load_meta();
    let train_meta: Value =
        serde_json::from_str(&std::fs::read_to_string(data_dir().join("train_meta.json")).unwrap())
            .unwrap();
    let config: GPTConfig =
        serde_json::from_value(train_meta["core_model_config"].clone()).unwrap();
    let weights =
        candle_core::safetensors::load(data_dir().join("core_model.safetensors"), &Device::Cpu)
            .unwrap();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let model = GPT::new(config, vb).unwrap();
    let tokenizer = Tokenizer::from_directory(&data_dir()).unwrap();

    let results = nanochat::base_eval::evaluate_core(&model, &tokenizer, &Device::Cpu, 8).unwrap();
    let expected = &meta["evaluate_core"];
    for (label, acc) in expected["results"].as_object().unwrap() {
        assert_eq!(
            results["results"][label].as_f64(),
            acc.as_f64(),
            "accuracy {label}"
        );
    }
    for (label, centered) in expected["centered_results"].as_object().unwrap() {
        let actual = results["centered_results"][label].as_f64().unwrap();
        assert!(
            (actual - centered.as_f64().unwrap()).abs() < 1e-12,
            "centered {label}"
        );
    }
    let core = results["core_metric"].as_f64().unwrap();
    assert!(
        (core - expected["core_metric"].as_f64().unwrap()).abs() < 1e-12,
        "core_metric"
    );
}

#[test]
fn test_chat_eval_loops() {
    set_base_dir();
    let meta = load_meta();
    // tiny core model (vocab matches the test tokenizer) + test tokenizer
    let train_meta: Value =
        serde_json::from_str(&std::fs::read_to_string(data_dir().join("train_meta.json")).unwrap())
            .unwrap();
    let config: GPTConfig =
        serde_json::from_value(train_meta["core_model_config"].clone()).unwrap();
    let weights =
        candle_core::safetensors::load(data_dir().join("core_model.safetensors"), &Device::Cpu)
            .unwrap();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let model = GPT::new(config, vb).unwrap();
    let tokenizer = Tokenizer::from_directory(&data_dir()).unwrap();

    let arc = ARC::from_rows(rows("arc_ARC-Easy_test"), Slice::default());
    let acc = run_categorical_eval(&arc, &tokenizer, &model, 2, None).unwrap();
    assert_eq!(
        acc,
        meta["chat_eval_categorical_acc"].as_f64().unwrap(),
        "categorical acc"
    );

    let gsm8k = GSM8K::from_rows(rows("gsm8k_main_test"), Slice::default());
    let engine = Engine::new(&model, &tokenizer);
    let acc =
        run_generative_eval(&gsm8k, &tokenizer, &model, &engine, 1, 8, 0.0, 50, None).unwrap();
    assert_eq!(
        acc,
        meta["chat_eval_generative_acc"].as_f64().unwrap(),
        "generative acc"
    );
}
