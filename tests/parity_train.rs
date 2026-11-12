//! Tests for the training stack against the inputs and expected outputs in
//! tests/data (train_fixtures.safetensors, train_meta.json, shards): optimizer
//! steps, autograd gradients, full training steps, dataloader packing,
//! bits-per-byte eval, and CORE eval.
//!
//! tests/data is generated, not checked in: `cargo run --release --example
//! gen_test_data` rebuilds it from this implementation, so these tests detect
//! behaviour changes since it was generated. The Python reference's fixture
//! scripts write the same layout, which turns them into parity tests.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};

use nanochat::core_eval;
use nanochat::dataloader::{DataLoader, LoaderState};
use nanochat::gpt::{GPTConfig, GPT};
use nanochat::loss_eval::evaluate_bpb;
use nanochat::optim::{MuonAdamW, ParamGroup};
use nanochat::pyrandom::PyRandom;
use nanochat::tokenizer::Tokenizer;

fn data_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data");
    assert!(dir.join("meta.json").exists(), "tests/data is generated, not checked in: run `cargo run --release --example gen_test_data` first");
    dir
}

fn load_fixtures() -> HashMap<String, Tensor> {
    candle_core::safetensors::load(data_dir().join("train_fixtures.safetensors"), &Device::Cpu)
        .unwrap()
}

fn load_meta() -> serde_json::Value {
    let data = std::fs::read_to_string(data_dir().join("train_meta.json")).unwrap();
    serde_json::from_str(&data).unwrap()
}

#[track_caller]
fn assert_close(actual: &Tensor, expected: &Tensor, atol: f64, name: &str) {
    let actual = actual
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let expected = expected
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    let mut max_diff = 0f64;
    for (a, e) in actual.iter().zip(expected.iter()) {
        max_diff = max_diff.max((a - e).abs() as f64);
    }
    assert!(
        max_diff <= atol,
        "{name}: max_diff={max_diff:.3e} > atol={atol:.1e}"
    );
}

/// Build the tiny GPT backed by a VarMap (trainable) with the reference weights.
fn load_train_model(weights_file: &str, config_key: &str) -> (GPT, VarMap) {
    let meta_file = if config_key == "core_model_config" {
        "train_meta.json"
    } else {
        "meta.json"
    };
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(data_dir().join(meta_file)).unwrap())
            .unwrap();
    let config: GPTConfig = serde_json::from_value(meta[config_key].clone()).unwrap();
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let model = GPT::new(config, vb).unwrap();
    varmap.load(data_dir().join(weights_file)).unwrap();
    (model, varmap)
}

fn test_tokenizer() -> Tokenizer {
    Tokenizer::from_directory(&data_dir()).unwrap()
}

// -----------------------------------------------------------------------------
// optimizer: AdamW group + Muon groups (wide and tall), 5 golden steps

#[test]
fn test_optimizer_parity() {
    let fx = load_fixtures();
    let meta = load_meta();
    let n_steps = meta["opt_n_steps"].as_u64().unwrap() as usize;

    let vars: Vec<Var> = (0..7)
        .map(|i| Var::from_tensor(&fx[&format!("opt_p{i}_init")]).unwrap())
        .collect();
    let groups = vec![
        ParamGroup::adamw(vars[0..2].to_vec(), 0.02, (0.8, 0.95), 1e-10, 0.1),
        ParamGroup::muon(vars[2..5].to_vec(), 0.02, 0.95, 5, 0.9, 0.1),
        ParamGroup::muon(vars[5..7].to_vec(), 0.02, 0.95, 5, 0.9, 0.1),
    ];
    let mut opt = MuonAdamW::new(groups);

    for step in 0..n_steps {
        let grads: Vec<Tensor> = (0..7)
            .map(|i| fx[&format!("opt_p{i}_grad{step}")].clone())
            .collect();
        let ids: Vec<_> = vars.iter().map(|v| v.id()).collect();
        opt.step_with(&|v: &Var| {
            let idx = ids.iter().position(|&id| id == v.id())?;
            Some(grads[idx].clone())
        })
        .unwrap();
    }
    for (i, v) in vars.iter().enumerate() {
        assert_close(
            v.as_tensor(),
            &fx[&format!("opt_p{i}_final")],
            1e-5,
            &format!("opt_p{i}_final"),
        );
    }
}

// -----------------------------------------------------------------------------
// autograd: every parameter gradient from a single backward pass

#[test]
fn test_backward_gradients() {
    let fx = load_fixtures();
    let (model, varmap) = load_train_model("model.safetensors", "config");
    let loss = model
        .forward(&fx["grad_x"], Some(&fx["grad_y"]), None, "mean")
        .unwrap();
    assert_close(&loss, &fx["grad_loss"], 1e-5, "grad loss");
    let grads = loss.backward().unwrap();
    let vars = varmap.data().lock().unwrap();
    for (name, var) in vars.iter() {
        let expected = &fx[&format!("grad::{name}")];
        let actual = grads
            .get(var)
            .unwrap_or_else(|| panic!("missing gradient for {name}"));
        assert_close(actual, expected, 2e-4, &format!("grad::{name}"));
    }
}

// -----------------------------------------------------------------------------
// full training steps: losses and final parameters after 3 optimizer steps

#[test]
fn test_training_steps() {
    let fx = load_fixtures();
    let meta = load_meta();
    let (model, varmap) = load_train_model("model.safetensors", "config");
    let mut optimizer = model
        .setup_optimizer(&varmap, 0.004, 0.2, 0.02, 0.1, 0.5)
        .unwrap();

    let expected_losses: Vec<f64> = meta["train_losses"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    for (step, expected_loss) in expected_losses.iter().enumerate() {
        let x = &fx[&format!("train_x{step}")];
        let y = &fx[&format!("train_y{step}")];
        let loss = model.forward(x, Some(y), None, "mean").unwrap();
        let loss_val = loss.to_scalar::<f32>().unwrap() as f64;
        assert!(
            (loss_val - expected_loss).abs() < 5e-4,
            "step {step}: loss {loss_val} != {expected_loss}"
        );
        let grads = loss.backward().unwrap();
        optimizer.step(&grads).unwrap();
    }

    let vars = varmap.data().lock().unwrap();
    for (name, var) in vars.iter() {
        let expected = &fx[&format!("trained::{name}")];
        assert_close(var.as_tensor(), expected, 1e-3, &format!("trained::{name}"));
    }
}

// -----------------------------------------------------------------------------
// dataloader: best-fit packing over the test parquet shards

#[test]
fn test_dataloader_parity() {
    let fx = load_fixtures();
    let meta = load_meta();
    let tokenizer = test_tokenizer();
    let shards = data_dir().join("shards");
    let expected_states: Vec<LoaderState> =
        serde_json::from_value(meta["dl_states"].clone()).unwrap();

    let mut loader = DataLoader::new(
        &tokenizer,
        2,
        32,
        "train",
        Some(&shards),
        4,
        &Device::Cpu,
        None,
        8,
    )
    .unwrap();
    for i in 0..4 {
        let (inputs, targets, state) = loader.next_batch().unwrap();
        assert_close(
            &inputs,
            &fx[&format!("dl_inputs{i}")],
            0.0,
            &format!("dl_inputs{i}"),
        );
        assert_close(
            &targets,
            &fx[&format!("dl_targets{i}")],
            0.0,
            &format!("dl_targets{i}"),
        );
        assert_eq!(state, expected_states[i], "dl_state{i}");
    }

    // resume from the state after batch 1
    let mut loader = DataLoader::new(
        &tokenizer,
        2,
        32,
        "train",
        Some(&shards),
        4,
        &Device::Cpu,
        Some(&expected_states[1]),
        8,
    )
    .unwrap();
    let (inputs, targets, state) = loader.next_batch().unwrap();
    assert_close(&inputs, &fx["dl_resume_inputs"], 0.0, "dl_resume_inputs");
    assert_close(&targets, &fx["dl_resume_targets"], 0.0, "dl_resume_targets");
    let expected: LoaderState = serde_json::from_value(meta["dl_resume_state"].clone()).unwrap();
    assert_eq!(state, expected, "dl_resume_state");

    // val split reads only the last shard
    let mut loader = DataLoader::new(
        &tokenizer,
        2,
        32,
        "val",
        Some(&shards),
        4,
        &Device::Cpu,
        None,
        8,
    )
    .unwrap();
    let (inputs, targets, state) = loader.next_batch().unwrap();
    assert_close(&inputs, &fx["dl_val_inputs"], 0.0, "dl_val_inputs");
    assert_close(&targets, &fx["dl_val_targets"], 0.0, "dl_val_targets");
    let expected: LoaderState = serde_json::from_value(meta["dl_val_state"].clone()).unwrap();
    assert_eq!(state, expected, "dl_val_state");
}

// -----------------------------------------------------------------------------
// bits per byte + per-position loss

#[test]
fn test_evaluate_bpb() {
    let fx = load_fixtures();
    let meta = load_meta();
    let (model, _varmap) = load_train_model("model.safetensors", "config");

    // per-position loss (loss_reduction='none')
    let loss_none = model
        .forward(&fx["bpb_x1"], Some(&fx["bpb_y1"]), None, "none")
        .unwrap();
    assert_close(&loss_none, &fx["loss_none"], 1e-5, "loss_none");

    let batches: Vec<(Tensor, Tensor)> = (0..2)
        .map(|i| {
            (
                fx[&format!("bpb_x{i}")].clone(),
                fx[&format!("bpb_y{i}")].clone(),
            )
        })
        .collect();
    let mut batch_iter = batches.into_iter().map(Ok);
    let bpb = evaluate_bpb(&model, &mut batch_iter, 2, &fx["bpb_token_bytes"]).unwrap();
    let expected = meta["bpb"].as_f64().unwrap();
    assert!((bpb - expected).abs() < 1e-6, "bpb {bpb} != {expected}");
}

#[test]
fn test_token_bytes() {
    let fx = load_fixtures();
    let tokenizer = test_tokenizer();
    let token_bytes: Vec<i64> = tokenizer.token_bytes().iter().map(|&b| b as i64).collect();
    let expected = fx["token_bytes_tok"].to_vec1::<i64>().unwrap();
    assert_eq!(token_bytes, expected, "token_bytes");
}

// -----------------------------------------------------------------------------
// core_eval: prompt rendering, CPython rng sampling, full task accuracies

#[test]
fn test_core_eval_rendering() {
    let meta = load_meta();
    let rendered = &meta["core_rendered"];
    let mc = meta["core_mc_data"].as_array().unwrap();
    let schema = meta["core_schema_data"].as_array().unwrap();
    let lm = meta["core_lm_data"].as_array().unwrap();

    let check = |actual: Vec<String>, key: &str| {
        let expected: Vec<String> = serde_json::from_value(rendered[key].clone()).unwrap();
        assert_eq!(actual, expected, "{key}");
    };
    check(core_eval::render_prompts_mc(&mc[0], ": ", &[]), "mc_0shot");
    check(
        core_eval::render_prompts_mc(&mc[0], ": ", &[mc[1].clone(), mc[2].clone()]),
        "mc_2shot",
    );
    check(
        core_eval::render_prompts_schema(&schema[0], " ", &[]),
        "schema_0shot",
    );
    check(
        core_eval::render_prompts_schema(&schema[0], " ", &[schema[3].clone(), schema[4].clone()]),
        "schema_2shot",
    );
    check(core_eval::render_prompts_lm(&lm[0], "", &[]), "lm_0shot");
    check(
        core_eval::render_prompts_lm(&lm[0], "", &[lm[5].clone(), lm[6].clone()]),
        "lm_2shot",
    );
}

#[test]
fn test_cpython_random_sample() {
    let meta = load_meta();
    for case in meta["core_sample_cases"].as_array().unwrap() {
        let n = case["n"].as_u64().unwrap() as usize;
        let k = case["k"].as_u64().unwrap() as usize;
        let idx = case["idx"].as_u64().unwrap() as usize;
        let expected: Vec<usize> = serde_json::from_value(case["picked"].clone()).unwrap();
        let mut rng = PyRandom::new(1234 + idx as u64);
        let available: Vec<usize> = (0..n).filter(|&i| i != idx).collect();
        let picked = rng.sample(&available, k);
        assert_eq!(picked, expected, "sample n={n} k={k} idx={idx}");
    }
}

#[test]
fn test_core_eval_tasks() {
    let meta = load_meta();
    let (model, _varmap) = load_train_model("core_model.safetensors", "core_model_config");
    let tokenizer = test_tokenizer();
    let cases = [
        (
            "mc",
            "core_mc_data",
            serde_json::json!({"task_type": "multiple_choice", "num_fewshot": 2, "continuation_delimiter": ": "}),
        ),
        (
            "schema",
            "core_schema_data",
            serde_json::json!({"task_type": "schema", "num_fewshot": 0, "continuation_delimiter": " "}),
        ),
        (
            "lm",
            "core_lm_data",
            serde_json::json!({"task_type": "language_modeling", "num_fewshot": 1, "continuation_delimiter": ""}),
        ),
    ];
    for (name, data_key, task_meta) in cases {
        let data: Vec<serde_json::Value> = serde_json::from_value(meta[data_key].clone()).unwrap();
        let acc =
            core_eval::evaluate_task(&model, &tokenizer, &data, &Device::Cpu, &task_meta).unwrap();
        let expected = meta["core_task_accs"][name].as_f64().unwrap();
        assert!(
            (acc - expected).abs() < 1e-9,
            "{name} accuracy {acc} != {expected}"
        );
    }
}
