//! CORE evaluation for base models: downloads the eval bundle if needed and
//! evaluates each ICL task, reporting per-task and centered accuracies.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Result};
use serde_json::{json, Value};

use super::common::{download_file_with_lock, get_base_dir, print0};
use super::core_eval::evaluate_task;
use super::gpt::GPT;
use super::pyrandom::PyRandom;
use super::tokenizer::Tokenizer;

pub const EVAL_BUNDLE_URL: &str =
    "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip";

fn msg(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

/// Unzip eval_bundle.zip and place it in the base directory.
fn place_eval_bundle(file_path: &Path) {
    let result = (|| -> anyhow::Result<()> {
        let base_dir = get_base_dir();
        let file = std::fs::File::open(file_path)?;
        let mut zip = zip::ZipArchive::new(file)?;
        zip.extract(&base_dir)?; // the archive contains the eval_bundle/ directory
        Ok(())
    })();
    match result {
        Ok(()) => print0(&format!(
            "Placed eval_bundle directory at {}",
            get_base_dir().join("eval_bundle").display()
        )),
        Err(e) => print0(&format!("Failed to extract eval bundle: {e}")),
    }
}

/// Minimal CSV line parser handling double-quoted fields.
fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    for c in line.chars() {
        match c {
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => {
                fields.push(std::mem::take(&mut field));
            }
            _ => field.push(c),
        }
    }
    fields.push(field);
    fields
}

/// Evaluate a base model on the CORE benchmark.
/// Returns {results, centered_results, core_metric}.
pub fn evaluate_core(
    model: &GPT,
    tokenizer: &Tokenizer,
    device: &Device,
    max_per_task: i64,
) -> Result<Value> {
    let base_dir = get_base_dir();
    let eval_bundle_dir = base_dir.join("eval_bundle");
    // Download the eval bundle if needed
    if !eval_bundle_dir.exists() {
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", Some(place_eval_bundle))
            .map_err(|e| msg(e.to_string()))?;
    }

    let config_path = eval_bundle_dir.join("core.yaml");
    let data_base_path = eval_bundle_dir.join("eval_data");
    let eval_meta_data = eval_bundle_dir.join("eval_meta_data.csv");

    let config: serde_yaml::Value = serde_yaml::from_str(
        &std::fs::read_to_string(&config_path)
            .map_err(|e| msg(format!("Error reading {}: {e}", config_path.display())))?,
    )
    .map_err(msg)?;
    let tasks = config["icl_tasks"]
        .as_sequence()
        .ok_or_else(|| msg("missing icl_tasks"))?;

    // Load random baseline values
    let mut random_baselines: HashMap<String, f64> = HashMap::new();
    let csv = std::fs::read_to_string(&eval_meta_data).map_err(msg)?;
    let mut lines = csv.lines();
    let header = parse_csv_line(lines.next().ok_or_else(|| msg("empty csv"))?);
    let task_col = header.iter().position(|h| h.trim() == "Eval Task").unwrap();
    let baseline_col = header
        .iter()
        .position(|h| h.trim() == "Random baseline")
        .unwrap();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields = parse_csv_line(line);
        let baseline: f64 = fields[baseline_col].trim().parse().map_err(msg)?;
        random_baselines.insert(fields[task_col].trim().to_string(), baseline);
    }

    // Evaluate each task
    let mut results = serde_json::Map::new();
    let mut centered_results = serde_json::Map::new();
    for task in tasks {
        let start_time = std::time::Instant::now();
        let label = task["label"].as_str().unwrap().to_string();
        let task_meta = json!({
            "task_type": task["icl_task_type"].as_str().unwrap(),
            "num_fewshot": task["num_fewshot"][0].as_u64().unwrap(),
            "continuation_delimiter": task.get("continuation_delimiter")
                .and_then(|d| d.as_str()).unwrap_or(" "),
        });
        print0(&format!(
            "Evaluating: {label} ({}-shot, type: {})... ",
            task_meta["num_fewshot"],
            task_meta["task_type"].as_str().unwrap()
        ));

        let data_path = data_base_path.join(task["dataset_uri"].as_str().unwrap());
        let raw = std::fs::read_to_string(&data_path)
            .map_err(|e| msg(format!("Error reading {}: {e}", data_path.display())))?;
        let mut data: Vec<Value> = raw
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l.trim()).map_err(msg))
            .collect::<Result<_>>()?;

        // Shuffle for consistent subsampling when using max_per_task
        let mut shuffle_rng = PyRandom::new(1337);
        shuffle_rng.shuffle(&mut data);
        if max_per_task > 0 {
            data.truncate(max_per_task as usize);
        }

        let accuracy = evaluate_task(model, tokenizer, &data, device, &task_meta)?;
        let random_baseline = random_baselines[&label];
        let centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline);
        let elapsed = start_time.elapsed().as_secs_f64();
        print0(&format!(
            "accuracy: {accuracy:.4} | centered: {centered_result:.4} | time: {elapsed:.2}s"
        ));
        results.insert(label.clone(), json!(accuracy));
        centered_results.insert(label, json!(centered_result));
    }

    let core_metric = centered_results
        .values()
        .map(|v| v.as_f64().unwrap())
        .sum::<f64>()
        / centered_results.len() as f64;
    Ok(json!({
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
    }))
}
