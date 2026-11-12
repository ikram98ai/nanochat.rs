//! Evaluate the Chat model.
//!
//!     cargo run --release --bin chat_eval -- -i sft -a ARC-Easy

use clap::Parser;

use nanochat::chat_eval::run_chat_eval;
use nanochat::checkpoint_manager::load_model;
use nanochat::common::{autodetect_device_type, compute_init, print0, setup_default_logging};
use nanochat::engine::Engine;

#[derive(Parser, Debug)]
#[command(about = "Evaluate the Chat model")]
struct Args {
    /// Source of the model: base|sft|rl
    #[arg(short = 'i', long, required = true)]
    source: String,

    /// Task name. Default = all tasks. Use | to split multiple tasks.
    #[arg(short = 'a', long)]
    task_name: Option<String>,

    #[arg(short = 't', long, default_value_t = 0.0)]
    temperature: f64,

    #[arg(short = 'm', long, default_value_t = 512)]
    max_new_tokens: usize,

    #[arg(short = 'n', long, default_value_t = 1)]
    num_samples: usize,

    #[arg(short = 'k', long, default_value_t = 50)]
    top_k: usize,

    /// Batch size for categorical evaluation
    #[arg(short = 'b', long, default_value_t = 8)]
    batch_size: usize,

    /// Model tag to load
    #[arg(short = 'g', long)]
    model_tag: Option<String>,

    /// Step to load
    #[arg(short = 's', long)]
    step: Option<usize>,

    /// Max problems to evaluate
    #[arg(short = 'x', long)]
    max_problems: Option<usize>,

    /// Device type for evaluation. empty => autodetect
    #[arg(long, default_value = "")]
    device_type: String,
}

fn main() -> anyhow::Result<()> {
    setup_default_logging();
    let args = Args::parse();

    let device_type = if args.device_type.is_empty() {
        autodetect_device_type()
    } else {
        args.device_type.clone()
    };
    let (_ddp, _rank, _local, _world, device) = compute_init(&device_type)?;

    let (model, tokenizer, _meta) = load_model(
        &args.source,
        &device,
        "eval",
        args.model_tag.as_deref(),
        args.step,
    )?;
    let engine = Engine::new(&model, &tokenizer);

    // Get the tasks to evaluate on (SpellingBee is not part of ChatCORE, but can
    // still be evaluated explicitly with -a SpellingBee)
    let all_tasks = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval"];
    // multiple choice 1 of 4 => 25%, open-ended => 0%
    let baseline_accuracies = |task: &str| match task {
        "ARC-Easy" | "ARC-Challenge" | "MMLU" => 0.25,
        _ => 0.0,
    };
    let task_names: Vec<String> = match &args.task_name {
        Some(name) => name.split('|').map(|s| s.to_string()).collect(),
        None => all_tasks.iter().map(|s| s.to_string()).collect(),
    };

    // Run all the task evaluations sequentially
    let mut results: Vec<(String, f64)> = Vec::new();
    for task_name in &task_names {
        let acc = run_chat_eval(
            task_name,
            &model,
            &tokenizer,
            &engine,
            args.batch_size,
            args.num_samples,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.max_problems,
        )?;
        results.push((task_name.clone(), acc));
        print0(&format!("{task_name} accuracy: {:.2}%", 100.0 * acc));
    }

    // ChatCORE metric: mean centered accuracy, 0 at random baseline to 1 at peak
    let all_tasks_were_evaluated = all_tasks
        .iter()
        .all(|t| results.iter().any(|(name, _)| name == t));
    if all_tasks_were_evaluated {
        let mut centered_mean = 0.0;
        for (task_name, acc) in &results {
            let baseline_acc = baseline_accuracies(task_name);
            centered_mean += (acc - baseline_acc) / (1.0 - baseline_acc);
        }
        let chatcore_metric = centered_mean / results.len() as f64;
        print0(&format!("ChatCORE metric: {chatcore_metric:.4}"));
    }

    Ok(())
}
