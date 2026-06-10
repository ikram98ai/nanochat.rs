//! Unified evaluation script for base models.
//!
//! Supports three evaluation modes (comma-separated):
//!   --eval core    : CORE metric (accuracy on ICL tasks)
//!   --eval bpb     : Bits per byte on train/val splits
//!   --eval sample  : Generate samples from the model
//!
//!     cargo run --release --bin base_eval -- --model-tag d2 --eval bpb,sample

use candle_core::Tensor;
use clap::Parser;

use nanochat::base_eval::evaluate_core;
use nanochat::checkpoint_manager::load_model;
use nanochat::common::{autodetect_device_type, compute_init, get_base_dir, print0, setup_default_logging};
use nanochat::dataloader::DataLoader;
use nanochat::engine::Engine;
use nanochat::loss_eval::evaluate_bpb;

#[derive(Parser, Debug)]
#[command(about = "Base model evaluation")]
struct Args {
    /// Comma-separated evaluations to run: core,bpb,sample
    #[arg(long, default_value = "core,bpb,sample")]
    eval: String,

    /// nanochat model tag to identify the checkpoint directory
    #[arg(long)]
    model_tag: Option<String>,

    /// Model step to load (default = last)
    #[arg(long)]
    step: Option<usize>,

    /// Max examples per CORE task (-1 = all)
    #[arg(long, default_value_t = -1)]
    max_per_task: i64,

    /// Per-device batch size for BPB evaluation
    #[arg(long, default_value_t = 32)]
    device_batch_size: usize,

    /// Number of tokens to evaluate per split for BPB
    #[arg(long, default_value_t = 40 * 524288)]
    split_tokens: usize,

    /// Device type. empty => autodetect
    #[arg(long, default_value = "")]
    device_type: String,
}

fn main() -> anyhow::Result<()> {
    setup_default_logging();
    let args = Args::parse();

    // Parse evaluation modes
    let eval_modes: Vec<String> = args.eval.split(',').map(|m| m.trim().to_string()).collect();
    for mode in &eval_modes {
        anyhow::ensure!(
            ["core", "bpb", "sample"].contains(&mode.as_str()),
            "Invalid eval mode: {mode}"
        );
    }
    let has = |mode: &str| eval_modes.iter().any(|m| m == mode);

    let device_type = if args.device_type.is_empty() {
        autodetect_device_type()
    } else {
        args.device_type.clone()
    };
    let (_ddp, _rank, _local, ddp_world_size, device) = compute_init(&device_type)?;
    let (model, tokenizer, meta) =
        load_model("base", &device, "eval", args.model_tag.as_deref(), args.step)?;
    let sequence_len = meta["model_config"]["sequence_len"].as_u64().unwrap() as usize;
    let token_bytes = Tensor::from_vec(tokenizer.token_bytes(), tokenizer.get_vocab_size(), &device)?;
    let model_name = format!("base_model (step {})", meta["step"]);
    let model_slug = format!("base_model_{:06}", meta["step"].as_u64().unwrap_or(0));

    print0(&format!("Evaluating model: {model_name}"));
    print0(&format!("Eval modes: {}", eval_modes.join(", ")));

    // --- Sampling ---
    if has("sample") {
        print0(&format!("\n{}", "=".repeat(80)));
        print0("Model Samples");
        print0(&"=".repeat(80));
        let prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ];
        let engine = Engine::new(&model, &tokenizer);
        print0("\nConditioned samples:");
        for prompt in prompts {
            let tokens = tokenizer.encode(prompt, Some("<|bos|>"), None);
            let (sample, _) = engine.generate_batch(&tokens, 1, Some(16), 0.0, None, 42)?;
            print0(&"-".repeat(80));
            print0(&tokenizer.decode(&sample[0]));
        }
        print0("\nUnconditioned samples:");
        let tokens = tokenizer.encode("", Some("<|bos|>"), None);
        let (uncond, _) = engine.generate_batch(&tokens, 8, Some(128), 1.0, None, 42)?;
        for sample in &uncond {
            print0(&"-".repeat(80));
            print0(&tokenizer.decode(sample));
        }
    }

    // --- BPB evaluation ---
    if has("bpb") {
        print0(&format!("\n{}", "=".repeat(80)));
        print0("BPB Evaluation");
        print0(&"=".repeat(80));
        let tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size;
        let mut split_tokens = args.split_tokens;
        if split_tokens % tokens_per_step != 0 {
            // Adjust to nearest multiple
            split_tokens = (split_tokens / tokens_per_step) * tokens_per_step;
            print0(&format!(
                "Adjusted split_tokens to {split_tokens} (must be divisible by {tokens_per_step})"
            ));
        }
        let steps = split_tokens / tokens_per_step;
        for split_name in ["train", "val"] {
            let mut loader = DataLoader::new(
                &tokenizer, args.device_batch_size, sequence_len, split_name, None, 128, &device,
                None, 1000,
            )?;
            let mut batches =
                std::iter::from_fn(|| Some(loader.next_batch().map(|(x, y, _)| (x, y))));
            let bpb = evaluate_bpb(&model, &mut batches, steps, &token_bytes)?;
            print0(&format!("{split_name} bpb: {bpb:.6}"));
        }
    }

    // --- CORE evaluation ---
    if has("core") {
        print0(&format!("\n{}", "=".repeat(80)));
        print0("CORE Evaluation");
        print0(&"=".repeat(80));
        let core_results = evaluate_core(&model, &tokenizer, &device, args.max_per_task)?;

        // Write CSV output
        let output_csv_path = get_base_dir().join("base_eval").join(format!("{model_slug}.csv"));
        std::fs::create_dir_all(output_csv_path.parent().unwrap())?;
        let mut csv = format!("{:<35}, {:<10}, {:<10}\n", "Task", "Accuracy", "Centered");
        for (label, acc) in core_results["results"].as_object().unwrap() {
            let centered = core_results["centered_results"][label].as_f64().unwrap();
            csv.push_str(&format!(
                "{label:<35}, {:<10.6}, {centered:<10.6}\n",
                acc.as_f64().unwrap()
            ));
        }
        let core_metric = core_results["core_metric"].as_f64().unwrap();
        csv.push_str(&format!("{:<35}, {:<10}, {core_metric:<10.6}\n", "CORE", ""));
        std::fs::write(&output_csv_path, csv)?;
        print0(&format!("\nResults written to: {}", output_csv_path.display()));
        print0(&format!("CORE metric: {core_metric:.4}"));
    }

    Ok(())
}
