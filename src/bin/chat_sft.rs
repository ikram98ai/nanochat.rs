//! Supervised fine-tuning (SFT) the model.
//!
//!     cargo run --release --bin chat_sft -- --num-iterations 20

use std::collections::HashMap;
use std::time::Instant;

use candle_core::{Tensor, TensorId, Var};
use clap::Parser;
use serde_json::json;

use nanochat::chat_eval::run_chat_eval;
use nanochat::checkpoint_manager::{
    build_model_for_training, find_largest_model, find_last_step, load_optimizer_state,
    save_checkpoint,
};
use nanochat::common::{autodetect_device_type, compute_init, get_base_dir, print0, setup_default_logging};
use nanochat::dataloader::SftDataLoader;
use nanochat::engine::Engine;
use nanochat::loss_eval::evaluate_bpb;
use nanochat::optim::GroupKind;
use nanochat::tasks::common::{Slice, Task, TaskMixture};
use nanochat::tasks::gsm8k::GSM8K;
use nanochat::tasks::mmlu::MMLU;
use nanochat::tasks::smoltalk::SmolTalk;

#[derive(Parser, Debug, Clone, serde::Serialize)]
#[command(about = "Supervised fine-tuning (SFT) the model")]
struct Args {
    /// device type: cuda|cpu|mps (empty = autodetect)
    #[arg(long, default_value = "")]
    device_type: String,
    /// model tag to load from
    #[arg(long)]
    model_tag: Option<String>,
    /// model step to load from
    #[arg(long)]
    model_step: Option<usize>,
    /// warm-start optimizer from pretrained checkpoint (0=no, 1=yes)
    #[arg(long, default_value_t = 1)]
    load_optimizer: usize,
    /// number of optimization steps (-1 = full epoch)
    #[arg(long, default_value_t = -1)]
    num_iterations: i64,
    /// max context length (default: inherit from pretrain)
    #[arg(long)]
    max_seq_len: Option<usize>,
    /// per-device batch size (default: inherit from pretrain)
    #[arg(long)]
    device_batch_size: Option<usize>,
    /// total batch size in tokens (default: inherit from pretrain)
    #[arg(long)]
    total_batch_size: Option<usize>,
    /// learning rate for embedding parameters (default: inherit from pretrain)
    #[arg(long)]
    embedding_lr: Option<f64>,
    /// learning rate for unembedding parameters (default: inherit from pretrain)
    #[arg(long)]
    unembedding_lr: Option<f64>,
    /// learning rate for matrix parameters (default: inherit from pretrain)
    #[arg(long)]
    matrix_lr: Option<f64>,
    /// initial LR as fraction of base LR
    #[arg(long, default_value_t = 0.8)]
    init_lr_frac: f64,
    /// ratio of iterations for LR warmup
    #[arg(long, default_value_t = 0.0)]
    warmup_ratio: f64,
    /// ratio of iterations for LR warmdown
    #[arg(long, default_value_t = 0.5)]
    warmdown_ratio: f64,
    /// final LR as fraction of initial LR
    #[arg(long, default_value_t = 0.0)]
    final_lr_frac: f64,
    /// evaluate val bpb every N steps (-1 = disable)
    #[arg(long, default_value_t = 200)]
    eval_every: i64,
    /// number of tokens to evaluate val loss on
    #[arg(long, default_value_t = 40 * 524288)]
    eval_tokens: usize,
    /// evaluate ChatCORE metric every N steps (-1 = disable)
    #[arg(long, default_value_t = 200)]
    chatcore_every: i64,
    /// max problems per categorical task for ChatCORE (-1 = no limit)
    #[arg(long, default_value_t = -1)]
    chatcore_max_cat: i64,
    /// max problems per generative task for ChatCORE
    #[arg(long, default_value_t = 24)]
    chatcore_max_sample: i64,
    /// number of epochs of MMLU in training mixture (teaches Multiple Choice)
    #[arg(long, default_value_t = 3)]
    mmlu_epochs: usize,
    /// number of epochs of GSM8K in training mixture (teaches Math and Tool Use)
    #[arg(long, default_value_t = 4)]
    gsm8k_epochs: usize,
}

/// Resolve a hyperparameter: CLI override > pretrained checkpoint > fallback.
fn inherit<T: Copy + std::fmt::Display + serde::de::DeserializeOwned + PartialEq>(
    name: &str,
    arg_val: Option<T>,
    source: &serde_json::Value,
    fallback: T,
) -> T {
    let pretrain_val: Option<T> = serde_json::from_value(source[name].clone()).ok();
    match (arg_val, pretrain_val) {
        (None, Some(v)) => {
            print0(&format!("Inherited {name}={v} from pretrained checkpoint"));
            v
        }
        (None, None) => {
            print0(&format!("Inherited {name}={fallback} from pretrained checkpoint"));
            fallback
        }
        (Some(a), Some(p)) if a != p => {
            print0(&format!("NOTE: --{name}={a} overrides pretrained value of {p}"));
            a
        }
        (Some(a), _) => {
            print0(&format!("Using {name}={a}"));
            a
        }
    }
}

fn main() -> anyhow::Result<()> {
    setup_default_logging();
    let args = Args::parse();
    let user_config = serde_json::to_value(&args)?;

    let device_type = if args.device_type.is_empty() {
        autodetect_device_type()
    } else {
        args.device_type.clone()
    };
    let (_ddp, _rank, _local, ddp_world_size, device) = compute_init(&device_type)?;

    // Load the model and tokenizer (trainable parameters)
    let base_dir = get_base_dir();
    let base_checkpoints = base_dir.join("base_checkpoints");
    let model_tag = match &args.model_tag {
        Some(tag) => tag.clone(),
        None => find_largest_model(&base_checkpoints)?,
    };
    let checkpoint_dir = base_checkpoints.join(&model_tag);
    let model_step = match args.model_step {
        Some(step) => step,
        None => find_last_step(&checkpoint_dir)?,
    };
    let (model, varmap, tokenizer, meta) =
        build_model_for_training(&checkpoint_dir, model_step, &device)?;

    // Inherit training hyperparameters from the pretrained checkpoint
    let pretrain_user_config = meta["user_config"].clone();
    let max_seq_len = inherit("max_seq_len", args.max_seq_len, &meta, 2048);
    let device_batch_size = inherit("device_batch_size", args.device_batch_size, &meta, 32);
    let total_batch_size = inherit("total_batch_size", args.total_batch_size, &meta, 524288);
    let embedding_lr = inherit("embedding_lr", args.embedding_lr, &pretrain_user_config, 0.3);
    let unembedding_lr = inherit("unembedding_lr", args.unembedding_lr, &pretrain_user_config, 0.004);
    let matrix_lr = inherit("matrix_lr", args.matrix_lr, &pretrain_user_config, 0.02);

    let depth = model.config.n_layer;
    let num_flops_per_token = model.estimate_flops();
    let tokens_per_fwdbwd = device_batch_size * max_seq_len;
    let world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size;
    assert!(total_batch_size % world_tokens_per_fwdbwd == 0);
    let grad_accum_steps = total_batch_size / world_tokens_per_fwdbwd;
    print0(&format!(
        "Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd}"
    ));
    print0(&format!(
        "Total batch size {total_batch_size} => gradient accumulation steps: {grad_accum_steps}"
    ));
    let token_bytes = Tensor::from_vec(tokenizer.token_bytes(), tokenizer.get_vocab_size(), &device)?;

    // Initialize the Optimizer. Pretraining ramps weight_decay to zero by the
    // end, so SFT continues with zero.
    let mut optimizer =
        model.setup_optimizer(&varmap, unembedding_lr, embedding_lr, matrix_lr, 0.0, 0.5)?;

    // Optionally warm-start optimizer from the pretrained checkpoint (momentum
    // buffers only; the param-group hyperparameters are kept fresh).
    if args.load_optimizer != 0 {
        match load_optimizer_state(&checkpoint_dir, model_step, &device)? {
            Some(optimizer_data) => {
                optimizer.load_state_dict(&optimizer_data)?;
                print0("Loaded optimizer state from pretrained checkpoint (momentum buffers only, LRs reset)");
            }
            None => print0(
                "WARNING: optimizer checkpoint not found, starting with fresh optimizer (slightly worse)",
            ),
        }
    }

    // Override the initial learning rate as a fraction of the base learning rate
    for group in optimizer.param_groups.iter_mut() {
        group.lr *= args.init_lr_frac;
        group.initial_lr = group.lr;
    }

    // SFT data mixture
    let mut train_tasks: Vec<Box<dyn Task>> = vec![
        Box::new(SmolTalk::new("train", Slice::default())?), // 460K rows of general conversations
    ];
    for _ in 0..args.mmlu_epochs {
        // 100K rows per epoch
        train_tasks.push(Box::new(MMLU::new("all", "auxiliary_train", Slice::default())?));
    }
    for _ in 0..args.gsm8k_epochs {
        // 8K rows per epoch
        train_tasks.push(Box::new(GSM8K::new("main", "train", Slice::default())?));
    }
    let train_dataset = TaskMixture::new(train_tasks);
    print0(&format!(
        "Training mixture: {} rows (MMLU x{}, GSM8K x{})",
        train_dataset.len(),
        args.mmlu_epochs,
        args.gsm8k_epochs
    ));
    let val_dataset = TaskMixture::new(vec![
        Box::new(SmolTalk::new("test", Slice::default())?),
        // use only part of the test sets to match the train ratios
        Box::new(MMLU::new("all", "test", Slice::new(0, Some(5200), 1))?),
        Box::new(GSM8K::new("main", "test", Slice::new(0, Some(420), 1))?),
    ]);

    let mut train_loader = SftDataLoader::new(
        &train_dataset, &tokenizer, device_batch_size, max_seq_len, true, args.num_iterations,
        &device, 100,
    );

    // Learning rate schedule based on progress (0 to 1), because SFT doesn't
    // always know num_iterations in advance (dataset-driven stopping)
    let get_lr_multiplier = |progress: f64| -> f64 {
        if progress < args.warmup_ratio {
            (progress + 1e-8) / args.warmup_ratio
        } else if progress <= 1.0 - args.warmdown_ratio {
            1.0
        } else {
            let decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio;
            (1.0 - decay) + decay * args.final_lr_frac
        }
    };
    // Momentum scheduler for Muon optimizer
    let get_muon_momentum =
        |it: usize| -> f64 { (1.0 - (it as f64 / 300.0).min(1.0)) * 0.85 + (it as f64 / 300.0).min(1.0) * 0.95 };

    let all_vars: Vec<(String, Var)> = {
        let data = varmap.data().lock().unwrap();
        data.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    };

    // Training loop
    let (mut x, mut y) = train_loader.next_batch()?; // prefetch the very first batch
    let mut min_val_bpb = f64::INFINITY;
    let mut val_bpb: Option<f64> = None;
    let mut smooth_train_loss = 0f64;
    let ema_beta = 0.9;
    let mut total_training_time = 0f64;
    let mut progress = 0f64;
    let mut step = 0usize;
    loop {
        let last_step = train_loader.last_step;

        // once in a while: evaluate the val bpb
        if last_step || (args.eval_every > 0 && step % args.eval_every as usize == 0) {
            let mut val_loader = SftDataLoader::new(
                &val_dataset, &tokenizer, device_batch_size, max_seq_len, false, -1, &device, 100,
            );
            let eval_steps = args.eval_tokens / (device_batch_size * max_seq_len * ddp_world_size);
            let mut batches = std::iter::from_fn(|| Some(val_loader.next_batch()));
            let bpb = evaluate_bpb(&model, &mut batches, eval_steps, &token_bytes)?;
            print0(&format!("Step {step:05} | Validation bpb: {bpb:.4}"));
            min_val_bpb = min_val_bpb.min(bpb);
            val_bpb = Some(bpb);
        }

        // once in a while: estimate the ChatCORE metric
        if args.chatcore_every > 0
            && (last_step || (step > 0 && step % args.chatcore_every as usize == 0))
        {
            let engine = Engine::new(&model, &tokenizer);
            let all_tasks = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval"];
            let categorical_tasks = ["ARC-Easy", "ARC-Challenge", "MMLU"];
            let mut task_results: HashMap<&str, f64> = HashMap::new();
            for task_name in all_tasks {
                let limit = if categorical_tasks.contains(&task_name) {
                    args.chatcore_max_cat
                } else {
                    args.chatcore_max_sample
                };
                let max_problems = if limit < 0 { None } else { Some(limit as usize) };
                let acc = run_chat_eval(
                    task_name, &model, &tokenizer, &engine, device_batch_size, 1, 512, 0.0, 50,
                    max_problems,
                )?;
                task_results.insert(task_name, acc);
                print0(&format!("  {task_name}: {:.2}%", 100.0 * acc));
            }
            // mean centered accuracy: 0 = random baseline, 1 = perfect
            let centered_mean = |tasks: &[&str]| -> f64 {
                tasks
                    .iter()
                    .map(|t| {
                        let baseline = if categorical_tasks.contains(t) { 0.25 } else { 0.0 };
                        (task_results[t] - baseline) / (1.0 - baseline)
                    })
                    .sum::<f64>()
                    / tasks.len() as f64
            };
            let chatcore = centered_mean(&all_tasks);
            let chatcore_cat = centered_mean(&categorical_tasks);
            print0(&format!(
                "Step {step:05} | ChatCORE: {chatcore:.4} | ChatCORE_cat: {chatcore_cat:.4}"
            ));
        }

        // save checkpoint at the end of the run
        if last_step {
            let output_dirname = args.model_tag.clone().unwrap_or_else(|| format!("d{depth}"));
            let sft_checkpoint_dir = base_dir.join("chatsft_checkpoints").join(output_dirname);
            let model_data: HashMap<String, Tensor> =
                all_vars.iter().map(|(k, v)| (k.clone(), v.as_tensor().clone())).collect();
            let meta_data = json!({
                "step": step,
                "val_bpb": val_bpb,
                "model_config": {
                    "sequence_len": max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": depth,
                    "n_head": model.config.n_head,
                    "n_kv_head": model.config.n_kv_head,
                    "n_embd": model.config.n_embd,
                    "window_pattern": model.config.window_pattern,
                },
                "user_config": user_config,
            });
            save_checkpoint(&sft_checkpoint_dir, step, &model_data, &meta_data)?;
            let optim_path = sft_checkpoint_dir.join(format!("optim_{step:06}_rank0.safetensors"));
            candle_core::safetensors::save(&optimizer.state_dict()?, &optim_path)?;
        }

        if last_step {
            break;
        }

        // ---------------------------------------------------------------------
        // single training step
        let t0 = Instant::now();
        let mut accum: HashMap<TensorId, Tensor> = HashMap::new();
        let mut train_loss_f = 0f64;
        for _micro_step in 0..grad_accum_steps {
            let loss = model.forward(&x, Some(&y), None, "mean")?;
            train_loss_f = loss.to_scalar::<f32>()? as f64;
            let loss = (loss / grad_accum_steps as f64)?;
            let grads = loss.backward()?;
            for (_, var) in &all_vars {
                if let Some(g) = grads.get(var) {
                    match accum.get_mut(&var.id()) {
                        Some(acc) => *acc = (acc.clone() + g)?,
                        None => {
                            accum.insert(var.id(), g.clone());
                        }
                    }
                }
            }
            (x, y) = train_loader.next_batch()?; // prefetch the next batch
            progress = progress.max(train_loader.approx_progress); // monotonic
        }
        // step the optimizer
        let lrm = get_lr_multiplier(progress);
        let muon_momentum = get_muon_momentum(step);
        for group in optimizer.param_groups.iter_mut() {
            group.lr = group.initial_lr * lrm;
            if group.kind == GroupKind::Muon {
                group.momentum = muon_momentum;
            }
        }
        optimizer.step_with(&|v: &Var| accum.get(&v.id()).cloned())?;
        let dt = t0.elapsed().as_secs_f64();
        // ---------------------------------------------------------------------

        step += 1;

        // logging
        smooth_train_loss = ema_beta * smooth_train_loss + (1.0 - ema_beta) * train_loss_f;
        let debiased_smooth_loss = smooth_train_loss / (1.0 - ema_beta.powi(step as i32 + 1));
        let tok_per_sec = (total_batch_size as f64 / dt) as usize;
        if step > 10 {
            total_training_time += dt;
        }
        print0(&format!(
            "step {step:05} ({:.2}%) | loss: {debiased_smooth_loss:.6} | lrm: {lrm:.2} | \
             dt: {:.2}ms | tok/sec: {tok_per_sec} | epoch: {} | total time: {:.2}m",
            100.0 * progress,
            dt * 1000.0,
            train_loader.current_epoch,
            total_training_time / 60.0
        ));
    }

    print0(&format!("Total training time: {:.2}m", total_training_time / 60.0));
    print0(&format!("Minimum validation bpb: {min_val_bpb:.4}"));
    Ok(())
}
