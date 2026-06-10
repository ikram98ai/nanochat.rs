//! Train the base model. From the root directory of the project, run as:
//!
//!     cargo run --release --bin base_train -- --depth=4 --max-seq-len=512 \
//!         --device-batch-size=1 --eval-tokens=512 --total-batch-size=512 --num-iterations=20

use std::collections::HashMap;
use std::time::Instant;

use candle_core::{DType, Tensor, TensorId, Var};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use serde_json::json;

use nanochat::checkpoint_manager::save_checkpoint;
use nanochat::common::{autodetect_device_type, compute_init, get_base_dir, print0, print_banner, setup_default_logging};
use nanochat::dataloader::{DataLoader, LoaderState};
use nanochat::engine::Engine;
use nanochat::gpt::{has_ve, GPTConfig, GPT};
use nanochat::loss_eval::evaluate_bpb;
use nanochat::optim::GroupKind;
use nanochat::tokenizer::get_tokenizer;

#[derive(Parser, Debug, Clone, serde::Serialize)]
#[command(about = "Pretrain base model")]
struct Args {
    /// device type: cpu (empty = autodetect)
    #[arg(long, default_value = "")]
    device_type: String,
    // Model architecture
    /// depth of the Transformer model
    #[arg(long, default_value_t = 20)]
    depth: usize,
    /// model_dim = depth * aspect_ratio
    #[arg(long, default_value_t = 64)]
    aspect_ratio: usize,
    /// target head dimension for attention
    #[arg(long, default_value_t = 128)]
    head_dim: usize,
    /// max context length
    #[arg(long, default_value_t = 2048)]
    max_seq_len: usize,
    /// sliding window pattern tiled across layers (e.g. 'SSL')
    #[arg(long, default_value = "SSSL")]
    window_pattern: String,
    // Training horizon (only one used, in order of precedence)
    /// explicit number of optimization steps (-1 = disable)
    #[arg(long, default_value_t = -1)]
    num_iterations: i64,
    /// calculate num_iterations to reach target_flops (-1 = disable)
    #[arg(long, default_value_t = -1.0)]
    target_flops: f64,
    /// calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)
    #[arg(long, default_value_t = 12.0)]
    target_param_data_ratio: f64,
    // Optimization
    /// per-device batch size
    #[arg(long, default_value_t = 32)]
    device_batch_size: usize,
    /// total batch size in tokens (-1 = auto-compute optimal)
    #[arg(long, default_value_t = -1)]
    total_batch_size: i64,
    /// learning rate for embedding parameters (Adam)
    #[arg(long, default_value_t = 0.3)]
    embedding_lr: f64,
    /// learning rate for unembedding parameters (Adam)
    #[arg(long, default_value_t = 0.008)]
    unembedding_lr: f64,
    /// cautious weight decay for the Muon optimizer (for weights)
    #[arg(long, default_value_t = 0.28)]
    weight_decay: f64,
    /// learning rate for matrix parameters (Muon)
    #[arg(long, default_value_t = 0.02)]
    matrix_lr: f64,
    /// learning rate for scalars (resid_lambdas, x0_lambdas)
    #[arg(long, default_value_t = 0.5)]
    scalar_lr: f64,
    /// number of steps for LR warmup
    #[arg(long, default_value_t = 40)]
    warmup_steps: usize,
    /// ratio of iterations for LR warmdown
    #[arg(long, default_value_t = 0.65)]
    warmdown_ratio: f64,
    /// final LR as fraction of initial LR
    #[arg(long, default_value_t = 0.05)]
    final_lr_frac: f64,
    // Evaluation
    /// evaluate val bpb every N steps (-1 = disable)
    #[arg(long, default_value_t = 250)]
    eval_every: i64,
    /// number of tokens to evaluate val loss on
    #[arg(long, default_value_t = 80 * 524288)]
    eval_tokens: usize,
    /// sample from model every N steps (-1 = disable)
    #[arg(long, default_value_t = 2000)]
    sample_every: i64,
    /// save checkpoints every N steps (-1 = only at end)
    #[arg(long, default_value_t = -1)]
    save_every: i64,
    // Output
    /// override model tag for checkpoint directory name
    #[arg(long)]
    model_tag: Option<String>,
}

/// Python-style round-half-to-even for the schedule/batch computations.
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// Model dim is nudged up to the nearest multiple of head_dim for clean division.
fn build_config(args: &Args, depth: usize, vocab_size: usize) -> GPTConfig {
    let base_dim = depth * args.aspect_ratio;
    let model_dim = ((base_dim + args.head_dim - 1) / args.head_dim) * args.head_dim;
    let num_heads = model_dim / args.head_dim;
    GPTConfig {
        sequence_len: args.max_seq_len,
        vocab_size,
        n_layer: depth,
        n_head: num_heads,
        n_kv_head: num_heads,
        n_embd: model_dim,
        window_pattern: args.window_pattern.clone(),
    }
}

/// transformer matrices + lm_head give the cleanest scaling laws.
fn get_scaling_params(config: &GPTConfig) -> usize {
    let padded_vocab = (config.vocab_size + 63) / 64 * 64;
    let n_embd = config.n_embd;
    let head_dim = n_embd / config.n_head;
    let kv_dim = config.n_kv_head * head_dim;
    let mut matrices = 0;
    for i in 0..config.n_layer {
        matrices += n_embd * n_embd; // c_q
        matrices += 2 * kv_dim * n_embd; // c_k, c_v
        matrices += n_embd * n_embd; // c_proj
        if has_ve(i, config.n_layer) {
            matrices += config.n_kv_head * 12; // ve_gate
        }
        matrices += 2 * 4 * n_embd * n_embd; // mlp c_fc, c_proj
    }
    matrices + padded_vocab * n_embd // + lm_head
}

fn main() -> anyhow::Result<()> {
    setup_default_logging();
    print_banner();
    let args = Args::parse();
    let user_config = serde_json::to_value(&args)?;

    let device_type = if args.device_type.is_empty() {
        autodetect_device_type()
    } else {
        args.device_type.clone()
    };
    let (_ddp, _rank, _local_rank, ddp_world_size, device) = compute_init(&device_type)?;

    // Tokenizer is useful for evaluation and we need the vocab size to init the model
    let tokenizer = get_tokenizer()?;
    let token_bytes_vec = tokenizer.token_bytes();
    let token_bytes = Tensor::from_vec(token_bytes_vec, tokenizer.get_vocab_size(), &device)?;
    let vocab_size = tokenizer.get_vocab_size();
    print0(&format!("Vocab size: {vocab_size}"));

    // Initialize the model (trainable: parameters live in a VarMap)
    let config = build_config(&args, args.depth, vocab_size);
    print0(&format!("Model config:\n{}", serde_json::to_string_pretty(&config)?));
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GPT::new(config.clone(), vb)?;
    let init = GPT::init_weights(&config, 64, &device, 42)?;
    varmap.set(init.iter().map(|(k, v)| (k, v)))?;

    let base_dir = get_base_dir();
    let output_dirname = args.model_tag.clone().unwrap_or_else(|| format!("d{}", args.depth));
    let checkpoint_dir = base_dir.join("base_checkpoints").join(&output_dirname);

    // Scaling laws and muP extrapolations: training horizon, batch size, LRs, weight decay
    let param_counts = model.num_scaling_params();
    print0("Parameter counts:");
    for key in ["wte", "value_embeds", "lm_head", "transformer_matrices", "scalars", "total"] {
        print0(&format!("{key:24}: {}", param_counts[key]));
    }
    let num_flops_per_token = model.estimate_flops();
    print0(&format!("Estimated FLOPs per token: {num_flops_per_token:e}"));

    // 1) optimal training horizon in tokens: data:param ratio over the scaling params
    let num_scaling_params = get_scaling_params(&config);
    let target_tokens = (args.target_param_data_ratio * num_scaling_params as f64) as i64;

    // Our reference model is d12: hyperparameters are tuned there and transferred (muP style)
    let d12_config = build_config(&args, 12, vocab_size);
    let d_ref = args.target_param_data_ratio * get_scaling_params(&d12_config) as f64;
    let b_ref = (1u64 << 19) as f64; // optimal batch size at d12 ~= 524,288 tokens

    // 2) optimal batch size follows the Power Lines paper: Bopt ∝ D^0.383
    let total_batch_size = if args.total_batch_size == -1 {
        let batch_size_ratio = target_tokens as f64 / d_ref;
        let predicted_batch_size = b_ref * batch_size_ratio.powf(0.383);
        let total = 2f64.powf(round_half_even(predicted_batch_size.log2())) as usize;
        print0(&format!("Auto-computed optimal batch size: {total} tokens"));
        total
    } else {
        args.total_batch_size as usize
    };

    // 3) learning rate correction: η ∝ √(B/B_ref)
    let batch_ratio = total_batch_size as f64 / b_ref;
    let batch_lr_scale = if batch_ratio != 1.0 {
        let scale = batch_ratio.sqrt();
        print0(&format!("Scaling LRs by {scale:.4} for batch size {total_batch_size} (reference: {b_ref})"));
        scale
    } else {
        1.0
    };

    // 4) weight decay scaling to keep T_epoch = B/(η·λ·D) constant: λ = λ_ref·√(B/B_ref)·(D_ref/D)
    let weight_decay_scaled =
        args.weight_decay * (total_batch_size as f64 / b_ref).sqrt() * (d_ref / target_tokens as f64);
    if weight_decay_scaled != args.weight_decay {
        print0(&format!(
            "Scaling weight decay from {:.6} to {weight_decay_scaled:.6} for depth {}",
            args.weight_decay, args.depth
        ));
    }

    // Initialize the combined MuonAdamW optimizer
    let mut optimizer = model.setup_optimizer(
        &varmap,
        args.unembedding_lr * batch_lr_scale,
        args.embedding_lr * batch_lr_scale,
        args.matrix_lr * batch_lr_scale,
        weight_decay_scaled,
        args.scalar_lr * batch_lr_scale,
    )?;

    // Initialize the DataLoaders for train/val
    let mut train_loader = DataLoader::new(
        &tokenizer, args.device_batch_size, args.max_seq_len, "train", None, 128, &device, None, 1000,
    )?;
    let (mut x, mut y, mut dataloader_state): (Tensor, Tensor, LoaderState) = train_loader.next_batch()?;

    // num_iterations: either given, or from target flops, or from target data:param ratio
    assert!(args.num_iterations > 0 || args.target_param_data_ratio > 0.0 || args.target_flops > 0.0);
    let num_iterations: usize = if args.num_iterations > 0 {
        print0(&format!("Using user-provided number of iterations: {}", args.num_iterations));
        args.num_iterations as usize
    } else if args.target_flops > 0.0 {
        let n = round_half_even(args.target_flops / (num_flops_per_token as f64 * total_batch_size as f64)) as usize;
        print0(&format!("Calculated number of iterations from target FLOPs: {n}"));
        n
    } else {
        let n = (target_tokens as usize) / total_batch_size;
        print0(&format!("Calculated number of iterations from target data:param ratio: {n}"));
        n
    };
    let total_tokens = total_batch_size * num_iterations;
    print0(&format!("Total number of training tokens: {total_tokens}"));
    print0(&format!(
        "Tokens : Scaling params ratio: {:.2}",
        (total_batch_size * num_iterations) as f64 / num_scaling_params as f64
    ));
    print0(&format!("Total training FLOPs estimate: {:e}", num_flops_per_token as f64 * total_tokens as f64));

    // Learning rate schedule (linear warmup, constant, linear warmdown)
    let warmdown_iters = round_half_even(args.warmdown_ratio * num_iterations as f64) as usize;
    let get_lr_multiplier = |it: usize| -> f64 {
        if it < args.warmup_steps {
            (it + 1) as f64 / args.warmup_steps as f64
        } else if it <= num_iterations - warmdown_iters {
            1.0
        } else {
            let progress = (num_iterations - it) as f64 / warmdown_iters as f64;
            progress + (1.0 - progress) * args.final_lr_frac
        }
    };
    // Momentum scheduler for Muon (warms up to 0.97, down to 0.90 during LR warmdown)
    let warmdown_start = num_iterations - warmdown_iters;
    let get_muon_momentum = |it: usize| -> f64 {
        if it < 400 {
            let frac = it as f64 / 400.0;
            (1.0 - frac) * 0.85 + frac * 0.97
        } else if it >= warmdown_start {
            let progress = (it - warmdown_start) as f64 / warmdown_iters as f64;
            0.97 * (1.0 - progress) + 0.90 * progress
        } else {
            0.97
        }
    };
    // Weight decay scheduler for Muon (cosine decay to zero over the course of training)
    let get_weight_decay = |it: usize| -> f64 {
        weight_decay_scaled * 0.5 * (1.0 + (std::f64::consts::PI * it as f64 / num_iterations as f64).cos())
    };

    // Loop state
    let mut step = 0usize;
    let mut val_bpb: Option<f64> = None;
    let mut min_val_bpb = f64::INFINITY;
    let mut smooth_train_loss = 0f64; // EMA of training loss
    let mut total_training_time = 0f64;

    // gradient accumulation micro-steps to reach the desired total batch size
    let tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len;
    let world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size;
    assert!(total_batch_size % world_tokens_per_fwdbwd == 0);
    let grad_accum_steps = total_batch_size / world_tokens_per_fwdbwd;
    print0(&format!(
        "Tokens / micro-batch / rank: {} x {} = {tokens_per_fwdbwd}",
        args.device_batch_size, args.max_seq_len
    ));
    print0(&format!("Total batch size {total_batch_size} => gradient accumulation steps: {grad_accum_steps}"));

    let all_vars: Vec<(String, Var)> = {
        let data = varmap.data().lock().unwrap();
        data.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    };

    // Go!
    loop {
        let last_step = step == num_iterations; // +1 loop so we can eval/save at the end

        // once in a while: evaluate the val bpb
        if args.eval_every > 0 && (last_step || step % args.eval_every as usize == 0) {
            let mut val_loader = DataLoader::new(
                &tokenizer, args.device_batch_size, args.max_seq_len, "val", None, 128, &device, None, 1000,
            )?;
            let eval_steps = args.eval_tokens / (args.device_batch_size * args.max_seq_len * ddp_world_size);
            let mut batches = std::iter::from_fn(|| {
                Some(val_loader.next_batch().map(|(x, y, _)| (x, y)))
            });
            let bpb = evaluate_bpb(&model, &mut batches, eval_steps, &token_bytes)?;
            print0(&format!("Step {step:05} | Validation bpb: {bpb:.6}"));
            min_val_bpb = min_val_bpb.min(bpb);
            val_bpb = Some(bpb);
        }

        // once in a while: sample from the model
        if args.sample_every > 0 && (last_step || (step > 0 && step % args.sample_every as usize == 0)) {
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
            for prompt in prompts {
                let tokens = tokenizer.encode(prompt, Some("<|bos|>"), None);
                let (sample, _) = engine.generate_batch(&tokens, 1, Some(16), 0.0, None, 42)?;
                print0(&tokenizer.decode(&sample[0]));
            }
        }

        // save checkpoint: at the end of the run, or every save_every steps (not at step 0)
        if last_step || (step > 0 && args.save_every > 0 && step % args.save_every as usize == 0) {
            let model_data: HashMap<String, Tensor> = all_vars
                .iter()
                .map(|(k, v)| (k.clone(), v.as_tensor().clone()))
                .collect();
            let meta_data = json!({
                "step": step,
                "val_bpb": val_bpb,
                "model_config": config,
                "user_config": user_config,
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state,
                "loop_state": {
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            });
            save_checkpoint(&checkpoint_dir, step, &model_data, &meta_data)?;
            let optim_path = checkpoint_dir.join(format!("optim_{step:06}_rank0.safetensors"));
            candle_core::safetensors::save(&optimizer.state_dict()?, &optim_path)?;
        }

        // termination conditions
        if last_step {
            break;
        }

        // ---------------------------------------------------------------------
        // single training step: evaluate the gradient with accumulation
        let t0 = Instant::now();
        let mut accum: HashMap<TensorId, Tensor> = HashMap::new();
        let mut train_loss_f = 0f64;
        for _micro_step in 0..grad_accum_steps {
            let loss = model.forward(&x, Some(&y), None, "mean")?;
            train_loss_f = loss.to_scalar::<f32>()? as f64; // for logging
            // each backward is a grad sum => normalize loss here
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
            (x, y, dataloader_state) = train_loader.next_batch()?; // fetch the next batch
        }
        // step the optimizer with the scheduled lr/momentum/weight decay
        let lrm = get_lr_multiplier(step);
        let muon_momentum = get_muon_momentum(step);
        let muon_weight_decay = get_weight_decay(step);
        for group in optimizer.param_groups.iter_mut() {
            group.lr = group.initial_lr * lrm;
            if group.kind == GroupKind::Muon {
                group.momentum = muon_momentum;
                group.weight_decay = muon_weight_decay;
            }
        }
        optimizer.step_with(&|v: &Var| accum.get(&v.id()).cloned())?;
        let dt = t0.elapsed().as_secs_f64();
        // ---------------------------------------------------------------------

        // logging
        let ema_beta = 0.9; // EMA decay factor for some smoothing just for nicer logging
        smooth_train_loss = ema_beta * smooth_train_loss + (1.0 - ema_beta) * train_loss_f;
        let debiased_smooth_loss = smooth_train_loss / (1.0 - ema_beta.powi(step as i32 + 1));
        let pct_done = 100.0 * step as f64 / num_iterations as f64;
        let tok_per_sec = (total_batch_size as f64 / dt) as usize;
        if step > 10 {
            total_training_time += dt; // only count the time after the first 10 steps
        }
        let steps_done = step as i64 - 10;
        let eta_str = if steps_done > 0 {
            let avg = total_training_time / steps_done as f64;
            format!(" | eta: {:.1}m", (num_iterations - step) as f64 * avg / 60.0)
        } else {
            String::new()
        };
        let epoch = format!(
            "{} pq: {} rg: {}",
            dataloader_state.epoch, dataloader_state.pq_idx, dataloader_state.rg_idx
        );
        print0(&format!(
            "step {step:05}/{num_iterations:05} ({pct_done:.2}%) | loss: {debiased_smooth_loss:.6} | \
             lrm: {lrm:.2} | dt: {:.2}ms | tok/sec: {tok_per_sec} | epoch: {epoch} | \
             total time: {:.2}m{eta_str}",
            dt * 1000.0,
            total_training_time / 60.0
        ));

        step += 1;
    }

    print0(&format!("Total training time: {:.2}m", total_training_time / 60.0));
    if val_bpb.is_some() {
        print0(&format!("Minimum validation bpb: {min_val_bpb:.6}"));
    }
    Ok(())
}
