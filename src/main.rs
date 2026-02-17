mod nanochat;

use clap::Parser;
use std::io::Write;
use crate::nanochat::common::{setup_default_logging, autodetect_device_type, compute_init};
use crate::nanochat::checkpoint_manager::load_model;
use crate::nanochat::engine::Engine;
use crate::nanochat::report::{get_report, ReportItem};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'i', long, default_value = "sft")]
    source: String,
    
    #[arg(short = 'g', long)]
    model_tag: Option<String>,
    
    #[arg(short = 's', long)]
    step: Option<usize>,
    
    #[arg(short = 'p', long, default_value = "")]
    prompt: String,
    
    #[arg(short = 't', long, default_value_t = 0.6)]
    temperature: f64,
    
    #[arg(short = 'k', long, default_value_t = 50)]
    top_k: usize,
    
    #[arg(long, default_value = "")]
    device_type: String,
    
    #[arg(short = 'd', long, default_value = "bfloat16")]
    dtype: String,
}

fn main() -> anyhow::Result<()> {
    setup_default_logging();
    let args = Args::parse();
    
    let device_type = if args.device_type.is_empty() {
        autodetect_device_type()
    } else {
        args.device_type
    };
    
    let (_, _, _, _, device) = compute_init(&device_type)?;
    
    // Load model
    let (model, tokenizer, _) = load_model(&args.source, &device)?;
    
    // Model metadata
    let flops = model.estimate_flops(2048);
    let params = model.num_scaling_params();
    log::info!("Model estimate: {:.2} TFLOPs peak, {} scaling params", flops / 1e12, params);

    // Initial report
    let report = get_report();
    report.log("System", &[
        ReportItem::Text { key: "device".into(), value: device_type.clone() },
        ReportItem::Text { key: "dtype".into(), value: args.dtype.clone() },
    ]);
    
    // Special tokens
    let bos = tokenizer.get_bos_token_id();
    let user_start = tokenizer.encode_special("<|user_start|>");
    let user_end = tokenizer.encode_special("<|user_end|>");
    let assistant_start = tokenizer.encode_special("<|assistant_start|>");
    let assistant_end = tokenizer.encode_special("<|assistant_end|>");
    
    let mut engine = Engine::new(model, tokenizer);
    
    println!("\nNanoChat Interactive Mode");
    println!("{}", "-".repeat(50));
    println!("Type 'quit' or 'exit' to end the conversation");
    println!("Type 'clear' to start a new conversation");
    println!("{}", "-".repeat(50));
    
    let mut conversation_tokens = vec![bos];
    
    loop {
        let user_input = if !args.prompt.is_empty() {
            args.prompt.clone()
        } else {
            let mut input = String::new();
            print!("\nUser: ");
            std::io::stdout().flush()?;
            std::io::stdin().read_line(&mut input)?;
            let trimmed = input.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.eq_ignore_ascii_case("quit") || trimmed.eq_ignore_ascii_case("exit") {
                println!("Goodbye!");
                break;
            }
            if trimmed.eq_ignore_ascii_case("clear") {
                conversation_tokens = vec![bos];
                println!("Conversation cleared.");
                continue;
            }
            trimmed.to_string()
        };
        
        // Add User message
        conversation_tokens.push(user_start);
        conversation_tokens.extend(engine.tokenizer.encode(&user_input, false)?);
        conversation_tokens.push(user_end);
        
        conversation_tokens.push(assistant_start);
        
        print!("\nAssistant: ");
        std::io::stdout().flush()?;
        
        let tokenizer = engine.tokenizer.clone();
        let gen_iter = engine.generate(&conversation_tokens, 256, args.temperature, args.top_k)?;
        let mut response_tokens = Vec::new();
        
        for token in gen_iter {
            response_tokens.push(token);
            let token_str = tokenizer.decode(&[token], true)?; 
            print!("{}", token_str);
            std::io::stdout().flush()?;
        }
        println!();
        
        if *response_tokens.last().unwrap_or(&0) != assistant_end {
            response_tokens.push(assistant_end);
        }
        conversation_tokens.extend(response_tokens);
        
        if !args.prompt.is_empty() {
            break;
        }
    }

    Ok(())
}