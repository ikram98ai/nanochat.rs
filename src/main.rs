//! Chat with the model on the command line.
//!
//! cargo run --release -- [-i sft] [-p "prompt"] [-t 0.6] [-k 50]

use std::io::{BufRead, Write};

use clap::Parser;

use nanochat::checkpoint_manager::load_model;
use nanochat::common::{autodetect_device_type, compute_init, setup_default_logging};
use nanochat::engine::Engine;

#[derive(Parser, Debug)]
#[command(about = "Chat with the model")]
struct Args {
    /// Source of the model: base|sft|rl
    #[arg(short = 'i', long, default_value = "sft")]
    source: String,

    /// Model tag to load
    #[arg(short = 'g', long)]
    model_tag: Option<String>,

    /// Step to load
    #[arg(short = 's', long)]
    step: Option<usize>,

    /// Prompt the model, get a single response back
    #[arg(short = 'p', long, default_value = "")]
    prompt: String,

    /// Temperature for generation
    #[arg(short = 't', long, default_value_t = 0.6)]
    temperature: f64,

    /// Top-k sampling parameter
    #[arg(short = 'k', long, default_value_t = 50)]
    top_k: usize,

    /// Device type for evaluation. empty => autodetect
    #[arg(long, default_value = "")]
    device_type: String,
}

fn main() -> anyhow::Result<()> {
    setup_default_logging();
    let args = Args::parse();

    // Init the model and tokenizer
    let device_type = if args.device_type.is_empty() {
        autodetect_device_type()
    } else {
        args.device_type.clone()
    };
    let (_ddp, _rank, _local_rank, _world_size, device) = compute_init(&device_type)?;
    let (model, tokenizer, _meta) =
        load_model(&args.source, &device, "eval", args.model_tag.as_deref(), args.step)?;

    // Special tokens for the chat state machine
    let bos = tokenizer.get_bos_token_id();
    let user_start = tokenizer.encode_special("<|user_start|>");
    let user_end = tokenizer.encode_special("<|user_end|>");
    let assistant_start = tokenizer.encode_special("<|assistant_start|>");
    let assistant_end = tokenizer.encode_special("<|assistant_end|>");

    // Create Engine for efficient generation
    let engine = Engine::new(&model, &tokenizer);

    println!("\nNanoChat Interactive Mode");
    println!("{}", "-".repeat(50));
    println!("Type 'quit' or 'exit' to end the conversation");
    println!("Type 'clear' to start a new conversation");
    println!("{}", "-".repeat(50));

    let mut conversation_tokens: Vec<u32> = vec![bos];
    let stdin = std::io::stdin();

    loop {
        let user_input = if !args.prompt.is_empty() {
            // Get the prompt from the launch command
            args.prompt.clone()
        } else {
            // Get the prompt interactively from the console
            print!("\nUser: ");
            std::io::stdout().flush()?;
            let mut line = String::new();
            if stdin.lock().read_line(&mut line)? == 0 {
                println!("\nGoodbye!");
                break;
            }
            line.trim().to_string()
        };

        // Handle special commands
        let lower = user_input.to_lowercase();
        if lower == "quit" || lower == "exit" {
            println!("Goodbye!");
            break;
        }
        if lower == "clear" {
            conversation_tokens = vec![bos];
            println!("Conversation cleared.");
            continue;
        }
        if user_input.is_empty() {
            continue;
        }

        // Add User message to the conversation
        conversation_tokens.push(user_start);
        conversation_tokens.extend(tokenizer.encode(&user_input, None, None));
        conversation_tokens.push(user_end);

        // Kick off the assistant
        conversation_tokens.push(assistant_start);
        let mut response_tokens: Vec<u32> = Vec::new();
        print!("\nAssistant: ");
        std::io::stdout().flush()?;
        let stream = engine.generate(
            &conversation_tokens,
            1,            // num_samples
            Some(256),    // max_tokens
            args.temperature,
            Some(args.top_k),
            42,
        )?;
        for step in stream {
            let (token_column, _token_masks) = step?;
            let token = token_column[0]; // pop the batch dimension (num_samples=1)
            response_tokens.push(token);
            let token_text = tokenizer.decode(&[token]);
            print!("{token_text}");
            std::io::stdout().flush()?;
        }
        println!();
        // we have to ensure that the assistant end token is the last token
        // so even if generation ends due to max tokens, we have to append it to the end
        if response_tokens.last() != Some(&assistant_end) {
            response_tokens.push(assistant_end);
        }
        conversation_tokens.extend(&response_tokens);

        // In the prompt mode, we only want a single response and exit
        if !args.prompt.is_empty() {
            break;
        }
    }

    Ok(())
}
