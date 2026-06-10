//! Train a tokenizer using our own BPE Tokenizer library.
//! In the style of GPT-4 tokenizer.
//!
//!     cargo run --release --bin tok_train -- --vocab-size 32768

use clap::Parser;

use nanochat::common::get_base_dir;
use nanochat::dataset::parquets_iter_batched;
use nanochat::tokenizer::Tokenizer;

#[derive(Parser, Debug)]
#[command(about = "Train a BPE tokenizer")]
struct Args {
    /// Maximum characters to train on
    #[arg(long, default_value_t = 2_000_000_000)]
    max_chars: usize,

    /// Maximum characters per document
    #[arg(long, default_value_t = 10_000)]
    doc_cap: usize,

    /// Vocabulary size
    #[arg(long, default_value_t = 32768)]
    vocab_size: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    println!("max_chars: {}", args.max_chars);
    println!("doc_cap: {}", args.doc_cap);
    println!("vocab_size: {}", args.vocab_size);

    // Text iterator: flatten the batches, crop every document to doc_cap
    // characters, stop after max_chars characters.
    let mut nchars = 0usize;
    let mut done = false;
    let batches = parquets_iter_batched("train", 0, 1)?;
    let text_iter = batches.flat_map(|batch| batch.unwrap_or_default()).map_while(move |doc| {
        if done {
            return None;
        }
        let mut doc_text = doc;
        if doc_text.chars().count() > args.doc_cap {
            doc_text = doc_text.chars().take(args.doc_cap).collect();
        }
        nchars += doc_text.chars().count();
        done = nchars > args.max_chars; // the current document is still included
        Some(doc_text)
    });

    // Train the tokenizer
    let t0 = std::time::Instant::now();
    let tokenizer = Tokenizer::train_from_iterator(text_iter, args.vocab_size)?;
    let train_time = t0.elapsed().as_secs_f64();
    println!("Training time: {train_time:.2}s");

    // Save the tokenizer to disk
    let tokenizer_dir = get_base_dir().join("tokenizer");
    tokenizer.save(&tokenizer_dir)?;

    // Quick inline sanity check
    let test_text = "Hello world! This is a test.\nNumbers: 123, 4567, 89\nContractions: I'm, you're, it's\nSpecial chars: @#$%^&*()\nUnicode: 你好世界 🌍";
    let encoded = tokenizer.encode(test_text, None, None);
    let decoded = tokenizer.decode(&encoded);
    assert!(decoded == test_text, "encode/decode round-trip failed");

    // token id -> byte length mapping stats (the bits-per-byte normalization);
    // the Rust port computes token_bytes from the tokenizer on the fly.
    let token_bytes = tokenizer.token_bytes();
    let nonzero: Vec<f64> = token_bytes.iter().filter(|&&b| b > 0).map(|&b| b as f64).collect();
    let mean = nonzero.iter().sum::<f64>() / nonzero.len() as f64;
    let var = nonzero.iter().map(|b| (b - mean).powi(2)).sum::<f64>() / (nonzero.len() - 1) as f64;
    println!("token_bytes_min: {}", nonzero.iter().cloned().fold(f64::INFINITY, f64::min));
    println!("token_bytes_max: {}", nonzero.iter().cloned().fold(0.0, f64::max));
    println!("token_bytes_mean: {mean:.4}");
    println!("token_bytes_std: {:.4}", var.sqrt());
    Ok(())
}
