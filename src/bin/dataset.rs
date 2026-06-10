//! Download pretraining dataset shards.
//!
//!     cargo run --release --bin dataset -- -n 8 -w 4

use clap::Parser;

use nanochat::dataset::{data_dir, download_single_file, MAX_SHARD};

#[derive(Parser, Debug)]
#[command(about = "Download pretraining dataset shards")]
struct Args {
    /// Number of train shards to download (-1 = all)
    #[arg(short = 'n', long, default_value_t = -1)]
    num_files: i64,

    /// Number of parallel download workers
    #[arg(short = 'w', long, default_value_t = 4)]
    num_workers: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Prepare the output directory
    std::fs::create_dir_all(data_dir())?;

    // The user specifies the number of train shards to download via the -n flag.
    // The validation shard is *always* downloaded and is pinned to be the last shard.
    let num_train_shards = if args.num_files == -1 {
        MAX_SHARD
    } else {
        (args.num_files as usize).min(MAX_SHARD)
    };
    let mut ids_to_download: Vec<usize> = (0..num_train_shards).collect();
    ids_to_download.push(MAX_SHARD); // always download the validation shard

    println!(
        "Downloading {} shards using {} workers...",
        ids_to_download.len(),
        args.num_workers
    );
    println!("Target directory: {}", data_dir().display());
    println!();

    // Download the shards with a simple worker pool
    let queue = std::sync::Mutex::new(ids_to_download.into_iter().collect::<Vec<_>>());
    let successful = std::sync::atomic::AtomicUsize::new(0);
    let total = std::sync::Mutex::new(0usize);
    std::thread::scope(|scope| {
        for _ in 0..args.num_workers {
            scope.spawn(|| loop {
                let index = {
                    let mut q = queue.lock().unwrap();
                    if q.is_empty() {
                        break;
                    }
                    q.remove(0)
                };
                *total.lock().unwrap() += 1;
                if download_single_file(index) {
                    successful.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            });
        }
    });

    println!(
        "Done! Downloaded: {}/{} shards to {}",
        successful.load(std::sync::atomic::Ordering::Relaxed),
        total.lock().unwrap(),
        data_dir().display()
    );
    Ok(())
}
