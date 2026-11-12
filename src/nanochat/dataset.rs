//! The base/pretraining dataset is a set of parquet files.
//! This file contains utilities for:
//! - iterating over the parquet files and yielding documents from it
//! - downloading the files on demand if they are not on disk

use std::fs::File;
use std::path::PathBuf;

use candle_core::Result;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;

use super::common::get_base_dir;

// The URL on the internet where the data is hosted and downloaded from on demand
pub const BASE_URL: &str =
    "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main";
/// the last datashard is shard_06542.parquet
pub const MAX_SHARD: usize = 6542;

pub fn index_to_filename(index: usize) -> String {
    format!("shard_{index:05}.parquet")
}

pub fn data_dir() -> PathBuf {
    get_base_dir().join("base_data_climbmix")
}

fn msg(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

/// Looks into a data dir and returns full paths to all parquet files, sorted.
pub fn list_parquet_files(
    dir: Option<&std::path::Path>,
    warn_on_legacy: bool,
) -> Result<Vec<PathBuf>> {
    let mut dir = dir.map(|d| d.to_path_buf()).unwrap_or_else(data_dir);

    // Legacy-supporting code due to the upgrade from FinewebEdu-100B to ClimbMix-400B
    if !dir.exists() {
        if warn_on_legacy {
            println!();
            println!("{}", "=".repeat(80));
            println!("  WARNING: DATASET UPGRADE REQUIRED");
            println!("{}", "=".repeat(80));
            println!();
            println!("  Could not find: {}", dir.display());
            println!();
            println!("  nanochat recently switched from FinewebEdu-100B to ClimbMix-400B.");
            println!("  To upgrade to the new ClimbMix-400B dataset, download the new shards");
            println!("  and re-train the tokenizer on the new ClimbMix data.");
            println!();
            println!("  For now, falling back to your old FinewebEdu-100B dataset...");
            println!("{}", "=".repeat(80));
            println!();
        }
        // attempt a fallback to the legacy data directory
        dir = get_base_dir().join("base_data");
    }

    let mut files: Vec<String> = std::fs::read_dir(&dir)
        .map_err(|e| msg(format!("Error listing {}: {e}", dir.display())))?
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|f| f.ends_with(".parquet"))
        .collect();
    files.sort();
    Ok(files.into_iter().map(|f| dir.join(f)).collect())
}

/// Number of row groups in a parquet file.
pub fn num_row_groups(path: &std::path::Path) -> Result<usize> {
    let file = File::open(path).map_err(msg)?;
    let reader = SerializedFileReader::new(file).map_err(msg)?;
    Ok(reader.metadata().num_row_groups())
}

/// Read the 'text' column of one row group as strings.
pub fn read_row_group_texts(path: &std::path::Path, rg_idx: usize) -> Result<Vec<String>> {
    let file = File::open(path).map_err(msg)?;
    let reader = SerializedFileReader::new(file).map_err(msg)?;
    let rg = reader.get_row_group(rg_idx).map_err(msg)?;
    let mut texts = Vec::new();
    for row in rg.get_row_iter(None).map_err(msg)? {
        let row = row.map_err(msg)?;
        for (name, field) in row.get_column_iter() {
            if name == "text" {
                match field {
                    Field::Str(s) => texts.push(s.clone()),
                    other => return Err(msg(format!("unexpected 'text' field: {other:?}"))),
                }
            }
        }
    }
    Ok(texts)
}

/// Iterate through the dataset, in batches of underlying row groups.
/// - split can be "train" or "val": the last parquet file is val.
/// - start/step are useful for skipping rows in DDP (start=rank, step=world_size).
#[allow(dead_code)]
pub fn parquets_iter_batched(
    split: &str,
    start: usize,
    step: usize,
) -> Result<impl Iterator<Item = Result<Vec<String>>>> {
    assert!(
        split == "train" || split == "val",
        "split must be 'train' or 'val'"
    );
    let paths = list_parquet_files(None, false)?;
    if paths.is_empty() {
        return Err(msg(
            "No dataset parquet files found: download the shards first (make data)",
        ));
    }
    let paths: Vec<PathBuf> = if split == "train" {
        paths[..paths.len() - 1].to_vec()
    } else {
        paths[paths.len() - 1..].to_vec()
    };
    let mut iter_state: Vec<(PathBuf, usize)> = Vec::new();
    for path in paths {
        let n = num_row_groups(&path)?;
        let mut rg = start;
        while rg < n {
            iter_state.push((path.clone(), rg));
            rg += step;
        }
    }
    Ok(iter_state
        .into_iter()
        .map(|(path, rg)| read_row_group_texts(&path, rg)))
}

/// Downloads a single file index, with some backoff.
pub fn download_single_file(index: usize) -> bool {
    let filename = index_to_filename(index);
    let filepath = data_dir().join(&filename);
    if filepath.exists() {
        println!("Skipping {} (already exists)", filepath.display());
        return true;
    }

    let url = format!("{BASE_URL}/{filename}");
    println!("Downloading {filename}...");

    // Write to a temporary file first, then move it to the final location. A
    // failed attempt keeps what it got, and the next one resumes from there, so
    // a connection that keeps dropping still makes progress.
    let temp_path = filepath.with_extension("parquet.tmp");
    let max_attempts = 10;
    for attempt in 1..=max_attempts {
        let result = (|| -> anyhow::Result<()> {
            let client = reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(30 * 60))
                .connect_timeout(std::time::Duration::from_secs(30))
                .build()?;
            let have = std::fs::metadata(&temp_path).map(|m| m.len()).unwrap_or(0);
            let mut request = client.get(&url);
            if have > 0 {
                request = request.header(reqwest::header::RANGE, format!("bytes={have}-"));
            }
            let response = request.send()?;
            if response.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
                // the partial file does not match the remote one: start over
                std::fs::remove_file(&temp_path)?;
                anyhow::bail!("cannot resume, restarting the download");
            }
            let mut response = response.error_for_status()?;
            let mut f = if response.status() == reqwest::StatusCode::PARTIAL_CONTENT {
                std::fs::OpenOptions::new().append(true).open(&temp_path)?
            } else {
                File::create(&temp_path)? // the server sent the whole file
            };
            std::io::copy(&mut response, &mut f)?;
            std::fs::rename(&temp_path, &filepath)?;
            Ok(())
        })();
        match result {
            Ok(()) => {
                println!("Successfully downloaded {filename}");
                return true;
            }
            Err(e) => {
                println!("Attempt {attempt}/{max_attempts} failed for {filename}: {e}");
                if attempt < max_attempts {
                    // exponential backoff: 2^attempt seconds, at most 30
                    let wait = (1u64 << attempt).min(30);
                    println!("Waiting {wait} seconds before retry...");
                    std::thread::sleep(std::time::Duration::from_secs(wait));
                } else {
                    println!("Failed to download {filename} after {max_attempts} attempts");
                }
            }
        }
    }
    false
}
