//! Minimal stand-in for HuggingFace `datasets.load_dataset(repo_id, subset, split=split)`.
//!
//! Every dataset on the hub has an auto-generated parquet export. We list the
//! parquet shards via the hub API, download them (once) into the local cache
//! directory, and read them with the parquet crate. The cache layout is the
//! same as the Python reference's ({base_dir}/task_data/...), so the two share
//! downloads when they point at the same base dir (set NANOCHAT_BASE_DIR for the
//! Python side, which defaults to ~/.cache/nanochat rather than ./models).

use std::fs::File;
use std::path::{Path, PathBuf};

use candle_core::Result;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Field, Row};
use serde_json::Value;

use crate::nanochat::common::get_base_dir;
use crate::nanochat::nprandom::permutation;

fn msg(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

fn http_client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .user_agent("nanochat")
        .timeout(std::time::Duration::from_secs(60 * 60))
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(msg)
}

/// Convert one parquet value to JSON (the equivalent of pyarrow's `.as_py()`).
fn field_to_json(field: &Field) -> Result<Value> {
    Ok(match field {
        Field::Null => Value::Null,
        Field::Bool(v) => Value::from(*v),
        Field::Byte(v) => Value::from(*v),
        Field::Short(v) => Value::from(*v),
        Field::Int(v) => Value::from(*v),
        Field::Long(v) => Value::from(*v),
        Field::UByte(v) => Value::from(*v),
        Field::UShort(v) => Value::from(*v),
        Field::UInt(v) => Value::from(*v),
        Field::ULong(v) => Value::from(*v),
        Field::Float(v) => Value::from(*v),
        Field::Double(v) => Value::from(*v),
        Field::Str(v) => Value::from(v.as_str()),
        Field::Group(row) => row_to_json(row)?,
        Field::ListInternal(list) => Value::Array(
            list.elements()
                .iter()
                .map(field_to_json)
                .collect::<Result<_>>()?,
        ),
        other => {
            return Err(msg(format!(
                "unsupported parquet value in task dataset: {other:?}"
            )))
        }
    })
}

fn row_to_json(row: &Row) -> Result<Value> {
    let mut object = serde_json::Map::new();
    for (name, field) in row.get_column_iter() {
        object.insert(name.clone(), field_to_json(field)?);
    }
    Ok(Value::Object(object))
}

/// All rows of a parquet file, each as a JSON object keyed by column name.
pub fn read_parquet_rows(path: &Path) -> Result<Vec<Value>> {
    let file =
        File::open(path).map_err(|e| msg(format!("Error opening {}: {e}", path.display())))?;
    let reader = SerializedFileReader::new(file).map_err(msg)?;
    let mut rows = Vec::with_capacity(reader.metadata().file_metadata().num_rows().max(0) as usize);
    for row in reader.get_row_iter(None).map_err(msg)? {
        rows.push(row_to_json(&row.map_err(msg)?)?);
    }
    Ok(rows)
}

/// Download the parquet shards of a dataset split into `shards_dir`, unless
/// already there. The manifest is written last, so its existence means the
/// download completed.
fn ensure_downloaded(
    repo_id: &str,
    subset: &str,
    split: &str,
    shards_dir: &Path,
) -> Result<Vec<PathBuf>> {
    let manifest_path = shards_dir.join("manifest.json");
    if !manifest_path.exists() {
        std::fs::create_dir_all(shards_dir).map_err(msg)?;
        let client = http_client()?;
        let listing_url =
            format!("https://huggingface.co/api/datasets/{repo_id}/parquet/{subset}/{split}");
        let listing = client
            .get(&listing_url)
            .send()
            .and_then(|r| r.error_for_status())
            .and_then(|r| r.text())
            .map_err(|e| msg(format!("Error listing {listing_url}: {e}")))?;
        let shard_urls: Vec<String> = serde_json::from_str(&listing).map_err(msg)?;
        let mut filenames = Vec::with_capacity(shard_urls.len());
        for (shard_index, shard_url) in shard_urls.iter().enumerate() {
            let filename = format!("{shard_index:05}.parquet");
            println!("Downloading {shard_url} ...");
            // write to a temporary file first, then move it into place
            let temp_path = shards_dir.join(format!("{filename}.tmp"));
            let mut response = client
                .get(shard_url)
                .send()
                .and_then(|r| r.error_for_status())
                .map_err(|e| msg(format!("Error downloading {shard_url}: {e}")))?;
            let mut file = File::create(&temp_path).map_err(msg)?;
            std::io::copy(&mut response, &mut file).map_err(msg)?;
            std::fs::rename(&temp_path, shards_dir.join(&filename)).map_err(msg)?;
            filenames.push(filename);
        }
        std::fs::write(
            &manifest_path,
            serde_json::to_string(&filenames).map_err(msg)?,
        )
        .map_err(msg)?;
    }
    let manifest = std::fs::read_to_string(&manifest_path).map_err(msg)?;
    let filenames: Vec<String> = serde_json::from_str(&manifest).map_err(msg)?;
    Ok(filenames.iter().map(|f| shards_dir.join(f)).collect())
}

/// The rows of a hub dataset split, in the stored order. Datasets without
/// named subsets use the subset "default" (or their single config name).
pub fn load_hub_dataset(repo_id: &str, subset: &str, split: &str) -> Result<Vec<Value>> {
    let slug = repo_id.replace('/', "--");
    let shards_dir = get_base_dir()
        .join("task_data")
        .join(slug)
        .join(subset)
        .join(split);
    let mut rows = Vec::new();
    for path in ensure_downloaded(repo_id, subset, split, &shards_dir)? {
        rows.extend(read_parquet_rows(&path)?);
    }
    Ok(rows)
}

/// Seeded shuffle that matches `datasets.Dataset.shuffle(seed=seed)` (and the
/// reference's HubDataset.shuffle) exactly: the row order comes out identical.
pub fn shuffle(rows: Vec<Value>, seed: u64) -> Vec<Value> {
    let mut rows: Vec<Option<Value>> = rows.into_iter().map(Some).collect();
    permutation(seed, rows.len())
        .into_iter()
        .map(|physical_index| {
            rows[physical_index]
                .take()
                .expect("permutation visits each row once")
        })
        .collect()
}

/// A hub dataset split in the canonical seed-42 shuffled order used by all tasks.
pub fn load_shuffled(repo_id: &str, subset: &str, split: &str) -> Result<Vec<Value>> {
    Ok(shuffle(load_hub_dataset(repo_id, subset, split)?, 42))
}
