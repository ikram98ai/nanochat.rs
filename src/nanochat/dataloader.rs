//! Dataloader for pretraining (single process).
//!
//! BOS-aligned bestfit:
//!    - Every row starts with BOS token
//!    - Documents packed using best-fit algorithm to minimize cropping
//!    - When no document fits remaining space, crops a document to fill exactly
//!    - 100% utilization (no padding), ~35% tokens cropped at T=2048

use std::path::PathBuf;

use candle_core::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

use super::dataset::{list_parquet_files, num_row_groups, read_row_group_texts};
use super::tasks::common::Task;
use super::tokenizer::Tokenizer;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoaderState {
    pub pq_idx: usize,
    pub rg_idx: usize,
    pub epoch: usize,
}

/// Infinite iterator over document batches (lists of text strings) from parquet files.
/// Handles approximate resume; each batch comes with (pq_idx, rg_idx, epoch) where the
/// indices track position for resumption and epoch counts dataset cycles (starts at 1).
struct DocumentBatches {
    parquet_paths: Vec<PathBuf>,
    tokenizer_batch_size: usize,
    resume_pq_idx: usize,
    resume_rg_idx: Option<usize>,
    first_pass: bool,
    epoch: usize,
    pq_idx: usize,
    rg_idx: Option<usize>, // current row group, None before the first of a file
    // unconsumed texts of the current row group
    current_texts: Vec<String>,
    current_offset: usize,
}

impl DocumentBatches {
    fn new(
        split: &str,
        data_dir: Option<&std::path::Path>,
        resume_state: Option<&LoaderState>,
        tokenizer_batch_size: usize,
    ) -> Result<Self> {
        let warn_on_legacy = split == "train";
        let parquet_paths = list_parquet_files(data_dir, warn_on_legacy)?;
        assert!(
            !parquet_paths.is_empty(),
            "No dataset parquet files found, did you download the dataset?"
        );
        let parquet_paths: Vec<PathBuf> = if split == "train" {
            parquet_paths[..parquet_paths.len() - 1].to_vec()
        } else {
            parquet_paths[parquet_paths.len() - 1..].to_vec()
        };
        let resume_pq_idx = resume_state.map_or(0, |s| s.pq_idx);
        let resume_rg_idx = resume_state.map(|s| s.rg_idx);
        let epoch = resume_state.map_or(1, |s| s.epoch);
        Ok(Self {
            parquet_paths,
            tokenizer_batch_size,
            resume_pq_idx,
            resume_rg_idx,
            first_pass: true,
            epoch,
            pq_idx: resume_pq_idx,
            rg_idx: None,
            current_texts: Vec::new(),
            current_offset: 0,
        })
    }

    /// Next batch of document strings with the position it came from.
    fn next_batch(&mut self) -> Result<(Vec<String>, LoaderState)> {
        loop {
            // serve from the current row group if anything is left
            if self.current_offset < self.current_texts.len() {
                let end =
                    (self.current_offset + self.tokenizer_batch_size).min(self.current_texts.len());
                let batch = self.current_texts[self.current_offset..end].to_vec();
                self.current_offset = end;
                let state = LoaderState {
                    pq_idx: self.pq_idx,
                    rg_idx: self.rg_idx.unwrap(),
                    epoch: self.epoch,
                };
                return Ok((batch, state));
            }

            // advance to the next row group / file / epoch
            if self.pq_idx >= self.parquet_paths.len() {
                self.first_pass = false;
                self.epoch += 1;
                self.pq_idx = 0;
                self.rg_idx = None;
                continue;
            }
            let filepath = &self.parquet_paths[self.pq_idx];
            let n_row_groups = num_row_groups(filepath)?;
            let next_rg = match self.rg_idx {
                Some(rg) => rg + 1, // single process: world_size=1
                None => {
                    // Start from resume point if resuming on same file, otherwise from the start
                    if self.first_pass
                        && self.resume_rg_idx.is_some()
                        && self.pq_idx == self.resume_pq_idx
                    {
                        // advance by 1 so we don't repeat data after resuming
                        let rg = self.resume_rg_idx.take().unwrap() + 1;
                        if rg >= n_row_groups {
                            self.pq_idx += 1;
                            continue;
                        }
                        rg
                    } else {
                        0
                    }
                }
            };
            if next_rg >= n_row_groups {
                self.pq_idx += 1;
                self.rg_idx = None;
                continue;
            }
            self.rg_idx = Some(next_rg);
            self.current_texts = read_row_group_texts(filepath, next_rg)?;
            self.current_offset = 0;
        }
    }
}

/// BOS-aligned dataloader with Best-Fit Cropping.
///
/// Algorithm for each row:
/// 1. From buffered docs, pick the LARGEST doc that fits entirely
/// 2. Repeat until no doc fits
/// 3. When nothing fits, crop a doc to fill remaining space exactly
///
/// Key properties:
/// - Every row starts with BOS
/// - 100% utilization (no padding, every token is trained on)
/// - Approximately 35% of all tokens are discarded due to cropping
pub struct DataLoader<'a> {
    tokenizer: &'a Tokenizer,
    batch_size: usize,
    seq_len: usize,
    buffer_size: usize,
    device: Device,
    batches: DocumentBatches,
    bos_token: u32,
    doc_buffer: Vec<Vec<u32>>,
    state: LoaderState,
    /// number of threads used to tokenize each batch of documents
    pub tokenizer_threads: usize,
}

impl<'a> DataLoader<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tokenizer: &'a Tokenizer,
        batch_size: usize,
        seq_len: usize,
        split: &str,
        data_dir: Option<&std::path::Path>,
        tokenizer_batch_size: usize,
        device: &Device,
        resume_state: Option<&LoaderState>,
        buffer_size: usize,
    ) -> Result<Self> {
        assert!(
            split == "train" || split == "val",
            "split must be 'train' or 'val'"
        );
        let batches = DocumentBatches::new(split, data_dir, resume_state, tokenizer_batch_size)?;
        let bos_token = tokenizer.get_bos_token_id();
        Ok(Self {
            tokenizer,
            batch_size,
            seq_len,
            buffer_size,
            device: device.clone(),
            batches,
            bos_token,
            doc_buffer: Vec::new(),
            state: LoaderState {
                pq_idx: 0,
                rg_idx: 0,
                epoch: 1,
            },
            tokenizer_threads: 4,
        })
    }

    fn refill_buffer(&mut self) -> Result<()> {
        let (doc_batch, state) = self.batches.next_batch()?;
        self.state = state;
        // tokenize the batch of documents in parallel, prepending BOS to each
        let token_lists = self
            .tokenizer
            .encode_batch_threaded(&doc_batch, self.tokenizer_threads);
        for ids in token_lists {
            let mut tokens = Vec::with_capacity(ids.len() + 1);
            tokens.push(self.bos_token);
            tokens.extend(ids);
            self.doc_buffer.push(tokens);
        }
        Ok(())
    }

    /// Build the next (inputs, targets, state) batch: (B, T) i64 tensors with
    /// targets shifted one token ahead of inputs.
    pub fn next_batch(&mut self) -> Result<(Tensor, Tensor, LoaderState)> {
        let (b, t) = (self.batch_size, self.seq_len);
        let row_capacity = t + 1;
        let mut row_buffer = vec![0i64; b * row_capacity];

        for row_idx in 0..b {
            let row = &mut row_buffer[row_idx * row_capacity..(row_idx + 1) * row_capacity];
            let mut pos = 0;
            while pos < row_capacity {
                // Ensure buffer has documents
                while self.doc_buffer.len() < self.buffer_size {
                    self.refill_buffer()?;
                }

                let remaining = row_capacity - pos;

                // Find largest doc that fits entirely (ties keep the first)
                let mut best_idx: Option<usize> = None;
                let mut best_len = 0;
                for (i, doc) in self.doc_buffer.iter().enumerate() {
                    let doc_len = doc.len();
                    if doc_len <= remaining && doc_len > best_len {
                        best_idx = Some(i);
                        best_len = doc_len;
                    }
                }

                match best_idx {
                    Some(best_idx) => {
                        let doc = self.doc_buffer.remove(best_idx);
                        for (j, &tok) in doc.iter().enumerate() {
                            row[pos + j] = tok as i64;
                        }
                        pos += doc.len();
                    }
                    None => {
                        // No doc fits - crop shortest in buffer to fill remaining and minimize waste
                        let shortest_idx = (0..self.doc_buffer.len())
                            .min_by_key(|&i| self.doc_buffer[i].len())
                            .unwrap();
                        let doc = self.doc_buffer.remove(shortest_idx);
                        for (j, &tok) in doc[..remaining].iter().enumerate() {
                            row[pos + j] = tok as i64;
                        }
                        pos += remaining;
                    }
                }
            }
        }

        // next-token shift: inputs are tokens[0..T], targets are tokens[1..T+1] of each row
        let mut inputs = Vec::with_capacity(b * t);
        let mut targets = Vec::with_capacity(b * t);
        for row_idx in 0..b {
            let row = &row_buffer[row_idx * row_capacity..(row_idx + 1) * row_capacity];
            inputs.extend_from_slice(&row[..t]);
            targets.extend_from_slice(&row[1..]);
        }
        let inputs = Tensor::from_vec(inputs, (b, t), &self.device)?.to_dtype(DType::I64)?;
        let targets = Tensor::from_vec(targets, (b, t), &self.device)?.to_dtype(DType::I64)?;
        Ok((inputs, targets, self.state.clone()))
    }
}

/// BOS-aligned dataloader for SFT with bestfit-pad packing.
///
/// Each row in the batch starts with BOS (beginning of a conversation).
/// Conversations are packed using a best-fit algorithm. When no conversation
/// fits, the row is padded (instead of cropping) so no tokens are discarded.
/// Padding positions and non-assistant tokens have targets masked with -1
/// (ignore_index for cross-entropy).
pub struct SftDataLoader<'a> {
    dataset: &'a dyn Task,
    tokenizer: &'a Tokenizer,
    batch_size: usize,
    seq_len: usize,
    buffer_size: usize,
    device: Device,
    is_train: bool,
    /// number of optimization steps (-1 = full epoch)
    num_iterations: i64,
    conv_buffer: Vec<(Vec<u32>, Vec<u8>)>,
    cursor: usize,
    consumed: usize,
    epoch: usize,
    it: usize,
    /// toggled when the end of the training dataset is reached
    pub last_step: bool,
    /// goes from 0 to 1 over the course of the epoch
    pub approx_progress: f64,
    pub current_epoch: usize,
}

impl<'a> SftDataLoader<'a> {
    pub fn new(
        dataset: &'a dyn Task,
        tokenizer: &'a Tokenizer,
        batch_size: usize,
        seq_len: usize,
        is_train: bool,
        num_iterations: i64,
        device: &Device,
        buffer_size: usize,
    ) -> Self {
        assert!(dataset.len() > 0);
        Self {
            dataset,
            tokenizer,
            batch_size,
            seq_len,
            buffer_size,
            device: device.clone(),
            is_train,
            num_iterations,
            conv_buffer: Vec::new(),
            cursor: 0,
            consumed: 0,
            epoch: 1,
            it: 0,
            last_step: false,
            approx_progress: 0.0,
            current_epoch: 1,
        }
    }

    fn refill_buffer(&mut self) -> Result<()> {
        let dataset_size = self.dataset.len();
        while self.conv_buffer.len() < self.buffer_size {
            let conversation = self.dataset.get(self.cursor)?;
            // Truncate to at most one row. The reference always truncates to 2048
            // tokens, which fits its default rows of 2049; with a shorter context a
            // longer conversation could never be packed, would stay in the buffer
            // forever and, once the buffer was full of them, every batch would be
            // all padding.
            let max_tokens = 2048.min(self.seq_len + 1);
            let (ids, mask) = self
                .tokenizer
                .render_conversation(&conversation, max_tokens)?;
            self.conv_buffer.push((ids, mask));
            self.cursor += 1;
            if self.cursor >= dataset_size {
                self.cursor %= dataset_size;
                self.epoch += 1;
                // last_step is triggered based on consumption, not fetching
            }
        }
        Ok(())
    }

    /// Build the next (inputs, targets) batch of (B, T) i64 tensors.
    pub fn next_batch(&mut self) -> Result<(Tensor, Tensor)> {
        let row_capacity = self.seq_len + 1; // +1 for target at last position
        let bos_token = self.tokenizer.get_bos_token_id();
        let dataset_size = self.dataset.len();

        let mut rows: Vec<Vec<u32>> = Vec::with_capacity(self.batch_size);
        let mut mask_rows: Vec<Vec<u8>> = Vec::with_capacity(self.batch_size);
        let mut row_lengths = Vec::with_capacity(self.batch_size); // content length per row
        for _ in 0..self.batch_size {
            let mut row: Vec<u32> = Vec::with_capacity(row_capacity);
            let mut mask_row: Vec<u8> = Vec::with_capacity(row_capacity);
            let mut content_len = row_capacity;
            while row.len() < row_capacity {
                self.refill_buffer()?;
                let remaining = row_capacity - row.len();

                // Find largest conversation that fits entirely (ties keep the first)
                let mut best_idx: Option<usize> = None;
                let mut best_len = 0;
                for (i, (conv, _)) in self.conv_buffer.iter().enumerate() {
                    if conv.len() <= remaining && conv.len() > best_len {
                        best_idx = Some(i);
                        best_len = conv.len();
                    }
                }

                match best_idx {
                    Some(best_idx) => {
                        // Found a conversation that fits - use it entirely
                        let (conv, conv_mask) = self.conv_buffer.remove(best_idx);
                        row.extend(&conv);
                        mask_row.extend(&conv_mask);
                        self.consumed += 1;
                    }
                    None => {
                        // No conversation fits - pad the remainder instead of cropping
                        content_len = row.len();
                        row.extend(std::iter::repeat(bos_token).take(remaining));
                        mask_row.extend(std::iter::repeat(0).take(remaining));
                        break;
                    }
                }
            }
            row.truncate(row_capacity);
            mask_row.truncate(row_capacity);
            row_lengths.push(content_len);
            rows.push(row);
            mask_rows.push(mask_row);
        }

        // Stopping condition to respect num_iterations, if given
        self.it += 1;
        if self.is_train {
            if self.num_iterations > 0 && self.num_iterations <= self.it as i64 {
                self.last_step = true;
            }
            self.current_epoch = self.epoch;
            self.approx_progress = if self.num_iterations > 0 {
                self.it as f64 / self.num_iterations as f64
            } else {
                self.consumed as f64 / dataset_size as f64
            };
            // Trigger last_step when we've consumed enough
            if self.consumed >= dataset_size {
                self.last_step = true;
            }
        }

        // Build tensors: inputs = tokens[:-1], targets = tokens[1:] with the
        // loss mask applied (mask=1 only for assistant completions) and padding
        // positions set to -1 (ignore_index)
        let (b, t) = (self.batch_size, self.seq_len);
        let mut inputs = Vec::with_capacity(b * t);
        let mut targets = Vec::with_capacity(b * t);
        for (i, (row, mask_row)) in rows.iter().zip(mask_rows.iter()).enumerate() {
            inputs.extend(row[..t].iter().map(|&tok| tok as i64));
            for pos in 0..t {
                let target_pos = pos + 1;
                let masked = mask_row[target_pos] == 0
                    || (row_lengths[i] < row_capacity && target_pos >= row_lengths[i]);
                targets.push(if masked {
                    -1i64
                } else {
                    row[target_pos] as i64
                });
            }
        }
        let inputs = Tensor::from_vec(inputs, (b, t), &self.device)?;
        let targets = Tensor::from_vec(targets, (b, t), &self.device)?;
        Ok((inputs, targets))
    }
}
