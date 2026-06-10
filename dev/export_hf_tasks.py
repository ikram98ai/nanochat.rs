"""
Materialize the HuggingFace-hosted task datasets as JSONL snapshots that the
Rust tasks read, preserving the seed-42 shuffled order used everywhere.

Run from the nanochat-python directory (downloads the datasets on first use):
    uv run python ../dev/export_hf_tasks.py [--out DIR]

Writes {base_dir}/hf_data/*.jsonl by default.
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanochat-python"))

from datasets import load_dataset
from nanochat.common import get_base_dir

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, default=None, help="output dir (default {base_dir}/hf_data)")
args = parser.parse_args()

out_dir = args.out or os.path.join(get_base_dir(), "hf_data")
os.makedirs(out_dir, exist_ok=True)

EXPORTS = [
    # (filename, hf path, subset, split)
    ("arc_ARC-Easy_test", "allenai/ai2_arc", "ARC-Easy", "test"),
    ("arc_ARC-Easy_train", "allenai/ai2_arc", "ARC-Easy", "train"),
    ("arc_ARC-Challenge_test", "allenai/ai2_arc", "ARC-Challenge", "test"),
    ("arc_ARC-Challenge_train", "allenai/ai2_arc", "ARC-Challenge", "train"),
    ("mmlu_all_test", "cais/mmlu", "all", "test"),
    ("mmlu_all_auxiliary_train", "cais/mmlu", "all", "auxiliary_train"),
    ("gsm8k_main_test", "openai/gsm8k", "main", "test"),
    ("gsm8k_main_train", "openai/gsm8k", "main", "train"),
    ("humaneval_test", "openai/openai_humaneval", None, "test"),
    ("smoltalk_train", "HuggingFaceTB/smol-smoltalk", None, "train"),
    ("smoltalk_test", "HuggingFaceTB/smol-smoltalk", None, "test"),
]

for filename, path, subset, split in EXPORTS:
    out_path = os.path.join(out_dir, f"{filename}.jsonl")
    if os.path.exists(out_path):
        print(f"Skipping {out_path} (already exists)")
        continue
    print(f"Exporting {path} {subset or ''} {split} -> {out_path}")
    ds = load_dataset(path, subset, split=split) if subset else load_dataset(path, split=split)
    ds = ds.shuffle(seed=42)  # the canonical order used by the tasks
    with open(out_path + ".tmp", "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.rename(out_path + ".tmp", out_path)

print("Done.")
