"""
Generate reference data for the Rust training-stack parity tests:
optimizer (AdamW + Muon), autograd gradients, full training steps,
dataloader packing over parquet shards, bits-per-byte eval, and CORE eval.

Outputs (in tests/data/):
- train_fixtures.safetensors  tensors for every case
- train_meta.json             scalars, losses, dataloader states, core_eval cases
- shards/shard_0000{0,1}.parquet  tiny parquet shards for the dataloader

Run from the nanochat-python directory:
    TORCHDYNAMO_DISABLE=1 uv run --with safetensors python ../tests/gen_reference_train.py
"""

import os
import sys
import json
import random

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # run fused kernels in eager mode
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanochat-python"))

import torch
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.torch import load_file, save_file

from nanochat.gpt import GPT, GPTConfig
from nanochat.optim import MuonAdamW
from nanochat.tokenizer import RustBPETokenizer
from nanochat.loss_eval import evaluate_bpb
from nanochat import core_eval

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
fixtures = {}
meta = {}

# -----------------------------------------------------------------------------
# Optimizer goldens: AdamW group + Muon groups (wide and tall), 5 steps

g = torch.Generator().manual_seed(42)
adamw_params = [
    torch.nn.Parameter(torch.randn(16, 8, generator=g) * 0.5),
    torch.nn.Parameter(torch.randn(5, generator=g) * 0.5),
]
muon_wide = [torch.nn.Parameter(torch.randn(8, 16, generator=g) * 0.1) for _ in range(3)]
muon_tall = [torch.nn.Parameter(torch.randn(16, 8, generator=g) * 0.1) for _ in range(2)]

param_groups = [
    dict(kind='adamw', params=adamw_params, lr=0.02, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.1),
    dict(kind='muon', params=muon_wide, lr=0.02, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=0.1),
    dict(kind='muon', params=muon_tall, lr=0.02, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=0.1),
]
all_params = adamw_params + muon_wide + muon_tall
for i, p in enumerate(all_params):
    fixtures[f"opt_p{i}_init"] = p.detach().clone()

opt = MuonAdamW(param_groups)
n_steps = 5
for step in range(n_steps):
    for i, p in enumerate(all_params):
        grad = torch.randn(p.shape, generator=g) * 0.1
        p.grad = grad
        fixtures[f"opt_p{i}_grad{step}"] = grad.clone()
    opt.step()
for i, p in enumerate(all_params):
    fixtures[f"opt_p{i}_final"] = p.detach().clone()
meta["opt_n_steps"] = n_steps

# -----------------------------------------------------------------------------
# Tiny GPT (same weights as the inference fixtures): gradients + training steps

config = GPTConfig(sequence_len=512, vocab_size=199, n_layer=5, n_head=4,
                   n_kv_head=2, n_embd=64, window_pattern="SSL")
model = GPT(config)
model.init_weights()
model.load_state_dict(load_file(os.path.join(OUT_DIR, "model.safetensors")))
model.train()

# 1) single backward pass: export every parameter gradient
g2 = torch.Generator().manual_seed(21)
xb = torch.randint(0, config.vocab_size, (2, 16), generator=g2)
yb = torch.randint(0, config.vocab_size, (2, 16), generator=g2)
yb[torch.rand(2, 16, generator=g2) < 0.1] = -1
fixtures["grad_x"] = xb
fixtures["grad_y"] = yb
loss = model(xb, targets=yb)
loss.backward()
fixtures["grad_loss"] = loss.detach().reshape(1)
for name, p in model.named_parameters():
    fixtures[f"grad::{name}"] = p.grad.detach().clone()
model.zero_grad(set_to_none=True)

# 2) three full training steps with the combined optimizer
optimizer = model.setup_optimizer(unembedding_lr=0.004, embedding_lr=0.2,
                                  matrix_lr=0.02, weight_decay=0.1, scalar_lr=0.5)
losses = []
batches = []
for step in range(3):
    x = torch.randint(0, config.vocab_size, (2, 16), generator=g2)
    y = torch.randint(0, config.vocab_size, (2, 16), generator=g2)
    batches.append((x, y))
    fixtures[f"train_x{step}"] = x
    fixtures[f"train_y{step}"] = y
    loss = model(x, targets=y)
    loss.backward()
    optimizer.step()
    model.zero_grad(set_to_none=True)
    losses.append(loss.item())
meta["train_losses"] = losses
for name, p in model.named_parameters():
    fixtures[f"trained::{name}"] = p.detach().clone()

# -----------------------------------------------------------------------------
# Tokenizer (re-train identically to gen_reference.py and sanity check)

train_text = [
    "Hello world! This is a tiny training corpus for the nanochat tokenizer. ",
    "The quick brown fox jumps over the lazy dog. 0123456789 42 100 2048. ",
    "I'd've thought it'll be fine, isn't it? We're sure they've gone. ",
    "def main():\n    print('hello')\n    return 0\n",
    "Numbers: 1, 22, 333, 4444. Punctuation?! (parens) [brackets] {braces}. ",
] * 20
tok = RustBPETokenizer.train_from_iterator(iter(train_text), vocab_size=300)
tok_fx = json.load(open(os.path.join(OUT_DIR, "tokenizer_fixtures.json")))
assert tok.encode("Hello world") == tok_fx["encode_cases"][0]["ids"], "tokenizer retrain mismatch"

# token_bytes as written by tokenizer training: utf-8 length of decode, 0 for specials
special_set = set(tok.get_special_tokens())
token_bytes_list = []
for token_id in range(tok.get_vocab_size()):
    token_str = tok.decode([token_id])
    token_bytes_list.append(0 if token_str in special_set else len(token_str.encode("utf-8")))
fixtures["token_bytes_tok"] = torch.tensor(token_bytes_list, dtype=torch.int32)

# -----------------------------------------------------------------------------
# Dataloader goldens: tiny parquet shards + best-fit packing

shard_dir = os.path.join(OUT_DIR, "shards")
os.makedirs(shard_dir, exist_ok=True)
rng = random.Random(7)
words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
         "hello", "world", "numbers", "42", "100", "fine", "sure", "gone"]
def make_doc():
    n = rng.randint(2, 30)
    return " ".join(rng.choice(words) for _ in range(n)) + "."
for shard_idx in range(2):
    docs = [make_doc() for _ in range(12)]
    table = pa.table({"text": docs})
    pq.write_table(table, os.path.join(shard_dir, f"shard_{shard_idx:05d}.parquet"),
                   row_group_size=4)

from nanochat import dataset as ds_mod
from nanochat import dataloader as dl_mod
ds_mod.DATA_DIR = shard_dir  # point the loader at the test shards
dl_mod.list_parquet_files = lambda **kw: ds_mod.list_parquet_files(shard_dir)

loader = dl_mod.tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tok, B=2, T=32, split="train", tokenizer_threads=1, tokenizer_batch_size=4,
    device="cpu", buffer_size=8)
states = []
for i in range(4):
    inputs, targets, state = next(loader)
    fixtures[f"dl_inputs{i}"] = inputs.clone()
    fixtures[f"dl_targets{i}"] = targets.clone()
    states.append(dict(state))
meta["dl_states"] = states

# resume from the state after batch 1
loader_resume = dl_mod.tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tok, B=2, T=32, split="train", tokenizer_threads=1, tokenizer_batch_size=4,
    device="cpu", buffer_size=8, resume_state_dict=states[1])
inputs, targets, state = next(loader_resume)
fixtures["dl_resume_inputs"] = inputs.clone()
fixtures["dl_resume_targets"] = targets.clone()
meta["dl_resume_state"] = dict(state)

# val split reads only the last shard
val_loader = dl_mod.tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tok, B=2, T=32, split="val", tokenizer_threads=1, tokenizer_batch_size=4,
    device="cpu", buffer_size=8)
inputs, targets, state = next(val_loader)
fixtures["dl_val_inputs"] = inputs.clone()
fixtures["dl_val_targets"] = targets.clone()
meta["dl_val_state"] = dict(state)

# -----------------------------------------------------------------------------
# evaluate_bpb golden (synthetic token_bytes over the model vocab)

# restore the pre-training weights so the bpb goldens use the same model
# the Rust test loads from model.safetensors
model.load_state_dict(load_file(os.path.join(OUT_DIR, "model.safetensors")))
model.eval()
token_bytes_model = torch.tensor(
    [0 if i < 3 else i % 7 for i in range(config.vocab_size)], dtype=torch.int32)
fixtures["bpb_token_bytes"] = token_bytes_model
g3 = torch.Generator().manual_seed(33)
bpb_batches = []
for i in range(2):
    x = torch.randint(0, config.vocab_size, (2, 16), generator=g3)
    y = torch.randint(0, config.vocab_size, (2, 16), generator=g3)
    if i == 1:
        y[torch.rand(2, 16, generator=g3) < 0.2] = -1  # exercise the masked path
    fixtures[f"bpb_x{i}"] = x
    fixtures[f"bpb_y{i}"] = y
    bpb_batches.append((x, y))
meta["bpb"] = evaluate_bpb(model, bpb_batches, steps=2, token_bytes=token_bytes_model)

# per-position loss fixture (loss_reduction='none')
with torch.no_grad():
    fixtures["loss_none"] = model(fixtures["bpb_x1"], targets=fixtures["bpb_y1"],
                                  loss_reduction='none').clone()

# -----------------------------------------------------------------------------
# core_eval goldens

rng = random.Random(99)
def make_text(n):
    return " ".join(rng.choice(words) for _ in range(n))

mc_data = [
    {"query": make_text(6) + "?", "choices": [make_text(2), make_text(3), make_text(2)],
     "gold": rng.randrange(3)}
    for _ in range(8)
]
schema_data = [
    {"context_options": [make_text(4), make_text(5), make_text(4)],
     "continuation": make_text(3), "gold": rng.randrange(3)}
    for _ in range(8)
]
lm_data = [
    {"context": make_text(6) + "  ", "continuation": " " + rng.choice(words)}
    for _ in range(8)
]
meta["core_mc_data"] = mc_data
meta["core_schema_data"] = schema_data
meta["core_lm_data"] = lm_data

# rendered prompts (0-shot and 2-shot)
meta["core_rendered"] = {
    "mc_0shot": core_eval.render_prompts_mc(mc_data[0], ": "),
    "mc_2shot": core_eval.render_prompts_mc(mc_data[0], ": ", [mc_data[1], mc_data[2]]),
    "schema_0shot": core_eval.render_prompts_schema(schema_data[0], " "),
    "schema_2shot": core_eval.render_prompts_schema(schema_data[0], " ", [schema_data[3], schema_data[4]]),
    "lm_0shot": core_eval.render_prompts_lm(lm_data[0], ""),
    "lm_2shot": core_eval.render_prompts_lm(lm_data[0], "", [lm_data[5], lm_data[6]]),
}

# CPython rng fewshot picks: random.Random(1234+idx).sample(available, k)
sample_cases = []
for n, k in [(8, 2), (8, 5), (50, 3), (200, 8)]:
    for idx in range(4):
        rng_s = random.Random(1234 + idx)
        available = [i for i in range(n) if i != idx]
        sample_cases.append({"n": n, "k": k, "idx": idx, "picked": rng_s.sample(available, k)})
meta["core_sample_cases"] = sample_cases

# full task accuracies need a model whose vocab covers the test tokenizer (300)
core_config = GPTConfig(sequence_len=512, vocab_size=tok.get_vocab_size(), n_layer=2,
                        n_head=2, n_kv_head=2, n_embd=32, window_pattern="L")
core_model = GPT(core_config)
core_model.init_weights()
gc = torch.Generator().manual_seed(555)
s = 3**0.5 * core_config.n_embd**-0.5
with torch.no_grad():
    for name, p in core_model.named_parameters():
        if "wte" in name or "value_embeds" in name:
            p.normal_(0.0, 0.5, generator=gc)
        elif "lm_head" in name:
            p.normal_(0.0, 0.05, generator=gc)
        elif "lambda" in name and "smear" not in name and "backout" not in name:
            p.uniform_(0.5, 1.0, generator=gc)
        elif "smear_lambda" in name:
            p.fill_(0.5)
        elif "backout_lambda" in name:
            p.fill_(0.25)
        else:
            p.uniform_(-s, s, generator=gc)
core_model.eval()
save_file({k: v.clone().float().contiguous() for k, v in core_model.state_dict().items()},
          os.path.join(OUT_DIR, "core_model.safetensors"))
meta["core_model_config"] = core_config.__dict__

task_accs = {}
for name, data, task_meta in [
    ("mc", mc_data, {"task_type": "multiple_choice", "num_fewshot": 2, "continuation_delimiter": ": "}),
    ("schema", schema_data, {"task_type": "schema", "num_fewshot": 0, "continuation_delimiter": " "}),
    ("lm", lm_data, {"task_type": "language_modeling", "num_fewshot": 1, "continuation_delimiter": ""}),
]:
    task_accs[name] = core_eval.evaluate_task(core_model, tok, data, torch.device("cpu"), task_meta)
meta["core_task_accs"] = task_accs

# -----------------------------------------------------------------------------
fixtures = {k: (v.float() if v.is_floating_point() else v.to(torch.int64)).contiguous()
            for k, v in fixtures.items()}
save_file(fixtures, os.path.join(OUT_DIR, "train_fixtures.safetensors"))
with open(os.path.join(OUT_DIR, "train_meta.json"), "w") as f:
    json.dump(meta, f, indent=1)

print(f"train losses: {meta['train_losses']}")
print(f"bpb: {meta['bpb']}")
print(f"task accs: {task_accs}")
print(f"dl states: {states}")
print("OK: wrote train fixtures")
