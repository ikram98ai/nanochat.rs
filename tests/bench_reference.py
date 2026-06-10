"""
Benchmark the Python engine on CPU for comparison with the Rust port.

Run from the nanochat-python directory:
    uv run --with safetensors python ../tests/bench_reference.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanochat-python"))

import torch
from safetensors.torch import load_file

from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import KVCache

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def bench(name, model, prompt_len, decode_steps):
    config = model.config
    g = torch.Generator().manual_seed(11)
    prompt = torch.randint(0, config.vocab_size, (1, prompt_len), generator=g)
    kv = KVCache(batch_size=1, num_heads=config.n_kv_head,
                 seq_len=prompt_len + decode_steps,
                 head_dim=config.n_embd // config.n_head,
                 num_layers=config.n_layer, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        t0 = time.perf_counter()
        logits = model(prompt, kv_cache=kv)
        t1 = time.perf_counter()
        for _ in range(decode_steps):
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            logits = model(next_tok, kv_cache=kv)
        t2 = time.perf_counter()
    prefill_tps = prompt_len / (t1 - t0)
    decode_tps = decode_steps / (t2 - t1)
    print(f"{name}: prefill {prompt_len} tok in {t1-t0:.3f}s ({prefill_tps:.0f} tok/s), "
          f"decode {decode_steps} tok in {t2-t1:.3f}s ({decode_tps:.1f} tok/s)")


# tiny model: identical weights to the Rust parity fixtures
config = GPTConfig(sequence_len=512, vocab_size=199, n_layer=5, n_head=4,
                   n_kv_head=2, n_embd=64, window_pattern="SSL")
model = GPT(config)
model.init_weights()
model.load_state_dict(load_file(os.path.join(DATA_DIR, "model.safetensors")))
model.eval()
bench("tiny  (d5/64emb)", model, prompt_len=150, decode_steps=100)

# medium model: realistic-ish depth/width, random weights (speed only)
torch.manual_seed(42)
config = GPTConfig(sequence_len=1024, vocab_size=32768, n_layer=8, n_head=8,
                   n_kv_head=4, n_embd=512, window_pattern="SSSL")
model = GPT(config)
model.init_weights()
model.eval()
bench("medium(d8/512emb)", model, prompt_len=200, decode_steps=100)
