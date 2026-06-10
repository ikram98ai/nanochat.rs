"""
Generate reference data for the Rust port parity tests.

Builds a tiny seeded GPT with the Python implementation, runs every inference
code path (training forward, loss, sliding windows, KV-cache prefill/decode,
naive generate, Engine batch generate, tokenizer, calculator) and exports:

- tests/data/model.safetensors      tiny model weights (fp32)
- tests/data/fixtures.safetensors   input/output tensors for each case
- tests/data/meta.json              config, scalars, token sequences, calculator cases
- tests/data/tokenizer.json         tiny trained tokenizer (ranks + specials, b64)
- tests/data/tokenizer_fixtures.json  encode/decode/render_conversation cases

Run from the nanochat-python directory:
    uv run --with safetensors python ../tests/gen_reference.py
"""

import os
import sys
import json
import base64

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanochat-python"))

import torch
from safetensors.torch import save_file

from nanochat.gpt import GPT, GPTConfig, norm, apply_rotary_emb
from nanochat.flash_attention import flash_attn_func, flash_attn_with_kvcache
from nanochat.engine import KVCache, Engine, use_calculator
from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(42)

# -----------------------------------------------------------------------------
# Tiny config exercising every path: GQA (4q/2kv heads), vocab padding
# (199 -> 256), sliding windows (short=128 < seq), value embeddings on layers
# 0/2/4, backout at layer 2.
config = GPTConfig(
    sequence_len=512,
    vocab_size=199,
    n_layer=5,
    n_head=4,
    n_kv_head=2,
    n_embd=64,
    window_pattern="SSL",
)

model = GPT(config)
model.init_weights()

# init_weights zeros all projections (c_proj, smear_lambda, ...) which would
# make most code paths no-ops. Re-randomize every parameter with a fixed
# generator so each feature contributes to the output.
gen = torch.Generator().manual_seed(1234)
s = 3**0.5 * config.n_embd**-0.5
with torch.no_grad():
    for name, p in model.named_parameters():
        if "wte" in name:
            p.normal_(0.0, 0.8, generator=gen)
        elif "value_embeds" in name:
            p.normal_(0.0, 0.5, generator=gen)
        elif "lm_head" in name:
            p.normal_(0.0, 0.02, generator=gen)
        elif "resid_lambdas" in name:
            p.uniform_(0.9, 1.2, generator=gen)
        elif "x0_lambdas" in name:
            p.uniform_(0.0, 0.3, generator=gen)
        elif "smear_lambda" in name:
            p.fill_(0.7)
        elif "backout_lambda" in name:
            p.fill_(0.25)
        elif "smear_gate" in name or "ve_gate" in name:
            p.uniform_(0.0, 0.5, generator=gen)
        elif "c_proj" in name:
            p.uniform_(-0.5 * s, 0.5 * s, generator=gen)
        else:  # c_q, c_k, c_v, c_fc
            p.uniform_(-s, s, generator=gen)

model.eval()
state = {k: v.clone().float().contiguous() for k, v in model.state_dict().items()}
save_file(state, os.path.join(OUT_DIR, "model.safetensors"))

fixtures = {}
meta = {
    "config": config.__dict__,
    "padded_vocab_size": model.transformer.wte.weight.shape[0],
    "window_sizes": model.window_sizes,
    "num_flops_per_token": model.estimate_flops(),
    "num_scaling_params": model.num_scaling_params(),
    "rms_norm_eps": torch.finfo(torch.float32).eps,
}

# -----------------------------------------------------------------------------
# Micro fixtures: rmsnorm, rotary, attention with/without windows and cache
g2 = torch.Generator().manual_seed(99)

x = torch.randn(2, 3, 32, generator=g2)
fixtures["rmsnorm_in"] = x
fixtures["rmsnorm_out"] = norm(x)

fixtures["rotary_cos"] = model.cos[:, :8].clone()
fixtures["rotary_sin"] = model.sin[:, :8].clone()
xr = torch.randn(1, 8, 2, 16, generator=g2)
fixtures["rotary_in"] = xr
fixtures["rotary_out"] = apply_rotary_emb(xr, fixtures["rotary_cos"], fixtures["rotary_sin"])

q = torch.randn(1, 6, 4, 16, generator=g2)
k = torch.randn(1, 6, 2, 16, generator=g2)
v = torch.randn(1, 6, 2, 16, generator=g2)
fixtures["attn_q"], fixtures["attn_k"], fixtures["attn_v"] = q, k, v
fixtures["attn_full_out"] = flash_attn_func(q, k, v, causal=True, window_size=(512, 0))
fixtures["attn_win2_out"] = flash_attn_func(q, k, v, causal=True, window_size=(2, 0))

# KV-cache attention: prefill 12, chunk of 5, then single-token decode
kc = torch.zeros(1, 20, 2, 16)
vc = torch.zeros(1, 20, 2, 16)
qp = torch.randn(1, 12, 4, 16, generator=g2)
kp = torch.randn(1, 12, 2, 16, generator=g2)
vp = torch.randn(1, 12, 2, 16, generator=g2)
seqlens = torch.zeros(1, dtype=torch.int32)
fixtures["kvattn_q_prefill"], fixtures["kvattn_k_prefill"], fixtures["kvattn_v_prefill"] = qp, kp, vp
fixtures["kvattn_prefill_out"] = flash_attn_with_kvcache(
    qp, kc, vc, k=kp, v=vp, cache_seqlens=seqlens, causal=True, window_size=(4, 0))
seqlens += 12
qc = torch.randn(1, 5, 4, 16, generator=g2)
kn = torch.randn(1, 5, 2, 16, generator=g2)
vn = torch.randn(1, 5, 2, 16, generator=g2)
fixtures["kvattn_q_chunk"], fixtures["kvattn_k_chunk"], fixtures["kvattn_v_chunk"] = qc, kn, vn
fixtures["kvattn_chunk_out"] = flash_attn_with_kvcache(
    qc, kc, vc, k=kn, v=vn, cache_seqlens=seqlens, causal=True, window_size=(4, 0))
seqlens += 5
qd = torch.randn(1, 1, 4, 16, generator=g2)
kd = torch.randn(1, 1, 2, 16, generator=g2)
vd = torch.randn(1, 1, 2, 16, generator=g2)
fixtures["kvattn_q_decode"], fixtures["kvattn_k_decode"], fixtures["kvattn_v_decode"] = qd, kd, vd
fixtures["kvattn_decode_out"] = flash_attn_with_kvcache(
    qd, kc, vc, k=kd, v=vd, cache_seqlens=seqlens, causal=True, window_size=(4, 0))
fixtures["kvattn_k_cache_final"] = kc.clone()
fixtures["kvattn_v_cache_final"] = vc.clone()

# -----------------------------------------------------------------------------
# Full model forward, training mode (no KV cache)
g3 = torch.Generator().manual_seed(7)
ids1 = torch.randint(0, config.vocab_size, (2, 16), generator=g3)
fixtures["fw1_ids"] = ids1
with torch.no_grad():
    fixtures["fw1_logits"] = model(ids1)

ids2 = torch.randint(0, config.vocab_size, (1, 200), generator=g3)  # windows bind: 128 < 200
fixtures["fw2_ids"] = ids2
with torch.no_grad():
    fixtures["fw2_logits"] = model(ids2)

targets = torch.randint(0, config.vocab_size, (2, 16), generator=g3)
targets[torch.rand(2, 16, generator=g3) < 0.15] = -1
fixtures["loss_targets"] = targets
with torch.no_grad():
    fixtures["loss_mean"] = model(ids1, targets=targets).reshape(1)
    fixtures["loss_sum"] = model(ids1, targets=targets, loss_reduction='sum').reshape(1)

# -----------------------------------------------------------------------------
# KV-cache inference: prefill + argmax decode steps
def kv_decode_case(prefix, prompt_len, cache_len, n_steps):
    g = torch.Generator().manual_seed(11)
    prompt = torch.randint(0, config.vocab_size, (1, prompt_len), generator=g)
    fixtures[f"{prefix}_prompt"] = prompt
    kv = KVCache(batch_size=1, num_heads=config.n_kv_head, seq_len=cache_len,
                 head_dim=config.n_embd // config.n_head, num_layers=config.n_layer,
                 device="cpu", dtype=torch.float32)
    with torch.no_grad():
        logits = model(prompt, kv_cache=kv)
        fixtures[f"{prefix}_prefill_logits"] = logits[:, -1, :].clone()
        tokens = []
        for step in range(n_steps):
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(next_tok.item())
            logits = model(next_tok, kv_cache=kv)
        fixtures[f"{prefix}_final_logits"] = logits[:, -1, :].clone()
    meta[f"{prefix}_tokens"] = tokens

kv_decode_case("kv1", prompt_len=40, cache_len=64, n_steps=6)
kv_decode_case("kv2", prompt_len=150, cache_len=170, n_steps=6)  # window binds: 128 < 150

# -----------------------------------------------------------------------------
# Naive model.generate (recomputes full forward each step), temperature=0
g4 = torch.Generator().manual_seed(13)
gen_prompt = torch.randint(0, config.vocab_size, (10,), generator=g4).tolist()
meta["naive_gen_prompt"] = gen_prompt
meta["naive_gen_tokens"] = list(model.generate(gen_prompt, max_tokens=8, temperature=0.0))

# -----------------------------------------------------------------------------
# Engine.generate / generate_batch, temperature=0
class SpecialOnlyTokenizer:
    """Engine only needs special-token ids; map them above the model vocab."""
    def __init__(self):
        self._special = {name: 500 + i for i, name in enumerate(SPECIAL_TOKENS)}
    def encode_special(self, s):
        return self._special[s]
    def get_bos_token_id(self):
        return self._special["<|bos|>"]
    def encode(self, s, prepend=None):
        raise RuntimeError("not used in this fixture")
    def decode(self, tokens):
        raise RuntimeError("not used in this fixture")

engine = Engine(model, SpecialOnlyTokenizer())
g5 = torch.Generator().manual_seed(17)
eng_prompt = torch.randint(0, config.vocab_size, (12,), generator=g5).tolist()
meta["engine_prompt"] = eng_prompt
columns, masks = [], []
for token_column, token_masks in engine.generate(eng_prompt, num_samples=2, max_tokens=10, temperature=0.0):
    columns.append(token_column)
    masks.append(token_masks)
meta["engine_token_columns"] = columns
meta["engine_token_masks"] = masks
results, rmasks = engine.generate_batch(eng_prompt, num_samples=2, max_tokens=10, temperature=0.0)
meta["engine_batch_results"] = results
meta["engine_batch_masks"] = rmasks

# -----------------------------------------------------------------------------
# Calculator tool
calc_cases = [
    "1+2", "3*4+5", "12/4", "1/3", "(8-3)*2", "10//3", "3.5*2", "2.5+0.1",
    "1,234+1", "  7 * 7 ", "2**3", "1+; ",
    "'strawberry'.count('r')", '"hello world".count("o")', "'aaa'.count('aa')",
    "'abc'.count('d')", "import os", "'x'.upper()", "__class__",
    # floored division, signs
    "7//-2", "-7//-2", "2//-3", "-7//2", "-3.5//2",
    # literal edge cases
    "007+1", "00", "000", "00.5+1", "5.",
    # float str() formatting incl. scientific notation
    "1/100000", "1.5*10000000000000000", "1/3+0", "0.1+0.2", "10/2-5",
    # string atoms, concatenation, parens, chained methods
    "'.count('", "'a' 'b'.count('b')", "('ab').count('a')", "'ab'.count(('a'))",
    "'abc'.upper().count('A')", "'  x  '.strip().count('x')", "''.count('')",
    "'ab'.count('')",
]
meta["calculator_cases"] = [
    {"expr": e, "result": None if (r := use_calculator(e)) is None else str(r)}
    for e in calc_cases
]

# -----------------------------------------------------------------------------
# Tokenizer: train a tiny BPE, export ranks + encode/decode fixtures
train_text = [
    "Hello world! This is a tiny training corpus for the nanochat tokenizer. ",
    "The quick brown fox jumps over the lazy dog. 0123456789 42 100 2048. ",
    "I'd've thought it'll be fine, isn't it? We're sure they've gone. ",
    "def main():\n    print('hello')\n    return 0\n",
    "Numbers: 1, 22, 333, 4444. Punctuation?! (parens) [brackets] {braces}. ",
] * 20
tok = RustBPETokenizer.train_from_iterator(iter(train_text), vocab_size=300)

ranks = [
    {"bytes": base64.b64encode(b).decode(), "rank": r}
    for b, r in sorted(tok.enc._mergeable_ranks.items(), key=lambda kv: kv[1])
]
tok_json = {
    "pattern": tok.enc._pat_str,
    "mergeable_ranks": ranks,
    "special_tokens": tok.enc._special_tokens,
}
with open(os.path.join(OUT_DIR, "tokenizer.json"), "w") as f:
    json.dump(tok_json, f)

test_strings = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    "12345 + 67890 = ?",
    "I'd've THOUGHT so!",
    "héllo wörld 你好 \U0001f600",
    "  leading spaces\n\nnewlines\t\ttabs   ",
    "def f(x):\n    return x*2  # comment\n",
    "",
    "a",
    "\n",
]
conversation = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 12 + 34?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Let me compute."},
            {"type": "python", "text": "12+34"},
            {"type": "python_output", "text": "46"},
            {"type": "text", "text": "The answer is 46."},
        ]},
        {"role": "user", "content": "Thanks!"},
        {"role": "assistant", "content": "You're welcome."},
    ]
}
conv_ids, conv_mask = tok.render_conversation(conversation)
trunc_ids, trunc_mask = tok.render_conversation(conversation, max_tokens=20)
tok_fixtures = {
    "vocab_size": tok.get_vocab_size(),
    "bos_token_id": tok.get_bos_token_id(),
    "special_tokens": {name: tok.encode_special(name) for name in SPECIAL_TOKENS},
    "encode_cases": [
        {"text": t, "ids": tok.encode(t), "decoded": tok.decode(tok.encode(t))}
        for t in test_strings
    ],
    "encode_prepend_append": {
        "text": "Hello world",
        "ids": tok.encode("Hello world", prepend="<|bos|>", append="<|assistant_end|>"),
    },
    "encode_batch": tok.encode(["Hello world", "foo bar"]),
    "render_conversation": {"conversation": conversation, "ids": conv_ids, "mask": conv_mask},
    "render_conversation_truncated": {"max_tokens": 20, "ids": trunc_ids, "mask": trunc_mask},
    "render_for_completion": {
        "conversation": conversation,
        "ids": tok.render_for_completion(conversation),
    },
    "id_to_token_samples": {str(i): tok.id_to_token(i) for i in [0, 1, 50, 255, 260, 290]},
}
with open(os.path.join(OUT_DIR, "tokenizer_fixtures.json"), "w") as f:
    json.dump(tok_fixtures, f, indent=1)

# -----------------------------------------------------------------------------
fixtures = {k: (v.float() if v.is_floating_point() else v.to(torch.int64)).contiguous()
            for k, v in fixtures.items()}
save_file(fixtures, os.path.join(OUT_DIR, "fixtures.safetensors"))
with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=1)

print(f"window_sizes: {model.window_sizes}")
print(f"flops/token: {meta['num_flops_per_token']}")
print(f"scaling params: {meta['num_scaling_params']}")
print(f"naive_gen_tokens: {meta['naive_gen_tokens']}")
print(f"kv1_tokens: {meta['kv1_tokens']}  kv2_tokens: {meta['kv2_tokens']}")
print(f"engine rows: {meta['engine_batch_results']}")
print(f"tokenizer vocab: {tok_fixtures['vocab_size']}, conv ids len: {len(conv_ids)}")
print("OK: wrote", OUT_DIR)
