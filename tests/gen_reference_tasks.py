"""
Generate reference data for the Rust tasks + chat_eval parity tests.

Synthetic dataset rows stand in for the HuggingFace datasets (the loaders are
exercised on identical JSONL snapshots); conversation rendering, evaluation,
slicing, mixtures, SpellingBee RNG, and the chat_eval loops are compared
against these goldens.

Run from the nanochat-python directory:
    uv run --with safetensors python ../tests/gen_reference_tasks.py
"""

import os
import sys
import json
import multiprocessing

# the sandboxed executor spawns children; fork avoids re-importing this module
multiprocessing.set_start_method("fork", force=True)

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.environ["NANOCHAT_BASE_DIR"] = OUT_DIR  # word list + hf_data live here
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanochat-python"))

import torch
from safetensors.torch import load_file

from tasks.common import Task, TaskMixture, render_mc
from tasks.arc import ARC
from tasks.mmlu import MMLU
from tasks.gsm8k import GSM8K, extract_answer
from tasks.humaneval import HumanEval, extract_imports, extract_program
from tasks.smoltalk import SmolTalk
import tasks.spellingbee as sb

meta = {}

# -----------------------------------------------------------------------------
# Synthetic dataset rows (stand-ins for the seed-42 shuffled HF datasets)

arc_rows = [
    {"question": f"What is {i} + {i}?",
     "choices": {"text": [str(i * 2), str(i * 2 + 1), str(i), "none"],
                 "label": ["A", "B", "C", "D"]},
     "answerKey": "A"}
    for i in range(6)
]
arc_rows[3]["choices"]["label"] = ["1", "2", "3", "4"]  # ARC sometimes uses numbers
arc_rows[3]["answerKey"] = "1"

mmlu_rows = [
    {"question": f"Question number {i}?", "choices": [f"opt{j}" for j in range(4)],
     "answer": i % 4, "subject": "anatomy"}
    for i in range(6)
]

gsm8k_rows = [
    {"question": "Weng earns $12 an hour. She did 50 minutes. How much?",
     "answer": "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10"},
    {"question": "Two plus two?", "answer": "<<2+2=4>>4 indeed <<4*1=4>><<1+1=2>>\n#### 4"},
    {"question": "No tools here?", "answer": "Just thinking.\n#### 1,234"},
    {"question": "No equals sign?", "answer": "Look: <<weird>> ok\n#### -5.5"},
]

humaneval_rows = [
    {"prompt": "from typing import List\n\n\ndef add(a: int, b: int) -> int:\n    \"\"\"Add two numbers.\"\"\"\n",
     "canonical_solution": "    return a + b\n",
     "entry_point": "add",
     "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n"},
]

smoltalk_rows = [
    {"messages": [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]},
    {"messages": [
        {"role": "user", "content": "What is up?"},
        {"role": "assistant", "content": "The sky."},
        {"role": "user", "content": "Thanks"},
        {"role": "assistant", "content": "Anytime."},
    ]},
]

# write the JSONL snapshots that the Rust file-based constructors read
hf_dir = os.path.join(OUT_DIR, "hf_data")
os.makedirs(hf_dir, exist_ok=True)
for name, rows in [("arc_ARC-Easy_test", arc_rows), ("mmlu_all_test", mmlu_rows),
                   ("gsm8k_main_test", gsm8k_rows), ("humaneval_test", humaneval_rows),
                   ("smoltalk_test", smoltalk_rows)]:
    with open(os.path.join(hf_dir, f"{name}.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

def make_task(cls, rows, **kwargs):
    """Instantiate a dataset-backed task on synthetic rows (bypass load_dataset)."""
    obj = cls.__new__(cls)
    Task.__init__(obj, **kwargs)
    obj.ds = rows
    obj.length = len(rows)  # used by SmolTalk
    return obj

# -----------------------------------------------------------------------------
# Conversations, evaluation outcomes, slicing

arc = make_task(ARC, arc_rows)
mmlu = make_task(MMLU, mmlu_rows)
gsm8k = make_task(GSM8K, gsm8k_rows)
humaneval = make_task(HumanEval, humaneval_rows)
smoltalk = make_task(SmolTalk, smoltalk_rows)

meta["render_mc"] = render_mc("Why?", ["A", "B"], ["because", "why not"])
meta["arc_conversations"] = [arc[i] for i in range(len(arc))]
meta["mmlu_conversations"] = [mmlu[i] for i in range(len(mmlu))]
meta["gsm8k_conversations"] = [gsm8k[i] for i in range(len(gsm8k))]
meta["humaneval_conversations"] = [humaneval[i] for i in range(len(humaneval))]
meta["smoltalk_conversations"] = [smoltalk[i] for i in range(len(smoltalk))]

meta["arc_eval"] = [
    {"idx": 0, "response": "A", "outcome": bool(arc.evaluate(arc[0], "A"))},
    {"idx": 0, "response": "B", "outcome": bool(arc.evaluate(arc[0], "B"))},
    {"idx": 3, "response": "1", "outcome": bool(arc.evaluate(arc[3], "1"))},
]
meta["gsm8k_eval"] = [
    {"idx": 0, "response": "Lots of math.\n#### 10", "outcome": gsm8k.evaluate(gsm8k[0], "Lots of math.\n#### 10")},
    {"idx": 0, "response": "#### 1,0", "outcome": gsm8k.evaluate(gsm8k[0], "#### 1,0")},
    {"idx": 0, "response": "no answer at all", "outcome": gsm8k.evaluate(gsm8k[0], "no answer at all")},
    {"idx": 2, "response": "#### 1234", "outcome": gsm8k.evaluate(gsm8k[2], "#### 1234")},
    {"idx": 3, "response": "#### -5.5", "outcome": gsm8k.evaluate(gsm8k[3], "#### -5.5")},
]
meta["gsm8k_rewards"] = [gsm8k.reward(gsm8k[0], "#### 10"), gsm8k.reward(gsm8k[0], "#### 9")]
meta["extract_answer_cases"] = [
    {"text": t, "answer": extract_answer(t)}
    for t in ["#### 42", "x #### 1,234.5 y", "#### -7", "nothing", "a #### 1 b #### 2"]
]

# humaneval: correct and broken completions through the real executor
he_conv = humaneval[0]
correct_completion = he_conv["messages"][1]["content"]
broken_completion = "```python\ndef add(a, b):\n    return a - b\n```"
meta["humaneval_eval"] = [
    {"response": correct_completion, "outcome": bool(humaneval.evaluate(he_conv, correct_completion))},
    {"response": broken_completion, "outcome": bool(humaneval.evaluate(he_conv, broken_completion))},
]
meta["extract_imports_cases"] = [
    {"text": he_conv["messages"][0]["content"], "imports": extract_imports(he_conv["messages"][0]["content"])},
    {"text": "import os\nfrom sys import path\n# comment\nimport re\nx = 1\nimport late", "imports": extract_imports("import os\nfrom sys import path\n# comment\nimport re\nx = 1\nimport late")},
]
meta["extract_program_cases"] = [
    {"text": t, "program": extract_program(t)}
    for t in [
        "```python\nx = 1\n```",
        "text before\n```\ny = 2\nz = 3\n```\ntext after",
        "plain code\nno blocks",
        "```python\nfirst\n```\n```python\nsecond\n```",
    ]
]

# slicing + mixture
sliced = make_task(MMLU, mmlu_rows, start=1, stop=6, step=2)
meta["sliced_len"] = len(sliced)
meta["sliced_conversations"] = [sliced[i] for i in range(len(sliced))]
mixture = TaskMixture([make_task(ARC, arc_rows), make_task(MMLU, mmlu_rows)])
meta["mixture_len"] = len(mixture)
meta["mixture_conversations"] = [mixture[i] for i in range(len(mixture))]

# -----------------------------------------------------------------------------
# SpellingBee / SimpleSpelling with a synthetic word list

words = ["strawberry", "banana", "letter", "mississippi", "queue", "rhythm",
         "bookkeeper", "committee", "balloon", "coffee", "pepper", "success",
         "assess", "footloose", "deed", "noon", "level", "radar", "kayak", "stats"]
with open(os.path.join(OUT_DIR, "words_alpha.txt"), "w") as f:
    f.write("\n".join(words) + "\n")

bee_train = sb.SpellingBee(size=1000, split="train")
bee_test = sb.SpellingBee(size=1000, split="test")
meta["spellingbee_train"] = [bee_train[i] for i in range(8)]
meta["spellingbee_test"] = [bee_test[i] for i in range(4)]
conv = bee_train[0]
meta["spellingbee_eval"] = [
    {"idx": 0, "response": "#### 3", "outcome": bee_train.evaluate(conv, "#### 3")},
    {"idx": 0, "response": "#### 2", "outcome": bee_train.evaluate(conv, "#### 2")},
]
simple = sb.SimpleSpelling(size=1000, split="train")
meta["simplespelling_train"] = [simple[i] for i in range(6)]

# -----------------------------------------------------------------------------
# chat_eval loops on the tiny core model + test tokenizer

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer
from nanochat.engine import Engine
from scripts.chat_eval import run_categorical_eval, run_generative_eval

train_text = [
    "Hello world! This is a tiny training corpus for the nanochat tokenizer. ",
    "The quick brown fox jumps over the lazy dog. 0123456789 42 100 2048. ",
    "I'd've thought it'll be fine, isn't it? We're sure they've gone. ",
    "def main():\n    print('hello')\n    return 0\n",
    "Numbers: 1, 22, 333, 4444. Punctuation?! (parens) [brackets] {braces}. ",
] * 20
tok = RustBPETokenizer.train_from_iterator(iter(train_text), vocab_size=300)

train_meta = json.load(open(os.path.join(OUT_DIR, "train_meta.json")))
core_config = GPTConfig(**train_meta["core_model_config"])
model = GPT(core_config)
model.init_weights()
model.load_state_dict(load_file(os.path.join(OUT_DIR, "core_model.safetensors")))
model.eval()

acc_cat = run_categorical_eval(arc, tok, model, batch_size=2)
meta["chat_eval_categorical_acc"] = acc_cat

engine = Engine(model, tok)
acc_gen = run_generative_eval(gsm8k, tok, model, engine, num_samples=1,
                              max_new_tokens=8, temperature=0.0, top_k=50)
meta["chat_eval_generative_acc"] = acc_gen

# -----------------------------------------------------------------------------
# base_eval.evaluate_core over a synthetic eval bundle

import random as pyrandom_mod
from scripts.base_eval import evaluate_core

bundle_dir = os.path.join(OUT_DIR, "eval_bundle")
os.makedirs(os.path.join(bundle_dir, "eval_data"), exist_ok=True)
core_rng = pyrandom_mod.Random(5)
core_mc_items = [
    {"query": " ".join(core_rng.choice(words) for _ in range(5)) + "?",
     "choices": [core_rng.choice(words) for _ in range(3)],
     "gold": core_rng.randrange(3)}
    for _ in range(10)
]
core_lm_items = [
    {"context": " ".join(core_rng.choice(words) for _ in range(5)) + " ",
     "continuation": " " + core_rng.choice(words)}
    for _ in range(10)
]
with open(os.path.join(bundle_dir, "eval_data", "tiny_mc.jsonl"), "w") as f:
    for item in core_mc_items:
        f.write(json.dumps(item) + "\n")
with open(os.path.join(bundle_dir, "eval_data", "tiny_lm.jsonl"), "w") as f:
    for item in core_lm_items:
        f.write(json.dumps(item) + "\n")
with open(os.path.join(bundle_dir, "core.yaml"), "w") as f:
    f.write("""icl_tasks:
- label: tiny_mc
  dataset_uri: tiny_mc.jsonl
  num_fewshot: [2]
  icl_task_type: multiple_choice
  continuation_delimiter: ': '
- label: tiny_lm
  dataset_uri: tiny_lm.jsonl
  num_fewshot: [0]
  icl_task_type: language_modeling
""")
with open(os.path.join(bundle_dir, "eval_meta_data.csv"), "w") as f:
    f.write("Eval Task,Random baseline\ntiny_mc,33.3\ntiny_lm,0.0\n")

meta["evaluate_core"] = evaluate_core(model, tok, torch.device("cpu"), max_per_task=8)

# -----------------------------------------------------------------------------
# SFT bestfit-pad data generator (the chat_sft packing logic, single rank)

def sft_data_generator_bos_bestfit(dataset, tokenizer, B, T, buffer_size=100):
    dataset_size = len(dataset)
    row_capacity = T + 1
    bos_token = tokenizer.get_bos_token_id()
    conv_buffer = []
    cursor = 0
    while True:
        rows, mask_rows, row_lengths = [], [], []
        for _ in range(B):
            row, mask_row = [], []
            padded = False
            while len(row) < row_capacity:
                while len(conv_buffer) < buffer_size:
                    conversation = dataset[cursor]
                    ids, mask = tokenizer.render_conversation(conversation)
                    conv_buffer.append((ids, mask))
                    cursor = (cursor + 1) % dataset_size
                remaining = row_capacity - len(row)
                best_idx, best_len = -1, 0
                for i, (conv, _) in enumerate(conv_buffer):
                    if len(conv) <= remaining and len(conv) > best_len:
                        best_idx, best_len = i, len(conv)
                if best_idx >= 0:
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                else:
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    mask_row.extend([0] * remaining)
                    padded = True
                    break
            row_lengths.append(content_len if padded else row_capacity)
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])
        batch_tensor = torch.tensor(rows, dtype=torch.long)
        inputs = batch_tensor[:, :-1].contiguous()
        targets = batch_tensor[:, 1:].clone()
        mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
        targets[mask_tensor[:, 1:] == 0] = -1
        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len - 1:] = -1
        yield inputs, targets

sft_mixture = TaskMixture([make_task(SmolTalk, smoltalk_rows), make_task(MMLU, mmlu_rows)])
sft_gen = sft_data_generator_bos_bestfit(sft_mixture, tok, B=2, T=64, buffer_size=6)
sft_batches = []
for _ in range(3):
    inputs, targets = next(sft_gen)
    sft_batches.append({"inputs": inputs.tolist(), "targets": targets.tolist()})
meta["sft_batches"] = sft_batches

# -----------------------------------------------------------------------------
with open(os.path.join(OUT_DIR, "tasks_meta.json"), "w") as f:
    json.dump(meta, f, indent=1)
print(f"categorical acc: {acc_cat}, generative acc: {acc_gen}")
print(f"spellingbee example 0 user msg: {meta['spellingbee_train'][0]['messages'][0]['content']!r}")
print("OK: wrote tasks fixtures")
