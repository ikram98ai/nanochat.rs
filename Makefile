# nanochat.rs: common commands. Run `make` to list them.
#
# Options (set on the command line, e.g. `make chat FEATURES=metal`):
#   FEATURES  GPU backend to build with: metal (Apple Silicon) or cuda (NVIDIA, untested). Empty = CPU.
#   BASE      directory holding checkpoints, tokenizer and data. Empty = ./models
#   SOURCE    which checkpoints to load: base, sft or rl (default sft)
#   TAG       model tag, e.g. d12. Empty = the largest model found
#   PROMPT    ask one question and exit. Empty = interactive chat
#   TASK      chat-eval task: ARC-Easy, ARC-Challenge, MMLU, GSM8K, HumanEval. Empty = all
#   N         number of training shards to download (default 8)
#   ARGS      extra arguments passed straight to the program

FEATURES ?=
BASE ?=
SOURCE ?= sft
TAG ?=
PROMPT ?=
TASK ?=
N ?= 8
ARGS ?=

RUN = $(if $(BASE),NANOCHAT_BASE_DIR=$(BASE) )cargo run --release $(if $(FEATURES),--features $(FEATURES) )
TAG_FLAG = $(if $(TAG),--model-tag $(TAG) )

HF = https://huggingface.co
MODELS = $(if $(BASE),$(BASE),models)
SHA256 = $(shell command -v sha256sum >/dev/null 2>&1 && echo sha256sum || echo shasum -a 256)
# $(call fetch,URL,FILE,SHA256): download FILE unless it is already there. Keeps
# resuming after dropped connections, and checks the checksum before the file is put in place.
fetch = @test -f $(2) || { echo "Downloading $(2)"; n=0; until curl -L --fail -C - --connect-timeout 30 --speed-limit 10240 --speed-time 60 -o $(2).part $(1); do n=$$((n+1)); [ $$n -lt 200 ] || exit 1; sleep 5; done; echo "$(3)  $(2).part" | $(SHA256) -c - >/dev/null || { echo "Checksum mismatch: delete $(2).part and try again"; exit 1; }; mv $(2).part $(2); }

.DEFAULT_GOAL := help
.PHONY: help build chat sample base-eval chat-eval get-d12 get-d8 data tok-train base-train chat-sft test-data test bench clean

help: ## List the commands
	@awk 'BEGIN {FS = ":.*## "} /^[a-z0-9-]+:.*## / {printf "  make %-11s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Compile everything in release mode
	cargo build --release $(if $(FEATURES),--features $(FEATURES))

# --- run a model -------------------------------------------------------------

chat: ## Chat with a model (PROMPT="..." asks one question)
	$(RUN)--bin nanochat -- -i $(SOURCE) $(TAG_FLAG)$(if $(PROMPT),-p "$(PROMPT)" )$(ARGS)

sample: ## Print text samples from a base model
	$(RUN)--bin base_eval -- --eval sample $(TAG_FLAG)$(ARGS)

base-eval: ## Evaluate a base model: CORE metric, bits per byte, samples
	$(RUN)--bin base_eval -- $(TAG_FLAG)$(ARGS)

chat-eval: ## Evaluate a chat model on the chat tasks (TASK=ARC-Easy for one)
	$(RUN)--bin chat_eval -- -i $(SOURCE) $(TAG_FLAG)$(if $(TASK),-a "$(TASK)" )$(ARGS)

# --- get weights from Hugging Face (community checkpoints, English) ----------

# both models were trained with the same tokenizer
TOKENIZER_SHA = 387cfc082b0bee45467774fd6f1310a922ad170886a58ccddcb468f275e06a6c

D12 = $(HF)/allbilly/nanochat-d12-sft/resolve/main
get-d12: ## Download an English chat model (793 MB) into models
	@mkdir -p $(MODELS)/chatsft_checkpoints/d12 $(MODELS)/tokenizer
	$(call fetch,$(D12)/tokenizer/tokenizer.pkl,$(MODELS)/tokenizer/tokenizer.pkl,$(TOKENIZER_SHA))
	$(call fetch,$(D12)/meta_000970.json,$(MODELS)/chatsft_checkpoints/d12/meta_000970.json,08306523dd602d9d857f8f6ea163374e0a79ba5ac4ba73ae1b7a0daa185db800)
	$(call fetch,$(D12)/model_000970.pt,$(MODELS)/chatsft_checkpoints/d12/model_000970.pt,a02fb1b64ab26b0f2a397f0efa1d6cd01fdba1b60d29e861f5daefd1e12b69eb)

D8 = $(HF)/EvgeniyZh/nanochat-d8/resolve/main
get-d8: ## Download a small English base model (335 MB) into models
	@mkdir -p $(MODELS)/base_checkpoints/d8 $(MODELS)/tokenizer
	$(call fetch,$(D8)/tokenizer/tokenizer.pkl,$(MODELS)/tokenizer/tokenizer.pkl,$(TOKENIZER_SHA))
	$(call fetch,$(D8)/meta_001680.json,$(MODELS)/base_checkpoints/d8/meta_001680.json,04315ce5be4e127ab8fe0810394a50f7fb93310f435c89da2d5450483ab25f85)
	$(call fetch,$(D8)/model.safetensors,$(MODELS)/base_checkpoints/d8/model_001680.safetensors,ae10f2435ce789c2e805f73f4d84871a0ed3f9f2e180e8439282433182426395)

# --- train your own ------------------------------------------------------------

data: ## Download N training shards plus the validation shard (about 90 MB each)
	$(RUN)--bin dataset -- -n $(N) $(ARGS)

tok-train: ## Train the tokenizer on the downloaded shards
	$(RUN)--bin tok_train -- $(ARGS)

base-train: ## Pretrain a base model (pass sizes in ARGS)
	$(RUN)--bin base_train -- $(ARGS)

chat-sft: ## Fine-tune the base model into a chat model
	$(RUN)--bin chat_sft -- $(TAG_FLAG)$(ARGS)

# --- tests ---------------------------------------------------------------------

tests/data/meta.json:
	cargo run --release --example gen_test_data

test-data: ## Rebuild tests/data from the current code
	cargo run --release --example gen_test_data

test: tests/data/meta.json ## Run the tests (builds tests/data first if missing)
	cargo test --release $(if $(FEATURES),--features $(FEATURES))

bench: tests/data/meta.json ## Time prefill and decode on CPU
	cargo run --release --example bench

clean: ## Remove build output
	cargo clean
