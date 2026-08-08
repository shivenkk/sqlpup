.PHONY: setup fmt lint typecheck test check \
        download clean dedup decontaminate stats eval-index \
        tokenizer-corpus tokenizer-train tokenizer-train-matched fertility fertility-matched \
        train smoke-train

setup:
	uv sync --dev
	uv run pre-commit install

fmt:
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test

# --- pipeline stages ---------------------------------------------------------
# Thin wrappers over the sqlpup CLI: one target per stage, each reading a
# versioned config and/or explicit --in/--out paths. Override any variable on
# the command line, e.g.
#   make download SOURCE=synsql
#   make clean IN=data/raw/synsql.jsonl.zst OUT=data/clean/synsql.jsonl.zst
#   make train CONFIG=configs/train/proxy_50m.yaml
IN    ?= data/raw/corpus.jsonl.zst
OUT   ?= data/clean/corpus.jsonl.zst
INDEX ?= artifacts/decontam/eval_index.json

download: CONFIG ?= configs/data/mix_baseline.yaml
download:
	@test -n "$(SOURCE)" || { echo "make download requires SOURCE=<name> (a source in $(CONFIG))"; exit 1; }
	uv run sqlpup data download --config $(CONFIG) --source $(SOURCE)

# `clean` here is the data-cleaning stage (normalise + quality-filter), not a
# build-artifact clean.
clean:
	uv run sqlpup data clean --in $(IN) --out $(OUT)

dedup:
	uv run sqlpup data dedup --in $(IN) --out $(OUT)

decontaminate:
	uv run sqlpup data decontaminate --in $(IN) --out $(OUT) --index-in $(INDEX)

stats: CONFIG ?= configs/data/mix_baseline.yaml
stats:
	uv run sqlpup data stats --config $(CONFIG)

eval-index:
	uv run sqlpup data build-eval-index

# Builds the shipped tokenizer's corpus: downloads the SQL-heavy tokenizer mix
# (mix_tokenizer_v2, ~780M proxy-tokens across five sources) and cleans+merges
# the five per-source files into the one corpus tokenizer-train reads. Run once
# before `make tokenizer-train`.
tokenizer-corpus: CONFIG ?= configs/data/mix_tokenizer_v2.yaml
tokenizer-corpus: CORPUS ?= data/clean/tokenizer_corpus_v2.jsonl.zst
tokenizer-corpus:
	uv run sqlpup data download --config $(CONFIG)
	uv run sqlpup data clean \
	    --in data/tokenizer_corpus/starcoder_sql.jsonl.zst \
	         data/tokenizer_corpus/synsql.jsonl.zst \
	         data/tokenizer_corpus/schemapile.jsonl.zst \
	         data/tokenizer_corpus/stack_python.jsonl.zst \
	         data/tokenizer_corpus/fineweb_edu.jsonl.zst \
	    --out $(CORPUS)

# Trains on the SQL-heavy tokenizer corpus (mix_tokenizer_v2), holding out a
# 1/50 slice for the fertility comparison (HOLDOUT=0 or a custom N overrides on
# the command line).
tokenizer-train: CONFIG ?= configs/tokenizer/bpe32k.yaml
tokenizer-train: CORPUS ?= data/clean/tokenizer_corpus_v2.jsonl.zst
tokenizer-train: HOLDOUT ?= 50
tokenizer-train:
	uv run sqlpup tokenizer train --config $(CONFIG) --corpus $(CORPUS) --holdout-mod $(HOLDOUT)

# Compares the trained tokenizer's compression against gpt-4/gpt-4o/qwen2.5 on
# the corpus holdout and the BIRD/Spider eval sets, regenerating the committed
# report. Needs the optional 'report' extra (tiktoken) and downloads the three
# reference tokenizers on first run.
fertility: TOKENIZER ?= artifacts/tokenizer/tokenizer.json
fertility: CORPUS ?= data/clean/tokenizer_corpus_v2.jsonl.zst
fertility: HOLDOUT ?= 50
fertility:
	uv run --extra report sqlpup tokenizer fertility \
	    --tokenizer $(TOKENIZER) --corpus $(CORPUS) --holdout-mod $(HOLDOUT) \
	    --report docs/tokenizer-fertility.md

# Trains the throwaway matched-vocabulary tokenizer at cl100k_base's exact vocab
# (100,277) on the same corpus and holdout. Saved to a separate artifacts path
# -- this is a science experiment, never the shipped 32k model tokenizer.
tokenizer-train-matched: CONFIG ?= configs/tokenizer/bpe32k.yaml
tokenizer-train-matched: CORPUS ?= data/clean/tokenizer_corpus_v2.jsonl.zst
tokenizer-train-matched: VOCAB ?= 100277
tokenizer-train-matched: OUTDIR ?= artifacts/tokenizer_matched_100k
tokenizer-train-matched: HOLDOUT ?= 50
tokenizer-train-matched:
	uv run sqlpup tokenizer train --config $(CONFIG) --vocab-size $(VOCAB) \
	    --out-dir $(OUTDIR) --corpus $(CORPUS) --holdout-mod $(HOLDOUT)

# The matched-vocabulary experiment: adds the ~100k-vocab sqlpup tokenizer as an
# extra column and regenerates the committed report (the base 32k section plus
# the added matched-vocab section). Run `make tokenizer-train-matched` first.
fertility-matched: TOKENIZER ?= artifacts/tokenizer/tokenizer.json
fertility-matched: MATCHED ?= artifacts/tokenizer_matched_100k/tokenizer.json
fertility-matched: MATCHED_LABEL ?= sqlpup-100k
fertility-matched: CORPUS ?= data/clean/tokenizer_corpus_v2.jsonl.zst
fertility-matched: HOLDOUT ?= 50
fertility-matched:
	uv run --extra report sqlpup tokenizer fertility \
	    --tokenizer $(TOKENIZER) --extra-tokenizer $(MATCHED):$(MATCHED_LABEL) \
	    --corpus $(CORPUS) --holdout-mod $(HOLDOUT) \
	    --out artifacts/tokenizer_matched_100k/fertility.json \
	    --report docs/tokenizer-fertility.md

# Proxy config by default; real runs override CONFIG (e.g. a 260M config) and
# pass --resume auto / --device on the CLI directly.
train: CONFIG ?= configs/train/proxy_30m.yaml
train:
	uv run sqlpup train --config $(CONFIG)

smoke-train:
	uv run python scripts/smoke_train.py
