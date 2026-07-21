# sqlpup tokenizer fertility

_Generated 2026-07-21T17:16:08.779003+00:00. Regenerate with `make fertility-matched`._

sqlpup ships a byte-level BPE tokenizer with a code-tuned pre-tokenizer, trained on the SQL-heavy corpus `configs/data/mix_tokenizer_v2.yaml`. It is compared against three general-purpose tokenizers with 3-6x larger vocabularies (GPT-4, GPT-4o, Qwen2.5) on held-out corpus text and on the BIRD/Spider eval sets. Lower **tokens/word** and higher **bytes/token** mean tighter compression: more text per token, so more context per training and inference step. The larger vocabularies hold more merges, so every ratio should be read against the vocab sizes below.

**Before -> after.** The initial tokenizer (a small ~95M-token, ~35%-SQL balanced sample; plain byte-level pre-tokenizer) trailed GPT-4's cl100k on SQL even at matched vocabulary (BIRD dev SQL +8.0%, Spider dev SQL +9.0% more tokens). Retraining on the bigger, SQL-heavier corpus alone barely moved the out-of-distribution SQL gap (~+7.8% / +8.2%) -- it was not corpus-size-bound. Adding the code-tuned pre-tokenizer (cl100k-style <=3-digit number groups + code-aware whitespace) closed and reversed it: at matched vocabulary sqlpup now uses fewer tokens than cl100k on every SQL surface below, at a modest ~0.4-0.6% cost on English questions.

## Tokenizers

| tokenizer | vocab size |
| --- | ---: |
| sqlpup-bpe32k | 32,768 |
| gpt-4 (cl100k_base) | 100,277 |
| gpt-4o (o200k_base) | 200,019 |
| qwen2.5-0.5b | 151,665 |

## Method

- **Corpus provenance.** The SQL-heavy tokenizer mix (`configs/data/mix_tokenizer_v2.yaml`, ~780M proxy-tokens across five sources) cleaned and merged into one corpus (`data/clean/tokenizer_corpus_v2.jsonl.zst`). Fertility is measured only on the **holdout slice** -- documents where `xxh64(id) % 50 == 0`, the text the BPE trainer never saw (trained with `--holdout-mod 50`).
- **Eval sets.** BIRD dev and Spider dev questions and gold SQL, loaded from the cached `data/eval` archives. No tokenizer here trained on them, so they are the most meaningful out-of-distribution comparison surface.
- **Metrics.** Per group: **bytes** = UTF-8 byte length summed; **words** = whitespace split (`len(text.split())`) summed; **tokens** = ids per tokenizer summed, no special tokens added. **tokens/word** = tokens / words; **bytes/token** = bytes / tokens. Ratios come from summed totals (not per-document averages) and are deterministic.

## Tokens per word (lower is better)

| group | docs | words | sqlpup-bpe32k | gpt-4 (cl100k_base) | gpt-4o (o200k_base) | qwen2.5-0.5b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fineweb_edu | 2,432 | 1,955,726 | 1.452 | 1.317 | 1.301 | 1.345 |
| schemapile | 414 | 130,022 | 2.007 | 2.115 | 2.079 | 2.124 |
| stack_python | 1,409 | 665,659 | 3.105 | 2.860 | 2.872 | 2.923 |
| starcoder_sql | 1,414 | 1,777,393 | 3.831 | 3.754 | 3.692 | 4.366 |
| synsql | 1,661 | 2,173,209 | 1.629 | 1.643 | 1.654 | 1.649 |
| **overall** | 7,330 | 6,702,009 | 2.315 | 2.238 | 2.220 | 2.417 |
| bird_dev_question | 1,534 | 22,325 | 1.357 | 1.272 | 1.262 | 1.340 |
| bird_dev_sql | 1,534 | 38,935 | 2.055 | 1.970 | 1.970 | 2.028 |
| spider_dev_question | 1,034 | 12,848 | 1.216 | 1.151 | 1.149 | 1.171 |
| spider_dev_sql | 1,034 | 16,146 | 2.049 | 1.942 | 1.933 | 1.958 |

## Bytes per token (higher is better)

| group | docs | bytes | sqlpup-bpe32k | gpt-4 (cl100k_base) | gpt-4o (o200k_base) | qwen2.5-0.5b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fineweb_edu | 2,432 | 12,138,948 | 4.274 | 4.713 | 4.771 | 4.615 |
| schemapile | 414 | 1,041,286 | 3.989 | 3.786 | 3.852 | 3.771 |
| stack_python | 1,409 | 7,811,025 | 3.779 | 4.102 | 4.086 | 4.015 |
| starcoder_sql | 1,414 | 19,000,180 | 2.790 | 2.848 | 2.896 | 2.448 |
| synsql | 1,661 | 15,572,216 | 4.400 | 4.361 | 4.333 | 4.347 |
| **overall** | 7,330 | 55,563,655 | 3.581 | 3.705 | 3.734 | 3.431 |
| bird_dev_question | 1,534 | 126,958 | 4.190 | 4.472 | 4.508 | 4.245 |
| bird_dev_sql | 1,534 | 247,705 | 3.095 | 3.230 | 3.229 | 3.138 |
| spider_dev_question | 1,034 | 70,370 | 4.505 | 4.760 | 4.766 | 4.676 |
| spider_dev_sql | 1,034 | 110,321 | 3.334 | 3.519 | 3.534 | 3.490 |

## Caveats

- **Vocab-size asymmetry (read the ratios with this in mind).** sqlpup's vocab is 32,768 against roughly 100k (GPT-4), 200k (GPT-4o) and 152k (Qwen2.5) -- 3-6x larger. Those larger vocabularies hold more merges and generally compress SQL more tightly here in absolute bytes/token; sqlpup's aim is competitive compression at a fraction of the embedding parameters, purpose-built for the SQL-heavy mix, not to beat far larger vocabularies on raw ratio. The asymmetry is stated, not hidden, because the ratios cannot be read fairly without it.
- **Fertility is a compression metric, not task quality.** Fewer tokens per word is a throughput and context-length win; it is not by itself a measure of downstream text-to-SQL accuracy.
- **The corpus holdout is same-distribution as the training mix.** It is text the trainer never saw, but drawn from the same five sources. The eval-set groups (BIRD/Spider) are the genuinely out-of-distribution surface and the more meaningful comparison.
- **The corpus is post-clean-filter.** The `min_chars=200` clean filter drops short documents (here ~25k of ~400k, mostly short StarCoder SQL files and SchemaPile schemas), so very short SQL snippets are underrepresented in the corpus-slice groups -- another reason the eval-set groups matter.

## Matched-vocabulary comparison

The 32k comparison above is confounded by vocab size: GPT-4/GPT-4o/Qwen2.5 carry 3-6x more merges than sqlpup's shipped 32,768-token vocabulary. To isolate *domain specialization* from *vocab size*, a throwaway sqlpup byte-level BPE was trained at GPT-4 cl100k_base's exact vocabulary (100,277) on the same corpus (`data/clean/tokenizer_corpus_v2.jsonl.zst`) and the same 1/50 holdout, then compared head-to-head. **sqlpup-100k** is an experiment only -- the model still ships the 32k tokenizer.

At matched vocabulary the domain-specialization effect is real: **sqlpup-100k** compresses SQL more tightly than GPT-4's cl100k on every SQL surface measured below.

- **BIRD dev SQL:** sqlpup-100k uses 1.1% fewer tokens than GPT-4 cl100k (75,823 vs 76,695).
- **Spider dev SQL:** sqlpup-100k uses 1.5% fewer tokens than GPT-4 cl100k (30,885 vs 31,348).
- **StarCoder SQL (corpus holdout):** sqlpup-100k uses 4.3% fewer tokens than GPT-4 cl100k (6,388,767 vs 6,672,354).

The same head-to-head on the English **question** surfaces, where a SQL-tuned vocabulary is expected to give ground:

- **BIRD dev question:** sqlpup-100k uses 0.6% more tokens than GPT-4 cl100k (28,559 vs 28,390).
- **Spider dev question:** sqlpup-100k uses 0.4% more tokens than GPT-4 cl100k (14,837 vs 14,784).

### Tokens per word (lower is better)

| group | docs | words | sqlpup-bpe32k (32,768) | sqlpup-100k (100,277) | gpt-4 (cl100k_base) (100,277) | gpt-4o (o200k_base) (200,019) | qwen2.5-0.5b (151,665) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| starcoder_sql | 1,414 | 1,777,393 | 3.831 | 3.594 | 3.754 | 3.692 | 4.366 |
| **overall** | 7,330 | 6,702,009 | 2.315 | 2.173 | 2.238 | 2.220 | 2.417 |
| bird_dev_question | 1,534 | 22,325 | 1.357 | 1.279 | 1.272 | 1.262 | 1.340 |
| bird_dev_sql | 1,534 | 38,935 | 2.055 | 1.947 | 1.970 | 1.970 | 2.028 |
| spider_dev_question | 1,034 | 12,848 | 1.216 | 1.155 | 1.151 | 1.149 | 1.171 |
| spider_dev_sql | 1,034 | 16,146 | 2.049 | 1.913 | 1.942 | 1.933 | 1.958 |

### Bytes per token (higher is better)

| group | docs | bytes | sqlpup-bpe32k (32,768) | sqlpup-100k (100,277) | gpt-4 (cl100k_base) (100,277) | gpt-4o (o200k_base) (200,019) | qwen2.5-0.5b (151,665) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| starcoder_sql | 1,414 | 19,000,180 | 2.790 | 2.974 | 2.848 | 2.896 | 2.448 |
| **overall** | 7,330 | 55,563,655 | 3.581 | 3.816 | 3.705 | 3.734 | 3.431 |
| bird_dev_question | 1,534 | 126,958 | 4.190 | 4.445 | 4.472 | 4.508 | 4.245 |
| bird_dev_sql | 1,534 | 247,705 | 3.095 | 3.267 | 3.230 | 3.229 | 3.138 |
| spider_dev_question | 1,034 | 70,370 | 4.505 | 4.743 | 4.760 | 4.766 | 4.676 |
| spider_dev_sql | 1,034 | 110,321 | 3.334 | 3.572 | 3.519 | 3.534 | 3.490 |

### Matched-vocab caveats

- **Less and narrower training data (cuts against sqlpup).** The 100k tokenizer was trained on the same ~780M-token SQL-heavy corpus -- far less and narrower text than cl100k saw. A matched-vocab win despite that handicap is strong; a loss is partly explained by it.
- **Experiment, not the shipped tokenizer.** The model ships the 32k tokenizer (`artifacts/tokenizer/tokenizer.json`); this 100k tokenizer is a science artifact only and is never used for training or inference.
- **Compression, not accuracy.** Fertility measures tokens per unit text, not downstream text-to-SQL accuracy.
