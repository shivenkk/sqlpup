# sqlpup tokenizer fertility

_Generated 2026-07-21T00:00:00+00:00. Regenerate with `make fertility`._

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

- **Corpus provenance.** The SQL-heavy tokenizer mix (`configs/data/mix_tokenizer_v2.yaml`, ~780M proxy-tokens across five sources) cleaned and merged into one corpus (`data/clean/corpus_sample.jsonl.zst`). Fertility is measured only on the **holdout slice** -- documents where `xxh64(id) % 50 == 0`, the text the BPE trainer never saw (trained with `--holdout-mod 50`).
- **Eval sets.** BIRD dev and Spider dev questions and gold SQL, loaded from the cached `data/eval` archives. No tokenizer here trained on them, so they are the most meaningful out-of-distribution comparison surface.
- **Metrics.** Per group: **bytes** = UTF-8 byte length summed; **words** = whitespace split (`len(text.split())`) summed; **tokens** = ids per tokenizer summed, no special tokens added. **tokens/word** = tokens / words; **bytes/token** = bytes / tokens. Ratios come from summed totals (not per-document averages) and are deterministic.

## Tokens per word (lower is better)

| group | docs | words | sqlpup-bpe32k | gpt-4 (cl100k_base) | gpt-4o (o200k_base) | qwen2.5-0.5b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| starcoder_sql | 10 | 40 | 1.250 | 2.000 | 1.500 | 2.500 |
| **overall** | 10 | 40 | 1.250 | 2.000 | 1.500 | 2.500 |
| bird_dev_sql | 5 | 20 | 1.500 | 2.500 | 1.250 | 3.000 |

## Bytes per token (higher is better)

| group | docs | bytes | sqlpup-bpe32k | gpt-4 (cl100k_base) | gpt-4o (o200k_base) | qwen2.5-0.5b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| starcoder_sql | 10 | 200 | 4.000 | 2.500 | 3.333 | 2.000 |
| **overall** | 10 | 200 | 4.000 | 2.500 | 3.333 | 2.000 |
| bird_dev_sql | 5 | 100 | 3.333 | 2.000 | 4.000 | 1.667 |

## Caveats

- **Vocab-size asymmetry (read the ratios with this in mind).** sqlpup's vocab is 32,768 against roughly 100k (GPT-4), 200k (GPT-4o) and 152k (Qwen2.5) -- 3-6x larger. Those larger vocabularies hold more merges and generally compress SQL more tightly here in absolute bytes/token; sqlpup's aim is competitive compression at a fraction of the embedding parameters, purpose-built for the SQL-heavy mix, not to beat far larger vocabularies on raw ratio. The asymmetry is stated, not hidden, because the ratios cannot be read fairly without it.
- **Fertility is a compression metric, not task quality.** Fewer tokens per word is a throughput and context-length win; it is not by itself a measure of downstream text-to-SQL accuracy.
- **The corpus holdout is same-distribution as the training mix.** It is text the trainer never saw, but drawn from the same five sources. The eval-set groups (BIRD/Spider) are the genuinely out-of-distribution surface and the more meaningful comparison.
- **The corpus is post-clean-filter.** The `min_chars=200` clean filter drops short documents (here ~25k of ~400k, mostly short StarCoder SQL files and SchemaPile schemas), so very short SQL snippets are underrepresented in the corpus-slice groups -- another reason the eval-set groups matter.
