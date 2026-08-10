# Results

Execution accuracy (EX) on the BIRD **dev** split: 1,534 questions, official
BIRD execution semantics. Every file here is the summary JSON emitted by
`sqlpup eval score`, copied unmodified; the tables in the top-level README are
read off these. Raw prediction dumps and per-example verdicts are regenerable
and not committed.

Three decoding configurations appear throughout:

- **greedy**: one greedy decode per question.
- **greedy + compaction**: the same, but over-context prompts are re-rendered
  with a gold-blind compacted schema rather than recorded as empty
  (`--compact-overflow`). 129 of the 1,534 dev prompts are affected.
- **stack**: k=7 self-consistency voting on top of compaction
  (`--self-consistency 7 --compact-overflow`). This is the submitted
  configuration.

Only runs in the same configuration are comparable. The `.meta.json` sidecar
written next to every prediction file records which flags were in force.

Self-consistency is deterministic given a seed: each draw is sampled under
`manual_seed(seed + draw)`, so repeating a run reproduces it exactly. Error
bars therefore come from re-running at seeds 0, 101 and 202, and the `±` values
quoted anywhere in this repository are **standard deviations over those three
seeds, not standard errors**.

## `bird-dev/`: the reported table

| file | model | decoding | EX |
|---|---|---|---:|
| `sqlpup-sft-greedy.json` | sqlpup 394M, SFT | greedy | 17.41% |
| `sqlpup-sft-stack-seed{0,101,202}.json` | sqlpup 394M, SFT | stack | 22.95 / 22.88 / 22.69% |
| `sqlpup-grpo-greedy.json` | sqlpup 394M, SFT + GRPO | greedy + compaction | 19.23% |
| `sqlpup-grpo-stack-seed{0,101,202}.json` | sqlpup 394M, SFT + GRPO | stack | 23.66 / 23.08 / 23.79% |
| `qwen2.5-0.5b-greedy.json` | Qwen2.5-0.5B, SFT | greedy | 23.27% |
| `../ablations/qwen2.5-0.5b-compaction.json` | Qwen2.5-0.5B, SFT | greedy + compaction | 25.42% |
| `qwen2.5-0.5b-stack-seed{0,101,202}.json` | Qwen2.5-0.5B, SFT | stack | 31.94 / 30.44 / 32.07% |

The Qwen2.5-0.5B arm is a control: the same SFT recipe, the same evaluation
harness, the same inference stack, on an industrially pretrained base model.
Comparisons against it are only sound between the two **SFT** arms, because
the GRPO arm carries a reinforcement-learning step the control never received.

## `ablations/`: inference stack, sqlpup SFT on full dev

| file | change from greedy | EX |
|---|---|---:|
| `sqlpup-sft-ckpt35000-greedy.json` | earlier SFT checkpoint | 17.60% |
| `sqlpup-sft-compaction.json` | `--compact-overflow` | 17.86% |
| `sqlpup-sft-block-budget.json` | `--block-budget 96` | 17.86% |
| `sqlpup-sft-vote-k8.json` | `--self-consistency 8` | 22.29% |
| `sqlpup-sft-vote-k8-compaction.json` | both | 23.08% |
| `qwen2.5-0.5b-compaction.json` | `--compact-overflow`, control arm | 25.42% |
| `sqlpup-grpo-greedy-unaided.json` | GRPO ckpt-500, no compaction | 18.12% |

Compaction acts on the same 129 over-context prompts in every arm, but the
number it converts tracks model strength: 7 for sqlpup SFT, 17 for sqlpup+GRPO,
33 for the control, worth +0.46 / +1.11 / +2.15pt respectively. Any claim about
the pretraining gap must therefore hold the decoding configuration fixed; the
lever does not transfer between models.

`sqlpup-grpo-greedy-unaided.json` exists to test whether the RL gain survives
without that lever. It does not clear significance: 17.41 to 18.12 is +0.72pt at
McNemar exact two-sided p = 0.080, against +1.37pt at p = 0.0023 in the
compacted pair. Both McNemar values are computed from the per-example `match`
fields of the corresponding `eval score` outputs.

`sqlpup-sft-compaction.json` (17.86%) is the matched baseline for the GRPO
greedy comparison: both it and `bird-dev/sqlpup-grpo-greedy.json` run greedy
with `--compact-overflow`, so the +1.37pt difference isolates the RL step.

k=7 rather than k=8 ships because the BIRD "Few" self-consistency category
admits 1 to 7 candidates; the 0.13pt difference between them is inside seed
noise.

## `diagnostics/`: verdict panels

`sqlpup eval diagnose` output for the two SFT greedy runs: EX and valid-SQL
rate split by difficulty and JOIN count, schema-linking F1, and linking-block
drift.

## Regenerating

```bash
uv run sqlpup eval generate --model-dir <exported-model> --subset dev \
    --self-consistency 7 --compact-overflow --seed 0 --out preds.json
uv run sqlpup eval score --predictions preds.json --subset dev --out score.json
```

`eval score` prints the summary JSON to stdout and writes per-example verdicts
to `--out`. `eval gold-check` executes every gold query against its own database
first, certifying the harness independently of any model.
