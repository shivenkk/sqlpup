"""Command-line interface: one subcommand per pipeline stage.

Each stage reads/writes JSONL(.zst) document files and prints a single JSON
stats line to stdout, so runs are easy to log, diff, and script.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from collections.abc import Callable, Iterator, Sequence
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, cast

from sqlpup.config import load_mix_config, load_tokenizer_config
from sqlpup.data.clean import CleanConfig, Cleaner
from sqlpup.data.decontaminate import (
    DEFAULT_NGRAM,
    Decontaminator,
    NgramIndex,
    load_eval_texts,
)
from sqlpup.data.dedup import ExactDeduper, NearDedupConfig, NearDeduper
from sqlpup.eval import DEFAULT_ROW_LIMIT, DEFAULT_TIMEOUT
from sqlpup.io import read_documents, write_documents

if TYPE_CHECKING:
    # Type-only import: the torch-backed loop is never imported at module load,
    # so `import sqlpup` and the `data` subcommands stay torch-free.
    from sqlpup.train.loop import Trainer

Handler = Callable[[argparse.Namespace], int]


def _emit(stats: dict[str, object]) -> None:
    print(json.dumps(stats, sort_keys=True))


def _download_fast_exit() -> None:
    """Exit the process immediately after a successful streaming download.

    ``datasets`` streams parquet sources (e.g. ``bigcode/the-stack-dedup``)
    through pyarrow, which lazily starts libarrow's process-global I/O thread
    pool. That pool's C++ destructor runs at C-runtime finalization
    (``__cxa_finalize``) and intermittently deadlocks in ``ThreadPool::Shutdown``,
    waiting on a condition variable its worker never re-signals -- so the command
    can hang after every output file, manifest, and the summary JSON are already
    written (confirmed by a native stack sample of a hung run: the main thread
    parked in ``arrow::internal::ThreadPool::~ThreadPool``). There is no public
    API to join that global pool cleanly, and closing our own stream iterator
    does not stop the static destructor from running, so we bypass the hanging
    teardown with ``os._exit``. Called only on the success path, after stdout is
    flushed and every manifest is fsynced, so no output can be truncated and no
    error is ever masked. Isolated in one function so tests can patch it without
    terminating the test runner.
    """
    os._exit(0)


def _cmd_data_download(args: argparse.Namespace) -> int:
    from sqlpup.data.download import download_source

    mix = load_mix_config(Path(args.config))
    out_dir = Path(args.out_dir) if args.out_dir else mix.out_dir
    sources = [mix.source(args.source)] if args.source else list(mix.sources)
    results = []
    for cfg in sources:
        if args.limit_docs is not None:
            cfg = dataclasses.replace(cfg, doc_cap=args.limit_docs)
        results.append(download_source(cfg, out_dir).as_dict())
    _emit({"downloads": results})
    # Every output file and manifest is now on disk. Flush the summary JSON, then
    # skip the interpreter/C-runtime teardown that libarrow's thread pool can hang
    # on (see _download_fast_exit). The `return 0` is reached only when the hook is
    # patched out (tests); a real run exits here with code 0.
    sys.stdout.flush()
    sys.stderr.flush()
    _download_fast_exit()
    return 0


def _cmd_data_clean(args: argparse.Namespace) -> int:
    cleaner = Cleaner(CleanConfig(min_chars=args.min_chars))
    # One or more inputs are streamed through a single cleaner into one output,
    # so the per-source files of a mix collapse into a combined corpus while each
    # document keeps its own Document.source provenance.
    docs = chain.from_iterable(read_documents(Path(infile)) for infile in args.infiles)
    count = write_documents(Path(args.out), cleaner.run(docs))
    _emit({"written": count, **cleaner.stats.as_dict()})
    return 0


def _cmd_data_dedup(args: argparse.Namespace) -> int:
    exact = ExactDeduper()
    stream = exact.filter(read_documents(Path(args.infile)))
    near: NearDeduper | None = None
    if args.near:
        near = NearDeduper(
            NearDedupConfig(
                shingle_size=args.near_shingle_size,
                num_perm=args.near_num_perm,
                bands=args.near_bands,
                threshold=args.near_threshold,
            )
        )
        stream = near.filter(stream)
    count = write_documents(Path(args.out), stream)
    stats: dict[str, object] = {"written": count, **exact.stats.as_dict()}
    if near is not None:
        stats.update(near.stats.as_dict())
    _emit(stats)
    return 0


def _cmd_data_decontaminate(args: argparse.Namespace) -> int:
    stats: dict[str, object] = {}
    if args.index_in:
        index = NgramIndex.load(Path(args.index_in))
    else:
        texts = load_eval_texts([Path(p) for p in args.eval_json])
        index = NgramIndex.build(texts, n=args.ngram)
        stats["eval_texts"] = len(texts)
    if args.index_out:
        index.save(Path(args.index_out))
    decon = Decontaminator(index, threshold=args.threshold)
    count = write_documents(Path(args.out), decon.filter(read_documents(Path(args.infile))))
    _emit({"written": count, **stats, **decon.stats.as_dict()})
    return 0


def _cmd_data_build_eval_index(args: argparse.Namespace) -> int:
    from sqlpup.config import load_eval_sets_config
    from sqlpup.data.eval_sets import build_eval_index

    config = load_eval_sets_config(Path(args.config))
    index_path = Path(args.index_out)
    report_path = (
        Path(args.report_out)
        if args.report_out
        else index_path.with_name(f"{index_path.stem}_report.json")
    )
    report = build_eval_index(config, Path(args.data_dir), index_path, report_path, n=args.ngram)
    _emit(
        {
            "index": str(index_path),
            "report": str(report_path),
            "n": report["n"],
            "total_shingles": report["total_shingles"],
            "sources": {
                s["name"]: {"examples": s["examples"], "text_entries": s["text_entries"]}
                for s in report["sources"]
            },
        }
    )
    return 0


def _cmd_data_stats(args: argparse.Namespace) -> int:
    from sqlpup.data.stats import compute_corpus_stats, render_table, write_report

    mix = load_mix_config(Path(args.config))
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else mix.out_dir
    stats = compute_corpus_stats(mix, corpus_dir)
    report = write_report(stats, Path(args.report_out))
    _emit(report)  # one JSON stats line on stdout, like every other stage
    print(render_table(stats), file=sys.stderr)  # human table -> stderr, off the stats stream
    return 0


def _cmd_tokenizer_train(args: argparse.Namespace) -> int:
    from sqlpup.tokenizer.train import is_holdout, train_bpe

    config = load_tokenizer_config(Path(args.config))
    if args.vocab_size is not None:
        config = dataclasses.replace(config, vocab_size=args.vocab_size)
    if args.out_dir is not None:
        config = dataclasses.replace(config, out_dir=Path(args.out_dir))

    holdout_mod: int | None = args.holdout_mod
    if holdout_mod is not None and holdout_mod < 1:
        raise SystemExit("--holdout-mod must be a positive integer")

    counts = {"used": 0, "held_out": 0}

    def corpus() -> Iterator[str]:
        # Streamed (not materialized) so the ~hundreds-of-MB sample corpus never
        # sits in memory as a list; counts settle once the trainer drains it.
        for doc in read_documents(Path(args.corpus)):
            if counts["used"] >= config.sample_docs:
                break
            if holdout_mod is not None and is_holdout(doc.id, holdout_mod):
                counts["held_out"] += 1
                continue
            counts["used"] += 1
            yield doc.text

    path = train_bpe(config, corpus())
    stats: dict[str, object] = {
        "tokenizer": str(path),
        "vocab_size": config.vocab_size,
        "docs_used": counts["used"],
    }
    if holdout_mod is not None:
        stats["holdout_mod"] = holdout_mod
        stats["docs_held_out"] = counts["held_out"]
    _emit(stats)
    return 0


def _cmd_tokenizer_fertility(args: argparse.Namespace) -> int:
    from datetime import UTC, datetime

    from sqlpup.config import load_eval_sets_config
    from sqlpup.tokenizer import fertility

    holdout_mod: int = args.holdout_mod
    if holdout_mod < 1:
        raise SystemExit("--holdout-mod must be a positive integer")

    # Parse any --extra-tokenizer PATH:LABEL specs first (cheap, offline) so a
    # malformed value fails before the network work below. rpartition splits on
    # the last colon, so filesystem paths without one are rejected cleanly.
    extra_specs: list[tuple[Path, str]] = []
    reserved_labels = {
        fertility.SQLPUP_NAME,
        fertility.GPT4_NAME,
        fertility.GPT4O_NAME,
        fertility.QWEN_NAME,
    }
    for spec in args.extra_tokenizers or []:
        path_str, sep, label = spec.rpartition(":")
        if not sep or not path_str or not label:
            raise SystemExit(f"--extra-tokenizer expects PATH:LABEL, got {spec!r}")
        # A label equal to the primary sqlpup column or a reference display name
        # would drop a real column from the base tables (or collide its counts).
        if label in reserved_labels:
            raise SystemExit(
                f"--extra-tokenizer label {label!r} collides with a built-in column; "
                f"pick a distinct label (reserved: {', '.join(sorted(reserved_labels))})"
            )
        extra_specs.append((Path(path_str), label))

    # The reference tokenizers need the optional 'report' extra (tiktoken) and a
    # one-time download; build them first so a missing extra fails loud and early
    # with an actionable hint, before any real work.
    try:
        references = fertility.load_reference_adapters()
    except ModuleNotFoundError:
        print(fertility.REPORT_EXTRA_HINT, file=sys.stderr)
        return 1

    # Column order: primary sqlpup, then any matched sqlpup-side tokenizers, then
    # the general-purpose references.
    adapters: list[fertility.TokenizerAdapter] = [
        fertility.load_sqlpup_adapter(Path(args.tokenizer)),
        *(fertility.load_sqlpup_adapter(path, name=label) for path, label in extra_specs),
        *references,
    ]
    matched = [label for _, label in extra_specs]

    corpus_groups = fertility.holdout_groups(Path(args.corpus), holdout_mod)
    eval_config = load_eval_sets_config(Path(args.eval_config))
    eval_groups = fertility.build_eval_groups(
        eval_config, Path(args.eval_dir), fertility.EVAL_GROUP_SPECS
    )

    report = fertility.build_report(
        adapters,
        corpus_groups,
        eval_groups,
        corpus=str(Path(args.corpus)),
        holdout_mod=holdout_mod,
        eval_dir=str(Path(args.eval_dir)),
        generated_at=datetime.now(UTC).isoformat(),
        matched=matched,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report_path: str | None = None
    if args.report:
        markdown_path = Path(args.report)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(fertility.render_markdown(report), encoding="utf-8")
        report_path = str(markdown_path)

    _emit(report.emit_summary(out=str(out_path), report=report_path))
    return 0


def _cmd_data_shard(args: argparse.Namespace) -> int:
    from sqlpup.shard import ShardWriter
    from sqlpup.tokenizer.train import is_holdout, load_tokenizer

    holdout_out_dir = args.holdout_out_dir
    holdout_mod = args.holdout_mod
    if holdout_out_dir is not None and holdout_mod < 1:
        raise SystemExit("--holdout-mod must be a positive integer")

    tokenizer = load_tokenizer(Path(args.tokenizer))
    eos_id = tokenizer.token_to_id(args.eos_token)
    if eos_id is None:
        raise SystemExit(f"eos token {args.eos_token!r} not in tokenizer vocab")

    writer = ShardWriter(Path(args.out_dir), eos_id=eos_id, shard_size_tokens=args.shard_size)
    # Optional doc-level train/eval split in the SAME streaming pass: documents
    # whose id falls in the deterministic is_holdout slice are routed to a second
    # writer (the eval shards ~ the tokenizer's reserved holdout), everything else
    # to the train writer. The corpus is never materialized -- one process, one pass.
    holdout_writer = (
        ShardWriter(Path(holdout_out_dir), eos_id=eos_id, shard_size_tokens=args.shard_size)
        if holdout_out_dir is not None
        else None
    )

    docs = 0
    holdout_docs = 0
    for infile in args.infiles:
        for doc in read_documents(Path(infile)):
            token_ids = tokenizer.encode(doc.text).ids
            if holdout_writer is not None and is_holdout(doc.id, holdout_mod):
                holdout_writer.add(token_ids)
                holdout_docs += 1
            else:
                writer.add(token_ids)
                docs += 1

    index = writer.close()
    stats: dict[str, object] = {
        "docs": docs,
        "total_tokens": index.total_tokens,
        "shards": len(index.shards),
    }
    if holdout_writer is not None:
        holdout_index = holdout_writer.close()
        stats["holdout_docs"] = holdout_docs
        stats["holdout_total_tokens"] = holdout_index.total_tokens
        stats["holdout_shards"] = len(holdout_index.shards)
    _emit(stats)
    return 0


def _cmd_eval_score(args: argparse.Namespace) -> int:
    # Imported lazily and torch-free: scoring runs CPU-only via the stdlib sqlite3
    # sandbox, so `sqlpup eval` never touches the optional torch extra.
    from sqlpup.eval import ExecutionScorer
    from sqlpup.eval.dataset import ensure_databases, load_examples
    from sqlpup.eval.scorer import load_predictions, score_predictions

    eval_dir = Path(args.eval_dir)
    examples = load_examples(eval_dir, args.subset)
    predictions = load_predictions(Path(args.predictions))
    ensure_databases(eval_dir, {example.db_id for example in examples})

    with ExecutionScorer(timeout=args.timeout, row_limit=args.row_limit) as scorer:
        report = score_predictions(examples, predictions, scorer, eval_dir, subset=args.subset)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report.detail_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    _emit(report.summary_dict())  # overall EX + per-difficulty on the stats stream
    return 0


def _cmd_rl_grpo(args: argparse.Namespace) -> int:
    """GRPO over the execution reward, from an SFT checkpoint."""
    # Lazy imports: the RL stack pulls trl + datasets, which the eval-only
    # boxes neither have nor need.
    from tokenizers import Tokenizer

    from sqlpup.eval.dataset import load_bird_file
    from sqlpup.rl.rollout_data import build_rollout_rows
    from sqlpup.rl.train import run_grpo

    tokenizer = Tokenizer.from_file(args.tokenizer)

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text).ids)

    if args.unseen_of is not None:
        # The RL pool: exactly the rows an SFT run at (rate, seed) skipped.
        # Measured necessity -- the first pilot trained on BIRD-train, which
        # SFT oversampled 10x, opened at 0.769 mean reward and never moved.
        from sqlpup.rl.unseen import iter_unseen_rows

        examples = list(
            iter_unseen_rows(
                Path(args.data),
                sample_rate=args.unseen_of,
                seed=args.unseen_seed,
                limit=args.limit,
            )
        )
    else:
        examples = load_bird_file(Path(args.data))
        if args.limit:
            examples = examples[: args.limit]

    rows, stats = build_rollout_rows(
        examples,
        args.db_root,
        count_tokens=count_tokens,
        context_limit=args.context_limit,
        max_completion_length=args.max_completion_length,
    )
    print(json.dumps({"rollout_rows": stats}, sort_keys=True), file=sys.stderr)
    if not rows:
        print("no usable rollout rows; refusing to start", file=sys.stderr)
        return 1

    receipt = run_grpo(
        model_dir=args.model_dir,
        rows=rows,
        out_dir=args.out_dir,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_completion_length=args.max_completion_length,
        context_limit=args.context_limit,
        learning_rate=args.lr,
        beta=args.beta,
        temperature=args.temperature,
        max_steps=args.max_steps,
        run_name=args.run_name,
        report_to="wandb" if args.run_name else "none",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
    )
    full = {**receipt, "rollout_rows": stats}
    if args.receipt:
        from sqlpup.rl.train import write_receipt

        write_receipt(args.receipt, full)
    _emit(full)
    # A pre-registered halt is a real outcome, not a crash: exit non-zero so the
    # box script stops rather than evaluating a collapsed policy.
    return 1 if receipt.get("halt_reason") else 0


def _cmd_eval_gold_check(args: argparse.Namespace) -> int:
    """Execute every gold query against itself -- the harness's own certificate.

    Any non-match here is a harness/data problem (extraction, timeout budget,
    comparison semantics), never a model problem; model numbers are meaningless
    until this exits 0 on the target subset.
    """
    # Lazy, torch-free imports -- same rationale as _cmd_eval_score.
    from sqlpup.eval import ExecutionScorer
    from sqlpup.eval.dataset import ensure_databases, load_examples, resolve_db_path

    eval_dir = Path(args.eval_dir)
    examples = load_examples(eval_dir, args.subset)
    ensure_databases(eval_dir, {example.db_id for example in examples})

    failures: list[dict[str, object]] = []
    with ExecutionScorer(timeout=args.timeout, row_limit=args.row_limit) as scorer:
        for example in examples:
            db_path = resolve_db_path(eval_dir, example.db_id)
            verdict = scorer.score(example.gold_sql, example.gold_sql, db_path)
            if not verdict.match:
                failures.append(
                    {
                        "index": example.index,
                        "question_id": example.question_id,
                        "db_id": example.db_id,
                        "category": verdict.category.value,
                    }
                )
    certificate: dict[str, object] = {
        "subset": args.subset,
        "total": len(examples),
        "passed": len(examples) - len(failures),
        "failures": failures,
    }
    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(certificate, sort_keys=True), encoding="utf-8")
    _emit(certificate)
    return 1 if failures else 0


def _cmd_eval_diagnose(args: argparse.Namespace) -> int:
    """Compute the pre-registered verdict panel for a predictions file."""
    from sqlpup.eval.dataset import load_examples
    from sqlpup.eval.diagnose import diagnose_predictions

    eval_dir = Path(args.eval_dir)
    examples = load_examples(eval_dir, args.subset)
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    score = json.loads(Path(args.score).read_text(encoding="utf-8"))
    panel = diagnose_predictions(predictions, examples, eval_dir, score=score)
    panel["subset"] = args.subset
    panel["predictions"] = str(args.predictions)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    _emit(panel)
    return 0


def _cmd_sft_prepare(args: argparse.Namespace) -> int:
    """Stream raw rows -> filtered tokenized pairs + manifest (torch-free)."""
    from tokenizers import Tokenizer

    from sqlpup.eval.prompts import SPECS
    from sqlpup.sft.prepare import iter_json_array, prepare_pairs

    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id(args.eos_token)
    if eos_id is None:
        raise SystemExit(f"tokenizer has no {args.eos_token} token")
    manifest = prepare_pairs(
        iter_json_array(Path(args.data)),
        Path(args.db_root),
        tokenizer,
        Path(args.out),
        eos_id=int(eos_id),
        spec=SPECS[args.prompt_spec],
        context_limit=args.context_limit,
        sample_rate=args.sample_rate,
        seed=args.seed,
        limit=args.limit,
        link_targets=args.link_targets,
        reduce_overflow=args.reduce_overflow,
        workers=args.workers,
        tokenizer_path=args.tokenizer,
    )
    _emit(manifest)
    return 0


def _resolve_pad_id(tokenizer: object, eos_token: str) -> int:
    """Padding id for *any* tokenizer: ours, or a control model's.

    Padding is masked out of the loss, so its identity is irrelevant as long
    as it exists -- what matters is never passing ``None`` downstream. The
    Qwen control arm died exactly here after an hour of data preparation
    because both candidate sentinels were hardcoded to our own vocabulary.
    """
    for candidate in ("<|pad|>", eos_token, "<|end|>"):
        token_id = tokenizer.token_to_id(candidate)  # type: ignore[attr-defined]
        if token_id is not None:
            return int(token_id)
    raise SystemExit(f"tokenizer has no pad candidate (tried <|pad|>, {eos_token}, <|end|>)")


def _cmd_sft_train(args: argparse.Namespace) -> int:
    """Fine-tune the exported model on a prepared pair file (torch extra)."""
    from tokenizers import Tokenizer

    from sqlpup.sft.train import run_sft

    tokenizer = Tokenizer.from_file(args.tokenizer)
    pad_id = _resolve_pad_id(tokenizer, args.eos_token)
    receipts = run_sft(
        model_dir=args.model_dir,
        pairs_path=Path(args.pairs),
        out_dir=Path(args.out_dir),
        pad_id=int(pad_id),
        epochs=args.epochs,
        learning_rate=args.lr,
        per_device_batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        seed=args.seed,
        save_steps=args.save_steps,
        resume=args.resume,
        run_name=args.run_name,
    )
    _emit(receipts)
    return 0


def _cmd_eval_generate(args: argparse.Namespace) -> int:
    """Render prompts, generate predictions, and write them with provenance."""
    # The one torch-touching import, resolved at call time from the module
    # attribute so tests can substitute a scripted generator.
    import sqlpup.eval.hf_generator as hf_generator
    from sqlpup.eval.dataset import ensure_databases, load_examples
    from sqlpup.eval.execution import ExecutionScorer
    from sqlpup.eval.predict import generate_predictions
    from sqlpup.eval.prompts import SPECS

    spec = SPECS[args.prompt_spec]

    eval_dir = Path(args.eval_dir)
    examples = load_examples(eval_dir, args.subset)
    if args.limit is not None:
        examples = examples[: args.limit]
    ensure_databases(eval_dir, {example.db_id for example in examples})

    generator = hf_generator.HFGreedyGenerator(
        args.model_dir,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        context_limit=args.context_limit,
    )
    if args.self_consistency:
        with ExecutionScorer(timeout=args.timeout) as scorer:
            records, meta = generate_predictions(
                examples,
                eval_dir,
                generator,
                spec=spec,
                self_consistency=args.self_consistency,
                temperature=args.temperature,
                seed=args.seed,
                scorer=scorer,
                compact_overflow=args.compact_overflow,
            )
    elif args.refine is None:
        records, meta = generate_predictions(
            examples,
            eval_dir,
            generator,
            spec=spec,
            block_constrain=args.constrain_block,
            two_pass=args.two_pass,
            column_hints=args.column_hints,
            expand_fk=args.expand_fk,
            compact_overflow=args.compact_overflow,
            block_budget=args.block_budget,
        )
    else:
        with ExecutionScorer(timeout=args.timeout) as scorer:
            records, meta = generate_predictions(
                examples,
                eval_dir,
                generator,
                spec=spec,
                refine_retries=args.refine,
                scorer=scorer,
                compact_overflow=args.compact_overflow,
            )

    meta["subset"] = args.subset
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    meta_path = Path(str(out_path) + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _emit(
        {
            "written": str(out_path),
            "meta": str(meta_path),
            "subset": args.subset,
            "examples": meta["examples"],
            "prompt_spec": meta["prompt_spec"],
            "generator": meta["generator"],
            "refine_retries": meta["refine_retries"],
        }
    )
    return 0


def _configure_run_log(path: Path) -> None:
    """Log progress to stderr and to a file a third party can read after a crash."""
    handlers: list[logging.Handler] = [logging.StreamHandler(), logging.FileHandler(path)]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger("sqlpup")
    root.handlers.clear()
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(logging.INFO)


def _cmd_eval_predict(args: argparse.Namespace) -> int:
    """Predict on a held-out split and write the file BIRD's evaluator reads."""
    import contextlib

    import sqlpup.eval.hf_generator as hf_generator
    from sqlpup.eval.dataset import BirdExample, db_path_under, load_prediction_examples
    from sqlpup.eval.execution import ExecutionScorer
    from sqlpup.eval.predict import generate_predictions
    from sqlpup.eval.prompts import SPECS
    from sqlpup.eval.submit import run_resumable, write_bird_predictions

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_run_log(out_dir / "run.log")

    spec = SPECS[args.prompt_spec]
    examples = load_prediction_examples(args.examples)
    if args.limit is not None:
        examples = examples[: args.limit]

    db_root = Path(args.db_root)
    absent = sorted({e.db_id for e in examples if not db_path_under(db_root, e.db_id).exists()})
    if absent:
        raise SystemExit(
            f"{len(absent)} database(s) not found under {db_root} "
            f"(expected <root>/<db_id>/<db_id>.sqlite; first missing: {absent[0]!r})"
        )

    generator = hf_generator.HFGreedyGenerator(
        args.model_dir,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        context_limit=args.context_limit,
    )

    with contextlib.ExitStack() as stack:
        # Voting and compaction both execute candidate SQL, so the sandbox opens
        # once for the whole run rather than once per chunk.
        scorer = (
            stack.enter_context(ExecutionScorer(timeout=args.timeout))
            if args.self_consistency
            else None
        )

        def predict(
            chunk: Sequence[BirdExample],
        ) -> tuple[list[dict[str, object]], dict[str, object]]:
            return generate_predictions(
                chunk,
                # Unused because db_root is set: the cached dev release is never consulted.
                db_root,
                generator,
                spec=spec,
                self_consistency=args.self_consistency,
                temperature=args.temperature,
                seed=args.seed,
                scorer=scorer,
                compact_overflow=args.compact_overflow,
                db_root=db_root,
            )

        records, meta = run_resumable(
            examples, predict, out_dir / "progress.jsonl", chunk_size=args.chunk_size
        )

    predict_path = out_dir / f"predict_{args.split}.json"
    empties = write_bird_predictions(records, examples, predict_path)
    records_path = out_dir / "records.json"
    records_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    meta = {**meta, "split": args.split, "db_root": str(db_root), "empty_predictions": empties}
    meta_path = Path(str(records_path) + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _emit(
        {
            "written": str(predict_path),
            "records": str(records_path),
            "log": str(out_dir / "run.log"),
            "examples": meta["examples"],
            "empty_predictions": empties,
            "compacted": meta.get("compacted"),
            "prompt_spec": meta["prompt_spec"],
        }
    )
    return 0


_TRAIN_EXTRA_HINT = (
    "sqlpup train requires PyTorch, the optional 'train' extra, which is not installed.\n"
    "Install it with one of:\n"
    '    pip install "sqlpup[train]"\n'
    "    uv sync --extra train"
)


def _load_trainer() -> type[Trainer]:
    """Import the torch-backed :class:`~sqlpup.train.loop.Trainer` on demand.

    Raises :class:`ImportError` if the optional ``train`` extra (PyTorch) is not
    installed. Kept as a tiny indirection so the missing-torch path is easy to
    exercise in tests and so nothing torch-shaped is imported until ``train`` runs.
    """
    from sqlpup.train.loop import Trainer

    return Trainer


def _resolve_resume(resume: str | None, out_dir: Path) -> Path | None:
    """Map the ``--resume`` value to a checkpoint path (or ``None`` for fresh).

    ``None`` (flag omitted) starts fresh. ``"auto"`` resumes from ``out_dir`` when
    it holds a ``latest.json`` pointer, else falls back to a fresh start. Any other
    value is treated as an explicit checkpoint path (a ``.pt`` file or a directory).
    """
    if resume is None:
        return None
    if resume == "auto":
        return out_dir if (out_dir / "latest.json").exists() else None
    return Path(resume)


def _latest_checkpoint(out_dir: Path) -> str | None:
    """The path recorded in ``out_dir/latest.json`` (rank-0 only), else ``None``."""
    latest = out_dir / "latest.json"
    if not latest.exists():
        return None
    meta = json.loads(latest.read_text(encoding="utf-8"))
    return str(out_dir / str(meta["path"]))


def _latest_step(out_dir: Path) -> int | None:
    """The step recorded in ``out_dir/latest.json``, or ``None`` if there is none."""
    latest = out_dir / "latest.json"
    if not latest.exists():
        return None
    step = json.loads(latest.read_text(encoding="utf-8")).get("step")
    return None if step is None else int(step)


def _fresh_over_existing_error(out_dir: Path) -> str:
    """Message shown when a fresh start would clobber an existing run in ``out_dir``."""
    step = _latest_step(out_dir)
    at = f" (latest checkpoint at step {step})" if step is not None else ""
    return (
        f"refusing to start fresh: {out_dir}/latest.json already records a run{at}.\n"
        "Starting fresh would overwrite and prune those checkpoints. Instead:\n"
        "    --resume auto    continue this run from its latest checkpoint\n"
        "    --resume PATH    resume from a specific checkpoint file/dir\n"
        "    --out-dir DIR    write this run to a different directory\n"
        "    --force-fresh    discard the existing checkpoints and start over (destructive)"
    )


def _cmd_train(args: argparse.Namespace) -> int:
    try:
        trainer_cls = _load_trainer()
    except ImportError:
        print(_TRAIN_EXTRA_HINT, file=sys.stderr)
        return 1

    # Route progress logs to stderr (stdout carries only the final stats line).
    # Under pytest a root handler already exists, so this is a no-op there.
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from sqlpup.train.config import load_train_config

    cfg = load_train_config(Path(args.config))
    if args.out_dir:
        cfg = dataclasses.replace(cfg, out_dir=Path(args.out_dir))

    device = None if args.device == "auto" else args.device
    out_dir = cfg.out_dir

    # Precedence: an explicit --resume resumes; --force-fresh starts fresh even over
    # an existing run; otherwise a populated out_dir is refused (never silently
    # overwritten -- spot-interrupted runs must not be destroyed by a bare relaunch).
    if args.resume is not None:
        resume_path = _resolve_resume(args.resume, out_dir)
        if resume_path is not None and not resume_path.exists():
            print(f"resume checkpoint not found: {resume_path}", file=sys.stderr)
            return 1
    elif not args.force_fresh and (out_dir / "latest.json").exists():
        print(_fresh_over_existing_error(out_dir), file=sys.stderr)
        return 1
    else:
        resume_path = None

    if resume_path is not None:
        trainer = trainer_cls.from_checkpoint(resume_path, cfg, device=device)
    else:
        trainer = trainer_cls(cfg, device=device)
    resumed = resume_path is not None

    with trainer:
        start_step = trainer.step
        stats = trainer.train(resumed=resumed)
        eval_stats: dict[str, object] = {}
        if cfg.eval_shard_index is not None:
            # Collective under DDP (all-reduce): every rank must call it in lockstep.
            eval_loss, eval_ppl = trainer.evaluate()
            eval_stats = {"eval_loss": eval_loss, "eval_ppl": eval_ppl}
        if trainer.rank == 0:
            step = int(stats["step"])
            _emit(
                {
                    "step": step,
                    "start_step": start_step,
                    "resumed": resumed,
                    "final_loss": stats["final_loss"],
                    **eval_stats,
                    "tokens_seen": int(stats["effective_tokens_per_step"]) * step,
                    "checkpoint": _latest_checkpoint(cfg.out_dir),
                }
            )
    return 0


_EXPORT_EXTRA_HINT = (
    "sqlpup export requires PyTorch and safetensors, the optional 'train' extra, "
    "which is not installed.\n"
    "Install it with one of:\n"
    '    pip install "sqlpup[train]"\n'
    "    uv sync --extra train"
)


def _cmd_export(args: argparse.Namespace) -> int:
    # torch (checkpoint load) and safetensors (weights write) are the optional
    # 'train' extra; keep them out of module import so the CLI stays torch-free.
    try:
        import safetensors  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        print(_EXPORT_EXTRA_HINT, file=sys.stderr)
        return 1

    from sqlpup.model.config import load_model_config
    from sqlpup.model.export import write_hf_directory
    from sqlpup.train.checkpoint import load_checkpoint

    ckpt = load_checkpoint(Path(args.checkpoint))
    # The architecture (n_heads/rope_theta/... are not recoverable from tensor
    # shapes alone) comes from the model config the run recorded in its snapshot.
    try:
        model_config_path = Path(str(ckpt["train_config"]["model_config"]))
    except (KeyError, TypeError):
        print(
            f"checkpoint {args.checkpoint} has no train_config.model_config; "
            "cannot recover the model architecture to export",
            file=sys.stderr,
        )
        return 1
    if not model_config_path.exists():
        print(
            f"model config referenced by the checkpoint not found: {model_config_path}\n"
            "Paths in the checkpoint are relative to the run's root -- run `sqlpup export` "
            "from there, or place the model config at that path.",
            file=sys.stderr,
        )
        return 1

    cfg = load_model_config(model_config_path)
    summary = write_hf_directory(
        Path(args.out),
        state_dict=ckpt["model"],
        cfg=cfg,
        dtype=args.dtype,
        tokenizer_path=Path(args.tokenizer) if args.tokenizer else None,
    )
    _emit(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sqlpup", description=__doc__)
    top = parser.add_subparsers(dest="command", required=True)

    data = top.add_parser("data", help="corpus pipeline stages").add_subparsers(
        dest="subcommand", required=True
    )

    download = data.add_parser("download", help="stream sources to jsonl.zst")
    download.add_argument("--config", required=True)
    download.add_argument("--source", help="single source name (default: all)")
    download.add_argument("--out-dir")
    download.add_argument("--limit-docs", type=int, help="cap docs per source")
    download.set_defaults(func=_cmd_data_download)

    clean = data.add_parser("clean", help="normalize and quality-filter")
    clean.add_argument(
        "--in",
        dest="infiles",
        nargs="+",
        required=True,
        help="one or more input jsonl(.zst) files, merged into a single output corpus",
    )
    clean.add_argument("--out", required=True)
    clean.add_argument("--min-chars", type=int, default=CleanConfig().min_chars)
    clean.set_defaults(func=_cmd_data_clean)

    dedup = data.add_parser("dedup", help="drop exact (and optionally near) duplicates")
    dedup.add_argument("--in", dest="infile", required=True)
    dedup.add_argument("--out", required=True)
    dedup.add_argument(
        "--near", action="store_true", help="also drop near-duplicates via minhash+lsh"
    )
    dedup.add_argument("--near-shingle-size", type=int, default=NearDedupConfig().shingle_size)
    dedup.add_argument("--near-num-perm", type=int, default=NearDedupConfig().num_perm)
    dedup.add_argument("--near-bands", type=int, default=NearDedupConfig().bands)
    dedup.add_argument("--near-threshold", type=float, default=NearDedupConfig().threshold)
    dedup.set_defaults(func=_cmd_data_dedup)

    decontaminate = data.add_parser("decontaminate", help="drop eval-set overlaps")
    decontaminate.add_argument("--in", dest="infile", required=True)
    decontaminate.add_argument("--out", required=True)
    eval_source = decontaminate.add_mutually_exclusive_group(required=True)
    eval_source.add_argument("--eval-json", nargs="+", help="build the index from eval JSON files")
    eval_source.add_argument("--index-in", help="load a prebuilt n-gram index instead")
    decontaminate.add_argument("--ngram", type=int, default=DEFAULT_NGRAM)
    decontaminate.add_argument("--threshold", type=int, default=1)
    decontaminate.add_argument("--index-out", help="persist the n-gram index")
    decontaminate.set_defaults(func=_cmd_data_decontaminate)

    build_eval_index = data.add_parser(
        "build-eval-index", help="fetch eval sets and build the decontamination index"
    )
    build_eval_index.add_argument("--config", default="configs/data/eval_sets.yaml")
    build_eval_index.add_argument("--data-dir", default="data/eval", help="download cache dir")
    build_eval_index.add_argument("--index-out", default="artifacts/decontam/eval_index.json")
    build_eval_index.add_argument("--report-out", help="default: <index>_report.json alongside")
    build_eval_index.add_argument("--ngram", type=int, default=DEFAULT_NGRAM)
    build_eval_index.set_defaults(func=_cmd_data_build_eval_index)

    stats = data.add_parser("stats", help="corpus stats and data-mix report")
    stats.add_argument("--config", required=True)
    stats.add_argument("--corpus-dir", help="corpus directory (default: mix out_dir)")
    stats.add_argument("--report-out", default="artifacts/stats/corpus_stats.json")
    stats.set_defaults(func=_cmd_data_stats)

    shard = data.add_parser("shard", help="tokenize into uint16 shards")
    shard.add_argument("--in", dest="infiles", nargs="+", required=True)
    shard.add_argument("--tokenizer", required=True)
    shard.add_argument("--out-dir", required=True)
    shard.add_argument("--shard-size", type=int, default=100_000_000)
    shard.add_argument("--eos-token", default="<|end|>")
    shard.add_argument(
        "--holdout-out-dir",
        help="also write a doc-level holdout split here (the eval shards): documents whose "
        "xxh64(id) %% N == 0 go here instead of --out-dir, matching the tokenizer's holdout",
    )
    shard.add_argument(
        "--holdout-mod",
        type=int,
        default=50,
        metavar="N",
        help="holdout modulus used with --holdout-out-dir (xxh64(id) %% N == 0); match the "
        "value used at tokenizer-training time (default: 50)",
    )
    shard.set_defaults(func=_cmd_data_shard)

    tokenizer = top.add_parser("tokenizer", help="tokenizer stages").add_subparsers(
        dest="subcommand", required=True
    )
    train = tokenizer.add_parser("train", help="train byte-level bpe")
    train.add_argument("--config", required=True)
    train.add_argument("--corpus", required=True)
    train.add_argument("--vocab-size", type=int, help="override config vocab size")
    train.add_argument("--out-dir", help="override config output dir")
    train.add_argument(
        "--holdout-mod",
        type=int,
        default=None,
        metavar="N",
        help="reserve a 1/N holdout slice for fertility eval by skipping documents "
        "whose xxh64(id) is divisible by N (default: off, train on every document)",
    )
    train.set_defaults(func=_cmd_tokenizer_train)

    fertility = tokenizer.add_parser(
        "fertility", help="compare tokenizer compression vs gpt-4/gpt-4o/qwen2.5"
    )
    fertility.add_argument("--tokenizer", required=True, help="sqlpup tokenizer.json path")
    fertility.add_argument(
        "--extra-tokenizer",
        action="append",
        dest="extra_tokenizers",
        metavar="PATH:LABEL",
        help="additional sqlpup-side tokenizer.json to compare as a labeled column "
        "(repeatable), e.g. artifacts/tokenizer_matched_100k/tokenizer.json:sqlpup-100k",
    )
    fertility.add_argument("--corpus", required=True, help="sample corpus jsonl(.zst) path")
    fertility.add_argument(
        "--holdout-mod",
        type=int,
        default=50,
        metavar="N",
        help="measure only the 1/N holdout slice (xxh64(id) %% N == 0); must match the value "
        "used at tokenizer-training time (default: 50)",
    )
    fertility.add_argument(
        "--eval-config", default="configs/data/eval_sets.yaml", help="eval-sets config"
    )
    fertility.add_argument("--eval-dir", default="data/eval", help="cached eval-set archives dir")
    fertility.add_argument(
        "--out", default="artifacts/tokenizer/fertility.json", help="JSON artifact output path"
    )
    fertility.add_argument("--report", help="markdown report output path (optional)")
    fertility.set_defaults(func=_cmd_tokenizer_fertility)

    train_cmd = top.add_parser("train", help="pretrain a model from a train config")
    train_cmd.add_argument("--config", required=True)
    train_cmd.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto", help="compute device"
    )
    train_cmd.add_argument(
        "--resume",
        metavar="{auto,PATH}",
        help="'auto' resumes from <out_dir>/latest.json when present (else fresh); "
        "or an explicit checkpoint file/dir. Omitting is refused when <out_dir> "
        "already holds a run (use --force-fresh to override).",
    )
    train_cmd.add_argument(
        "--force-fresh",
        action="store_true",
        help="start fresh even if <out_dir> already holds checkpoints "
        "(DESTRUCTIVE: overwrites and prunes them)",
    )
    train_cmd.add_argument("--out-dir", help="override the config's out_dir")
    train_cmd.set_defaults(func=_cmd_train)

    sft_sub = top.add_parser("sft", help="supervised fine-tuning stages").add_subparsers(
        dest="sft_command", required=True
    )
    sft_prepare = sft_sub.add_parser("prepare", help="rows -> filtered tokenized pairs")
    sft_prepare.add_argument("--data", required=True, help="JSON array of rows (SynSQL-style)")
    sft_prepare.add_argument("--db-root", required=True, help="dir of {db_id}/{db_id}.sqlite")
    sft_prepare.add_argument("--tokenizer", required=True, help="tokenizer.json path")
    sft_prepare.add_argument("--out", required=True, help="output pairs JSONL")
    sft_prepare.add_argument("--prompt-spec", default="bird-ddl-v1", choices=("bird-ddl-v1",))
    sft_prepare.add_argument("--context-limit", type=int, default=2048)
    sft_prepare.add_argument("--sample-rate", type=float, default=1.0)
    sft_prepare.add_argument("--seed", type=int, default=0)
    sft_prepare.add_argument(
        "--eos-token",
        default="<|end|>",
        help="sentinel closing each completion; the control arm's tokenizer uses its own",
    )
    sft_prepare.add_argument("--limit", type=int, default=None)
    sft_prepare.add_argument(
        "--link-targets",
        action="store_true",
        help="prefix completions with the gold-derived schema-linking block (v3 targets)",
    )
    sft_prepare.add_argument(
        "--reduce-overflow",
        action="store_true",
        help="retry oversized pairs with level-1/2 reduced DDL instead of filtering",
    )
    sft_prepare.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel pair-building processes (output stays byte-identical)",
    )
    sft_prepare.set_defaults(func=_cmd_sft_prepare)

    sft_train = sft_sub.add_parser("train", help="fine-tune on a prepared pair file")
    sft_train.add_argument("--model-dir", required=True, help="exported HF model directory")
    sft_train.add_argument("--pairs", required=True, help="prepared pairs JSONL")
    sft_train.add_argument("--out-dir", required=True)
    sft_train.add_argument("--tokenizer", required=True, help="tokenizer.json (pad/eos ids)")
    sft_train.add_argument(
        "--eos-token",
        default="<|end|>",
        help="the model's end token; also the pad fallback for foreign tokenizers",
    )
    sft_train.add_argument("--epochs", type=float, default=1.0)
    sft_train.add_argument("--lr", type=float, default=2e-5)
    sft_train.add_argument("--batch-size", type=int, default=4)
    sft_train.add_argument("--grad-accum", type=int, default=16)
    sft_train.add_argument("--seed", type=int, default=0)
    sft_train.add_argument(
        "--save-steps", type=int, default=None, help="periodic resumable checkpoints (spot safety)"
    )
    sft_train.add_argument("--resume", action="store_true", help="resume from last checkpoint")
    sft_train.add_argument("--run-name", default=None, help="W&B run name (needs WANDB_API_KEY)")
    sft_train.set_defaults(func=_cmd_sft_train)

    rl_sub = top.add_parser("rl", help="reinforcement learning stages").add_subparsers(
        dest="rl_command", required=True
    )
    grpo_cmd = rl_sub.add_parser("grpo", help="GRPO over the execution reward")
    grpo_cmd.add_argument("--model-dir", required=True, help="SFT checkpoint to start from")
    grpo_cmd.add_argument("--data", required=True, help="BIRD-format rows (train split)")
    grpo_cmd.add_argument("--db-root", required=True, help="<root>/<db_id>/<db_id>.sqlite")
    grpo_cmd.add_argument("--out-dir", required=True)
    grpo_cmd.add_argument("--tokenizer", required=True, help="tokenizer.json (prompt budgeting)")
    grpo_cmd.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="rollouts per prompt; group-relative advantage needs >= 2 (default: %(default)s)",
    )
    grpo_cmd.add_argument("--batch-size", type=int, default=4)
    grpo_cmd.add_argument("--grad-accum", type=int, default=4)
    grpo_cmd.add_argument("--max-completion-length", type=int, default=256)
    grpo_cmd.add_argument("--context-limit", type=int, default=2048)
    grpo_cmd.add_argument("--lr", type=float, default=1e-6)
    grpo_cmd.add_argument(
        "--beta",
        type=float,
        default=0.02,
        help="KL anchor to the SFT policy (default: %(default)s)",
    )
    grpo_cmd.add_argument("--temperature", type=float, default=1.0)
    grpo_cmd.add_argument("--max-steps", type=int, default=-1)
    grpo_cmd.add_argument("--limit", type=int, default=None, help="first N examples (pilot runs)")
    grpo_cmd.add_argument(
        "--unseen-of",
        type=float,
        default=None,
        metavar="RATE",
        help="train only on rows an SFT run at this --sample-rate skipped "
        "(the complement is exact: SFT selects by a per-row seeded hash)",
    )
    grpo_cmd.add_argument(
        "--unseen-seed", type=int, default=7, help="the SFT seed to invert (default: %(default)s)"
    )
    grpo_cmd.add_argument("--logging-steps", type=int, default=1)
    grpo_cmd.add_argument("--save-steps", type=int, default=100)
    grpo_cmd.add_argument("--run-name", default=None, help="W&B run name (needs WANDB_API_KEY)")
    grpo_cmd.add_argument(
        "--receipt",
        default=None,
        help="write the run receipt here as JSON (stdout also carries TRL logs)",
    )
    grpo_cmd.set_defaults(func=_cmd_rl_grpo)

    eval_sub = top.add_parser("eval", help="BIRD execution-accuracy evaluation").add_subparsers(
        dest="eval_command", required=True
    )
    score_cmd = eval_sub.add_parser("score", help="score predictions -> BIRD execution accuracy")
    score_cmd.add_argument(
        "--predictions",
        required=True,
        help="predictions jsonl/json: objects with predicted_sql (index defaults to position)",
    )
    score_cmd.add_argument("--subset", choices=["dev", "mini-dev"], default="dev")
    score_cmd.add_argument(
        "--eval-dir", default="data/eval", help="cached eval archives + extracted databases"
    )
    score_cmd.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="per-query wall-clock budget in seconds (default: %(default)s)",
    )
    score_cmd.add_argument(
        "--row-limit",
        type=int,
        default=DEFAULT_ROW_LIMIT,
        help="distinct-row cap per result set (default: %(default)s)",
    )
    score_cmd.add_argument(
        "--out", help="write the detailed per-example artifact here (gitignored)"
    )
    score_cmd.set_defaults(func=_cmd_eval_score)

    gold_cmd = eval_sub.add_parser(
        "gold-check", help="execute every gold query against itself (harness certificate)"
    )
    gold_cmd.add_argument("--subset", choices=["dev", "mini-dev"], default="mini-dev")
    gold_cmd.add_argument(
        "--eval-dir", default="data/eval", help="cached eval archives + extracted databases"
    )
    gold_cmd.add_argument(
        "--timeout",
        type=float,
        # Gold queries are trusted but occasionally slow; the official evaluator allows 30s.
        default=30.0,
        help="per-query wall-clock budget in seconds (default: %(default)s)",
    )
    gold_cmd.add_argument(
        "--row-limit",
        type=int,
        default=DEFAULT_ROW_LIMIT,
        help="distinct-row cap per result set (default: %(default)s)",
    )
    gold_cmd.add_argument(
        "--out",
        default=None,
        help="also write the certificate here (archived beside a run's other artefacts)",
    )
    gold_cmd.set_defaults(func=_cmd_eval_gold_check)

    generate_cmd = eval_sub.add_parser(
        "generate", help="render prompts and generate predictions from an exported model"
    )
    generate_cmd.add_argument("--model-dir", required=True, help="exported HF model directory")
    generate_cmd.add_argument("--subset", choices=["dev", "mini-dev"], default="dev")
    generate_cmd.add_argument(
        "--eval-dir", default="data/eval", help="cached eval archives + extracted databases"
    )
    generate_cmd.add_argument(
        "--out", required=True, help="predictions json path (a .meta.json sidecar is written too)"
    )
    generate_cmd.add_argument(
        "--refine",
        type=int,
        default=None,
        help="execution-feedback retry budget (omit for single-shot generation)",
    )
    generate_cmd.add_argument(
        "--constrain-block",
        action="store_true",
        help="mask the linking block to real schema identifiers (v3 checkpoints)",
    )
    generate_cmd.add_argument(
        "--two-pass",
        action="store_true",
        help="pass 1 predicts the tables, pass 2 re-asks with only those tables' DDL",
    )
    generate_cmd.add_argument(
        "--column-hints",
        action="store_true",
        help="with --two-pass: add BIRD column descriptions for the narrowed tables",
    )
    generate_cmd.add_argument(
        "--expand-fk",
        action="store_true",
        help="with --two-pass: also keep degree-1 FK neighbours of the named tables",
    )
    generate_cmd.add_argument(
        "--self-consistency",
        type=int,
        default=None,
        metavar="K",
        help="sample K completions and submit the answer most of them agree on",
    )
    generate_cmd.add_argument("--temperature", type=float, default=0.7)
    generate_cmd.add_argument(
        "--seed",
        type=int,
        default=0,
        help="sampling seed; each draw uses seed+draw, so a run is deterministic "
        "(vary it for error bars on a self-consistency number)",
    )
    generate_cmd.add_argument(
        "--block-budget",
        type=int,
        default=None,
        metavar="N",
        help="cap the linking block at N tokens, then force the SQL to start "
        "(targets the measured 'stuck in the block' failures)",
    )
    generate_cmd.add_argument(
        "--compact-overflow",
        action="store_true",
        help="re-render over-context prompts with a gold-blind compacted schema "
        "instead of recording an empty prediction (submission runs)",
    )
    generate_cmd.add_argument(
        "--context-limit",
        type=int,
        default=None,
        help="cap the prompt budget below the model's own (matched comparisons)",
    )
    generate_cmd.add_argument("--max-new-tokens", type=int, default=256)
    generate_cmd.add_argument("--batch-size", type=int, default=8)
    generate_cmd.add_argument("--device", default=None, help="cuda/mps/cpu (default: auto)")
    generate_cmd.add_argument(
        "--limit", type=int, default=None, help="generate for only the first K examples"
    )
    generate_cmd.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="probe budget in seconds when --refine is set (default: %(default)s)",
    )
    generate_cmd.add_argument(
        "--prompt-spec",
        default="bird-ddl-v1",
        choices=("bird-ddl-v1", "bird-ddl-v1-selectcue"),
        help="prompt format; -selectcue ends mid-statement (base models continue comments)",
    )
    generate_cmd.set_defaults(func=_cmd_eval_generate)

    predict_cmd = eval_sub.add_parser(
        "predict",
        help="predict on a held-out split (no gold) and write a BIRD submission file",
        description="Resumable prediction for a split whose answers we do not have, "
        "such as BIRD's test set. Writes predict_<split>.json in the format their "
        "evaluator reads, plus a log and a progress file so an interrupted run "
        "continues instead of restarting.",
    )
    predict_cmd.add_argument("--model-dir", required=True, help="exported HF model directory")
    predict_cmd.add_argument(
        "--examples", required=True, help="BIRD-format json (test.json); its SQL field may be empty"
    )
    predict_cmd.add_argument(
        "--db-root", required=True, help="databases root, laid out <root>/<db_id>/<db_id>.sqlite"
    )
    predict_cmd.add_argument("--out-dir", required=True, help="directory for every run artifact")
    predict_cmd.add_argument(
        "--split", default="test", help="names the output file, predict_<split>.json"
    )
    predict_cmd.add_argument(
        "--self-consistency",
        type=int,
        default=None,
        metavar="K",
        help="sample K completions and submit the answer most of them agree on",
    )
    predict_cmd.add_argument("--temperature", type=float, default=0.7)
    predict_cmd.add_argument("--seed", type=int, default=0)
    predict_cmd.add_argument(
        "--compact-overflow",
        action="store_true",
        help="re-render over-context prompts with a compacted schema instead of "
        "recording an empty prediction",
    )
    predict_cmd.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="examples per checkpoint to the progress file (default: %(default)s)",
    )
    predict_cmd.add_argument("--max-new-tokens", type=int, default=256)
    predict_cmd.add_argument("--batch-size", type=int, default=8)
    predict_cmd.add_argument("--device", default=None, help="cuda/mps/cpu (default: auto)")
    predict_cmd.add_argument("--context-limit", type=int, default=None)
    predict_cmd.add_argument("--limit", type=int, default=None, help="only the first K examples")
    predict_cmd.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="per-execution budget in seconds (default: %(default)s)",
    )
    predict_cmd.add_argument(
        "--prompt-spec", default="bird-ddl-v1", choices=("bird-ddl-v1", "bird-ddl-v1-selectcue")
    )
    predict_cmd.set_defaults(func=_cmd_eval_predict)

    diagnose_cmd = eval_sub.add_parser(
        "diagnose",
        help="verdict panel: difficulty/JOIN splits, valid rate, linking F1, block drift",
    )
    diagnose_cmd.add_argument("--predictions", required=True)
    diagnose_cmd.add_argument(
        "--score", required=True, help="the `eval score` output (EX of record, never recomputed)"
    )
    diagnose_cmd.add_argument("--subset", default="mini-dev", choices=("mini-dev", "dev"))
    diagnose_cmd.add_argument("--eval-dir", default="data/eval")
    diagnose_cmd.add_argument("--out", default=None, help="also write the panel to this path")
    diagnose_cmd.set_defaults(func=_cmd_eval_diagnose)

    export_cmd = top.add_parser(
        "export", help="convert a checkpoint to a HuggingFace LlamaForCausalLM directory"
    )
    export_cmd.add_argument(
        "--checkpoint", required=True, help="checkpoint .pt file or run directory (latest.json)"
    )
    export_cmd.add_argument("--out", required=True, help="output directory for the HF artifact")
    export_cmd.add_argument(
        "--tokenizer", help="tokenizer.json to bundle (also writes tokenizer_config.json)"
    )
    export_cmd.add_argument(
        "--dtype",
        choices=["fp32", "bf16"],
        default="fp32",
        help="weights dtype in the exported artifact (default: %(default)s)",
    )
    export_cmd.set_defaults(func=_cmd_export)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = cast(Handler, args.func)
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
