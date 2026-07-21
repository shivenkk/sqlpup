"""Byte-level BPE tokenizer training.

Byte-level means no unknown tokens ever; training on our own SQL-heavy mix
buys measurably better SQL compression than general-purpose tokenizers, which
is a real throughput and context-length win at small model scale.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import xxhash
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from sqlpup.config import TokenizerConfig

TOKENIZER_FILENAME = "tokenizer.json"

# cl100k_base's pre-tokenization regex (public, from tiktoken). It splits numbers
# into <=3-digit groups and isolates whitespace/newline runs, so long numerals
# and indentation do not blow up into many tokens -- the two code-relevant levers
# the "code" pre-tokenizer applies before byte-level. Only the split boundaries
# change; the merges are still learned from our own SQL-heavy corpus.
_CODE_PRETOKENIZER_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def build_pre_tokenizer(kind: str) -> pre_tokenizers.PreTokenizer:
    """Build the configured pre-tokenizer.

    ``byte_level`` is the vanilla GPT-2 ByteLevel split (default). ``code`` runs a
    cl100k-style regex split first, then byte-level with its own regex disabled.
    Byte-level is the last step in both, so a trained tokenizer round-trips any
    text exactly (byte-level guarantees no ``<unk>``).
    """
    if kind == "byte_level":
        return pre_tokenizers.ByteLevel(add_prefix_space=False)
    if kind == "code":
        return pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(_CODE_PRETOKENIZER_PATTERN), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
    raise ValueError(f"unknown pre_tokenizer {kind!r}; expected one of ('byte_level', 'code')")


def is_holdout(doc_id: str, holdout_mod: int) -> bool:
    """Whether ``doc_id`` falls in the fertility-eval holdout for ``holdout_mod``.

    Reserves a deterministic ``1/holdout_mod`` slice of documents -- those whose
    xxh64 id hash is divisible by ``holdout_mod`` (``xxh64(id) % N == 0``) -- so
    the tokenizer is trained with those documents excluded and the fertility
    comparison can measure on text the BPE trainer never saw. Pure and stable
    across runs and machines (xxh64 is seedless and platform-independent here).
    """
    return int(xxhash.xxh64(doc_id.encode()).intdigest()) % holdout_mod == 0


def train_bpe(config: TokenizerConfig, corpus: Iterable[str]) -> Path:
    """Train a byte-level BPE tokenizer on ``corpus``; return the saved path."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = build_pre_tokenizer(config.pre_tokenizer)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        special_tokens=list(config.special_tokens),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    missing = [t for t in config.special_tokens if tokenizer.token_to_id(t) is None]
    if missing:
        raise RuntimeError(f"special tokens missing from trained vocab: {missing}")

    config.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.out_dir / TOKENIZER_FILENAME
    tokenizer.save(str(out_path))
    return out_path


def load_tokenizer(path: Path) -> Tokenizer:
    """Load a tokenizer saved by :func:`train_bpe`."""
    return Tokenizer.from_file(str(path))
