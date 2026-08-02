"""One SFT example: prompt tokens masked, completion tokens supervised.

Masking uses the prefix-length method -- labels are ``IGNORE_INDEX`` for the
first ``len(encode(prompt))`` positions -- which is only correct if the
tokenizer never merges a token across the prompt/completion seam (byte-level
BPE happily merges ``"x" + "y"`` into one token when they are adjacent bytes).
Rather than trust that property, :func:`build_pair` verifies it per pair and
raises :class:`BoundaryError` on violation, so a silent mis-masking bug is
structurally impossible: the prompt's own rendering (ending in a newline cue)
makes the seam whitespace-stable for our code-tuned pre-tokenizer, and the
check proves it example by example.

Overflow (``total_tokens > context_limit``) is *flagged*, never truncated:
cutting DDL mid-table teaches the model malformed schemas, so the caller
filters flagged pairs and reports the count instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol

from sqlpup.eval.prompts import BIRD_DDL_V1, PromptSpec

# HF/torch cross-entropy convention for "no loss at this position".
IGNORE_INDEX: Final = -100

DEFAULT_CONTEXT_LIMIT: Final = 2048


class BoundaryError(ValueError):
    """The tokenizer merged across the prompt/completion seam; masking unsafe."""


class _Encoding(Protocol):
    ids: list[int]


class TokenizerLike(Protocol):
    """The one sliver of ``tokenizers.Tokenizer`` this module needs."""

    def encode(self, sequence: str) -> _Encoding: ...


@dataclass(frozen=True, slots=True)
class SFTPair:
    """A tokenized training example with loss-mask already applied."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]  # IGNORE_INDEX over the prompt, token id over the completion
    prompt_tokens: int
    total_tokens: int
    overflow: bool


def build_pair(
    tokenizer: TokenizerLike,
    *,
    ddl: str,
    question: str,
    evidence: str,
    sql: str,
    eos_id: int,
    spec: PromptSpec = BIRD_DDL_V1,
    context_limit: int = DEFAULT_CONTEXT_LIMIT,
    _prompt_override: str | None = None,
    _completion_override: str | None = None,
) -> SFTPair:
    """Render prompt + completion, tokenize, verify the seam, mask the prompt.

    The completion is ``sql.strip() + ";"`` followed by ``eos_id`` -- the model
    learns to terminate statements (extraction cuts at ``;``) and to close the
    document the way pretraining did. The two ``_*_override`` parameters exist
    only so tests can force a seam violation; production callers never pass
    them.
    """
    prompt = (
        _prompt_override
        if _prompt_override is not None
        else spec.render(ddl, question=question, evidence=evidence)
    )
    completion = _completion_override if _completion_override is not None else f"{sql.strip()};"

    prompt_ids = list(tokenizer.encode(prompt).ids)
    full_ids = list(tokenizer.encode(prompt + completion).ids)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise BoundaryError(
            "tokenizer merged across the prompt/completion seam; prefix-length "
            "masking would supervise the wrong positions for this example"
        )

    input_ids = (*full_ids, eos_id)
    prompt_tokens = len(prompt_ids)
    labels = (IGNORE_INDEX,) * prompt_tokens + input_ids[prompt_tokens:]
    total = len(input_ids)
    return SFTPair(
        input_ids=input_ids,
        labels=labels,
        prompt_tokens=prompt_tokens,
        total_tokens=total,
        overflow=total > context_limit,
    )
