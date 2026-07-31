"""Checkpoint-backed greedy generation over an exported HF directory.

The one :class:`~sqlpup.eval.generate.SQLGenerator` implementation that touches
a real model. It consumes the directory ``sqlpup export`` writes (HF
``LlamaForCausalLM`` + ``PreTrainedTokenizerFast``) and decodes greedily --
deterministic by construction, so an eval report is reproducible from
(checkpoint, prompt spec, this adapter) alone. torch/transformers are imported
lazily inside ``__init__`` and this module is deliberately NOT re-exported from
``sqlpup.eval``: importing the package stays torch-free (the GRPO reward
depends on that), while importing this module is an explicit opt-in to the
``train`` extra.

Context discipline (measured on BIRD dev under the sqlpup tokenizer: 3.8% of
full-DDL prompts exceed the model's 2048 positions, max 2139): prompts are
length-sorted and batched with like-sized neighbours, each batch's generation
budget shrinks to fit the context, and prompts that cannot fit at all yield an
empty completion -- scored conservatively as a miss -- rather than a crash or
out-of-range positions. Batches are left-padded (decoder-only models continue
from the sequence *end*; right-padding would have them continue from pad
tokens) and completions are sliced past the shared padded prompt length, so
extraction sees only new text.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

from sqlpup.eval.constrain import SQLConstraint

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps runtime torch-free
    import torch

# BIRD gold SQL tops out well under this many 32k-vocab tokens; the budget only
# bounds runaway rambling (extraction cuts at the first ';' anyway).
DEFAULT_MAX_NEW_TOKENS: Final = 256
DEFAULT_BATCH_SIZE: Final = 8
# Consecutive empty completions tolerated before declaring silent failure.
# Above one full batch, so a genuinely hard chunk cannot trip it.
EMPTY_STREAK_LIMIT: Final = 25


def _fit_new_tokens(requested: int, *, prompt_length: int, context_limit: int) -> int:
    """The generation budget that keeps ``prompt + new`` within the context."""
    return max(0, min(requested, context_limit - prompt_length))


def _plan_batches(
    lengths: Sequence[int], *, batch_size: int, context_limit: int
) -> tuple[list[int], list[list[int]]]:
    """Split prompt indices into (overflow, length-sorted batches).

    Overflow prompts (``length >= context_limit``) cannot be generated for at
    all. The rest are batched with like-sized neighbours so one long prompt
    neither blanks nor throttles a short chunk-mate's budget (with left
    padding, a batch's effective prompt length is its longest member's).
    """
    if batch_size < 1:
        raise ValueError(f"batch size must be >= 1, got {batch_size}")
    overflow = [i for i, length in enumerate(lengths) if length >= context_limit]
    rest = sorted(
        (i for i, length in enumerate(lengths) if length < context_limit),
        key=lambda i: lengths[i],
    )
    batches = [rest[start : start + batch_size] for start in range(0, len(rest), batch_size)]
    return overflow, batches


class _ConstraintProcessor:
    """Per-row logits masking for batched constrained generation.

    Duck-typed ``transformers`` logits processor: sets disallowed vocabulary
    to ``-inf`` row by row, slicing each row's *generated* suffix past the
    shared left-padded prompt before consulting its constraint.
    """

    def __init__(self, constraints: Sequence[SQLConstraint], *, padded_prompt_len: int) -> None:
        self._constraints = list(constraints)
        self._padded_prompt_len = padded_prompt_len

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        import torch

        vocab_size = int(scores.shape[-1])
        for row, constraint in enumerate(self._constraints):
            generated = input_ids[row, self._padded_prompt_len :].tolist()
            mask = torch.tensor(
                list(constraint.allowed_token_mask(generated, vocab_size)),
                dtype=torch.bool,
                device=scores.device,
            )
            # A mask admitting nothing would send every logit to -inf and the
            # rest of the sequence to garbage (measured: EX 4.4%, validity
            # *below* unconstrained). Decode this step unconstrained instead.
            if not bool(mask.any()):
                continue
            scores[row] = scores[row].masked_fill(~mask, float("-inf"))
        return scores


class EmptyCompletionStreakError(RuntimeError):
    """Raised when the model emits nothing for many consecutive prompts."""


def guard_empty_streak(streak: int, completion: str, *, limit: int) -> int:
    """Running count of consecutive empty completions; raises past ``limit``.

    A working model almost never returns nothing, so a long run of empties
    means something upstream broke silently -- half-loaded weights, a schema
    that narrowed to nothing, a wedged constraint. On a 1534-query submission
    run that is ten hours wasted before a post-hoc check would notice, so the
    run stops at the first sign instead.
    """
    if completion.strip():
        return 0
    streak += 1
    if streak > limit:
        raise EmptyCompletionStreakError(
            f"{streak} consecutive empty completions -- generation is failing silently"
        )
    return streak


def _pick_device() -> str:
    """cuda > mps > cpu, by availability."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # pragma: no cover - Apple-only branch
        return "mps"
    return "cpu"


class HFGreedyGenerator:
    """Batched greedy :class:`SQLGenerator` over an exported model directory."""

    def __init__(
        self,
        model_dir: Path | str,
        *,
        device: str | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        constraint: SQLConstraint | None = None,
        context_limit: int | None = None,
    ) -> None:
        # Checked before any model loading: never silently decode unconstrained.
        if constraint is not None:
            raise NotImplementedError(
                "SQL-constrained decoding is not implemented in this harness; "
                "it belongs to the batched inference engine."
            )
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._max_new_tokens = max_new_tokens
        self._batch_size = batch_size
        # Explicit override for matched comparisons: a control model with a
        # larger native window must be held to the same schema budget we had.
        self._context_override = context_limit
        self._device = device or _pick_device()

        dtype = torch.float32 if self._device in ("cpu", "mps") else "auto"
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=dtype)
        # transformers wraps Module.to in a stub that mistypes the first argument
        # as the bound instance; this is the standard Module.to(device) call.
        self._model = model.to(torch.device(self._device)).eval()  # type: ignore[arg-type]

        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        # Decoder-only batching: continue from the sequence end, not from pads.
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            if self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            else:
                # A bare tokenizer.json (no tokenizer_config.json) declares no
                # special tokens at all; the model config still knows its eos.
                # Without this fallback an SFT output dir missing that file
                # cannot pad a batch.
                self._tokenizer.pad_token_id = int(self._model.config.eos_token_id)

    def _context_limit(self) -> int:
        if self._context_override is not None:
            return self._context_override
        return int(self._model.config.max_position_embeddings)

    @property
    def context_limit(self) -> int:
        """Positions available to a prompt (public seam for the schema fallback)."""
        return self._context_limit()

    def count_tokens(self, text: str) -> int:
        """Length of *text* under this run's tokenizer, in tokens."""
        return len(self._tokenizer(text)["input_ids"])

    def _generate_batch(
        self,
        batch_prompts: list[str],
        constraints: Sequence[SQLConstraint] | None = None,
        temperature: float | None = None,
    ) -> list[str]:
        import torch
        from transformers import LogitsProcessorList

        encoded = self._tokenizer(batch_prompts, return_tensors="pt", padding=True)
        encoded = {key: tensor.to(self._device) for key, tensor in encoded.items()}
        prompt_length = int(encoded["input_ids"].shape[1])
        context_limit = self._context_limit()
        budget = _fit_new_tokens(
            self._max_new_tokens, prompt_length=prompt_length, context_limit=context_limit
        )
        if budget == 0:  # defensive: _plan_batches keeps these out already
            return [""] * len(batch_prompts)
        processors = (
            LogitsProcessorList(
                [_ConstraintProcessor(constraints, padded_prompt_len=prompt_length)]
            )
            if constraints is not None
            else None
        )
        with torch.no_grad():
            generated: torch.Tensor = self._model.generate(
                **encoded,
                do_sample=temperature is not None,
                temperature=temperature,
                max_new_tokens=budget,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                logits_processor=processors,
            )
        completions: list[str] = self._tokenizer.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True
        )
        return completions

    def generate(self, prompts: Sequence[str]) -> list[str]:
        """One raw completion per prompt, in order (greedy, deterministic).

        Prompts that exceed the model's context yield ``""`` (conservative
        miss); everything else is generated in length-sorted batches and
        restored to input order.
        """
        lengths = [len(self._tokenizer(prompt)["input_ids"]) for prompt in prompts]
        context_limit = self._context_limit()
        overflow, batches = _plan_batches(
            lengths, batch_size=self._batch_size, context_limit=context_limit
        )
        out: list[str] = [""] * len(prompts)
        streak = 0
        for batch_indices in batches:
            completions = self._generate_batch([prompts[i] for i in batch_indices])
            for index, completion in zip(batch_indices, completions, strict=True):
                out[index] = completion
                streak = guard_empty_streak(streak, completion, limit=EMPTY_STREAK_LIMIT)
        # overflow entries stay "" -- recorded, never crashed on
        del overflow
        return out

    def generate_samples(
        self, prompts: Sequence[str], k: int, temperature: float, seed: int
    ) -> list[list[str]]:
        """``k`` sampled completions per prompt (candidate 0 is greedy).

        Keeping the greedy sample first makes voting a strict improvement
        test: ties go to candidate 0, so sampling can only overrule greedy
        decoding when strictly more samples agree on something else.
        """
        import torch

        greedy = self.generate(prompts)
        out: list[list[str]] = [[g] for g in greedy]
        for draw in range(k - 1):
            torch.manual_seed(seed + draw)
            sampled = self._generate_all(prompts, temperature=temperature)
            for row, completion in enumerate(sampled):
                out[row].append(completion)
        return out

    def _generate_all(self, prompts: Sequence[str], *, temperature: float) -> list[str]:
        """``generate`` with sampling on, same batching and overflow rules."""
        lengths = [len(self._tokenizer(prompt)["input_ids"]) for prompt in prompts]
        context_limit = self._context_limit()
        overflow, batches = _plan_batches(
            lengths, batch_size=self._batch_size, context_limit=context_limit
        )
        out: list[str] = [""] * len(prompts)
        for batch_indices in batches:
            completions = self._generate_batch(
                [prompts[i] for i in batch_indices], temperature=temperature
            )
            for index, completion in zip(batch_indices, completions, strict=True):
                out[index] = completion
        del overflow
        return out

    def generate_block_constrained(
        self, prompts: Sequence[str], db_paths: Sequence[Path]
    ) -> list[str]:
        """Generate with each prompt's linking block masked to its real schema.

        Constraints see only the generated suffix (``prompt_len=0`` relative
        to the padded prompt boundary the processor slices at), so one shared
        schema cache serves repeated databases.
        """
        from sqlpup.eval.blockmask import LinkingBlockConstraint
        from sqlpup.sft.linking import schema_identifiers

        schema_cache: dict[str, dict[str, list[str]]] = {}
        constraints: list[SQLConstraint] = []
        for db_path in db_paths:
            key = str(db_path)
            if key not in schema_cache:
                schema_cache[key] = schema_identifiers(db_path)
            constraints.append(
                LinkingBlockConstraint(schema_cache[key], self._tokenizer, prompt_len=0)
            )
        return self.generate_constrained(prompts, constraints)

    def generate_block_budget(self, prompts: Sequence[str], budget_tokens: int) -> list[str]:
        """Generate, capping how much of each completion the linking block may eat.

        No database is consulted: the constraint is purely positional, so one
        constraint object per row is all it needs.
        """
        from sqlpup.eval.blockbudget import BlockBudgetConstraint

        constraints: list[SQLConstraint] = [
            BlockBudgetConstraint(self._tokenizer, prompt_len=0, budget_tokens=budget_tokens)
            for _ in prompts
        ]
        return self.generate_constrained(prompts, constraints)

    def generate_constrained(
        self, prompts: Sequence[str], constraints: Sequence[SQLConstraint]
    ) -> list[str]:
        """``generate`` with one :class:`SQLConstraint` mask applied per prompt.

        Constraints follow their prompts through length-sorted batching, so
        one example's schema never masks a batch-mate. Overflow prompts yield
        ``""`` exactly as in :meth:`generate`.
        """
        if len(prompts) != len(constraints):
            raise ValueError(f"{len(prompts)} prompts but {len(constraints)} constraints")
        lengths = [len(self._tokenizer(prompt)["input_ids"]) for prompt in prompts]
        context_limit = self._context_limit()
        overflow, batches = _plan_batches(
            lengths, batch_size=self._batch_size, context_limit=context_limit
        )
        out: list[str] = [""] * len(prompts)
        for batch_indices in batches:
            completions = self._generate_batch(
                [prompts[i] for i in batch_indices],
                constraints=[constraints[i] for i in batch_indices],
            )
            for index, completion in zip(batch_indices, completions, strict=True):
                out[index] = completion
        del overflow
        return out
