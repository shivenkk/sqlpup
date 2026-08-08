"""Selecting the examples SFT never saw.

The first GRPO pilot opened at 0.769 mean reward on BIRD-train, which v3's SFT
oversampled ten times: RL was sharpening answers the model had memorised, so
57% of groups carried zero advantage and the policy did not move.

SFT's sampling is a per-row seeded hash (``sha256(f"{seed}:{index}")``, kept
when below the rate), which makes its complement *exactly* computable rather
than approximately. This selects that complement, so RL trains where the model
is actually weak. Correctness here is load-bearing: an off-by-one in the index
would silently hand RL the memorised half again and reproduce the same null
result at full cost.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sqlpup.rl.unseen import iter_unseen_rows


def _corpus(tmp_path: Path, n: int = 500) -> Path:
    path = tmp_path / "data.json"
    rows = [
        {"db_id": f"db{i % 7}", "question": f"q{i}", "sql": f"SELECT {i}", "external_knowledge": ""}
        for i in range(n)
    ]
    path.write_text(json.dumps(rows))
    return path


def test_it_selects_exactly_what_sft_skipped(tmp_path: Path) -> None:
    """The complement must be the *exact* inverse of the SFT selection, not a
    fresh sample that happens to be about the right size."""
    from sqlpup.sft.prepare import _keep

    path = _corpus(tmp_path)
    picked = list(iter_unseen_rows(path, sample_rate=0.4, seed=7))
    got = {row.question for row in picked}
    expected = {f"q{i}" for i in range(500) if not _keep(i, 0.4, 7)}
    assert got == expected


def test_the_selection_is_disjoint_from_what_sft_trained_on(tmp_path: Path) -> None:
    from sqlpup.sft.prepare import _keep

    path = _corpus(tmp_path)
    unseen = {r.question for r in iter_unseen_rows(path, sample_rate=0.4, seed=7)}
    seen = {f"q{i}" for i in range(500) if _keep(i, 0.4, 7)}
    assert not (unseen & seen)
    assert len(unseen) + len(seen) == 500


def test_rows_arrive_as_bird_examples_the_rollout_builder_accepts(tmp_path: Path) -> None:
    path = _corpus(tmp_path, n=50)
    row = next(iter(iter_unseen_rows(path, sample_rate=0.4, seed=7)))
    assert row.db_id.startswith("db")
    assert row.gold_sql.startswith("SELECT")
    assert row.evidence == ""


def test_synsql_calls_its_evidence_field_something_else(tmp_path: Path) -> None:
    """SynSQL says ``external_knowledge`` where BIRD says ``evidence``. Dropping
    it would train RL on prompts missing the hint the evaluation prompt carries."""
    path = tmp_path / "d.json"
    path.write_text(
        json.dumps(
            [{"db_id": "a", "question": "q", "sql": "SELECT 1", "external_knowledge": "hint"}]
        )
    )
    rows = list(iter_unseen_rows(path, sample_rate=0.0, seed=7))
    assert rows[0].evidence == "hint"


def test_limit_stops_the_scan_early(tmp_path: Path) -> None:
    """The real corpus is multi-gigabyte; a pilot wanting 300 rows must not read
    all of it."""
    path = _corpus(tmp_path, n=5000)
    assert len(list(iter_unseen_rows(path, sample_rate=0.4, seed=7, limit=25))) == 25


def test_a_full_rate_leaves_nothing_unseen(tmp_path: Path) -> None:
    """sample_rate 1.0 means SFT took everything, so the RL pool is empty --
    and must come back empty rather than silently yielding the whole corpus."""
    path = _corpus(tmp_path, n=100)
    assert list(iter_unseen_rows(path, sample_rate=1.0, seed=7)) == []


@pytest.mark.parametrize("rate", [-0.1, 1.1])
def test_a_nonsense_rate_is_rejected(tmp_path: Path, rate: float) -> None:
    path = _corpus(tmp_path, n=10)
    with pytest.raises(ValueError):
        list(iter_unseen_rows(path, sample_rate=rate, seed=7))
