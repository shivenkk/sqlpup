"""Execution-guided self-consistency: keep the answer most samples agree on.

Measured motivation: majority voting across four existing
prediction sets scored 15.2% where the best single member scored 12.6%, and
the oracle union over those members reached 20.4%. The model's correct answer
is frequently *reachable* but not reliably produced, which is exactly the
regime voting exploits.

Candidates are grouped by the result set they produce rather than by their
text, because two differently-written queries returning the same rows are the
same answer. Grouping uses the evaluation sandbox's own fingerprint op, so a
candidate that cannot execute simply does not get a vote -- and if none of
them execute, candidate 0 is submitted anyway (an empty answer scores zero,
so there is nothing to gain by abstaining).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol


class _Fingerprinter(Protocol):
    def fingerprint(self, sql: str, db_path: Path | str) -> tuple[bool, str]: ...


def majority_vote(
    candidates: Sequence[str], db_path: Path | str, scorer: _Fingerprinter
) -> tuple[str, dict[str, Any]]:
    """The most-agreed-upon candidate, plus vote statistics.

    Ties go to the earliest candidate, which by convention is the greedy
    sample -- so voting can only overrule greedy decoding when strictly more
    samples agree on something else.
    """
    if not candidates:
        return "", {"valid": 0, "votes": 0, "distinct_answers": 0, "candidates": 0}

    groups: dict[str, list[int]] = {}
    for index, sql in enumerate(candidates):
        ok, digest = scorer.fingerprint(sql, db_path)
        if ok:
            groups.setdefault(digest, []).append(index)

    # An empty result set is a wrong answer on this benchmark -- measured: 0 of
    # 500 Mini-Dev gold queries return zero rows. It is also the answer wrong
    # queries agree on most readily (a bad filter, a bad join), so empty
    # clusters out-vote correct-but-distinct ones: they won 20% of votes, all
    # of them wrong. Demoting empties to a last resort measured +1.2pt.
    # Disclosed as part of the voting policy, not folded into the model.
    non_empty = {d: m for d, m in groups.items() if not d.startswith("0:")}
    empty_demoted = bool(non_empty) and len(non_empty) < len(groups)
    if non_empty:
        groups = non_empty

    stats: dict[str, Any] = {
        "candidates": len(candidates),
        "empty_demoted": empty_demoted,
        "valid": sum(len(members) for members in groups.values()),
        "distinct_answers": len(groups),
    }
    if not groups:
        stats["votes"] = 0
        return candidates[0], stats

    # most votes, then earliest occurrence -- both deterministic
    best = max(groups.values(), key=lambda members: (len(members), -members[0]))
    stats["votes"] = len(best)
    return candidates[best[0]], stats
