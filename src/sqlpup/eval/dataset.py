"""BIRD dev / Mini-Dev example loading and idempotent database extraction.

Reuses the cached-archive helpers from :mod:`sqlpup.data.eval_sets`
(:func:`fetch_archive`, :func:`read_member`). BIRD dev ships as one zip holding
``dev.json`` (the 1534 examples) and a nested ``dev_databases.zip`` (346MB of
per-database SQLite files); the nested zip is streamed to a temp file and
extracted, never read into memory. All 11 Mini-Dev databases live inside that
same bundle, so Mini-Dev needs no separate database download -- only its
``mini_dev_sqlite.json`` (obtained out of band; see :data:`MINI_DEV_URL`).

Examples are keyed by positional ``index`` because Mini-Dev's ``question_id`` is
not unique (498 distinct across 500 rows).
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from sqlpup.config import EvalSourceConfig
from sqlpup.data.eval_sets import fetch_archive, read_member

# BIRD dev: 2024-06-27 release linked from bird-bench.github.io.
BIRD_DEV_URL: Final = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
BIRD_DEV_JSON_MEMBER: Final = "dev_20240627/dev.json"
BIRD_DEV_DATABASES_MEMBER: Final = "dev_20240627/dev_databases.zip"
BIRD_DEV_EXAMPLES: Final = 1534
DEV_DATABASES_DIRNAME: Final = "dev_databases"

# Mini-Dev: the single needed member is ~278KB; the full minidev.zip is 801MB
# (HTTP-Range friendly) and deliberately NOT downloaded by this library.
MINI_DEV_FILENAME: Final = "mini_dev_sqlite.json"
MINI_DEV_URL: Final = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
MINI_DEV_MEMBER: Final = "minidev/MINIDEV/mini_dev_sqlite.json"

DIFFICULTIES: Final = ("simple", "moderate", "challenging")
SUBSETS: Final = ("dev", "mini-dev")

# Gold-check over the full extracted dev release: every gold query matches
# itself except these two, whose *gold* SQL is valid but exceeds
# the official 30s execution budget on commodity hardware -- 518 completes in
# ~36s and 701 in ~170s on an M-series laptop, so 518 may clear 30s on a faster
# eval host. They surface as ``gold_error`` in reports and cost at most
# 2/1534 = 0.13% EX; kept -- never silently excluded -- and footnoted wherever
# dev EX is stated.
KNOWN_SLOW_DEV_GOLDS: Final = (518, 701)

_COPY_CHUNK: Final = 1 << 20  # 1 MiB streamed-copy granularity

# A minimal EvalSourceConfig so the cached BIRD dev archive is located (and, if
# absent, fetched) exactly like every other eval set -- same URL-derived cache
# path used by sqlpup.data.eval_sets.
_BIRD_DEV_SOURCE = EvalSourceConfig(
    name="bird_dev",
    url=BIRD_DEV_URL,
    fields=("question", "SQL", "evidence"),
    archive_member=BIRD_DEV_JSON_MEMBER,
    expected_examples=BIRD_DEV_EXAMPLES,
)


@dataclass(frozen=True, slots=True)
class BirdExample:
    """One BIRD example, keyed by positional ``index`` (question_id is not unique)."""

    index: int
    question_id: str
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    difficulty: str


def _example_from(raw: Mapping[str, Any], index: int) -> BirdExample:
    return BirdExample(
        index=index,
        question_id=str(raw.get("question_id", index)),
        db_id=str(raw["db_id"]),
        question=str(raw.get("question", "")),
        evidence=str(raw.get("evidence") or ""),
        gold_sql=str(raw["SQL"]),
        difficulty=str(raw.get("difficulty", "")),
    )


def load_bird_file(path: Path | str) -> list[BirdExample]:
    """Examples from a raw BIRD-format JSON file (e.g. the train split).

    ``load_examples`` goes through the dev archive machinery; RL trains on
    BIRD-train, which is a plain file on disk. Both the ``SQL`` key BIRD-train
    uses and the ``sql`` key SynSQL-style rows use are accepted, so one loader
    covers every source we train on.
    """
    rows = _as_example_list(Path(path).read_bytes())
    examples: list[BirdExample] = []
    for index, raw in enumerate(rows):
        sql = raw.get("SQL") or raw.get("sql") or raw.get("query")
        if not sql:
            raise KeyError(f"row {index} has no SQL/sql/query field: {sorted(raw)}")
        examples.append(_example_from({**raw, "SQL": sql}, index))
    return examples


def load_prediction_examples(path: Path | str) -> list[BirdExample]:
    """Examples from a held-out split, where gold SQL is absent or empty.

    BIRD's ``test.json`` carries the same fields as ``dev.json`` except that
    ``SQL`` is an empty string, or missing outright. A missing key is an error in
    :func:`load_bird_file`, which feeds training and scoring; here it is the
    expected case, so gold is recorded as ``""``. Nothing on the prediction path
    reads it, and a held-out run must not be able to start reading it by
    accident.
    """
    rows = _as_example_list(Path(path).read_bytes())
    return [_example_from({**row, "SQL": row.get("SQL") or ""}, i) for i, row in enumerate(rows)]


def _as_example_list(raw: bytes) -> list[Mapping[str, Any]]:
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("expected a JSON list of BIRD examples")
    items: list[Mapping[str, Any]] = []
    for item in data:
        if not isinstance(item, Mapping):
            raise ValueError("expected JSON objects in the BIRD example list")
        items.append(item)
    return items


def _load_dev_json(eval_dir: Path) -> list[Mapping[str, Any]]:
    archive = fetch_archive(_BIRD_DEV_SOURCE, eval_dir)
    return _as_example_list(read_member(archive, BIRD_DEV_JSON_MEMBER))


def _load_mini_dev_json(eval_dir: Path) -> list[Mapping[str, Any]]:
    path = eval_dir / MINI_DEV_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Mini-Dev subset not found at {path}. Fetch member {MINI_DEV_MEMBER!r} "
            f"from {MINI_DEV_URL} (~278KB; the archive supports HTTP Range) and save it "
            f"there as {MINI_DEV_FILENAME}."
        )
    return _as_example_list(path.read_bytes())


def load_examples(eval_dir: Path, subset: str) -> list[BirdExample]:
    """Load the ``dev`` (1534) or ``mini-dev`` (500) examples, keyed by index."""
    if subset == "dev":
        rows = _load_dev_json(eval_dir)
    elif subset == "mini-dev":
        rows = _load_mini_dev_json(eval_dir)
    else:
        raise ValueError(f"unknown subset {subset!r}; expected one of {SUBSETS}")
    return [_example_from(row, i) for i, row in enumerate(rows)]


def db_path_under(db_root: Path | str, db_id: str) -> Path:
    """Path to a database's SQLite file under an explicit per-split databases root.

    Every BIRD split lays its databases out the same way, ``<root>/<db_id>/
    <db_id>.sqlite``, so pointing a run at ``test_databases`` needs a different
    root and nothing else.
    """
    return Path(db_root) / db_id / f"{db_id}.sqlite"


def resolve_db_path(eval_dir: Path, db_id: str) -> Path:
    """Path to a database's SQLite file under the extracted ``dev_databases`` tree."""
    return db_path_under(eval_dir / DEV_DATABASES_DIRNAME, db_id)


def _find_databases_member(outer: zipfile.ZipFile) -> str:
    names = outer.namelist()
    if BIRD_DEV_DATABASES_MEMBER in names:
        return BIRD_DEV_DATABASES_MEMBER
    for name in names:  # tolerate a differently-prefixed release
        if name.endswith("dev_databases.zip"):
            return name
    raise KeyError(f"{BIRD_DEV_DATABASES_MEMBER!r} not found in {outer.filename}")


def _extract_databases(archive: Path, eval_dir: Path) -> None:
    """Stream the nested dev_databases.zip out to a temp file, then extract it.

    Streaming the ~346MB member (never ``read()``-ing it whole) keeps memory flat
    regardless of the bundle size.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as outer:
        member = _find_databases_member(outer)
        fd, tmp_name = tempfile.mkstemp(dir=eval_dir, suffix=".dev_databases.zip")
        tmp_path = Path(tmp_name)
        try:
            with outer.open(member) as src, open(fd, "wb") as dst:
                shutil.copyfileobj(src, dst, _COPY_CHUNK)
            with zipfile.ZipFile(tmp_path) as inner:
                inner.extractall(eval_dir)
        finally:
            tmp_path.unlink(missing_ok=True)


def ensure_databases(eval_dir: Path, db_ids: Iterable[str]) -> Path:
    """Ensure every ``db_id``'s SQLite file exists; extract from the archive if not.

    Idempotent: if all requested databases are already present, returns without
    touching the archive (or the network). Returns the ``dev_databases`` root.
    """
    db_root = eval_dir / DEV_DATABASES_DIRNAME
    if all(resolve_db_path(eval_dir, db_id).exists() for db_id in db_ids):
        return db_root
    archive = fetch_archive(_BIRD_DEV_SOURCE, eval_dir)
    _extract_databases(archive, eval_dir)
    return db_root
