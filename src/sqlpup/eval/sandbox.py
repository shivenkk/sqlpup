"""Killable-worker-process sandbox for executing model-generated SQL.

Why a separate process, not a signal alarm
-------------------------------------------
A predicted query can loop forever inside SQLite's C engine (e.g. a recursive
CTE), where no Python-level ``signal.alarm``/``SIGALRM`` handler can interrupt
it -- the interpreter only runs Python signal handlers between bytecode ops, and
a blocking ``sqlite3_step`` is off in C. A prior attempt at this harness used a
signal-based timeout and died on macOS ``EINTR`` (interrupted system call): the
alarm interrupted blocking syscalls elsewhere and raised ``InterruptedError``
where correct results were expected. So instead:

* Each query runs in a **single long-lived worker process** the parent talks to
  over a ``multiprocessing`` ``Pipe``. The parent waits for a reply with a
  deadline; if the deadline passes it ``terminate()``s (escalating to
  ``kill()``) the worker -- an OS-level stop that works even while the worker is
  stuck in C -- and respawns a fresh one. This genuinely bounds wall-clock.
* PEP 475 already makes CPython retry ``EINTR`` inside ``poll``/``join``/``recv``
  internally, but we *additionally* treat any ``InterruptedError`` that reaches
  us while waiting on the worker as retry-not-failure (recompute the remaining
  time and wait again) -- never as a query result. That is the exact class of
  bug that killed the previous attempt.
* Exactly **one** worker is alive at a time (no pool fan-out); the batch scorer
  drives it sequentially. The worker is spawned with the ``"spawn"`` start
  method and imports only ``sqlite3`` + stdlib -- never torch -- so it stays
  light and identical across platforms.

Safety and memory
------------------
The worker opens every database **read-only** (``file:...?mode=ro``), sets
``PRAGMA query_only`` and an authorizer that denies ``ATTACH``/``DETACH`` and
every write action -- ``mode=ro`` blocks writes to the target file, but only the
authorizer stops ``ATTACH`` from creating a *new* file elsewhere. Result sets are
fetched in chunks (never an unbounded ``fetchall``) into a ``set`` capped at
``row_limit`` distinct rows, so a runaway query cannot exhaust memory. Gold and
prediction sets live in the killable worker, so the parent process holds almost
nothing.
"""

from __future__ import annotations

import contextlib
import hashlib
import multiprocessing as mp
import sqlite3
import time
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Final

from sqlpup.eval.results import MatchCategory

# Fetch granularity: chunked so a huge result set never lands in memory at once.
_FETCH_CHUNK: Final = 10_000

# Authorizer action codes that mutate the database or attach/detach another one.
# Reads (SQLITE_READ/SELECT/FUNCTION/RECURSIVE/...) are absent, so SELECTs, CTEs
# and subqueries run untouched while any write or file-creating action is denied.
_DENIED_ACTIONS: Final[frozenset[int]] = frozenset(
    {
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_REINDEX,
        sqlite3.SQLITE_ANALYZE,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_INDEX,
        sqlite3.SQLITE_CREATE_TEMP_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
        sqlite3.SQLITE_CREATE_TEMP_VIEW,
        sqlite3.SQLITE_CREATE_TRIGGER,
        sqlite3.SQLITE_CREATE_VIEW,
        sqlite3.SQLITE_CREATE_VTABLE,
        sqlite3.SQLITE_DROP_INDEX,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_INDEX,
        sqlite3.SQLITE_DROP_TEMP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_TRIGGER,
        sqlite3.SQLITE_DROP_TEMP_VIEW,
        sqlite3.SQLITE_DROP_TRIGGER,
        sqlite3.SQLITE_DROP_VIEW,
        sqlite3.SQLITE_DROP_VTABLE,
    }
)

Row = tuple[object, ...]

# Cap on error text shipped across the pipe (arbitrary model SQL -> arbitrary text).
_ERROR_TEXT_LIMIT: Final = 300

# Parent-side message for a probe the worker had to be killed over.
PROBE_TIMEOUT_MESSAGE: Final = "timeout: query exceeded the per-query budget"


def _error_text(exc: BaseException) -> str:
    """``Type: message`` truncated -- the repair-prompt signal, never a traceback."""
    return f"{type(exc).__name__}: {exc}"[:_ERROR_TEXT_LIMIT]


# --- child (worker) side -----------------------------------------------------


def _authorizer(
    action: int,
    arg1: str | None,
    arg2: str | None,
    db_name: str | None,
    source: str | None,
) -> int:
    """Deny writes and ATTACH/DETACH; allow everything read-only."""
    return sqlite3.SQLITE_DENY if action in _DENIED_ACTIONS else sqlite3.SQLITE_OK


def _open_readonly(db_path: str) -> sqlite3.Connection:
    """Open ``db_path`` read-only with query-only + authorizer guards."""
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.execute("PRAGMA query_only = ON")
    con.set_authorizer(_authorizer)
    return con


def _build_capped_set(cursor: sqlite3.Cursor, row_limit: int) -> tuple[set[Row], bool]:
    """Fetch in chunks into a distinct-row set, stopping if it exceeds ``row_limit``.

    Returns ``(rows, exceeded)``; when ``exceeded`` is true the set holds
    ``row_limit + 1`` rows and fetching stopped early.
    """
    rows: set[Row] = set()
    while True:
        chunk = cursor.fetchmany(_FETCH_CHUNK)
        if not chunk:
            return rows, False
        for row in chunk:
            rows.add(tuple(row))
            if len(rows) > row_limit:
                return rows, True


class _WorkerState:
    """Per-process cache of read-only connections plus the pending gold set.

    Gold and prediction of one example arrive as two requests (one per-query
    timeout each); gold's distinct-row set is held here between them so the
    result rows never cross the pipe. It is consumed (cleared) by the prediction
    request, and lost entirely if the worker is killed on a timeout -- which is
    fine, since after a gold timeout no prediction request is sent.
    """

    def __init__(self) -> None:
        self._conns: dict[str, sqlite3.Connection] = {}
        self._gold: set[Row] | None = None

    def _conn(self, db_path: str) -> sqlite3.Connection:
        con = self._conns.get(db_path)
        if con is None:
            con = _open_readonly(db_path)
            self._conns[db_path] = con
        return con

    def run_gold(self, db_path: str, sql: str, row_limit: int) -> tuple[str, str]:
        self._gold = None
        try:
            cursor = self._conn(db_path).execute(sql)
            gold, exceeded = _build_capped_set(cursor, row_limit)
        except Exception:  # any failure is reported to the parent, never raised
            return ("gold", "error")
        if exceeded:
            return ("gold", "row_limit")
        self._gold = gold
        return ("gold", "ok")

    def run_pred(self, db_path: str, sql: str, row_limit: int) -> tuple[str, bool, str]:
        gold = self._gold
        self._gold = None  # consume: one prediction per prepared gold
        if gold is None:  # defensive: should never happen given the parent protocol
            return ("pred", False, MatchCategory.GOLD_ERROR.value)
        try:
            cursor = self._conn(db_path).execute(sql)
            pred, exceeded = _build_capped_set(cursor, row_limit)
        except Exception:  # predicted SQL may be arbitrary/broken -> non-match
            return ("pred", False, MatchCategory.EXECUTION_ERROR.value)
        if exceeded:
            return ("pred", False, MatchCategory.ROW_LIMIT.value)
        if not gold and not pred:
            return ("pred", True, MatchCategory.EMPTY.value)
        return ("pred", pred == gold, MatchCategory.OK.value)

    def run_probe(self, db_path: str, sql: str, row_limit: int) -> tuple[str, bool, str]:
        """Execute ``sql`` alone; (ok, error_text) is the refine-loop signal.

        Exceeding ``row_limit`` still counts as ok -- the query *executes*;
        the cap exists only to bound worker memory while draining the cursor.
        """
        try:
            cursor = self._conn(db_path).execute(sql)
            _build_capped_set(cursor, row_limit)
        except Exception as exc:  # arbitrary model SQL -> report, never raise
            return ("probe", False, _error_text(exc))
        return ("probe", True, "")

    def run_fingerprint(self, db_path: str, sql: str, row_limit: int) -> tuple[str, bool, str]:
        """Execute ``sql`` and digest its result set.

        Voting needs to know whether two candidates *return the same rows*,
        which text comparison cannot answer. The digest is order-independent
        (the rows are a set) and stable across processes.
        """
        try:
            cursor = self._conn(db_path).execute(sql)
            rows, exceeded = _build_capped_set(cursor, row_limit)
        except Exception:
            return ("fingerprint", False, "")
        if exceeded:
            return ("fingerprint", False, "")
        digest = hashlib.sha256(
            "\x1f".join(sorted(repr(row) for row in rows)).encode("utf-8", "replace")
        ).hexdigest()
        # Row count is prefixed so callers can recognise an empty answer
        # without a second execution (voting demotes empties).
        return ("fingerprint", True, f"{len(rows)}:{digest}")

    def close(self) -> None:
        for con in self._conns.values():
            with contextlib.suppress(sqlite3.Error):
                con.close()
        self._conns.clear()
        self._gold = None


def _worker_main(conn: Connection) -> None:
    """Worker entry point: serve query requests until the pipe closes."""
    state = _WorkerState()
    try:
        while True:
            try:
                msg = conn.recv()
            except (EOFError, KeyboardInterrupt):
                break
            if msg is None:  # graceful shutdown sentinel
                break
            kind = msg[0]
            reply: tuple[object, ...]
            if kind == "gold":
                reply = state.run_gold(msg[1], msg[2], msg[3])
            elif kind == "pred":
                reply = state.run_pred(msg[1], msg[2], msg[3])
            elif kind == "probe":
                reply = state.run_probe(msg[1], msg[2], msg[3])
            elif kind == "fingerprint":
                reply = state.run_fingerprint(msg[1], msg[2], msg[3])
            else:  # pragma: no cover - defensive
                reply = ("error", f"unknown request {kind!r}")
            try:
                conn.send(reply)
            except (BrokenPipeError, OSError):
                break
    finally:
        state.close()
        conn.close()


# --- parent side -------------------------------------------------------------

_TIMEOUT: Final = object()  # sentinel: worker did not reply within the deadline
_CRASH: Final = object()  # sentinel: worker died/pipe broke unexpectedly


class SqliteWorker:
    """Parent-side handle to one killable SQLite worker process.

    Not thread-safe: a single worker is driven sequentially by one scorer.
    """

    def __init__(self) -> None:
        # A dedicated spawn context: identical across platforms and import-light
        # in the child (never inherits a fork of the parent's imported torch).
        self._ctx = mp.get_context("spawn")
        self._proc: BaseProcess | None = None
        self._conn: Connection | None = None

    # -- process lifecycle --

    def _spawn(self) -> None:
        parent_conn, child_conn = self._ctx.Pipe()
        proc = self._ctx.Process(target=_worker_main, args=(child_conn,), daemon=True)
        proc.start()
        child_conn.close()  # the parent keeps only its own end
        self._proc = proc
        self._conn = parent_conn

    def _ensure_alive(self) -> None:
        if self._proc is None or not self._proc.is_alive():
            self._teardown()
            self._spawn()

    def _close_conn(self) -> None:
        if self._conn is not None:
            with contextlib.suppress(OSError):
                self._conn.close()
            self._conn = None

    @staticmethod
    def _join(proc: BaseProcess, timeout: float) -> None:
        """``join`` that treats EINTR as retry, never as done."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                proc.join(remaining)
                return
            except InterruptedError:  # pragma: no cover - platform/timing dependent
                continue

    def _kill(self) -> None:
        proc = self._proc
        if proc is not None:
            try:
                if proc.is_alive():
                    proc.terminate()
                    self._join(proc, 0.5)
                if proc.is_alive():
                    proc.kill()
                    self._join(proc, 1.0)
            except (OSError, ValueError):  # pragma: no cover - defensive
                pass
        self._teardown()

    def _teardown(self) -> None:
        self._close_conn()
        self._proc = None

    def close(self) -> None:
        """Shut the worker down gracefully, escalating to a kill if needed."""
        proc = self._proc
        if proc is None:
            self._close_conn()
            return
        if self._conn is not None and proc.is_alive():
            with contextlib.suppress(BrokenPipeError, OSError):
                self._conn.send(None)  # shutdown sentinel
            self._join(proc, 0.5)
        if proc.is_alive():
            self._kill()
        else:
            self._teardown()

    # -- request/response with an EINTR-safe deadline --

    def _poll(self, timeout: float) -> bool:
        """True if a reply is ready within ``timeout``; retries on EINTR."""
        assert self._conn is not None
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                return self._conn.poll(remaining)
            except InterruptedError:  # pragma: no cover - platform/timing dependent
                continue

    def _request(self, msg: tuple[object, ...], timeout: float) -> object:
        assert self._conn is not None
        try:
            self._conn.send(msg)
        except (BrokenPipeError, OSError):
            return _CRASH
        if not self._poll(timeout):
            return _TIMEOUT
        try:
            return self._conn.recv()
        except (EOFError, OSError):
            return _CRASH

    # -- public API --

    def evaluate(
        self,
        db_path: str,
        gold_sql: str,
        pred_sql: str,
        timeout: float,
        row_limit: int,
    ) -> tuple[bool, MatchCategory]:
        """Score one prediction against gold under a per-query timeout.

        Gold is executed first (it establishes the reference result set); a gold
        failure/timeout is loud. Then the prediction is executed and compared.
        """
        self._ensure_alive()

        gold_reply = self._request(("gold", db_path, gold_sql, row_limit), timeout)
        if gold_reply is _TIMEOUT:
            self._kill()  # gold stuck in C -> OS-level stop, then respawn next call
            return (False, MatchCategory.GOLD_ERROR)
        if gold_reply is _CRASH:
            self._kill()
            return (False, MatchCategory.GOLD_ERROR)
        assert isinstance(gold_reply, tuple)
        if gold_reply[1] != "ok":  # "error" or "row_limit" -> loud gold failure
            return (False, MatchCategory.GOLD_ERROR)

        pred_reply = self._request(("pred", db_path, pred_sql, row_limit), timeout)
        if pred_reply is _TIMEOUT:
            self._kill()
            return (False, MatchCategory.TIMEOUT)
        if pred_reply is _CRASH:
            self._kill()
            return (False, MatchCategory.EXECUTION_ERROR)
        assert isinstance(pred_reply, tuple)
        _, match, category = pred_reply
        return (bool(match), MatchCategory(category))

    def fingerprint(
        self, db_path: str, sql: str, timeout: float, row_limit: int
    ) -> tuple[bool, str]:
        """Digest one query's result set; ``(ok, digest)``, same guarantees as probe."""
        self._ensure_alive()
        reply = self._request(("fingerprint", db_path, sql, row_limit), timeout)
        if reply is _TIMEOUT:
            self._kill()
            return (False, "")
        if reply is _CRASH:
            self._kill()
            return (False, "")
        assert isinstance(reply, tuple)
        _, ok, digest = reply
        return (bool(ok), str(digest))

    def probe(self, db_path: str, sql: str, timeout: float, row_limit: int) -> tuple[bool, str]:
        """Execute one SQL alone under the timeout; ``(ok, error_message)``."""
        self._ensure_alive()
        reply = self._request(("probe", db_path, sql, row_limit), timeout)
        if reply is _TIMEOUT:
            self._kill()  # stuck in C -> OS-level stop; respawned on next use
            return (False, PROBE_TIMEOUT_MESSAGE)
        if reply is _CRASH:
            self._kill()
            return (False, "worker crashed while executing the query")
        assert isinstance(reply, tuple)
        _, ok, error = reply
        return (bool(ok), str(error))
