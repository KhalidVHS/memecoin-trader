"""The durable ledger, and the one place that knows how to serialize our types.

Audit C11 — *"Persistence and execution are not transactional or idempotent"* —
is the whole reason this file is more than forty lines. The old shape was:
``broker._commit()`` appended a trade row to ``trades.jsonl`` and then wrote
``state.json``. Two files, two operations, no transaction, no IDs. A kill
between them leaves a ledger ahead of state with nothing to say which rows
belonged to which decision, and the audit's evidence line is blunt: *"No order
ID, decision ID, unique constraint, database transaction, sequence, WAL
recovery, process lock, or venue reconciliation exists."*

The replacement is an **event-sourced append-only ledger**. Positions are
derived from immutable rows rather than mirrored in a second file, so there is
no second file to disagree with. Four properties make that safe.

**1. Every row carries identity.** ``schema_version``, ``run_id``, and whichever
of ``decision_id``/``action_id``/``intent_id``/``order_id``/``fill_id`` apply.
That is what lets :func:`fills_by_intent` join a fill to the intent that caused
it *by ID*. The old code matched fills to actions by symbol, so a stop-loss and
a strategy SELL on the same coin in the same tick were indistinguishable — the
audit's ``prompts._decision_line`` finding — and the attribution in the report
was therefore a coin flip in exactly the situation you most want to understand.

**2. Rows sort themselves.** The IDs from ``ids.py`` are prefixed with a
zero-padded 18-digit microsecond timestamp, so **lexical sort equals
chronological sort**. ``sorted(rows, key=itemgetter("row_id"))`` is a correct
chronological sort without parsing a single timestamp, and ``sort`` on the raw
file does the same thing from a shell. Microseconds rather than seconds because
a slow tick emits several intents inside one second and they must still order.

**3. Appends are durable before they return.** Write, ``flush``, ``os.fsync``,
then return. Without the fsync, ``write`` has only reached the OS page cache: a
power loss or a hard kill can lose a row the caller was told was persisted, and
the one row you lose is the order that was in flight. Any operation that is not
an append — a compaction, a snapshot export — goes through
:func:`atomic_write_text`, which is temp-file + fsync + ``os.replace``, so a
reader never observes a half-written file.

**4. Re-appending the same row is a no-op.** :meth:`Ledger.append` is keyed on
the row's primary ID. Appending the same ``intent_id`` twice writes one row.
That is what makes recovery safe to run repeatedly, and it is the audit's
"unique constraint" in the only form a JSONL file can offer one.

---

**The crash-safety argument, written out.** Take a kill at each point in
``intent → submit → fill``:

* *Before the intent row is appended.* The ledger has no record and no side
  effect happened. Nothing to reconcile; the order simply never existed.
* *Mid-append of the intent row.* The process died with a partial line and no
  trailing newline. :func:`scan` parses lines independently, discards a
  trailing line that does not parse, and **reports it** via
  ``ScanResult.torn_final_line`` rather than swallowing it. The torn line is
  discarded rather than repaired because a half-written JSON object is not
  partially true — you cannot tell whether the missing half contained a field
  that changes its meaning. Every earlier row is intact, because the file is
  append-only and nothing rewrites them.
* *After the intent row, before submission.* :meth:`Ledger.open_intents`
  returns it: an intent with no terminal ``OrderState``. Startup calls that and
  refuses to trade the symbol until the operator or the reconciler resolves it.
  This is exactly the audit's "Append/state crash in current broker → **Refuse
  startup until repaired**".
* *After submission, before the fill row.* Identical ledger state to the
  previous case, and deliberately so: the ledger cannot tell them apart and
  must not pretend to. The intent is open, the truth is at the venue, and
  reconciliation is a venue/wallet query — not an inference. In ``PAPER`` mode
  the "venue" is this same ledger, so the intent is simply marked ``EXPIRED``.
* *Mid-append of the fill row.* Torn final line again: discarded and reported,
  and the intent is still open, so recovery lands in the case above. The order
  in which rows are written is what makes this safe — the intent is always
  durable *before* the thing it describes is attempted.
* *After the fill row.* The fill carries a terminal state, ``open_intents``
  does not return the intent, and there is no second file that could disagree.

There is deliberately no repair path that edits the ledger in place. A ledger
you can rewrite is a ledger you cannot learn from, and the audit's whole
complaint is about state that was changed without a record.

**Windows specifics**, since this runs on Windows 11 and the semantics differ:

* ``os.replace`` is atomic on Windows (it is ``MoveFileEx`` with
  ``REPLACE_EXISTING``) but it fails with ``PermissionError`` if the
  destination is open in another process. Nothing else opens these files for
  writing — there is one writer by design — but a reader holding the file (an
  editor, ``tail -f``) can make a snapshot rewrite fail. It fails loudly rather
  than corrupting, which is the correct trade.
* ``os.fsync`` on Windows maps to ``FlushFileBuffers``, which does flush the
  drive's own cache. It is honoured.
* Directory fsync — the POSIX step that makes a *rename* durable, not just the
  file's contents — is not available on Windows: you cannot ``os.open`` a
  directory. :func:`atomic_write_text` attempts it and ignores the failure,
  which means on Windows a crash in the microseconds between ``replace`` and
  the metadata flush can lose a *snapshot*. It cannot lose a ledger row,
  because rows are appended and fsynced in place, never replaced.

**Encoding is pinned to UTF-8 everywhere, explicitly.** Windows defaults to
cp1252 for both files and redirected stdio, and this codebase has already been
bitten: ``memetrader status | tail`` died with ``UnicodeEncodeError`` on a
``↓`` while the same command in a terminal was fine. Every ``open`` here names
``encoding="utf-8"`` and every dump uses ``ensure_ascii=False``, so the model's
em dashes and arrows land as themselves and a log you read with your eyes is
readable.

Two serialization decisions are load-bearing and both regress silently:

* ``Side`` has to land as ``"BUY"``, not ``"Side.BUY"``. That works only
  because it is a ``StrEnum``; a plain ``Enum`` makes ``json.dumps`` raise, and
  the ledger stops being written at the exact moment a trade happens.
* ``float('inf')`` has to land as the string ``"inf"``. ``TxnCounts.ratio``
  returns it for a pool with buys and no sells — the strongest buy-pressure
  reading the feed can produce — and JSON has no literal for it. A ``null``
  there would be indistinguishable from "we did not measure", which is the one
  distinction this system is built around.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .types import Fill, OrderIntent, OrderState

__all__ = [
    "SCHEMA_VERSION",
    "TERMINAL_STATES",
    "Ledger",
    "LedgerError",
    "OpenIntent",
    "RowKind",
    "ScanResult",
    "append",
    "atomic_write_text",
    "fills_by_intent",
    "read",
    "rows_for_decision",
    "scan",
    "tail",
    "to_jsonable",
]

log = logging.getLogger(__name__)

# Bumped whenever a row's shape changes in a way a reader must know about.
# Version 1 was the untagged pre-audit row: no IDs, no envelope, no version.
# Rows without a ``schema_version`` key are therefore version 1 by definition,
# which is how :func:`scan` can read a ledger written before this file existed.
SCHEMA_VERSION = 2

ENCODING = "utf-8"

TERMINAL_STATES: frozenset[OrderState] = frozenset(s for s in OrderState if s.is_terminal)


class LedgerError(RuntimeError):
    """Something is wrong with the ledger itself, not with a row's contents."""


class RowKind(StrEnum):
    """What a row is. The envelope's discriminator.

    One file holds every kind rather than one file per kind, because the
    ordering *between* kinds is the thing recovery needs: "the intent was
    written, then the process died" is only a fact if both would have gone to
    the same append-ordered file.
    """

    DECISION = "decision"
    INTENT = "intent"
    ORDER_STATE = "order_state"
    FILL = "fill"
    NOTE = "note"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses, pydantic models, enums and tuples into
    plain JSON types. Enums are StrEnum so they serialize as their value."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        # Sorted, not iteration-ordered. A set's iteration order varies with
        # PYTHONHASHSEED, which would make two replays of the same run produce
        # different bytes — and "the dry run reproduces byte-identically" is the
        # Phase 1 exit gate. `frozenset` is listed explicitly because it is not
        # a subclass of `set`; omitting it is how RiskState.quarantined_symbols
        # made a decision row unwritable.
        return sorted(to_jsonable(v) for v in obj)
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float):
        # inf and nan are legal Python but not legal JSON. TxnCounts.ratio
        # returns inf for a pool with zero sells, which is meaningful — encode
        # it as a string so it survives the round trip visibly.
        if obj != obj:  # NaN
            return None
        if obj in (float("inf"), float("-inf")):
            return str(obj)
        return obj
    return obj


def _dumps(record: Any) -> str:
    """One row, one line, UTF-8, no ASCII escaping.

    ``ensure_ascii=False`` is deliberate — see the module docstring's encoding
    note. The line must contain no raw newline, which ``json.dumps`` guarantees
    by escaping them inside strings; a model reason containing ``\\n`` becomes
    ``\\\\n`` and stays one physical row.
    """
    return json.dumps(to_jsonable(record), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Durable I/O
# ---------------------------------------------------------------------------


def append(path: Path, record: Any, *, fsync: bool = True) -> None:
    """Append one row and make it durable before returning.

    ``flush`` moves the bytes from Python's buffer to the OS; ``os.fsync``
    moves them from the OS page cache to the device. Only the second one makes
    the claim "this row is persisted" true across a power loss or a hard kill,
    and the row we would lose without it is by construction the most recent —
    the order that was in flight.

    The cost is real: an fsync per row is a few hundred microseconds to a few
    milliseconds. At this system's volume — a handful of rows per 60-second
    tick — that is free, and it is the single cheapest thing in C11's fix
    list. ``fsync=False`` exists only for bulk test fixtures that write
    thousands of rows and do not care.

    Opened with ``encoding="utf-8"`` explicitly: on Windows the default is
    cp1252, and a model reason containing an arrow would raise
    ``UnicodeEncodeError`` mid-tick. That has already happened in this
    codebase's output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding=ENCODING) as fh:
        fh.write(_dumps(record) + "\n")
        fh.flush()
        if fsync:
            os.fsync(fh.fileno())


def atomic_write_text(path: Path, text: str) -> None:
    """Replace a whole file's contents without a reader ever seeing a partial one.

    Temp file in the *same directory* (``os.replace`` across filesystems is not
    atomic and on Windows raises outright), fsync the temp file, then
    ``os.replace``. A reader either sees the old file or the new one.

    Used for snapshots and exports only. It is emphatically not how ledger rows
    are written — rewriting a ledger to add a row would make every earlier row
    depend on the success of the latest write, which is the failure mode the
    append-only design exists to remove.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding=ENCODING) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        # ``Path.replace`` *is* ``os.replace`` — same syscall, same atomicity
        # guarantee, and on Windows the same ``MoveFileEx(REPLACE_EXISTING)``.
        # Spelled this way only because the project lints for pathlib.
        Path(tmp_name).replace(path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    # POSIX: the rename itself is only durable once the *directory* is synced.
    # Windows cannot open a directory as a file descriptor, so this is a no-op
    # there — documented in the module docstring rather than pretended away.
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


# ---------------------------------------------------------------------------
# Reading, and the torn final line
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScanResult:
    """Every good row, plus an honest account of what was thrown away.

    The old ``read`` swallowed a bad line with a bare ``continue``. That is the
    right *behaviour* — the rest of the ledger is still a true record — but
    silently is the wrong way to do it, because "the ledger has 412 rows" and
    "the ledger has 412 rows and one that could not be read" are different
    facts and only the second one explains a missing trade.
    """

    path: Path
    rows: tuple[dict[str, Any], ...]
    # A final line with no trailing newline that did not parse: the signature of
    # a process killed mid-append. Distinguished from mid-file corruption
    # because they have different causes and different responses — this one is
    # expected and benign, that one means something rewrote the file.
    torn_final_line: bool = False
    torn_final_bytes: int = 0
    corrupt_line_numbers: tuple[int, ...] = ()

    @property
    def clean(self) -> bool:
        return not self.torn_final_line and not self.corrupt_line_numbers

    @property
    def discarded(self) -> int:
        return len(self.corrupt_line_numbers) + (1 if self.torn_final_line else 0)


def scan(path: Path, *, logger: logging.Logger | None = None) -> ScanResult:
    """Read the whole ledger, tolerating a torn final line and saying so.

    The only corruption an append-only file can produce by itself is a partial
    *last* line: the process died between the ``write`` and the ``\\n``. It is
    detected structurally — the line failed to parse **and** the file does not
    end in a newline — rather than by guessing from the content, so a
    genuinely malformed row in the middle is reported as corruption instead of
    being excused as a crash.

    Mid-file corruption is skipped too, with its line number recorded. A reader
    that stopped at the first bad line would hide every row after it, and those
    rows are still a true record of trades that happened.

    ``encoding="utf-8"`` is explicit: reading a UTF-8 ledger with the Windows
    default cp1252 would raise ``UnicodeDecodeError`` on the first em dash,
    which means the file becomes unreadable on the machine that wrote it.
    """
    lg = logger or log
    if not path.is_file():
        return ScanResult(path=path, rows=())

    raw = path.read_text(encoding=ENCODING)
    if not raw:
        return ScanResult(path=path, rows=())

    ends_clean = raw.endswith("\n")
    lines = raw.split("\n")
    if ends_clean:
        lines = lines[:-1]  # split leaves a trailing "" after the final newline

    rows: list[dict[str, Any]] = []
    corrupt: list[int] = []
    torn = False
    torn_bytes = 0

    last_index = len(lines) - 1
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            if index == last_index and not ends_clean:
                torn = True
                torn_bytes = len(line.encode(ENCODING))
                lg.warning(
                    "ledger torn final line discarded path=%s bytes=%d",
                    path,
                    torn_bytes,
                )
            else:
                corrupt.append(index + 1)
                lg.error("ledger corrupt row skipped path=%s line=%d", path, index + 1)
            continue
        if not isinstance(parsed, dict):
            corrupt.append(index + 1)
            lg.error("ledger non-object row skipped path=%s line=%d", path, index + 1)
            continue
        rows.append(parsed)

    return ScanResult(
        path=path,
        rows=tuple(rows),
        torn_final_line=torn,
        torn_final_bytes=torn_bytes,
        corrupt_line_numbers=tuple(corrupt),
    )


def read(path: Path) -> Iterator[dict[str, Any]]:
    """Yield every readable row. Kept for ``report.py`` and the prompt history.

    A thin wrapper over :func:`scan`, which means a torn line is now *logged*
    where it used to vanish. Anything doing recovery should call :func:`scan`
    directly and inspect ``torn_final_line`` — a caller that needs to know
    should not have to read the log to find out.
    """
    yield from scan(path).rows


def tail(path: Path, n: int) -> list[dict[str, Any]]:
    """The last ``n`` rows, oldest first.

    ``n <= 0`` returns nothing. ``rows[-0:]`` is the whole list, so the guard
    is the only thing standing between "show me no history" and "show me all of
    it", and ``prompt.decision_history = 0`` has to mean zero.
    """
    if n <= 0:
        return []
    return list(scan(path).rows[-n:])


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


def _row_envelope(
    *,
    kind: RowKind,
    run_id: str,
    ts: float,
    primary_id: str,
    payload: Any,
    decision_id: str | None = None,
    action_id: str | None = None,
    intent_id: str | None = None,
    order_id: str | None = None,
    fill_id: str | None = None,
    state: OrderState | None = None,
) -> dict[str, Any]:
    """The shape of every row. Flat IDs at the top level on purpose.

    The IDs sit beside ``kind`` rather than inside ``payload`` so that a join,
    a reconciliation query, or a ``grep`` for an order ID works without knowing
    what kind of row it is looking at. ``row_id`` is the primary ID, so the
    microsecond prefix from ``ids.py`` makes the file sort chronologically by
    that one key.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "row_id": primary_id,
        "kind": str(kind),
        "ts": ts,
        "run_id": run_id,
        "decision_id": decision_id,
        "action_id": action_id,
        "intent_id": intent_id,
        "order_id": order_id,
        "fill_id": fill_id,
        "state": None if state is None else str(state),
        "payload": to_jsonable(payload),
    }


@dataclass(frozen=True, slots=True)
class OpenIntent:
    """An intent the ledger has no terminal outcome for.

    This is what a crash leaves behind, and what startup must resolve before
    the symbol may be traded again. ``last_state`` is the furthest the order is
    known to have got — ``PROPOSED`` means nothing was attempted and it is safe
    to drop; ``SUBMITTED`` means the truth is at the venue and only a venue or
    wallet query can settle it. The audit's safe fallback for that row is
    "**Reconcile before any new order**", and the distinction between those two
    states is what makes the difference between a note and a halt.
    """

    intent_id: str
    run_id: str
    ts: float
    symbol: str
    side: str
    source: str
    decision_id: str | None
    action_id: str | None
    order_id: str | None
    last_state: OrderState
    last_state_ts: float

    @property
    def was_submitted(self) -> bool:
        return self.last_state is OrderState.SUBMITTED


class Ledger:
    """One append-only JSONL file, one run, idempotent appends.

    The index of already-written primary IDs is built by scanning the file at
    construction and kept in memory thereafter. That is O(file) per process
    start, which at this system's volume (single-digit rows per tick) is
    nothing, and it is the only way a plain JSONL file can offer a uniqueness
    constraint at all. It is also the reason there must be exactly one writer:
    two processes appending to the same file would each hold half the index and
    neither would detect the other's duplicates. ``run_id`` on every row is
    what makes that incident *visible* afterwards — two run IDs interleaved in
    one file — rather than inexplicable.

    Not a context manager and holding no open handle: each append opens,
    writes, fsyncs and closes. A long-lived handle would be faster and would
    also be a file that an ``os.replace`` elsewhere could not replace, and on
    Windows that is a ``PermissionError`` rather than a slow path.
    """

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        fsync: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.fsync = fsync
        self.log = logger or log
        initial = scan(self.path, logger=self.log)
        self._seen: set[str] = {
            key for key in (_idempotency_key(row) for row in initial.rows) if key
        }
        self.opened_with = initial

    # -- appends --------------------------------------------------------

    def _append_row(self, row: dict[str, Any]) -> bool:
        """Write ``row`` unless its key is already present. True if written.

        The check is in memory and the write is on disk, so a crash between
        them is possible — and harmless, because the crash means the row was
        *not* written and the next process rebuilds the index from the file.
        The ordering that would be unsafe is the reverse one.
        """
        key = _idempotency_key(row)
        if key and key in self._seen:
            self.log.info(
                "ledger duplicate suppressed kind=%s row_id=%s", row.get("kind"), key
            )
            return False
        append(self.path, row, fsync=self.fsync)
        if key:
            self._seen.add(key)
        return True

    def append_intent(
        self,
        intent: OrderIntent,
        *,
        state: OrderState = OrderState.PROPOSED,
    ) -> bool:
        """Record an intent **before** anything is quoted or submitted.

        The ordering is the crash-safety argument: durable intent, then side
        effect. Reversed, a kill after submission leaves no evidence the order
        ever existed, and reconciliation has nothing to look for.

        Idempotent on ``intent_id``, which ``ids.new_intent_id`` describes as
        the idempotency key: a retry reuses it, so a duplicated attempt writes
        one row instead of becoming a second order.
        """
        return self._append_row(
            _row_envelope(
                kind=RowKind.INTENT,
                run_id=intent.run_id or self.run_id,
                ts=intent.ts,
                primary_id=intent.intent_id,
                decision_id=intent.decision_id,
                action_id=intent.action_id,
                intent_id=intent.intent_id,
                state=state,
                payload=intent,
            )
        )

    def append_state(
        self,
        *,
        intent_id: str,
        state: OrderState,
        ts: float,
        order_id: str | None = None,
        decision_id: str | None = None,
        note: str | None = None,
    ) -> bool:
        """Record a transition in the C11 state machine.

        Idempotent on ``(intent_id, state)`` rather than on ``intent_id``: an
        order legitimately passes through several states and each is a distinct
        row, but recording the same transition twice — which a re-run of
        recovery does — must not create two.
        """
        return self._append_row(
            _row_envelope(
                kind=RowKind.ORDER_STATE,
                run_id=self.run_id,
                ts=ts,
                primary_id=f"{intent_id}:{state}",
                decision_id=decision_id,
                intent_id=intent_id,
                order_id=order_id,
                state=state,
                payload={"note": note},
            )
        )

    def append_fill(self, fill: Fill) -> bool:
        """Record a settled *or failed* attempt.

        A failed attempt is a row too, with zero amounts and non-zero gas.
        Dropping it would hide what the failure cost and would leave the intent
        looking open forever.

        Idempotent on ``fill_id``. The fill carries ``intent_id`` and
        ``decision_id`` in the envelope, which is what makes the join by ID
        possible — a stop-loss fill and a strategy fill on the same symbol in
        the same tick carry different ``intent_id`` values and are therefore
        distinguishable, which under the old symbol-matching scheme they were
        not.
        """
        return self._append_row(
            _row_envelope(
                kind=RowKind.FILL,
                run_id=self.run_id,
                ts=fill.ts,
                primary_id=fill.fill_id,
                decision_id=fill.decision_id,
                intent_id=fill.intent_id,
                order_id=fill.order_id,
                fill_id=fill.fill_id,
                state=fill.state,
                payload=fill,
            )
        )

    def append_decision(self, record: Any) -> bool:
        """Record one strategy invocation. Idempotent on ``decision_id``."""
        decision_id = getattr(record, "decision_id", None)
        if not decision_id:
            raise LedgerError("a decision row must carry a decision_id")
        return self._append_row(
            _row_envelope(
                kind=RowKind.DECISION,
                run_id=getattr(record, "run_id", None) or self.run_id,
                ts=float(getattr(record, "ts", 0.0)),
                primary_id=decision_id,
                decision_id=decision_id,
                payload=record,
            )
        )

    def append_note(self, *, note_id: str, ts: float, text: str, **ids: Any) -> bool:
        """An operator- or recovery-authored annotation.

        Notes are how a reconciliation decision gets into the record without
        editing a row. There is no other way — nothing in this module mutates
        an existing line.
        """
        return self._append_row(
            _row_envelope(
                kind=RowKind.NOTE,
                run_id=self.run_id,
                ts=ts,
                primary_id=note_id,
                decision_id=ids.get("decision_id"),
                action_id=ids.get("action_id"),
                intent_id=ids.get("intent_id"),
                order_id=ids.get("order_id"),
                fill_id=ids.get("fill_id"),
                payload={"text": text},
            )
        )

    # -- queries --------------------------------------------------------

    def scan(self) -> ScanResult:
        return scan(self.path, logger=self.log)

    def rows(self) -> tuple[dict[str, Any], ...]:
        return self.scan().rows

    def open_intents(self) -> tuple[OpenIntent, ...]:
        """Intents with no terminal ``OrderState``. Startup calls this.

        "Terminal" means ``LANDED``, ``FAILED``, ``EXPIRED`` or ``RECONCILED``
        — ``OrderState.is_terminal``. Anything else is an order that was in
        flight when the process died, and the audit's failure table is
        unambiguous about what that means: *reconcile before any new order*.

        Returned in ledger order, which — because ``intent_id`` carries a
        microsecond prefix and is the row's ``row_id`` — is chronological
        order, oldest first. The oldest open intent is the one most likely to
        have actually settled at the venue, so it is the one to resolve first.
        """
        return open_intents(self.scan().rows)


def _idempotency_key(row: dict[str, Any]) -> str | None:
    """The uniqueness constraint, derived from a row rather than stored twice.

    Legacy version-1 rows have no ``kind`` and no ``row_id``. They return
    ``None``, which means "not deduplicated" — correct, because they predate
    identity entirely and inventing one for them would make two genuinely
    distinct old rows collide.
    """
    kind = row.get("kind")
    row_id = row.get("row_id")
    if not kind or not isinstance(row_id, str):
        return None
    return f"{kind}:{row_id}"


# ---------------------------------------------------------------------------
# Joins and reconciliation — by ID, never by symbol
# ---------------------------------------------------------------------------


def fills_by_intent(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group fill rows by the ``intent_id`` that caused them.

    This function is the audit's ``prompts._decision_line`` fix. The old code
    matched fills to actions by *symbol*, so a stop-loss exit and a strategy
    SELL on BONK in the same tick were the same key — the report attributed
    both to whichever it looked at first, and the one case where you urgently
    want to know which policy sold is the one case it could not tell you.

    A symbol is a property of an order. An ``intent_id`` *is* the order.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("kind") != RowKind.FILL:
            continue
        intent_id = row.get("intent_id")
        if not isinstance(intent_id, str):
            continue
        grouped.setdefault(intent_id, []).append(row)
    return grouped


def rows_for_decision(
    rows: Iterable[dict[str, Any]], decision_id: str
) -> tuple[dict[str, Any], ...]:
    """Every row — decision, intents, transitions, fills — for one decision.

    Filtered on the envelope's ``decision_id``, so a row belongs to a decision
    because it says so, not because its timestamp is nearby. Two ticks that
    overlap (a slow tick still running when the next fires) would be
    inseparable under a time window and are trivially separable here.
    """
    return tuple(r for r in rows if r.get("decision_id") == decision_id)


def open_intents(rows: Iterable[dict[str, Any]]) -> tuple[OpenIntent, ...]:
    """Reconciliation query: intents with no terminal state. See :class:`Ledger`.

    The last state wins, taken in ledger order. That is correct because the
    ledger is append-only and rows are appended in the order the transitions
    happened; it would not be correct for a file that could be rewritten, which
    is one more reason nothing rewrites it.
    """
    intents: dict[str, dict[str, Any]] = {}
    last_state: dict[str, tuple[OrderState, float]] = {}
    last_order_id: dict[str, str] = {}

    for row in rows:
        intent_id = row.get("intent_id")
        if not isinstance(intent_id, str):
            continue
        kind = row.get("kind")
        if kind == RowKind.INTENT:
            intents.setdefault(intent_id, row)
        order_id = row.get("order_id")
        if isinstance(order_id, str):
            last_order_id[intent_id] = order_id
        raw_state = row.get("state")
        if isinstance(raw_state, str):
            try:
                state = OrderState(raw_state)
            except ValueError:
                # An unknown state is not a terminal state. Treating it as one
                # would silently close an order whose outcome we cannot read,
                # which is the opposite of what recovery is for.
                continue
            last_state[intent_id] = (state, float(row.get("ts") or 0.0))

    out: list[OpenIntent] = []
    for intent_id, row in intents.items():
        state, state_ts = last_state.get(intent_id, (OrderState.PROPOSED, 0.0))
        if state.is_terminal:
            continue
        payload = row.get("payload") or {}
        out.append(
            OpenIntent(
                intent_id=intent_id,
                run_id=str(row.get("run_id") or ""),
                ts=float(row.get("ts") or 0.0),
                symbol=str(payload.get("symbol") or ""),
                side=str(payload.get("side") or ""),
                source=str(payload.get("source") or ""),
                decision_id=row.get("decision_id"),
                action_id=row.get("action_id"),
                order_id=last_order_id.get(intent_id),
                last_state=state,
                last_state_ts=state_ts,
            )
        )
    # ``intent_id`` is microsecond-prefixed, so lexical sort is chronological
    # sort. This is the property ids.py promises, relied on here rather than
    # re-deriving order from the ``ts`` field (which a clock step can reorder).
    out.sort(key=lambda o: o.intent_id)
    return tuple(out)
