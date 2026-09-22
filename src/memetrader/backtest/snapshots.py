"""Crash/restart snapshots — resuming a replay must be indistinguishable from
never having stopped it.

A :class:`ReplaySnapshot` is a point-in-time capture of everything needed to
resume a backtest replay: the ledger's full accounting state (via
``BacktestLedger.to_state()``), the simulated clock's position, the run's
identity (``run_id`` plus the config/data hashes that say *which* run this
is), and whatever RNG and event-queue bookkeeping the caller hands in. Saving
and loading one must never change a single number the replay would otherwise
have produced — this module's entire job is durability and identity
checking, never interpretation of the economic state it carries.

Why this file does not itself define the shape of "the event queue's
position" or "an RNG stream's state"
=====================================

``backtest/engine.py`` — the component that actually owns a live
``EventQueue`` and a live per-stream RNG — does not exist yet (it is another
agent's module in this same effort, per ``docs/BACKTEST-CONTRACTS.md``'s
layout). Two things follow from that, and both are deliberate rather than
oversights:

1. **The event queue is not introspectable from here.**
   ``backtest/event_queue.py``'s public surface is ``push`` / ``__iter__`` /
   ``__next__`` / ``__bool__`` / ``peek`` — there is no accessor for
   per-stream cursor position, and popping to find out would consume events
   this module has no business consuming. This module therefore accepts
   whatever cursor bookkeeping the caller (eventually the engine, which
   already knows which row of which loader it is on) chooses to hand in as
   ``event_queue_state``, stores it opaquely, and round-trips it exactly.
   If the caller passes nothing, that is recorded honestly in
   ``not_captured`` — see the module-level note on the point in
   ``docs/BACKTEST-CONTRACTS.md`` §0 that "did not capture" must never render
   as "captured, and it was empty."

2. **RNG state is opaque here too, and that is the correct amount of
   knowledge for this module to have** — but it is worth writing out exactly
   why continuity is even possible, because it is the subtle part of this
   assignment.

   ``SimulatedClock.deterministic_seed(stream_name)`` derives a seed from
   ``(run_id, stream_name)`` via SHA-256. That seed is stable across
   processes, which is necessary for reproducibility, but it is **not**
   sufficient for *resuming* a stream: a ``random.Random(seed)`` that has
   already produced N draws is, internally, a completely different object
   from a freshly-seeded ``random.Random(seed)``. Re-deriving the seed at
   resume time and reseeding a fresh RNG from it would silently rewind the
   stream to draw #0 — every draw after the restore would repeat draws the
   uninterrupted run already made, which is exactly the kind of "looks fine,
   is silently wrong" bug §0 warns about.

   The only way to make draw N+1 after a restore equal draw N+1 without one
   is to persist the RNG's **consumed internal state**, not its seed, and
   restore that exact state before drawing again. ``clock.py`` does not own
   any RNG objects — by design, per its own docstring, callers are expected
   to keep one ``random.Random``/``numpy.random.Generator`` per named stream,
   seeded once via ``deterministic_seed`` — so *this* module cannot capture
   that state on its own either; it can only provide the storage contract
   (an opaque, JSON-safe blob per stream name) and, for the common case of
   Python's stdlib ``random.Random``, the two small helpers below
   (:func:`rng_state_from_random` / :func:`restore_random_state`) that do the
   capture/restore correctly. ``deep_tuple`` exists because JSON has no tuple
   type: ``random.Random.getstate()`` returns nested tuples, JSON round-trips
   them as nested lists, and ``random.Random.setstate`` requires the original
   nested-tuple shape back.

   **This means RNG continuity is exact, not approximate — but only if the
   caller captures and restores the consumed state, not merely the seed.** A
   caller that snapshots ``rng_states={}`` (or omits a stream) and reseeds
   fresh from ``deterministic_seed`` at resume will diverge from the
   uninterrupted run at the first draw after the restore. That divergence
   would be this module's caller's bug, not a limitation of the format —
   :func:`build_snapshot` records any stream name missing from
   ``rng_states`` as not captured for exactly this reason, and the
   crash-restart-equivalence test in ``tests/backtest/test_snapshots.py``
   fails immediately if a caller gets this wrong.

Durable, atomic writes
======================

``save`` writes through ``journal.atomic_write_text`` — the same
temp-file-in-the-same-directory, fsync, ``os.replace`` idiom ``journal.py``
uses for its own snapshot exports, and the one ``experiments/manifest.py``
already reuses for the same reason. A reader of the target path only ever
sees the old complete file or the new complete file; a crash mid-write
leaves a stray, unreferenced temp file (which ``atomic_write_text`` itself
unlinks on any exception) and never a half-written file at the path callers
actually read from.

Refused, not migrated
======================

``ReplaySnapshot.schema_version`` is checked exactly like
``BacktestLedger.from_state`` checks its own: an unrecognised version raises
rather than being guessed at. The two version numbers are independent and
both present in a saved file — the snapshot's own ``schema_version`` and the
nested ``ledger_state["schema_version"]`` — because the snapshot format and
the ledger's internal format can each change on their own schedule.

Known gap: lot-id labels are not guaranteed stable across a restart
====================================================================

Crash-restart equivalence holds **exactly** for every economic quantity —
cash, token quantities, realized PnL, fee totals, and therefore
``LedgerSnapshot`` equality and every accounting invariant in
``invariants.py``. It does **not** hold byte-for-byte for the *labels*
``BacktestLedger`` assigns to individual FIFO lots (``LedgerEntry
.lot_created`` / ``.lots_consumed``) in one specific case: ``to_state()``
serializes only *currently open* lots (its "lots" dict comprehension is
guarded by ``if dq``), and ``from_state()`` reconstructs the internal
``_lot_seq_counter`` solely from the lot IDs present in that serialized set.
A lot that was opened and then fully closed (sold down to zero) before the
snapshot was taken leaves no trace of the sequence number it used, so after
a restore the counter can resume lower than an uninterrupted run's would
have and reissue a lot-id string an earlier, already-closed lot also used.

No dollar, no token quantity, and no PnL figure is affected by this — the
label collision does not change what any invariant check computes, because
every check in ``invariants.py`` that touches ``lot_created``
(``check_fill_lot_integrity``) only requires that a consuming SELL's
``lot_id`` was created by *some* earlier BUY in the same, already-restored
log, which it still was. But a byte-identical comparison of raw
``ledger.entries()`` (or ``ledger.lot_views()``) across an interrupted vs.
uninterrupted run of the same input can fail on this field alone. This is a
gap in ``ledger.py``'s own serialization (out of scope for this module —
``ledger.py`` is a frozen file for this assignment), stated here rather than
silently worked around, per the instruction to report such a finding loudly.
``tests/backtest/test_snapshots.py``'s headline crash-restart test
documents and demonstrates exactly this: it asserts full equality of every
economic field in the entry log and of the resulting ``LedgerSnapshot``, but
deliberately excludes ``lot_created``/``lots_consumed`` from that
comparison, with the reasoning inline.

Resuming the wrong run is silent corruption, so it must not be silent
=====================================================================

``verify_resume`` (and ``resume_ledger``, which calls it) compares the
snapshot's ``run_id``, ``config_hash``, and ``data_partition_hashes`` against
the values the caller is about to resume with, and raises
:class:`SnapshotMismatch` on any disagreement. Resuming a snapshot against
different data is exactly the "would be nearly impossible to notice
downstream" bug this project's docs call out repeatedly — the equity curve
downstream still looks plausible, it is simply not a fact about the run
whose numbers it claims to be.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from memetrader.backtest.ledger import BacktestLedger
from memetrader.journal import atomic_write_text

if TYPE_CHECKING:
    from pathlib import Path

    from memetrader.backtest.clock import SimulatedClock

__all__ = [
    "SCHEMA_VERSION",
    "ReplaySnapshot",
    "SnapshotCorrupt",
    "SnapshotError",
    "SnapshotMismatch",
    "UnsupportedSchemaVersion",
    "build_snapshot",
    "deep_tuple",
    "load",
    "restore_random_state",
    "resume_ledger",
    "rng_state_from_random",
    "save",
    "verify_resume",
]

#: Bumped whenever the *snapshot envelope*'s shape changes in a way a reader
#: must know about. Independent of ``ledger.py``'s own ``schema_version`` —
#: see the module docstring's "Refused, not migrated" section.
SCHEMA_VERSION = 1

# The reason recorded in ``ReplaySnapshot.not_captured`` when a caller does
# not supply ``event_queue_state`` to :func:`build_snapshot`. Named as a
# module constant so every snapshot that hits this path records the exact
# same, greppable string rather than slightly different prose each time.
_EVENT_QUEUE_NOT_CAPTURED = "event_queue_stream_cursors"


class SnapshotError(RuntimeError):
    """Something is wrong with a snapshot — refused, never guessed at."""


class UnsupportedSchemaVersion(SnapshotError):
    """A snapshot's ``schema_version`` is not one this build knows how to
    read. Mirrors ``ledger.LedgerError``'s refusal for the same situation:
    move the file aside and write a migration, do not guess at its shape."""


class SnapshotCorrupt(SnapshotError):
    """A snapshot file could not be parsed as the JSON object this module
    writes. Covers a truncated write, a hand-edited file, and anything else
    that is not valid JSON or not a JSON object at the top level."""


class SnapshotMismatch(SnapshotError):
    """A snapshot's run identity does not match the run it is being resumed
    into. Raised by :func:`verify_resume` (and therefore by
    :func:`resume_ledger`) rather than allowed to pass silently — resuming
    against the wrong data or config is corruption, not a warning."""


# ---------------------------------------------------------------------------
# JSON-safety helpers
# ---------------------------------------------------------------------------


def _json_safe(obj: Any) -> Any:
    """Recursively coerce ``tuple``/``set``/``frozenset`` into JSON-native
    types. Mirrors ``journal.to_jsonable``'s idiom, for the same reason:
    ``random.Random.getstate()`` returns nested tuples, and every caller
    having to pre-flatten its own RNG/queue state before handing it to this
    module would be a footgun this function removes once, here, rather than
    at every call site."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        # Sorted, not iteration-ordered — see journal.to_jsonable's comment
        # on the same choice: iteration order over a set varies with
        # PYTHONHASHSEED, which would make two snapshots of identical state
        # serialize to different bytes.
        return sorted(_json_safe(v) for v in obj)
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def deep_tuple(obj: Any) -> Any:
    """Recursively convert ``list`` back into ``tuple``.

    JSON has no tuple type, so anything captured through :func:`_json_safe`
    (in particular a ``random.Random.getstate()`` blob, whose innermost
    element is itself a 625-int tuple) comes back from :func:`load` as
    nested lists. ``random.Random.setstate`` requires the original
    nested-tuple shape, so :func:`restore_random_state` calls this first.
    Exposed publicly because a caller restoring some *other* RNG's captured
    state (numpy, a custom PRNG) may need the same conversion.
    """
    if isinstance(obj, list):
        return tuple(deep_tuple(v) for v in obj)
    return obj


def rng_state_from_random(rng: random.Random) -> Any:
    """Capture a stdlib ``random.Random``'s full internal state, ready to be
    stored under ``ReplaySnapshot.rng_states[stream_name]``.

    See the module docstring's RNG-continuity section: this captures the
    *consumed* state, not the seed — that distinction is the entire reason
    a restored stream can continue exactly rather than restart.
    """
    return rng.getstate()


def restore_random_state(rng: random.Random, state: Any) -> None:
    """Restore a stdlib ``random.Random`` to exactly the state captured by
    :func:`rng_state_from_random`, undoing the JSON list/tuple conversion
    first via :func:`deep_tuple`."""
    rng.setstate(deep_tuple(state))


# ---------------------------------------------------------------------------
# The snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReplaySnapshot:
    """Everything captured at one point in simulated time.

    ``config_hash`` and ``data_partition_hashes`` mirror
    ``experiments/manifest.py``'s fields of the same name and meaning — this
    is deliberately not a third, incompatible hashing scheme. ``None`` for
    either means "the caller did not supply one to check against", per §0's
    "``None`` = could not find out" convention; it is not treated as a
    wildcard match by :func:`verify_resume` for ``run_id``, which is always
    checked.

    ``not_captured`` is the explicit list of state categories this snapshot
    does *not* contain a real value for — currently only ever
    ``"event_queue_stream_cursors"`` when the caller omits
    ``event_queue_state``, or any caller-supplied reason via
    :func:`build_snapshot`'s ``not_captured`` parameter. An empty tuple is a
    positive claim ("everything asked for was captured"), never a default
    that happens to look that way.
    """

    schema_version: int
    run_id: str
    config_hash: str | None
    data_partition_hashes: dict[str, str | None]
    clock_now: float
    ledger_state: dict[str, Any]
    rng_states: dict[str, Any]
    event_queue_state: dict[str, Any]
    not_captured: tuple[str, ...] = ()

    # -- serialization ------------------------------------------------------

    def to_state(self) -> dict[str, Any]:
        """A plain, JSON-safe dict. The inverse of :meth:`from_state`."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "data_partition_hashes": dict(sorted(self.data_partition_hashes.items())),
            "clock_now": self.clock_now,
            "ledger_state": self.ledger_state,
            "rng_states": _json_safe(self.rng_states),
            "event_queue_state": _json_safe(self.event_queue_state),
            "not_captured": list(self.not_captured),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> ReplaySnapshot:
        """The inverse of :meth:`to_state`. Refuses a schema it does not
        recognise, and refuses a state missing the fields a snapshot cannot
        be resumed without — both raise rather than guess."""
        version = int(state.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise UnsupportedSchemaVersion(
                f"snapshot has schema_version {version}, this build reads "
                f"version {SCHEMA_VERSION} — move it aside rather than guess "
                "at its shape"
            )
        required = ("run_id", "clock_now", "ledger_state")
        missing = [key for key in required if key not in state]
        if missing:
            raise SnapshotCorrupt(
                f"snapshot is missing required field(s) {missing!r} — a valid "
                "snapshot must always carry these"
            )
        ledger_state = state["ledger_state"]
        if not isinstance(ledger_state, dict):
            raise SnapshotCorrupt("snapshot's ledger_state is not an object")
        return cls(
            schema_version=version,
            run_id=str(state["run_id"]),
            config_hash=state.get("config_hash"),
            data_partition_hashes=dict(state.get("data_partition_hashes") or {}),
            clock_now=float(state["clock_now"]),
            ledger_state=dict(ledger_state),
            rng_states=dict(state.get("rng_states") or {}),
            event_queue_state=dict(state.get("event_queue_state") or {}),
            not_captured=tuple(state.get("not_captured") or ()),
        )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def build_snapshot(
    *,
    clock: SimulatedClock,
    ledger: BacktestLedger,
    config_hash: str | None = None,
    data_partition_hashes: Mapping[str, str | None] | None = None,
    rng_states: Mapping[str, Any] | None = None,
    event_queue_state: Mapping[str, Any] | None = None,
    not_captured: Sequence[str] = (),
) -> ReplaySnapshot:
    """Capture ``ledger`` and ``clock`` into a :class:`ReplaySnapshot`.

    ``run_id`` is taken from ``clock.run_id``. If ``ledger.run_id`` is also
    set and disagrees, that is a caller bug (a ledger and a clock from two
    different runs being snapshotted together) and raises immediately rather
    than picking one silently.

    ``event_queue_state`` and ``rng_states`` are opaque, caller-supplied
    blobs — see the module docstring for why this module cannot derive them
    itself. Omitting ``event_queue_state`` records
    ``"event_queue_stream_cursors"`` in the returned snapshot's
    ``not_captured``; omitting ``rng_states`` entirely (or a particular
    stream name within it) is the caller's own claim about what it chose to
    persist, and is not auto-flagged here because a replay with no RNG use
    at all is a legitimate, fully-captured state.
    """
    if ledger.run_id and ledger.run_id != clock.run_id:
        raise SnapshotError(
            f"ledger.run_id={ledger.run_id!r} does not match "
            f"clock.run_id={clock.run_id!r} — refusing to build a snapshot "
            "that mixes state from two different runs"
        )
    resolved_not_captured = list(not_captured)
    resolved_event_queue_state: dict[str, Any]
    if event_queue_state is None:
        resolved_event_queue_state = {}
        resolved_not_captured.append(_EVENT_QUEUE_NOT_CAPTURED)
    else:
        resolved_event_queue_state = dict(event_queue_state)
    return ReplaySnapshot(
        schema_version=SCHEMA_VERSION,
        run_id=clock.run_id,
        config_hash=config_hash,
        data_partition_hashes=dict(data_partition_hashes) if data_partition_hashes else {},
        clock_now=clock.now,
        ledger_state=ledger.to_state(),
        rng_states=dict(rng_states) if rng_states else {},
        event_queue_state=resolved_event_queue_state,
        not_captured=tuple(resolved_not_captured),
    )


# ---------------------------------------------------------------------------
# Durable I/O
# ---------------------------------------------------------------------------


def save(path: Path, snapshot: ReplaySnapshot) -> None:
    """Write ``snapshot`` to ``path`` atomically.

    Delegates to ``journal.atomic_write_text``: temp file in the same
    directory, fsync, ``os.replace``. A reader of ``path`` never observes a
    partially-written file — either the previous snapshot (if any) or the
    complete new one. See the module docstring's "Durable, atomic writes"
    section for why a plain ``open().write()`` is not acceptable here.
    """
    text = json.dumps(snapshot.to_state(), sort_keys=True, ensure_ascii=False, indent=2)
    atomic_write_text(path, text + "\n")


def load(path: Path) -> ReplaySnapshot:
    """Read and validate a snapshot written by :func:`save`.

    Raises :class:`SnapshotCorrupt` for a missing file, invalid JSON (the
    signature of a torn/partial write that somehow reached the target path),
    or a JSON value that is not an object; raises
    :class:`UnsupportedSchemaVersion` for a recognised-but-unreadable schema.
    Never returns a partially-populated snapshot silently.
    """
    if not path.is_file():
        raise SnapshotCorrupt(f"no snapshot file at {path}")
    text = path.read_text(encoding="utf-8")
    try:
        state = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SnapshotCorrupt(
            f"snapshot at {path} is not valid JSON ({exc}) — this is the "
            "signature of a truncated or corrupted write, not a schema change"
        ) from exc
    if not isinstance(state, dict):
        raise SnapshotCorrupt(f"snapshot at {path} did not decode to a JSON object")
    return ReplaySnapshot.from_state(state)


# ---------------------------------------------------------------------------
# Resuming — the identity check that must never be silent
# ---------------------------------------------------------------------------


def verify_resume(
    snapshot: ReplaySnapshot,
    *,
    run_id: str,
    config_hash: str | None = None,
    data_partition_hashes: Mapping[str, str | None] | None = None,
) -> None:
    """Raise :class:`SnapshotMismatch` unless ``snapshot`` belongs to the run
    identified by ``run_id`` (and, when supplied, ``config_hash`` /
    ``data_partition_hashes``).

    ``run_id`` is always checked — a snapshot is meaningless outside the run
    it was taken from. ``config_hash``/``data_partition_hashes`` are checked
    only when the caller supplies them (``None`` means "not checked", not
    "matches anything"): a caller that has not yet computed its config/data
    hashes should not be able to accidentally pass validation by omission
    once it does compute them and stops passing ``None``.
    """
    problems: list[str] = []
    if snapshot.run_id != run_id:
        problems.append(f"run_id: snapshot={snapshot.run_id!r} resume={run_id!r}")
    if config_hash is not None and snapshot.config_hash != config_hash:
        problems.append(
            f"config_hash: snapshot={snapshot.config_hash!r} resume={config_hash!r}"
        )
    if data_partition_hashes is not None:
        expected = dict(data_partition_hashes)
        if expected != snapshot.data_partition_hashes:
            problems.append(
                f"data_partition_hashes: snapshot={snapshot.data_partition_hashes!r} "
                f"resume={expected!r}"
            )
    if problems:
        raise SnapshotMismatch(
            "refusing to resume: snapshot identity does not match the run "
            "being resumed (" + "; ".join(problems) + ") — this would be "
            "silent corruption if allowed to continue"
        )


def resume_ledger(
    snapshot: ReplaySnapshot,
    *,
    run_id: str,
    config_hash: str | None = None,
    data_partition_hashes: Mapping[str, str | None] | None = None,
) -> BacktestLedger:
    """Verify ``snapshot`` matches the run being resumed, then rebuild its
    :class:`~.ledger.BacktestLedger` via ``BacktestLedger.from_state``.

    This is the single entry point most callers want: it composes
    :func:`verify_resume` (raises :class:`SnapshotMismatch` on identity
    disagreement) with ``BacktestLedger.from_state`` (raises
    ``ledger.LedgerError`` on an unrecognised *ledger* schema version — a
    separate, independent check from this module's own
    ``schema_version``). Either failure aborts the resume rather than
    producing a ledger that looks valid but is not the one this run
    actually had.
    """
    verify_resume(
        snapshot,
        run_id=run_id,
        config_hash=config_hash,
        data_partition_hashes=data_partition_hashes,
    )
    return BacktestLedger.from_state(snapshot.ledger_state)
