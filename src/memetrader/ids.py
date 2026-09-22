"""Immutable identifiers for every object in the decision-to-fill chain.

The audit's C11 and the `prompts._decision_line` finding share one root cause:
nothing in the old system had an identity. Fills were matched to actions by
symbol, so a stop-loss and a model SELL on the same coin in the same tick were
indistinguishable in the journal; and a crash between the trade append and the
state save left two files with no way to tell which rows belonged together.

Every record now carries an ID minted here. Two properties matter:

* **Ordering.** IDs are prefixed with a zero-padded microsecond timestamp, so
  lexical sort equals chronological sort. That makes a JSONL ledger sortable
  without parsing it.
* **Uniqueness without coordination.** A short random suffix means two
  processes that should never have been running at once still cannot silently
  produce colliding IDs — the duplicate shows up as two distinct rows rather
  than as one row overwriting another.

They are deliberately *not* content hashes. A content hash of an intent would
collide whenever the same order is legitimately placed twice, which is exactly
the case idempotency has to be able to distinguish. Content hashing is used
separately, by :func:`quote_fingerprint`, where equality *is* the question.
"""

from __future__ import annotations

import hashlib
import os
import time

__all__ = [
    "ids_state",
    "new_action_id",
    "new_decision_id",
    "new_fill_id",
    "new_intent_id",
    "new_order_id",
    "new_run_id",
    "quote_fingerprint",
    "restore_ids_state",
]

# ---------------------------------------------------------------------------
# Replay-aware minting
# ---------------------------------------------------------------------------
#
# Live, both halves of an ID come from the environment: the timestamp from
# ``time.time()`` and the suffix from ``os.urandom``.  Under replay neither is
# usable.  ``time.time()`` *raises* while a SimulatedClock is active (that is
# the whole point of its wall-clock guard), and ``os.urandom`` would make two
# replays of the same data produce different IDs — breaking the deterministic
# replay invariant in BACKTEST-CONTRACTS.md §6, which requires two runs over
# identical inputs to be comparable record by record.
#
# So under replay we substitute both: simulated time for the timestamp, and a
# per-run monotonic counter for the suffix.  The counter is what keeps IDs
# unique when several intents are minted inside a single simulated instant,
# which is the common case — a tick emits all its orders at one ``now``.
#
# The counter resets when the active clock instance changes, so two runs in
# one process each start from zero and produce identical ID sequences.  It is
# exposed via ids_state()/restore_ids_state() so a crash-restart snapshot can
# carry it across a resume; without that, a resumed run would restart the
# counter and re-issue IDs it had already used.

_sim_clock: object | None = None
_sim_seq: int = 0


def ids_state() -> dict[str, object]:
    """Capture the replay ID counter, for inclusion in a restart snapshot.

    Returns ``{}`` when no replay is active, because there is nothing to carry
    — live IDs are environment-derived and do not need to be resumed.
    """
    from memetrader.backtest.clock import active_clock

    # Ask the clock, not our own ``_sim_clock``: that global holds the last
    # replay we minted under and is deliberately not cleared on clock exit, so
    # consulting it here would report a finished replay's counter as live state.
    clock = active_clock()
    if clock is None:
        return {}
    return {"run_id": clock.run_id, "seq": _sim_seq if clock is _sim_clock else 0}


def restore_ids_state(state: dict[str, object]) -> None:
    """Restore a counter captured by :func:`ids_state`.

    Must be called *after* the SimulatedClock is active: the restore binds
    itself to the clock that is running, so the next mint continues the
    sequence instead of treating a fresh clock instance as a fresh run and
    resetting to zero.
    """
    global _sim_clock, _sim_seq
    from memetrader.backtest.clock import active_clock

    seq = state.get("seq", 0)
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        raise ValueError(f"ids state 'seq' must be a non-negative int, got {seq!r}")
    clock = active_clock()
    if clock is None:
        raise RuntimeError(
            "restore_ids_state() requires an active SimulatedClock — restoring "
            "with no replay running would be silently discarded by the next mint."
        )
    _sim_clock = clock
    _sim_seq = seq


def _next_sim_seq(clock: object) -> int:
    global _sim_clock, _sim_seq
    if clock is not _sim_clock:
        # A different replay: start its ID sequence from zero so the run is
        # reproducible regardless of what ran before it in this process.
        _sim_clock = clock
        _sim_seq = 0
    seq = _sim_seq
    _sim_seq = seq + 1
    return seq


def _mint(prefix: str) -> str:
    # Imported lazily: ids.py sits below the backtest package, and importing it
    # at module scope would invert that layering for the benefit of a branch
    # that only matters during replay.
    from memetrader.backtest.clock import active_clock

    clock = active_clock()
    if clock is not None:
        seq = _next_sim_seq(clock)
        micros = int(clock.now * 1_000_000)
        # blake2b, not urandom: the suffix must be a deterministic function of
        # (run, prefix, sequence) so the same replay mints the same IDs twice.
        key = f"{clock.run_id}|{prefix}|{seq}"
        suffix = hashlib.blake2b(key.encode("utf-8"), digest_size=4).hexdigest()
        return f"{prefix}-{micros:018d}-{suffix}"

    # Microseconds, not seconds: a slow tick can emit several intents inside one
    # second and they must still sort in the order they were created.
    micros = int(time.time() * 1_000_000)
    suffix = os.urandom(4).hex()
    return f"{prefix}-{micros:018d}-{suffix}"


def new_run_id() -> str:
    """One per process start. Every record made by that process carries it, so a
    duplicate-process incident is visible as two run IDs interleaved in one
    ledger rather than as inexplicable state."""
    return _mint("run")


def new_decision_id() -> str:
    """One per slow tick that reached a strategy."""
    return _mint("dec")


def new_action_id() -> str:
    """One per per-symbol action inside a decision."""
    return _mint("act")


def new_intent_id() -> str:
    """One per order the execution layer intends to place.

    This is the idempotency key. It is minted *before* anything is quoted or
    submitted and persisted before any side effect, so a crash mid-submission
    leaves a row that recovery can find and reconcile rather than an orphan.
    """
    return _mint("int")


def new_order_id() -> str:
    """One per quote-bound order derived from an intent."""
    return _mint("ord")


def new_fill_id() -> str:
    """One per settled (or failed) execution attempt."""
    return _mint("fil")


def quote_fingerprint(
    *,
    side: str,
    input_mint: str,
    output_mint: str,
    in_amount_atomic: int,
    out_amount_atomic: int,
    slot: int | None,
) -> str:
    """A content hash binding a quote to the exact swap it describes.

    C3 in the audit: risk could shrink an order after it was quoted, and the
    broker then executed the reduced notional against the original-size quote.
    Binding is the fix — an order carries the fingerprint of the quote it was
    built from, and the broker recomputes it from the quote it was handed. If
    they differ, the order is refused rather than filled at a price that was
    never offered for that size.

    Every economically load-bearing field of the swap is in the digest. The
    amounts are the atomic integers, not dollars, because dollars are a derived
    presentation of the swap and two different swaps can round to the same one.
    """
    payload = "|".join(
        (
            side,
            input_mint,
            output_mint,
            str(in_amount_atomic),
            str(out_amount_atomic),
            "none" if slot is None else str(slot),
        )
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()
