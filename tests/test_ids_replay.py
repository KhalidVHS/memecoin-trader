"""Tests for :mod:`memetrader.ids` under an active SimulatedClock.

Live, an ID is environment-derived: wall-clock microseconds plus random bytes.
Neither is available during a replay — ``time.time()`` raises under the clock
guard, and random bytes would make two replays of identical data produce
different IDs. These tests pin the substituted behaviour.
"""

from __future__ import annotations

import pytest

from memetrader import ids
from memetrader.backtest.clock import SimulatedClock


def _mint_all() -> list[str]:
    return [
        ids.new_run_id(),
        ids.new_decision_id(),
        ids.new_action_id(),
        ids.new_intent_id(),
        ids.new_order_id(),
        ids.new_fill_id(),
    ]


def test_minting_works_under_the_clock_guard() -> None:
    """Guard: minting an ID during a replay does not raise.

    Before the replay-aware branch existed, every one of these called
    ``time.time()`` and died with WallClockAccessError, which would have taken
    down the engine the first time it tried to identify an order.
    """
    with SimulatedClock(run_id="mint-live", start=1_700_000_000.0):
        minted = _mint_all()
    assert len(minted) == 6
    assert all(isinstance(value, str) and value for value in minted)


def test_ids_are_reproducible_across_runs() -> None:
    """Two replays with the same run_id mint byte-identical IDs.

    This is what makes two runs comparable record by record (contracts §6
    deterministic replay). If IDs differed between runs, every downstream
    diff would report every row as changed.
    """
    with SimulatedClock(run_id="repro", start=1_700_000_000.0):
        first = _mint_all()
    with SimulatedClock(run_id="repro", start=1_700_000_000.0):
        second = _mint_all()
    assert first == second


def test_ids_differ_across_run_ids() -> None:
    """Different runs must not collide, or one run's ledger could be read as
    another's."""
    with SimulatedClock(run_id="run-A", start=1_700_000_000.0):
        a = _mint_all()
    with SimulatedClock(run_id="run-B", start=1_700_000_000.0):
        b = _mint_all()
    assert set(a).isdisjoint(b)


def test_ids_unique_within_one_simulated_instant() -> None:
    """Guard: the per-run counter, not the timestamp, is what separates IDs.

    A tick emits all its orders at a single ``now``. Without the counter every
    intent in that tick would share a timestamp *and* a suffix, and the
    idempotency key would stop being a key.
    """
    with SimulatedClock(run_id="same-instant", start=1_700_000_000.0):
        minted = [ids.new_intent_id() for _ in range(64)]
    assert len(set(minted)) == 64


def test_ids_embed_simulated_time_not_wall_time() -> None:
    """The timestamp segment comes from the clock, so IDs sort in simulated
    order rather than in the order the replay happened to execute."""
    start = 1_700_000_000.0
    with SimulatedClock(run_id="simtime", start=start) as clock:
        early = ids.new_intent_id()
        clock.advance_to(start + 3_600.0)
        late = ids.new_intent_id()
    assert early.split("-")[1] == f"{int(start * 1_000_000):018d}"
    assert late.split("-")[1] == f"{int((start + 3_600.0) * 1_000_000):018d}"
    assert early < late, "lexical order must equal chronological order"


def test_live_ids_still_random_after_replay_exits() -> None:
    """Leaving a replay restores environment-derived minting.

    A counter that leaked into live trading would make IDs predictable and, far
    worse, would restart at zero on every process start.
    """
    with SimulatedClock(run_id="leak-check", start=1_700_000_000.0):
        ids.new_intent_id()
    live = {ids.new_intent_id() for _ in range(32)}
    assert len(live) == 32


def test_counter_survives_a_snapshot_restore() -> None:
    """Guard: a resumed run does not re-issue IDs it already used.

    ``ids_state``/``restore_ids_state`` exist so a crash-restart snapshot can
    carry the counter. Without the restore, the resumed half restarts at zero
    and mints duplicates of the first half's IDs.
    """
    start = 1_700_000_000.0
    with SimulatedClock(run_id="resume", start=start):
        before = [ids.new_intent_id() for _ in range(5)]
        captured = ids.ids_state()

    # A fresh clock instance stands in for the restarted process.
    with SimulatedClock(run_id="resume", start=start):
        ids.restore_ids_state(captured)
        after = [ids.new_intent_id() for _ in range(5)]

    assert set(before).isdisjoint(after)


def test_counter_without_restore_collides() -> None:
    """The negative half of the test above.

    Stated explicitly so the value of restoring is demonstrated, not assumed:
    resuming *without* the captured counter reproduces the earlier IDs.
    """
    start = 1_700_000_000.0
    with SimulatedClock(run_id="no-restore", start=start):
        before = [ids.new_intent_id() for _ in range(5)]
    with SimulatedClock(run_id="no-restore", start=start):
        after = [ids.new_intent_id() for _ in range(5)]
    assert before == after


def test_ids_state_is_empty_when_live() -> None:
    """Nothing to carry when no replay is active."""
    assert ids.ids_state() == {}


def test_restore_rejects_a_negative_sequence() -> None:
    with (
        SimulatedClock(run_id="bad-seq", start=0.0),
        pytest.raises(ValueError, match="non-negative"),
    ):
        ids.restore_ids_state({"seq": -1})


def test_restore_without_an_active_clock_raises() -> None:
    """Guard: restoring outside a replay would be silently discarded.

    The next mint would see no active clock, take the live branch, and the
    caller would never learn their resume state went nowhere.
    """
    with pytest.raises(RuntimeError, match="active SimulatedClock"):
        ids.restore_ids_state({"seq": 5})


def test_quote_fingerprint_is_unaffected_by_the_clock() -> None:
    """Guard: the fingerprint is a pure content hash.

    It must not acquire a time or run dependency — its whole job is to answer
    "is this the same swap?", which cannot depend on when it was asked.
    """
    kwargs = {
        "side": "buy",
        "input_mint": "So11111111111111111111111111111111111111112",
        "output_mint": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
        "in_amount_atomic": 1_000_000_000,
        "out_amount_atomic": 4_200_000_000,
        "slot": 123,
    }
    live = ids.quote_fingerprint(**kwargs)  # type: ignore[arg-type]
    with SimulatedClock(run_id="fingerprint", start=0.0):
        replayed = ids.quote_fingerprint(**kwargs)  # type: ignore[arg-type]
    assert live == replayed
