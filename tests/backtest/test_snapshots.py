"""Tests for backtest/snapshots.py.

Each test here targets a specific failure mode a crash-restart layer can
have: a field that quietly rounds through a float on the way to disk, a
resume that continues against the wrong data, a torn write that still reads
back as "valid", an RNG stream that silently rewinds to its seed instead of
continuing. Every guard is written so that reverting it (in ``snapshots.py``)
makes the corresponding test fail.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from memetrader.backtest.clock import SimulatedClock
from memetrader.backtest.invariants import check_all
from memetrader.backtest.ledger import MICRO, BacktestLedger
from memetrader.backtest.snapshots import (
    ReplaySnapshot,
    SnapshotCorrupt,
    SnapshotMismatch,
    UnsupportedSchemaVersion,
    build_snapshot,
    load,
    restore_random_state,
    resume_ledger,
    rng_state_from_random,
    save,
    verify_resume,
)
from memetrader.types import (
    CostBreakdown,
    ExecutionReport,
    FidelityTier,
    Fill,
    OrderState,
    Side,
)

_SYMBOL = "BONK"
_RUN_ID = "run-snapshot-test"


# ---------------------------------------------------------------------------
# Shared fixtures — mirrors tests/accounting/test_ledger.py's helpers, kept
# local rather than imported so this test module has no dependency on
# another agent's test file.
# ---------------------------------------------------------------------------


def make_fill(
    *,
    fill_id: str,
    order_id: str,
    side: Side,
    ts: float,
    token_amount_atomic: int,
    notional_usd: float,
    symbol: str = _SYMBOL,
    token_decimals: int = 6,
    gas_usd: float = 0.0,
    pool_fee_usd: float = 0.0,
    state: OrderState = OrderState.LANDED,
    intent_id: str = "intent-1",
) -> Fill:
    usd_atomic = max(round(notional_usd * MICRO), 0)
    if side is Side.BUY:
        in_amount_atomic, out_amount_atomic = usd_atomic, token_amount_atomic
    else:
        in_amount_atomic, out_amount_atomic = token_amount_atomic, usd_atomic
    return Fill(
        fill_id=fill_id,
        order_id=order_id,
        intent_id=intent_id,
        decision_id=None,
        ts=ts,
        symbol=symbol,
        side=side,
        state=state,
        in_amount_atomic=in_amount_atomic,
        out_amount_atomic=out_amount_atomic,
        token_amount_atomic=token_amount_atomic,
        token_decimals=token_decimals,
        quote_fingerprint="fp",
        price_usd=None,
        notional_usd=notional_usd,
        price_impact_pct=0.0,
        pool_fee_usd=pool_fee_usd,
        gas_usd=gas_usd,
        realized_pnl_usd=0.0,
    )


def make_report(
    fill: Fill | None,
    *,
    report_id: str,
    intent_id: str = "intent-1",
    order_id: str | None = None,
    state: OrderState | None = None,
    ts: float | None = None,
    costs: CostBreakdown | None = None,
    reason: str = "",
) -> ExecutionReport:
    resolved_state = (
        state if state is not None else (fill.state if fill else OrderState.FAILED)
    )
    resolved_ts = ts if ts is not None else (fill.ts if fill else 0.0)
    resolved_order_id = (
        order_id if order_id is not None else (fill.order_id if fill else None)
    )
    return ExecutionReport(
        report_id=report_id,
        intent_id=intent_id,
        order_id=resolved_order_id,
        state=resolved_state,
        ts=resolved_ts,
        fidelity=FidelityTier.TIER_0,
        fill=fill,
        costs=costs,
        reason=reason,
    )


def _build_reports(ops: list[tuple[int, int, float, int]]) -> list[ExecutionReport]:
    """Turn a hypothesis-generated op list into a valid sequence of
    ExecutionReports: each BUY is followed by an optional SELL of a fraction
    of whatever is currently held, so no op sequence ever oversells. Mirrors
    tests/accounting/test_ledger.py's own hypothesis strategy construction.
    """
    reports: list[ExecutionReport] = []
    held = 0
    seq = 0
    for buy_qty, buy_notional_micro, sell_frac, sell_notional_micro in ops:
        seq += 1
        buy = make_fill(
            fill_id=f"buy-{seq}",
            order_id=f"o-{seq}",
            side=Side.BUY,
            ts=float(seq),
            token_amount_atomic=buy_qty,
            notional_usd=buy_notional_micro / MICRO,
        )
        reports.append(make_report(buy, report_id=f"r-buy-{seq}"))
        held += buy_qty

        sell_qty = int(held * sell_frac)
        if sell_qty > 0:
            seq += 1
            sell = make_fill(
                fill_id=f"sell-{seq}",
                order_id=f"o-{seq}",
                side=Side.SELL,
                ts=float(seq),
                token_amount_atomic=sell_qty,
                notional_usd=sell_notional_micro / MICRO,
            )
            reports.append(make_report(sell, report_id=f"r-sell-{seq}"))
            held -= sell_qty
    return reports


def _serialize_ledger_snapshot(snapshot: object) -> str:
    """Canonical string form for exact (not approx) comparison, mirroring
    invariants._serialize_snapshot but kept local to avoid depending on that
    module's private helper."""
    from memetrader.backtest.ledger import LedgerSnapshot

    assert isinstance(snapshot, LedgerSnapshot)
    positions = {
        symbol: {
            "mint": position.mint,
            "quantity_atomic": position.quantity_atomic,
            "decimals": position.decimals,
            "avg_entry_price_usd": position.avg_entry_price_usd,
            "opened_at": position.opened_at,
            "cost_basis_usd": position.cost_basis_usd,
        }
        for symbol, position in sorted(snapshot.positions.items())
    }
    payload = {
        "cash_micro_usd": snapshot.cash_micro_usd,
        "starting_cash_micro_usd": snapshot.starting_cash_micro_usd,
        "realized_pnl_micro_usd": snapshot.realized_pnl_micro_usd,
        "venue_fee_micro_usd": snapshot.venue_fee_micro_usd,
        "network_fee_micro_usd": snapshot.network_fee_micro_usd,
        "priority_fee_micro_usd": snapshot.priority_fee_micro_usd,
        "positions": positions,
    }
    return json.dumps(payload, sort_keys=True)


def _economic_entries(entries: tuple) -> tuple:
    """The economically meaningful projection of a ledger's entry log:
    every field except `seq` (an internal counter, not an economic fact) and
    `lot_created`/`lots_consumed` (lot *label* identity, which — see the
    module docstring's "Known gap" note this test documents — is not
    guaranteed stable across a restart when a lot fully closes before the
    snapshot is taken, even though no dollar or token amount is affected).
    """
    return tuple(
        (
            entry.report_id,
            entry.kind,
            entry.ts,
            entry.symbol,
            entry.fill_id,
            entry.order_id,
            entry.cash_delta_micro_usd,
            entry.realized_pnl_delta_micro_usd,
            entry.quantity_delta_atomic,
            entry.venue_fee_micro_usd,
            entry.network_fee_micro_usd,
            entry.priority_fee_micro_usd,
        )
        for entry in entries
    )


def _make_ledger_with_fills() -> BacktestLedger:
    ledger = BacktestLedger(starting_cash_micro_usd=1_000 * MICRO, run_id=_RUN_ID)
    buy1 = make_fill(
        fill_id="f1",
        order_id="o1",
        side=Side.BUY,
        ts=1.0,
        token_amount_atomic=1_000_000,
        notional_usd=10.0,
    )
    buy2 = make_fill(
        fill_id="f2",
        order_id="o2",
        side=Side.BUY,
        ts=2.0,
        token_amount_atomic=500_000,
        notional_usd=6.0,
    )
    sell = make_fill(
        fill_id="f3",
        order_id="o3",
        side=Side.SELL,
        ts=3.0,
        token_amount_atomic=800_000,
        notional_usd=9.0,
    )
    ledger.apply_fill(make_report(buy1, report_id="r1"))
    ledger.apply_fill(make_report(buy2, report_id="r2"))
    ledger.apply_fill(make_report(sell, report_id="r3"))
    return ledger


# ---------------------------------------------------------------------------
# Round-trip equality
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_save_load_round_trips_every_field_exactly(self, tmp_path) -> None:
        clock = SimulatedClock(run_id=_RUN_ID, start=42.0)
        ledger = _make_ledger_with_fills()
        snapshot = build_snapshot(
            clock=clock,
            ledger=ledger,
            config_hash="cfg-hash-1",
            data_partition_hashes={"train": "h1", "val": "h2"},
            rng_states={"fills": [1, 2, 3]},
            event_queue_state={"stream_0": 17},
        )
        path = tmp_path / "snap.json"
        save(path, snapshot)
        loaded = load(path)

        assert loaded == snapshot
        assert loaded.run_id == _RUN_ID
        assert loaded.config_hash == "cfg-hash-1"
        assert loaded.data_partition_hashes == {"train": "h1", "val": "h2"}
        assert loaded.clock_now == 42.0
        assert loaded.rng_states == {"fills": [1, 2, 3]}
        assert loaded.event_queue_state == {"stream_0": 17}
        assert loaded.not_captured == ()

        restored = BacktestLedger.from_state(loaded.ledger_state)
        assert restored.cash_micro_usd == ledger.cash_micro_usd
        assert isinstance(restored.cash_micro_usd, int)
        assert restored.realized_pnl_micro_usd == ledger.realized_pnl_micro_usd
        assert restored.entries() == ledger.entries()
        assert restored.lot_views(_SYMBOL) == ledger.lot_views(_SYMBOL)

    def test_omitting_event_queue_state_is_recorded_as_not_captured(self, tmp_path) -> None:
        clock = SimulatedClock(run_id=_RUN_ID, start=0.0)
        ledger = BacktestLedger(starting_cash_micro_usd=100 * MICRO, run_id=_RUN_ID)
        snapshot = build_snapshot(clock=clock, ledger=ledger)
        assert "event_queue_stream_cursors" in snapshot.not_captured
        assert snapshot.event_queue_state == {}

        path = tmp_path / "snap.json"
        save(path, snapshot)
        loaded = load(path)
        assert "event_queue_stream_cursors" in loaded.not_captured


# ---------------------------------------------------------------------------
# Headline test: crash-restart equivalence
# ---------------------------------------------------------------------------

_BUY_TOKEN_ATOMIC = st.integers(min_value=1, max_value=10_000_000)
_BUY_NOTIONAL_MICRO = st.integers(min_value=1, max_value=1_000_000)
_SELL_FRACTION = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)
_SELL_NOTIONAL_MICRO = st.integers(min_value=0, max_value=2_000_000)


class TestCrashRestartEquivalence:
    @given(
        st.lists(
            st.tuples(
                _BUY_TOKEN_ATOMIC, _BUY_NOTIONAL_MICRO, _SELL_FRACTION, _SELL_NOTIONAL_MICRO
            ),
            min_size=2,
            max_size=20,
        )
    )
    @settings(max_examples=60)
    def test_snapshot_restore_matches_uninterrupted_run_exactly(
        self, ops: list[tuple[int, int, float, int]]
    ) -> None:
        starting_cash = 10**15
        reports = _build_reports(ops)
        mid = max(1, len(reports) // 2)

        # Uninterrupted run: apply everything to one ledger, never stopping.
        full_ledger = BacktestLedger(starting_cash_micro_usd=starting_cash, run_id=_RUN_ID)
        for report in reports:
            full_ledger.apply_fill(report)

        # Interrupted run: apply the first half, snapshot, save, load, resume,
        # then apply the rest to the *restored* ledger.
        partial_ledger = BacktestLedger(
            starting_cash_micro_usd=starting_cash, run_id=_RUN_ID
        )
        for report in reports[:mid]:
            partial_ledger.apply_fill(report)

        clock = SimulatedClock(run_id=_RUN_ID, start=0.0)
        snapshot = build_snapshot(
            clock=clock,
            ledger=partial_ledger,
            config_hash="cfg",
            data_partition_hashes={"train": "h1"},
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / f"snap-{len(reports)}.json"
            save(path, snapshot)
            loaded = load(path)
        restored_ledger = resume_ledger(
            loaded, run_id=_RUN_ID, config_hash="cfg", data_partition_hashes={"train": "h1"}
        )
        for report in reports[mid:]:
            restored_ledger.apply_fill(report)

        # Exact equality, not approximate, of every economic quantity: the
        # LedgerSnapshot (cash, positions, realized PnL, fee totals) that
        # portfolio.mark_book and every downstream report actually consume.
        assert _serialize_ledger_snapshot(
            restored_ledger.snapshot()
        ) == _serialize_ledger_snapshot(full_ledger.snapshot())
        assert restored_ledger.cash_micro_usd == full_ledger.cash_micro_usd
        assert restored_ledger.realized_pnl_micro_usd == full_ledger.realized_pnl_micro_usd

        # The entry log's economically meaningful fields also match exactly,
        # entry for entry. `lot_created`/`lots_consumed` are deliberately
        # excluded from this comparison — see the note below and the module
        # docstring's "Known gap" callout for why those two fields are not
        # guaranteed byte-identical across a restart.
        assert _economic_entries(restored_ledger.entries()) == _economic_entries(
            full_ledger.entries()
        )

        # And the restored ledger is still a fully law-abiding ledger.
        check_all(restored_ledger)

        # --- Known gap, stated loudly rather than buried -------------------
        # BacktestLedger.to_state() only serializes *currently open* lots
        # (`if dq` in its "lots" dict comprehension), and from_state()
        # reconstructs `_lot_seq_counter` solely from the lot_ids present in
        # that serialized set. A lot that was opened and then fully consumed
        # before the snapshot was taken leaves no trace of the sequence
        # number it used, so after a restart the counter can restart lower
        # than it would have in an uninterrupted run and reissue a lot_id
        # string an earlier, already-closed lot also used. This is purely a
        # label collision — no cash, quantity, or PnL is affected, and
        # invariants.check_fill_lot_integrity still passes because it only
        # requires a *consuming* SELL's lot_id to have been created by some
        # earlier BUY in the same (now-restored) ledger's own log, which it
        # was. It does mean `ledger.entries()` and `ledger.lot_views()` are
        # NOT guaranteed byte-identical to an uninterrupted run's when a lot
        # fully closes before a snapshot; `ledger.py` is out of scope for
        # this assignment (frozen file), so this is recorded as a finding
        # rather than patched here.


# ---------------------------------------------------------------------------
# RNG continuity
# ---------------------------------------------------------------------------


class TestRngContinuity:
    def test_restored_rng_draws_match_uninterrupted_stream_exactly(self, tmp_path) -> None:
        clock = SimulatedClock(run_id=_RUN_ID, start=0.0)
        seed = clock.deterministic_seed("fills")

        uninterrupted = random.Random(seed)
        full_sequence = [uninterrupted.random() for _ in range(20)]

        # Interrupted: draw the first half from a fresh RNG seeded the same
        # way, snapshot its *consumed* state, restore it, and draw the rest.
        live = random.Random(seed)
        first_half = [live.random() for _ in range(10)]

        ledger = BacktestLedger(starting_cash_micro_usd=100 * MICRO, run_id=_RUN_ID)
        snapshot = build_snapshot(
            clock=clock,
            ledger=ledger,
            rng_states={"fills": rng_state_from_random(live)},
        )
        path = tmp_path / "rng-snap.json"
        save(path, snapshot)
        loaded = load(path)

        restored_rng = random.Random()
        restore_random_state(restored_rng, loaded.rng_states["fills"])
        second_half = [restored_rng.random() for _ in range(10)]

        assert first_half + second_half == full_sequence

    def test_reseeding_from_seed_alone_would_diverge(self) -> None:
        """The negative case that motivates capturing consumed state at all:
        reseeding fresh from the same deterministic seed after some draws
        have already happened does NOT reproduce the continuation — proving
        that persisting only the seed (and not the state) is insufficient,
        which is exactly what the module docstring warns about."""
        clock = SimulatedClock(run_id=_RUN_ID, start=0.0)
        seed = clock.deterministic_seed("fills")

        uninterrupted = random.Random(seed)
        full_sequence = [uninterrupted.random() for _ in range(20)]

        live = random.Random(seed)
        first_half = [live.random() for _ in range(10)]
        # Wrong approach: reseed fresh from the deterministic seed instead of
        # restoring consumed state.
        reseeded = random.Random(seed)
        wrong_second_half = [reseeded.random() for _ in range(10)]

        assert first_half + wrong_second_half != full_sequence


# ---------------------------------------------------------------------------
# Identity mismatch must raise
# ---------------------------------------------------------------------------


class TestIdentityMismatch:
    def _snapshot(self, tmp_path) -> ReplaySnapshot:
        clock = SimulatedClock(run_id=_RUN_ID, start=0.0)
        ledger = BacktestLedger(starting_cash_micro_usd=100 * MICRO, run_id=_RUN_ID)
        return build_snapshot(
            clock=clock,
            ledger=ledger,
            config_hash="cfg-1",
            data_partition_hashes={"train": "h1"},
        )

    def test_mismatched_run_id_raises(self, tmp_path) -> None:
        snapshot = self._snapshot(tmp_path)
        with pytest.raises(SnapshotMismatch):
            verify_resume(snapshot, run_id="a-different-run")

    def test_mismatched_config_hash_raises(self, tmp_path) -> None:
        snapshot = self._snapshot(tmp_path)
        with pytest.raises(SnapshotMismatch):
            verify_resume(snapshot, run_id=_RUN_ID, config_hash="cfg-2")

    def test_mismatched_data_partition_hash_raises(self, tmp_path) -> None:
        snapshot = self._snapshot(tmp_path)
        with pytest.raises(SnapshotMismatch):
            verify_resume(
                snapshot,
                run_id=_RUN_ID,
                data_partition_hashes={"train": "different-hash"},
            )

    def test_matching_identity_does_not_raise(self, tmp_path) -> None:
        snapshot = self._snapshot(tmp_path)
        verify_resume(
            snapshot,
            run_id=_RUN_ID,
            config_hash="cfg-1",
            data_partition_hashes={"train": "h1"},
        )

    def test_resume_ledger_raises_on_mismatch_and_does_not_return_a_ledger(
        self, tmp_path
    ) -> None:
        snapshot = self._snapshot(tmp_path)
        with pytest.raises(SnapshotMismatch):
            resume_ledger(snapshot, run_id="wrong-run")


# ---------------------------------------------------------------------------
# Schema version refusal
# ---------------------------------------------------------------------------


class TestSchemaVersion:
    def test_unknown_schema_version_is_refused(self, tmp_path) -> None:
        path = tmp_path / "future.json"
        path.write_text(
            json.dumps({"schema_version": 999, "run_id": "x"}), encoding="utf-8"
        )
        with pytest.raises(UnsupportedSchemaVersion):
            load(path)

    def test_from_state_refuses_unknown_schema_version_directly(self) -> None:
        with pytest.raises(UnsupportedSchemaVersion):
            ReplaySnapshot.from_state({"schema_version": 2})


# ---------------------------------------------------------------------------
# Corruption is detected, not silently accepted
# ---------------------------------------------------------------------------


class TestCorruption:
    def test_truncated_file_is_detected(self, tmp_path) -> None:
        clock = SimulatedClock(run_id=_RUN_ID, start=0.0)
        ledger = BacktestLedger(starting_cash_micro_usd=100 * MICRO, run_id=_RUN_ID)
        snapshot = build_snapshot(clock=clock, ledger=ledger)
        path = tmp_path / "snap.json"
        save(path, snapshot)

        full_text = path.read_text(encoding="utf-8")
        truncated = full_text[: len(full_text) // 2]
        path.write_text(truncated, encoding="utf-8")

        with pytest.raises(SnapshotCorrupt):
            load(path)

    def test_missing_file_is_detected(self, tmp_path) -> None:
        with pytest.raises(SnapshotCorrupt):
            load(tmp_path / "does-not-exist.json")

    def test_non_object_json_is_detected(self, tmp_path) -> None:
        path = tmp_path / "snap.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(SnapshotCorrupt):
            load(path)

    def test_missing_required_field_is_detected(self, tmp_path) -> None:
        path = tmp_path / "snap.json"
        path.write_text(json.dumps({"schema_version": 1, "run_id": "x"}), encoding="utf-8")
        with pytest.raises(SnapshotCorrupt):
            load(path)


# ---------------------------------------------------------------------------
# Atomic write: a partial write must not leave a loadable file
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_failure_mid_write_leaves_no_file_at_the_target_path(
        self, tmp_path, monkeypatch
    ) -> None:
        clock = SimulatedClock(run_id=_RUN_ID, start=0.0)
        ledger = BacktestLedger(starting_cash_micro_usd=100 * MICRO, run_id=_RUN_ID)
        snapshot = build_snapshot(clock=clock, ledger=ledger)
        path = tmp_path / "snap.json"
        assert not path.exists()

        def boom(*_args: object, **_kwargs: object) -> None:
            raise OSError("simulated crash mid-write")

        monkeypatch.setattr(os, "fsync", boom)
        with pytest.raises(OSError):
            save(path, snapshot)

        # No file at the target path — a crash never leaves a half-written
        # snapshot that reads back as valid.
        assert not path.exists()
        # And no stray temp file left behind in the directory either.
        assert list(tmp_path.iterdir()) == []

    def test_failure_mid_overwrite_preserves_the_previous_snapshot(
        self, tmp_path, monkeypatch
    ) -> None:
        clock = SimulatedClock(run_id=_RUN_ID, start=0.0)
        ledger = BacktestLedger(starting_cash_micro_usd=100 * MICRO, run_id=_RUN_ID)
        first_snapshot = build_snapshot(clock=clock, ledger=ledger, config_hash="first")
        path = tmp_path / "snap.json"
        save(path, first_snapshot)
        original_bytes = path.read_bytes()

        second_ledger = _make_ledger_with_fills()
        second_snapshot = build_snapshot(
            clock=clock, ledger=second_ledger, config_hash="second"
        )

        def boom(*_args: object, **_kwargs: object) -> None:
            raise OSError("simulated crash mid-write")

        monkeypatch.setattr(os, "fsync", boom)
        with pytest.raises(OSError):
            save(path, second_snapshot)

        # The previous, complete snapshot is exactly intact.
        assert path.read_bytes() == original_bytes
        reloaded = load(path)
        assert reloaded.config_hash == "first"


# ---------------------------------------------------------------------------
# invariants.check_all passes on a restored ledger
# ---------------------------------------------------------------------------


class TestInvariantsOnRestoredLedger:
    def test_check_all_passes_after_save_load_resume(self, tmp_path) -> None:
        clock = SimulatedClock(run_id=_RUN_ID, start=0.0)
        ledger = _make_ledger_with_fills()
        snapshot = build_snapshot(
            clock=clock, ledger=ledger, config_hash="cfg", data_partition_hashes={}
        )
        path = tmp_path / "snap.json"
        save(path, snapshot)
        loaded = load(path)
        restored = resume_ledger(
            loaded, run_id=_RUN_ID, config_hash="cfg", data_partition_hashes={}
        )
        check_all(restored)
