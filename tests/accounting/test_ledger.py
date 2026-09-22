"""Tests for backtest/ledger.py.

Every property test here exists to catch a specific class of bug in the
accounting core: money or token quantities silently leaking into floats,
a SELL partially applying before it discovers it exceeds inventory, a fee
charged twice, or a duplicate report changing the book. Each test is
written so that reverting the guard it targets (in ``ledger.py``) makes it
fail — a passing suite with the guard removed would mean the test is
decorative, and none of these are meant to be.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from memetrader.backtest.invariants import check_all
from memetrader.backtest.ledger import (
    MICRO,
    BacktestLedger,
    InsufficientLots,
    LedgerError,
    NegativeCash,
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
    """Build a settled Fill. The in/out atomic legs are derived so the Fill's
    own validators are satisfied; the ledger only ever reads
    ``notional_usd``/``token_amount_atomic``/``gas_usd``/``pool_fee_usd``/
    ``side``/``state``/``symbol``/``token_decimals`` from it.
    """
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


def make_failed_fill(
    *, fill_id: str, order_id: str, ts: float, gas_usd: float, symbol: str = _SYMBOL
) -> Fill:
    """A failed attempt: zero amounts, non-zero gas — ``types.Fill``'s own
    documented shape for a failed row."""
    return Fill(
        fill_id=fill_id,
        order_id=order_id,
        intent_id="intent-1",
        decision_id=None,
        ts=ts,
        symbol=symbol,
        side=Side.BUY,
        state=OrderState.FAILED,
        in_amount_atomic=0,
        out_amount_atomic=0,
        token_amount_atomic=0,
        token_decimals=6,
        quote_fingerprint="fp",
        price_usd=None,
        notional_usd=0.0,
        price_impact_pct=None,
        pool_fee_usd=0.0,
        gas_usd=gas_usd,
        realized_pnl_usd=0.0,
        note="transaction failed",
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


class TestBuyAndSellBasics:
    def test_buy_creates_lot_and_moves_cash(self) -> None:
        ledger = BacktestLedger(starting_cash_micro_usd=100 * MICRO)
        fill = make_fill(
            fill_id="f1",
            order_id="o1",
            side=Side.BUY,
            ts=1.0,
            token_amount_atomic=1_000_000,
            notional_usd=10.0,
        )
        applied = ledger.apply_fill(make_report(fill, report_id="r1"))
        assert applied is True
        assert ledger.cash_micro_usd == 90 * MICRO
        lots = ledger.lot_views(_SYMBOL)
        assert len(lots) == 1
        assert lots[0].quantity_atomic == 1_000_000
        assert lots[0].basis_micro_usd == 10 * MICRO
        assert lots[0].fill_id == "f1"

    def test_sell_realizes_pnl_and_releases_basis(self) -> None:
        ledger = BacktestLedger(starting_cash_micro_usd=100 * MICRO)
        buy = make_fill(
            fill_id="f1",
            order_id="o1",
            side=Side.BUY,
            ts=1.0,
            token_amount_atomic=1_000_000,
            notional_usd=10.0,
        )
        ledger.apply_fill(make_report(buy, report_id="r1"))
        sell = make_fill(
            fill_id="f2",
            order_id="o2",
            side=Side.SELL,
            ts=2.0,
            token_amount_atomic=1_000_000,
            notional_usd=15.0,
        )
        ledger.apply_fill(make_report(sell, report_id="r2"))
        assert ledger.cash_micro_usd == 105 * MICRO
        assert ledger.realized_pnl_micro_usd == 5 * MICRO
        assert ledger.lot_views(_SYMBOL) == ()

    def test_fees_are_charged_exactly_once_and_split_by_category(self) -> None:
        ledger = BacktestLedger(starting_cash_micro_usd=100 * MICRO)
        costs = CostBreakdown(venue_fee_usd=0.5, network_fee_usd=0.25, priority_fee_usd=0.1)
        fill = make_fill(
            fill_id="f1",
            order_id="o1",
            side=Side.BUY,
            ts=1.0,
            token_amount_atomic=1_000_000,
            notional_usd=10.0,
        )
        ledger.apply_fill(make_report(fill, report_id="r1", costs=costs))
        assert ledger.venue_fee_micro_usd == round(0.5 * MICRO)
        assert ledger.network_fee_micro_usd == round(0.25 * MICRO)
        assert ledger.priority_fee_micro_usd == round(0.1 * MICRO)
        total_cost = 10.0 + 0.5 + 0.25 + 0.1
        assert ledger.cash_micro_usd == 100 * MICRO - round(total_cost * MICRO)
        # Fee reconciliation: sum of per-entry fees equals the running totals,
        # exactly. This must fail if a fee is ever double-counted or dropped.
        entries = ledger.entries()
        assert sum(e.venue_fee_micro_usd for e in entries) == ledger.venue_fee_micro_usd
        assert sum(e.network_fee_micro_usd for e in entries) == ledger.network_fee_micro_usd
        assert (
            sum(e.priority_fee_micro_usd for e in entries) == ledger.priority_fee_micro_usd
        )

    def test_fee_fallback_uses_fill_pool_fee_and_gas_when_no_costs_given(self) -> None:
        ledger = BacktestLedger(starting_cash_micro_usd=100 * MICRO)
        fill = make_fill(
            fill_id="f1",
            order_id="o1",
            side=Side.BUY,
            ts=1.0,
            token_amount_atomic=1_000_000,
            notional_usd=10.0,
            pool_fee_usd=0.3,
            gas_usd=0.05,
        )
        ledger.apply_fill(make_report(fill, report_id="r1", costs=None))
        assert ledger.venue_fee_micro_usd == round(0.3 * MICRO)
        assert ledger.network_fee_micro_usd == round(0.05 * MICRO)
        assert ledger.priority_fee_micro_usd == 0


class TestRealizedVsUnrealizedSplit:
    def test_partial_sell_splits_realized_and_unrealized_correctly(self) -> None:
        ledger = BacktestLedger(starting_cash_micro_usd=1_000 * MICRO)
        buy = make_fill(
            fill_id="f1",
            order_id="o1",
            side=Side.BUY,
            ts=1.0,
            token_amount_atomic=2_000_000,
            notional_usd=20.0,
        )
        ledger.apply_fill(make_report(buy, report_id="r1"))
        # Sell half at a gain.
        sell = make_fill(
            fill_id="f2",
            order_id="o2",
            side=Side.SELL,
            ts=2.0,
            token_amount_atomic=1_000_000,
            notional_usd=15.0,
        )
        ledger.apply_fill(make_report(sell, report_id="r2"))
        # Basis for the sold half is exactly half of the original basis.
        assert ledger.realized_pnl_micro_usd == 15 * MICRO - 10 * MICRO
        snapshot = ledger.snapshot()
        position = snapshot.positions[_SYMBOL]
        assert position.quantity_atomic == 1_000_000
        assert position.cost_basis_usd == 10.0
        # Unrealized PnL is computed against a mark distinct from realized.
        unrealized = snapshot.unrealized_pnl_usd({_SYMBOL: 12.0})
        assert unrealized[_SYMBOL] == 12.0 - 10.0
        # The two numbers must never be conflated.
        assert snapshot.realized_pnl_usd != unrealized[_SYMBOL]


class TestFifoMultiLotExits:
    def test_sell_spanning_two_lots_consumes_oldest_first(self) -> None:
        ledger = BacktestLedger(starting_cash_micro_usd=1_000 * MICRO)
        first = make_fill(
            fill_id="f1",
            order_id="o1",
            side=Side.BUY,
            ts=1.0,
            token_amount_atomic=1_000_000,
            notional_usd=10.0,
        )
        second = make_fill(
            fill_id="f2",
            order_id="o2",
            side=Side.BUY,
            ts=2.0,
            token_amount_atomic=1_000_000,
            notional_usd=30.0,
        )
        ledger.apply_fill(make_report(first, report_id="r1"))
        ledger.apply_fill(make_report(second, report_id="r2"))

        sell = make_fill(
            fill_id="f3",
            order_id="o3",
            side=Side.SELL,
            ts=3.0,
            token_amount_atomic=1_500_000,
            notional_usd=45.0,
        )
        ledger.apply_fill(make_report(sell, report_id="r3"))

        # First lot (basis 10) fully consumed, second lot (basis 30) half
        # consumed: basis released = 10 + 15 = 25.
        assert ledger.realized_pnl_micro_usd == 45 * MICRO - 25 * MICRO
        remaining = ledger.lot_views(_SYMBOL)
        assert len(remaining) == 1
        assert remaining[0].fill_id == "f2"
        assert remaining[0].quantity_atomic == 500_000
        assert remaining[0].basis_micro_usd == 15 * MICRO

        entries = ledger.entries()
        sell_entry = next(e for e in entries if e.kind == "sell")
        # FIFO order preserved in the recorded consumption trail.
        assert sell_entry.lots_consumed[0].startswith("lot:f1:")
        assert sell_entry.lots_consumed[1].startswith("lot:f2:")

    def test_forced_liquidation_of_entire_position_empties_book(self) -> None:
        ledger = BacktestLedger(starting_cash_micro_usd=1_000 * MICRO)
        buy = make_fill(
            fill_id="f1",
            order_id="o1",
            side=Side.BUY,
            ts=1.0,
            token_amount_atomic=1_000_000,
            notional_usd=10.0,
        )
        ledger.apply_fill(make_report(buy, report_id="r1"))
        liquidate = make_fill(
            fill_id="f2",
            order_id="o2",
            side=Side.SELL,
            ts=2.0,
            token_amount_atomic=1_000_000,
            notional_usd=7.5,
        )
        ledger.apply_fill(make_report(liquidate, report_id="r2"))
        snapshot = ledger.snapshot()
        assert _SYMBOL not in snapshot.positions
        assert ledger.realized_pnl_micro_usd == round(7.5 * MICRO) - 10 * MICRO
        # A liquidated book has nothing left to mark — unrealized PnL for the
        # symbol is simply absent, not zero-by-coincidence.
        assert _SYMBOL not in snapshot.unrealized_pnl_usd({_SYMBOL: 999.0})


class TestAllOrNothing:
    def test_oversell_raises_and_leaves_ledger_untouched(self) -> None:
        ledger = BacktestLedger(starting_cash_micro_usd=1_000 * MICRO)
        buy = make_fill(
            fill_id="f1",
            order_id="o1",
            side=Side.BUY,
            ts=1.0,
            token_amount_atomic=1_000_000,
            notional_usd=10.0,
        )
        ledger.apply_fill(make_report(buy, report_id="r1"))
        before_cash = ledger.cash_micro_usd
        before_entries = len(ledger.entries())
        before_lots = ledger.lot_views(_SYMBOL)

        oversell = make_fill(
            fill_id="f2",
            order_id="o2",
            side=Side.SELL,
            ts=2.0,
            token_amount_atomic=2_000_000,
            notional_usd=20.0,
        )
        with pytest.raises(InsufficientLots):
            ledger.apply_fill(make_report(oversell, report_id="r2"))

        # Zero side effects from the failed attempt — this is the guard that
        # makes "partially applied" impossible rather than merely unlikely.
        assert ledger.cash_micro_usd == before_cash
        assert len(ledger.entries()) == before_entries
        assert ledger.lot_views(_SYMBOL) == before_lots

    def test_buy_exceeding_cash_raises_and_leaves_ledger_untouched(self) -> None:
        ledger = BacktestLedger(starting_cash_micro_usd=5 * MICRO)
        buy = make_fill(
            fill_id="f1",
            order_id="o1",
            side=Side.BUY,
            ts=1.0,
            token_amount_atomic=1_000_000,
            notional_usd=10.0,
        )
        with pytest.raises(NegativeCash):
            ledger.apply_fill(make_report(buy, report_id="r1"))
        assert ledger.cash_micro_usd == 5 * MICRO
        assert ledger.entries() == ()
        assert ledger.lot_views(_SYMBOL) == ()


class TestFailureHandling:
    def test_execution_failure_charges_gas_only_no_position_change(self) -> None:
        ledger = BacktestLedger(starting_cash_micro_usd=100 * MICRO)
        buy = make_fill(
            fill_id="f1",
            order_id="o1",
            side=Side.BUY,
            ts=1.0,
            token_amount_atomic=1_000_000,
            notional_usd=10.0,
        )
        ledger.apply_fill(make_report(buy, report_id="r1"))
        before = ledger.snapshot().positions

        failed = make_failed_fill(fill_id="f2", order_id="o2", ts=2.0, gas_usd=0.02)
        report = make_report(failed, report_id="r2", state=OrderState.FAILED)
        ledger.apply_fill(report)

        assert ledger.cash_micro_usd == 90 * MICRO - round(0.02 * MICRO)
        assert ledger.snapshot().positions == before
        entry = ledger.entries()[-1]
        assert entry.kind == "failure"
        assert entry.quantity_delta_atomic == 0
        assert entry.realized_pnl_delta_micro_usd == 0

    def test_rejected_report_with_no_fill_charges_nothing(self) -> None:
        ledger = BacktestLedger(starting_cash_micro_usd=100 * MICRO)
        report = ExecutionReport(
            report_id="r1",
            intent_id="intent-1",
            order_id=None,
            state=OrderState.FAILED,
            ts=1.0,
            fidelity=FidelityTier.TIER_0,
            fill=None,
            reason="route not found",
        )
        applied = ledger.apply_fill(report)
        assert applied is True
        assert ledger.cash_micro_usd == 100 * MICRO
        assert ledger.entries()[-1].kind == "rejected"


class TestIdempotency:
    def test_duplicate_report_id_is_noop(self) -> None:
        ledger = BacktestLedger(starting_cash_micro_usd=100 * MICRO)
        fill = make_fill(
            fill_id="f1",
            order_id="o1",
            side=Side.BUY,
            ts=1.0,
            token_amount_atomic=1_000_000,
            notional_usd=10.0,
        )
        report = make_report(fill, report_id="r1")
        assert ledger.apply_fill(report) is True
        cash_after_first = ledger.cash_micro_usd
        entries_after_first = len(ledger.entries())

        assert ledger.apply_fill(report) is False
        assert ledger.cash_micro_usd == cash_after_first
        # A duplicate is recorded (auditable) but with zero economic effect.
        assert len(ledger.entries()) == entries_after_first + 1
        assert ledger.entries()[-1].kind == "duplicate"
        assert ledger.entries()[-1].cash_delta_micro_usd == 0


class TestDryRun:
    def test_preview_fill_mutates_nothing(self) -> None:
        ledger = BacktestLedger(starting_cash_micro_usd=100 * MICRO)
        fill = make_fill(
            fill_id="f1",
            order_id="o1",
            side=Side.BUY,
            ts=1.0,
            token_amount_atomic=1_000_000,
            notional_usd=10.0,
        )
        report = make_report(fill, report_id="r1")
        preview = ledger.preview_fill(report)
        assert preview.would_apply is True
        assert preview.cash_delta_micro_usd == -10 * MICRO
        assert ledger.cash_micro_usd == 100 * MICRO
        assert ledger.entries() == ()
        assert ledger.seen("r1") is False


class TestNoFloatLeakage:
    def test_accumulated_fees_across_many_fills_have_no_float_drift(self) -> None:
        """Runs enough operations that any float creeping into the running
        totals would eventually produce a mismatch against the exact integer
        sum of the recorded entries. Exact (not approx) equality is the
        point."""
        ledger = BacktestLedger(starting_cash_micro_usd=10**12)
        cash = 0.0037  # a value with no exact binary fraction
        for i in range(200):
            fill = make_fill(
                fill_id=f"f{i}",
                order_id=f"o{i}",
                side=Side.BUY,
                ts=float(i),
                token_amount_atomic=1_000,
                notional_usd=1.0 + cash,
                pool_fee_usd=cash,
                gas_usd=cash,
            )
            ledger.apply_fill(make_report(fill, report_id=f"r{i}"))
        entries = ledger.entries()
        assert sum(e.venue_fee_micro_usd for e in entries) == ledger.venue_fee_micro_usd
        assert sum(e.network_fee_micro_usd for e in entries) == ledger.network_fee_micro_usd
        assert sum(
            e.cash_delta_micro_usd for e in entries
        ) + ledger.starting_cash_micro_usd == (ledger.cash_micro_usd)
        assert isinstance(ledger.cash_micro_usd, int)
        assert isinstance(ledger.venue_fee_micro_usd, int)


class TestCrashRestartRoundTrip:
    def test_to_state_from_state_round_trips_exactly(self) -> None:
        ledger = BacktestLedger(starting_cash_micro_usd=1_000 * MICRO, run_id="run-1")
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

        state = ledger.to_state()
        restored = BacktestLedger.from_state(state)

        assert restored.cash_micro_usd == ledger.cash_micro_usd
        assert restored.realized_pnl_micro_usd == ledger.realized_pnl_micro_usd
        assert restored.venue_fee_micro_usd == ledger.venue_fee_micro_usd
        assert restored.network_fee_micro_usd == ledger.network_fee_micro_usd
        assert restored.priority_fee_micro_usd == ledger.priority_fee_micro_usd
        assert restored.entries() == ledger.entries()
        assert restored.lot_views(_SYMBOL) == ledger.lot_views(_SYMBOL)
        assert restored.seen("r1") is True
        assert restored.seen("r3") is True

        # The restored ledger must continue idempotency and sequencing
        # correctly — re-applying an already-seen report is still a no-op,
        # and a new report gets a fresh, non-colliding seq/lot id.
        assert restored.apply_fill(make_report(buy1, report_id="r1")) is False
        buy3 = make_fill(
            fill_id="f4",
            order_id="o4",
            side=Side.BUY,
            ts=4.0,
            token_amount_atomic=100_000,
            notional_usd=2.0,
        )
        assert restored.apply_fill(make_report(buy3, report_id="r4")) is True
        new_lot_ids = {lot.lot_id for lot in restored.lot_views(_SYMBOL)}
        old_lot_ids = {lot.lot_id for lot in ledger.lot_views(_SYMBOL)}
        assert not (new_lot_ids & old_lot_ids) or new_lot_ids != old_lot_ids

    def test_lot_ordinal_resumes_above_fully_closed_lots(self) -> None:
        """Guard: the lot ordinal counts every lot ever opened, not just open ones.

        ``to_state`` serializes only *open* lots, so a lot that was fully
        closed before the snapshot leaves no trace in ``lots`` — but its label
        is still in the entry log. Rebuilding the counter from open lots alone
        rewinds the ordinal, and a resumed run starts numbering from a value it
        has already used. The ``lot:{fill_id}:{seq}`` format means the label
        itself does not collide (a new fill carries a new fill_id), so no
        dollar amount or invariant is at risk; what breaks is the ordinal's
        meaning. It is supposed to be a monotonic "nth lot opened by this
        ledger", and after a rewind it no longer orders lots correctly when
        read back out of the log.
        """
        ledger = BacktestLedger(starting_cash_micro_usd=1_000 * MICRO, run_id="run-lot")
        buy = make_fill(
            fill_id="f1",
            order_id="o1",
            side=Side.BUY,
            ts=1.0,
            token_amount_atomic=1_000_000,
            notional_usd=10.0,
        )
        # Sell the entire position, so the lot closes and disappears from `lots`.
        sell = make_fill(
            fill_id="f2",
            order_id="o2",
            side=Side.SELL,
            ts=2.0,
            token_amount_atomic=1_000_000,
            notional_usd=12.0,
        )
        ledger.apply_fill(make_report(buy, report_id="r1"))
        ledger.apply_fill(make_report(sell, report_id="r2"))
        assert ledger.lot_views(_SYMBOL) == (), "position should be flat"

        def ordinals(entries: object) -> set[int]:
            found: set[int] = set()
            for entry in entries:  # type: ignore[attr-defined]
                for lot_id in (entry.lot_created, *entry.lots_consumed):
                    if not lot_id:
                        continue
                    tail = lot_id.rsplit(":", 1)[-1]
                    if tail.isdigit():
                        found.add(int(tail))
            return found

        used_before = ordinals(ledger.entries())
        assert used_before, "the closed lot should have left an ordinal in the log"

        restored = BacktestLedger.from_state(ledger.to_state())
        buy2 = make_fill(
            fill_id="f3",
            order_id="o3",
            side=Side.BUY,
            ts=3.0,
            token_amount_atomic=500_000,
            notional_usd=6.0,
        )
        assert restored.apply_fill(make_report(buy2, report_id="r3")) is True

        new_lot = restored.lot_views(_SYMBOL)[0]
        new_ordinal = int(new_lot.lot_id.rsplit(":", 1)[-1])
        assert new_ordinal > max(used_before), (
            f"resumed ledger reused ordinal {new_ordinal}; the pre-snapshot log "
            f"already used {sorted(used_before)}"
        )

    def test_from_state_refuses_unknown_schema_version(self) -> None:
        with pytest.raises(LedgerError):
            BacktestLedger.from_state({"schema_version": 999})


# ---------------------------------------------------------------------------
# Hypothesis: cash/token conservation across arbitrary fill sequences
# ---------------------------------------------------------------------------

# Each tuple drives one BUY (bounded token amount + notional in micro-USD)
# followed by an optional SELL of a random fraction of whatever is currently
# held. Bounds are kept realistic-but-generous: large enough to exercise
# multi-lot exits, small enough that hypothesis explores many shapes quickly.
_BUY_TOKEN_ATOMIC = st.integers(min_value=1, max_value=10_000_000)
_BUY_NOTIONAL_MICRO = st.integers(min_value=1, max_value=1_000_000)
_SELL_FRACTION = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)
_SELL_NOTIONAL_MICRO = st.integers(min_value=0, max_value=2_000_000)


@given(
    st.lists(
        st.tuples(
            _BUY_TOKEN_ATOMIC, _BUY_NOTIONAL_MICRO, _SELL_FRACTION, _SELL_NOTIONAL_MICRO
        ),
        min_size=1,
        max_size=15,
    )
)
@settings(max_examples=100)
def test_cash_token_conservation_across_arbitrary_fill_sequences(
    ops: list[tuple[int, int, float, int]],
) -> None:
    """cash + sum(open lot basis) must equal starting_cash + realized_pnl,
    exactly, in integer micro-USD, no matter how many buys/sells were folded
    in. This is the accounting identity the whole ledger exists to hold —
    if any BUY/SELL branch drops or double-counts a delta, this breaks.
    """
    starting_cash = 10**15
    ledger = BacktestLedger(starting_cash_micro_usd=starting_cash)
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
        ledger.apply_fill(make_report(buy, report_id=f"r-buy-{seq}"))
        held += buy_qty
        check_all(ledger)

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
            ledger.apply_fill(make_report(sell, report_id=f"r-sell-{seq}"))
            held -= sell_qty
            check_all(ledger)

    total_basis_micro = sum(lot.basis_micro_usd for lot in ledger.lot_views(_SYMBOL))
    lhs = ledger.cash_micro_usd + total_basis_micro
    rhs = ledger.starting_cash_micro_usd + ledger.realized_pnl_micro_usd
    assert lhs == rhs
