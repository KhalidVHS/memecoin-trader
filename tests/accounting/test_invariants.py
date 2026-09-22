"""Tests for backtest/invariants.py — one section per invariant in
docs/BACKTEST-CONTRACTS.md §6. Each check function is exercised both on a
state that satisfies it (must not raise) and, where the check's shape allows
constructing one, on a state that violates it (must raise InvariantBreach).
The negative cases are the ones that would fail silently if a guard were
ever weakened to a boolean return.
"""

from __future__ import annotations

import pytest

from memetrader.backtest.invariants import (
    InvariantBreach,
    check_all,
    check_cash_matches_entry_log,
    check_cash_never_negative,
    check_deterministic_replay,
    check_dry_run_no_mutation,
    check_duplicate_entries_are_zero,
    check_duplicate_report_is_noop,
    check_execution_failure_no_position_change,
    check_fee_reconciliation,
    check_fees_charged_once,
    check_fill_lot_integrity,
    check_no_oversell,
    check_quantity_matches_entry_log,
    check_risk_resize_requotes,
    check_sell_quantity_matches_quote,
)
from memetrader.backtest.ledger import MICRO, BacktestLedger, LedgerEntry, LedgerSnapshot
from memetrader.types import (
    ExecutionReport,
    FidelityTier,
    Fill,
    OrderState,
    Quote,
    Side,
    TokenMeta,
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
    state: OrderState = OrderState.LANDED,
) -> Fill:
    usd_atomic = max(round(notional_usd * MICRO), 0)
    if side is Side.BUY:
        in_amount_atomic, out_amount_atomic = usd_atomic, token_amount_atomic
    else:
        in_amount_atomic, out_amount_atomic = token_amount_atomic, usd_atomic
    return Fill(
        fill_id=fill_id,
        order_id=order_id,
        intent_id="intent-1",
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
        pool_fee_usd=0.0,
        gas_usd=0.0,
        realized_pnl_usd=0.0,
    )


def make_failed_fill(*, fill_id: str, order_id: str, ts: float, gas_usd: float) -> Fill:
    return Fill(
        fill_id=fill_id,
        order_id=order_id,
        intent_id="intent-1",
        decision_id=None,
        ts=ts,
        symbol=_SYMBOL,
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
    )


def make_report(
    fill: Fill | None,
    *,
    report_id: str,
    state: OrderState | None = None,
    ts: float | None = None,
) -> ExecutionReport:
    resolved_state = (
        state if state is not None else (fill.state if fill else OrderState.FAILED)
    )
    resolved_ts = ts if ts is not None else (fill.ts if fill else 0.0)
    resolved_order_id = fill.order_id if fill else None
    return ExecutionReport(
        report_id=report_id,
        intent_id="intent-1",
        order_id=resolved_order_id,
        state=resolved_state,
        ts=resolved_ts,
        fidelity=FidelityTier.TIER_0,
        fill=fill,
    )


def make_quote(*, side: Side, token_amount_atomic: int) -> Quote:
    usd = TokenMeta(
        mint="So11111111111111111111111111111111111111112", decimals=9, source="test"
    )
    token = TokenMeta(mint="mint-bonk", decimals=6, source="test")
    if side is Side.BUY:
        input_token, output_token = usd, token
        in_amount_atomic, out_amount_atomic = 10_000_000_000, token_amount_atomic
    else:
        input_token, output_token = token, usd
        in_amount_atomic, out_amount_atomic = token_amount_atomic, 10_000_000_000
    return Quote(
        symbol=_SYMBOL,
        side=side,
        input_token=input_token,
        output_token=output_token,
        in_amount_atomic=in_amount_atomic,
        out_amount_atomic=out_amount_atomic,
        min_out_amount_atomic=out_amount_atomic,
        price_impact_pct=0.0,
        route_labels=("test",),
        fingerprint="fp",
        requested_at=0.0,
        received_at=0.1,
    )


def _basic_ledger() -> BacktestLedger:
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
    return ledger


# ---------------------------------------------------------------------------
# 1. Cash never negative
# ---------------------------------------------------------------------------


class TestCashNeverNegative:
    def test_passes_for_non_negative_cash(self) -> None:
        check_cash_never_negative(_basic_ledger().snapshot())

    def test_raises_for_negative_cash(self) -> None:
        snap = LedgerSnapshot(
            ts=0.0,
            cash_micro_usd=-1,
            starting_cash_micro_usd=0,
            positions={},
            realized_pnl_micro_usd=0,
            venue_fee_micro_usd=0,
            network_fee_micro_usd=0,
            priority_fee_micro_usd=0,
        )
        with pytest.raises(InvariantBreach):
            check_cash_never_negative(snap)


# ---------------------------------------------------------------------------
# 2. No sale exceeding settled inventory
# ---------------------------------------------------------------------------


class TestNoOversell:
    def test_passes_for_a_valid_buy_then_sell_log(self) -> None:
        ledger = _basic_ledger()
        sell = make_fill(
            fill_id="f2",
            order_id="o2",
            side=Side.SELL,
            ts=2.0,
            token_amount_atomic=500_000,
            notional_usd=6.0,
        )
        ledger.apply_fill(make_report(sell, report_id="r2"))
        check_no_oversell(ledger.entries())

    def test_raises_for_a_log_implying_negative_inventory(self) -> None:
        entries = (
            LedgerEntry(
                seq=0,
                report_id="r1",
                kind="sell",
                ts=1.0,
                symbol=_SYMBOL,
                fill_id="f1",
                order_id="o1",
                cash_delta_micro_usd=1,
                realized_pnl_delta_micro_usd=0,
                quantity_delta_atomic=-100,
                venue_fee_micro_usd=0,
                network_fee_micro_usd=0,
                priority_fee_micro_usd=0,
            ),
        )
        with pytest.raises(InvariantBreach):
            check_no_oversell(entries)


# ---------------------------------------------------------------------------
# 3. Cash / tokens change only through ledger entries
# ---------------------------------------------------------------------------


class TestChangesOnlyThroughEntries:
    def test_passes_for_untampered_ledger(self) -> None:
        ledger = _basic_ledger()
        check_cash_matches_entry_log(ledger)
        check_quantity_matches_entry_log(ledger)

    def test_raises_if_cash_is_mutated_outside_apply_fill(self) -> None:
        ledger = _basic_ledger()
        ledger._cash_micro += 1  # simulate a future bug bypassing apply_fill
        with pytest.raises(InvariantBreach):
            check_cash_matches_entry_log(ledger)

    def test_raises_if_inventory_is_mutated_outside_apply_fill(self) -> None:
        ledger = _basic_ledger()
        next(iter(ledger._lots[_SYMBOL])).quantity_atomic += 1
        with pytest.raises(InvariantBreach):
            check_quantity_matches_entry_log(ledger)


# ---------------------------------------------------------------------------
# 4. Fees charged exactly once
# ---------------------------------------------------------------------------


class TestFeesChargedOnce:
    def test_passes_for_untampered_log(self) -> None:
        ledger = _basic_ledger()
        check_fees_charged_once(ledger.entries())
        check_fee_reconciliation(ledger)

    def test_raises_if_same_fill_id_appears_in_two_entries(self) -> None:
        entries = (
            LedgerEntry(
                seq=0,
                report_id="r1",
                kind="buy",
                ts=1.0,
                symbol=_SYMBOL,
                fill_id="dup",
                order_id="o1",
                cash_delta_micro_usd=-1,
                realized_pnl_delta_micro_usd=0,
                quantity_delta_atomic=1,
                venue_fee_micro_usd=1,
                network_fee_micro_usd=0,
                priority_fee_micro_usd=0,
                lot_created="lot:dup:0",
            ),
            LedgerEntry(
                seq=1,
                report_id="r2",
                kind="buy",
                ts=2.0,
                symbol=_SYMBOL,
                fill_id="dup",
                order_id="o2",
                cash_delta_micro_usd=-1,
                realized_pnl_delta_micro_usd=0,
                quantity_delta_atomic=1,
                venue_fee_micro_usd=1,
                network_fee_micro_usd=0,
                priority_fee_micro_usd=0,
                lot_created="lot:dup:1",
            ),
        )
        with pytest.raises(InvariantBreach):
            check_fees_charged_once(entries)

    def test_fee_reconciliation_raises_if_running_total_is_corrupted(self) -> None:
        ledger = _basic_ledger()
        ledger._venue_fee_micro += 1
        with pytest.raises(InvariantBreach):
            check_fee_reconciliation(ledger)


# ---------------------------------------------------------------------------
# 5. Every fill belongs to one order and one position lot
# ---------------------------------------------------------------------------


class TestFillLotIntegrity:
    def test_passes_for_a_normal_buy_sell_sequence(self) -> None:
        ledger = _basic_ledger()
        sell = make_fill(
            fill_id="f2",
            order_id="o2",
            side=Side.SELL,
            ts=2.0,
            token_amount_atomic=1_000_000,
            notional_usd=12.0,
        )
        ledger.apply_fill(make_report(sell, report_id="r2"))
        check_fill_lot_integrity(ledger)

    def test_raises_if_a_sell_entry_consumes_an_unknown_lot(self) -> None:
        entries = (
            LedgerEntry(
                seq=0,
                report_id="r1",
                kind="sell",
                ts=1.0,
                symbol=_SYMBOL,
                fill_id="f1",
                order_id="o1",
                cash_delta_micro_usd=1,
                realized_pnl_delta_micro_usd=0,
                quantity_delta_atomic=-1,
                venue_fee_micro_usd=0,
                network_fee_micro_usd=0,
                priority_fee_micro_usd=0,
                lots_consumed=("lot:never-created:0",),
            ),
        )

        class _FakeLedger:
            def entries(self) -> tuple[LedgerEntry, ...]:
                return entries

        with pytest.raises(InvariantBreach):
            check_fill_lot_integrity(_FakeLedger())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 6. Same inputs + hashes -> byte-identical economic output
# ---------------------------------------------------------------------------


class TestDeterministicReplay:
    def test_replaying_the_same_reports_twice_matches(self) -> None:
        buy = make_fill(
            fill_id="f1",
            order_id="o1",
            side=Side.BUY,
            ts=1.0,
            token_amount_atomic=1_000_000,
            notional_usd=10.0,
        )
        sell = make_fill(
            fill_id="f2",
            order_id="o2",
            side=Side.SELL,
            ts=2.0,
            token_amount_atomic=1_000_000,
            notional_usd=12.0,
        )
        reports = [make_report(buy, report_id="r1"), make_report(sell, report_id="r2")]
        # Must not raise: two fresh ledgers folding the same report sequence
        # produce byte-identical snapshots.
        check_deterministic_replay(reports, starting_cash_micro_usd=1_000 * MICRO)


# ---------------------------------------------------------------------------
# 7. A dry run mutates nothing
# ---------------------------------------------------------------------------


class TestDryRunNoMutation:
    def test_preview_fill_leaves_ledger_untouched(self) -> None:
        ledger = BacktestLedger(starting_cash_micro_usd=1_000 * MICRO)
        buy = make_fill(
            fill_id="f1",
            order_id="o1",
            side=Side.BUY,
            ts=1.0,
            token_amount_atomic=1_000_000,
            notional_usd=10.0,
        )
        check_dry_run_no_mutation(ledger, make_report(buy, report_id="r1"))
        # The check itself only calls preview_fill; confirm state truly is
        # still empty afterward.
        assert ledger.entries() == ()
        assert ledger.cash_micro_usd == 1_000 * MICRO


# ---------------------------------------------------------------------------
# 8. An execution failure can never become a synthetic fill
# ---------------------------------------------------------------------------


class TestExecutionFailureNoPositionChange:
    def test_passes_when_a_failure_only_burns_gas(self) -> None:
        ledger = _basic_ledger()
        failed = make_failed_fill(fill_id="f2", order_id="o2", ts=2.0, gas_usd=0.01)
        report = make_report(failed, report_id="r2", state=OrderState.FAILED)
        check_execution_failure_no_position_change(ledger, report)

    def test_raises_if_called_with_a_non_terminal_failure_state(self) -> None:
        ledger = _basic_ledger()
        buy = make_fill(
            fill_id="f2",
            order_id="o2",
            side=Side.BUY,
            ts=2.0,
            token_amount_atomic=1,
            notional_usd=0.01,
        )
        report = make_report(buy, report_id="r2", state=OrderState.LANDED)
        with pytest.raises(InvariantBreach):
            check_execution_failure_no_position_change(ledger, report)


# ---------------------------------------------------------------------------
# 9. Risk resizing triggers a new quote at the approved size (audit C3)
# ---------------------------------------------------------------------------


class TestRiskResizeRequotes:
    def test_passes_when_every_approved_size_was_requoted(self) -> None:
        check_risk_resize_requotes([(1_000, 1_000), (500, 500)])

    def test_raises_when_an_order_kept_the_stale_quote_size(self) -> None:
        with pytest.raises(InvariantBreach):
            check_risk_resize_requotes([(1_000, 1_000), (500, 800)])


# ---------------------------------------------------------------------------
# 10. SELL quantity == quoted quantity, exactly
# ---------------------------------------------------------------------------


class TestSellQuantityMatchesQuote:
    def test_passes_when_the_settled_sell_matches_the_quote(self) -> None:
        fill = make_fill(
            fill_id="f1",
            order_id="o1",
            side=Side.SELL,
            ts=1.0,
            token_amount_atomic=500_000,
            notional_usd=5.0,
        )
        quote = make_quote(side=Side.SELL, token_amount_atomic=500_000)
        check_sell_quantity_matches_quote(fill, quote)

    def test_raises_when_the_settled_sell_diverges_from_the_quote(self) -> None:
        fill = make_fill(
            fill_id="f1",
            order_id="o1",
            side=Side.SELL,
            ts=1.0,
            token_amount_atomic=500_001,
            notional_usd=5.0,
        )
        quote = make_quote(side=Side.SELL, token_amount_atomic=500_000)
        with pytest.raises(InvariantBreach):
            check_sell_quantity_matches_quote(fill, quote)

    def test_buy_side_fills_are_exempt(self) -> None:
        fill = make_fill(
            fill_id="f1",
            order_id="o1",
            side=Side.BUY,
            ts=1.0,
            token_amount_atomic=999,
            notional_usd=5.0,
        )
        quote = make_quote(side=Side.BUY, token_amount_atomic=500_000)
        check_sell_quantity_matches_quote(fill, quote)  # must not raise


# ---------------------------------------------------------------------------
# 11. Duplicate ExecutionReport.report_id is a no-op
# ---------------------------------------------------------------------------


class TestDuplicateReportIsNoop:
    def test_reapplying_the_same_report_is_a_noop(self) -> None:
        ledger = BacktestLedger(starting_cash_micro_usd=1_000 * MICRO)
        buy = make_fill(
            fill_id="f1",
            order_id="o1",
            side=Side.BUY,
            ts=1.0,
            token_amount_atomic=1_000_000,
            notional_usd=10.0,
        )
        check_duplicate_report_is_noop(ledger, make_report(buy, report_id="r1"))

    def test_duplicate_entries_are_inert(self) -> None:
        ledger = BacktestLedger(starting_cash_micro_usd=1_000 * MICRO)
        buy = make_fill(
            fill_id="f1",
            order_id="o1",
            side=Side.BUY,
            ts=1.0,
            token_amount_atomic=1_000_000,
            notional_usd=10.0,
        )
        report = make_report(buy, report_id="r1")
        ledger.apply_fill(report)
        ledger.apply_fill(report)
        check_duplicate_entries_are_zero(ledger.entries())

    def test_raises_if_a_duplicate_entry_somehow_carries_a_nonzero_delta(self) -> None:
        entries = (
            LedgerEntry(
                seq=0,
                report_id="r1",
                kind="duplicate",
                ts=1.0,
                symbol=_SYMBOL,
                fill_id="f1",
                order_id="o1",
                cash_delta_micro_usd=1,
                realized_pnl_delta_micro_usd=0,
                quantity_delta_atomic=0,
                venue_fee_micro_usd=0,
                network_fee_micro_usd=0,
                priority_fee_micro_usd=0,
            ),
        )
        with pytest.raises(InvariantBreach):
            check_duplicate_entries_are_zero(entries)


# ---------------------------------------------------------------------------
# check_all — the aggregate the engine calls every tick
# ---------------------------------------------------------------------------


class TestCheckAll:
    def test_passes_on_a_well_formed_mixed_history(self) -> None:
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
        check_all(ledger)

        sell = make_fill(
            fill_id="f2",
            order_id="o2",
            side=Side.SELL,
            ts=2.0,
            token_amount_atomic=500_000,
            notional_usd=6.0,
        )
        ledger.apply_fill(make_report(sell, report_id="r2"))
        check_all(ledger)

        failed = make_failed_fill(fill_id="f3", order_id="o3", ts=3.0, gas_usd=0.01)
        ledger.apply_fill(make_report(failed, report_id="r3", state=OrderState.FAILED))
        check_all(ledger)

        # A repeated report_id is still checked cleanly (inert duplicate).
        ledger.apply_fill(make_report(buy, report_id="r1"))
        check_all(ledger)

    def test_raises_if_the_book_has_been_tampered_with(self) -> None:
        ledger = _basic_ledger()
        ledger._cash_micro = -1
        with pytest.raises(InvariantBreach):
            check_all(ledger)
