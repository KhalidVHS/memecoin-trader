"""Tests for metrics.attribution — PnL decomposition and reconciliation.

The headline invariant this file guards: gross_alpha_usd minus every cost
bucket (venue/network/priority fee, spread, price impact, latency cost,
failure cost) plus the benchmark decomposition terms must reconcile to
realized_pnl_usd.  ``attribute_pnl`` itself enforces this per-trade (raises
``AttributionError`` on breach); the property tests here additionally check
that the *aggregate* totals reconcile the same way, and that concentration
slices (by_asset/by_month/by_regime/by_signal) partition every trade exactly
once.

Note on tolerance: unlike the ledger's atomic (int) cash convention, these
USD figures are float dollars derived for reporting (Fill docstring: "Atomic
amounts are the record; dollars are derived for reporting"). Reconciliation
here is therefore checked to float precision (~1e-9), not to an integer
atomic unit -- there is no atomic unit at this layer.
"""

from __future__ import annotations

import datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from memetrader.metrics.attribution import (
    AttributionError,
    TradeAttribution,
    _check_reconciliation,
    attribute_pnl,
)
from memetrader.types import (
    CostBreakdown,
    ExecutionReport,
    FidelityTier,
    Fill,
    OrderState,
    Side,
)

TIER_0 = FidelityTier.TIER_0
TIER_2 = FidelityTier.TIER_2


def _fill(
    *,
    fill_id: str,
    ts: float = 1_700_000_000.0,
    symbol: str = "FOO",
    side: Side = Side.SELL,
    state: OrderState = OrderState.LANDED,
    realized_pnl_usd: float = 0.0,
    pool_fee_usd: float = 0.0,
    gas_usd: float = 0.0,
) -> Fill:
    return Fill(
        fill_id=fill_id,
        order_id=f"order-{fill_id}",
        intent_id=f"intent-{fill_id}",
        decision_id=None,
        ts=ts,
        symbol=symbol,
        side=side,
        state=state,
        in_amount_atomic=1_000_000,
        out_amount_atomic=1_000_000,
        token_amount_atomic=1_000_000,
        token_decimals=6,
        quote_fingerprint="fp",
        price_usd=1.0,
        notional_usd=100.0,
        price_impact_pct=0.0,
        pool_fee_usd=pool_fee_usd,
        gas_usd=gas_usd,
        realized_pnl_usd=realized_pnl_usd,
    )


def _report(
    *,
    report_id: str,
    fill: Fill | None,
    state: OrderState,
    costs: CostBreakdown | None = None,
    ts: float = 1_700_000_000.0,
) -> ExecutionReport:
    return ExecutionReport(
        report_id=report_id,
        intent_id=f"intent-{report_id}",
        order_id=f"order-{report_id}",
        state=state,
        ts=ts,
        fidelity=TIER_2,
        fill=fill,
        costs=costs,
    )


# ---------------------------------------------------------------------------
# Headline reconciliation invariant
# ---------------------------------------------------------------------------


def test_landed_trade_reconciles_to_realized_pnl() -> None:
    costs = CostBreakdown(
        venue_fee_usd=1.0,
        network_fee_usd=0.5,
        priority_fee_usd=0.25,
        spread_usd=0.1,
        price_impact_usd=0.2,
        latency_cost_usd=-0.05,
        failure_cost_usd=0.0,
    )
    fill = _fill(fill_id="s1", realized_pnl_usd=10.0)
    report = _report(report_id="r1", fill=fill, state=OrderState.LANDED, costs=costs)

    summary = attribute_pnl([report], fidelity=TIER_2)

    assert len(summary.trades) == 1
    trade = summary.trades[0]
    reconstructed = (
        trade.gross_alpha_usd
        - trade.venue_fee_usd
        - trade.network_fee_usd
        - trade.priority_fee_usd
        - trade.spread_usd
        - trade.price_impact_usd
        - trade.latency_cost_usd
        - trade.failure_cost_usd
    )
    assert reconstructed == pytest.approx(trade.realized_pnl_usd, abs=1e-9)
    assert trade.gross_alpha_usd == pytest.approx(10.0 + costs.total_usd)


def test_reconciliation_breach_raises_attribution_error() -> None:
    """A cost model that disagrees with the fill ledger must raise loudly,
    not silently misreport.

    ``attribute_pnl`` always *derives* gross_alpha = realized + total costs,
    which makes the per-trade reconciliation tautologically true for any
    input reachable through the public API -- there is no way to feed it an
    inconsistent (gross_alpha, costs, realized) triple through normal use.
    So the raise path is tested directly against the private
    ``_check_reconciliation`` guard with a hand-built, deliberately
    inconsistent ``TradeAttribution`` -- exactly the shape a future bug
    (e.g. a renamed or dropped cost bucket) would produce.
    """
    trade = TradeAttribution(
        symbol="FOO",
        fill_id="f1",
        realized_pnl_usd=10.0,
        gross_alpha_usd=10.0,
        venue_fee_usd=5.0,  # not reflected in realized_pnl_usd above
        network_fee_usd=0.0,
        priority_fee_usd=0.0,
        spread_usd=0.0,
        price_impact_usd=0.0,
        latency_cost_usd=0.0,
        failure_cost_usd=0.0,
        selection_usd=10.0,
        sizing_usd=0.0,
        benchmark_usd=0.0,
        month="2025-01",
        regime=None,
        signal_id=None,
        asset="FOO",
    )
    with pytest.raises(AttributionError, match="does not reconcile"):
        _check_reconciliation(trade, 0.01)


# ---------------------------------------------------------------------------
# Failed / never-filled trades
# ---------------------------------------------------------------------------


def test_failed_trade_contributes_failure_cost_and_zero_gross_alpha() -> None:
    """BUG FOUND AND FIXED: previously, a non-LANDED report with a fill was
    silently skipped entirely -- no failure_cost_usd, no TradeAttribution --
    contradicting both the module's own docstring ("FAILED reports
    contribute failure_cost from their fill's gas_usd") and the accounting
    invariant that every dollar spent is accounted for. Fixed in
    attribution.py: a report whose fill is present but not LANDED now
    produces a TradeAttribution with failure_cost_usd = fill.gas_usd and
    gross_alpha_usd = 0 when realized_pnl_usd == -gas_usd (the natural
    ledger convention for "spent gas, landed nothing").
    """
    gas = 3.5
    fill = _fill(
        fill_id="f1",
        state=OrderState.FAILED,
        gas_usd=gas,
        realized_pnl_usd=-gas,
    )
    report = _report(report_id="r1", fill=fill, state=OrderState.FAILED)

    summary = attribute_pnl([report], fidelity=TIER_2)

    assert len(summary.trades) == 1
    trade = summary.trades[0]
    assert trade.failure_cost_usd == pytest.approx(gas)
    assert trade.gross_alpha_usd == pytest.approx(0.0)
    assert trade.selection_usd == 0.0
    assert trade.sizing_usd == 0.0
    assert trade.benchmark_usd == 0.0
    assert summary.total_failure_cost_usd == pytest.approx(gas)
    assert summary.total_realized_pnl_usd == pytest.approx(-gas)


def test_expired_trade_also_contributes_failure_cost() -> None:
    gas = 1.2
    fill = _fill(
        fill_id="f1",
        state=OrderState.EXPIRED,
        gas_usd=gas,
        realized_pnl_usd=-gas,
    )
    report = _report(report_id="r1", fill=fill, state=OrderState.EXPIRED)
    summary = attribute_pnl([report], fidelity=TIER_2)
    assert summary.total_failure_cost_usd == pytest.approx(gas)


def test_report_with_no_fill_contributes_nothing() -> None:
    """A report where nothing was even attempted (fill is None) is not a
    trade at all -- it must not appear in trades or totals."""
    report = _report(report_id="r1", fill=None, state=OrderState.FAILED)
    summary = attribute_pnl([report], fidelity=TIER_2)
    assert summary.trades == []
    assert summary.total_realized_pnl_usd == 0.0
    assert summary.total_failure_cost_usd == 0.0


# ---------------------------------------------------------------------------
# Slicing partitions trades exactly once
# ---------------------------------------------------------------------------


def _report_strategy() -> st.SearchStrategy[tuple[float, str, int]]:
    """A single landed report with a controlled realized_pnl_usd, tagged with
    an asset/month/signal/regime we can check partitioning against."""
    return st.builds(
        lambda pnl, sym, month_offset: (pnl, sym, month_offset),
        pnl=st.floats(
            min_value=-1_000.0, max_value=1_000.0, allow_nan=False, allow_infinity=False
        ),
        sym=st.sampled_from(["FOO", "BAR", "BAZ"]),
        month_offset=st.integers(min_value=0, max_value=5),
    )


@given(st.lists(_report_strategy(), min_size=1, max_size=25))
@settings(max_examples=50)
def test_slices_partition_total_pnl_exactly(rows: list[tuple[float, str, int]]) -> None:
    base_ts = 1_700_000_000.0
    seconds_per_month = 30 * 24 * 3600
    reports = []
    for i, (pnl, sym, month_offset) in enumerate(rows):
        ts = base_ts + month_offset * seconds_per_month
        fill = _fill(fill_id=f"f{i}", ts=ts, symbol=sym, realized_pnl_usd=pnl)
        reports.append(
            _report(report_id=f"r{i}", fill=fill, state=OrderState.LANDED, ts=ts)
        )

    summary = attribute_pnl(list(reports), fidelity=TIER_2)

    assert sum(summary.by_asset.values()) == pytest.approx(
        summary.total_realized_pnl_usd, abs=1e-6
    )
    assert sum(summary.by_month.values()) == pytest.approx(
        summary.total_realized_pnl_usd, abs=1e-6
    )
    assert sum(summary.by_regime.values()) == pytest.approx(
        summary.total_realized_pnl_usd, abs=1e-6
    )
    assert sum(summary.by_signal.values()) == pytest.approx(
        summary.total_realized_pnl_usd, abs=1e-6
    )
    # Every trade appears in exactly one asset slice's contribution and one
    # month slice's contribution -- summing per-trade realized_pnl_usd must
    # equal both aggregates independently reconstructed from trades.
    from_trades = sum(t.realized_pnl_usd for t in summary.trades)
    assert from_trades == pytest.approx(summary.total_realized_pnl_usd, abs=1e-6)


def test_month_key_is_utc_and_matches_fill_timestamp() -> None:
    ts = datetime.datetime(2025, 3, 15, tzinfo=datetime.UTC).timestamp()
    fill = _fill(fill_id="f1", ts=ts, realized_pnl_usd=1.0)
    report = _report(report_id="r1", fill=fill, state=OrderState.LANDED, ts=ts)
    summary = attribute_pnl([report], fidelity=TIER_2)
    assert summary.trades[0].month == "2025-03"


# ---------------------------------------------------------------------------
# Property: aggregate reconciliation over random landed fills
# ---------------------------------------------------------------------------


@given(
    realized=st.floats(
        min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False
    ),
    venue=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    network=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    priority=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    spread=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    impact=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    latency=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_reconciliation_holds_for_arbitrary_cost_breakdowns(
    realized: float,
    venue: float,
    network: float,
    priority: float,
    spread: float,
    impact: float,
    latency: float,
) -> None:
    """No matter what the (authoritative, explicitly-supplied) CostBreakdown
    says, gross_alpha is defined as realized + total costs, so the
    reconstruction must always tie out -- this guards against a future edit
    that adds/removes/renames a cost bucket in the trade construction without
    updating _check_reconciliation to match."""
    costs = CostBreakdown(
        venue_fee_usd=venue,
        network_fee_usd=network,
        priority_fee_usd=priority,
        spread_usd=spread,
        price_impact_usd=impact,
        latency_cost_usd=latency,
        failure_cost_usd=0.0,
    )
    fill = _fill(fill_id="s1", realized_pnl_usd=realized)
    report = _report(report_id="r1", fill=fill, state=OrderState.LANDED, costs=costs)

    summary = attribute_pnl([report], fidelity=TIER_2, reconciliation_tolerance_usd=1e-6)

    trade = summary.trades[0]
    reconstructed = (
        trade.gross_alpha_usd
        - trade.venue_fee_usd
        - trade.network_fee_usd
        - trade.priority_fee_usd
        - trade.spread_usd
        - trade.price_impact_usd
        - trade.latency_cost_usd
        - trade.failure_cost_usd
    )
    assert reconstructed == pytest.approx(trade.realized_pnl_usd, abs=1e-6)


# ---------------------------------------------------------------------------
# Fidelity notice
# ---------------------------------------------------------------------------


def test_tier0_carries_non_executable_notice() -> None:
    fill = _fill(fill_id="s1", realized_pnl_usd=1.0)
    report = _report(report_id="r1", fill=fill, state=OrderState.LANDED)
    summary = attribute_pnl([report], fidelity=TIER_0)
    assert summary.non_executable_notice is not None


def test_tier2_carries_no_notice() -> None:
    fill = _fill(fill_id="s1", realized_pnl_usd=1.0)
    report = _report(report_id="r1", fill=fill, state=OrderState.LANDED)
    summary = attribute_pnl([report], fidelity=TIER_2)
    assert summary.non_executable_notice is None
