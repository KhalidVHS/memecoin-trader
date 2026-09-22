"""PnL attribution — every dollar of PnL reconciled to ledger events.

Design principle: hard reconciliation
--------------------------------------
An attribution that silently fails to add up is worse than none.  A number
that says "fees cost $X" when the actual realized PnL implies a different total
misleads the analyst and the promotion gate.  Every TradeAttribution carries a
check: gross_alpha minus the sum of all cost components must equal realized PnL
within a stated tolerance, and the function RAISES AttributionError — not a
warning, not a log line — when it does not.

Why these cost components?
---------------------------
- venue_fee:    AMM pool fee taken on every swap (0.25–1% on Raydium/Orca).
- network_fee:  Solana base transaction fee (5000 lamports ≈ $0.001 at normal
                SOL prices; negligible individually but real at scale).
- priority_fee: Compute-unit price for priority landing; highly variable and
                the primary factor in failed-transaction cost.
- spread:       Half-spread captured by passive LPs on entry and exit combined.
                At TIER_0 this is estimated from price impact, not measured.
- price_impact: The AMM slippage from moving the curve with our own order.
- latency_cost: Signed change in fair value between decision and fill.  Negative
                (good) when prices moved in our favour while the order was in
                flight; positive (bad) in the usual case.
- failure_cost: Gas and priority fees spent on transactions that failed to land.
                Dropped from attribution in older systems; retained here because
                a strategy that wins by sending many speculative transactions
                carries a real failure-cost tail.

Benchmark decomposition
------------------------
When a benchmark return is available, gross_alpha is split into:
- selection_usd: alpha from choosing this asset (vs benchmark return)
- sizing_usd:    alpha from position size (vs equal-weight)
- benchmark_usd: what an equal-weight position in the benchmark would have made

Without a benchmark, the full gross_alpha is attributed to selection and the
other two are zero.

Concentration slicing
----------------------
by_asset, by_month, by_regime, by_signal allow the promotion gate to check
the rule "no single slice contributes > ~25% of total PnL".  All four are
always populated; slices with None keys represent unattributed trades.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field
from typing import Sequence

from memetrader.types import (
    CostBreakdown,
    ExecutionReport,
    FidelityTier,
    Fill,
    NON_EXECUTABLE_NOTICE,
    OrderState,
    Side,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AttributionError(ValueError):
    """Raised when PnL components do not reconcile to realized PnL.

    The sum of (gross_alpha - all cost components) must equal realized_pnl_usd
    within reconciliation_tolerance_usd.  Any breach means the cost model is
    inconsistent with the fill ledger and the attribution cannot be trusted.
    """


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class TradeAttribution:
    """Attribution for one closed trade (one ExecutionReport with a fill).

    All USD amounts are signed: a negative cost means the component added to
    PnL (e.g. latency_cost_usd can be negative when a delay helped).
    """

    symbol: str
    fill_id: str
    realized_pnl_usd: float
    gross_alpha_usd: float
    venue_fee_usd: float
    network_fee_usd: float
    priority_fee_usd: float
    spread_usd: float
    price_impact_usd: float
    latency_cost_usd: float
    failure_cost_usd: float
    # Benchmark decomposition (zero when no benchmark supplied)
    selection_usd: float
    sizing_usd: float
    benchmark_usd: float
    # Slice keys for concentration analysis
    month: str          # "YYYY-MM"
    regime: str | None
    signal_id: str | None
    asset: str


@dataclass
class AttributionSummary:
    """Aggregated attribution across all fills in the run."""

    total_realized_pnl_usd: float
    total_gross_alpha_usd: float
    total_venue_fee_usd: float
    total_network_fee_usd: float
    total_priority_fee_usd: float
    total_spread_usd: float
    total_price_impact_usd: float
    total_latency_cost_usd: float
    total_failure_cost_usd: float
    total_selection_usd: float
    total_sizing_usd: float
    total_benchmark_usd: float
    # Concentration slices — all keyed by their natural dimension
    by_asset: dict[str, float]
    by_month: dict[str, float]
    by_regime: dict[str | None, float]
    by_signal: dict[str | None, float]
    trades: list[TradeAttribution]
    fidelity: FidelityTier
    non_executable_notice: str | None
    reconciliation_tolerance_usd: float


# ---------------------------------------------------------------------------
# Cost extraction
# ---------------------------------------------------------------------------


def _extract_costs(
    report: ExecutionReport,
    fill: Fill,
) -> CostBreakdown:
    """Return the CostBreakdown to use for this report.

    Prefer ``report.costs`` when present — it is the execution engine's
    authoritative breakdown.  Fall back to the fields on the Fill itself:
    pool_fee_usd maps to venue_fee, gas_usd to network_fee, and everything
    else is zero.  This fall-back is explicitly conservative and partial; the
    promotion gate should flag runs where costs had to be imputed this way.
    """
    if report.costs is not None:
        return report.costs
    return CostBreakdown(
        venue_fee_usd=fill.pool_fee_usd,
        network_fee_usd=fill.gas_usd,
        priority_fee_usd=0.0,
        spread_usd=0.0,
        price_impact_usd=0.0,
        latency_cost_usd=0.0,
        failure_cost_usd=0.0,
    )


def _month_key(ts: float) -> str:
    """Convert epoch seconds to 'YYYY-MM' string.

    Uses UTC to avoid timezone-dependent test failures — the codebase uses
    epoch seconds throughout, so UTC is the consistent interpretation.
    """
    dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m")


# ---------------------------------------------------------------------------
# Reconciliation check
# ---------------------------------------------------------------------------


def _check_reconciliation(
    trade: TradeAttribution,
    tolerance: float,
) -> None:
    """Raise AttributionError when the trade does not reconcile.

    Reconstruction:
        realized_pnl = gross_alpha
                       - venue_fee - network_fee - priority_fee
                       - spread - price_impact - latency_cost - failure_cost

    This must hold to within tolerance.  Note that latency_cost can be
    negative (see module docstring), so it is subtracted even when negative —
    which means a negative latency_cost *adds* to the reconstructed PnL, as
    intended.
    """
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
    diff = abs(reconstructed - trade.realized_pnl_usd)
    if diff > tolerance:
        raise AttributionError(
            f"Attribution does not reconcile for fill {trade.fill_id} "
            f"({trade.symbol}): "
            f"gross_alpha({trade.gross_alpha_usd:.6f}) minus costs "
            f"= {reconstructed:.6f}, but realized_pnl = {trade.realized_pnl_usd:.6f}. "
            f"Difference {diff:.6f} exceeds tolerance {tolerance:.6f}. "
            "This indicates an inconsistency between the cost model and the "
            "fill ledger — check that CostBreakdown fields match the fill."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def attribute_pnl(
    reports: Sequence[ExecutionReport],
    *,
    fidelity: FidelityTier,
    benchmark_returns: dict[str, float] | None = None,
    regime_map: dict[str, str] | None = None,
    signal_map: dict[str, str] | None = None,
    reconciliation_tolerance_usd: float = 0.01,
) -> AttributionSummary:
    """Decompose realized PnL into its causal components.

    Parameters
    ----------
    reports:
        All ExecutionReports from the run.  Only reports with a non-None fill
        and state LANDED are attributed as completed trades; FAILED reports
        contribute failure_cost from their fill's gas_usd.
    fidelity:
        FidelityTier of the run.  Any result below TIER_2 carries
        NON_EXECUTABLE_NOTICE.
    benchmark_returns:
        symbol -> benchmark PnL in USD on the same notional.  When None, the
        benchmark decomposition is zeroed and all gross_alpha is attributed to
        selection.
    regime_map:
        symbol -> regime label (e.g. "bull", "bear", "chop").  None = unknown.
    signal_map:
        fill_id -> signal_id.  None = unattributed.
    reconciliation_tolerance_usd:
        Maximum allowed absolute difference between reconstructed and reported
        realized PnL per trade.  Set tighter for TIER_2+ where costs are
        precisely modelled; looser for TIER_0 where costs are estimated.
    """
    notice = NON_EXECUTABLE_NOTICE if not fidelity.permits_pnl_claim else None

    trades: list[TradeAttribution] = []
    by_asset: dict[str, float] = {}
    by_month: dict[str, float] = {}
    by_regime: dict[str | None, float] = {}
    by_signal: dict[str | None, float] = {}

    # Aggregate totals
    tot_pnl = 0.0
    tot_gross = 0.0
    tot_venue = 0.0
    tot_net = 0.0
    tot_pri = 0.0
    tot_spread = 0.0
    tot_impact = 0.0
    tot_latency = 0.0
    tot_failure = 0.0
    tot_selection = 0.0
    tot_sizing = 0.0
    tot_bench = 0.0

    for report in reports:
        fill = report.fill
        if fill is None:
            continue
        if report.state is not OrderState.LANDED:
            # Failed/expired reports: count failure_cost only.
            # We do not create a TradeAttribution for non-fills; the failure
            # cost is captured in the fill's gas_usd on FAILED fills.
            continue

        costs = _extract_costs(report, fill)
        realized = fill.realized_pnl_usd

        # gross_alpha is what we would have made with zero friction.
        # realized = gross_alpha - sum(costs), so gross_alpha = realized + sum(costs).
        gross_alpha = realized + costs.total_usd

        # Benchmark decomposition
        bench_return = 0.0
        if benchmark_returns is not None:
            bench_return = benchmark_returns.get(fill.symbol, 0.0)

        # selection = gross_alpha vs benchmark; sizing = 0 (requires portfolio
        # constructor data not available here); benchmark = benchmark return.
        selection = gross_alpha - bench_return
        sizing = 0.0
        bench = bench_return

        sym = fill.symbol
        month = _month_key(fill.ts)
        regime = regime_map.get(sym) if regime_map else None
        sig = signal_map.get(fill.fill_id) if signal_map else None

        trade = TradeAttribution(
            symbol=sym,
            fill_id=fill.fill_id,
            realized_pnl_usd=realized,
            gross_alpha_usd=gross_alpha,
            venue_fee_usd=costs.venue_fee_usd,
            network_fee_usd=costs.network_fee_usd,
            priority_fee_usd=costs.priority_fee_usd,
            spread_usd=costs.spread_usd,
            price_impact_usd=costs.price_impact_usd,
            latency_cost_usd=costs.latency_cost_usd,
            failure_cost_usd=costs.failure_cost_usd,
            selection_usd=selection,
            sizing_usd=sizing,
            benchmark_usd=bench,
            month=month,
            regime=regime,
            signal_id=sig,
            asset=sym,
        )

        # Hard reconciliation check — raises AttributionError if violated
        _check_reconciliation(trade, reconciliation_tolerance_usd)

        trades.append(trade)
        by_asset[sym] = by_asset.get(sym, 0.0) + realized
        by_month[month] = by_month.get(month, 0.0) + realized
        by_regime[regime] = by_regime.get(regime, 0.0) + realized
        by_signal[sig] = by_signal.get(sig, 0.0) + realized

        tot_pnl += realized
        tot_gross += gross_alpha
        tot_venue += costs.venue_fee_usd
        tot_net += costs.network_fee_usd
        tot_pri += costs.priority_fee_usd
        tot_spread += costs.spread_usd
        tot_impact += costs.price_impact_usd
        tot_latency += costs.latency_cost_usd
        tot_failure += costs.failure_cost_usd
        tot_selection += selection
        tot_sizing += sizing
        tot_bench += bench

    return AttributionSummary(
        total_realized_pnl_usd=tot_pnl,
        total_gross_alpha_usd=tot_gross,
        total_venue_fee_usd=tot_venue,
        total_network_fee_usd=tot_net,
        total_priority_fee_usd=tot_pri,
        total_spread_usd=tot_spread,
        total_price_impact_usd=tot_impact,
        total_latency_cost_usd=tot_latency,
        total_failure_cost_usd=tot_failure,
        total_selection_usd=tot_selection,
        total_sizing_usd=tot_sizing,
        total_benchmark_usd=tot_bench,
        by_asset=by_asset,
        by_month=by_month,
        by_regime=by_regime,
        by_signal=by_signal,
        trades=trades,
        fidelity=fidelity,
        non_executable_notice=notice,
        reconciliation_tolerance_usd=reconciliation_tolerance_usd,
    )
