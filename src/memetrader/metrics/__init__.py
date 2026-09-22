"""Metrics layer — performance, attribution, benchmarks and capacity.

This package turns an equity curve and a fill/execution-report log into the
numbers the promotion gate, the experiment registry and the CLI report read.
See ``docs/BACKTEST-CONTRACTS.md`` §6 and §8 for the accounting invariants
these modules are held to.

Submodules:

    ``performance``  — Risk-adjusted return ratios (Sharpe, Sortino, Calmar),
                       drawdown, trade-level and portfolio-level statistics.
                       ``periods_per_year`` is always an explicit argument;
                       there is no hardcoded annualisation constant.

    ``attribution``  — PnL decomposition. Every ``TradeAttribution`` reconciles
                       ``gross_alpha_usd`` minus every cost bucket to
                       ``realized_pnl_usd``; a breach raises ``AttributionError``
                       rather than reporting a silently wrong number.

    ``benchmarks``   — Cash/no-trade, random-entry, random-asset, equal-weight
                       basket and buy-and-hold-SOL nulls, each returned as a
                       ``BenchmarkResult`` whose ``.metrics`` is directly
                       comparable to the strategy's own ``PerformanceMetrics``.

    ``capacity``     — Position-level and strategy-level capacity/participation
                       limits, derived from size-specific price impact.
"""

from __future__ import annotations

from .attribution import (
    AttributionError,
    AttributionSummary,
    TradeAttribution,
    attribute_pnl,
)
from .benchmarks import (
    BenchmarkResult,
    buy_hold_sol_benchmark,
    cash_benchmark,
    equal_weight_basket_benchmark,
    random_asset_benchmark,
    random_entry_benchmark,
)
from .capacity import (
    CapacityReport,
    PositionCapacity,
    estimate_capacity,
    liquidation_value,
)
from .performance import (
    FoldMetrics,
    PerformanceMetrics,
    compute_performance,
    max_drawdown,
)

__all__ = [
    "AttributionError",
    "AttributionSummary",
    "BenchmarkResult",
    "CapacityReport",
    "FoldMetrics",
    "PerformanceMetrics",
    "PositionCapacity",
    "TradeAttribution",
    "attribute_pnl",
    "buy_hold_sol_benchmark",
    "cash_benchmark",
    "compute_performance",
    "equal_weight_basket_benchmark",
    "estimate_capacity",
    "liquidation_value",
    "max_drawdown",
    "random_asset_benchmark",
    "random_entry_benchmark",
]
