"""Baseline strategies and portfolio construction for the backtest engine.

See ``docs/BACKTEST-CONTRACTS.md`` §2: this package is plural (``strategies/``)
because ``strategy.py`` (singular) already owns the live-trading ``Strategy``
protocol and ``BaselineStrategy``. Everything here is backtest-only and reads
history exclusively through ``histdata.point_in_time.PointInTimeState``.

``baselines.py`` holds the reference strategies every candidate model must
beat. ``construction.py`` turns strategy proposals into cash-safe, cap-respecting
orders. See each module's docstring for the details, in particular the
documented rule for what happens when a token leaves the point-in-time
universe (referenced from both modules, applied consistently).
"""

from __future__ import annotations

from .baselines import (
    BacktestStrategy,
    BuyAndHoldStrategy,
    RandomEntryStrategy,
    RandomTradeSpec,
    TradeStat,
    build_random_entry_baseline,
    equal_weight_strategy,
    matched_random_schedule,
    mean_reversion_strategy,
    momentum_strategy,
)
from .construction import (
    MICRO_USD,
    ConstructionLimits,
    EqualRiskForecastConstructor,
    PortfolioConstructor,
    size_orders,
)

__all__ = [
    "MICRO_USD",
    "BacktestStrategy",
    "BuyAndHoldStrategy",
    "ConstructionLimits",
    "EqualRiskForecastConstructor",
    "PortfolioConstructor",
    "RandomEntryStrategy",
    "RandomTradeSpec",
    "TradeStat",
    "build_random_entry_baseline",
    "equal_weight_strategy",
    "matched_random_schedule",
    "mean_reversion_strategy",
    "momentum_strategy",
    "size_orders",
]
