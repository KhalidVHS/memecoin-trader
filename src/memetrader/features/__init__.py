"""Feature computation layer for the backtest engine.

This package implements the feature pipeline described in
``docs/BACKTEST-CONTRACTS.md``. Every module here reads history exclusively
through the ``PointInTimeState`` protocol, which enforces the central invariant:
at simulated time ``t``, no feature may use data with ``available_time > t``.

Submodules:

    ``registry``       — Named registry of feature definitions. Each definition
                         carries a mandatory warm-up length (in bars and in
                         wall-clock seconds) that the embargo calculation
                         consumes. A feature that cannot state its warm-up
                         cannot be embargoed correctly.

    ``pipeline``       — Prefix-safe feature computation. ``fit(train_slice)``
                         and ``transform(slice)`` are separate so that fitting
                         on test data is a visible API misuse rather than a
                         quiet default.

    ``market``         — Price/volume features: returns over horizons, realized
                         volatility, volume ratios, abnormal volume, residual
                         momentum. Gap handling is the load-bearing concern:
                         missing bars mean no trades, not forward-filled price.

    ``microstructure`` — Spread, depth, price impact. Requires TIER_2 data;
                         returns ``None`` at TIER_0.

    ``onchain``        — Holder concentration, wallet flow, smart wallet labels.
                         Requires TIER_1+; returns ``None`` at TIER_0. Smart
                         wallet labels enforce per-window computation so they
                         cannot leak future trades.

    ``attention``      — Social/attention features. Available time is collector
                         receipt time, not post creation time. Currently
                         returns ``None`` because ``[sentiment] enabled = false``.
"""

from __future__ import annotations

from .attention import (
    contributor_to_post_ratio,
    mention_velocity_1h,
    mention_velocity_24h,
    mention_zscore_7d,
    unique_contributors_24h,
)
from .market import (
    abnormal_volume,
    realized_vol_pct,
    residual_momentum,
    return_pct,
    sector_return,
    volume_ratio,
)
from .microstructure import (
    depth_usd_at_size,
    price_impact_pct_at_size,
    quote_ladder_age_seconds,
    spread_bps,
)
from .onchain import (
    holder_count,
    net_wallet_flow,
    smart_wallet_labels,
    top_holder_concentration,
)
from .pipeline import (
    FeatureResult,
    Pipeline,
    PointInTimeState,
    RobustScaler,
    StandardScaler,
)
from .registry import FeatureDefinition, FeatureRegistry

__all__ = [
    # registry
    "FeatureDefinition",
    "FeatureRegistry",
    # pipeline
    "FeatureResult",
    "Pipeline",
    "PointInTimeState",
    "RobustScaler",
    "StandardScaler",
    # market
    "abnormal_volume",
    "realized_vol_pct",
    "residual_momentum",
    "return_pct",
    "sector_return",
    "volume_ratio",
    # microstructure
    "depth_usd_at_size",
    "price_impact_pct_at_size",
    "quote_ladder_age_seconds",
    "spread_bps",
    # onchain
    "holder_count",
    "net_wallet_flow",
    "smart_wallet_labels",
    "top_holder_concentration",
    # attention
    "contributor_to_post_ratio",
    "mention_velocity_1h",
    "mention_velocity_24h",
    "mention_zscore_7d",
    "unique_contributors_24h",
]
