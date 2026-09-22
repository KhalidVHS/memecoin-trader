"""Historical data layer for the backtest engine.

This package is the only legal gateway through which the backtest reads history.
Every piece of recorded data — candles, pool states, universe membership, quote
ladders — must flow through ``point_in_time.ReplayState``, which enforces the
central invariant: at simulated time ``t``, nothing with ``available_time > t``
may be read.

The package is named ``histdata`` rather than ``data`` for the reason recorded in
``BACKTEST-CONTRACTS.md §2``: ``.gitignore`` line 2 was an unanchored ``data/``,
and ``Config.data_dir`` already occupies the "data" namespace in config.

Submodules:
    schemas         — versioned record types for every stored stream
    catalog         — immutable partition discovery and content hashing
    quality         — per-asset/timeframe data-quality reports
    point_in_time   — PointInTimeState protocol and ReplayState implementation
    universe        — point-in-time tradable-universe reconstruction
"""

from __future__ import annotations

from .catalog import Catalog, PartitionInfo
from .point_in_time import PointInTimeState, ReplayState
from .quality import QualityReport, usable_for
from .schemas import (
    CandleRecord,
    LabelRecord,
    PoolState,
    QuoteLadder,
    QuoteLadderRung,
    SocialEventRecord,
    SwapRecord,
    UniverseMembershipRecord,
)
from .universe import UniverseCatalog, UniverseEntry

__all__ = [
    "CandleRecord",
    "Catalog",
    "LabelRecord",
    "PartitionInfo",
    "PointInTimeState",
    "PoolState",
    "QualityReport",
    "QuoteLadder",
    "QuoteLadderRung",
    "ReplayState",
    "SocialEventRecord",
    "SwapRecord",
    "UniverseCatalog",
    "UniverseEntry",
    "UniverseMembershipRecord",
    "usable_for",
]
