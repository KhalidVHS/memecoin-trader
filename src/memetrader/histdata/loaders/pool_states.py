"""Pool-state loader — scaffolding. No pool-state snapshots exist yet.

Distinct from a quote ladder (``quote_ladders.py``): a pool state is the raw
on-chain reserves/fee-tier/tick-range you would see by reading the pool
account directly, not a Jupiter route output for a specific input size. Either
one alone (per ``point_in_time.PointInTimeState.pool_state`` /
``.quote_ladder``) is enough to reach TIER_2; today this repository has
neither.

Storage convention (not yet produced by anything): one gzipped-JSONL file per
pool at ``<root>/pool_states/<pool_id>.jsonl.gz``, one JSON object per line
matching ``schemas.PoolState``'s fields (``asset_id``, ``pool_id``, ``venue``,
``event_time``, ``available_time``, ``received_time``, ``reserve_in_atomic``,
``reserve_out_atomic``, ``fee_rate_bps``, ``price_usd``, ``liquidity_usd``,
``source``, plus the common provenance/versioning fields).

``available_time`` is the time this process finished reading the account, not
the slot's block time — a snapshot read is a live RPC call whenever it exists,
so the honest floor is receipt time, exactly as for a swap.

Fidelity
========

``FIDELITY_TIER = FidelityTier.TIER_2``: reserves plus a fee tier are enough to
compute exact constant-product execution cost at any size, which is what
separates this from a flat bar-based slippage assumption.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from memetrader.histdata.loaders import DataAvailability
from memetrader.histdata.schemas import PoolState
from memetrader.types import EventKind, FidelityTier, HistoricalEvent

__all__ = ["FIDELITY_TIER", "available", "partition_path", "stream_partition"]

FIDELITY_TIER = FidelityTier.TIER_2


def partition_path(root: Path, pool_id: str) -> Path:
    """``<root>/pool_states/<pool_id>.jsonl.gz``."""
    return root / "pool_states" / f"{pool_id}.jsonl.gz"


def available(root: Path, pool_id: str) -> DataAvailability:
    """Whether a pool-state partition exists for ``pool_id``, and if so,
    whether it has rows.

    Always ``NOT_COLLECTED`` today — nothing writes to ``partition_path`` yet.
    """
    path = partition_path(root, pool_id)
    if not path.exists():
        return DataAvailability.NOT_COLLECTED
    for _ in _read_rows(path):
        return DataAvailability.PRESENT
    return DataAvailability.PRESENT_EMPTY


def stream_partition(
    root: Path, pool_id: str, *, source: str = "pool_states"
) -> Iterator[HistoricalEvent]:
    """Lazily stream one pool's state snapshots as ``HistoricalEvent``s.

    Yields nothing when the partition does not exist, which today is every
    pool. Rows must be pre-sorted ascending by ``event_time`` on disk (the
    collector's responsibility); this loader does not re-sort.
    """
    path = partition_path(root, pool_id)
    if not path.exists():
        return
    event_source = f"{source}:{pool_id}"
    for row in _read_rows(path):
        state = _row_to_state(row)
        yield HistoricalEvent(
            kind=EventKind.POOL_STATE,
            available_time=state.available_time,
            asset_id=state.asset_id,
            payload=state,
            event_time=state.event_time,
            received_time=state.received_time,
            sequence=state.sequence,
            source=event_source,
            pool_id=state.pool_id,
            quality_flags=state.quality_flags,
        )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _read_rows(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _row_to_state(row: dict[str, Any]) -> PoolState:
    return PoolState(
        asset_id=row["asset_id"],
        pool_id=row["pool_id"],
        venue=row["venue"],
        event_time=float(row["event_time"]),
        available_time=float(row["available_time"]),
        received_time=float(row["received_time"]),
        reserve_in_atomic=int(row["reserve_in_atomic"]),
        reserve_out_atomic=int(row["reserve_out_atomic"]),
        fee_rate_bps=int(row["fee_rate_bps"]),
        price_usd=(float(row["price_usd"]) if row.get("price_usd") is not None else None),
        liquidity_usd=(
            float(row["liquidity_usd"]) if row.get("liquidity_usd") is not None else None
        ),
        source=row["source"],
        quality_flags=tuple(row.get("quality_flags") or ()),
        slot=row.get("slot"),
        sequence=row.get("sequence"),
    )
