"""Swap loader — scaffolding. No swap-level data has been collected yet.

This repository currently has no decoded on-chain swap history: the only
collected data is OHLCV bars (see ``ohlcv.py``) and, once ``shadow.py`` has
run, prospective quote ladders (see ``quote_ladders.py``). This module exists
so that the moment a swap collector is built, it has an on-disk convention and
a loader ready rather than needing both designed at once.

Storage convention (not yet produced by anything): one gzipped-JSONL file per
pool at ``<root>/swaps/<pool_id>.jsonl.gz``, one JSON object per line matching
the field names of ``schemas.SwapRecord`` (``asset_id``, ``pool_id``, ``venue``,
``event_time``, ``available_time``, ``received_time``, ``in_mint``, ``out_mint``,
``in_amount_atomic``, ``out_amount_atomic``, ``price_impact_pct``,
``fees_atomic``, ``side``, ``slot``, ``tx_signature``, ``source``, and the
common ``schema_version``/``ingestion_version``/``quality_flags``/``sequence``
fields). This mirrors the append-only-JSONL-stream convention
``BACKTEST-CONTRACTS.md`` §2 requires for raw event data, and the per-pool
keying mirrors ``backfill.series_path`` for the same reason documented there:
a swap is priced by the pool it executed against, and concatenating swaps from
two pools for the same mint would be a synthetic-jump join.

``available_time`` for a decoded swap is the time this process finished
decoding the transaction, not the block time — the same publication-delay
principle as a bar, applied to a much smaller and source-dependent delay that
a swap collector must set explicitly when it exists.

Fidelity
========

``FIDELITY_TIER = FidelityTier.TIER_1``: individual swaps plus conservative
cost estimates, one tier below the exact size-specific ladders TIER_2 needs.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from memetrader.histdata.loaders import DataAvailability
from memetrader.histdata.schemas import SwapRecord
from memetrader.types import EventKind, FidelityTier, HistoricalEvent

__all__ = ["FIDELITY_TIER", "available", "partition_path", "stream_partition"]

FIDELITY_TIER = FidelityTier.TIER_1


def partition_path(root: Path, pool_id: str) -> Path:
    """``<root>/swaps/<pool_id>.jsonl.gz`` — see module docstring for why
    pool-keyed rather than mint-keyed."""
    return root / "swaps" / f"{pool_id}.jsonl.gz"


def available(root: Path, pool_id: str) -> DataAvailability:
    """Whether a swap partition exists for ``pool_id``, and if so, whether it
    has rows.

    Always ``NOT_COLLECTED`` today for every pool: nothing in this repository
    writes to ``partition_path``. This function is not a stub that always
    returns the same constant, though — it genuinely checks the filesystem, so
    it starts reporting correctly the day a collector begins writing here
    without any change to this module.
    """
    path = partition_path(root, pool_id)
    if not path.exists():
        return DataAvailability.NOT_COLLECTED
    for _ in _read_rows(path):
        return DataAvailability.PRESENT
    return DataAvailability.PRESENT_EMPTY


def stream_partition(
    root: Path, pool_id: str, *, source: str = "swaps"
) -> Iterator[HistoricalEvent]:
    """Lazily stream one pool's swaps as ``HistoricalEvent``s.

    Yields nothing when the partition does not exist, which today is every
    pool. Rows are required to be pre-sorted ascending by ``event_time`` on
    disk (the collector's responsibility, mirroring
    ``backfill.write_series``'s oldest-first contract); this loader does not
    re-sort, since re-sorting would require materialising the whole file and
    defeat the laziness the event queue depends on.
    """
    path = partition_path(root, pool_id)
    if not path.exists():
        return
    event_source = f"{source}:{pool_id}"
    for row in _read_rows(path):
        record = _row_to_record(row)
        yield HistoricalEvent(
            kind=EventKind.SWAP,
            available_time=record.available_time,
            asset_id=record.asset_id,
            payload=record,
            event_time=record.event_time,
            received_time=record.received_time,
            sequence=record.sequence,
            source=event_source,
            pool_id=record.pool_id,
            quality_flags=record.quality_flags,
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


def _row_to_record(row: dict[str, Any]) -> SwapRecord:
    return SwapRecord(
        asset_id=row["asset_id"],
        pool_id=row["pool_id"],
        venue=row["venue"],
        event_time=float(row["event_time"]),
        available_time=float(row["available_time"]),
        received_time=float(row["received_time"]),
        in_mint=row["in_mint"],
        out_mint=row["out_mint"],
        in_amount_atomic=int(row["in_amount_atomic"]),
        out_amount_atomic=int(row["out_amount_atomic"]),
        price_impact_pct=(
            float(row["price_impact_pct"])
            if row.get("price_impact_pct") is not None
            else None
        ),
        fees_atomic=int(row["fees_atomic"]),
        side=row["side"],
        slot=row.get("slot"),
        tx_signature=row.get("tx_signature"),
        source=row["source"],
        quality_flags=tuple(row.get("quality_flags") or ()),
        sequence=row.get("sequence"),
    )
