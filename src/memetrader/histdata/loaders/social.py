"""Social signal loader — scaffolding. No Reddit/Arctic-Shift capture wired yet.

Built directly against ``schemas.SocialEventRecord``, which already encodes
the central rule for this data source: ``available_time`` is the collector's
receipt time, never the underlying post's creation time. A post created at
10:00 but collected at 10:03 was not available at 10:00 — replaying against
post-creation time hands the strategy three minutes of foresight, and that
three minutes is exactly the kind of gap a 0.2% edge lives or dies on.

``observed_through`` (on ``SocialEventRecord``) matters for the same reason:
an indexer like Arctic Shift runs behind live Reddit, so the most recent
collection window is structurally incomplete. A loader that ignores this
field would let a replay treat "we haven't finished indexing this window yet"
as "we looked and found nothing" — exactly the ``None`` vs. ``0`` collapse
this package's ``DataAvailability`` type and ``schemas.py``'s convention both
exist to prevent.

Fidelity tier — a deliberate judgment call
===========================================

``FIDELITY_TIER`` is set to ``FidelityTier.TIER_0`` here, not because social
data is worthless, but because ``FidelityTier`` is defined purely in terms of
*execution* realism — whether a given tier can support a PnL claim
(``permits_pnl_claim``) — and social signal presence or richness does not
change whether a fill can be priced. A run with rich social data and only
OHLCV execution data is still TIER_0 for PnL purposes; a run with TIER_2
execution data and zero social data can still be promoted. This is flagged
explicitly because it is a genuine judgment call, not a fact read directly off
``FidelityTier``'s definition — a future maintainer who disagrees should treat
this as the place to start the conversation, not as a bug.

Storage convention (not yet produced by anything): one gzipped-JSONL file per
mint at ``<root>/social/<mint>.jsonl.gz``, one JSON object per line matching
``SocialEventRecord``'s fields, oldest-first by ``received_time``.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from memetrader.histdata.loaders import DataAvailability
from memetrader.histdata.schemas import SocialEventRecord
from memetrader.types import EventKind, FidelityTier, HistoricalEvent

__all__ = ["FIDELITY_TIER", "available", "partition_path", "stream_partition"]

#: See the module docstring: this is a deliberate judgment call, not a value
#: read directly off ``FidelityTier``'s own definition.
FIDELITY_TIER = FidelityTier.TIER_0


def partition_path(root: Path, asset_id: str) -> Path:
    """``<root>/social/<asset_id>.jsonl.gz``."""
    return root / "social" / f"{asset_id}.jsonl.gz"


def available(root: Path, asset_id: str) -> DataAvailability:
    """Whether a social partition exists for ``asset_id``, and if so, whether
    it has rows.

    Always ``NOT_COLLECTED`` today — nothing writes to ``partition_path`` yet.
    A future collector that runs and finds zero mentions for a quiet coin
    should still write a row (or an explicit empty-window marker) so this
    reports ``PRESENT_EMPTY`` rather than looking identical to "never ran".
    """
    path = partition_path(root, asset_id)
    if not path.exists():
        return DataAvailability.NOT_COLLECTED
    for _ in _read_rows(path):
        return DataAvailability.PRESENT
    return DataAvailability.PRESENT_EMPTY


def stream_partition(
    root: Path, asset_id: str, *, source: str = "social"
) -> Iterator[HistoricalEvent]:
    """Lazily stream one mint's social observations as ``HistoricalEvent``s.

    Yields nothing when the partition does not exist, which today is every
    mint. Rows must be pre-sorted ascending by ``received_time`` on disk (the
    collector's responsibility); this loader does not re-sort.
    """
    path = partition_path(root, asset_id)
    if not path.exists():
        return
    event_source = f"{source}:{asset_id}"
    for row in _read_rows(path):
        record = _row_to_record(row)
        yield HistoricalEvent(
            kind=EventKind.SOCIAL,
            available_time=record.available_time,
            asset_id=record.asset_id,
            payload=record,
            event_time=record.event_time,
            received_time=record.received_time,
            sequence=record.sequence,
            source=event_source,
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


def _row_to_record(row: dict[str, Any]) -> SocialEventRecord:
    return SocialEventRecord(
        asset_id=row["asset_id"],
        event_time=float(row["event_time"]),
        available_time=float(row["available_time"]),
        received_time=float(row["received_time"]),
        source=row["source"],
        mention_velocity_1h=(
            float(row["mention_velocity_1h"])
            if row.get("mention_velocity_1h") is not None
            else None
        ),
        mention_velocity_24h=(
            float(row["mention_velocity_24h"])
            if row.get("mention_velocity_24h") is not None
            else None
        ),
        mention_zscore_7d=(
            float(row["mention_zscore_7d"])
            if row.get("mention_zscore_7d") is not None
            else None
        ),
        unique_contributors_24h=(
            int(row["unique_contributors_24h"])
            if row.get("unique_contributors_24h") is not None
            else None
        ),
        contributor_to_post_ratio=(
            float(row["contributor_to_post_ratio"])
            if row.get("contributor_to_post_ratio") is not None
            else None
        ),
        observed_through=(
            float(row["observed_through"])
            if row.get("observed_through") is not None
            else None
        ),
        baseline_hours=int(row["baseline_hours"]),
        quality_flags=tuple(row.get("quality_flags") or ()),
        sequence=row.get("sequence"),
    )
