"""On-chain metadata loader — scaffolding. No collector exists yet.

**Judgment call flagged for review**: ``schemas.py`` has no record type for
token-metadata changes (renames, symbol/decimals/authority changes), and this
loaders package was told not to modify ``schemas.py``. Rather than silently
repurposing an unrelated schema (``UniverseMembershipRecord`` is close but
means something different — eligibility, not identity) or inventing a schema
change unilaterally, this module defines its own minimal, clearly-provisional
record type, :class:`TokenMetadataChange`, scoped to this file. If on-chain
metadata tracking becomes a real feature, this type should move into
``schemas.py`` proper (versioned alongside the others) rather than staying
here — flagging that migration explicitly rather than doing it, since
``schemas.py`` is out of scope for this loaders package.

The central rule this loader exists to enforce: **a rename is not
retroactive.** If a token symbol changes at time T, a replay running before T
must still see the old symbol; a replay running at or after T sees the new
one. This is why ``TokenMetadataChange`` carries ``available_time`` (the
collector's observation time — the earliest a replay may treat the new value
as current) separately from ``event_time`` (when the on-chain change actually
landed), following exactly the same discipline as every other loader in this
package.

Storage convention (not yet produced by anything): one gzipped-JSONL file per
mint at ``<root>/onchain/<mint>.jsonl.gz``, one JSON object per line matching
``TokenMetadataChange``'s fields, oldest-first.

Fidelity
========

``FIDELITY_TIER = FidelityTier.TIER_1``: on-chain metadata is orthogonal to
execution-cost fidelity in the strict sense used by ``FidelityTier``, but a
loader that can identify *which* on-chain events occurred (not just bar
prices) sits alongside swap-level TIER_1 data rather than bar-only TIER_0.
"""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memetrader.histdata.loaders import DataAvailability
from memetrader.types import EventKind, FidelityTier, HistoricalEvent

__all__ = [
    "FIDELITY_TIER",
    "TokenMetadataChange",
    "available",
    "partition_path",
    "stream_partition",
]

FIDELITY_TIER = FidelityTier.TIER_1

#: Same versioning discipline as ``schemas.py``: bump when this provisional
#: shape changes incompatibly.
SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class TokenMetadataChange:
    """One observed change to a token's on-chain metadata.

    ``field_name`` is the name of the changed attribute (``"symbol"``,
    ``"decimals"``, ``"mint_authority"``, ...); ``old_value``/``new_value``
    are its string representations (kept as strings so this one record type
    covers every field without a union type). ``None`` for ``old_value`` means
    this is the first observation of the field, not a change from a known
    prior value — the same ``None`` != ``0``/empty-string convention as
    everywhere else in this package.
    """

    asset_id: str  # mint address
    field_name: str
    old_value: str | None
    new_value: str
    event_time: float  # when the on-chain change landed
    available_time: float  # when the collector observed it — never retroactive
    received_time: float
    source: str
    schema_version: str = SCHEMA_VERSION
    quality_flags: tuple[str, ...] = ()
    sequence: int | None = None

    def __post_init__(self) -> None:
        for name in ("event_time", "available_time", "received_time"):
            v = getattr(self, name)
            if not math.isfinite(v):
                raise ValueError(f"TokenMetadataChange.{name} must be finite, got {v}")
        if self.available_time < self.event_time:
            raise ValueError(
                f"TokenMetadataChange for {self.asset_id}.{self.field_name}: "
                f"available_time {self.available_time} < event_time {self.event_time} "
                "— a metadata change cannot be knowable before it happens, and a "
                "rename must never be applied retroactively"
            )


def partition_path(root: Path, asset_id: str) -> Path:
    """``<root>/onchain/<asset_id>.jsonl.gz``."""
    return root / "onchain" / f"{asset_id}.jsonl.gz"


def available(root: Path, asset_id: str) -> DataAvailability:
    """Whether an on-chain metadata partition exists for ``asset_id``, and if
    so, whether it has rows.

    Always ``NOT_COLLECTED`` today — nothing writes to ``partition_path`` yet.
    """
    path = partition_path(root, asset_id)
    if not path.exists():
        return DataAvailability.NOT_COLLECTED
    for _ in _read_rows(path):
        return DataAvailability.PRESENT
    return DataAvailability.PRESENT_EMPTY


def stream_partition(
    root: Path, asset_id: str, *, source: str = "onchain"
) -> Iterator[HistoricalEvent]:
    """Lazily stream one mint's metadata changes as ``HistoricalEvent``s.

    Yields nothing when the partition does not exist, which today is every
    mint. There is no dedicated ``EventKind`` for metadata changes; ``MARK``
    is reused (priority 60, between state updates and decisions) since a
    metadata change is best modelled as an update to the replay's view of the
    world rather than a tradable signal or an order-side event.
    """
    path = partition_path(root, asset_id)
    if not path.exists():
        return
    event_source = f"{source}:{asset_id}"
    for row in _read_rows(path):
        change = _row_to_change(row)
        yield HistoricalEvent(
            kind=EventKind.MARK,
            available_time=change.available_time,
            asset_id=change.asset_id,
            payload=change,
            event_time=change.event_time,
            received_time=change.received_time,
            sequence=change.sequence,
            source=event_source,
            quality_flags=change.quality_flags,
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


def _row_to_change(row: dict[str, Any]) -> TokenMetadataChange:
    return TokenMetadataChange(
        asset_id=row["asset_id"],
        field_name=row["field_name"],
        old_value=row.get("old_value"),
        new_value=row["new_value"],
        event_time=float(row["event_time"]),
        available_time=float(row["available_time"]),
        received_time=float(row["received_time"]),
        source=row["source"],
        quality_flags=tuple(row.get("quality_flags") or ()),
        sequence=row.get("sequence"),
    )
