"""Quote-ladder loader — reads exactly what ``shadow.py`` writes.

``shadow.py`` prospectively collects Jupiter quote ladders into
``<root>/shadow/<date>/<symbol>.jsonl.gz`` (one :class:`LadderSnapshot` per
line, gzip in append mode — multi-member gzip, which ``gzip.open`` handles
transparently) plus a per-day manifest at
``<root>/shadow/<date>/manifest.json``. This is the TIER_2 path: rung-level
depth data lets the execution model interpolate size-specific cost instead of
assuming a flat slippage.

Two structural gaps between what ``shadow.py`` stores and what
``schemas.QuoteLadder``/``QuoteLadderRung`` require (both required fields,
neither optional):

* ``QuoteLadderRung.fees_atomic: int`` — the shadow collector's ``Quote``
  object (see ``quotes.py``) never captures a fee amount. There is no honest
  non-zero value to reconstruct here, and guessing one would misstate cost.
  This loader sets ``fees_atomic=0`` and appends ``"fees_atomic_unknown"`` to
  the record's ``quality_flags`` — a confident zero would be a silent lie
  (§0: ``None`` means "could not find out", ``0`` means "looked, and it is
  quiet" — this is the former, dressed as the latter only because the schema
  has no ``int | None`` slot for it), so the flag is what carries the honest
  "we could not find out" fact instead.
* ``QuoteLadder.pool_id: str`` — ``LadderSnapshot`` is keyed by symbol, not
  pool (Jupiter picks the route; ``shadow.py`` records which AMMs it touched
  in each rung's ``route_labels``, never a pool address). This loader sets
  ``pool_id=""`` and appends ``"pool_id_unknown"`` to ``quality_flags`` rather
  than repurposing a venue label (e.g. ``"raydium"``) as a fake pool address —
  a venue name is not a pool address and treating it as one would silently
  corrupt any code that joins on ``pool_id``.

Rungs whose ``quote`` is ``None`` (``quality`` is ``"no_route"``,
``"parse_error"``, or ``"degraded"``) are omitted from the reconstructed
ladder rather than being zero-filled. This matches
``QuoteLadder.best_rung_for``'s documented semantics (``None`` return means
"we never probed a rung at or below this size", not "the pool has zero
depth") and the project-wide rule that an absence must never collapse into a
zero.

Fidelity
========

``FIDELITY_TIER = FidelityTier.TIER_2``. A ladder with at least one real rung
supports size-specific execution cost estimation, which is what separates
TIER_2 from a flat bar-based slippage assumption.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from memetrader.histdata.loaders import DataAvailability
from memetrader.histdata.schemas import QuoteLadder, QuoteLadderRung
from memetrader.shadow import ShadowMeta, ladder_path, read_manifest
from memetrader.types import EventKind, FidelityTier, HistoricalEvent, Side

__all__ = [
    "FIDELITY_TIER",
    "available",
    "read_ladders",
    "stream_symbol",
]

FIDELITY_TIER = FidelityTier.TIER_2

# Rung qualities that mean "Jupiter returned an actual route". Every other
# quality string ("no_route", "parse_error", "degraded") means the rung's
# ``quote`` field is None and must be omitted, not zero-filled.
_ROUTED_QUALITY = "ok"


def available(root: Path, date_str: str, symbol: str) -> DataAvailability:
    """Whether ``symbol``'s shadow file exists for ``date_str``, and if so,
    whether it has snapshot lines.

    Distinguishing "never collected" from "collected and empty" here means
    checking two independent facts: does the file exist, and does the day's
    manifest record a snapshot count for this symbol (or, absent a manifest
    entry, does the file itself have at least one line).
    """
    path = ladder_path(root, date_str, symbol)
    if not path.exists():
        return DataAvailability.NOT_COLLECTED
    meta = _manifest_entry(root, date_str, symbol)
    if meta is not None:
        return (
            DataAvailability.PRESENT
            if meta.snapshots > 0
            else DataAvailability.PRESENT_EMPTY
        )
    for _ in _read_snapshot_dicts(path):
        return DataAvailability.PRESENT
    return DataAvailability.PRESENT_EMPTY


def stream_symbol(
    root: Path,
    date_str: str,
    symbol: str,
    *,
    source: str = "shadow-jupiter",
) -> Iterator[HistoricalEvent]:
    """Lazily stream one symbol-day's ladder snapshots as ``HistoricalEvent``s.

    Two events per snapshot line (one per side present in the snapshot's
    rungs), each carrying a reconstructed :class:`QuoteLadder` payload. Yields
    nothing when the partition does not exist — call :func:`available` first
    if the caller needs to distinguish that from "exists but empty".

    Snapshots within one file are appended in observation order by
    ``collect()``, and ``available_time`` (the last rung's ``received_at``) is
    monotonically non-decreasing across a sweep, so the stream satisfies
    ``EventQueue``'s non-decreasing ``sort_key`` requirement. ``sequence`` is
    set from the snapshot's ``slot`` where known, giving a finer ordering
    token than wall-clock seconds when two snapshots share a second.
    """
    path = ladder_path(root, date_str, symbol)
    if not path.exists():
        return
    event_source = f"{source}:{symbol}"
    for snap in _read_snapshot_dicts(path):
        for ladder in _ladders_from_snapshot(snap, source=event_source):
            yield HistoricalEvent(
                kind=EventKind.QUOTE,
                available_time=ladder.available_time,
                asset_id=snap["mint"],
                payload=ladder,
                event_time=ladder.event_time,
                received_time=ladder.available_time,
                sequence=snap.get("slot"),
                source=f"{event_source}:{ladder.side}",
                pool_id=ladder.pool_id or None,
                quality_flags=ladder.quality_flags,
            )


def read_ladders(root: Path, date_str: str, symbol: str) -> list[QuoteLadder]:
    """Eagerly read every reconstructed ``QuoteLadder`` for one symbol-day.

    A convenience for tests and small offline checks; :func:`stream_symbol` is
    the lazy interface real replay code should use.
    """
    path = ladder_path(root, date_str, symbol)
    if not path.exists():
        return []
    out: list[QuoteLadder] = []
    for snap in _read_snapshot_dicts(path):
        out.extend(_ladders_from_snapshot(snap, source=f"shadow-jupiter:{symbol}"))
    return out


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _manifest_entry(root: Path, date_str: str, symbol: str) -> ShadowMeta | None:
    for m in read_manifest(root, date_str):
        if m.symbol == symbol:
            return m
    return None


def _read_snapshot_dicts(path: Path) -> Iterator[dict[str, Any]]:
    """Stream parsed JSON snapshot dicts from a shadow ladder file.

    Mirrors ``backfill.read_series``'s laziness: one line read per element
    yielded. ``gzip.open`` in text mode transparently concatenates the
    multi-member gzip stream ``_append_snapshot`` produces (one member per
    call), so this reads exactly what was written regardless of how many
    ``_append_snapshot`` calls built the file.
    """
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _ladders_from_snapshot(snap: dict[str, Any], *, source: str) -> list[QuoteLadder]:
    """Reconstruct one ``QuoteLadder`` per side present in a snapshot dict.

    A snapshot mixes BUY and SELL rungs (``LadderConfig.sides`` defaults to
    both); ``QuoteLadder`` is per-side, so this splits the snapshot by
    ``rung["side"]`` and builds one ladder per side that has at least one
    routed rung. A side with zero routed rungs (every rung was no_route,
    parse_error, or degraded) produces no ladder at all — an empty
    ``QuoteLadder`` would misreport "we have depth data with zero rungs"
    when the truth is "every probe at this size, this side, failed".
    """
    by_side: dict[str, list[QuoteLadderRung]] = {}
    for rung in snap.get("rungs") or []:
        if rung.get("quality") != _ROUTED_QUALITY or rung.get("quote") is None:
            continue
        q = rung["quote"]
        by_side.setdefault(rung["side"], []).append(
            QuoteLadderRung(
                in_amount_atomic=int(rung["in_amount_atomic"]),
                out_amount_atomic=int(q["out_amount_atomic"]),
                price_impact_pct=float(q["price_impact_pct"])
                if q["price_impact_pct"] is not None
                else 0.0,
                route_labels=tuple(q.get("route_labels") or ()),
                fees_atomic=0,  # shadow.py's Quote never captures a fee amount
                min_out_atomic=int(q["min_out_amount_atomic"]),
            )
        )

    ladders: list[QuoteLadder] = []
    for side_str, rungs in by_side.items():
        try:
            side = Side(side_str)
        except ValueError:
            continue
        ladders.append(
            QuoteLadder(
                asset_id=snap["mint"],
                pool_id="",  # shadow.py records venue labels, never a pool address
                side=side.value,
                event_time=float(snap["event_time"]),
                available_time=float(snap["available_time"]),
                received_time=float(snap["available_time"]),
                rungs=tuple(rungs),
                context_slot=snap.get("slot"),
                source=source,
                quality_flags=("fees_atomic_unknown", "pool_id_unknown"),
            )
        )
    return ladders
