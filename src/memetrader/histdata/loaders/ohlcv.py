"""OHLCV loader — the only loader with real data today.

Reads the gzipped-JSONL bar files ``backfill.py`` writes
(``<root>/<pool>/<timeframe>.jsonl.gz``, ~801,957 bars across 24 coins, 1h and
5m) and turns each closed bar into a :class:`~memetrader.types.HistoricalEvent`
with ``kind=EventKind.BAR_CLOSE``.

Publication delay
==================

A bar opens at ``ts`` and closes at ``ts + interval``. It is not published by
the vendor at that exact instant, so ``available_time`` is
``ts + interval + publication_delay_seconds`` — never ``ts``, and never even
``ts + interval`` unless the caller explicitly sets the delay to zero.
``publication_delay_seconds`` is always a caller-supplied parameter (mirroring
``backtest.config.BacktestConfig.publication_delay_seconds`` and
``histdata.point_in_time.ReplayState.publication_delay_seconds``); this module
does not hardcode a value, and increasing it strictly delays availability —
never event_time, since the bar still happened when it happened.

Laziness
========

``backfill.read_series`` is a generator that streams one JSON line at a time
from the gzip file. :func:`stream_partition` wraps it with another generator,
so pulling the first event from the returned iterator reads exactly one line
of the underlying file — the ~801,957-bar dataset is never materialised as a
list. This is what lets ``EventQueue`` hold O(k) bars in memory for k active
streams instead of the whole catalog.

Fidelity
========

``FIDELITY_TIER = FidelityTier.TIER_0``. A closed bar tells you a price moved;
it says nothing about size-specific execution cost, so no PnL claim may rest
on this loader alone (see ``FidelityTier.permits_pnl_claim``).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from memetrader.backfill import RawBar, read_series, series_path
from memetrader.histdata.loaders import DataAvailability
from memetrader.histdata.schemas import CandleRecord
from memetrader.types import EventKind, FidelityTier, HistoricalEvent, Timeframe

if TYPE_CHECKING:
    from memetrader.histdata.catalog import Catalog

__all__ = [
    "FIDELITY_TIER",
    "INTERVAL_SECONDS",
    "available",
    "available_time_for",
    "interval_seconds",
    "stream_partition",
    "streams_for_catalog",
]

#: What this loader can support when its data is present. A ceiling, not a
#: claim: whether a given run actually reaches TIER_0 is decided by the run
#: manifest, not by this constant.
FIDELITY_TIER = FidelityTier.TIER_0

#: Bar length in seconds per timeframe. Duplicated (not imported) from
#: ``histdata.point_in_time._INTERVAL_SECONDS`` deliberately: that name is
#: private to that module, and this loader must not depend on point_in_time's
#: internals to compute the same public fact. Keep these two dicts in sync if
#: a new timeframe is ever added.
INTERVAL_SECONDS: dict[Timeframe, float] = {
    Timeframe.M5: 300.0,
    Timeframe.H1: 3600.0,
}


def interval_seconds(timeframe: Timeframe) -> float:
    """Bar length in seconds for ``timeframe``."""
    return INTERVAL_SECONDS[timeframe]


def available_time_for(
    ts: float, timeframe: Timeframe, *, publication_delay_seconds: float
) -> float:
    """When a bar opened at ``ts`` becomes knowable to a replay.

    ``ts + interval + publication_delay_seconds`` — never ``ts``. Increasing
    ``publication_delay_seconds`` strictly increases the result; it never
    moves ``event_time`` (the bar's open), only how long the replay must wait
    before it may act on the bar.
    """
    return ts + interval_seconds(timeframe) + publication_delay_seconds


def available(path: Path) -> DataAvailability:
    """Whether a partition file exists, and if so, whether it has rows.

    Reads at most one line to answer "does it have rows" — this must stay
    cheap, since a caller may check availability for many partitions before
    deciding which ones to stream.
    """
    if not path.exists():
        return DataAvailability.NOT_COLLECTED
    for _ in read_series(path):
        return DataAvailability.PRESENT
    return DataAvailability.PRESENT_EMPTY


def stream_partition(
    path: Path,
    *,
    asset_id: str,
    pool_id: str,
    timeframe: Timeframe,
    publication_delay_seconds: float,
    source: str = "geckoterminal-backfill",
) -> Iterator[HistoricalEvent]:
    """Lazily stream one (pool, timeframe) partition as sorted ``HistoricalEvent``s.

    Each yielded event carries a :class:`~memetrader.histdata.schemas.CandleRecord`
    payload with ``closed=True`` (the only kind ``backfill.py`` ever stores) and
    ``available_time`` computed by :func:`available_time_for`. Bars are read
    oldest-first, which ``backfill.write_series`` guarantees on write, so the
    resulting stream is non-decreasing in ``available_time`` and therefore in
    ``sort_key`` — the property ``EventQueue`` requires.

    Yields nothing (not an error) when ``path`` does not exist — use
    :func:`available` first if you need to distinguish "no partition" from
    "partition exists but is empty" or "genuinely mid-stream".
    """
    if not path.exists():
        return
    interval = interval_seconds(timeframe)
    # Unique per (pool, timeframe) so two partitions for the same asset never
    # collide on sort_key even when a bar from each lands at the identical
    # available_time — EventQueue raises on duplicate sort_keys, and the
    # queue merges many partition streams together.
    event_source = f"{source}:{pool_id}:{timeframe.value}"
    for raw in read_series(path):
        yield _bar_event(
            raw,
            asset_id=asset_id,
            pool_id=pool_id,
            timeframe=timeframe,
            interval=interval,
            publication_delay_seconds=publication_delay_seconds,
            source=event_source,
        )


def streams_for_catalog(
    catalog: Catalog,
    *,
    pool_to_asset: dict[str, str],
    publication_delay_seconds: float,
    timeframe: Timeframe | None = None,
    source: str = "geckoterminal-backfill",
) -> list[Iterator[HistoricalEvent]]:
    """Build one lazy stream per partition known to ``catalog``.

    ``pool_to_asset`` maps pool address to mint address — the catalog only
    knows about pools, not which mint each belongs to (mirroring
    ``ReplayState.load_from_catalog``'s ``pool_to_asset`` parameter). Pools not
    present in the mapping are skipped rather than raising: a catalog scan may
    legitimately find partitions for pools outside the caller's universe.

    Returns a list rather than one merged stream because ``EventQueue`` itself
    is the merge point — passing many small pre-sorted streams is exactly the
    lazy heap-merge design the queue exists for.
    """
    streams: list[Iterator[HistoricalEvent]] = []
    for info in catalog.partitions():
        asset_id = pool_to_asset.get(info.pool)
        if asset_id is None:
            continue
        try:
            tf = Timeframe(info.timeframe)
        except ValueError:
            continue
        if timeframe is not None and tf is not timeframe:
            continue
        path = series_path(catalog.root, info.pool, tf)
        streams.append(
            stream_partition(
                path,
                asset_id=asset_id,
                pool_id=info.pool,
                timeframe=tf,
                publication_delay_seconds=publication_delay_seconds,
                source=source,
            )
        )
    return streams


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _bar_event(
    raw: RawBar,
    *,
    asset_id: str,
    pool_id: str,
    timeframe: Timeframe,
    interval: float,
    publication_delay_seconds: float,
    source: str,
) -> HistoricalEvent:
    event_time = raw.ts
    available_time = event_time + interval + publication_delay_seconds
    record = CandleRecord(
        asset_id=asset_id,
        pool_id=pool_id,
        timeframe=timeframe.value,
        ts=raw.ts,
        event_time=event_time,
        available_time=available_time,
        received_time=available_time,
        open=raw.open,
        high=raw.high,
        low=raw.low,
        close=raw.close,
        volume=raw.volume,
        closed=True,
        source=source,
    )
    return HistoricalEvent(
        kind=EventKind.BAR_CLOSE,
        available_time=available_time,
        asset_id=asset_id,
        payload=record,
        event_time=event_time,
        received_time=available_time,
        source=source,
        pool_id=pool_id,
    )
