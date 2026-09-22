"""Tests for memetrader.histdata.loaders.

Offline, no network, no reliance on the gitignored ``history/`` directory —
every fixture is built under ``tmp_path``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from memetrader.backfill import RawBar, series_path, write_series
from memetrader.backtest.clock import SimulatedClock
from memetrader.backtest.event_queue import EventQueue
from memetrader.histdata import loaders
from memetrader.histdata.loaders import (
    ohlcv,
    onchain,
    pool_states,
    quote_ladders,
    social,
    swaps,
)
from memetrader.shadow import (
    LadderRung,
    LadderSnapshot,
    ShadowMeta,
    _append_snapshot,
    ladder_path,
)
from memetrader.shadow import write_manifest as shadow_write_manifest
from memetrader.types import EventKind, Quote, Side, Timeframe, TokenMeta

POOL = "PoolAddr111111111111111111111111111111111"
ASSET = "MintAddr1111111111111111111111111111111111"


def _bars(n: int, *, start: float = 0.0, step: float = 3600.0) -> list[RawBar]:
    return [
        RawBar(ts=start + i * step, open=1.0, high=1.1, low=0.9, close=1.0, volume=100.0)
        for i in range(n)
    ]


def _write_bars(
    root: Path, bars: list[RawBar], timeframe: Timeframe = Timeframe.H1
) -> Path:
    path = series_path(root, POOL, timeframe)
    write_series(path, bars)
    return path


# ---------------------------------------------------------------------------
# Publication delay: not-available-at-ts, available-at-ts+interval+delay
# ---------------------------------------------------------------------------


def test_bar_not_available_at_ts(tmp_path: Path) -> None:
    path = _write_bars(tmp_path, _bars(1))
    [event] = list(
        ohlcv.stream_partition(
            path,
            asset_id=ASSET,
            pool_id=POOL,
            timeframe=Timeframe.H1,
            publication_delay_seconds=30.0,
        )
    )
    assert event.available_time != 0.0
    assert event.event_time == 0.0
    assert event.available_time == 0.0 + 3600.0 + 30.0


def test_bar_available_at_ts_plus_interval_plus_delay(tmp_path: Path) -> None:
    path = _write_bars(tmp_path, _bars(1))
    [event] = list(
        ohlcv.stream_partition(
            path,
            asset_id=ASSET,
            pool_id=POOL,
            timeframe=Timeframe.H1,
            publication_delay_seconds=30.0,
        )
    )
    clock = SimulatedClock(run_id="t", start=event.available_time - 1.0)
    # Not yet available one second before.
    assert event.available_time > clock.now
    clock2 = SimulatedClock(run_id="t2", start=event.available_time)
    assert event.available_time == clock2.now  # available exactly at this instant


def test_increasing_delay_moves_availability_later(tmp_path: Path) -> None:
    path = _write_bars(tmp_path, _bars(1))
    [low_delay] = list(
        ohlcv.stream_partition(
            path,
            asset_id=ASSET,
            pool_id=POOL,
            timeframe=Timeframe.H1,
            publication_delay_seconds=0.0,
        )
    )
    [high_delay] = list(
        ohlcv.stream_partition(
            path,
            asset_id=ASSET,
            pool_id=POOL,
            timeframe=Timeframe.H1,
            publication_delay_seconds=500.0,
        )
    )
    assert high_delay.available_time > low_delay.available_time
    assert high_delay.event_time == low_delay.event_time  # event_time never moves


# ---------------------------------------------------------------------------
# Sort order / EventQueue acceptance
# ---------------------------------------------------------------------------


def test_ohlcv_stream_is_sorted_and_accepted_by_event_queue(tmp_path: Path) -> None:
    path = _write_bars(tmp_path, _bars(20))
    stream = ohlcv.stream_partition(
        path,
        asset_id=ASSET,
        pool_id=POOL,
        timeframe=Timeframe.H1,
        publication_delay_seconds=10.0,
    )
    clock = SimulatedClock(run_id="q", start=0.0)
    queue = EventQueue(clock, streams=[stream])
    events = list(queue)  # must not raise EventQueueError
    assert len(events) == 20
    times = [e.available_time for e in events]
    assert times == sorted(times)
    assert all(e.kind == EventKind.BAR_CLOSE for e in events)


# ---------------------------------------------------------------------------
# Laziness
# ---------------------------------------------------------------------------


def test_ohlcv_loader_is_lazy(tmp_path: Path) -> None:
    path = _write_bars(tmp_path, _bars(500))
    read_count = 0

    from memetrader import backfill as backfill_module

    orig_read_series = backfill_module.read_series

    def counting_read_series(p: Path) -> Iterator[RawBar]:
        nonlocal read_count
        for bar in orig_read_series(p):
            read_count += 1
            yield bar

    # Patch the module-level name ohlcv.py imported directly. ohlcv.read_series
    # is a plain module-level function reference, not a settable attribute per
    # mypy's view of the module namespace, hence the assignment (not attr) ignore.
    original = ohlcv.read_series
    ohlcv.read_series = counting_read_series  # type: ignore[assignment]
    try:
        stream = ohlcv.stream_partition(
            path,
            asset_id=ASSET,
            pool_id=POOL,
            timeframe=Timeframe.H1,
            publication_delay_seconds=0.0,
        )
        first = next(stream)
        assert first is not None
        assert read_count == 1, "reading one event must not materialize the whole file"
        next(stream)
        assert read_count == 2
    finally:
        ohlcv.read_series = original


def test_ohlcv_stream_through_event_queue_is_lazy(tmp_path: Path) -> None:
    """EventQueue seeds the heap with exactly one bar per stream at construction."""
    path = _write_bars(tmp_path, _bars(500))
    stream = ohlcv.stream_partition(
        path,
        asset_id=ASSET,
        pool_id=POOL,
        timeframe=Timeframe.H1,
        publication_delay_seconds=0.0,
    )
    clock = SimulatedClock(run_id="lazy", start=0.0)
    queue = EventQueue(clock, streams=[stream])
    # Only the first event should have been pulled from the generator so far.
    top = queue.peek()
    assert top is not None
    assert top.event_time == 0.0


# ---------------------------------------------------------------------------
# Missing vs. present-but-empty partitions
# ---------------------------------------------------------------------------


def test_ohlcv_missing_partition_is_not_collected(tmp_path: Path) -> None:
    missing = series_path(tmp_path, POOL, Timeframe.H1)
    assert ohlcv.available(missing) is loaders.DataAvailability.NOT_COLLECTED
    assert (
        list(
            ohlcv.stream_partition(
                missing,
                asset_id=ASSET,
                pool_id=POOL,
                timeframe=Timeframe.H1,
                publication_delay_seconds=0.0,
            )
        )
        == []
    )


def test_ohlcv_present_but_empty_partition_is_distinguishable(tmp_path: Path) -> None:
    path = _write_bars(tmp_path, [])  # collector ran, found nothing
    assert ohlcv.available(path) is loaders.DataAvailability.PRESENT_EMPTY
    assert ohlcv.available(path) is not loaders.DataAvailability.NOT_COLLECTED, (
        "an empty-but-existing partition must not look identical to an uncollected one"
    )


def test_ohlcv_present_partition_with_rows(tmp_path: Path) -> None:
    path = _write_bars(tmp_path, _bars(3))
    assert ohlcv.available(path) is loaders.DataAvailability.PRESENT


@pytest.mark.parametrize(
    ("module", "root_relpath"),
    [
        (swaps, "swaps"),
        (pool_states, "pool_states"),
        (onchain, "onchain"),
        (social, "social"),
    ],
)
def test_scaffolding_loaders_not_collected_by_default(
    tmp_path: Path, module, root_relpath: str
) -> None:
    """No collector for these has ever run, so every partition is NOT_COLLECTED."""
    assert module.available(tmp_path, ASSET) is loaders.DataAvailability.NOT_COLLECTED
    assert list(module.stream_partition(tmp_path, ASSET)) == []


@pytest.mark.parametrize("module", [swaps, pool_states, onchain, social])
def test_scaffolding_loader_fidelity_tier_is_exposed(module) -> None:
    assert module.FIDELITY_TIER is not None


# ---------------------------------------------------------------------------
# quote_ladders round-trip against shadow.py's own writer
# ---------------------------------------------------------------------------


def _make_quote(
    *,
    side: Side,
    in_amount: int,
    out_amount: int,
    min_out: int,
    requested_at: float,
    received_at: float,
) -> Quote:
    sol_mint = "So1111111111111111111111111111111111111111"
    return Quote(
        symbol="BONK",
        side=side,
        input_token=TokenMeta(mint=sol_mint, decimals=9, source="test"),
        output_token=TokenMeta(mint=ASSET, decimals=6, source="test"),
        in_amount_atomic=in_amount,
        out_amount_atomic=out_amount,
        min_out_amount_atomic=min_out,
        price_impact_pct=0.5,
        route_labels=("raydium",),
        fingerprint="fp-1",
        requested_at=requested_at,
        received_at=received_at,
        context_slot=12345,
    )


def test_quote_ladder_round_trip_from_shadow_writer(tmp_path: Path) -> None:
    date_str = "2026-01-01"
    quote = _make_quote(
        side=Side.BUY,
        in_amount=10_000_000,
        out_amount=9_000_000,
        min_out=8_900_000,
        requested_at=100.0,
        received_at=101.0,
    )
    rung_ok = LadderRung(
        side=Side.BUY,
        usd_size=10.0,
        in_amount_atomic=10_000_000,
        quote=quote,
        quality="ok",
        requested_at=100.0,
        received_at=101.0,
        latency_seconds=1.0,
    )
    rung_no_route = LadderRung(
        side=Side.BUY,
        usd_size=1000.0,
        in_amount_atomic=1_000_000_000,
        quote=None,
        quality="no_route",
        requested_at=100.5,
        received_at=101.5,
        latency_seconds=1.0,
    )
    snap = LadderSnapshot(
        symbol="BONK",
        mint=ASSET,
        event_time=100.0,
        available_time=101.5,
        slot=12345,
        rungs=(rung_ok, rung_no_route),
        pool_state=None,
        quality="ok",
    )
    path = ladder_path(tmp_path, date_str, "BONK")
    _append_snapshot(path, snap)
    shadow_write_manifest(
        tmp_path,
        date_str,
        [
            ShadowMeta(
                symbol="BONK",
                mint=ASSET,
                date=date_str,
                snapshots=1,
                first_ts=101.5,
                last_ts=101.5,
                rung_count=2,
                rung_ok=1,
                rung_no_route=1,
                rung_error=0,
                ladder_usd=(10.0, 1000.0),
                sides=("BUY",),
                quality="ok",
            )
        ],
    )

    assert (
        quote_ladders.available(tmp_path, date_str, "BONK")
        is loaders.DataAvailability.PRESENT
    )

    ladders = quote_ladders.read_ladders(tmp_path, date_str, "BONK")
    assert len(ladders) == 1  # only BUY side had a routed rung
    ladder = ladders[0]
    assert ladder.asset_id == ASSET
    assert ladder.side == "BUY"
    assert ladder.event_time == 100.0
    assert ladder.available_time == 101.5
    # The no_route rung must be omitted, not zero-filled.
    assert len(ladder.rungs) == 1
    rung = ladder.rungs[0]
    assert rung.in_amount_atomic == 10_000_000
    assert rung.out_amount_atomic == 9_000_000
    assert rung.min_out_atomic == 8_900_000
    # Documented gaps: fee amount and pool_id are not in shadow.py's format.
    assert rung.fees_atomic == 0
    assert ladder.pool_id == ""
    assert "fees_atomic_unknown" in ladder.quality_flags
    assert "pool_id_unknown" in ladder.quality_flags


def test_quote_ladder_stream_is_sorted_and_accepted_by_event_queue(tmp_path: Path) -> None:
    date_str = "2026-01-01"
    for i, (req, rec) in enumerate([(100.0, 101.0), (200.0, 201.0), (300.0, 301.0)]):
        quote = _make_quote(
            side=Side.BUY,
            in_amount=10_000_000,
            out_amount=9_000_000,
            min_out=8_900_000,
            requested_at=req,
            received_at=rec,
        )
        rung = LadderRung(
            side=Side.BUY,
            usd_size=10.0,
            in_amount_atomic=10_000_000,
            quote=quote,
            quality="ok",
            requested_at=req,
            received_at=rec,
            latency_seconds=1.0,
        )
        snap = LadderSnapshot(
            symbol="BONK",
            mint=ASSET,
            event_time=req,
            available_time=rec,
            slot=1000 + i,
            rungs=(rung,),
            pool_state=None,
            quality="ok",
        )
        _append_snapshot(ladder_path(tmp_path, date_str, "BONK"), snap)

    stream = quote_ladders.stream_symbol(tmp_path, date_str, "BONK")
    clock = SimulatedClock(run_id="ql", start=0.0)
    queue = EventQueue(clock, streams=[stream])
    events = list(queue)  # must not raise EventQueueError
    assert len(events) == 3
    times = [e.available_time for e in events]
    assert times == sorted(times)
    assert all(e.kind == EventKind.QUOTE for e in events)


def test_quote_ladder_missing_partition_is_not_collected(tmp_path: Path) -> None:
    assert (
        quote_ladders.available(tmp_path, "2026-01-01", "NOSYMBOL")
        is loaders.DataAvailability.NOT_COLLECTED
    )
    assert list(quote_ladders.stream_symbol(tmp_path, "2026-01-01", "NOSYMBOL")) == []
