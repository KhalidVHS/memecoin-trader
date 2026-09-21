"""Offline tests for the historical backfill.

Nothing here touches the network. An ``httpx.MockTransport`` answers every
route and a fake clock drives every timestamp, which is a hard rule rather than
a preference: a test that needs GeckoTerminal to be up is a test that will one
day fail for a reason that has nothing to do with this code, and a backfill test
that really paged a year of history would take an hour.

The cases are chosen around the places a backfill actually goes wrong — the
duplicate bar at every page boundary, the still-forming newest bar, a vendor
that omits quiet intervals, and the time-reversed payload this codebase has
already been bitten by once.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import httpx
import pytest

from memetrader.backfill import (
    BackfillError,
    Gap,
    RawBar,
    RejectedRow,
    SeriesMeta,
    _find_gaps,
    _merge,
    _parse_page,
    backfill,
    fetch_series,
    read_manifest,
    read_series,
    series_path,
    to_candle_series,
    window_gaps,
    write_manifest,
    write_series,
)
from memetrader.http import make_client
from memetrader.types import DataQuality, Timeframe

BASE = "https://api.geckoterminal.com/api/v2"
POOL = "EP2ib6dYdEeqD8MfE2ezHCxX3kP3K2eLKkirfPm5eyMx"
MINT = "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm"

# A round epoch-seconds anchor inside the 2020..2100 sanity window, on an exact
# hour so bars land on the grid without arithmetic in every test.
T0 = 1_760_000_400.0
HOUR = 3600.0


def row(ts: float, close: float = 10.0, volume: float = 5.0) -> list[float]:
    """One vendor OHLCV row: [ts, o, h, l, c, v]."""
    return [ts, close, close + 1.0, close - 1.0, close, volume]


def payload(rows: list[list[float]]) -> dict[str, object]:
    return {"data": {"attributes": {"ohlcv_list": rows}}}


def client_for(handler) -> httpx.Client:
    return make_client(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_page_returns_oldest_first_from_newest_first_payload() -> None:
    """The vendor serves newest-first; every indicator walks forward in time.

    This is the bug the codebase hit once already (market.py:675). If it ever
    regresses, EMAs and ATR compute over time-reversed data and produce
    confident garbage rather than failing.
    """
    rows = [row(T0 + 2 * HOUR), row(T0 + HOUR), row(T0)]
    bars, rejected = _parse_page(rows, now=T0 + 10 * HOUR)

    assert [b.ts for b in bars] == [T0, T0 + HOUR, T0 + 2 * HOUR]
    assert rejected == []


def test_parse_page_rejects_rather_than_raises_on_bad_rows() -> None:
    """A malformed row must not abort an hour-long walk, but must leave evidence.

    This is the one deliberate divergence from ``market._parse_candle_rows``,
    which raises. Rejecting-and-recording keeps the bad row visible in the
    manifest instead of discarding it silently.
    """
    rows = [
        row(T0),
        [T0 + HOUR, 10.0, 5.0, 9.0, 10.0, 1.0],  # low > high
        [T0 + 2 * HOUR, 10.0, 11.0, 9.0, None, 1.0],  # null close
        [T0 + 3 * HOUR],  # too short
        [T0 + 4 * HOUR, 0.0, 1.0, 0.0, 0.5, 1.0],  # non-positive open
        [T0 + 5 * HOUR, 10.0, 11.0, 9.0, 10.0, -1.0],  # negative volume
        row(T0 + 6 * HOUR),
    ]
    bars, rejected = _parse_page(rows, now=T0 + 100 * HOUR)

    assert [b.ts for b in bars] == [T0, T0 + 6 * HOUR]
    assert len(rejected) == 5
    assert all(isinstance(r, RejectedRow) and r.reason and r.raw for r in rejected)


def test_parse_page_rejects_future_dated_bars() -> None:
    """A bar that opens in the future cannot be an observation."""
    bars, rejected = _parse_page([row(T0), row(T0 + 5 * HOUR)], now=T0 + HOUR)

    assert [b.ts for b in bars] == [T0]
    assert "future-dated" in rejected[0].reason


def test_parse_page_normalises_millisecond_timestamps() -> None:
    """Catches a vendor unit change rather than shifting bars 56,000 years out."""
    bars, rejected = _parse_page([row(T0 * 1000.0)], now=T0 + HOUR)

    assert rejected == []
    assert bars[0].ts == pytest.approx(T0)


def test_parse_page_rejects_non_finite_prices() -> None:
    """NaN survives float() and then poisons every mean and stdev downstream."""
    bars, rejected = _parse_page([[T0, float("nan"), 11.0, 9.0, 10.0, 1.0]], now=T0 + HOUR)

    assert bars == []
    assert rejected


# ---------------------------------------------------------------------------
# Merging and gaps
# ---------------------------------------------------------------------------


def test_merge_dedups_the_shared_bar_at_a_page_boundary() -> None:
    """Page boundaries share exactly one bar. Dedup is mandatory, not defensive.

    Without it ``_find_gaps`` sees a zero delta and the series is refused —
    which is precisely why ``market.py``'s parser cannot be reused here.
    """
    page_a = [RawBar(T0 + HOUR, 1.0, 1.0, 1.0, 1.0, 1.0)]
    page_b = [
        RawBar(T0, 2.0, 2.0, 2.0, 2.0, 2.0),
        RawBar(T0 + HOUR, 9.0, 9.0, 9.0, 9.0, 9.0),  # duplicate ts
    ]

    merged = _merge([page_a, page_b])

    assert [b.ts for b in merged] == [T0, T0 + HOUR]
    assert merged[1].open == 1.0  # first seen wins


def test_find_gaps_records_absent_intervals_without_filling_them() -> None:
    """Missing is never zero: a hole is recorded, never forward- or zero-filled."""
    bars = [
        RawBar(T0, 1.0, 1.0, 1.0, 1.0, 1.0),
        RawBar(T0 + 4 * HOUR, 1.0, 1.0, 1.0, 1.0, 1.0),  # 3 bars absent
        RawBar(T0 + 5 * HOUR, 1.0, 1.0, 1.0, 1.0, 1.0),
    ]

    gaps, missing = _find_gaps(bars, HOUR)

    assert missing == 3
    assert gaps == [Gap(first_missing_ts=T0 + HOUR, last_missing_ts=T0 + 3 * HOUR, bars=3)]


def test_find_gaps_raises_on_off_grid_spacing() -> None:
    """Off-grid bars invalidate every window length in signals.py."""
    bars = [
        RawBar(T0, 1.0, 1.0, 1.0, 1.0, 1.0),
        RawBar(T0 + 1800.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    ]

    with pytest.raises(BackfillError, match="not a multiple"):
        _find_gaps(bars, HOUR)


def test_window_gaps_counts_only_the_overlap() -> None:
    """A backtest asks this before trusting a feature window."""
    meta = SeriesMeta(
        symbol="WIF",
        mint=MINT,
        pool=POOL,
        timeframe="1h",
        interval_seconds=HOUR,
        rows=10,
        first_ts=T0,
        last_ts=T0 + 10 * HOUR,
        gaps=(Gap(T0 + 2 * HOUR, T0 + 5 * HOUR, 4),),
        missing_bars=4,
    )

    assert window_gaps(meta, T0, T0 + 20 * HOUR) == 4
    assert window_gaps(meta, T0 + 3 * HOUR, T0 + 4 * HOUR) == 2
    assert window_gaps(meta, T0 + 6 * HOUR, T0 + 9 * HOUR) == 0


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def test_fetch_series_pages_backwards_and_stops_at_since() -> None:
    """Two pages, a shared boundary bar, and a clean stop once ``since`` is passed."""
    seen: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        before = request.url.params.get("before_timestamp")
        if before is None:
            rows = [row(T0 + i * HOUR) for i in range(10, 4, -1)]
        else:
            rows = [row(T0 + i * HOUR) for i in range(5, -1, -1)]
        return httpx.Response(200, json=payload(rows))

    with client_for(handler) as client:
        bars, meta = fetch_series(
            client,
            symbol="WIF",
            mint=MINT,
            pool=POOL,
            timeframe=Timeframe.H1,
            since=T0,
            base_url=BASE,
            pace_seconds=0.0,
            now=lambda: T0 + 100 * HOUR,
            sleep=lambda _: None,
        )

    assert len(seen) == 2
    assert seen[1]["before_timestamp"] == str(int(T0 + 5 * HOUR))
    assert [b.ts for b in bars] == [T0 + i * HOUR for i in range(11)]
    assert meta.rows == 11
    assert meta.missing_bars == 0
    assert meta.quality == DataQuality.OK.value


def test_fetch_series_never_writes_the_still_forming_bar() -> None:
    """History stores closed bars only, so no reader can forget to filter.

    ``now`` sits half an hour into the bar opening at ``T0 + 3h``, so that bar
    is incomplete and must not survive.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=payload([row(T0 + i * HOUR) for i in range(3, -1, -1)])
        )

    with client_for(handler) as client:
        bars, meta = fetch_series(
            client,
            symbol="WIF",
            mint=MINT,
            pool=POOL,
            timeframe=Timeframe.H1,
            since=T0,
            base_url=BASE,
            pace_seconds=0.0,
            max_pages=1,
            now=lambda: T0 + 3 * HOUR + 1800.0,
            sleep=lambda _: None,
        )

    assert [b.ts for b in bars] == [T0, T0 + HOUR, T0 + 2 * HOUR]
    assert meta.last_ts == T0 + 2 * HOUR


def test_fetch_series_stops_when_the_vendor_stops_producing_history() -> None:
    """A pool younger than ``since`` must not page forever."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, json=payload([row(T0 + i * HOUR) for i in range(2, -1, -1)])
        )

    with client_for(handler) as client:
        bars, meta = fetch_series(
            client,
            symbol="WIF",
            mint=MINT,
            pool=POOL,
            timeframe=Timeframe.H1,
            since=T0 - 10_000 * HOUR,
            base_url=BASE,
            pace_seconds=0.0,
            now=lambda: T0 + 100 * HOUR,
            sleep=lambda _: None,
        )

    assert calls == 2  # second page repeats the window, so the walk stops
    assert meta.pages == 2
    assert len(bars) == 3


def test_fetch_series_marks_a_gapped_series_degraded() -> None:
    """Gaps degrade the series, which the existing forecast gate already refuses."""

    def handler(request: httpx.Request) -> httpx.Response:
        rows = [row(T0 + 5 * HOUR), row(T0 + HOUR), row(T0)]
        return httpx.Response(200, json=payload(rows))

    with client_for(handler) as client:
        _, meta = fetch_series(
            client,
            symbol="WIF",
            mint=MINT,
            pool=POOL,
            timeframe=Timeframe.H1,
            since=T0,
            base_url=BASE,
            pace_seconds=0.0,
            max_pages=1,
            now=lambda: T0 + 100 * HOUR,
            sleep=lambda _: None,
        )

    assert meta.missing_bars == 3
    assert meta.quality == DataQuality.DEGRADED.value
    assert meta.quality_reason is not None and "missing" in meta.quality_reason


def test_fetch_series_treats_422_as_range_not_rate_limit() -> None:
    """Recorded finding: 422 means 'narrow the range', never 'slow down'.

    Retrying a 422 as if it were rate limiting is an infinite backoff against a
    request that will never succeed.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"errors": [{"status": "422"}]})

    with client_for(handler) as client, pytest.raises(BackfillError, match="422"):
        fetch_series(
            client,
            symbol="WIF",
            mint=MINT,
            pool=POOL,
            timeframe=Timeframe.H1,
            since=T0,
            base_url=BASE,
            pace_seconds=0.0,
            now=lambda: T0 + 100 * HOUR,
            sleep=lambda _: None,
        )


def test_fetch_series_keeps_its_bars_when_the_horizon_is_reached() -> None:
    """A 401 mid-walk is a horizon, not a failure.

    Measured against the live vendor: the keyless tier answers 401 with "You
    can only access data from the past 180 days with Public API" once
    ``before_timestamp`` passes that mark. Raising here is what made a first
    real run write zero files while holding thousands of good bars in memory.
    The bars survive, the walk stops, and the metadata says why the series
    starts where it does.
    """
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json=payload([row(T0 + i * HOUR) for i in range(3)]))
        return httpx.Response(
            401,
            json={
                "errors": [
                    {
                        "status": "401",
                        "title": (
                            "You can only access data from the past 180 days "
                            "with Public API."
                        ),
                    }
                ]
            },
        )

    with client_for(handler) as client:
        bars, meta = fetch_series(
            client,
            symbol="WIF",
            mint=MINT,
            pool=POOL,
            timeframe=Timeframe.H1,
            since=T0 - 5000 * HOUR,
            base_url=BASE,
            pace_seconds=0.0,
            now=lambda: T0 + 100 * HOUR,
            sleep=lambda _: None,
        )

    assert [b.ts for b in bars] == [T0, T0 + HOUR, T0 + 2 * HOUR]
    assert meta.horizon_reached is True
    assert meta.quality_reason is not None
    assert "180-day" in meta.quality_reason
    # The horizon is not a data defect: these three bars are perfectly good.
    assert meta.quality == DataQuality.OK.value


def test_fetch_series_retries_a_429() -> None:
    """The live path forgoes http.execute's retry; this one must not.

    POPCAT lost 38 of 39 live fetches to 429s. A backfill that gives up on the
    first one is not worth running.
    """
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json=payload([row(T0)]))

    with client_for(handler) as client:
        bars, _ = fetch_series(
            client,
            symbol="WIF",
            mint=MINT,
            pool=POOL,
            timeframe=Timeframe.H1,
            since=T0,
            base_url=BASE,
            pace_seconds=0.0,
            max_pages=1,
            now=lambda: T0 + 100 * HOUR,
            sleep=lambda _: None,
        )

    assert attempts == 2
    assert len(bars) == 1


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def test_write_then_read_series_round_trips(tmp_path: Path) -> None:
    bars = [
        RawBar(T0 + i * HOUR, 1.0 + i, 2.0 + i, 0.5 + i, 1.5 + i, 10.0 * i)
        for i in range(5)
    ]
    path = series_path(tmp_path, POOL, Timeframe.H1)

    write_series(path, bars)

    assert path.name == "1h.jsonl.gz"
    assert path.parent.name == POOL  # keyed by pool, never by symbol
    assert list(read_series(path)) == bars


def test_write_series_is_byte_stable_across_runs(tmp_path: Path) -> None:
    """An unchanged pull must produce an identical file, so a re-run is a visible no-op."""
    bars = [RawBar(T0, 1.0, 2.0, 0.5, 1.5, 3.0)]
    first = tmp_path / "a.jsonl.gz"
    second = tmp_path / "b.jsonl.gz"

    write_series(first, bars)
    write_series(second, bars)

    assert first.read_bytes() == second.read_bytes()


def test_write_series_refuses_bars_that_are_not_oldest_first(tmp_path: Path) -> None:
    bars = [RawBar(T0 + HOUR, 1.0, 1.0, 1.0, 1.0, 1.0), RawBar(T0, 1.0, 1.0, 1.0, 1.0, 1.0)]

    with pytest.raises(BackfillError, match="oldest-first"):
        write_series(tmp_path / "x.jsonl.gz", bars)


def test_stored_rows_are_plain_json_readable(tmp_path: Path) -> None:
    """The format has to stay boring: one JSON object per line, no framing."""
    path = tmp_path / "x.jsonl.gz"
    write_series(path, [RawBar(T0, 1.0, 2.0, 0.5, 1.5, 3.0)])

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    assert rows == [{"ts": T0, "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 3.0}]


def test_manifest_round_trips_including_gaps_and_rejects(tmp_path: Path) -> None:
    """A dataset without provenance is not evidence, so this must survive a reload."""
    meta = SeriesMeta(
        symbol="WIF",
        mint=MINT,
        pool=POOL,
        timeframe="1h",
        interval_seconds=HOUR,
        rows=3,
        first_ts=T0,
        last_ts=T0 + 2 * HOUR,
        gaps=(Gap(T0 + HOUR, T0 + HOUR, 1),),
        missing_bars=1,
        rejected=(RejectedRow(reason="low > high", raw="[1,2,3]"),),
        pages=2,
        fetched_at=T0,
        base_url=BASE,
        quality=DataQuality.DEGRADED.value,
        quality_reason="1 missing bars in 1 gaps",
    )

    write_manifest(tmp_path, [meta])
    loaded = read_manifest(tmp_path)

    assert loaded == [meta]
    assert loaded[0].complete is False


def test_read_manifest_is_empty_when_absent(tmp_path: Path) -> None:
    assert read_manifest(tmp_path) == []


# ---------------------------------------------------------------------------
# Consumption
# ---------------------------------------------------------------------------


def test_to_candle_series_yields_only_closed_bars_bound_to_the_pool() -> None:
    bars = [RawBar(T0 + i * HOUR, 1.0, 2.0, 0.5, 1.5, 3.0) for i in range(3)]

    series = to_candle_series(bars, timeframe=Timeframe.H1, pool=POOL)

    assert series.pool_address == POOL
    assert series.interval_seconds == HOUR
    assert len(series.closed_candles) == len(series.candles) == 3
    assert series.complete
    # event_time is the newest bar's *open*, so a series can never look fresher
    # than it is.
    assert series.provenance.event_time == T0 + 2 * HOUR


def test_to_candle_series_degrades_a_gapped_window() -> None:
    """No new gate: BaselineStrategy._forecast already refuses DEGRADED."""
    bars = [
        RawBar(T0, 1.0, 2.0, 0.5, 1.5, 3.0),
        RawBar(T0 + 3 * HOUR, 1.0, 2.0, 0.5, 1.5, 3.0),
    ]

    series = to_candle_series(bars, timeframe=Timeframe.H1, pool=POOL)

    assert series.missing_intervals == 2
    assert series.provenance.quality is DataQuality.DEGRADED
    assert not series.complete


def test_to_candle_series_refuses_an_empty_series() -> None:
    with pytest.raises(BackfillError):
        to_candle_series([], timeframe=Timeframe.H1, pool=POOL)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=payload([row(T0 + HOUR), row(T0)]))


def test_backfill_writes_every_pair_and_a_manifest(tmp_path: Path) -> None:
    coins = [
        {"symbol": "WIF", "mint": MINT, "pool": POOL},
        {"symbol": "BONK", "mint": "Dez", "pool": "PoolB"},
    ]

    with client_for(_ok_handler) as client:
        report = backfill(
            client,
            coins=coins,
            timeframes=[Timeframe.H1, Timeframe.M5],
            since=T0,
            root=tmp_path,
            base_url=BASE,
            pace_seconds=0.0,
            now=lambda: T0 + 100 * HOUR,
            sleep=lambda _: None,
        )

    assert len(report.written) == 4
    assert report.failed == []
    assert series_path(tmp_path, POOL, Timeframe.H1).exists()
    assert series_path(tmp_path, "PoolB", Timeframe.M5).exists()
    assert len(read_manifest(tmp_path)) == 4


def test_backfill_resume_skips_pairs_already_covered(tmp_path: Path) -> None:
    """An interrupted hour-long pull must be cheap to restart."""
    coins = [{"symbol": "WIF", "mint": MINT, "pool": POOL}]

    with client_for(_ok_handler) as client:
        backfill(
            client,
            coins=coins,
            timeframes=[Timeframe.H1],
            since=T0,
            root=tmp_path,
            base_url=BASE,
            pace_seconds=0.0,
            now=lambda: T0 + 100 * HOUR,
            sleep=lambda _: None,
        )

        calls = 0

        def counting(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return _ok_handler(request)

    with client_for(counting) as client:
        report = backfill(
            client,
            coins=coins,
            timeframes=[Timeframe.H1],
            since=T0,
            root=tmp_path,
            base_url=BASE,
            resume=True,
            pace_seconds=0.0,
            now=lambda: T0 + 100 * HOUR,
            sleep=lambda _: None,
        )

    assert calls == 0
    assert report.skipped == ["WIF 1h"]
    assert report.written == []


def test_backfill_records_a_failure_without_aborting_the_run(tmp_path: Path) -> None:
    """A partial dataset with an honest manifest beats no dataset."""
    coins = [
        {"symbol": "BAD", "mint": "m", "pool": "PoolBad"},
        {"symbol": "WIF", "mint": MINT, "pool": POOL},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if "PoolBad" in str(request.url):
            return httpx.Response(422)
        return _ok_handler(request)

    with client_for(handler) as client:
        report = backfill(
            client,
            coins=coins,
            timeframes=[Timeframe.H1],
            since=T0,
            root=tmp_path,
            base_url=BASE,
            pace_seconds=0.0,
            now=lambda: T0 + 100 * HOUR,
            sleep=lambda _: None,
        )

    assert [name for name, _ in report.failed] == ["BAD 1h"]
    assert [m.symbol for m in report.written] == ["WIF"]
    assert len(read_manifest(tmp_path)) == 1
