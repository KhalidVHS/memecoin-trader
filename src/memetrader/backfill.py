"""Historical OHLCV acquisition for backtesting.

This module downloads and stores closed bars. It is deliberately **not** a
change to ``market.py``: that module's parser is built for a live tick and
actively fights a backfill. ``market._audit_spacing`` raises on the duplicate
bar that every page boundary produces, ``market.py`` rejects any bar newer than
the injected ``now``, and a single malformed row aborts the whole series — which
is correct for a tick that can be retried in sixty seconds, and wrong for an
hour-long walk backwards through a year of history. The *validation ideas* are
reused here; the call path is not.

Four decisions worth stating, because each one is a trap avoided:

**Series are keyed by pool address, never by symbol.** ``CandleSeries`` is bound
to ``pool_address`` for the reason given at ``types.py:338-345`` — a pool
migration creates a synthetic price regime, and concatenating two pools under
one ticker produces a jump that every momentum and volatility feature reads as a
real move. A file named ``BONK/1h.jsonl.gz`` invites exactly that splice. A file
named after the pool cannot be spliced by accident.

**The open bar is never written.** Live code keeps it, watermarked
``closed=False``, because current volume is legitimate state. History has no
such need: an in-progress bar is a partial observation that will be revised, and
the only honest thing to do with it in a stored dataset is to not store it.
Every row on disk is closed, so no ``closed`` column exists and no reader can
forget to filter on it.

**Missing bars are recorded, never filled.** GeckoTerminal omits intervals in
which nothing traded rather than sending a zero-volume bar — confirmed, zero
zero-volume bars across 3,000 sampled while POPCAT was missing 9.3% of its
intervals outright. Forward-filling would invent trades; zero-filling would
claim a price of zero. Both violate the project's rule that ``None`` means
"could not find out" and ``0`` means "looked, and it is quiet". A gap is
*knowable* information — it means no trades printed — so it is stored as an
explicit :class:`Gap` and surfaced through :func:`window_gaps`.

**A gapped window degrades the series rather than adding new logic.**
:func:`to_candle_series` marks a window containing gaps ``DataQuality.DEGRADED``,
which the existing ``BaselineStrategy._forecast`` gate already refuses to act
on. This matters more than it looks: ``realized_vol_pct`` is a stdev of log
returns, and over a gapped series those returns span unequal times, overstating
volatility exactly for the sparse coins that can least afford it. The backtest
does not need to learn a new rule — the rule it already has now fires.
"""

from __future__ import annotations

import gzip
import itertools
import json
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx

from .http import RequestFailed, RetryPolicy, execute
from .journal import atomic_write_text, to_jsonable
from .types import Candle, CandleSeries, DataQuality, Provenance, Timeframe

__all__ = [
    "BACKFILL_RETRY",
    "BackfillError",
    "Gap",
    "RawBar",
    "RejectedRow",
    "SeriesMeta",
    "backfill",
    "fetch_series",
    "read_manifest",
    "read_series",
    "series_path",
    "to_candle_series",
    "window_gaps",
    "write_manifest",
    "write_series",
]

TOOL_VERSION = "backfill/1"

# Mirrors ``market._ROUTES``: route segment, aggregate, and the bar length it
# implies. The interval is what makes gap counting possible at all — without it
# "the next row is 7200s later" is indistinguishable from "one bar is missing".
_ROUTES: dict[Timeframe, tuple[str, int, float]] = {
    Timeframe.M5: ("minute", 5, 300.0),
    Timeframe.H1: ("hour", 1, 3600.0),
}

# Verified against the live free tier: 1000 is accepted and is the documented
# ceiling. The live path only ever sends 100, so a backfill page is ten times
# cheaper per bar than the tick path would suggest.
MAX_LIMIT = 1000

# Verified empirically: 429s begin around 2.1s spacing on the keyless tier, so
# the default sits above that *and* every call still goes through
# ``http.execute``, which treats 429 as retryable and honours ``Retry-After``.
# ``market.py`` forgoes that retry wrapper; this module deliberately does not.
DEFAULT_PACE_SECONDS = 3.0

# ``DEFAULT_RETRY`` is tuned for a 60-second tick: 3 attempts inside a 20-second
# budget, backoff capped at 4s. Measured against this vendor that is far too
# impatient — a first smoke run lost 18 of 24 series to 429s, with the attempts
# exhausted in under a second. A backfill has no tick deadline, so it trades
# latency for completeness: more attempts, a budget measured in minutes, and a
# cap long enough to sit out a rolling-window limit rather than hammer it.
# Sleeping 45s once is cheaper than failing a coin and re-running the hour.
BACKFILL_RETRY = RetryPolicy(
    max_attempts=8,
    total_budget_seconds=240.0,
    backoff_base_seconds=2.0,
    backoff_multiplier=2.0,
    backoff_max_seconds=45.0,
    max_retry_after_seconds=90.0,
)

# Sanity window for normalized timestamps, 2020-01-01 .. 2100-01-01. Catches a
# vendor unit slip, not bad data.
_TS_FLOOR = 1_577_836_800.0
_TS_CEILING = 4_102_444_800.0

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_HEADERS = {"User-Agent": _BROWSER_UA, "Accept": "application/json"}

# A page whose rejected fraction exceeds this is not "a few bad rows", it is a
# vendor malfunction, and the series is quarantined rather than trimmed.
_QUARANTINE_REJECT_FRACTION = 0.05


class BackfillError(RuntimeError):
    """A pull could not be completed or produced something unusable."""


class HistoryHorizon(BackfillError):
    """The vendor refused to look further back. A horizon, not a failure.

    Measured 2026-09-21, and it is the single most important constraint on this
    whole dataset: the keyless tier answers **401** with *"You can only access
    data from the past 180 days with Public API"* as soon as
    ``before_timestamp`` passes that mark. It applies to 1h and 5m alike, so
    "pull 1h back as far as the free tier allows" and "pull 5m for six months"
    are the same window — there is no long history to be had without a key.

    The depth is slightly better than 180 days because the cap is on the
    *requested* ``before_timestamp``, not on the rows returned: a request at the
    180-day mark still yields the 1000 bars ending there. That reaches roughly
    2026-02-11 for 1h and 2026-03-21 for 5m.

    This is a subclass of :class:`BackfillError` so nothing can catch it by
    accident, but ``fetch_series`` handles it explicitly and keeps every bar
    already collected. Treating it as a plain error is what made a first run
    write zero files while holding thousands of perfectly good bars in memory.
    """


@dataclass(frozen=True, slots=True)
class RawBar:
    """One closed OHLCV bar as stored. ``ts`` is the bar's open, epoch seconds.

    Deliberately not ``types.Candle``: there is no ``closed`` field because
    nothing unclosed is ever stored, and the OHLC invariants are checked at
    parse time so a stored row is known-good without re-validating 1.7M rows on
    every read. :func:`to_candle_series` converts to ``Candle`` at the boundary,
    which is where the strict validation belongs.
    """

    ts: float
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class Gap:
    """A run of intervals the vendor did not report.

    ``first_missing_ts``/``last_missing_ts`` are the opens of the first and last
    absent bars, so a one-bar gap has them equal. Stored rather than derived
    because the point is to make a hole legible to a human reading the manifest.
    """

    first_missing_ts: float
    last_missing_ts: float
    bars: int


@dataclass(frozen=True, slots=True)
class RejectedRow:
    """A row that could not become a bar, kept as evidence.

    Counting rejects without keeping them makes a vendor malfunction
    indistinguishable from a quiet market. The raw payload is truncated because
    the manifest is meant to be read, not to be a second copy of the feed.
    """

    reason: str
    raw: str


@dataclass(frozen=True, slots=True)
class SeriesMeta:
    """Provenance for one stored series. A dataset without this is not evidence."""

    symbol: str
    mint: str
    pool: str
    timeframe: str
    interval_seconds: float
    rows: int
    first_ts: float | None
    last_ts: float | None
    gaps: tuple[Gap, ...] = ()
    missing_bars: int = 0
    rejected: tuple[RejectedRow, ...] = ()
    pages: int = 0
    fetched_at: float = 0.0
    source: str = "geckoterminal"
    base_url: str = ""
    tool_version: str = TOOL_VERSION
    quality: str = DataQuality.OK.value
    quality_reason: str | None = None
    horizon_reached: bool = False

    @property
    def complete(self) -> bool:
        return self.missing_bars == 0 and not self.rejected


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _opt_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except TypeError, ValueError:
        return None
    # NaN and infinity are not prices. They survive float() and then poison
    # every mean and stdev computed downstream without ever raising.
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def _parse_page(rows: Any, *, now: float) -> tuple[list[RawBar], list[RejectedRow]]:
    """Vendor rows -> bars, oldest-first, with unusable rows kept as evidence.

    Where ``market._parse_candle_rows`` *raises* on a malformed row, this
    rejects and records it. That difference is intentional and is not a
    loosening of standards: aborting an hour-long walk because one bar in
    400,000 has ``low > high`` loses far more information than it protects, and
    the reject is still fatal at the series level via
    ``_QUARANTINE_REJECT_FRACTION``. The evidence is preserved either way — the
    live path preserves it by refusing to trade, this path by writing it into
    the manifest.
    """
    bars: list[RawBar] = []
    rejected: list[RejectedRow] = []

    def reject(reason: str, row: Any) -> None:
        rejected.append(RejectedRow(reason=reason, raw=repr(row)[:200]))

    for row in rows or []:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            reject("row is not a 6-element sequence", row)
            continue
        ts = _opt_float(row[0])
        if ts is None:
            reject("ts is null or non-numeric", row)
            continue
        # GeckoTerminal serves seconds. This only catches a future unit change,
        # so a silent switch to milliseconds fails here instead of shifting
        # every bar 56,000 years into the future.
        if ts > _TS_CEILING:
            ts /= 1000.0
        if not _TS_FLOOR <= ts <= _TS_CEILING:
            reject(f"ts {ts} outside 2020..2100", row)
            continue
        if ts > now:
            # A bar that opens in the future cannot be an observation. Unlike
            # the live path there is no clock-tolerance slack, because a
            # backfill asks only for bars that are already long past.
            reject(f"bar at {ts} is future-dated against {now}", row)
            continue
        values: list[float] = []
        bad = False
        for index, name in ((1, "open"), (2, "high"), (3, "low"), (4, "close")):
            parsed = _opt_float(row[index])
            if parsed is None:
                reject(f"{name} is null or non-numeric", row)
                bad = True
                break
            if parsed <= 0.0:
                reject(f"{name} is not positive ({parsed})", row)
                bad = True
                break
            values.append(parsed)
        if bad:
            continue
        volume = _opt_float(row[5])
        if volume is None or volume < 0.0:
            reject("volume is null, non-numeric or negative", row)
            continue
        open_, high, low, close = values
        # The same invariants ``Candle.__post_init__`` enforces. A vendor
        # reporting low > high silently produces a negative range in every
        # range-based indicator downstream, which understates volatility exactly
        # when it matters most.
        if low > high or not (low <= open_ <= high) or not (low <= close <= high):
            reject(f"OHLC invariant violated (o={open_} h={high} l={low} c={close})", row)
            continue
        bars.append(
            RawBar(ts=ts, open=open_, high=high, low=low, close=close, volume=volume)
        )

    # The vendor serves newest-first and every indicator in signals.py walks
    # forward in time. This codebase has hit the un-reversed-series bug once
    # already; see the note at market.py:675.
    bars.sort(key=lambda b: b.ts)
    return bars, rejected


def _merge(pages: Iterable[Sequence[RawBar]]) -> list[RawBar]:
    """Concatenate pages, dedup by timestamp, return oldest-first.

    Page boundaries share **exactly one** duplicate bar (verified against the
    live free tier), so dedup is mandatory rather than defensive. Where two
    copies of a timestamp disagree the first seen wins; they have never been
    observed to disagree, and picking arbitrarily is still better than the
    alternative, which is ``market._audit_spacing`` raising on every boundary.
    """
    seen: dict[float, RawBar] = {}
    for page in pages:
        for bar in page:
            seen.setdefault(bar.ts, bar)
    return [seen[ts] for ts in sorted(seen)]


def _find_gaps(bars: Sequence[RawBar], interval: float) -> tuple[list[Gap], int]:
    """Locate absent intervals. Raises if the bars are not on the expected grid.

    A duplicate timestamp cannot reach here (``_merge`` removes them) so it is
    an assertion failure rather than a vendor problem. Off-grid spacing *is* a
    vendor problem and is fatal: it means the bars are not the bars we think
    they are, which invalidates every window length in ``signals.py``.
    """
    gaps: list[Gap] = []
    missing = 0
    for previous, current in itertools.pairwise(bars):
        delta = current.ts - previous.ts
        if delta <= 0.0:
            raise BackfillError(f"non-increasing timestamps at {current.ts}")
        steps = delta / interval
        if abs(steps - round(steps)) > 1e-6:
            raise BackfillError(
                f"bar spacing {delta}s is not a multiple of the {interval}s "
                "interval — the bars are not on the expected grid"
            )
        absent = round(steps) - 1
        if absent > 0:
            gaps.append(
                Gap(
                    first_missing_ts=previous.ts + interval,
                    last_missing_ts=current.ts - interval,
                    bars=absent,
                )
            )
            missing += absent
    return gaps, missing


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _fetch_page(
    client: httpx.Client,
    *,
    base_url: str,
    pool: str,
    timeframe: Timeframe,
    limit: int = MAX_LIMIT,
    before_timestamp: float | None = None,
    retry: RetryPolicy | None = None,
) -> Any:
    """One OHLCV page. Returns the raw ``ohlcv_list``.

    Routed through ``http.execute`` rather than ``client.get``: ``make_client``
    supplies TLS and timeouts but no retry, backoff or breaker
    (``http.py:767``), and a backfill that gives up on the first 429 halfway
    through a year of history is not worth running. ``execute`` already treats
    429 as retryable and honours ``Retry-After``.
    """
    route, aggregate, _ = _ROUTES[timeframe]
    url = f"{base_url.rstrip('/')}/networks/solana/pools/{pool}/ohlcv/{route}"
    params: dict[str, Any] = {"aggregate": aggregate, "limit": limit}
    if before_timestamp is not None:
        # Epoch *seconds*, and exclusive of the bar at that timestamp in
        # practice — which is why the caller pages from the oldest bar it
        # already holds rather than one interval below it.
        params["before_timestamp"] = int(before_timestamp)

    outcome = execute(
        client,
        "GET",
        url,
        params=params,
        headers=_HEADERS,
        retry=retry or BACKFILL_RETRY,
        idempotent=True,
    )
    response = outcome.response
    if response is None:
        raise BackfillError(
            f"{pool} {timeframe.value}: no response after "
            f"{outcome.attempt_count} attempts ({outcome.stopped_by})"
        ) from outcome.error
    if response.status_code == 422:
        # Recorded finding (http.py:246-269): 422 from this vendor means
        # "narrow the range", never "you are going too fast". Treating it as
        # rate limiting produces an infinite backoff against a request that
        # will never succeed.
        raise BackfillError(f"{pool} {timeframe.value}: 422, range not serviceable")
    if response.status_code == 401:
        # Not "you are unauthenticated" — the keyless tier is the intended tier
        # here. It means ``before_timestamp`` went past the 180-day public
        # window. See HistoryHorizon: the caller keeps what it already has.
        raise HistoryHorizon(
            f"{pool} {timeframe.value}: public API history horizon reached"
        )
    if response.status_code >= 400:
        raise BackfillError(
            f"{pool} {timeframe.value}: HTTP {response.status_code} from {route} ohlcv"
        )
    payload = response.json() or {}
    attributes = (payload.get("data") or {}).get("attributes") or {}
    return attributes.get("ohlcv_list") or []


def fetch_series(
    client: httpx.Client,
    *,
    symbol: str,
    mint: str,
    pool: str,
    timeframe: Timeframe,
    since: float,
    base_url: str,
    limit: int = MAX_LIMIT,
    pace_seconds: float = DEFAULT_PACE_SECONDS,
    max_pages: int = 400,
    retry: RetryPolicy | None = None,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[RawBar], SeriesMeta]:
    """Walk backwards from now to ``since``, returning closed bars oldest-first.

    Stops on the first of: reaching ``since``, an empty page, a page that adds
    no new oldest bar (the vendor has no more history), or ``max_pages``. The
    last of those is a guard against an infinite walk, not an expected exit, and
    it is recorded in the metadata when it fires.
    """
    if timeframe not in _ROUTES:
        raise BackfillError(f"unsupported timeframe {timeframe!r}")
    _, _, interval = _ROUTES[timeframe]

    started = now()
    pages: list[list[RawBar]] = []
    rejected: list[RejectedRow] = []
    before: float | None = None
    oldest: float | None = None
    page_count = 0
    truncated = False
    horizon = False

    for index in range(max_pages):
        if index and pace_seconds > 0:
            sleep(pace_seconds)
        try:
            rows = _fetch_page(
                client,
                base_url=base_url,
                pool=pool,
                timeframe=timeframe,
                limit=limit,
                before_timestamp=before,
                retry=retry,
            )
        except HistoryHorizon:
            # The vendor will not look further back. Everything already paged
            # is still good data; stop walking and say so in the metadata.
            horizon = True
            break
        page_count += 1
        bars, page_rejects = _parse_page(rows, now=started)
        rejected.extend(page_rejects)
        if not bars:
            break
        pages.append(bars)
        page_oldest = bars[0].ts
        # No progress: the vendor is returning the same window, so there is no
        # more history behind it. Without this check a pool younger than
        # ``since`` would page forever.
        if oldest is not None and page_oldest >= oldest:
            break
        oldest = page_oldest
        if page_oldest <= since:
            break
        before = page_oldest
    else:
        truncated = True

    merged = _merge(pages)
    # Trim to the requested window. Vendors overshoot on the last page and a
    # dataset whose start depends on where a page boundary happened to land is
    # not reproducible.
    merged = [b for b in merged if b.ts >= since]

    # Drop the newest bar if it is still forming. The live path keeps it
    # watermarked ``closed=False``; history simply does not store it, so every
    # row on disk is closed and no reader can forget to filter.
    while merged and (merged[-1].ts + interval) > started:
        merged.pop()

    gaps, missing = _find_gaps(merged, interval)

    quality = DataQuality.OK
    reasons: list[str] = []
    if missing:
        quality = DataQuality.DEGRADED
        reasons.append(f"{missing} missing bars in {len(gaps)} gaps")
    if rejected:
        quality = DataQuality.DEGRADED
        reasons.append(f"{len(rejected)} rejected rows")
    total = len(merged) + len(rejected)
    if total and len(rejected) / total > _QUARANTINE_REJECT_FRACTION:
        # Past this point it is not a few bad rows, it is a malfunctioning
        # feed, and no feature may be computed from it.
        quality = DataQuality.QUARANTINED
        reasons.append("reject fraction above quarantine threshold")
    if truncated:
        reasons.append(f"stopped at max_pages={max_pages}")
    if horizon:
        # Deliberately not a quality downgrade. The bars that are here are
        # sound; the series is simply shorter than asked for, and a reader
        # comparing first_ts against --since needs to know which of the two
        # reasons applies: the pool is young, or the vendor said no.
        reasons.append("public API 180-day history horizon")

    meta = SeriesMeta(
        symbol=symbol,
        mint=mint,
        pool=pool,
        timeframe=timeframe.value,
        interval_seconds=interval,
        rows=len(merged),
        first_ts=merged[0].ts if merged else None,
        last_ts=merged[-1].ts if merged else None,
        gaps=tuple(gaps),
        missing_bars=missing,
        rejected=tuple(rejected[:50]),
        pages=page_count,
        fetched_at=started,
        base_url=base_url,
        quality=quality.value,
        quality_reason="; ".join(reasons) or None,
        horizon_reached=horizon,
    )
    return merged, meta


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def series_path(root: Path, pool: str, timeframe: Timeframe | str) -> Path:
    """``<root>/<pool>/<timeframe>.jsonl.gz``.

    Keyed by pool, not symbol — see the module docstring. A ticker can point at
    two pools across a migration; a pool address cannot.
    """
    name = timeframe.value if isinstance(timeframe, Timeframe) else timeframe
    return root / pool / f"{name}.jsonl.gz"


def write_series(path: Path, bars: Sequence[RawBar]) -> None:
    """Write bars oldest-first as gzipped JSONL.

    Gzipped JSONL rather than Parquet because no Parquet engine is installed
    (neither pyarrow nor fastparquet), it matches the repo's existing
    ``journal.py`` idiom, it streams without holding 1.7M rows in memory, and
    ``pandas.read_json(lines=True)`` reads it directly.

    ``mtime=0`` makes the gzip header deterministic, so an unchanged pull
    produces a byte-identical file and a re-run is visibly a no-op.
    """
    if any(b.ts > n.ts for b, n in itertools.pairwise(bars)):
        raise BackfillError("refusing to write bars that are not oldest-first")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.GzipFile(filename="", mode="wb", fileobj=tmp.open("wb"), mtime=0) as raw:
        for bar in bars:
            row = {
                "ts": bar.ts,
                "o": bar.open,
                "h": bar.high,
                "l": bar.low,
                "c": bar.close,
                "v": bar.volume,
            }
            raw.write((json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8"))
    tmp.replace(path)


def read_series(path: Path) -> Iterator[RawBar]:
    """Stream bars back. Lazy so a 400k-row 5m series need not be materialised."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BackfillError(f"{path}:{number}: malformed JSON") from exc
            yield RawBar(
                ts=float(row["ts"]),
                open=float(row["o"]),
                high=float(row["h"]),
                low=float(row["l"]),
                close=float(row["c"]),
                volume=float(row["v"]),
            )


def write_manifest(root: Path, entries: Sequence[SeriesMeta]) -> None:
    """Replace the manifest atomically.

    Reuses ``journal.to_jsonable``/``atomic_write_text`` so NaN becomes null
    rather than the bare ``NaN`` token that strict JSON readers reject, and so a
    reader never sees a half-written file — including on Windows, where
    ``os.replace`` across filesystems raises outright.
    """
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "tool_version": TOOL_VERSION,
        "written_at": time.time(),
        "series": [to_jsonable(entry) for entry in entries],
    }
    atomic_write_text(root / "manifest.json", json.dumps(payload, indent=1) + "\n")


def read_manifest(root: Path) -> list[SeriesMeta]:
    path = root / "manifest.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: list[SeriesMeta] = []
    for raw in payload.get("series") or []:
        gaps = tuple(
            Gap(
                first_missing_ts=float(g["first_missing_ts"]),
                last_missing_ts=float(g["last_missing_ts"]),
                bars=int(g["bars"]),
            )
            for g in raw.get("gaps") or []
        )
        rejected = tuple(
            RejectedRow(reason=str(r["reason"]), raw=str(r["raw"]))
            for r in raw.get("rejected") or []
        )
        known = {
            f for f in SeriesMeta.__dataclass_fields__ if f not in ("gaps", "rejected")
        }
        kwargs = {k: v for k, v in raw.items() if k in known}
        out.append(SeriesMeta(gaps=gaps, rejected=rejected, **kwargs))
    return out


# ---------------------------------------------------------------------------
# Consumption
# ---------------------------------------------------------------------------


def window_gaps(meta: SeriesMeta, start_ts: float, end_ts: float) -> int:
    """How many bars are absent in ``[start_ts, end_ts]``.

    The whole point of storing gaps rather than filling them. A backtest asks
    this before computing a feature over a window, because ``realized_vol_pct``
    is a stdev of log returns and a gapped window silently computes returns
    across unequal spans — overstating volatility for exactly the sparse coins
    that can least afford it.
    """
    total = 0
    for gap in meta.gaps:
        first = max(gap.first_missing_ts, start_ts)
        last = min(gap.last_missing_ts, end_ts)
        if first > last:
            continue
        interval = meta.interval_seconds
        total += round((last - first) / interval) + 1
    return total


def to_candle_series(
    bars: Sequence[RawBar],
    *,
    timeframe: Timeframe,
    pool: str,
    meta: SeriesMeta | None = None,
    receive_time: float | None = None,
) -> CandleSeries:
    """Stored bars -> a validated ``CandleSeries`` the live code already accepts.

    Every candle is ``closed=True`` because nothing unclosed is ever stored.
    ``Candle.__post_init__`` re-validates at this boundary, which is the point:
    the strict check lives where data enters the decision path, not on every
    read of a 400k-row file.

    A window containing gaps is marked ``DataQuality.DEGRADED``, which
    ``BaselineStrategy._forecast`` already refuses to act on. No new gate is
    introduced; the existing one simply starts firing on historical data.
    """
    if not bars:
        raise BackfillError("cannot build a CandleSeries from zero bars")
    _, _, interval = _ROUTES[timeframe]
    candles = tuple(
        Candle(
            ts=b.ts,
            open=b.open,
            high=b.high,
            low=b.low,
            close=b.close,
            volume=b.volume,
            closed=True,
        )
        for b in bars
    )
    _, missing = _find_gaps(bars, interval)

    quality = DataQuality.OK
    reason: str | None = None
    if meta is not None and meta.quality == DataQuality.QUARANTINED.value:
        quality, reason = DataQuality.QUARANTINED, meta.quality_reason
    elif missing:
        quality = DataQuality.DEGRADED
        reason = f"{missing} missing intervals in window"

    return CandleSeries(
        timeframe=timeframe,
        pool_address=pool,
        candles=candles,
        interval_seconds=interval,
        provenance=Provenance(
            source="geckoterminal-backfill",
            # The read is offline, so ``receive_time`` defaults to when the data
            # was actually fetched rather than to now — a replay must not be
            # able to make year-old bars look freshly observed.
            receive_time=(
                receive_time
                if receive_time is not None
                else (meta.fetched_at if meta else candles[-1].ts)
            ),
            # Deliberately the newest bar's *open*, so a series can never look
            # fresher than it is.
            event_time=candles[-1].ts,
            source_time=None,
            quality=quality,
            quality_reason=reason,
        ),
        missing_intervals=missing,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Target:
    symbol: str
    mint: str
    pool: str
    timeframe: Timeframe


@dataclass(slots=True)
class BackfillReport:
    """What a run did. Returned rather than printed so the CLI owns rendering."""

    written: list[SeriesMeta] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)


def load_universe(path: Path) -> list[dict[str, Any]]:
    """Read the committed universe file. Kept here so the CLI stays thin."""
    import tomllib

    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    coins = payload.get("coins") or []
    if not coins:
        raise BackfillError(f"{path}: no [[coins]] entries")
    return list(coins)


def backfill(
    client: httpx.Client,
    *,
    coins: Sequence[dict[str, Any]],
    timeframes: Sequence[Timeframe],
    since: float,
    root: Path,
    base_url: str,
    resume: bool = False,
    pace_seconds: float = DEFAULT_PACE_SECONDS,
    limit: int = MAX_LIMIT,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[str], None] | None = None,
) -> BackfillReport:
    """Pull every (coin, timeframe) pair and write it, updating the manifest.

    ``resume`` skips pairs already covering ``since`` in the manifest, so an
    interrupted hour-long pull is cheap to restart. One coin failing does not
    abort the run — a partial dataset with an honest manifest is worth more than
    no dataset, and the failures are reported rather than swallowed.

    The manifest is rewritten after **every** series rather than once at the
    end, so a run killed at minute 50 leaves a manifest describing exactly the
    files that exist.
    """
    existing = {(m.pool, m.timeframe): m for m in read_manifest(root)}
    report = BackfillReport()

    targets = [
        _Target(
            symbol=str(coin["symbol"]),
            mint=str(coin["mint"]),
            pool=str(coin["pool"]),
            timeframe=tf,
        )
        for coin in coins
        for tf in timeframes
    ]

    for target in targets:
        key = (target.pool, target.timeframe.value)
        prior = existing.get(key)
        if (
            resume
            and prior is not None
            and prior.first_ts is not None
            and prior.first_ts <= since + prior.interval_seconds
            and series_path(root, target.pool, target.timeframe).exists()
        ):
            report.skipped.append(f"{target.symbol} {target.timeframe.value}")
            continue

        if on_progress:
            on_progress(f"{target.symbol} {target.timeframe.value}")
        try:
            bars, meta = fetch_series(
                client,
                symbol=target.symbol,
                mint=target.mint,
                pool=target.pool,
                timeframe=target.timeframe,
                since=since,
                base_url=base_url,
                limit=limit,
                pace_seconds=pace_seconds,
                now=now,
                sleep=sleep,
            )
        except (BackfillError, RequestFailed, httpx.HTTPError, ValueError, OSError) as exc:
            report.failed.append((f"{target.symbol} {target.timeframe.value}", str(exc)))
            continue

        if not bars:
            report.failed.append(
                (f"{target.symbol} {target.timeframe.value}", "no bars in window")
            )
            continue

        write_series(series_path(root, target.pool, target.timeframe), bars)
        existing[key] = meta
        report.written.append(meta)
        write_manifest(root, list(existing.values()))

    return report


def refresh_meta(meta: SeriesMeta, **changes: Any) -> SeriesMeta:
    """Copy a metadata record with fields replaced. Thin, but keeps callers pure."""
    return replace(meta, **changes)
