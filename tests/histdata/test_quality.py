"""Tests for quality.py — data-quality reports and usability verdicts.

Tests verify that:
  - Missing-rate calculation is correct
  - vol_biased is set for high missing rates
  - usable_for returns False for coins with too many missing bars
  - The measured facts from CANNOT-REPLAY.md are reflected in verdicts
  - Missing rates over a sub-window are calculated correctly
"""

from __future__ import annotations

import time

from memetrader.backfill import Gap, SeriesMeta
from memetrader.histdata.quality import (
    MEASURED_1H_MISSING,
    MEASURED_5M_MISSING,
    build_report,
    missing_rate_pct_for_window,
    usable_for,
)
from memetrader.types import DataQuality, FidelityTier

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _meta(
    *,
    symbol: str = "TEST",
    mint: str = "MINT_TEST",
    pool: str = "POOL_TEST",
    timeframe: str = "1h",
    rows: int = 1000,
    missing_bars: int = 0,
    gaps: tuple[Gap, ...] = (),
    first_ts: float = 1_700_000_000.0,
    last_ts: float | None = None,
    interval_seconds: float = 3600.0,
    horizon_reached: bool = True,
    quality: str = DataQuality.OK.value,
) -> SeriesMeta:
    if last_ts is None:
        last_ts = first_ts + (rows + missing_bars - 1) * interval_seconds
    return SeriesMeta(
        symbol=symbol,
        mint=mint,
        pool=pool,
        timeframe=timeframe,
        interval_seconds=interval_seconds,
        rows=rows,
        first_ts=first_ts,
        last_ts=last_ts,
        gaps=gaps,
        missing_bars=missing_bars,
        fetched_at=time.time(),
        source="test",
        horizon_reached=horizon_reached,
        quality=quality,
    )


# ---------------------------------------------------------------------------
# Test: missing rate calculation
# ---------------------------------------------------------------------------


def test_missing_rate_zero() -> None:
    """A complete series has 0% missing rate."""
    meta = _meta(rows=4995, missing_bars=0)
    report = build_report(meta)
    assert report.missing_rate_pct == 0.0


def test_missing_rate_one_percent() -> None:
    """49 missing in 4991+49=5040 expected bars ≈ 0.97%."""
    meta = _meta(rows=4991, missing_bars=49)
    report = build_report(meta)
    # missing / (present + missing) * 100
    expected = 100.0 * 49 / (4991 + 49)
    assert abs(report.missing_rate_pct - expected) < 0.01


def test_slerf_high_missing_rate() -> None:
    """SLERF-like 5m series: 86.5% missing makes vol_biased=True, usable=False."""
    # SLERF 5m: 7276 rows, ~86.5% missing
    total_expected = round(7276 / (1 - 0.865))
    missing = total_expected - 7276
    meta = _meta(
        timeframe="5m",
        rows=7276,
        missing_bars=missing,
        interval_seconds=300.0,
        last_ts=1_700_000_000.0 + (total_expected - 1) * 300,
    )
    report = build_report(meta, symbol="SLERF")
    assert report.vol_biased
    assert not report.usable


def test_bonk_low_missing_rate() -> None:
    """BONK 1h: 0% missing rate, usable=True."""
    meta = _meta(
        timeframe="1h",
        rows=4995,
        missing_bars=0,
        symbol="BONK",
        last_ts=1_700_000_000.0 + 4994 * 3600,
    )
    report = build_report(meta, symbol="BONK")
    assert report.missing_rate_pct == 0.0
    assert not report.vol_biased
    assert report.usable


# ---------------------------------------------------------------------------
# Test: coverage
# ---------------------------------------------------------------------------


def test_coverage_days_calculated() -> None:
    """coverage_days = (last_ts - first_ts) / 86400."""
    days = 90.0
    first = 1_700_000_000.0
    last = first + days * 86400.0
    meta = _meta(rows=100, first_ts=first, last_ts=last)
    report = build_report(meta)
    assert abs(report.coverage_days - days) < 0.001


def test_short_coverage_flagged() -> None:
    """A series shorter than 60 days has coverage_short=True and usable=False."""
    first = 1_700_000_000.0
    last = first + 30.0 * 86400.0  # only 30 days
    meta = _meta(rows=100, first_ts=first, last_ts=last)
    report = build_report(meta)
    assert report.coverage_short
    assert not report.usable


def test_act_short_5m_coverage() -> None:
    """ACT 5m: 51 days of coverage, should be flagged as short."""
    first = 1_700_000_000.0
    last = first + 51.0 * 86400.0
    meta = _meta(
        timeframe="5m",
        rows=10000,
        missing_bars=0,
        first_ts=first,
        last_ts=last,
        interval_seconds=300.0,
    )
    report = build_report(meta, symbol="ACT")
    assert report.coverage_short
    assert not report.usable


# ---------------------------------------------------------------------------
# Test: usable_for by tier
# ---------------------------------------------------------------------------


def test_tier0_usable_good_series() -> None:
    """A clean 1h series with 200+ days coverage is usable for TIER_0."""
    first = 1_700_000_000.0
    last = first + 210.0 * 86400.0
    meta = _meta(timeframe="1h", rows=5000, missing_bars=10, first_ts=first, last_ts=last)
    report = build_report(meta)
    assert usable_for(report, FidelityTier.TIER_0)


def test_tier0_unusable_high_missing_1h() -> None:
    """A 1h series with 36% missing is not usable for TIER_0."""
    first = 1_700_000_000.0
    # 36% missing: 100 rows in 100+56=156 expected
    rows = 100
    missing = 57  # 57 / (100+57) ≈ 36.3%
    last = first + (rows + missing - 1) * 3600.0
    meta = _meta(
        timeframe="1h", rows=rows, missing_bars=missing, first_ts=first, last_ts=last
    )
    report = build_report(meta)
    assert not usable_for(report, FidelityTier.TIER_0)


def test_tier1_relaxed_threshold() -> None:
    """TIER_1 allows up to 50% missing.

    Uses enough rows that coverage stays well above ``_MIN_COVERAGE_DAYS``
    (60 days) — a 100-row/1h series only spans ~7.5 days and would be
    rejected by the coverage_short guard before the missing-rate threshold
    is ever exercised, regardless of tier. That guard is intentional (see
    test_coverage_short_blocks_all_tiers) and must stay in force; the fix
    here is to give this test's fixture a realistic coverage window instead
    of loosening the guard.
    """
    first = 1_700_000_000.0
    # 45% missing: under TIER_1 threshold (50%) but over TIER_0 threshold (35% for 1h)
    rows = 1000
    missing = 818  # 818 / 1818 ≈ 45%
    last = first + (rows + missing - 1) * 3600.0
    meta = _meta(
        timeframe="1h", rows=rows, missing_bars=missing, first_ts=first, last_ts=last
    )
    report = build_report(meta)
    assert not report.coverage_short
    assert usable_for(report, FidelityTier.TIER_1)


def test_coverage_short_blocks_all_tiers() -> None:
    """Short coverage blocks usability regardless of tier."""
    first = 1_700_000_000.0
    last = first + 30.0 * 86400.0
    meta = _meta(rows=1000, missing_bars=0, first_ts=first, last_ts=last)
    report = build_report(meta)
    for tier in FidelityTier:
        assert not usable_for(report, tier)


# ---------------------------------------------------------------------------
# Test: gap histogram
# ---------------------------------------------------------------------------


def test_gap_histogram_populated() -> None:
    """Gap histogram buckets are populated correctly."""
    gaps = (
        Gap(first_missing_ts=1.0, last_missing_ts=1.0, bars=1),  # bucket [1,1]
        Gap(first_missing_ts=2.0, last_missing_ts=5.0, bars=3),  # bucket [2,5]
        Gap(first_missing_ts=6.0, last_missing_ts=20.0, bars=10),  # bucket [6,20]
    )
    meta = _meta(rows=100, missing_bars=14, gaps=gaps)
    report = build_report(meta)

    # Find bucket [1,1]: count=1, total=1
    bucket_1 = next(b for b in report.gap_histogram if b.min_bars == 1)
    assert bucket_1.count == 1
    assert bucket_1.total_missing_bars == 1

    # Find bucket [2,5]: count=1, total=3
    bucket_2_5 = next(b for b in report.gap_histogram if b.min_bars == 2)
    assert bucket_2_5.count == 1
    assert bucket_2_5.total_missing_bars == 3


# ---------------------------------------------------------------------------
# Test: missing rate over a sub-window
# ---------------------------------------------------------------------------


def test_window_missing_rate_clean_window() -> None:
    """A window with no gaps returns 0% missing."""
    meta = _meta(
        rows=100,
        missing_bars=5,
        gaps=(Gap(first_missing_ts=2_000_000.0, last_missing_ts=2_000_000.0, bars=1),),
        first_ts=1_700_000_000.0,
    )
    # Query a window far from the gap
    rate = missing_rate_pct_for_window(meta, 1_700_000_000.0, 1_700_000_000.0 + 10 * 3600)
    assert rate == 0.0


def test_window_missing_rate_with_gap_inside() -> None:
    """A window that contains a gap returns a positive missing rate."""
    gap_ts = 1_700_010_000.0
    gap = Gap(first_missing_ts=gap_ts, last_missing_ts=gap_ts, bars=1)
    meta = _meta(
        rows=100,
        missing_bars=1,
        gaps=(gap,),
        first_ts=1_700_000_000.0,
        interval_seconds=3600.0,
    )

    # Window that contains the gap
    start = gap_ts - 3600
    end = gap_ts + 3600
    rate = missing_rate_pct_for_window(meta, start, end)
    assert rate > 0.0


# ---------------------------------------------------------------------------
# Test: horizon flag
# ---------------------------------------------------------------------------


def test_horizon_covered_reported() -> None:
    """horizon_covered reflects the meta's horizon_reached flag."""
    meta_with = _meta(horizon_reached=True)
    meta_without = _meta(horizon_reached=False)
    assert build_report(meta_with).horizon_covered is True
    assert build_report(meta_without).horizon_covered is False


# ---------------------------------------------------------------------------
# Test: measured facts consistency
# ---------------------------------------------------------------------------


def test_measured_tables_have_expected_coins() -> None:
    """The measured missing-rate tables contain all 24 universe coins."""
    universe_coins = {
        "SLERF",
        "BOME",
        "MEW",
        "WIF",
        "POPCAT",
        "PNUT",
        "MOODENG",
        "GIGA",
        "AURA",
        "GOAT",
        "CHILLGUY",
        "PONKE",
        "ACT",
        "FWOG",
        "GME",
        "DADDY",
        "MICHI",
        "SC",
        "BILLY",
        "RETARDIO",
        "BODEN",
        "LOCKIN",
        "BONK",
        "MOTHER",
    }
    assert set(MEASURED_5M_MISSING.keys()) == universe_coins
    assert set(MEASURED_1H_MISSING.keys()) == universe_coins


def _measured_5m_rates() -> list[float]:
    """The 5m rates that were actually measured, dropping the unmeasured ones.

    ``None`` entries are not zero and not comparable — they mean the rate was
    never measured for that coin.  Folding them into a ``min()``/``max()`` is
    exactly the mistake these tests exist to catch, so they are excluded here
    explicitly rather than by accident.
    """
    return [v for v in MEASURED_5M_MISSING.values() if v is not None]


def test_act_5m_is_unmeasured() -> None:
    """Guard: ACT's 5m rate is None, not a number.

    CANNOT-REPLAY.md's 5m table omits ACT — its 5m series spans 51 days
    against the ~209-day window behind every other figure.  If this ever
    becomes a float again, someone has substituted an estimate for a
    measurement, and the floor/ceiling tests below silently start ranking
    against invented data.
    """
    assert MEASURED_5M_MISSING["ACT"] is None


def test_slerf_5m_is_highest_missing() -> None:
    """SLERF has the highest 5m missing rate in the measured table."""
    assert MEASURED_5M_MISSING["SLERF"] == max(_measured_5m_rates())


def test_bonk_5m_is_lowest_missing() -> None:
    """BONK has the lowest 5m missing rate in the measured table."""
    assert MEASURED_5M_MISSING["BONK"] == min(_measured_5m_rates())
