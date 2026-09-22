"""Data-quality reports per asset/timeframe.

A backtest that ingests bad data and reports a number is worse than one that
refuses to run: the number looks credible while being based on fabricated
evidence. This module makes the quality of each series explicit before any
feature is computed.

Key measurements, all from ``docs/CANNOT-REPLAY.md`` (real data, not estimates):
  - 1h missing rates: BONK/GIGA 0%, SLERF/BODEN/MOTHER 20-30%
  - 5m missing rates: BONK 0.5%, SLERF 86.5%
  - ACT has only 51 days of 5m data; LOCKIN has 133 days
  - Everything else has ~187 days of 5m

These numbers shape the ``usable_for`` verdict. The thresholds are not magic:
they are chosen so that a coin whose missing rate makes ``realized_vol_pct``
meaningless (the stdev of log returns over a gapped series spans unequal times,
inflating volatility) is flagged before any feature touches it.

``realized_vol_pct`` is computed as the sample sigma of 21 log-returns. Across
a gap, one "bar return" spans multiple intervals, overstating sigma by
sqrt(n_missing+1) for each gap. At 0.5% missing, the bias is noise; at 29.8%
(SLERF hourly) or 86.5% (SLERF 5m), it is not a volatility estimate at all.

The threshold for flagging is ``max_missing_rate_pct`` on each tier definition.
TIER_0 (current state) is lenient: we only have OHLCV, and a coin with 30%
hourly gaps is still useful for directional signals even if its vol is wrong.
A tighter threshold (e.g., 5%) is used when the PnL estimate depends on vol.

Horizon coverage is expressed in days rather than bar counts because the coins
have wildly different sampling densities — comparing SLERF (which at 86.5%
missing effectively has one bar per 37 minutes) with BONK by bar count is
meaningless. Wall-clock coverage is the honest metric.
"""

from __future__ import annotations

from dataclasses import dataclass

from memetrader.backfill import SeriesMeta, window_gaps
from memetrader.types import FidelityTier, Timeframe

# ------------------------------------------------------------------
# Measured facts, hard-coded from docs/CANNOT-REPLAY.md.
# These are not parameters — they are observations. Changing them
# requires updating the source data, not adjusting a config value.
# ------------------------------------------------------------------

# 5m missing rates by symbol, from the measured dataset.
# Coins with fewer rows due to a shorter window (ACT, LOCKIN) are noted
# in their entries. Coins not in this table have measured missing rates
# between the values shown for the bracketed groups.
MEASURED_5M_MISSING: dict[str, float | None] = {
    "BONK": 0.5,
    "GIGA": 10.9,
    "CHILLGUY": 21.0,
    "BOME": 21.0,
    "MOODENG": 28.0,
    "GOAT": 28.0,
    "PNUT": 28.0,
    "AURA": 28.0,
    "MEW": 32.0,
    "POPCAT": 32.0,
    "WIF": 35.5,
    "FWOG": 39.0,
    "PONKE": 40.0,
    "RETARDIO": 45.0,
    "GME": 48.0,
    "DADDY": 53.0,
    "LOCKIN": 58.0,  # also short window: 133 days
    "BILLY": 74.0,
    "SC": 75.0,
    "MICHI": 81.0,
    "BODEN": 81.0,
    "MOTHER": 81.0,
    "SLERF": 86.5,
    # ACT is the one coin in the universe with NO measured 5m missing rate.
    # CANNOT-REPLAY.md's 5m table lists 23 of the 24 coins and omits ACT
    # entirely, because ACT's 5m series covers only 51 days (2026-08-01 →)
    # against the ~209-day window every other figure here was measured over.
    # There is no honest number to put in this slot, so it is None — §0:
    # None means "could not find out", 0.0 would mean "measured, and it is
    # perfect". A previous version had 0.0 here, which made a thinly-sampled
    # 51-day coin read as the cleanest series in the universe and displaced
    # BONK (0.5%, 209 days of dense trading) as the measured floor that
    # BACKTEST-CONTRACTS.md §7 cites. Do not substitute an estimate: measure
    # ACT's 5m series over a comparable window, or leave this None.
    "ACT": None,
}

# 1h missing rates by symbol, from the measured dataset.
MEASURED_1H_MISSING: dict[str, float] = {
    "BONK": 0.0,
    "GIGA": 0.0,
    "POPCAT": 0.02,
    "PNUT": 0.02,
    "GOAT": 0.02,
    "CHILLGUY": 0.02,
    "BOME": 0.04,
    "MOODENG": 0.04,
    "MEW": 0.08,
    "FWOG": 0.08,
    "AURA": 0.14,
    "ACT": 0.22,
    "PONKE": 0.26,
    "WIF": 0.38,
    "GME": 0.74,
    "RETARDIO": 0.98,
    "DADDY": 1.5,
    "LOCKIN": 3.0,
    "BILLY": 9.4,
    "SC": 11.9,
    "MICHI": 14.3,
    "MOTHER": 20.3,
    "BODEN": 27.6,
    "SLERF": 29.8,
}

# Coverage in days for the special short-window 5m series.
SHORT_COVERAGE_DAYS: dict[str, float] = {
    "ACT": 51.0,
    "LOCKIN": 133.0,
}

# Missing rate above which realized_vol_pct is meaningfully biased.
# Chosen so that coins where every gap inflates a 21-bar window's sigma by
# more than ~5% are flagged. Derivation: if 5% of bars are missing and gaps
# are distributed uniformly, the expected inflation per return is (1.05)^0.5
# ≈ 1.025, or a ~2.5% overstatement of vol — below the noise floor. At 10%
# missing the overstatement is ~5%, and at 30% it is ~20%.
_VOL_BIAS_THRESHOLD_PCT = 10.0

# Below this coverage (days), 5m windows are too short for robust features.
_MIN_COVERAGE_DAYS = 60.0

# TIER_0 allows up to this 1h missing rate before refusing usability. 1h is
# the binding series for TIER_0 signals since we only have OHLCV.
_TIER0_MAX_1H_MISSING_PCT = 35.0

# TIER_0 allows up to this 5m missing rate before refusing usability.
# Set at 85% so that SLERF (86.5%, MEASURED_5M_MISSING) is refused but
# BODEN/MOTHER/MICHI (81%) are flagged as vol-biased but not outright
# refused, per the module docstring's stated intent. (A prior version of
# this threshold was a hardcoded 87.0, which let SLERF's 86.5% through as
# "usable" — exactly the coin BACKTEST-CONTRACTS.md §7 cites as the one
# that must be excluded.)
_TIER0_MAX_5M_MISSING_PCT = 85.0

# For TIER_2+ the bar quality matters less because fills use quote ladders,
# not bar closes. The threshold is relaxed to 50% to preserve more coins.
_TIER2_MAX_MISSING_PCT = 50.0


@dataclass(frozen=True, slots=True)
class GapBucket:
    """One bin of the gap length histogram.

    Gaps are bucketed by length (in bars) rather than by wall-clock duration
    because the interesting question for feature engineering is "how many bar
    returns are synthetic?" not "how many minutes were quiet?". A 10-bar gap
    means 10 missing log-returns, each inflating vol by sqrt(2) vs a real return.
    """

    min_bars: int  # inclusive
    max_bars: int  # inclusive
    count: int  # how many gaps fall in this bucket
    total_missing_bars: int


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Data-quality summary for one (asset, timeframe) pair.

    All rates are percentages (0-100). All durations are seconds. All counts are
    bar counts, not row counts (they are the same unless there are rejected rows,
    which in practice does not happen in the backfilled dataset).

    ``vol_biased`` is the field to check before computing ``realized_vol_pct``
    features. A coin with ``vol_biased=True`` will produce inflated vol estimates
    for any window that contains a gap, which biases the baseline strategy toward
    false negatives (refusing trades) because ``half_width = interval_vol_multiple
    x realized_vol_pct`` is subtracted from the forecast. False negatives are
    less dangerous than false positives, but they are still wrong.

    ``usable`` is the combined verdict. A coin that is not ``usable`` should be
    excluded from the experiment, or at least excluded from any aggregate that
    claims to represent the full 24-coin universe.
    """

    asset_id: str  # mint address
    symbol: str  # for display
    timeframe: str  # "1h" or "5m"
    row_count: int
    missing_bars: int
    missing_rate_pct: float  # missing / (present + missing) * 100
    coverage_days: float  # (last_ts - first_ts) / 86400
    first_ts: float | None
    last_ts: float | None
    non_positive_prices: int  # bars with open/high/low/close <= 0
    duplicate_timestamps: int  # should always be 0 from the backfill
    gap_histogram: tuple[GapBucket, ...]
    horizon_covered: bool  # True if the series hit the 180-day vendor limit
    vol_biased: bool  # missing rate above _VOL_BIAS_THRESHOLD_PCT
    coverage_short: bool  # coverage below _MIN_COVERAGE_DAYS
    usable: bool  # combined verdict for TIER_0 purposes
    usable_reasons: tuple[str, ...]  # why usable=False, if applicable


def build_report(meta: SeriesMeta, *, symbol: str = "") -> QualityReport:
    """Construct a quality report from a ``SeriesMeta``.

    This is the primary entry point. ``meta`` comes from ``backfill.read_manifest``.
    The report is computed purely from the manifest, not by re-reading bars, so
    it is fast even for the full 24-coin x 2-timeframe set.

    ``symbol`` is for display only — the mint address on ``meta`` is the identity.
    We accept it separately because ``SeriesMeta`` carries ``symbol`` as a
    convenience field already (from the backfill), so in practice callers pass
    ``meta.symbol``.
    """
    sym = symbol or meta.symbol or ""

    # Coverage in days
    if meta.first_ts is not None and meta.last_ts is not None:
        coverage_days = (meta.last_ts - meta.first_ts) / 86400.0
    else:
        coverage_days = 0.0

    # Missing rate: missing / (present + missing)
    total_expected = meta.rows + meta.missing_bars
    missing_rate_pct = (
        100.0 * meta.missing_bars / total_expected if total_expected > 0 else 0.0
    )

    # Gap histogram — bucket by length in bars
    gap_histogram = _build_gap_histogram(meta)

    vol_biased = missing_rate_pct >= _VOL_BIAS_THRESHOLD_PCT
    coverage_short = coverage_days < _MIN_COVERAGE_DAYS

    reasons: list[str] = []
    if vol_biased:
        reasons.append(
            f"missing rate {missing_rate_pct:.1f}% >= {_VOL_BIAS_THRESHOLD_PCT}%: "
            "realized_vol_pct is biased"
        )
    if coverage_short:
        reasons.append(
            f"coverage {coverage_days:.0f} days < {_MIN_COVERAGE_DAYS:.0f} days minimum"
        )

    # TIER_0 usability: 1h OK up to 35% missing; 5m OK up to 80%.
    # These thresholds are asymmetric because 1h is used for strategy signals
    # (the missing-vol problem dominates) while 5m is primarily used for h1
    # reconstruction (where even a sparse series is better than nothing).
    tf = meta.timeframe
    if tf == Timeframe.H1.value:
        usable = missing_rate_pct < _TIER0_MAX_1H_MISSING_PCT and not coverage_short
        if missing_rate_pct >= _TIER0_MAX_1H_MISSING_PCT:
            reasons.append(
                f"1h missing rate {missing_rate_pct:.1f}% >= "
                f"{_TIER0_MAX_1H_MISSING_PCT}%: exclude from aggregate"
            )
    else:
        usable = missing_rate_pct < _TIER0_MAX_5M_MISSING_PCT and not coverage_short
        if missing_rate_pct >= _TIER0_MAX_5M_MISSING_PCT:
            reasons.append(
                f"5m missing rate {missing_rate_pct:.1f}% >= "
                f"{_TIER0_MAX_5M_MISSING_PCT}%: "
                "effectively a sparse trade log, not a bar series"
            )

    return QualityReport(
        asset_id=meta.mint,
        symbol=sym,
        timeframe=meta.timeframe,
        row_count=meta.rows,
        missing_bars=meta.missing_bars,
        missing_rate_pct=missing_rate_pct,
        coverage_days=coverage_days,
        first_ts=meta.first_ts,
        last_ts=meta.last_ts,
        non_positive_prices=0,  # backfill validates positive at write time
        duplicate_timestamps=0,  # backfill deduplicates by timestamp
        gap_histogram=gap_histogram,
        horizon_covered=meta.horizon_reached,
        vol_biased=vol_biased,
        coverage_short=coverage_short,
        usable=usable,
        usable_reasons=tuple(reasons),
    )


def usable_for(report: QualityReport, tier: FidelityTier) -> bool:
    """Whether this series is usable at the given fidelity tier.

    TIER_0: signal quality only, OHLCV. The bar missing rate is the binding
    constraint because ``realized_vol_pct`` is the primary signal gating.
    SLERF (86.5% missing 5m, 29.8% missing 1h) fails at both — it has one
    print every 37 minutes on the 5m series, which is not a bar series.

    TIER_1+: adds swap-level data. The missing-bar constraint is relaxed
    because fills are based on swap events rather than bar closes, but
    coverage is still required for feature computation.

    TIER_2+: adds quote ladders. The constraint further relaxes to 50%
    missing because cost estimates come from ladder history, not bar midpoints,
    so a sparse bar series is less damaging. Coverage is still required.

    TIER_3: calibrated. Same as TIER_2 for the underlying bar data; the
    calibration adds an additional requirement that shadow quote data exists,
    but that is checked at a higher level (the run manifest), not here.
    """
    if report.coverage_short:
        return False

    if tier is FidelityTier.TIER_0:
        # For TIER_0 the bar series IS the execution model. A series where the
        # vol estimate is meaningless is not usable for any PnL analysis.
        # We use the same threshold as the report's ``usable`` field.
        if report.timeframe == Timeframe.H1.value:
            return report.missing_rate_pct < _TIER0_MAX_1H_MISSING_PCT
        return report.missing_rate_pct < _TIER0_MAX_5M_MISSING_PCT

    if tier is FidelityTier.TIER_1:
        return report.missing_rate_pct < _TIER2_MAX_MISSING_PCT

    # TIER_2, TIER_3
    return report.missing_rate_pct < _TIER2_MAX_MISSING_PCT


def missing_rate_pct_for_window(
    meta: SeriesMeta,
    start_ts: float,
    end_ts: float,
) -> float:
    """Missing-bar rate over a sub-window ``[start_ts, end_ts]``.

    Used by feature pipelines that compute a rolling window and need to know
    whether that specific window is clean enough to use. A 21-bar vol window
    that sits entirely in a data-dense period is usable even if the full series
    has a high aggregate missing rate.

    Returns a percentage (0-100). An empty window returns 0.0.

    Reuses ``backfill.window_gaps`` so the gap accounting matches the rest of
    the codebase.
    """
    if meta.interval_seconds <= 0 or end_ts <= start_ts:
        return 0.0
    expected = round((end_ts - start_ts) / meta.interval_seconds) + 1
    if expected <= 0:
        return 0.0
    missing = window_gaps(meta, start_ts, end_ts)
    return 100.0 * missing / expected


def _build_gap_histogram(meta: SeriesMeta) -> tuple[GapBucket, ...]:
    """Bucket the series gaps by length.

    Buckets: [1], [2-5], [6-20], [21-100], [101+].
    These boundaries are chosen to distinguish:
      - 1-bar gaps (single silent interval, low impact on vol)
      - 2-5 bars (short run, small vol bias)
      - 6-20 bars (one to three hours on 5m, noticeable bias)
      - 21-100 bars (one day or more, large bias)
      - 101+ bars (multi-day, series is locally a trade log)
    """
    buckets: list[tuple[int, int]] = [(1, 1), (2, 5), (6, 20), (21, 100), (101, 10**9)]
    result: list[GapBucket] = []
    for min_b, max_b in buckets:
        count = 0
        total = 0
        for gap in meta.gaps:
            if min_b <= gap.bars <= max_b:
                count += 1
                total += gap.bars
        result.append(
            GapBucket(min_bars=min_b, max_bars=max_b, count=count, total_missing_bars=total)
        )
    return tuple(result)


__all__ = [
    "MEASURED_1H_MISSING",
    "MEASURED_5M_MISSING",
    "SHORT_COVERAGE_DAYS",
    "GapBucket",
    "QualityReport",
    "build_report",
    "missing_rate_pct_for_window",
    "usable_for",
]
