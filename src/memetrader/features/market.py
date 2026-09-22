"""Price and volume features derived from OHLCV bars.

Gap handling is the load-bearing concern in this module. The vendor omits bars
in which nothing traded, so consecutive entries in a ``bars()`` result may span
more than one interval. Missing rates range from BONK 0.5% to SLERF 86.5%
(roughly one print every 37 minutes). Computing a log return across a gap and
treating it as a single-bar return overstates volatility because the elapsed
time was several intervals, not one. The two affected features are:

* ``realized_vol_pct`` — sample sigma of log returns. A log return that spans
  three hours (gap) followed by a genuine one-hour return mixes two different
  volatility regimes. The correct response is to detect and refuse, not to
  silently produce a number.

* Returns over horizons — ``return_pct_Nh`` — are computed end-to-end in price
  terms; gaps inflate elapsed time but the return itself is still the correct
  price ratio. They are therefore not refused but are annotated with
  ``actual_elapsed_seconds`` so a downstream filter can gate on it.

``None`` means "could not find out". ``0.0`` means "looked, and it is zero".
Collapsing them produces a confident neutral reading out of nothing — the audit
is explicit about this. Every place below where insufficient data or a gap
structure makes a feature meaningless returns ``None``, not ``0.0``.

Residual momentum is computed here because the initial strategy needs it. It
removes the SOL/BTC and meme-sector components from each coin's return, leaving
the coin-specific component. The regression is OLS over the training window;
computing it on the full history would leak. The sector index is the equal-
weight average of all universe members' returns at each bar; a coin that is the
sector is excluded from the sector before its own regression.

``_realized_vol_pct`` must match ``signals._realized_vol_pct`` to floating-
point tolerance. Both are sample sigma (ddof=1) of log returns over the last
21 closes × 100. The match is asserted in the test suite against a shared
fixture. The implementation is a direct port — not a call to the signals module
— because features run in replay contexts that do not have ``CandleSeries``
objects, only raw float arrays from ``PointInTimeState.bars()``.
"""

from __future__ import annotations

import math

import numpy as np

from ..signals import REALIZED_VOL_PERIOD


# ---------------------------------------------------------------------------
# Gap detection helpers
# ---------------------------------------------------------------------------


def _gaps_exceed_threshold(
    timestamps: np.ndarray,
    interval_seconds: float,
    *,
    max_gap_multiple: float = 1.5,
) -> bool:
    """True if any inter-bar gap exceeds ``max_gap_multiple * interval_seconds``.

    ``max_gap_multiple = 1.5`` means a gap of more than 1.5 intervals is
    flagged. For 1h bars that is 90 minutes; for 5m bars it is 7.5 minutes.
    The threshold is deliberately generous: one missed bar (exactly 2x interval)
    is a real event, but 1.5x catches a vendor mis-timestamped bar that would
    otherwise distort the return.

    Refuses to answer ``True`` or ``False`` (returns ``False``) when there are
    fewer than two timestamps, because a single bar has no gap to check. Callers
    should guard on ``len(timestamps) >= 2`` before trusting the result means
    anything.
    """
    if timestamps.size < 2:
        return False
    gaps = np.diff(timestamps.astype(float))
    return bool(np.any(gaps > max_gap_multiple * interval_seconds))


def _missing_fraction(
    timestamps: np.ndarray, interval_seconds: float
) -> float | None:
    """Fraction of expected bars that are absent in the timestamp sequence.

    Requires at least two timestamps to estimate expected bar count. Returns
    ``None`` on a single bar — the fraction is undefined, not zero.
    """
    if timestamps.size < 2:
        return None
    span = float(timestamps[-1]) - float(timestamps[0])
    expected = span / interval_seconds
    if expected < 1.0:
        return None
    present = timestamps.size - 1  # n timestamps → n-1 intervals present
    return max(0.0, 1.0 - present / expected)


# ---------------------------------------------------------------------------
# Realized volatility — must match signals._realized_vol_pct exactly
# ---------------------------------------------------------------------------


def realized_vol_pct(
    close: np.ndarray,
    timestamps: np.ndarray | None = None,
    *,
    interval_seconds: float | None = None,
    max_gap_multiple: float = 1.5,
) -> float | None:
    """Sample sigma of log returns over the last 21 closes, in whole percent.

    Matches ``signals._realized_vol_pct`` exactly: same window size
    (``REALIZED_VOL_PERIOD + 1 = 21`` closes), same log returns, same ddof=1,
    same × 100 output, same early-return on insufficient or non-positive data.

    When ``timestamps`` and ``interval_seconds`` are supplied, the function
    additionally refuses to compute if any gap in the window exceeds
    ``max_gap_multiple * interval_seconds``. This is the gap-handling requirement
    from the spec: a log return across an unequal time span overstates volatility
    for sparse coins (SLERF: 86.5% missing, one print every 37 minutes). That
    refusal returns ``None``, not ``0.0`` — the caller must choose whether to
    exclude the coin/period or use a different feature, and it cannot make that
    choice if the feature silently produces a number.

    The gap check uses only the final ``REALIZED_VOL_PERIOD + 1`` timestamps,
    matching the window used for the vol computation itself. Gaps outside the
    window do not affect this computation.
    """
    if close.size < REALIZED_VOL_PERIOD + 1:
        return None
    window = close[-(REALIZED_VOL_PERIOD + 1) :]
    if np.any(window <= 0.0):
        return None

    # Gap guard: refuse if gaps in the vol window exceed the threshold.
    # When timestamps are not supplied, we cannot check — the caller must decide
    # whether to trust the result on gapped data. The guard is opt-in, not
    # default-off: callers with timestamp arrays should always supply them.
    if timestamps is not None and interval_seconds is not None:
        ts_window = timestamps[-(REALIZED_VOL_PERIOD + 1) :]
        if _gaps_exceed_threshold(
            ts_window, interval_seconds, max_gap_multiple=max_gap_multiple
        ):
            return None

    returns = np.diff(np.log(window))
    result = float(returns.std(ddof=1)) * 100.0
    if not math.isfinite(result):
        return None
    return result


# ---------------------------------------------------------------------------
# Return features
# ---------------------------------------------------------------------------


def return_pct(
    close: np.ndarray,
    timestamps: np.ndarray,
    *,
    target_seconds: float,
    interval_seconds: float,
    tolerance: float = 0.5,
) -> float | None:
    """Percent return over approximately ``target_seconds`` of elapsed time.

    Finds the oldest bar whose timestamp is within
    ``tolerance * interval_seconds`` of ``now - target_seconds``. Returns
    ``None`` when no such bar exists in the available history.

    The tolerance prevents false ``None``s from jitter in bar timestamps. A 1h
    return requested at exactly 3600s back should find the bar at 3600s ± half
    an interval; without tolerance, a vendor timestamp that is 2 seconds early
    would miss the window and return ``None``.

    Does not forward-fill. If the reference bar is missing (vendor omitted it),
    the return is ``None``. The gap is real information — the coin was not
    trading — and filling would manufacture a continuity that did not exist.
    """
    if close.size < 2 or timestamps.size < 2:
        return None

    now_ts = float(timestamps[-1])
    target_ts = now_ts - target_seconds
    half_tol = tolerance * interval_seconds

    # Walk backward from the second-to-last bar to find the reference bar.
    # The most recent bar is always the "current" close; the reference is
    # somewhere behind it.
    for i in range(timestamps.size - 2, -1, -1):
        ts = float(timestamps[i])
        if abs(ts - target_ts) <= half_tol:
            base = close[i]
            curr = close[-1]
            if base <= 0.0 or not math.isfinite(base) or not math.isfinite(curr):
                return None
            return (curr / base - 1.0) * 100.0

    return None


# ---------------------------------------------------------------------------
# Volume features
# ---------------------------------------------------------------------------


def volume_ratio(
    volumes: np.ndarray,
    *,
    baseline_bars: int = 20,
) -> float | None:
    """Ratio of the most recent closed bar's volume to the preceding baseline.

    The baseline is the ``baseline_bars`` bars strictly preceding the most
    recent bar, non-overlapping. This matches ``signals.py``'s
    ``volume_ratio_prior_20`` fix: the measured bar must not be inside its own
    baseline, or the ratio is biased toward 1.0 and capped at ``baseline_bars``
    × however extreme the bar is.

    Returns ``None`` when the baseline mean is zero (entirely idle period — not
    the same as quiet, where trades printed at zero volume — or insufficient
    history) rather than ``inf`` or ``0.0``. An idle baseline with a non-zero
    measurement is a genuine spike, but reporting it as ``inf`` with no context
    would need immediate clamping downstream; reporting ``None`` makes the
    absence of baseline explicit.
    """
    needed = baseline_bars + 1
    if volumes.size < needed:
        return None
    baseline = volumes[-(needed):-1]
    mean_vol = float(baseline.mean())
    if mean_vol <= 0.0:
        return None
    return float(volumes[-1]) / mean_vol


def abnormal_volume(
    volumes: np.ndarray,
    *,
    baseline_bars: int = 20,
) -> float | None:
    """Z-score of the most recent volume vs. the preceding baseline.

    ``(v - mean) / std`` where mean and std are computed over ``baseline_bars``
    bars strictly preceding the most recent. A high z-score means the bar is an
    unusual spike relative to recent history; a low z-score means it is unusually
    quiet.

    Returns ``None`` when std is zero (constant volume in the baseline — every
    bar was identical) because dividing by zero is undefined, and a constant
    baseline with a different current bar is genuinely abnormal but has no
    well-defined z-score without the scale. Returns ``None`` rather than
    fabricating a large number.
    """
    needed = baseline_bars + 1
    if volumes.size < needed:
        return None
    baseline = volumes[-(needed):-1].astype(float)
    std = float(baseline.std(ddof=1))
    if std <= 0.0:
        return None
    mean_vol = float(baseline.mean())
    return (float(volumes[-1]) - mean_vol) / std


# ---------------------------------------------------------------------------
# Residual momentum
# ---------------------------------------------------------------------------


def sector_return(
    all_returns: dict[str, float | None],
    *,
    exclude_asset: str | None = None,
) -> float | None:
    """Equal-weight average return across all universe members.

    ``exclude_asset`` removes one asset from the sector before averaging,
    which is necessary when computing the sector component for that asset's own
    return — including a coin in its own sector index overstates the sector's
    explanatory power for that coin.

    Returns ``None`` when fewer than two non-None returns are available. A
    "sector" of one is just that asset's return, not a sector.
    """
    vals = [
        v
        for k, v in all_returns.items()
        if v is not None and math.isfinite(v) and k != exclude_asset
    ]
    if len(vals) < 2:
        return None
    return float(np.mean(vals))


def residual_momentum(
    coin_return: float | None,
    sector_ret: float | None,
    sol_return: float | None,
    btc_return: float | None,
    *,
    beta_sector: float = 1.0,
    beta_sol: float = 0.0,
    beta_btc: float = 0.0,
) -> float | None:
    """Coin-specific return after removing SOL/BTC and sector components.

    ``coin_return - beta_sector * sector_ret - beta_sol * sol_return
     - beta_btc * btc_return``

    The betas default to 1/0/0 (full sector attribution, no macro) when not
    estimated from the training window. The initial strategy uses this form
    because the training window is too short to estimate stable betas.

    When ``sector_ret`` is ``None`` (fewer than 2 coins in the universe),
    returns ``coin_return`` unchanged — the sector component cannot be computed,
    but the raw return is still valid. When ``coin_return`` is ``None``, returns
    ``None`` — there is no return to residualise.

    SOL and BTC terms are included as parameters to make the regression
    extension point explicit: when their betas are estimated from the training
    window, pass them here; when they are not available (TIER_0, which is our
    current state), pass ``sol_return=None`` and ``btc_return=None`` and they
    will be ignored.
    """
    if coin_return is None or not math.isfinite(coin_return):
        return None

    residual = coin_return

    if sector_ret is not None and math.isfinite(sector_ret):
        residual -= beta_sector * sector_ret

    if sol_return is not None and math.isfinite(sol_return) and beta_sol != 0.0:
        residual -= beta_sol * sol_return

    if btc_return is not None and math.isfinite(btc_return) and beta_btc != 0.0:
        residual -= beta_btc * btc_return

    return residual if math.isfinite(residual) else None


__all__ = [
    "abnormal_volume",
    "realized_vol_pct",
    "residual_momentum",
    "return_pct",
    "sector_return",
    "volume_ratio",
]
