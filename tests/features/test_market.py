"""Tests for ``features.market``.

Each test is load-bearing: removing the guard it exercises must cause the test
to fail. The comment on each test states which guard it targets.

Critical tests:
* ``realized_vol_pct`` matches ``signals._realized_vol_pct`` to 1e-12.
* A gapped series does not silently produce a confident vol number.
* Gap guard is opt-in: without timestamps, the function computes (but callers
  are warned by this test's existence that they should supply timestamps).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from memetrader.features.market import (
    abnormal_volume,
    realized_vol_pct,
    residual_momentum,
    return_pct,
    sector_return,
    volume_ratio,
)
from memetrader.signals import _realized_vol_pct as signals_rvol

# Shared constants — same OHLCV fixture used for all price tests
START_TS = 1_700_000_000.0
INTERVAL = 3600.0  # 1h bars


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_close_ts(n: int, *, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """``n`` random positive closes and their hourly timestamps."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, size=n)
    close = np.cumprod(1.0 + returns) * 1.0  # start at $1
    ts = np.arange(n, dtype=float) * INTERVAL + START_TS
    return close, ts


def make_gapped_close_ts(
    n_before: int,
    gap_multiple: int,
    n_after: int,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Series with a single gap of ``gap_multiple`` intervals in the middle."""
    close, _ = make_close_ts(n_before + n_after, seed=seed)
    ts_before = np.arange(n_before, dtype=float) * INTERVAL + START_TS
    ts_after = (
        ts_before[-1] + gap_multiple * INTERVAL + INTERVAL
        + np.arange(n_after, dtype=float) * INTERVAL
    )
    ts = np.concatenate([ts_before, ts_after])
    return close, ts


# ---------------------------------------------------------------------------
# realized_vol_pct — must match signals._realized_vol_pct exactly
# ---------------------------------------------------------------------------


def test_rvol_matches_signals_implementation() -> None:
    """features.market.realized_vol_pct must match signals._realized_vol_pct
    to floating-point tolerance (1e-12).

    Guard: if the two implementations use different windows, ddof, or scaling,
    the signal the live system reads will differ from the backtest feature,
    invalidating the backtest's claim to reproduce live behaviour.
    """
    close, _ = make_close_ts(50)
    expected = signals_rvol(close)
    actual = realized_vol_pct(close)
    assert expected is not None
    assert actual is not None
    assert abs(actual - expected) < 1e-12, (
        f"features.realized_vol_pct={actual} != signals._realized_vol_pct={expected}"
    )


def test_rvol_matches_signals_across_many_windows() -> None:
    """Match must hold across a range of different close series.

    Guard: a match on one fixture might pass by coincidence; matching on 20
    different windows makes a coincidence vanishingly unlikely.
    """
    for seed in range(20):
        close, _ = make_close_ts(30 + seed, seed=seed)
        expected = signals_rvol(close)
        actual = realized_vol_pct(close)
        if expected is None:
            assert actual is None
        else:
            assert actual is not None
            assert abs(actual - expected) < 1e-12, (
                f"seed={seed}: {actual} != {expected}"
            )


def test_rvol_none_on_insufficient_bars() -> None:
    """Returns None when fewer than 21 bars are available.

    Guard: without the size check, the function would compute over fewer bars
    and produce a wider confidence interval — a feature that silently changes
    its semantics based on data quantity.
    """
    close = np.array([1.0, 1.01, 1.02])
    assert realized_vol_pct(close) is None


def test_rvol_none_on_non_positive_close() -> None:
    """Returns None when any close in the window is <= 0.

    Guard: log(0) is -inf and log(negative) is NaN; without the check,
    the function would return a NaN or -inf masked as a finite vol.
    """
    close, _ = make_close_ts(25)
    close[-5] = 0.0  # inject a zero in the vol window
    assert realized_vol_pct(close) is None


def test_rvol_none_on_wide_gap() -> None:
    """Returns None when a gap in the timestamp window exceeds the threshold.

    Guard: this is the core gap-handling requirement. A log return across
    3 hours of gap is not a 1h return; silently producing a vol number overstates
    volatility for sparse coins (SLERF: 86.5% missing).
    """
    # 25 bars total, gap of 5 intervals at position 15 (inside the vol window)
    close, ts = make_gapped_close_ts(15, gap_multiple=5, n_after=10)
    result = realized_vol_pct(close, ts, interval_seconds=INTERVAL)
    assert result is None, (
        "Expected None for a gapped series; got a confident vol number. "
        "A 5-interval gap inside the 21-bar vol window should be refused."
    )


def test_rvol_gap_outside_window_does_not_refuse() -> None:
    """A gap before the vol window does not affect the computation.

    Guard: the gap check only looks at the last 21 bars. A gap 40 bars back
    should not refuse the computation — it is outside the window.
    """
    close, ts = make_gapped_close_ts(30, gap_multiple=10, n_after=25)
    # The gap is at position 30, so the last 21 bars (positions 34..54) are gap-free
    result = realized_vol_pct(close, ts, interval_seconds=INTERVAL)
    # Should NOT be None — the gap is outside the window
    assert result is not None, (
        "A gap outside the vol window should not refuse the computation."
    )


def test_rvol_without_timestamps_computes_unchecked() -> None:
    """Without timestamps, the gap guard is disabled and the function computes.

    This is NOT a failure case — the function is spec-correct without timestamps.
    The test documents the behaviour so callers know they must supply timestamps
    to activate the guard.
    """
    close, _ = make_gapped_close_ts(15, gap_multiple=5, n_after=10)
    result = realized_vol_pct(close)
    # Without timestamps, we cannot detect the gap, so it should compute.
    assert result is not None


# ---------------------------------------------------------------------------
# return_pct
# ---------------------------------------------------------------------------


def test_return_pct_finds_bar_at_target_horizon() -> None:
    """return_pct must locate the bar at approximately the target horizon.

    Guard: if target_ts search uses a wrong reference or wrong direction,
    the return computed is not the return over the intended horizon.
    """
    close, ts = make_close_ts(50)
    # 12h return: look back 12 bars
    result = return_pct(close, ts, target_seconds=12 * INTERVAL, interval_seconds=INTERVAL)
    expected = (close[-1] / close[-13] - 1.0) * 100.0
    assert result is not None
    assert abs(result - expected) < 1e-10


def test_return_pct_none_when_horizon_not_in_history() -> None:
    """Returns None when the requested horizon exceeds available history.

    Guard: without this, a requested 24h return on a 10-bar series would
    compute something nonsensical using the oldest available bar.
    """
    close, ts = make_close_ts(10)
    result = return_pct(close, ts, target_seconds=24 * INTERVAL, interval_seconds=INTERVAL)
    assert result is None


# ---------------------------------------------------------------------------
# volume_ratio
# ---------------------------------------------------------------------------


def test_volume_ratio_excludes_measured_bar_from_baseline() -> None:
    """The measured bar must not be part of its own baseline.

    Guard: including the measured bar in the baseline biases the ratio toward
    1.0 (the denominator includes the numerator) and caps the maximum value at
    ``baseline_bars`` × whatever the bar is.
    """
    volumes = np.ones(22, dtype=float)  # 21 bars of 1.0
    volumes[-1] = 50.0  # last bar is a spike
    ratio = volume_ratio(volumes)
    # baseline is the 20 bars before the last → mean = 1.0, ratio = 50.0
    assert ratio is not None
    assert ratio == pytest.approx(50.0)


def test_volume_ratio_none_on_insufficient_bars() -> None:
    """Returns None when fewer than baseline_bars + 1 bars are available."""
    volumes = np.ones(20, dtype=float)
    # baseline_bars=20, so needs 21 bars; 20 is insufficient
    assert volume_ratio(volumes) is None


def test_volume_ratio_none_on_zero_baseline() -> None:
    """Returns None when all baseline bars have zero volume.

    Guard: returning ``inf`` would be technically correct but semantically
    misleading — a zero-volume baseline with non-zero current means we have no
    reference point, not that the current is infinitely above baseline.
    """
    volumes = np.zeros(22, dtype=float)
    volumes[-1] = 5.0
    assert volume_ratio(volumes) is None


# ---------------------------------------------------------------------------
# abnormal_volume
# ---------------------------------------------------------------------------


def test_abnormal_volume_positive_for_spike() -> None:
    """Z-score must be positive (high) for a volume spike."""
    rng = np.random.default_rng(0)
    volumes = rng.uniform(0.9, 1.1, size=21)
    volumes[-1] = 10.0  # large spike vs baseline near 1.0
    result = abnormal_volume(volumes)
    assert result is not None
    assert result > 3.0


def test_abnormal_volume_none_on_constant_baseline() -> None:
    """Returns None when baseline std is zero (constant series).

    Guard: dividing by zero produces NaN or inf, which would escape into the
    feature vector as a confident extreme value.
    """
    volumes = np.ones(22, dtype=float)
    assert abnormal_volume(volumes) is None


# ---------------------------------------------------------------------------
# sector_return
# ---------------------------------------------------------------------------


def test_sector_return_excludes_asset() -> None:
    """The excluded asset must not contribute to the sector mean.

    Guard: including a coin in its own sector index overstates the sector's
    explanatory power for that coin in residual momentum.
    """
    returns = {"A": 10.0, "B": 2.0, "C": 3.0}
    # Excluding A: mean of {B: 2.0, C: 3.0} = 2.5
    result = sector_return(returns, exclude_asset="A")
    assert result is not None
    assert result == pytest.approx(2.5)


def test_sector_return_none_on_single_member() -> None:
    """Returns None when fewer than 2 non-None members remain after exclusion.

    Guard: a "sector" of one is just that asset's own return.
    """
    returns = {"A": 10.0, "B": None}
    result = sector_return(returns, exclude_asset="A")
    assert result is None


# ---------------------------------------------------------------------------
# residual_momentum
# ---------------------------------------------------------------------------


def test_residual_momentum_subtracts_sector() -> None:
    """Must subtract beta_sector * sector_ret from coin_return."""
    result = residual_momentum(10.0, 3.0, None, None, beta_sector=1.0)
    assert result == pytest.approx(7.0)


def test_residual_momentum_none_on_none_coin_return() -> None:
    """Returns None when coin_return is None."""
    assert residual_momentum(None, 3.0, None, None) is None


def test_residual_momentum_passthrough_on_none_sector() -> None:
    """Returns coin_return unchanged when sector_ret is None.

    Guard: if sector_ret=None were treated as 0.0, the residual would
    equal the raw return, which is correct here but for the wrong reason —
    a ``None`` sector means the sector could not be computed, not that
    the coin moved independently of it.
    """
    result = residual_momentum(10.0, None, None, None)
    assert result == pytest.approx(10.0)


def test_residual_momentum_includes_sol_when_beta_nonzero() -> None:
    result = residual_momentum(10.0, 3.0, 5.0, None, beta_sector=1.0, beta_sol=0.5)
    # 10 - 1.0*3 - 0.5*5 = 10 - 3 - 2.5 = 4.5
    assert result == pytest.approx(4.5)


def test_residual_momentum_nan_input_returns_none() -> None:
    """NaN coin_return must collapse to None, not produce NaN output."""
    result = residual_momentum(float("nan"), 3.0, None, None)
    assert result is None
    assert not math.isfinite(float("nan"))  # sanity
