"""Tests for memetrader.validation.leakage.

Every test here is offline and deterministic. Per plan §13, each test must
FAIL if the guard it exercises is removed — the comments above each test
class say what change would flip it.

Critical tests:
- test_prefix_equivalence_real_feature_passes     — realized_vol_pct is prefix-safe
- test_future_sentinel_real_feature_passes         — poisoning the future doesn't
                                                      move a correctly-scoped feature
- test_leaky_full_mean_feature_is_caught_*         — a feature that centers on the
                                                      full-series mean IS flagged
- test_label_horizon_*_purge_{enabled,disabled}    — contamination appears with
                                                      enable_purge=False and is
                                                      absent with it True
- test_timestamp_monotonicity_*                    — a state that leaks a future
                                                      or unclosed bar IS flagged
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from memetrader.features import market
from memetrader.histdata.point_in_time import ReplayState
from memetrader.types import Candle, Timeframe
from memetrader.validation import leakage
from memetrader.validation.splits import Interval, LabeledObservation

_HOUR = 3600.0

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _price_series(n: int, *, seed: int = 7) -> np.ndarray:
    """A deterministic, strictly positive synthetic price series."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0, scale=0.01, size=n)
    return np.cumprod(1.0 + steps) * 100.0


def _vol_feature(series: np.ndarray, at_index: int) -> float | None:
    """Adapter: realized_vol_pct as a point-in-time FeatureFn.

    Truncates to ``series[: at_index + 1]`` before calling the real feature,
    which is exactly how a point-in-time pipeline must invoke it (see
    features/pipeline.py's fit/transform split docstring).
    """
    return market.realized_vol_pct(series[: at_index + 1])


def _vol_feature_forgets_to_truncate(series: np.ndarray, at_index: int) -> float | None:
    """Adapter that ignores at_index — models a caller who forgot to
    truncate before calling the feature. Demonstrates the check catches this
    caller mistake even though realized_vol_pct itself is not at fault."""
    return market.realized_vol_pct(series)


def _leaky_full_mean_centered(series: np.ndarray, at_index: int) -> float | None:
    """A deliberately leaky feature: centers on the mean of whatever array it
    is given, rather than only the data up to ``at_index``. The classic leak
    named in the assignment — a full-series mean is not knowable at ``t``."""
    return float(series[at_index] - series.mean())


# ---------------------------------------------------------------------------
# 1. Prefix equivalence on a real feature
# ---------------------------------------------------------------------------


class TestPrefixEquivalenceRealFeature:
    def test_passes_when_adapter_truncates_correctly(self) -> None:
        series = _price_series(60)
        result = leakage.check_prefix_equivalence(_vol_feature, series, at_index=40)
        assert result.status == leakage.CheckStatus.PASSED
        assert result.violations == 0

    def test_catches_a_caller_that_forgets_to_truncate(self) -> None:
        """Not a bug in realized_vol_pct — a bug in how it is called. The
        check must catch this too, because the invariant is "the value the
        pipeline actually produced at t", not "the feature function is
        theoretically capable of being called safely"."""
        series = _price_series(60)
        result = leakage.check_prefix_equivalence(
            _vol_feature_forgets_to_truncate, series, at_index=40
        )
        assert result.status == leakage.CheckStatus.FAILED
        assert result.violations == 1


# ---------------------------------------------------------------------------
# 2. Future sentinel on a real feature
# ---------------------------------------------------------------------------


class TestFutureSentinelRealFeature:
    def test_poisoning_the_future_does_not_move_a_correct_feature(self) -> None:
        series = _price_series(60)
        result = leakage.check_future_sentinel(_vol_feature, series, at_index=40)
        assert result.status == leakage.CheckStatus.PASSED
        assert result.violations == 0

    def test_nan_sentinel_also_leaves_a_correct_feature_unchanged(self) -> None:
        series = _price_series(60)
        result = leakage.check_future_sentinel(
            _vol_feature, series, at_index=40, sentinel=math.nan
        )
        assert result.status == leakage.CheckStatus.PASSED


# ---------------------------------------------------------------------------
# 3. A deliberately leaky feature IS caught
# ---------------------------------------------------------------------------


class TestLeakyFeatureIsCaught:
    def test_prefix_equivalence_flags_full_mean_centering(self) -> None:
        series = np.array([100.0, 101.0, 99.0, 500.0, 500.0, 500.0], dtype=float)
        # The future values (500.0, 500.0, 500.0) pull the full-series mean
        # well away from the prefix mean, so the leaky mean must differ.
        result = leakage.check_prefix_equivalence(
            _leaky_full_mean_centered, series, at_index=2
        )
        assert result.status == leakage.CheckStatus.FAILED
        assert result.violations == 1

    def test_future_sentinel_flags_full_mean_centering(self) -> None:
        series = np.array([100.0, 101.0, 99.0, 102.0, 98.0], dtype=float)
        result = leakage.check_future_sentinel(
            _leaky_full_mean_centered, series, at_index=2, sentinel=1.0e9
        )
        assert result.status == leakage.CheckStatus.FAILED
        assert result.violations == 1

    @given(
        values=st.lists(
            st.floats(
                min_value=-1.0e6,
                max_value=1.0e6,
                allow_nan=False,
                allow_infinity=False,
                width=64,
            ),
            min_size=3,
            max_size=25,
        )
    )
    def test_prefix_equivalence_matches_manual_mean_diff(self, values: list[float]) -> None:
        """Property test: the check's verdict must track whether the
        full-series mean actually differs from the prefix mean — not just
        "usually fail", but fail exactly when the leak would manifest."""
        series = np.array(values, dtype=float)
        at_index = series.size - 2  # always leaves >= 1 future observation
        prefix_mean = float(series[: at_index + 1].mean())
        full_mean = float(series.mean())
        would_leak = not math.isclose(prefix_mean, full_mean, rel_tol=1e-9, abs_tol=1e-9)

        result = leakage.check_prefix_equivalence(
            _leaky_full_mean_centered, series, at_index=at_index
        )
        expected = leakage.CheckStatus.FAILED if would_leak else leakage.CheckStatus.PASSED
        assert result.status == expected


# ---------------------------------------------------------------------------
# 4. Label horizon: contamination appears exactly when purge is disabled
# ---------------------------------------------------------------------------


class TestLabelHorizonPurgeGuard:
    def _scenario(self) -> tuple[list[LabeledObservation], list[Interval], Interval]:
        train_interval = Interval(0.0, 100.0)
        protected = [Interval(100.0, 200.0)]  # validation window
        observations = [
            # Fully inside train, label fully inside train: never a problem.
            LabeledObservation(
                asset_id="BONK", obs_time=10.0, label_start_ts=10.0, label_end_ts=20.0
            ),
            # obs_time inside train, but label spills into the protected
            # (validation) interval — must be purged.
            LabeledObservation(
                asset_id="BONK", obs_time=90.0, label_start_ts=90.0, label_end_ts=150.0
            ),
        ]
        return observations, protected, train_interval

    def test_no_contamination_when_purge_enabled(self) -> None:
        observations, protected, train_interval = self._scenario()
        result = leakage.check_label_horizon(
            observations,
            protected_intervals=protected,
            train_interval=train_interval,
            embargo_secs=0.0,
            enable_purge=True,
        )
        assert result.status == leakage.CheckStatus.PASSED
        assert result.violations == 0

    def test_contamination_appears_when_purge_disabled(self) -> None:
        observations, protected, train_interval = self._scenario()
        result = leakage.check_label_horizon(
            observations,
            protected_intervals=protected,
            train_interval=train_interval,
            embargo_secs=0.0,
            enable_purge=False,
        )
        assert result.status == leakage.CheckStatus.FAILED
        assert result.violations == 1


# ---------------------------------------------------------------------------
# 5. Timestamp monotonicity against PointInTimeState
# ---------------------------------------------------------------------------


class TestTimestampMonotonicity:
    def test_replay_state_never_exposes_a_future_or_open_bar(self) -> None:
        state = ReplayState(now=0.0, publication_delay_seconds=0.0)
        candle = Candle(ts=1000.0, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)
        state.add_bar(candle, asset_id="BONK", timeframe=Timeframe.H1)

        # ReplayState.bars is typed with the concrete Timeframe/Candle types, which
        # is a narrower (and therefore incompatible-by-protocol-variance) signature
        # than PointInTimeStateLike's deliberately loose object-typed bars(); the
        # cast asserts what's true at runtime — ReplayState does satisfy the check.
        checked_state = cast(leakage.PointInTimeStateLike, state)

        # Not yet available: available_time = 1000 + 3600 = 4600.
        state.now = 4599.0
        result_before = leakage.check_timestamp_monotonicity(
            checked_state,
            asset_id="BONK",
            timeframe=Timeframe.H1,
            lookback=10,
            interval_seconds=3600.0,
        )
        assert result_before.status == leakage.CheckStatus.PASSED

        state.now = 4600.0
        result_after = leakage.check_timestamp_monotonicity(
            checked_state,
            asset_id="BONK",
            timeframe=Timeframe.H1,
            lookback=10,
            interval_seconds=3600.0,
        )
        assert result_after.status == leakage.CheckStatus.PASSED

    def test_catches_a_state_that_exposes_a_bar_too_early(self) -> None:
        """A hand-built fake state that (incorrectly) returns a bar before its
        available_time. Proves the check fires on a genuine violation rather
        than only ever passing on well-behaved fixtures."""

        class _CheatingState:
            def __init__(self, now: float, candles: tuple[Candle, ...]) -> None:
                self.now = now
                self._candles = candles

            def bars(
                self, asset_id: str, timeframe: object, *, lookback: int
            ) -> tuple[Candle, ...]:
                return self._candles

        candle = Candle(ts=5000.0, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)
        state = _CheatingState(now=5000.0, candles=(candle,))  # available at 8600, not 5000

        result = leakage.check_timestamp_monotonicity(
            state,
            asset_id="BONK",
            timeframe=Timeframe.H1,
            lookback=10,
            interval_seconds=3600.0,
        )
        assert result.status == leakage.CheckStatus.FAILED
        assert result.violations == 1

    def test_catches_an_unclosed_bar(self) -> None:
        class _CheatingState:
            def __init__(self, now: float, candles: tuple[Candle, ...]) -> None:
                self.now = now
                self._candles = candles

            def bars(
                self, asset_id: str, timeframe: object, *, lookback: int
            ) -> tuple[Candle, ...]:
                return self._candles

        candle = Candle(
            ts=0.0, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0, closed=False
        )
        state = _CheatingState(now=100_000.0, candles=(candle,))

        result = leakage.check_timestamp_monotonicity(
            state,
            asset_id="BONK",
            timeframe=Timeframe.H1,
            lookback=10,
            interval_seconds=3600.0,
        )
        assert result.status == leakage.CheckStatus.FAILED
        assert result.violations == 1


# ---------------------------------------------------------------------------
# Report semantics: skipped != clean
# ---------------------------------------------------------------------------


class TestLeakageReport:
    def test_check_all_with_nothing_supplied_is_all_skipped_and_not_clean(self) -> None:
        report = leakage.check_all()
        assert len(report.checks) == 4
        assert all(c.status == leakage.CheckStatus.SKIPPED for c in report.checks)
        assert report.clean is False

    def test_clean_requires_every_check_to_have_run_and_passed(self) -> None:
        series = _price_series(60)
        report = leakage.check_all(
            prefix_checks=[
                leakage.PrefixCheckSpec(
                    name="vol", feature_fn=_vol_feature, series=series, at_index=40
                )
            ],
        )
        # Three of four families were never supplied: still not clean, even
        # though the one check that ran passed.
        assert report.passed
        assert report.skipped
        assert report.clean is False

    def test_clean_true_only_when_all_supplied_checks_pass(self) -> None:
        series = _price_series(60)
        observations, protected, train_interval = TestLabelHorizonPurgeGuard()._scenario()
        state = ReplayState(now=4600.0, publication_delay_seconds=0.0)
        candle = Candle(ts=1000.0, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)
        state.add_bar(candle, asset_id="BONK", timeframe=Timeframe.H1)

        report = leakage.check_all(
            prefix_checks=[
                leakage.PrefixCheckSpec(
                    name="vol", feature_fn=_vol_feature, series=series, at_index=40
                )
            ],
            sentinel_checks=[
                leakage.SentinelCheckSpec(
                    name="vol_sentinel", feature_fn=_vol_feature, series=series, at_index=40
                )
            ],
            timestamp_checks=[
                leakage.TimestampCheckSpec(
                    # See the cast note in TestTimestampMonotonicity above: ReplayState
                    # satisfies PointInTimeStateLike at runtime despite the narrower
                    # concrete bars() signature tripping up structural typing.
                    state=cast(leakage.PointInTimeStateLike, state),
                    asset_id="BONK",
                    timeframe=Timeframe.H1,
                    lookback=10,
                    interval_seconds=3600.0,
                )
            ],
            label_horizon_checks=[
                leakage.LabelHorizonCheckSpec(
                    observations=observations,
                    protected_intervals=protected,
                    train_interval=train_interval,
                    embargo_secs=0.0,
                )
            ],
        )
        assert not report.skipped
        assert not report.failed
        assert report.clean is True


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_prefix_equivalence_rejects_out_of_range_index(self) -> None:
        series = _price_series(10)
        with pytest.raises(ValueError, match="out of range"):
            leakage.check_prefix_equivalence(_vol_feature, series, at_index=100)

    def test_future_sentinel_rejects_out_of_range_index(self) -> None:
        series = _price_series(10)
        with pytest.raises(ValueError, match="out of range"):
            leakage.check_future_sentinel(_vol_feature, series, at_index=-1)
