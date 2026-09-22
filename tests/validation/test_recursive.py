"""Tests for memetrader.validation.recursive.

Critical tests (each must FAIL if the guard it exercises is removed):
- test_same_seed_produces_identical_model_state / different seed diverges
- test_refit_never_observes_data_past_its_cutoff
- test_expanding_and_rolling_schedule_cutoffs_are_monotonic_and_bounded (hypothesis)
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from memetrader.backtest.clock import SimulatedClock
from memetrader.histdata.point_in_time import ReplayState
from memetrader.types import Candle, Timeframe
from memetrader.validation.recursive import (
    ExpandingWindowSchedule,
    RecursiveRefitter,
    RefitPoint,
    RollingWindowSchedule,
    hash_model_state,
    state_bound_data_provider,
    trial_count,
)

_HOUR = 3600.0


# ---------------------------------------------------------------------------
# RefitPoint validation
# ---------------------------------------------------------------------------


class TestRefitPoint:
    def test_valid_point_constructs(self) -> None:
        point = RefitPoint(cutoff=100.0, train_start=0.0, train_end=100.0)
        assert point.cutoff == point.train_end

    def test_cutoff_must_equal_train_end(self) -> None:
        with pytest.raises(ValueError, match="cutoff"):
            RefitPoint(cutoff=90.0, train_start=0.0, train_end=100.0)

    def test_train_end_must_exceed_train_start(self) -> None:
        with pytest.raises(ValueError, match="train_end"):
            RefitPoint(cutoff=0.0, train_start=0.0, train_end=0.0)


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


class TestSchedules:
    def test_expanding_window_train_start_always_data_start(self) -> None:
        schedule = ExpandingWindowSchedule(
            data_start=0.0, data_end=1000.0, min_train_secs=200.0, refit_interval_secs=100.0
        ).refit_points()
        assert schedule
        assert all(p.train_start == 0.0 for p in schedule)
        assert schedule[0].cutoff == 200.0

    def test_rolling_window_width_is_constant(self) -> None:
        schedule = RollingWindowSchedule(
            data_start=0.0, data_end=1000.0, window_secs=300.0, refit_interval_secs=100.0
        ).refit_points()
        assert schedule
        for p in schedule:
            assert math.isclose(p.train_end - p.train_start, 300.0)

    @given(
        data_start=st.floats(
            min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
        span=st.floats(
            min_value=100.0, max_value=100_000.0, allow_nan=False, allow_infinity=False
        ),
        min_train=st.floats(
            min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False
        ),
        interval=st.floats(
            min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False
        ),
    )
    def test_expanding_schedule_cutoffs_monotonic_and_bounded(
        self, data_start: float, span: float, min_train: float, interval: float
    ) -> None:
        data_end = data_start + span
        schedule = ExpandingWindowSchedule(
            data_start=data_start,
            data_end=data_end,
            min_train_secs=min_train,
            refit_interval_secs=interval,
        ).refit_points()
        cutoffs = [p.cutoff for p in schedule]
        assert cutoffs == sorted(cutoffs)
        assert len(set(cutoffs)) == len(cutoffs)
        for p in schedule:
            assert p.train_start == data_start
            assert data_start < p.cutoff <= data_end

    @given(
        data_start=st.floats(
            min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
        span=st.floats(
            min_value=100.0, max_value=100_000.0, allow_nan=False, allow_infinity=False
        ),
        window=st.floats(
            min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False
        ),
        interval=st.floats(
            min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False
        ),
    )
    def test_rolling_schedule_cutoffs_monotonic_and_bounded(
        self, data_start: float, span: float, window: float, interval: float
    ) -> None:
        data_end = data_start + span
        schedule = RollingWindowSchedule(
            data_start=data_start,
            data_end=data_end,
            window_secs=window,
            refit_interval_secs=interval,
        ).refit_points()
        cutoffs = [p.cutoff for p in schedule]
        assert cutoffs == sorted(cutoffs)
        assert len(set(cutoffs)) == len(cutoffs)
        for p in schedule:
            assert math.isclose(p.train_end - p.train_start, window)
            assert data_start < p.cutoff <= data_end


# ---------------------------------------------------------------------------
# hash_model_state
# ---------------------------------------------------------------------------


class TestHashModelState:
    def test_identical_structures_hash_identically(self) -> None:
        a = {"w": [1.0, 2.0, 3.0], "n": 3, "tag": "m"}
        b = {"tag": "m", "n": 3, "w": [1.0, 2.0, 3.0]}  # different key order
        assert hash_model_state(a) == hash_model_state(b)

    def test_different_values_hash_differently(self) -> None:
        a = {"w": [1.0, 2.0, 3.0]}
        b = {"w": [1.0, 2.0, 3.1]}
        assert hash_model_state(a) != hash_model_state(b)

    def test_numpy_arrays_supported(self) -> None:
        a = {"weights": np.array([1.0, 2.0, 3.0])}
        b = {"weights": np.array([1.0, 2.0, 3.0])}
        assert hash_model_state(a) == hash_model_state(b)

    def test_unsupported_type_raises(self) -> None:
        class _Unsupported:
            pass

        with pytest.raises(TypeError):
            hash_model_state(_Unsupported())


# ---------------------------------------------------------------------------
# Determinism: same seed -> identical state, different seed -> different state
# ---------------------------------------------------------------------------


def _provider(train_start: float, train_end: float) -> np.ndarray:
    return np.linspace(train_start, train_end, 50)


def _fit(train_data: object, seed: int) -> dict[str, object]:
    arr = np.asarray(train_data, dtype=float)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=5).tolist()
    return {"noise": noise, "mean": float(arr.mean()), "seed_echo": seed}


class TestRefitDeterminism:
    def test_same_run_id_same_seed_identical_state(self) -> None:
        schedule = [RefitPoint(cutoff=100.0, train_start=0.0, train_end=100.0)]

        clock_a = SimulatedClock(run_id="run-x")
        clock_b = SimulatedClock(run_id="run-x")
        results_a = RecursiveRefitter(clock=clock_a).run(
            schedule, data_provider=_provider, fit_fn=_fit
        )
        results_b = RecursiveRefitter(clock=clock_b).run(
            schedule, data_provider=_provider, fit_fn=_fit
        )

        assert results_a[0].record.seed == results_b[0].record.seed
        assert results_a[0].record.model_state_hash == results_b[0].record.model_state_hash
        assert results_a[0].model_state == results_b[0].model_state

    def test_different_run_id_diverges(self) -> None:
        schedule = [RefitPoint(cutoff=100.0, train_start=0.0, train_end=100.0)]

        clock_a = SimulatedClock(run_id="run-x")
        clock_b = SimulatedClock(run_id="run-y")
        results_a = RecursiveRefitter(clock=clock_a).run(
            schedule, data_provider=_provider, fit_fn=_fit
        )
        results_b = RecursiveRefitter(clock=clock_b).run(
            schedule, data_provider=_provider, fit_fn=_fit
        )

        assert results_a[0].record.seed != results_b[0].record.seed
        assert results_a[0].record.model_state_hash != results_b[0].record.model_state_hash

    def test_refit_record_stamps_exact_cutoff(self) -> None:
        schedule = ExpandingWindowSchedule(
            data_start=0.0, data_end=500.0, min_train_secs=100.0, refit_interval_secs=100.0
        ).refit_points()
        clock = SimulatedClock(run_id="stamp-check")
        results = RecursiveRefitter(clock=clock).run(
            schedule, data_provider=_provider, fit_fn=_fit
        )
        for point, result in zip(schedule, results, strict=True):
            assert result.record.cutoff == point.cutoff
            assert result.record.train_start == point.train_start
            assert result.record.train_end == point.train_end

    def test_trial_hyperparameters_and_note_are_carried_and_counted(self) -> None:
        schedule = [RefitPoint(cutoff=100.0, train_start=0.0, train_end=100.0)]
        clock = SimulatedClock(run_id="trial-run")
        results = RecursiveRefitter(clock=clock).run(
            schedule,
            data_provider=_provider,
            fit_fn=_fit,
            hyperparameters={"lr": 0.01},
            is_trial=True,
            trial_note="hyperparameter sweep candidate",
        )
        assert results[0].record.is_trial is True
        assert results[0].record.hyperparameters == {"lr": 0.01}
        assert results[0].record.trial_note == "hyperparameter sweep candidate"
        assert trial_count(results) == 1

        non_trial_results = RecursiveRefitter(clock=SimulatedClock(run_id="non-trial")).run(
            schedule, data_provider=_provider, fit_fn=_fit
        )
        assert trial_count(non_trial_results) == 0


# ---------------------------------------------------------------------------
# A refit never sees data past its cutoff
# ---------------------------------------------------------------------------


class TestRefitNeverSeesFuture:
    def test_manual_provider_that_would_leak_is_caught_by_the_assertion_itself(
        self,
    ) -> None:
        """Sanity check on the test's own guard: a provider that hands back
        data beyond train_end must make the assertion below fail. This proves
        the test would actually catch a real leak, not just always pass."""
        schedule = [RefitPoint(cutoff=100.0, train_start=0.0, train_end=100.0)]

        def leaking_provider(train_start: float, train_end: float) -> np.ndarray:
            return np.array([train_end + 50.0])  # deliberately leaks 50s into the future

        def fit_checking_future(train_data: object, seed: int) -> dict[str, object]:
            arr = np.asarray(train_data, dtype=float)
            return {"max_ts": float(arr.max())}

        clock = SimulatedClock(run_id="leak-check")
        results = RecursiveRefitter(clock=clock).run(
            schedule, data_provider=leaking_provider, fit_fn=fit_checking_future
        )
        # model_state is declared ``object`` on RefitResult since the refitter is
        # generic over fit_fn's return type; here we know it's the dict our own
        # fit_checking_future returned.
        state = cast(dict[str, float], results[0].model_state)
        max_ts = state["max_ts"]
        assert max_ts > results[0].record.cutoff  # confirms the leak is detectable

    def test_state_bound_data_provider_never_hands_back_a_future_bar(self) -> None:
        state = ReplayState(now=0.0, publication_delay_seconds=0.0)
        t = 0.0
        while t < 50_000.0:
            candle = Candle(ts=t, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)
            state.add_bar(candle, asset_id="BONK", timeframe=Timeframe.H1)
            t += _HOUR

        provider = state_bound_data_provider(
            state, asset_id="BONK", timeframe=Timeframe.H1, lookback=10_000
        )
        schedule = ExpandingWindowSchedule(
            data_start=0.0,
            data_end=50_000.0,
            min_train_secs=10_000.0,
            refit_interval_secs=5_000.0,
        ).refit_points()

        def fit_recording_max_ts(train_data: object, seed: int) -> dict[str, object]:
            # fit_fn's contract types train_data as ``object`` (the refitter is
            # generic over the data shape); state_bound_data_provider is known to
            # hand back a tuple[Candle, ...] here.
            candles = cast(tuple[Candle, ...], train_data)
            max_ts = max((c.ts for c in candles), default=-math.inf)
            return {"max_ts": max_ts, "n": len(candles)}

        clock = SimulatedClock(run_id="future-guard")
        results = RecursiveRefitter(clock=clock).run(
            schedule, data_provider=provider, fit_fn=fit_recording_max_ts
        )

        assert results  # schedule produced at least one refit point
        for result in results:
            model_state = cast(dict[str, float], result.model_state)
            max_ts = model_state["max_ts"]
            if max_ts == -math.inf:
                continue  # no bars available yet at this cutoff
            available_time = max_ts + _HOUR  # publication_delay_seconds == 0
            assert available_time <= result.record.cutoff
