"""Tests for memetrader.validation.walk_forward.

Every test is offline and deterministic. Fixtures use the same helpers as
test_splits.py (copied inline to avoid a cross-test-file import dependency).

Critical tests:
- test_test_read_exactly_once        — EvaluationAccessViolation on second read
- test_leakage_clean_after_run       — assert_leakage_clean passes after valid run
- test_zero_test_reads_caught        — access_count == 0 would fail leakage check
- test_phases_called_in_order        — fit before select before refit before test
- test_train_does_not_see_val_test   — train obs times precede val/test times
- test_refit_uses_train_val_combined — refit gets more obs than fit-only train
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import pytest

from memetrader.validation.splits import (
    FoldRecord,
    LabeledObservation,
    PurgedWalkForward,
)
from memetrader.validation.walk_forward import (
    EvaluationAccessViolation,
    FoldResult,
    WalkForwardError,
    WalkForwardRunner,
    WalkForwardRunResult,
    _GuardedTestView,
)

# ---------------------------------------------------------------------------
# Helpers (duplicated from test_splits.py to avoid cross-file import)
# ---------------------------------------------------------------------------

_DAY = 86400.0
_HOUR = 3600.0


def _epoch(day_offset: float, hour: float = 0.0) -> float:
    return day_offset * _DAY + hour * _HOUR


def _make_obs(
    asset_id: str,
    obs_time: float,
    label_horizon_hours: float = 4.0,
) -> LabeledObservation:
    return LabeledObservation(
        asset_id=asset_id,
        obs_time=obs_time,
        label_start_ts=obs_time,
        label_end_ts=obs_time + label_horizon_hours * _HOUR,
    )


def _dense_observations(
    asset_id: str,
    start: float,
    end: float,
    interval_hours: float = 1.0,
    label_horizon_hours: float = 4.0,
) -> list[LabeledObservation]:
    out: list[LabeledObservation] = []
    t = start
    while t < end:
        out.append(_make_obs(asset_id, t, label_horizon_hours))
        t += interval_hours * _HOUR
    return out


# ---------------------------------------------------------------------------
# Minimal model stubs for the four-phase protocol
# ---------------------------------------------------------------------------


@dataclass
class _CallLog:
    """Records which phases were called, in order, and with what data sizes."""

    events: list[str] = field(default_factory=list)
    fit_n_train: list[int] = field(default_factory=list)
    select_n_val: list[int] = field(default_factory=list)
    refit_n_train_val: list[int] = field(default_factory=list)
    test_n_test: list[int] = field(default_factory=list)


def _make_runner(
    call_log: _CallLog,
    splitter: PurgedWalkForward | None = None,
    double_read_test: bool = False,
    skip_test_read: bool = False,
) -> WalkForwardRunner:
    """Build a WalkForwardRunner with instrumented phase callables.

    ``double_read_test``: test_fn reads the test view twice (leakage probe).
    ``skip_test_read``: test_fn never reads the test view (skipped evaluation probe).
    """

    def fit_fn(train_obs: list[LabeledObservation], fold: FoldRecord) -> dict[str, Any]:
        call_log.events.append("fit")
        call_log.fit_n_train.append(len(train_obs))
        return {"phase": "fit", "n": len(train_obs)}

    def select_fn(
        fitted_model: Any, val_obs: list[LabeledObservation], fold: FoldRecord
    ) -> dict[str, Any]:
        call_log.events.append("select")
        call_log.select_n_val.append(len(val_obs))
        return {"phase": "select", "threshold": 0.5}

    def refit_fn(
        selected_config: Any, train_val_obs: list[LabeledObservation], fold: FoldRecord
    ) -> dict[str, Any]:
        call_log.events.append("refit")
        call_log.refit_n_train_val.append(len(train_val_obs))
        return {"phase": "refit", "n": len(train_val_obs)}

    if double_read_test:

        def test_fn(
            refitted_model: Any, test_view: _GuardedTestView, fold: FoldRecord
        ) -> dict[str, Any]:
            call_log.events.append("test")
            obs = test_view.read()  # first read
            call_log.test_n_test.append(len(obs))
            obs2 = test_view.read()  # second read — must raise
            return {"phase": "test", "n": len(obs2)}

    elif skip_test_read:

        def test_fn(
            refitted_model: Any, test_view: _GuardedTestView, fold: FoldRecord
        ) -> dict[str, Any]:
            call_log.events.append("test")
            # Deliberately does not call test_view.read().
            return {"phase": "test", "n": 0}

    else:

        def test_fn(
            refitted_model: Any, test_view: _GuardedTestView, fold: FoldRecord
        ) -> dict[str, Any]:
            call_log.events.append("test")
            obs = test_view.read()
            call_log.test_n_test.append(len(obs))
            return {"phase": "test", "n": len(obs)}

    if splitter is None:
        splitter = PurgedWalkForward(
            train_days=120.0,
            val_days=20.0,
            test_days=20.0,
            roll_days=20.0,
        )

    return WalkForwardRunner(
        splitter=splitter,
        fit_fn=fit_fn,
        select_fn=select_fn,
        refit_fn=refit_fn,
        test_fn=test_fn,
    )


# ---------------------------------------------------------------------------
# _GuardedTestView unit tests
# ---------------------------------------------------------------------------


class TestGuardedTestView:
    def _make_view(self, n: int = 5) -> _GuardedTestView:
        obs = [_make_obs("BONK", float(i)) for i in range(n)]
        return _GuardedTestView(_observations=obs, _fold_id="fold_000")

    def test_first_read_succeeds(self) -> None:
        view = self._make_view(5)
        result = view.read()
        assert len(result) == 5
        assert view.access_count == 1

    def test_second_read_raises(self) -> None:
        """GUARD TEST: second read must raise EvaluationAccessViolation."""
        view = self._make_view(5)
        view.read()
        with pytest.raises(EvaluationAccessViolation, match="already read"):
            view.read()

    def test_access_count_increments(self) -> None:
        view = self._make_view(3)
        assert view.access_count == 0
        view.read()
        assert view.access_count == 1

    def test_fold_id_in_error_message(self) -> None:
        view = _GuardedTestView(_observations=[], _fold_id="my_fold_007")
        view.read()
        with pytest.raises(EvaluationAccessViolation, match="my_fold_007"):
            view.read()


# ---------------------------------------------------------------------------
# WalkForwardRunner — normal operation
# ---------------------------------------------------------------------------


class TestWalkForwardRunnerNormal:
    def _build(self) -> tuple[list[LabeledObservation], float, float]:
        data_start = 0.0
        data_end = 209.0 * _DAY
        obs = _dense_observations("BONK", data_start, data_end, 1.0, 4.0)
        return obs, data_start, data_end

    def test_phases_called_in_order(self) -> None:
        """fit → select → refit → test must be called in that order per fold."""
        obs, data_start, data_end = self._build()
        log = _CallLog()
        runner = _make_runner(log)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = runner.run(obs, data_start=data_start, data_end=data_end)

        n_folds = len(result.fold_results)
        assert n_folds > 0
        # Each fold contributes fit, select, refit, test in that order.
        expected_cycle = ["fit", "select", "refit", "test"] * n_folds
        assert log.events == expected_cycle

    def test_leakage_clean_after_run(self) -> None:
        obs, data_start, data_end = self._build()
        log = _CallLog()
        runner = _make_runner(log)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = runner.run(obs, data_start=data_start, data_end=data_end)

        assert result.leakage_clean
        result.assert_leakage_clean()  # must not raise

    def test_test_read_exactly_once(self) -> None:
        """Every fold's test view must have access_count == 1 after the run."""
        obs, data_start, data_end = self._build()
        log = _CallLog()
        runner = _make_runner(log)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = runner.run(obs, data_start=data_start, data_end=data_end)

        for fr in result.fold_results:
            assert fr.test_view.access_count == 1, (
                f"Fold '{fr.fold.fold_id}': expected access_count=1, "
                f"got {fr.test_view.access_count}"
            )

    def test_refit_uses_train_val_combined(self) -> None:
        """refit_fn sees train+val observations, more than fit_fn alone."""
        obs, data_start, data_end = self._build()
        log = _CallLog()
        runner = _make_runner(log)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            runner.run(obs, data_start=data_start, data_end=data_end)

        for n_fit, n_refit, n_val in zip(
            log.fit_n_train, log.refit_n_train_val, log.select_n_val, strict=True
        ):
            assert n_refit >= n_fit, (
                f"refit saw {n_refit} obs but fit saw {n_fit}; refit must see ≥ fit"
            )
            assert n_refit == n_fit + n_val, (
                f"refit={n_refit} != fit({n_fit}) + val({n_val})"
            )

    def test_train_does_not_see_val_test(self) -> None:
        """Training observations must not have obs_time in the val or test window."""
        obs, data_start, data_end = self._build()
        splitter = PurgedWalkForward(
            train_days=120.0,
            val_days=20.0,
            test_days=20.0,
            roll_days=20.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            splits = splitter.split(obs, data_start=data_start, data_end=data_end)

        for sr in splits:
            val_start = sr.fold.val_start
            test_end = sr.fold.test_end
            for i in sr.train_indices:
                t = obs[i].obs_time
                assert t < val_start or t >= test_end, (
                    f"Training obs at {t:.0f} is inside val/test window "
                    f"[{val_start:.0f}, {test_end:.0f})"
                )

    def test_fold_records_match_results(self) -> None:
        obs, data_start, data_end = self._build()
        log = _CallLog()
        runner = _make_runner(log)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = runner.run(obs, data_start=data_start, data_end=data_end)

        assert len(result.fold_records) == len(result.fold_results)
        for fr, rec in zip(result.fold_results, result.fold_records, strict=True):
            assert fr.fold.fold_id == rec.fold_id


# ---------------------------------------------------------------------------
# Leakage detection tests
# ---------------------------------------------------------------------------


class TestLeakageDetection:
    def _build(self) -> tuple[list[LabeledObservation], float, float]:
        data_start = 0.0
        data_end = 209.0 * _DAY
        obs = _dense_observations("BONK", data_start, data_end, 1.0, 4.0)
        return obs, data_start, data_end

    def test_double_read_raises_during_run(self) -> None:
        """GUARD TEST: test_fn that reads test twice must raise EvaluationAccessViolation.

        This tests the structural invariant: the runner raises immediately
        when test_fn accesses the test view more than once, so a leakage event
        cannot go undetected until the end of the run.
        """
        obs, data_start, data_end = self._build()
        log = _CallLog()
        runner = _make_runner(log, double_read_test=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(EvaluationAccessViolation):
                runner.run(obs, data_start=data_start, data_end=data_end)

    def test_skip_test_read_caught_by_walk_forward_error(self) -> None:
        """GUARD TEST: test_fn that never reads the test view must be caught.

        If test_fn forgets to call test_view.read(), the driver raises
        WalkForwardError (access_count == 0, not 1). An evaluation step that
        produced no metrics silently is as bad as one that leaked.
        """
        obs, data_start, data_end = self._build()
        log = _CallLog()
        runner = _make_runner(log, skip_test_read=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(WalkForwardError, match="0 time"):
                runner.run(obs, data_start=data_start, data_end=data_end)

    def test_assert_leakage_clean_raises_on_contaminated(self) -> None:
        """assert_leakage_clean raises WalkForwardError when access_count != 1.

        This test manually builds a FoldResult with access_count=0 (simulating
        a skipped evaluation) to exercise assert_leakage_clean directly.
        """
        obs = [_make_obs("BONK", float(i)) for i in range(5)]
        view = _GuardedTestView(_observations=obs, _fold_id="fake_fold")
        # Do NOT call view.read() — access_count stays 0.
        from memetrader.validation.splits import FoldRecord

        fake_fold = FoldRecord(
            fold_id="fake_fold",
            splitter_id="test",
            train_start=0.0,
            train_end=1.0,
            val_start=1.0,
            val_end=2.0,
            test_start=2.0,
            test_end=3.0,
            embargo_end=4.0,
        )
        fake_result = FoldResult(
            fold=fake_fold,
            test_view=view,
            fit_output=None,
            select_output=None,
            refit_output=None,
            metrics=None,
            n_train=0,
            n_val=0,
            n_test=0,
        )
        run_result = WalkForwardRunResult(
            fold_results=[fake_result],
            fold_records=[fake_fold],
        )
        assert not run_result.leakage_clean
        with pytest.raises(WalkForwardError, match="0 time"):
            run_result.assert_leakage_clean()


# ---------------------------------------------------------------------------
# Empty-fold warning
# ---------------------------------------------------------------------------


class TestEmptyFoldWarning:
    def test_no_folds_emits_warning(self) -> None:
        """A tiny dataset that cannot fit even one fold should warn, not crash."""
        # 10-day dataset, asking for 180+30+30 = 240-day minimum window.
        data_start = 0.0
        data_end = 10.0 * _DAY
        obs = _dense_observations("BONK", data_start, data_end, 1.0, 4.0)
        log = _CallLog()
        runner = _make_runner(log)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = runner.run(obs, data_start=data_start, data_end=data_end)

        assert result.fold_results == []
        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        # Both the holdout warning and the no-folds warning may fire.
        texts = [str(w.message).lower() for w in user_warns]
        assert any("fold" in t or "no fold" in t for t in texts), (
            f"Expected a warning about no folds, got: {texts}"
        )
