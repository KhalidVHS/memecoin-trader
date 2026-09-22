"""Tests for memetrader.validation.ablation.

Critical tests (each must FAIL if the guard it exercises is removed):
- test_no_features_supplied_is_not_run_not_survived (§0 None-vs-0 convention
  applied at the whole-ablation-category level)
- test_all_features_droppable_without_harm_fails
- test_break_even_multiple_below_required_fails
- test_worst_regime_drives_the_verdict_not_the_best
- test_cliff_perturbation_is_flagged
- test_seed_spread_above_threshold_fails
- test_ablation_report_robust_requires_nonempty_and_all_survived
"""

from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from memetrader.validation.ablation import (
    AblationCheckResult,
    AblationReport,
    AblationStatus,
    RegimeSliceResult,
    cost_sensitivity_check,
    feature_ablation_check,
    parameter_perturbation_check,
    regime_slicing_check,
    run_cost_sensitivity,
    run_feature_ablation,
    run_parameter_perturbation,
    run_regime_slicing,
    run_seed_sensitivity,
    seed_sensitivity_check,
)

# ---------------------------------------------------------------------------
# AblationReport / AblationCheckResult basics
# ---------------------------------------------------------------------------


class TestAblationReport:
    def test_empty_report_is_not_robust(self) -> None:
        report = AblationReport(checks=())
        assert report.robust is False

    def test_report_with_a_not_run_check_is_not_robust(self) -> None:
        report = AblationReport(
            checks=(
                AblationCheckResult(name="a", status=AblationStatus.SURVIVED),
                AblationCheckResult(name="b", status=AblationStatus.NOT_RUN),
            )
        )
        assert report.robust is False
        assert len(report.not_run) == 1

    def test_report_all_survived_is_robust(self) -> None:
        report = AblationReport(
            checks=(
                AblationCheckResult(name="a", status=AblationStatus.SURVIVED),
                AblationCheckResult(name="b", status=AblationStatus.SURVIVED),
            )
        )
        assert report.robust is True

    def test_as_dict_counts_match(self) -> None:
        report = AblationReport(
            checks=(
                AblationCheckResult(name="a", status=AblationStatus.SURVIVED),
                AblationCheckResult(name="b", status=AblationStatus.FAILED),
                AblationCheckResult(name="c", status=AblationStatus.NOT_RUN),
            )
        )
        d = report.as_dict()
        assert d["robust"] is False
        assert d["n_survived"] == 1
        assert d["n_failed"] == 1
        assert d["n_not_run"] == 1

    def test_check_result_ok_only_true_when_survived(self) -> None:
        assert AblationCheckResult(name="x", status=AblationStatus.SURVIVED).ok is True
        assert AblationCheckResult(name="x", status=AblationStatus.FAILED).ok is False
        assert AblationCheckResult(name="x", status=AblationStatus.NOT_RUN).ok is False


# ---------------------------------------------------------------------------
# Feature ablation
# ---------------------------------------------------------------------------


class TestFeatureAblation:
    def test_no_features_supplied_is_not_run_not_survived(self) -> None:
        """A missing ablation must never read as a passed one."""
        result = feature_ablation_check([])
        assert result.status == AblationStatus.NOT_RUN
        assert result.ok is False

    def test_load_bearing_feature_survives(self) -> None:
        def evaluate(dropped: frozenset[str]) -> float | None:
            if "momentum" in dropped:
                return 0.1  # dropping the real driver hurts a lot
            return 1.0

        results = run_feature_ablation(
            ["momentum", "volume"],
            evaluate,
            baseline_metric=1.0,
            degradation_threshold=0.05,
        )
        check = feature_ablation_check(results)
        assert check.status == AblationStatus.SURVIVED

    def test_all_features_droppable_without_harm_fails(self) -> None:
        """A strategy whose edge survives dropping every feature never had a
        feature-driven edge — this must FAIL, not SURVIVE."""

        def evaluate(dropped: frozenset[str]) -> float | None:
            return 1.0  # unchanged no matter what is dropped

        results = run_feature_ablation(
            ["momentum", "volume", "onchain"],
            evaluate,
            baseline_metric=1.0,
            degradation_threshold=0.05,
        )
        check = feature_ablation_check(results)
        assert check.status == AblationStatus.FAILED

    def test_evaluate_fn_returning_none_marks_not_evaluable(self) -> None:
        results = run_feature_ablation(
            ["momentum"],
            lambda dropped: None,
            baseline_metric=1.0,
            degradation_threshold=0.05,
        )
        assert results[0].edge_depends_on_feature is None
        assert results[0].ablated_metric is None
        check = feature_ablation_check(results)
        assert check.status == AblationStatus.NOT_RUN


# ---------------------------------------------------------------------------
# Cost sensitivity
# ---------------------------------------------------------------------------


class TestCostSensitivity:
    def test_no_run_is_not_run(self) -> None:
        check = cost_sensitivity_check(None, required_multiple=1.5)
        assert check.status == AblationStatus.NOT_RUN

    def test_break_even_interpolated_correctly(self) -> None:
        # metric = 2.0 - multiplier: crosses zero at multiplier == 2.0
        def evaluate(multiplier: float) -> float:
            return 2.0 - multiplier

        result = run_cost_sensitivity([1.0, 1.5, 2.0, 2.5, 3.0], evaluate)
        assert result.break_even_multiple is not None
        assert math.isclose(result.break_even_multiple, 2.0, abs_tol=1e-6)

    def test_break_even_multiple_below_required_fails(self) -> None:
        """A strategy dying at 1.2x assumed costs is not deployable."""

        def evaluate(multiplier: float) -> float:
            return 2.0 - multiplier  # breaks even at 2.0x

        result = run_cost_sensitivity([1.0, 1.2, 1.5, 2.0, 2.5], evaluate)
        check = cost_sensitivity_check(result, required_multiple=1.2)
        # break-even (2.0x) is ABOVE the required 1.2x -> should survive
        assert check.status == AblationStatus.SURVIVED

        check_strict = cost_sensitivity_check(result, required_multiple=2.5)
        assert check_strict.status == AblationStatus.FAILED

    def test_edge_survives_entire_tested_range_but_below_required_is_not_run(self) -> None:
        def evaluate(multiplier: float) -> float:
            return 10.0 - multiplier  # never crosses zero in tested range

        result = run_cost_sensitivity([1.0, 1.2], evaluate)
        check = cost_sensitivity_check(result, required_multiple=5.0)
        assert check.status == AblationStatus.NOT_RUN

    def test_edge_survives_entire_range_and_range_covers_required_multiple(self) -> None:
        def evaluate(multiplier: float) -> float:
            return 10.0 - multiplier

        result = run_cost_sensitivity([1.0, 2.0, 3.0], evaluate)
        check = cost_sensitivity_check(result, required_multiple=2.0)
        assert check.status == AblationStatus.SURVIVED

    def test_edge_negative_at_smallest_multiple_fails(self) -> None:
        def evaluate(multiplier: float) -> float:
            return -1.0 - multiplier

        result = run_cost_sensitivity([1.0, 2.0], evaluate)
        check = cost_sensitivity_check(result, required_multiple=1.0)
        assert check.status == AblationStatus.FAILED

    @given(
        multipliers=st.lists(
            st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=8,
            unique=True,
        )
    )
    def test_break_even_is_none_or_within_tested_bounds(
        self, multipliers: list[float]
    ) -> None:
        def evaluate(multiplier: float) -> float:
            return 1.0 - 0.3 * multiplier

        result = run_cost_sensitivity(multipliers, evaluate)
        if result.break_even_multiple is not None:
            assert min(multipliers) <= result.break_even_multiple <= max(multipliers)


# ---------------------------------------------------------------------------
# Regime slicing
# ---------------------------------------------------------------------------


class TestRegimeSlicing:
    def test_no_regimes_is_not_run(self) -> None:
        check = regime_slicing_check([], min_metric=0.0)
        assert check.status == AblationStatus.NOT_RUN

    def test_worst_regime_drives_the_verdict_not_the_best(self) -> None:
        results = [
            RegimeSliceResult(regime_label="low_vol", metric=2.0, n_obs=100),
            RegimeSliceResult(regime_label="high_vol", metric=-1.0, n_obs=100),
        ]
        check = regime_slicing_check(results, min_metric=0.0)
        assert check.status == AblationStatus.FAILED
        assert "high_vol" in check.detail

    def test_all_regimes_clear_threshold_survives(self) -> None:
        results = [
            RegimeSliceResult(regime_label="jan", metric=1.0, n_obs=50),
            RegimeSliceResult(regime_label="feb", metric=1.5, n_obs=50),
        ]
        check = regime_slicing_check(results, min_metric=0.5)
        assert check.status == AblationStatus.SURVIVED

    def test_regimes_with_zero_observations_are_ignored_not_counted_as_zero(self) -> None:
        results = [
            RegimeSliceResult(regime_label="empty", metric=None, n_obs=0),
            RegimeSliceResult(regime_label="real", metric=1.0, n_obs=10),
        ]
        check = regime_slicing_check(results, min_metric=0.5)
        assert check.status == AblationStatus.SURVIVED

    def test_run_regime_slicing_calls_evaluate_fn_per_label(self) -> None:
        seen: list[str] = []

        def evaluate(label: str) -> tuple[float | None, int]:
            seen.append(label)
            return (1.0, 10)

        run_regime_slicing(["a", "b", "c"], evaluate)
        assert seen == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Parameter perturbation
# ---------------------------------------------------------------------------


class TestParameterPerturbation:
    def test_no_perturbations_is_not_run(self) -> None:
        check = parameter_perturbation_check([])
        assert check.status == AblationStatus.NOT_RUN

    def test_cliff_perturbation_is_flagged(self) -> None:
        """A cliff means overfitting to a specific value."""

        def evaluate(value: float) -> float:
            return -100.0  # huge drop from any nudge

        results = run_parameter_perturbation(
            "lookback", 20.0, [-0.05, 0.05], evaluate, base_metric=1.0, max_sensitivity=5.0
        )
        check = parameter_perturbation_check(results)
        assert check.status == AblationStatus.FAILED
        assert any(r.is_cliff for r in results)

    def test_smooth_perturbation_survives(self) -> None:
        def evaluate(value: float) -> float:
            return 1.0 - 0.01 * abs(value - 20.0)

        results = run_parameter_perturbation(
            "lookback",
            20.0,
            [-0.05, 0.05, -0.1, 0.1],
            evaluate,
            base_metric=1.0,
            max_sensitivity=1.0,
        )
        check = parameter_perturbation_check(results)
        assert check.status == AblationStatus.SURVIVED

    def test_zero_fraction_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="nonzero"):
            run_parameter_perturbation(
                "x", 1.0, [0.0], lambda v: 1.0, base_metric=1.0, max_sensitivity=1.0
            )

    def test_none_metric_is_not_a_cliff_and_not_counted(self) -> None:
        results = run_parameter_perturbation(
            "x", 1.0, [0.1], lambda v: None, base_metric=1.0, max_sensitivity=1.0
        )
        assert results[0].is_cliff is None
        check = parameter_perturbation_check(results)
        assert check.status == AblationStatus.NOT_RUN


# ---------------------------------------------------------------------------
# Seed sensitivity
# ---------------------------------------------------------------------------


class TestSeedSensitivity:
    def test_no_seeds_is_not_run(self) -> None:
        check = seed_sensitivity_check(None, max_spread=1.0)
        assert check.status == AblationStatus.NOT_RUN

    def test_single_seed_is_not_run(self) -> None:
        result = run_seed_sensitivity([1], lambda s: 1.0)
        check = seed_sensitivity_check(result, max_spread=1.0)
        assert check.status == AblationStatus.NOT_RUN

    def test_seed_spread_above_threshold_fails(self) -> None:
        """A strategy whose seeds disagree wildly is a seed lottery, not a
        strategy."""
        values = {1: 5.0, 2: -5.0, 3: 0.0}
        result = run_seed_sensitivity(list(values), lambda s: values[s])
        assert result.spread == 10.0
        check = seed_sensitivity_check(result, max_spread=1.0)
        assert check.status == AblationStatus.FAILED

    def test_seed_spread_within_threshold_survives(self) -> None:
        values = {1: 1.0, 2: 1.05, 3: 0.98}
        result = run_seed_sensitivity(list(values), lambda s: values[s])
        check = seed_sensitivity_check(result, max_spread=0.5)
        assert check.status == AblationStatus.SURVIVED

    def test_reports_mean_not_just_spread(self) -> None:
        values = {1: 1.0, 2: 3.0}
        result = run_seed_sensitivity(list(values), lambda s: values[s])
        assert result.mean == 2.0
        assert result.spread == 2.0

    @given(
        values=st.lists(
            st.floats(
                min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
            ),
            min_size=2,
            max_size=10,
        )
    )
    def test_spread_is_always_nonnegative(self, values: list[float]) -> None:
        seeds = list(range(len(values)))
        table = dict(zip(seeds, values, strict=True))
        result = run_seed_sensitivity(seeds, lambda s: table[s])
        assert result.spread is not None
        assert result.spread >= 0.0
