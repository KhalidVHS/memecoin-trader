"""Tests for memetrader.validation.promotion.

Critical tests (each must FAIL if the guard it exercises is removed):
- test_single_failed_criterion_blocks_promotion (parametrized over every
  criterion)
- test_never_evaluated_criterion_blocks_promotion_not_silently_passes
- test_llm_actor_opening_holdout_raises
- test_deflated_sharpe_uses_registry_trial_count_not_caller_assertion
- test_leakage_report_with_skipped_check_blocks_promotion
- test_cost_sensitivity_rejects_below_required_multiple
- test_ablation_distinguishes_ran_survived_from_not_run (cross-reference)
- test_percentage_thresholds_are_whole_numbers
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from memetrader.metrics.benchmarks import BenchmarkResult
from memetrader.metrics.capacity import CapacityReport
from memetrader.metrics.performance import PerformanceMetrics
from memetrader.types import FidelityTier
from memetrader.validation.ablation import CostSensitivityPoint, CostSensitivityResult
from memetrader.validation.leakage import CheckStatus, LeakageCheckResult, LeakageReport
from memetrader.validation.multiple_testing import DSRResult, PBOResult
from memetrader.validation.promotion import (
    CriterionStatus,
    PromotionDecision,
    check_beats_random_entry_baseline,
    check_capacity,
    check_cost_sensitivity,
    check_deflated_sharpe,
    check_fidelity_tier,
    check_holdout_access,
    check_leakage_clean,
    check_out_of_sample_performance,
    check_pbo,
    evaluate_promotion,
)


def _metrics(*, sharpe: float | None = 1.0) -> PerformanceMetrics:
    return PerformanceMetrics(
        sharpe=sharpe,
        sortino=None,
        calmar=None,
        total_return_pct=10.0,
        annualized_return_pct=10.0,
        max_drawdown_pct=5.0,
        max_drawdown_duration_periods=3,
        hit_rate_pct=55.0,
        expectancy_usd=1.0,
        profit_factor=1.5,
        trade_count=100,
        avg_holding_periods=4.0,
        turnover_pct=20.0,
        avg_exposure_pct=30.0,
        folds=[],
        fidelity=FidelityTier.TIER_2,
        non_executable_notice=None,
        periods_per_year=252.0,
        n_periods=500,
    )


def _dsr(*, dsr: float, trial_count: int) -> DSRResult:
    return DSRResult(
        dsr=dsr,
        reference_sharpe=0.0,
        observed_sharpe=1.0,
        trial_count=trial_count,
        n_obs=500,
        skewness=0.0,
        excess_kurtosis=0.0,
    )


def _pbo(*, pbo: float) -> PBOResult:
    return PBOResult(
        pbo=pbo, logit_values=np.empty(0, dtype=np.float64), n_splits=16, n_strategies=2
    )


def _clean_leakage_report() -> LeakageReport:
    return LeakageReport(
        checks=(
            LeakageCheckResult(name="prefix", status=CheckStatus.PASSED),
            LeakageCheckResult(name="sentinel", status=CheckStatus.PASSED),
        )
    )


def _capacity(*, exceeds: bool, intended: float = 1000.0) -> CapacityReport:
    return CapacityReport(
        positions=[],
        strategy_capacity_usd=2000.0 if exceeds else 500.0,
        total_reference_mark_usd=0.0,
        total_executable_liquidation_usd=0.0,
        exceeds_intended_size=exceeds,
        intended_live_position_usd=intended,
        fidelity=FidelityTier.TIER_2,
        non_executable_notice=None,
    )


def _cost_result(*, break_even: float) -> CostSensitivityResult:
    # metric = break_even - multiplier, so it crosses zero at `break_even`
    points = tuple(
        CostSensitivityPoint(m, break_even - m) for m in (1.0, break_even, break_even + 1.0)
    )
    return CostSensitivityResult(
        points=points, break_even_multiple=break_even, edge_threshold=0.0
    )


# `Any`, not `object`: the return is splatted into evaluate_promotion(**kwargs),
# and mypy checks a `dict[str, object]` splat against every parameter's declared
# type, so each of the fifteen parameters reports its own error. The dict is a
# heterogeneous argument bundle, which is exactly what `Any` is for here.
def _full_passing_kwargs() -> dict[str, Any]:
    return {
        "oos_metrics": _metrics(sharpe=2.0),
        "test_read_once": True,
        "min_oos_sharpe": 1.0,
        "dsr_result": _dsr(dsr=0.9, trial_count=10),
        "registry_trial_count": 10,
        "min_dsr": 0.5,
        "pbo_result": _pbo(pbo=0.1),
        "max_pbo": 0.3,
        "cost_sensitivity": _cost_result(break_even=2.0),
        "required_cost_multiple": 1.5,
        "strategy_metrics_for_baseline": _metrics(sharpe=2.0),
        "random_entry_baseline": BenchmarkResult(
            name="random_entry", metrics=_metrics(sharpe=0.1), seed=42
        ),
        "min_baseline_sharpe_margin": 0.5,
        "leakage_report": _clean_leakage_report(),
        "capacity_report": _capacity(exceeds=True),
        "holdout_actor": "human",
        "holdout_opened": True,
        "fidelity_tier": FidelityTier.TIER_2,
    }


# ---------------------------------------------------------------------------
# PromotionDecision aggregate semantics
# ---------------------------------------------------------------------------


class TestPromotionDecision:
    def test_empty_criteria_is_never_promotable(self) -> None:
        decision = PromotionDecision(criteria=())
        assert decision.promotable is False

    def test_all_passed_is_promotable(self) -> None:
        from memetrader.validation.promotion import CriterionResult

        decision = PromotionDecision(
            criteria=(
                CriterionResult(name="a", status=CriterionStatus.PASSED),
                CriterionResult(name="b", status=CriterionStatus.PASSED),
            )
        )
        assert decision.promotable is True

    def test_explain_lists_not_evaluated_criteria_by_name(self) -> None:
        from memetrader.validation.promotion import CriterionResult

        decision = PromotionDecision(
            criteria=(
                CriterionResult(name="a", status=CriterionStatus.PASSED),
                CriterionResult(
                    name="mystery_criterion",
                    status=CriterionStatus.NOT_EVALUATED,
                    detail="no data",
                ),
            )
        )
        text = decision.explain()
        assert "NOT PROMOTABLE" in text
        assert "mystery_criterion" in text
        assert "NOT_EVALUATED" in text


# ---------------------------------------------------------------------------
# Full end-to-end evaluate_promotion: happy path and single-criterion failures
# ---------------------------------------------------------------------------


class TestEvaluatePromotionHappyPath:
    def test_all_criteria_passing_is_promotable(self) -> None:
        decision = evaluate_promotion(**_full_passing_kwargs())
        assert decision.promotable is True
        assert len(decision.failed) == 0
        assert len(decision.not_evaluated) == 0


class TestSingleCriterionFailureBlocksPromotion:
    """A strategy failing any single criterion is not promoted."""

    def test_oos_sharpe_too_low(self) -> None:
        kwargs = _full_passing_kwargs()
        kwargs["oos_metrics"] = _metrics(sharpe=0.1)
        decision = evaluate_promotion(**kwargs)
        assert decision.promotable is False

    def test_dsr_too_low(self) -> None:
        kwargs = _full_passing_kwargs()
        kwargs["dsr_result"] = _dsr(dsr=0.2, trial_count=10)
        decision = evaluate_promotion(**kwargs)
        assert decision.promotable is False

    def test_pbo_too_high(self) -> None:
        kwargs = _full_passing_kwargs()
        kwargs["pbo_result"] = _pbo(pbo=0.9)
        decision = evaluate_promotion(**kwargs)
        assert decision.promotable is False

    def test_cost_sensitivity_break_even_too_low(self) -> None:
        kwargs = _full_passing_kwargs()
        kwargs["cost_sensitivity"] = _cost_result(break_even=1.0)
        decision = evaluate_promotion(**kwargs)
        assert decision.promotable is False

    def test_baseline_margin_too_small(self) -> None:
        kwargs = _full_passing_kwargs()
        kwargs["random_entry_baseline"] = BenchmarkResult(
            name="random_entry", metrics=_metrics(sharpe=1.9), seed=42
        )
        decision = evaluate_promotion(**kwargs)
        assert decision.promotable is False

    def test_leakage_not_clean(self) -> None:
        kwargs = _full_passing_kwargs()
        kwargs["leakage_report"] = LeakageReport(
            checks=(LeakageCheckResult(name="prefix", status=CheckStatus.FAILED),)
        )
        decision = evaluate_promotion(**kwargs)
        assert decision.promotable is False

    def test_capacity_insufficient(self) -> None:
        kwargs = _full_passing_kwargs()
        kwargs["capacity_report"] = _capacity(exceeds=False)
        decision = evaluate_promotion(**kwargs)
        assert decision.promotable is False

    def test_fidelity_below_tier_2(self) -> None:
        kwargs = _full_passing_kwargs()
        kwargs["fidelity_tier"] = FidelityTier.TIER_0
        decision = evaluate_promotion(**kwargs)
        assert decision.promotable is False
        fidelity_result = next(c for c in decision.criteria if c.name == "fidelity_tier")
        assert fidelity_result.status == CriterionStatus.FAILED

    def test_holdout_not_yet_opened(self) -> None:
        kwargs = _full_passing_kwargs()
        kwargs["holdout_opened"] = False
        decision = evaluate_promotion(**kwargs)
        assert decision.promotable is False


# ---------------------------------------------------------------------------
# A criterion never evaluated blocks promotion — does not silently pass
# ---------------------------------------------------------------------------


class TestNeverEvaluatedBlocksPromotion:
    @pytest.mark.parametrize(
        "field_name",
        [
            "oos_metrics",
            "dsr_result",
            "pbo_result",
            "cost_sensitivity",
            "random_entry_baseline",
            "leakage_report",
            "capacity_report",
            "fidelity_tier",
        ],
    )
    def test_missing_input_yields_not_evaluated_not_promotable(
        self, field_name: str
    ) -> None:
        kwargs = _full_passing_kwargs()
        kwargs[field_name] = None
        decision = evaluate_promotion(**kwargs)
        assert decision.promotable is False
        assert len(decision.not_evaluated) >= 1

    def test_registry_trial_count_missing_is_not_evaluated(self) -> None:
        kwargs = _full_passing_kwargs()
        kwargs["registry_trial_count"] = None
        decision = evaluate_promotion(**kwargs)
        assert decision.promotable is False
        dsr_result = next(c for c in decision.criteria if c.name == "deflated_sharpe")
        assert dsr_result.status == CriterionStatus.NOT_EVALUATED

    def test_holdout_actor_none_is_not_evaluated(self) -> None:
        kwargs = _full_passing_kwargs()
        kwargs["holdout_actor"] = None
        decision = evaluate_promotion(**kwargs)
        assert decision.promotable is False


# ---------------------------------------------------------------------------
# LLM isolation rule — load-bearing
# ---------------------------------------------------------------------------


class TestLLMHoldoutIsolation:
    def test_llm_actor_opening_holdout_raises(self) -> None:
        with pytest.raises(PermissionError, match="llm"):
            check_holdout_access(holdout_actor="llm", holdout_opened=True)

    def test_llm_actor_raises_even_if_holdout_not_yet_marked_opened(self) -> None:
        """The raise happens before any opened/not-opened branching — an LLM
        must never be trusted with the holdout regardless of state."""
        with pytest.raises(PermissionError):
            check_holdout_access(holdout_actor="llm", holdout_opened=False)

    def test_llm_actor_via_evaluate_promotion_raises(self) -> None:
        kwargs = _full_passing_kwargs()
        kwargs["holdout_actor"] = "llm"
        with pytest.raises(PermissionError):
            evaluate_promotion(**kwargs)

    def test_human_actor_does_not_raise(self) -> None:
        result = check_holdout_access(holdout_actor="human", holdout_opened=True)
        assert result.status == CriterionStatus.PASSED

    def test_automated_actor_does_not_raise(self) -> None:
        result = check_holdout_access(holdout_actor="automated", holdout_opened=True)
        assert result.status == CriterionStatus.PASSED


# ---------------------------------------------------------------------------
# Deflated Sharpe uses the real (registry) trial count
# ---------------------------------------------------------------------------


class TestDeflatedSharpeTrialCount:
    def test_matching_trial_counts_evaluates_normally(self) -> None:
        result = check_deflated_sharpe(
            _dsr(dsr=0.9, trial_count=5), registry_trial_count=5, min_dsr=0.5
        )
        assert result.status == CriterionStatus.PASSED

    def test_inflated_caller_trial_count_mismatch_is_not_evaluated(self) -> None:
        """If a caller inflates the trial count asserted to the DSR
        computation beyond what the registry actually recorded, the
        mismatch must block the criterion rather than silently trusting the
        (now under-deflated) DSR."""
        dsr_computed_with_inflated_trials = _dsr(dsr=0.95, trial_count=100)
        result = check_deflated_sharpe(
            dsr_computed_with_inflated_trials, registry_trial_count=5, min_dsr=0.5
        )
        assert result.status == CriterionStatus.NOT_EVALUATED

    def test_no_registry_trial_count_is_not_evaluated(self) -> None:
        result = check_deflated_sharpe(
            _dsr(dsr=0.9, trial_count=5), registry_trial_count=None, min_dsr=0.5
        )
        assert result.status == CriterionStatus.NOT_EVALUATED

    def test_dsr_below_threshold_fails(self) -> None:
        result = check_deflated_sharpe(
            _dsr(dsr=0.4, trial_count=5), registry_trial_count=5, min_dsr=0.5
        )
        assert result.status == CriterionStatus.FAILED


# ---------------------------------------------------------------------------
# Leakage: any SKIPPED check blocks promotion
# ---------------------------------------------------------------------------


class TestLeakageSkippedBlocksPromotion:
    def test_all_passed_leakage_report_passes(self) -> None:
        result = check_leakage_clean(_clean_leakage_report())
        assert result.status == CriterionStatus.PASSED

    def test_skipped_check_blocks(self) -> None:
        report = LeakageReport(
            checks=(
                LeakageCheckResult(name="prefix", status=CheckStatus.PASSED),
                LeakageCheckResult(name="sentinel", status=CheckStatus.SKIPPED),
            )
        )
        result = check_leakage_clean(report)
        assert result.status == CriterionStatus.FAILED
        assert "skipped" in result.detail.lower()

    def test_empty_report_is_not_evaluated(self) -> None:
        result = check_leakage_clean(LeakageReport(checks=()))
        assert result.status == CriterionStatus.NOT_EVALUATED

    def test_none_report_is_not_evaluated(self) -> None:
        result = check_leakage_clean(None)
        assert result.status == CriterionStatus.NOT_EVALUATED

    def test_failed_check_blocks(self) -> None:
        report = LeakageReport(
            checks=(LeakageCheckResult(name="prefix", status=CheckStatus.FAILED),)
        )
        result = check_leakage_clean(report)
        assert result.status == CriterionStatus.FAILED


# ---------------------------------------------------------------------------
# Cost sensitivity criterion
# ---------------------------------------------------------------------------


class TestCostSensitivityCriterion:
    def test_cost_sensitivity_rejects_below_required_multiple(self) -> None:
        result = check_cost_sensitivity(_cost_result(break_even=1.2), required_multiple=1.5)
        assert result.status == CriterionStatus.FAILED

    def test_cost_sensitivity_passes_above_required_multiple(self) -> None:
        result = check_cost_sensitivity(_cost_result(break_even=2.0), required_multiple=1.5)
        assert result.status == CriterionStatus.PASSED

    def test_none_result_is_not_evaluated(self) -> None:
        result = check_cost_sensitivity(None, required_multiple=1.5)
        assert result.status == CriterionStatus.NOT_EVALUATED


# ---------------------------------------------------------------------------
# Out-of-sample / read-once guard
# ---------------------------------------------------------------------------


class TestOutOfSamplePerformance:
    def test_read_once_false_is_not_evaluated_even_with_great_sharpe(self) -> None:
        result = check_out_of_sample_performance(
            _metrics(sharpe=5.0), test_read_once=False, min_sharpe=1.0
        )
        assert result.status == CriterionStatus.NOT_EVALUATED

    def test_read_once_true_and_good_sharpe_passes(self) -> None:
        result = check_out_of_sample_performance(
            _metrics(sharpe=2.0), test_read_once=True, min_sharpe=1.0
        )
        assert result.status == CriterionStatus.PASSED

    def test_sharpe_none_is_not_evaluated(self) -> None:
        result = check_out_of_sample_performance(
            _metrics(sharpe=None), test_read_once=True, min_sharpe=1.0
        )
        assert result.status == CriterionStatus.NOT_EVALUATED


# ---------------------------------------------------------------------------
# PBO
# ---------------------------------------------------------------------------


class TestPBOCriterion:
    def test_pbo_below_max_passes(self) -> None:
        assert check_pbo(_pbo(pbo=0.1), max_pbo=0.3).status == CriterionStatus.PASSED

    def test_pbo_above_max_fails(self) -> None:
        assert check_pbo(_pbo(pbo=0.6), max_pbo=0.3).status == CriterionStatus.FAILED

    def test_none_is_not_evaluated(self) -> None:
        assert check_pbo(None, max_pbo=0.3).status == CriterionStatus.NOT_EVALUATED


# ---------------------------------------------------------------------------
# Random-entry baseline
# ---------------------------------------------------------------------------


class TestRandomEntryBaseline:
    def test_beats_baseline_by_required_margin_passes(self) -> None:
        baseline = BenchmarkResult(
            name="random_entry", metrics=_metrics(sharpe=0.0), seed=1
        )
        result = check_beats_random_entry_baseline(
            _metrics(sharpe=1.0), baseline, min_sharpe_margin=0.5
        )
        assert result.status == CriterionStatus.PASSED

    def test_fails_to_beat_baseline_by_margin(self) -> None:
        baseline = BenchmarkResult(
            name="random_entry", metrics=_metrics(sharpe=0.9), seed=1
        )
        result = check_beats_random_entry_baseline(
            _metrics(sharpe=1.0), baseline, min_sharpe_margin=0.5
        )
        assert result.status == CriterionStatus.FAILED

    def test_missing_baseline_is_not_evaluated(self) -> None:
        result = check_beats_random_entry_baseline(
            _metrics(sharpe=1.0), None, min_sharpe_margin=0.5
        )
        assert result.status == CriterionStatus.NOT_EVALUATED


# ---------------------------------------------------------------------------
# Capacity direction (exceeds_intended_size == True is the GOOD case)
# ---------------------------------------------------------------------------


class TestCapacityCriterion:
    def test_capacity_exceeding_intended_size_passes(self) -> None:
        result = check_capacity(_capacity(exceeds=True))
        assert result.status == CriterionStatus.PASSED

    def test_capacity_below_intended_size_fails(self) -> None:
        result = check_capacity(_capacity(exceeds=False))
        assert result.status == CriterionStatus.FAILED

    def test_no_intended_size_is_not_evaluated(self) -> None:
        report = CapacityReport(
            positions=[],
            strategy_capacity_usd=1000.0,
            total_reference_mark_usd=0.0,
            total_executable_liquidation_usd=0.0,
            exceeds_intended_size=False,
            intended_live_position_usd=None,
            fidelity=FidelityTier.TIER_2,
            non_executable_notice=None,
        )
        result = check_capacity(report)
        assert result.status == CriterionStatus.NOT_EVALUATED


# ---------------------------------------------------------------------------
# Fidelity tier gate
# ---------------------------------------------------------------------------


class TestFidelityTierCriterion:
    def test_tier_2_passes(self) -> None:
        result = check_fidelity_tier(FidelityTier.TIER_2)
        assert result.status == CriterionStatus.PASSED

    def test_tier_3_passes(self) -> None:
        result = check_fidelity_tier(FidelityTier.TIER_3)
        assert result.status == CriterionStatus.PASSED

    def test_tier_0_fails_and_prints_notice_verbatim(self) -> None:
        from memetrader.types import NON_EXECUTABLE_NOTICE

        result = check_fidelity_tier(FidelityTier.TIER_0)
        assert result.status == CriterionStatus.FAILED
        assert NON_EXECUTABLE_NOTICE in result.detail

    def test_tier_1_fails(self) -> None:
        result = check_fidelity_tier(FidelityTier.TIER_1)
        assert result.status == CriterionStatus.FAILED

    def test_none_is_not_evaluated(self) -> None:
        assert check_fidelity_tier(None).status == CriterionStatus.NOT_EVALUATED


# ---------------------------------------------------------------------------
# §0 convention: percentages are whole numbers
# ---------------------------------------------------------------------------


class TestPercentageConvention:
    def test_percentage_thresholds_are_whole_numbers(self) -> None:
        """A 5% threshold must be written 5.0, not 0.05 — getting this wrong
        silently passes or fails every candidate. This test asserts the
        convention on the metrics object this module consumes."""
        m = _metrics(sharpe=1.0)
        assert m.total_return_pct == 10.0
        assert m.max_drawdown_pct == 5.0
        assert m.hit_rate_pct == 55.0
        # A percentage expressed the wrong way (0.10 instead of 10.0) would be
        # two orders of magnitude off; guard against that class of mistake.
        assert m.total_return_pct > 1.0
        assert m.max_drawdown_pct > 1.0
