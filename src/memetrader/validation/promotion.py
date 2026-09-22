"""The promotion gate — deciding whether a strategy is allowed near real money.

A promotion gate that fails open is worse than no gate: a caller who sees a
"promoted" decision will act on it, so every path that could produce that
decision without genuine evidence must instead produce a not-evaluated /
failed criterion. This module has one job — combine independently inspectable
criteria into a single aggregate decision that is ``True`` only when every
criterion is ``True`` — and it trusts nothing it can check for itself:

* Deflated Sharpe's trial count is cross-checked against the experiment
  registry's own count (``registry.Registry.trial_count`` /
  ``recursive.trial_count``), not taken on the caller's word — an inflated
  caller-supplied trial count would understate how many trials were run and
  overstate the DSR.
* "Out-of-sample" is tied to ``walk_forward.WalkForwardRunResult.leakage_clean``
  — the guarded read-once view's own verdict — not a caller's assertion that
  they only looked at test data.
* A :class:`~memetrader.validation.leakage.LeakageReport` with any
  ``SKIPPED`` check does not satisfy the leakage criterion, mirroring
  ``LeakageReport.clean`` exactly (a leakage check that never ran is not a
  leakage check that passed).
* The LLM isolation rule (BACKTEST-CONTRACTS.md and
  ``experiments.registry.Registry.open_holdout``: ``actor == "llm"`` may never
  open the holdout) is enforced a second time, independently, inside
  :func:`check_holdout_access` — defense in depth, since ``registry.py`` is
  frozen and this module cannot add new introspection to it. If the registry
  already denied the actor, that is a separate, upstream guarantee; this
  module does not rely on it alone.
* Below ``FidelityTier.TIER_2`` a PnL claim is not supportable at all
  (``types.FidelityTier`` docstring: "``validation.promotion`` refuses to
  promote below TIER_2"), so :func:`check_fidelity_tier` hard-fails and the
  decision explanation prints ``types.NON_EXECUTABLE_NOTICE`` verbatim.

Every numeric threshold below (``min_dsr``, ``max_pbo``, ``required_cost_
multiple``, ...) is a required keyword argument with no default, mirroring the
deliberate no-default convention on ``trial_count`` throughout
``multiple_testing.py``. ``docs/BACKTEST-CONTRACTS.md`` (331 lines, current as
of this writing) does not itemize specific numeric thresholds for DSR, PBO, or
cost multiple anywhere in its text — there is no §10 or §11 in the file, and
its actual §9 is "Run artifacts", not promotion criteria. Rather than invent
numbers the contracts do not state, every threshold is pushed to the caller so
the choice is explicit and visible at the call site instead of silently
baked in here.

Percentage fields follow §0: whole numbers, so a 5% threshold is ``5.0``, not
``0.05``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from memetrader.types import NON_EXECUTABLE_NOTICE, FidelityTier
from memetrader.validation.ablation import (
    AblationStatus,
    CostSensitivityResult,
    cost_sensitivity_check,
)

if TYPE_CHECKING:
    from memetrader.metrics.benchmarks import BenchmarkResult
    from memetrader.metrics.capacity import CapacityReport
    from memetrader.metrics.performance import PerformanceMetrics
    from memetrader.validation.leakage import LeakageReport
    from memetrader.validation.multiple_testing import DSRResult, PBOResult

__all__ = [
    "CriterionResult",
    "CriterionStatus",
    "PromotionDecision",
    "check_beats_random_entry_baseline",
    "check_capacity",
    "check_cost_sensitivity",
    "check_deflated_sharpe",
    "check_fidelity_tier",
    "check_holdout_access",
    "check_leakage_clean",
    "check_out_of_sample_performance",
    "check_pbo",
    "evaluate_promotion",
]


# ---------------------------------------------------------------------------
# Shared status vocabulary
# ---------------------------------------------------------------------------


class CriterionStatus(StrEnum):
    """Outcome of one promotion criterion.

    ``NOT_EVALUATED`` is distinct from ``PASSED`` on purpose: a criterion the
    caller never supplied evidence for must never be read as one that was
    checked and cleared. This is the same three-state shape as
    ``leakage.CheckStatus`` and ``ablation.AblationStatus``.
    """

    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"


@dataclass(frozen=True, slots=True)
class CriterionResult:
    """One promotion criterion's verdict, with the reasoning behind it."""

    name: str
    status: CriterionStatus
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == CriterionStatus.PASSED


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """The aggregate promotion decision: every criterion, individually
    inspectable, with an aggregate that is ``True`` only if every single one
    of them is ``PASSED``.

    An empty ``criteria`` tuple is never promotable — the same "non-empty AND
    all-passed" rule as ``LeakageReport.clean`` and ``AblationReport.robust``.
    """

    criteria: tuple[CriterionResult, ...] = field(default_factory=tuple)

    @property
    def failed(self) -> tuple[CriterionResult, ...]:
        return tuple(c for c in self.criteria if c.status == CriterionStatus.FAILED)

    @property
    def not_evaluated(self) -> tuple[CriterionResult, ...]:
        return tuple(c for c in self.criteria if c.status == CriterionStatus.NOT_EVALUATED)

    @property
    def passed(self) -> tuple[CriterionResult, ...]:
        return tuple(c for c in self.criteria if c.status == CriterionStatus.PASSED)

    @property
    def promotable(self) -> bool:
        """``True`` only when every criterion present is ``PASSED``.

        Zero criteria, any ``FAILED``, or any ``NOT_EVALUATED`` all yield
        ``False`` — a candidate is promotable only if it clears *every*
        criterion, and "we never checked" is not clearing it.
        """
        return bool(self.criteria) and all(
            c.status == CriterionStatus.PASSED for c in self.criteria
        )

    def explain(self) -> str:
        """Human-readable explanation of the decision, including exactly
        which criteria were not evaluated — a promotion gate must say *why*,
        not just emit a boolean."""
        if not self.criteria:
            return "NOT PROMOTABLE: no criteria were evaluated."
        lines = [
            "PROMOTABLE" if self.promotable else "NOT PROMOTABLE",
            f" ({len(self.passed)} passed, {len(self.failed)} failed, "
            f"{len(self.not_evaluated)} not evaluated)",
        ]
        header = "".join(lines) + ":"
        body = [
            f"  - [{c.status.value.upper()}] {c.name}: {c.detail}" for c in self.criteria
        ]
        return "\n".join([header, *body])

    def as_dict(self) -> dict[str, object]:
        return {
            "promotable": self.promotable,
            "criteria": [
                {"name": c.name, "status": c.status.value, "detail": c.detail}
                for c in self.criteria
            ],
            "n_passed": len(self.passed),
            "n_failed": len(self.failed),
            "n_not_evaluated": len(self.not_evaluated),
        }


# ---------------------------------------------------------------------------
# Individual criteria
# ---------------------------------------------------------------------------


def check_out_of_sample_performance(
    metrics: PerformanceMetrics | None,
    *,
    test_read_once: bool,
    min_sharpe: float,
    name: str = "out_of_sample_performance",
) -> CriterionResult:
    """Out-of-sample performance on purged walk-forward folds.

    ``test_read_once`` must come from
    ``walk_forward.WalkForwardRunResult.leakage_clean`` (or an equivalent,
    independently verified guarded-read-once check) — not from a caller's
    unverified claim that they only evaluated on held-out folds. If the guard
    was not confirmed, the criterion is ``NOT_EVALUATED``, never ``PASSED``.
    """
    if metrics is None:
        return CriterionResult(
            name=name, status=CriterionStatus.NOT_EVALUATED, detail="no metrics supplied"
        )
    if not test_read_once:
        return CriterionResult(
            name=name,
            status=CriterionStatus.NOT_EVALUATED,
            detail=(
                "test-fold read-once guard was not confirmed clean "
                "(WalkForwardRunResult.leakage_clean was not True) — cannot certify "
                "these metrics as genuinely out-of-sample"
            ),
        )
    if metrics.sharpe is None:
        return CriterionResult(
            name=name,
            status=CriterionStatus.NOT_EVALUATED,
            detail="PerformanceMetrics.sharpe is None",
        )
    if metrics.sharpe >= min_sharpe:
        return CriterionResult(
            name=name,
            status=CriterionStatus.PASSED,
            detail=f"OOS sharpe {metrics.sharpe:.4f} >= required {min_sharpe:g}",
        )
    return CriterionResult(
        name=name,
        status=CriterionStatus.FAILED,
        detail=f"OOS sharpe {metrics.sharpe:.4f} below required {min_sharpe:g}",
    )


def check_deflated_sharpe(
    dsr_result: DSRResult | None,
    *,
    registry_trial_count: int | None,
    min_dsr: float,
    name: str = "deflated_sharpe",
) -> CriterionResult:
    """Deflated Sharpe must be positive/above threshold after correcting for
    the number of trials ACTUALLY run.

    ``registry_trial_count`` must come from the experiment registry
    (``registry.Registry.trial_count`` / ``recursive.trial_count``), not
    asserted by the caller. It is cross-checked against
    ``dsr_result.trial_count`` (the count the DSR computation itself was
    given); if they disagree, the DSR was computed against the wrong trial
    count and the criterion is ``NOT_EVALUATED`` rather than trusted. Fewer
    trials than the DSR assumed would *inflate* an already-run DSR — this
    also catches that direction, not just deliberate under-counting.
    """
    if dsr_result is None:
        return CriterionResult(
            name=name, status=CriterionStatus.NOT_EVALUATED, detail="no DSR result supplied"
        )
    if registry_trial_count is None:
        return CriterionResult(
            name=name,
            status=CriterionStatus.NOT_EVALUATED,
            detail="no registry-sourced trial count supplied",
        )
    if registry_trial_count != dsr_result.trial_count:
        return CriterionResult(
            name=name,
            status=CriterionStatus.NOT_EVALUATED,
            detail=(
                f"DSR was computed with trial_count={dsr_result.trial_count}, but the "
                f"registry recorded {registry_trial_count} actual trial(s) — recompute "
                "the DSR against the real trial count before this can be evaluated"
            ),
        )
    if dsr_result.dsr >= min_dsr:
        return CriterionResult(
            name=name,
            status=CriterionStatus.PASSED,
            detail=(
                f"DSR {dsr_result.dsr:.4f} >= required {min_dsr:g} "
                f"(trial_count={dsr_result.trial_count})"
            ),
        )
    return CriterionResult(
        name=name,
        status=CriterionStatus.FAILED,
        detail=(
            f"DSR {dsr_result.dsr:.4f} below required {min_dsr:g} "
            f"(trial_count={dsr_result.trial_count})"
        ),
    )


def check_pbo(
    pbo_result: PBOResult | None,
    *,
    max_pbo: float,
    name: str = "probability_of_backtest_overfitting",
) -> CriterionResult:
    """Probability of backtest overfitting must be below ``max_pbo``.

    ``max_pbo`` is a probability in ``[0, 1]``, matching
    ``multiple_testing.PBOResult.pbo``'s own scale (this is not a §0
    "percentage" field, it is already a probability).
    """
    if pbo_result is None:
        return CriterionResult(
            name=name, status=CriterionStatus.NOT_EVALUATED, detail="no PBO result supplied"
        )
    if pbo_result.pbo <= max_pbo:
        return CriterionResult(
            name=name,
            status=CriterionStatus.PASSED,
            detail=f"PBO {pbo_result.pbo:.4f} <= required max {max_pbo:g}",
        )
    return CriterionResult(
        name=name,
        status=CriterionStatus.FAILED,
        detail=f"PBO {pbo_result.pbo:.4f} exceeds required max {max_pbo:g}",
    )


def check_cost_sensitivity(
    result: CostSensitivityResult | None,
    *,
    required_multiple: float,
    name: str = "cost_sensitivity",
) -> CriterionResult:
    """Survives cost sensitivity at a stated multiple.

    Delegates the actual grading to ``ablation.cost_sensitivity_check`` so
    the two modules agree on exactly what "survives" means, then maps
    ``ablation.AblationStatus`` onto ``CriterionStatus`` (``NOT_RUN`` ->
    ``NOT_EVALUATED``, never onto ``PASSED``).
    """
    ablation_result = cost_sensitivity_check(result, required_multiple=required_multiple)
    status = {
        AblationStatus.SURVIVED: CriterionStatus.PASSED,
        AblationStatus.FAILED: CriterionStatus.FAILED,
        AblationStatus.NOT_RUN: CriterionStatus.NOT_EVALUATED,
    }[ablation_result.status]
    return CriterionResult(name=name, status=status, detail=ablation_result.detail)


def check_beats_random_entry_baseline(
    strategy_metrics: PerformanceMetrics | None,
    baseline: BenchmarkResult | None,
    *,
    min_sharpe_margin: float,
    name: str = "beats_random_entry_baseline",
) -> CriterionResult:
    """Beats a matched random-entry baseline (not just cash) by at least
    ``min_sharpe_margin``.

    ``baseline`` must be a ``benchmarks.BenchmarkResult`` produced by
    ``random_entry_benchmark`` (or an equivalent matched-random construction)
    so the comparison is against something that traded the same market at the
    same size, not the risk-free/no-trade case.
    """
    if strategy_metrics is None or baseline is None:
        return CriterionResult(
            name=name,
            status=CriterionStatus.NOT_EVALUATED,
            detail="strategy metrics or random-entry baseline not supplied",
        )
    if strategy_metrics.sharpe is None or baseline.metrics.sharpe is None:
        return CriterionResult(
            name=name,
            status=CriterionStatus.NOT_EVALUATED,
            detail="sharpe missing on strategy or baseline metrics",
        )
    margin = strategy_metrics.sharpe - baseline.metrics.sharpe
    if margin >= min_sharpe_margin:
        return CriterionResult(
            name=name,
            status=CriterionStatus.PASSED,
            detail=(
                f"strategy sharpe {strategy_metrics.sharpe:.4f} beats random-entry "
                f"baseline '{baseline.name}' sharpe {baseline.metrics.sharpe:.4f} by "
                f"{margin:.4f} (required margin {min_sharpe_margin:g})"
            ),
        )
    return CriterionResult(
        name=name,
        status=CriterionStatus.FAILED,
        detail=(
            f"strategy sharpe {strategy_metrics.sharpe:.4f} vs random-entry baseline "
            f"'{baseline.name}' sharpe {baseline.metrics.sharpe:.4f}: margin {margin:.4f} "
            f"below required {min_sharpe_margin:g}"
        ),
    )


def check_leakage_clean(
    report: LeakageReport | None,
    *,
    name: str = "leakage_clean",
) -> CriterionResult:
    """Leakage checks all passed.

    Delegates to ``LeakageReport.clean``, which is ``False`` if the report is
    empty or if any single check is ``SKIPPED`` — a report with a SKIPPED
    check must NOT satisfy this criterion, so this function never maps a
    SKIPPED-containing report onto ``PASSED``.
    """
    if report is None:
        return CriterionResult(
            name=name,
            status=CriterionStatus.NOT_EVALUATED,
            detail="no leakage report supplied",
        )
    if not report.checks:
        return CriterionResult(
            name=name,
            status=CriterionStatus.NOT_EVALUATED,
            detail="leakage report contains no checks",
        )
    if report.clean:
        return CriterionResult(
            name=name,
            status=CriterionStatus.PASSED,
            detail=f"all {len(report.checks)} leakage check(s) passed",
        )
    skipped = [c.name for c in report.skipped]
    failed = [c.name for c in report.failed]
    detail_parts = []
    if failed:
        detail_parts.append(f"failed: {failed}")
    if skipped:
        detail_parts.append(f"skipped (never run): {skipped}")
    return CriterionResult(
        name=name,
        status=CriterionStatus.FAILED,
        detail="leakage report not clean — " + "; ".join(detail_parts),
    )


def check_capacity(
    report: CapacityReport | None,
    *,
    name: str = "capacity",
) -> CriterionResult:
    """Capacity sufficient at the intended live size.

    ``CapacityReport.exceeds_intended_size`` means the strategy's safe
    capacity *exceeds* the intended live position size — i.e. there is
    headroom. That is the good/sufficient case, not an "over-capacity"
    warning, despite the name; this check must not invert that direction.
    """
    if report is None:
        return CriterionResult(
            name=name,
            status=CriterionStatus.NOT_EVALUATED,
            detail="no capacity report supplied",
        )
    if report.intended_live_position_usd is None:
        return CriterionResult(
            name=name,
            status=CriterionStatus.NOT_EVALUATED,
            detail="capacity report has no intended live position size to compare against",
        )
    if report.exceeds_intended_size:
        return CriterionResult(
            name=name,
            status=CriterionStatus.PASSED,
            detail=(
                f"strategy capacity ${report.strategy_capacity_usd:,.0f} covers intended "
                f"live size ${report.intended_live_position_usd:,.0f}"
            ),
        )
    return CriterionResult(
        name=name,
        status=CriterionStatus.FAILED,
        detail=(
            f"strategy capacity ${report.strategy_capacity_usd:,.0f} is below intended "
            f"live size ${report.intended_live_position_usd:,.0f}"
        ),
    )


def check_holdout_access(
    *,
    holdout_actor: str | None,
    holdout_opened: bool,
    name: str = "holdout_access",
) -> CriterionResult:
    """Holdout opened at most once, by a permitted actor.

    The LLM isolation rule is load-bearing and enforced here a second time,
    independently of ``experiments.registry.Registry.open_holdout``: an
    actor of ``"llm"`` may never open the holdout. This raises immediately —
    it does not degrade to a FAILED criterion — because an LLM-opened
    holdout is not a promotion-time data point to weigh, it is a protocol
    violation that must stop execution.
    """
    if holdout_actor == "llm":
        msg = (
            "actor='llm' may never open the holdout — this is enforced "
            "independently in validation.promotion as well as "
            "experiments.registry.Registry.open_holdout"
        )
        raise PermissionError(msg)
    if holdout_actor is None:
        return CriterionResult(
            name=name,
            status=CriterionStatus.NOT_EVALUATED,
            detail="no holdout actor supplied",
        )
    if not holdout_opened:
        return CriterionResult(
            name=name,
            status=CriterionStatus.NOT_EVALUATED,
            detail="holdout has not been opened yet",
        )
    return CriterionResult(
        name=name,
        status=CriterionStatus.PASSED,
        detail=f"holdout opened once by permitted actor '{holdout_actor}'",
    )


def check_fidelity_tier(
    tier: FidelityTier | None,
    *,
    name: str = "fidelity_tier",
) -> CriterionResult:
    """Hard-fails below TIER_2, per the ``FidelityTier`` docstring in
    ``memetrader.types``: "``validation.promotion`` refuses to promote below
    TIER_2." Below TIER_2 there is no PnL claim to promote, so the failure
    detail prints ``NON_EXECUTABLE_NOTICE`` verbatim rather than paraphrasing
    it.
    """
    if tier is None:
        return CriterionResult(
            name=name,
            status=CriterionStatus.NOT_EVALUATED,
            detail="no fidelity tier supplied",
        )
    if tier.permits_pnl_claim:
        return CriterionResult(
            name=name,
            status=CriterionStatus.PASSED,
            detail=f"fidelity tier {tier.value} permits a PnL claim",
        )
    return CriterionResult(
        name=name,
        status=CriterionStatus.FAILED,
        detail=f"fidelity tier {tier.value} is below TIER_2. {NON_EXECUTABLE_NOTICE}",
    )


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def evaluate_promotion(
    *,
    oos_metrics: PerformanceMetrics | None,
    test_read_once: bool,
    min_oos_sharpe: float,
    dsr_result: DSRResult | None,
    registry_trial_count: int | None,
    min_dsr: float,
    pbo_result: PBOResult | None,
    max_pbo: float,
    cost_sensitivity: CostSensitivityResult | None,
    required_cost_multiple: float,
    strategy_metrics_for_baseline: PerformanceMetrics | None,
    random_entry_baseline: BenchmarkResult | None,
    min_baseline_sharpe_margin: float,
    leakage_report: LeakageReport | None,
    capacity_report: CapacityReport | None,
    holdout_actor: str | None,
    holdout_opened: bool,
    fidelity_tier: FidelityTier | None,
) -> PromotionDecision:
    """Assemble every promotion criterion into one aggregate decision.

    Every threshold argument is required with no default (mirroring
    ``multiple_testing.trial_count``'s no-default convention) because
    ``docs/BACKTEST-CONTRACTS.md`` does not itemize specific numeric
    thresholds for these criteria; the choice is pushed to the caller so it
    is explicit at the call site rather than silently guessed here.

    ``check_holdout_access`` runs first and can raise ``PermissionError`` for
    ``holdout_actor == "llm"`` — that is deliberate: an LLM-opened holdout is
    a protocol violation, not a criterion to record and continue past.
    """
    holdout_criterion = check_holdout_access(
        holdout_actor=holdout_actor, holdout_opened=holdout_opened
    )
    criteria = (
        check_out_of_sample_performance(
            oos_metrics, test_read_once=test_read_once, min_sharpe=min_oos_sharpe
        ),
        check_deflated_sharpe(
            dsr_result, registry_trial_count=registry_trial_count, min_dsr=min_dsr
        ),
        check_pbo(pbo_result, max_pbo=max_pbo),
        check_cost_sensitivity(cost_sensitivity, required_multiple=required_cost_multiple),
        check_beats_random_entry_baseline(
            strategy_metrics_for_baseline,
            random_entry_baseline,
            min_sharpe_margin=min_baseline_sharpe_margin,
        ),
        check_leakage_clean(leakage_report),
        check_capacity(capacity_report),
        holdout_criterion,
        check_fidelity_tier(fidelity_tier),
    )
    return PromotionDecision(criteria=criteria)
