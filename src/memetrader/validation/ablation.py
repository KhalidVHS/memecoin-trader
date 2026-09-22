"""Robustness testing — ablation, cost sensitivity, regime slicing, and more.

BACKTEST-CONTRACTS.md's robustness discussion (§7 validation strategy, and the
promotion gate it feeds in ``validation/promotion.py``) requires more than a
single aggregate performance number before a strategy is allowed near real
money. This module runs the perturbations that separate a real edge from a
fitted one:

* **Feature ablation** (:func:`run_feature_ablation`) — drop each feature and
  measure the damage. A strategy whose performance is unchanged by dropping
  every feature never had a feature-driven edge; it had something else
  (survivorship in the sample, a benchmark quirk, noise that happened to
  correlate with the label window used to pick the strategy).

* **Cost sensitivity** (:func:`run_cost_sensitivity`) — sweep a cost
  multiplier and find the **break-even cost multiple**: the point at which
  the edge crosses a stated threshold. A strategy that breaks even at 1.2x
  assumed costs has no margin for slippage estimation error and is not
  deployable, however good it looks at the assumed cost level.

* **Regime slicing** (:func:`run_regime_slicing`) — performance sliced by
  volatility regime, calendar month, or universe subset. A strategy whose
  entire edge lives in one regime is not evidence of a general effect.

* **Parameter perturbation** (:func:`run_parameter_perturbation`) — nudge one
  parameter by a small amount and check the metric moves by a proportionally
  small amount. A cliff (large metric swing from a small nudge) means the
  result is fitted to one specific value, not to a real, smooth effect.

* **Random seed sensitivity** (:func:`run_seed_sensitivity`) — re-run across
  seeds and report the spread (max - min), not just the mean. A strategy
  whose seeds disagree wildly is not a strategy, it is a seed lottery.

The load-bearing convention (§0: ``None`` means "could not find out", not
"looked, and it is quiet") applies here at the level of whole ablations, the
same way ``leakage.CheckStatus.SKIPPED`` applies at the level of whole leakage
checks. :class:`AblationStatus` has three states —
``SURVIVED`` / ``FAILED`` / ``NOT_RUN`` — and ``NOT_RUN`` is returned whenever
an ablation was never supplied inputs to run on. A missing ablation must never
render as a passed one; see :class:`AblationReport.robust`, which mirrors
``LeakageReport.clean`` exactly: ``True`` only when every check present
actually ran and passed.

This module implements no strategy and no feature — every ``evaluate_fn`` is
supplied by the caller, exactly as ``validation.walk_forward`` and
``validation.recursive`` delegate model behaviour to caller-supplied
callables. The module's job is to drive the perturbation and grade the
result, not to know what a "feature" or a "parameter" means for any specific
strategy.
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "AblationCheckResult",
    "AblationReport",
    "AblationStatus",
    "CostSensitivityPoint",
    "CostSensitivityResult",
    "FeatureAblationResult",
    "ParameterPerturbationResult",
    "RegimeSliceResult",
    "SeedSensitivityResult",
    "cost_sensitivity_check",
    "feature_ablation_check",
    "parameter_perturbation_check",
    "regime_slicing_check",
    "run_cost_sensitivity",
    "run_feature_ablation",
    "run_parameter_perturbation",
    "run_regime_slicing",
    "run_seed_sensitivity",
    "seed_sensitivity_check",
]


# ---------------------------------------------------------------------------
# Shared status vocabulary
# ---------------------------------------------------------------------------


class AblationStatus(StrEnum):
    """Outcome of one robustness check.

    ``SURVIVED`` — the check ran and the edge held up under the perturbation.
    ``FAILED``   — the check ran and the edge did not hold up.
    ``NOT_RUN``  — the check was never run (no inputs supplied, or every
    evaluation call returned ``None``). Distinct from ``SURVIVED`` on purpose:
    an ablation that was never run must never be read as one that passed.
    """

    SURVIVED = "survived"
    FAILED = "failed"
    NOT_RUN = "not_run"


@dataclass(frozen=True, slots=True)
class AblationCheckResult:
    """One ablation category's verdict, with enough detail to act on it."""

    name: str
    status: AblationStatus
    detail: str = ""

    @property
    def ok(self) -> bool:
        """True only when the check ran and the edge survived."""
        return self.status == AblationStatus.SURVIVED


@dataclass(frozen=True, slots=True)
class AblationReport:
    """Summary of every robustness category run (or explicitly not run).

    Mirrors ``leakage.LeakageReport`` exactly: ``robust`` is ``True`` only when
    every supplied check ran and survived. A report with zero checks, or any
    check marked ``NOT_RUN``, is not robust.
    """

    checks: tuple[AblationCheckResult, ...] = field(default_factory=tuple)

    @property
    def failed(self) -> tuple[AblationCheckResult, ...]:
        return tuple(c for c in self.checks if c.status == AblationStatus.FAILED)

    @property
    def not_run(self) -> tuple[AblationCheckResult, ...]:
        return tuple(c for c in self.checks if c.status == AblationStatus.NOT_RUN)

    @property
    def survived(self) -> tuple[AblationCheckResult, ...]:
        return tuple(c for c in self.checks if c.status == AblationStatus.SURVIVED)

    @property
    def robust(self) -> bool:
        """True only when every check present ran and survived.

        Zero checks, or any check left ``NOT_RUN``, is not robust — "we did
        not check" must never render as "robust" (§0's None-vs-0 rule applied
        to whole ablation categories rather than individual numeric fields).
        """
        return bool(self.checks) and all(
            c.status == AblationStatus.SURVIVED for c in self.checks
        )

    def as_dict(self) -> dict[str, object]:
        """Flat JSON-serializable representation, mirroring leakage_report.json."""
        return {
            "robust": self.robust,
            "checks": [
                {"name": c.name, "status": c.status.value, "detail": c.detail}
                for c in self.checks
            ],
            "n_survived": len(self.survived),
            "n_failed": len(self.failed),
            "n_not_run": len(self.not_run),
        }


# ---------------------------------------------------------------------------
# 1. Feature ablation
# ---------------------------------------------------------------------------

FeatureEvalFn = Callable[[frozenset[str]], float | None]
"""``(dropped_features) -> metric``. ``metric`` is any scalar where higher is
better (Sharpe, total return %, DSR — caller's choice, consistent across all
calls in one run). ``None`` means the strategy could not be evaluated with
those features dropped (e.g. the remaining feature set is degenerate)."""


@dataclass(frozen=True, slots=True)
class FeatureAblationResult:
    """One feature's ablation outcome.

    ``edge_depends_on_feature`` is ``None`` when the ablated metric could not
    be computed (``evaluate_fn`` returned ``None``) — distinct from ``False``,
    which means the feature was dropped and it *did not matter*.
    """

    feature_name: str
    baseline_metric: float
    ablated_metric: float | None
    delta: float | None
    edge_depends_on_feature: bool | None


def run_feature_ablation(
    feature_names: Sequence[str],
    evaluate_fn: FeatureEvalFn,
    *,
    baseline_metric: float,
    degradation_threshold: float,
) -> list[FeatureAblationResult]:
    """Drop each feature in turn and measure the damage to ``baseline_metric``.

    ``degradation_threshold`` is the minimum drop (``baseline - ablated``)
    required to call a feature load-bearing. It has no default: a threshold
    of 0 would call any nonzero noise "load-bearing", and a caller-chosen
    value forces the choice to be explicit and visible at the call site,
    matching the no-silent-default convention used for ``trial_count``
    throughout ``multiple_testing.py``.
    """
    results: list[FeatureAblationResult] = []
    for name in feature_names:
        ablated = evaluate_fn(frozenset({name}))
        if ablated is None:
            results.append(
                FeatureAblationResult(
                    feature_name=name,
                    baseline_metric=baseline_metric,
                    ablated_metric=None,
                    delta=None,
                    edge_depends_on_feature=None,
                )
            )
            continue
        delta = ablated - baseline_metric
        depends = delta < -degradation_threshold
        results.append(
            FeatureAblationResult(
                feature_name=name,
                baseline_metric=baseline_metric,
                ablated_metric=ablated,
                delta=delta,
                edge_depends_on_feature=depends,
            )
        )
    return results


def feature_ablation_check(
    results: Sequence[FeatureAblationResult],
    *,
    name: str = "feature_ablation",
) -> AblationCheckResult:
    """Grade a batch of :func:`run_feature_ablation` results.

    ``SURVIVED`` means at least one feature is load-bearing (dropping it hurt
    performance) — evidence the edge is genuinely feature-driven. ``FAILED``
    means every evaluated feature could be dropped with no material harm to
    performance: the exact "a strategy whose edge survives dropping every
    feature never had a feature-driven edge" case this check exists to catch.
    """
    if not results:
        return AblationCheckResult(
            name=name, status=AblationStatus.NOT_RUN, detail="no features supplied"
        )
    evaluated = [r for r in results if r.edge_depends_on_feature is not None]
    if not evaluated:
        return AblationCheckResult(
            name=name,
            status=AblationStatus.NOT_RUN,
            detail="every feature ablation failed to produce a metric",
        )
    load_bearing = [r.feature_name for r in evaluated if r.edge_depends_on_feature]
    if load_bearing:
        return AblationCheckResult(
            name=name,
            status=AblationStatus.SURVIVED,
            detail=f"edge depends on: {load_bearing}",
        )
    return AblationCheckResult(
        name=name,
        status=AblationStatus.FAILED,
        detail=(
            f"dropping every one of {len(evaluated)} evaluated feature(s) left "
            "performance unchanged or improved — the edge is not feature-driven"
        ),
    )


# ---------------------------------------------------------------------------
# 2. Cost sensitivity
# ---------------------------------------------------------------------------

CostEvalFn = Callable[[float], float | None]
"""``(cost_multiplier) -> metric``. ``metric`` is any scalar where the sign
matters (e.g. total return %, or Sharpe); ``edge_threshold`` (usually 0.0)
marks the boundary between "has edge" and "does not"."""


@dataclass(frozen=True, slots=True)
class CostSensitivityPoint:
    multiplier: float
    metric: float | None


@dataclass(frozen=True, slots=True)
class CostSensitivityResult:
    """A cost-multiplier sweep and its interpolated break-even point.

    ``break_even_multiple`` is ``None`` when no crossing was observed inside
    the tested range — either because the edge never breaks (in which case
    the caller should not claim a break-even beyond the tested range) or
    because no multiplier produced a usable metric.
    """

    points: tuple[CostSensitivityPoint, ...]
    break_even_multiple: float | None
    edge_threshold: float


def _interpolate_break_even(
    points: Sequence[CostSensitivityPoint], edge_threshold: float
) -> float | None:
    evaluated = sorted(
        (p for p in points if p.metric is not None), key=lambda p: p.multiplier
    )
    if not evaluated:
        return None
    for prev, curr in itertools.pairwise(evaluated):
        assert prev.metric is not None
        assert curr.metric is not None
        if prev.metric >= edge_threshold > curr.metric:
            span = prev.metric - curr.metric
            if span == 0.0:
                return curr.multiplier
            frac = (prev.metric - edge_threshold) / span
            return prev.multiplier + frac * (curr.multiplier - prev.multiplier)
    return None


def run_cost_sensitivity(
    multipliers: Sequence[float],
    evaluate_fn: CostEvalFn,
    *,
    edge_threshold: float = 0.0,
) -> CostSensitivityResult:
    """Sweep ``multipliers`` (applied to fee/spread/slippage/latency jointly
    or however the caller's ``evaluate_fn`` chooses to apply them) and locate
    the break-even cost multiple by linear interpolation between the last
    point at or above ``edge_threshold`` and the first point below it.
    """
    sorted_mults = sorted(multipliers)
    points = tuple(CostSensitivityPoint(m, evaluate_fn(m)) for m in sorted_mults)
    break_even = _interpolate_break_even(points, edge_threshold)
    return CostSensitivityResult(
        points=points, break_even_multiple=break_even, edge_threshold=edge_threshold
    )


def cost_sensitivity_check(
    result: CostSensitivityResult | None,
    *,
    required_multiple: float,
    name: str = "cost_sensitivity",
) -> AblationCheckResult:
    """Grade a :class:`CostSensitivityResult` against a required multiple.

    A strategy whose break-even multiple is below ``required_multiple`` (e.g.
    it dies at 1.2x assumed costs when 1.5x is required) is not deployable —
    it has no margin for slippage-estimation error.
    """
    if result is None or not result.points:
        return AblationCheckResult(
            name=name, status=AblationStatus.NOT_RUN, detail="cost sensitivity not run"
        )
    evaluated = [p for p in result.points if p.metric is not None]
    if not evaluated:
        return AblationCheckResult(
            name=name,
            status=AblationStatus.NOT_RUN,
            detail="no cost multiplier produced a usable metric",
        )
    if result.break_even_multiple is None:
        max_tested = max(p.multiplier for p in evaluated)
        if all((p.metric or 0.0) >= result.edge_threshold for p in evaluated):
            if max_tested >= required_multiple:
                return AblationCheckResult(
                    name=name,
                    status=AblationStatus.SURVIVED,
                    detail=(
                        f"edge held at every tested multiple up to {max_tested:g}x "
                        f"(>= required {required_multiple:g}x)"
                    ),
                )
            return AblationCheckResult(
                name=name,
                status=AblationStatus.NOT_RUN,
                detail=(
                    f"edge held up to the largest tested multiple {max_tested:g}x, but "
                    f"that is below the required {required_multiple:g}x — sweep further "
                    "before certifying"
                ),
            )
        return AblationCheckResult(
            name=name,
            status=AblationStatus.FAILED,
            detail="edge did not survive even the smallest tested cost multiple",
        )
    survived = result.break_even_multiple >= required_multiple
    status = AblationStatus.SURVIVED if survived else AblationStatus.FAILED
    return AblationCheckResult(
        name=name,
        status=status,
        detail=(
            f"break-even at {result.break_even_multiple:.3f}x assumed costs; "
            f"required {required_multiple:g}x"
        ),
    )


# ---------------------------------------------------------------------------
# 3. Regime slicing
# ---------------------------------------------------------------------------

RegimeEvalFn = Callable[[str], "tuple[float | None, int]"]
"""``(regime_label) -> (metric, n_obs)``. ``n_obs`` lets the check tell "this
regime had zero observations" (never evaluated) from "this regime's metric
was computed from data and happens to equal zero"."""


@dataclass(frozen=True, slots=True)
class RegimeSliceResult:
    regime_label: str
    metric: float | None
    n_obs: int


def run_regime_slicing(
    regime_labels: Sequence[str],
    evaluate_fn: RegimeEvalFn,
) -> list[RegimeSliceResult]:
    """Compute the performance metric within each named regime slice.

    ``regime_labels`` is caller-supplied: volatility-regime buckets, calendar
    months, universe subsets — whatever partition the caller wants sliced.
    """
    return [
        RegimeSliceResult(regime_label=label, metric=metric, n_obs=n)
        for label, (metric, n) in ((lbl, evaluate_fn(lbl)) for lbl in regime_labels)
    ]


def regime_slicing_check(
    results: Sequence[RegimeSliceResult],
    *,
    min_metric: float,
    name: str = "regime_slicing",
) -> AblationCheckResult:
    """Grade regime slices: the worst evaluated slice must clear ``min_metric``.

    A strategy whose entire edge lives in one favourable regime should not
    pass on the strength of its best slice — the worst slice is the honest
    read.
    """
    if not results:
        return AblationCheckResult(
            name=name, status=AblationStatus.NOT_RUN, detail="no regimes supplied"
        )
    evaluated = [r for r in results if r.metric is not None and r.n_obs > 0]
    if not evaluated:
        return AblationCheckResult(
            name=name,
            status=AblationStatus.NOT_RUN,
            detail="no regime slice produced a metric from observed data",
        )
    worst = min(evaluated, key=lambda r: r.metric)  # type: ignore[arg-type,return-value]
    assert worst.metric is not None
    if worst.metric >= min_metric:
        return AblationCheckResult(
            name=name,
            status=AblationStatus.SURVIVED,
            detail=(
                f"worst regime '{worst.regime_label}' metric {worst.metric:.4f} "
                f">= {min_metric:g}"
            ),
        )
    return AblationCheckResult(
        name=name,
        status=AblationStatus.FAILED,
        detail=f"regime '{worst.regime_label}' fell to {worst.metric:.4f}, below {min_metric:g}",
    )


# ---------------------------------------------------------------------------
# 4. Parameter perturbation
# ---------------------------------------------------------------------------

ParamEvalFn = Callable[[float], float | None]
"""``(perturbed_value) -> metric``."""


@dataclass(frozen=True, slots=True)
class ParameterPerturbationResult:
    """One perturbation's effect on the metric.

    ``sensitivity`` = ``|delta| / |relative_fraction|`` — the metric change
    per unit of relative parameter change. ``is_cliff`` compares this to the
    caller's ``max_sensitivity``; ``None`` when the metric could not be
    computed at the perturbed value.
    """

    parameter_name: str
    base_value: float
    perturbed_value: float
    relative_fraction: float
    base_metric: float
    perturbed_metric: float | None
    delta: float | None
    sensitivity: float | None
    is_cliff: bool | None


def run_parameter_perturbation(
    parameter_name: str,
    base_value: float,
    relative_fractions: Sequence[float],
    evaluate_fn: ParamEvalFn,
    *,
    base_metric: float,
    max_sensitivity: float,
) -> list[ParameterPerturbationResult]:
    """Nudge ``base_value`` by each fraction in ``relative_fractions`` and
    grade whether the metric moved smoothly.

    ``relative_fractions`` are small fractional nudges, e.g. ``[-0.1, -0.05,
    0.05, 0.1]``. ``max_sensitivity`` is the maximum tolerated
    ``|delta_metric| / |fraction|`` before a perturbation is flagged as a
    cliff; it has no default because "how much metric movement per 1% of
    parameter change is acceptable" is a strategy-specific judgment the
    caller must make explicitly.
    """
    results: list[ParameterPerturbationResult] = []
    for frac in relative_fractions:
        if frac == 0.0:
            msg = "relative_fractions must be nonzero (a zero nudge tests nothing)"
            raise ValueError(msg)
        perturbed_value = base_value * (1.0 + frac)
        metric = evaluate_fn(perturbed_value)
        if metric is None:
            results.append(
                ParameterPerturbationResult(
                    parameter_name=parameter_name,
                    base_value=base_value,
                    perturbed_value=perturbed_value,
                    relative_fraction=frac,
                    base_metric=base_metric,
                    perturbed_metric=None,
                    delta=None,
                    sensitivity=None,
                    is_cliff=None,
                )
            )
            continue
        delta = metric - base_metric
        sensitivity = abs(delta) / abs(frac)
        is_cliff = sensitivity > max_sensitivity
        results.append(
            ParameterPerturbationResult(
                parameter_name=parameter_name,
                base_value=base_value,
                perturbed_value=perturbed_value,
                relative_fraction=frac,
                base_metric=base_metric,
                perturbed_metric=metric,
                delta=delta,
                sensitivity=sensitivity,
                is_cliff=is_cliff,
            )
        )
    return results


def parameter_perturbation_check(
    results: Sequence[ParameterPerturbationResult],
    *,
    name: str = "parameter_perturbation",
) -> AblationCheckResult:
    """Grade a batch of :func:`run_parameter_perturbation` results.

    ``FAILED`` if any evaluated perturbation is a cliff — a fitted parameter
    reads as fine on average but a single cliff means the "small change,
    small effect" property this check exists to verify is false.
    """
    if not results:
        return AblationCheckResult(
            name=name, status=AblationStatus.NOT_RUN, detail="no perturbations supplied"
        )
    evaluated = [r for r in results if r.is_cliff is not None]
    if not evaluated:
        return AblationCheckResult(
            name=name,
            status=AblationStatus.NOT_RUN,
            detail="no perturbation produced a metric",
        )
    cliffs = [r for r in evaluated if r.is_cliff]
    if cliffs:
        fracs = [r.relative_fraction for r in cliffs]
        return AblationCheckResult(
            name=name,
            status=AblationStatus.FAILED,
            detail=(
                f"{len(cliffs)} of {len(evaluated)} perturbation(s) are a cliff "
                f"at fractions {fracs}"
            ),
        )
    return AblationCheckResult(
        name=name,
        status=AblationStatus.SURVIVED,
        detail=f"no cliff detected across {len(evaluated)} tested perturbation(s)",
    )


# ---------------------------------------------------------------------------
# 5. Random seed sensitivity
# ---------------------------------------------------------------------------

SeedEvalFn = Callable[[int], float | None]
"""``(seed) -> metric``."""


@dataclass(frozen=True, slots=True)
class SeedSensitivityResult:
    """Metrics across a batch of seeds, with the spread that matters more
    than the mean: a wide spread means the result is a seed lottery."""

    seeds: tuple[int, ...]
    metrics: tuple[float | None, ...]

    @property
    def evaluated_metrics(self) -> tuple[float, ...]:
        return tuple(m for m in self.metrics if m is not None)

    @property
    def mean(self) -> float | None:
        vals = self.evaluated_metrics
        return statistics.fmean(vals) if vals else None

    @property
    def spread(self) -> float | None:
        """``max - min`` across evaluated seeds. ``None`` if fewer than 2."""
        vals = self.evaluated_metrics
        if len(vals) < 2:
            return None
        return max(vals) - min(vals)


def run_seed_sensitivity(
    seeds: Sequence[int],
    evaluate_fn: SeedEvalFn,
) -> SeedSensitivityResult:
    """Re-run ``evaluate_fn`` across ``seeds`` and collect the spread."""
    metrics = tuple(evaluate_fn(s) for s in seeds)
    return SeedSensitivityResult(seeds=tuple(seeds), metrics=metrics)


def seed_sensitivity_check(
    result: SeedSensitivityResult | None,
    *,
    max_spread: float,
    name: str = "seed_sensitivity",
) -> AblationCheckResult:
    """Grade seed sensitivity: the spread across seeds must not exceed
    ``max_spread``. No default — how much seed-to-seed variation is
    acceptable is a strategy-specific judgment, same rationale as
    ``max_sensitivity`` in :func:`parameter_perturbation_check`.
    """
    if result is None or not result.seeds:
        return AblationCheckResult(
            name=name, status=AblationStatus.NOT_RUN, detail="no seeds supplied"
        )
    spread = result.spread
    if spread is None:
        return AblationCheckResult(
            name=name,
            status=AblationStatus.NOT_RUN,
            detail="fewer than 2 seeds produced a usable metric",
        )
    if not math.isfinite(spread):
        return AblationCheckResult(
            name=name,
            status=AblationStatus.FAILED,
            detail=f"spread is non-finite ({spread!r})",
        )
    if spread <= max_spread:
        return AblationCheckResult(
            name=name,
            status=AblationStatus.SURVIVED,
            detail=(
                f"spread {spread:.4f} <= {max_spread:g} across "
                f"{len(result.evaluated_metrics)} seed(s)"
            ),
        )
    return AblationCheckResult(
        name=name,
        status=AblationStatus.FAILED,
        detail=(
            f"spread {spread:.4f} exceeds {max_spread:g} across "
            f"{len(result.evaluated_metrics)} seed(s)"
        ),
    )
