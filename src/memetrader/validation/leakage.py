"""Automated leakage detection — the checks that prove a backtest is not cheating.

BACKTEST-CONTRACTS.md §7 requires four independent guards, each of which
targets a distinct way a backtest grants itself foresight:

1. **Prefix equivalence** (:func:`check_prefix_equivalence`) — the feature
   computed "as of" index ``t`` from the full series must equal the feature
   computed "as of" the same logical point from data truncated at ``t``. This
   is the strongest single leakage test there is: any feature that reads its
   own future, however indirectly (a global mean, a full-series scaler, a
   lookahead join), fails it.

2. **Future sentinel** (:func:`check_future_sentinel`) — poison every
   observation after ``t`` with an absurd value and recompute; nothing at or
   before ``t`` may change. Complementary to prefix equivalence: prefix
   equivalence catches a feature that silently *uses* the full array, the
   sentinel catches one that reads specific future values without going
   through a length check at all.

3. **Timestamp monotonicity** (:func:`check_timestamp_monotonicity`) — no
   record returned by a ``PointInTimeState`` may have ``available_time >
   now``, and no returned candle may be unclosed. This is checked directly
   against the point-in-time protocol described in BACKTEST-CONTRACTS.md §4,
   not re-derived from feature values.

4. **Label horizon** (:func:`check_label_horizon`) — cross-checks
   ``validation.splits.purge_training``: no observation whose
   ``[label_start_ts, label_end_ts]`` overlaps a protected (validation/test)
   interval may survive as a "safe" training index.

Every check returns a :class:`LeakageCheckResult` with an explicit
``CheckStatus``. §0's convention — ``None`` means "could not find out", ``0``
means "looked, and it is quiet" — applies here to whole checks: a check that
was not run is ``SKIPPED``, never silently reported as ``PASSED``. See
:class:`LeakageReport.clean`, which is ``True`` only when every check that
was included actually ran and passed; a report with any ``SKIPPED`` entry is
never "clean".
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np

from .splits import Interval, LabeledObservation, purge_training

# ---------------------------------------------------------------------------
# PointInTimeState protocol — reproduced locally (same pattern as
# features/pipeline.py) so this module does not force an import ordering
# against histdata/, which the point-in-time invariant depends on but which
# this module must not need at import time.
# ---------------------------------------------------------------------------


@runtime_checkable
class PointInTimeStateLike(Protocol):
    """The minimal slice of ``PointInTimeState`` the timestamp check needs."""

    @property
    def now(self) -> float: ...

    def bars(
        self, asset_id: str, timeframe: object, *, lookback: int
    ) -> tuple[object, ...]: ...


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class CheckStatus(StrEnum):
    """Outcome of one leakage check.

    ``SKIPPED`` is a distinct state from ``PASSED`` on purpose: §0's rule that
    ``None`` ("could not find out") must never collapse into ``0`` ("looked,
    and it is quiet") applies at the level of whole checks. A caller who
    forgets to wire up the label-horizon check must see "skipped", never a
    clean report that implies the check ran.
    """

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class LeakageCheckResult:
    """The outcome of one leakage check, with enough detail to act on it.

    ``violations`` is a count, not a boolean, so a batch check_all() summary
    can report "3 of 500 training observations leaked" rather than a bare
    fail/pass that loses the magnitude of the problem.
    """

    name: str
    status: CheckStatus
    detail: str = ""
    violations: int = 0

    @property
    def ok(self) -> bool:
        """True when the check ran and found nothing. False if it failed OR
        was skipped — a skipped check is never evidence of cleanliness."""
        return self.status == CheckStatus.PASSED


@dataclass(frozen=True, slots=True)
class LeakageReport:
    """Summary of every leakage check run (or explicitly skipped) for a backtest.

    Written to ``leakage_report.json`` per BACKTEST-CONTRACTS.md §9.
    """

    checks: tuple[LeakageCheckResult, ...] = field(default_factory=tuple)

    @property
    def failed(self) -> tuple[LeakageCheckResult, ...]:
        return tuple(c for c in self.checks if c.status == CheckStatus.FAILED)

    @property
    def skipped(self) -> tuple[LeakageCheckResult, ...]:
        return tuple(c for c in self.checks if c.status == CheckStatus.SKIPPED)

    @property
    def passed(self) -> tuple[LeakageCheckResult, ...]:
        return tuple(c for c in self.checks if c.status == CheckStatus.PASSED)

    @property
    def clean(self) -> bool:
        """True only when every check ran and every one of them passed.

        A report with zero checks, or with any check skipped, is not clean —
        "we did not check" must never render as "clean" (§0). Callers that
        want a partial-coverage summary should read ``.skipped`` explicitly
        rather than trusting a truthy ``clean``.
        """
        return bool(self.checks) and all(
            c.status == CheckStatus.PASSED for c in self.checks
        )

    def as_dict(self) -> dict[str, object]:
        """Flat JSON-serializable representation for ``leakage_report.json``."""
        return {
            "clean": self.clean,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "detail": c.detail,
                    "violations": c.violations,
                }
                for c in self.checks
            ],
            "n_passed": len(self.passed),
            "n_failed": len(self.failed),
            "n_skipped": len(self.skipped),
        }


# ---------------------------------------------------------------------------
# Shared comparison helper
# ---------------------------------------------------------------------------


def _values_equal(a: float | None, b: float | None, *, rtol: float, atol: float) -> bool:
    """Compare two optional floats, treating None-None and NaN-NaN as equal.

    None and NaN are both "no value" in different vocabularies (§0: a
    computed feature converts NaN to None, per features/pipeline.py's
    FeatureResult.set), so both must compare equal to themselves without
    tripping the leakage detector on a benign "still warming up" case that
    happens identically on both sides of the comparison.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    a_nan = math.isnan(a)
    b_nan = math.isnan(b)
    if a_nan and b_nan:
        return True
    if a_nan or b_nan:
        return False
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


# A point-in-time feature: given a 1-D series and the index that represents
# "now" within that series, return the feature value at that index (or None
# if it cannot be computed, e.g. insufficient warm-up). An honestly point-in-
# time feature must produce the same answer whether it is handed the full
# series (and told which index is "now") or only the prefix ending at that
# index — because it must not read past ``at_index`` either way.
FeatureFn = Callable[[np.ndarray, int], float | None]


# ---------------------------------------------------------------------------
# Check 1 — prefix equivalence
# ---------------------------------------------------------------------------


def check_prefix_equivalence(
    feature_fn: FeatureFn,
    series: np.ndarray,
    *,
    at_index: int,
    name: str = "prefix_equivalence",
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> LeakageCheckResult:
    """Recompute ``feature_fn`` over the full series vs. the truncated prefix.

    Calls ``feature_fn(series, at_index)`` and ``feature_fn(series[:at_index+1],
    at_index)`` and asserts they agree. A feature whose value depends on any
    observation past ``at_index`` — a full-series mean, a leaked join, a
    scaler fit on the whole run — will disagree between the two calls even
    though both calls describe "the value at the same point in time".

    Per BACKTEST-CONTRACTS.md §7 / plan §13: "feature at t from the full
    dataset == feature at t from data truncated at t."
    """
    if series.ndim != 1:
        raise ValueError(f"series must be 1-D, got shape {series.shape}")
    if not 0 <= at_index < series.size:
        raise ValueError(
            f"at_index={at_index} out of range for series of size {series.size}"
        )

    value_from_full = feature_fn(series, at_index)
    prefix = series[: at_index + 1]
    value_from_prefix = feature_fn(prefix, prefix.size - 1)

    if _values_equal(value_from_full, value_from_prefix, rtol=rtol, atol=atol):
        return LeakageCheckResult(
            name=name,
            status=CheckStatus.PASSED,
            detail=(
                f"value at index {at_index} agrees between full-series and "
                f"prefix-truncated computation ({value_from_full!r})"
            ),
        )
    return LeakageCheckResult(
        name=name,
        status=CheckStatus.FAILED,
        detail=(
            f"value at index {at_index} DIFFERS between full-series computation "
            f"({value_from_full!r}) and prefix-truncated computation "
            f"({value_from_prefix!r}) — the feature reads data beyond {at_index}"
        ),
        violations=1,
    )


# ---------------------------------------------------------------------------
# Check 2 — future sentinel
# ---------------------------------------------------------------------------


def check_future_sentinel(
    feature_fn: FeatureFn,
    series: np.ndarray,
    *,
    at_index: int,
    sentinel: float = 1.0e12,
    name: str = "future_sentinel",
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> LeakageCheckResult:
    """Poison everything after ``at_index`` and assert the value at
    ``at_index`` does not move.

    ``sentinel`` defaults to an absurd finite value rather than NaN so that a
    feature computing e.g. ``np.mean`` over an accidentally-included future
    slice produces a wildly different (not merely NaN-propagated, possibly
    masked) result — a large finite outlier cannot hide behind a NaN-guard
    that a feature already has for missing data. Pass ``sentinel=math.nan``
    explicitly to also exercise NaN-propagation paths.
    """
    if series.ndim != 1:
        raise ValueError(f"series must be 1-D, got shape {series.shape}")
    if not 0 <= at_index < series.size:
        raise ValueError(
            f"at_index={at_index} out of range for series of size {series.size}"
        )

    baseline = feature_fn(series, at_index)
    poisoned = series.astype(float, copy=True)
    n_poisoned = poisoned.size - (at_index + 1)
    if n_poisoned > 0:
        poisoned[at_index + 1 :] = sentinel
    poisoned_value = feature_fn(poisoned, at_index)

    if _values_equal(baseline, poisoned_value, rtol=rtol, atol=atol):
        return LeakageCheckResult(
            name=name,
            status=CheckStatus.PASSED,
            detail=(
                f"poisoning {n_poisoned} observation(s) after index {at_index} "
                f"left the value unchanged ({baseline!r})"
            ),
        )
    return LeakageCheckResult(
        name=name,
        status=CheckStatus.FAILED,
        detail=(
            f"poisoning the future changed the value at index {at_index} from "
            f"{baseline!r} to {poisoned_value!r} — something read past the cutoff"
        ),
        violations=1,
    )


# ---------------------------------------------------------------------------
# Check 3 — timestamp monotonicity
# ---------------------------------------------------------------------------


def check_timestamp_monotonicity(
    state: PointInTimeStateLike,
    *,
    asset_id: str,
    timeframe: object,
    lookback: int,
    interval_seconds: float,
    publication_delay_seconds: float = 0.0,
    name: str = "timestamp_monotonicity",
) -> LeakageCheckResult:
    """Assert every bar ``state.bars(...)`` returns is legally available now.

    Per BACKTEST-CONTRACTS.md §1: "A candle is unavailable until ts + interval
    + publication_delay." and every returned ``Candle`` must have
    ``closed is True``. This check calls the live ``PointInTimeState`` at its
    current ``now`` and verifies both properties directly against the
    protocol, rather than trusting an internal implementation detail.
    """
    candles = state.bars(asset_id, timeframe, lookback=lookback)
    now = state.now

    late: list[float] = []
    unclosed: list[float] = []
    for candle in candles:
        ts = float(candle.ts)  # type: ignore[attr-defined]
        available_at = ts + interval_seconds + publication_delay_seconds
        if available_at > now:
            late.append(ts)
        if not bool(candle.closed):  # type: ignore[attr-defined]
            unclosed.append(ts)

    violations = len(late) + len(unclosed)
    if violations == 0:
        return LeakageCheckResult(
            name=name,
            status=CheckStatus.PASSED,
            detail=f"all {len(candles)} bar(s) returned at now={now} are closed and available",
        )

    detail_parts: list[str] = []
    if late:
        detail_parts.append(
            f"{len(late)} bar(s) with available_time > now={now}: ts={late}"
        )
    if unclosed:
        detail_parts.append(f"{len(unclosed)} bar(s) not closed: ts={unclosed}")
    return LeakageCheckResult(
        name=name,
        status=CheckStatus.FAILED,
        detail="; ".join(detail_parts),
        violations=violations,
    )


# ---------------------------------------------------------------------------
# Check 4 — label horizon vs. splits.purge_training
# ---------------------------------------------------------------------------


def check_label_horizon(
    observations: Sequence[LabeledObservation],
    *,
    protected_intervals: Sequence[Interval],
    train_interval: Interval,
    embargo_secs: float,
    enable_purge: bool = True,
    name: str = "label_horizon",
) -> LeakageCheckResult:
    """Cross-check ``splits.purge_training`` against the raw label overlap rule.

    An observation is a violation if it is reported "safe" for training by
    ``purge_training`` yet its ``[label_start_ts, label_end_ts]`` overlaps any
    protected (validation/test) interval. With ``enable_purge=True`` this
    check must find zero violations; the paired test in
    ``tests/validation/test_leakage.py`` also runs this with
    ``enable_purge=False`` and asserts violations appear, proving the check
    would actually catch contamination if the purge guard were ever disabled
    or broken.
    """
    safe_indices, _n_purged, _n_embargoed = purge_training(
        observations,
        protected_intervals=protected_intervals,
        train_interval=train_interval,
        embargo_secs=embargo_secs,
        enable_purge=enable_purge,
    )

    violating: list[int] = []
    for i in safe_indices:
        obs = observations[i]
        if any(obs.label_interval.overlaps(pi) for pi in protected_intervals):
            violating.append(i)

    if not violating:
        return LeakageCheckResult(
            name=name,
            status=CheckStatus.PASSED,
            detail=(
                f"{len(safe_indices)} training observation(s) marked safe; none has a "
                "label interval overlapping a protected interval"
            ),
        )
    return LeakageCheckResult(
        name=name,
        status=CheckStatus.FAILED,
        detail=(
            f"{len(violating)} of {len(safe_indices)} 'safe' training observation(s) "
            f"have a label interval overlapping a protected interval: indices {violating}"
        ),
        violations=len(violating),
    )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PrefixCheckSpec:
    """Inputs for one :func:`check_prefix_equivalence` call in :func:`check_all`."""

    name: str
    feature_fn: FeatureFn
    series: np.ndarray
    at_index: int
    rtol: float = 1e-9
    atol: float = 1e-9


@dataclass(frozen=True, slots=True)
class SentinelCheckSpec:
    """Inputs for one :func:`check_future_sentinel` call in :func:`check_all`."""

    name: str
    feature_fn: FeatureFn
    series: np.ndarray
    at_index: int
    sentinel: float = 1.0e12
    rtol: float = 1e-9
    atol: float = 1e-9


@dataclass(frozen=True, slots=True)
class TimestampCheckSpec:
    """Inputs for one :func:`check_timestamp_monotonicity` call in :func:`check_all`."""

    state: PointInTimeStateLike
    asset_id: str
    timeframe: object
    lookback: int
    interval_seconds: float
    publication_delay_seconds: float = 0.0
    name: str = "timestamp_monotonicity"


@dataclass(frozen=True, slots=True)
class LabelHorizonCheckSpec:
    """Inputs for one :func:`check_label_horizon` call in :func:`check_all`."""

    observations: Sequence[LabeledObservation]
    protected_intervals: Sequence[Interval]
    train_interval: Interval
    embargo_secs: float
    enable_purge: bool = True
    name: str = "label_horizon"


def check_all(
    *,
    prefix_checks: Sequence[PrefixCheckSpec] = (),
    sentinel_checks: Sequence[SentinelCheckSpec] = (),
    timestamp_checks: Sequence[TimestampCheckSpec] = (),
    label_horizon_checks: Sequence[LabelHorizonCheckSpec] = (),
) -> LeakageReport:
    """Run every supplied check and record explicit skips for any check family
    that received no specs.

    Each of the four check families is independently optional so a caller
    that, say, has no ``PointInTimeState`` wired up yet still gets a report —
    but that report's ``timestamp_monotonicity`` entry is ``SKIPPED``, not
    silently absent, and ``LeakageReport.clean`` is ``False`` until it is
    wired up. This is the direct implementation of §0's "None != 0": a
    report must never look clean because a check was never run.
    """
    results: list[LeakageCheckResult] = []

    if prefix_checks:
        results.extend(
            check_prefix_equivalence(
                spec.feature_fn,
                spec.series,
                at_index=spec.at_index,
                name=spec.name,
                rtol=spec.rtol,
                atol=spec.atol,
            )
            for spec in prefix_checks
        )
    else:
        results.append(
            LeakageCheckResult(
                name="prefix_equivalence",
                status=CheckStatus.SKIPPED,
                detail="no PrefixCheckSpec supplied to check_all",
            )
        )

    if sentinel_checks:
        results.extend(
            check_future_sentinel(
                spec.feature_fn,
                spec.series,
                at_index=spec.at_index,
                sentinel=spec.sentinel,
                name=spec.name,
                rtol=spec.rtol,
                atol=spec.atol,
            )
            for spec in sentinel_checks
        )
    else:
        results.append(
            LeakageCheckResult(
                name="future_sentinel",
                status=CheckStatus.SKIPPED,
                detail="no SentinelCheckSpec supplied to check_all",
            )
        )

    if timestamp_checks:
        results.extend(
            check_timestamp_monotonicity(
                tspec.state,
                asset_id=tspec.asset_id,
                timeframe=tspec.timeframe,
                lookback=tspec.lookback,
                interval_seconds=tspec.interval_seconds,
                publication_delay_seconds=tspec.publication_delay_seconds,
                name=tspec.name,
            )
            for tspec in timestamp_checks
        )
    else:
        results.append(
            LeakageCheckResult(
                name="timestamp_monotonicity",
                status=CheckStatus.SKIPPED,
                detail="no TimestampCheckSpec supplied to check_all",
            )
        )

    if label_horizon_checks:
        results.extend(
            check_label_horizon(
                lspec.observations,
                protected_intervals=lspec.protected_intervals,
                train_interval=lspec.train_interval,
                embargo_secs=lspec.embargo_secs,
                enable_purge=lspec.enable_purge,
                name=lspec.name,
            )
            for lspec in label_horizon_checks
        )
    else:
        results.append(
            LeakageCheckResult(
                name="label_horizon",
                status=CheckStatus.SKIPPED,
                detail="no LabelHorizonCheckSpec supplied to check_all",
            )
        )

    return LeakageReport(checks=tuple(results))


__all__ = [
    "CheckStatus",
    "FeatureFn",
    "LabelHorizonCheckSpec",
    "LeakageCheckResult",
    "LeakageReport",
    "PointInTimeStateLike",
    "PrefixCheckSpec",
    "SentinelCheckSpec",
    "TimestampCheckSpec",
    "check_all",
    "check_future_sentinel",
    "check_label_horizon",
    "check_prefix_equivalence",
    "check_timestamp_monotonicity",
]
