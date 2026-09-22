"""Interval-aware purged cross-validation splitters.

Why this module exists instead of using skfolio
-----------------------------------------------
skfolio's ``WalkForward`` and ``CombinatorialPurgedCV`` express purge and
embargo windows as observation *counts*. That is correct for a single asset
sampled uniformly, and wrong for our dataset: BONK emits a 5m bar every 5
minutes, SLERF misses 86.5% of 5m bars and has one print roughly every 37
minutes. An embargo of "21 observations back" means 1h45m for BONK and ~13h for
SLERF from the same wall-clock instant. When BONK observations near a fold
boundary are purged but the equivalent SLERF observations are not, the model can
learn from SLERF's signal around market events that the BONK observations were
purged to exclude. The contamination is invisible in a row-count view and
structurally impossible to express in skfolio's API.

The fix: every boundary, purge window, and embargo window is expressed in
**wall-clock seconds**. All assets are split on the same clock boundaries.
Purging removes any training observation whose ``[label_start_ts, label_end_ts]``
overlaps the validation or test interval, regardless of how many rows that
removes from each asset.

Embargo components (each an explicit named parameter, not one opaque number)
---------------------------------------------------------------------------
``holding_horizon_secs``
    Maximum prediction/position holding horizon. A training label that closes
    one minute before the validation window still depends on prices *inside* the
    validation window if the holding period extends that far.

``publication_delay_secs``
    Time between the last bar of a label and when that bar becomes available.
    Modelled as ``interval_seconds + publication_delay`` in the loaders; here we
    add the full delay again as a safety margin because the loaders' definition
    lives in histdata/ which this module must not import.

``rolling_state_secs``
    Rolling features such as ``realized_vol_pct`` (21 closes × interval) carry
    state backward from the point they are computed. An embargo that does not
    cover this window will leave observations whose *feature values* were
    computed from bars inside the test set, even though the *label* does not
    overlap. Default: 21 × 3600s = 75600s (21 hourly closes), the longest
    rolling window used by any current feature.

``serial_dependence_secs``
    Extra wall-clock margin for serial dependence that is not captured by label
    overlap (momentum, autocorrelated returns). Justified by the data: at 1h the
    autocorrelation of log-returns across all coins has a half-life of roughly
    3 hours in the in-sample period; we conservatively double that. Default: 6h.

The total embargo after each validation/test period is:
    max(holding_horizon_secs, publication_delay_secs)
    + rolling_state_secs
    + serial_dependence_secs

The ``max`` is because the holding horizon and publication delay are two
different bounds on the same thing (when an observation's world stops), not
additive costs.

Holdout policy and the 209-day warning
---------------------------------------
The plan specifies 90 days or ~20% as a holdout. Our measured horizon is ~209
days of 1h data (2026-02-03 → 2026-09-21). 90 days is 43% of the record—too
much; it would leave the outer folds with almost no training data.

``locked_holdout_days`` defaults to ``max(45, int(0.20 * total_days))``.
A loud ``warnings.warn(..., stacklevel=2)`` fires whenever the holdout is below
the plan's 90-day recommendation, quoting the shortfall explicitly. The warning
is *never* suppressed inside this module—the caller may filter it, but they
will have seen it.

Serialisation
-------------
Every split method returns ``list[FoldRecord]``. ``FoldRecord`` is a plain
dataclass (no numpy, no pandas in its definition) expressible as one row of
``folds.parquet`` via ``pyarrow``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterator, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Constants from the data horizon (documented in docs/CANNOT-REPLAY.md)
# ---------------------------------------------------------------------------

# Measured depth of the 1h series: 2026-02-03 → 2026-09-21 = ~209 days.
# Used only to size the holdout warning; the splitters themselves are data-agnostic.
_MEASURED_HORIZON_DAYS: int = 209

# Plan's recommended minimum holdout. We cannot hit it at 209 days without
# gutting the training set, so we default lower and warn loudly.
_PLAN_HOLDOUT_DAYS: int = 90

# Default embargo components (wall-clock seconds). Each documented above.
_DEFAULT_HOLDING_HORIZON_SECS: float = 4 * 3600.0  # 4h max position horizon
_DEFAULT_PUBLICATION_DELAY_SECS: float = 3600.0 + 60.0  # 1h interval + 60s buffer
_DEFAULT_ROLLING_STATE_SECS: float = 21 * 3600.0  # 21 × 1h close window
_DEFAULT_SERIAL_DEPENDENCE_SECS: float = 6 * 3600.0  # 6h autocorrelation margin

# Default outer-fold widths (seconds).
_DAY: float = 86400.0
_DEFAULT_TRAIN_DAYS: float = 180.0
_DEFAULT_VAL_DAYS: float = 30.0
_DEFAULT_TEST_DAYS: float = 30.0
_DEFAULT_ROLL_DAYS: float = 30.0


# ---------------------------------------------------------------------------
# Core data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Interval:
    """A half-open wall-clock interval [start, end) in epoch seconds.

    Half-open so that abutting intervals do not overlap: the validation period
    [val_start, val_end) and the subsequent test period [val_end, test_end)
    share no instant. All comparisons in this module use this convention.
    """

    start: float
    end: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError(f"Interval timestamps must be finite: {self!r}")
        if self.end <= self.start:
            raise ValueError(
                f"Interval end {self.end} must be strictly after start {self.start}"
            )

    def overlaps(self, other: Interval) -> bool:
        """True when [self.start, self.end) and [other.start, other.end) share any instant.

        Two half-open intervals overlap iff neither ends before the other begins.
        Touching at a single point (self.end == other.start) does NOT overlap,
        which is the correct semantics for abutting folds: a training bar whose
        label ends exactly at the validation boundary does not overlap validation.
        """
        return self.start < other.end and other.start < self.end

    def duration_seconds(self) -> float:
        return self.end - self.start

    def duration_days(self) -> float:
        return self.duration_seconds() / _DAY


@dataclass(frozen=True, slots=True)
class LabeledObservation:
    """A single training-eligible observation with its label interval.

    ``asset_id``       — which coin this belongs to (e.g. "BONK").
    ``obs_time``       — when the observation's features were computed (epoch s).
    ``label_start_ts`` — start of the future window the label is computed over.
    ``label_end_ts``   — end of the future window the label is computed over.

    The interval [label_start_ts, label_end_ts] must not overlap any
    validation or test interval for this observation to remain in training.
    Using the full label interval (not just label_end_ts) is the right choice
    because a label computed from prices *inside* the validation window leaks
    even if the label end is outside it.
    """

    asset_id: str
    obs_time: float
    label_start_ts: float
    label_end_ts: float

    def __post_init__(self) -> None:
        if self.label_end_ts <= self.label_start_ts:
            raise ValueError(
                f"label_end_ts {self.label_end_ts} must be after "
                f"label_start_ts {self.label_start_ts}"
            )

    @property
    def label_interval(self) -> Interval:
        return Interval(self.label_start_ts, self.label_end_ts)


@dataclass(frozen=True, slots=True)
class FoldRecord:
    """One fold's boundaries and metadata — one row of folds.parquet.

    All timestamps are epoch seconds (float), matching the project-wide
    convention in types.py §0. There are no datetime objects here.

    ``purged_count`` and ``embargoed_count`` are totals across all assets,
    so the fold audit can verify that non-zero purge actually happened at
    fold boundaries (and that zero purge *also* happens away from them).
    """

    fold_id: str
    splitter_id: str
    train_start: float
    train_end: float
    val_start: float
    val_end: float
    test_start: float
    test_end: float
    embargo_end: float  # wall-clock end of the embargo zone after test_end
    purged_count: int = 0
    embargoed_count: int = 0
    notes: str = ""

    @property
    def train_interval(self) -> Interval:
        return Interval(self.train_start, self.train_end)

    @property
    def val_interval(self) -> Interval:
        return Interval(self.val_start, self.val_end)

    @property
    def test_interval(self) -> Interval:
        return Interval(self.test_start, self.test_end)

    def as_dict(self) -> dict[str, object]:
        """Flat dict for a parquet row; avoids a pyarrow dependency here."""
        return {
            "fold_id": self.fold_id,
            "splitter_id": self.splitter_id,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "val_start": self.val_start,
            "val_end": self.val_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "embargo_end": self.embargo_end,
            "purged_count": self.purged_count,
            "embargoed_count": self.embargoed_count,
            "notes": self.notes,
        }


@dataclass(slots=True)
class SplitResult:
    """The output of one fold's purge pass.

    ``train_indices`` — integer positions in the original observations list
      that survived purging and are safe for training.
    ``val_indices``   — integer positions in the validation window.
    ``test_indices``  — integer positions in the test window.
    ``fold``          — the ``FoldRecord`` with counts filled in.

    Indices are into the same sequence of ``LabeledObservation`` objects
    passed to the splitter, so the caller can use them for numpy fancy-indexing
    or pandas ``.iloc[]`` on any parallel array.
    """

    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]
    fold: FoldRecord


# ---------------------------------------------------------------------------
# Embargo helpers
# ---------------------------------------------------------------------------


def embargo_window(
    *,
    holding_horizon_secs: float = _DEFAULT_HOLDING_HORIZON_SECS,
    publication_delay_secs: float = _DEFAULT_PUBLICATION_DELAY_SECS,
    rolling_state_secs: float = _DEFAULT_ROLLING_STATE_SECS,
    serial_dependence_secs: float = _DEFAULT_SERIAL_DEPENDENCE_SECS,
) -> float:
    """Total embargo in seconds after a validation/test interval ends.

    Components are named and explicit rather than a single opaque number so
    that the risk each covers survives into code review and experiment logs.
    See module docstring for full justification.

    The max() between holding_horizon_secs and publication_delay_secs is
    intentional: both express "how long after the last label bar is an
    observation's world still entangled with the fold boundary?" They are
    not additive—the longer one subsumes the shorter.
    """
    return (
        max(holding_horizon_secs, publication_delay_secs)
        + rolling_state_secs
        + serial_dependence_secs
    )


# ---------------------------------------------------------------------------
# Core purge logic
# ---------------------------------------------------------------------------


def purge_training(
    observations: Sequence[LabeledObservation],
    *,
    protected_intervals: Sequence[Interval],
    train_interval: Interval,
    embargo_secs: float,
    enable_purge: bool = True,
) -> tuple[list[int], int, int]:
    """Return (safe_indices, n_purged, n_embargoed) for training observations.

    This is the anti-overfitting core.

    An observation is **purged** when its ``[label_start_ts, label_end_ts]``
    overlaps any interval in ``protected_intervals`` (val + test intervals).
    Purging removes observations whose label was computed using prices that
    the model will later be evaluated on — without this, a model trained on
    "what happened in the test week" is not a generalisation test, it is an
    in-sample fit presented as out-of-sample.

    An observation is **embargoed** when it falls in the wall-clock zone
    ``[pi.end, pi.end + embargo_secs)`` for any protected interval ``pi``.
    The embargo removes observations whose *features* (rolling vol, momentum)
    were computed from bars inside or immediately after the protected interval,
    even when the label itself does not overlap. Without the rolling_state
    component of the embargo, a realized_vol computed from 21 bars that straddle
    the boundary would appear in training with no purge removing it.

    ``enable_purge=False`` disables all filtering and returns all indices inside
    ``train_interval``. This parameter exists only to make the "purge-off" test
    possible: a test that asserts contamination exists when purging is disabled
    proves the guard actually matters, not just that the happy path passes.

    All assets are subject to the same wall-clock boundaries. There is no
    per-asset row-count logic anywhere in this function, which is the structural
    guarantee that a thin-bar asset like SLERF cannot slip contaminated
    observations through that BONK's observations would not.
    """
    safe: list[int] = []
    n_purged = 0
    n_embargoed = 0

    # Pre-compute embargo zones: [pi.end, pi.end + embargo_secs) for each
    # protected interval.  We keep them as (start, end) pairs to avoid
    # constructing Interval objects in the inner loop.
    embargo_zones: list[tuple[float, float]] = [
        (pi.end, pi.end + embargo_secs) for pi in protected_intervals
    ]

    for i, obs in enumerate(observations):
        # Only consider observations within the train interval.
        if obs.obs_time < train_interval.start or obs.obs_time >= train_interval.end:
            continue

        if not enable_purge:
            safe.append(i)
            continue

        label = obs.label_interval

        # Purge check: label interval overlaps any protected interval.
        purged = any(label.overlaps(pi) for pi in protected_intervals)
        if purged:
            n_purged += 1
            continue

        # Embargo check: obs_time falls in any embargo zone.
        embargoed = any(
            ez_start <= obs.obs_time < ez_end for ez_start, ez_end in embargo_zones
        )
        if embargoed:
            n_embargoed += 1
            continue

        safe.append(i)

    return safe, n_purged, n_embargoed


# ---------------------------------------------------------------------------
# Holdout warning
# ---------------------------------------------------------------------------


def _check_holdout(
    holdout_days: float,
    total_days: float,
    splitter_name: str,
) -> None:
    """Emit a loud warning when holdout is below the plan's recommendation.

    The plan says 90 days or ~20%. At our 209-day horizon, 90 days is 43%,
    which would leave the outer folds with almost no training data. We default
    to max(45, 20%) = ~42 days and warn explicitly so no result can be
    interpreted without knowing the shortfall.

    The warning always fires—it is the *caller's* responsibility to filter it
    with ``warnings.filterwarnings`` if they have a good reason, and that
    filtering decision is then visible in the code.
    """
    if holdout_days < _PLAN_HOLDOUT_DAYS:
        shortfall = _PLAN_HOLDOUT_DAYS - holdout_days
        warnings.warn(
            f"[{splitter_name}] Locked holdout is {holdout_days:.1f} days "
            f"({100.0 * holdout_days / total_days:.1f}% of {total_days:.0f}-day horizon), "
            f"which is {shortfall:.1f} days below the plan's recommended "
            f"{_PLAN_HOLDOUT_DAYS} days. This shortfall exists because the measured "
            f"data horizon is only {_MEASURED_HORIZON_DAYS} days; 90 days would be "
            f"{100.0 * _PLAN_HOLDOUT_DAYS / total_days:.0f}% of the record, leaving "
            f"insufficient training data for the outer folds. All results from this "
            f"run must be read with this constraint in mind.",
            UserWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# PurgedWalkForward
# ---------------------------------------------------------------------------


@dataclass
class PurgedWalkForward:
    """Rolling walk-forward with purge + embargo on wall-clock boundaries.

    Produces outer folds of the form:
        [train_start ... train_end) [val_start ... val_end) [test_start ... test_end)
    where each successive fold shifts forward by ``roll_days``.

    All boundary arithmetic is in wall-clock seconds. Observation-count-based
    boundaries are not used anywhere in this class; see module docstring.

    Parameters
    ----------
    train_days, val_days, test_days, roll_days:
        Width of each window in calendar days. Defaults match the plan's
        "starting values" (180/30/30/30 days).

    holding_horizon_secs, publication_delay_secs, rolling_state_secs,
    serial_dependence_secs:
        Embargo components. See :func:`embargo_window` and module docstring.

    locked_holdout_days:
        How much of the *end* of the dataset to reserve as a final holdout.
        ``None`` triggers the default: max(45, 20% of total span). A warning
        fires if this is below 90 days (the plan's recommendation).

    splitter_id:
        Stable identifier written into every FoldRecord for traceability.
    """

    train_days: float = _DEFAULT_TRAIN_DAYS
    val_days: float = _DEFAULT_VAL_DAYS
    test_days: float = _DEFAULT_TEST_DAYS
    roll_days: float = _DEFAULT_ROLL_DAYS
    holding_horizon_secs: float = _DEFAULT_HOLDING_HORIZON_SECS
    publication_delay_secs: float = _DEFAULT_PUBLICATION_DELAY_SECS
    rolling_state_secs: float = _DEFAULT_ROLLING_STATE_SECS
    serial_dependence_secs: float = _DEFAULT_SERIAL_DEPENDENCE_SECS
    locked_holdout_days: float | None = None
    splitter_id: str = "purged_walk_forward_v1"

    def _embargo_secs(self) -> float:
        return embargo_window(
            holding_horizon_secs=self.holding_horizon_secs,
            publication_delay_secs=self.publication_delay_secs,
            rolling_state_secs=self.rolling_state_secs,
            serial_dependence_secs=self.serial_dependence_secs,
        )

    def _resolve_holdout(self, total_span_secs: float) -> float:
        """Return the locked holdout in seconds, emitting a warning if needed."""
        total_days = total_span_secs / _DAY
        if self.locked_holdout_days is not None:
            ho = self.locked_holdout_days
        else:
            ho = max(45.0, 0.20 * total_days)
        _check_holdout(ho, total_days, self.splitter_id)
        return ho * _DAY

    def split(
        self,
        observations: Sequence[LabeledObservation],
        *,
        data_start: float,
        data_end: float,
        enable_purge: bool = True,
    ) -> list[SplitResult]:
        """Generate all folds over [data_start, data_end - holdout].

        ``data_start`` and ``data_end`` are epoch seconds. The locked holdout
        is carved from the *end* of the span before any fold is produced.

        ``enable_purge=False`` disables purge+embargo (for the "guard test"
        only — never in production).
        """
        holdout_secs = self._resolve_holdout(data_end - data_start)
        effective_end = data_end - holdout_secs

        train_secs = self.train_days * _DAY
        val_secs = self.val_days * _DAY
        test_secs = self.test_days * _DAY
        roll_secs = self.roll_days * _DAY
        embargo_secs = self._embargo_secs()

        results: list[SplitResult] = []
        fold_num = 0
        cursor = data_start

        while True:
            train_start = cursor
            train_end = cursor + train_secs
            val_start = train_end
            val_end = val_start + val_secs
            test_start = val_end
            test_end = test_start + test_secs

            # Stop when the test window would exceed the effective end.
            if test_end > effective_end:
                break

            embargo_end = test_end + embargo_secs

            val_interval = Interval(val_start, val_end)
            test_interval = Interval(test_start, test_end)
            train_interval = Interval(train_start, train_end)
            protected = [val_interval, test_interval]

            train_idx, n_purged, n_embargoed = purge_training(
                observations,
                protected_intervals=protected,
                train_interval=train_interval,
                embargo_secs=embargo_secs,
                enable_purge=enable_purge,
            )

            # Val/test indices: obs_time in [window.start, window.end).
            val_idx = [
                i
                for i, obs in enumerate(observations)
                if val_start <= obs.obs_time < val_end
            ]
            test_idx = [
                i
                for i, obs in enumerate(observations)
                if test_start <= obs.obs_time < test_end
            ]

            fold_record = FoldRecord(
                fold_id=f"{self.splitter_id}_fold{fold_num:03d}",
                splitter_id=self.splitter_id,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
                embargo_end=embargo_end,
                purged_count=n_purged,
                embargoed_count=n_embargoed,
            )
            results.append(
                SplitResult(
                    train_indices=train_idx,
                    val_indices=val_idx,
                    test_indices=test_idx,
                    fold=fold_record,
                )
            )

            fold_num += 1
            cursor += roll_secs

        return results


# ---------------------------------------------------------------------------
# CombinatorialPurgedCV (Lopez de Prado CPCV)
# ---------------------------------------------------------------------------


@dataclass
class CombinatorialPurgedCV:
    """Combinatorial purged cross-validation (Lopez de Prado, AFML §12).

    Divides the non-holdout span into ``n_groups`` equal-width time groups.
    For each combination of ``k`` groups chosen as test, the remaining
    ``n_groups - k`` groups form the training set (after purge + embargo).
    The validation window is not a separate entity in CPCV: the combinatorial
    averaging across paths serves the same role.

    Number of test paths produced:
        C(n_groups, k) = n_groups! / (k! * (n_groups - k)!)

    A conventional split with n_groups=2, k=1 reproduces a single train/test
    split; n_groups=6, k=2 gives C(6,2)=15 paths, each with a different
    view of the same dataset.

    Parameters
    ----------
    n_groups:
        Number of equal-width time groups to divide the data into.
    k:
        Number of groups to use as the test set in each combination.
    holding_horizon_secs, publication_delay_secs, rolling_state_secs,
    serial_dependence_secs:
        Embargo components; see module docstring.
    locked_holdout_days:
        Same as in PurgedWalkForward; default is max(45, 20%).
    splitter_id:
        Identifier written into FoldRecord rows.
    """

    n_groups: int = 6
    k: int = 2
    holding_horizon_secs: float = _DEFAULT_HOLDING_HORIZON_SECS
    publication_delay_secs: float = _DEFAULT_PUBLICATION_DELAY_SECS
    rolling_state_secs: float = _DEFAULT_ROLLING_STATE_SECS
    serial_dependence_secs: float = _DEFAULT_SERIAL_DEPENDENCE_SECS
    locked_holdout_days: float | None = None
    splitter_id: str = "cpcv_v1"

    def __post_init__(self) -> None:
        if self.k >= self.n_groups:
            raise ValueError(
                f"k={self.k} must be strictly less than n_groups={self.n_groups}; "
                "at least one group must remain for training."
            )
        if self.k < 1:
            raise ValueError(f"k={self.k} must be >= 1.")
        if self.n_groups < 2:
            raise ValueError(f"n_groups={self.n_groups} must be >= 2.")

    def n_paths(self) -> int:
        """Number of test paths this configuration produces: C(n_groups, k)."""
        n = self.n_groups
        kk = self.k
        # math.comb is exact integer arithmetic — no floating-point rounding.
        return math.comb(n, kk)

    def _embargo_secs(self) -> float:
        return embargo_window(
            holding_horizon_secs=self.holding_horizon_secs,
            publication_delay_secs=self.publication_delay_secs,
            rolling_state_secs=self.rolling_state_secs,
            serial_dependence_secs=self.serial_dependence_secs,
        )

    def _resolve_holdout(self, total_span_secs: float) -> float:
        total_days = total_span_secs / _DAY
        if self.locked_holdout_days is not None:
            ho = self.locked_holdout_days
        else:
            ho = max(45.0, 0.20 * total_days)
        _check_holdout(ho, total_days, self.splitter_id)
        return ho * _DAY

    def split(
        self,
        observations: Sequence[LabeledObservation],
        *,
        data_start: float,
        data_end: float,
        enable_purge: bool = True,
    ) -> list[SplitResult]:
        """Generate one SplitResult per combination of k groups as test.

        Each SplitResult's ``fold.val_start == fold.val_end == fold.test_start``:
        CPCV has no separate validation interval. The fold_id encodes the
        specific combination of groups used as test (e.g. "cpcv_v1_g2g4") so
        results from different paths are distinguishable in folds.parquet.

        The total embargo applied after each contiguous run of test groups
        is the same as in PurgedWalkForward; see module docstring.
        """
        holdout_secs = self._resolve_holdout(data_end - data_start)
        effective_end = data_end - holdout_secs
        total_secs = effective_end - data_start

        group_width = total_secs / self.n_groups
        embargo_secs = self._embargo_secs()

        # Build group boundaries: list of (start, end) in epoch seconds.
        group_bounds: list[tuple[float, float]] = [
            (data_start + i * group_width, data_start + (i + 1) * group_width)
            for i in range(self.n_groups)
        ]

        results: list[SplitResult] = []

        for test_group_indices in combinations(range(self.n_groups), self.k):
            test_group_set = set(test_group_indices)
            train_group_indices = [
                i for i in range(self.n_groups) if i not in test_group_set
            ]

            # Test intervals: one per test group.
            test_intervals = [
                Interval(group_bounds[gi][0], group_bounds[gi][1])
                for gi in test_group_indices
            ]

            # Val interval: not meaningful in CPCV, but FoldRecord requires it.
            # We use the first test interval's start/end as a degenerate val.
            val_start = test_intervals[0].start
            val_end = test_intervals[0].start  # zero-width: no val in CPCV

            # Test span: from earliest to latest test group boundary.
            test_start = min(iv.start for iv in test_intervals)
            test_end = max(iv.end for iv in test_intervals)

            embargo_end = test_end + embargo_secs

            # Training indices: obs_time in any train group, after purge.
            train_obs_indices_in_groups: list[int] = []
            for gi in train_group_indices:
                g_start, g_end = group_bounds[gi]
                for i, obs in enumerate(observations):
                    if g_start <= obs.obs_time < g_end:
                        train_obs_indices_in_groups.append(i)

            # Now apply purge + embargo against all test intervals.
            # We build a synthetic "train interval" covering all train groups
            # by treating the full span as train and relying on the group filter
            # above—but purge_training needs a single Interval. Instead, we
            # run purge manually here since training observations are already
            # limited to train groups.
            if enable_purge:
                embargo_zones = [(iv.end, iv.end + embargo_secs) for iv in test_intervals]
                safe_train: list[int] = []
                n_purged = 0
                n_embargoed = 0
                for i in train_obs_indices_in_groups:
                    obs = observations[i]
                    label = obs.label_interval
                    purged = any(label.overlaps(iv) for iv in test_intervals)
                    if purged:
                        n_purged += 1
                        continue
                    embargoed = any(
                        ez_s <= obs.obs_time < ez_e for ez_s, ez_e in embargo_zones
                    )
                    if embargoed:
                        n_embargoed += 1
                        continue
                    safe_train.append(i)
            else:
                safe_train = list(train_obs_indices_in_groups)
                n_purged = 0
                n_embargoed = 0

            # Test indices: obs_time in any test group.
            test_idx = [
                i
                for i, obs in enumerate(observations)
                if any(iv.start <= obs.obs_time < iv.end for iv in test_intervals)
            ]

            group_label = "g" + "g".join(str(gi) for gi in test_group_indices)
            fold_id = f"{self.splitter_id}_{group_label}"

            fold_record = FoldRecord(
                fold_id=fold_id,
                splitter_id=self.splitter_id,
                train_start=min(group_bounds[gi][0] for gi in train_group_indices),
                train_end=max(group_bounds[gi][1] for gi in train_group_indices),
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
                embargo_end=embargo_end,
                purged_count=n_purged,
                embargoed_count=n_embargoed,
                notes=f"CPCV test groups: {test_group_indices}",
            )
            results.append(
                SplitResult(
                    train_indices=safe_train,
                    val_indices=[],  # no separate val in CPCV
                    test_indices=test_idx,
                    fold=fold_record,
                )
            )

        return results


# ---------------------------------------------------------------------------
# Convenience: locked holdout extractor
# ---------------------------------------------------------------------------


def extract_locked_holdout(
    observations: Sequence[LabeledObservation],
    *,
    data_start: float,
    data_end: float,
    locked_holdout_days: float | None = None,
    splitter_id: str = "holdout",
) -> tuple[list[int], list[int], FoldRecord]:
    """Split observations into (development_indices, holdout_indices, record).

    The holdout is the *end* of the timeline and must be opened at most once.
    The FoldRecord encodes the boundary; the experiment registry
    (``experiments/registry.py``) records when it was opened so that a second
    opening invalidates the run.

    Returns a triple so the caller can immediately check that holdout_indices
    is non-empty before opening it.
    """
    total_days = (data_end - data_start) / _DAY
    if locked_holdout_days is not None:
        ho = locked_holdout_days
    else:
        ho = max(45.0, 0.20 * total_days)

    _check_holdout(ho, total_days, splitter_id)
    holdout_start = data_end - ho * _DAY

    dev_idx = [i for i, o in enumerate(observations) if o.obs_time < holdout_start]
    ho_idx = [i for i, o in enumerate(observations) if o.obs_time >= holdout_start]

    record = FoldRecord(
        fold_id=f"{splitter_id}_locked",
        splitter_id=splitter_id,
        train_start=data_start,
        train_end=holdout_start,
        val_start=holdout_start,
        val_end=holdout_start,
        test_start=holdout_start,
        test_end=data_end,
        embargo_end=data_end,
        notes=f"locked holdout: {ho:.1f} days",
    )
    return dev_idx, ho_idx, record


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "CombinatorialPurgedCV",
    "FoldRecord",
    "Interval",
    "LabeledObservation",
    "PurgedWalkForward",
    "SplitResult",
    "embargo_window",
    "extract_locked_holdout",
    "purge_training",
]
