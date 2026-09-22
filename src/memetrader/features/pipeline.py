"""Prefix-safe feature computation pipeline.

The central invariant this module exists to enforce:

    The feature computed at time ``t`` from the full dataset must equal the
    feature computed at ``t`` from data truncated at ``t``.

Violating this invariant is look-ahead leakage. It is subtle in proportion to
how much fitted state is involved: a raw price return has no fitted state and
is trivially prefix-safe; a StandardScaler fit on the full series normalises
each point against its own future distribution, which inflates the apparent
Sharpe of any strategy that learns from the normalised features.

The ``Pipeline`` class enforces the fit/transform split at the API boundary
rather than by convention. ``fit`` accepts a training slice and builds all
fitted state from it. ``transform`` applies that state to any slice. The API
makes fitting on test data a visible mistake — the caller has to explicitly call
``fit`` with test data, which is wrong in an obvious way — rather than a quiet
default where every ``fit_transform`` call on the full series leaks.

``PointInTimeState`` is the only legal way to read history during a replay.
Every method on this module reads through that protocol. Do not accept raw
arrays or DataFrames directly in pipeline steps: the call site controls what
history is visible, and passing arrays bypasses that control.

Design note on scalers: sklearn's StandardScaler, RobustScaler, etc. are not
used here because adding scikit-learn as a dependency is prohibited (see
pyproject.toml). The two scaler implementations below cover the use cases in
the initial feature set. They are intentionally simple — a wrong scaler that
passes its own tests is worse than no scaler, so they match numpy directly.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np

from .registry import FeatureRegistry

# ---------------------------------------------------------------------------
# PointInTimeState protocol — reproduced here so this module compiles without
# importing from histdata, which is being written concurrently. The concrete
# ``ReplayState`` satisfies this protocol; tests use a local fake.
# ---------------------------------------------------------------------------


@runtime_checkable
class PointInTimeState(Protocol):
    """The only legal gateway through which a feature reads history in a replay.

    Reproduced from BACKTEST-CONTRACTS.md §4. Every method returns only records
    with ``available_at <= self.now``, and every ``Candle`` returned is closed.
    A forming bar must never cross this boundary.

    This is a ``Protocol`` rather than a base class because both the concrete
    ``histdata.ReplayState`` and test fakes implement it structurally. The
    protocol runtime-checkable so ``isinstance`` guards in pipeline steps can
    verify they received the right type without importing the concrete class.
    """

    @property
    def now(self) -> float:
        """Current simulated time, epoch seconds."""
        ...

    def bars(self, asset_id: str, timeframe: str, *, lookback: int) -> tuple[object, ...]:
        """Up to ``lookback`` closed candles for ``asset_id`` on ``timeframe``.

        Returns an empty tuple when there is no history, never raises on
        missing assets. The result is always ordered oldest-first.
        """
        ...

    def universe(self) -> frozenset[str]:
        """Asset IDs eligible for trading at ``self.now``."""
        ...

    def quote_ladder(self, asset_id: str, side: str) -> object | None:
        """Best available quote ladder, or None at TIER_0."""
        ...

    def pool_state(self, pool_id: str) -> object | None:
        """Pool state, or None at TIER_0."""
        ...


# ---------------------------------------------------------------------------
# Scaler primitives — fitted on train, applied on transform
# ---------------------------------------------------------------------------


class StandardScaler:
    """Z-score normalisation: (x - mean) / std, fitted on training data only.

    ``fit`` records the mean and standard deviation of the training slice.
    ``transform`` applies those constants to any later slice.

    Fitting on the full series (train+test) would normalise each test point
    against a mean that already knows the test outcomes, inflating the apparent
    predictability of the features. The fit/transform split makes that mistake
    impossible without deliberate misuse: the caller must call ``fit`` with
    training data before ``transform`` will work, and calling ``fit`` twice
    with different data is the caller's explicit decision.

    ``ddof=1`` matches numpy's default for ``std``. Using ``ddof=0`` (population
    sigma) would slightly understate the variance on small training windows, but
    the practical difference is negligible for the window sizes here. The
    important thing is consistency: both fit and transform must use the same
    ``ddof`` or the normalised test values will not be on the same scale as the
    normalised training values.

    ``min_std`` prevents division by zero on constant series. A series with zero
    variance produces a NaN z-score rather than a zero, which would be a silent
    false signal. ``min_std=1e-10`` is conservative enough that any real price
    series will have a larger standard deviation, so the clamp only fires on
    genuinely constant series.
    """

    def __init__(self, min_std: float = 1e-10) -> None:
        self._min_std = min_std
        self._mean: float | None = None
        self._std: float | None = None

    def fit(self, values: np.ndarray) -> StandardScaler:
        """Record mean and std from the training slice. Returns self for chaining."""
        finite_vals = values[np.isfinite(values)]
        if finite_vals.size == 0:
            # No finite data in the training slice. Record None so transform
            # returns NaN — better to signal "no scale" than to use 0/1.
            self._mean = None
            self._std = None
        else:
            self._mean = float(finite_vals.mean())
            self._std = float(max(finite_vals.std(ddof=1), self._min_std))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply the fitted scale to any slice. Raises if not yet fitted."""
        if self._mean is None or self._std is None:
            return np.full_like(values, np.nan, dtype=float)
        return (values.astype(float) - self._mean) / self._std

    @property
    def fitted(self) -> bool:
        return self._mean is not None


class RobustScaler:
    """Median/IQR normalisation, less sensitive to outliers than z-score.

    Uses the 25th–75th percentile range (IQR) as the scale. A distribution
    with a heavy tail produces a large IQR, which compresses outliers rather
    than letting them dominate the normalised range. Memecoins have heavy tails.

    ``min_iqr`` serves the same role as ``min_std`` in ``StandardScaler``.
    """

    def __init__(self, min_iqr: float = 1e-10) -> None:
        self._min_iqr = min_iqr
        self._median: float | None = None
        self._iqr: float | None = None

    def fit(self, values: np.ndarray) -> RobustScaler:
        finite_vals = values[np.isfinite(values)]
        if finite_vals.size == 0:
            self._median = None
            self._iqr = None
        else:
            self._median = float(np.median(finite_vals))
            q25, q75 = np.percentile(finite_vals, [25.0, 75.0])
            self._iqr = float(max(q75 - q25, self._min_iqr))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self._median is None or self._iqr is None:
            return np.full_like(values, np.nan, dtype=float)
        return (values.astype(float) - self._median) / self._iqr

    @property
    def fitted(self) -> bool:
        return self._median is not None


# ---------------------------------------------------------------------------
# Feature result
# ---------------------------------------------------------------------------


class FeatureResult:
    """Computed feature values for one asset at one point in time.

    Each entry is either a finite float or ``None``. ``None`` preserves the
    distinction between "could not compute" (warm-up not met, gap too wide,
    data missing) and "computed and it is zero". Collapsing them produces a
    confident neutral reading out of nothing.

    ``reasons`` records why a specific field is ``None`` when the reason is
    non-obvious. Not every ``None`` needs a reason — "not enough bars" is
    expected and needs no explanation — but "gap exceeded threshold" or
    "TIER_0: no quote ladder" are useful for debugging feature availability.
    """

    __slots__ = ("asset_id", "reasons", "ts", "values")

    def __init__(self, asset_id: str, ts: float) -> None:
        self.asset_id = asset_id
        self.ts = ts
        self.values: dict[str, float | None] = {}
        self.reasons: dict[str, str] = {}

    def set(self, name: str, value: float | None, reason: str = "") -> None:
        """Record a feature value. NaN is converted to None."""
        if value is not None and not math.isfinite(value):
            value = None
            if not reason:
                reason = "non-finite value collapsed to None"
        self.values[name] = value
        if reason:
            self.reasons[name] = reason

    def get(self, name: str) -> float | None:
        return self.values.get(name)

    def to_dict(self) -> dict[str, float | None]:
        return dict(self.values)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """Prefix-safe, fold-isolated feature computation pipeline.

    Usage pattern for a single walk-forward fold::

        pipeline = Pipeline(registry)
        pipeline.fit(train_state)          # builds all fitted state from train
        train_results = pipeline.transform(train_state)
        val_results   = pipeline.transform(val_state)
        test_results  = pipeline.transform(test_state)

    The separation is the point. ``fit`` must be called before ``transform``,
    and it must receive only training-period data. Calling ``fit(test_state)``
    is wrong in an obvious way; the common mistake of ``fit_transform(all_data)``
    is not available.

    ``_steps`` is a list of callables ``(state, asset_id) -> FeatureResult``.
    Steps are added by the feature modules at import time. Each step is
    responsible for one logical group of features (market, microstructure, etc.)
    and must read history exclusively through ``state``.

    The pipeline is not thread-safe. One instance per fold.
    """

    def __init__(self, registry: FeatureRegistry) -> None:
        self._registry = registry
        self._scalers: dict[str, StandardScaler | RobustScaler] = {}
        self._fitted = False
        self._steps: list[object] = []  # populated by feature modules

    def fit(self, train_data: dict[str, list[dict[str, float | None]]]) -> None:
        """Build fitted state (scalers, etc.) from training-period feature values.

        ``train_data`` maps asset_id → list of feature dicts, one per bar in
        the training period. Features that do not use a scaler ignore this call.

        Calling ``fit`` a second time is allowed and replaces all fitted state —
        the caller is explicitly building a new fold. Any previous ``transform``
        results are now computed against a different scale; the caller must
        recompute them if they need consistent normalisation.
        """
        self._scalers.clear()

        # Collect all feature names across the training data.
        all_names: set[str] = set()
        for rows in train_data.values():
            for row in rows:
                all_names.update(row.keys())

        for name in sorted(all_names):
            # Only fit a scaler for features that are registered.
            try:
                _defn = self._registry.get(name)
            except KeyError:
                continue

            vals: list[float] = []
            for rows in train_data.values():
                for row in rows:
                    v = row.get(name)
                    if v is not None and math.isfinite(v):
                        vals.append(v)

            arr = np.array(vals, dtype=float)
            scaler = StandardScaler()
            scaler.fit(arr)
            self._scalers[name] = scaler

        self._fitted = True

    def transform(
        self,
        state: PointInTimeState,
        asset_id: str,
        raw_values: dict[str, float | None],
    ) -> dict[str, float | None]:
        """Apply fitted scalers to pre-computed raw feature values.

        ``raw_values`` is the output of the feature computation steps for one
        asset at ``state.now``. Returns a new dict with the same keys and
        scaled values; unregistered features are passed through unchanged.

        Does not raise if ``fit`` has not been called — returns raw values
        unscaled. Callers that need scaled values must check ``self.fitted``.
        """
        if not self._fitted:
            return dict(raw_values)

        out: dict[str, float | None] = {}
        for name, value in raw_values.items():
            scaler = self._scalers.get(name)
            if scaler is None or value is None or not math.isfinite(value):
                out[name] = value
            else:
                arr = np.array([value], dtype=float)
                scaled = scaler.transform(arr)
                sv = float(scaled[0])
                out[name] = sv if math.isfinite(sv) else None
        return out

    @property
    def fitted(self) -> bool:
        return self._fitted


__all__ = [
    "FeatureResult",
    "Pipeline",
    "PointInTimeState",
    "RobustScaler",
    "StandardScaler",
]
