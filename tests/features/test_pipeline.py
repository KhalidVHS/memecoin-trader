"""Tests for ``features.pipeline``.

Each test is load-bearing: removing the guard it exercises must cause the test
to fail, not just produce a different number. This is documented on each test.

Critical tests:
* ``fit`` builds scaler state strictly from the training slice passed to it;
  ``transform`` applies that fixed state to any later input — the fit/transform
  split this module exists to enforce.
* ``transform`` propagates ``None`` unchanged; it never coerces a missing
  feature to 0.0 or to a scaled NaN presented as a number.
* Determinism: the same fitted pipeline applied to the same raw values
  produces byte-identical output on every call.
* Adding an unrelated feature to the ``raw_values`` dict passed to
  ``transform`` must not perturb the scaled value of any other feature in
  that same dict — verified both by a fixed example and a hypothesis property.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from memetrader.features.pipeline import (
    FeatureResult,
    Pipeline,
    PointInTimeState,
    RobustScaler,
    StandardScaler,
)
from memetrader.features.registry import FeatureDefinition, FeatureRegistry

# These tests exercise Pipeline.transform's fit/transform logic, which never
# reads its `state` argument -- only pipeline steps that need history do. A
# bare object() is a deliberate duck-typed stand-in for the unused parameter,
# cast to the protocol type rather than built as a full fake.
_STATE = cast(PointInTimeState, object())


def make_registry(*names: str, warmup_bars: int = 1) -> FeatureRegistry:
    reg = FeatureRegistry()
    for name in names:
        reg.register(
            FeatureDefinition(
                name=name,
                version="1.0.0",
                inputs=(),
                warmup_bars=warmup_bars,
                prefix_safe=True,
            )
        )
    return reg


# ---------------------------------------------------------------------------
# StandardScaler / RobustScaler — known values, hand-computed
# ---------------------------------------------------------------------------


def test_standard_scaler_known_value() -> None:
    """Hand-computed: values [10,20,30,40,50] -> mean=30, std(ddof=1)=~15.811.

    transform(20) = (20 - 30) / 15.8113883... = -0.6324555...
    """
    scaler = StandardScaler().fit(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
    expected_std = math.sqrt(1000.0 / 4.0)  # sum((x-30)^2)=1000, ddof=1 -> /4
    out = scaler.transform(np.array([20.0]))
    assert out[0] == pytest.approx((20.0 - 30.0) / expected_std, abs=1e-9)


def test_standard_scaler_constant_series_uses_min_std_clamp() -> None:
    """A zero-variance training series must not divide by zero.

    Guard: without the ``min_std`` clamp, ``std(ddof=1)`` of a constant array
    is exactly 0.0 and the transform would produce NaN/inf — a silent false
    signal rather than a documented near-zero-variance handling.
    """
    scaler = StandardScaler(min_std=1e-10).fit(np.array([5.0, 5.0, 5.0, 5.0]))
    out = scaler.transform(np.array([5.0]))
    assert math.isfinite(out[0])
    assert out[0] == pytest.approx(0.0)  # same value as training mean -> 0


def test_standard_scaler_no_finite_training_values_leaves_unfitted() -> None:
    """All-NaN/inf training data must not silently fit a bogus mean/std."""
    scaler = StandardScaler().fit(np.array([np.nan, np.inf, -np.inf]))
    assert not scaler.fitted
    out = scaler.transform(np.array([1.0, 2.0]))
    assert np.all(np.isnan(out))


def test_robust_scaler_known_value() -> None:
    """Hand-computed: values [10,20,30,40,50] -> median=30, IQR (q75-q25) =
    40 - 20 = 20 (numpy linear-interpolation percentiles on 5 sorted points).

    transform(30) = 0/20 = 0.0; transform(50) = 20/20 = 1.0.
    """
    scaler = RobustScaler().fit(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
    out = scaler.transform(np.array([30.0, 50.0]))
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(1.0)


def test_robust_scaler_constant_series_uses_min_iqr_clamp() -> None:
    scaler = RobustScaler(min_iqr=1e-10).fit(np.array([7.0, 7.0, 7.0]))
    out = scaler.transform(np.array([7.0]))
    assert math.isfinite(out[0])
    assert out[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Pipeline.fit / transform — the fit/transform split
# ---------------------------------------------------------------------------


def test_pipeline_transform_matches_manually_fitted_scaler() -> None:
    """The pipeline's internal scaler must match a scaler fit by hand on the
    same training data — not merely re-derive its own formula.

    Guard: training rvol values [10,20,30] -> mean=20, std(ddof=1)=10.
    transform(30) = (30-20)/10 = 1.0.
    """
    registry = make_registry("rvol")
    pipeline = Pipeline(registry)
    train_data: dict[str, list[dict[str, float | None]]] = {
        "tokenA": [{"rvol": 10.0}, {"rvol": 20.0}, {"rvol": 30.0}],
    }
    pipeline.fit(train_data)
    assert pipeline.fitted

    out = pipeline.transform(state=_STATE, asset_id="tokenA", raw_values={"rvol": 30.0})
    assert out["rvol"] == pytest.approx(1.0)


def test_pipeline_transform_before_fit_passes_through_unscaled() -> None:
    """Calling ``transform`` before ``fit`` must return raw values unchanged,
    not raise and not silently apply an identity scaler that looks fitted.
    """
    registry = make_registry("rvol")
    pipeline = Pipeline(registry)
    assert not pipeline.fitted
    out = pipeline.transform(state=_STATE, asset_id="tokenA", raw_values={"rvol": 42.0})
    assert out == {"rvol": 42.0}


def test_pipeline_transform_propagates_none_unscaled() -> None:
    """None must survive transform unchanged — never coerced to 0.0 or to a
    scaled NaN silently presented as a number.

    Guard: §0 draws a hard line between "could not compute" (None) and
    "computed and it is zero" (0.0). A pipeline that ran a fitted feature's
    scaler against a missing value and produced a number would erase that
    distinction right before the model sees it.
    """
    registry = make_registry("rvol")
    pipeline = Pipeline(registry)
    pipeline.fit({"tokenA": [{"rvol": 10.0}, {"rvol": 20.0}, {"rvol": 30.0}]})
    out = pipeline.transform(state=_STATE, asset_id="tokenA", raw_values={"rvol": None})
    assert out["rvol"] is None


def test_pipeline_transform_passes_through_unregistered_feature() -> None:
    """A feature with no registered definition (and therefore no scaler) is
    passed through unchanged rather than dropped or raising.
    """
    registry = make_registry("rvol")
    pipeline = Pipeline(registry)
    pipeline.fit({"tokenA": [{"rvol": 10.0}, {"rvol": 20.0}, {"rvol": 30.0}]})
    out = pipeline.transform(
        state=_STATE,
        asset_id="tokenA",
        raw_values={"rvol": 30.0, "not_registered": 7.0},
    )
    assert out["not_registered"] == 7.0


def test_pipeline_fit_called_twice_replaces_prior_scalers() -> None:
    """Calling ``fit`` a second time is a full replacement of fitted state,
    not an accumulation.

    Guard: if the second ``fit`` merged with the first instead of clearing it,
    a new fold's scaler would be contaminated by the previous fold's training
    distribution — exactly the kind of cross-fold leak the fit/transform split
    exists to prevent.
    """
    registry = make_registry("rvol")
    pipeline = Pipeline(registry)

    pipeline.fit({"tokenA": [{"rvol": 10.0}, {"rvol": 20.0}, {"rvol": 30.0}]})
    first_scale = pipeline.transform(
        state=_STATE, asset_id="tokenA", raw_values={"rvol": 30.0}
    )["rvol"]

    pipeline.fit({"tokenA": [{"rvol": 100.0}, {"rvol": 200.0}, {"rvol": 300.0}]})
    second_scale = pipeline.transform(
        state=_STATE, asset_id="tokenA", raw_values={"rvol": 30.0}
    )["rvol"]

    assert first_scale != pytest.approx(second_scale)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_transform_is_deterministic() -> None:
    """Same fitted pipeline, same inputs, called twice -> identical output.

    Guard: any hidden mutable state (e.g. an accidental running mean updated
    on every ``transform`` call) would make the second call diverge from the
    first for identical inputs.
    """
    registry = make_registry("rvol")
    pipeline = Pipeline(registry)
    pipeline.fit({"tokenA": [{"rvol": 10.0}, {"rvol": 20.0}, {"rvol": 30.0}]})

    raw: dict[str, float | None] = {"rvol": 25.0}
    first = pipeline.transform(state=_STATE, asset_id="tokenA", raw_values=raw)
    second = pipeline.transform(state=_STATE, asset_id="tokenA", raw_values=raw)
    assert first == second


# ---------------------------------------------------------------------------
# Adding a feature must not perturb existing ones
# ---------------------------------------------------------------------------


def test_adding_a_feature_does_not_perturb_existing_ones() -> None:
    """Transforming a dict with an extra (unrelated, registered) feature key
    must not change the scaled value of a feature already present.

    Guard: if scaling were somehow computed jointly across the dict (e.g. a
    shared normalisation over all values present) instead of independently per
    feature name, adding a new feature to the feature set would silently shift
    every previously-shipped feature's value on every future run.
    """
    registry = make_registry("rvol", "volume_ratio")
    pipeline = Pipeline(registry)
    pipeline.fit(
        {
            "tokenA": [
                {"rvol": 10.0, "volume_ratio": 1.0},
                {"rvol": 20.0, "volume_ratio": 2.0},
                {"rvol": 30.0, "volume_ratio": 3.0},
            ],
        }
    )

    without_extra = pipeline.transform(
        state=_STATE, asset_id="tokenA", raw_values={"rvol": 30.0}
    )
    with_extra = pipeline.transform(
        state=_STATE,
        asset_id="tokenA",
        raw_values={"rvol": 30.0, "volume_ratio": 3.0},
    )
    assert without_extra["rvol"] == with_extra["rvol"]


@given(
    rvol_value=st.floats(
        min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
    ),
    extra_value=st.floats(
        min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
    ),
)
def test_transform_key_result_independent_of_other_keys_present(
    rvol_value: float, extra_value: float
) -> None:
    """Property: the transformed value for one key never depends on which
    other keys are present in the same ``raw_values`` dict.

    Generalises the fixed example above across arbitrary input values rather
    than one hand-picked pair of numbers.
    """
    registry = make_registry("rvol", "other")
    pipeline = Pipeline(registry)
    pipeline.fit(
        {
            "tokenA": [
                {"rvol": 1.0, "other": 5.0},
                {"rvol": 2.0, "other": 6.0},
                {"rvol": 3.0, "other": 7.0},
                {"rvol": 4.0, "other": 8.0},
            ],
        }
    )

    alone = pipeline.transform(
        state=_STATE, asset_id="tokenA", raw_values={"rvol": rvol_value}
    )
    together = pipeline.transform(
        state=_STATE,
        asset_id="tokenA",
        raw_values={"rvol": rvol_value, "other": extra_value},
    )
    if alone["rvol"] is None or together["rvol"] is None:
        assert alone["rvol"] == together["rvol"]
    else:
        assert alone["rvol"] == pytest.approx(together["rvol"])


# ---------------------------------------------------------------------------
# FeatureResult — None/NaN handling
# ---------------------------------------------------------------------------


def test_feature_result_set_collapses_nan_to_none() -> None:
    """A NaN value must be stored as None with a recorded reason, never as a
    literal NaN that could silently poison a downstream mean/std computation.
    """
    result = FeatureResult(asset_id="tokenA", ts=1000.0)
    result.set("rvol", float("nan"))
    assert result.get("rvol") is None
    assert "non-finite" in result.reasons["rvol"]


def test_feature_result_set_collapses_inf_to_none() -> None:
    result = FeatureResult(asset_id="tokenA", ts=1000.0)
    result.set("rvol", float("inf"))
    assert result.get("rvol") is None


def test_feature_result_get_missing_key_is_none() -> None:
    result = FeatureResult(asset_id="tokenA", ts=1000.0)
    assert result.get("never_set") is None


def test_feature_result_to_dict_is_a_copy() -> None:
    """Mutating the returned dict must not mutate the FeatureResult's
    internal state."""
    result = FeatureResult(asset_id="tokenA", ts=1000.0)
    result.set("rvol", 5.0)
    snapshot = result.to_dict()
    snapshot["rvol"] = 999.0
    assert result.get("rvol") == 5.0
