"""Tests for metrics.capacity — position and strategy capacity limits.

Core properties: more size implies more (or equal) price impact, and the
strategy-level capacity ceiling (min across positions of capacity_at_1pct)
is respected by the exceeds_intended_size flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from memetrader.metrics.capacity import (
    estimate_capacity,
    liquidation_value,
)
from memetrader.types import FidelityTier, Mark

TIER_0 = FidelityTier.TIER_0
TIER_2 = FidelityTier.TIER_2


@dataclass
class _Position:
    quantity: float


def _mark(
    price: float | None = 1.0,
    basis: Literal["route", "mid", "estimate", "unavailable"] = "estimate",
) -> Mark:
    return Mark(symbol="FOO", price_usd=price, basis=basis, provenance=None)


# ---------------------------------------------------------------------------
# Monotonicity: more size -> more (or equal) price impact
# ---------------------------------------------------------------------------


@given(
    adv=st.floats(min_value=1_000.0, max_value=10_000_000.0, allow_nan=False),
    small=st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False),
    extra=st.floats(min_value=0.01, max_value=10_000.0, allow_nan=False),
    model=st.sampled_from(["sqrt", "linear"]),
)
def test_impact_pct_monotonic_in_size(
    adv: float, small: float, extra: float, model: str
) -> None:
    """A larger reference size against the same ADV must never produce a
    smaller estimated price impact."""
    large = small + extra
    mark_small = _mark(price=1.0)
    mark_large = _mark(price=1.0)

    _, small_exec = liquidation_value(
        mark_small,
        small,
        fidelity=TIER_0,
        bar_volume_usd=adv * 0.01,
        participation_cap=0.01,
        price_impact_model=model,
    )
    _, large_exec = liquidation_value(
        mark_large,
        large,
        fidelity=TIER_0,
        bar_volume_usd=adv * 0.01,
        participation_cap=0.01,
        price_impact_model=model,
    )
    small_impact_pct = 100.0 * (1.0 - small_exec / (1.0 * small)) if small > 0 else 0.0
    large_impact_pct = 100.0 * (1.0 - large_exec / (1.0 * large)) if large > 0 else 0.0
    assert large_impact_pct >= small_impact_pct - 1e-9


def test_impact_pct_known_values_sqrt_model() -> None:
    """impact_pct = k * sqrt(size / adv) * 100, k=1: size=100, adv=10000 ->
    sqrt(0.01)*100 = 10.0."""
    from memetrader.metrics.capacity import _compute_impact_pct

    impact = _compute_impact_pct(100.0, 10_000.0, "sqrt")
    assert impact == pytest.approx(10.0)


def test_impact_pct_known_values_linear_model() -> None:
    """impact_pct = (size / adv) * 100: size=100, adv=10000 -> 1.0."""
    from memetrader.metrics.capacity import _compute_impact_pct

    impact = _compute_impact_pct(100.0, 10_000.0, "linear")
    assert impact == pytest.approx(1.0)


def test_capacity_at_1pct_known_values() -> None:
    from memetrader.metrics.capacity import _compute_capacity_at_1pct

    # linear: size = adv * 0.01
    assert _compute_capacity_at_1pct(10_000.0, "linear") == pytest.approx(100.0)
    # sqrt: size = adv * (0.01/k)^2, k=1 -> adv * 0.0001
    assert _compute_capacity_at_1pct(10_000.0, "sqrt") == pytest.approx(1.0)


def test_capacity_at_1pct_none_adv_is_zero() -> None:
    from memetrader.metrics.capacity import _compute_capacity_at_1pct

    assert _compute_capacity_at_1pct(None, "sqrt") == 0.0


# ---------------------------------------------------------------------------
# liquidation_value: executable basis at TIER_2 bypasses haircut
# ---------------------------------------------------------------------------


def test_executable_mark_at_tier2_has_no_haircut() -> None:
    mark = _mark(price=2.0, basis="route")
    reference, executable = liquidation_value(
        mark,
        50.0,
        fidelity=TIER_2,
        bar_volume_usd=1.0,
        participation_cap=0.01,
    )
    assert reference == pytest.approx(100.0)
    assert executable == pytest.approx(100.0)


def test_non_executable_mark_at_tier0_has_haircut() -> None:
    mark = _mark(price=2.0, basis="estimate")
    reference, executable = liquidation_value(
        mark,
        50.0,
        fidelity=TIER_0,
        bar_volume_usd=1000.0,
        participation_cap=0.01,
    )
    assert reference == pytest.approx(100.0)
    assert executable < reference


def test_liquidation_value_no_volume_data_uses_conservative_haircut() -> None:
    """No ADV data at all -> fixed conservative 10% haircut, not zero and not
    a crash."""
    mark = _mark(price=1.0, basis="estimate")
    reference, executable = liquidation_value(
        mark,
        100.0,
        fidelity=TIER_0,
        bar_volume_usd=None,
    )
    assert reference == pytest.approx(100.0)
    assert executable == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# estimate_capacity: monotonicity and the ceiling
# ---------------------------------------------------------------------------


def test_estimate_capacity_larger_position_has_larger_or_equal_haircut() -> None:
    marks = {"FOO": _mark(price=1.0, basis="estimate")}
    small_positions = {"FOO": _Position(quantity=100.0)}
    large_positions = {"FOO": _Position(quantity=10_000.0)}
    bar_volume = {"FOO": 1_000_000.0}

    small_report = estimate_capacity(
        marks,
        small_positions,
        fidelity=TIER_0,
        bar_volume_usd=bar_volume,
    )
    large_report = estimate_capacity(
        marks,
        large_positions,
        fidelity=TIER_0,
        bar_volume_usd=bar_volume,
    )
    assert large_report.positions[0].haircut_pct >= small_report.positions[0].haircut_pct
    assert (
        large_report.positions[0].estimated_price_impact_pct
        >= small_report.positions[0].estimated_price_impact_pct
    )


def test_exceeds_intended_size_flag_respects_ceiling() -> None:
    marks = {"FOO": _mark(price=1.0, basis="estimate")}
    positions = {"FOO": _Position(quantity=100.0)}
    bar_volume = {"FOO": 1_000_000.0}  # adv_usd = 1_000_000 / 0.01 = 1e8
    # capacity_at_1pct for sqrt model, k=1: adv * 0.0001 = 10_000.0
    report = estimate_capacity(
        marks,
        positions,
        fidelity=TIER_0,
        bar_volume_usd=bar_volume,
        intended_live_position_usd=5_000.0,
    )
    assert report.strategy_capacity_usd == pytest.approx(10_000.0)
    assert report.exceeds_intended_size is True

    report_high_ceiling = estimate_capacity(
        marks,
        positions,
        fidelity=TIER_0,
        bar_volume_usd=bar_volume,
        intended_live_position_usd=50_000.0,
    )
    assert report_high_ceiling.exceeds_intended_size is False


def test_exceeds_intended_size_none_when_no_intended_size_given() -> None:
    marks = {"FOO": _mark(price=1.0, basis="estimate")}
    positions = {"FOO": _Position(quantity=100.0)}
    report = estimate_capacity(marks, positions, fidelity=TIER_0)
    assert report.exceeds_intended_size is False
    assert report.intended_live_position_usd is None


def test_missing_mark_produces_zeroed_warned_entry() -> None:
    positions = {"FOO": _Position(quantity=100.0)}
    report = estimate_capacity({}, positions, fidelity=TIER_0)
    assert len(report.positions) == 1
    assert report.positions[0].warning is not None
    assert report.positions[0].reference_mark_usd == 0.0


def test_empty_positions_yield_zero_strategy_capacity() -> None:
    report = estimate_capacity({}, {}, fidelity=TIER_0)
    assert report.strategy_capacity_usd == 0.0
    assert report.positions == []


def test_tier0_carries_non_executable_notice() -> None:
    marks = {"FOO": _mark(price=1.0, basis="route")}
    positions = {"FOO": _Position(quantity=10.0)}
    report = estimate_capacity(marks, positions, fidelity=TIER_0)
    assert report.non_executable_notice is not None


def test_tier2_executable_route_mark_carries_no_notice() -> None:
    marks = {"FOO": _mark(price=1.0, basis="route")}
    positions = {"FOO": _Position(quantity=10.0)}
    report = estimate_capacity(marks, positions, fidelity=TIER_2)
    assert report.non_executable_notice is None
    assert report.positions[0].is_executable_mark is True
    assert report.positions[0].warning is None
