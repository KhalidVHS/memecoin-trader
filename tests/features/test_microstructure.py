"""Tests for ``features.microstructure``.

Each test is load-bearing: removing the guard it exercises must cause the test
to fail, not just produce a different number. This is documented on each test.

Critical tests:
* All four features degrade honestly to ``None`` when the fidelity tier lacks
  quote data (``ladder is None``, the current TIER_0 state) — never a
  fabricated spread/depth/impact of 0.0, which would read as "flat market,
  zero cost" in exactly the regime (thin memecoin liquidity) where that
  reading is most dangerous.
* ``price_impact_pct_at_size`` and ``spread_bps`` preserve the whole-number
  percentage convention: an impact of ``0.5`` means 0.5%, not 0.005.
* ``quote_ladder_age_seconds`` is exactly ``now - available_time``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from hypothesis import given
from hypothesis import strategies as st

from memetrader.features.microstructure import (
    depth_usd_at_size,
    price_impact_pct_at_size,
    quote_ladder_age_seconds,
    spread_bps,
)


@dataclass
class FakeRung:
    in_amount_atomic: int
    price_impact_pct: float


@dataclass
class FakeLadder:
    """Minimal stand-in for ``histdata.schemas.QuoteLadder``.

    ``best_rung_for`` mirrors the real schema's floor-lookup: the largest
    rung whose ``in_amount_atomic`` is <= the requested size, or None if no
    rung qualifies.
    """

    rungs: tuple[FakeRung, ...]
    available_time: float | None = None

    def best_rung_for(self, in_amount_atomic: int) -> FakeRung | None:
        candidates = [r for r in self.rungs if r.in_amount_atomic <= in_amount_atomic]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.in_amount_atomic)


# ---------------------------------------------------------------------------
# TIER_0 degradation — every feature, ladder is None
# ---------------------------------------------------------------------------


def test_all_features_degrade_to_none_when_no_ladder() -> None:
    """The central microstructure guarantee: at TIER_0, every feature reports
    None, never a fabricated neutral value.

    Guard: a flat 0.0 spread or 0.0 impact would read to a downstream model as
    "liquid market, no cost" — exactly backwards for a fidelity tier that has
    no evidence about liquidity at all. thin-market microstructure is the kill
    condition for memecoin positions, not the baseline, so the absence of data
    must not silently default to the safe-looking case.
    """
    spread, spread_reason = spread_bps(None, mid_price_usd=1.0)
    depth, depth_reason = depth_usd_at_size(None, 1000, price_usd_per_atomic=0.01)
    impact, impact_reason = price_impact_pct_at_size(None, 1000)
    age, age_reason = quote_ladder_age_seconds(None, now=1000.0)

    assert (spread, depth, impact, age) == (None, None, None, None)
    for reason in (spread_reason, depth_reason, impact_reason, age_reason):
        assert "TIER_0" in reason


# ---------------------------------------------------------------------------
# spread_bps
# ---------------------------------------------------------------------------


def test_spread_bps_known_value() -> None:
    """Hand-computed: smallest rung (100 atomic) has impact_pct=0.5 (0.5%) ->
    bps = 2 * 0.5 * 100 = 100.0 bps.
    """
    ladder = FakeLadder(
        rungs=(
            FakeRung(in_amount_atomic=100, price_impact_pct=0.5),
            FakeRung(in_amount_atomic=1000, price_impact_pct=1.2),
        )
    )
    value, reason = spread_bps(ladder, mid_price_usd=1.0)
    assert value == pytest.approx(100.0)
    assert reason == ""


def test_spread_bps_uses_smallest_rung_not_first_in_list() -> None:
    """The smallest-size rung must be selected regardless of list order.

    Guard: if the function used ``rungs[0]`` instead of the min by size, an
    unsorted ladder would silently use the wrong rung's impact.
    """
    ladder = FakeLadder(
        rungs=(
            FakeRung(in_amount_atomic=1000, price_impact_pct=9.0),  # listed first
            FakeRung(in_amount_atomic=50, price_impact_pct=0.25),  # smallest size
        )
    )
    value, reason = spread_bps(ladder, mid_price_usd=1.0)
    assert value == pytest.approx(2.0 * 0.25 * 100.0)
    assert reason == ""


def test_spread_bps_none_mid_price() -> None:
    ladder = FakeLadder(rungs=(FakeRung(in_amount_atomic=100, price_impact_pct=0.5),))
    value, reason = spread_bps(ladder, mid_price_usd=None)
    assert value is None
    assert "mid-price" in reason


def test_spread_bps_non_positive_mid_price() -> None:
    ladder = FakeLadder(rungs=(FakeRung(in_amount_atomic=100, price_impact_pct=0.5),))
    value, reason = spread_bps(ladder, mid_price_usd=0.0)
    assert value is None
    assert "mid-price" in reason


def test_spread_bps_empty_rungs() -> None:
    ladder = FakeLadder(rungs=())
    value, reason = spread_bps(ladder, mid_price_usd=1.0)
    assert value is None
    assert "empty" in reason


def test_spread_bps_nonfinite_impact_returns_none() -> None:
    ladder = FakeLadder(
        rungs=(FakeRung(in_amount_atomic=100, price_impact_pct=float("nan")),)
    )
    value, reason = spread_bps(ladder, mid_price_usd=1.0)
    assert value is None
    assert "non-finite" in reason


# ---------------------------------------------------------------------------
# depth_usd_at_size
# ---------------------------------------------------------------------------


def test_depth_usd_at_size_known_value() -> None:
    """Hand-computed: rung at 500 atomic, price 0.01 USD/atomic -> $5.00."""
    ladder = FakeLadder(rungs=(FakeRung(in_amount_atomic=500, price_impact_pct=0.3),))
    value, reason = depth_usd_at_size(ladder, 500, price_usd_per_atomic=0.01)
    assert value == pytest.approx(5.0)
    assert reason == ""


def test_depth_usd_at_size_no_price_oracle() -> None:
    ladder = FakeLadder(rungs=(FakeRung(in_amount_atomic=500, price_impact_pct=0.3),))
    value, reason = depth_usd_at_size(ladder, 500, price_usd_per_atomic=None)
    assert value is None
    assert "price_usd_per_atomic" in reason


def test_depth_usd_at_size_no_rung_covers_requested_size() -> None:
    """Requesting a size smaller than every rung on the ladder finds nothing.

    Guard: without the floor-rung check, this could silently fall back to the
    smallest rung and report depth for a size nobody asked about.
    """
    ladder = FakeLadder(rungs=(FakeRung(in_amount_atomic=10_000, price_impact_pct=0.3),))
    value, reason = depth_usd_at_size(ladder, 100, price_usd_per_atomic=0.01)
    assert value is None
    assert "no rung" in reason


# ---------------------------------------------------------------------------
# price_impact_pct_at_size — whole-number percentage convention
# ---------------------------------------------------------------------------


def test_price_impact_pct_at_size_known_value_is_whole_number_percent() -> None:
    """A rung recording 1.5 means 1.5%, not 0.015 — §0's whole-number
    percentage convention. The function must pass the rung's value through
    unscaled.
    """
    ladder = FakeLadder(rungs=(FakeRung(in_amount_atomic=500, price_impact_pct=1.5),))
    value, reason = price_impact_pct_at_size(ladder, 500)
    assert value == pytest.approx(1.5)
    assert reason == ""


def test_price_impact_pct_at_size_no_rung_found() -> None:
    ladder = FakeLadder(rungs=())
    value, reason = price_impact_pct_at_size(ladder, 500)
    assert value is None
    assert "no rung" in reason


def test_price_impact_pct_at_size_nonfinite_returns_none() -> None:
    ladder = FakeLadder(
        rungs=(FakeRung(in_amount_atomic=500, price_impact_pct=float("inf")),)
    )
    value, _reason = price_impact_pct_at_size(ladder, 500)
    assert value is None


# ---------------------------------------------------------------------------
# quote_ladder_age_seconds
# ---------------------------------------------------------------------------


def test_quote_ladder_age_seconds_known_value() -> None:
    ladder = FakeLadder(rungs=(), available_time=700.0)
    value, reason = quote_ladder_age_seconds(ladder, now=1000.0)
    assert value == pytest.approx(300.0)
    assert reason == ""


def test_quote_ladder_age_seconds_none_available_time() -> None:
    ladder = FakeLadder(rungs=(), available_time=None)
    value, reason = quote_ladder_age_seconds(ladder, now=1000.0)
    assert value is None
    assert "available_time" in reason


@given(
    available_time=st.floats(
        min_value=0, max_value=1e9, allow_nan=False, allow_infinity=False
    ),
    delta=st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_quote_ladder_age_seconds_is_exactly_now_minus_available_time(
    available_time: float, delta: float
) -> None:
    """Property: age is always exactly ``now - available_time``, for any
    non-negative elapsed ``delta``. Generalises the fixed known-value test
    above across the float range so a coincidental match on one pair of
    numbers cannot hide a wrong formula (e.g. an inverted sign or an added
    constant).
    """
    now = available_time + delta
    ladder = FakeLadder(rungs=(), available_time=available_time)
    value, _ = quote_ladder_age_seconds(ladder, now=now)
    assert value == pytest.approx(delta, abs=1e-6)
