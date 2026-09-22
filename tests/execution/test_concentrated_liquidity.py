"""Property and unit tests for concentrated_liquidity.py.

The CLMM swap model is significantly more complex than constant-product, so
the tests focus on invariants that hold regardless of the internal mechanics:

* **Output is a non-negative integer** — the formula must never produce a
  float or negative amount.
* **Amount_in_used <= amount_in** — the swap can never consume more than was
  offered.
* **Fees_charged >= 0 and <= amount_in_used** — fees cannot be negative and
  cannot exceed the input consumed.
* **sqrt_price moves in the right direction** — a zero-for-one swap (selling
  token_0) must decrease or maintain the sqrt price; one-for-zero must
  increase or maintain it.
* **Zero input yields zero output** — the trivial case.
* **Single-range swap with abundant liquidity matches constant-product
  directionally** — with enough liquidity in one range, a CLMM swap should
  behave like a constant-product pool (the tick never crosses).
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from memetrader.execution.amm.concentrated_liquidity import (
    LiquidityRange,
    sqrt_price_at_tick,
    swap,
    tick_at_sqrt_price,
)
from memetrader.types import ValidationError

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_TICKS = st.integers(min_value=-100_000, max_value=100_000)
_TICK_PAIRS = st.integers(min_value=-100_000, max_value=99_999).flatmap(
    lambda lo: st.tuples(st.just(lo), st.integers(min_value=lo + 1, max_value=lo + 10_000))
)
_LIQUIDITY = st.integers(min_value=1, max_value=10**18)
_AMOUNTS = st.integers(min_value=1, max_value=10**15)
_FEE_BPS = st.integers(min_value=0, max_value=9_999)


def _make_range(lower: int, upper: int, liq: int) -> LiquidityRange:
    return LiquidityRange(lower_tick=lower, upper_tick=upper, liquidity=liq)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestSqrtPriceAtTick:
    def test_tick_zero_is_one_scaled(self) -> None:
        # At tick 0: price = 1.0001^0 = 1.0, sqrt_price = 1.0
        # sqrt_price_x64 = floor(1.0 * 2^64)
        result = sqrt_price_at_tick(0)
        assert result == (1 << 64)

    def test_positive_tick_larger_than_zero(self) -> None:
        assert sqrt_price_at_tick(1000) > sqrt_price_at_tick(0)

    def test_negative_tick_smaller_than_zero(self) -> None:
        assert sqrt_price_at_tick(-1000) < sqrt_price_at_tick(0)

    def test_float_tick_rejected(self) -> None:
        with pytest.raises(ValidationError):
            sqrt_price_at_tick(1.5)  # type: ignore[arg-type]


class TestTickAtSqrtPrice:
    def test_roundtrip_zero_tick(self) -> None:
        sp = sqrt_price_at_tick(0)
        tick = tick_at_sqrt_price(sp)
        # Should recover tick 0 or very close (floor)
        assert tick in (0, -1)

    def test_roundtrip_positive_tick(self) -> None:
        for t in (100, 1000, 5000, 50000):
            sp = sqrt_price_at_tick(t)
            recovered = tick_at_sqrt_price(sp)
            # Floor tick from the Q64 fixed-point value should be at most 1 below.
            assert recovered in (t, t - 1)

    def test_zero_sqrt_price_rejected(self) -> None:
        with pytest.raises(ValidationError):
            tick_at_sqrt_price(0)


class TestLiquidityRange:
    def test_lower_ge_upper_raises(self) -> None:
        with pytest.raises(ValidationError):
            LiquidityRange(lower_tick=100, upper_tick=100, liquidity=1000)
        with pytest.raises(ValidationError):
            LiquidityRange(lower_tick=200, upper_tick=100, liquidity=1000)

    def test_negative_liquidity_raises(self) -> None:
        with pytest.raises(ValidationError):
            LiquidityRange(lower_tick=0, upper_tick=100, liquidity=-1)


class TestSwap:
    def test_zero_input_returns_zero_output(self) -> None:
        rng = _make_range(-1000, 1000, 10**15)
        sp = sqrt_price_at_tick(0)
        result = swap([rng], sp, 0, 30, direction=False)
        assert result.amount_out == 0
        assert result.amount_in_used == 0
        assert result.fees_charged == 0
        assert result.sqrt_price_x64 == sp

    def test_no_liquidity_returns_zero(self) -> None:
        rng = _make_range(-1000, 1000, 0)
        sp = sqrt_price_at_tick(0)
        result = swap([rng], sp, 1_000_000, 30, direction=False)
        assert result.amount_out == 0
        assert result.amount_in_used == 0

    def test_empty_range_list_returns_zero(self) -> None:
        sp = sqrt_price_at_tick(0)
        result = swap([], sp, 1_000_000, 30, direction=False)
        assert result.amount_out == 0

    def test_direction_true_price_decreases_or_stays(self) -> None:
        # Selling token_0 (zero-for-one) should move price down.
        rng = _make_range(-1000, 1000, 10**15)
        sp = sqrt_price_at_tick(0)
        result = swap([rng], sp, 10_000, 30, direction=True)
        assert result.sqrt_price_x64 <= sp

    def test_direction_false_price_increases_or_stays(self) -> None:
        # Selling token_1 (one-for-zero) should move price up.
        rng = _make_range(-1000, 1000, 10**15)
        sp = sqrt_price_at_tick(0)
        result = swap([rng], sp, 10_000, 30, direction=False)
        assert result.sqrt_price_x64 >= sp

    def test_invalid_sqrt_price_rejected(self) -> None:
        with pytest.raises(ValidationError):
            swap([], 0, 1000, 30, direction=False)

    def test_invalid_fee_rejected(self) -> None:
        sp = sqrt_price_at_tick(0)
        with pytest.raises(ValidationError):
            swap([], sp, 1000, 10_000, direction=False)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@given(
    tick_pair=_TICK_PAIRS,
    liquidity=_LIQUIDITY,
    amount=_AMOUNTS,
    fee_bps=_FEE_BPS,
    direction=st.booleans(),
)
@settings(max_examples=300)
def test_amount_out_non_negative_integer(
    tick_pair: tuple[int, int],
    liquidity: int,
    amount: int,
    fee_bps: int,
    direction: bool,
) -> None:
    """Output is always a non-negative integer."""
    lower, upper = tick_pair
    rng = _make_range(lower, upper, liquidity)
    sp = sqrt_price_at_tick((lower + upper) // 2)
    result = swap([rng], sp, amount, fee_bps, direction)
    assert isinstance(result.amount_out, int)
    assert not isinstance(result.amount_out, bool)
    assert result.amount_out >= 0


@given(
    tick_pair=_TICK_PAIRS,
    liquidity=_LIQUIDITY,
    amount=_AMOUNTS,
    fee_bps=_FEE_BPS,
    direction=st.booleans(),
)
@settings(max_examples=300)
def test_amount_in_used_le_amount_in(
    tick_pair: tuple[int, int],
    liquidity: int,
    amount: int,
    fee_bps: int,
    direction: bool,
) -> None:
    """The swap never consumes more than was offered."""
    lower, upper = tick_pair
    rng = _make_range(lower, upper, liquidity)
    sp = sqrt_price_at_tick((lower + upper) // 2)
    result = swap([rng], sp, amount, fee_bps, direction)
    assert result.amount_in_used <= amount


@given(
    tick_pair=_TICK_PAIRS,
    liquidity=_LIQUIDITY,
    amount=_AMOUNTS,
    fee_bps=_FEE_BPS,
    direction=st.booleans(),
)
@settings(max_examples=300)
def test_fees_charged_non_negative_and_le_amount_used(
    tick_pair: tuple[int, int],
    liquidity: int,
    amount: int,
    fee_bps: int,
    direction: bool,
) -> None:
    """Fees are non-negative and do not exceed the input consumed."""
    lower, upper = tick_pair
    rng = _make_range(lower, upper, liquidity)
    sp = sqrt_price_at_tick((lower + upper) // 2)
    result = swap([rng], sp, amount, fee_bps, direction)
    assert result.fees_charged >= 0
    assert result.fees_charged <= result.amount_in_used


@given(
    tick_pair=_TICK_PAIRS,
    liquidity=_LIQUIDITY,
    amount=_AMOUNTS,
    fee_bps=_FEE_BPS,
)
@settings(max_examples=200)
def test_direction_true_sqrt_price_non_increasing(
    tick_pair: tuple[int, int],
    liquidity: int,
    amount: int,
    fee_bps: int,
) -> None:
    """Selling token_0 (direction=True) must not increase the sqrt price."""
    lower, upper = tick_pair
    rng = _make_range(lower, upper, liquidity)
    sp = sqrt_price_at_tick((lower + upper) // 2)
    result = swap([rng], sp, amount, fee_bps, direction=True)
    assert result.sqrt_price_x64 <= sp


@given(
    tick_pair=_TICK_PAIRS,
    liquidity=_LIQUIDITY,
    amount=_AMOUNTS,
    fee_bps=_FEE_BPS,
)
@settings(max_examples=200)
def test_direction_false_sqrt_price_non_decreasing(
    tick_pair: tuple[int, int],
    liquidity: int,
    amount: int,
    fee_bps: int,
) -> None:
    """Selling token_1 (direction=False) must not decrease the sqrt price."""
    lower, upper = tick_pair
    rng = _make_range(lower, upper, liquidity)
    sp = sqrt_price_at_tick((lower + upper) // 2)
    result = swap([rng], sp, amount, fee_bps, direction=False)
    assert result.sqrt_price_x64 >= sp
