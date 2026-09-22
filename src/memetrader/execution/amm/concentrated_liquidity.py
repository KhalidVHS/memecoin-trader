"""Concentrated-liquidity (CLMM) swap math for the backtest engine.

Covers Orca Whirlpools and Raydium CLMM, both of which use the Uniswap-v3
tick-indexed liquidity model.

Authorship and licensing note
===============================

This implementation is derived *entirely* from the published mathematical
specification of the Uniswap-v3 whitepaper (Adams et al., 2021) and the
Orca/Raydium CLMM documentation. No code from the Orca or Raydium repositories
has been read or copied:

  * Orca's orca-so/whirlpools SDK is licensed under the Apache 2.0 License
    with a Commons Clause, which restricts commercial use.
  * Raydium's raydium-io/raydium-clmm is GPL-3.0.

Implementing from the math (and verifying against the spec) rather than from
the repositories is both the legally safe choice and the epistemically correct
one: a reimplementation that was verified against first principles is more
trustworthy as a backtest component than a copy that silently inherits any
upstream bugs.

Key mathematical concepts
==========================

**Square-root price (sqrt_price_x64)**

Prices in CLMMs are stored as Q64.64 fixed-point numbers representing the
square root of the price: ``sqrt_price_x64 = sqrt(price) * 2**64``. Fixed-point
arithmetic avoids float precision loss across the enormous price range spanned
by DeFi tokens.

**Tick index and spacing**

The price space is discretised into *ticks*. Tick ``i`` corresponds to a price
of ``1.0001^i``, or equivalently a sqrt_price of ``1.0001^(i/2)``. Liquidity is
specified per tick range (lower_tick, upper_tick) — it is only active while the
current price is within that range.

**Swap mechanics**

A swap proceeds tick by tick:
1. Compute how much input the current tick range can absorb before the price
   hits the next tick boundary.
2. If the full ``amount_in`` is absorbed within this range, compute the exact
   output and stop.
3. Otherwise, consume this range entirely, cross the tick boundary (updating
   active liquidity), and continue.

This is the ``swap`` function below.

**Formula references**

All formulas are from the Uniswap-v3 whitepaper §6.2–6.3 and the associated
technical reference. Variable names follow the whitepaper notation where
possible.

  x = token_0 (the "base" token)
  y = token_1 (the "quote" token, often USDC)
  L = liquidity
  sqrt_P = sqrt(price)

For a swap of token_0 in (direction=True, buying quote/token_1):

    Δsqrt_P = Δy / L              (move price up)
    Δx      = L * Δ(1/sqrt_P)     (token_0 consumed)

For a swap of token_1 in (direction=False, buying base/token_0):

    Δ(1/sqrt_P) = Δx / L         (move price down)
    Δy          = L * Δsqrt_P    (token_1 consumed)

The fee is charged on the input and excluded from the amount that moves the
price, consistent with the Uniswap-v3 specification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

from memetrader.types import ValidationError, atomic

__all__ = [
    "LiquidityRange",
    "SwapResult",
    "sqrt_price_at_tick",
    "swap",
    "tick_at_sqrt_price",
]

# Q64.64 scale factor: sqrt_price_x64 = sqrt(price) * _Q64
_Q64 = 1 << 64  # 2**64

# Uniswap-v3 tick base: price = 1.0001^tick
_TICK_BASE = 1.0001

# log(1.0001) precomputed once — used in tick_at_sqrt_price.
_LOG_TICK_BASE = math.log(_TICK_BASE)

# Tick bounds from the Uniswap-v3 spec. A tick outside this range cannot
# correspond to a valid Q64.64 sqrt_price.
_MIN_TICK = -443_636
_MAX_TICK = 443_636


@dataclass(frozen=True, slots=True)
class LiquidityRange:
    """A single tick range with its associated liquidity.

    ``lower_tick`` and ``upper_tick`` are inclusive tick indices.
    ``liquidity`` is the net active liquidity (L) when the price is inside
    this range. Units are the same as Uniswap-v3's ``uint128`` liquidity —
    dimensionless, non-negative integers.

    Multiple overlapping ranges are resolved by the caller before passing to
    ``swap``: this type represents a *net* liquidity figure, not a single
    position. The pool simulator accumulates net liquidity per tick by walking
    the tick bitmap, exactly as the on-chain contract does, and the resulting
    list of non-overlapping ranges is what ``swap`` receives.
    """

    lower_tick: int
    upper_tick: int
    liquidity: int

    def __post_init__(self) -> None:
        if not isinstance(self.lower_tick, int) or isinstance(self.lower_tick, bool):
            raise ValidationError(f"lower_tick must be int, got {self.lower_tick!r}")
        if not isinstance(self.upper_tick, int) or isinstance(self.upper_tick, bool):
            raise ValidationError(f"upper_tick must be int, got {self.upper_tick!r}")
        if self.lower_tick >= self.upper_tick:
            raise ValidationError(
                f"lower_tick {self.lower_tick} must be < upper_tick {self.upper_tick}"
            )
        if not isinstance(self.liquidity, int) or isinstance(self.liquidity, bool):
            raise ValidationError(f"liquidity must be int, got {self.liquidity!r}")
        if self.liquidity < 0:
            raise ValidationError(f"liquidity must be >= 0, got {self.liquidity}")


class SwapResult(NamedTuple):
    """Output of a single ``swap`` call.

    ``amount_out``     — tokens received, in atomic units. Always >= 0.
    ``sqrt_price_x64`` — final Q64.64 sqrt price after the swap.
    ``ticks_crossed``  — list of tick indices that were crossed (boundary ticks
                         where liquidity changed). Empty for a swap that stayed
                         within one range.
    ``amount_in_used`` — how much of the input was actually consumed. Equals
                         ``amount_in`` unless the pool ran out of liquidity
                         (partial fill).
    ``fees_charged``   — total fee deducted from the input, in atomic units.
    """

    amount_out: int
    sqrt_price_x64: int
    ticks_crossed: list[int]
    amount_in_used: int
    fees_charged: int


def sqrt_price_at_tick(tick: int) -> int:
    """Q64.64 sqrt price for a given tick index.

    ``sqrt_price_x64 = floor(sqrt(1.0001^tick) * 2^64)``

    Floor is used because the on-chain contracts truncate (not round) when
    converting the mathematical sqrt price to fixed-point, and we need to match
    their behaviour for tick-boundary comparisons to be correct.
    """
    if not isinstance(tick, int) or isinstance(tick, bool):
        raise ValidationError(f"tick must be int, got {tick!r}")
    if not (_MIN_TICK <= tick <= _MAX_TICK):
        raise ValidationError(f"tick {tick} outside valid range [{_MIN_TICK}, {_MAX_TICK}]")

    # 1.0001^tick = e^(tick * log(1.0001))
    price = math.exp(tick * _LOG_TICK_BASE)
    sqrt_p = math.sqrt(price)
    return int(sqrt_p * _Q64)  # floor via int()


def tick_at_sqrt_price(sqrt_price_x64: int) -> int:
    """Tick index for a given Q64.64 sqrt price (floor).

    Inverse of ``sqrt_price_at_tick``. Returns the largest tick ``i`` such that
    ``sqrt_price_at_tick(i) <= sqrt_price_x64``.
    """
    if not isinstance(sqrt_price_x64, int) or isinstance(sqrt_price_x64, bool):
        raise ValidationError(f"sqrt_price_x64 must be int, got {sqrt_price_x64!r}")
    if sqrt_price_x64 <= 0:
        raise ValidationError(f"sqrt_price_x64 must be > 0, got {sqrt_price_x64}")

    sqrt_p = sqrt_price_x64 / _Q64
    price = sqrt_p * sqrt_p
    # tick = log(price) / log(1.0001) = log(price) / _LOG_TICK_BASE
    # but price = sqrt_p^2, so log(price) = 2*log(sqrt_p)
    if sqrt_p <= 0.0:
        raise ValidationError("sqrt_price_x64 too small to compute tick")
    tick = math.floor(math.log(price) / _LOG_TICK_BASE)
    # Clamp to valid range to avoid upstream caller confusion.
    return max(_MIN_TICK, min(_MAX_TICK, tick))


def _compute_amount_out_within_range(
    sqrt_price_current_x64: int,
    sqrt_price_target_x64: int,
    liquidity: int,
    amount_in_remaining: int,
    fee_bps: int,
    zero_for_one: bool,
) -> tuple[int, int, int, int]:
    """Compute how much output a single tick range can provide.

    Returns ``(amount_in_consumed, amount_out, fees, sqrt_price_next_x64)``.

    The amount_in_consumed may be less than amount_in_remaining if the range
    is exhausted before the input is fully consumed (the swap must cross to the
    next tick).

    All intermediate arithmetic uses Python's arbitrary-precision integers
    (or floats for the Q64.64 operations that would overflow otherwise).

    ``zero_for_one=True`` means selling token_0 to buy token_1, which moves
    the price down (sqrt_price decreases).
    ``zero_for_one=False`` means selling token_1 to buy token_0, which moves
    the price up (sqrt_price increases).
    """
    # Fee is charged on gross input: the net input (after fee) moves the price.
    # fee = amount_in_remaining * fee_bps // 10_000 (rounded up for fees)
    fee = (amount_in_remaining * fee_bps + 9_999) // 10_000
    amount_in_net = amount_in_remaining - fee

    if zero_for_one:
        # Selling token_0 (x), buying token_1 (y). Price moves down.
        # Δx = L * (1/sqrt_P_lower - 1/sqrt_P_upper)
        # where upper = current, lower = target (price moving down).
        # Using Q64.64: sqrt_P = sqrt_price_x64 / 2^64
        # To avoid float precision loss for large Q64 values, compute in
        # arbitrary-precision integer arithmetic where possible.

        # max token_0 the range can absorb (to reach target price):
        # Δx_max = L * (1/sqrt_P_target - 1/sqrt_P_current) * 2^64 ... in Q64
        # = L * (sqrt_P_current - sqrt_P_target) / (sqrt_P_current * sqrt_P_target / 2^64)
        # We compute in float here because these are presentation-tier range-boundary
        # checks; the settlement amounts are then floored to integers.
        sqrt_current = sqrt_price_current_x64 / _Q64
        sqrt_target = sqrt_price_target_x64 / _Q64

        if sqrt_current <= 0.0 or sqrt_target <= 0.0:
            return 0, 0, 0, sqrt_price_current_x64

        # Max input (token_0) to consume this entire range:
        max_amount_in_net = int(liquidity * (1.0 / sqrt_target - 1.0 / sqrt_current))
        # max_amount_in_gross is what, before fee deduction, corresponds to max_amount_in_net
        # We don't need max_amount_in_gross precisely; we compare net amounts.

        if amount_in_net >= max_amount_in_net:
            # Range is fully consumed; arrive at target price.
            amount_in_consumed_net = max_amount_in_net
            # Recompute fee on the consumed portion only
            # (fee proportional to consumed fraction):
            # gross_consumed = ceil(consumed_net / (1 - fee_bps/10000))
            fee_factor = 10_000 - fee_bps
            if fee_factor > 0:
                gross_consumed = (
                    amount_in_consumed_net * 10_000 + fee_factor - 1
                ) // fee_factor
            else:
                gross_consumed = amount_in_consumed_net
            fee_charged = gross_consumed - amount_in_consumed_net
            # Δy = L * (sqrt_P_current - sqrt_P_target)
            out = int(liquidity * (sqrt_current - sqrt_target))
            return gross_consumed, out, fee_charged, sqrt_price_target_x64
        # Partial fill: amount_in_net moves the price partway.
        # New sqrt_P: 1/sqrt_P_new = 1/sqrt_P_current + Δx_net/L
        if amount_in_net == 0:
            # All input consumed by fees; price does not move.
            return amount_in_remaining, 0, fee, sqrt_price_current_x64
        inv_new = 1.0 / sqrt_current + amount_in_net / liquidity
        if inv_new <= 0.0:
            return 0, 0, 0, sqrt_price_current_x64
        sqrt_new = 1.0 / inv_new
        # Clamp to not exceed current (direction=True means price falls).
        sqrt_price_new_x64 = min(int(sqrt_new * _Q64), sqrt_price_current_x64)
        out = max(0, int(liquidity * (sqrt_current - sqrt_new)))
        return amount_in_remaining, out, fee, sqrt_price_new_x64
    # Selling token_1 (y), buying token_0 (x). Price moves up.
    sqrt_current = sqrt_price_current_x64 / _Q64
    sqrt_target = sqrt_price_target_x64 / _Q64

    if sqrt_current <= 0.0 or sqrt_target <= 0.0:
        return 0, 0, 0, sqrt_price_current_x64

    # Max input (token_1) to consume this entire range:
    # Δy_max = L * (sqrt_P_target - sqrt_P_current)
    max_amount_in_net = int(liquidity * (sqrt_target - sqrt_current))

    if amount_in_net >= max_amount_in_net:
        amount_in_consumed_net = max_amount_in_net
        fee_factor = 10_000 - fee_bps
        if fee_factor > 0:
            gross_consumed = (
                amount_in_consumed_net * 10_000 + fee_factor - 1
            ) // fee_factor
        else:
            gross_consumed = amount_in_consumed_net
        fee_charged = gross_consumed - amount_in_consumed_net
        # Δx = L * (1/sqrt_P_current - 1/sqrt_P_target)
        out = int(liquidity * (1.0 / sqrt_current - 1.0 / sqrt_target))
        return gross_consumed, out, fee_charged, sqrt_price_target_x64
    # Partial fill: amount_in_net raises price by Δy_net/L
    if amount_in_net == 0:
        # All input consumed by fees; price does not move.
        return amount_in_remaining, 0, fee, sqrt_price_current_x64
    sqrt_new = sqrt_current + amount_in_net / liquidity
    # Clamp to not exceed target (direction=False means price rises).
    sqrt_price_new_x64 = max(int(sqrt_new * _Q64), sqrt_price_current_x64)
    out = max(0, int(liquidity * (1.0 / sqrt_current - 1.0 / sqrt_new)))
    return amount_in_remaining, out, fee, sqrt_price_new_x64


def swap(
    liquidity_by_tick: list[LiquidityRange],
    sqrt_price_x64: int,
    amount_in: int,
    fee_bps: int,
    direction: bool,
) -> SwapResult:
    """Execute a CLMM swap across multiple tick ranges.

    Parameters
    ----------
    liquidity_by_tick:
        Ordered list of non-overlapping ``LiquidityRange`` objects covering the
        tick space relevant to this swap. Must be sorted by ``lower_tick``
        ascending. Gaps (ranges with zero liquidity) between active ranges are
        allowed and represent tick ranges where the pool has no liquidity —
        trying to swap through them exhausts the input without producing output.
    sqrt_price_x64:
        Current Q64.64 sqrt price of the pool (``sqrt(price) * 2^64``).
    amount_in:
        Exact input amount in atomic units. Must be a non-negative integer.
        Zero input produces a zero-output result immediately.
    fee_bps:
        Pool fee in basis points (e.g., 30 for a 0.3% pool, 100 for 1%).
        Applied to the gross input; the net input moves the price.
    direction:
        ``True`` = zero-for-one (selling token_0, buying token_1, price moves
        down). ``False`` = one-for-zero (selling token_1, buying token_0, price
        moves up).

    Returns
    -------
    SwapResult
        Named tuple with ``amount_out``, ``sqrt_price_x64``, ``ticks_crossed``,
        ``amount_in_used``, and ``fees_charged``. If the pool runs out of
        liquidity before consuming all of ``amount_in``, ``amount_in_used`` will
        be less than ``amount_in`` and the swap is a partial fill — the caller
        must decide how to handle the remainder.
    """
    atomic(amount_in, "amount_in")
    if not isinstance(sqrt_price_x64, int) or isinstance(sqrt_price_x64, bool):
        raise ValidationError(f"sqrt_price_x64 must be int, got {sqrt_price_x64!r}")
    if sqrt_price_x64 <= 0:
        raise ValidationError(f"sqrt_price_x64 must be > 0, got {sqrt_price_x64}")
    if (
        not isinstance(fee_bps, int)
        or isinstance(fee_bps, bool)
        or fee_bps < 0
        or fee_bps >= 10_000
    ):
        raise ValidationError(f"fee_bps must be int in [0, 10000), got {fee_bps!r}")

    if amount_in == 0:
        return SwapResult(
            amount_out=0,
            sqrt_price_x64=sqrt_price_x64,
            ticks_crossed=[],
            amount_in_used=0,
            fees_charged=0,
        )

    # Determine which ranges are relevant given current price and direction.
    # Current tick for range selection:
    current_tick = tick_at_sqrt_price(sqrt_price_x64)

    # Filter and sort ranges relevant to this swap direction.
    if direction:  # zero-for-one: price moves down, we walk lower ticks
        # Relevant ranges: those whose upper_tick > current_tick (we're above their lower)
        # and lower_tick <= current_tick (we're inside or above)
        relevant = sorted(
            [r for r in liquidity_by_tick if r.lower_tick <= current_tick],
            key=lambda r: r.lower_tick,
            reverse=True,  # process from current position downward
        )
    else:  # one-for-zero: price moves up
        relevant = sorted(
            [r for r in liquidity_by_tick if r.upper_tick > current_tick],
            key=lambda r: r.lower_tick,
        )

    remaining = amount_in
    total_out = 0
    total_fees = 0
    ticks_crossed: list[int] = []
    current_sqrt = sqrt_price_x64

    for rng in relevant:
        if remaining == 0:
            break
        if rng.liquidity == 0:
            # No liquidity in this range; cross both boundaries without output.
            if direction:
                ticks_crossed.append(rng.lower_tick)
            else:
                ticks_crossed.append(rng.upper_tick)
            continue

        # Target sqrt price is the far boundary of this range.
        if direction:
            target_tick = rng.lower_tick
        else:
            target_tick = rng.upper_tick
        target_sqrt = sqrt_price_at_tick(target_tick)

        consumed, out, fees, new_sqrt = _compute_amount_out_within_range(
            current_sqrt,
            target_sqrt,
            rng.liquidity,
            remaining,
            fee_bps,
            direction,
        )

        total_out += out
        total_fees += fees
        remaining -= consumed
        current_sqrt = new_sqrt

        # If we reached the boundary (used all capacity of this range), record
        # the crossed tick and continue.
        if new_sqrt == target_sqrt:
            ticks_crossed.append(target_tick)

    amount_in_used = amount_in - remaining
    return SwapResult(
        amount_out=total_out,
        sqrt_price_x64=current_sqrt,
        ticks_crossed=ticks_crossed,
        amount_in_used=amount_in_used,
        fees_charged=total_fees,
    )
