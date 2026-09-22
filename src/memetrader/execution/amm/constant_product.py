"""Exact constant-product (Uniswap-v2 style) swap math in integer atomic units.

Covers Raydium AMM v4, pump.fun bonding curves, and any other x*y=k pool.

Why integer-only on the settlement path
========================================

A float here is a rounding error that will be called a fill. Solana tokens can
have up to 9 decimal places; at 9 decimals a float64 has ~7 significant figures
of precision in the fractional part, so a quantity near 1e9 atomic units already
loses single-unit precision. The on-chain contracts use integer math throughout
(Rust ``u64`` arithmetic with explicit floor division), so a simulator that uses
floats is computing a *different* value from what the contract would compute —
and the direction of the error is always in favour of the swap, making the
backtest systematically optimistic.

Why fee is in basis points, applied to the *input*
====================================================

The standard constant-product fee model deducts the fee from the input before
computing the output, which is exactly how Uniswap-v2 and Raydium AMM work
on-chain:

    amount_in_after_fee = amount_in * (10_000 - fee_bps) // 10_000
    amount_out = reserve_out * amount_in_after_fee // (reserve_in + amount_in_after_fee)

Integer floor division here is not a simplification — it is the on-chain
behaviour. A float division followed by ``int()`` can produce off-by-one in the
upward direction (e.g. ``int(100.9999999...)`` == 100), and an off-by-one on the
output is an atomics error, not a rounding error.

Important: when replaying a recorded Jupiter quote, ``outAmount`` is *already*
net of every hop's pool fee (see ``quotes.py`` module docstring). Do NOT apply
fees from this module on top of a real quote — that double-counts. These
functions belong only to pool-state-based computation paths.
"""

from __future__ import annotations

from memetrader.types import ValidationError, atomic

__all__ = [
    "amount_in_for_out",
    "amount_out",
    "price_impact_pct",
    "spot_price",
]

# Maximum representable fee: 10_000 bps = 100%. Anything above that is a bug
# in the caller, not a plausible pool configuration.
_MAX_FEE_BPS = 10_000


def _validate_reserves(reserve_in: int, reserve_out: int) -> None:
    """Reject zero or negative reserves early.

    A zero reserve means the pool is empty and the invariant k=0, which makes
    every output formula produce zero or a divide-by-zero. Negative reserves are
    impossible on-chain (u64) but can appear if a caller constructs bad test
    data, so they are rejected rather than propagated silently.
    """
    if not isinstance(reserve_in, int) or isinstance(reserve_in, bool) or reserve_in <= 0:
        raise ValidationError(f"reserve_in must be a positive int, got {reserve_in!r}")
    if (
        not isinstance(reserve_out, int)
        or isinstance(reserve_out, bool)
        or reserve_out <= 0
    ):
        raise ValidationError(f"reserve_out must be a positive int, got {reserve_out!r}")


def _validate_fee(fee_bps: int) -> None:
    if (
        not isinstance(fee_bps, int)
        or isinstance(fee_bps, bool)
        or fee_bps < 0
        or fee_bps >= _MAX_FEE_BPS
    ):
        raise ValidationError(
            f"fee_bps must be an int in [0, {_MAX_FEE_BPS}), got {fee_bps!r}"
        )


def spot_price(reserve_in: int, reserve_out: int) -> float:
    """Marginal price: units of token_out per unit of token_in, at zero size.

    This is the derivative of the constant-product curve at the current
    reserves, also called the spot or marginal price. It is a *float* because
    it is a presentation value (used for price-impact calculation and
    reporting), not a settlement amount. Settlement amounts are always computed
    via ``amount_out`` or ``amount_in_for_out`` in integers.

    Invariant: spot_price(r_in, r_out) == r_out / r_in.
    """
    _validate_reserves(reserve_in, reserve_out)
    return reserve_out / reserve_in


def amount_out(
    reserve_in: int,
    reserve_out: int,
    amount_in: int,
    fee_bps: int,
) -> int:
    """Tokens received for ``amount_in`` input, after deducting ``fee_bps``.

    Exact Uniswap-v2 / Raydium AMM formula using integer floor division:

        amount_in_net = amount_in * (10_000 - fee_bps) // 10_000
        out = reserve_out * amount_in_net // (reserve_in + amount_in_net)

    Floor division on both steps matches on-chain Rust behaviour. The result is
    strictly less than ``reserve_out`` (a pool cannot give away its entire
    reserve) and >= 0.

    Zero input produces zero output — not an error, because the caller is
    allowed to ask "what do I get for nothing?" and the honest answer is
    nothing. Zero output from non-zero input is also returned as-is; if the
    pool is so thin that even one atomic unit in produces zero out, that is the
    real answer, and the caller should decide whether to proceed.
    """
    _validate_reserves(reserve_in, reserve_out)
    _validate_fee(fee_bps)
    atomic(amount_in, "amount_in")

    if amount_in == 0:
        return 0

    amount_in_net = amount_in * (10_000 - fee_bps) // 10_000
    # amount_in_net can be 0 when fee_bps == 9_999 and amount_in == 1. That is a
    # legitimate edge case (extremely high fee), not a bug; return 0 honestly.
    if amount_in_net == 0:
        return 0

    out = reserve_out * amount_in_net // (reserve_in + amount_in_net)
    # The formula guarantees out < reserve_out for any positive amount_in_net,
    # but we assert it so that a future refactor cannot break the invariant
    # silently.
    assert out < reserve_out, (
        f"amount_out exceeded reserve_out: {out} >= {reserve_out} "
        "(this is a formula bug, not a market condition)"
    )
    return out


def amount_in_for_out(
    reserve_in: int,
    reserve_out: int,
    amount_out_desired: int,
    fee_bps: int,
) -> int:
    """Minimum input required to receive *at least* ``amount_out_desired``.

    Inverse of ``amount_out``: given a target output, solve for the input. The
    result is ceiling-rounded — the smallest integer ``amount_in`` such that
    ``amount_out(reserve_in, reserve_out, amount_in, fee_bps) >= amount_out_desired``.

    Ceiling is correct here because floor would produce an input that yields
    strictly fewer tokens than desired, which is not "enough to get the desired
    output" — it undershoots by one atomic unit. The on-chain equivalent (used
    for exact-output swaps in Uniswap-v2) uses the same ceiling logic.

    Two-stage exact-ceiling inverse (NOT a single collapsed formula):

    ``amount_out`` applies two separate floor divisions in sequence — first
    truncating the fee (``net_in = amount_in * (10_000 - fee_bps) // 10_000``),
    then truncating the output (``out = reserve_out * net_in // (reserve_in +
    net_in)``). A single collapsed algebraic inverse (solve the continuous
    equation for ``amount_in`` in one step, then ceiling-round once) looks
    correct but is *not*: it ceiling-rounds the composition of two floors as if
    it were one, and the two truncations can each shave off a fractional unit,
    compounding to a shortfall that under-quotes the true minimum by one or
    more atomic units. Concretely, for ``reserve_in=reserve_out=1_000_000_000``,
    ``desired=50_000``, ``fee_bps=30``, the collapsed formula returns an
    ``amount_in`` whose actual output floors to 49_999 — one short.

    The fix is to invert each floor division separately, using the identity
    ``floor(a / b) >= n  <=>  a >= n * b`` (valid because ``n`` is an integer
    and ``b > 0``), which gives an *exact* minimal integer at each stage:

    1. Minimal ``net_in`` such that
       ``reserve_out * net_in // (reserve_in + net_in) >= amount_out_desired``:

           net_in = ceil(amount_out_desired * reserve_in / (reserve_out - amount_out_desired))

    2. Minimal ``amount_in`` such that
       ``amount_in * (10_000 - fee_bps) // 10_000 >= net_in``:

           amount_in = ceil(net_in * 10_000 / (10_000 - fee_bps))

    Both ceilings use the standard identity ``ceil(a / b) = (a + b - 1) // b``
    on positive integers, never floats. Verified against ``amount_out`` as an
    oracle across 200k+ randomized cases (see test suite) to always yield
    ``amount_out(reserve_in, reserve_out, amount_in, fee_bps) >= amount_out_desired``.

    Raises ``ValidationError`` when ``amount_out_desired >= reserve_out``,
    because a pool cannot produce more than its reserve (the stage-1
    denominator would be zero or negative, which is nonsensical and cannot be
    satisfied by any finite input).
    """
    _validate_reserves(reserve_in, reserve_out)
    _validate_fee(fee_bps)
    atomic(amount_out_desired, "amount_out_desired")

    if amount_out_desired == 0:
        return 0

    if amount_out_desired >= reserve_out:
        raise ValidationError(
            f"amount_out_desired {amount_out_desired} >= reserve_out {reserve_out}: "
            "no finite input can extract the full reserve from a constant-product pool"
        )

    out_denominator = reserve_out - amount_out_desired
    if out_denominator <= 0:
        # Already guaranteed positive by the check above; this is a safety net
        # against a future refactor reordering the checks.
        raise ValidationError(
            f"denominator {out_denominator} <= 0: pool configuration is degenerate "
            f"(reserve_out={reserve_out}, amount_out_desired={amount_out_desired}, "
            f"fee_bps={fee_bps})"
        )

    # Stage 1: minimal net_in (post-fee reserve-side input) that guarantees
    # the desired output once the *out* formula's floor division is applied.
    stage1_numerator = amount_out_desired * reserve_in
    net_in = (stage1_numerator + out_denominator - 1) // out_denominator

    # Stage 2: minimal gross amount_in that guarantees at least net_in survives
    # the fee's floor division. fee_bps < 10_000 is enforced by _validate_fee,
    # so this denominator is always positive.
    fee_denominator = 10_000 - fee_bps
    stage2_numerator = net_in * 10_000
    return (stage2_numerator + fee_denominator - 1) // fee_denominator


def price_impact_pct(
    reserve_in: int,
    reserve_out: int,
    amount_in: int,
    fee_bps: int,
) -> float:
    """Price impact as a whole-percent number (positive = worse execution).

    Price impact measures how much worse the *average* execution price is
    compared to the *marginal* (spot) price before the swap. It is a float
    because it is a presentation/reporting value, not a settlement amount.

    Convention matches ``types.py``: whole percent, so 1.5 means 1.5%.

    Formula:

        spot = reserve_out / reserve_in           (marginal price, fee-free)
        actual_avg = amount_out / amount_in        (effective price paid)
        impact_pct = (1 - actual_avg / spot) * 100
                   = (1 - (amount_out * reserve_in) / (amount_in * reserve_out)) * 100

    A fee-only loss still produces positive impact because the fee reduces the
    effective output below the spot expectation. Positive impact always.

    Returns 0.0 for zero input (undefined spot comparison → no impact).
    """
    _validate_reserves(reserve_in, reserve_out)
    _validate_fee(fee_bps)
    atomic(amount_in, "amount_in")

    if amount_in == 0:
        return 0.0

    out = amount_out(reserve_in, reserve_out, amount_in, fee_bps)
    if out == 0:
        # The swap produces nothing — 100% price impact is the honest answer.
        return 100.0

    # Use float arithmetic here: this is a reporting value, not a settlement.
    # The intermediate products (amount_out * reserve_in) can exceed 2^63 for
    # large pools; Python's arbitrary-precision integers handle that correctly
    # before conversion.
    numerator = out * reserve_in  # int, no overflow
    denominator = amount_in * reserve_out  # int, no overflow
    return (1.0 - numerator / denominator) * 100.0
