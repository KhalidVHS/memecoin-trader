"""Reconstruct a multi-hop swap across historical pool states.

This module answers the question: "given what the pools looked like at time T,
what would this route have produced?"  It does so by applying each hop of the
route in sequence using ``constant_product.amount_out``, accumulating fees
(only when computing from pool reserves — NOT when replaying a recorded Jupiter
quote), and computing end-to-end price impact and ``min_out`` at a given
slippage tolerance.

Why provenance labels matter
==============================

There are two fundamentally different things a caller can hand this module:

1. **A route Jupiter was observed to take** — the ``routePlan`` from a real
   recorded quote.  The recorded ``outAmount`` is already the ground truth for
   that instant; this module can *verify* the hop sequence is plausible but
   should not override what was actually observed.

2. **A candidate route we are hypothesising** — a sequence of pools chosen by
   a routing algorithm that was never executed.  The computed output is a model
   prediction, not evidence, and any downstream attribution that treats it as
   evidence has introduced look-ahead (the route might not have been available,
   might have been quoted differently, etc.).

These two cases are labelled ``RouteProvenance.HISTORICAL`` and
``RouteProvenance.COUNTERFACTUAL`` respectively, and the label lives on
``RouteResult`` so that every downstream consumer can separate them.  A
backtest that collapses this distinction will silently overstate the strategy's
access to good routes, which is the most flattering direction an execution
realism error can take.

Why "no route" produces no trade
==================================

Audit §6 (C4 equivalent for the backtest): a route that is unavailable must
not become a synthetic fill.  ``replay_route`` returns ``None`` — not a
degraded ``RouteResult``, not a zero-output ``RouteResult`` — when any hop
fails or the route is otherwise unusable.  Callers must check for ``None``
before doing anything with the result.

Fee double-counting hazard
===========================

When the input is a recorded Jupiter quote, ``outAmount`` is already net of
every hop's fee.  Applying fees here on top of a real quote's amounts is the
double-count the ``quotes.py`` module docstring warns about.  ``replay_route``
accepts an ``apply_hop_fees`` flag: set it ``False`` when the pool states are
being used only to *verify* plausibility of a recorded quote, ``True`` when
computing a fresh forward simulation from pool reserves.  The default is
``True`` because the common use is forward simulation; callers replaying a
recorded route should pass ``apply_hop_fees=False``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

from memetrader.ids import quote_fingerprint
from memetrader.types import (
    Quote,
    Side,
    TokenMeta,
    ValidationError,
    atomic,
)

from .constant_product import amount_out as cp_amount_out
from .constant_product import price_impact_pct as cp_price_impact_pct

__all__ = [
    "Hop",
    "RouteProvenance",
    "RouteResult",
    "replay_route",
]


class RouteProvenance(StrEnum):
    """Whether a replayed route was observed in the historical record or inferred.

    ``HISTORICAL``    — every hop of this route was seen in a recorded Jupiter
                        quote at approximately this timestamp.  The ``outAmount``
                        we compute is a re-derivation for verification, not a new
                        claim; the ground truth is the recorded quote.

    ``COUNTERFACTUAL`` — we composed this route from pool states that existed at
                        the time, but no evidence exists that Jupiter would have
                        chosen it, that it was available, or that it would have
                        been priced exactly as computed.  Use for sensitivity
                        analysis, not for PnL attribution.
    """

    HISTORICAL = "historical"
    COUNTERFACTUAL = "counterfactual"


@dataclass(frozen=True, slots=True)
class Hop:
    """One pool in a multi-hop route.

    ``reserve_in`` and ``reserve_out`` are the pool reserves *before* this hop,
    in atomic units of each token respectively.  They must come from the
    historical ``PoolState`` snapshot at the replay timestamp — using current
    reserves would introduce look-forward.

    ``fee_bps`` is the pool's fee tier (e.g., 25 for 0.25%, 30 for 0.3%).

    ``input_token`` and ``output_token`` identify the tokens on each leg.  For
    a multi-hop route the output token of hop N must equal the input token of
    hop N+1; ``replay_route`` validates this and refuses if the route is
    disconnected.
    """

    reserve_in: int
    reserve_out: int
    fee_bps: int
    input_token: TokenMeta
    output_token: TokenMeta

    def __post_init__(self) -> None:
        if (
            not isinstance(self.reserve_in, int)
            or isinstance(self.reserve_in, bool)
            or self.reserve_in <= 0
        ):
            raise ValidationError(
                f"reserve_in must be a positive int, got {self.reserve_in!r}"
            )
        if (
            not isinstance(self.reserve_out, int)
            or isinstance(self.reserve_out, bool)
            or self.reserve_out <= 0
        ):
            raise ValidationError(
                f"reserve_out must be a positive int, got {self.reserve_out!r}"
            )
        if (
            not isinstance(self.fee_bps, int)
            or isinstance(self.fee_bps, bool)
            or self.fee_bps < 0
            or self.fee_bps >= 10_000
        ):
            raise ValidationError(
                f"fee_bps must be int in [0, 10000), got {self.fee_bps!r}"
            )


class RouteResult(NamedTuple):
    """The output of ``replay_route``.

    ``quote``           — a ``Quote``-shaped record for downstream consumption.
                          Fingerprinted via ``ids.quote_fingerprint`` so it
                          carries a unique identity and can be passed to code
                          that expects a real ``Quote``.
    ``provenance``      — ``HISTORICAL`` or ``COUNTERFACTUAL``.  Never collapse
                          these; the distinction is what separates evidence from
                          hypothesis.
    ``price_impact_pct`` — end-to-end price impact, whole percent (positive =
                          worse for the swapper), computed as the ratio of
                          actual output to zero-fee theoretical output.
    ``hops_computed``   — number of hops that were actually computed (useful for
                          debugging partial routes in testing).
    """

    quote: Quote
    provenance: RouteProvenance
    price_impact_pct: float
    hops_computed: int


def _validate_route_connectivity(hops: list[Hop]) -> None:
    """Raise if the route has disconnected token pairs.

    A hop whose output token does not match the next hop's input token is a
    route specification error, not a market condition.  Failing loudly here
    prevents a silent "the intermediate token was wrong and the amounts are
    garbage" from propagating into attribution.
    """
    for i in range(len(hops) - 1):
        if hops[i].output_token.mint != hops[i + 1].input_token.mint:
            raise ValidationError(
                f"Route is disconnected at hop {i}: output token "
                f"{hops[i].output_token.mint!r} != next input token "
                f"{hops[i + 1].input_token.mint!r}"
            )


def _zero_fee_output(hops: list[Hop], amount_in: int) -> int:
    """Compute the theoretical output with all fees set to zero.

    Used only for price-impact calculation.  A zero-fee forward pass gives the
    "ideal" output at the current reserves, which is the denominator of the
    price-impact ratio.  Never used for settlement amounts.
    """
    current = amount_in
    for hop in hops:
        current = cp_amount_out(hop.reserve_in, hop.reserve_out, current, fee_bps=0)
        if current == 0:
            return 0
    return current


def replay_route(
    hops: list[Hop],
    *,
    symbol: str,
    side: Side,
    in_amount_atomic: int,
    slippage_bps: int,
    provenance: RouteProvenance,
    apply_hop_fees: bool = True,
    now: float | None = None,
) -> RouteResult | None:
    """Reconstruct a swap across a candidate route from historical pool states.

    Returns ``None`` when the route cannot produce a trade — any hop that
    outputs zero tokens (pool too thin, reserve exhaustion, or zero input after
    fee) is treated as route failure.  A failed route must never become a
    synthetic fill; the caller must handle ``None`` explicitly.

    Parameters
    ----------
    hops:
        Ordered list of ``Hop`` objects, one per pool in the route.  Must be
        non-empty.  The first hop's ``input_token`` is the overall input; the
        last hop's ``output_token`` is the overall output.
    symbol:
        Token symbol for the traded asset (used on the resulting ``Quote``).
    side:
        ``BUY`` or ``SELL``.  Determines which leg is the traded token.
    in_amount_atomic:
        Exact input in atomic units of the first hop's input token.
    slippage_bps:
        Slippage tolerance in basis points.  Used to compute ``min_out``:
        ``min_out = out * (10_000 - slippage_bps) // 10_000``.
    provenance:
        Whether this is a ``HISTORICAL`` (observed) or ``COUNTERFACTUAL``
        (hypothesised) route.  Never override the caller's label.
    apply_hop_fees:
        If ``True`` (default), each hop's ``fee_bps`` is applied to the input
        before computing output.  Set to ``False`` when replaying a recorded
        Jupiter quote whose ``outAmount`` already nets the fees, to avoid
        double-counting.
    now:
        Epoch-seconds timestamp for the ``Quote`` timestamps.  Defaults to
        ``time.time()``.

    Returns
    -------
    RouteResult or None
        ``None`` if the route fails (any hop produces zero output, route is
        empty, or slippage computation would produce a negative ``min_out``).
        ``RouteResult`` otherwise.
    """
    if not hops:
        return None

    atomic(in_amount_atomic, "in_amount_atomic")
    if in_amount_atomic == 0:
        return None

    if (
        not isinstance(slippage_bps, int)
        or isinstance(slippage_bps, bool)
        or slippage_bps < 0
        or slippage_bps > 10_000
    ):
        raise ValidationError(
            f"slippage_bps must be int in [0, 10000], got {slippage_bps!r}"
        )

    _validate_route_connectivity(hops)

    ts = time.time() if now is None else now

    # Forward pass: apply each hop in sequence.
    current_amount = in_amount_atomic
    for hop in hops:
        effective_fee = hop.fee_bps if apply_hop_fees else 0
        out = cp_amount_out(
            hop.reserve_in,
            hop.reserve_out,
            current_amount,
            effective_fee,
        )
        if out == 0:
            # Route failure: pool is too thin or fee too high. Return None —
            # never synthesise a fill from a failed route.
            return None
        current_amount = out

    final_out = current_amount

    # min_out at the requested slippage tolerance.
    min_out = final_out * (10_000 - slippage_bps) // 10_000
    if min_out <= 0:
        # Slippage would consume the entire output — the route is unusable at
        # this tolerance.
        return None

    # End-to-end price impact: compare actual output to zero-fee theoretical.
    zero_fee_out = _zero_fee_output(hops, in_amount_atomic)
    if zero_fee_out > 0:
        # Positive impact = the swapper gets less than the zero-fee ideal.
        # Matches the convention in types.py: whole percent, positive = worse.
        e2e_impact = (1.0 - final_out / zero_fee_out) * 100.0
    else:
        e2e_impact = 100.0

    input_token = hops[0].input_token
    output_token = hops[-1].output_token

    # Determine a context_slot from now (no real slot available in replay;
    # use None and let the fingerprint use 0 internally — consistent with how
    # ids.quote_fingerprint handles None).
    fp = quote_fingerprint(
        side=str(side),
        input_mint=input_token.mint,
        output_mint=output_token.mint,
        in_amount_atomic=in_amount_atomic,
        out_amount_atomic=final_out,
        slot=None,
    )

    # Route labels: pool addresses or "hop_N" for anonymous hops.
    route_labels = tuple(
        f"{hop.input_token.mint[:8]}→{hop.output_token.mint[:8]}" for hop in hops
    )

    quote = Quote(
        symbol=symbol,
        side=side,
        input_token=input_token,
        output_token=output_token,
        in_amount_atomic=in_amount_atomic,
        out_amount_atomic=final_out,
        min_out_amount_atomic=min_out,
        price_impact_pct=e2e_impact,
        route_labels=route_labels,
        fingerprint=fp,
        requested_at=ts,
        received_at=ts,
        context_slot=None,
        expires_at=None,  # Replay quotes do not expire — they are point-in-time snapshots.
        reference_price_usd=None,
    )

    return RouteResult(
        quote=quote,
        provenance=provenance,
        price_impact_pct=e2e_impact,
        hops_computed=len(hops),
    )
