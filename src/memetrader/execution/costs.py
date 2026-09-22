"""Cost accounting for simulated execution.

Builds :class:`~memetrader.types.CostBreakdown` instances from a fill and its
context. The central correctness constraint this module enforces:

**Do not double-count pool fees on Jupiter-routed quotes.**

Jupiter's ``outAmount`` is already net of every hop's AMM fee; the BONK
round-trip evidence in ``broker.py`` is 4.4 bp observed vs. 200 bp expected
from a 2-hop 0.25%-per-hop double-charge — a 45x error. ``broker.pool_fee_micro``
returns zero for exactly this reason. Venue fees are non-zero *only* when the
swap was computed from pool reserves (``TIER_1`` or below) rather than replayed
from a live Jupiter quote.

To make that distinction uncollapsable, :func:`build_cost_breakdown` requires
the caller to pass an explicit ``quote_replayed`` flag. The flag is not
``True`` by default — the dangerous default would be to silently zero the fee
and hide it; the dangerous default is instead to force the caller to choose.
Callers in the AMM path pass ``quote_replayed=False``; callers replaying a
Jupiter quote pass ``quote_replayed=True``. The API cannot be used silently
wrong because there is no default.

**Gas is charged on every attempt, including failed ones.**

A failed Solana swap still pays the validator. ``broker.py`` has always done
this: a failed fill has ``state=FAILED``, zero trade amounts, and non-zero
``gas_usd``. :func:`build_cost_breakdown` reads ``fill.gas_usd`` directly,
so it inherits that behaviour automatically. Callers must not subtract gas
from a failed fill before passing it here.

**Latency cost is signed.**

``latency_cost_usd`` is the opportunity cost of waiting. It is negative when
the delay was beneficial (price moved in favour), zero for flat, and positive
when the delay cost money. Clamping to non-negative would make latency look
like a one-way tax and systematically overstate gross alpha by the entire
favourable half of the distribution. The field in
:class:`~memetrader.types.CostBreakdown` is already signed; this module
preserves the sign.

**Cost-stress multiplier.**

The robustness suite reruns with multiplied costs to identify strategies whose
performance evaporates under realistic rather than optimistic assumptions.
:func:`apply_stress` scales only the components that a real adverse scenario
would amplify — venue fee, network fee, priority fee, spread, price impact —
and leaves gas (already a near-fixed platform cost) and latency cost (already
signed; amplifying a signed quantity changes its economic meaning) unchanged.
Failure cost is also amplified because adverse conditions correlate with higher
failure rates.
"""

from __future__ import annotations

from typing import Literal

from ..config import ExecutionConfig
from ..types import CostBreakdown, Fill, OrderState, finite, non_negative

# ---------------------------------------------------------------------------
# Public sentinel type
# ---------------------------------------------------------------------------

StressLevel = Literal[1, 2, 3]
"""Robustness-suite multiplier: 1x (baseline), 2x, or 3x.

Only three values are permitted so the suite has named, comparable scenarios
rather than an arbitrary slider. ``apply_stress`` enforces this via the
``Literal`` annotation; passing ``4`` is a mypy error.
"""

# ---------------------------------------------------------------------------
# Which cost components scale under stress, and which do not
# ---------------------------------------------------------------------------
#
# Gas is a near-fixed platform charge (a constant lamports per compute unit on
# Solana at a given priority fee level); adverse conditions do not multiply it.
#
# Latency cost is already a signed quantity: amplifying it changes its economic
# meaning (a beneficial delay becoming more beneficial is not a stress scenario).
# Leaving it unchanged is the only interpretation that makes the stressed run
# a proper upper bound on costs.
#
# Everything else — venue fees, network fees, priority fees, spread, price
# impact, failure cost — correlates with congestion and adverse market
# conditions and should scale.

_STRESS_FIELDS = (
    "venue_fee_usd",
    "network_fee_usd",
    "priority_fee_usd",
    "spread_usd",
    "price_impact_usd",
    "failure_cost_usd",
)


# ---------------------------------------------------------------------------
# Primary API
# ---------------------------------------------------------------------------


def build_cost_breakdown(
    fill: Fill,
    cfg: ExecutionConfig,
    *,
    quote_replayed: bool,
    latency_seconds: float = 0.0,
    price_at_decision: float | None = None,
    price_at_fill: float | None = None,
) -> CostBreakdown:
    """Decompose the cost of one fill into labelled components.

    Parameters
    ----------
    fill:
        The fill to analyse. Must be a terminal fill (``LANDED``, ``FAILED``,
        or ``EXPIRED``). Passing a non-terminal fill raises ``ValueError``
        because the amounts are not yet settled.
    cfg:
        The execution config, used for gas and pool-fee rates. Passed
        explicitly rather than imported globally so the function is testable
        without a full config file.
    quote_replayed:
        ``True`` when the fill was computed by replaying a Jupiter quote
        (``outAmount`` already net of all fees). ``False`` when the fill was
        computed from pool reserves, which requires an explicit fee charge.
        **There is no default.** The caller must choose, because the two paths
        produce different venue fees, and silently defaulting to either would
        make one of the two paths silently wrong.
    latency_seconds:
        Total time between decision and fill, in seconds. Used to compute
        ``latency_cost_usd`` — positive means the delay cost money, negative
        means it helped. Defaults to 0.0 (no latency modelled) so callers
        that do not have a latency estimate do not have to pass one, but
        see the module docstring on why clamping to zero is wrong.
    price_at_decision:
        The token price in USD at the time the decision was made. Used to
        sign the latency cost: if the price moved against the order during
        the latency window, latency cost is positive; if it moved in favour,
        it is negative. ``None`` when the price is unavailable — latency cost
        is then 0.0, not guessed.
    price_at_fill:
        The token price in USD at fill time. See ``price_at_decision``.

    Returns
    -------
    CostBreakdown
        All fields finite; ``latency_cost_usd`` signed per the module
        docstring; ``venue_fee_usd`` zero when ``quote_replayed=True``.
    """
    # Refuse non-terminal fills early: the amounts on a SUBMITTED fill are
    # provisional and building a cost breakdown from them would look like real
    # cost accounting but would be wrong.
    if not fill.state.is_terminal:
        raise ValueError(
            f"fill {fill.fill_id} has state {fill.state}, which is not terminal — "
            "cost can only be computed once the fill has settled"
        )

    gas_usd = finite(fill.gas_usd, "fill.gas_usd")

    # Network fee: the base transaction fee embedded in gas_usd. On Solana the
    # whole gas charge is a validator fee, so we record it as network_fee_usd and
    # leave priority_fee_usd for an explicit priority-fee component if the caller
    # breaks it out. For now both come from gas_usd: the split is preserved in
    # the type system so a future calibrated model can separate them.
    network_fee_usd = gas_usd
    priority_fee_usd = 0.0  # not broken out at TIER_0/TIER_1; zero is honest

    # Venue fee: zero for Jupiter-replayed quotes (outAmount is already net of
    # every hop's fee), non-zero for reserve-computed swaps. The distinction is
    # the whole reason this parameter exists; see the module docstring.
    if quote_replayed:
        # Pool fees are baked into outAmount. Charging them again is the 45x
        # double-count documented in broker.py. The assumed_pool_fee_pct in
        # RiskParams is explicitly 0.0 for this path.
        venue_fee_usd = 0.0
    else:
        # Reserve-computed path: reconstruct the fee from the route labels.
        # fill.pool_fee_usd carries the fee the broker charged; use it directly
        # rather than recomputing from cfg.fee_pct_for, so that cost accounting
        # matches the ledger exactly.
        venue_fee_usd = non_negative(fill.pool_fee_usd, "fill.pool_fee_usd")

    # Spread: the difference between the mid price and the execution price, as
    # a dollar amount. At TIER_0 we only have OHLCV, so the spread is estimated
    # from slippage_bps_vs_quote when available, else from the configured
    # fallback. The sign convention: a fill worse than the quote incurs a
    # positive spread cost.
    spread_usd = 0.0
    if fill.slippage_bps_vs_quote is not None and fill.notional_usd > 0:
        # slippage_bps_vs_quote is negative when fill is worse than quote
        # (broker uses sign-negative-for-worse). Spread cost is the absolute
        # loss, so we negate: a negative bps means a positive cost.
        slip_pct = (-fill.slippage_bps_vs_quote) / 10_000.0
        spread_usd = slip_pct * fill.notional_usd
    elif fill.notional_usd > 0:
        slip_pct = cfg.slippage_bps_fallback / 10_000.0
        spread_usd = slip_pct * fill.notional_usd

    # Price impact: from the quote's stated impact, already embedded in
    # outAmount on the Jupiter path. We record it for attribution even though it
    # is not an additive charge — it is what we gave up relative to the
    # infinite-liquidity mid.
    price_impact_usd = 0.0
    if fill.price_impact_pct is not None and fill.notional_usd > 0:
        price_impact_usd = (fill.price_impact_pct / 100.0) * fill.notional_usd

    # Latency cost: signed opportunity cost of the delay. Negative when the
    # delay was favourable (price moved toward us), positive when it cost money.
    # See the module docstring on why clamping to zero is wrong.
    latency_cost_usd = _latency_cost(
        fill=fill,
        latency_seconds=latency_seconds,
        price_at_decision=price_at_decision,
        price_at_fill=price_at_fill,
    )

    # Failure cost: the economic cost of a failed attempt beyond gas. For a
    # failed fill the gas is already in network_fee_usd; failure_cost_usd
    # captures opportunity cost — if the strategy needed to re-quote and the
    # price moved, that delta lives here. At TIER_0 we cannot measure it, so
    # it is zero. A TIER_3 calibrated model would populate it from observed
    # failure patterns.
    failure_cost_usd = 0.0
    if fill.state is OrderState.FAILED:
        # At TIER_0: gas is already counted; no additional opportunity cost
        # can be modelled from OHLCV. The field exists so a calibrated model
        # can fill it in; we do not invent a number here.
        failure_cost_usd = 0.0

    return CostBreakdown(
        venue_fee_usd=venue_fee_usd,
        network_fee_usd=network_fee_usd,
        priority_fee_usd=priority_fee_usd,
        spread_usd=spread_usd,
        price_impact_usd=price_impact_usd,
        latency_cost_usd=latency_cost_usd,
        failure_cost_usd=failure_cost_usd,
    )


def apply_stress(costs: CostBreakdown, level: StressLevel) -> CostBreakdown:
    """Scale variable cost components by ``level`` for robustness testing.

    Only the components that a real adverse scenario would amplify are scaled;
    gas and latency cost are left unchanged. See the module docstring for the
    full rationale.

    ``level`` must be 1, 2, or 3 (the type annotation enforces this at static
    check time). Passing 1 is a no-op but is permitted so the suite can iterate
    over all three levels uniformly.

    The stress scenario is an upper bound on costs: the stressed P&L is the
    worst-case estimate under adverse-but-plausible conditions. A strategy that
    is unprofitable at 2x costs but profitable at 1x is a warning signal, not
    a green light.
    """
    if level not in (1, 2, 3):
        raise ValueError(f"StressLevel must be 1, 2, or 3, got {level!r}")

    m = float(level)
    return CostBreakdown(
        venue_fee_usd=costs.venue_fee_usd * m,
        network_fee_usd=costs.network_fee_usd,  # near-fixed platform charge
        priority_fee_usd=costs.priority_fee_usd * m,
        spread_usd=costs.spread_usd * m,
        price_impact_usd=costs.price_impact_usd * m,
        latency_cost_usd=costs.latency_cost_usd,  # signed; do not amplify
        failure_cost_usd=costs.failure_cost_usd * m,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _latency_cost(
    fill: Fill,
    latency_seconds: float,
    price_at_decision: float | None,
    price_at_fill: float | None,
) -> float:
    """Compute the signed latency cost for one fill.

    The cost is the P&L impact of the price move during the latency window:

    * BUY: if price rose during latency, we paid more → positive cost.
    * SELL: if price fell during latency, we received less → positive cost.
    * Either direction can be negative (latency was beneficial).

    Returns 0.0 when:
    * ``latency_seconds == 0`` (no latency modelled).
    * Either price is None (cannot measure the move).
    * The fill is terminal-failed (no quantity traded; no opportunity cost).
    * ``fill.notional_usd == 0`` (dust fill or failed).

    The sign is preserved without clamping. See the module docstring.
    """
    finite(latency_seconds, "latency_seconds")

    if latency_seconds == 0.0:
        return 0.0
    if price_at_decision is None or price_at_fill is None:
        return 0.0
    if fill.state is OrderState.FAILED or fill.notional_usd == 0.0:
        return 0.0
    if fill.token_amount_atomic == 0:
        return 0.0

    # Price move as a fraction of the decision price.
    price_move = price_at_fill - price_at_decision
    if price_at_decision == 0.0:
        return 0.0

    # Quantity in UI units — the fill records atomic units, token_decimals
    # lets us reconstruct the UI quantity without access to the token meta.
    quantity_ui = fill.token_amount_atomic / (10**fill.token_decimals)

    from ..types import Side

    if fill.side is Side.BUY:
        # A price rise hurts a buyer: we paid more than we expected to.
        return price_move * quantity_ui
    else:
        # A price fall hurts a seller: we received less than we expected to.
        return -price_move * quantity_ui


__all__ = [
    "StressLevel",
    "apply_stress",
    "build_cost_breakdown",
]
