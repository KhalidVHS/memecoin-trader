"""Property and unit tests for route_replay.py.

Key properties under test:

* **No route produces no result** — a failed hop (zero output) must return
  ``None``, never a synthetic Quote.
* **Provenance is preserved** — the caller's HISTORICAL/COUNTERFACTUAL label
  must survive round-trip through ``replay_route``.
* **Two-hop output <= better single hop** — routing through an intermediary
  can only add fees, never remove them; the direct route (if it exists) is
  at least as good as a two-hop route through the same reserves.
* **min_out <= out_amount** — the slippage floor must never exceed the expected
  output (enforced by the Quote invariant and by our own slippage formula).
* **apply_hop_fees=False produces more output than True for fee > 0** —
  disabling fee deduction should only help, never hurt.
* **Disconnected route raises ValidationError** — a route where hop N's output
  token != hop N+1's input token is a specification error, not a market
  condition, and must be rejected loudly.
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from memetrader.execution.amm.route_replay import (
    Hop,
    RouteProvenance,
    RouteResult,
    replay_route,
)
from memetrader.types import Quote, Side, TokenMeta, ValidationError

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

USDC = TokenMeta(
    mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    decimals=6,
    source="test",
    verified=True,
)
SOL = TokenMeta(
    mint="So11111111111111111111111111111111111111112",
    decimals=9,
    source="test",
    verified=True,
)
BONK = TokenMeta(
    mint="DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    decimals=5,
    source="test",
    verified=True,
)


def _make_hop(
    reserve_in: int,
    reserve_out: int,
    fee_bps: int,
    tok_in: TokenMeta,
    tok_out: TokenMeta,
) -> Hop:
    return Hop(
        reserve_in=reserve_in,
        reserve_out=reserve_out,
        fee_bps=fee_bps,
        input_token=tok_in,
        output_token=tok_out,
    )


_RESERVES = st.integers(min_value=1_000, max_value=10_000_000_000)
_AMOUNTS = st.integers(min_value=1, max_value=100_000_000)
_FEE = st.integers(min_value=0, max_value=500)  # up to 5%
_SLIPPAGE = st.integers(min_value=0, max_value=500)  # up to 5%


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestBasicReplay:
    def test_single_hop_buy_returns_result(self) -> None:
        hop = _make_hop(10_000_000, 10_000_000, 30, USDC, SOL)
        result = replay_route(
            [hop],
            symbol="SOL",
            side=Side.BUY,
            in_amount_atomic=1_000_000,
            slippage_bps=50,
            provenance=RouteProvenance.COUNTERFACTUAL,
            now=1_700_000_000.0,
        )
        assert result is not None
        assert isinstance(result, RouteResult)
        assert isinstance(result.quote, Quote)
        assert result.quote.out_amount_atomic > 0

    def test_provenance_historical_preserved(self) -> None:
        hop = _make_hop(10_000_000, 10_000_000, 30, USDC, SOL)
        result = replay_route(
            [hop],
            symbol="SOL",
            side=Side.BUY,
            in_amount_atomic=1_000_000,
            slippage_bps=50,
            provenance=RouteProvenance.HISTORICAL,
            now=1_700_000_000.0,
        )
        assert result is not None
        assert result.provenance is RouteProvenance.HISTORICAL

    def test_provenance_counterfactual_preserved(self) -> None:
        hop = _make_hop(10_000_000, 10_000_000, 30, USDC, SOL)
        result = replay_route(
            [hop],
            symbol="SOL",
            side=Side.BUY,
            in_amount_atomic=1_000_000,
            slippage_bps=50,
            provenance=RouteProvenance.COUNTERFACTUAL,
            now=1_700_000_000.0,
        )
        assert result is not None
        assert result.provenance is RouteProvenance.COUNTERFACTUAL

    def test_empty_hops_returns_none(self) -> None:
        result = replay_route(
            [],
            symbol="SOL",
            side=Side.BUY,
            in_amount_atomic=1_000_000,
            slippage_bps=50,
            provenance=RouteProvenance.COUNTERFACTUAL,
        )
        assert result is None

    def test_zero_amount_in_returns_none(self) -> None:
        hop = _make_hop(10_000_000, 10_000_000, 30, USDC, SOL)
        result = replay_route(
            [hop],
            symbol="SOL",
            side=Side.BUY,
            in_amount_atomic=0,
            slippage_bps=50,
            provenance=RouteProvenance.COUNTERFACTUAL,
        )
        assert result is None

    def test_pool_too_thin_returns_none(self) -> None:
        # Reserve is 1 atomic unit — any realistic swap produces zero output.
        hop = _make_hop(1, 1, 30, USDC, SOL)
        result = replay_route(
            [hop],
            symbol="SOL",
            side=Side.BUY,
            in_amount_atomic=1_000_000,
            slippage_bps=50,
            provenance=RouteProvenance.COUNTERFACTUAL,
        )
        assert result is None

    def test_disconnected_route_raises(self) -> None:
        hop1 = _make_hop(10_000_000, 10_000_000, 30, USDC, SOL)
        # hop2 input is BONK, not SOL — disconnected from hop1
        hop2 = _make_hop(10_000_000, 10_000_000, 30, BONK, USDC)
        with pytest.raises(ValidationError):
            replay_route(
                [hop1, hop2],
                symbol="USDC",
                side=Side.SELL,
                in_amount_atomic=1_000_000,
                slippage_bps=50,
                provenance=RouteProvenance.COUNTERFACTUAL,
            )

    def test_min_out_le_out_amount(self) -> None:
        hop = _make_hop(10_000_000, 10_000_000, 30, USDC, SOL)
        result = replay_route(
            [hop],
            symbol="SOL",
            side=Side.BUY,
            in_amount_atomic=1_000_000,
            slippage_bps=100,
            provenance=RouteProvenance.COUNTERFACTUAL,
            now=1_700_000_000.0,
        )
        assert result is not None
        assert result.quote.min_out_amount_atomic <= result.quote.out_amount_atomic

    def test_apply_hop_fees_false_increases_output(self) -> None:
        hop = _make_hop(10_000_000, 10_000_000, 100, USDC, SOL)  # 1% fee
        result_with_fee = replay_route(
            [hop],
            symbol="SOL",
            side=Side.BUY,
            in_amount_atomic=1_000_000,
            slippage_bps=50,
            provenance=RouteProvenance.HISTORICAL,
            apply_hop_fees=True,
            now=1_700_000_000.0,
        )
        result_no_fee = replay_route(
            [hop],
            symbol="SOL",
            side=Side.BUY,
            in_amount_atomic=1_000_000,
            slippage_bps=50,
            provenance=RouteProvenance.HISTORICAL,
            apply_hop_fees=False,
            now=1_700_000_000.0,
        )
        assert result_with_fee is not None
        assert result_no_fee is not None
        assert result_no_fee.quote.out_amount_atomic >= result_with_fee.quote.out_amount_atomic

    def test_two_hop_connected_route(self) -> None:
        hop1 = _make_hop(10_000_000, 10_000_000, 30, USDC, SOL)
        hop2 = _make_hop(10_000_000, 10_000_000_000, 30, SOL, BONK)
        result = replay_route(
            [hop1, hop2],
            symbol="BONK",
            side=Side.BUY,
            in_amount_atomic=1_000_000,
            slippage_bps=50,
            provenance=RouteProvenance.COUNTERFACTUAL,
            now=1_700_000_000.0,
        )
        assert result is not None
        assert result.hops_computed == 2

    def test_price_impact_non_negative(self) -> None:
        hop = _make_hop(10_000_000, 10_000_000, 30, USDC, SOL)
        result = replay_route(
            [hop],
            symbol="SOL",
            side=Side.BUY,
            in_amount_atomic=1_000_000,
            slippage_bps=50,
            provenance=RouteProvenance.COUNTERFACTUAL,
            now=1_700_000_000.0,
        )
        assert result is not None
        assert result.price_impact_pct >= 0.0

    def test_invalid_slippage_raises(self) -> None:
        hop = _make_hop(10_000_000, 10_000_000, 30, USDC, SOL)
        with pytest.raises(ValidationError):
            replay_route(
                [hop],
                symbol="SOL",
                side=Side.BUY,
                in_amount_atomic=1_000_000,
                slippage_bps=10_001,  # > 10_000
                provenance=RouteProvenance.COUNTERFACTUAL,
            )


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@given(
    reserve_in=_RESERVES,
    reserve_out=_RESERVES,
    amount=_AMOUNTS,
    fee_bps=_FEE,
    slippage_bps=_SLIPPAGE,
)
@settings(max_examples=400)
def test_min_out_le_out_amount_property(
    reserve_in: int,
    reserve_out: int,
    amount: int,
    fee_bps: int,
    slippage_bps: int,
) -> None:
    """min_out must never exceed out_amount."""
    hop = _make_hop(reserve_in, reserve_out, fee_bps, USDC, SOL)
    result = replay_route(
        [hop],
        symbol="SOL",
        side=Side.BUY,
        in_amount_atomic=amount,
        slippage_bps=slippage_bps,
        provenance=RouteProvenance.COUNTERFACTUAL,
        now=1_700_000_000.0,
    )
    if result is not None:
        assert result.quote.min_out_amount_atomic <= result.quote.out_amount_atomic


@given(
    reserve_in=_RESERVES,
    reserve_out=_RESERVES,
    amount=_AMOUNTS,
    fee_bps=st.integers(min_value=1, max_value=500),  # fee > 0 required
    slippage_bps=_SLIPPAGE,
)
@settings(max_examples=300)
def test_no_fee_replay_ge_fee_replay(
    reserve_in: int,
    reserve_out: int,
    amount: int,
    fee_bps: int,
    slippage_bps: int,
) -> None:
    """Disabling fees (apply_hop_fees=False) never reduces the output."""
    hop = _make_hop(reserve_in, reserve_out, fee_bps, USDC, SOL)
    kwargs: dict = dict(
        symbol="SOL",
        side=Side.BUY,
        in_amount_atomic=amount,
        slippage_bps=slippage_bps,
        provenance=RouteProvenance.HISTORICAL,
        now=1_700_000_000.0,
    )
    result_fee = replay_route([hop], **kwargs, apply_hop_fees=True)
    result_nofee = replay_route([hop], **kwargs, apply_hop_fees=False)
    if result_fee is not None and result_nofee is not None:
        assert (
            result_nofee.quote.out_amount_atomic >= result_fee.quote.out_amount_atomic
        )


@given(
    r1=_RESERVES,
    r2=_RESERVES,
    r3=_RESERVES,
    r4=_RESERVES,
    amount=_AMOUNTS,
    fee_bps=_FEE,
    slippage_bps=_SLIPPAGE,
)
@settings(max_examples=200)
def test_two_hop_le_direct_through_same_reserves(
    r1: int,
    r2: int,
    r3: int,
    r4: int,
    amount: int,
    fee_bps: int,
    slippage_bps: int,
) -> None:
    """A two-hop route's output must not exceed the direct single-hop output.

    This is a fundamental invariant: routing through an intermediary adds at
    least one more fee application, so the two-hop path is always <= the
    direct path (assuming the direct path exists with the same reserves).

    We construct:
      - Direct:  USDC -> SOL with reserves (r1, r2)
      - Two-hop: USDC -> BONK (reserves r1, r3) -> SOL (reserves r3, r2)

    Note: the two-hop route uses different intermediate reserves, so it will
    generally produce different (and higher-fee-impacted) results than the
    direct route. This test asserts the more conservative invariant: that the
    two-hop route's output with fee applied twice is <= zero-fee single hop.
    """
    assume(r3 > 1)  # intermediate reserve must be positive

    direct_hop = _make_hop(r1, r2, fee_bps, USDC, SOL)
    hop1 = _make_hop(r1, r3, fee_bps, USDC, BONK)
    hop2 = _make_hop(r3, r2, fee_bps, BONK, SOL)

    direct = replay_route(
        [direct_hop],
        symbol="SOL",
        side=Side.BUY,
        in_amount_atomic=amount,
        slippage_bps=0,
        provenance=RouteProvenance.COUNTERFACTUAL,
        now=1_700_000_000.0,
        apply_hop_fees=False,  # zero-fee direct is the upper bound
    )
    two_hop = replay_route(
        [hop1, hop2],
        symbol="SOL",
        side=Side.BUY,
        in_amount_atomic=amount,
        slippage_bps=0,
        provenance=RouteProvenance.COUNTERFACTUAL,
        now=1_700_000_000.0,
        apply_hop_fees=True,  # fees applied twice
    )

    if direct is not None and two_hop is not None:
        # Two-hop with fees must not exceed zero-fee direct (conservative bound).
        assert two_hop.quote.out_amount_atomic <= direct.quote.out_amount_atomic + 1, (
            f"Two-hop output {two_hop.quote.out_amount_atomic} exceeded zero-fee "
            f"direct output {direct.quote.out_amount_atomic} "
            f"(r1={r1}, r2={r2}, r3={r3}, amount={amount}, fee={fee_bps})"
        )


@given(
    reserve_in=_RESERVES,
    reserve_out=_RESERVES,
    amount=_AMOUNTS,
    fee_bps=_FEE,
    slippage_bps=_SLIPPAGE,
    prov=st.sampled_from([RouteProvenance.HISTORICAL, RouteProvenance.COUNTERFACTUAL]),
)
@settings(max_examples=300)
def test_provenance_round_trips(
    reserve_in: int,
    reserve_out: int,
    amount: int,
    fee_bps: int,
    slippage_bps: int,
    prov: RouteProvenance,
) -> None:
    """The provenance label must survive through the result unchanged."""
    hop = _make_hop(reserve_in, reserve_out, fee_bps, USDC, SOL)
    result = replay_route(
        [hop],
        symbol="SOL",
        side=Side.BUY,
        in_amount_atomic=amount,
        slippage_bps=slippage_bps,
        provenance=prov,
        now=1_700_000_000.0,
    )
    if result is not None:
        assert result.provenance is prov


@given(
    reserve_in=_RESERVES,
    reserve_out=_RESERVES,
    amount=_AMOUNTS,
    fee_bps=_FEE,
    slippage_bps=_SLIPPAGE,
)
@settings(max_examples=300)
def test_price_impact_non_negative_property(
    reserve_in: int,
    reserve_out: int,
    amount: int,
    fee_bps: int,
    slippage_bps: int,
) -> None:
    """Price impact must always be non-negative (swappers never get better than spot)."""
    hop = _make_hop(reserve_in, reserve_out, fee_bps, USDC, SOL)
    result = replay_route(
        [hop],
        symbol="SOL",
        side=Side.BUY,
        in_amount_atomic=amount,
        slippage_bps=slippage_bps,
        provenance=RouteProvenance.COUNTERFACTUAL,
        now=1_700_000_000.0,
    )
    if result is not None:
        assert result.price_impact_pct >= 0.0
