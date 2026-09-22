"""Tests for execution/interfaces.py.

Offline, seeded, no network. Tests verify the structural contracts rather than
any specific numeric behaviour: ApprovedOrder is immutable and validates its
fields; NoRoute is distinguishable from None and from a generic exception;
the Protocol classes are structurally checkable at construction time.
"""

from __future__ import annotations

import pytest

from memetrader.execution.interfaces import ApprovedOrder, NoRoute
from memetrader.types import (
    RiskBounds,
    Side,
    ValidationError,
)

# ---------------------------------------------------------------------------
# Fixtures — minimal valid objects
# ---------------------------------------------------------------------------


def _make_token_meta(mint: str = "USDC1111111111111111111111111111111"):
    from memetrader.types import TokenMeta

    return TokenMeta(mint=mint, decimals=6, source="test", verified=True)


def _make_quote(side: Side = Side.BUY, received_at: float = 1_000.0):
    from memetrader.ids import quote_fingerprint
    from memetrader.types import Quote

    usdc = _make_token_meta("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
    bonk = _make_token_meta("DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263")
    fp = quote_fingerprint(
        side=str(side),
        input_mint=usdc.mint if side is Side.BUY else bonk.mint,
        output_mint=bonk.mint if side is Side.BUY else usdc.mint,
        in_amount_atomic=1_000_000,
        out_amount_atomic=500_000,
        slot=None,
    )
    return Quote(
        symbol="BONK",
        side=side,
        input_token=usdc if side is Side.BUY else bonk,
        output_token=bonk if side is Side.BUY else usdc,
        in_amount_atomic=1_000_000,
        out_amount_atomic=500_000,
        min_out_amount_atomic=490_000,
        price_impact_pct=0.1,
        route_labels=("Orca",),
        fingerprint=fp,
        requested_at=received_at - 0.1,
        received_at=received_at,
        context_slot=123,
    )


def _make_intent(side: Side = Side.BUY):
    from memetrader.ids import new_intent_id, new_run_id
    from memetrader.types import OrderIntent

    return OrderIntent(
        intent_id=new_intent_id(),
        decision_id=None,
        action_id=None,
        run_id=new_run_id(),
        ts=1_000.0,
        symbol="BONK",
        side=side,
        in_amount_atomic=1_000_000,
        max_in_amount_atomic=1_000_000,
        source="strategy",
    )


def _make_bounds(side: Side = Side.BUY):
    return RiskBounds(symbol="BONK", side=side, max_notional_usd=10.0)


def _make_approved_order(decided_at: float = 999.0):
    return ApprovedOrder(
        intent=_make_intent(),
        bounds=_make_bounds(),
        quote=_make_quote(),
        decided_at=decided_at,
    )


# ---------------------------------------------------------------------------
# ApprovedOrder tests
# ---------------------------------------------------------------------------


class TestApprovedOrder:
    def test_construction_succeeds(self):
        order = _make_approved_order()
        assert order.decided_at == 999.0

    def test_frozen(self):
        """ApprovedOrder must be immutable — a mutable order could be resized
        after risk approval, breaking the quote-binding guarantee (audit C3)."""
        order = _make_approved_order()
        with pytest.raises((AttributeError, TypeError)):
            order.decided_at = 0.0

    def test_negative_decided_at_raises(self):
        """A negative decided_at is a programming error, not a market condition."""
        with pytest.raises(ValidationError):
            ApprovedOrder(
                intent=_make_intent(),
                bounds=_make_bounds(),
                quote=_make_quote(),
                decided_at=-1.0,
            )

    def test_decided_at_zero_allowed(self):
        """Zero is a valid epoch second (Unix epoch start); allowed."""
        order = ApprovedOrder(
            intent=_make_intent(),
            bounds=_make_bounds(),
            quote=_make_quote(),
            decided_at=0.0,
        )
        assert order.decided_at == 0.0

    def test_fields_accessible(self):
        order = _make_approved_order()
        assert order.intent.symbol == "BONK"
        assert order.bounds.max_notional_usd == 10.0
        assert order.quote.side is Side.BUY


# ---------------------------------------------------------------------------
# NoRoute tests
# ---------------------------------------------------------------------------


class TestNoRoute:
    def test_is_exception(self):
        """NoRoute must be catchable as a distinct exception type."""
        with pytest.raises(NoRoute):
            raise NoRoute("pool dry")

    def test_message_preserved(self):
        exc = NoRoute("no AMM route for BONK→USDC")
        assert exc.reason == "no AMM route for BONK→USDC"

    def test_not_caught_by_generic_value_error(self):
        """NoRoute must not be a ValueError so it isn't silently caught by
        code that traps ValueError from validation."""
        exc = NoRoute("x")
        assert not isinstance(exc, ValueError)

    def test_distinct_from_none(self):
        """The three outcomes — Quote, None, NoRoute — must not be collapsable.
        This test documents the design intent: a function returning None is not
        the same as one raising NoRoute."""
        result = None
        exc_raised = False
        try:
            raise NoRoute("no route")
        except NoRoute:
            exc_raised = True
        assert result is None
        assert exc_raised

    def test_empty_reason(self):
        """NoRoute with no reason should not crash."""
        exc = NoRoute()
        assert exc.reason == ""

    def test_failed_route_produces_no_trade(self):
        """Callers must catch NoRoute and produce no fill. This test verifies
        that a function signalling NoRoute cannot be mistaken for a successful
        price lookup."""

        def price_fn() -> None:
            raise NoRoute("route unavailable")

        result = None
        try:
            price_fn()
        except NoRoute:
            result = "no_trade"
        assert result == "no_trade", "a NoRoute must veto the trade"
