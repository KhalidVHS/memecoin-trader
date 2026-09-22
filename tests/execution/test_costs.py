"""Tests for execution/costs.py.

Offline, seeded, no network. Critical invariants under test:

1. Replaying a Jupiter quote does NOT add a venue fee.
2. A reserve-computed swap DOES add a venue fee.
3. Gas is charged on a failed attempt (gas_usd is non-zero on FAILED fills).
4. A 2x stress multiplier doubles the right components and leaves gas + latency
   cost unchanged.
5. Latency cost is signed: negative when delay was beneficial.
6. Stress multiplier is only valid for levels 1, 2, 3.
7. Non-terminal fills are refused.
"""

from __future__ import annotations

import pytest

from memetrader.config import ExecutionConfig
from memetrader.execution.costs import StressLevel, apply_stress, build_cost_breakdown
from memetrader.types import OrderState, Side


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXEC_CFG = ExecutionConfig(
    slippage_bps_fallback=50.0,
    gas_usd_per_swap=0.21,
    failed_tx_rate=0.06,
    default_pool_fee_pct=0.25,
    pool_fee_pct={"Orca": 0.30, "Raydium": 0.25},
)


def _make_fill(
    state: OrderState = OrderState.LANDED,
    notional_usd: float = 100.0,
    gas_usd: float = 0.21,
    pool_fee_usd: float = 0.0,
    slippage_bps: float | None = -10.0,
    price_impact_pct: float | None = 0.5,
    token_amount_atomic: int = 1_000_000,
    token_decimals: int = 6,
    side: Side = Side.BUY,
):
    from memetrader.ids import new_fill_id, new_intent_id, new_order_id
    from memetrader.types import Fill

    in_atomic = 1_000_000 if side is Side.BUY else token_amount_atomic
    out_atomic = token_amount_atomic if side is Side.BUY else 1_000_000

    return Fill(
        fill_id=new_fill_id(),
        order_id=new_order_id(),
        intent_id=new_intent_id(),
        decision_id=None,
        ts=1000.0,
        symbol="BONK",
        side=side,
        state=state,
        in_amount_atomic=in_atomic if state is not OrderState.FAILED else 0,
        out_amount_atomic=out_atomic if state is not OrderState.FAILED else 0,
        token_amount_atomic=token_amount_atomic if state is not OrderState.FAILED else 0,
        token_decimals=token_decimals,
        quote_fingerprint="fp123",
        price_usd=0.00001 if state is not OrderState.FAILED else None,
        notional_usd=notional_usd if state is not OrderState.FAILED else 0.0,
        price_impact_pct=price_impact_pct,
        pool_fee_usd=pool_fee_usd if state is not OrderState.FAILED else 0.0,
        gas_usd=gas_usd,
        realized_pnl_usd=0.0,
        slippage_bps_vs_quote=slippage_bps if state is not OrderState.FAILED else None,
        note=None,
    )


# ---------------------------------------------------------------------------
# Venue fee tests — the double-count guard
# ---------------------------------------------------------------------------


class TestVenueFee:
    def test_quote_replayed_has_zero_venue_fee(self):
        """A Jupiter-replayed fill must not charge a venue fee. outAmount is
        already net of all hop fees; adding one would be the 45x double-count
        documented in broker.py and the module docstring."""
        fill = _make_fill()
        costs = build_cost_breakdown(fill, _EXEC_CFG, quote_replayed=True)
        assert costs.venue_fee_usd == 0.0, (
            "replaying a quote must not add a venue fee — outAmount is already net"
        )

    def test_reserve_computed_has_nonzero_venue_fee(self):
        """A reserve-computed fill must charge the pool fee from fill.pool_fee_usd."""
        fill = _make_fill(pool_fee_usd=0.25)
        costs = build_cost_breakdown(fill, _EXEC_CFG, quote_replayed=False)
        assert costs.venue_fee_usd == pytest.approx(0.25)

    def test_no_default_for_quote_replayed(self):
        """quote_replayed has no default — omitting it is a TypeError, not a
        silent wrong answer."""
        fill = _make_fill()
        with pytest.raises(TypeError):
            build_cost_breakdown(fill, _EXEC_CFG)  # type: ignore[call-arg]

    def test_replayed_zero_not_reserve_zero(self):
        """Both paths can produce zero venue fee, but for different reasons.
        A reserve-computed fill with pool_fee_usd=0.0 is genuinely zero pool
        fee (unusual), not the same as a replayed quote's structural zero."""
        fill_replayed = _make_fill(pool_fee_usd=0.0)
        fill_reserve = _make_fill(pool_fee_usd=0.0)

        costs_replayed = build_cost_breakdown(fill_replayed, _EXEC_CFG, quote_replayed=True)
        costs_reserve = build_cost_breakdown(fill_reserve, _EXEC_CFG, quote_replayed=False)

        # Both happen to be zero, but arrived there differently.
        assert costs_replayed.venue_fee_usd == 0.0
        assert costs_reserve.venue_fee_usd == 0.0


# ---------------------------------------------------------------------------
# Gas on failed attempts
# ---------------------------------------------------------------------------


class TestGasOnFailedAttempts:
    def test_failed_fill_has_gas(self):
        """A failed Solana swap still pays the validator. Gas must be non-zero
        on a FAILED fill. Dropping it hides the real cost of failed transactions."""
        fill = _make_fill(state=OrderState.FAILED, gas_usd=0.21)
        costs = build_cost_breakdown(fill, _EXEC_CFG, quote_replayed=True)
        assert costs.network_fee_usd == pytest.approx(0.21), (
            "gas must be charged on failed attempts, not dropped"
        )

    def test_failed_fill_has_zero_venue_fee(self):
        """A failed swap never reached a pool; pool fee is zero regardless of
        the quote_replayed flag."""
        fill = _make_fill(state=OrderState.FAILED, gas_usd=0.21, pool_fee_usd=0.0)
        costs = build_cost_breakdown(fill, _EXEC_CFG, quote_replayed=False)
        assert costs.venue_fee_usd == 0.0

    def test_expired_fill_has_gas(self):
        """OrderState.EXPIRED is also terminal-failed; gas is charged."""
        fill = _make_fill(state=OrderState.EXPIRED, gas_usd=0.21)
        costs = build_cost_breakdown(fill, _EXEC_CFG, quote_replayed=True)
        assert costs.network_fee_usd == pytest.approx(0.21)

    def test_non_terminal_fill_raises(self):
        """A non-terminal fill has unsettled amounts. Building costs from one
        would look like real accounting but would be wrong."""
        fill = _make_fill(state=OrderState.SUBMITTED)
        with pytest.raises(ValueError, match="not terminal"):
            build_cost_breakdown(fill, _EXEC_CFG, quote_replayed=True)


# ---------------------------------------------------------------------------
# Stress multiplier
# ---------------------------------------------------------------------------


class TestStressMultiplier:
    def _baseline_costs(self) -> object:
        fill = _make_fill(pool_fee_usd=0.25)
        return build_cost_breakdown(fill, _EXEC_CFG, quote_replayed=False)

    def test_level_1_is_identity(self):
        costs = self._baseline_costs()
        stressed = apply_stress(costs, 1)
        assert stressed.venue_fee_usd == pytest.approx(costs.venue_fee_usd)
        assert stressed.network_fee_usd == pytest.approx(costs.network_fee_usd)
        assert stressed.total_usd == pytest.approx(costs.total_usd)

    def test_level_2_doubles_venue_fee(self):
        costs = self._baseline_costs()
        stressed = apply_stress(costs, 2)
        assert stressed.venue_fee_usd == pytest.approx(costs.venue_fee_usd * 2)

    def test_level_2_does_not_change_network_fee(self):
        """Gas is a near-fixed platform charge and must not be scaled by stress."""
        costs = self._baseline_costs()
        stressed = apply_stress(costs, 2)
        assert stressed.network_fee_usd == pytest.approx(costs.network_fee_usd)

    def test_level_2_does_not_change_latency_cost(self):
        """Latency cost is signed; amplifying a signed quantity changes its
        economic meaning. A beneficial delay becoming more beneficial is not a
        stress scenario."""
        fill = _make_fill()
        costs = build_cost_breakdown(
            fill,
            _EXEC_CFG,
            quote_replayed=True,
            latency_seconds=1.0,
            price_at_decision=0.00001,
            price_at_fill=0.000008,  # price fell — latency was beneficial (BUY)
        )
        assert costs.latency_cost_usd < 0.0, "setup: latency was beneficial"
        stressed = apply_stress(costs, 2)
        assert stressed.latency_cost_usd == pytest.approx(costs.latency_cost_usd)

    def test_level_3_triples_spread(self):
        fill = _make_fill(slippage_bps=-20.0)
        costs = build_cost_breakdown(fill, _EXEC_CFG, quote_replayed=True)
        stressed = apply_stress(costs, 3)
        assert stressed.spread_usd == pytest.approx(costs.spread_usd * 3)

    def test_invalid_level_raises(self):
        costs = self._baseline_costs()
        with pytest.raises(ValueError):
            apply_stress(costs, 4)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Latency cost signing
# ---------------------------------------------------------------------------


class TestLatencyCost:
    def test_latency_cost_positive_when_price_moved_against_buyer(self):
        """For a BUY: price rose during latency → we paid more → positive cost."""
        fill = _make_fill(side=Side.BUY)
        costs = build_cost_breakdown(
            fill,
            _EXEC_CFG,
            quote_replayed=True,
            latency_seconds=1.0,
            price_at_decision=0.00001,
            price_at_fill=0.000012,  # price rose → bad for buyer
        )
        assert costs.latency_cost_usd > 0.0

    def test_latency_cost_negative_when_price_fell_for_buyer(self):
        """For a BUY: price fell during latency → we paid less → negative cost
        (beneficial). Clamping this to zero would overstate gross alpha."""
        fill = _make_fill(side=Side.BUY)
        costs = build_cost_breakdown(
            fill,
            _EXEC_CFG,
            quote_replayed=True,
            latency_seconds=1.0,
            price_at_decision=0.00001,
            price_at_fill=0.000008,  # price fell → good for buyer
        )
        assert costs.latency_cost_usd < 0.0, (
            "beneficial latency must produce negative cost, not zero"
        )

    def test_latency_cost_zero_when_no_prices(self):
        """Without price data the latency cost cannot be computed; zero is
        honest, not a guess."""
        fill = _make_fill()
        costs = build_cost_breakdown(
            fill,
            _EXEC_CFG,
            quote_replayed=True,
            latency_seconds=1.0,
            price_at_decision=None,
            price_at_fill=None,
        )
        assert costs.latency_cost_usd == 0.0

    def test_latency_cost_zero_when_latency_zero(self):
        fill = _make_fill()
        costs = build_cost_breakdown(
            fill,
            _EXEC_CFG,
            quote_replayed=True,
            latency_seconds=0.0,
            price_at_decision=0.00001,
            price_at_fill=0.000015,
        )
        assert costs.latency_cost_usd == 0.0

    def test_latency_cost_zero_on_failed_fill(self):
        """A failed fill traded nothing; there is no opportunity cost to sign."""
        fill = _make_fill(state=OrderState.FAILED)
        costs = build_cost_breakdown(
            fill,
            _EXEC_CFG,
            quote_replayed=True,
            latency_seconds=2.0,
            price_at_decision=0.00001,
            price_at_fill=0.00002,
        )
        assert costs.latency_cost_usd == 0.0

    def test_sell_beneficial_when_price_rose(self):
        """For a SELL: price rose during latency → we received more → negative
        cost (beneficial)."""
        fill = _make_fill(side=Side.SELL)
        costs = build_cost_breakdown(
            fill,
            _EXEC_CFG,
            quote_replayed=True,
            latency_seconds=1.0,
            price_at_decision=0.00001,
            price_at_fill=0.000012,  # price rose → good for seller
        )
        assert costs.latency_cost_usd < 0.0


# ---------------------------------------------------------------------------
# CostBreakdown structural invariants
# ---------------------------------------------------------------------------


class TestBreakdownInvariants:
    def test_all_fields_finite(self):
        """CostBreakdown.__post_init__ validates all fields are finite; build
        must not produce NaN or inf."""
        import math

        fill = _make_fill()
        costs = build_cost_breakdown(fill, _EXEC_CFG, quote_replayed=True)
        for attr in (
            "venue_fee_usd",
            "network_fee_usd",
            "priority_fee_usd",
            "spread_usd",
            "price_impact_usd",
            "latency_cost_usd",
            "failure_cost_usd",
        ):
            val = getattr(costs, attr)
            assert math.isfinite(val), f"{attr} is not finite: {val}"

    def test_total_usd_is_sum_of_parts(self):
        fill = _make_fill()
        costs = build_cost_breakdown(fill, _EXEC_CFG, quote_replayed=True)
        expected = (
            costs.venue_fee_usd
            + costs.network_fee_usd
            + costs.priority_fee_usd
            + costs.spread_usd
            + costs.price_impact_usd
            + costs.latency_cost_usd
            + costs.failure_cost_usd
        )
        assert costs.total_usd == pytest.approx(expected)
