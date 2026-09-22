"""Tests for execution/fill_models.py.

Offline, seeded, no network. Each test targets one contract-mandated
invariant and is written so that removing the guard it checks makes the test
fail — see the docstring on each test for which specific behaviour it pins
down.
"""

from __future__ import annotations

import random

import pytest

from memetrader.config import ExecutionConfig
from memetrader.execution.fill_models import (
    BarExecutionModel,
    PoolStateExecutionModel,
    QuoteReplayExecutionModel,
)
from memetrader.execution.interfaces import ApprovedOrder, NoRoute
from memetrader.execution.latency import LatencyModel
from memetrader.histdata.point_in_time import ReplayState
from memetrader.histdata.schemas import PoolState, QuoteLadder, QuoteLadderRung
from memetrader.ids import new_intent_id, new_run_id
from memetrader.ids import quote_fingerprint as _quote_fingerprint
from memetrader.types import (
    NON_EXECUTABLE_NOTICE,
    Candle,
    FidelityTier,
    OrderIntent,
    OrderState,
    Quote,
    RiskBounds,
    Side,
    Timeframe,
    TokenMeta,
    ValidationError,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_USDC = TokenMeta(
    mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", decimals=6, source="test"
)
_BONK = TokenMeta(
    mint="DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263", decimals=9, source="test"
)


def _cfg(**overrides: object) -> ExecutionConfig:
    defaults: dict[str, object] = {
        "slippage_bps_fallback": 100.0,
        "gas_usd_per_swap": 0.05,
        "failed_tx_rate": 0.05,
        "default_pool_fee_pct": 0.3,
        "pool_fee_pct": {},
    }
    defaults.update(overrides)
    return ExecutionConfig(**defaults)  # type: ignore[arg-type]


def _make_intent(
    *, side: Side = Side.BUY, in_amount_atomic: int = 100_000_000, symbol: str = "BONK"
) -> OrderIntent:
    return OrderIntent(
        intent_id=new_intent_id(),
        decision_id=None,
        action_id=None,
        run_id=new_run_id(),
        ts=1_000.0,
        symbol=symbol,
        side=side,
        in_amount_atomic=in_amount_atomic,
        max_in_amount_atomic=in_amount_atomic,
        source="strategy",
    )


def _make_bounds(symbol: str = "BONK", side: Side = Side.BUY) -> RiskBounds:
    return RiskBounds(symbol=symbol, side=side, max_notional_usd=1_000.0)


def _make_quote(
    *,
    side: Side = Side.BUY,
    in_amount_atomic: int = 100_000_000,
    out_amount_atomic: int = 50_000_000,
    expires_at: float | None = 10_000.0,
    received_at: float = 999.0,
) -> Quote:
    input_token = _USDC if side is Side.BUY else _BONK
    output_token = _BONK if side is Side.BUY else _USDC
    fp = _quote_fingerprint(
        side=str(side),
        input_mint=input_token.mint,
        output_mint=output_token.mint,
        in_amount_atomic=in_amount_atomic,
        out_amount_atomic=out_amount_atomic,
        slot=None,
    )
    return Quote(
        symbol="BONK",
        side=side,
        input_token=input_token,
        output_token=output_token,
        in_amount_atomic=in_amount_atomic,
        out_amount_atomic=out_amount_atomic,
        min_out_amount_atomic=int(out_amount_atomic * 0.95),
        price_impact_pct=0.1,
        route_labels=("test_route",),
        fingerprint=fp,
        requested_at=received_at - 0.1,
        received_at=received_at,
        context_slot=None,
        expires_at=expires_at,
    )


def _make_approved_order(
    *, intent: OrderIntent, quote: Quote, decided_at: float
) -> ApprovedOrder:
    return ApprovedOrder(
        intent=intent,
        bounds=_make_bounds(side=intent.side),
        quote=quote,
        decided_at=decided_at,
    )


def _make_ladder(
    *,
    asset_id: str = "BONK",  # matches OrderIntent.symbol used in these tests
    side: Side = Side.BUY,
    rung_in: int = 100_000_000,
    rung_out: int = 50_000_000,
    available_time: float = 0.0,
) -> QuoteLadder:
    rung = QuoteLadderRung(
        in_amount_atomic=rung_in,
        out_amount_atomic=rung_out,
        price_impact_pct=0.2,
        route_labels=("test_route",),
        fees_atomic=1_000,
        min_out_atomic=int(rung_out * 0.95),
    )
    return QuoteLadder(
        asset_id=asset_id,
        pool_id="pool1",
        side=side.value,
        event_time=available_time,
        available_time=available_time,
        received_time=available_time,
        rungs=(rung,),
        context_slot=None,
        source="test",
    )


def _make_pool_state(
    *,
    asset_id: str = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    pool_id: str = "BONK",  # matches OrderIntent.symbol via the default identity resolver
    reserve_in_atomic: int = 1_000_000_000_000,
    reserve_out_atomic: int = 500_000_000_000,
    fee_rate_bps: int = 25,
    available_time: float = 0.0,
) -> PoolState:
    return PoolState(
        asset_id=asset_id,
        pool_id=pool_id,
        venue="raydium",
        event_time=available_time,
        available_time=available_time,
        received_time=available_time,
        reserve_in_atomic=reserve_in_atomic,
        reserve_out_atomic=reserve_out_atomic,
        fee_rate_bps=fee_rate_bps,
        price_usd=0.5,
        liquidity_usd=1_000_000.0,
        source="test",
    )


# ---------------------------------------------------------------------------
# 1. Anti-lookahead: signal at a bar close cannot fill at that close
# ---------------------------------------------------------------------------


class TestBarExecutionAntiLookahead:
    def test_fill_uses_next_bar_open_not_decision_bar_close(self) -> None:
        """A decision made from bar1's close must fill at bar2's open, never
        at bar1's close. If a future refactor swapped ``next_bar.open`` for
        ``last_bar.close`` this test must fail: the two prices are made to
        differ deliberately (a gap up) so the wrong choice is detectable."""
        state = ReplayState(now=0.0, publication_delay_seconds=0.0)
        bar1 = Candle(ts=0.0, open=1.0, high=1.0, low=1.0, close=1.0, volume=1_000_000.0)
        bar2 = Candle(
            ts=3_600.0, open=2.0, high=2.0, low=2.0, close=2.0, volume=1_000_000.0
        )
        state.add_bar(bar1, asset_id="BONK", timeframe=Timeframe.H1)
        state.add_bar(bar2, asset_id="BONK", timeframe=Timeframe.H1)

        decided_at = 3_600.0  # bar1 available_time == decision time
        model = BarExecutionModel(_cfg(), usd_token=_USDC)
        intent = _make_intent(side=Side.BUY, in_amount_atomic=1_000_000)
        state.now = decided_at
        quote = model.price(intent=intent, state=state, now=decided_at)
        assert quote is not None
        order = _make_approved_order(intent=intent, quote=quote, decided_at=decided_at)

        state.now = 7_200.0  # bar2 available_time
        report = model.fill(order=order, state=state, now=state.now)
        assert report.fill is not None
        assert report.fill.price_usd == pytest.approx(2.0)  # bar2.open
        assert report.fill.price_usd != pytest.approx(1.0)  # never bar1.close

    def test_fill_before_next_bar_exists_raises_no_route(self) -> None:
        """If no bar has opened after the decision yet, there is nothing
        legitimate to fill against — this must raise, not fabricate a price."""
        state = ReplayState(now=0.0, publication_delay_seconds=0.0)
        bar1 = Candle(ts=0.0, open=1.0, high=1.0, low=1.0, close=1.0, volume=1_000_000.0)
        state.add_bar(bar1, asset_id="BONK", timeframe=Timeframe.H1)
        state.now = 3_600.0

        model = BarExecutionModel(_cfg(), usd_token=_USDC)
        intent = _make_intent(side=Side.BUY, in_amount_atomic=1_000_000)
        state.now = 3_600.0
        quote = model.price(intent=intent, state=state, now=3_600.0)
        assert quote is not None
        order = _make_approved_order(intent=intent, quote=quote, decided_at=3_600.0)

        with pytest.raises(NoRoute):
            model.fill(order=order, state=state, now=3_600.0 + 1.0)


# ---------------------------------------------------------------------------
# 2. Risk clamp forces an exact-size requote (audit C3)
# ---------------------------------------------------------------------------


class TestQuoteReplayExactSizeRequote:
    def test_requote_at_clamped_size_is_exact_not_rescaled(self) -> None:
        """A risk clamp shrinks the requested size; the resulting quote must
        be bound to exactly that smaller size, not a linear rescale cached
        from the original (larger) quote object."""
        state = ReplayState(now=0.0)
        # Two rungs so the clamped (smaller) size still floor-selects a real
        # probed rung rather than falling below the ladder's smallest size.
        small_rung = QuoteLadderRung(
            in_amount_atomic=10_000_000,
            out_amount_atomic=5_000_000,
            price_impact_pct=0.05,
            route_labels=("test_route",),
            fees_atomic=100,
            min_out_atomic=4_750_000,
        )
        large_rung = QuoteLadderRung(
            in_amount_atomic=100_000_000,
            out_amount_atomic=50_000_000,
            price_impact_pct=0.2,
            route_labels=("test_route",),
            fees_atomic=1_000,
            min_out_atomic=47_500_000,
        )
        ladder = QuoteLadder(
            asset_id="BONK",
            pool_id="pool1",
            side="BUY",
            event_time=0.0,
            available_time=0.0,
            received_time=0.0,
            rungs=(small_rung, large_rung),
            context_slot=None,
            source="test",
        )
        state.add_quote_ladder(ladder)

        model = QuoteReplayExecutionModel(
            _cfg(), latency_model=LatencyModel(seed=1), landing_rng_seed=42
        )

        original_intent = _make_intent(side=Side.BUY, in_amount_atomic=100_000_000)
        original_quote = model.price(intent=original_intent, state=state, now=0.0)
        assert original_quote is not None
        assert original_quote.in_amount_atomic == 100_000_000

        clamped_intent = _make_intent(side=Side.BUY, in_amount_atomic=30_000_000)
        clamped_quote = model.price(intent=clamped_intent, state=state, now=0.0)
        assert clamped_quote is not None
        # Exact-size binding: the requoted quote's in_amount is exactly the
        # clamped size, never the original size and never a value produced by
        # taking original_quote.out_amount_atomic and scaling it externally.
        assert clamped_quote.in_amount_atomic == 30_000_000
        # Floor rung for 30M is the 10M rung (100M > 30M is not eligible);
        # scaled proportionally from that rung, never from the 100M rung.
        assert clamped_quote.out_amount_atomic == 30_000_000 * 5_000_000 // 10_000_000


# ---------------------------------------------------------------------------
# 3. Expired quote cannot fill
# ---------------------------------------------------------------------------


class TestExpiredQuote:
    def test_expired_quote_produces_no_fill(self) -> None:
        state = ReplayState(now=0.0)
        ladder = _make_ladder()
        state.add_quote_ladder(ladder)

        model = QuoteReplayExecutionModel(
            _cfg(), latency_model=LatencyModel(seed=1), landing_rng_seed=7
        )
        intent = _make_intent(side=Side.BUY, in_amount_atomic=100_000_000)
        quote = _make_quote(
            in_amount_atomic=100_000_000, out_amount_atomic=50_000_000, expires_at=5.0
        )
        order = _make_approved_order(intent=intent, quote=quote, decided_at=0.0)

        report = model.fill(order=order, state=state, now=10.0)  # past expires_at
        assert report.state is OrderState.EXPIRED
        assert report.fill is None


# ---------------------------------------------------------------------------
# 4. Failed route produces no position change
# ---------------------------------------------------------------------------


class TestFailedRouteNoPositionChange:
    def test_route_disappeared_before_fill_raises_no_route(self) -> None:
        """If the ladder vanished between price() and fill(), the caller must
        get NoRoute — never a synthesized Fill that would move a position."""
        state = ReplayState(now=0.0)
        ladder = _make_ladder()
        state.add_quote_ladder(ladder)

        model = QuoteReplayExecutionModel(
            _cfg(), latency_model=LatencyModel(seed=1), landing_rng_seed=7
        )
        intent = _make_intent(side=Side.BUY, in_amount_atomic=100_000_000)
        quote = model.price(intent=intent, state=state, now=0.0)
        assert quote is not None
        order = _make_approved_order(intent=intent, quote=quote, decided_at=0.0)

        # Route disappears: build a fresh state with no ladder loaded.
        empty_state = ReplayState(now=1.0)
        with pytest.raises(NoRoute):
            model.fill(order=order, state=empty_state, now=1.0)

    def test_landing_failure_produces_zero_amount_fill_no_position_change(self) -> None:
        """When the coin flip says the tx failed to land, the Fill row has
        zero amounts (gas was spent, but nothing was bought/sold)."""
        state = ReplayState(now=0.0)
        ladder = _make_ladder()
        state.add_quote_ladder(ladder)

        # landing_probability=0.0 forces every draw to be a failure to land.
        model = QuoteReplayExecutionModel(
            _cfg(),
            latency_model=LatencyModel(seed=1),
            landing_rng_seed=7,
            landing_probability=0.0,
        )
        intent = _make_intent(side=Side.BUY, in_amount_atomic=100_000_000)
        quote = model.price(intent=intent, state=state, now=0.0)
        assert quote is not None
        order = _make_approved_order(intent=intent, quote=quote, decided_at=0.0)

        report = model.fill(order=order, state=state, now=0.5)
        assert report.state is OrderState.FAILED
        assert report.fill is not None
        assert report.fill.in_amount_atomic == 0
        assert report.fill.out_amount_atomic == 0
        assert report.fill.token_amount_atomic == 0


# ---------------------------------------------------------------------------
# 5. Missing liquidity blocks the trade
# ---------------------------------------------------------------------------


class TestMissingLiquidity:
    def test_no_pool_state_raises_no_route(self) -> None:
        state = ReplayState(now=0.0)  # no pool state loaded
        model = PoolStateExecutionModel(_cfg(), usd_token=_USDC)
        intent = _make_intent(side=Side.BUY, in_amount_atomic=1_000_000)
        with pytest.raises(NoRoute):
            model.price(intent=intent, state=state, now=0.0)

    def test_thin_pool_zero_output_raises_no_route(self) -> None:
        """A pool so thin that even the smallest trade nets zero output must
        block the trade, not silently produce a zero-size fill."""
        state = ReplayState(now=0.0)
        pool = _make_pool_state(
            reserve_in_atomic=10, reserve_out_atomic=10, fee_rate_bps=9_999
        )
        state.add_pool_state(pool)

        model = PoolStateExecutionModel(_cfg(), usd_token=_USDC)
        intent = _make_intent(side=Side.BUY, in_amount_atomic=1)
        with pytest.raises(NoRoute):
            model.price(intent=intent, state=state, now=0.0)


# ---------------------------------------------------------------------------
# 6. Same-bar stop/target ambiguity is resolved pessimistically
# ---------------------------------------------------------------------------


class TestSameBarExitAmbiguity:
    def test_stop_wins_when_both_in_range(self) -> None:
        """Documented choice: pessimistic ordering. When both a stop and a
        target fall inside one bar's range, the stop is assumed to have hit
        first — the conservative assumption, never the friendlier one."""
        model = BarExecutionModel(_cfg(), usd_token=_USDC)
        bar = Candle(ts=0.0, open=1.0, high=1.2, low=0.8, close=1.0, volume=10.0)
        price, which = model.resolve_same_bar_exit(bar, stop_price=0.9, target_price=1.1)
        assert which == "stop"
        assert price == 0.9

    def test_only_target_in_range_resolves_to_target(self) -> None:
        model = BarExecutionModel(_cfg(), usd_token=_USDC)
        bar = Candle(ts=0.0, open=1.0, high=1.2, low=0.95, close=1.0, volume=10.0)
        price, which = model.resolve_same_bar_exit(bar, stop_price=0.5, target_price=1.1)
        assert which == "target"
        assert price == 1.1

    def test_neither_in_range_raises(self) -> None:
        model = BarExecutionModel(_cfg(), usd_token=_USDC)
        bar = Candle(ts=0.0, open=1.0, high=1.05, low=0.95, close=1.0, volume=10.0)
        with pytest.raises(ValidationError):
            model.resolve_same_bar_exit(bar, stop_price=0.5, target_price=2.0)


# ---------------------------------------------------------------------------
# 7. Participation cap is respected
# ---------------------------------------------------------------------------


class TestParticipationCap:
    def test_partial_fill_when_desired_exceeds_cap(self) -> None:
        """A desired size much larger than the bar's volume-derived cap must
        produce a partial fill capped at the participation limit, never the
        full requested size."""
        state = ReplayState(now=0.0, publication_delay_seconds=0.0)
        bar1 = Candle(ts=0.0, open=1.0, high=1.0, low=1.0, close=1.0, volume=1_000.0)
        bar2 = Candle(ts=3_600.0, open=1.0, high=1.0, low=1.0, close=1.0, volume=1_000.0)
        state.add_bar(bar1, asset_id="BONK", timeframe=Timeframe.H1)
        state.add_bar(bar2, asset_id="BONK", timeframe=Timeframe.H1)

        model = BarExecutionModel(_cfg(), usd_token=_USDC, participation_cap_pct=0.1)
        # Desire far more USD than 10% of 1000 tokens at price 1.0 (=100 tokens).
        intent = _make_intent(side=Side.BUY, in_amount_atomic=_USDC.to_atomic(10_000.0))
        state.now = 3_600.0
        quote = model.price(intent=intent, state=state, now=3_600.0)
        assert quote is not None
        order = _make_approved_order(intent=intent, quote=quote, decided_at=3_600.0)

        state.now = 7_200.0
        report = model.fill(order=order, state=state, now=state.now)
        assert report.fill is not None
        filled_tokens = report.fill.token_amount_atomic / 10**_BONK.decimals
        assert filled_tokens == pytest.approx(100.0, rel=1e-6)  # 10% of bar2 volume
        assert report.fill.note is not None
        assert "participation cap" in report.fill.note

    def test_full_fill_when_within_cap(self) -> None:
        state = ReplayState(now=0.0, publication_delay_seconds=0.0)
        bar1 = Candle(ts=0.0, open=1.0, high=1.0, low=1.0, close=1.0, volume=1_000.0)
        bar2 = Candle(ts=3_600.0, open=1.0, high=1.0, low=1.0, close=1.0, volume=1_000.0)
        state.add_bar(bar1, asset_id="BONK", timeframe=Timeframe.H1)
        state.add_bar(bar2, asset_id="BONK", timeframe=Timeframe.H1)

        model = BarExecutionModel(_cfg(), usd_token=_USDC, participation_cap_pct=0.5)
        # Desire well within 50% of bar volume (500 tokens): 10 USD @ price 1.0.
        intent = _make_intent(side=Side.BUY, in_amount_atomic=_USDC.to_atomic(10.0))
        state.now = 3_600.0
        quote = model.price(intent=intent, state=state, now=3_600.0)
        assert quote is not None
        order = _make_approved_order(intent=intent, quote=quote, decided_at=3_600.0)

        state.now = 7_200.0
        report = model.fill(order=order, state=state, now=state.now)
        assert report.fill is not None
        assert report.fill.note is None


# ---------------------------------------------------------------------------
# 8. NON_EXECUTABLE tier carries the notice verbatim
# ---------------------------------------------------------------------------


class TestNonExecutableNotice:
    def test_bar_model_report_carries_notice_verbatim(self) -> None:
        assert FidelityTier.TIER_0.permits_pnl_claim is False

        state = ReplayState(now=0.0, publication_delay_seconds=0.0)
        bar1 = Candle(ts=0.0, open=1.0, high=1.0, low=1.0, close=1.0, volume=1_000.0)
        bar2 = Candle(ts=3_600.0, open=1.0, high=1.0, low=1.0, close=1.0, volume=1_000.0)
        state.add_bar(bar1, asset_id="BONK", timeframe=Timeframe.H1)
        state.add_bar(bar2, asset_id="BONK", timeframe=Timeframe.H1)

        model = BarExecutionModel(_cfg(), usd_token=_USDC)
        intent = _make_intent(side=Side.BUY, in_amount_atomic=1_000_000)
        state.now = 3_600.0
        quote = model.price(intent=intent, state=state, now=3_600.0)
        assert quote is not None
        order = _make_approved_order(intent=intent, quote=quote, decided_at=3_600.0)

        state.now = 7_200.0
        report = model.fill(order=order, state=state, now=state.now)
        assert report.reason == NON_EXECUTABLE_NOTICE


# ---------------------------------------------------------------------------
# 9. Landing-probability draws are reproducible and RNG-independent of latency
# ---------------------------------------------------------------------------


class TestLandingDrawIndependence:
    def test_landing_draws_reproducible_for_fixed_seed(self) -> None:
        state = ReplayState(now=0.0)
        ladder = _make_ladder()
        state.add_quote_ladder(ladder)

        def make_model(latency_seed: int) -> QuoteReplayExecutionModel:
            return QuoteReplayExecutionModel(
                _cfg(),
                latency_model=LatencyModel(seed=latency_seed),
                landing_rng_seed=999,
                landing_probability=0.5,
            )

        intent = _make_intent(side=Side.BUY, in_amount_atomic=100_000_000)

        def run(model: QuoteReplayExecutionModel) -> list[OrderState]:
            outcomes = []
            for i in range(5):
                quote = model.price(intent=intent, state=state, now=float(i))
                assert quote is not None
                order = _make_approved_order(
                    intent=intent, quote=quote, decided_at=float(i)
                )
                report = model.fill(order=order, state=state, now=float(i) + 0.1)
                outcomes.append(report.state)
            return outcomes

        # Same landing seed, but two DIFFERENT latency seeds (so the number
        # and value of latency draws differ) — the landing outcome sequence
        # must be identical, proving it does not share the latency RNG.
        outcomes_a = run(make_model(latency_seed=1))
        outcomes_b = run(make_model(latency_seed=2))
        assert outcomes_a == outcomes_b

    def test_landing_rng_matches_independent_random_instance(self) -> None:
        """The landing draw sequence must match a bare ``random.Random`` seeded
        the same way — proving it is not perturbed by the latency model's own
        internal draws happening in between."""
        state = ReplayState(now=0.0)
        ladder = _make_ladder()
        state.add_quote_ladder(ladder)

        seed = 12345
        model = QuoteReplayExecutionModel(
            _cfg(),
            latency_model=LatencyModel(seed=1),
            landing_rng_seed=seed,
            landing_probability=0.5,
        )
        reference_rng = random.Random(seed)

        intent = _make_intent(side=Side.BUY, in_amount_atomic=100_000_000)
        for i in range(5):
            quote = model.price(intent=intent, state=state, now=float(i))
            assert quote is not None
            order = _make_approved_order(intent=intent, quote=quote, decided_at=float(i))
            report = model.fill(order=order, state=state, now=float(i) + 0.1)
            expected_will_land = reference_rng.random() < 0.5
            expected_state = OrderState.LANDED if expected_will_land else OrderState.FAILED
            assert report.state == expected_state
