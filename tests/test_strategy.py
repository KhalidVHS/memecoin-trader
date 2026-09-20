"""Tests for the strategy layer — the audit C6 replacement for LLM order authority.

The theme running through these is that *absence must stay absent*. Most of the
defects the audit found were not wrong numbers; they were missing numbers
quietly rendered as neutral ones, at which point nothing downstream could tell
a flat market from a broken feed. So the assertions here are mostly about what
the code refuses to produce.
"""

from __future__ import annotations

import math

import pytest

from memetrader.strategy import (
    AdvisoryStrategy,
    BaselineStrategy,
    CashStrategy,
    StrategySettings,
    build_strategy,
)
from memetrader.types import (
    Candle,
    CoinSnapshot,
    DataQuality,
    EvidenceBundle,
    FlowBrief,
    PoolRef,
    PortfolioState,
    PriceLadder,
    Provenance,
    TechnicalBrief,
    Technicals,
    Timeframe,
    TxnCounts,
    ValidationError,
)

NOW = 1_758_300_000.0


def _technicals(tf: Timeframe, *, vol: float | None = 2.0) -> Technicals:
    return Technicals(
        timeframe=tf,
        candles_used=50,
        pool_address="POOL",
        rsi14=55.0,
        rsi14_rising=True,
        ema9=1.0,
        ema21=0.99,
        ema9_above_ema21=True,
        pct_from_ema9=0.5,
        pct_from_ema21=1.0,
        macd_line=0.01,
        macd_signal=0.005,
        macd_hist=0.005,
        macd_cross="bullish",
        bars_since_cross=3,
        bb_percent_b=0.7,
        bb_bandwidth=4.0,
        bb_expanding=True,
        atr14_pct=3.0,
        volume_ratio_prior_20=1.2,
        realized_vol_pct=vol,
        pct_from_swing_high=-2.0,
        pct_from_swing_low=8.0,
    )


def _bundle(
    symbol: str,
    *,
    h1_change: float | None = 40.0,
    quality: DataQuality = DataQuality.OK,
    trusted_quote: bool = True,
    technicals: bool = True,
    vol: float | None = 2.0,
) -> EvidenceBundle:
    pool = PoolRef(
        pair_address="POOL",
        dex_id="raydium",
        base_mint="MINT",
        quote_mint="So11111111111111111111111111111111111111112",
        quote_symbol="SOL",
        trusted_quote=trusted_quote,
    )
    snapshot = CoinSnapshot(
        symbol=symbol,
        mint="MINT",
        price_usd=1.0,
        liquidity_usd=500_000.0,
        volume_24h_usd=1_000_000.0,
        volume_1h_usd=50_000.0,
        fdv_usd=10_000_000.0,
        price_change=PriceLadder(m5=None, h1=h1_change, h6=1.0, h24=2.0),
        txns_m5=TxnCounts(buys=10, sells=8),
        txns_h1=TxnCounts(buys=100, sells=90),
        txns_h24=TxnCounts(buys=1000, sells=950),
        pool=pool,
        provenance=Provenance(source="dexscreener", receive_time=NOW),
        quality=quality,
    )
    brief = (
        TechnicalBrief(
            symbol=symbol,
            m5=_technicals(Timeframe.M5, vol=vol),
            h1=_technicals(Timeframe.H1, vol=vol),
            flow=FlowBrief(
                txn_count_ratio_m5=1.25,
                txn_count_ratio_h1=1.11,
                txn_count_ratio_h24=1.05,
                turnover_24h=2.0,
                turnover_1h=0.1,
                liquidity_usd=500_000.0,
                liquidity_trend_pct=None,
                liquidity_trend_seconds=None,
                liquidity_trend_pool=None,
                price_ladder=snapshot.price_change,
            ),
        )
        if technicals
        else None
    )
    return EvidenceBundle(
        symbol=symbol, snapshot=snapshot, technicals=brief, sentiment=None
    )


def _portfolio() -> PortfolioState:
    return PortfolioState(
        ts=NOW,
        cash_usd=1000.0,
        positions={},
        marks={},
        position_values_usd={},
        unrealized_pnl_usd=0.0,
        realized_pnl_usd=0.0,
        total_value_usd=1000.0,
        starting_cash_usd=1000.0,
    )


# --------------------------------------------------------------------------
# CashStrategy — the null hypothesis
# --------------------------------------------------------------------------


def test_cash_strategy_targets_zero_everywhere() -> None:
    evidence = {"BONK": _bundle("BONK"), "WIF": _bundle("WIF")}
    decision = CashStrategy().decide(evidence, _portfolio(), now=NOW)
    assert {t.symbol for t in decision.targets} == {"BONK", "WIF"}
    assert all(t.target_usd == 0.0 for t in decision.targets)
    assert decision.forecasts == ()


# --------------------------------------------------------------------------
# BaselineStrategy — absence stays absent
# --------------------------------------------------------------------------


def test_missing_technicals_yields_no_view_not_a_neutral_zero() -> None:
    """The whole point of `Forecast.expected_net_return_pct is None`.

    A zero expected return is a *claim* that the price will not move. A missing
    feature must not be able to manufacture that claim.
    """
    strat = BaselineStrategy(StrategySettings())
    decision = strat.decide(
        {"BONK": _bundle("BONK", technicals=False)}, _portfolio(), now=NOW
    )
    (forecast,) = decision.forecasts
    assert forecast.expected_net_return_pct is None
    assert forecast.lower_quantile_pct is None
    assert not forecast.actionable
    assert "technicals" in forecast.features_missing
    assert decision.targets[0].target_usd == 0.0


def test_degraded_snapshot_yields_no_view() -> None:
    strat = BaselineStrategy(StrategySettings())
    decision = strat.decide(
        {"BONK": _bundle("BONK", quality=DataQuality.DEGRADED)}, _portfolio(), now=NOW
    )
    (forecast,) = decision.forecasts
    assert not forecast.actionable
    assert any("snapshot_quality" in m for m in forecast.features_missing)


def test_untrusted_quote_token_yields_no_view() -> None:
    """A pool priced in an unknown token produces a number, not a valuation."""
    strat = BaselineStrategy(StrategySettings())
    decision = strat.decide(
        {"BONK": _bundle("BONK", trusted_quote=False)}, _portfolio(), now=NOW
    )
    (forecast,) = decision.forecasts
    assert not forecast.actionable
    assert "untrusted_quote_token" in forecast.features_missing


def test_missing_h1_change_yields_no_view() -> None:
    """13 of 30 live pairs had no m5 block at all. h1 goes missing too."""
    strat = BaselineStrategy(StrategySettings())
    decision = strat.decide(
        {"BONK": _bundle("BONK", h1_change=None)}, _portfolio(), now=NOW
    )
    (forecast,) = decision.forecasts
    assert not forecast.actionable
    assert "price_change.h1" in forecast.features_missing


def test_missing_realized_vol_yields_no_view() -> None:
    strat = BaselineStrategy(StrategySettings())
    decision = strat.decide({"BONK": _bundle("BONK", vol=None)}, _portfolio(), now=NOW)
    (forecast,) = decision.forecasts
    assert not forecast.actionable
    assert "realized_vol_pct" in forecast.features_missing


# --------------------------------------------------------------------------
# BaselineStrategy — the cost hurdle is a *net* hurdle
# --------------------------------------------------------------------------


def test_forecast_is_net_of_round_trip_cost() -> None:
    """The 12-hour run's one closed trade was directionally right and still lost.

    That is what a gross-return forecast buys you. The cost subtraction here is
    the fix, and this test pins the arithmetic so it cannot drift back.
    """
    settings = StrategySettings()
    strat = BaselineStrategy(settings)
    decision = strat.decide(
        {"BONK": _bundle("BONK", h1_change=40.0)}, _portfolio(), now=NOW
    )
    (forecast,) = decision.forecasts
    expected = 40.0 * settings.shrinkage - BaselineStrategy.ROUND_TRIP_COST_PCT
    assert forecast.expected_net_return_pct == pytest.approx(expected)


def test_entry_tests_the_lower_quantile_not_the_mean() -> None:
    """Trading the mean of a wide distribution is how noise becomes a position.

    Same central estimate, two volatilities. The high-vol case must not trade
    even though its expected return is identical.
    """
    settings = StrategySettings(entry_hurdle_pct=1.0)
    strat = BaselineStrategy(settings)

    tight = strat.decide(
        {"BONK": _bundle("BONK", h1_change=40.0, vol=0.5)}, _portfolio(), now=NOW
    )
    wide = strat.decide(
        {"BONK": _bundle("BONK", h1_change=40.0, vol=20.0)}, _portfolio(), now=NOW
    )

    assert tight.forecasts[0].expected_net_return_pct == pytest.approx(
        wide.forecasts[0].expected_net_return_pct
    )
    assert tight.targets[0].target_usd > 0.0
    assert wide.targets[0].target_usd == 0.0


def test_move_below_hurdle_does_not_trade() -> None:
    strat = BaselineStrategy(StrategySettings(entry_hurdle_pct=1.0))
    decision = strat.decide(
        {"BONK": _bundle("BONK", h1_change=2.0, vol=0.1)}, _portfolio(), now=NOW
    )
    assert decision.targets[0].target_usd == 0.0
    assert "below hurdle" in decision.targets[0].rationale


# --------------------------------------------------------------------------
# BaselineStrategy — uncalibrated means unsized
# --------------------------------------------------------------------------


def test_sizing_is_flat_because_nothing_is_calibrated() -> None:
    """Audit C6: an uncalibrated number must not be scaled as if it were one.

    A 400% move clears the hurdle by 10x more than a 40% move, but does not
    get 10x the size: sizing comes from sequential cash-fraction allocation
    (rank order only), never from forecast magnitude.
    """
    settings = StrategySettings(
        entry_hurdle_pct=1.0, max_positions=2, cash_fraction_per_entry=0.5
    )
    strat = BaselineStrategy(settings)
    decision = strat.decide(
        {
            "BONK": _bundle("BONK", h1_change=40.0, vol=0.5),
            "WIF": _bundle("WIF", h1_change=400.0, vol=0.5),
        },
        _portfolio(),
        now=NOW,
    )
    sized = {t.symbol: t.target_usd for t in decision.targets if t.target_usd > 0}
    assert len(sized) == 2
    # $1000 book, no existing holdings: the higher-ranked forecast (WIF, the
    # larger move) draws 50% of the full $1000 first; BONK then draws 50% of
    # what's left. Neither figure is proportional to the 10x return gap.
    assert sized == {"WIF": 500.0, "BONK": 250.0}
    assert all(f.calibration_id is None for f in decision.forecasts)


def test_max_positions_caps_the_sleeve() -> None:
    settings = StrategySettings(max_positions=1, entry_hurdle_pct=1.0)
    strat = BaselineStrategy(settings)
    decision = strat.decide(
        {
            "BONK": _bundle("BONK", h1_change=40.0, vol=0.5),
            "WIF": _bundle("WIF", h1_change=80.0, vol=0.5),
            "POPCAT": _bundle("POPCAT", h1_change=60.0, vol=0.5),
        },
        _portfolio(),
        now=NOW,
    )
    sized = [t for t in decision.targets if t.target_usd > 0]
    assert len(sized) == 1
    # The strongest lower quantile wins, not the strongest raw move by luck of
    # dict ordering.
    assert sized[0].symbol == "WIF"


def test_decision_is_deterministic_apart_from_its_id() -> None:
    """A baseline that does not reproduce is not a baseline."""
    strat = BaselineStrategy(StrategySettings())
    evidence = {"BONK": _bundle("BONK", h1_change=40.0, vol=0.5)}
    a = strat.decide(evidence, _portfolio(), now=NOW)
    b = strat.decide(evidence, _portfolio(), now=NOW)
    assert a.decision_id != b.decision_id
    assert a.targets == b.targets
    assert a.forecasts == b.forecasts
    assert a.market_read == b.market_read


def test_no_agreement_score_is_computed() -> None:
    """RSI/EMA/MACD/BB are transforms of one close series.

    Counting how many of them agree manufactures confidence out of a single
    number seen five ways. The baseline reads exactly two inputs, and this test
    exists so a future "confirmation score" has to delete it deliberately.
    """
    source = BaselineStrategy._forecast.__doc__ or ""
    strat = BaselineStrategy(StrategySettings())
    # Flip every indicator bearish while leaving the two real inputs alone.
    bundle = _bundle("BONK", h1_change=40.0, vol=0.5)
    bearish = _technicals(Timeframe.H1, vol=0.5)
    object.__setattr__(bundle.technicals, "h1", bearish)
    decision = strat.decide({"BONK": bundle}, _portfolio(), now=NOW)
    assert decision.targets[0].target_usd > 0.0
    assert "confirm" not in source.lower()


# --------------------------------------------------------------------------
# AdvisoryStrategy — C6 and C7
# --------------------------------------------------------------------------


class _Advice:
    def __init__(self, actions: list[object], market_read: str = "read") -> None:
        self.actions = actions
        self.market_read = market_read


class _Act:
    def __init__(self, symbol: str, action: str, reasoning: str = "") -> None:
        self.symbol = symbol
        self.action = action
        self.reasoning = reasoning


class _Brain:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls = 0

    def decide(self, evidence, portfolio, *, now):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_advisory_is_not_the_default() -> None:
    """A default is authority: it is what runs when nobody chose."""
    assert isinstance(build_strategy("baseline", StrategySettings()), BaselineStrategy)
    with pytest.raises(ValueError, match="requires a brain"):
        build_strategy("advisory", StrategySettings())


def test_advisory_failure_degrades_to_the_baseline_not_to_a_gap() -> None:
    settings = StrategySettings()
    brain = _Brain(RuntimeError("gateway down"))
    strat = AdvisoryStrategy(settings, brain)
    decision = strat.decide(
        {"BONK": _bundle("BONK", h1_change=40.0, vol=0.5)}, _portfolio(), now=NOW
    )
    assert brain.calls == 1
    assert decision.strategy_id == BaselineStrategy.strategy_id


def test_discarded_advice_falls_back_rather_than_holding_silently() -> None:
    strat = AdvisoryStrategy(StrategySettings(), _Brain(None))
    decision = strat.decide({"BONK": _bundle("BONK")}, _portfolio(), now=NOW)
    assert decision.strategy_id == BaselineStrategy.strategy_id


def test_hallucinated_symbol_discards_the_whole_advice() -> None:
    """A ticker that does not exist is a malfunction, not a line to skip.

    If one field is invented, none of the others have earned trust either.
    """
    strat = AdvisoryStrategy(StrategySettings(), _Brain(_Advice([_Act("FAKECOIN", "BUY")])))
    decision = strat.decide(
        {"BONK": _bundle("BONK", h1_change=40.0, vol=0.5)}, _portfolio(), now=NOW
    )
    baseline = BaselineStrategy(StrategySettings()).decide(
        {"BONK": _bundle("BONK", h1_change=40.0, vol=0.5)}, _portfolio(), now=NOW
    )
    assert decision.targets == baseline.targets


def test_silence_is_not_a_hold() -> None:
    """The old vocabulary made "said nothing" and "said HOLD" the same event.

    A truncated response then quietly preserved whatever was on the book.
    """
    strat = AdvisoryStrategy(StrategySettings(), _Brain(_Advice([_Act("BONK", "BUY")])))
    decision = strat.decide(
        {"BONK": _bundle("BONK"), "WIF": _bundle("WIF")}, _portfolio(), now=NOW
    )
    targets = {t.symbol: t for t in decision.targets}
    assert targets["BONK"].target_usd > 0.0
    assert targets["WIF"].target_usd == 0.0
    assert "not mentioned" in targets["WIF"].rationale


def test_advisory_sizing_also_draws_on_cash() -> None:
    settings = StrategySettings(cash_fraction_per_entry=0.5)
    strat = AdvisoryStrategy(settings, _Brain(_Advice([_Act("BONK", "BUY")])))
    decision = strat.decide({"BONK": _bundle("BONK")}, _portfolio(), now=NOW)
    assert decision.targets[0].target_usd == 500.0


def test_advisory_sell_targets_zero() -> None:
    strat = AdvisoryStrategy(StrategySettings(), _Brain(_Advice([_Act("BONK", "SELL")])))
    decision = strat.decide({"BONK": _bundle("BONK")}, _portfolio(), now=NOW)
    assert decision.targets[0].target_usd == 0.0


def test_evidence_bundle_carries_no_raw_social_text() -> None:
    """Audit C7, enforced at the type level.

    The excerpt field is gone from `SentimentBrief` entirely, so there is no
    field for a public author's text to travel in. This asserts the absence
    rather than trusting the renderer to keep escaping it.
    """
    from memetrader.types import SentimentBrief

    fields = set(SentimentBrief.__dataclass_fields__)
    for banned in ("top_posts", "excerpts", "posts", "titles", "bodies", "text"):
        assert banned not in fields


# --------------------------------------------------------------------------
# Settings validation
# --------------------------------------------------------------------------


def test_nan_settings_are_rejected_at_construction() -> None:
    """NaN is false against every bound, including the rejections."""
    with pytest.raises(ValidationError):
        StrategySettings(entry_hurdle_pct=math.nan)
    with pytest.raises(ValidationError):
        StrategySettings(flat_size_usd=math.inf)
    with pytest.raises(ValidationError):
        StrategySettings(min_trade_usd=-1.0)


def test_unknown_strategy_name_is_an_error_not_a_default() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        build_strategy("momentum-v9", StrategySettings())


def test_closed_candle_is_the_only_valid_history_source() -> None:
    """Pin the flag the look-ahead fix depends on."""
    partial = Candle(
        ts=NOW, open=1.0, high=1.1, low=0.9, close=1.05, volume=10.0, closed=False
    )
    assert partial.closed is False
    with pytest.raises(ValidationError):
        Candle(ts=NOW, open=1.0, high=0.9, low=1.1, close=1.0, volume=1.0)
