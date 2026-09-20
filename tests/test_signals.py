"""Tests for ``signals.py``.

Every series here is synthetic and built inline. That is deliberate: the shared
JSON fixtures are real captured market data, which is excellent for exercising
the adapters but useless for checking arithmetic — you cannot hand-verify an
RSI against a coin that moved 400% in an hour. Indicator maths needs inputs
whose correct answer is known before the code runs.

Where a magic number appears in an assertion, the comment above it shows the
arithmetic that produced it.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from memetrader.signals import brief, flow_brief, technicals
from memetrader.types import (
    Candle,
    CoinSnapshot,
    PriceLadder,
    Technicals,
    Timeframe,
    TxnCounts,
)

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def candles(closes, *, spread=0.0, volume=1000.0, start_ts=1_700_000_000.0, step=300.0):
    """Turn a list of closes into candles.

    ``spread`` widens high/low symmetrically around the close, which lets a test
    choose whether true range is dominated by the bar's own range or by the gap
    from the previous close. With ``spread=0`` every bar is a doji and TR is
    purely the gap, which is the branch most implementations get wrong.
    """
    out = []
    prev = closes[0] if closes else 0.0
    for i, c in enumerate(closes):
        out.append(
            Candle(
                ts=start_ts + i * step,
                open=prev,
                high=max(c, prev) + spread,
                low=min(c, prev) - spread,
                close=c,
                volume=volume,
            )
        )
        prev = c
    return tuple(out)


def snapshot(
    *,
    price=1.0,
    liquidity=100_000.0,
    volume_24h=500_000.0,
    volume_1h=25_000.0,
    m5=(10, 5),
    h1=(120, 80),
    h24=(2000, 1600),
    ladder=(1.0, 2.0, 3.0, 4.0),
    candles_5m=(),
    candles_1h=(),
):
    return CoinSnapshot(
        symbol="TEST",
        mint="mint-test",
        price_usd=price,
        liquidity_usd=liquidity,
        volume_24h_usd=volume_24h,
        volume_1h_usd=volume_1h,
        fdv_usd=1_000_000.0,
        price_change=PriceLadder(*ladder),
        txns_m5=TxnCounts(*m5),
        txns_h1=TxnCounts(*h1),
        txns_h24=TxnCounts(*h24),
        pair_address="pair-test",
        dex_id="raydium",
        pair_created_at=1_699_000_000.0,
        candles_5m=candles_5m,
        candles_1h=candles_1h,
    )


def float_fields(tech: Technicals):
    """Every ``float | None`` field that is currently not None."""
    out = {}
    for field in dataclasses.fields(tech):
        value = getattr(tech, field.name)
        if isinstance(value, float):
            out[field.name] = value
    return out


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

# The series from Wilder's worked example, as reproduced in essentially every
# TA reference. Only the first 15 closes are needed to produce the first RSI.
WILDER_CLOSES = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
    45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00,
    46.03, 46.41, 46.22, 45.64, 46.21, 46.25, 45.71, 46.45,
]


def test_rsi_matches_hand_computed_wilder_seed():
    # 15 closes -> 14 changes, which is exactly the seeding window.
    #   gains  = .06+.72+.50+.27+.32+.42+.24+.14+.67 = 3.34  -> avg 3.34/14
    #   losses = .25+.54+.19+.42                     = 1.40  -> avg 1.40/14
    #   RS  = 3.34/1.40 = 2.3857142857
    #   RSI = 100 - 100/(1 + RS) = 100 * 3.34/(3.34 + 1.40) = 70.4641350211
    tech = technicals(candles(WILDER_CLOSES[:15]), Timeframe.M5)
    assert tech.rsi14 == pytest.approx(100.0 * 3.34 / 4.74, abs=1e-9)
    assert tech.rsi14 == pytest.approx(70.4641350211, abs=1e-9)
    # One bar of seeding only — there is no prior RSI to compare against yet.
    assert tech.rsi14_rising is None


def test_rsi_rising_flag_uses_the_previous_bar():
    # Close 16 is 46.00, a -0.28 change, so the smoothed RSI must fall. One
    # Wilder step (drop 1/14 of the old average, add 1/14 of the new value):
    #   avg gain = (3.34/14) * 13/14        = 43.42/196
    #   avg loss = (1.40/14) * 13/14 + 0.28/14 = (18.20 + 3.92)/196 = 22.12/196
    #   RSI = 100 * 43.42/(43.42 + 22.12) = 4342/65.54 = 66.2496185536
    tech = technicals(candles(WILDER_CLOSES[:16]), Timeframe.M5)
    assert tech.rsi14 == pytest.approx(100.0 * 43.42 / 65.54, abs=1e-9)
    assert tech.rsi14 == pytest.approx(66.2496185536, abs=1e-9)
    assert tech.rsi14_rising is False


def test_rsi_saturates_at_100_and_0():
    rising = technicals(candles([float(i) for i in range(1, 31)]), Timeframe.M5)
    assert rising.rsi14 == 100.0
    assert rising.rsi14_rising is False  # 100 -> 100 is flat, not rising

    falling = technicals(candles([float(i) for i in range(30, 0, -1)]), Timeframe.M5)
    assert falling.rsi14 == 0.0


def test_rsi_needs_fifteen_candles():
    assert technicals(candles([float(i) for i in range(14)]), Timeframe.M5).rsi14 is None
    assert technicals(candles([float(i) for i in range(15)]), Timeframe.M5).rsi14 is not None


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


def test_ema9_seed_is_the_simple_mean_of_the_first_nine():
    # closes 1..9 -> seed = mean = 5.0, and that is the first EMA9 value.
    tech = technicals(candles([float(i) for i in range(1, 10)]), Timeframe.M5)
    assert tech.ema9 == pytest.approx(5.0, abs=1e-12)
    # close == 9, ema == 5 -> (9-5)/5*100 = 80% above the EMA.
    assert tech.pct_from_ema9 == pytest.approx(80.0, abs=1e-12)
    # 21 bars are needed before EMA21 exists, so the comparison is undefined.
    assert tech.ema21 is None
    assert tech.ema9_above_ema21 is None


def test_ema9_recursion_matches_hand_computation():
    # Tenth bar: alpha = 2/(9+1) = 0.2, prev = 5.0, close = 10.0
    #   ema = 5.0 + 0.2 * (10.0 - 5.0) = 6.0
    tech = technicals(candles([float(i) for i in range(1, 11)]), Timeframe.M5)
    assert tech.ema9 == pytest.approx(6.0, abs=1e-12)
    # Eleventh bar: 6.0 + 0.2 * (11.0 - 6.0) = 7.0
    tech = technicals(candles([float(i) for i in range(1, 12)]), Timeframe.M5)
    assert tech.ema9 == pytest.approx(7.0, abs=1e-12)


def test_ema_ordering_flag():
    rising = technicals(candles([float(i) for i in range(1, 41)]), Timeframe.M5)
    assert rising.ema9_above_ema21 is True
    assert rising.ema9 > rising.ema21

    falling = technicals(candles([float(i) for i in range(40, 0, -1)]), Timeframe.M5)
    assert falling.ema9_above_ema21 is False


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


def macd_turn_closes():
    """A long accelerating decline, then an accelerating rally.

    Two properties are deliberate. The decline is long enough that the histogram
    is already valid (34 bars) and negative *before* the turn — otherwise the
    crossover happens inside the warm-up window and there is nothing to detect.
    And both legs are curved rather than straight: on a perfectly linear ramp
    both EMAs settle at a constant lag, the histogram sits at exactly zero, and
    the test would be measuring floating-point noise instead of a crossover.
    """
    down = [100.0 - 0.02 * i * i for i in range(46)]  # 100.00 .. 59.50
    up = [down[-1] + 1.5 * j + 0.02 * j * j for j in range(1, 26)]
    return down + up


def test_macd_hist_sign_flips_at_the_cross():
    closes = macd_turn_closes()
    series = candles(closes)

    # Walk forward to the bar where the implementation reports a brand-new cross.
    cross_at = None
    for n in range(35, len(series) + 1):
        if technicals(series[:n], Timeframe.M5).bars_since_cross == 0:
            cross_at = n
            break
    assert cross_at is not None, "constructed series should contain a MACD cross"

    at_cross = technicals(series[:cross_at], Timeframe.M5)
    before = technicals(series[: cross_at - 1], Timeframe.M5)

    assert at_cross.macd_cross == "bullish"
    # The cross claim must be backed by an actual sign flip in the histogram.
    assert at_cross.macd_hist > 0.0
    assert before.macd_hist <= 0.0
    assert at_cross.macd_hist == pytest.approx(
        at_cross.macd_line - at_cross.macd_signal, abs=1e-12
    )


def test_bars_since_cross_counts_bars_not_crosses():
    closes = macd_turn_closes()
    series = candles(closes)

    cross_at = None
    for n in range(35, len(series) + 1):
        if technicals(series[:n], Timeframe.M5).bars_since_cross == 0:
            cross_at = n
            break
    assert cross_at is not None

    # The rally continues, so no second cross: the counter just ages.
    for extra in range(1, 6):
        tech = technicals(series[: cross_at + extra], Timeframe.M5)
        assert tech.macd_cross == "bullish"
        assert tech.bars_since_cross == extra


def test_macd_reports_none_without_a_cross_in_range():
    # A steadily compounding uptrend keeps MACD above its signal throughout the
    # valid window, so there is nothing to report.
    tech = technicals(candles([100.0 * 1.02**i for i in range(60)]), Timeframe.M5)
    assert tech.macd_hist is not None
    assert tech.macd_cross == "none"
    assert tech.bars_since_cross is None


def test_macd_signal_needs_thirty_four_bars():
    ramp = [100.0 + math.sin(i / 3.0) * 5.0 for i in range(40)]
    assert technicals(candles(ramp[:33]), Timeframe.M5).macd_hist is None
    assert technicals(candles(ramp[:34]), Timeframe.M5).macd_hist is not None


# ---------------------------------------------------------------------------
# Bollinger
# ---------------------------------------------------------------------------


def upper_band_close(prefix):
    """The close that lands exactly on the 20/2 upper band, in closed form.

    With 19 known closes (sum S, sum of squares Q) and one unknown x, requiring
    ``x - mean == 2 * sigma`` over the 20-value window gives

        (19x - S)^2 / 400 = 4 * [20(Q + x^2) - (S + x)^2] / 400
        =>  57 x^2 - 6 S x + (S^2 - 16 Q) = 0
        =>  x = [6S + sqrt(3648 Q - 192 S^2)] / 114        (upper root)

    using the population sigma (ddof=0) that Bollinger's definition specifies.
    """
    s = sum(prefix)
    q = sum(v * v for v in prefix)
    return (6 * s + math.sqrt(3648 * q - 192 * s * s)) / 114


def test_percent_b_is_exactly_one_on_the_upper_band():
    prefix = [10.0 + i for i in range(1, 20)]  # 11 .. 29
    x = upper_band_close(prefix)
    tech = technicals(candles(prefix + [x]), Timeframe.M5)
    assert tech.bb_percent_b == pytest.approx(1.0, abs=1e-9)


def test_percent_b_exceeds_one_above_the_band_and_is_not_clamped():
    prefix = [10.0 + i for i in range(1, 20)]
    x = upper_band_close(prefix)
    tech = technicals(candles(prefix + [x + 5.0]), Timeframe.M5)
    assert tech.bb_percent_b > 1.0
    # Below the lower band it must go negative for the same reason.
    low = technicals(candles(prefix + [0.0]), Timeframe.M5)
    assert low.bb_percent_b < 0.0


def test_bandwidth_is_a_whole_percent_and_expansion_is_relative():
    quiet = [100.0, 100.5] * 10
    tech_quiet = technicals(candles(quiet), Timeframe.M5)
    # Widening the last bar must widen the bands.
    tech_loud = technicals(candles(quiet[:-1] + [130.0]), Timeframe.M5)
    assert tech_loud.bb_bandwidth > tech_quiet.bb_bandwidth
    assert tech_loud.bb_bandwidth > 1.0  # whole percent, not a 0..1 fraction
    assert technicals(candles(quiet + [130.0]), Timeframe.M5).bb_expanding is True


def test_bollinger_needs_twenty_bars():
    ramp = [float(i) for i in range(1, 25)]
    assert technicals(candles(ramp[:19]), Timeframe.M5).bb_percent_b is None
    assert technicals(candles(ramp[:20]), Timeframe.M5).bb_percent_b is not None
    assert technicals(candles(ramp[:20]), Timeframe.M5).bb_expanding is None
    assert technicals(candles(ramp[:21]), Timeframe.M5).bb_expanding is not None


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


def test_atr_uses_the_gap_from_the_previous_close():
    # 14 flat bars (high 101 / low 100 / close 100, so TR = 1), then a bar that
    # gaps to 120/121. Its true range is |121 - 100| = 21, not the 1.0 you get
    # from high-low alone.
    bars = []
    for i in range(14):
        bars.append(Candle(ts=i * 300.0, open=100.0, high=101.0, low=100.0, close=100.0, volume=10.0))
    bars.append(Candle(ts=14 * 300.0, open=120.0, high=121.0, low=120.0, close=120.0, volume=10.0))

    # 14 true ranges (bars 1..14): thirteen 1.0s and one 21.0.
    #   Wilder seed = (13 * 1 + 21) / 14 = 34/14 = 2.4285714286
    #   as a percent of the 120.0 close = 2.4285714286 / 120 * 100
    tech = technicals(tuple(bars), Timeframe.M5)
    assert tech.atr14_pct == pytest.approx(34.0 / 14.0 / 120.0 * 100.0, abs=1e-10)
    # Sanity: the high-low-only answer would have been 1/120*100 = 0.833%.
    assert tech.atr14_pct > 2.0


def test_atr_smoothing_is_wilder_not_a_rolling_mean():
    # Same bars plus one more flat bar. Wilder: 34/14 + (1 - 34/14)/14.
    bars = [
        Candle(ts=i * 300.0, open=100.0, high=101.0, low=100.0, close=100.0, volume=10.0)
        for i in range(14)
    ]
    bars.append(Candle(ts=4200.0, open=120.0, high=121.0, low=120.0, close=120.0, volume=10.0))
    bars.append(Candle(ts=4500.0, open=120.0, high=121.0, low=120.0, close=120.0, volume=10.0))

    seed = 34.0 / 14.0
    expected_atr = seed + (1.0 - seed) / 14.0
    tech = technicals(tuple(bars), Timeframe.M5)
    assert tech.atr14_pct == pytest.approx(expected_atr / 120.0 * 100.0, abs=1e-10)


def test_atr_needs_fifteen_candles():
    ramp = [float(i) for i in range(1, 17)]
    assert technicals(candles(ramp[:14], spread=0.5), Timeframe.M5).atr14_pct is None
    assert technicals(candles(ramp[:15], spread=0.5), Timeframe.M5).atr14_pct is not None


# ---------------------------------------------------------------------------
# Volume and swings
# ---------------------------------------------------------------------------


def test_volume_ratio_is_latest_over_the_twenty_bar_mean():
    bars = list(candles([100.0] * 19, volume=100.0))
    bars.append(Candle(ts=99_999.0, open=100.0, high=100.0, low=100.0, close=100.0, volume=400.0))
    # mean of nineteen 100s and one 400 = (1900 + 400)/20 = 115.0
    tech = technicals(tuple(bars), Timeframe.M5)
    assert tech.volume_ratio_20 == pytest.approx(400.0 / 115.0, abs=1e-12)


def test_volume_ratio_is_none_when_nothing_traded():
    tech = technicals(candles([100.0] * 25, volume=0.0), Timeframe.M5)
    assert tech.volume_ratio_20 is None


def test_swing_distances_have_the_documented_signs():
    # Last 20 bars run 100 up to 119, then pull back to 110.
    closes = [100.0 + i for i in range(20)] + [110.0]
    tech = technicals(candles(closes), Timeframe.M5)
    # Lookback window is the last 20 bars: closes 101..119 plus the 110 pullback.
    # Highest high in that window is 119, lowest low is 100 (bar 101's open).
    assert tech.pct_from_swing_high < 0.0
    assert tech.pct_from_swing_low > 0.0
    assert tech.pct_from_swing_high == pytest.approx((110.0 - 119.0) / 119.0 * 100.0, abs=1e-9)

    # Making a new high puts price exactly on it.
    at_high = technicals(candles([100.0 + i for i in range(25)]), Timeframe.M5)
    assert at_high.pct_from_swing_high == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_five_candles_yields_all_none():
    tech = technicals(candles([1.0, 2.0, 3.0, 2.5, 3.5]), Timeframe.M5)
    assert tech.candles_used == 5
    assert tech.timeframe is Timeframe.M5
    for field in dataclasses.fields(tech):
        if field.name in {"timeframe", "candles_used"}:
            continue
        assert getattr(tech, field.name) is None, f"{field.name} should be None on 5 candles"


def test_empty_sequence_does_not_raise():
    tech = technicals((), Timeframe.H1)
    assert tech.candles_used == 0
    assert tech.timeframe is Timeframe.H1
    for field in dataclasses.fields(tech):
        if field.name in {"timeframe", "candles_used"}:
            continue
        assert getattr(tech, field.name) is None

    # A list, not just a tuple, and a single bar.
    assert technicals([], Timeframe.M5).candles_used == 0
    assert technicals(list(candles([1.0])), Timeframe.M5).candles_used == 1


def test_identical_closes_produce_no_nan_or_inf():
    bars = tuple(
        Candle(ts=i * 300.0, open=5.0, high=5.0, low=5.0, close=5.0, volume=100.0)
        for i in range(60)
    )
    tech = technicals(bars, Timeframe.M5)
    assert tech.candles_used == 60
    for name, value in float_fields(tech).items():
        assert math.isfinite(value), f"{name} is {value} on a flat series"

    # A flat market is neither overbought nor oversold, and has zero volatility.
    assert tech.rsi14 == pytest.approx(50.0, abs=1e-12)
    assert tech.atr14_pct == pytest.approx(0.0, abs=1e-12)
    assert tech.bb_bandwidth == pytest.approx(0.0, abs=1e-12)
    assert tech.volume_ratio_20 == pytest.approx(1.0, abs=1e-12)
    assert tech.macd_cross == "none"
    assert tech.bars_since_cross is None
    # Bands of zero width make %B a 0/0 question, which has no answer worth
    # showing a trading model.
    assert tech.bb_percent_b is None


def test_zero_prices_do_not_divide_by_zero():
    bars = tuple(
        Candle(ts=i * 300.0, open=0.0, high=0.0, low=0.0, close=0.0, volume=0.0)
        for i in range(60)
    )
    tech = technicals(bars, Timeframe.M5)
    for name, value in float_fields(tech).items():
        assert math.isfinite(value), f"{name} is {value} on a zero-priced series"


def test_all_float_fields_finite_on_a_normal_series():
    closes = [100.0 + math.sin(i / 4.0) * 8.0 + i * 0.3 for i in range(80)]
    tech = technicals(candles(closes, spread=0.4), Timeframe.H1)
    assert tech.timeframe is Timeframe.H1
    assert tech.candles_used == 80
    for name, value in float_fields(tech).items():
        assert math.isfinite(value), f"{name} is {value}"
    assert tech.rsi14 is not None and 0.0 <= tech.rsi14 <= 100.0


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------


def test_flow_brief_passes_through_and_derives_turnover():
    snap = snapshot(liquidity=200_000.0, volume_24h=1_000_000.0, volume_1h=50_000.0)
    flow = flow_brief(snap)

    assert flow.buy_sell_ratio_m5 == pytest.approx(10 / 5)
    assert flow.buy_sell_ratio_h1 == pytest.approx(120 / 80)
    assert flow.buy_sell_ratio_h24 == pytest.approx(2000 / 1600)
    assert flow.turnover_24h == pytest.approx(5.0)
    assert flow.turnover_1h == pytest.approx(0.25)
    assert flow.liquidity_usd == 200_000.0
    assert flow.liquidity_trend_pct is None  # no previous snapshot
    assert flow.price_ladder is snap.price_change


def test_flow_brief_survives_an_infinite_buy_sell_ratio():
    snap = snapshot(m5=(7, 0))
    flow = flow_brief(snap)
    assert flow.buy_sell_ratio_m5 == math.inf
    # Nothing else may have been contaminated by that infinity.
    assert math.isfinite(flow.turnover_24h)
    assert math.isfinite(flow.turnover_1h)


def test_flow_brief_guards_zero_liquidity():
    flow = flow_brief(snapshot(liquidity=0.0))
    assert flow.turnover_24h == 0.0
    assert flow.turnover_1h == 0.0
    assert math.isfinite(flow.turnover_24h)


def test_liquidity_trend_is_a_whole_percent_against_the_previous_snapshot():
    previous = snapshot(liquidity=100_000.0)
    drained = flow_brief(snapshot(liquidity=75_000.0), previous)
    assert drained.liquidity_trend_pct == pytest.approx(-25.0)

    grown = flow_brief(snapshot(liquidity=130_000.0), previous)
    assert grown.liquidity_trend_pct == pytest.approx(30.0)

    # A previous snapshot with no liquidity gives no percentage, not infinity.
    assert flow_brief(snapshot(), snapshot(liquidity=0.0)).liquidity_trend_pct is None


def test_liquidity_trend_carries_the_window_it_was_measured_over():
    """The percentage is unreadable without the window: -6% is a wobble over a
    day and an exit over a quarter of an hour. The caller supplies the gap
    because only the caller knows which of its two cadences produced the pair."""
    previous = snapshot(liquidity=100_000.0)
    drained = flow_brief(snapshot(liquidity=94_000.0), previous, elapsed_seconds=880.0)
    assert drained.liquidity_trend_pct == pytest.approx(-6.0)
    assert drained.liquidity_trend_seconds == pytest.approx(880.0)

    # The same two reads 60 seconds apart are the same percentage over a wholly
    # different window, and the brief has to be able to say so.
    fast = flow_brief(snapshot(liquidity=94_000.0), previous, elapsed_seconds=60.0)
    assert fast.liquidity_trend_pct == pytest.approx(-6.0)
    assert fast.liquidity_trend_seconds == pytest.approx(60.0)


def test_the_liquidity_window_is_none_whenever_the_trend_is():
    """Missing is never zero, and a window without a trend is worse than either:
    "stable over 15m" is a claim, and one point of data does not support it."""
    alone = flow_brief(snapshot())
    assert alone.liquidity_trend_pct is None
    assert alone.liquidity_trend_seconds is None

    # An elapsed time offered with no baseline to pair it with is still no trend.
    orphan = flow_brief(snapshot(), None, elapsed_seconds=900.0)
    assert orphan.liquidity_trend_pct is None
    assert orphan.liquidity_trend_seconds is None

    # A zero-liquidity baseline yields no percentage, so it yields no window.
    undefined = flow_brief(snapshot(), snapshot(liquidity=0.0), elapsed_seconds=900.0)
    assert undefined.liquidity_trend_pct is None
    assert undefined.liquidity_trend_seconds is None

    # And a comparison the caller could not time reports the trend without
    # inventing a window for it.
    untimed = flow_brief(snapshot(liquidity=75_000.0), snapshot(liquidity=100_000.0))
    assert untimed.liquidity_trend_pct == pytest.approx(-25.0)
    assert untimed.liquidity_trend_seconds is None


# ---------------------------------------------------------------------------
# brief()
# ---------------------------------------------------------------------------


def test_brief_composes_both_timeframes_and_flow():
    closes = [100.0 + math.sin(i / 5.0) * 6.0 for i in range(60)]
    snap = snapshot(
        candles_5m=candles(closes, spread=0.3),
        candles_1h=candles(closes[:40], spread=0.3, step=3600.0),
    )
    out = brief(snap)

    assert out.symbol == "TEST"
    assert out.m5.timeframe is Timeframe.M5
    assert out.h1.timeframe is Timeframe.H1
    assert out.m5.candles_used == 60
    assert out.h1.candles_used == 40
    assert out.flow.turnover_1h == pytest.approx(25_000.0 / 100_000.0)
    assert out.flow.liquidity_trend_pct is None


def test_brief_without_candles_is_all_none_but_still_has_flow():
    snap = snapshot()
    out = brief(snap, previous=snapshot(liquidity=50_000.0))

    assert out.m5.candles_used == 0
    assert out.h1.candles_used == 0
    assert out.m5.rsi14 is None
    assert out.flow.liquidity_trend_pct == pytest.approx(100.0)


def test_brief_passes_the_liquidity_window_through_to_the_flow():
    out = brief(
        snapshot(liquidity=120_000.0),
        previous=snapshot(liquidity=100_000.0),
        elapsed_seconds=900.0,
    )
    assert out.flow.liquidity_trend_pct == pytest.approx(20.0)
    assert out.flow.liquidity_trend_seconds == pytest.approx(900.0)
