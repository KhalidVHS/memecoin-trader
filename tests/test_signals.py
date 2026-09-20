"""Tests for ``signals.py``: indicator arithmetic, and the audit's two rules.

The arithmetic tests are hand-computed rather than compared against a reference
implementation, because a reference implementation is exactly the thing we do
not have — there is no TA-Lib here, and "it matches what the code did yesterday"
is a snapshot test that locks in whatever was wrong. Every expected number below
can be worked out on paper from the definition in the indicator's docstring.

The other half of the file exists because of the audit, and asserts absences:

* **Look-ahead.** ``test_mutating_the_open_bar_moves_nothing`` is the load-
  bearing test in this file. Every feature must be a function of closed bars
  only, so rewriting the in-progress bar into an enormous spike must change
  nothing at all.
* **Pool identity.** A liquidity trend may not be reported across a pair switch.
* **Missing is not zero.** A ratio with an unobserved side is ``None``; turnover
  against zero or unknown depth is ``None``.

Tests deleted, and why:

* ``test_volume_ratio_20_*`` — renamed to ``volume_ratio_prior_20`` and
  re-specified. The old tests asserted the buggy definition (the measured bar
  inside its own baseline), so they could not be kept; the replacements assert
  a strictly-preceding baseline.
* ``test_turnover_is_zero_when_liquidity_is_zero`` — inverted, not deleted. It
  asserted the exact bug this remediation removes. Its replacement,
  ``test_turnover_is_none_when_depth_is_zero_or_unknown``, asserts ``None``.
* ``test_flow_ratio_defaults_to_one_for_quiet_pairs`` — deleted outright. A 1.0
  computed from an absent block is the manufactured-neutrality bug; there is
  nothing in its intent worth keeping.
* ``test_liquidity_trend_matches_previous_snapshot`` — superseded by the pair-
  switch tests. The original matched on symbol only, which is the bug.
* The ``technicals(list_of_candles)`` call signature — ``technicals`` now takes
  a ``CandleSeries`` so the result can name the pool and the closed-bar
  watermark can be enforced in one place rather than at each call site.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from memetrader import signals
from memetrader.types import (
    Candle,
    CandleSeries,
    CoinSnapshot,
    PoolRef,
    PriceLadder,
    Provenance,
    Timeframe,
    TxnCounts,
)

START_TS = 1_700_000_000.0
POOL_A = "PoolAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
POOL_B = "PoolBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_candles(
    closes: list[float],
    *,
    spread: float = 0.01,
    volume: float | list[float] = 100.0,
    start_ts: float = START_TS,
    step: float = 300.0,
    last_open: bool = False,
) -> tuple[Candle, ...]:
    """OHLCV bars with ``close`` as given and a symmetric ``spread`` around it.

    ``last_open`` marks the final bar in progress, which is the live case: a
    read almost always lands inside the current period.
    """
    volumes = volume if isinstance(volume, list) else [volume] * len(closes)
    out = []
    for i, (close, vol) in enumerate(zip(closes, volumes, strict=True)):
        out.append(
            Candle(
                ts=start_ts + i * step,
                open=close,
                high=close * (1.0 + spread),
                low=close * (1.0 - spread),
                close=close,
                volume=vol,
                closed=not (last_open and i == len(closes) - 1),
            )
        )
    return tuple(out)


def make_series(
    closes: list[float],
    *,
    timeframe: Timeframe = Timeframe.M5,
    pool_address: str = POOL_A,
    interval_seconds: float = 300.0,
    missing_intervals: int = 0,
    **kwargs,
) -> CandleSeries:
    candles = make_candles(closes, step=interval_seconds, **kwargs)
    return CandleSeries(
        timeframe=timeframe,
        pool_address=pool_address,
        candles=candles,
        interval_seconds=interval_seconds,
        provenance=Provenance(source="test", receive_time=START_TS + 100_000.0),
        missing_intervals=missing_intervals,
    )


def pool(address: str = POOL_A, **kwargs) -> PoolRef:
    base = {
        "pair_address": address,
        "dex_id": "orca",
        "base_mint": "BaseMint",
        "quote_mint": "So11111111111111111111111111111111111111112",
        "quote_symbol": "SOL",
        "created_at": START_TS - 90 * 86_400.0,
    }
    return PoolRef(**{**base, **kwargs})


def make_snapshot(
    *,
    symbol: str = "BONK",
    price_usd: float | None = 1.0,
    liquidity_usd: float | None = 100_000.0,
    volume_24h_usd: float | None = 500_000.0,
    volume_1h_usd: float | None = 20_000.0,
    fdv_usd: float | None = 5_000_000.0,
    price_change: PriceLadder | None = None,
    txns_m5: TxnCounts | None = None,
    txns_h1: TxnCounts | None = None,
    txns_h24: TxnCounts | None = None,
    pool_ref: PoolRef | None = None,
    receive_time: float = START_TS,
    candles_5m: CandleSeries | None = None,
    candles_1h: CandleSeries | None = None,
) -> CoinSnapshot:
    return CoinSnapshot(
        symbol=symbol,
        mint="BaseMint",
        price_usd=price_usd,
        liquidity_usd=liquidity_usd,
        volume_24h_usd=volume_24h_usd,
        volume_1h_usd=volume_1h_usd,
        fdv_usd=fdv_usd,
        price_change=price_change or PriceLadder(m5=None, h1=1.0, h6=2.0, h24=-3.0),
        txns_m5=txns_m5 or TxnCounts(buys=10, sells=5),
        txns_h1=txns_h1 or TxnCounts(buys=100, sells=50),
        txns_h24=txns_h24 or TxnCounts(buys=1000, sells=500),
        pool=pool_ref or pool(),
        provenance=Provenance(source="dexscreener", receive_time=receive_time),
        candles_5m=candles_5m,
        candles_1h=candles_1h,
    )


# ---------------------------------------------------------------------------
# Smoothing primitives — the two that everything else is built on
# ---------------------------------------------------------------------------


def test_wilder_smoothing_uses_alpha_one_over_period():
    """Hand-computed: seed = mean(1, 2) = 1.5, then level += (x - level)/2.

    ``ewm(span=2)`` would use alpha = 2/3 and give 2.333/3.444 instead. Both
    look like plausible indicator values, which is why this is pinned.
    """
    out = signals._wilder_smoothed(np.array([1.0, 2.0, 3.0, 4.0]), 2)
    assert math.isnan(out[0])  # warm-up, deliberately not a number
    assert out[1] == pytest.approx(1.5)
    assert out[2] == pytest.approx(2.25)
    assert out[3] == pytest.approx(3.125)


def test_wilder_smoothing_is_all_nan_when_too_short():
    out = signals._wilder_smoothed(np.array([1.0, 2.0]), 14)
    assert np.isnan(out).all()


def test_ema_is_seeded_with_the_sma_not_the_first_observation():
    """Hand-computed with alpha = 2/(2+1) = 2/3 and seed mean(1, 2) = 1.5.

    ``ewm(adjust=False)`` seeds at the first observation (1.0) and would give
    1.667/2.556/3.519 — one arbitrary early bar steering the series.
    """
    out = signals._sma_seeded_ema(np.array([1.0, 2.0, 3.0, 4.0]), 2)
    assert math.isnan(out[0])
    assert out[1] == pytest.approx(1.5)
    assert out[2] == pytest.approx(2.5)
    assert out[3] == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# Indicator arithmetic
# ---------------------------------------------------------------------------


def test_rsi_pins_at_100_when_every_bar_rises():
    tech = signals.technicals(make_series([float(i) for i in range(1, 41)]))
    assert tech.rsi14 == pytest.approx(100.0)
    assert tech.rsi14_rising is False  # already at the ceiling, so not rising


def test_rsi_pins_at_0_when_every_bar_falls():
    tech = signals.technicals(make_series([float(i) for i in range(40, 0, -1)]))
    assert tech.rsi14 == pytest.approx(0.0)


def test_rsi_is_50_on_a_flat_series_not_100():
    """0/0 and x/0 are different questions. A flat market is neither overbought
    nor oversold; only *gains against zero losses* is 100."""
    tech = signals.technicals(make_series([100.0] * 40))
    assert tech.rsi14 == pytest.approx(50.0)


def test_rsi_is_none_before_its_warmup_completes():
    """14 changes need 15 closes. Reporting 0.0 or 50.0 here would be the
    missing-is-zero bug wearing an indicator's name."""
    tech = signals.technicals(make_series([float(i) for i in range(1, 15)]))
    assert tech.rsi14 is None
    assert tech.candles_used == 14


def test_ema_ordering_and_distance_on_a_rising_series():
    closes = [float(i) for i in range(1, 61)]
    tech = signals.technicals(make_series(closes))
    assert tech.ema9 is not None and tech.ema21 is not None
    assert tech.ema9 > tech.ema21  # fast leads on a monotone rise
    assert tech.ema9_above_ema21 is True
    # Price is above both, so both distances are positive whole percents.
    assert tech.pct_from_ema9 > 0.0
    assert tech.pct_from_ema21 > tech.pct_from_ema9


def test_ema21_is_none_with_twenty_candles():
    tech = signals.technicals(make_series([float(i) for i in range(1, 21)]))
    assert tech.ema9 is not None
    assert tech.ema21 is None
    assert tech.ema9_above_ema21 is None  # not False: it is unknown
    assert tech.pct_from_ema21 is None


def test_macd_is_zero_on_a_flat_series():
    """Both EMAs converge on the same constant, so the line, signal and
    histogram are all exactly zero — and there is no cross to report."""
    tech = signals.technicals(make_series([100.0] * 60))
    assert tech.macd_line == pytest.approx(0.0, abs=1e-9)
    assert tech.macd_signal == pytest.approx(0.0, abs=1e-9)
    assert tech.macd_hist == pytest.approx(0.0, abs=1e-9)
    assert tech.macd_cross == "none"
    assert tech.bars_since_cross is None  # a cross that never happened has no age


def test_macd_line_is_positive_while_price_rises():
    tech = signals.technicals(make_series([float(i) for i in range(1, 61)]))
    assert tech.macd_line > 0.0


def test_macd_cross_reports_direction_and_freshness():
    """Down for 40 bars then up for 8: the histogram must flip bullish, and the
    flip must be recent. Age is the whole point — a 40-bar-old cross is already
    priced in."""
    closes = [100.0 - i for i in range(40)] + [61.0 + 3.0 * i for i in range(1, 9)]
    tech = signals.technicals(make_series(closes))
    assert tech.macd_cross == "bullish"
    assert tech.bars_since_cross is not None
    assert tech.bars_since_cross < 8


def test_macd_is_none_before_warmup():
    tech = signals.technicals(make_series([float(i) for i in range(1, 26)]))
    assert tech.macd_line is None
    assert tech.macd_cross is None


def test_bollinger_percent_b_is_half_at_the_midline():
    """A flat window has zero width: %B is 0/0, which has no answer, so it is
    None rather than the superficially reasonable 0.5."""
    tech = signals.technicals(make_series([100.0] * 30))
    assert tech.bb_percent_b is None
    assert tech.bb_bandwidth == pytest.approx(0.0)


def test_bollinger_percent_b_exceeds_one_on_a_breakout():
    """Left unclamped on purpose: the excursion outside the band is the signal,
    and clamping to [0, 1] deletes exactly the information wanted."""
    closes = [100.0] * 25 + [140.0]
    tech = signals.technicals(make_series(closes))
    assert tech.bb_percent_b > 1.0


def test_bollinger_bandwidth_expands_when_volatility_rises():
    quiet = [100.0 + (i % 2) * 0.1 for i in range(40)]
    tech_quiet = signals.technicals(make_series(quiet))
    loud = quiet[:30] + [100.0 + (i % 2) * 20.0 for i in range(10)]
    tech_loud = signals.technicals(make_series(loud))
    assert tech_loud.bb_bandwidth > tech_quiet.bb_bandwidth
    assert tech_loud.bb_expanding is True


def test_atr_percent_is_hand_computable_on_a_flat_series():
    """Every bar is 100 with a 1% spread, so high=101, low=99 and the previous
    close is 100. True range = max(101-99, |101-100|, |99-100|) = 2, ATR = 2,
    and 2/100 is 2.0%."""
    tech = signals.technicals(make_series([100.0] * 30, spread=0.01))
    assert tech.atr14_pct == pytest.approx(2.0)


def test_atr_uses_gaps_not_just_the_bar_range():
    """Memecoins gap constantly. An ATR built from high-low alone understates
    volatility badly and makes a -15% stop look far safer than it is."""
    # Each bar's own range is exactly 1.0, but each one opens 10.0 above the
    # previous close. True range is therefore max(1, 10, 9) = 10 on every bar.
    close = np.array([10.0 * i for i in range(1, 21)])
    high = close
    low = close - 1.0
    atr = signals._atr(high, low, close)
    assert atr[-1] == pytest.approx(10.0)
    # The number a high-minus-low ATR would have produced — ten times too small,
    # which is how a -15% stop ends up looking comfortably wide.
    assert float((high - low).mean()) == pytest.approx(1.0)


def test_realized_vol_is_zero_on_a_flat_series_and_none_when_short():
    assert signals.technicals(make_series([100.0] * 25)).realized_vol_pct == pytest.approx(
        0.0
    )
    assert signals.technicals(make_series([100.0] * 10)).realized_vol_pct is None


def test_realized_vol_is_not_annualized():
    """A 1% per-bar move must read as roughly 1, not as a four-figure annualized
    number. Annualizing a 5-minute memecoin bar with gaps is a claim about how
    many such bars fit in a year, and that claim is wrong."""
    closes = [100.0 * (1.01 ** (i % 2)) for i in range(30)]
    vol = signals.technicals(make_series(closes)).realized_vol_pct
    assert 0.5 < vol < 2.0


def test_distance_to_swing_high_and_low():
    """Closes are flat at 100 with a 1% spread, so the 20-bar swing high is 101
    and the swing low is 99. (100-101)/101 = -0.9901%, (100-99)/99 = +1.0101%."""
    tech = signals.technicals(make_series([100.0] * 30, spread=0.01))
    assert tech.pct_from_swing_high == pytest.approx(-0.990099, rel=1e-4)
    assert tech.pct_from_swing_low == pytest.approx(1.010101, rel=1e-4)


# ---------------------------------------------------------------------------
# The volume baseline (audit: the bar was inside its own benchmark)
# ---------------------------------------------------------------------------


def test_volume_ratio_uses_the_twenty_bars_before_the_measured_bar():
    """Twenty bars at 100 followed by one bar at 500. The answer is 5.0.

    Under the old definition the measured bar sat inside its own 20-bar mean —
    (100*19 + 500)/20 = 120, giving 4.17x — which biases every reading toward
    1.0 and caps the ratio at 20x no matter how extreme the bar is.
    """
    volumes = [100.0] * 20 + [500.0]
    tech = signals.technicals(make_series([100.0] * 21, volume=volumes))
    assert tech.volume_ratio_prior_20 == pytest.approx(5.0)


def test_volume_ratio_can_exceed_twenty_times():
    """The structural consequence of a non-overlapping baseline: a 50x bar is
    allowed to report 50x. The old definition could not exceed 20x."""
    volumes = [100.0] * 20 + [5000.0]
    tech = signals.technicals(make_series([100.0] * 21, volume=volumes))
    assert tech.volume_ratio_prior_20 == pytest.approx(50.0)


def test_volume_ratio_ignores_the_in_progress_bar_entirely():
    """The second half of the audit's finding: the measured bar was usually
    *incomplete*, so early in a period a genuine spike read as normal volume.

    Here the open bar carries a huge partial volume and the last closed bar is
    ordinary. The ratio must describe the closed bar.
    """
    volumes = [100.0] * 20 + [200.0, 99_999.0]
    series = make_series([100.0] * 22, volume=volumes, last_open=True)
    tech = signals.technicals(series)
    assert tech.candles_used == 21
    assert tech.volume_ratio_prior_20 == pytest.approx(2.0)


def test_volume_ratio_is_none_with_exactly_twenty_bars():
    """Twenty bars is a baseline with nothing left to measure against it."""
    tech = signals.technicals(make_series([100.0] * 20))
    assert tech.volume_ratio_prior_20 is None


def test_volume_ratio_is_none_when_the_baseline_is_all_zero():
    volumes = [0.0] * 20 + [500.0]
    tech = signals.technicals(make_series([100.0] * 21, volume=volumes))
    assert tech.volume_ratio_prior_20 is None  # x/0 is undefined, not infinite volume


# ---------------------------------------------------------------------------
# Look-ahead: the load-bearing test in this file
# ---------------------------------------------------------------------------


def test_mutating_the_open_bar_moves_nothing():
    """No feature may be a function of the in-progress bar. Audit look-ahead.

    The open bar is rewritten into a 10x spike with 1000x the volume. If any
    single field of ``Technicals`` moves, some indicator is reading the partial
    current period as though it were history — which is the leak, and it is
    invisible in backtests because a backtest replays completed bars.
    """
    closes = [100.0 + math.sin(i / 3.0) * 5.0 for i in range(60)]
    series = make_series(closes, last_open=True)
    before = signals.technicals(series)

    spike = Candle(
        ts=series.candles[-1].ts,
        open=1000.0,
        high=5000.0,
        low=900.0,
        close=4800.0,
        volume=1_000_000.0,
        closed=False,
    )
    mutated = CandleSeries(
        timeframe=series.timeframe,
        pool_address=series.pool_address,
        candles=(*series.candles[:-1], spike),
        interval_seconds=series.interval_seconds,
        provenance=series.provenance,
    )
    after = signals.technicals(mutated)

    assert before == after
    # And the guard that makes the test meaningful: the open bar really was
    # different, and really was present on the series.
    assert series.candles[-1] != spike
    assert len(mutated.candles) == 60
    assert before.candles_used == 59


def test_closing_the_last_bar_does_change_the_answer():
    """The other direction. If nothing changed when a bar *closed*, the test
    above would be passing for the wrong reason — an indicator stuck on stale
    data rather than one correctly excluding the open bar."""
    closes = [100.0 + math.sin(i / 3.0) * 5.0 for i in range(60)]
    open_ended = signals.technicals(make_series(closes, last_open=True))
    all_closed = signals.technicals(make_series(closes, last_open=False))
    assert open_ended.candles_used == 59
    assert all_closed.candles_used == 60
    assert open_ended.rsi14 != all_closed.rsi14


def test_technicals_carries_the_pool_it_was_computed_from():
    """A ``Technicals`` that cannot name its pool cannot be compared with the
    one before it — a pair switch creates a synthetic price regime."""
    tech = signals.technicals(make_series([100.0] * 30, pool_address=POOL_B))
    assert tech.pool_address == POOL_B
    assert tech.timeframe is Timeframe.M5


# ---------------------------------------------------------------------------
# Degenerate input: never raise, always answer None
# ---------------------------------------------------------------------------


def test_missing_series_produces_an_all_none_brief_rather_than_an_exception():
    """GeckoTerminal being down must not take the tick down with it."""
    tech = signals.technicals(None, Timeframe.H1)
    assert tech.timeframe is Timeframe.H1
    assert tech.candles_used == 0
    assert tech.pool_address == ""
    assert tech.rsi14 is None
    assert tech.atr14_pct is None
    assert tech.volume_ratio_prior_20 is None


def test_a_series_of_only_open_bars_has_nothing_to_compute_from():
    series = make_series([100.0], last_open=True)
    tech = signals.technicals(series)
    assert tech.candles_used == 0
    assert tech.pool_address == POOL_A
    assert tech.rsi14 is None


def test_three_candles_is_a_brief_full_of_none_not_a_crash():
    """A coin that launched an hour ago has three candles. That is a normal
    state for this system, not an error."""
    tech = signals.technicals(make_series([1.0, 2.0, 3.0]))
    assert tech.candles_used == 3
    assert tech.rsi14 is None
    assert tech.bb_percent_b is None
    assert tech.realized_vol_pct is None


def test_no_indicator_ever_returns_nan():
    """A NaN reaching a prompt renders as the string "nan", which a model will
    reason about as though it were data."""
    for closes in ([100.0] * 40, [float(i) for i in range(1, 41)], [1.0, 2.0, 3.0]):
        tech = signals.technicals(make_series(closes))
        for name in tech.__slots__:
            value = getattr(tech, name)
            if isinstance(value, float):
                assert math.isfinite(value), f"{name} is not finite"


# ---------------------------------------------------------------------------
# Flow brief: counts, turnover, and pool-gated liquidity trend
# ---------------------------------------------------------------------------


def test_txn_count_ratios_are_counts_and_are_named_as_such():
    flow = signals.flow_brief(make_snapshot())
    assert flow.txn_count_ratio_m5 == pytest.approx(2.0)
    assert flow.txn_count_ratio_h1 == pytest.approx(2.0)
    assert flow.txn_count_ratio_h24 == pytest.approx(2.0)
    # The field names carry the correction. Anything called "flow" would be
    # claiming signed notional, which is not implemented anywhere in this system.
    assert not any("flow_ratio" in f for f in type(flow).__slots__)


def test_txn_count_ratio_is_none_when_a_side_was_not_observed():
    """Not 1.0. A neutral ratio computed from an absent block is a confident
    claim about a balanced two-sided market, manufactured out of silence."""
    snap = make_snapshot(
        txns_m5=TxnCounts(buys=None, sells=None),
        txns_h1=TxnCounts(buys=40, sells=None),
    )
    flow = signals.flow_brief(snap)
    assert flow.txn_count_ratio_m5 is None
    assert flow.txn_count_ratio_h1 is None
    assert flow.txn_count_ratio_h24 == pytest.approx(2.0)


def test_txn_count_ratio_is_infinite_when_nobody_is_selling():
    """Observed buys against observed *zero* sells is a real state, and it is
    different from an unobserved one. It is reported as-is and never divided by."""
    snap = make_snapshot(txns_m5=TxnCounts(buys=25, sells=0))
    assert signals.flow_brief(snap).txn_count_ratio_m5 == math.inf


def test_turnover_is_volume_over_depth():
    snap = make_snapshot(
        liquidity_usd=100_000.0, volume_24h_usd=500_000.0, volume_1h_usd=20_000.0
    )
    flow = signals.flow_brief(snap)
    assert flow.turnover_24h == pytest.approx(5.0)
    assert flow.turnover_1h == pytest.approx(0.2)


def test_turnover_is_none_when_depth_is_zero_or_unknown():
    """Replaces a test that asserted 0.0 for zero liquidity. ``x/0`` is
    undefined, and "nothing changing hands relative to depth" is the *opposite*
    of what a zero-depth pool with volume means."""
    assert signals.flow_brief(make_snapshot(liquidity_usd=0.0)).turnover_24h is None
    assert signals.flow_brief(make_snapshot(liquidity_usd=None)).turnover_24h is None
    assert signals.flow_brief(make_snapshot(volume_1h_usd=None)).turnover_1h is None


def test_liquidity_trend_is_none_on_the_first_tick():
    flow = signals.flow_brief(make_snapshot())
    assert flow.liquidity_trend_pct is None
    assert flow.liquidity_trend_seconds is None
    assert flow.liquidity_trend_pool is None


def test_liquidity_trend_is_measured_within_one_pool():
    previous = make_snapshot(liquidity_usd=100_000.0, receive_time=START_TS)
    current = make_snapshot(liquidity_usd=85_000.0, receive_time=START_TS + 900.0)
    flow = signals.flow_brief(current, previous)
    assert flow.liquidity_trend_pct == pytest.approx(-15.0)
    assert flow.liquidity_trend_pool == POOL_A
    # Derived from the two observations' provenance when the caller does not say.
    assert flow.liquidity_trend_seconds == pytest.approx(900.0)


def test_the_caller_may_state_the_interval_itself():
    """``loop.py`` used to hand this a 60-second-old snapshot while the prompt
    labelled the result "trend vs last tick", so a pool draining 10% across a
    15-minute decision interval reached the model as -0.7% — noise, against a
    prompt that ranks a draining pool above every other signal in the system."""
    previous = make_snapshot(liquidity_usd=100_000.0)
    current = make_snapshot(liquidity_usd=90_000.0)
    flow = signals.flow_brief(current, previous, elapsed_seconds=900.0)
    assert flow.liquidity_trend_pct == pytest.approx(-10.0)
    assert flow.liquidity_trend_seconds == pytest.approx(900.0)


def test_liquidity_trend_is_not_reported_across_a_pool_switch():
    """The audit's ``signals.py:382-428`` finding, which was live in the code:
    the comparison matched on symbol and ignored ``pair_address``.

    ``_best_pair`` can legitimately select a different pool between two reads.
    When it does, the delta across them is the difference between two unrelated
    pools — a fabricated 60% "drain" that no liquidity provider caused, arriving
    at a prompt that treats a draining pool as the strongest sell signal there is.
    """
    previous = make_snapshot(liquidity_usd=250_000.0, pool_ref=pool(POOL_A))
    current = make_snapshot(liquidity_usd=100_000.0, pool_ref=pool(POOL_B))
    flow = signals.flow_brief(current, previous)

    assert flow.liquidity_trend_pct is None
    assert flow.liquidity_trend_seconds is None
    assert flow.liquidity_trend_pool is None
    # The current liquidity itself is still reported: it is a fact about now.
    assert flow.liquidity_usd == 100_000.0
    # And the naive answer this refuses to give would have been -60%.
    assert pytest.approx(-60.0) == (100_000.0 - 250_000.0) / 250_000.0 * 100.0


def test_liquidity_trend_is_none_when_either_reading_is_missing():
    previous = make_snapshot(liquidity_usd=None)
    current = make_snapshot(liquidity_usd=100_000.0)
    assert signals.flow_brief(current, previous).liquidity_trend_pct is None
    assert (
        signals.flow_brief(
            make_snapshot(liquidity_usd=None), previous=None
        ).liquidity_trend_pct
        is None
    )


def test_liquidity_trend_is_none_against_a_zero_previous_depth():
    previous = make_snapshot(liquidity_usd=0.0)
    current = make_snapshot(liquidity_usd=100_000.0)
    assert signals.flow_brief(current, previous).liquidity_trend_pct is None


def test_price_ladder_passes_through_with_its_gaps_intact():
    snap = make_snapshot(price_change=PriceLadder(m5=None, h1=0.93, h6=None, h24=-4.2))
    ladder = signals.flow_brief(snap).price_ladder
    assert ladder.m5 is None  # not 0.0: DexScreener omitted the window
    assert ladder.h1 == pytest.approx(0.93)
    assert ladder.h6 is None
    assert ladder.h24 == pytest.approx(-4.2)


# ---------------------------------------------------------------------------
# brief()
# ---------------------------------------------------------------------------


def test_brief_assembles_both_timeframes_and_the_flow():
    snap = make_snapshot(
        candles_5m=make_series([100.0 + i for i in range(40)], timeframe=Timeframe.M5),
        candles_1h=make_series(
            [100.0 + i for i in range(40)],
            timeframe=Timeframe.H1,
            interval_seconds=3600.0,
        ),
    )
    out = signals.brief(snap)
    assert out.symbol == "BONK"
    assert out.m5.timeframe is Timeframe.M5
    assert out.h1.timeframe is Timeframe.H1
    assert out.m5.rsi14 is not None
    assert out.flow.liquidity_usd == 100_000.0


def test_brief_still_answers_when_candles_are_missing_entirely():
    """Both timeframes are always present even with no series, because
    ``prompts.py`` renders the ``None`` fields as explicitly unavailable — which
    is information the consumer needs rather than a section to hide."""
    out = signals.brief(make_snapshot())
    assert out.m5.candles_used == 0
    assert out.h1.candles_used == 0
    assert out.m5.rsi14 is None
    assert out.flow.turnover_24h == pytest.approx(5.0)


def test_there_is_no_agreement_or_confluence_score():
    """Audit §7/§9, asserted structurally so it cannot be reintroduced quietly.

    RSI, the EMAs, MACD and %B are deterministic transforms of *one* close
    series. When they "agree", one price path has been described four times.
    Any "N of M signals bullish" number is a correlation artefact wearing a
    confidence interval, and none may exist in this module.
    """
    banned = ("score", "confluence", "agree", "bullish_count", "signal_count")
    for name in signals.Technicals.__slots__:
        assert not any(word in name.lower() for word in banned), name
    for name in dir(signals):
        if name.startswith("_"):
            continue
        assert not any(word in name.lower() for word in banned), name
