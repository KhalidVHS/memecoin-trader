"""Technical indicators and on-chain flow, derived from a ``CoinSnapshot``.

Two rules govern everything in this module.

**Deliberately few indicators.** The predecessor to this project ran eight
signals and finished -16.62%; cutting to two finished +5.39%. Every additional
oscillator is another chance for the model to find a story it likes in noise,
and the measured effect of adding them has been negative. The list below is
therefore closed: RSI, the two EMAs, MACD, Bollinger, ATR, relative volume and
distance-to-swing. Resist adding more.

**Missing is never zero.** Every field on ``Technicals`` is ``X | None``, and a
``None`` here means "there is not enough history to make this claim" — never
"the value happens to be zero". Handing a model ``rsi14=0.0`` because only
twelve candles arrived is how you get a confidently wrong trade, so every
indicator below checks its own warm-up length and every division checks its own
denominator.

Implementation note: pandas and numpy only. There is no TA-Lib dependency and
there must never be one — it needs a C toolchain on Windows, and this project
has to stay one ``uv sync`` away from running.

All percentage outputs are whole percents (``-4.2`` means -4.2%), per the
convention at the top of ``types.py``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from .types import (
    Candle,
    CoinSnapshot,
    FlowBrief,
    TechnicalBrief,
    Technicals,
    Timeframe,
)

# ---------------------------------------------------------------------------
# Periods. Named rather than inlined so the warm-up lengths below read as
# statements about the indicator instead of as magic numbers.
# ---------------------------------------------------------------------------

RSI_PERIOD = 14
EMA_FAST_SPAN = 9
EMA_SLOW_SPAN = 21
MACD_FAST_SPAN = 12
MACD_SLOW_SPAN = 26
MACD_SIGNAL_SPAN = 9
BB_PERIOD = 20
BB_SIGMA = 2.0
ATR_PERIOD = 14
VOLUME_MEAN_PERIOD = 20
SWING_LOOKBACK = 20  # bars of high/low history used for distance-to-swing


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _finite(value: float | None) -> float | None:
    """Collapse NaN and +/-inf to ``None``.

    Every value leaving this module passes through here. A NaN that escapes into
    a prompt renders as the string "nan", which a model will happily reason
    about as though it were data.
    """
    if value is None:
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _last(values: np.ndarray, back: int = 0) -> float | None:
    """The value ``back`` bars from the end, or ``None`` if it is not there."""
    if values.size <= back:
        return None
    return _finite(values[values.size - 1 - back])


def _ratio_pct(value: float | None, base: float | None) -> float | None:
    """``(value - base) / base`` as a whole percent, guarding a zero base."""
    if value is None or base is None or base == 0.0:
        return None
    return _finite((value - base) / base * 100.0)


def _rising(values: np.ndarray) -> bool | None:
    """Strictly greater than the previous bar. ``None`` if there is no previous.

    Flat counts as not rising: a flat RSI is not momentum, and reporting it as
    such would be the same lie as reporting a missing value as zero.
    """
    latest, prior = _last(values, 0), _last(values, 1)
    if latest is None or prior is None:
        return None
    return latest > prior


def _sma_seeded_ema(values: np.ndarray, span: int) -> np.ndarray:
    """EMA seeded with the simple mean of the first ``span`` values.

    pandas' ``ewm(adjust=False)`` seeds with the *first observation*, which lets
    one arbitrary early bar dominate the series for dozens of bars afterwards.
    Every TA reference definition of EMA/MACD seeds with the SMA instead, and on
    a 100-bar memecoin window the difference is large enough to flip a MACD
    cross. Output is NaN until the seed window is full, which is what makes the
    warm-up checks downstream honest.
    """
    count = values.size
    out = np.full(count, np.nan, dtype=float)
    if count < span or span <= 0:
        return out
    alpha = 2.0 / (span + 1.0)
    level = float(values[:span].mean())
    out[span - 1] = level
    for i in range(span, count):
        level += alpha * (float(values[i]) - level)
        out[i] = level
    return out


def _wilder_smoothed(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing: alpha = 1/period, seeded with the first mean.

    This is *not* ``ewm(span=period)`` (which is alpha = 2/(period+1)) and *not*
    a rolling simple mean. RSI and ATR are both defined on Wilder's smoothing;
    substituting either alternative gives numbers that look plausible and
    disagree with every charting platform the user will sanity-check against.
    """
    count = values.size
    out = np.full(count, np.nan, dtype=float)
    if count < period or period <= 0:
        return out
    level = float(values[:period].mean())
    out[period - 1] = level
    for i in range(period, count):
        level += (float(values[i]) - level) / period
        out[i] = level
    return out


# ---------------------------------------------------------------------------
# Individual indicators. Each returns a full-length array aligned to the candle
# index, NaN-padded through its own warm-up.
# ---------------------------------------------------------------------------


def _rsi(close: np.ndarray, period: int = RSI_PERIOD) -> np.ndarray:
    """Wilder's RSI. Needs ``period + 1`` candles: N changes require N+1 closes."""
    count = close.size
    out = np.full(count, np.nan, dtype=float)
    if count < period + 1:
        return out

    delta = np.diff(close)
    avg_gain = _wilder_smoothed(np.where(delta > 0.0, delta, 0.0), period)
    avg_loss = _wilder_smoothed(np.where(delta < 0.0, -delta, 0.0), period)

    for i in range(delta.size):
        gain, loss = avg_gain[i], avg_loss[i]
        if math.isnan(gain) or math.isnan(loss):
            continue
        if loss == 0.0:
            # No downside at all in the window. A flat market is neither
            # overbought nor oversold, so it is 50 rather than 100 — 0/0 and
            # x/0 are different questions.
            out[i + 1] = 50.0 if gain == 0.0 else 100.0
        else:
            out[i + 1] = 100.0 - 100.0 / (1.0 + gain / loss)
    return out


def _macd(close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD line, signal and histogram (12/26/9), aligned to the candle index."""
    count = close.size
    fast = _sma_seeded_ema(close, MACD_FAST_SPAN)
    slow = _sma_seeded_ema(close, MACD_SLOW_SPAN)
    line = fast - slow  # NaN wherever either EMA is still warming up

    signal = np.full(count, np.nan, dtype=float)
    valid = ~np.isnan(line)
    if valid.any():
        # The signal line is an EMA *of the MACD line*, so its own seed window
        # starts where the MACD line starts, not where the price series does.
        first = int(np.argmax(valid))
        signal[first:] = _sma_seeded_ema(line[first:], MACD_SIGNAL_SPAN)

    return line, signal, line - signal


def _macd_cross(
    hist: np.ndarray,
) -> tuple[Literal["bullish", "bearish", "none"] | None, int | None]:
    """The most recent histogram sign flip and how many bars ago it happened.

    Freshness is the point. A bullish cross forty bars old is a trend that is
    already priced in; the same cross one bar ago is the trade. ``bars_since``
    is 0 when the flip is on the latest bar.
    """
    if hist.size == 0 or math.isnan(hist[-1]):
        return None, None

    last_index = hist.size - 1
    for i in range(last_index, 0, -1):
        current, previous = hist[i], hist[i - 1]
        if math.isnan(current) or math.isnan(previous):
            break  # walked off the front of the valid window
        if current > 0.0 >= previous:
            return "bullish", last_index - i
        if current < 0.0 <= previous:
            return "bearish", last_index - i
    # No flip inside the history we have. That is a real answer ("none"), but
    # the age of a cross that never happened is not, hence the None.
    return "none", None


def _bollinger(close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """%B and bandwidth for a ``BB_PERIOD``/``BB_SIGMA`` band.

    Population sigma (ddof=0), which is Bollinger's own definition. %B is left
    unclamped on purpose: values above 1 or below 0 mean price has broken out of
    the band, and that excursion is the signal — clamping deletes it.
    """
    series = pd.Series(close, dtype=float)
    mid = series.rolling(BB_PERIOD).mean()
    sigma = series.rolling(BB_PERIOD).std(ddof=0)

    width = 2.0 * BB_SIGMA * sigma  # upper - lower
    lower = mid - BB_SIGMA * sigma

    # Zero width means a perfectly flat window: %B is 0/0, which has no answer.
    # Zero mid means a zero-priced series. Both become NaN and then None.
    percent_b = np.where(width.to_numpy() > 0.0, (series - lower) / width, np.nan)
    bandwidth = np.where(mid.to_numpy() != 0.0, width / mid * 100.0, np.nan)
    return np.asarray(percent_b, dtype=float), np.asarray(bandwidth, dtype=float)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Wilder's ATR on true range, aligned to the candle index.

    True range is the widest of the bar's own range and its two gaps from the
    previous close. Memecoins gap constantly, so the high-vs-previous-close
    branch is the one that usually wins; an ATR built from high-low alone
    understates volatility badly and makes a -15% stop look much safer than it
    is.
    """
    count = close.size
    out = np.full(count, np.nan, dtype=float)
    if count < 2:
        return out

    previous_close = close[:-1]
    true_range = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - previous_close),
            np.abs(low[1:] - previous_close),
        ),
    )
    out[1:] = _wilder_smoothed(true_range, ATR_PERIOD)
    return out


def _empty(timeframe: Timeframe, candles_used: int) -> Technicals:
    """Every field unavailable. Used for empty and too-short histories."""
    return Technicals(
        timeframe=timeframe,
        candles_used=candles_used,
        rsi14=None,
        rsi14_rising=None,
        ema9=None,
        ema21=None,
        ema9_above_ema21=None,
        pct_from_ema9=None,
        pct_from_ema21=None,
        macd_line=None,
        macd_signal=None,
        macd_hist=None,
        macd_cross=None,
        bars_since_cross=None,
        bb_percent_b=None,
        bb_bandwidth=None,
        bb_expanding=None,
        atr14_pct=None,
        volume_ratio_20=None,
        pct_from_swing_high=None,
        pct_from_swing_low=None,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def technicals(candles: Sequence[Candle], timeframe: Timeframe) -> Technicals:
    """Compute the full indicator set for one timeframe.

    Never raises on short, empty or degenerate input — a coin that just launched
    has three candles, and that must produce a brief full of ``None`` rather than
    an exception that takes the whole tick down.
    """
    count = len(candles)
    if count == 0:
        return _empty(timeframe, 0)

    close = np.fromiter((c.close for c in candles), dtype=float, count=count)
    high = np.fromiter((c.high for c in candles), dtype=float, count=count)
    low = np.fromiter((c.low for c in candles), dtype=float, count=count)
    volume = np.fromiter((c.volume for c in candles), dtype=float, count=count)
    latest_close = _finite(close[-1])

    rsi = _rsi(close)
    ema_fast = _sma_seeded_ema(close, EMA_FAST_SPAN)
    ema_slow = _sma_seeded_ema(close, EMA_SLOW_SPAN)
    macd_line, macd_signal, macd_hist = _macd(close)
    percent_b, bandwidth = _bollinger(close)
    atr = _atr(high, low, close)

    ema9 = _last(ema_fast)
    ema21 = _last(ema_slow)
    cross, bars_since_cross = _macd_cross(macd_hist)

    # ATR is only meaningful relative to price; at a zero close it is not a
    # percentage of anything.
    atr_value = _last(atr)
    atr14_pct = None
    if atr_value is not None and latest_close is not None and latest_close != 0.0:
        atr14_pct = _finite(atr_value / latest_close * 100.0)

    # Relative volume. A zero mean means nothing traded in the window, so the
    # ratio is undefined rather than zero.
    volume_ratio_20 = None
    if count >= VOLUME_MEAN_PERIOD:
        mean_volume = float(volume[-VOLUME_MEAN_PERIOD:].mean())
        if mean_volume > 0.0:
            volume_ratio_20 = _finite(float(volume[-1]) / mean_volume)

    # Distance to the swing extremes over the lookback window, inclusive of the
    # current bar. Negative from the high, positive from the low.
    pct_from_swing_high = None
    pct_from_swing_low = None
    if count >= SWING_LOOKBACK:
        swing_high = float(high[-SWING_LOOKBACK:].max())
        swing_low = float(low[-SWING_LOOKBACK:].min())
        pct_from_swing_high = _ratio_pct(latest_close, swing_high)
        pct_from_swing_low = _ratio_pct(latest_close, swing_low)

    return Technicals(
        timeframe=timeframe,
        candles_used=count,
        rsi14=_last(rsi),
        rsi14_rising=_rising(rsi),
        ema9=ema9,
        ema21=ema21,
        ema9_above_ema21=None if ema9 is None or ema21 is None else ema9 > ema21,
        pct_from_ema9=_ratio_pct(latest_close, ema9),
        pct_from_ema21=_ratio_pct(latest_close, ema21),
        macd_line=_last(macd_line),
        macd_signal=_last(macd_signal),
        macd_hist=_last(macd_hist),
        macd_cross=cross,
        bars_since_cross=bars_since_cross,
        bb_percent_b=_last(percent_b),
        bb_bandwidth=_last(bandwidth),
        bb_expanding=_rising(bandwidth),
        atr14_pct=atr14_pct,
        volume_ratio_20=volume_ratio_20,
        pct_from_swing_high=pct_from_swing_high,
        pct_from_swing_low=pct_from_swing_low,
    )


def flow_brief(
    snap: CoinSnapshot,
    previous: CoinSnapshot | None = None,
    *,
    elapsed_seconds: float | None = None,
) -> FlowBrief:
    """On-chain flow, mostly passed straight through from DexScreener.

    These numbers have no traditional-TA analogue and for this asset class they
    are arguably the highest-signal evidence available, because they are actual
    money moving rather than talk about money moving.

    ``previous`` is whichever earlier read the caller wants liquidity measured
    against, and ``elapsed_seconds`` is how far back that read is. The caller
    owns both because only the caller knows its own cadence: ``loop.py`` used to
    hand this the snapshot from 60 seconds ago while the prompt labelled the
    result "trend vs last tick", so a pool draining 10% across a 15-minute
    decision interval reached the model as -0.7% — noise, against a system
    prompt that ranks a draining pool above every other signal in the system.
    """
    liquidity = snap.liquidity_usd

    # Turnover relative to the pool. A pool with no liquidity is untradeable, so
    # reporting 0.0 (rather than an infinity that would poison every downstream
    # comparison) is both safe and the correct read for risk purposes.
    if liquidity > 0.0:
        turnover_24h = snap.volume_24h_usd / liquidity
        turnover_1h = snap.volume_1h_usd / liquidity
    else:
        turnover_24h = 0.0
        turnover_1h = 0.0

    # A draining pool is the single most dangerous thing that can happen to a
    # memecoin position, and price alone will not show it until the exit is
    # already gone. None on the first tick — there is no trend from one point.
    liquidity_trend_pct = None
    liquidity_trend_seconds = None
    if previous is not None and previous.liquidity_usd > 0.0:
        liquidity_trend_pct = _finite(
            (liquidity - previous.liquidity_usd) / previous.liquidity_usd * 100.0
        )
        # The window is attached to the percentage rather than reported on its
        # own, so a caller can never render an interval next to a trend that was
        # not measured — "stable over 15m" is a claim, and an absent trend is
        # not.
        if liquidity_trend_pct is not None:
            liquidity_trend_seconds = _finite(elapsed_seconds)

    # TxnCounts.ratio is intentionally allowed to be inf (buys with zero sells);
    # it is reported as-is and never used as a divisor here.
    return FlowBrief(
        buy_sell_ratio_m5=snap.txns_m5.ratio,
        buy_sell_ratio_h1=snap.txns_h1.ratio,
        buy_sell_ratio_h24=snap.txns_h24.ratio,
        turnover_24h=turnover_24h,
        turnover_1h=turnover_1h,
        liquidity_usd=liquidity,
        liquidity_trend_pct=liquidity_trend_pct,
        liquidity_trend_seconds=liquidity_trend_seconds,
        price_ladder=snap.price_change,
    )


def brief(
    snap: CoinSnapshot,
    previous: CoinSnapshot | None = None,
    *,
    elapsed_seconds: float | None = None,
) -> TechnicalBrief:
    """The full technical picture for one coin: 5m, 1h and flow.

    Both timeframes are always present, even when their candle lists are empty —
    ``prompts.py`` renders the ``None`` fields as explicitly unavailable, which
    is information the model needs rather than a section to hide.
    """
    return TechnicalBrief(
        symbol=snap.symbol,
        m5=technicals(snap.candles_5m, Timeframe.M5),
        h1=technicals(snap.candles_1h, Timeframe.H1),
        flow=flow_brief(snap, previous, elapsed_seconds=elapsed_seconds),
    )


__all__ = ["brief", "flow_brief", "technicals"]
