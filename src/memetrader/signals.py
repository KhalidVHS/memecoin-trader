"""Technical indicators and pool state, derived from a ``CoinSnapshot``.

Four rules govern everything in this module. The first two predate the
adversarial audit; the last two are its findings.

**Deliberately few indicators.** The predecessor to this project ran eight
signals and finished -16.62%; cutting to two finished +5.39%. Every additional
oscillator is another chance for the model to find a story it likes in noise,
and the measured effect of adding them has been negative. The list below is
therefore closed: RSI, the two EMAs, MACD, Bollinger, ATR, relative volume,
realized volatility and distance-to-swing. Resist adding more.

**Missing is never zero.** Every field on ``Technicals`` is ``X | None``, and a
``None`` here means "there is not enough history to make this claim" — never
"the value happens to be zero". Handing a model ``rsi14=0.0`` because only
twelve candles arrived is how you get a confidently wrong trade, so every
indicator below checks its own warm-up length and every division checks its own
denominator.

**These indicators are not independent evidence, and their agreement is not
confirmation.** This is audit §7 and §9, and it is the most important sentence
in the module. RSI, EMA(9/21) and their distances, MACD line/signal/histogram,
%B and bandwidth are all deterministic transforms of *one* close series. When
they "agree", what has happened is that one price path was described five
times. Presenting that as five confirming signals creates false confidence and
multiplies degrees of freedom; the audit classifies MACD in particular as
"REMOVE from production until ablation" for exactly this reason. They are kept
here as a **labelled benchmark feature set** with no demonstrated alpha, so a
future ablation has something to ablate against. Consequently:

    There is no agreement score, confluence count, or "N of M signals bullish"
    number in this module, and none may be added. Any such score is a
    correlation artefact wearing a confidence interval.

**Only closed bars may feed a feature.** The audit's look-ahead finding: every
indicator and the volume ratio were computed over the *partial current bar*,
which leaks the in-progress period into a window labelled "historical" and
understates that bar's volume. Everything below reads
``CandleSeries.closed_candles``. The in-progress bar is retained on the series
because live price and running volume are legitimate state, but nothing in here
can see it. A direct consequence: mutating the open bar must not move a single
indicator value, and there is a test that asserts exactly that.

Implementation note: pandas and numpy only. There is no TA-Lib dependency and
there must never be one — it needs a C toolchain on Windows, and this project
has to stay one ``uv sync`` away from running.

All percentage outputs are whole percents (``-4.2`` means -4.2%), per the
convention at the top of ``types.py``.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

from .types import (
    CandleSeries,
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
#: Bars in the volume baseline. The baseline is the ``VOLUME_MEAN_PERIOD`` bars
#: *preceding* the most recent closed bar, so the comparison needs one more bar
#: than the period and the measured bar is never part of its own benchmark.
VOLUME_MEAN_PERIOD = 20
REALIZED_VOL_PERIOD = 20  # log returns, so it needs one extra close
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
#
# All of them are transforms of the same close series. See the module docstring:
# they do not corroborate each other.
# ---------------------------------------------------------------------------


def _rsi(close: np.ndarray, period: int = RSI_PERIOD) -> np.ndarray:
    """Wilder's RSI. Needs ``period + 1`` candles: N changes require N+1 closes.

    Audit §7 classifies this as "TEST as benchmark": it is a monotonic
    transform of past returns whose scale and horizon depend entirely on bar
    construction, with no current evidence of incremental net alpha.
    """
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
    """MACD line, signal and histogram (12/26/9), aligned to the candle index.

    Audit §7 classifies MACD as "REMOVE from production until ablation":
    it "largely repackages the same EMAs, multiplying degrees of freedom". It
    stays in the benchmark set and out of any confirmation logic.
    """
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
    is 0 when the flip is on the latest *closed* bar.
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

    Audit §7: bandwidth is plausible as a volatility/regime descriptor; %B
    overlaps normalized price momentum and is not separate information.
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

    This is the one indicator the audit keeps unreservedly — as a *risk* input
    for volatility scaling and data validation, not as a return forecast.
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


def _realized_vol_pct(close: np.ndarray) -> float | None:
    """Per-bar realized volatility: sample sigma of log returns, whole percent.

    Deliberately *not* annualized. An annualization factor is a claim about how
    many of these bars fit in a year, and for a 5-minute memecoin bar with gaps
    that claim is wrong in a way nobody would notice. Risk sizing wants the
    per-bar number and can scale it itself with a horizon it actually knows.

    Log returns rather than simple returns because they are additive across
    bars, which is the property that makes any later scaling defensible.
    """
    if close.size < REALIZED_VOL_PERIOD + 1:
        return None
    window = close[-(REALIZED_VOL_PERIOD + 1) :]
    if np.any(window <= 0.0):
        return None
    returns = np.diff(np.log(window))
    return _finite(float(returns.std(ddof=1)) * 100.0)


def _empty(timeframe: Timeframe, candles_used: int, pool_address: str) -> Technicals:
    """Every field unavailable. Used for empty and too-short histories."""
    return Technicals(
        timeframe=timeframe,
        candles_used=candles_used,
        pool_address=pool_address,
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
        volume_ratio_prior_20=None,
        realized_vol_pct=None,
        pct_from_swing_high=None,
        pct_from_swing_low=None,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def technicals(
    series: CandleSeries | None,
    timeframe: Timeframe | None = None,
) -> Technicals:
    """Compute the full benchmark indicator set for one timeframe.

    **Reads ``series.closed_candles`` and nothing else.** That is the audit's
    look-ahead fix, and it is enforced here rather than at each call site
    because "remember to drop the last bar" is not a control. ``candles_used``
    therefore counts closed bars, which is smaller than ``len(series.candles)``
    on almost every live read and is the honest number.

    Takes a ``CandleSeries`` rather than a bare sequence so the result can carry
    ``pool_address``. Audit §7 is explicit that a pool switch creates a
    synthetic price regime; a ``Technicals`` that cannot name the pool it was
    computed from cannot be compared with the one before it.

    Never raises on short, empty, missing or degenerate input — a coin that just
    launched has three candles, and that must produce a brief full of ``None``
    rather than an exception that takes the whole tick down. ``series=None``
    (GeckoTerminal was down, or the caller asked for a price-only snapshot) is
    the same answer for the same reason.

    A reminder that belongs next to the code and not only in the module
    docstring: these outputs are correlated transforms of one close series. Do
    not count them.
    """
    if series is None:
        return _empty(timeframe or Timeframe.M5, 0, "")

    timeframe = timeframe or series.timeframe
    closed = series.closed_candles
    count = len(closed)
    if count == 0:
        return _empty(timeframe, 0, series.pool_address)

    close = np.fromiter((c.close for c in closed), dtype=float, count=count)
    high = np.fromiter((c.high for c in closed), dtype=float, count=count)
    low = np.fromiter((c.low for c in closed), dtype=float, count=count)
    volume = np.fromiter((c.volume for c in closed), dtype=float, count=count)
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

    # Relative volume, renamed from ``volume_ratio_20``. The audit's finding was
    # two separate errors stacked: the measured bar was inside its own 20-bar
    # baseline, which biases the ratio toward 1.0 and caps it at 20x however
    # extreme the bar is; and that bar was usually *incomplete*, so early in a
    # period a genuine volume spike read as normal volume. Here the numerator is
    # the last CLOSED bar and the denominator is the 20 bars strictly preceding
    # it — non-overlapping, so a 50x bar can report 50x.
    #
    # Not done, and worth naming: the audit also asks for seasonal matching
    # (compare a 03:00 bar against prior 03:00 bars). That needs a longer
    # history than a 100-bar window holds.
    volume_ratio_prior_20 = None
    if count >= VOLUME_MEAN_PERIOD + 1:
        baseline = volume[-(VOLUME_MEAN_PERIOD + 1) : -1]
        mean_volume = float(baseline.mean())
        if mean_volume > 0.0:
            volume_ratio_prior_20 = _finite(float(volume[-1]) / mean_volume)

    # Distance to the swing extremes over the lookback window, inclusive of the
    # latest closed bar. Negative from the high, positive from the low.
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
        pool_address=series.pool_address,
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
        volume_ratio_prior_20=volume_ratio_prior_20,
        realized_vol_pct=_realized_vol_pct(close),
        pct_from_swing_high=pct_from_swing_high,
        pct_from_swing_low=pct_from_swing_low,
    )


def _turnover(volume: float | None, liquidity: float | None) -> float | None:
    """Volume as a multiple of pool depth, or ``None`` if it is not defined.

    The old code returned ``0.0`` when liquidity was zero, reasoning that an
    untradeable pool is safe to report as having no turnover. That was wrong in
    the same way everything else this remediation removes was wrong: ``x/0`` is
    undefined, not zero, and "nothing is changing hands relative to depth" is
    the *opposite* of what a zero-depth pool with volume means. A missing
    numerator or denominator is likewise ``None``, never a neutral number.
    """
    if volume is None or liquidity is None or liquidity <= 0.0:
        return None
    return _finite(volume / liquidity)


def flow_brief(
    snap: CoinSnapshot,
    previous: CoinSnapshot | None = None,
    *,
    elapsed_seconds: float | None = None,
) -> FlowBrief:
    """Pool state and transaction **counts**.

    Read the name of every field before using one. These are
    ``txn_count_ratio_*``, not flow. Audit §7 is blunt about why the old name
    was a lie: "Counts are not notional, aggressor volume, or unique wallets.
    The prompt's 'actual money moving' description is false." One wallet can
    emit a thousand transactions for the price of a thousand signatures, which
    makes a buy/sell count ratio one of the cheapest numbers in this system to
    manipulate. It is retained as a weak, explicitly manipulable feature.

    **Signed notional flow from decoded swaps is the feature that would deserve
    the name "flow", and it is not implemented.** It requires decoding swap
    instructions, deduplicating routed legs across hops, and flagging self- and
    wash-trading patterns. Nothing here approximates it, and nothing downstream
    may describe these counts as if it did.

    ``previous`` is whichever earlier read the caller wants liquidity measured
    against, and ``elapsed_seconds`` is how far back that read is. The caller
    owns both because only the caller knows its own cadence: ``loop.py`` used to
    hand this the snapshot from 60 seconds ago while the prompt labelled the
    result "trend vs last tick", so a pool draining 10% across a 15-minute
    decision interval reached the model as -0.7% — noise, against a system
    prompt that ranks a draining pool above every other signal in the system.
    When the caller does not supply the gap, it is derived from the two
    observations' provenance, which is now recorded per observation precisely so
    that this is possible.

    **A liquidity trend is only reported within one pool.** This is the audit's
    ``signals.py:382-428`` finding, which was live in the code: the comparison
    matched on symbol and ignored ``pair_address``. ``_best_pair`` can
    legitimately select a different pool between two reads, and when it does,
    the delta across them is the difference between two unrelated pools — a
    fabricated 60% "drain" or "inflow" that no liquidity provider caused.
    ``liquidity_trend_pool`` records which pool the surviving comparison was
    made within, so a consumer can never render a trend without knowing where
    it came from.
    """
    liquidity = snap.liquidity_usd

    liquidity_trend_pct: float | None = None
    liquidity_trend_seconds: float | None = None
    liquidity_trend_pool: str | None = None

    # A draining pool is the single most dangerous thing that can happen to a
    # memecoin position, and price alone will not show it until the exit is
    # already gone. None on the first tick — there is no trend from one point.
    if (
        previous is not None
        and liquidity is not None
        and previous.liquidity_usd is not None
        and previous.liquidity_usd > 0.0
        and snap.pool.same_pool(previous.pool)
    ):
        liquidity_trend_pct = _finite(
            (liquidity - previous.liquidity_usd) / previous.liquidity_usd * 100.0
        )
        if liquidity_trend_pct is not None:
            liquidity_trend_pool = snap.pool.pair_address
            # The window is attached to the percentage rather than reported on
            # its own, so a caller can never render an interval next to a trend
            # that was not measured — "stable over 15m" is a claim, and an
            # absent trend is not.
            if elapsed_seconds is None:
                gap = snap.provenance.effective_time - previous.provenance.effective_time
                liquidity_trend_seconds = _finite(gap) if gap > 0 else None
            else:
                liquidity_trend_seconds = _finite(elapsed_seconds)

    # TxnCounts.ratio is intentionally allowed to be inf (observed buys against
    # observed zero sells — a pool nobody is selling, which is a real state) and
    # None (either side unobserved). Both are reported as-is; neither is used as
    # a divisor here.
    return FlowBrief(
        txn_count_ratio_m5=snap.txns_m5.ratio,
        txn_count_ratio_h1=snap.txns_h1.ratio,
        txn_count_ratio_h24=snap.txns_h24.ratio,
        turnover_24h=_turnover(snap.volume_24h_usd, liquidity),
        turnover_1h=_turnover(snap.volume_1h_usd, liquidity),
        liquidity_usd=liquidity,
        liquidity_trend_pct=liquidity_trend_pct,
        liquidity_trend_seconds=liquidity_trend_seconds,
        liquidity_trend_pool=liquidity_trend_pool,
        price_ladder=snap.price_change,
    )


def brief(
    snap: CoinSnapshot,
    previous: CoinSnapshot | None = None,
    *,
    elapsed_seconds: float | None = None,
) -> TechnicalBrief:
    """The full technical picture for one coin: 5m, 1h and pool state.

    Both timeframes are always present, even when their series is missing —
    ``prompts.py`` renders the ``None`` fields as explicitly unavailable, which
    is information the consumer needs rather than a section to hide.
    """
    return TechnicalBrief(
        symbol=snap.symbol,
        m5=technicals(snap.candles_5m, Timeframe.M5),
        h1=technicals(snap.candles_1h, Timeframe.H1),
        flow=flow_brief(snap, previous, elapsed_seconds=elapsed_seconds),
    )


__all__ = ["brief", "flow_brief", "technicals"]
