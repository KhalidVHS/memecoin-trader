"""Shared shapes for every module in the system.

These are our *internal normalized* types, deliberately not the raw shapes of
any API. Adapters (``market.py``, ``quotes.py``, ``sentiment.py``) translate
vendor JSON into these at their boundary, which is what keeps a surprise in a
vendor response from rippling through the rest of the codebase.

Conventions that everything depends on:

* **All timestamps are epoch seconds as ``float``.** DexScreener returns
  milliseconds and GeckoTerminal returns seconds; adapters normalize at the
  boundary. Nothing downstream should ever divide by 1000.
* **All percentages are whole numbers, not fractions.** ``-4.2`` means -4.2%.
  Jupiter returns ``priceImpactPct`` as a string fraction in 0..1; ``quotes.py``
  multiplies by 100 at the boundary. The one exception is config values whose
  name ends in ``_pct`` and are documented as fractions (``max_position_pct``,
  ``stop_loss_pct``) — those are fractions because they are multipliers.
* **Token amounts are integers in atomic units.** A float quantity of a
  9-decimal token is a rounding error waiting to be called a fill. Dollars are
  a *derived presentation*; the atomic integer is the fact.
* **Every numeric field is validated finite at construction.** NaN compares
  false against every bound, so an unvalidated NaN walks through a risk check
  untouched. See :func:`finite`.

Three conventions were added in response to the adversarial audit:

* **Missing is never zero, and now it is not constructible either.** Fields that
  can be absent are ``X | None`` *and* the parsers refuse to coerce a malformed
  value into a plausible number (audit C-list: ``market._as_float``).
* **Every observation carries its provenance** — when the event happened at the
  source, when the source published it, and when we received it. A snapshot
  timestamped at local completion time hides that its first constituent read is
  minutes old.
* **Quotes are not fills, and non-executable valuations are a different type.**
  :class:`Quote` can reach the broker; :class:`ValuationEstimate` structurally
  cannot. That is audit C4 enforced by the type system rather than by a flag
  everyone remembers to check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Validation primitives
# ---------------------------------------------------------------------------


class ValidationError(ValueError):
    """A value that should never have been constructed was constructed.

    Deliberately not caught anywhere in the trading path. These represent
    upstream bugs or corrupt vendor data, and the correct response is to abort
    the affected observation, not to substitute a guess.
    """


def finite(value: float, name: str) -> float:
    """Return ``value`` if it is a finite real number, else raise.

    The audit's finding was specific: ``Action.size_usd`` had no finite
    constraint, and ``_normalize`` only caught values ``< 0``. NaN is not
    ``< 0`` — it is not anything — so a NaN size passed the repair, passed every
    risk comparison (all of which are false against NaN, including the
    rejections), and arrived at the broker.

    Infinity is rejected for the same reason: it is not a size, and arithmetic
    on it produces NaN one step later, at which point the origin is lost.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValidationError(f"{name} must be a real number, got {value!r}")
    v = float(value)
    if math.isnan(v):
        raise ValidationError(f"{name} is NaN — a missing value must be None, not NaN")
    if math.isinf(v):
        raise ValidationError(f"{name} is infinite: {v}")
    return v


def finite_or_none(value: float | None, name: str) -> float | None:
    """``finite`` that permits an explicit absence.

    This is the only sanctioned way to express "we do not know". A NaN is still
    rejected: NaN is what a broken computation produces, ``None`` is what an
    honest one produces when the input was missing, and collapsing the two loses
    exactly the distinction this system is built around.
    """
    return None if value is None else finite(value, name)


def positive(value: float, name: str) -> float:
    v = finite(value, name)
    if v <= 0:
        raise ValidationError(f"{name} must be > 0, got {v}")
    return v


def non_negative(value: float, name: str) -> float:
    v = finite(value, name)
    if v < 0:
        raise ValidationError(f"{name} must be >= 0, got {v}")
    return v


def atomic(value: int, name: str) -> int:
    """Validate an atomic token amount: a non-negative integer, never a float.

    Accepting a float here would reintroduce the whole class of bug this type
    exists to prevent, so a float is refused even when it is integral — the
    caller has a rounding decision to make and must make it explicitly.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} must be an int in atomic units, got {value!r}")
    if value < 0:
        raise ValidationError(f"{name} must be >= 0, got {value}")
    return value


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class Timeframe(StrEnum):
    M5 = "5m"
    H1 = "1h"


class ExecutionMode(StrEnum):
    """What a given run is *permitted* to do. Audit C12.

    ``--dry-run`` claimed to be non-mutating but the stop-loss path never
    consulted the flag, so an operator testing a decision could liquidate a
    position. A boolean checked at one call site cannot fix that; the capability
    has to travel with the broker itself, so that a component which is not
    allowed to mutate state does not *have* an object that can.

    * ``READ_ONLY``  — compute everything, mutate nothing, journal nothing.
    * ``PAPER``      — simulate fills against the local ledger.
    * ``LIVE``       — submit real transactions. No implementation exists, and
                       :func:`assert_live_supported` refuses to let one appear
                       by accident.
    """

    READ_ONLY = "read_only"
    PAPER = "paper"
    LIVE = "live"

    @property
    def may_mutate(self) -> bool:
        return self is not ExecutionMode.READ_ONLY


def assert_live_supported(mode: ExecutionMode) -> None:
    """Refuse ``LIVE`` until something actually implements it.

    The audit's verdict is the first line of its executive summary: *do not
    deploy this system with real money*. The enum needs a ``LIVE`` member so
    that every capability check is written against the real three-way
    distinction rather than against a boolean that would have to be widened
    later — but a member that exists is a member somebody can pass. This is the
    guard that makes the gap between "the type admits it" and "the code does it"
    explicit and loud, rather than a ``NotImplementedError`` discovered halfway
    through a submission.
    """
    if mode is ExecutionMode.LIVE:
        raise NotImplementedError(
            "ExecutionMode.LIVE is not implemented. This is a paper trader: there is "
            "no wallet, no signer and no settlement path, and the adversarial audit "
            "records unresolved accounting, execution-realism and risk blockers. "
            "Real-money deployment requires clearing those first."
        )


class OrderState(StrEnum):
    """The execution state machine. Audit C11.

    Linear and explicit, because the old code had exactly two states — "a Fill
    row exists" and "it does not" — and a crash between the ledger append and
    the state save produced a third that nothing could name. A terminal order is
    one of ``LANDED``, ``FAILED`` or ``EXPIRED``; ``RECONCILED`` means the
    settled truth has been compared against what we expected.
    """

    PROPOSED = "proposed"
    RISK_APPROVED = "risk_approved"
    QUOTE_BOUND = "quote_bound"
    SUBMITTED = "submitted"
    LANDED = "landed"
    FAILED = "failed"
    EXPIRED = "expired"
    RECONCILED = "reconciled"

    @property
    def is_terminal(self) -> bool:
        return self in (
            OrderState.LANDED,
            OrderState.FAILED,
            OrderState.EXPIRED,
            OrderState.RECONCILED,
        )


class DataQuality(StrEnum):
    """Whether an observation may be used, and for what.

    ``QUARANTINED`` data is retained — it is evidence about a source's health —
    but no feature may be computed from it and no order may depend on it.
    """

    OK = "ok"
    DEGRADED = "degraded"
    QUARANTINED = "quarantined"


# ---------------------------------------------------------------------------
# Provenance and time
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Provenance:
    """When an observation was true, published, and received.

    The audit's leakage finding: ``MarketSnapshot.ts`` was ``time.time()`` taken
    after all the sequential HTTP calls returned, so a snapshot looked fresh
    while its first constituent read could be minutes old. Freshness must be
    asserted per observation, not per batch.

    * ``event_time``   — when the thing happened on-chain/at the venue. Often
                         unavailable from a REST summary endpoint, hence
                         optional; ``None`` is honest, a copy of ``receive_time``
                         is a lie that makes stale data look current.
    * ``source_time``  — when the source says it computed the value.
    * ``receive_time`` — when this process finished reading it. Always known.
    * ``available_time`` — the earliest simulated clock at which a *replay* is
                         permitted to look at this value. See below.

    ``age_seconds`` deliberately measures from the *oldest* known time, so a
    source that omits ``event_time`` cannot look fresher than one that reports
    it honestly.

    ``available_time`` is the backtest's central invariant and it is a different
    question from every other field here. ``event_time`` asks "when was this
    true?"; ``available_time`` asks "when could this system first have acted on
    it?". The two diverge constantly and the gap is always the direction that
    flatters a backtest: an hourly candle is *true* from its open timestamp but
    is not knowable until it closes and the vendor publishes it, and a social
    post created at 10:00 but collected at 10:03 was not available at 10:00.
    Replaying against ``event_time`` silently grants the strategy three minutes
    of foresight, which on a 0.2% hurdle is the entire edge.

    It defaults to ``None`` rather than to ``receive_time`` so that the gap is
    visible: ``available_at`` falls back to ``receive_time``, which is the
    honest floor for a live read (you had it once you received it), while a
    backfilled row must set it explicitly to close-plus-publication-delay. A
    default of "available when the event happened" would have been the one
    choice that leaks.

    ``sequence`` is the source's own ordering token — a Solana slot, a block
    height, a vendor cursor. It breaks ties deterministically when two records
    share an ``available_time``, which is what makes a replay reproducible
    rather than dependent on dict iteration order.
    """

    source: str
    receive_time: float
    event_time: float | None = None
    source_time: float | None = None
    quality: DataQuality = DataQuality.OK
    quality_reason: str | None = None
    available_time: float | None = None
    sequence: int | None = None

    def __post_init__(self) -> None:
        finite(self.receive_time, "receive_time")
        finite_or_none(self.event_time, "event_time")
        finite_or_none(self.source_time, "source_time")
        finite_or_none(self.available_time, "available_time")

    @property
    def available_at(self) -> float:
        """The earliest simulated time this value may be read. Never earlier
        than ``receive_time`` unless a loader states otherwise explicitly."""
        return self.available_time if self.available_time is not None else self.receive_time

    @property
    def effective_time(self) -> float:
        """The oldest timestamp we have. What staleness must be measured from."""
        candidates = [t for t in (self.event_time, self.source_time) if t is not None]
        return min(candidates) if candidates else self.receive_time

    def age_seconds(self, now: float) -> float:
        return now - self.effective_time

    @property
    def usable(self) -> bool:
        return self.quality is not DataQuality.QUARANTINED


@dataclass(frozen=True, slots=True)
class Observed[T]:
    """A value together with where and when it came from.

    Used where a single field's freshness differs from its neighbours' — a
    liquidity reading refreshed every minute sitting beside an hourly candle
    series. Wrapping everything would be noise; wrapping the fields that drive
    risk decisions is the point.
    """

    value: T
    provenance: Provenance


# ---------------------------------------------------------------------------
# Market data — owned by market.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candle:
    """One OHLCV bar. ``ts`` is the bar's open time, epoch seconds.

    ``closed`` records whether the bar was complete when we read it. The audit's
    look-ahead finding: indicators and the volume ratio were computed over a
    partial current bar, which both leaks the in-progress period into a
    "historical" window and understates its volume. Only closed bars may feed a
    feature; the current bar is kept because it is legitimate *state*, but it is
    labelled so nothing can silently treat it as history.
    """

    ts: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool = True

    def __post_init__(self) -> None:
        finite(self.ts, "candle.ts")
        for name in ("open", "high", "low", "close"):
            positive(getattr(self, name), f"candle.{name}")
        non_negative(self.volume, "candle.volume")
        # OHLC invariants. A vendor that reports low > high has malfunctioned,
        # and every range-based indicator downstream (ATR above all) silently
        # produces a negative range from it instead of failing.
        if self.low > self.high:
            raise ValidationError(f"candle low {self.low} > high {self.high}")
        if not (self.low <= self.open <= self.high):
            raise ValidationError(
                f"candle open {self.open} outside [{self.low}, {self.high}]"
            )
        if not (self.low <= self.close <= self.high):
            raise ValidationError(
                f"candle close {self.close} outside [{self.low}, {self.high}]"
            )


@dataclass(frozen=True, slots=True)
class CandleSeries:
    """A validated, gap-checked run of candles for one pool and timeframe.

    Bound to ``pool_address`` because the audit is right that a pool switch
    creates a synthetic price and liquidity regime: concatenating candles from
    two pools produces a jump that every momentum and volatility feature reads
    as a real move. A series that cannot name its pool cannot be trusted to be
    continuous, so the pool is part of the identity rather than a note.
    """

    timeframe: Timeframe
    pool_address: str
    candles: tuple[Candle, ...]
    interval_seconds: float
    provenance: Provenance
    missing_intervals: int = 0

    @property
    def closed_candles(self) -> tuple[Candle, ...]:
        """The only thing a feature may be computed from."""
        return tuple(c for c in self.candles if c.closed)

    @property
    def complete(self) -> bool:
        return self.missing_intervals == 0


@dataclass(frozen=True, slots=True)
class TxnCounts:
    """Buy/sell transaction *counts* over a window.

    Renamed in spirit by the audit: these are counts of transactions, not
    notional flow and not unique actors. One wallet can emit a thousand of them.
    The prompt used to describe this as "actual money moving", which was false;
    ``signals.py`` now labels it as a count and the system treats it as a weak,
    manipulable feature rather than as flow.

    ``buys``/``sells`` are ``int | None`` because DexScreener omits the block
    entirely for quiet pairs. The old code coerced the absence to 0, which made
    ``ratio`` return a confident, neutral 1.0 computed from nothing.
    """

    buys: int | None
    sells: int | None

    @property
    def ratio(self) -> float | None:
        """Buys per sell. ``None`` when either count is unobserved.

        Still returns ``inf`` for observed buys against observed zero sells,
        which is a real and meaningful state: a pool nobody is selling.
        """
        if self.buys is None or self.sells is None:
            return None
        if self.sells == 0:
            return float("inf") if self.buys else 1.0
        return self.buys / self.sells

    @property
    def total(self) -> int | None:
        if self.buys is None or self.sells is None:
            return None
        return self.buys + self.sells


@dataclass(frozen=True, slots=True)
class PriceLadder:
    """Percent price change over each window. Whole percents, e.g. -4.2.

    ``None`` means DexScreener did not report that window, which is common — it
    omits the key rather than sending zero, and on a live sample 13 of 30 pairs
    had no ``m5`` at all (including BONK's best pool). "No 5-minute move was
    reported" and "the price was flat over 5 minutes" are very different claims,
    so we keep them distinct all the way to the consumer.
    """

    m5: float | None
    h1: float | None
    h6: float | None
    h24: float | None

    def __post_init__(self) -> None:
        for name in ("m5", "h1", "h6", "h24"):
            finite_or_none(getattr(self, name), f"price_change.{name}")


@dataclass(frozen=True, slots=True)
class PoolRef:
    """Stable identity for one liquidity pool.

    The audit's pair-switching finding needs an identity to compare against.
    ``_best_pair`` can legitimately choose a different pool between reads, and
    when it does, a liquidity delta computed across the two is not a liquidity
    trend — it is the difference between two unrelated pools. Anything that
    compares two observations now compares their ``PoolRef`` first and refuses
    if they differ.
    """

    pair_address: str
    dex_id: str
    base_mint: str
    quote_mint: str
    quote_symbol: str
    created_at: float | None = None
    # False when no SOL/USDC/USDT-quoted pool existed and we fell back to a pool
    # priced in an unknown token. Such a pool may be *diagnostic* — it tells you
    # the token trades somewhere — but its price is not a valuation, because the
    # quote token's own price is unknown. Audit C8: this must veto entries.
    trusted_quote: bool = True

    def same_pool(self, other: PoolRef | None) -> bool:
        return other is not None and self.pair_address == other.pair_address


@dataclass(frozen=True, slots=True)
class CoinSnapshot:
    """Everything ``market.py`` knows about one coin at one instant.

    Every optional field is optional because the source genuinely omits it. The
    parser raises rather than substituting, so a value present here was actually
    reported.
    """

    symbol: str
    mint: str
    price_usd: float | None
    liquidity_usd: float | None
    volume_24h_usd: float | None
    volume_1h_usd: float | None
    fdv_usd: float | None
    price_change: PriceLadder
    txns_m5: TxnCounts
    txns_h1: TxnCounts
    txns_h24: TxnCounts
    pool: PoolRef
    provenance: Provenance
    candles_5m: CandleSeries | None = None
    candles_1h: CandleSeries | None = None
    quality: DataQuality = DataQuality.OK
    quality_reason: str | None = None

    def __post_init__(self) -> None:
        finite_or_none(self.price_usd, "price_usd")
        finite_or_none(self.liquidity_usd, "liquidity_usd")
        finite_or_none(self.volume_24h_usd, "volume_24h_usd")
        finite_or_none(self.volume_1h_usd, "volume_1h_usd")
        finite_or_none(self.fdv_usd, "fdv_usd")

    @property
    def tradeable(self) -> bool:
        """Whether this snapshot may support an *entry*.

        Exits are deliberately not gated on this — see ``risk.py``. A snapshot
        being untrustworthy is a reason not to buy and frequently a reason to
        sell, so the same predicate must not govern both.
        """
        return (
            self.quality is DataQuality.OK
            and self.pool.trusted_quote
            and self.price_usd is not None
            and self.liquidity_usd is not None
        )

    def age_seconds(self, now: float) -> float:
        return self.provenance.age_seconds(now)


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """A full read of every configured coin.

    ``ts`` is when the read *completed*. It is retained for logging and
    explicitly must not be used for staleness: use ``oldest_age_seconds``, or
    better, the per-coin provenance. The old code's single batch timestamp is
    precisely what let a three-minute-old price look ninety seconds fresh.
    """

    ts: float
    coins: dict[str, CoinSnapshot]

    def age_seconds(self, now: float) -> float:
        return now - self.ts

    def oldest_age_seconds(self, now: float) -> float:
        """Age of the *stalest* constituent observation. The honest number."""
        if not self.coins:
            return now - self.ts
        return max(c.age_seconds(now) for c in self.coins.values())


# ---------------------------------------------------------------------------
# Technicals — owned by signals.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Technicals:
    """Indicators for one coin on one timeframe.

    Every field is ``None`` when there is not enough *closed* candle history to
    compute it. The audit is right that these are correlated transforms of one
    close series rather than independent evidence, and that they are unproven as
    alpha; they are retained as labelled benchmark features, computed from
    closed bars only, and nothing in the system treats their agreement as
    confirmation.
    """

    timeframe: Timeframe
    candles_used: int
    pool_address: str

    rsi14: float | None
    rsi14_rising: bool | None

    ema9: float | None
    ema21: float | None
    ema9_above_ema21: bool | None
    pct_from_ema9: float | None
    pct_from_ema21: float | None

    macd_line: float | None
    macd_signal: float | None
    macd_hist: float | None
    macd_cross: Literal["bullish", "bearish", "none"] | None
    bars_since_cross: int | None

    bb_percent_b: float | None
    bb_bandwidth: float | None
    bb_expanding: bool | None

    atr14_pct: float | None

    # Renamed from volume_ratio_20. The audit's finding: the current bar was
    # included in its own 20-bar baseline *and* was frequently incomplete, so
    # the ratio was biased toward 1.0 early in a bar and could never exceed 20x.
    # This compares the last closed bar against the 20 bars preceding it.
    volume_ratio_prior_20: float | None

    realized_vol_pct: float | None

    pct_from_swing_high: float | None
    pct_from_swing_low: float | None


@dataclass(frozen=True, slots=True)
class FlowBrief:
    """Pool state and transaction counts.

    Deliberately no longer called "flow" in its field names. Counts are not
    notional and not unique actors; calling them flow is what let the old prompt
    describe them as "actual money moving". Signed notional flow from decoded
    swaps is the feature that would deserve the name, and it is not implemented,
    so it is not claimed.
    """

    txn_count_ratio_m5: float | None
    txn_count_ratio_h1: float | None
    txn_count_ratio_h24: float | None
    turnover_24h: float | None
    turnover_1h: float | None
    liquidity_usd: float | None
    # Set only when both reads came from the *same pool*. A liquidity delta
    # across a pair switch is the difference between two pools, not a trend.
    liquidity_trend_pct: float | None
    liquidity_trend_seconds: float | None
    liquidity_trend_pool: str | None
    price_ladder: PriceLadder


@dataclass(frozen=True, slots=True)
class TechnicalBrief:
    symbol: str
    m5: Technicals
    h1: Technicals
    flow: FlowBrief


# ---------------------------------------------------------------------------
# Sentiment — owned by sentiment.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SentimentBrief:
    """Attention, not polarity — and, after the audit, not text either.

    Audit C7: raw Reddit post and comment bodies were interpolated into the same
    prompt that had order authority, which is textbook indirect prompt
    injection. Any public author could address the trader directly. The excerpt
    field is gone from this type entirely; what survives is counts and rates,
    which are numbers and cannot issue instructions.

    The whole stream is disabled by default (``[sentiment] enabled``) pending
    the locked ablation the audit specifies. It is retained as an experiment
    source, not as a production input.
    """

    symbol: str
    ts: float
    source: Literal["praw", "arctic_shift"]

    # ``None`` means the period was not observed, which is emphatically not the
    # same as "nobody posted". Arctic Shift's index runs behind live Reddit, so
    # the most recent hour routinely contains nothing indexed; reporting that as
    # 0.0 mentions/hour manufactures a confident bearish reading out of a gap.
    mention_velocity_1h: float | None
    mention_velocity_24h: float | None
    # Computed against strictly non-overlapping prior buckets. The audit found
    # the old baseline included the current window through its rolling buckets,
    # which shrinks the very anomaly the z-score exists to detect.
    mention_zscore_7d: float | None
    unique_contributors_24h: int | None
    contributor_to_post_ratio: float | None

    # When the counts became *available to us*, which is later than when the
    # posts were made. Using post time as availability time is look-ahead.
    observed_through: float | None = None
    baseline_hours: int = 0

    degraded_reason: str | None = None


# ---------------------------------------------------------------------------
# Execution — owned by quotes.py and broker.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TokenMeta:
    """A mint and its decimals, from an authoritative source.

    Audit §15: decimals must never be *inferred* for execution. The old code
    could derive them from a base/UI amount pair and cache the guess; a wrong
    guess is a silent factor-of-1000 error in every subsequent quantity. An
    inferred value may be used to render a diagnostic, never to size a swap, so
    ``verified`` travels with the number and the broker refuses unverified.
    """

    mint: str
    decimals: int
    source: str
    verified: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.decimals, int) or isinstance(self.decimals, bool):
            raise ValidationError(f"decimals must be int, got {self.decimals!r}")
        if not 0 <= self.decimals <= 18:
            raise ValidationError(f"decimals {self.decimals} outside plausible range 0..18")

    def to_ui(self, amount_atomic: int) -> float:
        return amount_atomic / (10**self.decimals)

    def to_atomic(self, amount_ui: float) -> int:
        return int(finite(amount_ui, "amount_ui") * (10**self.decimals))


@dataclass(frozen=True, slots=True)
class Quote:
    """An executable route, priced at an exact input amount.

    This type is the audit's C2 and C5 fixes made structural. The old
    ``FillQuote`` carried a dollar notional and an effective price and discarded
    Jupiter's exact ``inAmount``/``outAmount``; the broker then reconstructed a
    quantity from dollars ÷ price, producing a pair of numbers that corresponded
    to no swap that was ever offered. A swap has an exact input and an expected
    output. Both are here, as integers, and the broker consumes *them* rather
    than re-deriving anything.

    ``fingerprint`` binds the quote to that exact swap (see ``ids.py``). An
    order carries the fingerprint it was built from and the broker recomputes
    it, so a quote obtained for one size can never be used to fill another —
    audit C3.

    There is deliberately no ``degraded`` flag. A route we could not obtain is
    not a worse route; it is not a route. See :class:`ValuationEstimate`.
    """

    symbol: str
    side: Side
    input_token: TokenMeta
    output_token: TokenMeta
    in_amount_atomic: int
    out_amount_atomic: int
    # The worst output the route may deliver at the configured slippage. What a
    # conservative simulation and any real submission must assume.
    min_out_amount_atomic: int
    price_impact_pct: float
    route_labels: tuple[str, ...]
    fingerprint: str
    requested_at: float
    received_at: float
    context_slot: int | None = None
    expires_at: float | None = None
    # USD value of the *input* leg at quote time, for reporting only. Never an
    # input to sizing: sizing uses atomic amounts.
    reference_price_usd: float | None = None

    def __post_init__(self) -> None:
        atomic(self.in_amount_atomic, "in_amount_atomic")
        atomic(self.out_amount_atomic, "out_amount_atomic")
        atomic(self.min_out_amount_atomic, "min_out_amount_atomic")
        finite(self.price_impact_pct, "price_impact_pct")
        finite(self.requested_at, "requested_at")
        finite(self.received_at, "received_at")
        finite_or_none(self.reference_price_usd, "reference_price_usd")
        if self.in_amount_atomic == 0:
            raise ValidationError("a quote for zero input is not a quote")
        if self.min_out_amount_atomic > self.out_amount_atomic:
            raise ValidationError(
                f"min_out {self.min_out_amount_atomic} exceeds expected out "
                f"{self.out_amount_atomic}"
            )

    @property
    def latency_seconds(self) -> float:
        return self.received_at - self.requested_at

    def age_seconds(self, now: float) -> float:
        return now - self.received_at

    def is_expired(self, now: float) -> bool:
        return self.expires_at is not None and now >= self.expires_at

    @property
    def effective_price_usd(self) -> float | None:
        """Token price in USD implied by the route, in UI units.

        ``None`` when the USD leg cannot be identified. Both legs are converted
        to UI units before dividing — a scaling error does not look like an
        error, it looks like a plausible price off by a power of ten.
        """
        in_ui = self.input_token.to_ui(self.in_amount_atomic)
        out_ui = self.output_token.to_ui(self.out_amount_atomic)
        if in_ui == 0 or out_ui == 0:
            return None
        return out_ui / in_ui if self.side is Side.SELL else in_ui / out_ui

    @property
    def token_amount_atomic(self) -> int:
        """Atomic amount of the *traded token*, whichever leg it is on."""
        return self.out_amount_atomic if self.side is Side.BUY else self.in_amount_atomic

    @property
    def token_meta(self) -> TokenMeta:
        return self.output_token if self.side is Side.BUY else self.input_token

    @property
    def usd_notional(self) -> float:
        """Dollars on the USD leg of this route, in UI units.

        The cash side is the input on a BUY and the output on a SELL, so this is
        the amount the book actually moves — not a price multiplied back out by
        a quantity. It lives here rather than at each call site because risk's
        post-quote reconfirmation and the reported fill notional must be the
        same number, and two reconstructions of it eventually will not be.
        """
        usd_leg = self.input_token if self.side is Side.BUY else self.output_token
        atomic = self.in_amount_atomic if self.side is Side.BUY else self.out_amount_atomic
        return usd_leg.to_ui(atomic)


@dataclass(frozen=True, slots=True)
class ValuationEstimate:
    """A non-executable estimate of what a position is worth.

    Audit C4. When Jupiter is unavailable the old code manufactured a quote from
    the DexScreener mid plus a fixed spread, flagged it ``degraded``, and let
    the broker fill against it — so a routing outage, which is exactly when a
    mid-price fiction is least credible, produced trades that could not have
    happened.

    This type exists so that failure has somewhere to go that is not the order
    path. It is structurally not a :class:`Quote`: the broker's signature will
    not accept it. Risk and reporting use it to mark a book conservatively and
    to escalate, and it carries its own haircut so no consumer forgets to apply
    one.
    """

    symbol: str
    mid_price_usd: float | None
    haircut_pct: float
    reason: str
    at: float
    source: str

    def __post_init__(self) -> None:
        finite_or_none(self.mid_price_usd, "mid_price_usd")
        non_negative(self.haircut_pct, "haircut_pct")

    @property
    def conservative_price_usd(self) -> float | None:
        """The mid, marked down. Never marked up.

        A degraded valuation must not be able to look better than a real one, or
        an outage becomes a trading signal.
        """
        if self.mid_price_usd is None:
            return None
        return self.mid_price_usd * (1.0 - self.haircut_pct / 100.0)


@dataclass(frozen=True, slots=True)
class Fill:
    """A settled or failed execution attempt.

    Atomic amounts are the record; dollars are derived for reporting. A failed
    attempt is still a Fill row with ``state=FAILED`` and zero amounts but
    non-zero gas — that is what a failed Solana swap costs, and dropping the row
    would hide the cost.
    """

    fill_id: str
    order_id: str
    intent_id: str
    decision_id: str | None
    ts: float
    symbol: str
    side: Side
    state: OrderState
    in_amount_atomic: int
    out_amount_atomic: int
    token_amount_atomic: int
    token_decimals: int
    quote_fingerprint: str
    price_usd: float | None
    notional_usd: float
    price_impact_pct: float | None
    pool_fee_usd: float
    gas_usd: float
    realized_pnl_usd: float = 0.0
    slippage_bps_vs_quote: float | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        atomic(self.in_amount_atomic, "in_amount_atomic")
        atomic(self.out_amount_atomic, "out_amount_atomic")
        atomic(self.token_amount_atomic, "token_amount_atomic")
        finite_or_none(self.price_usd, "price_usd")
        non_negative(self.notional_usd, "notional_usd")
        non_negative(self.pool_fee_usd, "pool_fee_usd")
        non_negative(self.gas_usd, "gas_usd")
        finite(self.realized_pnl_usd, "realized_pnl_usd")

    @property
    def failed(self) -> bool:
        return self.state in (OrderState.FAILED, OrderState.EXPIRED)

    @property
    def quantity(self) -> float:
        """Token quantity in UI units. Derived — the atomic amount is the fact."""
        return self.token_amount_atomic / (10**self.token_decimals)


# Why an order exists, not who asked for it. Three values, because three is the
# complete list of reasons this system places an order: a strategy target, a
# stop, or a risk-driven unwind. A strategy *name* is not a member of this set —
# putting one here was how the journal lost the ability to answer "was this a
# forced exit?" by filter.
OrderSource = Literal["strategy", "stop_loss", "risk_exit"]


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """The durable record of an order we are about to attempt.

    Written before anything is quoted or submitted, because audit C11's
    recovery story requires it: a crash mid-submission must leave a row that
    startup reconciliation can find. ``intent_id`` is the idempotency key — a
    retry reuses it, so a duplicate attempt is detectable rather than being a
    second order.
    """

    intent_id: str
    decision_id: str | None
    action_id: str | None
    run_id: str
    ts: float
    symbol: str
    side: Side
    # The *bound* size: exactly what was quoted, in atomic units of the input.
    in_amount_atomic: int
    max_in_amount_atomic: int
    source: OrderSource
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Position:
    """An open position. Quantity is atomic; UI quantity is derived.

    Cost basis includes fees and gas, so a position's break-even is its true
    break-even.
    """

    symbol: str
    mint: str
    quantity_atomic: int
    decimals: int
    avg_entry_price_usd: float
    opened_at: float
    cost_basis_usd: float

    def __post_init__(self) -> None:
        atomic(self.quantity_atomic, "quantity_atomic")
        non_negative(self.avg_entry_price_usd, "avg_entry_price_usd")
        non_negative(self.cost_basis_usd, "cost_basis_usd")

    @property
    def quantity(self) -> float:
        return self.quantity_atomic / (10**self.decimals)

    def value_usd(self, mark_price: float | None) -> float | None:
        return None if mark_price is None else self.quantity * mark_price

    def unrealized_pnl_usd(self, mark_price: float | None) -> float | None:
        value = self.value_usd(mark_price)
        return None if value is None else value - self.cost_basis_usd

    def unrealized_pnl_pct(self, mark_price: float | None) -> float | None:
        pnl = self.unrealized_pnl_usd(mark_price)
        if pnl is None or self.cost_basis_usd == 0:
            return None
        return 100.0 * pnl / self.cost_basis_usd

    def age_seconds(self, now: float) -> float:
        return now - self.opened_at


class Broker(Protocol):
    """The seam that keeps this reversible.

    ``LocalPaperBroker`` implements it against a transactional local ledger. A
    real venue would attach here.

    Two things changed after the audit. ``place_order`` takes a :class:`Quote`
    and an :class:`OrderIntent` rather than a dollar amount, so the broker
    cannot invent a quantity and cannot be handed a non-executable valuation.
    And ``mode`` is part of the protocol, so a read-only run holds an object
    that is structurally incapable of mutating the book rather than one that
    promises not to.
    """

    @property
    def mode(self) -> ExecutionMode: ...

    def place_order(self, intent: OrderIntent, quote: Quote, *, now: float) -> Fill: ...

    def get_positions(self) -> dict[str, Position]: ...


# ---------------------------------------------------------------------------
# Portfolio — owned by portfolio.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Mark:
    """What one position is worth, and how much that number can be trusted.

    Audit C8: the old ``mark()`` carried an unmarkable position at *cost basis*,
    which shows zero loss during exactly the two events that matter — a rug and
    a data outage — and so guarantees the stop never fires when it is most
    needed. A mark is now either present with a stated basis, or explicitly
    absent, and absence is a risk incident rather than a number.
    """

    symbol: str
    price_usd: float | None
    basis: Literal["route", "mid", "estimate", "unavailable"]
    provenance: Provenance | None
    haircut_pct: float = 0.0
    reason: str | None = None

    def __post_init__(self) -> None:
        finite_or_none(self.price_usd, "mark.price_usd")
        non_negative(self.haircut_pct, "mark.haircut_pct")

    @property
    def usable(self) -> bool:
        return self.price_usd is not None and self.basis != "unavailable"

    @property
    def is_executable_basis(self) -> bool:
        """Only a route-derived mark is an estimate of liquidation value."""
        return self.basis == "route"


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """The book, marked at ``ts``.

    ``total_value_usd`` is ``None`` when any held position is unmarkable. That
    is deliberate and it propagates: a book whose value is unknown must not
    report a confident number, because every downstream percentage — return,
    drawdown, position cap — would then be computed against a figure partly
    made of cost basis standing in for a price nobody could obtain.
    """

    ts: float
    cash_usd: float
    positions: dict[str, Position]
    marks: dict[str, Mark]
    position_values_usd: dict[str, float | None]
    unrealized_pnl_usd: float | None
    realized_pnl_usd: float
    total_value_usd: float | None
    starting_cash_usd: float
    fees_paid_usd: float = 0.0
    gas_paid_usd: float = 0.0
    unmarkable: tuple[str, ...] = ()

    @property
    def total_return_pct(self) -> float | None:
        if self.total_value_usd is None or self.starting_cash_usd == 0:
            return None
        return (
            100.0 * (self.total_value_usd - self.starting_cash_usd) / self.starting_cash_usd
        )

    @property
    def gross_exposure_usd(self) -> float | None:
        values = list(self.position_values_usd.values())
        if any(v is None for v in values):
            return None
        return sum(v for v in values if v is not None)

    @property
    def gross_exposure_pct(self) -> float | None:
        gross = self.gross_exposure_usd
        if gross is None or not self.total_value_usd:
            return None
        return 100.0 * gross / self.total_value_usd

    @property
    def fully_marked(self) -> bool:
        return not self.unmarkable


# ---------------------------------------------------------------------------
# Risk — owned by risk.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RiskBounds:
    """The maximum the risk layer permits — not an approval of an order.

    Audit C3's structural half. The old ``check()`` returned an approved dollar
    amount that the caller then executed against a quote obtained for a
    *different* amount. Risk now answers a different question: "what is the most
    you may do?" The execution layer takes that bound, requotes at the size it
    actually intends, and only then binds and submits. Risk never mutates an
    order, so it cannot produce one that was never priced.

    ``max_notional_usd == 0`` with ``vetoes`` non-empty is a refusal. Every
    binding rule names itself so the reason survives into the journal.
    """

    symbol: str
    side: Side
    max_notional_usd: float
    vetoes: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    binding_rule: str | None = None
    notes: tuple[str, ...] = ()
    bypassed_rules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        non_negative(self.max_notional_usd, "max_notional_usd")

    @property
    def permitted(self) -> bool:
        return not self.vetoes and self.max_notional_usd > 0

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons) if self.reasons else ""


@dataclass(frozen=True, slots=True)
class RiskState:
    """Continuous, cross-order risk. Audit C10.

    Per-order checks cannot see that three positions are one bet. This carries
    the portfolio-level facts a pre-trade check must consult, plus the kill
    switch, which is the only control that reliably works when something
    unanticipated is happening.
    """

    ts: float
    halted: bool = False
    halt_reasons: tuple[str, ...] = ()
    gross_exposure_pct: float | None = None
    rolling_loss_pct: float | None = None
    peak_value_usd: float | None = None
    drawdown_pct: float | None = None
    consecutive_failures: int = 0
    quarantined_symbols: frozenset[str] = frozenset()
    data_health_ok: bool = True

    @property
    def may_open(self) -> bool:
        return not self.halted and self.data_health_ok


# ---------------------------------------------------------------------------
# Strategy I/O — owned by strategy.py, consumed by loop.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Forecast:
    """A horizon-specific view with an explicit economic target.

    Audit C6's replacement for a categorical BUY/SELL/HOLD. A direction with no
    horizon and no magnitude cannot be compared against a cost hurdle, so the
    old system could not tell whether a trade was worth its own gas — and the
    12-hour run contains a directionally correct trade that lost money for
    exactly that reason.

    ``expected_net_return_pct`` is *after* estimated costs. ``lower_quantile``
    is the conservative end of the predictive interval and is what the entry
    hurdle is actually tested against; trading the mean of a wide distribution
    is how noisy estimates become positions.

    ``calibration_id`` is ``None`` for a baseline that makes no calibration
    claim. It is the field a future calibrated model fills in, and its absence
    is what stops an uncalibrated number being sized as if it were calibrated.
    """

    symbol: str
    horizon_seconds: float
    expected_net_return_pct: float | None
    lower_quantile_pct: float | None
    upper_quantile_pct: float | None
    model_id: str
    calibration_id: str | None = None
    features_missing: tuple[str, ...] = ()
    note: str = ""
    # When this view goes stale. A forecast generated from a bar that closed at
    # 10:00 with a one-hour horizon is not still actionable at 11:30, and a
    # replay that lets a queued order fill against a forecast the live system
    # would have discarded is measuring a strategy nobody could run. ``None``
    # means the producer makes no staleness claim.
    valid_until: float | None = None
    # P(return > 0) at this horizon. Distinct from the quantiles: a forecast can
    # be confidently small-positive or barely-positive-but-huge, and sizing
    # should be able to tell those apart. ``None`` for a model making no
    # probabilistic claim — a baseline point estimate must not be read as 100%.
    probability_positive: float | None = None

    def __post_init__(self) -> None:
        positive(self.horizon_seconds, "horizon_seconds")
        finite_or_none(self.expected_net_return_pct, "expected_net_return_pct")
        finite_or_none(self.lower_quantile_pct, "lower_quantile_pct")
        finite_or_none(self.upper_quantile_pct, "upper_quantile_pct")
        finite_or_none(self.valid_until, "valid_until")
        finite_or_none(self.probability_positive, "probability_positive")
        if self.probability_positive is not None and not (
            0.0 <= self.probability_positive <= 1.0
        ):
            raise ValidationError(
                f"probability_positive {self.probability_positive} outside [0, 1]"
            )

    @property
    def actionable(self) -> bool:
        return (
            self.expected_net_return_pct is not None and self.lower_quantile_pct is not None
        )


@dataclass(frozen=True, slots=True)
class TargetPosition:
    """Desired inventory for one symbol, in dollars. The portfolio layer's output.

    A *target*, not a trade. The execution layer diffs targets against held
    inventory and schedules the delta, which is what makes "hold what you have"
    and "buy more" the same statement rather than two code paths. Cash is
    expressible: zero targets everywhere is a valid, expected portfolio.
    """

    symbol: str
    target_usd: float
    forecast: Forecast | None = None
    rationale: str = ""

    def __post_init__(self) -> None:
        non_negative(self.target_usd, "target_usd")


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    """Everything one strategy invocation produced."""

    decision_id: str
    ts: float
    strategy_id: str
    market_read: str
    targets: tuple[TargetPosition, ...]
    forecasts: tuple[Forecast, ...] = ()
    diagnostics: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Advisory LLM I/O — owned by brain.py
# ---------------------------------------------------------------------------
#
# Audit C6 removes the LLM from trade selection and sizing. These types survive
# only because the audit's Phase 1 permits a clearly-labelled alternative path
# and its §8 leaves the door open to an isolated, ablated extractor. They are
# named for what they are — advisory — and `strategy.py` refuses to route them
# into orders unless the operator has explicitly opted in.


class AdvisoryAction(BaseModel):
    """One advisory opinion about one coin. Not an order.

    Validated strictly: the audit found ``size_usd`` had no finite or
    non-negative constraint and that ``_normalize`` silently repaired bad
    output. Silent repair is the problem — a model that emits an invalid size
    has malfunctioned, and the correct response is to discard the whole output,
    not to guess what it meant and trade the guess.
    """

    action: Literal["BUY", "SELL", "HOLD"]
    symbol: str
    size_usd: float = Field(ge=0.0, description="USD notional. Exactly 0.0 for HOLD.")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Which specific number drove this.")

    @field_validator("size_usd", "confidence")
    @classmethod
    def _finite(cls, v: float) -> float:
        # Pydantic's ge/le do not reject NaN: every comparison against NaN is
        # false, so `ge=0.0` passes it through. This is the audit's exact
        # finding and it has to be an explicit check.
        if not math.isfinite(v):
            raise ValueError("must be finite")
        return v


class AdvisoryDecision(BaseModel):
    market_read: str = Field(description="One paragraph on what is happening now.")
    actions: list[AdvisoryAction] = Field(description="One entry per configured coin.")


# ---------------------------------------------------------------------------
# Journal records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """One row in ``decisions.jsonl``.

    Carries ``decision_id`` so fills join to actions by identity rather than by
    symbol — the audit's ``_decision_line`` finding, where a stop-loss and a
    strategy order on the same coin in the same tick were indistinguishable.
    """

    decision_id: str
    run_id: str
    ts: float
    strategy_id: str
    market_read: str
    targets: tuple[TargetPosition, ...]
    bounds: tuple[RiskBounds, ...]
    intents: tuple[OrderIntent, ...]
    fills: tuple[Fill, ...]
    mode: ExecutionMode
    forecasts: tuple[Forecast, ...] = ()
    risk_state: RiskState | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    model: str = ""
    effort: str = ""
    thinking: str | None = None
    advisory_used: bool = False


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """Everything one symbol contributes to a decision.

    A ``None`` stream means unavailable. Consumers must represent that as
    unavailable rather than substituting a neutral value — and, after the audit,
    a *critical* unavailable stream vetoes entries rather than merely inviting a
    model to "discount" it.
    """

    symbol: str
    snapshot: CoinSnapshot
    technicals: TechnicalBrief | None
    sentiment: SentimentBrief | None
    sentiment_unavailable_reason: str | None = None
    mark: Mark | None = None


# ---------------------------------------------------------------------------
# Replay — owned by backtest/, consumed by histdata/ and execution/
# ---------------------------------------------------------------------------


class FidelityTier(StrEnum):
    """How much of a PnL claim the available data can actually support.

    This exists because the most dangerous output of a backtest is a number
    that looks like dollars but was computed from data that cannot price a
    swap. Solana execution depends on size-specific AMM depth, route
    availability, priority fees and whether the transaction landed at all.
    OHLCV knows none of that, so a bar-based fill is an estimate of *signal*,
    not of *money*, and every artifact derived from one must say so.

    The tier is carried on the run manifest and on every execution report, and
    :mod:`memetrader.validation.promotion` refuses to promote below TIER_2.
    """

    # Candles only. What this repository currently has: 24 coins, 1h and 5m,
    # bounded by the vendor's 180-day public horizon.
    TIER_0 = "tier_0_ohlcv"
    # Individual swaps/trades plus conservative cost estimates.
    TIER_1 = "tier_1_swaps"
    # Historical pool states or exact size-specific quote ladders.
    TIER_2 = "tier_2_executable"
    # Prospective shadow quotes and observed landed/failed transactions, used
    # to calibrate the TIER_2 simulator against reality.
    TIER_3 = "tier_3_calibrated"

    @property
    def permits_pnl_claim(self) -> bool:
        return self in (FidelityTier.TIER_2, FidelityTier.TIER_3)


# The exact sentence a run must print when its tier cannot price a swap. It is
# a constant so that no report can quietly soften the wording.
NON_EXECUTABLE_NOTICE = (
    "This experiment estimates signal quality, but executable after-cost PnL "
    "has not been demonstrated."
)


class EventKind(StrEnum):
    """What a replayed event is. Ordering within one timestamp depends on it."""

    UNIVERSE = "universe"
    BAR_CLOSE = "bar_close"
    POOL_STATE = "pool_state"
    SWAP = "swap"
    QUOTE = "quote"
    SOCIAL = "social"
    MARK = "mark"
    DECISION_TICK = "decision_tick"
    ORDER_READY = "order_ready"
    EXECUTION = "execution"


# Deterministic intra-timestamp ordering. Every state update lands before the
# decision that reads it, and the decision lands before any order it produces —
# so a strategy cannot act on a bar in the same breath as that bar arriving,
# even when both carry an identical ``available_time``. The gaps leave room to
# insert kinds later without renumbering.
EVENT_PRIORITY: dict[EventKind, int] = {
    EventKind.UNIVERSE: 0,
    EventKind.BAR_CLOSE: 10,
    EventKind.POOL_STATE: 20,
    EventKind.SWAP: 30,
    EventKind.QUOTE: 40,
    EventKind.SOCIAL: 50,
    EventKind.MARK: 60,
    EventKind.DECISION_TICK: 70,
    EventKind.ORDER_READY: 80,
    EventKind.EXECUTION: 90,
}


@dataclass(frozen=True, slots=True)
class HistoricalEvent:
    """One thing that became knowable at ``available_time``.

    The replay queue is sorted by :attr:`sort_key` and nothing else. Note what
    is *not* in the key: ``event_time``. Sorting a replay by when things
    happened rather than by when they were knowable is the single most common
    way a backtest grants itself foresight, and it is invisible in the results
    because the equity curve still looks like a plausible equity curve.
    """

    kind: EventKind
    available_time: float
    asset_id: str | None
    payload: object
    event_time: float | None = None
    received_time: float | None = None
    sequence: int | None = None
    source: str = ""
    pool_id: str | None = None
    quality_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        finite(self.available_time, "available_time")
        finite_or_none(self.event_time, "event_time")
        finite_or_none(self.received_time, "received_time")
        if self.event_time is not None and self.event_time > self.available_time:
            raise ValidationError(
                f"{self.kind} for {self.asset_id}: event_time {self.event_time} is after "
                f"available_time {self.available_time} — an event cannot be knowable "
                "before it happens"
            )

    @property
    def sort_key(self) -> tuple[float, int, int, str, str]:
        """Total order over events. Must be deterministic for replay equality."""
        return (
            self.available_time,
            EVENT_PRIORITY[self.kind],
            self.sequence if self.sequence is not None else -1,
            self.source,
            self.asset_id or "",
        )


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Where a trade's money went, split so attribution can subtract it.

    Every field is signed rather than non-negative: ``latency_cost_usd`` is
    genuinely negative when the delay happened to help, and clamping it at zero
    would make latency look like a one-way tax and overstate the strategy's
    gross alpha by exactly the favourable half of the distribution.
    """

    venue_fee_usd: float = 0.0
    network_fee_usd: float = 0.0
    priority_fee_usd: float = 0.0
    spread_usd: float = 0.0
    price_impact_usd: float = 0.0
    latency_cost_usd: float = 0.0
    failure_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "venue_fee_usd",
            "network_fee_usd",
            "priority_fee_usd",
            "spread_usd",
            "price_impact_usd",
            "latency_cost_usd",
            "failure_cost_usd",
        ):
            finite(getattr(self, name), f"cost.{name}")

    @property
    def total_usd(self) -> float:
        return (
            self.venue_fee_usd
            + self.network_fee_usd
            + self.priority_fee_usd
            + self.spread_usd
            + self.price_impact_usd
            + self.latency_cost_usd
            + self.failure_cost_usd
        )


@dataclass(frozen=True, slots=True)
class OrderReceipt:
    """The venue's acknowledgement that an order was accepted for execution.

    Separate from the fill because acceptance and settlement are separate
    facts separated by time, and collapsing them is how a simulator ends up
    filling an order the real network would have dropped. ``ready_at`` is when
    the order may first meet a market state — submission time plus quote,
    signing and landing latency.
    """

    receipt_id: str
    intent_id: str
    accepted: bool
    submitted_at: float
    ready_at: float
    quote_fingerprint: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        finite(self.submitted_at, "submitted_at")
        finite(self.ready_at, "ready_at")
        if self.ready_at < self.submitted_at:
            raise ValidationError(
                f"ready_at {self.ready_at} precedes submitted_at {self.submitted_at}"
            )


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    """What a venue reports back about one order attempt.

    ``report_id`` is the idempotency key: a venue may legitimately emit the
    same report twice and the ledger must treat the second as a no-op rather
    than as a second fill. ``fill`` is ``None`` for a terminal non-fill —
    a failed route, an expired quote, a dropped transaction — and that is a
    real, costly outcome, not an absence to be skipped.
    """

    report_id: str
    intent_id: str
    order_id: str | None
    state: OrderState
    ts: float
    fidelity: FidelityTier
    fill: Fill | None = None
    costs: CostBreakdown | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        finite(self.ts, "execution_report.ts")
        if self.fill is not None and self.fill.state is not self.state:
            raise ValidationError(
                f"report state {self.state} disagrees with fill state {self.fill.state}"
            )


__all__ = [
    "EVENT_PRIORITY",
    "NON_EXECUTABLE_NOTICE",
    "AdvisoryAction",
    "AdvisoryDecision",
    "Broker",
    "Candle",
    "CandleSeries",
    "CoinSnapshot",
    "CostBreakdown",
    "DataQuality",
    "DecisionRecord",
    "EventKind",
    "EvidenceBundle",
    "ExecutionMode",
    "ExecutionReport",
    "FidelityTier",
    "Fill",
    "FlowBrief",
    "Forecast",
    "HistoricalEvent",
    "Mark",
    "MarketSnapshot",
    "Observed",
    "OrderIntent",
    "OrderReceipt",
    "OrderState",
    "PoolRef",
    "PortfolioState",
    "Position",
    "PriceLadder",
    "Provenance",
    "Quote",
    "RiskBounds",
    "RiskState",
    "SentimentBrief",
    "Side",
    "StrategyDecision",
    "TargetPosition",
    "TechnicalBrief",
    "Technicals",
    "Timeframe",
    "TokenMeta",
    "TxnCounts",
    "ValidationError",
    "ValuationEstimate",
    "assert_live_supported",
    "atomic",
    "finite",
    "finite_or_none",
    "non_negative",
    "positive",
]
