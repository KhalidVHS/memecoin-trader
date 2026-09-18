"""Shared shapes for every module in the system.

These are our *internal normalized* types, deliberately not the raw shapes of
any API. Adapters (``market.py``, ``quotes.py``, ``sentiment.py``) translate
vendor JSON into these at their boundary, which is what keeps a surprise in a
vendor response from rippling through the rest of the codebase.

Two conventions that everything depends on:

* **All timestamps are epoch seconds as ``float``.** DexScreener returns
  milliseconds and GeckoTerminal returns seconds; adapters normalize at the
  boundary. Nothing downstream should ever divide by 1000.
* **All percentages are whole numbers, not fractions.** ``-4.2`` means -4.2%.
  Jupiter returns ``priceImpactPct`` as a string fraction in 0..1; ``quotes.py``
  multiplies by 100 at the boundary. The one exception is config values whose
  name ends in ``_pct`` and are documented as fractions (``max_position_pct``,
  ``stop_loss_pct``) — those are fractions because they are multipliers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class Timeframe(StrEnum):
    M5 = "5m"
    H1 = "1h"


# ---------------------------------------------------------------------------
# Market data — owned by market.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candle:
    """One OHLCV bar. ``ts`` is the bar's open time, epoch seconds."""

    ts: float
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class PriceLadder:
    """Percent price change over each window. Whole percents, e.g. -4.2."""

    m5: float
    h1: float
    h6: float
    h24: float


@dataclass(frozen=True, slots=True)
class TxnCounts:
    """Buy/sell transaction counts over a window."""

    buys: int
    sells: int

    @property
    def ratio(self) -> float:
        """Buys per sell. Returns ``float('inf')`` when there are no sells."""
        if self.sells == 0:
            return float("inf") if self.buys else 1.0
        return self.buys / self.sells


@dataclass(frozen=True, slots=True)
class CoinSnapshot:
    """Everything ``market.py`` knows about one coin at one instant.

    Sourced from the single highest-liquidity DexScreener pair for the mint,
    plus GeckoTerminal OHLCV for that same pool.
    """

    symbol: str
    mint: str
    price_usd: float
    liquidity_usd: float
    volume_24h_usd: float
    volume_1h_usd: float
    fdv_usd: float | None
    price_change: PriceLadder
    txns_m5: TxnCounts
    txns_h1: TxnCounts
    txns_h24: TxnCounts
    pair_address: str
    dex_id: str
    pair_created_at: float | None  # epoch seconds
    candles_5m: tuple[Candle, ...] = ()
    candles_1h: tuple[Candle, ...] = ()
    # True when some part of this snapshot came from a degraded path (e.g.
    # candles unavailable, so technicals will be partial). Surfaced in the
    # prompt so the model can discount rather than trade a blank.
    degraded: bool = False
    degraded_reason: str | None = None


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """A full read of every configured coin at one instant."""

    ts: float  # epoch seconds, when the snapshot was taken
    coins: dict[str, CoinSnapshot]  # keyed by symbol

    def age_seconds(self, now: float) -> float:
        return now - self.ts


# ---------------------------------------------------------------------------
# Technicals — owned by signals.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Technicals:
    """Indicators for one coin on one timeframe.

    Every field is ``None`` when there is not enough candle history to compute
    it. Consumers must render "n/a" rather than substituting a zero — a missing
    indicator and a zero-valued indicator are very different claims.
    """

    timeframe: Timeframe
    candles_used: int

    rsi14: float | None
    rsi14_rising: bool | None  # direction matters more than level

    ema9: float | None
    ema21: float | None
    ema9_above_ema21: bool | None
    pct_from_ema9: float | None  # whole percent
    pct_from_ema21: float | None

    macd_line: float | None
    macd_signal: float | None
    macd_hist: float | None
    macd_cross: Literal["bullish", "bearish", "none"] | None
    bars_since_cross: int | None  # freshness of the cross

    bb_percent_b: float | None  # 0 = lower band, 1 = upper band
    bb_bandwidth: float | None  # (upper-lower)/mid, whole percent
    bb_expanding: bool | None

    atr14_pct: float | None  # ATR as whole percent of price

    volume_ratio_20: float | None  # latest volume / 20-period mean

    pct_from_swing_high: float | None  # negative = below the high
    pct_from_swing_low: float | None  # positive = above the low


@dataclass(frozen=True, slots=True)
class FlowBrief:
    """On-chain flow, passed through from the DexScreener snapshot.

    No traditional-TA equivalent exists for these, and for this asset class they
    are arguably the highest-signal numbers available: actual money moving.
    """

    buy_sell_ratio_m5: float
    buy_sell_ratio_h1: float
    buy_sell_ratio_h24: float
    turnover_24h: float  # volume_24h / liquidity
    turnover_1h: float
    liquidity_usd: float
    liquidity_trend_pct: float | None  # vs the previous snapshot; None on first
    price_ladder: PriceLadder


@dataclass(frozen=True, slots=True)
class TechnicalBrief:
    """What the model sees for one coin's technical picture."""

    symbol: str
    m5: Technicals
    h1: Technicals
    flow: FlowBrief


# ---------------------------------------------------------------------------
# Sentiment — owned by sentiment.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TopPost:
    title: str
    score: int
    age_hours: float
    subreddit: str


@dataclass(frozen=True, slots=True)
class SentimentBrief:
    """Attention, not polarity.

    For memecoins sentiment polarity is manufactured — shill farms produce
    bullish text on demand. What survives scrutiny is velocity (is attention
    accelerating), breadth (how many distinct people) and novelty. Fields are
    ordered by how much they are trusted, and the prompt says so explicitly.
    """

    symbol: str
    ts: float  # epoch seconds
    source: Literal["praw", "arctic_shift"]

    mention_velocity_1h: float  # mentions/hour over the last hour
    mention_velocity_24h: float  # mentions/hour averaged over 24h
    mention_zscore_7d: float | None  # is current attention unusual for this coin
    unique_contributors_24h: int
    contributor_to_post_ratio: float | None  # low = few accounts posting a lot

    top_posts: tuple[TopPost, ...] = ()
    polarity: float | None = None  # -1..1, explicitly low-trust in the prompt

    # Set when the source answered but with partial data (e.g. no 7d baseline
    # yet). A total failure returns ``None`` instead of a degraded brief.
    degraded_reason: str | None = None


# ---------------------------------------------------------------------------
# Execution — owned by quotes.py and broker.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FillQuote:
    """A real route priced at an exact notional, before fees and gas."""

    symbol: str
    mint: str
    side: Side
    usd_notional: float
    price_usd: float  # effective price = outAmount/inAmount, normalized
    price_impact_pct: float  # whole percent; Jupiter's fraction x100
    route_labels: tuple[str, ...]  # e.g. ("Raydium",) — drives the fee lookup
    pool_fee_pct: float  # resolved from route_labels via config
    degraded: bool = False  # True when this is a DexScreener mid + slippage
    degraded_reason: str | None = None


@dataclass(frozen=True, slots=True)
class Fill:
    """A completed (or failed) paper trade. One row in ``trades.jsonl``."""

    ts: float
    symbol: str
    side: Side
    requested_usd: float
    filled_usd: float  # 0.0 when the transaction failed
    price_usd: float
    quantity: float  # token units; negative is never used, side carries sign
    price_impact_pct: float
    pool_fee_usd: float
    gas_usd: float  # charged even when the transaction failed
    realized_pnl_usd: float = 0.0  # non-zero only on SELL
    failed: bool = False
    degraded: bool = False
    note: str | None = None


@dataclass(frozen=True, slots=True)
class Position:
    """An open position. Cost basis is average, and we never DCA a loser, so in
    practice a position has a single entry — but the field is an average so that
    a scale-in remains representable if that rule is ever relaxed."""

    symbol: str
    quantity: float
    avg_entry_price_usd: float
    opened_at: float  # epoch seconds
    cost_basis_usd: float  # what was actually paid, fees and gas included

    def unrealized_pnl_usd(self, mark_price: float) -> float:
        return self.quantity * mark_price - self.cost_basis_usd

    def unrealized_pnl_pct(self, mark_price: float) -> float:
        if self.cost_basis_usd == 0:
            return 0.0
        return 100.0 * self.unrealized_pnl_usd(mark_price) / self.cost_basis_usd

    def age_seconds(self, now: float) -> float:
        return now - self.opened_at


class Broker(Protocol):
    """The seam that keeps this reversible.

    ``LocalPaperBroker`` implements it against a JSON state file. A real venue
    would attach here — and the LLM layer would never learn the difference.
    """

    def place_order(self, symbol: str, side: Side, usd_notional: float) -> Fill: ...

    def get_positions(self) -> dict[str, Position]: ...


# ---------------------------------------------------------------------------
# Portfolio — owned by portfolio.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """The book, marked to market at ``ts``."""

    ts: float
    cash_usd: float
    positions: dict[str, Position]
    marks: dict[str, float]  # symbol -> mark price used
    position_values_usd: dict[str, float]
    unrealized_pnl_usd: float
    realized_pnl_usd: float  # cumulative, since inception
    total_value_usd: float  # cash + position values
    starting_cash_usd: float
    fees_paid_usd: float = 0.0
    gas_paid_usd: float = 0.0

    @property
    def total_return_pct(self) -> float:
        if self.starting_cash_usd == 0:
            return 0.0
        return 100.0 * (self.total_value_usd - self.starting_cash_usd) / self.starting_cash_usd


# ---------------------------------------------------------------------------
# Risk — owned by risk.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TradeProposal:
    """A model action, resolved against live data, ready to be judged.

    ``loop.py`` builds these from ``Action`` objects; ``risk.check()`` is a pure
    function of a proposal plus the current book and snapshot.
    """

    symbol: str
    side: Side
    usd_notional: float
    quote: FillQuote | None  # None when quoting itself failed
    source: Literal["model", "stop_loss"] = "model"
    confidence: float = 0.0
    reasoning: str = ""


@dataclass(frozen=True, slots=True)
class RiskVerdict:
    """Code decides, the model proposes.

    ``approved_usd`` may be *less* than the proposal when a rule clamps rather
    than rejects (e.g. position sizing). A clamp is still an approval, and the
    clamp is recorded in ``notes`` so it shows up in the decision log.
    """

    approved: bool
    approved_usd: float
    rule: str | None = None  # the rule that fired, e.g. "max_position_pct"
    reason: str | None = None  # human-readable, fed back to the model next tick
    notes: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Model I/O — owned by brain.py, consumed by loop.py
# ---------------------------------------------------------------------------


class Action(BaseModel):
    """One decision, for one coin."""

    action: Literal["BUY", "SELL", "HOLD"]
    symbol: str
    size_usd: float = Field(
        description="USD notional to trade. Exactly 0.0 for HOLD.",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(
        description=(
            "Why. Name the specific evidence — which stream, which number. "
            "This field is the point of the whole log."
        )
    )


class TradeDecision(BaseModel):
    """The model's full output for one slow tick."""

    market_read: str = Field(
        description="One paragraph on what you think is happening right now."
    )
    actions: list[Action] = Field(
        description="Exactly one entry per configured coin, in any order."
    )


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """One row in ``decisions.jsonl`` — the model's proposal, the code's verdict,
    and what it cost. Rejections are recorded too; that is the point."""

    ts: float
    market_read: str
    actions: tuple[Action, ...]
    verdicts: tuple[RiskVerdict, ...]  # parallel to actions
    fills: tuple[Fill, ...]
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    model: str
    effort: str
    dry_run: bool = False
    thinking: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """Everything the model is shown for one coin, assembled by ``loop.py`` and
    rendered by ``prompts.py``. A ``None`` field means that evidence stream was
    unavailable this tick and must be rendered as explicitly unavailable."""

    symbol: str
    snapshot: CoinSnapshot
    technicals: TechnicalBrief | None
    sentiment: SentimentBrief | None
    sentiment_unavailable_reason: str | None = None


__all__ = [
    "Action",
    "Broker",
    "Candle",
    "CoinSnapshot",
    "DecisionRecord",
    "EvidenceBundle",
    "Fill",
    "FillQuote",
    "FlowBrief",
    "MarketSnapshot",
    "PortfolioState",
    "Position",
    "PriceLadder",
    "RiskVerdict",
    "SentimentBrief",
    "Side",
    "Technicals",
    "TechnicalBrief",
    "Timeframe",
    "TopPost",
    "TradeDecision",
    "TradeProposal",
    "TxnCounts",
]
