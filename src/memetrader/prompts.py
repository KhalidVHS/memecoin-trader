"""Everything the model reads, and the cache boundary that makes it affordable.

The whole design of this module is one constraint: **prompt caching is a prefix
match**. The API hashes the exact bytes of ``tools`` -> ``system`` -> ``messages``
up to each ``cache_control`` breakpoint. One byte of drift in the prefix and the
cache silently misses — no error, no warning, just a bill several times larger.

So the split is absolute:

* ``build_system(cfg)`` returns *only* text that is byte-identical from one tick
  to the next: the role, the hard limits, how to read each evidence stream, the
  lessons from the prior project, and the output contract. The last block
  carries the ``cache_control`` breakpoint.
* ``render_user(...)`` returns *everything* that moves: prices, technicals,
  sentiment, the book, the decision log, last tick's rejections. It goes after
  the breakpoint, so it costs full price and invalidates nothing.

There is no timestamp, no price and no symbol-specific number anywhere in
``build_system``'s output, and ``tests/test_prompts.py`` asserts that by building
the blocks twice against different evidence and comparing bytes.

One number worth knowing: on Claude Opus 5 the **minimum cacheable prefix is 512
tokens**. Below that the breakpoint is ignored silently — ``cache_creation`` and
``cache_read`` both stay at zero and the marker buys nothing. The stable text
below is comfortably past that (~2k tokens), which is exactly why it is worth
keeping it verbose and keeping it frozen; if it is ever trimmed to a couple of
paragraphs, caching stops engaging at all.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .types import (
    DecisionRecord,
    EvidenceBundle,
    PortfolioState,
    RiskVerdict,
    SentimentBrief,
    Technicals,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import Config

__all__ = [
    "SYSTEM_PROMPT",
    "build_system",
    "render_user",
]

UNAVAILABLE = "UNAVAILABLE — this stream is missing this tick"

# ---------------------------------------------------------------------------
# The stable prefix. Never interpolate anything into this string.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an active trader of Solana memecoins, running a $1,000 paper book. You
wake up every 15 minutes, read the evidence, and decide what the book should
look like for the next 15 minutes. You are not an analyst writing a note. You
are the one holding the position.

# The rules of the world

These are not requests. They are enforced in code, after you answer, by a risk
module that never negotiates:

* A single position may not exceed 30% of total book value. A larger proposal is
  clamped down to the limit, not rejected — but you have then spent your turn
  asking for something you cannot have.
* A position is force-closed at -15% from its average entry. That stop fires on
  a fast tick without consulting you. Do not plan around holding through it.
* No trade smaller than $10 is executed at all. Below that, fees and gas eat the
  trade.
* Any route with more than 3% price impact is rejected outright. On a thin pool
  that is a hard ceiling on size, regardless of conviction.

Code clamps or rejects regardless of what you ask for, so proposing an illegal
size does not get you a bigger position — it gets you a wasted turn and a
rejection line in your next prompt.

Two limits that deliberately do **not** exist: there is no cap on trade
frequency, and there is no minimum hold time. Both were removed on purpose. You
may enter and exit within a single 15-minute tick if the evidence changed. You
may also sit still for a day. HOLD is always a legitimate answer and is the
correct one most of the time. Churn is not rewarded — but neither is patience
for its own sake. Trade when the evidence says to and not otherwise.

# How to read the evidence, and how much to trust it

The three streams are not equal. They are listed here in descending order of
trust, and the brief labels them so you always know which one you are leaning on.

## 1. Price and flow (DexScreener, on-chain) — the most trustworthy stream

These are actual transactions. Buy/sell counts are real money moving, and
liquidity is the pool that has to absorb your exit.

A draining pool is the single most important thing that can happen to a memecoin
position, and price alone will not tell you in time. Liquidity leaves before
price does: the LPs who know something withdraw first, the price holds for a
while on thin volume, and then it does not. If liquidity is trending down
materially while price is flat or up, that is a sell signal on its own, ahead of
anything the chart says. It also silently raises your price impact, which is how
a position you thought you could exit becomes one you cannot exit at these
levels.

Read the buy/sell ratio across windows together. m5 above 1 against an h1 below 1
is a bounce inside a distribution; both above 1 and rising is real accumulation.
Turnover (volume divided by liquidity) tells you whether the pool is being
actively traded or is a parked bag.

A price-change window printed as "n/a" was not reported for this pool — thinly
traded pools frequently have no m5 figure at all. That is not a flat 5 minutes;
it is no observation. Fall back to the longer windows rather than reading it as
a zero.

## 2. Technicals — real, but noisy at 5m

Treat the 5m frame as noise with signal in it, not as truth. The point of having
both frames is the agreement or disagreement between them:

* A 5m move **confirming** the 1h trend is a continuation trade. You can size it.
* A 5m move **fighting** the 1h trend is a different trade entirely — a fade or a
  fakeout — and deserves less size, or none.

RSI *direction* matters more than RSI *level*. "RSI 35 and rising" and "RSI 35
and falling" are opposite trades that a level-only reading calls the same thing.
Same for MACD: a cross two bars old is information, a cross thirty bars old is
history. Bars-since-cross is printed for exactly that reason.

ATR% is what makes the -15% stop sane or insane for a given coin. On a coin with
2% ATR on the hour, -15% is a genuine thesis-is-wrong level and you have room. On
a coin running 12% ATR, -15% is one ordinary candle and you will be stopped out
by noise; size down or stand aside rather than donating to variance.

Any indicator printed as "n/a" could not be computed — there was not enough candle
history. That is not a zero and it is not neutral. It means you are reading a
partial chart and should weight it accordingly.

## 3. Sentiment — attention, not polarity

What sentiment measures here is how much attention a coin is getting, not how
people feel about it. Polarity is manufactured: shill farms produce bullish text
on demand and it costs them nothing. Polarity is printed but explicitly flagged
low-trust, and it should almost never be the reason for a trade.

What survives scrutiny is velocity and breadth. Mention velocity accelerating
against its own 7-day baseline is a real change in attention. Unique contributors
is breadth — how many distinct people, not how many posts.

A low contributor_to_post_ratio means a handful of accounts are producing most of
the posts. That is a warning sign, not enthusiasm: it is the signature of a
coordinated push, and the accounts doing the pushing are usually the ones already
holding and looking for exit liquidity. Read it as a reason to be more careful,
never as confirmation.

A sentiment brief marked UNAVAILABLE means **we could not find out**. That is not
the same claim as "nobody is talking about it" — it is an absence of information,
not the information that there is nothing. It should reduce your confidence in
anything that would have leaned on sentiment, not be quietly treated as neutral.

# What the last project taught us

These are empirical findings from a prior system that traded this way for
months. They are not style preferences, and they are not up for relitigation
tick by tick:

1. **Trailing stops were catastrophic.** They produced 269 exits at roughly
   -2.26% each — death by a thousand cuts, with every single exit locally
   defensible. Ordinary volatility kept tapping the trail and closing positions
   that were fine. A wide, fixed stop measured from entry is the correct
   structure, which is why the -15% here is fixed and not trailing.

2. **Signal concentration beats diversification.** Cutting from eight entry
   signals to one or two turned a -16.62% run into +5.39%. More signals did not
   mean more confirmation; it meant that at any moment something was flashing, so
   something was always a reason to trade. Conviction from one or two strong,
   agreeing signals beats a weak consensus across many. If your reasoning has to
   list five mediocre reasons, that is a HOLD.

3. **Never DCA a loser.** Adding to a losing position was reliably destructive.
   If a position is down and you still like it, the answer is to hold it, not to
   buy more of it at a better price. The better price is the market disagreeing
   with you.

4. **Nearly every added layer of sophistication measurably hurt returns.** The
   simpler read of the evidence outperformed the clever one, consistently. When
   you find yourself constructing a multi-step story about why a bad-looking
   setup is actually good, that is the failure mode, not insight.

# Output contract

Return exactly one action per configured coin, every single tick — including the
coins you are doing nothing with. A missing coin is an error; an explicit HOLD is
an answer.

* ``size_usd`` is exactly 0.0 for HOLD. Not a small number. Zero.
* ``size_usd`` for BUY is the USD notional to spend. For SELL it is the USD
  notional of the position to close (use the current position value to close it
  fully).
* ``confidence`` is 0..1 and should actually vary. If everything is 0.8, it
  carries no information.
* ``reasoning`` must name the *specific* evidence: which stream, which number.
  "Flow: m5 buy/sell 1.9 vs h1 0.8, liquidity -6.2% — distribution" is useful.
  "Momentum looks weak" is not. This field is the entire audit trail for the
  trade. When a trade loses money, this is the only record of why it was made,
  and a vague one makes the loss unlearnable.
* ``market_read`` is one paragraph on what you think is actually happening right
  now across the three coins — the view the individual actions follow from.
"""


def _limits_block(cfg: Config) -> str:
    """The configured numbers, restated from config so the prompt cannot drift
    from what risk.py actually enforces.

    This is *stable within a run* but changes if the user edits ``config.toml``,
    so it lives in its own block after the frozen text. That way editing a risk
    number re-caches only this block's suffix rather than the whole prompt.
    """
    symbols = ", ".join(cfg.symbols)
    return (
        "# This run's configuration\n"
        "\n"
        f"Coins under management (one action required for each, every tick): {symbols}\n"
        f"Starting book: ${cfg.starting_cash_usd:,.2f}\n"
        "\n"
        "Enforced limits, as configured:\n"
        f"* max position: {cfg.risk.max_position_pct * 100:.0f}% of total book value\n"
        f"* forced stop-loss: -{cfg.risk.stop_loss_pct * 100:.0f}% from average entry\n"
        f"* minimum trade: ${cfg.risk.min_trade_usd:,.2f}\n"
        f"* maximum price impact: {cfg.risk.max_price_impact_pct:.1f}%\n"
        f"* minimum pool liquidity to trade: ${cfg.risk.min_liquidity_usd:,.0f}\n"
        f"* decision cadence: every {cfg.cadence.slow_tick_seconds // 60} minutes; "
        f"stops are checked every {cfg.cadence.fast_tick_seconds} seconds without you\n"
    )


def build_system(cfg: Config) -> list[dict[str, Any]]:
    """The system blocks, with the cache breakpoint on the last stable block.

    Two blocks, both cached, in increasing order of volatility:

    0. ``SYSTEM_PROMPT`` — frozen forever. Survives even a config edit.
    1. the configured limits — frozen for the life of a run.

    The breakpoint on block 0 means a config change only re-caches block 1; the
    breakpoint on block 1 is the one that matters tick to tick, and it is the one
    the tests assert on. Two of the four available breakpoints, which leaves room
    for ``messages`` caching later if the loop ever becomes multi-turn.
    """
    return [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": _limits_block(cfg),
            "cache_control": {"type": "ephemeral"},
        },
    ]


# ---------------------------------------------------------------------------
# Volatile rendering helpers
#
# Every one of these renders ``None`` as "n/a". Substituting 0.0 for a missing
# indicator is how you get a confidently wrong trade: "RSI 0" reads as maximally
# oversold when what actually happened is that we had eleven candles.
# ---------------------------------------------------------------------------

NA = "n/a"


def _f(value: float | int | None, fmt: str = ".2f", suffix: str = "") -> str:
    if value is None:
        return NA
    return f"{value:{fmt}}{suffix}"


def _pct(value: float | None, fmt: str = "+.2f") -> str:
    if value is None:
        return NA
    return f"{value:{fmt}}%"


def _usd(value: float | None, fmt: str = ",.2f") -> str:
    if value is None:
        return NA
    return f"${value:{fmt}}"


def _level(value: float | None) -> str:
    """A price-like number. Memecoin prices span ten orders of magnitude, so
    neither fixed decimals nor bare ``%g`` works: ``%g`` prints ``2.134e-05``,
    which nobody reads as a price, and ``.8f`` prints ``2.41000000``."""
    if value is None:
        return NA
    if value == 0:
        return "0"
    if abs(value) >= 1:
        return f"{value:,.6g}"
    if abs(value) < 1e-9:  # genuinely tiny — exponent is the honest rendering
        return f"{value:.4g}"
    return f"{value:.12f}".rstrip("0")


def _price(value: float | None) -> str:
    if value is None:
        return NA
    return f"${_level(value)}"


def _ratio(value: float | None) -> str:
    if value is None:
        return NA
    if value == float("inf"):
        return "inf (no sells)"
    return f"{value:.2f}"


def _flag(value: bool | None, yes: str, no: str) -> str:
    if value is None:
        return NA
    return yes if value else no


def _age(seconds: float | None) -> str:
    if seconds is None:
        return NA
    if seconds < 0:
        seconds = 0.0
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _clock(ts: float | None) -> str:
    if ts is None:
        return NA
    return time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(ts))


def _technicals_lines(label: str, t: Technicals) -> list[str]:
    return [
        f"  {label} ({t.candles_used} candles)",
        f"    rsi14      {_f(t.rsi14, '.1f')} "
        f"({_flag(t.rsi14_rising, 'rising', 'falling')})",
        f"    ema        9 {_level(t.ema9)} / 21 {_level(t.ema21)} — "
        f"{_flag(t.ema9_above_ema21, '9 above 21 (bull)', '9 below 21 (bear)')}; "
        f"price vs ema9 {_pct(t.pct_from_ema9)}, vs ema21 {_pct(t.pct_from_ema21)}",
        f"    macd       line {_f(t.macd_line, '.8g')} signal {_f(t.macd_signal, '.8g')} "
        f"hist {_f(t.macd_hist, '.8g')}; cross {t.macd_cross or NA} "
        f"({_f(t.bars_since_cross, 'd')} bars ago)",
        f"    bollinger  %b {_f(t.bb_percent_b, '.2f')} bandwidth "
        f"{_f(t.bb_bandwidth, '.2f', '%')} "
        f"({_flag(t.bb_expanding, 'expanding', 'contracting')})",
        f"    atr14      {_f(t.atr14_pct, '.2f', '%')} of price"
        f"   <- sanity-check the -15% stop against this",
        f"    volume     {_f(t.volume_ratio_20, '.2f')} x the 20-period mean",
        f"    swing      {_pct(t.pct_from_swing_high)} from high, "
        f"{_pct(t.pct_from_swing_low)} from low",
    ]


def _sentiment_lines(s: SentimentBrief, now: float) -> list[str]:
    lines = [
        f"  SENTIMENT (source {s.source}, {_age(now - s.ts)} old)"
        + (f" [PARTIAL: {s.degraded_reason}]" if s.degraded_reason else ""),
        # n/a here means the hour was never indexed, not that it was silent.
        # The distinction decides trades, so the line says which it is rather
        # than leaving the model to read 0.00 as quiet.
        f"    velocity   {_f(s.mention_velocity_1h, '.2f')} mentions/h over 1h "
        f"vs {_f(s.mention_velocity_24h, '.2f')}/h over the observed window "
        f"(z-score vs 7d baseline {_f(s.mention_zscore_7d, '+.2f')})"
        + (
            "  <- n/a = not yet indexed by the source, NOT zero attention"
            if s.mention_velocity_1h is None or s.mention_velocity_24h is None
            else ""
        ),
        f"    breadth    {s.unique_contributors_24h} unique contributors in 24h; "
        f"contributor/post ratio {_f(s.contributor_to_post_ratio, '.2f')}"
        f"  <- low means few accounts posting a lot; treat as a warning",
        f"    polarity   {_f(s.polarity, '+.2f')}  [LOW TRUST — manufactured; "
        f"weigh attention, not mood]",
    ]
    if s.top_posts:
        lines.append("    top posts:")
        for p in s.top_posts[:5]:
            title = p.title if len(p.title) <= 110 else p.title[:107] + "..."
            lines.append(
                f"      [{p.score:>5}] r/{p.subreddit} {p.age_hours:.1f}h — {title}"
            )
    return lines


def _coin_section(bundle: EvidenceBundle, now: float) -> list[str]:
    snap = bundle.snapshot
    tech = bundle.technicals
    lines: list[str] = [f"--- {bundle.symbol} ---"]

    if snap.degraded:
        lines.append(
            f"  !! DEGRADED SNAPSHOT: {snap.degraded_reason or 'reason not recorded'}"
        )

    liq_trend = tech.flow.liquidity_trend_pct if tech is not None else None
    lines += [
        "  PRICE/FLOW (DexScreener, on-chain) — highest-trust stream",
        f"    price      {_price(snap.price_usd)}   fdv {_usd(snap.fdv_usd, ',.0f')}"
        f"   dex {snap.dex_id}",
        f"    liquidity  {_usd(snap.liquidity_usd, ',.0f')}  "
        f"trend vs last tick {_pct(liq_trend)}"
        f"   <- a draining pool outranks everything else here",
        f"    change     m5 {_pct(snap.price_change.m5)}  "
        f"h1 {_pct(snap.price_change.h1)}  "
        f"h6 {_pct(snap.price_change.h6)}  "
        f"h24 {_pct(snap.price_change.h24)}",
        f"    volume     1h {_usd(snap.volume_1h_usd, ',.0f')}  "
        f"24h {_usd(snap.volume_24h_usd, ',.0f')}",
        f"    txns m5    {snap.txns_m5.buys}B / {snap.txns_m5.sells}S  "
        f"ratio {_ratio(snap.txns_m5.ratio)}",
        f"    txns h1    {snap.txns_h1.buys}B / {snap.txns_h1.sells}S  "
        f"ratio {_ratio(snap.txns_h1.ratio)}",
        f"    txns h24   {snap.txns_h24.buys}B / {snap.txns_h24.sells}S  "
        f"ratio {_ratio(snap.txns_h24.ratio)}",
    ]
    if tech is not None:
        lines.append(
            f"    turnover   1h {tech.flow.turnover_1h:.3f}x liquidity  "
            f"24h {tech.flow.turnover_24h:.3f}x"
        )
    lines.append(
        f"    pool age   {_age(now - snap.pair_created_at) if snap.pair_created_at else NA}"
    )

    if tech is None:
        lines.append(f"  TECHNICALS: {UNAVAILABLE}")
    else:
        lines.append("  TECHNICALS — noisy at 5m; the 5m/1h agreement is the point")
        lines += _technicals_lines("5m", tech.m5)
        lines += _technicals_lines("1h", tech.h1)

    if bundle.sentiment is None:
        reason = bundle.sentiment_unavailable_reason or "reason not recorded"
        lines.append(f"  SENTIMENT: {UNAVAILABLE} ({reason})")
        lines.append(
            "    This means we could not find out — NOT that nobody is talking. "
            "Lower confidence on anything that would lean on attention."
        )
    else:
        lines += _sentiment_lines(bundle.sentiment, now)

    return lines


def _portfolio_section(
    cfg: Config, portfolio: PortfolioState, evidence: dict[str, EvidenceBundle]
) -> list[str]:
    lines = [
        "=== PORTFOLIO ===",
        f"  cash          {_usd(portfolio.cash_usd)}",
        f"  total value   {_usd(portfolio.total_value_usd)} "
        f"(started {_usd(portfolio.starting_cash_usd)})",
        f"  total return  {_pct(portfolio.total_return_pct)}",
        f"  realized P&L  {_usd(portfolio.realized_pnl_usd)} to date   "
        f"unrealized {_usd(portfolio.unrealized_pnl_usd)}",
        f"  costs paid    fees {_usd(portfolio.fees_paid_usd)}  "
        f"gas {_usd(portfolio.gas_paid_usd)}",
        f"  max position  {_usd(portfolio.total_value_usd * cfg.risk.max_position_pct)} "
        f"({cfg.risk.max_position_pct * 100:.0f}% of book) — anything larger is clamped",
    ]

    if not portfolio.positions:
        lines.append("  positions: none — the book is entirely cash")
        return lines

    lines.append("  positions:")
    for symbol in sorted(portfolio.positions):
        pos = portfolio.positions[symbol]
        mark = portfolio.marks.get(symbol)
        if mark is None and symbol in evidence:
            mark = evidence[symbol].snapshot.price_usd
        value = portfolio.position_values_usd.get(symbol)
        stop_price = pos.avg_entry_price_usd * (1.0 - cfg.risk.stop_loss_pct)
        if mark is None:
            pnl_usd = pnl_pct = None
            to_stop = None
        else:
            pnl_usd = pos.unrealized_pnl_usd(mark)
            pnl_pct = pos.unrealized_pnl_pct(mark)
            to_stop = 100.0 * (stop_price - mark) / mark if mark else None
        lines += [
            f"    {symbol}",
            f"      qty {pos.quantity:,.4f}  entry {_price(pos.avg_entry_price_usd)}  "
            f"cost basis {_usd(pos.cost_basis_usd)}",
            f"      mark {_price(mark)}  value {_usd(value)}  "
            f"age {_age(portfolio.ts - pos.opened_at)}",
            f"      unrealized {_usd(pnl_usd)} ({_pct(pnl_pct)})",
            f"      forced stop at {_price(stop_price)} "
            f"(-{cfg.risk.stop_loss_pct * 100:.0f}% from entry) — "
            f"{_pct(to_stop)} from here",
        ]
    return lines


def _decision_line(
    record: DecisionRecord, now: float, portfolio: PortfolioState
) -> str:
    """One decision compressed to exactly one line: what was done and how it went.

    Bounded by construction — the caller slices to ``cfg.prompt.decision_history``
    before calling — so the prompt cannot grow without limit across a multi-day
    run.
    """
    parts: list[str] = []
    holds = 0
    for i, action in enumerate(record.actions):
        if action.action == "HOLD":
            holds += 1
            continue
        seg = f"{action.action} {action.symbol} {_usd(action.size_usd, ',.0f')}"
        verdict = record.verdicts[i] if i < len(record.verdicts) else None
        if verdict is not None and not verdict.approved:
            seg += f" REJECTED[{verdict.rule or 'unknown rule'}]"
        elif verdict is not None and verdict.approved_usd < action.size_usd - 0.01:
            seg += f" clamped->{_usd(verdict.approved_usd, ',.0f')}"
        fill = next((f for f in record.fills if f.symbol == action.symbol), None)
        if fill is not None:
            if fill.failed:
                seg += " FILL FAILED"
            elif action.action == "SELL":
                seg += f" filled {_usd(fill.filled_usd, ',.0f')}, realized {_usd(fill.realized_pnl_usd)}"
            else:
                seg += f" filled {_usd(fill.filled_usd, ',.0f')} @ {_price(fill.price_usd)}"
                pos = portfolio.positions.get(action.symbol)
                mark = portfolio.marks.get(action.symbol)
                if pos is not None and mark is not None:
                    seg += f", now {_pct(pos.unrealized_pnl_pct(mark))}"
                else:
                    seg += ", position since closed"
        parts.append(seg)

    if not parts:
        parts.append(f"HOLD x{holds}" if holds else "no actions")
    elif holds:
        parts.append(f"HOLD x{holds}")
    return f"  t-{_age(now - record.ts):>5}  " + "; ".join(parts)


def _history_section(
    cfg: Config,
    history: Sequence[DecisionRecord],
    now: float,
    portfolio: PortfolioState,
) -> list[str]:
    limit = max(0, cfg.prompt.decision_history)
    recent = list(history)[-limit:] if limit else []
    header = f"=== YOUR LAST {limit} DECISIONS (newest first) ==="
    if not recent:
        return [header, "  none yet — this is the first decision of the run"]
    return [header] + [
        _decision_line(r, now, portfolio) for r in reversed(recent)
    ]


def _rejections_section(rejections: Sequence[RiskVerdict]) -> list[str]:
    header = "=== RISK VERDICTS FROM LAST TICK ==="
    if not rejections:
        return [header, "  none — nothing was clamped or rejected last tick"]
    lines = [
        header,
        "  These fired in code after your last answer. Do not re-propose them.",
    ]
    for v in rejections:
        status = "REJECTED" if not v.approved else "CLAMPED"
        lines.append(
            f"  {status} by rule `{v.rule or 'unknown'}`: "
            f"{v.reason or 'no reason recorded'} "
            f"(approved {_usd(v.approved_usd, ',.2f')})"
        )
        for note in v.notes:
            lines.append(f"      note: {note}")
    return lines


def render_user(
    cfg: Config,
    evidence: dict[str, EvidenceBundle],
    portfolio: PortfolioState,
    history: Sequence[DecisionRecord],
    rejections: Sequence[RiskVerdict],
) -> str:
    """The volatile brief — everything that moves, after the cache breakpoint.

    Nothing in here is cached and nothing in here needs to be. It is deliberately
    compact and labelled by stream, so that when a trade goes wrong the decision
    log shows which stream drove it.
    """
    now = portfolio.ts or time.time()
    lines: list[str] = [
        f"=== TICK {_clock(now)} ===",
        "",
        "=== EVIDENCE ===",
    ]

    # Iterate in configured order so the coins always appear in the same order,
    # then append anything unexpected rather than silently dropping it.
    ordered = [s for s in cfg.symbols if s in evidence]
    ordered += [s for s in evidence if s not in set(ordered)]
    for symbol in ordered:
        lines += _coin_section(evidence[symbol], now)
        lines.append("")
    for symbol in cfg.symbols:
        if symbol not in evidence:
            lines.append(f"--- {symbol} ---")
            lines.append(f"  ALL EVIDENCE: {UNAVAILABLE}")
            lines.append("  Default to HOLD unless you already hold it.")
            lines.append("")

    lines += _portfolio_section(cfg, portfolio, evidence)
    lines.append("")
    lines += _history_section(cfg, history, now, portfolio)
    lines.append("")
    lines += _rejections_section(rejections)
    lines.append("")
    lines.append(
        f"Decide now. Exactly one action for each of: {', '.join(cfg.symbols)}. "
        "size_usd is 0.0 for HOLD. Name the specific number you traded on."
    )
    return "\n".join(lines)
