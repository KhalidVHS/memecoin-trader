"""What the operator reads, and the arithmetic behind it — kept apart.

This module is the ``memetrader report`` command. It reads the persisted record
of a run and says what happened. It computes nothing a trading decision depends
on and it makes no network call: ``report`` must work offline, after the fact,
on a machine that can no longer reach Jupiter.

Four audit findings shaped the rewrite, and each one is a structural property of
this file rather than a line of copy:

**C1 — no unfounded performance claims.** The headline is the *sample*, not the
return. A 12-hour run over three correlated memecoins with five fills and one
round trip cannot support an alpha claim, and the audit's own acceptance gate is
"8-12 weeks and roughly 500 sufficiently independent round trips". So
:class:`Sample` is computed first, rendered first, and carries the caveat as
data — :attr:`Sample.caveat` is a string a test can assert on, not a sentence
buried in a Rich markup literal. The return number is shown underneath it,
beside the do-nothing counterfactual, because "up $12" means nothing until you
know what the alternative was.

**C8 — marks are not prices.** Every valuation line carries its ``basis``
(``route``/``mid``/``estimate``/``unavailable``) and the haircut that was
applied. A position that cannot be marked renders as ``UNMARKABLE``; it is never
carried at cost basis (the exact defect C8 named) and never quietly dropped from
the table. When any position is unmarkable, ``PortfolioState.total_value_usd`` is
``None`` and the total renders as unavailable rather than as a sum of whatever
happened to be priceable.

**C11 — the record must be auditable.** Torn and corrupt ledger lines, open
(unreconciled) intents, failed fills and the gas they burned, and decisions whose
intents never reached a terminal fill are all reasons the numbers below may be
wrong. They are therefore rendered **above** the numbers. A footnote is where you
put something you hope nobody reads.

**Costs are decomposed as far as the record allows, and no further.** Realised
P&L, gas (including gas on *failed* swaps, which cost full gas and move no
inventory), and slippage versus the bound quote are separate lines.
``pool_fees_usd`` is 0.00 on routed quotes by construction — the AMM fee is
embedded in the router's ``outAmount`` — and :attr:`CostBreakdown.
pool_fee_decomposed` is ``False`` to say so. Zero *reported* pool fee is not zero
economic trading cost, and the audit calls that confusion out by name.

**Missing is never zero.** ``None`` renders as ``n/a`` or ``unavailable``,
everywhere, without exception. A blank cell and a 0.0% are both claims.

Structure: frozen dataclasses built by pure functions (:func:`build_sample`,
:func:`build_run_summary`, :func:`build_trade_stats`, :func:`build_cost_breakdown`,
:func:`build_model_spend`, :func:`build_integrity`), a disk reader
(:func:`collect`) that is the only part that touches the filesystem, and
``render_*`` functions that take a :class:`~rich.console.Console`. Tests assert
on the dataclasses; nothing has to parse terminal output to check a number.

**Deliberately not computed here**, because computing them would itself be a C1
violation: Sharpe, Sortino, Calmar, maximum drawdown and their confidence
intervals. Every one of those needs an equity curve sampled at fixed intervals,
and the record contains fills, not a curve — the fast tick marks the book but
does not persist the mark. Deriving a Sharpe from five fills would produce a
number with no standard error that somebody would quote. See
:attr:`Sample.caveat` for what is said instead.
"""

from __future__ import annotations

import contextlib
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import journal
from .config import Config, ModelConfig
from .http import BreakerState, BreakerStatus
from .journal import OpenIntent, RowKind, ScanResult
from .portfolio import failed_attempts, landed_exits, stop_price_usd
from .types import (
    DecisionRecord,
    EvidenceBundle,
    ExecutionMode,
    Fill,
    OrderIntent,
    OrderState,
    PortfolioState,
    Side,
    TechnicalBrief,
    Technicals,
)

__all__ = [
    "ADEQUATE_ROUND_TRIPS",
    "ADEQUATE_RUN_HOURS",
    "CostBreakdown",
    "DecisionGap",
    "DecisionSummary",
    "IntegrityReport",
    "LedgerFacts",
    "ModelSpend",
    "PositionLine",
    "Report",
    "RunSummary",
    "Sample",
    "TradeStats",
    "build_cost_breakdown",
    "build_integrity",
    "build_model_spend",
    "build_position_lines",
    "build_report",
    "build_run_summary",
    "build_sample",
    "build_trade_stats",
    "collect",
    "console",
    "load_report",
    "render_book",
    "render_costs",
    "render_evidence",
    "render_fills",
    "render_integrity",
    "render_report",
    "render_spend",
    "render_summary",
    "render_trades",
    "summarize_decision",
]


def _utf8(stream: object) -> None:
    """Force UTF-8 on a stdio stream.

    On Windows, Python picks the ANSI code page (cp1252 here) for a *redirected*
    stream, and this module renders arrows, em dashes and box characters —
    ``memetrader status | tail`` died with UnicodeEncodeError on a '↓' while the
    same command in a terminal was fine. A display-layer encoding detail must
    never be able to take down a command that reads the book.

    ``errors="replace"`` is the belt to the UTF-8 braces: if a stream cannot be
    reconfigured at all, an unrenderable glyph degrades to '?' instead of an
    exception.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:  # not a TextIOWrapper (pytest capture, a pipe shim)
        return
    # Already detached, or not reconfigurable: either way the stream is still
    # usable, just not in UTF-8, and `errors="replace"` was the fallback anyway.
    with contextlib.suppress(ValueError, OSError):
        reconfigure(encoding="utf-8", errors="replace")


_utf8(sys.stdout)
_utf8(sys.stderr)

console = Console()

#: The audit's prospective-evidence gate, §10: "at least 8-12 weeks and roughly
#: 500 sufficiently independent round trips, or longer if effective sample size
#: is smaller." These are not thresholds this system expects to clear; they are
#: here so the report can state how far short of them it is by a factor, which
#: is harder to misread than an adjective.
ADEQUATE_ROUND_TRIPS = 500
ADEQUATE_RUN_HOURS = 8 * 7 * 24.0

#: A high-effort thinking trace runs to thousands of tokens and would bury the
#: other nine decisions. The full text is never discarded — it stays in the
#: ledger row, which is where to read one decision in depth.
_THINKING_PREVIEW_CHARS = 240

#: What an absent number renders as. One constant so that every table, panel and
#: test agrees, and so that grepping for the string finds every site.
NA = "n/a"
UNAVAILABLE = "unavailable"


# ---------------------------------------------------------------------------
# Formatting. None is never zero.
# ---------------------------------------------------------------------------


def _usd(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return NA
    return f"${value:+,.2f}" if signed else f"${value:,.2f}"


def _pct(value: float | None, digits: int = 2) -> Text:
    """A percent, coloured by sign. ``None`` renders ``n/a`` — never 0.0%,
    because "not reported" and "flat" are different claims."""
    if value is None:
        return Text(NA, style="dim")
    style = "green" if value > 0 else "red" if value < 0 else "white"
    return Text(f"{value:+.{digits}f}%", style=style)


def _num(value: float | int | None, fmt: str = ",.2f") -> str:
    return NA if value is None else format(value, fmt)


def _unit(value: float | int | None, suffix: str, fmt: str = ",.2f") -> str:
    """A number with its unit, or the bare "n/a" — never the two glued together.

    Appending a suffix to :func:`_num` produces strings like "n/ax" and
    "n/a/h", which read as values and get quoted back as if they were. The
    unit belongs to the number; when there is no number there is no unit.
    """
    return NA if value is None else f"{format(value, fmt)}{suffix}"


def _price(value: float | None) -> str:
    """Memecoin prices span nine orders of magnitude; a fixed precision either
    prints $0.00 for BONK or a wall of zeros for WIF."""
    if value is None:
        return NA
    if value == 0:
        return "$0"
    if value >= 0.01:
        return f"${value:,.4f}"
    return f"${value:.3e}".replace("e-0", "e-")


def _age(seconds: float | None) -> str:
    if seconds is None:
        return NA
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _mean(values: Sequence[float]) -> float | None:
    """The mean, or ``None`` for an empty sample. Not 0.0.

    ``statistics.mean(())`` raises, and every call site here would otherwise
    have to guard — and one of them would eventually guard with ``or 0.0``.
    """
    return statistics.fmean(values) if values else None


# ---------------------------------------------------------------------------
# Sample size — computed first, rendered first. Audit C1.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Sample:
    """How much evidence exists. The number that governs every other number.

    The audit's C1 finding is not that the reported return was wrong; it is that
    a return computed over one round trip was presented as if it described a
    strategy. This type exists so the denominator is impossible to skip: it is
    built before the P&L, rendered above it, and its :attr:`caveat` is an
    assertable string rather than a flourish in a render function.
    """

    decisions: int
    fills: int
    landed_fills: int
    failed_fills: int
    closed_round_trips: int
    first_ts: float | None
    last_ts: float | None

    @property
    def elapsed_hours(self) -> float | None:
        """Wall-clock span of the record. ``None`` when fewer than two events
        exist — one timestamp is an instant, not a duration, and reporting 0.0
        hours would make a fresh run look like a long flat one."""
        if self.first_ts is None or self.last_ts is None:
            return None
        span = self.last_ts - self.first_ts
        return span / 3600.0 if span > 0 else None

    @property
    def round_trip_shortfall(self) -> float | None:
        """How many times short of the audit's gate this sample is."""
        if self.closed_round_trips <= 0:
            return None
        return ADEQUATE_ROUND_TRIPS / self.closed_round_trips

    @property
    def adequate(self) -> bool:
        """Always false at any plausible size of this system's record, and
        computed rather than hard-coded so that the day it becomes true, it
        becomes true for a reason."""
        hours = self.elapsed_hours
        return (
            self.closed_round_trips >= ADEQUATE_ROUND_TRIPS
            and hours is not None
            and hours >= ADEQUATE_RUN_HOURS
        )

    @property
    def caveat(self) -> str:
        """The sentence that must accompany every performance number here."""
        if self.adequate:
            return (
                f"{self.closed_round_trips} closed round trips over "
                f"{self.elapsed_hours:.0f}h meets the audit's minimum prospective "
                "sample. Statistical tests are still required before any alpha claim."
            )
        span = (
            f"{self.elapsed_hours:.1f}h"
            if self.elapsed_hours is not None
            else "no elapsed time"
        )
        return (
            f"NO ALPHA CLAIM IS SUPPORTABLE AT THIS SAMPLE: {self.closed_round_trips} "
            f"closed round trip(s), {self.landed_fills} landed fill(s), "
            f"{self.decisions} decision(s) over {span}, across correlated "
            "memecoins. The audit's prospective gate is ~"
            f"{ADEQUATE_ROUND_TRIPS} independent round trips over "
            f"{ADEQUATE_RUN_HOURS / 24 / 7:.0f}+ weeks. Every figure below is a "
            "description of what this record contains, not an estimate of "
            "expected value."
        )


def build_sample(
    *,
    fills: Sequence[Fill],
    decisions: int,
    now: float,
) -> Sample:
    """Count the evidence. ``now`` extends the span so an idle run still ages.

    A round trip is counted as a *landed SELL*, via ``portfolio.landed_exits``
    rather than a local predicate, because that function is the one definition of
    "an exit occurred" — audit §11's "failed stop reported as exit" row exists
    because there used to be two.
    """
    timestamps = [f.ts for f in fills]
    first = min(timestamps) if timestamps else None
    last = max([*timestamps, now]) if timestamps else None
    return Sample(
        decisions=decisions,
        fills=len(fills),
        landed_fills=len([f for f in fills if not f.failed]),
        failed_fills=len(failed_attempts(fills)),
        closed_round_trips=len(landed_exits(fills)),
        first_ts=first,
        last_ts=last,
    )


# ---------------------------------------------------------------------------
# The book. Audit C8.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PositionLine:
    """One position, with the provenance of its valuation attached.

    ``basis`` and ``haircut_pct`` travel with the number because the audit's C8
    complaint was that a mark was rendered as though it were a price. A
    ``mid``-basis mark at a 2% haircut and a ``route``-basis mark at zero are
    different claims about what the position could be sold for, and a table that
    prints only the dollars cannot tell you which you are looking at.
    """

    symbol: str
    quantity: float
    avg_entry_price_usd: float
    cost_basis_usd: float
    mark_price_usd: float | None
    basis: str
    haircut_pct: float
    mark_reason: str | None
    value_usd: float | None
    unrealized_pnl_usd: float | None
    unrealized_pnl_pct: float | None
    stop_price_usd: float | None
    age_seconds: float

    @property
    def markable(self) -> bool:
        return self.mark_price_usd is not None and self.basis != "unavailable"

    @property
    def executable_basis(self) -> bool:
        """Only a sell route estimates liquidation value. Everything else is a
        display price wearing a dollar sign."""
        return self.basis == "route"


def build_position_lines(
    state: PortfolioState, *, stop_loss_pct: float
) -> tuple[PositionLine, ...]:
    """One line per held position, sorted by symbol. Nothing is dropped.

    An unmarkable position gets a line with ``mark_price_usd=None`` and
    ``basis="unavailable"``. It is deliberately *not* filtered out and
    deliberately *not* valued at cost: both of those were the C8 defect, one by
    omission and one by fabrication. A reader must see that the book contains
    something nobody can price.
    """
    lines: list[PositionLine] = []
    for symbol in sorted(state.positions):
        position = state.positions[symbol]
        mark = state.marks.get(symbol)
        price = mark.price_usd if mark is not None and mark.usable else None
        lines.append(
            PositionLine(
                symbol=symbol,
                quantity=position.quantity,
                avg_entry_price_usd=position.avg_entry_price_usd,
                cost_basis_usd=position.cost_basis_usd,
                mark_price_usd=price,
                basis=mark.basis if mark is not None else "unavailable",
                haircut_pct=mark.haircut_pct if mark is not None else 0.0,
                mark_reason=mark.reason if mark is not None else "no mark supplied",
                # Straight from the marked state rather than recomputed, so the
                # report cannot disagree with the thing risk.py acted on.
                value_usd=state.position_values_usd.get(symbol),
                unrealized_pnl_usd=position.unrealized_pnl_usd(price),
                unrealized_pnl_pct=position.unrealized_pnl_pct(price),
                stop_price_usd=stop_price_usd(position, stop_loss_pct),
                age_seconds=position.age_seconds(state.ts),
            )
        )
    return tuple(lines)


@dataclass(frozen=True, slots=True)
class RunSummary:
    """The book and what it is worth, or an honest statement that we cannot say.

    ``total_value_usd`` is whatever ``PortfolioState`` said, including ``None``.
    This type never repairs it — it propagates it — and ``total_return_pct`` is
    ``None`` in lockstep, because a percentage of an unknown total is a fiction
    with a decimal point.
    """

    marked_at: float
    starting_cash_usd: float
    cash_usd: float
    total_value_usd: float | None
    total_return_pct: float | None
    realized_pnl_usd: float
    unrealized_pnl_usd: float | None
    gross_exposure_usd: float | None
    positions: tuple[PositionLine, ...]
    unmarkable: tuple[str, ...]
    sample: Sample
    mode: ExecutionMode

    @property
    def value_available(self) -> bool:
        return self.total_value_usd is not None

    @property
    def cash_counterfactual_usd(self) -> float:
        """What the book would be worth having done nothing.

        In a paper book with no cash yield this is exactly the starting cash, and
        that is the point: the entire measured contribution of the strategy is
        :attr:`excess_vs_cash_usd`. Stated explicitly because "up $12" and "$12
        better than sitting in cash" are the same number here and are routinely
        confused when they are not.
        """
        return self.starting_cash_usd

    @property
    def excess_vs_cash_usd(self) -> float | None:
        """Strategy minus do-nothing. ``None`` when the book cannot be valued."""
        if self.total_value_usd is None:
            return None
        return self.total_value_usd - self.cash_counterfactual_usd

    @property
    def benchmark_note(self) -> str:
        """Why the *other* obvious benchmark is absent.

        Buy-and-hold over the same three coins is the comparison a reader will
        reach for, and it is not computable from this record: the ledger stores
        fills, not a price series, so there is no t0 price for a coin that was
        never bought. Inventing one from today's mid would be a backtest with
        look-ahead, which is the audit's §10 complaint in miniature.
        """
        return (
            "buy-and-hold benchmark unavailable: the record stores fills, not a "
            "point-in-time price series, so there is no honest t0 price for a coin "
            "that was never traded"
        )


def build_run_summary(
    *,
    state: PortfolioState,
    sample: Sample,
    stop_loss_pct: float,
    mode: ExecutionMode,
) -> RunSummary:
    """Assemble the book view from an already-marked ``PortfolioState``.

    Takes the marked state rather than a broker on purpose: ``report`` must be a
    pure function of the record it was handed, and a version of this that reached
    into a live broker would produce different numbers depending on when it ran.
    """
    return RunSummary(
        marked_at=state.ts,
        starting_cash_usd=state.starting_cash_usd,
        cash_usd=state.cash_usd,
        total_value_usd=state.total_value_usd,
        total_return_pct=state.total_return_pct,
        realized_pnl_usd=state.realized_pnl_usd,
        unrealized_pnl_usd=state.unrealized_pnl_usd,
        gross_exposure_usd=state.gross_exposure_usd,
        positions=build_position_lines(state, stop_loss_pct=stop_loss_pct),
        unmarkable=state.unmarkable,
        sample=sample,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Trading behaviour
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TradeStats:
    """Counts and per-trade outcomes. Descriptive statistics, nothing inferred.

    Every ratio here is ``None`` on an empty denominator. A win rate of 0.0% on
    zero closed trades is the single most misleading cell this report could
    contain, and it is the one a reader would quote.

    ``largest_win_share_pct`` is the audit's concentration check in its cheapest
    form: §10 asks that no single trade dominate, and cites a 190-trade
    memecoin study whose profitability reversed when its best three trades were
    removed. At this sample the answer is usually 100%, which is the finding.
    """

    fills: int
    landed: int
    failed: int
    buys: int
    sells: int
    closed_round_trips: int
    wins: int
    losses: int
    win_rate_pct: float | None
    avg_win_usd: float | None
    avg_loss_usd: float | None
    expectancy_usd: float | None
    profit_factor: float | None
    best_trade_usd: float | None
    worst_trade_usd: float | None
    largest_win_share_pct: float | None
    notional_traded_usd: float


def build_trade_stats(fills: Sequence[Fill]) -> TradeStats:
    """Per-trade outcomes, keyed on landed exits.

    A round trip is realised at the SELL, so realised P&L per exit is the unit of
    outcome. ``realized_pnl_usd`` on a BUY fill is 0.0 by construction and is
    excluded rather than counted as a flat trade — a purchase is not a trade
    outcome, and including them would halve every win rate.
    """
    exits = landed_exits(fills)
    outcomes = [f.realized_pnl_usd for f in exits]
    wins = [p for p in outcomes if p > 0]
    losses = [p for p in outcomes if p < 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)

    return TradeStats(
        fills=len(fills),
        landed=len([f for f in fills if not f.failed]),
        failed=len(failed_attempts(fills)),
        buys=len([f for f in fills if f.side is Side.BUY]),
        sells=len([f for f in fills if f.side is Side.SELL]),
        closed_round_trips=len(exits),
        wins=len(wins),
        losses=len(losses),
        win_rate_pct=(100.0 * len(wins) / len(outcomes)) if outcomes else None,
        avg_win_usd=_mean(wins),
        avg_loss_usd=_mean(losses),
        # Expectancy in the audit's form: P(win)E[win] - P(loss)E[|loss|], which
        # over this sample is arithmetically the mean outcome. Written as the
        # mean because the decomposed form invites reading the two halves as
        # independently estimated, and at n<10 neither is estimated at all.
        expectancy_usd=_mean(outcomes),
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else None,
        best_trade_usd=max(outcomes) if outcomes else None,
        worst_trade_usd=min(outcomes) if outcomes else None,
        largest_win_share_pct=(100.0 * max(wins) / gross_win)
        if wins and gross_win
        else None,
        notional_traded_usd=sum(f.notional_usd for f in fills),
    )


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Where the money went, decomposed as far as the record permits.

    ``pool_fee_decomposed`` is the honest half of this type. On a routed quote
    the AMM fee and the price impact are already inside the router's
    ``outAmount``, so ``pool_fees_usd`` is 0.00 — and the audit is explicit that
    "$0 explicit pool fee" must not be read as zero economic trading cost. The
    flag exists so a consumer cannot render the zero without also having been
    handed the reason it is zero.

    ``gas_on_failed_usd`` is broken out because a failed Solana swap pays full
    gas and moves no inventory. It is a pure loss that never appears in any P&L
    line and is the cost most likely to be forgotten.
    """

    realized_pnl_usd: float
    gas_paid_usd: float
    gas_on_failed_usd: float
    pool_fees_usd: float
    pool_fee_decomposed: bool
    slippage_samples: int
    mean_slippage_bps: float | None
    worst_slippage_bps: float | None
    impact_samples: int
    mean_price_impact_pct: float | None
    notional_traded_usd: float

    @property
    def cost_bps_of_notional(self) -> float | None:
        """Gas as basis points of traded notional. ``None`` on zero notional —
        a run that traded nothing has no cost ratio, it has no denominator."""
        if self.notional_traded_usd <= 0:
            return None
        return 10_000.0 * self.gas_paid_usd / self.notional_traded_usd

    @property
    def pool_fee_note(self) -> str:
        if self.pool_fee_decomposed:
            return "pool fee decomposed from the route"
        return (
            "pool fee reported as $0.00 because it is embedded in the router's "
            "outAmount — NOT because trading was free; the AMM fee and price "
            "impact are paid inside the quoted output"
        )


def build_cost_breakdown(
    fills: Sequence[Fill], *, pool_fee_decomposed: bool = False
) -> CostBreakdown:
    """Sum the costs off the fills. Failed attempts included, deliberately.

    ``realized_pnl_usd`` is summed from the fills rather than read off the
    portfolio state so that this breakdown reconciles internally against the same
    rows the slippage and gas numbers came from. The two should agree; if they
    ever do not, that disagreement is itself the finding, and hiding it behind a
    single shared source would prevent anyone noticing.
    """
    slippage = [
        f.slippage_bps_vs_quote for f in fills if f.slippage_bps_vs_quote is not None
    ]
    impact = [f.price_impact_pct for f in fills if f.price_impact_pct is not None]
    return CostBreakdown(
        realized_pnl_usd=sum(f.realized_pnl_usd for f in fills),
        gas_paid_usd=sum(f.gas_usd for f in fills),
        gas_on_failed_usd=sum(f.gas_usd for f in failed_attempts(fills)),
        pool_fees_usd=sum(f.pool_fee_usd for f in fills),
        pool_fee_decomposed=pool_fee_decomposed,
        slippage_samples=len(slippage),
        mean_slippage_bps=_mean(slippage),
        # Worst by magnitude: a fill that came in 40bps *better* than quoted is
        # not the tail this number exists to describe.
        worst_slippage_bps=(max(slippage, key=abs) if slippage else None),
        impact_samples=len(impact),
        mean_price_impact_pct=_mean(impact),
        notional_traded_usd=sum(f.notional_usd for f in fills),
    )


# ---------------------------------------------------------------------------
# Model spend
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelSpend:
    """What the LLM cost, and whether it was in the loop at all.

    ``relevant`` is false whenever ``strategy_kind != "advisory"``: audit C6 took
    the model out of trade selection, so a non-advisory run makes no calls and
    this whole section describes a historical cost, not an operating one. Showing
    "$0.00/day at this cadence" without that context reads as a claim that the
    model is cheap rather than that it is absent.
    """

    calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float
    strategy_kind: str
    relevant: bool

    @property
    def cost_per_call_usd(self) -> float | None:
        return self.cost_usd / self.calls if self.calls else None

    @property
    def cache_hit_rate(self) -> float | None:
        """Cached input over all input that could have been cached.

        The denominator includes cache *writes*, because a token you paid 1.25x
        to write is a token that was not a hit. Omitting them is how the old
        report and ``brain.Usage`` disagreed about the same run.
        """
        total = self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
        return self.cache_read_tokens / total if total else None

    def projected_daily_usd(self, slow_tick_seconds: float) -> float | None:
        per_call = self.cost_per_call_usd
        if per_call is None or slow_tick_seconds <= 0 or not self.relevant:
            return None
        return per_call * 86400.0 / slow_tick_seconds


def build_model_spend(
    decisions: Sequence[DecisionSummary],
    *,
    model: ModelConfig,
    strategy_kind: str,
) -> ModelSpend:
    """Total the token bill through ``ModelConfig.cost_usd``, the one formula.

    There were three cost formulas in this codebase before 2026-09-19 and none
    agreed: one had no cache-write price, one passed only ``input_tokens``, and
    two display call sites folded cache-creation into input and so billed it at
    1x instead of 1.25x. This calls the config's method with all four buckets and
    does no arithmetic of its own.
    """
    billed = [d for d in decisions if d.has_usage]
    input_tokens = sum(d.input_tokens for d in billed)
    output_tokens = sum(d.output_tokens for d in billed)
    cache_read = sum(d.cache_read_input_tokens for d in billed)
    cache_write = sum(d.cache_creation_input_tokens for d in billed)
    return ModelSpend(
        calls=len(billed),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        cost_usd=model.cost_usd(input_tokens, output_tokens, cache_read, cache_write),
        strategy_kind=strategy_kind,
        relevant=strategy_kind == "advisory",
    )


# ---------------------------------------------------------------------------
# Integrity. Audit C11 — rendered above the numbers, not below them.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecisionGap:
    """A decision whose intents never reached a terminal fill.

    On this paper venue that means nothing moved. On a live venue it would mean
    an order of unknown status, and the audit's §11 response is the same either
    way: reconcile before any new order. The report's job is only to make it
    impossible to read the P&L without seeing that one exists.
    """

    decision_id: str
    ts: float
    symbols: tuple[str, ...]
    intent_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """Every reason the numbers below might be wrong, in one place.

    ``ok`` being false does not mean the report is useless — a torn final line is
    expected and benign, and the rows before it are a true record. It means the
    reader must know before reading anything else.
    """

    torn_lines: tuple[str, ...]
    corrupt_lines: tuple[str, ...]
    unreadable_rows: tuple[str, ...]
    open_intents: tuple[OpenIntent, ...]
    failed_fills: tuple[Fill, ...]
    gas_burned_on_failures_usd: float
    decision_gaps: tuple[DecisionGap, ...]
    open_breaker_hosts: tuple[str, ...]
    breakers: tuple[BreakerStatus, ...]

    @property
    def ok(self) -> bool:
        return self.issue_count == 0

    @property
    def issue_count(self) -> int:
        return (
            len(self.torn_lines)
            + len(self.corrupt_lines)
            + len(self.unreadable_rows)
            + len(self.open_intents)
            + len(self.failed_fills)
            + len(self.decision_gaps)
            + len(self.open_breaker_hosts)
        )


def build_integrity(
    *,
    scans: Sequence[ScanResult] = (),
    fills: Sequence[Fill] = (),
    intents: Sequence[OrderIntent] = (),
    decisions: Sequence[DecisionSummary] = (),
    open_intents: Sequence[OpenIntent] = (),
    unreadable_rows: Sequence[str] = (),
    breakers: Sequence[BreakerStatus] = (),
) -> IntegrityReport:
    """Collate what the record cannot vouch for.

    ``open_intents`` is taken from the caller (``journal.open_intents`` or the
    broker's ``Reconciliation``) rather than re-derived, because the state
    machine's definition of "terminal" lives in ``journal`` and a second copy
    here would eventually disagree about ``EXPIRED``.

    The decision-gap join is by ``intent_id``, never by symbol. Audit
    ``prompts._decision_line``: a stop-loss and a strategy SELL on the same coin
    in the same tick are the same symbol and different orders, and matching on
    symbol attributed both to whichever was looked at first.
    """
    terminal = {f.intent_id for f in fills if f.state.is_terminal}
    by_decision: dict[str, list[OrderIntent]] = {}
    for intent in intents:
        if intent.decision_id is None or intent.intent_id in terminal:
            continue
        by_decision.setdefault(intent.decision_id, []).append(intent)

    ts_of = {d.decision_id: d.ts for d in decisions}
    gaps = tuple(
        DecisionGap(
            decision_id=decision_id,
            ts=ts_of.get(decision_id, min(i.ts for i in stranded)),
            symbols=tuple(sorted({i.symbol for i in stranded})),
            intent_ids=tuple(sorted(i.intent_id for i in stranded)),
        )
        for decision_id, stranded in sorted(by_decision.items())
    )

    failed = failed_attempts(fills)
    return IntegrityReport(
        torn_lines=tuple(
            f"{s.path} (final line, {s.torn_final_bytes} bytes discarded)"
            for s in scans
            if s.torn_final_line
        ),
        corrupt_lines=tuple(
            f"{s.path}:{line}" for s in scans for line in s.corrupt_line_numbers
        ),
        unreadable_rows=tuple(unreadable_rows),
        open_intents=tuple(open_intents),
        failed_fills=failed,
        gas_burned_on_failures_usd=sum(f.gas_usd for f in failed),
        decision_gaps=gaps,
        # OPEN only. A HALF_OPEN host is mid-recovery and is being probed, which
        # is a different fact from "every call to it is failing fast", and
        # flattening the two would make a recovering source look like a dead one.
        open_breaker_hosts=tuple(b.host for b in breakers if b.state is BreakerState.OPEN),
        breakers=tuple(breakers),
    )


# ---------------------------------------------------------------------------
# Reading the record. The only part of this module that touches a disk.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecisionSummary:
    """One decision, reduced to what the report needs.

    A projection rather than a rehydrated :class:`~.types.DecisionRecord`,
    because a full rehydration would have to reconstruct every nested type from
    JSON and would therefore be able to *fail* on a ledger the report is supposed
    to be able to describe. Reporting must degrade; it must not refuse.
    """

    decision_id: str
    ts: float
    strategy_id: str
    market_read: str
    model: str
    effort: str
    thinking: str | None
    advisory_used: bool
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int

    @property
    def has_usage(self) -> bool:
        """Whether this decision involved a billed call at all.

        A deterministic strategy writes a decision row with four zeroes, and
        counting those as calls would divide the real spend by the number of
        ticks and report a flatteringly low cost per call.
        """
        return bool(
            self.input_tokens
            or self.output_tokens
            or self.cache_read_input_tokens
            or self.cache_creation_input_tokens
        )


@dataclass(frozen=True, slots=True)
class LedgerFacts:
    """Everything :func:`collect` could read, plus what it could not."""

    fills: tuple[Fill, ...]
    intents: tuple[OrderIntent, ...]
    decisions: tuple[DecisionSummary, ...]
    open_intents: tuple[OpenIntent, ...]
    scans: tuple[ScanResult, ...]
    unreadable_rows: tuple[str, ...]


def _int(value: Any) -> int:
    """Coerce a ledger field to int, or 0 when it is absent.

    Zero is correct here and nowhere else in this file: these are token *counts*
    written by our own writer, and a missing key means the row predates the field
    rather than meaning the count is unknown.
    """
    try:
        return int(value)
    except TypeError, ValueError:
        return 0


def _unwrap(row: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a journal envelope into a single mapping.

    ``journal`` puts the IDs beside ``kind`` and the object under ``payload``;
    ``broker`` writes flat rows to ``trades.jsonl``. Both are legitimate readers'
    inputs, so this collapses them rather than making every parser below know
    which file it came from. Envelope IDs win on conflict — they are what the
    joins are done on.
    """
    payload = row.get("payload")
    if not isinstance(payload, dict):
        return dict(row)
    merged = dict(payload)
    for key in ("decision_id", "action_id", "intent_id", "order_id", "fill_id", "run_id"):
        value = row.get(key)
        if value is not None:
            merged[key] = value
    merged.setdefault("ts", row.get("ts"))
    return merged


def _fill_from_row(row: Mapping[str, Any]) -> Fill | None:
    """Parse a fill row, or return ``None`` and let the caller record the miss.

    Deliberately not ``broker._fill_from_row``, which raises ``LedgerCorrupt``.
    The broker is right to refuse to *trade* on a ledger it cannot read; a report
    that refuses to render because one row out of four hundred is malformed
    removes the only tool available for investigating that row.
    """
    data = _unwrap(row)
    try:
        return Fill(
            fill_id=str(data["fill_id"]),
            order_id=str(data["order_id"]),
            intent_id=str(data["intent_id"]),
            decision_id=data.get("decision_id"),
            ts=float(data["ts"]),
            symbol=str(data["symbol"]),
            side=Side(data["side"]),
            state=OrderState(data["state"]),
            in_amount_atomic=int(data["in_amount_atomic"]),
            out_amount_atomic=int(data["out_amount_atomic"]),
            token_amount_atomic=int(data["token_amount_atomic"]),
            token_decimals=int(data["token_decimals"]),
            quote_fingerprint=str(data.get("quote_fingerprint", "")),
            price_usd=data.get("price_usd"),
            notional_usd=float(data["notional_usd"]),
            price_impact_pct=data.get("price_impact_pct"),
            pool_fee_usd=float(data.get("pool_fee_usd", 0.0)),
            gas_usd=float(data.get("gas_usd", 0.0)),
            realized_pnl_usd=float(data.get("realized_pnl_usd", 0.0)),
            slippage_bps_vs_quote=data.get("slippage_bps_vs_quote"),
            note=data.get("note"),
        )
    except KeyError, TypeError, ValueError:
        return None


def _intent_from_row(row: Mapping[str, Any]) -> OrderIntent | None:
    data = _unwrap(row)
    try:
        return OrderIntent(
            intent_id=str(data["intent_id"]),
            decision_id=data.get("decision_id"),
            action_id=data.get("action_id"),
            run_id=str(data.get("run_id", "")),
            ts=float(data["ts"]),
            symbol=str(data["symbol"]),
            side=Side(data["side"]),
            in_amount_atomic=int(data["in_amount_atomic"]),
            max_in_amount_atomic=int(
                data.get("max_in_amount_atomic", data["in_amount_atomic"])
            ),
            source=data.get("source", "strategy"),
            reason=str(data.get("reason", "")),
        )
    except KeyError, TypeError, ValueError:
        return None


def _decision_from_row(row: Mapping[str, Any]) -> DecisionSummary | None:
    data = _unwrap(row)
    decision_id = data.get("decision_id")
    if not isinstance(decision_id, str) or not decision_id:
        return None
    thinking = data.get("thinking")
    return DecisionSummary(
        decision_id=decision_id,
        ts=float(data.get("ts") or 0.0),
        strategy_id=str(data.get("strategy_id") or ""),
        market_read=str(data.get("market_read") or ""),
        model=str(data.get("model") or ""),
        effort=str(data.get("effort") or ""),
        thinking=thinking if isinstance(thinking, str) else None,
        advisory_used=bool(data.get("advisory_used", False)),
        input_tokens=_int(data.get("input_tokens")),
        output_tokens=_int(data.get("output_tokens")),
        cache_read_input_tokens=_int(data.get("cache_read_input_tokens")),
        cache_creation_input_tokens=_int(data.get("cache_creation_input_tokens")),
    )


def summarize_decision(record: DecisionRecord) -> DecisionSummary:
    """Project a live :class:`~.types.DecisionRecord` into the report's view.

    Lets ``loop.py`` hand the report an in-memory decision — for the ``once``
    command, which has not journaled anything yet — without a round trip through
    JSON.
    """
    return DecisionSummary(
        decision_id=record.decision_id,
        ts=record.ts,
        strategy_id=record.strategy_id,
        market_read=record.market_read,
        model=record.model,
        effort=record.effort,
        thinking=record.thinking,
        advisory_used=record.advisory_used,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        cache_read_input_tokens=record.cache_read_input_tokens,
        cache_creation_input_tokens=record.cache_creation_input_tokens,
    )


def collect(cfg: Config) -> LedgerFacts:
    """Read every persisted file the report describes. No network, never raises.

    Four files, because the system writes four and a reader that knows about only
    some of them under-reports:

    * ``ledger.jsonl``   — the general ledger (decisions, intents, transitions,
      fills), written by ``journal.Ledger``;
    * ``trades.jsonl``   — the broker's own fill ledger;
    * ``intents.jsonl``  — the broker's pre-submission intent journal;
    * ``decisions.jsonl``— the legacy decision log, still read so a report over
      an older run is not silently empty.

    Fills and intents are deduplicated by ID across files, taking the general
    ledger's copy first. A fill written to both is one fill; counting it twice
    would double the gas and halve the win rate.
    """
    scans: list[ScanResult] = []
    unreadable: list[str] = []

    def scan_path(path: Path) -> ScanResult:
        result = journal.scan(path)
        scans.append(result)
        return result

    ledger = scan_path(cfg.ledger_path)
    trades = scan_path(cfg.trades_path)
    intents_file = scan_path(cfg.intents_path)
    decisions_file = scan_path(cfg.decisions_path)

    fills: dict[str, Fill] = {}
    intents: dict[str, OrderIntent] = {}
    decisions: dict[str, DecisionSummary] = {}

    def take_fill(row: Mapping[str, Any], where: Path, index: int) -> None:
        fill = _fill_from_row(row)
        if fill is None:
            unreadable.append(f"{where}#{index} (fill)")
        else:
            fills.setdefault(fill.fill_id, fill)

    def take_intent(row: Mapping[str, Any], where: Path, index: int) -> None:
        intent = _intent_from_row(row)
        if intent is None:
            unreadable.append(f"{where}#{index} (intent)")
        else:
            intents.setdefault(intent.intent_id, intent)

    for index, row in enumerate(ledger.rows, start=1):
        kind = row.get("kind")
        if kind == RowKind.FILL:
            take_fill(row, ledger.path, index)
        elif kind == RowKind.INTENT:
            take_intent(row, ledger.path, index)
        elif kind == RowKind.DECISION:
            summary = _decision_from_row(row)
            if summary is None:
                unreadable.append(f"{ledger.path}#{index} (decision)")
            else:
                decisions.setdefault(summary.decision_id, summary)

    for index, row in enumerate(trades.rows, start=1):
        take_fill(row, trades.path, index)
    for index, row in enumerate(intents_file.rows, start=1):
        take_intent(row, intents_file.path, index)
    for row in decisions_file.rows:
        # No unreadable-row note for this file: a legacy decision log predates
        # decision_id entirely, and flagging every one of its rows as corrupt
        # would drown the integrity panel in a schema change.
        summary = _decision_from_row(row)
        if summary is not None:
            decisions.setdefault(summary.decision_id, summary)

    # An intent is open when no *terminal* fill carries its ID. The general
    # ledger's own answer is authoritative where it has one, because it can see
    # state-transition rows that the broker's flat files do not contain; the
    # flat-file fallback covers a run that only ever wrote trades.jsonl.
    open_by_id: dict[str, OpenIntent] = {
        o.intent_id: o for o in journal.open_intents(ledger.rows)
    }
    terminal = {f.intent_id for f in fills.values() if f.state.is_terminal}
    for intent in intents.values():
        if intent.intent_id in terminal or intent.intent_id in open_by_id:
            continue
        open_by_id[intent.intent_id] = OpenIntent(
            intent_id=intent.intent_id,
            run_id=intent.run_id,
            ts=intent.ts,
            symbol=intent.symbol,
            side=str(intent.side),
            source=intent.source,
            decision_id=intent.decision_id,
            action_id=intent.action_id,
            order_id=None,
            last_state=OrderState.PROPOSED,
            last_state_ts=intent.ts,
        )

    return LedgerFacts(
        fills=tuple(sorted(fills.values(), key=lambda f: f.ts)),
        intents=tuple(sorted(intents.values(), key=lambda i: i.ts)),
        decisions=tuple(sorted(decisions.values(), key=lambda d: d.ts)),
        open_intents=tuple(sorted(open_by_id.values(), key=lambda o: o.intent_id)),
        scans=tuple(scans),
        unreadable_rows=tuple(unreadable),
    )


# ---------------------------------------------------------------------------
# The whole report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Report:
    """Every computed section. Rendering this object produces the command."""

    generated_at: float
    integrity: IntegrityReport
    summary: RunSummary
    trades: TradeStats
    costs: CostBreakdown
    spend: ModelSpend
    recent_fills: tuple[Fill, ...]
    recent_decisions: tuple[DecisionSummary, ...]
    slow_tick_seconds: float


def build_report(
    *,
    facts: LedgerFacts,
    state: PortfolioState,
    model: ModelConfig,
    strategy_kind: str,
    stop_loss_pct: float,
    mode: ExecutionMode,
    slow_tick_seconds: float,
    now: float,
    breakers: Sequence[BreakerStatus] = (),
    fill_limit: int = 20,
    decision_limit: int = 5,
) -> Report:
    """Compose every section from already-read facts and an already-marked book.

    Pure: given the same ``facts``, ``state`` and ``now`` it produces the same
    report. That is what lets the tests build a book by hand and assert on
    dollars without a filesystem, a broker or a terminal.
    """
    sample = build_sample(fills=facts.fills, decisions=len(facts.decisions), now=now)
    return Report(
        generated_at=now,
        integrity=build_integrity(
            scans=facts.scans,
            fills=facts.fills,
            intents=facts.intents,
            decisions=facts.decisions,
            open_intents=facts.open_intents,
            unreadable_rows=facts.unreadable_rows,
            breakers=breakers,
        ),
        summary=build_run_summary(
            state=state, sample=sample, stop_loss_pct=stop_loss_pct, mode=mode
        ),
        trades=build_trade_stats(facts.fills),
        costs=build_cost_breakdown(facts.fills),
        spend=build_model_spend(facts.decisions, model=model, strategy_kind=strategy_kind),
        recent_fills=facts.fills[-fill_limit:] if fill_limit > 0 else (),
        recent_decisions=facts.decisions[-decision_limit:] if decision_limit > 0 else (),
        slow_tick_seconds=slow_tick_seconds,
    )


def load_report(
    cfg: Config,
    *,
    state: PortfolioState,
    now: float,
    breakers: Sequence[BreakerStatus] = (),
    fill_limit: int = 20,
    decision_limit: int = 5,
) -> Report:
    """``collect`` then ``build_report``. The CLI's single entry point.

    ``state`` is supplied by the caller because marking the book requires quotes
    and this module makes no network call. The CLI decides whether it could reach
    a router and passes either a route-marked or an unmarkable book; either way
    the basis of every price travels with it in ``PortfolioState.marks``.
    """
    return build_report(
        facts=collect(cfg),
        state=state,
        model=cfg.model,
        strategy_kind=cfg.strategy_kind,
        stop_loss_pct=cfg.risk.stop_loss_pct,
        mode=cfg.execution_mode,
        slow_tick_seconds=float(cfg.cadence.slow_tick_seconds),
        now=now,
        breakers=breakers,
        fill_limit=fill_limit,
        decision_limit=decision_limit,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_integrity(console: Console, integrity: IntegrityReport) -> None:
    """The caveats, first. Audit C11.

    Printed above the P&L on purpose. Everything in this panel is a reason the
    numbers underneath may be wrong, and a reader who has already formed a view
    of the return before reaching them will discount them.
    """
    if integrity.ok:
        console.print(
            "[green]record integrity ok[/green] — no torn lines, no open "
            "intents, no failed fills, no unreconciled decisions"
        )
        return

    lines: list[Text] = []
    lines.extend(
        Text(f"  torn ledger line   {entry}", style="yellow")
        for entry in integrity.torn_lines
    )
    lines.extend(
        Text(f"  CORRUPT row        {entry}", style="red")
        for entry in integrity.corrupt_lines
    )
    lines.extend(
        Text(f"  unreadable row     {entry}", style="red")
        for entry in integrity.unreadable_rows
    )
    lines.extend(
        Text(
            f"  OPEN INTENT        {o.symbol} {o.side} ({o.source}) "
            f"last_state={o.last_state} {o.intent_id}",
            # A SUBMITTED intent is the serious one: on a live venue the truth
            # would be at the venue and only a wallet query could settle it.
            style="red" if o.was_submitted else "yellow",
        )
        for o in integrity.open_intents
    )
    if integrity.failed_fills:
        lines.append(
            Text(
                f"  FAILED FILLS       {len(integrity.failed_fills)} attempt(s) burned "
                f"{_usd(integrity.gas_burned_on_failures_usd)} in gas and moved no "
                "inventory",
                style="red",
            )
        )
        lines.extend(
            Text(
                f"                     {f.symbol} {f.side} {f.state} "
                f"gas {_usd(f.gas_usd)} — {f.note or 'no reason recorded'}",
                style="dim",
            )
            for f in integrity.failed_fills
        )
    lines.extend(
        Text(
            f"  NO TERMINAL FILL   decision {gap.decision_id} "
            f"({', '.join(gap.symbols) or 'unknown symbol'}): "
            f"{len(gap.intent_ids)} intent(s) never settled",
            style="red",
        )
        for gap in integrity.decision_gaps
    )
    lines.extend(
        Text(f"  CIRCUIT OPEN       {host}", style="red")
        for host in integrity.open_breaker_hosts
    )

    lines.append(
        Text(
            "\nThese are reasons the figures below may be wrong. They are shown "
            "first for that reason.",
            style="dim",
        )
    )
    console.print(
        Panel(
            Text("\n").join(lines),
            title=f"[bold red]record integrity — {integrity.issue_count} issue(s)[/bold red]",
            title_align="left",
            border_style="red",
        )
    )


def render_summary(console: Console, summary: RunSummary) -> None:
    """Sample size, then the caveat, then the money. In that order. Audit C1."""
    s = summary.sample
    # The hour suffix travels with the number, not with the "n/a" that stands in
    # for it — "n/ah elapsed" reads as a duration and is not one.
    elapsed = "n/a" if s.elapsed_hours is None else f"{s.elapsed_hours:,.1f}h"
    console.print(
        f"\n[bold]Sample[/bold]  {s.decisions} decisions · {s.fills} fills "
        f"({s.landed_fills} landed, {s.failed_fills} failed) · "
        f"{s.closed_round_trips} closed round trips · "
        f"{elapsed} elapsed · mode {summary.mode}"
    )
    console.print(Panel(Text(s.caveat, style="bold yellow"), border_style="yellow"))

    total = summary.total_value_usd
    if total is None:
        headline = Text(UNAVAILABLE.upper(), style="bold red")
        detail = Text(
            f" — {len(summary.unmarkable)} unmarkable position(s): "
            f"{', '.join(summary.unmarkable)}. The book's value is unknown, so it is "
            "not summed from the positions that happen to be priceable.",
            style="red",
        )
    else:
        headline = Text(_usd(total), style="bold")
        detail = Text("")
    line = Text("  book value  ")
    line.append(headline)
    line.append("   return ")
    line.append(_pct(summary.total_return_pct))
    line.append(detail)
    console.print(line)
    console.print(
        f"  cash {_usd(summary.cash_usd)}   realised {_usd(summary.realized_pnl_usd, signed=True)}"
        f"   unrealised {_usd(summary.unrealized_pnl_usd, signed=True)}"
        f"   started {_usd(summary.starting_cash_usd)}"
    )
    console.print(
        f"  [dim]do-nothing counterfactual (all cash): "
        f"{_usd(summary.cash_counterfactual_usd)} — strategy contribution "
        f"{_usd(summary.excess_vs_cash_usd, signed=True)}[/dim]"
    )
    console.print(f"  [dim]{summary.benchmark_note}[/dim]")


def render_book(console: Console, summary: RunSummary) -> None:
    """The positions table, with the basis of every mark in its own column."""
    t = Table(title="Book", title_justify="left", header_style="bold")
    t.add_column("symbol")
    t.add_column("qty", justify="right")
    t.add_column("entry", justify="right")
    t.add_column("mark", justify="right")
    t.add_column("basis")
    t.add_column("haircut", justify="right")
    t.add_column("value", justify="right")
    t.add_column("unreal $", justify="right")
    t.add_column("unreal %", justify="right")
    t.add_column("stop", justify="right")
    t.add_column("age", justify="right")

    for line in summary.positions:
        if not line.markable:
            # Never at cost basis, never omitted. Audit C8: the old table did the
            # former, and a table that quietly dropped the row would do the
            # latter — both of which hide a position nobody can price.
            t.add_row(
                line.symbol,
                f"{line.quantity:,.4g}",
                _price(line.avg_entry_price_usd),
                Text("UNMARKABLE", style="bold red"),
                Text("unavailable", style="red"),
                NA,
                Text(UNAVAILABLE, style="red"),
                NA,
                Text(NA, style="dim"),
                _price(line.stop_price_usd),
                _age(line.age_seconds),
            )
            continue
        pnl = line.unrealized_pnl_usd
        t.add_row(
            line.symbol,
            f"{line.quantity:,.4g}",
            _price(line.avg_entry_price_usd),
            _price(line.mark_price_usd),
            Text(line.basis, style="green" if line.executable_basis else "yellow"),
            f"{line.haircut_pct:.1f}%" if line.haircut_pct else "—",
            _usd(line.value_usd),
            Text(_usd(pnl, signed=True), style="green" if (pnl or 0) >= 0 else "red"),
            _pct(line.unrealized_pnl_pct),
            _price(line.stop_price_usd),
            _age(line.age_seconds),
        )
    if not summary.positions:
        t.add_row("[dim]— flat —[/dim]", *[""] * 10)

    t.add_section()
    t.add_row(
        "[bold]cash[/bold]", *[""] * 5, f"[bold]{_usd(summary.cash_usd)}[/bold]", *[""] * 4
    )
    total = (
        Text(_usd(summary.total_value_usd), style="bold")
        if summary.value_available
        else Text(UNAVAILABLE, style="bold red")
    )
    t.add_row(
        "[bold]total[/bold]", *[""] * 5, total, "", _pct(summary.total_return_pct), "", ""
    )
    console.print(t)

    for line in summary.positions:
        if line.mark_reason and not line.executable_basis:
            console.print(f"  [dim]{line.symbol}: {line.mark_reason}[/dim]")


def render_trades(console: Console, stats: TradeStats) -> None:
    win_rate = NA if stats.win_rate_pct is None else f"{stats.win_rate_pct:.1f}%"
    share = (
        NA if stats.largest_win_share_pct is None else f"{stats.largest_win_share_pct:.0f}%"
    )
    console.print(
        f"\n[bold]Trading[/bold]  {stats.fills} fills ({stats.buys} buy / "
        f"{stats.sells} sell, {stats.failed} failed) · "
        f"{stats.closed_round_trips} closed round trips · "
        f"notional {_usd(stats.notional_traded_usd)}"
    )
    console.print(
        f"  win rate {win_rate}  ({stats.wins}W/{stats.losses}L)   "
        f"avg win {_usd(stats.avg_win_usd, signed=True)}   "
        f"avg loss {_usd(stats.avg_loss_usd, signed=True)}   "
        f"expectancy {_usd(stats.expectancy_usd, signed=True)}"
    )
    console.print(
        f"  profit factor {_num(stats.profit_factor)}   "
        f"best {_usd(stats.best_trade_usd, signed=True)}   "
        f"worst {_usd(stats.worst_trade_usd, signed=True)}   "
        f"largest win is {share} of gross profit"
    )
    console.print(
        "  [dim]Sharpe, Sortino, Calmar and max drawdown are deliberately absent: "
        "they need an equity curve at fixed intervals, and the record stores "
        "fills.[/dim]"
    )


def render_costs(console: Console, costs: CostBreakdown) -> None:
    console.print(
        f"\n[bold]Costs[/bold]  realised P&L {_usd(costs.realized_pnl_usd, signed=True)}"
        f"   gas {_usd(costs.gas_paid_usd)} (of which "
        f"{_usd(costs.gas_on_failed_usd)} on FAILED swaps)"
        f"   pool fees {_usd(costs.pool_fees_usd)}"
    )
    console.print(f"  [yellow]! {costs.pool_fee_note}[/yellow]")
    slippage = (
        NA
        if costs.mean_slippage_bps is None
        else f"{costs.mean_slippage_bps:+.1f}bps over {costs.slippage_samples} fills"
    )
    worst = (
        NA if costs.worst_slippage_bps is None else f"{costs.worst_slippage_bps:+.1f}bps"
    )
    impact = (
        NA
        if costs.mean_price_impact_pct is None
        else f"{costs.mean_price_impact_pct:.3f}% over {costs.impact_samples} fills"
    )
    console.print(
        f"  slippage vs quote {slippage} (worst {worst})   mean route impact {impact}"
    )
    # The unit is glued to the number when there is one and dropped when there
    # is not: "n/abps" reads as a number and is exactly the kind of figure a
    # reader would quote back.
    bps = (
        "n/a"
        if costs.cost_bps_of_notional is None
        else f"{costs.cost_bps_of_notional:,.1f}bps"
    )
    console.print(f"  gas as {bps} of traded notional ({_usd(costs.notional_traded_usd)})")


def render_spend(console: Console, spend: ModelSpend, *, slow_tick_seconds: float) -> None:
    if not spend.relevant:
        console.print(
            f"\n[bold]Model spend[/bold]  [dim]not applicable: strategy_kind="
            f"{spend.strategy_kind!r} makes no model calls (audit C6 removed the "
            f"LLM from trade selection). Historical spend in the record: "
            f"{_usd(spend.cost_usd)} over {spend.calls} call(s).[/dim]"
        )
        return
    if spend.calls == 0:
        console.print(
            "\n[bold]Model spend[/bold]  [dim]no billed calls in the record[/dim]"
        )
        return
    hit_rate = spend.cache_hit_rate
    daily = spend.projected_daily_usd(slow_tick_seconds)
    per_call = spend.cost_per_call_usd
    per_call_text = NA if per_call is None else f"${per_call:,.4f}"
    console.print(
        f"\n[bold]Model spend[/bold]  {spend.calls} calls   "
        f"in {spend.input_tokens:,}  out {spend.output_tokens:,}  "
        f"cache-write {spend.cache_write_tokens:,}  "
        f"cache-read {spend.cache_read_tokens:,} "
        f"({NA if hit_rate is None else f'{100 * hit_rate:.0f}% hit'})"
    )
    console.print(
        f"  [bold]{_usd(spend.cost_usd)}[/bold] to date   "
        f"{per_call_text}/call   "
        f"≈ {_usd(daily)}/day at this cadence"
    )
    if spend.cache_read_tokens == 0 and spend.calls > 1:
        console.print(
            "[yellow]  ! zero cache reads across multiple calls — the stable prompt "
            "prefix is being invalidated, and you are paying several times over.[/yellow]"
        )


def render_fills(console: Console, fills: Sequence[Fill], *, now: float) -> None:
    if not fills:
        console.print("\n[dim]no fills in the record[/dim]")
        return
    t = Table(title=f"Last {len(fills)} fills", title_justify="left", header_style="bold")
    for col in (
        "when",
        "side",
        "symbol",
        "state",
        "usd",
        "price",
        "impact",
        "slip",
        "fee",
        "gas",
        "realised",
    ):
        t.add_column(col, justify="left" if col in ("side", "symbol", "state") else "right")
    for fill in fills:
        slip = (
            NA
            if fill.slippage_bps_vs_quote is None
            else f"{fill.slippage_bps_vs_quote:+.1f}"
        )
        impact = NA if fill.price_impact_pct is None else f"{fill.price_impact_pct:.3f}%"
        t.add_row(
            _age(now - fill.ts),
            str(fill.side),
            fill.symbol,
            Text(str(fill.state), style="red" if fill.failed else "green"),
            _usd(fill.notional_usd),
            _price(fill.price_usd),
            impact,
            slip,
            _usd(fill.pool_fee_usd),
            _usd(fill.gas_usd),
            Text(
                _usd(fill.realized_pnl_usd, signed=True),
                style="green" if fill.realized_pnl_usd >= 0 else "red",
            ),
        )
    console.print(t)


def render_decisions(
    console: Console, decisions: Sequence[DecisionSummary], *, now: float
) -> None:
    if not decisions:
        console.print("\n[dim]no decisions in the record[/dim]")
        return
    for record in decisions:
        console.print(
            f"\n[bold]{_age(now - record.ts)} ago[/bold]  "
            f"[dim]{record.strategy_id or 'unknown strategy'}"
            + (f" · {record.model} effort={record.effort}" if record.model else "")
            + (" · advisory used" if record.advisory_used else "")
            + "[/dim]"
        )
        if record.market_read:
            console.print(Text(f"  {_clip(record.market_read, 400)}", style="italic dim"))
        if record.thinking:
            console.print(
                Text(
                    f"  thinking: {_clip(record.thinking, _THINKING_PREVIEW_CHARS)}",
                    style="dim",
                )
            )


def render_report(console: Console, report: Report) -> None:
    """The whole command, in the order the audit requires: caveats, sample,
    money, behaviour, costs, spend, raw rows."""
    render_integrity(console, report.integrity)
    render_summary(console, report.summary)
    render_book(console, report.summary)
    render_trades(console, report.trades)
    render_costs(console, report.costs)
    render_spend(console, report.spend, slow_tick_seconds=report.slow_tick_seconds)
    render_fills(console, report.recent_fills, now=report.generated_at)
    render_decisions(console, report.recent_decisions, now=report.generated_at)


# ---------------------------------------------------------------------------
# Evidence — used by `status`, not by `report`
# ---------------------------------------------------------------------------


def _tech_row(t: Technicals) -> str:
    # Three-way, not two: `rsi14_rising is None` means there was not enough
    # closed history to know the direction, which is not the same as "flat".
    arrow = "↑" if t.rsi14_rising else "↓" if t.rsi14_rising is False else ""
    rsi = NA if t.rsi14 is None else f"{t.rsi14:.1f}{arrow}"
    ema = NA if t.ema9_above_ema21 is None else ("9>21" if t.ema9_above_ema21 else "9<21")
    macd = NA if t.macd_hist is None else f"{t.macd_hist:+.3g}"
    cross = ""
    if t.macd_cross and t.macd_cross != "none":
        cross = f" {t.macd_cross[:4]}@{t.bars_since_cross}"
    bb = NA if t.bb_percent_b is None else f"{t.bb_percent_b:.2f}"
    atr = NA if t.atr14_pct is None else f"{t.atr14_pct:.2f}%"
    vol = NA if t.volume_ratio_prior_20 is None else f"{t.volume_ratio_prior_20:.2f}x"
    return f"RSI {rsi}  EMA {ema}  MACD {macd}{cross}  %B {bb}  ATR {atr}  Vol {vol}"


def _flow_line(brief: TechnicalBrief) -> Text:
    flow = Text("  liq    ")
    flow.append(_usd(brief.flow.liquidity_usd))
    window = brief.flow.liquidity_trend_seconds
    # The window is printed because the trend is measured against the previous
    # *decision*, not the previous read, so its span varies with how late a tick
    # ran or how long an outage lasted. A bare percentage invites reading it as
    # the cadence.
    flow.append(f"  trend/{_age(window)} " if window is not None else "  trend ")
    flow.append(_pct(brief.flow.liquidity_trend_pct))
    flow.append(f"   turnover24h {_unit(brief.flow.turnover_24h, 'x')}")
    return flow


def evidence_panel(bundle: EvidenceBundle) -> Panel:
    c = bundle.snapshot
    lines: list[Text] = []

    ladder = Text("  price  ")
    ladder.append(_price(c.price_usd))
    for label, val in (
        ("m5", c.price_change.m5),
        ("h1", c.price_change.h1),
        ("h6", c.price_change.h6),
        ("h24", c.price_change.h24),
    ):
        ladder.append(f"   {label} ")
        ladder.append(_pct(val))
    lines.append(ladder)

    if bundle.technicals is None:
        lines.append(Text(f"  liq    {_usd(c.liquidity_usd)}"))
        lines.append(Text("  tech   UNAVAILABLE this tick", style="yellow"))
    else:
        lines.append(_flow_line(bundle.technicals))
        lines.append(Text(f"  5m     {_tech_row(bundle.technicals.m5)}", style="cyan"))
        lines.append(Text(f"  1h     {_tech_row(bundle.technicals.h1)}", style="cyan"))

    txn = Text("  count  ")
    txn.append(
        f"buy/sell m5 {_num(c.txns_m5.ratio)}  h1 {_num(c.txns_h1.ratio)}  "
        f"h24 {_num(c.txns_h24.ratio)}   [transaction counts, not notional flow]"
    )
    lines.append(txn)

    if bundle.mark is not None:
        mark = bundle.mark
        style = "green" if mark.is_executable_basis else "yellow"
        lines.append(
            Text(
                f"  mark   {_price(mark.price_usd)} basis={mark.basis} "
                f"haircut={mark.haircut_pct:.1f}%"
                + (f" — {mark.reason}" if mark.reason else ""),
                style=style,
            )
        )

    s = bundle.sentiment
    if s is None:
        reason = bundle.sentiment_unavailable_reason or "no data"
        lines.append(Text(f"  social UNAVAILABLE — {reason}", style="yellow"))
    else:
        line = Text("  social ", style="magenta")
        line.append(
            f"vel1h {_unit(s.mention_velocity_1h, '/h', ',.1f')}  "
            f"vel24h {_unit(s.mention_velocity_24h, '/h', ',.1f')}  "
            f"z7d {_num(s.mention_zscore_7d, '+.2f')}  "
            f"contributors {_num(s.unique_contributors_24h, 'd')}  "
            f"ratio {_num(s.contributor_to_post_ratio)}",
            style="magenta",
        )
        lines.append(line)
        if s.mention_velocity_1h is None:
            lines.append(
                Text(
                    "         ^ vel1h n/a: the source has not indexed this hour — "
                    "unobserved, not quiet",
                    style="yellow",
                )
            )
        if s.contributor_to_post_ratio is not None and s.contributor_to_post_ratio < 0.5:
            lines.append(
                Text(
                    "         ^ few accounts posting a lot — shill-farm signature",
                    style="yellow",
                )
            )

    if c.quality_reason:
        lines.append(Text(f"  ! {c.quality}: {c.quality_reason}", style="yellow"))

    body = Text("\n").join(lines)
    return Panel(
        body, title=f"[bold]{c.symbol}[/bold]  {c.pool.dex_id}", title_align="left"
    )


def render_evidence(console: Console, evidence: Iterable[EvidenceBundle]) -> None:
    for bundle in evidence:
        console.print(evidence_panel(bundle))
