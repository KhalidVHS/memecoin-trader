"""A long, fully-documented paper-trading run.

Wraps ``loop.Trader`` and writes Markdown, because the point of this run is not
the P&L — it is having enough on disk afterwards to ask *why* each trade looked
like a good idea at the time, and to feed that back into the strategy.

Three design notes worth reading before changing anything here:

**It captures the prompt, it does not reconstruct it — and usually there is no
prompt to capture.** Audit C6 took the model off the decision path: ``[strategy]
kind`` defaults to ``baseline``, a deterministic function of two closed candles,
and ``brain.advise`` is only reached when an operator opts into ``advisory``. So
this installs a capture around ``brain.build_system``/``brain.render_user`` —
they are looked up out of ``brain``'s own module namespace, so rebinding them
records exactly what a call was given — but it counts the calls, and a tick that
made none says so in as many words. Re-rendering a prompt afterwards from the
``TickResult`` would be a near-miss (the decision history has grown by a row and
last tick's bounds have been replaced), and a near-miss is worse than nothing
when the file claims to be the input to a specific decision. An empty prompt
block on a tick that never called a model is worse still: it implies a call
happened and was lost.

**The supervisor restarts, it does not resume.** If ``Trader.run`` dies for any
reason, a fresh ``Trader`` is constructed and the run continues until the
deadline. That is safe because the book is an append-only ledger
(``data/ledger.jsonl`` plus the broker's own fill log), fsynced before each side
effect, and ``Trader.preflight`` replays it on the way up — so a restart picks
up the same positions, cash and realized P&L, and *refuses* rather than guessing
if an order was in flight when the process died. A refusal is therefore fatal to
the run, not something to retry: restarting into the same unreconciled intent
would spin forever. The only things a restart loses are in-memory:
``decision_baseline`` (so the next liquidity trend spans the outage and says so)
and ``previous``. Both are reported in EVENTS.md rather than papered over.

**Nothing here writes to the trader's decisions.** This is an observer. It adds
no risk rule, no spend cap and no retry that the live loop would not have done
on its own — it calls ``preflight`` and ``run`` exactly as ``memetrader run``
does — so the decisions documented are the decisions ``memetrader run`` would
have made unattended.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memetrader import brain, config, journal
from memetrader.broker import LiveModeUnsupported
from memetrader.config import Config
from memetrader.loop import StartupRefusal, TickResult, Trader
from memetrader.types import Fill, PortfolioState, RiskState

log = logging.getLogger("document_run")

# Tildes, not backticks. The captured prompt is free text we do not control and
# a stray triple-backtick inside it would end the fence early and scramble the
# rest of the file.
FENCE = "~~~"


# ---------------------------------------------------------------------------
# Prompt capture
# ---------------------------------------------------------------------------

# ``calls`` is the load-bearing field. ``system``/``user`` persist after a call,
# so without a counter a tick that made no model call would silently redisplay
# the previous tick's prompt as though it were its own.
_captured: dict[str, Any] = {"system": None, "user": None, "calls": 0}


def install_prompt_capture() -> None:
    """Record what ``brain`` was actually given, if it is ever called.

    ``prompts.build_system`` is keyword-only, so the wrapper forwards ``**kwargs``
    untouched rather than naming parameters it would then have to keep in step.
    Nothing is altered on the way through: this must not be able to change what
    a model would be sent.
    """
    real_build_system = brain.build_system
    real_render_user = brain.render_user

    def build_system(**kwargs: Any) -> list[dict[str, Any]]:
        out = real_build_system(**kwargs)
        _captured["system"] = out
        return out

    def render_user(*args: Any, **kwargs: Any) -> str:
        out = real_render_user(*args, **kwargs)
        _captured["user"] = out
        # Incremented here rather than in build_system because a run renders one
        # user turn per model call, whereas the frozen prefix is rebuilt for the
        # same call and would double-count nothing useful.
        _captured["calls"] = int(_captured["calls"]) + 1
        return out

    brain.build_system = build_system
    brain.render_user = render_user


def system_text(blocks: Any) -> str:
    """Flatten the cached system blocks into the text the model actually read."""
    if blocks is None:
        return ""
    if isinstance(blocks, str):
        return blocks
    parts = []
    for i, block in enumerate(blocks):
        if isinstance(block, dict):
            cache = " [cache breakpoint]" if block.get("cache_control") else ""
            parts.append(f"--- system block {i + 1}{cache} ---\n{block.get('text', '')}")
        else:
            parts.append(str(block))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def local(ts: float) -> str:
    return datetime.fromtimestamp(ts).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def price(v: float | None) -> str:
    """Memecoin prices span nine orders of magnitude; %g keeps BONK readable."""
    return "n/a" if v is None else f"{v:.10g}"


def usd(v: float | None) -> str:
    return "n/a" if v is None else f"${v:,.2f}"


def pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.2f}%"


def mag(v: float | None) -> str:
    """A percentage that is a magnitude, not a direction.

    Exposure, drawdown and price impact are never "up" — printing them with a
    leading ``+`` reads as a gain, which is the opposite of what a drawdown is.
    """
    return "n/a" if v is None else f"{v:.2f}%"


def qty(v: float | None) -> str:
    return "n/a" if v is None else f"{v:,.6g}"


def cell(text: str) -> str:
    """Make free text safe inside a Markdown table cell."""
    return text.replace("\n", " ").replace("|", "\\|").strip() or "-"


def fence(text: str, lang: str = "text") -> str:
    return f"{FENCE}{lang}\n{text}\n{FENCE}"


def jsonblock(obj: Any) -> str:
    return fence(json.dumps(journal.to_jsonable(obj), indent=2, ensure_ascii=False), "json")


def book_table(state: PortfolioState) -> str:
    """The book, including the fact that part of it may be unpriceable.

    ``total_value_usd`` is ``None`` when any held position cannot be marked, and
    that ``None`` is deliberate upstream (audit C8) — so it is rendered as
    ``n/a`` with the reason beside it rather than as a plausible sum. Every
    number here comes off ``PortfolioState``; nothing is recomputed, because a
    second arithmetic path is a second answer.
    """
    lines = [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total book value | {usd(state.total_value_usd)} |",
        f"| Cash | {usd(state.cash_usd)} |",
        f"| Unrealized P&L | {usd(state.unrealized_pnl_usd)} |",
        f"| Realized P&L (cumulative) | {usd(state.realized_pnl_usd)} |",
        f"| Total return vs ${state.starting_cash_usd:,.0f} start | "
        f"{pct(state.total_return_pct)} |",
        f"| Gross exposure | {usd(state.gross_exposure_usd)} "
        f"({mag(state.gross_exposure_pct)} of book) |",
        f"| Fees paid (cumulative) | {usd(state.fees_paid_usd)} |",
        f"| Gas paid (cumulative, incl. failed tx) | {usd(state.gas_paid_usd)} |",
    ]
    if state.unmarkable:
        lines.append(
            f"| Unmarkable positions | {', '.join(state.unmarkable)} — totals above "
            f"are incomplete for exactly that reason |"
        )
    if not state.positions:
        lines.append("")
        lines.append("_No open positions._")
        return "\n".join(lines)
    lines += [
        "",
        "| Position | Qty | Avg entry | Mark | Basis | Cost basis | Value "
        "| Unrealized | Unreal % | Age |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for sym, p in sorted(state.positions.items()):
        mark = state.marks.get(sym)
        # `Mark.price_usd` is None when the position could not be priced at all,
        # and `Mark.basis` says which of route/mid/estimate produced the number.
        # A mark and a price are not the same object any more; conflating them is
        # how this table used to hand a Mark to arithmetic expecting a float.
        mark_px = mark.price_usd if mark is not None else None
        basis = mark.basis if mark is not None else "unavailable"
        value = state.position_values_usd.get(sym)
        lines.append(
            f"| {sym} | {qty(p.quantity)} | {price(p.avg_entry_price_usd)} "
            f"| {price(mark_px)} | {basis} | {usd(p.cost_basis_usd)} | {usd(value)} "
            f"| {usd(p.unrealized_pnl_usd(mark_px))} | {pct(p.unrealized_pnl_pct(mark_px))} "
            f"| {p.age_seconds(state.ts) / 3600.0:.2f}h |"
        )
    return "\n".join(lines)


FILL_HEADER = (
    "| Time | Coin | Side | State | Notional | Fill price | Quantity | Price impact "
    "| Slippage vs quote | Pool fee | Gas | Realized P&L | Note |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
)


def fill_rows(fills: tuple[Fill, ...] | list[Fill]) -> list[str]:
    """One row per execution attempt, settled or failed.

    A failed attempt is a real row with zero amounts and non-zero gas — that is
    what a failed Solana swap costs — so it is rendered rather than filtered.
    ``quantity`` is the derived UI amount; ``token_amount_atomic`` is the fact,
    and it is in the JSON dump beside this table.
    """
    rows = []
    for f in fills:
        flags = []
        if f.failed:
            flags.append("TX FAILED (gas still paid)")
        if f.note:
            flags.append(f.note)
        slip = (
            "n/a"
            if f.slippage_bps_vs_quote is None
            else f"{f.slippage_bps_vs_quote:+.1f}bps"
        )
        rows.append(
            f"| {utc(f.ts)} | {f.symbol} | {f.side.value} | {f.state.value} "
            f"| {usd(f.notional_usd)} | {price(f.price_usd)} | {qty(f.quantity)} "
            f"| {mag(f.price_impact_pct)} | {slip} | {usd(f.pool_fee_usd)} "
            f"| {usd(f.gas_usd)} | {usd(f.realized_pnl_usd)} "
            f"| {cell('; '.join(flags))} |"
        )
    return rows


def risk_state_table(state: RiskState) -> str:
    """Continuous, cross-order risk (audit C10), including the kill switch."""
    quarantined = ", ".join(sorted(state.quarantined_symbols)) or "none"
    lines = [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Halted | {'YES' if state.halted else 'no'} |",
        f"| Halt reasons | {cell('; '.join(state.halt_reasons))} |",
        f"| May open new risk | {'yes' if state.may_open else 'no'} |",
        f"| Gross exposure | {mag(state.gross_exposure_pct)} |",
        f"| Rolling window loss | {mag(state.rolling_loss_pct)} |",
        f"| Peak book value | {usd(state.peak_value_usd)} |",
        f"| Drawdown from peak | {mag(state.drawdown_pct)} |",
        f"| Consecutive execution failures | {state.consecutive_failures} |",
        f"| Quarantined symbols | {quarantined} |",
        f"| Data health OK | {'yes' if state.data_health_ok else 'no'} |",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The recorder
# ---------------------------------------------------------------------------


class Recorder:
    def __init__(self, cfg: Config, out: Path, hours: float) -> None:
        self.cfg = cfg
        self.out = out
        self.hours = hours
        self.ticks_dir = out / "ticks"
        self.ticks_dir.mkdir(parents=True, exist_ok=True)

        # The model is only on the decision path when an operator has opted into
        # the advisory strategy. Everything that renders a prompt, a token count
        # or a bill is gated on this, so a baseline run cannot produce a section
        # implying a call that never happened.
        self.advisory = cfg.strategy_kind == "advisory"

        self.started = time.time()
        self.deadline = self.started + hours * 3600.0
        self.slow_n = 0
        self.fast_n = 0
        self.errors = 0
        self.restarts = 0
        self.cost = 0.0
        self.model_calls = 0
        self.all_fills: list[Fill] = []
        self.last_state: PortfolioState | None = None
        self.system_sha: str | None = None
        self.deadline_stop = False
        self.user_stop = False
        self.aborted: str | None = None
        self._seen_notes: set[str] = set()
        self._captured_calls = 0

        self._write("EVENTS.md", f"# Events\n\nRun started {utc(self.started)}.\n\n")
        self._write(
            "FAST-TICKS.md",
            "# Fast ticks (book marks)\n\n"
            "No strategy call on these — prices, marks and stop-loss enforcement "
            "only. This is the minute-by-minute equity curve. A `n/a` book value "
            "means a held position could not be priced at that instant, which is a "
            "data incident and not a flat book.\n\n"
            "| Time (UTC) | Book value | Cash | Unrealized | Realized | Return | Note |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n",
        )
        self._write(
            "TRADES.md",
            "# Trade ledger\n\nEvery paper fill, in order. `Realized P&L` is "
            "non-zero only on a SELL (it is booked when the position closes). "
            "A `failed`/`expired` state is an attempt that paid gas and filled "
            "nothing.\n\n" + FILL_HEADER + "\n",
        )

    # -- io ---------------------------------------------------------------

    def _write(self, name: str, text: str) -> None:
        (self.out / name).write_text(text, encoding="utf-8")

    def _append(self, name: str, text: str) -> None:
        with (self.out / name).open("a", encoding="utf-8") as fh:
            fh.write(text)

    def event(self, text: str) -> None:
        line = f"- **{utc(time.time())}** — {text}\n"
        self._append("EVENTS.md", line)
        log.info(text)

    # -- the tick callback ------------------------------------------------

    def on_tick(self, result: TickResult) -> None:
        try:
            self._notes(result)
            if result.kind == "slow":
                self._slow(result)
            else:
                self._fast(result)
        except Exception:  # noqa: BLE001 - deliberate; see below
            # A bug in the *documentation* must never take down the trading run.
            # Narrowing this would mean predicting every way a formatter can
            # fail on a value it has never seen, and being wrong once costs the
            # run. Nothing is swallowed: the traceback goes to the event log.
            self.event(
                f"recorder failed on a {result.kind} tick:"
                f"\n\n{fence(traceback.format_exc())}"
            )
        self.last_state = result.portfolio
        try:
            self.write_summary()
        except Exception:
            log.exception("summary write failed")

    def _notes(self, r: TickResult) -> None:
        """Startup/recovery notes ride on every tick; report each one once."""
        for note in r.notes:
            if note not in self._seen_notes:
                self._seen_notes.add(note)
                self.event(f"trader note: {note}")

    def _fast(self, r: TickResult) -> None:
        self.fast_n += 1
        s = r.portfolio
        note = []
        if r.stop_exits:
            note.append("STOP-LOSS: " + ", ".join(r.stop_exits))
        if r.risk_state.halted:
            note.append("risk HALTED: " + ("; ".join(r.risk_state.halt_reasons) or "-"))
        if r.error:
            note.append(f"error: {r.error}")
        if r.fills:
            self._record_fills(r.fills)
        self._append(
            "FAST-TICKS.md",
            f"| {utc(r.ts)} | {usd(s.total_value_usd)} | {usd(s.cash_usd)} "
            f"| {usd(s.unrealized_pnl_usd)} | {usd(s.realized_pnl_usd)} "
            f"| {pct(s.total_return_pct)} | {cell('; '.join(note)) if note else ''} |\n",
        )
        if note:
            self.event(f"fast tick: {'; '.join(note)}")

    def _record_fills(self, fills: tuple[Fill, ...] | list[Fill]) -> None:
        self.all_fills.extend(fills)
        self._append("TRADES.md", "\n".join(fill_rows(fills)) + "\n")

    def _slow(self, r: TickResult) -> None:
        self.slow_n += 1
        n = self.slow_n
        if r.error:
            self.errors += 1
        if r.fills:
            self._record_fills(r.fills)

        # Did a model call happen *on this tick*? The counter, not the presence
        # of stale captured text, is what answers that.
        calls_now = int(_captured["calls"])
        fresh_prompt = calls_now > self._captured_calls
        self._captured_calls = calls_now
        if fresh_prompt:
            self.model_calls += 1

        usage = r.usage
        # `Usage.cost_usd` takes the model settings object, not the whole Config:
        # it is the one cost formula in the codebase and it is given exactly the
        # prices it needs. Only a call that actually happened is billed.
        cost = (
            usage.cost_usd(self.cfg.model) if (usage is not None and fresh_prompt) else 0.0
        )
        self.cost += cost

        stamp = datetime.fromtimestamp(r.ts, UTC).strftime("%Y%m%dT%H%M%SZ")
        name = f"tick-{n:04d}-{stamp}.md"
        (self.ticks_dir / name).write_text(
            self._slow_md(r, n, cost, fresh_prompt), encoding="utf-8"
        )

        traded = [f"{f.side.value} {f.symbol} {usd(f.notional_usd)}" for f in r.fills]
        self.event(
            f"slow tick #{n} -> [{name}](ticks/{name}) — "
            + (", ".join(traded) if traded else "no fills")
            + (f" — ERROR: {r.error}" if r.error else "")
            + (f" — {usd(cost)}" if fresh_prompt else "")
        )

    def _slow_md(self, r: TickResult, n: int, cost: float, fresh_prompt: bool) -> str:
        p: list[str] = []
        a = p.append
        elapsed = (r.ts - self.started) / 3600.0

        a(f"# Slow tick #{n} — {utc(r.ts)}")
        a("")
        a(f"*{local(r.ts)} · {elapsed:.2f}h into a {self.hours:g}h run*")
        a("")
        a(
            f"A full decision cycle: evidence -> strategy "
            f"(`{self.cfg.strategy_kind}`) -> risk bounds -> quote -> execution, "
            f"in `{r.mode.value}` mode."
        )
        a("")
        if r.stop_exits:
            a(
                "> **Stop-losses fired before the strategy was consulted**: "
                + ", ".join(r.stop_exits)
                + ". Getting out is not a decision anything gets to veto."
            )
            a("")
        if r.notes:
            a("> Trader notes this tick: " + "; ".join(r.notes))
            a("")

        # -- 1. what the strategy was given ---------------------------------
        a("## 1. What the strategy was given")
        a("")
        p.extend(self._prompt_section(r, fresh_prompt))

        a("### The evidence, structured")
        a("")
        a(
            "Every field the strategy read, including the ones a renderer would "
            "show as `n/a`. A `null` here means *we could not find out*; it is "
            "never a zero. Only `closed` candles may feed a feature."
        )
        a("")
        symbols = list(dict.fromkeys((*self.cfg.symbols, *r.evidence)))
        for sym in symbols:
            bundle = r.evidence.get(sym)
            a(f"<details><summary><b>{sym}</b> — full evidence bundle</summary>")
            a("")
            a(
                jsonblock(bundle)
                if bundle is not None
                else "_No evidence bundle for this coin — the stream did not return._"
            )
            a("")
            a("</details>")
            a("")

        # -- 2. what the strategy decided -----------------------------------
        a("## 2. What the strategy decided, and why")
        a("")
        if r.error:
            a(f"**The strategy call failed: {cell(r.error)}**")
            a("")
            a(
                "No decision was made. A failure is not a decision to hold — the "
                "tick was skipped and the book left as the stops above left it."
            )
            a("")
            p.extend(self._risk_section(r))
            p.extend(self._fills_section(r))
            p.extend(self._book_section(r))
            return "\n".join(p)

        d = r.decision
        if d is None:
            a(
                "_No decision this tick._ The strategy was not called — see the "
                "risk section below, which is the only thing that skips it "
                "without an error (a halt means exits only)."
            )
            a("")
            p.extend(self._risk_section(r))
            p.extend(self._fills_section(r))
            p.extend(self._book_section(r))
            return "\n".join(p)

        a(f"Decision `{d.decision_id}` from strategy `{d.strategy_id}`.")
        a("")
        a("### Market read")
        a("")
        a("> " + str(d.market_read or "(none)").replace("\n", "\n> "))
        a("")

        # `Usage` is loosely typed on TickResult and the thinking trace is only
        # populated on the advisory path, so ask rather than assume.
        thinking = getattr(r.usage, "thinking", None) if fresh_prompt else None
        if self.advisory and thinking:
            a("### Reasoning trace (the model's own thinking)")
            a("")
            a(
                "Raw adaptive-thinking output. This is most of what the tick cost, "
                "and it is the only record of the reasoning the advice compresses. "
                "It is advice, not authority: the targets below still passed "
                "through the same hurdle and the same risk bounds as the baseline."
            )
            a("")
            a("<details><summary>Expand thinking</summary>")
            a("")
            a(fence(str(thinking)))
            a("")
            a("</details>")
            a("")

        a("### Target positions")
        a("")
        a(
            "A strategy emits *desired inventory*, not trades. `loop.py` diffs "
            "these against what is held and schedules the delta, which is why a "
            "target of $0 on a coin you do not hold is not an order."
        )
        a("")
        if not d.targets:
            a("_The strategy emitted no targets._")
        else:
            a("| Coin | Target $ | Held $ | Rationale |")
            a("| --- | --- | --- | --- |")
            for t in d.targets:
                # Absent from `positions` means flat, which is $0.00 and not an
                # unknown. Present with a `None` value means held but unpriceable,
                # which is the only honest `n/a` in this column.
                if t.symbol not in r.portfolio.positions:
                    held = "$0.00"
                else:
                    value = r.portfolio.position_values_usd.get(t.symbol)
                    held = usd(value) if value is not None else "n/a (unmarkable)"
                a(f"| {t.symbol} | {usd(t.target_usd)} | {held} | {cell(t.rationale)} |")
        a("")

        a("### Forecasts")
        a("")
        a(
            "Every view carries a horizon and a **net** (after estimated cost) "
            "magnitude, plus a predictive interval. The entry hurdle is tested "
            "against `lower quantile`, not the mean. `calibration` is `none` on "
            "everything here, which is what forbids sizing on conviction."
        )
        a("")
        if not d.forecasts:
            a("_No forecasts were produced._")
        else:
            a(
                "| Coin | Horizon | Expected net | Lower q | Upper q | Model "
                "| Calibration | Missing features | Note |"
            )
            a("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for f in d.forecasts:
                a(
                    f"| {f.symbol} | {f.horizon_seconds / 60:.0f}m "
                    f"| {pct(f.expected_net_return_pct)} | {pct(f.lower_quantile_pct)} "
                    f"| {pct(f.upper_quantile_pct)} | {f.model_id} "
                    f"| {f.calibration_id or 'none'} "
                    f"| {cell(', '.join(f.features_missing))} | {cell(f.note)} |"
                )
        a("")
        if d.diagnostics:
            a("Strategy diagnostics:")
            a("")
            a(jsonblock(d.diagnostics))
            a("")

        p.extend(self._risk_section(r))
        p.extend(self._intents_section(r))
        p.extend(self._fills_section(r))
        p.extend(self._book_section(r))
        p.extend(self._cost_section(r, cost, fresh_prompt))
        return "\n".join(p)

    # -- sections ----------------------------------------------------------

    def _prompt_section(self, r: TickResult, fresh_prompt: bool) -> list[str]:
        """The prompt, when there was one — and an honest heading when there was not.

        The default path makes no model call at all, so the usual outcome of this
        method is a short paragraph saying which deterministic strategy ran. The
        alternative — an empty fenced block labelled "prompt" — would read as a
        call whose text we failed to record, which is a different and much worse
        claim.
        """
        out: list[str] = []
        a = out.append
        if not self.advisory:
            a(f"### No model call — deterministic strategy `{self.cfg.strategy_kind}`")
            a("")
            a(
                f"There is no prompt for this tick because there was no model call. "
                f"`[strategy] kind` is `{self.cfg.strategy_kind}`, a deterministic "
                f"function of the evidence below; audit C6 removed the language "
                f"model from trade selection and sizing, and the advisory path is "
                f"opt-in. The complete input to this decision is the structured "
                f"evidence below — for a deterministic strategy that *is* the whole "
                f"prompt, and it is reproducible from it."
            )
            a("")
            return out

        if not fresh_prompt:
            a("### No model call on this tick")
            a("")
            a(
                "The advisory strategy is enabled, but `brain.advise` was not "
                "reached this tick — a risk halt skips the strategy entirely, and "
                "an advisory failure falls back to the deterministic baseline "
                "before any prompt is rendered. The previous tick's prompt is "
                "deliberately not shown here: it was not this tick's input."
            )
            a("")
            return out

        a(
            f"Model `{self.cfg.model.name}`, effort `{self.cfg.model.effort}`, "
            f"max_tokens {self.cfg.model.max_tokens}, adaptive thinking."
        )
        a("")
        sys_text = system_text(_captured.get("system"))
        sha = hashlib.sha256(sys_text.encode("utf-8")).hexdigest()[:16]
        a("### System prompt (cached, frozen for the life of the run)")
        a("")
        if self.system_sha is None:
            self.system_sha = sha
            (self.out / "SYSTEM-PROMPT.md").write_text(
                f"# System prompt\n\nsha256:{sha} — captured {utc(r.ts)}.\n\n"
                "Sent as cached blocks on every tick and byte-frozen for the life "
                "of the run. Reproduced here once rather than in every tick file; "
                "each tick verifies the hash and inlines the text if it ever "
                "differs.\n\n" + fence(sys_text),
                encoding="utf-8",
            )
        if sha == self.system_sha:
            a(
                f"Identical to [SYSTEM-PROMPT.md](../SYSTEM-PROMPT.md) "
                f"(sha256:`{sha}`). Not repeated here."
            )
        else:
            a(
                f"**CHANGED** this tick (sha256:`{sha}`, was `{self.system_sha}`) "
                f"— the frozen-prefix contract was broken; inlined in full:"
            )
            a("")
            a(fence(sys_text))
        a("")
        a("### User message (the volatile brief — verbatim)")
        a("")
        a("This is the exact text the model read this tick.")
        a("")
        a(fence(str(_captured.get("user") or "")))
        a("")
        return out

    def _risk_section(self, r: TickResult) -> list[str]:
        """Bounds, not approvals — audit C3's structural half."""
        out: list[str] = []
        a = out.append
        a("## 3. What the risk layer said")
        a("")
        a(
            "Risk does not approve or clamp an order. It answers *what is the most "
            "you may do?*; the execution layer then requotes at that size and "
            "`confirm_quote` re-checks the quote it is actually about to send. "
            "A bound with a veto and a $0 ceiling is a refusal."
        )
        a("")
        a("### Continuous risk state")
        a("")
        a(risk_state_table(r.risk_state))
        a("")
        a("### Per-order bounds")
        a("")
        if not r.bounds:
            a(
                "_No bounds were computed (no target differed from inventory enough to trade)._"
            )
        else:
            a("| Coin | Side | Permitted | Max notional | Binding rule | Reason | Notes |")
            a("| --- | --- | --- | --- | --- | --- | --- |")
            for b in r.bounds:
                extra = list(b.notes)
                if b.bypassed_rules:
                    extra.append("bypassed: " + ", ".join(b.bypassed_rules))
                if b.vetoes:
                    extra.append("vetoes: " + ", ".join(b.vetoes))
                a(
                    f"| {b.symbol} | {b.side.value} "
                    f"| {'PERMITTED' if b.permitted else 'REFUSED'} "
                    f"| {usd(b.max_notional_usd)} | {b.binding_rule or '-'} "
                    f"| {cell(b.reason)} | {cell('; '.join(extra))} |"
                )
        a("")
        return out

    def _intents_section(self, r: TickResult) -> list[str]:
        """The durable order records, written before anything was submitted."""
        out: list[str] = []
        a = out.append
        a("## 4. Orders that were actually attempted")
        a("")
        a(
            "An intent is journaled to `data/ledger.jsonl` *before* the side "
            "effect, and `in_amount_atomic` is exactly what was quoted — the "
            "broker refuses any other amount. That is audit C3 and C11 enforced "
            "rather than asserted."
        )
        a("")
        if not r.intents:
            a("_No orders were attempted this tick._")
        else:
            a("| Intent | Coin | Side | Source | In amount (atomic) | Reason |")
            a("| --- | --- | --- | --- | --- | --- |")
            for i in r.intents:
                a(
                    f"| `{i.intent_id}` | {i.symbol} | {i.side.value} | {i.source} "
                    f"| {i.in_amount_atomic:,} | {cell(i.reason)} |"
                )
        a("")
        return out

    def _fills_section(self, r: TickResult) -> list[str]:
        out: list[str] = []
        a = out.append
        a("## 5. What actually traded")
        a("")
        if not r.fills:
            a("_No fills this tick._")
        else:
            a(FILL_HEADER)
            out.extend(fill_rows(r.fills))
            a("")
            a(
                "`Realized P&L` is booked on the SELL that closes a position, "
                "against average cost basis including the fees and gas paid to "
                "open it. A failed transaction fills $0 and still pays gas."
            )
            a("")
            a("<details><summary>The same fills, in full</summary>")
            a("")
            a(jsonblock(list(r.fills)))
            a("")
            a("</details>")
        a("")
        return out

    def _book_section(self, r: TickResult) -> list[str]:
        """One book, labelled as one book.

        ``TickResult`` carries a single ``PortfolioState``, re-marked after the
        tick's fills. There is no "before" snapshot to print, so this does not
        print the same table twice under two headings and call one of them the
        opening book — the position and cash columns *did* move, and showing an
        identical table as "going in" would be a fabrication.
        """
        out: list[str] = []
        a = out.append
        a("## 6. The book coming out")
        a("")
        a(
            f"Marked at {utc(r.portfolio.ts)}, after this tick's fills. This is the "
            "only book snapshot the tick reports; the state going in is the "
            "previous tick's closing book."
        )
        a("")
        a(book_table(r.portfolio))
        a("")
        return out

    def _cost_section(self, r: TickResult, cost: float, fresh_prompt: bool) -> list[str]:
        out: list[str] = []
        a = out.append
        a("## 7. What this tick cost")
        a("")
        u = r.usage
        if not self.advisory:
            a(
                f"**$0.00 in model spend.** The `{self.cfg.strategy_kind}` strategy "
                "is arithmetic over the evidence above and makes no API calls. The "
                "only costs this tick could incur are the paper trading costs — "
                "pool fee, price impact and gas — which are in the fills table."
            )
        elif u is None or not fresh_prompt:
            a("_No model call on this tick, so nothing was billed._")
        else:
            a("| Metric | Value |")
            a("| --- | --- |")
            a(f"| Input tokens (uncached) | {u.input_tokens:,} |")
            a(f"| Cache read | {u.cache_read_input_tokens:,} |")
            a(f"| Cache write | {u.cache_creation_input_tokens:,} |")
            a(f"| Total prompt | {u.total_input_tokens:,} |")
            a(f"| Cache hit rate | {u.cache_hit_rate:.1%} |")
            a(f"| Output tokens | {u.output_tokens:,} |")
            a(f"| System prefix fingerprint | `{u.prompt_fingerprint or 'n/a'}` |")
            a(f"| Cost this tick | ${cost:.4f} |")
            a(f"| Cost run-to-date | ${self.cost:.4f} |")
        a("")
        return out

    # -- summary -----------------------------------------------------------

    def write_summary(self) -> None:
        now = time.time()
        elapsed = (now - self.started) / 3600.0
        remaining = max(0.0, (self.deadline - now) / 3600.0)
        s = self.last_state
        done = (
            self.deadline_stop
            or self.user_stop
            or self.aborted is not None
            or remaining <= 0
        )

        settled = [f for f in self.all_fills if not f.failed]
        sells = [f for f in settled if f.side.value == "SELL"]
        wins = [f for f in sells if f.realized_pnl_usd > 0]
        losses = [f for f in sells if f.realized_pnl_usd < 0]
        failed = [f for f in self.all_fills if f.failed]

        p: list[str] = []
        a = p.append
        a("# Run summary")
        a("")
        a(f"**Status:** {'COMPLETE' if done else 'RUNNING'}  ")
        if self.aborted:
            a(f"**Aborted:** {self.aborted}  ")
        a(f"**Strategy:** `{self.cfg.strategy_kind}`  ")
        a(
            f"**Execution mode:** `{self.cfg.execution_mode.value}` (paper book, no wallet)  "
        )
        a(f"**Started:** {utc(self.started)} ({local(self.started)})  ")
        a(f"**Deadline:** {utc(self.deadline)} ({local(self.deadline)})  ")
        a(f"**Elapsed:** {elapsed:.2f}h of {self.hours:g}h — {remaining:.2f}h remaining  ")
        a(f"**Last updated:** {utc(now)}")
        a("")
        a("## Cadence")
        a("")
        a("| | |")
        a("| --- | --- |")
        a(f"| Slow ticks (decision cycles) | {self.slow_n} |")
        a(f"| Fast ticks (book marks) | {self.fast_n} |")
        a(f"| Ticks where the strategy failed | {self.errors} |")
        a(f"| Supervisor restarts | {self.restarts} |")
        if self.advisory:
            a(f"| Model calls | {self.model_calls} |")
            a(f"| Model spend so far | ${self.cost:.4f} |")
            a(
                f"| Projected full-run spend | "
                f"${(self.cost / elapsed * self.hours) if elapsed > 0.05 else 0.0:.2f} |"
            )
        else:
            a(
                "| Model calls | 0 — the deterministic strategy never reaches "
                "`brain.advise` |"
            )
            a("| Model spend | $0.0000 |")
        a("")
        a("## Results")
        a("")
        if s is None:
            a("_No tick has completed yet._")
        else:
            a(book_table(s))
        a("")
        a("## Trades")
        a("")
        a("| | |")
        a("| --- | --- |")
        a(f"| Settled fills | {len(settled)} |")
        a(f"| Closing sells | {len(sells)} |")
        a(f"| Winners | {len(wins)} |")
        a(f"| Losers | {len(losses)} |")
        a(f"| Hit rate | {(100.0 * len(wins) / len(sells)) if sells else 0.0:.1f}% |")
        a(f"| Failed transactions (gas paid, nothing filled) | {len(failed)} |")
        if wins:
            a(
                f"| Average winner | {usd(sum(f.realized_pnl_usd for f in wins) / len(wins))} |"
            )
        if losses:
            a(
                f"| Average loser | "
                f"{usd(sum(f.realized_pnl_usd for f in losses) / len(losses))} |"
            )
        a("")
        if self.all_fills:
            a(FILL_HEADER)
            p.extend(fill_rows(self.all_fills))
        else:
            a("_No trades._" if done else "_No trades yet._")
        a("")
        a("## Where everything is")
        a("")
        a("| File | What it holds |")
        a("| --- | --- |")
        a(
            "| `ticks/tick-NNNN-*.md` | One per decision cycle: the full evidence "
            "bundle, the target positions and forecasts behind them, the risk "
            "state and bounds, the intents, the fills, the book and the cost |"
        )
        if self.advisory:
            a(
                "| `SYSTEM-PROMPT.md` | The frozen system prefix, verbatim, "
                "captured on the first model call |"
            )
        a(
            "| `TRADES.md` | Every attempt in order, with price, quantity, fees, "
            "gas, slippage-vs-quote and realized P&L |"
        )
        a("| `FAST-TICKS.md` | The book-mark equity curve and stop-loss enforcement |")
        a(
            "| `EVENTS.md` | Errors, restarts, halts, stop-losses — anything that "
            "was not a clean tick |"
        )
        a(
            "| `../../data/ledger.jsonl` | The joined decision / intent / "
            "state-transition / fill stream the trader itself writes. Rows sort "
            "chronologically by `row_id` and join by ID, never by symbol |"
        )
        a(
            "| `../../data/risk_ledger.json` | Persisted risk state: peak book "
            "value, loss-window anchors, consecutive failures, post-stop "
            "quarantines and the manual halt |"
        )
        a("| `../../data/trades.jsonl` | The broker's own fill log |")
        a("| `../../data/state.json` | The broker's replayable book snapshot |")
        a("")
        self._write("SUMMARY.md", "\n".join(p) + "\n")


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Document a long paper-trading run.")
    ap.add_argument("--hours", type=float, default=12.0, help="How long to run.")
    ap.add_argument("--out", type=Path, default=None, help="Output directory.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    cfg = config.load()
    install_prompt_capture()

    root = Path(__file__).resolve().parent.parent
    out = args.out or (root / "runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True, exist_ok=True)

    rec = Recorder(cfg, out, args.hours)
    log.info("documenting a %.2fh run into %s", args.hours, out)
    rec.event(
        f"documenting {args.hours:g}h; strategy {cfg.strategy_kind}; "
        f"mode {cfg.execution_mode.value}; slow tick every "
        f"{cfg.cadence.slow_tick_seconds}s, fast tick every "
        f"{cfg.cadence.fast_tick_seconds}s; book ${cfg.starting_cash_usd:,.0f}"
    )
    if rec.advisory:
        rec.event(
            f"advisory (LLM) strategy is enabled — {cfg.model.name} at effort "
            f"{cfg.model.effort}. Audit C6 records that this component has no "
            f"measured predictive value; sizing stays flat and risk still bounds "
            f"every order."
        )

    # ``Trader`` has no public stop, but it polls ``_stop`` between naps and the
    # documented contract of that flag is "finish this tick, then return with
    # state saved". Setting it is exactly what the signal handler does.
    holder: dict[str, Trader | None] = {"trader": None}

    def on_tick(result: TickResult) -> None:
        rec.on_tick(result)
        if time.time() >= rec.deadline:
            rec.deadline_stop = True
            trader = holder["trader"]
            if trader is not None:
                trader._stop = True

    # The supervisor. ``Trader.run`` already isolates a failure in any one
    # evidence stream and already survives a vendor outage, so reaching here
    # means something it does not handle — a socket the http client did not
    # wrap, an OS hiccup, a bug. The deadline is the only thing that ends this
    # loop, other than the operator or a refusal it would be wrong to retry.
    while time.time() < rec.deadline and not rec.user_stop and rec.aborted is None:
        try:
            with Trader(cfg) as trader:
                holder["trader"] = trader
                # Exactly what `memetrader run` does, and for the same reason:
                # an intent with no terminal state is an order whose outcome is
                # unknown, and restarting into it would place it twice.
                for note in trader.preflight():
                    rec.event(f"recovered on startup: {note}")
                trader.run(on_tick=on_tick)
            # A clean return with time left and no deadline stop means the
            # trader's own signal handler fired: the operator asked to stop.
            if not rec.deadline_stop:
                rec.user_stop = True
                rec.event("stopped by operator (SIGINT/SIGTERM); book is saved")
        except StartupRefusal as exc:
            # Fatal on purpose. The ledger says an order was in flight and
            # nothing in this process can settle it; restarting would spin on
            # the same refusal until the deadline while documenting nothing.
            rec.aborted = f"refused to trade: {exc}"
            rec.event(f"REFUSED TO TRADE — the run stops here.\n\n{fence(str(exc))}")
        except LiveModeUnsupported as exc:
            rec.aborted = f"refused to run: {exc}"
            rec.event(f"REFUSED TO RUN — the run stops here.\n\n{fence(str(exc))}")
        except KeyboardInterrupt:
            rec.user_stop = True
            rec.event("interrupted by operator; book is saved")
        except Exception:  # noqa: BLE001 - this is the supervisor
            # The point of this process is to keep a 12-hour run alive across a
            # crash it cannot anticipate. Catching narrowly would end the run on
            # the first surprise, which is what the restart loop exists to
            # prevent. The traceback is recorded in EVENTS.md either way.
            rec.restarts += 1
            rec.event(
                f"trader crashed (restart #{rec.restarts}); restarting in 30s. "
                f"In-memory liquidity baseline is lost, so the next tick's "
                f"liquidity trend spans the outage and will say so.\n\n"
                + fence(traceback.format_exc())
            )
            # Sleep in slices so the deadline still ends the run promptly.
            until = min(time.time() + 30.0, rec.deadline)
            while time.time() < until:
                time.sleep(1.0)

    if rec.aborted is None:
        rec.deadline_stop = rec.deadline_stop or time.time() >= rec.deadline
        rec.event(
            "run finished: "
            + ("deadline reached" if rec.deadline_stop else "stopped early by operator")
        )
    rec.write_summary()
    log.info("done — %s", out / "SUMMARY.md")
    return 2 if rec.aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
