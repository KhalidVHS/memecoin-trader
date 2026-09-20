"""A long, fully-documented paper-trading run.

Wraps ``loop.Trader`` and writes Markdown, because the point of this run is not
the P&L — it is having enough on disk afterwards to ask *why* each trade looked
like a good idea at the time, and to feed that back into the strategy.

Three design notes worth reading before changing anything here:

**It captures the prompt, it does not reconstruct it.** ``brain.decide`` calls
``build_system`` and ``render_user`` out of its own module namespace, so this
rebinds those two names and records what they actually returned. Re-rendering
the prompt afterwards from the ``TickResult`` would produce a near-miss: by the
time ``slow_tick`` returns, ``pending_rejections`` has been replaced with *this*
tick's rejections and the decision journal has grown by a row, so the history
and rejection sections would both differ from what the model was really shown.
A near-miss is worse than nothing here — the whole file claims to be the input
to a specific decision.

**The supervisor restarts, it does not resume.** If ``Trader.run`` dies for any
reason, a fresh ``Trader`` is constructed and the run continues until the
deadline. That is safe because the book lives in ``data/state.json`` and is
written atomically after every mutation, so a restart picks up the same
positions, cash and realized P&L. The only thing lost is in-memory:
``decision_baseline`` (so the next liquidity trend spans the outage and says so)
and ``previous``. Both are reported in EVENTS.md rather than papered over.

**Nothing here writes to the model's context.** This is an observer. It adds no
risk rule, no spend cap and no retry that the live loop would not have done on
its own, so the decisions documented are the decisions ``memetrader run`` would
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memetrader import brain, config, journal
from memetrader.loop import TickResult, Trader
from memetrader.types import Fill, PortfolioState

log = logging.getLogger("document_run")

# Tildes, not backticks. The captured prompt is free text we do not control and
# a stray triple-backtick inside it would end the fence early and scramble the
# rest of the file.
FENCE = "~~~"


# ---------------------------------------------------------------------------
# Prompt capture
# ---------------------------------------------------------------------------

_captured: dict[str, Any] = {"system": None, "user": None}


def install_prompt_capture() -> None:
    real_build_system = brain.build_system
    real_render_user = brain.render_user

    def build_system(cfg):  # type: ignore[no-untyped-def]
        out = real_build_system(cfg)
        _captured["system"] = out
        return out

    def render_user(*args, **kwargs):  # type: ignore[no-untyped-def]
        out = real_render_user(*args, **kwargs)
        _captured["user"] = out
        return out

    brain.build_system = build_system  # type: ignore[assignment]
    brain.render_user = render_user  # type: ignore[assignment]


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
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def local(ts: float) -> str:
    return datetime.fromtimestamp(ts).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def price(v: float | None) -> str:
    """Memecoin prices span nine orders of magnitude; %g keeps BONK readable."""
    return "n/a" if v is None else f"{v:.10g}"


def usd(v: float | None) -> str:
    return "n/a" if v is None else f"${v:,.2f}"


def pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.2f}%"


def fence(text: str, lang: str = "text") -> str:
    return f"{FENCE}{lang}\n{text}\n{FENCE}"


def jsonblock(obj: Any) -> str:
    return fence(json.dumps(journal.to_jsonable(obj), indent=2, ensure_ascii=False), "json")


def book_table(state: PortfolioState) -> str:
    lines = [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total book value | {usd(state.total_value_usd)} |",
        f"| Cash | {usd(state.cash_usd)} |",
        f"| Unrealized P&L | {usd(state.unrealized_pnl_usd)} |",
        f"| Realized P&L (cumulative) | {usd(state.realized_pnl_usd)} |",
        f"| Total return vs ${state.starting_cash_usd:,.0f} start | {pct(state.total_return_pct)} |",
        f"| Fees paid (cumulative) | {usd(state.fees_paid_usd)} |",
        f"| Gas paid (cumulative) | {usd(state.gas_paid_usd)} |",
    ]
    if not state.positions:
        lines.append("")
        lines.append("_No open positions._")
        return "\n".join(lines)
    lines += [
        "",
        "| Position | Qty | Avg entry | Mark | Cost basis | Value | Unrealized | Unreal % | Age |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for sym, p in state.positions.items():
        mark = state.marks.get(sym)
        value = state.position_values_usd.get(sym)
        upnl = None if mark is None else p.unrealized_pnl_usd(mark)
        upct = None if mark is None else p.unrealized_pnl_pct(mark)
        age_h = p.age_seconds(state.ts) / 3600.0
        lines.append(
            f"| {sym} | {p.quantity:,.6g} | {price(p.avg_entry_price_usd)} | {price(mark)} "
            f"| {usd(p.cost_basis_usd)} | {usd(value)} | {usd(upnl)} | {pct(upct)} "
            f"| {age_h:.2f}h |"
        )
    return "\n".join(lines)


def fill_rows(fills: list[Fill]) -> list[str]:
    rows = []
    for f in fills:
        flags = []
        if f.failed:
            flags.append("TX FAILED (gas still paid)")
        if f.degraded:
            flags.append("degraded quote")
        if f.note:
            flags.append(f.note)
        rows.append(
            f"| {utc(f.ts)} | {f.symbol} | {f.side} | {usd(f.requested_usd)} "
            f"| {usd(f.filled_usd)} | {price(f.price_usd)} | {f.quantity:,.6g} "
            f"| {pct(f.price_impact_pct)} | {usd(f.pool_fee_usd)} | {usd(f.gas_usd)} "
            f"| {usd(f.realized_pnl_usd)} | {'; '.join(flags) or '-'} |"
        )
    return rows


FILL_HEADER = (
    "| Time | Coin | Side | Requested | Filled | Fill price | Quantity "
    "| Price impact | Pool fee | Gas | Realized P&L | Flags |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
)


# ---------------------------------------------------------------------------
# The recorder
# ---------------------------------------------------------------------------


class Recorder:
    def __init__(self, cfg, out: Path, hours: float) -> None:
        self.cfg = cfg
        self.out = out
        self.hours = hours
        self.ticks_dir = out / "ticks"
        self.ticks_dir.mkdir(parents=True, exist_ok=True)

        self.started = time.time()
        self.deadline = self.started + hours * 3600.0
        self.slow_n = 0
        self.fast_n = 0
        self.errors = 0
        self.restarts = 0
        self.cost = 0.0
        self.all_fills: list[Fill] = []
        self.last_state: PortfolioState | None = None
        self.system_sha: str | None = None
        self.deadline_stop = False
        self.user_stop = False

        self._write("EVENTS.md", f"# Events\n\nRun started {utc(self.started)}.\n\n")
        self._write(
            "FAST-TICKS.md",
            "# Fast ticks (60s book marks)\n\n"
            "No model call on these — prices, marks and stop-loss enforcement only. "
            "This is the minute-by-minute equity curve.\n\n"
            "| Time (UTC) | Book value | Cash | Unrealized | Realized | Return | Note |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n",
        )
        self._write(
            "TRADES.md",
            "# Trade ledger\n\nEvery paper fill, in order. `Realized P&L` is "
            "non-zero only on a SELL (it is booked when the position closes).\n\n"
            + FILL_HEADER
            + "\n",
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
            if result.kind == "slow":
                self._slow(result)
            else:
                self._fast(result)
        except Exception:
            # A bug in the *documentation* must never take down the trading run.
            self.event(f"recorder failed on a {result.kind} tick:\n\n{fence(traceback.format_exc())}")
        self.last_state = result.portfolio
        try:
            self.write_summary()
        except Exception:
            log.exception("summary write failed")

    def _fast(self, r: TickResult) -> None:
        self.fast_n += 1
        s = r.portfolio
        note = []
        if r.stop_loss_exits:
            note.append("STOP-LOSS: " + ", ".join(r.stop_loss_exits))
        if r.error:
            note.append(f"error: {r.error}")
        if r.fills:
            self._record_fills(r.fills)
        self._append(
            "FAST-TICKS.md",
            f"| {utc(r.ts)} | {usd(s.total_value_usd)} | {usd(s.cash_usd)} "
            f"| {usd(s.unrealized_pnl_usd)} | {usd(s.realized_pnl_usd)} "
            f"| {pct(s.total_return_pct)} | {'; '.join(note) or ''} |\n",
        )
        if note:
            self.event(f"fast tick: {'; '.join(note)}")

    def _record_fills(self, fills: list[Fill]) -> None:
        self.all_fills.extend(fills)
        self._append("TRADES.md", "\n".join(fill_rows(list(fills))) + "\n")

    def _slow(self, r: TickResult) -> None:
        self.slow_n += 1
        n = self.slow_n
        if r.error:
            self.errors += 1
        if r.fills:
            self._record_fills(r.fills)
        usage = r.usage
        cost = usage.cost_usd(self.cfg) if usage else 0.0
        self.cost += cost

        stamp = datetime.fromtimestamp(r.ts, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"tick-{n:04d}-{stamp}.md"
        (self.ticks_dir / name).write_text(self._slow_md(r, n, cost), encoding="utf-8")

        traded = [f"{f.side} {f.symbol} {usd(f.filled_usd)}" for f in r.fills]
        self.event(
            f"slow tick #{n} -> [{name}](ticks/{name}) — "
            + (", ".join(traded) if traded else "no fills")
            + (f" — ERROR: {r.error}" if r.error else "")
            + f" — {usd(cost)}"
        )

    def _slow_md(self, r: TickResult, n: int, cost: float) -> str:
        p: list[str] = []
        a = p.append
        elapsed = (r.ts - self.started) / 3600.0

        a(f"# Slow tick #{n} — {utc(r.ts)}")
        a("")
        a(f"*{local(r.ts)} · {elapsed:.2f}h into a {self.hours:g}h run*")
        a("")
        a("A full decision cycle: evidence -> Opus 5 -> risk rules -> execution.")
        a("")

        # -- 1. book before ------------------------------------------------
        a("## 1. The book going in")
        a("")
        a(book_table(r.portfolio))
        a("")
        if r.stop_loss_exits:
            a(
                "> **Stop-losses fired before the model was consulted**: "
                + ", ".join(r.stop_loss_exits)
                + ". Getting out is not a decision the model gets to veto."
            )
            a("")

        # -- 2. what was sent ----------------------------------------------
        a("## 2. What was sent to the model")
        a("")
        a(
            f"Model `{self.cfg.model.name}`, effort `{self.cfg.model.effort}`, "
            f"max_tokens {self.cfg.model.max_tokens}, adaptive thinking."
        )
        a("")

        sys_text = system_text(_captured.get("system"))
        sha = hashlib.sha256(sys_text.encode("utf-8")).hexdigest()[:16]
        a("### 2a. System prompt (cached, stable across ticks)")
        a("")
        if self.system_sha is None:
            self.system_sha = sha
            (self.out / "SYSTEM-PROMPT.md").write_text(
                f"# System prompt\n\nsha256:{sha} — captured {utc(r.ts)}.\n\n"
                "Sent as cached blocks on every tick. Reproduced here once rather "
                "than in all ~48 tick files; each tick verifies the hash and "
                "inlines the text if it ever differs.\n\n" + fence(sys_text),
                encoding="utf-8",
            )
        if sha == self.system_sha:
            a(
                f"Identical to [SYSTEM-PROMPT.md](../SYSTEM-PROMPT.md) "
                f"(sha256:`{sha}`). Not repeated here."
            )
        else:
            a(f"**CHANGED** this tick (sha256:`{sha}`, was `{self.system_sha}`) — inlined in full:")
            a("")
            a(fence(sys_text))
        a("")

        a("### 2b. User message (the volatile brief — verbatim)")
        a("")
        a("This is the exact text the model read this tick.")
        a("")
        a(fence(str(_captured.get("user") or "(not captured — the call never reached the model)")))
        a("")

        a("### 2c. The same evidence, structured")
        a("")
        a(
            "Every field behind the prose above, including the ones the prompt "
            "renders as `n/a`. A `null` here means *we could not find out*; it is "
            "never a zero."
        )
        a("")
        for sym in self.cfg.symbols:
            bundle = r.evidence.get(sym)
            a(f"<details><summary><b>{sym}</b> — full evidence bundle</summary>")
            a("")
            a(jsonblock(bundle) if bundle is not None else "_No evidence bundle for this coin._")
            a("")
            a("</details>")
            a("")

        # -- 3. what came back ----------------------------------------------
        a("## 3. What the model decided, and why")
        a("")
        if r.error:
            a(f"**The model call failed: {r.error}**")
            a("")
            a(
                "No decision was made. An API failure is not a decision to hold — "
                "the tick was skipped and the book left untouched."
            )
            a("")
            return "\n".join(p)

        d = r.decision
        a("### 3a. Market read")
        a("")
        a("> " + str(getattr(d, "market_read", "")).replace("\n", "\n> "))
        a("")

        if r.usage is not None and r.usage.thinking:
            a("### 3b. Reasoning trace (the model's own thinking)")
            a("")
            a(
                "Raw adaptive-thinking output. This is most of what the tick cost, "
                "and it is the only record of the reasoning that the structured "
                "`reasoning` field below compresses."
            )
            a("")
            a("<details><summary>Expand thinking</summary>")
            a("")
            a(fence(r.usage.thinking))
            a("")
            a("</details>")
            a("")
        else:
            a("### 3b. Reasoning trace")
            a("")
            a("_No thinking block was returned on this call._")
            a("")

        a("### 3c. Proposed actions")
        a("")
        actions = list(getattr(d, "actions", []) or [])
        if not actions:
            a("_The model proposed no actions._")
        else:
            a("| Coin | Action | Size | Confidence | Justification |")
            a("| --- | --- | --- | --- | --- |")
            for act in actions:
                reason = str(act.reasoning).replace("\n", " ").replace("|", "\\|")
                a(
                    f"| {act.symbol} | **{act.action}** | {usd(act.size_usd)} "
                    f"| {act.confidence:.2f} | {reason} |"
                )
        a("")

        # -- 4. risk ---------------------------------------------------------
        a("## 4. What the risk rules said")
        a("")
        a("Code decides, the model proposes. A clamp is still an approval.")
        a("")
        if not r.verdicts:
            a("_No verdicts (no actions reached the risk layer)._")
        else:
            a("| Coin | Verdict | Rule | Approved $ | Reason | Notes |")
            a("| --- | --- | --- | --- | --- | --- |")
            for v in r.verdicts:
                notes = "; ".join(v.notes).replace("|", "\\|") or "-"
                reason = (v.reason or "-").replace("\n", " ").replace("|", "\\|")
                a(
                    f"| {v.symbol or '-'} | {'APPROVED' if v.approved else 'REJECTED'} "
                    f"| {v.rule or '-'} | {usd(v.approved_usd)} | {reason} | {notes} |"
                )
        a("")

        # -- 5. fills --------------------------------------------------------
        a("## 5. What actually traded")
        a("")
        if not r.fills:
            a("_No fills this tick._")
        else:
            a(FILL_HEADER)
            for row in fill_rows(list(r.fills)):
                a(row)
            a("")
            a(
                "`Realized P&L` is booked on the SELL that closes a position, "
                "against average cost basis including the fees and gas paid to "
                "open it. A failed transaction fills $0 and still pays gas."
            )
        a("")

        # -- 6. book after ---------------------------------------------------
        a("## 6. The book coming out")
        a("")
        a(book_table(r.portfolio))
        a("")

        # -- 7. cost ---------------------------------------------------------
        a("## 7. What this tick cost")
        a("")
        u = r.usage
        if u is None:
            a("_No usage reported._")
        else:
            a("| Metric | Value |")
            a("| --- | --- |")
            a(f"| Input tokens (uncached) | {u.input_tokens:,} |")
            a(f"| Cache read | {u.cache_read_input_tokens:,} |")
            a(f"| Cache write | {u.cache_creation_input_tokens:,} |")
            a(f"| Total prompt | {u.total_input_tokens:,} |")
            a(f"| Cache hit rate | {u.cache_hit_rate:.1%} |")
            a(f"| Output tokens | {u.output_tokens:,} |")
            a(f"| Cost this tick | ${cost:.4f} |")
            a(f"| Cost run-to-date | ${self.cost:.4f} |")
        a("")
        return "\n".join(p)

    # -- summary -----------------------------------------------------------

    def write_summary(self) -> None:
        now = time.time()
        elapsed = (now - self.started) / 3600.0
        remaining = max(0.0, (self.deadline - now) / 3600.0)
        s = self.last_state
        done = self.deadline_stop or self.user_stop or remaining <= 0

        sells = [f for f in self.all_fills if f.side == "SELL"]
        wins = [f for f in sells if f.realized_pnl_usd > 0]
        losses = [f for f in sells if f.realized_pnl_usd < 0]
        failed = [f for f in self.all_fills if f.failed]

        p: list[str] = []
        a = p.append
        a("# Run summary")
        a("")
        a(f"**Status:** {'COMPLETE' if done else 'RUNNING'}  ")
        a(f"**Started:** {utc(self.started)} ({local(self.started)})  ")
        a(f"**Deadline:** {utc(self.deadline)} ({local(self.deadline)})  ")
        a(f"**Elapsed:** {elapsed:.2f}h of {self.hours:g}h — {remaining:.2f}h remaining  ")
        a(f"**Last updated:** {utc(now)}")
        a("")
        a("## Cadence")
        a("")
        a("| | |")
        a("| --- | --- |")
        a(f"| Slow ticks (model calls) | {self.slow_n} |")
        a(f"| Fast ticks (book marks) | {self.fast_n} |")
        a(f"| Ticks that failed to reach the model | {self.errors} |")
        a(f"| Supervisor restarts | {self.restarts} |")
        a(f"| Model spend so far | ${self.cost:.4f} |")
        a(
            f"| Projected full-run spend | "
            f"${(self.cost / elapsed * self.hours) if elapsed > 0.05 else 0.0:.2f} |"
        )
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
        a(f"| Total fills | {len(self.all_fills)} |")
        a(f"| Closing sells | {len(sells)} |")
        a(f"| Winners | {len(wins)} |")
        a(f"| Losers | {len(losses)} |")
        a(
            f"| Hit rate | "
            f"{(100.0 * len(wins) / len(sells)) if sells else 0.0:.1f}% |"
        )
        a(f"| Failed transactions (gas paid, nothing filled) | {len(failed)} |")
        if wins:
            a(f"| Average winner | {usd(sum(f.realized_pnl_usd for f in wins) / len(wins))} |")
        if losses:
            a(f"| Average loser | {usd(sum(f.realized_pnl_usd for f in losses) / len(losses))} |")
        a("")
        if self.all_fills:
            a(FILL_HEADER)
            for row in fill_rows(self.all_fills):
                a(row)
        else:
            a("_No trades yet._")
        a("")
        a("## Where everything is")
        a("")
        a("| File | What it holds |")
        a("| --- | --- |")
        a("| `ticks/tick-NNNN-*.md` | One per model call: the exact prompt, the full evidence bundle, the reasoning trace, the actions and their justifications, risk verdicts, fills, book, cost |")
        a("| `SYSTEM-PROMPT.md` | The cached system prompt, verbatim, captured once |")
        a("| `TRADES.md` | Every fill in order, with price, quantity, fees, gas and realized P&L |")
        a("| `FAST-TICKS.md` | The 60-second equity curve and stop-loss enforcement |")
        a("| `EVENTS.md` | Errors, restarts, stop-losses — anything that was not a clean tick |")
        a("| `../../data/decisions.jsonl` | The machine-readable decision log the trader itself writes |")
        a("| `../../data/trades.jsonl` | The machine-readable trade ledger |")
        a("")
        self._write("SUMMARY.md", "\n".join(p) + "\n")


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    cfg = config.load()
    install_prompt_capture()

    root = Path(__file__).resolve().parent.parent
    out = args.out or (
        root / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    out.mkdir(parents=True, exist_ok=True)

    rec = Recorder(cfg, out, args.hours)
    log.info("documenting a %.2fh run into %s", args.hours, out)
    rec.event(
        f"documenting {args.hours:g}h; slow tick every "
        f"{cfg.cadence.slow_tick_seconds}s, fast tick every "
        f"{cfg.cadence.fast_tick_seconds}s; "
        f"{cfg.model.name} effort={cfg.model.effort}; "
        f"book ${cfg.starting_cash_usd:,.0f}"
    )

    def on_tick(result: TickResult) -> None:
        rec.on_tick(result)
        if time.time() >= rec.deadline:
            rec.deadline_stop = True
            trader = getattr(on_tick, "trader", None)
            if trader is not None:
                trader._stop = True

    # The supervisor. ``Trader.run`` already isolates a failure in any one
    # evidence stream and already survives a model outage, so reaching here
    # means something it does not handle — a socket the http client did not
    # wrap, an OS hiccup, a bug. The deadline is the only thing that ends this
    # loop, other than the operator.
    while time.time() < rec.deadline and not rec.user_stop:
        try:
            with Trader(cfg) as trader:
                on_tick.trader = trader  # type: ignore[attr-defined]
                trader.run(on_tick=on_tick)
            # A clean return with time left and no deadline stop means the
            # trader's own signal handler fired: the operator asked to stop.
            if not rec.deadline_stop:
                rec.user_stop = True
                rec.event("stopped by operator (SIGINT/SIGTERM); book is saved")
        except KeyboardInterrupt:
            rec.user_stop = True
            rec.event("interrupted by operator; book is saved")
        except Exception:
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

    rec.deadline_stop = rec.deadline_stop or time.time() >= rec.deadline
    rec.event(
        "run finished: "
        + ("deadline reached" if rec.deadline_stop else "stopped early by operator")
    )
    rec.write_summary()
    log.info("done — %s", out / "SUMMARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
