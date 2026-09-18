"""Rendering. Everything the user actually looks at.

Nothing here computes anything a trading decision depends on — it reads the
logs and the marked book and formats them. Keeping the arithmetic out of the
display layer is why the numbers in `report` can be trusted to match the ledger.
"""

from __future__ import annotations

import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import journal
from .config import Config
from .types import EvidenceBundle, PortfolioState, Technicals


def _utf8(stream: object) -> None:
    """Force UTF-8 on a stdio stream.

    On Windows, Python picks the ANSI code page (cp1252 here) for a redirected
    stream, and this module renders arrows, em dashes and box characters —
    ``memetrader status | tail`` died with UnicodeEncodeError on a '↓' while
    the same command in a terminal was fine. A display-layer encoding detail
    must never be able to take down a command that reads the book.

    ``errors="replace"`` is the belt to the UTF-8 braces: if a stream cannot be
    reconfigured at all, an unrenderable glyph degrades to '?' instead of an
    exception.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:  # not a TextIOWrapper (pytest capture, a pipe shim)
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):  # already detached, or not reconfigurable
        pass


_utf8(sys.stdout)
_utf8(sys.stderr)

console = Console()


def _pct(value: float | None, digits: int = 2) -> Text:
    """Render a percent, coloured by sign. ``None`` renders as n/a — never as
    0.0, because "not reported" and "flat" are different claims."""
    if value is None:
        return Text("n/a", style="dim")
    style = "green" if value > 0 else "red" if value < 0 else "white"
    return Text(f"{value:+.{digits}f}%", style=style)


def _num(value: float | None, fmt: str = ",.2f") -> str:
    return "n/a" if value is None else format(value, fmt)


def _price(value: float) -> str:
    """Memecoin prices span nine orders of magnitude; a fixed precision either
    prints $0.00 for BONK or a wall of zeros for WIF."""
    if value == 0:
        return "$0"
    if value >= 0.01:
        return f"${value:,.4f}"
    return f"${value:.3e}".replace("e-0", "e-")


def _age(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def _tech_row(t: Technicals) -> str:
    rsi = "n/a" if t.rsi14 is None else f"{t.rsi14:.1f}{'↑' if t.rsi14_rising else '↓' if t.rsi14_rising is False else ''}"
    ema = "n/a" if t.ema9_above_ema21 is None else ("9>21" if t.ema9_above_ema21 else "9<21")
    macd = "n/a" if t.macd_hist is None else f"{t.macd_hist:+.3g}"
    cross = ""
    if t.macd_cross and t.macd_cross != "none":
        cross = f" {t.macd_cross[:4]}@{t.bars_since_cross}"
    bb = "n/a" if t.bb_percent_b is None else f"{t.bb_percent_b:.2f}"
    atr = "n/a" if t.atr14_pct is None else f"{t.atr14_pct:.2f}%"
    vol = "n/a" if t.volume_ratio_20 is None else f"{t.volume_ratio_20:.2f}x"
    return (
        f"RSI {rsi}  EMA {ema}  MACD {macd}{cross}  %B {bb}  ATR {atr}  Vol {vol}"
    )


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

    flow = Text("  liq    ")
    flow.append(f"${c.liquidity_usd:,.0f}")
    if bundle.technicals is not None:
        trend = bundle.technicals.flow.liquidity_trend_pct
        flow.append("  trend ")
        flow.append(_pct(trend))
        flow.append(f"   turnover24h {bundle.technicals.flow.turnover_24h:.2f}x")
    lines.append(flow)

    txn = Text("  flow   ")
    txn.append(f"buy/sell m5 {c.txns_m5.ratio:.2f}  h1 {c.txns_h1.ratio:.2f}  h24 {c.txns_h24.ratio:.2f}")
    lines.append(txn)

    if bundle.technicals is None:
        lines.append(Text("  tech   UNAVAILABLE this tick", style="yellow"))
    else:
        lines.append(Text(f"  5m     {_tech_row(bundle.technicals.m5)}", style="cyan"))
        lines.append(Text(f"  1h     {_tech_row(bundle.technicals.h1)}", style="cyan"))

    s = bundle.sentiment
    if s is None:
        reason = bundle.sentiment_unavailable_reason or "no data"
        lines.append(Text(f"  social UNAVAILABLE — {reason}", style="yellow"))
    else:
        ratio = "n/a" if s.contributor_to_post_ratio is None else f"{s.contributor_to_post_ratio:.2f}"
        z = "n/a" if s.mention_zscore_7d is None else f"{s.mention_zscore_7d:+.2f}"
        v1 = "n/a" if s.mention_velocity_1h is None else f"{s.mention_velocity_1h:.1f}/h"
        v24 = "n/a" if s.mention_velocity_24h is None else f"{s.mention_velocity_24h:.1f}/h"
        line = Text("  social ", style="magenta")
        line.append(
            f"vel1h {v1}  vel24h {v24}  "
            f"z7d {z}  contributors {s.unique_contributors_24h}  ratio {ratio}",
            style="magenta",
        )
        lines.append(line)
        if s.mention_velocity_1h is None:
            lines.append(
                Text(
                    "         ^ vel1h n/a: the source has not indexed this hour — unobserved, not quiet",
                    style="yellow",
                )
            )
        if s.contributor_to_post_ratio is not None and s.contributor_to_post_ratio < 0.5:
            lines.append(
                Text("         ^ few accounts posting a lot — shill-farm signature", style="yellow")
            )
        for p in s.top_posts[:2]:
            lines.append(Text(f"         r/{p.subreddit} [{p.score}] {p.title[:70]}", style="dim"))

    if c.degraded:
        lines.append(Text(f"  ! degraded: {c.degraded_reason}", style="yellow"))

    body = Text("\n").join(lines)
    return Panel(body, title=f"[bold]{c.symbol}[/bold]  {c.dex_id}", title_align="left")


def print_evidence(evidence: dict[str, EvidenceBundle]) -> None:
    for bundle in evidence.values():
        console.print(evidence_panel(bundle))


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


def portfolio_table(state: PortfolioState, cfg: Config) -> Table:
    t = Table(title="Book", title_justify="left", header_style="bold")
    t.add_column("symbol")
    t.add_column("qty", justify="right")
    t.add_column("entry", justify="right")
    t.add_column("mark", justify="right")
    t.add_column("value", justify="right")
    t.add_column("unreal $", justify="right")
    t.add_column("unreal %", justify="right")
    t.add_column("to stop", justify="right")
    t.add_column("age", justify="right")

    stop_pct = cfg.risk.stop_loss_pct * 100
    for sym, pos in sorted(state.positions.items()):
        mark = state.marks.get(sym, pos.avg_entry_price_usd)
        pnl = pos.unrealized_pnl_usd(mark)
        pnl_pct = pos.unrealized_pnl_pct(mark)
        t.add_row(
            sym,
            f"{pos.quantity:,.4g}",
            _price(pos.avg_entry_price_usd),
            _price(mark),
            f"${state.position_values_usd.get(sym, 0.0):,.2f}",
            Text(f"{pnl:+,.2f}", style="green" if pnl >= 0 else "red"),
            _pct(pnl_pct),
            _pct(pnl_pct + stop_pct),
            _age(pos.age_seconds(state.ts)),
        )
    if not state.positions:
        t.add_row("[dim]— flat —[/dim]", "", "", "", "", "", "", "", "")

    t.add_section()
    t.add_row(
        "[bold]cash[/bold]", "", "", "", f"[bold]${state.cash_usd:,.2f}[/bold]", "", "", "", ""
    )
    total = Text(f"${state.total_value_usd:,.2f}", style="bold")
    t.add_row("[bold]total[/bold]", "", "", "", total, "", _pct(state.total_return_pct), "", "")
    return t


def print_portfolio(state: PortfolioState, cfg: Config) -> None:
    console.print(portfolio_table(state, cfg))
    console.print(
        f"  realized P&L [b]${state.realized_pnl_usd:+,.2f}[/b]   "
        f"fees ${state.fees_paid_usd:,.2f}   gas ${state.gas_paid_usd:,.2f}   "
        f"started ${state.starting_cash_usd:,.2f}"
    )


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


def print_trades(cfg: Config, limit: int = 20) -> None:
    rows = journal.tail(cfg.trades_path, limit)
    if not rows:
        console.print("[dim]no trades yet[/dim]")
        return
    t = Table(title=f"Last {len(rows)} fills", title_justify="left", header_style="bold")
    for col in ("when", "side", "symbol", "usd", "price", "impact", "fee", "gas", "realized", ""):
        t.add_column(col, justify="right" if col not in ("side", "symbol", "") else "left")
    now = time.time()
    for r in rows:
        failed = r.get("failed")
        t.add_row(
            _age(now - float(r.get("ts", now))),
            str(r.get("side", "")),
            str(r.get("symbol", "")),
            f"${float(r.get('filled_usd', 0)):,.2f}",
            _price(float(r.get("price_usd", 0))),
            f"{float(r.get('price_impact_pct', 0)):.3f}%",
            f"${float(r.get('pool_fee_usd', 0)):.2f}",
            f"${float(r.get('gas_usd', 0)):.2f}",
            Text(
                f"{float(r.get('realized_pnl_usd', 0)):+,.2f}",
                style="green" if float(r.get("realized_pnl_usd", 0)) >= 0 else "red",
            ),
            Text("TX FAILED", style="red") if failed else (
                Text("degraded", style="yellow") if r.get("degraded") else Text("")
            ),
        )
    console.print(t)


def print_decisions(cfg: Config, limit: int = 10) -> None:
    rows = journal.tail(cfg.decisions_path, limit)
    if not rows:
        console.print("[dim]no decisions yet[/dim]")
        return
    now = time.time()
    for r in rows:
        age = _age(now - float(r.get("ts", now)))
        console.print(f"\n[bold]{age} ago[/bold]  [dim]{r.get('model')} effort={r.get('effort')}[/dim]")
        console.print(Text(f"  {r.get('market_read', '')}", style="italic dim"))
        verdicts = r.get("verdicts") or []
        for i, a in enumerate(r.get("actions") or []):
            v = verdicts[i] if i < len(verdicts) else {}
            act = a.get("action", "?")
            style = {"BUY": "green", "SELL": "red"}.get(act, "dim")
            head = f"  [{style}]{act:<4}[/{style}] {a.get('symbol','?'):<7} ${a.get('size_usd',0):>8,.2f}  conf {a.get('confidence',0):.2f}"
            if not v.get("approved", True):
                head += f"  [yellow]REJECTED ({v.get('rule')})[/yellow]"
            elif v.get("notes"):
                head += f"  [yellow]clamped → ${v.get('approved_usd',0):,.2f}[/yellow]"
            console.print(head)
            console.print(Text(f"        {a.get('reasoning','')}", style="dim"))
            if not v.get("approved", True) and v.get("reason"):
                console.print(Text(f"        ! {v.get('reason')}", style="yellow"))


def print_spend(cfg: Config) -> None:
    """What this has cost so far. You should know from day one, not from the
    invoice."""
    rows = list(journal.read(cfg.decisions_path))
    if not rows:
        console.print("[dim]no model calls yet[/dim]")
        return
    tin = sum(int(r.get("input_tokens", 0)) for r in rows)
    tout = sum(int(r.get("output_tokens", 0)) for r in rows)
    tcache = sum(int(r.get("cache_read_input_tokens", 0)) for r in rows)
    tcreate = sum(int(r.get("cache_creation_input_tokens", 0)) for r in rows)
    cost = cfg.model.cost_usd(tin + tcreate, tout, tcache)
    hit = 100.0 * tcache / (tin + tcache) if (tin + tcache) else 0.0
    per_call = cost / len(rows)
    console.print(
        f"\n[bold]Spend[/bold]  {len(rows)} calls   in {tin:,}  out {tout:,}  "
        f"cache-read {tcache:,} ({hit:.0f}% hit)\n"
        f"  [bold]${cost:,.2f}[/bold] to date   ${per_call:.4f}/call   "
        f"≈ ${per_call * 86400 / cfg.cadence.slow_tick_seconds:,.2f}/day at this cadence"
    )
    if tcache == 0 and len(rows) > 1:
        console.print(
            "[yellow]  ! zero cache reads across multiple calls — the stable prompt "
            "prefix is being invalidated, and you are paying several times over.[/yellow]"
        )
