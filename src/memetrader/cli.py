"""CLI entry point: run | once | status | report | reset."""

from __future__ import annotations

import logging
import time

import typer
from rich.logging import RichHandler

from . import config as config_mod
from . import market, portfolio, report
from .broker import LocalPaperBroker
from .loop import Trader, TickResult
from .report import console

app = typer.Typer(
    add_completion=False,
    help="Claude Opus 5 trading three Solana memecoins against a paper book.",
    no_args_is_help=True,
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _load():
    try:
        return config_mod.load()
    except config_mod.ConfigError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command()
def status(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """All three evidence streams for every coin. No model call, no cost."""
    _setup_logging(verbose)
    cfg = _load()
    with Trader(cfg) as trader:
        snap = trader.snapshot()
        evidence = trader.evidence(snap)
        report.print_evidence(evidence)
        state = portfolio.mark(trader.broker, trader.marks(snap))
        report.print_portfolio(state, cfg)
    console.print(f"[dim]snapshot age {time.time() - snap.ts:.1f}s[/dim]")


@app.command()
def once(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Decide and show everything, but do not mutate state."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """One full decision cycle: evidence -> model -> risk -> execution."""
    _setup_logging(verbose)
    cfg = _load()
    with Trader(cfg) as trader:
        result = trader.slow_tick(dry_run=dry_run)
    _render_tick(result, cfg)
    if result.error:
        raise typer.Exit(1)


@app.command()
def run(
    max_ticks: int = typer.Option(
        None, "--max-ticks", help="Stop after this many decision cycles."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """The dual-cadence loop. Ctrl+C is safe — state is saved after every change."""
    _setup_logging(verbose)
    cfg = _load()
    console.print(
        f"[bold]memetrader[/bold] — {', '.join(cfg.symbols)} · "
        f"{cfg.model.name} effort={cfg.model.effort} · "
        f"decide every {cfg.cadence.slow_tick_seconds // 60}m, "
        f"mark every {cfg.cadence.fast_tick_seconds}s. Ctrl+C to stop."
    )

    def on_tick(result: TickResult) -> None:
        if result.kind == "slow":
            _render_tick(result, cfg)
        else:
            state = result.portfolio
            console.print(
                f"[dim]{time.strftime('%H:%M:%S')}  mark  "
                f"${state.total_value_usd:,.2f}  ({state.total_return_pct:+.2f}%)[/dim]"
            )
            for sym in result.stop_loss_exits:
                console.print(f"[red]  STOP-LOSS exited {sym}[/red]")

    with Trader(cfg) as trader:
        trader.run(max_slow_ticks=max_ticks, on_tick=on_tick)
    console.print("[dim]stopped. state saved.[/dim]")


@app.command()
def report_(
    trades: int = typer.Option(20, "--trades", help="How many fills to show."),
    decisions: int = typer.Option(5, "--decisions", help="How many decisions to show."),
) -> None:
    """P&L, trade history, decision history, and spend to date."""
    cfg = _load()
    broker = LocalPaperBroker(cfg)
    # Mark against the last known entry prices rather than making a network call
    # — `report` should work offline, on a plane, after the fact.
    marks = {s: p.avg_entry_price_usd for s, p in broker.get_positions().items()}
    try:
        with Trader(cfg) as trader:
            # Prices only. `report` renders no technicals, and skipping candles
            # keeps it from competing for GeckoTerminal's rate limit with a
            # `run` loop in another terminal.
            marks = trader.marks(trader.snapshot(with_candles=False))
    except Exception as exc:
        console.print(f"[yellow]live marks unavailable ({exc}); showing at cost basis[/yellow]")
    report.print_portfolio(portfolio.mark(broker, marks), cfg)
    report.print_trades(cfg, trades)
    report.print_decisions(cfg, decisions)
    report.print_spend(cfg)


# typer derives the command name from the function name; `report_` would become
# "report-". Name it explicitly instead of shadowing the module import.
app.command(name="report")(report_)
app.registered_commands = [c for c in app.registered_commands if c.name != "report_"]


@app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    """Wipe data/ back to the starting cash. Irreversible."""
    cfg = _load()
    paths = [p for p in (cfg.state_path, cfg.trades_path, cfg.decisions_path) if p.is_file()]
    extra = cfg.data_dir / "sentiment_cache.json"
    if extra.is_file():
        paths.append(extra)
    if not paths:
        console.print("[dim]nothing to reset[/dim]")
        return
    console.print("This deletes the whole ledger — trades, decisions and the book:")
    for p in paths:
        console.print(f"  [red]{p}[/red]")
    if not yes and not typer.confirm(f"Reset to ${cfg.starting_cash_usd:,.2f}?"):
        console.print("[dim]cancelled[/dim]")
        raise typer.Exit(1)
    for p in paths:
        p.unlink()
    console.print(f"[green]reset[/green] — book is ${cfg.starting_cash_usd:,.2f} in cash")


@app.command()
def check() -> None:
    """Validate config and confirm every data source answers. Run this first."""
    cfg = _load()
    console.print(f"[green]config ok[/green]  {len(cfg.coins)} coins, "
                  f"${cfg.starting_cash_usd:,.2f} book, {cfg.model.name} effort={cfg.model.effort}")
    if not cfg.anthropic_api_key:
        console.print("[red]ANTHROPIC_API_KEY is not set[/red] — `once` and `run` will fail")
    try:
        pairs = market.resolve_pairs(cfg)
        for sym, pool in pairs.items():
            console.print(f"  [green]ok[/green] {sym:<7} pool {pool}")
    except Exception as exc:
        console.print(f"  [red]market data failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(
        "[dim]reddit credentials: "
        + ("present" if cfg.sentiment.has_reddit_credentials else "absent (using Arctic Shift)")
        + "  ·  jupiter key: "
        + ("present" if cfg.data.jupiter_api_key else "absent (using lite-api)")
        + "[/dim]"
    )


def _render_tick(result: TickResult, cfg) -> None:
    if result.error:
        console.print(f"[red]tick failed:[/red] {result.error}")
        console.print("[dim]no decision was made — this is not a HOLD[/dim]")
        return
    report.print_evidence(result.evidence)
    decision = result.decision
    if decision is not None:
        console.print(f"\n[bold]market read[/bold]\n{decision.market_read}\n")
        for i, a in enumerate(decision.actions):
            v = result.verdicts[i] if i < len(result.verdicts) else None
            style = {"BUY": "green", "SELL": "red"}.get(a.action, "dim")
            line = f"[{style}]{a.action:<4}[/{style}] {a.symbol:<7} ${a.size_usd:>8,.2f}  conf {a.confidence:.2f}"
            if v is not None and not v.approved:
                line += f"  [yellow]REJECTED ({v.rule})[/yellow]"
            elif v is not None and v.notes:
                line += f"  [yellow]clamped → ${v.approved_usd:,.2f}[/yellow]"
            console.print(line)
            console.print(f"[dim]      {a.reasoning}[/dim]")
            if v is not None and not v.approved and v.reason:
                console.print(f"[yellow]      ! {v.reason}[/yellow]")
    console.print()
    report.print_portfolio(result.portfolio, cfg)
    if result.usage:
        u = result.usage
        # Through Usage.cost_usd, not a second hand-rolled sum: this call site
        # used to fold cache-creation into input_tokens and so priced it at 1x
        # instead of 1.25x.
        cost = u.cost_usd(cfg)
        console.print(
            f"[dim]tokens in {u.input_tokens:,} out {u.output_tokens:,} "
            f"cache-read {u.cache_read_input_tokens:,} · ${cost:.4f}[/dim]"
        )
    if result.dry_run:
        console.print("[yellow]dry run — nothing was executed and nothing was logged[/yellow]")


if __name__ == "__main__":
    app()
