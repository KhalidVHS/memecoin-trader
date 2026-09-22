"""CLI entry point: check | status | once | run | report | reset.

Two things in here are audit consequences rather than taste.

**``--dry-run`` is gone.** It was a boolean threaded through the loop and
checked at some call sites and not others, which is how C12 happened: a
"dry" run could still liquidate a position through the stop-loss path, because
that path never looked at the flag. The replacement is ``--mode``, which
selects an :class:`ExecutionMode` that the *broker* enforces — a READ_ONLY
broker raises on any mutation rather than relying on every caller to remember.
A flag you must check everywhere will eventually not be checked somewhere; a
capability you do not have cannot be forgotten.

**Startup can refuse.** If the persisted record has an order whose outcome is
unknown, ``once`` and ``run`` stop with a non-zero exit instead of trading on
top of it. That is deliberately inconvenient. The alternative is a process that
places an order, dies, restarts and places it again.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import time
from collections.abc import Sequence
from pathlib import Path

import typer
from rich.logging import RichHandler

from . import backfill as backfill_mod
from . import cli_backtest, market, portfolio, report
from . import config as config_mod
from .broker import LiveModeUnsupported
from .config import Config
from .http import make_client
from .loop import StartupRefusal, TickResult, Trader
from .report import console
from .types import (
    ExecutionMode,
    Forecast,
    MarketSnapshot,
    PortfolioState,
    TargetPosition,
    Timeframe,
)

app = typer.Typer(
    add_completion=False,
    help="A deterministic strategy (optionally advised by Claude) trading three "
    "Solana memecoins against a paper book.",
    no_args_is_help=True,
)

# Mounted as a sub-app rather than shipped as its own console script: the
# backtester is another way to run this same strategy, so `memetrader backtest
# run` keeps one entry point and one place to discover what the tool can do.
# ``add_typer`` needs the sub-app object at registration time, so this import
# cannot be deferred — every `memetrader` invocation pays for importing the
# replay stack. That is a real cost, accepted here because it is import-time
# only (no I/O, no config read) and the alternative — a second console script —
# splits the tool's surface in two to save it.
app.add_typer(
    cli_backtest.app,
    name="backtest",
    help="Replay a strategy against recorded history.",
)

_MODE_HELP = (
    "read_only = decide and show everything, mutate nothing. "
    "paper = trade the simulated book. "
    "live = refused; there is no wallet. "
    "Defaults to config.toml's execution_mode."
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _load(mode: str | None = None) -> Config:
    try:
        cfg = config_mod.load()
    except config_mod.ConfigError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(1) from exc
    if mode is None:
        return cfg
    try:
        chosen = ExecutionMode(mode.lower())
    except ValueError as exc:
        allowed = ", ".join(m.value for m in ExecutionMode)
        console.print(f"[red]--mode must be one of:[/red] {allowed}")
        raise typer.Exit(1) from exc
    return _with_mode(cfg, chosen)


def _with_mode(cfg: Config, mode: ExecutionMode) -> Config:
    """The one place a loaded Config is ever changed at runtime.

    Narrow on purpose: `--mode` is the only override the CLI offers, so this
    takes an ExecutionMode rather than arbitrary keywords. A general
    `**changes` helper would make "what can the command line change about the
    config?" a question you answer by reading every call site.
    """
    return dataclasses.replace(cfg, execution_mode=mode)


def _trader(cfg: Config, *, preflight: bool = True) -> Trader:
    try:
        trader = Trader(cfg)
    except LiveModeUnsupported as exc:
        # A refusal, not a crash. The broker raises this in its constructor so
        # that `--mode live` cannot get as far as building a book; catching it
        # here is what turns it into a message an operator can read instead of
        # a traceback that looks like the system fell over.
        console.print(f"[red]refusing to run:[/red] {exc}")
        raise typer.Exit(2) from exc
    if not preflight:
        return trader
    try:
        for note in trader.preflight():
            console.print(f"[yellow]recovered:[/yellow] {note}")
    except StartupRefusal as exc:
        trader.close()
        console.print(f"[red]refusing to trade:[/red] {exc}")
        raise typer.Exit(2) from exc
    return trader


@app.command()
def status(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Every evidence stream for every coin, plus the book. No strategy call."""
    _setup_logging(verbose)
    cfg = _load("read_only")
    with _trader(cfg, preflight=False) as trader:
        snap = trader.snapshot()
        now = time.time()
        evidence = trader.evidence(snap)
        report.render_evidence(console, evidence.values())
        _render_safety(snap, cfg, now=now)
        _render_book(trader.book(snap, now=now), cfg)
    console.print(
        f"[dim]oldest observation {snap.oldest_age_seconds(time.time()):.1f}s old[/dim]"
    )


@app.command()
def once(
    mode: str = typer.Option(None, "--mode", help=_MODE_HELP),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """One decision cycle: evidence -> strategy -> risk -> quote -> execute."""
    _setup_logging(verbose)
    cfg = _load(mode)
    with _trader(cfg) as trader:
        result = trader.slow_tick()
    _render_tick(result, cfg)
    if result.error:
        raise typer.Exit(1)


@app.command()
def run(
    max_ticks: int = typer.Option(
        None, "--max-ticks", help="Stop after this many decision cycles."
    ),
    mode: str = typer.Option(None, "--mode", help=_MODE_HELP),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """The dual-cadence loop. Ctrl+C is safe — state is saved after every change."""
    _setup_logging(verbose)
    cfg = _load(mode)
    console.print(
        f"[bold]memetrader[/bold] — {', '.join(cfg.symbols)} · "
        f"strategy [bold]{cfg.strategy_kind}[/bold] · mode "
        f"[bold]{cfg.execution_mode.value}[/bold] · "
        f"decide every {cfg.cadence.slow_tick_seconds // 60}m, "
        f"mark every {cfg.cadence.fast_tick_seconds}s. Ctrl+C to stop."
    )
    if cfg.strategy_kind == "advisory":
        # The one place an operator is told, at the moment it matters, that the
        # component with no measured predictive value is switched on.
        console.print(
            "[yellow]advisory (LLM) strategy is enabled — audit C6 records that this "
            "component has never been shown to predict returns. Sizing stays flat and "
            "risk still bounds every order.[/yellow]"
        )

    def on_tick(result: TickResult) -> None:
        if result.kind == "slow":
            _render_tick(result, cfg)
        else:
            _render_mark(result)

    with _trader(cfg) as trader:
        trader.run(max_slow_ticks=max_ticks, on_tick=on_tick)
    console.print("[dim]stopped. state saved.[/dim]")


@app.command(name="report")
def report_cmd(
    trades: int = typer.Option(20, "--trades", help="How many fills to show."),
    decisions: int = typer.Option(5, "--decisions", help="How many decisions to show."),
    live_marks: bool = typer.Option(
        True,
        "--live-marks/--no-live-marks",
        help="Fetch current marks. Off makes the report fully offline.",
    ),
) -> None:
    """P&L, trade history, decision history, integrity and spend to date."""
    cfg = _load("read_only")
    now = time.time()
    state = None
    if live_marks:
        try:
            with _trader(cfg, preflight=False) as trader:
                # Prices only. `report` renders no technicals, and skipping
                # candles keeps it from competing for GeckoTerminal's rate limit
                # with a `run` loop in another terminal.
                state = trader.book(trader.snapshot(with_candles=False), now=now)
        except Exception as exc:
            console.print(f"[yellow]live marks unavailable ({exc})[/yellow]")
    if state is None:
        # Offline: the book is marked against nothing, so every open position is
        # unmarkable and the total is None. That renders as "unavailable", which
        # is the honest answer — not a stale number dressed up as a current one.
        state = _offline_book(cfg, now=now)
    rendered = report.load_report(
        cfg, state=state, now=now, fill_limit=trades, decision_limit=decisions
    )
    report.render_report(console, rendered)


@app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    """Wipe data/ back to the starting cash. Irreversible."""
    cfg = _load()
    candidates = [
        cfg.state_path,
        cfg.trades_path,
        cfg.intents_path,
        cfg.decisions_path,
        cfg.ledger_path,
        cfg.data_dir / "risk_ledger.json",
        cfg.data_dir / "sentiment_cache.json",
    ]
    paths = [p for p in candidates if p.is_file()]
    if not paths:
        console.print("[dim]nothing to reset[/dim]")
        return
    console.print("This deletes the whole record — book, ledger, risk history:")
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
    """Validate config, confirm every data source answers, screen every token."""
    cfg = _load("read_only")
    console.print(
        f"[green]config ok[/green]  {len(cfg.coins)} coins, "
        f"${cfg.starting_cash_usd:,.2f} book, strategy={cfg.strategy_kind}, "
        f"mode={cfg.execution_mode.value}"
    )
    if cfg.strategy_kind == "advisory" and not cfg.anthropic_api_key:
        console.print(
            "[red]ANTHROPIC_API_KEY is not set[/red] — the advisory strategy will fail"
        )

    try:
        pools = market.resolve_pairs(cfg.coins, cfg.market)
    except Exception as exc:
        console.print(f"  [red]market data failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    for sym, pool in pools.items():
        trust = "" if pool.trusted_quote else " [yellow](untrusted quote token)[/yellow]"
        console.print(
            f"  [green]ok[/green] {sym:<7} {pool.dex_id} {pool.pair_address}{trust}"
        )

    missing = sorted(set(cfg.symbols) - set(pools))
    if missing:
        console.print(f"  [red]no pool resolved for:[/red] {', '.join(missing)}")

    try:
        snap = market.snapshot(cfg.coins, cfg.market, with_candles=False)
        _render_safety(snap, cfg, now=time.time())
    except Exception as exc:
        console.print(f"  [yellow]safety screen skipped:[/yellow] {exc}")

    console.print(
        "[dim]reddit credentials: "
        + ("present" if cfg.sentiment.has_reddit_credentials else "absent (Arctic Shift)")
        + "  ·  jupiter key: "
        + ("present" if cfg.data.jupiter_api_key else "absent (lite-api)")
        + "  ·  sentiment stream: "
        + ("enabled" if cfg.sentiment.enabled else "disabled")
        + "[/dim]"
    )


def _offline_book(cfg: Config, *, now: float) -> PortfolioState:
    """The book with no marks at all.

    Used when `report --no-live-marks` runs. Every open position lands in
    ``unmarkable`` and ``total_value_usd`` is None, which is exactly right: an
    offline report knows the cash and the token balances and does not know what
    the tokens are worth. C8 is the rule that this must not be papered over with
    cost basis or a cached price.
    """
    from .broker import LocalPaperBroker

    broker = LocalPaperBroker(cfg, mode=ExecutionMode.READ_ONLY)
    broker.load()
    return portfolio.mark_book(
        cash_usd=broker.cash_usd,
        positions=broker.get_positions(),
        marks={},
        realized_pnl_usd=broker.realized_pnl_usd,
        starting_cash_usd=broker.starting_cash_usd,
        fees_paid_usd=broker.fees_paid_usd,
        gas_paid_usd=broker.gas_paid_usd + broker.failed_gas_usd,
        now=now,
    )


def _render_book(book: PortfolioState, cfg: Config) -> None:
    """The book, through ``report``'s own summary builder.

    Routed through ``build_sample`` / ``build_run_summary`` rather than printed
    directly so that the sample size and its caveat travel with the money here
    too. C1 was performance numbers presented without the n that makes them
    meaningless, and a second rendering path is how that reappears.
    """
    sample = report.build_sample(fills=(), decisions=0, now=book.ts)
    summary = report.build_run_summary(
        state=book,
        sample=sample,
        stop_loss_pct=cfg.risk.stop_loss_pct,
        mode=cfg.execution_mode,
    )
    report.render_book(console, summary)


def _render_safety(snap: MarketSnapshot, cfg: Config, *, now: float) -> None:
    """The fail-closed screen, with vetoes and unknowns kept apart.

    A veto is "we looked and this token fails the rule". An unknown is "we could
    not find out". They print differently because they mean different things and
    the audit's whole "missing is never zero" thread is that collapsing them is
    what lets a blind spot read as a pass.
    """
    console.print("\n[bold]safety screen[/bold]")
    for symbol, coin in sorted(snap.coins.items()):
        verdict = market.screen(coin, cfg.safety, now=now)
        if verdict.eligible:
            mark = "[green]eligible[/green]"
        elif verdict.vetoes:
            mark = "[red]vetoed[/red]"
        else:
            mark = "[yellow]unknown[/yellow]"
        console.print(f"  {symbol:<7} {mark}  {verdict.reason}")
        for note in verdict.notes:
            console.print(f"    [dim]{note}[/dim]")


def _render_forecasts(forecasts: Sequence[Forecast]) -> None:
    if not forecasts:
        return
    console.print("[bold]forecasts[/bold]")
    for f in forecasts:
        # The interval, not just the point estimate. A +2% expectation spanning
        # -18% to +22% is not the same claim as one spanning +1% to +3%, and
        # printing only the middle number erases that distinction.
        band = (
            f"[{f.lower_quantile_pct:+.1f}%, {f.upper_quantile_pct:+.1f}%]"
            if f.lower_quantile_pct is not None and f.upper_quantile_pct is not None
            else "[interval unavailable]"
        )
        missing = ""
        if f.features_missing:
            missing = f"  [yellow]missing: {', '.join(f.features_missing)}[/yellow]"
        console.print(
            f"  {f.symbol:<7} {f.expected_net_return_pct:+.2f}% net over "
            f"{f.horizon_seconds / 60:.0f}m  {band}{missing}"
        )


def _render_targets(targets: Sequence[TargetPosition], book: PortfolioState) -> None:
    """Targets against what is actually held.

    Shown as a diff because a target is not an order: the loop trades the gap
    between the two, and an operator reading "BONK $25" with no idea what is
    already held cannot tell whether that means buy, sell or do nothing.
    """
    if not targets:
        console.print("[dim]no targets — flat is a position[/dim]")
        return
    console.print("[bold]targets[/bold]")
    for t in targets:
        # Three states, not two. A symbol missing from `position_values_usd`
        # is either not held at all — worth exactly $0.00, and a target above
        # it is an ordinary buy — or held and unpriceable, which is the only
        # one that blocks sizing. Collapsing the first into the second printed
        # "cannot size" against a flat book and made every target look stuck.
        held = book.position_values_usd.get(t.symbol)
        if held is not None:
            held_s, delta_s = f"${held:,.2f}", f"{t.target_usd - held:+,.2f}"
        elif t.symbol in book.unmarkable:
            held_s, delta_s = "unmarkable", "[yellow]no trade — cannot size[/yellow]"
        else:
            held_s, delta_s = "$0.00", f"{t.target_usd:+,.2f}"
        console.print(
            f"  {t.symbol:<7} target ${t.target_usd:>8,.2f}  held {held_s:>12}  "
            f"delta {delta_s}"
        )
        if t.rationale:
            console.print(f"    [dim]{t.rationale}[/dim]")


def _render_mark(result: TickResult) -> None:
    book = result.portfolio
    # total_value_usd is None when any position could not be marked. Rendering
    # it as $0.00 or omitting the unmarkable leg would be the C8 failure in the
    # display layer: a number that looks like a valuation but is not one.
    if book.total_value_usd is None:
        console.print(
            f"[yellow]{time.strftime('%H:%M:%S')}  mark  unavailable — "
            f"{', '.join(book.unmarkable)} cannot be priced[/yellow]"
        )
    else:
        ret = book.total_return_pct
        ret_s = "n/a" if ret is None else f"{ret:+.2f}%"
        console.print(
            f"[dim]{time.strftime('%H:%M:%S')}  mark  "
            f"${book.total_value_usd:,.2f}  ({ret_s})[/dim]"
        )
    for sym in result.stop_exits:
        console.print(f"[red]  STOP-LOSS exited {sym}[/red]")
    if result.risk_state.halted:
        console.print(f"[red]  HALTED: {'; '.join(result.risk_state.halt_reasons)}[/red]")


def _render_tick(result: TickResult, cfg: Config) -> None:
    for note in result.notes:
        console.print(f"[yellow]{note}[/yellow]")
    if result.error:
        console.print(f"[red]tick failed:[/red] {result.error}")
        console.print("[dim]no decision was made — this is not a decision to hold[/dim]")
        return

    report.render_evidence(console, result.evidence.values())

    state = result.risk_state
    if state.halted:
        console.print(f"[red]RISK HALTED: {'; '.join(state.halt_reasons)}[/red]")
    if not state.data_health_ok:
        console.print("[yellow]data health degraded — entries are blocked[/yellow]")

    decision = result.decision
    if decision is not None:
        if decision.market_read:
            console.print(f"\n[bold]market read[/bold]\n{decision.market_read}\n")
        _render_forecasts(decision.forecasts)
        _render_targets(decision.targets, result.portfolio)

    for bounds in result.bounds:
        if not bounds.permitted:
            console.print(
                f"[yellow]REJECT[/yellow] {bounds.side.value:<4} {bounds.symbol:<7} "
                f"— {bounds.reason}"
            )
        elif bounds.bypassed_rules:
            console.print(
                f"[yellow]BYPASS[/yellow] {bounds.side.value:<4} {bounds.symbol:<7} "
                f"— {', '.join(bounds.bypassed_rules)}"
            )

    for fill in result.fills:
        style = "red" if fill.failed else ("green" if fill.side.value == "BUY" else "cyan")
        suffix = "  [red]TX FAILED — gas still paid[/red]" if fill.failed else ""
        console.print(
            f"[{style}]{fill.side.value:<4}[/{style}] {fill.symbol:<7} "
            f"${fill.notional_usd:>9,.2f} @ {fill.price_usd:.10g}{suffix}"
        )

    console.print()
    _render_book(result.portfolio, cfg)

    usage = result.usage
    if usage is not None:
        # Through ModelConfig.cost_usd, not a second hand-rolled sum. This call
        # site used to fold cache-creation into input_tokens and so priced it at
        # 1x instead of 1.25x.
        cost = cfg.model.cost_usd(
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_input_tokens,
            usage.cache_creation_input_tokens,
        )
        console.print(
            f"[dim]tokens in {usage.input_tokens:,} out {usage.output_tokens:,} "
            f"cache-read {usage.cache_read_input_tokens:,} "
            f"cache-write {usage.cache_creation_input_tokens:,} · ${cost:.4f}[/dim]"
        )
    if not result.mode.may_mutate:
        console.print(
            f"[yellow]{result.mode.value} — orders were priced and bounded but "
            f"nothing was executed and nothing was written[/yellow]"
        )


@app.command(name="backfill")
def backfill_cmd(
    since: str = typer.Option(
        ..., "--since", help="Earliest bar to keep, YYYY-MM-DD (UTC)."
    ),
    universe: str = typer.Option(
        "universe/solana_memecoins.toml",
        "--universe",
        help="Committed universe file: the definition of the experiment.",
    ),
    timeframes: str = typer.Option(
        "1h,5m", "--timeframes", "-t", help="Comma-separated: 1h, 5m, or both."
    ),
    out: str = typer.Option("history", "--out", help="Output root."),
    resume: bool = typer.Option(
        False, "--resume", help="Skip pairs the manifest already covers."
    ),
    pace: float = typer.Option(
        backfill_mod.DEFAULT_PACE_SECONDS,
        "--pace",
        help="Seconds between calls. 429s begin around 2.1s on the keyless tier.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Download historical OHLCV for backtesting. Writes closed bars only.

    Long-running and resumable: a full 1h+5m pull over ~24 coins is roughly an
    hour of wall clock, almost all of it spent pacing to stay under the keyless
    rate limit. ``--resume`` makes an interrupted run cheap to restart, and the
    manifest is rewritten after every series so a run killed partway through
    still describes exactly the files that exist.
    """
    _setup_logging(verbose)
    cfg = _load("read_only")
    root = Path(out)

    try:
        frames = [Timeframe(v.strip()) for v in timeframes.split(",") if v.strip()]
    except ValueError as exc:
        console.print(f"[red]unsupported timeframe: {exc}[/red]")
        raise typer.Exit(2) from exc
    if not frames:
        console.print("[red]--timeframes is empty[/red]")
        raise typer.Exit(2)

    try:
        start = dt.datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=dt.UTC)
    except ValueError as exc:
        console.print(f"[red]--since must be YYYY-MM-DD: {exc}[/red]")
        raise typer.Exit(2) from exc

    coins = backfill_mod.load_universe(Path(universe))
    console.print(
        f"[dim]{len(coins)} coins x {len(frames)} timeframes since {since} -> {root}[/dim]"
    )

    with make_client(cfg.data.http_timeout_seconds) as client:
        result = backfill_mod.backfill(
            client,
            coins=coins,
            timeframes=frames,
            since=start.timestamp(),
            root=root,
            base_url=cfg.data.geckoterminal_base,
            resume=resume,
            pace_seconds=pace,
            on_progress=lambda label: console.print(f"[dim]  {label}[/dim]"),
        )

    for meta in result.written:
        # Gaps are reported, never closed up: a "20-bar" window that actually
        # spans 26 bars of wall clock has to be visible to whoever reads this.
        note = f" [yellow]{meta.missing_bars} missing[/yellow]" if meta.missing_bars else ""
        console.print(
            f"  {meta.symbol:9s} {meta.timeframe:3s} {meta.rows:>7,} rows "
            f"in {meta.pages} pages{note}"
        )
    if result.skipped:
        console.print(f"[dim]skipped {len(result.skipped)} already-complete[/dim]")
    for label, reason in result.failed:
        console.print(f"[red]  {label}: {reason}[/red]")

    console.print(
        f"[bold]{len(result.written)} written, {len(result.skipped)} skipped, "
        f"{len(result.failed)} failed[/bold]"
    )
    if result.failed and not result.written:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
