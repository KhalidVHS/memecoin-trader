"""CLI entry point for the backtest engine: run | resume.

``memetrader.backtest.runner`` is deliberately low-level — it takes a
pre-built :class:`~memetrader.backtest.runner.RunnerInputs` (streams, a
strategy, an execution model) and does not know how to load a catalog or a
config file (see ``runner.py``'s own module docstring: that plumbing is
"catalog- and execution-model-registry" work, explicitly out of scope for the
engine's core). This module is where that plumbing lives for a real,
run-from-the-command-line invocation.

**What "wiring a config into ``RunnerInputs``" concretely means here**, since
nothing in the codebase did this before this file:

* ``[data] timeframes`` + a committed universe file (default
  ``universe/solana_memecoins.toml``, the same file ``cli.py backfill`` reads)
  select which ``(pool, timeframe)`` bar partitions to stream, via
  :func:`memetrader.histdata.loaders.ohlcv.streams_for_catalog`.
* A ``UNIVERSE`` event is synthesised at every point-in-time membership change
  (``UniverseEntry.eligible_from``/``removed_at``), because no universe loader
  exists yet and ``ReplayEngine`` needs at least one ``UNIVERSE`` event to ever
  see a non-empty tradable set.
* A ``POOL_STATE`` event is synthesised once per pool, at the replay start
  (or resume point), carrying only the universe file's own resolution-time
  ``liquidity_usd`` — because no pool-state loader exists either, and without
  one ``EligibilityRisk.assess`` unconditionally vetoes every entry with
  ``no_liquidity_observation`` (see ``_pool_state_event``'s docstring).
* A ``DECISION_TICK`` event is synthesised every ``decision_tick_seconds``,
  because — per ``event_queue.py``'s module docstring and
  ``engine.ReplayEngine``'s low-level construction contract — the engine does
  not inject these itself; the caller building the streams does.
* The strategy is fixed at :class:`~memetrader.strategies.baselines.BuyAndHoldStrategy`
  and the execution model at :class:`~memetrader.execution.fill_models.BarExecutionModel`
  (TIER_0). ``BacktestConfig`` has no ``[strategy]`` section and no execution
  model registry exists, so there is currently no config-driven way to select
  either one; ``BuyAndHoldStrategy`` is the only baseline that needs no extra
  parameters to construct. Widening this is future work, not a CLI concern to
  invent here (see the final report for what a registry would need).

Every event stream this module builds is bounded to
``[config.start_ts, config.end_ts)`` — ``BacktestConfig``'s own contract for
both endpoints — because none of the loaders filter by date themselves. For
``resume``, the lower bound is the snapshot's ``clock_now`` instead of
``config.start_ts``: see ``_build_runner_inputs``'s docstring.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import typer

from memetrader.backtest.config import BacktestConfig, BacktestConfigError
from memetrader.backtest.config import load as load_backtest_config
from memetrader.backtest.runner import RunnerInputs, RunOutputs
from memetrader.backtest.runner import run as run_backtest
from memetrader.backtest.snapshots import SnapshotError
from memetrader.backtest.snapshots import load as load_snapshot
from memetrader.config import ExecutionConfig
from memetrader.execution.fill_models import BarExecutionModel
from memetrader.execution.interfaces import ExecutionModel
from memetrader.histdata.catalog import Catalog
from memetrader.histdata.loaders.ohlcv import interval_seconds, streams_for_catalog
from memetrader.histdata.schemas import PoolState
from memetrader.histdata.universe import UniverseCatalog, UniverseEntry
from memetrader.report import console
from memetrader.risk import RiskEngine, RiskParams
from memetrader.strategies.baselines import BuyAndHoldStrategy
from memetrader.types import (
    NON_EXECUTABLE_NOTICE,
    EventKind,
    HistoricalEvent,
    Timeframe,
    TokenMeta,
)

app = typer.Typer(
    add_completion=False,
    help="Run and resume backtests over the deterministic replay engine "
    "(memetrader.backtest.runner). TIER_0 (OHLCV) only, today.",
    no_args_is_help=True,
)

DEFAULT_HISTORY_ROOT = Path("history")
DEFAULT_UNIVERSE_PATH = Path("universe/solana_memecoins.toml")
DEFAULT_OUTPUT_ROOT = Path("runs/backtests")

# The cash-leg TokenMeta ``BarExecutionModel`` prices every swap against.
# 6 decimals mirrors USDC / the ledger's micro-USD atomic unit convention
# (1 USD == 10**6 units), so an ``OrderIntent.in_amount_atomic`` denominated in
# micro-USD (per BuyAndHoldStrategy's docstring) round-trips through
# ``TokenMeta.to_ui``/``to_atomic`` without a scale error.
_USD_TOKEN = TokenMeta(mint="USD", decimals=6, source="cli_backtest", verified=False)


# ---------------------------------------------------------------------------
# Config loading, with cli.py's error-reporting idiom
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> BacktestConfig:
    try:
        return load_backtest_config(path)
    except BacktestConfigError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(1) from exc
    except tomllib.TOMLDecodeError as exc:
        # backtest.config.load's docstring promises BacktestConfigError "on
        # any structural or semantic error", but its tomllib.load(fh) call is
        # not wrapped in a try/except (config.py:396-398 at the time of
        # writing) — a malformed TOML file therefore raises a raw
        # TOMLDecodeError instead. Reported as a defect rather than fixed
        # here (config.py is owned by another agent); caught here only so
        # this CLI's own error-reporting path stays uniform.
        console.print(f"[red]config error:[/red] {path} is not valid TOML: {exc}")
        raise typer.Exit(1) from exc


def _decision_timeframe(config: BacktestConfig) -> Timeframe:
    """Which OHLCV resolution the strategy/execution model reads.

    ``BacktestConfig`` has no explicit field for this — only ``timeframes``,
    the set of resolutions loaded at all. The finest configured resolution is
    the natural choice: it is what a decision tick every
    ``decision_tick_seconds`` (often finer than 1h) actually has fresh data
    for. Falls back to H1 when 5m was not requested.
    """
    return Timeframe.M5 if Timeframe.M5.value in config.timeframes else Timeframe.H1


# ---------------------------------------------------------------------------
# Synthesised streams: DECISION_TICK and UNIVERSE
# ---------------------------------------------------------------------------


def _bounded(
    events: Iterator[HistoricalEvent], *, start_ts: float, end_ts: float
) -> Iterator[HistoricalEvent]:
    """Drop events outside ``[start_ts, end_ts)``.

    Mirrors ``BacktestConfig.start_ts``/``end_ts``'s own contract ("the first
    event yielded by any loader must have available_time >= start_ts";
    "events after [end_ts] are dropped") — no loader in this codebase applies
    that filter itself, so whoever builds the streams must.
    """
    for event in events:
        if start_ts <= event.available_time < end_ts:
            yield event


def _decision_ticks(
    config: BacktestConfig, *, start_ts: float | None = None
) -> Iterator[HistoricalEvent]:
    """One ``DECISION_TICK`` every ``decision_tick_seconds``, first one tick
    interval after ``config.start_ts`` (never at ``start_ts`` itself).

    Nothing in ``event_queue.py``/``engine.py`` injects these automatically —
    both modules' docstrings describe a ``DECISION_TICK`` arriving as an
    ordinary stream event (or a mid-replay ``push()``), and ``ReplayEngine``'s
    own docstring says construction is intentionally low-level: the caller
    supplies everything. This is that supply, for the CLI's own default wiring.

    The grid deliberately starts at ``config.start_ts + decision_tick_seconds``,
    not at ``config.start_ts``. ``BacktestConfig.start_ts``'s own contract
    ("the first event yielded by any loader must have available_time >=
    start_ts") means no ``BAR_CLOSE`` event can ever be available *at*
    start_ts itself — ``available_time_for`` (histdata/loaders/ohlcv.py) always
    adds a strictly positive ``interval + publication_delay_seconds`` to a
    bar's open time, so the earliest possible bar close is available strictly
    after start_ts. A tick fired at start_ts would therefore *always* find
    ``ReplayEngine._synthesize_snapshot`` returning ``None`` for every symbol,
    by construction, regardless of how much history exists — not a fixable
    data problem, an artifact of checking at the instant zero bars could
    possibly have closed yet. Waiting one tick interval before the first check
    gives the replay a real chance to have ingested at least one bar.

    ``start_ts``, when given (a resume), does not shift the tick grid — the
    grid is always anchored relative to ``config.start_ts`` so a resumed run's
    ticks fall on exactly the same instants a fresh run's would. Ticks
    strictly before ``start_ts`` are simply not yielded.
    """
    floor = config.start_ts if start_ts is None else start_ts
    ts = config.start_ts + config.decision_tick_seconds
    sequence = 0
    while ts < config.end_ts:
        if ts >= floor:
            yield HistoricalEvent(
                kind=EventKind.DECISION_TICK,
                available_time=ts,
                asset_id=None,
                payload=None,
                sequence=sequence,
                source="cli_backtest:decision_tick",
            )
        sequence += 1
        ts += config.decision_tick_seconds


def _universe_events(
    universe: UniverseCatalog, *, start_ts: float, end_ts: float
) -> Iterator[HistoricalEvent]:
    """One ``UNIVERSE`` event per point-in-time membership change.

    ``ReplayState.universe()`` returns the members of the most recent
    ``UNIVERSE`` event at or before ``now`` (empty set if none has arrived
    yet), so at least one event at ``start_ts`` is required for the replay to
    ever see a non-empty universe. Each payload is the *full* eligible set at
    that instant (``UniverseCatalog.eligible_at``), not a delta — that is what
    ``engine._ingest_universe`` expects (it calls ``state.set_universe``
    directly with whatever collection the payload is).
    """
    transitions = {start_ts}
    for entry in universe.entries:
        if start_ts < entry.eligible_from < end_ts:
            transitions.add(entry.eligible_from)
        if entry.removed_at is not None and start_ts < entry.removed_at < end_ts:
            transitions.add(entry.removed_at)
    for ts in sorted(transitions):
        yield HistoricalEvent(
            kind=EventKind.UNIVERSE,
            available_time=ts,
            asset_id=None,
            payload=universe.eligible_at(ts),
            source="cli_backtest:universe",
        )


def _pool_state_event(entry: UniverseEntry, *, available_time: float) -> HistoricalEvent:
    """One synthetic ``POOL_STATE`` event carrying the universe file's own
    (resolution-time, not historical) liquidity figure for ``entry``.

    Without this, ``EligibilityRisk.assess`` (risk.py) unconditionally vetoes
    every entry with ``no_liquidity_observation`` — ``snapshot.liquidity_usd
    is None`` is checked before ``RiskParams.min_liquidity_usd`` is even
    consulted, so there is no threshold to relax around a missing
    observation. ``engine._synthesize_snapshot`` only ever gets a
    ``liquidity_usd`` from ``ReplayState.pool_state(pool_id)``, and no
    pool-state loader exists in this codebase (only the OHLCV loader does) —
    so a TIER_0 OHLCV-only backtest could never clear that veto for any
    symbol, ever, without this event. ``reserve_*_atomic``/``fee_rate_bps``
    are left at 0: unused here, since ``BarExecutionModel`` (the only
    execution model this CLI wires) prices fills from OHLCV bars, never from
    AMM reserves. ``price_usd`` is left ``None`` so ``_synthesize_snapshot``
    keeps using the bar's own close as the traded price rather than a second,
    potentially-inconsistent price source.
    """
    return HistoricalEvent(
        kind=EventKind.POOL_STATE,
        available_time=available_time,
        asset_id=entry.asset_id,
        payload=PoolState(
            asset_id=entry.asset_id,
            pool_id=entry.pool_id,
            venue=entry.dex,
            event_time=available_time,
            available_time=available_time,
            received_time=available_time,
            reserve_in_atomic=0,
            reserve_out_atomic=0,
            fee_rate_bps=0,
            price_usd=None,
            liquidity_usd=entry.liquidity_usd,
            source="cli_backtest:universe_liquidity",
        ),
        source="cli_backtest:pool_state",
    )


def _pool_state_events(
    universe: UniverseCatalog, *, start_ts: float
) -> Iterator[HistoricalEvent]:
    """One :func:`_pool_state_event` per universe entry with a known
    liquidity figure, anchored at ``start_ts`` (``floor_ts`` on resume — see
    ``_universe_events``, which this mirrors) rather than emitted throughout
    the replay: this is a single static observation, not a fabricated
    historical liquidity series.
    """
    for entry in universe.entries:
        if entry.liquidity_usd is not None:
            yield _pool_state_event(entry, available_time=start_ts)


# ---------------------------------------------------------------------------
# RunnerInputs assembly
# ---------------------------------------------------------------------------


def _build_runner_inputs(
    config: BacktestConfig,
    *,
    history_root: Path,
    universe_path: Path,
    stream_start_ts: float | None = None,
) -> RunnerInputs:
    """Turn a loaded config into everything ``runner.run`` needs.

    ``stream_start_ts``, when given, replaces ``config.start_ts`` as the lower
    bound on every generated stream — used only by the ``resume`` command,
    which passes the snapshot's ``clock_now``. This matters because
    ``SimulatedClock.advance_to`` refuses to move backwards (see
    ``clock.py``): rebuilding fresh streams from ``config.start_ts`` for a
    resumed run would hand the engine events earlier than the clock it is
    about to restore, which is exactly the "clock moves backwards" case the
    clock raises ``ClockError`` for. Bounding from the snapshot's own
    ``clock_now`` instead means the first event a resumed engine ever sees is
    at or after where it left off.

    Raises ``BacktestConfigError`` (not a bare ``FileNotFoundError``) when the
    universe file is missing, so the CLI's single error-reporting path in
    ``_load_config``-style callers can catch one exception type.
    """
    if not universe_path.exists():
        raise BacktestConfigError(f"universe file not found: {universe_path}")
    universe = UniverseCatalog.from_toml(universe_path)
    catalog = Catalog(root=history_root)
    pool_to_asset = universe.pool_to_asset()
    floor_ts = config.start_ts if stream_start_ts is None else stream_start_ts

    streams: list[Iterator[HistoricalEvent]] = []
    for tf_str in config.timeframes:
        try:
            timeframe = Timeframe(tf_str)
        except ValueError as exc:
            raise BacktestConfigError(
                f"[data] timeframes contains unsupported value: {tf_str!r}"
            ) from exc
        streams.extend(
            _bounded(stream, start_ts=floor_ts, end_ts=config.end_ts)
            for stream in streams_for_catalog(
                catalog,
                pool_to_asset=pool_to_asset,
                publication_delay_seconds=config.publication_delay_seconds,
                timeframe=timeframe,
            )
        )

    # _universe_events anchors its own full-membership snapshot at start_ts —
    # passing floor_ts (not config.start_ts) here means a resumed run gets a
    # correct anchor snapshot at the point it resumes from, rather than
    # relying on _bounded to drop the true start_ts anchor and hope a later
    # transition arrives before anything asks ReplayState.universe().
    streams.append(_universe_events(universe, start_ts=floor_ts, end_ts=config.end_ts))
    # Same anchoring rationale as _universe_events: one static liquidity
    # observation per pool, dated at floor_ts so a resume still has it.
    streams.append(_pool_state_events(universe, start_ts=floor_ts))
    # _decision_ticks keeps the grid anchored at config.start_ts even when
    # floor_ts differs (a resume) — see its own docstring for why.
    streams.append(_decision_ticks(config, start_ts=floor_ts))

    decision_timeframe = _decision_timeframe(config)
    execution_config = ExecutionConfig(
        slippage_bps_fallback=50.0,
        gas_usd_per_swap=0.21,
        failed_tx_rate=config.latency.failed_tx_rate,
        default_pool_fee_pct=0.25,
        pool_fee_pct={},
    )
    execution_model: ExecutionModel = BarExecutionModel(
        execution_config, usd_token=_USD_TOKEN, timeframe=decision_timeframe
    )

    # RiskEngine's own default (RiskParams.universe=frozenset()) means
    # "nothing is eligible" (risk.py's own docstring: "Empty means nothing
    # is eligible, not no restriction" — a deliberate fail-closed default).
    # It is a *separate* static allow-list from ReplayState.universe()'s
    # point-in-time membership, and nothing wires one from the other
    # automatically, so every mint this universe file ever lists must be
    # supplied here or every entry order is vetoed with "not_in_universe"
    # regardless of what the point-in-time universe stream says.
    #
    # Two more defaults need relaxing for TIER_0 to ever clear an entry at
    # all, for reasons that are structural to this engine, not bugs in this
    # file's own wiring:
    #  * require_known_pool_age=True (RiskParams' default) vetoes with
    #    "unknown_pool_age" whenever CoinSnapshot.pool.created_at is None —
    #    and engine._synthesize_snapshot's PoolRef is always built with no
    #    created_at (there is nowhere in this engine for it to come from), so
    #    this veto would otherwise fire unconditionally, forever.
    #  * require_volatility_estimate=True (RiskParams' default) vetoes with
    #    "no_volatility_estimate" whenever entry_bounds() is not given a
    #    volatility_pct — and engine._route_entry never passes one to
    #    broker.attempt_entry (see engine.py's own call site), so this veto
    #    would also otherwise fire unconditionally, forever.
    # max_snapshot_age_seconds' default (90s) assumes near-real-time market
    # data; CoinSnapshot.provenance.event_time here is a bar's *open* time
    # (engine._synthesize_snapshot), so its age against "now" is naturally on
    # the order of one decision tick plus one bar interval plus the
    # publication delay — raised generously (2x, plus a fixed cushion) so a
    # tier built entirely on slow OHLCV bars is not held to a live-feed
    # staleness bar it cannot ever meet.
    bar_interval = interval_seconds(decision_timeframe)
    max_snapshot_age_seconds = 2.0 * (
        config.decision_tick_seconds + bar_interval + config.publication_delay_seconds
    )
    risk_engine = RiskEngine(
        params=RiskParams(
            universe=frozenset(e.asset_id for e in universe.entries),
            require_known_pool_age=False,
            require_volatility_estimate=False,
            max_snapshot_age_seconds=max_snapshot_age_seconds,
        )
    )

    return RunnerInputs(
        config=config,
        streams=streams,
        strategy=BuyAndHoldStrategy(),
        execution_model=execution_model,
        risk_engine=risk_engine,
        decision_timeframe=decision_timeframe,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_outputs(outputs: RunOutputs, *, config: BacktestConfig) -> None:
    result = outputs.result
    # EngineResult.equity_curve is (ts, equity_usd) — plain USD, not
    # micro-USD (see engine._route_entry / _mark_book: book.total_value_usd
    # is built from BrokerBook.cash_usd, already dollars) — unlike the
    # ledger's *_micro_usd fields, this one needs no scaling.
    final_equity = result.equity_curve[-1][1] if result.equity_curve else None
    equity_s = f"${final_equity:,.2f}" if final_equity is not None else "n/a"
    console.print(
        f"[green]run {outputs.run_id}[/green]  {result.ticks_checked} ticks  "
        f"final equity {equity_s}"
    )
    console.print(f"  manifest  {outputs.manifest_path}")
    console.print(f"  ledger    {outputs.ledger_path}")
    console.print(f"  metrics   {outputs.metrics_path}")
    if not config.fidelity_tier.permits_pnl_claim:
        # Contracts §3: any report from a sub-TIER_2 run must print this
        # verbatim. Printed here rather than left to a downstream report
        # generator so the notice cannot be dropped by never calling one.
        # soft_wrap=True: rich's default wrapping would break the sentence
        # across lines depending on terminal width, which would make a
        # substring check for the "verbatim" text fail non-deterministically.
        console.print(f"[yellow]{NON_EXECUTABLE_NOTICE}[/yellow]", soft_wrap=True)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

_HISTORY_ROOT_HELP = "Root of the backfilled bar tree (<root>/<pool>/<timeframe>.jsonl.gz)."
_UNIVERSE_HELP = "Committed universe file: the point-in-time tradable set."
_OUTPUT_DIR_HELP = "Where manifest.json/ledger.json/metrics.json are written."
_RUN_ID_HELP = "Override the generated run ID."
_SNAPSHOT_TO_HELP = "Write a resumable snapshot here, every --snapshot-interval-ticks."
_SNAPSHOT_INTERVAL_HELP = "Ticks between snapshot writes (and always on the last tick)."
_EXPERIMENT_ID_HELP = "Manifest experiment_id. Defaults to the config's [run] label."
_TRIAL_NUMBER_HELP = "Manifest trial_number, for a sweep of the same experiment_id."

# `Annotated[..., typer.Option(...)]` rather than `= typer.Option(...)` as the
# parameter default: ruff's B008 (function-call-in-default-argument) only
# auto-exempts typer's own Option/Argument calls when the annotation is one of
# a handful of builtin immutable types (bool/str/int) — cli.py's options are
# all one of those and so never trip it, but every option below is `Path` or
# `Path | None`, which does. `Annotated` moves the call out of the default
# position entirely (the default becomes a plain value again), which is both
# ruff-clean and the typer-recommended style for exactly this case.
_ConfigArg = Annotated[Path, typer.Argument(help="Path to a backtest TOML config.")]
_OutputDirOpt = Annotated[Path | None, typer.Option("--output-dir", help=_OUTPUT_DIR_HELP)]
_HistoryRootOpt = Annotated[Path, typer.Option("--history-root", help=_HISTORY_ROOT_HELP)]
_UniverseOpt = Annotated[Path, typer.Option("--universe", help=_UNIVERSE_HELP)]
_RunIdOpt = Annotated[str | None, typer.Option("--run-id", help=_RUN_ID_HELP)]
_SnapshotToOpt = Annotated[
    Path | None, typer.Option("--snapshot-to", help=_SNAPSHOT_TO_HELP)
]
_SnapshotIntervalOpt = Annotated[
    int, typer.Option("--snapshot-interval-ticks", help=_SNAPSHOT_INTERVAL_HELP)
]
_ExperimentIdOpt = Annotated[
    str | None, typer.Option("--experiment-id", help=_EXPERIMENT_ID_HELP)
]
_TrialNumberOpt = Annotated[int, typer.Option("--trial-number", help=_TRIAL_NUMBER_HELP)]


@app.command()
def run(
    config: _ConfigArg,
    output_dir: _OutputDirOpt = None,
    history_root: _HistoryRootOpt = DEFAULT_HISTORY_ROOT,
    universe: _UniverseOpt = DEFAULT_UNIVERSE_PATH,
    run_id: _RunIdOpt = None,
    snapshot_to: _SnapshotToOpt = None,
    snapshot_interval_ticks: _SnapshotIntervalOpt = 1,
    experiment_id: _ExperimentIdOpt = None,
    trial_number: _TrialNumberOpt = 0,
) -> None:
    """Run a backtest from a config file to completion."""
    cfg = _load_config(config)
    resolved_output_dir = (
        output_dir if output_dir is not None else DEFAULT_OUTPUT_ROOT / cfg.label
    )
    try:
        inputs = _build_runner_inputs(
            cfg, history_root=history_root, universe_path=universe
        )
        outputs = run_backtest(
            inputs,
            output_dir=resolved_output_dir,
            run_id=run_id,
            snapshot_to=snapshot_to,
            snapshot_interval_ticks=snapshot_interval_ticks,
            experiment_id=experiment_id,
            trial_number=trial_number,
        )
    except (BacktestConfigError, SnapshotError) as exc:
        console.print(f"[red]run failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _render_outputs(outputs, config=cfg)


_ResumeFromArg = Annotated[
    Path, typer.Argument(help="Snapshot written by an earlier --snapshot-to.")
]


@app.command()
def resume(
    config: _ConfigArg,
    resume_from: _ResumeFromArg,
    output_dir: _OutputDirOpt = None,
    history_root: _HistoryRootOpt = DEFAULT_HISTORY_ROOT,
    universe: _UniverseOpt = DEFAULT_UNIVERSE_PATH,
    run_id: _RunIdOpt = None,
    snapshot_to: _SnapshotToOpt = None,
    snapshot_interval_ticks: _SnapshotIntervalOpt = 1,
    experiment_id: _ExperimentIdOpt = None,
    trial_number: _TrialNumberOpt = 0,
) -> None:
    """Resume a backtest from a snapshot written by an earlier run.

    Rebuilds fresh streams from the same config (streams are consumed exactly
    once — see ``RunnerInputs``'s docstring), bounded from the snapshot's own
    ``clock_now`` rather than the config's ``start_ts`` (see
    ``_build_runner_inputs``'s docstring for why replaying from the true start
    would hand the resumed engine events the clock has already moved past),
    and hands ``resume_from`` to ``runner.run``, which independently verifies
    the snapshot's run/config/data identity against this config before
    restoring anything.
    """
    cfg = _load_config(config)
    resolved_output_dir = (
        output_dir if output_dir is not None else DEFAULT_OUTPUT_ROOT / cfg.label
    )
    try:
        # Read only for clock_now, to bound the rebuilt streams — runner.run
        # below reloads and verifies this same file's run/config/data identity
        # before trusting it for anything else.
        snapshot = load_snapshot(resume_from)
        inputs = _build_runner_inputs(
            cfg,
            history_root=history_root,
            universe_path=universe,
            stream_start_ts=snapshot.clock_now,
        )
        outputs = run_backtest(
            inputs,
            output_dir=resolved_output_dir,
            run_id=run_id,
            resume_from=resume_from,
            snapshot_to=snapshot_to,
            snapshot_interval_ticks=snapshot_interval_ticks,
            experiment_id=experiment_id,
            trial_number=trial_number,
        )
    except (BacktestConfigError, SnapshotError) as exc:
        console.print(f"[red]resume failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _render_outputs(outputs, config=cfg)


if __name__ == "__main__":
    app()
