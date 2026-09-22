"""Orchestration: turn a :class:`~memetrader.backtest.config.BacktestConfig`
and a set of already-built data streams into a completed
:class:`~memetrader.backtest.engine.ReplayEngine` run, and write its outputs.

**Scope trim, stated up front rather than discovered by surprise later:**
resolving a ``BacktestConfig`` into concrete ``HistoricalEvent`` streams (which
catalog partitions to open, which loader to call for which timeframe, how to
turn ``execution_model: str`` into a constructed ``ExecutionModel`` with the
right ``TokenMeta``/``ExecutionConfig`` wiring) is catalog- and
execution-model-registry plumbing that belongs to whichever module owns
``histdata/catalog.py`` and the execution model registry — not to the six
files this assignment covers. ``runner.py`` therefore accepts pre-built
streams, a pre-built ``ExecutionModel``, and a pre-built ``BacktestStrategy``
as :class:`RunnerInputs`, and owns everything *after* that point: constructing
the clock/queue/state/ledger/broker, running the engine, and writing the run's
outputs (manifest, ledger, metrics) per ``docs/BACKTEST-CONTRACTS.md`` §9,
plus snapshot save/resume.

Output layout, under ``output_dir``:

* ``manifest.json`` — ``experiments.manifest.Manifest``, keyed to this run's
  ``config_hash``.
* ``ledger.json`` — the full ``BacktestLedger.to_state()`` dump: every entry,
  every fee total, every open lot.
* ``metrics.json`` — ``metrics.performance.PerformanceMetrics`` computed from
  the engine's recorded equity curve and settled fills.
* ``snapshot.json`` — written only when ``snapshot_path`` is requested;
  everything :mod:`memetrader.backtest.snapshots` needs to resume this run
  exactly, including the ID-counter and RNG states a resume must restore.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memetrader import ids
from memetrader.backtest.broker import SimulatedBroker
from memetrader.backtest.clock import SimulatedClock
from memetrader.backtest.config import BacktestConfig
from memetrader.backtest.engine import EngineConfig, EngineResult, ReplayEngine
from memetrader.backtest.event_queue import EventQueue
from memetrader.backtest.ledger import BacktestLedger
from memetrader.backtest.snapshots import (
    ReplaySnapshot,
    build_snapshot,
    deep_tuple,
    resume_ledger,
    rng_state_from_random,
)
from memetrader.backtest.snapshots import load as load_snapshot
from memetrader.backtest.snapshots import save as save_snapshot
from memetrader.execution.interfaces import ExecutionModel
from memetrader.experiments.manifest import build_manifest, write_manifest
from memetrader.histdata.loaders.ohlcv import interval_seconds
from memetrader.histdata.point_in_time import ReplayState
from memetrader.journal import atomic_write_text
from memetrader.metrics.performance import PerformanceMetrics, compute_performance
from memetrader.risk import RiskEngine
from memetrader.strategies.baselines import BacktestStrategy
from memetrader.types import HistoricalEvent, Timeframe

__all__ = ["RunOutputs", "RunnerInputs", "build_engine", "run"]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunnerInputs:
    """Everything :func:`build_engine`/:func:`run` need beyond the config.

    ``streams`` are consumed exactly once — building an :class:`EventQueue`
    from them starts pulling immediately. A caller that wants to run the same
    logical replay twice (e.g. the determinism test) must build two fresh
    ``RunnerInputs`` with two fresh stream iterators, not reuse one.
    """

    config: BacktestConfig
    streams: list[Iterator[HistoricalEvent]]
    strategy: BacktestStrategy
    execution_model: ExecutionModel
    mint_for: dict[str, str] = field(default_factory=dict)
    pool_for: dict[str, str] = field(default_factory=dict)
    risk_engine: RiskEngine = field(default_factory=RiskEngine)
    decision_timeframe: Timeframe = Timeframe.H1
    dry_run: bool = False
    rng_streams: dict[str, random.Random] = field(default_factory=dict)
    """Named stdlib RNG streams the caller wants captured/restored across a
    snapshot. The engine itself draws no randomness (see ``EngineConfig``'s
    docstring on why settlement latency is a fixed constant, not a draw); this
    exists for a caller-supplied strategy or execution model that does."""
    rebalance_band_usd: float = 15.0
    min_trade_usd: float = 10.0
    """Passed straight through to ``EngineConfig`` (contracts §5 step 9).
    ``BacktestConfig`` carries no equivalent fields today, so there is nothing
    for this module to read them from on its own; a caller that wants these to
    track a particular live ``StrategySettings``/``BacktestConfig`` value is
    responsible for reading it and passing it in here. Defaults match
    ``StrategySettings``'s (``src/memetrader/strategy.py``) so an unconfigured
    run still behaves the same as the live loop by default."""


@dataclass(frozen=True, slots=True)
class RunOutputs:
    run_id: str
    output_dir: Path
    manifest_path: Path
    ledger_path: Path
    metrics_path: Path
    result: EngineResult


# ---------------------------------------------------------------------------
# Building the engine
# ---------------------------------------------------------------------------


def _engine_config(
    config: BacktestConfig,
    *,
    run_id: str,
    decision_timeframe: Timeframe,
    rebalance_band_usd: float = 15.0,
    min_trade_usd: float = 10.0,
) -> EngineConfig:
    # A settled fill needs a bar whose ts >= decided_at to already be knowable
    # (available_time <= now — see fill_models.BarExecutionModel.fill and
    # histdata.loaders.ohlcv.available_time_for). The earliest such bar's
    # available_time is decided_at + interval_seconds(decision_timeframe) +
    # publication_delay_seconds, so settlement latency must be at least that
    # long or every entry raises NoRoute("no bar opened after decided_at=...")
    # forever, regardless of how the strategy or config are tuned.
    # fast_tick_seconds alone (the clock's step size) has no necessary
    # relationship to that floor and can be far smaller than it.
    min_settlement_latency = (
        interval_seconds(decision_timeframe) + config.publication_delay_seconds
    )
    settlement_latency_seconds = max(
        min_settlement_latency, float(config.fast_tick_seconds), 1.0
    )
    return EngineConfig(
        run_id=run_id,
        starting_cash_micro_usd=config.starting_cash_micro_usd,
        settlement_latency_seconds=settlement_latency_seconds,
        decision_timeframe=decision_timeframe,
        rebalance_band_usd=rebalance_band_usd,
        min_trade_usd=min_trade_usd,
    )


def build_engine(
    inputs: RunnerInputs,
    *,
    run_id: str,
    resume_snapshot: ReplaySnapshot | None = None,
) -> ReplayEngine:
    """Construct a fresh (or resumed) :class:`ReplayEngine`, not yet run.

    When ``resume_snapshot`` is given, the ledger is rebuilt via
    ``snapshots.resume_ledger`` (which itself verifies run/config/data
    identity and raises ``SnapshotMismatch`` rather than resuming against the
    wrong thing), the clock starts from ``snapshot.clock_now`` instead of
    ``config.start_ts``, and every RNG stream present in both
    ``inputs.rng_streams`` and the snapshot is restored to its exact consumed
    state via ``on_clock_active`` — restoring before any event of the resumed
    run is processed, and after the clock (which ``ids.restore_ids_state``
    requires to be active) has entered.
    """
    config = inputs.config
    start = resume_snapshot.clock_now if resume_snapshot is not None else config.start_ts
    clock = SimulatedClock(run_id, start=start)
    queue = EventQueue(clock, streams=inputs.streams)
    state = ReplayState(
        now=start, publication_delay_seconds=config.publication_delay_seconds
    )

    if resume_snapshot is not None:
        ledger = resume_ledger(
            resume_snapshot,
            run_id=run_id,
            config_hash=config.config_hash,
        )
    else:
        ledger = BacktestLedger(
            starting_cash_micro_usd=config.starting_cash_micro_usd, run_id=run_id
        )

    broker = SimulatedBroker(
        execution_model=inputs.execution_model, risk_engine=inputs.risk_engine
    )
    engine = ReplayEngine(
        clock=clock,
        queue=queue,
        state=state,
        ledger=ledger,
        broker=broker,
        strategy=inputs.strategy,
        config=_engine_config(
            config,
            run_id=run_id,
            decision_timeframe=inputs.decision_timeframe,
            rebalance_band_usd=inputs.rebalance_band_usd,
            min_trade_usd=inputs.min_trade_usd,
        ),
        mint_for=dict(inputs.mint_for),
        pool_for=dict(inputs.pool_for),
        dry_run=inputs.dry_run,
    )

    def _on_clock_active(_: SimulatedClock) -> None:
        if resume_snapshot is None:
            return
        # The replay ID counter is stashed under a reserved key in
        # ``rng_states`` — ``ReplaySnapshot`` has no dedicated field for it,
        # and ``rng_states`` is already documented as an opaque, caller-owned
        # blob keyed by stream name, so a reserved name here is exactly that
        # contract, not a workaround of it.
        ids_state = resume_snapshot.rng_states.get(_IDS_STATE_KEY)
        if ids_state is not None:
            ids.restore_ids_state(dict(ids_state))
        for name, rng in inputs.rng_streams.items():
            captured = resume_snapshot.rng_states.get(name)
            if captured is not None:
                rng.setstate(deep_tuple(captured))

    engine._resume_on_clock_active = _on_clock_active  # type: ignore[attr-defined]
    return engine


_IDS_STATE_KEY = "__ids_state__"


def _snapshot_now(
    engine: ReplayEngine, *, inputs: RunnerInputs, config: BacktestConfig, path: Path
) -> None:
    """Capture and durably save a snapshot while ``engine.clock`` is still
    the active clock — required for ``ids.ids_state()`` to return anything
    (see its docstring: ``{}`` when no replay is active) and for
    ``rng_state_from_random`` to reflect draws made so far in this run."""
    rng_states: dict[str, Any] = {
        name: rng_state_from_random(rng) for name, rng in inputs.rng_streams.items()
    }
    rng_states[_IDS_STATE_KEY] = ids.ids_state()
    snapshot = build_snapshot(
        clock=engine.clock,
        ledger=engine.ledger,
        config_hash=config.config_hash,
        rng_states=rng_states,
    )
    save_snapshot(path, snapshot)


# ---------------------------------------------------------------------------
# Running and writing outputs
# ---------------------------------------------------------------------------


def run(
    inputs: RunnerInputs,
    *,
    output_dir: Path,
    run_id: str | None = None,
    resume_from: Path | None = None,
    snapshot_to: Path | None = None,
    snapshot_interval_ticks: int = 1,
    experiment_id: str | None = None,
    trial_number: int = 0,
) -> RunOutputs:
    """Run ``inputs`` to completion and write manifest/ledger/metrics.

    Resuming: pass ``resume_from`` (a path written by an earlier
    ``snapshot_to``). The resumed clock, ledger, and (for any RNG stream named
    in both ``inputs.rng_streams`` and the snapshot) RNG state are restored
    before the first event of the resumed run is processed — see
    :func:`build_engine`'s docstring for exactly when and why.

    ``snapshot_to``, when given, is (over)written every ``snapshot_interval_ticks``
    ticks *and* after the final tick — always via :class:`ReplayEngine`'s
    ``on_tick`` hook, i.e. while the clock is still active, which is the only
    time ``ids.ids_state()`` reports anything real. A crash between two
    snapshot writes loses at most ``snapshot_interval_ticks - 1`` ticks of
    progress, never correctness: resuming from any one of these files must
    reproduce the same continuation an uninterrupted run would have taken
    from that same point (the property ``tests/backtest/test_runner.py``'s
    snapshot/resume test checks).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    config = inputs.config

    resume_snapshot: ReplaySnapshot | None = None
    if resume_from is not None:
        resume_snapshot = load_snapshot(resume_from)
        resolved_run_id = run_id or resume_snapshot.run_id
    else:
        resolved_run_id = run_id or ids.new_run_id()

    engine = build_engine(inputs, run_id=resolved_run_id, resume_snapshot=resume_snapshot)
    on_clock_active = getattr(engine, "_resume_on_clock_active", None)

    on_tick = None
    if snapshot_to is not None:
        snap_path: Path = snapshot_to

        def on_tick(eng: ReplayEngine) -> None:
            is_last = not eng.queue
            if is_last or eng.ticks_checked % max(1, snapshot_interval_ticks) == 0:
                _snapshot_now(eng, inputs=inputs, config=config, path=snap_path)

    result = engine.run(on_clock_active=on_clock_active, on_tick=on_tick)

    manifest = build_manifest(
        experiment_id=experiment_id or config.label,
        trial_number=trial_number,
        fidelity_tier=config.fidelity_tier,
        universe_version=config.universe_version,
        config_hash=config.config_hash,
        random_seeds=dict(config.seeds),
    )
    manifest_path = output_dir / "manifest.json"
    write_manifest(manifest_path, manifest)

    ledger_path = output_dir / "ledger.json"
    atomic_write_text(
        ledger_path,
        json.dumps(result.ledger.to_state(), sort_keys=True, indent=2) + "\n",
    )

    metrics_path = output_dir / "metrics.json"
    metrics = _compute_metrics(
        result, config=config, decision_timeframe=inputs.decision_timeframe
    )
    atomic_write_text(
        metrics_path,
        json.dumps(_metrics_to_dict(metrics), sort_keys=True, indent=2) + "\n",
    )

    return RunOutputs(
        run_id=resolved_run_id,
        output_dir=output_dir,
        manifest_path=manifest_path,
        ledger_path=ledger_path,
        metrics_path=metrics_path,
        result=result,
    )


_PERIODS_PER_YEAR: dict[Timeframe, float] = {
    Timeframe.M5: 365 * 24 * 12,
    Timeframe.H1: 365 * 24,
}


def _compute_metrics(
    result: EngineResult, *, config: BacktestConfig, decision_timeframe: Timeframe
) -> PerformanceMetrics:
    equity = [value for _, value in result.equity_curve]
    fills = [report.fill for report in result.reports if report.fill is not None]
    periods_per_year = _PERIODS_PER_YEAR.get(decision_timeframe, 365 * 24)
    return compute_performance(
        equity,
        fills,
        periods_per_year=periods_per_year,
        fidelity=config.fidelity_tier,
    )


def _metrics_to_dict(m: PerformanceMetrics) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(m)
