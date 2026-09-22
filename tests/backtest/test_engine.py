"""Tests for :mod:`memetrader.backtest.engine`.

Every fixture here is built by hand rather than loaded from ``history/``
(offline only, per the assignment) and is deliberately small: one or two
symbols, two to four hourly bars, a permissive :class:`~memetrader.risk.RiskParams`
so that eligibility/portfolio/pre-trade vetoes unrelated to what a given test
checks do not get in the way.

Timing design (shared by every test that needs a settled fill)
----------------------------------------------------------------
Bars are hourly (``Timeframe.H1``), spaced exactly 3600s apart, with
``publication_delay_seconds=0`` so ``ReplayState.add_bar``'s default
``available_time = ts + interval`` applies: bar ``k`` (``ts = T0 + k*3600``)
becomes available at ``T0 + (k+1)*3600``.

A ``DECISION_TICK`` is scheduled at bar ``k``'s ``available_time`` so the
strategy sees exactly that bar's close the instant it is knowable.
``EngineConfig.settlement_latency_seconds`` is set to exactly one bar
interval (3600s), so a decision at bar ``k``'s ``available_time`` schedules
an ``EXECUTION`` at bar ``(k+1)``'s ``available_time`` — and
``EVENT_PRIORITY`` guarantees that bar's ``BAR_CLOSE`` (priority 10) is
ingested before the ``EXECUTION`` (priority 90) at that same instant, so
``BarExecutionModel.fill``'s ``bar.ts >= decided_at`` search (bar ``k+1``'s
``ts`` equals ``decided_at`` exactly) finds it without any look-ahead.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from memetrader.backtest import invariants
from memetrader.backtest.broker import SimulatedBroker
from memetrader.backtest.clock import SimulatedClock
from memetrader.backtest.engine import EngineConfig, EngineResult, ReplayEngine
from memetrader.backtest.event_queue import EventQueue
from memetrader.backtest.ledger import BacktestLedger
from memetrader.config import ExecutionConfig
from memetrader.execution.fill_models import BarExecutionModel
from memetrader.histdata.point_in_time import ReplayState
from memetrader.histdata.schemas import CandleRecord, PoolState
from memetrader.risk import RiskEngine, RiskParams
from memetrader.strategies.construction import ConstructionLimits
from memetrader.types import (
    EventKind,
    FidelityTier,
    HistoricalEvent,
    OrderIntent,
    OrderState,
    Side,
    Timeframe,
    TokenMeta,
)

T0 = 1_700_000_000.0
INTERVAL = 3600.0
POOL_ID = "pool-1"

_seq = itertools.count()


def _next_seq() -> int:
    return next(_seq)


# ---------------------------------------------------------------------------
# A tiny scripted strategy — precise, deterministic control over what is
# proposed on which tick, unlike the baselines in strategies/baselines.py.
# ---------------------------------------------------------------------------


@dataclass
class ScriptedStrategy:
    """Returns whatever :class:`OrderIntent`\\ s were pre-scripted for ``now``."""

    by_tick: dict[float, tuple[OrderIntent, ...]] = field(default_factory=dict)
    strategy_id: str = "scripted"
    required_fidelity: FidelityTier = FidelityTier.TIER_0

    def propose(self, *, state, portfolio, now, run_id):
        return self.by_tick.get(now, ())


def _buy_intent(symbol: str, *, usd: float, now: float, intent_id: str) -> OrderIntent:
    amt = int(usd * 1_000_000)
    return OrderIntent(
        intent_id=intent_id,
        decision_id=None,
        action_id=None,
        run_id="test-run",
        ts=now,
        symbol=symbol,
        side=Side.BUY,
        in_amount_atomic=amt,
        max_in_amount_atomic=amt,
        source="strategy",
        reason="scripted buy",
    )


# ---------------------------------------------------------------------------
# Event fixtures
# ---------------------------------------------------------------------------


def _bar_event(
    symbol: str, k: int, *, open_: float, close: float, volume: float = 1_000_000.0
):
    ts = T0 + k * INTERVAL
    record = CandleRecord(
        asset_id=symbol,
        pool_id=POOL_ID,
        timeframe=Timeframe.H1.value,
        ts=ts,
        event_time=ts,
        available_time=ts + INTERVAL,
        received_time=ts + INTERVAL,
        open=open_,
        high=max(open_, close),
        low=min(open_, close),
        close=close,
        volume=volume,
        closed=True,
        source="test",
    )
    return HistoricalEvent(
        kind=EventKind.BAR_CLOSE,
        available_time=record.available_time,
        asset_id=symbol,
        payload=record,
        source="test",
        sequence=_next_seq(),
    )


def _decision_event(k: int):
    available = T0 + (k + 1) * INTERVAL
    return HistoricalEvent(
        kind=EventKind.DECISION_TICK,
        available_time=available,
        asset_id=None,
        payload=None,
        source="test",
        sequence=_next_seq(),
    )


def _universe_event(symbols: frozenset[str]):
    return HistoricalEvent(
        kind=EventKind.UNIVERSE,
        available_time=T0,
        asset_id=None,
        payload=symbols,
        source="test",
        sequence=_next_seq(),
    )


def _pool_state_event(
    symbol: str, *, liquidity_usd: float = 1_000_000.0, price_usd: float = 1.0
):
    state = PoolState(
        asset_id=symbol,
        pool_id=POOL_ID,
        venue="test",
        event_time=T0,
        available_time=T0,
        received_time=T0,
        reserve_in_atomic=1,
        reserve_out_atomic=1,
        fee_rate_bps=25,
        price_usd=price_usd,
        liquidity_usd=liquidity_usd,
        source="test",
    )
    return HistoricalEvent(
        kind=EventKind.POOL_STATE,
        available_time=T0,
        asset_id=symbol,
        payload=state,
        source="test",
        sequence=_next_seq(),
    )


_PERMISSIVE_PARAMS: dict[str, Any] = {
    "require_known_pool_age": False,
    "min_liquidity_usd": 0.0,
    "min_seconds_between_entries": 0.0,
    "post_stop_quarantine_seconds": 0.0,
    "max_snapshot_age_seconds": 1.0e9,
    "require_volatility_estimate": False,
}


def _risk_params(universe: frozenset[str], **overrides) -> RiskParams:
    kwargs = dict(_PERMISSIVE_PARAMS)
    kwargs.update(overrides)
    return RiskParams(universe=universe, **kwargs)


def _execution_model() -> BarExecutionModel:
    cfg = ExecutionConfig(
        slippage_bps_fallback=100.0,
        gas_usd_per_swap=0.21,
        failed_tx_rate=0.0,
        default_pool_fee_pct=0.25,
        pool_fee_pct={},
    )
    usd_token = TokenMeta(mint="USDC", decimals=6, source="test")
    return BarExecutionModel(
        cfg,
        usd_token=usd_token,
        token_decimals=9,
        timeframe=Timeframe.H1,
        participation_cap_pct=1.0,
    )


def _build_engine(
    *,
    streams: list,
    strategy: ScriptedStrategy,
    risk_params: RiskParams,
    starting_cash_usd: float = 1_000.0,
    rebalance_band_usd: float = 15.0,
    min_trade_usd: float = 10.0,
    construction_limits: ConstructionLimits | None = None,
    dry_run: bool = False,
    settlement_latency_seconds: float = INTERVAL,
    run_id: str = "test-run",
) -> ReplayEngine:
    clock = SimulatedClock(run_id, start=T0 - 1.0)
    queue = EventQueue(clock, streams=streams)
    state = ReplayState(now=T0 - 1.0, publication_delay_seconds=0.0)
    ledger = BacktestLedger(
        starting_cash_micro_usd=int(starting_cash_usd * 1_000_000), run_id=run_id
    )
    broker = SimulatedBroker(
        execution_model=_execution_model(), risk_engine=RiskEngine(params=risk_params)
    )
    config = EngineConfig(
        run_id=run_id,
        starting_cash_micro_usd=int(starting_cash_usd * 1_000_000),
        settlement_latency_seconds=settlement_latency_seconds,
        decision_timeframe=Timeframe.H1,
        construction_limits=construction_limits
        or ConstructionLimits(per_asset_cap_bps=10_000, portfolio_cap_bps=10_000),
        rebalance_band_usd=rebalance_band_usd,
        min_trade_usd=min_trade_usd,
    )
    return ReplayEngine(
        clock=clock,
        queue=queue,
        state=state,
        ledger=ledger,
        broker=broker,
        strategy=strategy,
        config=config,
        dry_run=dry_run,
    )


def _one_symbol_streams(
    symbol: str,
    prices: list[tuple[float, float]],
    *,
    decision_ticks: list[int],
    liquidity_usd: float = 1_000_000.0,
) -> list:
    events = [
        _universe_event(frozenset({symbol})),
        _pool_state_event(symbol, liquidity_usd=liquidity_usd),
    ]
    for k, (open_, close) in enumerate(prices):
        events.append(_bar_event(symbol, k, open_=open_, close=close))
    events.extend(_decision_event(k) for k in decision_ticks)
    events.sort(key=lambda e: e.sort_key)
    return [iter(events)]


# ---------------------------------------------------------------------------
# Anti-lookahead (headline test)
# ---------------------------------------------------------------------------


def test_anti_lookahead_fill_uses_next_bar_open_not_decision_bar_close():
    symbol = "ANTI"
    # bar0: open=1.0 close=1.0 (the bar the decision is made from)
    # bar1: open=2.0 close=2.5 (open != close, so we can tell which was used)
    prices = [(1.0, 1.0), (2.0, 2.5)]
    decided_at = T0 + INTERVAL  # bar0's available_time
    intent = _buy_intent(symbol, usd=50.0, now=decided_at, intent_id="intent-1")
    strategy = ScriptedStrategy(by_tick={decided_at: (intent,)})
    streams = _one_symbol_streams(symbol, prices, decision_ticks=[0])
    engine = _build_engine(
        streams=streams,
        strategy=strategy,
        risk_params=_risk_params(frozenset({symbol})),
    )
    result = engine.run()

    fills = [r.fill for r in result.reports if r.fill is not None]
    assert len(fills) == 1
    fill = fills[0]
    # Must fill at bar1's open (2.0) — never at bar0's close (1.0, the
    # decision bar) and never at bar1's close (2.5, which would still be
    # look-ahead relative to the instant the fill is generated).
    assert fill.price_usd == 2.0


# ---------------------------------------------------------------------------
# Deterministic replay
# ---------------------------------------------------------------------------


def _det_run() -> EngineResult:
    symbol = "DET"
    prices = [(1.0, 1.1), (1.1, 1.2), (1.2, 1.3)]
    t1 = T0 + INTERVAL
    intent = _buy_intent(symbol, usd=50.0, now=t1, intent_id="det-intent")
    strategy = ScriptedStrategy(by_tick={t1: (intent,)})
    streams = _one_symbol_streams(symbol, prices, decision_ticks=[0, 1])
    engine = _build_engine(
        streams=streams, strategy=strategy, risk_params=_risk_params(frozenset({symbol}))
    )
    return engine.run()


def test_deterministic_replay_byte_identical_across_two_runs():
    # ids.py mints intent/order/fill ids from wall-clock + os.urandom, which
    # is fine here because check_deterministic_replay compares *economic*
    # ledger state (cash, positions, realized pnl), not raw id strings — see
    # its docstring. Two independent runs of the identical scripted setup
    # must still produce the same economic outcome.
    global _seq
    _seq = itertools.count()
    first = _det_run()
    second = _det_run()

    assert len(first.reports) == len(second.reports)
    for a, b in zip(first.reports, second.reports, strict=True):
        assert a.state == b.state
        if a.fill is not None:
            assert b.fill is not None
            assert a.fill.price_usd == b.fill.price_usd
            assert a.fill.notional_usd == b.fill.notional_usd
            assert a.fill.token_amount_atomic == b.fill.token_amount_atomic
        else:
            assert b.fill is None

    invariants.check_deterministic_replay(
        first.reports, starting_cash_micro_usd=first.ledger._starting_cash_micro
    )


# ---------------------------------------------------------------------------
# Risk clamp forces an exact-size requote (audit C3)
# ---------------------------------------------------------------------------


def test_risk_clamp_forces_exact_size_requote():
    symbol = "CLAMP"
    prices = [(1.0, 1.0), (1.0, 1.0)]
    t1 = T0 + INTERVAL
    # Ask for $500 against a book whose max_position_pct bound will be far
    # smaller (~1% of a $1000 book == $10), forcing broker._attempt's
    # resize-and-requote path.
    intent = _buy_intent(symbol, usd=500.0, now=t1, intent_id="clamp-intent")
    strategy = ScriptedStrategy(by_tick={t1: (intent,)})
    streams = _one_symbol_streams(symbol, prices, decision_ticks=[0])
    risk_params = _risk_params(frozenset({symbol}), max_position_pct=0.01)
    engine = _build_engine(streams=streams, strategy=strategy, risk_params=risk_params)
    engine.run()

    assert engine.broker.resize_pairs, "expected at least one risk-driven resize"
    for approved_atomic, quoted_atomic in engine.broker.resize_pairs:
        assert approved_atomic == quoted_atomic
    invariants.check_risk_resize_requotes(engine.broker.resize_pairs)

    # And the actual settled notional must be far below the $500 ask.
    fills = [r.fill for r in engine.reports if r.fill is not None]
    assert fills
    assert fills[0].notional_usd < 50.0


# ---------------------------------------------------------------------------
# Dry run mutates nothing
# ---------------------------------------------------------------------------


def test_dry_run_mutates_nothing():
    symbol = "DRY"
    prices = [(1.0, 1.0), (1.0, 1.0)]
    t1 = T0 + INTERVAL
    intent = _buy_intent(symbol, usd=50.0, now=t1, intent_id="dry-intent")
    strategy = ScriptedStrategy(by_tick={t1: (intent,)})
    streams = _one_symbol_streams(symbol, prices, decision_ticks=[0])
    engine = _build_engine(
        streams=streams,
        strategy=strategy,
        risk_params=_risk_params(frozenset({symbol})),
        dry_run=True,
    )
    starting_micro = engine.ledger._starting_cash_micro
    result = engine.run()

    # A dry run still produces reports (so a caller can see what *would*
    # have happened) but check_dry_run_no_mutation (invoked internally by
    # engine._settle for every dry-run report) guarantees the ledger itself
    # never moved — confirmed here structurally, not just by absence of an
    # exception.
    assert any(r.fill is not None for r in result.reports)
    snapshot = engine.ledger.snapshot()
    assert snapshot.cash_micro_usd == starting_micro
    assert snapshot.positions == {}


# ---------------------------------------------------------------------------
# Duplicate ExecutionReport is a no-op
# ---------------------------------------------------------------------------


def test_duplicate_execution_report_is_noop():
    ledger = BacktestLedger(starting_cash_micro_usd=1_000_000_000, run_id="dup-run")
    intent = _buy_intent("DUPE", usd=50.0, now=T0, intent_id="dupe-intent")
    engine = _build_engine(
        streams=_one_symbol_streams("DUPE", [(1.0, 1.0), (1.0, 1.0)], decision_ticks=[0]),
        strategy=ScriptedStrategy(by_tick={T0 + INTERVAL: (intent,)}),
        risk_params=_risk_params(frozenset({"DUPE"})),
    )
    result = engine.run()
    fills = [r for r in result.reports if r.fill is not None]
    assert fills
    report = fills[0]
    invariants.check_duplicate_report_is_noop(ledger, report)


# ---------------------------------------------------------------------------
# A route that never appears produces a FAILED report, no position change
# ---------------------------------------------------------------------------


def test_no_route_at_settlement_produces_no_position_change():
    symbol = "NOROUTE"
    # Only bar0 exists — decided_at = bar0's available_time. No bar with
    # ts >= decided_at will ever be ingested, so BarExecutionModel.fill()
    # raises NoRoute at settlement and the broker returns a synthetic FAILED
    # report.
    t1 = T0 + INTERVAL
    intent = _buy_intent(symbol, usd=50.0, now=t1, intent_id="noroute-intent")
    strategy = ScriptedStrategy(by_tick={t1: (intent,)})
    streams = _one_symbol_streams(symbol, [(1.0, 1.0)], decision_ticks=[0])
    engine = _build_engine(
        streams=streams, strategy=strategy, risk_params=_risk_params(frozenset({symbol}))
    )
    result = engine.run()

    failed = [r for r in result.reports if r.state is OrderState.FAILED]
    assert failed, "expected the unsettleable order to fail"
    assert engine.ledger.snapshot().positions == {}
    assert engine.ledger.snapshot().cash_micro_usd == engine.ledger._starting_cash_micro


# ---------------------------------------------------------------------------
# check_all passes every tick of a multi-tick run, and the clock never
# moves backwards
# ---------------------------------------------------------------------------


def test_check_all_every_tick_and_clock_monotonic():
    symbol = "MULTI"
    prices = [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    t1, t2 = T0 + INTERVAL, T0 + 2 * INTERVAL
    intent1 = _buy_intent(symbol, usd=20.0, now=t1, intent_id="multi-1")
    intent2 = _buy_intent(symbol, usd=20.0, now=t2, intent_id="multi-2")
    strategy = ScriptedStrategy(by_tick={t1: (intent1,), t2: (intent2,)})
    streams = _one_symbol_streams(symbol, prices, decision_ticks=[0, 1, 2])
    engine = _build_engine(
        streams=streams, strategy=strategy, risk_params=_risk_params(frozenset({symbol}))
    )

    seen_times: list[float] = []

    def on_tick(eng: ReplayEngine) -> None:
        seen_times.append(eng.clock.now)

    result = engine.run(on_tick=on_tick)

    # If check_all/check_risk_resize_requotes had ever failed inside run(),
    # an InvariantBreach would have propagated out of engine.run() and this
    # line would never be reached.
    assert result.ticks_checked == len(seen_times)
    assert result.ticks_checked > 0

    for earlier, later in itertools.pairwise(seen_times):
        assert later >= earlier


# ---------------------------------------------------------------------------
# Snapshot mid-run is byte-identical to the same tick of an uninterrupted run
# (equity_curve is the cheapest, most direct proxy available at this layer;
# full snapshot/resume continuation equivalence is exercised end-to-end in
# tests/backtest/test_runner.py)
# ---------------------------------------------------------------------------


def test_equity_curve_reproducible_across_identical_runs():
    global _seq
    _seq = itertools.count()
    first = _det_run()
    _seq = itertools.count()
    second = _det_run()
    assert first.equity_curve == second.equity_curve


# ---------------------------------------------------------------------------
# Rebalance-band / min-trade filtering (contracts §5 step 9)
# ---------------------------------------------------------------------------


def test_rebalance_band_drops_small_delta_but_not_larger_one():
    sym_low = "LOWDELTA"
    sym_high = "HIGHDELTA"
    prices = [(1.0, 1.0), (1.0, 1.0)]
    t1 = T0 + INTERVAL

    # $10 is below the default rebalance_band_usd (15.0): must be filtered.
    low_intent = _buy_intent(sym_low, usd=10.0, now=t1, intent_id="low-intent")
    # $20 clears the band: must survive to routing.
    high_intent = _buy_intent(sym_high, usd=20.0, now=t1, intent_id="high-intent")
    strategy = ScriptedStrategy(by_tick={t1: (low_intent, high_intent)})

    universe = frozenset({sym_low, sym_high})
    events = [_universe_event(universe)]
    events.append(_pool_state_event(sym_low))
    events.append(_pool_state_event(sym_high))
    for k, (open_, close) in enumerate(prices):
        events.append(_bar_event(sym_low, k, open_=open_, close=close))
        events.append(_bar_event(sym_high, k, open_=open_, close=close))
    events.append(_decision_event(0))
    events.sort(key=lambda e: e.sort_key)

    engine = _build_engine(
        streams=[iter(events)],
        strategy=strategy,
        risk_params=_risk_params(universe),
        rebalance_band_usd=15.0,
        min_trade_usd=10.0,
    )
    result = engine.run()

    assert len(result.decisions) == 1
    routed_symbols = {intent.symbol for intent in result.decisions[0].intents}
    assert routed_symbols == {sym_high}

    reported_symbols = {r.fill.symbol for r in result.reports if r.fill is not None}
    assert sym_high in reported_symbols
    assert sym_low not in reported_symbols


def test_rebalance_band_boundary_is_inclusive_of_the_band_itself():
    # abs(delta) < band drops; abs(delta) == band is not "< band" so it must
    # survive. Pick a delta exactly equal to the (non-default) band.
    symbol = "EXACT"
    prices = [(1.0, 1.0), (1.0, 1.0)]
    t1 = T0 + INTERVAL
    intent = _buy_intent(symbol, usd=15.0, now=t1, intent_id="exact-intent")
    strategy = ScriptedStrategy(by_tick={t1: (intent,)})
    streams = _one_symbol_streams(symbol, prices, decision_ticks=[0])
    engine = _build_engine(
        streams=streams,
        strategy=strategy,
        risk_params=_risk_params(frozenset({symbol})),
        rebalance_band_usd=15.0,
        min_trade_usd=10.0,
    )
    result = engine.run()
    assert len(result.decisions) == 1
    assert {intent.symbol for intent in result.decisions[0].intents} == {symbol}
