"""Tests for :mod:`memetrader.backtest.runner`.

Three things are checked here, deliberately not duplicated from
``test_engine.py`` (which already covers the tick-order/invariant behavior at
the ``ReplayEngine`` layer directly): that ``run()`` writes well-formed
artifacts, that a snapshot/resume/finish continuation reproduces an
uninterrupted run's economics exactly, and that the reserved
``__ids_state__`` key round-trips through a saved snapshot.

**A structural gap this file works around, not silently:** ``ReplaySnapshot``
(``snapshots.py``) captures the ledger, the clock, and opaque caller-owned RNG
blobs — it does not capture :class:`~memetrader.histdata.point_in_time.ReplayState`
at all. ``build_engine`` always constructs a resumed run's ``ReplayState``
fresh and empty (see its docstring). That means a resumed run's strategy
would see an empty universe and no bars until new data streams back in —
which would make any BUY placed on the very first post-resume tick get
dropped by ``strategies.construction.size_orders`` (BUYs outside
``state.universe()`` are dropped) and any entry attempt refused by
``risk.EligibilityRisk.assess``'s unconditional ``missing_snapshot``/
``no_liquidity_observation`` vetoes (no bar, no pool state ingested yet).

The fix applied here mirrors what a real resume implementation would
plausibly do: re-open the historical catalog and re-feed every
universe/bar/pool-state record already true as of the resume point, so the
fresh ``ReplayState`` catches back up to where the discarded one was. The one
wrinkle is that :class:`~memetrader.backtest.clock.SimulatedClock` refuses to
move backwards (``advance_to`` requires ``t >= now``), so the re-fed events'
own *outer* ``HistoricalEvent.available_time`` (which drives the clock and
queue ordering) is set to the resume point itself, while each payload's own
internal ``available_time`` field — what ``ReplayState.add_bar``/
``add_pool_state`` actually index on — keeps its true historical value. Bars
and pool states already tolerate being loaded more than once (no dedup check
in ``point_in_time.py``; harmless since the content re-fed here is identical
to what was already ingested pre-snapshot), so this is a safe, idempotent
re-feed rather than a hack around a broken guard.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from memetrader.backtest import runner as runner_mod
from memetrader.backtest.config import BacktestConfig
from memetrader.backtest.config import load as load_config
from memetrader.backtest.runner import RunnerInputs, run
from memetrader.backtest.snapshots import load as load_snapshot
from memetrader.config import ExecutionConfig
from memetrader.execution.fill_models import BarExecutionModel
from memetrader.histdata.schemas import CandleRecord, PoolState
from memetrader.risk import RiskEngine, RiskParams
from memetrader.types import (
    EventKind,
    FidelityTier,
    HistoricalEvent,
    OrderIntent,
    Side,
    Timeframe,
    TokenMeta,
)

T0 = 1_700_000_000.0
INTERVAL = 3600.0
SYMBOL = "RESUME"
POOL_ID = "pool-resume"

_seq = itertools.count()


def _next_seq() -> int:
    return next(_seq)


# ---------------------------------------------------------------------------
# A tiny scripted strategy, identical in spirit to test_engine.py's — precise
# control over what is proposed on which tick.
# ---------------------------------------------------------------------------


@dataclass
class ScriptedStrategy:
    by_tick: dict[float, tuple[OrderIntent, ...]] = field(default_factory=dict)
    strategy_id: str = "scripted"
    required_fidelity: FidelityTier = FidelityTier.TIER_0

    def propose(self, *, state, portfolio, now, run_id):
        return self.by_tick.get(now, ())


def _buy_intent(*, usd: float, now: float, intent_id: str) -> OrderIntent:
    amt = int(usd * 1_000_000)
    return OrderIntent(
        intent_id=intent_id,
        decision_id=None,
        action_id=None,
        run_id="resume-test",
        ts=now,
        symbol=SYMBOL,
        side=Side.BUY,
        in_amount_atomic=amt,
        max_in_amount_atomic=amt,
        source="strategy",
        reason="scripted buy",
    )


# ---------------------------------------------------------------------------
# Event fixtures
# ---------------------------------------------------------------------------


def _bar_record(k: int) -> CandleRecord:
    """Flat $1.00 bars — keeps the economics simple enough that the
    continuation-equivalence check is about ledger/order-flow continuity, not
    about reproducing a particular price path."""
    ts = T0 + k * INTERVAL
    return CandleRecord(
        asset_id=SYMBOL,
        pool_id=POOL_ID,
        timeframe=Timeframe.H1.value,
        ts=ts,
        event_time=ts,
        available_time=ts + INTERVAL,
        received_time=ts + INTERVAL,
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=1_000_000.0,
        closed=True,
        source="test",
    )


def _bar_event(k: int, *, outer_available_time: float | None = None) -> HistoricalEvent:
    """``outer_available_time`` overrides only the wrapping event's clock-facing
    timestamp — used for the resumed run's re-feed, where the payload's own
    ``available_time`` (what ``ReplayState.add_bar`` indexes on) must stay at
    its true historical value even though the clock itself has moved past it."""
    record = _bar_record(k)
    outer = (
        outer_available_time if outer_available_time is not None else record.available_time
    )
    return HistoricalEvent(
        kind=EventKind.BAR_CLOSE,
        available_time=outer,
        asset_id=SYMBOL,
        payload=record,
        source="test",
        sequence=_next_seq(),
    )


def _decision_event(k: int) -> HistoricalEvent:
    available = T0 + (k + 1) * INTERVAL
    return HistoricalEvent(
        kind=EventKind.DECISION_TICK,
        available_time=available,
        asset_id=None,
        payload=None,
        source="test",
        sequence=_next_seq(),
    )


def _universe_event(*, outer_available_time: float = T0) -> HistoricalEvent:
    return HistoricalEvent(
        kind=EventKind.UNIVERSE,
        available_time=outer_available_time,
        asset_id=None,
        payload=frozenset({SYMBOL}),
        source="test",
        sequence=_next_seq(),
    )


def _pool_state_payload() -> PoolState:
    return PoolState(
        asset_id=SYMBOL,
        pool_id=POOL_ID,
        venue="test",
        event_time=T0,
        available_time=T0,
        received_time=T0,
        reserve_in_atomic=1,
        reserve_out_atomic=1,
        fee_rate_bps=25,
        price_usd=1.0,
        liquidity_usd=1_000_000.0,
        source="test",
    )


def _pool_state_event(*, outer_available_time: float = T0) -> HistoricalEvent:
    return HistoricalEvent(
        kind=EventKind.POOL_STATE,
        available_time=outer_available_time,
        asset_id=SYMBOL,
        payload=_pool_state_payload(),
        source="test",
        sequence=_next_seq(),
    )


# ---------------------------------------------------------------------------
# Risk / execution model / config plumbing
# ---------------------------------------------------------------------------

_PERMISSIVE_PARAMS: dict[str, Any] = {
    "require_known_pool_age": False,
    "min_liquidity_usd": 0.0,
    "min_seconds_between_entries": 0.0,
    "post_stop_quarantine_seconds": 0.0,
    "max_snapshot_age_seconds": 1.0e9,
    "require_volatility_estimate": False,
}


def _risk_engine() -> RiskEngine:
    return RiskEngine(params=RiskParams(universe=frozenset({SYMBOL}), **_PERMISSIVE_PARAMS))


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


_TOML_TEMPLATE = """\
[run]
label = "resume_test"
run_id_prefix = "resume"

[date_range]
start_ts = {start_ts}
end_ts   = {end_ts}

[data]
universe_version = "v1"
timeframes = ["1h"]

[cadence]
fast_tick_seconds     = 3600
decision_tick_seconds = 3600

[portfolio]
starting_cash_usd = {starting_cash_usd}

[execution]
publication_delay_seconds = 0.0
execution_model  = "ohlcv_midpoint"
execution_mode   = "paper"
fidelity_tier    = "tier_0_ohlcv"

[execution.latency]
mu             = -1.5
sigma          = 0.4
failed_tx_rate = 0.06

[folds]
train_seconds   = 7776000
val_seconds     = 2592000
test_seconds    = 2592000
roll_seconds    = 2592000
embargo_seconds = 172800
n_outer_folds   = 3

[holdout]
opened = false

[seeds]
latency_draw = 42

[cost_stress]
multipliers = [1, 2, 3]
"""


def _config(tmp_path: Path, *, name: str = "resume_test.toml") -> BacktestConfig:
    content = _TOML_TEMPLATE.format(
        start_ts=T0 - 1.0, end_ts=T0 + 20 * INTERVAL, starting_cash_usd=1000.0
    )
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return load_config(path)


# ---------------------------------------------------------------------------
# Shared fixture: 5 bars (k=0..4), a decision tick at each bar's availability
# (t1..t5), one BUY at t1 (pre-split) and one BUY at t3 (post-resume).
# ---------------------------------------------------------------------------

_T1 = T0 + INTERVAL
_T2 = T0 + 2 * INTERVAL
_T3 = T0 + 3 * INTERVAL
_T4 = T0 + 4 * INTERVAL
_T5 = T0 + 5 * INTERVAL


def _full_events() -> list[HistoricalEvent]:
    events = [_universe_event(), _pool_state_event()]
    events.extend(_bar_event(k) for k in range(5))
    events.extend(_decision_event(k) for k in range(5))
    events.sort(key=lambda e: e.sort_key)
    return events


def _strategy() -> ScriptedStrategy:
    # Two modest buys of the same symbol — small enough, individually and
    # combined, to clear size_orders' default per-asset/portfolio caps
    # (20%/100% of a $1000 book) as well as the rebalance band ($15) and
    # min-trade floor ($10), so both are expected to settle in full.
    return ScriptedStrategy(
        by_tick={
            _T1: (_buy_intent(usd=60.0, now=_T1, intent_id="pre-split-buy"),),
            _T3: (_buy_intent(usd=60.0, now=_T3, intent_id="post-resume-buy"),),
        }
    )


def _inputs(config: BacktestConfig, streams: list) -> RunnerInputs:
    return RunnerInputs(
        config=config,
        streams=streams,
        strategy=_strategy(),
        execution_model=_execution_model(),
        risk_engine=_risk_engine(),
    )


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------


def test_run_writes_manifest_ledger_and_metrics(tmp_path: Path) -> None:
    global _seq
    _seq = itertools.count()
    config = _config(tmp_path)
    inputs = _inputs(config, [iter(_full_events())])

    outputs = run(inputs, output_dir=tmp_path / "out", run_id="basic-run")

    assert outputs.manifest_path.exists()
    assert outputs.ledger_path.exists()
    assert outputs.metrics_path.exists()

    manifest = json.loads(outputs.manifest_path.read_text())
    assert manifest["config_hash"] == config.config_hash
    assert manifest["fidelity_tier"] == config.fidelity_tier.value

    ledger_state = json.loads(outputs.ledger_path.read_text())
    assert ledger_state["run_id"] == "basic-run"
    assert "lots" in ledger_state
    assert ledger_state["cash_micro_usd"] < ledger_state["starting_cash_micro_usd"], (
        "the pre-split buy should have spent cash"
    )

    metrics = json.loads(outputs.metrics_path.read_text())
    assert isinstance(metrics, dict)

    assert outputs.result.reports
    assert any(r.fill is not None for r in outputs.result.reports)


# ---------------------------------------------------------------------------
# Snapshot / resume / finish — continuation equivalence
# ---------------------------------------------------------------------------


def test_resume_from_snapshot_matches_uninterrupted_run(tmp_path: Path) -> None:
    # -- Ground truth: one uninterrupted run over the full event stream. --
    global _seq
    _seq = itertools.count()
    config = _config(tmp_path, name="baseline.toml")
    baseline_inputs = _inputs(config, [iter(_full_events())])
    baseline = run(baseline_inputs, output_dir=tmp_path / "baseline", run_id="same-run-id")

    # -- Split run: first half (ticks 0, 1 -> t1, t2), snapshot at the end. --
    _seq = itertools.count()
    config = _config(tmp_path, name="split.toml")
    assert config.config_hash == baseline_inputs.config.config_hash

    first_half = [
        _universe_event(),
        _pool_state_event(),
        _bar_event(0),
        _bar_event(1),
        _decision_event(0),
        _decision_event(1),
    ]
    first_half.sort(key=lambda e: e.sort_key)
    first_inputs = _inputs(config, [iter(first_half)])
    snapshot_path = tmp_path / "snap" / "snapshot.json"
    run(
        first_inputs,
        output_dir=tmp_path / "first",
        run_id="same-run-id",
        snapshot_to=snapshot_path,
        snapshot_interval_ticks=1,
    )
    assert snapshot_path.exists()
    snapshot = load_snapshot(snapshot_path)
    assert snapshot.clock_now == _T2, "the split run's clock must stop at t2"

    # -- Resume: re-feed the foundational universe/pool-state/bar records --
    # (see module docstring) at the resume point, then the genuinely new
    # remaining events (bars 2-4, decision ticks 2-4).
    resumed_prefix = [
        _universe_event(outer_available_time=snapshot.clock_now),
        _pool_state_event(outer_available_time=snapshot.clock_now),
        _bar_event(0, outer_available_time=snapshot.clock_now),
        _bar_event(1, outer_available_time=snapshot.clock_now),
    ]
    remainder = [_bar_event(k) for k in range(2, 5)]
    remainder.extend(_decision_event(k) for k in range(2, 5))
    resumed_stream = resumed_prefix + remainder
    resumed_stream.sort(key=lambda e: e.sort_key)

    second_inputs = _inputs(config, [iter(resumed_stream)])
    second_outputs = run(
        second_inputs,
        output_dir=tmp_path / "second",
        run_id="same-run-id",
        resume_from=snapshot_path,
    )

    # The resumed run's final ledger must match the uninterrupted run's,
    # modulo the documented lot-id-label gap (snapshots.py's module
    # docstring: lot ids are not guaranteed stable across a restart since
    # only open lots are serialized). Every dollar/quantity figure must agree.
    baseline_state = baseline.result.ledger.to_state()
    resumed_state = second_outputs.result.ledger.to_state()
    for key in (
        "cash_micro_usd",
        "starting_cash_micro_usd",
        "realized_pnl_micro_usd",
        "venue_fee_micro_usd",
        "network_fee_micro_usd",
        "priority_fee_micro_usd",
    ):
        assert baseline_state[key] == resumed_state[key], key

    baseline_lots = baseline_state["lots"]
    resumed_lots = resumed_state["lots"]
    assert set(baseline_lots) == set(resumed_lots)
    for symbol, lots in baseline_lots.items():
        resumed_symbol_lots = resumed_lots[symbol]
        assert len(lots) == len(resumed_symbol_lots)
        for base_lot, resumed_lot in zip(lots, resumed_symbol_lots, strict=True):
            for field_name in (
                "symbol",
                "mint",
                "decimals",
                "quantity_atomic",
                "basis_micro_usd",
            ):
                assert base_lot[field_name] == resumed_lot[field_name], (
                    symbol,
                    field_name,
                )

    # Both runs must have seen and settled the same two orders.
    baseline_fills = [r.fill for r in baseline.result.reports if r.fill is not None]
    assert len(baseline_fills) == 2

    # The equity curve's final NAV — the run's headline economic figure —
    # must agree too, even though the resumed run only ever recorded the
    # tail end of it. Both runs are built entirely from flat $1.00 bars and
    # integer-micro-USD arithmetic, so this is an exact float equality, not
    # an approximation.
    assert baseline.result.equity_curve[-1][1] == second_outputs.result.equity_curve[-1][1]


# ---------------------------------------------------------------------------
# The reserved __ids_state__ key round-trips through a saved snapshot
# ---------------------------------------------------------------------------


def test_ids_state_key_round_trips_through_snapshot(tmp_path: Path) -> None:
    global _seq
    _seq = itertools.count()
    config = _config(tmp_path)
    inputs = _inputs(config, [iter(_full_events())])
    snapshot_path = tmp_path / "snap.json"

    run(
        inputs,
        output_dir=tmp_path / "out",
        run_id="ids-state-run",
        snapshot_to=snapshot_path,
        snapshot_interval_ticks=1,
    )

    snapshot = load_snapshot(snapshot_path)
    assert runner_mod._IDS_STATE_KEY in snapshot.rng_states
    ids_state = snapshot.rng_states[runner_mod._IDS_STATE_KEY]
    assert isinstance(ids_state, dict)
    assert ids_state, "expected a non-empty ids_state after minting several ids"

    # And the raw file on disk carries it too, under the same reserved key —
    # confirming this is not an artifact of in-memory object identity.
    raw = json.loads(snapshot_path.read_text())
    assert runner_mod._IDS_STATE_KEY in raw["rng_states"]


# ---------------------------------------------------------------------------
# Regression: the shipped baseline config must actually be fillable.
#
# ``_engine_config`` used to derive ``settlement_latency_seconds`` from
# ``fast_tick_seconds`` alone. For ``configs/backtests/baseline_price_volume
# .toml`` (fast_tick_seconds=300, publication_delay_seconds=60.0, decision
# timeframe 5m per ``cli_backtest._decision_timeframe``) that produced a
# settlement latency of 300s. A bar opened at ``decided_at`` is only knowable
# once ``available_time = ts + interval_seconds(timeframe) +
# publication_delay_seconds <= now`` (``fill_models.BarExecutionModel.fill``),
# and the candidate filter additionally requires ``ts >= decided_at`` — so the
# earliest usable bar needs ``now - decided_at >= interval_seconds(timeframe)
# + publication_delay_seconds`` = 360s here, strictly more than the 300s the
# old expression produced. Every entry therefore raised
# ``NoRoute("no bar opened after decided_at=...")`` forever: this config,
# loaded exactly as shipped, could never produce a single fill.
# ---------------------------------------------------------------------------

_BASELINE_CONFIG_PATH = Path(__file__).resolve().parents[2] / (
    "configs/backtests/baseline_price_volume.toml"
)


def test_engine_config_settlement_latency_covers_decision_timeframe_bar() -> None:
    """Unit-level pin on the exact regression: the derived latency must be at
    least one full decision-timeframe bar plus the publication delay, not just
    ``fast_tick_seconds``."""
    real_config = load_config(_BASELINE_CONFIG_PATH)
    assert real_config.fast_tick_seconds == 300
    assert real_config.publication_delay_seconds == 60.0
    assert Timeframe.M5.value in real_config.timeframes

    engine_config = runner_mod._engine_config(
        real_config, run_id="baseline-latency-check", decision_timeframe=Timeframe.M5
    )

    # interval_seconds(M5) + publication_delay_seconds = 300 + 60 = 360,
    # strictly greater than fast_tick_seconds (300) alone — the old buggy
    # expression's value.
    assert engine_config.settlement_latency_seconds >= 360.0
    assert engine_config.settlement_latency_seconds > real_config.fast_tick_seconds


def test_baseline_config_as_shipped_produces_at_least_one_fill(tmp_path: Path) -> None:
    """End-to-end proof that the real, unmodified baseline config can settle a
    fill. Only the date range is narrowed (to keep the test fast and avoid any
    dependency on real historical data under ``history/``); every cadence,
    publication-delay, and timeframe value is exactly what ships."""
    global _seq
    _seq = itertools.count()

    interval = 300.0  # 5m, matching real_config's decision timeframe
    start_ts = T0 - 1.0
    real_config = load_config(_BASELINE_CONFIG_PATH)
    config = replace(real_config, start_ts=start_ts, end_ts=T0 + 20 * interval)
    config.validate()
    decision_timeframe = Timeframe.M5

    def bar_record(k: int) -> CandleRecord:
        ts = T0 + k * interval
        return CandleRecord(
            asset_id=SYMBOL,
            pool_id=POOL_ID,
            timeframe=decision_timeframe.value,
            ts=ts,
            event_time=ts,
            available_time=ts + interval + config.publication_delay_seconds,
            received_time=ts + interval + config.publication_delay_seconds,
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=1_000_000.0,
            closed=True,
            source="test",
        )

    def bar_event(k: int) -> HistoricalEvent:
        record = bar_record(k)
        return HistoricalEvent(
            kind=EventKind.BAR_CLOSE,
            available_time=record.available_time,
            asset_id=SYMBOL,
            payload=record,
            source="test",
            sequence=_next_seq(),
        )

    def decision_event(available: float) -> HistoricalEvent:
        return HistoricalEvent(
            kind=EventKind.DECISION_TICK,
            available_time=available,
            asset_id=None,
            payload=None,
            source="test",
            sequence=_next_seq(),
        )

    # ``decide_at`` is deliberately pinned to bar 2's own ``ts`` (T0 + 2 *
    # interval), the tightest legal case: settlement_latency_seconds is
    # exactly interval + publication_delay_seconds (see the test above), so
    # the only bar whose `ts >= decided_at` *and* whose `available_time <=
    # decided_at + settlement_latency_seconds` both hold is the one whose
    # `ts` equals `decided_at` exactly. Two bars are needed before it purely
    # so ``price()`` (which reads the *last already-closed* bar as of
    # ``decide_at``, independent of ``decided_at``'s own alignment) has
    # something to quote against — bar 0's available_time (T0 + interval +
    # publication_delay_seconds) is <= decide_at by construction below.
    decide_at = T0 + 2 * interval
    assert decide_at >= T0 + interval + config.publication_delay_seconds

    events = [_universe_event(), _pool_state_event()]
    events.extend(bar_event(k) for k in range(6))
    events.append(decision_event(decide_at))
    events.sort(key=lambda e: e.sort_key)

    strategy = ScriptedStrategy(
        by_tick={
            decide_at: (_buy_intent(usd=60.0, now=decide_at, intent_id="baseline-buy"),)
        }
    )
    usd_token = TokenMeta(mint="USDC", decimals=6, source="test")
    execution_model = BarExecutionModel(
        ExecutionConfig(
            slippage_bps_fallback=100.0,
            gas_usd_per_swap=0.21,
            failed_tx_rate=0.0,
            default_pool_fee_pct=0.25,
            pool_fee_pct={},
        ),
        usd_token=usd_token,
        token_decimals=9,
        timeframe=decision_timeframe,
        participation_cap_pct=1.0,
    )
    inputs = RunnerInputs(
        config=config,
        streams=[iter(events)],
        strategy=strategy,
        execution_model=execution_model,
        risk_engine=_risk_engine(),
        decision_timeframe=decision_timeframe,
    )

    outputs = run(
        inputs, output_dir=tmp_path / "baseline-fillable", run_id="baseline-fillable"
    )

    assert outputs.result.reports
    assert any(r.fill is not None for r in outputs.result.reports), (
        "the real, unmodified baseline_price_volume.toml settlement latency "
        "must allow at least one fill to settle"
    )
