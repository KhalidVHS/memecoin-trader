"""The replay event loop — turns a merged ``HistoricalEvent`` stream into
ledger mutations, one ``available_time`` at a time.

This mirrors ``docs/BACKTEST-CONTRACTS.md`` §5's tick order exactly:

1. Pop the next event from the :class:`~memetrader.backtest.event_queue.EventQueue`
   (already in deterministic ``available_time`` order).
2. Advance the :class:`~memetrader.backtest.clock.SimulatedClock` to that
   event's ``available_time`` — nothing below this point may read data with a
   later ``available_time`` than what the clock now reads.
3. Advance :class:`~memetrader.histdata.point_in_time.ReplayState` to the same
   instant (``state.now = event.available_time``) and, for a data-kind event,
   load its payload into the state store (``add_bar``/``add_pool_state``/
   ``set_universe``).
4. On a ``DECISION_TICK`` event, run the full per-tick pipeline: mark the book
   → advance the continuous risk ledger → evaluate ``ContinuousRisk`` → fire
   any stop-loss breaches as *forced* exits → re-mark if a stop fired → ask
   the strategy for new intents (skipped while halted, since a kill switch
   permits exits but not new entries) → size them via
   ``strategies.construction.size_orders`` → route sells before buys through
   ``backtest.broker.SimulatedBroker`` → schedule a future ``EXECUTION`` event
   for every order the broker approved.
5. On an ``EXECUTION`` event, settle the ``ApprovedOrder`` it carries through
   the broker, apply the resulting ``ExecutionReport`` to the ledger, and run
   the fill-shaped invariants that need the settled ``Fill``/``Quote`` pair.
6. Run ``invariants.check_all`` (and the cheap, always-available subset of the
   five invariants that need engine-level data) after *every* event, not just
   at the end of the run — contracts §5's explicit requirement.

``ReplayState.snapshot()`` always returns ``None`` at TIER_0 — its own
docstring says the execution model must build its own snapshot from bars
rather than relying on it. This engine is that caller: :func:`_synthesize_snapshot`
builds the minimal ``CoinSnapshot`` ``risk.EligibilityRisk.assess`` needs from
whatever bar/pool-state data the replay has actually ingested by ``now``.

**Symbol identity convention** (an explicit design decision, not a frozen
contract): this engine treats ``OrderIntent.symbol`` / ``Position.symbol`` /
``ReplayState``'s ``asset_id`` as the same string — the token mint address.
Every baseline strategy already does this (``state.universe()`` returns
asset_ids and baselines pass them straight through as ``symbol``), so the
engine does not invent a second identity space. A caller wiring in a strategy
that uses a different symbol convention must supply ``mint_for``/``pool_for``
translation tables.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field

from memetrader import ids
from memetrader.backtest import invariants
from memetrader.backtest.broker import BrokerAttempt, SimulatedBroker
from memetrader.backtest.clock import SimulatedClock
from memetrader.backtest.event_queue import EventQueue
from memetrader.backtest.ledger import BacktestLedger
from memetrader.execution.interfaces import ApprovedOrder
from memetrader.histdata.point_in_time import ReplayState
from memetrader.histdata.schemas import CandleRecord, PoolState
from memetrader.portfolio import (
    DEFAULT_MARK_PARAMS,
    Mark,
    MarkParams,
    StopBreach,
    build_mark,
    mark_book,
    stop_loss_breaches,
)
from memetrader.risk import EMPTY_LEDGER, RiskLedger, update_ledger
from memetrader.strategies.baselines import BacktestStrategy
from memetrader.strategies.construction import ConstructionLimits, size_orders
from memetrader.types import (
    Candle,
    CoinSnapshot,
    DecisionRecord,
    EventKind,
    ExecutionMode,
    ExecutionReport,
    Fill,
    HistoricalEvent,
    OrderIntent,
    OrderState,
    PoolRef,
    PortfolioState,
    PriceLadder,
    Provenance,
    Side,
    Timeframe,
    TxnCounts,
)

__all__ = ["EngineConfig", "EngineResult", "ReplayEngine"]


# ---------------------------------------------------------------------------
# Configuration and result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """The knobs one replay run needs that are not already frozen elsewhere.

    ``settlement_latency_seconds`` is a deterministic constant rather than a
    drawn ``execution.latency.LatencyModel`` sample: the assigned scope is the
    event loop, and a real latency model's RNG draws would need their state
    captured in every snapshot for exact resume, which is out of scope for
    this pass. A caller that wants drawn latency can still get it — see the
    module docstring's note in the final report — by pre-scheduling
    ``EXECUTION`` events at model-drawn times instead of relying on this
    field; the loop only requires ``ready_at > decided_at``, however it was
    computed.

    ``rebalance_band_usd``/``min_trade_usd`` mirror the live loop's
    ``StrategySettings.rebalance_band_usd``/``.min_trade_usd``
    (``src/memetrader/strategy.py``) and default to the same values (15.0 /
    10.0). They live here, on ``EngineConfig``, rather than on
    ``backtest.config.BacktestConfig``: that dataclass carries no such fields
    today, and adding them is outside this module's assigned scope. A caller
    that wants a run's band/floor to track a particular
    ``BacktestConfig``/live-strategy value is responsible for reading it and
    passing it through when constructing this ``EngineConfig`` — see
    ``runner.py``'s ``_engine_config``.
    """

    run_id: str
    starting_cash_micro_usd: int
    stop_loss_pct: float = 0.15
    settlement_latency_seconds: float = 1.0
    decision_timeframe: Timeframe = Timeframe.H1
    mark_params: MarkParams = DEFAULT_MARK_PARAMS
    construction_limits: ConstructionLimits = field(default_factory=ConstructionLimits)
    rebalance_band_usd: float = 15.0
    min_trade_usd: float = 10.0


@dataclass(frozen=True, slots=True)
class EngineResult:
    """What one call to :meth:`ReplayEngine.run` produced."""

    ledger: BacktestLedger
    reports: tuple[ExecutionReport, ...]
    risk_ledger: RiskLedger
    ticks_checked: int
    equity_curve: tuple[tuple[float, float], ...]
    """``(decision_tick_ts, total_value_usd)`` pairs, one per ``DECISION_TICK``
    processed, in chronological order — recorded pre-settlement (any orders
    routed on that tick have not yet landed), which is the run's authoritative
    NAV series for ``metrics.performance.compute_performance``."""
    decisions: tuple[DecisionRecord, ...] = ()
    """One :class:`~memetrader.types.DecisionRecord` per ``DECISION_TICK``
    processed (contracts §5 step 10), appended by ``_run_decision_tick``
    regardless of whether that tick produced any order. ``targets`` is always
    ``()``: ``BacktestStrategy.propose()`` — the already-blessed narrowing of
    the live ``Strategy.decide()``/``StrategyDecision`` surface — never
    produces ``TargetPosition``s, so there is nothing honest to put there.
    ``market_read`` is a fixed placeholder string for the same reason: no
    baseline strategy narrates a market read. ``bounds`` carries every
    :class:`~memetrader.types.RiskBounds` the tick's routed attempts (forced
    stop-loss exits included) actually received from the broker. ``fills`` is
    always ``()`` here — a decision's fills, if any, settle on a later
    ``EXECUTION`` event once ``settlement_latency_seconds`` elapses, not
    synchronously within the tick that decided them, so there is nothing to
    record at construction time."""


# ---------------------------------------------------------------------------
# Snapshot synthesis — the bridge ReplayState.snapshot()'s docstring asks for
# ---------------------------------------------------------------------------

_EMPTY_TXNS = TxnCounts(buys=None, sells=None)


def _synthesize_snapshot(
    symbol: str,
    *,
    mint: str,
    pool_id: str | None,
    state: ReplayState,
    now: float,
    timeframe: Timeframe,
) -> CoinSnapshot | None:
    """Build the minimal ``CoinSnapshot`` an entry/exit risk check needs.

    Returns ``None`` (honestly: no data) when no closed bar has ever been
    ingested for ``symbol`` at ``timeframe`` by ``now`` — that correctly
    blocks entries through ``EligibilityRisk.assess``'s unconditional
    ``missing_snapshot`` veto, and is not a workaround for it.
    """
    bars = state.bars(mint, timeframe, lookback=1)
    if not bars:
        return None
    last = bars[-1]
    pool_state = state.pool_state(pool_id) if pool_id else None
    price = (
        pool_state.price_usd
        if pool_state is not None and pool_state.price_usd is not None
        else last.close
    )
    liquidity = pool_state.liquidity_usd if pool_state is not None else None
    pool_ref = PoolRef(
        pair_address=pool_id or f"synthetic:{mint}",
        dex_id="replay",
        base_mint=mint,
        quote_mint="",
        quote_symbol="",
        trusted_quote=True,
    )
    provenance = Provenance(
        source="replay-synthesized",
        receive_time=now,
        event_time=last.ts,
        available_time=now,
    )
    return CoinSnapshot(
        symbol=symbol,
        mint=mint,
        price_usd=price,
        liquidity_usd=liquidity,
        volume_24h_usd=None,
        volume_1h_usd=None,
        fdv_usd=None,
        price_change=PriceLadder(m5=None, h1=None, h6=None, h24=None),
        txns_m5=_EMPTY_TXNS,
        txns_h1=_EMPTY_TXNS,
        txns_h24=_EMPTY_TXNS,
        pool=pool_ref,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


@dataclass
class ReplayEngine:
    """Drives one replay from an ``EventQueue`` to a settled ``BacktestLedger``.

    Construction is intentionally low-level: the caller (``runner.py`` or a
    test) builds the ``SimulatedClock``, the ``EventQueue`` and its streams,
    the ``ReplayState``, the ``BacktestLedger`` and the ``SimulatedBroker``
    ahead of time and hands them all in. This engine does not know how to
    load a catalog or a config file — that is ``runner.py``'s job.
    """

    clock: SimulatedClock
    queue: EventQueue
    state: ReplayState
    ledger: BacktestLedger
    broker: SimulatedBroker
    strategy: BacktestStrategy
    config: EngineConfig
    mint_for: dict[str, str] = field(default_factory=dict)
    pool_for: dict[str, str] = field(default_factory=dict)
    risk_ledger: RiskLedger = EMPTY_LEDGER
    dry_run: bool = False

    reports: list[ExecutionReport] = field(default_factory=list, init=False)
    ticks_checked: int = field(default=0, init=False)
    equity_curve: list[tuple[float, float]] = field(default_factory=list, init=False)
    decision_records: list[DecisionRecord] = field(default_factory=list, init=False)
    _tick_fills: list[Fill] = field(default_factory=list, init=False, repr=False)
    _exec_seq: itertools.count = field(
        default_factory=itertools.count, init=False, repr=False
    )

    # -- entry point ---------------------------------------------------------

    def run(
        self,
        *,
        on_clock_active: Callable[[SimulatedClock], None] | None = None,
        on_tick: Callable[[ReplayEngine], None] | None = None,
    ) -> EngineResult:
        """Run to completion inside the ``SimulatedClock`` context.

        ``on_clock_active`` fires immediately after the clock's wall-clock
        guard is installed and before any event is processed. It exists for
        crash-restart resume: ``ids.restore_ids_state`` and
        ``snapshots.restore_random_state`` both require an *active* clock,
        which only exists once ``self.clock.__enter__`` has run — this hook
        is the one point in this method where that is true and no event has
        been touched yet.

        ``on_tick`` fires after every event is handled and checked, while the
        clock is still active — this is what lets a caller (``runner.py``)
        take a mid-run snapshot: ``ids.ids_state()`` and an RNG's consumed
        state are only meaningful to capture while the clock they were minted
        under is still the active one.
        """
        with self.clock as clock:
            if on_clock_active is not None:
                on_clock_active(clock)
            for event in self.queue:
                clock.advance_to(event.available_time)
                self.state.now = event.available_time
                self._handle_event(event)
                invariants.check_all(self.ledger)
                invariants.check_risk_resize_requotes(self.broker.resize_pairs)
                self.ticks_checked += 1
                if on_tick is not None:
                    on_tick(self)
        return EngineResult(
            ledger=self.ledger,
            reports=tuple(self.reports),
            risk_ledger=self.risk_ledger,
            ticks_checked=self.ticks_checked,
            equity_curve=tuple(self.equity_curve),
            decisions=tuple(self.decision_records),
        )

    # -- per-event dispatch ----------------------------------------------------

    def _handle_event(self, event: HistoricalEvent) -> None:
        now = event.available_time
        if event.kind is EventKind.BAR_CLOSE:
            self._ingest_bar(event)
        elif event.kind is EventKind.POOL_STATE:
            self._ingest_pool_state(event)
        elif event.kind is EventKind.UNIVERSE:
            self._ingest_universe(event)
        elif event.kind is EventKind.DECISION_TICK:
            self._run_decision_tick(now)
        elif event.kind is EventKind.EXECUTION:
            self._settle(event, now=now)
        # QUOTE/SOCIAL/MARK/SWAP/ORDER_READY: no loader in the frozen
        # dependency surface produces these today (only OHLCV and pool-state
        # loaders exist — see the final report). Left as structural no-ops
        # rather than guessed-at handling, so a future loader for one of them
        # only needs a branch added here, not a redesign.

    def _ingest_bar(self, event: HistoricalEvent) -> None:
        record = event.payload
        if not isinstance(record, CandleRecord):
            return
        candle = Candle(
            ts=record.ts,
            open=record.open,
            high=record.high,
            low=record.low,
            close=record.close,
            volume=record.volume,
            closed=True,
        )
        timeframe = Timeframe(record.timeframe)
        self.state.add_bar(
            candle,
            asset_id=record.asset_id,
            timeframe=timeframe,
            available_time=record.available_time,
        )
        self.pool_for.setdefault(record.asset_id, record.pool_id)

    def _ingest_pool_state(self, event: HistoricalEvent) -> None:
        payload = event.payload
        if not isinstance(payload, PoolState):
            return
        self.state.add_pool_state(payload)
        if payload.asset_id:
            self.pool_for.setdefault(payload.asset_id, payload.pool_id)

    def _ingest_universe(self, event: HistoricalEvent) -> None:
        payload = event.payload
        if isinstance(payload, (frozenset, set, tuple, list)):
            members = frozenset(str(m) for m in payload)
            self.state.set_universe(members, available_time=event.available_time)

    # -- the decision tick pipeline (contracts §5) ------------------------------

    def _run_decision_tick(self, now: float) -> None:
        book = self._mark_book(now=now)
        self.risk_ledger = update_ledger(
            self.risk_ledger, book=book, fills=tuple(self._tick_fills), now=now
        )
        self._tick_fills = []
        risk_state = self.broker.risk_engine.continuous.evaluate(
            book=book, ledger=self.risk_ledger, data_quality_ok=True, now=now
        )

        tick_attempts: list[BrokerAttempt] = []

        # Forced stop-loss exits precede everything else the tick might do.
        breaches = stop_loss_breaches(
            book, self.config.stop_loss_pct, params=self.config.mark_params
        )
        fired = False
        for breach in breaches:
            intent = self._stop_exit_intent(breach, book, now=now)
            if intent is None:
                continue
            fired = True
            tick_attempts.append(
                self._route_exit(
                    intent, book=book, risk_state=risk_state, now=now, forced=True
                )
            )

        if fired:
            # Re-mark and re-evaluate: a stop that just fired changed cash and
            # inventory, and the rest of this tick must see the post-stop book.
            book = self._mark_book(now=now)
            risk_state = self.broker.risk_engine.continuous.evaluate(
                book=book, ledger=self.risk_ledger, data_quality_ok=True, now=now
            )

        if risk_state.halted:
            # The kill switch permits exits, never new entries (contracts §5 /
            # risk.py's ContinuousRisk docstring).
            proposed: tuple[OrderIntent, ...] = ()
        else:
            proposed = self.strategy.propose(
                state=self.state, portfolio=book, now=now, run_id=self.config.run_id
            )

        existing_exposure = {
            symbol: self._exposure_micro(book, symbol) for symbol in book.positions
        }
        sized = size_orders(
            proposed,
            cash_micro_usd=round(book.cash_usd * 1_000_000),
            portfolio_value_micro_usd=round((book.total_value_usd or 0.0) * 1_000_000),
            existing_exposure_micro_usd=existing_exposure,
            positions=book.positions,
            universe=self.state.universe(),
            limits=self.config.construction_limits,
        )

        # Contracts §5 step 9: diff (here, the already-sized order) against
        # inventory and drop it if the delta it represents is too small to be
        # worth paying spread/impact/gas for — the same band the live loop
        # applies in ``loop.py._diff_targets``, same order (band, then the
        # separate min-trade floor), just applied to a sized ``OrderIntent``
        # rather than a ``TargetPosition`` since ``BacktestStrategy.propose()``
        # does not produce targets. Forced stop-loss exits above are exempt by
        # construction — they never go through ``sized``.
        sized = tuple(
            intent for intent in sized if self._passes_rebalance_band(intent, book=book)
        )

        # Sells before buys — freeing cash/quarantine slots before anything
        # competes for the freshly-opened headroom.
        tick_attempts.extend(
            self._route_exit(
                intent, book=book, risk_state=risk_state, now=now, forced=False
            )
            for intent in sized
            if intent.side is Side.SELL
        )
        tick_attempts.extend(
            self._route_entry(intent, book=book, risk_state=risk_state, now=now)
            for intent in sized
            if intent.side is Side.BUY
        )

        equity_value = (
            book.total_value_usd if book.total_value_usd is not None else book.cash_usd
        )
        self.equity_curve.append((now, equity_value))

        # Contracts §5 step 10: append a DecisionRecord for this tick. See
        # EngineResult.decisions's docstring for exactly what is (and is not)
        # populated and why.
        self.decision_records.append(
            DecisionRecord(
                decision_id=ids.new_decision_id(),
                run_id=self.config.run_id,
                ts=now,
                strategy_id=self.strategy.strategy_id,
                market_read=(
                    "backtest replay: BacktestStrategy.propose() emits sized "
                    "OrderIntents directly, not a StrategyDecision with a market "
                    "narrative — no market_read text exists at this fidelity."
                ),
                targets=(),
                bounds=tuple(attempt.bounds for attempt in tick_attempts),
                intents=tuple(sized),
                fills=(),
                mode=ExecutionMode.PAPER,
                risk_state=risk_state,
            )
        )

    def _order_delta_usd(self, intent: OrderIntent, *, book: PortfolioState) -> float:
        """The USD size of ``intent``, for rebalance-band comparison.

        A BUY's ``in_amount_atomic`` is already micro-USD (the convention
        documented in ``strategies/baselines.py``). A SELL's is token-atomic,
        so it is converted via the fraction of the held position it disposes
        of, applied to that position's currently marked USD value — avoiding
        a second, independent price lookup that could disagree with the mark
        the rest of this tick is using.
        """
        if intent.side is Side.BUY:
            return intent.in_amount_atomic / 1_000_000.0
        position = book.positions.get(intent.symbol)
        if position is None or position.quantity_atomic <= 0:
            return 0.0
        exposure_usd = self._exposure_micro(book, intent.symbol) / 1_000_000.0
        fraction = min(1.0, intent.in_amount_atomic / position.quantity_atomic)
        return exposure_usd * fraction

    def _passes_rebalance_band(self, intent: OrderIntent, *, book: PortfolioState) -> bool:
        """Contracts §5 step 9's filter, mirroring ``loop.py._diff_targets``:
        a delta under the band is left alone, and — checked separately, same
        as the live loop — so is one under the flat minimum-trade floor.
        """
        delta_usd = abs(self._order_delta_usd(intent, book=book))
        if delta_usd < self.config.rebalance_band_usd:
            return False
        return not delta_usd < self.config.min_trade_usd

    def _stop_exit_intent(
        self, breach: StopBreach, book: PortfolioState, *, now: float
    ) -> OrderIntent | None:
        position = book.positions.get(breach.symbol)
        if position is None or position.quantity_atomic <= 0:
            return None
        return OrderIntent(
            intent_id=ids.new_intent_id(),
            decision_id=None,
            action_id=None,
            run_id=self.config.run_id,
            ts=now,
            symbol=breach.symbol,
            side=Side.SELL,
            in_amount_atomic=position.quantity_atomic,
            max_in_amount_atomic=position.quantity_atomic,
            source="stop_loss",
            reason=breach.reason,
        )

    def _mark_book(self, *, now: float) -> PortfolioState:
        snapshot = self.ledger.snapshot(ts=now)
        marks: dict[str, Mark] = {}
        for symbol in snapshot.positions:
            price = self._mid_price(symbol, now=now)
            provenance = Provenance(
                source="replay-mark", receive_time=now, event_time=now, available_time=now
            )
            marks[symbol] = build_mark(
                symbol,
                mid_price_usd=price,
                mid_provenance=provenance,
                params=self.config.mark_params,
                now=now,
            )
        return mark_book(
            cash_usd=snapshot.cash_usd,
            positions=snapshot.positions,
            marks=marks,
            realized_pnl_usd=snapshot.realized_pnl_usd,
            starting_cash_usd=snapshot.starting_cash_usd,
            fees_paid_usd=snapshot.fees_paid_usd,
            gas_paid_usd=snapshot.gas_paid_usd,
            now=now,
        )

    def _mid_price(self, symbol: str, *, now: float) -> float | None:
        del now  # ReplayState.bars/pool_state already gate on state.now
        bars = self.state.bars(symbol, self.config.decision_timeframe, lookback=1)
        if bars:
            return bars[-1].close
        pool_id = self.pool_for.get(symbol)
        pool_state = self.state.pool_state(pool_id) if pool_id else None
        return pool_state.price_usd if pool_state is not None else None

    def _exposure_micro(self, book: PortfolioState, symbol: str) -> int:
        value = book.position_values_usd.get(symbol)
        if value is None:
            position = book.positions.get(symbol)
            value = position.cost_basis_usd if position is not None else 0.0
        return max(0, round(value * 1_000_000))

    def _snapshot_for(self, symbol: str, *, now: float) -> CoinSnapshot | None:
        mint = self.mint_for.get(symbol, symbol)
        pool_id = self.pool_for.get(symbol)
        return _synthesize_snapshot(
            symbol,
            mint=mint,
            pool_id=pool_id,
            state=self.state,
            now=now,
            timeframe=self.config.decision_timeframe,
        )

    # -- routing through the broker, then scheduling settlement -----------------

    def _route_exit(
        self,
        intent: OrderIntent,
        *,
        book: PortfolioState,
        risk_state: object,
        now: float,
        forced: bool,
    ) -> BrokerAttempt:
        snapshot = self._snapshot_for(intent.symbol, now=now)
        attempt = self.broker.attempt_exit(
            intent,
            state=self.state,
            book=book,
            risk_state=risk_state,  # type: ignore[arg-type]
            snapshot=snapshot,
            forced=forced,
            now=now,
        )
        self._schedule(attempt, now=now)
        return attempt

    def _route_entry(
        self, intent: OrderIntent, *, book: PortfolioState, risk_state: object, now: float
    ) -> BrokerAttempt:
        snapshot = self._snapshot_for(intent.symbol, now=now)
        attempt = self.broker.attempt_entry(
            intent,
            state=self.state,
            book=book,
            risk_state=risk_state,  # type: ignore[arg-type]
            snapshot=snapshot,
            ledger=self.risk_ledger,
            now=now,
        )
        self._schedule(attempt, now=now)
        return attempt

    def _schedule(self, attempt: BrokerAttempt, *, now: float) -> None:
        if not attempt.permitted:
            return
        approved = attempt.approved
        assert approved is not None  # `.permitted` guarantees this
        ready_at = now + self.config.settlement_latency_seconds
        event = HistoricalEvent(
            kind=EventKind.EXECUTION,
            available_time=ready_at,
            asset_id=approved.intent.symbol,
            payload=approved,
            source="engine:settlement",
            sequence=next(self._exec_seq),
        )
        self.queue.push(event)

    # -- settlement --------------------------------------------------------

    def _settle(self, event: HistoricalEvent, *, now: float) -> None:
        approved = event.payload
        if not isinstance(approved, ApprovedOrder):
            return
        report = self.broker.settle(approved, state=self.state, now=now)
        mint = self.mint_for.get(approved.intent.symbol, approved.intent.symbol)

        if self.dry_run:
            invariants.check_dry_run_no_mutation(self.ledger, report)
        elif report.state in (OrderState.FAILED, OrderState.EXPIRED):
            # check_execution_failure_no_position_change *is* the application
            # step here (invariant #8) — it applies the report itself and
            # asserts no position moved, only cash/gas.
            invariants.check_execution_failure_no_position_change(self.ledger, report)
        else:
            self.ledger.apply_fill(report, mint=mint)

        if report.fill is not None:
            invariants.check_sell_quantity_matches_quote(report.fill, approved.quote)
            if not self.dry_run:
                self._tick_fills.append(report.fill)

        self.reports.append(report)
