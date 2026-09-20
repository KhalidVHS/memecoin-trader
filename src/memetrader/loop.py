"""The dual-cadence scheduler, and the only place the modules meet.

A **fast tick** every 60 seconds re-reads prices, re-marks the book and
enforces stops. No strategy call, so it is nearly free. It exists because a
-15% stop that only checks every 15 minutes is not a stop; by the time a
memecoin decides to move, fifteen minutes is several lifetimes.

A **slow tick** every 15 minutes builds the evidence bundle, asks the strategy
for target positions, turns the difference between targets and inventory into
orders, and puts each one through risk before it is quoted for execution.

Because the two cadences interleave, anything measured *between* reads has to
name which cadence it belongs to. The liquidity trend the strategy sees spans
one decision interval, measured against ``decision_baseline``; measuring it
against the freshest read of any kind — which is what this used to do — made it
a 60-second delta wearing the decision cadence's label.

What the audit changed here
---------------------------

The old loop asked a language model for a list of BUY/SELL actions and executed
them. Four of the twelve critical findings lived in the seam:

* **C3** — risk could shrink an order *after* it was quoted and the broker then
  filled the reduced size against the original-size quote. The order of
  operations is now fixed and is the spine of :meth:`_execute`: risk returns a
  *bound*, the bound sets the size, the size gets its **own** quote, and
  :meth:`RiskEngine.confirm_quote` re-checks that quote before anything is
  placed. A quote is never reused across a size change.
* **C11** — a crash between "trade appended" and "state saved" left two files
  with no way to tell which rows belonged together. Every order now has an
  intent ID minted and journaled *before* the side effect, and startup refuses
  to trade while any intent is unreconciled.
* **C12** — ``--dry-run`` could still liquidate, because the dry-run flag was
  checked in some paths and not others. The flag is gone. Mode is a property of
  the broker (``ExecutionMode``), it is decided once at construction, and a
  READ_ONLY broker raises rather than mutating. A boolean that must be checked
  everywhere will eventually not be checked somewhere.
* **C6** — the model no longer selects or sizes orders. A :class:`Strategy`
  emits *target positions*; this file diffs targets against inventory. Which
  strategy runs is config, and the default is the deterministic baseline.

The resulting shape is: **the strategy proposes an inventory, risk bounds what
may be traded toward it, and the broker executes exactly what was quoted.**
"""

from __future__ import annotations

import logging
import signal
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from . import brain, journal, market, portfolio, quotes, risk, sentiment, signals, strategy
from .broker import BrokerError, LocalPaperBroker
from .config import Config
from .http import CircuitBreaker, make_client
from .ids import new_action_id, new_intent_id
from .journal import Ledger
from .sentiment import SentimentSettings
from .types import (
    DecisionRecord,
    EvidenceBundle,
    ExecutionMode,
    Fill,
    Mark,
    MarketSnapshot,
    OrderIntent,
    OrderSource,
    OrderState,
    PoolRef,
    PortfolioState,
    Quote,
    RiskBounds,
    RiskState,
    Side,
    StrategyDecision,
    TokenMeta,
    ValuationEstimate,
)

log = logging.getLogger("memetrader")


@dataclass(frozen=True, slots=True)
class TickResult:
    """What one tick did.

    A tick that failed before reaching the strategy still returns a result with
    ``decision is None`` and ``error`` set, because a failed tick and a tick
    that decided to hold nothing must not render the same. Conflating them is
    how an outage reads as a flat market in the record afterwards.
    """

    ts: float
    kind: str  # "fast" | "slow"
    mode: ExecutionMode
    portfolio: PortfolioState
    risk_state: RiskState
    evidence: dict[str, EvidenceBundle] = field(default_factory=dict)
    decision: StrategyDecision | None = None
    bounds: tuple[RiskBounds, ...] = ()
    intents: tuple[OrderIntent, ...] = ()
    fills: tuple[Fill, ...] = ()
    stop_exits: tuple[str, ...] = ()
    usage: Any | None = None
    error: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def traded(self) -> bool:
        return any(not f.failed for f in self.fills)


class StartupRefusal(RuntimeError):
    """Raised when the persisted record is not safe to trade on top of.

    Not a warning. The audit's C11 scenario is a process that died mid-order and
    a successor that cheerfully places the order again, and the only correct
    response to "I cannot tell what happened" is to stop.
    """


class _BrainAdapter:
    """Gives :class:`strategy.AdvisoryStrategy` the ``decide()`` it looks for.

    ``brain.advise`` deliberately takes its settings as explicit arguments and
    imports no ``Config`` — that is what lets it be tested without a TOML file.
    Binding those arguments is this file's job, because this file is where
    configuration and behaviour are allowed to meet.

    Token usage accumulates here rather than being returned through the
    strategy, because a strategy that had to carry a bill would have to know it
    was a model.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.last_usage: Any | None = None

    def decide(
        self,
        evidence: Mapping[str, EvidenceBundle],
        book: PortfolioState,
        *,
        now: float,
        history: tuple[DecisionRecord, ...] = (),
        bounds: tuple[RiskBounds, ...] = (),
    ) -> Any:
        cfg = self.cfg
        advice, usage = brain.advise(
            evidence,
            book,
            history,
            bounds,
            symbols=cfg.symbols,
            model=cfg.model,
            risk=cfg.risk,
            starting_cash_usd=cfg.starting_cash_usd,
            cadence=cfg.cadence,
            api_key=cfg.anthropic_api_key,
            decision_history=cfg.prompt.decision_history,
            now=now,
        )
        self.last_usage = usage
        return advice


class Trader:
    """Owns the book, the HTTP client, the risk ledger and the tick cadence."""

    def __init__(
        self,
        cfg: Config,
        *,
        client: httpx.Client | None = None,
        broker: LocalPaperBroker | None = None,
        strategy_impl: strategy.Strategy | None = None,
        breaker: CircuitBreaker | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.cfg = cfg
        self._now = now
        self._client = client
        self._owns_client = client is None

        # One breaker per Trader, not a module-level singleton. A singleton
        # would make two Traders in one process (the test suite, `status`
        # running alongside `run`) share outage state they did not observe, and
        # "this host is down" is an observation, not a global fact.
        self.breaker = (
            breaker
            if breaker is not None
            else CircuitBreaker(
                failure_threshold=cfg.http.breaker_failure_threshold,
                cooldown_seconds=cfg.http.breaker_cooldown_seconds,
            )
        )

        self.broker = (
            broker
            if broker is not None
            else LocalPaperBroker(
                cfg,
                mode=cfg.execution_mode,
                max_price_impact_pct=cfg.risk.max_price_impact_pct,
            )
        )
        self.run_id = self.broker.run_id
        self.ledger = Ledger(cfg.ledger_path, run_id=self.run_id)

        self.risk = risk.RiskEngine(cfg.risk)
        self.risk_ledger = _load_risk_ledger(cfg)

        if strategy_impl is not None:
            self.strategy = strategy_impl
            self._brain: _BrainAdapter | None = None
        else:
            self._brain = _BrainAdapter(cfg) if cfg.strategy_kind == "advisory" else None
            self.strategy = strategy.build_strategy(
                cfg.strategy_kind, cfg.strategy, brain=self._brain
            )

        self.sentiment_settings = SentimentSettings.from_config(cfg)

        # The freshest read of any kind. Not used for cross-snapshot deltas —
        # see decision_baseline.
        self.previous: MarketSnapshot | None = None
        # What the next decision measures liquidity against: the snapshot the
        # strategy last saw, one decision interval back. When both ticks shared
        # one baseline the compared pair was 60 seconds apart, so a pool
        # shedding 10% of its depth between decisions rendered as -0.7% and read
        # as noise.
        self.decision_baseline: MarketSnapshot | None = None

        self._pools: dict[str, PoolRef] = {}
        self._tokens: dict[str, TokenMeta] = {}
        self._stop = False
        self._startup_notes: tuple[str, ...] = ()

    # -- startup ----------------------------------------------------------

    def preflight(self) -> tuple[str, ...]:
        """Reconcile the persisted record and refuse to trade on a broken one.

        Three questions, in the order that matters:

        1. Does the broker's own ledger replay into the state it claims? The
           broker answers this in ``load()``/``reconcile()``.
        2. Is the journal intact — no torn final line, no unparseable rows? A
           torn line is the signature of a crash mid-append and means the last
           thing that happened is exactly the thing we cannot read.
        3. Is any intent still open? An intent with no terminal state is an
           order whose outcome is unknown. Re-running the loop would either
           place it twice or abandon it, and there is no third option available
           from inside this process.

        Notes are returned rather than printed so the CLI can render them and
        the tests can assert on them.
        """
        notes: list[str] = []
        recon = self.broker.reconcile()
        if not recon.clean:
            notes.append(
                f"broker ledger replayed {len(recon.replayed_fill_ids)} fills to rebuild state"
            )

        scan = self.ledger.scan()
        if scan.torn_final_line:
            notes.append(
                f"ledger had a torn final line ({scan.torn_final_bytes} bytes) — "
                f"the process died mid-append; that row is discarded"
            )
        if scan.corrupt_line_numbers:
            notes.append(
                f"ledger has {len(scan.corrupt_line_numbers)} unparseable rows at lines "
                f"{list(scan.corrupt_line_numbers)[:5]}"
            )

        # Two sources, two shapes. The broker's reconciliation returns the
        # `OrderIntent`s it journaled and cannot match to a fill; the ledger
        # returns `OpenIntent` summaries that also carry the last state seen.
        # They are described separately because the second knows something the
        # first does not, and flattening them would throw that away.
        broker_open = tuple(recon.open_intents)
        ledger_open = self.ledger.open_intents()
        described = [
            f"{oi.intent_id} ({oi.symbol} {oi.side.value}, unfilled)" for oi in broker_open
        ]
        described += [
            f"{oi.intent_id} ({oi.symbol} {oi.side}, last state {oi.last_state})"
            for oi in ledger_open
        ]
        if described:
            detail = ", ".join(described[:5])
            raise StartupRefusal(
                f"{len(described)} order intent(s) have no terminal state: {detail}. "
                f"Their outcome is unknown, so placing new orders risks duplicating "
                f"them. Resolve them in {self.cfg.ledger_path} (or `memetrader reset` "
                f"if this is a paper book you are willing to discard) before trading."
            )

        self._startup_notes = tuple(notes)
        return self._startup_notes

    # -- plumbing ---------------------------------------------------------

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            # Via http.make_client: verifies against the OS trust store, which
            # is the only store with the corporate TLS-inspection CA in it. A
            # bare httpx.Client dies with CERTIFICATE_VERIFY_FAILED here.
            self._client = make_client(self.cfg.http.timeouts)
        return self._client

    def close(self) -> None:
        _save_risk_ledger(self.cfg, self.risk_ledger)
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def __enter__(self) -> Trader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- market data ------------------------------------------------------

    def pools(self) -> dict[str, PoolRef]:
        """Resolved once per process. A pool that migrates mid-run is a real
        event, but re-resolving every tick would let a single bad DexScreener
        response silently repoint a position's price source, which is worse."""
        if not self._pools:
            self._pools = market.resolve_pairs(
                self.cfg.coins, self.cfg.market, client=self.client, now=self._now
            )
        return self._pools

    def snapshot(self, *, with_candles: bool = True) -> MarketSnapshot:
        return market.snapshot(
            self.cfg.coins,
            self.cfg.market,
            client=self.client,
            with_candles=with_candles,
            now=self._now,
        )

    def token(self, symbol: str, mint: str) -> TokenMeta | None:
        """Decimals for a mint, cached. ``None`` means we could not find out —
        and an unknown decimal count makes every atomic amount for that token a
        guess, so callers must refuse rather than assume 9 or 6."""
        cached = self._tokens.get(mint)
        if cached is not None:
            return cached
        meta = quotes.token_meta(self.cfg, mint, client=self.client)
        if meta is None:
            log.error(
                "%s: token decimals unavailable for %s; cannot size an order", symbol, mint
            )
            return None
        self._tokens[mint] = meta
        return meta

    # -- evidence ---------------------------------------------------------

    def evidence(self, snap: MarketSnapshot) -> dict[str, EvidenceBundle]:
        """Assemble the streams. Each degrades independently, and a stream that
        failed is reported as explicitly unavailable rather than as a neutral
        value — the whole "missing is never zero" invariant is enforced by
        every consumer downstream, and it only works if this function is honest
        about which of the two happened."""
        briefs: dict[str, Any] = {}
        reasons: dict[str, str | None] = {}
        if self.sentiment_settings.enabled:
            try:
                briefs = sentiment.briefs(self.cfg.coins, self.sentiment_settings)
            except Exception as exc:  # pragma: no cover - briefs catches its own
                log.warning("sentiment unavailable: %s", exc)
                reasons = dict.fromkeys(self.cfg.symbols, f"sentiment lookup failed: {exc}")
        else:
            reasons = dict.fromkeys(
                self.cfg.symbols,
                "sentiment stream disabled — no measured contribution (audit C7/§7)",
            )

        baseline = self.decision_baseline
        # The real gap, not ``cadence.slow_tick_seconds``. A long tick, a vendor
        # outage or a restart all move it, and the liquidity trend is labelled
        # with this number. Claiming 15 minutes over a 3-hour gap would
        # misdescribe the single signal the system ranks highest.
        elapsed = None if baseline is None else snap.ts - baseline.ts

        bundles: dict[str, EvidenceBundle] = {}
        for sym, coin in snap.coins.items():
            prev = baseline.coins.get(sym) if baseline is not None else None
            try:
                tech = signals.brief(coin, prev, elapsed_seconds=elapsed)
            except Exception as exc:
                log.warning("%s technicals failed: %s", sym, exc)
                tech = None
            b = briefs.get(sym)
            bundles[sym] = EvidenceBundle(
                symbol=sym,
                snapshot=coin,
                technicals=tech,
                sentiment=b,
                sentiment_unavailable_reason=(
                    reasons.get(sym)
                    or (None if b is not None else "no sentiment data returned")
                ),
            )
        return bundles

    # -- marking ----------------------------------------------------------

    def marks(self, snap: MarketSnapshot, *, now: float) -> dict[str, Mark]:
        """What each held position is worth, and on what basis.

        Audit C8: the old code marked at the DexScreener mid and, when that was
        missing, at cost. Both are wrong in the same direction — they report a
        number you cannot sell into, and marking at cost reports a loss as zero.
        A mark now carries its basis and its haircut, and a position that cannot
        be priced comes back with basis ``unavailable``, which propagates a
        ``None`` total all the way to the report rather than a plausible sum.

        The route quote is the real answer: it is what a router says it would
        actually pay for the exact size held. It costs one Jupiter call per
        position per tick, which is why the mid is kept as the fallback rather
        than as the default.
        """
        out: dict[str, Mark] = {}
        for symbol, position in self.broker.get_positions().items():
            if position.quantity_atomic <= 0:
                continue
            coin = snap.coins.get(symbol)
            mid = coin.price_usd if coin is not None else None
            mid_prov = coin.provenance if coin is not None else None
            route: Quote | None = None
            estimate: ValuationEstimate | None = None
            try:
                priced = quotes.mark_route(
                    self.cfg,
                    symbol=symbol,
                    mint=position.mint,
                    quantity_atomic=position.quantity_atomic,
                    mid_price_usd=mid,
                    client=self.client,
                    now=now,
                )
            except Exception as exc:
                log.warning("%s mark route failed: %s", symbol, exc)
                priced = None
            if isinstance(priced, Quote):
                route = priced
            elif isinstance(priced, ValuationEstimate):
                estimate = priced

            out[symbol] = portfolio.build_mark(
                symbol,
                route=route,
                mid_price_usd=mid,
                mid_provenance=mid_prov,
                estimate=estimate,
                params=self.cfg.mark,
                now=now,
            )
        return out

    def book(self, snap: MarketSnapshot, *, now: float) -> PortfolioState:
        return portfolio.mark_book(
            cash_usd=self.broker.cash_usd,
            positions=self.broker.get_positions(),
            marks=self.marks(snap, now=now),
            realized_pnl_usd=self.broker.realized_pnl_usd,
            starting_cash_usd=self.broker.starting_cash_usd,
            fees_paid_usd=self.broker.fees_paid_usd,
            gas_paid_usd=self.broker.gas_paid_usd + self.broker.failed_gas_usd,
            now=now,
        )

    def _data_health_ok(self) -> bool:
        """Whether the feeds are healthy enough to open risk against.

        An open circuit breaker means a vendor has failed repeatedly and is
        being left alone. Trading through that is trading on a book marked from
        whatever happened to be cached, which is precisely the condition under
        which a stop cannot fire.
        """
        return not self.breaker.open_hosts

    # -- ticks ------------------------------------------------------------

    def fast_tick(self, *, snap: MarketSnapshot | None = None) -> TickResult:
        """Re-mark and enforce stops. No strategy call and no candles: this tick
        reads prices only, and pulling OHLCV it never looks at was spending a
        fifth of GeckoTerminal's keyless rate limit every minute — which the
        slow tick then went without."""
        snap = snap if snap is not None else self.snapshot(with_candles=False)
        now = self._now()
        book = self.book(snap, now=now)
        state = self._evaluate_risk(book, fills=(), now=now)

        fills, exits = self._enforce_stops(snap, book, state, now=now)
        if fills:
            book = self.book(snap, now=now)
            state = self._evaluate_risk(book, fills=fills, now=now)

        self.previous = snap
        return TickResult(
            ts=now,
            kind="fast",
            mode=self.broker.mode,
            portfolio=book,
            risk_state=state,
            fills=fills,
            stop_exits=exits,
            notes=self._startup_notes,
        )

    def slow_tick(self) -> TickResult:
        """evidence -> strategy -> targets -> risk bounds -> quote -> execute."""
        snap = self.snapshot()
        now = self._now()
        book = self.book(snap, now=now)
        state = self._evaluate_risk(book, fills=(), now=now)

        # Stops first. A position past its stop should not survive long enough
        # for a strategy to have an opinion about it.
        fills, exits = self._enforce_stops(snap, book, state, now=now)
        if fills:
            book = self.book(snap, now=now)
            state = self._evaluate_risk(book, fills=fills, now=now)

        evidence = self.evidence(snap)

        if state.halted:
            # Halted means exits only, and the stops above already ran. Calling
            # the strategy anyway would burn a model call to produce targets
            # that are guaranteed to be vetoed, and would log a decision that
            # never had a chance of executing.
            log.warning("risk halted: %s", "; ".join(state.halt_reasons) or "unspecified")
            self.previous = snap
            return TickResult(
                ts=now,
                kind="slow",
                mode=self.broker.mode,
                portfolio=book,
                risk_state=state,
                evidence=evidence,
                fills=fills,
                stop_exits=exits,
                error=None,
                notes=(*self._startup_notes, "risk halted; exits only"),
            )

        try:
            decision = self.strategy.decide(evidence, book, now=now)
        except Exception as exc:
            # A strategy failure is not a decision to hold. Skip the tick and
            # leave the distinction in the record.
            log.exception("strategy failed, skipping tick")
            self.previous = snap
            return TickResult(
                ts=now,
                kind="slow",
                mode=self.broker.mode,
                portfolio=book,
                risk_state=state,
                evidence=evidence,
                fills=fills,
                stop_exits=exits,
                error=str(exc),
                notes=self._startup_notes,
            )

        usage = self._brain.last_usage if self._brain is not None else None

        bounds, intents, new_fills, book, state = self._rebalance(
            decision, snap, book, state, evidence, now=now
        )
        fills = fills + new_fills

        record = DecisionRecord(
            decision_id=decision.decision_id,
            run_id=self.run_id,
            ts=now,
            strategy_id=decision.strategy_id,
            market_read=decision.market_read,
            targets=decision.targets,
            bounds=bounds,
            intents=intents,
            fills=fills,
            mode=self.broker.mode,
            forecasts=decision.forecasts,
            risk_state=state,
            model=self.cfg.model.name if usage is not None else "",
            effort=self.cfg.model.effort if usage is not None else "",
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0),
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0),
            thinking=getattr(usage, "thinking", None),
            advisory_used=self._brain is not None,
        )
        if self.broker.mode.may_mutate:
            self.ledger.append_decision(record)

        self.previous = snap
        # Advanced only on a tick that reached the strategy. A tick that died
        # earlier is not a decision, so the next successful one measures
        # liquidity across the whole outage and prints that longer window,
        # rather than claiming 15 minutes over evidence nobody read.
        self.decision_baseline = snap
        return TickResult(
            ts=now,
            kind="slow",
            mode=self.broker.mode,
            portfolio=book,
            risk_state=state,
            evidence=evidence,
            decision=decision,
            bounds=bounds,
            intents=intents,
            fills=fills,
            stop_exits=exits,
            usage=usage,
            notes=self._startup_notes,
        )

    # -- risk -------------------------------------------------------------

    def _evaluate_risk(
        self, book: PortfolioState, *, fills: tuple[Fill, ...], now: float
    ) -> RiskState:
        self.risk_ledger = risk.update_ledger(
            self.risk_ledger,
            book=book,
            fills=fills,
            params=self.cfg.risk,
            now=now,
        )
        return self.risk.continuous.evaluate(
            book=book,
            ledger=self.risk_ledger,
            data_quality_ok=self._data_health_ok(),
            now=now,
        )

    # -- stops ------------------------------------------------------------

    def _enforce_stops(
        self, snap: MarketSnapshot, book: PortfolioState, state: RiskState, *, now: float
    ) -> tuple[tuple[Fill, ...], tuple[str, ...]]:
        """Forced exits. Deliberately independent of the strategy — getting out
        is not a decision anything gets to veto.

        ``portfolio.stop_loss_breaches`` is the one definition of the stop
        price; this function does not recompute it. There used to be three
        definitions that disagreed at the third decimal, which meant a position
        could be past its stop in the fast tick and not in the slow one.
        """
        breaches = portfolio.stop_loss_breaches(
            book, self.cfg.risk.stop_loss_pct, params=self.cfg.mark
        )
        fills: list[Fill] = []
        exits: list[str] = []
        for breach in breaches:
            if breach.degraded_reference:
                # The stop is firing against a haircut mid rather than a route
                # quote. That is allowed (freezing stops during an outage is the
                # more dangerous failure on an asset that gaps) but it is not
                # the same event and must not read as one.
                log.warning(
                    "%s stop firing against a %s reference: %s",
                    breach.symbol,
                    breach.reference_basis,
                    breach.reason,
                )
            fill = self._force_exit(breach.symbol, snap, book, state, now=now)
            if fill is not None:
                fills.append(fill)
                if not fill.failed:
                    exits.append(breach.symbol)
                    self.risk_ledger = risk.update_ledger(
                        self.risk_ledger,
                        book=book,
                        risk_events=(f"stop:{breach.symbol}",),
                        params=self.cfg.risk,
                        now=now,
                    )
        return tuple(fills), tuple(exits)

    def _force_exit(
        self,
        symbol: str,
        snap: MarketSnapshot,
        book: PortfolioState,
        state: RiskState,
        *,
        now: float,
    ) -> Fill | None:
        position = self.broker.get_positions().get(symbol)
        if position is None or position.quantity_atomic <= 0:
            return None
        bounds = self.risk.exit_bounds(
            symbol,
            book=book,
            risk_state=state,
            snapshot=snap.coins.get(symbol),
            forced=True,
            now=now,
        )
        if not bounds.permitted:
            # A forced exit bypasses the sizing and pool-health rules, so a veto
            # here means something structural went wrong. Loudly, because the
            # consequence is an open position with a dead stop.
            log.error(
                "%s STOP BLOCKED by %s: %s", symbol, bounds.binding_rule, bounds.reason
            )
            return None
        for note in bounds.bypassed_rules:
            log.warning("%s stop bypassed %s", symbol, note)
        return self._execute(
            symbol=symbol,
            side=Side.SELL,
            bounds=bounds,
            token_amount_atomic=position.quantity_atomic,
            decision_id=None,
            source="stop_loss",
            reason=f"stop-loss: {symbol} past -{self.cfg.risk.stop_loss_pct:.0%} from entry",
            now=now,
        )

    # -- rebalancing ------------------------------------------------------

    def _rebalance(
        self,
        decision: StrategyDecision,
        snap: MarketSnapshot,
        book: PortfolioState,
        state: RiskState,
        evidence: Mapping[str, EvidenceBundle],
        *,
        now: float,
    ) -> tuple[
        tuple[RiskBounds, ...],
        tuple[OrderIntent, ...],
        tuple[Fill, ...],
        PortfolioState,
        RiskState,
    ]:
        """Turn target positions into orders.

        This is the diff the strategy layer exists to make possible. A target is
        a dollar amount of inventory to hold; the order is the difference
        between that and what is held. The old design had the model emit orders
        directly, which meant "BUY $50" twice in a row was two positions rather
        than one — the model had to remember the book, and it did not.

        Exits are processed before entries so that freed cash is available to
        the same tick, and so a de-risking decision cannot be blocked by a cash
        floor it was itself about to satisfy.
        """
        all_bounds: list[RiskBounds] = []
        all_intents: list[OrderIntent] = []
        all_fills: list[Fill] = []

        orders = self._diff_targets(decision, book)
        for side in (Side.SELL, Side.BUY):
            for symbol, delta_usd in orders:
                if (delta_usd < 0) != (side is Side.SELL):
                    continue
                bounds, intent, fill = self._trade_toward(
                    symbol,
                    side,
                    abs(delta_usd),
                    snap=snap,
                    book=book,
                    state=state,
                    evidence=evidence,
                    decision=decision,
                    now=now,
                )
                all_bounds.append(bounds)
                if intent is not None:
                    all_intents.append(intent)
                if fill is not None:
                    all_fills.append(fill)
                    book = self.book(snap, now=now)
                    state = self._evaluate_risk(book, fills=(fill,), now=now)
        return tuple(all_bounds), tuple(all_intents), tuple(all_fills), book, state

    def _diff_targets(
        self, decision: StrategyDecision, book: PortfolioState
    ) -> list[tuple[str, float]]:
        """Target minus held, filtered by the rebalance band.

        The band is not a nicety. Every round trip costs gas plus spread plus
        whatever the router's price impact is, and rebalancing a $50 position by
        $3 pays all of that to move nothing. A drift smaller than the band is
        left alone.
        """
        out: list[tuple[str, float]] = []
        for target in decision.targets:
            held = book.position_values_usd.get(target.symbol)
            if held is None and target.symbol in book.unmarkable:
                # We hold it and cannot price it. Sizing a delta against an
                # unknown is arithmetic on a guess; skip and say so.
                log.warning(
                    "%s is unmarkable, so no rebalance can be sized against it",
                    target.symbol,
                )
                continue
            delta = target.target_usd - (held or 0.0)
            if abs(delta) < self.cfg.strategy.rebalance_band_usd:
                continue
            if abs(delta) < self.cfg.strategy.min_trade_usd:
                continue
            out.append((target.symbol, delta))
        return out

    def _trade_toward(
        self,
        symbol: str,
        side: Side,
        notional_usd: float,
        *,
        snap: MarketSnapshot,
        book: PortfolioState,
        state: RiskState,
        evidence: Mapping[str, EvidenceBundle],
        decision: StrategyDecision,
        now: float,
    ) -> tuple[RiskBounds, OrderIntent | None, Fill | None]:
        coin = snap.coins.get(symbol)
        bundle = evidence.get(symbol)
        forecast = next((f for f in decision.forecasts if f.symbol == symbol), None)
        vol = None
        if bundle is not None and bundle.technicals is not None:
            vol = bundle.technicals.h1.realized_vol_pct if bundle.technicals.h1 else None

        if side is Side.SELL:
            bounds = self.risk.exit_bounds(
                symbol, book=book, risk_state=state, snapshot=coin, forced=False, now=now
            )
        else:
            # No quote passed here on purpose. The bound is computed *first*, at
            # a size risk is willing to see; only then is that size quoted. C3
            # was the other order — quote, then clamp, then fill the clamped
            # size against the unclamped quote.
            bounds = self.risk.entry_bounds(
                symbol,
                book=book,
                risk_state=state,
                snapshot=coin,
                forecast=forecast,
                volatility_pct=vol,
                ledger=self.risk_ledger,
                now=now,
            )

        if not bounds.permitted:
            log.info("REJECT %s %s — %s", side.value, symbol, bounds.reason)
            return bounds, None, None

        size_usd = min(notional_usd, bounds.max_notional_usd)
        if size_usd < self.cfg.risk.min_trade_usd:
            return (
                _veto(
                    symbol,
                    side,
                    "below_min_trade",
                    f"${size_usd:,.2f} is below the ${self.cfg.risk.min_trade_usd:,.2f} minimum",
                ),
                None,
                None,
            )

        token_amount_atomic: int | None = None
        if side is Side.SELL:
            position = self.broker.get_positions().get(symbol)
            if position is None or position.quantity_atomic <= 0:
                return _veto(symbol, side, "no_position", f"no {symbol} held"), None, None
            held_value = book.position_values_usd.get(symbol)
            if held_value is None or held_value <= 0:
                # Cannot price the position, so cannot convert dollars to
                # tokens. Selling "about that much" of an unpriced holding is
                # how an oversell becomes a clamp becomes a silent partial exit.
                return (
                    _veto(
                        symbol,
                        side,
                        "unmarkable",
                        f"{symbol} cannot be priced to size an exit",
                    ),
                    None,
                    None,
                )
            fraction = min(1.0, size_usd / held_value)
            token_amount_atomic = int(position.quantity_atomic * fraction)
            if token_amount_atomic <= 0:
                return (
                    _veto(symbol, side, "dust", "rounds to zero atomic units"),
                    None,
                    None,
                )

        fill = self._execute(
            symbol=symbol,
            side=side,
            bounds=bounds,
            notional_usd=size_usd if side is Side.BUY else None,
            token_amount_atomic=token_amount_atomic,
            decision_id=decision.decision_id,
            source="strategy",
            reason=_rationale(decision, symbol),
            now=now,
        )
        return bounds, self._last_intent, fill

    # -- execution --------------------------------------------------------

    def _execute(
        self,
        *,
        symbol: str,
        side: Side,
        bounds: RiskBounds,
        now: float,
        decision_id: str | None,
        source: OrderSource,
        reason: str,
        notional_usd: float | None = None,
        token_amount_atomic: int | None = None,
    ) -> Fill | None:
        """Quote at the permitted size, re-confirm, journal, then place.

        The ordering is the whole point and is worth stating plainly:

        1. **Quote the size risk permitted**, not the size that was wanted. A
           quote describes one specific swap; it is not a price curve you may
           evaluate at another point.
        2. **``confirm_quote``** re-runs the quote-dependent rules — price
           impact, depth participation, quote age — against the quote that will
           actually be sent. A quote that was fine at $25 may be 6% impact at
           the same $25 thirty seconds later.
        3. **Journal the intent before the side effect.** The intent ID is the
           idempotency key; if the process dies after this line and before the
           fill, :meth:`preflight` finds an open intent and refuses to trade
           rather than placing it a second time.
        4. **Place, then journal the outcome.** The broker writes its own fill
           row inside ``place_order`` before it persists state, so a fill can
           never exist in the book without a row explaining it.

        ``None`` is returned for every refusal. It is never an exception,
        because a refused order is an ordinary outcome of a tick and the loop
        must keep running.
        """
        self._last_intent = None
        coin_cfg = next((c for c in self.cfg.coins if c.symbol == symbol), None)
        if coin_cfg is None:
            log.error("%s is not a configured coin", symbol)
            return None
        token = self.token(symbol, coin_cfg.mint)
        if token is None:
            return None

        if side is Side.BUY:
            assert notional_usd is not None
            quote = quotes.quote_buy_usd(
                self.cfg,
                symbol=symbol,
                token=token,
                usd_notional=notional_usd,
                ttl_seconds=self.cfg.risk.max_quote_age_seconds,
                now=now,
                client=self.client,
            )
        else:
            assert token_amount_atomic is not None
            quote = quotes.quote_sell_tokens(
                self.cfg,
                symbol=symbol,
                token=token,
                token_amount_atomic=token_amount_atomic,
                ttl_seconds=self.cfg.risk.max_quote_age_seconds,
                now=now,
                client=self.client,
            )
        if quote is None:
            # A missing quote is a missing price, and the audit's C4 was that a
            # synthetic "degraded" quote was manufactured here and then
            # executed as though a router had offered it. There is no such
            # thing as a fallback quote.
            log.warning(
                "%s %s: no executable quote available; not trading", side.value, symbol
            )
            return None

        confirmed = self.risk.confirm_quote(
            bounds,
            quote,
            notional_usd=quote.usd_notional,
            now=now,
        )
        if not confirmed.permitted:
            log.info("REJECT %s %s at quote — %s", side.value, symbol, confirmed.reason)
            return None

        intent = OrderIntent(
            intent_id=new_intent_id(),
            decision_id=decision_id,
            action_id=new_action_id(),
            run_id=self.run_id,
            ts=now,
            symbol=symbol,
            side=side,
            # Exactly what was quoted. The broker refuses any other amount, and
            # that refusal is the enforcement of C3 rather than a sanity check.
            in_amount_atomic=quote.in_amount_atomic,
            max_in_amount_atomic=quote.in_amount_atomic,
            source=source,
            reason=reason,
        )
        self._last_intent = intent

        if not self.broker.mode.may_mutate:
            log.info(
                "[%s] would %s %s (%s) — not executed",
                self.broker.mode.value,
                side.value,
                symbol,
                intent.intent_id,
            )
            return None

        self.ledger.append_intent(intent, state=OrderState.QUOTE_BOUND)
        try:
            fill = self.broker.place_order(intent, quote, now=now)
        except BrokerError as exc:
            log.error("%s %s execution refused: %s", side.value, symbol, exc)
            self.ledger.append_state(
                intent_id=intent.intent_id,
                state=OrderState.FAILED,
                ts=self._now(),
                decision_id=decision_id,
                note=str(exc),
            )
            self.risk_ledger = risk.update_ledger(
                self.risk_ledger,
                book=self._blank_book(now),
                risk_events=(f"execution_failed:{symbol}",),
                params=self.cfg.risk,
                now=now,
            )
            return None

        self.ledger.append_fill(fill)
        self.ledger.append_state(
            intent_id=intent.intent_id,
            state=OrderState.RECONCILED,
            ts=fill.ts,
            order_id=fill.order_id,
            decision_id=decision_id,
        )
        log.info(
            "%s %s %s at %.10g USD%s",
            side.value,
            symbol,
            f"${fill.notional_usd:,.2f}",
            fill.price_usd,
            " (TX FAILED — gas still paid)" if fill.failed else "",
        )
        return fill

    def _blank_book(self, now: float) -> PortfolioState:
        """A book snapshot for the failure path, where re-marking would mean
        another round of network calls in the middle of an error."""
        return portfolio.mark_book(
            cash_usd=self.broker.cash_usd,
            positions=self.broker.get_positions(),
            marks={},
            realized_pnl_usd=self.broker.realized_pnl_usd,
            starting_cash_usd=self.broker.starting_cash_usd,
            fees_paid_usd=self.broker.fees_paid_usd,
            gas_paid_usd=self.broker.gas_paid_usd + self.broker.failed_gas_usd,
            now=now,
        )

    # -- the loop ---------------------------------------------------------

    def run(
        self,
        *,
        max_slow_ticks: int | None = None,
        on_tick: Callable[[TickResult], None] | None = None,
    ) -> None:
        installed = _install_signal_handlers(self._request_stop)
        fast = self.cfg.cadence.fast_tick_seconds
        slow = self.cfg.cadence.slow_tick_seconds
        # Monotonic, not wall clock. An NTP correction or a DST change moving
        # time.time() backwards used to stall the scheduler until wall time
        # caught up; moving it forwards fired every missed tick at once.
        next_slow = time.monotonic()
        slow_count = 0
        try:
            while not self._stop:
                mono = time.monotonic()
                if mono >= next_slow:
                    result = self.slow_tick()
                    # Fixed cadence from the deadline, not from now: scheduling
                    # the next tick relative to completion lets a run of slow
                    # ticks drift the decision interval out indefinitely.
                    next_slow = max(mono, next_slow) + slow
                    slow_count += 1
                else:
                    result = self.fast_tick()
                if on_tick is not None:
                    on_tick(result)
                if max_slow_ticks is not None and slow_count >= max_slow_ticks:
                    break
                deadline = min(time.monotonic() + fast, next_slow)
                # Sleep in slices so Ctrl+C is answered in under a second
                # rather than waiting out a full nap.
                while not self._stop and time.monotonic() < deadline:
                    time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
        finally:
            for sig, handler in installed:
                signal.signal(sig, handler)
            self.close()

    def _request_stop(self, *_: object) -> None:
        if self._stop:
            raise KeyboardInterrupt
        self._stop = True
        log.warning("stopping after this tick — state is saved; Ctrl+C again to force")

    # -- history ----------------------------------------------------------

    def recent_decisions(self) -> list[dict[str, Any]]:
        """The last N decisions as raw rows.

        Returned as dicts rather than rehydrated dataclasses. Rehydration used
        to be done here and was a recurring source of second-tick crashes,
        because a record written by an older schema version does not construct
        under the current one — and the consumer (a prompt renderer) reads a
        handful of scalars off each row. Parsing a whole object to read four
        fields is how a log-format change becomes a trading outage.
        """
        rows = journal.tail(self.cfg.ledger_path, self.cfg.prompt.decision_history * 8)
        return [r for r in rows if r.get("kind") == journal.RowKind.DECISION.value]


def _veto(symbol: str, side: Side, rule: str, reason: str) -> RiskBounds:
    return RiskBounds(
        symbol=symbol,
        side=side,
        max_notional_usd=0.0,
        vetoes=(rule,),
        reasons=(reason,),
        binding_rule=rule,
    )


def _rationale(decision: StrategyDecision, symbol: str) -> str:
    for target in decision.targets:
        if target.symbol == symbol:
            return target.rationale or decision.strategy_id
    return decision.strategy_id


def _install_signal_handlers(handler: Callable[..., None]) -> list[tuple[int, Any]]:
    installed: list[tuple[int, Any]] = []
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            installed.append((sig, signal.signal(sig, handler)))
        except ValueError, OSError:
            # Not on the main thread, or unsupported on this platform.
            continue
    return installed


# ---------------------------------------------------------------------------
# Risk ledger persistence
# ---------------------------------------------------------------------------
#
# The risk ledger holds the peak book value, the day and window anchors, the
# consecutive-failure count and the post-stop quarantines. All of it is
# *history*, and history that lives only in memory means a restart silently
# clears a drawdown halt and un-quarantines a symbol that just stopped out —
# which turns "halt the run" into "halt until someone restarts it", the exact
# opposite of a circuit breaker.


def _risk_ledger_path(cfg: Config) -> Path:
    return cfg.data_dir / "risk_ledger.json"


def _load_risk_ledger(cfg: Config) -> risk.RiskLedger:
    path = _risk_ledger_path(cfg)
    if not path.is_file():
        return risk.EMPTY_LEDGER
    try:
        import json

        raw = json.loads(path.read_text(encoding="utf-8"))
        return risk.RiskLedger(
            peak_value_usd=raw.get("peak_value_usd"),
            day_start_value_usd=raw.get("day_start_value_usd"),
            day_start_ts=raw.get("day_start_ts"),
            window_start_value_usd=raw.get("window_start_value_usd"),
            window_start_ts=raw.get("window_start_ts"),
            consecutive_failures=int(raw.get("consecutive_failures", 0)),
            quarantined_until={
                k: float(v) for k, v in raw.get("quarantined_until", {}).items()
            },
            last_entry_ts={k: float(v) for k, v in raw.get("last_entry_ts", {}).items()},
            manual_halt=bool(raw.get("manual_halt", False)),
            manual_halt_reason=raw.get("manual_halt_reason"),
        )
    except Exception as exc:
        # Loud, and NOT a silent reset to empty: an unreadable risk ledger means
        # the halt history is gone, and resuming as though nothing ever went
        # wrong is the failure this file is trying to prevent. Halt instead.
        log.error("risk ledger at %s is unreadable (%s); starting halted", path, exc)
        return risk.RiskLedger(
            manual_halt=True,
            manual_halt_reason=f"risk ledger unreadable: {exc}",
        )


def _save_risk_ledger(cfg: Config, ledger: risk.RiskLedger) -> None:
    try:
        import json

        journal.atomic_write_text(
            _risk_ledger_path(cfg),
            json.dumps(journal.to_jsonable(ledger), indent=2),
        )
    except Exception as exc:  # pragma: no cover - disk failure
        log.error("could not persist risk ledger: %s", exc)
