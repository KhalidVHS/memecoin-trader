"""The dual-cadence scheduler — the one piece of real design in this project.

A **fast tick** every 60 seconds refreshes prices, marks the book and enforces
stop-losses. No model call, so it is nearly free. This exists because a -15%
stop that only checks every 15 minutes is not a stop; by the time a memecoin
decides to move, fifteen minutes is several lifetimes.

A **slow tick** every 15 minutes builds the full evidence bundle, calls the
model once, runs every proposed action through ``risk.check()``, and executes
what survives.

State is written atomically after every mutation, so Ctrl+C or a crash resumes
with the book intact.
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field

import httpx

from . import brain, journal, market, portfolio, quotes, risk, sentiment, signals
from .broker import BrokerError, LocalPaperBroker
from .config import Config
from .http import make_client
from .types import (
    DecisionRecord,
    EvidenceBundle,
    Fill,
    MarketSnapshot,
    PortfolioState,
    RiskVerdict,
    Side,
    TradeProposal,
)

log = logging.getLogger("memetrader")


@dataclass
class TickResult:
    """What one tick did, for the CLI to render. A slow tick that never reached
    the model still returns a result — ``decision is None`` and ``error`` set —
    because a failed tick and a tick that decided to hold must look different."""

    ts: float
    kind: str  # "fast" | "slow"
    portfolio: PortfolioState
    evidence: dict[str, EvidenceBundle] = field(default_factory=dict)
    decision: object | None = None
    verdicts: list[RiskVerdict] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    stop_loss_exits: list[str] = field(default_factory=list)
    usage: brain.Usage | None = None
    error: str | None = None
    dry_run: bool = False


class Trader:
    """Owns the book, the HTTP client and the tick cadence."""

    def __init__(self, cfg: Config, *, client: httpx.Client | None = None) -> None:
        self.cfg = cfg
        self.broker = LocalPaperBroker(cfg)
        self._client = client
        self._owns_client = client is None
        # Kept so signals.py can compute liquidity_trend_pct: a pool draining is
        # the most important thing that can happen to a position, and you only
        # see it by comparing two reads.
        self.previous: MarketSnapshot | None = None
        # Rejections from the last slow tick, fed back to the model so it learns
        # the boundaries instead of re-proposing illegal trades.
        self.pending_rejections: list[RiskVerdict] = []
        self._stop = False

    # -- plumbing ---------------------------------------------------------

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            # Via http.make_client: verifies against the OS trust store, which is
            # the only store that has the corporate TLS-inspection CA in it.
            self._client = make_client(self.cfg.data.http_timeout_seconds)
        return self._client

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def __enter__(self) -> Trader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- evidence ---------------------------------------------------------

    def snapshot(self, *, with_candles: bool = True) -> MarketSnapshot:
        return market.snapshot(
            self.cfg,
            client=self.client,
            previous=self.previous,
            with_candles=with_candles,
        )

    def evidence(self, snap: MarketSnapshot) -> dict[str, EvidenceBundle]:
        """Assemble all three streams. Each degrades independently — a stream
        that fails is reported to the model as explicitly unavailable, so it can
        discount its confidence rather than silently trading on a blank."""
        briefs: dict[str, object] = {}
        reasons: dict[str, str | None] = {}
        if self.cfg.sentiment.enabled:
            try:
                briefs = sentiment.briefs(self.cfg)  # type: ignore[assignment]
            except Exception as exc:  # pragma: no cover - sentiment.briefs catches its own
                log.warning("sentiment unavailable: %s", exc)
                reasons = {s: f"sentiment lookup failed: {exc}" for s in self.cfg.symbols}
        else:
            reasons = dict.fromkeys(self.cfg.symbols, "sentiment disabled in config.toml")

        bundles: dict[str, EvidenceBundle] = {}
        for sym, coin in snap.coins.items():
            prev = self.previous.coins.get(sym) if self.previous else None
            try:
                tech = signals.brief(coin, prev)
            except Exception as exc:
                log.warning("%s technicals failed: %s", sym, exc)
                tech = None
            brief = briefs.get(sym)
            bundles[sym] = EvidenceBundle(
                symbol=sym,
                snapshot=coin,
                technicals=tech,
                sentiment=brief,  # type: ignore[arg-type]
                sentiment_unavailable_reason=(
                    reasons.get(sym)
                    or (None if brief is not None else "no sentiment data returned")
                ),
            )
        return bundles

    def marks(self, snap: MarketSnapshot) -> dict[str, float]:
        return {sym: c.price_usd for sym, c in snap.coins.items()}

    # -- ticks ------------------------------------------------------------

    def fast_tick(self, *, snap: MarketSnapshot | None = None) -> TickResult:
        """Mark the book and enforce stop-losses. No model call, and no
        candles: this tick reads prices and liquidity only, and pulling OHLCV
        it never looks at was spending a fifth of GeckoTerminal's keyless rate
        limit every minute — which the slow tick then went without."""
        snap = snap or self.snapshot(with_candles=False)
        now = time.time()
        state = portfolio.mark(self.broker, self.marks(snap), now=now)
        result = TickResult(ts=now, kind="fast", portfolio=state)

        breaches = portfolio.stop_loss_breaches(state, self.cfg.risk.stop_loss_pct)
        for sym in breaches:
            fill = self._force_exit(sym, snap, state, now)
            if fill is not None:
                result.fills.append(fill)
                result.stop_loss_exits.append(sym)

        if result.fills:
            # Re-mark so the caller sees the book after the exits, not before.
            result.portfolio = portfolio.mark(self.broker, self.marks(snap), now=now)
        self.previous = snap
        return result

    def _force_exit(
        self, symbol: str, snap: MarketSnapshot, state: PortfolioState, now: float
    ) -> Fill | None:
        """A stop-loss exit. Deliberately independent of the model — getting out
        is not a decision the model gets to veto."""
        value = state.position_values_usd.get(symbol, 0.0)
        if value <= 0:
            return None
        coin = snap.coins.get(symbol)
        if coin is not None:
            mint, mid = coin.mint, coin.price_usd
        else:
            # No snapshot for this coin, which is itself a bad sign — but the
            # mint is a config constant, and ``mid_price_usd`` is only a hint
            # used to derive token decimals when Jupiter's token API is down.
            # The entry price is a fine hint for that. Refusing to exit here
            # would make a data outage cost the position.
            position = self.broker.get_positions().get(symbol)
            if position is None:
                return None
            cfg_coin = next((c for c in self.cfg.coins if c.symbol == symbol), None)
            if cfg_coin is None:
                log.error("%s stop-loss fired but the coin is not in config", symbol)
                return None
            log.warning("%s stop-loss firing with no market snapshot to exit against", symbol)
            mint, mid = cfg_coin.mint, position.avg_entry_price_usd
        try:
            quote = quotes.fill_quote(
                self.cfg,
                symbol,
                mint,
                Side.SELL,
                value,
                mid_price_usd=mid,
                client=self.client,
            )
        except Exception as exc:
            log.error("%s stop-loss quote failed: %s", symbol, exc)
            return None

        proposal = TradeProposal(
            symbol=symbol,
            side=Side.SELL,
            usd_notional=value,
            quote=quote,
            source="stop_loss",
            reasoning=f"stop-loss: {symbol} breached -{self.cfg.risk.stop_loss_pct:.0%}",
        )
        verdict = risk.check(proposal, state, snap, self.cfg, now=now)
        if not verdict.approved:
            # A stop bypasses the sizing and pool-health rules, so this means
            # something real went wrong (a stale snapshot, or no quote). Loudly.
            log.error("%s stop-loss BLOCKED by %s: %s", symbol, verdict.rule, verdict.reason)
            return None
        for note in verdict.notes:
            # Every bypassed rule surfaces here. A stop that got out through a
            # collapsing pool should be visible in the log, not just in the P&L.
            log.warning("%s stop-loss: %s", symbol, note)
        try:
            fill = self.broker.place_order(
                symbol, Side.SELL, verdict.approved_usd, quote=quote, now=now
            )
        except BrokerError as exc:
            log.error("%s stop-loss execution failed: %s", symbol, exc)
            return None
        # No journal.append here: the broker writes the trades.jsonl row inside
        # place_order, before it persists state, so that a fill can never exist
        # in the book without a row explaining it. Logging it again here is how
        # every fill ended up in the ledger twice.
        log.warning(
            "STOP-LOSS %s sold $%.2f at %.10g%s",
            symbol,
            fill.filled_usd,
            fill.price_usd,
            " (TX FAILED)" if fill.failed else "",
        )
        return fill

    def slow_tick(self, *, dry_run: bool = False) -> TickResult:
        """The full decision cycle: evidence -> model -> risk -> execution."""
        snap = self.snapshot()
        now = time.time()
        state = portfolio.mark(self.broker, self.marks(snap), now=now)
        result = TickResult(ts=now, kind="slow", portfolio=state, dry_run=dry_run)

        # Stops first. A position past its stop should not survive long enough
        # for the model to have an opinion about it.
        for sym in portfolio.stop_loss_breaches(state, self.cfg.risk.stop_loss_pct):
            fill = self._force_exit(sym, snap, state, now)
            if fill is not None:
                result.fills.append(fill)
                result.stop_loss_exits.append(sym)
        if result.fills:
            state = portfolio.mark(self.broker, self.marks(snap), now=now)
            result.portfolio = state

        result.evidence = self.evidence(snap)
        history = self.recent_decisions()

        try:
            decision, usage = brain.decide(
                self.cfg,
                result.evidence,
                state,
                history,
                self.pending_rejections,
                client=None,
            )
        except Exception as exc:
            # An API failure is not a decision to hold. Skip the tick and leave
            # the distinction visible in the log.
            log.error("model call failed, skipping tick: %s", exc)
            result.error = str(exc)
            self.previous = snap
            return result

        result.decision = decision
        result.usage = usage

        verdicts: list[RiskVerdict] = []
        for action in decision.actions:
            verdict, fill = self._apply(action, snap, state, now, dry_run=dry_run)
            verdicts.append(verdict)
            if fill is not None:
                result.fills.append(fill)
                state = portfolio.mark(self.broker, self.marks(snap), now=now)
        result.verdicts = verdicts
        result.portfolio = state

        self.pending_rejections = [v for v in verdicts if not v.approved]

        record = DecisionRecord(
            ts=now,
            market_read=decision.market_read,
            actions=tuple(decision.actions),
            verdicts=tuple(verdicts),
            fills=tuple(result.fills),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            model=self.cfg.model.name,
            effort=self.cfg.model.effort,
            dry_run=dry_run,
        )
        if not dry_run:
            journal.append(self.cfg.decisions_path, record)
        self.previous = snap
        return result

    def _apply(
        self,
        action,
        snap: MarketSnapshot,
        state: PortfolioState,
        now: float,
        *,
        dry_run: bool,
    ) -> tuple[RiskVerdict, Fill | None]:
        symbol = action.symbol
        if action.action == "HOLD":
            return RiskVerdict(True, 0.0, symbol=symbol, reason="HOLD"), None

        coin = snap.coins.get(symbol)
        if coin is None:
            return (
                RiskVerdict(
                    False, 0.0, symbol=symbol, rule="unknown_symbol",
                    reason=f"{symbol} is not a configured coin",
                ),
                None,
            )

        side = Side(action.action)
        try:
            quote = quotes.fill_quote(
                self.cfg, symbol, coin.mint, side, action.size_usd,
                mid_price_usd=coin.price_usd, client=self.client,
            )
        except Exception as exc:
            log.warning("%s quote failed: %s", symbol, exc)
            quote = None

        proposal = TradeProposal(
            symbol=symbol,
            side=side,
            usd_notional=action.size_usd,
            quote=quote,
            confidence=action.confidence,
            reasoning=action.reasoning,
        )
        verdict = risk.check(proposal, state, snap, self.cfg, now=now)
        if not verdict.approved:
            log.info("REJECT %s %s $%.2f — %s", side, symbol, action.size_usd, verdict.reason)
            return verdict, None
        if dry_run or quote is None:
            return verdict, None

        try:
            fill = self.broker.place_order(
                symbol, side, verdict.approved_usd, quote=quote, now=now
            )
        except BrokerError as exc:
            return (
                RiskVerdict(
                    False, 0.0, symbol=symbol, rule="execution_failed", reason=str(exc)
                ),
                None,
            )
        # The broker already logged this fill — see _force_exit.
        log.info(
            "%s %s $%.2f at %.10g%s",
            side, symbol, fill.filled_usd, fill.price_usd,
            " (TX FAILED — gas still paid)" if fill.failed else "",
        )
        return verdict, fill

    # -- history ----------------------------------------------------------

    def recent_decisions(self) -> list[DecisionRecord]:
        """The last N decisions, bounded by construction so context never grows
        without limit across a multi-day run."""
        rows = journal.tail(self.cfg.decisions_path, self.cfg.prompt.decision_history)
        out: list[DecisionRecord] = []
        for row in rows:
            try:
                out.append(_record_from_row(row))
            except Exception as exc:
                # Skipping a row keeps an old or malformed log from bricking the
                # run, but it must not be silent: the failure mode is the model
                # quietly losing its memory of what it just did, which looks
                # like erratic trading rather than like a bug.
                log.warning("skipping unreadable decision-log row: %s", exc)
        return out

    # -- the loop ---------------------------------------------------------

    def run(self, *, max_slow_ticks: int | None = None, on_tick=None) -> None:
        installed = _install_signal_handlers(self._request_stop)
        fast = self.cfg.cadence.fast_tick_seconds
        slow = self.cfg.cadence.slow_tick_seconds
        next_slow = time.time()  # decide immediately on start
        slow_count = 0
        try:
            while not self._stop:
                now = time.time()
                if now >= next_slow:
                    result = self.slow_tick()
                    next_slow = now + slow
                    slow_count += 1
                else:
                    result = self.fast_tick()
                if on_tick is not None:
                    on_tick(result)
                if max_slow_ticks is not None and slow_count >= max_slow_ticks:
                    break
                # Sleep in short slices so Ctrl+C is responsive rather than
                # waiting out a full 60-second nap.
                deadline = min(time.time() + fast, next_slow)
                while not self._stop and time.time() < deadline:
                    time.sleep(min(0.5, max(0.0, deadline - time.time())))
        finally:
            for sig, handler in installed:
                signal.signal(sig, handler)
            self.close()

    def _request_stop(self, *_: object) -> None:
        if self._stop:
            raise KeyboardInterrupt
        self._stop = True
        log.warning("stopping after this tick — state is already saved; Ctrl+C again to force")


def _install_signal_handlers(handler) -> list[tuple[int, object]]:
    installed: list[tuple[int, object]] = []
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            installed.append((sig, signal.signal(sig, handler)))
        except (ValueError, OSError):
            # Not on the main thread, or unsupported on this platform.
            continue
    return installed


def _record_from_row(row: dict) -> DecisionRecord:
    """Rehydrate a logged decision into the same shape ``slow_tick`` builds.

    Every nested member is reconstructed, not left as the raw dict json.load
    produced. An earlier version kept ``fills`` raw on the theory that the
    history renderer only read scalars off them; it reads ``fill.symbol``, so
    the second slow tick of every run died with "'dict' object has no attribute
    'symbol'" — and only ever the second, because the first runs against an
    empty log. If it is typed as a ``Fill`` here, it is a ``Fill``.
    """
    from .types import Action, Fill

    return DecisionRecord(
        ts=float(row.get("ts", 0.0)),
        market_read=str(row.get("market_read", "")),
        actions=tuple(Action(**a) for a in row.get("actions", [])),
        verdicts=tuple(
            RiskVerdict(**{**v, "notes": tuple(v.get("notes", ()))})
            for v in row.get("verdicts", [])
        ),
        fills=tuple(
            Fill(**{**f, "side": Side(f["side"])}) for f in row.get("fills", [])
        ),
        input_tokens=int(row.get("input_tokens", 0)),
        output_tokens=int(row.get("output_tokens", 0)),
        cache_read_input_tokens=int(row.get("cache_read_input_tokens", 0)),
        cache_creation_input_tokens=int(row.get("cache_creation_input_tokens", 0)),
        model=str(row.get("model", "")),
        effort=str(row.get("effort", "")),
        dry_run=bool(row.get("dry_run", False)),
    )
