"""Baseline strategies — the reference every candidate model must beat.

Every strategy here reads history **exclusively** through
``histdata.point_in_time.PointInTimeState`` (bars and universe membership) and
its own ``PortfolioState`` (cash and current positions). None of them touch a
raw file, a full series, or the wall clock: randomness is seeded explicitly
(see ``RandomEntryStrategy``) and every other decision is a pure function of
``(state, portfolio, now)``.

A strategy here **proposes**; it never sizes a final order and never
executes one. Every ``propose`` method returns ``tuple[OrderIntent, ...]`` —
never a ``Fill``, never an ``execution.interfaces.ApprovedOrder``. Sizing
against available cash, exposure caps and turnover control are
``strategies/construction.py``'s job, not this module's — see
:func:`memetrader.strategies.construction.size_orders`.

``OrderIntent.in_amount_atomic`` is always denominated in the *input* leg of
the proposed swap. That happens to make sizing trivial here without any token
decimals: a BUY's input is cash, so ``in_amount_atomic`` is the proposed spend
in **micro-USD** (matching ``broker.py``'s ``cash_micro_usd`` convention — see
``MICRO_USD`` in ``construction.py``); a SELL's input is the token itself, so
this module never invents a sell quantity — it always sells the position's
full ``quantity_atomic``, copied verbatim from the ``Position`` record. Partial
trims are deliberately out of scope for a baseline: they are a sizing decision,
which belongs to ``construction.py``.

Universe-shrinkage rule (documented, load-bearing — a delisted or "dead" token
is a frequent, real memecoin event, not an edge case)
------------------------------------------------------------------------------
* :class:`BuyAndHoldStrategy` is exempt by definition: it never rebalances, so
  it holds a position through delisting. That is exactly the risk buy-and-hold
  is meant to measure, not a bug to route around.
* :class:`_RebalancingStrategy` (equal-weight, momentum, mean-reversion) treats
  "no longer in ``state.universe()``" as an unconditional full-exit signal: on
  the next ``propose`` call after a symbol drops out, it emits a SELL for the
  entire held position, unconditionally, independent of whether the momentum
  or mean-reversion score could still be computed from stale bars. A token
  that cannot be rebalanced cannot be held under an active allocation rule.
* :class:`RandomEntryStrategy` schedules exits purely by elapsed holding time,
  not by universe membership — closing every held trade at its pre-committed
  exit time regardless of what else happened to the token in between, because
  the baseline exists to measure timing skill in isolation. If a scheduled
  *entry* tick arrives and ``state.universe()`` is empty, that entry is
  skipped and not retried: a real random trader attempting to enter a dead
  market takes no position, not a deferred one.

Determinism
-----------
No strategy in this module reads ``time.time()`` or any other wall clock —
that would raise under ``backtest.clock.SimulatedClock`` and would break
byte-identical replay even if it did not. Order identifiers are minted from a
SHA-256 of ``(run_id, strategy_id, symbol, side, ts)``, not from
``ids.new_intent_id()`` (which mixes in wall-clock microseconds and
``os.urandom`` — appropriate for the live system's crash-recovery idempotency
key, wrong for a replay that must produce the same bytes twice).
``RandomEntryStrategy`` seeds its own ``random.Random`` from an integer the
caller supplies; the documented, required source of that integer is
``backtest.clock.SimulatedClock.deterministic_seed("random_baseline")``.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from memetrader.features.market import return_pct
from memetrader.types import (
    FidelityTier,
    OrderIntent,
    PortfolioState,
    Position,
    Side,
    Timeframe,
)

if TYPE_CHECKING:
    from memetrader.histdata.point_in_time import PointInTimeState

_INTERVAL_SECONDS: dict[Timeframe, float] = {
    Timeframe.M5: 300.0,
    Timeframe.H1: 3600.0,
}


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class BacktestStrategy(Protocol):
    """What every baseline (and every candidate strategy compared against them)
    must implement.

    Deliberately narrower than ``strategy.py``'s live ``Strategy`` protocol:
    that one is keyed to ``EvidenceBundle``/venue-shaped inputs. This one is
    keyed to the replay's own point-in-time state and reads no live-only
    fields.
    """

    strategy_id: str
    required_fidelity: FidelityTier

    def propose(
        self,
        *,
        state: PointInTimeState,
        portfolio: PortfolioState,
        now: float,
        run_id: str,
    ) -> tuple[OrderIntent, ...]: ...


# ---------------------------------------------------------------------------
# Deterministic order construction helpers
# ---------------------------------------------------------------------------


def _intent_id(
    *, run_id: str, strategy_id: str, symbol: str, side: Side, now: float
) -> str:
    """A reproducible idempotency key: same inputs, same id, every time.

    Unlike ``ids.new_intent_id()``, this never reads the wall clock and never
    draws from ``os.urandom`` — both would break byte-identical replay, and
    the wall-clock read would raise outright under ``SimulatedClock``.
    """
    payload = f"{run_id}:{strategy_id}:{symbol}:{side.value}:{now!r}"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:24]
    return f"intent-{digest}"


def _buy_intent(
    *, strategy_id: str, symbol: str, notional_micro_usd: int, now: float, run_id: str
) -> OrderIntent | None:
    """A proposed BUY sized in micro-USD (the cash/input leg). ``None`` if the
    proposed spend rounds to nothing worth proposing."""
    amt = max(0, int(notional_micro_usd))
    if amt <= 0:
        return None
    intent_id = _intent_id(
        run_id=run_id, strategy_id=strategy_id, symbol=symbol, side=Side.BUY, now=now
    )
    return OrderIntent(
        intent_id=intent_id,
        decision_id=None,
        action_id=None,
        run_id=run_id,
        ts=now,
        symbol=symbol,
        side=Side.BUY,
        in_amount_atomic=amt,
        max_in_amount_atomic=amt,
        source="strategy",
        reason=f"{strategy_id}: entry",
    )


def _sell_intent(
    *, strategy_id: str, symbol: str, position: Position, now: float, run_id: str
) -> OrderIntent | None:
    """A proposed full-exit SELL. Sizes directly off ``Position.quantity_atomic``
    — never derived, so there is no decimals conversion to get wrong."""
    amt = position.quantity_atomic
    if amt <= 0:
        return None
    intent_id = _intent_id(
        run_id=run_id, strategy_id=strategy_id, symbol=symbol, side=Side.SELL, now=now
    )
    return OrderIntent(
        intent_id=intent_id,
        decision_id=None,
        action_id=None,
        run_id=run_id,
        ts=now,
        symbol=symbol,
        side=Side.SELL,
        in_amount_atomic=amt,
        max_in_amount_atomic=amt,
        source="strategy",
        reason=f"{strategy_id}: exit",
    )


def _usd_to_micro(usd: float) -> int:
    """Floor-convert dollars to integer micro-USD. The one float boundary a
    strategy is allowed to cross: ``PortfolioState.cash_usd`` is a float by
    contract (``types.PortfolioState``), and this is where it becomes an int
    the same way ``backtest/ledger.py`` converts ``Fill.notional_usd`` once at
    its own boundary. Floors, never rounds up, so a strategy can never
    propose spending a fraction of a cent more than it can see."""
    if usd <= 0.0:
        return 0
    return int(usd * 1_000_000)


# ---------------------------------------------------------------------------
# Buy and hold
# ---------------------------------------------------------------------------


@dataclass
class BuyAndHoldStrategy:
    """Buy each token once, the first tick it becomes point-in-time eligible,
    and never sell it.

    Cash still un-invested is split evenly across the symbols that are
    eligible-but-not-yet-bought at the moment each one first appears, so a
    universe that grows over time (new listings) is handled without any
    foresight: a symbol earns a share of whatever cash remains when *it*
    shows up, never of cash that has not been raised yet.
    """

    strategy_id: str = "buy_and_hold"
    required_fidelity: FidelityTier = FidelityTier.TIER_0
    min_notional_micro_usd: int = 1_000_000  # do not bother with sub-$1 allocations
    entry_reservation_seconds: float = 3_600.0
    """How long a proposed-but-not-yet-settled entry suppresses a re-proposal.

    Proposing is not buying. An intent can still be vetoed by risk, or fail to
    route, and the ledger moves no cash until a fill is applied — so between
    proposal and settlement the symbol is in neither ``portfolio.positions``
    nor anywhere else this strategy can see. Without a reservation, every tick
    in that window would propose the same full-size buy again and they would
    stack.

    The reservation therefore has to expire rather than be permanent: a
    reservation that never lifts turns one vetoed entry into a symbol this
    strategy will never buy, and a buy-and-hold benchmark that quietly holds
    nothing makes every strategy measured against it look good. One hour is
    comfortably longer than settlement takes at the bar cadences this runs on,
    and short enough that a transient veto costs one retry rather than the
    whole run.
    """
    _reserved_at: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def propose(
        self, *, state: PointInTimeState, portfolio: PortfolioState, now: float, run_id: str
    ) -> tuple[OrderIntent, ...]:
        universe = sorted(state.universe())
        candidates = [
            s
            for s in universe
            if s not in portfolio.positions and not self._is_reserved(s, now=now)
        ]
        if not candidates:
            return ()

        share_micro = _usd_to_micro(portfolio.cash_usd) // len(candidates)
        if share_micro < self.min_notional_micro_usd:
            return ()

        intents: list[OrderIntent] = []
        for symbol in candidates:
            intent = _buy_intent(
                strategy_id=self.strategy_id,
                symbol=symbol,
                notional_micro_usd=share_micro,
                now=now,
                run_id=run_id,
            )
            if intent is not None:
                self._reserved_at[symbol] = now
                intents.append(intent)
        return tuple(intents)

    def _is_reserved(self, symbol: str, *, now: float) -> bool:
        """True while an earlier proposal for ``symbol`` is still in flight."""
        reserved_at = self._reserved_at.get(symbol)
        if reserved_at is None:
            return False
        return now - reserved_at < self.entry_reservation_seconds


# ---------------------------------------------------------------------------
# Equal-weight / momentum / mean-reversion — shared rebalancing machinery
# ---------------------------------------------------------------------------


@dataclass
class _RebalancingStrategy:
    """Shared machinery for equal-weight, momentum and mean-reversion.

    ``top_k=None`` means "target the whole universe" (equal weight). Otherwise
    the ``top_k`` symbols with the highest (``contrarian=False``, momentum) or
    lowest (``contrarian=True``, mean-reversion) trailing return over
    ``lookback_seconds`` are targeted, equally weighted among themselves.

    Rebalancing here is binary per symbol — enter to an equal share or exit
    entirely — never a partial trim of an existing holding. Partial-position
    sizing against caps and turnover is ``construction.py``'s job; mixing it
    into the strategy layer would make the "a strategy proposes, it never
    decides final size" rule impossible to test.
    """

    strategy_id: str
    timeframe: Timeframe
    lookback_seconds: float
    top_k: int | None
    contrarian: bool = False
    min_notional_micro_usd: int = 1_000_000
    required_fidelity: FidelityTier = FidelityTier.TIER_0

    def _select(self, state: PointInTimeState) -> tuple[str, ...]:
        universe = sorted(state.universe())
        if self.top_k is None:
            return tuple(universe)

        interval = _INTERVAL_SECONDS[self.timeframe]
        lookback_bars = int(self.lookback_seconds / interval) + 5
        scored: list[tuple[float, str]] = []
        for symbol in universe:
            candles = state.bars(symbol, self.timeframe, lookback=lookback_bars)
            if len(candles) < 2:
                continue
            close = np.array([c.close for c in candles], dtype=float)
            ts = np.array([c.ts for c in candles], dtype=float)
            r = return_pct(
                close, ts, target_seconds=self.lookback_seconds, interval_seconds=interval
            )
            if r is None:
                continue
            scored.append((r, symbol))

        if not scored:
            return ()
        # Momentum: highest return first. Mean-reversion: lowest return first.
        # The symbol name is a deterministic tie-breaker so the ranking never
        # depends on dict/set iteration order.
        scored.sort(key=lambda pair: (pair[0] if self.contrarian else -pair[0], pair[1]))
        return tuple(sym for _, sym in scored[: self.top_k])

    def propose(
        self, *, state: PointInTimeState, portfolio: PortfolioState, now: float, run_id: str
    ) -> tuple[OrderIntent, ...]:
        target = set(self._select(state))
        universe = state.universe()
        held = {s for s, p in portfolio.positions.items() if p.quantity_atomic > 0}

        intents: list[OrderIntent] = []

        # Exits first: no longer targeted, or (redundantly, for clarity) no
        # longer in the universe at all. See module docstring, universe rule.
        for symbol in sorted(held):
            if symbol in target and symbol in universe:
                continue
            exit_intent = _sell_intent(
                strategy_id=self.strategy_id,
                symbol=symbol,
                position=portfolio.positions[symbol],
                now=now,
                run_id=run_id,
            )
            if exit_intent is not None:
                intents.append(exit_intent)

        # Entries: targeted symbols not already held.
        to_enter = sorted(s for s in target if s not in held)
        if to_enter:
            share_micro = _usd_to_micro(portfolio.cash_usd) // len(to_enter)
            if share_micro >= self.min_notional_micro_usd:
                for symbol in to_enter:
                    entry_intent = _buy_intent(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        notional_micro_usd=share_micro,
                        now=now,
                        run_id=run_id,
                    )
                    if entry_intent is not None:
                        intents.append(entry_intent)

        return tuple(intents)


def equal_weight_strategy(
    *, timeframe: Timeframe = Timeframe.H1, min_notional_micro_usd: int = 1_000_000
) -> _RebalancingStrategy:
    """Equal-weight rebalanced across the entire point-in-time universe."""
    return _RebalancingStrategy(
        strategy_id="equal_weight",
        timeframe=timeframe,
        lookback_seconds=0.0,  # unused: top_k=None never scores a return
        top_k=None,
        min_notional_micro_usd=min_notional_micro_usd,
    )


def momentum_strategy(
    *,
    timeframe: Timeframe = Timeframe.H1,
    lookback_seconds: float,
    top_k: int = 3,
    min_notional_micro_usd: int = 1_000_000,
) -> _RebalancingStrategy:
    """Long the ``top_k`` symbols with the highest trailing return."""
    return _RebalancingStrategy(
        strategy_id="momentum_top_k",
        timeframe=timeframe,
        lookback_seconds=lookback_seconds,
        top_k=top_k,
        contrarian=False,
        min_notional_micro_usd=min_notional_micro_usd,
    )


def mean_reversion_strategy(
    *,
    timeframe: Timeframe = Timeframe.H1,
    lookback_seconds: float,
    top_k: int = 3,
    min_notional_micro_usd: int = 1_000_000,
) -> _RebalancingStrategy:
    """Long the ``top_k`` symbols with the lowest (most negative) trailing
    return — the contrarian counterpart to :func:`momentum_strategy`."""
    return _RebalancingStrategy(
        strategy_id="mean_reversion_bottom_k",
        timeframe=timeframe,
        lookback_seconds=lookback_seconds,
        top_k=top_k,
        contrarian=True,
        min_notional_micro_usd=min_notional_micro_usd,
    )


# ---------------------------------------------------------------------------
# Random entry — the baseline that matters most
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TradeStat:
    """One round-trip actually taken by some other strategy, used only to
    build a matched random schedule. Not an ``OrderIntent`` — this never
    reaches a strategy's ``propose`` output, it is purely a matching input."""

    entry_time: float
    exit_time: float
    notional_micro_usd: int

    def __post_init__(self) -> None:
        if self.exit_time < self.entry_time:
            raise ValueError(
                f"exit_time {self.exit_time} precedes entry_time {self.entry_time}"
            )
        if self.notional_micro_usd < 0:
            raise ValueError("notional_micro_usd must be >= 0")


@dataclass(frozen=True, slots=True)
class RandomTradeSpec:
    """One scheduled random-entry attempt: when to try, how long to hold, how
    much to spend. The symbol is deliberately *not* part of the schedule — it
    is drawn live from ``state.universe()`` at the moment the entry actually
    fires, which is what keeps the schedule itself point-in-time-safe: it
    commits only to a wall-clock-independent time and a duration, never to
    a symbol that might not exist yet when the schedule is built."""

    entry_time: float
    hold_seconds: float
    notional_micro_usd: int


def matched_random_schedule(
    trades: list[TradeStat], *, tick_grid: list[float], seed: int
) -> tuple[RandomTradeSpec, ...]:
    """Build a random-entry schedule with the same trade count and the same
    holding-period *values* (not just the same mean) as ``trades``.

    For each reference trade, draw a uniformly random tick from ``tick_grid``
    as the random entry time, and reuse that trade's exact holding duration
    and notional. Because the duration is copied rather than re-sampled, the
    matched schedule's mean holding period equals the reference's mean holding
    period exactly, by construction — not approximately, and not only in
    expectation.

    ``tick_grid`` must be non-empty and is not required to be sorted; it is
    the set of ``now`` values at which ``propose`` will actually be called
    during the replay (e.g. every bar close), so an entry never lands between
    ticks where nothing could have fired anyway.
    """
    if not tick_grid:
        raise ValueError("tick_grid must not be empty")
    rng = random.Random(seed)
    schedule = []
    for trade in trades:
        idx = rng.randrange(len(tick_grid))
        entry_time = tick_grid[idx]
        schedule.append(
            RandomTradeSpec(
                entry_time=entry_time,
                hold_seconds=trade.exit_time - trade.entry_time,
                notional_micro_usd=trade.notional_micro_usd,
            )
        )
    return tuple(schedule)


@dataclass
class RandomEntryStrategy:
    """Random entry, matched trade count and matched holding period.

    The strongest null a candidate strategy must beat: same number of trades,
    same holding-period distribution, but *when* to enter and *what* to enter
    are decided by a seeded RNG instead of by any signal. A strategy that
    cannot beat this has no timing/selection edge — only exposure.

    Construct via :func:`build_random_entry_baseline`, which derives
    ``schedule`` from :func:`matched_random_schedule`. ``seed`` must come from
    ``backtest.clock.SimulatedClock.deterministic_seed("random_baseline")`` so
    a replay is reproducible byte-for-byte under a fixed run id, and differs
    under a different one.
    """

    schedule: tuple[RandomTradeSpec, ...]
    seed: int
    strategy_id: str = "random_entry_baseline"
    required_fidelity: FidelityTier = FidelityTier.TIER_0
    _symbol_rng: random.Random = field(init=False, repr=False)
    _open: dict[int, tuple[str, float]] = field(
        default_factory=dict, init=False, repr=False
    )
    _resolved: set[int] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self._symbol_rng = random.Random(self.seed)

    def propose(
        self, *, state: PointInTimeState, portfolio: PortfolioState, now: float, run_id: str
    ) -> tuple[OrderIntent, ...]:
        intents: list[OrderIntent] = []

        # Exits: every open trade whose pre-committed hold duration has
        # elapsed, regardless of universe membership (see module docstring).
        for idx, (symbol, exit_time) in list(self._open.items()):
            if now < exit_time:
                continue
            position = portfolio.positions.get(symbol)
            if position is not None:
                exit_intent = _sell_intent(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    position=position,
                    now=now,
                    run_id=run_id,
                )
                if exit_intent is not None:
                    intents.append(exit_intent)
            del self._open[idx]
            self._resolved.add(idx)

        # Entries: any scheduled slot whose time has arrived and has not yet
        # fired or been skipped.
        for idx, spec in enumerate(self.schedule):
            if idx in self._open or idx in self._resolved:
                continue
            if now < spec.entry_time:
                continue
            universe = sorted(state.universe())
            if not universe:
                # Documented rule: a missed entry is not retried later.
                self._resolved.add(idx)
                continue
            symbol = self._symbol_rng.choice(universe)
            entry_intent = _buy_intent(
                strategy_id=self.strategy_id,
                symbol=symbol,
                notional_micro_usd=spec.notional_micro_usd,
                now=now,
                run_id=run_id,
            )
            if entry_intent is None:
                self._resolved.add(idx)
                continue
            intents.append(entry_intent)
            self._open[idx] = (symbol, spec.entry_time + spec.hold_seconds)

        return tuple(intents)


def build_random_entry_baseline(
    trades: list[TradeStat], *, tick_grid: list[float], seed: int
) -> RandomEntryStrategy:
    """Build a :class:`RandomEntryStrategy` matched to ``trades`` on both trade
    count and holding-period distribution. ``seed`` should come from
    ``SimulatedClock.deterministic_seed("random_baseline")``."""
    schedule = matched_random_schedule(trades, tick_grid=tick_grid, seed=seed)
    return RandomEntryStrategy(schedule=schedule, seed=seed)


__all__ = [
    "BacktestStrategy",
    "BuyAndHoldStrategy",
    "RandomEntryStrategy",
    "RandomTradeSpec",
    "TradeStat",
    "build_random_entry_baseline",
    "equal_weight_strategy",
    "matched_random_schedule",
    "mean_reversion_strategy",
    "momentum_strategy",
]
