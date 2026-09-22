"""Tests for ``memetrader.strategies.baselines``.

Fixtures are built inline against ``histdata.point_in_time.ReplayState`` — the
concrete ``PointInTimeState`` — never against the gitignored ``history/``
directory. Every test is offline.
"""

from __future__ import annotations

from memetrader.backtest.clock import SimulatedClock
from memetrader.execution.interfaces import ApprovedOrder
from memetrader.histdata.point_in_time import ReplayState
from memetrader.strategies.baselines import (
    BacktestStrategy,
    BuyAndHoldStrategy,
    RandomEntryStrategy,
    RandomTradeSpec,
    TradeStat,
    build_random_entry_baseline,
    equal_weight_strategy,
    mean_reversion_strategy,
    momentum_strategy,
)
from memetrader.types import (
    Candle,
    Fill,
    OrderIntent,
    PortfolioState,
    Position,
    Side,
    Timeframe,
)

_H1 = 3600.0


def _candle(ts: float, close: float, *, volume: float = 100.0) -> Candle:
    return Candle(
        ts=ts, open=close, high=close, low=close, close=close, volume=volume, closed=True
    )


def _portfolio(
    *, cash_usd: float = 10_000.0, positions: dict[str, Position] | None = None
) -> PortfolioState:
    positions = positions if positions is not None else {}
    return PortfolioState(
        ts=0.0,
        cash_usd=cash_usd,
        positions=positions,
        marks={},
        position_values_usd={},
        unrealized_pnl_usd=0.0,
        realized_pnl_usd=0.0,
        total_value_usd=cash_usd,
        starting_cash_usd=cash_usd,
    )


def _position(
    symbol: str, *, quantity_atomic: int = 1_000_000, decimals: int = 6
) -> Position:
    return Position(
        symbol=symbol,
        mint=symbol,
        quantity_atomic=quantity_atomic,
        decimals=decimals,
        avg_entry_price_usd=1.0,
        opened_at=0.0,
        cost_basis_usd=1.0,
    )


def _replay_state(
    bars_by_symbol: dict[str, list[Candle]],
    *,
    now: float,
    universe: set[str],
    timeframe: Timeframe = Timeframe.H1,
) -> ReplayState:
    state = ReplayState(now=now, publication_delay_seconds=0.0)
    for symbol, candles in bars_by_symbol.items():
        for candle in candles:
            state.add_bar(candle, asset_id=symbol, timeframe=timeframe)
    state.set_universe(frozenset(universe), available_time=0.0)
    return state


def _tick_grid(n: int, interval: float = _H1) -> list[float]:
    return [i * interval for i in range(n)]


# ---------------------------------------------------------------------------
# Prefix / future-sentinel safety
# ---------------------------------------------------------------------------


def test_momentum_strategy_cannot_see_a_future_bar() -> None:
    """A candle whose available_time is after `now` must not change the pick.

    This must fail if a strategy ever bypasses PointInTimeState and reads a
    raw series directly.
    """
    candles = [_candle(i * _H1, close=100.0 + i) for i in range(10)]
    future_spike = _candle(100 * _H1, close=1_000_000.0)
    now = 9 * _H1 + _H1  # bar #9 has just become available

    without_future = _replay_state(
        {"AAA": candles, "BBB": candles}, now=now, universe={"AAA", "BBB"}
    )
    with_future = _replay_state(
        {"AAA": [*candles, future_spike], "BBB": candles}, now=now, universe={"AAA", "BBB"}
    )

    strat_a = momentum_strategy(lookback_seconds=5 * _H1, top_k=1)
    strat_b = momentum_strategy(lookback_seconds=5 * _H1, top_k=1)
    portfolio = _portfolio()

    out_a = strat_a.propose(
        state=without_future, portfolio=portfolio, now=now, run_id="run"
    )
    out_b = strat_b.propose(state=with_future, portfolio=portfolio, now=now, run_id="run")

    assert out_a == out_b


def test_bars_available_only_at_or_after_publication() -> None:
    """A strategy queried before a bar's close cannot see that bar at all."""
    candles = [_candle(i * _H1, close=100.0 + i) for i in range(5)]
    early_now = 2 * _H1  # only bars 0 and 1 have closed by here (bar N closes at (N+1)*H1)
    state = _replay_state({"AAA": candles}, now=early_now, universe={"AAA"})
    visible = state.bars("AAA", Timeframe.H1, lookback=100)
    assert all(c.ts < early_now for c in visible)
    assert len(visible) == 2


# ---------------------------------------------------------------------------
# Strategies emit only OrderIntent
# ---------------------------------------------------------------------------


def test_strategies_emit_only_order_intent() -> None:
    candles = [_candle(i * _H1, close=100.0 + (-1) ** i) for i in range(10)]
    now = 9 * _H1 + _H1
    state = _replay_state(
        {"AAA": candles, "BBB": candles}, now=now, universe={"AAA", "BBB"}
    )
    portfolio = _portfolio()

    strategies: list[BacktestStrategy] = [
        BuyAndHoldStrategy(),
        equal_weight_strategy(),
        momentum_strategy(lookback_seconds=5 * _H1, top_k=1),
        mean_reversion_strategy(lookback_seconds=5 * _H1, top_k=1),
    ]
    for strat in strategies:
        out = strat.propose(state=state, portfolio=portfolio, now=now, run_id="run")
        assert out, f"{strat.strategy_id} produced nothing to check"
        for item in out:
            assert isinstance(item, OrderIntent)
            assert not isinstance(item, Fill)
            assert not isinstance(item, ApprovedOrder)


# ---------------------------------------------------------------------------
# Universe shrinkage
# ---------------------------------------------------------------------------


def test_equal_weight_full_exits_on_universe_departure() -> None:
    candles = [_candle(i * _H1, close=100.0) for i in range(3)]
    now = 2 * _H1 + _H1
    positions = {"AAA": _position("AAA"), "BBB": _position("BBB")}
    portfolio = _portfolio(cash_usd=0.0, positions=positions)

    # BBB has left the universe.
    state = _replay_state({"AAA": candles, "BBB": candles}, now=now, universe={"AAA"})
    strat = equal_weight_strategy()
    out = strat.propose(state=state, portfolio=portfolio, now=now, run_id="run")

    sells = [o for o in out if o.side is Side.SELL]
    assert {o.symbol for o in sells} == {"BBB"}
    assert sells[0].in_amount_atomic == positions["BBB"].quantity_atomic


def test_buy_and_hold_survives_delisting_without_selling() -> None:
    candles = [_candle(i * _H1, close=100.0) for i in range(3)]
    now = _H1
    strat = BuyAndHoldStrategy()
    state_present = _replay_state({"AAA": candles}, now=now, universe={"AAA"})
    portfolio = _portfolio()
    bought = strat.propose(state=state_present, portfolio=portfolio, now=now, run_id="run")
    assert len(bought) == 1
    portfolio = _portfolio(
        cash_usd=portfolio.cash_usd - bought[0].in_amount_atomic / 1_000_000,
        positions={"AAA": _position("AAA")},
    )

    # AAA is delisted (out of the universe) on the next tick.
    state_delisted = _replay_state({"AAA": candles}, now=now + _H1, universe=set())
    out = strat.propose(
        state=state_delisted, portfolio=portfolio, now=now + _H1, run_id="run"
    )
    assert out == ()  # buy-and-hold never sells


def test_buy_and_hold_retries_an_entry_that_never_settled() -> None:
    """A proposal that is vetoed must not cost the symbol permanently.

    Proposing is not buying: the intent can be vetoed by risk or fail to
    route, and the ledger moves no cash until a fill lands. If the strategy
    treats "proposed" as "bought", one transient veto means it never holds
    that symbol at all — and a buy-and-hold benchmark holding nothing flatters
    every strategy measured against it.
    """
    candles = [_candle(i * _H1, close=100.0) for i in range(6)]
    strat = BuyAndHoldStrategy()
    portfolio = _portfolio()

    first = strat.propose(
        state=_replay_state({"AAA": candles}, now=_H1, universe={"AAA"}),
        portfolio=portfolio,
        now=_H1,
        run_id="run",
    )
    assert len(first) == 1

    # The entry is vetoed: no fill, so cash is untouched and AAA never enters
    # the book. Within the reservation window the strategy must stay quiet
    # rather than stack a second full-size buy on top of the in-flight one.
    held = _H1 + strat.entry_reservation_seconds / 2
    assert (
        strat.propose(
            state=_replay_state({"AAA": candles}, now=held, universe={"AAA"}),
            portfolio=portfolio,
            now=held,
            run_id="run",
        )
        == ()
    )

    # Once the reservation lapses with still no position, it tries again.
    later = _H1 + strat.entry_reservation_seconds + 1.0
    retry = strat.propose(
        state=_replay_state({"AAA": candles}, now=later, universe={"AAA"}),
        portfolio=portfolio,
        now=later,
        run_id="run",
    )
    assert [o.symbol for o in retry] == ["AAA"]


def test_buy_and_hold_does_not_rebuy_a_settled_position() -> None:
    """The lapsing reservation must not turn into repeat buying.

    Once the fill lands the symbol is in ``portfolio.positions``, and that —
    not the reservation — is what keeps buy-and-hold from buying twice.
    """
    candles = [_candle(i * _H1, close=100.0) for i in range(6)]
    strat = BuyAndHoldStrategy()
    bought = strat.propose(
        state=_replay_state({"AAA": candles}, now=_H1, universe={"AAA"}),
        portfolio=_portfolio(),
        now=_H1,
        run_id="run",
    )
    assert len(bought) == 1

    settled = _portfolio(cash_usd=5_000.0, positions={"AAA": _position("AAA")})
    long_after = _H1 + strat.entry_reservation_seconds * 10
    assert (
        strat.propose(
            state=_replay_state({"AAA": candles}, now=long_after, universe={"AAA"}),
            portfolio=settled,
            now=long_after,
            run_id="run",
        )
        == ()
    )


# ---------------------------------------------------------------------------
# Random entry baseline
# ---------------------------------------------------------------------------


def test_random_entry_seed_reproducible_and_seed_sensitive() -> None:
    trades = [
        TradeStat(entry_time=0.0, exit_time=_H1, notional_micro_usd=5_000_000),
        TradeStat(entry_time=_H1, exit_time=3 * _H1, notional_micro_usd=3_000_000),
        TradeStat(
            entry_time=2 * _H1, exit_time=2 * _H1 + 5 * _H1, notional_micro_usd=2_000_000
        ),
    ]
    grid = _tick_grid(20)

    seed_a1 = SimulatedClock(run_id="run-a").deterministic_seed("random_baseline")
    seed_a2 = SimulatedClock(run_id="run-a").deterministic_seed("random_baseline")
    seed_b = SimulatedClock(run_id="run-b").deterministic_seed("random_baseline")
    assert seed_a1 == seed_a2
    assert seed_a1 != seed_b

    strat_1 = build_random_entry_baseline(trades, tick_grid=grid, seed=seed_a1)
    strat_2 = build_random_entry_baseline(trades, tick_grid=grid, seed=seed_a1)
    strat_3 = build_random_entry_baseline(trades, tick_grid=grid, seed=seed_b)

    assert strat_1.schedule == strat_2.schedule
    assert strat_1.schedule != strat_3.schedule

    assert len(strat_1.schedule) == len(trades)
    mean_ref = sum(t.exit_time - t.entry_time for t in trades) / len(trades)
    mean_random = sum(s.hold_seconds for s in strat_1.schedule) / len(strat_1.schedule)
    assert mean_ref == mean_random


def test_random_entry_byte_identical_orders_for_same_seed() -> None:
    trades = [TradeStat(entry_time=0.0, exit_time=2 * _H1, notional_micro_usd=4_000_000)]
    grid = _tick_grid(10)
    universe = {"AAA", "BBB", "CCC"}
    candles = {sym: [_candle(t, close=1.0) for t in grid] for sym in universe}

    def run(seed: int) -> list[OrderIntent]:
        strat = build_random_entry_baseline(trades, tick_grid=grid, seed=seed)
        portfolio = _portfolio()
        collected: list[OrderIntent] = []
        for now in grid:
            state = _replay_state(candles, now=now, universe=universe)
            out = strat.propose(state=state, portfolio=portfolio, now=now, run_id="run-x")
            collected.extend(out)
            for intent in out:
                if intent.side is Side.BUY:
                    portfolio.positions[intent.symbol] = _position(intent.symbol)
                else:
                    portfolio.positions.pop(intent.symbol, None)
        return collected

    out_1 = run(seed=1234)
    out_2 = run(seed=1234)
    out_3 = run(seed=5678)

    assert out_1 == out_2
    assert out_1 != out_3
    assert len(out_1) >= 1


def test_random_entry_skips_missed_window_without_retry() -> None:
    """No universe at the scheduled entry tick -> the entry is skipped, not deferred.

    Built directly from an explicit schedule (rather than through
    ``build_random_entry_baseline``) so the entry tick under test is pinned,
    not left to whatever tick the RNG happens to draw.
    """
    schedule = (
        RandomTradeSpec(entry_time=0.0, hold_seconds=_H1, notional_micro_usd=1_000_000),
    )
    strat = RandomEntryStrategy(schedule=schedule, seed=1)
    portfolio = _portfolio()

    empty_state = _replay_state({}, now=0.0, universe=set())
    out = strat.propose(state=empty_state, portfolio=portfolio, now=0.0, run_id="run")
    assert out == ()  # scheduled entry tick had no universe -> skipped

    grid = _tick_grid(5)
    later_state = _replay_state(
        {"AAA": [_candle(t, 1.0) for t in grid]}, now=_H1, universe={"AAA"}
    )
    out_later = strat.propose(state=later_state, portfolio=portfolio, now=_H1, run_id="run")
    assert out_later == ()  # not retried
