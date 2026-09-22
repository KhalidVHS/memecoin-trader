"""Tests for point_in_time.ReplayState — the most critical module.

Every test here is designed to fail if the point-in-time guard is removed.
Specifically:
  - Refusing a bar that has not closed
  - Making data appear later when available_time increases
  - Making future-dated records invisible
  - The bars() method returning oldest-first within the lookback window

These are the mandatory tests from BACKTEST-CONTRACTS.md §8.
"""

from __future__ import annotations

import pytest

from memetrader.histdata.point_in_time import ReplayState
from memetrader.histdata.schemas import PoolState, QuoteLadderRung, QuoteLadder
from memetrader.types import Candle, Side, Timeframe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _closed_candle(ts: float, close: float = 1.0) -> Candle:
    return Candle(ts=ts, open=close, high=close * 1.01, low=close * 0.99, close=close,
                  volume=100.0, closed=True)


def _open_candle(ts: float) -> Candle:
    """A forming bar — not yet closed."""
    return Candle(ts=ts, open=1.0, high=1.01, low=0.99, close=1.0,
                  volume=50.0, closed=False)


MINT_A = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"  # BONK mint
POOL_A = "5zpyutJu9ee6jFymDGoK7F6S5Kczqtc9FomP3ueKuyA9"


# ---------------------------------------------------------------------------
# Test: closed-bar enforcement
# ---------------------------------------------------------------------------


def test_refuses_unclosed_bar() -> None:
    """ReplayState.add_bar must reject a bar that has not closed.

    This test fails if the guard in _CandleStore.add is removed. The forming
    bar is the look-ahead bug the spec explicitly calls out.
    """
    state = ReplayState(now=0.0)
    forming = _open_candle(ts=1_000_000.0)
    with pytest.raises(ValueError, match="has not closed"):
        state.add_bar(forming, asset_id=MINT_A, timeframe=Timeframe.H1,
                      available_time=1_000_000.0)


def test_accepts_closed_bar() -> None:
    """Closed bars are accepted without error."""
    state = ReplayState(now=0.0)
    bar = _closed_candle(ts=1_000_000.0)
    state.add_bar(bar, asset_id=MINT_A, timeframe=Timeframe.H1,
                  available_time=1_000_001.0)
    # Nothing raised


# ---------------------------------------------------------------------------
# Test: future records are invisible
# ---------------------------------------------------------------------------


def test_future_bar_invisible() -> None:
    """A bar with available_time > now must not be returned by bars().

    This is the 'Future sentinel' test from BACKTEST-CONTRACTS.md §8.
    If the bisect guard is removed, the future bar would appear in results
    and any feature computed from it would change, which is detectable.
    """
    now = 1_700_000_000.0
    state = ReplayState(now=now)

    # This bar is available 1 second in the future — must not appear
    future_bar = _closed_candle(ts=1_000_000.0, close=99999.0)
    state.add_bar(future_bar, asset_id=MINT_A, timeframe=Timeframe.H1,
                  available_time=now + 1.0)

    # Current-time bar
    past_bar = _closed_candle(ts=999_999.0, close=1.0)
    state.add_bar(past_bar, asset_id=MINT_A, timeframe=Timeframe.H1,
                  available_time=now - 1.0)

    result = state.bars(MINT_A, Timeframe.H1, lookback=100)
    assert len(result) == 1
    assert result[0].close == 1.0, "future bar must not appear"


def test_future_bar_appears_after_clock_advance() -> None:
    """The same bar becomes visible once now advances past its available_time."""
    t0 = 1_700_000_000.0
    state = ReplayState(now=t0)

    bar = _closed_candle(ts=1_000_000.0, close=42.0)
    state.add_bar(bar, asset_id=MINT_A, timeframe=Timeframe.H1,
                  available_time=t0 + 100.0)

    # Not visible yet
    assert len(state.bars(MINT_A, Timeframe.H1, lookback=100)) == 0

    # Advance clock
    state.now = t0 + 100.0
    result = state.bars(MINT_A, Timeframe.H1, lookback=100)
    assert len(result) == 1
    assert result[0].close == 42.0


# ---------------------------------------------------------------------------
# Test: raising available_time makes data appear later (timestamp delay)
# ---------------------------------------------------------------------------


def test_raising_available_time_delays_visibility() -> None:
    """Increasing available_time must make a bar visible later.

    This is the 'Timestamp delay' test from BACKTEST-CONTRACTS.md §8.
    Two ReplayState instances with different available_times for the same bar:
    the one with the later available_time must not return the bar at the
    earlier time.
    """
    t_bar = 1_000_000.0
    t_early_available = 1_001_000.0  # bar + 1000s
    t_late_available = 1_002_000.0   # bar + 2000s
    t_query = 1_001_500.0  # between early and late

    # State with early available_time
    state_early = ReplayState(now=t_query)
    state_early.add_bar(_closed_candle(t_bar, close=1.0), asset_id=MINT_A,
                        timeframe=Timeframe.H1, available_time=t_early_available)

    # State with late available_time
    state_late = ReplayState(now=t_query)
    state_late.add_bar(_closed_candle(t_bar, close=1.0), asset_id=MINT_A,
                       timeframe=Timeframe.H1, available_time=t_late_available)

    # Early state sees the bar at t_query
    assert len(state_early.bars(MINT_A, Timeframe.H1, lookback=10)) == 1
    # Late state does NOT see it — the same test failing proves look-ahead
    assert len(state_late.bars(MINT_A, Timeframe.H1, lookback=10)) == 0


# ---------------------------------------------------------------------------
# Test: default available_time = ts + interval + delay
# ---------------------------------------------------------------------------


def test_default_available_time_uses_interval_plus_delay() -> None:
    """Without an explicit available_time, a bar is available at ts + interval + delay.

    This is the publication-delay invariant from BACKTEST-CONTRACTS.md §1:
    'A candle is unavailable until ts + interval + publication_delay.'
    """
    delay = 30.0
    state = ReplayState(now=0.0, publication_delay_seconds=delay)

    ts = 1_700_000_000.0
    bar = _closed_candle(ts=ts, close=2.0)
    state.add_bar(bar, asset_id=MINT_A, timeframe=Timeframe.H1)

    interval = 3600.0
    expected_available = ts + interval + delay

    # Just before available: invisible
    state.now = expected_available - 0.001
    assert len(state.bars(MINT_A, Timeframe.H1, lookback=10)) == 0

    # Exactly at available: visible
    state.now = expected_available
    assert len(state.bars(MINT_A, Timeframe.H1, lookback=10)) == 1


def test_default_available_time_m5() -> None:
    """Same publication-delay test for the 5m timeframe (interval = 300s)."""
    state = ReplayState(now=0.0, publication_delay_seconds=0.0)
    ts = 1_700_000_000.0
    state.add_bar(_closed_candle(ts=ts), asset_id=MINT_A, timeframe=Timeframe.M5)

    # 5m interval = 300s
    state.now = ts + 299.0
    assert len(state.bars(MINT_A, Timeframe.M5, lookback=10)) == 0

    state.now = ts + 300.0
    assert len(state.bars(MINT_A, Timeframe.M5, lookback=10)) == 1


# ---------------------------------------------------------------------------
# Test: lookback window
# ---------------------------------------------------------------------------


def test_lookback_limits_returned_bars() -> None:
    """bars() returns at most lookback bars, the most recent ones."""
    base_ts = 1_700_000_000.0
    interval = 3600.0
    n = 10
    # Set now past all bars: bar[9].available_time = base_ts + (9+1)*interval
    state = ReplayState(now=base_ts + n * interval + 1.0)

    # Load 10 bars, each available at ts + interval (publication delay = 0)
    for i in range(n):
        ts = base_ts + i * interval
        bar = _closed_candle(ts=ts, close=float(i + 1))
        state.add_bar(bar, asset_id=MINT_A, timeframe=Timeframe.H1,
                      available_time=ts + interval)

    result = state.bars(MINT_A, Timeframe.H1, lookback=3)
    assert len(result) == 3
    # Should be the last 3 bars (indices 7, 8, 9 → close=8, 9, 10)
    assert result[0].close == 8.0
    assert result[2].close == 10.0


def test_bars_oldest_first() -> None:
    """bars() returns bars in ascending ts order (oldest first)."""
    base_ts = 1_700_000_000.0
    interval = 3600.0
    n = 5
    state = ReplayState(now=base_ts + n * interval + 1.0)

    for i in range(n):
        ts = base_ts + i * interval
        state.add_bar(_closed_candle(ts=ts), asset_id=MINT_A, timeframe=Timeframe.H1,
                      available_time=ts + interval)

    result = state.bars(MINT_A, Timeframe.H1, lookback=10)
    assert all(result[i].ts < result[i + 1].ts for i in range(len(result) - 1))


def test_empty_when_no_data() -> None:
    """bars() returns empty tuple when no data has been loaded."""
    state = ReplayState(now=1_700_000_000.0)
    assert state.bars(MINT_A, Timeframe.H1, lookback=10) == ()


# ---------------------------------------------------------------------------
# Test: universe integration
# ---------------------------------------------------------------------------


def test_universe_empty_before_first_update() -> None:
    """universe() returns empty frozenset when no updates have been loaded."""
    state = ReplayState(now=1_700_000_000.0)
    assert state.universe() == frozenset()


def test_universe_point_in_time() -> None:
    """universe() returns the most recent snapshot at or before now."""
    t0 = 1_700_000_000.0
    state = ReplayState(now=t0)

    coins_early = frozenset(["MINT_A", "MINT_B"])
    coins_late = frozenset(["MINT_A", "MINT_B", "MINT_C"])

    state.set_universe(coins_early, available_time=t0 - 1000.0)
    state.set_universe(coins_late, available_time=t0 + 1000.0)

    # At t0, the late update is not yet available
    assert state.universe() == coins_early

    state.now = t0 + 1000.0
    assert state.universe() == coins_late


def test_universe_future_update_invisible() -> None:
    """A universe update in the future must not affect current membership."""
    t0 = 1_700_000_000.0
    state = ReplayState(now=t0)
    state.set_universe(frozenset(["A"]), available_time=t0 + 1.0)

    # Future update is invisible at t0
    assert state.universe() == frozenset()


# ---------------------------------------------------------------------------
# Test: pool state and quote ladder
# ---------------------------------------------------------------------------


def _make_pool_state(pool_id: str, available_time: float) -> PoolState:
    return PoolState(
        asset_id=MINT_A,
        pool_id=pool_id,
        venue="raydium",
        event_time=available_time - 1.0,
        available_time=available_time,
        received_time=available_time,
        reserve_in_atomic=1_000_000,
        reserve_out_atomic=2_000_000,
        fee_rate_bps=25,
        price_usd=1.0,
        liquidity_usd=100_000.0,
        source="test",
    )


def test_pool_state_point_in_time() -> None:
    """pool_state() returns the most recent state at or before now."""
    t0 = 1_700_000_000.0
    state = ReplayState(now=t0)

    early = _make_pool_state(POOL_A, available_time=t0 - 100.0)
    late = _make_pool_state(POOL_A, available_time=t0 + 100.0)

    state.add_pool_state(early)
    state.add_pool_state(late)

    ps = state.pool_state(POOL_A)
    assert ps is not None
    assert ps.available_time == t0 - 100.0  # early one, not future one


def test_pool_state_none_before_first() -> None:
    """pool_state() returns None when now is before the first state."""
    state = ReplayState(now=0.0)
    state.add_pool_state(_make_pool_state(POOL_A, available_time=1000.0))
    assert state.pool_state(POOL_A) is None


def _make_ladder(asset_id: str, available_time: float) -> QuoteLadder:
    rung = QuoteLadderRung(
        in_amount_atomic=1_000_000,
        out_amount_atomic=950_000,
        price_impact_pct=0.5,
        route_labels=("raydium",),
        fees_atomic=5_000,
        min_out_atomic=940_000,
    )
    return QuoteLadder(
        asset_id=asset_id,
        pool_id=POOL_A,
        side="BUY",
        event_time=available_time - 1.0,
        available_time=available_time,
        received_time=available_time,
        rungs=(rung,),
        context_slot=None,
        source="test",
    )


def test_quote_ladder_point_in_time() -> None:
    """quote_ladder() returns the most recent ladder at or before now."""
    t0 = 1_700_000_000.0
    state = ReplayState(now=t0)

    early = _make_ladder(MINT_A, available_time=t0 - 200.0)
    late = _make_ladder(MINT_A, available_time=t0 + 200.0)

    state.add_quote_ladder(early)
    state.add_quote_ladder(late)

    ladder = state.quote_ladder(MINT_A, Side.BUY)
    assert ladder is not None
    assert ladder.available_time == t0 - 200.0


# ---------------------------------------------------------------------------
# Test: bar_count helper
# ---------------------------------------------------------------------------


def test_bar_count_all_bars_regardless_of_now() -> None:
    """bar_count() returns total loaded bars, not filtered by now."""
    state = ReplayState(now=0.0)
    for i in range(5):
        bar = _closed_candle(ts=float(i * 3600), close=1.0)
        state.add_bar(bar, asset_id=MINT_A, timeframe=Timeframe.H1,
                      available_time=float((i + 1) * 3600))
    # now=0 means none are visible via bars(), but bar_count sees all
    assert state.bar_count(MINT_A, Timeframe.H1) == 5
    assert state.bars(MINT_A, Timeframe.H1, lookback=100) == ()


# ---------------------------------------------------------------------------
# Test: protocol structural subtyping
# ---------------------------------------------------------------------------


def test_replay_state_satisfies_protocol() -> None:
    """ReplayState must be an instance of PointInTimeState (runtime_checkable)."""
    from memetrader.histdata.point_in_time import PointInTimeState

    state = ReplayState(now=0.0)
    assert isinstance(state, PointInTimeState)
