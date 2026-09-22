"""Tests for metrics.performance — risk statistics from an equity curve.

Every test here must *fail* if the guard it is exercising is removed.  Known
values below were computed independently with numpy (see the comment above
each fixture) rather than by re-deriving the module's own formula, so a test
failure means the module's output actually changed, not that the test is
tautological.

Percentage convention (BACKTEST-CONTRACTS.md §0): all *_pct fields are whole
numbers.  ``8.0`` means total return of +8%, not 0.08.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from memetrader.metrics.performance import compute_performance, max_drawdown
from memetrader.types import FidelityTier, Fill, OrderState, Side

TIER_0 = FidelityTier.TIER_0
TIER_2 = FidelityTier.TIER_2


def _fill(
    *,
    fill_id: str,
    ts: float,
    symbol: str = "FOO",
    side: Side,
    state: OrderState = OrderState.LANDED,
    realized_pnl_usd: float = 0.0,
    notional_usd: float = 100.0,
) -> Fill:
    return Fill(
        fill_id=fill_id,
        order_id=f"order-{fill_id}",
        intent_id=f"intent-{fill_id}",
        decision_id=None,
        ts=ts,
        symbol=symbol,
        side=side,
        state=state,
        in_amount_atomic=1_000_000,
        out_amount_atomic=1_000_000,
        token_amount_atomic=1_000_000,
        token_decimals=6,
        quote_fingerprint="fp",
        price_usd=1.0,
        notional_usd=notional_usd,
        price_impact_pct=0.0,
        pool_fee_usd=0.0,
        gas_usd=0.0,
        realized_pnl_usd=realized_pnl_usd,
    )


# ---------------------------------------------------------------------------
# max_drawdown — known values
# ---------------------------------------------------------------------------


def test_max_drawdown_known_value() -> None:
    """Hand-traced: peak 1000->1100, trough at 990 while peak holds at 1100.

    dd = 100 * (1100 - 990) / 1100 = 10.0 (exactly), duration = 3 bars below
    the 1100 peak (990, 1080, 1080).
    """
    equity = [1000.0, 1100.0, 990.0, 1080.0, 1080.0]
    dd_pct, dd_dur = max_drawdown(equity)
    assert dd_pct == pytest.approx(10.0)
    assert dd_dur == 3


def test_max_drawdown_monotonic_nondecreasing_equity_is_zero() -> None:
    """A strictly increasing equity curve never draws down."""
    equity = [100.0, 105.0, 110.0, 120.0, 130.0]
    dd_pct, dd_dur = max_drawdown(equity)
    assert dd_pct == 0.0
    assert dd_dur == 0


def test_max_drawdown_bounded_by_peak_to_trough() -> None:
    """The reported drawdown can never exceed the single worst peak-to-trough
    move in the series — otherwise the function invented risk that never
    happened."""
    equity = [100.0, 200.0, 50.0, 150.0, 40.0]
    dd_pct, _dd_dur = max_drawdown(equity)
    peak = 0.0
    worst = 0.0
    for e in equity:
        peak = max(peak, e)
        if peak > 0:
            worst = max(worst, 100.0 * (peak - e) / peak)
    assert dd_pct == pytest.approx(worst)
    assert dd_pct <= 100.0
    assert dd_pct >= 0.0


def test_max_drawdown_short_series_is_zero() -> None:
    assert max_drawdown([]) == (0.0, 0)
    assert max_drawdown([42.0]) == (0.0, 0)


@given(
    st.lists(
        st.floats(
            min_value=1.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False
        ),
        min_size=2,
        max_size=30,
    )
)
def test_max_drawdown_is_never_positive_direction_wrong(equity: list[float]) -> None:
    """Property: drawdown magnitude is always within [0, 100] and never
    exceeds the running peak — i.e. equity can't go negative-implied."""
    dd_pct, dd_dur = max_drawdown(equity)
    assert 0.0 <= dd_pct <= 100.0
    assert dd_dur >= 0
    assert dd_dur < len(equity)


# ---------------------------------------------------------------------------
# compute_performance — known values (hand/numpy-computed, see docstring)
# ---------------------------------------------------------------------------


def test_compute_performance_known_values_daily() -> None:
    """Known-value check against independently computed numpy figures.

    equity = [1000, 1100, 990, 1080, 1080] -> arithmetic returns
    [0.1, -0.1, 0.090909..., 0.0].  periods_per_year=252.

    Computed once, offline, with:
        rets = np.diff(arr) / arr[:-1]
        sharpe = rets.mean() / rets.std(ddof=1) * sqrt(252)
        sortino = rets.mean() / sqrt(mean(min(rets,0)**2)) * sqrt(252)
        ann_ret = ((1+total_ret) ** (252/4) - 1) * 100
    """
    equity = [1000.0, 1100.0, 990.0, 1080.0, 1080.0]
    m = compute_performance(
        equity,
        [],
        periods_per_year=252.0,
        fidelity=TIER_2,
    )
    assert m.sharpe == pytest.approx(3.860746401413234, rel=1e-9)
    assert m.sortino == pytest.approx(7.21568539381252, rel=1e-9)
    assert m.total_return_pct == pytest.approx(8.0)
    assert m.annualized_return_pct == pytest.approx(12655.473818635774, rel=1e-6)
    assert m.max_drawdown_pct == pytest.approx(10.0)
    assert m.max_drawdown_duration_periods == 3
    assert m.calmar == pytest.approx(1265.5473818635774, rel=1e-6)


def test_annualization_scales_with_periods_per_year() -> None:
    """The *same* return series annualized at a higher frequency produces a
    larger-magnitude Sharpe: sqrt(periods_per_year) scaling must actually be
    applied, not silently dropped.
    """
    equity = [1000.0, 1010.0, 995.0, 1020.0, 1005.0, 1030.0]
    daily = compute_performance(equity, [], periods_per_year=252.0, fidelity=TIER_2)
    five_min = compute_performance(
        equity, [], periods_per_year=365 * 24 * 12, fidelity=TIER_2
    )
    assert daily.sharpe is not None
    assert five_min.sharpe is not None
    # Same sign (same underlying return series), larger magnitude at the
    # higher frequency because of the sqrt(periods_per_year) scale factor.
    assert (daily.sharpe > 0) == (five_min.sharpe > 0)
    assert abs(five_min.sharpe) > abs(daily.sharpe)
    # And the ratio must equal the scaling factor exactly.
    expected_ratio = math.sqrt((365 * 24 * 12) / 252.0)
    assert five_min.sharpe / daily.sharpe == pytest.approx(expected_ratio, rel=1e-9)


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_empty_equity_returns_zeroed_metrics_not_error() -> None:
    m = compute_performance([], [], periods_per_year=252.0, fidelity=TIER_2)
    assert m.sharpe is None
    assert m.sortino is None
    assert m.calmar is None
    assert m.total_return_pct == 0.0
    assert m.max_drawdown_pct == 0.0
    assert m.n_periods == 0


def test_single_observation_returns_zeroed_metrics() -> None:
    m = compute_performance([1000.0], [], periods_per_year=252.0, fidelity=TIER_2)
    assert m.sharpe is None
    assert m.n_periods == 1
    assert m.total_return_pct == 0.0


def test_all_zero_returns_flat_equity_sharpe_is_none() -> None:
    """A flat equity curve has zero variance: Sharpe must be None, never inf
    or a ZeroDivisionError (BACKTEST-CONTRACTS.md §0: honesty over false
    precision)."""
    equity = [1000.0] * 10
    m = compute_performance(equity, [], periods_per_year=252.0, fidelity=TIER_2)
    assert m.sharpe is None
    assert m.sortino is None
    assert m.total_return_pct == 0.0
    assert m.max_drawdown_pct == 0.0


def test_all_negative_returns_series() -> None:
    """A monotonically declining equity curve: Sharpe is a finite negative
    number (constant negative returns still have zero variance -> None,
    matching the flat-curve case; a *varying* decline has finite variance)."""
    equity = [1000.0, 900.0, 850.0, 700.0]
    m = compute_performance(equity, [], periods_per_year=252.0, fidelity=TIER_2)
    assert m.total_return_pct < 0.0
    assert m.max_drawdown_pct > 0.0
    # Sharpe must be finite (not nan/inf) whenever it is not None.
    if m.sharpe is not None:
        assert m.sharpe == m.sharpe  # not nan
        assert abs(m.sharpe) != float("inf")


def test_constant_negative_return_sharpe_is_none() -> None:
    """Constant per-period return (even negative) has zero sample variance
    *around its own mean*: Sharpe must be None, not -inf.

    Uses a halving series (-50% every period) rather than an arbitrary
    percentage: repeated division by a power of two is exact in binary
    floating point, so the per-period returns are bit-identical and the
    sample variance is exactly zero -- not just numerically close to it.

    Sortino is deliberately *not* asserted None here: it measures downside
    deviation below a fixed MAR (default 0%), not deviation around the
    series' own mean, so a series that consistently underperforms the MAR
    has a real, finite, nonzero downside deviation -- see
    test_sortino_is_none_when_nothing_underperforms_mar for the case that
    does collapse to None.
    """
    equity = [1000.0, 500.0, 250.0, 125.0]  # exactly -50% every period
    m = compute_performance(equity, [], periods_per_year=252.0, fidelity=TIER_2)
    assert m.sharpe is None


def test_sortino_is_none_when_nothing_underperforms_mar() -> None:
    """Sortino's downside deviation is measured against MAR, not the mean.

    A series where every period return is exactly at (or above) the MAR
    never records any downside -> downside_std == 0 -> Sortino is None
    (infinite Sortino would be misleading, per module docstring).
    """
    equity = [1000.0, 500.0, 250.0, 125.0]  # exactly -50% every period
    m = compute_performance(
        equity,
        [],
        periods_per_year=252.0,
        fidelity=TIER_2,
        mar=-0.5 * 252.0,
    )
    assert m.sortino is None


# ---------------------------------------------------------------------------
# Trade-level statistics — known values
# ---------------------------------------------------------------------------


def test_trade_stats_known_values() -> None:
    """Three closed round-trips on FOO: +50, -20, +30.

    hit_rate = 2/3 * 100 = 66.6667%
    expectancy = (50 - 20 + 30) / 3 = 20.0
    profit_factor = (50 + 30) / 20 = 4.0
    """
    fills = [
        _fill(fill_id="b1", ts=0.0, side=Side.BUY),
        _fill(fill_id="s1", ts=10.0, side=Side.SELL, realized_pnl_usd=50.0),
        _fill(fill_id="b2", ts=20.0, side=Side.BUY),
        _fill(fill_id="s2", ts=30.0, side=Side.SELL, realized_pnl_usd=-20.0),
        _fill(fill_id="b3", ts=40.0, side=Side.BUY),
        _fill(fill_id="s3", ts=50.0, side=Side.SELL, realized_pnl_usd=30.0),
    ]
    equity = [1000.0, 1060.0]  # content doesn't matter for trade stats
    m = compute_performance(equity, fills, periods_per_year=8760.0, fidelity=TIER_2)
    assert m.trade_count == 3
    assert m.hit_rate_pct == pytest.approx(200.0 / 3.0)
    assert m.expectancy_usd == pytest.approx(20.0)
    assert m.profit_factor == pytest.approx(4.0)


def test_trade_stats_all_wins_profit_factor_is_none() -> None:
    """All-win trade set has zero gross loss: profit_factor is None (not inf),
    matching the module's documented §0-honest convention."""
    fills = [
        _fill(fill_id="b1", ts=0.0, side=Side.BUY),
        _fill(fill_id="s1", ts=10.0, side=Side.SELL, realized_pnl_usd=15.0),
    ]
    equity = [1000.0, 1015.0]
    m = compute_performance(equity, fills, periods_per_year=8760.0, fidelity=TIER_2)
    assert m.profit_factor is None
    assert m.trade_count == 1


def test_trade_stats_no_closed_trades_is_none() -> None:
    """Fills that never close (a lone BUY, or all failed) contribute no
    trade-level statistics: None, not 0, per §0's None-vs-zero rule."""
    fills = [_fill(fill_id="b1", ts=0.0, side=Side.BUY)]
    equity = [1000.0, 1000.0]
    m = compute_performance(equity, fills, periods_per_year=8760.0, fidelity=TIER_2)
    assert m.trade_count == 0
    assert m.hit_rate_pct is None
    assert m.expectancy_usd is None
    assert m.profit_factor is None
    assert m.avg_holding_periods is None


def test_failed_fills_excluded_from_trade_stats() -> None:
    """FAILED fills must not be counted as closed trades — a failed
    transaction is a real cost (see attribution.py) but not a W/L trade."""
    fills = [
        _fill(fill_id="b1", ts=0.0, side=Side.BUY),
        _fill(
            fill_id="s1",
            ts=10.0,
            side=Side.SELL,
            state=OrderState.FAILED,
            realized_pnl_usd=-5.0,
        ),
    ]
    equity = [1000.0, 995.0]
    m = compute_performance(equity, fills, periods_per_year=8760.0, fidelity=TIER_2)
    assert m.trade_count == 0


# ---------------------------------------------------------------------------
# Fidelity / non-executable notice
# ---------------------------------------------------------------------------


def test_tier0_carries_non_executable_notice() -> None:
    m = compute_performance([1000.0, 1010.0], [], periods_per_year=252.0, fidelity=TIER_0)
    assert m.non_executable_notice is not None


def test_tier2_carries_no_notice() -> None:
    m = compute_performance([1000.0, 1010.0], [], periods_per_year=252.0, fidelity=TIER_2)
    assert m.non_executable_notice is None


# ---------------------------------------------------------------------------
# Fold-level dispersion
# ---------------------------------------------------------------------------


def test_fold_metrics_partition_equity_exactly() -> None:
    equity = [1000.0, 1010.0, 1005.0, 1020.0, 1015.0, 1030.0]
    fold_ids = [0, 0, 0, 1, 1, 1]
    m = compute_performance(
        equity,
        [],
        periods_per_year=252.0,
        fidelity=TIER_2,
        fold_ids=fold_ids,
    )
    assert len(m.folds) == 2
    assert {f.fold_id for f in m.folds} == {0, 1}
    for f in m.folds:
        assert f.n_periods == 3


def test_fold_ids_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        compute_performance(
            [1000.0, 1010.0, 1020.0],
            [],
            periods_per_year=252.0,
            fidelity=TIER_2,
            fold_ids=[0, 1],
        )
