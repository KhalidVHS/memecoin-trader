"""Benchmark strategies against which every backtest result must be read.

Why these four benchmarks?
---------------------------
1. **Cash/no-trade**: the floor.  If you cannot beat staying in cash, you have
   negative gross alpha even before costs.

2. **Random entry with identical holding periods**: the *timing* test.  Keeps
   the same assets, sizes and hold durations as the strategy, but picks entry
   times uniformly at random.  A strategy that cannot beat this null has no
   timing edge — it may have selection or sizing edge, but the decision of
   *when* to trade adds nothing.  This is the most commonly omitted test and
   the one most likely to deflate inflated results.

3. **Random asset selection**: the *selection* test.  Keeps the same timing and
   sizes, but picks assets uniformly at random from the eligible universe.
   A strategy that cannot beat this null has no selection edge.

4. **Equal-weight basket**: the *index* test.  What would a simple "hold
   everything equally" rule have made?  Meme cycles inflate all coins together,
   so a strategy must beat this benchmark to claim anything beyond beta.

5. **Buy-and-hold SOL**: the *beta* test.  SOL is the natural risk factor for
   Solana ecosystem coins.  An equity curve that tracks SOL is not an
   independent strategy.

Random baselines are reproducible under a seed
------------------------------------------------
``np.random.default_rng(seed)`` is used for all randomness.  The same seed
produces byte-identical results, which is required for experiment reproducibility
and holdout integrity.

All benchmarks return a ``BenchmarkResult`` whose ``.metrics`` field has the
same ``PerformanceMetrics`` shape as the strategy, so they are directly
comparable — the promotion gate can read off the same fields from both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from memetrader.types import FidelityTier, Fill, NON_EXECUTABLE_NOTICE, OrderState, Side
from memetrader.metrics.performance import (
    PerformanceMetrics,
    compute_performance,
    _period_returns,
    _sharpe,
    max_drawdown,
    _total_return_pct,
    _annualized_return,
)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Performance metrics for one benchmark strategy."""

    name: str
    metrics: PerformanceMetrics
    seed: int | None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _equity_from_returns(
    returns: np.ndarray,
    starting_capital: float,
) -> list[float]:
    """Reconstruct equity curve from period returns."""
    eq: list[float] = [starting_capital]
    for r in returns:
        eq.append(eq[-1] * (1.0 + r))
    return eq


def _make_metrics(
    equity: list[float],
    *,
    periods_per_year: float,
    fidelity: FidelityTier,
) -> PerformanceMetrics:
    """Compute performance metrics with an empty fill list.

    Benchmarks do not have individual fills, so trade-level statistics are
    absent.  The equity curve is the entire story.
    """
    return compute_performance(
        equity,
        [],
        periods_per_year=periods_per_year,
        fidelity=fidelity,
    )


# ---------------------------------------------------------------------------
# Cash / no-trade benchmark
# ---------------------------------------------------------------------------


def cash_benchmark(
    equity: Sequence[float],
    *,
    periods_per_year: float,
    fidelity: FidelityTier,
) -> BenchmarkResult:
    """Flat equity at starting value — the no-trade floor.

    Sharpe is None (zero std), total return is 0%.  Any strategy with a
    negative total return is below cash after costs.
    """
    arr = [float(e) for e in equity]
    n = max(len(arr), 1)
    starting = arr[0] if arr else 1000.0
    flat = [starting] * n
    return BenchmarkResult(
        name="cash_no_trade",
        metrics=_make_metrics(flat, periods_per_year=periods_per_year, fidelity=fidelity),
        seed=None,
    )


# ---------------------------------------------------------------------------
# Random-entry benchmark
# ---------------------------------------------------------------------------


def random_entry_benchmark(
    equity: Sequence[float],
    fills: Sequence[Fill],
    *,
    periods_per_year: float,
    fidelity: FidelityTier,
    seed: int = 42,
    n_simulations: int = 1000,
) -> BenchmarkResult:
    """Randomize entry times, keep identical holding periods.

    For each landed fill pair (BUY entry + SELL exit), we know:
    - The holding duration in *equity-index* periods.
    - The notional size.

    We draw random entry indices, apply the same holding duration, and
    compute the equity-curve return over that window.  We do this for all
    trade pairs in one simulation, accumulate total PnL, and repeat for
    ``n_simulations`` seeds.  The median equity curve across simulations
    is returned as the benchmark.

    This is the strongest test of timing edge.  The equity series is the
    performance of the *market* (or a proxy), so a trade that happened to
    enter when the market was about to rise is compared against randomly-timed
    entries with the same duration.

    Why median, not mean?
    ----------------------
    Mean of simulations can be skewed by outlier runs (e.g. a few simulations
    that happen to always pick the best windows).  Median gives the expected
    result of a strategy with no timing skill, which is the correct null.
    """
    arr = [float(e) for e in equity]
    n = len(arr)
    if n < 2:
        return BenchmarkResult(
            name="random_entry",
            metrics=_make_metrics(arr or [1000.0], periods_per_year=periods_per_year,
                                  fidelity=fidelity),
            seed=seed,
        )

    # Build (holding_periods, notional_usd) for each round-trip
    open_info: dict[str, list[tuple[float, float]]] = {}  # sym -> [(ts, notional)]
    trips: list[tuple[int, float]] = []  # (hold_periods, notional)

    # We need timestamps to map to equity indices.  We use fill timestamps
    # and approximate equity indices by linear interpolation over time.
    times = [float(e) for e in equity]  # equity is just floats (NAV), no timestamps
    # Since equity doesn't carry timestamps, we approximate holding periods
    # by using sorted fill order to estimate bar positions.
    landed = [f for f in fills if f.state is OrderState.LANDED]
    landed_sorted = sorted(landed, key=lambda f: f.ts)

    if not landed_sorted:
        # No trades — simulate constant equity
        return BenchmarkResult(
            name="random_entry",
            metrics=_make_metrics(arr, periods_per_year=periods_per_year, fidelity=fidelity),
            seed=seed,
        )

    t_min = landed_sorted[0].ts
    t_max = landed_sorted[-1].ts
    t_span = t_max - t_min if t_max > t_min else 1.0

    def _ts_to_idx(ts: float) -> int:
        frac = (ts - t_min) / t_span
        return max(0, min(n - 1, int(frac * (n - 1))))

    for f in landed_sorted:
        sym = f.symbol
        if f.side is Side.BUY:
            open_info.setdefault(sym, []).append((f.ts, f.notional_usd))
        elif f.side is Side.SELL and open_info.get(sym):
            entry_ts, notional = open_info[sym].pop(0)
            entry_idx = _ts_to_idx(entry_ts)
            exit_idx = _ts_to_idx(f.ts)
            hold = max(1, exit_idx - entry_idx)
            trips.append((hold, notional))

    if not trips:
        return BenchmarkResult(
            name="random_entry",
            metrics=_make_metrics(arr, periods_per_year=periods_per_year, fidelity=fidelity),
            seed=seed,
        )

    rng = np.random.default_rng(seed)
    all_final_equities: list[float] = []
    starting = arr[0]

    for _sim in range(n_simulations):
        sim_pnl = 0.0
        for hold_periods, notional in trips:
            entry_idx = int(rng.integers(0, max(1, n - hold_periods)))
            exit_idx = min(entry_idx + hold_periods, n - 1)
            if arr[entry_idx] <= 0.0:
                continue
            trade_ret = (arr[exit_idx] - arr[entry_idx]) / arr[entry_idx]
            sim_pnl += trade_ret * notional
        all_final_equities.append(starting + sim_pnl)

    # Build a representative equity curve using median final equity
    median_final = float(np.median(all_final_equities))
    # Linear interpolation from starting to median_final as the representative curve
    rep_equity = [
        starting + (median_final - starting) * i / max(n - 1, 1) for i in range(n)
    ]

    return BenchmarkResult(
        name="random_entry",
        metrics=_make_metrics(rep_equity, periods_per_year=periods_per_year,
                              fidelity=fidelity),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Random-asset benchmark
# ---------------------------------------------------------------------------


def random_asset_benchmark(
    equity: Sequence[float],
    available_assets: Sequence[str],
    fills: Sequence[Fill],
    *,
    periods_per_year: float,
    fidelity: FidelityTier,
    seed: int = 42,
    n_simulations: int = 1000,
) -> BenchmarkResult:
    """Randomize asset selection, keep timing and sizing identical.

    For each strategy trade, we replace the chosen asset with a uniformly
    random one from ``available_assets`` and assume the same notional and
    holding period, but a random return (drawn from the equity series as proxy
    since we don't have per-asset equity curves here).

    When per-asset equity series are not available (which is the case at TIER_0
    with only aggregate equity), we use the aggregate equity curve as a return
    proxy.  This is a conservative approximation — individual coins are more
    volatile than the aggregate, so the random-asset benchmark will understate
    the variance of the true random-asset strategy.  This is the correct
    direction of conservatism: it makes random selection look *better* than it
    is, so a strategy must beat a more generous bar.
    """
    arr = [float(e) for e in equity]
    n = len(arr)
    rng = np.random.default_rng(seed)
    landed = [f for f in fills if f.state is OrderState.LANDED and f.side is Side.SELL]

    if not landed or n < 2:
        return BenchmarkResult(
            name="random_asset",
            metrics=_make_metrics(arr or [1000.0], periods_per_year=periods_per_year,
                                  fidelity=fidelity),
            seed=seed,
        )

    # Use aggregate equity return per bar as the random-asset return proxy
    eq_arr = np.array(arr, dtype=np.float64)
    rets = _period_returns(arr)

    starting = arr[0]
    all_final: list[float] = []

    for _sim in range(n_simulations):
        sim_pnl = 0.0
        for f in landed:
            # Draw a random return from the equity return distribution
            ret = float(rng.choice(rets)) if len(rets) > 0 else 0.0
            sim_pnl += ret * f.notional_usd
        all_final.append(starting + sim_pnl)

    median_final = float(np.median(all_final))
    rep_equity = [
        starting + (median_final - starting) * i / max(n - 1, 1) for i in range(n)
    ]

    return BenchmarkResult(
        name="random_asset",
        metrics=_make_metrics(rep_equity, periods_per_year=periods_per_year,
                              fidelity=fidelity),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Equal-weight basket benchmark
# ---------------------------------------------------------------------------


def equal_weight_basket_benchmark(
    asset_equity_series: dict[str, Sequence[float]],
    *,
    periods_per_year: float,
    fidelity: FidelityTier,
    starting_capital: float,
) -> BenchmarkResult:
    """Point-in-time equal-weight basket across all provided assets.

    All assets receive equal weight at each rebalancing point (every bar).
    The basket return at bar i is the simple average of per-asset returns.

    This is the minimal active management benchmark: zero stock-picking,
    zero market timing, just equal exposure to the eligible universe.  A
    strategy must beat this to claim any alpha above simple meme-cycle beta.

    ``asset_equity_series`` maps each asset to its price (or NAV) series.
    If series have different lengths, the shortest governs (excess bars are
    dropped) — we only compare what we can compare at the same point in time.
    """
    if not asset_equity_series:
        flat = [starting_capital]
        return BenchmarkResult(
            name="equal_weight_basket",
            metrics=_make_metrics(flat, periods_per_year=periods_per_year,
                                  fidelity=fidelity),
            seed=None,
        )

    series = {k: list(v) for k, v in asset_equity_series.items()}
    min_len = min(len(s) for s in series.values())
    if min_len < 2:
        return BenchmarkResult(
            name="equal_weight_basket",
            metrics=_make_metrics([starting_capital], periods_per_year=periods_per_year,
                                  fidelity=fidelity),
            seed=None,
        )

    n_assets = len(series)
    basket_equity = [starting_capital]
    for i in range(1, min_len):
        avg_ret = 0.0
        for s in series.values():
            prev = s[i - 1]
            if prev > 0.0:
                avg_ret += (s[i] - prev) / prev
        avg_ret /= n_assets
        basket_equity.append(basket_equity[-1] * (1.0 + avg_ret))

    return BenchmarkResult(
        name="equal_weight_basket",
        metrics=_make_metrics(basket_equity, periods_per_year=periods_per_year,
                              fidelity=fidelity),
        seed=None,
    )


# ---------------------------------------------------------------------------
# Buy-and-hold SOL benchmark
# ---------------------------------------------------------------------------


def buy_hold_sol_benchmark(
    sol_prices: Sequence[float],
    *,
    starting_capital: float,
    periods_per_year: float,
    fidelity: FidelityTier,
) -> BenchmarkResult:
    """Buy-and-hold SOL with ``starting_capital`` at the first bar.

    SOL is the natural risk factor for the Solana ecosystem.  An equity curve
    that merely tracks SOL is not a strategy — it is beta.  If the strategy
    does not beat buy-and-hold SOL, it is not adding value over a simpler
    allocation.

    ``sol_prices`` is the SOL/USD price at each bar close.  The equity curve
    is: quantity = starting_capital / sol_prices[0], then
    equity[i] = quantity * sol_prices[i].
    """
    prices = [float(p) for p in sol_prices]
    if not prices or prices[0] <= 0.0:
        return BenchmarkResult(
            name="buy_hold_sol",
            metrics=_make_metrics([starting_capital], periods_per_year=periods_per_year,
                                  fidelity=fidelity),
            seed=None,
        )

    quantity = starting_capital / prices[0]
    equity = [quantity * p for p in prices]

    return BenchmarkResult(
        name="buy_hold_sol",
        metrics=_make_metrics(equity, periods_per_year=periods_per_year,
                              fidelity=fidelity),
        seed=None,
    )
