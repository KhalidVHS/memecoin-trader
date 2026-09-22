"""Backtest performance metrics — PnL, risk-adjusted returns and trade statistics.

Why this module exists as a separate unit rather than living inside backtest/:
metrics are consumed by the promotion gate (validation/), the experiment
registry (experiments/), the CLI report (report.py) and — critically — by the
benchmarks in this same package. Keeping them out of backtest/ prevents a
circular dependency and makes them independently testable against known series.

Annualisation design decision
------------------------------
Every function that produces an annualised number takes ``periods_per_year`` as
an *explicit* argument.  A hardcoded 252 (equity days) is silently wrong for a
5-minute memecoin series where the correct value is 365 * 24 * 12 = 105,120.
Getting this wrong produces Sharpe ratios off by sqrt(105120/252) ≈ 20×, which
is the difference between "looks strong" and "might beat cash".  Callers must
make the choice they mean.

Drawdown definition
--------------------
We replicate risk.py's definition verbatim so that live and backtest drawdown
mean the same thing and halt thresholds are directly comparable:

    running_peak = max(equity[:i+1])
    drawdown_pct[i] = 100 * (running_peak - equity[i]) / running_peak

See ContinuousRisk.evaluate in risk.py lines 992–997.

Fold-level dispersion
----------------------
A single aggregate Sharpe can hide a strategy whose entire edge came from one
meme cycle or one calendar quarter.  FoldMetrics are first-class outputs so the
promotion gate can check "no single fold contributes > 25% of total PnL" and
the analyst can see median/worst-fold behaviour rather than just the aggregate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from memetrader.types import (
    FidelityTier, Fill, NON_EXECUTABLE_NOTICE, OrderState, Side, finite
)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class FoldMetrics:
    """Performance for one time-fold, e.g. one walk-forward window.

    Fold-level metrics are the honest unit of evidence.  An aggregate Sharpe
    averaged over N folds with wildly different per-fold Sharpes is not N times
    more reliable — it may be hiding that one fold drove the entire result.
    """

    fold_id: int
    sharpe: float | None
    total_return_pct: float
    max_drawdown_pct: float
    n_periods: int


@dataclass
class PerformanceMetrics:
    """All performance numbers for one backtest run or sub-window.

    Fields are grouped: annualised risk-adjusted ratios, absolute return/risk,
    trade-level statistics, portfolio-level statistics, dispersion across folds,
    and provenance metadata.

    Any ``None`` means "could not be computed from the available data", not
    "zero".  A Sharpe of None on a flat equity curve is more honest than 0.0.
    """

    # -- Annualised risk-adjusted ----------------------------------------
    sharpe: float | None
    sortino: float | None
    calmar: float | None

    # -- Absolute return / risk ------------------------------------------
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    max_drawdown_duration_periods: int

    # -- Trade-level statistics ------------------------------------------
    # Only closed (LANDED) fills with non-zero realized_pnl contribute.
    hit_rate_pct: float | None
    expectancy_usd: float | None
    profit_factor: float | None
    trade_count: int
    avg_holding_periods: float | None  # in units of the sampling period

    # -- Portfolio-level statistics --------------------------------------
    # turnover = sum(|notional_traded|) / mean_equity * 100 per period
    turnover_pct: float | None
    avg_exposure_pct: float | None

    # -- Fold-level dispersion ------------------------------------------
    folds: list[FoldMetrics]

    # -- Provenance / fidelity labelling --------------------------------
    fidelity: FidelityTier
    non_executable_notice: str | None
    periods_per_year: float
    n_periods: int


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


def max_drawdown(equity: Sequence[float]) -> tuple[float, int]:
    """Maximum drawdown and its duration, matching risk.py's definition exactly.

    Returns (max_drawdown_pct, max_duration_periods) where:
    - max_drawdown_pct = 100 * (running_peak - trough) / running_peak
    - max_duration_periods = longest consecutive run of periods below the peak

    Duration counts every bar below the then-current peak, whether or not the
    eventual trough has been reached yet.  This is the definition that matches
    the live system's halt condition rather than the "time to recovery"
    alternative used in some libraries.

    If equity has fewer than 2 points, both outputs are 0.
    """
    arr = [float(e) for e in equity]
    if len(arr) < 2:
        return 0.0, 0

    peak = arr[0]
    max_dd = 0.0
    max_dur = 0
    cur_dur = 0

    for e in arr[1:]:
        if e > peak:
            peak = e
            cur_dur = 0
        else:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)

        if peak > 0.0:
            dd = 100.0 * (peak - e) / peak
            max_dd = max(max_dd, dd)

    return max_dd, max_dur


def _period_returns(equity: Sequence[float]) -> np.ndarray:
    """Arithmetic period returns as a numpy array.

    Uses arithmetic (not log) returns because fill PnL is arithmetic and the
    Sharpe formula here is the standard arithmetic version.  Using log returns
    with arithmetic Sharpe would understate variance on high-vol series.
    """
    arr = np.array([float(e) for e in equity], dtype=np.float64)
    if len(arr) < 2:
        return np.empty(0, dtype=np.float64)
    prev = arr[:-1]
    # Avoid divide-by-zero: a zero equity value cannot produce a meaningful
    # return; treat as 0.0 rather than inf or nan.
    with np.errstate(invalid="ignore", divide="ignore"):
        rets = np.where(prev > 0.0, (arr[1:] - prev) / prev, 0.0)
    return rets


def _sharpe(
    returns: np.ndarray,
    periods_per_year: float,
    risk_free_rate_annual: float,
) -> float | None:
    """Annualised Sharpe from a return series.

    Returns None rather than inf or nan when std is zero (flat equity).
    A Sharpe of None is more honest than claiming infinite precision.
    """
    if len(returns) < 2:
        return None
    rf_per_period = risk_free_rate_annual / periods_per_year
    excess = returns - rf_per_period
    std = float(np.std(excess, ddof=1))
    if std == 0.0:
        return None
    mean = float(np.mean(excess))
    return float(mean / std * math.sqrt(periods_per_year))


def _sortino(
    returns: np.ndarray,
    periods_per_year: float,
    mar_annual: float,
) -> float | None:
    """Annualised Sortino ratio using downside deviation below MAR.

    MAR (minimum acceptable return) is supplied as an annualised percentage.
    We convert to per-period before comparing so the threshold is consistent
    regardless of sampling frequency — another place where hardcoding ruins
    cross-frequency comparisons.
    """
    if len(returns) < 2:
        return None
    mar_per_period = mar_annual / periods_per_year
    # Downside deviation uses min(r - mar, 0) over ALL periods (not just downside
    # periods). This matches the standard Sortino formula and means a strategy
    # that never underperforms MAR produces downside_dev = 0 → return None.
    below = np.minimum(returns - mar_per_period, 0.0)
    downside_std = float(np.sqrt(np.mean(below**2)))
    if downside_std == 0.0:
        # No underperformance at all — infinite Sortino, return None to avoid
        # misleading the promotion gate with a non-finite value.
        return None
    mean_excess = float(np.mean(returns - mar_per_period))
    return float(mean_excess / downside_std * math.sqrt(periods_per_year))


def _annualized_return(
    equity: Sequence[float], periods_per_year: float
) -> float:
    """Annualised arithmetic return expressed as a percentage.

    Uses the geometric compounding formula so that a 100% gain followed by a
    50% loss shows as 0%, not a misleading positive average of period returns.
    """
    arr = [float(e) for e in equity]
    if len(arr) < 2 or arr[0] <= 0.0:
        return 0.0
    total_ret = (arr[-1] - arr[0]) / arr[0]
    n_periods = len(arr) - 1
    # Geometric annualisation — clamp at -100% to avoid math domain errors.
    clamped = max(1.0 + total_ret, 1e-10)
    return float((clamped ** (periods_per_year / n_periods) - 1.0) * 100.0)


def _total_return_pct(equity: Sequence[float]) -> float:
    arr = [float(e) for e in equity]
    if len(arr) < 2 or arr[0] <= 0.0:
        return 0.0
    return float((arr[-1] - arr[0]) / arr[0] * 100.0)


# ---------------------------------------------------------------------------
# Trade-level statistics
# ---------------------------------------------------------------------------


def _trade_stats(
    fills: Sequence[Fill],
) -> tuple[float | None, float | None, float | None, int, float | None]:
    """Return (hit_rate_pct, expectancy_usd, profit_factor, trade_count,
    avg_holding_periods).

    Only LANDED fills with nonzero realized_pnl_usd are counted as closed
    trades.  Failed / expired fills are real costs (captured in attribution)
    but do not constitute a "trade" in the W/L sense — including them as losses
    would double-count costs.

    Holding period: we estimate from paired BUY→SELL sequences per symbol.
    Each BUY opens a position; the next SELL of the same symbol closes it.
    We record the time gap in *seconds* and report it in terms of the period
    length implied by periods_per_year — but since we do not have that here, we
    return the raw second count and let compute_performance convert it.  This
    avoids threading periods_per_year deep into a private helper.
    """
    closed: list[float] = []
    holding_seconds: list[float] = []

    # Build per-symbol open-time stack from BUY fills
    open_ts: dict[str, list[float]] = {}
    for f in sorted(fills, key=lambda x: x.ts):  # noqa: E731
        if f.state is not OrderState.LANDED:
            continue
        sym = f.symbol
        if f.side is Side.BUY:
            open_ts.setdefault(sym, []).append(f.ts)
        elif f.side is Side.SELL:
            if open_ts.get(sym):
                entry_ts = open_ts[sym].pop(0)
                holding_seconds.append(f.ts - entry_ts)
            if f.realized_pnl_usd != 0.0:
                closed.append(f.realized_pnl_usd)

    trade_count = len(closed)
    if trade_count == 0:
        return None, None, None, 0, None

    wins = [p for p in closed if p > 0.0]
    losses = [p for p in closed if p < 0.0]
    hit_rate = float(len(wins) / trade_count * 100.0)
    expectancy = float(sum(closed) / trade_count)

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor: float | None = None
    if gross_loss > 0.0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0.0:
        profit_factor = None  # all wins — infinite, report as None (see docstring)

    avg_hold: float | None = (
        float(sum(holding_seconds) / len(holding_seconds)) if holding_seconds else None
    )

    return hit_rate, expectancy, profit_factor, trade_count, avg_hold


# ---------------------------------------------------------------------------
# Portfolio-level statistics
# ---------------------------------------------------------------------------


def _portfolio_stats(
    equity: Sequence[float],
    fills: Sequence[Fill],
) -> tuple[float | None, float | None]:
    """Return (turnover_pct, avg_exposure_pct).

    Turnover = sum(|notional traded|) / mean_equity * 100.  A common
    alternative is "as a fraction of starting capital", but mean equity is
    more informative for a strategy where capital compounds.

    Exposure requires per-bar position values, which we do not have in this
    signature — only equity and fills.  We cannot reconstruct bar-level
    exposure from fills alone without a replay, so we return None rather than
    a misleading estimate.  A caller with access to PortfolioState snapshots
    can compute this independently.
    """
    arr = [float(e) for e in equity]
    mean_eq = float(np.mean(arr)) if arr else 0.0
    if mean_eq <= 0.0:
        return None, None

    total_notional = sum(f.notional_usd for f in fills if not f.failed)
    turnover = float(total_notional / mean_eq * 100.0) if total_notional > 0 else 0.0
    return turnover, None  # exposure requires bar snapshots


# ---------------------------------------------------------------------------
# Fold metrics
# ---------------------------------------------------------------------------


def _fold_metrics(
    equity: Sequence[float],
    fold_ids: Sequence[int],
    periods_per_year: float,
    risk_free_rate_annual: float,
) -> list[FoldMetrics]:
    """Compute per-fold Sharpe/return/drawdown.

    fold_ids is parallel to equity (same length). Each unique fold_id produces
    one FoldMetrics entry.  Folds with fewer than 2 points produce None Sharpe.
    """
    arr = list(equity)
    ids = list(fold_ids)
    if len(arr) != len(ids):
        raise ValueError(
            f"equity length {len(arr)} != fold_ids length {len(ids)}"
        )

    # Group indices by fold
    fold_indices: dict[int, list[int]] = {}
    for i, fid in enumerate(ids):
        fold_indices.setdefault(fid, []).append(i)

    results: list[FoldMetrics] = []
    for fid in sorted(fold_indices):
        idxs = fold_indices[fid]
        fold_eq = [arr[i] for i in idxs]
        rets = _period_returns(fold_eq)
        sh = _sharpe(rets, periods_per_year, risk_free_rate_annual)
        dd, _ = max_drawdown(fold_eq)
        tr = _total_return_pct(fold_eq)
        results.append(
            FoldMetrics(
                fold_id=fid,
                sharpe=sh,
                total_return_pct=tr,
                max_drawdown_pct=dd,
                n_periods=len(fold_eq),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_performance(
    equity: Sequence[float],
    fills: Sequence[Fill],
    *,
    periods_per_year: float,
    fidelity: FidelityTier,
    risk_free_rate_annual: float = 0.0,
    fold_ids: Sequence[int] | None = None,
    mar: float = 0.0,
) -> PerformanceMetrics:
    """Compute all performance metrics from an equity curve and fill log.

    Parameters
    ----------
    equity:
        NAV at each bar close, in dollars.  Length N.
    fills:
        All fills from the run.  Used for trade-level statistics only; the
        equity curve is the authoritative PnL source.
    periods_per_year:
        Sampling frequency expressed as periods per calendar year.  Required.
        5-minute bars: 365 * 24 * 12 = 105,120.
        Hourly bars:   365 * 24     =   8,760.
        Daily bars:    252.
        Passing the wrong value here is the single most common source of
        inflated Sharpe ratios in crypto backtests.
    fidelity:
        FidelityTier of the underlying data.  Any result below TIER_2 carries
        NON_EXECUTABLE_NOTICE verbatim — see BACKTEST-CONTRACTS.md §3.
    risk_free_rate_annual:
        Annual risk-free rate as a decimal (0.05 = 5 %).  Applied per period.
    fold_ids:
        Integer fold assignment parallel to equity.  When supplied, fold-level
        Sharpe/return/drawdown are computed and stored in folds.
    mar:
        Minimum acceptable return for Sortino, annualised as a decimal.
    """
    arr = [float(e) for e in equity]
    n = len(arr)
    notice = NON_EXECUTABLE_NOTICE if not fidelity.permits_pnl_claim else None

    if n < 2:
        return PerformanceMetrics(
            sharpe=None,
            sortino=None,
            calmar=None,
            total_return_pct=0.0,
            annualized_return_pct=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_duration_periods=0,
            hit_rate_pct=None,
            expectancy_usd=None,
            profit_factor=None,
            trade_count=0,
            avg_holding_periods=None,
            turnover_pct=None,
            avg_exposure_pct=None,
            folds=[],
            fidelity=fidelity,
            non_executable_notice=notice,
            periods_per_year=periods_per_year,
            n_periods=n,
        )

    rets = _period_returns(arr)
    sh = _sharpe(rets, periods_per_year, risk_free_rate_annual)
    so = _sortino(rets, periods_per_year, mar)
    ann_ret = _annualized_return(arr, periods_per_year)
    tot_ret = _total_return_pct(arr)
    dd_pct, dd_dur = max_drawdown(arr)

    # Calmar: annualised_return_pct / max_drawdown_pct.  Both are percentages
    # so the ratio is dimensionless.  None when drawdown is zero (division by
    # zero) rather than infinity — infinity is not a useful promotion gate value.
    calmar: float | None = None
    if dd_pct > 0.0:
        calmar = ann_ret / dd_pct

    hit, exp, pf, tc, avg_hold_sec = _trade_stats(fills)

    # Convert holding seconds to periods
    avg_hold_periods: float | None = None
    if avg_hold_sec is not None and periods_per_year > 0.0:
        seconds_per_year = 365.25 * 24 * 3600
        seconds_per_period = seconds_per_year / periods_per_year
        avg_hold_periods = avg_hold_sec / seconds_per_period

    turnover, exposure = _portfolio_stats(arr, fills)

    folds: list[FoldMetrics] = []
    if fold_ids is not None:
        folds = _fold_metrics(arr, fold_ids, periods_per_year, risk_free_rate_annual)

    return PerformanceMetrics(
        sharpe=sh,
        sortino=so,
        calmar=calmar,
        total_return_pct=tot_ret,
        annualized_return_pct=ann_ret,
        max_drawdown_pct=dd_pct,
        max_drawdown_duration_periods=dd_dur,
        hit_rate_pct=hit,
        expectancy_usd=exp,
        profit_factor=pf,
        trade_count=tc,
        avg_holding_periods=avg_hold_periods,
        turnover_pct=turnover,
        avg_exposure_pct=exposure,
        folds=folds,
        fidelity=fidelity,
        non_executable_notice=notice,
        periods_per_year=periods_per_year,
        n_periods=n,
    )
