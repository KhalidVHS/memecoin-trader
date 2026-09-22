"""Resampling tools for serially dependent return series.

Why not iid bootstrap?  A standard iid bootstrap destroys autocorrelation:
each resample independently draws single observations, so the dependence
structure of daily or hourly returns is lost entirely.  The consequence is
that bootstrap confidence intervals for statistics like Sharpe ratio are
over-narrow — the estimator thinks it has more independent observations than
it does.  Block bootstraps fix this by resampling contiguous *blocks* of
returns, preserving the local dependence structure within each block.

Two variants are provided:

* **Circular block bootstrap** (CBB, Politis & Romano 1992) — fixed block
  length L, treats the series as circular (wraps around the end), so every
  observation appears in exactly L blocks.  Simple and symmetric, but the
  fixed length is a strong assumption about the autocorrelation decay rate.

* **Stationary bootstrap** (SB, Politis & Romano 1994) — geometric random
  block lengths with mean 1/p, which produces a resampled series that is
  itself strictly stationary (the name is exact).  The randomised block
  length smooths away the sharp edges that make CBB sensitive to L.

References
----------
Politis, D.N. & Romano, J.P. (1992).  A circular block-resampling procedure
for stationary data.  In *Exploring the Limits of Bootstrap*, pp. 263-270.

Politis, D.N. & Romano, J.P. (1994).  The stationary bootstrap.  *Journal of
the American Statistical Association*, 89(428), 1303-1313.

Politis, D.N. & White, H. (2004).  Automatic block-length selection for the
dependent bootstrap.  *Econometric Reviews*, 23(1), 53-70.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


class BootstrapCI(NamedTuple):
    """A confidence interval for one statistic, with its own provenance.

    ``block_length`` is surfaced here rather than hidden inside the function
    because block length is a researcher degree of freedom that changes the
    answer: too-short blocks leave autocorrelation intact and the CI is still
    over-narrow; too-long blocks give you too few effective blocks and the CI
    is over-wide.  Every experiment that uses bootstrap CIs must record it in
    the experiment registry, hence it travels with the result.
    """

    lower: float
    upper: float
    point_estimate: float
    method: str  # "percentile" or "bca"
    bootstrap_kind: str  # "stationary" or "circular"
    block_length: float  # mean block length (float for SB geometric mean)
    n_resamples: int
    confidence_level: float


# ---------------------------------------------------------------------------
# Automatic block-length selector
# ---------------------------------------------------------------------------


def select_block_length(
    x: npt.NDArray[np.float64],
    *,
    method: str = "politis_white",
) -> float:
    """Estimate an appropriate mean block length for a return series.

    Uses the Politis-White (2004) plug-in estimator, which is based on the
    spectral density at zero frequency and the sum of autocovariances.  The
    estimator targets the optimal block length for the stationary bootstrap
    under squared-error loss of the variance of the sample mean.

    The result is clipped to [2, n/4] — a block of 1 is iid resampling, and
    a block of n/2 or more gives too few independent blocks for a useful
    confidence interval.  The raw estimate is returned (after clipping) so
    the caller can log it.

    Parameters
    ----------
    x:
        1-D array of returns (or any real-valued time series).
    method:
        Only ``"politis_white"`` is implemented.  Accepted as a keyword so
        future methods can be added without changing call sites.
    """
    if method != "politis_white":
        msg = f"Unknown block-length method {method!r}; only 'politis_white' is supported"
        raise ValueError(msg)

    n = len(x)
    if n < 4:
        return 2.0

    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean()

    # Compute autocorrelations up to lag K_n = max(5, sqrt(n)).
    # The bandwidth K_n is the Politis-White recommendation; it limits how many
    # lags go into the spectral estimate without requiring a subjective choice.
    k_n = max(5, int(math.sqrt(n)))
    k_n = min(k_n, n - 1)

    # Biased autocovariance (divide by n, not n-lag) — consistent with PW2004.
    gamma = np.array([float(np.dot(xc[: n - lag], xc[lag:])) / n for lag in range(k_n + 1)])

    # Flat-top kernel weights: k(x) = 1 for |x| <= 0.5, linearly taper to 0
    # at |x| = 1 (Politis & White 2004 eq. 3.3).  The kernel suppresses
    # aliasing from high-lag autocovariances that are dominated by noise.
    lags = np.arange(k_n + 1, dtype=np.float64) / k_n
    weights = np.where(lags <= 0.5, 1.0, np.where(lags <= 1.0, 2.0 * (1.0 - lags), 0.0))

    # G_hat = 2 * sum_{l=1}^{K_n} l * k(l/K_n) * gamma_l (PW2004 eq. 3.4)
    g_hat = 2.0 * float(
        np.dot(np.arange(1, k_n + 1, dtype=np.float64) * weights[1:], gamma[1:])
    )

    # D_hat = 2 * sum_{l=-K_n}^{K_n} k(l/K_n) * gamma_|l| — the spectral
    # density at zero, used as the denominator.
    d_hat = 2.0 * float(np.dot(weights, gamma)) - gamma[0]
    d_hat = max(d_hat, 1e-12)  # guard divide-by-zero for constant series

    # Optimal block length: b* = (2 G^2 / D)^{1/3} * n^{1/3}  (PW2004 eq. 3.6)
    b_star = (2.0 * g_hat**2 / d_hat) ** (1.0 / 3.0) * n ** (1.0 / 3.0)

    # Clip to sensible range: at least 2, at most n/4.
    return float(np.clip(b_star, 2.0, n / 4.0))


# ---------------------------------------------------------------------------
# Core bootstrap engines
# ---------------------------------------------------------------------------


def stationary_bootstrap(
    x: npt.NDArray[np.float64],
    n_resamples: int,
    *,
    block_length: float,
    seed: int,
) -> npt.NDArray[np.float64]:
    """Resample ``x`` via the stationary bootstrap of Politis & Romano (1994).

    Block lengths are drawn from a geometric distribution with mean
    ``block_length`` (equivalently, continuation probability 1 - 1/L).  This
    produces a resampled series that is *itself* stationary — unlike the
    circular block bootstrap, the stationary bootstrap does not impose a
    periodic structure on the data.

    The resampled series is the same length as the input.

    Parameters
    ----------
    x:
        1-D array of returns, length n.
    n_resamples:
        Number of bootstrap replications.
    block_length:
        Mean geometric block length.  Must be >= 1.
    seed:
        RNG seed for exact reproducibility.  Using a fixed seed is required
        for any result that enters the experiment registry; a run with a
        different seed is a different experiment.

    Returns
    -------
    Array of shape ``(n_resamples, len(x))``.
    """
    if block_length < 1.0:
        msg = f"block_length must be >= 1, got {block_length}"
        raise ValueError(msg)

    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    rng = np.random.default_rng(seed)

    # Continuation probability p = 1/L — at each step we start a new block
    # with probability 1/L.  Expected block length is then 1/p = L.
    p = 1.0 / block_length

    out = np.empty((n_resamples, n), dtype=np.float64)
    for i in range(n_resamples):
        # Draw start indices and geometric continuation decisions together.
        # Vectorised: sample all n positions at once.
        starts = rng.integers(0, n, size=n)
        new_block = rng.random(size=n) < p

        idx = np.empty(n, dtype=np.intp)
        cur = int(starts[0])
        for j in range(n):
            cur = int(starts[j]) if j == 0 or new_block[j] else (cur + 1) % n
            idx[j] = cur
        out[i] = x[idx]

    return out


def circular_block_bootstrap(
    x: npt.NDArray[np.float64],
    n_resamples: int,
    *,
    block_length: int,
    seed: int,
) -> npt.NDArray[np.float64]:
    """Resample ``x`` via the circular block bootstrap (Politis & Romano 1992).

    Blocks of exactly ``block_length`` observations are drawn with uniformly
    random start indices on the circle [0, n).  The circular wrap-around
    ensures every observation is the start of exactly one block — a useful
    symmetry property missing from the non-overlapping block bootstrap.

    Parameters
    ----------
    x:
        1-D array of returns, length n.
    n_resamples:
        Number of bootstrap replications.
    block_length:
        Fixed block size.  Must satisfy 1 <= block_length <= n.
    seed:
        RNG seed.  Same contract as :func:`stationary_bootstrap`.

    Returns
    -------
    Array of shape ``(n_resamples, len(x))``.
    """
    if not 1 <= block_length <= len(x):
        msg = f"block_length must be in [1, {len(x)}], got {block_length}"
        raise ValueError(msg)

    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    rng = np.random.default_rng(seed)

    # How many blocks cover n observations?  Ceiling, then truncate the last
    # block if it would overrun n.
    n_blocks = math.ceil(n / block_length)
    out = np.empty((n_resamples, n), dtype=np.float64)

    # Double the array to avoid modular indexing inside the inner loop.
    x2 = np.concatenate([x, x])

    for i in range(n_resamples):
        starts = rng.integers(0, n, size=n_blocks)
        row = np.concatenate([x2[s : s + block_length] for s in starts])
        out[i] = row[:n]

    return out


# ---------------------------------------------------------------------------
# Confidence interval computation
# ---------------------------------------------------------------------------


def _bca_acceleration(
    x: npt.NDArray[np.float64],
    stat_fn: Callable[[npt.NDArray[np.float64]], float],
) -> float:
    """Jackknife estimate of the BCa acceleration constant.

    The acceleration ``a`` measures how quickly the standard error of the
    statistic changes with the parameter value.  It is estimated via the
    jackknife influence values (Efron & Tibshirani 1993, §14.3).

    A non-zero acceleration corrects the BCa endpoints for skewness in the
    sampling distribution of the statistic; without it BCa degenerates to
    the bias-corrected (BC) interval.
    """
    n = len(x)
    theta_jack = np.array([stat_fn(np.delete(x, i)) for i in range(n)], dtype=np.float64)
    # Influence values (centred jackknife pseudo-values, sign convention from
    # Efron & Tibshirani §14.3, eq. 14.14).
    u = theta_jack.mean() - theta_jack
    denom = float((u**2).sum()) ** 1.5
    if denom == 0.0:
        return 0.0
    return float((u**3).sum()) / (6.0 * denom)


def _normal_ppf(p: float) -> float:
    """Inverse standard normal CDF via ``math.erf``.

    Implemented from scratch to avoid scipy (see BACKTEST-CONTRACTS.md §7).
    Uses the identity Φ(x) = (1 + erf(x/√2)) / 2, inverted by a Newton step
    after an initial rational approximation from Abramowitz & Stegun 26.2.17.

    Accurate to better than 1e-10 for p in (1e-15, 1-1e-15).
    """
    if p <= 0.0 or p >= 1.0:
        msg = f"p must be in (0, 1), got {p}"
        raise ValueError(msg)

    # Rational approximation for the initial estimate (A&S 26.2.17).
    # The formula approximates Φ^{-1}(p) for p in (0, 0.5] by computing a
    # positive value from the left-tail probability.  For p > 0.5 we reflect:
    # Φ^{-1}(p) = -Φ^{-1}(1-p), so we work on q = 1-p and negate at the end.
    sign = 1.0
    q = p
    if p > 0.5:
        # Right tail: formula gives the magnitude; result is positive.
        q = 1.0 - p
    else:
        # Left tail: formula gives the magnitude; result is negative.
        sign = -1.0

    t = math.sqrt(-2.0 * math.log(q))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    x_approx = t - (c0 + c1 * t + c2 * t**2) / (1.0 + d1 * t + d2 * t**2 + d3 * t**3)
    # x_approx is positive; apply sign to get the correct tail.
    x = sign * x_approx

    # Two Newton-Raphson steps to refine: x_{n+1} = x_n - (Φ(x_n) - p) / φ(x_n).
    # Guard fpx against underflow at extreme x (|x| > ~37 causes exp underflow).
    for _ in range(2):
        fx = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))) - p
        raw_fpx = -0.5 * x**2
        if raw_fpx < -700:
            break  # underflow: initial estimate is already machine-accurate
        fpx = math.exp(raw_fpx) / math.sqrt(2.0 * math.pi)
        if fpx == 0.0:
            break
        x -= fx / fpx

    return x


def bootstrap_ci(
    x: npt.NDArray[np.float64],
    stat_fn: Callable[[npt.NDArray[np.float64]], float],
    *,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    method: str = "bca",
    bootstrap_kind: str = "stationary",
    block_length: float | None = None,
    seed: int,
) -> BootstrapCI:
    """Bootstrap confidence interval for an arbitrary statistic.

    Parameters
    ----------
    x:
        1-D array of returns (or any real series).
    stat_fn:
        Callable that accepts a 1-D array and returns a scalar.  Common
        choices: ``np.mean``, a Sharpe estimator, profit factor, expectancy.
    n_resamples:
        Bootstrap replications.  2000 is adequate for 95% CIs; use 5000 for
        99% or for BCa with large acceleration.
    confidence_level:
        Nominal coverage, e.g. 0.95.
    method:
        ``"percentile"`` — naive percentile of the bootstrap distribution.
        Faster but biased when the sampling distribution is asymmetric.

        ``"bca"`` — bias-corrected and accelerated (Efron 1987).  Adjusts
        for both bias and skewness in the bootstrap distribution.  Preferred
        for small samples and non-Gaussian statistics like Sharpe ratio.
    bootstrap_kind:
        ``"stationary"`` (default) or ``"circular"``.
    block_length:
        Mean block length.  ``None`` triggers :func:`select_block_length`,
        whose output is then stored in the returned ``BootstrapCI`` so the
        experiment registry can log it.
    seed:
        RNG seed; required, no default.

    Returns
    -------
    :class:`BootstrapCI` with lower/upper endpoints and full provenance.
    """
    if not 0.0 < confidence_level < 1.0:
        msg = f"confidence_level must be in (0, 1), got {confidence_level}"
        raise ValueError(msg)
    if method not in ("percentile", "bca"):
        msg = f"method must be 'percentile' or 'bca', got {method!r}"
        raise ValueError(msg)
    if bootstrap_kind not in ("stationary", "circular"):
        msg = f"bootstrap_kind must be 'stationary' or 'circular', got {bootstrap_kind!r}"
        raise ValueError(msg)

    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    if block_length is None:
        block_length = select_block_length(x)

    # Generate resamples.
    if bootstrap_kind == "stationary":
        resamples = stationary_bootstrap(
            x, n_resamples, block_length=block_length, seed=seed
        )
    else:
        bl_int = max(1, min(round(block_length), n))
        resamples = circular_block_bootstrap(x, n_resamples, block_length=bl_int, seed=seed)

    # Evaluate the statistic on each resample.
    boot_stats = np.array([stat_fn(resamples[i]) for i in range(n_resamples)])

    theta_hat = stat_fn(x)
    alpha = 1.0 - confidence_level

    if method == "percentile":
        lower = float(np.quantile(boot_stats, alpha / 2.0))
        upper = float(np.quantile(boot_stats, 1.0 - alpha / 2.0))

    else:  # bca
        # Bias-correction z0: proportion of bootstrap replicates below the
        # observed statistic, mapped through the normal quantile.
        prop_below = float(np.mean(boot_stats < theta_hat))
        # Guard the edges so _normal_ppf doesn't blow up on degenerate series.
        prop_below = max(1e-10, min(1.0 - 1e-10, prop_below))
        z0 = _normal_ppf(prop_below)

        acc = _bca_acceleration(x, stat_fn)

        # BCa adjusted quantile levels (Efron 1987, eq. 22.36).
        z_alpha_lo = _normal_ppf(alpha / 2.0)
        z_alpha_hi = _normal_ppf(1.0 - alpha / 2.0)

        def _adj_q(z_alpha: float) -> float:
            numer = z0 + z_alpha
            denom = 1.0 - acc * (z0 + z_alpha)
            arg = z0 + numer / denom
            # Φ(arg) — standard normal CDF via erf.
            return 0.5 * (1.0 + math.erf(arg / math.sqrt(2.0)))

        q_lo = _adj_q(z_alpha_lo)
        q_hi = _adj_q(z_alpha_hi)

        lower = float(np.quantile(boot_stats, max(0.0, min(1.0, q_lo))))
        upper = float(np.quantile(boot_stats, max(0.0, min(1.0, q_hi))))

    return BootstrapCI(
        lower=lower,
        upper=upper,
        point_estimate=theta_hat,
        method=method,
        bootstrap_kind=bootstrap_kind,
        block_length=float(block_length),
        n_resamples=n_resamples,
        confidence_level=confidence_level,
    )


# ---------------------------------------------------------------------------
# Convenience statistics
# ---------------------------------------------------------------------------


def sharpe(
    returns: npt.NDArray[np.float64],
    *,
    periods_per_year: float = 252.0,
) -> float:
    """Annualised Sharpe ratio.  Returns 0.0 for a constant series.

    The denominator is the sample standard deviation, not the population
    stddev — consistent with how virtually every practitioner computes Sharpe
    from a finite return series.  Using the population std would overstate
    the ratio for small samples.
    """
    std = float(np.std(returns, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(returns)) / std * math.sqrt(periods_per_year)


def expectancy(returns: npt.NDArray[np.float64]) -> float:
    """Mean return.  A thin wrapper so ``stat_fn`` signatures are consistent."""
    return float(np.mean(returns))


def profit_factor(returns: npt.NDArray[np.float64]) -> float:
    """Gross profit divided by gross loss.

    Returns ``inf`` when there are no losing trades (every trade profitable),
    and ``0.0`` when there are no winning trades.  Both are meaningful and
    must not be silently converted to NaN.
    """
    wins = returns[returns > 0.0]
    losses = returns[returns < 0.0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    if gross_loss == 0.0:
        return math.inf
    return gross_profit / gross_loss
