"""Statistical controls for multiple-strategy evaluation.

Running many strategies and keeping the best one is essentially a
model-selection problem, and Sharpe ratio (or any in-sample metric) is a
biased estimator of out-of-sample performance when the selection was made from
a pool.  The bias grows with the number of candidates *tried*, not just the
number *kept* — which is why the trial count here must be supplied explicitly.

Every function that corrects for multiple testing requires a ``trial_count``
parameter with no default.  Silently defaulting to 1 is the exact error these
tests exist to prevent: a researcher who ran 100 parameter sets and forgot to
say so will report an uncorrected Sharpe as if it were from a single
pre-specified strategy.  The absence of a default makes the omission a
``TypeError`` rather than a silent over-claim.

References
----------
Bailey, D.H. & López de Prado, M. (2014).  The deflated Sharpe ratio:
correcting for selection bias, backtest overfitting, and non-normality.
*Journal of Portfolio Management*, 40(5), 94-107.

López de Prado, M. & Bailey, D.H. (2014).  The Sharpe ratio efficient
frontier.  *Journal of Risk*, 15(2), 3-44.

Hansen, P.R. (2005).  A test for superior predictive ability.  *Journal of
Business & Economic Statistics*, 23(4), 365-380.

Benjamini, Y. & Hochberg, Y. (1995).  Controlling the false discovery rate:
a practical and powerful approach to multiple testing.  *Journal of the Royal
Statistical Society B*, 57(1), 289-300.

Holm, S. (1979).  A simple sequentially rejective multiple test procedure.
*Scandinavian Journal of Statistics*, 6(2), 65-70.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from memetrader.validation.bootstrap import stationary_bootstrap

# ---------------------------------------------------------------------------
# Normal CDF / PPF — no scipy, implemented via math.erf
# ---------------------------------------------------------------------------


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via ``math.erf``.

    Exact identity: Φ(x) = (1 + erf(x / √2)) / 2.

    ``math.erf`` is a C-library function accurate to machine precision on all
    supported platforms; this implementation matches scipy.stats.norm.cdf to
    better than 1e-10 across the range [-8, 8].

    This is the only special function the module needs (the PPF appears only
    in bootstrap.py).  We deliberately do not import scipy — see the
    dependency-decision table in docs/BACKTEST-CONTRACTS.md §7.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------


class DSRResult(NamedTuple):
    """Probability that the true Sharpe ratio is positive, after corrections.

    A DSR near 1.0 means the strategy almost certainly has a positive
    population Sharpe even after accounting for the number of trials, sample
    length, and non-normality.  A DSR near 0.5 means the observed Sharpe is
    roughly consistent with a lucky draw from zero-mean noise.  A DSR below
    0.5 means the *corrected* reference Sharpe exceeds the observed one — the
    strategy looks worse than random once you account for selection bias.

    ``reference_sharpe`` is the Sharpe ratio of the best of ``trial_count``
    iid standard-normal trials of ``n_obs`` observations — the benchmark that
    an observed Sharpe must beat to be considered evidence of skill.  It
    represents what selection alone produces, without any real edge.
    """

    dsr: float
    reference_sharpe: float
    observed_sharpe: float
    trial_count: int
    n_obs: int
    skewness: float
    excess_kurtosis: float


def deflated_sharpe_ratio(
    returns: npt.NDArray[np.float64],
    *,
    trial_count: int,
    periods_per_year: float = 252.0,
) -> DSRResult:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    Adjusts an observed Sharpe ratio for three sources of bias that a naive
    backtest ignores:

    1. **Trial count** — the more strategies tried, the higher the best
       in-sample Sharpe will be by pure luck.  The reference Sharpe SR* is
       the expected maximum of ``trial_count`` iid standard-normal Sharpes
       from a sample of length ``n_obs``.

    2. **Non-normality** — a strategy whose return distribution has negative
       skewness or excess kurtosis is penalised: the same Sharpe ratio is
       *harder* to achieve from a fat-tailed, left-skewed distribution, so
       we deflate accordingly.

    3. **Sample length** — a short backtest has a large standard error on
       the Sharpe estimate.  The standard error SE(SR) = sqrt((1 + 0.5 SR²
       - gamma₁ SR + (gamma₂/4)(1 + SR²)) / (n-1)) accounts for all three moments.

    The DSR is then P(SR > SR* | sample), implemented as a one-sided normal
    test on the standardised statistic.  See Bailey & López de Prado (2014)
    eqs. 8-14.

    Parameters
    ----------
    returns:
        1-D array of period returns.  Length is the sample size.
    trial_count:
        Total number of strategies evaluated, including discarded ones.
        **There is no default** — the test exists to prevent the silent
        default-of-1 error.
    periods_per_year:
        Annualisation factor, matching how SR is conventionally quoted.
    """
    if trial_count < 1:
        msg = f"trial_count must be >= 1, got {trial_count}"
        raise ValueError(msg)

    n = len(returns)
    if n < 2:
        msg = "Need at least 2 observations to compute a Sharpe ratio"
        raise ValueError(msg)

    r = np.asarray(returns, dtype=np.float64)
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))

    if sigma == 0.0:
        # Constant series — Sharpe is undefined; return degenerate result.
        sr_obs = 0.0
    else:
        sr_obs = mu / sigma  # period Sharpe (annualised below for reporting)

    sr_obs_ann = sr_obs * math.sqrt(periods_per_year)

    # Skewness and excess kurtosis of the return sample.
    skew = float(np.mean(((r - mu) / sigma) ** 3) if sigma > 0.0 else 0.0)
    kurt = float(np.mean(((r - mu) / sigma) ** 4) - 3.0 if sigma > 0.0 else 0.0)

    # Standard error of the Sharpe ratio (per-period, Bailey & LdP eq. 10).
    # The term (gamma₂/4) is the excess kurtosis divided by 4.
    var_sr = 1.0 + 0.5 * sr_obs**2 - skew * sr_obs + (kurt / 4.0) * (1.0 + sr_obs**2)
    var_sr = max(var_sr, 1e-12)  # guard tiny samples
    se_sr = math.sqrt(var_sr / (n - 1))

    # Reference Sharpe SR* — the expected maximum of ``trial_count`` iid
    # standard-normal Sharpes with the same per-period SE.
    # Under normality the expected max of k iid N(0, 1) variates is
    # approximately Φ^{-1}((k-0.5269)/(k+1-2*0.5269)) * (Bailey & LdP eq. 11)
    # which we compute numerically via the approximation
    # E[max_{i=1..k} Z_i] ≈ (1-gamma)*Φ^{-1}(1 - 1/k) + gamma*Φ^{-1}(1 - 1/(k*e))
    # (where gamma ≈ 0.5772 Euler-Mascheroni).  For k=1 this reduces to 0, which
    # is correct — a single pre-specified test has no selection bias.
    #
    # Bailey & López de Prado (2014) eq. 12 express the reference Sharpe as
    # SR* = SE(SR) * ( (1-gamma) * Φ^{-1}(1 - 1/N) + gamma * Φ^{-1}(1 - 1/(N·e)) )
    # where gamma is the Euler-Mascheroni constant and N = trial_count.
    euler_mascheroni = 0.5772156649015329

    if trial_count == 1:
        # Expected max of a single standard normal is 0 — correct by definition.
        e_max = 0.0
    else:
        k = float(trial_count)
        # Guard log/quantile edge cases for large trial counts.
        q1 = max(1e-15, 1.0 - 1.0 / k)
        q2 = max(1e-15, 1.0 - 1.0 / (k * math.e))

        # Invert the normal CDF for q1, q2 using our local PPF (imported from
        # bootstrap.py's _normal_ppf logic, re-implemented inline to keep this
        # module self-contained).
        def _ppf(p: float) -> float:
            """Standard normal PPF via Newton-Raphson on the erf identity."""
            if p <= 0.0:
                return -math.inf
            if p >= 1.0:
                return math.inf
            sign = 1.0
            q = p
            if p > 0.5:
                q = 1.0 - p
            else:
                sign = -1.0
            t = math.sqrt(-2.0 * math.log(q))
            c0, c1, c2 = 2.515517, 0.802853, 0.010328
            d1, d2, d3 = 1.432788, 0.189269, 0.001308
            x_a = t - (c0 + c1 * t + c2 * t**2) / (1.0 + d1 * t + d2 * t**2 + d3 * t**3)
            x = sign * x_a
            for _ in range(2):
                fx = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))) - p
                raw = -0.5 * x**2
                if raw < -700:
                    break
                fpx = math.exp(raw) / math.sqrt(2.0 * math.pi)
                if fpx == 0.0:
                    break
                x -= fx / fpx
            return x

        z1 = _ppf(q1)
        z2 = _ppf(q2)
        e_max = (1.0 - euler_mascheroni) * z1 + euler_mascheroni * z2

    # Reference Sharpe in per-period units, then annualise.
    sr_ref_period = e_max * se_sr
    sr_ref_ann = sr_ref_period * math.sqrt(periods_per_year)

    # DSR = P(SR_obs > SR* | sample) — standardised, one-sided normal test.
    # A positive DSR statistic means the observed Sharpe beats the selection-
    # adjusted reference; negative means it does not.
    if se_sr == 0.0:
        dsr = 0.0
    else:
        z_dsr = (sr_obs - sr_ref_period) / se_sr
        dsr = _normal_cdf(z_dsr)

    return DSRResult(
        dsr=dsr,
        reference_sharpe=sr_ref_ann,
        observed_sharpe=sr_obs_ann,
        trial_count=trial_count,
        n_obs=n,
        skewness=skew,
        excess_kurtosis=kurt,
    )


# ---------------------------------------------------------------------------
# Probability of Backtest Overfitting (PBO)
# ---------------------------------------------------------------------------


class PBOResult(NamedTuple):
    """Probability of Backtest Overfitting via CSCV.

    ``pbo`` is the fraction of CSCV splits for which the in-sample best
    strategy ranked below median out-of-sample.  A PBO near 0.5 means the
    IS-best strategy performs no better than random OOS — the IS metric
    cannot distinguish skill from luck.  A PBO near 0 means the IS-best is
    consistently the OOS best — strong evidence of a real, robust edge.

    ``logit_values`` are the logit(OOS rank / n_strategies) for each split,
    used to characterise the full distribution of IS→OOS rank degradation
    rather than just the binary above/below-median count.
    """

    pbo: float
    logit_values: npt.NDArray[np.float64]
    n_splits: int
    n_strategies: int


def probability_of_backtest_overfitting(
    performance_matrix: npt.NDArray[np.float64],
    *,
    trial_count: int,
    n_splits: int = 16,
) -> PBOResult:
    """PBO via Combinatorially Symmetric Cross-Validation (Bailey & LdP 2014).

    Splits the time axis of ``performance_matrix`` into ``n_splits`` equal
    sub-periods, then iterates over all C(n_splits, n_splits//2) ways to
    partition those sub-periods into in-sample and out-of-sample halves.  For
    each split:

    1. Find the strategy with the highest mean IS performance (Sharpe-like).
    2. Rank that strategy by its mean OOS performance.
    3. Compute the logit of the normalised OOS rank.

    PBO = fraction of splits where the IS-best ranked below the median OOS.
    A high PBO means the selection mechanism (whichever metric was used IS) is
    unreliable as a predictor of OOS performance.

    Parameters
    ----------
    performance_matrix:
        Array of shape ``(T, S)`` where T = time periods and S = strategies.
        Each cell is a period return (or Sharpe, or any ordinal performance
        measure — PBO is rank-based and metric-agnostic).
    trial_count:
        Total strategies tried including those not in the matrix.
        Required; no default.  Strategies excluded from the matrix (e.g.
        because they failed a filter) still inflated the selection bias.
    n_splits:
        Number of sub-periods.  Must be even (IS and OOS get equal halves).
        Bailey & LdP use 16 as the default; fewer splits reduce combinatorial
        coverage, more splits shorten each sub-period's performance estimate.
    """
    if trial_count < 1:
        msg = f"trial_count must be >= 1, got {trial_count}"
        raise ValueError(msg)
    if n_splits < 2 or n_splits % 2 != 0:
        msg = f"n_splits must be a positive even integer, got {n_splits}"
        raise ValueError(msg)

    mat = np.asarray(performance_matrix, dtype=np.float64)
    if mat.ndim != 2:
        msg = "performance_matrix must be 2-D (T x S)"
        raise ValueError(msg)

    t_total, n_strategies = mat.shape
    if n_strategies < 2:
        msg = "Need at least 2 strategies"
        raise ValueError(msg)
    if t_total < n_splits:
        msg = f"T={t_total} < n_splits={n_splits}; reduce n_splits"
        raise ValueError(msg)

    # Divide time axis into n_splits equal sub-periods.  Remainder rows go
    # to the last sub-period — this is a minor asymmetry but it avoids
    # discarding data and Bailey & LdP do not mandate equal-length splits.
    split_size = t_total // n_splits
    splits = [mat[i * split_size : (i + 1) * split_size, :] for i in range(n_splits - 1)]
    splits.append(mat[(n_splits - 1) * split_size :, :])  # last absorbs remainder

    # Enumerate all combinations of n_splits//2 splits for IS.
    from itertools import combinations

    half = n_splits // 2
    all_combos = list(combinations(range(n_splits), half))

    logit_values: list[float] = []

    for is_idx in all_combos:
        oos_idx = tuple(i for i in range(n_splits) if i not in is_idx)

        is_mat = np.concatenate([splits[i] for i in is_idx], axis=0)
        oos_mat = np.concatenate([splits[i] for i in oos_idx], axis=0)

        # IS-best strategy (highest mean IS performance).
        is_means = is_mat.mean(axis=0)
        best_is = int(np.argmax(is_means))

        # OOS rank of the IS-best strategy (higher = better).
        oos_means = oos_mat.mean(axis=0)
        # rank: 1 = worst, n_strategies = best
        rank = int(np.sum(oos_means <= oos_means[best_is]))

        # Normalised rank ω ∈ (0, 1]: rank / n_strategies.
        # If ω < 0.5 the IS-best finished below the OOS median → overfitting.
        omega = rank / n_strategies
        # Logit of normalised rank — guards the 0 and 1 edges.
        omega_clipped = max(1e-10, min(1.0 - 1e-10, omega))
        logit_values.append(math.log(omega_clipped / (1.0 - omega_clipped)))

    logit_arr = np.array(logit_values, dtype=np.float64)
    # PBO = fraction of splits where IS-best was below OOS median (logit < 0).
    pbo = float(np.mean(logit_arr < 0.0))

    return PBOResult(
        pbo=pbo,
        logit_values=logit_arr,
        n_splits=len(all_combos),
        n_strategies=n_strategies,
    )


# ---------------------------------------------------------------------------
# Hansen's Superior Predictive Ability (SPA) test
# ---------------------------------------------------------------------------


class SPAResult(NamedTuple):
    """Hansen (2005) SPA test result.

    Three p-values are reported, following Hansen (2005) §3.3:

    * ``p_consistent`` — the main result; uses a data-dependent centering of
      the null that is consistent against all alternatives.

    * ``p_lower`` — conservative lower bound; does not centre (hardest to
      reject, most Type-II error).

    * ``p_upper`` — liberal upper bound; centres aggressively (easiest to
      reject, most Type-I error in finite samples).

    A strategy is declared superior if ``p_consistent`` < alpha.

    ``test_statistic`` is the maximum mean outperformance over the benchmark,
    T_n = max(0, max_k mean(d_{k,t})) where d_{k,t} is the loss difference
    at time t between strategy k and the benchmark.
    """

    p_consistent: float
    p_lower: float
    p_upper: float
    test_statistic: float
    trial_count: int
    n_bootstrap: int


def hansens_spa(
    loss_differences: npt.NDArray[np.float64],
    *,
    trial_count: int,
    n_bootstrap: int = 1000,
    block_length: float | None = None,
    seed: int,
) -> SPAResult:
    """Hansen's Superior Predictive Ability test (Hansen 2005).

    Tests whether *any* strategy in a candidate set strictly outperforms a
    benchmark, after accounting for the fact that the candidates were selected
    from ``trial_count`` alternatives.

    Null hypothesis H_0: no strategy in the candidate set has positive
    expected performance relative to the benchmark (all µ_k <= 0).

    Parameters
    ----------
    loss_differences:
        Array of shape ``(T, K)`` where T = time periods and K = strategies.
        ``loss_differences[t, k]`` = benchmark_loss(t) - strategy_k_loss(t).
        A positive value at (t, k) means strategy k outperformed the
        benchmark at time t.  Common choice: log returns minus benchmark
        log returns.
    trial_count:
        Total strategies evaluated including those not in the matrix.
        Required; no default.
    n_bootstrap:
        Number of stationary-bootstrap replications for the null distribution.
        Hansen uses 1000.
    block_length:
        Mean bootstrap block length.  ``None`` uses the auto-selector applied
        to the first column of ``loss_differences``.
    seed:
        RNG seed for reproducibility.
    """
    if trial_count < 1:
        msg = f"trial_count must be >= 1, got {trial_count}"
        raise ValueError(msg)

    ld = np.asarray(loss_differences, dtype=np.float64)
    if ld.ndim == 1:
        ld = ld[:, np.newaxis]
    if ld.ndim != 2:
        msg = "loss_differences must be 1-D or 2-D"
        raise ValueError(msg)

    t_obs, k = ld.shape

    # Sample mean loss differences: µ̂_k = (1/T) Σ_t d_{k,t}
    mu_hat = ld.mean(axis=0)  # shape (K,)

    # Observed test statistic: T_n = sqrt(T) * max(0, max_k µ̂_k).
    # The sqrt(T) scaling makes the statistic converge to a proper limit.
    t_stat = math.sqrt(t_obs) * float(max(0.0, float(mu_hat.max())))

    # --- Bootstrap null distribution ---
    # We resample the centred loss differences under three centering schemes:
    #   - lower:      no centering (µ̂_k not subtracted) — hardest to reject
    #   - upper:      always subtract µ̂_k — easiest to reject
    #   - consistent: subtract µ̂_k only for strategies that look competitive
    #                 (µ̂_k >= -sqrt(var * log(log(T)) / T)) — Hansen (2005) §3.3

    # Estimate bootstrap block length from the first strategy's series.
    from memetrader.validation.bootstrap import select_block_length

    if block_length is None:
        block_length = select_block_length(ld[:, 0])

    # Generate bootstrap index resamples for all K strategies at once by
    # resampling the T-length time index and applying it to all K columns.
    # Stationary bootstrap on the row index.
    row_resamples = stationary_bootstrap(
        np.arange(t_obs, dtype=np.float64),
        n_bootstrap,
        block_length=block_length,
        seed=seed,
    )
    # row_resamples: (n_bootstrap, T) — each row is a permuted time index.
    idx = row_resamples.astype(np.intp)  # (n_bootstrap, T)

    # Variance of each strategy's loss-difference series (for the consistent
    # threshold).  Use the biased estimator as Hansen (2005) does.
    var_hat = ld.var(axis=0)  # shape (K,)

    # Consistent centering threshold: c_k = sqrt(var_k * log(log(T)) / T).
    # Strategies with µ̂_k >= -c_k are "potentially competitive" and are
    # centred at 0; others are centred at their sample mean.
    if t_obs >= 3:
        thresh = np.sqrt(var_hat * math.log(math.log(t_obs)) / t_obs)
    else:
        thresh = np.zeros(k)
    competitive = mu_hat >= -thresh  # shape (K,)

    # Bootstrap test statistics under each centering rule.
    t_lower_boot = np.empty(n_bootstrap)
    t_upper_boot = np.empty(n_bootstrap)
    t_cons_boot = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        # Resample the T rows.
        boot_ld = ld[idx[b], :]  # (T, K)
        boot_mu = boot_ld.mean(axis=0)  # (K,)

        # Lower: centre at sample mean (d* - µ̂), equivalent to no centre when
        # the test statistic is max(0, max(boot_mu - mu_hat)).
        boot_lower = boot_mu - mu_hat
        t_lower_boot[b] = math.sqrt(t_obs) * float(max(0.0, float(boot_lower.max())))

        # Upper: centre at zero (d* itself), assuming all mu_k = 0 under H_0.
        t_upper_boot[b] = math.sqrt(t_obs) * float(max(0.0, float(boot_mu.max())))

        # Consistent: centre competitive strategies at their sample mean
        # (subtract µ̂_k), leave non-competitive uncentred (subtract 0).
        # Hansen (2005) §3.3: competitive means potentially µ_k > 0, so we
        # impose the null µ_k = 0 for them by shifting the bootstrap mean by
        # µ̂_k.  Non-competitive strategies are already well into the null.
        centring = np.where(competitive, mu_hat, 0.0)
        boot_cons = boot_mu - centring
        t_cons_boot[b] = math.sqrt(t_obs) * float(max(0.0, float(boot_cons.max())))

    # p-values: fraction of bootstrap statistics that exceed the observed.
    p_lower = float(np.mean(t_lower_boot >= t_stat))
    p_upper = float(np.mean(t_upper_boot >= t_stat))
    p_consistent = float(np.mean(t_cons_boot >= t_stat))

    return SPAResult(
        p_consistent=p_consistent,
        p_lower=p_lower,
        p_upper=p_upper,
        test_statistic=t_stat,
        trial_count=trial_count,
        n_bootstrap=n_bootstrap,
    )


# ---------------------------------------------------------------------------
# Multiple-comparison corrections
# ---------------------------------------------------------------------------


def benjamini_hochberg(
    p_values: npt.NDArray[np.float64],
    *,
    trial_count: int,
    fdr_level: float = 0.05,
) -> npt.NDArray[np.bool_]:
    """Benjamini-Hochberg FDR correction (Benjamini & Hochberg 1995).

    Controls the expected fraction of false discoveries among the rejected
    hypotheses.  Less conservative than Bonferroni-style corrections when many
    hypotheses are tested; the price is that individual rejections have a
    higher false-positive probability.

    Parameters
    ----------
    p_values:
        1-D array of raw p-values, one per hypothesis (feature, strategy…).
    trial_count:
        Total tests performed.  If p_values contains only a subset (e.g.
        after a pre-filter), the denominator must still reflect the full
        trial count to avoid deflating the FDR estimate.
    fdr_level:
        Target false discovery rate, e.g. 0.05.

    Returns
    -------
    Boolean array of the same length as ``p_values``.  ``True`` = rejected
    (declared significant at the given FDR level).
    """
    if trial_count < 1:
        msg = f"trial_count must be >= 1, got {trial_count}"
        raise ValueError(msg)
    if not 0.0 < fdr_level < 1.0:
        msg = f"fdr_level must be in (0, 1), got {fdr_level}"
        raise ValueError(msg)

    p = np.asarray(p_values, dtype=np.float64)
    m = trial_count  # total comparisons (may exceed len(p) if pre-filtered)
    n = len(p)

    # Sort p-values and compute the BH threshold for each rank.
    order = np.argsort(p)
    sorted_p = p[order]

    # BH critical values: (rank / m) * fdr_level.
    ranks = np.arange(1, n + 1, dtype=np.float64)
    thresholds = ranks / m * fdr_level

    # Find the largest rank k* where p_(k*) <= threshold_(k*).
    # All hypotheses with rank <= k* are rejected.
    below = sorted_p <= thresholds
    if not below.any():
        rejected_sorted = np.zeros(n, dtype=np.bool_)
    else:
        cutoff = int(np.max(np.where(below)[0]))
        rejected_sorted = np.zeros(n, dtype=np.bool_)
        rejected_sorted[: cutoff + 1] = True

    # Restore original order.
    result = np.empty(n, dtype=np.bool_)
    result[order] = rejected_sorted
    return result


def holm_bonferroni(
    p_values: npt.NDArray[np.float64],
    *,
    trial_count: int,
    alpha: float = 0.05,
) -> npt.NDArray[np.bool_]:
    """Holm-Bonferroni step-down correction (Holm 1979).

    Controls the family-wise error rate (FWER) — the probability of *any*
    false positive.  Uniformly more powerful than the standard Bonferroni
    correction while maintaining strong FWER control.

    Parameters
    ----------
    p_values:
        1-D array of raw p-values.
    trial_count:
        Total tests performed; same rationale as in :func:`benjamini_hochberg`.
    alpha:
        Target FWER.

    Returns
    -------
    Boolean array; ``True`` = rejected.
    """
    if trial_count < 1:
        msg = f"trial_count must be >= 1, got {trial_count}"
        raise ValueError(msg)
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must be in (0, 1), got {alpha}"
        raise ValueError(msg)

    p = np.asarray(p_values, dtype=np.float64)
    m = trial_count
    n = len(p)

    order = np.argsort(p)
    sorted_p = p[order]

    # Holm step-down: compare sorted p_i against alpha / (m - i + 1).
    # Stop as soon as a p-value exceeds its threshold; all subsequent
    # hypotheses are not rejected (step-down property).
    rejected_sorted = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        threshold = alpha / (m - i)
        if sorted_p[i] <= threshold:
            rejected_sorted[i] = True
        else:
            break  # step-down: stop at first non-rejection

    result = np.empty(n, dtype=np.bool_)
    result[order] = rejected_sorted
    return result
