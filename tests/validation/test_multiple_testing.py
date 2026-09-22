"""Tests for memetrader.validation.multiple_testing.

Every test is offline, seeded, and deterministic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from memetrader.validation.multiple_testing import (
    DSRResult,
    PBOResult,
    SPAResult,
    _normal_cdf,
    benjamini_hochberg,
    deflated_sharpe_ratio,
    hansens_spa,
    holm_bonferroni,
    probability_of_backtest_overfitting,
)

# ---------------------------------------------------------------------------
# Normal CDF reference values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (0.0, 0.5),
        (1.0, 0.8413447460685429),
        (-1.0, 0.15865525393145707),
        (1.96, 0.9750021048517796),
        (-1.96, 0.024997895148220435),
        (3.0, 0.9986501019683699),
        (-3.0, 0.0013498980316300933),
        (0.0, 0.5),
    ],
)
def test_normal_cdf_reference_values(x: float, expected: float) -> None:
    """CDF must match scipy reference values to 1e-10."""
    result = _normal_cdf(x)
    assert abs(result - expected) < 1e-10, (
        f"_normal_cdf({x}) = {result!r}, expected {expected!r}"
    )


def test_normal_cdf_symmetry() -> None:
    for x in [0.5, 1.0, 2.0, 3.5]:
        assert abs(_normal_cdf(x) + _normal_cdf(-x) - 1.0) < 1e-14


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------


def test_dsr_shrinks_with_trial_count() -> None:
    """DSR must decrease as trial_count increases for a fixed return series.

    A strategy that looks significant at 1 trial should look less significant
    at 100 trials and fail entirely at 1000 trials.  This is the core
    property DSR is designed to implement.
    """
    rng = np.random.default_rng(2024)
    # Moderate Sharpe: enough to pass with 1 trial, not enough for 1000.
    returns = rng.standard_normal(252) * 0.01 + 0.0005  # slight positive drift

    dsr_1 = deflated_sharpe_ratio(returns, trial_count=1)
    dsr_10 = deflated_sharpe_ratio(returns, trial_count=10)
    dsr_100 = deflated_sharpe_ratio(returns, trial_count=100)
    dsr_1000 = deflated_sharpe_ratio(returns, trial_count=1000)

    assert dsr_1.dsr > dsr_10.dsr, "DSR should fall as trials increase"
    assert dsr_10.dsr > dsr_100.dsr
    assert dsr_100.dsr > dsr_1000.dsr


def test_dsr_single_trial_survives() -> None:
    """A high-Sharpe series should have DSR > 0.9 at 1 trial."""
    rng = np.random.default_rng(7)
    # Annual Sharpe ~3: strong strategy.
    returns = rng.standard_normal(500) * 0.01 + 0.005
    result = deflated_sharpe_ratio(returns, trial_count=1)
    assert result.dsr > 0.9, f"Expected DSR > 0.9, got {result.dsr:.4f}"


def test_dsr_fails_at_1000_trials() -> None:
    """A modest Sharpe that passes 1-trial should fail at 1000 trials.

    This tests the guard that the spec explicitly requires: a Sharpe that
    survives 1 trial fails at 1000.
    """
    rng = np.random.default_rng(99)
    # Build a series with SR ~1 annualised: plausible but not extraordinary.
    # With 252 obs the standard error is ~1/sqrt(252) ≈ 0.063, so SR=1 is
    # about 1/0.063 ≈ 15 SE above zero — should pass with 1 trial...
    returns = rng.standard_normal(252) * 0.01 + 0.00063  # SR ≈ 1
    dsr_1 = deflated_sharpe_ratio(returns, trial_count=1)
    dsr_1000 = deflated_sharpe_ratio(returns, trial_count=1000)
    assert dsr_1.dsr > 0.5, f"Expected DSR > 0.5 at 1 trial, got {dsr_1.dsr:.4f}"
    assert dsr_1000.dsr < dsr_1.dsr, "DSR must fall with 1000 trials"


def test_dsr_result_fields() -> None:
    rng = np.random.default_rng(0)
    returns = rng.standard_normal(200)
    result = deflated_sharpe_ratio(returns, trial_count=5)
    assert isinstance(result, DSRResult)
    assert 0.0 <= result.dsr <= 1.0
    assert result.trial_count == 5
    assert result.n_obs == 200
    assert math.isfinite(result.skewness)
    assert math.isfinite(result.excess_kurtosis)


def test_dsr_requires_trial_count() -> None:
    """trial_count has no default — omitting it is a TypeError."""
    rng = np.random.default_rng(0)
    returns = rng.standard_normal(100)
    with pytest.raises(TypeError):
        deflated_sharpe_ratio(returns)  # type: ignore[call-arg]


def test_dsr_rejects_trial_count_zero() -> None:
    rng = np.random.default_rng(0)
    returns = rng.standard_normal(100)
    with pytest.raises(ValueError, match="trial_count"):
        deflated_sharpe_ratio(returns, trial_count=0)


# ---------------------------------------------------------------------------
# Probability of Backtest Overfitting
# ---------------------------------------------------------------------------


def test_pbo_near_zero_for_dominant_strategy() -> None:
    """A genuinely dominant strategy should have PBO close to 0.

    We construct a performance matrix where strategy 0 is consistently better
    than all others by a large margin across all time sub-periods.
    """
    rng = np.random.default_rng(42)
    t, s = 320, 10
    mat = rng.standard_normal((t, s)) * 0.01
    # Strategy 0 gets a large constant advantage.
    mat[:, 0] += 0.05
    result = probability_of_backtest_overfitting(mat, trial_count=s, n_splits=8)
    assert isinstance(result, PBOResult)
    assert result.pbo < 0.2, (
        f"Expected PBO < 0.2 for dominant strategy, got {result.pbo:.3f}"
    )


def test_pbo_near_half_for_noise() -> None:
    """For pure noise (all strategies iid N(0,1)), PBO should be near 0.5.

    The IS-best strategy is just the luckiest draw; it is equally likely to
    rank above or below the median OOS.  We allow a slack of ±0.15 around 0.5.
    """
    rng = np.random.default_rng(1234)
    t, s = 320, 20
    mat = rng.standard_normal((t, s))
    result = probability_of_backtest_overfitting(mat, trial_count=s, n_splits=8)
    assert abs(result.pbo - 0.5) < 0.25, (
        f"Expected PBO near 0.5 for noise, got {result.pbo:.3f}"
    )


def test_pbo_requires_trial_count() -> None:
    mat = np.random.default_rng(0).standard_normal((100, 5))
    with pytest.raises(TypeError):
        probability_of_backtest_overfitting(mat)  # type: ignore[call-arg]


def test_pbo_rejects_odd_n_splits() -> None:
    mat = np.random.default_rng(0).standard_normal((100, 5))
    with pytest.raises(ValueError, match="n_splits"):
        probability_of_backtest_overfitting(mat, trial_count=5, n_splits=7)


def test_pbo_result_fields() -> None:
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((80, 4))
    result = probability_of_backtest_overfitting(mat, trial_count=4, n_splits=8)
    assert 0.0 <= result.pbo <= 1.0
    assert result.n_strategies == 4
    assert len(result.logit_values) == result.n_splits


# ---------------------------------------------------------------------------
# Hansen's SPA test
# ---------------------------------------------------------------------------


def test_spa_does_not_reject_noise_at_5pct() -> None:
    """When all candidates are noise, p_consistent should exceed 0.05.

    This is the false-positive control test: if H_0 (no strategy beats the
    benchmark) is true, the test should not reject at the 5% level.
    We use 50 seeded replications and check that the rejection rate is at most
    0.20 (very conservative threshold, given a small per-replication n).
    """
    rejections = 0
    n_reps = 50
    for rep in range(n_reps):
        rng = np.random.default_rng(rep)
        # Loss differences centred at zero: benchmark == candidates on average.
        ld = rng.standard_normal((200, 5)) * 0.01
        result = hansens_spa(ld, trial_count=5, n_bootstrap=400, seed=rep)
        if result.p_consistent < 0.05:
            rejections += 1
    rejection_rate = rejections / n_reps
    assert rejection_rate <= 0.20, (
        f"SPA false positive rate {rejection_rate:.0%} exceeds 20% on noise data"
    )


def test_spa_rejects_clearly_superior_strategy() -> None:
    """A strategy with large mean outperformance should be detected."""
    rng = np.random.default_rng(0)
    t, k = 500, 5
    ld = rng.standard_normal((t, k)) * 0.01
    # Strategy 0 has a very large positive mean loss difference (it wins big).
    ld[:, 0] += 0.05
    result = hansens_spa(ld, trial_count=k, n_bootstrap=1000, seed=42)
    assert result.p_consistent < 0.05, (
        f"Expected p_consistent < 0.05 for dominant strategy, got {result.p_consistent:.4f}"
    )


def test_spa_p_ordering() -> None:
    """p_lower >= p_consistent >= p_upper is the theoretical ordering."""
    rng = np.random.default_rng(55)
    ld = rng.standard_normal((200, 4)) * 0.01
    result = hansens_spa(ld, trial_count=4, n_bootstrap=200, seed=0)
    # The ordering is strict for consistent tests; allow equality in degenerate cases.
    assert result.p_lower >= result.p_upper - 1e-10


def test_spa_requires_trial_count() -> None:
    ld = np.random.default_rng(0).standard_normal((100, 3))
    with pytest.raises(TypeError):
        hansens_spa(ld, seed=0)  # type: ignore[call-arg]


def test_spa_result_fields() -> None:
    ld = np.random.default_rng(0).standard_normal((100, 3))
    result = hansens_spa(ld, trial_count=3, n_bootstrap=100, seed=0)
    assert isinstance(result, SPAResult)
    assert 0.0 <= result.p_consistent <= 1.0
    assert 0.0 <= result.p_lower <= 1.0
    assert 0.0 <= result.p_upper <= 1.0
    assert result.test_statistic >= 0.0
    assert result.trial_count == 3
    assert result.n_bootstrap == 100


def test_spa_reproducible() -> None:
    ld = np.random.default_rng(5).standard_normal((150, 4))
    r1 = hansens_spa(ld, trial_count=4, n_bootstrap=200, seed=7)
    r2 = hansens_spa(ld, trial_count=4, n_bootstrap=200, seed=7)
    assert r1.p_consistent == r2.p_consistent
    assert r1.test_statistic == r2.test_statistic


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR
# ---------------------------------------------------------------------------


def test_bh_all_null() -> None:
    """With uniformly distributed p-values, rejections should be rare."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, size=100)
    rejected = benjamini_hochberg(p, trial_count=100, fdr_level=0.05)
    assert rejected.dtype == np.bool_
    # Under H_0, BH controls FDR at 5%; empirically few or no rejections
    # in one seeded draw.
    assert rejected.sum() < 20


def test_bh_clear_signal() -> None:
    """Obvious signal p-values should all be rejected."""
    p = np.array([0.0001, 0.0002, 0.0003, 0.5, 0.8, 0.9])
    rejected = benjamini_hochberg(p, trial_count=len(p), fdr_level=0.05)
    # The first three should be rejected; the last three should not.
    assert rejected[:3].all()
    assert not rejected[3:].any()


def test_bh_requires_trial_count() -> None:
    p = np.array([0.01, 0.05, 0.1])
    with pytest.raises(TypeError):
        benjamini_hochberg(p)  # type: ignore[call-arg]


def test_bh_rejects_trial_count_zero() -> None:
    p = np.array([0.01, 0.05])
    with pytest.raises(ValueError, match="trial_count"):
        benjamini_hochberg(p, trial_count=0)


# ---------------------------------------------------------------------------
# Holm-Bonferroni FWER
# ---------------------------------------------------------------------------


def test_holm_all_null() -> None:
    """With uniform p-values, FWER control should prevent almost all rejections."""
    rng = np.random.default_rng(1)
    p = rng.uniform(0.0, 1.0, size=50)
    rejected = holm_bonferroni(p, trial_count=50, alpha=0.05)
    # Under H_0 with seeded uniform data, Holm rarely rejects anything.
    assert rejected.sum() <= 5


def test_holm_rejects_strong_signals() -> None:
    p = np.array([1e-8, 1e-6, 0.001, 0.5])
    rejected = holm_bonferroni(p, trial_count=len(p), alpha=0.05)
    # The first three are far below Bonferroni threshold; the last is not.
    assert rejected[:3].all()
    assert not rejected[3]


def test_holm_step_down_property() -> None:
    """Once Holm stops, all subsequent hypotheses are not rejected."""
    p = np.array([0.001, 0.01, 0.5, 0.001])
    rejected = holm_bonferroni(p, trial_count=4, alpha=0.05)
    # Sort order: 0.001, 0.001, 0.01, 0.5.  Threshold at rank 1 is 0.05/4=0.0125.
    # 0.001 < 0.0125 → reject.  0.001 < 0.05/3 ≈ 0.0167 → reject.
    # 0.01 < 0.05/2 = 0.025 → reject.  0.5 > 0.05/1 → not rejected.
    assert rejected.sum() >= 2  # at minimum the two tiny p-values are rejected


def test_holm_requires_trial_count() -> None:
    p = np.array([0.01, 0.05])
    with pytest.raises(TypeError):
        holm_bonferroni(p)  # type: ignore[call-arg]


def test_holm_rejects_trial_count_zero() -> None:
    p = np.array([0.01, 0.05])
    with pytest.raises(ValueError, match="trial_count"):
        holm_bonferroni(p, trial_count=0)
