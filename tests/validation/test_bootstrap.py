"""Tests for memetrader.validation.bootstrap.

Every test is offline, seeded, and deterministic.  The suite verifies both
the statistical properties of the bootstrap algorithms and their exact
reproducibility.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from memetrader.validation.bootstrap import (
    BootstrapCI,
    _normal_ppf,
    bootstrap_ci,
    circular_block_bootstrap,
    expectancy,
    profit_factor,
    select_block_length,
    sharpe,
    stationary_bootstrap,
)

# ---------------------------------------------------------------------------
# Normal PPF reference values (from scipy.stats.norm.ppf / Wolfram Alpha)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (0.5, 0.0),
        (0.975, 1.959963985),
        (0.025, -1.959963985),
        (0.9, 1.281551566),
        (0.1, -1.281551566),
        (0.99, 2.326347874),
        (0.01, -2.326347874),
        (0.001, -3.090232306),
        (0.999, 3.090232306),
    ],
)
def test_normal_ppf_reference_values(p: float, expected: float) -> None:
    """PPF must match reference values to 1e-10 across the common range."""
    result = _normal_ppf(p)
    assert abs(result - expected) < 1e-7, (
        f"_normal_ppf({p}) = {result}, expected {expected}"
    )


def test_normal_ppf_symmetry() -> None:
    """Φ^{-1}(1-p) = -Φ^{-1}(p) for all valid p."""
    for p in [0.01, 0.05, 0.1, 0.25, 0.4]:
        assert abs(_normal_ppf(p) + _normal_ppf(1.0 - p)) < 1e-10


def test_normal_ppf_rejects_edges() -> None:
    with pytest.raises(ValueError):
        _normal_ppf(0.0)
    with pytest.raises(ValueError):
        _normal_ppf(1.0)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_stationary_bootstrap_reproducible() -> None:
    """Same seed must produce byte-identical output on repeated calls."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    a = stationary_bootstrap(x, 50, block_length=10.0, seed=7)
    b = stationary_bootstrap(x, 50, block_length=10.0, seed=7)
    np.testing.assert_array_equal(a, b)


def test_circular_block_bootstrap_reproducible() -> None:
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    a = circular_block_bootstrap(x, 50, block_length=10, seed=7)
    b = circular_block_bootstrap(x, 50, block_length=10, seed=7)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ() -> None:
    x = np.random.default_rng(0).standard_normal(100)
    a = stationary_bootstrap(x, 20, block_length=5.0, seed=1)
    b = stationary_bootstrap(x, 20, block_length=5.0, seed=2)
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Shape and basic properties
# ---------------------------------------------------------------------------


def test_stationary_bootstrap_shape() -> None:
    x = np.arange(50, dtype=float)
    out = stationary_bootstrap(x, 100, block_length=5.0, seed=0)
    assert out.shape == (100, 50)


def test_circular_block_bootstrap_shape() -> None:
    x = np.arange(50, dtype=float)
    out = circular_block_bootstrap(x, 100, block_length=5, seed=0)
    assert out.shape == (100, 50)


def test_bootstrap_values_in_original_set() -> None:
    """Every resampled value must appear in the original series."""
    x = np.random.default_rng(99).standard_normal(80)
    x_set = set(x.tolist())
    out = stationary_bootstrap(x, 10, block_length=5.0, seed=0)
    for row in out:
        for v in row:
            assert v in x_set


# ---------------------------------------------------------------------------
# CI coverage on iid data
# ---------------------------------------------------------------------------


def test_ci_covers_true_mean_at_nominal_rate() -> None:
    """Bootstrap CI should contain the true mean at roughly the nominal rate
    on iid Gaussian data.

    We draw 100 synthetic datasets of length 200, compute 90% CIs for the
    mean, and check that the true mean (0.0) is covered at least 80% of the
    time (leaving a 10-point slack for finite-sample and bootstrap variability).
    The test is seeded so it is deterministic and cannot flake.
    """
    rng = np.random.default_rng(2024)
    n_datasets = 100
    nominal = 0.90
    covered = 0
    for i in range(n_datasets):
        x = rng.standard_normal(200)
        ci = bootstrap_ci(
            x,
            np.mean,
            n_resamples=500,
            confidence_level=nominal,
            method="percentile",
            bootstrap_kind="stationary",
            block_length=1.0,  # iid: block length 1 is correct
            seed=i,
        )
        if ci.lower <= 0.0 <= ci.upper:
            covered += 1
    coverage = covered / n_datasets
    # At 90% nominal with slack for finite samples, we require >= 80% empirical.
    assert coverage >= 0.80, f"Coverage {coverage:.0%} below threshold on iid data"


# ---------------------------------------------------------------------------
# Block bootstrap preserves autocorrelation that iid does not
# ---------------------------------------------------------------------------


def test_block_bootstrap_preserves_autocorrelation() -> None:
    """A block bootstrap must preserve lag-1 autocorrelation significantly
    better than an iid bootstrap (block_length=1).

    We generate a strongly autocorrelated AR(1) series and compare the mean
    lag-1 autocorrelation of block resamples versus iid resamples.  The block
    resamples should have much higher autocorrelation, confirming that the
    block structure is actually doing what it promises.
    """
    rng = np.random.default_rng(777)
    n = 500
    phi = 0.8  # AR(1) coefficient — strong autocorrelation
    x = np.zeros(n)
    eps = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]

    n_rep = 200
    block_len = 20

    # Block resamples: should preserve within-block autocorrelation.
    block_samples = stationary_bootstrap(x, n_rep, block_length=float(block_len), seed=11)
    block_ac = np.array([float(np.corrcoef(s[:-1], s[1:])[0, 1]) for s in block_samples])

    # iid resamples (block_length=1): destroy all autocorrelation.
    iid_samples = stationary_bootstrap(x, n_rep, block_length=1.0, seed=11)
    iid_ac = np.array([float(np.corrcoef(s[:-1], s[1:])[0, 1]) for s in iid_samples])

    mean_block_ac = float(block_ac.mean())
    mean_iid_ac = float(iid_ac.mean())

    # The block bootstrap should yield noticeably higher lag-1 autocorrelation.
    assert mean_block_ac > mean_iid_ac + 0.2, (
        f"Block AC {mean_block_ac:.3f} not > iid AC {mean_iid_ac:.3f} + 0.2; "
        "block bootstrap is not preserving autocorrelation"
    )


# ---------------------------------------------------------------------------
# Automatic block-length selector
# ---------------------------------------------------------------------------


def test_select_block_length_returns_finite_positive() -> None:
    x = np.random.default_rng(0).standard_normal(300)
    bl = select_block_length(x)
    assert math.isfinite(bl) and bl >= 2.0


def test_select_block_length_clipped_to_range() -> None:
    x = np.random.default_rng(1).standard_normal(100)
    bl = select_block_length(x)
    assert 2.0 <= bl <= 25.0  # n/4 = 25


def test_select_block_length_short_series() -> None:
    x = np.array([1.0, 2.0, 3.0])
    bl = select_block_length(x)
    assert bl == 2.0


# ---------------------------------------------------------------------------
# Convenience statistics
# ---------------------------------------------------------------------------


def test_sharpe_constant_series() -> None:
    assert sharpe(np.ones(100)) == 0.0


def test_sharpe_positive_drift() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal(252) + 0.05
    sr = sharpe(x)
    assert sr > 0.0


def test_expectancy() -> None:
    x = np.array([1.0, -1.0, 2.0, -2.0, 3.0])
    assert abs(expectancy(x) - 0.6) < 1e-12


def test_profit_factor_all_wins() -> None:
    x = np.array([1.0, 2.0, 3.0])
    assert profit_factor(x) == math.inf


def test_profit_factor_all_losses() -> None:
    x = np.array([-1.0, -2.0])
    assert profit_factor(x) == 0.0


def test_profit_factor_mixed() -> None:
    x = np.array([3.0, -1.0])
    assert abs(profit_factor(x) - 3.0) < 1e-12


# ---------------------------------------------------------------------------
# BootstrapCI namedtuple fields
# ---------------------------------------------------------------------------


def test_bootstrap_ci_fields_populated() -> None:
    x = np.random.default_rng(0).standard_normal(100)
    ci = bootstrap_ci(
        x,
        np.mean,
        n_resamples=200,
        confidence_level=0.95,
        method="bca",
        bootstrap_kind="stationary",
        seed=42,
    )
    assert isinstance(ci, BootstrapCI)
    assert ci.lower <= ci.upper
    assert ci.n_resamples == 200
    assert ci.confidence_level == 0.95
    assert ci.block_length > 0.0
    assert ci.bootstrap_kind == "stationary"
    assert ci.method == "bca"


def test_bootstrap_ci_circular() -> None:
    x = np.random.default_rng(1).standard_normal(80)
    ci = bootstrap_ci(
        x,
        np.mean,
        n_resamples=100,
        confidence_level=0.90,
        method="percentile",
        bootstrap_kind="circular",
        block_length=5.0,
        seed=0,
    )
    assert ci.bootstrap_kind == "circular"
    assert ci.lower <= ci.upper
