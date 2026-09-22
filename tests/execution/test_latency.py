"""Tests for execution/latency.py.

Offline, seeded, no network. Critical invariants under test:

1. Latency is never zero by default (the bar-close invariant).
2. Same seed → identical latency draws (reproducibility).
3. Landing probability and landing delay are independent knobs.
4. ready_at(submitted_at) always returns > submitted_at (with default config).
5. Presets (P50, P90, P99) are valid LatencyConfig instances.
6. Constant distributions are deterministic; lognormal is variable.
7. Explicitly setting min_total_s=0.0 allows zero (deliberate override only).
"""

from __future__ import annotations

import pytest

from memetrader.execution.latency import (
    PRESET_P50,
    PRESET_P90,
    PRESET_P99,
    LatencyConfig,
    LatencyModel,
    StageConfig,
)

# ---------------------------------------------------------------------------
# No-zero invariant
# ---------------------------------------------------------------------------


class TestNoZeroLatency:
    def test_default_config_never_zero(self):
        """The bar-close invariant: a strategy decision at bar close must not
        fill at that close. The structural enforcement is that total_s > 0."""
        model = LatencyModel(seed=42)
        for _ in range(100):
            draw = model.draw()
            assert draw.total_s > 0.0, (
                "zero total latency would allow a fill at bar-close time"
            )

    def test_ready_at_always_greater_than_submitted_at(self):
        """ready_at must be strictly > submitted_at with the default config."""
        model = LatencyModel(seed=0)
        submitted_at = 1_234_567.0
        for _ in range(50):
            ra = model.ready_at(submitted_at)
            assert ra > submitted_at, (
                "ready_at <= submitted_at would let an order fill in the same bar"
            )

    def test_explicit_zero_min_allows_zero_only_on_constant_zero_config(self):
        """Returning zero requires *both* min_total_s=0.0 AND all constant
        stages drawing zero. This test documents that the caller must be
        deliberate about both decisions."""
        zero_stage = StageConfig(kind="constant", p50_s=0.0)
        cfg = LatencyConfig(
            quote_request=zero_stage,
            quote_response=zero_stage,
            risk_confirm=zero_stage,
            signing=zero_stage,
            submission=zero_stage,
            landing=zero_stage,
            min_total_s=0.0,
        )
        model = LatencyModel(cfg, seed=1)
        draw = model.draw()
        assert draw.total_s == 0.0, "explicit zero config should allow zero"

    def test_floor_applied_when_stages_sum_below_minimum(self):
        """If all stages happen to draw near-zero, the floor raises total to
        min_total_s. This is the structural guarantee."""
        tiny_stage = StageConfig(kind="constant", p50_s=0.001)
        cfg = LatencyConfig(
            quote_request=tiny_stage,
            quote_response=tiny_stage,
            risk_confirm=tiny_stage,
            signing=tiny_stage,
            submission=tiny_stage,
            landing=tiny_stage,
            min_total_s=1.0,  # much larger than the sum of stages (0.006)
        )
        model = LatencyModel(cfg, seed=7)
        for _ in range(20):
            draw = model.draw()
            assert draw.total_s >= 1.0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_draws(self):
        """Two models with the same seed must produce identical draw sequences.
        The robustness suite relies on this to compare baseline vs. stressed runs
        over identical random scenarios."""
        model_a = LatencyModel(PRESET_P50, seed=42)
        model_b = LatencyModel(PRESET_P50, seed=42)
        for _ in range(20):
            a = model_a.draw()
            b = model_b.draw()
            assert a.total_s == b.total_s
            assert a.will_land == b.will_land
            assert a.landing_delay_s == b.landing_delay_s

    def test_different_seeds_different_draws(self):
        """Two models with different seeds should produce different sequences
        (the probability of all 20 draws matching is astronomically small)."""
        model_a = LatencyModel(PRESET_P50, seed=1)
        model_b = LatencyModel(PRESET_P50, seed=2)
        totals_a = [model_a.draw().total_s for _ in range(20)]
        totals_b = [model_b.draw().total_s for _ in range(20)]
        assert totals_a != totals_b

    def test_same_model_consecutive_draws_vary(self):
        """The RNG state advances between draws; consecutive calls should not
        return the same value (would indicate a broken constant distribution)."""
        model = LatencyModel(PRESET_P50, seed=99)
        draws = [model.draw().total_s for _ in range(10)]
        # With a lognormal distribution, the probability of all 10 being equal
        # is effectively zero.
        assert len(set(draws)) > 1


# ---------------------------------------------------------------------------
# Landing probability and delay are independent
# ---------------------------------------------------------------------------


class TestLandingIndependence:
    def test_can_configure_low_prob_but_fast_landing(self):
        """Landing probability and landing delay are separate parameters. A
        configuration with low probability but fast delay should produce some
        non-landing fast draws."""
        cfg = LatencyConfig(
            landing_probability=0.1,  # mostly fails
            extra_landing_spread_s=0.0,  # but fast when it does land
            min_total_s=0.0,
        )
        model = LatencyModel(cfg, seed=42)
        draws = [model.draw() for _ in range(200)]
        not_landing = [d for d in draws if not d.will_land]
        landing = [d for d in draws if d.will_land]
        # With p=0.1 and 200 draws, we expect ~20 landings
        assert len(not_landing) > 100, "low probability should produce many non-landings"
        assert len(landing) > 0, "some draws should still land"

    def test_can_configure_high_prob_but_slow_landing(self):
        """High probability but extra delay — most land but are slow."""
        cfg = LatencyConfig(
            landing_probability=0.99,
            extra_landing_spread_s=5.0,
            min_total_s=0.1,
        )
        model = LatencyModel(cfg, seed=7)
        draws = [model.draw() for _ in range(100)]
        landing = [d for d in draws if d.will_land]
        assert len(landing) > 90, "high probability should land most often"
        delayed = [d for d in draws if d.landing_delay_s > 0.0]
        assert len(delayed) > 0, "extra_landing_spread_s > 0 should produce delays"

    def test_landing_probability_zero_never_lands(self):
        """landing_probability=0.0 means the transaction always drops."""
        cfg = LatencyConfig(landing_probability=0.0, min_total_s=0.1)
        model = LatencyModel(cfg, seed=3)
        for _ in range(50):
            draw = model.draw()
            assert not draw.will_land

    def test_landing_probability_one_always_lands(self):
        """landing_probability=1.0 means the transaction always lands."""
        cfg = LatencyConfig(landing_probability=1.0, min_total_s=0.1)
        model = LatencyModel(cfg, seed=3)
        for _ in range(50):
            draw = model.draw()
            assert draw.will_land

    def test_landing_delay_zero_when_no_extra_spread(self):
        """Without extra_landing_spread_s, landing_delay_s should be 0.0."""
        cfg = LatencyConfig(extra_landing_spread_s=0.0, min_total_s=0.1)
        model = LatencyModel(cfg, seed=5)
        for _ in range(20):
            draw = model.draw()
            assert draw.landing_delay_s == 0.0


# ---------------------------------------------------------------------------
# Presets are valid configs
# ---------------------------------------------------------------------------


class TestPresets:
    @pytest.mark.parametrize("preset", [PRESET_P50, PRESET_P90, PRESET_P99])
    def test_preset_is_valid_config(self, preset):
        """All presets must be constructible LatencyConfig instances with valid
        landing_probability and non-negative stage parameters."""
        model = LatencyModel(preset, seed=11)
        draw = model.draw()
        assert draw.total_s > 0.0
        assert isinstance(draw.will_land, bool)
        assert draw.landing_delay_s >= 0.0

    def test_p99_has_lower_landing_probability_than_p50(self):
        """Tail congestion should have more failures than median conditions."""
        assert PRESET_P99.landing_probability < PRESET_P50.landing_probability

    def test_p90_between_p50_and_p99(self):
        """P90 landing probability should be between P50 and P99."""
        assert (
            PRESET_P99.landing_probability
            <= PRESET_P90.landing_probability
            <= PRESET_P50.landing_probability
        )

    def test_p99_has_higher_extra_spread(self):
        """Tail congestion should have more extra landing delay."""
        assert PRESET_P99.extra_landing_spread_s >= PRESET_P50.extra_landing_spread_s


# ---------------------------------------------------------------------------
# Distribution shapes
# ---------------------------------------------------------------------------


class TestDistributions:
    def test_constant_stage_always_returns_same_value(self):
        stage = StageConfig(kind="constant", p50_s=0.42)
        import random

        rng = random.Random(1)
        for _ in range(20):
            assert stage.sample(rng) == pytest.approx(0.42)

    def test_lognormal_stage_varies(self):
        stage = StageConfig(kind="lognormal", p50_s=0.1, spread_s=0.5)
        import random

        rng = random.Random(77)
        samples = [stage.sample(rng) for _ in range(50)]
        assert len(set(samples)) > 1, "lognormal must produce varied samples"
        assert all(s >= 0 for s in samples), "lognormal is always non-negative"

    def test_uniform_stage_within_bounds(self):
        stage = StageConfig(kind="uniform", p50_s=0.2, spread_s=0.1)
        import random

        rng = random.Random(33)
        for _ in range(50):
            s = stage.sample(rng)
            assert s >= max(0.0, 0.2 - 0.1)
            assert s <= 0.2 + 0.1

    def test_unknown_kind_raises(self):
        """An unknown distribution kind should fail at sampling, not silently
        return a default value."""
        import random

        # Bypass frozen dataclass to inject bad kind for test
        stage = StageConfig.__new__(StageConfig)
        object.__setattr__(stage, "kind", "magic_dist")
        object.__setattr__(stage, "p50_s", 0.1)
        object.__setattr__(stage, "spread_s", 0.1)
        rng = random.Random(1)
        from memetrader.types import ValidationError

        with pytest.raises(ValidationError):
            stage.sample(rng)


# ---------------------------------------------------------------------------
# LatencyDraw non-negative invariants
# ---------------------------------------------------------------------------


class TestLatencyDrawInvariants:
    def test_all_stage_fields_non_negative(self):
        """All timing fields in a draw must be non-negative."""
        model = LatencyModel(PRESET_P50, seed=0)
        for _ in range(50):
            draw = model.draw()
            for attr in (
                "quote_request_s",
                "quote_response_s",
                "risk_confirm_s",
                "signing_s",
                "submission_s",
                "landing_s",
                "total_s",
                "landing_delay_s",
            ):
                assert getattr(draw, attr) >= 0.0, f"{attr} is negative"

    def test_ready_at_equals_submitted_plus_total(self):
        """ready_at draws once and returns submitted_at + total_s. The test
        verifies the relationship by checking readiness is always ahead of
        submission."""
        model = LatencyModel(seed=13)
        for t in (0.0, 1000.0, 1_700_000_000.0):
            ra = model.ready_at(t)
            assert ra > t or (ra == t and model.config.min_total_s == 0.0)
