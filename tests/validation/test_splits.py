"""Tests for memetrader.validation.splits.

Every test is offline and deterministic.  Fixtures are constructed in-memory
from calendar arithmetic so the tests are reproducible without any history
files.

Critical tests (each must FAIL if the guard they exercise is removed):
- test_fold_audit_no_overlap         — no training label overlaps val/test
- test_fold_audit_fails_without_purge — contamination IS present when purge off
- test_embargo_wall_clock_across_density — embargo holds for sparse vs dense assets
- test_holdout_warning_fires          — warning fires at 209-day horizon
- test_cpcv_path_count                — CPCV produces C(n, k) paths
- test_all_assets_same_boundaries     — same wall-clock split for every asset
"""

from __future__ import annotations

import math
import warnings

import pytest

from memetrader.validation.splits import (
    CombinatorialPurgedCV,
    FoldRecord,
    Interval,
    LabeledObservation,
    PurgedWalkForward,
    SplitResult,
    embargo_window,
    extract_locked_holdout,
    purge_training,
)

# ---------------------------------------------------------------------------
# Helpers / fixture builders
# ---------------------------------------------------------------------------

_DAY = 86400.0
_HOUR = 3600.0


def _epoch(day_offset: float, hour: float = 0.0) -> float:
    """Epoch seconds for a fixed origin + day_offset days + hour hours.

    Using a fixed origin (0.0 = "day 0") keeps the fixture arithmetic simple
    and removes any dependence on real calendar conversions.
    """
    return day_offset * _DAY + hour * _HOUR


def _make_obs(
    asset_id: str,
    obs_time: float,
    label_horizon_hours: float = 4.0,
) -> LabeledObservation:
    """One observation whose label spans ``label_horizon_hours`` forward."""
    return LabeledObservation(
        asset_id=asset_id,
        obs_time=obs_time,
        label_start_ts=obs_time,
        label_end_ts=obs_time + label_horizon_hours * _HOUR,
    )


def _dense_observations(
    asset_id: str,
    start: float,
    end: float,
    interval_hours: float = 1.0,
    label_horizon_hours: float = 4.0,
) -> list[LabeledObservation]:
    """Regular grid of observations (like BONK at 1h: one bar per hour)."""
    out: list[LabeledObservation] = []
    t = start
    while t < end:
        out.append(_make_obs(asset_id, t, label_horizon_hours))
        t += interval_hours * _HOUR
    return out


def _sparse_observations(
    asset_id: str,
    start: float,
    end: float,
    avg_interval_hours: float = 37.0,
    label_horizon_hours: float = 4.0,
) -> list[LabeledObservation]:
    """Irregularly-spaced observations (like SLERF: one print per ~37 min).

    We use a fixed-period grid at avg_interval_hours to keep tests
    deterministic while still modelling the sparsity.
    """
    out: list[LabeledObservation] = []
    t = start
    while t < end:
        out.append(_make_obs(asset_id, t, label_horizon_hours))
        t += avg_interval_hours * _HOUR
    return out


# ---------------------------------------------------------------------------
# Interval unit tests
# ---------------------------------------------------------------------------


class TestInterval:
    def test_basic_construction(self) -> None:
        iv = Interval(0.0, 100.0)
        assert iv.start == 0.0
        assert iv.end == 100.0

    def test_raises_for_equal_endpoints(self) -> None:
        with pytest.raises(ValueError, match="strictly after"):
            Interval(50.0, 50.0)

    def test_raises_for_reversed_endpoints(self) -> None:
        with pytest.raises(ValueError, match="strictly after"):
            Interval(100.0, 50.0)

    def test_overlaps_overlapping(self) -> None:
        a = Interval(0.0, 100.0)
        b = Interval(50.0, 150.0)
        assert a.overlaps(b)
        assert b.overlaps(a)

    def test_overlaps_touching_does_not_overlap(self) -> None:
        # Abutting intervals: [0, 100) and [100, 200) share no instant.
        a = Interval(0.0, 100.0)
        b = Interval(100.0, 200.0)
        assert not a.overlaps(b)
        assert not b.overlaps(a)

    def test_overlaps_contained(self) -> None:
        outer = Interval(0.0, 200.0)
        inner = Interval(50.0, 100.0)
        assert outer.overlaps(inner)
        assert inner.overlaps(outer)

    def test_no_overlap_disjoint(self) -> None:
        a = Interval(0.0, 50.0)
        b = Interval(60.0, 100.0)
        assert not a.overlaps(b)

    def test_duration_days(self) -> None:
        iv = Interval(0.0, _DAY * 30)
        assert iv.duration_days() == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# purge_training unit tests
# ---------------------------------------------------------------------------


class TestPurgeTraining:
    def _val_test(self) -> tuple[Interval, Interval]:
        val = Interval(_epoch(180), _epoch(210))
        test = Interval(_epoch(210), _epoch(240))
        return val, test

    def test_observations_before_boundary_kept(self) -> None:
        val, test = self._val_test()
        train_interval = Interval(_epoch(0), _epoch(180))
        obs = _dense_observations("BONK", _epoch(0), _epoch(170))
        safe, n_purged, n_embargoed = purge_training(
            obs,
            protected_intervals=[val, test],
            train_interval=train_interval,
            embargo_secs=embargo_window(),
        )
        assert n_purged == 0
        assert n_embargoed > 0 or len(safe) == len(obs)  # some may be embargoed

    def test_observations_with_label_in_val_are_purged(self) -> None:
        """An obs whose label extends into val must be purged."""
        val = Interval(_epoch(10), _epoch(20))
        test = Interval(_epoch(20), _epoch(30))
        train_interval = Interval(_epoch(0), _epoch(10))
        # A label that starts at day 9.9 and has a 4-hour horizon extends into val.
        obs = [
            LabeledObservation(
                asset_id="BONK",
                obs_time=_epoch(9, 22),  # hour 22 of day 9
                label_start_ts=_epoch(9, 22),
                label_end_ts=_epoch(10, 2),  # 4 hours later, crosses into val
            )
        ]
        safe, n_purged, n_embargoed = purge_training(
            obs,
            protected_intervals=[val, test],
            train_interval=train_interval,
            embargo_secs=1.0,  # minimal embargo so we isolate purge
        )
        assert n_purged == 1
        assert len(safe) == 0

    def test_no_purge_when_disabled(self) -> None:
        """With enable_purge=False, contaminated obs are kept (for guard test)."""
        val = Interval(_epoch(10), _epoch(20))
        test = Interval(_epoch(20), _epoch(30))
        train_interval = Interval(_epoch(0), _epoch(10))
        obs = [
            LabeledObservation(
                asset_id="BONK",
                obs_time=_epoch(9, 22),
                label_start_ts=_epoch(9, 22),
                label_end_ts=_epoch(10, 2),  # overlaps val
            )
        ]
        safe, n_purged, n_embargoed = purge_training(
            obs,
            protected_intervals=[val, test],
            train_interval=train_interval,
            embargo_secs=1.0,
            enable_purge=False,
        )
        # With purge off, the contaminated observation survives.
        assert n_purged == 0
        assert n_embargoed == 0
        assert 0 in safe  # the observation was kept despite its label overlapping val

    def test_embargo_removes_obs_near_boundary(self) -> None:
        """Observations inside the embargo zone are counted separately from purged."""
        val = Interval(_epoch(10), _epoch(20))
        test = Interval(_epoch(20), _epoch(30))
        # Embargo is 1 day after val.end and after test.end.
        embargo_secs = _DAY
        train_interval = Interval(_epoch(0), _epoch(10))
        # An obs at day 9.5 with a 1-second label (no purge overlap with val/test),
        # but within embargo_secs of val.start.
        # val.end = epoch(20); embargo zone = [epoch(20), epoch(21)).
        # obs_time = epoch(9.5) is in train, label ends at epoch(9.5)+1 < epoch(10).
        # Not purged (label doesn't overlap val/test), but in embargo zone? No:
        # embargo zone starts at epoch(20), obs is at epoch(9.5). Not embargoed either.
        # Let's put obs in [val.end, val.end + embargo_secs) = [epoch(20), epoch(21))
        # but that's outside train_interval [epoch(0), epoch(10)).
        # The correct test: obs_time in train interval but near the END of train.
        # val starts at epoch(10). Embargo before val = not modelled; embargo is
        # AFTER test/val. So let's test embargo after test.
        # An observation at epoch(30, 0.5h) = epoch(30) + 1800s would be in
        # the embargo zone [test.end, test.end+embargo_secs) = [epoch(30), epoch(31)).
        # But it must also be in train_interval. Let's use a train that goes to epoch(32).
        train_interval2 = Interval(_epoch(0), _epoch(32))
        obs2 = [
            LabeledObservation(
                asset_id="BONK",
                obs_time=_epoch(30, 0.5),  # inside embargo zone after test
                label_start_ts=_epoch(30, 0.5),
                label_end_ts=_epoch(30, 0.5) + 1.0,  # tiny label, no overlap with test
            )
        ]
        safe2, n_purged2, n_embargoed2 = purge_training(
            obs2,
            protected_intervals=[val, test],
            train_interval=train_interval2,
            embargo_secs=embargo_secs,
        )
        assert n_embargoed2 == 1
        assert len(safe2) == 0


# ---------------------------------------------------------------------------
# Fold audit — the spec's primary requirement
# ---------------------------------------------------------------------------


class TestFoldAudit:
    """The fold audit: no training label interval overlaps validation/test.

    This is BACKTEST-CONTRACTS.md §8: "Fold audit: no training label interval
    overlaps validation/test." The test must also FAIL when purging is disabled,
    which proves the guard is doing real work, not just passing.
    """

    def _build_obs(self) -> tuple[list[LabeledObservation], float, float]:
        """Build a 209-day dataset matching the real horizon, 1h bars, 3 assets."""
        total_days = 209.0
        data_start = 0.0
        data_end = total_days * _DAY
        bonk = _dense_observations("BONK", data_start, data_end, 1.0, 4.0)
        slerf = _sparse_observations("SLERF", data_start, data_end, 37.0 / 60.0, 4.0)
        giga = _dense_observations("GIGA", data_start, data_end, 1.0, 4.0)
        all_obs = bonk + slerf + giga
        return all_obs, data_start, data_end

    def test_fold_audit_no_overlap(self) -> None:
        """No training observation's label interval overlaps val or test.

        This is the canonical fold audit test from the spec. If this fails,
        the purge logic is broken.
        """
        all_obs, data_start, data_end = self._build_obs()
        splitter = PurgedWalkForward(
            train_days=120.0,
            val_days=20.0,
            test_days=20.0,
            roll_days=20.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            splits = splitter.split(
                all_obs, data_start=data_start, data_end=data_end, enable_purge=True
            )

        assert splits, "Expected at least one fold"
        for sr in splits:
            val_iv = sr.fold.val_interval
            test_iv = sr.fold.test_interval
            for i in sr.train_indices:
                obs = all_obs[i]
                label = obs.label_interval
                assert not label.overlaps(val_iv), (
                    f"Training obs {obs.asset_id}@{obs.obs_time:.0f} has label "
                    f"[{obs.label_start_ts:.0f},{obs.label_end_ts:.0f}) overlapping "
                    f"val [{val_iv.start:.0f},{val_iv.end:.0f})"
                )
                assert not label.overlaps(test_iv), (
                    f"Training obs {obs.asset_id}@{obs.obs_time:.0f} has label "
                    f"[{obs.label_start_ts:.0f},{obs.label_end_ts:.0f}) overlapping "
                    f"test [{test_iv.start:.0f},{test_iv.end:.0f})"
                )

    def test_fold_audit_fails_without_purge(self) -> None:
        """GUARD TEST: contamination EXISTS when purge is disabled.

        This test must PASS (i.e. contamination is found). If this test starts
        failing, it means either the data fixture no longer produces observations
        near fold boundaries, or purge_training's enable_purge=False path has
        been changed to still filter—both would mean the guard test is no longer
        proving anything.
        """
        all_obs, data_start, data_end = self._build_obs()
        splitter = PurgedWalkForward(
            train_days=120.0,
            val_days=20.0,
            test_days=20.0,
            roll_days=20.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            splits_no_purge = splitter.split(
                all_obs,
                data_start=data_start,
                data_end=data_end,
                enable_purge=False,
            )

        contaminated_found = False
        for sr in splits_no_purge:
            val_iv = sr.fold.val_interval
            test_iv = sr.fold.test_interval
            for i in sr.train_indices:
                obs = all_obs[i]
                label = obs.label_interval
                if label.overlaps(val_iv) or label.overlaps(test_iv):
                    contaminated_found = True
                    break
            if contaminated_found:
                break

        assert contaminated_found, (
            "Expected contamination when purge is disabled, but none was found. "
            "The guard test is no longer meaningful — either the fixture needs more "
            "observations near fold boundaries, or enable_purge=False is still filtering."
        )


# ---------------------------------------------------------------------------
# Embargo across different asset densities
# ---------------------------------------------------------------------------


class TestEmbargoAcrossDensity:
    """Verify that embargo is wall-clock, not row-count based.

    BONK has one observation per hour. SLERF has one per ~37 minutes.
    After a fold boundary at ``fold_end``, both assets must have observations
    in [fold_end, fold_end + embargo_secs) removed, regardless of how many
    rows that is per asset.
    """

    def test_embargo_wall_clock_across_density(self) -> None:
        # Short data span: 60 days, fold at day 30, val [30,35), test [35,40).
        data_start = 0.0
        fold_end = _epoch(40)  # after test
        embargo_secs = _DAY * 2  # 2 days

        val = Interval(_epoch(30), _epoch(35))
        test = Interval(_epoch(35), _epoch(40))
        # Train interval goes from day 0 to day 45 (encompasses embargo zone).
        train_interval = Interval(_epoch(0), _epoch(45))

        # Dense asset: 1 obs/hour. In embargo zone [epoch(40), epoch(42)):
        # 48 hours × 1 obs/hour = 48 observations should be embargoed.
        bonk = _dense_observations("BONK", _epoch(0), _epoch(45), 1.0, 0.1)
        # Sparse asset: 1 obs per 37 minutes. Same zone has ~48×60/37 ≈ 78 obs.
        slerf = _sparse_observations("SLERF", _epoch(0), _epoch(45), 37.0 / 60.0, 0.1)

        all_obs = bonk + slerf

        safe_b, n_purged_b, n_embargoed_b = purge_training(
            bonk,
            protected_intervals=[val, test],
            train_interval=train_interval,
            embargo_secs=embargo_secs,
        )
        safe_s, n_purged_s, n_embargoed_s = purge_training(
            slerf,
            protected_intervals=[val, test],
            train_interval=train_interval,
            embargo_secs=embargo_secs,
        )

        # Both assets must have some embargoed observations.
        assert n_embargoed_b > 0, "BONK: no observations were embargoed"
        assert n_embargoed_s > 0, "SLERF: no observations were embargoed"

        # SLERF has more obs in the same wall-clock window (it's denser here).
        # The wall-clock embargo means the *count* differs between assets — that
        # is correct and expected. What must NOT differ is the wall-clock end of
        # the embargo zone.
        # Verify: no safe observation for either asset falls in [test.end, test.end+embargo).
        embargo_start = test.end
        embargo_end_ts = test.end + embargo_secs

        for i in safe_b:
            obs = bonk[i]
            assert not (embargo_start <= obs.obs_time < embargo_end_ts), (
                f"BONK obs at {obs.obs_time:.0f} survived embargo "
                f"[{embargo_start:.0f}, {embargo_end_ts:.0f})"
            )
        for i in safe_s:
            obs = slerf[i]
            assert not (embargo_start <= obs.obs_time < embargo_end_ts), (
                f"SLERF obs at {obs.obs_time:.0f} survived embargo "
                f"[{embargo_start:.0f}, {embargo_end_ts:.0f})"
            )


# ---------------------------------------------------------------------------
# Holdout warning
# ---------------------------------------------------------------------------


class TestHoldoutWarning:
    def test_holdout_warning_fires_at_209_day_horizon(self) -> None:
        """Warning fires when holdout is below plan's 90-day recommendation.

        At 209 days, the default holdout is max(45, 0.20 × 209) = 41.8 days,
        which is below 90 days. The warning must fire. This test asserts that
        the warning contains the shortfall information (not just that it fires).
        """
        data_start = 0.0
        data_end = 209.0 * _DAY
        obs = _dense_observations("BONK", data_start, data_end, 1.0, 4.0)
        splitter = PurgedWalkForward(
            train_days=120.0,
            val_days=20.0,
            test_days=20.0,
            roll_days=20.0,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            splitter.split(obs, data_start=data_start, data_end=data_end)

        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert user_warnings, "Expected a UserWarning about holdout shortfall"
        msg = str(user_warnings[0].message)
        assert (
            "209" in msg
            or "209.0" in msg
            or "shortfall" in msg.lower()
            or "below" in msg.lower()
        ), f"Warning message did not mention the data horizon or shortfall: {msg}"

    def test_holdout_warning_not_fired_for_large_holdout(self) -> None:
        """No warning when holdout is at or above 90 days."""
        data_start = 0.0
        data_end = 500.0 * _DAY
        obs = _dense_observations("BONK", data_start, data_end, 1.0, 4.0)
        splitter = PurgedWalkForward(
            train_days=120.0,
            val_days=20.0,
            test_days=20.0,
            roll_days=20.0,
            locked_holdout_days=90.0,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            splitter.split(obs, data_start=data_start, data_end=data_end)

        holdout_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning) and "shortfall" in str(w.message).lower()
        ]
        assert not holdout_warnings, (
            "Unexpected holdout shortfall warning when holdout == 90 days"
        )


# ---------------------------------------------------------------------------
# All assets split on same wall-clock boundaries
# ---------------------------------------------------------------------------


class TestSameBoundariesAllAssets:
    """All assets must be split on the same wall-clock fold boundaries.

    Assigning observations to folds by row count instead of timestamp lets the
    model see one market event in both train and test (the event lands at
    different row indices for different assets). The structural protection is
    that all boundaries in splits.py are wall-clock seconds, and this test
    checks the resulting fold boundaries are asset-agnostic.
    """

    def test_all_assets_same_boundaries(self) -> None:
        data_start = 0.0
        data_end = 209.0 * _DAY
        bonk = _dense_observations("BONK", data_start, data_end, 1.0, 4.0)
        slerf = _sparse_observations("SLERF", data_start, data_end, 37.0 / 60.0, 4.0)
        all_obs = bonk + slerf

        splitter = PurgedWalkForward(
            train_days=120.0,
            val_days=20.0,
            test_days=20.0,
            roll_days=20.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            splits_combined = splitter.split(
                all_obs, data_start=data_start, data_end=data_end
            )

        # Run split separately for each asset.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            splits_bonk = splitter.split(bonk, data_start=data_start, data_end=data_end)
            splits_slerf = splitter.split(slerf, data_start=data_start, data_end=data_end)

        # The combined split must produce the same number of folds as per-asset.
        assert len(splits_combined) == len(splits_bonk) == len(splits_slerf)

        # Each fold's wall-clock boundaries must be identical regardless of which
        # asset list was used.
        for sc, sb, ss in zip(splits_combined, splits_bonk, splits_slerf):
            assert sc.fold.train_start == sb.fold.train_start == ss.fold.train_start
            assert sc.fold.train_end == sb.fold.train_end == ss.fold.train_end
            assert sc.fold.val_start == sb.fold.val_start == ss.fold.val_start
            assert sc.fold.val_end == sb.fold.val_end == ss.fold.val_end
            assert sc.fold.test_start == sb.fold.test_start == ss.fold.test_start
            assert sc.fold.test_end == sb.fold.test_end == ss.fold.test_end


# ---------------------------------------------------------------------------
# CPCV path count
# ---------------------------------------------------------------------------


class TestCPCVPathCount:
    def test_cpcv_path_count(self) -> None:
        """CPCV with n_groups=6, k=2 must produce C(6,2)=15 paths."""
        data_start = 0.0
        data_end = 209.0 * _DAY
        obs = _dense_observations("BONK", data_start, data_end, 1.0, 4.0)
        cpcv = CombinatorialPurgedCV(n_groups=6, k=2)
        assert cpcv.n_paths() == 15

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            results = cpcv.split(obs, data_start=data_start, data_end=data_end)

        assert len(results) == 15, f"Expected 15 CPCV paths for C(6,2), got {len(results)}"

    def test_cpcv_path_count_other_config(self) -> None:
        """C(4,1)=4 paths for n_groups=4, k=1."""
        cpcv = CombinatorialPurgedCV(n_groups=4, k=1)
        assert cpcv.n_paths() == 4

        data_start = 0.0
        data_end = 209.0 * _DAY
        obs = _dense_observations("BONK", data_start, data_end, 1.0, 4.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            results = cpcv.split(obs, data_start=data_start, data_end=data_end)
        assert len(results) == 4

    def test_cpcv_k_gte_n_groups_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly less than"):
            CombinatorialPurgedCV(n_groups=4, k=4)

    def test_cpcv_fold_audit(self) -> None:
        """No training label interval overlaps any test interval in CPCV folds."""
        data_start = 0.0
        data_end = 120.0 * _DAY
        obs = _dense_observations("BONK", data_start, data_end, 1.0, 4.0)
        cpcv = CombinatorialPurgedCV(n_groups=4, k=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            results = cpcv.split(obs, data_start=data_start, data_end=data_end)

        for sr in results:
            test_start = sr.fold.test_start
            test_end = sr.fold.test_end
            test_iv = Interval(test_start, test_end)
            for i in sr.train_indices:
                label = obs[i].label_interval
                assert not label.overlaps(test_iv), (
                    f"CPCV fold '{sr.fold.fold_id}': training label overlaps test"
                )


# ---------------------------------------------------------------------------
# FoldRecord serialisation
# ---------------------------------------------------------------------------


class TestFoldRecordSerialisation:
    def test_as_dict_round_trip(self) -> None:
        record = FoldRecord(
            fold_id="test_fold_000",
            splitter_id="purged_walk_forward_v1",
            train_start=0.0,
            train_end=_epoch(180),
            val_start=_epoch(180),
            val_end=_epoch(210),
            test_start=_epoch(210),
            test_end=_epoch(240),
            embargo_end=_epoch(241),
            purged_count=12,
            embargoed_count=5,
            notes="test",
        )
        d = record.as_dict()
        assert d["fold_id"] == "test_fold_000"
        assert d["purged_count"] == 12
        assert d["embargoed_count"] == 5
        assert math.isfinite(d["train_start"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# embargo_window
# ---------------------------------------------------------------------------


class TestEmbargoWindow:
    def test_default_embargo_positive(self) -> None:
        ew = embargo_window()
        assert ew > 0

    def test_components_additive(self) -> None:
        """The components should sum correctly (modulo the max() for horizon/delay)."""
        ew = embargo_window(
            holding_horizon_secs=4 * _HOUR,
            publication_delay_secs=1 * _HOUR,
            rolling_state_secs=21 * _HOUR,
            serial_dependence_secs=6 * _HOUR,
        )
        # max(4h, 1h) + 21h + 6h = 4h + 21h + 6h = 31h
        expected = 4 * _HOUR + 21 * _HOUR + 6 * _HOUR
        assert ew == pytest.approx(expected)

    def test_max_of_holding_and_publication(self) -> None:
        """When publication_delay > holding_horizon, publication_delay dominates."""
        ew = embargo_window(
            holding_horizon_secs=1 * _HOUR,
            publication_delay_secs=10 * _HOUR,
            rolling_state_secs=0.0,
            serial_dependence_secs=0.0,
        )
        assert ew == pytest.approx(10 * _HOUR)


# ---------------------------------------------------------------------------
# extract_locked_holdout
# ---------------------------------------------------------------------------


class TestExtractLockedHoldout:
    def test_holdout_at_end(self) -> None:
        data_start = 0.0
        data_end = 209.0 * _DAY
        obs = _dense_observations("BONK", data_start, data_end, 1.0, 4.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dev_idx, ho_idx, record = extract_locked_holdout(
                obs,
                data_start=data_start,
                data_end=data_end,
                locked_holdout_days=45.0,
            )
        assert dev_idx
        assert ho_idx
        # Every holdout obs must be after every dev obs.
        dev_times = {obs[i].obs_time for i in dev_idx}
        ho_times = {obs[i].obs_time for i in ho_idx}
        assert max(dev_times) < min(ho_times)

    def test_non_overlapping_indices(self) -> None:
        data_start = 0.0
        data_end = 100.0 * _DAY
        obs = _dense_observations("BONK", data_start, data_end, 1.0, 4.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dev_idx, ho_idx, _ = extract_locked_holdout(
                obs,
                data_start=data_start,
                data_end=data_end,
            )
        overlap = set(dev_idx) & set(ho_idx)
        assert not overlap, f"dev and holdout indices overlap: {overlap}"
