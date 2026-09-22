"""Manifest: provenance correctness, hash stability, and non-git safety.

Each test states the failure it guards against rather than restating the
function under test. A manifest that hashes identically to a different run
is indistinguishable from that run, which makes any comparison between them
meaningless. A manifest that crashes in a tmp_path (no git) blocks the test
suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from memetrader.experiments.manifest import (
    Manifest,
    build_manifest,
    manifest_hash,
    read_manifest,
    write_manifest,
)
from memetrader.types import FidelityTier

EXP_ID = "exp-test-001"


def _base_manifest(tmp_path: Path, **kwargs: Any) -> Manifest:
    """Construct a manifest in a tmp_path (no git repo present).

    Callers may override any field via kwargs. Defaults are applied first so
    that kwargs always win — ``build_manifest`` does not accept duplicate keys,
    so we merge the defaults dict then pass it unpacked.
    """
    defaults: dict[str, Any] = {
        "experiment_id": EXP_ID,
        "trial_number": 1,
        "fidelity_tier": FidelityTier.TIER_0,
        "data_partition_hashes": {"train": "abc123", "holdout": "def456"},
        "universe_version": "v1",
        "random_seeds": {"data_split": 42},
        "holdout_previously_seen": False,
        "repo_root": tmp_path,
    }
    defaults.update(kwargs)
    return build_manifest(**defaults)


class TestNoGit:
    """The manifest must work in a directory with no git repository."""

    def test_build_in_no_git_dir(self, tmp_path: Path) -> None:
        """Building a manifest in a non-git directory records None for git fields
        rather than raising. Raising would block every test run in a tmp fixture.
        """
        m = _base_manifest(tmp_path)
        # git fields must be None (we tried and found nothing), not an error
        assert m.git_commit is None
        assert m.git_diff_hash is None

    def test_write_and_read_round_trip(self, tmp_path: Path) -> None:
        """A manifest written and then read back produces the same hash.

        If the round-trip changes the hash, two runs that serialize to the
        same file are not the same run by the manifest's own definition, which
        is the property that makes manifest_hash useless as a run identity.
        """
        m = _base_manifest(tmp_path)
        path = tmp_path / "manifest.json"
        write_manifest(path, m)
        m2 = read_manifest(path)
        assert manifest_hash(m) == manifest_hash(m2)

    def test_hash_stability_across_reserialization(self, tmp_path: Path) -> None:
        """Hash must be identical across multiple read-write cycles."""
        m = _base_manifest(tmp_path)
        path = tmp_path / "manifest.json"
        write_manifest(path, m)
        m2 = read_manifest(path)
        write_manifest(path, m2)
        m3 = read_manifest(path)
        h1 = manifest_hash(m)
        h2 = manifest_hash(m2)
        h3 = manifest_hash(m3)
        assert h1 == h2 == h3


class TestHashSensitivity:
    """Changing any field must change the hash.

    These tests are the guard against a hash function that ignores a field —
    which would make two manifests that differ on that field look identical.
    """

    def test_different_trial_number_different_hash(self, tmp_path: Path) -> None:
        m1 = _base_manifest(tmp_path, trial_number=1)
        m2 = _base_manifest(tmp_path, trial_number=2)
        # Override via dataclass mutation (build_manifest doesn't take trial_number kw
        # conflict, use it directly)
        assert manifest_hash(m1) != manifest_hash(m2)

    def test_different_fidelity_tier_different_hash(self, tmp_path: Path) -> None:
        m1 = build_manifest(
            experiment_id=EXP_ID,
            trial_number=1,
            fidelity_tier=FidelityTier.TIER_0,
            repo_root=tmp_path,
        )
        m2 = build_manifest(
            experiment_id=EXP_ID,
            trial_number=1,
            fidelity_tier=FidelityTier.TIER_1,
            repo_root=tmp_path,
        )
        assert manifest_hash(m1) != manifest_hash(m2)

    def test_different_seeds_different_hash(self, tmp_path: Path) -> None:
        m1 = _base_manifest(tmp_path, random_seeds={"data_split": 42})
        m2 = _base_manifest(tmp_path, random_seeds={"data_split": 99})
        assert manifest_hash(m1) != manifest_hash(m2)

    def test_different_partition_hash_different_hash(self, tmp_path: Path) -> None:
        m1 = _base_manifest(tmp_path, data_partition_hashes={"train": "aaa"})
        m2 = _base_manifest(tmp_path, data_partition_hashes={"train": "bbb"})
        assert manifest_hash(m1) != manifest_hash(m2)

    def test_holdout_seen_flag_changes_hash(self, tmp_path: Path) -> None:
        """holdout_previously_seen=False and True must hash differently.

        A manifest that claims the holdout was not seen when it was is a false
        research record. The hash must reflect the flag so that the claim is
        part of the run identity.
        """
        m1 = _base_manifest(tmp_path, holdout_previously_seen=False)
        m2 = _base_manifest(tmp_path, holdout_previously_seen=True)
        assert manifest_hash(m1) != manifest_hash(m2)

    def test_different_experiment_id_different_hash(self, tmp_path: Path) -> None:
        m1 = build_manifest(
            experiment_id="exp-A",
            trial_number=1,
            fidelity_tier=FidelityTier.TIER_0,
            repo_root=tmp_path,
        )
        m2 = build_manifest(
            experiment_id="exp-B",
            trial_number=1,
            fidelity_tier=FidelityTier.TIER_0,
            repo_root=tmp_path,
        )
        assert manifest_hash(m1) != manifest_hash(m2)

    def test_feature_definitions_hash_captured(self, tmp_path: Path) -> None:
        m1 = _base_manifest(tmp_path, feature_definitions={"rsi": {"window": 14}})
        m2 = _base_manifest(tmp_path, feature_definitions={"rsi": {"window": 21}})
        assert manifest_hash(m1) != manifest_hash(m2)

    def test_lock_file_hash_captured(self, tmp_path: Path) -> None:
        """If a uv.lock file is present, its hash is part of the manifest."""
        lock = tmp_path / "uv.lock"
        lock.write_text("version 1\ndep==1.0.0\n", encoding="utf-8")
        m1 = build_manifest(
            experiment_id=EXP_ID,
            trial_number=1,
            fidelity_tier=FidelityTier.TIER_0,
            lock_file=lock,
            repo_root=tmp_path,
        )
        assert m1.lock_hash is not None

        lock.write_text("version 1\ndep==2.0.0\n", encoding="utf-8")
        m2 = build_manifest(
            experiment_id=EXP_ID,
            trial_number=1,
            fidelity_tier=FidelityTier.TIER_0,
            lock_file=lock,
            repo_root=tmp_path,
        )
        assert manifest_hash(m1) != manifest_hash(m2)


class TestWriteRead:
    """Persistence correctness."""

    def test_atomic_write_no_partial_read(self, tmp_path: Path) -> None:
        """write_manifest uses atomic_write_text, so the file is always complete."""
        m = _base_manifest(tmp_path)
        path = tmp_path / "manifest.json"
        write_manifest(path, m)
        assert path.is_file()
        # read back must not raise
        m2 = read_manifest(path)
        assert m2.experiment_id == EXP_ID

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises((FileNotFoundError, OSError)):
            read_manifest(tmp_path / "nonexistent.json")

    def test_parent_experiment_id_preserved(self, tmp_path: Path) -> None:
        m = build_manifest(
            experiment_id=EXP_ID,
            trial_number=2,
            fidelity_tier=FidelityTier.TIER_0,
            parent_experiment_id="exp-parent-000",
            repo_root=tmp_path,
        )
        path = tmp_path / "manifest.json"
        write_manifest(path, m)
        m2 = read_manifest(path)
        assert m2.parent_experiment_id == "exp-parent-000"
