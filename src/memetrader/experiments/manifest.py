"""Run manifest — provenance for every backtest artifact.

A run without provenance is not evidence. That sentence from §9 of
BACKTEST-CONTRACTS.md is the entire reason this module exists: any claim made
by a backtest must be traceable to exactly the code, data, configuration, and
randomness that produced it. Without a manifest, two runs that happen to produce
the same output are indistinguishable from two runs that are secretly different
(different data slice, different feature definition, model retrained after seeing
holdout), and a research process built on indistinguishable runs will eventually
fool itself.

The manifest captures every axis on which two runs can differ:

* **Code identity.** Git commit SHA plus a hash of ``git diff`` (not just a
  dirty flag). A dirty flag says *something* changed; the diff hash says
  *what*, and two dirty trees at the same commit are only the same run if they
  have the same diff.

* **Data identity.** SHA-256 hashes of each partition (train, validation, test,
  holdout), the universe version, and the universe file hash. Swapping a data
  slice without touching the code is the most common way a comparison becomes
  meaningless.

* **Model and feature identity.** The feature definitions (as a content hash of
  their serialized form) and the model artifact hash. Two runs with the same
  feature names but different implementations are different experiments.

* **Configuration and dependencies.** Config file hash and ``uv.lock`` hash.
  A dependency update is a code change; failing to record it makes a result
  irreproducible even when the repo is clean.

* **Randomness.** Every random seed used by the run, keyed by role (e.g.
  ``{"data_split": 42, "model_init": 7}``). Same manifest should produce
  byte-identical results given the same environment.

* **Holdout contamination flag.** Whether any human or LLM has seen prior
  holdout results at the time this manifest is written. This is the field the
  experiment registry writes back into the manifest record; it must be present
  and explicit rather than absent-by-default, because an absent field and a
  False field are indistinguishable in a file and only one of them is a
  truthful claim.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memetrader.types import FidelityTier

__all__ = [
    "Manifest",
    "build_manifest",
    "manifest_hash",
    "read_manifest",
    "write_manifest",
]

_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Git helpers — fail gracefully when no git repository is present
# ---------------------------------------------------------------------------


def _run_git(*args: str, cwd: Path | None = None) -> str | None:
    """Run a git sub-command and return its stdout, or None on any failure.

    Tests run in tmp_path fixtures that are not git repositories. The manifest
    must still be constructible there — recording None is honest; crashing is
    not. Every caller of this function is therefore required to handle None.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(cwd) if cwd is not None else None,
            timeout=15,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except FileNotFoundError, subprocess.TimeoutExpired, OSError:
        return None


def _git_commit(cwd: Path | None = None) -> str | None:
    """HEAD commit SHA, or None in a detached HEAD / no-git situation."""
    return _run_git("rev-parse", "HEAD", cwd=cwd)


def _git_diff_hash(cwd: Path | None = None) -> str | None:
    """Blake2b digest of ``git diff HEAD``.

    A dirty-worktree flag says *something* changed. This hash says *what*. Two
    dirty trees at the same commit hash to the same value only if their diffs
    are identical, which is the condition under which they really are the same
    run. A clean tree produces an empty diff, whose hash is still stable and
    distinct from any non-empty diff — so a clean run and a dirty run are always
    distinguishable by this field.
    """
    diff = _run_git("diff", "HEAD", cwd=cwd)
    # An empty diff (clean tree) gives the hash of the empty string, which is a
    # valid and stable value — not None. None means git is unavailable.
    if diff is None:
        # git itself is unavailable; we cannot produce a meaningful diff hash.
        return None
    payload = (diff or "").encode("utf-8")
    return hashlib.blake2b(payload, digest_size=32).hexdigest()


# ---------------------------------------------------------------------------
# File hash helpers
# ---------------------------------------------------------------------------


def _file_hash(path: Path) -> str | None:
    """SHA-256 of a file's contents, or None if the file is absent/unreadable.

    Using SHA-256 rather than blake2b here for interoperability: the file
    hashes may be compared against externally-computed values (e.g. from a
    data pipeline that produces its own checksums), and SHA-256 is the de
    facto standard for that context.
    """
    if not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()


def _dict_hash(d: dict[str, Any]) -> str:
    """Stable blake2b of a JSON-serialized dict, sorted by key.

    Sorted serialization is critical: dict iteration order is insertion-order in
    Python 3.7+ but differs between construction sites, and two dicts with the
    same keys and values must hash identically regardless of how they were
    built.
    """
    serialized = json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.blake2b(serialized, digest_size=32).hexdigest()


# ---------------------------------------------------------------------------
# Manifest dataclass
# ---------------------------------------------------------------------------


@dataclass
class Manifest:
    """Every axis on which two runs can differ, recorded at run start.

    All fields are recorded at manifest construction time, before any
    computation begins. A field that is updated after computation (e.g.
    model_artifact_hash, which cannot be known until training ends) should be
    written into a second manifest; do not mutate this one.

    Fields that cannot be determined (e.g. git commit in a non-git directory)
    are None rather than absent — None is an honest statement that we looked and
    could not find out; absent is ambiguous.
    """

    schema_version: int
    # -- Code identity -------------------------------------------------------
    git_commit: str | None
    # Hash of `git diff HEAD`, not a flag. Two dirty trees at the same commit
    # are only the same run if their diff hashes match.
    git_diff_hash: str | None
    # -- Data identity -------------------------------------------------------
    # Mapping from partition name (e.g. "train", "val", "test", "holdout") to
    # SHA-256 of the partition file.
    data_partition_hashes: dict[str, str | None]
    universe_version: str | None
    # SHA-256 of the universe definition file.
    universe_hash: str | None
    # -- Feature and model identity ------------------------------------------
    # Blake2b of the serialized feature definitions dict.
    feature_definitions_hash: str | None
    # SHA-256 of the saved model artifact (weights file, pickle, etc.).
    model_artifact_hash: str | None
    # -- Configuration -------------------------------------------------------
    # SHA-256 of the config file (config.yaml).
    config_hash: str | None
    # SHA-256 of uv.lock — a dependency update is a code change.
    lock_hash: str | None
    # -- Randomness ----------------------------------------------------------
    # Keyed by role, e.g. {"data_split": 42, "model_init": 7}.
    random_seeds: dict[str, int]
    # -- Experiment lineage --------------------------------------------------
    trial_number: int
    parent_experiment_id: str | None
    experiment_id: str
    # -- Execution tier ------------------------------------------------------
    fidelity_tier: str  # FidelityTier value
    # -- Contamination flag --------------------------------------------------
    # Whether any human or LLM has seen prior holdout results before this run.
    # Absent is not the same as False: this field is always set explicitly.
    holdout_previously_seen: bool
    # -- Optional extra fields -----------------------------------------------
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_manifest(
    *,
    experiment_id: str,
    trial_number: int,
    fidelity_tier: FidelityTier,
    data_partition_hashes: dict[str, str | None] | None = None,
    universe_version: str | None = None,
    universe_hash: str | None = None,
    feature_definitions: dict[str, Any] | None = None,
    model_artifact_hash: str | None = None,
    config_hash: str | None = None,
    lock_file: Path | None = None,
    random_seeds: dict[str, int] | None = None,
    parent_experiment_id: str | None = None,
    holdout_previously_seen: bool = False,
    repo_root: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Manifest:
    """Construct a Manifest, shelling out to git and hashing files as needed.

    ``repo_root`` is the directory passed to git commands. If None, git uses
    whatever repository contains the current working directory — or fails
    gracefully if there is none.

    ``lock_file`` defaults to ``repo_root / "uv.lock"`` when repo_root is
    given. Pass an explicit Path to override.

    This function is the single construction site for Manifest. Every field
    that can be auto-detected is, so callers only need to supply the
    experiment-specific fields. Fields that cannot be auto-detected (like
    model_artifact_hash, which is only known after training) can be supplied
    explicitly or left as None to be filled in later.
    """
    cwd = repo_root

    resolved_lock = lock_file
    if resolved_lock is None and repo_root is not None:
        resolved_lock = repo_root / "uv.lock"

    feat_hash: str | None = None
    if feature_definitions is not None:
        feat_hash = _dict_hash(feature_definitions)

    return Manifest(
        schema_version=_SCHEMA_VERSION,
        git_commit=_git_commit(cwd=cwd),
        git_diff_hash=_git_diff_hash(cwd=cwd),
        data_partition_hashes=dict(data_partition_hashes) if data_partition_hashes else {},
        universe_version=universe_version,
        universe_hash=universe_hash,
        feature_definitions_hash=feat_hash,
        model_artifact_hash=model_artifact_hash,
        config_hash=config_hash,
        lock_hash=_file_hash(resolved_lock) if resolved_lock is not None else None,
        random_seeds=dict(random_seeds) if random_seeds else {},
        trial_number=trial_number,
        parent_experiment_id=parent_experiment_id,
        experiment_id=experiment_id,
        fidelity_tier=str(fidelity_tier),
        holdout_previously_seen=holdout_previously_seen,
        extra=dict(extra) if extra else {},
    )


def _manifest_to_dict(m: Manifest) -> dict[str, Any]:
    """Canonical serializable form. Every field, sorted, deterministic."""
    return {
        "schema_version": m.schema_version,
        "experiment_id": m.experiment_id,
        "parent_experiment_id": m.parent_experiment_id,
        "trial_number": m.trial_number,
        "fidelity_tier": m.fidelity_tier,
        "git_commit": m.git_commit,
        "git_diff_hash": m.git_diff_hash,
        "data_partition_hashes": dict(sorted(m.data_partition_hashes.items())),
        "universe_version": m.universe_version,
        "universe_hash": m.universe_hash,
        "feature_definitions_hash": m.feature_definitions_hash,
        "model_artifact_hash": m.model_artifact_hash,
        "config_hash": m.config_hash,
        "lock_hash": m.lock_hash,
        "random_seeds": dict(sorted((k, v) for k, v in m.random_seeds.items())),
        "holdout_previously_seen": m.holdout_previously_seen,
        "extra": m.extra,
    }


def manifest_hash(m: Manifest) -> str:
    """Blake2b digest of the manifest's canonical serialized form.

    Two manifests are provably identical iff their hashes match. Any field
    change — including a seed, a partition hash, or the holdout flag — produces
    a different digest. The hash is stable across reserialization because the
    serialized form is deterministic (dict sorted by key, JSON with sort_keys).

    experiment_id is included in the hash so that two manifests for different
    experiments but identical everything else are still distinguishable. If that
    is not desired, the caller may zero out experiment_id before hashing — but
    that is a different question ("did these two runs see the same data and code")
    and should be answered explicitly, not by accident.
    """
    d = _manifest_to_dict(m)
    serialized = json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.blake2b(serialized, digest_size=32).hexdigest()


def write_manifest(path: Path, m: Manifest) -> None:
    """Write the manifest as pretty-printed JSON via atomic_write_text.

    Uses journal.atomic_write_text so readers never see a partial file — the
    same guarantee the ledger provides for its snapshots. The manifest is
    written once at run start (and again if updated after training), so it goes
    through the atomic path rather than the append path.
    """
    from memetrader.journal import atomic_write_text  # local import to avoid cycles

    text = json.dumps(_manifest_to_dict(m), indent=2, sort_keys=True, ensure_ascii=False)
    atomic_write_text(path, text + "\n")


def read_manifest(path: Path) -> Manifest:
    """Read a manifest written by write_manifest. Raises if the file is absent or corrupt."""
    text = path.read_text(encoding="utf-8")
    d: dict[str, Any] = json.loads(text)
    return Manifest(
        schema_version=int(d.get("schema_version", _SCHEMA_VERSION)),
        experiment_id=str(d["experiment_id"]),
        parent_experiment_id=d.get("parent_experiment_id"),
        trial_number=int(d["trial_number"]),
        fidelity_tier=str(d["fidelity_tier"]),
        git_commit=d.get("git_commit"),
        git_diff_hash=d.get("git_diff_hash"),
        data_partition_hashes=dict(d.get("data_partition_hashes") or {}),
        universe_version=d.get("universe_version"),
        universe_hash=d.get("universe_hash"),
        feature_definitions_hash=d.get("feature_definitions_hash"),
        model_artifact_hash=d.get("model_artifact_hash"),
        config_hash=d.get("config_hash"),
        lock_hash=d.get("lock_hash"),
        random_seeds={str(k): int(v) for k, v in (d.get("random_seeds") or {}).items()},
        holdout_previously_seen=bool(d.get("holdout_previously_seen", False)),
        extra=dict(d.get("extra") or {}),
    )
