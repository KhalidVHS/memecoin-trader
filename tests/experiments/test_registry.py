"""Registry: trial accounting, holdout access control, actor isolation.

Each test names the failure it guards against. The registry's job is to make
the anti-overfitting story real — these tests are what make that job verifiable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memetrader.experiments.registry import (
    HoldoutAccessError,
    PermissionError,
    Registry,
    TrialRegistration,
)

EXP = "exp-test-registry-001"


def _trial(
    trial_id: str, *, actor="human", revision_of=None, abandoned=False
) -> TrialRegistration:
    return TrialRegistration(
        trial_id=trial_id,
        experiment_id=EXP,
        actor=actor,
        feature_set=["rsi14", "ema9"],
        parameter_set={"lr": 0.01},
        universe_filter={"min_mcap": 1_000_000},
        model_name="xgb_v1",
        horizon="24h",
        exit_rule="trailing_stop_5pct",
        cost_assumption="taker_30bps",
        prompt="baseline",
        revision_of=revision_of,
        abandoned=abandoned,
    )


class TestTrialCount:
    """trial_count must include every registration, including revisions and abandoned ones.

    Undercounting trials is the primary way a research process fools the DSR.
    """

    def test_empty_registry_count_zero(self, tmp_path: Path) -> None:
        reg = Registry(tmp_path / "reg.jsonl")
        assert reg.trial_count(EXP) == 0

    def test_count_rises_with_each_registration(self, tmp_path: Path) -> None:
        reg = Registry(tmp_path / "reg.jsonl")
        reg.register_trial(_trial("t1"))
        assert reg.trial_count(EXP) == 1
        reg.register_trial(_trial("t2", revision_of="t1"))
        assert reg.trial_count(EXP) == 2

    def test_abandoned_trials_count(self, tmp_path: Path) -> None:
        """An abandoned trial still consumed a degree of freedom."""
        reg = Registry(tmp_path / "reg.jsonl")
        reg.register_trial(_trial("t1"))
        reg.register_trial(_trial("t2", abandoned=True))
        assert reg.trial_count(EXP) == 2

    def test_count_is_per_experiment(self, tmp_path: Path) -> None:
        """Trials in different experiments must not bleed into each other's count."""
        reg = Registry(tmp_path / "reg.jsonl")
        t = _trial("t1")
        reg.register_trial(t)
        other_t = TrialRegistration(
            trial_id="other-t1",
            experiment_id="other-exp",
            actor="human",
            feature_set=[],
            parameter_set={},
            universe_filter={},
            model_name="m",
            horizon="1h",
            exit_rule="none",
            cost_assumption="zero",
            prompt="",
        )
        reg.register_trial(other_t)
        assert reg.trial_count(EXP) == 1
        assert reg.trial_count("other-exp") == 1


class TestIdempotency:
    """Same row_id twice must produce one row in the file."""

    def test_duplicate_trial_id_is_one_row(self, tmp_path: Path) -> None:
        """Appending the same trial twice must not double-count it."""
        path = tmp_path / "reg.jsonl"
        reg = Registry(path)
        t = _trial("t-idem")
        first = reg.register_trial(t)
        second = reg.register_trial(t)
        assert first is True
        assert second is False
        assert reg.trial_count(EXP) == 1
        # Confirm the file has only one row
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 1

    def test_idempotency_survives_reload(self, tmp_path: Path) -> None:
        """After reloading from disk, duplicate suppression must still work."""
        path = tmp_path / "reg.jsonl"
        reg = Registry(path)
        reg.register_trial(_trial("t-reload"))
        # Create fresh instance from same file
        reg2 = Registry(path)
        wrote = reg2.register_trial(_trial("t-reload"))
        assert wrote is False
        assert reg2.trial_count(EXP) == 1


class TestHoldout:
    """Holdout access must be recorded, locked, and enforced in code."""

    def test_opening_holdout_is_recorded(self, tmp_path: Path) -> None:
        reg = Registry(tmp_path / "reg.jsonl")
        reg.register_trial(_trial("t1"))
        reg.open_holdout(EXP, actor="human", manifest_hash="abc123")
        result = reg.certify(EXP)
        assert result.holdout_opened is True

    def test_second_open_is_refused(self, tmp_path: Path) -> None:
        """The second open attempt must raise HoldoutAccessError.

        The holdout is no longer unseen after the first open. A second open
        can only happen after the researcher has already seen the result, so
        refusing it is not overly strict — it is exactly the right response.
        """
        reg = Registry(tmp_path / "reg.jsonl")
        reg.register_trial(_trial("t1"))
        reg.open_holdout(EXP, actor="human", manifest_hash="abc123")
        with pytest.raises(HoldoutAccessError):
            reg.open_holdout(EXP, actor="human", manifest_hash="abc123")

    def test_revision_after_holdout_flagged_and_certify_fails(self, tmp_path: Path) -> None:
        """certify() must fail and name the post-holdout trial.

        A researcher who opens the holdout and then registers a new trial has
        seen the holdout result before making a modelling decision. The trial
        is tagged post_holdout=True and certify() refuses to certify the lineage.
        """
        reg = Registry(tmp_path / "reg.jsonl")
        reg.register_trial(_trial("t1"))
        reg.open_holdout(EXP, actor="human", manifest_hash="abc123")
        # Registering after holdout open — must be flagged automatically
        reg.register_trial(_trial("t2", revision_of="t1"))
        result = reg.certify(EXP)
        assert result.passed is False
        # The reason must name the specific trial
        assert any("t2" in r for r in result.reasons)

    def test_certify_passes_before_holdout_opened(self, tmp_path: Path) -> None:
        reg = Registry(tmp_path / "reg.jsonl")
        reg.register_trial(_trial("t1"))
        result = reg.certify(EXP)
        assert result.passed is True
        assert result.holdout_opened is False

    def test_certify_passes_after_holdout_with_no_revisions(self, tmp_path: Path) -> None:
        """Opening the holdout without any subsequent revisions is a clean process."""
        reg = Registry(tmp_path / "reg.jsonl")
        reg.register_trial(_trial("t1"))
        reg.open_holdout(EXP, actor="human", manifest_hash="abc123")
        result = reg.certify(EXP)
        assert result.passed is True

    def test_certify_fails_with_no_trials(self, tmp_path: Path) -> None:
        reg = Registry(tmp_path / "reg.jsonl")
        result = reg.certify(EXP)
        assert result.passed is False
        assert result.trial_count == 0

    def test_llm_cannot_open_holdout(self, tmp_path: Path) -> None:
        """An LLM actor must be refused at the open_holdout call, not silently allowed."""
        reg = Registry(tmp_path / "reg.jsonl")
        reg.register_trial(_trial("t1"))
        with pytest.raises(PermissionError):
            reg.open_holdout(EXP, actor="llm", manifest_hash="abc123")

    def test_holdout_open_survives_reload(self, tmp_path: Path) -> None:
        """Holdout lock must be durable: a fresh Registry instance from the same
        file must still refuse a second open."""
        path = tmp_path / "reg.jsonl"
        reg = Registry(path)
        reg.register_trial(_trial("t1"))
        reg.open_holdout(EXP, actor="human", manifest_hash="abc123")
        reg2 = Registry(path)
        with pytest.raises(HoldoutAccessError):
            reg2.open_holdout(EXP, actor="human", manifest_hash="abc456")


class TestActorIsolation:
    """LLM actors must not be able to request views they are not permitted."""

    def test_llm_cannot_view_holdout(self, tmp_path: Path) -> None:
        reg = Registry(tmp_path / "reg.jsonl")
        with pytest.raises(PermissionError):
            reg.request_view(EXP, actor="llm", view="holdout")

    def test_llm_cannot_view_holdout_pnl(self, tmp_path: Path) -> None:
        reg = Registry(tmp_path / "reg.jsonl")
        with pytest.raises(PermissionError):
            reg.request_view(EXP, actor="llm", view="holdout_pnl")

    def test_llm_cannot_view_validation_pnl(self, tmp_path: Path) -> None:
        """LLMs must not see validation PnL — it would allow steering on out-of-sample."""
        reg = Registry(tmp_path / "reg.jsonl")
        with pytest.raises(PermissionError):
            reg.request_view(EXP, actor="llm", view="val_pnl")

    def test_llm_cannot_view_test_pnl(self, tmp_path: Path) -> None:
        reg = Registry(tmp_path / "reg.jsonl")
        with pytest.raises(PermissionError):
            reg.request_view(EXP, actor="llm", view="test_pnl")

    def test_llm_can_view_training_diagnostics(self, tmp_path: Path) -> None:
        """LLMs may read training-fold diagnostics — that is their permitted role."""
        reg = Registry(tmp_path / "reg.jsonl")
        # Must not raise
        reg.request_view(EXP, actor="llm", view="train_diagnostics")

    def test_llm_cannot_view_unknown_resource(self, tmp_path: Path) -> None:
        """An unknown view string is not whitelisted for LLMs."""
        reg = Registry(tmp_path / "reg.jsonl")
        with pytest.raises(PermissionError):
            reg.request_view(EXP, actor="llm", view="some_future_view")

    def test_automated_cannot_view_holdout(self, tmp_path: Path) -> None:
        reg = Registry(tmp_path / "reg.jsonl")
        with pytest.raises(PermissionError):
            reg.request_view(EXP, actor="automated", view="holdout")

    def test_automated_can_view_validation(self, tmp_path: Path) -> None:
        reg = Registry(tmp_path / "reg.jsonl")
        # Must not raise
        reg.request_view(EXP, actor="automated", view="validation")

    def test_human_can_view_holdout(self, tmp_path: Path) -> None:
        reg = Registry(tmp_path / "reg.jsonl")
        # Must not raise — humans may view anything
        reg.request_view(EXP, actor="human", view="holdout")


class TestLineage:
    """lineage() must reconstruct the revision chain correctly."""

    def test_single_trial_lineage(self, tmp_path: Path) -> None:
        reg = Registry(tmp_path / "reg.jsonl")
        reg.register_trial(_trial("t1"))
        assert reg.lineage("t1") == ["t1"]

    def test_chain_lineage(self, tmp_path: Path) -> None:
        reg = Registry(tmp_path / "reg.jsonl")
        reg.register_trial(_trial("t1"))
        reg.register_trial(_trial("t2", revision_of="t1"))
        reg.register_trial(_trial("t3", revision_of="t2"))
        assert reg.lineage("t3") == ["t1", "t2", "t3"]

    def test_unknown_trial_lineage_empty(self, tmp_path: Path) -> None:
        reg = Registry(tmp_path / "reg.jsonl")
        assert reg.lineage("nonexistent") == []


class TestNoGit:
    """Registry must work in a tmp_path with no git repository."""

    def test_register_in_no_git_dir(self, tmp_path: Path) -> None:
        reg = Registry(tmp_path / "reg.jsonl")
        reg.register_trial(_trial("t1"))
        assert reg.trial_count(EXP) == 1

    def test_certify_in_no_git_dir(self, tmp_path: Path) -> None:
        reg = Registry(tmp_path / "reg.jsonl")
        reg.register_trial(_trial("t1"))
        result = reg.certify(EXP)
        assert result.passed is True
