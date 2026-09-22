"""Append-only experiment registry — the governance layer for anti-overfitting.

This module enforces the properties that make a Deflated Sharpe Ratio (DSR)
calculation meaningful. The DSR's whole point is to correct for the number of
independent strategy variations that were tried. Undercounting trials is the
single most common way a research process fools itself: a researcher who tries
200 parameter sets, reports the best, and claims trial_count=1 has produced a
number that looks like evidence but is not.

The registry makes undercounting hard by accident:

**Every trial is registered.** Feature set, parameter set, universe filter,
model, horizon, exit rule, cost assumption, prompt, and every manual or
LLM-suggested revision. A trial that was tried and abandoned still happened and
still consumed a degree of freedom. The DSR denominator is the count of all
registered trials in a lineage, including abandoned ones.

**Holdout access is recorded and locked.** The holdout data may be inspected
exactly once. The moment it is opened, the registry writes a ``holdout_opened``
row and refuses to certify any subsequent revision in that lineage. If the model
is revised after seeing the holdout, it is no longer a holdout — a new,
unseen one is required. This is enforced in code: ``certify()`` inspects the
row sequence and returns a failing result with an explicit reason string if it
finds any revision after ``holdout_opened``.

**LLM actor isolation.** An LLM may propose hypotheses, generate code, read
training-fold diagnostics, and help investigate failures — all of which are
legitimate uses that do not consume holdout degrees of freedom. It may NOT
inspect final-holdout results, alter the strategy in response to holdout losses,
or select a prompt after seeing test PnL. The ``actor`` field (``human`` /
``llm`` / ``automated``) on every row, combined with ``permitted_views`` checks,
makes the LLM's permitted operations explicit and machine-enforced rather than
relying on a convention that a tired researcher might accidentally violate.

Storage mirrors ``journal.py`` exactly: append-only JSONL, ``schema_version`` +
``row_id`` + ``kind`` + ``ts`` in every envelope, idempotent appends keyed on
``row_id``, reusing ``journal.append`` and ``journal.atomic_write_text``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from memetrader.journal import ENCODING, append, scan

__all__ = [
    "Actor",
    "CertifyResult",
    "HoldoutAccessError",
    "PermissionError",
    "Registry",
    "RegistryRow",
    "TrialRegistration",
]


_SCHEMA_VERSION = 1

# Actors that may request a holdout view. LLMs may never request one; that
# would undermine the isolation this registry exists to enforce.
_HOLDOUT_PERMITTED_ACTORS: frozenset[str] = frozenset({"human", "automated"})

Actor = Literal["human", "llm", "automated"]
RowKind = Literal["trial", "holdout_opened", "view_request"]


class HoldoutAccessError(RuntimeError):
    """Raised when the holdout has already been opened or cannot be opened.

    Raised on the second open attempt, not on the first — the first is correct
    and expected. The caller catching this error is encountering the guard that
    prevents silent re-use of a seen holdout as if it were still unseen.
    """


class PermissionError(RuntimeError):  # noqa: A001
    """Raised when an actor requests a view it is not permitted to make.

    The name shadows the builtin deliberately: this is a domain-level permission
    denial, not an OS-level one, and callers in this package should be catching
    this type, not the builtin.
    """


# ---------------------------------------------------------------------------
# Row dataclasses (in-memory representations, not stored directly)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrialRegistration:
    """Everything that defines one candidate strategy configuration.

    All fields that vary between trials must be captured here — a field omitted
    is a dimension along which two "different" trials look identical to the DSR
    calculation, which means their degrees of freedom are silently merged. When
    in doubt, add a field.

    ``revision_of`` links this trial to the one it was derived from, building
    the lineage graph. A trial that was tried and then abandoned is still a leaf
    in that graph and still counts toward trial_count.

    ``abandoned`` marks trials that were discarded without producing a result.
    Abandoned trials still count: the researcher saw enough of the training
    behaviour to decide this direction was unpromising, which is information
    about the search space.
    """

    trial_id: str
    experiment_id: str
    actor: Actor
    # The configuration axes that vary between trials
    feature_set: list[str]
    parameter_set: dict[str, Any]
    universe_filter: dict[str, Any]
    model_name: str
    horizon: str
    exit_rule: str
    cost_assumption: str
    prompt: str
    # Lineage
    revision_of: str | None = None
    abandoned: bool = False
    # Free-form note about why this trial was created or abandoned
    note: str = ""


@dataclass(frozen=True, slots=True)
class RegistryRow:
    """The envelope every row is stored in. Mirrors journal.py's _row_envelope."""

    schema_version: int
    row_id: str
    kind: RowKind
    ts: float
    experiment_id: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CertifyResult:
    """The outcome of certify(), with explicit reasons for any failure.

    A passing result means: the holdout was opened at most once, no revisions
    were registered after it was opened, and trial_count is non-zero. A failing
    result carries a list of human-readable reasons, each naming the specific
    row or condition that caused the failure.

    The reasons are strings rather than error codes so they can be pasted
    directly into a research log without translation.
    """

    passed: bool
    trial_count: int
    holdout_opened: bool
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class Registry:
    """Append-only experiment registry with holdout access control.

    One Registry instance manages one JSONL file. Multiple experiments can share
    the same file (they are distinguished by ``experiment_id`` in every row), or
    each experiment can have its own file — the public API is the same either way.

    The in-memory ``_seen`` set mirrors journal.Ledger._seen: built by scanning
    the file at construction, maintained by appending. One writer, no concurrent
    access — the same contract as the Ledger.
    """

    def __init__(self, path: Path, *, fsync: bool = True) -> None:
        self.path = Path(path)
        self.fsync = fsync
        result = scan(self.path)
        self._rows: list[dict[str, Any]] = list(result.rows)
        self._seen: set[str] = {
            r["row_id"] for r in self._rows if isinstance(r.get("row_id"), str)
        }

    # -- Internal append -----------------------------------------------

    def _append(self, row: dict[str, Any]) -> bool:
        """Write row unless its row_id is already present. Returns True if written.

        Idempotency contract: same row_id twice → one row in the file. This
        mirrors journal.Ledger._append_row and is the only way a JSONL file can
        enforce a uniqueness constraint without a database.
        """
        row_id = row.get("row_id")
        if isinstance(row_id, str) and row_id in self._seen:
            return False
        append(self.path, row, fsync=self.fsync)
        self._rows.append(row)
        if isinstance(row_id, str):
            self._seen.add(row_id)
        return True

    def _make_envelope(
        self, *, row_id: str, kind: RowKind, experiment_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "row_id": row_id,
            "kind": kind,
            "ts": time.time(),
            "experiment_id": experiment_id,
            "payload": payload,
        }

    # -- Public API ----------------------------------------------------

    def register_trial(self, reg: TrialRegistration) -> bool:
        """Record one trial. Returns True if newly written, False if duplicate.

        Every trial — including revisions and abandoned ones — must be
        registered before any computation begins. A trial registered after the
        fact (to retroactively lower trial_count) is indistinguishable from a
        new trial, and the registry does not distinguish them — it just appends.
        The honesty constraint is on the researcher, not on this function.

        The ``post_holdout`` flag is set automatically: if the holdout has
        already been opened for this experiment, this registration is tagged
        so that ``certify()`` can identify it as a violation.
        """
        experiment_id = reg.experiment_id
        post_holdout = self._holdout_was_opened(experiment_id)
        payload: dict[str, Any] = {
            "trial_id": reg.trial_id,
            "experiment_id": experiment_id,
            "actor": reg.actor,
            "feature_set": list(reg.feature_set),
            "parameter_set": dict(reg.parameter_set),
            "universe_filter": dict(reg.universe_filter),
            "model_name": reg.model_name,
            "horizon": reg.horizon,
            "exit_rule": reg.exit_rule,
            "cost_assumption": reg.cost_assumption,
            "prompt": reg.prompt,
            "revision_of": reg.revision_of,
            "abandoned": reg.abandoned,
            "note": reg.note,
            "post_holdout": post_holdout,
        }
        row = self._make_envelope(
            row_id=reg.trial_id,
            kind="trial",
            experiment_id=experiment_id,
            payload=payload,
        )
        return self._append(row)

    def open_holdout(
        self,
        experiment_id: str,
        *,
        actor: Actor,
        manifest_hash: str,
        note: str = "",
    ) -> None:
        """Record the one-time opening of the holdout partition.

        After this call, ``certify()`` will fail for any revision registered
        against this experiment_id. The record includes who opened the holdout,
        when, and under which manifest hash — so the audit trail is complete.

        Raises HoldoutAccessError if the holdout has already been opened for
        this experiment. Opening it twice is refused because the second open
        can only happen after the researcher has seen the holdout result, at
        which point the holdout is no longer unseen, and any comparison that
        treats it as unseen is false.
        """
        if actor == "llm":
            raise PermissionError(
                f"actor='llm' may not open the holdout for experiment {experiment_id!r}. "
                "An LLM is permitted to read training-fold diagnostics and propose "
                "hypotheses, but inspecting final holdout results violates the isolation "
                "that makes the holdout's evidence value non-zero. If an LLM needs to "
                "assist with holdout analysis, a human actor must open it."
            )
        if self._holdout_was_opened(experiment_id):
            # Find the original open row to report when it happened.
            opened_at = self._holdout_open_ts(experiment_id)
            raise HoldoutAccessError(
                f"Holdout for experiment {experiment_id!r} has already been opened "
                f"(at t={opened_at}). Opening it a second time is refused: the "
                "holdout is no longer unseen, and any revision made after the first "
                "open cannot be claimed as pre-holdout. If you need a fresh holdout, "
                "collect new prospective data and start a new experiment."
            )
        row_id = f"holdout:{experiment_id}"
        payload: dict[str, Any] = {
            "experiment_id": experiment_id,
            "actor": actor,
            "manifest_hash": manifest_hash,
            "note": note,
        }
        row = self._make_envelope(
            row_id=row_id,
            kind="holdout_opened",
            experiment_id=experiment_id,
            payload=payload,
        )
        self._append(row)

    def request_view(
        self,
        experiment_id: str,
        *,
        actor: Actor,
        view: str,
    ) -> None:
        """Record an actor's request to view a result, and enforce permissions.

        Permitted views by actor:
        - ``human``: all views, including holdout results.
        - ``automated``: training and validation fold metrics only.
        - ``llm``: training fold diagnostics only (not validation PnL, not holdout).

        The permitted-views check is enforced here rather than documented in a
        comment, because a comment that is not code is a comment that will be
        ignored at the worst possible moment — when a convenient shortcut is
        available.

        Raises PermissionError if the actor is not permitted to view the
        requested resource. The error message names the actor, the view, and
        the reason so the researcher understands what the constraint is without
        reading the source code.
        """
        _holdout_views = frozenset({"holdout", "holdout_pnl", "holdout_results"})
        _val_views = frozenset({"validation", "val_pnl", "val_metrics", "test_pnl"})
        _train_views = frozenset({"training", "train_metrics", "train_diagnostics"})

        if actor == "llm":
            # LLMs may read training diagnostics. They may not read validation
            # PnL, test PnL, or holdout results — any of those would allow an
            # LLM to steer parameter selection based on out-of-sample performance,
            # which is precisely the overfitting pathway this registry guards.
            if view in _holdout_views or view in _val_views:
                raise PermissionError(
                    f"actor='llm' is not permitted to view {view!r} for experiment "
                    f"{experiment_id!r}. LLMs may read training-fold diagnostics and "
                    "propose hypotheses, but not validation PnL, test PnL, or holdout "
                    "results — those views would allow steering based on out-of-sample "
                    "performance, consuming holdout degrees of freedom silently."
                )
            if view not in _train_views:
                raise PermissionError(
                    f"actor='llm' may only request views in {sorted(_train_views)!r}. "
                    f"Requested: {view!r}."
                )

        elif actor == "automated":
            if view in _holdout_views:
                raise PermissionError(
                    f"actor='automated' is not permitted to view holdout results "
                    f"({view!r}) for experiment {experiment_id!r}. "
                    "Automated systems may access training and validation metrics "
                    "but holdout access requires a human actor."
                )

        # actor == "human": all views are permitted.

        row_id = f"view:{experiment_id}:{actor}:{view}:{time.time()}"
        payload: dict[str, Any] = {
            "experiment_id": experiment_id,
            "actor": actor,
            "view": view,
        }
        row = self._make_envelope(
            row_id=row_id,
            kind="view_request",
            experiment_id=experiment_id,
            payload=payload,
        )
        self._append(row)

    # -- Queries -------------------------------------------------------

    def trial_count(self, experiment_id: str) -> int:
        """Count of all registered trials in this experiment, including abandoned ones.

        This is the number that the DSR denominator should use. Every trial
        that was attempted — regardless of whether it was finished, abandoned,
        or superseded — consumed a degree of freedom by virtue of the researcher
        having looked at training behaviour and decided what to try next.
        """
        return sum(
            1
            for r in self._rows
            if r.get("kind") == "trial"
            and r.get("experiment_id") == experiment_id
        )

    def lineage(self, trial_id: str) -> list[str]:
        """Ancestor trial_ids from oldest to this one, inclusive.

        Follows ``revision_of`` links to reconstruct the chain. A trial with
        no ``revision_of`` is the root; the returned list starts there. Cycles
        are broken after 1000 steps (they indicate a bug in how trials were
        registered, not a legitimate research lineage).
        """
        by_id: dict[str, dict[str, Any]] = {}
        for r in self._rows:
            if r.get("kind") == "trial":
                p = r.get("payload", {})
                tid = p.get("trial_id")
                if isinstance(tid, str):
                    by_id[tid] = p

        chain: list[str] = []
        current: str | None = trial_id
        seen_in_chain: set[str] = set()
        for _ in range(1000):
            if current is None or current not in by_id:
                break
            if current in seen_in_chain:
                break  # cycle guard
            seen_in_chain.add(current)
            chain.append(current)
            current = by_id[current].get("revision_of")

        chain.reverse()
        return chain

    def certify(self, experiment_id: str) -> CertifyResult:
        """Return a pass/fail certification for this experiment's research process.

        Fails with explicit reasons if:
        - trial_count is zero (no trials registered — nothing to certify).
        - A revision was registered after the holdout was opened (post_holdout
          flag set on any trial row). The reason names the trial_id so the
          researcher can find the offending row.
        - The holdout was opened more than once (structurally impossible via
          open_holdout, but checked defensively in case of manual file edits).

        A passing result does NOT certify that the research process was good —
        it certifies that the recorded process did not violate the structural
        rules this registry can enforce. Certify is a necessary condition, not
        a sufficient one.
        """
        reasons: list[str] = []

        count = self.trial_count(experiment_id)
        if count == 0:
            reasons.append(
                f"No trials registered for experiment {experiment_id!r}. "
                "Register at least one trial before certifying."
            )

        holdout_open_count = sum(
            1
            for r in self._rows
            if r.get("kind") == "holdout_opened"
            and r.get("experiment_id") == experiment_id
        )
        holdout_opened = holdout_open_count > 0

        if holdout_open_count > 1:
            reasons.append(
                f"Holdout for {experiment_id!r} was opened {holdout_open_count} times. "
                "Only one opening is permitted; subsequent openings suggest manual file "
                "manipulation."
            )

        # Find post-holdout revisions — trials registered after the holdout was
        # opened. These invalidate the holdout as evidence because the researcher
        # had access to holdout-derived information when deciding what to try next.
        post_holdout_trials: list[str] = []
        for r in self._rows:
            if r.get("kind") == "trial" and r.get("experiment_id") == experiment_id:
                p = r.get("payload", {})
                if p.get("post_holdout"):
                    trial_id = p.get("trial_id", "<unknown>")
                    post_holdout_trials.append(str(trial_id))

        if post_holdout_trials:
            names = ", ".join(post_holdout_trials)
            reasons.append(
                f"The following trials were registered after the holdout was opened "
                f"for experiment {experiment_id!r}: {names}. A model revised after "
                "seeing the holdout is no longer being evaluated on unseen data — the "
                "holdout has been consumed. A new prospective holdout is required to "
                "make a fresh claim."
            )

        return CertifyResult(
            passed=len(reasons) == 0,
            trial_count=count,
            holdout_opened=holdout_opened,
            reasons=reasons,
        )

    # -- Internal helpers ----------------------------------------------

    def _holdout_was_opened(self, experiment_id: str) -> bool:
        return any(
            r.get("kind") == "holdout_opened" and r.get("experiment_id") == experiment_id
            for r in self._rows
        )

    def _holdout_open_ts(self, experiment_id: str) -> float | None:
        for r in self._rows:
            if r.get("kind") == "holdout_opened" and r.get("experiment_id") == experiment_id:
                ts = r.get("ts")
                return float(ts) if ts is not None else None
        return None

    def reload(self) -> None:
        """Re-scan the file and rebuild the in-memory index.

        Call this if you suspect another process has written to the registry
        file (e.g. in a multi-process research pipeline). Under normal use,
        the in-memory state is always consistent with the file because the
        Registry is the sole writer and updates both on every append.
        """
        result = scan(self.path)
        self._rows = list(result.rows)
        self._seen = {r["row_id"] for r in self._rows if isinstance(r.get("row_id"), str)}


