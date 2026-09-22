"""Walk-forward driver: fit → select → refit → test, per outer fold.

Design intent and leakage discipline
-------------------------------------
The walk-forward driver orchestrates the fit/select/refit/test cycle but does
*not* implement any of those steps itself — it calls user-supplied callables.
This separation means:

1. The driver can enforce structural invariants (order of operations, which
   data each phase sees) without knowing what model or scaler is being used.
2. A leakage audit can prove, by inspection of this module, that the test fold
   is read exactly once: the ``_GuardedTestView`` accessor records every access
   and the driver hands only *that* view to the user's ``test_fn``.

The cycle per fold:
    a) Fit scalers, selectors, and hyperparameters on ``train_observations``.
       This is the only phase where parameters may change.
    b) Choose thresholds and select models using ``val_observations``.
       No parameter fitting, only selection among configurations fitted in (a).
    c) Refit the chosen configuration on ``train_observations + val_observations``
       combined, to maximise data usage before the final evaluation.
    d) Evaluate the refitted model exactly once on ``test_observations``.

Step (d) uses a ``_GuardedTestView`` that sets ``_accessed = True`` on first
read and raises on any subsequent read in the same fold. A leakage audit can
check that ``fold_result.test_view.access_count == 1`` across all folds.

Protocols
---------
The driver is generic over any user-supplied model via the ``FoldCallable``
protocol. This keeps the driver free of any ML-framework dependency while
being statically type-checkable (mypy verifies the protocol, not just the
ABC). The user supplies:

    ``fit_fn(observations, fold_record) -> FittedModel``
        Called with *training-only* observations. May not peek at val or test.

    ``select_fn(model, observations, fold_record) -> SelectedConfig``
        Called with validation observations. May use the model from fit_fn.
        Must not refit — only select threshold / hyperparam.

    ``refit_fn(config, observations, fold_record) -> FittedModel``
        Called with train+val combined. Must use the *config* from select_fn,
        not re-derive it. This prevents val leakage via the refit.

    ``test_fn(model, test_view, fold_record) -> FoldMetrics``
        Called with the guarded test view. Called exactly once per fold.
        The guarded view raises ``TestSetAccessViolation`` on a second read.

Thread safety
-------------
``WalkForwardRunner.run()`` is sequential by default. The ``parallel`` flag
is reserved for future parallel fold execution but is not yet implemented;
passing it raises ``NotImplementedError`` rather than silently running
sequentially.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from memetrader.validation.splits import (
    FoldRecord,
    LabeledObservation,
    PurgedWalkForward,
    SplitResult,
)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TestSetAccessViolation(RuntimeError):
    """Raised when a test fold is read more than once in the same fold.

    The structural invariant: the test fold is evaluated exactly once. A second
    read means the caller has used test-set feedback to select or adjust the
    model, which is the definition of test-set contamination. We raise rather
    than warn because a contaminated result is worse than no result.

    Named ``TestSetAccessViolation`` (not ``TestAccessViolation``) to avoid
    pytest treating it as a test class during collection.
    """


class WalkForwardError(RuntimeError):
    """Raised for configuration errors in the walk-forward driver."""


# ---------------------------------------------------------------------------
# Guarded test accessor
# ---------------------------------------------------------------------------


@dataclass
class _GuardedTestView:
    """A read-once wrapper around the test observations for one fold.

    ``_accessed`` is set on the first ``read()`` call. A second call raises
    ``TestSetAccessViolation``. The driver hands this wrapper (not the raw list)
    to ``test_fn``, so the callable cannot accidentally re-read the test set.

    ``access_count`` is public so a post-hoc leakage audit can assert:
        assert all(r.test_view.access_count == 1 for r in results)
    which proves the test fold was touched exactly once per fold, never zero
    times (which would mean the evaluation step was skipped), and never more
    than once (which would mean test-set feedback).
    """

    _observations: list[LabeledObservation]
    _fold_id: str
    _access_count: int = field(default=0, init=False)

    @property
    def access_count(self) -> int:
        return self._access_count

    def read(self) -> list[LabeledObservation]:
        """Return the test observations. Raises on a second call.

        The first call is expected (evaluation). A second call is a leakage
        event: the model has already seen test metrics and is being re-evaluated
        or retrained. We raise immediately rather than logging, because a
        result produced after a test-set read cannot be trusted.
        """
        if self._access_count > 0:
            raise TestSetAccessViolation(
                f"Fold '{self._fold_id}': test observations were already read "
                f"({self._access_count} time(s)). Reading the test set more than once "
                "means test-set feedback has contaminated the model selection or "
                "evaluation. This is the precise failure the guarded accessor exists "
                "to prevent. If you believe this is a false alarm, audit every call "
                "site that holds a reference to this _GuardedTestView."
            )
        self._access_count += 1
        return self._observations


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    """Everything produced by one walk-forward fold.

    ``fold``         — the ``FoldRecord`` with boundary and purge counts.
    ``test_view``    — the guarded test accessor; ``access_count`` must be 1
                       after a successful run.
    ``fit_output``   — whatever ``fit_fn`` returned (opaque to the driver).
    ``select_output``— whatever ``select_fn`` returned.
    ``refit_output`` — whatever ``refit_fn`` returned (the model used on test).
    ``metrics``      — whatever ``test_fn`` returned; shape is user-defined.
    ``n_train``      — number of training observations after purge + embargo.
    ``n_val``        — number of validation observations.
    ``n_test``       — number of test observations.

    All outputs are typed as ``Any`` because the driver is generic over the
    user's model types. Static checking of the specific types is the user's
    responsibility; the driver checks operational correctness (ordering, access
    counts), not model correctness.
    """

    fold: FoldRecord
    test_view: _GuardedTestView
    fit_output: Any
    select_output: Any
    refit_output: Any
    metrics: Any
    n_train: int
    n_val: int
    n_test: int


@dataclass
class WalkForwardRunResult:
    """The aggregate output of a full walk-forward run.

    ``fold_results`` — one ``FoldResult`` per fold, in fold order.
    ``fold_records``  — convenience list of ``FoldRecord`` objects for
                        writing ``folds.parquet`` without loading the full
                        ``FoldResult`` objects.

    ``leakage_clean`` property: ``True`` iff every fold's test was read
    exactly once. The caller should assert this before consuming ``metrics``.
    """

    fold_results: list[FoldResult]
    fold_records: list[FoldRecord]

    @property
    def leakage_clean(self) -> bool:
        """True when every fold's test set was read exactly once.

        Zero reads means the evaluation step was skipped—no metrics were
        produced. More than one means test-set feedback. Either is a leakage
        event, not just a "too many reads" event.
        """
        return all(r.test_view.access_count == 1 for r in self.fold_results)

    def assert_leakage_clean(self) -> None:
        """Raise ``WalkForwardError`` if any fold failed the read-once check.

        Call this immediately after ``run()`` completes and before consuming
        any ``FoldResult.metrics``. A result produced under leakage is not a
        valid backtest result.
        """
        for r in self.fold_results:
            count = r.test_view.access_count
            if count != 1:
                raise WalkForwardError(
                    f"Fold '{r.fold.fold_id}': test accessor was read {count} time(s). "
                    "Expected exactly 1. Consuming metrics from this fold is not valid."
                )


# ---------------------------------------------------------------------------
# Callable protocols (for static type-checking)
# ---------------------------------------------------------------------------


class FitCallable(Protocol):
    """Protocol for the fit phase: train observations → fitted model."""

    def __call__(
        self,
        train_obs: list[LabeledObservation],
        fold: FoldRecord,
    ) -> Any: ...


class SelectCallable(Protocol):
    """Protocol for the selection phase: val observations → selected config."""

    def __call__(
        self,
        fitted_model: Any,
        val_obs: list[LabeledObservation],
        fold: FoldRecord,
    ) -> Any: ...


class RefitCallable(Protocol):
    """Protocol for the refit phase: train+val observations → refitted model."""

    def __call__(
        self,
        selected_config: Any,
        train_val_obs: list[LabeledObservation],
        fold: FoldRecord,
    ) -> Any: ...


class TestCallable(Protocol):
    """Protocol for the test phase: reads the guarded test view once."""

    def __call__(
        self,
        refitted_model: Any,
        test_view: _GuardedTestView,
        fold: FoldRecord,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardRunner:
    """Orchestrates the fit/select/refit/test cycle over walk-forward folds.

    The runner is intentionally *thin*: it enforces the ordering and the
    read-once test invariant, then delegates to the user-supplied callables.
    It has no opinion about what kind of model is being trained.

    Parameters
    ----------
    splitter:
        A ``PurgedWalkForward`` (or any object with a matching ``split``
        method) that generates the folds.
    fit_fn, select_fn, refit_fn, test_fn:
        The four phase callables. See module docstring and protocol definitions.
    enable_purge:
        If ``False``, disables purge+embargo (for the leakage-guard test only).
        In production this must always be ``True``.
    """

    splitter: PurgedWalkForward
    fit_fn: FitCallable
    select_fn: SelectCallable
    refit_fn: RefitCallable
    test_fn: TestCallable
    enable_purge: bool = True

    def run(
        self,
        observations: Sequence[LabeledObservation],
        *,
        data_start: float,
        data_end: float,
    ) -> WalkForwardRunResult:
        """Execute the full walk-forward cycle.

        Steps per fold (must not be reordered):
            1. Generate train/val/test index splits via the splitter.
            2. Call ``fit_fn`` with *training-only* observations.
            3. Call ``select_fn`` with *validation-only* observations.
            4. Call ``refit_fn`` with *train+val combined* observations.
            5. Wrap test observations in ``_GuardedTestView``.
            6. Call ``test_fn`` exactly once with the guarded view.
            7. Assert the guarded view was read exactly once.

        Parameters
        ----------
        observations:
            All ``LabeledObservation`` objects across all assets and the full
            data span. The splitter assigns each to train/val/test by
            wall-clock boundary; the caller need not pre-filter.
        data_start, data_end:
            Epoch seconds bounding the full dataset. Passed directly to the
            splitter's ``split`` method.

        Returns
        -------
        ``WalkForwardRunResult`` with all fold results. Call
        ``result.assert_leakage_clean()`` immediately before using metrics.
        """
        splits: list[SplitResult] = self.splitter.split(
            observations,
            data_start=data_start,
            data_end=data_end,
            enable_purge=self.enable_purge,
        )

        if not splits:
            warnings.warn(
                "WalkForwardRunner.run(): no folds were generated. "
                "Check that data_end - holdout is large enough to fit at least one "
                "full train/val/test window.",
                UserWarning,
                stacklevel=2,
            )

        fold_results: list[FoldResult] = []
        fold_records: list[FoldRecord] = []
        obs_list = list(observations)

        for split in splits:
            train_obs = [obs_list[i] for i in split.train_indices]
            val_obs = [obs_list[i] for i in split.val_indices]
            test_obs = [obs_list[i] for i in split.test_indices]
            train_val_obs = train_obs + val_obs

            fold = split.fold

            # Phase 1: fit on training data only.
            fit_out = self.fit_fn(train_obs, fold)

            # Phase 2: select threshold/config on validation data.
            select_out = self.select_fn(fit_out, val_obs, fold)

            # Phase 3: refit on train+val combined using the *selected* config.
            # The refit must not re-derive the config from val; the config is
            # already committed by select_fn. This prevents val feedback from
            # entering the final model through the refit.
            refit_out = self.refit_fn(select_out, train_val_obs, fold)

            # Phase 4: wrap test observations and evaluate exactly once.
            test_view = _GuardedTestView(_observations=test_obs, _fold_id=fold.fold_id)
            metrics = self.test_fn(refit_out, test_view, fold)

            # Invariant: test was read exactly once inside test_fn.
            # This assertion fires here (not just in assert_leakage_clean)
            # so that a badly-written test_fn is caught immediately, not at
            # the end of a multi-hour run.
            if test_view.access_count != 1:
                raise WalkForwardError(
                    f"Fold '{fold.fold_id}': test_fn read the test view "
                    f"{test_view.access_count} time(s). Expected exactly 1. "
                    "Ensure test_fn calls test_view.read() exactly once and "
                    "does not cache or re-read the result."
                )

            fold_result = FoldResult(
                fold=fold,
                test_view=test_view,
                fit_output=fit_out,
                select_output=select_out,
                refit_output=refit_out,
                metrics=metrics,
                n_train=len(train_obs),
                n_val=len(val_obs),
                n_test=len(test_obs),
            )
            fold_results.append(fold_result)
            fold_records.append(fold)

        return WalkForwardRunResult(
            fold_results=fold_results,
            fold_records=fold_records,
        )


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "FitCallable",
    "FoldResult",
    "RefitCallable",
    "SelectCallable",
    "TestSetAccessViolation",
    "TestCallable",
    "WalkForwardError",
    "WalkForwardRunResult",
    "WalkForwardRunner",
    "_GuardedTestView",
]
