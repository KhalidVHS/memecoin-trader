"""Recursive / expanding-window refit — the model only ever sees the past.

BACKTEST-CONTRACTS.md §7: "the model must be refit only on data available at
each decision point." This module provides:

* :class:`ExpandingWindowSchedule` / :class:`RollingWindowSchedule` — generate
  the sequence of ``[train_start, train_end)`` windows and their
  ``available_time`` cutoffs. Both are pure wall-clock arithmetic, matching
  ``validation.splits``' interval-aware convention: no row counts, no
  per-asset logic.

* :class:`RecursiveRefitter` — drives a caller-supplied ``fit_fn`` through a
  schedule. Every refit is stamped with the exact cutoff used
  (:class:`RefitRecord`), so a run is auditable after the fact: given the
  record, a reviewer can answer "what was visible when this model was
  fitted?" without re-running anything.

* :func:`hash_model_state` — a stable content hash of a fitted model's state,
  used to prove that two runs seeded from the same
  ``SimulatedClock.deterministic_seed`` produce byte-identical output, and
  that two different seeds do not.

This module does not implement a model. ``fit_fn`` is supplied by the caller
and may wrap anything — a linear model, a gradient booster, a lookup table —
as long as it returns a state object built only from ``hash_model_state``'s
supported primitives (bytes, ``numpy.ndarray``, and JSON-safe scalars/
containers).

Hyperparameter retuning inside the refit loop consumes a degree of freedom
exactly like any other trial (BACKTEST-CONTRACTS.md's DSR discussion in
``multiple_testing.py``). This module does not write to the experiments
registry — ``experiments/registry.py`` is out of scope here — but every
:class:`RefitRecord` carries ``is_trial`` and ``hyperparameters`` so the
caller can build a ``TrialRegistration`` from it without recomputing anything.
:func:`trial_count` totals the refits that were marked as trials, which is the
number a caller should fold into their own registry bookkeeping.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from memetrader.backtest.clock import SimulatedClock

# ---------------------------------------------------------------------------
# Refit schedules
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RefitPoint:
    """One scheduled refit: the training window and its available_time cutoff.

    ``cutoff`` is the exact ``available_time`` boundary the refit must not see
    past — it equals ``train_end`` by construction, kept as its own field so
    downstream code names it the way BACKTEST-CONTRACTS.md §1 does ("every
    component may access only records whose available_time <= t") rather than
    reaching into ``train_end`` and hoping the reader makes the connection.
    """

    cutoff: float
    train_start: float
    train_end: float

    def __post_init__(self) -> None:
        for name, value in (
            ("cutoff", self.cutoff),
            ("train_start", self.train_start),
            ("train_end", self.train_end),
        ):
            if not math.isfinite(value):
                raise ValueError(f"RefitPoint.{name} must be finite, got {value!r}")
        if self.train_end <= self.train_start:
            raise ValueError(
                f"RefitPoint.train_end {self.train_end} must be after "
                f"train_start {self.train_start}"
            )
        if self.cutoff != self.train_end:
            raise ValueError(
                f"RefitPoint.cutoff {self.cutoff} must equal train_end {self.train_end}"
            )


@dataclass(frozen=True, slots=True)
class ExpandingWindowSchedule:
    """Refit points whose training window always starts at ``data_start``.

    Each successive refit sees strictly more history than the last — the
    "recursive" scheme in the classic walk-forward literature. The first
    refit point is at ``data_start + min_train_secs``; subsequent points are
    spaced ``refit_interval_secs`` apart, stopping once a point would exceed
    ``data_end``.
    """

    data_start: float
    data_end: float
    min_train_secs: float
    refit_interval_secs: float

    def __post_init__(self) -> None:
        if self.data_end <= self.data_start:
            raise ValueError(
                f"data_end {self.data_end} must be after data_start {self.data_start}"
            )
        if self.min_train_secs <= 0.0:
            raise ValueError(f"min_train_secs must be > 0, got {self.min_train_secs}")
        if self.refit_interval_secs <= 0.0:
            raise ValueError(
                f"refit_interval_secs must be > 0, got {self.refit_interval_secs}"
            )

    def refit_points(self) -> list[RefitPoint]:
        points: list[RefitPoint] = []
        cutoff = self.data_start + self.min_train_secs
        while cutoff <= self.data_end:
            points.append(
                RefitPoint(cutoff=cutoff, train_start=self.data_start, train_end=cutoff)
            )
            cutoff += self.refit_interval_secs
        return points


@dataclass(frozen=True, slots=True)
class RollingWindowSchedule:
    """Refit points whose training window has a fixed maximum width.

    Unlike :class:`ExpandingWindowSchedule`, old history falls out of the
    window as new history enters — useful when the researcher believes the
    data-generating process drifts and stale history should stop informing
    the model. ``window_secs`` bounds ``train_end - train_start``; the first
    few windows may be shorter than ``window_secs`` if ``data_start +
    window_secs`` would otherwise require data before ``data_start``, but this
    schedule never does that — the first cutoff is at ``data_start +
    window_secs`` exactly, so every window is full-width by construction.
    """

    data_start: float
    data_end: float
    window_secs: float
    refit_interval_secs: float

    def __post_init__(self) -> None:
        if self.data_end <= self.data_start:
            raise ValueError(
                f"data_end {self.data_end} must be after data_start {self.data_start}"
            )
        if self.window_secs <= 0.0:
            raise ValueError(f"window_secs must be > 0, got {self.window_secs}")
        if self.refit_interval_secs <= 0.0:
            raise ValueError(
                f"refit_interval_secs must be > 0, got {self.refit_interval_secs}"
            )

    def refit_points(self) -> list[RefitPoint]:
        points: list[RefitPoint] = []
        cutoff = self.data_start + self.window_secs
        while cutoff <= self.data_end:
            train_start = cutoff - self.window_secs
            points.append(
                RefitPoint(cutoff=cutoff, train_start=train_start, train_end=cutoff)
            )
            cutoff += self.refit_interval_secs
        return points


# ---------------------------------------------------------------------------
# Deterministic model-state hashing
# ---------------------------------------------------------------------------


def _encode_state(value: object) -> bytes:
    """Recursively encode a model-state value into a canonical byte string.

    Supported types: ``bytes``, ``numpy.ndarray``, ``bool``, ``int``,
    ``float``, ``str``, ``None``, ``list``/``tuple`` and ``dict`` of the same
    (recursively). Anything else raises ``TypeError`` rather than falling
    back to ``repr()`` or ``id()`` — an unsupported type hashing successfully
    by object identity would make two runs "look" non-identical (or, worse,
    identical) for reasons unrelated to the model state itself.

    Dict keys are sorted by ``repr`` before encoding so insertion order never
    affects the hash — two model states built by populating the same dict in
    a different order must hash identically.
    """
    if isinstance(value, bytes):
        return b"bytes:" + value
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        return (
            b"ndarray:" + str(arr.dtype).encode() + str(arr.shape).encode() + arr.tobytes()
        )
    if isinstance(value, bool):
        return f"bool:{value}".encode()
    if isinstance(value, int):
        return f"int:{value}".encode()
    if isinstance(value, float):
        return f"float:{value!r}".encode()
    if value is None:
        return b"none"
    if isinstance(value, str):
        return b"str:" + value.encode()
    if isinstance(value, (list, tuple)):
        parts = [f"seq:{len(value)}:".encode()]
        parts.extend(_encode_state(item) for item in value)
        return b"".join(parts)
    if isinstance(value, dict):
        parts = [f"map:{len(value)}:".encode()]
        for key in sorted(value.keys(), key=repr):
            parts.append(_encode_state(key))
            parts.append(_encode_state(value[key]))
        return b"".join(parts)
    raise TypeError(
        f"hash_model_state cannot hash {type(value)!r}; supported types are bytes, "
        "numpy.ndarray, bool, int, float, str, None, list, tuple, dict (recursively). "
        "Serialize custom model objects to one of these before returning them from fit_fn."
    )


def hash_model_state(state: object) -> str:
    """Stable SHA-256 hex digest of a fitted model's state.

    Two refits that used the same seed and the same training data must
    produce a state with an identical hash; two refits with different seeds
    (and any seed-sensitive fitting procedure) must not. This is the
    mechanical check behind "same seed -> byte-identical model state."
    """
    return hashlib.sha256(_encode_state(state)).hexdigest()


# ---------------------------------------------------------------------------
# Refit records and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RefitRecord:
    """Audit stamp for one refit — enough to answer "what could this see?"
    without re-running the backtest.

    ``cutoff`` is the ``available_time`` boundary honored for this refit
    (BACKTEST-CONTRACTS.md §1). ``seed`` is the exact integer drawn from
    ``SimulatedClock.deterministic_seed`` for this refit's named stream.
    ``model_state_hash`` is :func:`hash_model_state` applied to whatever
    ``fit_fn`` returned, so two runs can be compared without carrying the
    model objects themselves around.

    ``is_trial`` / ``hyperparameters`` / ``trial_note`` are not written to
    ``experiments/registry.py`` by this module — that module is owned
    elsewhere — but they carry exactly the fields a caller needs to build a
    ``TrialRegistration`` (feature_set/parameter_set/etc. still come from the
    caller's own config) whenever a refit retuned hyperparameters rather than
    reusing a pre-registered configuration.
    """

    refit_id: str
    cutoff: float
    train_start: float
    train_end: float
    seed: int
    model_state_hash: str
    hyperparameters: dict[str, object] = field(default_factory=dict)
    is_trial: bool = False
    trial_note: str = ""


@dataclass(slots=True)
class RefitResult:
    """One refit's audit record plus the live model state it produced.

    Kept separate from ``RefitRecord`` because the record is the
    JSON-serializable, parquet-row-shaped artifact (mirrors ``FoldRecord`` in
    ``validation/splits.py``), while ``model_state`` may be an arbitrary
    Python object the caller still needs for inference.
    """

    record: RefitRecord
    model_state: object


TrainDataProvider = Callable[[float, float], object]
"""``(train_start, train_end) -> train_data``. Must return data drawn only
from ``[train_start, train_end)`` — see :func:`state_bound_data_provider` for
a ``PointInTimeState``-backed implementation that enforces this by construction."""

FitFn = Callable[[object, int], object]
"""``(train_data, seed) -> model_state``. Must be a deterministic function of
its two arguments for the determinism guarantee to hold; any wall-clock or
unseeded randomness inside ``fit_fn`` breaks it, the same way a stray
``time.time()`` breaks ``SimulatedClock`` (see ``backtest/clock.py``)."""


# ---------------------------------------------------------------------------
# RecursiveRefitter
# ---------------------------------------------------------------------------


class RecursiveRefitter:
    """Drives a refit schedule, stamping every refit with an auditable record.

    One instance per backtest run. ``clock`` supplies
    ``deterministic_seed(stream_name)`` for each refit's RNG seed — per-stream,
    not shared, for the same reason ``backtest/clock.py`` gives every RNG use
    its own named stream: drawing from a shared sequence means adding one
    refit shifts the seed of every subsequent one, which destroys the ability
    to compare two runs "on the same seed".
    """

    def __init__(self, *, clock: SimulatedClock, stream_prefix: str = "refit") -> None:
        self._clock = clock
        self._stream_prefix = stream_prefix

    def seed_for(self, refit_id: str) -> int:
        """The deterministic seed this refitter would use for ``refit_id``.

        Exposed publicly so a caller (or a test) can pre-compute the seed for
        a given refit_id without running the schedule, e.g. to seed a
        baseline model to compare against.
        """
        return self._clock.deterministic_seed(f"{self._stream_prefix}:{refit_id}")

    def run(
        self,
        schedule: Sequence[RefitPoint],
        *,
        data_provider: TrainDataProvider,
        fit_fn: FitFn,
        hyperparameters: dict[str, object] | None = None,
        is_trial: bool = False,
        trial_note: str = "",
    ) -> list[RefitResult]:
        """Execute every point in ``schedule`` in order, earliest cutoff first.

        For each point: fetch training data via ``data_provider(train_start,
        train_end)``, fit via ``fit_fn(train_data, seed)`` where ``seed`` is
        derived deterministically from this refitter's clock and the refit's
        id, then stamp a :class:`RefitRecord`.

        ``hyperparameters``/``is_trial``/``trial_note`` are recorded verbatim
        on every resulting record — callers that retune hyperparameters
        per-refit should call ``run`` once per distinct hyperparameter set
        (or per-point, building a fresh ``schedule`` of length 1 in a loop)
        rather than trying to vary them within a single call, so each trial
        is a distinct, auditable event.
        """
        results: list[RefitResult] = []
        for i, point in enumerate(schedule):
            refit_id = f"refit_{i:04d}_{int(point.cutoff)}"
            seed = self.seed_for(refit_id)
            train_data = data_provider(point.train_start, point.train_end)
            model_state = fit_fn(train_data, seed)
            record = RefitRecord(
                refit_id=refit_id,
                cutoff=point.cutoff,
                train_start=point.train_start,
                train_end=point.train_end,
                seed=seed,
                model_state_hash=hash_model_state(model_state),
                hyperparameters=dict(hyperparameters or {}),
                is_trial=is_trial,
                trial_note=trial_note,
            )
            results.append(RefitResult(record=record, model_state=model_state))
        return results


def trial_count(results: Sequence[RefitResult]) -> int:
    """Count of refits marked ``is_trial=True`` — the number to add to the
    caller's own experiment-registry trial count.

    Retuning hyperparameters inside a refit loop is itself a trial (§7 /
    ``multiple_testing.py``'s trial_count discussion): this function is the
    bookkeeping hook so that count is never silently dropped on the floor
    between the refit loop and the registry.
    """
    return sum(1 for r in results if r.record.is_trial)


# ---------------------------------------------------------------------------
# PointInTimeState-backed data provider
# ---------------------------------------------------------------------------


def state_bound_data_provider(
    state: object,
    *,
    asset_id: str,
    timeframe: object,
    lookback: int,
) -> TrainDataProvider:
    """Build a :data:`TrainDataProvider` backed by a ``PointInTimeState``.

    Advances ``state.now`` to ``train_end`` before reading bars, then filters
    to ``ts >= train_start``. Because ``state.bars`` itself enforces the
    point-in-time invariant (BACKTEST-CONTRACTS.md §1 / ``histdata.
    point_in_time.ReplayState``), the returned data cannot contain anything
    with ``available_time > train_end`` — the guarantee comes from the state
    object, not from this function re-implementing the check.

    ``state`` is typed as ``object`` (not the ``PointInTimeState`` protocol)
    to avoid importing ``histdata`` from ``validation``; it only needs a
    mutable ``now`` attribute and a ``bars`` method, exactly like
    ``leakage.PointInTimeStateLike``.
    """

    def _provider(train_start: float, train_end: float) -> object:
        state.now = train_end  # type: ignore[attr-defined]
        bars = state.bars(asset_id, timeframe, lookback=lookback)  # type: ignore[attr-defined]
        return tuple(b for b in bars if float(b.ts) >= train_start)

    return _provider


__all__ = [
    "ExpandingWindowSchedule",
    "FitFn",
    "RecursiveRefitter",
    "RefitPoint",
    "RefitRecord",
    "RefitResult",
    "RollingWindowSchedule",
    "TrainDataProvider",
    "hash_model_state",
    "state_bound_data_provider",
    "trial_count",
]
