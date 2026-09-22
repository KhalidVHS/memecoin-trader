"""Simulated replay clock — the single source of time truth inside a backtest.

Every timestamp a replayed component reads must come from here, never from
``time.time()``, ``datetime.now()``, or anything that reads the wall clock.
That invariant is what makes a replay deterministic: two runs over identical
data with identical seeds must produce byte-identical economic output, and a
stray ``time.time()`` call anywhere in the call graph breaks that guarantee
silently — the equity curve still looks like an equity curve, but it cannot
be reproduced.

The guard is enforced at the module level by monkey-patching ``time.time``
and ``time.time_ns`` (module-attribute assignment, which is legal because
``time`` is an ordinary module) to raise when a ``SimulatedClock`` is active.
``time.monotonic``/``time.perf_counter`` are deliberately left unguarded —
they measure durations, not wall-clock instants, and guarding them would
break timing/profiling code that is harmlessly used inside a replay without
buying any determinism.

``datetime.datetime`` cannot be patched the same way: ``datetime`` is an
immutable C type, and assigning to ``datetime.now`` raises
``TypeError: cannot set 'now' attribute of immutable type 'datetime.datetime'``.
Instead we rebind the *class* in the ``datetime`` module's namespace to a
subclass whose ``now``/``utcnow`` raise, and restore the original class on
close.

.. important:: **Documented gap.** The subclass swap only catches call sites
   that do ``import datetime`` and then call ``datetime.datetime.now(...)``
   (or a module that imports ``memetrader.backtest.clock`` after this module
   has already rebound the name). It does **not** catch
   ``from datetime import datetime`` performed *before* the guard is
   installed, because that import binds the name directly to the original
   class object, and rebinding ``datetime.datetime`` afterwards does not
   change an already-bound local/module name elsewhere. Guard against this
   gap statically: ``tests/backtest/test_clock.py`` scans
   ``src/memetrader/**/*.py`` for direct ``datetime.now(``/``datetime.utcnow(``
   call sites and fails the build if one appears outside an explicit
   allowlist — this converts the unenforceable runtime gap into an
   enforceable static one.

Both guards are installed by ``SimulatedClock.__enter__`` under a single
exception-safe sequence: if any step of installing the guard fails, every
already-applied patch is unwound, ``_active_clock`` is reset to ``None``, and
the original exception is re-raised. A partially-applied guard — the bug that
motivated this design — is impossible by construction: either every patch is
applied and ``_active_clock`` is set, or none are and it is ``None``. The
guard is undone by ``SimulatedClock.close()``, which is why the clock
implements the context-manager protocol.

``deterministic_seed`` lives here rather than in a utilities module because it
is tightly coupled to the clock's ``run_id`` — seeds that are not bound to the
run cannot guarantee cross-run reproducibility.  A single shared RNG for all
streams is the naive approach and is the specific failure this helper prevents:
if the failed-TX coin flip and the latency draw share an RNG, adding one extra
latency call between a strategy decision and the next coin flip shifts every
subsequent outcome in that run, which destroys the ability to compare two code
versions on the same seed.  One RNG per named stream, each seeded from the run
ID plus its own name, keeps streams orthogonal.
"""

from __future__ import annotations

import datetime as _datetime_module
import hashlib
import math
import sys
import time
import types as _types
from typing import NoReturn


class ClockError(RuntimeError):
    """Raised when the replay clock is used incorrectly.

    Separate from ``ValueError`` so callers can distinguish a programming
    error (backwards advance, duplicate activation) from a data error in the
    event stream.
    """


class WallClockAccessError(ClockError):
    """Raised when code attempts to read the wall clock during a replay.

    The message includes the offending call's name so a stack trace is not
    needed to identify the source.
    """


# ---------------------------------------------------------------------------
# Wall-clock guard
# ---------------------------------------------------------------------------

# We replace ``time.time``, ``time.time_ns``, and the ``datetime.datetime``
# class with guards that raise when a SimulatedClock is active.  The
# originals are stashed here so ``close()`` (and the exception-safe unwind in
# ``__enter__``) can restore them.  Using module-level state rather than
# instance state means the guard is process-wide, which is what we need: a
# replayed strategy that reads the clock via an import alias still triggers
# the error.
#
# ``time.monotonic`` and ``time.perf_counter`` are NOT guarded. They measure
# elapsed durations, not wall-clock instants, so reading them during a replay
# does not make the economic output non-deterministic — it is safe (and
# common) for profiling/timing code to call them inside a replay.

_orig_time_time = time.time
_orig_time_time_ns = time.time_ns
_orig_datetime_class = _datetime_module.datetime
_active_clock: SimulatedClock | None = None


def active_clock() -> SimulatedClock | None:
    """Return the clock currently driving a replay, or ``None`` if live.

    This is the supported way for code outside this module to ask "am I inside
    a replay?".  Reading the private ``_active_clock`` global directly works
    today but couples callers to an implementation detail that the guard
    install/unwind logic owns.

    The motivating caller is :mod:`memetrader.ids`, which mints identifiers
    from the wall clock when live and must mint them from simulated time when
    replaying — it cannot simply call ``time.time()``, because the guard this
    module installs makes that raise.
    """
    return _active_clock


def _called_from_logging() -> bool:
    """Whether the immediate caller of a guard is the stdlib ``logging`` module.

    ``logging.LogRecord.__init__`` timestamps every record with
    ``time.time_ns()``. Without this exemption the guard turns any
    ``logger.warning(...)`` reached during a replay into a crashed run — and
    the replay's own production code legitimately warns (``portfolio
    .stop_loss_breaches`` on an unmarkable position, for one).

    Exempting it is sound for the same reason ``time.monotonic`` is not
    guarded at all: a log record's ``created`` timestamp is observability
    metadata that is written out and never read back into the simulation, so
    it cannot make the economic output differ between two runs over identical
    data. What the guard exists to catch — replay *logic* branching on the
    real time of day — is untouched, because that logic is not the ``logging``
    module.

    Identified by the calling frame's module name rather than by comparing
    code objects, so it keeps working across CPython versions that move the
    ``time_ns()`` call between ``LogRecord.__init__`` and its callers. The
    narrow cost is that a third-party module literally named ``logging`` would
    also be exempt; nothing in this codebase shadows that name.
    """
    frame = sys._getframe(2)  # 0 = here, 1 = the guard, 2 = whoever called it
    module = frame.f_globals.get("__name__", "")
    return module == "logging" or module.startswith("logging.")


def _guarded_time() -> float:
    if _called_from_logging():
        return _orig_time_time()
    raise WallClockAccessError(
        "time.time() called during a replay — read the SimulatedClock instead. "
        "A wall-clock read makes the replay non-deterministic: two runs over "
        "identical data will produce different outputs whenever this is called "
        "at a different real-world moment."
    )


def _guarded_time_ns() -> int:
    if _called_from_logging():
        return _orig_time_time_ns()
    raise WallClockAccessError(
        "time.time_ns() called during a replay — read the SimulatedClock instead. "
        "See time.time() guard for the rationale."
    )


class _GuardedDatetime(_orig_datetime_class):
    """``datetime.datetime`` subclass whose ``now``/``utcnow`` raise.

    Swapped in for the real class while a ``SimulatedClock`` is active (see
    the module docstring for why a subclass swap is used instead of patching
    ``datetime.now`` directly, and for the documented gap this leaves).
    """

    @classmethod
    def now(cls, tz: _datetime_module.tzinfo | None = None) -> NoReturn:
        raise WallClockAccessError(
            "datetime.now() called during a replay — read the SimulatedClock instead. "
            "See time.time() guard for the rationale."
        )

    @classmethod
    def utcnow(cls) -> NoReturn:
        raise WallClockAccessError(
            "datetime.utcnow() called during a replay — read the SimulatedClock instead. "
            "See time.time() guard for the rationale."
        )


# ---------------------------------------------------------------------------
# SimulatedClock
# ---------------------------------------------------------------------------


class SimulatedClock:
    """Monotonic, explicit-advancement-only simulated clock.

    Usage::

        with SimulatedClock(run_id="run-abc", start=1_700_000_000.0) as clock:
            clock.advance_to(1_700_003_600.0)  # one hour later
            assert clock.now == 1_700_003_600.0

    The clock refuses to move backwards (``ClockError``), refuses to be
    created while another clock is already active (``ClockError``), and guards
    against wall-clock reads for the duration of its activation.

    ``start`` defaults to 0.0 if omitted.  The engine should always supply an
    explicit start equal to the first event's ``available_time`` minus one
    tick, so nothing is accidentally available at t=0.
    """

    def __init__(self, run_id: str, start: float = 0.0) -> None:
        finite(start, "start")
        self._run_id = run_id
        self._now: float = start

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def now(self) -> float:
        """Current simulated time as epoch seconds."""
        return self._now

    def advance_to(self, t: float) -> None:
        """Advance the clock to ``t``.

        ``t`` must be finite and >= ``self.now``.  Refusing to go backwards is
        the monotonicity guarantee that makes every component's "was this event
        already available?" check correct: if the clock could go back, an event
        that was available at t=100 could silently re-enter the visible window
        at t=90.
        """
        finite(t, "t")
        if t < self._now:
            raise ClockError(
                f"advance_to({t}) would move the clock backwards from {self._now}. "
                "Simulated time is monotonic: events must arrive in non-decreasing "
                "available_time order."
            )
        self._now = t

    def deterministic_seed(self, stream_name: str) -> int:
        """Return a reproducible integer seed for a named RNG stream.

        Each stream gets an independent seed derived from the run ID and the
        stream name.  Callers should use this to seed a ``random.Random`` or
        ``numpy.random.default_rng`` that is *local to that stream*, not shared
        with any other stream.

        Why per-stream rather than per-run?  Because every time the calling code
        draws from a shared RNG, it shifts the sequence seen by every subsequent
        draw — including draws in completely unrelated subsystems.  Adding a
        latency draw between two strategy decisions silently changes every
        subsequent failed-TX coin flip, which means comparing two code versions
        on "the same seed" is not actually the same experiment.  Independent
        streams are orthogonal by construction: changing one stream's draw count
        never affects another.

        The seed is a non-negative int derived by SHA-256 over UTF-8 bytes,
        truncated to 64 bits.  SHA-256 is used for its avalanche property
        (nearby inputs produce uncorrelated outputs) rather than for security.
        The truncation to 64 bits keeps numpy's legacy API happy; the high bits
        are discarded.
        """
        key = f"{self._run_id}:{stream_name}"
        digest = hashlib.sha256(key.encode()).digest()
        # Interpret the first 8 bytes as a big-endian unsigned int.
        return int.from_bytes(digest[:8], "big")

    # ------------------------------------------------------------------
    # Context manager — installs / removes wall-clock guard
    # ------------------------------------------------------------------

    def __enter__(self) -> SimulatedClock:
        global _active_clock
        if _active_clock is not None:
            raise ClockError(
                f"A SimulatedClock (run_id={_active_clock.run_id!r}) is already "
                "active in this process.  Nested or concurrent simulated clocks "
                "are not supported: the wall-clock guard is process-wide and "
                "cannot be shared between two independent replay sessions."
            )
        # Install every patch, but if any step fails, unwind whatever was
        # already applied and reset _active_clock before re-raising.  A
        # partially-applied guard must be impossible: either every patch is
        # active and _active_clock is self, or none are and it is None.
        _active_clock = self
        applied: list[str] = []
        try:
            time.time = _guarded_time
            applied.append("time")
            time.time_ns = _guarded_time_ns
            applied.append("time_ns")
            setattr(_datetime_module, "datetime", _GuardedDatetime)  # noqa: B010
            applied.append("datetime")
        except BaseException:
            if "datetime" in applied:
                setattr(_datetime_module, "datetime", _orig_datetime_class)  # noqa: B010
            if "time_ns" in applied:
                time.time_ns = _orig_time_time_ns
            if "time" in applied:
                time.time = _orig_time_time
            _active_clock = None
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: _types.TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Remove the wall-clock guard and deactivate the clock.

        Safe to call more than once; subsequent calls are no-ops.
        """
        global _active_clock
        if _active_clock is self:
            time.time = _orig_time_time
            time.time_ns = _orig_time_time_ns
            setattr(_datetime_module, "datetime", _orig_datetime_class)  # noqa: B010
            _active_clock = None


# ---------------------------------------------------------------------------
# Helpers used internally
# ---------------------------------------------------------------------------


def finite(value: float, name: str) -> float:
    """Minimal copy of ``types.finite`` to avoid a circular import.

    ``clock.py`` is the lowest layer of the backtest package and must not
    import from the broader ``memetrader`` namespace, because the engine will
    import both — and Python's import system is not required to make that work
    in any particular order.  The validation logic is identical; the duplication
    is deliberate and documented here so a future reader does not "helpfully"
    replace it with an import that introduces a cycle.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        raise ValueError(f"{name} must be finite, got {v}")
    return v
