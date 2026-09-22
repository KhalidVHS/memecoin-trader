"""Simulated replay clock — the single source of time truth inside a backtest.

Every timestamp a replayed component reads must come from here, never from
``time.time()``, ``datetime.now()``, or anything that reads the wall clock.
That invariant is what makes a replay deterministic: two runs over identical
data with identical seeds must produce byte-identical economic output, and a
stray ``time.time()`` call anywhere in the call graph breaks that guarantee
silently — the equity curve still looks like an equity curve, but it cannot
be reproduced.

The guard is enforced at the module level by monkey-patching ``time.time`` and
``datetime.now`` to raise when a ``SimulatedClock`` is active.  The patch is
re-entrant (a second ``SimulatedClock`` while one is already active raises
immediately) and is undone by ``SimulatedClock.close()``, which is why the
clock implements the context-manager protocol.

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

import hashlib
import math
import time
import types as _types
from datetime import datetime, timezone


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

# We replace ``time.time`` and ``datetime.now`` with guards that raise when a
# SimulatedClock is active.  The originals are stashed here so ``close()``
# can restore them.  Using module-level state rather than instance state means
# the guard is process-wide, which is what we need: a replayed strategy that
# reads the clock via an import alias still triggers the error.

_orig_time_time = time.time
_orig_datetime_now = datetime.now
_active_clock: SimulatedClock | None = None


def _guarded_time() -> float:
    raise WallClockAccessError(
        "time.time() called during a replay — read the SimulatedClock instead. "
        "A wall-clock read makes the replay non-deterministic: two runs over "
        "identical data will produce different outputs whenever this is called "
        "at a different real-world moment."
    )


def _guarded_datetime_now(tz: timezone | None = None) -> datetime:  # noqa: ARG001
    raise WallClockAccessError(
        "datetime.now() called during a replay — read the SimulatedClock instead. "
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
        global _active_clock  # noqa: PLW0603
        if _active_clock is not None:
            raise ClockError(
                f"A SimulatedClock (run_id={_active_clock.run_id!r}) is already "
                "active in this process.  Nested or concurrent simulated clocks "
                "are not supported: the wall-clock guard is process-wide and "
                "cannot be shared between two independent replay sessions."
            )
        _active_clock = self
        time.time = _guarded_time  # type: ignore[assignment]
        datetime.now = _guarded_datetime_now  # type: ignore[assignment]
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
        global _active_clock  # noqa: PLW0603
        if _active_clock is self:
            time.time = _orig_time_time  # type: ignore[assignment]
            datetime.now = _orig_datetime_now  # type: ignore[assignment]
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
