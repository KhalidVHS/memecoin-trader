"""Tests for backtest.clock.SimulatedClock.

Every test here must *fail* if the guard it is exercising is removed.  The
comment above each test states which guard that is.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from memetrader.backtest.clock import (
    ClockError,
    SimulatedClock,
    WallClockAccessError,
)


# ---------------------------------------------------------------------------
# Advancement
# ---------------------------------------------------------------------------


def test_advance_to_moves_clock_forward() -> None:
    """Basic forward advance."""
    clock = SimulatedClock(run_id="test", start=1_000.0)
    clock.advance_to(2_000.0)
    assert clock.now == 2_000.0


def test_advance_to_same_time_is_allowed() -> None:
    """Advancing to the current time is a no-op, not an error.

    Two events at the same ``available_time`` both advance the clock to
    the same value.  The second call must succeed.
    """
    clock = SimulatedClock(run_id="test", start=1_000.0)
    clock.advance_to(1_000.0)  # idempotent
    assert clock.now == 1_000.0


def test_advance_to_backwards_raises() -> None:
    """Guard: clock refuses to move backwards.

    If this guard were removed, an event with a smaller ``available_time``
    than a previously yielded event could silently re-enter the visible window,
    corrupting the point-in-time invariant.
    """
    clock = SimulatedClock(run_id="test", start=5_000.0)
    with pytest.raises(ClockError, match="backwards"):
        clock.advance_to(4_999.0)


def test_advance_to_nan_raises() -> None:
    """NaN is not a valid time."""
    clock = SimulatedClock(run_id="test", start=0.0)
    with pytest.raises(ValueError):
        clock.advance_to(float("nan"))


def test_advance_to_inf_raises() -> None:
    """Infinity is not a valid time."""
    clock = SimulatedClock(run_id="test", start=0.0)
    with pytest.raises(ValueError):
        clock.advance_to(float("inf"))


def test_start_must_be_finite() -> None:
    """A non-finite start is rejected at construction."""
    with pytest.raises(ValueError):
        SimulatedClock(run_id="test", start=float("nan"))


# ---------------------------------------------------------------------------
# Wall-clock guard
# ---------------------------------------------------------------------------


def test_wall_clock_guard_time_time() -> None:
    """Guard: time.time() raises while a SimulatedClock is active.

    If this guard were removed, a module that reads ``time.time()`` during
    replay would produce non-deterministic results: two runs over the same
    data at different real-world times would produce different outputs.
    """
    with SimulatedClock(run_id="guard-test", start=0.0):
        with pytest.raises(WallClockAccessError):
            time.time()


def test_wall_clock_guard_datetime_now() -> None:
    """Guard: datetime.now() raises while a SimulatedClock is active."""
    with SimulatedClock(run_id="guard-test", start=0.0):
        with pytest.raises(WallClockAccessError):
            datetime.now(tz=timezone.utc)


def test_wall_clock_guard_restored_after_close() -> None:
    """After the clock exits, time.time() works again."""
    with SimulatedClock(run_id="restore-test", start=0.0):
        pass  # __exit__ calls close()
    t = time.time()
    assert t > 0.0, "time.time() should be callable after clock exits"


def test_wall_clock_guard_restored_on_exception() -> None:
    """The guard is restored even when an exception terminates the with-block."""
    with pytest.raises(RuntimeError, match="deliberate"):
        with SimulatedClock(run_id="exc-test", start=0.0):
            raise RuntimeError("deliberate")
    # Should not raise:
    time.time()


def test_nested_clocks_raise() -> None:
    """Guard: two simultaneous SimulatedClocks raise at the second activation.

    The wall-clock guard is process-wide and cannot be shared between two
    independent replay sessions.
    """
    with SimulatedClock(run_id="outer", start=0.0):
        with pytest.raises(ClockError, match="already active"):
            SimulatedClock(run_id="inner", start=0.0).__enter__()


# ---------------------------------------------------------------------------
# Deterministic seeds
# ---------------------------------------------------------------------------


def test_deterministic_seed_is_reproducible() -> None:
    """Same run_id + stream_name → same seed every time."""
    clock = SimulatedClock(run_id="repro", start=0.0)
    seed1 = clock.deterministic_seed("latency_draw")
    seed2 = clock.deterministic_seed("latency_draw")
    assert seed1 == seed2


def test_deterministic_seed_differs_by_stream() -> None:
    """Different stream names → different seeds (orthogonality guarantee)."""
    clock = SimulatedClock(run_id="repro", start=0.0)
    s1 = clock.deterministic_seed("latency_draw")
    s2 = clock.deterministic_seed("failed_tx")
    assert s1 != s2, (
        "Two different stream names produced the same seed.  If streams share a "
        "seed, adding one draw to one stream shifts every subsequent draw in the "
        "other, destroying replay equality across code changes."
    )


def test_deterministic_seed_differs_by_run_id() -> None:
    """Different run_ids → different seeds for the same stream name."""
    c1 = SimulatedClock(run_id="run-A", start=0.0)
    c2 = SimulatedClock(run_id="run-B", start=0.0)
    assert c1.deterministic_seed("latency_draw") != c2.deterministic_seed("latency_draw")


def test_deterministic_seed_is_non_negative_int() -> None:
    """Seeds must be non-negative ints for numpy/random compatibility."""
    clock = SimulatedClock(run_id="seed-type-test", start=0.0)
    seed = clock.deterministic_seed("stream")
    assert isinstance(seed, int)
    assert seed >= 0


def test_deterministic_seed_is_process_stable() -> None:
    """Seeds do not depend on PYTHONHASHSEED or process restart timing.

    We verify stability by checking a known expected value.  The expected
    value was computed by running ``deterministic_seed`` once and recording
    the result; this test will catch any change to the hash algorithm.
    """
    clock = SimulatedClock(run_id="stable", start=0.0)
    seed = clock.deterministic_seed("latency_draw")
    # Recompute manually: SHA-256("stable:latency_draw"), first 8 bytes big-endian.
    import hashlib

    digest = hashlib.sha256(b"stable:latency_draw").digest()
    expected = int.from_bytes(digest[:8], "big")
    assert seed == expected


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_now_at_start() -> None:
    clock = SimulatedClock(run_id="props", start=1_700_000_000.0)
    assert clock.now == 1_700_000_000.0


def test_run_id_preserved() -> None:
    clock = SimulatedClock(run_id="my-run", start=0.0)
    assert clock.run_id == "my-run"
