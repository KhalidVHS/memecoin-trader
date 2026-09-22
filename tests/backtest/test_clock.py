"""Tests for backtest.clock.SimulatedClock.

Every test here must *fail* if the guard it is exercising is removed.  The
comment above each test states which guard that is.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import pytest

from memetrader.backtest.clock import (
    ClockError,
    SimulatedClock,
    WallClockAccessError,
)

# Root of the source tree the static datetime.now/utcnow scan below covers.
_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "memetrader"

# Modules allowed to call datetime.now()/datetime.utcnow() directly, because
# they are not part of the replay call graph (see clock.py's documented gap:
# the SimulatedClock guard cannot catch `from datetime import datetime`
# imports performed before the guard is installed, so non-replay call sites
# are policed statically instead). Empty today — nothing in memetrader needs
# a raw wall-clock read; new offenders must either use SimulatedClock or be
# justified here.
_DATETIME_NOW_ALLOWLIST: frozenset[str] = frozenset()


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
    with (
        SimulatedClock(run_id="guard-test", start=0.0),
        pytest.raises(WallClockAccessError),
    ):
        time.time()


def test_wall_clock_guard_datetime_now() -> None:
    """Guard: datetime.datetime.now() raises while a SimulatedClock is active.

    The guard works by rebinding ``datetime.datetime`` to a raising subclass
    (patching ``datetime.now`` directly is impossible: ``datetime`` is an
    immutable C type). That means this test must go through the module
    attribute (``import datetime as dt`` ... ``dt.datetime.now(...)``) rather
    than binding the name early via ``from datetime import datetime`` — the
    latter would capture the original class before the guard is installed
    and is, by design, NOT caught by the runtime guard. See the documented
    gap in clock.py's module docstring, and
    ``test_no_direct_datetime_now_in_src`` below for the static check that
    covers that gap.
    """
    with (
        SimulatedClock(run_id="guard-test", start=0.0),
        pytest.raises(WallClockAccessError),
    ):
        dt.datetime.now(tz=dt.UTC)


def test_wall_clock_guard_datetime_utcnow() -> None:
    """Guard: datetime.datetime.utcnow() also raises while active."""
    with (
        SimulatedClock(run_id="guard-test-utcnow", start=0.0),
        pytest.raises(WallClockAccessError),
    ):
        dt.datetime.utcnow()  # noqa: DTZ003


def test_wall_clock_guard_time_time_ns() -> None:
    """Guard: time.time_ns() raises while a SimulatedClock is active."""
    with (
        SimulatedClock(run_id="guard-test-ns", start=0.0),
        pytest.raises(WallClockAccessError),
    ):
        time.time_ns()


def test_wall_clock_guard_restored_after_close() -> None:
    """After the clock exits, time.time() and datetime.now() work again."""
    with SimulatedClock(run_id="restore-test", start=0.0):
        pass  # __exit__ calls close()
    t = time.time()
    assert t > 0.0, "time.time() should be callable after clock exits"
    now = dt.datetime.now(tz=dt.UTC)
    assert now.year >= 2024, "datetime.now() should be callable after clock exits"


def test_wall_clock_guard_restored_on_exception() -> None:
    """The guard is restored even when an exception terminates the with-block."""
    with (
        pytest.raises(RuntimeError, match="deliberate"),
        SimulatedClock(run_id="exc-test", start=0.0),
    ):
        raise RuntimeError("deliberate")
    # Should not raise:
    time.time()


def test_nested_clocks_raise() -> None:
    """Guard: two simultaneous SimulatedClocks raise at the second activation.

    The wall-clock guard is process-wide and cannot be shared between two
    independent replay sessions.
    """
    with (
        SimulatedClock(run_id="outer", start=0.0),
        pytest.raises(ClockError, match="already active"),
    ):
        SimulatedClock(run_id="inner", start=0.0).__enter__()


def test_nested_clock_rejection_leaves_outer_guard_intact() -> None:
    """Regression test for the bug this module was rewritten to fix.

    A rejected nested activation must not disturb the outer clock's guard:
    time.time()/time.time_ns()/datetime.now() must still raise, and the
    active clock must still be the outer one (i.e. the outer clock's own
    ``close()`` still tears its guard down cleanly afterwards). Before the
    fix, any failure partway through installing a guard (e.g. the
    ``datetime.now`` assignment that used to raise ``TypeError``) could
    leave ``time.time`` permanently patched with no way to undo it, because
    the exception escaped ``__enter__`` before ``__exit__``/``close()`` ever
    ran. The nested-clock rejection path exercises the same "raise inside
    __enter__" shape from the caller's side.
    """
    outer = SimulatedClock(run_id="outer-survives", start=0.0)
    with outer:
        with pytest.raises(ClockError, match="already active"):
            SimulatedClock(run_id="inner-rejected", start=0.0).__enter__()

        # Outer guard must still be fully active after the rejected nested
        # activation attempt.
        with pytest.raises(WallClockAccessError):
            time.time()
        with pytest.raises(WallClockAccessError):
            time.time_ns()
        with pytest.raises(WallClockAccessError):
            dt.datetime.now(tz=dt.UTC)

    # And the outer clock's own close() still restores everything cleanly.
    time.time()
    time.time_ns()
    dt.datetime.now(tz=dt.UTC)


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


# ---------------------------------------------------------------------------
# Static coverage for the documented runtime-guard gap
# ---------------------------------------------------------------------------


def test_no_direct_datetime_now_in_src() -> None:
    """Static check covering the gap the runtime guard cannot close.

    ``SimulatedClock``'s guard works by rebinding ``datetime.datetime`` to a
    raising subclass, which cannot intercept a ``from datetime import
    datetime`` binding made before the guard was installed (see clock.py's
    module docstring). Rather than rely on every call site going through
    ``import datetime`` correctly, we forbid direct ``datetime.now(``/
    ``datetime.utcnow(`` call sites in the source tree outright: any replayed
    code should be getting its time from ``SimulatedClock`` instead, and any
    non-replay code that legitimately needs a wall-clock read should be
    listed in ``_DATETIME_NOW_ALLOWLIST`` with a justification in the comment
    next to it.
    """
    # clock.py itself defines the guard's now()/utcnow() overrides, which
    # raise rather than read the wall clock, and its docstrings/comments
    # discuss ``datetime.now()`` in prose — it is the one file the scan
    # would otherwise (correctly, but uselessly) flag, so it is excluded
    # here rather than via the allowlist, which is reserved for call sites
    # that genuinely execute a wall-clock read.
    _clock_module_path = (_SRC_ROOT / "backtest" / "clock.py").resolve()

    offenders: list[str] = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        if path.resolve() == _clock_module_path:
            continue
        rel = path.relative_to(_SRC_ROOT.parents[1]).as_posix()
        if rel in _DATETIME_NOW_ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "datetime.now(" in line or ".utcnow(" in line:
                offenders.append(f"{rel}:{lineno}: {stripped}")
    assert not offenders, (
        "Direct datetime.now()/datetime.utcnow() call sites found outside the "
        "SimulatedClock guard's own implementation. Read simulated time from "
        "SimulatedClock instead, or add a justified entry to "
        "_DATETIME_NOW_ALLOWLIST:\n" + "\n".join(offenders)
    )
