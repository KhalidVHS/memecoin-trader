"""Tests for backtest.event_queue.EventQueue.

Every test must *fail* if the guard it exercises is removed.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

import pytest

from memetrader.backtest.clock import SimulatedClock
from memetrader.backtest.event_queue import EventQueue, EventQueueError
from memetrader.types import EventKind, HistoricalEvent


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_event(
    available_time: float,
    kind: EventKind = EventKind.BAR_CLOSE,
    sequence: int = 0,
    source: str = "test",
    asset_id: str = "BONK",
) -> HistoricalEvent:
    return HistoricalEvent(
        kind=kind,
        available_time=available_time,
        asset_id=asset_id,
        payload=None,
        sequence=sequence,
        source=source,
    )


def _clock(start: float = 0.0) -> SimulatedClock:
    return SimulatedClock(run_id="queue-test", start=start)


def _drain(queue: EventQueue) -> list[HistoricalEvent]:
    return list(queue)


# ---------------------------------------------------------------------------
# Ordering correctness
# ---------------------------------------------------------------------------


def test_single_stream_order_preserved() -> None:
    """Events from a single sorted stream arrive in sort_key order."""
    events = [
        _make_event(1.0, sequence=0),
        _make_event(2.0, sequence=1),
        _make_event(3.0, sequence=2),
    ]
    clock = _clock()
    queue = EventQueue(clock, streams=[iter(events)])
    result = _drain(queue)
    assert result == events


def test_multi_stream_merge_is_sorted() -> None:
    """Events from two streams are merged in sort_key order.

    This is the core invariant: the heap merge must produce the same global
    order as if all events were sorted together up front.
    """
    stream_a = [_make_event(t, sequence=i, source="A") for i, t in enumerate([1.0, 3.0, 5.0])]
    stream_b = [_make_event(t, sequence=i, source="B") for i, t in enumerate([2.0, 4.0, 6.0])]
    clock = _clock()
    queue = EventQueue(clock, streams=[iter(stream_a), iter(stream_b)])
    result = _drain(queue)
    expected_times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert [e.available_time for e in result] == expected_times


def test_queue_order_is_stable_under_shuffled_insertion() -> None:
    """Guard: queue order must not depend on insertion order of push() calls.

    Two runs over the same shuffled input must produce identical event sequences.
    If this were not enforced, replay equality would be an illusion: passing the
    same seed to the engine would still produce different results if push() calls
    arrived in different orders.
    """
    events = [
        _make_event(1.0, sequence=0, source="A"),
        _make_event(1.0, sequence=0, source="B"),  # same time, different source
        _make_event(2.0, sequence=0, source="A"),
        _make_event(3.0, sequence=0, source="B"),
    ]

    def run_with_shuffle(seed: int) -> list[tuple]:
        rng = random.Random(seed)
        shuffled = events.copy()
        rng.shuffle(shuffled)
        clock = _clock()
        queue = EventQueue(clock)
        for e in shuffled:
            queue.push(e)
        return [e.sort_key for e in _drain(queue)]

    keys_1 = run_with_shuffle(1)
    keys_2 = run_with_shuffle(2)
    keys_3 = run_with_shuffle(3)
    assert keys_1 == keys_2 == keys_3, (
        "Queue order changed across shuffled insertions.  The sort_key total "
        "order must be independent of push() call order."
    )


def test_two_runs_same_shuffled_input_identical_sequence() -> None:
    """Two runs over the same shuffled input produce identical event sequences."""
    events = [
        _make_event(float(t), sequence=i, source="src")
        for i, t in enumerate(range(20))
    ]
    shuffled = events.copy()
    random.Random(99).shuffle(shuffled)

    def run() -> list[tuple]:
        clock = _clock()
        queue = EventQueue(clock)
        for e in shuffled:
            queue.push(e)
        return [e.sort_key for e in _drain(queue)]

    assert run() == run()


def test_event_priority_respected_at_same_time() -> None:
    """Within the same available_time, EVENT_PRIORITY controls order.

    A DECISION_TICK (priority 70) must arrive after a BAR_CLOSE (priority 10)
    at the same simulated time — otherwise the strategy could act on a bar in
    the same breath as it arrives.
    """
    bar = _make_event(100.0, kind=EventKind.BAR_CLOSE, sequence=0)
    decision = _make_event(100.0, kind=EventKind.DECISION_TICK, sequence=0)
    clock = _clock()
    queue = EventQueue(clock)
    # Push decision first to check that priority overrides push order.
    queue.push(decision)
    queue.push(bar)
    result = _drain(queue)
    assert result[0].kind == EventKind.BAR_CLOSE
    assert result[1].kind == EventKind.DECISION_TICK


# ---------------------------------------------------------------------------
# Past-time push guard
# ---------------------------------------------------------------------------


def test_push_past_available_time_raises() -> None:
    """Guard: pushing an event whose available_time < clock.now raises.

    If this guard were removed, the engine could claim that an event was
    knowable before the clock reached its availability time — a lookahead bug
    that is invisible in the equity curve.
    """
    clock = _clock(start=1_000.0)
    queue = EventQueue(clock)
    past_event = _make_event(999.0)
    with pytest.raises(EventQueueError, match="lookahead"):
        queue.push(past_event)


def test_push_at_current_time_is_allowed() -> None:
    """Pushing an event at the current clock time is not in the past."""
    clock = _clock(start=500.0)
    queue = EventQueue(clock)
    event = _make_event(500.0, sequence=0)
    queue.push(event)  # must not raise
    result = _drain(queue)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Duplicate sort_key guard
# ---------------------------------------------------------------------------


def test_duplicate_sort_key_in_push_raises() -> None:
    """Guard: two events with identical sort_keys raise on the second push.

    If this guard were absent, two events with the same sort_key would compare
    equal under the heap invariant, and the queue would fall back to insertion
    order to break the tie — making replay order non-deterministic.
    """
    clock = _clock()
    queue = EventQueue(clock)
    e1 = _make_event(1.0, sequence=7, source="X", asset_id="BONK")
    e2 = _make_event(1.0, sequence=7, source="X", asset_id="BONK")
    assert e1.sort_key == e2.sort_key, "fixture error: sort_keys must be equal"
    queue.push(e1)
    with pytest.raises(EventQueueError, match="Duplicate sort_key"):
        queue.push(e2)


def test_duplicate_sort_key_across_stream_and_push_raises() -> None:
    """A stream event and a pushed event with the same sort_key raise."""
    e_stream = _make_event(1.0, sequence=0, source="src")
    e_push = _make_event(1.0, sequence=0, source="src")
    assert e_stream.sort_key == e_push.sort_key

    clock = _clock()
    queue = EventQueue(clock, streams=[iter([e_stream])])
    # The stream event is on the heap.  Pushing an identical key must raise.
    with pytest.raises(EventQueueError):
        queue.push(e_push)


# ---------------------------------------------------------------------------
# Out-of-order stream guard
# ---------------------------------------------------------------------------


def test_out_of_order_stream_raises() -> None:
    """Guard: a stream that yields events out of sort_key order raises.

    An unsorted stream would let events bypass the heap's ordering guarantee.
    The error must be raised when the offending event is encountered, not at
    queue construction time (construction does not consume the whole stream).
    """

    def bad_stream() -> Iterator[HistoricalEvent]:
        yield _make_event(5.0, sequence=0)
        yield _make_event(3.0, sequence=0)  # out of order

    clock = _clock()
    queue = EventQueue(clock, streams=[bad_stream()])
    # First event is fine.
    next(queue)
    # Second event triggers the guard.
    with pytest.raises(EventQueueError, match="not sorted"):
        next(queue)


# ---------------------------------------------------------------------------
# Lazy streaming (memory)
# ---------------------------------------------------------------------------


def test_lazy_stream_does_not_load_all_at_once() -> None:
    """The queue drives the stream lazily; the iterator is not exhausted up front.

    We verify this by using a generator that counts how many elements have been
    yielded and asserting the count after one pop.
    """
    yielded: list[int] = []

    def counting_stream() -> Iterator[HistoricalEvent]:
        for i in range(100):
            yielded.append(i)
            yield _make_event(float(i), sequence=i)

    clock = _clock()
    queue = EventQueue(clock, streams=[counting_stream()])
    # The queue seeds the heap with the first event from each stream.
    assert len(yielded) == 1, (
        "Stream should be advanced exactly once at construction — one peek "
        "to seed the heap.  Pulling all events defeats lazy streaming."
    )
    next(queue)
    # After one pop, the queue advances the stream once more.
    assert len(yielded) == 2


# ---------------------------------------------------------------------------
# Peek and bool
# ---------------------------------------------------------------------------


def test_peek_returns_next_without_consuming() -> None:
    events = [_make_event(1.0, sequence=0), _make_event(2.0, sequence=1)]
    clock = _clock()
    queue = EventQueue(clock, streams=[iter(events)])
    top = queue.peek()
    assert top is not None
    assert top.available_time == 1.0
    # peek does not consume:
    assert next(queue).available_time == 1.0


def test_peek_returns_none_when_empty() -> None:
    clock = _clock()
    queue = EventQueue(clock)
    assert queue.peek() is None


def test_bool_false_when_empty() -> None:
    clock = _clock()
    queue = EventQueue(clock)
    assert not queue


def test_bool_true_when_nonempty() -> None:
    clock = _clock()
    queue = EventQueue(clock)
    queue.push(_make_event(1.0, sequence=0))
    assert queue


# ---------------------------------------------------------------------------
# Empty queue
# ---------------------------------------------------------------------------


def test_empty_queue_stops_iteration() -> None:
    clock = _clock()
    queue = EventQueue(clock)
    result = list(queue)
    assert result == []


def test_empty_stream_stops_iteration() -> None:
    clock = _clock()
    queue = EventQueue(clock, streams=[iter([])])
    assert list(queue) == []
