"""Deterministic priority queue for the backtest replay loop.

Every simulated event — a bar close, a decision tick, a fill landing — passes
through this queue.  The queue's job is exactly one thing: produce events in a
total, deterministic, reproducible order.

## Why total order matters

Two events with the same ``sort_key`` would compare equal under Python's
heap invariant, which falls back to the heap's insertion order.  Insertion
order in a streaming merge depends on which source iterator happened to
advance first, which depends on I/O scheduling, which is not deterministic
across machines or Python versions.  The result is a replay that passes every
test on the author's laptop and produces a different equity curve on CI.

``HistoricalEvent.sort_key`` is a 5-tuple:
    (available_time, EVENT_PRIORITY[kind], sequence, source, asset_id)

For two distinct events to collide, all five fields must match.  The first
two are data constraints.  ``sequence`` is the source's own ordering token
(Solana slot, vendor cursor); two events from the same source with the same
slot and the same asset at the same available_time are genuinely the same
event and should not both be in the queue.  An assertion enforces this:
pushing a duplicate is an error, not a silent no-op, because a silent no-op
would mean an event that should have been processed was dropped.

## Why lazy merging matters

The full history is 801,957 bars.  Loading everything into a heap at once
requires O(N) memory and O(N log N) upfront sorting.  Lazy merging of
already-sorted streams (each loaded bar file is sorted by ``available_time``)
uses O(k) heap size where k is the number of active streams, and each
``heappush`` / ``heappop`` pair is O(log k) rather than O(log N).

## Pushed events (generated during replay)

An order becoming ready, a fill landing, a DECISION_TICK injected by the
engine — these arrive as ``push()`` calls during replay.  They must have
``available_time >= clock.now`` at the moment of push.  An event pushed with
a past time is a lookahead bug: the engine is claiming that something was
knowable before it was computed.

Pushed events and streamed events share the same heap, so their interleaving
is governed by the same total-order rules as everything else.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memetrader.backtest.clock import SimulatedClock

from memetrader.types import HistoricalEvent


class EventQueueError(RuntimeError):
    """Raised when the queue detects a determinism-breaking condition.

    Separate from ``ValueError`` so callers can distinguish queue-level
    contract violations from data-level validation failures.
    """


# ---------------------------------------------------------------------------
# Internal heap entry
# ---------------------------------------------------------------------------

# The heap stores 2-tuples of (sort_key, event).  Python's heapq is a
# min-heap, and sort_key is already a 5-tuple that implements the total order
# we want, so no wrapper class is needed.  The second element of the tuple is
# compared only if sort_keys are equal, and since HistoricalEvent is a
# frozen dataclass with slots, Python will try to compare it field-by-field —
# but we assert that no two events ever have equal sort_keys, so the fallback
# comparison never executes in correct code.

_HeapEntry = tuple[tuple[float, int, int, str, str], HistoricalEvent]


class EventQueue:
    """Deterministic, lazily-merged priority queue for replay events.

    Parameters
    ----------
    clock:
        The active ``SimulatedClock``.  The queue uses it to validate that
        pushed events are not in the past, and to reject ``next()`` calls
        that would advance the clock past a pushed event's time before that
        event has been yielded.
    streams:
        Zero or more iterators over pre-sorted ``HistoricalEvent`` sequences.
        Each iterator must yield events in non-decreasing ``sort_key`` order;
        the queue does not re-sort within a stream.  Out-of-order events within
        a stream are detected and raise ``EventQueueError``.

    Usage::

        queue = EventQueue(clock, streams=[bar_events, social_events])
        queue.push(HistoricalEvent(...))  # inject a generated event
        for event in queue:
            clock.advance_to(event.available_time)
            handle(event)
    """

    def __init__(
        self,
        clock: SimulatedClock,
        streams: list[Iterator[HistoricalEvent]] | None = None,
    ) -> None:
        self._clock = clock
        self._heap: list[_HeapEntry] = []
        # Track the last sort_key seen from each stream to detect out-of-order
        # events.  Key is the stream's index in the original list.
        self._stream_cursors: dict[int, tuple[float, int, int, str, str]] = {}
        # All seen sort_keys, to detect pushed duplicates.  We track keys
        # rather than events because HistoricalEvent.payload is an arbitrary
        # object and is not hashable in general.
        self._seen_keys: set[tuple[float, int, int, str, str]] = set()

        # Initialise one sentinel per stream.  ``_advance_stream`` pulls the
        # first event from each and pushes it onto the heap.
        self._streams: list[Iterator[HistoricalEvent]] = []
        for stream in streams or []:
            idx = len(self._streams)
            self._streams.append(stream)
            self._advance_stream(idx)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, event: HistoricalEvent) -> None:
        """Inject an event generated during the replay.

        Preconditions (each raises ``EventQueueError`` if violated):

        * ``event.available_time >= clock.now`` — a generated event cannot
          be placed in the past.  The engine is about to act on it; if it
          were already in the past, the engine has already had the chance to
          act on it and is now claiming it learned something retroactively.

        * The event's ``sort_key`` must not collide with any previously seen
          key.  A collision that is not a genuine duplicate is a data error
          (two distinct events with identical 5-tuple keys); a genuine
          duplicate should be handled by the caller before it reaches the queue
          (idempotency is the caller's problem, not the queue's).
        """
        if event.available_time < self._clock.now:
            raise EventQueueError(
                f"push() received an event with available_time={event.available_time} "
                f"which is earlier than the current clock ({self._clock.now}). "
                "This is a lookahead bug: the engine is claiming an event was "
                "knowable before the clock reached its availability time."
            )
        key = event.sort_key
        if key in self._seen_keys:
            raise EventQueueError(
                f"Duplicate sort_key {key!r} pushed to EventQueue. "
                "Two distinct events must never share a sort_key — the total "
                "order would become undefined and replay would not be "
                "reproducible.  If this is a genuine duplicate event, handle "
                "idempotency before calling push()."
            )
        self._seen_keys.add(key)
        heapq.heappush(self._heap, (key, event))

    def __iter__(self) -> Iterator[HistoricalEvent]:
        """Yield events in total sort_key order until the queue is empty."""
        return self

    def __next__(self) -> HistoricalEvent:
        """Pop and return the next event.

        Also advances any stream whose leading event is now the cheapest
        candidate on the heap.  This keeps the heap size at O(k) where k is
        the number of active streams plus the number of pushed-but-not-yet-
        yielded events.
        """
        if not self._heap:
            raise StopIteration
        key, event = heapq.heappop(self._heap)
        # Pull the next event from whichever stream this event came from.
        # We store the stream index in a side channel keyed by sort_key so we
        # can identify the stream without adding a wrapper tuple element that
        # would change the comparison semantics.
        if key in self._key_to_stream:
            idx = self._key_to_stream.pop(key)
            self._advance_stream(idx)
        return event

    def __bool__(self) -> bool:
        """``True`` iff the queue has at least one more event."""
        return bool(self._heap)

    def peek(self) -> HistoricalEvent | None:
        """Return the next event without consuming it, or ``None`` if empty."""
        if not self._heap:
            return None
        return self._heap[0][1]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _key_to_stream(self) -> dict[tuple[float, int, int, str, str], int]:
        """Lazy-initialised mapping from the sort_key of a stream's leading
        event to that stream's index.

        This is how ``__next__`` knows which stream to advance after popping:
        we tag each stream-sourced heap entry when we push it, and remove the
        tag when we pop it.
        """
        try:
            return self.__key_to_stream  # type: ignore[attr-defined]
        except AttributeError:
            self.__key_to_stream: dict[
                tuple[float, int, int, str, str], int
            ] = {}
            return self.__key_to_stream

    def _advance_stream(self, stream_idx: int) -> None:
        """Pull one event from stream ``stream_idx`` and push it to the heap.

        If the stream is exhausted, does nothing.  If the event's sort_key
        is less than the previous key from the same stream, raises
        ``EventQueueError`` — the stream was not sorted, which would let
        events from it bypass the heap's ordering guarantee.
        """
        stream = self._streams[stream_idx]
        try:
            event = next(stream)
        except StopIteration:
            return
        key = event.sort_key
        prev = self._stream_cursors.get(stream_idx)
        if prev is not None and key < prev:
            raise EventQueueError(
                f"Stream {stream_idx} is not sorted: received sort_key {key!r} "
                f"after {prev!r}.  Each stream passed to EventQueue must yield "
                "events in non-decreasing sort_key order.  An unsorted stream "
                "would silently produce out-of-order events after the heap merges "
                "them."
            )
        self._stream_cursors[stream_idx] = key
        if key in self._seen_keys:
            raise EventQueueError(
                f"Stream {stream_idx} produced a sort_key {key!r} that has "
                "already been seen (from a push() or an earlier stream). "
                "Two events must never share a sort_key."
            )
        self._seen_keys.add(key)
        self._key_to_stream[key] = stream_idx
        heapq.heappush(self._heap, (key, event))
