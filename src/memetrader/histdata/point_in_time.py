"""Point-in-time state for the backtest replay engine.

THIS IS THE MOST IMPORTANT FILE IN THE PACKAGE.

The central invariant of the entire backtest: at simulated time ``t``, a strategy
may only read records with ``available_time <= t``. Every candle it sees must
have ``closed is True``. A forming bar must never escape this boundary.

The two most common ways backtests silently grant themselves foresight:

1. **Sorting by event_time instead of available_time.** A bar whose ``ts`` is
   10:00 is true from 10:00 but not *knowable* until 11:00 + publication_delay.
   Sorting by ``ts`` instead of ``available_time`` hands the strategy 60+
   minutes of foresight on every bar, which on a 0.2% hurdle is the entire edge.

2. **Using a forming bar.** A bar whose close time is in the future is not a
   closed observation — it is a prediction. Including it in a "historical"
   window is look-ahead by the bar's remaining duration.

**The design here makes both bugs structurally hard:**

The internal store is a list sorted by ``available_time``, and every query uses
``bisect_right`` to find the cut-off index. Returning a record with
``available_time > self.now`` requires actively bypassing the bisect, not just
forgetting a filter. The ``available_time`` of every candle is set to
``ts + interval + publication_delay``, so the bar for ``ts=10:00`` on a 1h
timeframe with zero delay is not available until ``11:00`` at the earliest.

**Protocol vs implementation:**

``PointInTimeState`` is the Protocol from ``BACKTEST-CONTRACTS.md §4``. It is
the only legal interface for reading history during a replay — no backtest module
may import raw bar files directly. ``ReplayState`` is the concrete implementation.

The protocol is a Protocol (not an ABC) because ``execution/interfaces.py``
already depends on it and we do not want to force all implementations to inherit
from a common base. Structural subtyping is the right tool here.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..types import (
    Candle,
    CandleSeries,
    DataQuality,
    Provenance,
    Side,
    Timeframe,
)
from .schemas import PoolState, QuoteLadder

# Default publication delay for bar data. A candle closes at ``ts + interval``
# and is then published and indexed by the vendor. In practice GeckoTerminal's
# bar data appears within seconds of the close, but we apply a small safety
# margin to avoid "bar appears exactly at close" races. Configurable at
# ``ReplayState`` construction time so tests can set it to zero.
DEFAULT_PUBLICATION_DELAY_SECONDS = 0.0


@runtime_checkable
class PointInTimeState(Protocol):
    """The only legal way to read history during a backtest replay.

    Every method on this protocol returns only records with
    ``available_time <= self.now``, and every ``Candle`` satisfies
    ``closed is True``.

    ``now`` is the simulated clock. The replay engine advances it; a component
    that wants to "look ahead" must accept a future ``now``, which means
    explicitly lying about the clock — not just forgetting a filter.

    This matches the interface in ``BACKTEST-CONTRACTS.md §4`` exactly.
    ``pool_state`` and ``quote_ladder`` return ``None`` at TIER_0 (current
    state) — the catalog has no pool state or ladder history — and that
    absence is the signal to the execution model to fall back to bar-based
    cost estimation.
    """

    now: float

    def bars(
        self, asset_id: str, timeframe: Timeframe, *, lookback: int
    ) -> tuple[Candle, ...]:
        """The last ``lookback`` closed bars for ``asset_id`` at ``timeframe``.

        Returns an empty tuple when no data is available, never raises.
        Bars are returned oldest-first (ascending ``ts``).
        """
        ...

    def snapshot(self, asset_id: str) -> None:
        """The most recent CoinSnapshot for ``asset_id``, or None.

        At TIER_0 this always returns None — the replay has no live snapshot
        data beyond what was reconstructed from bars. The execution model must
        construct its own snapshot from bars rather than relying on this.
        """
        ...

    def universe(self) -> frozenset[str]:
        """The set of asset_ids eligible for trading at ``self.now``.

        Point-in-time: a coin is eligible only from its ``eligible_from``
        timestamp, and only while it has data. Today's universe must not
        be back-applied to yesterday's replay.
        """
        ...

    def pool_state(self, pool_id: str) -> PoolState | None:
        """The most recent PoolState for ``pool_id``, or None.

        None at TIER_0 (no pool state history collected).
        """
        ...

    def quote_ladder(self, asset_id: str, side: Side) -> QuoteLadder | None:
        """The most recent QuoteLadder for ``asset_id``+``side``, or None.

        None at TIER_0 (no ladder history collected).
        """
        ...


@dataclass
class _CandleStore:
    """Sorted index of closed candles for one (asset_id, timeframe) pair.

    Sorted by ``available_time`` so every query is a bisect, not a scan.
    The sort is the structural guarantee against look-ahead: a query for
    ``now=t`` finds only entries up to the rightmost index where
    ``available_time <= t``, which is exactly what bisect_right gives us
    when we search on the ``available_time`` projection.

    We keep a parallel list of ``available_time`` values for bisect, because
    bisect requires a key-extractable sorted sequence and Python's bisect does
    not support a key argument before 3.10 — even though we target 3.14, the
    parallel list is clearer and avoids a callable allocation on every query.
    """

    _candles: list[Candle] = field(default_factory=list)
    _available_times: list[float] = field(default_factory=list)

    def add(self, candle: Candle, available_time: float) -> None:
        """Insert a candle in sorted position by ``available_time``.

        We use ``bisect_right`` to find the insertion point, which keeps equal
        ``available_time`` values in insertion order — FIFO within a timestamp,
        which is deterministic for a given feed ordering.
        """
        if not candle.closed:
            raise ValueError(
                f"ReplayState refuses to store a bar that has not closed "
                f"(ts={candle.ts}). A forming bar must never enter the store: "
                "it is a partial observation that will be revised, and its "
                "presence in a 'historical' window is look-ahead."
            )
        idx = bisect.bisect_right(self._available_times, available_time)
        self._candles.insert(idx, candle)
        self._available_times.insert(idx, available_time)

    def query(self, now: float, lookback: int) -> tuple[Candle, ...]:
        """Return the last ``lookback`` candles available at ``now``.

        ``bisect_right`` gives the first index where ``available_time > now``,
        so everything before that index is available. We take the last
        ``lookback`` entries from that slice — oldest-first is preserved because
        the list is sorted by ``available_time``, which for same-interval bars
        is also oldest-first by ``ts``.

        Note: bars with the same ``available_time`` are in insertion order,
        which is oldest-first because the loader adds them chronologically.
        """
        cut = bisect.bisect_right(self._available_times, now)
        start = max(0, cut - lookback)
        return tuple(self._candles[start:cut])

    def __len__(self) -> int:
        return len(self._candles)


@dataclass
class ReplayState:
    """Concrete implementation of PointInTimeState.

    Built once from a ``Catalog`` (or from an in-memory fixture in tests) and
    then advanced by setting ``now`` as the replay clock ticks forward.

    **Thread safety**: not thread-safe. The replay engine is single-threaded
    by design — parallelism is at the fold level (separate ``ReplayState``
    instances per fold), not within a replay.

    **Publication delay**: a candle at ``ts`` on a ``tf``-second timeframe is
    available at ``ts + tf + publication_delay_seconds``. Never at ``ts``. The
    delay is zero by default but must be set to a value > 0 in any run that
    claims to model realistic data latency. The test suite sets it to zero so
    it can control ``available_time`` precisely via the ``add_bar`` method.

    **Why ``now`` is mutable**: the replay engine advances the clock. Making
    ``now`` immutable would require rebuilding the state on every tick, which
    is O(bars) per tick. The alternative — a mutable ``now`` — means callers
    must be careful not to let the clock go backwards, which the engine
    enforces by construction (bars are processed in ``available_time`` order).

    **Internal store layout**: one ``_CandleStore`` per (asset_id, timeframe)
    key. Pool states and quote ladders use a simpler model: one list per key,
    sorted by ``available_time``. Because we expect O(1) or O(small-constant)
    pool-state and ladder queries per tick, a bisect on a flat list is simpler
    and fast enough.
    """

    now: float
    publication_delay_seconds: float = DEFAULT_PUBLICATION_DELAY_SECONDS

    _candle_stores: dict[tuple[str, str], _CandleStore] = field(
        default_factory=dict, init=False, repr=False
    )
    _pool_states: dict[str, list[tuple[float, PoolState]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _pool_state_times: dict[str, list[float]] = field(
        default_factory=dict, init=False, repr=False
    )
    _ladders: dict[tuple[str, str], list[tuple[float, QuoteLadder]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _ladder_times: dict[tuple[str, str], list[float]] = field(
        default_factory=dict, init=False, repr=False
    )
    _universe_schedule: list[tuple[float, frozenset[str]]] = field(
        default_factory=list, init=False, repr=False
    )
    _universe_times: list[float] = field(default_factory=list, init=False, repr=False)

    # ------------------------------------------------------------------
    # PointInTimeState protocol implementation
    # ------------------------------------------------------------------

    def bars(
        self, asset_id: str, timeframe: Timeframe, *, lookback: int
    ) -> tuple[Candle, ...]:
        """The last ``lookback`` closed bars available at ``self.now``.

        Returns oldest-first. Returns an empty tuple when:
        - no data has been loaded for this (asset_id, timeframe)
        - ``self.now`` is before the first available bar
        - ``lookback`` is 0

        Never raises: a missing series is represented as empty data, not
        as an error. The caller (feature pipeline) is responsible for
        checking that it has enough bars before computing a feature.
        """
        key = (asset_id, timeframe.value)
        store = self._candle_stores.get(key)
        if store is None or lookback <= 0:
            return ()
        return store.query(self.now, lookback)

    def snapshot(self, asset_id: str) -> None:
        """Always None at TIER_0.

        The protocol says ``CoinSnapshot | None``; at TIER_0 we have no
        snapshot history, so None is the honest answer. The execution model
        must construct its own snapshot from bars via ``build_report``-style
        reconstruction.

        This method exists on the protocol so that TIER_1+ implementations
        can return a real snapshot without changing callers.
        """
        return None

    def universe(self) -> frozenset[str]:
        """The tradable universe at ``self.now``.

        Returns the most recent universe update whose ``available_time <=
        self.now``. Returns an empty frozenset if no universe update has
        been loaded or if ``self.now`` is before the first update.

        The empty-set default is safe: a strategy that cannot see any coins
        produces no forecasts and no orders, which is a conservative error
        rather than a dangerous one.
        """
        if not self._universe_times:
            return frozenset()
        idx = bisect.bisect_right(self._universe_times, self.now)
        if idx == 0:
            return frozenset()
        _, members = self._universe_schedule[idx - 1]
        return members

    def pool_state(self, pool_id: str) -> PoolState | None:
        """Most recent pool state at or before ``self.now``, or None."""
        times = self._pool_state_times.get(pool_id)
        if not times:
            return None
        idx = bisect.bisect_right(times, self.now)
        if idx == 0:
            return None
        _, state = self._pool_states[pool_id][idx - 1]
        return state

    def quote_ladder(self, asset_id: str, side: Side) -> QuoteLadder | None:
        """Most recent quote ladder at or before ``self.now``, or None."""
        key = (asset_id, side.value)
        times = self._ladder_times.get(key)
        if not times:
            return None
        idx = bisect.bisect_right(times, self.now)
        if idx == 0:
            return None
        _, ladder = self._ladders[key][idx - 1]
        return ladder

    # ------------------------------------------------------------------
    # Mutation methods (called by the replay engine loader, not by strategy)
    # ------------------------------------------------------------------

    def add_bar(
        self,
        candle: Candle,
        *,
        asset_id: str,
        timeframe: Timeframe,
        available_time: float | None = None,
    ) -> None:
        """Load one closed bar into the store.

        ``available_time`` defaults to ``candle.ts + interval + publication_delay``.
        The caller may override it for testing (e.g., to make a bar visible
        immediately) but must not set it earlier than ``candle.ts``.

        Raises ``ValueError`` if the candle is not closed — see ``_CandleStore.add``.
        This is the structural guard against forming-bar leakage: the only path
        into the store raises if ``closed is False``.
        """
        if available_time is None:
            interval = _INTERVAL_SECONDS[timeframe]
            available_time = candle.ts + interval + self.publication_delay_seconds

        key = (asset_id, timeframe.value)
        if key not in self._candle_stores:
            self._candle_stores[key] = _CandleStore()
        self._candle_stores[key].add(candle, available_time)

    def add_pool_state(self, state: PoolState) -> None:
        """Load a pool state into the sorted store."""
        pool_id = state.pool_id
        if pool_id not in self._pool_states:
            self._pool_states[pool_id] = []
            self._pool_state_times[pool_id] = []
        idx = bisect.bisect_right(self._pool_state_times[pool_id], state.available_time)
        self._pool_states[pool_id].insert(idx, (state.available_time, state))
        self._pool_state_times[pool_id].insert(idx, state.available_time)

    def add_quote_ladder(self, ladder: QuoteLadder) -> None:
        """Load a quote ladder into the sorted store."""
        key = (ladder.asset_id, ladder.side)
        if key not in self._ladders:
            self._ladders[key] = []
            self._ladder_times[key] = []
        idx = bisect.bisect_right(self._ladder_times[key], ladder.available_time)
        self._ladders[key].insert(idx, (ladder.available_time, ladder))
        self._ladder_times[key].insert(idx, ladder.available_time)

    def set_universe(
        self, members: frozenset[str], *, available_time: float
    ) -> None:
        """Register a universe snapshot valid from ``available_time``.

        Multiple calls accumulate; the most recent snapshot at or before
        ``self.now`` is returned by ``universe()``. This is how the replay
        engine implements point-in-time universe reconstruction: it loads
        all historical universe snapshots up front, then ``universe()`` returns
        the right one as the clock advances.

        The empty frozenset is a valid universe (all coins delisted). It is
        distinct from "no snapshot loaded" (unknown membership).
        """
        idx = bisect.bisect_right(self._universe_times, available_time)
        self._universe_schedule.insert(idx, (available_time, members))
        self._universe_times.insert(idx, available_time)

    # ------------------------------------------------------------------
    # Convenience: bulk-load from the catalog
    # ------------------------------------------------------------------

    def load_from_catalog(
        self,
        catalog: object,  # Catalog — forward reference to avoid circular import
        *,
        pool_to_asset: dict[str, str],  # pool_id -> mint address
        pool_to_timeframe: dict[str, Timeframe] | None = None,
    ) -> None:
        """Load all bars from a Catalog into this state.

        ``pool_to_asset`` maps pool addresses to mint addresses (asset_ids).
        This mapping must be provided by the caller from the universe file —
        the catalog does not know which mint a pool belongs to.

        ``pool_to_timeframe`` optionally restricts which timeframes are loaded.
        If None, all timeframes present in the catalog are loaded.

        Bars are loaded in ascending ``ts`` order per partition. The
        ``available_time`` is computed from the bar's ``ts``, the interval, and
        ``self.publication_delay_seconds``.
        """
        from ..backfill import RawBar
        from .catalog import Catalog

        if not isinstance(catalog, Catalog):
            raise TypeError(f"expected Catalog, got {type(catalog)!r}")

        for partition in catalog.partitions():
            asset_id = pool_to_asset.get(partition.pool)
            if asset_id is None:
                continue  # skip pools not in the universe mapping

            try:
                tf = Timeframe(partition.timeframe)
            except ValueError:
                continue  # skip unknown timeframes

            if pool_to_timeframe is not None and tf not in pool_to_timeframe.values():
                continue

            raw_bars = catalog.read_bars(partition.pool, tf)
            interval = _INTERVAL_SECONDS[tf]

            for raw in raw_bars:
                candle = Candle(
                    ts=raw.ts,
                    open=raw.open,
                    high=raw.high,
                    low=raw.low,
                    close=raw.close,
                    volume=raw.volume,
                    closed=True,
                )
                available_time = raw.ts + interval + self.publication_delay_seconds
                self.add_bar(
                    candle,
                    asset_id=asset_id,
                    timeframe=tf,
                    available_time=available_time,
                )

    def bar_count(self, asset_id: str, timeframe: Timeframe) -> int:
        """Total bars loaded for (asset_id, timeframe), regardless of ``now``."""
        key = (asset_id, timeframe.value)
        store = self._candle_stores.get(key)
        return 0 if store is None else len(store)

    def as_candle_series(
        self, asset_id: str, timeframe: Timeframe, pool_id: str
    ) -> CandleSeries | None:
        """Return all bars available at ``self.now`` as a ``CandleSeries``.

        Used by the signal pipeline when it needs a ``CandleSeries`` rather
        than a raw tuple of candles. The provenance is constructed from the
        most recent bar's timestamp, not from a backfill manifest, so the
        series is marked as coming from "replay" rather than from the original
        backfill source. This is intentional: the replay has already filtered
        for point-in-time correctness, and labelling the series with the
        backfill source would imply a provenance that is not accurate.

        Returns ``None`` when no bars are available.
        """
        interval = _INTERVAL_SECONDS[timeframe]
        candles = self.bars(asset_id, timeframe, lookback=10_000)
        if not candles:
            return None
        return CandleSeries(
            timeframe=timeframe,
            pool_address=pool_id,
            candles=candles,
            interval_seconds=interval,
            provenance=Provenance(
                source="replay",
                receive_time=candles[-1].ts + interval,
                event_time=candles[-1].ts,
                available_time=candles[-1].ts + interval + self.publication_delay_seconds,
                quality=DataQuality.OK,
            ),
            missing_intervals=0,  # gaps were recorded at load time, not tracked here
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INTERVAL_SECONDS: dict[Timeframe, float] = {
    Timeframe.M5: 300.0,
    Timeframe.H1: 3600.0,
}


__all__ = [
    "DEFAULT_PUBLICATION_DELAY_SECONDS",
    "PointInTimeState",
    "ReplayState",
]
