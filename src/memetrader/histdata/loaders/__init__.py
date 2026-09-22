"""Historical loaders: turn on-disk history into point-in-time event streams.

Every loader in this package shares one contract:

* It yields ``types.HistoricalEvent`` in **non-decreasing** ``sort_key`` order —
  the ``EventQueue`` (``backtest/event_queue.py``) requires pre-sorted streams
  per source and raises ``EventQueueError`` on an out-of-order one, so a loader
  that gets this wrong fails loudly the first time it is wired into a replay,
  not quietly in a report six weeks later.
* It sets ``available_time`` following ``BACKTEST-CONTRACTS.md`` §1: a record is
  never available at the instant it happened. Bars are available at
  ``ts + interval + publication_delay``; social data at collector receipt time;
  everything else at its own honest receive time.
* It never raises on missing data. A loader whose backing partition does not
  exist on disk returns a clean empty stream — but that emptiness is not the
  same fact as "the partition exists and is quiet", and every loader exposes an
  explicit ``available(...)`` query (returning a :class:`DataAvailability`) so
  a caller can tell the two apart without guessing from an empty iterator.
* It exposes a module-level ``FIDELITY_TIER`` constant: the
  ``types.FidelityTier`` this loader's data can support *when present*. This is
  a ceiling, not a claim about what any given run actually has — the run
  manifest is what says which tier a run achieved.

Modules:
    ohlcv          — TIER_0. The only loader with real data today: ~801,957
                     bars across 24 coins, 1h + 5m, lazily streamed from the
                     gzipped JSONL backfill files.
    quote_ladders  — TIER_2. Reads exactly what ``shadow.py`` writes:
                     ``history/shadow/<date>/<symbol>.jsonl.gz`` plus its
                     per-day manifest.
    swaps          — TIER_1. Scaffolding: no swap-level data has been
                     collected yet.
    pool_states    — TIER_2. Scaffolding: no pool-state snapshots exist yet.
    onchain        — TIER_1. Scaffolding: token-metadata/rename events. No
                     schema for this exists in ``schemas.py`` yet, so this
                     module defines its own minimal on-disk record locally
                     (documented in the module docstring) rather than
                     touching the frozen schema file.
    social         — Feature-only; does not raise the execution fidelity
                     tier. Scaffolding: no Reddit/Arctic-Shift capture has
                     been wired to this on-disk layout yet.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["DataAvailability"]


class DataAvailability(StrEnum):
    """Whether a loader's backing partition exists, and if so, whether it has rows.

    This is the structural fix for the §0 rule that ``None`` and ``0`` must
    never collapse into each other, applied to whole partitions rather than
    single fields: "we never collected this" and "we collected it and it was
    quiet" are different facts, and a loader that only returns an empty
    iterator for both has thrown that distinction away.
    """

    # The partition's file/directory does not exist on disk at all. Nobody has
    # collected this stream for this asset/date/pool.
    NOT_COLLECTED = "not_collected"
    # The partition exists (a file/manifest entry is present) but contains zero
    # records. This is a real, positive observation: the collector ran and
    # found nothing to report.
    PRESENT_EMPTY = "present_empty"
    # The partition exists and has at least one record.
    PRESENT = "present"
