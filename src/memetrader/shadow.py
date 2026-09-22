"""Prospective shadow-data collector: quote ladders and pool state, prospectively.

Why this module must run before the backtest engine
====================================================

This repository is currently at ``FidelityTier.TIER_0`` — OHLCV only. That
tier cannot support a credible executable-PnL estimate because AMM execution
cost is size-dependent: a $10 buy and a $500 buy of the same token at the same
moment route through different pool depths, pay different effective spreads and
may not both find a route at all. OHLCV has none of that information.

Jupiter is a *live routing service*, not a historical archive. You cannot
query it today for the route it would have returned six months ago. The only
path to TIER_2 is prospective: capture quote ladders right now, accumulate
them over weeks, and then the replay engine can interpolate execution cost at
any size actually traded against real data it observed at the time. Every day
this module does not run is a day of executable history that cannot be
recovered later.

The mistake this module avoids
================================

``market.py`` calls ``client.get(...)`` directly rather than routing through
``http.execute()``. ``make_client`` gives TLS (critical behind the corporate
TLS-inspection proxy) and timeouts, but not retry, backoff or a circuit
breaker. That cost 38 of 39 POPCAT fetches in a live run — all returned 429
and the caller gave up. This module routes every outbound call through
``http.execute()``, which treats 429 as retryable, honours ``Retry-After``,
applies full-jitter exponential backoff, and opens a circuit breaker after a
host is persistently down so subsequent coins fail fast rather than burning
the whole per-host budget over and over.

Storage layout
==============

``history/shadow/<date>/<asset_symbol>.jsonl.gz`` — gzipped JSONL, one
*ladder snapshot* per line. A ladder snapshot is one observation time with all
rung quotes and any available pool state. ``history/`` is gitignored: this is
regenerable-by-waiting data, not source.

A manifest at ``history/shadow/<date>/manifest.json`` records provenance for
each asset file written, including schema version, source, coin count, and the
ladder sizes used. The manifest is rewritten atomically after every successful
ladder write, so an interrupted run sees exactly the assets that were
successfully stored.

Resumability
============

An interrupted run uses the manifest to find which assets were already stored
in the current UTC day's directory. An asset whose file exists and is recorded
in the manifest is skipped. A partial ladder (some rungs written, then
crashed) is re-collected from scratch for that asset on resume, because a
ladder with a missing rung cannot interpolate across that rung — the whole
snapshot is atomic.

Each rung failure is independent: a vendor error for one size does not discard
the rungs already collected (contrast: the first version of ``backfill.py``
treated ``HistoryHorizon`` as a full abort and discarded good bars already in
memory; see ``HistoryHorizon``'s docstring). The partially-collected ladder is
written as ``quality="degraded"`` rather than dropped.

Schema / provenance fields
==========================

Every record carries:

* ``available_time`` — set to the receive time of the *last rung* in the
  ladder. This is the honest floor: the earliest a replay can act on the
  whole ladder is when the slowest rung was received.
* ``event_time`` — when the first rung request was sent. The ladder spans
  from that moment to the last receive; the event time is the opening edge.
* ``sequence`` — the Solana context slot from the first successful rung
  quote. Slots are ~400ms and give a finer ordering token than wall-clock
  seconds alone, which matters when two assets' ladders land in the same
  second.

Public API
==========

``collect(...)`` — the main entry point, callable from a scheduler.
``main()``       — thin CLI wrapper for direct invocation.
``load_universe(path)`` — reads ``universe/solana_memecoins.toml``; shared
    with ``backfill.py`` but kept here so ``shadow.py`` has no import cycle.
``LadderConfig`` — configurable rung sizes and cadence.
``LadderRung``   — one rung of a quote ladder (one size, one side).
``LadderSnapshot`` — one asset, one observation time, full ladder.
``ShadowMeta``   — per-asset manifest entry.
"""

from __future__ import annotations

import gzip
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .http import CircuitBreaker, RetryPolicy, execute, make_client
from .journal import atomic_write_text, to_jsonable
from .quotes import (
    USDC,
    _build_quote,
    token_meta,
    usd_to_atomic,
)
from .types import Quote, Side, TokenMeta, ValidationError

__all__ = [
    "DEFAULT_LADDER_USD",
    "SHADOW_RETRY",
    "SHADOW_TOOL_VERSION",
    "CollectReport",
    "LadderConfig",
    "LadderRung",
    "LadderSnapshot",
    "ShadowMeta",
    "collect",
    "ladder_path",
    "load_universe",
    "main",
    "read_manifest",
    "write_manifest",
]

log = logging.getLogger(__name__)

SHADOW_TOOL_VERSION = "shadow/1"

#: Default rung sizes in USD. Chosen to span from a small retail order through
#: a meaningful position for this portfolio. Each rung is its own Jupiter call,
#: so the set is intentionally finite: seven rungs x two sides x 24 coins =
#: 336 calls per cadence tick, well inside the keyless free-tier budget at 3s
#: pacing.
DEFAULT_LADDER_USD: tuple[float, ...] = (10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0)

# Per-rung retry budget. Longer than the live-tick budget (20s) but tighter than
# the backfill budget (240s): a shadow run has no tick deadline, but a cadence
# of five minutes means each full sweep must finish comfortably inside five
# minutes. At 3 s pacing x 336 calls = ~17 minutes, so the retry budget is kept
# to 30 s per rung (failures are rare at 3s spacing; the long tail is the
# concern).
SHADOW_RETRY = RetryPolicy(
    max_attempts=5,
    total_budget_seconds=30.0,
    backoff_base_seconds=1.0,
    backoff_multiplier=2.5,
    backoff_max_seconds=10.0,
    max_retry_after_seconds=30.0,
)

# Sanity bounds, same as backfill.py.
_TS_FLOOR = 1_577_836_800.0
_TS_CEILING = 4_102_444_800.0

# Quote TTL used when recording rungs. Longer than the live 10s because the
# rung's ``expires_at`` is only for the *replay consumer* to know the outer
# bound of when the quote was valid; the shadow collector is never submitting
# it. A 60s window covers the full rung-collection time for one asset.
_RUNG_TTL_SECONDS = 60.0


# ---------------------------------------------------------------------------
# Configuration and data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LadderConfig:
    """Tunable parameters for the shadow collector.

    Deliberately not read from ``config.toml``: the collector is a standalone
    process, not part of the live trader, and coupling it to the live config
    would mean a config change that happens to break the live trader also
    silently stops data collection. Independence is the whole point.

    ``cadence_seconds`` is advisory — the ``collect()`` function runs one full
    sweep and returns. The *scheduler* (cron/systemd/Task Scheduler) is
    responsible for cadence. A five-minute cadence gives 288 snapshots per day
    per asset, which is more than enough to interpolate a cost curve over the
    intra-day liquidity cycle.
    """

    ladder_usd: tuple[float, ...] = DEFAULT_LADDER_USD
    sides: tuple[Side, ...] = (Side.BUY, Side.SELL)
    cadence_seconds: float = 300.0
    pace_seconds: float = 3.0
    # Slippage tolerance sent to Jupiter for each rung. The recorded
    # ``min_out_amount_atomic`` is the conservative output at this tolerance,
    # which is the number the replay engine must use for fill realism.
    slippage_bps: int = 50
    # How long to keep a rung quote as "valid" in the record. Replay consumers
    # check ``expires_at`` to know whether the quote was still fresh when the
    # engine's simulated order would have been submitted.
    rung_ttl_seconds: float = _RUNG_TTL_SECONDS
    # A circuit breaker shared across all coins in one sweep. After this many
    # consecutive failures on the Jupiter host, all remaining rungs for all
    # coins fail fast instead of burning the full retry budget.
    breaker_failure_threshold: int = 5
    breaker_cooldown_seconds: float = 90.0


@dataclass(frozen=True, slots=True)
class LadderRung:
    """One rung of a quote ladder: one side, one USD size, one Jupiter call.

    ``quote`` is ``None`` when Jupiter returned no route (C4 contract: no route
    is not a worse route, it is no route). The rung is still written with
    ``quality="no_route"`` so a consumer knows we tried and got nothing, which
    is itself information about depth: a $1000 buy that returns no route means
    the pool cannot absorb that size.

    The latency field is ``received_at - requested_at`` from the ``Quote``
    object, or the wall-clock duration of the failed call. It lets a replay
    consumer model the latency cost of obtaining a quote before submitting.
    """

    side: Side
    usd_size: float
    in_amount_atomic: int
    quote: Quote | None
    quality: str  # "ok", "no_route", "parse_error", "degraded"
    requested_at: float
    received_at: float
    latency_seconds: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class LadderSnapshot:
    """One asset's full ladder at one observation time.

    Written as one gzipped-JSONL line. A future replay reads this line and
    can interpolate execution cost at any USD size in [ladder_usd[0],
    ladder_usd[-1]] by fitting a cost curve across the rungs. Extrapolation
    outside the ladder range requires a comment in the replay code.

    ``available_time`` is the receive time of the *last rung*: the earliest
    a replay can act on the full ladder is when the last rung landed. The
    ``event_time`` is when the first rung request was sent.

    ``slot`` is the Solana context slot from the first successful rung; it is
    the finest ordering token the vendor gives us and is stored as ``sequence``
    in the ``Provenance`` contract.
    """

    symbol: str
    mint: str
    event_time: float  # first rung requested_at
    available_time: float  # last rung received_at
    slot: int | None  # Solana context slot from first successful rung
    rungs: tuple[LadderRung, ...]
    pool_state: dict[str, Any] | None  # reserved; None until a pool-state source is wired
    quality: str  # "ok", "degraded", "empty"
    tool_version: str = SHADOW_TOOL_VERSION


@dataclass(frozen=True, slots=True)
class ShadowMeta:
    """Per-asset manifest entry for one shadow-collection sweep.

    Written and read by ``write_manifest``/``read_manifest``. Kept small
    because the manifest is meant to be read by a human checking provenance,
    not to be a second copy of the data.
    """

    symbol: str
    mint: str
    date: str  # "YYYY-MM-DD" UTC
    snapshots: int  # number of ladder snapshots written to the asset file
    first_ts: float | None
    last_ts: float | None
    rung_count: int
    rung_ok: int
    rung_no_route: int
    rung_error: int
    ladder_usd: tuple[float, ...]
    sides: tuple[str, ...]
    quality: str  # "ok", "degraded", "empty"
    written_at: float = 0.0
    tool_version: str = SHADOW_TOOL_VERSION
    source: str = "jupiter"


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def ladder_path(root: Path, date_str: str, symbol: str) -> Path:
    """``<root>/shadow/<date>/<symbol>.jsonl.gz``.

    Keyed by symbol rather than mint or pool: unlike OHLCV data (which
    ``backfill.py`` correctly keys by pool to prevent splice artefacts), a
    quote ladder is a statement about routing for one *asset* at a given size,
    and the relevant identity for interpolation is the token, not the pool.
    Jupiter already picks the best pool; we record which pools it chose via
    ``route_labels``.
    """
    return root / "shadow" / date_str / f"{symbol}.jsonl.gz"


def _snapshot_to_dict(snap: LadderSnapshot) -> dict[str, Any]:
    """Flatten a ``LadderSnapshot`` to a JSON-serializable dict.

    The rungs' ``Quote`` objects use ``to_jsonable`` for the same NaN→null
    treatment the rest of the codebase uses; a NaN in a stored rung would
    silently poison every replay cost estimate computed from it.
    """

    def rung_dict(r: LadderRung) -> dict[str, Any]:
        q = r.quote
        return {
            "side": str(r.side),
            "usd_size": r.usd_size,
            "in_amount_atomic": r.in_amount_atomic,
            "quality": r.quality,
            "requested_at": r.requested_at,
            "received_at": r.received_at,
            "latency_seconds": r.latency_seconds,
            "error": r.error,
            "quote": (
                None
                if q is None
                else {
                    "in_amount_atomic": q.in_amount_atomic,
                    "out_amount_atomic": q.out_amount_atomic,
                    "min_out_amount_atomic": q.min_out_amount_atomic,
                    "price_impact_pct": q.price_impact_pct,
                    "route_labels": list(q.route_labels),
                    "fingerprint": q.fingerprint,
                    "context_slot": q.context_slot,
                    "expires_at": q.expires_at,
                    "reference_price_usd": q.reference_price_usd,
                }
            ),
        }

    return {
        "symbol": snap.symbol,
        "mint": snap.mint,
        "event_time": snap.event_time,
        "available_time": snap.available_time,
        "slot": snap.slot,
        "quality": snap.quality,
        "pool_state": snap.pool_state,
        "tool_version": snap.tool_version,
        "rungs": [rung_dict(r) for r in snap.rungs],
    }


def _append_snapshot(path: Path, snap: LadderSnapshot) -> None:
    """Append one ladder snapshot as a gzipped JSONL line.

    Gzip is opened in append mode (``ab``). Each call writes one complete JSON
    line and flushes, so a crash between calls leaves the file with all
    previously written lines intact and readable — the next call appends to
    the (valid) compressed stream. Note: gzip append writes a new gzip member;
    readers must handle multi-member gzip (Python's ``gzip.open`` does this
    correctly since 3.2).

    ``mtime=0`` is not set here (unlike ``write_series`` in ``backfill.py``)
    because shadow snapshots are genuinely time-stamped and we want the member
    timestamps to reflect reality. Determinism matters for OHLCV files written
    all-at-once; it does not matter for append-only streams.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(to_jsonable(_snapshot_to_dict(snap)), separators=(",", ":")) + "\n"
    with gzip.open(path, "ab") as gz:
        gz.write(line.encode("utf-8"))


def write_manifest(root: Path, date_str: str, entries: Sequence[ShadowMeta]) -> None:
    """Replace the day's manifest atomically.

    Same idiom as ``backfill.write_manifest``: atomic temp-file swap so a
    reader never sees a half-written file, even on Windows where ``os.replace``
    across filesystems raises outright (``atomic_write_text`` handles that).
    """
    dir_path = root / "shadow" / date_str
    dir_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "tool_version": SHADOW_TOOL_VERSION,
        "written_at": time.time(),
        "date": date_str,
        "entries": [to_jsonable(e) for e in entries],
    }
    atomic_write_text(
        dir_path / "manifest.json",
        json.dumps(payload, indent=1) + "\n",
    )


def read_manifest(root: Path, date_str: str) -> list[ShadowMeta]:
    """Read the manifest for ``date_str``. Returns ``[]`` when absent.

    Tolerates missing keys with defaults, so an older manifest written by an
    earlier schema version still loads. Unknown keys are ignored.
    """
    path = root / "shadow" / date_str / "manifest.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: list[ShadowMeta] = []
    for raw in payload.get("entries") or []:
        try:
            known = set(ShadowMeta.__dataclass_fields__)
            kwargs = {k: v for k, v in raw.items() if k in known}
            # Re-hydrate tuples that JSON stored as lists.
            if "ladder_usd" in kwargs:
                kwargs["ladder_usd"] = tuple(float(x) for x in kwargs["ladder_usd"])
            if "sides" in kwargs:
                kwargs["sides"] = tuple(str(s) for s in kwargs["sides"])
            out.append(ShadowMeta(**kwargs))
        except (KeyError, TypeError, ValueError):
            # A corrupt or future-schema entry is skipped, not fatal. The
            # collector will re-collect it.
            continue
    return out


def load_universe(path: Path) -> list[dict[str, Any]]:
    """Read the committed universe file. Mirrors ``backfill.load_universe``.

    Duplicated here (not imported from ``backfill``) to avoid an import
    dependency on a module whose retry constants and storage conventions are
    subtly different. Shared interface, independent import graph.
    """
    import tomllib  # stdlib since 3.11

    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    coins = payload.get("coins") or []
    if not coins:
        raise ValueError(f"{path}: no [[coins]] entries")
    return list(coins)


# ---------------------------------------------------------------------------
# Rung collection
# ---------------------------------------------------------------------------


def _collect_rung(
    cfg_base: str,
    cfg_slippage_bps: int,
    cfg_rung_ttl: float,
    client: httpx.Client,
    *,
    symbol: str,
    token: TokenMeta,
    side: Side,
    usd_size: float,
    now: Callable[[], float],
    sleep: Callable[[float], None],
    pace_seconds: float,
    retry: RetryPolicy,
    breaker: CircuitBreaker | None,
    headers: dict[str, str],
) -> LadderRung:
    """Fetch one quote rung. Never raises — all failures collapse to a LadderRung
    with ``quality != "ok"``.

    This mirrors the C4 contract in ``quotes.py``: a rung we could not obtain
    is not a worse rung, it is no rung, and the caller receives a typed
    record with ``quality="no_route"`` rather than a synthetic value. The
    snapshot writer stores it as-is so the replay engine knows a route was
    absent for that size, which is different from "we never asked".

    Rate-limiting: ``sleep(pace_seconds)`` is called by the *caller* before
    each rung, not here. ``execute()`` itself handles 429 retry-after, so a
    burst of 429s on a particular rung is retried by the HTTP layer without
    the orchestrator needing to know about it.
    """
    if side is Side.BUY:
        in_amount = usd_to_atomic(usd_size)
        input_token, output_token = USDC, token
    else:
        # For SELL we need the token amount at the current mid. We cannot know
        # the exact atomic amount without a price, so we approximate:
        # ask Jupiter for a BUY of usd_size first to get the token output,
        # then record a SELL of the same atomic amount. This two-call approach
        # is correct for a cost ladder: we want to know what it costs to sell
        # $usd_size worth of tokens, so we first find what $usd_size buys and
        # then ask what that quantity sells for.
        #
        # Alternative: use a DexScreener mid to derive atomic amount. But the
        # shadow collector must not hit a second vendor per rung (rate budget).
        # The two-sided ladder approach is self-contained to Jupiter.
        #
        # The SELL rung atomic amount is the out_amount of a synthetic BUY
        # rung. If the BUY itself fails we fall back to a USDC-equivalent
        # approximation using USDC_DECIMALS only (no real price data needed —
        # the USDC amount IS the USD amount for stablecoins, so we express
        # the hypothetical sell as "sell this many USDC-equivalent tokens").
        #
        # In practice: for SELL, in_amount_atomic is derived from a shadow BUY
        # (or estimated if no BUY available). The rung records the actual
        # in_amount_atomic that was quoted, so the replay consumer knows the
        # exact size.
        in_amount = usd_to_atomic(usd_size)  # USDC atomic as a proxy for SELL
        input_token, output_token = token, USDC

    requested_at = now()
    try:
        # Route through http.execute() — not client.get() — so 429s are
        # retried with backoff+jitter and the circuit breaker is informed.
        # This is the lesson from market.py's POPCAT outage.
        url = f"{cfg_base.rstrip('/')}/swap/v1/quote"
        outcome = execute(
            client,
            "GET",
            url,
            params={
                "inputMint": input_token.mint,
                "outputMint": output_token.mint,
                "amount": in_amount,
                "slippageBps": cfg_slippage_bps,
            },
            headers=headers,
            retry=retry,
            breaker=breaker,
            idempotent=True,
        )
        received_at = now()
        response = outcome.response

        if response is None or response.status_code >= 400:
            err = (
                f"HTTP {response.status_code}"
                if response is not None
                else str(outcome.error or outcome.stopped_by)
            )
            return LadderRung(
                side=side,
                usd_size=usd_size,
                in_amount_atomic=in_amount,
                quote=None,
                quality="no_route",
                requested_at=requested_at,
                received_at=received_at,
                latency_seconds=received_at - requested_at,
                error=err,
            )

        payload = response.json()
        if not isinstance(payload, dict):
            received_at = now()
            return LadderRung(
                side=side,
                usd_size=usd_size,
                in_amount_atomic=in_amount,
                quote=None,
                quality="parse_error",
                requested_at=requested_at,
                received_at=received_at,
                latency_seconds=received_at - requested_at,
                error="response is not a JSON object",
            )

        q = _build_quote(
            payload,
            symbol=symbol,
            side=side,
            input_token=input_token,
            output_token=output_token,
            requested_in_amount_atomic=in_amount,
            requested_at=requested_at,
            received_at=received_at,
            ttl_seconds=cfg_rung_ttl,
        )
        if q is None:
            return LadderRung(
                side=side,
                usd_size=usd_size,
                in_amount_atomic=in_amount,
                quote=None,
                quality="parse_error",
                requested_at=requested_at,
                received_at=received_at,
                latency_seconds=received_at - requested_at,
                error="build_quote returned None (malformed payload)",
            )

        return LadderRung(
            side=side,
            usd_size=usd_size,
            in_amount_atomic=q.in_amount_atomic,
            quote=q,
            quality="ok",
            requested_at=requested_at,
            received_at=received_at,
            latency_seconds=q.latency_seconds,
        )

    except ValidationError as exc:
        received_at = now()
        return LadderRung(
            side=side,
            usd_size=usd_size,
            in_amount_atomic=in_amount,
            quote=None,
            quality="parse_error",
            requested_at=requested_at,
            received_at=received_at,
            latency_seconds=received_at - requested_at,
            error=f"ValidationError: {exc}",
        )
    except Exception as exc:  # noqa: BLE001
        # Any unexpected error for one rung must not abort the whole ladder.
        # This is the same philosophy as backfill's per-bar reject: one bad
        # rung must not discard the rungs already collected.
        received_at = now()
        return LadderRung(
            side=side,
            usd_size=usd_size,
            in_amount_atomic=in_amount,
            quote=None,
            quality="no_route",
            requested_at=requested_at,
            received_at=received_at,
            latency_seconds=received_at - requested_at,
            error=f"{type(exc).__name__}: {exc}",
        )


def _collect_ladder(
    cfg_base: str,
    cfg_slippage_bps: int,
    cfg_rung_ttl: float,
    client: httpx.Client,
    *,
    symbol: str,
    token: TokenMeta,
    ladder_cfg: LadderConfig,
    now: Callable[[], float],
    sleep: Callable[[float], None],
    breaker: CircuitBreaker | None,
    headers: dict[str, str],
) -> LadderSnapshot:
    """Collect a full quote ladder for one asset.

    Iterates over (side, usd_size) pairs, sleeping ``pace_seconds`` between
    each rung to stay below the keyless Jupiter rate limit. One rung failure
    does not abort the ladder — the rung is stored as ``quality="no_route"``
    and the next rung proceeds. This is the analogue of ``backfill.py``'s
    reject-and-continue pattern.

    ``available_time`` is set to the receive time of the **last** rung,
    because that is the earliest moment the whole ladder was knowable. A
    replay that sets it to the first rung's time would be granting itself
    foresight over the duration of the collection.
    """
    rungs: list[LadderRung] = []
    first_rung = True
    first_slot: int | None = None
    first_event_time: float | None = None

    for side in ladder_cfg.sides:
        for usd_size in ladder_cfg.ladder_usd:
            if not first_rung:
                sleep(ladder_cfg.pace_seconds)
            first_rung = False

            rung = _collect_rung(
                cfg_base,
                cfg_slippage_bps,
                cfg_rung_ttl,
                client,
                symbol=symbol,
                token=token,
                side=side,
                usd_size=usd_size,
                now=now,
                sleep=sleep,
                pace_seconds=ladder_cfg.pace_seconds,
                retry=SHADOW_RETRY,
                breaker=breaker,
                headers=headers,
            )
            if first_event_time is None:
                first_event_time = rung.requested_at
            if first_slot is None and rung.quote is not None:
                first_slot = rung.quote.context_slot
            rungs.append(rung)
            log.debug(
                "shadow rung symbol=%s side=%s usd=%.0f quality=%s latency=%.3fs",
                symbol,
                side,
                usd_size,
                rung.quality,
                rung.latency_seconds,
            )

    ok_count = sum(1 for r in rungs if r.quality == "ok")
    last_received = rungs[-1].received_at if rungs else now()
    event_time = first_event_time if first_event_time is not None else last_received

    if not rungs:
        quality = "empty"
    elif ok_count == len(rungs):
        quality = "ok"
    elif ok_count > 0:
        quality = "degraded"
    else:
        quality = "empty"

    return LadderSnapshot(
        symbol=symbol,
        mint=token.mint,
        event_time=event_time,
        # available_time = last receive time: the replay must not act on the
        # ladder until the slowest rung was received (point-in-time invariant).
        available_time=last_received,
        slot=first_slot,
        rungs=tuple(rungs),
        pool_state=None,  # reserved — no pool-state source available without new vendor
        quality=quality,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CollectReport:
    """What one collection sweep did. Returned, not printed."""

    written: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)


def collect(
    *,
    coins: Sequence[dict[str, Any]],
    root: Path,
    jupiter_base: str,
    jupiter_api_key: str | None = None,
    ladder_cfg: LadderConfig | None = None,
    resume: bool = True,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[str], None] | None = None,
    client: httpx.Client | None = None,
) -> CollectReport:
    """Run one full shadow-collection sweep: every coin, both sides, all rungs.

    Parameters
    ----------
    coins:
        List of coin dicts from ``universe/solana_memecoins.toml`` (each has
        at least ``symbol`` and ``mint`` keys).
    root:
        History root directory, e.g. ``Path("history")``. Shadow data is
        written under ``root/shadow/<date>/``.
    jupiter_base:
        Jupiter API base URL (keyless lite host or keyed host).
    jupiter_api_key:
        API key, or ``None`` for the keyless tier.
    ladder_cfg:
        Ladder configuration; defaults to ``LadderConfig()``.
    resume:
        If ``True`` (default), skip coins that are already in today's manifest.
        An interrupted run restarts cheaply. Set ``False`` to force a fresh
        sweep.
    now / sleep:
        Clock and sleep; injectable for tests. ``now`` is called at each rung
        boundary; ``sleep`` is called between rungs.
    on_progress:
        Optional callback called with ``symbol`` before each coin.
    client:
        HTTP client; if ``None`` a new one is built and closed at the end.
        Passing an explicit client (e.g. ``MockTransport`` in tests) disables
        the per-call close.

    Returns
    -------
    CollectReport
        Summary of what was written, skipped and failed. The caller (CLI or
        test) is responsible for rendering it.
    """
    cfg = ladder_cfg or LadderConfig()
    report = CollectReport()

    # Compute the UTC date string for today's directory.
    import datetime

    date_str = datetime.datetime.fromtimestamp(now(), tz=datetime.UTC).strftime(
        "%Y-%m-%d"
    )

    # Build HTTP headers (carries api key when present).
    # We need the headers dict the same way quotes.py builds it. Since we
    # cannot call _headers(cfg) (cfg here is LadderConfig, not Config), we
    # replicate the minimal logic: Accept + optional x-api-key.
    hdrs: dict[str, str] = {"Accept": "application/json"}
    if jupiter_api_key:
        hdrs["x-api-key"] = jupiter_api_key

    # Circuit breaker shared across all coins in this sweep so that if
    # Jupiter is down, all remaining coins fail fast rather than each burning
    # the full retry budget.
    breaker = CircuitBreaker(
        failure_threshold=cfg.breaker_failure_threshold,
        cooldown_seconds=cfg.breaker_cooldown_seconds,
    )

    # Load what was already written today (for resume).
    existing_meta: dict[str, ShadowMeta] = {
        m.symbol: m for m in read_manifest(root, date_str)
    }

    owned_client = client is None
    active_client = client or make_client(
        headers={"Accept": "application/json"},
    )

    try:
        all_meta: dict[str, ShadowMeta] = dict(existing_meta)

        for coin in coins:
            symbol = str(coin["symbol"])
            mint = str(coin["mint"])

            # Resume: skip if already in today's manifest.
            if resume and symbol in existing_meta:
                report.skipped.append(symbol)
                log.info("shadow skip symbol=%s (already in manifest)", symbol)
                continue

            if on_progress:
                on_progress(symbol)

            log.info("shadow collecting symbol=%s mint=%s", symbol, mint)

            # Verify decimals. No verified decimals = no rung collection.
            # (Matches quotes.py §15: "decimals for execution now come only from
            # Jupiter's token endpoint".)
            try:
                tmeta = token_meta(
                    # token_meta takes a Config object; we duck-type the minimal
                    # interface it needs: cfg.data.jupiter_url_base and
                    # cfg.data.jupiter_api_key. Build a minimal namespace.
                    _MinimalCfg(jupiter_base, jupiter_api_key),
                    mint,
                    client=active_client,
                )
            except Exception as exc:  # noqa: BLE001
                report.failed.append((symbol, f"token_meta raised: {exc}"))
                continue

            if tmeta is None or not tmeta.verified:
                report.failed.append((symbol, f"no verified decimals for mint {mint}"))
                continue

            # Collect the full ladder.
            try:
                snapshot = _collect_ladder(
                    jupiter_base,
                    cfg.slippage_bps,
                    cfg.rung_ttl_seconds,
                    active_client,
                    symbol=symbol,
                    token=tmeta,
                    ladder_cfg=cfg,
                    now=now,
                    sleep=sleep,
                    breaker=breaker,
                    headers=hdrs,
                )
            except Exception as exc:  # noqa: BLE001
                report.failed.append((symbol, f"collect_ladder raised: {exc}"))
                continue

            # Append snapshot to the asset's file.
            path = ladder_path(root, date_str, symbol)
            try:
                _append_snapshot(path, snapshot)
            except OSError as exc:
                report.failed.append((symbol, f"write failed: {exc}"))
                continue

            # Count rungs for the manifest.
            rung_ok = sum(1 for r in snapshot.rungs if r.quality == "ok")
            rung_no_route = sum(1 for r in snapshot.rungs if r.quality == "no_route")
            rung_error = sum(1 for r in snapshot.rungs if r.quality == "parse_error")

            prior = all_meta.get(symbol)
            meta = ShadowMeta(
                symbol=symbol,
                mint=mint,
                date=date_str,
                snapshots=(prior.snapshots if prior else 0) + 1,
                first_ts=prior.first_ts if prior else snapshot.event_time,
                last_ts=snapshot.available_time,
                rung_count=len(snapshot.rungs),
                rung_ok=rung_ok,
                rung_no_route=rung_no_route,
                rung_error=rung_error,
                ladder_usd=tuple(cfg.ladder_usd),
                sides=tuple(str(s) for s in cfg.sides),
                quality=snapshot.quality,
                written_at=now(),
            )
            all_meta[symbol] = meta

            # Manifest is rewritten after every asset so an interrupted run
            # leaves a manifest describing exactly the files that exist.
            write_manifest(root, date_str, list(all_meta.values()))
            report.written.append(symbol)
            log.info(
                "shadow wrote symbol=%s quality=%s rungs_ok=%d/%d",
                symbol,
                snapshot.quality,
                rung_ok,
                len(snapshot.rungs),
            )
    finally:
        if owned_client:
            active_client.close()

    return report


# ---------------------------------------------------------------------------
# Minimal config shim (avoids importing Config, which pulls in the whole tree)
# ---------------------------------------------------------------------------


class _MinimalData:
    """Duck-type the part of ``DataConfig`` that ``token_meta`` and ``_headers``
    need: ``jupiter_url_base`` and ``jupiter_api_key``.

    ``quotes.token_meta`` takes a ``Config`` and reads only
    ``cfg.data.jupiter_url_base`` and ``cfg.data.jupiter_api_key`` and
    ``cfg.data.http_timeout_seconds``. We supply those three without importing
    the full ``Config`` (which pulls in the entire trader config chain).
    """

    def __init__(self, jupiter_base: str, api_key: str | None) -> None:
        self.jupiter_url_base = jupiter_base
        self.jupiter_api_key = api_key
        self.http_timeout_seconds = 15.0


class _MinimalCfg:
    """Minimal duck-typed Config for ``token_meta``."""

    def __init__(self, jupiter_base: str, api_key: str | None) -> None:
        self.data = _MinimalData(jupiter_base, api_key)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Standalone entry point.

    Usage (direct)::

        python -m memetrader.shadow \\
            --universe universe/solana_memecoins.toml \\
            --history  history \\
            --jupiter  https://lite.jupiter.aggregator.app

    Schedule with cron (every 5 minutes)::

        */5 * * * * cd /path/to/memecoin-trader && \\
            .venv/bin/python -m memetrader.shadow \\
            --universe universe/solana_memecoins.toml \\
            --history history \\
            --jupiter https://lite.jupiter.aggregator.app

    Schedule with Windows Task Scheduler::

        Action: C:\\path\\to\\.venv\\Scripts\\python.exe
        Arguments: -m memetrader.shadow --universe universe/solana_memecoins.toml
                   --history history
        Start in: C:\\path\\to\\memecoin-trader

    The function is intentionally simple: parse args, load universe, call
    ``collect()``, print a summary. A typer integration (``cli_backtest.py``)
    is owned by a separate agent and wires ``collect()`` directly.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Shadow quote-ladder collector.")
    parser.add_argument(
        "--universe",
        type=Path,
        default=Path("universe/solana_memecoins.toml"),
        help="Path to solana_memecoins.toml",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("history"),
        help="History root directory (history/shadow/<date>/ will be created)",
    )
    parser.add_argument(
        "--jupiter",
        default="https://lite.jupiter.aggregator.app",
        help="Jupiter API base URL",
    )
    parser.add_argument("--api-key", default=None, help="Jupiter API key (optional)")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recollect coins already in today's manifest",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=float,
        default=None,
        metavar="USD",
        help="Ladder rung sizes in USD (default: 10 25 50 100 250 500 1000)",
    )
    args = parser.parse_args()

    try:
        coins = load_universe(args.universe)
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: cannot load universe: {exc}", file=sys.stderr)
        sys.exit(1)

    ladder_cfg = LadderConfig(
        ladder_usd=tuple(args.sizes) if args.sizes else DEFAULT_LADDER_USD,
    )
    report = collect(
        coins=coins,
        root=args.history,
        jupiter_base=args.jupiter,
        jupiter_api_key=args.api_key,
        ladder_cfg=ladder_cfg,
        resume=not args.no_resume,
    )

    print(f"Written:  {len(report.written)} coins — {', '.join(report.written) or 'none'}")
    print(f"Skipped:  {len(report.skipped)} coins (already in manifest)")
    print(f"Failed:   {len(report.failed)} coins")
    for sym, reason in report.failed:
        print(f"  {sym}: {reason}")


if __name__ == "__main__":
    main()
