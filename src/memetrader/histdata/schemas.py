"""Versioned record schemas for every stream that flows through the backtest.

Each dataclass in this module is the canonical shape for one category of stored
data. All time-varying records carry the fields that make point-in-time replay
safe: ``event_time``, ``available_time``, ``received_time``, ``asset_id`` (a
mint address, never a bare symbol), ``source``, ``schema_version``, and
``quality_flags``. The distinction between these time fields is load-bearing:

* ``event_time``     — when the thing happened on-chain or at the venue.
* ``received_time``  — when this process read it.
* ``available_time`` — the earliest simulated clock at which a replay may act
                       on this value. Always >= ``event_time`` (an event cannot
                       be knowable before it happens) and always >=
                       ``received_time`` for live reads. For backfilled bars it
                       is ``ts + interval + publication_delay``, never ``ts``.

Carrying all three prevents the classic leakage: using ``event_time`` as
``available_time`` grants the strategy foresight equal to the publication delay,
which on a 0.2% hurdle is the entire edge of the baseline strategy.

``asset_id`` is always a mint address. A symbol is not unique: the same ticker
has been reused by impostor tokens, and at least two (PNUT, MYRO) had impostors
surfaced by DexScreener above the real token. A mint is unique; a symbol is a
label for display.

``schema_version`` is a semver string baked at definition time. A change to any
field that breaks backward compatibility must bump the major version, so a
catalog scan can detect mixed-schema partitions and refuse to merge them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Bump the major version when any field is renamed, removed, or changes its
# semantics. A catalog scan will detect the mismatch and refuse to concatenate
# records from two different major versions, which is the whole point: a schema
# change that passes silently produces features computed over a mix of shapes.
SCHEMA_VERSION = "1.0.0"

# The ingestion pipeline version. Separate from schema version so we can
# re-process the same raw data with a fixed parser without also bumping the
# record shape. A manifest can then tell whether a partition was ingested with
# a buggy loader even if the records look structurally correct.
INGESTION_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class CandleRecord:
    """One OHLCV bar as it flows through the replay pipeline.

    Distinct from ``types.Candle`` — that type is the live boundary object
    validated at construction; this one is the stored record before it enters
    the feature pipeline. The difference matters: we do not want to pay
    ``Candle.__post_init__`` validation on every row of a 687k-row table, but we
    do want the richer provenance fields that the backtest needs.

    ``closed`` is always ``True`` here — ``backfill.py`` never writes a forming
    bar, and a schema that admits ``closed=False`` records would make it possible
    to store them by accident.

    ``pool_id`` is the pool that priced this bar. Required alongside ``asset_id``
    because two pools for the same mint produce a synthetic price jump when
    concatenated, and the catalog must be able to refuse that join.
    """

    asset_id: str  # mint address
    pool_id: str  # pool address (pair address)
    timeframe: str  # "1h" or "5m"
    ts: float  # bar open, epoch seconds
    event_time: float  # same as ts — bar opened at this time
    available_time: float  # ts + interval + publication_delay
    received_time: float  # when the backfill downloaded it
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool  # always True in stored records; kept so readers cannot forget
    source: str  # "geckoterminal-backfill" or similar
    schema_version: str = SCHEMA_VERSION
    ingestion_version: str = INGESTION_VERSION
    quality_flags: tuple[str, ...] = ()
    slot: int | None = None  # Solana slot if known
    sequence: int | None = None  # ordering token within source

    def __post_init__(self) -> None:
        # Guard that prevents the one mistake that would silently corrupt a
        # replay: a bar that claims to have been knowable before it opened.
        # This is the same invariant types.HistoricalEvent enforces, applied
        # at the record layer so it is caught long before replay.
        if self.available_time < self.event_time:
            raise ValueError(
                f"CandleRecord for {self.asset_id} at {self.ts}: "
                f"available_time {self.available_time} < event_time {self.event_time} "
                "— a bar cannot be knowable before it opens"
            )
        for name in ("open", "high", "low", "close"):
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(
                    f"CandleRecord.{name} must be finite and positive, got {v}"
                )
        if not math.isfinite(self.volume) or self.volume < 0:
            raise ValueError(f"CandleRecord.volume must be >= 0, got {self.volume}")


@dataclass(frozen=True, slots=True)
class SwapRecord:
    """One decoded swap event from on-chain data (TIER_1 and above).

    At TIER_0 (OHLCV only) these records do not exist — the dataset has no swap
    history. This schema exists so the catalog can declare that a partition
    contains swap data and the quality layer can report coverage.

    ``in_amount_atomic`` and ``out_amount_atomic`` are integers in the token's
    smallest unit. A float amount is a rounding decision waiting to be a bug;
    the convention is the same as ``types.Quote``.

    ``venue`` identifies the AMM (raydium, orca, meteora …). Pool-keyed because
    a single mint trades on many pools and each has its own liquidity depth.
    """

    asset_id: str  # mint of the base token
    pool_id: str  # pool (pair) address
    venue: str  # dex identifier
    event_time: float  # block timestamp, epoch seconds
    available_time: float  # when replay may read this
    received_time: float
    in_mint: str
    out_mint: str
    in_amount_atomic: int
    out_amount_atomic: int
    price_impact_pct: float | None
    fees_atomic: int
    side: str  # "BUY" or "SELL" from the perspective of the base token
    slot: int | None
    tx_signature: str | None
    source: str
    schema_version: str = SCHEMA_VERSION
    ingestion_version: str = INGESTION_VERSION
    quality_flags: tuple[str, ...] = ()
    sequence: int | None = None


@dataclass(frozen=True, slots=True)
class QuoteLadderRung:
    """One (size, out, impact, route, fees, min_out) rung of a quote ladder.

    A ladder is what makes size-specific execution replay possible at TIER_2.
    Rather than one mid-price, we store the actual AMM output for each of
    several input sizes, so the execution model can interpolate rather than
    apply a constant slippage assumption.

    All amounts are in atomic units. A float amount is a rounding decision the
    caller must make explicitly, not a representation error waiting to be treated
    as a fact.

    ``min_out_atomic`` is the worst-case output at the configured slippage
    tolerance, i.e. what the real transaction would set as its slippage guard.
    A simulator that uses ``out_amount_atomic`` instead of ``min_out_atomic``
    will systematically overstate fill quality in stressed markets.
    """

    in_amount_atomic: int
    out_amount_atomic: int
    price_impact_pct: float
    route_labels: tuple[str, ...]  # which AMMs the route touches
    fees_atomic: int
    min_out_atomic: int  # out_amount_atomic * (1 - slippage_tolerance)


@dataclass(frozen=True, slots=True)
class QuoteLadder:
    """The full depth curve for one asset+side at one moment in time.

    A set of rungs, each representing the Jupiter route output for a specific
    input size. The set of rungs defines the price-impact curve: small sizes
    near the mid, large sizes showing the real cost of moving the pool.

    This is the data ``shadow.py`` collects prospectively and what separates
    TIER_2 (exact route history) from TIER_0 (bar midpoint + fallback slippage).
    Without ladder history, a 1M-USDC trade is priced the same as a $100 trade,
    which understates cost in exactly the regime where it matters most.

    ``asset_id`` is the mint of the token being bought or sold. ``side`` is
    "BUY" (USDC → token) or "SELL" (token → USDC).

    ``pool_id`` is the primary pool the route touches at the first rung; routes
    may touch different pools at different sizes, which is captured per-rung in
    ``route_labels``.

    ``context_slot`` ties the ladder to a specific chain state. A ladder
    generated from a different slot than the simulated bar close cannot be
    assumed to reflect the same depth; the execution model must apply a
    staleness penalty.
    """

    asset_id: str
    pool_id: str
    side: str  # "BUY" or "SELL"
    event_time: float  # when the quote was requested
    available_time: float  # when replay may use it
    received_time: float
    rungs: tuple[QuoteLadderRung, ...]
    context_slot: int | None
    source: str
    schema_version: str = SCHEMA_VERSION
    ingestion_version: str = INGESTION_VERSION
    quality_flags: tuple[str, ...] = ()
    sequence: int | None = None

    def best_rung_for(self, in_amount_atomic: int) -> QuoteLadderRung | None:
        """The rung with the largest ``in_amount_atomic`` <= requested size.

        Using the floor rung rather than interpolating avoids manufacturing a
        depth estimate that the AMM never actually quoted. Interpolation between
        two rungs is acceptable in the execution model but must be done there,
        with the two concrete rungs as inputs, not here.

        Returns ``None`` when the ladder is empty or the requested size is below
        every rung (the strategy is sizing smaller than we ever probed, which
        is not an error but means we have no depth data at that size).
        """
        eligible = [r for r in self.rungs if r.in_amount_atomic <= in_amount_atomic]
        if not eligible:
            return None
        return max(eligible, key=lambda r: r.in_amount_atomic)


@dataclass(frozen=True, slots=True)
class PoolState:
    """The observable state of one liquidity pool at one moment.

    Distinct from a quote: a pool state is what you would see by inspecting the
    on-chain account directly — reserves, fee tier, tick range for CLMM, etc.
    A quote is what Jupiter returned for a specific input size, which is derived
    from pool state but is not the same thing.

    At TIER_0 we have neither; at TIER_2 we may have one or both. This schema
    is the canonical shape so the execution model can declare what it needs and
    the catalog can say what it has.

    ``reserve_in_atomic`` and ``reserve_out_atomic`` are the pool's token
    reserves in atomic units at the snapshot time. A Raydium CPMM pool has two
    reserves; a CLMM pool has virtual reserves that depend on current price and
    tick range. Both are represented the same way here.

    ``fee_rate_bps`` is the pool's fee tier in basis points (e.g., 25 for a
    0.25% Raydium pool). Needed to compute the net output without re-deriving it
    from a quote, and to separate venue cost from price impact in attribution.
    """

    asset_id: str  # base token mint
    pool_id: str  # pair address
    venue: str  # dex identifier
    event_time: float
    available_time: float
    received_time: float
    reserve_in_atomic: int  # base token reserve
    reserve_out_atomic: int  # quote token reserve
    fee_rate_bps: int  # fee in basis points
    price_usd: float | None  # mid price at snapshot time
    liquidity_usd: float | None  # total pool value in USD
    source: str
    schema_version: str = SCHEMA_VERSION
    ingestion_version: str = INGESTION_VERSION
    quality_flags: tuple[str, ...] = ()
    slot: int | None = None
    sequence: int | None = None


@dataclass(frozen=True, slots=True)
class SocialEventRecord:
    """One social signal observation (Reddit mention count, velocity, etc.).

    ``available_time`` is the collector receipt time, not the post creation
    time. Using post creation time as ``available_time`` is look-ahead: a post
    created at 10:00 but collected at 10:03 was not in the dataset at 10:00, and
    a replay that treats it as available at 10:00 grants three minutes of
    foresight — smaller than it sounds only until you realise a 0.2% threshold
    is the whole margin.

    ``observed_through`` is the latest post timestamp that is fully indexed by
    the collector. Arctic Shift's index runs behind live Reddit, so the most
    recent window is structurally incomplete; this field makes that visible
    rather than letting the replay treat an incomplete window as a confident
    low-count observation.
    """

    asset_id: str  # mint of the associated token
    event_time: float  # when the underlying posts were created (approx)
    available_time: float  # when the collector received this data
    received_time: float
    source: str  # "praw" or "arctic_shift"
    mention_velocity_1h: float | None
    mention_velocity_24h: float | None
    mention_zscore_7d: float | None
    unique_contributors_24h: int | None
    contributor_to_post_ratio: float | None
    observed_through: float | None
    baseline_hours: int
    schema_version: str = SCHEMA_VERSION
    ingestion_version: str = INGESTION_VERSION
    quality_flags: tuple[str, ...] = ()
    sequence: int | None = None


@dataclass(frozen=True, slots=True)
class UniverseMembershipRecord:
    """Point-in-time eligibility of one coin for trading.

    Carries explicit start and end timestamps so the catalog can reconstruct the
    universe at any historical moment without assuming current membership applies
    to the past. A coin removed from the universe today must not be retroactively
    invisible in yesterday's replay — that would be a form of survivorship bias
    applied to the strategy input rather than just to the backtest result.

    ``eligible_from`` is ``pool_created_at + min_age_seconds``, not just
    ``pool_created_at``: a coin that launched yesterday is not a candidate for
    a strategy that requires several weeks of history to warm its features.

    ``eligible_until`` is ``None`` when the coin is still in the universe. A
    replay at time ``t`` includes this coin only when
    ``eligible_from <= t < (eligible_until or inf)``.
    """

    asset_id: str  # mint address
    pool_id: str
    symbol: str  # for display only; never used for identity
    eligible_from: float  # epoch seconds
    eligible_until: float | None  # None = still eligible
    pool_created_at: float
    dex: str
    quote_token: str  # "SOL", "USDC", etc.
    liquidity_usd: float | None
    fdv_usd: float | None
    source: str
    event_time: float  # when this membership decision was recorded
    available_time: float
    received_time: float
    schema_version: str = SCHEMA_VERSION
    ingestion_version: str = INGESTION_VERSION
    quality_flags: tuple[str, ...] = ()
    sequence: int | None = None


@dataclass(frozen=True, slots=True)
class LabelRecord:
    """A training label for one asset over one forward window.

    ``label_start_ts`` and ``label_end_ts`` define the full interval that this
    observation depends on. The purge step before a train/val/test split must
    remove any training example whose ``[label_start_ts, label_end_ts]`` overlaps
    the validation or test period. Without both timestamps, the purge would have
    to assume the worst-case forward horizon, which either over-purges (too much
    data lost) or under-purges (leakage).

    ``feature_ts`` is when the features were computed — the last bar close that
    the model saw. This is distinct from ``event_time``, which is when the label
    became knowable (i.e., the bar at ``label_end_ts`` closed and was published).
    Features computed before ``feature_ts`` and labels observed before
    ``event_time`` produce a valid, non-leaking training example.

    ``target_return_pct`` is the raw observed return; ``net_return_pct`` is after
    estimated costs. Only the net version should be the optimization target,
    because a model that maximises gross return will systematically overestimate
    the strategy's value.
    """

    asset_id: str
    pool_id: str
    feature_ts: float  # last feature observation time (last bar close)
    label_start_ts: float  # start of the forward window
    label_end_ts: float  # end of the forward window (purge boundary)
    event_time: float  # when the label became observable (label_end_ts + pub delay)
    available_time: float
    received_time: float
    target_return_pct: float | None  # gross return over the window
    net_return_pct: float | None  # after estimated round-trip costs
    horizon_seconds: float  # label_end_ts - label_start_ts
    label_kind: str  # "forward_return", "binary_direction", etc.
    source: str
    schema_version: str = SCHEMA_VERSION
    ingestion_version: str = INGESTION_VERSION
    quality_flags: tuple[str, ...] = ()
    sequence: int | None = None

    def __post_init__(self) -> None:
        if self.label_end_ts <= self.label_start_ts:
            raise ValueError(
                f"LabelRecord for {self.asset_id}: label_end_ts {self.label_end_ts} "
                f"<= label_start_ts {self.label_start_ts}"
            )
        if self.available_time < self.label_end_ts:
            raise ValueError(
                f"LabelRecord for {self.asset_id}: available_time {self.available_time} "
                f"< label_end_ts {self.label_end_ts} — a label cannot be known before "
                "the window it describes closes"
            )

    @property
    def overlaps(self) -> tuple[float, float]:
        """The interval that must not overlap val/test. Both endpoints inclusive."""
        return (self.label_start_ts, self.label_end_ts)


__all__ = [
    "INGESTION_VERSION",
    "SCHEMA_VERSION",
    "CandleRecord",
    "LabelRecord",
    "PoolState",
    "QuoteLadder",
    "QuoteLadderRung",
    "SocialEventRecord",
    "SwapRecord",
    "UniverseMembershipRecord",
]
