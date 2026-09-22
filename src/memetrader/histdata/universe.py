"""Point-in-time tradable universe reconstruction.

A universe is the set of coins eligible for trading at a given moment. Getting
this wrong is survivorship bias: if the replay uses today's universe to select
coins in a historical backtest, it will only see coins that survived — the
failures that the strategy must learn to avoid are simply absent.

This module addresses three specific forms of survivorship bias:

1. **Membership creep**: a coin added to the universe yesterday is not eligible
   last month. ``eligible_from = pool_created_at + min_age_seconds`` is the
   hard floor; a coin does not appear in any replay before that time.

2. **Dead token omission**: coins that have since been delisted or abandoned
   must remain in the historical universe. They are the failures. A backtest
   without them trains on a cherry-picked dataset.

3. **Future metadata leak**: the resolution data (liquidity, FDV, pool address)
   was collected on a specific date. Applying it to a backtest that predates
   the collection is a form of look-ahead. Liquidity from today does not
   describe yesterday's pool depth, and a strategy screened on today's liquidity
   would have had access to different (usually worse) pools historically.

The ``UniverseEntry`` dataclass carries an ``added_at`` field that records when
this entry was added to the universe file. The ``eligible_from`` field encodes
the minimum age the pool must have reached before the coin is tradable. The
``removed_at`` field records when the coin was delisted or dropped from the
universe (``None`` for still-active coins).

A replay at time ``t`` includes a coin only when:
    ``eligible_from <= t < (removed_at if removed_at is not None else +inf)``

The ``added_at`` field does NOT gate eligibility: a coin that was discoverable
before it was added to the universe file should be eligible from
``eligible_from``, not from ``added_at``. The universe file is a research
artifact; ``eligible_from`` is the economic fact.

Reading ``universe/solana_memecoins.toml``:
The real file has ``pool_created`` as a date string (e.g., "2024-03-18").
We parse it to midnight UTC (epoch seconds) to use as a timestamp. The minimum
age floor (``min_pool_age_days``) is applied on top to compute ``eligible_from``.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path


# Default minimum pool age before a coin enters the universe. Chosen so that
# a coin has at least some price history before we expect feature pipelines to
# compute meaningful signals. 14 days is enough for a 1h RSI(14) to warm up.
# Set to 0 in tests that exercise the eligibility logic directly.
DEFAULT_MIN_POOL_AGE_DAYS: float = 14.0


@dataclass(frozen=True, slots=True)
class UniverseEntry:
    """One coin's membership record, with point-in-time eligibility.

    ``symbol`` is for display. ``asset_id`` (mint) is the identity.
    ``pool_id`` is the primary pool used for pricing. Both are required
    because a symbol can point to impostors and a pool can be migrated.

    ``eligible_from`` is computed from ``pool_created_at + min_age_seconds``,
    not from any collection date. This is the economic fact: a coin that
    launched six months ago was eligible six months ago (plus the age floor),
    regardless of when we added it to this file.

    ``removed_at`` is ``None`` when the coin is still in the universe. A
    non-None value means the coin was delisted, the pool dried up, or the
    research team decided to drop it. It is still part of the historical
    universe up to that timestamp.

    ``added_at`` is the timestamp when this entry was committed to the universe
    file. It is informational — used for auditing, not for eligibility.
    """

    asset_id: str  # mint address — the stable identity
    pool_id: str  # primary pool address
    symbol: str  # display only
    dex: str
    quote_token: str  # "SOL", "USDC", etc.
    pool_created_at: float  # epoch seconds (midnight UTC of pool_created date)
    eligible_from: float  # pool_created_at + min_age_seconds
    added_at: float  # when this entry was committed to the universe file
    removed_at: float | None  # None = still eligible
    liquidity_usd: float | None  # at resolution time — not a historical series
    fdv_usd: float | None  # at resolution time
    liquidity_fdv_ratio: float | None

    def is_eligible_at(self, ts: float) -> bool:
        """Whether this coin is tradable at simulated time ``ts``.

        The condition is: ``eligible_from <= ts < removed_at`` (or no
        ``removed_at``). Strict less-than on removed_at so that a coin removed
        at exactly ``ts`` is considered already removed — the removal event has
        been processed before the decision tick.

        NOT gated on ``added_at``: a coin that was eligible before we discovered
        it is still historically eligible. Back-applying the discovery date as
        an eligibility gate would be survivorship bias in reverse.
        """
        if ts < self.eligible_from:
            return False
        if self.removed_at is not None and ts >= self.removed_at:
            return False
        return True


@dataclass
class UniverseCatalog:
    """All coins ever in the universe, with point-in-time eligibility.

    Built from ``universe/solana_memecoins.toml`` (or equivalent). Contains
    both active and retired coins, so the historical replay sees the failures
    that a live leaderboard would exclude.

    ``entries`` is the full membership list (immutable after construction).
    Use ``eligible_at(ts)`` to get the point-in-time tradable set.

    ``min_pool_age_days`` is the configurable minimum age applied uniformly.
    It defaults to ``DEFAULT_MIN_POOL_AGE_DAYS``. A run that changes this
    parameter must re-register the universe in its manifest — changing the age
    floor changes which coins appear in which folds, and two runs with different
    floors are not comparable.
    """

    entries: tuple[UniverseEntry, ...]
    min_pool_age_days: float = DEFAULT_MIN_POOL_AGE_DAYS
    source_path: Path | None = None  # for audit trail

    @classmethod
    def from_toml(
        cls,
        path: Path,
        *,
        min_pool_age_days: float = DEFAULT_MIN_POOL_AGE_DAYS,
        now: float | None = None,
    ) -> UniverseCatalog:
        """Read the universe from a TOML file with the solana_memecoins.toml shape.

        ``now`` is the timestamp to use as ``added_at`` for all entries. If not
        provided, defaults to the epoch timestamp of the ``[meta] resolved_at``
        field in the TOML, which records when the file was constructed.

        The TOML shape is described in the comments at the top of
        ``universe/solana_memecoins.toml``. The key fields:
            [[coins]]
            symbol, mint, pool, dex, quote, pool_created, liquidity_usd,
            fdv_usd, liquidity_fdv_ratio, total_pools_seen

        We do not filter on liquidity or FDV here — that was the resolver's job
        when building the file. The universe file is the committed research
        artifact; we read it as-is.
        """
        import tomllib

        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        meta = raw.get("meta") or {}

        resolved_at_str = meta.get("resolved_at")
        if resolved_at_str and now is None:
            # Parse the ISO 8601 string with Z suffix
            resolved_at_str = resolved_at_str.replace("Z", "+00:00")
            added_at = datetime.datetime.fromisoformat(resolved_at_str).timestamp()
        else:
            added_at = now or 0.0

        entries: list[UniverseEntry] = []
        min_age_seconds = min_pool_age_days * 86400.0

        for coin in raw.get("coins") or []:
            pool_created_str = str(coin.get("pool_created") or "")
            if pool_created_str:
                # Parse "YYYY-MM-DD" as midnight UTC
                try:
                    pool_created_dt = datetime.datetime.strptime(
                        pool_created_str, "%Y-%m-%d"
                    ).replace(tzinfo=datetime.timezone.utc)
                    pool_created_at = pool_created_dt.timestamp()
                except ValueError:
                    pool_created_at = 0.0
            else:
                pool_created_at = 0.0

            eligible_from = pool_created_at + min_age_seconds

            mint = str(coin.get("mint") or "")
            pool = str(coin.get("pool") or "")
            symbol = str(coin.get("symbol") or "")

            # liquidity_fdv_ratio: some entries are > 1.0 (SLERF, MICHI) due to
            # burned LP — that is a genuine observation, not a data error.
            liq = coin.get("liquidity_usd")
            fdv = coin.get("fdv_usd")
            ratio = coin.get("liquidity_fdv_ratio")

            entries.append(
                UniverseEntry(
                    asset_id=mint,
                    pool_id=pool,
                    symbol=symbol,
                    dex=str(coin.get("dex") or ""),
                    quote_token=str(coin.get("quote") or "SOL"),
                    pool_created_at=pool_created_at,
                    eligible_from=eligible_from,
                    added_at=added_at,
                    removed_at=None,  # no removals in the current file
                    liquidity_usd=float(liq) if liq is not None else None,
                    fdv_usd=float(fdv) if fdv is not None else None,
                    liquidity_fdv_ratio=float(ratio) if ratio is not None else None,
                )
            )

        return cls(
            entries=tuple(entries),
            min_pool_age_days=min_pool_age_days,
            source_path=path,
        )

    # ------------------------------------------------------------------
    # Point-in-time queries
    # ------------------------------------------------------------------

    def eligible_at(self, ts: float) -> frozenset[str]:
        """Mint addresses of coins eligible for trading at time ``ts``.

        This is the correct answer to "what could the strategy have traded at
        time t?" It includes dead coins (which were alive then) and excludes
        coins not yet born. It does NOT depend on when we collected this data.

        If ``ts`` is after all coins were removed, the result is empty — but
        we do not remove coins in the current file, so in practice this
        returns a non-empty set for any ``ts`` after the youngest coin's
        ``eligible_from``.
        """
        return frozenset(e.asset_id for e in self.entries if e.is_eligible_at(ts))

    def entry_for(self, asset_id: str) -> UniverseEntry | None:
        """The entry for ``asset_id`` (mint), or None if not in this universe."""
        for e in self.entries:
            if e.asset_id == asset_id:
                return e
        return None

    def entry_for_pool(self, pool_id: str) -> UniverseEntry | None:
        """The entry whose ``pool_id`` matches, or None."""
        for e in self.entries:
            if e.pool_id == pool_id:
                return e
        return None

    def pool_to_asset(self) -> dict[str, str]:
        """Mapping from pool address to mint address for all entries.

        Used by ``ReplayState.load_from_catalog`` to associate bar files
        (keyed by pool) with asset_ids (mints) used throughout the strategy.
        """
        return {e.pool_id: e.asset_id for e in self.entries}

    def asset_to_pool(self) -> dict[str, str]:
        """Mapping from mint address to primary pool address."""
        return {e.asset_id: e.pool_id for e in self.entries}

    def symbols(self) -> dict[str, str]:
        """Mapping from mint address to symbol (for display)."""
        return {e.asset_id: e.symbol for e in self.entries}

    def coverage_start(self) -> float:
        """Earliest ``eligible_from`` across all entries.

        A replay before this time has an empty universe, which is the correct
        and safe answer — not "use all coins regardless of age".
        """
        if not self.entries:
            return float("inf")
        return min(e.eligible_from for e in self.entries)

    def coverage_end(self) -> float:
        """Latest ``removed_at`` (or ``inf`` if any coin is still active).

        A replay after this time has an empty universe (all coins delisted).
        In the current dataset, no coins have been removed, so this returns
        ``inf``.
        """
        if not self.entries:
            return float("-inf")
        ends = [e.removed_at for e in self.entries if e.removed_at is not None]
        if len(ends) < len(self.entries):
            return float("inf")  # at least one coin is still active
        return max(ends)

    def with_entry_removed_at(
        self, asset_id: str, removed_at: float
    ) -> UniverseCatalog:
        """Return a copy with one entry's ``removed_at`` set.

        Used in survivorship tests: we can build a universe that looks like
        a coin was delisted at a specific time, then verify that historical
        eligibility before that time is unchanged. This is the immutable-data
        test pattern the spec requires.

        The original ``UniverseCatalog`` is not modified.
        """
        from dataclasses import replace as dc_replace

        new_entries: list[UniverseEntry] = []
        for e in self.entries:
            if e.asset_id == asset_id:
                new_entries.append(dc_replace(e, removed_at=removed_at))
            else:
                new_entries.append(e)
        return UniverseCatalog(
            entries=tuple(new_entries),
            min_pool_age_days=self.min_pool_age_days,
            source_path=self.source_path,
        )

    def with_entry_added_at_future(
        self, asset_id: str, new_added_at: float
    ) -> UniverseCatalog:
        """Return a copy where one entry's ``added_at`` is moved to the future.

        Used to test the survivorship-bias contract: changing ``added_at``
        (when the entry was recorded in the file) must NOT change historical
        eligibility. ``eligible_from`` is the economic fact; ``added_at``
        is the bookkeeping timestamp.
        """
        from dataclasses import replace as dc_replace

        new_entries: list[UniverseEntry] = []
        for e in self.entries:
            if e.asset_id == asset_id:
                new_entries.append(dc_replace(e, added_at=new_added_at))
            else:
                new_entries.append(e)
        return UniverseCatalog(
            entries=tuple(new_entries),
            min_pool_age_days=self.min_pool_age_days,
            source_path=self.source_path,
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return (
            f"UniverseCatalog(entries={len(self.entries)}, "
            f"min_pool_age_days={self.min_pool_age_days})"
        )


__all__ = [
    "DEFAULT_MIN_POOL_AGE_DAYS",
    "UniverseCatalog",
    "UniverseEntry",
]
