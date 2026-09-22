"""Tests for universe.UniverseCatalog — point-in-time universe reconstruction.

Key tests:
  - Survivorship bias: removing future metadata does not change historical eligibility
  - Membership is gated by eligible_from, not by added_at
  - Coins with removed_at are excluded after that time but included before
  - The min-age floor is applied consistently
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from memetrader.histdata.universe import (
    DEFAULT_MIN_POOL_AGE_DAYS,
    UniverseCatalog,
    UniverseEntry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A real-looking but non-network timestamp: 2024-01-01 midnight UTC
_T_JAN_2024 = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
_T_JUNE_2024 = datetime.datetime(2024, 6, 1, tzinfo=datetime.timezone.utc).timestamp()
_T_SEP_2024 = datetime.datetime(2024, 9, 1, tzinfo=datetime.timezone.utc).timestamp()
_T_SEP_2026 = datetime.datetime(2026, 9, 1, tzinfo=datetime.timezone.utc).timestamp()


def _entry(
    *,
    asset_id: str = "MINT_A",
    pool_id: str = "POOL_A",
    symbol: str = "TESTA",
    pool_created_at: float = _T_JAN_2024,
    min_age_days: float = 0.0,
    added_at: float = _T_JUNE_2024,
    removed_at: float | None = None,
) -> UniverseEntry:
    eligible_from = pool_created_at + min_age_days * 86400.0
    return UniverseEntry(
        asset_id=asset_id,
        pool_id=pool_id,
        symbol=symbol,
        dex="raydium",
        quote_token="SOL",
        pool_created_at=pool_created_at,
        eligible_from=eligible_from,
        added_at=added_at,
        removed_at=removed_at,
        liquidity_usd=1_000_000.0,
        fdv_usd=5_000_000.0,
        liquidity_fdv_ratio=0.2,
    )


def _catalog(*entries: UniverseEntry, min_age: float = 0.0) -> UniverseCatalog:
    return UniverseCatalog(entries=entries, min_pool_age_days=min_age)


# ---------------------------------------------------------------------------
# Test: basic eligibility
# ---------------------------------------------------------------------------


def test_eligible_from_pool_created_plus_age() -> None:
    """eligible_from = pool_created_at + min_age_seconds."""
    created = _T_JAN_2024
    min_days = 14.0
    e = _entry(pool_created_at=created, min_age_days=min_days)
    assert e.eligible_from == created + min_days * 86400.0


def test_not_eligible_before_eligible_from() -> None:
    """A coin is not eligible before its eligible_from timestamp."""
    created = _T_JAN_2024
    e = _entry(pool_created_at=created, min_age_days=14.0)
    # Query 1 second before eligible_from
    assert not e.is_eligible_at(e.eligible_from - 1.0)


def test_eligible_at_eligible_from() -> None:
    """A coin is eligible exactly at eligible_from."""
    e = _entry(pool_created_at=_T_JAN_2024, min_age_days=0.0)
    assert e.is_eligible_at(e.eligible_from)


def test_eligible_long_after() -> None:
    """A coin with no removed_at is eligible indefinitely."""
    e = _entry(pool_created_at=_T_JAN_2024, removed_at=None)
    assert e.is_eligible_at(_T_SEP_2026)


# ---------------------------------------------------------------------------
# Test: removed_at gating
# ---------------------------------------------------------------------------


def test_not_eligible_after_removed_at() -> None:
    """A coin is not eligible at or after its removed_at."""
    e = _entry(pool_created_at=_T_JAN_2024, removed_at=_T_SEP_2024)
    assert not e.is_eligible_at(_T_SEP_2024)


def test_eligible_before_removed_at() -> None:
    """A coin is still eligible before its removed_at."""
    e = _entry(pool_created_at=_T_JAN_2024, removed_at=_T_SEP_2024)
    assert e.is_eligible_at(_T_SEP_2024 - 1.0)


# ---------------------------------------------------------------------------
# Survivorship bias test: the critical one
# ---------------------------------------------------------------------------


def test_survivorship_future_added_at_unchanged() -> None:
    """Removing future metadata (adding_at in the future) must leave
    historical eligibility unchanged.

    This is the 'Universe survivorship' test from BACKTEST-CONTRACTS.md §8.

    A coin that was eligible at time T must remain eligible at time T even if
    we change when the coin was *recorded in our file* (added_at). The
    eligible_from field is the economic fact; added_at is bookkeeping.

    If this test fails, the universe is applying the research collection date
    as an eligibility gate — which is survivorship bias: only coins we had
    already discovered would appear in historical replays.
    """
    created = _T_JAN_2024
    eligible_from = created  # no age floor
    original_added_at = _T_JUNE_2024  # well before the test query time

    # Original universe: coin added in June 2024
    original = _catalog(_entry(
        asset_id="MINT_A",
        pool_created_at=created,
        added_at=original_added_at,
    ))

    # Modified universe: coin "discovered" in Sep 2026 (far future)
    modified = original.with_entry_added_at_future("MINT_A", new_added_at=_T_SEP_2026)

    # Query at _T_JUNE_2024 (before the "future" added_at)
    query_time = _T_JUNE_2024

    original_eligible = original.eligible_at(query_time)
    modified_eligible = modified.eligible_at(query_time)

    assert "MINT_A" in original_eligible, "coin must be eligible in original"
    assert original_eligible == modified_eligible, (
        "changing added_at must not change historical eligibility: "
        "survivorship bias would result if added_at were used as an eligibility gate"
    )


def test_survivorship_removed_at_does_not_change_prior_eligibility() -> None:
    """Setting removed_at in the future must not change eligibility before removed_at.

    Variant: if we mark a coin as removed at time T2, its eligibility before T2
    must be unchanged. This verifies that the removal event is handled correctly
    and does not accidentally back-propagate.
    """
    e = _entry(pool_created_at=_T_JAN_2024, removed_at=None)
    original = _catalog(e)

    # "Delist" the coin at Sep 2026
    modified = original.with_entry_removed_at("MINT_A", removed_at=_T_SEP_2026)

    # Before the removal time: eligibility is the same
    query_before = _T_JUNE_2024
    assert original.eligible_at(query_before) == modified.eligible_at(query_before)

    # After the removal time: coin is gone in modified, still there in original
    query_after = _T_SEP_2026 + 1.0
    assert "MINT_A" in original.eligible_at(query_after)
    assert "MINT_A" not in modified.eligible_at(query_after)


# ---------------------------------------------------------------------------
# Test: multi-coin catalog
# ---------------------------------------------------------------------------


def test_eligible_at_multi_coin() -> None:
    """eligible_at returns exactly the coins eligible at the given time."""
    # Coin A: eligible from Jan 2024
    # Coin B: eligible from June 2024
    # Coin C: eligible from Jan 2024 but removed in June 2024
    a = _entry(asset_id="MINT_A", pool_id="POOL_A", pool_created_at=_T_JAN_2024)
    b = _entry(asset_id="MINT_B", pool_id="POOL_B", pool_created_at=_T_JUNE_2024)
    c = _entry(asset_id="MINT_C", pool_id="POOL_C", pool_created_at=_T_JAN_2024,
               removed_at=_T_JUNE_2024)

    cat = _catalog(a, b, c)

    # Before June 2024: A and C eligible, B not yet
    before_june = _T_JUNE_2024 - 1.0
    eligible = cat.eligible_at(before_june)
    assert "MINT_A" in eligible
    assert "MINT_B" not in eligible
    assert "MINT_C" in eligible

    # After June 2024: A and B eligible, C removed
    after_june = _T_JUNE_2024 + 1.0
    eligible = cat.eligible_at(after_june)
    assert "MINT_A" in eligible
    assert "MINT_B" in eligible
    assert "MINT_C" not in eligible


# ---------------------------------------------------------------------------
# Test: from_toml reads the real universe file
# ---------------------------------------------------------------------------


def test_from_toml_reads_real_file() -> None:
    """from_toml reads the committed universe file and produces 24 entries."""
    universe_path = Path(__file__).parents[2] / "universe" / "solana_memecoins.toml"
    if not universe_path.exists():
        pytest.skip("universe file not found")

    cat = UniverseCatalog.from_toml(universe_path, min_pool_age_days=0.0)
    assert len(cat) == 24, f"expected 24 coins, got {len(cat)}"

    symbols = {e.symbol for e in cat.entries}
    assert "BONK" in symbols
    assert "SLERF" in symbols
    assert "WIF" in symbols


def test_from_toml_eligible_at_sep_2026() -> None:
    """All 24 coins should be eligible at Sep 2026 (no removals in current file)."""
    universe_path = Path(__file__).parents[2] / "universe" / "solana_memecoins.toml"
    if not universe_path.exists():
        pytest.skip("universe file not found")

    cat = UniverseCatalog.from_toml(universe_path, min_pool_age_days=0.0)
    eligible = cat.eligible_at(_T_SEP_2026)
    assert len(eligible) == 24, f"expected all 24 coins eligible, got {len(eligible)}"


def test_from_toml_min_age_reduces_early_eligibility() -> None:
    """With min_pool_age_days=14, coins are not eligible in the first 14 days."""
    universe_path = Path(__file__).parents[2] / "universe" / "solana_memecoins.toml"
    if not universe_path.exists():
        pytest.skip("universe file not found")

    cat_no_age = UniverseCatalog.from_toml(universe_path, min_pool_age_days=0.0)
    cat_with_age = UniverseCatalog.from_toml(universe_path, min_pool_age_days=14.0)

    # Find a coin and verify eligible_from shifted
    bonk_no_age = cat_no_age.entry_for_pool(
        "5zpyutJu9ee6jFymDGoK7F6S5Kczqtc9FomP3ueKuyA9"
    )
    bonk_with_age = cat_with_age.entry_for_pool(
        "5zpyutJu9ee6jFymDGoK7F6S5Kczqtc9FomP3ueKuyA9"
    )
    assert bonk_no_age is not None
    assert bonk_with_age is not None
    assert bonk_with_age.eligible_from == bonk_no_age.pool_created_at + 14.0 * 86400.0


# ---------------------------------------------------------------------------
# Test: helpers
# ---------------------------------------------------------------------------


def test_pool_to_asset_mapping() -> None:
    """pool_to_asset returns a complete mint->pool mapping."""
    a = _entry(asset_id="MINT_A", pool_id="POOL_A")
    b = _entry(asset_id="MINT_B", pool_id="POOL_B")
    cat = _catalog(a, b)
    mapping = cat.pool_to_asset()
    assert mapping == {"POOL_A": "MINT_A", "POOL_B": "MINT_B"}


def test_coverage_start_is_earliest_eligible_from() -> None:
    """coverage_start returns the earliest eligible_from."""
    a = _entry(asset_id="MINT_A", pool_created_at=_T_JAN_2024)
    b = _entry(asset_id="MINT_B", pool_created_at=_T_JUNE_2024)
    cat = _catalog(a, b)
    assert cat.coverage_start() == a.eligible_from


def test_coverage_end_is_inf_when_no_removals() -> None:
    """coverage_end returns inf when no coins have been removed."""
    a = _entry(asset_id="MINT_A", removed_at=None)
    b = _entry(asset_id="MINT_B", removed_at=None)
    cat = _catalog(a, b)
    assert cat.coverage_end() == float("inf")
