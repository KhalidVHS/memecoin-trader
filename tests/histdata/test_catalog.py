"""Tests for catalog.Catalog — partition discovery and content hashing.

All tests use in-memory fixtures written to a tmp_path. No network. No real
history directory required.
"""

from __future__ import annotations

import time
from pathlib import Path

from memetrader.backfill import RawBar, SeriesMeta, write_manifest, write_series
from memetrader.histdata.catalog import Catalog
from memetrader.types import Timeframe

# ---------------------------------------------------------------------------
# Helpers to build fixture data
# ---------------------------------------------------------------------------


def _write_bar_file(root: Path, pool: str, tf: str, bars: list[RawBar]) -> Path:
    """Write a gzipped JSONL bar file at the expected path."""
    path = root / pool / f"{tf}.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_series(path, bars)
    return path


def _make_raw_bars(
    count: int, base_ts: float = 1_700_000_000.0, interval: float = 3600.0
) -> list[RawBar]:
    """Generate synthetic closed bars."""
    return [
        RawBar(
            ts=base_ts + i * interval,
            open=1.0 + i * 0.01,
            high=1.02 + i * 0.01,
            low=0.99 + i * 0.01,
            close=1.0 + i * 0.01,
            volume=100.0,
        )
        for i in range(count)
    ]


def _make_meta(pool: str, tf: str, bars: list[RawBar]) -> SeriesMeta:
    return SeriesMeta(
        symbol="TEST",
        mint="MINT_TEST_123",
        pool=pool,
        timeframe=tf,
        interval_seconds=3600.0 if tf == "1h" else 300.0,
        rows=len(bars),
        first_ts=bars[0].ts if bars else None,
        last_ts=bars[-1].ts if bars else None,
        fetched_at=time.time(),
        source="geckoterminal",
        horizon_reached=True,
    )


# ---------------------------------------------------------------------------
# Test: basic discovery
# ---------------------------------------------------------------------------


def test_empty_root_produces_empty_catalog(tmp_path: Path) -> None:
    """An empty history root produces a catalog with no partitions."""
    cat = Catalog(root=tmp_path)
    assert len(cat) == 0
    assert cat.partitions() == []


def test_discovers_one_partition(tmp_path: Path) -> None:
    """A single bar file is discovered and exposed as a PartitionInfo."""
    bars = _make_raw_bars(10)
    pool = "POOL_ABCDEF"
    _write_bar_file(tmp_path, pool, "1h", bars)

    cat = Catalog(root=tmp_path)
    assert len(cat) == 1

    parts = cat.partitions()
    assert len(parts) == 1
    assert parts[0].pool == pool
    assert parts[0].timeframe == "1h"
    assert parts[0].row_count == 10
    assert parts[0].first_ts == bars[0].ts
    assert parts[0].last_ts == bars[-1].ts


def test_discovers_multiple_partitions(tmp_path: Path) -> None:
    """Multiple pools and timeframes are all discovered."""
    bars_1h = _make_raw_bars(5, interval=3600.0)
    bars_5m = _make_raw_bars(12, interval=300.0)

    _write_bar_file(tmp_path, "POOL_AAA", "1h", bars_1h)
    _write_bar_file(tmp_path, "POOL_BBB", "1h", bars_1h)
    _write_bar_file(tmp_path, "POOL_AAA", "5m", bars_5m)

    cat = Catalog(root=tmp_path)
    assert len(cat) == 3
    assert len(cat.pools()) == 2


def test_partition_filter_by_pool(tmp_path: Path) -> None:
    """partitions(pool=...) filters by pool address."""
    _write_bar_file(tmp_path, "POOL_X", "1h", _make_raw_bars(5))
    _write_bar_file(tmp_path, "POOL_Y", "1h", _make_raw_bars(5))

    cat = Catalog(root=tmp_path)
    result = cat.partitions(pool="POOL_X")
    assert len(result) == 1
    assert result[0].pool == "POOL_X"


def test_partition_filter_by_timeframe(tmp_path: Path) -> None:
    """partitions(timeframe=...) filters by timeframe."""
    _write_bar_file(tmp_path, "POOL_Z", "1h", _make_raw_bars(5))
    _write_bar_file(tmp_path, "POOL_Z", "5m", _make_raw_bars(5, interval=300.0))

    cat = Catalog(root=tmp_path)
    result = cat.partitions(timeframe=Timeframe.H1)
    assert all(p.timeframe == "1h" for p in result)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Test: content hashing
# ---------------------------------------------------------------------------


def test_file_hash_stable_across_builds(tmp_path: Path) -> None:
    """The same bars produce the same file hash on two separate writes.

    This is the reproducibility property: two runs on the same data produce
    the same data_manifest, which is what makes manifest equality meaningful.
    """
    bars = _make_raw_bars(5)
    path_a = tmp_path / "a" / "1h.jsonl.gz"
    path_b = tmp_path / "b" / "1h.jsonl.gz"

    path_a.parent.mkdir(parents=True)
    path_b.parent.mkdir(parents=True)

    write_series(path_a, bars)
    write_series(path_b, bars)

    # Both catalogs should produce the same hash
    cat_a = Catalog(root=tmp_path / "a")
    cat_b = Catalog(root=tmp_path / "b")

    # Read the single partition from each
    parts_a = cat_a.partitions()
    parts_b = cat_b.partitions()
    assert len(parts_a) == 1
    assert len(parts_b) == 1
    # Hashes match because write_series uses mtime=0 (deterministic gzip)
    assert parts_a[0].file_hash == parts_b[0].file_hash


def test_different_data_different_hash(tmp_path: Path) -> None:
    """Different bars produce different hashes."""
    bars_a = _make_raw_bars(5)
    bars_b = _make_raw_bars(6)  # one extra bar

    path_a = tmp_path / "a" / "POOL_X" / "1h.jsonl.gz"
    path_b = tmp_path / "b" / "POOL_X" / "1h.jsonl.gz"
    path_a.parent.mkdir(parents=True)
    path_b.parent.mkdir(parents=True)

    write_series(path_a, bars_a)
    write_series(path_b, bars_b)

    cat_a = Catalog(root=tmp_path / "a")
    cat_b = Catalog(root=tmp_path / "b")
    hash_a = cat_a.partitions()[0].file_hash
    hash_b = cat_b.partitions()[0].file_hash
    assert hash_a != hash_b


# ---------------------------------------------------------------------------
# Test: data_manifest
# ---------------------------------------------------------------------------


def test_data_manifest_structure(tmp_path: Path) -> None:
    """data_manifest returns a stable dict with expected keys."""
    bars = _make_raw_bars(3)
    pool = "POOL_MANIFEST_TEST"
    _write_bar_file(tmp_path, pool, "1h", bars)

    cat = Catalog(root=tmp_path)
    manifest = cat.data_manifest()

    key = f"{pool}/1h"
    assert key in manifest
    entry = manifest[key]
    assert "file_hash" in entry
    assert "row_count" in entry
    assert "first_ts" in entry
    assert "last_ts" in entry
    assert entry["row_count"] == 3


def test_data_manifest_is_sorted(tmp_path: Path) -> None:
    """data_manifest keys are sorted lexicographically."""
    _write_bar_file(tmp_path, "POOL_ZZZ", "1h", _make_raw_bars(3))
    _write_bar_file(tmp_path, "POOL_AAA", "1h", _make_raw_bars(3))

    cat = Catalog(root=tmp_path)
    keys = list(cat.data_manifest().keys())
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Test: manifest enrichment
# ---------------------------------------------------------------------------


def test_enriched_from_manifest(tmp_path: Path) -> None:
    """PartitionInfo.meta is populated when history/manifest.json exists."""
    bars = _make_raw_bars(7)
    pool = "POOL_WITH_META"
    _write_bar_file(tmp_path, pool, "1h", bars)
    meta = _make_meta(pool, "1h", bars)
    write_manifest(tmp_path, [meta])

    cat = Catalog(root=tmp_path)
    parts = cat.partitions(pool=pool)
    assert len(parts) == 1
    assert parts[0].meta is not None
    assert parts[0].meta.symbol == "TEST"
    assert parts[0].row_count == 7


# ---------------------------------------------------------------------------
# Test: read_bars
# ---------------------------------------------------------------------------


def test_read_bars_returns_all_bars(tmp_path: Path) -> None:
    """read_bars returns every bar in a partition."""
    bars = _make_raw_bars(15)
    pool = "POOL_READ_TEST"
    _write_bar_file(tmp_path, pool, "1h", bars)

    cat = Catalog(root=tmp_path)
    loaded = cat.read_bars(pool, Timeframe.H1)
    assert len(loaded) == 15
    assert loaded[0].ts == bars[0].ts


def test_read_bars_missing_partition(tmp_path: Path) -> None:
    """read_bars returns [] for a partition that does not exist."""
    cat = Catalog(root=tmp_path)
    result = cat.read_bars("NONEXISTENT_POOL", Timeframe.H1)
    assert result == []
