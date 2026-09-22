"""Immutable dataset discovery over ``history/``.

The catalog answers two questions:
1. What data exists? (partition selection)
2. Can I reproduce a run? (content hashing)

It is deliberately read-only and side-effect-free: the catalog represents a
point-in-time snapshot of the on-disk dataset, not a live view. A run that
starts with one catalog object will not see data written by another process
mid-run, which is the property needed for reproducible manifests.

**No DuckDB.** ``pyarrow.dataset`` gives partition discovery and predicate
pushdown for any columnar data we might write in the future. The existing
bar files are gzipped JSONL keyed by pool/timeframe — the catalog reads them
with ``backfill.read_series`` and hashes their content. Pyarrow is used only
where it genuinely helps: scanning Parquet artifacts written at run end, not
for the bar files that already exist in JSONL form.

**Stable ``data_manifest``**: the dict produced by :meth:`Catalog.data_manifest`
maps ``"<pool>/<tf>"`` to a ``PartitionInfo`` containing a SHA-256 of the raw
file bytes, the row count, and the first/last timestamp. That struct is stable
across runs as long as the data does not change, which is what makes two runs
comparable: if their manifests match, they consumed the same bytes.

The hash covers the *file bytes*, not parsed bar values, because a parser bug
that changes how bytes are interpreted would not change the hash — but that is
the right trade-off: two runs that used the same file definitely used the same
source, while two runs that re-parsed the same bytes might have interpreted a
float differently.
"""

from __future__ import annotations

import gzip
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from memetrader.backfill import SeriesMeta, read_manifest, read_series, series_path
from memetrader.types import Timeframe

if TYPE_CHECKING:
    pass


@dataclass(frozen=True, slots=True)
class PartitionInfo:
    """Provenance for one (pool, timeframe) partition.

    ``file_hash`` is SHA-256 of the raw gzipped bytes. Hashing the compressed
    bytes rather than the decompressed content is deliberate: gzip with
    ``mtime=0`` (as ``backfill.write_series`` sets it) is deterministic, so
    two files with the same bars produce the same hash. Hashing decompressed
    content would require parsing, which is slower and could produce the same
    output from semantically different inputs (e.g., reordered JSON keys).

    ``row_count`` is the number of bars, not bytes, because "the partition has
    X bars from Y to Z" is what the manifest needs to report — a byte count
    tells you the file size, which is much less useful for auditing coverage.
    """

    pool: str
    timeframe: str
    file_hash: str  # SHA-256 hex of raw gzipped file bytes
    row_count: int
    first_ts: float | None  # epoch seconds, bar open time
    last_ts: float | None
    meta: SeriesMeta | None = None  # from manifest.json if present


@dataclass
class Catalog:
    """Immutable view of one history root directory.

    Constructed once at run start. After construction the on-disk state is
    irrelevant — all answers come from ``self._partitions``. This is what
    makes the ``data_manifest`` stable: it is computed at construction time
    and does not change if files are added later.

    ``_partitions`` is keyed by ``"<pool>/<tf>"``, which matches the on-disk
    layout (``history/<pool>/<tf>.jsonl.gz``) and the manifest key used in
    run artifacts.

    Partitions are discovered by scanning for ``*.jsonl.gz`` files under
    ``root``. Subdirectories named ``manifest.json`` are not partitions.
    The manifest file is read if present to enrich each partition with
    provenance (gaps, missing bar counts, horizon flag).
    """

    root: Path
    _partitions: dict[str, PartitionInfo] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        metas: dict[tuple[str, str], SeriesMeta] = {}
        for m in read_manifest(self.root):
            metas[(m.pool, m.timeframe)] = m

        for gz_path in sorted(self.root.rglob("*.jsonl.gz")):
            # Path layout: <root>/<pool>/<tf>.jsonl.gz
            # We need the pool (parent dir name) and tf (stem without .jsonl)
            pool = gz_path.parent.name
            tf_name = gz_path.name.removesuffix(".jsonl.gz")
            key = f"{pool}/{tf_name}"

            file_hash = _sha256_file(gz_path)
            meta = metas.get((pool, tf_name))

            if meta is not None:
                first_ts = meta.first_ts
                last_ts = meta.last_ts
                row_count = meta.rows
            else:
                # Fall back to reading bars — slower but always correct.
                bars = list(read_series(gz_path))
                first_ts = bars[0].ts if bars else None
                last_ts = bars[-1].ts if bars else None
                row_count = len(bars)

            self._partitions[key] = PartitionInfo(
                pool=pool,
                timeframe=tf_name,
                file_hash=file_hash,
                row_count=row_count,
                first_ts=first_ts,
                last_ts=last_ts,
                meta=meta,
            )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def partitions(
        self,
        *,
        pool: str | None = None,
        timeframe: Timeframe | str | None = None,
    ) -> list[PartitionInfo]:
        """All partitions, optionally filtered.

        Filtering by pool or timeframe is by exact string match. We do not
        do prefix or glob matching because the pool address is a 44-character
        base58 string — it does not have meaningful sub-parts to match on —
        and allowing fuzzy matches would make the catalog's results depend on
        which files happen to be present, which breaks reproducibility.
        """
        tf_str = timeframe.value if isinstance(timeframe, Timeframe) else timeframe
        result = []
        for info in self._partitions.values():
            if pool is not None and info.pool != pool:
                continue
            if tf_str is not None and info.timeframe != tf_str:
                continue
            result.append(info)
        return result

    def partition(self, pool: str, timeframe: Timeframe | str) -> PartitionInfo | None:
        """The single partition for (pool, timeframe), or None if absent."""
        tf_str = timeframe.value if isinstance(timeframe, Timeframe) else timeframe
        return self._partitions.get(f"{pool}/{tf_str}")

    def pools(self) -> frozenset[str]:
        """All pool addresses that have at least one partition."""
        return frozenset(info.pool for info in self._partitions.values())

    def schema_version_hash(self) -> str:
        """A hash of all schema versions seen across partitions.

        Used in the run manifest to record whether any partition was ingested
        with a non-current schema version. Two catalogs whose
        ``schema_version_hash`` differs consumed records with different shapes,
        and their features cannot be compared directly.
        """
        from .schemas import INGESTION_VERSION, SCHEMA_VERSION

        parts = sorted(f"{k}:{v.file_hash}" for k, v in self._partitions.items())
        combined = "|".join([SCHEMA_VERSION, INGESTION_VERSION, *parts])
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Run-manifest support
    # ------------------------------------------------------------------

    def data_manifest(self) -> dict[str, dict[str, object]]:
        """Stable mapping from partition key to provenance.

        The result is suitable for embedding directly in ``manifest.json`` as
        the ``data_manifest`` section. Each value is a plain dict so the
        manifest can be serialised with ``json.dumps`` without a custom encoder.

        The sort is deterministic (lexicographic on the key) so two runs on
        the same data produce the same manifest bytes, which is what makes
        manifest equality a reliable signal for "same data".
        """
        out: dict[str, dict[str, object]] = {}
        for key in sorted(self._partitions):
            info = self._partitions[key]
            out[key] = {
                "pool": info.pool,
                "timeframe": info.timeframe,
                "file_hash": info.file_hash,
                "row_count": info.row_count,
                "first_ts": info.first_ts,
                "last_ts": info.last_ts,
            }
        return out

    # ------------------------------------------------------------------
    # Bar access (wraps backfill.read_series)
    # ------------------------------------------------------------------

    def read_bars(self, pool: str, timeframe: Timeframe | str) -> list:
        """Return all bars for the partition, or [] if not present.

        Returns ``list[RawBar]`` (from ``backfill``). The return type is
        not annotated with the concrete type to avoid importing the heavy
        backfill module at class-definition time — callers that need the
        type can import it directly.

        This is a convenience wrapper, not the primary interface: the
        ``point_in_time`` module builds its sorted store from these bars once
        at construction and then never touches the disk again.
        """
        tf_str = timeframe.value if isinstance(timeframe, Timeframe) else timeframe
        path = series_path(self.root, pool, tf_str)
        if not path.exists():
            return []
        return list(read_series(path))

    def __len__(self) -> int:
        return len(self._partitions)

    def __repr__(self) -> str:
        return (
            f"Catalog(root={self.root!r}, partitions={len(self._partitions)}, "
            f"pools={len(self.pools())})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """SHA-256 hex digest of ``path``'s raw bytes.

    Reads in 1 MiB chunks to avoid holding an entire 50 MB gzipped history
    file in memory. The chunk size is large enough that the loop overhead is
    negligible but small enough that a 512 MiB file does not cause an OOM.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _gz_line_count(path: Path) -> int:
    """Count lines in a gzipped file without loading it all at once."""
    count = 0
    with gzip.open(path, "rb") as fh:
        for _ in fh:
            count += 1
    return count


# Parquet scanning helper — only called when pyarrow.dataset is available and
# the caller has a Parquet directory to scan. Kept as a free function so it is
# easy to mock in tests without mocking the whole Catalog.
def scan_parquet_partition(path: Path) -> dict[str, object]:
    """Return {row_count, first_ts, last_ts} for a Parquet partition directory.

    Uses ``pyarrow.dataset`` for the scan so predicate pushdown keeps memory
    usage bounded. Falls back gracefully when the directory is empty or not a
    valid Parquet dataset.

    This is called by the catalog when it encounters a ``.parquet`` directory
    alongside the bar files — e.g., for swap-level TIER_1 data that was written
    as Parquet rather than JSONL. It is not called for the bar files, which are
    always JSONL.
    """
    try:
        import pyarrow.dataset as ds  # local import — only needed here

        dataset = ds.dataset(path, format="parquet")
        # We need min/max ts and total rows. Scanning just those columns is
        # much cheaper than loading all columns for a wide swap schema.
        tbl = dataset.to_table(columns=["event_time"])
        if tbl.num_rows == 0:
            return {"row_count": 0, "first_ts": None, "last_ts": None}
        col = tbl.column("event_time")
        return {
            "row_count": tbl.num_rows,
            "first_ts": float(col.min().as_py()),
            "last_ts": float(col.max().as_py()),
        }
    except Exception:  # noqa: BLE001
        return {"row_count": 0, "first_ts": None, "last_ts": None}


__all__ = [
    "Catalog",
    "PartitionInfo",
    "scan_parquet_partition",
]
