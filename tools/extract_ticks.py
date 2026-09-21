"""Pull the live evidence bundles out of the documented-run tick pages.

Each slow tick was written as a markdown page with one ``~~~json`` fenced block
per coin holding the complete :class:`EvidenceBundle` the strategy actually
read. That is the only surviving record of what the live system was served, and
it is what makes the backfill checkable rather than merely plausible.

**What is in there, and what is not.** The bundles carry the snapshot
(including DexScreener's ``price_change.h1``) and the *derived* technicals.
They do **not** carry the raw candle arrays — the plan assumed they did. This
turns out to be the stronger position rather than the weaker one:
``realized_vol_pct`` is a pure function of the last 21 closes
(``signals.py:319``), so recomputing it from backfilled bars and comparing
against the recorded value tests the downloaded bars close-for-close. A
reconstruction that agrees to floating-point noise cannot be agreeing by
accident.

Two fields make the join to the backfill possible and both are recorded per
coin per tick: ``technicals.<tf>.pool_address`` (the backfill is keyed by pool,
never by symbol) and ``snapshot.provenance.receive_time`` (the instant the
evidence was true, so the comparison uses only bars that had closed by then).

Emits one JSON object per coin per tick on stdout, oldest tick first.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# The fence is ``~~~`` rather than ``` because the bundles are nested inside a
# <details> block that already uses backticks.
_BLOCK = re.compile(r"~~~json\n(.*?)\n~~~", re.DOTALL)


def _get(obj: Any, *path: str) -> Any:
    """Walk a nested mapping, returning None at the first missing or non-map.

    None here means "the recorded bundle did not have it", which is a fact
    about the run worth keeping, not an error to raise on.
    """
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def iter_bundles(tick_path: Path) -> Iterator[dict[str, Any]]:
    """Yield every evidence bundle embedded in one tick page."""
    text = tick_path.read_text(encoding="utf-8")
    for raw in _BLOCK.findall(text):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # A prompt or a report block that happens to be JSON-fenced. Skip
            # rather than fail: this tool reads pages written for humans.
            continue
        if isinstance(parsed, dict) and "snapshot" in parsed:
            yield parsed


def flatten(bundle: dict[str, Any], *, run: str, tick: str) -> dict[str, Any]:
    """Reduce a bundle to the fields the fidelity check needs.

    Everything kept here is either a join key or a value that will be
    independently recomputed from backfilled bars and compared.
    """
    snapshot = bundle.get("snapshot") or {}
    technicals = bundle.get("technicals") or {}
    return {
        "run": run,
        "tick": tick,
        "symbol": bundle.get("symbol"),
        "mint": snapshot.get("mint"),
        "receive_time": _get(snapshot, "provenance", "receive_time"),
        "quality": _get(snapshot, "provenance", "quality"),
        "price_usd": snapshot.get("price_usd"),
        # The live ground truth for the h1 reconstruction. Rolling trailing
        # window, computed by the vendor at request time and not retrievable
        # afterwards at any price.
        "price_change_h1": _get(snapshot, "price_change", "h1"),
        "price_change_m5": _get(snapshot, "price_change", "m5"),
        "pool": _get(snapshot, "pool", "pair_address"),
        "h1_pool": _get(technicals, "h1", "pool_address"),
        "h1_candles_used": _get(technicals, "h1", "candles_used"),
        "h1_realized_vol_pct": _get(technicals, "h1", "realized_vol_pct"),
        "m5_pool": _get(technicals, "m5", "pool_address"),
        "m5_candles_used": _get(technicals, "m5", "candles_used"),
        "m5_realized_vol_pct": _get(technicals, "m5", "realized_vol_pct"),
    }


def extract(runs_root: Path) -> list[dict[str, Any]]:
    """Every bundle under ``<runs_root>/*/ticks/*.md``, oldest tick first.

    Sorted by path because the tick filenames embed a zero-padded index and a
    UTC stamp, so lexical order is chronological order.
    """
    records: list[dict[str, Any]] = []
    for tick_path in sorted(runs_root.glob("*/ticks/*.md")):
        run = tick_path.parent.parent.name
        records.extend(
            flatten(bundle, run=run, tick=tick_path.name)
            for bundle in iter_bundles(tick_path)
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", default="runs", type=Path, help="Root holding the documented runs."
    )
    parser.add_argument("--out", type=Path, help="Write JSONL here instead of stdout.")
    args = parser.parse_args()

    records = extract(args.runs)
    if not records:
        print(f"no evidence bundles under {args.runs}", file=sys.stderr)
        return 1

    lines = "".join(json.dumps(r) + "\n" for r in records)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(lines, encoding="utf-8")
        ticks = len({(r["run"], r["tick"]) for r in records})
        print(f"{len(records)} bundles from {ticks} ticks -> {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
