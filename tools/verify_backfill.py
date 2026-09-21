"""Check backfilled bars against what the live system was actually served.

The backfill is only evidence if the bars it downloaded are the same bars the
live run saw. Nothing about that is guaranteed: the vendor could be serving a
different pool's history, a different alignment, or a silently revised series.

Three checks, in descending order of how much they prove.

**A. realized_vol_pct, recomputed.** The decisive one.
``signals._realized_vol_pct`` is a pure function of the last 21 closes
(``signals.py:319-336``) and ``BaselineStrategy`` reads its output directly, so
recomputing it from backfilled bars at the recorded ``receive_time`` and
comparing against the value the live strategy used tests the downloaded series
close-for-close. Agreement to floating-point noise cannot happen by chance
across 21 closes; any real disagreement localises to a bar.

**B. price_change.h1, reconstructed.** The honest one. DexScreener's ``h1`` is a
*rolling* trailing window computed vendor-side at request time and is not
retrievable afterwards, so a backtest must reconstruct it from bars. This
measures the error that substitution introduces. It is expected to be nonzero —
the point is to size it against ``entry_hurdle_pct``, not to drive it to zero.

**C. Cross-timeframe.** Resampling 5m to hourly and comparing against the
fetched 1h series. Disagreement means one of the two pulls is wrong and says
which bar to look at.

Exit status is 1 if check A fails its tolerance, because that one is a
correctness claim about the data. B is a measurement, not a pass/fail.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memetrader import backfill as bf

# Mirrors signals.REALIZED_VOL_PERIOD: 20 log returns, so 21 closes.
VOL_PERIOD = 20
# Twelve 5-minute bars span the hour DexScreener's h1 looks back over.
H1_LOOKBACK_5M = 12
# Check A tolerance, in percentage points of realized vol. Generous relative to
# float noise (~1e-12) but far tighter than any real bar mismatch would give.
VOL_TOLERANCE_PP = 1e-6


def realized_vol_pct(closes: Sequence[float]) -> float | None:
    """Reimplementation of signals._realized_vol_pct, kept deliberately separate.

    Importing the original would make this check circular: it would prove the
    two call sites agree, not that the bars do. Writing it out means a change
    to either implementation shows up as a disagreement.
    """
    if len(closes) < VOL_PERIOD + 1:
        return None
    window = list(closes[-(VOL_PERIOD + 1) :])
    if any(c <= 0.0 for c in window):
        return None
    returns = [math.log(b / a) for a, b in itertools.pairwise(window)]
    return statistics.stdev(returns) * 100.0


def closed_before(bars: list[bf.RawBar], cutoff: float, interval: float) -> list[bf.RawBar]:
    """Bars whose interval had fully elapsed by ``cutoff``.

    The live path never lets a forming bar reach a feature
    (``market.py:764-775``); replay must not either, or it hands the backtest a
    close the live system could not have known.
    """
    return [b for b in bars if b.ts + interval <= cutoff]


def load_series(root: Path, pool: str, timeframe: str) -> list[bf.RawBar]:
    path = bf.series_path(root, pool, timeframe)
    if not path.exists():
        return []
    return list(bf.read_series(path))


def check_vol(
    records: list[dict[str, Any]], root: Path, timeframe: str, field: str
) -> dict[str, Any]:
    """Check A for one timeframe."""
    interval = 3600.0 if timeframe == "1h" else 300.0
    cache: dict[str, list[bf.RawBar]] = {}
    errors: list[float] = []
    worst: tuple[float, str, str] | None = None
    skipped_no_series = 0
    skipped_short = 0

    for rec in records:
        recorded = rec.get(field)
        pool = rec.get(f"{timeframe}_pool") or rec.get("pool")
        when = rec.get("receive_time")
        if recorded is None or not pool or when is None:
            continue
        if pool not in cache:
            cache[pool] = load_series(root, pool, timeframe)
        bars = cache[pool]
        if not bars:
            skipped_no_series += 1
            continue
        usable = closed_before(bars, float(when), interval)
        got = realized_vol_pct([b.close for b in usable])
        if got is None:
            skipped_short += 1
            continue
        err = abs(got - float(recorded))
        errors.append(err)
        if worst is None or err > worst[0]:
            worst = (err, str(rec.get("symbol")), str(rec.get("tick")))

    return {
        "timeframe": timeframe,
        "compared": len(errors),
        "max_error_pp": max(errors) if errors else None,
        "median_error_pp": statistics.median(errors) if errors else None,
        "within_tolerance": sum(1 for e in errors if e <= VOL_TOLERANCE_PP),
        "worst": worst,
        "skipped_no_series": skipped_no_series,
        "skipped_short_history": skipped_short,
    }


def check_h1_reconstruction(
    records: list[dict[str, Any]], root: Path, timeframe: str
) -> dict[str, Any]:
    """Check B: how wrong is a reconstructed price_change.h1?"""
    interval = 3600.0 if timeframe == "1h" else 300.0
    lookback = 1 if timeframe == "1h" else H1_LOOKBACK_5M
    cache: dict[str, list[bf.RawBar]] = {}
    errors: list[tuple[float, str, str]] = []

    for rec in records:
        live = rec.get("price_change_h1")
        pool = rec.get("pool")
        when = rec.get("receive_time")
        if live is None or not pool or when is None:
            continue
        if pool not in cache:
            cache[pool] = load_series(root, pool, timeframe)
        usable = closed_before(cache[pool], float(when), interval)
        if len(usable) <= lookback:
            continue
        now_close = usable[-1].close
        then_close = usable[-1 - lookback].close
        if then_close <= 0.0:
            continue
        rebuilt = (now_close / then_close - 1.0) * 100.0
        errors.append(
            (abs(rebuilt - float(live)), str(rec.get("symbol")), str(rec.get("tick")))
        )

    values = [e[0] for e in errors]
    return {
        "timeframe": timeframe,
        "compared": len(values),
        "median_error_pp": statistics.median(values) if values else None,
        "p90_error_pp": (
            statistics.quantiles(values, n=10)[-1] if len(values) >= 10 else None
        ),
        "max_error_pp": max(values) if values else None,
        "worst": max(errors, default=None),
    }


def check_cross_timeframe(root: Path, pools: list[str]) -> dict[str, Any]:
    """Check C: 5m resampled to hourly against the fetched 1h series."""
    results = []
    for pool in pools:
        m5 = load_series(root, pool, "5m")
        h1 = load_series(root, pool, "1h")
        if not m5 or not h1:
            continue
        # An hourly bar's close is the close of the last 5m bar inside it.
        buckets: dict[float, bf.RawBar] = {}
        for bar in m5:
            hour = bar.ts - (bar.ts % 3600.0)
            prev = buckets.get(hour)
            if prev is None or bar.ts > prev.ts:
                buckets[hour] = bar
        diffs = []
        for bar in h1:
            got = buckets.get(bar.ts)
            if got is None or bar.close <= 0:
                continue
            diffs.append(abs(got.close / bar.close - 1.0) * 100.0)
        if diffs:
            results.append(
                {
                    "pool": pool,
                    "hours_compared": len(diffs),
                    "median_pct": statistics.median(diffs),
                    "max_pct": max(diffs),
                    "exact": sum(1 for d in diffs if d < 1e-9),
                }
            )
    return {"pools": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ticks", required=True, type=Path, help="JSONL from extract_ticks."
    )
    parser.add_argument("--history", default=Path("history"), type=Path)
    parser.add_argument("--json", type=Path, help="Write the full report here.")
    args = parser.parse_args()

    records = [json.loads(line) for line in args.ticks.read_text().splitlines() if line]

    report: dict[str, Any] = {
        "records": len(records),
        "check_a_realized_vol": [
            check_vol(records, args.history, "1h", "h1_realized_vol_pct"),
            check_vol(records, args.history, "5m", "m5_realized_vol_pct"),
        ],
        "check_b_h1_reconstruction": [
            check_h1_reconstruction(records, args.history, "5m"),
            check_h1_reconstruction(records, args.history, "1h"),
        ],
        "check_c_cross_timeframe": check_cross_timeframe(
            args.history,
            sorted({str(r["pool"]) for r in records if r.get("pool")}),
        ),
    }

    print(json.dumps(report, indent=2, default=str))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    failed = False
    for result in report["check_a_realized_vol"]:
        compared = result["compared"]
        if compared and result["within_tolerance"] != compared:
            failed = True
            print(
                f"FAIL check A {result['timeframe']}: "
                f"{compared - result['within_tolerance']} of {compared} outside "
                f"{VOL_TOLERANCE_PP}pp; worst {result['worst']}",
                file=sys.stderr,
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
