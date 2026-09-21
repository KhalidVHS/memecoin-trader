"""Tests for the replay tooling that checks backfilled bars against live records.

These cover the two tools that turn the backfill from "a download" into
"evidence": the extractor that recovers live evidence bundles from the
documented-run tick pages, and the checks that compare them against downloaded
bars. Everything here is offline and constructed — no network, no dependence on
whether a real ``history/`` tree exists.

The point of testing a verification tool is that a broken checker reports
success, which is worse than no checker at all.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import extract_ticks
import verify_backfill as vb

POOL = "EP2ib6dYdEeqD8MfE2ezHCxX3kP3K2eLKkirfPm5eyMx"
MINT = "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm"
T0 = 1_760_000_400.0
HOUR = 3600.0


def bundle(symbol: str = "WIF", *, h1_vol: float | None = 1.5) -> dict[str, object]:
    """A minimal evidence bundle shaped like the ones the runs actually wrote."""
    return {
        "symbol": symbol,
        "snapshot": {
            "symbol": symbol,
            "mint": MINT,
            "price_usd": 0.2,
            "price_change": {"m5": -0.6, "h1": 1.13},
            "pool": {"pair_address": POOL, "trusted_quote": True},
            "provenance": {"receive_time": T0 + 50 * HOUR, "quality": "ok"},
        },
        "technicals": {
            "symbol": symbol,
            "h1": {"pool_address": POOL, "candles_used": 99, "realized_vol_pct": h1_vol},
            "m5": {"pool_address": POOL, "candles_used": 99, "realized_vol_pct": 0.5},
        },
    }


def tick_page(*bundles: dict[str, object]) -> str:
    """Render bundles the way document_run writes them: fenced inside <details>."""
    parts = ["# Slow tick #1\n\nsome prose\n"]
    parts.extend(
        f"<details><summary><b>{item['symbol']}</b></summary>\n\n"
        f"~~~json\n{json.dumps(item, indent=2)}\n~~~\n\n</details>\n"
        for item in bundles
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# extract_ticks
# ---------------------------------------------------------------------------


def test_extract_recovers_every_bundle_on_a_page(tmp_path: Path) -> None:
    ticks = tmp_path / "run-a" / "ticks"
    ticks.mkdir(parents=True)
    (ticks / "tick-0001-x.md").write_text(
        tick_page(bundle("BONK"), bundle("WIF"), bundle("POPCAT")), encoding="utf-8"
    )

    records = extract_ticks.extract(tmp_path)

    assert [r["symbol"] for r in records] == ["BONK", "WIF", "POPCAT"]
    assert all(r["run"] == "run-a" for r in records)
    assert all(r["pool"] == POOL for r in records)


def test_extract_skips_non_bundle_json_blocks(tmp_path: Path) -> None:
    """Tick pages also carry prompt and report blocks. Those are not evidence.

    Failing on them would make the extractor depend on page layout; silently
    counting them would inflate the comparison set with records that have no
    snapshot to compare against.
    """
    ticks = tmp_path / "run-a" / "ticks"
    ticks.mkdir(parents=True)
    page = tick_page(bundle("WIF"))
    page += '\n~~~json\n{"not_a_bundle": true}\n~~~\n'
    page += "\n~~~json\nthis is not json at all\n~~~\n"
    (ticks / "tick-0001-x.md").write_text(page, encoding="utf-8")

    records = extract_ticks.extract(tmp_path)

    assert len(records) == 1
    assert records[0]["symbol"] == "WIF"


def test_extract_orders_ticks_chronologically(tmp_path: Path) -> None:
    """Filenames are zero-padded, so lexical order is chronological order."""
    ticks = tmp_path / "run-a" / "ticks"
    ticks.mkdir(parents=True)
    for index in (10, 2, 1):
        (ticks / f"tick-{index:04d}-x.md").write_text(
            tick_page(bundle("WIF")), encoding="utf-8"
        )

    records = extract_ticks.extract(tmp_path)

    assert [r["tick"] for r in records] == [
        "tick-0001-x.md",
        "tick-0002-x.md",
        "tick-0010-x.md",
    ]


def test_extract_keeps_a_missing_field_as_none(tmp_path: Path) -> None:
    """A live outage is a fact about the run, not a parse error.

    The core invariant: None means "could not find out", never zero. One real
    tick recorded price_change.h1 as absent; flattening that to 0.0 would hand
    the fidelity check a fabricated agreement.
    """
    ticks = tmp_path / "run-a" / "ticks"
    ticks.mkdir(parents=True)
    broken = bundle("WIF")
    broken["technicals"] = None
    del broken["snapshot"]["price_change"]["h1"]  # type: ignore[index]
    (ticks / "tick-0001-x.md").write_text(tick_page(broken), encoding="utf-8")

    (record,) = extract_ticks.extract(tmp_path)

    assert record["price_change_h1"] is None
    assert record["h1_realized_vol_pct"] is None
    assert record["h1_pool"] is None
    assert record["pool"] == POOL  # the snapshot half still resolved


# ---------------------------------------------------------------------------
# verify_backfill
# ---------------------------------------------------------------------------


def test_realized_vol_matches_the_shipped_implementation() -> None:
    """The reimplementation must agree with signals._realized_vol_pct.

    It is written out separately so that a change to either side shows up as a
    disagreement rather than being masked by a shared import — but it still has
    to be the same function today.
    """
    import numpy as np

    from memetrader.signals import _realized_vol_pct

    closes = [1.0 + 0.01 * math.sin(i) for i in range(40)]
    expected = _realized_vol_pct(np.array(closes))
    got = vb.realized_vol_pct(closes)

    assert expected is not None
    assert got is not None
    assert got == pytest.approx(expected, abs=1e-12)


def test_realized_vol_needs_a_full_window() -> None:
    assert vb.realized_vol_pct([1.0] * vb.VOL_PERIOD) is None
    assert vb.realized_vol_pct([1.0] * (vb.VOL_PERIOD + 1)) is not None


def test_realized_vol_refuses_non_positive_closes() -> None:
    """A log return needs a positive price. Zero here would mean 'no trades',
    and taking its log would turn missing data into -inf volatility."""
    closes = [1.0] * (vb.VOL_PERIOD + 1)
    closes[3] = 0.0
    assert vb.realized_vol_pct(closes) is None


def test_closed_before_excludes_the_forming_bar() -> None:
    """Replay must not see a close the live system could not have known."""
    from memetrader.backfill import RawBar

    bars = [
        RawBar(ts=T0 + i * HOUR, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)
        for i in range(4)
    ]
    # Standing inside the bar that opened at T0+3h: it has not closed yet.
    usable = vb.closed_before(bars, T0 + 3 * HOUR + 60.0, HOUR)

    assert [b.ts for b in usable] == [T0, T0 + HOUR, T0 + 2 * HOUR]


def test_closed_before_includes_a_bar_that_closed_exactly_now() -> None:
    from memetrader.backfill import RawBar

    bars = [RawBar(ts=T0, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)]

    assert vb.closed_before(bars, T0 + HOUR, HOUR) == bars


def test_check_a_flags_a_series_that_does_not_match(tmp_path: Path) -> None:
    """The check must fail when the bars are wrong. A checker that always
    passes is worse than no checker."""
    from memetrader import backfill as bf
    from memetrader.backfill import RawBar
    from memetrader.types import Timeframe

    closes = [1.0 + 0.01 * math.sin(i) for i in range(40)]
    bars = [
        RawBar(ts=T0 + i * HOUR, open=c, high=c, low=c, close=c, volume=1.0)
        for i, c in enumerate(closes)
    ]
    bf.write_series(bf.series_path(tmp_path, POOL, "1h"), bars)

    truth = vb.realized_vol_pct([b.close for b in bars])
    assert truth is not None

    record = {
        "symbol": "WIF",
        "tick": "t",
        "pool": POOL,
        "receive_time": bars[-1].ts + HOUR,
        "h1_realized_vol_pct": truth,
        "h1_pool": POOL,
    }
    good = vb.check_vol([record], tmp_path, "1h", "h1_realized_vol_pct")
    assert good["compared"] == 1
    assert good["within_tolerance"] == 1

    # Same bars, a live value that disagrees: must not be tolerated.
    bad = vb.check_vol(
        [{**record, "h1_realized_vol_pct": truth + 0.5}],
        tmp_path,
        "1h",
        "h1_realized_vol_pct",
    )
    assert bad["compared"] == 1
    assert bad["within_tolerance"] == 0
    assert bad["max_error_pp"] == pytest.approx(0.5, abs=1e-9)
    assert Timeframe.H1.value == "1h"


def test_check_a_reports_a_missing_series_rather_than_passing(tmp_path: Path) -> None:
    """No file must never look like agreement."""
    record = {
        "symbol": "WIF",
        "tick": "t",
        "pool": POOL,
        "receive_time": T0,
        "h1_realized_vol_pct": 1.5,
        "h1_pool": POOL,
    }

    result = vb.check_vol([record], tmp_path, "1h", "h1_realized_vol_pct")

    assert result["compared"] == 0
    assert result["skipped_no_series"] == 1
