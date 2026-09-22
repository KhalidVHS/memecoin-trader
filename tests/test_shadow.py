"""Offline tests for ``shadow.py``.

Everything here is strictly offline: all HTTP calls are intercepted by an
``httpx.MockTransport``, which is the established pattern in this repo (see
``tests/test_backfill.py`` and ``tests/test_quotes.py``). There are no live
network calls and there must not be — a test that calls Jupiter in CI fails on
a bad API day and obscures the real failure.

Cases covered
-------------
* Ladder has one rung per configured size and side.
* A 429 on a rung is retried (via http.execute) rather than counted as a
  permanent failure.
* A partial / interrupted run resumes from the manifest and does not
  double-write already-collected coins.
* ``available_time`` on every snapshot is >= the receive time of its last rung.
* A vendor error for one rung does not discard the rungs already collected
  in that ladder snapshot (same lesson as ``backfill.HistoryHorizon``).
* The manifest round-trips (write / read preserves all fields).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from memetrader.http import make_client
from memetrader.shadow import (
    DEFAULT_LADDER_USD,
    LadderConfig,
    ShadowMeta,
    collect,
    ladder_path,
    load_universe,
    read_manifest,
    write_manifest,
)
from memetrader.types import Side

# ---------------------------------------------------------------------------
# Constants shared across tests
# ---------------------------------------------------------------------------

MINT = "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm"
SYMBOL = "BONK"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
JUPITER_BASE = "https://lite.jupiter.aggregator.app"

# A real epoch timestamp well inside 2020..2100.
T0 = 1_760_000_400.0
DATE_STR = "2025-10-09"  # matches T0 UTC date


def _token_search_response(mint: str, decimals: int = 5) -> list[dict[str, Any]]:
    """Minimal /tokens/v2/search response for a given mint."""
    return [{"id": mint, "decimals": decimals, "symbol": "BONK"}]


def _quote_response(
    in_amount: int,
    out_amount: int,
    slot: int = 12345,
    impact_pct: str = "0.001",
) -> dict[str, Any]:
    """Minimal /swap/v1/quote response."""
    return {
        "inAmount": str(in_amount),
        "outAmount": str(out_amount),
        "otherAmountThreshold": str(max(1, out_amount - 10)),
        "priceImpactPct": impact_pct,
        "contextSlot": slot,
        "routePlan": [{"swapInfo": {"label": "Orca"}}],
        "swapUsdValue": "100.0",
    }


def _make_mock_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _make_client(handler) -> httpx.Client:
    return make_client(transport=_make_mock_transport(handler))


# ---------------------------------------------------------------------------
# Helpers for deterministic clock/sleep
# ---------------------------------------------------------------------------


class FakeClock:
    """A monotonically advancing fake clock.

    Each call to ``tick()`` advances by ``step`` seconds. Used to give every
    rung a distinct ``requested_at`` / ``received_at`` without real sleeps.
    """

    def __init__(self, start: float = T0, step: float = 0.05) -> None:
        self._t = start
        self._step = step

    def __call__(self) -> float:
        t = self._t
        self._t += self._step
        return t

    @property
    def current(self) -> float:
        return self._t


def _noop_sleep(_: float) -> None:
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def coins() -> list[dict[str, Any]]:
    return [{"symbol": SYMBOL, "mint": MINT}]


@pytest.fixture()
def ladder_cfg_small() -> LadderConfig:
    """A tiny ladder (two sizes, one side) so tests run fast."""
    return LadderConfig(
        ladder_usd=(10.0, 50.0),
        sides=(Side.BUY,),
        pace_seconds=0.0,  # no real sleeping in tests
        slippage_bps=50,
    )


def _build_handler_always_ok(
    *,
    buy_out: int = 500_000,
    slot: int = 99_000,
) -> Any:
    """An httpx handler that serves valid token-meta and quote responses."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "tokens/v2/search" in url:
            body = _token_search_response(MINT, decimals=5)
            return httpx.Response(200, json=body)
        if "swap/v1/quote" in url:
            # Parse in_amount from query string.
            params = dict(request.url.params)
            in_amount = int(params.get("amount", "1000000"))
            return httpx.Response(200, json=_quote_response(in_amount, buy_out, slot=slot))
        return httpx.Response(404, json={"error": "not found"})

    return handler


# ---------------------------------------------------------------------------
# Test: ladder rung count matches configuration
# ---------------------------------------------------------------------------


def test_ladder_requests_one_rung_per_configured_size_and_side(
    tmp_root: Path,
    coins: list[dict[str, Any]],
    ladder_cfg_small: LadderConfig,
) -> None:
    """``collect`` must request exactly (len(sides) * len(ladder_usd)) rungs
    per coin. A missing rung means a gap in the cost curve that cannot be
    interpolated across.
    """
    handler = _build_handler_always_ok()
    client = _make_client(handler)

    clock = FakeClock()
    report = collect(
        coins=coins,
        root=tmp_root,
        jupiter_base=JUPITER_BASE,
        ladder_cfg=ladder_cfg_small,
        resume=False,
        now=clock,
        sleep=_noop_sleep,
        client=client,
    )

    assert SYMBOL in report.written
    assert not report.failed

    # Verify the stored file.
    shadow_dir = tmp_root / "shadow"
    assert shadow_dir.exists()
    written_date = next(iter(shadow_dir.iterdir())).name

    path = ladder_path(tmp_root, written_date, SYMBOL)
    assert path.exists()

    # Read the snapshot lines.
    snapshots = _read_snapshots(path)
    assert len(snapshots) == 1

    snap = snapshots[0]
    expected_rung_count = len(ladder_cfg_small.sides) * len(ladder_cfg_small.ladder_usd)
    assert len(snap["rungs"]) == expected_rung_count


def _read_snapshots(path: Path) -> list[dict[str, Any]]:
    """Read all JSON lines from a gzipped JSONL file."""
    lines = []
    with gzip.open(path, "rt", encoding="utf-8") as gz:
        for line in gz:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines


# ---------------------------------------------------------------------------
# Test: 429 is retried, not dropped
# ---------------------------------------------------------------------------


def test_429_is_retried_not_dropped(
    tmp_root: Path,
    coins: list[dict[str, Any]],
    ladder_cfg_small: LadderConfig,
) -> None:
    """A 429 on the first attempt of a rung must be retried by http.execute,
    not counted as a permanent failure. This is the lesson from the POPCAT
    outage: market.py used client.get() directly and lost 38 of 39 fetches to
    429s it never retried.
    """
    call_counts: dict[str, int] = {"quote": 0, "token": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "tokens/v2/search" in url:
            call_counts["token"] += 1
            return httpx.Response(200, json=_token_search_response(MINT))
        if "swap/v1/quote" in url:
            call_counts["quote"] += 1
            params = dict(request.url.params)
            in_amount = int(params.get("amount", "1000000"))
            # First call for each rung returns 429; second returns OK.
            if call_counts["quote"] % 2 == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(200, json=_quote_response(in_amount, 500_000))
        return httpx.Response(404)

    client = _make_client(handler)
    clock = FakeClock()

    report = collect(
        coins=coins,
        root=tmp_root,
        jupiter_base=JUPITER_BASE,
        ladder_cfg=ladder_cfg_small,
        resume=False,
        now=clock,
        sleep=_noop_sleep,
        client=client,
    )

    assert SYMBOL in report.written
    # The quote endpoint must have been called more than once per rung (proving
    # the 429 caused a retry).
    expected_rungs = len(ladder_cfg_small.sides) * len(ladder_cfg_small.ladder_usd)
    assert call_counts["quote"] > expected_rungs

    # All rungs must have succeeded (quality = ok).
    shadow_dir = tmp_root / "shadow"
    date_dirs = list(shadow_dir.iterdir())
    path = ladder_path(tmp_root, date_dirs[0].name, SYMBOL)
    snaps = _read_snapshots(path)
    assert all(r["quality"] == "ok" for r in snaps[0]["rungs"])


# ---------------------------------------------------------------------------
# Test: resume does not duplicate
# ---------------------------------------------------------------------------


def test_resume_skips_already_collected_coins(
    tmp_root: Path,
    coins: list[dict[str, Any]],
    ladder_cfg_small: LadderConfig,
) -> None:
    """An interrupted run resumed with ``resume=True`` must not write a second
    snapshot for coins already in the manifest. Double-writing would make the
    replay engine see the same quote twice and count it as more evidence than
    it is.
    """
    handler = _build_handler_always_ok()
    client = _make_client(handler)
    clock = FakeClock()

    # First sweep.
    report1 = collect(
        coins=coins,
        root=tmp_root,
        jupiter_base=JUPITER_BASE,
        ladder_cfg=ladder_cfg_small,
        resume=True,
        now=clock,
        sleep=_noop_sleep,
        client=client,
    )
    assert SYMBOL in report1.written

    # Locate the date dir.
    shadow_dir = tmp_root / "shadow"
    date_dirs = list(shadow_dir.iterdir())
    date_str = date_dirs[0].name
    path = ladder_path(tmp_root, date_str, SYMBOL)
    snap_count_after_first = len(_read_snapshots(path))

    # Second sweep with resume=True — the coin is in the manifest.
    report2 = collect(
        coins=coins,
        root=tmp_root,
        jupiter_base=JUPITER_BASE,
        ladder_cfg=ladder_cfg_small,
        resume=True,
        now=clock,
        sleep=_noop_sleep,
        client=client,
    )
    assert SYMBOL in report2.skipped
    assert SYMBOL not in report2.written

    snap_count_after_second = len(_read_snapshots(path))
    assert snap_count_after_second == snap_count_after_first, (
        "resume=True must not append a second snapshot for an already-collected coin"
    )


# ---------------------------------------------------------------------------
# Test: available_time >= last rung receive time
# ---------------------------------------------------------------------------


def test_available_time_is_set_to_receive_time_on_every_snapshot(
    tmp_root: Path,
    coins: list[dict[str, Any]],
    ladder_cfg_small: LadderConfig,
) -> None:
    """``available_time`` on every snapshot must be set to the receive time of
    the last rung (the honest floor for a live read). A replay using
    ``event_time`` (the first rung's request time) would grant itself the
    duration of the collection as foresight — typically 5-30 seconds on 14
    rungs, which is enough to see several AMM state updates.
    """
    handler = _build_handler_always_ok()
    client = _make_client(handler)
    clock = FakeClock(start=T0, step=0.1)

    report = collect(
        coins=coins,
        root=tmp_root,
        jupiter_base=JUPITER_BASE,
        ladder_cfg=ladder_cfg_small,
        resume=False,
        now=clock,
        sleep=_noop_sleep,
        client=client,
    )
    assert SYMBOL in report.written

    shadow_dir = tmp_root / "shadow"
    date_str = next(iter(shadow_dir.iterdir())).name
    path = ladder_path(tmp_root, date_str, SYMBOL)
    snaps = _read_snapshots(path)
    assert len(snaps) == 1
    snap = snaps[0]

    # available_time must be >= event_time (time advanced during collection).
    assert snap["available_time"] >= snap["event_time"]

    # available_time must be >= the received_at of every rung.
    for rung in snap["rungs"]:
        assert snap["available_time"] >= rung["received_at"], (
            "available_time must be >= every rung's received_at to satisfy "
            "the point-in-time invariant"
        )


# ---------------------------------------------------------------------------
# Test: one rung error does not discard other rungs
# ---------------------------------------------------------------------------


def test_rung_error_does_not_discard_other_rungs(
    tmp_root: Path,
    coins: list[dict[str, Any]],
    ladder_cfg_small: LadderConfig,
) -> None:
    """A vendor error for one rung size must not discard the rungs already
    collected. This is the analogue of ``backfill.HistoryHorizon``: treating
    it as a fatal error would discard the other good rungs, making the cost
    curve unusable for the sizes that did succeed.
    """
    # The 50 USD rung fails permanently, on every retry attempt.
    # 50 USD = 50_000_000 micro-USDC (USDC has 6 decimals)
    fail_amount = 50_000_000

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "tokens/v2/search" in url:
            return httpx.Response(200, json=_token_search_response(MINT))
        if "swap/v1/quote" in url:
            params = dict(request.url.params)
            in_amount = int(params.get("amount", "1000000"))
            # Permanently fail the 50 USD rung on all attempts (including
            # retries). 404 is non-retryable so a single response is enough.
            if in_amount == fail_amount:
                return httpx.Response(404, json={"error": "not found"})
            return httpx.Response(200, json=_quote_response(in_amount, 500_000))
        return httpx.Response(404)

    client = _make_client(handler)
    clock = FakeClock()

    report = collect(
        coins=coins,
        root=tmp_root,
        jupiter_base=JUPITER_BASE,
        ladder_cfg=ladder_cfg_small,  # 2 sizes x 1 side = 2 rungs
        resume=False,
        now=clock,
        sleep=_noop_sleep,
        client=client,
    )

    # The snapshot is still written (degraded), not dropped.
    assert SYMBOL in report.written

    shadow_dir = tmp_root / "shadow"
    date_str = next(iter(shadow_dir.iterdir())).name
    path = ladder_path(tmp_root, date_str, SYMBOL)
    snaps = _read_snapshots(path)
    assert len(snaps) == 1, "snapshot must be written even when one rung fails"

    rungs = snaps[0]["rungs"]
    assert len(rungs) == 2  # both rungs attempted

    ok_rungs = [r for r in rungs if r["quality"] == "ok"]
    bad_rungs = [r for r in rungs if r["quality"] != "ok"]
    assert len(ok_rungs) == 1, "one rung must succeed"
    assert len(bad_rungs) == 1, "one rung must fail"


# ---------------------------------------------------------------------------
# Test: manifest round-trips
# ---------------------------------------------------------------------------


def test_manifest_round_trips(tmp_root: Path) -> None:
    """``write_manifest`` / ``read_manifest`` must preserve all fields exactly.

    The manifest is the provenance record for the shadow dataset. A field
    silently lost on round-trip is a provenance claim that cannot be verified.
    """
    date_str = "2025-10-09"
    meta = ShadowMeta(
        symbol=SYMBOL,
        mint=MINT,
        date=date_str,
        snapshots=3,
        first_ts=T0,
        last_ts=T0 + 3600.0,
        rung_count=14,
        rung_ok=12,
        rung_no_route=2,
        rung_error=0,
        ladder_usd=DEFAULT_LADDER_USD,
        sides=("BUY", "SELL"),
        quality="degraded",
        written_at=T0 + 3700.0,
    )

    write_manifest(tmp_root, date_str, [meta])
    entries = read_manifest(tmp_root, date_str)

    assert len(entries) == 1
    recovered = entries[0]
    assert recovered.symbol == meta.symbol
    assert recovered.mint == meta.mint
    assert recovered.date == meta.date
    assert recovered.snapshots == meta.snapshots
    assert recovered.first_ts == pytest.approx(meta.first_ts)
    assert recovered.last_ts == pytest.approx(meta.last_ts)
    assert recovered.rung_count == meta.rung_count
    assert recovered.rung_ok == meta.rung_ok
    assert recovered.rung_no_route == meta.rung_no_route
    assert recovered.rung_error == meta.rung_error
    assert recovered.ladder_usd == meta.ladder_usd
    assert recovered.sides == meta.sides
    assert recovered.quality == meta.quality
    assert recovered.written_at == pytest.approx(meta.written_at)
    assert recovered.tool_version == meta.tool_version


# ---------------------------------------------------------------------------
# Test: two coins — failure of one does not stop the other
# ---------------------------------------------------------------------------


def test_one_coin_failing_does_not_abort_other_coins(
    tmp_root: Path,
    ladder_cfg_small: LadderConfig,
) -> None:
    """A token_meta failure for one coin (e.g. Jupiter doesn't know its mint)
    must not prevent the next coin from being collected. One bad coin is filed
    under ``report.failed``; the rest continue.
    """
    mint2 = "UNKNOWN_MINT_XYZ"
    symbol2 = "UNKNOWN"
    coins = [{"symbol": SYMBOL, "mint": MINT}, {"symbol": symbol2, "mint": mint2}]

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "tokens/v2/search" in url:
            query = str(request.url.params.get("query", ""))
            if MINT in query:
                return httpx.Response(200, json=_token_search_response(MINT))
            # Return empty list for unknown mint — token_meta returns None.
            return httpx.Response(200, json=[])
        if "swap/v1/quote" in url:
            params = dict(request.url.params)
            in_amount = int(params.get("amount", "1_000_000"))
            return httpx.Response(200, json=_quote_response(in_amount, 500_000))
        return httpx.Response(404)

    client = _make_client(handler)
    clock = FakeClock()

    report = collect(
        coins=coins,
        root=tmp_root,
        jupiter_base=JUPITER_BASE,
        ladder_cfg=ladder_cfg_small,
        resume=False,
        now=clock,
        sleep=_noop_sleep,
        client=client,
    )

    assert SYMBOL in report.written
    # UNKNOWN should appear in failed (no verified decimals).
    failed_syms = [sym for sym, _ in report.failed]
    assert symbol2 in failed_syms


# ---------------------------------------------------------------------------
# Test: load_universe
# ---------------------------------------------------------------------------


def test_load_universe_reads_coins_from_toml(tmp_path: Path) -> None:
    """``load_universe`` must parse a minimal TOML and return coin dicts."""
    toml_text = """
[[coins]]
symbol = "BONK"
mint = "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm"

[[coins]]
symbol = "WIF"
mint = "EbMg8ab7y4RxLxTHDSRzNtcGmKMmViS89YGh8cPxbR5e"
"""
    toml_file = tmp_path / "universe.toml"
    toml_file.write_text(toml_text, encoding="utf-8")

    coins = load_universe(toml_file)
    assert len(coins) == 2
    assert coins[0]["symbol"] == "BONK"
    assert coins[1]["symbol"] == "WIF"


def test_load_universe_raises_on_empty_coins(tmp_path: Path) -> None:
    toml_text = "[meta]\nfoo = 1\n"
    toml_file = tmp_path / "empty.toml"
    toml_file.write_text(toml_text, encoding="utf-8")
    with pytest.raises(ValueError, match="no \\[\\[coins\\]\\]"):
        load_universe(toml_file)
