"""Offline tests for ``market.py``, driven entirely by saved live responses.

Every fixture in ``tests/fixtures`` is a verbatim capture of a real response,
so these tests fail if our parsing drifts from what the vendors actually send —
which is the only failure mode that matters for an adapter. No test here may
touch the network: an ``httpx.MockTransport`` answers every route, and an
unrouted request is an explicit failure rather than a silent live call.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from pathlib import Path

import httpx
import pytest

from memetrader import config, market
from memetrader.market import MarketDataError

FIXTURES = Path(__file__).parent / "fixtures"

BONK_MINT = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture(scope="module")
def cfg():
    return config.load()


@pytest.fixture(scope="module")
def bonk_cfg(cfg):
    """The hand-built selection fixture only contains BONK pairs, and a coin
    with no pair is (correctly) fatal — so narrow the config rather than pad
    the fixture with noise that has nothing to do with what is being tested."""
    return dataclasses.replace(cfg, coins=(cfg.coin("BONK"),))


@pytest.fixture(autouse=True)
def _no_polite_delay(monkeypatch):
    """The real inter-request delay exists for GeckoTerminal's rate limiter and
    has nothing to say about correctness; paying it 6x per test is pure waste."""
    monkeypatch.setattr(market, "_GECKO_DELAY_SECONDS", 0.0)


def make_client(
    *,
    tokens: dict | None = None,
    tokens_status: int = 200,
    ohlcv_5m: dict | None = None,
    ohlcv_1h: dict | None = None,
    ohlcv_status: int = 200,
) -> httpx.Client:
    """An httpx.Client whose transport serves fixtures instead of the internet."""
    tokens = tokens if tokens is not None else load_fixture("dexscreener_tokens.json")
    ohlcv_5m = ohlcv_5m or load_fixture("geckoterminal_5m.json")
    ohlcv_1h = ohlcv_1h or load_fixture("geckoterminal_1h.json")

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "/dex/tokens/" in path:
            if tokens_status != 200:
                return httpx.Response(tokens_status, text="blocked")
            return httpx.Response(200, json=tokens)
        if path.endswith("/ohlcv/minute"):
            if ohlcv_status != 200:
                return httpx.Response(ohlcv_status, text="rate limited")
            return httpx.Response(200, json=ohlcv_5m)
        if path.endswith("/ohlcv/hour"):
            if ohlcv_status != 200:
                return httpx.Response(ohlcv_status, text="rate limited")
            return httpx.Response(200, json=ohlcv_1h)
        raise AssertionError(f"unrouted request (would have hit the network): {request.url}")

    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------


def test_picks_highest_liquidity_pair_not_pairs_zero(bonk_cfg):
    """The synthetic fixture is arranged so pairs[0] is a dust pool."""
    tokens = load_fixture("dexscreener_pair_selection.json")
    assert tokens["pairs"][0]["pairAddress"] == "5zDeadPoolNeverPickMe"

    with make_client(tokens=tokens) as client:
        snap = market.snapshot(bonk_cfg, client=client)

    bonk = snap.coins["BONK"]
    assert bonk.pair_address == "3UBestRealUsdcPool"
    assert bonk.liquidity_usd == 900000.0


def test_ignores_fake_liquidity_pool_quoted_in_a_junk_token(bonk_cfg):
    """Highest raw ``liquidity.usd`` is a DLMM quoted in a pump.fun token that
    reports BONK at $0.01434. Selecting it would poison every downstream number."""
    tokens = load_fixture("dexscreener_pair_selection.json")
    fake = next(p for p in tokens["pairs"] if p["pairAddress"] == "CdFakeDlmmQuotedInJunk")
    assert fake["liquidity"]["usd"] == max(p["liquidity"]["usd"] for p in tokens["pairs"])

    with make_client(tokens=tokens) as client:
        snap = market.snapshot(bonk_cfg, client=client)

    assert snap.coins["BONK"].pair_address != "CdFakeDlmmQuotedInJunk"
    assert snap.coins["BONK"].price_usd < 1e-4


def test_live_capture_selects_sane_pairs_for_all_three_coins(cfg):
    """Same rule against the real captured response, where BONK's top-liquidity
    pair really is the fake one."""
    with make_client() as client:
        snap = market.snapshot(cfg, client=client)

    assert set(snap.coins) == {"BONK", "WIF", "POPCAT"}
    assert snap.coins["BONK"].price_usd == pytest.approx(2.944e-06, rel=0.01)
    assert snap.coins["WIF"].price_usd == pytest.approx(0.2043, rel=0.01)
    assert snap.coins["POPCAT"].price_usd == pytest.approx(0.04890, rel=0.01)
    for coin in snap.coins.values():
        assert not coin.degraded, coin.degraded_reason


def test_degrades_when_no_numeraire_quoted_pool_exists(bonk_cfg):
    """If the junk pool is the *only* pool, we still price off it — but the
    snapshot must say out loud that the price is untrusted."""
    tokens = load_fixture("dexscreener_pair_selection.json")
    tokens["pairs"] = [
        p for p in tokens["pairs"] if p["pairAddress"] == "CdFakeDlmmQuotedInJunk"
    ]
    with make_client(tokens=tokens) as client:
        snap = market.snapshot(bonk_cfg, client=client)

    bonk = snap.coins["BONK"]
    assert bonk.degraded
    assert "L$L" in bonk.degraded_reason


def test_missing_pair_raises_rather_than_reporting_a_zero_price(cfg):
    with make_client(tokens={"schemaVersion": "1.0.0", "pairs": []}) as client:
        with pytest.raises(MarketDataError, match="no tradeable pair"):
            market.snapshot(cfg, client=client)


def test_resolve_pairs_returns_symbol_to_address(cfg):
    with make_client() as client:
        resolved = market.resolve_pairs(cfg, client=client)

    assert resolved["BONK"] == "5zpyutJu9ee6jFymDGoK7F6S5Kczqtc9FomP3ueKuyA9"
    assert resolved["WIF"] == "EP2ib6dYdEeqD8MfE2ezHCxX3kP3K2eLKkirfPm5eyMx"
    assert resolved["POPCAT"] == "FRhB8L7Y9Qq41qZXYLtC2nw8An1RJfLLxRF2x9RwLLMo"


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------


def test_string_numerics_are_coerced_to_float(bonk_cfg):
    tokens = load_fixture("dexscreener_pair_selection.json")
    best = next(p for p in tokens["pairs"] if p["pairAddress"] == "3UBestRealUsdcPool")
    assert isinstance(best["priceUsd"], str)  # guard: the fixture must stay stringly-typed

    with make_client(tokens=tokens) as client:
        bonk = market.snapshot(bonk_cfg, client=client).coins["BONK"]

    assert isinstance(bonk.price_usd, float)
    assert bonk.price_usd == pytest.approx(2.943e-06)
    assert isinstance(bonk.liquidity_usd, float)
    assert isinstance(bonk.volume_24h_usd, float)
    assert bonk.volume_24h_usd == pytest.approx(1250000.5)
    assert isinstance(bonk.txns_h1.buys, int)


def test_absent_price_change_window_is_none_not_zero(bonk_cfg):
    """DexScreener omits ``priceChange.m5`` entirely on quiet pools — observed
    on 13 of 30 live pairs, including the pool we select for BONK.

    It must survive as ``None``. Collapsing it to 0.0 would tell the model the
    price was flat over five minutes when in fact nothing was reported, and a
    flat reading is a tradeable claim in a way that silence is not.
    """
    tokens = load_fixture("dexscreener_pair_selection.json")
    best = next(p for p in tokens["pairs"] if p["pairAddress"] == "3UBestRealUsdcPool")
    assert "m5" not in best["priceChange"]

    with make_client(tokens=tokens) as client:
        bonk = market.snapshot(bonk_cfg, client=client).coins["BONK"]

    assert bonk.price_change.m5 is None
    assert bonk.price_change.h1 == pytest.approx(0.93)


# ---------------------------------------------------------------------------
# The timestamp bug
# ---------------------------------------------------------------------------


def test_dexscreener_millisecond_timestamp_lands_in_a_sane_year(cfg):
    """``pairCreatedAt`` is 1671980424000 — epoch *milliseconds*, Dec 2022.

    Left unconverted it is the year 54,977, and every staleness check that ever
    looks at it passes without complaint.
    """
    with make_client() as client:
        bonk = market.snapshot(cfg, client=client).coins["BONK"]

    created = dt.datetime.fromtimestamp(bonk.pair_created_at, tz=dt.UTC)
    assert created.year == 2022
    assert 2020 <= created.year <= 2100


def test_geckoterminal_second_timestamps_pass_through_unscaled(cfg):
    with make_client() as client:
        bonk = market.snapshot(cfg, client=client).coins["BONK"]

    for series in (bonk.candles_5m, bonk.candles_1h):
        for candle in series:
            year = dt.datetime.fromtimestamp(candle.ts, tz=dt.UTC).year
            assert 2020 <= year <= 2100


def test_snapshot_ts_is_seconds(cfg):
    with make_client() as client:
        snap = market.snapshot(cfg, client=client)
    assert 2020 <= dt.datetime.fromtimestamp(snap.ts, tz=dt.UTC).year <= 2100


# ---------------------------------------------------------------------------
# Candles
# ---------------------------------------------------------------------------


def test_candles_are_returned_oldest_first(cfg):
    """GeckoTerminal serves ohlcv_list newest-first; un-reversed, every
    indicator in signals.py computes over time-reversed data."""
    raw = load_fixture("geckoterminal_5m.json")["data"]["attributes"]["ohlcv_list"]
    assert raw[0][0] > raw[-1][0]  # guard: the capture really is descending

    with make_client() as client:
        bonk = market.snapshot(cfg, client=client).coins["BONK"]

    timestamps = [c.ts for c in bonk.candles_5m]
    assert timestamps == sorted(timestamps)
    assert bonk.candles_5m[-1].ts == float(raw[0][0])
    assert len(bonk.candles_5m) == 100
    assert len(bonk.candles_1h) == 100


def test_candle_five_minute_spacing(cfg):
    with make_client() as client:
        bonk = market.snapshot(cfg, client=client).coins["BONK"]
    deltas = {
        bonk.candles_5m[i + 1].ts - bonk.candles_5m[i].ts
        for i in range(len(bonk.candles_5m) - 1)
    }
    assert deltas == {300.0}


def test_geckoterminal_failure_degrades_instead_of_crashing_the_tick(cfg):
    with make_client(ohlcv_status=429) as client:
        snap = market.snapshot(cfg, client=client)

    for coin in snap.coins.values():
        assert coin.candles_5m == ()
        assert coin.candles_1h == ()
        assert coin.degraded
        assert "candles unavailable" in coin.degraded_reason
        # The part that matters: price and liquidity survived.
        assert coin.price_usd > 0
        assert coin.liquidity_usd > 0


def _counting_client() -> tuple[httpx.Client, list[str]]:
    """Like ``make_client``, but records every GeckoTerminal path it serves."""
    tokens = load_fixture("dexscreener_tokens.json")
    ohlcv_5m = load_fixture("geckoterminal_5m.json")
    ohlcv_1h = load_fixture("geckoterminal_1h.json")
    gecko: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "/dex/tokens/" in path:
            return httpx.Response(200, json=tokens)
        if path.endswith("/ohlcv/minute"):
            gecko.append(path)
            return httpx.Response(200, json=ohlcv_5m)
        if path.endswith("/ohlcv/hour"):
            gecko.append(path)
            return httpx.Response(200, json=ohlcv_1h)
        raise AssertionError(f"unrouted request (would have hit the network): {request.url}")

    return httpx.Client(transport=httpx.MockTransport(handler)), gecko


def test_price_only_snapshot_makes_no_geckoterminal_calls(cfg):
    """The absence of the call is the whole point, so it is what gets asserted.

    ``fast_tick`` marks the book off ``price_usd`` and never reads a candle,
    but it used to fetch them anyway: 2 calls per coin per minute against
    GeckoTerminal's ~30/min keyless budget, plus 1.5s of spacing between each.
    The 429s that bought landed on the *slow* tick, the one caller whose
    technicals actually depend on candles.
    """
    client, gecko = _counting_client()
    with client:
        snap = market.snapshot(cfg, client=client, with_candles=False)

    assert gecko == []
    for coin in snap.coins.values():
        assert coin.candles_5m == ()
        assert coin.candles_1h == ()
        # Not degraded: nothing failed, the caller asked for prices. Marking
        # this degraded would put a permanent false warning in front of the
        # model on every tick, which is how real warnings stop being read.
        assert not coin.degraded
        assert coin.price_usd > 0
        assert coin.liquidity_usd > 0


def test_the_default_snapshot_still_fetches_candles(cfg):
    """Guards the other direction: ``with_candles`` defaulting to False would
    silently strip technicals from the slow tick and nothing else would fail."""
    client, gecko = _counting_client()
    with client:
        snap = market.snapshot(cfg, client=client)

    assert len(gecko) == 2 * len(cfg.coins)
    assert all(coin.candles_5m and coin.candles_1h for coin in snap.coins.values())


def test_malformed_ohlcv_rows_are_skipped(cfg):
    broken = {"data": {"attributes": {"ohlcv_list": [[1789755000, 1, 2], "junk", None]}}}
    with make_client(ohlcv_5m=broken, ohlcv_1h=broken) as client:
        bonk = market.snapshot(cfg, client=client).coins["BONK"]
    assert bonk.candles_5m == ()


# ---------------------------------------------------------------------------
# Cloudflare
# ---------------------------------------------------------------------------


def test_403_raises_a_named_error_and_is_never_retried(cfg):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(403, text="cloudflare")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MarketDataError, match="403"):
            market.snapshot(cfg, client=client)

    assert calls["n"] == 1, "a 403 must not be retried"


def test_snapshot_sends_a_browser_user_agent(cfg):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("user-agent", ""))
        if "/dex/tokens/" in request.url.path:
            return httpx.Response(200, json=load_fixture("dexscreener_tokens.json"))
        return httpx.Response(200, json=load_fixture("geckoterminal_5m.json"))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        market.snapshot(cfg, client=client)

    assert seen and all("Mozilla/5.0" in ua for ua in seen)
    assert not any("python" in ua.lower() for ua in seen)
