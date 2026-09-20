"""Offline tests for ``market.py``, driven entirely by saved live responses.

Every fixture in ``tests/fixtures`` is a verbatim capture of a real response,
so these tests fail if our parsing drifts from what the vendors actually send —
which is the only failure mode that matters for an adapter. No test here may
touch the network: an ``httpx.MockTransport`` answers every route, and an
unrouted request is an explicit failure rather than a silent live call.

Two structural changes from the pre-audit suite:

* **The clock is injected.** ``market.snapshot(..., now=...)`` is frozen at
  ``NOW`` in every test, because the closed-bar watermark is a function of the
  read time and a watermark you cannot control is a watermark you cannot test.
  It also stops the captured fixtures ageing into "future-dated bar" rejections
  as wall-clock time passes them.
* **No ``Config``.** The adapter takes a ``MarketParams`` and a sequence of
  ``CoinRef``, so these tests construct both inline. Previously every test here
  depended on ``config.load()`` reading ``config.toml`` off disk, which coupled
  an adapter test to a file it does not own.

Tests deleted, and why:

* ``test_string_numerics_are_coerced_to_float`` — split into the strict-parser
  tests below. "Coerced" was the bug: the old helper coerced malformed input
  too. The surviving half (strings parse to floats) is asserted in
  ``test_string_numerics_parse_to_float``.
* ``test_snapshot_ts_is_seconds`` — replaced by
  ``test_provenance_is_per_observation_not_per_batch``. ``MarketSnapshot.ts``
  is no longer a freshness input (audit C8), so asserting its plausibility was
  asserting the wrong property.
* ``test_malformed_ohlcv_rows_are_skipped`` — replaced by
  ``test_structurally_broken_rows_are_skipped_but_an_empty_series_raises``.
  Structurally broken rows are still skipped; a series left with nothing usable
  is now an error rather than a silent empty tuple.
* The ``coin.degraded`` / ``coin.degraded_reason`` assertions throughout —
  those fields were replaced by ``CoinSnapshot.quality`` (a three-valued
  ``DataQuality``) because "degraded" could not distinguish a missing candle
  series from a corrupt price.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from pathlib import Path

import httpx
import pytest

from memetrader import market
from memetrader.market import (
    MalformedField,
    MarketDataError,
    MarketParams,
    SafetyScreenParams,
)
from memetrader.types import DataQuality, Timeframe

FIXTURES = Path(__file__).parent / "fixtures"

BONK_MINT = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
WIF_MINT = "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm"
POPCAT_MINT = "7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr"

#: Frozen read time, chosen to sit *inside* the newest bar of both captures:
#: the 5m capture's newest bar opens at 1789755000 (closes 1789755300) and the
#: 1h capture's opens at 1789754400 (closes 1789758000). So both series end in
#: an in-progress bar, which is the live case and the one the watermark exists
#: for.
NOW = 1789755100.0


@dataclasses.dataclass(frozen=True, slots=True)
class Coin:
    """A minimal ``market.CoinRef``. ``config.CoinConfig`` satisfies the same
    protocol structurally; nothing here needs the rest of a ``Config``."""

    symbol: str
    mint: str


COINS = (
    Coin("BONK", BONK_MINT),
    Coin("WIF", WIF_MINT),
    Coin("POPCAT", POPCAT_MINT),
)
BONK_ONLY = (COINS[0],)

PARAMS = MarketParams(
    dexscreener_base="https://api.dexscreener.com",
    geckoterminal_base="https://api.geckoterminal.com/api/v2",
    candles_5m=100,
    candles_1h=100,
)


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def clock(value: float = NOW):
    return lambda: value


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
        raise AssertionError(
            f"unrouted request (would have hit the network): {request.url}"
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def snap_all(client, coins=COINS, **kwargs):
    return market.snapshot(coins, PARAMS, client=client, now=clock(), **kwargs)


def ohlcv(rows: list) -> dict:
    return {"data": {"attributes": {"ohlcv_list": rows}}}


def row(ts: float, *, o=1.0, h=1.2, low=0.9, c=1.1, v=10.0) -> list:
    return [ts, o, h, low, c, v]


# ---------------------------------------------------------------------------
# Strict parsing: absent, malformed and present-and-zero are three things
# ---------------------------------------------------------------------------


def test_absent_field_is_none_and_malformed_field_raises():
    """The single most damaging habit in the old file, in one test.

    ``_as_float`` returned 0.0 for a missing value *and* for garbage. That made
    "DexScreener omitted the m5 block for this quiet pair" indistinguishable
    from "the price was exactly flat", and made ``"n/a"`` a number.
    """
    # Absent.
    assert market._opt_float(None, "x") is None

    # Present and zero — a real observation, and it stays one.
    assert market._opt_float(0, "x") == 0.0
    assert market._opt_float("0", "x") == 0.0

    # Malformed. Every one of these used to be 0.0.
    for bad in ("n/a", "", "  ", [], {}, object(), True, False):
        with pytest.raises(MalformedField):
            market._opt_float(bad, "x")

    # NaN and infinity are malformed, not missing: a missing value must be
    # None, and NaN compares false against every bound including the rejections.
    with pytest.raises(MalformedField):
        market._opt_float(float("nan"), "x")
    with pytest.raises(MalformedField):
        market._opt_float(float("inf"), "x")


def test_counts_parse_strictly_and_refuse_to_invent_a_transaction():
    assert market._opt_int(None, "n") is None
    assert market._opt_int(0, "n") == 0
    assert market._opt_int("93", "n") == 93
    assert market._opt_int(93.0, "n") == 93

    # isinstance(True, int) is True in Python, so without an explicit bool
    # check `"buys": true` would parse as one transaction.
    with pytest.raises(MalformedField):
        market._opt_int(True, "n")
    # A fractional count means the field is not what we think it is.
    with pytest.raises(MalformedField):
        market._opt_int(93.5, "n")
    with pytest.raises(MalformedField):
        market._opt_int(-1, "n")
    with pytest.raises(MalformedField):
        market._opt_int("lots", "n")


def test_string_numerics_parse_to_float(bonk_pairs_fixture=None):
    tokens = load_fixture("dexscreener_pair_selection.json")
    best = next(p for p in tokens["pairs"] if p["pairAddress"] == "3UBestRealUsdcPool")
    assert isinstance(best["priceUsd"], str)  # guard: the fixture stays stringly-typed

    with make_client(tokens=tokens) as client:
        bonk = snap_all(client, BONK_ONLY).coins["BONK"]

    assert isinstance(bonk.price_usd, float)
    assert bonk.price_usd == pytest.approx(2.943e-06)
    assert isinstance(bonk.liquidity_usd, float)
    assert bonk.volume_24h_usd == pytest.approx(1250000.5)
    assert isinstance(bonk.txns_h1.buys, int)


def test_absent_price_change_window_is_none_not_zero():
    """DexScreener omits ``priceChange.m5`` entirely on quiet pools — observed
    on 13 of 30 live pairs, including the pool we select for BONK.

    It must survive as ``None``. Collapsing it to 0.0 would say the price was
    flat over five minutes when in fact nothing was reported, and a flat reading
    is a tradeable claim in a way that silence is not.
    """
    tokens = load_fixture("dexscreener_pair_selection.json")
    best = next(p for p in tokens["pairs"] if p["pairAddress"] == "3UBestRealUsdcPool")
    assert "m5" not in best["priceChange"]

    with make_client(tokens=tokens) as client:
        bonk = snap_all(client, BONK_ONLY).coins["BONK"]

    assert bonk.price_change.m5 is None
    assert bonk.price_change.h1 == pytest.approx(0.93)


def test_absent_txn_block_gives_none_counts_and_a_none_ratio():
    """``TxnCounts.ratio`` must be None, not a neutral 1.0 computed from nothing.

    The old ``_as_int`` made an omitted block into 0 buys and 0 sells, and
    ``types.py:85-90`` turned that into a confident ratio of 1.0 — "balanced
    two-sided market" manufactured out of an absence.
    """
    tokens = load_fixture("dexscreener_pair_selection.json")
    for pair in tokens["pairs"]:
        pair.pop("txns", None)

    with make_client(tokens=tokens) as client:
        bonk = snap_all(client, BONK_ONLY).coins["BONK"]

    assert bonk.txns_m5.buys is None
    assert bonk.txns_m5.sells is None
    assert bonk.txns_m5.ratio is None
    assert bonk.txns_m5.total is None
    # One side present and one absent is still no ratio.
    assert bonk.quality is DataQuality.OK  # absence is not corruption


def test_one_missing_side_of_a_txn_block_still_yields_no_ratio():
    tokens = load_fixture("dexscreener_pair_selection.json")
    best = next(p for p in tokens["pairs"] if p["pairAddress"] == "3UBestRealUsdcPool")
    best["txns"]["h1"].pop("sells")

    with make_client(tokens=tokens) as client:
        bonk = snap_all(client, BONK_ONLY).coins["BONK"]

    assert bonk.txns_h1.buys is not None
    assert bonk.txns_h1.sells is None
    assert bonk.txns_h1.ratio is None


def test_a_malformed_field_quarantines_the_observation_rather_than_zeroing_it():
    """Quarantined data is retained — it is evidence about a source's health —
    but it is not tradeable and no feature may be computed from it."""
    tokens = load_fixture("dexscreener_pair_selection.json")
    best = next(p for p in tokens["pairs"] if p["pairAddress"] == "3UBestRealUsdcPool")
    best["volume"]["h24"] = "not a number"

    with make_client(tokens=tokens) as client:
        bonk = snap_all(client, BONK_ONLY).coins["BONK"]

    assert bonk.volume_24h_usd is None  # emphatically not 0.0
    assert bonk.quality is DataQuality.QUARANTINED
    assert "volume.h24" in bonk.quality_reason
    assert not bonk.tradeable
    # The fields that parsed cleanly are still there: quarantine is a label on
    # the observation, not a reason to throw the evidence away.
    assert bonk.price_usd == pytest.approx(2.943e-06)


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------


def test_picks_highest_liquidity_pair_not_pairs_zero():
    """The synthetic fixture is arranged so pairs[0] is a dust pool."""
    tokens = load_fixture("dexscreener_pair_selection.json")
    assert tokens["pairs"][0]["pairAddress"] == "5zDeadPoolNeverPickMe"

    with make_client(tokens=tokens) as client:
        bonk = snap_all(client, BONK_ONLY).coins["BONK"]

    assert bonk.pool.pair_address == "3UBestRealUsdcPool"
    assert bonk.liquidity_usd == 900000.0


def test_the_4900x_fake_liquidity_pool_is_still_rejected():
    """The measurement that motivated the whole selection heuristic.

    Observed live: the single highest-``liquidity.usd`` BONK pool was a Meteora
    DLMM quoted against a pump.fun token, reporting $2.4M of "liquidity" and a
    BONK price of $0.01434 — roughly 4,900x the real $0.0000029.
    ``liquidity.usd`` is derived from the quote token's own fabricated
    valuation, so a worthless quote token mints unlimited fake liquidity.
    Selecting it would poison every downstream number.
    """
    tokens = load_fixture("dexscreener_pair_selection.json")
    fake = next(p for p in tokens["pairs"] if p["pairAddress"] == "CdFakeDlmmQuotedInJunk")
    assert fake["liquidity"]["usd"] == max(p["liquidity"]["usd"] for p in tokens["pairs"])
    # The fixture's fake price really is ~4,900x the real one.
    real = next(p for p in tokens["pairs"] if p["pairAddress"] == "3UBestRealUsdcPool")
    assert float(fake["priceUsd"]) / float(real["priceUsd"]) == pytest.approx(
        4872, rel=0.01
    )

    with make_client(tokens=tokens) as client:
        bonk = snap_all(client, BONK_ONLY).coins["BONK"]

    assert bonk.pool.pair_address != "CdFakeDlmmQuotedInJunk"
    assert bonk.price_usd < 1e-4
    assert bonk.pool.trusted_quote


def test_live_capture_selects_sane_pairs_for_all_three_coins():
    """Same rule against the real captured response, where BONK's top-liquidity
    pair really is the fake one."""
    with make_client() as client:
        snap = snap_all(client)

    assert set(snap.coins) == {"BONK", "WIF", "POPCAT"}
    assert snap.coins["BONK"].price_usd == pytest.approx(2.944e-06, rel=0.01)
    assert snap.coins["WIF"].price_usd == pytest.approx(0.2043, rel=0.01)
    assert snap.coins["POPCAT"].price_usd == pytest.approx(0.04890, rel=0.01)
    for coin in snap.coins.values():
        assert coin.quality is DataQuality.OK, coin.quality_reason
        assert coin.tradeable


def test_an_untrusted_quote_token_marks_the_pool_not_just_the_snapshot():
    """If the junk pool is the *only* pool we still see it — but the price is
    structurally untrusted, and audit C8 requires that to veto entries."""
    tokens = load_fixture("dexscreener_pair_selection.json")
    tokens["pairs"] = [
        p for p in tokens["pairs"] if p["pairAddress"] == "CdFakeDlmmQuotedInJunk"
    ]
    with make_client(tokens=tokens) as client:
        bonk = snap_all(client, BONK_ONLY).coins["BONK"]

    assert bonk.pool.trusted_quote is False
    assert bonk.pool.quote_symbol == "L$L"
    assert bonk.quality is DataQuality.DEGRADED
    assert "L$L" in bonk.quality_reason
    # The structural consequence: this snapshot cannot support an entry.
    assert not bonk.tradeable


def test_missing_pair_raises_rather_than_reporting_a_zero_price():
    empty = {"schemaVersion": "1.0.0", "pairs": []}
    with (
        make_client(tokens=empty) as client,
        pytest.raises(MarketDataError, match="no tradeable pair"),
    ):
        snap_all(client)


def test_resolve_pairs_returns_pool_refs_not_bare_strings():
    """A bare address string is what let pool identity go unchecked; the caller
    needs ``trusted_quote`` and ``created_at`` to decide anything."""
    with make_client() as client:
        resolved = market.resolve_pairs(COINS, PARAMS, client=client, now=clock())

    assert resolved["BONK"].pair_address == "5zpyutJu9ee6jFymDGoK7F6S5Kczqtc9FomP3ueKuyA9"
    assert resolved["WIF"].pair_address == "EP2ib6dYdEeqD8MfE2ezHCxX3kP3K2eLKkirfPm5eyMx"
    assert resolved["POPCAT"].pair_address == "FRhB8L7Y9Qq41qZXYLtC2nw8An1RJfLLxRF2x9RwLLMo"
    assert all(ref.trusted_quote for ref in resolved.values())
    assert resolved["BONK"].quote_symbol == "SOL"


def test_pool_ref_identity_survives_a_read():
    with make_client() as client:
        bonk = snap_all(client).coins["BONK"]
    pool = bonk.pool
    assert pool.base_mint == BONK_MINT
    assert pool.quote_mint == "So11111111111111111111111111111111111111112"
    assert pool.dex_id == "orca"
    assert pool.same_pool(pool)
    assert not pool.same_pool(None)


# ---------------------------------------------------------------------------
# Timestamps and provenance
# ---------------------------------------------------------------------------


def test_dexscreener_millisecond_timestamp_lands_in_a_sane_year():
    """``pairCreatedAt`` is 1671980424000 — epoch *milliseconds*, Dec 2022.

    Left unconverted it is the year 54,977, and every staleness check that ever
    looks at it passes without complaint.
    """
    with make_client() as client:
        bonk = snap_all(client).coins["BONK"]

    created = dt.datetime.fromtimestamp(bonk.pool.created_at, tz=dt.UTC)
    assert created.year == 2022


def test_an_out_of_range_timestamp_is_malformed_not_unknown():
    """A unit or encoding change must surface as a failure, not as an absence:
    reporting it as unknown would hide the bug while still producing a snapshot."""
    with pytest.raises(MalformedField, match=r"2020\.\.2100"):
        market._ms_to_seconds(10**20, "pairCreatedAt")
    with pytest.raises(MalformedField):
        market._ms_to_seconds(-5, "pairCreatedAt")
    assert market._ms_to_seconds(None, "pairCreatedAt") is None
    # Already-in-seconds values pass through untouched.
    assert market._ms_to_seconds(1671980424, "x") == 1671980424.0
    assert market._ms_to_seconds(1671980424000, "x") == 1671980424.0


def test_geckoterminal_second_timestamps_pass_through_unscaled():
    with make_client() as client:
        bonk = snap_all(client).coins["BONK"]

    for series in (bonk.candles_5m, bonk.candles_1h):
        for candle in series.candles:
            year = dt.datetime.fromtimestamp(candle.ts, tz=dt.UTC).year
            assert 2020 <= year <= 2100


def test_provenance_is_per_observation_not_per_batch():
    """Audit C8. ``MarketSnapshot.ts`` used to be ``time.time()`` taken after
    every sequential call returned, so a snapshot looked fresh while its first
    constituent read could be minutes old.

    The DexScreener read now carries its own receive time, each candle series
    carries its own, and ``event_time`` is ``None`` where the vendor does not
    tell us — never a copy of ``receive_time``.
    """
    # A clock that jumps 800 seconds between the read and the completion of the
    # tick — a slow tick, which is the case the old batch timestamp hid.
    times = iter([100.0])
    # The candle fixtures would be future-dated against this toy clock, so this
    # test asserts the provenance plumbing on a price-only read.
    with make_client() as client:
        snap = market.snapshot(
            COINS,
            PARAMS,
            client=client,
            with_candles=False,
            now=lambda: next(times, 900.0),
        )

    bonk = snap.coins["BONK"]
    assert bonk.provenance.source == "dexscreener"
    assert bonk.provenance.receive_time == 100.0  # when the response arrived
    assert bonk.provenance.event_time is None  # the summary has no "as of"
    assert bonk.provenance.source_time is None
    # ts is the completion time and is for logging only.
    assert snap.ts == 900.0
    # The honest number: measured from the oldest constituent observation.
    assert snap.oldest_age_seconds(1000.0) == pytest.approx(900.0)
    assert snap.age_seconds(1000.0) == pytest.approx(100.0)
    assert snap.oldest_age_seconds(1000.0) > snap.age_seconds(1000.0)


def test_candle_series_provenance_is_its_own_read():
    with make_client() as client:
        bonk = snap_all(client).coins["BONK"]

    prov = bonk.candles_5m.provenance
    assert prov.source == "geckoterminal"
    assert prov.receive_time == NOW
    # event_time is the newest bar's *open*, a time at which the data was
    # demonstrably true — deliberately the open, so a series can never look
    # fresher than it is.
    assert prov.event_time == 1789755000.0
    assert prov.effective_time == 1789755000.0
    assert prov.age_seconds(NOW) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Candle hygiene
# ---------------------------------------------------------------------------


def test_candles_are_returned_oldest_first():
    """GeckoTerminal serves ohlcv_list newest-first; un-reversed, every
    indicator in signals.py computes over time-reversed data."""
    raw = load_fixture("geckoterminal_5m.json")["data"]["attributes"]["ohlcv_list"]
    assert raw[0][0] > raw[-1][0]  # guard: the capture really is descending

    with make_client() as client:
        bonk = snap_all(client).coins["BONK"]

    timestamps = [c.ts for c in bonk.candles_5m.candles]
    assert timestamps == sorted(timestamps)
    assert timestamps[-1] == float(raw[0][0])
    assert len(bonk.candles_5m.candles) == 100
    assert len(bonk.candles_1h.candles) == 100
    assert bonk.candles_5m.interval_seconds == 300.0
    assert bonk.candles_1h.interval_seconds == 3600.0
    assert bonk.candles_5m.pool_address == bonk.pool.pair_address


def test_candle_five_minute_spacing_and_no_gaps():
    with make_client() as client:
        bonk = snap_all(client).coins["BONK"]
    candles = bonk.candles_5m.candles
    deltas = {candles[i + 1].ts - candles[i].ts for i in range(len(candles) - 1)}
    assert deltas == {300.0}
    assert bonk.candles_5m.missing_intervals == 0
    assert bonk.candles_5m.complete


def test_the_in_progress_bar_is_labelled_open_and_excluded_from_closed_candles():
    """The look-ahead watermark. The newest bar opens at 1789755000 and closes
    at 1789755300; the read happens at 1789755100, i.e. inside it."""
    with make_client() as client:
        bonk = snap_all(client).coins["BONK"]

    series = bonk.candles_5m
    assert series.candles[-1].closed is False
    assert all(c.closed for c in series.candles[:-1])
    assert len(series.closed_candles) == 99
    assert series.closed_candles[-1].ts == 1789754700.0
    # The 1h series ends in an in-progress bar too (opens 1789754400).
    assert bonk.candles_1h.candles[-1].closed is False
    assert len(bonk.candles_1h.closed_candles) == 99


def test_every_bar_is_closed_once_the_read_happens_after_it():
    later = 1789755400.0  # past the close of the newest 5m bar
    with make_client() as client:
        snap = market.snapshot(BONK_ONLY, PARAMS, client=client, now=clock(later))
    series = snap.coins["BONK"].candles_5m
    assert series.candles[-1].closed is True
    assert len(series.closed_candles) == 100


def test_ohlc_invariant_violation_quarantines_the_whole_series():
    """A vendor reporting low > high has malfunctioned. Left alone it produces a
    negative range in every ATR downstream, which understates volatility exactly
    when it matters — so the series goes, not just the bar."""
    rows = [row(1789754700.0, o=1.0, h=0.5, low=1.5, c=1.0), row(1789754400.0)]
    with make_client(ohlcv_5m=ohlcv(rows), ohlcv_1h=ohlcv(rows)) as client:
        bonk = snap_all(client, BONK_ONLY).coins["BONK"]

    assert bonk.candles_5m is None
    assert bonk.quality is DataQuality.DEGRADED
    assert "candles unavailable" in bonk.quality_reason
    # The part that matters: price and liquidity survived.
    assert bonk.price_usd > 0
    assert bonk.liquidity_usd > 0


def test_close_outside_the_bar_range_is_rejected():
    rows = [row(1789754700.0, o=1.0, h=1.2, low=0.9, c=1.9)]
    with make_client(ohlcv_5m=ohlcv(rows), ohlcv_1h=ohlcv(rows)) as client:
        bonk = snap_all(client, BONK_ONLY).coins["BONK"]
    assert bonk.candles_5m is None


def test_duplicate_timestamps_are_rejected():
    """The same bar twice: there is no way to know which copy is real."""
    rows = [row(1789754700.0), row(1789754700.0, c=1.15), row(1789754400.0)]
    with make_client(ohlcv_5m=ohlcv(rows), ohlcv_1h=ohlcv(rows)) as client:
        bonk = snap_all(client, BONK_ONLY).coins["BONK"]

    assert bonk.candles_5m is None
    assert "duplicate candle timestamp" in bonk.quality_reason


def test_future_dated_bars_are_rejected():
    """A bar that opens after the read time cannot be an observation."""
    rows = [row(NOW + 600.0), row(1789754700.0)]
    with make_client(ohlcv_5m=ohlcv(rows), ohlcv_1h=ohlcv(rows)) as client:
        bonk = snap_all(client, BONK_ONLY).coins["BONK"]

    assert bonk.candles_5m is None
    assert "future-dated" in bonk.quality_reason


def test_gaps_are_counted_not_closed_up():
    """CoinGecko documents that empty intervals may be skipped, so a gap is
    expected and is information: a '20-bar' window that actually spans 26 bars
    of wall clock must be visible rather than silently compressed."""
    rows = [
        row(1789754700.0),
        # 1789754400 and 1789754100 are missing: two skipped intervals.
        row(1789753800.0),
        row(1789753500.0),
    ]
    with make_client(ohlcv_5m=ohlcv(rows), ohlcv_1h=ohlcv(rows)) as client:
        bonk = snap_all(client, BONK_ONLY).coins["BONK"]

    series = bonk.candles_5m
    assert series is not None
    assert series.missing_intervals == 2
    assert not series.complete
    assert len(series.candles) == 3
    assert series.provenance.quality is DataQuality.DEGRADED


def test_bars_off_the_interval_grid_are_rejected():
    """Spacing that is not a whole multiple of the interval means the bars are
    not on the grid every window length in signals.py assumes."""
    rows = [row(1789754700.0), row(1789754555.0)]
    with make_client(ohlcv_5m=ohlcv(rows), ohlcv_1h=ohlcv(rows)) as client:
        bonk = snap_all(client, BONK_ONLY).coins["BONK"]
    assert bonk.candles_5m is None
    assert "not a multiple" in bonk.quality_reason


def test_malformed_candle_numbers_reject_the_series():
    rows = [row(1789754700.0, c="n/a"), row(1789754400.0)]
    with make_client(ohlcv_5m=ohlcv(rows), ohlcv_1h=ohlcv(rows)) as client:
        bonk = snap_all(client, BONK_ONLY).coins["BONK"]
    assert bonk.candles_5m is None


def test_structurally_broken_rows_are_skipped_but_an_empty_series_raises():
    """A truncated or non-list row is a transport artefact, not a claim about
    the market, so it is skipped and counted. A response left with nothing
    usable is an error — the old code returned an empty tuple, which read
    downstream as "this pool has no history"."""
    broken = ohlcv([[1789754700, 1, 2], "junk", None])
    with make_client(ohlcv_5m=broken, ohlcv_1h=broken) as client:
        bonk = snap_all(client, BONK_ONLY).coins["BONK"]
    assert bonk.candles_5m is None
    assert "no usable rows" in bonk.quality_reason

    # Mixed: one good row survives and the skip is recorded as degradation.
    mixed = ohlcv(["junk", row(1789754700.0)])
    with make_client(ohlcv_5m=mixed, ohlcv_1h=mixed) as client:
        bonk = snap_all(client, BONK_ONLY).coins["BONK"]
    assert len(bonk.candles_5m.candles) == 1
    assert bonk.candles_5m.provenance.quality is DataQuality.DEGRADED


def test_geckoterminal_failure_degrades_instead_of_crashing_the_tick():
    with make_client(ohlcv_status=429) as client:
        snap = snap_all(client)

    for coin in snap.coins.values():
        assert coin.candles_5m is None
        assert coin.candles_1h is None
        assert coin.quality is DataQuality.DEGRADED
        assert "candles unavailable" in coin.quality_reason
        # The part that matters: price and liquidity survived.
        assert coin.price_usd > 0
        assert coin.liquidity_usd > 0


# ---------------------------------------------------------------------------
# Request budget
# ---------------------------------------------------------------------------


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
        raise AssertionError(
            f"unrouted request (would have hit the network): {request.url}"
        )

    return httpx.Client(transport=httpx.MockTransport(handler)), gecko


def test_price_only_snapshot_makes_no_geckoterminal_calls():
    """The absence of the call is the whole point, so it is what gets asserted.

    ``fast_tick`` marks the book off ``price_usd`` and never reads a candle,
    but it used to fetch them anyway: 2 calls per coin per minute against
    GeckoTerminal's ~30/min keyless budget, plus 1.5s of spacing between each.
    The 429s that bought landed on the *slow* tick, the one caller whose
    technicals actually depend on candles.
    """
    client, gecko = _counting_client()
    with client:
        snap = snap_all(client, with_candles=False)

    assert gecko == []
    for coin in snap.coins.values():
        assert coin.candles_5m is None
        assert coin.candles_1h is None
        # Not degraded: nothing failed, the caller asked for prices. Marking
        # this degraded would put a permanent false warning in front of every
        # consumer, which is how real warnings stop being read.
        assert coin.quality is DataQuality.OK
        assert coin.price_usd > 0


def test_the_default_snapshot_still_fetches_candles():
    """Guards the other direction: ``with_candles`` defaulting to False would
    silently strip technicals from the slow tick and nothing else would fail."""
    client, gecko = _counting_client()
    with client:
        snap = snap_all(client)

    assert len(gecko) == 2 * len(COINS)
    assert all(coin.candles_5m and coin.candles_1h for coin in snap.coins.values())


def test_too_many_mints_is_refused_rather_than_silently_truncated():
    """DexScreener caps this route at 30 addresses and truncates past it without
    erroring, so the coins past the cap would silently vanish from the read."""
    many = tuple(Coin(f"C{i}", f"mint{i}") for i in range(31))
    with (
        make_client() as client,
        pytest.raises(MarketDataError, match="exceeds DexScreener"),
    ):
        snap_all(client, many)


# ---------------------------------------------------------------------------
# Cloudflare and the misread status codes
# ---------------------------------------------------------------------------


def test_403_raises_a_named_error_and_is_never_retried():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(403, text="cloudflare")

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(MarketDataError, match="403"),
    ):
        snap_all(client)

    assert calls["n"] == 1, "a 403 must not be retried"


def test_422_is_named_as_a_query_timeout_and_never_retried():
    """Established 2026-09-19 across 12 back-to-back 200s, a cold-process
    reproduction, four spacing values and a 105s backoff: a 422 from these hosts
    is a ~3s server-side query timeout on the request range, not rate limiting.
    Backing off does not help; narrowing the range does."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            422, json={"data": None, "error": "Timeout. Maybe slow down a bit"}
        )

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(MarketDataError, match="NOT rate limiting"),
    ):
        snap_all(client)

    assert calls["n"] == 1


def test_snapshot_sends_a_browser_user_agent():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("user-agent", ""))
        if "/dex/tokens/" in request.url.path:
            return httpx.Response(200, json=load_fixture("dexscreener_tokens.json"))
        return httpx.Response(200, json=load_fixture("geckoterminal_5m.json"))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        snap_all(client)

    assert seen and all("Mozilla/5.0" in ua for ua in seen)
    assert not any("python" in ua.lower() for ua in seen)


# ---------------------------------------------------------------------------
# The data-quality screen (audit C9, scoped)
# ---------------------------------------------------------------------------


def screen_params(**overrides) -> SafetyScreenParams:
    base = {
        "min_liquidity_usd": 50_000.0,
        "min_pool_age_seconds": 7.0 * 86_400.0,
        "min_volume_24h_usd": 250_000.0,
        "min_liquidity_to_fdv": 0.0005,
        "max_snapshot_age_seconds": 900.0,
    }
    return SafetyScreenParams(**{**base, **overrides})


def test_screen_passes_a_healthy_established_pool():
    with make_client() as client:
        bonk = snap_all(client).coins["BONK"]

    verdict = market.screen(bonk, screen_params(), now=NOW)
    assert verdict.eligible, verdict.reason
    assert verdict.vetoes == ()
    assert verdict.unknowns == ()


def test_screen_vetoes_a_thin_young_untrusted_pool():
    tokens = load_fixture("dexscreener_pair_selection.json")
    tokens["pairs"] = [
        p for p in tokens["pairs"] if p["pairAddress"] == "CdFakeDlmmQuotedInJunk"
    ]
    # The fake pool was created 1784508320000 ms = 2026-07-18, but pin the clock
    # relationship explicitly rather than relying on the fixture's age.
    with make_client(tokens=tokens) as client:
        bonk = snap_all(client, BONK_ONLY).coins["BONK"]

    verdict = market.screen(bonk, screen_params(min_pool_age_seconds=10**9), now=NOW)
    assert not verdict.eligible
    joined = verdict.reason
    assert "untrusted quote token" in joined
    assert "pool is" in joined  # too young


def test_screen_fails_closed_on_an_unmeasurable_input():
    """A screen that passes on missing data is not a screen. An unknown is
    reported separately from a measured failure because 'this pool is too small'
    and 'we never learned how big it is' call for different fixes."""
    tokens = load_fixture("dexscreener_pair_selection.json")
    # Every pair, not just the best one: a pair that reports no liquidity sorts
    # last in selection, so stripping one would simply change which pool wins.
    for pair in tokens["pairs"]:
        pair.pop("liquidity", None)
        pair.pop("fdv", None)

    with make_client(tokens=tokens) as client:
        bonk = snap_all(client, BONK_ONLY).coins["BONK"]

    verdict = market.screen(bonk, screen_params(), now=NOW)
    assert not verdict.eligible
    assert any("liquidity_usd not reported" in u for u in verdict.unknowns)
    assert any("liquidity/FDV" in u for u in verdict.unknowns)
    # Unknowns are kept apart from measured failures: "this pool is too thin"
    # and "we never learned how thin it is" call for different fixes, and the
    # second must never be filed as the first.
    assert not any("liquidity" in v for v in verdict.vetoes)
    assert "not reported" in verdict.reason


def test_screen_vetoes_a_stale_observation():
    with make_client() as client:
        bonk = snap_all(client).coins["BONK"]
    verdict = market.screen(bonk, screen_params(), now=NOW + 5000.0)
    assert not verdict.eligible
    assert any("old" in v for v in verdict.vetoes)


def test_screen_names_the_on_chain_checks_it_does_not_perform():
    """Audit C9's on-chain half is deferred, not done. The gap travels with the
    answer so an operator cannot mistake this for a token-safety screen."""
    with make_client() as client:
        bonk = snap_all(client).coins["BONK"]
    verdict = market.screen(bonk, screen_params(), now=NOW)
    assert "mint_authority" in verdict.deferred
    assert "freeze_authority" in verdict.deferred
    assert "lp_ownership_and_lock" in verdict.deferred
    assert "holder_and_dev_concentration" in verdict.deferred
    assert "token_2022_extensions" in verdict.deferred


def test_screen_vetoes_a_quarantined_observation():
    tokens = load_fixture("dexscreener_pair_selection.json")
    best = next(p for p in tokens["pairs"] if p["pairAddress"] == "3UBestRealUsdcPool")
    best["fdv"] = "???"
    with make_client(tokens=tokens) as client:
        bonk = snap_all(client, BONK_ONLY).coins["BONK"]

    verdict = market.screen(bonk, screen_params(), now=NOW)
    assert not verdict.eligible
    assert any("quarantined" in v for v in verdict.vetoes)


def test_timeframe_routes_are_the_ones_geckoterminal_actually_serves():
    """The route and aggregate pair is what makes the interval known, and the
    interval is what makes gap counting and the watermark possible at all."""
    assert market._ROUTES[Timeframe.M5] == ("minute", 5, 300.0)
    assert market._ROUTES[Timeframe.H1] == ("hour", 1, 3600.0)
