"""Offline tests for ``quotes.py``, driven by saved live Jupiter responses.

The effective-price arithmetic here *is* the P&L, and it fails silently: a
missing decimals adjustment produces a number that looks like a price, just off
by a power of ten. So the two direction tests restate the conversion from first
principles against a verbatim capture rather than trusting a round-trip.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import httpx
import pytest

from memetrader import config, quotes
from memetrader.types import Side

FIXTURES = Path(__file__).parent / "fixtures"

USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
BONK_MINT = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
BONK_DECIMALS = 5  # verified live via /tokens/v2/search
BONK_MID = 2.944e-06


def load_fixture(name: str) -> dict | list:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture(scope="module")
def cfg():
    return config.load()


@pytest.fixture(autouse=True)
def _clear_decimals_cache():
    """The cache is process-wide by design; leaking it across tests would let
    one test satisfy another's lookup and hide a broken fallback path."""
    quotes._DECIMALS_CACHE.clear()
    yield
    quotes._DECIMALS_CACHE.clear()


def make_client(
    *,
    quote_status: int = 200,
    tokens_status: int = 200,
    seen: list[httpx.Request] | None = None,
    raise_on_quote: bool = False,
) -> httpx.Client:
    buy = load_fixture("jupiter_quote_buy.json")
    sell = load_fixture("jupiter_quote_sell.json")
    token_info = load_fixture("jupiter_token_info.json")

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        path = request.url.path
        if path.endswith("/tokens/v2/search"):
            if tokens_status != 200:
                return httpx.Response(tokens_status, text="gone")
            query = request.url.params.get("query", "")
            return httpx.Response(
                200, json=[t for t in token_info if t["id"] in query.split(",")]
            )
        if path.endswith("/swap/v1/quote"):
            if raise_on_quote:
                raise httpx.ConnectError("boom", request=request)
            if quote_status != 200:
                return httpx.Response(quote_status, text="no route")
            is_buy = request.url.params.get("inputMint") == USDC_MINT
            return httpx.Response(200, json=buy if is_buy else sell)
        raise AssertionError(f"unrouted request (would have hit the network): {request.url}")

    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# Effective price — the number that is the P&L
# ---------------------------------------------------------------------------


def test_buy_effective_price_applies_both_token_decimals(cfg):
    """BUY is USDC(6) -> BONK(5): 100000000 base USDC in, 3387150000000 base
    BONK out. Divide base units by base units and you get 0.0000295 * 10 — the
    right order of magnitude by luck and wrong by construction."""
    raw = load_fixture("jupiter_quote_buy.json")
    usd_in = int(raw["inAmount"]) / 10**6
    tokens_out = int(raw["outAmount"]) / 10**BONK_DECIMALS
    expected = usd_in / tokens_out

    with make_client() as client:
        quote = quotes.fill_quote(
            cfg, "BONK", BONK_MINT, Side.BUY, 100.0, mid_price_usd=BONK_MID, client=client
        )

    assert not quote.degraded, quote.degraded_reason
    assert quote.price_usd == pytest.approx(expected, rel=1e-12)
    assert quote.price_usd == pytest.approx(2.952346e-06, rel=1e-5)
    # A power-of-ten slip is the failure mode; pin the magnitude independently.
    assert quote.price_usd == pytest.approx(BONK_MID, rel=0.05)
    assert quote.side is Side.BUY
    assert quote.usd_notional == 100.0


def test_sell_effective_price_applies_both_token_decimals(cfg):
    """SELL is BONK(5) -> USDC(6), i.e. the decimals swap sides."""
    raw = load_fixture("jupiter_quote_sell.json")
    tokens_in = int(raw["inAmount"]) / 10**BONK_DECIMALS
    usd_out = int(raw["outAmount"]) / 10**6
    expected = usd_out / tokens_in

    with make_client() as client:
        quote = quotes.fill_quote(
            cfg, "BONK", BONK_MINT, Side.SELL, 100.0, mid_price_usd=BONK_MID, client=client
        )

    assert not quote.degraded, quote.degraded_reason
    assert quote.price_usd == pytest.approx(expected, rel=1e-12)
    assert quote.price_usd == pytest.approx(2.950077e-06, rel=1e-5)
    assert quote.price_usd == pytest.approx(BONK_MID, rel=0.05)


def test_sell_input_amount_is_derived_from_the_mid_price(cfg):
    """A USD notional has to become token base units before Jupiter can price
    it, and the only available conversion is the DexScreener mid."""
    seen: list[httpx.Request] = []
    with make_client(seen=seen) as client:
        quotes.fill_quote(
            cfg, "BONK", BONK_MINT, Side.SELL, 250.0, mid_price_usd=BONK_MID, client=client
        )

    quote_request = next(r for r in seen if r.url.path.endswith("/swap/v1/quote"))
    assert quote_request.url.params["inputMint"] == BONK_MINT
    assert quote_request.url.params["outputMint"] == USDC_MINT
    assert int(quote_request.url.params["amount"]) == int(
        250.0 / BONK_MID * 10**BONK_DECIMALS
    )


def test_buy_input_amount_is_the_notional_in_usdc_base_units(cfg):
    seen: list[httpx.Request] = []
    with make_client(seen=seen) as client:
        quotes.fill_quote(
            cfg, "BONK", BONK_MINT, Side.BUY, 100.0, mid_price_usd=BONK_MID, client=client
        )

    quote_request = next(r for r in seen if r.url.path.endswith("/swap/v1/quote"))
    assert int(quote_request.url.params["amount"]) == 100_000_000


# ---------------------------------------------------------------------------
# Price impact: fraction -> whole percent
# ---------------------------------------------------------------------------


def test_price_impact_fraction_is_converted_to_whole_percent(cfg):
    raw = load_fixture("jupiter_quote_buy.json")
    fraction = float(raw["priceImpactPct"])
    assert isinstance(raw["priceImpactPct"], str)
    assert 0.0 < fraction < 1.0  # guard: it is a fraction, not a percent

    with make_client() as client:
        quote = quotes.fill_quote(
            cfg, "BONK", BONK_MINT, Side.BUY, 100.0, mid_price_usd=BONK_MID, client=client
        )

    assert quote.price_impact_pct == pytest.approx(fraction * 100.0)
    assert quote.price_impact_pct == pytest.approx(0.04214469956918417)
    # Without the x100 this reads as 0.00042%, and risk.max_price_impact_pct
    # (3.0) would never reject anything again.
    assert quote.price_impact_pct > fraction


# ---------------------------------------------------------------------------
# Routing and fees
# ---------------------------------------------------------------------------


def test_route_labels_come_from_route_plan_swap_info(cfg):
    with make_client() as client:
        quote = quotes.fill_quote(
            cfg, "BONK", BONK_MINT, Side.BUY, 100.0, mid_price_usd=BONK_MID, client=client
        )

    assert quote.route_labels == ("AlphaQ", "Scorch")


def test_every_hop_pays_a_pool_fee_including_unknown_venues(cfg):
    """Live routes hit venues absent from config.toml ("AlphaQ", "Scorch").
    Unknown must mean default_pool_fee_pct, never free."""
    with make_client() as client:
        quote = quotes.fill_quote(
            cfg, "BONK", BONK_MINT, Side.BUY, 100.0, mid_price_usd=BONK_MID, client=client
        )

    assert len(quote.route_labels) == 2
    assert quote.pool_fee_pct == pytest.approx(2 * cfg.execution.default_pool_fee_pct)
    assert quote.pool_fee_pct == cfg.execution.fee_pct_for(quote.route_labels)


# ---------------------------------------------------------------------------
# Decimals
# ---------------------------------------------------------------------------


def test_decimals_come_from_the_jupiter_token_search(cfg):
    with make_client() as client:
        quotes.fill_quote(
            cfg, "BONK", BONK_MINT, Side.BUY, 100.0, mid_price_usd=BONK_MID, client=client
        )
    assert quotes._DECIMALS_CACHE[BONK_MINT] == BONK_DECIMALS


def test_decimals_are_derived_from_the_quote_when_token_search_is_gone(cfg):
    """``/tokens/v1/token/{mint}`` already 404s; assume the v2 route can go the
    same way. Decimals are a power of ten, so a rough mid pins them exactly."""
    with make_client(tokens_status=404) as client:
        quote = quotes.fill_quote(
            cfg, "BONK", BONK_MINT, Side.SELL, 100.0, mid_price_usd=BONK_MID, client=client
        )

    assert quotes._DECIMALS_CACHE[BONK_MINT] == BONK_DECIMALS
    assert not quote.degraded, quote.degraded_reason
    assert quote.price_usd == pytest.approx(BONK_MID, rel=0.05)


def test_decimals_derivation_tolerates_a_badly_wrong_mid(cfg):
    """Being 3x off moves log10 by 0.48 — still short of the 0.5 that would
    round to the wrong exponent."""
    assert quotes._derive_decimals(base_units=3387150000000, ui_units=33871500) == 5
    assert quotes._derive_decimals(base_units=3387150000000, ui_units=33871500 * 3) == 5
    assert quotes._derive_decimals(base_units=3387150000000, ui_units=33871500 / 3) == 5
    assert quotes._derive_decimals(base_units=0, ui_units=1) is None
    assert quotes._derive_decimals(base_units=1, ui_units=0) is None


# ---------------------------------------------------------------------------
# The degraded path
# ---------------------------------------------------------------------------


def test_jupiter_failure_falls_back_to_mid_plus_adverse_slippage_on_buy(cfg):
    with make_client(quote_status=500) as client:
        quote = quotes.fill_quote(
            cfg, "BONK", BONK_MINT, Side.BUY, 100.0, mid_price_usd=BONK_MID, client=client
        )

    adjustment = cfg.execution.slippage_bps_fallback / 10_000.0
    assert quote.degraded
    assert quote.degraded_reason
    assert quote.price_usd == pytest.approx(BONK_MID * (1 + adjustment))
    assert quote.price_usd > BONK_MID, "a BUY fallback must not flatter the fill"
    assert quote.route_labels == ()
    assert quote.pool_fee_pct == pytest.approx(cfg.execution.default_pool_fee_pct)
    assert quote.price_impact_pct == pytest.approx(
        cfg.execution.slippage_bps_fallback / 100.0
    )


def test_jupiter_failure_falls_back_to_mid_minus_adverse_slippage_on_sell(cfg):
    with make_client(quote_status=500) as client:
        quote = quotes.fill_quote(
            cfg, "BONK", BONK_MINT, Side.SELL, 100.0, mid_price_usd=BONK_MID, client=client
        )

    adjustment = cfg.execution.slippage_bps_fallback / 10_000.0
    assert quote.degraded
    assert quote.price_usd == pytest.approx(BONK_MID * (1 - adjustment))
    assert quote.price_usd < BONK_MID, "a SELL fallback must not flatter the fill"


def test_a_transport_error_degrades_rather_than_raising(cfg):
    with make_client(raise_on_quote=True) as client:
        quote = quotes.fill_quote(
            cfg, "BONK", BONK_MINT, Side.BUY, 100.0, mid_price_usd=BONK_MID, client=client
        )
    assert quote.degraded
    assert quote.price_usd > 0


def test_a_nonsense_mid_price_degrades_rather_than_raising(cfg):
    with make_client() as client:
        quote = quotes.fill_quote(
            cfg, "BONK", BONK_MINT, Side.SELL, 100.0, mid_price_usd=0.0, client=client
        )
    assert quote.degraded
    assert quote.price_usd == 0.0
    assert "mid_price_usd" in quote.degraded_reason


def test_unknown_decimals_degrade_rather_than_guessing(cfg):
    """Token search gone *and* no route to probe with: there is no honest way
    to scale base units, so do not pretend."""
    with make_client(tokens_status=404, quote_status=500) as client:
        quote = quotes.fill_quote(
            cfg, "BONK", BONK_MINT, Side.SELL, 100.0, mid_price_usd=BONK_MID, client=client
        )
    assert quote.degraded
    assert "decimals" in quote.degraded_reason


# ---------------------------------------------------------------------------
# Auth header
# ---------------------------------------------------------------------------


def test_api_key_header_is_omitted_when_no_key_is_configured(cfg):
    assert cfg.data.jupiter_api_key is None
    seen: list[httpx.Request] = []
    with make_client(seen=seen) as client:
        quotes.fill_quote(
            cfg, "BONK", BONK_MINT, Side.BUY, 100.0, mid_price_usd=BONK_MID, client=client
        )
    assert seen
    assert all("x-api-key" not in r.headers for r in seen)


def test_api_key_header_is_sent_when_a_key_is_configured(cfg):
    keyed = dataclasses.replace(
        cfg, data=dataclasses.replace(cfg.data, jupiter_api_key="secret-key")
    )
    assert keyed.data.jupiter_url_base == keyed.data.jupiter_base_keyed

    seen: list[httpx.Request] = []
    with make_client(seen=seen) as client:
        quotes.fill_quote(
            keyed, "BONK", BONK_MINT, Side.BUY, 100.0, mid_price_usd=BONK_MID, client=client
        )
    assert seen
    assert all(r.headers.get("x-api-key") == "secret-key" for r in seen)
    assert all(r.url.host == "api.jup.ag" for r in seen)
