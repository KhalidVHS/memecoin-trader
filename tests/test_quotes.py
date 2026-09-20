"""Offline tests for ``quotes.py``, driven by saved live Jupiter responses.

Two things this file exists to pin, both of which fail *silently* when they
break:

1. **The unit arithmetic.** A missing decimals adjustment produces a number that
   looks like a price, just off by a power of ten, and the book compounds it for
   hours. So the direction tests restate the conversion from first principles
   against a verbatim capture rather than round-tripping through the code.
2. **That a failure cannot become an executable object.** Audit C4. Every
   failure path is asserted to yield ``None`` or a ``ValuationEstimate``, and
   there is a test that walks the module's public surface and asserts no
   function can return a ``Quote`` from a broken router.

Tests carried over unchanged in intent
--------------------------------------
Both effective-price direction tests, the priceImpactPct fraction-to-percent
test, the route-label test, the decimals-from-token-search test and the two
api-key header tests all still assert exactly what they asserted before; only
the call shape changed.

Tests deleted, and why
----------------------
* ``test_jupiter_failure_falls_back_to_mid_plus_adverse_slippage_on_buy`` /
  ``..._on_sell`` / ``test_a_transport_error_degrades_rather_than_raising`` /
  ``test_a_nonsense_mid_price_degrades_rather_than_raising``. These asserted the
  existence and the arithmetic of ``_fallback``, the synthetic mid-plus-spread
  quote that the broker filled against. Audit C4 deletes the object, so the
  tests that guaranteed it would keep working are deleted with it and replaced
  by tests asserting the opposite: a router failure yields no quote at all.
* ``test_decimals_are_derived_from_the_quote_when_token_search_is_gone`` and
  ``test_decimals_derivation_tolerates_a_badly_wrong_mid``. These tested
  ``_derive_decimals``. Audit §15 forbids inferring decimals for execution — the
  inference was *usually* right, which is precisely the failure mode, since a
  wrong exponent is a silent factor-of-1000 error. Replaced by
  ``test_no_decimals_means_no_quote``.
* ``test_every_hop_pays_a_pool_fee_including_unknown_venues``. The per-hop fee
  billing was the 45x double-count; ``outAmount`` is already net of it. The
  surviving half of that test — that route labels are captured un-deduplicated
  for diagnostics — is now ``test_route_labels_are_not_deduplicated``.
* ``test_sell_input_amount_is_derived_from_the_mid_price``. SELL no longer takes
  a USD notional and divides by the mid; it takes the exact atomic quantity the
  caller holds. Replaced by ``test_sell_quotes_the_exact_held_amount``.

No network: every test routes through ``httpx.MockTransport``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import httpx
import pytest

from memetrader import config, quotes
from memetrader.ids import quote_fingerprint
from memetrader.types import Quote, Side, TokenMeta, ValidationError, ValuationEstimate

FIXTURES = Path(__file__).parent / "fixtures"

USDC_MINT = quotes.USDC_MINT
BONK_MINT = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
BONK_DECIMALS = 5  # verified live via /tokens/v2/search
BONK_MID = 2.944e-06

# The captures: BUY is 100_000000 micro-USDC in, SELL is 3396739130434 atomic
# BONK in. Quoting any other amount would make Jupiter's ExactIn `inAmount`
# disagree with the request, which `_build_quote` refuses by design.
BUY_IN_ATOMIC = 100_000_000
SELL_IN_ATOMIC = 3_396_739_130_434


def load_fixture(name: str) -> dict | list:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture(scope="module")
def cfg():
    return config.load()


@pytest.fixture(autouse=True)
def _clear_token_cache():
    """The cache is process-wide by design; leaking it across tests would let
    one test satisfy another's lookup and hide a broken failure path."""
    quotes.clear_token_cache()
    yield
    quotes.clear_token_cache()


def make_client(
    *,
    quote_status: int = 200,
    tokens_status: int = 200,
    seen: list[httpx.Request] | None = None,
    raise_on_quote: bool = False,
    mangle_quote: dict | None = None,
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
            payload = dict(buy if is_buy else sell)
            if mangle_quote is not None:
                payload.update(mangle_quote)
            return httpx.Response(200, json=payload)
        raise AssertionError(
            f"unrouted request (would have hit the network): {request.url}"
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def bonk(cfg, client) -> TokenMeta:
    meta = quotes.token_meta(cfg, BONK_MINT, client=client)
    assert meta is not None
    return meta


# ---------------------------------------------------------------------------
# Exact amounts — audit C2
# ---------------------------------------------------------------------------


def test_buy_preserves_jupiter_exact_atomic_amounts(cfg) -> None:
    """The whole of C2. The integers on the Quote must be the integers Jupiter
    sent, byte for byte, not a dollar amount reconstructed from a price."""
    raw = load_fixture("jupiter_quote_buy.json")
    with make_client() as client:
        q = quotes.quote_buy_usd(
            cfg, symbol="BONK", token=bonk(cfg, client), usd_notional=100.0, client=client
        )

    assert q is not None
    assert q.in_amount_atomic == int(raw["inAmount"]) == BUY_IN_ATOMIC
    assert q.out_amount_atomic == int(raw["outAmount"]) == 3_387_150_000_000
    assert q.min_out_amount_atomic == int(raw["otherAmountThreshold"]) == 3_370_214_250_000
    assert isinstance(q.in_amount_atomic, int)
    assert isinstance(q.out_amount_atomic, int)
    assert q.input_token.mint == USDC_MINT
    assert q.output_token.mint == BONK_MINT
    assert q.token_amount_atomic == q.out_amount_atomic
    assert q.context_slot == 448161273


def test_sell_preserves_jupiter_exact_atomic_amounts(cfg) -> None:
    raw = load_fixture("jupiter_quote_sell.json")
    with make_client() as client:
        q = quotes.quote_sell_tokens(
            cfg,
            symbol="BONK",
            token=bonk(cfg, client),
            token_amount_atomic=SELL_IN_ATOMIC,
            client=client,
        )

    assert q is not None
    assert q.in_amount_atomic == int(raw["inAmount"])
    assert q.out_amount_atomic == int(raw["outAmount"]) == 100_206_366
    assert q.min_out_amount_atomic == int(raw["otherAmountThreshold"]) == 99_705_335
    assert q.input_token.mint == BONK_MINT
    assert q.output_token.mint == USDC_MINT
    assert q.token_amount_atomic == q.in_amount_atomic


def test_min_out_is_strictly_worse_than_expected_out(cfg) -> None:
    """``otherAmountThreshold`` is the slippage-worst output. The broker fills
    at it (audit C5a), so it must never silently equal ``outAmount``."""
    with make_client() as client:
        q = quotes.quote_buy_usd(
            cfg, symbol="BONK", token=bonk(cfg, client), usd_notional=100.0, client=client
        )
    assert q is not None
    assert q.min_out_amount_atomic < q.out_amount_atomic
    shortfall_bps = (
        10_000 * (q.out_amount_atomic - q.min_out_amount_atomic) / q.out_amount_atomic
    )
    assert shortfall_bps == pytest.approx(50.0, abs=0.1)  # the 50 bps we asked for


def test_a_missing_threshold_is_not_backfilled_with_the_optimistic_amount(cfg) -> None:
    """Substituting ``outAmount`` would silently turn the conservative fill
    model into the optimistic one. Refuse the quote instead."""
    with make_client(mangle_quote={"otherAmountThreshold": None}) as client:
        q = quotes.quote_buy_usd(
            cfg, symbol="BONK", token=bonk(cfg, client), usd_notional=100.0, client=client
        )
    assert q is None


def test_an_inamount_that_is_not_what_we_asked_for_is_refused(cfg) -> None:
    """Jupiter honours ExactIn. A differing ``inAmount`` means the response is
    not the swap we requested, and filling it is C2 from the other side."""
    with make_client(mangle_quote={"inAmount": "99999999"}) as client:
        q = quotes.quote_buy_usd(
            cfg, symbol="BONK", token=bonk(cfg, client), usd_notional=100.0, client=client
        )
    assert q is None


def test_huge_atomic_amounts_survive_parsing(cfg) -> None:
    """``int(float(x))`` loses precision above 2^53, which is well inside the
    range of a 5-decimal token position. The parser must stay integral."""
    big = 2**60 + 1
    with make_client(
        mangle_quote={
            "inAmount": str(big),
            "outAmount": str(big),
            "otherAmountThreshold": str(big - 7),
        }
    ) as client:
        q = quotes.quote_sell_tokens(
            cfg,
            symbol="BONK",
            token=bonk(cfg, client),
            token_amount_atomic=big,
            client=client,
        )
    assert q is not None
    assert q.in_amount_atomic == big
    assert q.min_out_amount_atomic == big - 7


# ---------------------------------------------------------------------------
# Effective price — the number that is the P&L
# ---------------------------------------------------------------------------


def test_buy_effective_price_applies_both_token_decimals(cfg) -> None:
    """BUY is USDC(6) -> BONK(5): 100000000 base USDC in, 3387150000000 base
    BONK out. Divide base units by base units and you get 0.0000295 * 10 — the
    right order of magnitude by luck and wrong by construction."""
    raw = load_fixture("jupiter_quote_buy.json")
    usd_in = int(raw["inAmount"]) / 10**6
    tokens_out = int(raw["outAmount"]) / 10**BONK_DECIMALS
    expected = usd_in / tokens_out

    with make_client() as client:
        q = quotes.quote_buy_usd(
            cfg, symbol="BONK", token=bonk(cfg, client), usd_notional=100.0, client=client
        )

    assert q is not None
    assert q.effective_price_usd == pytest.approx(expected, rel=1e-12)
    assert q.effective_price_usd == pytest.approx(2.952346e-06, rel=1e-5)
    # A power-of-ten slip is the failure mode; pin the magnitude independently.
    assert q.effective_price_usd == pytest.approx(BONK_MID, rel=0.05)
    assert q.side is Side.BUY


def test_sell_effective_price_applies_both_token_decimals(cfg) -> None:
    """SELL is BONK(5) -> USDC(6), i.e. the decimals swap sides."""
    raw = load_fixture("jupiter_quote_sell.json")
    tokens_in = int(raw["inAmount"]) / 10**BONK_DECIMALS
    usd_out = int(raw["outAmount"]) / 10**6
    expected = usd_out / tokens_in

    with make_client() as client:
        q = quotes.quote_sell_tokens(
            cfg,
            symbol="BONK",
            token=bonk(cfg, client),
            token_amount_atomic=SELL_IN_ATOMIC,
            client=client,
        )

    assert q is not None
    assert q.effective_price_usd == pytest.approx(expected, rel=1e-12)
    assert q.effective_price_usd == pytest.approx(2.950077e-06, rel=1e-5)
    assert q.effective_price_usd == pytest.approx(BONK_MID, rel=0.05)


def test_buy_input_amount_is_the_notional_in_usdc_base_units(cfg) -> None:
    seen: list[httpx.Request] = []
    with make_client(seen=seen) as client:
        quotes.quote_buy_usd(
            cfg, symbol="BONK", token=bonk(cfg, client), usd_notional=100.0, client=client
        )

    request = next(r for r in seen if r.url.path.endswith("/swap/v1/quote"))
    assert int(request.url.params["amount"]) == 100_000_000
    assert request.url.params["inputMint"] == USDC_MINT
    assert request.url.params["outputMint"] == BONK_MINT


def test_sell_quotes_the_exact_held_amount(cfg) -> None:
    """The old code turned a USD notional into tokens via the DexScreener mid,
    so an inaccurate mid changed *how much of the position was sold*. The caller
    holds an exact integer; that integer is what gets quoted."""
    seen: list[httpx.Request] = []
    with make_client(seen=seen) as client:
        quotes.quote_sell_tokens(
            cfg,
            symbol="BONK",
            token=bonk(cfg, client),
            token_amount_atomic=SELL_IN_ATOMIC,
            client=client,
        )

    request = next(r for r in seen if r.url.path.endswith("/swap/v1/quote"))
    assert int(request.url.params["amount"]) == SELL_IN_ATOMIC
    assert request.url.params["inputMint"] == BONK_MINT
    assert request.url.params["outputMint"] == USDC_MINT


def test_usd_to_atomic_is_the_only_dollar_conversion() -> None:
    assert quotes.usd_to_atomic(100.0) == 100_000_000
    assert quotes.usd_to_atomic(0.000001) == 1
    with pytest.raises(ValidationError):
        quotes.usd_to_atomic(0.0)
    with pytest.raises(ValidationError):
        quotes.usd_to_atomic(-5.0)


# ---------------------------------------------------------------------------
# Binding — audit C3
# ---------------------------------------------------------------------------


def test_fingerprint_is_computed_from_the_swap_the_quote_describes(cfg) -> None:
    with make_client() as client:
        q = quotes.quote_buy_usd(
            cfg, symbol="BONK", token=bonk(cfg, client), usd_notional=100.0, client=client
        )

    assert q is not None
    assert q.fingerprint == quote_fingerprint(
        side="BUY",
        input_mint=USDC_MINT,
        output_mint=BONK_MINT,
        in_amount_atomic=q.in_amount_atomic,
        out_amount_atomic=q.out_amount_atomic,
        slot=q.context_slot,
    )


def test_a_different_size_produces_a_different_fingerprint(cfg) -> None:
    """The point of the fingerprint: a quote obtained for one size cannot be
    used to fill another."""
    with make_client() as client:
        a = quotes.quote_buy_usd(
            cfg, symbol="BONK", token=bonk(cfg, client), usd_notional=100.0, client=client
        )
        with make_client(mangle_quote={"inAmount": "50000000"}) as other:
            b = quotes.quote_buy_usd(
                cfg, symbol="BONK", token=bonk(cfg, other), usd_notional=50.0, client=other
            )
    assert a is not None and b is not None
    assert a.fingerprint != b.fingerprint


def test_quote_carries_an_expiry_and_a_latency(cfg) -> None:
    """Jupiter publishes no expiry, so we assert one — audit C5d needs something
    for the broker to refuse against."""
    with make_client() as client:
        q = quotes.quote_exact_in(
            cfg,
            symbol="BONK",
            side=Side.BUY,
            token=bonk(cfg, client),
            in_amount_atomic=BUY_IN_ATOMIC,
            now=1000.0,
            client=client,
        )
    assert q is not None
    assert q.requested_at == 1000.0
    assert q.received_at == 1000.0
    assert q.latency_seconds == 0.0
    assert q.expires_at == 1000.0 + quotes.DEFAULT_QUOTE_TTL_SECONDS
    assert not q.is_expired(1005.0)
    assert q.is_expired(1010.0)


# ---------------------------------------------------------------------------
# Price impact: fraction -> whole percent
# ---------------------------------------------------------------------------


def test_price_impact_fraction_is_converted_to_whole_percent(cfg) -> None:
    raw = load_fixture("jupiter_quote_buy.json")
    fraction = float(raw["priceImpactPct"])
    assert isinstance(raw["priceImpactPct"], str)
    assert 0.0 < fraction < 1.0  # guard: it is a fraction, not a percent

    with make_client() as client:
        q = quotes.quote_buy_usd(
            cfg, symbol="BONK", token=bonk(cfg, client), usd_notional=100.0, client=client
        )

    assert q is not None
    assert q.price_impact_pct == pytest.approx(fraction * 100.0)
    assert q.price_impact_pct == pytest.approx(0.04214469956918417)
    # Without the x100 this reads as 0.00042%, and risk.max_price_impact_pct
    # (3.0) would never reject anything again.
    assert q.price_impact_pct > fraction


def test_a_missing_price_impact_is_not_zero(cfg) -> None:
    """ "Missing is never zero." A 0.0 impact reads as a free fill."""
    with make_client(mangle_quote={"priceImpactPct": None}) as client:
        q = quotes.quote_buy_usd(
            cfg, symbol="BONK", token=bonk(cfg, client), usd_notional=100.0, client=client
        )
    assert q is None


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_route_labels_come_from_route_plan_swap_info(cfg) -> None:
    with make_client() as client:
        q = quotes.quote_buy_usd(
            cfg, symbol="BONK", token=bonk(cfg, client), usd_notional=100.0, client=client
        )
    assert q is not None
    # Live routes hit venues absent from config.toml ("AlphaQ", "Scorch").
    # Unknown is not an error — the labels are diagnostic, not billable.
    assert q.route_labels == ("AlphaQ", "Scorch")


def test_route_labels_are_not_deduplicated(cfg) -> None:
    """A route that touches the same venue twice is a real thing to see. It is
    no longer a thing to *bill* — that per-hop billing was the 45x
    double-count, since outAmount is already net of every hop's fee."""
    doubled = load_fixture("jupiter_quote_buy.json")["routePlan"]
    doubled = [doubled[0], doubled[0], doubled[1]]
    with make_client(mangle_quote={"routePlan": doubled}) as client:
        q = quotes.quote_buy_usd(
            cfg, symbol="BONK", token=bonk(cfg, client), usd_notional=100.0, client=client
        )
    assert q is not None
    assert q.route_labels == ("AlphaQ", "AlphaQ", "Scorch")


# ---------------------------------------------------------------------------
# Decimals — audit §15
# ---------------------------------------------------------------------------


def test_decimals_come_from_the_jupiter_token_search(cfg) -> None:
    with make_client() as client:
        meta = quotes.token_meta(cfg, BONK_MINT, client=client)
    assert meta is not None
    assert meta.decimals == BONK_DECIMALS
    assert meta.verified is True
    assert "jupiter" in meta.source


def test_no_decimals_means_no_quote(cfg) -> None:
    """The token endpoint is down. The old code derived the exponent from a
    probe quote and the DexScreener mid and cached the guess; audit §15 forbids
    it, because a wrong exponent is a silent factor-of-1000 error that arrives
    looking like a plausible price. No decimals, no trade."""
    with make_client(tokens_status=404) as client:
        assert quotes.token_meta(cfg, BONK_MINT, client=client) is None
        mark = quotes.mark_route(
            cfg,
            symbol="BONK",
            mint=BONK_MINT,
            quantity_atomic=SELL_IN_ATOMIC,
            client=client,
        )
    assert isinstance(mark, ValuationEstimate)
    assert "decimals" in mark.reason


def test_unverified_decimals_cannot_reach_an_executable_quote(cfg) -> None:
    """Not a market condition — a programming error, so it raises."""
    inferred = TokenMeta(
        mint=BONK_MINT, decimals=5, source="derived:mid-price", verified=False
    )
    with make_client() as client, pytest.raises(ValidationError, match="unverified"):
        quotes.quote_exact_in(
            cfg,
            symbol="BONK",
            side=Side.BUY,
            token=inferred,
            in_amount_atomic=BUY_IN_ATOMIC,
            client=client,
        )


def test_absurd_decimals_from_the_vendor_are_refused(cfg) -> None:
    info = load_fixture("jupiter_token_info.json")
    broken = [
        dict(entry, decimals=42) if entry["id"] == BONK_MINT else entry for entry in info
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=[t for t in broken if t["id"] in request.url.params.get("query", "")]
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert quotes.token_meta(cfg, BONK_MINT, client=client) is None


def test_only_verified_metadata_is_cached(cfg) -> None:
    with make_client(tokens_status=500) as client:
        assert quotes.token_meta(cfg, BONK_MINT, client=client) is None
    assert BONK_MINT not in quotes._TOKEN_META_CACHE


# ---------------------------------------------------------------------------
# No degraded route can execute — audit C4, the Phase 1 exit gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"quote_status": 500},
        {"quote_status": 404},
        {"raise_on_quote": True},
        {"mangle_quote": {"outAmount": None}},
        {"mangle_quote": {"outAmount": "0"}},
        {"mangle_quote": {"outAmount": "not-a-number"}},
        {"mangle_quote": {"otherAmountThreshold": "99999999999999999999"}},
    ],
)
def test_no_router_failure_can_produce_an_executable_quote(cfg, kwargs) -> None:
    """The heart of C4. The old ``_fallback`` synthesised a mid-plus-spread
    quote for every one of these conditions and the broker filled against it, so
    a Jupiter outage manufactured trades. Every one of them must now yield
    nothing at all."""
    with make_client(**kwargs) as client:
        token = bonk(cfg, client)
        assert (
            quotes.quote_buy_usd(
                cfg, symbol="BONK", token=token, usd_notional=100.0, client=client
            )
            is None
        )
        assert (
            quotes.quote_sell_tokens(
                cfg,
                symbol="BONK",
                token=token,
                token_amount_atomic=SELL_IN_ATOMIC,
                client=client,
            )
            is None
        )


def test_a_zero_or_negative_request_is_not_a_quote(cfg) -> None:
    with make_client() as client:
        token = bonk(cfg, client)
        assert (
            quotes.quote_buy_usd(
                cfg, symbol="BONK", token=token, usd_notional=0.0, client=client
            )
            is None
        )
        assert (
            quotes.quote_buy_usd(
                cfg, symbol="BONK", token=token, usd_notional=-1.0, client=client
            )
            is None
        )
        assert (
            quotes.quote_sell_tokens(
                cfg, symbol="BONK", token=token, token_amount_atomic=0, client=client
            )
            is None
        )
        with pytest.raises(ValidationError):
            quotes.quote_sell_tokens(
                cfg, symbol="BONK", token=token, token_amount_atomic=-1, client=client
            )


# ---------------------------------------------------------------------------
# The marking path
# ---------------------------------------------------------------------------


def test_mark_route_returns_a_real_exit_route_when_one_exists(cfg) -> None:
    with make_client() as client:
        mark = quotes.mark_route(
            cfg,
            symbol="BONK",
            mint=BONK_MINT,
            quantity_atomic=SELL_IN_ATOMIC,
            client=client,
        )
    assert isinstance(mark, Quote)
    assert mark.side is Side.SELL
    assert mark.in_amount_atomic == SELL_IN_ATOMIC


def test_mark_route_degrades_to_a_non_executable_estimate(cfg) -> None:
    with make_client(quote_status=503) as client:
        mark = quotes.mark_route(
            cfg,
            symbol="BONK",
            mint=BONK_MINT,
            quantity_atomic=SELL_IN_ATOMIC,
            mid_price_usd=BONK_MID,
            client=client,
        )
    assert isinstance(mark, ValuationEstimate)
    assert not isinstance(mark, Quote)
    assert mark.mid_price_usd == BONK_MID
    assert mark.haircut_pct > 0
    # A degraded valuation must never look better than a real one, or an outage
    # becomes a buy signal.
    assert mark.conservative_price_usd < BONK_MID


def test_an_estimate_with_no_mid_is_unmarkable_not_zero(cfg) -> None:
    with make_client(quote_status=503) as client:
        mark = quotes.mark_route(
            cfg, symbol="BONK", mint=BONK_MINT, quantity_atomic=1, client=client
        )
    assert isinstance(mark, ValuationEstimate)
    assert mark.mid_price_usd is None
    assert mark.conservative_price_usd is None


def test_nothing_held_marks_as_an_estimate_not_a_zero_quote(cfg) -> None:
    with make_client() as client:
        mark = quotes.mark_route(
            cfg, symbol="BONK", mint=BONK_MINT, quantity_atomic=0, client=client
        )
    assert isinstance(mark, ValuationEstimate)
    assert "nothing held" in mark.reason


def test_the_outage_haircut_is_at_least_the_adverse_move_assumption(cfg) -> None:
    assert quotes.outage_haircut_pct(cfg) >= cfg.execution.slippage_bps_fallback / 100.0


# ---------------------------------------------------------------------------
# Auth header
# ---------------------------------------------------------------------------


def test_api_key_header_is_omitted_when_no_key_is_configured(cfg) -> None:
    assert cfg.data.jupiter_api_key is None
    seen: list[httpx.Request] = []
    with make_client(seen=seen) as client:
        quotes.quote_buy_usd(
            cfg, symbol="BONK", token=bonk(cfg, client), usd_notional=100.0, client=client
        )
    assert seen
    assert all("x-api-key" not in r.headers for r in seen)


def test_api_key_header_is_sent_when_a_key_is_configured(cfg) -> None:
    keyed = dataclasses.replace(
        cfg, data=dataclasses.replace(cfg.data, jupiter_api_key="secret-key")
    )
    assert keyed.data.jupiter_url_base == keyed.data.jupiter_base_keyed

    seen: list[httpx.Request] = []
    with make_client(seen=seen) as client:
        quotes.quote_buy_usd(
            keyed,
            symbol="BONK",
            token=bonk(keyed, client),
            usd_notional=100.0,
            client=client,
        )
    assert seen
    assert all(r.headers.get("x-api-key") == "secret-key" for r in seen)
    assert all(r.url.host == "api.jup.ag" for r in seen)
