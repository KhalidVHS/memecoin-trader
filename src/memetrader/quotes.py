"""Jupiter adapter: the exact swap a given input amount would actually get.

A DexScreener mid price is what the last trade printed at. It is not what you
would get. For an asset class where a $500 order can move the pool, the gap
between the two is the entire difference between a backtest that looks good and
one that is true — so every trade is priced by asking Jupiter for a real route
at the exact size.

What the adversarial audit changed here
=======================================

**C2 — the exact amounts are the fact, dollars are a rendering.** The old
``fill_quote`` threw away Jupiter's ``inAmount``/``outAmount`` and returned a
dollar notional plus an effective price. The broker then set ``filled_usd`` to
the *requested* dollars and back-derived a quantity as ``filled_usd /
price_usd``. Worked example from the audit: $100 requested at a $1 mid, Jupiter
quotes 100 tokens for $95, and the broker records $100 of proceeds and 105.263
tokens. Neither number describes a swap that was ever offered. This module now
preserves the integer atomic amounts verbatim onto :class:`~.types.Quote` and
the broker consumes *those*; nothing anywhere re-derives a quantity from
dollars.

**C3 — a quote is bound to the size it was obtained for.** Risk used to be able
to shrink an order after it had been quoted, and the broker would then execute
the reduced notional against the original-size quote. Every ``Quote`` built here
carries a :func:`~.ids.quote_fingerprint` over (side, both mints, both atomic
amounts, context slot). The broker recomputes it from the quote it was handed
and refuses on mismatch. Binding, not trust.

**C4 — a route we could not obtain is not a worse route, it is no route.** The
old ``_fallback`` synthesised a quote from the DexScreener mid plus a fixed
slippage assumption, flagged it ``degraded``, and the broker filled against it.
That meant a Jupiter outage — precisely when a mid-price fiction is least
credible — *manufactured trades*. There is no fallback quote any more. The
executable functions return ``Quote | None``; ``None`` means no route was
obtained and there is nothing to execute. Failure has one other place to go:
:func:`mark_route` returns ``Quote | ValuationEstimate`` for the *marking* path,
and ``ValuationEstimate`` is structurally unable to reach ``place_order``
because the broker's signature does not accept it.

**C5 / §15 — decimals are looked up, never inferred.** ``_derive_decimals`` is
gone. It recovered a token's exponent from a base-unit/UI-unit pair using the
DexScreener mid, and the reasoning it rested on was sound as far as it went
(decimals are an integer power of ten, so being 3x off moves ``log10`` by 0.48,
short of the 0.5 that rounds wrong). But the audit's objection is not about the
error rate, it is about the failure *mode*: a wrong exponent is a silent
factor-of-1000 error in every subsequent quantity, and it arrives looking like a
plausible price. Decimals for execution now come only from Jupiter's token
endpoint and travel as :class:`~.types.TokenMeta` with ``verified=True``. An
unverifiable mint yields no quote at all.

Facts about the venue that were expensive to learn and must not be re-lost
==========================================================================

* **Jupiter's ``outAmount`` is already net of every hop's pool fee.** It is the
  number of tokens the pools actually send you, so slippage, price impact and
  each AMM fee are deducted *inside* the route. Charging ``pool_fee_pct`` on top
  double-counts. The live evidence: BONK quoted at 2.9636e-6 to buy and 2.9623e-6
  to sell, a 4.4 bp round trip, where a 2-hop route billed at 0.25% per hop would
  imply 200 bp — 45x the observed spread. Only gas/priority fee is additive.
  ``route_labels`` is still carried, un-deduplicated and in hop order, because it
  is *diagnostic* (a route that touches the same venue twice is a real thing to
  see), not because anyone should bill it.
* **``priceImpactPct`` is a fraction string in 0..1.** ``"0.00042..."`` is
  0.042%. ``types.py`` wants whole percent, so it is multiplied by 100 at this
  boundary. Forget the x100 and ``risk.max_price_impact_pct`` (3.0) never
  rejects anything again, forever.
* **``otherAmountThreshold`` is the slippage-worst output** at the
  ``slippageBps`` we asked for, and it is what a conservative simulation and any
  real submission must assume. It lands on ``Quote.min_out_amount_atomic``.
* **``/tokens/v1/token/{mint}`` 404s.** ``/tokens/v2/search?query={mint}`` is the
  live route and takes a comma-separated list, so batching stays cheap.
* **Jupiter's keyed host rejects an unexpected ``x-api-key``** and the lite host
  ignores it, so the header is sent only when a key actually exists.
* Observed live route labels include venues absent from ``config.toml``
  ("AlphaQ", "Scorch", "Byreal"). Unknown venues are not an error.

Public API
==========

``token_meta(cfg, mint, *, client=None) -> TokenMeta | None``
    Authoritative decimals, or ``None``. Never guesses.

``quote_exact_in(cfg, *, symbol, side, token, in_amount_atomic, ...) -> Quote | None``
    The only executable path. One Jupiter call at an exact input amount.

``quote_buy_usd(cfg, *, symbol, token, usd_notional, ...) -> Quote | None``
``quote_sell_tokens(cfg, *, symbol, token, token_amount_atomic, ...) -> Quote | None``
    Thin, direction-specific wrappers that do the unit conversion once, in one
    place, so no caller has to remember which leg carries which decimals.

``mark_route(cfg, *, symbol, mint, quantity_atomic, ...) -> Quote | ValuationEstimate``
    The marking path. Prefers a real exit route; degrades to a haircut mid.
    Never returns something the broker would accept.

Nothing in this module raises for a network or vendor failure — a tick that
cannot quote must still be able to mark its book and escalate. It *does* raise
``ValidationError`` for a programming error (a negative amount, unverified
decimals passed to an executable path), because that is a bug, not weather.
"""

from __future__ import annotations

import time

import httpx

from .config import Config
from .http import make_client
from .ids import quote_fingerprint
from .types import (
    Quote,
    Side,
    TokenMeta,
    ValidationError,
    ValuationEstimate,
    atomic,
)

# USDC is the unit of account for the whole book, and its 6 decimals are a fixed
# property of the mint — not something worth a network round-trip, and not
# something an outage may make unavailable.
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDC_DECIMALS = 6
USDC = TokenMeta(
    mint=USDC_MINT,
    decimals=USDC_DECIMALS,
    source="constant:usdc-spl-mint",
    verified=True,
)

_HEADERS = {"Accept": "application/json"}

#: How long a route stays executable. Jupiter does not publish an expiry, so we
#: assert one: a quote is a statement about the pools at ``contextSlot``, and
#: Solana slots are ~400ms. Ten seconds is roughly 25 slots — long enough for a
#: risk check and a submission, short enough that the pool state it describes is
#: still recognisable. ``Quote.expires_at`` is set from this and the broker
#: refuses an expired quote (audit C5d).
DEFAULT_QUOTE_TTL_SECONDS = 10.0

# Token decimals never change for a live mint, so one lookup per process is
# plenty. Module-level rather than on Config because it is a cache, not a
# setting, and Config is frozen. Only *verified* metadata is ever cached — a
# guess must never become a cache hit for something that sizes a swap.
_TOKEN_META_CACHE: dict[str, TokenMeta] = {}


def _new_client(cfg: Config) -> httpx.Client:
    # See http.make_client — certifi's bundle fails behind a TLS-inspecting
    # proxy (this machine sits behind Zscaler) and every call dies with
    # CERTIFICATE_VERIFY_FAILED. Never construct a bare httpx.Client here.
    return make_client(cfg.data.http_timeout_seconds, _HEADERS)


def _headers(cfg: Config) -> dict[str, str]:
    """Jupiter's keyed host rejects an unexpected ``x-api-key``, and the lite
    host ignores it — so the header is sent only when a key actually exists."""
    headers = dict(_HEADERS)
    if cfg.data.jupiter_api_key:
        headers["x-api-key"] = cfg.data.jupiter_api_key
    return headers


def _as_int(value: object) -> int | None:
    """Parse an integer atomic amount, or return ``None``.

    "Missing is never zero": a malformed ``outAmount`` is not a swap for nothing
    tokens, it is an unusable response, and the caller must be able to tell the
    difference. Jupiter sends these as decimal *strings*; ``int(float(x))``
    would quietly lose precision above 2^53, which is well inside the range of a
    5-decimal token position.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except TypeError, ValueError:
        return None


def _as_float_or_none(value: object) -> float | None:
    # `bool` first, because it is an `int` and `float(True)` is 1.0 — a JSON
    # `true` arriving where a price belongs is malformed, not one dollar.
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        return None


# ---------------------------------------------------------------------------
# Token metadata
# ---------------------------------------------------------------------------


def token_meta(
    cfg: Config,
    mint: str,
    *,
    client: httpx.Client | None = None,
) -> TokenMeta | None:
    """Authoritative decimals for ``mint``, or ``None`` if we could not get them.

    There is deliberately no third outcome. The old code had one — derive the
    exponent from a probe quote and a DexScreener mid — and audit §15 removes
    it: decimals must never be inferred for execution, because an inferred value
    that is wrong is a silent factor-of-1000 error in every quantity downstream,
    and it does not look like an error. If the token endpoint is down, this
    returns ``None``, the quote functions return ``None``, and the tick trades
    nothing rather than trading a guess.

    ``/tokens/v1/token/{mint}`` — the endpoint most docs still reference — now
    404s. ``/tokens/v2/search?query={mint}`` is the live route and accepts a
    comma-separated list, so this stays cheap if it ever needs batching.
    """
    cached = _TOKEN_META_CACHE.get(mint)
    if cached is not None:
        return cached

    owned = client is None
    client = client or _new_client(cfg)
    try:
        response = client.get(
            f"{cfg.data.jupiter_url_base.rstrip('/')}/tokens/v2/search",
            params={"query": mint},
            headers=_headers(cfg),
        )
        if response.status_code >= 400:
            return None
        payload = response.json()
    except httpx.HTTPError, ValueError, OSError:
        return None
    finally:
        if owned:
            client.close()

    for entry in payload or []:
        if not isinstance(entry, dict) or entry.get("id") != mint:
            continue
        decimals = entry.get("decimals")
        if not isinstance(decimals, int) or isinstance(decimals, bool):
            continue
        try:
            meta = TokenMeta(
                mint=mint,
                decimals=decimals,
                source="jupiter:/tokens/v2/search",
                verified=True,
            )
        except ValidationError:
            # A decimals outside 0..18 means the endpoint is returning something
            # other than SPL metadata. Refusing is the whole point.
            return None
        _TOKEN_META_CACHE[mint] = meta
        return meta
    return None


def clear_token_cache() -> None:
    """Drop the process-wide metadata cache. For tests and for a long-lived
    process that wants to re-verify after a vendor incident."""
    _TOKEN_META_CACHE.clear()


# ---------------------------------------------------------------------------
# Jupiter routing
# ---------------------------------------------------------------------------


def _request_quote(
    cfg: Config,
    client: httpx.Client,
    *,
    input_mint: str,
    output_mint: str,
    amount: int,
    slippage_bps: int,
) -> dict | None:
    """One ``/swap/v1/quote`` call. Returns ``None`` on any failure.

    Every failure mode collapses to ``None`` on purpose: a 4xx, a 5xx, a
    connection reset and a body that is not JSON are all "we did not obtain a
    route", and the only honest thing to do with that is not trade. The old code
    distinguished them so it could write a prettier ``degraded_reason`` onto a
    synthetic quote — which is exactly the object C4 deletes.
    """
    if amount <= 0:
        return None
    try:
        response = client.get(
            f"{cfg.data.jupiter_url_base.rstrip('/')}/swap/v1/quote",
            params={
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": amount,
                "slippageBps": slippage_bps,
            },
            headers=_headers(cfg),
        )
        if response.status_code >= 400:
            return None
        payload = response.json()
    except httpx.HTTPError, ValueError, OSError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _route_labels(payload: dict) -> tuple[str, ...]:
    """Labels for every hop, in order and **not** deduplicated.

    Kept for diagnostics only. It used to feed ``ExecutionConfig.fee_pct_for``,
    and that is the 45x double-count described in the module docstring: the
    per-hop AMM fee is already inside ``outAmount``. A route that touches the
    same venue twice is still worth *seeing*, so the duplicate is preserved.
    """
    labels: list[str] = []
    for hop in payload.get("routePlan") or []:
        label = ((hop or {}).get("swapInfo") or {}).get("label")
        if label:
            labels.append(str(label))
    return tuple(labels)


def _build_quote(
    payload: dict,
    *,
    symbol: str,
    side: Side,
    input_token: TokenMeta,
    output_token: TokenMeta,
    requested_in_amount_atomic: int,
    requested_at: float,
    received_at: float,
    ttl_seconds: float,
) -> Quote | None:
    """Translate a Jupiter response into a :class:`~.types.Quote`, or ``None``.

    ``None`` for anything malformed. The one thing this must never do is repair:
    a response whose ``outAmount`` will not parse is not a small output, and a
    response whose ``inAmount`` differs from what we asked for is not the swap
    we asked for. Jupiter honours ExactIn, so an ``inAmount`` mismatch means the
    request or the response is not what we think it is, and filling against it
    would reintroduce C2 from the other side.
    """
    in_amount = _as_int(payload.get("inAmount"))
    out_amount = _as_int(payload.get("outAmount"))
    if in_amount is None or out_amount is None or in_amount <= 0 or out_amount <= 0:
        return None
    if in_amount != requested_in_amount_atomic:
        return None

    # ``otherAmountThreshold`` is the slippage-worst output on an ExactIn quote.
    # If it is absent we do *not* substitute ``outAmount`` — that would silently
    # turn the conservative fill model into the optimistic one (C5a).
    min_out = _as_int(payload.get("otherAmountThreshold"))
    if min_out is None or min_out <= 0 or min_out > out_amount:
        return None

    impact_fraction = _as_float_or_none(payload.get("priceImpactPct"))
    if impact_fraction is None:
        return None

    slot = _as_int(payload.get("contextSlot"))

    try:
        return Quote(
            symbol=symbol,
            side=side,
            input_token=input_token,
            output_token=output_token,
            in_amount_atomic=in_amount,
            out_amount_atomic=out_amount,
            min_out_amount_atomic=min_out,
            # Jupiter sends a *fraction* string in 0..1 ("0.00042..." is
            # 0.042%). types.py wants whole percent.
            price_impact_pct=impact_fraction * 100.0,
            route_labels=_route_labels(payload),
            fingerprint=quote_fingerprint(
                side=str(side),
                input_mint=input_token.mint,
                output_mint=output_token.mint,
                in_amount_atomic=in_amount,
                out_amount_atomic=out_amount,
                slot=slot,
            ),
            requested_at=requested_at,
            received_at=received_at,
            context_slot=slot,
            expires_at=received_at + ttl_seconds,
            reference_price_usd=_as_float_or_none(payload.get("swapUsdValue")),
        )
    except ValidationError:
        # The invariants on Quote are the last line of defence against a vendor
        # shape change. A quote that cannot be constructed is not a quote.
        return None


# ---------------------------------------------------------------------------
# Public API — executable
# ---------------------------------------------------------------------------


def quote_exact_in(
    cfg: Config,
    *,
    symbol: str,
    side: Side,
    token: TokenMeta,
    in_amount_atomic: int,
    slippage_bps: int | None = None,
    ttl_seconds: float = DEFAULT_QUOTE_TTL_SECONDS,
    now: float | None = None,
    client: httpx.Client | None = None,
) -> Quote | None:
    """Price an exact atomic input amount against a live Jupiter route.

    This is the *only* function in the codebase that can produce something the
    broker will execute, and it returns ``None`` rather than anything synthetic
    when it cannot. That is audit C4 reduced to a single reviewable fact: grep
    for ``Quote(`` and every construction site is in this module, downstream of
    a real router response.

    ``token`` must be verified (audit §15). Passing unverified metadata is a
    programming error, not a market condition, so it raises rather than
    returning ``None`` — a caller that would have silently skipped the trade
    would also have silently skipped the bug.

    ``side`` determines which leg the traded token is on: BUY is USDC -> token,
    SELL is token -> USDC. ``in_amount_atomic`` is always in units of whichever
    of those is the *input*.
    """
    atomic(in_amount_atomic, "in_amount_atomic")
    if in_amount_atomic == 0:
        return None
    if not token.verified:
        raise ValidationError(
            f"refusing to quote {symbol} on unverified decimals for {token.mint} "
            f"(source={token.source}): an inferred exponent is a silent "
            f"factor-of-1000 error in every quantity downstream"
        )

    side = Side(side)
    if side is Side.BUY:
        input_token, output_token = USDC, token
    else:
        input_token, output_token = token, USDC

    bps = int(cfg.execution.slippage_bps_fallback if slippage_bps is None else slippage_bps)
    owned = client is None
    client = client or _new_client(cfg)
    try:
        requested_at = time.time() if now is None else now
        payload = _request_quote(
            cfg,
            client,
            input_mint=input_token.mint,
            output_mint=output_token.mint,
            amount=in_amount_atomic,
            slippage_bps=bps,
        )
        received_at = time.time() if now is None else now
        if payload is None:
            return None
        return _build_quote(
            payload,
            symbol=symbol,
            side=side,
            input_token=input_token,
            output_token=output_token,
            requested_in_amount_atomic=in_amount_atomic,
            requested_at=requested_at,
            received_at=received_at,
            ttl_seconds=ttl_seconds,
        )
    except ValidationError:
        raise
    except Exception:  # noqa: BLE001
        # A tick that cannot quote must still be able to mark its book and
        # honour its stops. What must *not* happen — and what used to — is that
        # the failure becomes a fillable object.
        return None
    finally:
        if owned:
            client.close()


def usd_to_atomic(usd: float) -> int:
    """Whole-dollar USD to micro-USDC. The only place this conversion lives.

    Truncation, not rounding: asking for one atomic unit more than the operator
    authorised is worse than asking for one less, and the difference is 1e-6 of
    a dollar either way.
    """
    if not usd > 0:
        raise ValidationError(f"usd must be > 0, got {usd!r}")
    return int(usd * 10**USDC_DECIMALS)


def quote_buy_usd(
    cfg: Config,
    *,
    symbol: str,
    token: TokenMeta,
    usd_notional: float,
    slippage_bps: int | None = None,
    ttl_seconds: float = DEFAULT_QUOTE_TTL_SECONDS,
    now: float | None = None,
    client: httpx.Client | None = None,
) -> Quote | None:
    """Quote a BUY of ``usd_notional`` dollars. USDC in, ``token`` out.

    The dollar amount is converted to micro-USDC *once*, here, and from that
    point on the integer is the order. Nothing downstream converts back.
    """
    if not usd_notional > 0:
        return None
    return quote_exact_in(
        cfg,
        symbol=symbol,
        side=Side.BUY,
        token=token,
        in_amount_atomic=usd_to_atomic(usd_notional),
        slippage_bps=slippage_bps,
        ttl_seconds=ttl_seconds,
        now=now,
        client=client,
    )


def quote_sell_tokens(
    cfg: Config,
    *,
    symbol: str,
    token: TokenMeta,
    token_amount_atomic: int,
    slippage_bps: int | None = None,
    ttl_seconds: float = DEFAULT_QUOTE_TTL_SECONDS,
    now: float | None = None,
    client: httpx.Client | None = None,
) -> Quote | None:
    """Quote a SELL of an exact held token amount. ``token`` in, USDC out.

    Takes atomic tokens, never dollars. The old code took a USD notional and
    divided it by the DexScreener mid to get a token amount — which meant an
    inaccurate mid changed *how much of the position was sold*, and the error
    only became visible as a position that would not close. The caller holds an
    exact integer quantity; that integer is what gets quoted and what gets sold.
    """
    return quote_exact_in(
        cfg,
        symbol=symbol,
        side=Side.SELL,
        token=token,
        in_amount_atomic=token_amount_atomic,
        slippage_bps=slippage_bps,
        ttl_seconds=ttl_seconds,
        now=now,
        client=client,
    )


# ---------------------------------------------------------------------------
# Public API — non-executable marking
# ---------------------------------------------------------------------------


def outage_haircut_pct(cfg: Config) -> float:
    """Whole-percent markdown applied when no route could be obtained.

    Floored at the configured adverse-move assumption
    (``slippage_bps_fallback``, 50 bp = 0.5%) because that is the *minimum*
    credible cost of getting out, and doubled on top of it because the condition
    being priced is not a normal exit — it is an exit during a router outage or
    into a pool with no route at this size, which is when realised slippage is
    worst. The number is an assumption and is labelled as one on
    ``ValuationEstimate.haircut_pct``; what matters structurally is the sign. A
    degraded valuation must never be able to look *better* than a real one, or
    an outage becomes a buy signal.
    """
    return 2.0 * cfg.execution.slippage_bps_fallback / 100.0


def mark_route(
    cfg: Config,
    *,
    symbol: str,
    mint: str,
    quantity_atomic: int,
    mid_price_usd: float | None = None,
    token: TokenMeta | None = None,
    slippage_bps: int | None = None,
    ttl_seconds: float = DEFAULT_QUOTE_TTL_SECONDS,
    now: float | None = None,
    client: httpx.Client | None = None,
) -> Quote | ValuationEstimate:
    """What the held ``quantity_atomic`` of ``symbol`` is worth, and how solidly.

    The marking path, and the reason :class:`~.types.ValuationEstimate` exists.
    ``portfolio.py`` calls this to mark a book; ``risk.py`` reads the result to
    decide whether a stop may fire on it.

    * A :class:`~.types.Quote` here is a *real exit route at the real size* —
      the only honest estimate of liquidation value, and the only basis a
      ``Mark`` may label ``"route"``.
    * A :class:`~.types.ValuationEstimate` is everything else: no route, no
      decimals, nothing held. It carries its own haircut so no consumer can
      forget to apply one, and it has ``mid_price_usd=None`` when we could not
      even find a mid, because "unmarkable" is a risk incident and not a number.

    The broker's signature accepts only ``Quote``, so the degraded branch is
    structurally unable to reach execution. That is audit C4 enforced by the
    type system rather than by a ``degraded`` flag everyone remembers to check.
    """
    at = time.time() if now is None else now

    def estimate(reason: str) -> ValuationEstimate:
        return ValuationEstimate(
            symbol=symbol,
            mid_price_usd=mid_price_usd if (mid_price_usd or 0) > 0 else None,
            haircut_pct=outage_haircut_pct(cfg),
            reason=reason,
            at=at,
            source="quotes.mark_route",
        )

    if quantity_atomic <= 0:
        return estimate("nothing held to mark")

    owned = client is None
    client = client or _new_client(cfg)
    try:
        meta = token or token_meta(cfg, mint, client=client)
        if meta is None or not meta.verified:
            return estimate(f"no verified decimals for mint {mint}")
        routed = quote_exact_in(
            cfg,
            symbol=symbol,
            side=Side.SELL,
            token=meta,
            in_amount_atomic=quantity_atomic,
            slippage_bps=slippage_bps,
            ttl_seconds=ttl_seconds,
            now=now,
            client=client,
        )
        if routed is None:
            return estimate("no jupiter exit route at the held size")
        return routed
    finally:
        if owned:
            client.close()


__all__ = [
    "DEFAULT_QUOTE_TTL_SECONDS",
    "USDC",
    "USDC_DECIMALS",
    "USDC_MINT",
    "clear_token_cache",
    "mark_route",
    "outage_haircut_pct",
    "quote_buy_usd",
    "quote_exact_in",
    "quote_sell_tokens",
    "token_meta",
    "usd_to_atomic",
]
