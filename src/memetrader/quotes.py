"""Jupiter adapter: what a given notional would *actually* fill at.

A DexScreener mid price is what the last trade printed at. It is not what you
would get. For an asset class where a $500 order can move the pool, the gap
between the two is the entire difference between a backtest that looks good and
one that is true — so every trade is priced by asking Jupiter for a real route
at the exact size, and the mid is used only as an explicitly-flagged fallback.

This module never raises. A tick that cannot quote must still be able to mark
its book and honour its stops; a quote that is visibly degraded is strictly
better than an exception that takes the loop down. Degradation is recorded on
the ``FillQuote`` itself so it reaches the decision log and the model's prompt
rather than being swallowed here.
"""

from __future__ import annotations

import httpx

from .config import Config
from .http import make_client
from .types import FillQuote, Side

# USDC is the unit of account for the whole book, and its 6 decimals are a fixed
# property of the mint — not something worth a network round-trip.
_USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
_USDC_DECIMALS = 6

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_HEADERS = {"User-Agent": _BROWSER_UA, "Accept": "application/json"}

# Token decimals never change for a live mint, so one lookup per process is
# plenty. Module-level rather than on Config because it is a cache, not a
# setting, and Config is frozen.
_DECIMALS_CACHE: dict[str, int] = {}

# Plausible SPL decimals. Used to reject a derived value that came out absurd,
# which is the signal that ``mid_price_usd`` was itself wrong.
_MIN_DECIMALS = 0
_MAX_DECIMALS = 18


def _new_client(cfg: Config) -> httpx.Client:
    # See http.make_client — certifi's bundle fails behind a TLS-inspecting proxy.
    return make_client(cfg.data.http_timeout_seconds, _HEADERS)


def _headers(cfg: Config) -> dict[str, str]:
    """Jupiter's keyed host rejects an unexpected ``x-api-key``, and the lite
    host ignores it — so the header is sent only when a key actually exists."""
    headers = dict(_HEADERS)
    if cfg.data.jupiter_api_key:
        headers["x-api-key"] = cfg.data.jupiter_api_key
    return headers


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Decimals
# ---------------------------------------------------------------------------


def _derive_decimals(base_units: float, ui_units: float) -> int | None:
    """Recover a token's decimals from one base-unit/UI-unit pair.

    Decimals are an integer power of ten, so an estimate of the UI quantity
    that is merely in the right ballpark pins the exponent exactly: being off
    by 3x moves ``log10`` by 0.48, nowhere near the 0.5 needed to round wrong.
    That is what makes a DexScreener mid good enough to derive decimals from
    even though it is not good enough to price a fill with.
    """
    if base_units <= 0 or ui_units <= 0:
        return None
    ratio = base_units / ui_units
    decimals = 0
    while ratio >= 10 ** (decimals + 0.5) and decimals < _MAX_DECIMALS:
        decimals += 1
    if not _MIN_DECIMALS <= decimals <= _MAX_DECIMALS:
        return None
    return decimals


def _lookup_decimals(cfg: Config, client: httpx.Client, mint: str) -> int | None:
    """Ask Jupiter's token search for a mint's decimals.

    ``/tokens/v1/token/{mint}`` — the endpoint most docs still reference — now
    404s; ``/tokens/v2/search?query={mint}`` is the live route and accepts a
    comma-separated list, so this stays cheap if it ever needs batching.
    """
    if mint in _DECIMALS_CACHE:
        return _DECIMALS_CACHE[mint]
    try:
        response = client.get(
            f"{cfg.data.jupiter_url_base.rstrip('/')}/tokens/v2/search",
            params={"query": mint},
            headers=_headers(cfg),
        )
        if response.status_code >= 400:
            return None
        for entry in response.json() or []:
            if not isinstance(entry, dict):
                continue
            if entry.get("id") == mint and isinstance(entry.get("decimals"), int):
                _DECIMALS_CACHE[mint] = entry["decimals"]
                return entry["decimals"]
    except (httpx.HTTPError, ValueError):
        return None
    return None


def _token_decimals(
    cfg: Config,
    client: httpx.Client,
    mint: str,
    mid_price_usd: float,
) -> int | None:
    """Best-effort decimals, cheapest source first.

    1. Process cache.
    2. Jupiter's token search.
    3. A throwaway BUY-direction quote, from whose ``outAmount`` the exponent
       is derived against ``mid_price_usd``. Only reached when the token API is
       down, and only worth doing because SELL needs decimals *before* it can
       build its request — without this, an outage would degrade every exit,
       which is the one direction where pricing accuracy is realized P&L.
    """
    cached = _DECIMALS_CACHE.get(mint)
    if cached is not None:
        return cached

    found = _lookup_decimals(cfg, client, mint)
    if found is not None:
        return found

    if mid_price_usd <= 0:
        return None
    probe_usd = 100.0
    payload = _request_quote(
        cfg,
        client,
        input_mint=_USDC_MINT,
        output_mint=mint,
        amount=int(probe_usd * 10**_USDC_DECIMALS),
    )
    if payload is None:
        return None
    derived = _derive_decimals(
        base_units=_as_float(payload.get("outAmount")),
        ui_units=probe_usd / mid_price_usd,
    )
    if derived is not None:
        _DECIMALS_CACHE[mint] = derived
    return derived


# ---------------------------------------------------------------------------
# Jupiter
# ---------------------------------------------------------------------------


def _request_quote(
    cfg: Config,
    client: httpx.Client,
    *,
    input_mint: str,
    output_mint: str,
    amount: int,
) -> dict | None:
    """One ``/swap/v1/quote`` call. Returns ``None`` on any failure."""
    if amount <= 0:
        return None
    try:
        response = client.get(
            f"{cfg.data.jupiter_url_base.rstrip('/')}/swap/v1/quote",
            params={
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": amount,
                "slippageBps": int(cfg.execution.slippage_bps_fallback),
            },
            headers=_headers(cfg),
        )
        if response.status_code >= 400:
            return None
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    if not isinstance(payload, dict) or not payload.get("outAmount"):
        return None
    return payload


def _route_labels(payload: dict) -> tuple[str, ...]:
    """Labels for every hop, in order and **not** deduplicated.

    Each hop pays its own pool fee, so a route that touches the same venue
    twice owes that fee twice — ``ExecutionConfig.fee_pct_for`` sums over this
    tuple and is built on that assumption. Observed live labels include venues
    absent from ``config.toml`` ("AlphaQ", "Scorch", "Byreal"), which is fine:
    unknown venues fall back to ``default_pool_fee_pct`` rather than free.
    """
    labels: list[str] = []
    for hop in payload.get("routePlan") or []:
        label = ((hop or {}).get("swapInfo") or {}).get("label")
        if label:
            labels.append(str(label))
    return tuple(labels)


def _effective_price(
    side: Side,
    *,
    in_amount: float,
    out_amount: float,
    token_decimals: int,
) -> float | None:
    """USD per token for the route, both sides' decimals applied.

    This number is the P&L. Getting the scaling wrong does not look wrong — it
    looks like a plausible price off by a power of ten, which the book will
    happily compound for hours. Hence: convert both legs to UI units first,
    then divide USD by tokens, never base units by base units.
    """
    if in_amount <= 0 or out_amount <= 0:
        return None
    if side is Side.BUY:
        usd_ui = in_amount / 10**_USDC_DECIMALS
        token_ui = out_amount / 10**token_decimals
    else:
        token_ui = in_amount / 10**token_decimals
        usd_ui = out_amount / 10**_USDC_DECIMALS
    if token_ui <= 0:
        return None
    return usd_ui / token_ui


def _fallback(
    cfg: Config,
    symbol: str,
    mint: str,
    side: Side,
    usd_notional: float,
    mid_price_usd: float,
    reason: str,
) -> FillQuote:
    """Mid price pushed in the direction that hurts.

    Always adverse, never optimistic: a fallback that flattered the fill would
    make the degraded path *more* attractive than the real one and quietly bias
    the whole strategy toward trading during outages.
    """
    adjustment = cfg.execution.slippage_bps_fallback / 10_000.0
    price = mid_price_usd * (1 + adjustment if side is Side.BUY else 1 - adjustment)
    return FillQuote(
        symbol=symbol,
        mint=mint,
        side=side,
        usd_notional=usd_notional,
        price_usd=max(price, 0.0),
        # The assumed adverse move, as whole percent, so risk.py sees a real
        # cost estimate instead of a 0.0 that reads as "free".
        price_impact_pct=cfg.execution.slippage_bps_fallback / 100.0,
        route_labels=(),
        pool_fee_pct=cfg.execution.fee_pct_for(()),
        degraded=True,
        degraded_reason=reason,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fill_quote(
    cfg: Config,
    symbol: str,
    mint: str,
    side: Side,
    usd_notional: float,
    *,
    mid_price_usd: float,
    client: httpx.Client | None = None,
) -> FillQuote:
    """Price ``usd_notional`` of ``symbol`` against a live Jupiter route.

    BUY quotes USDC -> mint, so the input amount is the notional directly.
    SELL quotes mint -> USDC, and since the caller specifies a *USD* size, the
    token input is derived as ``usd_notional / mid_price_usd``. The mid is used
    only to size the request — the price that comes back is still the real
    route's, so an inaccurate mid changes how much we asked to sell, not what
    we believe we got for it.
    """
    owned = client is None
    client = client or _new_client(cfg)
    try:
        if usd_notional <= 0 or mid_price_usd <= 0:
            return _fallback(
                cfg,
                symbol,
                mint,
                side,
                usd_notional,
                mid_price_usd,
                f"cannot quote: usd_notional={usd_notional}, "
                f"mid_price_usd={mid_price_usd}",
            )

        decimals = _token_decimals(cfg, client, mint, mid_price_usd)
        if decimals is None:
            return _fallback(
                cfg,
                symbol,
                mint,
                side,
                usd_notional,
                mid_price_usd,
                f"could not determine decimals for mint {mint}",
            )

        if side is Side.BUY:
            input_mint, output_mint = _USDC_MINT, mint
            amount = int(usd_notional * 10**_USDC_DECIMALS)
        else:
            input_mint, output_mint = mint, _USDC_MINT
            amount = int(usd_notional / mid_price_usd * 10**decimals)

        payload = _request_quote(
            cfg,
            client,
            input_mint=input_mint,
            output_mint=output_mint,
            amount=amount,
        )
        if payload is None:
            return _fallback(
                cfg,
                symbol,
                mint,
                side,
                usd_notional,
                mid_price_usd,
                "jupiter quote unavailable",
            )

        price = _effective_price(
            side,
            in_amount=_as_float(payload.get("inAmount")),
            out_amount=_as_float(payload.get("outAmount")),
            token_decimals=decimals,
        )
        if price is None or price <= 0:
            return _fallback(
                cfg,
                symbol,
                mint,
                side,
                usd_notional,
                mid_price_usd,
                "jupiter returned an unusable amount pair",
            )

        labels = _route_labels(payload)
        return FillQuote(
            symbol=symbol,
            mint=mint,
            side=side,
            usd_notional=usd_notional,
            price_usd=price,
            # Jupiter sends a *fraction* string in 0..1 ("0.00042..." is 0.042%).
            # types.py wants whole percent; forget the x100 and every impact
            # check passes, forever.
            price_impact_pct=_as_float(payload.get("priceImpactPct")) * 100.0,
            route_labels=labels,
            pool_fee_pct=cfg.execution.fee_pct_for(labels),
        )
    except Exception as exc:  # noqa: BLE001 - see module docstring: never raise
        return _fallback(
            cfg,
            symbol,
            mint,
            side,
            usd_notional,
            mid_price_usd,
            f"unexpected quoting failure: {type(exc).__name__}: {exc}",
        )
    finally:
        if owned:
            client.close()


__all__ = ["fill_quote"]
