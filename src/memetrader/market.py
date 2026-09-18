"""DexScreener + GeckoTerminal adapter: vendor JSON in, ``MarketSnapshot`` out.

This module is the only place in the codebase allowed to know what DexScreener
and GeckoTerminal actually return. Everything it emits already obeys the two
conventions in ``types.py`` — epoch **seconds** as float, whole percents — so
nothing downstream ever has to ask "which unit is this?".

Three hazards live here, and all three are handled at this boundary because
handling them anywhere else means handling them in five places:

1. **Unit skew.** DexScreener's ``pairCreatedAt`` is epoch *milliseconds*;
   GeckoTerminal's OHLCV timestamps are epoch *seconds*. A millisecond value
   that leaks downstream does not crash anything — it quietly reports a
   timestamp in the year 58,700, and every age/staleness check silently passes.
   That is the worst kind of bug, so it is killed on entry.
2. **Stringly-typed numerics.** ``priceUsd``, ``priceNative`` and friends arrive
   as JSON *strings*. Comparing those lexicographically "works" often enough to
   ship and is wrong every time.
3. **Pair selection.** See ``_best_pair`` — the naive choice is actively
   dangerous for this asset class.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

import httpx

from .config import Config
from .http import make_client
from .types import Candle, CoinSnapshot, MarketSnapshot, PriceLadder, TxnCounts

# Cloudflare fronts DexScreener and answers the stock httpx/python-requests
# user-agent with a 403 interstitial whenever it feels like it. A plain browser
# UA is enough to be let through; nothing else about the request needs to lie.
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_HEADERS = {"User-Agent": _BROWSER_UA, "Accept": "application/json"}

# Quote tokens we are willing to price a memecoin against. See ``_best_pair``.
_WSOL = "So11111111111111111111111111111111111111112"
_USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
_USDT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
_NUMERAIRES = frozenset({_WSOL, _USDC, _USDT})

# GeckoTerminal allows ~30 requests/minute without a key. A snapshot costs two
# calls per coin, so at a 15-minute cadence we are three orders of magnitude
# under the cap — this delay exists only to avoid arriving as an instantaneous
# burst, which is what naive rate limiters actually punish. Tests set it to 0.
_GECKO_DELAY_SECONDS = 1.5

# Sanity window for normalized timestamps, used to catch a unit slip rather than
# to validate data. 2020-01-01 .. 2100-01-01 in epoch seconds.
_TS_FLOOR = 1_577_836_800.0
_TS_CEILING = 4_102_444_800.0


class MarketDataError(RuntimeError):
    """A market read failed in a way that must stop the tick.

    Raised only when there is no usable price at all. A merely *incomplete*
    read (missing candles) degrades instead — see ``CoinSnapshot.degraded``.
    """


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _as_float(value: Any, default: float = 0.0) -> float:
    """Coerce vendor JSON to float, tolerating strings, nulls and junk.

    Returns ``default`` rather than raising: one malformed field on one pair
    should not cost us the whole snapshot.
    """
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_optional_float(value: Any) -> float | None:
    """Like ``_as_float`` but preserves "absent" as ``None``.

    Used for ``fdv``, where 0.0 and "unknown" are very different claims.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ms_to_seconds(value: Any) -> float | None:
    """Normalize a DexScreener epoch-millisecond field to epoch seconds.

    Defensive rather than a blind ``/1000``: DexScreener has been observed
    serving both units during migrations, and a value that is already in
    seconds must pass through untouched. Anything that lands outside a sane
    calendar window after conversion is reported as unknown instead of as a
    confident wrong answer.
    """
    raw = _as_optional_float(value)
    if raw is None or raw <= 0:
        return None
    seconds = raw / 1000.0 if raw > _TS_CEILING else raw
    if not _TS_FLOOR <= seconds <= _TS_CEILING:
        return None
    return seconds


def _txns(block: Any, window: str) -> TxnCounts:
    entry = (block or {}).get(window) or {}
    return TxnCounts(buys=_as_int(entry.get("buys")), sells=_as_int(entry.get("sells")))


def _price_ladder(block: Any) -> PriceLadder:
    """Build the percent-change ladder.

    ``priceChange.m5`` is absent on roughly half of live pairs — DexScreener
    omits the key entirely on quiet pools rather than sending 0, and the pool we
    actually select for BONK is one of them. Absent stays ``None`` rather than
    collapsing to 0.0: "no 5-minute move was reported" and "the price was flat"
    are different claims, and the prompt renders the former as n/a.
    """
    block = block or {}
    return PriceLadder(
        m5=_as_optional_float(block.get("m5")),
        h1=_as_optional_float(block.get("h1")),
        h6=_as_optional_float(block.get("h6")),
        h24=_as_optional_float(block.get("h24")),
    )


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------


def _pair_liquidity(pair: dict) -> float:
    return _as_float((pair.get("liquidity") or {}).get("usd"))


def _best_pair(pairs: Iterable[dict]) -> tuple[dict | None, str | None]:
    """Pick the pool whose price we are willing to believe.

    Two rules, in order, and both were written against observed live data:

    *Never ``pairs[0]``.* DexScreener returns pairs in no documented order. A
    major mint resolves to dozens of pools, most of them dust; the first entry
    is routinely a dead pool whose last trade was hours ago.

    *Only pools quoted in a real numéraire.* Taking the global max of
    ``liquidity.usd`` is the obvious rule and it is a trap. Observed live: the
    single highest-``liquidity.usd`` BONK pool was a Meteora DLMM quoted
    against a pump.fun token, reporting $2.4M of "liquidity" and a BONK price
    of $0.01434 — roughly 4,900x the real $0.0000029. ``liquidity.usd`` is
    derived from the quote token's own (fabricated) valuation, so a worthless
    quote token mints unlimited fake liquidity. Restricting to SOL/USDC/USDT
    makes that attack cost real money.

    Returns ``(pair, warning)``. ``warning`` is non-None when the numéraire
    filter had to be abandoned, so the caller can mark the snapshot degraded
    rather than pretend the price is trustworthy.
    """
    pairs = [p for p in pairs if isinstance(p, dict) and _as_float(p.get("priceUsd")) > 0]
    if not pairs:
        return None, None

    quoted = [
        p for p in pairs if (p.get("quoteToken") or {}).get("address") in _NUMERAIRES
    ]
    if quoted:
        return max(quoted, key=_pair_liquidity), None

    # No SOL/USDC/USDT pool at all. Still better to trade on *something* than to
    # blind the tick, but the price is explicitly untrusted from here on.
    best = max(pairs, key=_pair_liquidity)
    quote_symbol = (best.get("quoteToken") or {}).get("symbol") or "?"
    return best, (
        f"no SOL/USDC/USDT-quoted pool; priced against {quote_symbol}, "
        "price may be fabricated"
    )


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _new_client(cfg: Config) -> httpx.Client:
    # Via http.make_client so TLS is verified against the OS trust store; on a
    # TLS-inspecting corporate network certifi's bundle fails every request.
    return make_client(cfg.data.http_timeout_seconds, _HEADERS)


def _check(response: httpx.Response, what: str) -> None:
    """Raise a diagnosable error, and make a 403 unmistakable.

    A 403 here is Cloudflare, not rate limiting, and retrying it just burns the
    tick budget against a wall — so it gets its own message that names the
    actual fix instead of surfacing as a generic HTTP error somebody will
    "solve" by adding a retry loop.
    """
    if response.status_code == 403:
        raise MarketDataError(
            f"{what}: HTTP 403 (Cloudflare). This is a blocked client, not a "
            "transient failure - do not retry. Check the User-Agent header "
            f"and whether the host now requires a key. URL: {response.request.url}"
        )
    if response.status_code >= 400:
        raise MarketDataError(
            f"{what}: HTTP {response.status_code} for {response.request.url}: "
            f"{response.text[:200]}"
        )


def _fetch_pairs(cfg: Config, client: httpx.Client) -> dict[str, list[dict]]:
    """One batched call for every configured mint, grouped by base token.

    DexScreener accepts up to 30 comma-separated addresses on this route, which
    comfortably covers any plausible ``[[coins]]`` list, so a snapshot costs
    exactly one request no matter how many coins are configured.
    """
    mints = [c.mint for c in cfg.coins]
    url = f"{cfg.data.dexscreener_base.rstrip('/')}/latest/dex/tokens/{','.join(mints)}"
    response = client.get(url, headers=_HEADERS)
    _check(response, "dexscreener tokens")

    payload = response.json()
    # The route has returned both a bare list and a {"pairs": [...]} envelope
    # across versions; accept either rather than being broken by a redeploy.
    raw = payload.get("pairs") if isinstance(payload, dict) else payload
    grouped: dict[str, list[dict]] = {}
    for pair in raw or []:
        if not isinstance(pair, dict):
            continue
        address = (pair.get("baseToken") or {}).get("address")
        if address:
            grouped.setdefault(address.lower(), []).append(pair)
    return grouped


def _fetch_candles(
    cfg: Config,
    client: httpx.Client,
    pool: str,
    timeframe: str,
    aggregate: int,
    limit: int,
) -> tuple[Candle, ...]:
    """Fetch one OHLCV series, returned oldest-first.

    GeckoTerminal serves ``ohlcv_list`` **newest-first** (verified live), while
    every indicator in ``signals.py`` walks forward in time. Reversing here is
    not cosmetic: an un-reversed series makes EMAs, MACD and ATR compute over
    time-reversed data and produce confident garbage.
    """
    url = (
        f"{cfg.data.geckoterminal_base.rstrip('/')}"
        f"/networks/solana/pools/{pool}/ohlcv/{timeframe}"
    )
    response = client.get(
        url,
        params={"aggregate": aggregate, "limit": limit},
        headers=_HEADERS,
    )
    _check(response, f"geckoterminal {timeframe} ohlcv")

    rows = (
        ((response.json() or {}).get("data") or {}).get("attributes") or {}
    ).get("ohlcv_list") or []

    candles: list[Candle] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            continue
        ts = _as_float(row[0])
        # GeckoTerminal is already in seconds; this only catches a future unit
        # change, so that a silent switch to milliseconds fails a test instead
        # of shifting every bar 56,000 years into the future.
        if ts > _TS_CEILING:
            ts /= 1000.0
        candles.append(
            Candle(
                ts=ts,
                open=_as_float(row[1]),
                high=_as_float(row[2]),
                low=_as_float(row[3]),
                close=_as_float(row[4]),
                volume=_as_float(row[5]),
            )
        )
    candles.sort(key=lambda c: c.ts)
    return tuple(candles)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_pairs(cfg: Config, client: httpx.Client | None = None) -> dict[str, str]:
    """Map symbol -> best pair address. Used by startup validation.

    Deliberately does not touch GeckoTerminal: startup wants a fast, cheap
    answer to "do these mints exist and trade", and a missing candle series is
    not a reason to refuse to boot.
    """
    owned = client is None
    client = client or _new_client(cfg)
    try:
        grouped = _fetch_pairs(cfg, client)
        resolved: dict[str, str] = {}
        for coin in cfg.coins:
            pair, _ = _best_pair(grouped.get(coin.mint.lower(), []))
            if pair is None:
                raise MarketDataError(
                    f"{coin.symbol}: DexScreener returned no tradeable pair for "
                    f"mint {coin.mint} - the mint is wrong, or the token has no "
                    "live pool."
                )
            resolved[coin.symbol] = str(pair.get("pairAddress") or "")
        return resolved
    finally:
        if owned:
            client.close()


def snapshot(
    cfg: Config,
    *,
    client: httpx.Client | None = None,
    previous: MarketSnapshot | None = None,
    with_candles: bool = True,
) -> MarketSnapshot:
    """Read every configured coin: one DexScreener call plus, by default, 2
    GeckoTerminal calls per coin.

    ``with_candles=False`` skips GeckoTerminal entirely and returns a
    price-only snapshot. The fast tick wants this: it marks the book and checks
    stops off ``price_usd`` and ``liquidity_usd`` and never touches a candle,
    but fetching them anyway cost 6 of GeckoTerminal's ~30 keyless requests per
    minute *and* ~7.5s of inter-request spacing on every 60s tick — a fifth of
    the rate budget and most of the tick's wall clock, spent on data nothing
    read. It was enough to earn 429s and degrade the slow tick's technicals,
    which is the one place the candles actually matter.

    A candle-less snapshot is still safe to carry forward as ``previous``:
    ``signals.flow_brief`` reads only ``liquidity_usd`` off it.

    ``previous`` is accepted but intentionally unused. Derived, cross-snapshot
    quantities (``liquidity_trend_pct``) belong to ``signals.py``, which owns
    the comparison semantics; if that computation lived here, a snapshot's
    meaning would depend on how it was obtained. The parameter exists so the
    call site stays stable if that ever changes.
    """
    owned = client is None
    client = client or _new_client(cfg)
    try:
        grouped = _fetch_pairs(cfg, client)
        coins: dict[str, CoinSnapshot] = {}
        gecko_calls = 0

        for coin in cfg.coins:
            pair, pair_warning = _best_pair(grouped.get(coin.mint.lower(), []))
            if pair is None:
                # No price at all. Unlike missing candles this is not
                # degradable — there is nothing to mark a book against.
                raise MarketDataError(
                    f"{coin.symbol}: DexScreener returned no tradeable pair for "
                    f"mint {coin.mint} - the mint is wrong, or the token has no "
                    "live pool."
                )

            pool = str(pair.get("pairAddress") or "")
            reasons: list[str] = [pair_warning] if pair_warning else []
            candles_5m: tuple[Candle, ...] = ()
            candles_1h: tuple[Candle, ...] = ()

            # Deliberately skipped candles are not a degraded read, so nothing
            # is appended to ``reasons`` here — the caller asked for prices.
            wanted = (
                (
                    ("5m", "minute", 5, cfg.data.candles_5m),
                    ("1h", "hour", 1, cfg.data.candles_1h),
                )
                if with_candles
                else ()
            )
            for target, timeframe, aggregate, limit in wanted:
                if gecko_calls and _GECKO_DELAY_SECONDS:
                    time.sleep(_GECKO_DELAY_SECONDS)
                gecko_calls += 1
                try:
                    series = _fetch_candles(
                        cfg, client, pool, timeframe, aggregate, limit
                    )
                except (MarketDataError, httpx.HTTPError, ValueError) as exc:
                    # Candles are enrichment, not the read itself. Losing them
                    # costs us technicals; crashing here costs us the tick, the
                    # stop-loss check and the mark-to-market.
                    reasons.append(f"{target} candles unavailable: {exc}")
                    continue
                if target == "5m":
                    candles_5m = series
                else:
                    candles_1h = series

            volume = pair.get("volume") or {}
            coins[coin.symbol] = CoinSnapshot(
                symbol=coin.symbol,
                mint=coin.mint,
                price_usd=_as_float(pair.get("priceUsd")),
                liquidity_usd=_pair_liquidity(pair),
                volume_24h_usd=_as_float(volume.get("h24")),
                volume_1h_usd=_as_float(volume.get("h1")),
                fdv_usd=_as_optional_float(pair.get("fdv")),
                price_change=_price_ladder(pair.get("priceChange")),
                txns_m5=_txns(pair.get("txns"), "m5"),
                txns_h1=_txns(pair.get("txns"), "h1"),
                txns_h24=_txns(pair.get("txns"), "h24"),
                pair_address=pool,
                dex_id=str(pair.get("dexId") or ""),
                pair_created_at=_ms_to_seconds(pair.get("pairCreatedAt")),
                candles_5m=candles_5m,
                candles_1h=candles_1h,
                degraded=bool(reasons),
                degraded_reason="; ".join(reasons) or None,
            )

        return MarketSnapshot(ts=time.time(), coins=coins)
    finally:
        if owned:
            client.close()


__all__ = ["MarketDataError", "resolve_pairs", "snapshot"]
