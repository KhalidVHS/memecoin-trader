"""DexScreener + GeckoTerminal adapter: vendor JSON in, ``MarketSnapshot`` out.

This module is the only place in the codebase allowed to know what DexScreener
and GeckoTerminal actually return. Everything it emits already obeys the two
conventions in ``types.py`` — epoch **seconds** as float, whole percents — so
nothing downstream ever has to ask "which unit is this?".

Four hazards live here, and all four are handled at this boundary because
handling them anywhere else means handling them in five places:

1. **Unit skew.** DexScreener's ``pairCreatedAt`` is epoch *milliseconds*;
   GeckoTerminal's OHLCV timestamps are epoch *seconds*. A millisecond value
   that leaks downstream does not crash anything — it quietly reports a
   timestamp in the year 58,700, and every age/staleness check silently passes.
   That is the worst kind of bug, so it is killed on entry.
2. **Stringly-typed numerics.** ``priceUsd``, ``priceNative`` and friends arrive
   as JSON *strings*. Comparing those lexicographically "works" often enough to
   ship and is wrong every time.
3. **Pair selection.** See :func:`_best_pair` — the naive choice is actively
   dangerous for this asset class.
4. **Missingness.** See below. This is the change the audit cared about most.

What the adversarial audit changed here
---------------------------------------

**Missing is no longer zero.** The old ``_as_float``/``_as_int`` (audit §11,
"Additional current code hazards", ``market.py:76-96``) returned ``0`` for a
value that was absent *and* for a value that was garbage. That single habit
turned "DexScreener omitted the m5 block for this quiet pair" into a confident
"the price was exactly flat over five minutes", and turned the string
``"n/a"`` into a real-looking number. On a live 30-pair sample, 13 pairs had no
``m5`` block at all — including BONK's best pool — so this was not a rare path.
:func:`_opt_float` and :func:`_opt_int` now keep the three cases apart:

* **absent** (key missing or JSON ``null``) -> ``None``;
* **malformed** (``"n/a"``, ``[]``, ``NaN``, ``Infinity``) -> raises
  :class:`MalformedField`, and the coin's observation is *quarantined* with a
  reason rather than silently repaired;
* **present and zero** -> ``0.0``, which is a real observation and stays one.

**Provenance is per observation, not per batch.** Audit C8 and §11: the old
``MarketSnapshot.ts`` was ``time.time()`` taken *after* every sequential HTTP
call returned, so a snapshot looked fresh while its first constituent read could
be minutes old. Every :class:`CoinSnapshot` and every :class:`CandleSeries` now
carries its own :class:`Provenance` recording when the response actually
arrived. ``event_time`` is ``None`` where the vendor genuinely does not tell us
— DexScreener's pair summary has no "as of" field at all — because a copy of
``receive_time`` is a lie that makes stale data look current.

**Pool identity is explicit.** Audit C8 again: ``_best_pair`` can legitimately
choose a different pool between two reads, and a liquidity delta computed across
that switch is the difference between two unrelated pools rather than a trend.
Every snapshot now carries a :class:`PoolRef`, and ``signals.flow_brief``
refuses to compare two observations whose ``pair_address`` differs.

**Candles are validated, not merely parsed.** Audit §7 and the §11 failure table
require unique increasing timestamps, expected intervals, OHLC invariants,
future-bar rejection and a closed-bar watermark. :func:`_fetch_candles` does all
of it and quarantines the whole series when a vendor violates one, because a
source that reports ``low > high`` on one bar has malfunctioned and we have no
basis for trusting the neighbouring bars it sent in the same response.

**A data-quality screen exists.** Audit C9 asks for universe/token safety.
Full on-chain risk screening (mint authority, freeze authority, LP lock, holder
concentration, Token-2022 extensions) needs an external service and is
deliberately **not** built here — see :class:`SafetyScreenParams`. What *is*
built is the subset computable from what we already fetch, exposed as an
explicit, named, testable screen that ``risk.py`` consumes: :func:`screen`.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .http import make_client
from .types import (
    Candle,
    CandleSeries,
    CoinSnapshot,
    DataQuality,
    MarketSnapshot,
    PoolRef,
    PriceLadder,
    Provenance,
    Timeframe,
    TxnCounts,
    ValidationError,
    finite,
)

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

# GeckoTerminal route parameters and the bar length they imply, in seconds. The
# interval is what makes gap counting and the closed-bar watermark possible at
# all: without it, "the next row is 900s later" is indistinguishable from "three
# bars are missing".
_ROUTES: dict[Timeframe, tuple[str, int, float]] = {
    Timeframe.M5: ("minute", 5, 300.0),
    Timeframe.H1: ("hour", 1, 3600.0),
}

# Slack allowed when deciding whether the newest bar is closed, and when
# rejecting a future-dated bar. Vendor clocks and ours are not synchronised;
# without a tolerance every read would either reject a legitimate bar or mark a
# closed one open, depending on which way the drift ran.
_CLOCK_TOLERANCE_SECONDS = 2.0


class MarketDataError(RuntimeError):
    """A market read failed in a way that must stop the affected observation.

    Raised when there is no usable price at all, or when a vendor response is
    self-contradictory. A merely *incomplete* read (missing candles) degrades
    instead — see ``CoinSnapshot.quality``.
    """


class MalformedField(ValidationError):
    """A vendor sent a value where a number was required, and it is not one.

    Deliberately a :class:`~memetrader.types.ValidationError`: this is corrupt
    upstream data, and the sanctioned response is to abort the observation, not
    to substitute a plausible-looking guess. The old code's guess was ``0``.
    """


class CoinRef(Protocol):
    """The two things this module needs to know about a configured coin.

    A ``Protocol`` rather than a concrete type so ``config.CoinConfig`` satisfies
    it structurally. ``market.py`` must not import ``config`` — the audit's
    config finding is that strategy, infrastructure and risk settings are
    tangled, and an adapter that reaches into the global config object is how
    that tangle propagates. Settings arrive as :class:`MarketParams`.
    """

    @property
    def symbol(self) -> str: ...

    @property
    def mint(self) -> str: ...


@dataclass(frozen=True, slots=True)
class MarketParams:
    """Everything tunable about a market read.

    Defaults are the values the system has actually been run with. The
    coordinator maps ``config.toml`` onto this; nothing here reads config
    itself, so a test can vary one number without constructing a whole
    ``Config``.
    """

    dexscreener_base: str = "https://api.dexscreener.com"
    geckoterminal_base: str = "https://api.geckoterminal.com/api/v2"
    http_timeout_seconds: float = 15.0
    candles_5m: int = 100
    candles_1h: int = 100
    # DexScreener accepts up to 30 comma-separated addresses on the tokens
    # route. Exceeding it does not error usefully, it silently truncates.
    max_mints_per_request: int = 30


@dataclass(frozen=True, slots=True)
class SafetyScreenParams:
    """Thresholds for the data-quality half of audit C9.

    C9 asks for a real token-safety screen: mint authority, freeze authority,
    LP ownership and lock, holder and dev concentration, Token-2022 extensions
    (transfer fees, hooks, permanent delegate, default-frozen). **None of that
    is implemented here and none of it is claimed.** All of it requires an
    on-chain RPC or an external risk service, and a screen that pretends to
    check authorities it never read is worse than no screen, because the
    operator stops looking. :attr:`SafetyVerdict.deferred` names each missing
    check by hand so the gap stays visible in the output rather than in a
    comment.

    What *is* implemented is the subset computable from the DexScreener pair we
    already fetch. Every threshold is a claim about tradeability and
    manipulability, not about the token's intentions:

    * ``min_liquidity_usd`` — below this the exit is the risk, not the entry.
    * ``min_pool_age_seconds`` — a pool minutes old has no price history to
      validate and is the population where the rug base rate lives (audit §11
      cites a Solana preprint reporting a very high rug-label incidence among
      newly issued tokens; that population is not BONK/WIF, which is precisely
      why age is the discriminator).
    * ``min_volume_24h_usd`` — a pool nobody trades cannot be exited at the
      displayed price. Note the audit's caveat: DEX volume is contaminated by
      wash trading, so this is a floor, never evidence of health.
    * ``min_liquidity_to_fdv`` — the low-float sanity check. A $400M FDV over a
      $30k pool is not a $400M asset; it is a $30k asset with a number attached.
    * ``max_snapshot_age_seconds`` — an observation too old to act on.
    """

    min_liquidity_usd: float = 50_000.0
    min_pool_age_seconds: float = 7.0 * 86_400.0
    min_volume_24h_usd: float = 250_000.0
    min_liquidity_to_fdv: float = 0.005  # fraction, 0.005 = 0.5%
    max_snapshot_age_seconds: float = 900.0
    require_trusted_quote: bool = True


@dataclass(frozen=True, slots=True)
class SafetyVerdict:
    """The screen's answer for one coin.

    ``vetoes`` are measured failures. ``unknowns`` are screens that could not be
    *evaluated* because the input was absent — and they veto too, because a
    screen that passes on missing data is not a screen. They are reported
    separately so an operator can tell "this pool is too small" from "we never
    learned how big this pool is", which call for different fixes.

    ``deferred`` is the list of C9 checks this screen does not perform. It is
    part of the return value on purpose.
    """

    symbol: str
    eligible: bool
    vetoes: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    deferred: tuple[str, ...] = (
        "mint_authority",
        "freeze_authority",
        "lp_ownership_and_lock",
        "holder_and_dev_concentration",
        "token_2022_extensions",
        "sellability_probe",
    )

    @property
    def reason(self) -> str:
        return "; ".join((*self.vetoes, *self.unknowns))


# ---------------------------------------------------------------------------
# Strict parsers
# ---------------------------------------------------------------------------


def _opt_float(value: Any, name: str) -> float | None:
    """Parse a vendor numeric. ``None`` for absent; raise for malformed.

    The audit's ``market.py:76-96`` finding in one function. Three input classes
    must never collapse into each other:

    * ``None`` (key missing, or an explicit JSON ``null``) is *absent*. The
      caller learns nothing and must represent that as nothing.
    * ``"n/a"``, ``[]``, ``{}``, ``"NaN"``, ``float("inf")`` are *malformed*.
      A source that sends these has malfunctioned. Returning a default here is
      how a parse failure becomes a market observation.
    * ``0`` and ``"0"`` are *present and zero*, which is a real reading and is
      returned as ``0.0``.

    ``bool`` is rejected explicitly. ``isinstance(True, int)`` is ``True`` in
    Python, so without this check ``"buys": true`` would be parsed as one
    transaction.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise MalformedField(f"{name}: expected a number, got bool {value!r}")
    if isinstance(value, (int, float)):
        try:
            return finite(float(value), name)
        except ValidationError as exc:  # NaN / inf arriving as a JSON literal
            raise MalformedField(f"{name}: {exc}") from exc
    if isinstance(value, str):
        text = value.strip()
        if not text:
            # An empty string is not an absence. Something produced it, and we
            # do not know what it meant.
            raise MalformedField(f"{name}: empty string is not a number")
        try:
            parsed = float(text)
        except ValueError as exc:
            raise MalformedField(f"{name}: {value!r} is not a number") from exc
        try:
            return finite(parsed, name)
        except ValidationError as exc:
            raise MalformedField(f"{name}: {exc}") from exc
    raise MalformedField(f"{name}: expected a number, got {type(value).__name__}")


def _opt_int(value: Any, name: str) -> int | None:
    """Parse a vendor count. ``None`` for absent; raise for malformed.

    Counts get their own parser because the failure they cause is specific:
    ``TxnCounts`` with a coerced ``0`` for each side produced a *neutral ratio
    of 1.0 computed from nothing* (audit §11, ``types.py:85-90``). A ratio of
    1.0 reads as "balanced two-sided market" and was in fact "DexScreener did
    not send this block".

    A non-integral float is malformed rather than truncated. Transaction counts
    are integers; a ``93.5`` means the field is not what we think it is.
    """
    parsed = _opt_float(value, name)
    if parsed is None:
        return None
    as_int = int(parsed)
    if as_int != parsed:
        raise MalformedField(f"{name}: {value!r} is not a whole count")
    if as_int < 0:
        raise MalformedField(f"{name}: negative count {as_int}")
    return as_int


def _as_dict(value: Any) -> dict[str, Any]:
    """A nested object from the API, or an empty one if it is not an object.

    DexScreener omits sub-objects entirely for some pairs, and has been seen to
    send ``null`` where it usually sends ``{"h24": ...}``. Both collapse to an
    empty mapping so that the caller's ``.get`` reads ``None`` — which is the
    "could not find out" value the rest of the pipeline already understands.
    Substituting a zero here instead would be the exact confusion the core
    invariant forbids.
    """
    return value if isinstance(value, dict) else {}


def _ms_to_seconds(value: Any, name: str) -> float | None:
    """Normalize a DexScreener epoch-millisecond field to epoch seconds.

    Defensive rather than a blind ``/1000``: DexScreener has been observed
    serving both units during migrations, and a value that is already in
    seconds must pass through untouched. Anything that lands outside a sane
    calendar window after conversion is *malformed*, not unknown — an epoch
    timestamp in the year 54,977 is a bug somewhere, and reporting it as an
    absence would hide the bug while still producing a usable snapshot.
    """
    raw = _opt_float(value, name)
    if raw is None:
        return None
    if raw <= 0:
        raise MalformedField(f"{name}: non-positive timestamp {raw}")
    seconds = raw / 1000.0 if raw > _TS_CEILING else raw
    if not _TS_FLOOR <= seconds <= _TS_CEILING:
        raise MalformedField(
            f"{name}: {raw} normalizes to {seconds}, outside 2020..2100 — "
            "this is a unit or encoding change, not a stale value"
        )
    return seconds


def _txns(block: Any, window: str) -> TxnCounts:
    """One window of buy/sell counts, with absence preserved on each side.

    Each side is parsed independently because DexScreener can and does omit one
    of them. ``TxnCounts.ratio`` returns ``None`` when either is missing, which
    is the whole point of this being nullable.
    """
    entry = (block or {}).get(window)
    if not isinstance(entry, dict):
        return TxnCounts(buys=None, sells=None)
    return TxnCounts(
        buys=_opt_int(entry.get("buys"), f"txns.{window}.buys"),
        sells=_opt_int(entry.get("sells"), f"txns.{window}.sells"),
    )


def _price_ladder(block: Any) -> PriceLadder:
    """Build the percent-change ladder.

    ``priceChange.m5`` is absent on roughly half of live pairs — DexScreener
    omits the key entirely on quiet pools rather than sending 0, measured at 13
    of 30 pairs on a live sample, and the pool we actually select for BONK is
    one of them. Absent stays ``None`` rather than collapsing to 0.0: "no
    5-minute move was reported" and "the price was flat" are different claims.
    """
    block = block if isinstance(block, dict) else {}
    return PriceLadder(
        m5=_opt_float(block.get("m5"), "priceChange.m5"),
        h1=_opt_float(block.get("h1"), "priceChange.h1"),
        h6=_opt_float(block.get("h6"), "priceChange.h6"),
        h24=_opt_float(block.get("h24"), "priceChange.h24"),
    )


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------


def _pair_liquidity(pair: dict) -> float | None:
    liquidity = pair.get("liquidity")
    if not isinstance(liquidity, dict):
        return None
    return _opt_float(liquidity.get("usd"), "liquidity.usd")


def _sort_liquidity(pair: dict) -> float:
    """Liquidity for *ranking only*, with unknown sorting last.

    A pair whose liquidity is absent or malformed must not win a ``max()`` by
    accident, and must not be coerced to a number that then escapes into a
    snapshot. This value never leaves the sort.
    """
    try:
        value = _pair_liquidity(pair)
    except MalformedField:
        return -1.0
    return -1.0 if value is None else value


def _candidate_price(pair: dict) -> float | None:
    """``priceUsd`` if it is a real positive number, else ``None``.

    A pair with a malformed price is dropped from *selection* rather than
    raising, because one broken pool among thirty must not cost us the coin.
    That is not the old coercion bug: nothing is substituted, the candidate is
    simply not a candidate.
    """
    try:
        price = _opt_float(pair.get("priceUsd"), "priceUsd")
    except MalformedField:
        return None
    return price if price is not None and price > 0 else None


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
    filter had to be abandoned. The caller turns that into
    ``PoolRef.trusted_quote=False``, which audit C8 requires to veto entries:
    such a pool is *diagnostic* — it says the token trades somewhere — but its
    price is not a valuation, because the quote token's own price is unknown.
    """
    candidates = [p for p in pairs if isinstance(p, dict) and _candidate_price(p)]
    if not candidates:
        return None, None

    quoted = [
        p for p in candidates if (p.get("quoteToken") or {}).get("address") in _NUMERAIRES
    ]
    if quoted:
        return max(quoted, key=_sort_liquidity), None

    # No SOL/USDC/USDT pool at all. Still better to see *something* than to
    # blind the tick, but the price is explicitly untrusted from here on and the
    # snapshot is not tradeable.
    best = max(candidates, key=_sort_liquidity)
    quote_symbol = (best.get("quoteToken") or {}).get("symbol") or "?"
    return best, (
        f"no SOL/USDC/USDT-quoted pool; priced against {quote_symbol}, "
        "price may be fabricated"
    )


def _pool_ref(pair: dict, mint: str, *, trusted_quote: bool) -> PoolRef:
    """Stable identity for the selected pool.

    ``created_at`` is parsed here and not later because this is the only place
    that knows ``pairCreatedAt`` is milliseconds. It feeds the pool-age screen.
    """
    quote = _as_dict(pair.get("quoteToken"))
    base = _as_dict(pair.get("baseToken"))
    return PoolRef(
        pair_address=str(pair.get("pairAddress") or ""),
        dex_id=str(pair.get("dexId") or ""),
        base_mint=str(base.get("address") or mint),
        quote_mint=str(quote.get("address") or ""),
        quote_symbol=str(quote.get("symbol") or "?"),
        created_at=_ms_to_seconds(pair.get("pairCreatedAt"), "pairCreatedAt"),
        trusted_quote=trusted_quote,
    )


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _new_client(params: MarketParams) -> httpx.Client:
    # Via http.make_client so TLS is verified against the OS trust store; on a
    # TLS-inspecting corporate network (this machine sits behind Zscaler) a bare
    # httpx.Client fails every request with CERTIFICATE_VERIFY_FAILED, because
    # certifi has never heard of the proxy's private CA.
    return make_client(params.http_timeout_seconds, _HEADERS)


def _check(response: httpx.Response, what: str) -> None:
    """Raise a diagnosable error, and make the two misread statuses unmistakable.

    A **403** here is Cloudflare, not rate limiting, and retrying it just burns
    the tick budget against a wall — so it gets its own message that names the
    actual fix instead of surfacing as a generic HTTP error somebody will
    "solve" by adding a retry loop.

    A **422** is not rate limiting either, and this codebase paid for that
    finding once already: a full day (2026-09-19) established that the archive
    host's HTTP 422 ``{"data": null, "error": "Timeout. Maybe slow down a
    bit"}`` is a **~3-second server-side query timeout on a dense request
    range**, not throttling. The evidence is recorded in full at
    ``sentiment._ARCTIC_SPACING_S``: 12 back-to-back 200s at the failing
    spacing, a reproduction on the *first cold request of a fresh process* with
    no prior traffic to be limited for, identical failures at 0.6s/1.0s/2.0s/
    2.5s spacing and after a 105-second backoff, and 33 consecutive 200s across
    11 field combinations. The message says "slow down"; slowing down does not
    help. The correct response is to narrow the request and report the
    shortfall, never to sleep and retry the same range.
    """
    if response.status_code == 403:
        raise MarketDataError(
            f"{what}: HTTP 403 (Cloudflare). This is a blocked client, not a "
            "transient failure - do not retry. Check the User-Agent header "
            f"and whether the host now requires a key. URL: {response.request.url}"
        )
    if response.status_code == 422:
        raise MarketDataError(
            f"{what}: HTTP 422. Established 2026-09-19 (see "
            "sentiment._ARCTIC_SPACING_S): a 422 from these hosts is a "
            "server-side query timeout on the request range, NOT rate "
            "limiting. Backing off does not help - narrow the range. "
            f"URL: {response.request.url}"
        )
    if response.status_code >= 400:
        raise MarketDataError(
            f"{what}: HTTP {response.status_code} for {response.request.url}: "
            f"{response.text[:200]}"
        )


def _fetch_pairs(
    mints: Sequence[str],
    params: MarketParams,
    client: httpx.Client,
    *,
    now: Callable[[], float],
) -> tuple[dict[str, list[dict]], float]:
    """One batched call for every configured mint, grouped by base token.

    DexScreener accepts up to 30 comma-separated addresses on this route, which
    comfortably covers any plausible ``[[coins]]`` list, so a snapshot costs
    exactly one request no matter how many coins are configured.

    Returns the grouping **and the receive time of this specific response**.
    That second value is the audit's C8 fix in miniature: it is when *these*
    prices were read, and it is what every resulting ``CoinSnapshot`` is
    timestamped with — not the moment the whole tick happened to finish.
    """
    if len(mints) > params.max_mints_per_request:
        raise MarketDataError(
            f"{len(mints)} mints exceeds DexScreener's "
            f"{params.max_mints_per_request}-address limit on this route; it "
            "truncates silently rather than erroring, so the request is "
            "refused here instead."
        )
    url = f"{params.dexscreener_base.rstrip('/')}/latest/dex/tokens/{','.join(mints)}"
    response = client.get(url, headers=_HEADERS)
    receive_time = now()
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
    return grouped, receive_time


def _parse_candle_rows(rows: Any, *, now: float) -> tuple[list[Candle], int]:
    """Rows -> candles, oldest-first, with structurally broken rows counted.

    Ordering is not cosmetic. GeckoTerminal serves ``ohlcv_list``
    **newest-first** (verified live), while every indicator in ``signals.py``
    walks forward in time. An un-reversed series makes EMAs, MACD and ATR
    compute over time-reversed data and produce confident garbage; this codebase
    hit exactly that bug once.

    A row that is not a 6-element sequence is skipped and counted — that is a
    transport-level artefact, not a claim about the market. A row that *is* a
    row but whose numbers are malformed or violate the OHLC invariants raises,
    because that is a malfunctioning vendor and the caller must quarantine the
    whole series rather than silently trade a hole in it.
    """
    candles: list[Candle] = []
    skipped = 0
    for row in rows or []:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            skipped += 1
            continue
        ts = _opt_float(row[0], "candle.ts")
        if ts is None:
            raise MalformedField("candle.ts is null")
        # GeckoTerminal is already in seconds; this only catches a future unit
        # change, so that a silent switch to milliseconds fails a test instead
        # of shifting every bar 56,000 years into the future.
        if ts > _TS_CEILING:
            ts /= 1000.0
        if not _TS_FLOOR <= ts <= _TS_CEILING:
            raise MalformedField(f"candle.ts {ts} outside 2020..2100")
        if ts > now + _CLOCK_TOLERANCE_SECONDS:
            # A bar that opens in the future is the purest form of look-ahead:
            # it cannot be an observation, and anything computed from it is a
            # prediction laundered as history. Audit §7/§11.
            raise MalformedField(f"candle at {ts} is future-dated against read time {now}")
        values: list[float] = []
        for index, name in ((1, "open"), (2, "high"), (3, "low"), (4, "close")):
            parsed = _opt_float(row[index], f"candle.{name}")
            if parsed is None:
                raise MalformedField(f"candle.{name} is null")
            values.append(parsed)
        volume = _opt_float(row[5], "candle.volume")
        if volume is None:
            raise MalformedField("candle.volume is null")
        # Candle.__post_init__ enforces low <= open/close <= high and low <=
        # high. A vendor reporting low > high silently produces a negative range
        # in every ATR downstream, which understates volatility exactly when it
        # matters; we would rather lose the series.
        candles.append(
            Candle(
                ts=ts,
                open=values[0],
                high=values[1],
                low=values[2],
                close=values[3],
                volume=volume,
                closed=True,  # provisional; the watermark is applied below
            )
        )
    candles.sort(key=lambda c: c.ts)
    return candles, skipped


def _audit_spacing(candles: Sequence[Candle], interval: float) -> int:
    """Reject duplicates and misalignment; count genuine gaps.

    CoinGecko documents that empty intervals may be skipped rather than sent as
    zero-volume bars, so a gap is *expected* and is information (audit §7). A
    duplicate timestamp is not: it means the same bar arrived twice, and there
    is no way to know which copy is the real one, so the series is refused.
    A spacing that is not a whole multiple of the interval means the bars are
    not on the grid we think they are, which invalidates every window length in
    ``signals.py``.
    """
    missing = 0
    for previous, current in itertools.pairwise(candles):
        delta = current.ts - previous.ts
        if delta == 0.0:
            raise MalformedField(f"duplicate candle timestamp {current.ts}")
        steps = delta / interval
        if abs(steps - round(steps)) > 1e-6:
            raise MalformedField(
                f"candle spacing {delta}s is not a multiple of the {interval}s "
                "interval — the bars are not on the expected grid"
            )
        missing += round(steps) - 1
    return missing


def _fetch_candles(
    params: MarketParams,
    client: httpx.Client,
    pool: str,
    timeframe: Timeframe,
    limit: int,
    *,
    now: Callable[[], float],
) -> CandleSeries:
    """Fetch, validate and watermark one OHLCV series for one pool.

    Everything the audit's failure table demands of a candle feed happens here,
    at the boundary, because a series that reaches ``signals.py`` unvalidated is
    a series nobody will validate:

    * **ordering** — returned oldest-first (the vendor sends newest-first);
    * **OHLC invariants** — enforced by ``Candle``; a violation quarantines;
    * **duplicates** — refused outright;
    * **gaps** — counted into ``CandleSeries.missing_intervals`` rather than
      silently closed up, so a "20-bar" window that actually spans 26 bars of
      wall clock is visible;
    * **future bars** — refused;
    * **the closed-bar watermark** — the newest bar is almost always still
      being written. It is kept, because current volume and the running price
      are legitimate *state*, but it is labelled ``closed=False`` and
      ``signals.py`` computes every feature from ``closed_candles`` only. That
      is the look-ahead fix: an in-progress bar in a historical window both
      leaks the current period and understates its own volume.

    The series is bound to ``pool_address``. Audit §7 is explicit that a pool
    switch creates a synthetic price and liquidity regime, so a series that
    cannot name its pool cannot be trusted to be continuous.
    """
    route, aggregate, interval = _ROUTES[timeframe]
    url = (
        f"{params.geckoterminal_base.rstrip('/')}"
        f"/networks/solana/pools/{pool}/ohlcv/{route}"
    )
    response = client.get(
        url,
        params={"aggregate": aggregate, "limit": limit},
        headers=_HEADERS,
    )
    receive_time = now()
    _check(response, f"geckoterminal {route} ohlcv")

    rows = (((response.json() or {}).get("data") or {}).get("attributes") or {}).get(
        "ohlcv_list"
    ) or []

    parsed, skipped = _parse_candle_rows(rows, now=receive_time)
    if not parsed:
        raise MarketDataError(
            f"geckoterminal {route} ohlcv for {pool}: no usable rows "
            f"({skipped} unparseable)"
        )
    missing = _audit_spacing(parsed, interval)

    # The watermark. A bar is closed only once its whole interval is behind us.
    watermarked = tuple(
        Candle(
            ts=c.ts,
            open=c.open,
            high=c.high,
            low=c.low,
            close=c.close,
            volume=c.volume,
            closed=(c.ts + interval) <= (receive_time + _CLOCK_TOLERANCE_SECONDS),
        )
        for c in parsed
    )

    quality = DataQuality.OK
    reason: str | None = None
    if skipped or missing:
        quality = DataQuality.DEGRADED
        reason = f"{skipped} unparseable rows, {missing} missing intervals"

    return CandleSeries(
        timeframe=timeframe,
        pool_address=pool,
        candles=watermarked,
        interval_seconds=interval,
        # ``event_time`` is the open of the newest bar: a time at which this
        # data was demonstrably true at the source. It is deliberately the bar's
        # *open* and not its close, so a series can never look fresher than it
        # is. ``source_time`` is None because GeckoTerminal does not tell us
        # when it computed the response, and guessing is what C8 is about.
        provenance=Provenance(
            source="geckoterminal",
            receive_time=receive_time,
            event_time=watermarked[-1].ts,
            source_time=None,
            quality=quality,
            quality_reason=reason,
        ),
        missing_intervals=missing,
    )


# ---------------------------------------------------------------------------
# Snapshot assembly
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Accumulator:
    """Field-by-field parse with failures collected instead of thrown away.

    One malformed field must not cost us the whole coin, and must not be
    repaired into a number either. So each field is attempted independently;
    a failure yields ``None`` *and* a recorded reason *and* quarantines the
    observation. Quarantined data is retained — it is evidence about a source's
    health — but ``CoinSnapshot.tradeable`` is false, so no feature may be
    computed from it and no order may depend on it.
    """

    malformed: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)

    def number(self, value: Any, name: str) -> float | None:
        try:
            return _opt_float(value, name)
        except MalformedField as exc:
            self.malformed.append(str(exc))
            return None

    def counts(self, block: Any, window: str) -> TxnCounts:
        try:
            return _txns(block, window)
        except MalformedField as exc:
            self.malformed.append(str(exc))
            return TxnCounts(buys=None, sells=None)

    def ladder(self, block: Any) -> PriceLadder:
        try:
            return _price_ladder(block)
        except MalformedField as exc:
            self.malformed.append(str(exc))
            return PriceLadder(m5=None, h1=None, h6=None, h24=None)

    @property
    def quality(self) -> DataQuality:
        if self.malformed:
            return DataQuality.QUARANTINED
        if self.degraded:
            return DataQuality.DEGRADED
        return DataQuality.OK

    @property
    def reason(self) -> str | None:
        parts = [*self.malformed, *self.degraded]
        return "; ".join(parts) or None


def _coin_snapshot(
    coin: CoinRef,
    pair: dict,
    pair_warning: str | None,
    *,
    receive_time: float,
    candles_5m: CandleSeries | None,
    candles_1h: CandleSeries | None,
    candle_reasons: Sequence[str],
) -> CoinSnapshot:
    acc = _Accumulator()
    acc.degraded.extend(candle_reasons)
    if pair_warning:
        acc.degraded.append(pair_warning)

    volume = _as_dict(pair.get("volume"))
    liquidity = _as_dict(pair.get("liquidity"))

    try:
        pool = _pool_ref(pair, coin.mint, trusted_quote=pair_warning is None)
    except MalformedField as exc:
        # A pool we cannot even identify is not a pool we may trade. The
        # identity is load-bearing (C8), so this is fatal for the coin rather
        # than a degraded field.
        acc.malformed.append(str(exc))
        pool = PoolRef(
            pair_address=str(pair.get("pairAddress") or ""),
            dex_id=str(pair.get("dexId") or ""),
            base_mint=coin.mint,
            quote_mint="",
            quote_symbol="?",
            created_at=None,
            trusted_quote=False,
        )

    return CoinSnapshot(
        symbol=coin.symbol,
        mint=coin.mint,
        price_usd=acc.number(pair.get("priceUsd"), "priceUsd"),
        liquidity_usd=acc.number(liquidity.get("usd"), "liquidity.usd"),
        volume_24h_usd=acc.number(volume.get("h24"), "volume.h24"),
        volume_1h_usd=acc.number(volume.get("h1"), "volume.h1"),
        fdv_usd=acc.number(pair.get("fdv"), "fdv"),
        price_change=acc.ladder(pair.get("priceChange")),
        txns_m5=acc.counts(pair.get("txns"), "m5"),
        txns_h1=acc.counts(pair.get("txns"), "h1"),
        txns_h24=acc.counts(pair.get("txns"), "h24"),
        pool=pool,
        # The provenance of *this* observation: when the DexScreener response
        # that produced it arrived. Not when the tick finished. ``event_time``
        # is None because the pair summary carries no "as of" field, and a copy
        # of ``receive_time`` would make a stale read look current — the exact
        # shape of audit C8.
        provenance=Provenance(
            source="dexscreener",
            receive_time=receive_time,
            event_time=None,
            source_time=None,
            quality=acc.quality,
            quality_reason=acc.reason,
        ),
        candles_5m=candles_5m,
        candles_1h=candles_1h,
        quality=acc.quality,
        quality_reason=acc.reason,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_pairs(
    coins: Sequence[CoinRef],
    params: MarketParams | None = None,
    *,
    client: httpx.Client | None = None,
    now: Callable[[], float] = time.time,
) -> dict[str, PoolRef]:
    """Map symbol -> selected :class:`PoolRef`. Used by startup validation.

    Returns the whole ``PoolRef`` rather than a bare address string because the
    caller needs ``trusted_quote`` and ``created_at`` to decide whether the mint
    is merely *present* or actually *tradeable* — and because a bare string is
    what let pool identity go unchecked in the first place.

    Deliberately does not touch GeckoTerminal: startup wants a fast, cheap
    answer to "do these mints exist and trade", and a missing candle series is
    not a reason to refuse to boot.
    """
    params = params or MarketParams()
    owned = client is None
    client = client or _new_client(params)
    try:
        grouped, _ = _fetch_pairs([c.mint for c in coins], params, client, now=now)
        resolved: dict[str, PoolRef] = {}
        for coin in coins:
            pair, warning = _best_pair(grouped.get(coin.mint.lower(), []))
            if pair is None:
                raise MarketDataError(
                    f"{coin.symbol}: DexScreener returned no tradeable pair for "
                    f"mint {coin.mint} - the mint is wrong, or the token has no "
                    "live pool."
                )
            resolved[coin.symbol] = _pool_ref(
                pair, coin.mint, trusted_quote=warning is None
            )
        return resolved
    finally:
        if owned:
            client.close()


def snapshot(
    coins: Sequence[CoinRef],
    params: MarketParams | None = None,
    *,
    client: httpx.Client | None = None,
    with_candles: bool = True,
    now: Callable[[], float] = time.time,
) -> MarketSnapshot:
    """Read every configured coin: one DexScreener call plus, by default, two
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
    ``signals.flow_brief`` reads only ``liquidity_usd`` and ``pool`` off it.

    On timestamps — this is the C8 fix. ``MarketSnapshot.ts`` is still the
    completion time and is still only for logging. Every ``CoinSnapshot``
    carries the receive time of the DexScreener response that produced it, and
    every ``CandleSeries`` carries the receive time of *its own* GeckoTerminal
    response, which on a three-coin read is up to ~7.5s later. Staleness must be
    asserted per observation, or via ``MarketSnapshot.oldest_age_seconds``,
    which is honest by construction. Nothing may use ``ts`` for freshness.

    ``now`` is injected so a test can freeze the clock; the closed-bar watermark
    depends on it, and a watermark you cannot control is a watermark you cannot
    test.
    """
    params = params or MarketParams()
    owned = client is None
    client = client or _new_client(params)
    try:
        grouped, receive_time = _fetch_pairs(
            [c.mint for c in coins], params, client, now=now
        )
        result: dict[str, CoinSnapshot] = {}
        gecko_calls = 0

        for coin in coins:
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
            series: dict[Timeframe, CandleSeries | None] = {
                Timeframe.M5: None,
                Timeframe.H1: None,
            }
            candle_reasons: list[str] = []

            # Deliberately skipped candles are not a degraded read, so nothing
            # is appended to ``candle_reasons`` here — the caller asked for
            # prices. A permanent false warning in front of every consumer is
            # how real warnings stop being read.
            wanted = (
                (
                    (Timeframe.M5, params.candles_5m),
                    (Timeframe.H1, params.candles_1h),
                )
                if with_candles
                else ()
            )
            for timeframe, limit in wanted:
                if gecko_calls and _GECKO_DELAY_SECONDS:
                    time.sleep(_GECKO_DELAY_SECONDS)
                gecko_calls += 1
                try:
                    series[timeframe] = _fetch_candles(
                        params, client, pool, timeframe, limit, now=now
                    )
                except (MarketDataError, httpx.HTTPError, ValidationError) as exc:
                    # Candles are enrichment, not the read itself. Losing them
                    # costs us technicals; crashing here costs us the tick, the
                    # stop-loss check and the mark-to-market.
                    candle_reasons.append(f"{timeframe.value} candles unavailable: {exc}")

            result[coin.symbol] = _coin_snapshot(
                coin,
                pair,
                pair_warning,
                receive_time=receive_time,
                candles_5m=series[Timeframe.M5],
                candles_1h=series[Timeframe.H1],
                candle_reasons=candle_reasons,
            )

        return MarketSnapshot(ts=now(), coins=result)
    finally:
        if owned:
            client.close()


def screen(
    snap: CoinSnapshot,
    params: SafetyScreenParams | None = None,
    *,
    now: float,
) -> SafetyVerdict:
    """The data-quality half of audit C9, as an explicit named screen.

    ``risk.py`` calls this and treats ``eligible=False`` as a hard veto on
    *entries*. It is deliberately not consulted for exits: a snapshot being
    untrustworthy is a reason not to buy and frequently a reason to sell, and
    one predicate must not govern both (see ``CoinSnapshot.tradeable``).

    Every rule fails closed. An absent input produces an entry in ``unknowns``
    and blocks, because the alternative — passing a screen you could not run —
    is precisely the missing-is-zero habit this whole remediation is about.

    What this does **not** do is the on-chain half of C9: mint and freeze
    authority, LP ownership and lock, holder and dev concentration, Token-2022
    transfer hooks/fees/permanent-delegate/default-frozen, and a sellability
    probe. Those need an RPC or a risk vendor, they are deferred, and they are
    enumerated in ``SafetyVerdict.deferred`` so the gap travels with the answer.
    Until they exist, universe expansion beyond hand-listed established tokens
    is not safe, which is exactly what the audit says.
    """
    params = params or SafetyScreenParams()
    vetoes: list[str] = []
    unknowns: list[str] = []
    notes: list[str] = []

    if snap.quality is DataQuality.QUARANTINED:
        vetoes.append(f"observation quarantined: {snap.quality_reason}")
    elif snap.quality is DataQuality.DEGRADED:
        notes.append(f"degraded: {snap.quality_reason}")

    if params.require_trusted_quote and not snap.pool.trusted_quote:
        vetoes.append(
            f"untrusted quote token {snap.pool.quote_symbol!r}: the price is "
            "derived from a token whose own value is unknown (the 4,900x "
            "fake-liquidity case)"
        )

    age = snap.age_seconds(now)
    if age > params.max_snapshot_age_seconds:
        vetoes.append(
            f"observation is {age:.0f}s old, limit {params.max_snapshot_age_seconds:.0f}s"
        )

    if snap.price_usd is None:
        unknowns.append("price_usd not reported")
    elif snap.price_usd <= 0:
        vetoes.append(f"non-positive price {snap.price_usd}")

    if snap.liquidity_usd is None:
        unknowns.append("liquidity_usd not reported")
    elif snap.liquidity_usd < params.min_liquidity_usd:
        vetoes.append(
            f"liquidity ${snap.liquidity_usd:,.0f} below minimum "
            f"${params.min_liquidity_usd:,.0f}"
        )

    if snap.volume_24h_usd is None:
        unknowns.append("volume_24h_usd not reported")
    elif snap.volume_24h_usd < params.min_volume_24h_usd:
        vetoes.append(
            f"24h volume ${snap.volume_24h_usd:,.0f} below minimum "
            f"${params.min_volume_24h_usd:,.0f}"
        )

    created = snap.pool.created_at
    if created is None:
        unknowns.append("pool creation time not reported")
    else:
        pool_age = now - created
        if pool_age < params.min_pool_age_seconds:
            vetoes.append(
                f"pool is {pool_age / 3600.0:.1f}h old, minimum "
                f"{params.min_pool_age_seconds / 3600.0:.1f}h"
            )

    # Liquidity-to-FDV. Not a valuation claim: it is the ratio between what the
    # market says the token is worth and what could actually be sold today.
    if snap.fdv_usd is None or snap.liquidity_usd is None:
        unknowns.append("liquidity/FDV ratio not computable")
    elif snap.fdv_usd > 0:
        ratio = snap.liquidity_usd / snap.fdv_usd
        if ratio < params.min_liquidity_to_fdv:
            vetoes.append(
                f"liquidity is {ratio * 100:.3f}% of FDV, minimum "
                f"{params.min_liquidity_to_fdv * 100:.3f}% — the valuation is "
                "not backed by exitable depth"
            )
        else:
            notes.append(f"liquidity/FDV {ratio * 100:.2f}%")

    return SafetyVerdict(
        symbol=snap.symbol,
        eligible=not vetoes and not unknowns,
        vetoes=tuple(vetoes),
        unknowns=tuple(unknowns),
        notes=tuple(notes),
    )


__all__ = [
    "CoinRef",
    "MalformedField",
    "MarketDataError",
    "MarketParams",
    "SafetyScreenParams",
    "SafetyVerdict",
    "resolve_pairs",
    "screen",
    "snapshot",
]
