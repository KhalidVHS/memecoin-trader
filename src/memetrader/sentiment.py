"""Reddit *attention*, not Reddit sentiment.

For memecoins, polarity is manufactured. A shill farm will produce bullish text
on demand and for a few dollars, which makes "73% of posts are positive" a
number with no information in it. What survives scrutiny is:

* **velocity** — is attention accelerating right now (``1h`` rate vs the ``24h``
  baseline rate; the model reads the *ratio*),
* **breadth** — how many distinct humans, versus one account posting forty
  times (``contributor_to_post_ratio``, the shill-farm detector),
* **novelty** — is this level of attention unusual *for this coin*
  (``mention_zscore_7d``).

So this module counts posts and counts people. ``polarity`` exists, is a
twelve-line keyword lexicon with no dependencies, and is flagged low-trust in
the prompt. There is deliberately no classifier and no ``transformers``.

Two sources, one shape:

``ArcticShiftProvider``
    Keyless community mirror of the Reddit archive. Always available, needs no
    credentials, and is therefore the path that can never be blocked on the
    user creating an OAuth app.
``PrawProvider``
    The official API. Preferred when ``cfg.sentiment.has_reddit_credentials``
    is true, because it is authoritative and fresher.

**Failure policy.** ``brief()`` returns ``None`` on total failure and never
raises. A missing sentiment brief is rendered to the model as *explicitly
unavailable* so it can discount its own confidence; a sentiment outage must
never fail a tick, and must never silently look like "nobody is talking about
this". Those are different claims and the type system keeps them apart:
``None`` means we could not find out, a brief with ``mention_velocity_1h == 0``
means we looked and it is quiet.

**On Reddit's Data API terms.** §3.2 prohibits deriving revenue from use of the
API and there is a 48-hour deletion recommendation for stored user content.
Both are aimed at resellers and are invisible at personal-research scale, but
they shape the design here: the on-disk cache holds **derived counts,
timestamps and salted author hashes only** — never post bodies, never post
titles, never plaintext usernames. The three ``top_posts`` titles exist in
memory for the current tick and are handed straight to the prompt; they are not
persisted, which is why a cache *hit* returns a brief with an empty
``top_posts`` and says so in ``degraded_reason``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

import httpx

from .config import CoinConfig, Config, SentimentConfig
from .http import make_client
from .types import SentimentBrief, TopPost

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARCTIC_SHIFT_BASE = "https://arctic-shift.photon-reddit.com/api"

# Verified live 2026-09-18: the server rejects any field it does not know
# (400 "'permalink' is not a valid field"), so this list is exactly what the
# endpoint will return. Asking for a subset keeps a 24h page of r/CryptoCurrency
# at ~9 KB instead of ~250 KB.
_ARCTIC_FIELDS = "id,created_utc,author,title,score,subreddit,num_comments,selftext"

# Verified live: 400 "'limit' must be between 1 and 100".
_ARCTIC_MAX_LIMIT = 100
_ARCTIC_MAX_PAGES = 6  # 600 posts per subreddit per window is plenty
_ARCTIC_SPACING_S = 0.6  # the host 422s with "Timeout. Maybe slow down a bit"

# How long one subreddit sweep is reused across coins within a single tick.
# Every coin in a tick shares the same lookback window, so re-pulling it per
# coin would triple the request count against a host that already rate-limits.
_MEMO_SECONDS = 60.0

CACHE_FILENAME = "sentiment_cache.json"
CACHE_VERSION = 1

# How far a source's index may trail live before the most recent hour stops
# being reportable. Measured against the newest post in the whole sweep, not
# the per-coin matches — a coin with no mentions tells you nothing about
# freshness. Arctic Shift was ~10h behind on a live check (2026-09-18), so this
# threshold is routinely crossed; PRAW queries live Reddit and never trips it.
# 15 minutes is one slow tick: a lag under that cannot hide a burst from the
# next decision.
_MAX_INDEX_LAG_SECONDS = 900.0

# Bots and tombstones are not contributors. Compared case-insensitively.
_NON_CONTRIBUTORS = frozenset({"[deleted]", "[removed]", "automoderator", "none", ""})

# How many completed hourly buckets we insist on before we will emit a z-score.
# One fetch observes ``lookback_hours`` (24) buckets at once, so this is reached
# on the second day of running. Below it we return ``None`` and say why rather
# than inventing a number out of one day of history.
MIN_BASELINE_HOURS = 48

# Titles are handed to the model verbatim; cap them so one pathological title
# cannot dominate the prompt budget.
_TITLE_MAX_CHARS = 200

_TOP_POSTS = 3

_SECONDS_PER_HOUR = 3600.0


class SourceUnavailable(RuntimeError):
    """Every configured source failed. Caught by ``brief()``, becomes ``None``."""


# ---------------------------------------------------------------------------
# Normalized post
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Post:
    """One post, normalized out of whichever vendor shape produced it.

    Exists for the duration of one tick and is never serialized. ``created_utc``
    is epoch **seconds** (both sources already use seconds; nothing here divides
    by 1000).
    """

    id: str
    created_utc: float
    author: str
    title: str
    body: str
    score: int
    subreddit: str

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.body}"

    def age_hours(self, now: float) -> float:
        return max(0.0, (now - self.created_utc) / _SECONDS_PER_HOUR)


# ---------------------------------------------------------------------------
# Alias matching
# ---------------------------------------------------------------------------


def alias_pattern(aliases: Iterable[str]) -> re.Pattern[str]:
    """Case-insensitive whole-word matcher for a coin's aliases.

    Naive substring matching is not a small inaccuracy here, it destroys the
    signal: ``"WIF" in text`` fires on *wife*, *swift*, *wifi*, *midwife* and
    every ``t.me/...wif...`` spam link, and r/CryptoCurrency is full of all of
    them. So each alias is fenced with explicit lookarounds.

    The fences exclude ASCII letters and digits but **not** ``$``, ``_`` or
    ``#``, because ``$WIF``, ``#WIF`` and ``WIF_army`` are exactly the mentions
    we want. Using lookarounds rather than ``\\b`` is what buys that asymmetry:
    ``\\b`` counts ``_`` as a word character and would drop ``WIF_army``.

    What this still lets through, honestly:

    * **Genuine homonyms.** "WIF" is also *Wallet Import Format* in Bitcoin
      contexts, and "popcat" is a meme that predates the token. Neither is
      distinguishable from the ticker without reading the post. A real hit from
      r/pumpfun during development: *"Imagine yourself being an ALIEN WIF HAT"*
      — "wif" as eye-dialect for "with". This is why ``top_posts`` is handed to
      the model raw: it can see that the mention is noise, and we cannot.
    * **Sentiment-negative attention.** "WIF is dead" counts as a mention. That
      is intentional — this module measures attention, not approval.
    * **Obfuscation.** ``W I F``, ``W1F`` and unicode homoglyphs are missed.
      Shill farms do use these; treat the counts as a floor, not a census.
    * **Link bodies.** An alias inside a URL path still matches.
    """
    parts = sorted({a.strip() for a in aliases if a and a.strip()}, key=len, reverse=True)
    if not parts:
        # No aliases is a config error, but returning a pattern that matches
        # nothing beats raising from inside a data path.
        return re.compile(r"(?!x)x")
    body = "|".join(re.escape(p) for p in parts)
    return re.compile(rf"(?<![0-9A-Za-z])(?:{body})(?![0-9A-Za-z])", re.IGNORECASE)


def matching_posts(posts: Iterable[Post], aliases: Iterable[str]) -> list[Post]:
    """Posts whose title or body mentions any alias as a whole word."""
    pattern = alias_pattern(aliases)
    return [p for p in posts if pattern.search(p.text)]


# ---------------------------------------------------------------------------
# Polarity — deliberately trivial, deliberately low-trust
# ---------------------------------------------------------------------------

_BULLISH = frozenset(
    """moon mooning pump pumping bullish bull long send sending ath breakout gem
    rocket green hodl accumulate undervalued bounce rally parabolic buying""".split()
)
_BEARISH = frozenset(
    """dump dumping bearish bear short rug rugged rugpull scam dead crash red
    exit rekt bleeding bagholder bagholding avoid dumped selling""".split()
)
_WORD = re.compile(r"[a-z]+")


def keyword_polarity(posts: Sequence[Post]) -> float | None:
    """-1..1 from a flat keyword count, or ``None`` when no keyword appears.

    This is a stopgap, not an analysis: no negation handling, no sarcasm, no
    weighting. It is here only because a coarse tilt is occasionally worth a
    sentence in the prompt, where it is labelled explicitly as low-trust. If you
    ever feel tempted to improve it, improve the *breadth* metrics instead —
    they are the ones that cost a shill farm money to fake.
    """
    bull = bear = 0
    for post in posts:
        for word in _WORD.findall(post.text.lower()):
            if word in _BULLISH:
                bull += 1
            elif word in _BEARISH:
                bear += 1
    if bull + bear == 0:
        return None
    return (bull - bear) / (bull + bear)


# ---------------------------------------------------------------------------
# Cache — counts only, never content
# ---------------------------------------------------------------------------


def cache_path(cfg: Config):
    return cfg.data_dir / CACHE_FILENAME


def _author_hash(author: str) -> str:
    """One-way, non-reversible handle for an author.

    We only ever need "is this the same person as that other post", never "who".
    Hashing at the boundary means a plaintext username cannot reach disk even by
    accident. The digest is short because collisions across ~600 posts are
    irrelevant to a contributor *count*.
    """
    return hashlib.blake2b(author.strip().lower().encode("utf-8"), digest_size=8).hexdigest()


def load_cache(cfg: Config) -> dict[str, Any]:
    path = cache_path(cfg)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"version": CACHE_VERSION, "symbols": {}}
    except Exception as exc:  # corrupt cache is never worth failing a tick over
        log.warning("sentiment cache at %s unreadable (%s); starting fresh", path, exc)
        return {"version": CACHE_VERSION, "symbols": {}}
    if not isinstance(raw, dict) or raw.get("version") != CACHE_VERSION:
        return {"version": CACHE_VERSION, "symbols": {}}
    raw.setdefault("symbols", {})
    return raw


def save_cache(cfg: Config, cache: dict[str, Any]) -> None:
    path = cache_path(cfg)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        log.warning("could not write sentiment cache to %s: %s", path, exc)


def _brief_to_cache(brief: SentimentBrief) -> dict[str, Any]:
    """Serialize a brief for disk, dropping ``top_posts`` entirely.

    See the module docstring: titles are user content and stay in memory. The
    cached brief is counts and timestamps, which is what the TTL is protecting.
    """
    return {
        "symbol": brief.symbol,
        "ts": brief.ts,
        "source": brief.source,
        "mention_velocity_1h": brief.mention_velocity_1h,
        "mention_velocity_24h": brief.mention_velocity_24h,
        "mention_zscore_7d": brief.mention_zscore_7d,
        "unique_contributors_24h": brief.unique_contributors_24h,
        "contributor_to_post_ratio": brief.contributor_to_post_ratio,
        "polarity": brief.polarity,
        "degraded_reason": brief.degraded_reason,
    }


def _brief_from_cache(entry: dict[str, Any]) -> SentimentBrief:
    reasons = [r for r in (entry.get("degraded_reason"), "served from cache; top posts are not persisted") if r]
    return SentimentBrief(
        symbol=str(entry["symbol"]),
        ts=float(entry["ts"]),
        source=entry["source"],
        # ``None`` survives the round trip: a cached "hour not indexed" must not
        # rehydrate as a confident zero.
        mention_velocity_1h=(
            None if entry.get("mention_velocity_1h") is None else float(entry["mention_velocity_1h"])
        ),
        mention_velocity_24h=(
            None
            if entry.get("mention_velocity_24h") is None
            else float(entry["mention_velocity_24h"])
        ),
        mention_zscore_7d=(
            None if entry.get("mention_zscore_7d") is None else float(entry["mention_zscore_7d"])
        ),
        unique_contributors_24h=int(entry["unique_contributors_24h"]),
        contributor_to_post_ratio=(
            None
            if entry.get("contributor_to_post_ratio") is None
            else float(entry["contributor_to_post_ratio"])
        ),
        top_posts=(),
        polarity=None if entry.get("polarity") is None else float(entry["polarity"]),
        degraded_reason="; ".join(reasons),
    )


# ---------------------------------------------------------------------------
# The arithmetic — one implementation, shared by every provider
# ---------------------------------------------------------------------------


def _hour_bucket(ts: float) -> str:
    return str(int(ts // 3600))


def _update_history(
    history: dict[str, int],
    posts: Sequence[Post],
    cfg: SentimentConfig,
    now: float,
    observed_end: float | None = None,
) -> dict[str, int]:
    """Fold this tick's observation into the rolling hourly history.

    Each fetch sees the whole ``lookback_hours`` window, so we *overwrite* every
    bucket inside it rather than adding. That makes the history self-healing:
    a tick that was skipped, rate-limited or run on a laptop that was asleep
    gets backfilled by the next successful fetch, and running twice in one
    minute cannot double-count.

    ``observed_end`` bounds how far the zero-seeding may reach. Seeding up to
    ``now`` when the source's index stops 10 hours short writes ten fabricated
    "zero mentions this hour" buckets into a 7-day baseline, which drags the
    mean down and inflates every later z-score. The self-healing overwrite does
    eventually correct them — but only while the lag stays under
    ``lookback_hours``, and a baseline should not depend on that.

    Only integers land here. No ids, no authors, no text.
    """
    window_start = now - cfg.lookback_hours * _SECONDS_PER_HOUR
    end = now if observed_end is None else min(observed_end, now)
    observed: dict[str, int] = {}
    # Seed every *observed* hour in the window at zero — an hour with no
    # mentions is an observation, and dropping it would bias the baseline
    # upward. An hour the source has not reached is not an observation.
    start_bucket = int(window_start // 3600)
    end_bucket = int(end // 3600)
    for bucket in range(start_bucket, end_bucket + 1):
        observed[str(bucket)] = 0
    for post in posts:
        if post.created_utc < window_start:
            continue
        observed[_hour_bucket(post.created_utc)] = observed.get(_hour_bucket(post.created_utc), 0) + 1

    merged = dict(history)
    merged.update(observed)

    cutoff = int((now - cfg.baseline_days * 24 * _SECONDS_PER_HOUR) // 3600)
    return {k: int(v) for k, v in merged.items() if k.isdigit() and int(k) >= cutoff}


def _zscore(history: dict[str, int], current_rate: float, now: float) -> tuple[float | None, str | None]:
    """Z-score of the current hourly rate against completed historical hours.

    The current hour is excluded from the baseline: it is partial, so including
    it would drag the mean toward the very value we are testing.

    ``current_rate`` is a *rolling* hour (``now-3600`` to ``now``) while the
    baseline buckets are wall-clock hours. Both are in mentions-per-hour so the
    comparison is sound, but the rolling window is why the current bucket and
    ``mention_velocity_1h`` will usually differ: a post 12 minutes before a
    clock-aligned ``now`` is inside the rolling hour and inside the *previous*
    bucket. That is the correct behaviour — a rolling window is what makes a
    burst visible at 12:05 instead of at 13:00.
    """
    current_bucket = _hour_bucket(now)
    baseline = [v for k, v in history.items() if k != current_bucket]
    if len(baseline) < MIN_BASELINE_HOURS:
        return None, (
            f"no 7d baseline yet ({len(baseline)}/{MIN_BASELINE_HOURS} hourly "
            "buckets accumulated); z-score will appear once history builds"
        )
    mean = sum(baseline) / len(baseline)
    variance = sum((v - mean) ** 2 for v in baseline) / len(baseline)
    stdev = math.sqrt(variance)
    if stdev < 1e-9:
        # Flat baseline, usually all zeros. A z-score would be ±inf, which is
        # worse than admitting we cannot scale this.
        return None, "baseline has zero variance (no mentions in the baseline window)"
    return (current_rate - mean) / stdev, None


def build_brief(
    coin: CoinConfig,
    cfg: Config,
    posts: Sequence[Post],
    source: str,
    now: float,
    *,
    history: dict[str, int] | None = None,
    extra_reasons: Sequence[str] = (),
    observed_through: float | None = None,
) -> SentimentBrief:
    """Turn a window of already-matched posts into a ``SentimentBrief``.

    Every provider funnels through here so the arithmetic exists once. ``posts``
    must already be filtered to ``coin``'s aliases; this function does not match.

    ``history`` is the rolling hourly-count dict for this symbol and is
    **mutated in place** (the caller owns persisting it). Pass ``None`` to skip
    baseline accumulation entirely, in which case ``mention_zscore_7d`` is
    ``None`` with a reason.

    ``observed_through`` is the timestamp the source's index actually reaches —
    not the timestamp we asked for. These differ, and the gap is not small:
    Arctic Shift was running ~10 hours behind live Reddit when this was
    written. Everything after it is *unobserved*, and unobserved is not zero.
    Passing ``None`` asserts the source is live through ``now``.

    Zero posts inside the observed window is a perfectly good answer and
    produces a valid brief with zero velocity — "nobody is talking about this"
    is information. Zero posts because the hour has not been indexed yet is not
    information, and comes back as ``None``.
    """
    scfg = cfg.sentiment
    window_hours = float(scfg.lookback_hours)
    window_start = now - window_hours * _SECONDS_PER_HOUR

    # Never trust a source to be ahead of the clock; ``min`` also absorbs the
    # vendor-bug case of a future timestamp.
    observed_end = now if observed_through is None else min(observed_through, now)
    lag_seconds = max(now - observed_end, 0.0)

    reasons: list[str] = list(extra_reasons)
    if lag_seconds > _MAX_INDEX_LAG_SECONDS:
        reasons.append(
            f"source index is {lag_seconds / _SECONDS_PER_HOUR:.1f}h behind live; "
            f"counts cover only through {time.strftime('%H:%M UTC', time.gmtime(observed_end))}"
        )

    # The window is closed at both ends. The upper bound matters: a clock skew
    # or a vendor bug that hands back a future timestamp would otherwise create
    # an hourly bucket beyond ``now`` which the retention trim (a lower bound)
    # can never remove, quietly poisoning the baseline for every later run.
    in_window = [p for p in posts if window_start <= p.created_utc <= observed_end]

    # The rolling hour only means something if the source has indexed it. With a
    # 10h lag it never has, and every tick would otherwise report a confident
    # 0.0 mentions/hour — the single most misleading number this module could
    # emit, because flat attention reads bearish and would be wrong every time.
    if lag_seconds > _MAX_INDEX_LAG_SECONDS:
        velocity_1h = None
    else:
        last_hour = [p for p in in_window if p.created_utc >= now - _SECONDS_PER_HOUR]
        velocity_1h = float(len(last_hour))  # one hour of data, so count == rate

    # Average over the hours actually observed, not the hours requested —
    # dividing a 14h count by 24h understates attention by 40% and the error
    # grows with the lag.
    observed_hours = max(observed_end - window_start, 0.0) / _SECONDS_PER_HOUR
    velocity_24h = len(in_window) / observed_hours if observed_hours > 0 else None

    contributors = {
        _author_hash(p.author)
        for p in in_window
        if p.author.strip().lower() not in _NON_CONTRIBUTORS
    }
    unique_contributors = len(contributors)
    # Denominator is *all* matched posts, including bot/deleted ones: a wall of
    # deleted posts is exactly the pattern this ratio is meant to expose.
    ratio = (unique_contributors / len(in_window)) if in_window else None

    if history is None:
        zscore = None
        reasons.append("no cross-run baseline available (called without a cache)")
    else:
        merged = _update_history(history, in_window, scfg, now, observed_end)
        history.clear()  # mutate in place; the caller holds the reference
        history.update(merged)
        if velocity_1h is None:
            # Nothing to score. The baseline is still updated above, so the
            # history keeps building for whenever the index does catch up.
            zscore = None
            reasons.append("no z-score: the current hour is not indexed yet")
        else:
            zscore, why = _zscore(history, velocity_1h, now)
            if why:
                reasons.append(why)

    top = tuple(
        TopPost(
            title=p.title[:_TITLE_MAX_CHARS],
            score=p.score,
            age_hours=round(p.age_hours(now), 2),
            subreddit=p.subreddit,
        )
        for p in sorted(in_window, key=lambda p: p.score, reverse=True)[:_TOP_POSTS]
    )

    return SentimentBrief(
        symbol=coin.symbol,
        ts=now,
        source=source,  # type: ignore[arg-type]
        mention_velocity_1h=velocity_1h,
        mention_velocity_24h=velocity_24h,
        mention_zscore_7d=zscore,
        unique_contributors_24h=unique_contributors,
        contributor_to_post_ratio=ratio,
        top_posts=top,
        polarity=keyword_polarity(in_window),
        degraded_reason="; ".join(reasons) or None,
    )


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class SentimentProvider(Protocol):
    """The seam. Anything with this shape can feed ``brief()``."""

    def fetch(self, coin: CoinConfig, cfg: Config) -> SentimentBrief | None: ...


def sweep_observed_through(provider: object, sweep: Sequence[Post], now: float) -> float | None:
    """How far this provider's index actually reaches.

    ``None`` means "live through ``now``". For a lagging source it is the newest
    post in the **whole sweep** — deliberately computed before the per-coin
    alias filter, because a coin nobody mentioned would otherwise look like a
    stale index and have its (genuine) zero suppressed.

    An empty sweep from a lagging source is the one case that stays ``None``:
    zero posts across every subreddit is a broken read, and it is already
    reported through ``failures``; inventing a lag from it would add a second,
    wrong explanation.
    """
    if not getattr(provider, "index_lags", False):
        return None
    return max((p.created_utc for p in sweep), default=None)


@runtime_checkable
class PostSource(Protocol):
    """The *richer* seam, which both real providers also implement.

    ``fetch()`` alone cannot produce ``mention_zscore_7d``, because the baseline
    lives in a cache file the provider deliberately knows nothing about. A
    provider that can hand back raw posts lets ``brief()`` fold the exact
    per-hour counts into the persisted history instead of trying to reconstruct
    a distribution from two aggregate velocities — which would mean inventing
    numbers, the one thing this module refuses to do.

    ``fetch()`` remains the contract; this is an optional upgrade. A provider
    implementing only ``fetch()`` works fine and simply reports no baseline.

    Deliberately method-only: ``runtime_checkable`` protocols raise ``TypeError``
    on ``isinstance`` if they declare data members, and ``brief()`` needs the
    ``isinstance`` check. Implementations also carry a ``source`` string and an
    ``index_lags`` flag, both read via ``getattr``.

    ``index_lags`` says whether this source's index can trail live. It must be
    declared rather than inferred, because the only thing measurable from the
    outside — the newest post in the sweep — means "how fresh the index is"
    only for a source that sweeps whole subreddits (Arctic Shift). For a source
    that runs a per-coin query (PRAW), an empty or old result set means the coin
    is quiet, and treating that as index lag would suppress the true reading.
    """

    def posts(
        self, coin: CoinConfig, cfg: Config, now: float
    ) -> tuple[list[Post], list[str]]:
        """Return ``(unfiltered posts in window, names of failed sources)``."""
        ...


class ArcticShiftProvider:
    """Keyless Reddit archive mirror. The fallback that is always available.

    Verified live 2026-09-18 against ``GET /api/posts/search``:

    * envelope is ``{"data": [...]}``  on success and
      ``{"data": null, "error": "..."}`` on failure, always HTTP-coded too;
    * ``data`` items are raw Reddit submission objects (``created_utc`` in
      seconds, ``author`` as a plain username string, ``selftext``, ``score``);
    * ``limit`` is capped at 100;
    * ``fields`` accepts a comma-separated whitelist and 400s on unknown names;
    * ``after``/``before`` are epoch seconds and ``sort`` takes ``desc``/``asc``.

    Note what this class does **not** do: it never sends the ``query``
    parameter. Server-side full-text search on this host is both fragile
    (repeated 422 ``"Timeout. Maybe slow down a bit"`` on
    ``subreddit``+``query``) and lossy (it returned zero rows for terms that are
    demonstrably present in the same window). Pulling the subreddit window and
    matching locally is one extra page of JSON and gives us control of the
    word-boundary rule, which is the part that actually determines signal
    quality.
    """

    source = "arctic_shift"
    #: An archive mirror, so its index trails live Reddit — by ~10h on a live
    #: check on 2026-09-18. It sweeps whole subreddits, so the newest post in a
    #: sweep is a true measure of that lag.
    index_lags = True

    def __init__(self, client: httpx.Client | None = None, base: str = ARCTIC_SHIFT_BASE) -> None:
        self._client = client
        self._owns_client = client is None
        self._base = base.rstrip("/")
        self._memo: tuple[float, list[Post], list[str]] | None = None

    def _http(self, cfg: Config) -> httpx.Client:
        if self._client is None:
            # http.make_client verifies against the OS trust store rather than
            # certifi; arctic-shift fails CERTIFICATE_VERIFY_FAILED otherwise on
            # any TLS-inspecting corporate network. Reddit asks for a descriptive
            # UA, which overrides the browser default make_client sends.
            self._client = make_client(
                max(cfg.data.http_timeout_seconds, 30.0),
                {"User-Agent": cfg.sentiment.reddit_user_agent},
            )
        return self._client

    def _page(self, cfg: Config, subreddit: str, after: int, before: int) -> list[dict[str, Any]]:
        resp = self._http(cfg).get(
            f"{self._base}/posts/search",
            params={
                "subreddit": subreddit,
                "after": str(after),
                "before": str(before),
                "limit": str(_ARCTIC_MAX_LIMIT),
                "sort": "desc",
                "fields": _ARCTIC_FIELDS,
            },
        )
        if resp.status_code != 200:
            detail = ""
            try:
                detail = str(resp.json().get("error"))
            except Exception:
                detail = resp.text[:120]
            raise RuntimeError(f"arctic-shift {resp.status_code}: {detail}")
        data = resp.json().get("data")
        return data if isinstance(data, list) else []

    def posts(
        self, coin: CoinConfig, cfg: Config, now: float
    ) -> tuple[list[Post], list[str]]:
        """All posts in the lookback window across every configured subreddit.

        ``coin`` is unused: this source pulls the whole subreddit window once
        and lets ``matching_posts`` do the per-coin filtering, so three coins
        cost one sweep rather than three.

        Returns ``(posts, failures)``. A subreddit that errors is skipped and
        named in ``failures`` so the brief can be marked degraded rather than
        quietly reporting a lower mention count.
        """
        # One sweep serves every coin: the window is identical for all of them
        # and this host rate-limits hard, so re-pulling it per coin would triple
        # the request count for identical bytes.
        if self._memo is not None and abs(now - self._memo[0]) < _MEMO_SECONDS:
            return list(self._memo[1]), list(self._memo[2])

        after = int(now - cfg.sentiment.lookback_hours * _SECONDS_PER_HOUR)
        out: dict[str, Post] = {}
        failures: list[str] = []
        for subreddit in cfg.sentiment.subreddits:
            cursor = int(now) + 1
            try:
                for _ in range(_ARCTIC_MAX_PAGES):
                    raw = self._page(cfg, subreddit, after, cursor)
                    for item in raw:
                        post = _post_from_arctic(item)
                        if post is not None:
                            out[post.id] = post
                    if len(raw) < _ARCTIC_MAX_LIMIT:
                        break
                    oldest = min(int(i.get("created_utc", cursor)) for i in raw)
                    if oldest >= cursor or oldest <= after:
                        break
                    cursor = oldest
                    time.sleep(_ARCTIC_SPACING_S)
            except Exception as exc:
                log.warning("arctic-shift: r/%s failed: %s", subreddit, exc)
                failures.append(subreddit)
            time.sleep(_ARCTIC_SPACING_S)

        if failures and len(failures) == len(cfg.sentiment.subreddits):
            raise SourceUnavailable(
                f"arctic-shift returned nothing for any of {len(failures)} subreddits"
            )
        self._memo = (now, list(out.values()), list(failures))
        return list(out.values()), failures

    def fetch(self, coin: CoinConfig, cfg: Config) -> SentimentBrief | None:
        """Standalone brief, with no cross-run baseline.

        ``brief()`` normally goes through ``posts()`` instead so it can maintain
        the persisted hourly history; this path exists so the class satisfies
        ``SentimentProvider`` on its own.
        """
        now = time.time()
        raw_posts, failures = self.posts(coin, cfg, now)
        return build_brief(
            coin,
            cfg,
            matching_posts(raw_posts, coin.aliases),
            self.source,
            now,
            history=None,
            extra_reasons=_failure_reasons(failures),
            observed_through=sweep_observed_through(self, raw_posts, now),
        )

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None


def _post_from_arctic(item: dict[str, Any]) -> Post | None:
    """Normalize one Arctic Shift row. Returns ``None`` for unusable rows."""
    try:
        return Post(
            id=str(item.get("id") or ""),
            created_utc=float(item["created_utc"]),
            author=str(item.get("author") or "[deleted]"),
            title=str(item.get("title") or ""),
            body=str(item.get("selftext") or ""),
            score=int(item.get("score") or 0),
            subreddit=str(item.get("subreddit") or ""),
        )
    except (KeyError, TypeError, ValueError):
        return None


class PrawProvider:
    """The official Reddit API. Primary path whenever credentials exist.

    PRAW 8.0.3 notes (checked against the installed package, not 7.x memory):

    * ``Subreddit.search(query, *, sort=, syntax=, time_filter=, **kwargs)`` —
      ``sort``/``syntax``/``time_filter`` are **keyword-only** in 8.x. The 7.x
      habit of ``search(q, "new", "lucene", "day")`` is a ``TypeError`` now.
    * ``Reddit.__init__(site_name=None, *, config_interpolation=None,
      requestor_class=None, requestor_kwargs=None, **config_settings)`` —
      ``site_name`` is the only positional parameter; credentials still arrive
      as keywords, so the 7.x construction call is unchanged.
    * ``reddit.subreddit`` is an *instance* attribute in 8.x, not a class
      method, so ``hasattr(praw.Reddit, "subreddit")`` is now ``False``. Only
      matters if you were duck-typing against the class.
    * Listing generators are typed ``**Unpack[ListingGeneratorKwargs]``;
      ``limit=`` still works.
    * Riding on prawcore 4.0.0.

    Subreddits are queried as one ``a+b+c`` multireddit, which is a single
    search per coin instead of one per subreddit. Reddit's own tokenizer is
    treated as a *prefilter* only — every result is re-checked against the
    word-boundary pattern locally, because Reddit stems and will happily return
    "wife" for a "wif" query.
    """

    source = "praw"
    #: Queries live Reddit, and its result set is already narrowed to one coin —
    #: so an old newest-post means the coin is quiet, not that the index is
    #: stale. Inferring lag from it would silently suppress real readings.
    index_lags = False

    def __init__(self, reddit: Any | None = None) -> None:
        self._reddit = reddit

    def _client(self, cfg: Config) -> Any:
        if self._reddit is None:
            import praw  # imported lazily: the keyless path must not need it

            self._reddit = praw.Reddit(
                client_id=cfg.sentiment.reddit_client_id,
                client_secret=cfg.sentiment.reddit_client_secret,
                user_agent=cfg.sentiment.reddit_user_agent,
                check_for_updates=False,
            )
            # Read-only is the default for an app-only script grant, but say it
            # out loud — nothing here should ever be able to post.
            self._reddit.read_only = True
        return self._reddit

    @staticmethod
    def _time_filter(lookback_hours: int) -> str:
        if lookback_hours <= 24:
            return "day"
        if lookback_hours <= 24 * 7:
            return "week"
        return "month"

    def posts(
        self, coin: CoinConfig, cfg: Config, now: float
    ) -> tuple[list[Post], list[str]]:
        scfg = cfg.sentiment
        if not scfg.subreddits:
            raise SourceUnavailable("no subreddits configured")
        multi = "+".join(scfg.subreddits)
        # Quoted terms OR'd together. Reddit's index is only a prefilter here —
        # every hit is re-checked locally by ``matching_posts``.
        query = " OR ".join(f'"{a}"' for a in coin.aliases)
        subreddit = self._client(cfg).subreddit(multi)
        results = subreddit.search(
            query,
            # PRAW 8: these three are keyword-only. Passing them positionally,
            # as 7.x allowed, is a TypeError.
            sort="new",
            syntax="lucene",
            time_filter=self._time_filter(scfg.lookback_hours),
            limit=None,
        )
        out: dict[str, Post] = {}
        for submission in results:
            post = _post_from_praw(submission)
            if post is not None:
                out[post.id] = post
        return list(out.values()), []

    def fetch(self, coin: CoinConfig, cfg: Config) -> SentimentBrief | None:
        """Standalone brief, with no cross-run baseline. See the note on
        ``ArcticShiftProvider.fetch``."""
        now = time.time()
        found, failures = self.posts(coin, cfg, now)
        return build_brief(
            coin,
            cfg,
            matching_posts(found, coin.aliases),
            self.source,
            now,
            history=None,
            extra_reasons=_failure_reasons(failures),
        )


def _post_from_praw(submission: Any) -> Post | None:
    """Normalize one PRAW ``Submission``.

    ``getattr`` throughout rather than attribute access: PRAW objects are lazy
    and a network hiccup on attribute resolution should cost us one post, not
    the tick. It also makes the object trivially stubbable in tests.
    """
    try:
        author = getattr(submission, "author", None)
        name = getattr(author, "name", None) or (str(author) if author else "[deleted]")
        subreddit = getattr(submission, "subreddit", "")
        return Post(
            id=str(getattr(submission, "id", "") or ""),
            created_utc=float(getattr(submission, "created_utc", 0.0)),
            author=str(name),
            title=str(getattr(submission, "title", "") or ""),
            body=str(getattr(submission, "selftext", "") or ""),
            score=int(getattr(submission, "score", 0) or 0),
            subreddit=str(getattr(subreddit, "display_name", None) or subreddit or ""),
        )
    except Exception:
        return None


def default_provider(cfg: Config) -> SentimentProvider:
    """PRAW when credentials exist, Arctic Shift otherwise."""
    if cfg.sentiment.has_reddit_credentials:
        return PrawProvider()
    return ArcticShiftProvider()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def brief(
    coin: CoinConfig,
    cfg: Config,
    *,
    provider: SentimentProvider | None = None,
    now: float | None = None,
    cache: dict[str, Any] | None = None,
) -> SentimentBrief | None:
    """One coin's attention brief, or ``None`` if we could not find out.

    Checks the on-disk TTL cache first, then the provider. **Never raises.**

    ``cache`` lets a caller (``briefs()``) load and save the cache file once for
    a whole sweep instead of once per coin; pass ``None`` for standalone use.
    """
    scfg = cfg.sentiment
    if not scfg.enabled:
        log.info("sentiment disabled in config; %s brief is unavailable", coin.symbol)
        return None

    now = time.time() if now is None else now
    owns_cache = cache is None
    cache = load_cache(cfg) if cache is None else cache
    entry = cache.setdefault("symbols", {}).setdefault(coin.symbol, {})

    cached = entry.get("brief")
    fetched_at = entry.get("fetched_at")
    if cached and isinstance(fetched_at, (int, float)):
        if now - float(fetched_at) < scfg.cache_ttl_seconds:
            log.debug(
                "%s: sentiment cache hit (%.0fs old, ttl %ss)",
                coin.symbol,
                now - float(fetched_at),
                scfg.cache_ttl_seconds,
            )
            try:
                return _brief_from_cache(cached)
            except Exception as exc:
                log.warning("%s: unusable cached brief (%s); refetching", coin.symbol, exc)

    provider = provider or default_provider(cfg)

    # The persisted hourly history for this symbol. ``build_brief`` mutates it
    # in place with this tick's exact per-hour counts; we write it straight back
    # into the cache entry afterwards.
    history = {
        k: int(v)
        for k, v in (entry.get("hourly") or {}).items()
        if str(k).isdigit() and isinstance(v, (int, float))
    }

    # A broad catch is correct here and is the whole contract of this module:
    # httpx, praw, prawcore, JSON decoding, DNS and a TLS-inspecting corporate
    # proxy can each raise something different, and *none* of them justify
    # failing a trading tick. Sentiment is one evidence stream out of several;
    # losing it must degrade the prompt, not stop it. Anything unexpected is
    # logged with a traceback (at DEBUG) so it stays debuggable.
    try:
        if isinstance(provider, PostSource):
            found, failures = provider.posts(coin, cfg, now)
            result = build_brief(
                coin,
                cfg,
                matching_posts(found, coin.aliases),
                getattr(provider, "source", "arctic_shift"),
                now,
                history=history,
                extra_reasons=_failure_reasons(failures),
                observed_through=sweep_observed_through(provider, found, now),
            )
        else:
            # A provider that only speaks ``fetch()`` owns its own arithmetic,
            # including whatever it decided about the baseline. We do not
            # second-guess it and we do not fabricate history for it.
            result = provider.fetch(coin, cfg)
    except Exception as exc:
        log.warning(
            "%s: sentiment provider %s failed: %s",
            coin.symbol,
            type(provider).__name__,
            exc,
            exc_info=log.isEnabledFor(logging.DEBUG),
        )
        return None

    if result is None:
        log.info("%s: sentiment provider returned no brief", coin.symbol)
        return None

    entry["hourly"] = history
    entry["brief"] = _brief_to_cache(result)
    entry["fetched_at"] = now

    if owns_cache:
        save_cache(cfg, cache)
    return result


def _failure_reasons(failures: Sequence[str]) -> list[str]:
    return [f"sources unreachable: {', '.join(failures)}"] if failures else []


def briefs(cfg: Config, **kw: Any) -> dict[str, SentimentBrief | None]:
    """Every configured coin's brief, keyed by symbol.

    One cache read and one cache write for the whole sweep, and one provider
    instance shared across coins so the HTTP connection (and the Arctic Shift
    rate-limit spacing) is reused. Coins are independent: one failing produces a
    ``None`` for that symbol only.
    """
    cache = load_cache(cfg)
    provider = kw.pop("provider", None) or (default_provider(cfg) if cfg.sentiment.enabled else None)
    out: dict[str, SentimentBrief | None] = {}
    try:
        for coin in cfg.coins:
            out[coin.symbol] = brief(coin, cfg, provider=provider, cache=cache, **kw)
    finally:
        save_cache(cfg, cache)
        close = getattr(provider, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    return out


__all__ = [
    "ARCTIC_SHIFT_BASE",
    "ArcticShiftProvider",
    "MIN_BASELINE_HOURS",
    "Post",
    "PostSource",
    "PrawProvider",
    "SentimentProvider",
    "default_provider",
    "SourceUnavailable",
    "alias_pattern",
    "brief",
    "briefs",
    "build_brief",
    "sweep_observed_through",
    "cache_path",
    "keyword_polarity",
    "load_cache",
    "matching_posts",
    "save_cache",
]
