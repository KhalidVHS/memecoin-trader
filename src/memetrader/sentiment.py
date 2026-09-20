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

That rule applies to *every* count, not just the one it was first written for.
``build_brief``'s ``sweep_size`` is what carries it: a sweep that came back
empty produces ``None`` for both velocities, no z-score and no contributor
count, because zero posts read from five subreddits cannot distinguish silence
from a failed read. Every field on ``SentimentBrief`` that reports a quantity is
now optional for that reason, so no consumer can be handed a fabricated zero.

**On the size of a zero.** Attention on these coins is sparse enough that a
zero usually means very little. Across the five configured subreddits, the
whole 90 days ending 2026-09-19 held 23 BONK submissions, 3 popcat and 1
dogwifhat, with the newest BONK submission 14.7 days old — about 0.26 BONK
posts a day, and a tenth of that for the other two. A 24-hour window with no
mentions is therefore the ordinary outcome rather than a signal, so a brief
reporting one says how many items it swept to get there.

**Comments are swept alongside submissions** (``sentiment.include_comments``,
default on) because that is where the chatter actually is. Measured across the
five configured subreddits for the 24h ending 2026-09-20 00:13 UTC: 136
submissions against at least 1008 comments, a 7.4:1 ratio, and 103 unique
authors against 498. Be honest about what that buys, though — over the same
window the alias matcher found 0 BONK / 0 WIF / 0 POPCAT mentions in
submissions and 1 / 0 / 0 in comments. It is a 7.4x larger *denominator* and a
0 → 1 change in the numerator, not a 7.4x increase in signal. The value is that
it makes a reported zero mean something: 1008 items read and none of them
mentioned the coin is a measurement, 136 is barely a sample.

**On Reddit's Data API terms.** §3.2 prohibits deriving revenue from use of the
API and there is a 48-hour deletion recommendation for stored user content.
Both are aimed at resellers and are invisible at personal-research scale, but
they shape the design here: the on-disk cache holds **derived counts,
timestamps and salted author hashes only** — never post bodies, never comment
bodies, never post titles, never plaintext usernames. A comment body is user
content on exactly the same terms as a submission; adding comments to the sweep
changed nothing about that rule. The three ``top_posts`` titles exist in
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
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

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

# ``GET /api/comments/search`` has the *same envelope* as /posts/search and a
# *different* field whitelist. Verified live 2026-09-19 by probing each name on
# its own. Accepted (13): id, created_utc, author, body, score, subreddit,
# link_id, parent_id, author_fullname, subreddit_id, distinguished,
# retrieved_on, author_flair_text. Rejected with 400 "'X' is not a valid
# field": permalink, title, selftext, num_comments, ups, controversiality,
# is_submitter, edited, gilded, stickied, total_awards_received, body_html,
# score_hidden.
#
# Read that rejected list before reaching for _ARCTIC_FIELDS here: ``title``,
# ``selftext`` and ``num_comments`` are all valid on /posts/search and all 400
# on this one. Sharing a single constant would not degrade the comment sweep,
# it would take the endpoint down entirely, for every subreddit, silently
# (every request raises, every subreddit lands in ``failures``).
#
# ``link_id`` is requested but unused today. It is the cheapest field that
# makes a comment traceable back to its submission, and ``parent_id ==
# link_id`` is how you tell a top-level comment from a reply — kept because
# dropping it costs nothing and re-adding it costs a live probe.
_ARCTIC_COMMENT_FIELDS = "id,created_utc,author,body,score,subreddit,link_id"

# Verified live: 400 "'limit' must be between 1 and 100".
_ARCTIC_MAX_LIMIT = 100

# 600 posts per subreddit per window, and nowhere near binding at observed
# volumes: on 2026-09-19 a 24h window returned 72 rows for r/CryptoCurrency, 34
# for r/solana, 24 for r/pumpfun, 10 for r/CryptoMoonShots and 3 for
# r/SatoshiStreetBets — every page under _ARCTIC_MAX_LIMIT, so the walk stops
# on page one and the cursor logic never runs. Kept at 6 for the days that do
# exceed it: r/CryptoCurrency hit the 100-row cap on 4 of the 9 days ending
# 2026-09-18.
#
# Comments blow straight through it — r/CryptoCurrency alone ran 7 pages of the
# 2026-09-19 24h window before the walk died, r/solana 2 — but raising it does
# not help. See _ARCTIC_SPACING_S.
_ARCTIC_MAX_PAGES = 6

# The host answers some requests with HTTP 422 {"data": null, "error":
# "Timeout. Maybe slow down a bit"}. **This is not rate limiting**, despite
# what the message says, and 2026-09-19 was spent establishing that:
#
# * 12 back-to-back comment pages at this spacing returned 12/12 x 200;
# * a failing request reproduces on the *first cold request of a fresh
#   process*, with no prior traffic to be limited for;
# * 0.6s, 1.0s, 2.0s and 2.5s spacing all failed identically, as did a 105s
#   backoff before the retry;
# * 11 different ``fields`` combinations x 3 tries each = 33 consecutive 200s,
#   so the field list is not the trigger.
#
# It is a property of the specific ``(after, before)`` range. Failures take
# 3.1-3.4s and successes 0.1-1.6s, which is the shape of a ~3s server-side
# query timeout on a range holding too many rows: r/CryptoCurrency comments
# with ``after=-24h before=-21.8h`` 422s, while ``before=-12h`` on the same
# subreddit returns 200.
#
# The consequence is the part that matters, and it is a permanent one: **no
# page budget completes a 24h comment walk of a busy subreddit.** The older
# pages are not slow, they are unreachable. So this module does not engineer
# around the 422 — it stops the walk, records how much of the window it
# actually covered, and reports the shortfall in ``degraded_reason`` so a low
# comment count cannot be mistaken for a quiet subreddit.
_ARCTIC_SPACING_S = 0.6

# Ceiling on one ``/new`` sweep. Reddit pages listings 100 at a time and caps
# pagination at 1000, so this is 5 requests per subreddit worst case, against a
# 100 req/min budget. It only binds on a subreddit posting faster than
# _PRAW_NEW_LIMIT per lookback window — r/CryptoCurrency runs a few hundred a
# day, so the window closes first and the sweep stops early.
_PRAW_NEW_LIMIT = 500

# Ceiling on one ``/r/<sub>/comments/`` sweep. Checked against the installed
# praw 8.0.3: ``Subreddit.comments`` is a ``praw.util.cache.cachedproperty`` on
# ``SubredditListingMixin`` (so it is absent from ``Subreddit.__dict__`` and
# ``inspect.signature`` on it raises ``TypeError`` — do not reach for either).
# It yields a ``CommentHelper``, which is *callable*:
# ``reddit.subreddit(name).comments(limit=N)``. Its ``_path`` is
# ``/r/<sub>/comments/`` — a **listing, not a search**, so every reason this
# module prefers ``/new`` over ``search()`` carries over unchanged.
#
# 1000 is Reddit's own pagination ceiling, so this is "as deep as the listing
# goes", 10 requests per subreddit. A busy subreddit still outruns it — at the
# measured ≥700 comments/24h for r/CryptoCurrency there is headroom, but not a
# lot — so the sweep reports when it stopped on the limit rather than on the
# window edge.
_PRAW_COMMENT_LIMIT = 1000

# Consecutive out-of-window posts required before a ``/new`` sweep concludes it
# has walked off the end of the window. A listing is reverse-chronological, so
# one *ought* to be enough — but a pinned or recently-approved submission can
# surface out of order, and treating a single stray as the boundary would
# truncate the whole subreddit to nothing.
_PRAW_STALE_RUN = 3

# How long one subreddit sweep is reused across coins within a single tick.
# Every coin in a tick shares the same lookback window, so re-pulling it per
# coin would triple the request count against a host that already rate-limits.
_MEMO_SECONDS = 60.0

CACHE_FILENAME = "sentiment_cache.json"

# 2 (2026-09-19): comments joined the sweep, so an hourly bucket counts a
# different thing than it did under version 1 and the two cannot share a
# baseline — mixing them would show every coin's attention "rising" by exactly
# the amount the denominator grew. Bumping discards the v1 file wholesale,
# which also clears the fabricated-zero buckets a pre-``sweep_size`` run left
# behind (the checked-in cache held 15 zero buckets per coin and
# ``unique_contributors_24h: 0`` from a sweep that read nothing).
#
# The version alone is too blunt for what happens *after* the upgrade, though:
# toggling ``include_comments`` changes the unit again, and a global bump would
# throw away every symbol's history to fix one. So each symbol's entry also
# carries the unit its buckets were counted in — see ``_baseline_unit``.
CACHE_VERSION = 2

# How far a source's index may trail live before the most recent hour stops
# being reportable. Measured against the newest post in the whole sweep, not
# the per-coin matches — a coin with no mentions tells you nothing about
# freshness. Arctic Shift's lag is *variable*, which is the part that makes
# measuring it per tick necessary: ~10h behind on 2026-09-18, but 0.01h on a
# re-check the next day (newest indexed post 22 seconds old). Neither state can
# be assumed. PRAW queries live Reddit and never trips it. 15 minutes is one
# slow tick: a lag under that cannot hide a burst from the next decision.
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
    """One submission **or comment**, normalized out of whichever vendor shape
    produced it.

    Exists for the duration of one tick and is never serialized. ``created_utc``
    is epoch **seconds** (both sources already use seconds; nothing here divides
    by 1000).

    A comment carries ``title=""`` and its body in ``body``, so ``text`` and
    therefore ``matching_posts`` keep working untouched. The alternative —
    synthesizing a title like ``"[comment] ..."`` — was considered and rejected:
    it lies in the type, and ``report.py`` and the prompt would both print the
    lie. ``kind`` is what consumers use to render the difference, and it is
    **trailing and defaulted** on purpose, because callers (tests especially)
    construct ``Post`` positionally.
    """

    id: str
    created_utc: float
    author: str
    title: str
    body: str
    score: int
    subreddit: str
    kind: Literal["submission", "comment"] = "submission"

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
    """Items whose title or body mentions any alias as a whole word.

    Submissions and comments go through the same call and the same pattern: a
    comment is a ``Post`` with an empty title, so ``text`` is its body and the
    word-boundary rule applies identically. Nothing here branches on ``kind``,
    which is the point — the matching semantics stay in exactly one place.
    """
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


def _baseline_unit(cfg: Config) -> str:
    """What one hourly bucket counts, recorded per symbol in the cache.

    ``CACHE_VERSION`` handles the one-time upgrade; this handles the toggle.
    ``include_comments`` can be flipped in config at any time, and a bucket
    counted over submissions+comments is not comparable to one counted over
    submissions alone — a z-score computed across the boundary would read the
    denominator change as a burst of attention. Storing the unit on the entry
    means flipping it invalidates the affected symbols' history and nothing
    else, instead of forcing a global version bump that discards every coin's
    baseline to fix one.
    """
    return "submissions+comments" if cfg.sentiment.include_comments else "submissions"


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
        unique_contributors_24h=(
            None
            if entry.get("unique_contributors_24h") is None
            else int(entry["unique_contributors_24h"])
        ),
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
    sweep_size: int | None = None,
    sweep_unit: str = "posts",
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

    ``sweep_size`` is how many items the provider's **whole sweep** returned —
    submissions plus comments, when comments are enabled — counted before the
    per-coin alias filter. It is the denominator this coin's count sits over,
    and the second half of the same distinction ``observed_through`` draws,
    along the other axis:

    * ``0`` means nothing at all was read, so there is no window to speak
      about. Every rate comes back ``None``, because a sweep that saw no posts
      cannot tell "nobody mentioned this coin" apart from "we did not look".
    * ``> 0`` means the window *was* read, so a zero for this coin is a real
      measurement and stays zero — and the size travels into
      ``degraded_reason`` so the model can see how thin the denominator is.
    * ``None`` means the caller is not reporting one and the window is assumed
      observed, which is what a hand-built ``posts`` list wants.

    ``sweep_unit`` names what ``sweep_size`` counted, for the text handed to the
    model. It is not cosmetic: "the sweep observed 1008 posts" would be a false
    claim about the composition of the evidence when 872 of them are comments.

    Zero posts inside the observed window is a perfectly good answer and
    produces a valid brief with zero velocity — "nobody is talking about this"
    is information. Zero posts because the hour has not been indexed yet, or
    because the sweep came back empty, is not information, and comes back as
    ``None``.
    """
    scfg = cfg.sentiment
    window_hours = float(scfg.lookback_hours)
    window_start = now - window_hours * _SECONDS_PER_HOUR

    # Never trust a source to be ahead of the clock; ``min`` also absorbs the
    # vendor-bug case of a future timestamp.
    observed_end = now if observed_through is None else min(observed_through, now)
    lag_seconds = max(now - observed_end, 0.0)

    # An empty sweep is a read that returned nothing, not a quiet Reddit, and
    # it reaches here *without any subreddit having errored*: arctic-shift
    # answers a subreddit with no posts in the window with HTTP 200 and
    # ``{"data": []}``, which never lands in ``failures``. That is not a rare
    # shape — r/SatoshiStreetBets came back that way for two entire days
    # (2026-09-15 and -16) and for 24 of 28 consecutive hours scanned on
    # 2026-09-19. Before this flag existed the all-empty case produced
    # ``0.0 mentions/hour``, ``0`` contributors and ``degraded_reason = None``:
    # a maximally confident "attention is flat" — the bearish read — with no
    # marker on it at all, which is strictly worse than the unindexed-hour case
    # below because that one at least says why.
    observed = sweep_size is None or sweep_size > 0

    reasons: list[str] = list(extra_reasons)
    if not observed:
        reasons.append(
            "sweep returned no posts at all from any of the "
            f"{len(scfg.subreddits)} configured subreddits; that is a failed "
            "read, not silence — every count here is unmeasured, not zero"
        )
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
    if not observed or lag_seconds > _MAX_INDEX_LAG_SECONDS:
        velocity_1h = None
    else:
        last_hour = [p for p in in_window if p.created_utc >= now - _SECONDS_PER_HOUR]
        velocity_1h = float(len(last_hour))  # one hour of data, so count == rate

    # Average over the hours actually observed, not the hours requested —
    # dividing a 14h count by 24h understates attention by 40% and the error
    # grows with the lag. With nothing observed there is no denominator at all,
    # and "0 posts / 24h = 0.0 mentions/hour" is the same manufactured bearish
    # claim one field over from the one guarded above.
    observed_hours = max(observed_end - window_start, 0.0) / _SECONDS_PER_HOUR
    if not observed or observed_hours <= 0:
        velocity_24h = None
    else:
        velocity_24h = len(in_window) / observed_hours

    contributors = {
        _author_hash(p.author)
        for p in in_window
        if p.author.strip().lower() not in _NON_CONTRIBUTORS
    }
    # "Zero unique contributors" is the same false bearish claim as a zero
    # velocity — it reads as "nobody is talking about this coin" — so a sweep
    # that read nothing reports no breadth rather than no people.
    unique_contributors = None if not observed else len(contributors)
    # Denominator is *all* matched items, including bot/deleted ones and
    # including comments: a wall of deleted posts is exactly the pattern this
    # ratio is meant to expose, and so is one account replying to itself forty
    # times. Counting comments in the numerator but not the denominator — or
    # splitting this into two ratios — would both blur what the number means,
    # which is "how many distinct humans produced the attention we matched".
    # One person writing one submission and nineteen comments about it should
    # score 0.05, and does.
    ratio = (len(contributors) / len(in_window)) if in_window else None

    if history is None:
        zscore = None
        reasons.append("no cross-run baseline available (called without a cache)")
    elif not observed:
        # Seeding "0 mentions this hour" for hours nothing was read from writes
        # fabrications straight into the 7-day baseline, and unlike the
        # index-lag case the self-healing overwrite can never repair them: no
        # later sweep re-observes an hour, it only re-reads whatever the index
        # holds now. The recorded 2026-09-18 run persisted 15 such buckets per
        # coin, and a baseline of manufactured zeros drags the mean down and
        # inflates every z-score computed against it afterwards.
        zscore = None
        reasons.append("no z-score: nothing was observed this tick, so the baseline is untouched")
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

    # A zero is only as strong as its denominator, and here the denominator is
    # small. Measured 2026-09-19, these five subreddits produced 23 BONK
    # submissions, 3 popcat and 1 dogwifhat across the whole preceding 90 days
    # — roughly 0.26 BONK posts/day, and the newest BONK submission in the
    # index was 14.7 days old. A 24h window with no mentions is therefore the
    # *expected* outcome and is nowhere near evidence that attention fell off.
    # Stating the sweep size lets the model weigh the zero instead of trading
    # on it; without it, "0.0 mentions/hour" looks like a reading of the same
    # quality as "12.0 mentions/hour".
    if observed and sweep_size and not in_window:
        reasons.append(
            f"0 mentions of {coin.symbol}; the sweep observed {sweep_size} "
            f"{sweep_unit} over {observed_hours:.1f}h"
        )

    # A comment has no title, so its excerpt comes from the body — same cap,
    # same "handed to the model raw" contract. ``kind`` travels with it so the
    # prompt and the terminal report can label what they are showing instead of
    # rendering a comment as an empty-titled post, which is what happened
    # before: ``report.py`` printed ``r/solana [12]`` and no text at all.
    top = tuple(
        TopPost(
            title=(p.title or p.body)[:_TITLE_MAX_CHARS],
            score=p.score,
            age_hours=round(p.age_hours(now), 2),
            subreddit=p.subreddit,
            kind=p.kind,
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

    An empty sweep stays ``None`` because there is no newest post to measure
    and a fabricated lag would be a second, wrong explanation. It emphatically
    does **not** mean the sweep was fine: an earlier version of this docstring
    claimed an empty sweep "is already reported through ``failures``", and that
    was false. ``failures`` only records subreddits whose request *raised*;
    arctic-shift answers a subreddit with nothing in the window with HTTP 200
    and ``{"data": []}``, so an all-empty sweep leaves ``failures`` empty and
    this function returning ``None``, which together read as a healthy, live,
    silent Reddit. The "we saw nothing" signal therefore travels separately, as
    ``build_brief``'s ``sweep_size``.
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

    Implementations may also expose ``sweep_notes``: a list of plain-English
    caveats about the sweep that *just* returned, read via ``getattr`` and
    folded into ``degraded_reason``. It exists because ``failures`` can only say
    "this subreddit raised", and partial coverage is neither a failure nor a
    success — a comment walk that reached 21.8h of a 24h window returned real
    data for a window narrower than the one requested, and the difference
    between "few comments mention BONK" and "we could only read 90% of the
    window" has to reach the model. Kept off the ``posts()`` return tuple
    deliberately: widening that to three elements would break every fake
    provider in the test suite to carry something optional.
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
    quality. (The host also refuses ``query`` on its own: 400 ``"'query' query
    parameter requires one of: author, subreddit"``.)

    **The window sweep was audited on 2026-09-19 and returns everything the
    index holds**, which is what makes a zero from it trustworthy enough to
    report. Re-running r/solana's 24h window as 24 separate one-hour queries
    surfaced no post the single 24h query had missed (33 from the hourly walk,
    34 from the one-shot, the extra one posted during the run). ``sort=desc``
    orders newest-first as documented and ``limit`` truncates from the new end
    — ``limit=10`` returned the 10 most recent. So when this sweep reports no
    mentions, the posts genuinely do not contain them; the place to look next
    is the base rate, not the fetch.
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
        self._memo: tuple[float, list[Post], list[str], list[str]] | None = None
        #: Caveats about the sweep that just ran. See ``PostSource``.
        self.sweep_notes: list[str] = []

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

    def _page(
        self, cfg: Config, endpoint: str, fields: str, subreddit: str, after: int, before: int
    ) -> list[dict[str, Any]]:
        """One page from ``/{endpoint}/search``. Raises on any non-200.

        ``endpoint`` and ``fields`` travel together and must match: the two
        search endpoints share an envelope and a parameter set but not a field
        whitelist, so ``_ARCTIC_COMMENT_FIELDS`` against ``posts`` or
        ``_ARCTIC_FIELDS`` against ``comments`` is a 400 on every request.
        """
        resp = self._http(cfg).get(
            f"{self._base}/{endpoint}/search",
            params={
                "subreddit": subreddit,
                "after": str(after),
                "before": str(before),
                "limit": str(_ARCTIC_MAX_LIMIT),
                "sort": "desc",
                "fields": fields,
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

    def _walk(
        self,
        cfg: Config,
        endpoint: str,
        fields: str,
        normalize: Callable[[dict[str, Any]], Post | None],
        subreddit: str,
        after: int,
        now: float,
        out: dict[tuple[str, str], Post],
    ) -> float:
        """Page backwards through one subreddit's window. Returns coverage.

        The return value is the oldest epoch second this walk actually reached
        — ``after`` when the window was covered end to end, something larger
        when it stopped short. Stopping short is normal on the comments
        endpoint and unreachable-by-design (see ``_ARCTIC_SPACING_S``), so a
        partial walk keeps everything it did read and reports the shortfall
        rather than discarding the page or retrying into the same timeout.

        A raise on the *first* page means we got nothing and the caller should
        treat the subreddit as failed; a raise on a later page means we got
        something, and throwing it away would be strictly worse than reporting
        it with a caveat. That asymmetry is why this catches per page instead
        of letting the exception reach the caller.
        """
        cursor = int(now) + 1
        covered = float(cursor)
        for page in range(_ARCTIC_MAX_PAGES):
            try:
                raw = self._page(cfg, endpoint, fields, subreddit, after, cursor)
            except Exception as exc:
                if page == 0:
                    raise
                log.info(
                    "arctic-shift: r/%s %s walk stopped after %d page(s): %s",
                    subreddit,
                    endpoint,
                    page,
                    exc,
                )
                return covered
            for item in raw:
                post = normalize(item)
                if post is not None:
                    # Keyed by (kind, id): submission and comment ids come from
                    # different Reddit namespaces and a bare id can collide
                    # across them, which would silently drop one of the two.
                    out[(post.kind, post.id)] = post
            if len(raw) < _ARCTIC_MAX_LIMIT:
                # A short page is the end of the data, not the end of a budget.
                return float(after)
            oldest = min(int(i.get("created_utc", cursor)) for i in raw)
            covered = float(min(covered, oldest))
            if oldest >= cursor or oldest <= after:
                return float(after)
            cursor = oldest
            time.sleep(_ARCTIC_SPACING_S)
        return covered

    def posts(
        self, coin: CoinConfig, cfg: Config, now: float
    ) -> tuple[list[Post], list[str]]:
        """All submissions and comments in the lookback window, every subreddit.

        ``coin`` is unused: this source pulls the whole subreddit window once
        and lets ``matching_posts`` do the per-coin filtering, so three coins
        cost one sweep rather than three.

        Comments are swept unless ``cfg.sentiment.include_comments`` is off, in
        which case this behaves exactly as it did before comments existed here.
        They roughly double the request count (measured 2026-09-19: ~10 comment
        pages against 5 submission pages for the five configured subreddits) and
        multiply the rows read by 7.4.

        Returns ``(posts, failures)``. A subreddit that errors is skipped and
        named in ``failures`` so the brief can be marked degraded rather than
        quietly reporting a lower mention count. A subreddit whose comment walk
        merely *stopped short* is not a failure and is reported through
        ``sweep_notes`` instead.
        """
        # One sweep serves every coin: the window is identical for all of them
        # and this host rate-limits hard, so re-pulling it per coin would triple
        # the request count for identical bytes.
        if self._memo is not None and abs(now - self._memo[0]) < _MEMO_SECONDS:
            self.sweep_notes = list(self._memo[3])
            return list(self._memo[1]), list(self._memo[2])

        scfg = cfg.sentiment
        window_hours = float(scfg.lookback_hours)
        after = int(now - window_hours * _SECONDS_PER_HOUR)
        out: dict[tuple[str, str], Post] = {}
        failures: list[str] = []
        # subreddit -> hours of the window its comment walk actually covered.
        short: dict[str, float] = {}

        for subreddit in scfg.subreddits:
            try:
                self._walk(
                    cfg, "posts", _ARCTIC_FIELDS, _post_from_arctic, subreddit, after, now, out
                )
            except Exception as exc:
                log.warning("arctic-shift: r/%s failed: %s", subreddit, exc)
                failures.append(subreddit)
                # No comment walk for a subreddit whose submissions are already
                # unreachable: it is the same host and the same failure.
                time.sleep(_ARCTIC_SPACING_S)
                continue
            time.sleep(_ARCTIC_SPACING_S)

            if not scfg.include_comments:
                continue
            try:
                covered = self._walk(
                    cfg,
                    "comments",
                    _ARCTIC_COMMENT_FIELDS,
                    _comment_from_arctic,
                    subreddit,
                    after,
                    now,
                    out,
                )
            except Exception as exc:
                # Submissions for this subreddit are already in ``out``, so this
                # is a gap in the evidence rather than a dead subreddit. Naming
                # it in ``failures`` would claim we read nothing from it, which
                # is false and would understate the brief's quality in the
                # opposite direction.
                log.warning("arctic-shift: r/%s comments failed: %s", subreddit, exc)
                short[subreddit] = 0.0
            else:
                if covered > after:
                    short[subreddit] = max(now - covered, 0.0) / _SECONDS_PER_HOUR
            time.sleep(_ARCTIC_SPACING_S)

        notes: list[str] = []
        if short:
            detail = ", ".join(
                f"r/{name} {hours:.1f}h" for name, hours in sorted(short.items())
            )
            # Two causes, one consequence, so one message: the walk either
            # exhausted its page budget or hit the per-range 422, and either
            # way the older end of the window was never read. Measured live
            # 2026-09-19: r/CryptoCurrency reached 17.4h of 24h at 600
            # comments, r/solana 20.7h. Naming only the timeout would be wrong
            # about which one fired on any given tick.
            notes.append(
                f"comment sweep covered less than the requested {window_hours:.0f}h "
                f"window ({detail}); a busy subreddit outruns the "
                f"{_ARCTIC_MAX_PAGES}-page budget and the host times out on the "
                "older pages, so comment-derived counts there are a floor"
            )

        if failures and len(failures) == len(scfg.subreddits):
            raise SourceUnavailable(
                f"arctic-shift returned nothing for any of {len(failures)} subreddits"
            )
        self.sweep_notes = notes
        self._memo = (now, list(out.values()), list(failures), list(notes))
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
            extra_reasons=[*_failure_reasons(failures), *self.sweep_notes],
            observed_through=sweep_observed_through(self, raw_posts, now),
            sweep_size=len(raw_posts),
            sweep_unit=_sweep_unit(cfg),
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


def _comment_from_arctic(item: dict[str, Any]) -> Post | None:
    """Normalize one Arctic Shift comment row. ``None`` for unusable rows.

    Two shape differences from ``_post_from_arctic``, both forced by the
    endpoint rather than chosen: the text arrives in ``body`` (there is no
    ``selftext`` on a comment and asking for one 400s), and there is no title to
    read, so ``title`` stays empty and the ``text`` property falls back to the
    body alone.
    """
    try:
        return Post(
            id=str(item.get("id") or ""),
            created_utc=float(item["created_utc"]),
            author=str(item.get("author") or "[deleted]"),
            title="",
            body=str(item.get("body") or ""),
            score=int(item.get("score") or 0),
            subreddit=str(item.get("subreddit") or ""),
            kind="comment",
        )
    except (KeyError, TypeError, ValueError):
        return None


class PrawProvider:
    """The official Reddit API. Primary path whenever credentials exist.

    PRAW 8.0.3 notes (checked against the installed package, not 7.x memory):

    * ``Subreddit.new(**Unpack[ListingGeneratorKwargs])`` — ``limit=`` still
      works and still means "stop after N", with ``None`` for "as many as
      Reddit will paginate" (1000).
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

    **Sweeps ``/new`` rather than running a search.** This is the difference
    between current data and nearly-current data, and it is the whole reason to
    prefer PRAW over the archive mirror. Reddit's search index is populated
    asynchronously, so a post is live on ``/new`` before it is findable by
    ``search()`` — the same class of staleness this module already refuses to
    paper over in ``ArcticShiftProvider``, merely smaller. ``/new`` is a
    listing, not a query, so there is nothing to wait for.

    Three things fall out of the switch, all good:

    * Reddit's tokenizer leaves the picture entirely. It was only ever a
      prefilter — every hit is re-checked locally by ``matching_posts``,
      because Reddit stems and will happily return "wife" for a "wif" query —
      and a prefilter that can silently drop a real mention is worse than none.
    * One sweep serves every coin, so three coins cost one pass over each
      subreddit instead of three searches.
    * It is the same shape as the Arctic Shift sweep, so both providers now
      hand ``matching_posts`` the same unfiltered window and the matching
      semantics live in exactly one place.
    """

    source = "praw"
    #: Live Reddit via ``/new``: no search index sits between the post being
    #: made and us seeing it, so there is no lag to measure or report.
    index_lags = False

    def __init__(self, reddit: Any | None = None) -> None:
        self._reddit = reddit
        self._memo: tuple[float, list[Post], list[str], list[str]] | None = None
        #: Caveats about the sweep that just ran. See ``PostSource``.
        self.sweep_notes: list[str] = []

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

    def posts(
        self, coin: CoinConfig, cfg: Config, now: float
    ) -> tuple[list[Post], list[str]]:
        """Every submission and comment in the window, across the subreddits.

        ``coin`` is unused, exactly as in ``ArcticShiftProvider.posts``: one
        ``/new`` sweep covers all three coins and ``matching_posts`` does the
        per-coin filtering afterwards.

        ``/r/<sub>/comments/`` is swept the same way unless
        ``cfg.sentiment.include_comments`` is off. It is a listing rather than a
        search, so the argument for preferring ``/new`` over ``search()``
        applies unchanged: nothing sits between a comment being written and us
        reading it.

        Returns ``(posts, failures)``. A subreddit that errors is skipped and
        named in ``failures``, so a dead sub shows up as a degraded brief rather
        than a quietly lower mention count. All of them failing is a dead
        source, not a quiet day, and raises ``SourceUnavailable``.
        """
        scfg = cfg.sentiment
        if not scfg.subreddits:
            raise SourceUnavailable("no subreddits configured")

        # One sweep serves every coin — the window does not depend on which coin
        # is asking, and Reddit's 100 queries/min is worth not spending three
        # times over on identical bytes.
        if self._memo is not None and abs(now - self._memo[0]) < _MEMO_SECONDS:
            self.sweep_notes = list(self._memo[3])
            return list(self._memo[1]), list(self._memo[2])

        after = now - scfg.lookback_hours * _SECONDS_PER_HOUR
        reddit = self._client(cfg)
        out: dict[tuple[str, str], Post] = {}
        failures: list[str] = []
        # Subreddits whose comment listing ran out of pagination before it
        # reached the window edge. Not a failure — real data, narrower window.
        truncated: list[str] = []

        for name in scfg.subreddits:
            try:
                subreddit = reddit.subreddit(name)
                self._drain(subreddit.new(limit=_PRAW_NEW_LIMIT), _post_from_praw, after, out)
                if scfg.include_comments:
                    # ``comments`` is a cachedproperty returning a callable
                    # CommentHelper; ``subreddit.comments(limit=N)`` is the
                    # supported call in praw 8.0.3.
                    seen = self._drain(
                        subreddit.comments(limit=_PRAW_COMMENT_LIMIT),
                        _comment_from_praw,
                        after,
                        out,
                    )
                    if seen >= _PRAW_COMMENT_LIMIT:
                        # We stopped on Reddit's pagination ceiling rather than
                        # on the window edge, so there are older in-window
                        # comments we never saw.
                        truncated.append(name)
            except Exception as exc:
                log.warning("praw: r/%s failed: %s", name, exc)
                failures.append(name)

        notes: list[str] = []
        if truncated:
            notes.append(
                f"comment listing hit Reddit's {_PRAW_COMMENT_LIMIT}-item pagination "
                f"limit before the window edge for {', '.join(f'r/{n}' for n in truncated)}; "
                "comment-derived counts there are a floor"
            )

        if failures and len(failures) == len(scfg.subreddits):
            raise SourceUnavailable(
                f"praw returned nothing for any of {len(failures)} subreddits"
            )
        self.sweep_notes = notes
        self._memo = (now, list(out.values()), list(failures), list(notes))
        return list(out.values()), failures

    @staticmethod
    def _drain(
        listing: Iterable[Any],
        normalize: Callable[[Any], Post | None],
        after: float,
        out: dict[tuple[str, str], Post],
    ) -> int:
        """Consume one reverse-chronological listing. Returns items seen.

        The count is how many rows the listing actually yielded, which is what
        tells the caller *why* the walk ended: short of the limit means we
        reached the window edge, at the limit means Reddit stopped paginating
        and the window is only partly read.
        """
        seen = 0
        stale_run = 0
        for item in listing:
            seen += 1
            post = normalize(item)
            if post is None:
                continue
            if post.created_utc < after:
                # A listing is reverse-chronological, so the first old item is
                # normally the end of the window — but a pinned or
                # recently-approved submission can surface out of order, and
                # stopping on one of those would silently truncate an entire
                # subreddit to nothing. Require a short run before believing we
                # are past the edge.
                stale_run += 1
                if stale_run >= _PRAW_STALE_RUN:
                    break
                continue
            stale_run = 0
            # Keyed by (kind, id): a comment id and a submission id come from
            # different Reddit namespaces and can collide as bare strings.
            out[(post.kind, post.id)] = post
        return seen

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
            extra_reasons=[*_failure_reasons(failures), *self.sweep_notes],
            observed_through=sweep_observed_through(self, found, now),
            sweep_size=len(found),
            sweep_unit=_sweep_unit(cfg),
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


def _comment_from_praw(comment: Any) -> Post | None:
    """Normalize one PRAW ``Comment``. Same ``getattr`` discipline as above.

    A ``Comment`` has ``body`` and no ``title``, and its ``score`` can be hidden
    for the first hour — PRAW reports that as ``0`` rather than ``None``, which
    only affects the ordering of ``top_posts`` and never a count.
    """
    try:
        author = getattr(comment, "author", None)
        name = getattr(author, "name", None) or (str(author) if author else "[deleted]")
        subreddit = getattr(comment, "subreddit", "")
        return Post(
            id=str(getattr(comment, "id", "") or ""),
            created_utc=float(getattr(comment, "created_utc", 0.0)),
            author=str(name),
            title="",
            body=str(getattr(comment, "body", "") or ""),
            score=int(getattr(comment, "score", 0) or 0),
            subreddit=str(getattr(subreddit, "display_name", None) or subreddit or ""),
            kind="comment",
        )
    except Exception:
        return None


def _sweep_unit(cfg: Config) -> str:
    """What ``sweep_size`` counted, for the sentence handed to the model."""
    return "posts and comments" if cfg.sentiment.include_comments else "posts"


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

    # What this entry's hourly buckets count, versus what we are about to count.
    # ``include_comments`` can be flipped between runs and the two units are not
    # comparable — a 7.4x larger denominator would read as a burst of attention
    # on every coin at once, which is the most expensive possible false signal.
    # Scoped to the symbol on purpose (the user's call): flipping the toggle
    # should cost the affected symbols' baselines, not every coin's.
    unit = _baseline_unit(cfg)
    cached_unit = entry.get("unit")
    unit_changed = cached_unit is not None and cached_unit != unit
    unit_reasons: list[str] = []
    if unit_changed:
        log.info(
            "%s: sentiment baseline unit changed (%s -> %s); discarding this "
            "symbol's history",
            coin.symbol,
            cached_unit,
            unit,
        )
        entry.pop("hourly", None)
        entry.pop("brief", None)
        entry.pop("fetched_at", None)
        unit_reasons.append(
            f"baseline discarded: hourly counts now cover {unit} rather than "
            f"{cached_unit}, and the two are not comparable"
        )

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
                extra_reasons=[
                    *unit_reasons,
                    *_failure_reasons(failures),
                    # Partial-coverage caveats. ``getattr`` because this is an
                    # optional part of the seam; a provider without it is fine.
                    *getattr(provider, "sweep_notes", ()),
                ],
                observed_through=sweep_observed_through(provider, found, now),
                sweep_size=len(found),
                sweep_unit=_sweep_unit(cfg),
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
    entry["unit"] = unit

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
