"""Tests for ``memetrader.sentiment``. No network, ever.

The most important test in this file is ``TestDegradation``: the module's real
contract is not "compute a z-score correctly", it is "a Reddit outage must not
be able to fail a trading tick, and must not be able to masquerade as silence".
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import httpx
import pytest

from memetrader import sentiment as S
from memetrader.config import (
    CadenceConfig,
    CoinConfig,
    Config,
    DataConfig,
    ExecutionConfig,
    ModelConfig,
    PromptConfig,
    RiskConfig,
    SentimentConfig,
)
from memetrader.types import SentimentBrief

FIXTURES = Path(__file__).parent / "fixtures"

# A fixed clock. Aligned to an exact hour boundary so "the current hour bucket"
# is unambiguous and the arithmetic below is checkable by hand.
NOW = 1_700_000_000.0 - (1_700_000_000.0 % 3600)
HOUR = 3600.0

WIF = CoinConfig(symbol="WIF", mint="EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm", aliases=("WIF", "dogwifhat"))
BONK = CoinConfig(symbol="BONK", mint="DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263", aliases=("BONK",))


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def make_config(tmp_path: Path, **sentiment_overrides) -> Config:
    """A Config good enough for this module, with ``data_dir`` in a tmpdir.

    Built by hand rather than via ``config.load()`` so these tests never touch
    the real ``config.toml``, the real ``data/`` directory or the environment.
    """
    scfg_kwargs = dict(
        enabled=True,
        cache_ttl_seconds=600,
        lookback_hours=24,
        baseline_days=7,
        subreddits=("CryptoCurrency", "solana"),
        reddit_client_id=None,
        reddit_client_secret=None,
        reddit_user_agent="memetrader-test/0.1",
    )
    scfg_kwargs.update(sentiment_overrides)
    return Config(
        root=tmp_path,
        data_dir=tmp_path,
        starting_cash_usd=1000.0,
        coins=(WIF, BONK),
        model=ModelConfig("claude-opus-5", "high", 8000, 5.0, 25.0, 0.5, 6.25),
        cadence=CadenceConfig(60, 900),
        risk=RiskConfig(0.3, 0.15, 10.0, 3.0, 90.0, 50_000.0),
        execution=ExecutionConfig(50.0, 0.21, 0.06, 0.25, {}),
        data=DataConfig(
            "https://api.dexscreener.com",
            "https://api.geckoterminal.com/api/v2",
            "https://lite-api.jup.ag",
            "https://api.jup.ag",
            15.0,
            100,
            100,
            None,
        ),
        sentiment=SentimentConfig(**scfg_kwargs),
        prompt=PromptConfig(10),
        anthropic_api_key=None,
    )


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return make_config(tmp_path)


def post(
    *,
    hours_ago: float,
    author: str = "someone",
    title: str = "WIF is pumping",
    body: str = "",
    score: int = 1,
    subreddit: str = "solana",
    id: str | None = None,
) -> S.Post:
    return S.Post(
        id=id or f"p{hours_ago}-{author}-{title[:8]}-{score}",
        created_utc=NOW - hours_ago * HOUR,
        author=author,
        title=title,
        body=body,
        score=score,
        subreddit=subreddit,
    )


class StubProvider:
    """Speaks only ``fetch()`` — the minimal ``SentimentProvider``."""

    source = "arctic_shift"

    def __init__(self, posts: list[S.Post] | None = None, raises: Exception | None = None):
        self.posts_ = posts or []
        self.raises = raises
        self.calls = 0

    def fetch(self, coin: CoinConfig, cfg: Config) -> SentimentBrief | None:
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return S.build_brief(
            coin, cfg, S.matching_posts(self.posts_, coin.aliases), self.source, NOW, history=None
        )


class StubPostSource:
    """Speaks ``posts()`` too, so ``brief()`` maintains the persisted baseline."""

    source = "arctic_shift"

    def __init__(self, posts: list[S.Post] | None = None, raises: Exception | None = None):
        self.posts_ = posts or []
        self.raises = raises
        self.calls = 0

    def posts(self, coin: CoinConfig, cfg: Config, now: float):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return list(self.posts_), []

    def fetch(self, coin: CoinConfig, cfg: Config) -> SentimentBrief | None:  # pragma: no cover
        raise AssertionError("brief() should prefer posts() on a PostSource")


# ---------------------------------------------------------------------------
# Word-boundary matching — the difference between signal and noise
# ---------------------------------------------------------------------------


class TestAliasMatching:
    @pytest.mark.parametrize(
        "text",
        [
            "WIF is pumping",
            "$WIF",
            "buying $WIF right now",
            "wif looks strong",  # case-insensitive
            "#WIF",
            "(WIF)",
            "WIF's chart",
            "WIF, BONK and POPCAT",
            "sold my WIF.",
            "dogwifhat gang",
        ],
    )
    def test_matches(self, text):
        assert S.alias_pattern(WIF.aliases).search(text), text

    @pytest.mark.parametrize(
        "text",
        [
            "my wife is asking about crypto",
            "swift protocol launch",
            "the wifi is down",
            "midwife",
            "wifey",
            "W1F",  # digit substitution is not matched; documented limitation
            "swifties",
        ],
    )
    def test_does_not_match(self, text):
        assert not S.alias_pattern(WIF.aliases).search(text), text

    def test_the_naive_version_would_have_been_wrong(self):
        """The bug this pattern exists to prevent, stated explicitly."""
        noise = "my wife used wifi to watch swift videos"
        assert "wif" in noise.lower()  # substring matching says yes
        assert not S.alias_pattern(("WIF",)).search(noise)  # we say no

    def test_matches_in_body_not_just_title(self):
        posts = [post(hours_ago=1.0, title="daily thread", body="anyone still holding dogwifhat?")]
        assert len(S.matching_posts(posts, WIF.aliases)) == 1

    def test_other_coin_is_not_matched(self):
        posts = [post(hours_ago=1.0, title="WIF is pumping")]
        assert S.matching_posts(posts, BONK.aliases) == []

    def test_empty_aliases_match_nothing(self):
        assert not S.alias_pattern(()).search("WIF BONK anything")

    def test_real_indexed_titles_match(self):
        """Titles pulled from Arctic Shift on 2026-09-19 by searching the five
        configured subreddits over 90 days. All 24 submissions the index holds
        for these coins were matched by this pattern, 24/24 — which is what
        rules the matcher out as the cause of a zero-mention sweep.
        """
        bonk = S.alias_pattern(BONK.aliases)
        assert bonk.search("Anyone seen BONK tanking?")
        assert bonk.search("Heard of the BONK DAO Exploit? | A Funny, Straightforward One")
        # ``_`` is deliberately not a fence, which is the entire reason for
        # lookarounds over ``\b``: "Bonk_inu" is a mention, "Bonkplay" is not.
        assert bonk.search("Bonk_inu Launched Bonkplay With Over $1m In Rewards")
        assert not bonk.search("Bonkplay launched with rewards")
        assert S.alias_pattern(("POPCAT", "popcat")).search("Popke meme, predates Popcat")

    def test_the_live_corpus_that_matched_nothing_really_contains_nothing(self):
        """Real titles from the 2026-09-19 sweep, which returned 143 posts and
        zero matches for all three coins. A naive substring scan of that whole
        41 KB corpus found no "bonk", "wif" or "popcat" either, so the zero is
        the corpus and not the pattern.
        """
        titles = [
            "Robux has to move right??!",
            "Solana up 12% in a day while Congress kills crypto legislation",
            "What is the most amount of money you have made in memecoins?",
            "My first 3 days trading memecoins on Pump.fun",
            "Why shiba inu is failing traders ?",
        ]
        posts = [post(hours_ago=1.0, title=t, id=f"live{i}") for i, t in enumerate(titles)]
        for coin in (WIF, BONK):
            assert S.matching_posts(posts, coin.aliases) == []


# ---------------------------------------------------------------------------
# Velocity
# ---------------------------------------------------------------------------


class TestVelocity:
    def test_known_timestamps(self, cfg):
        # 5 posts inside the last hour, 12 more spread across the 24h window,
        # and 3 that are older than the window and must be ignored entirely.
        posts = (
            [post(hours_ago=0.1 * i, author=f"a{i}") for i in range(5)]
            + [post(hours_ago=2.0 + i, author=f"b{i}") for i in range(12)]
            + [post(hours_ago=30.0 + i, author=f"c{i}") for i in range(3)]
        )
        brief = S.build_brief(WIF, cfg, posts, "arctic_shift", NOW, history=None)

        assert brief.mention_velocity_1h == 5.0  # 5 mentions in 1 hour
        assert brief.mention_velocity_24h == pytest.approx(17 / 24)  # 5 + 12 in 24h
        assert brief.symbol == "WIF"
        assert brief.source == "arctic_shift"
        assert brief.ts == NOW

    def test_acceleration_is_the_ratio(self, cfg):
        """A burst reads as 1h rate far above the 24h baseline rate."""
        quiet = [post(hours_ago=3.0 + i, author=f"q{i}") for i in range(12)]
        burst = [post(hours_ago=0.2, author=f"burst{i}") for i in range(20)]
        b = S.build_brief(WIF, cfg, quiet + burst, "arctic_shift", NOW, history=None)
        assert b.mention_velocity_1h / b.mention_velocity_24h > 10

    def test_window_boundary_is_exclusive_of_older_posts(self, cfg):
        just_inside = post(hours_ago=23.99, author="x")
        just_outside = post(hours_ago=24.01, author="y")
        b = S.build_brief(WIF, cfg, [just_inside, just_outside], "arctic_shift", NOW, history=None)
        assert b.mention_velocity_24h == pytest.approx(1 / 24)

    def test_future_timestamps_are_excluded(self, cfg):
        """Clock skew or a vendor bug must not create an hourly bucket past
        ``now`` — the retention trim is a lower bound and could never evict it,
        so it would sit in the baseline forever."""
        history: dict[str, int] = {}
        posts = [post(hours_ago=-5.0, author="from_the_future"), post(hours_ago=1.0)]
        b = S.build_brief(WIF, cfg, posts, "arctic_shift", NOW, history=history)
        assert b.mention_velocity_24h == pytest.approx(1 / 24)
        assert max(int(k) for k in history) <= int(NOW // 3600)

    def test_lookback_hours_is_honoured(self, tmp_path):
        cfg = make_config(tmp_path, lookback_hours=48)
        posts = [post(hours_ago=float(h), author=f"a{h}") for h in range(1, 48)]
        b = S.build_brief(WIF, cfg, posts, "arctic_shift", NOW, history=None)
        assert b.mention_velocity_24h == pytest.approx(47 / 48)


# ---------------------------------------------------------------------------
# Index lag
# ---------------------------------------------------------------------------


class TestIndexLag:
    """A source that has not indexed the last hour must say so, not say zero.

    Arctic Shift is an archive mirror and was running ~10h behind live Reddit
    on 2026-09-18. Before this was handled, every brief reported
    ``mention_velocity_1h = 0.0`` on every tick forever — not because the coins
    were quiet, but because the hour did not exist in the index yet. Zero
    attention reads bearish, so the model was being handed a confident, wrong,
    permanently-stuck signal on the evidence stream that matters most for
    memecoins.
    """

    LAG = 10 * HOUR

    def test_an_unindexed_hour_is_none_not_zero(self, cfg):
        """The whole point: absence of data must not render as absence of talk."""
        posts = [post(hours_ago=self.LAG / HOUR + i, author=f"a{i}") for i in range(5)]
        b = S.build_brief(
            WIF, cfg, posts, "arctic_shift", NOW, history=None,
            observed_through=NOW - self.LAG,
        )
        assert b.mention_velocity_1h is None
        assert "behind live" in (b.degraded_reason or "")

    def test_a_fresh_source_still_reports_a_genuine_zero(self, cfg):
        """The other half of the distinction. If this ever returns None, the
        fix has overshot and destroyed the 'we looked, it is quiet' signal."""
        b = S.build_brief(
            WIF, cfg, [post(hours_ago=5.0)], "arctic_shift", NOW, history=None,
            observed_through=NOW,
        )
        assert b.mention_velocity_1h == 0.0
        assert "behind live" not in (b.degraded_reason or "")

    def test_no_observed_through_means_live(self, cfg):
        """Omitting the argument asserts the source is current — every existing
        caller and test depends on this staying the default."""
        b = S.build_brief(WIF, cfg, [post(hours_ago=5.0)], "arctic_shift", NOW, history=None)
        assert b.mention_velocity_1h == 0.0

    def test_small_lag_is_tolerated(self, cfg):
        """Under the threshold the rolling hour is still meaningful, and
        blanking it would throw away a good reading."""
        b = S.build_brief(
            WIF, cfg, [post(hours_ago=0.1)], "arctic_shift", NOW, history=None,
            observed_through=NOW - 300.0,
        )
        assert b.mention_velocity_1h == 1.0

    def test_24h_rate_divides_by_observed_hours_not_requested_hours(self, cfg):
        """14 posts over the 14 observed hours is 1.0/h. Dividing by the 24h
        window that was *asked* for gives 0.58/h — a 40% understatement that
        grows with the lag, and it would land on the baseline the 1h rate is
        compared against."""
        posts = [post(hours_ago=self.LAG / HOUR + i, author=f"a{i}") for i in range(14)]
        b = S.build_brief(
            WIF, cfg, posts, "arctic_shift", NOW, history=None,
            observed_through=NOW - self.LAG,
        )
        assert b.mention_velocity_24h == pytest.approx(1.0)

    def test_posts_after_the_observed_end_are_excluded(self, cfg):
        """A source cannot hand back a post newer than its own index. If one
        appears it is a bug in the source or in our freshness measurement, and
        counting it would inflate exactly the window we just declared blind."""
        b = S.build_brief(
            WIF, cfg, [post(hours_ago=0.1)], "arctic_shift", NOW, history=None,
            observed_through=NOW - self.LAG,
        )
        assert b.mention_velocity_24h == 0.0

    def test_unobserved_hours_are_not_written_to_the_baseline(self, cfg):
        """The quieter bug. ``_update_history`` seeds every hour in the window
        at zero so that "no mentions" counts as an observation — correct, until
        10 of those hours were never looked at. Those fabricated zeros drag the
        7-day mean down and inflate every later z-score."""
        history: dict[str, int] = {}
        S.build_brief(
            WIF, cfg, [], "arctic_shift", NOW, history=history,
            observed_through=NOW - self.LAG,
        )
        newest_bucket = max(int(k) for k in history)
        assert newest_bucket <= int((NOW - self.LAG) // 3600)
        # And the observed part of the window *was* recorded, so this is not
        # passing simply because history came back empty.
        assert len(history) > 12

    def test_no_zscore_is_emitted_for_an_unindexed_hour(self, cfg):
        """A z-score is a statement about the current hour. With no current
        hour there is nothing to score, and scoring a None would crash."""
        history = {str(int((NOW - (50 + i) * HOUR) // 3600)): 2 for i in range(60)}
        b = S.build_brief(
            WIF, cfg, [], "arctic_shift", NOW, history=history,
            observed_through=NOW - self.LAG,
        )
        assert b.mention_zscore_7d is None
        assert "not indexed" in (b.degraded_reason or "")


class TestSweepObservedThrough:
    class Lagging:
        index_lags = True

    class Live:
        index_lags = False

    def test_lagging_source_reports_its_newest_post(self):
        sweep = [post(hours_ago=10.0), post(hours_ago=12.0)]
        assert S.sweep_observed_through(self.Lagging(), sweep, NOW) == NOW - 10.0 * HOUR

    def test_live_source_is_never_treated_as_lagging(self):
        """PRAW searches per coin, so its newest result says how recently the
        *coin* was mentioned, not how fresh the index is. Inferring lag from it
        would suppress every genuine quiet reading."""
        sweep = [post(hours_ago=20.0)]
        assert S.sweep_observed_through(self.Live(), sweep, NOW) is None

    def test_an_empty_sweep_is_a_failure_not_a_lag(self):
        """Zero posts across every subreddit is a broken read, already reported
        through ``failures``. Calling it a 10h lag would be a second, wrong
        explanation for the same symptom."""
        assert S.sweep_observed_through(self.Lagging(), [], NOW) is None


# ---------------------------------------------------------------------------
# Empty sweeps — "missing is never zero", applied to the fields next door
# ---------------------------------------------------------------------------


class TestEmptySweep:
    """A sweep that read nothing is a failed read, not a silent Reddit.

    ``TestIndexLag`` above protects ``mention_velocity_1h`` from exactly this
    mistake. The fields beside it were not protected: the 2026-09-18 cache
    shipped ``mention_velocity_24h = 0.0`` and ``unique_contributors_24h = 0``
    for all three coins, and had the sweep been empty rather than merely 10h
    behind, those would have gone to the model with ``degraded_reason = None``
    on them. "Zero unique contributors" reads as "nobody is talking about this
    coin", which is the same manufactured bearish claim one field over.

    The trigger is ordinary, not exotic. arctic-shift answers a subreddit with
    nothing in the window with HTTP 200 and ``{"data": []}``, which never
    reaches ``failures``: r/SatoshiStreetBets came back that way for two entire
    days (2026-09-15 and -16) and for 24 of 28 consecutive hours scanned on
    2026-09-19.
    """

    SWEPT = 143  # what the live 2026-09-19 sweep actually returned

    def test_rates_are_none_not_zero(self, cfg):
        b = S.build_brief(WIF, cfg, [], "arctic_shift", NOW, history=None, sweep_size=0)
        assert b.mention_velocity_1h is None
        assert b.mention_velocity_24h is None
        assert b.mention_zscore_7d is None
        assert b.contributor_to_post_ratio is None
        assert b.polarity is None

    def test_it_says_why(self, cfg):
        b = S.build_brief(WIF, cfg, [], "arctic_shift", NOW, history=None, sweep_size=0)
        assert b.degraded_reason and "no posts at all" in b.degraded_reason

    def test_a_genuine_zero_survives(self, cfg):
        """The other half of the distinction, and the one a fix can destroy by
        overshooting. The sweep looked at 143 posts and none mentioned this
        coin: that is a measurement, and blanking it would throw away the
        "we looked and it is quiet" signal the module exists to preserve."""
        b = S.build_brief(WIF, cfg, [], "arctic_shift", NOW, history=None, sweep_size=self.SWEPT)
        assert b.mention_velocity_1h == 0.0
        assert b.mention_velocity_24h == 0.0
        assert b.unique_contributors_24h == 0

    def test_a_genuine_zero_carries_its_denominator(self, cfg):
        """0 out of 143 and 0 out of 3 are different claims, and the model
        cannot weigh the first without the second number."""
        b = S.build_brief(WIF, cfg, [], "arctic_shift", NOW, history=None, sweep_size=self.SWEPT)
        assert "0 mentions of WIF" in (b.degraded_reason or "")
        assert "143 posts" in (b.degraded_reason or "")

    def test_a_nonzero_count_needs_no_denominator_note(self, cfg):
        b = S.build_brief(
            WIF, cfg, [post(hours_ago=0.5)], "arctic_shift", NOW,
            history=None, sweep_size=self.SWEPT,
        )
        assert "0 mentions" not in (b.degraded_reason or "")

    def test_omitting_sweep_size_still_means_observed(self, cfg):
        """Every hand-built ``posts`` list and every ``fetch()``-only provider
        depends on this: declining to report a sweep size must not blank the
        arithmetic."""
        b = S.build_brief(WIF, cfg, [], "arctic_shift", NOW, history=None)
        assert b.mention_velocity_1h == 0.0
        assert b.mention_velocity_24h == 0.0
        assert "no posts at all" not in (b.degraded_reason or "")

    def test_the_baseline_is_not_seeded_from_an_empty_sweep(self, cfg):
        """Fabricated zeros here are permanent, unlike the index-lag case.
        A lagging index is self-healing because the next sweep re-reads those
        hours; an hour nothing was read from is never revisited, so a
        manufactured zero sits in the 7-day baseline dragging the mean down and
        inflating every later z-score for good."""
        history: dict[str, int] = {}
        S.build_brief(WIF, cfg, [], "arctic_shift", NOW, history=history, sweep_size=0)
        assert history == {}

    def test_an_observed_sweep_still_seeds_the_baseline(self, cfg):
        """The contrast: observed quiet hours are observations and must land,
        otherwise the baseline is built only out of busy hours."""
        history: dict[str, int] = {}
        S.build_brief(WIF, cfg, [], "arctic_shift", NOW, history=history, sweep_size=self.SWEPT)
        assert len(history) == cfg.sentiment.lookback_hours + 1
        assert set(history.values()) == {0}

    def test_an_existing_baseline_is_left_intact(self, cfg):
        """Not seeding must not mean discarding — ``brief()`` writes this dict
        straight back to the cache file."""
        history = {str(int(NOW // 3600) - 1 - i): 2 for i in range(60)}
        before = dict(history)
        S.build_brief(WIF, cfg, [], "arctic_shift", NOW, history=history, sweep_size=0)
        assert history == before

    def test_no_zscore_is_invented_from_an_unobserved_sweep(self, cfg):
        history = {str(int(NOW // 3600) - 1 - i): (0 if i % 2 else 4) for i in range(60)}
        b = S.build_brief(WIF, cfg, [], "arctic_shift", NOW, history=history, sweep_size=0)
        assert b.mention_zscore_7d is None
        assert "nothing was observed" in (b.degraded_reason or "")

    def test_contributors_are_unmeasured_not_zero(self, cfg):
        """Breadth is the last count that used to escape the rule. A ``0`` here
        reads as "nobody is talking about this coin" — the bearish claim — so an
        unobserved sweep reports no breadth rather than no people."""
        b = S.build_brief(WIF, cfg, [], "arctic_shift", NOW, history=None, sweep_size=0)
        assert b.unique_contributors_24h is None
        assert "unmeasured, not zero" in (b.degraded_reason or "")

    def test_an_observed_sweep_that_misses_this_coin_still_reports_zero(self, cfg):
        """The other half of the distinction, and the reason ``sweep_size`` is
        measured before the alias filter: 135 posts read and none of them
        mentioning WIF is a real zero, and must not be softened into ``None``."""
        b = S.build_brief(WIF, cfg, [], "arctic_shift", NOW, history=None, sweep_size=135)
        assert b.unique_contributors_24h == 0
        assert b.mention_velocity_24h == 0.0

    def test_end_to_end_through_brief(self, cfg):
        """Still a brief, not a ``None``: the outage is described rather than
        hidden, which keeps the reason on its way to the prompt."""
        b = S.brief(WIF, cfg, provider=StubPostSource(posts=[]), now=NOW)
        assert b is not None
        assert b.mention_velocity_1h is None
        assert b.mention_velocity_24h is None

    def test_the_cache_round_trip_keeps_the_nones(self, cfg):
        """The TTL is 10 minutes, so a cached "not observed" is reused for the
        rest of the window. Rehydrating it as a confident zero would put the
        bug back one layer down."""
        S.brief(WIF, cfg, provider=StubPostSource(posts=[]), now=NOW)
        cached = S.brief(WIF, cfg, provider=StubPostSource(posts=[]), now=NOW + 60)
        assert cached is not None
        assert cached.mention_velocity_1h is None
        assert cached.mention_velocity_24h is None

    def test_an_empty_sweep_and_a_quiet_one_are_distinguishable(self, cfg):
        """The whole point, stated in one assertion."""
        unobserved = S.build_brief(WIF, cfg, [], "arctic_shift", NOW, history=None, sweep_size=0)
        quiet = S.build_brief(WIF, cfg, [], "arctic_shift", NOW, history=None, sweep_size=self.SWEPT)
        assert unobserved.mention_velocity_24h is None
        assert quiet.mention_velocity_24h == 0.0


# ---------------------------------------------------------------------------
# Z-score
# ---------------------------------------------------------------------------


class TestZScore:
    def test_hand_computed(self):
        """Baseline of 24 zeros and 24 twos: mean 1.0, population stdev 1.0."""
        bucket = int(NOW // 3600)
        history = {str(bucket - 1 - i): (0 if i < 24 else 2) for i in range(48)}
        z, why = S._zscore(history, current_rate=4.0, now=NOW)
        assert why is None
        assert z == pytest.approx(3.0)  # (4 - 1) / 1

    def test_current_hour_is_excluded_from_its_own_baseline(self):
        bucket = int(NOW // 3600)
        history = {str(bucket - 1 - i): (0 if i < 24 else 2) for i in range(48)}
        history[str(bucket)] = 999  # a huge current hour must not move the mean
        z, why = S._zscore(history, current_rate=4.0, now=NOW)
        assert why is None
        assert z == pytest.approx(3.0)

    def test_none_without_enough_baseline(self, cfg):
        """A missing baseline is reported, not fabricated."""
        b = S.build_brief(WIF, cfg, [post(hours_ago=0.5)], "arctic_shift", NOW, history={})
        assert b.mention_zscore_7d is None
        assert b.degraded_reason and "baseline" in b.degraded_reason
        assert f"/{S.MIN_BASELINE_HOURS}" in b.degraded_reason

    def test_none_when_baseline_has_no_variance(self):
        bucket = int(NOW // 3600)
        history = {str(bucket - 1 - i): 0 for i in range(S.MIN_BASELINE_HOURS + 10)}
        z, why = S._zscore(history, current_rate=3.0, now=NOW)
        assert z is None
        assert why and "variance" in why

    def test_spike_against_a_real_accumulated_baseline(self, cfg):
        """End to end through build_brief, with an independent expectation."""
        bucket = int(NOW // 3600)
        # 48 completed hours older than the 24h window, so build_brief will not
        # overwrite them: alternating 0 and 4.
        history = {str(bucket - 25 - i): (0 if i % 2 else 4) for i in range(48)}
        # Inside the window: 2 mentions in each of the 24 previous hours, and a
        # spike of 8 in the current hour.
        posts = [
            post(hours_ago=h + 0.5, author=f"a{h}-{k}")
            for h in range(1, 25)
            for k in range(2)
        ] + [post(hours_ago=0.1, author=f"spike{k}") for k in range(8)]

        b = S.build_brief(WIF, cfg, posts, "arctic_shift", NOW, history=history)

        expected_baseline = [v for k, v in history.items() if k != str(bucket)]
        expected = (8.0 - statistics.mean(expected_baseline)) / statistics.pstdev(
            expected_baseline
        )
        assert b.mention_zscore_7d == pytest.approx(expected)
        assert b.mention_zscore_7d > 2.0  # a spike reads as a spike

    def test_history_is_trimmed_to_baseline_days(self, cfg):
        bucket = int(NOW // 3600)
        history = {str(bucket - 5000 - i): 1 for i in range(20)}  # ancient
        S.build_brief(WIF, cfg, [], "arctic_shift", NOW, history=history)
        oldest_allowed = bucket - cfg.sentiment.baseline_days * 24
        assert all(int(k) >= oldest_allowed for k in history)

    def test_history_accumulates_across_calls_without_double_counting(self, cfg):
        """Two fetches in the same window must not inflate the buckets."""
        posts = [post(hours_ago=0.2, author=f"a{i}") for i in range(3)]
        history: dict[str, int] = {}
        S.build_brief(WIF, cfg, posts, "arctic_shift", NOW, history=history)
        first = dict(history)
        S.build_brief(WIF, cfg, posts, "arctic_shift", NOW, history=history)
        assert history == first
        assert sum(history.values()) == 3  # three posts, counted once

    def test_quiet_hours_are_recorded_as_zero_not_dropped(self, cfg):
        """An hour with no mentions is an observation and must land in the
        history, otherwise the baseline is computed only over busy hours."""
        history: dict[str, int] = {}
        S.build_brief(WIF, cfg, [post(hours_ago=0.2)], "arctic_shift", NOW, history=history)
        assert len(history) == cfg.sentiment.lookback_hours + 1
        assert sum(history.values()) == 1


# ---------------------------------------------------------------------------
# Breadth — the shill-farm detector
# ---------------------------------------------------------------------------


class TestContributorRatio:
    def test_shill_pattern_scores_low(self, cfg):
        posts = [
            post(hours_ago=0.5 + i * 0.01, author=f"shill{i % 2}", id=f"s{i}")
            for i in range(40)
        ]
        b = S.build_brief(WIF, cfg, posts, "arctic_shift", NOW, history=None)
        assert b.unique_contributors_24h == 2
        assert b.contributor_to_post_ratio == pytest.approx(2 / 40)
        assert b.contributor_to_post_ratio < 0.1

    def test_organic_pattern_scores_high(self, cfg):
        authors = [f"user{i}" for i in range(38)] + ["user0", "user1"]
        posts = [
            post(hours_ago=0.5 + i * 0.01, author=a, id=f"o{i}")
            for i, a in enumerate(authors)
        ]
        b = S.build_brief(WIF, cfg, posts, "arctic_shift", NOW, history=None)
        assert b.unique_contributors_24h == 38
        assert b.contributor_to_post_ratio == pytest.approx(38 / 40)
        assert b.contributor_to_post_ratio > 0.9

    def test_shill_and_organic_are_separable(self, cfg):
        """The whole point: same post count, opposite verdicts."""
        shill = [post(hours_ago=0.5, author=f"s{i % 2}", id=f"a{i}") for i in range(40)]
        organic = [post(hours_ago=0.5, author=f"u{i}", id=f"b{i}") for i in range(40)]
        lo = S.build_brief(WIF, cfg, shill, "arctic_shift", NOW, history=None)
        hi = S.build_brief(WIF, cfg, organic, "arctic_shift", NOW, history=None)
        assert lo.mention_velocity_1h == hi.mention_velocity_1h == 40.0
        assert lo.contributor_to_post_ratio < 0.1 < 0.9 < hi.contributor_to_post_ratio

    def test_none_with_zero_posts(self, cfg):
        b = S.build_brief(WIF, cfg, [], "arctic_shift", NOW, history=None)
        assert b.contributor_to_post_ratio is None
        assert b.unique_contributors_24h == 0

    @pytest.mark.parametrize("name", ["[deleted]", "AutoModerator", "automoderator", "[removed]"])
    def test_bots_and_tombstones_are_not_contributors(self, cfg, name):
        posts = [
            post(hours_ago=0.5, author="real_person", id="r1"),
            post(hours_ago=0.6, author=name, id="r2"),
        ]
        b = S.build_brief(WIF, cfg, posts, "arctic_shift", NOW, history=None)
        assert b.unique_contributors_24h == 1
        # Denominator stays at 2: a wall of deleted posts should *depress* the
        # ratio, because that is exactly the pattern it exists to expose.
        assert b.contributor_to_post_ratio == pytest.approx(0.5)

    def test_author_case_is_not_a_second_contributor(self, cfg):
        posts = [
            post(hours_ago=0.5, author="SameGuy", id="c1"),
            post(hours_ago=0.6, author="sameguy", id="c2"),
        ]
        b = S.build_brief(WIF, cfg, posts, "arctic_shift", NOW, history=None)
        assert b.unique_contributors_24h == 1


# ---------------------------------------------------------------------------
# Top posts and polarity
# ---------------------------------------------------------------------------


class TestTopPostsAndPolarity:
    def test_top_three_by_score(self, cfg):
        posts = [
            post(hours_ago=1.0, author=f"u{i}", score=s, title=f"WIF post {s}", id=f"t{s}")
            for i, s in enumerate([5, 900, 12, 300, 1])
        ]
        b = S.build_brief(WIF, cfg, posts, "arctic_shift", NOW, history=None)
        assert [t.score for t in b.top_posts] == [900, 300, 12]
        assert b.top_posts[0].subreddit == "solana"
        assert b.top_posts[0].age_hours == pytest.approx(1.0)

    def test_titles_are_capped(self, cfg):
        b = S.build_brief(
            WIF, cfg, [post(hours_ago=0.5, title="WIF " + "x" * 5000)], "arctic_shift", NOW, history=None
        )
        assert len(b.top_posts[0].title) <= 200

    def test_no_posts_no_top_posts(self, cfg):
        assert S.build_brief(WIF, cfg, [], "arctic_shift", NOW, history=None).top_posts == ()

    def test_polarity_is_none_without_keywords(self, cfg):
        b = S.build_brief(
            WIF, cfg, [post(hours_ago=0.5, title="WIF", body="a b c")], "arctic_shift", NOW, history=None
        )
        assert b.polarity is None

    def test_polarity_signs(self, cfg):
        bull = [post(hours_ago=0.5, title="WIF moon pump bullish rocket")]
        bear = [post(hours_ago=0.5, title="WIF rug scam dump dead")]
        assert S.keyword_polarity(bull) == pytest.approx(1.0)
        assert S.keyword_polarity(bear) == pytest.approx(-1.0)
        assert S.keyword_polarity([]) is None


# ---------------------------------------------------------------------------
# Degradation — the important one
# ---------------------------------------------------------------------------


class TestDegradation:
    def test_raising_provider_returns_none_and_does_not_propagate(self, cfg):
        provider = StubProvider(raises=httpx.ConnectError("proxy said no"))
        assert S.brief(WIF, cfg, provider=provider, now=NOW) is None

    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ConnectError("dns"),
            httpx.ReadTimeout("slow"),
            json.JSONDecodeError("bad", "", 0),
            S.SourceUnavailable("everything is down"),
            KeyError("created_utc"),
            RuntimeError("something nobody predicted"),
        ],
    )
    def test_every_failure_shape_becomes_none(self, cfg, exc):
        assert S.brief(WIF, cfg, provider=StubProvider(raises=exc), now=NOW) is None
        assert S.brief(WIF, cfg, provider=StubPostSource(raises=exc), now=NOW) is None

    def test_zero_posts_is_a_valid_brief_not_none(self, cfg):
        """'Nobody is talking about this' is information, and is *not* the same
        claim as 'we could not find out'. The type must keep them apart."""
        b = S.brief(WIF, cfg, provider=StubProvider(posts=[]), now=NOW)
        assert b is not None
        assert isinstance(b, SentimentBrief)
        assert b.mention_velocity_1h == 0.0
        assert b.mention_velocity_24h == 0.0
        assert b.unique_contributors_24h == 0
        assert b.contributor_to_post_ratio is None
        assert b.top_posts == ()

    def test_silence_and_failure_are_distinguishable(self, cfg):
        silence = S.brief(WIF, cfg, provider=StubProvider(posts=[]), now=NOW)
        failure = S.brief(BONK, cfg, provider=StubProvider(raises=OSError("down")), now=NOW)
        assert silence is not None and failure is None

    def test_partial_source_failure_degrades_rather_than_fails(self, cfg):
        class PartiallyBroken(StubPostSource):
            def posts(self, coin, cfg, now):
                return [post(hours_ago=0.5)], ["CryptoCurrency"]

        b = S.brief(WIF, cfg, provider=PartiallyBroken(), now=NOW)
        assert b is not None
        assert b.degraded_reason and "CryptoCurrency" in b.degraded_reason

    def test_failure_is_not_cached(self, cfg):
        """A failed fetch must not poison the cache with a fake brief."""
        S.brief(WIF, cfg, provider=StubProvider(raises=OSError("down")), now=NOW)
        cache = S.load_cache(cfg)
        assert not cache["symbols"].get("WIF", {}).get("brief")

    def test_disabled_sentiment_returns_none(self, tmp_path):
        cfg = make_config(tmp_path, enabled=False)
        provider = StubProvider(posts=[post(hours_ago=0.5)])
        assert S.brief(WIF, cfg, provider=provider, now=NOW) is None
        assert provider.calls == 0

    def test_briefs_isolates_coins(self, cfg):
        class OnlyWifWorks:
            source = "arctic_shift"

            def fetch(self, coin, cfg):
                if coin.symbol != "WIF":
                    raise RuntimeError("BONK lookup exploded")
                return S.build_brief(coin, cfg, [], "arctic_shift", NOW, history=None)

        out = S.briefs(cfg, provider=OnlyWifWorks(), now=NOW)
        assert set(out) == {"WIF", "BONK"}
        assert out["WIF"] is not None
        assert out["BONK"] is None

    def test_corrupt_cache_file_does_not_fail_a_tick(self, cfg):
        S.cache_path(cfg).write_text("{not json at all", encoding="utf-8")
        b = S.brief(WIF, cfg, provider=StubProvider(posts=[post(hours_ago=0.5)]), now=NOW)
        assert b is not None


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCache:
    def test_inside_ttl_does_not_refetch(self, cfg):
        provider = StubPostSource(posts=[post(hours_ago=0.5, author="u1")])
        first = S.brief(WIF, cfg, provider=provider, now=NOW)
        second = S.brief(WIF, cfg, provider=provider, now=NOW + 60)
        third = S.brief(WIF, cfg, provider=provider, now=NOW + 599)

        assert provider.calls == 1
        assert first is not None and second is not None and third is not None
        assert second.mention_velocity_1h == first.mention_velocity_1h
        assert second.ts == first.ts  # the cached brief keeps its own timestamp

    def test_outside_ttl_refetches(self, cfg):
        provider = StubPostSource(posts=[post(hours_ago=0.5, author="u1")])
        S.brief(WIF, cfg, provider=provider, now=NOW)
        S.brief(WIF, cfg, provider=provider, now=NOW + 601)
        assert provider.calls == 2

    def test_cache_survives_a_process_restart(self, cfg):
        provider = StubPostSource(posts=[post(hours_ago=0.5, author="u1")])
        S.brief(WIF, cfg, provider=provider, now=NOW)
        # Nothing in memory is reused; the second call re-reads the file.
        again = S.brief(WIF, cfg, provider=StubPostSource(raises=OSError("boom")), now=NOW + 10)
        assert again is not None  # served from disk, so the outage is invisible

    def test_cache_hit_says_top_posts_are_missing(self, cfg):
        provider = StubPostSource(posts=[post(hours_ago=0.5, author="u1", score=9)])
        fresh = S.brief(WIF, cfg, provider=provider, now=NOW)
        cached = S.brief(WIF, cfg, provider=provider, now=NOW + 60)
        assert fresh.top_posts and len(fresh.top_posts) == 1
        assert cached.top_posts == ()
        assert "cache" in (cached.degraded_reason or "")

    def test_cache_is_per_symbol(self, cfg):
        provider = StubPostSource(posts=[post(hours_ago=0.5, title="WIF and BONK")])
        S.brief(WIF, cfg, provider=provider, now=NOW)
        S.brief(BONK, cfg, provider=provider, now=NOW)
        assert set(S.load_cache(cfg)["symbols"]) == {"WIF", "BONK"}

    def test_briefs_writes_the_cache_once(self, cfg):
        provider = StubPostSource(posts=[post(hours_ago=0.5, title="WIF and BONK too")])
        S.briefs(cfg, provider=provider, now=NOW)
        assert S.cache_path(cfg).exists()
        assert set(S.load_cache(cfg)["symbols"]) == {"WIF", "BONK"}

    def test_baseline_accumulates_across_runs(self, cfg):
        """The whole reason the history is on disk."""
        provider = StubPostSource(posts=[post(hours_ago=0.5, author="u1")])
        S.brief(WIF, cfg, provider=provider, now=NOW)
        hourly_1 = S.load_cache(cfg)["symbols"]["WIF"]["hourly"]
        assert len(hourly_1) == cfg.sentiment.lookback_hours + 1

        # A day later, with the cache expired, the old buckets are still there.
        # The second sweep reads posts that say nothing about WIF rather than
        # no posts at all: an empty sweep now means "we did not look", and a
        # tick that did not look must not extend the baseline. Quiet hours that
        # were genuinely observed still have to land, which is what this
        # asserts.
        later = NOW + 24 * HOUR
        quiet = StubPostSource(
            posts=[S.Post("q1", later - 600, "u9", "BONK chatter only", "", 1, "solana")]
        )
        S.brief(WIF, cfg, provider=quiet, now=later)
        hourly_2 = S.load_cache(cfg)["symbols"]["WIF"]["hourly"]
        assert len(hourly_2) > len(hourly_1)
        assert set(hourly_1) < set(hourly_2)
        assert sum(hourly_2.values()) == 1  # the one WIF post from the first run


class TestCachePrivacy:
    """Reddit's Data API terms plus basic hygiene: the cache is counts, not a
    corpus. See the note in the module docstring."""

    SECRET_BODY = "ZZZ_SECRET_POST_BODY_ZZZ"
    SECRET_TITLE = "ZZZ_SECRET_TITLE_WIF_ZZZ"
    SECRET_AUTHOR = "ZZZ_secret_username_ZZZ"

    def _run(self, cfg):
        provider = StubPostSource(
            posts=[
                S.Post(
                    id="abc123",
                    created_utc=NOW - 600,
                    author=self.SECRET_AUTHOR,
                    title=self.SECRET_TITLE,
                    body=self.SECRET_BODY,
                    score=42,
                    subreddit="solana",
                )
            ]
        )
        b = S.brief(WIF, cfg, provider=provider, now=NOW)
        assert b is not None and b.unique_contributors_24h == 1
        return S.cache_path(cfg).read_text(encoding="utf-8")

    def test_no_post_body_on_disk(self, cfg):
        assert self.SECRET_BODY not in self._run(cfg)

    def test_no_title_on_disk(self, cfg):
        raw = self._run(cfg)
        assert self.SECRET_TITLE not in raw
        assert "title" not in raw

    def test_no_plaintext_username_on_disk(self, cfg):
        raw = self._run(cfg)
        assert self.SECRET_AUTHOR not in raw
        assert self.SECRET_AUTHOR.lower() not in raw.lower()

    def test_cache_holds_only_counts_and_timestamps(self, cfg):
        self._run(cfg)
        entry = S.load_cache(cfg)["symbols"]["WIF"]
        # ``unit`` names what the hourly buckets counted. It is a fixed label,
        # not user content, and it is what lets an ``include_comments`` flip
        # invalidate one symbol instead of the whole file.
        assert set(entry) == {"brief", "fetched_at", "hourly", "unit"}
        assert entry["unit"] in {"submissions", "submissions+comments"}
        assert all(isinstance(v, int) for v in entry["hourly"].values())
        assert all(k.isdigit() for k in entry["hourly"])
        assert "top_posts" not in entry["brief"]
        # Every persisted value is a number, a symbol, a source name or a reason.
        assert set(entry["brief"]) == {
            "symbol",
            "ts",
            "source",
            "mention_velocity_1h",
            "mention_velocity_24h",
            "mention_zscore_7d",
            "unique_contributors_24h",
            "contributor_to_post_ratio",
            "polarity",
            "degraded_reason",
        }

    def test_author_hash_is_stable_and_not_reversible(self):
        h = S._author_hash("SomeUser")
        assert h == S._author_hash("someuser  ")
        assert "someuser" not in h
        assert h != S._author_hash("someuser2")


# ---------------------------------------------------------------------------
# ArcticShiftProvider against a mock transport — no network
# ---------------------------------------------------------------------------


def _no_comments(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"data": []})


def _arctic_client(handler, comments=None) -> httpx.Client:
    """Mock transport routing the two search endpoints separately.

    They are genuinely different endpoints with different field whitelists, so
    a single handler answering both would let a test pass that the live host
    would 400. ``comments`` defaults to an empty page: a test about the
    submission sweep says so by not modelling comments, rather than by
    accidentally receiving submission rows on the comment endpoint and
    double-counting every post.
    """

    def route(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/comments/search"):
            return (comments or _no_comments)(request)
        return handler(request)

    return httpx.Client(transport=httpx.MockTransport(route), base_url="http://test")


class TestArcticShiftProvider:
    @pytest.fixture(autouse=True)
    def _no_sleeping(self, monkeypatch):
        monkeypatch.setattr(S, "_ARCTIC_SPACING_S", 0.0)

    def test_parses_the_real_recorded_response(self, cfg):
        """The fixture is a genuine response captured from the live host."""
        payload = json.loads((FIXTURES / "arctic_shift_sample.json").read_text(encoding="utf-8"))
        assert "data" in payload and isinstance(payload["data"], list) and payload["data"]

        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if len(seen) > len(cfg.sentiment.subreddits):
                return httpx.Response(200, json={"data": []})
            return httpx.Response(200, json=payload)

        provider = S.ArcticShiftProvider(client=_arctic_client(handler))
        posts, failures = provider.posts(WIF, cfg, time.time())

        assert failures == []
        assert posts and all(isinstance(p, S.Post) for p in posts)
        assert all(p.created_utc > 1_600_000_000 for p in posts)  # seconds, not ms

        params = seen[0].url.params
        assert params["subreddit"] == "CryptoCurrency"
        assert params["limit"] == str(S._ARCTIC_MAX_LIMIT)
        assert params["sort"] == "desc"
        assert "query" not in params  # server-side search is deliberately unused
        assert int(params["before"]) - int(params["after"]) == pytest.approx(
            cfg.sentiment.lookback_hours * 3600, abs=5
        )

    def test_one_dead_subreddit_degrades(self, cfg):
        def handler(request):
            if "solana" in str(request.url):
                return httpx.Response(500, json={"data": None, "error": "boom"})
            return httpx.Response(200, json={"data": [_raw(id="k1", title="WIF up")]})

        provider = S.ArcticShiftProvider(client=_arctic_client(handler))
        posts, failures = provider.posts(WIF, cfg, NOW)
        assert failures == ["solana"]
        assert len(posts) == 1

    def test_all_subreddits_dead_raises_source_unavailable(self, cfg):
        provider = S.ArcticShiftProvider(
            client=_arctic_client(
                lambda r: httpx.Response(422, json={"data": None, "error": "Timeout. Maybe slow down a bit"})
            )
        )
        with pytest.raises(S.SourceUnavailable):
            provider.posts(WIF, cfg, NOW)
        # ...and brief() turns that into None rather than an exception.
        assert S.brief(WIF, cfg, provider=provider, now=NOW) is None

    def test_paginates_backwards_on_a_full_page(self, cfg):
        now = NOW
        pages = {0: [_raw(id=f"a{i}", created=now - i * 60) for i in range(100)]}

        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(200, json={"data": pages[0]})
            return httpx.Response(200, json={"data": [_raw(id="tail", created=now - 20000)]})

        provider = S.ArcticShiftProvider(client=_arctic_client(handler))
        posts, _ = provider.posts(WIF, cfg, now)
        assert calls["n"] > len(cfg.sentiment.subreddits)  # it turned a page
        assert any(p.id == "tail" for p in posts)

    def test_malformed_rows_are_skipped_not_fatal(self, cfg):
        good = _raw(id="ok", title="WIF up")
        bad = {"id": "broken", "title": "WIF up"}  # no created_utc
        provider = S.ArcticShiftProvider(
            client=_arctic_client(lambda r: httpx.Response(200, json={"data": [good, bad]}))
        )
        posts, failures = provider.posts(WIF, cfg, NOW)
        assert failures == []
        assert {p.id for p in posts} == {"ok"}

    def test_one_sweep_serves_every_coin(self, cfg):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(200, json={"data": [_raw(id="x", title="WIF and BONK")]})

        provider = S.ArcticShiftProvider(client=_arctic_client(handler))
        out = S.briefs(cfg, provider=provider, now=NOW)
        # Both coins served, and only one sweep of HTTP calls to do it. The
        # "both coins served" half matters: without it, a crash on the second
        # coin looks identical to a cache hit from here.
        assert out["WIF"] is not None and out["BONK"] is not None
        assert out["WIF"].mention_velocity_24h > 0
        assert out["BONK"].mention_velocity_24h > 0
        assert calls["n"] == len(cfg.sentiment.subreddits)  # not x2 for two coins

    def test_every_subreddit_answering_empty_is_not_a_failure(self, cfg):
        """HTTP 200 with ``{"data": []}`` is the live shape for a subreddit
        with nothing in the window, so it never raises and never lands in
        ``failures``. That is correct on its own and was the trap: nothing
        downstream could tell this apart from a healthy read of a silent
        Reddit."""
        provider = S.ArcticShiftProvider(
            client=_arctic_client(lambda r: httpx.Response(200, json={"data": []}))
        )
        posts, failures = provider.posts(WIF, cfg, NOW)
        assert posts == []
        assert failures == []
        assert S.sweep_observed_through(provider, posts, NOW) is None

    def test_an_all_empty_sweep_reaches_the_model_as_unmeasured(self, cfg):
        """The end of that trap. Before ``sweep_size``, this exact response
        produced ``mention_velocity_1h = 0.0``, ``mention_velocity_24h = 0.0``,
        ``unique_contributors_24h = 0`` and ``degraded_reason = None`` — the
        most confident wrong number this module can emit, with no marker on
        it at all."""
        provider = S.ArcticShiftProvider(
            client=_arctic_client(lambda r: httpx.Response(200, json={"data": []}))
        )
        b = S.brief(WIF, cfg, provider=provider, now=NOW)
        assert b is not None
        assert b.mention_velocity_1h is None
        assert b.mention_velocity_24h is None
        assert b.mention_zscore_7d is None
        assert "no posts at all" in (b.degraded_reason or "")
        # And nothing fabricated was persisted for the baseline to inherit.
        assert S.load_cache(cfg)["symbols"]["WIF"]["hourly"] == {}

    def test_a_sweep_with_no_mention_of_this_coin_is_a_real_zero(self, cfg):
        """What the live sweep actually looks like. On 2026-09-19 it returned
        143 posts across the five subreddits and not one mentioned BONK, WIF
        or POPCAT — which is the expected outcome at these base rates (23 BONK
        submissions in the preceding 90 days), not a defect. It must come
        through as 0.0 with the denominator attached, never as ``None``."""
        rows = [
            _raw(id="n1", title="Robux has to move right??!"),
            _raw(id="n2", title="What is the most amount of money you have made in memecoins?"),
        ]
        provider = S.ArcticShiftProvider(
            client=_arctic_client(lambda r: httpx.Response(200, json={"data": rows}))
        )
        b = S.brief(WIF, cfg, provider=provider, now=NOW)
        assert b is not None
        assert b.mention_velocity_24h == 0.0
        assert b.unique_contributors_24h == 0
        assert "0 mentions of WIF" in (b.degraded_reason or "")
        assert "2 posts" in (b.degraded_reason or "")

    def test_base_url_is_the_verified_one(self):
        assert S.ARCTIC_SHIFT_BASE == "https://arctic-shift.photon-reddit.com/api"


def _raw(*, id: str, created: float | None = None, title: str = "WIF is up", author: str = "u1") -> dict:
    return {
        "id": id,
        "created_utc": created if created is not None else NOW - 600,
        "author": author,
        "title": title,
        "score": 3,
        "subreddit": "solana",
        "num_comments": 1,
        "selftext": "",
    }


# ---------------------------------------------------------------------------
# PrawProvider against a stub — the live path cannot be tested without an app
# ---------------------------------------------------------------------------


class FakeAuthor:
    def __init__(self, name):
        self.name = name


class FakeSubmission:
    def __init__(self, id, title, author, created_utc, score=1, selftext="", subreddit="solana"):
        self.id = id
        self.title = title
        self.author = FakeAuthor(author) if author else None
        self.created_utc = created_utc
        self.score = score
        self.selftext = selftext
        self.subreddit = subreddit


class FakeComment:
    """A PRAW ``Comment``: a body, no title. The shape difference is the point."""

    def __init__(self, id, body, author, created_utc, score=1, subreddit="solana"):
        self.id = id
        self.body = body
        self.author = FakeAuthor(author) if author else None
        self.created_utc = created_utc
        self.score = score
        self.subreddit = subreddit


class FakeSubreddit:
    def __init__(self, display_name, results, recorder, comments=()):
        self.display_name = display_name
        self._results = results
        self._recorder = recorder
        self._comments = list(comments)

    def comments(self, **kwargs):
        # In praw 8.0.3 ``Subreddit.comments`` is a cachedproperty yielding a
        # callable CommentHelper, so ``subreddit.comments(limit=N)`` is the real
        # call shape; a plain method reproduces it from the caller's side.
        self._recorder.append({"subreddit": self.display_name, "listing": "comments", **kwargs})
        ordered = sorted(self._comments, key=lambda c: c.created_utc, reverse=True)
        limit = kwargs.get("limit")
        return iter(ordered if limit is None else ordered[:limit])

    def new(self, **kwargs):
        # ``new`` takes only listing kwargs in PRAW 8. Results are handed back
        # newest-first, which is what the real listing guarantees and what the
        # sweep's early stop depends on.
        self._recorder.append({"subreddit": self.display_name, **kwargs})
        ordered = sorted(self._results, key=lambda s: s.created_utc, reverse=True)
        limit = kwargs.get("limit")
        return iter(ordered if limit is None else ordered[:limit])

    def search(self, *a, **kw):  # pragma: no cover - must never be reached
        raise AssertionError(
            "PrawProvider must sweep /new, not search: Reddit's search index is "
            "populated asynchronously and lags the listing."
        )


class FakeReddit:
    def __init__(self, results, comments=()):
        self._results = results
        self._comments = list(comments)
        self.listings: list[dict] = []
        self.requested: list[str] = []
        self.read_only = False

    def subreddit(self, display_name):
        self.requested.append(display_name)
        return FakeSubreddit(display_name, self._results, self.listings, self._comments)


class UnsortedReddit(FakeReddit):
    """Hands back results in the order given, not newest-first.

    Reddit's ``/new`` is reverse-chronological *except* for stickies, so the
    sorted stub cannot express the one case the sweep's early stop has to
    survive.
    """

    def subreddit(self, display_name):
        self.requested.append(display_name)
        sub = FakeSubreddit(display_name, self._results, self.listings)
        sub.new = lambda **kw: iter(self._results)  # type: ignore[method-assign]
        return sub


class TestPrawProvider:
    def test_builds_posts_from_submissions(self, cfg):
        now = time.time()
        reddit = FakeReddit(
            [
                FakeSubmission("s1", "WIF is pumping", "alice", now - 600, score=50),
                FakeSubmission("s2", "my wife hates crypto", "bob", now - 900),
                FakeSubmission("s3", "dogwifhat gang", None, now - 1200),  # deleted author
            ]
        )
        provider = S.PrawProvider(reddit=reddit)
        posts, failures = provider.posts(WIF, cfg, now)
        assert failures == []
        assert {p.id for p in posts} == {"s1", "s2", "s3"}
        # "my wife" must not survive the local word-boundary re-check. Under
        # `search` Reddit's stemmer was what handed it to us; under `/new` the
        # sweep is unfiltered, so this check is now the *only* thing standing
        # between a stray substring and a mention count.
        assert {p.id for p in S.matching_posts(posts, WIF.aliases)} == {"s1", "s3"}

    def test_sweeps_new_per_subreddit_not_a_search(self, cfg):
        reddit = FakeReddit([])
        S.PrawProvider(reddit=reddit).posts(WIF, cfg, time.time())
        # One listing per subreddit — and emphatically not a `a+b` multireddit
        # search, which would put Reddit's lagging search index in the path.
        assert reddit.requested == ["CryptoCurrency", "solana"]
        # Two listings per subreddit now — ``/new`` and ``/comments`` — and
        # still emphatically not a `a+b` multireddit search, which would put
        # Reddit's lagging search index in the path.
        assert [c["subreddit"] for c in reddit.listings] == [
            "CryptoCurrency",
            "CryptoCurrency",
            "solana",
            "solana",
        ]
        new = [c for c in reddit.listings if "listing" not in c]
        comments = [c for c in reddit.listings if c.get("listing") == "comments"]
        assert [c["subreddit"] for c in new] == ["CryptoCurrency", "solana"]
        assert all(c["limit"] == S._PRAW_NEW_LIMIT for c in new)
        assert [c["subreddit"] for c in comments] == ["CryptoCurrency", "solana"]
        assert all(c["limit"] == S._PRAW_COMMENT_LIMIT for c in comments)

    def test_the_sweep_is_shared_across_coins(self, cfg):
        now = time.time()
        reddit = FakeReddit([FakeSubmission("s1", "WIF and BONK", "alice", now - 300)])
        provider = S.PrawProvider(reddit=reddit)
        provider.posts(WIF, cfg, now)
        provider.posts(BONK, cfg, now)
        # Three coins must not cost three sweeps: the window does not depend on
        # which coin is asking.
        assert reddit.requested == ["CryptoCurrency", "solana"]

    def test_posts_outside_the_window_are_dropped(self, cfg):
        now = time.time()
        old = now - 30 * HOUR  # cfg lookback is 24h
        reddit = FakeReddit(
            [
                FakeSubmission("fresh", "WIF now", "alice", now - 300),
                FakeSubmission("old1", "WIF then", "bob", old),
                FakeSubmission("old2", "WIF then", "carol", old - 60),
                FakeSubmission("old3", "WIF then", "dave", old - 120),
                FakeSubmission("old4", "WIF then", "erin", old - 180),
            ]
        )
        posts, _ = S.PrawProvider(reddit=reddit).posts(WIF, cfg, now)
        assert {p.id for p in posts} == {"fresh"}

    def test_a_single_out_of_order_post_does_not_end_the_sweep(self, cfg):
        """A pinned submission surfacing out of order must not truncate the sub.

        ``/new`` is reverse-chronological, so it is tempting to stop at the
        first old post — but one stray would then cost the entire subreddit.
        """
        now = time.time()
        reddit = UnsortedReddit(
            # An old sticky at the head of an otherwise fresh listing.
            [FakeSubmission("pinned", "WIF megathread", "mod", now - 40 * HOUR)]
            + [FakeSubmission(f"s{i}", "WIF talk", f"u{i}", now - 300 - i) for i in range(4)]
        )
        posts, _ = S.PrawProvider(reddit=reddit).posts(WIF, cfg, now)
        assert {p.id for p in posts} == {"s0", "s1", "s2", "s3"}

    def test_a_run_of_old_posts_does_end_the_sweep(self, cfg):
        """The tolerance is for strays, not a licence to read the whole listing."""
        now = time.time()
        old = now - 40 * HOUR
        reddit = UnsortedReddit(
            [FakeSubmission(f"old{i}", "WIF", f"u{i}", old - i) for i in range(S._PRAW_STALE_RUN)]
            + [FakeSubmission("unreached", "WIF", "late", now - 300)]
        )
        posts, _ = S.PrawProvider(reddit=reddit).posts(WIF, cfg, now)
        assert posts == []

    def test_one_dead_subreddit_is_reported_not_swallowed(self, cfg):
        now = time.time()

        class HalfDead(FakeReddit):
            def subreddit(self, display_name):
                if display_name == "solana":
                    raise RuntimeError("prawcore: 404 banned")
                return super().subreddit(display_name)

        reddit = HalfDead([FakeSubmission("s1", "WIF up", "alice", now - 300)])
        posts, failures = S.PrawProvider(reddit=reddit).posts(WIF, cfg, now)
        assert [p.id for p in posts] == ["s1"]
        # A lower mention count with no explanation is worse than no count.
        assert failures == ["solana"]

    def test_end_to_end_brief(self, cfg):
        now = time.time()
        reddit = FakeReddit(
            [FakeSubmission(f"s{i}", "WIF moon", f"u{i}", now - 300, score=i) for i in range(5)]
        )
        b = S.brief(WIF, cfg, provider=S.PrawProvider(reddit=reddit), now=now)
        assert b is not None
        assert b.source == "praw"
        assert b.mention_velocity_1h == 5.0
        assert b.unique_contributors_24h == 5

    def test_deleted_author_is_not_a_contributor(self, cfg):
        now = time.time()
        reddit = FakeReddit([FakeSubmission("s1", "WIF up", None, now - 300)])
        posts, _ = S.PrawProvider(reddit=reddit).posts(WIF, cfg, now)
        assert posts[0].author == "[deleted]"
        b = S.build_brief(WIF, cfg, posts, "praw", now, history=None)
        assert b.unique_contributors_24h == 0

    def test_no_subreddits_configured_is_source_unavailable(self, tmp_path):
        cfg = make_config(tmp_path, subreddits=())
        with pytest.raises(S.SourceUnavailable):
            S.PrawProvider(reddit=FakeReddit([])).posts(WIF, cfg, time.time())
        assert S.brief(WIF, cfg, provider=S.PrawProvider(reddit=FakeReddit([])), now=NOW) is None

    def test_an_empty_new_sweep_is_not_reported_as_silence(self, cfg):
        """PRAW sweeps whole subreddits too, so an empty sweep here means the
        same thing it means on the archive path: we read nothing. The lag
        machinery cannot catch it — ``index_lags`` is ``False`` for a live
        listing, so ``observed_through`` stays ``None`` and every rate would
        otherwise come out a confident zero."""
        b = S.brief(WIF, cfg, provider=S.PrawProvider(reddit=FakeReddit([])), now=NOW)
        assert b is not None
        assert b.mention_velocity_1h is None
        assert b.mention_velocity_24h is None
        assert "no posts at all" in (b.degraded_reason or "")

    def test_a_praw_explosion_never_escapes_brief(self, cfg):
        class Exploding(FakeReddit):
            def subreddit(self, display_name):
                raise RuntimeError("prawcore.exceptions.ResponseException: 401")

        assert S.brief(WIF, cfg, provider=S.PrawProvider(reddit=Exploding([])), now=NOW) is None


class TestProviderSelection:
    def test_arctic_shift_without_credentials(self, cfg):
        assert isinstance(S.default_provider(cfg), S.ArcticShiftProvider)

    def test_praw_with_credentials(self, tmp_path):
        cfg = make_config(tmp_path, reddit_client_id="abc", reddit_client_secret="def")
        assert cfg.sentiment.has_reddit_credentials
        assert isinstance(S.default_provider(cfg), S.PrawProvider)

    def test_partial_credentials_fall_back(self, tmp_path):
        cfg = make_config(tmp_path, reddit_client_id="abc", reddit_client_secret=None)
        assert isinstance(S.default_provider(cfg), S.ArcticShiftProvider)


# ---------------------------------------------------------------------------
# Comments — the second half of the sweep
# ---------------------------------------------------------------------------


def _raw_comment(
    *, id: str, created: float | None = None, body: str = "WIF is up", author: str = "u1"
) -> dict:
    """One row as ``/api/comments/search`` actually returns it.

    Note what is *absent*: no ``title``, no ``selftext``, no ``num_comments``.
    Those three are valid on ``/posts/search`` and 400 on this endpoint, which
    is why the two normalizers cannot be one function.
    """
    return {
        "id": id,
        "created_utc": created if created is not None else NOW - 600,
        "author": author,
        "body": body,
        "score": 3,
        "subreddit": "solana",
        "link_id": "t3_abc",
    }


class TestCommentNormalization:
    def test_a_comment_becomes_a_post_with_no_title(self):
        c = S._comment_from_arctic(_raw_comment(id="c1", body="bought more $WIF"))
        assert c is not None
        assert c.kind == "comment"
        assert c.title == ""
        assert c.body == "bought more $WIF"
        # ``text`` is what ``matching_posts`` reads, so an empty title must not
        # cost the body its match.
        assert "bought more $WIF" in c.text

    def test_a_submission_still_defaults_to_submission_kind(self):
        p = S._post_from_arctic(_raw(id="s1"))
        assert p is not None and p.kind == "submission"

    def test_post_kind_is_trailing_and_defaulted(self):
        """Positional construction is used throughout this file and elsewhere."""
        p = S.Post("q1", NOW - 600, "u9", "BONK chatter only", "", 1, "solana")
        assert p.kind == "submission"

    def test_unusable_comment_rows_are_dropped_not_faked(self):
        assert S._comment_from_arctic({"id": "c1"}) is None  # no created_utc
        assert S._comment_from_arctic({"id": "c1", "created_utc": "nope"}) is None

    def test_a_comment_is_matched_on_its_body_alone(self):
        c = S._comment_from_arctic(_raw_comment(id="c1", body="my wife says no"))
        assert c is not None
        # The word-boundary rule is shared, so "wife" is still not a WIF mention.
        assert S.matching_posts([c], WIF.aliases) == []

    def test_praw_comment_normalizes_the_same_way(self):
        c = S._comment_from_praw(
            FakeComment("c1", "WIF to the moon", "alice", NOW - 600, score=7)
        )
        assert c is not None
        assert (c.kind, c.title, c.body, c.score) == ("comment", "", "WIF to the moon", 7)


class TestCommentSweep:
    @pytest.fixture(autouse=True)
    def _no_sleeping(self, monkeypatch):
        monkeypatch.setattr(S, "_ARCTIC_SPACING_S", 0.0)

    def test_arctic_uses_the_comment_field_whitelist(self, cfg):
        """A shared ``fields`` constant would 400 every comment request."""
        seen: list[httpx.Request] = []

        def comments(request):
            seen.append(request)
            return httpx.Response(200, json={"data": []})

        provider = S.ArcticShiftProvider(
            client=_arctic_client(lambda r: httpx.Response(200, json={"data": []}), comments)
        )
        provider.posts(WIF, cfg, NOW)

        assert seen, "the comments endpoint was never called"
        fields = seen[0].url.params["fields"].split(",")
        assert "body" in fields
        # The three that are valid for posts and rejected for comments.
        assert not {"title", "selftext", "num_comments"} & set(fields)

    def test_comments_and_submissions_merge_into_one_sweep(self, cfg):
        provider = S.ArcticShiftProvider(
            client=_arctic_client(
                lambda r: httpx.Response(200, json={"data": [_raw(id="s1", title="WIF up")]}),
                lambda r: httpx.Response(
                    200, json={"data": [_raw_comment(id="c1", body="WIF up too")]}
                ),
            )
        )
        posts, failures = provider.posts(WIF, cfg, NOW)
        assert failures == []
        assert sorted(p.kind for p in posts) == ["comment", "submission"]

    def test_a_shared_id_across_kinds_does_not_collide(self, cfg):
        """Submission and comment ids come from different Reddit namespaces.

        Keying the sweep by bare id would silently drop one of the two and
        understate both the mention count and ``sweep_size``.
        """
        provider = S.ArcticShiftProvider(
            client=_arctic_client(
                lambda r: httpx.Response(200, json={"data": [_raw(id="dup", title="WIF up")]}),
                lambda r: httpx.Response(
                    200, json={"data": [_raw_comment(id="dup", body="WIF up")]}
                ),
            )
        )
        posts, _ = provider.posts(WIF, cfg, NOW)
        assert len(posts) == 2

    def test_sweep_size_counts_both_kinds(self, cfg):
        """The denominator behind a reported zero must include comments."""
        provider = S.ArcticShiftProvider(
            client=_arctic_client(
                lambda r: httpx.Response(200, json={"data": [_raw(id="s1", title="doge news")]}),
                lambda r: httpx.Response(
                    200,
                    json={
                        "data": [
                            _raw_comment(id="c1", body="nothing here"),
                            _raw_comment(id="c2", body="nor here"),
                        ]
                    },
                ),
            )
        )
        b = S.brief(WIF, cfg, provider=provider, now=NOW)
        assert b is not None
        assert "the sweep observed 3 posts and comments" in (b.degraded_reason or "")

    def test_comments_off_behaves_exactly_as_before(self, tmp_path):
        cfg = make_config(tmp_path, include_comments=False)
        called: list[str] = []

        def comments(request):  # pragma: no cover - must never be reached
            called.append(str(request.url))
            return httpx.Response(200, json={"data": []})

        provider = S.ArcticShiftProvider(
            client=_arctic_client(
                lambda r: httpx.Response(200, json={"data": [_raw(id="s1", title="WIF up")]}),
                comments,
            )
        )
        posts, _ = provider.posts(WIF, cfg, NOW)
        assert called == []
        assert posts and all(p.kind == "submission" for p in posts)

    def test_comments_off_says_posts_not_posts_and_comments(self, tmp_path):
        cfg = make_config(tmp_path, include_comments=False)
        provider = S.ArcticShiftProvider(
            client=_arctic_client(
                lambda r: httpx.Response(200, json={"data": [_raw(id="s1", title="doge news")]})
            )
        )
        b = S.brief(WIF, cfg, provider=provider, now=NOW)
        assert b is not None
        assert "the sweep observed 1 posts over" in (b.degraded_reason or "")

    def test_praw_sweeps_comments_alongside_new(self, cfg):
        reddit = FakeReddit(
            [FakeSubmission("s1", "doge news", "alice", NOW - 600)],
            [FakeComment("c1", "WIF looks strong", "bob", NOW - 300, score=5)],
        )
        posts, failures = S.PrawProvider(reddit=reddit).posts(WIF, cfg, NOW)
        assert failures == []
        matched = S.matching_posts(posts, WIF.aliases)
        assert matched and all(p.kind == "comment" for p in matched)

    def test_praw_comments_off_skips_the_listing(self, tmp_path):
        cfg = make_config(tmp_path, include_comments=False)
        reddit = FakeReddit(
            [FakeSubmission("s1", "WIF up", "alice", NOW - 600)],
            [FakeComment("c1", "WIF looks strong", "bob", NOW - 300)],
        )
        posts, _ = S.PrawProvider(reddit=reddit).posts(WIF, cfg, NOW)
        assert all(c.get("listing") != "comments" for c in reddit.listings)
        assert posts and all(p.kind == "submission" for p in posts)

    def test_praw_comments_outside_the_window_are_dropped(self, cfg):
        reddit = FakeReddit(
            [],
            [
                FakeComment("c1", "WIF now", "bob", NOW - 300),
                FakeComment("c2", "WIF ages ago", "bob", NOW - 40 * HOUR),
                FakeComment("c3", "WIF older still", "bob", NOW - 41 * HOUR),
                FakeComment("c4", "WIF oldest", "bob", NOW - 42 * HOUR),
            ],
        )
        posts, _ = S.PrawProvider(reddit=reddit).posts(WIF, cfg, NOW)
        assert {p.id for p in posts} == {"c1"}


class TestPartialCommentCoverage:
    """The 422 is a per-range server-side timeout, not rate limiting.

    No page budget completes a 24h comment walk of a busy subreddit, so the
    only honest response is to keep what was read and say how much of the
    window it covered. A silently short window would read as a quiet subreddit.
    """

    @pytest.fixture(autouse=True)
    def _no_sleeping(self, monkeypatch):
        monkeypatch.setattr(S, "_ARCTIC_SPACING_S", 0.0)

    def _provider(self, pages_before_422: int):
        state = {"n": 0}

        def comments(request):
            state["n"] += 1
            if state["n"] > pages_before_422:
                return httpx.Response(
                    422, json={"data": None, "error": "Timeout. Maybe slow down a bit"}
                )
            # A full page, so the walk tries to page backwards again.
            page = state["n"]
            return httpx.Response(
                200,
                json={
                    "data": [
                        _raw_comment(id=f"c{page}-{i}", created=NOW - 60 - i * 60)
                        for i in range(S._ARCTIC_MAX_LIMIT)
                    ]
                },
            )

        return S.ArcticShiftProvider(
            client=_arctic_client(
                lambda r: httpx.Response(200, json={"data": [_raw(id="s1", title="WIF up")]}),
                comments,
            )
        )

    def test_a_walk_that_stops_short_keeps_what_it_read(self, cfg):
        provider = self._provider(pages_before_422=2)
        posts, failures = provider.posts(WIF, cfg, NOW)
        # Not a failure: the submissions came back and so did two comment pages.
        assert failures == []
        assert sum(1 for p in posts if p.kind == "comment") > 0

    def test_partial_coverage_is_reported_not_hidden(self, cfg):
        provider = self._provider(pages_before_422=2)
        provider.posts(WIF, cfg, NOW)
        note = " ".join(provider.sweep_notes)
        assert "comment sweep covered less than the requested 24h window" in note
        assert "floor" in note

    def test_the_note_reaches_the_brief(self, cfg):
        b = S.brief(WIF, cfg, provider=self._provider(pages_before_422=2), now=NOW)
        assert b is not None
        assert "comment sweep covered less than" in (b.degraded_reason or "")

    def test_a_first_page_failure_is_a_gap_not_a_dead_subreddit(self, cfg):
        """Submissions already landed, so the subreddit is not in ``failures``."""
        provider = S.ArcticShiftProvider(
            client=_arctic_client(
                lambda r: httpx.Response(200, json={"data": [_raw(id="s1", title="WIF up")]}),
                lambda r: httpx.Response(422, json={"data": None, "error": "Timeout."}),
            )
        )
        posts, failures = provider.posts(WIF, cfg, NOW)
        assert failures == []
        assert posts and all(p.kind == "submission" for p in posts)
        assert "comment sweep covered less than" in " ".join(provider.sweep_notes)

    def test_a_complete_walk_says_nothing(self, cfg):
        provider = S.ArcticShiftProvider(
            client=_arctic_client(
                lambda r: httpx.Response(200, json={"data": [_raw(id="s1", title="WIF up")]}),
                lambda r: httpx.Response(200, json={"data": [_raw_comment(id="c1")]}),
            )
        )
        provider.posts(WIF, cfg, NOW)
        assert provider.sweep_notes == []

    def test_praw_reports_hitting_the_pagination_ceiling(self, cfg, monkeypatch):
        monkeypatch.setattr(S, "_PRAW_COMMENT_LIMIT", 3)
        reddit = FakeReddit(
            [],
            [FakeComment(f"c{i}", "WIF", "bob", NOW - 60 - i) for i in range(10)],
        )
        provider = S.PrawProvider(reddit=reddit)
        provider.posts(WIF, cfg, NOW)
        note = " ".join(provider.sweep_notes)
        assert "pagination limit" in note and "floor" in note


class TestBaselineUnit:
    """A bucket counted over submissions+comments is not comparable to one
    counted over submissions alone, and mixing them would read as a burst of
    attention on every coin at once."""

    def _run(self, cfg, now):
        provider = StubPostSource([post(hours_ago=0.2, title="WIF is pumping", author="alice")])
        return S.brief(WIF, cfg, provider=provider, now=now)

    def test_the_unit_is_recorded_on_the_entry(self, cfg):
        self._run(cfg, NOW)
        assert S.load_cache(cfg)["symbols"]["WIF"]["unit"] == "submissions+comments"

    def test_turning_comments_off_records_the_other_unit(self, tmp_path):
        cfg = make_config(tmp_path, include_comments=False)
        self._run(cfg, NOW)
        assert S.load_cache(cfg)["symbols"]["WIF"]["unit"] == "submissions"

    def test_flipping_the_toggle_discards_that_symbols_history(self, tmp_path):
        on = make_config(tmp_path, include_comments=True, cache_ttl_seconds=0)
        self._run(on, NOW)
        assert S.load_cache(on)["symbols"]["WIF"]["hourly"]

        off = make_config(tmp_path, include_comments=False, cache_ttl_seconds=0)
        b = self._run(off, NOW + 3600)
        assert b is not None
        assert "baseline discarded" in (b.degraded_reason or "")
        entry = S.load_cache(off)["symbols"]["WIF"]
        assert entry["unit"] == "submissions"
        # Rebuilt from this tick's window only — the old buckets are gone, not
        # merged into a baseline that would now be measuring two things.
        assert len(entry["hourly"]) <= off.sentiment.lookback_hours + 2

    def test_an_unchanged_unit_keeps_the_history(self, tmp_path):
        cfg = make_config(tmp_path, include_comments=True, cache_ttl_seconds=0)
        self._run(cfg, NOW)
        before = dict(S.load_cache(cfg)["symbols"]["WIF"]["hourly"])
        b = self._run(cfg, NOW + 3600)
        assert b is not None
        assert "baseline discarded" not in (b.degraded_reason or "")
        after = S.load_cache(cfg)["symbols"]["WIF"]["hourly"]
        assert set(before) & set(after)

    def test_the_version_bump_discards_a_v1_file(self, cfg):
        """Version 1 buckets counted submissions only and cannot be rescued."""
        S.cache_path(cfg).write_text(
            json.dumps({"version": 1, "symbols": {"WIF": {"hourly": {"1": 9}}}}),
            encoding="utf-8",
        )
        assert S.CACHE_VERSION == 2
        assert S.load_cache(cfg) == {"version": 2, "symbols": {}}


class TestCommentPrivacy:
    """A comment body is user content on the same terms as a submission."""

    SECRET = "zzsecretcommentbodyzz"

    def test_no_comment_body_or_author_reaches_disk(self, cfg):
        provider = StubPostSource(
            [
                S.Post(
                    "c1",
                    NOW - 600,
                    "zzsecretcommenterzz",
                    "",
                    f"WIF {self.SECRET}",
                    4,
                    "solana",
                    "comment",
                )
            ]
        )
        b = S.brief(WIF, cfg, provider=provider, now=NOW)
        assert b is not None and b.unique_contributors_24h == 1
        # It did reach the *prompt* — that is the point of top_posts.
        assert self.SECRET in b.top_posts[0].title

        raw = S.cache_path(cfg).read_text(encoding="utf-8")
        assert self.SECRET not in raw
        assert "zzsecretcommenterzz" not in raw.lower()
        assert "body" not in raw

    def test_a_comment_top_post_carries_its_kind_and_its_text(self, cfg):
        """``report.py`` rendered ``p.title`` and printed nothing for a comment."""
        provider = StubPostSource(
            [S.Post("c1", NOW - 600, "bob", "", "WIF looks strong", 4, "solana", "comment")]
        )
        b = S.brief(WIF, cfg, provider=provider, now=NOW)
        assert b is not None
        top = b.top_posts[0]
        assert top.kind == "comment"
        assert top.title == "WIF looks strong"
