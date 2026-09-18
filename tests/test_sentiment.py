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
        model=ModelConfig("claude-opus-5", "high", 8000, 5.0, 25.0, 0.5),
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
        later = NOW + 24 * HOUR
        S.brief(WIF, cfg, provider=StubPostSource(posts=[]), now=later)
        hourly_2 = S.load_cache(cfg)["symbols"]["WIF"]["hourly"]
        assert len(hourly_2) > len(hourly_1)
        assert set(hourly_1) < set(hourly_2)


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
        assert set(entry) == {"brief", "fetched_at", "hourly"}
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


def _arctic_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")


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


class FakeSubreddit:
    def __init__(self, display_name, results, recorder):
        self.display_name = display_name
        self._results = results
        self._recorder = recorder

    def search(self, query, *, sort="relevance", syntax="lucene", time_filter="all", **kwargs):
        # Keyword-only, matching PRAW 8.0.3's actual signature. If sentiment.py
        # ever reverts to the 7.x positional style this raises TypeError.
        self._recorder.append(
            {"query": query, "sort": sort, "syntax": syntax, "time_filter": time_filter, **kwargs}
        )
        return iter(self._results)


class FakeReddit:
    def __init__(self, results):
        self._results = results
        self.searches: list[dict] = []
        self.requested: list[str] = []
        self.read_only = False

    def subreddit(self, display_name):
        self.requested.append(display_name)
        return FakeSubreddit(display_name, self._results, self.searches)


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
        # "my wife" must not survive the local word-boundary re-check, even
        # though Reddit's stemmer handed it to us.
        assert {p.id for p in S.matching_posts(posts, WIF.aliases)} == {"s1", "s3"}

    def test_searches_one_multireddit_with_ord_aliases(self, cfg):
        reddit = FakeReddit([])
        S.PrawProvider(reddit=reddit).posts(WIF, cfg, time.time())
        assert reddit.requested == ["CryptoCurrency+solana"]  # one call, not two
        call = reddit.searches[0]
        assert '"WIF"' in call["query"] and '"dogwifhat"' in call["query"]
        assert " OR " in call["query"]
        assert call["sort"] == "new"
        assert call["syntax"] == "lucene"
        assert call["time_filter"] == "day"  # lookback_hours == 24

    @pytest.mark.parametrize(
        "hours,expected", [(1, "day"), (24, "day"), (48, "week"), (168, "week"), (400, "month")]
    )
    def test_time_filter_mapping(self, hours, expected):
        assert S.PrawProvider._time_filter(hours) == expected

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
