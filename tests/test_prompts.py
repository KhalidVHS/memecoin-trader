"""Tests for the prompt layer and the brain.

Fixtures are hand-built here rather than loaded from ``tests/fixtures/*.json`` on
purpose: this file must be able to construct evidence that no live API would
produce — a wholly missing sentiment brief, a technicals block where every
indicator is ``None`` — and it must not break when another module's fixtures
change shape.

The expensive bug this file exists to catch is the cache one. Everything else
here is cheap correctness; ``test_live_call_hits_the_prompt_cache`` is the test
that decides whether this design costs $3/day or $30/day.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from memetrader import config as config_mod
from memetrader.brain import BrainError, Usage, _normalize, decide
from memetrader.prompts import build_system, render_user
from memetrader.types import (
    Action,
    Candle,
    CoinSnapshot,
    DecisionRecord,
    EvidenceBundle,
    Fill,
    FlowBrief,
    PortfolioState,
    Position,
    PriceLadder,
    RiskVerdict,
    SentimentBrief,
    Side,
    TechnicalBrief,
    Technicals,
    Timeframe,
    TopPost,
    TxnCounts,
)

NOW = 1_764_000_000.0  # a fixed epoch-seconds "now" so nothing in here drifts

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cfg():
    """The real config.toml — it is what the live test must exercise."""
    return config_mod.load(REPO_ROOT / "config.toml")


def _technicals(timeframe: Timeframe, *, blank: bool = False) -> Technicals:
    if blank:
        # Every indicator missing: not enough candle history. This is the case
        # that must render "n/a" and never "0.0".
        return Technicals(
            timeframe=timeframe,
            candles_used=7,
            rsi14=None,
            rsi14_rising=None,
            ema9=None,
            ema21=None,
            ema9_above_ema21=None,
            pct_from_ema9=None,
            pct_from_ema21=None,
            macd_line=None,
            macd_signal=None,
            macd_hist=None,
            macd_cross=None,
            bars_since_cross=None,
            bb_percent_b=None,
            bb_bandwidth=None,
            bb_expanding=None,
            atr14_pct=None,
            volume_ratio_20=None,
            pct_from_swing_high=None,
            pct_from_swing_low=None,
        )
    return Technicals(
        timeframe=timeframe,
        candles_used=100,
        rsi14=35.2,
        rsi14_rising=True,
        ema9=0.00002181,
        ema21=0.00002240,
        ema9_above_ema21=False,
        pct_from_ema9=-1.4,
        pct_from_ema21=-3.9,
        macd_line=-1.2e-07,
        macd_signal=-1.6e-07,
        macd_hist=4.0e-08,
        macd_cross="bullish",
        bars_since_cross=2,
        bb_percent_b=0.18,
        bb_bandwidth=6.4,
        bb_expanding=True,
        atr14_pct=2.9,
        volume_ratio_20=1.8,
        pct_from_swing_high=-11.2,
        pct_from_swing_low=3.4,
    )


def _snapshot(
    symbol: str,
    price: float,
    *,
    liquidity: float = 1_200_000.0,
    ladder: PriceLadder | None = None,
) -> CoinSnapshot:
    return CoinSnapshot(
        symbol=symbol,
        mint=f"mint-for-{symbol}",
        price_usd=price,
        liquidity_usd=liquidity,
        volume_24h_usd=9_100_000.0,
        volume_1h_usd=412_000.0,
        fdv_usd=1_420_000_000.0,
        price_change=ladder or PriceLadder(m5=-0.41, h1=2.3, h6=-1.1, h24=8.4),
        txns_m5=TxnCounts(buys=120, sells=98),
        txns_h1=TxnCounts(buys=1440, sells=1190),
        txns_h24=TxnCounts(buys=30_100, sells=28_600),
        pair_address=f"pair-{symbol}",
        dex_id="raydium",
        pair_created_at=NOW - 400 * 86400,
        candles_5m=(Candle(NOW - 300, price, price, price, price, 1.0),),
        candles_1h=(),
    )


def _flow() -> FlowBrief:
    return FlowBrief(
        buy_sell_ratio_m5=1.22,
        buy_sell_ratio_h1=1.21,
        buy_sell_ratio_h24=1.05,
        turnover_24h=7.55,
        turnover_1h=0.34,
        liquidity_usd=1_200_000.0,
        liquidity_trend_pct=-6.2,
        # A decision interval that ran 60 seconds late, so the rendered window
        # has to come from this number rather than from the configured cadence.
        liquidity_trend_seconds=840.0,
        price_ladder=PriceLadder(m5=-0.41, h1=2.3, h6=-1.1, h24=8.4),
    )


def _sentiment(symbol: str) -> SentimentBrief:
    return SentimentBrief(
        symbol=symbol,
        ts=NOW - 120,
        source="praw",
        mention_velocity_1h=14.0,
        mention_velocity_24h=4.2,
        mention_zscore_7d=2.4,
        unique_contributors_24h=61,
        contributor_to_post_ratio=0.31,
        top_posts=(
            TopPost(title=f"{symbol} is going parabolic", score=412, age_hours=3.1, subreddit="solana"),
        ),
        polarity=0.62,
    )


@pytest.fixture
def evidence(cfg) -> dict[str, EvidenceBundle]:
    """Three coins exercising the three interesting shapes:

    * full evidence
    * sentiment missing entirely
    * technicals present but every indicator ``None``
    """
    symbols = list(cfg.symbols)
    prices = [0.00002134, 2.41, 0.813]
    bundles: dict[str, EvidenceBundle] = {}
    for i, symbol in enumerate(symbols):
        # The third coin is the "degraded everything" case: no candle history,
        # so every indicator is None, and DexScreener reported no m5/h6 window
        # for its pool either.
        ladder = (
            PriceLadder(m5=None, h1=2.3, h6=None, h24=8.4) if i == 2 else None
        )
        snap = _snapshot(symbol, prices[i % len(prices)], ladder=ladder)
        blank = i == 2
        tech = TechnicalBrief(
            symbol=symbol,
            m5=_technicals(Timeframe.M5, blank=blank),
            h1=_technicals(Timeframe.H1, blank=blank),
            flow=_flow(),
        )
        if i == 1:
            bundles[symbol] = EvidenceBundle(
                symbol=symbol,
                snapshot=snap,
                technicals=tech,
                sentiment=None,
                sentiment_unavailable_reason="reddit returned 503",
            )
        else:
            bundles[symbol] = EvidenceBundle(
                symbol=symbol,
                snapshot=snap,
                technicals=tech,
                sentiment=_sentiment(symbol),
            )
    return bundles


@pytest.fixture
def other_evidence(cfg) -> dict[str, EvidenceBundle]:
    """Wholly different numbers, for the prefix-stability test."""
    bundles: dict[str, EvidenceBundle] = {}
    for i, symbol in enumerate(cfg.symbols):
        snap = _snapshot(
            symbol,
            99.9 + i,
            liquidity=41_000.0,
            ladder=PriceLadder(m5=None, h1=None, h6=None, h24=None),
        )
        bundles[symbol] = EvidenceBundle(
            symbol=symbol,
            snapshot=snap,
            technicals=None,
            sentiment=None,
            sentiment_unavailable_reason="sentiment disabled",
        )
    return bundles


@pytest.fixture
def portfolio(cfg) -> PortfolioState:
    symbol = cfg.symbols[0]
    pos = Position(
        symbol=symbol,
        quantity=8_411_200.0,
        avg_entry_price_usd=0.00002210,
        opened_at=NOW - 8_700,
        cost_basis_usd=185.89,
    )
    mark = 0.00002134
    value = pos.quantity * mark
    return PortfolioState(
        ts=NOW,
        cash_usd=812.44,
        positions={symbol: pos},
        marks={symbol: mark},
        position_values_usd={symbol: value},
        unrealized_pnl_usd=pos.unrealized_pnl_usd(mark),
        realized_pnl_usd=-12.10,
        total_value_usd=812.44 + value,
        starting_cash_usd=cfg.starting_cash_usd,
        fees_paid_usd=2.10,
        gas_paid_usd=0.63,
    )


@pytest.fixture
def empty_portfolio(cfg) -> PortfolioState:
    return PortfolioState(
        ts=NOW,
        cash_usd=cfg.starting_cash_usd,
        positions={},
        marks={},
        position_values_usd={},
        unrealized_pnl_usd=0.0,
        realized_pnl_usd=0.0,
        total_value_usd=cfg.starting_cash_usd,
        starting_cash_usd=cfg.starting_cash_usd,
    )


def _record(cfg, i: int) -> DecisionRecord:
    symbol = cfg.symbols[i % len(cfg.symbols)]
    return DecisionRecord(
        ts=NOW - (i + 1) * 900,
        market_read=f"read number {i}",
        actions=(
            Action(
                action="BUY",
                symbol=symbol,
                size_usd=120.0 + i,
                confidence=0.6,
                reasoning="flow: m5 buy/sell 1.9",
            ),
        ),
        verdicts=(RiskVerdict(approved=True, approved_usd=120.0 + i),),
        fills=(
            Fill(
                ts=NOW - (i + 1) * 900,
                symbol=symbol,
                side=Side.BUY,
                requested_usd=120.0 + i,
                filled_usd=119.4 + i,
                price_usd=0.00002210,
                quantity=5_400_000.0,
                price_impact_pct=0.4,
                pool_fee_usd=0.3,
                gas_usd=0.21,
            ),
        ),
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=2000,
        cache_creation_input_tokens=0,
        model=cfg.model.name,
        effort=cfg.model.effort,
    )


@pytest.fixture
def history(cfg) -> list[DecisionRecord]:
    # Deliberately more records than the configured cap.
    return [_record(cfg, i) for i in range(cfg.prompt.decision_history + 7)][::-1]


@pytest.fixture
def rejections() -> list[RiskVerdict]:
    return [
        RiskVerdict(
            approved=False,
            approved_usd=0.0,
            rule="max_position_pct",
            reason="proposed $480 on a $1,004 book; the cap is 30% ($301)",
            notes=("clamp would have left it above the cap anyway",),
        )
    ]


# ---------------------------------------------------------------------------
# render_user
# ---------------------------------------------------------------------------


def test_every_coin_appears(cfg, evidence, portfolio, history, rejections):
    text = render_user(cfg, evidence, portfolio, history, rejections)
    for symbol in cfg.symbols:
        assert f"--- {symbol} ---" in text, f"{symbol} missing from the brief"


def test_missing_sentiment_is_explicitly_unavailable(
    cfg, evidence, portfolio, history, rejections
):
    text = render_user(cfg, evidence, portfolio, history, rejections)
    missing = cfg.symbols[1]
    section = text.split(f"--- {missing} ---", 1)[1].split("---", 1)[0]
    assert "SENTIMENT: UNAVAILABLE" in section
    assert "this stream is missing this tick" in section
    assert "reddit returned 503" in section
    # ...and the phrase that distinguishes "we don't know" from "nobody cares".
    assert "could not find out" in section


def test_missing_technicals_is_explicitly_unavailable(
    cfg, other_evidence, empty_portfolio, rejections
):
    text = render_user(cfg, other_evidence, empty_portfolio, [], rejections)
    assert text.count("TECHNICALS: UNAVAILABLE") == len(cfg.symbols)


def test_none_indicators_render_as_na_not_zero(
    cfg, evidence, portfolio, history, rejections
):
    """The load-bearing one: a missing indicator must never look like a zero."""
    text = render_user(cfg, evidence, portfolio, history, rejections)
    blank_symbol = cfg.symbols[2]
    start = text.index(f"--- {blank_symbol} ---")
    section = text[start : text.index("=== PORTFOLIO ===")]
    tech = section[section.index("TECHNICALS") :]

    assert "n/a" in tech
    for field in ("rsi14", "macd", "atr14", "bollinger", "volume", "swing"):
        line = next(ln for ln in tech.splitlines() if field in ln)
        assert "n/a" in line, f"{field!r} line has no n/a marker: {line}"
        assert "0.0" not in line, f"{field!r} line renders a None as a zero: {line}"

    # And nowhere in the whole blank-technicals block does a zero appear.
    assert "0.0" not in tech
    assert "0.00" not in tech

    # Same discipline for the price ladder: DexScreener omits a window rather
    # than reporting zero, so a missing m5 must not read as "flat over 5m".
    change = next(ln for ln in section.splitlines() if ln.strip().startswith("change"))
    assert "m5 n/a" in change
    assert "h6 n/a" in change
    assert "h1 +2.30%" in change  # the windows that *were* reported still render
    assert "0.0" not in change


def test_all_none_price_ladder_renders_entirely_na(
    cfg, other_evidence, empty_portfolio
):
    text = render_user(cfg, other_evidence, empty_portfolio, [], [])
    changes = [ln for ln in text.splitlines() if ln.strip().startswith("change")]
    assert len(changes) == len(cfg.symbols)
    for line in changes:
        assert line.strip() == "change     m5 n/a  h1 n/a  h6 n/a  h24 n/a", line
        assert "0.0" not in line


def test_the_liquidity_trend_names_the_window_it_covers(
    cfg, evidence, portfolio, history, rejections
):
    """A liquidity percentage with no window attached is unreadable, and this line
    used to attach the wrong one: it said "trend vs last tick", which reads as the
    15-minute decision cadence the model is told it wakes up on, while the number
    behind it was the delta since a fast tick 60 seconds earlier."""
    text = render_user(cfg, evidence, portfolio, history, rejections)
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("liquidity"))

    # The fixture's gap is 840s, which _age formats as 14m — so what is printed
    # is the interval that was measured and not the configured cadence restated.
    assert "trend over 14m -6.20%" in line
    assert "vs last tick" not in text
    assert "a draining pool outranks everything else here" in line


def test_a_missing_liquidity_trend_renders_na_and_never_a_zero(
    cfg, evidence, portfolio
):
    """The first decision of a run has nothing to compare against, so there is
    neither a trend nor a window. A fabricated 0.00% would claim a perfectly
    stable pool — the opposite of an absence of information — against a system
    prompt that ranks a draining pool above every other signal it is given."""
    symbol = cfg.symbols[0]
    bundle = evidence[symbol]
    flow = replace(
        bundle.technicals.flow,
        liquidity_trend_pct=None,
        liquidity_trend_seconds=None,
    )
    first_tick = {
        symbol: replace(bundle, technicals=replace(bundle.technicals, flow=flow))
    }

    text = render_user(cfg, first_tick, portfolio, [], [])
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("liquidity"))

    assert "trend n/a" in line
    assert "NOT a stable pool" in line, "n/a here must not read as flat"
    assert "0.0" not in line and "0.00" not in line


def test_a_stretched_liquidity_window_is_rendered_as_the_real_gap(
    cfg, evidence, portfolio
):
    """After a run of failed model calls the two compared reads are hours apart.
    The window follows the measurement, so the line widens instead of restating
    the cadence: -6.2% over three hours is a different trade from -6.2% over
    fifteen minutes."""
    symbol = cfg.symbols[0]
    bundle = evidence[symbol]
    flow = replace(bundle.technicals.flow, liquidity_trend_seconds=10_800.0)
    stretched = {
        symbol: replace(bundle, technicals=replace(bundle.technicals, flow=flow))
    }

    text = render_user(cfg, stretched, portfolio, [], [])
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("liquidity"))
    assert "trend over 3.0h -6.20%" in line


def test_history_is_capped_at_config_limit(
    cfg, evidence, portfolio, history, rejections
):
    text = render_user(cfg, evidence, portfolio, history, rejections)
    assert len(history) > cfg.prompt.decision_history  # the fixture must over-supply
    header = f"=== YOUR LAST {cfg.prompt.decision_history} DECISIONS (newest first) ==="
    block = text.split(header, 1)[1].split("===", 1)[0]
    lines = [ln for ln in block.splitlines() if ln.strip()]
    assert len(lines) == cfg.prompt.decision_history
    # One line per decision, not one per action.
    assert all(ln.lstrip().startswith("t-") for ln in lines)


def test_history_lines_carry_the_outcome(
    cfg, evidence, portfolio, history, rejections
):
    text = render_user(cfg, evidence, portfolio, history, rejections)
    held = cfg.symbols[0]
    line = next(
        ln for ln in text.splitlines() if ln.lstrip().startswith("t-") and held in ln
    )
    assert "BUY" in line and "filled" in line
    assert "now " in line  # how it has worked out so far


def test_portfolio_block_has_pnl_and_stop_distance(
    cfg, evidence, portfolio, history, rejections
):
    text = render_user(cfg, evidence, portfolio, history, rejections)
    block = text.split("=== PORTFOLIO ===", 1)[1]
    assert "cash" in block and "812.44" in block
    assert "total return" in block
    assert "realized P&L" in block
    assert "unrealized" in block
    assert "forced stop at" in block
    assert "age" in block


def test_rejections_name_the_rule(cfg, evidence, portfolio, history, rejections):
    text = render_user(cfg, evidence, portfolio, history, rejections)
    assert "max_position_pct" in text
    assert "the cap is 30%" in text


def test_empty_history_and_rejections_are_stated_not_blank(
    cfg, evidence, empty_portfolio
):
    text = render_user(cfg, evidence, empty_portfolio, [], [])
    assert "none yet" in text
    assert "none — nothing was clamped or rejected" in text
    assert "positions: none" in text


def test_missing_coin_entirely_is_flagged(cfg, evidence, empty_portfolio):
    partial = {k: v for k, v in evidence.items() if k != cfg.symbols[0]}
    text = render_user(cfg, partial, empty_portfolio, [], [])
    assert f"--- {cfg.symbols[0]} ---" in text
    assert "ALL EVIDENCE: UNAVAILABLE" in text


# ---------------------------------------------------------------------------
# build_system — the cache prefix
# ---------------------------------------------------------------------------


def test_system_blocks_are_byte_identical_across_ticks(
    cfg, evidence, other_evidence, portfolio, empty_portfolio, history, rejections
):
    """The single most expensive bug in this design would be a drifting prefix.

    Build the system blocks twice, with completely different evidence and
    portfolios rendered in between, and require byte equality. If this ever
    fails, every tick pays full price for the whole stable prompt.
    """
    first = build_system(cfg)
    render_user(cfg, evidence, portfolio, history, rejections)
    second = build_system(cfg)
    render_user(cfg, other_evidence, empty_portfolio, [], [])
    third = build_system(cfg)

    assert first == second == third
    for a, b in zip(first, third, strict=True):
        assert a["text"].encode() == b["text"].encode()


def test_cache_breakpoint_is_on_the_last_stable_block(cfg):
    blocks = build_system(cfg)
    assert len(blocks) >= 1
    assert blocks[-1]["cache_control"] == {"type": "ephemeral"}
    # At most 4 breakpoints are allowed per request.
    n = sum(1 for b in blocks if "cache_control" in b)
    assert 1 <= n <= 4


def test_stable_text_contains_nothing_volatile(cfg, evidence):
    blob = "\n".join(b["text"] for b in build_system(cfg))

    # No timestamps of any flavour.
    import re

    assert not re.search(r"\d{4}-\d{2}-\d{2}", blob), "a date leaked into the prefix"
    assert not re.search(r"\b\d{2}:\d{2}:\d{2}\b", blob), "a clock leaked into the prefix"
    assert not re.search(r"\b1[67]\d{8}\b", blob), "an epoch timestamp leaked in"

    # No live prices or per-tick numbers.
    for bundle in evidence.values():
        assert f"{bundle.snapshot.price_usd}" not in blob
        assert f"{bundle.snapshot.liquidity_usd}" not in blob

    # It must still be long enough to clear Opus 5's 512-token minimum cacheable
    # prefix — below that the breakpoint is silently ignored.
    assert len(blob) > 4000, "stable prefix may be too short to cache at all"


def test_stable_text_states_the_non_negotiables(cfg):
    blob = "\n".join(b["text"] for b in build_system(cfg))
    lowered = blob.lower()
    # The hard limits, as facts about the world.
    assert "30%" in blob and "-15%" in blob
    assert "$10" in blob
    assert "3%" in blob
    assert "clamp" in lowered and "reject" in lowered
    # No frequency cap, no minimum hold.
    assert "no cap on trade" in lowered or "no limit on trade" in lowered
    assert "no minimum hold time" in lowered
    assert "hold is always a legitimate answer" in lowered
    # Stream trust ordering.
    assert "draining pool" in lowered
    assert "attention, not polarity" in lowered
    assert "contributor_to_post_ratio" in blob
    # The prior project's findings.
    assert "269" in blob and "2.26" in blob
    assert "16.62" in blob and "5.39" in blob
    assert "never dca a loser" in lowered
    # Output contract.
    assert "exactly 0.0 for hold" in lowered
    # The configured coins.
    for symbol in cfg.symbols:
        assert symbol in blob


# ---------------------------------------------------------------------------
# brain post-validation
# ---------------------------------------------------------------------------


def _decision(cfg, **overrides):
    from memetrader.types import TradeDecision

    actions = overrides.pop(
        "actions",
        [
            Action(action="HOLD", symbol=s, size_usd=0.0, confidence=0.3, reasoning="x")
            for s in cfg.symbols
        ],
    )
    return TradeDecision(market_read=overrides.pop("market_read", "flat"), actions=actions)


def test_normalize_fills_missing_coin_with_hold(cfg):
    partial = _decision(
        cfg,
        actions=[
            Action(action="HOLD", symbol=cfg.symbols[0], size_usd=0.0, confidence=0.2, reasoning="x")
        ],
    )
    out, notes = _normalize(cfg, partial)
    assert [a.symbol for a in out.actions] == list(cfg.symbols)
    assert all(a.action == "HOLD" for a in out.actions[1:])
    assert any("no action returned" in n for n in notes)
    assert "brain.py corrections" in out.market_read


def test_normalize_forces_hold_size_to_zero(cfg):
    bad = _decision(
        cfg,
        actions=[
            Action(action="HOLD", symbol=cfg.symbols[0], size_usd=42.0, confidence=0.9, reasoning="x")
        ],
    )
    out, notes = _normalize(cfg, bad)
    assert out.actions[0].size_usd == 0.0
    assert any("forced to 0.0" in n for n in notes)


def test_normalize_drops_unknown_symbols_and_duplicates(cfg):
    bad = _decision(
        cfg,
        actions=[
            Action(action="BUY", symbol="DOGE", size_usd=50.0, confidence=0.9, reasoning="x"),
            Action(action="BUY", symbol=cfg.symbols[0], size_usd=50.0, confidence=0.9, reasoning="a"),
            Action(action="SELL", symbol=cfg.symbols[0], size_usd=20.0, confidence=0.4, reasoning="b"),
        ],
    )
    out, notes = _normalize(cfg, bad)
    assert len(out.actions) == len(cfg.symbols)
    assert "DOGE" not in {a.symbol for a in out.actions}
    first = next(a for a in out.actions if a.symbol == cfg.symbols[0])
    assert first.action == "BUY" and first.reasoning == "a"
    assert any("unknown symbol" in n for n in notes)
    assert any("duplicate" in n for n in notes)


def test_normalize_clamps_negative_size(cfg):
    bad = _decision(
        cfg,
        actions=[
            Action(action="BUY", symbol=cfg.symbols[0], size_usd=-5.0, confidence=0.5, reasoning="x")
        ],
    )
    out, _ = _normalize(cfg, bad)
    assert out.actions[0].size_usd == 0.0


def test_normalize_lowercase_symbol_is_matched(cfg):
    bad = _decision(
        cfg,
        actions=[
            Action(
                action="BUY",
                symbol=cfg.symbols[0].lower(),
                size_usd=25.0,
                confidence=0.5,
                reasoning="x",
            )
        ],
    )
    out, _ = _normalize(cfg, bad)
    assert out.actions[0].symbol == cfg.symbols[0]
    assert out.actions[0].size_usd == 25.0


def test_api_failure_raises_and_never_returns_a_hold(
    cfg, evidence, portfolio, history, rejections
):
    """A failed tick and a decision to hold are different events."""
    import anthropic
    import httpx2

    class Boom:
        class messages:  # noqa: N801
            @staticmethod
            def parse(**_kwargs):
                request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
                response = httpx2.Response(429, request=request)
                raise anthropic.RateLimitError(
                    "rate limited", response=response, body=None
                )

    with pytest.raises(BrainError) as excinfo:
        decide(cfg, evidence, portfolio, history, rejections, client=Boom())
    assert excinfo.value.retryable is True


def test_usage_reports_the_four_fields():
    class U:
        input_tokens = 11
        output_tokens = 22
        cache_read_input_tokens = 33
        cache_creation_input_tokens = 44

    class R:
        usage = U()

    u = Usage.from_response(R())
    assert (u.input_tokens, u.output_tokens) == (11, 22)
    assert (u.cache_read_input_tokens, u.cache_creation_input_tokens) == (33, 44)
    assert u.total_input_tokens == 88


# ---------------------------------------------------------------------------
# The live call. Two requests, guarded, and it must prove the cache is working.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping the live model call",
)
def test_live_call_hits_the_prompt_cache(
    cfg, evidence, other_evidence, portfolio, empty_portfolio, history, rejections, capsys
):
    """Two real calls. Costs real money — keep it at two.

    Asserts the two things that cannot be verified offline:

    1. a validated ``TradeDecision`` comes back, with exactly one action per coin;
    2. ``cache_read_input_tokens > 0`` on the second call.

    If (2) is zero, the stable prefix is being invalidated between calls and
    every tick of the real run pays full price for the whole system prompt. That
    is the single most expensive bug in this design — do not xfail it, find it.
    """
    first, usage_1 = decide(cfg, evidence, portfolio, history, rejections)
    second, usage_2 = decide(cfg, other_evidence, empty_portfolio, [], [])

    for decision in (first, second):
        assert [a.symbol for a in decision.actions] == list(cfg.symbols)
        assert len(decision.actions) == len(cfg.symbols)
        for a in decision.actions:
            assert a.action in {"BUY", "SELL", "HOLD"}
            assert a.size_usd >= 0.0
            if a.action == "HOLD":
                assert a.size_usd == 0.0
            assert a.reasoning.strip(), "empty reasoning destroys the audit trail"
        assert decision.market_read.strip()

    with capsys.disabled():
        print("\n--- live call 1 ---")
        print(f"usage: {usage_1}")
        print(f"market_read: {first.market_read}")
        for a in first.actions:
            print(f"  {a.action:4} {a.symbol:8} ${a.size_usd:8.2f} "
                  f"conf {a.confidence:.2f}  {a.reasoning}")
        print("--- live call 2 ---")
        print(f"usage: {usage_2}")
        print(f"cache_read_input_tokens: {usage_2.cache_read_input_tokens}")
        print(f"cache hit rate: {usage_2.cache_hit_rate:.1%}")
        print(f"cost of the two calls: "
              f"${usage_1.cost_usd(cfg) + usage_2.cost_usd(cfg):.4f}")

    assert usage_2.cache_read_input_tokens > 0, (
        "the second live call read nothing from cache: the stable prefix is being "
        f"invalidated between ticks (usage_1={usage_1}, usage_2={usage_2})"
    )
