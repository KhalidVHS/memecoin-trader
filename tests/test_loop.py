"""The wiring, not the arithmetic.

Every module below ``loop.py`` is tested on its own terms elsewhere. What is
only testable here is what happens when they are connected: who writes to the
ledger, who is allowed to fail, and whether a tick that could not reach the
model is distinguishable from a tick that decided to do nothing.

The first test in this file exists because of a real bug. ``_apply`` and
``_force_exit`` both called ``journal.append(trades_path, fill)`` on a fill that
``LocalPaperBroker.place_order`` had *already* logged, so every single trade
landed in ``trades.jsonl`` twice while ``state.json`` moved once. Nothing caught
it — the book was correct, the reconciliation identity held, and only counting
the rows revealed it. So the rows get counted.

No network: ``market.snapshot`` and ``quotes.fill_quote`` are monkeypatched, and
every test gets its own ``tmp_path`` data dir.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import pytest

from memetrader import config, loop, quotes
from memetrader.loop import Trader
from memetrader.types import (
    CoinSnapshot,
    FillQuote,
    MarketSnapshot,
    PriceLadder,
    Side,
    TxnCounts,
)

MINT = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
NOW = 1_700_000_000.0
PRICE = 0.00002


def make_cfg(tmp_path: Path) -> config.Config:
    """The real config with the data dir redirected and randomness removed.

    ``failed_tx_rate`` is pinned to 0 so a 6% coin flip cannot make these tests
    flaky — the failed-transaction path has its own deterministic tests in
    ``test_broker.py``.
    """
    base = config.load()
    return replace(
        base,
        data_dir=tmp_path,
        execution=replace(base.execution, failed_tx_rate=0.0, gas_usd_per_swap=0.21),
        sentiment=replace(base.sentiment, enabled=False),
    )


def coin(symbol: str = "BONK", *, price_usd: float = PRICE) -> CoinSnapshot:
    return CoinSnapshot(
        symbol=symbol,
        mint=MINT,
        price_usd=price_usd,
        liquidity_usd=250_000.0,
        volume_24h_usd=1_000_000.0,
        volume_1h_usd=50_000.0,
        fdv_usd=None,
        price_change=PriceLadder(m5=0.1, h1=0.2, h6=0.3, h24=0.4),
        txns_m5=TxnCounts(buys=10, sells=8),
        txns_h1=TxnCounts(buys=100, sells=90),
        txns_h24=TxnCounts(buys=1000, sells=900),
        pair_address="pool",
        dex_id="raydium",
        pair_created_at=NOW - 86_400.0,
        candles_5m=(),
        candles_1h=(),
    )


def snapshot(*coins: CoinSnapshot, ts: float | None = None) -> MarketSnapshot:
    """Fresh by default. ``fast_tick`` reads the real clock to age the snapshot
    against ``max_snapshot_age_seconds``, so a snapshot pinned to a fixed epoch
    is rejected as stale before any stop-loss logic is reached."""
    coins = coins or (coin(),)
    return MarketSnapshot(ts=time.time() if ts is None else ts, coins={c.symbol: c for c in coins})


def quote(
    *, side: Side = Side.SELL, usd_notional: float = 200.0, price_usd: float = PRICE
) -> FillQuote:
    return FillQuote(
        symbol="BONK",
        mint=MINT,
        side=side,
        usd_notional=usd_notional,
        price_usd=price_usd,
        price_impact_pct=0.4,
        route_labels=("Raydium",),
        pool_fee_pct=0.0,
        degraded=False,
    )


@pytest.fixture
def stub_network(monkeypatch: pytest.MonkeyPatch):
    """Pin both network boundaries. Returns the snapshot so tests can mutate it."""
    snap = snapshot()

    def fake_fill_quote(cfg, symbol, mint, side, usd_notional, *, mid_price_usd, **kw):
        # Fill at the true market price, which is what Jupiter returns — note
        # this is deliberately *not* ``mid_price_usd``. The real ``fill_quote``
        # uses that argument only as a hint for deriving token decimals and gets
        # its price from the route, which is why ``_force_exit`` can safely pass
        # a stale entry price when the snapshot is missing. A stub that filled at
        # the hint would make that fallback look broken when it is not.
        market_price = snap.coins[symbol].price_usd if symbol in snap.coins else mid_price_usd
        return quote(side=side, usd_notional=usd_notional, price_usd=market_price)

    monkeypatch.setattr(loop.market, "snapshot", lambda cfg, **kw: snap)
    monkeypatch.setattr(loop.quotes, "fill_quote", fake_fill_quote)
    return snap


def trades(cfg: config.Config) -> list[dict]:
    if not cfg.trades_path.is_file():
        return []
    return [
        json.loads(line)
        for line in cfg.trades_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# The ledger is written exactly once
# ---------------------------------------------------------------------------


def test_a_stop_loss_exit_writes_exactly_one_trade_row(
    tmp_path: Path, stub_network: MarketSnapshot
) -> None:
    """The regression. ``place_order`` owns the trades.jsonl row; the loop must
    not write a second one. Asserting ``== 1`` rather than ``>= 1`` is the whole
    point of the test."""
    cfg = make_cfg(tmp_path)
    trader = Trader(cfg, client=object())  # type: ignore[arg-type]
    trader.broker.place_order("BONK", Side.BUY, 200.0, quote=quote(side=Side.BUY), now=NOW)
    assert len(trades(cfg)) == 1, "the opening buy itself should be one row"

    # Halve the price: a -50% drawdown is well past the -15% stop.
    stub_network.coins["BONK"] = coin(price_usd=PRICE / 2)
    result = trader.fast_tick()

    assert result.stop_loss_exits == ["BONK"]
    rows = trades(cfg)
    assert len(rows) == 2, f"expected buy + stop-loss sell, got {len(rows)} rows"
    assert [r["side"] for r in rows] == ["BUY", "SELL"]


def test_the_book_and_the_trade_log_agree_after_a_stop_loss(
    tmp_path: Path, stub_network: MarketSnapshot
) -> None:
    """The check that the duplicate-row bug slipped past, kept as the companion
    to the one that catches it: the book was always right, which is exactly why
    nothing noticed the log was wrong."""
    cfg = make_cfg(tmp_path)
    trader = Trader(cfg, client=object())  # type: ignore[arg-type]
    trader.broker.place_order("BONK", Side.BUY, 200.0, quote=quote(side=Side.BUY), now=NOW)
    stub_network.coins["BONK"] = coin(price_usd=PRICE / 2)
    trader.fast_tick()

    assert trader.broker.get_positions() == {}
    reloaded = json.loads(cfg.state_path.read_text(encoding="utf-8"))
    assert reloaded["positions"] == {}
    # Sum of the log equals the move in the book: paid 200 + gas, got ~100 back.
    rows = trades(cfg)
    net = sum(
        (r["filled_usd"] if r["side"] == "SELL" else -r["filled_usd"]) - r["gas_usd"]
        for r in rows
    )
    assert reloaded["cash_usd"] == pytest.approx(1000.0 + net, abs=0.01)


# ---------------------------------------------------------------------------
# Stop-losses fire through a collapsing pool
# ---------------------------------------------------------------------------


def test_a_stop_loss_still_exits_a_pool_below_the_liquidity_floor(
    tmp_path: Path, stub_network: MarketSnapshot
) -> None:
    """The reason the forced-exit bypass was widened. A pool draining below the
    floor is the scenario the stop exists for; blocking the exit there traps the
    position instead of protecting it."""
    cfg = make_cfg(tmp_path)
    trader = Trader(cfg, client=object())  # type: ignore[arg-type]
    trader.broker.place_order("BONK", Side.BUY, 200.0, quote=quote(side=Side.BUY), now=NOW)

    collapsed = replace(coin(price_usd=PRICE / 2), liquidity_usd=900.0)
    stub_network.coins["BONK"] = collapsed
    result = trader.fast_tick()

    assert result.stop_loss_exits == ["BONK"]
    assert trader.broker.get_positions() == {}


def test_a_stop_loss_is_not_blocked_by_a_missing_snapshot(
    tmp_path: Path, stub_network: MarketSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mint is a config constant and the entry price is a good enough hint
    for decimals, so losing the market read must not cost the position."""
    cfg = make_cfg(tmp_path)
    trader = Trader(cfg, client=object())  # type: ignore[arg-type]
    trader.broker.place_order("BONK", Side.BUY, 200.0, quote=quote(side=Side.BUY), now=NOW)

    # The market really is at half price; it is the *read* of it that failed.
    from memetrader import portfolio

    stub_network.coins["BONK"] = coin(price_usd=PRICE / 2)
    now = time.time()
    book = portfolio.mark(trader.broker, {"BONK": PRICE / 2}, now=now)
    empty = MarketSnapshot(ts=now, coins={})
    fill = trader._force_exit("BONK", empty, book, now)

    assert fill is not None, "a data outage must not trap a position past its stop"
    assert fill.side is Side.SELL
    assert trader.broker.get_positions() == {}


def test_a_stop_loss_is_blocked_by_a_stale_snapshot(
    tmp_path: Path, stub_network: MarketSnapshot
) -> None:
    """Deliberately still enforced. The fast tick retries in 60s, so refusing
    here costs a minute; exiting on a price we cannot vouch for costs the fill."""
    from memetrader import portfolio

    cfg = make_cfg(tmp_path)
    trader = Trader(cfg, client=object())  # type: ignore[arg-type]
    trader.broker.place_order("BONK", Side.BUY, 200.0, quote=quote(side=Side.BUY), now=NOW)

    now = time.time()
    book = portfolio.mark(trader.broker, {"BONK": PRICE / 2}, now=now)
    stale = snapshot(coin(price_usd=PRICE / 2), ts=now - 600.0)
    fill = trader._force_exit("BONK", stale, book, now)

    assert fill is None
    assert "BONK" in trader.broker.get_positions()
    assert len(trades(cfg)) == 1, "the opening buy only — no sell row"


# ---------------------------------------------------------------------------
# A failed tick is not a HOLD
# ---------------------------------------------------------------------------


def test_a_model_failure_leaves_the_book_untouched_and_says_so(
    tmp_path: Path, stub_network: MarketSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tick that never reached the model must be distinguishable from a tick
    that chose to do nothing — otherwise an outage reads as conviction."""
    cfg = make_cfg(tmp_path)
    monkeypatch.setattr(
        loop.brain, "decide", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("529 overloaded"))
    )
    trader = Trader(cfg, client=object())  # type: ignore[arg-type]
    result = trader.slow_tick()

    assert result.decision is None
    assert result.error is not None and "529" in result.error
    assert result.fills == []
    assert trades(cfg) == []
    assert not cfg.decisions_path.is_file(), "a failed tick must not log a decision"


# ---------------------------------------------------------------------------
# The decision log round-trips
# ---------------------------------------------------------------------------


def test_a_logged_decision_rehydrates_into_a_renderable_record(
    tmp_path: Path, stub_network: MarketSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write a decision, read it back, render it into the next prompt.

    This is the test that was missing. Every earlier check ran against an empty
    ``decisions.jsonl``, so the replay path was only ever exercised on the
    *second* slow tick of a run — where ``_record_from_row`` handed the prompt
    renderer raw dicts and it died on ``fill.symbol``. Two ticks, not one.
    """
    from memetrader import portfolio, prompts
    from memetrader.types import Action, TradeDecision

    cfg = make_cfg(tmp_path)
    decision = TradeDecision(
        market_read="fabricated",
        actions=[
            Action(action="BUY", symbol="BONK", size_usd=200.0,
                   confidence=0.8, reasoning="because"),
        ],
    )
    monkeypatch.setattr(loop.brain, "decide", lambda *a, **kw: (decision, brain_usage()))

    trader = Trader(cfg, client=object())  # type: ignore[arg-type]
    trader.slow_tick()
    assert cfg.decisions_path.is_file()

    # Tick two: the record now comes back off disk rather than out of memory.
    history = trader.recent_decisions()
    assert len(history) == 1, "the decision just written must be readable"
    record = history[0]
    assert record.actions[0].symbol == "BONK"
    assert record.fills and record.fills[0].symbol == "BONK"
    assert record.fills[0].side is Side.BUY, "side must rehydrate as the enum"

    # The part that actually broke: rendering it into the next user turn.
    book = portfolio.mark(trader.broker, {"BONK": PRICE}, now=time.time())
    text = prompts.render_user(cfg, trader.evidence(trader.snapshot()), book, history, [])
    assert "BUY BONK" in text


def test_the_models_thinking_reaches_the_decision_log_and_reads_back(
    tmp_path: Path, stub_network: MarketSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thinking tokens are the majority of what this project spends, and the
    field sat unpopulated — so the reasoning behind every trade was paid for and
    then discarded, leaving the log unable to answer why a loss looked like a good
    idea at the time."""
    from memetrader.types import Action, TradeDecision

    cfg = make_cfg(tmp_path)
    thinking = "liquidity -12% over the interval; exiting rather than sizing down"
    decision = TradeDecision(
        market_read="fabricated",
        actions=[
            Action(action="HOLD", symbol="BONK", size_usd=0.0,
                   confidence=0.2, reasoning="waiting"),
        ],
    )
    monkeypatch.setattr(
        loop.brain, "decide", lambda *a, **kw: (decision, brain_usage(thinking=thinking))
    )

    trader = Trader(cfg, client=object())  # type: ignore[arg-type]
    trader.slow_tick()

    assert [r["thinking"] for r in decisions(cfg)] == [thinking]
    assert trader.recent_decisions()[0].thinking == thinking

    # A call that returned no thinking block at all — the API can redact it —
    # records the absence rather than an empty string.
    monkeypatch.setattr(loop.brain, "decide", lambda *a, **kw: (decision, brain_usage()))
    trader.slow_tick()
    assert decisions(cfg)[1]["thinking"] is None
    assert trader.recent_decisions()[1].thinking is None

    # Rows written before the field was populated have no key at all, and a
    # missing thinking block is not an empty one.
    older = {k: v for k, v in decisions(cfg)[0].items() if k != "thinking"}
    assert loop._record_from_row(older).thinking is None


# ---------------------------------------------------------------------------
# The risk feedback channel survives a restart
# ---------------------------------------------------------------------------


def test_last_ticks_rejections_survive_a_restart(
    tmp_path: Path, stub_network: MarketSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``pending_rejections`` was in-memory only, so a restart dropped the risk
    feedback on the one tick most likely to repeat itself: nothing in the model's
    context says it already proposed this and was refused. Every verdict is
    already in decisions.jsonl, so a fresh Trader over the same data dir has to
    come up knowing them."""
    from memetrader import portfolio, prompts
    from memetrader.types import Action, TradeDecision

    cfg = make_cfg(tmp_path)
    # $4 is under the $10 floor, which risk.py rejects outright. A size over the
    # position cap would be *clamped* instead, and a clamp is still an approval.
    decision = TradeDecision(
        market_read="fabricated",
        actions=[
            Action(action="BUY", symbol="BONK", size_usd=4.0,
                   confidence=0.7, reasoning="because"),
        ],
    )
    monkeypatch.setattr(loop.brain, "decide", lambda *a, **kw: (decision, brain_usage()))

    trader = Trader(cfg, client=object())  # type: ignore[arg-type]
    assert trader.pending_rejections == [], "an empty log has nothing to replay"
    trader.slow_tick()
    assert [v.rule for v in trader.pending_rejections] == ["min_trade_size"]

    # The restart: a second Trader over the same data dir, sharing no memory.
    restarted = Trader(cfg, client=object())  # type: ignore[arg-type]
    assert restarted.pending_rejections == trader.pending_rejections
    assert all(not v.approved for v in restarted.pending_rejections), (
        "approvals are not feedback — only what was refused goes back to the model"
    )

    # And it has to reach the prompt, not just the attribute.
    book = portfolio.mark(restarted.broker, {"BONK": PRICE}, now=time.time())
    text = prompts.render_user(
        cfg,
        restarted.evidence(restarted.snapshot()),
        book,
        restarted.recent_decisions(),
        restarted.pending_rejections,
    )
    section = text.split("=== RISK VERDICTS FROM LAST TICK ===", 1)[1]
    assert "min_trade_size" in section
    assert "Do not re-propose them" in section


# ---------------------------------------------------------------------------
# The liquidity window the model is shown
# ---------------------------------------------------------------------------


def test_the_slow_ticks_liquidity_window_is_a_decision_interval(
    tmp_path: Path, stub_network: MarketSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression for the 60-second liquidity trend.

    Both ticks used to end by assigning ``self.previous``, and the slow tick built
    its evidence against that — so the two snapshots actually being compared were
    one *fast* tick apart. A pool shedding 10% of its depth across a decision
    interval reached the model as -0.7%, which reads as noise, while the prompt
    labelled it "trend vs last tick" and the system prompt calls a draining pool
    the single most important thing that can happen to a position.

    Only the real tick sequence catches that, so this runs one: decide, fourteen
    fast ticks, decide.
    """
    cfg = make_cfg(tmp_path)
    fast = cfg.cadence.fast_tick_seconds
    slow = cfg.cadence.slow_tick_seconds
    start = time.time() - slow  # so the last read lands at roughly "now"
    elapsed = 0.0

    def read(_cfg, **_kw) -> MarketSnapshot:
        return snapshot(
            replace(coin(), liquidity_usd=draining(slow, elapsed)),
            ts=start + elapsed,
        )

    seen: list[dict] = []

    def capture(cfg_, evidence, *_a, **_kw):
        seen.append(evidence)
        return hold_everything(cfg_), brain_usage()

    monkeypatch.setattr(loop.market, "snapshot", read)
    monkeypatch.setattr(loop.brain, "decide", capture)

    trader = Trader(cfg, client=object())  # type: ignore[arg-type]
    trader.slow_tick()
    for step in range(1, slow // fast):  # fourteen fast ticks, a minute apart
        elapsed = float(step * fast)
        trader.fast_tick()

    # The two references have to diverge here — that divergence is the fix. The
    # fast ticks carried the freshest read forward and left the baseline alone.
    assert trader.previous is not None
    assert trader.previous.ts == pytest.approx(start + 14 * fast)
    assert trader.decision_baseline is not None
    assert trader.decision_baseline.ts == pytest.approx(start)

    elapsed = float(slow)
    trader.slow_tick()

    assert len(seen) == 2, "both decisions must have reached the stubbed model"
    first = seen[0]["BONK"].technicals.flow
    assert first.liquidity_trend_pct is None, "the first decision has no baseline"
    assert first.liquidity_trend_seconds is None, "and so no window either"

    second = seen[1]["BONK"].technicals.flow
    # The fast tick immediately before this one read
    # 100_000 * (1 - 0.10 * 840/900) = $90,666.67, against which $90,000 is
    # (90_000 - 90_666.67) / 90_666.67 = -0.735%: the noise-shaped number the
    # defect handed the model instead of the drain it was looking at.
    assert second.liquidity_trend_pct == pytest.approx(-10.0, abs=1e-9), (
        "a fast-tick-wide window would have reported -0.735% here"
    )
    assert second.liquidity_trend_seconds == pytest.approx(float(slow), abs=1e-9)


def test_a_failed_model_call_does_not_move_the_liquidity_baseline(
    tmp_path: Path, stub_network: MarketSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tick that never reached the model is not a decision, so it must not
    become the thing the next decision is measured against. The next successful
    tick then reports the drain across the whole outage — and says the window was
    that long, instead of claiming one cadence over evidence nobody read."""
    cfg = make_cfg(tmp_path)
    slow = cfg.cadence.slow_tick_seconds
    start = time.time() - 2 * slow
    elapsed = 0.0

    def read(_cfg, **_kw) -> MarketSnapshot:
        return snapshot(
            replace(coin(), liquidity_usd=draining(slow, elapsed)),
            ts=start + elapsed,
        )

    seen: list[dict] = []

    def capture(cfg_, evidence, *_a, **_kw):
        seen.append(evidence)
        return hold_everything(cfg_), brain_usage()

    monkeypatch.setattr(loop.market, "snapshot", read)
    monkeypatch.setattr(loop.brain, "decide", capture)

    trader = Trader(cfg, client=object())  # type: ignore[arg-type]
    trader.slow_tick()  # the baseline read

    elapsed = float(slow)
    monkeypatch.setattr(
        loop.brain, "decide",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("529 overloaded")),
    )
    assert trader.slow_tick().error is not None

    elapsed = 2.0 * slow
    monkeypatch.setattr(loop.brain, "decide", capture)
    trader.slow_tick()

    flow = seen[-1]["BONK"].technicals.flow
    assert flow.liquidity_trend_pct == pytest.approx(-20.0, abs=1e-9)
    assert flow.liquidity_trend_seconds == pytest.approx(2.0 * slow, abs=1e-9)


def draining(slow: float, elapsed: float) -> float:
    """Pool depth on a straight line from $100k to $90k across one decision
    interval — a 10% drain, which no single 60-second step can see."""
    return 100_000.0 * (1.0 - 0.10 * elapsed / slow)


def decisions(cfg: config.Config) -> list[dict]:
    if not cfg.decisions_path.is_file():
        return []
    return [
        json.loads(line)
        for line in cfg.decisions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def hold_everything(cfg: config.Config):
    """One HOLD per configured coin — the only decision that leaves the book
    exactly as it was, which is what the cadence tests want to measure against."""
    from memetrader.types import Action, TradeDecision

    return TradeDecision(
        market_read="holding",
        actions=[
            Action(action="HOLD", symbol=s, size_usd=0.0, confidence=0.3, reasoning="x")
            for s in cfg.symbols
        ],
    )


def brain_usage(*, thinking: str | None = None):
    from memetrader.brain import Usage

    return Usage(
        input_tokens=10, output_tokens=5,
        cache_read_input_tokens=0, cache_creation_input_tokens=0,
        thinking=thinking,
    )
