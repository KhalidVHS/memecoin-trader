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


def brain_usage():
    from memetrader.brain import Usage

    return Usage(
        input_tokens=10, output_tokens=5,
        cache_read_input_tokens=0, cache_creation_input_tokens=0,
    )
