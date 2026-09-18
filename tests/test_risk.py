"""Every risk rule, on both sides of its boundary, plus the clamps.

``risk.check`` is a pure function, so these tests need no broker, no files and
no clock beyond the ``now`` they pass in.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from memetrader import config, risk
from memetrader.types import (
    CoinSnapshot,
    FillQuote,
    MarketSnapshot,
    PortfolioState,
    Position,
    PriceLadder,
    Side,
    TradeProposal,
    TxnCounts,
)

MINT = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
NOW = 1_700_000_000.0

# Defaults from config.toml, restated so the tests read as arithmetic:
#   max_position_pct         0.30       max_price_impact_pct      3.0
#   min_trade_usd           10.0        max_snapshot_age_seconds  90
#   min_liquidity_usd   50_000.0        gas_usd_per_swap          0.21


@pytest.fixture(scope="module")
def cfg() -> config.Config:
    base = config.load()
    return replace(base, execution=replace(base.execution, gas_usd_per_swap=0.21))


def coin(symbol: str = "BONK", *, liquidity_usd: float = 250_000.0,
         price_usd: float = 0.00002) -> CoinSnapshot:
    return CoinSnapshot(
        symbol=symbol,
        mint=MINT,
        price_usd=price_usd,
        liquidity_usd=liquidity_usd,
        volume_24h_usd=1_000_000.0,
        volume_1h_usd=50_000.0,
        fdv_usd=None,
        price_change=PriceLadder(m5=0.1, h1=0.5, h6=-1.0, h24=2.0),
        txns_m5=TxnCounts(buys=10, sells=8),
        txns_h1=TxnCounts(buys=100, sells=90),
        txns_h24=TxnCounts(buys=1000, sells=900),
        pair_address="pair",
        dex_id="raydium",
        pair_created_at=None,
    )


def snapshot(*coins: CoinSnapshot, age_seconds: float = 5.0) -> MarketSnapshot:
    return MarketSnapshot(
        ts=NOW - age_seconds, coins={c.symbol: c for c in coins or (coin(),)}
    )


def quote(
    *,
    symbol: str = "BONK",
    side: Side = Side.BUY,
    usd_notional: float = 100.0,
    price_usd: float = 0.00002,
    price_impact_pct: float = 0.4,
    pool_fee_pct: float = 0.25,
    degraded: bool = False,
) -> FillQuote:
    """A routed Jupiter quote by default.

    On a routed quote the pool fee is already inside ``price_usd``, so the cash
    clamp only has to reserve gas. ``degraded=True`` is the DexScreener-mid
    fallback, the one path where the fee is a real uncounted cost.
    """
    return FillQuote(
        symbol=symbol,
        mint=MINT,
        side=side,
        usd_notional=usd_notional,
        price_usd=price_usd,
        price_impact_pct=price_impact_pct,
        route_labels=("Raydium",),
        pool_fee_pct=pool_fee_pct,
        degraded=degraded,
    )


def book(
    *, cash: float = 1000.0, holdings: dict[str, tuple[float, float]] | None = None
) -> PortfolioState:
    """``holdings`` maps symbol -> (quantity, mark price). Positions are flat."""
    holdings = holdings or {}
    positions: dict[str, Position] = {}
    values: dict[str, float] = {}
    marks: dict[str, float] = {}
    for symbol, (quantity, price) in holdings.items():
        positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            avg_entry_price_usd=price,
            opened_at=NOW - 60.0,
            cost_basis_usd=quantity * price,
        )
        values[symbol] = quantity * price
        marks[symbol] = price
    return PortfolioState(
        ts=NOW,
        cash_usd=cash,
        positions=positions,
        marks=marks,
        position_values_usd=values,
        unrealized_pnl_usd=0.0,
        realized_pnl_usd=0.0,
        total_value_usd=cash + sum(values.values()),
        starting_cash_usd=1000.0,
    )


def proposal(
    *,
    symbol: str = "BONK",
    side: Side = Side.BUY,
    usd_notional: float = 100.0,
    source: str = "model",
    q: FillQuote | None | object = ...,
) -> TradeProposal:
    if q is ...:
        q = quote(symbol=symbol, side=side, usd_notional=usd_notional)
    return TradeProposal(
        symbol=symbol,
        side=side,
        usd_notional=usd_notional,
        quote=q,  # type: ignore[arg-type]
        source=source,  # type: ignore[arg-type]
        confidence=0.7,
        reasoning="because",
    )


def check(p: TradeProposal, cfg: config.Config, **kw):
    state = kw.pop("state", None) or book()
    snap = kw.pop("snapshot", None) or snapshot()
    return risk.check(p, state, snap, cfg, now=kw.pop("now", NOW))


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_a_plain_buy_is_approved_at_full_size(cfg: config.Config) -> None:
    verdict = check(proposal(usd_notional=100.0), cfg)
    assert verdict.approved is True
    assert verdict.approved_usd == pytest.approx(100.0)
    assert verdict.rule is None
    assert verdict.notes == ()


def test_a_plain_sell_is_approved_at_full_size(cfg: config.Config) -> None:
    state = book(holdings={"BONK": (10_000_000.0, 0.00002)})  # worth 200
    verdict = check(
        proposal(side=Side.SELL, usd_notional=150.0), cfg, state=state
    )
    assert verdict.approved is True
    assert verdict.approved_usd == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# stale_data
# ---------------------------------------------------------------------------


def test_stale_data_rejects(cfg: config.Config) -> None:
    verdict = check(proposal(), cfg, snapshot=snapshot(age_seconds=91.0))
    assert verdict.approved is False
    assert verdict.approved_usd == 0.0
    assert verdict.rule == "stale_data"
    assert "91" in verdict.reason and "90" in verdict.reason


def test_fresh_data_passes(cfg: config.Config) -> None:
    assert check(proposal(), cfg, snapshot=snapshot(age_seconds=89.0)).approved


def test_snapshot_exactly_at_the_age_limit_passes(cfg: config.Config) -> None:
    """The rule is ``>``, so the limit itself is still tradeable."""
    assert check(proposal(), cfg, snapshot=snapshot(age_seconds=90.0)).approved


def test_stale_data_outranks_every_other_breach(cfg: config.Config) -> None:
    """Blind is blind; do not report a liquidity opinion formed from old data."""
    verdict = check(
        proposal(usd_notional=1.0),
        cfg,
        snapshot=snapshot(coin(liquidity_usd=1.0), age_seconds=600.0),
    )
    assert verdict.rule == "stale_data"


# ---------------------------------------------------------------------------
# missing_quote
# ---------------------------------------------------------------------------


def test_missing_quote_rejects(cfg: config.Config) -> None:
    verdict = check(proposal(q=None), cfg)
    assert verdict.approved is False
    assert verdict.rule == "missing_quote"
    assert verdict.approved_usd == 0.0


def test_missing_quote_rejects_even_a_stop_loss(cfg: config.Config) -> None:
    """The one rule a forced exit cannot argue with: there is nothing to fill."""
    state = book(holdings={"BONK": (10_000_000.0, 0.00002)})
    verdict = check(
        proposal(side=Side.SELL, usd_notional=200.0, source="stop_loss", q=None),
        cfg,
        state=state,
    )
    assert verdict.approved is False
    assert verdict.rule == "missing_quote"


# ---------------------------------------------------------------------------
# max_price_impact
# ---------------------------------------------------------------------------


def test_price_impact_above_the_limit_rejects(cfg: config.Config) -> None:
    verdict = check(proposal(q=quote(price_impact_pct=3.1)), cfg)
    assert verdict.approved is False
    assert verdict.rule == "max_price_impact"
    assert "3.1" in verdict.reason


def test_price_impact_below_the_limit_passes(cfg: config.Config) -> None:
    assert check(proposal(q=quote(price_impact_pct=2.9)), cfg).approved


def test_price_impact_exactly_at_the_limit_passes(cfg: config.Config) -> None:
    assert check(proposal(q=quote(price_impact_pct=3.0)), cfg).approved


# ---------------------------------------------------------------------------
# min_liquidity
# ---------------------------------------------------------------------------


def test_thin_liquidity_rejects(cfg: config.Config) -> None:
    verdict = check(proposal(), cfg, snapshot=snapshot(coin(liquidity_usd=49_999.0)))
    assert verdict.approved is False
    assert verdict.rule == "min_liquidity"


def test_deep_liquidity_passes(cfg: config.Config) -> None:
    assert check(
        proposal(), cfg, snapshot=snapshot(coin(liquidity_usd=50_001.0))
    ).approved


def test_liquidity_exactly_at_the_floor_passes(cfg: config.Config) -> None:
    assert check(
        proposal(), cfg, snapshot=snapshot(coin(liquidity_usd=50_000.0))
    ).approved


def test_a_symbol_absent_from_the_snapshot_rejects(cfg: config.Config) -> None:
    verdict = check(proposal(symbol="WIF", q=quote(symbol="WIF")), cfg)
    assert verdict.approved is False
    assert verdict.rule == "missing_snapshot"
    assert "WIF" in verdict.reason


# ---------------------------------------------------------------------------
# min_trade_size
# ---------------------------------------------------------------------------


def test_dust_trade_rejects(cfg: config.Config) -> None:
    verdict = check(proposal(usd_notional=9.99), cfg)
    assert verdict.approved is False
    assert verdict.rule == "min_trade_size"


def test_trade_exactly_at_the_minimum_passes(cfg: config.Config) -> None:
    verdict = check(proposal(usd_notional=10.0), cfg)
    assert verdict.approved is True
    assert verdict.approved_usd == pytest.approx(10.0)


def test_zero_size_rejects(cfg: config.Config) -> None:
    assert check(proposal(usd_notional=0.0), cfg).rule == "min_trade_size"


# ---------------------------------------------------------------------------
# max_position_pct — clamps rather than rejecting
# ---------------------------------------------------------------------------


def test_position_limit_clamps_a_too_large_buy(cfg: config.Config) -> None:
    # Book 1000 -> limit 300. Already holding 100 of BONK, so 200 is legal.
    state = book(cash=900.0, holdings={"BONK": (5_000_000.0, 0.00002)})
    verdict = check(proposal(usd_notional=400.0), cfg, state=state)

    assert verdict.approved is True
    assert verdict.approved_usd == pytest.approx(200.0)
    assert verdict.notes and any("30" in n for n in verdict.notes)
    assert any("200" in n for n in verdict.notes)


def test_buy_inside_the_position_limit_is_untouched(cfg: config.Config) -> None:
    state = book(cash=900.0, holdings={"BONK": (5_000_000.0, 0.00002)})
    verdict = check(proposal(usd_notional=150.0), cfg, state=state)

    assert verdict.approved is True
    assert verdict.approved_usd == pytest.approx(150.0)
    assert verdict.notes == ()


def test_buy_exactly_at_the_position_limit_is_untouched(cfg: config.Config) -> None:
    state = book(cash=900.0, holdings={"BONK": (5_000_000.0, 0.00002)})
    verdict = check(proposal(usd_notional=200.0), cfg, state=state)
    assert verdict.approved_usd == pytest.approx(200.0)
    assert verdict.notes == ()


def test_position_limit_rejects_when_the_clamp_would_be_dust(
    cfg: config.Config,
) -> None:
    # Holding 295 of a 1000 book: only 5 of headroom, below the 10 minimum.
    state = book(cash=705.0, holdings={"BONK": (14_750_000.0, 0.00002)})
    verdict = check(proposal(usd_notional=400.0), cfg, state=state)

    assert verdict.approved is False
    assert verdict.approved_usd == 0.0
    assert verdict.rule == "max_position_pct"


def test_position_limit_rejects_when_already_over(cfg: config.Config) -> None:
    state = book(cash=600.0, holdings={"BONK": (20_000_000.0, 0.00002)})  # 400/1000
    verdict = check(proposal(usd_notional=50.0), cfg, state=state)
    assert verdict.approved is False
    assert verdict.rule == "max_position_pct"


def test_position_limit_reason_is_actionable(cfg: config.Config) -> None:
    """Spec: 'would put BONK at 41% of book, limit is 30%' beats a rule name."""
    state = book(cash=705.0, holdings={"BONK": (14_750_000.0, 0.00002)})
    reason = check(proposal(usd_notional=400.0), cfg, state=state).reason

    assert "BONK" in reason
    assert "30" in reason  # the limit, as a percent of book
    assert "%" in reason
    assert "400" in reason  # what was asked for


def test_position_limit_does_not_apply_to_sells(cfg: config.Config) -> None:
    """Selling always reduces concentration; the rule must not block an exit."""
    state = book(cash=100.0, holdings={"BONK": (30_000_000.0, 0.00002)})  # 600/700
    verdict = check(
        proposal(side=Side.SELL, usd_notional=500.0), cfg, state=state
    )
    assert verdict.approved is True
    assert verdict.approved_usd == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# insufficient_cash — clamps rather than rejecting
# ---------------------------------------------------------------------------


def test_insufficient_cash_clamps(cfg: config.Config) -> None:
    # Cash 100, routed quote (no pool fee on top), gas 0.21.
    #   largest N with N + 0.21 <= 100  ->  N = 99.79
    state = book(cash=100.0, holdings={"WIF": (1000.0, 1.9)})  # book = 2000
    verdict = check(proposal(usd_notional=400.0), cfg, state=state)

    assert verdict.approved is True
    assert verdict.approved_usd == pytest.approx(99.79)
    assert verdict.approved_usd < 100.0
    assert any("cash" in n.lower() for n in verdict.notes)


def test_the_routed_cash_clamp_leaves_room_for_gas(cfg: config.Config) -> None:
    state = book(cash=100.0, holdings={"WIF": (1000.0, 1.9)})
    approved = check(proposal(usd_notional=400.0), cfg, state=state).approved_usd
    total = approved + 0.21  # routed: no pool fee charged on top
    assert total == pytest.approx(100.0)
    assert total <= 100.0 + 1e-9


def test_the_degraded_cash_clamp_leaves_room_for_fee_and_gas(
    cfg: config.Config,
) -> None:
    state = book(cash=100.0, holdings={"WIF": (1000.0, 1.9)})
    approved = check(
        proposal(usd_notional=400.0, q=quote(degraded=True)), cfg, state=state
    ).approved_usd
    total = approved + approved * 0.25 / 100.0 + 0.21
    assert approved == pytest.approx(99.79 / 1.0025)
    assert total == pytest.approx(100.0)
    assert total <= 100.0 + 1e-9


def test_the_clamp_is_always_affordable_at_the_broker(cfg: config.Config) -> None:
    """The clamp and the broker's charge must never disagree: an approved size
    the broker then refuses with InsufficientCash is the failure this guards.
    Both read the reservation from ``broker.pool_fee_rate``."""
    for degraded in (False, True):
        for cash in (12.0, 99.9, 100.0, 250.0, 1_000.0):
            q = quote(degraded=degraded, pool_fee_pct=0.55)
            state = book(cash=cash, holdings={"WIF": (1000.0, 1.9)})
            verdict = check(proposal(usd_notional=5_000.0, q=q), cfg, state=state)
            if not verdict.approved:
                continue
            charged = verdict.approved_usd * (
                1.0 + (0.55 / 100.0 if degraded else 0.0)
            ) + 0.21
            assert charged <= cash, (degraded, cash, charged)


def test_insufficient_cash_rejects_when_the_clamp_would_be_dust(
    cfg: config.Config,
) -> None:
    state = book(cash=5.0, holdings={"WIF": (1000.0, 1.9)})
    verdict = check(proposal(usd_notional=400.0), cfg, state=state)

    assert verdict.approved is False
    assert verdict.rule == "insufficient_cash"
    assert verdict.approved_usd == 0.0


def test_no_cash_at_all_rejects(cfg: config.Config) -> None:
    state = book(cash=0.0, holdings={"WIF": (1000.0, 1.9)})
    verdict = check(proposal(usd_notional=400.0), cfg, state=state)
    assert verdict.approved is False
    assert verdict.rule == "insufficient_cash"


def test_ample_cash_passes_untouched(cfg: config.Config) -> None:
    verdict = check(proposal(usd_notional=100.0), cfg, state=book(cash=1000.0))
    assert verdict.approved_usd == pytest.approx(100.0)
    assert verdict.notes == ()


def test_cash_is_not_checked_for_sells(cfg: config.Config) -> None:
    state = book(cash=0.0, holdings={"BONK": (10_000_000.0, 0.00002)})
    verdict = check(
        proposal(side=Side.SELL, usd_notional=200.0), cfg, state=state
    )
    assert verdict.approved is True


def test_both_clamps_compose_to_the_tighter_one(cfg: config.Config) -> None:
    """Position limit says 300, cash says ~99.54; the smaller must win."""
    state = book(cash=100.0, holdings={"WIF": (1000.0, 0.9)})  # book = 1000
    verdict = check(proposal(usd_notional=800.0), cfg, state=state)

    assert verdict.approved is True
    assert verdict.approved_usd == pytest.approx(99.79)
    assert len(verdict.notes) == 2


# ---------------------------------------------------------------------------
# no_position and the sell clamp
# ---------------------------------------------------------------------------


def test_sell_with_no_position_rejects(cfg: config.Config) -> None:
    verdict = check(proposal(side=Side.SELL, usd_notional=100.0), cfg)
    assert verdict.approved is False
    assert verdict.rule == "no_position"
    assert "BONK" in verdict.reason


def test_sell_clamps_to_the_position_value(cfg: config.Config) -> None:
    state = book(holdings={"BONK": (10_000_000.0, 0.00002)})  # worth 200
    verdict = check(
        proposal(side=Side.SELL, usd_notional=500.0), cfg, state=state
    )

    assert verdict.approved is True
    assert verdict.approved_usd == pytest.approx(200.0)
    assert any("200" in n for n in verdict.notes)


def test_sell_of_exactly_the_position_value_is_untouched(
    cfg: config.Config,
) -> None:
    state = book(holdings={"BONK": (10_000_000.0, 0.00002)})
    verdict = check(
        proposal(side=Side.SELL, usd_notional=200.0), cfg, state=state
    )
    assert verdict.approved_usd == pytest.approx(200.0)
    assert verdict.notes == ()


def test_model_sell_clamped_below_the_minimum_rejects(cfg: config.Config) -> None:
    state = book(holdings={"BONK": (150_000.0, 0.00002)})  # worth 3
    verdict = check(
        proposal(side=Side.SELL, usd_notional=100.0), cfg, state=state
    )
    assert verdict.approved is False
    assert verdict.rule == "min_trade_size"


# ---------------------------------------------------------------------------
# stop_loss bypass
# ---------------------------------------------------------------------------


def test_stop_loss_bypasses_min_trade_size(cfg: config.Config) -> None:
    state = book(holdings={"BONK": (150_000.0, 0.00002)})  # worth 3
    verdict = check(
        proposal(side=Side.SELL, usd_notional=3.0, source="stop_loss"),
        cfg,
        state=state,
    )
    assert verdict.approved is True
    assert verdict.approved_usd == pytest.approx(3.0)


def test_stop_loss_bypasses_min_trade_size_after_the_sell_clamp(
    cfg: config.Config,
) -> None:
    state = book(holdings={"BONK": (150_000.0, 0.00002)})  # worth 3
    verdict = check(
        proposal(side=Side.SELL, usd_notional=100.0, source="stop_loss"),
        cfg,
        state=state,
    )
    assert verdict.approved is True
    assert verdict.approved_usd == pytest.approx(3.0)


def test_stop_loss_bypasses_max_position_pct(cfg: config.Config) -> None:
    state = book(cash=900.0, holdings={"BONK": (5_000_000.0, 0.00002)})
    verdict = check(
        proposal(usd_notional=400.0, source="stop_loss"), cfg, state=state
    )
    assert verdict.approved is True
    assert verdict.approved_usd == pytest.approx(400.0)


def test_stop_loss_bypasses_max_price_impact(cfg: config.Config) -> None:
    """The bypass that matters. Price impact blows out exactly when a pool is
    collapsing, which is the scenario the stop exists for — enforcing the limit
    there does not protect the position, it traps it."""
    state = book(holdings={"BONK": (10_000_000.0, 0.00002)})  # worth 200
    verdict = check(
        proposal(
            side=Side.SELL,
            usd_notional=200.0,
            source="stop_loss",
            q=quote(side=Side.SELL, usd_notional=200.0, price_impact_pct=41.0),
        ),
        cfg,
        state=state,
    )
    assert verdict.approved is True
    assert verdict.approved_usd == pytest.approx(200.0)
    assert any("41.00% price impact" in n for n in verdict.notes)


def test_a_model_sell_is_still_blocked_by_max_price_impact(cfg: config.Config) -> None:
    """The same trade without ``source="stop_loss"`` still gets the veto, so the
    bypass is genuinely conditional and not an accidental blanket hole."""
    state = book(holdings={"BONK": (10_000_000.0, 0.00002)})
    verdict = check(
        proposal(
            side=Side.SELL,
            usd_notional=200.0,
            q=quote(side=Side.SELL, usd_notional=200.0, price_impact_pct=41.0),
        ),
        cfg,
        state=state,
    )
    assert verdict.approved is False
    assert verdict.rule == "max_price_impact"


def test_stop_loss_bypasses_min_liquidity(cfg: config.Config) -> None:
    """A draining pool is the reason to get out, not a reason to stay in."""
    state = book(holdings={"BONK": (10_000_000.0, 0.00002)})
    verdict = check(
        proposal(side=Side.SELL, usd_notional=200.0, source="stop_loss"),
        cfg,
        state=state,
        snapshot=snapshot(coin(liquidity_usd=1_200.0)),
    )
    assert verdict.approved is True
    assert verdict.approved_usd == pytest.approx(200.0)
    assert any("draining pool" in n for n in verdict.notes)


def test_stop_loss_bypasses_a_missing_snapshot(cfg: config.Config) -> None:
    """``missing_snapshot`` exists only to guard the liquidity check, so once a
    forced exit bypasses that check there is nothing left for it to protect."""
    state = book(holdings={"WIF": (100.0, 2.0)})  # worth 200
    verdict = check(
        proposal(
            symbol="WIF", side=Side.SELL, usd_notional=200.0, source="stop_loss"
        ),
        cfg,
        state=state,
    )
    assert verdict.approved is True
    assert verdict.approved_usd == pytest.approx(200.0)
    assert any("verify liquidity" in n for n in verdict.notes)


def test_stop_loss_is_still_blocked_by_stale_data(cfg: config.Config) -> None:
    """Deliberately *not* bypassed. The fast tick retries in 60s, so blocking
    here costs a minute; exiting on a price we cannot vouch for costs the fill."""
    state = book(holdings={"BONK": (10_000_000.0, 0.00002)})
    verdict = check(
        proposal(side=Side.SELL, usd_notional=200.0, source="stop_loss"),
        cfg,
        state=state,
        snapshot=snapshot(age_seconds=91.0),
    )
    assert verdict.approved is False
    assert verdict.rule == "stale_data"


def test_stop_loss_still_obeys_no_position(cfg: config.Config) -> None:
    verdict = check(
        proposal(side=Side.SELL, usd_notional=100.0, source="stop_loss"), cfg
    )
    assert verdict.approved is False
    assert verdict.rule == "no_position"


def test_stop_loss_still_clamps_to_the_position_value(cfg: config.Config) -> None:
    state = book(holdings={"BONK": (10_000_000.0, 0.00002)})  # worth 200
    verdict = check(
        proposal(side=Side.SELL, usd_notional=999.0, source="stop_loss"),
        cfg,
        state=state,
    )
    assert verdict.approved_usd == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Rules that must NOT exist
# ---------------------------------------------------------------------------


def test_there_is_no_minimum_hold_time(cfg: config.Config) -> None:
    """A position opened one second ago can be sold. Deliberate."""
    state = book(holdings={"BONK": (10_000_000.0, 0.00002)})
    state.positions["BONK"] = replace(state.positions["BONK"], opened_at=NOW - 1.0)
    verdict = check(
        proposal(side=Side.SELL, usd_notional=200.0), cfg, state=state
    )
    assert verdict.approved is True


def test_there_is_no_max_trades_per_day(cfg: config.Config) -> None:
    """``check`` is pure — calling it a hundred times changes nothing."""
    verdicts = [check(proposal(usd_notional=50.0), cfg) for _ in range(100)]
    assert all(v.approved for v in verdicts)
    assert len({(v.approved, v.approved_usd) for v in verdicts}) == 1


def test_check_does_not_mutate_its_inputs(cfg: config.Config) -> None:
    state = book(cash=900.0, holdings={"BONK": (5_000_000.0, 0.00002)})
    snap = snapshot()
    p = proposal(usd_notional=400.0)
    before = (dict(state.positions), dict(state.position_values_usd), state.cash_usd)

    check(p, cfg, state=state, snapshot=snap)

    assert (dict(state.positions), dict(state.position_values_usd), state.cash_usd) == before
    assert p.usd_notional == 400.0
    assert snap.coins["BONK"].liquidity_usd == 250_000.0


# ---------------------------------------------------------------------------
# Config is honoured, nothing is hardcoded
# ---------------------------------------------------------------------------


def test_limits_come_from_config_not_constants(cfg: config.Config) -> None:
    loose = replace(
        cfg,
        risk=replace(
            cfg.risk,
            max_position_pct=0.9,
            min_trade_usd=1.0,
            max_price_impact_pct=10.0,
            min_liquidity_usd=1.0,
            max_snapshot_age_seconds=10_000.0,
        ),
    )
    verdict = check(
        proposal(usd_notional=2.0, q=quote(price_impact_pct=8.0)),
        loose,
        snapshot=snapshot(coin(liquidity_usd=100.0), age_seconds=5_000.0),
    )
    assert verdict.approved is True
    assert verdict.approved_usd == pytest.approx(2.0)


def test_a_higher_gas_cost_tightens_the_cash_clamp(cfg: config.Config) -> None:
    pricey = replace(cfg, execution=replace(cfg.execution, gas_usd_per_swap=5.0))
    state = book(cash=100.0, holdings={"WIF": (1000.0, 1.9)})

    cheap_usd = check(proposal(usd_notional=400.0), cfg, state=state).approved_usd
    pricey_usd = check(proposal(usd_notional=400.0), pricey, state=state).approved_usd

    assert pricey_usd < cheap_usd
    assert pricey_usd == pytest.approx(95.0)


def test_a_multi_hop_fee_tightens_the_clamp_on_the_degraded_path(
    cfg: config.Config,
) -> None:
    """A two-hop degraded quote resolves to 0.55% via ``fee_pct_for``, and that
    has to come out of the clamp because nothing else has counted it."""
    state = book(cash=100.0, holdings={"WIF": (1000.0, 1.9)})
    verdict = check(
        proposal(usd_notional=400.0, q=quote(pool_fee_pct=0.55, degraded=True)),
        cfg,
        state=state,
    )
    assert verdict.approved_usd == pytest.approx(99.79 / 1.0055)


def test_a_routed_pool_fee_does_not_tighten_the_clamp(cfg: config.Config) -> None:
    """The mirror image, and the actual correction: on a real Jupiter route the
    fee is already inside ``price_usd``, so reserving it again would shrink
    every order by a fee the user is not going to be charged twice."""
    state = book(cash=100.0, holdings={"WIF": (1000.0, 1.9)})
    cheap = check(
        proposal(usd_notional=400.0, q=quote(pool_fee_pct=0.25)), cfg, state=state
    )
    pricey = check(
        proposal(usd_notional=400.0, q=quote(pool_fee_pct=5.00)), cfg, state=state
    )
    assert cheap.approved_usd == pytest.approx(pricey.approved_usd)
    assert cheap.approved_usd == pytest.approx(99.79)


# ---------------------------------------------------------------------------
# Every verdict names its coin
# ---------------------------------------------------------------------------
#
# The decision log replays rejections to the model on the next tick. With three
# coins in play, a bare rule name leaves the model to guess which coin it fired
# on, so `symbol` is set on approvals and rejections alike.
# ---------------------------------------------------------------------------


def test_approved_verdicts_carry_the_symbol(cfg: config.Config) -> None:
    assert check(proposal(usd_notional=100.0), cfg).symbol == "BONK"


def test_clamped_verdicts_carry_the_symbol(cfg: config.Config) -> None:
    state = book(cash=900.0, holdings={"BONK": (5_000_000.0, 0.00002)})
    verdict = check(proposal(usd_notional=400.0), cfg, state=state)
    assert verdict.approved is True
    assert verdict.symbol == "BONK"


def test_every_rejection_path_carries_the_symbol(cfg: config.Config) -> None:
    """One case per rule, so a rule added later without a symbol is caught."""
    held = book(holdings={"BONK": (10_000_000.0, 0.00002)})
    thin = book(holdings={"BONK": (150_000.0, 0.00002)})
    broke = book(cash=5.0, holdings={"WIF": (1000.0, 1.9)})
    full = book(cash=705.0, holdings={"BONK": (14_750_000.0, 0.00002)})

    cases = {
        "stale_data": (proposal(), book(), snapshot(age_seconds=600.0)),
        "missing_quote": (proposal(q=None), book(), snapshot()),
        "max_price_impact": (
            proposal(q=quote(price_impact_pct=9.9)),
            book(),
            snapshot(),
        ),
        "missing_snapshot": (
            proposal(symbol="BONK"),
            book(),
            MarketSnapshot(ts=NOW, coins={}),
        ),
        "min_liquidity": (
            proposal(),
            book(),
            snapshot(coin(liquidity_usd=1.0)),
        ),
        "min_trade_size": (proposal(usd_notional=1.0), book(), snapshot()),
        "max_position_pct": (proposal(usd_notional=400.0), full, snapshot()),
        "insufficient_cash": (proposal(usd_notional=400.0), broke, snapshot()),
        "no_position": (
            proposal(side=Side.SELL, usd_notional=100.0),
            book(),
            snapshot(),
        ),
    }
    for expected_rule, (p, state, snap) in cases.items():
        verdict = risk.check(p, state, snap, cfg, now=NOW)
        assert verdict.approved is False, expected_rule
        assert verdict.rule == expected_rule
        assert verdict.symbol == "BONK", expected_rule
        assert verdict.reason

    # And the clamp-into-dust rejection, which reports min_trade_size.
    dust = check(proposal(side=Side.SELL, usd_notional=100.0), cfg, state=thin)
    assert dust.rule == "min_trade_size"
    assert dust.symbol == "BONK"

    # A SELL that is fine, for contrast.
    assert check(
        proposal(side=Side.SELL, usd_notional=100.0), cfg, state=held
    ).symbol == "BONK"


def test_the_symbol_is_the_proposal_symbol_not_a_constant(
    cfg: config.Config,
) -> None:
    verdict = check(
        proposal(symbol="POPCAT", q=quote(symbol="POPCAT", price_usd=0.41)),
        cfg,
        snapshot=snapshot(coin("POPCAT", price_usd=0.41)),
    )
    assert verdict.approved is True
    assert verdict.symbol == "POPCAT"
