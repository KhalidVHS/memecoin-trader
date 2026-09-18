"""Marking the book, and the stop-loss trigger.

The stop-loss tests exist mostly to pin down one conversion: ``stop_loss_pct``
is a *fraction* (0.15) and ``unrealized_pnl_pct`` is a *whole percent* (-15.0).
Getting that wrong is a silent 100x, and the boundary case is worse than it
looks because ``0.15 * 100`` is ``15.000000000000002`` in binary floating point,
so a naive ``<=`` misses an exactly-15% drawdown.
"""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import pytest

from memetrader import config
from memetrader.broker import LocalPaperBroker
from memetrader.portfolio import mark, stop_loss_breaches
from memetrader.types import FillQuote, Side

MINT = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"


def make_cfg(tmp_path: Path, *, starting_cash_usd: float = 1000.0) -> config.Config:
    base = config.load()
    return replace(
        base,
        data_dir=tmp_path,
        starting_cash_usd=starting_cash_usd,
        execution=replace(base.execution, failed_tx_rate=0.0, gas_usd_per_swap=0.21),
    )


def quote(
    *,
    symbol: str = "BONK",
    side: Side = Side.BUY,
    usd_notional: float = 400.0,
    price_usd: float = 0.00002,
    pool_fee_pct: float = 0.25,
) -> FillQuote:
    return FillQuote(
        symbol=symbol,
        mint=MINT,
        side=side,
        usd_notional=usd_notional,
        price_usd=price_usd,
        price_impact_pct=0.4,
        route_labels=("Raydium",),
        pool_fee_pct=pool_fee_pct,
    )


def funded(tmp_path: Path) -> LocalPaperBroker:
    """Cash 1000 -> BUY $400 of BONK @ 0.00002 on a routed quote.

    A routed price already includes the pool fee, so only gas is charged on
    top. Leaves cash 599.79, 20,000,000 BONK, cost basis 400.21.
    """
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=random.Random(0))
    broker.place_order("BONK", Side.BUY, 400.0, quote=quote())
    return broker


# ---------------------------------------------------------------------------
# mark()
# ---------------------------------------------------------------------------


def test_empty_book_marks_to_cash(tmp_path: Path) -> None:
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=random.Random(0))
    state = mark(broker, {}, now=1_700_000_000.0)

    assert state.ts == 1_700_000_000.0
    assert state.cash_usd == 1000.0
    assert state.total_value_usd == 1000.0
    assert state.positions == {}
    assert state.position_values_usd == {}
    assert state.unrealized_pnl_usd == 0.0
    assert state.realized_pnl_usd == 0.0
    assert state.total_return_pct == 0.0
    assert state.starting_cash_usd == 1000.0


def test_total_value_is_cash_plus_marks(tmp_path: Path) -> None:
    broker = funded(tmp_path)
    state = mark(broker, {"BONK": 0.000025}, now=1.0)

    assert state.cash_usd == pytest.approx(599.79)
    assert state.position_values_usd["BONK"] == pytest.approx(500.0)
    assert state.total_value_usd == pytest.approx(1099.79)
    assert state.marks["BONK"] == 0.000025


def test_unrealized_pnl_is_against_cost_basis_not_notional(tmp_path: Path) -> None:
    broker = funded(tmp_path)
    state = mark(broker, {"BONK": 0.000025}, now=1.0)

    # 500.00 mark - 400.21 cost basis. A simulator that used the 400 notional
    # would overstate this by the 0.21 of entry gas.
    assert state.unrealized_pnl_usd == pytest.approx(99.79)


def test_total_return_pct(tmp_path: Path) -> None:
    broker = funded(tmp_path)
    state = mark(broker, {"BONK": 0.000025}, now=1.0)

    # (1099.79 - 1000) / 1000 * 100
    assert state.total_return_pct == pytest.approx(9.979)

    down = mark(broker, {"BONK": 0.00001}, now=1.0)
    # cash 599.79 + 200.00 = 799.79 -> -20.021%
    assert down.total_value_usd == pytest.approx(799.79)
    assert down.total_return_pct == pytest.approx(-20.021)


def test_fees_gas_and_realized_are_carried_through(tmp_path: Path) -> None:
    broker = funded(tmp_path)
    broker.place_order(
        "BONK", Side.SELL, 250.0, quote=quote(side=Side.SELL, price_usd=0.000025)
    )
    state = mark(broker, {"BONK": 0.000025}, now=1.0)

    # Routed quotes, so no pool fee is charged on top; gas on both legs.
    assert state.fees_paid_usd == 0.0
    assert state.gas_paid_usd == pytest.approx(0.42)
    # 250 - (400.21 / 2) - 0.21
    assert state.realized_pnl_usd == pytest.approx(49.685)


def test_missing_mark_does_not_crash_and_carries_at_cost(tmp_path: Path) -> None:
    broker = funded(tmp_path)
    state = mark(broker, {}, now=1.0)  # no price for BONK at all

    assert state.position_values_usd["BONK"] == pytest.approx(400.21)
    assert state.unrealized_pnl_usd == pytest.approx(0.0)
    assert state.total_value_usd == pytest.approx(599.79 + 400.21)
    # The implied mark is recorded so downstream code has a number to print.
    assert state.marks["BONK"] == pytest.approx(400.21 / 20_000_000.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_unusable_mark_is_treated_as_missing(tmp_path: Path, bad: float) -> None:
    broker = funded(tmp_path)
    state = mark(broker, {"BONK": bad}, now=1.0)

    assert state.position_values_usd["BONK"] == pytest.approx(400.21)
    assert state.unrealized_pnl_usd == pytest.approx(0.0)


def test_marks_for_unheld_symbols_are_ignored(tmp_path: Path) -> None:
    broker = funded(tmp_path)
    state = mark(broker, {"BONK": 0.00002, "WIF": 1.83, "POPCAT": 0.41}, now=1.0)

    assert set(state.marks) == {"BONK"}
    assert set(state.position_values_usd) == {"BONK"}


def test_now_defaults_to_wall_clock(tmp_path: Path) -> None:
    broker = funded(tmp_path)
    state = mark(broker, {"BONK": 0.00002})
    assert state.ts > 1_700_000_000.0


def test_mark_does_not_mutate_broker_state(tmp_path: Path) -> None:
    broker = funded(tmp_path)
    before = broker.get_positions()
    state = mark(broker, {"BONK": 0.000025}, now=1.0)
    state.positions.clear()

    assert broker.get_positions() == before


# ---------------------------------------------------------------------------
# stop_loss_breaches()
# ---------------------------------------------------------------------------


def price_for_drawdown(cost_basis: float, quantity: float, pct: float) -> float:
    """Mark price that puts the position exactly ``pct`` below cost basis."""
    return cost_basis * (1.0 + pct / 100.0) / quantity


def test_stop_loss_fires_at_exactly_minus_fifteen(tmp_path: Path) -> None:
    broker = funded(tmp_path)
    pos = broker.get_positions()["BONK"]
    price = price_for_drawdown(pos.cost_basis_usd, pos.quantity, -15.0)

    state = mark(broker, {"BONK": price}, now=1.0)
    assert pos.unrealized_pnl_pct(price) == pytest.approx(-15.0, abs=1e-9)
    assert stop_loss_breaches(state, 0.15) == ["BONK"]


def test_stop_loss_does_not_fire_at_minus_fourteen_point_nine(
    tmp_path: Path,
) -> None:
    broker = funded(tmp_path)
    pos = broker.get_positions()["BONK"]
    price = price_for_drawdown(pos.cost_basis_usd, pos.quantity, -14.9)

    state = mark(broker, {"BONK": price}, now=1.0)
    assert stop_loss_breaches(state, 0.15) == []


def test_stop_loss_fires_well_below_the_threshold(tmp_path: Path) -> None:
    broker = funded(tmp_path)
    pos = broker.get_positions()["BONK"]
    price = price_for_drawdown(pos.cost_basis_usd, pos.quantity, -40.0)

    state = mark(broker, {"BONK": price}, now=1.0)
    assert stop_loss_breaches(state, 0.15) == ["BONK"]


def test_stop_loss_ignores_winners(tmp_path: Path) -> None:
    broker = funded(tmp_path)
    state = mark(broker, {"BONK": 0.000025}, now=1.0)
    assert stop_loss_breaches(state, 0.15) == []


def test_stop_loss_pct_is_a_fraction_not_a_percent(tmp_path: Path) -> None:
    """If the conversion were dropped, passing 0.15 would mean -0.15% and this
    -1% position would breach. It must not."""
    broker = funded(tmp_path)
    pos = broker.get_positions()["BONK"]
    price = price_for_drawdown(pos.cost_basis_usd, pos.quantity, -1.0)

    state = mark(broker, {"BONK": price}, now=1.0)
    assert stop_loss_breaches(state, 0.15) == []
    # ...and if it were multiplied twice, a -15% position would need -1500%.
    deep = mark(
        broker,
        {"BONK": price_for_drawdown(pos.cost_basis_usd, pos.quantity, -15.0)},
        now=1.0,
    )
    assert stop_loss_breaches(deep, 0.15) == ["BONK"]


def test_breaches_are_ordered_worst_first(tmp_path: Path) -> None:
    broker = LocalPaperBroker(make_cfg(tmp_path, starting_cash_usd=5000.0),
                              rng=random.Random(0))
    broker.place_order("BONK", Side.BUY, 300.0, quote=quote(usd_notional=300.0))
    broker.place_order(
        "WIF", Side.BUY, 300.0,
        quote=quote(symbol="WIF", usd_notional=300.0, price_usd=1.83),
    )
    positions = broker.get_positions()

    marks = {
        "BONK": price_for_drawdown(
            positions["BONK"].cost_basis_usd, positions["BONK"].quantity, -20.0
        ),
        "WIF": price_for_drawdown(
            positions["WIF"].cost_basis_usd, positions["WIF"].quantity, -60.0
        ),
    }
    state = mark(broker, marks, now=1.0)
    assert stop_loss_breaches(state, 0.15) == ["WIF", "BONK"]


def test_stop_loss_on_an_unmarked_position_does_not_fire(tmp_path: Path) -> None:
    """Carried at cost means flat, and you do not stop out of a flat position
    on the strength of data you do not have."""
    broker = funded(tmp_path)
    state = mark(broker, {}, now=1.0)
    assert stop_loss_breaches(state, 0.15) == []


def test_stop_loss_on_an_empty_book(tmp_path: Path) -> None:
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=random.Random(0))
    assert stop_loss_breaches(mark(broker, {}, now=1.0), 0.15) == []
