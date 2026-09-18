"""Broker arithmetic, asserted against hand-computed numbers.

Every expected value in this file was worked out on paper first and is written
as a literal, not as a re-implementation of the code under test. A test that
recomputes the production formula proves nothing; these prove the formula.

No network, no real files — every test gets its own ``tmp_path`` data dir.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from memetrader import config
from memetrader.broker import InsufficientCash, LocalPaperBroker, NoPosition
from memetrader.types import FillQuote, Side

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

MINT = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"


def make_cfg(
    tmp_path: Path,
    *,
    starting_cash_usd: float = 1000.0,
    failed_tx_rate: float = 0.0,
    gas_usd_per_swap: float = 0.21,
) -> config.Config:
    """A real config with the data dir redirected and execution pinned."""
    base = config.load()
    execution = replace(
        base.execution,
        failed_tx_rate=failed_tx_rate,
        gas_usd_per_swap=gas_usd_per_swap,
    )
    return replace(
        base,
        data_dir=tmp_path,
        starting_cash_usd=starting_cash_usd,
        execution=execution,
    )


def quote(
    *,
    symbol: str = "BONK",
    side: Side = Side.BUY,
    usd_notional: float = 400.0,
    price_usd: float = 0.00002,
    pool_fee_pct: float = 0.25,
    price_impact_pct: float = 0.4,
    degraded: bool = False,
) -> FillQuote:
    """A routed Jupiter quote by default. ``degraded=True`` is the DexScreener
    mid fallback, which is the only path that pays an explicit pool fee."""
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


def never_fails() -> random.Random:
    """An rng is still injected so the draw sequence is explicit in tests."""
    return random.Random(1234)


# ---------------------------------------------------------------------------
# Cold start
# ---------------------------------------------------------------------------


def test_first_run_initializes_cash_to_starting_cash(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, starting_cash_usd=2500.0)
    broker = LocalPaperBroker(cfg, rng=never_fails())

    assert broker.cash_usd == 2500.0
    assert broker.get_positions() == {}
    assert broker.realized_pnl_usd == 0.0
    assert broker.fees_paid_usd == 0.0
    assert broker.gas_paid_usd == 0.0
    # No state file is written until something happens.
    assert not cfg.state_path.exists()


def test_get_positions_returns_a_copy(tmp_path: Path) -> None:
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    broker.place_order("BONK", Side.BUY, 400.0, quote=quote())

    positions = broker.get_positions()
    positions.clear()

    assert "BONK" in broker.get_positions()


# ---------------------------------------------------------------------------
# The round trip, computed by hand — routed (normal) path
# ---------------------------------------------------------------------------
#
#   A real Jupiter route already has every hop's pool fee deducted inside
#   `outAmount`, so `price_usd` is net of it and `pool_fee_usd` is 0.00.
#   Gas is still charged, because gas is paid to validators, not to the pool.
#
#   starting cash                                              1000.00
#
#   BUY  $400 of BONK @ 0.00002, gas 0.21
#     pool fee   (already inside the routed price)             =   0.00
#     gas                                                      =   0.21
#     quantity   = 400 / 0.00002                               = 20,000,000
#     cash       = 1000 - 400 - 0.21                           = 599.79
#     cost basis = 400 + 0.21                                  = 400.21
#
#   SELL the whole position @ 0.000025
#     proceeds   = 20,000,000 * 0.000025                       = 500.00
#     gas                                                      =   0.21
#     realized   = 500.00 - 400.21 - 0.21                      =  99.58
#     cash       = 599.79 + 500.00 - 0.21                      = 1099.58
#
#   Sanity: ending cash - starting cash = 1099.58 - 1000 = 99.58 = realized.
# ---------------------------------------------------------------------------


def test_buy_then_sell_round_trip_realized_pnl(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    broker = LocalPaperBroker(cfg, rng=never_fails())

    buy = broker.place_order("BONK", Side.BUY, 400.0, quote=quote())

    assert buy.failed is False
    assert buy.filled_usd == pytest.approx(400.0)
    assert buy.pool_fee_usd == 0.0  # already inside the routed price
    assert buy.gas_usd == pytest.approx(0.21)
    assert buy.quantity == pytest.approx(20_000_000.0)
    assert buy.realized_pnl_usd == 0.0
    assert broker.cash_usd == pytest.approx(599.79)

    pos = broker.get_positions()["BONK"]
    assert pos.quantity == pytest.approx(20_000_000.0)
    assert pos.avg_entry_price_usd == pytest.approx(0.00002)
    assert pos.cost_basis_usd == pytest.approx(400.21)

    sell = broker.place_order(
        "BONK",
        Side.SELL,
        500.0,
        quote=quote(side=Side.SELL, usd_notional=500.0, price_usd=0.000025),
    )

    assert sell.failed is False
    assert sell.filled_usd == pytest.approx(500.00)
    assert sell.pool_fee_usd == 0.0
    assert sell.gas_usd == pytest.approx(0.21)
    assert sell.realized_pnl_usd == pytest.approx(99.58)

    assert broker.cash_usd == pytest.approx(1099.58)
    assert broker.realized_pnl_usd == pytest.approx(99.58)
    assert broker.get_positions() == {}

    assert broker.fees_paid_usd == 0.0
    assert broker.gas_paid_usd == pytest.approx(0.42)

    # A simulator that ignored gas entirely would report 500 - 400 = 100.00.
    assert broker.realized_pnl_usd == pytest.approx(100.00 - 0.42)


# ---------------------------------------------------------------------------
# The same round trip on the degraded path, where the fee is real
# ---------------------------------------------------------------------------
#
#   A degraded quote is a DexScreener mid plus a slippage assumption. A mid has
#   no fee baked into it, so the pool fee is charged explicitly:
#
#   BUY  $400 @ 0.00002, pool fee 0.25%, gas 0.21
#     pool fee   = 400 * 0.25 / 100                            =   1.00
#     cash       = 1000 - 400 - 1.00 - 0.21                    = 598.79
#     cost basis = 400 + 1.00 + 0.21                           = 401.21
#   SELL all @ 0.000025
#     pool fee   = 500 * 0.25 / 100                            =   1.25
#     realized   = 500.00 - 401.21 - 1.25 - 0.21               =  97.33
#     cash       = 598.79 + 500.00 - 1.25 - 0.21               = 1097.33
# ---------------------------------------------------------------------------


def test_degraded_round_trip_charges_the_pool_fee(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    broker = LocalPaperBroker(cfg, rng=never_fails())

    buy = broker.place_order("BONK", Side.BUY, 400.0, quote=quote(degraded=True))
    assert buy.pool_fee_usd == pytest.approx(1.00)
    assert buy.degraded is True
    assert broker.cash_usd == pytest.approx(598.79)
    assert broker.get_positions()["BONK"].cost_basis_usd == pytest.approx(401.21)

    sell = broker.place_order(
        "BONK",
        Side.SELL,
        500.0,
        quote=quote(
            side=Side.SELL, usd_notional=500.0, price_usd=0.000025, degraded=True
        ),
    )
    assert sell.pool_fee_usd == pytest.approx(1.25)
    assert sell.realized_pnl_usd == pytest.approx(97.33)
    assert broker.cash_usd == pytest.approx(1097.33)
    assert broker.fees_paid_usd == pytest.approx(2.25)
    assert broker.gas_paid_usd == pytest.approx(0.42)


def test_the_fee_branch_is_decided_by_the_quote_not_the_config(
    tmp_path: Path,
) -> None:
    """Same notional, same pool_fee_pct, same config — only ``degraded`` differs,
    and it is worth exactly the fee."""
    routed = LocalPaperBroker(make_cfg(tmp_path / "a"), rng=never_fails())
    degraded = LocalPaperBroker(make_cfg(tmp_path / "b"), rng=never_fails())

    routed.place_order("BONK", Side.BUY, 400.0, quote=quote(degraded=False))
    degraded.place_order("BONK", Side.BUY, 400.0, quote=quote(degraded=True))

    assert routed.cash_usd - degraded.cash_usd == pytest.approx(1.00)
    assert routed.fees_paid_usd == 0.0
    assert degraded.fees_paid_usd == pytest.approx(1.00)


def test_cost_basis_includes_costs_not_just_notional(tmp_path: Path) -> None:
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    broker.place_order("BONK", Side.BUY, 400.0, quote=quote())

    pos = broker.get_positions()["BONK"]
    assert pos.cost_basis_usd > 400.0
    assert pos.cost_basis_usd == pytest.approx(400.21)
    # Marking at the entry price is therefore a small *loss*, which is correct:
    # you are down the gas the moment you are filled.
    assert pos.unrealized_pnl_usd(0.00002) == pytest.approx(-0.21)


# ---------------------------------------------------------------------------
# Partial sell proportionality
# ---------------------------------------------------------------------------
#
#   After the routed BUY above: quantity 20,000,000, basis 400.21, cash 599.79.
#   SELL $250 @ 0.000025  ->  quantity sold = 250 / 0.000025 = 10,000,000
#     fraction     = 10,000,000 / 20,000,000                  = 0.5
#     basis share  = 400.21 * 0.5                             = 200.105
#     gas                                                     =   0.21
#     realized     = 250 - 200.105 - 0.21                     =  49.685
#     cash         = 599.79 + 250 - 0.21                      = 849.58
#     remaining    : quantity 10,000,000, cost basis 200.105
# ---------------------------------------------------------------------------


def test_partial_sell_is_proportional(tmp_path: Path) -> None:
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    broker.place_order("BONK", Side.BUY, 400.0, quote=quote())

    sell = broker.place_order(
        "BONK",
        Side.SELL,
        250.0,
        quote=quote(side=Side.SELL, usd_notional=250.0, price_usd=0.000025),
    )

    assert sell.quantity == pytest.approx(10_000_000.0)
    assert sell.filled_usd == pytest.approx(250.0)
    assert sell.pool_fee_usd == 0.0
    assert sell.realized_pnl_usd == pytest.approx(49.685)
    assert sell.note is None  # no clamp

    pos = broker.get_positions()["BONK"]
    assert pos.quantity == pytest.approx(10_000_000.0)
    assert pos.cost_basis_usd == pytest.approx(200.105)
    # Average entry price is untouched by a partial exit.
    assert pos.avg_entry_price_usd == pytest.approx(0.00002)

    assert broker.cash_usd == pytest.approx(849.58)


def test_partial_sell_is_proportional_on_the_degraded_path(tmp_path: Path) -> None:
    #   basis 401.21, half of it is 200.605
    #   pool fee = 250 * 0.25 / 100 = 0.625
    #   realized = 250 - 200.605 - 0.625 - 0.21 = 48.56
    #   cash     = 598.79 + 250 - 0.625 - 0.21  = 847.955
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    broker.place_order("BONK", Side.BUY, 400.0, quote=quote(degraded=True))

    sell = broker.place_order(
        "BONK",
        Side.SELL,
        250.0,
        quote=quote(
            side=Side.SELL, usd_notional=250.0, price_usd=0.000025, degraded=True
        ),
    )

    assert sell.pool_fee_usd == pytest.approx(0.625)
    assert sell.realized_pnl_usd == pytest.approx(48.56)
    assert broker.get_positions()["BONK"].cost_basis_usd == pytest.approx(200.605)
    assert broker.cash_usd == pytest.approx(847.955)


def test_two_partial_sells_equal_one_full_sell(tmp_path: Path) -> None:
    """Halving twice must land exactly where selling once lands, minus the
    extra lot of gas."""
    cfg = make_cfg(tmp_path)
    broker = LocalPaperBroker(cfg, rng=never_fails())
    broker.place_order("BONK", Side.BUY, 400.0, quote=quote())

    q = quote(side=Side.SELL, usd_notional=250.0, price_usd=0.000025)
    a = broker.place_order("BONK", Side.SELL, 250.0, quote=q)
    b = broker.place_order("BONK", Side.SELL, 250.0, quote=q)

    # One-shot exit realizes 99.58 with one lot of gas; two exits pay gas twice.
    assert a.realized_pnl_usd + b.realized_pnl_usd == pytest.approx(99.58 - 0.21)
    assert broker.get_positions() == {}


# ---------------------------------------------------------------------------
# Over-sell clamps, never goes short
# ---------------------------------------------------------------------------


def test_oversell_clamps_to_position_and_never_goes_short(tmp_path: Path) -> None:
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    broker.place_order("BONK", Side.BUY, 400.0, quote=quote())

    # Position is worth 500 at this price; ask to sell 5,000.
    sell = broker.place_order(
        "BONK",
        Side.SELL,
        5_000.0,
        quote=quote(side=Side.SELL, usd_notional=5_000.0, price_usd=0.000025),
    )

    assert sell.requested_usd == pytest.approx(5_000.0)
    assert sell.filled_usd == pytest.approx(500.0)
    assert sell.quantity == pytest.approx(20_000_000.0)
    assert sell.note is not None and "clamp" in sell.note.lower()
    assert sell.realized_pnl_usd == pytest.approx(99.58)

    assert broker.get_positions() == {}
    assert broker.cash_usd == pytest.approx(1099.58)
    assert broker.cash_usd > 0


def test_oversell_fee_is_charged_on_the_clamped_size(tmp_path: Path) -> None:
    """On the degraded path the fee must follow what actually traded ($500),
    not the fantasy notional that was asked for ($5,000)."""
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    broker.place_order("BONK", Side.BUY, 400.0, quote=quote(degraded=True))

    sell = broker.place_order(
        "BONK",
        Side.SELL,
        5_000.0,
        quote=quote(
            side=Side.SELL, usd_notional=5_000.0, price_usd=0.000025, degraded=True
        ),
    )

    assert sell.filled_usd == pytest.approx(500.0)
    assert sell.pool_fee_usd == pytest.approx(1.25)  # not 12.50
    assert sell.realized_pnl_usd == pytest.approx(97.33)


def test_position_is_deleted_on_full_exit_despite_float_dust(tmp_path: Path) -> None:
    """A price with no exact binary representation must still close cleanly."""
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    broker.place_order(
        "WIF", Side.BUY, 333.33, quote=quote(symbol="WIF", price_usd=1.7e-05)
    )
    pos = broker.get_positions()["WIF"]

    broker.place_order(
        "WIF",
        Side.SELL,
        pos.quantity * 1.9e-05,
        quote=quote(symbol="WIF", side=Side.SELL, price_usd=1.9e-05),
    )

    assert broker.get_positions() == {}


def test_sell_with_no_position_raises(tmp_path: Path) -> None:
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    with pytest.raises(NoPosition):
        broker.place_order(
            "BONK", Side.SELL, 100.0, quote=quote(side=Side.SELL)
        )
    # Nothing moved, not even gas — the swap was never attempted.
    assert broker.cash_usd == 1000.0
    assert broker.gas_paid_usd == 0.0


# ---------------------------------------------------------------------------
# Failed transactions
# ---------------------------------------------------------------------------


def test_failed_transaction_charges_gas_and_moves_nothing_else(
    tmp_path: Path,
) -> None:
    cfg = make_cfg(tmp_path, failed_tx_rate=1.0)
    broker = LocalPaperBroker(cfg, rng=random.Random(7))

    fill = broker.place_order("BONK", Side.BUY, 400.0, quote=quote())

    assert fill.failed is True
    assert fill.filled_usd == 0.0
    assert fill.quantity == 0.0
    assert fill.pool_fee_usd == 0.0  # the pool never executed
    assert fill.gas_usd == pytest.approx(0.21)  # but the validator still ate
    assert fill.realized_pnl_usd == 0.0

    assert broker.cash_usd == pytest.approx(999.79)
    assert broker.get_positions() == {}
    assert broker.fees_paid_usd == 0.0
    assert broker.gas_paid_usd == pytest.approx(0.21)
    assert broker.realized_pnl_usd == 0.0


def test_failed_sell_leaves_the_position_intact(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, failed_tx_rate=0.0)
    broker = LocalPaperBroker(cfg, rng=random.Random(7))
    broker.place_order("BONK", Side.BUY, 400.0, quote=quote())
    before = broker.get_positions()["BONK"]
    cash_before = broker.cash_usd

    broker = LocalPaperBroker(
        replace(cfg, execution=replace(cfg.execution, failed_tx_rate=1.0)),
        rng=random.Random(7),
    )
    fill = broker.place_order(
        "BONK", Side.SELL, 500.0, quote=quote(side=Side.SELL, price_usd=0.000025)
    )

    assert fill.failed is True
    assert broker.get_positions()["BONK"] == before
    assert broker.cash_usd == pytest.approx(cash_before - 0.21)


def test_failure_draw_is_deterministic_for_a_seeded_rng(tmp_path: Path) -> None:
    """Same seed, same config, same sequence of outcomes — twice."""

    def run(tag: str) -> list[bool]:
        run_dir = tmp_path / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = make_cfg(run_dir, failed_tx_rate=0.5)
        broker = LocalPaperBroker(cfg, rng=random.Random(99))
        out = []
        for _ in range(12):
            fill = broker.place_order("BONK", Side.BUY, 10.0, quote=quote())
            out.append(fill.failed)
        return out

    first = run("a")
    second = run("b")
    assert first == second
    # A 50% rate over 12 draws must produce both outcomes, or the draw is dead.
    assert any(first) and not all(first)


# ---------------------------------------------------------------------------
# Insufficient cash
# ---------------------------------------------------------------------------


def test_buy_larger_than_cash_raises(tmp_path: Path) -> None:
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    with pytest.raises(InsufficientCash):
        broker.place_order("BONK", Side.BUY, 2_000.0, quote=quote())
    assert broker.cash_usd == 1000.0
    assert broker.gas_paid_usd == 0.0


def test_buy_of_exactly_cash_raises_because_of_gas(tmp_path: Path) -> None:
    """1000 notional needs 1000 + 0.21 gas. Cash is exactly 1000."""
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    with pytest.raises(InsufficientCash):
        broker.place_order("BONK", Side.BUY, 1_000.0, quote=quote())
    assert broker.cash_usd == 1000.0


def test_degraded_buy_reserves_the_pool_fee_too(tmp_path: Path) -> None:
    """998 * 1.0025 + 0.21 = 1000.705 > 1000, but 998 + 0.21 would have fit —
    so this raises only if the degraded fee is reserved."""
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    with pytest.raises(InsufficientCash):
        broker.place_order("BONK", Side.BUY, 998.0, quote=quote(degraded=True))
    # The same order on a routed quote fits, because there is no extra fee.
    broker.place_order("BONK", Side.BUY, 998.0, quote=quote())
    assert broker.cash_usd == pytest.approx(1.79)


def test_buy_just_inside_the_cash_limit_succeeds(tmp_path: Path) -> None:
    # 990 + 0.21 = 990.21 <= 1000
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    broker.place_order("BONK", Side.BUY, 990.0, quote=quote())
    assert broker.cash_usd == pytest.approx(9.79)


def test_non_positive_notional_is_rejected(tmp_path: Path) -> None:
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    with pytest.raises(ValueError):
        broker.place_order("BONK", Side.BUY, 0.0, quote=quote())
    with pytest.raises(ValueError):
        broker.place_order("BONK", Side.BUY, -5.0, quote=quote())


def test_non_positive_price_is_rejected(tmp_path: Path) -> None:
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    with pytest.raises(ValueError):
        broker.place_order("BONK", Side.BUY, 100.0, quote=quote(price_usd=0.0))


# ---------------------------------------------------------------------------
# Scale-in
# ---------------------------------------------------------------------------


def test_scale_in_averages_entry_price_and_sums_cost_basis(tmp_path: Path) -> None:
    #  BUY 100 @ 0.00002  -> 5,000,000 tokens, basis 100 + 0.21 = 100.21
    #  BUY 100 @ 0.00004  -> 2,500,000 tokens, basis 100 + 0.21 = 100.21
    #  total 7,500,000 tokens, basis 200.42
    #  avg entry = (100 + 100) / 7,500,000 = 2.6666...e-05
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    first = broker.place_order(
        "BONK", Side.BUY, 100.0, quote=quote(usd_notional=100.0, price_usd=0.00002)
    )
    broker.place_order(
        "BONK", Side.BUY, 100.0, quote=quote(usd_notional=100.0, price_usd=0.00004)
    )

    pos = broker.get_positions()["BONK"]
    assert pos.quantity == pytest.approx(7_500_000.0)
    assert pos.cost_basis_usd == pytest.approx(200.42)
    assert pos.avg_entry_price_usd == pytest.approx(200.0 / 7_500_000.0)
    # opened_at is the *first* entry, so age is measured from the original.
    assert pos.opened_at == first.ts


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_state_survives_save_and_load_byte_exactly(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    first = LocalPaperBroker(cfg, rng=never_fails())
    first.place_order("BONK", Side.BUY, 400.0, quote=quote())
    first.place_order(
        "WIF", Side.BUY, 123.45, quote=quote(symbol="WIF", price_usd=1.83)
    )
    first.place_order(
        "BONK",
        Side.SELL,
        137.0,
        quote=quote(side=Side.SELL, price_usd=0.0000237),
    )
    first.save()
    raw_first = cfg.state_path.read_bytes()

    second = LocalPaperBroker(cfg, rng=never_fails())
    assert second.cash_usd == first.cash_usd  # exact, not approx
    assert second.get_positions() == first.get_positions()
    assert second.realized_pnl_usd == first.realized_pnl_usd
    assert second.fees_paid_usd == first.fees_paid_usd
    assert second.gas_paid_usd == first.gas_paid_usd
    assert second.starting_cash_usd == first.starting_cash_usd

    second.save()
    assert cfg.state_path.read_bytes() == raw_first


def test_state_file_carries_a_schema_version(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    broker = LocalPaperBroker(cfg, rng=never_fails())
    broker.save()
    payload = json.loads(cfg.state_path.read_text(encoding="utf-8"))
    assert isinstance(payload["schema_version"], int)
    assert payload["schema_version"] >= 1


def test_a_state_file_from_the_future_is_refused(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    cfg.state_path.write_text(
        json.dumps({"schema_version": 9_999, "cash_usd": 1.0}), encoding="utf-8"
    )
    with pytest.raises(Exception) as excinfo:
        LocalPaperBroker(cfg, rng=never_fails())
    assert "schema" in str(excinfo.value).lower()


def test_a_corrupt_state_file_is_refused_not_silently_reset(tmp_path: Path) -> None:
    """Resetting to starting cash would quietly erase the entire P&L history."""
    cfg = make_cfg(tmp_path)
    cfg.state_path.write_text("{ truncated", encoding="utf-8")
    with pytest.raises(Exception):
        LocalPaperBroker(cfg, rng=never_fails())


def test_save_is_atomic_and_leaves_no_debris(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    broker = LocalPaperBroker(cfg, rng=never_fails())
    broker.place_order("BONK", Side.BUY, 400.0, quote=quote())
    broker.save()

    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "state.json"]
    assert leftovers == ["trades.jsonl"]


def test_every_attempt_appends_one_row_to_trades_jsonl(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, failed_tx_rate=1.0)
    broker = LocalPaperBroker(cfg, rng=random.Random(3))
    broker.place_order("BONK", Side.BUY, 50.0, quote=quote(usd_notional=50.0))
    broker.place_order("BONK", Side.BUY, 50.0, quote=quote(usd_notional=50.0))

    rows = [
        json.loads(line)
        for line in cfg.trades_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert all(row["failed"] is True for row in rows)
    assert all(row["side"] == "BUY" for row in rows)
    assert all(row["gas_usd"] == pytest.approx(0.21) for row in rows)


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------
#
# The ledger identity. Every buy moves money from cash into cost basis one for
# one; every sell moves it back and books the difference as realized P&L; every
# failed transaction burns gas and nothing else. Therefore, at all times:
#
#     cash + sum(cost_basis) == starting_cash + realized_pnl - failed_tx_gas
#
# If any fee, gas charge or proportional split is wrong anywhere in broker.py,
# this equation stops balancing.
# ---------------------------------------------------------------------------


def _drive(broker: LocalPaperBroker, seed: int, steps: int = 60) -> None:
    rng = random.Random(seed)
    prices = {"BONK": 0.00002, "WIF": 1.83, "POPCAT": 0.41}
    fees = {"BONK": 0.25, "WIF": 0.30, "POPCAT": 0.55}
    for _ in range(steps):
        symbol = rng.choice(list(prices))
        prices[symbol] *= rng.uniform(0.85, 1.2)
        side = Side.BUY if rng.random() < 0.55 else Side.SELL
        notional = round(rng.uniform(5.0, 300.0), 2)
        q = quote(
            symbol=symbol,
            side=side,
            usd_notional=notional,
            price_usd=prices[symbol],
            pool_fee_pct=fees[symbol],
            # Mix routed and degraded quotes so the identity has to hold across
            # both fee branches, not just the zero-fee one.
            degraded=rng.random() < 0.3,
        )
        try:
            broker.place_order(symbol, side, notional, quote=q)
        except (InsufficientCash, NoPosition):
            continue


def _assert_balances(broker: LocalPaperBroker, failed_gas: float) -> None:
    basis = sum(p.cost_basis_usd for p in broker.get_positions().values())
    left = broker.cash_usd + basis
    right = broker.starting_cash_usd + broker.realized_pnl_usd - failed_gas
    assert left == pytest.approx(right, abs=1e-6)


def _failed_gas(cfg: config.Config) -> float:
    total = 0.0
    for line in cfg.trades_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["failed"]:
                total += row["gas_usd"]
    return total


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 17, 404])
def test_conservation_without_failures(tmp_path: Path, seed: int) -> None:
    cfg = make_cfg(tmp_path, failed_tx_rate=0.0)
    broker = LocalPaperBroker(cfg, rng=random.Random(seed))
    _drive(broker, seed)

    _assert_balances(broker, failed_gas=0.0)
    assert all(p.quantity > 0 for p in broker.get_positions().values())
    assert broker.cash_usd >= 0.0
    # Both fee branches were exercised: some fills paid a pool fee, some did not.
    assert broker.fees_paid_usd > 0.0


@pytest.mark.parametrize("seed", [5, 11, 23])
def test_conservation_with_failures(tmp_path: Path, seed: int) -> None:
    cfg = make_cfg(tmp_path, failed_tx_rate=0.2)
    broker = LocalPaperBroker(cfg, rng=random.Random(seed))
    _drive(broker, seed)

    _assert_balances(broker, failed_gas=_failed_gas(cfg))
    # Failures happened, or the test is not exercising what it claims to.
    assert _failed_gas(cfg) > 0.0


def test_trades_jsonl_replays_to_the_same_cash(tmp_path: Path) -> None:
    """Rebuild cash from the trade log alone and compare to the state file.

    This catches a state/ledger divergence that the balance identity above,
    which reads only the state, could not see.
    """
    cfg = make_cfg(tmp_path, failed_tx_rate=0.15)
    broker = LocalPaperBroker(cfg, rng=random.Random(2024))
    _drive(broker, 2024)

    cash = cfg.starting_cash_usd
    fees = gas = 0.0
    for line in cfg.trades_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        gas += row["gas_usd"]
        fees += row["pool_fee_usd"]
        cash -= row["gas_usd"] + row["pool_fee_usd"]
        if row["failed"]:
            continue
        cash += row["filled_usd"] if row["side"] == "SELL" else -row["filled_usd"]

    assert cash == pytest.approx(broker.cash_usd, abs=1e-6)
    assert fees == pytest.approx(broker.fees_paid_usd, abs=1e-9)
    assert gas == pytest.approx(broker.gas_paid_usd, abs=1e-9)


def test_realized_pnl_equals_the_sum_of_sell_fills(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, failed_tx_rate=0.1)
    broker = LocalPaperBroker(cfg, rng=random.Random(8))
    _drive(broker, 8)

    booked = 0.0
    for line in cfg.trades_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            booked += json.loads(line)["realized_pnl_usd"]

    assert booked == pytest.approx(broker.realized_pnl_usd, abs=1e-9)
