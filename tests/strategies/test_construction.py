"""Tests for ``memetrader.strategies.construction``.

Offline, no history/ dependency. Property tests use ``hypothesis`` to fuzz the
cash/exposure-cap invariants that :func:`size_orders` asserts internally —
these tests exist so a regression that weakens or removes an internal assert
is still caught from the outside.
"""

from __future__ import annotations

from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from memetrader.strategies.construction import ConstructionLimits, size_orders
from memetrader.types import OrderIntent, Position, Side


def _order(symbol: str, side: Side, amount: int, *, tag: str = "") -> OrderIntent:
    return OrderIntent(
        intent_id=f"i-{symbol}-{side.value}-{amount}-{tag}",
        decision_id=None,
        action_id=None,
        run_id="r",
        ts=0.0,
        symbol=symbol,
        side=side,
        in_amount_atomic=amount,
        max_in_amount_atomic=amount,
        source="strategy",
    )


def _buy(symbol: str, amount: int, *, tag: str = "") -> OrderIntent:
    return _order(symbol, Side.BUY, amount, tag=tag)


def _sell(symbol: str, amount: int, *, tag: str = "") -> OrderIntent:
    return _order(symbol, Side.SELL, amount, tag=tag)


def _position(symbol: str, quantity_atomic: int) -> Position:
    return Position(
        symbol=symbol,
        mint=symbol,
        quantity_atomic=quantity_atomic,
        decimals=6,
        avg_entry_price_usd=1.0,
        opened_at=0.0,
        cost_basis_usd=1.0,
    )


# ---------------------------------------------------------------------------
# Cash safety (hypothesis)
# ---------------------------------------------------------------------------


@given(
    cash=st.integers(min_value=0, max_value=10**12),
    portfolio_value=st.integers(min_value=0, max_value=10**12),
    amounts=st.lists(st.integers(min_value=0, max_value=10**11), min_size=1, max_size=6),
)
def test_size_orders_never_exceeds_available_cash(cash, portfolio_value, amounts) -> None:
    intents = [_buy(f"SYM{i}", amt, tag=str(i)) for i, amt in enumerate(amounts)]
    universe = frozenset(i.symbol for i in intents)
    out = size_orders(
        intents,
        cash_micro_usd=cash,
        portfolio_value_micro_usd=portfolio_value,
        existing_exposure_micro_usd={},
        positions={},
        universe=universe,
        limits=ConstructionLimits(per_asset_cap_bps=10_000, portfolio_cap_bps=10_000),
    )
    assert sum(o.in_amount_atomic for o in out) <= cash


# ---------------------------------------------------------------------------
# Exposure caps (hypothesis)
# ---------------------------------------------------------------------------


@given(
    cash=st.integers(min_value=0, max_value=10**10),
    portfolio_value=st.integers(min_value=0, max_value=10**10),
    per_asset_bps=st.integers(min_value=1, max_value=10_000),
    portfolio_bps=st.integers(min_value=1, max_value=10_000),
    amounts=st.dictionaries(
        st.sampled_from(["AAA", "BBB", "CCC"]),
        st.integers(min_value=0, max_value=10**10),
        min_size=1,
    ),
    existing=st.dictionaries(
        st.sampled_from(["AAA", "BBB", "CCC"]), st.integers(min_value=0, max_value=10**10)
    ),
)
def test_caps_and_cash_hold_under_fuzzing(
    cash, portfolio_value, per_asset_bps, portfolio_bps, amounts, existing
) -> None:
    intents = [_buy(sym, amt) for sym, amt in amounts.items()]
    universe = frozenset(amounts.keys())
    limits = ConstructionLimits(
        per_asset_cap_bps=per_asset_bps, portfolio_cap_bps=portfolio_bps
    )

    out = size_orders(
        intents,
        cash_micro_usd=cash,
        portfolio_value_micro_usd=portfolio_value,
        existing_exposure_micro_usd=existing,
        positions={},
        universe=universe,
        limits=limits,
    )

    assert sum(o.in_amount_atomic for o in out) <= cash

    # Headroom-based, not total-based: existing exposure can already sit at
    # or above a cap from pure price movement, which sizing cannot undo.
    # What construction must guarantee is that it adds nothing beyond
    # remaining headroom.
    asset_cap = (portfolio_value * per_asset_bps) // 10_000
    for o in out:
        asset_headroom = max(0, asset_cap - existing.get(o.symbol, 0))
        assert o.in_amount_atomic <= asset_headroom

    portfolio_cap = (portfolio_value * portfolio_bps) // 10_000
    portfolio_headroom = max(0, portfolio_cap - sum(existing.values()))
    assert sum(o.in_amount_atomic for o in out) <= portfolio_headroom


def test_per_asset_cap_holds_concrete() -> None:
    out = size_orders(
        [_buy("AAA", 10_000_000_000)],
        cash_micro_usd=10_000_000_000,
        portfolio_value_micro_usd=1_000_000_000,
        existing_exposure_micro_usd={},
        positions={},
        universe=frozenset({"AAA"}),
        limits=ConstructionLimits(per_asset_cap_bps=2_000, portfolio_cap_bps=10_000),
    )
    assert len(out) == 1
    assert out[0].in_amount_atomic <= 200_000_000  # 20% of 1e9


def test_portfolio_cap_holds_concrete() -> None:
    out = size_orders(
        [_buy("AAA", 5_000_000_000), _buy("BBB", 5_000_000_000)],
        cash_micro_usd=10_000_000_000,
        portfolio_value_micro_usd=1_000_000_000,
        existing_exposure_micro_usd={},
        positions={},
        universe=frozenset({"AAA", "BBB"}),
        limits=ConstructionLimits(per_asset_cap_bps=10_000, portfolio_cap_bps=5_000),
    )
    assert sum(o.in_amount_atomic for o in out) <= 500_000_000  # 50% of 1e9


# ---------------------------------------------------------------------------
# Universe shrinkage
# ---------------------------------------------------------------------------


def test_buy_outside_universe_is_dropped_silently() -> None:
    out = size_orders(
        [_buy("DEAD", 1_000_000)],
        cash_micro_usd=10**9,
        portfolio_value_micro_usd=10**9,
        existing_exposure_micro_usd={},
        positions={},
        universe=frozenset(),
        limits=ConstructionLimits(),
    )
    assert out == ()


def test_sell_survives_universe_departure_and_is_clamped_to_holding() -> None:
    position = _position("DEAD", 500)
    out = size_orders(
        [_sell("DEAD", 10_000)],  # claims more than is actually held
        cash_micro_usd=0,
        portfolio_value_micro_usd=0,
        existing_exposure_micro_usd={},
        positions={"DEAD": position},
        universe=frozenset(),  # DEAD has left the universe
        limits=ConstructionLimits(),
    )
    assert len(out) == 1
    assert out[0].side is Side.SELL
    assert out[0].symbol == "DEAD"
    assert out[0].in_amount_atomic == 500


def test_sell_of_unheld_symbol_is_dropped_without_crash() -> None:
    out = size_orders(
        [_sell("GHOST", 1_000)],
        cash_micro_usd=0,
        portfolio_value_micro_usd=0,
        existing_exposure_micro_usd={},
        positions={},
        universe=frozenset(),
        limits=ConstructionLimits(),
    )
    assert out == ()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_size_orders_is_deterministic() -> None:
    intents = [_buy("AAA", 123_456), _buy("BBB", 654_321)]
    kwargs: dict[str, Any] = {
        "cash_micro_usd": 500_000,
        "portfolio_value_micro_usd": 10**7,
        "existing_exposure_micro_usd": {},
        "positions": {},
        "universe": frozenset({"AAA", "BBB"}),
        "limits": ConstructionLimits(),
    }
    out_1 = size_orders(intents, **kwargs)
    out_2 = size_orders(intents, **kwargs)
    assert out_1 == out_2


def test_size_orders_orders_sells_before_buys_and_buys_sorted() -> None:
    position = _position("ZZZ", 10)
    out = size_orders(
        [_buy("BBB", 1_000), _buy("AAA", 1_000), _sell("ZZZ", 10)],
        cash_micro_usd=10**9,
        portfolio_value_micro_usd=10**9,
        existing_exposure_micro_usd={},
        positions={"ZZZ": position},
        universe=frozenset({"AAA", "BBB"}),
        limits=ConstructionLimits(),
    )
    sides = [o.side for o in out]
    assert sides == [Side.SELL, Side.BUY, Side.BUY]
    buy_symbols = [o.symbol for o in out if o.side is Side.BUY]
    assert buy_symbols == sorted(buy_symbols)
