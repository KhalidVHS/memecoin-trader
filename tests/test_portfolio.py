"""Marking the book, the mark hierarchy, and the one stop-price definition.

Rewritten against audit C8. What changed, and why:

**Removed.** ``test_missing_mark_does_not_crash_and_carries_at_cost`` and
``test_unusable_mark_is_treated_as_missing`` asserted the exact defect C8
identified: a position that could not be priced was carried at cost basis, which
displays zero unrealised loss during a rug and during a data outage, and so
guaranteed the stop could not fire when it was most needed. Both are replaced by
tests asserting the opposite — an unmarkable position makes ``total_value_usd``
``None`` and that ``None`` propagates.

``test_stop_loss_on_an_unmarked_position_does_not_fire`` kept its *assertion* but
lost its *reason*: it used to pass because the position had been overwritten
with its cost basis and therefore looked flat. It is now a deliberate decision
with its own test and its own docstring.

``test_now_defaults_to_wall_clock`` is removed. ``now`` is a required argument:
a pure function of state does not get to read a clock.

Everything else that still had meaning was kept — the cost-basis-not-notional
P&L test, ``total_return_pct``, the fee/gas pass-through, worst-first ordering,
the fraction-versus-whole-percent boundary, and the exactly-at-threshold stop.
They no longer go through ``LocalPaperBroker``: ``mark_book`` takes plain values,
so these tests construct positions directly and need no broker, no files and no
clock.
"""

from __future__ import annotations

import math

import pytest

from memetrader.portfolio import (
    MarkParams,
    build_mark,
    failed_attempts,
    landed_exits,
    mark_book,
    stop_loss_breaches,
    stop_price_usd,
)
from memetrader.types import (
    DataQuality,
    Fill,
    Mark,
    OrderState,
    Position,
    Provenance,
    Quote,
    Side,
    TokenMeta,
    ValidationError,
    ValuationEstimate,
)

NOW = 1_700_000_000.0
BONK_MINT = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

USDC = TokenMeta(mint=USDC_MINT, decimals=6, source="test")
BONK = TokenMeta(mint=BONK_MINT, decimals=5, source="test")

PARAMS = MarkParams()


# ---------------------------------------------------------------------------
# Fixtures, built by hand so nothing here depends on the broker
# ---------------------------------------------------------------------------


def position(
    symbol: str = "BONK",
    *,
    quantity: float = 20_000_000.0,
    decimals: int = 5,
    avg_entry_price_usd: float = 0.00002,
    cost_basis_usd: float = 400.21,
) -> Position:
    """The project's canonical fixture: $400 of BONK at 0.00002, plus $0.21 gas.

    Cost basis 400.21 against a 400.00 notional is exactly the 0.05% gap that
    audit §11 found between the engine's stop and the displayed one.
    """
    return Position(
        symbol=symbol,
        mint=BONK_MINT,
        quantity_atomic=int(quantity * 10**decimals),
        decimals=decimals,
        avg_entry_price_usd=avg_entry_price_usd,
        opened_at=NOW - 600.0,
        cost_basis_usd=cost_basis_usd,
    )


def route_quote(
    *,
    symbol: str = "BONK",
    price_usd: float = 0.000025,
    quantity: float = 20_000_000.0,
    received_at: float = NOW,
    expires_at: float | None = None,
) -> Quote:
    """A SELL route: ``quantity`` BONK in, dollars out. Prices liquidation."""
    in_atomic = int(quantity * 10**BONK.decimals)
    out_atomic = int(quantity * price_usd * 10**USDC.decimals)
    return Quote(
        symbol=symbol,
        side=Side.SELL,
        input_token=BONK,
        output_token=USDC,
        in_amount_atomic=in_atomic,
        out_amount_atomic=out_atomic,
        min_out_amount_atomic=int(out_atomic * 0.99),
        price_impact_pct=0.4,
        route_labels=("Raydium",),
        fingerprint="fp-route",
        requested_at=received_at - 0.2,
        received_at=received_at,
        expires_at=expires_at,
    )


def fill(
    *,
    symbol: str = "BONK",
    side: Side = Side.SELL,
    state: OrderState = OrderState.LANDED,
    ts: float = NOW,
) -> Fill:
    landed = state in (OrderState.LANDED, OrderState.RECONCILED)
    return Fill(
        fill_id="fil-1",
        order_id="ord-1",
        intent_id="int-1",
        decision_id="dec-1",
        ts=ts,
        symbol=symbol,
        side=side,
        state=state,
        in_amount_atomic=1_000 if landed else 0,
        out_amount_atomic=1_000 if landed else 0,
        token_amount_atomic=1_000 if landed else 0,
        token_decimals=5,
        quote_fingerprint="fp",
        price_usd=0.00002 if landed else None,
        notional_usd=200.0 if landed else 0.0,
        price_impact_pct=0.4,
        pool_fee_usd=0.0,
        gas_usd=0.21,
    )


def book(
    *,
    cash_usd: float = 599.79,
    positions: dict[str, Position] | None = None,
    marks: dict[str, Mark] | None = None,
    realized_pnl_usd: float = 0.0,
    starting_cash_usd: float = 1000.0,
    **kw: float,
):
    positions = {"BONK": position()} if positions is None else positions
    return mark_book(
        cash_usd=cash_usd,
        positions=positions,
        marks=marks or {},
        realized_pnl_usd=realized_pnl_usd,
        starting_cash_usd=starting_cash_usd,
        now=NOW,
        **kw,
    )


def mark_at(price: float, *, basis: str = "route", symbol: str = "BONK") -> Mark:
    return Mark(
        symbol=symbol,
        price_usd=price,
        basis=basis,  # type: ignore[arg-type]
        provenance=Provenance(source="test", receive_time=NOW),
    )


# ---------------------------------------------------------------------------
# build_mark — the hierarchy
# ---------------------------------------------------------------------------


def test_a_route_is_preferred_and_carries_no_haircut() -> None:
    m = build_mark(
        "BONK", route=route_quote(), mid_price_usd=0.00009, params=PARAMS, now=NOW
    )
    assert m.basis == "route"
    assert m.haircut_pct == 0.0
    assert m.is_executable_basis
    assert m.price_usd == pytest.approx(0.000025, rel=1e-6)


def test_a_mid_is_used_when_no_route_exists_and_is_haircut() -> None:
    m = build_mark(
        "BONK",
        mid_price_usd=0.00002,
        mid_provenance=Provenance(source="dexscreener", receive_time=NOW),
        params=PARAMS,
        now=NOW,
    )
    assert m.basis == "mid"
    assert m.haircut_pct == PARAMS.mid_haircut_pct
    assert not m.is_executable_basis
    assert m.price_usd == pytest.approx(0.00002 * 0.98)


def test_a_valuation_estimate_is_the_last_resort_and_never_undercuts_its_own_haircut() -> (
    None
):
    estimate = ValuationEstimate(
        symbol="BONK",
        mid_price_usd=0.00002,
        haircut_pct=1.0,  # more optimistic than the floor
        reason="jupiter unavailable",
        at=NOW,
        source="dexscreener",
    )
    m = build_mark("BONK", estimate=estimate, params=PARAMS, now=NOW)
    assert m.basis == "estimate"
    assert m.haircut_pct == PARAMS.estimate_min_haircut_pct
    assert m.price_usd == pytest.approx(0.00002 * 0.95)


def test_a_degraded_mark_can_never_look_better_than_the_raw_mid() -> None:
    """An outage that improves reported P&L is an outage that reads as a signal."""
    raw = 0.00002
    mid = build_mark(
        "BONK",
        mid_price_usd=raw,
        mid_provenance=Provenance(source="dex", receive_time=NOW),
        params=PARAMS,
        now=NOW,
    )
    est = build_mark(
        "BONK",
        estimate=ValuationEstimate(
            symbol="BONK",
            mid_price_usd=raw,
            haircut_pct=0.0,
            reason="no route",
            at=NOW,
            source="dex",
        ),
        params=PARAMS,
        now=NOW,
    )
    assert mid.price_usd is not None and mid.price_usd < raw
    assert est.price_usd is not None and est.price_usd < mid.price_usd


def test_nothing_at_all_is_unavailable_not_zero_and_not_cost() -> None:
    m = build_mark("BONK", params=PARAMS, now=NOW)
    assert m.basis == "unavailable"
    assert m.price_usd is None
    assert not m.usable


def test_a_stale_route_is_not_a_mark() -> None:
    old = route_quote(received_at=NOW - PARAMS.max_mark_age_seconds - 1.0)
    m = build_mark("BONK", route=old, params=PARAMS, now=NOW)
    assert m.basis == "unavailable"


def test_an_expired_route_is_not_a_mark() -> None:
    m = build_mark("BONK", route=route_quote(expires_at=NOW - 1.0), params=PARAMS, now=NOW)
    assert m.basis == "unavailable"


def test_a_stale_mid_is_not_a_mark() -> None:
    prov = Provenance(
        source="dex", receive_time=NOW, event_time=NOW - PARAMS.max_mark_age_seconds - 1.0
    )
    m = build_mark(
        "BONK", mid_price_usd=0.00002, mid_provenance=prov, params=PARAMS, now=NOW
    )
    assert m.basis == "unavailable"


def test_a_quarantined_source_is_not_a_mark() -> None:
    prov = Provenance(
        source="dex",
        receive_time=NOW,
        quality=DataQuality.QUARANTINED,
        quality_reason="sequence gap",
    )
    m = build_mark(
        "BONK", mid_price_usd=0.00002, mid_provenance=prov, params=PARAMS, now=NOW
    )
    assert m.basis == "unavailable"


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_non_positive_mid_is_not_a_price(bad: float) -> None:
    m = build_mark(
        "BONK",
        mid_price_usd=bad,
        mid_provenance=Provenance(source="dex", receive_time=NOW),
        params=PARAMS,
        now=NOW,
    )
    assert m.basis == "unavailable"


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_mid_is_rejected_at_construction(bad: float) -> None:
    """NaN compares false against every bound, including ``<= 0``, so it cannot
    be caught by a range check downstream. It has to be refused at the type."""
    with pytest.raises(ValidationError):
        Mark(symbol="BONK", price_usd=bad, basis="mid", provenance=None)


def test_mark_params_reject_a_fraction_where_a_whole_percent_belongs() -> None:
    with pytest.raises(ValidationError):
        MarkParams(mid_haircut_pct=150.0)


# ---------------------------------------------------------------------------
# mark_book — C8's central invariant
# ---------------------------------------------------------------------------


def test_empty_book_marks_to_cash() -> None:
    state = book(cash_usd=1000.0, positions={})
    assert state.ts == NOW
    assert state.cash_usd == 1000.0
    assert state.total_value_usd == 1000.0
    assert state.positions == {}
    assert state.position_values_usd == {}
    assert state.unrealized_pnl_usd == 0.0
    assert state.total_return_pct == 0.0
    assert state.unmarkable == ()
    assert state.fully_marked


def test_total_value_is_cash_plus_marks() -> None:
    state = book(marks={"BONK": mark_at(0.000025)})
    assert state.position_values_usd["BONK"] == pytest.approx(500.0)
    assert state.total_value_usd == pytest.approx(1099.79)


def test_unrealized_pnl_is_against_cost_basis_not_notional() -> None:
    """500.00 marked against a 400.21 basis. A simulator using the 400 notional
    would overstate this by the 0.21 of entry gas."""
    state = book(marks={"BONK": mark_at(0.000025)})
    assert state.unrealized_pnl_usd == pytest.approx(99.79)


def test_total_return_pct() -> None:
    up = book(marks={"BONK": mark_at(0.000025)})
    assert up.total_return_pct == pytest.approx(9.979)
    down = book(marks={"BONK": mark_at(0.00001)})
    assert down.total_value_usd == pytest.approx(799.79)
    assert down.total_return_pct == pytest.approx(-20.021)


def test_fees_gas_and_realized_are_carried_through() -> None:
    state = book(
        marks={"BONK": mark_at(0.000025)},
        realized_pnl_usd=49.685,
        fees_paid_usd=0.0,
        gas_paid_usd=0.42,
    )
    assert state.fees_paid_usd == 0.0
    assert state.gas_paid_usd == pytest.approx(0.42)
    assert state.realized_pnl_usd == pytest.approx(49.685)


def test_an_unmarkable_position_makes_total_value_none_and_is_not_flat_pnl() -> None:
    """Audit C8, the whole point. The old code returned 599.79 + 400.21 here and
    reported 0.00 unrealised P&L, which is the number a rug produces and the
    number a data outage produces, and they are not the same event as 'flat'."""
    state = book(marks={})

    assert state.total_value_usd is None
    assert state.unrealized_pnl_usd is None
    assert state.position_values_usd["BONK"] is None
    assert state.unmarkable == ("BONK",)
    assert not state.fully_marked
    # Specifically: it is not cost basis, and it is not zero P&L.
    assert state.position_values_usd["BONK"] != pytest.approx(400.21)
    assert state.unrealized_pnl_usd != 0.0


def test_the_none_propagates_to_every_downstream_percentage() -> None:
    state = book(marks={})
    assert state.total_return_pct is None
    assert state.gross_exposure_usd is None
    assert state.gross_exposure_pct is None


def test_one_unmarkable_position_poisons_the_whole_total() -> None:
    """A book whose value is partly unknown is a book whose value is unknown."""
    state = book(
        positions={"BONK": position(), "WIF": position("WIF", quantity=100.0, decimals=6)},
        marks={"BONK": mark_at(0.000025)},
    )
    assert state.position_values_usd["BONK"] == pytest.approx(500.0)
    assert state.total_value_usd is None
    assert state.unmarkable == ("WIF",)


def test_an_explicitly_unavailable_mark_counts_as_unmarkable() -> None:
    state = book(
        marks={
            "BONK": Mark(
                symbol="BONK", price_usd=None, basis="unavailable", provenance=None
            )
        }
    )
    assert state.unmarkable == ("BONK",)
    assert state.total_value_usd is None


def test_a_zero_quantity_position_does_not_poison_the_book() -> None:
    """Ledger residue is not exposure, and halting entries over an accounting
    artefact would be a self-inflicted outage."""
    state = book(
        positions={"BONK": position(quantity=0.0, cost_basis_usd=0.0)},
        marks={},
        cash_usd=1000.0,
    )
    assert state.unmarkable == ()
    assert state.total_value_usd == pytest.approx(1000.0)


def test_marks_for_unheld_symbols_are_ignored() -> None:
    state = book(marks={"BONK": mark_at(0.00002), "WIF": mark_at(1.83, symbol="WIF")})
    assert set(state.marks) == {"BONK"}
    assert set(state.position_values_usd) == {"BONK"}


def test_mark_book_does_not_alias_the_caller_s_positions() -> None:
    positions = {"BONK": position()}
    state = book(positions=positions, marks={"BONK": mark_at(0.000025)})
    state.positions.clear()
    assert set(positions) == {"BONK"}


def test_mark_book_rejects_a_non_finite_cash_balance() -> None:
    with pytest.raises(ValidationError):
        book(cash_usd=float("nan"))


# ---------------------------------------------------------------------------
# stop_price_usd — one number, audit §11
# ---------------------------------------------------------------------------


def test_engine_stop_and_displayed_stop_are_the_same_number() -> None:
    """§11: the engine compared mark value against cost basis while the display
    showed ``avg_entry * 0.85``. There is now one function and both call it."""
    pos = position()
    displayed = stop_price_usd(pos, 0.15)
    assert displayed is not None

    # The engine's trigger, reached through the public API, fires at exactly
    # that price and not a tick above it.
    at_stop = book(marks={"BONK": mark_at(displayed)})
    just_above = book(marks={"BONK": mark_at(displayed * (1.0 + 1e-6))})

    breaches = stop_loss_breaches(at_stop, 0.15)
    assert [b.symbol for b in breaches] == ["BONK"]
    assert breaches[0].stop_price_usd == displayed
    assert breaches[0].reference_price_usd == displayed
    assert stop_loss_breaches(just_above, 0.15) == ()


def test_the_stop_uses_cost_basis_not_average_entry() -> None:
    """They differ by the entry cost, which is money that is genuinely gone."""
    pos = position()
    stop = stop_price_usd(pos, 0.15)
    naive = pos.avg_entry_price_usd * 0.85
    assert stop is not None
    assert stop == pytest.approx(pos.cost_basis_usd / pos.quantity * 0.85)
    assert stop != pytest.approx(naive, rel=1e-6)


def test_stop_price_of_a_position_with_no_quantity_is_none_not_zero() -> None:
    assert stop_price_usd(position(quantity=0.0, cost_basis_usd=0.0), 0.15) is None


def test_stop_price_rejects_a_non_finite_threshold() -> None:
    with pytest.raises(ValidationError):
        stop_price_usd(position(), float("nan"))


# ---------------------------------------------------------------------------
# stop_loss_breaches
# ---------------------------------------------------------------------------


def price_for_drawdown(pos: Position, pct: float) -> float:
    return pos.cost_basis_usd * (1.0 + pct / 100.0) / pos.quantity


def test_stop_fires_at_exactly_minus_fifteen() -> None:
    pos = position()
    price = price_for_drawdown(pos, -15.0)
    state = book(marks={"BONK": mark_at(price)})
    assert pos.unrealized_pnl_pct(price) == pytest.approx(-15.0, abs=1e-9)
    assert [b.symbol for b in stop_loss_breaches(state, 0.15)] == ["BONK"]


def test_stop_does_not_fire_at_minus_fourteen_point_nine() -> None:
    state = book(marks={"BONK": mark_at(price_for_drawdown(position(), -14.9))})
    assert stop_loss_breaches(state, 0.15) == ()


def test_stop_fires_well_below_the_threshold() -> None:
    state = book(marks={"BONK": mark_at(price_for_drawdown(position(), -40.0))})
    assert [b.symbol for b in stop_loss_breaches(state, 0.15)] == ["BONK"]


def test_stop_ignores_winners() -> None:
    assert stop_loss_breaches(book(marks={"BONK": mark_at(0.000025)}), 0.15) == ()


def test_stop_loss_pct_is_a_fraction_not_a_whole_percent() -> None:
    """If the conversion were dropped, 0.15 would mean -0.15% and this -1%
    position would breach; if it were applied twice, -15% would need -1500%."""
    pos = position()
    shallow = book(marks={"BONK": mark_at(price_for_drawdown(pos, -1.0))})
    assert stop_loss_breaches(shallow, 0.15) == ()
    deep = book(marks={"BONK": mark_at(price_for_drawdown(pos, -15.0))})
    assert [b.symbol for b in stop_loss_breaches(deep, 0.15)] == ["BONK"]


def test_breaches_are_ordered_worst_first() -> None:
    bonk, wif = (
        position(),
        position("WIF", quantity=100.0, decimals=6, cost_basis_usd=300.0),
    )
    state = book(
        positions={"BONK": bonk, "WIF": wif},
        marks={
            "BONK": mark_at(price_for_drawdown(bonk, -20.0)),
            "WIF": mark_at(price_for_drawdown(wif, -60.0), symbol="WIF"),
        },
    )
    assert [b.symbol for b in stop_loss_breaches(state, 0.15)] == ["WIF", "BONK"]


def test_an_unmarkable_position_raises_no_breach_but_is_an_incident() -> None:
    """You do not force a liquidation on the strength of data you do not have.
    The old code reached this answer by accident, having already overwritten the
    position with its cost basis; here it is a decision, and the incident is
    surfaced instead of swallowed."""
    state = book(marks={})
    assert stop_loss_breaches(state, 0.15) == ()
    assert state.unmarkable == ("BONK",)
    assert state.total_value_usd is None


def test_a_mid_basis_stop_still_fires_but_is_flagged_degraded() -> None:
    price = price_for_drawdown(position(), -40.0)
    state = book(marks={"BONK": mark_at(price, basis="mid")})
    (breach,) = stop_loss_breaches(state, 0.15)
    assert breach.degraded_reference is True
    assert breach.reference_basis == "mid"
    assert "not executable" in breach.reason


def test_a_route_basis_stop_is_not_flagged_degraded() -> None:
    price = price_for_drawdown(position(), -40.0)
    state = book(marks={"BONK": mark_at(price, basis="route")})
    (breach,) = stop_loss_breaches(state, 0.15)
    assert breach.degraded_reference is False
    assert breach.reference_basis == "route"


def test_a_mid_basis_stop_can_be_suppressed_by_config() -> None:
    price = price_for_drawdown(position(), -40.0)
    state = book(marks={"BONK": mark_at(price, basis="mid")})
    params = MarkParams(allow_mid_stop_reference=False)
    assert stop_loss_breaches(state, 0.15, params=params) == ()
    # ...and the executable reference is unaffected by that switch.
    route = book(marks={"BONK": mark_at(price, basis="route")})
    assert len(stop_loss_breaches(route, 0.15, params=params)) == 1


def test_stop_on_an_empty_book() -> None:
    assert stop_loss_breaches(book(positions={}), 0.15) == ()


def test_breach_reason_names_the_coin_and_both_numbers() -> None:
    price = price_for_drawdown(position(), -40.0)
    (breach,) = stop_loss_breaches(book(marks={"BONK": mark_at(price)}), 0.15)
    assert "BONK" in breach.reason
    assert breach.unrealized_pnl_pct == pytest.approx(-40.0)
    assert math.isfinite(breach.stop_price_usd)


# ---------------------------------------------------------------------------
# Counting exits — audit §11's "failed stop reported as exit"
# ---------------------------------------------------------------------------


def test_a_failed_stop_fill_is_not_counted_as_an_exit() -> None:
    """The exact bug: a ``failed=True`` stop fill was appended to
    ``stop_loss_exits``, so the CLI said 'exited' while the inventory remained."""
    attempts = [fill(state=OrderState.FAILED)]
    assert landed_exits(attempts) == ()
    assert len(failed_attempts(attempts)) == 1


def test_an_expired_stop_fill_is_not_counted_as_an_exit() -> None:
    assert landed_exits([fill(state=OrderState.EXPIRED)]) == ()


def test_a_submitted_fill_is_not_yet_an_exit() -> None:
    """An attempt with an unknown outcome is not an exit. The audit's required
    detection is landed && reconciled."""
    assert landed_exits([fill(state=OrderState.SUBMITTED)]) == ()


def test_a_landed_sell_is_an_exit() -> None:
    assert len(landed_exits([fill(state=OrderState.LANDED)])) == 1
    assert len(landed_exits([fill(state=OrderState.RECONCILED)])) == 1


def test_a_landed_buy_is_not_an_exit() -> None:
    assert landed_exits([fill(side=Side.BUY, state=OrderState.LANDED)]) == ()


def test_mixed_attempts_are_separated_correctly() -> None:
    attempts = [
        fill(state=OrderState.LANDED),
        fill(state=OrderState.FAILED),
        fill(side=Side.BUY, state=OrderState.LANDED),
        fill(state=OrderState.EXPIRED),
    ]
    assert len(landed_exits(attempts)) == 1
    assert len(failed_attempts(attempts)) == 2
