"""Every risk layer, on both sides of its boundary, and the bound invariant.

The layers are pure functions of state, so these tests need no broker, no files
and no clock beyond the ``now`` they pass in. That property is itself tested.

Rewritten against audit C3, C4, C8 and C10. What was removed and why:

* **Every clamp test.** ``test_position_limit_clamps_a_too_large_buy``,
  ``test_insufficient_cash_clamps``, ``test_sell_clamps_to_the_position_value``,
  ``test_both_clamps_compose_to_the_tighter_one`` and the five cash-clamp
  arithmetic tests asserted the C3 defect directly: risk shrinking an order that
  had already been quoted at a different size. Risk no longer returns an order,
  so a clamp is not a thing it can do. Their *arithmetic* survives as cap tests —
  the cash reservation, the gas reservation and the composition-by-``min`` are
  all still pinned, but as bounds rather than as approvals.
* **``test_the_clamp_is_always_affordable_at_the_broker``** and the two pool-fee
  clamp tests. They coupled risk to ``broker.pool_fee_rate``; risk no longer
  imports the broker, and the fee assumption is an explicit parameter
  (``assumed_pool_fee_pct``) with its default and its reasoning documented.
* **``test_a_symbol_absent_from_the_snapshot_rejects``'s exit half.** The old
  ``missing_snapshot`` rule existed only to guard ``min_liquidity``, which an
  exit bypasses; on the entry path it survives, renamed.
* **``test_there_is_no_max_trades_per_day``** survives in spirit as the purity
  test. The rule is still deliberately absent.

Kept, because their intent still holds: the stale-data veto and its exact
boundary, the liquidity floor and its boundary, the price-impact limit and its
boundary, the minimum notional, ``no_position``, every stop-loss bypass and the
measured reason each one exists, the "every verdict names its coin" family, and
the "nothing is hardcoded, it all comes from config" family.
"""

from __future__ import annotations

from dataclasses import replace
from types import EllipsisType

import pytest

from memetrader import risk
from memetrader.risk import (
    ContinuousRisk,
    EligibilityRisk,
    PortfolioRisk,
    PreTradeRisk,
    RiskEngine,
    RiskLedger,
    RiskParams,
    update_ledger,
)
from memetrader.types import (
    CoinSnapshot,
    DataQuality,
    Fill,
    Forecast,
    Mark,
    OrderState,
    PoolRef,
    PortfolioState,
    Position,
    PriceLadder,
    Provenance,
    Quote,
    RiskState,
    Side,
    TokenMeta,
    TxnCounts,
    ValidationError,
    ValuationEstimate,
)

NOW = 1_700_000_000.0
BONK_MINT = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

USDC = TokenMeta(mint=USDC_MINT, decimals=6, source="test")
BONK = TokenMeta(mint=BONK_MINT, decimals=5, source="test")

SLEEVE = frozenset({"BONK", "WIF", "POPCAT"})

#: The parameters the arithmetic in this file is written against.
#:   max_position_pct    0.30 (fraction)   max_gross_exposure_pct   60.0
#:   min_cash_floor_pct  10.0              default_entry_usd        25.0
#:   max_entry_usd      100.0              min_trade_usd            10.0
#:   max_price_impact_pct 3.0              max_snapshot_age_seconds 90.0
#:   min_liquidity_usd  50_000             gas_usd_per_swap          0.21
PARAMS = RiskParams(universe=SLEEVE, correlated_sleeve=SLEEVE)
ENGINE = RiskEngine(PARAMS)

OPEN_STATE = RiskState(ts=NOW)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def coin(
    symbol: str = "BONK",
    *,
    liquidity_usd: float | None = 250_000.0,
    price_usd: float | None = 0.00002,
    age_seconds: float = 5.0,
    quality: DataQuality = DataQuality.OK,
    trusted_quote: bool = True,
    pool_age_seconds: float | None = 86_400.0 * 30,
) -> CoinSnapshot:
    created = None if pool_age_seconds is None else NOW - pool_age_seconds
    return CoinSnapshot(
        symbol=symbol,
        mint=BONK_MINT,
        price_usd=price_usd,
        liquidity_usd=liquidity_usd,
        volume_24h_usd=1_000_000.0,
        volume_1h_usd=50_000.0,
        fdv_usd=None,
        price_change=PriceLadder(m5=0.1, h1=0.5, h6=-1.0, h24=2.0),
        txns_m5=TxnCounts(buys=10, sells=8),
        txns_h1=TxnCounts(buys=100, sells=90),
        txns_h24=TxnCounts(buys=1000, sells=900),
        pool=PoolRef(
            pair_address="pair",
            dex_id="raydium",
            base_mint=BONK_MINT,
            quote_mint=USDC_MINT,
            quote_symbol="USDC",
            created_at=created,
            trusted_quote=trusted_quote,
        ),
        provenance=Provenance(source="dexscreener", receive_time=NOW - age_seconds),
        quality=quality,
    )


def position(
    symbol: str = "BONK", *, value_price: float = 0.00002, quantity: float = 10_000_000.0
):
    return Position(
        symbol=symbol,
        mint=BONK_MINT,
        quantity_atomic=int(quantity * 10**5),
        decimals=5,
        avg_entry_price_usd=value_price,
        opened_at=NOW - 600.0,
        cost_basis_usd=quantity * value_price,
    )


def book(
    *,
    cash: float = 1000.0,
    holdings: dict[str, float] | None = None,
    unmarkable: tuple[str, ...] = (),
    starting_cash_usd: float = 1000.0,
) -> PortfolioState:
    """``holdings`` maps symbol -> marked USD value. Positions are otherwise flat."""
    holdings = holdings or {}
    positions: dict[str, Position] = {}
    values: dict[str, float | None] = {}
    marks: dict[str, Mark] = {}
    for symbol, value in holdings.items():
        quantity = 10_000_000.0
        price = value / quantity
        positions[symbol] = position(symbol, value_price=price, quantity=quantity)
        values[symbol] = None if symbol in unmarkable else value
        marks[symbol] = Mark(
            symbol=symbol,
            price_usd=None if symbol in unmarkable else price,
            basis="unavailable" if symbol in unmarkable else "route",
            provenance=None,
        )
    marked = [v for v in values.values() if v is not None]
    complete = not unmarkable
    return PortfolioState(
        ts=NOW,
        cash_usd=cash,
        positions=positions,
        marks=marks,
        position_values_usd=values,
        unrealized_pnl_usd=0.0 if complete else None,
        realized_pnl_usd=0.0,
        total_value_usd=(cash + sum(marked)) if complete else None,
        starting_cash_usd=starting_cash_usd,
        unmarkable=tuple(unmarkable),
    )


def quote(
    *,
    symbol: str = "BONK",
    side: Side = Side.BUY,
    price_impact_pct: float = 0.4,
    received_at: float = NOW,
    expires_at: float | None = None,
) -> Quote:
    if side is Side.BUY:
        in_tok, out_tok, in_atomic, out_atomic = USDC, BONK, 100_000_000, 5_000_000_000_000
    else:
        in_tok, out_tok, in_atomic, out_atomic = BONK, USDC, 5_000_000_000_000, 100_000_000
    return Quote(
        symbol=symbol,
        side=side,
        input_token=in_tok,
        output_token=out_tok,
        in_amount_atomic=in_atomic,
        out_amount_atomic=out_atomic,
        min_out_amount_atomic=int(out_atomic * 0.99),
        price_impact_pct=price_impact_pct,
        route_labels=("Raydium",),
        fingerprint="fp-1",
        requested_at=received_at - 0.2,
        received_at=received_at,
        expires_at=expires_at,
    )


def estimate(symbol: str = "BONK") -> ValuationEstimate:
    return ValuationEstimate(
        symbol=symbol,
        mid_price_usd=0.00002,
        haircut_pct=5.0,
        reason="jupiter unreachable",
        at=NOW,
        source="dexscreener",
    )


def entry(
    symbol: str = "BONK",
    *,
    engine: RiskEngine = ENGINE,
    state: PortfolioState | None = None,
    risk_state: RiskState = OPEN_STATE,
    snapshot: CoinSnapshot | EllipsisType | None = ...,
    volatility_pct: float | None = 8.0,
    now: float = NOW,
    **kw,
):
    if snapshot is ...:
        snapshot = coin(symbol)
    return engine.entry_bounds(
        symbol,
        book=state if state is not None else book(),
        risk_state=risk_state,
        snapshot=snapshot,
        volatility_pct=volatility_pct,
        now=now,
        **kw,
    )


def exit_(
    symbol: str = "BONK",
    *,
    engine: RiskEngine = ENGINE,
    state: PortfolioState | None = None,
    risk_state: RiskState = OPEN_STATE,
    snapshot: CoinSnapshot | None = None,
    forced: bool = False,
    now: float = NOW,
):
    return engine.exit_bounds(
        symbol,
        book=state if state is not None else book(cash=600.0, holdings={"BONK": 200.0}),
        risk_state=risk_state,
        snapshot=snapshot,
        forced=forced,
        now=now,
    )


def fill(
    *,
    symbol: str = "BONK",
    side: Side = Side.BUY,
    state: OrderState = OrderState.LANDED,
    ts: float = NOW,
) -> Fill:
    landed = state in (OrderState.LANDED, OrderState.RECONCILED)
    return Fill(
        fill_id="fil",
        order_id="ord",
        intent_id="int",
        decision_id="dec",
        ts=ts,
        symbol=symbol,
        side=side,
        state=state,
        in_amount_atomic=1 if landed else 0,
        out_amount_atomic=1 if landed else 0,
        token_amount_atomic=1 if landed else 0,
        token_decimals=5,
        quote_fingerprint="fp",
        price_usd=0.00002 if landed else None,
        notional_usd=25.0 if landed else 0.0,
        price_impact_pct=0.4,
        pool_fee_usd=0.0,
        gas_usd=0.21,
    )


# ---------------------------------------------------------------------------
# C3 — risk returns bounds, never an order
# ---------------------------------------------------------------------------


def test_a_permitted_entry_returns_a_bound_and_names_its_binding_rule() -> None:
    bounds = entry()
    assert bounds.permitted
    assert bounds.side is Side.BUY
    assert bounds.symbol == "BONK"
    # 1000 book, no existing position: concentration 500 (50% of book), gross
    # 900, cash floor 899.79, entry ceiling 600, uncalibrated cash-based
    # fallback 500 (50% of $1000 cash). Concentration and the cash-based
    # fallback tie at 500; concentration is listed first and binds.
    assert bounds.max_notional_usd == pytest.approx(500.0)
    assert bounds.binding_rule == "max_position_pct"
    assert bounds.reason


def test_risk_bounds_expose_nothing_that_looks_like_an_order() -> None:
    """C3, structurally. There is no quantity, no quote, no approved amount and
    no side effect to mistake for an executable instruction."""
    from dataclasses import fields as dataclass_fields

    names = {f.name for f in dataclass_fields(entry())}
    forbidden = {"approved", "approved_usd", "quantity", "quote", "order", "size_usd"}
    assert not (names & forbidden)
    assert "max_notional_usd" in names


def test_confirm_quote_refuses_a_notional_over_the_bound() -> None:
    """The C3 incident in one test: a $600 quote may not be used for a $600
    order when the bound is $500. Risk refuses; it does not shrink."""
    bounds = entry()
    refusal = ENGINE.confirm_quote(bounds, quote(), notional_usd=600.0, now=NOW)
    assert not refusal.permitted
    assert refusal.vetoes == ("exceeds_bound",)
    assert refusal.max_notional_usd == 0.0


def test_confirm_quote_accepts_a_notional_at_the_bound_and_changes_nothing() -> None:
    bounds = entry()
    confirmed = ENGINE.confirm_quote(
        bounds, quote(), notional_usd=bounds.max_notional_usd, now=NOW
    )
    assert confirmed.permitted
    assert confirmed.max_notional_usd == bounds.max_notional_usd
    assert confirmed.binding_rule == bounds.binding_rule
    assert any("confirmed" in n for n in confirmed.notes)


def test_confirm_quote_rejects_a_quote_for_the_wrong_symbol_or_side() -> None:
    bounds = entry()
    wrong = ENGINE.confirm_quote(
        bounds, quote(symbol="WIF"), notional_usd=bounds.max_notional_usd, now=NOW
    )
    assert "quote_mismatch" in wrong.vetoes
    sell = ENGINE.confirm_quote(
        bounds, quote(side=Side.SELL), notional_usd=bounds.max_notional_usd, now=NOW
    )
    assert "quote_mismatch" in sell.vetoes


def test_confirm_quote_rejects_a_stale_quote() -> None:
    bounds = entry()
    stale = quote(received_at=NOW - PARAMS.max_quote_age_seconds - 1.0)
    refusal = ENGINE.confirm_quote(bounds, stale, notional_usd=25.0, now=NOW)
    assert "quote_age" in refusal.vetoes


def test_confirm_quote_rejects_an_expired_quote() -> None:
    bounds = entry()
    refusal = ENGINE.confirm_quote(
        bounds, quote(expires_at=NOW - 1.0), notional_usd=25.0, now=NOW
    )
    assert "quote_expired" in refusal.vetoes


def test_confirm_quote_refuses_against_bounds_that_permit_nothing() -> None:
    refusal = ENGINE.confirm_quote(
        entry(risk_state=RiskState(ts=NOW, halted=True, halt_reasons=("test",))),
        quote(),
        notional_usd=10.0,
        now=NOW,
    )
    assert not refusal.permitted


# ---------------------------------------------------------------------------
# The bound invariant — the property this module exists to guarantee
# ---------------------------------------------------------------------------


CAP_SCENARIOS = [
    ("flat book", book(), 8.0),
    ("rich book", book(cash=10_000.0), 8.0),
    ("concentrated", book(cash=100.0, holdings={"BONK": 280.0}), 8.0),
    ("sleeve loaded", book(cash=200.0, holdings={"WIF": 400.0, "POPCAT": 300.0}), 8.0),
    ("thin cash", book(cash=30.0, holdings={"WIF": 900.0}), 8.0),
    ("high vol", book(cash=10_000.0), 80.0),
    ("low vol", book(cash=10_000.0), 1.0),
]


@pytest.mark.parametrize("label,state,vol", CAP_SCENARIOS)
def test_the_bound_never_exceeds_any_individual_cap(
    label: str, state: PortfolioState, vol: float
) -> None:
    """The invariant. ``max_notional_usd`` is a ``min`` over every cap, so no
    arrangement of inputs can produce a bound that any one rule would refuse."""
    caps, vetoes, _, _ = ENGINE.entry_caps(
        "BONK",
        book=state,
        risk_state=OPEN_STATE,
        snapshot=coin(),
        volatility_pct=vol,
        now=NOW,
    )
    bounds = entry(state=state, volatility_pct=vol)
    if vetoes:
        assert bounds.max_notional_usd == 0.0, label
        return
    assert caps, label
    for cap in caps:
        assert bounds.max_notional_usd <= cap.max_notional_usd + 1e-9, (label, cap.rule)
    assert bounds.max_notional_usd == pytest.approx(
        min(c.max_notional_usd for c in caps)
    ), label


@pytest.mark.parametrize("label,state,vol", CAP_SCENARIOS)
def test_the_binding_rule_is_the_cap_that_actually_bound(
    label: str, state: PortfolioState, vol: float
) -> None:
    caps, vetoes, _, _ = ENGINE.entry_caps(
        "BONK",
        book=state,
        risk_state=OPEN_STATE,
        snapshot=coin(),
        volatility_pct=vol,
        now=NOW,
    )
    bounds = entry(state=state, volatility_pct=vol)
    if vetoes or not bounds.permitted:
        return
    tightest = min(caps, key=lambda c: c.max_notional_usd)
    assert bounds.binding_rule == tightest.rule, label


def test_a_bound_is_never_negative() -> None:
    """Negative headroom is a refusal, not a negative order."""
    over = book(cash=10.0, holdings={"BONK": 990.0})
    caps, _, _, _ = ENGINE.entry_caps(
        "BONK",
        book=over,
        risk_state=OPEN_STATE,
        snapshot=coin(),
        volatility_pct=8.0,
        now=NOW,
    )
    assert all(c.max_notional_usd >= 0.0 for c in caps)
    assert entry(state=over).max_notional_usd == 0.0


# ---------------------------------------------------------------------------
# NaN discipline
# ---------------------------------------------------------------------------


def test_a_nan_volatility_is_rejected_rather_than_passing() -> None:
    """Every comparison against NaN is false, *including* ``vol <= 0``, so an
    unvalidated NaN would skip the veto and then scale the size by NaN."""
    bounds = entry(volatility_pct=float("nan"))
    assert not bounds.permitted
    assert bounds.vetoes == ("non_finite_input",)
    assert bounds.max_notional_usd == 0.0


def test_a_nan_now_is_rejected() -> None:
    assert entry(now=float("nan")).vetoes == ("non_finite_input",)


def test_an_infinite_volatility_is_rejected() -> None:
    assert entry(volatility_pct=float("inf")).vetoes == ("non_finite_input",)


def test_a_nan_notional_cannot_be_bound_to_a_quote() -> None:
    refusal = ENGINE.confirm_quote(entry(), quote(), notional_usd=float("nan"), now=NOW)
    assert refusal.vetoes == ("non_finite_input",)


def test_risk_params_reject_a_whole_percent_supplied_as_a_fraction_and_vice_versa() -> None:
    with pytest.raises(ValidationError):
        RiskParams(max_position_pct=30.0)  # fraction field given a whole percent
    with pytest.raises(ValidationError):
        RiskParams(max_gross_exposure_pct=600.0)  # whole-percent field out of range
    with pytest.raises(ValidationError):
        RiskParams(stop_loss_pct=float("nan"))


# ---------------------------------------------------------------------------
# C4 — the entry/exit asymmetry
# ---------------------------------------------------------------------------


def test_a_valuation_estimate_vetoes_an_entry() -> None:
    """A routing outage is when a mid-price fiction is least credible."""
    bounds = entry(valuation=estimate())
    assert not bounds.permitted
    assert "degraded_valuation" in bounds.vetoes
    assert "C4" in bounds.reason


def test_a_valuation_estimate_does_not_block_an_exit() -> None:
    """The asymmetry. An untrustworthy snapshot is a reason not to buy and
    frequently a reason to sell, so the same predicate must not govern both.
    ``exit_bounds`` does not even accept a valuation — it structurally cannot
    apply C4's veto."""
    bounds = exit_()
    assert bounds.permitted
    assert bounds.max_notional_usd == pytest.approx(200.0)


def test_a_degraded_snapshot_vetoes_an_entry_but_not_an_exit() -> None:
    degraded = coin(quality=DataQuality.DEGRADED)
    assert "degraded_snapshot" in entry(snapshot=degraded).vetoes
    assert exit_(snapshot=degraded).permitted


def test_an_untrusted_quote_token_vetoes_an_entry() -> None:
    """C8: a pool priced in a token whose own USD price is unknown is diagnostic
    data, never an entry price."""
    bounds = entry(snapshot=coin(trusted_quote=False))
    assert "untrusted_quote_token" in bounds.vetoes


# ---------------------------------------------------------------------------
# C8 — an unmarkable book blocks entries and cannot be divided by
# ---------------------------------------------------------------------------


def test_an_unmarkable_book_vetoes_every_entry() -> None:
    state = book(cash=600.0, holdings={"BONK": 200.0, "WIF": 200.0}, unmarkable=("WIF",))
    bounds = entry(state=state)
    assert not bounds.permitted
    assert "book_unmarkable" in bounds.vetoes
    assert "C8" in bounds.reason


def test_an_unmarkable_book_still_permits_an_exit() -> None:
    state = book(cash=600.0, holdings={"BONK": 200.0}, unmarkable=("BONK",))
    bounds = exit_(state=state)
    assert bounds.permitted
    # Bounded by cost basis as a permission ceiling, loudly noted, never as a
    # valuation. Cost basis is only tolerable here because the execution layer
    # is bounded by inventory in atomic units.
    assert "mark_unavailable" in bounds.bypassed_rules
    assert any("UNMARKABLE" in n for n in bounds.notes)
    assert bounds.max_notional_usd == pytest.approx(200.0)


def test_continuous_risk_turns_an_unmarkable_position_into_an_incident() -> None:
    state = book(cash=600.0, holdings={"BONK": 200.0}, unmarkable=("BONK",))
    rs = ContinuousRisk(PARAMS).evaluate(book=state, now=NOW)
    assert rs.data_health_ok is False
    assert "BONK" in rs.quarantined_symbols
    assert rs.may_open is False


# ---------------------------------------------------------------------------
# C10 — the kill switch and the continuous breakers
# ---------------------------------------------------------------------------


def test_the_kill_switch_blocks_entries_and_permits_exits() -> None:
    halted = RiskState(ts=NOW, halted=True, halt_reasons=("operator pulled the switch",))
    blocked = entry(risk_state=halted)
    assert not blocked.permitted
    assert "halted" in blocked.vetoes

    allowed = exit_(risk_state=halted)
    assert allowed.permitted
    assert allowed.max_notional_usd == pytest.approx(200.0)
    assert any("exit-only" in n for n in allowed.notes)


def test_the_manual_kill_switch_halts_with_its_reason() -> None:
    ledger = RiskLedger(manual_halt=True, manual_halt_reason="RPC quorum lost")
    rs = ContinuousRisk(PARAMS).evaluate(book=book(), ledger=ledger, now=NOW)
    assert rs.halted
    assert any("RPC quorum lost" in r for r in rs.halt_reasons)


def test_the_drawdown_breaker_halts() -> None:
    ledger = RiskLedger(peak_value_usd=1000.0)
    ok = ContinuousRisk(PARAMS).evaluate(book=book(cash=850.0), ledger=ledger, now=NOW)
    assert not ok.halted
    assert ok.drawdown_pct == pytest.approx(15.0)

    blown = ContinuousRisk(PARAMS).evaluate(book=book(cash=700.0), ledger=ledger, now=NOW)
    assert blown.halted
    assert blown.drawdown_pct == pytest.approx(30.0)
    assert any("drawdown" in r for r in blown.halt_reasons)


def test_the_daily_loss_budget_halts() -> None:
    ledger = RiskLedger(day_start_value_usd=1000.0, day_start_ts=NOW - 3600.0)
    ok = ContinuousRisk(PARAMS).evaluate(book=book(cash=960.0), ledger=ledger, now=NOW)
    assert not ok.halted
    assert ok.rolling_loss_pct == pytest.approx(-4.0)

    blown = ContinuousRisk(PARAMS).evaluate(book=book(cash=940.0), ledger=ledger, now=NOW)
    assert blown.halted
    assert any("daily loss budget" in r for r in blown.halt_reasons)


def test_the_rolling_window_loss_budget_halts() -> None:
    ledger = RiskLedger(
        day_start_value_usd=900.0,
        day_start_ts=NOW - 3600.0,
        window_start_value_usd=1000.0,
        window_start_ts=NOW - 86_400.0 * 3,
        peak_value_usd=1000.0,
    )
    blown = ContinuousRisk(
        replace(PARAMS, max_drawdown_pct=90.0, max_daily_loss_pct=90.0)
    ).evaluate(book=book(cash=880.0), ledger=ledger, now=NOW)
    assert blown.halted
    assert any("book change" in r and "budget" in r for r in blown.halt_reasons)


def test_the_consecutive_failure_breaker_halts() -> None:
    ok = ContinuousRisk(PARAMS).evaluate(
        book=book(), ledger=RiskLedger(consecutive_failures=2), now=NOW
    )
    assert not ok.halted
    blown = ContinuousRisk(PARAMS).evaluate(
        book=book(), ledger=RiskLedger(consecutive_failures=3), now=NOW
    )
    assert blown.halted
    assert any("consecutive execution failures" in r for r in blown.halt_reasons)


def test_degraded_data_health_blocks_entries_only() -> None:
    rs = ContinuousRisk(PARAMS).evaluate(book=book(), data_quality_ok=False, now=NOW)
    assert rs.data_health_ok is False
    assert rs.halted is False
    assert not entry(risk_state=rs).permitted
    assert exit_(risk_state=rs).permitted


def test_an_unknown_book_value_cannot_be_read_as_flat() -> None:
    """Missing is never zero. A loss budget that reads an unmarkable book as
    'unchanged' is a loss budget disabled at the worst possible moment."""
    state = book(cash=600.0, holdings={"BONK": 200.0}, unmarkable=("BONK",))
    rs = ContinuousRisk(PARAMS).evaluate(
        book=state, ledger=RiskLedger(day_start_value_usd=1000.0, day_start_ts=NOW), now=NOW
    )
    assert rs.rolling_loss_pct is None
    assert rs.drawdown_pct is None


def test_gross_and_sleeve_caps_bind_before_the_concentration_cap() -> None:
    """Three coins are one bet. The sleeve cap must be able to bind even when no
    single name is close to its own limit."""
    loose = replace(
        PARAMS,
        default_entry_usd=10_000.0,
        max_entry_usd=10_000.0,
        max_position_pct=0.90,
        max_sleeve_exposure_pct=50.0,
    )
    engine = RiskEngine(loose)
    state = book(cash=600.0, holdings={"WIF": 200.0, "POPCAT": 200.0})
    bounds = engine.entry_bounds(
        "BONK",
        book=state,
        risk_state=OPEN_STATE,
        snapshot=coin(),
        volatility_pct=8.0,
        now=NOW,
    )
    # NAV 1000, sleeve holds 400, sleeve cap 500 -> 100 of headroom.
    assert bounds.binding_rule == "max_sleeve_exposure_pct"
    assert bounds.max_notional_usd == pytest.approx(100.0)


def test_the_sleeve_cap_is_skipped_for_a_symbol_outside_the_sleeve() -> None:
    outside = replace(PARAMS, universe=SLEEVE | {"SOL"}, correlated_sleeve=SLEEVE)
    caps, vetoes, _, _ = RiskEngine(outside).entry_caps(
        "SOL",
        book=book(),
        risk_state=OPEN_STATE,
        snapshot=coin("SOL"),
        volatility_pct=8.0,
        now=NOW,
    )
    assert not vetoes
    assert "max_sleeve_exposure_pct" not in {c.rule for c in caps}


def test_the_cash_floor_reserves_gas_and_a_percentage_of_the_book() -> None:
    caps, _, _, _ = PortfolioRisk(PARAMS).caps("BONK", book=book(), volatility_pct=8.0)
    floor = next(c for c in caps if c.rule == "min_cash_floor_pct")
    # 1000 cash, 10% of a 1000 book held back, 0.21 gas reserved.
    assert floor.max_notional_usd == pytest.approx(899.79, rel=1e-9)


def test_vol_scaling_shrinks_size_as_realised_vol_rises() -> None:
    loose = replace(PARAMS, default_entry_usd=10_000.0, max_entry_usd=10_000.0)
    engine = RiskEngine(loose)
    calm = engine.entry_bounds(
        "BONK",
        book=book(cash=10_000.0),
        risk_state=OPEN_STATE,
        snapshot=coin(),
        volatility_pct=8.0,
        now=NOW,
    )
    wild = engine.entry_bounds(
        "BONK",
        book=book(cash=10_000.0),
        risk_state=OPEN_STATE,
        snapshot=coin(),
        volatility_pct=32.0,
        now=NOW,
    )
    assert wild.max_notional_usd < calm.max_notional_usd
    # Concentration allows 5000 of a 10k book; at 4x the target vol the vol cap
    # is 5000/4 = 1250 and becomes the binding rule. (Calm is bound elsewhere —
    # by depth participation at 1% of a 250k pool — which is the point: vol
    # targeting only ever tightens.)
    assert wild.binding_rule == "volatility_target"
    assert wild.max_notional_usd == pytest.approx(1250.0)


def test_vol_scaling_never_scales_a_position_up() -> None:
    """A quiet asset does not earn a bigger position than the concentration cap
    allows — ATR is backward-looking and its quietest readings often precede a
    jump rather than describe safety."""
    loose = replace(PARAMS, default_entry_usd=10_000.0, max_entry_usd=10_000.0)
    caps, _, _, _ = PortfolioRisk(loose).caps(
        "BONK", book=book(cash=10_000.0), volatility_pct=0.5
    )
    vol_cap = next(c for c in caps if c.rule == "volatility_target")
    concentration = next(c for c in caps if c.rule == "max_position_pct")
    assert vol_cap.max_notional_usd == pytest.approx(concentration.max_notional_usd)


def test_a_missing_volatility_estimate_vetoes_rather_than_assuming_calm() -> None:
    bounds = entry(volatility_pct=None)
    assert not bounds.permitted
    assert "no_volatility_estimate" in bounds.vetoes


def test_the_volatility_veto_is_configurable() -> None:
    engine = RiskEngine(replace(PARAMS, require_volatility_estimate=False))
    bounds = engine.entry_bounds(
        "BONK",
        book=book(),
        risk_state=OPEN_STATE,
        snapshot=coin(),
        volatility_pct=None,
        now=NOW,
    )
    assert bounds.permitted
    assert any("vol targeting not applied" in n for n in bounds.notes)


def test_depth_participation_caps_an_order_against_the_pool() -> None:
    caps, _, _, _ = PreTradeRisk(PARAMS).caps(
        "BONK", Side.BUY, book=book(), snapshot=coin(liquidity_usd=60_000.0), now=NOW
    )
    depth = next(c for c in caps if c.rule == "max_depth_participation_pct")
    assert depth.max_notional_usd == pytest.approx(600.0)


# ---------------------------------------------------------------------------
# Quarantine and entry spacing — audit §11's stop-then-rebuy
# ---------------------------------------------------------------------------


def test_post_stop_quarantine_blocks_an_immediate_rebuy() -> None:
    """The audit found the slow tick could stop out and re-enter the same asset
    in the same tick, converting a stop into a round-trip fee generator."""
    stopped = update_ledger(
        RiskLedger(), book=book(), risk_events=("BONK",), params=PARAMS, now=NOW
    )
    blocked = entry(ledger=stopped)
    assert not blocked.permitted
    assert "quarantined" in blocked.vetoes

    # ...and it expires.
    later = NOW + PARAMS.post_stop_quarantine_seconds + 1.0
    assert entry(
        ledger=stopped,
        now=later,
        snapshot=coin(age_seconds=5.0 - (later - NOW)),
    ).permitted


def test_a_quarantine_never_blocks_an_exit() -> None:
    stopped = update_ledger(
        RiskLedger(), book=book(), risk_events=("BONK",), params=PARAMS, now=NOW
    )
    rs = ContinuousRisk(PARAMS).evaluate(book=book(), ledger=stopped, now=NOW)
    bounds = exit_(risk_state=rs)
    assert bounds.permitted
    assert any("quarantined for re-entry" in n for n in bounds.notes)


def test_entry_spacing_blocks_a_second_entry_in_the_same_symbol() -> None:
    ledger = RiskLedger(last_entry_ts={"BONK": NOW - 60.0})
    assert "entry_cooldown" in entry(ledger=ledger).vetoes
    old = RiskLedger(last_entry_ts={"BONK": NOW - PARAMS.min_seconds_between_entries - 1.0})
    assert entry(ledger=old).permitted


def test_update_ledger_records_a_landed_entry_and_resets_the_failure_streak() -> None:
    before = RiskLedger(consecutive_failures=2)
    after = update_ledger(
        before, book=book(), fills=[fill(state=OrderState.LANDED)], params=PARAMS, now=NOW
    )
    assert after.consecutive_failures == 0
    assert after.last_entry_ts["BONK"] == NOW


def test_update_ledger_counts_failed_and_expired_attempts_as_failures() -> None:
    """§11: a fill row's existence is not proof that anything happened."""
    after = update_ledger(
        RiskLedger(),
        book=book(),
        fills=[fill(state=OrderState.FAILED), fill(state=OrderState.EXPIRED)],
        params=PARAMS,
        now=NOW,
    )
    assert after.consecutive_failures == 2
    assert after.last_entry_ts == {}


def test_update_ledger_only_advances_the_peak_from_a_fully_marked_book() -> None:
    partial = book(cash=600.0, holdings={"BONK": 200.0}, unmarkable=("BONK",))
    after = update_ledger(
        RiskLedger(peak_value_usd=1000.0), book=partial, params=PARAMS, now=NOW
    )
    assert after.peak_value_usd == 1000.0
    assert "BONK" in after.quarantined_until


def test_update_ledger_rolls_the_day_over() -> None:
    old = RiskLedger(day_start_value_usd=500.0, day_start_ts=NOW - 86_401.0)
    after = update_ledger(old, book=book(cash=1000.0), params=PARAMS, now=NOW)
    assert after.day_start_value_usd == pytest.approx(1000.0)
    assert after.day_start_ts == NOW


# ---------------------------------------------------------------------------
# Eligibility — rules kept from the old suite, with their boundaries
# ---------------------------------------------------------------------------


def test_stale_data_vetoes_an_entry() -> None:
    bounds = entry(snapshot=coin(age_seconds=91.0))
    assert "stale_data" in bounds.vetoes
    assert "91" in bounds.reason and "90" in bounds.reason


def test_snapshot_exactly_at_the_age_limit_passes() -> None:
    """The rule is ``>``, so the limit itself is still tradeable."""
    assert entry(snapshot=coin(age_seconds=90.0)).permitted


def test_fresh_data_passes() -> None:
    assert entry(snapshot=coin(age_seconds=89.0)).permitted


def test_thin_liquidity_vetoes() -> None:
    assert "min_liquidity" in entry(snapshot=coin(liquidity_usd=49_999.0)).vetoes


def test_liquidity_exactly_at_the_floor_passes() -> None:
    assert entry(snapshot=coin(liquidity_usd=50_000.0)).permitted


def test_missing_liquidity_vetoes_because_missing_is_not_deep() -> None:
    bounds = entry(snapshot=coin(liquidity_usd=None))
    assert "no_liquidity_observation" in bounds.vetoes


def test_a_missing_price_vetoes() -> None:
    assert "no_price" in entry(snapshot=coin(price_usd=None)).vetoes


def test_a_missing_snapshot_vetoes_an_entry() -> None:
    bounds = entry("WIF", snapshot=None)
    assert "missing_snapshot" in bounds.vetoes
    assert "WIF" in bounds.reason


def test_a_symbol_outside_the_universe_vetoes() -> None:
    bounds = entry("DOGE", snapshot=coin("DOGE"))
    assert "not_in_universe" in bounds.vetoes


def test_an_empty_universe_is_closed_not_open() -> None:
    engine = RiskEngine(replace(PARAMS, universe=frozenset()))
    bounds = engine.entry_bounds(
        "BONK",
        book=book(),
        risk_state=OPEN_STATE,
        snapshot=coin(),
        volatility_pct=8.0,
        now=NOW,
    )
    assert "not_in_universe" in bounds.vetoes


def test_a_young_pool_vetoes() -> None:
    assert "new_pool" in entry(snapshot=coin(pool_age_seconds=3600.0)).vetoes


def test_an_unknown_pool_age_vetoes_because_missing_is_not_old() -> None:
    assert "unknown_pool_age" in entry(snapshot=coin(pool_age_seconds=None)).vetoes


def test_the_unknown_pool_age_veto_is_configurable() -> None:
    engine = RiskEngine(replace(PARAMS, require_known_pool_age=False))
    bounds = engine.entry_bounds(
        "BONK",
        book=book(),
        risk_state=OPEN_STATE,
        snapshot=coin(pool_age_seconds=None),
        volatility_pct=8.0,
        now=NOW,
    )
    assert bounds.permitted


def test_eligibility_is_testable_on_its_own() -> None:
    vetoes, reasons = EligibilityRisk(PARAMS).assess("BONK", snapshot=coin(), now=NOW)
    assert vetoes == ()
    assert reasons == ()


# ---------------------------------------------------------------------------
# Pre-trade
# ---------------------------------------------------------------------------


def test_price_impact_above_the_limit_vetoes() -> None:
    bounds = entry(quote=quote(price_impact_pct=3.1))
    assert "max_price_impact" in bounds.vetoes
    assert "3.1" in bounds.reason


def test_price_impact_exactly_at_the_limit_passes() -> None:
    assert entry(quote=quote(price_impact_pct=3.0)).permitted


def test_the_minimum_notional_refuses_rather_than_permitting_dust() -> None:
    tight = RiskEngine(replace(PARAMS, default_entry_usd=9.99, max_cash_fraction_pct=0.0))
    bounds = tight.entry_bounds(
        "BONK",
        book=book(),
        risk_state=OPEN_STATE,
        snapshot=coin(),
        volatility_pct=8.0,
        now=NOW,
    )
    assert not bounds.permitted
    assert "min_notional" in bounds.vetoes
    assert "uncalibrated_forecast" in bounds.vetoes  # the binding rule is named


def test_no_cash_at_all_refuses() -> None:
    bounds = entry(state=book(cash=0.0, holdings={"WIF": 1000.0}))
    assert not bounds.permitted


# ---------------------------------------------------------------------------
# Sizing on an uncalibrated forecast — audit C1
# ---------------------------------------------------------------------------


def uncalibrated(symbol: str = "BONK") -> Forecast:
    return Forecast(
        symbol=symbol,
        horizon_seconds=900.0,
        expected_net_return_pct=4.0,
        lower_quantile_pct=0.5,
        upper_quantile_pct=9.0,
        model_id="baseline-v0",
    )


def calibrated(symbol: str = "BONK") -> Forecast:
    return replace(uncalibrated(symbol), calibration_id="cal-2026-09")


#: Isolates the uncalibrated_forecast cap from the concentration cap, which
#: otherwise ties with it at the module's default 50%/50% settings on the
#: default 1000-cash test book.
_ISOLATED = RiskEngine(replace(PARAMS, max_position_pct=1.0))


def test_an_uncalibrated_forecast_falls_back_to_a_cash_based_size() -> None:
    """One closed round trip is not evidence of alpha, so the magnitude of an
    uncalibrated forecast is a number with units of nothing — the fallback
    sizes on available cash instead (50% of a 1000 book, floored at
    default_entry_usd)."""
    bounds = entry(engine=_ISOLATED, forecast=uncalibrated())
    assert bounds.binding_rule == "uncalibrated_forecast"
    assert bounds.max_notional_usd == pytest.approx(500.0)
    assert "calibration_id" in bounds.reason


def test_no_forecast_at_all_also_falls_back_to_a_cash_based_size() -> None:
    assert entry(engine=_ISOLATED).max_notional_usd == pytest.approx(500.0)


def test_a_calibrated_forecast_lifts_the_cash_based_cap() -> None:
    bounds = entry(engine=_ISOLATED, forecast=calibrated())
    assert bounds.max_notional_usd > 500.0
    assert bounds.binding_rule == "max_entry_usd"
    assert any("calibrated forecast" in n for n in bounds.notes)


def test_a_calibrated_but_unusable_forecast_vetoes() -> None:
    hollow = Forecast(
        symbol="BONK",
        horizon_seconds=900.0,
        expected_net_return_pct=None,
        lower_quantile_pct=None,
        upper_quantile_pct=None,
        model_id="m",
        calibration_id="cal-1",
    )
    assert "unusable_forecast" in entry(forecast=hollow).vetoes


# ---------------------------------------------------------------------------
# Exits, and the stop-loss bypasses (each with the reason it exists)
# ---------------------------------------------------------------------------


def test_a_plain_exit_is_bounded_by_the_position_value() -> None:
    bounds = exit_()
    assert bounds.permitted
    assert bounds.side is Side.SELL
    assert bounds.max_notional_usd == pytest.approx(200.0)
    assert bounds.binding_rule == "position_value"


def test_an_exit_with_no_position_refuses() -> None:
    bounds = exit_(state=book())
    assert not bounds.permitted
    assert bounds.vetoes == ("no_position",)
    assert "BONK" in bounds.reason


def test_an_exit_of_a_zero_quantity_position_refuses() -> None:
    state = book(cash=600.0, holdings={"BONK": 200.0})
    state.positions["BONK"] = replace(state.positions["BONK"], quantity_atomic=0)
    assert exit_(state=state).vetoes == ("no_position",)


def test_a_forced_exit_bypasses_the_liquidity_floor() -> None:
    """A draining pool is the reason to get out, not a reason to stay in. The
    rule fires hardest exactly when the stop exists — enforcing it there does
    not protect the position, it traps it."""
    thin = coin(liquidity_usd=1_200.0)
    assert not exit_(snapshot=thin).permitted
    forced = exit_(snapshot=thin, forced=True)
    assert forced.permitted
    assert "min_liquidity" in forced.bypassed_rules
    assert any("draining pool" in n for n in forced.notes)


def test_a_forced_exit_bypasses_price_impact_at_bind_time() -> None:
    forced = exit_(forced=True)
    assert "max_price_impact" in forced.bypassed_rules
    confirmed = ENGINE.confirm_quote(
        forced, quote(side=Side.SELL, price_impact_pct=41.0), notional_usd=200.0, now=NOW
    )
    assert confirmed.permitted


def test_an_ordinary_exit_is_still_blocked_by_price_impact() -> None:
    """So the bypass is genuinely conditional and not an accidental blanket hole."""
    ordinary = exit_()
    refusal = ENGINE.confirm_quote(
        ordinary, quote(side=Side.SELL, price_impact_pct=41.0), notional_usd=200.0, now=NOW
    )
    assert "max_price_impact" in refusal.vetoes


def test_a_forced_exit_bypasses_the_minimum_notional() -> None:
    """A forced exit must not be blocked by a rule about how big a trade should
    be."""
    dust = book(cash=600.0, holdings={"BONK": 3.0})
    assert not exit_(state=dust).permitted
    forced = exit_(state=dust, forced=True)
    assert forced.permitted
    assert "min_notional" in forced.bypassed_rules
    assert forced.max_notional_usd == pytest.approx(3.0)


def test_a_forced_exit_is_still_blocked_by_stale_data() -> None:
    """Deliberately *not* bypassed, and the reason is measured rather than
    aesthetic: the fast tick retries in 60s, so blocking here costs a minute,
    whereas exiting on a price we cannot vouch for costs the fill."""
    bounds = exit_(snapshot=coin(age_seconds=91.0), forced=True)
    assert not bounds.permitted
    assert bounds.vetoes == ("stale_data",)
    assert "60s" in bounds.reason


def test_an_exit_needs_no_snapshot_at_all() -> None:
    """``missing_snapshot`` existed only to guard the liquidity floor, which an
    exit bypasses, so there is nothing left for it to protect."""
    bounds = exit_(snapshot=None)
    assert bounds.permitted
    assert any("does not need one" in n for n in bounds.notes)


def test_there_is_no_minimum_hold_time() -> None:
    """A position opened one second ago can be sold. Deliberate."""
    state = book(cash=600.0, holdings={"BONK": 200.0})
    state.positions["BONK"] = replace(state.positions["BONK"], opened_at=NOW - 1.0)
    assert exit_(state=state).permitted


# ---------------------------------------------------------------------------
# Purity, and every bound naming its coin
# ---------------------------------------------------------------------------


def test_risk_is_a_pure_function_of_its_arguments() -> None:
    """Calling it a hundred times changes nothing. There is no max-trades-per-day
    cap and no hidden counter that could create one by accident."""
    results = {(entry().permitted, entry().max_notional_usd) for _ in range(100)}
    assert len(results) == 1


def test_risk_does_not_mutate_its_inputs() -> None:
    state = book(cash=900.0, holdings={"BONK": 100.0})
    snap = coin()
    before = (dict(state.positions), dict(state.position_values_usd), state.cash_usd)

    entry(state=state, snapshot=snap)
    exit_(state=state, snapshot=snap)

    assert (
        dict(state.positions),
        dict(state.position_values_usd),
        state.cash_usd,
    ) == before
    assert snap.liquidity_usd == 250_000.0


def test_risk_reads_no_clock_of_its_own() -> None:
    """``now`` is an argument on every public entry point, so a test can place
    the system at any instant and a replay is reproducible."""
    far_future = entry(now=NOW + 86_400.0, snapshot=coin(age_seconds=5.0 - 86_400.0))
    assert far_future.permitted


@pytest.mark.parametrize(
    "rule,kwargs",
    [
        ("stale_data", {"snapshot": coin(age_seconds=600.0)}),
        ("min_liquidity", {"snapshot": coin(liquidity_usd=1.0)}),
        ("missing_snapshot", {"snapshot": None}),
        ("degraded_snapshot", {"snapshot": coin(quality=DataQuality.DEGRADED)}),
        ("untrusted_quote_token", {"snapshot": coin(trusted_quote=False)}),
        ("new_pool", {"snapshot": coin(pool_age_seconds=60.0)}),
        ("max_price_impact", {"quote": quote(price_impact_pct=99.0)}),
        ("degraded_valuation", {"valuation": estimate()}),
        ("no_volatility_estimate", {"volatility_pct": None}),
        (
            "book_unmarkable",
            {"state": book(cash=600.0, holdings={"BONK": 200.0}, unmarkable=("BONK",))},
        ),
    ],
)
def test_every_veto_path_names_its_coin_and_gives_a_reason(rule: str, kwargs) -> None:
    """The decision log replays refusals on the next tick. With three coins in
    play a bare rule name leaves the reader to guess which one fired."""
    bounds = entry(**kwargs)
    assert not bounds.permitted, rule
    assert rule in bounds.vetoes, (rule, bounds.vetoes)
    assert bounds.symbol == "BONK"
    assert bounds.reason
    assert bounds.max_notional_usd == 0.0


def test_the_symbol_is_the_argument_not_a_constant() -> None:
    bounds = entry("POPCAT", snapshot=coin("POPCAT"))
    assert bounds.permitted
    assert bounds.symbol == "POPCAT"


def test_limits_come_from_params_not_constants() -> None:
    loose = replace(
        PARAMS,
        max_position_pct=0.9,
        min_trade_usd=1.0,
        max_price_impact_pct=10.0,
        min_liquidity_usd=1.0,
        max_snapshot_age_seconds=10_000.0,
        min_pool_age_seconds=0.0,
        default_entry_usd=2.0,
        max_cash_fraction_pct=0.0,
    )
    bounds = RiskEngine(loose).entry_bounds(
        "BONK",
        book=book(),
        risk_state=OPEN_STATE,
        snapshot=coin(liquidity_usd=500.0, age_seconds=5_000.0),
        quote=quote(price_impact_pct=8.0),
        volatility_pct=8.0,
        now=NOW,
    )
    # Every one of those inputs would be vetoed under the default parameters.
    assert bounds.permitted
    assert bounds.max_notional_usd == pytest.approx(2.0)


def test_a_higher_gas_cost_tightens_the_cash_caps() -> None:
    cheap, _, _, _ = PortfolioRisk(PARAMS).caps("BONK", book=book(), volatility_pct=8.0)
    pricey, _, _, _ = PortfolioRisk(replace(PARAMS, gas_usd_per_swap=5.0)).caps(
        "BONK", book=book(), volatility_pct=8.0
    )
    cheap_floor = next(c for c in cheap if c.rule == "min_cash_floor_pct")
    pricey_floor = next(c for c in pricey if c.rule == "min_cash_floor_pct")
    assert pricey_floor.max_notional_usd < cheap_floor.max_notional_usd
    assert pricey_floor.max_notional_usd == pytest.approx(895.0)


def test_the_module_exports_only_the_layered_api() -> None:
    """``check`` is gone. A caller that still imports it should fail loudly
    rather than find something with a similar name and a different contract."""
    assert not hasattr(risk, "check")
    assert set(risk.__all__) >= {
        "ContinuousRisk",
        "EligibilityRisk",
        "PortfolioRisk",
        "PreTradeRisk",
        "RiskEngine",
        "RiskLedger",
        "RiskParams",
        "update_ledger",
    }
