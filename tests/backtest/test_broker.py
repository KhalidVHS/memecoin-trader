"""Tests for :mod:`memetrader.backtest.broker`.

Unlike ``test_engine.py``, these tests do not drive a real
:class:`~memetrader.execution.fill_models.BarExecutionModel` through bar
search — they use a small hand-built ``FakeExecutionModel`` that implements
the ``ExecutionModel`` protocol (``price``/``fill``/``fidelity``) with
injectable callables, so each test can script the exact ``Quote``/
``ExecutionReport``/``NoRoute`` outcome it needs and exercise
``SimulatedBroker`` in isolation:

* C3 — a risk-driven resize forces a *fresh* re-quote, never a rescale of
  the quote already in hand (``resize_pairs`` records the equal pair).
* A settled ``Fill``'s ``quote_fingerprint`` must match the quote the order
  was approved against, or ``settle`` raises ``QuoteBindingError``.
* ``NoRoute``/degraded (``None``) quotes short-circuit ``_attempt`` with a
  descriptive ``veto_reason`` and no ``approved`` order.
* ``settle()`` refuses to generate a fill at or before ``decided_at``
  (look-ahead guard) and turns a ``NoRoute`` from ``fill()`` into a
  synthetic ``FAILED`` ``ExecutionReport`` rather than propagating.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from memetrader import ids
from memetrader.backtest.broker import QuoteBindingError, SimulatedBroker
from memetrader.execution.interfaces import ApprovedOrder, NoRoute
from memetrader.histdata.point_in_time import PointInTimeState
from memetrader.risk import RiskEngine, RiskParams
from memetrader.types import (
    CoinSnapshot,
    ExecutionReport,
    FidelityTier,
    Fill,
    OrderIntent,
    OrderState,
    PoolRef,
    PortfolioState,
    PriceLadder,
    Provenance,
    Quote,
    RiskBounds,
    RiskState,
    Side,
    TokenMeta,
    TxnCounts,
    ValidationError,
)

_EMPTY_TXNS = TxnCounts(buys=None, sells=None)

T0 = 1_700_000_000.0
USD = TokenMeta(mint="USDC", decimals=6, source="test")
TOKEN = TokenMeta(mint="TOKEN", decimals=9, source="test")

# ``FakeExecutionModel`` never dereferences ``state`` (its price_fn/fill_fn
# only ever see the arguments each test scripts), so these tests have no real
# ``PointInTimeState`` to pass. ``broker.py``'s ``attempt_entry``/``_attempt``/
# ``settle`` all declare ``state: PointInTimeState`` as non-Optional — this
# file must not weaken that signature (it belongs to a different owner), so a
# precise `cast` documents "this is deliberately not a real state, and every
# call site here relies on the fake execution model never touching it" rather
# than silencing the checker with a bare ignore.
_NO_STATE = cast(PointInTimeState, None)


def _intent(
    symbol: str, *, usd: float, now: float, intent_id: str = "intent-1"
) -> OrderIntent:
    amt = int(usd * 1_000_000)
    return OrderIntent(
        intent_id=intent_id,
        decision_id=None,
        action_id=None,
        run_id="test-run",
        ts=now,
        symbol=symbol,
        side=Side.BUY,
        in_amount_atomic=amt,
        max_in_amount_atomic=amt,
        source="strategy",
        reason="scripted",
    )


def _quote(
    symbol: str,
    *,
    in_amount_atomic: int,
    out_amount_atomic: int | None = None,
    now: float = T0,
    context_slot: int | None = 1,
) -> Quote:
    out = out_amount_atomic if out_amount_atomic is not None else in_amount_atomic * 1_000
    fp = ids.quote_fingerprint(
        side=str(Side.BUY),
        input_mint=USD.mint,
        output_mint=TOKEN.mint,
        in_amount_atomic=in_amount_atomic,
        out_amount_atomic=out,
        slot=context_slot,
    )
    return Quote(
        symbol=symbol,
        side=Side.BUY,
        input_token=USD,
        output_token=TOKEN,
        in_amount_atomic=in_amount_atomic,
        out_amount_atomic=out,
        min_out_amount_atomic=out,
        price_impact_pct=0.0,
        route_labels=("fake",),
        fingerprint=fp,
        requested_at=now,
        received_at=now,
        context_slot=context_slot,
    )


def _book(*, cash_usd: float = 1_000.0, ts: float = T0) -> PortfolioState:
    return PortfolioState(
        ts=ts,
        cash_usd=cash_usd,
        positions={},
        marks={},
        position_values_usd={},
        unrealized_pnl_usd=0.0,
        realized_pnl_usd=0.0,
        total_value_usd=cash_usd,
        starting_cash_usd=cash_usd,
    )


def _risk_state(ts: float = T0) -> RiskState:
    return RiskState(ts=ts)


def _snapshot(
    symbol: str, mint: str, *, price_usd: float = 1.0, liquidity_usd: float = 1_000_000.0
) -> CoinSnapshot:
    pool = PoolRef(
        pair_address=f"pool-{symbol}",
        dex_id="test",
        base_mint=mint,
        quote_mint="",
        quote_symbol="",
        trusted_quote=True,
    )
    provenance = Provenance(
        source="test", receive_time=T0, event_time=T0, available_time=T0
    )
    return CoinSnapshot(
        symbol=symbol,
        mint=mint,
        price_usd=price_usd,
        liquidity_usd=liquidity_usd,
        volume_24h_usd=None,
        volume_1h_usd=None,
        fdv_usd=None,
        price_change=PriceLadder(m5=None, h1=None, h6=None, h24=None),
        txns_m5=_EMPTY_TXNS,
        txns_h1=_EMPTY_TXNS,
        txns_h24=_EMPTY_TXNS,
        pool=pool,
        provenance=provenance,
    )


_PERMISSIVE: dict[str, Any] = {
    "require_known_pool_age": False,
    "min_liquidity_usd": 0.0,
    "min_seconds_between_entries": 0.0,
    "post_stop_quarantine_seconds": 0.0,
    "max_snapshot_age_seconds": 1.0e9,
    "require_volatility_estimate": False,
}


def _risk_engine(universe: frozenset[str], **overrides) -> RiskEngine:
    kwargs = dict(_PERMISSIVE)
    kwargs.update(overrides)
    return RiskEngine(params=RiskParams(universe=universe, **kwargs))


def _approved_bounds(symbol: str) -> RiskBounds:
    return RiskBounds(symbol=symbol, side=Side.BUY, max_notional_usd=1_000.0)


@dataclass
class FakeExecutionModel:
    """A hand-scripted stand-in for a real ``ExecutionModel``.

    ``price_fn``/``fill_fn`` are called with the same keyword arguments the
    protocol specifies; each defaults to a trivial pass-through so a test
    only overrides what it actually needs to control.
    """

    price_fn: Callable[..., Quote | None] | None = None
    fill_fn: Callable[..., ExecutionReport] | None = None
    fidelity_tier: FidelityTier = FidelityTier.TIER_1
    price_calls: list = field(default_factory=list)
    fill_calls: list = field(default_factory=list)

    @property
    def fidelity(self) -> FidelityTier:
        return self.fidelity_tier

    def price(self, *, intent: OrderIntent, state, now: float) -> Quote | None:
        self.price_calls.append(intent)
        if self.price_fn is None:
            return _quote(intent.symbol, in_amount_atomic=intent.in_amount_atomic, now=now)
        return self.price_fn(intent=intent, state=state, now=now)

    def fill(self, *, order: ApprovedOrder, state, now: float) -> ExecutionReport:
        self.fill_calls.append(order)
        if self.fill_fn is None:
            return _default_report(order, now=now)
        return self.fill_fn(order=order, state=state, now=now)


def _default_report(
    order: ApprovedOrder, *, now: float, fingerprint: str | None = None
) -> ExecutionReport:
    quote = order.quote
    fp = fingerprint if fingerprint is not None else quote.fingerprint
    fill = Fill(
        fill_id=ids.new_fill_id(),
        order_id="order-1",
        intent_id=order.intent.intent_id,
        decision_id=None,
        ts=now,
        symbol=order.intent.symbol,
        side=Side.BUY,
        state=OrderState.LANDED,
        in_amount_atomic=quote.in_amount_atomic,
        out_amount_atomic=quote.out_amount_atomic,
        token_amount_atomic=quote.out_amount_atomic,
        token_decimals=TOKEN.decimals,
        quote_fingerprint=fp,
        price_usd=1.0,
        notional_usd=USD.to_ui(quote.in_amount_atomic),
        price_impact_pct=0.0,
        pool_fee_usd=0.0,
        gas_usd=0.21,
    )
    return ExecutionReport(
        report_id=ids.new_fill_id(),
        intent_id=order.intent.intent_id,
        order_id="order-1",
        state=fill.state,
        ts=fill.ts,
        fidelity=FidelityTier.TIER_1,
        fill=fill,
    )


# ---------------------------------------------------------------------------
# attempt_entry — happy path
# ---------------------------------------------------------------------------


def test_attempt_entry_permitted_under_permissive_risk():
    symbol = "OK"
    intent = _intent(symbol, usd=50.0, now=T0)
    model = FakeExecutionModel()
    broker = SimulatedBroker(
        execution_model=model, risk_engine=_risk_engine(frozenset({symbol}))
    )

    attempt = broker.attempt_entry(
        intent,
        state=_NO_STATE,
        book=_book(),
        risk_state=_risk_state(),
        snapshot=_snapshot(symbol, symbol),
        now=T0,
    )

    assert attempt.permitted
    assert attempt.approved is not None
    assert attempt.approved.quote.fingerprint == attempt.quote.fingerprint
    assert attempt.approved.intent.in_amount_atomic == intent.in_amount_atomic


def test_attempt_entry_risk_veto_short_circuits_before_pricing():
    symbol = "VETOED"
    # Symbol not in the universe -> entry_bounds refuses outright.
    intent = _intent(symbol, usd=50.0, now=T0)
    model = FakeExecutionModel()
    broker = SimulatedBroker(
        execution_model=model, risk_engine=_risk_engine(frozenset({"OTHER"}))
    )

    attempt = broker.attempt_entry(
        intent, state=_NO_STATE, book=_book(), risk_state=_risk_state(), now=T0
    )

    assert not attempt.permitted
    assert attempt.veto_reason
    assert attempt.quote is None
    assert not model.price_calls, "price() must never be called once bounds refuse"


# ---------------------------------------------------------------------------
# NoRoute / degraded route
# ---------------------------------------------------------------------------


def test_attempt_entry_no_route_from_price_is_a_veto():
    symbol = "NOROUTE"

    def price_fn(*, intent, state, now):
        raise NoRoute("no liquidity")

    model = FakeExecutionModel(price_fn=price_fn)
    broker = SimulatedBroker(
        execution_model=model, risk_engine=_risk_engine(frozenset({symbol}))
    )
    intent = _intent(symbol, usd=50.0, now=T0)

    attempt = broker.attempt_entry(
        intent,
        state=_NO_STATE,
        book=_book(),
        risk_state=_risk_state(),
        snapshot=_snapshot(symbol, symbol),
        now=T0,
    )

    assert not attempt.permitted
    assert attempt.veto_reason.startswith("no_route:")


def test_attempt_entry_degraded_route_is_a_veto():
    symbol = "DEGRADED"

    def price_fn(*, intent, state, now):
        return None

    model = FakeExecutionModel(price_fn=price_fn)
    broker = SimulatedBroker(
        execution_model=model, risk_engine=_risk_engine(frozenset({symbol}))
    )
    intent = _intent(symbol, usd=50.0, now=T0)

    attempt = broker.attempt_entry(
        intent,
        state=_NO_STATE,
        book=_book(),
        risk_state=_risk_state(),
        snapshot=_snapshot(symbol, symbol),
        now=T0,
    )

    assert not attempt.permitted
    assert attempt.veto_reason == "degraded_route"


# ---------------------------------------------------------------------------
# C3 — risk resize forces a fresh re-quote, never a rescale
# ---------------------------------------------------------------------------


def test_resize_and_requote_never_rescales_the_original_quote():
    symbol = "RESIZE"
    # max_position_pct is tiny relative to a $1000 book, so entry_bounds'
    # max_notional_usd will land far below the $500 the intent asks for.
    model = FakeExecutionModel()
    broker = SimulatedBroker(
        execution_model=model,
        risk_engine=_risk_engine(frozenset({symbol}), max_position_pct=0.01),
    )
    intent = _intent(symbol, usd=500.0, now=T0)

    attempt = broker.attempt_entry(
        intent,
        state=_NO_STATE,
        book=_book(),
        risk_state=_risk_state(),
        snapshot=_snapshot(symbol, symbol),
        now=T0,
    )

    assert attempt.permitted
    assert broker.resize_pairs, "expected a risk-driven resize"
    for approved_atomic, quoted_atomic in broker.resize_pairs:
        assert approved_atomic == quoted_atomic
    # price() must have been called twice: once for the original ask, once
    # for the fresh working_intent at the clamped size.
    assert len(model.price_calls) == 2
    first_call, second_call = model.price_calls
    assert first_call.in_amount_atomic == intent.in_amount_atomic
    assert second_call.in_amount_atomic < intent.in_amount_atomic
    # The approved order is bound to the fresh quote's exact size, not a
    # rescale of the first quote.
    assert attempt.approved.intent.in_amount_atomic == second_call.in_amount_atomic
    assert attempt.approved.quote.in_amount_atomic == second_call.in_amount_atomic


def test_resize_to_zero_size_is_a_veto_not_a_zero_order():
    symbol = "ZERO"
    # A vanishingly small bound relative to the quoted notional clamps to
    # zero atomic units — this must refuse rather than attempt a zero-size
    # re-quote. Exercises the shared _attempt() path directly (white-box)
    # with a hand-built RiskBounds, since coaxing a real RiskParams-derived
    # bound down to sub-atomic-unit size is impractical (RiskParams'
    # fractions are validated to be > 0).
    intent = _intent(symbol, usd=0.000002, now=T0)  # 2 micro-usd atomic
    model = FakeExecutionModel()
    broker = SimulatedBroker(
        execution_model=model, risk_engine=_risk_engine(frozenset({symbol}))
    )
    tiny_bounds = RiskBounds(symbol=symbol, side=Side.BUY, max_notional_usd=1e-7)

    attempt = broker._attempt(intent, bounds=tiny_bounds, state=_NO_STATE, now=T0)

    assert not attempt.permitted
    assert "leaves no positive size to re-quote" in attempt.veto_reason
    assert not broker.resize_pairs


# ---------------------------------------------------------------------------
# settle() — look-ahead guard
# ---------------------------------------------------------------------------


def test_settle_refuses_a_fill_at_or_before_decided_at():
    symbol = "LATE"
    model = FakeExecutionModel()
    broker = SimulatedBroker(
        execution_model=model, risk_engine=_risk_engine(frozenset({symbol}))
    )
    intent = _intent(symbol, usd=50.0, now=T0)
    quote = _quote(symbol, in_amount_atomic=intent.in_amount_atomic, now=T0)
    approved = ApprovedOrder(
        intent=intent, bounds=_approved_bounds(symbol), quote=quote, decided_at=T0
    )

    try:
        broker.settle(approved, state=_NO_STATE, now=T0)
    except ValidationError:
        pass
    else:
        raise AssertionError("settle() must refuse now == decided_at")

    try:
        broker.settle(approved, state=_NO_STATE, now=T0 - 1.0)
    except ValidationError:
        pass
    else:
        raise AssertionError("settle() must refuse now < decided_at")


# ---------------------------------------------------------------------------
# settle() — NoRoute from fill() becomes a synthetic FAILED report
# ---------------------------------------------------------------------------


def test_settle_catches_no_route_and_returns_synthetic_failed_report():
    symbol = "VANISHED"

    def fill_fn(*, order, state, now):
        raise NoRoute("route disappeared")

    model = FakeExecutionModel(fill_fn=fill_fn)
    broker = SimulatedBroker(
        execution_model=model, risk_engine=_risk_engine(frozenset({symbol}))
    )
    intent = _intent(symbol, usd=50.0, now=T0)
    quote = _quote(symbol, in_amount_atomic=intent.in_amount_atomic, now=T0)
    approved = ApprovedOrder(
        intent=intent, bounds=_approved_bounds(symbol), quote=quote, decided_at=T0
    )

    report = broker.settle(approved, state=_NO_STATE, now=T0 + 1.0)

    assert report.state is OrderState.FAILED
    assert report.fill is None
    assert "route disappeared" in report.reason


# ---------------------------------------------------------------------------
# settle() — a settled fill's quote_fingerprint must match the approved quote
# ---------------------------------------------------------------------------


def test_settle_raises_quote_binding_error_on_fingerprint_mismatch():
    symbol = "MISMATCH"
    intent = _intent(symbol, usd=50.0, now=T0)
    quote = _quote(symbol, in_amount_atomic=intent.in_amount_atomic, now=T0)
    approved = ApprovedOrder(
        intent=intent, bounds=_approved_bounds(symbol), quote=quote, decided_at=T0
    )

    def fill_fn(*, order, state, now):
        return _default_report(order, now=now, fingerprint="not-the-real-fingerprint")

    model = FakeExecutionModel(fill_fn=fill_fn)
    broker = SimulatedBroker(
        execution_model=model, risk_engine=_risk_engine(frozenset({symbol}))
    )

    try:
        broker.settle(approved, state=_NO_STATE, now=T0 + 1.0)
    except QuoteBindingError:
        pass
    else:
        raise AssertionError(
            "settle() must raise QuoteBindingError on a fingerprint mismatch"
        )


def test_settle_succeeds_when_fingerprint_matches():
    symbol = "MATCH"
    intent = _intent(symbol, usd=50.0, now=T0)
    quote = _quote(symbol, in_amount_atomic=intent.in_amount_atomic, now=T0)
    approved = ApprovedOrder(
        intent=intent, bounds=_approved_bounds(symbol), quote=quote, decided_at=T0
    )
    model = FakeExecutionModel()
    broker = SimulatedBroker(
        execution_model=model, risk_engine=_risk_engine(frozenset({symbol}))
    )

    report = broker.settle(approved, state=_NO_STATE, now=T0 + 1.0)

    assert report.state is OrderState.LANDED
    assert report.fill is not None
    assert report.fill.quote_fingerprint == quote.fingerprint
