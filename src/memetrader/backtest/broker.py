"""The simulated broker — turns an ``OrderIntent`` into an ``ExecutionReport``.

This is the backtest analogue of the live paper broker (``src/memetrader/
broker.py``), but it is not that file: the live broker talks to a real venue
through ``ExecutionVenue``-shaped clients, while this one drives the frozen
``ExecutionModel`` implementations in ``execution/fill_models.py`` directly,
because none of them actually need the submit/settle split (see
``execution/interfaces.py``'s ``ExecutionVenue`` docstring and
``fill_models.py``: every concrete model does both steps synchronously inside
``.fill()``).

Two audit findings this module exists to close, both from
``docs/BACKTEST-CONTRACTS.md``:

* **C3 — a risk resize must force a new quote, never a rescale.** When the
  size a strategy asked for is quoted, and the resulting notional exceeds what
  ``RiskEngine.entry_bounds``/``exit_bounds`` permits, the fix is not to scale
  the existing ``Quote``'s numbers down. It is to build a *new* ``OrderIntent``
  at the bound's exact size and re-price it from scratch, because a route's
  price impact is not linear and a rescaled quote describes a swap nobody ever
  priced. ``_attempt`` below does exactly this, and records every
  ``(approved_atomic, quoted_atomic)`` pair it produces on ``resize_pairs`` so
  ``backtest/invariants.check_risk_resize_requotes`` can assert the two always
  match.
* **Binding a fill to the quote it was priced against.** ``settle`` recomputes
  ``ids.quote_fingerprint`` from the ``ApprovedOrder``'s bound quote and
  refuses (raises ``QuoteBindingError``) if the fingerprint the execution model
  attached to the settled ``Fill`` does not match. A model that filled the
  order against some other route than the one it was approved for is a bug in
  that model, not something this broker silently accepts.

Nothing here mutates a ``BacktestLedger``. ``settle`` returns an
``ExecutionReport``; applying it to the ledger is ``engine.py``'s job, because
only the engine knows the tick order in which that must happen (contracts
§5, step 9).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from memetrader import ids
from memetrader.execution.interfaces import ApprovedOrder, ExecutionModel, NoRoute
from memetrader.risk import EMPTY_LEDGER, RiskEngine, RiskLedger
from memetrader.types import (
    CoinSnapshot,
    ExecutionReport,
    Forecast,
    OrderIntent,
    OrderState,
    PortfolioState,
    Quote,
    RiskBounds,
    RiskState,
    ValidationError,
    ValuationEstimate,
)

if TYPE_CHECKING:
    from memetrader.histdata.point_in_time import PointInTimeState

__all__ = ["BrokerAttempt", "QuoteBindingError", "SimulatedBroker"]


class QuoteBindingError(RuntimeError):
    """A settled fill's quote fingerprint does not match the quote the order
    was approved against.

    Raised by :meth:`SimulatedBroker.settle`, never swallowed — like
    ``backtest.invariants.InvariantBreach``, this is a structural violation of
    audit C3, not a recoverable condition. A caller that catches this broadly
    has reintroduced the exact bug the fingerprint check exists to catch.
    """


@dataclass(frozen=True, slots=True)
class BrokerAttempt:
    """The full record of one attempt to turn an intent into an approved order.

    ``approved`` is ``None`` whenever the attempt did not reach a bindable
    order — risk vetoed it outright (``bounds.permitted`` is ``False``), the
    execution model found no route at all (``NoRoute``), the model returned a
    degraded ``None`` price, or ``RiskEngine.confirm_quote`` refused the final
    quote. ``bounds`` and ``quote`` are populated as far as the attempt got,
    so a caller can always see *why* it stopped even on failure — that is what
    ``veto_reason`` summarizes in one string for logging/tests.

    ``intent`` is the *working* intent actually priced: identical to what the
    caller passed in unless a risk resize forced a smaller one (audit C3).
    """

    intent: OrderIntent
    bounds: RiskBounds
    quote: Quote | None = None
    approved: ApprovedOrder | None = None
    veto_reason: str = ""

    @property
    def permitted(self) -> bool:
        return self.approved is not None


def _clamp_in_amount(intent: OrderIntent, quote: Quote, max_notional_usd: float) -> int:
    """The largest ``in_amount_atomic`` consistent with ``max_notional_usd``,
    assuming the quote's realized price holds at the smaller size.

    This is only ever used to pick the size for a *fresh* re-quote — the
    execution model, not this arithmetic, decides the actual price at that
    size. A linear scale-down of ``quote.usd_notional`` is therefore a
    starting point for the next ``price()`` call, never something filled
    against directly (that would be exactly the rescale audit C3 forbids).
    """
    notional = quote.usd_notional
    if notional <= 0:
        return 0
    scalar = max_notional_usd / notional
    if scalar <= 0:
        return 0
    clamped = int(intent.in_amount_atomic * scalar)
    # Floor arithmetic only (contracts §0): never round up into a size that
    # was not actually approved. Never exceed the original ask either — a
    # clamp can only shrink an order.
    clamped = min(clamped, intent.in_amount_atomic)
    return max(clamped, 0)


@dataclass(slots=True)
class SimulatedBroker:
    """Routes approved intents through one ``ExecutionModel``.

    ``resize_pairs`` accumulates every ``(approved_atomic, quoted_atomic)``
    pair produced by a risk-driven resize-and-requote, across every attempt
    this instance has handled. It feeds
    ``backtest.invariants.check_risk_resize_requotes`` directly and by
    construction the two elements of every pair are equal (see ``_attempt``),
    which is the property that check asserts.
    """

    execution_model: ExecutionModel
    risk_engine: RiskEngine = field(default_factory=RiskEngine)
    _resize_pairs: list[tuple[int, int]] = field(
        default_factory=list, init=False, repr=False
    )

    @property
    def resize_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._resize_pairs)

    # -- entries -------------------------------------------------------------

    def attempt_entry(
        self,
        intent: OrderIntent,
        *,
        state: PointInTimeState,
        book: PortfolioState,
        risk_state: RiskState,
        snapshot: CoinSnapshot | None = None,
        quote: Quote | None = None,
        valuation: ValuationEstimate | None = None,
        forecast: Forecast | None = None,
        volatility_pct: float | None = None,
        ledger: RiskLedger = EMPTY_LEDGER,
        now: float,
    ) -> BrokerAttempt:
        bounds = self.risk_engine.entry_bounds(
            intent.symbol,
            book=book,
            risk_state=risk_state,
            snapshot=snapshot,
            quote=quote,
            valuation=valuation,
            forecast=forecast,
            volatility_pct=volatility_pct,
            ledger=ledger,
            now=now,
        )
        return self._attempt(intent, bounds=bounds, state=state, now=now)

    # -- exits -----------------------------------------------------------

    def attempt_exit(
        self,
        intent: OrderIntent,
        *,
        state: PointInTimeState,
        book: PortfolioState,
        risk_state: RiskState,
        snapshot: CoinSnapshot | None = None,
        forced: bool = False,
        now: float,
    ) -> BrokerAttempt:
        bounds = self.risk_engine.exit_bounds(
            intent.symbol,
            book=book,
            risk_state=risk_state,
            snapshot=snapshot,
            forced=forced,
            now=now,
        )
        return self._attempt(intent, bounds=bounds, state=state, now=now)

    # -- shared quote/confirm/bind path ---------------------------------

    def _attempt(
        self,
        intent: OrderIntent,
        *,
        bounds: RiskBounds,
        state: PointInTimeState,
        now: float,
    ) -> BrokerAttempt:
        if not bounds.permitted:
            return BrokerAttempt(
                intent=intent,
                bounds=bounds,
                veto_reason="; ".join(bounds.vetoes) or "not permitted",
            )

        try:
            quote = self.execution_model.price(intent=intent, state=state, now=now)
        except NoRoute as exc:
            return BrokerAttempt(
                intent=intent, bounds=bounds, veto_reason=f"no_route: {exc.reason}"
            )
        if quote is None:
            return BrokerAttempt(intent=intent, bounds=bounds, veto_reason="degraded_route")

        working_intent = intent
        if quote.usd_notional > bounds.max_notional_usd:
            # The size the strategy asked for prices out above what risk will
            # permit. Audit C3: clamp to the bound and re-quote FRESH — never
            # rescale the quote already in hand, which would describe a swap
            # nobody ever priced.
            clamped_atomic = _clamp_in_amount(intent, quote, bounds.max_notional_usd)
            if clamped_atomic <= 0:
                return BrokerAttempt(
                    intent=intent,
                    bounds=bounds,
                    quote=quote,
                    veto_reason=(
                        f"risk bound {bounds.max_notional_usd} usd leaves no positive "
                        "size to re-quote at"
                    ),
                )
            working_intent = replace(
                intent,
                in_amount_atomic=clamped_atomic,
                max_in_amount_atomic=min(intent.max_in_amount_atomic, clamped_atomic),
            )
            try:
                quote = self.execution_model.price(
                    intent=working_intent, state=state, now=now
                )
            except NoRoute as exc:
                return BrokerAttempt(
                    intent=working_intent,
                    bounds=bounds,
                    veto_reason=f"no_route_after_resize: {exc.reason}",
                )
            if quote is None:
                return BrokerAttempt(
                    intent=working_intent,
                    bounds=bounds,
                    veto_reason="degraded_route_after_resize",
                )
            # By construction the fresh quote is priced at working_intent's
            # exact size, so this pair is always equal — that equality is
            # the property check_risk_resize_requotes asserts.
            self._resize_pairs.append(
                (working_intent.in_amount_atomic, quote.in_amount_atomic)
            )

        confirmed = self.risk_engine.confirm_quote(
            bounds, quote, notional_usd=quote.usd_notional, now=now
        )
        if not confirmed.permitted:
            return BrokerAttempt(
                intent=working_intent,
                bounds=confirmed,
                quote=quote,
                veto_reason="; ".join(confirmed.vetoes) or "quote refused at bind time",
            )

        approved = ApprovedOrder(
            intent=working_intent, bounds=confirmed, quote=quote, decided_at=now
        )
        return BrokerAttempt(
            intent=working_intent, bounds=confirmed, quote=quote, approved=approved
        )

    # -- settlement -------------------------------------------------------

    def settle(
        self, approved: ApprovedOrder, *, state: PointInTimeState, now: float
    ) -> ExecutionReport:
        """Fill (or fail) ``approved`` at ``now``.

        ``now`` must be strictly after ``approved.decided_at`` — a fill at or
        before the decision timestamp is look-ahead (contracts §5's headline
        rule). The caller (``engine.py``) is expected to have advanced the
        clock past a latency-derived ready time before calling this.
        """
        if now <= approved.decided_at:
            raise ValidationError(
                f"settle() called with now={now} <= decided_at={approved.decided_at} — "
                "a fill may not be generated at or before the decision timestamp"
            )

        try:
            report = self.execution_model.fill(order=approved, state=state, now=now)
        except NoRoute as exc:
            return ExecutionReport(
                report_id=ids.new_fill_id(),
                intent_id=approved.intent.intent_id,
                order_id=ids.new_order_id(),
                state=OrderState.FAILED,
                ts=now,
                fidelity=self.execution_model.fidelity,
                fill=None,
                costs=None,
                reason=f"route disappeared between price() and fill(): {exc.reason}",
            )

        if report.fill is not None:
            quote = approved.quote
            recomputed = ids.quote_fingerprint(
                side=str(quote.side),
                input_mint=quote.input_token.mint,
                output_mint=quote.output_token.mint,
                in_amount_atomic=quote.in_amount_atomic,
                out_amount_atomic=quote.out_amount_atomic,
                slot=quote.context_slot,
            )
            fill_fingerprint = report.fill.quote_fingerprint
            if recomputed != quote.fingerprint or fill_fingerprint != quote.fingerprint:
                raise QuoteBindingError(
                    f"settled fill {report.fill.fill_id} carries quote_fingerprint "
                    f"{report.fill.quote_fingerprint!r} but the approved order was bound "
                    f"to {quote.fingerprint!r} (recomputed: {recomputed!r}) — refusing to "
                    "accept a fill that does not match the quote it was priced against"
                )

        return report
