"""Mark the book to market, and decide who has to be stopped out.

This module was rewritten against audit **C8**, which named it the single most
dangerous file in the project. The old ``mark()`` carried a position it could
not price **at cost basis**. That is not a conservative default; it is a
falsehood with a specific shape. It displays exactly zero unrealised loss during
the only two events that matter — a rug and a data outage — and because
``stop_loss_breaches`` measured drawdown against that same fabricated value, the
stop was guaranteed not to fire at precisely the moment it existed for. The old
test suite even asserted this behaviour (``test_missing_mark_does_not_crash_and
_carries_at_cost``), which is how a bug survives for months: it had a passing
test describing it.

What replaces it:

* A mark is a :class:`~.types.Mark` — a price with a stated ``basis``
  (``"route" | "mid" | "estimate" | "unavailable"``) and a
  :class:`~.types.Provenance`, or an explicit absence. **Never cost basis.**
* ``PortfolioState.total_value_usd`` is ``None`` when any held position is
  unmarkable, and that ``None`` propagates. A book whose value is unknown must
  not report a confident number, because every downstream percentage — return,
  drawdown, the position cap in ``risk.py`` — would then be computed against a
  figure partly made of cost basis standing in for a price nobody could obtain.
  ``risk.PortfolioRisk`` vetoes entries rather than dividing by it.
* ``PortfolioState.unmarkable`` names the offenders. An unmarkable position is a
  **risk incident**, not a number; ``risk.ContinuousRisk`` turns it into a
  data-health failure and a per-symbol quarantine.

Marks are ranked by how close they are to a number someone would actually
receive:

1. ``"route"`` — the output of a *sell* route quoted at the position's own size.
   This is the only basis that estimates liquidation value, so it alone carries
   no haircut and is the only one ``risk`` will treat as executable.
2. ``"mid"`` — a pool mid price. It is a display number: it is one pool's last
   trade, it does not account for depth, and the audit's pair-switching finding
   means consecutive mids may not even describe the same pool. Haircut applied.
3. ``"estimate"`` — a :class:`~.types.ValuationEstimate`, which exists precisely
   because a route could not be obtained. Haircut applied, never smaller than
   the estimate's own.
4. ``"unavailable"`` — we do not know. This is the honest outcome and it is
   allowed to be the outcome.

``Mark.price_usd`` is always **post-haircut**: it is the number to value the
book at, and ``haircut_pct`` records how much was taken off. Storing the raw mid
and hoping every consumer remembers to discount it is the mistake that produced
``ValuationEstimate.conservative_price_usd`` in the first place.

**The stop, unified.** Audit §11 found two different numbers presented as one:
the engine compared mark value against *cost basis* (which includes entry fees
and gas) while the operator-facing "stop price" was ``avg_entry * 0.85``. On the
funded fixture below that is a $400.21 basis against a $400.00 notional — a
0.05% disagreement, small enough never to be noticed and large enough to make an
operator's screen wrong about when the engine will act. There is now exactly one
definition, :func:`stop_price_usd`, and both the trigger and the display call
it:

    stop_price = (cost_basis_usd / quantity) * (1 - stop_loss_pct)

Cost basis wins over average entry because fees and gas are money that is
genuinely gone: break-even is what you paid, not what the token cost. The
trigger is stated in *price* space rather than percent space so that the number
on the screen is the number being compared, not a re-derivation of it.

**The stop reference.** A stop is an instruction to attempt an exit, and it is
triggered against a reference price. Prefer the executable one. When only a mid
is available the breach is still raised — refusing to stop out because the data
is bad is how a -80% position becomes a -100% one — but it is flagged
``degraded_reference=True`` so the incident is visible rather than silent. An
*unmarkable* position raises no breach at all and instead appears in
``PortfolioState.unmarkable``: you do not force a liquidation on the strength of
data you do not have, you escalate.

Percent conventions, because this module straddles the boundary that has bitten
this codebase before: ``stop_loss_pct`` is a **fraction** (0.15 = -15%), every
``_pct`` on a computed value is a **whole percent** (-15.0 = -15%), and every
knob on :class:`MarkParams` is a **whole percent**.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .types import (
    Fill,
    Mark,
    OrderState,
    PortfolioState,
    Position,
    Provenance,
    Quote,
    Side,
    ValidationError,
    ValuationEstimate,
    finite,
    non_negative,
)

logger = logging.getLogger(__name__)

#: Relative slack on the stop comparison. ``0.15 * 100`` is
#: ``15.000000000000002`` in binary floating point and
#: ``cost_basis * 0.85 / quantity`` need not be bit-identical to
#: ``(cost_basis / quantity) * 0.85``, so a position sitting at exactly the
#: threshold can miss a naive ``<=`` by one ulp. One part in 10^9 of a price is
#: many orders of magnitude larger than that error and many orders of magnitude
#: smaller than any price move anyone would act on.
_STOP_RELATIVE_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class MarkParams:
    """How much to distrust each class of mark, in **whole percent**.

    The haircuts are not calibrated — there is no sample of route-versus-mid
    deviations in this system to calibrate them from, and pretending otherwise
    would be the same error the audit found everywhere else. They are stated as
    what they are: deliberately conservative constants whose only job is to stop
    a non-executable price from ever looking *better* than an executable one,
    because an outage that improves your reported P&L is an outage that reads as
    a trading signal.
    """

    #: Applied to a pool mid. A mid ignores depth entirely: the audit's whole
    #: C8 point is that a displayed price and a price you can sell 100% of a
    #: position into are different numbers.
    mid_haircut_pct: float = 2.0
    #: Floor under a :class:`ValuationEstimate`'s own haircut. The estimate is
    #: allowed to be more pessimistic than this; it is not allowed to be less.
    estimate_min_haircut_pct: float = 5.0
    #: A mark older than this is not a mark. Measured from
    #: ``Provenance.effective_time``, i.e. the *oldest* timestamp we have, so a
    #: source that omits ``event_time`` cannot look fresher than an honest one.
    max_mark_age_seconds: float = 120.0
    #: Whether a mid-basis mark may trigger a stop. Default true, loudly: see
    #: the module docstring. Set false only if you would rather hold an
    #: unquotable position than exit on a display price.
    allow_mid_stop_reference: bool = True

    def __post_init__(self) -> None:
        non_negative(self.mid_haircut_pct, "mid_haircut_pct")
        non_negative(self.estimate_min_haircut_pct, "estimate_min_haircut_pct")
        non_negative(self.max_mark_age_seconds, "max_mark_age_seconds")
        for name in ("mid_haircut_pct", "estimate_min_haircut_pct"):
            if getattr(self, name) >= 100.0:
                raise ValidationError(f"{name} is a whole percent and must be < 100")


DEFAULT_MARK_PARAMS = MarkParams()


@dataclass(frozen=True, slots=True)
class StopBreach:
    """One position at or beyond its stop, with the evidence that put it there.

    Carries ``reference_basis`` and ``degraded_reference`` because audit §11's
    complaint was not that the stop fired, it was that nobody could tell what it
    fired *against*. A journal row that records "stopped out" without recording
    whether the trigger price was executable cannot later be used to work out
    whether the exit was justified.
    """

    symbol: str
    stop_price_usd: float
    reference_price_usd: float
    reference_basis: str
    unrealized_pnl_pct: float
    degraded_reference: bool
    reason: str


def stop_price_usd(position: Position, stop_loss_pct: float) -> float | None:
    """The one stop-price definition. Engine and display both call this.

    ``stop_loss_pct`` is a **fraction** (0.15 means -15%). Returns ``None`` for a
    position with no quantity, because a stop price per unit of nothing is not a
    number and returning 0.0 would render as "stop at $0.00" on an operator's
    screen.

    Measured against cost basis per unit rather than ``avg_entry_price_usd``:
    entry fees and gas are money that is gone, so the price at which the
    position is down 15% *to its owner* is 15% below what was actually paid. The
    two differ by the entry cost — on the project's own $400 BONK fixture,
    $400.21 of basis against $400.00 of notional — which is exactly the
    disagreement §11 identified between the engine and the display.
    """
    quantity = position.quantity
    if quantity <= 0.0 or position.cost_basis_usd <= 0.0:
        return None
    breakeven = position.cost_basis_usd / quantity
    return breakeven * (1.0 - abs(finite(stop_loss_pct, "stop_loss_pct")))


def build_mark(
    symbol: str,
    *,
    route: Quote | None = None,
    mid_price_usd: float | None = None,
    mid_provenance: Provenance | None = None,
    estimate: ValuationEstimate | None = None,
    params: MarkParams = DEFAULT_MARK_PARAMS,
    now: float,
) -> Mark:
    """Choose the best available mark for one symbol, most executable first.

    ``route`` should be a **SELL** quote obtained at the position's own size —
    that is what "what is this worth" means for inventory you intend to be able
    to leave. A BUY route is accepted (it is better than a mid) but it prices the
    wrong side of the book, so it is noted.

    Every candidate must clear the same three gates before it is considered: the
    price must be finite and positive, the provenance must not be quarantined,
    and the observation must be younger than ``max_mark_age_seconds``. A stale
    price is not a conservative price — it is an arbitrary one, and during the
    outages this function exists for it is arbitrary in the optimistic
    direction, because the last value before a collapse is the pre-collapse one.

    Returning an ``"unavailable"`` mark is a normal, expected outcome and is the
    entire point of the rewrite. It is not an error and it is not zero.
    """
    finite(now, "now")

    if route is not None:
        price = route.effective_price_usd
        if price is None or price <= 0.0:
            logger.warning("%s: route quote implies no usable price; ignoring", symbol)
        elif route.is_expired(now):
            logger.warning(
                "%s: route quote expired at %.0f; ignoring", symbol, route.expires_at or 0.0
            )
        elif route.age_seconds(now) > params.max_mark_age_seconds:
            logger.warning(
                "%s: route quote is %.0fs old, limit %.0fs; ignoring",
                symbol,
                route.age_seconds(now),
                params.max_mark_age_seconds,
            )
        else:
            note = (
                None
                if route.side is Side.SELL
                else "BUY-side route: prices the wrong side of the book"
            )
            return Mark(
                symbol=symbol,
                price_usd=price,
                basis="route",
                provenance=Provenance(source="route", receive_time=route.received_at),
                haircut_pct=0.0,
                reason=note,
            )

    if mid_price_usd is not None:
        age = mid_provenance.age_seconds(now) if mid_provenance is not None else None
        usable_source = mid_provenance is None or mid_provenance.usable
        if mid_price_usd <= 0.0:
            logger.warning(
                "%s: mid price %r is not a price; ignoring", symbol, mid_price_usd
            )
        elif not usable_source:
            logger.warning("%s: mid price source is quarantined; ignoring", symbol)
        elif age is not None and age > params.max_mark_age_seconds:
            logger.warning(
                "%s: mid price is %.0fs old, limit %.0fs; ignoring",
                symbol,
                age,
                params.max_mark_age_seconds,
            )
        else:
            return Mark(
                symbol=symbol,
                price_usd=mid_price_usd * (1.0 - params.mid_haircut_pct / 100.0),
                basis="mid",
                provenance=mid_provenance,
                haircut_pct=params.mid_haircut_pct,
                reason=(
                    "pool mid, not an executable route; "
                    f"{params.mid_haircut_pct:.1f}% haircut applied"
                ),
            )

    if estimate is not None and estimate.mid_price_usd is not None:
        if estimate.mid_price_usd <= 0.0:
            logger.warning("%s: valuation estimate is not a price; ignoring", symbol)
        elif now - estimate.at > params.max_mark_age_seconds:
            logger.warning(
                "%s: valuation estimate is %.0fs old, limit %.0fs; ignoring",
                symbol,
                now - estimate.at,
                params.max_mark_age_seconds,
            )
        else:
            haircut = max(estimate.haircut_pct, params.estimate_min_haircut_pct)
            return Mark(
                symbol=symbol,
                price_usd=estimate.mid_price_usd * (1.0 - haircut / 100.0),
                basis="estimate",
                provenance=Provenance(source=estimate.source, receive_time=estimate.at),
                haircut_pct=haircut,
                reason=(
                    f"non-executable valuation ({estimate.reason}); "
                    f"{haircut:.1f}% haircut applied"
                ),
            )

    logger.warning(
        "%s: UNMARKABLE — no route, no usable mid, no valuation estimate. "
        "This is a risk incident: entries are blocked and the position is quarantined.",
        symbol,
    )
    return Mark(
        symbol=symbol,
        price_usd=None,
        basis="unavailable",
        provenance=None,
        haircut_pct=0.0,
        reason="no route, no usable mid, no valuation estimate",
    )


def mark_book(
    *,
    cash_usd: float,
    positions: Mapping[str, Position],
    marks: Mapping[str, Mark],
    realized_pnl_usd: float,
    starting_cash_usd: float,
    fees_paid_usd: float = 0.0,
    gas_paid_usd: float = 0.0,
    now: float,
) -> PortfolioState:
    """Value the book. ``None`` where the value is unknown, and it propagates.

    Deliberately takes plain values rather than a broker. ``mark`` used to reach
    into ``LocalPaperBroker`` for cash and realised P&L, which made the one
    function that must be trustworthy during an outage depend on a mutable
    object with I/O attached. It is now a pure function of its arguments and the
    caller does the reaching.

    A symbol with no entry in ``marks`` is treated as unmarkable, not skipped: a
    caller that forgets to mark a position must not thereby cause it to vanish
    from the book. Positions with zero quantity are ignored entirely — they are
    ledger residue, not exposure, and letting one of them turn
    ``total_value_usd`` into ``None`` would halt entries over an accounting
    artefact.

    Audit C8, restated as an invariant this function guarantees: if
    ``unmarkable`` is non-empty then ``total_value_usd`` and
    ``unrealized_pnl_usd`` are both ``None``. There is no arrangement of inputs
    that produces a confident total over an incompletely marked book.
    """
    finite(now, "now")
    finite(cash_usd, "cash_usd")
    finite(realized_pnl_usd, "realized_pnl_usd")

    values: dict[str, float | None] = {}
    marks_used: dict[str, Mark] = {}
    unmarkable: list[str] = []
    marked_value = 0.0
    marked_cost = 0.0

    for symbol, position in positions.items():
        if position.quantity_atomic == 0:
            values[symbol] = 0.0
            marks_used[symbol] = marks.get(
                symbol,
                Mark(
                    symbol=symbol,
                    price_usd=None,
                    basis="unavailable",
                    provenance=None,
                    reason="zero quantity: no exposure to mark",
                ),
            )
            continue

        mark_ = marks.get(symbol)
        if mark_ is None:
            mark_ = Mark(
                symbol=symbol,
                price_usd=None,
                basis="unavailable",
                provenance=None,
                reason="no mark supplied for a held position",
            )
        marks_used[symbol] = mark_

        if not mark_.usable:
            values[symbol] = None
            unmarkable.append(symbol)
            continue

        value = position.value_usd(mark_.price_usd)
        values[symbol] = value
        if value is not None:
            marked_value += value
            marked_cost += position.cost_basis_usd

    unmarkable.sort()
    fully_marked = not unmarkable

    return PortfolioState(
        ts=now,
        cash_usd=cash_usd,
        positions=dict(positions),
        marks=marks_used,
        position_values_usd=values,
        unrealized_pnl_usd=(marked_value - marked_cost) if fully_marked else None,
        realized_pnl_usd=realized_pnl_usd,
        total_value_usd=(cash_usd + marked_value) if fully_marked else None,
        starting_cash_usd=starting_cash_usd,
        fees_paid_usd=fees_paid_usd,
        gas_paid_usd=gas_paid_usd,
        unmarkable=tuple(unmarkable),
    )


def stop_loss_breaches(
    state: PortfolioState,
    stop_loss_pct: float,
    *,
    params: MarkParams = DEFAULT_MARK_PARAMS,
) -> tuple[StopBreach, ...]:
    """Positions at or beyond the stop, worst drawdown first.

    Worst-first ordering matters when acting on the result: if only some exits
    can be filled this tick, the deepest loser should go first. That property
    survives from the original implementation and is still tested.

    ``stop_loss_pct`` is a fraction from config (0.15). The comparison happens in
    *price* space against :func:`stop_price_usd`, so the number compared is
    literally the number an operator is shown. ``unrealized_pnl_pct`` is carried
    on the breach for the journal, in whole percent.

    Three outcomes, and the distinction between the last two is the C8 fix:

    * marked on a route — a clean breach;
    * marked on a mid or an estimate — still a breach (holding an unquotable
      loser is worse than exiting on an imperfect price) but flagged
      ``degraded_reference``, and suppressible via
      ``MarkParams.allow_mid_stop_reference``;
    * unmarkable — **no breach**. You do not force a liquidation on the strength
      of data you do not have. It surfaces instead in
      ``PortfolioState.unmarkable``, which ``risk.ContinuousRisk`` escalates into
      a data-health failure. The old code reached the same "do not fire"
      conclusion by accident, having already overwritten the position's value
      with its cost basis; here it is a decision with a reason attached.
    """
    breached: list[tuple[float, StopBreach]] = []

    for symbol, position in state.positions.items():
        if position.quantity_atomic == 0:
            continue
        stop = stop_price_usd(position, stop_loss_pct)
        if stop is None:
            continue

        mark_ = state.marks.get(symbol)
        if mark_ is None or not mark_.usable or mark_.price_usd is None:
            logger.warning(
                "%s: unmarkable, so its stop cannot be evaluated. Escalating as a "
                "risk incident instead of forcing an exit blind.",
                symbol,
            )
            continue

        degraded = not mark_.is_executable_basis
        if degraded and not params.allow_mid_stop_reference:
            logger.warning(
                "%s: at/near its stop on a %s-basis mark, but "
                "allow_mid_stop_reference is off, so no breach is raised.",
                symbol,
                mark_.basis,
            )
            continue

        reference = mark_.price_usd
        if reference > stop * (1.0 + _STOP_RELATIVE_EPSILON):
            continue

        pnl_pct = position.unrealized_pnl_pct(reference)
        if pnl_pct is None:
            continue

        if degraded:
            logger.warning(
                "%s: stop triggered on a NON-EXECUTABLE %s-basis reference "
                "($%.10g vs stop $%.10g). No sell route was available to price "
                "this; the exit may not fill near here.",
                symbol,
                mark_.basis,
                reference,
                stop,
            )

        breached.append(
            (
                pnl_pct,
                StopBreach(
                    symbol=symbol,
                    stop_price_usd=stop,
                    reference_price_usd=reference,
                    reference_basis=mark_.basis,
                    unrealized_pnl_pct=pnl_pct,
                    degraded_reference=degraded,
                    reason=(
                        f"{symbol} marked at ${reference:.10g} on a {mark_.basis} basis, "
                        f"at or below its ${stop:.10g} stop "
                        f"({pnl_pct:.2f}% against a ${position.cost_basis_usd:,.2f} cost basis)"
                        + (" — reference is not executable" if degraded else "")
                    ),
                ),
            )
        )

    breached.sort(key=lambda item: item[0])
    return tuple(breach for _, breach in breached)


def landed_exits(fills: Sequence[Fill]) -> tuple[Fill, ...]:
    """The SELL fills that actually happened.

    Audit §11, the "failed stop reported as exit" row: a stop fill returning
    ``failed=True`` was still appended to ``stop_loss_exits``, so the CLI
    reported "exited" while the inventory was still there — a false sense of
    safety at the exact moment the operator most needs an accurate one. The
    audit's required detection is ``landed && reconciled``; this is the one
    predicate for "an exit occurred" and everything that counts exits must go
    through it.

    ``EXPIRED`` counts as failure for the same reason ``FAILED`` does: nothing
    moved. ``SUBMITTED`` is deliberately excluded — it is an attempt with an
    unknown outcome, and an unknown outcome is not an exit.
    """
    return tuple(
        f
        for f in fills
        if f.side is Side.SELL
        and not f.failed
        and f.state in (OrderState.LANDED, OrderState.RECONCILED)
    )


def failed_attempts(fills: Sequence[Fill]) -> tuple[Fill, ...]:
    """Attempts that cost gas and moved no inventory.

    Feeds ``risk.ContinuousRisk``'s consecutive-failure breaker. Kept here
    beside :func:`landed_exits` so the two definitions cannot drift apart.
    """
    return tuple(f for f in fills if f.failed)


__all__ = [
    "DEFAULT_MARK_PARAMS",
    "MarkParams",
    "StopBreach",
    "build_mark",
    "failed_attempts",
    "landed_exits",
    "mark_book",
    "stop_loss_breaches",
    "stop_price_usd",
]
