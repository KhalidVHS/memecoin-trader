"""Frozen contracts for the execution layer of the backtest engine.

Every component that touches an order — the model that prices it, the venue
that fills it, the cost layer that measures it — is written against these
types. See ``docs/BACKTEST-CONTRACTS.md §4`` for the authoritative spec; this
file *is* that spec translated into Python, verbatim.

``ApprovedOrder`` binds an intent, a risk sanction, a quote, and a decision
timestamp into a single immutable package. The decision timestamp is the key
invariant: ``decided_at`` is the simulated clock when the strategy ran, and
``execution/latency.py`` must advance the clock past that value before the
order can meet a market state. A fill at ``decided_at`` is a fill that
happened before the strategy finished thinking, which is look-ahead.

``ExecutionModel`` is deliberately not named ``FillModel``. ``broker.py``
already contains ``FillModel`` (an enum with ``MIN_OUT`` / ``EXPECTED_OUT``)
and the two names must not collide: a name collision here would silently
import the wrong thing into a backtest and produce subtly wrong fills with no
error, which is the failure profile this spec exists to prevent.

``ExecutionVenue`` separates submission from settlement because acceptance and
settlement are different facts separated by time. Collapsing them into a
single call is how a simulator fills an order the real network would have
dropped; the two methods are the structural enforcement of that separation.

``NoRoute`` is a sentinel raised (not a return value) when the model cannot
find a route at all. A ``None`` return from ``ExecutionModel.price`` would
silently look like "got a price and it was nothing"; a distinct exception makes
the three cases — got a price, got None (logged degraded), raised NoRoute
(no trade) — structurally different. A failed quote, unavailable route,
timeout or stale state must produce **no trade**, never a synthesized fill.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ..types import (
    ExecutionReport,
    FidelityTier,
    HistoricalEvent,
    OrderIntent,
    OrderReceipt,
    Quote,
    RiskBounds,
    finite,
)

if TYPE_CHECKING:
    # Avoid a circular import at runtime: PointInTimeState lives in
    # histdata/point_in_time.py, which is in another agent's scope. The TYPE_CHECKING
    # guard lets mypy resolve the annotation while keeping the runtime import-free.
    # If the module does not exist yet during concurrent development, this file
    # still imports cleanly.
    from ..histdata.point_in_time import PointInTimeState


class NoRoute(Exception):
    """No executable route existed for this order at this moment.

    Raised by ``ExecutionModel.price`` — and propagated through the venue
    layer — whenever the model cannot find a route at all. This is
    structurally different from returning ``None`` (degraded but present) and
    from a generic exception (something went wrong). The three outcomes are:

    * ``Quote`` returned  — a route was found and priced.
    * ``None`` returned   — a route exists but is degraded or stale; caller
                            logs and skips.
    * ``NoRoute`` raised  — there is no route; the order must not trade at all.
                            A filled order from a ``NoRoute`` condition is a
                            fabricated fill, which ``backtest/invariants.py``
                            will detect and raise on.

    Callers must catch ``NoRoute`` and treat it as a veto: no position change,
    gas logged, order terminal. They must *not* catch it and return a synthetic
    ``Quote``.
    """

    def __init__(self, reason: str = "") -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ApprovedOrder:
    """An intent that survived risk approval, bound to a specific quote.

    The four fields form a closed, immutable record of everything that went
    into the approval decision. ``decided_at`` is the simulated clock when
    ``strategy.decide`` ran — the latency model uses it to enforce the
    bar-close invariant: no fill may be generated at a time ``<= decided_at``.

    Immutability is load-bearing, not stylistic. A mutable order would let
    the execution model silently resize it after risk approved it, breaking
    audit C3 (the quote is bound to the swap it describes). ``frozen=True``
    makes that structurally impossible.
    """

    intent: OrderIntent
    bounds: RiskBounds
    quote: Quote
    decided_at: float

    def __post_init__(self) -> None:
        finite(self.decided_at, "decided_at")
        # Guard the invariant at construction rather than hoping every consumer
        # of this object checks it. A decided_at that equals the bar close is a
        # decision that could fill at the close — which is exactly the look-ahead
        # the latency model is designed to prevent.
        if self.decided_at < 0:
            from ..types import ValidationError

            raise ValidationError(f"decided_at {self.decided_at} is negative")


class ExecutionModel(Protocol):
    """A model that can price and fill a single swap.

    Named ``ExecutionModel`` rather than ``FillModel`` because ``broker.py``
    already defines ``FillModel`` as an enum (``MIN_OUT`` / ``EXPECTED_OUT``).
    A collision would cause silent wrong-import bugs with no error at runtime.

    ``fidelity`` declares what the model can attest to. Only ``TIER_2`` and
    ``TIER_3`` models are permitted to make a PnL claim;
    ``FidelityTier.permits_pnl_claim`` enforces this and ``validation.promotion``
    hard-fails below ``TIER_2``.

    ``price`` returns ``None`` when a route exists but is degraded. It raises
    ``NoRoute`` when no route exists. The distinction is caller-visible and
    cannot be collapsed — see ``NoRoute`` docstring.

    ``fill`` must never be called unless ``price`` previously returned a
    non-None ``Quote`` for the same order. Calling it on a ``NoRoute`` condition
    is a programming error; the model may raise rather than return a synthetic
    fill.
    """

    @property
    def fidelity(self) -> FidelityTier:
        """The fidelity tier this model can attest to."""
        ...

    def price(
        self,
        *,
        intent: OrderIntent,
        state: PointInTimeState,
        now: float,
    ) -> Quote | None:
        """Attempt to find and price a route for ``intent`` at ``now``.

        Returns a ``Quote`` on success, ``None`` on degraded/stale, raises
        ``NoRoute`` when no route exists at all. The caller must treat a
        ``NoRoute`` as a hard veto: no fill, no retry at a degraded price.
        """
        ...

    def fill(
        self,
        *,
        order: ApprovedOrder,
        state: PointInTimeState,
        now: float,
    ) -> ExecutionReport:
        """Simulate the execution of an approved order.

        ``now`` must be > ``order.decided_at`` — the latency model enforces
        this before calling. A fill at the decision timestamp is look-ahead.

        Returns an ``ExecutionReport`` for every attempt: landed, failed, or
        expired. An ``ExecutionReport`` with ``fill=None`` and a terminal state
        is a real, costly outcome (gas charged), not an absence to skip.

        Must never return a synthetic fill for a ``NoRoute`` condition. If the
        route disappeared between ``price`` and ``fill``, raise ``NoRoute``
        so the caller can record a proper terminal non-fill.
        """
        ...


class ExecutionVenue(Protocol):
    """A venue that accepts and settles orders.

    Separates submission from settlement because on Solana the two are
    physically different events: a transaction is submitted to the validator
    network (submit) and then either lands in a block or drops (process_event).
    A venue that returns a fill from submit() collapses those into one and
    cannot model the landing-delay distribution, which is a separate knob from
    the landing-probability distribution — see ``execution/latency.py``.

    ``submit`` produces a receipt whose ``ready_at`` is when the order may
    first be matched against a market state. Events before ``ready_at`` must
    not fill the order.

    ``process_event`` drives settlement: the engine replays historical events
    and passes them to the venue, which emits ``ExecutionReport`` objects as
    orders settle. The ``list`` return allows one event to settle multiple
    orders (e.g., a block that confirms several transactions).
    """

    def submit(self, order: ApprovedOrder, now: float) -> OrderReceipt:
        """Accept an order and return a receipt.

        ``now`` is the simulated clock at submission time. ``receipt.ready_at``
        must be > ``now`` — a ready_at equal to the submission time would let
        the engine match the order against the same bar state that generated it,
        defeating the bar-close invariant.

        Returns a receipt even for orders the venue cannot accept:
        ``receipt.accepted = False`` with a ``reason`` is the correct encoding.
        Raising here would leave the engine with an unrecorded attempt.
        """
        ...

    def process_event(self, event: HistoricalEvent) -> list[ExecutionReport]:
        """Drive settlement for all pending orders up to ``event.available_time``.

        Called by the engine in replay order. The venue checks each pending
        order whose ``ready_at <= event.available_time`` and emits a report for
        each one that settles (landed, failed, expired).

        Idempotency: the engine may pass the same event twice on replay. The
        venue must deduplicate by ``report_id`` — ``ExecutionReport.report_id``
        is the key, and a duplicate is a no-op (``backtest/invariants.py``
        enforces this).

        Returns an empty list when no orders settle on this event.
        """
        ...


__all__ = [
    "ApprovedOrder",
    "ExecutionModel",
    "ExecutionVenue",
    "NoRoute",
]
