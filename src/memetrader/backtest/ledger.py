"""Conservation-based accounting core for the backtest replay.

This is the module the rest of ``backtest/`` cannot be wrong about. Every
number ``metrics/attribution.py`` reconciles, every equity curve
``validation/promotion`` gates on, and every dollar the report prints traces
back to a mutation this file recorded. If a fill is applied twice, or half
applied, or a fee is silently dropped, no downstream test can tell the
difference between a working strategy and a bookkeeping error, and the
adversarial audit's whole complaint about the live system is exactly that
kind of error.

Design, in one sentence: **cash and token inventory are integers, and the
only way either one changes is through :meth:`BacktestLedger.apply_fill`,
which records exactly one :class:`LedgerEntry` per call and mutates
all-or-nothing.**

Why this is not "a dict with a lock on it"
===========================================

* **Integers, always.** Cash lives as micro-USD (``1e-6`` USD, matching
  ``broker.cash_micro_usd`` and ``journal``'s conventions) and token
  quantities as atomic units. ``docs/BACKTEST-CONTRACTS.md`` §0 is explicit:
  a float token quantity is a rounding error waiting to be called a fill.
  The one boundary crossing is unavoidable and deliberate: ``types.Fill``
  itself already reports its cash legs as ``float`` dollars
  (``notional_usd``, ``gas_usd``, ``pool_fee_usd`` — see ``types.Fill``'s own
  docstring, "dollars are derived for reporting"), so this ledger converts
  those *once*, at the moment a report is applied, with the same
  round-to-nearest-micro rule ``broker._to_micro`` uses. Nothing downstream
  of that conversion ever touches a float again.

* **FIFO lots, not a single averaged position.** A position collapsed to one
  average entry price cannot answer "how much of this exit's PnL came from
  the tranche bought before the pump versus the one bought during it", and
  it cannot support a partial exit that only closes the oldest shares. Each
  BUY creates one lot; each SELL walks the lots oldest-first
  (:func:`_fifo_consume_plan`) and only *then* mutates them
  (:func:`_apply_consume_plan`) — the plan is computed read-only first so a
  sell that turns out to exceed held inventory raises before anything is
  touched, which is what makes "partially applied" impossible rather than
  merely unlikely.

* **One mutation path.** ``_cash_micro`` and ``_lots`` are private. There is
  no setter. The only method that touches them is :meth:`apply_fill`'s
  private helpers, and every one of those helpers appends the
  :class:`LedgerEntry` that describes what it just did in the same call.
  ``invariants.check_cash_matches_entry_log`` re-derives cash from that log
  and would catch a future change that mutates state without recording one.

* **Idempotency mirrors ``journal.Ledger``.** A ``_seen`` set keyed on
  ``ExecutionReport.report_id`` — the same pattern as
  ``journal.Ledger._append_row``'s ``_seen`` set keyed on row identity.
  Applying the same report twice is a no-op that still returns ``False``
  rather than raising, because a venue re-emitting a report it already sent
  is an expected, not exceptional, event (``types.ExecutionReport``'s own
  docstring says as much).

What this module deliberately does not do
==========================================

It does not know about ``Quote``, ``RiskBounds`` or ``OrderIntent`` — those
belong to the execution/risk layers, and binding a fill to the quote it was
priced against is ``broker.py``'s job (audit C3) before a report ever
reaches here. It does not mark positions to market — that is
``portfolio.mark_book``, fed by :meth:`BacktestLedger.snapshot`. And it does
not touch a filesystem; ``to_state``/``from_state`` produce and consume
plain, JSON-safe dicts so that whatever owns the run's persistence
(presumably a sibling of ``journal.py`` for the backtest tree) can decide how
and when to write them.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

from memetrader.types import CostBreakdown, ExecutionReport, Fill, Position, Side

__all__ = [
    "MICRO",
    "BacktestLedger",
    "FillPreview",
    "InsufficientLots",
    "LedgerEntry",
    "LedgerError",
    "LedgerSnapshot",
    "LotView",
    "NegativeCash",
]

#: Micro-USD per dollar. Matches ``broker.py``'s ``_MICRO`` / ``cash_micro_usd``
#: and ``journal.py``'s conventions, so a run's numbers compare directly
#: against the live paper broker's without a unit conversion at the seam.
MICRO = 10**6

_EntryKind = Literal["buy", "sell", "failure", "rejected", "duplicate"]


class LedgerError(RuntimeError):
    """An accounting operation could not be completed without violating
    conservation. Raised, never swallowed. See ``backtest/invariants.py``."""


class InsufficientLots(LedgerError):
    """A SELL fill's quantity exceeds what is held for that symbol.

    The old broker's equivalent (``broker.InsufficientPosition``) refuses to
    clamp: audit C3 is that the executed size must equal the priced size, and
    silently filling the smaller, held amount would execute a size that was
    never quoted. This ledger inherits the same refusal rather than inventing
    a softer one.
    """


class NegativeCash(LedgerError):
    """A fill would drive cash below zero. No leverage is modelled here —
    contracts §6's first invariant — so this can only mean an upstream bug
    let a BUY (or the gas on a failed attempt) through that the broker/risk
    layer should have refused before it ever reached the ledger."""


# ---------------------------------------------------------------------------
# Internal inventory
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Lot:
    """One FIFO acquisition layer. Mutated in place as later exits partially
    consume it — shrinking the same object is what makes "the remainder"
    unambiguous; replacing it with a new smaller lot would need its own
    identity and lose the line back to the fill that created it.

    Never exposed directly: callers see :class:`LotView`, an immutable copy,
    so nothing outside this module can mutate inventory without going
    through :meth:`BacktestLedger.apply_fill`.
    """

    lot_id: str
    order_id: str
    fill_id: str
    symbol: str
    mint: str
    decimals: int
    quantity_atomic: int
    basis_micro_usd: int
    opened_at: float


@dataclass(frozen=True, slots=True)
class LotView:
    """An immutable snapshot of one FIFO lot, for tests and diagnostics that
    need to see individual tranches rather than the aggregated
    :class:`~.types.Position`."""

    lot_id: str
    order_id: str
    fill_id: str
    symbol: str
    mint: str
    decimals: int
    quantity_atomic: int
    basis_micro_usd: int
    opened_at: float

    @property
    def basis_usd(self) -> float:
        return self.basis_micro_usd / MICRO


def _lot_view(lot: _Lot) -> LotView:
    return LotView(
        lot_id=lot.lot_id,
        order_id=lot.order_id,
        fill_id=lot.fill_id,
        symbol=lot.symbol,
        mint=lot.mint,
        decimals=lot.decimals,
        quantity_atomic=lot.quantity_atomic,
        basis_micro_usd=lot.basis_micro_usd,
        opened_at=lot.opened_at,
    )


# ---------------------------------------------------------------------------
# The operation log
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One recorded mutation — the *only* record of why cash or inventory
    moved. This is the "operation log" ``invariants.py`` checks #6
    (deterministic replay), #9 (risk requote, engine-side) and the
    log-replay form of #2/#11 are built on: they are properties of a
    *sequence* of these rows, not of a single ledger snapshot.

    ``quantity_delta_atomic`` is signed from the ledger's point of view: positive
    when a BUY adds inventory, negative when a SELL removes it, zero for a
    failure, rejection or suppressed duplicate. Recording it here (rather
    than making a reader reconstruct it from ``lots_consumed``) is what lets
    invariant checks operate on the log alone, without touching lot internals.
    """

    seq: int
    report_id: str
    kind: _EntryKind
    ts: float
    symbol: str | None
    fill_id: str | None
    order_id: str | None
    cash_delta_micro_usd: int
    realized_pnl_delta_micro_usd: int
    quantity_delta_atomic: int
    venue_fee_micro_usd: int
    network_fee_micro_usd: int
    priority_fee_micro_usd: int
    lots_consumed: tuple[str, ...] = ()
    lot_created: str | None = None

    @property
    def total_fee_micro_usd(self) -> int:
        return (
            self.venue_fee_micro_usd
            + self.network_fee_micro_usd
            + self.priority_fee_micro_usd
        )


def _entry_to_state(entry: LedgerEntry) -> dict[str, Any]:
    return {
        "seq": entry.seq,
        "report_id": entry.report_id,
        "kind": entry.kind,
        "ts": entry.ts,
        "symbol": entry.symbol,
        "fill_id": entry.fill_id,
        "order_id": entry.order_id,
        "cash_delta_micro_usd": entry.cash_delta_micro_usd,
        "realized_pnl_delta_micro_usd": entry.realized_pnl_delta_micro_usd,
        "quantity_delta_atomic": entry.quantity_delta_atomic,
        "venue_fee_micro_usd": entry.venue_fee_micro_usd,
        "network_fee_micro_usd": entry.network_fee_micro_usd,
        "priority_fee_micro_usd": entry.priority_fee_micro_usd,
        "lots_consumed": list(entry.lots_consumed),
        "lot_created": entry.lot_created,
    }


def _entry_from_state(state: dict[str, Any]) -> LedgerEntry:
    return LedgerEntry(
        seq=int(state["seq"]),
        report_id=str(state["report_id"]),
        kind=state["kind"],
        ts=float(state["ts"]),
        symbol=state.get("symbol"),
        fill_id=state.get("fill_id"),
        order_id=state.get("order_id"),
        cash_delta_micro_usd=int(state["cash_delta_micro_usd"]),
        realized_pnl_delta_micro_usd=int(state["realized_pnl_delta_micro_usd"]),
        quantity_delta_atomic=int(state["quantity_delta_atomic"]),
        venue_fee_micro_usd=int(state["venue_fee_micro_usd"]),
        network_fee_micro_usd=int(state["network_fee_micro_usd"]),
        priority_fee_micro_usd=int(state["priority_fee_micro_usd"]),
        lots_consumed=tuple(state.get("lots_consumed") or ()),
        lot_created=state.get("lot_created"),
    )


# ---------------------------------------------------------------------------
# Snapshot — what portfolio.mark_book consumes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    """A consistent view of the book, for marking. Everything
    ``portfolio.mark_book(cash_usd=..., positions=..., realized_pnl_usd=...,
    starting_cash_usd=..., fees_paid_usd=..., gas_paid_usd=...)`` needs, plus
    the fee split ``metrics/attribution.py`` reconciles against.

    ``fees_paid_usd``/``gas_paid_usd`` mirror ``LocalPaperBroker``'s split:
    venue and priority fees are "fees", network fee (gas) is reported
    separately, because gas is paid even on a failed attempt and the two
    have different economic meanings.
    """

    ts: float
    cash_micro_usd: int
    starting_cash_micro_usd: int
    positions: dict[str, Position]
    realized_pnl_micro_usd: int
    venue_fee_micro_usd: int
    network_fee_micro_usd: int
    priority_fee_micro_usd: int

    @property
    def cash_usd(self) -> float:
        return self.cash_micro_usd / MICRO

    @property
    def starting_cash_usd(self) -> float:
        return self.starting_cash_micro_usd / MICRO

    @property
    def realized_pnl_usd(self) -> float:
        return self.realized_pnl_micro_usd / MICRO

    @property
    def fees_paid_usd(self) -> float:
        return (self.venue_fee_micro_usd + self.priority_fee_micro_usd) / MICRO

    @property
    def gas_paid_usd(self) -> float:
        return self.network_fee_micro_usd / MICRO

    def unrealized_pnl_usd(self, marks: dict[str, float | None]) -> dict[str, float | None]:
        """Per-symbol unrealised PnL against external marks.

        Marking is deliberately not this module's job (see the module
        docstring): a mark is a market read with its own provenance and
        haircut, owned by ``portfolio.py``. This just applies whatever marks
        the caller already produced to this snapshot's cost bases, keeping
        realized and unrealized PnL visibly distinct outputs rather than one
        blended number.
        """
        return {
            symbol: position.unrealized_pnl_usd(marks.get(symbol))
            for symbol, position in self.positions.items()
        }


@dataclass(frozen=True, slots=True)
class FillPreview:
    """What :meth:`BacktestLedger.apply_fill` *would* do, computed without
    calling it. Invariant #7 — "a dry run mutates nothing" — is enforced by
    :meth:`BacktestLedger.preview_fill` never touching ``_cash_micro``,
    ``_lots`` or ``_seen``; this is its return value, not a side effect.
    """

    report_id: str
    would_apply: bool
    reason: str
    cash_delta_micro_usd: int
    realized_pnl_delta_micro_usd: int
    quantity_delta_atomic: int


# ---------------------------------------------------------------------------
# Pure helpers — no ledger state, so they are trivially testable in isolation
# ---------------------------------------------------------------------------


def _to_micro(usd: float) -> int:
    """Dollars to integer micro-USD, rounded to the nearest unit.

    Identical rule to ``broker._to_micro``: round rather than truncate, so
    that a dollar amount with no exact binary representation does not depend
    on how it was spelled, and round-trips exactly back through division by
    :data:`MICRO`. This is the *only* place a float becomes an int in this
    module — see the module docstring's note on where the float boundary is.
    """
    return round(float(usd) * MICRO)


def _extract_costs(report: ExecutionReport, fill: Fill) -> CostBreakdown:
    """The cash-affecting cost split to charge for one fill.

    Mirrors ``metrics.attribution._extract_costs`` exactly: prefer the
    engine's authoritative ``report.costs`` breakdown; fall back to the
    ``Fill``'s own two cost fields when the engine did not supply one. Kept
    as a separate copy rather than importing ``metrics.attribution`` because
    that module is another agent's file and because a ledger should not
    depend on the reporting layer that depends on it.
    """
    if report.costs is not None:
        return report.costs
    return CostBreakdown(
        venue_fee_usd=fill.pool_fee_usd,
        network_fee_usd=fill.gas_usd,
    )


def _proportional(total: int, part: int, whole: int) -> int:
    """``total * part / whole``, floored, exact when ``part == whole``.

    Mirrors ``broker._proportional``. Floor (never round) is what stops a
    sequence of partial exits from releasing more basis in aggregate than the
    lot holds — the remainder always stays with the lot and is released in
    full by whichever exit empties it.
    """
    if whole <= 0:
        return 0
    if part >= whole:
        return total
    return total * part // whole


def _fifo_consume_plan(dq: deque[_Lot], quantity: int) -> list[tuple[_Lot, int, int]]:
    """Read-only FIFO consumption plan for a SELL of ``quantity`` atomic units.

    Walks the deque oldest-first and returns ``(lot, atomic_taken,
    basis_released)`` triples without mutating anything. Computing the whole
    plan before applying any of it is what makes a sell that turns out to
    exceed held inventory raise :class:`InsufficientLots` with *zero* partial
    effect — the "all-or-nothing" half of the fill-application invariant.
    """
    remaining = quantity
    plan: list[tuple[_Lot, int, int]] = []
    for lot in dq:
        if remaining <= 0:
            break
        take = min(remaining, lot.quantity_atomic)
        if take <= 0:
            continue
        basis_share = _proportional(lot.basis_micro_usd, take, lot.quantity_atomic)
        plan.append((lot, take, basis_share))
        remaining -= take
    if remaining > 0:
        held = sum(lot.quantity_atomic for lot in dq)
        raise InsufficientLots(
            f"sell of {quantity} atomic units exceeds held inventory of {held} "
            "atomic units — no sale may exceed settled inventory (contracts §6)"
        )
    return plan


def _apply_consume_plan(dq: deque[_Lot], plan: list[tuple[_Lot, int, int]]) -> None:
    """Mutate the lots named in a plan already validated by
    :func:`_fifo_consume_plan`, then drop any lot the plan emptied.

    Safe to pop from the left only: the plan is always built oldest-first, so
    an exhausted lot is always at the front of what remains.
    """
    for lot, take, basis_share in plan:
        lot.quantity_atomic -= take
        lot.basis_micro_usd -= basis_share
    while dq and dq[0].quantity_atomic == 0:
        dq.popleft()


def _aggregate_position(symbol: str, dq: deque[_Lot]) -> Position:
    """Collapse a symbol's FIFO lots into the single :class:`~.types.Position`
    ``portfolio.mark_book`` expects. ``opened_at`` is the *first* lot's time —
    a position's age is measured from its first entry, not from a later
    scale-in, matching ``broker._apply``'s comment to the same effect."""
    total_qty = sum(lot.quantity_atomic for lot in dq)
    total_basis = sum(lot.basis_micro_usd for lot in dq)
    decimals = dq[0].decimals
    mint = dq[0].mint
    opened_at = min(lot.opened_at for lot in dq)
    qty_ui = total_qty / (10**decimals)
    avg_entry = (total_basis / MICRO) / qty_ui if qty_ui > 0 else 0.0
    return Position(
        symbol=symbol,
        mint=mint,
        quantity_atomic=total_qty,
        decimals=decimals,
        avg_entry_price_usd=avg_entry,
        opened_at=opened_at,
        cost_basis_usd=total_basis / MICRO,
    )


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


class BacktestLedger:
    """Single source of truth for cash and inventory during a replay.

    Conservation identity this maintains, in integer micro-USD, on every
    successful call to :meth:`apply_fill` (compare ``broker.py``'s docstring
    for the live equivalent)::

        cash + sum(lot.basis) == starting_cash + realized_pnl - failed_gas

    A BUY moves ``notional + fees`` from cash into a new lot's basis: both
    sides move by the same amount. A SELL adds ``proceeds - fees`` to cash,
    releases ``basis_share`` from the consumed lots, and books
    ``proceeds - basis_share - fees`` as realized: again, both sides move by
    the same amount. A FAILED attempt burns gas on the left side only, and
    ``docs/BACKTEST-CONTRACTS.md``'s ``failed_gas`` accounting keeps the
    identity honest by tracking it as network fee. If any split above is
    wrong the identity stops balancing exactly, which is what
    ``invariants.check_cash_matches_entry_log`` is for.
    """

    def __init__(self, *, starting_cash_micro_usd: int, run_id: str = "") -> None:
        if starting_cash_micro_usd < 0:
            raise LedgerError(
                f"starting_cash_micro_usd must be >= 0, got {starting_cash_micro_usd}"
            )
        self.run_id = run_id
        self._starting_cash_micro = starting_cash_micro_usd
        self._cash_micro = starting_cash_micro_usd
        self._lots: dict[str, deque[_Lot]] = {}
        self._realized_micro = 0
        self._venue_fee_micro = 0
        self._network_fee_micro = 0
        self._priority_fee_micro = 0
        self._seen: set[str] = set()
        self._entries: list[LedgerEntry] = []
        self._seq_counter = itertools.count()
        self._lot_seq_counter = itertools.count()

    # -- read-only views ----------------------------------------------------

    @property
    def cash_micro_usd(self) -> int:
        return self._cash_micro

    @property
    def starting_cash_micro_usd(self) -> int:
        return self._starting_cash_micro

    @property
    def realized_pnl_micro_usd(self) -> int:
        return self._realized_micro

    @property
    def venue_fee_micro_usd(self) -> int:
        return self._venue_fee_micro

    @property
    def network_fee_micro_usd(self) -> int:
        return self._network_fee_micro

    @property
    def priority_fee_micro_usd(self) -> int:
        return self._priority_fee_micro

    def entries(self) -> tuple[LedgerEntry, ...]:
        """The full operation log, in application order. This is the
        sequence ``invariants.py`` replays for the checks that are properties
        of a history rather than of one snapshot."""
        return tuple(self._entries)

    def lot_views(self, symbol: str) -> tuple[LotView, ...]:
        """Immutable snapshots of ``symbol``'s FIFO lots, oldest first."""
        return tuple(_lot_view(lot) for lot in self._lots.get(symbol, ()))

    def seen(self, report_id: str) -> bool:
        return report_id in self._seen

    def snapshot(self, *, ts: float = 0.0) -> LedgerSnapshot:
        positions = {
            symbol: _aggregate_position(symbol, dq)
            for symbol, dq in self._lots.items()
            if dq
        }
        return LedgerSnapshot(
            ts=ts,
            cash_micro_usd=self._cash_micro,
            starting_cash_micro_usd=self._starting_cash_micro,
            positions=positions,
            realized_pnl_micro_usd=self._realized_micro,
            venue_fee_micro_usd=self._venue_fee_micro,
            network_fee_micro_usd=self._network_fee_micro,
            priority_fee_micro_usd=self._priority_fee_micro,
        )

    # -- the one mutation path ------------------------------------------------

    def apply_fill(self, report: ExecutionReport, *, mint: str | None = None) -> bool:
        """Apply one :class:`~.types.ExecutionReport`. Returns ``True`` iff
        this call mutated ledger state.

        Idempotent on ``report.report_id`` (invariant #11): a report whose
        ``report_id`` has already been applied is a no-op that returns
        ``False`` and still appends a zero-delta ``"duplicate"``
        :class:`LedgerEntry`, so a re-emitted report is visible in the log
        without being counted twice anywhere it reconciles.

        All-or-nothing: everything that can fail (fee/notional validation,
        FIFO consumption planning) happens before any attribute is mutated,
        so a raised exception leaves the ledger exactly as it was before the
        call. See :func:`_fifo_consume_plan` for the SELL side of that
        argument.

        ``mint`` is accepted because neither :class:`~.types.ExecutionReport`
        nor :class:`~.types.Fill` carries a mint (only ``token_decimals`` —
        the live broker has the same gap and is handed the mint separately by
        its caller, see ``LocalPaperBroker._apply``). It only affects the
        cosmetic ``Position.mint`` field in :meth:`snapshot`; every accounting
        computation here uses ``symbol`` and atomic quantities, never mint.
        """
        if report.report_id in self._seen:
            self._record_duplicate(report)
            return False
        self._seen.add(report.report_id)
        self._apply_report(report, mint=mint)
        return True

    def preview_fill(
        self, report: ExecutionReport, *, mint: str | None = None
    ) -> FillPreview:
        """What :meth:`apply_fill` would do, computed with zero side effects.

        Deliberately does not consult or update ``_seen`` — a preview is not
        an application, and the same report may be legitimately previewed
        many times (an operator inspecting a decision) before it is ever
        actually applied, or never applied at all.
        """
        del mint  # not needed for the delta-only preview; kept for symmetry
        if report.report_id in self._seen:
            return FillPreview(
                report_id=report.report_id,
                would_apply=False,
                reason="report_id already applied",
                cash_delta_micro_usd=0,
                realized_pnl_delta_micro_usd=0,
                quantity_delta_atomic=0,
            )
        fill = report.fill
        if fill is None or fill.failed:
            gas_micro = _to_micro(fill.gas_usd) if fill is not None else 0
            return FillPreview(
                report_id=report.report_id,
                would_apply=True,
                reason="failure" if fill is not None else "rejected, no fill",
                cash_delta_micro_usd=-gas_micro,
                realized_pnl_delta_micro_usd=0,
                quantity_delta_atomic=0,
            )
        costs = _extract_costs(report, fill)
        fee_micro = (
            _to_micro(costs.venue_fee_usd)
            + _to_micro(costs.network_fee_usd)
            + _to_micro(costs.priority_fee_usd)
        )
        notional_micro = _to_micro(fill.notional_usd)
        if fill.side is Side.BUY:
            return FillPreview(
                report_id=report.report_id,
                would_apply=True,
                reason="buy",
                cash_delta_micro_usd=-(notional_micro + fee_micro),
                realized_pnl_delta_micro_usd=0,
                quantity_delta_atomic=fill.token_amount_atomic,
            )
        # SELL: plan against current lots, read-only, to report the realized
        # delta a real apply_fill would book — but never mutate them.
        dq = self._lots.get(fill.symbol)
        if not dq:
            return FillPreview(
                report_id=report.report_id,
                would_apply=False,
                reason=f"no open lots for {fill.symbol}",
                cash_delta_micro_usd=0,
                realized_pnl_delta_micro_usd=0,
                quantity_delta_atomic=0,
            )
        try:
            plan = _fifo_consume_plan(dq, fill.token_amount_atomic)
        except InsufficientLots as exc:
            return FillPreview(
                report_id=report.report_id,
                would_apply=False,
                reason=str(exc),
                cash_delta_micro_usd=0,
                realized_pnl_delta_micro_usd=0,
                quantity_delta_atomic=0,
            )
        basis_consumed = sum(basis_share for _, _, basis_share in plan)
        realized_delta = notional_micro - basis_consumed - fee_micro
        return FillPreview(
            report_id=report.report_id,
            would_apply=True,
            reason="sell",
            cash_delta_micro_usd=notional_micro - fee_micro,
            realized_pnl_delta_micro_usd=realized_delta,
            quantity_delta_atomic=-fill.token_amount_atomic,
        )

    # -- internals: one branch per report shape ------------------------------

    def _next_seq(self) -> int:
        return next(self._seq_counter)

    def _record_duplicate(self, report: ExecutionReport) -> None:
        fill = report.fill
        self._entries.append(
            LedgerEntry(
                seq=self._next_seq(),
                report_id=report.report_id,
                kind="duplicate",
                ts=report.ts,
                symbol=fill.symbol if fill is not None else None,
                fill_id=fill.fill_id if fill is not None else None,
                order_id=report.order_id,
                cash_delta_micro_usd=0,
                realized_pnl_delta_micro_usd=0,
                quantity_delta_atomic=0,
                venue_fee_micro_usd=0,
                network_fee_micro_usd=0,
                priority_fee_micro_usd=0,
            )
        )

    def _apply_report(self, report: ExecutionReport, *, mint: str | None) -> None:
        fill = report.fill
        if fill is None or fill.failed:
            self._apply_failure_or_rejection(report, fill)
            return
        if fill.side is Side.BUY:
            self._apply_buy(report, fill, mint=mint)
        else:
            self._apply_sell(report, fill)

    def _apply_failure_or_rejection(
        self, report: ExecutionReport, fill: Fill | None
    ) -> None:
        """Invariant #8: an execution failure can never become a synthetic
        fill. No lot is created or consumed here under any circumstance —
        the only thing that can happen is gas being spent, because a failed
        Solana transaction still costs the validator (mirrors
        ``broker._apply``'s FAILED branch)."""
        gas_micro = _to_micro(fill.gas_usd) if fill is not None else 0
        new_cash = self._cash_micro - gas_micro
        if new_cash < 0:
            raise NegativeCash(
                f"failed attempt's gas ({gas_micro / MICRO:.6f} USD) would drive "
                f"cash below zero from {self._cash_micro / MICRO:.6f} USD"
            )
        self._cash_micro = new_cash
        self._network_fee_micro += gas_micro
        self._entries.append(
            LedgerEntry(
                seq=self._next_seq(),
                report_id=report.report_id,
                kind="failure" if fill is not None else "rejected",
                ts=report.ts,
                symbol=fill.symbol if fill is not None else None,
                fill_id=fill.fill_id if fill is not None else None,
                order_id=report.order_id,
                cash_delta_micro_usd=-gas_micro,
                realized_pnl_delta_micro_usd=0,
                quantity_delta_atomic=0,
                venue_fee_micro_usd=0,
                network_fee_micro_usd=gas_micro,
                priority_fee_micro_usd=0,
            )
        )

    def _apply_buy(self, report: ExecutionReport, fill: Fill, *, mint: str | None) -> None:
        costs = _extract_costs(report, fill)
        venue_micro = _to_micro(costs.venue_fee_usd)
        network_micro = _to_micro(costs.network_fee_usd)
        priority_micro = _to_micro(costs.priority_fee_usd)
        notional_micro = _to_micro(fill.notional_usd)
        total_fee_micro = venue_micro + network_micro + priority_micro
        total_cost_micro = notional_micro + total_fee_micro

        new_cash = self._cash_micro - total_cost_micro
        if new_cash < 0:
            raise NegativeCash(
                f"BUY {fill.symbol} costing {total_cost_micro / MICRO:.6f} USD "
                f"(notional {notional_micro / MICRO:.6f} + fees "
                f"{total_fee_micro / MICRO:.6f}) would drive cash below zero "
                f"from {self._cash_micro / MICRO:.6f} USD — no leverage is "
                "modelled (contracts §6)"
            )

        lot_id = f"lot:{fill.fill_id}:{next(self._lot_seq_counter)}"
        lot = _Lot(
            lot_id=lot_id,
            order_id=fill.order_id,
            fill_id=fill.fill_id,
            symbol=fill.symbol,
            mint=mint or "",
            decimals=fill.token_decimals,
            quantity_atomic=fill.token_amount_atomic,
            # Cost basis includes fees and gas, so a lot's break-even is its
            # true break-even — the same rule types.Position documents.
            basis_micro_usd=total_cost_micro,
            opened_at=fill.ts,
        )
        self._lots.setdefault(fill.symbol, deque()).append(lot)

        self._cash_micro = new_cash
        self._venue_fee_micro += venue_micro
        self._network_fee_micro += network_micro
        self._priority_fee_micro += priority_micro

        self._entries.append(
            LedgerEntry(
                seq=self._next_seq(),
                report_id=report.report_id,
                kind="buy",
                ts=report.ts,
                symbol=fill.symbol,
                fill_id=fill.fill_id,
                order_id=report.order_id,
                cash_delta_micro_usd=-total_cost_micro,
                realized_pnl_delta_micro_usd=0,
                quantity_delta_atomic=fill.token_amount_atomic,
                venue_fee_micro_usd=venue_micro,
                network_fee_micro_usd=network_micro,
                priority_fee_micro_usd=priority_micro,
                lot_created=lot_id,
            )
        )

    def _apply_sell(self, report: ExecutionReport, fill: Fill) -> None:
        dq = self._lots.get(fill.symbol)
        if not dq:
            raise InsufficientLots(f"SELL {fill.symbol}: no open lots are held")

        # Read-only plan first: a sell that exceeds held inventory must raise
        # before a single lot is touched (invariant: no sale exceeds settled
        # inventory, and the fill-application is all-or-nothing).
        plan = _fifo_consume_plan(dq, fill.token_amount_atomic)

        costs = _extract_costs(report, fill)
        venue_micro = _to_micro(costs.venue_fee_usd)
        network_micro = _to_micro(costs.network_fee_usd)
        priority_micro = _to_micro(costs.priority_fee_usd)
        total_fee_micro = venue_micro + network_micro + priority_micro
        notional_micro = _to_micro(fill.notional_usd)

        basis_consumed = sum(basis_share for _, _, basis_share in plan)
        cash_delta = notional_micro - total_fee_micro
        new_cash = self._cash_micro + cash_delta
        if new_cash < 0:
            raise NegativeCash(
                f"SELL {fill.symbol} would drive cash below zero: "
                f"{self._cash_micro / MICRO:.6f} USD + {cash_delta / MICRO:.6f} USD"
            )
        realized_delta = notional_micro - basis_consumed - total_fee_micro

        # Only now, with every raise-able check behind us, does state move.
        _apply_consume_plan(dq, plan)
        if not dq:
            del self._lots[fill.symbol]
        self._cash_micro = new_cash
        self._realized_micro += realized_delta
        self._venue_fee_micro += venue_micro
        self._network_fee_micro += network_micro
        self._priority_fee_micro += priority_micro

        self._entries.append(
            LedgerEntry(
                seq=self._next_seq(),
                report_id=report.report_id,
                kind="sell",
                ts=report.ts,
                symbol=fill.symbol,
                fill_id=fill.fill_id,
                order_id=report.order_id,
                cash_delta_micro_usd=cash_delta,
                realized_pnl_delta_micro_usd=realized_delta,
                quantity_delta_atomic=-fill.token_amount_atomic,
                venue_fee_micro_usd=venue_micro,
                network_fee_micro_usd=network_micro,
                priority_fee_micro_usd=priority_micro,
                lots_consumed=tuple(lot.lot_id for lot, _, _ in plan),
            )
        )

    # -- serialization: crash/restart round-trip -----------------------------

    def to_state(self) -> dict[str, Any]:
        """A plain, JSON-safe dict capturing everything needed to resume this
        ledger exactly. Whoever owns the run's on-disk journal (a backtest
        sibling of ``journal.Ledger``) decides when and how this is written;
        this module only promises the round trip through
        :meth:`from_state` is exact.
        """
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "starting_cash_micro_usd": self._starting_cash_micro,
            "cash_micro_usd": self._cash_micro,
            "realized_pnl_micro_usd": self._realized_micro,
            "venue_fee_micro_usd": self._venue_fee_micro,
            "network_fee_micro_usd": self._network_fee_micro,
            "priority_fee_micro_usd": self._priority_fee_micro,
            "seen_report_ids": sorted(self._seen),
            "lots": {
                symbol: [
                    {
                        "lot_id": lot.lot_id,
                        "order_id": lot.order_id,
                        "fill_id": lot.fill_id,
                        "symbol": lot.symbol,
                        "mint": lot.mint,
                        "decimals": lot.decimals,
                        "quantity_atomic": lot.quantity_atomic,
                        "basis_micro_usd": lot.basis_micro_usd,
                        "opened_at": lot.opened_at,
                    }
                    for lot in dq
                ]
                for symbol, dq in self._lots.items()
                if dq
            },
            "entries": [_entry_to_state(entry) for entry in self._entries],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> BacktestLedger:
        """The inverse of :meth:`to_state`. Refuses a schema it does not
        recognise rather than guessing at a shape that might have changed —
        the same refusal ``broker.LedgerCorrupt`` makes for its state file."""
        version = int(state.get("schema_version", 0))
        if version != 1:
            raise LedgerError(
                f"ledger state has schema_version {version}, this build reads "
                "version 1 — move it aside rather than guess at its shape"
            )
        ledger = cls(
            starting_cash_micro_usd=int(state["starting_cash_micro_usd"]),
            run_id=str(state.get("run_id", "")),
        )
        ledger._cash_micro = int(state["cash_micro_usd"])
        ledger._realized_micro = int(state["realized_pnl_micro_usd"])
        ledger._venue_fee_micro = int(state["venue_fee_micro_usd"])
        ledger._network_fee_micro = int(state["network_fee_micro_usd"])
        ledger._priority_fee_micro = int(state["priority_fee_micro_usd"])
        ledger._seen = set(state.get("seen_report_ids") or ())
        max_lot_seq = -1
        for symbol, rows in (state.get("lots") or {}).items():
            dq: deque[_Lot] = deque()
            for row in rows:
                dq.append(
                    _Lot(
                        lot_id=str(row["lot_id"]),
                        order_id=str(row["order_id"]),
                        fill_id=str(row["fill_id"]),
                        symbol=str(row["symbol"]),
                        mint=str(row.get("mint") or ""),
                        decimals=int(row["decimals"]),
                        quantity_atomic=int(row["quantity_atomic"]),
                        basis_micro_usd=int(row["basis_micro_usd"]),
                        opened_at=float(row["opened_at"]),
                    )
                )
                lot_id = str(row["lot_id"])
                tail = lot_id.rsplit(":", 1)[-1]
                if tail.isdigit():
                    max_lot_seq = max(max_lot_seq, int(tail))
            if dq:
                ledger._lots[symbol] = dq
        ledger._entries = [_entry_from_state(row) for row in state.get("entries") or ()]
        max_seq = max((e.seq for e in ledger._entries), default=-1)
        ledger._seq_counter = itertools.count(max_seq + 1)
        # The open-lot scan above is not sufficient on its own. A lot that was
        # fully closed before the snapshot is gone from ``lots``, but its label
        # still appears in the entry log, so resuming the counter from the open
        # lots alone can re-issue a label an earlier closed lot already used.
        # Nothing economic depends on the label, but a duplicated lot_id makes
        # the log ambiguous to read back, which is the entire reason lots carry
        # identity. Scan the entries too and resume above the true high-water
        # mark.
        for entry in ledger._entries:
            for label in (entry.lot_created, *entry.lots_consumed):
                if not label:
                    continue
                tail = label.rsplit(":", 1)[-1]
                if tail.isdigit():
                    max_lot_seq = max(max_lot_seq, int(tail))
        ledger._lot_seq_counter = itertools.count(max_lot_seq + 1)
        return ledger
