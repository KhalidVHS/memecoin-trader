"""Accounting invariants — checked continuously, never at the end.

``docs/BACKTEST-CONTRACTS.md`` §6 lists eleven rules and is explicit about
what a breach means: *"a backtest that silently violates conservation is
worse than no backtest."* Every function below raises :class:`InvariantBreach`
on violation. None of them return a boolean sentinel, because a sentinel is a
result someone has to remember to check, and the whole point of this module
is that nobody has to remember.

Two kinds of check live here
=============================

Most of the eleven are properties of a single :class:`~.ledger.LedgerSnapshot`
or of the :class:`~.ledger.BacktestLedger`'s recorded
:class:`~.ledger.LedgerEntry` log, and :func:`check_all` runs all of those
every tick.

Three of them — #6 (byte-identical replay), #7 (a dry run mutates nothing)
and #9 (a risk resize triggers a new quote at the approved size) — are
properties of a *sequence of operations*, not of one state. ``backtest/
engine.py`` does not exist yet (it is another agent's module in this same
effort), so these are written against synthetic operation logs / explicit
ledger calls instead of against a live engine run. Each function's docstring
names the exact call the engine should make once it exists; until then, the
tests in ``tests/accounting/test_invariants.py`` exercise them directly.

Invariant → function map
=========================

1.  Cash never negative                          → :func:`check_cash_never_negative`
2.  No sale exceeding settled inventory           → :func:`check_no_oversell`
    (also enforced structurally at apply-time — see ``ledger.InsufficientLots``)
3.  Cash/tokens change only via ledger entries    → :func:`check_cash_matches_entry_log`,
                                                      :func:`check_quantity_matches_entry_log`
4.  Fees charged exactly once                     → :func:`check_fees_charged_once`,
                                                      :func:`check_fee_reconciliation`
5.  Every fill belongs to one order & one lot      → :func:`check_fill_lot_integrity`
6.  Same inputs → byte-identical output            → :func:`check_deterministic_replay`
7.  A dry run mutates nothing                      → :func:`check_dry_run_no_mutation`
8.  A failure can never become a synthetic fill    →
    :func:`check_execution_failure_no_position_change`
9.  Risk resizing triggers a new quote              → :func:`check_risk_resize_requotes`
10. SELL quantity == quoted quantity, exactly       → :func:`check_sell_quantity_matches_quote`
11. Duplicate report_id is a no-op                  → :func:`check_duplicate_report_is_noop`,
                                                       :func:`check_duplicate_entries_are_zero`
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from memetrader.backtest.ledger import (
    BacktestLedger,
    LedgerEntry,
    LedgerSnapshot,
)
from memetrader.types import ExecutionReport, Fill, OrderState, Quote, Side

__all__ = [
    "InvariantBreach",
    "check_all",
    "check_cash_matches_entry_log",
    "check_cash_never_negative",
    "check_deterministic_replay",
    "check_dry_run_no_mutation",
    "check_duplicate_entries_are_zero",
    "check_duplicate_report_is_noop",
    "check_execution_failure_no_position_change",
    "check_fee_reconciliation",
    "check_fees_charged_once",
    "check_fill_lot_integrity",
    "check_no_oversell",
    "check_quantity_matches_entry_log",
    "check_risk_resize_requotes",
    "check_sell_quantity_matches_quote",
]


class InvariantBreach(RuntimeError):
    """A recorded accounting invariant was violated.

    Deliberately not caught anywhere in the replay path — see the module
    docstring and ``docs/BACKTEST-CONTRACTS.md`` §6. A caller that wraps a
    call into this module in a broad ``except Exception`` has reintroduced
    the exact failure mode this module exists to make loud.
    """


# ---------------------------------------------------------------------------
# #1 — cash never negative
# ---------------------------------------------------------------------------


def check_cash_never_negative(snapshot: LedgerSnapshot) -> None:
    """No leverage is modelled. A negative cash balance would mean the
    ledger financed a BUY (or a failed attempt's gas) it could not afford,
    which ``ledger.NegativeCash`` should already have refused at apply-time —
    this is the independent, state-only check :func:`check_all` runs every
    tick regardless of how the state was reached."""
    if snapshot.cash_micro_usd < 0:
        raise InvariantBreach(
            f"cash is negative: {snapshot.cash_micro_usd} micro-USD "
            f"({snapshot.cash_usd:.6f} USD) — no leverage is modelled"
        )


# ---------------------------------------------------------------------------
# #2 — no sale exceeding settled inventory
# ---------------------------------------------------------------------------


def check_no_oversell(entries: Sequence[LedgerEntry]) -> None:
    """Replay the recorded quantity deltas and confirm inventory never goes
    negative for any symbol.

    ``ledger.BacktestLedger._apply_sell`` already refuses this at apply-time
    via ``InsufficientLots`` (computed all-or-nothing, before any lot is
    touched). This function is the independent check over the *log*: it
    would catch a future code path that mutated inventory without going
    through ``apply_fill``, which is exactly the kind of regression a single
    enforcement point at construction time cannot catch by itself.
    """
    held: dict[str, int] = {}
    for entry in entries:
        if entry.symbol is None or entry.quantity_delta_atomic == 0:
            continue
        held[entry.symbol] = held.get(entry.symbol, 0) + entry.quantity_delta_atomic
        if held[entry.symbol] < 0:
            raise InvariantBreach(
                f"{entry.symbol}: entry log implies inventory went negative "
                f"({held[entry.symbol]} atomic units) at seq={entry.seq} "
                f"report_id={entry.report_id}"
            )


# ---------------------------------------------------------------------------
# #3 — cash and tokens change only through ledger entries
# ---------------------------------------------------------------------------


def check_cash_matches_entry_log(ledger: BacktestLedger) -> None:
    """Cash is structurally private to :class:`~.ledger.BacktestLedger` — the
    only method that assigns ``_cash_micro`` also appends the
    :class:`~.ledger.LedgerEntry` describing that assignment. This check
    re-derives cash purely from the recorded log and compares it against the
    ledger's live value, so a future edit that mutates cash without
    recording an entry (or records the wrong delta) is caught rather than
    trusted."""
    total_delta = sum(entry.cash_delta_micro_usd for entry in ledger.entries())
    expected = ledger.starting_cash_micro_usd + total_delta
    if expected != ledger.cash_micro_usd:
        raise InvariantBreach(
            f"cash {ledger.cash_micro_usd} micro-USD does not equal "
            f"starting_cash ({ledger.starting_cash_micro_usd}) + entry log deltas "
            f"({total_delta}) = {expected} — cash changed outside the entry log"
        )


def check_quantity_matches_entry_log(ledger: BacktestLedger) -> None:
    """As :func:`check_cash_matches_entry_log`, but for token inventory: the
    sum of every symbol's held quantity must equal the sum of its entry-log
    deltas."""
    held: dict[str, int] = {}
    for entry in ledger.entries():
        if entry.symbol is None:
            continue
        held[entry.symbol] = held.get(entry.symbol, 0) + entry.quantity_delta_atomic
    snapshot = ledger.snapshot()
    live = {
        symbol: position.quantity_atomic for symbol, position in snapshot.positions.items()
    }
    for symbol, expected_qty in held.items():
        actual_qty = live.get(symbol, 0)
        if expected_qty != actual_qty:
            raise InvariantBreach(
                f"{symbol}: entry log implies {expected_qty} atomic units held, "
                f"ledger snapshot has {actual_qty} — inventory changed outside "
                "the entry log"
            )
    for symbol in live:
        if symbol not in held:
            raise InvariantBreach(
                f"{symbol}: ledger snapshot holds a position with no entry-log "
                "history at all"
            )


# ---------------------------------------------------------------------------
# #4 — fees charged exactly once
# ---------------------------------------------------------------------------


def check_fees_charged_once(entries: Sequence[LedgerEntry]) -> None:
    """Each ``fill_id`` may contribute fees through at most one non-duplicate
    entry. ``BacktestLedger.apply_fill`` already guarantees this structurally
    — a repeat ``report_id`` produces a zero-delta ``"duplicate"`` entry — but
    this checks the *log*, so it also catches two distinct ``report_id``
    values that somehow carried the same ``fill_id`` (a venue or engine bug
    upstream of the ledger, not something ``apply_fill`` alone can rule out).
    """
    seen_fill_ids: set[str] = set()
    for entry in entries:
        if entry.kind not in ("buy", "sell", "failure"):
            continue
        if entry.fill_id is None:
            continue
        if entry.fill_id in seen_fill_ids:
            raise InvariantBreach(
                f"fill_id {entry.fill_id} charged fees more than once "
                f"(seq={entry.seq}, report_id={entry.report_id})"
            )
        seen_fill_ids.add(entry.fill_id)


def check_fee_reconciliation(ledger: BacktestLedger) -> None:
    """The sum of every entry's per-category fee must equal the ledger's
    running fee totals, exactly — the "fee reconciliation" property
    ``metrics/attribution.py`` depends on being true before it ever sees a
    report."""
    venue_total = sum(e.venue_fee_micro_usd for e in ledger.entries())
    network_total = sum(e.network_fee_micro_usd for e in ledger.entries())
    priority_total = sum(e.priority_fee_micro_usd for e in ledger.entries())
    if venue_total != ledger.venue_fee_micro_usd:
        raise InvariantBreach(
            f"venue fee total {ledger.venue_fee_micro_usd} does not match the "
            f"sum of entry-log venue fees {venue_total}"
        )
    if network_total != ledger.network_fee_micro_usd:
        raise InvariantBreach(
            f"network fee total {ledger.network_fee_micro_usd} does not match "
            f"the sum of entry-log network fees {network_total}"
        )
    if priority_total != ledger.priority_fee_micro_usd:
        raise InvariantBreach(
            f"priority fee total {ledger.priority_fee_micro_usd} does not "
            f"match the sum of entry-log priority fees {priority_total}"
        )


# ---------------------------------------------------------------------------
# #5 — every fill belongs to one order and one position lot
# ---------------------------------------------------------------------------


def check_fill_lot_integrity(ledger: BacktestLedger) -> None:
    """Every BUY entry must create exactly one lot, tied to exactly one
    ``fill_id``/``order_id``; every SELL entry's consumed lots must all trace
    back to a lot this same log actually created. A lot_id appearing in
    ``lots_consumed`` that was never ``lot_created`` by an earlier BUY entry
    would mean inventory was consumed that this ledger never recorded
    acquiring — a break in the one-fill-one-lot chain of custody.
    """
    lot_owner: dict[str, str] = {}
    for entry in ledger.entries():
        if entry.kind == "buy":
            if entry.lot_created is None or entry.fill_id is None:
                raise InvariantBreach(
                    f"BUY entry (seq={entry.seq}) did not create exactly one "
                    "lot tied to one fill"
                )
            if entry.lot_created in lot_owner:
                raise InvariantBreach(
                    f"lot_id {entry.lot_created} was created more than once"
                )
            lot_owner[entry.lot_created] = entry.fill_id
        elif entry.kind == "sell":
            for lot_id in entry.lots_consumed:
                if lot_id not in lot_owner:
                    raise InvariantBreach(
                        f"SELL entry (seq={entry.seq}) consumed lot {lot_id!r} "
                        "which no BUY entry in this log ever created"
                    )


# ---------------------------------------------------------------------------
# #6 — same inputs + hashes → byte-identical economic output
# ---------------------------------------------------------------------------


def _serialize_snapshot(snapshot: LedgerSnapshot) -> str:
    """A canonical, deterministic string form of a snapshot for byte-identical
    comparison. Sorted keys and no float formatting ambiguity: every field on
    ``LedgerSnapshot`` that participates in accounting is already an int, and
    ``Position`` fields are compared through their own dataclass equality via
    a plain dict of primitives here rather than ``repr``, so dict ordering
    cannot make two equal snapshots compare unequal."""
    positions: dict[str, Any] = {
        symbol: {
            "mint": position.mint,
            "quantity_atomic": position.quantity_atomic,
            "decimals": position.decimals,
            "avg_entry_price_usd": position.avg_entry_price_usd,
            "opened_at": position.opened_at,
            "cost_basis_usd": position.cost_basis_usd,
        }
        for symbol, position in sorted(snapshot.positions.items())
    }
    payload = {
        "cash_micro_usd": snapshot.cash_micro_usd,
        "starting_cash_micro_usd": snapshot.starting_cash_micro_usd,
        "realized_pnl_micro_usd": snapshot.realized_pnl_micro_usd,
        "venue_fee_micro_usd": snapshot.venue_fee_micro_usd,
        "network_fee_micro_usd": snapshot.network_fee_micro_usd,
        "priority_fee_micro_usd": snapshot.priority_fee_micro_usd,
        "positions": positions,
    }
    return json.dumps(payload, sort_keys=True)


def check_deterministic_replay(
    reports: Sequence[ExecutionReport],
    *,
    starting_cash_micro_usd: int,
) -> None:
    """Replay the same ``ExecutionReport`` sequence through two fresh ledgers
    and require byte-identical economic output.

    This is written against an explicit report sequence rather than against
    a live run because ``backtest/engine.py`` — the component that would
    actually produce two runs of "the same inputs and hashes" — does not
    exist yet. Once it does, its entry point is: run the engine twice with
    an identical manifest (same data partition hashes, config hash, seeds)
    and call this function (or the equivalent snapshot comparison) on the
    two resulting ledgers instead of on two fresh ones built here.
    """
    first = BacktestLedger(starting_cash_micro_usd=starting_cash_micro_usd)
    second = BacktestLedger(starting_cash_micro_usd=starting_cash_micro_usd)
    for report in reports:
        first.apply_fill(report)
    for report in reports:
        second.apply_fill(report)
    left = _serialize_snapshot(first.snapshot())
    right = _serialize_snapshot(second.snapshot())
    if left != right:
        raise InvariantBreach(
            "identical inputs produced different economic output:\n"
            f"  first:  {left}\n  second: {right}"
        )


# ---------------------------------------------------------------------------
# #7 — a dry run mutates nothing
# ---------------------------------------------------------------------------


def check_dry_run_no_mutation(ledger: BacktestLedger, report: ExecutionReport) -> None:
    """``BacktestLedger.preview_fill`` is the dry-run entry point. This
    confirms it holds the contract's promise: identical snapshot,
    identical idempotency state, before and after."""
    before_snapshot = _serialize_snapshot(ledger.snapshot())
    before_seen = ledger.seen(report.report_id)
    before_entry_count = len(ledger.entries())
    ledger.preview_fill(report)
    after_snapshot = _serialize_snapshot(ledger.snapshot())
    after_seen = ledger.seen(report.report_id)
    after_entry_count = len(ledger.entries())
    if (
        before_snapshot != after_snapshot
        or before_seen != after_seen
        or before_entry_count != after_entry_count
    ):
        raise InvariantBreach(
            f"preview_fill (dry run) mutated ledger state for report {report.report_id}"
        )


# ---------------------------------------------------------------------------
# #8 — an execution failure can never become a synthetic fill
# ---------------------------------------------------------------------------


def check_execution_failure_no_position_change(
    ledger: BacktestLedger, report: ExecutionReport
) -> None:
    """Apply a FAILED/EXPIRED report and require every symbol's position to
    be byte-identical before and after — only cash (gas) may move."""
    if report.state not in (OrderState.FAILED, OrderState.EXPIRED):
        raise InvariantBreach(
            f"check_execution_failure_no_position_change called with a "
            f"non-terminal-failure report state {report.state!r}"
        )
    before = _serialize_positions(ledger.snapshot())
    ledger.apply_fill(report)
    after = _serialize_positions(ledger.snapshot())
    if before != after:
        raise InvariantBreach(
            f"execution failure {report.report_id} changed position state: "
            f"{before!r} -> {after!r}"
        )


def _serialize_positions(snapshot: LedgerSnapshot) -> str:
    return json.dumps(
        {
            symbol: {
                "quantity_atomic": position.quantity_atomic,
                "cost_basis_usd": position.cost_basis_usd,
            }
            for symbol, position in sorted(snapshot.positions.items())
        },
        sort_keys=True,
    )


# ---------------------------------------------------------------------------
# #9 — risk resizing triggers a new quote at the approved size (audit C3)
# ---------------------------------------------------------------------------


def check_risk_resize_requotes(approved_vs_quoted: Sequence[tuple[int, int]]) -> None:
    """Each pair is ``(approved_size_atomic, quoted_size_atomic)`` for one
    order attempt after risk has bound it. Audit C3 / contracts §6: risk may
    lower a *bound*, but the execution layer must re-quote at the size it
    actually intends rather than shrinking an order that was already priced.

    ``backtest/engine.py`` does not exist yet, so this takes an explicit
    sequence rather than reading from a live run. Its integration point,
    once the engine exists, is step 9 of the tick order in contracts §5:
    every time ``risk.entry_bounds``/``exit_bounds`` produces a
    ``RiskBounds.max_notional_usd`` smaller than the strategy's originally
    intended size, the engine must obtain a *new* ``Quote`` at that bound
    before binding an order — record ``(bound_size, quote.token_amount_atomic)``
    pairs from that step and pass them here. ``broker._assert_bound`` already
    enforces the equivalent check at the live-broker layer
    (``QuoteBindingError`` on a size mismatch); this is the same rule stated
    as a property the engine's requoting logic must satisfy.
    """
    for approved, quoted in approved_vs_quoted:
        if approved != quoted:
            raise InvariantBreach(
                f"risk approved size {approved} atomic units but the order was "
                f"quoted at {quoted} — a resize must trigger a new quote at "
                "the approved size, not shrink an already-priced order"
            )


# ---------------------------------------------------------------------------
# #10 — SELL quantity == quoted quantity, exactly
# ---------------------------------------------------------------------------


def check_sell_quantity_matches_quote(fill: Fill, quote: Quote) -> None:
    """A settled SELL's traded token quantity must equal exactly what the
    quote it was bound to priced — no rounding, no partial fill silently
    accepted as a full one. Mirrors ``broker._assert_bound``'s
    ``intent.in_amount_atomic != quote.in_amount_atomic`` check, applied to
    the settled ``Fill`` instead of the pre-trade ``OrderIntent``, which is
    the version of the check a replay's execution model needs once a fill
    has actually landed.
    """
    if fill.side is not Side.SELL:
        return
    if fill.token_amount_atomic != quote.token_amount_atomic:
        raise InvariantBreach(
            f"SELL fill {fill.fill_id} settled {fill.token_amount_atomic} "
            f"atomic units but the bound quote priced "
            f"{quote.token_amount_atomic} — SELL quantity must equal quoted "
            "quantity exactly"
        )


# ---------------------------------------------------------------------------
# #11 — duplicate ExecutionReport.report_id is a no-op
# ---------------------------------------------------------------------------


def check_duplicate_report_is_noop(ledger: BacktestLedger, report: ExecutionReport) -> None:
    """Apply ``report`` twice in a row and require the second call to report
    ``False`` and leave the ledger byte-identical to how the first call left
    it."""
    ledger.apply_fill(report)
    before = _serialize_snapshot(ledger.snapshot())
    before_entry_count = len(ledger.entries())
    applied_again = ledger.apply_fill(report)
    after = _serialize_snapshot(ledger.snapshot())
    if applied_again:
        raise InvariantBreach(
            f"re-applying report_id {report.report_id} returned True instead "
            "of being treated as a no-op"
        )
    if before != after:
        raise InvariantBreach(
            f"re-applying report_id {report.report_id} changed ledger state"
        )
    # A duplicate is still visible in the log (as a zero-delta row), so the
    # entry count must grow by exactly one — auditable, but inert.
    if len(ledger.entries()) != before_entry_count + 1:
        raise InvariantBreach(
            f"duplicate application of {report.report_id} did not append "
            "exactly one zero-delta log entry"
        )


def check_duplicate_entries_are_zero(entries: Sequence[LedgerEntry]) -> None:
    """The state-only half of #11: every ``"duplicate"``-kind entry in the
    log must carry zero deltas everywhere. Unlike
    :func:`check_duplicate_report_is_noop`, this needs no ledger to call
    against — it is a property of the log alone, so :func:`check_all` can run
    it every tick without re-applying anything."""
    for entry in entries:
        if entry.kind != "duplicate":
            continue
        if (
            entry.cash_delta_micro_usd
            or entry.realized_pnl_delta_micro_usd
            or entry.quantity_delta_atomic
            or entry.venue_fee_micro_usd
            or entry.network_fee_micro_usd
            or entry.priority_fee_micro_usd
            or entry.lots_consumed
            or entry.lot_created is not None
        ):
            raise InvariantBreach(
                f"duplicate entry (seq={entry.seq}, report_id={entry.report_id}) "
                "carries a non-zero delta — a suppressed duplicate must be "
                "completely inert"
            )


# ---------------------------------------------------------------------------
# Aggregate — what the engine calls every tick
# ---------------------------------------------------------------------------


def check_all(ledger: BacktestLedger) -> None:
    """Every invariant that is checkable from ledger state alone, run in one
    call. Intended to be called every tick of the (not yet written) replay
    engine, immediately after each ``apply_fill``.

    Not included here, because they are properties of a sequence of *calls*
    rather than of the ledger's current state, and each needs data this
    function does not have:

    * #6 :func:`check_deterministic_replay` — needs a second, independent run.
    * #7 :func:`check_dry_run_no_mutation` — needs the report being previewed.
    * #9 :func:`check_risk_resize_requotes` — needs the risk/execution layer's
      approved-vs-quoted size log.
    * #10 :func:`check_sell_quantity_matches_quote` — needs the ``Quote`` a
      fill was bound to, which the ledger does not retain.
    * #11's call-level half, :func:`check_duplicate_report_is_noop` — needs
      the report to re-apply; :func:`check_duplicate_entries_are_zero` below
      *is* included, since it only reads the log.

    The engine should call all of the above directly at the point in the
    tick order (contracts §5) where the relevant data is in scope, in
    addition to calling this function.
    """
    entries = ledger.entries()
    snapshot = ledger.snapshot()
    check_cash_never_negative(snapshot)
    check_no_oversell(entries)
    check_cash_matches_entry_log(ledger)
    check_quantity_matches_entry_log(ledger)
    check_fees_charged_once(entries)
    check_fee_reconciliation(ledger)
    check_fill_lot_integrity(ledger)
    check_duplicate_entries_are_zero(entries)
