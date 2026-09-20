"""A paper venue that charges you for everything a real Solana swap charges,
and a ledger that survives being killed halfway through.

The point of this module is that the P&L it reports is *believable* and the
files it leaves on disk are *recoverable*. Both of those were audit findings.

What costs money, and what does not
===================================

* **The pool fee is zero on a routed quote, and that is not a bug.** Jupiter's
  ``outAmount`` is the number of tokens the pools actually send you, so every
  hop's AMM fee — along with slippage and price impact — is already deducted
  *inside* the route. Charging ``pool_fee_pct`` again bills the user twice. The
  live evidence is unambiguous: BONK quoted at 2.9636e-6 to buy and 2.9623e-6 to
  sell, a 4.4 bp round trip, where a 2-hop route billed at 0.25% per hop would
  imply 200 bp — **45x the observed spread**. The old code had a second,
  "degraded" path where a mid price with no fee baked in justified charging one;
  audit C4 deleted that path along with the object it filled against, so the
  fee is now unconditionally zero and :func:`pool_fee_micro` exists only to say
  so in one place where it can be tested.
* **Gas is additive and is charged on every attempt, including failed ones.**
  A failed Solana swap still pays the validator. Dropping the row would hide
  the cost, so a failure is a ``Fill`` with ``state=FAILED``, zero amounts and
  non-zero gas (audit C5c).
* **The failure model is a configured iid coin flip and the audit is right that
  this is not realism.** §15 asks for failures conditioned on congestion, quote
  age, route, priority fee and program error, and for observed priority fees
  rather than a constant ``gas_usd_per_swap``. Neither is implemented; both are
  deferred, and ``failed_tx_rate``/``gas_usd_per_swap`` are labelled here as
  assumptions rather than measurements so nobody cites them as evidence.

Everything the audit made structural
====================================

**C2 — atomic units are the fact; dollars are a rendering.** The old code set
``filled_usd`` to the *requested* dollars and derived ``quantity = filled_usd /
quote.price_usd``. Audit's worked example: $100 requested at a $1 mid, Jupiter
quotes 100 tokens for $95, the broker records $100 of proceeds and 105.263
tokens. This broker never sees a dollar amount it did not compute from an
integer. Cash is held internally as **integer micro-USDC** and token inventory
as integer atomic units, so every fill reconciles *exactly* — not to 1e-9, not
to a tolerance. Float dollars appear only on the way out, in the reporting
properties and on :class:`~.types.Fill`.

**C3 — a quote is bound to the swap it describes.** ``place_order`` recomputes
:func:`~.ids.quote_fingerprint` from the quote it was handed and refuses on
mismatch, and refuses when ``intent.in_amount_atomic != quote.in_amount_atomic``.
Risk may lower a *bound*; it may not shrink an order that has already been
priced. The execution layer re-quotes at the size it actually intends.

**C4 — a degraded route is structurally unable to get here.** ``place_order``
accepts :class:`~.types.Quote` and nothing else. ``quotes.py`` returns
:class:`~.types.ValuationEstimate` on failure, which has no path into this file.

**C5 — a quote is not a fill, and where we cannot model the gap we lean the
conservative way.** Fills settle at ``min_out_amount_atomic``, the
slippage-worst output, not the optimistic ``out_amount_atomic``
(:class:`FillModel` makes this configurable; the default is the conservative
one). ``slippage_bps_vs_quote`` records the difference. Quotes past
``expires_at`` or older than ``max_quote_age_seconds`` are refused, as are
quotes whose ``price_impact_pct`` exceeds the configured ceiling. What is still
*not* modelled — latency to signing, block height, confirmation, partial
completion of a sliced parent order, adverse selection, MEV — is not modelled,
and this docstring is the place that says so.

**§15 — decimals must be verified.** Both legs of the quote must carry
``TokenMeta.verified``. An inferred exponent is a silent factor-of-1000 error
in every quantity, and it arrives looking like a plausible price.

**C12 — the execution mode is a capability, not a flag.** ``--dry-run`` was
checked at exactly one call site (``loop.py:406``) and the stop-loss path went
around it, so an operator testing a decision could liquidate a position. A
``READ_ONLY`` broker computes a full Fill-shaped simulation and **persists
nothing at all** — no intent row, no ledger row, no state write, no in-memory
mutation. The exit-gate test asserts state and ledger are byte-identical across
a read-only run. ``ExecutionMode.LIVE`` is refused at construction by
:func:`assert_live_supported`: there is no live implementation and the failure
mode of one appearing by accident is unbounded.

Crash safety — the argument in full (C11)
=========================================

Three files, all append-or-replace, never partial-truncate:

* ``intents.jsonl`` — one row per intent, written **before any side effect**.
* ``trades.jsonl``  — one row per terminal Fill. Append-only. The ledger.
* ``state.json``    — the folded balances, plus ``last_fill_id``: the id of the
  last ledger row folded into them. Written with temp-file +
  atomic rename (``Path.replace``), which is atomic on POSIX and Windows alike.

``place_order`` does, in this exact order:

1. **Refuse** on any binding/validation failure. Nothing has been written, so
   there is nothing to undo.
2. **Return the existing Fill** if this ``intent_id`` already has a terminal row
   in the ledger. Idempotency: a retry of the same intent is one order.
3. **Append the intent row** and fsync.
4. Compute the Fill in memory.
5. **Append the fill row** to the ledger and fsync.
6. Apply the Fill to the in-memory book.
7. **Atomically write state.json** with ``last_fill_id`` set to that fill.

Every crash window is recoverable:

* *Between 3 and 5.* An intent row exists with no fill. Nothing was applied and
  no money moved. :meth:`LocalPaperBroker.reconcile` reports it as an open
  intent, which is exactly the state C11 says the old code could not name.
* *Between 5 and 7.* The ledger is ahead of the state. This is the split the
  old ``_commit`` produced and could not repair. :meth:`load` now replays every
  ledger row after ``last_fill_id`` onto the loaded balances, reproducing step 6
  deterministically — the fold is a pure function of the Fill and the book, and
  both are integers, so the replay is exact rather than approximately right.
* *During 7.* The atomic rename means the file is either wholly old or wholly
  new.
  Wholly old is the case above; wholly new is complete.
* *A partially written last line of a JSONL file.* Refused, loudly, rather than
  skipped: a truncated fill row is a fill whose amounts are unknown, and
  guessing is how a ledger stops being a ledger.

The ledger is therefore the source of truth and ``state.json`` is a checkpoint
of it. That ordering — log first, fold second — is the whole recovery story: a
trade that happened but was not checkpointed can be replayed, while a
checkpointed balance with no row behind it can never be explained.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .config import Config
from .ids import new_fill_id, new_order_id, new_run_id
from .ids import quote_fingerprint as _fingerprint
from .quotes import USDC_DECIMALS, USDC_MINT
from .types import (
    ExecutionMode,
    Fill,
    OrderIntent,
    OrderState,
    Position,
    Quote,
    Side,
)

#: Bumped whenever the on-disk shape changes. Version 1 stored float dollars and
#: had no run_id, no last_fill_id and no intent journal; it cannot be replayed
#: into version 2 because the micro-USDC integers it would need were never
#: recorded. A state file this build does not understand is refused rather than
#: silently misread.
SCHEMA_VERSION = 2

#: Micro-USDC per dollar. USDC's 6 decimals are a fixed property of the mint.
_MICRO = 10**USDC_DECIMALS

#: Oldest quote this broker will fill against, independent of ``expires_at``.
#: Jupiter publishes no expiry; ``quotes.DEFAULT_QUOTE_TTL_SECONDS`` asserts one
#: at construction, and this is the broker's own belt-and-braces bound for a
#: quote that arrived from somewhere else or from a replayed fixture. Audit C5d
#: and §11's "Stale price/quote → reject stale quote; re-quote exact amount →
#: **No new trade**".
MAX_QUOTE_AGE_SECONDS = 15.0


class FillModel(StrEnum):
    """Which output amount a simulated fill settles at. Audit C5a.

    ``MIN_OUT`` — ``otherAmountThreshold``, the slippage-worst output the route
    may deliver. The default, and the only defensible one for a system whose
    whole complaint about itself is that it used to be optimistic. It is still
    a *bound*, not a model: the real distribution of realised output between
    ``min_out`` and ``out`` depends on latency, competing flow and MEV, none of
    which are modelled. Assuming the bound means paper P&L understates rather
    than overstates, which is the direction an unvalidated simulator should err.

    ``EXPECTED_OUT`` — ``outAmount``. Available so that the cost of the
    conservatism can be *measured* (run both, diff the P&L) rather than argued
    about. Never the default.
    """

    MIN_OUT = "min_out"
    EXPECTED_OUT = "expected_out"


def pool_fee_micro(quote: Quote) -> int:
    """The pool fee to charge on top of a routed quote: always zero.

    Kept as a function rather than inlined as a ``0`` because the reasoning is
    the thing worth preserving and it needs somewhere to live and be tested.
    Jupiter's ``outAmount`` is already net of every hop's AMM fee; the observed
    BONK round trip was 4.4 bp against the 200 bp a 2-hop 0.25%-per-hop billing
    would have implied — a 45x double-count. Only gas is additive.
    """
    del quote
    return 0


class BrokerError(RuntimeError):
    """Base class for everything this module refuses to do.

    Everything below is raised *before* anything is submitted, journaled or
    applied, so a refusal costs nothing and leaves no row. That is the deliberate
    dividing line in this file: a **refusal** is an upstream bug or a stale input
    and raises; a **failure** is a market outcome, costs gas, and is a ``Fill``
    with ``state=FAILED``. Writing a FAILED row for a fingerprint mismatch would
    file a programming error in the ledger as though the chain had rejected it.
    Callers (``loop.py``) must catch ``BrokerError`` around ``place_order``.
    """


class InsufficientCash(BrokerError):
    """A BUY was larger than the cash available, gas included."""


class NoPosition(BrokerError):
    """A SELL was placed for a symbol with no open position."""


class InsufficientPosition(BrokerError):
    """A SELL was quoted for more tokens than are held.

    The old broker *clamped* here and filled the smaller amount against the
    larger quote. That is audit C3 in miniature — the executed size no longer
    matches the size that was priced — so clamping is gone. The caller must
    quote the amount it actually holds.
    """


class QuoteBindingError(BrokerError):
    """The quote does not describe the swap the intent authorised. Audit C3."""


class QuoteRejected(BrokerError):
    """The quote is structurally unusable: stale, expired, too much impact, or
    built on unverified decimals. Audit C5d/C5e and §15."""


class ReadOnlyViolation(BrokerError):
    """A mutating operation was attempted on a ``READ_ONLY`` broker. Audit C12."""


class LiveModeUnsupported(BrokerError):
    """``ExecutionMode.LIVE`` was requested and there is no implementation."""


class LedgerCorrupt(BrokerError):
    """A journal or state file could not be read as what it claims to be.

    Never repaired and never skipped. Resetting to starting cash would quietly
    erase the entire P&L history, and skipping an unparseable ledger row would
    silently drop a trade — the two worst things a ledger can do.
    """


def assert_live_supported(mode: ExecutionMode) -> None:
    """Refuse ``LIVE`` until a real venue adapter exists. Audit C12 / Phase 1.

    The audit's Phase 1 opens with "disable/deprecate any live-broker work and
    place a clear 'research only' gate at startup". This is that gate, and it is
    a function rather than a comment so that the day someone writes a live
    adapter they have to delete a line whose name says what they are doing.
    """
    if mode is ExecutionMode.LIVE:
        raise LiveModeUnsupported(
            "ExecutionMode.LIVE has no implementation. This is a research "
            "system: there is no signer, no transaction builder, no "
            "confirmation monitor and no venue reconciliation. Running live "
            "would submit nothing and report success."
        )


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """What startup found on disk. The C11 recovery report.

    ``open_intents`` is the state the old code could not name: an intent that
    was journaled and never reached a terminal Fill. It means a crash happened
    between the intent write and the ledger append, so nothing moved — but on a
    live venue it would mean "we may or may not have an order out there", which
    is why §11 says **reconcile before any new order**.
    """

    ledger_fills: int
    journaled_intents: int
    replayed_fill_ids: tuple[str, ...]
    open_intents: tuple[OrderIntent, ...]

    @property
    def clean(self) -> bool:
        return not self.replayed_fill_ids and not self.open_intents


@dataclass(slots=True)
class _Lot:
    """Inventory for one symbol, in integers only.

    ``basis_micro`` includes gas, so a position's break-even is its true
    break-even and a fill marks at a small loss the instant it lands — which is
    correct and which no amount of optimism should paper over.
    ``entry_notional_micro`` is the fee-free notional, kept separately because
    the *average entry price* is a price and must not have gas folded into it;
    the cost drag belongs in the basis, which is what P&L is measured against.
    """

    mint: str
    decimals: int
    quantity_atomic: int
    basis_micro: int
    entry_notional_micro: int
    opened_at: float


class LocalPaperBroker:
    """The :class:`~.types.Broker` protocol, backed by an append-only ledger.

    ``place_order(intent, quote, *, now)`` matches the protocol exactly. The old
    signature took a symbol, a side and a dollar amount, which is what let it
    invent a quantity.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        mode: ExecutionMode = ExecutionMode.PAPER,
        run_id: str | None = None,
        rng: random.Random | None = None,
        fill_model: FillModel = FillModel.MIN_OUT,
        max_quote_age_seconds: float = MAX_QUOTE_AGE_SECONDS,
        max_price_impact_pct: float | None = None,
    ) -> None:
        assert_live_supported(mode)
        self.cfg = cfg
        self._mode = mode
        self._run_id = run_id or new_run_id()
        # Injectable so tests are deterministic. The default is seeded from the
        # OS, because a reproducible live run would be a lie of a different kind.
        self._rng = rng if rng is not None else random.Random()
        self._fill_model = FillModel(fill_model)
        self._max_quote_age_seconds = float(max_quote_age_seconds)
        # Defaults to the risk layer's ceiling so the broker cannot be *more*
        # permissive than the thing that is supposed to be gating it. A 3%
        # one-way allowance already implies more than 6% round trip (audit C10).
        self._max_price_impact_pct = (
            float(cfg.risk.max_price_impact_pct)
            if max_price_impact_pct is None
            else float(max_price_impact_pct)
        )

        self._cash_micro = _to_micro(cfg.starting_cash_usd)
        self._starting_cash_micro = self._cash_micro
        self._positions: dict[str, _Lot] = {}
        self._realized_micro = 0
        self._fees_micro = 0
        self._gas_micro = 0
        self._failed_gas_micro = 0
        self._last_fill_id: str | None = None
        self._fill_by_intent: dict[str, Fill] = {}

        self._reconciliation = self.load()

    # -- identity ----------------------------------------------------------

    @property
    def mode(self) -> ExecutionMode:
        return self._mode

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def fill_model(self) -> FillModel:
        return self._fill_model

    @property
    def intents_path(self) -> Path:
        return self.cfg.data_dir / "intents.jsonl"

    @property
    def ledger_path(self) -> Path:
        return self.cfg.trades_path

    # -- read-only view ----------------------------------------------------

    @property
    def cash_usd(self) -> float:
        return self._cash_micro / _MICRO

    @property
    def cash_micro_usd(self) -> int:
        """Cash in integer micro-USDC. The authoritative number; ``cash_usd`` is
        a rendering of it. Exposed because the exit-gate reconciliation test has
        to compare exactly, and an exact comparison of floats is a wish."""
        return self._cash_micro

    @property
    def realized_pnl_usd(self) -> float:
        """Cumulative since inception, net of the gas of each exit."""
        return self._realized_micro / _MICRO

    @property
    def fees_paid_usd(self) -> float:
        """Explicit pool fees. Zero by construction on routed quotes — see
        :func:`pool_fee_micro`. Not the same as zero economic trading cost: the
        AMM fee and impact are embedded in ``outAmount`` (audit §10)."""
        return self._fees_micro / _MICRO

    @property
    def gas_paid_usd(self) -> float:
        """Includes gas burned on failed transactions."""
        return self._gas_micro / _MICRO

    @property
    def failed_gas_usd(self) -> float:
        return self._failed_gas_micro / _MICRO

    @property
    def starting_cash_usd(self) -> float:
        """From the state file once it exists, so editing ``config.toml``
        mid-run cannot retroactively rewrite the return number."""
        return self._starting_cash_micro / _MICRO

    @property
    def reconciliation(self) -> Reconciliation:
        """What the last :meth:`load` found. Startup should log this."""
        return self._reconciliation

    def get_positions(self) -> dict[str, Position]:
        return {
            symbol: _lot_to_position(symbol, lot) for symbol, lot in self._positions.items()
        }

    # -- orders ------------------------------------------------------------

    def place_order(self, intent: OrderIntent, quote: Quote, *, now: float) -> Fill:
        """Attempt one swap against an exact, bound quote.

        Returns a ``Fill`` for every attempt that was actually made, successful
        or not. Raises :class:`BrokerError` for anything refused before an
        attempt — see that class for why the two are not the same thing.

        Idempotent on ``intent.intent_id``: re-submitting an intent that already
        has a terminal row in the ledger returns that row and does nothing else.
        """
        # Idempotency comes before validation on purpose. A retry after a crash
        # is replaying an order that already happened; whether its quote has
        # since expired is irrelevant and refusing would turn a completed trade
        # into an exception.
        existing = self._fill_by_intent.get(intent.intent_id)
        if existing is not None:
            return existing

        self._assert_bound(intent, quote, now=now)

        gas_micro = _to_micro(self.cfg.execution.gas_usd_per_swap)
        fee_micro = pool_fee_micro(quote)
        token = quote.token_meta

        if quote.side is Side.BUY:
            required = quote.in_amount_atomic + gas_micro + fee_micro
            if self._cash_micro < required:
                raise InsufficientCash(
                    f"BUY {quote.symbol} needs {required / _MICRO:,.6f} USDC "
                    f"(input {quote.in_amount_atomic / _MICRO:,.6f} + gas "
                    f"{gas_micro / _MICRO:,.6f}) but cash is "
                    f"{self._cash_micro / _MICRO:,.6f}"
                )
        else:
            lot = self._positions.get(quote.symbol)
            if lot is None:
                raise NoPosition(f"SELL {quote.symbol}: no open position")
            if quote.in_amount_atomic > lot.quantity_atomic:
                raise InsufficientPosition(
                    f"SELL {quote.symbol}: quote is for {quote.in_amount_atomic} "
                    f"atomic units but only {lot.quantity_atomic} are held. "
                    f"Re-quote at the held size; clamping would execute a size "
                    f"that was never priced (audit C3)."
                )

        if self._mode is ExecutionMode.READ_ONLY:
            # C12: compute the answer, write nothing, touch nothing. The failure
            # coin flip is deliberately not drawn — a read-only run is answering
            # "what would this order look like", and a random failure would make
            # that answer irreproducible without making it more informative.
            return self._build_fill(
                intent,
                quote,
                now=now,
                gas_micro=gas_micro,
                fee_micro=fee_micro,
                failed=False,
                note="read_only simulation; nothing persisted",
            )

        # Step 3 of the crash-safety ordering: the intent is durable before any
        # side effect, so a crash from here on leaves a row recovery can find.
        self._journal_intent(intent)

        # The chain's coin flip. Drawn after validation and after the intent is
        # durable, so a seeded rng replays a run exactly.
        failed = self._rng.random() < self.cfg.execution.failed_tx_rate
        fill = self._build_fill(
            intent,
            quote,
            now=now,
            gas_micro=gas_micro,
            fee_micro=fee_micro,
            failed=failed,
            note="transaction failed; gas charged" if failed else None,
        )

        self._append_fill(fill, mint=token.mint)  # step 5: the ledger, before the fold
        self._apply(fill, mint=token.mint, decimals=token.decimals)  # step 6
        self._last_fill_id = fill.fill_id
        self._fill_by_intent[intent.intent_id] = fill
        self.save()  # step 7
        return fill

    # -- binding and admission --------------------------------------------

    def _assert_bound(self, intent: OrderIntent, quote: Quote, *, now: float) -> None:
        """Every reason this order may not be sent. All of them raise.

        Ordered cheapest-and-most-structural first, so the error a developer
        sees is the root one rather than a downstream symptom.
        """
        if intent.symbol != quote.symbol:
            raise QuoteBindingError(
                f"intent is for {intent.symbol} but quote is for {quote.symbol}"
            )
        if intent.side is not quote.side:
            raise QuoteBindingError(f"intent side {intent.side} != quote side {quote.side}")

        # C3: recompute, never trust. The quote object could have been built
        # anywhere; the fingerprint is the only thing that says it describes the
        # swap it claims to.
        recomputed = _fingerprint(
            side=str(quote.side),
            input_mint=quote.input_token.mint,
            output_mint=quote.output_token.mint,
            in_amount_atomic=quote.in_amount_atomic,
            out_amount_atomic=quote.out_amount_atomic,
            slot=quote.context_slot,
        )
        if recomputed != quote.fingerprint:
            raise QuoteBindingError(
                f"quote fingerprint {quote.fingerprint} does not match the swap "
                f"it describes (recomputed {recomputed}) — the quote was "
                f"mutated after it was obtained (audit C3)"
            )
        if intent.in_amount_atomic != quote.in_amount_atomic:
            raise QuoteBindingError(
                f"intent authorises {intent.in_amount_atomic} atomic units but "
                f"the quote prices {quote.in_amount_atomic}. Re-quote at the "
                f"intended size; risk may lower a bound, not resize a priced "
                f"order (audit C3)."
            )
        if intent.in_amount_atomic > intent.max_in_amount_atomic:
            raise QuoteBindingError(
                f"intent size {intent.in_amount_atomic} exceeds its own bound "
                f"{intent.max_in_amount_atomic}"
            )

        # §15: decimals for execution must be authoritative. An inferred
        # exponent is a silent factor-of-1000 error in every quantity below.
        for leg in (quote.input_token, quote.output_token):
            if not leg.verified:
                raise QuoteRejected(
                    f"{quote.symbol}: refusing to fill on unverified decimals "
                    f"for {leg.mint} (source={leg.source})"
                )

        # The USD leg must actually be USDC, or "notional_usd" is not dollars.
        usd_leg = quote.input_token if quote.side is Side.BUY else quote.output_token
        if usd_leg.mint != USDC_MINT:
            raise QuoteRejected(
                f"{quote.symbol}: the USD leg is {usd_leg.mint}, not USDC — "
                f"this broker's unit of account is USDC and nothing here can "
                f"price another quote token"
            )

        # A route that outputs nothing is not a trade, it is a donation of gas.
        # It is reachable with legitimate inputs — selling a few atomic units of
        # a 5-decimal token worth 2.9e-6 rounds to zero micro-USDC — so it is a
        # refusal rather than an assertion. ``min_out`` is checked too because
        # the conservative fill model settles at *that* number, and a quote whose
        # worst case is zero output is a quote whose worst case is a total loss.
        if quote.out_amount_atomic == 0 or quote.min_out_amount_atomic == 0:
            raise QuoteRejected(
                f"{quote.symbol}: route outputs {quote.out_amount_atomic} "
                f"(worst case {quote.min_out_amount_atomic}) for "
                f"{quote.in_amount_atomic} in — dust below one unit of the "
                f"output token cannot be sold for more than it costs in gas"
            )

        # C5d. Two independent bounds: the quote's own stated expiry, and our
        # own maximum age for a quote that arrived without one or with a
        # generous one.
        if quote.is_expired(now):
            raise QuoteRejected(
                f"{quote.symbol}: quote expired at {quote.expires_at} and it is now {now}"
            )
        age = quote.age_seconds(now)
        if age > self._max_quote_age_seconds:
            raise QuoteRejected(
                f"{quote.symbol}: quote is {age:.2f}s old, ceiling is "
                f"{self._max_quote_age_seconds:.2f}s — re-quote at the exact "
                f"amount rather than filling against stale pool state"
            )
        if age < 0:
            raise QuoteRejected(
                f"{quote.symbol}: quote was received {-age:.2f}s in the future; "
                f"a clock problem must halt rather than trade"
            )

        # C5e.
        if quote.price_impact_pct > self._max_price_impact_pct:
            raise QuoteRejected(
                f"{quote.symbol}: price impact {quote.price_impact_pct:.4f}% "
                f"exceeds the {self._max_price_impact_pct:.4f}% ceiling"
            )

    # -- fill construction -------------------------------------------------

    def _build_fill(
        self,
        intent: OrderIntent,
        quote: Quote,
        *,
        now: float,
        gas_micro: int,
        fee_micro: int,
        failed: bool,
        note: str | None,
    ) -> Fill:
        """Turn a bound quote into the Fill it would settle as.

        Pure: it reads no broker state and writes none, which is what lets
        READ_ONLY use the identical code path as PAPER and lets the exit-gate
        test compare the two.
        """
        token = quote.token_meta

        if failed:
            # C5c. Zero amounts, non-zero gas. A failed swap still costs money
            # and a missing row would hide that.
            return Fill(
                fill_id=new_fill_id(),
                order_id=new_order_id(),
                intent_id=intent.intent_id,
                decision_id=intent.decision_id,
                ts=now,
                symbol=quote.symbol,
                side=quote.side,
                state=OrderState.FAILED,
                in_amount_atomic=0,
                out_amount_atomic=0,
                token_amount_atomic=0,
                token_decimals=token.decimals,
                quote_fingerprint=quote.fingerprint,
                price_usd=None,  # nothing traded, so no price was realised
                notional_usd=0.0,
                price_impact_pct=quote.price_impact_pct,
                pool_fee_usd=0.0,  # the pool never executed; only gas was spent
                gas_usd=gas_micro / _MICRO,
                realized_pnl_usd=0.0,
                slippage_bps_vs_quote=None,
                note=note,
            )

        out_atomic = (
            quote.min_out_amount_atomic
            if self._fill_model is FillModel.MIN_OUT
            else quote.out_amount_atomic
        )
        # Negative means worse than the expected output, which is the only
        # direction the conservative model can produce. Sign is meaningful.
        slippage_bps = (
            10_000.0 * (out_atomic - quote.out_amount_atomic) / quote.out_amount_atomic
        )

        if quote.side is Side.BUY:
            usd_micro = quote.in_amount_atomic
            token_atomic = out_atomic
        else:
            usd_micro = out_atomic
            token_atomic = quote.in_amount_atomic

        token_ui = token_atomic / (10**token.decimals)
        price_usd = (usd_micro / _MICRO) / token_ui if token_ui > 0 else None

        realized_micro = 0
        if quote.side is Side.SELL:
            lot = self._positions[quote.symbol]
            realized_micro = (
                usd_micro - _basis_share(lot, token_atomic) - gas_micro - fee_micro
            )

        return Fill(
            fill_id=new_fill_id(),
            order_id=new_order_id(),
            intent_id=intent.intent_id,
            decision_id=intent.decision_id,
            ts=now,
            symbol=quote.symbol,
            side=quote.side,
            state=OrderState.LANDED,
            in_amount_atomic=quote.in_amount_atomic,
            out_amount_atomic=out_atomic,
            token_amount_atomic=token_atomic,
            token_decimals=token.decimals,
            quote_fingerprint=quote.fingerprint,
            price_usd=price_usd,
            notional_usd=usd_micro / _MICRO,
            price_impact_pct=quote.price_impact_pct,
            pool_fee_usd=fee_micro / _MICRO,
            gas_usd=gas_micro / _MICRO,
            realized_pnl_usd=realized_micro / _MICRO,
            slippage_bps_vs_quote=slippage_bps,
            note=note,
        )

    # -- the fold ----------------------------------------------------------

    def _apply(self, fill: Fill, *, mint: str, decimals: int) -> None:
        """Fold one Fill into the book. Integers only, so it is exact.

        Called once by :meth:`place_order` and again by :meth:`load` when the
        ledger is ahead of the checkpoint. It must therefore be deterministic in
        the Fill and the current book and in nothing else — no clock, no rng, no
        config. That property is what makes crash recovery a replay rather than
        a guess.

        The conservation identity this maintains, in integer micro-USDC::

            cash + sum(basis) == starting_cash + realized - failed_gas

        BUY moves ``input + gas`` from cash into basis, so the left side is
        unchanged and so is the right. SELL adds ``proceeds - gas`` to cash,
        removes ``basis_share`` from basis, and books
        ``proceeds - basis_share - gas`` as realized: both sides move by the
        same amount. A FAILED attempt burns gas on both sides. If any split is
        wrong anywhere in this method the identity stops balancing, exactly.
        """
        gas_micro = _to_micro(fill.gas_usd)
        fee_micro = _to_micro(fill.pool_fee_usd)
        self._gas_micro += gas_micro

        if fill.failed:
            self._cash_micro -= gas_micro
            self._failed_gas_micro += gas_micro
            return

        self._fees_micro += fee_micro

        if fill.side is Side.BUY:
            cost = fill.in_amount_atomic + gas_micro + fee_micro
            self._cash_micro -= cost
            lot = self._positions.get(fill.symbol)
            if lot is None:
                self._positions[fill.symbol] = _Lot(
                    mint=mint,
                    decimals=decimals,
                    quantity_atomic=fill.token_amount_atomic,
                    basis_micro=cost,
                    entry_notional_micro=fill.in_amount_atomic,
                    opened_at=fill.ts,
                )
            else:
                lot.quantity_atomic += fill.token_amount_atomic
                lot.basis_micro += cost
                lot.entry_notional_micro += fill.in_amount_atomic
                # opened_at is untouched: a position's age is measured from the
                # first entry, not from the last scale-in.
            return

        lot = self._positions[fill.symbol]
        sold = fill.token_amount_atomic
        basis_share = _basis_share(lot, sold)
        entry_share = _proportional(lot.entry_notional_micro, sold, lot.quantity_atomic)
        proceeds = fill.out_amount_atomic

        self._cash_micro += proceeds - gas_micro - fee_micro
        self._realized_micro += proceeds - basis_share - gas_micro - fee_micro

        lot.quantity_atomic -= sold
        lot.basis_micro -= basis_share
        lot.entry_notional_micro -= entry_share
        if lot.quantity_atomic == 0:
            # Exact, because the quantity is an integer. The old code needed a
            # relative epsilon here to stop 1e-16 of a token being left behind;
            # atomic units make dust a thing that either exists or does not.
            del self._positions[fill.symbol]

    # -- persistence -------------------------------------------------------

    def load(self) -> Reconciliation:
        """Read the checkpoint, replay the ledger past it, report what was found.

        A missing state file is a cold start, not an error. A corrupt or
        future-versioned one *is* an error — see :class:`LedgerCorrupt`.
        """
        self._cash_micro = _to_micro(self.cfg.starting_cash_usd)
        self._starting_cash_micro = self._cash_micro
        self._positions = {}
        self._realized_micro = 0
        self._fees_micro = 0
        self._gas_micro = 0
        self._failed_gas_micro = 0
        self._last_fill_id = None
        self._fill_by_intent = {}

        path = self.cfg.state_path
        if path.exists():
            payload = _read_json(path)
            version = int(payload.get("schema_version", 0))
            if version != SCHEMA_VERSION:
                raise LedgerCorrupt(
                    f"{path} has schema_version {version}, this build writes and "
                    f"reads {SCHEMA_VERSION}. Version 1 stored float dollars and "
                    f"no fill checkpoint, so it cannot be migrated without "
                    f"inventing the integers it never recorded. Move it aside."
                )
            self._cash_micro = int(payload["cash_micro_usd"])
            self._starting_cash_micro = int(payload["starting_cash_micro_usd"])
            self._realized_micro = int(payload.get("realized_pnl_micro_usd", 0))
            self._fees_micro = int(payload.get("fees_micro_usd", 0))
            self._gas_micro = int(payload.get("gas_micro_usd", 0))
            self._failed_gas_micro = int(payload.get("failed_gas_micro_usd", 0))
            self._last_fill_id = payload.get("last_fill_id") or None
            self._positions = {
                symbol: _Lot(
                    mint=str(raw["mint"]),
                    decimals=int(raw["decimals"]),
                    quantity_atomic=int(raw["quantity_atomic"]),
                    basis_micro=int(raw["basis_micro_usd"]),
                    entry_notional_micro=int(raw["entry_notional_micro_usd"]),
                    opened_at=float(raw["opened_at"]),
                )
                for symbol, raw in (payload.get("positions") or {}).items()
            }

        ledger = _read_jsonl(self.ledger_path)
        fills = [_fill_from_row(row, self.ledger_path) for row in ledger]
        mints = {
            row["fill_id"]: (str(row["mint"]), int(row["token_decimals"])) for row in ledger
        }

        # Everything after the checkpoint is a fill the ledger recorded and the
        # state never folded in. That is the crash-between-5-and-7 window.
        replayed: list[str] = []
        seen_checkpoint = self._last_fill_id is None
        for fill in fills:
            self._fill_by_intent[fill.intent_id] = fill
            if not seen_checkpoint:
                if fill.fill_id == self._last_fill_id:
                    seen_checkpoint = True
                continue
            mint, decimals = mints[fill.fill_id]
            self._apply(fill, mint=mint, decimals=decimals)
            replayed.append(fill.fill_id)

        if not seen_checkpoint:
            raise LedgerCorrupt(
                f"{self.cfg.state_path} checkpoints fill {self._last_fill_id}, "
                f"which is not in {self.ledger_path}. The checkpoint is ahead of "
                f"the ledger, which this design makes impossible — the ledger is "
                f"always written first. Something truncated or replaced a file."
            )
        if replayed:
            self._last_fill_id = replayed[-1]

        intents = {
            row["intent_id"]: _intent_from_row(row)
            for row in _read_jsonl(self.intents_path)
        }
        open_intents = tuple(
            intent
            for intent_id, intent in intents.items()
            if intent_id not in self._fill_by_intent
        )

        self._reconciliation = Reconciliation(
            ledger_fills=len(fills),
            journaled_intents=len(intents),
            replayed_fill_ids=tuple(replayed),
            open_intents=open_intents,
        )
        return self._reconciliation

    def reconcile(self) -> Reconciliation:
        """Re-read everything from disk and report. The startup entry point.

        ``loop.py`` should call this before creating any intent and refuse to
        trade while ``open_intents`` is non-empty — §11: "Transaction
        expires/drops → **Reconcile before any new order**". On this paper venue
        an open intent means no money moved; on a live venue it would mean an
        order of unknown status, and the correct response is the same either way.
        """
        return self.load()

    def save(self) -> None:
        """Checkpoint the folded balances. Atomic; never a partial truncate.

        Temp file in the same directory, fsync, then ``os.replace``, which is
        atomic on both POSIX and Windows. ``last_fill_id`` is what makes this a
        checkpoint of the ledger rather than a second, competing source of
        truth.
        """
        if not self._mode.may_mutate:
            raise ReadOnlyViolation(
                f"a {self._mode} broker may not write {self.cfg.state_path}"
            )
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self._run_id,
            "last_fill_id": self._last_fill_id,
            "cash_micro_usd": self._cash_micro,
            "starting_cash_micro_usd": self._starting_cash_micro,
            "realized_pnl_micro_usd": self._realized_micro,
            "fees_micro_usd": self._fees_micro,
            "gas_micro_usd": self._gas_micro,
            "failed_gas_micro_usd": self._failed_gas_micro,
            "positions": {
                symbol: {
                    "mint": lot.mint,
                    "decimals": lot.decimals,
                    "quantity_atomic": lot.quantity_atomic,
                    "basis_micro_usd": lot.basis_micro,
                    "entry_notional_micro_usd": lot.entry_notional_micro,
                    "opened_at": lot.opened_at,
                }
                for symbol, lot in sorted(self._positions.items())
            },
        }
        _atomic_write_text(
            self.cfg.state_path,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )

    # -- internals ---------------------------------------------------------

    def _journal_intent(self, intent: OrderIntent) -> None:
        if not self._mode.may_mutate:
            raise ReadOnlyViolation(f"a {self._mode} broker may not journal intents")
        _append_jsonl(
            self.intents_path,
            {
                "intent_id": intent.intent_id,
                "decision_id": intent.decision_id,
                "action_id": intent.action_id,
                "run_id": intent.run_id,
                "ts": intent.ts,
                "symbol": intent.symbol,
                "side": str(intent.side),
                "in_amount_atomic": intent.in_amount_atomic,
                "max_in_amount_atomic": intent.max_in_amount_atomic,
                "source": intent.source,
                "reason": intent.reason,
            },
        )

    def _append_fill(self, fill: Fill, *, mint: str) -> None:
        if not self._mode.may_mutate:
            raise ReadOnlyViolation(f"a {self._mode} broker may not append fills")
        _append_jsonl(
            self.ledger_path,
            {
                "fill_id": fill.fill_id,
                "order_id": fill.order_id,
                "intent_id": fill.intent_id,
                "decision_id": fill.decision_id,
                "run_id": self._run_id,
                "ts": fill.ts,
                "symbol": fill.symbol,
                # The mint is not on Fill but recovery needs it to rebuild a
                # position, so the ledger row carries it.
                "mint": mint,
                "side": str(fill.side),
                "state": str(fill.state),
                "in_amount_atomic": fill.in_amount_atomic,
                "out_amount_atomic": fill.out_amount_atomic,
                "token_amount_atomic": fill.token_amount_atomic,
                "token_decimals": fill.token_decimals,
                "quote_fingerprint": fill.quote_fingerprint,
                "price_usd": fill.price_usd,
                "notional_usd": fill.notional_usd,
                "price_impact_pct": fill.price_impact_pct,
                "pool_fee_usd": fill.pool_fee_usd,
                "gas_usd": fill.gas_usd,
                "realized_pnl_usd": fill.realized_pnl_usd,
                "slippage_bps_vs_quote": fill.slippage_bps_vs_quote,
                "note": fill.note,
            },
        )


# ---------------------------------------------------------------------------
# Integer money
# ---------------------------------------------------------------------------


def _to_micro(usd: float) -> int:
    """Dollars to integer micro-USDC, rounded to the nearest unit.

    Rounding, not truncation: ``0.21`` has no exact binary representation, and
    ``int(0.21 * 1e6)`` can land on 209999, which would make gas cost a
    different amount depending on how it was spelled. Rounding is also what
    makes the reverse trip exact — a float produced as ``micro / 1e6`` converts
    back to the same integer for any micro amount below 2^53 (about 9 billion
    dollars), which is what lets :meth:`LocalPaperBroker._apply` read gas and
    fees back off a ``Fill`` without drift.
    """
    return round(float(usd) * _MICRO)


def _proportional(total: int, part: int, whole: int) -> int:
    """``total * part / whole`` in integers, floor, with an exact full case.

    Floor rather than round so that repeated partial exits can never release
    more basis than the position holds; the remainder stays with the position
    and is released by the final exit, which takes the whole of what is left.
    """
    if whole <= 0:
        return 0
    if part >= whole:
        return total
    return total * part // whole


def _basis_share(lot: _Lot, sold_atomic: int) -> int:
    return _proportional(lot.basis_micro, sold_atomic, lot.quantity_atomic)


def _lot_to_position(symbol: str, lot: _Lot) -> Position:
    quantity_ui = lot.quantity_atomic / (10**lot.decimals)
    avg_entry = (
        (lot.entry_notional_micro / _MICRO) / quantity_ui if quantity_ui > 0 else 0.0
    )
    return Position(
        symbol=symbol,
        mint=lot.mint,
        quantity_atomic=lot.quantity_atomic,
        decimals=lot.decimals,
        avg_entry_price_usd=avg_entry,
        opened_at=lot.opened_at,
        cost_basis_usd=lot.basis_micro / _MICRO,
    )


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise LedgerCorrupt(
            f"{path} is corrupt and will not be overwritten: {exc}. "
            f"Move it aside to start fresh — resetting it automatically would "
            f"erase the entire P&L history."
        ) from exc
    if not isinstance(payload, dict):
        raise LedgerCorrupt(f"{path} is not a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict]:
    """Every row, in order. A malformed row is fatal, never skipped."""
    if not path.exists():
        return []
    rows: list[dict] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LedgerCorrupt(
                f"{path}:{number} is not valid JSON: {exc}. A truncated row is a "
                f"trade whose amounts are unknown; skipping it would silently "
                f"drop it from the book."
            ) from exc
        if not isinstance(row, dict):
            raise LedgerCorrupt(f"{path}:{number} is not a JSON object")
        rows.append(row)
    return rows


def _fill_from_row(row: dict, path: Path) -> Fill:
    try:
        return Fill(
            fill_id=str(row["fill_id"]),
            order_id=str(row["order_id"]),
            intent_id=str(row["intent_id"]),
            decision_id=row.get("decision_id"),
            ts=float(row["ts"]),
            symbol=str(row["symbol"]),
            side=Side(row["side"]),
            state=OrderState(row["state"]),
            in_amount_atomic=int(row["in_amount_atomic"]),
            out_amount_atomic=int(row["out_amount_atomic"]),
            token_amount_atomic=int(row["token_amount_atomic"]),
            token_decimals=int(row["token_decimals"]),
            quote_fingerprint=str(row["quote_fingerprint"]),
            price_usd=row.get("price_usd"),
            notional_usd=float(row["notional_usd"]),
            price_impact_pct=row.get("price_impact_pct"),
            pool_fee_usd=float(row["pool_fee_usd"]),
            gas_usd=float(row["gas_usd"]),
            realized_pnl_usd=float(row["realized_pnl_usd"]),
            slippage_bps_vs_quote=row.get("slippage_bps_vs_quote"),
            note=row.get("note"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LedgerCorrupt(f"{path}: unreadable fill row {row!r}: {exc}") from exc


def _intent_from_row(row: dict) -> OrderIntent:
    return OrderIntent(
        intent_id=str(row["intent_id"]),
        decision_id=row.get("decision_id"),
        action_id=row.get("action_id"),
        run_id=str(row.get("run_id", "")),
        ts=float(row["ts"]),
        symbol=str(row["symbol"]),
        side=Side(row["side"]),
        in_amount_atomic=int(row["in_amount_atomic"]),
        max_in_amount_atomic=int(row["max_in_amount_atomic"]),
        source=row.get("source", "strategy"),
        reason=row.get("reason", ""),
    )


def _append_jsonl(path: Path, row: dict) -> None:
    """Append one row and fsync it.

    The fsync is the difference between "the row is in the page cache" and "the
    row survives a power loss", and the crash-safety argument in the module
    docstring depends on the second one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _atomic_write_text(path: Path, text: str) -> None:
    """Replace ``path`` wholesale, or leave it exactly as it was. Audit C11.

    The old code opened the state file for writing and serialised into it, so a
    crash — or a full disk — left a truncated JSON document where the balances
    used to be, and the next start could not tell that from a cold start. Here
    the new contents are written to a temp file in the *same directory* (so the
    rename cannot cross a filesystem boundary), fsynced, and then moved over the
    target with :meth:`Path.replace`, which is atomic on POSIX and on Windows.
    A reader therefore sees either the whole previous checkpoint or the whole
    new one.

    The cleanup catches ``BaseException`` rather than ``Exception`` so that
    Ctrl+C during a write does not leave a ``.state.json.*.tmp`` behind; the
    debris is harmless but it is the kind of thing that gets mistaken for
    evidence later.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        tmp_path.replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise


__all__ = [
    "MAX_QUOTE_AGE_SECONDS",
    "SCHEMA_VERSION",
    "BrokerError",
    "FillModel",
    "InsufficientCash",
    "InsufficientPosition",
    "LedgerCorrupt",
    "LiveModeUnsupported",
    "LocalPaperBroker",
    "NoPosition",
    "QuoteBindingError",
    "QuoteRejected",
    "ReadOnlyViolation",
    "Reconciliation",
    "assert_live_supported",
    "pool_fee_micro",
]
