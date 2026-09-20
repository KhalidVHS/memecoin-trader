"""What the risk layer permits — never what it approves.

The old ``risk.check()`` answered "is this order OK, and if not, what smaller
order would be?" and returned an approved dollar amount. Audit **C3** showed
what that costs: ``loop._apply()`` quoted ``action.size_usd``, ``check()``
clamped it, and the broker then executed the *reduced* notional against the
*original-size* quote. AMM prices are size-dependent, so a $100 quote became a
$40 fill at $100's price. Two rules clamped rather than rejected, which is what
made the mismatch routine rather than exceptional.

So risk now answers a different question: **what is the maximum you may do?**
It returns :class:`~.types.RiskBounds` and nothing else. There is no field on
that type that could be mistaken for an order — no side-effect, no quantity, no
quote, no "approved" flag. The execution layer takes the bound, re-quotes at the
size it actually intends, and calls :meth:`RiskEngine.confirm_quote` to bind.
Risk never mutates an order, so it cannot produce one that was never priced.

**Four layers** (audit §15), because the old monolith mixed questions that
cannot be answered at the same altitude:

* :class:`EligibilityRisk` — *may this symbol be traded at all?* Universe
  membership, data quality, quote-token trust, pool age, risk-event quarantine.
  Per-symbol, order-independent.
* :class:`PortfolioRisk` — *how much of the book may this become?* Gross and net
  exposure, per-name concentration, the correlated-sleeve cap, the cash floor,
  vol targeting. **Cannot be answered per order**, which is precisely why the
  old code could not answer it at all.
* :class:`PreTradeRisk` — *is this specific order sane?* Size floor, cash, quote
  age, price impact, pool-depth participation.
* :class:`ContinuousRisk` — *should we be trading at all right now?* Loss
  budgets, drawdown breaker, consecutive-failure breaker, data health, kill
  switch. Evaluated **every tick**, not only before an order, because the
  conditions it detects arrive between orders.

Each is a small frozen object with pure methods. There is no I/O, no clock of
its own — ``now`` is always an argument — and no hidden state; the mutable facts
(peak value, failure streak, quarantines) live in :class:`RiskLedger`, which the
caller owns and :func:`update_ledger` advances.

**C4 — the entry/exit asymmetry, stated once, in full.** A
:class:`~.types.ValuationEstimate` is what ``quotes.py`` produces when Jupiter
could not route. The old code manufactured a mid-price quote, flagged it
``degraded``, and let the broker fill against it; ``check()`` had no veto on it
at all. Here, a valuation estimate **vetoes an entry outright** — a routing
outage is exactly when a mid-price fiction is least credible, and a trade that
could not have happened is worse than no trade. It **does not block an exit**.
That is not an inconsistency: an untrustworthy snapshot is a reason not to buy
and frequently a reason to sell, and the two decisions have opposite failure
costs. Being wrong about an entry costs you the trade; being wrong about an exit
costs you the position. The same predicate must not govern both, and in this
module it does not — see :meth:`RiskEngine.entry_bounds` against
:meth:`RiskEngine.exit_bounds`.

**C10 — three coins are one bet.** BONK, WIF and POPCAT are the same factor with
different tickers, and the old controls (30% per name, a -15% stop, 3% impact, a
liquidity floor) permit 90% gross in one trade that gaps together. Implemented
here: gross and net exposure caps, a per-position concentration cap, a
correlated-sleeve bucket cap, ATR-scaled sizing against a volatility target,
rolling day and window loss budgets, a drawdown breaker against peak book value,
a consecutive-execution-failure breaker, a data-health gate, a post-stop
quarantine, a quote-age gate, and a kill switch. Their honest limitations are
documented on each knob rather than in a footnote.

**Sizing, and what it is not sized on.** Audit C1: the system has no evidence
of alpha — one closed round trip — and sizing on an uncalibrated signal is
unjustified. Where a rule wants a forecast it consumes :class:`~.types.Forecast`
and treats ``calibration_id is None`` as *uncalibrated: do not size on this*,
falling back to ``max(RiskParams.default_entry_usd, cash_usd *
RiskParams.max_cash_fraction_pct / 100)`` — a size drawn from available cash,
floored at a small configured minimum. A confidence score that has never been
scored against outcomes is a number with units of nothing; how much cash is
free to deploy is a different, legitimate axis and is what this sizes on
instead (operator request, 2026-09-20).

**Missing is never zero.** A rule that cannot evaluate vetoes. If
``PortfolioState.total_value_usd`` is ``None`` — which C8 guarantees happens
whenever any held position is unmarkable — every percentage-of-book cap is
unevaluable, and the answer is no entries, not a cap computed against a guess.

**NaN discipline.** Every comparison against NaN is false, *including the
rejections*, so an unvalidated NaN walks straight through a risk check. Every
scalar entering this module goes through ``types.finite``; a non-finite input
produces a ``non_finite_input`` veto rather than an exception, because a
malformed number upstream must fail the trade, not the tick.

Deliberately still absent: any max-trades-per-day cap, and any minimum *hold*
time. Both were considered and removed before the audit and neither was
reinstated by it. The post-stop quarantine below is a different control — it
bounds re-entry after a *risk event*, not after an ordinary exit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from .types import (
    CoinSnapshot,
    DataQuality,
    Fill,
    Forecast,
    OrderState,
    PortfolioState,
    Quote,
    RiskBounds,
    RiskState,
    Side,
    ValidationError,
    ValuationEstimate,
    finite,
    finite_or_none,
)

#: Bounds are shaved by this factor where they are derived from an exact cash
#: balance. A broker that compares on a strict ``<`` can reject a size computed
#: as "exactly all the cash" once the fee is re-multiplied on its side. One part
#: in 10^12 of a dollar is invisible and removes the failure mode entirely.
_CASH_SAFETY = 1.0 - 1e-12

#: Tolerance when checking a re-quoted notional against its bound. Floating
#: point round-trips through atomic units and back; a bound of exactly the
#: notional must not fail on the last bit.
_BIND_TOLERANCE = 1.0 + 1e-9

_ONE_DAY_SECONDS = 86_400.0


def _usd(amount: float) -> str:
    return f"${amount:,.2f}"


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RiskParams:
    """Every knob, with its units asserted rather than remembered.

    The percent/fraction boundary in this codebase has caused real bugs, so it
    is machine-checked here. **Exactly two fields are fractions**, both
    inherited from the existing config and both multipliers:
    ``max_position_pct`` (0.30 = 30% of book) and ``stop_loss_pct`` (0.15 =
    -15%). **Every other ``_pct`` field is a whole percent** (3.0 = 3%), and
    ``__post_init__`` rejects a fraction accidentally supplied to one of them by
    refusing values in (0, 1) where a whole percent is meant. That check is
    crude — it cannot catch 0.5% meant literally — so the constructor also
    refuses a whole-percent field ``>= 100`` and a fraction field ``> 1``. Two
    cheap asymmetric checks catch the 100x error, which is the one that has
    actually happened.

    Defined here rather than in ``config.py`` so that risk takes its settings as
    an argument and remains a pure function of them. The coordinator wires the
    TOML keys; the defaults below are the values this module was designed and
    tested against.
    """

    # -- Eligibility -------------------------------------------------------
    #: The point-in-time tradeable universe. **Empty means nothing is
    #: eligible**, not "no restriction" — audit C9 wants an explicit eligibility
    #: record, and a default-open universe is how an unscreened token gets
    #: traded because someone forgot to populate a list.
    universe: frozenset[str] = frozenset()
    #: Staleness ceiling on the per-coin snapshot, measured from
    #: ``Provenance.effective_time`` (the oldest timestamp we have), not from
    #: the batch completion time. That distinction is the audit's leakage
    #: finding: ``MarketSnapshot.ts`` let a three-minute-old price look ninety
    #: seconds fresh.
    max_snapshot_age_seconds: float = 90.0
    min_liquidity_usd: float = 50_000.0
    #: Pools younger than this are refused. A thin, days-old pool is where the
    #: rug-pull base rate lives (§11), and it is also where a 5m price move is
    #: one swap rather than a trend.
    min_pool_age_seconds: float = 86_400.0
    #: When a pool's creation time is unknown, treat it as young and refuse.
    #: Missing is never zero and it is certainly not "old enough".
    require_known_pool_age: bool = True
    #: Cooling-off after a risk event (a stop, a failed exit, an unmarkable
    #: mark). Audit §11: the slow tick could stop out and immediately re-enter
    #: the same asset through the model, which converts a stop into a round-trip
    #: fee generator. One hour is a judgement call, not a measurement.
    post_stop_quarantine_seconds: float = 3_600.0
    #: Minimum spacing between entries in the same symbol regardless of stops.
    #: Set to one slow tick so a single decision cannot be executed twice.
    min_seconds_between_entries: float = 900.0

    # -- Portfolio ---------------------------------------------------------
    #: FRACTION of book value, one name. 0.50 = 50%, raised from 0.30 at the
    #: operator's explicit request (2026-09-20) to permit up to half the book
    #: in a single name; paired with `max_cash_fraction_pct` below so this cap
    #: does not silently clip a cash-based target back down.
    max_position_pct: float = 0.50
    #: FRACTION from entry at which the stop fires. 0.15 = -15%. Lives here so
    #: ``portfolio.stop_price_usd`` and the risk layer read one number.
    stop_loss_pct: float = 0.15
    #: Whole percent of NAV. Gross = sum of |position values|. With a long-only
    #: book gross and net coincide; both caps exist anyway so that adding a
    #: short does not silently escape one of them.
    #: Raised from 60.0 to 90.0 (2026-09-20) so that up to 3 positions each
    #: sized against `max_cash_fraction_pct` of remaining cash cannot be
    #: silently overridden by a tighter exposure ceiling — 90% deployed pairs
    #: with the 10% `min_cash_floor_pct` below rather than binding first.
    max_gross_exposure_pct: float = 90.0
    max_net_exposure_pct: float = 90.0
    #: Whole percent of NAV for the whole correlated sleeve.
    #:
    #: **This is the correlation control, and it is deliberately crude.**
    #: Estimating a pairwise correlation matrix over the sample this system has
    #: — a handful of days of 5m bars on three names — produces an estimate
    #: whose sampling error exceeds the quantity being estimated, and which
    #: collapses to ~1 in exactly the regime where it would matter. So no
    #: correlation is estimated. Instead every configured memecoin is assumed to
    #: be the *same bet* with a different ticker and shares one bucket cap. That
    #: is an assumption, not a measurement, and it errs in the safe direction:
    #: if the names turn out to be less correlated than assumed, the cost is
    #: foregone diversification benefit, whereas the reverse error is the 90%-
    #: gross-in-one-factor position audit C10 describes — which is worth
    #: repeating now that this cap is deliberately raised to 90.0 (2026-09-20,
    #: matching `max_gross_exposure_pct`/`max_net_exposure_pct` above): with
    #: all three configured coins in the sleeve, this and the gross/net caps
    #: are the same ceiling in practice, and 90% concentrated in one factor is
    #: now an explicit, accepted tradeoff for larger per-coin sizing, not an
    #: oversight.
    max_sleeve_exposure_pct: float = 90.0
    #: Symbols in that one bucket. Empty means the sleeve cap is not applied,
    #: which is only correct if you genuinely have no correlated names.
    correlated_sleeve: frozenset[str] = frozenset()
    #: Whole percent of NAV that must remain in cash. A book with no cash cannot
    #: pay gas to exit, which turns an execution problem into a solvency one.
    min_cash_floor_pct: float = 10.0
    #: Volatility target, whole percent, expressed on the same horizon as the
    #: ATR estimate fed to :meth:`PortfolioRisk.caps`.
    #:
    #: **Limitation, stated plainly:** ATR is a realised-range estimator and it
    #: is the wrong tool for a jump process. It is backward-looking, it
    #: understates risk in the quiet period immediately before a rug or a
    #: liquidity withdrawal, and scaling size by it produces its *largest*
    #: positions exactly when realised vol is lowest — which for these assets is
    #: often the calm before a discontinuity, not evidence of safety. It is
    #: implemented because vol-unaware sizing is worse, not because it is
    #: adequate. The gross/sleeve caps and the kill switch are what actually
    #: bound the tail; this only shapes size within them.
    target_volatility_pct: float = 8.0
    #: Refuse an entry when no volatility estimate is available. A rule that
    #: cannot evaluate vetoes.
    require_volatility_estimate: bool = True
    #: Floor on the size used when the forecast is uncalibrated, which today is
    #: always (audit C1: one closed round trip is not evidence of alpha). The
    #: actual fallback is ``max(default_entry_usd, cash_usd *
    #: max_cash_fraction_pct / 100)`` — this only matters when the cash-based
    #: figure would be smaller than a viable minimum entry.
    default_entry_usd: float = 25.0
    #: WHOLE PERCENT of currently uncommitted cash usable as the fallback entry
    #: size for an uncalibrated forecast (today, every forecast). Added
    #: 2026-09-20 at the operator's explicit request to size against available
    #: capital rather than a flat dollar figure regardless of book size — see
    #: `strategy.StrategySettings.cash_fraction_per_entry`, which this mirrors
    #: as the risk layer's own independent ceiling.
    max_cash_fraction_pct: float = 50.0
    #: Hard ceiling on any single entry, independent of book size. Raised from
    #: 100.0 (2026-09-20) so it does not clip an entry sized at up to 50% of a
    #: roughly $1,000 book; still a sanity ceiling, not the binding rule in
    #: normal operation.
    max_entry_usd: float = 600.0

    # -- Pre-trade ---------------------------------------------------------
    min_trade_usd: float = 10.0
    #: Whole percent. One-way. Note that a 3% allowance implies >6% round trip
    #: before any other cost, which audit C10 flags as already unreasonable for
    #: a strategy with no measured edge.
    max_price_impact_pct: float = 3.0
    #: A quote older than this may not be bound. Jupiter's own guidance is to
    #: sign and submit promptly; a quote is a price for an instant, and the
    #: audit's "quote changes after risk approval" row has **cancel** as its
    #: safe fallback.
    max_quote_age_seconds: float = 10.0
    #: Whole percent of the pool's reported liquidity that one order may be.
    #: A cap on impact is not a cap on depth participation: impact is quoted for
    #: the current pool state and says nothing about what happens to the *exit*.
    max_depth_participation_pct: float = 1.0
    gas_usd_per_swap: float = 0.21
    #: Whole percent, added to the cash reservation. On a real Jupiter route the
    #: pool fee is already inside the quoted output, so reserving it again would
    #: shrink every order by a fee that is not charged twice — hence 0.0. The
    #: knob exists for a venue where that is not true.
    assumed_pool_fee_pct: float = 0.0

    # -- Continuous --------------------------------------------------------
    #: Whole percent of the day's opening NAV. Trading loss budget only; there
    #: is deliberately no cap here on LLM API spend.
    max_daily_loss_pct: float = 5.0
    max_window_loss_pct: float = 10.0
    #: The rolling window the second budget is measured over. One week.
    loss_window_seconds: float = 604_800.0
    #: Whole percent from peak book value. Peak-to-trough, not day-to-day, so a
    #: slow bleed cannot stay under a daily budget forever.
    max_drawdown_pct: float = 20.0
    #: Consecutive failed execution attempts before halting. A failed Solana
    #: swap still pays full gas, so a failure streak is both a symptom (route or
    #: RPC trouble) and a cost in its own right.
    max_consecutive_failures: int = 3

    def __post_init__(self) -> None:
        fractions = ("max_position_pct", "stop_loss_pct")
        for name in fractions:
            value = finite(getattr(self, name), name)
            if not 0.0 < value <= 1.0:
                raise ValidationError(
                    f"{name} is a FRACTION (0.30 = 30%) and must be in (0, 1], got {value}"
                )
        whole_percents = (
            "max_gross_exposure_pct",
            "max_net_exposure_pct",
            "max_sleeve_exposure_pct",
            "min_cash_floor_pct",
            "max_cash_fraction_pct",
            "target_volatility_pct",
            "max_price_impact_pct",
            "max_depth_participation_pct",
            "assumed_pool_fee_pct",
            "max_daily_loss_pct",
            "max_window_loss_pct",
            "max_drawdown_pct",
        )
        for name in whole_percents:
            value = finite(getattr(self, name), name)
            if value < 0.0 or value >= 100.0:
                raise ValidationError(
                    f"{name} is a WHOLE PERCENT (3.0 = 3%) and must be in [0, 100), got {value}"
                )
        for name in (
            "max_snapshot_age_seconds",
            "min_liquidity_usd",
            "min_pool_age_seconds",
            "post_stop_quarantine_seconds",
            "min_seconds_between_entries",
            "default_entry_usd",
            "max_entry_usd",
            "min_trade_usd",
            "max_quote_age_seconds",
            "gas_usd_per_swap",
            "loss_window_seconds",
        ):
            value = finite(getattr(self, name), name)
            if value < 0.0:
                raise ValidationError(f"{name} must be >= 0, got {value}")
        if self.max_consecutive_failures < 1:
            raise ValidationError("max_consecutive_failures must be >= 1")


DEFAULT_PARAMS = RiskParams()


# ---------------------------------------------------------------------------
# Mutable continuous state, owned by the caller
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RiskLedger:
    """The facts a continuous control needs that a single tick cannot see.

    Kept out of the risk classes on purpose. Risk must be a pure function of
    state, and a breaker that carries its own counter is a breaker whose
    behaviour depends on how many times you happened to call it. The caller owns
    this, persists it, and advances it through :func:`update_ledger`.

    Every value is ``None`` until observed. ``peak_value_usd = None`` means "we
    have never seen a book value", which is not the same as "peak is zero" — the
    latter would make every drawdown 0% and disable the breaker silently.
    """

    peak_value_usd: float | None = None
    day_start_value_usd: float | None = None
    day_start_ts: float | None = None
    window_start_value_usd: float | None = None
    window_start_ts: float | None = None
    consecutive_failures: int = 0
    #: symbol -> epoch seconds at which its risk-event quarantine expires.
    quarantined_until: Mapping[str, float] = field(default_factory=dict)
    #: symbol -> epoch seconds of its last *landed* entry.
    last_entry_ts: Mapping[str, float] = field(default_factory=dict)
    #: The kill switch an operator throws by hand. Audit §11 is explicit that
    #: this is the only control that reliably works when something
    #: unanticipated is happening, so it is a plain boolean with no override
    #: path and no expiry.
    manual_halt: bool = False
    manual_halt_reason: str | None = None


EMPTY_LEDGER = RiskLedger()


def update_ledger(
    ledger: RiskLedger,
    *,
    book: PortfolioState,
    fills: Sequence[Fill] = (),
    risk_events: Sequence[str] = (),
    params: RiskParams = DEFAULT_PARAMS,
    now: float,
) -> RiskLedger:
    """Advance the continuous state by one tick. Pure; returns a new ledger.

    ``fills`` are this tick's attempts **in order**. The failure streak
    increments on every ``Fill.failed`` and resets on a landed one — audit §11's
    "failed stop reported as exit" row is the same defect seen from the other
    side, and both come from treating a ``Fill`` row's existence as proof that
    something happened. Here only ``LANDED``/``RECONCILED`` counts as something
    happening.

    ``risk_events`` are symbols that just had a stop, a failed exit or an
    unmarkable mark, and they start the re-entry quarantine. The audit found a
    stop-then-immediate-rebuy was possible in one slow tick.

    The peak is only advanced from a **fully marked** book. Advancing it from a
    partially marked one would let a book that is missing its worst position set
    a peak it never reached, and the drawdown breaker would then be measured
    against a number that never existed.
    """
    finite(now, "now")
    nav = finite_or_none(book.total_value_usd, "total_value_usd")

    peak = ledger.peak_value_usd
    if nav is not None:
        peak = nav if peak is None else max(peak, nav)

    day_start_value = ledger.day_start_value_usd
    day_start_ts = ledger.day_start_ts
    if day_start_ts is None or now - day_start_ts >= _ONE_DAY_SECONDS:
        day_start_value, day_start_ts = nav, now
    elif day_start_value is None:
        day_start_value = nav

    window_start_value = ledger.window_start_value_usd
    window_start_ts = ledger.window_start_ts
    if window_start_ts is None or now - window_start_ts >= params.loss_window_seconds:
        window_start_value, window_start_ts = nav, now
    elif window_start_value is None:
        window_start_value = nav

    failures = ledger.consecutive_failures
    last_entry = dict(ledger.last_entry_ts)
    for fill in fills:
        if fill.failed:
            failures += 1
            continue
        if fill.state in (OrderState.LANDED, OrderState.RECONCILED):
            failures = 0
            if fill.side is Side.BUY:
                last_entry[fill.symbol] = fill.ts

    quarantined = dict(ledger.quarantined_until)
    for symbol in (*risk_events, *book.unmarkable):
        quarantined[symbol] = now + params.post_stop_quarantine_seconds
    quarantined = {s: t for s, t in quarantined.items() if t > now}

    return RiskLedger(
        peak_value_usd=peak,
        day_start_value_usd=day_start_value,
        day_start_ts=day_start_ts,
        window_start_value_usd=window_start_value,
        window_start_ts=window_start_ts,
        consecutive_failures=failures,
        quarantined_until=quarantined,
        last_entry_ts=last_entry,
        manual_halt=ledger.manual_halt,
        manual_halt_reason=ledger.manual_halt_reason,
    )


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Cap:
    """One rule's ceiling, in dollars, with the sentence that explains it.

    A cap is not an approval. It is one term in a ``min()``, and every cap that
    fires appears in ``RiskBounds.reasons`` whether or not it was the binding
    one — a trader who can see that the cash cap was $99.79 and the
    concentration cap was $200 knows what to change; "max_position_pct
    violated" tells them nothing.
    """

    rule: str
    max_notional_usd: float
    reason: str


def _cap(rule: str, amount: float, reason: str) -> Cap:
    """Clamp at zero. A negative headroom is a refusal, not a negative order."""
    return Cap(rule=rule, max_notional_usd=max(0.0, amount), reason=reason)


def _refuse(
    symbol: str,
    side: Side,
    vetoes: Sequence[str],
    reasons: Sequence[str],
    *,
    notes: Sequence[str] = (),
) -> RiskBounds:
    return RiskBounds(
        symbol=symbol,
        side=side,
        max_notional_usd=0.0,
        vetoes=tuple(vetoes),
        reasons=tuple(reasons),
        binding_rule=vetoes[0] if vetoes else None,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Layer 1 — Eligibility
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EligibilityRisk:
    """May this symbol be traded at all, ignoring size and book entirely?

    Audit C9's foothold. The old system had exactly three config entries as a
    permanent universe with no point-in-time eligibility record, no token safety
    screen, and no way to remove a name that had gone bad. This does not
    implement the on-chain screen the audit asks for — authorities, Token-2022
    extensions, LP ownership, holder concentration — and does not pretend to.
    What it does implement is the *shape*: an explicit universe, a data-quality
    gate, a quote-token trust gate, a pool-age gate and a quarantine, all
    evaluated before anything about the order is considered. The on-chain screen
    plugs in here when it exists, and is listed as deferred in the report.

    Returns vetoes. An empty tuple means eligible, and nothing else.
    """

    params: RiskParams = DEFAULT_PARAMS

    def assess(
        self,
        symbol: str,
        *,
        snapshot: CoinSnapshot | None,
        ledger: RiskLedger = EMPTY_LEDGER,
        now: float,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Returns ``(vetoes, reasons)``."""
        p = self.params
        vetoes: list[str] = []
        reasons: list[str] = []

        def veto(rule: str, why: str) -> None:
            vetoes.append(rule)
            reasons.append(why)

        if symbol not in p.universe:
            veto(
                "not_in_universe",
                f"{symbol} is not in the eligible universe "
                f"({', '.join(sorted(p.universe)) or 'empty'}) — an unscreened token is "
                f"not tradeable by default",
            )

        until = ledger.quarantined_until.get(symbol)
        if until is not None and until > now:
            veto(
                "quarantined",
                f"{symbol} is under a risk-event quarantine for another "
                f"{until - now:.0f}s — a stop or an unmarkable position is not an "
                f"invitation to re-enter",
            )

        last = ledger.last_entry_ts.get(symbol)
        if last is not None and now - last < p.min_seconds_between_entries:
            veto(
                "entry_cooldown",
                f"{symbol} was entered {now - last:.0f}s ago and the minimum spacing "
                f"is {p.min_seconds_between_entries:.0f}s",
            )

        if snapshot is None:
            veto(
                "missing_snapshot",
                f"no market snapshot for {symbol} this tick, so nothing about it can "
                f"be verified",
            )
            return tuple(vetoes), tuple(reasons)

        if snapshot.quality is not DataQuality.OK:
            veto(
                "degraded_snapshot",
                f"{symbol} snapshot quality is {snapshot.quality.value}"
                + (f" ({snapshot.quality_reason})" if snapshot.quality_reason else ""),
            )
        if not snapshot.pool.trusted_quote:
            veto(
                "untrusted_quote_token",
                f"{symbol}'s best pool is priced in {snapshot.pool.quote_symbol}, whose "
                f"own USD price is unknown — that pool is diagnostic data, never an "
                f"entry price (C8)",
            )
        if snapshot.price_usd is None:
            veto("no_price", f"{symbol} reported no price this tick")
        if snapshot.liquidity_usd is None:
            veto(
                "no_liquidity_observation",
                f"{symbol} reported no liquidity — missing is not zero and it is not "
                f"deep either, so it cannot clear the floor",
            )
        elif snapshot.liquidity_usd < p.min_liquidity_usd:
            veto(
                "min_liquidity",
                f"{symbol} pool liquidity is {_usd(snapshot.liquidity_usd)}, below the "
                f"{_usd(p.min_liquidity_usd)} floor — at this depth a 5m price move is "
                f"one swap, not a trend",
            )

        age = snapshot.age_seconds(now)
        if age > p.max_snapshot_age_seconds:
            veto(
                "stale_data",
                f"{symbol} market data is {age:.0f}s old against a "
                f"{p.max_snapshot_age_seconds:.0f}s limit — refusing to trade blind",
            )

        created = snapshot.pool.created_at
        if created is None:
            if p.require_known_pool_age:
                veto(
                    "unknown_pool_age",
                    f"{symbol}'s pool creation time is unknown, so it cannot be shown "
                    f"to be older than {p.min_pool_age_seconds / 3600.0:.0f}h",
                )
        elif now - created < p.min_pool_age_seconds:
            veto(
                "new_pool",
                f"{symbol}'s pool is {(now - created) / 3600.0:.1f}h old, under the "
                f"{p.min_pool_age_seconds / 3600.0:.0f}h minimum",
            )

        return tuple(vetoes), tuple(reasons)


# ---------------------------------------------------------------------------
# Layer 2 — Portfolio
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PortfolioRisk:
    """How much of the book may this position become?

    Every rule here is cross-position and therefore unanswerable by a per-order
    check — which is exactly why the old monolith, which only ever saw one
    proposal at a time, had none of them. Audit C10.

    All caps are evaluated against ``PortfolioState.total_value_usd``. When that
    is ``None`` — C8 guarantees it is whenever a held position is unmarkable —
    every one of them is unevaluable and the layer vetoes. That is the whole
    point of the ``None`` propagating: the alternative is a 30%-of-book cap
    computed against a book value partly made of cost basis.
    """

    params: RiskParams = DEFAULT_PARAMS

    def caps(
        self,
        symbol: str,
        *,
        book: PortfolioState,
        forecast: Forecast | None = None,
        volatility_pct: float | None = None,
    ) -> tuple[tuple[Cap, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Returns ``(caps, vetoes, reasons, notes)``."""
        p = self.params
        caps: list[Cap] = []
        vetoes: list[str] = []
        reasons: list[str] = []
        notes: list[str] = []

        nav = finite_or_none(book.total_value_usd, "total_value_usd")
        if nav is None:
            vetoes.append("book_unmarkable")
            reasons.append(
                "book value is unknown because "
                + (", ".join(book.unmarkable) or "a position")
                + " could not be marked — every percentage-of-book cap is unevaluable, "
                "so no entry is permitted (C8)"
            )
            return (), tuple(vetoes), tuple(reasons), tuple(notes)
        if nav <= 0.0:
            vetoes.append("no_book_value")
            reasons.append(f"book value is {_usd(nav)} — there is nothing to risk")
            return (), tuple(vetoes), tuple(reasons), tuple(notes)

        held = book.position_values_usd.get(symbol) or 0.0
        gross = book.gross_exposure_usd
        if gross is None:  # pragma: no cover - implied by nav is None
            vetoes.append("book_unmarkable")
            reasons.append("gross exposure is unknown")
            return (), tuple(vetoes), tuple(reasons), tuple(notes)

        # -- concentration -------------------------------------------------
        limit = nav * p.max_position_pct
        caps.append(
            _cap(
                "max_position_pct",
                limit - held,
                f"{symbol} may reach {_usd(limit)} ({100.0 * p.max_position_pct:.0f}% of a "
                f"{_usd(nav)} book) and already holds {_usd(held)}",
            )
        )

        # -- gross / net ---------------------------------------------------
        gross_limit = nav * p.max_gross_exposure_pct / 100.0
        caps.append(
            _cap(
                "max_gross_exposure_pct",
                gross_limit - gross,
                f"gross exposure is {_usd(gross)} against a {_usd(gross_limit)} cap "
                f"({p.max_gross_exposure_pct:.0f}% of book)",
            )
        )
        # Long-only today, so net equals gross. The cap is evaluated separately
        # anyway: if a short is ever added, a net limit that was never wired up
        # is worse than one that was redundant.
        net_limit = nav * p.max_net_exposure_pct / 100.0
        caps.append(
            _cap(
                "max_net_exposure_pct",
                net_limit - gross,
                f"net exposure is {_usd(gross)} against a {_usd(net_limit)} cap "
                f"({p.max_net_exposure_pct:.0f}% of book)",
            )
        )

        # -- correlated sleeve ---------------------------------------------
        if p.correlated_sleeve:
            sleeve_held = sum(
                value
                for name, value in book.position_values_usd.items()
                if name in p.correlated_sleeve and value is not None
            )
            sleeve_limit = nav * p.max_sleeve_exposure_pct / 100.0
            if symbol in p.correlated_sleeve:
                caps.append(
                    _cap(
                        "max_sleeve_exposure_pct",
                        sleeve_limit - sleeve_held,
                        f"the correlated memecoin sleeve holds {_usd(sleeve_held)} against a "
                        f"{_usd(sleeve_limit)} cap — these names are treated as one bet, "
                        f"because they are",
                    )
                )

        # -- cash floor ----------------------------------------------------
        fee_rate = p.assumed_pool_fee_pct / 100.0
        floor = nav * p.min_cash_floor_pct / 100.0
        spendable = (book.cash_usd - floor - p.gas_usd_per_swap) / (1.0 + fee_rate)
        caps.append(
            _cap(
                "min_cash_floor_pct",
                spendable * _CASH_SAFETY,
                f"cash is {_usd(book.cash_usd)} and {_usd(floor)} "
                f"({p.min_cash_floor_pct:.0f}% of book) must stay behind to pay gas on the "
                f"way out",
            )
        )

        # -- volatility target ---------------------------------------------
        vol = finite_or_none(volatility_pct, "volatility_pct")
        if vol is None or vol <= 0.0:
            if p.require_volatility_estimate:
                vetoes.append("no_volatility_estimate")
                reasons.append(
                    f"no usable volatility estimate for {symbol} (got {volatility_pct!r}) — "
                    f"a sizing rule that cannot evaluate vetoes rather than assuming calm"
                )
            else:
                notes.append(
                    f"no volatility estimate for {symbol}; vol targeting not applied"
                )
        else:
            scalar = min(1.0, p.target_volatility_pct / vol)
            caps.append(
                _cap(
                    "volatility_target",
                    limit * scalar - held,
                    f"{symbol} realised vol is {vol:.1f}% against a "
                    f"{p.target_volatility_pct:.1f}% target, so the concentration cap is "
                    f"scaled to {scalar:.2f}x (ATR-based: backward-looking and blind to "
                    f"jumps — see RiskParams.target_volatility_pct)",
                )
            )

        # -- forecast-conditioned size -------------------------------------
        if forecast is None or forecast.calibration_id is None:
            why = "no forecast" if forecast is None else f"model {forecast.model_id}"
            cash_based = book.cash_usd * p.max_cash_fraction_pct / 100.0
            fallback_size = max(p.default_entry_usd, cash_based)
            caps.append(
                _cap(
                    "uncalibrated_forecast",
                    fallback_size,
                    f"{why} carries no calibration_id, so its magnitude is not a number "
                    f"anything may be sized on (C1) — falling back to "
                    f"{p.max_cash_fraction_pct:.0f}% of available cash "
                    f"({_usd(cash_based)}), floored at {_usd(p.default_entry_usd)}",
                )
            )
        elif not forecast.actionable:
            vetoes.append("unusable_forecast")
            reasons.append(
                f"forecast {forecast.calibration_id} has no expected return or no lower "
                f"quantile, so there is nothing to test against the cost hurdle"
            )
        else:
            notes.append(
                f"sizing on calibrated forecast {forecast.calibration_id} "
                f"(lower quantile {forecast.lower_quantile_pct:.2f}%)"
            )

        caps.append(
            _cap(
                "max_entry_usd",
                p.max_entry_usd,
                f"single-entry ceiling of {_usd(p.max_entry_usd)}",
            )
        )

        return tuple(caps), tuple(vetoes), tuple(reasons), tuple(notes)


# ---------------------------------------------------------------------------
# Layer 3 — Pre-trade
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PreTradeRisk:
    """Is this specific order sane, given a price we could actually get?

    The narrowest layer and the only one that looks at a :class:`Quote`. It is
    also where audit C4 is enforced for entries: a
    :class:`~.types.ValuationEstimate` is not a route and may not support a buy.
    """

    params: RiskParams = DEFAULT_PARAMS

    def caps(
        self,
        symbol: str,
        side: Side,
        *,
        book: PortfolioState,
        snapshot: CoinSnapshot | None = None,
        quote: Quote | None = None,
        valuation: ValuationEstimate | None = None,
        now: float,
    ) -> tuple[tuple[Cap, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Returns ``(caps, vetoes, reasons, notes)``."""
        p = self.params
        caps: list[Cap] = []
        vetoes: list[str] = []
        reasons: list[str] = []
        notes: list[str] = []

        if side is Side.BUY and valuation is not None:
            vetoes.append("degraded_valuation")
            reasons.append(
                f"{symbol} has only a non-executable valuation estimate "
                f"({valuation.reason}) and no route — a routing outage is when a "
                f"mid-price fiction is least credible, so it may not support an entry "
                f"(C4). It does not block an exit."
            )

        fee_rate = p.assumed_pool_fee_pct / 100.0
        if side is Side.BUY:
            affordable = (book.cash_usd - p.gas_usd_per_swap) / (1.0 + fee_rate)
            caps.append(
                _cap(
                    "insufficient_cash",
                    affordable * _CASH_SAFETY,
                    f"cash is {_usd(book.cash_usd)} and {_usd(p.gas_usd_per_swap)} of gas "
                    f"must be reserved",
                )
            )

        if snapshot is not None and snapshot.liquidity_usd is not None:
            depth = snapshot.liquidity_usd * p.max_depth_participation_pct / 100.0
            caps.append(
                _cap(
                    "max_depth_participation_pct",
                    depth,
                    f"one order may be {p.max_depth_participation_pct:.2f}% of "
                    f"{symbol}'s {_usd(snapshot.liquidity_usd)} pool, i.e. {_usd(depth)}",
                )
            )

        if quote is not None:
            vetoes_q, reasons_q = self.check_quote(symbol, side, quote=quote, now=now)
            vetoes.extend(vetoes_q)
            reasons.extend(reasons_q)

        return tuple(caps), tuple(vetoes), tuple(reasons), tuple(notes)

    def check_quote(
        self, symbol: str, side: Side, *, quote: Quote, now: float
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Quote-age, expiry, identity and impact. Shared by sizing and binding.

        Kept as one method so that the checks run at bind time are *literally*
        the checks run at sizing time. The audit's "quote changes after risk
        approval" row exists because the two were different code.
        """
        p = self.params
        vetoes: list[str] = []
        reasons: list[str] = []

        if quote.symbol != symbol or quote.side is not side:
            vetoes.append("quote_mismatch")
            reasons.append(
                f"quote is {quote.side} {quote.symbol} but the bound is for {side} {symbol}"
            )
        if quote.is_expired(now):
            vetoes.append("quote_expired")
            reasons.append(
                f"{symbol} quote expired {now - (quote.expires_at or now):.1f}s ago"
            )
        age = quote.age_seconds(now)
        if age > p.max_quote_age_seconds:
            vetoes.append("quote_age")
            reasons.append(
                f"{symbol} quote is {age:.1f}s old against a "
                f"{p.max_quote_age_seconds:.1f}s limit — a quote is a price for an instant"
            )
        if quote.price_impact_pct > p.max_price_impact_pct:
            vetoes.append("max_price_impact")
            reasons.append(
                f"{symbol} price impact {quote.price_impact_pct:.2f}% exceeds the "
                f"{p.max_price_impact_pct:.2f}% limit — the pool is too thin for this size"
            )
        return tuple(vetoes), tuple(reasons)


# ---------------------------------------------------------------------------
# Layer 4 — Continuous
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContinuousRisk:
    """Should we be trading at all right now?

    Evaluated every tick, including ticks with no order in them, because the
    conditions it detects — a drawdown, a failure streak, a data outage — arrive
    between orders and the old system could only notice them at the moment it
    was about to trade anyway.

    The kill switch is the last item and the most important one. Audit §11 is
    explicit that it is the only control that reliably works when something
    unanticipated is happening, which is why it is a plain boolean with no
    conditions on it, and why ``halted`` permits exits: halting a system that
    holds inventory it cannot sell is not safety, it is a trap.
    """

    params: RiskParams = DEFAULT_PARAMS

    def evaluate(
        self,
        *,
        book: PortfolioState,
        ledger: RiskLedger = EMPTY_LEDGER,
        data_quality_ok: bool = True,
        now: float,
    ) -> RiskState:
        p = self.params
        finite(now, "now")
        nav = finite_or_none(book.total_value_usd, "total_value_usd")

        halt_reasons: list[str] = []

        if ledger.manual_halt:
            halt_reasons.append(
                "kill switch engaged"
                + (f": {ledger.manual_halt_reason}" if ledger.manual_halt_reason else "")
            )

        peak = ledger.peak_value_usd
        if peak is not None and nav is not None:
            peak = max(peak, nav)
        drawdown_pct: float | None = None
        if peak is not None and peak > 0.0 and nav is not None:
            drawdown_pct = 100.0 * (peak - nav) / peak
            if drawdown_pct > p.max_drawdown_pct:
                halt_reasons.append(
                    f"drawdown {drawdown_pct:.1f}% from a {_usd(peak)} peak exceeds the "
                    f"{p.max_drawdown_pct:.1f}% breaker"
                )

        day_loss_pct = _change_pct(ledger.day_start_value_usd, nav)
        if day_loss_pct is not None and day_loss_pct <= -p.max_daily_loss_pct:
            halt_reasons.append(
                f"today's book change is {day_loss_pct:.1f}%, past the "
                f"{p.max_daily_loss_pct:.1f}% daily loss budget"
            )
        window_loss_pct = _change_pct(ledger.window_start_value_usd, nav)
        if window_loss_pct is not None and window_loss_pct <= -p.max_window_loss_pct:
            halt_reasons.append(
                f"the {p.loss_window_seconds / 86400.0:.0f}-day book change is "
                f"{window_loss_pct:.1f}%, past the {p.max_window_loss_pct:.1f}% budget"
            )

        if ledger.consecutive_failures >= p.max_consecutive_failures:
            halt_reasons.append(
                f"{ledger.consecutive_failures} consecutive execution failures, limit is "
                f"{p.max_consecutive_failures} — each one pays full gas and moves nothing"
            )

        # An unmarkable position is a risk incident, not a number (C8). It does
        # not engage the kill switch — that would be disproportionate to one
        # missing price — but it does fail data health, which blocks every entry
        # while leaving every exit open. That is the audit's prescribed fallback
        # for this exact row: "Block entries; escalate exit".
        healthy = bool(data_quality_ok) and book.fully_marked
        quarantined = frozenset(
            {s for s, until in ledger.quarantined_until.items() if until > now}
            | set(book.unmarkable)
        )

        return RiskState(
            ts=now,
            halted=bool(halt_reasons),
            halt_reasons=tuple(halt_reasons),
            gross_exposure_pct=book.gross_exposure_pct,
            rolling_loss_pct=day_loss_pct,
            peak_value_usd=peak,
            drawdown_pct=drawdown_pct,
            consecutive_failures=ledger.consecutive_failures,
            quarantined_symbols=quarantined,
            data_health_ok=healthy,
        )


def _change_pct(start: float | None, end: float | None) -> float | None:
    """Whole-percent change, ``None`` when either end is unobserved.

    ``None`` rather than 0.0, always. A loss budget that reads a missing
    observation as "flat" is a loss budget that is disabled exactly when the
    book cannot be valued.
    """
    if start is None or end is None or start <= 0.0:
        return None
    return 100.0 * (end - start) / start


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RiskEngine:
    """Composes the four layers into the two questions execution actually asks.

    Both return :class:`~.types.RiskBounds`. Neither returns, builds, mutates or
    accepts an order. ``max_notional_usd`` is the minimum over every cap that
    fired, which is the property the test suite asserts exhaustively: **the
    bound can never exceed any individual cap.**
    """

    params: RiskParams = DEFAULT_PARAMS

    @property
    def eligibility(self) -> EligibilityRisk:
        return EligibilityRisk(self.params)

    @property
    def portfolio(self) -> PortfolioRisk:
        return PortfolioRisk(self.params)

    @property
    def pre_trade(self) -> PreTradeRisk:
        return PreTradeRisk(self.params)

    @property
    def continuous(self) -> ContinuousRisk:
        return ContinuousRisk(self.params)

    # -- entries -----------------------------------------------------------

    def entry_caps(
        self,
        symbol: str,
        *,
        book: PortfolioState,
        risk_state: RiskState,
        snapshot: CoinSnapshot | None,
        quote: Quote | None = None,
        valuation: ValuationEstimate | None = None,
        forecast: Forecast | None = None,
        volatility_pct: float | None = None,
        ledger: RiskLedger = EMPTY_LEDGER,
        now: float,
    ) -> tuple[tuple[Cap, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Every cap and veto that applies to an entry, unreduced.

        Exposed so that the invariant "the bound never exceeds any cap" can be
        checked against the caps themselves rather than against a restatement of
        them in a test.
        """
        vetoes: list[str] = []
        reasons: list[str] = []
        notes: list[str] = []

        if risk_state.halted:
            vetoes.append("halted")
            reasons.append(
                "the kill switch is engaged ("
                + ("; ".join(risk_state.halt_reasons) or "no reason recorded")
                + ") — exits only"
            )
        if not risk_state.data_health_ok:
            vetoes.append("data_health")
            reasons.append(
                "data health has failed this tick, so no new entries. Exits are "
                "unaffected: bad data is a reason not to buy and often a reason to sell."
            )
        if symbol in risk_state.quarantined_symbols:
            vetoes.append("quarantined")
            reasons.append(f"{symbol} is under a risk-event quarantine")

        e_vetoes, e_reasons = self.eligibility.assess(
            symbol, snapshot=snapshot, ledger=ledger, now=now
        )
        vetoes.extend(e_vetoes)
        reasons.extend(e_reasons)

        p_caps, p_vetoes, p_reasons, p_notes = self.portfolio.caps(
            symbol, book=book, forecast=forecast, volatility_pct=volatility_pct
        )
        vetoes.extend(p_vetoes)
        reasons.extend(p_reasons)
        notes.extend(p_notes)

        t_caps, t_vetoes, t_reasons, t_notes = self.pre_trade.caps(
            symbol,
            Side.BUY,
            book=book,
            snapshot=snapshot,
            quote=quote,
            valuation=valuation,
            now=now,
        )
        vetoes.extend(t_vetoes)
        reasons.extend(t_reasons)
        notes.extend(t_notes)

        # Deduplicate while preserving order: several layers can legitimately
        # reach the same conclusion and a journal line that says "quarantined"
        # three times reads as three problems.
        return (
            (*p_caps, *t_caps),
            tuple(dict.fromkeys(vetoes)),
            tuple(dict.fromkeys(reasons)),
            tuple(dict.fromkeys(notes)),
        )

    def entry_bounds(
        self,
        symbol: str,
        *,
        book: PortfolioState,
        risk_state: RiskState,
        snapshot: CoinSnapshot | None,
        quote: Quote | None = None,
        valuation: ValuationEstimate | None = None,
        forecast: Forecast | None = None,
        volatility_pct: float | None = None,
        ledger: RiskLedger = EMPTY_LEDGER,
        now: float,
    ) -> RiskBounds:
        """The maximum USD notional that may be bought in ``symbol``.

        Not an order. The caller re-quotes at whatever size it intends up to
        this bound and calls :meth:`confirm_quote` before submitting — that
        round trip is audit C3's fix, and it is the reason nothing on
        ``RiskBounds`` resembles an approval.

        ``quote`` is optional and normally absent at this stage: you obtain a
        bound, then a quote for the size you chose. When one is supplied it is
        checked, which is how a caller can ask "would this exact quote still be
        acceptable?" without a second code path.
        """
        try:
            finite(now, "now")
        except ValidationError as exc:
            return _refuse(symbol, Side.BUY, ("non_finite_input",), (str(exc),))

        try:
            caps, vetoes, reasons, notes = self.entry_caps(
                symbol,
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
        except ValidationError as exc:
            # A NaN or an infinity reached a risk input. Every comparison
            # against NaN is false, including the rejections, so the one
            # unacceptable outcome is letting it through. Fail closed and name
            # it, rather than raising and taking the whole tick down.
            return _refuse(symbol, Side.BUY, ("non_finite_input",), (str(exc),))

        if vetoes:
            return _refuse(symbol, Side.BUY, vetoes, reasons, notes=notes)

        if not caps:  # pragma: no cover - a vetoless empty cap set cannot occur
            return _refuse(
                symbol,
                Side.BUY,
                ("no_caps_evaluated",),
                ("no sizing rule could be evaluated, so no size is permitted",),
                notes=notes,
            )

        binding = min(caps, key=lambda c: c.max_notional_usd)
        bound = binding.max_notional_usd

        if bound < self.params.min_trade_usd:
            return _refuse(
                symbol,
                Side.BUY,
                ("min_notional", binding.rule),
                (
                    f"the largest legal {symbol} entry is {_usd(bound)}, below the "
                    f"{_usd(self.params.min_trade_usd)} minimum — fees and gas would eat it "
                    f"(binding rule: {binding.rule}; {binding.reason})",
                    *reasons,
                ),
                notes=notes,
            )

        return RiskBounds(
            symbol=symbol,
            side=Side.BUY,
            max_notional_usd=bound,
            vetoes=(),
            reasons=(
                f"{binding.rule}: {binding.reason}",
                *(c.reason for c in caps if c is not binding),
            ),
            binding_rule=binding.rule,
            notes=tuple(notes),
        )

    # -- exits -------------------------------------------------------------

    def exit_bounds(
        self,
        symbol: str,
        *,
        book: PortfolioState,
        risk_state: RiskState,
        snapshot: CoinSnapshot | None = None,
        forced: bool = False,
        now: float,
    ) -> RiskBounds:
        """The maximum USD notional that may be sold in ``symbol``.

        The asymmetry with :meth:`entry_bounds`, in full and on purpose:

        * ``risk_state.halted`` does **not** block an exit. Once halted, exits
          are the only thing permitted.
        * A degraded snapshot or a :class:`ValuationEstimate` does **not** block
          an exit. C4's veto is an entry veto.
        * A quarantine does **not** block an exit. It exists to stop a rebuy.
        * ``min_liquidity`` and ``max_price_impact`` are **bypassed on a forced
          exit**, and this is the bypass worth stating out loud: both rules fire
          hardest when a pool is collapsing, which is the exact scenario the stop
          exists for. Enforcing them there does not protect the position, it
          traps it, and the cost of being trapped is unbounded while the cost of
          a bad fill is bounded by what is left. Every bypass is recorded in
          ``RiskBounds.bypassed_rules`` so a bad forced fill is visible after the
          fact rather than silent.
        * ``stale_data`` is **still enforced**, forced or not. This is the one
          rule a stop does not get to argue with, and the reason is measured
          rather than aesthetic: the fast tick retries in 60 seconds, so blocking
          here costs a minute, whereas exiting on a price we cannot vouch for
          costs the fill. It applies only when a snapshot was supplied at all —
          the old ``missing_snapshot`` rejection existed solely to guard the
          liquidity floor, which an exit bypasses, so there is nothing left for
          it to protect.

        The bound itself is the position's marked value. When the position is
        **unmarkable** the bound falls back to cost basis, noted loudly. That is
        a ceiling, not a size: the execution layer is bounded by inventory in
        atomic units, so a ceiling that is too high cannot cause an over-sell,
        while a ceiling that is too low would trap exactly the position that
        most needs to leave. This is the only place in the system where cost
        basis is still allowed near a price, and it is allowed only because it
        is an upper bound on permission rather than a valuation (C8).
        """
        try:
            finite(now, "now")
        except ValidationError as exc:
            return _refuse(symbol, Side.SELL, ("non_finite_input",), (str(exc),))

        p = self.params
        notes: list[str] = []
        bypassed: list[str] = []

        position = book.positions.get(symbol)
        if position is None or position.quantity_atomic == 0:
            return _refuse(
                symbol,
                Side.SELL,
                ("no_position",),
                (f"there is no open {symbol} position to sell",),
            )

        if risk_state.halted:
            notes.append(
                "system is halted ("
                + ("; ".join(risk_state.halt_reasons) or "no reason recorded")
                + ") — exit-only mode, this exit is permitted"
            )
        if symbol in risk_state.quarantined_symbols:
            notes.append(f"{symbol} is quarantined for re-entry; the exit is unaffected")

        if snapshot is not None:
            age = snapshot.age_seconds(now)
            if age > p.max_snapshot_age_seconds:
                return _refuse(
                    symbol,
                    Side.SELL,
                    ("stale_data",),
                    (
                        f"{symbol} market data is {age:.0f}s old against a "
                        f"{p.max_snapshot_age_seconds:.0f}s limit. Not bypassed for a forced "
                        f"exit: the fast tick retries in 60s, so blocking costs a minute, "
                        f"not the position.",
                    ),
                    notes=notes,
                )
            if (
                snapshot.liquidity_usd is not None
                and snapshot.liquidity_usd < p.min_liquidity_usd
            ):
                if forced:
                    bypassed.append("min_liquidity")
                    notes.append(
                        f"forced exit from a draining pool: liquidity "
                        f"{_usd(snapshot.liquidity_usd)} is below the "
                        f"{_usd(p.min_liquidity_usd)} floor"
                    )
                else:
                    return _refuse(
                        symbol,
                        Side.SELL,
                        ("min_liquidity",),
                        (
                            f"{symbol} pool liquidity is {_usd(snapshot.liquidity_usd)}, below "
                            f"the {_usd(p.min_liquidity_usd)} floor",
                        ),
                        notes=notes,
                    )
        else:
            notes.append(
                f"no {symbol} snapshot this tick; an exit does not need one, since the "
                f"snapshot only ever guarded the liquidity floor"
            )

        value = book.position_values_usd.get(symbol)
        if value is None:
            value = position.cost_basis_usd
            notes.append(
                f"{symbol} is UNMARKABLE: bounding the exit by its "
                f"{_usd(position.cost_basis_usd)} cost basis as a permission ceiling only. "
                f"This is not a valuation and must not be journalled as one."
            )
            bypassed.append("mark_unavailable")

        if forced:
            bypassed.append("max_price_impact")
            if value < p.min_trade_usd:
                bypassed.append("min_notional")
                notes.append(
                    f"forced exit of {_usd(value)}, under the {_usd(p.min_trade_usd)} "
                    f"minimum — a stop is not blocked by a rule about how big a trade "
                    f"should be"
                )
        elif value < p.min_trade_usd:
            return _refuse(
                symbol,
                Side.SELL,
                ("min_notional",),
                (
                    f"the whole {symbol} position is worth {_usd(value)}, below the "
                    f"{_usd(p.min_trade_usd)} minimum — fees and gas would eat it",
                ),
                notes=notes,
            )

        return RiskBounds(
            symbol=symbol,
            side=Side.SELL,
            max_notional_usd=value,
            vetoes=(),
            reasons=(f"position_value: the whole {symbol} position is {_usd(value)}",),
            binding_rule="position_value",
            notes=tuple(notes),
            bypassed_rules=tuple(dict.fromkeys(bypassed)),
        )

    # -- binding -----------------------------------------------------------

    def confirm_quote(
        self,
        bounds: RiskBounds,
        quote: Quote,
        *,
        notional_usd: float,
        now: float,
    ) -> RiskBounds:
        """Re-run the order-specific checks against the quote actually obtained.

        This is the second half of audit C3. ``entry_bounds`` says how much you
        *may* do; the execution layer then requests an exact-input quote at the
        size it chose, and this confirms that the quote it got is still
        acceptable *at that size*. On success the original ``bounds`` is
        returned **unchanged** apart from a note — deliberately, so that there
        is no moment at which this function hands back something shaped like an
        approved order. On failure it returns a refusal.

        The impact check is skipped when ``max_price_impact`` is in
        ``bounds.bypassed_rules``, i.e. on a forced exit, so that the bypass is
        decided once in :meth:`exit_bounds` and honoured here rather than
        re-litigated.
        """
        try:
            amount = finite(notional_usd, "notional_usd")
            finite(now, "now")
        except ValidationError as exc:
            return _refuse(bounds.symbol, bounds.side, ("non_finite_input",), (str(exc),))

        if not bounds.permitted:
            return _refuse(
                bounds.symbol,
                bounds.side,
                bounds.vetoes or ("not_permitted",),
                bounds.reasons or ("these bounds permit nothing",),
            )

        if amount <= 0.0:
            return _refuse(
                bounds.symbol,
                bounds.side,
                ("non_positive_notional",),
                (f"a {_usd(amount)} order is not an order",),
            )

        if amount > bounds.max_notional_usd * _BIND_TOLERANCE:
            return _refuse(
                bounds.symbol,
                bounds.side,
                ("exceeds_bound",),
                (
                    f"{_usd(amount)} exceeds the {_usd(bounds.max_notional_usd)} bound set by "
                    f"{bounds.binding_rule} — risk does not shrink orders, it refuses them "
                    f"(C3)",
                ),
            )

        vetoes, reasons = self.pre_trade.check_quote(
            bounds.symbol, bounds.side, quote=quote, now=now
        )
        if "max_price_impact" in bounds.bypassed_rules:
            keep = [
                (v, r)
                for v, r in zip(vetoes, reasons, strict=True)
                if v != "max_price_impact"
            ]
            vetoes = tuple(v for v, _ in keep)
            reasons = tuple(r for _, r in keep)

        if vetoes:
            return _refuse(bounds.symbol, bounds.side, vetoes, reasons, notes=bounds.notes)

        return replace(
            bounds,
            notes=(
                *bounds.notes,
                f"quote {quote.fingerprint} confirmed at {_usd(amount)} against a "
                f"{_usd(bounds.max_notional_usd)} bound",
            ),
        )


__all__ = [
    "DEFAULT_PARAMS",
    "EMPTY_LEDGER",
    "Cap",
    "ContinuousRisk",
    "EligibilityRisk",
    "PortfolioRisk",
    "PreTradeRisk",
    "RiskEngine",
    "RiskLedger",
    "RiskParams",
    "update_ledger",
]
