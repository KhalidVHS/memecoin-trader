"""What to hold, and why. The layer the audit says must not be a language model.

Audit C6 is the finding this module exists to answer: the LLM had order
authority without any measured predictive value behind it. The evidence for
that is in the 12-hour live run — 147 decisions, of which 142 were HOLD, four
BUY and one SELL, producing exactly one closed round trip. One trade is not a
sample. It cannot distinguish skill from noise, it cannot be tested for
calibration, and it cannot justify the $3.48 of model spend that produced
-$3.19 of trading P&L. A system that cannot measure its own edge must not size
on it.

So the default strategy here is deterministic, cheap, and *labelled as a
baseline rather than as alpha*. The point of a baseline is not that it makes
money; it is that it gives every future claim something to be measured against.
Without one, "the model did well this week" has no denominator.

Three design choices worth stating plainly:

**Targets, not trades.** A strategy returns desired inventory per symbol as a
:class:`TargetPosition`. The execution layer diffs targets against what is
actually held and schedules the delta. This collapses "hold what you have" and
"buy more" into one statement instead of two code paths, and it makes an
all-cash portfolio expressible as the ordinary case (every target zero) rather
than as a special "do nothing" branch. The old ``BUY``/``SELL``/``HOLD``
vocabulary could not represent "I want $40 of this and I have $55".

**Every view carries a horizon and a cost-inclusive magnitude.** A direction
with no horizon and no size cannot be compared against a cost hurdle, so the
old system had no way to ask whether a trade was worth its own gas. The 12-hour
run contains a directionally correct trade that still lost money, which is
exactly the failure that omission produces. :class:`Forecast` therefore carries
``horizon_seconds`` and ``expected_net_return_pct`` — *net*, after estimated
costs — plus a predictive interval. The entry hurdle is tested against the
conservative end of that interval, not the mean, because trading the mean of a
wide distribution is how a noisy estimate becomes a position.

**Uncalibrated means unsized.** ``Forecast.calibration_id`` is ``None`` for any
model that makes no calibration claim, which includes everything currently in
this file. A ``None`` calibration forbids scaling size with conviction; the
caller falls back to a size drawn from available cash instead — see
``StrategySettings.cash_fraction_per_entry``. This is what stops a confident
number from being *treated* as a calibrated number: cash on hand is not a
forecast, so sizing on it does not smuggle conviction back in.

The LLM path survives behind :class:`AdvisoryStrategy`, off by default, taking
no raw social text (audit C7) and producing advice that still passes through
the same risk bounds and the same cost hurdle as everything else. It is
retained because the audit's own §8 leaves the door open to a measured,
ablated language component — not because it has earned order authority. It has
not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from .ids import new_decision_id
from .types import (
    DataQuality,
    EvidenceBundle,
    Forecast,
    PortfolioState,
    StrategyDecision,
    TargetPosition,
    ValidationError,
    finite,
    non_negative,
)

log = logging.getLogger(__name__)

__all__ = [
    "AdvisoryStrategy",
    "BaselineStrategy",
    "CashStrategy",
    "Strategy",
    "StrategySettings",
    "build_strategy",
]


@dataclass(frozen=True, slots=True)
class StrategySettings:
    """Knobs every strategy shares.

    These are deliberately not read from a module-level config object. A
    strategy that reaches out for its own configuration cannot be evaluated
    twice under two parameterisations in the same process, which is precisely
    what a sweep or a walk-forward test has to do.
    """

    horizon_seconds: float = 3600.0
    # The hurdle the conservative end of the forecast must clear. Set from the
    # measured round-trip cost, not from taste: the 12-hour run's single closed
    # trade paid roughly 0.6% in pool fee plus gas on a position that moved less
    # than that, which is the whole reason a hurdle exists.
    entry_hurdle_pct: float = 1.0
    # Flat size used whenever the forecast is uncalibrated, which is currently
    # always. Expressed in dollars rather than as a fraction so a shrinking book
    # does not silently shrink the experiment's sample size too. Retained as
    # the sizing fallback for the advisory path's non-BUY bookkeeping and for
    # any caller that has no cash figure to size against; ordinary entries are
    # sized by `cash_fraction_per_entry` instead — see below.
    flat_size_usd: float = 50.0
    # Fraction of currently uncommitted cash a *new* entry may target, at the
    # operator's explicit request (2026-09-20) to size positions against
    # available capital rather than a flat dollar figure regardless of book
    # size. This is a bankroll/allocation rule, not a conviction score — it
    # does not touch the "uncalibrated means unsized" rule above, which is
    # about scaling by an unproven forecast magnitude, not about scaling by
    # what cash happens to be free. A symbol already held keeps its current
    # marked value as its target instead of being resized to this fraction
    # every cycle; see `decide()`.
    cash_fraction_per_entry: float = 0.5
    # Below this, a target change is not worth an order: two round trips of gas
    # on a $3 delta is a guaranteed loss regardless of direction.
    min_trade_usd: float = 10.0
    # A target within this band of current inventory is left alone. Without it,
    # a target that drifts by a dollar a tick generates an order a tick.
    rebalance_band_usd: float = 15.0
    max_positions: int = 2
    # Fraction of the trailing move carried into the forecast. Higher means the
    # strategy takes the observed trend more at face value; lower means it
    # treats the trend as mostly noise. This is a live parameter, not fitted
    # to this sample, and moving it up trades statistical caution for trade
    # frequency — it does not change the round-trip cost the hurdle nets out.
    shrinkage: float = 0.10
    # Half-width of the predictive interval, as a multiple of realised
    # volatility. The entry hurdle is tested against the *lower* end of this
    # interval (net - half_width), so a smaller multiplier narrows the interval
    # and lets more forecasts clear the same hurdle without changing the
    # hurdle or the point estimate itself.
    interval_vol_multiple: float = 1.5

    def __post_init__(self) -> None:
        non_negative(self.entry_hurdle_pct, "entry_hurdle_pct")
        non_negative(self.flat_size_usd, "flat_size_usd")
        non_negative(self.min_trade_usd, "min_trade_usd")
        non_negative(self.rebalance_band_usd, "rebalance_band_usd")
        non_negative(self.shrinkage, "shrinkage")
        non_negative(self.interval_vol_multiple, "interval_vol_multiple")
        if not 0.0 < self.cash_fraction_per_entry <= 1.0:
            raise ValidationError(
                "cash_fraction_per_entry is a FRACTION (0.5 = 50%) and must be "
                f"in (0, 1], got {self.cash_fraction_per_entry}"
            )


class Strategy(Protocol):
    """The seam that makes the LLM replaceable rather than load-bearing.

    Anything implementing this can be swapped in without the execution, risk or
    accounting layers knowing which one is running — which is the precondition
    for ever comparing two of them honestly.
    """

    @property
    def strategy_id(self) -> str: ...

    def decide(
        self,
        evidence: dict[str, EvidenceBundle],
        portfolio: PortfolioState,
        *,
        now: float,
    ) -> StrategyDecision: ...


class CashStrategy:
    """Hold cash. Target zero everywhere.

    This is the audit's Phase 1 instruction taken literally — "replace decisions
    temporarily with HOLD/cash" — and it is also the null hypothesis every other
    strategy has to beat. It is not a placeholder: a system whose measured edge
    is indistinguishable from this one should be running this one, because this
    one costs nothing.
    """

    strategy_id = "cash-v1"

    def decide(
        self,
        evidence: dict[str, EvidenceBundle],
        portfolio: PortfolioState,
        *,
        now: float,
    ) -> StrategyDecision:
        return StrategyDecision(
            decision_id=new_decision_id(),
            ts=now,
            strategy_id=self.strategy_id,
            market_read="Flat by construction. No view is being expressed.",
            targets=tuple(TargetPosition(symbol=s, target_usd=0.0) for s in evidence),
            forecasts=(),
            diagnostics={"note": "cash baseline"},
        )


class BaselineStrategy:
    """A deterministic, documented, deliberately unimpressive momentum baseline.

    It exists to be a denominator. Every property that makes it weak is also
    what makes it a usable control: it is a closed-form function of two closed
    bars, it has no fitted parameters, it cannot overfit a 12-hour sample, and
    it reproduces exactly from the same inputs.

    The rule: take the 1-hour close-to-close return over *closed* bars only,
    shrink it hard toward zero, subtract estimated round-trip cost, and take a
    position only if the conservative end of the resulting interval still clears
    the hurdle. The shrinkage factor is the honest part. A raw trailing return
    is a terrible estimate of the next return in this asset class — the audit's
    §9 is blunt that momentum on a jump process is mostly noise — so the
    estimate is scaled down by an order of magnitude rather than pretending the
    trailing move will repeat. A strategy that would not trade after honest
    shrinkage should not trade.

    Deliberately NOT here: any score counting how many indicators agree. RSI,
    EMA, MACD and Bollinger are correlated transforms of one close series, so
    their agreement is arithmetic, not confirmation, and summing them
    manufactures confidence out of a single number seen five ways. The
    codebase already measured this — eight nominal signals collapsed to two
    independent ones — and the audit repeats it.
    """

    strategy_id = "baseline-momentum-v1"

    # ``shrinkage`` and ``interval_vol_multiple`` used to be fixed class
    # constants (0.10 and 1.5). They now live on `StrategySettings` so they can
    # be tuned from config.toml without a code change — see the docstrings
    # there for what moving each one trades away.
    # Estimated round-trip cost in whole percent: pool fee both ways plus gas
    # plus expected slippage. Subtracted before the hurdle test, which is what
    # makes the hurdle a *net* hurdle.
    ROUND_TRIP_COST_PCT = 0.8

    def __init__(self, settings: StrategySettings) -> None:
        self.settings = settings

    def decide(
        self,
        evidence: dict[str, EvidenceBundle],
        portfolio: PortfolioState,
        *,
        now: float,
    ) -> StrategyDecision:
        forecasts: list[Forecast] = []
        for symbol, bundle in sorted(evidence.items()):
            forecasts.append(self._forecast(symbol, bundle))

        # Rank only on forecasts that are both actionable and clear the hurdle.
        # An unactionable forecast is not a weak buy, it is an absence of a
        # view, and the two must not be allowed to sort against each other.
        eligible = [
            f
            for f in forecasts
            if f.actionable
            and f.lower_quantile_pct is not None
            and f.lower_quantile_pct >= self.settings.entry_hurdle_pct
        ]
        eligible.sort(key=lambda f: f.lower_quantile_pct or 0.0, reverse=True)
        chosen = {f.symbol: f for f in eligible[: self.settings.max_positions]}

        # Sizing is against available cash, not against forecast magnitude.
        # `calibration_id is None` on every forecast this class produces, and
        # an uncalibrated number must still not scale a position — see the
        # module docstring — but *how much cash happens to be free* is a
        # different axis and is exactly what this sizes on. A symbol already
        # held keeps its current marked value as target so being re-chosen on
        # a later tick does not force a resize; only a genuinely new entry
        # draws from the (sequentially shrinking) free-cash pool, in rank
        # order — the strongest-clearing forecast gets first claim on cash —
        # so several new entries chosen in one decision cannot jointly
        # overcommit it.
        free_cash = portfolio.cash_usd
        target_usd_by_symbol: dict[str, float] = dict.fromkeys(evidence, 0.0)
        for symbol in chosen:
            held = portfolio.position_values_usd.get(symbol)
            if held:
                target_usd_by_symbol[symbol] = held
                continue
            size = free_cash * self.settings.cash_fraction_per_entry
            target_usd_by_symbol[symbol] = size
            free_cash = max(0.0, free_cash - size)

        targets = tuple(
            TargetPosition(
                symbol=symbol,
                target_usd=target_usd_by_symbol[symbol],
                forecast=chosen.get(symbol),
                rationale=self._rationale(symbol, chosen, forecasts),
            )
            for symbol in sorted(evidence)
        )

        return StrategyDecision(
            decision_id=new_decision_id(),
            ts=now,
            strategy_id=self.strategy_id,
            market_read=self._market_read(forecasts, chosen),
            targets=targets,
            forecasts=tuple(forecasts),
            diagnostics={
                "eligible": ",".join(chosen) or "none",
                "hurdle_pct": f"{self.settings.entry_hurdle_pct:.2f}",
                "shrinkage": f"{self.settings.shrinkage:.2f}",
            },
        )

    def _forecast(self, symbol: str, bundle: EvidenceBundle) -> Forecast:
        """One symbol's view, or an explicit absence of one.

        Every path that cannot compute an honest number returns a forecast with
        ``expected_net_return_pct=None`` and the reason recorded in
        ``features_missing``. None of them return a neutral zero: a zero
        expected return is a *view* that the price will not move, and it must
        not be produced by a missing input.
        """
        missing: list[str] = []

        if bundle.snapshot.quality is not DataQuality.OK:
            missing.append(f"snapshot_quality={bundle.snapshot.quality}")
        if not bundle.snapshot.pool.trusted_quote:
            # A pool priced in an unknown token gives a number, not a valuation.
            missing.append("untrusted_quote_token")
        if bundle.technicals is None:
            missing.append("technicals")

        if missing:
            return Forecast(
                symbol=symbol,
                horizon_seconds=self.settings.horizon_seconds,
                expected_net_return_pct=None,
                lower_quantile_pct=None,
                upper_quantile_pct=None,
                model_id=self.strategy_id,
                calibration_id=None,
                features_missing=tuple(missing),
                note="no view: required evidence unavailable",
            )

        assert bundle.technicals is not None
        h1 = bundle.technicals.h1
        trailing = bundle.snapshot.price_change.h1
        vol = h1.realized_vol_pct

        if trailing is None:
            missing.append("price_change.h1")
        if vol is None:
            missing.append("realized_vol_pct")
        if missing:
            return Forecast(
                symbol=symbol,
                horizon_seconds=self.settings.horizon_seconds,
                expected_net_return_pct=None,
                lower_quantile_pct=None,
                upper_quantile_pct=None,
                model_id=self.strategy_id,
                calibration_id=None,
                features_missing=tuple(missing),
                note="no view: required features unavailable",
            )

        assert trailing is not None and vol is not None
        gross = finite(trailing, "trailing_h1_pct") * self.settings.shrinkage
        net = gross - self.ROUND_TRIP_COST_PCT
        half_width = self.settings.interval_vol_multiple * finite(vol, "realized_vol_pct")

        return Forecast(
            symbol=symbol,
            horizon_seconds=self.settings.horizon_seconds,
            expected_net_return_pct=net,
            lower_quantile_pct=net - half_width,
            upper_quantile_pct=net + half_width,
            model_id=self.strategy_id,
            # Never set by this class. Nothing here has been calibrated against
            # realised outcomes, and claiming otherwise would let the caller
            # size on it.
            calibration_id=None,
            features_missing=(),
            note=(
                f"h1 {trailing:+.2f}% shrunk x{self.settings.shrinkage:.2f} = {gross:+.2f}%, "
                f"less {self.ROUND_TRIP_COST_PCT:.2f}% cost; vol {vol:.2f}%"
            ),
        )

    def _rationale(
        self,
        symbol: str,
        chosen: dict[str, Forecast],
        forecasts: list[Forecast],
    ) -> str:
        if symbol in chosen:
            return chosen[symbol].note
        for f in forecasts:
            if f.symbol == symbol:
                if not f.actionable:
                    return f"no view ({', '.join(f.features_missing) or f.note})"
                return f"below hurdle: {f.note}"
        return "no view"

    def _market_read(self, forecasts: list[Forecast], chosen: dict[str, Forecast]) -> str:
        blind = [f.symbol for f in forecasts if not f.actionable]
        parts = [
            f"Deterministic baseline over {len(forecasts)} symbols; "
            f"{len(chosen)} cleared a {self.settings.entry_hurdle_pct:.2f}% net hurdle."
        ]
        if blind:
            parts.append(f"No view available for {', '.join(blind)} — evidence missing.")
        parts.append(
            f"Uncalibrated: an entry takes "
            f"{self.settings.cash_fraction_per_entry:.0%} of free cash, "
            f"not a conviction-weighted size."
        )
        return " ".join(parts)


class AdvisoryStrategy:
    """The language model, demoted from decision-maker to one bounded input.

    Enabled only by explicit operator opt-in. Three things constrain it, each
    tracing to a specific audit finding:

    * It receives **no raw social text** (C7). Public Reddit bodies were being
      interpolated into the same prompt that had order authority, which meant
      any stranger could address the trader directly. Numbers cannot issue
      instructions; text can, so text no longer reaches it.
    * It produces **advice, not orders** (C6). Its output is converted to the
      same :class:`TargetPosition` shape as the baseline and passes through the
      same cost hurdle and the same risk bounds. It gets no privileged path.
    * Its advice is **discarded whole on any invalid field**, never repaired.
      ``brain`` used to silently fix bad output; a NaN size survived that repair
      because NaN is not ``< 0`` — it is not anything — and every risk
      comparison against it, including the rejections, evaluated false. A model
      that emits an invalid number has malfunctioned, and the right response is
      to drop the output, not to guess what it meant and trade the guess.

    Sizing stays flat for the same reason as the baseline: ``confidence`` from a
    language model is not a calibrated probability, and no measurement in this
    repository says otherwise.
    """

    strategy_id = "advisory-llm-v1"

    def __init__(self, settings: StrategySettings, brain: object) -> None:
        self.settings = settings
        self.brain = brain
        self._fallback = BaselineStrategy(settings)

    def decide(
        self,
        evidence: dict[str, EvidenceBundle],
        portfolio: PortfolioState,
        *,
        now: float,
    ) -> StrategyDecision:
        # The baseline runs regardless. Its forecasts are what the advisory
        # output gets attributed against later, and computing it here means an
        # advisory failure degrades to a working strategy rather than to a gap.
        base = self._fallback.decide(evidence, portfolio, now=now)
        decide_fn = getattr(self.brain, "decide", None)
        if decide_fn is None:
            log.error(
                "advisory strategy enabled but brain exposes no decide(); using baseline"
            )
            return base

        try:
            advice = decide_fn(evidence, portfolio, now=now)
        except Exception:
            # A failed advisory call must not halt trading and must not produce
            # a neutral-looking decision that hides the failure. Fall back
            # loudly to the deterministic baseline.
            log.exception(
                "advisory decision failed; falling back to deterministic baseline"
            )
            return base

        if advice is None:
            log.warning(
                "advisory decision discarded as invalid; using deterministic baseline"
            )
            return base

        targets = self._targets_from_advice(advice, evidence, base, portfolio)
        return StrategyDecision(
            decision_id=new_decision_id(),
            ts=now,
            strategy_id=self.strategy_id,
            market_read=getattr(advice, "market_read", ""),
            targets=targets,
            forecasts=base.forecasts,
            diagnostics={
                "baseline_decision_id": base.decision_id,
                "advisory": "used",
            },
        )

    def _targets_from_advice(
        self,
        advice: object,
        evidence: dict[str, EvidenceBundle],
        base: StrategyDecision,
        portfolio: PortfolioState,
    ) -> tuple[TargetPosition, ...]:
        """Translate advisory actions into inventory targets.

        A symbol the model did not mention gets a target of zero rather than
        being skipped. Silence is not a hold: the old vocabulary made "the model
        said nothing" and "the model said hold" the same event, which meant a
        truncated or partial response quietly preserved whatever was already on
        the book.
        """
        by_symbol: dict[str, float] = dict.fromkeys(evidence, 0.0)
        forecast_by_symbol = {f.symbol: f for f in base.forecasts}
        rationale: dict[str, str] = {}
        free_cash = portfolio.cash_usd

        for action in getattr(advice, "actions", ()):
            symbol = getattr(action, "symbol", None)
            verb = getattr(action, "action", None)
            if symbol not in by_symbol:
                # A hallucinated ticker is a malfunction, not a no-op. The whole
                # output is suspect, so nothing from it is trusted.
                log.error("advisory named unknown symbol %r; discarding advice", symbol)
                return base.targets
            if verb == "BUY":
                held = portfolio.position_values_usd.get(symbol)
                if held:
                    by_symbol[symbol] = held
                else:
                    size = free_cash * self.settings.cash_fraction_per_entry
                    by_symbol[symbol] = size
                    free_cash = max(0.0, free_cash - size)
            elif verb == "SELL":
                by_symbol[symbol] = 0.0
            rationale[symbol] = getattr(action, "reasoning", "")

        return tuple(
            TargetPosition(
                symbol=symbol,
                target_usd=by_symbol[symbol],
                forecast=forecast_by_symbol.get(symbol),
                rationale=rationale.get(
                    symbol, "not mentioned by advisory; treated as flat"
                ),
            )
            for symbol in sorted(by_symbol)
        )


def build_strategy(
    kind: str,
    settings: StrategySettings,
    *,
    brain: object | None = None,
) -> Strategy:
    """Select a strategy by name.

    The default is deliberately *not* the language model. Audit C6 says an
    unmeasured component must not hold order authority, and a default is
    authority — it is what runs when nobody made a decision. Turning the model
    on is now an explicit act that appears in configuration and in the journal,
    which is also what makes an A/B against the baseline possible at all.
    """
    match kind:
        case "cash":
            return CashStrategy()
        case "baseline":
            return BaselineStrategy(settings)
        case "advisory":
            if brain is None:
                raise ValueError("advisory strategy requires a brain")
            log.warning(
                "advisory (LLM) strategy enabled — audit C6 records that this component "
                "has no measured predictive value; sizing remains flat"
            )
            return AdvisoryStrategy(settings, brain)
        case _:
            raise ValueError(f"unknown strategy {kind!r}")
