"""Portfolio construction — turn strategy proposals into cash-safe orders.

Every strategy in ``baselines.py`` proposes; it never sizes a final order and
never executes one (see that module's docstring). This module is where sizing
against available cash, per-asset and portfolio-level exposure caps, and
turnover control actually happen. The one non-negotiable invariant is in the
assignment's own words: **sizing must never produce a position that exceeds
available cash — assert this, don't assume it.** :func:`size_orders` does,
twice: once structurally (every intermediate step only ever scales amounts
*down*, never up) and once with an explicit ``assert`` on the final result.

No float arithmetic on money, anywhere in this module. Every cap is expressed
in **basis points** (an integer, 1 bp = 0.01%) rather than the whole-number-
percent convention used elsewhere in the codebase (``docs/BACKTEST-
CONTRACTS.md`` §0), specifically so that "20%" is the integer ``2000``, not
the float ``20.0`` that would otherwise multiply an integer cash amount.
``_usd_to_micro`` is the one documented float→int boundary conversion, used
only by callers translating a ``PortfolioState.cash_usd`` (a float, by
``types.py``'s own contract) into the integer micro-USD this module operates
on — never used internally by :func:`size_orders` itself.

Universe-shrinkage rule (documented, load-bearing; consistent with
``baselines.py``'s own rule for the same event)
--------------------------------------------------------------------
A BUY naming a symbol that is not a member of the ``universe`` passed to
:func:`size_orders` is dropped silently — no exception, no zero-size order
left behind to be submitted. This is deliberately permissive of a stale
proposal: a strategy's ``propose`` call and the tick at which its output
reaches construction are not guaranteed to observe exactly the same universe
snapshot in every possible pipeline, and crashing a whole replay over an
ordinary delisting is a worse failure mode than silently declining to open a
position in a token that cannot be bought. SELL/exit intents are **never**
filtered by universe membership, for the opposite reason: universe governs
what may be *entered*, never what may be *exited*. The fastest way to strand
cash forever in a token that will never trade again is to refuse to let it be
sold once it disappears from a universe snapshot.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

from memetrader.types import (
    Forecast,
    OrderIntent,
    PortfolioState,
    Position,
    Side,
    TargetPosition,
)

if TYPE_CHECKING:
    from memetrader.histdata.point_in_time import PointInTimeState

#: Matches ``broker.py``'s ``USDC_DECIMALS``-derived micro-USD convention
#: (``_MICRO = 10**USDC_DECIMALS = 1_000_000``): 1 USD = 1_000_000 micro-USD.
MICRO_USD = 1_000_000


def _usd_to_micro(usd: float) -> int:
    """Floor-convert a dollar amount to integer micro-USD.

    The only float→int boundary in this module. Floors rather than rounds:
    rounding up on a cash conversion is exactly how a sizer ends up proposing
    a spend a fraction of a cent larger than what is actually available.
    """
    if usd <= 0.0 or not math.isfinite(usd):
        return 0
    return int(usd * MICRO_USD)


def _scale_to_budget(amounts: Mapping[str, int], budget: int) -> dict[str, int]:
    """Proportionally scale ``amounts`` down so their sum is <= ``budget``.

    A no-op if already within budget (including ``budget <= 0`` and an empty
    or all-zero ``amounts``, both of which are already <= any non-negative
    budget only when their own sum is <= budget — handled by the same branch).
    Otherwise scales every entry by ``budget / total`` using integer floor
    division, which guarantees ``sum(scaled) <= budget`` unconditionally: the
    sum of floors of shares of a total can never exceed the floor of their
    (equal) sum, and ``budget`` is already an integer. No remainder
    redistribution pass is needed because the invariant required here is
    "<=", never "==".
    """
    total = sum(amounts.values())
    if total <= 0 or total <= budget:
        return dict(amounts)
    scaled = {symbol: (amt * budget) // total for symbol, amt in amounts.items()}
    assert sum(scaled.values()) <= budget, (
        f"_scale_to_budget produced {sum(scaled.values())} > budget {budget}"
    )
    return scaled


@dataclass(frozen=True, slots=True)
class ConstructionLimits:
    """Caps applied uniformly by :func:`size_orders`.

    Expressed in basis points (int), not the whole-number-percent convention
    used elsewhere — see module docstring for why. ``2000`` bps == 20%.
    """

    per_asset_cap_bps: int = 2_000
    portfolio_cap_bps: int = 10_000
    # Turnover control: caps total new BUY notional sized in one call. ``None``
    # means uncapped (the cash and exposure caps alone still bind).
    max_new_exposure_micro_usd: int | None = None

    def __post_init__(self) -> None:
        if not 0 < self.per_asset_cap_bps <= 10_000:
            raise ValueError(
                f"per_asset_cap_bps must be in (0, 10_000], got {self.per_asset_cap_bps}"
            )
        if not 0 < self.portfolio_cap_bps <= 10_000:
            raise ValueError(
                f"portfolio_cap_bps must be in (0, 10_000], got {self.portfolio_cap_bps}"
            )
        if (
            self.max_new_exposure_micro_usd is not None
            and self.max_new_exposure_micro_usd < 0
        ):
            raise ValueError("max_new_exposure_micro_usd must be >= 0")


#: Shared, immutable default — used instead of constructing a fresh
#: ``ConstructionLimits()`` in a function signature (ruff B008: a mutable- or
#: call-in-default is evaluated once at import time either way, so naming the
#: singleton makes that explicit instead of looking like a fresh call).
_DEFAULT_LIMITS = ConstructionLimits()


def size_orders(
    intents: Sequence[OrderIntent],
    *,
    cash_micro_usd: int,
    portfolio_value_micro_usd: int,
    existing_exposure_micro_usd: Mapping[str, int],
    positions: Mapping[str, Position],
    universe: frozenset[str],
    limits: ConstructionLimits = _DEFAULT_LIMITS,
) -> tuple[OrderIntent, ...]:
    """Resize a batch of strategy-proposed ``OrderIntent``s into cash-safe orders.

    ``existing_exposure_micro_usd`` is the already-marked value of each held
    position in micro-USD, supplied by the caller — this module does no
    pricing of its own; sizing and marking are different responsibilities.

    Processing order, each step only ever shrinking a proposal, never growing
    one past what the strategy itself proposed:

    1. SELLs are defensively clamped to at most the actually-held
       ``quantity_atomic`` (never more, even if the intent claims otherwise)
       and pass through unconditionally otherwise — never capped by cash or
       exposure limits, and never filtered by universe membership. See module
       docstring.
    2. BUYs naming a symbol outside ``universe`` are dropped.
    3. Remaining BUYs are capped per-asset: existing exposure plus new spend
       may not exceed ``per_asset_cap_bps`` of ``portfolio_value_micro_usd``.
    4. The total of what remains is capped at the portfolio level the same
       way, scaled down proportionally if it does not fit.
    5. If ``limits.max_new_exposure_micro_usd`` is set, the total is scaled
       down to fit it (turnover control).
    6. Finally, the total is scaled down to fit ``cash_micro_usd`` — this is
       the invariant that must never be violated, and it is asserted.

    Returns SELLs first, then BUYs sorted by symbol — deterministic given
    deterministic input, so two identical calls produce byte-identical output.
    """
    if cash_micro_usd < 0:
        raise ValueError(f"cash_micro_usd must be >= 0, got {cash_micro_usd}")
    if portfolio_value_micro_usd < 0:
        raise ValueError(
            f"portfolio_value_micro_usd must be >= 0, got {portfolio_value_micro_usd}"
        )

    sells: list[OrderIntent] = []
    buy_amounts: dict[str, int] = {}
    buy_source: dict[str, OrderIntent] = {}

    for intent in intents:
        if intent.side is Side.SELL:
            held = positions.get(intent.symbol)
            held_qty = held.quantity_atomic if held is not None else 0
            safe_amt = min(intent.in_amount_atomic, held_qty)
            if safe_amt <= 0:
                continue
            sells.append(
                replace(
                    intent,
                    in_amount_atomic=safe_amt,
                    max_in_amount_atomic=min(intent.max_in_amount_atomic, safe_amt),
                )
            )
            continue

        if intent.symbol not in universe:
            continue
        buy_amounts[intent.symbol] = buy_amounts.get(intent.symbol, 0) + max(
            0, intent.in_amount_atomic
        )
        buy_source.setdefault(intent.symbol, intent)

    # Per-asset cap.
    asset_cap_micro = (portfolio_value_micro_usd * limits.per_asset_cap_bps) // 10_000
    capped: dict[str, int] = {}
    for symbol, amt in buy_amounts.items():
        existing = existing_exposure_micro_usd.get(symbol, 0)
        headroom = max(0, asset_cap_micro - existing)
        clamped = min(amt, headroom)
        if clamped > 0:
            capped[symbol] = clamped

    # Portfolio-level cap.
    portfolio_cap_micro = (portfolio_value_micro_usd * limits.portfolio_cap_bps) // 10_000
    existing_total = sum(existing_exposure_micro_usd.values())
    portfolio_headroom = max(0, portfolio_cap_micro - existing_total)
    capped = {
        s: a for s, a in _scale_to_budget(capped, portfolio_headroom).items() if a > 0
    }

    # Turnover control.
    if limits.max_new_exposure_micro_usd is not None:
        capped = {
            s: a
            for s, a in _scale_to_budget(capped, limits.max_new_exposure_micro_usd).items()
            if a > 0
        }

    # Cash safety — must never be violated.
    capped = {s: a for s, a in _scale_to_budget(capped, cash_micro_usd).items() if a > 0}
    assert sum(capped.values()) <= cash_micro_usd, (
        "size_orders produced a spend exceeding available cash: "
        f"{sum(capped.values())} > {cash_micro_usd}"
    )
    # These assert that construction never *adds* exposure beyond headroom —
    # not that the resulting total sits under the cap outright. Existing
    # exposure can already be at or above a cap purely from price movement
    # (e.g. a token that has since rallied hard), which is not something
    # sizing new orders can retroactively undo; headroom is already floored
    # at 0 in that case, so the only thing this module can guarantee is that
    # it adds nothing new on top.
    for symbol, amt in capped.items():
        asset_existing = existing_exposure_micro_usd.get(symbol, 0)
        asset_headroom = max(0, asset_cap_micro - asset_existing)
        assert amt <= asset_headroom, (
            f"size_orders breached the per-asset headroom for {symbol}: "
            f"{amt} > {asset_headroom}"
        )
    assert sum(capped.values()) <= portfolio_headroom, (
        "size_orders breached the portfolio-level exposure headroom: "
        f"{sum(capped.values())} > {portfolio_headroom}"
    )

    buys = [
        replace(
            buy_source[symbol],
            in_amount_atomic=amt,
            max_in_amount_atomic=min(buy_source[symbol].max_in_amount_atomic, amt),
        )
        for symbol, amt in sorted(capped.items())
    ]

    return tuple(sells) + tuple(buys)


# ---------------------------------------------------------------------------
# Frozen contract: docs/BACKTEST-CONTRACTS.md §4's PortfolioConstructor
# ---------------------------------------------------------------------------


class PortfolioConstructor(Protocol):
    """Reproduced verbatim from ``docs/BACKTEST-CONTRACTS.md`` §4.

    ``baselines.py``'s strategies propose ``OrderIntent`` directly (see that
    module's docstring for why), so the module's actually-tested, load-
    bearing sizing path is :func:`size_orders`, not this protocol. This
    Protocol and :class:`EqualRiskForecastConstructor` are provided so the
    module also satisfies the frozen contract literally for any producer that
    speaks in ``Forecast``\\ s instead.
    """

    def build_targets(
        self,
        forecasts: Sequence[Forecast],
        portfolio: PortfolioState,
        market: PointInTimeState,
    ) -> tuple[TargetPosition, ...]: ...


@dataclass(frozen=True, slots=True)
class EqualRiskForecastConstructor:
    """Minimal ``PortfolioConstructor``: equal-dollar sizing among actionable,
    in-universe forecasts, capped the same integer-safe way as
    :func:`size_orders`.

    "Actionable" per ``types.Forecast.actionable``; additionally requires a
    positive ``lower_quantile_pct`` (contracts §0: the entry hurdle is tested
    against the conservative quantile, not the point estimate) and universe
    membership (see module docstring, universe rule).
    """

    limits: ConstructionLimits = _DEFAULT_LIMITS

    def build_targets(
        self,
        forecasts: Sequence[Forecast],
        portfolio: PortfolioState,
        market: PointInTimeState,
    ) -> tuple[TargetPosition, ...]:
        universe = market.universe()
        actionable = [
            f
            for f in forecasts
            if f.actionable
            and f.lower_quantile_pct is not None
            and f.lower_quantile_pct > 0
            and f.symbol in universe
        ]
        if not actionable:
            return ()

        cash_micro = _usd_to_micro(portfolio.cash_usd)
        share_micro = cash_micro // len(actionable)
        if share_micro <= 0:
            return ()

        asset_cap_micro = (
            _usd_to_micro(portfolio.total_value_usd or 0.0) * self.limits.per_asset_cap_bps
        ) // 10_000
        share_micro = (
            min(share_micro, asset_cap_micro) if asset_cap_micro > 0 else share_micro
        )

        return tuple(
            TargetPosition(
                symbol=f.symbol,
                target_usd=share_micro / MICRO_USD,
                forecast=f,
                rationale="equal_risk_forecast_constructor",
            )
            for f in sorted(actionable, key=lambda x: x.symbol)
        )


__all__ = [
    "MICRO_USD",
    "ConstructionLimits",
    "EqualRiskForecastConstructor",
    "PortfolioConstructor",
    "size_orders",
]
