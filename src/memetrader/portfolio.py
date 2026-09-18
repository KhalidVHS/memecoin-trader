"""Mark the book to market, and decide who has to be stopped out.

Two small functions, one of which contains the single easiest 100x bug in the
project: ``stop_loss_pct`` is a *fraction* (0.15 means -15%) while every ``_pct``
on a computed value is a *whole percent* (-15.0 means -15%). The conversion
lives in exactly one place, below, and is tested at the boundary.
"""

from __future__ import annotations

import logging
import math
import time

from .broker import LocalPaperBroker
from .types import PortfolioState

logger = logging.getLogger(__name__)

#: ``0.15 * 100`` is ``15.000000000000002`` in binary floating point, so a
#: drawdown of exactly -15.0% would miss a naive ``<=`` against the threshold.
#: This slack is far larger than that representation error and far smaller than
#: any price move anyone cares about.
_PCT_EPSILON = 1e-9


def mark(
    broker: LocalPaperBroker,
    marks: dict[str, float],
    *,
    now: float | None = None,
) -> PortfolioState:
    """Value the book at ``marks``, filling in every ``PortfolioState`` field.

    A position with no usable mark is **carried at cost basis** rather than
    dropped or zeroed: a missing price is an absence of information, not news
    that the coin went to zero, and either alternative would corrupt
    ``total_value_usd``. Such a position shows zero unrealized P&L and is
    logged as a warning. ``PortfolioState`` has no free-text field, so the
    implied cost-basis price is recorded in ``marks`` to keep that dict
    parallel with ``positions`` for anything downstream that renders it.
    """
    ts = time.time() if now is None else now
    positions = broker.get_positions()

    marks_used: dict[str, float] = {}
    values: dict[str, float] = {}
    unrealized = 0.0

    for symbol, position in positions.items():
        price = marks.get(symbol)
        if price is None or not math.isfinite(price) or price <= 0.0:
            value = position.cost_basis_usd
            implied = value / position.quantity if position.quantity else 0.0
            logger.warning(
                "no usable mark for %s (got %r); carrying at cost basis $%.2f",
                symbol,
                price,
                value,
            )
            marks_used[symbol] = implied
        else:
            value = position.quantity * price
            marks_used[symbol] = price

        values[symbol] = value
        unrealized += value - position.cost_basis_usd

    return PortfolioState(
        ts=ts,
        cash_usd=broker.cash_usd,
        positions=positions,
        marks=marks_used,
        position_values_usd=values,
        unrealized_pnl_usd=unrealized,
        realized_pnl_usd=broker.realized_pnl_usd,
        total_value_usd=broker.cash_usd + sum(values.values()),
        starting_cash_usd=broker.starting_cash_usd,
        fees_paid_usd=broker.fees_paid_usd,
        gas_paid_usd=broker.gas_paid_usd,
    )


def stop_loss_breaches(state: PortfolioState, stop_loss_pct: float) -> list[str]:
    """Symbols at or beyond the stop, worst drawdown first.

    ``stop_loss_pct`` is a fraction from config (0.15); the comparison is done
    in whole percents against ``Position.unrealized_pnl_pct``, which measures
    against cost basis — so entry fees and gas count toward the drawdown, as
    they should, since they are money that is genuinely gone.

    Worst-first ordering matters when acting on the result: if only some exits
    can be filled this tick, the deepest loser should go first.
    """
    threshold_pct = -abs(stop_loss_pct) * 100.0

    breached: list[tuple[float, str]] = []
    for symbol, position in state.positions.items():
        price = state.marks.get(symbol)
        if price is None:
            # Unmarkable, so unjudgeable. Do not force an exit on the strength
            # of data we do not have.
            continue
        pnl_pct = position.unrealized_pnl_pct(price)
        if pnl_pct <= threshold_pct + _PCT_EPSILON:
            breached.append((pnl_pct, symbol))

    breached.sort()
    return [symbol for _, symbol in breached]


__all__ = ["mark", "stop_loss_breaches"]
