from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from memetrader.types import FidelityTier, Mark, NON_EXECUTABLE_NOTICE, finite


@dataclass
class PositionCapacity:
    """Capacity analysis for one open position."""

    symbol: str
    quantity_ui: float
    reference_mark_usd: float          # mid * quantity — may overstate
    executable_liquidation_usd: float  # size-specific, conservative
    mark_basis: str                    # from Mark.basis
    is_executable_mark: bool
    estimated_price_impact_pct: float  # at current size
    capacity_at_1pct_impact_usd: float # size at which impact = 1%
    haircut_pct: float                 # applied to get executable value
    warning: str | None                # set when mark is non-executable


@dataclass
class CapacityReport:
    positions: list[PositionCapacity]
    strategy_capacity_usd: float        # min(capacity_at_1pct_impact) across positions
    total_reference_mark_usd: float
    total_executable_liquidation_usd: float
    exceeds_intended_size: bool         # capacity > intended_live_position_usd
    intended_live_position_usd: float | None
    fidelity: FidelityTier
    non_executable_notice: str | None


def _compute_impact_pct(
    reference_mark_usd: float,
    adv_usd: float | None,
    price_impact_model: str,
    k: float = 1.0,
) -> float:
    """Compute price impact percentage for a given trade size and ADV."""
    if adv_usd is not None and reference_mark_usd > 0 and adv_usd > 0:
        if price_impact_model == "sqrt":
            return k * math.sqrt(reference_mark_usd / adv_usd) * 100.0
        else:  # linear
            return (reference_mark_usd / adv_usd) * 100.0
    return 10.0  # fixed conservative haircut when no volume data


def _compute_capacity_at_1pct(
    adv_usd: float | None,
    price_impact_model: str,
    k: float = 1.0,
) -> float:
    """Compute the position size (USD) at which price impact equals 1%."""
    if adv_usd is None:
        return 0.0
    if price_impact_model == "sqrt":
        # impact_pct = k * sqrt(size / adv) * 100 = 1
        # sqrt(size / adv) = 0.01 / k
        # size = adv * (0.01 / k)^2
        return adv_usd * (0.01 / k) ** 2
    else:  # linear
        # impact_pct = (size / adv) * 100 = 1
        # size = adv * 0.01
        return adv_usd * 0.01


def liquidation_value(
    mark: Mark,
    quantity_ui: float,
    *,
    fidelity: FidelityTier,
    bar_volume_usd: float | None = None,
    participation_cap: float = 0.01,
    price_impact_model: str = "sqrt",
) -> tuple[float, float]:
    """Return (reference_mark_usd, executable_liquidation_usd).

    At TIER_2+, if mark.is_executable_basis, executable = reference (route-derived).
    At TIER_0 or non-route marks, apply estimated price impact haircut.
    """
    reference_mark_usd = (mark.price_usd or 0.0) * quantity_ui

    if fidelity.permits_pnl_claim and mark.is_executable_basis:
        executable_liquidation_usd = reference_mark_usd
    else:
        adv_usd: float | None = None
        if bar_volume_usd is not None and participation_cap > 0:
            adv_usd = bar_volume_usd / participation_cap

        impact_pct = _compute_impact_pct(reference_mark_usd, adv_usd, price_impact_model)
        executable_liquidation_usd = reference_mark_usd * (1.0 - impact_pct / 100.0)
        executable_liquidation_usd = max(0.0, executable_liquidation_usd)

    return (reference_mark_usd, executable_liquidation_usd)


def estimate_capacity(
    marks: dict[str, Mark],
    positions: dict[str, Any],  # str -> Position (duck-typed: has .quantity float)
    *,
    fidelity: FidelityTier,
    bar_volume_usd: dict[str, float] | None = None,
    participation_cap: float = 0.01,
    intended_live_position_usd: float | None = None,
    price_impact_model: str = "sqrt",  # "sqrt" or "linear"
) -> CapacityReport:
    """Estimate capacity for all open positions and produce a CapacityReport."""
    position_capacities: list[PositionCapacity] = []

    for symbol, position in positions.items():
        mark = marks.get(symbol)
        if mark is None:
            # No mark available — skip with zeroed-out entry
            position_capacities.append(
                PositionCapacity(
                    symbol=symbol,
                    quantity_ui=float(position.quantity),
                    reference_mark_usd=0.0,
                    executable_liquidation_usd=0.0,
                    mark_basis="unavailable",
                    is_executable_mark=False,
                    estimated_price_impact_pct=0.0,
                    capacity_at_1pct_impact_usd=0.0,
                    haircut_pct=0.0,
                    warning=NON_EXECUTABLE_NOTICE,
                )
            )
            continue

        quantity_ui = float(position.quantity)
        sym_bar_volume = (bar_volume_usd or {}).get(symbol)

        reference_mark_usd, executable_liquidation_usd = liquidation_value(
            mark,
            quantity_ui,
            fidelity=fidelity,
            bar_volume_usd=sym_bar_volume,
            participation_cap=participation_cap,
            price_impact_model=price_impact_model,
        )

        # Compute ADV for impact and capacity calculations
        adv_usd: float | None = None
        if sym_bar_volume is not None and participation_cap > 0:
            adv_usd = sym_bar_volume / participation_cap

        estimated_price_impact_pct = _compute_impact_pct(
            reference_mark_usd, adv_usd, price_impact_model
        )

        capacity_at_1pct_impact_usd = _compute_capacity_at_1pct(adv_usd, price_impact_model)

        if reference_mark_usd > 0:
            haircut_pct = 100.0 * (1.0 - executable_liquidation_usd / reference_mark_usd)
        else:
            haircut_pct = 0.0

        is_executable = mark.is_executable_basis and fidelity.permits_pnl_claim
        warning: str | None = None
        if not mark.is_executable_basis or not fidelity.permits_pnl_claim:
            warning = NON_EXECUTABLE_NOTICE

        position_capacities.append(
            PositionCapacity(
                symbol=symbol,
                quantity_ui=quantity_ui,
                reference_mark_usd=reference_mark_usd,
                executable_liquidation_usd=executable_liquidation_usd,
                mark_basis=mark.basis,
                is_executable_mark=is_executable,
                estimated_price_impact_pct=estimated_price_impact_pct,
                capacity_at_1pct_impact_usd=capacity_at_1pct_impact_usd,
                haircut_pct=haircut_pct,
                warning=warning,
            )
        )

    strategy_capacity_usd = (
        min(pc.capacity_at_1pct_impact_usd for pc in position_capacities)
        if position_capacities
        else 0.0
    )

    total_reference_mark_usd = sum(pc.reference_mark_usd for pc in position_capacities)
    total_executable_liquidation_usd = sum(
        pc.executable_liquidation_usd for pc in position_capacities
    )

    exceeds_intended_size = (
        strategy_capacity_usd > intended_live_position_usd
        if intended_live_position_usd is not None
        else False
    )

    non_executable_notice: str | None = (
        None if fidelity.permits_pnl_claim else NON_EXECUTABLE_NOTICE
    )

    return CapacityReport(
        positions=position_capacities,
        strategy_capacity_usd=strategy_capacity_usd,
        total_reference_mark_usd=total_reference_mark_usd,
        total_executable_liquidation_usd=total_executable_liquidation_usd,
        exceeds_intended_size=exceeds_intended_size,
        intended_live_position_usd=intended_live_position_usd,
        fidelity=fidelity,
        non_executable_notice=non_executable_notice,
    )
