"""Microstructure features derived from quote ladders.

These features require TIER_2 data (live quote ladders collected by
``shadow.py``). At TIER_0 — which is the current operational tier — no quote
ladders exist, and every feature in this module returns ``None`` with a recorded
reason rather than fabricating a value.

The distinction matters: ``None`` means "could not find out". Returning ``0.0``
or some neutral placeholder would silently suggest a flat spread and zero price
impact to any downstream model, which is optimistic in exactly the regime where
it is most dangerous (thin-market microstructure is the kill condition for
memecoin positions, not the baseline).

The ``QuoteLadder`` schema is defined in ``histdata.schemas``. Features are
implemented against that schema so they will compute correctly the moment TIER_2
data arrives, without requiring changes to this module.

Feature list:
* **spread_bps** — implied bid-ask spread from the first rung vs. mid-price.
* **depth_usd_at_size** — dollar depth available at a specified input size.
* **price_impact_pct_at_size** — price impact reading from the matching rung.
* **quote_ladder_age_seconds** — staleness of the ladder at simulation time.

Each is structured as a function that accepts a ``QuoteLadder | None`` and
returns ``float | None``, so the TIER_0 case is simply passing ``None`` as the
ladder.
"""

from __future__ import annotations

import math


# ``QuoteLadder`` and ``QuoteLadderRung`` are imported lazily inside the
# functions that need them. This avoids a hard dependency on ``histdata``
# when this module is used in contexts where that package is not yet wired up.
# The type annotations use string literals to avoid the circular import.


# ---------------------------------------------------------------------------
# Reason constants — descriptive strings attached to ``None`` results so
# downstream code can distinguish TIER_0 degradation from a genuine data error.
# ---------------------------------------------------------------------------

_REASON_NO_LADDER = "TIER_0: no quote ladder available at this fidelity level"
_REASON_NO_RUNGS = "quote ladder is empty — no rungs recorded"
_REASON_NO_MID = "mid-price is None — cannot compute spread"
_REASON_RUNG_NOT_FOUND = "no rung at or below requested size"


def spread_bps(
    ladder: object | None,
    mid_price_usd: float | None,
) -> tuple[float | None, str]:
    """Implied bid-ask spread in basis points, derived from the first rung.

    The spread proxy is ``2 * price_impact_pct * 100 bps / pct`` at the
    smallest rung size. Jupiter's ``priceImpactPct`` at the smallest rung is the
    best approximation of the half-spread we can extract from OHLCV + ladder
    data; real spread would require a live order book, which is TIER_3.

    Returns ``(value, reason)`` where ``reason`` is an empty string on success
    and a descriptive string on ``None``. This two-tuple pattern makes it
    possible for callers to log why a feature is absent without a separate query.

    When the ladder is ``None`` (TIER_0), returns ``(None, _REASON_NO_LADDER)``.
    """
    if ladder is None:
        return None, _REASON_NO_LADDER
    if mid_price_usd is None or not math.isfinite(mid_price_usd) or mid_price_usd <= 0.0:
        return None, _REASON_NO_MID

    # Access rungs through the schema-defined interface without hard-importing.
    rungs = getattr(ladder, "rungs", ())
    if not rungs:
        return None, _REASON_NO_RUNGS

    # Use the smallest rung as the spread proxy. ``QuoteLadderRung`` is
    # guaranteed to have ``price_impact_pct`` as a finite float by its schema.
    smallest = min(rungs, key=lambda r: r.in_amount_atomic)  # type: ignore[attr-defined]
    impact_pct = float(smallest.price_impact_pct)
    if not math.isfinite(impact_pct):
        return None, "price_impact_pct on smallest rung is non-finite"

    # Convert price impact percentage (e.g., 0.15 means 0.15%) to basis points.
    # 2× because price impact is one-sided (sell or buy), and bid-ask spread
    # is the round-trip cost — a proxy, not a measured spread.
    bps = 2.0 * impact_pct * 100.0
    return bps if math.isfinite(bps) else None, ""


def depth_usd_at_size(
    ladder: object | None,
    in_amount_atomic: int,
    price_usd_per_atomic: float | None,
) -> tuple[float | None, str]:
    """Dollar depth available at a specified input size.

    Returns the USD value of the best rung at or below ``in_amount_atomic``.
    ``price_usd_per_atomic`` converts atomic units to USD; it is separate from
    the ladder because the ladder records amounts without a price oracle.

    At TIER_0 (no ladder), returns ``(None, reason)`` rather than fabricating a
    depth estimate from bar data. A depth estimate derived from price alone would
    be optimistic and inconsistent with the ladder-based estimate at TIER_2.
    """
    if ladder is None:
        return None, _REASON_NO_LADDER
    if price_usd_per_atomic is None or not math.isfinite(price_usd_per_atomic):
        return None, "price_usd_per_atomic is None — cannot convert to USD"

    rung = getattr(ladder, "best_rung_for", lambda _: None)(in_amount_atomic)
    if rung is None:
        return None, _REASON_RUNG_NOT_FOUND

    depth_usd = float(rung.in_amount_atomic) * price_usd_per_atomic  # type: ignore[attr-defined]
    return depth_usd if math.isfinite(depth_usd) else None, ""


def price_impact_pct_at_size(
    ladder: object | None,
    in_amount_atomic: int,
) -> tuple[float | None, str]:
    """Price impact at a specific input size, from the matching rung.

    Uses the floor rung (largest rung whose ``in_amount_atomic`` is <=
    requested size). Interpolating between rungs is not done here — that is the
    execution model's responsibility with concrete rung pairs as inputs. This
    function returns what the AMM actually quoted, not a manufactured estimate.

    Returns ``(None, reason)`` at TIER_0 or when no rung covers the requested
    size.
    """
    if ladder is None:
        return None, _REASON_NO_LADDER

    rung = getattr(ladder, "best_rung_for", lambda _: None)(in_amount_atomic)
    if rung is None:
        return None, _REASON_RUNG_NOT_FOUND

    impact = float(rung.price_impact_pct)  # type: ignore[attr-defined]
    return impact if math.isfinite(impact) else None, ""


def quote_ladder_age_seconds(
    ladder: object | None,
    now: float,
) -> tuple[float | None, str]:
    """Staleness of the ladder at simulation time ``now``.

    ``ladder.available_time`` is when the replay is permitted to first use this
    ladder. ``now - available_time`` is how old it is at the point of use.
    A stale ladder is a real signal — it means the execution model would have
    to rely on older depth information — but it is not a disqualifier on its own.
    The execution model decides what age is acceptable.

    Returns ``(None, reason)`` at TIER_0.
    """
    if ladder is None:
        return None, _REASON_NO_LADDER

    available_time = getattr(ladder, "available_time", None)
    if available_time is None:
        return None, "ladder.available_time is None"
    age = now - float(available_time)
    return age if math.isfinite(age) else None, ""


__all__ = [
    "depth_usd_at_size",
    "price_impact_pct_at_size",
    "quote_ladder_age_seconds",
    "spread_bps",
]
