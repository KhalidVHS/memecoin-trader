"""AMM swap mathematics for the backtest execution layer.

Three modules:

* ``constant_product`` — Uniswap-v2 / Raydium AMM / pump.fun bonding-curve
  swap math in integer atomic units.
* ``concentrated_liquidity`` — CLMM (Orca Whirlpools / Raydium CLMM) tick-
  indexed swap math derived from published formulas.
* ``route_replay`` — multi-hop route reconstruction from historical pool
  states, producing ``Quote``-shaped results with explicit provenance labels.

None of these modules modify live code (``types.py``, ``quotes.py``,
``broker.py``); they import from it.
"""

from __future__ import annotations

from .concentrated_liquidity import swap as clmm_swap
from .constant_product import amount_in_for_out, amount_out, price_impact_pct, spot_price
from .route_replay import RouteProvenance, RouteResult, replay_route

__all__ = [
    "RouteProvenance",
    "RouteResult",
    "amount_in_for_out",
    "amount_out",
    "clmm_swap",
    "price_impact_pct",
    "replay_route",
    "spot_price",
]
