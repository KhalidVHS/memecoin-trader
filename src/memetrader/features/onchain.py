"""On-chain features: holder concentration and wallet flow.

These features require decoded swap history (TIER_1) or wallet-level analytics
(TIER_2 and above). At TIER_0 — OHLCV only — no on-chain data exists and every
feature here returns ``None`` with a recorded reason.

Two design choices in this module require explicit documentation:

**1. Smart wallet labels must be computed inside each training window.**

A "smart wallet" is a wallet whose historical trades preceded price moves. That
label is computed from a walk-forward window's training data. Computing it
globally — once across all history — is a leakage in the same category as
fitting a scaler on the full series: the label for a wallet in the training
period is computed using information the wallet's future trades provide.

The ``smart_wallet_labels`` function enforces this with a hard check: it
requires an explicit ``window_start`` and ``window_end`` that must bracket the
training period. Passing ``window_end`` beyond the current simulated time is
an error that raises rather than silently computing a leaked label. Callers
must supply the window from the fold's training slice.

**2. Holder concentration features are computed over the wallets observed
during the training window, never globally.**

A wallet that held tokens only during the test period would be invisible during
training, and including it in a "top-10 holder" metric computed over all
history would give the training model information about future holders.

These constraints are documented in the function signatures and enforced where
possible. Where enforcement would require knowledge the function does not have
(e.g., whether a caller's ``window_end`` is after ``state.now``), the
constraint is stated as a precondition and the function trusts the caller.

At TIER_0, neither concern is live: there is no data to leak. The constraints
are coded now so that when TIER_1 data arrives, the leakage guard is already in
place and tested rather than being bolted on afterwards.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Reason constants
# ---------------------------------------------------------------------------

_REASON_TIER0 = "TIER_0: no on-chain swap history available at this fidelity level"
_REASON_NO_HOLDERS = "holder snapshot not available"
_REASON_WINDOW = "smart wallet labels must be computed inside a training window"


# ---------------------------------------------------------------------------
# Holder concentration
# ---------------------------------------------------------------------------


def top_holder_concentration(
    holder_snapshot: object | None,
    *,
    top_n: int = 10,
) -> tuple[float | None, str]:
    """Fraction of supply held by the top ``top_n`` wallets.

    At TIER_0, ``holder_snapshot`` is always ``None`` and this returns
    ``(None, reason)``. The feature is defined against the schema so it
    computes correctly when TIER_1 data arrives.

    A high concentration is a rug-pull indicator: if the top 10 wallets
    control 80%+ of supply, a coordinated sell unwinds price regardless of
    any technical signal. This feature is explicitly one-sided — high
    concentration is bad, low concentration is baseline.

    The holder snapshot must be computed as of the training window's end, not
    globally. Passing a snapshot that includes future holders is a subtle
    survivorship-bias in reverse: it makes a post-rug token look concentrated
    because the surviving holders post-rug are a subset of the original set.
    """
    if holder_snapshot is None:
        return None, _REASON_TIER0

    # Access the holder list through the schema-defined interface.
    holders = getattr(holder_snapshot, "holders", None)
    if holders is None or len(holders) == 0:
        return None, _REASON_NO_HOLDERS

    total_supply = getattr(holder_snapshot, "total_supply_atomic", None)
    if total_supply is None or total_supply <= 0:
        return None, "total_supply_atomic is zero or unavailable"

    # Sort by balance descending and take top_n.
    sorted_holders = sorted(
        holders,
        key=lambda h: getattr(h, "balance_atomic", 0),
        reverse=True,
    )
    top = sorted_holders[:top_n]
    top_balance = sum(getattr(h, "balance_atomic", 0) for h in top)
    concentration = float(top_balance) / float(total_supply)
    return concentration if math.isfinite(concentration) else None, ""


def holder_count(holder_snapshot: object | None) -> tuple[float | None, str]:
    """Number of distinct wallets holding the token.

    Returns ``None`` at TIER_0. Holder count is a liquidity proxy — more holders
    means more distributed supply, which typically means more depth. But it can
    be gamed (airdrop farming produces thousands of wallets with dust balances),
    so it must be used alongside concentration metrics.
    """
    if holder_snapshot is None:
        return None, _REASON_TIER0
    holders = getattr(holder_snapshot, "holders", None)
    if holders is None:
        return None, _REASON_NO_HOLDERS
    count = len(holders)
    return float(count), ""


# ---------------------------------------------------------------------------
# Wallet flow
# ---------------------------------------------------------------------------


def net_wallet_flow(
    swap_records: object | None,
    *,
    window_seconds: float = 3600.0,
    now: float,
) -> tuple[float | None, str]:
    """Net token inflow (buys - sells) in atomic units over ``window_seconds``.

    Requires TIER_1 (decoded swap history). Returns ``None`` at TIER_0.

    ``None`` means "no data" — a completely absent flow record is not "zero net
    flow". A coin that had no swap data collected genuinely cannot be
    distinguished from a coin that traded with perfectly balanced buys and sells.
    Reporting ``0.0`` would manufacture a neutral reading from nothing.

    ``window_seconds`` is measured backward from ``now``. Only swaps with
    ``available_time <= now`` are included — the point-in-time invariant applies
    to every record this function touches.
    """
    if swap_records is None:
        return None, _REASON_TIER0

    records = getattr(swap_records, "records", None)
    if records is None:
        return None, "swap_records.records is None"

    cutoff = now - window_seconds
    net: float = 0.0
    count = 0
    for rec in records:
        available_time = getattr(rec, "available_time", None)
        if available_time is None or float(available_time) > now:
            continue
        event_time = getattr(rec, "event_time", None)
        if event_time is None or float(event_time) < cutoff:
            continue
        side = getattr(rec, "side", None)
        amount = getattr(rec, "in_amount_atomic", 0)
        if side == "BUY":
            net += float(amount)
            count += 1
        elif side == "SELL":
            net -= float(amount)
            count += 1

    if count == 0:
        return None, "no swap records in the requested window"
    return net if math.isfinite(net) else None, ""


# ---------------------------------------------------------------------------
# Smart wallet labels
# ---------------------------------------------------------------------------


def smart_wallet_labels(
    swap_records: object | None,
    price_returns: dict[float, float],
    *,
    window_start: float,
    window_end: float,
    now: float,
    lead_bars: int = 3,
    min_trades: int = 5,
    min_precision: float = 0.6,
) -> tuple[frozenset[str], str]:
    """Wallets whose trades preceded price moves within the training window.

    **Hard constraint: ``window_end`` must be <= ``now``.**

    A smart wallet label must be computed from data entirely within the training
    window. Using future price returns to label wallets as "smart" leaks those
    returns into the training features. The check below enforces this: if
    ``window_end > now``, the function raises ``ValueError`` rather than
    silently computing a label against future data.

    ``window_start`` and ``window_end`` define the training window. Only swaps
    with ``available_time`` in ``[window_start, window_end]`` are considered.

    ``price_returns`` maps timestamp → return. A wallet is labelled "smart" if
    at least ``min_precision`` fraction of its trades within the window were
    followed by a positive return within ``lead_bars`` bars.

    Returns ``(frozenset of wallet addresses, reason)``. An empty frozenset
    with a reason string indicates why no wallets were labelled.

    At TIER_0, returns ``(frozenset(), reason)`` because no swap data exists.
    """
    if window_end > now:
        raise ValueError(
            f"smart_wallet_labels: window_end {window_end} > now {now}. "
            "Labels must be computed inside the training window — using future "
            "price returns to label wallets as 'smart' leaks those returns. "
            "Pass window_end <= now, where now is the simulated time."
        )

    if swap_records is None:
        return frozenset(), _REASON_TIER0

    records = getattr(swap_records, "records", None)
    if records is None:
        return frozenset(), "swap_records.records is None"

    # Group trades by wallet within the window.
    wallet_trades: dict[str, list[tuple[float, str]]] = {}
    for rec in records:
        available_time = getattr(rec, "available_time", None)
        if available_time is None:
            continue
        at = float(available_time)
        if at < window_start or at > window_end:
            continue
        wallet = getattr(rec, "wallet_address", None)
        side = getattr(rec, "side", None)
        if wallet is None or side is None:
            continue
        if wallet not in wallet_trades:
            wallet_trades[wallet] = []
        wallet_trades[wallet].append((at, side))

    if not wallet_trades:
        return frozenset(), "no swap records in the training window"

    # Score each wallet.
    sorted_returns = sorted(price_returns.items())
    return_times = [t for t, _ in sorted_returns]
    return_vals = dict(sorted_returns)

    smart: set[str] = set()
    for wallet, trades in wallet_trades.items():
        buys = [(ts, s) for ts, s in trades if s == "BUY"]
        if len(buys) < min_trades:
            continue
        hits = 0
        for trade_ts, _ in buys:
            # Find the return ``lead_bars`` timestamps after the trade.
            future_times = [t for t in return_times if t > trade_ts]
            if len(future_times) < lead_bars:
                continue
            lead_ts = future_times[lead_bars - 1]
            ret = return_vals.get(lead_ts)
            if ret is not None and ret > 0.0:
                hits += 1
        precision = hits / len(buys)
        if precision >= min_precision:
            smart.add(wallet)

    return frozenset(smart), ""


__all__ = [
    "holder_count",
    "net_wallet_flow",
    "smart_wallet_labels",
    "top_holder_concentration",
]
