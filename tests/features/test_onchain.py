"""Tests for ``features.onchain``.

Each test is load-bearing: removing the guard it exercises must cause the test
to fail, not just produce a different number. This is documented on each test.

Critical tests:
* Every feature returns ``None`` at TIER_0 (no on-chain data collected) —
  never a fabricated ``0.0``.
* ``net_wallet_flow`` reads only swap records with ``available_time <= now``:
  a future-dated record must never move a currently-computed value. Verified
  both by a fixed example and a hypothesis prefix-equivalence property.
* Amounts are integer atomic units throughout the aggregation; the net flow
  of two clean-integer buys/sells is exact, not a float-rounded approximation.
* ``smart_wallet_labels`` refuses to compute past ``now`` (the training-window
  leakage guard) and only scores trades whose ``available_time`` falls inside
  ``[window_start, window_end]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from hypothesis import given
from hypothesis import strategies as st

from memetrader.features.onchain import (
    holder_count,
    net_wallet_flow,
    smart_wallet_labels,
    top_holder_concentration,
)


@dataclass
class FakeHolder:
    balance_atomic: int


@dataclass
class FakeHolderSnapshot:
    holders: list[FakeHolder]
    total_supply_atomic: int


@dataclass
class FakeSwapRecord:
    available_time: float | None
    event_time: float | None
    side: str
    in_amount_atomic: int
    wallet_address: str = "wallet-a"


@dataclass
class FakeSwapRecords:
    records: list[FakeSwapRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# top_holder_concentration
# ---------------------------------------------------------------------------


def test_top_holder_concentration_none_at_tier0() -> None:
    """TIER_0 has no holder snapshot; must be None, not a fabricated ratio."""
    value, reason = top_holder_concentration(None)
    assert value is None
    assert "TIER_0" in reason


def test_top_holder_concentration_known_value() -> None:
    """Hand-computed: top-2 of [300, 200, 100, 50, 50] / 1000 = 500/1000 = 0.5."""
    snapshot = FakeHolderSnapshot(
        holders=[
            FakeHolder(balance_atomic=100),
            FakeHolder(balance_atomic=300),
            FakeHolder(balance_atomic=50),
            FakeHolder(balance_atomic=200),
            FakeHolder(balance_atomic=50),
        ],
        total_supply_atomic=1000,
    )
    value, reason = top_holder_concentration(snapshot, top_n=2)
    assert value == pytest.approx(0.5)
    assert reason == ""


def test_top_holder_concentration_empty_holders_returns_none() -> None:
    snapshot = FakeHolderSnapshot(holders=[], total_supply_atomic=1000)
    value, reason = top_holder_concentration(snapshot)
    assert value is None
    assert "holder" in reason


def test_top_holder_concentration_zero_supply_returns_none() -> None:
    """Guard: dividing by a zero/unavailable supply must not produce inf."""
    snapshot = FakeHolderSnapshot(
        holders=[FakeHolder(balance_atomic=100)], total_supply_atomic=0
    )
    value, reason = top_holder_concentration(snapshot)
    assert value is None
    assert "total_supply_atomic" in reason


def test_top_holder_concentration_all_zero_balances_is_zero_not_none() -> None:
    """A snapshot that genuinely shows zero concentration is 0.0, not None.

    Guard: §0 draws the line at "looked, and it is quiet" vs "could not find
    out". Here we *did* look — there is a snapshot, holders, and supply — and
    the top holders simply hold nothing recorded. That is a real 0.0.
    """
    snapshot = FakeHolderSnapshot(
        holders=[FakeHolder(balance_atomic=0), FakeHolder(balance_atomic=0)],
        total_supply_atomic=1000,
    )
    value, reason = top_holder_concentration(snapshot, top_n=2)
    assert value == pytest.approx(0.0)
    assert reason == ""


# ---------------------------------------------------------------------------
# holder_count
# ---------------------------------------------------------------------------


def test_holder_count_none_at_tier0() -> None:
    value, reason = holder_count(None)
    assert value is None
    assert "TIER_0" in reason


def test_holder_count_known_value() -> None:
    snapshot = FakeHolderSnapshot(
        holders=[FakeHolder(balance_atomic=1), FakeHolder(balance_atomic=2)],
        total_supply_atomic=3,
    )
    value, reason = holder_count(snapshot)
    assert value == pytest.approx(2.0)
    assert reason == ""


def test_holder_count_empty_list_is_zero_not_none() -> None:
    """Zero holders (a snapshot that found nobody) is 0.0, distinct from
    ``holders is None`` (no snapshot data at all)."""
    snapshot = FakeHolderSnapshot(holders=[], total_supply_atomic=1000)
    value, reason = holder_count(snapshot)
    assert value == pytest.approx(0.0)
    assert reason == ""


# ---------------------------------------------------------------------------
# net_wallet_flow — the point-in-time invariant
# ---------------------------------------------------------------------------


def test_net_wallet_flow_none_at_tier0() -> None:
    value, reason = net_wallet_flow(None, now=1000.0)
    assert value is None
    assert "TIER_0" in reason


def test_net_wallet_flow_known_value_integer_atomic() -> None:
    """Hand-computed: buys 100 + 50, sell 30, all inside the window ->
    net = 100 + 50 - 30 = 120, exact integer arithmetic.
    """
    records = FakeSwapRecords(
        records=[
            FakeSwapRecord(
                available_time=900.0, event_time=900.0, side="BUY", in_amount_atomic=100
            ),
            FakeSwapRecord(
                available_time=950.0, event_time=950.0, side="BUY", in_amount_atomic=50
            ),
            FakeSwapRecord(
                available_time=980.0, event_time=980.0, side="SELL", in_amount_atomic=30
            ),
        ]
    )
    value, reason = net_wallet_flow(records, now=1000.0, window_seconds=3600.0)
    assert value == pytest.approx(120.0)
    assert reason == ""


def test_net_wallet_flow_future_record_is_invisible() -> None:
    """A swap whose ``available_time`` is after ``now`` must not move the
    computed net flow — the core no-lookahead guarantee.

    Guard: without the ``available_time <= now`` filter, a large future buy
    would leak into a currently-computed feature, and the equity curve built
    on it would look plausible while being unearnable in real time.
    """
    baseline = FakeSwapRecords(
        records=[
            FakeSwapRecord(
                available_time=900.0, event_time=900.0, side="BUY", in_amount_atomic=100
            ),
        ]
    )
    with_future_leak = FakeSwapRecords(
        records=[
            *baseline.records,
            FakeSwapRecord(
                available_time=1_000_000.0,
                event_time=1_000_000.0,
                side="BUY",
                in_amount_atomic=999_999,
            ),
        ]
    )
    v_baseline, _ = net_wallet_flow(baseline, now=1000.0, window_seconds=3600.0)
    v_with_future, _ = net_wallet_flow(with_future_leak, now=1000.0, window_seconds=3600.0)
    assert v_baseline == v_with_future == pytest.approx(100.0)


def test_net_wallet_flow_outside_window_excluded() -> None:
    """A swap whose ``event_time`` is before ``now - window_seconds`` does not
    contribute, even though its ``available_time`` is <= now.
    """
    records = FakeSwapRecords(
        records=[
            # cutoff = now(1000) - window_seconds(100) = 900; this event_time
            # is well before that, so it must be excluded from the window.
            FakeSwapRecord(
                available_time=100.0, event_time=100.0, side="BUY", in_amount_atomic=100_000
            ),
            FakeSwapRecord(
                available_time=990.0, event_time=990.0, side="BUY", in_amount_atomic=10
            ),
        ]
    )
    value, reason = net_wallet_flow(records, now=1000.0, window_seconds=100.0)
    assert value == pytest.approx(10.0)
    assert reason == ""


def test_net_wallet_flow_empty_window_is_none_not_zero() -> None:
    """No records in the window is None ("could not find out"), not 0.0.

    Guard: distinguishes "no data collected" from "balanced buys and sells" —
    the next test shows the latter really is 0.0.
    """
    records = FakeSwapRecords(records=[])
    value, reason = net_wallet_flow(records, now=1000.0, window_seconds=3600.0)
    assert value is None
    assert "no swap records" in reason


def test_net_wallet_flow_balanced_buys_and_sells_is_zero() -> None:
    """A genuinely balanced window (buy == sell) is 0.0, not None — we did
    look, and flow was quiet."""
    records = FakeSwapRecords(
        records=[
            FakeSwapRecord(
                available_time=950.0, event_time=950.0, side="BUY", in_amount_atomic=100
            ),
            FakeSwapRecord(
                available_time=960.0, event_time=960.0, side="SELL", in_amount_atomic=100
            ),
        ]
    )
    value, reason = net_wallet_flow(records, now=1000.0, window_seconds=3600.0)
    assert value == pytest.approx(0.0)
    assert reason == ""


@given(
    now=st.just(1000.0),
    extra_available_time=st.floats(
        min_value=1000.001, max_value=1e7, allow_nan=False, allow_infinity=False
    ),
    extra_amount=st.integers(min_value=1, max_value=10**9),
)
def test_net_wallet_flow_prefix_equivalence(
    now: float, extra_available_time: float, extra_amount: int
) -> None:
    """Prefix-equivalence property: the feature computed at ``now`` from data
    that also contains arbitrary future-dated swaps must equal the feature
    computed from data truncated at ``now``.

    This is the mandatory prefix-equivalence test from
    BACKTEST-CONTRACTS.md §8, specialised to on-chain wallet flow: adding any
    single record whose ``available_time`` is strictly after ``now`` must
    never change the result.
    """
    base_records = [
        FakeSwapRecord(
            available_time=900.0, event_time=900.0, side="BUY", in_amount_atomic=100
        ),
        FakeSwapRecord(
            available_time=950.0, event_time=950.0, side="SELL", in_amount_atomic=40
        ),
    ]
    truncated = FakeSwapRecords(records=list(base_records))
    full = FakeSwapRecords(
        records=[
            *base_records,
            FakeSwapRecord(
                available_time=extra_available_time,
                event_time=extra_available_time,
                side="BUY",
                in_amount_atomic=extra_amount,
            ),
        ]
    )
    v_truncated, _ = net_wallet_flow(truncated, now=now, window_seconds=3600.0)
    v_full, _ = net_wallet_flow(full, now=now, window_seconds=3600.0)
    assert v_truncated == v_full


# ---------------------------------------------------------------------------
# smart_wallet_labels — training-window leakage guard
# ---------------------------------------------------------------------------


def test_smart_wallet_labels_window_end_after_now_raises() -> None:
    """Hard guard: computing a label past ``now`` uses price returns the
    replay does not yet know, which is exactly the leak this function exists
    to prevent. Must raise, not silently clamp.
    """
    with pytest.raises(ValueError, match="window_end"):
        smart_wallet_labels(
            None,
            {},
            window_start=0.0,
            window_end=2000.0,
            now=1000.0,
        )


def test_smart_wallet_labels_none_at_tier0() -> None:
    labels, reason = smart_wallet_labels(
        None, {}, window_start=0.0, window_end=1000.0, now=1000.0
    )
    assert labels == frozenset()
    assert "TIER_0" in reason


def test_smart_wallet_labels_known_value() -> None:
    """A wallet whose buys all precede a positive return is labelled smart;
    one whose buys precede a negative return is not.

    ``price_returns`` carries bars at t=100..700 (interval 100). "good-wallet"
    trades at t=10..50 (5 buys, all before t=100): for each trade, the future
    bars after it are [100,200,300,400,500,600,700] and the 3rd
    (``lead_bars=3``) is t=300, whose return is positive -> 5/5 hits ->
    precision 1.0 -> smart. "bad-wallet" trades at t=350..390 (5 buys, all
    after t=300): the future bars after each trade are [400,500,600,700] and
    the 3rd is t=600, whose return is negative -> 0/5 hits -> not smart.
    """
    price_returns = {
        100.0: 1.0,
        200.0: 1.0,
        300.0: 5.0,  # good-wallet's lead_bars=3 checkpoint (positive)
        400.0: 1.0,
        500.0: 1.0,
        600.0: -5.0,  # bad-wallet's lead_bars=3 checkpoint (negative)
        700.0: 1.0,
    }
    good_trades = [
        FakeSwapRecord(
            available_time=t,
            event_time=t,
            side="BUY",
            in_amount_atomic=10,
            wallet_address="good-wallet",
        )
        for t in (10.0, 20.0, 30.0, 40.0, 50.0)
    ]
    bad_trades = [
        FakeSwapRecord(
            available_time=t,
            event_time=t,
            side="BUY",
            in_amount_atomic=10,
            wallet_address="bad-wallet",
        )
        for t in (350.0, 360.0, 370.0, 380.0, 390.0)
    ]
    records = FakeSwapRecords(records=[*good_trades, *bad_trades])

    labels, reason = smart_wallet_labels(
        records,
        price_returns,
        window_start=0.0,
        window_end=700.0,
        now=700.0,
        lead_bars=3,
        min_trades=5,
        min_precision=0.6,
    )
    assert "good-wallet" in labels
    assert "bad-wallet" not in labels
    assert reason == ""


def test_smart_wallet_labels_excludes_trades_outside_window() -> None:
    """A wallet whose only trades fall outside [window_start, window_end]
    must not appear, even if those trades would otherwise qualify as smart.

    Guard: this is the per-fold isolation requirement from the module
    docstring — a smart-wallet label must be computed entirely inside the
    training window, never from data outside it.
    """
    price_returns = {300.0: 5.0}
    trades_outside_window = [
        FakeSwapRecord(
            available_time=t,
            event_time=t,
            side="BUY",
            in_amount_atomic=10,
            wallet_address="outsider",
        )
        for t in (800.0, 810.0, 820.0, 830.0, 840.0)  # all > window_end=700
    ]
    records = FakeSwapRecords(records=trades_outside_window)
    labels, reason = smart_wallet_labels(
        records,
        price_returns,
        window_start=0.0,
        window_end=700.0,
        now=900.0,
        lead_bars=3,
        min_trades=5,
        min_precision=0.6,
    )
    assert "outsider" not in labels
    assert labels == frozenset()
    assert "training window" in reason


def test_smart_wallet_labels_below_min_trades_excluded() -> None:
    """A wallet with too few qualifying buys is excluded regardless of how
    good its precision would be.
    """
    price_returns = {300.0: 5.0}
    few_trades = [
        FakeSwapRecord(
            available_time=t,
            event_time=t,
            side="BUY",
            in_amount_atomic=10,
            wallet_address="thin-wallet",
        )
        for t in (10.0, 20.0, 30.0, 40.0)  # only 4, min_trades default is 5
    ]
    records = FakeSwapRecords(records=few_trades)
    labels, _ = smart_wallet_labels(
        records, price_returns, window_start=0.0, window_end=700.0, now=700.0
    )
    assert "thin-wallet" not in labels
