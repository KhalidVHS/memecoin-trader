"""Tests for ``features.attention``.

Each test is load-bearing: removing the guard it exercises must cause the test
to fail, not just produce a different number. This is documented on each test.

Critical tests:
* Availability is collector *receipt* time (``available_time``), not post
  *creation* time — a post created before ``now`` but collected after ``now``
  must be invisible (BACKTEST-CONTRACTS.md §1: "Social data is available at
  collector receipt time, not post creation time").
* Sentiment is disabled by default (``[sentiment] enabled = false``); every
  function must return ``None`` regardless of what the record contains.
* A record with the requested field set to ``None`` degrades to ``None``,
  never to ``0.0``.

Bug found and fixed as part of writing these tests: ``unique_contributors_24h``
did not guard against a non-finite value before ``int(value)``, unlike every
sibling function in this module. ``int(float("nan"))`` raises ``ValueError``
and ``int(float("inf"))`` raises ``OverflowError`` — a single malformed record
would crash the feature pipeline instead of degrading to ``None``. See
``test_unique_contributors_24h_nonfinite_value_does_not_crash``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from hypothesis import given
from hypothesis import strategies as st

from memetrader.features import attention


@dataclass
class FakeSocialRecord:
    """Minimal stand-in for ``histdata.schemas.SocialEventRecord``.

    Carries only the attributes these functions read via ``getattr``.
    ``available_time`` is collector receipt time — the field every function in
    this module checks before using anything else on the record.
    """

    available_time: float | None
    mention_velocity_1h: float | None = None
    mention_velocity_24h: float | None = None
    mention_zscore_7d: float | None = None
    unique_contributors_24h: float | None = None
    contributor_to_post_ratio: float | None = None


# (function, attribute it reads, a representative in-range value)
ALL_FUNCS = [
    (attention.mention_velocity_1h, "mention_velocity_1h", 12.5),
    (attention.mention_velocity_24h, "mention_velocity_24h", 3.25),
    (attention.mention_zscore_7d, "mention_zscore_7d", 2.1),
    (attention.unique_contributors_24h, "unique_contributors_24h", 40.0),
    (attention.contributor_to_post_ratio, "contributor_to_post_ratio", 0.8),
]


# ---------------------------------------------------------------------------
# Sentiment disabled (default / current live state)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func,attr,value", ALL_FUNCS)
def test_disabled_by_default_returns_none(func, attr, value) -> None:
    """Sentiment disabled forces every attention feature to None regardless of
    record contents.

    Guard: without this check running first, a record with plausible-looking
    data would still produce a value even though ``[sentiment] enabled =
    false`` — reporting a feature that was never actually computed by a live
    collector.
    """
    record = FakeSocialRecord(available_time=100.0, **{attr: value})
    result, reason = func(record, now=100.0, sentiment_enabled=False)
    assert result is None
    assert "disabled" in reason


@pytest.mark.parametrize("func,attr,value", ALL_FUNCS)
def test_disabled_ignores_now_and_record(func, attr, value) -> None:
    """Disabled state short-circuits before any time check runs.

    Guard: if the disabled check were moved after the point-in-time check, a
    record with a bad/future ``available_time`` would surface a *different*
    (point-in-time) reason instead of the disabled reason, silently implying
    sentiment collection is active.
    """
    record = FakeSocialRecord(available_time=None, **{attr: value})
    result, reason = func(record, now=100.0, sentiment_enabled=False)
    assert result is None
    assert "disabled" in reason


# ---------------------------------------------------------------------------
# No record
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func,attr,value", ALL_FUNCS)
def test_no_record_returns_none(func, attr, value) -> None:
    """A missing record (TIER_0 / no collector data yet) is None, not 0."""
    result, reason = func(None, now=100.0, sentiment_enabled=True)
    assert result is None
    assert "no social event record" in reason


# ---------------------------------------------------------------------------
# The point-in-time invariant: receipt time, not creation time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func,attr,value", ALL_FUNCS)
def test_receipt_time_after_now_is_invisible(func, attr, value) -> None:
    """A post created before ``now`` but *collected* after ``now`` must be
    invisible to a replay standing at ``now``.

    This is the module-specific leak BACKTEST-CONTRACTS.md §1 calls out by
    name: "Social data is available at collector receipt time, not post
    creation time." The record below models a post that existed (was created)
    well before the replay's current time, but whose collector receipt
    (``available_time``) has not happened yet from the replay's point of view.

    Guard: without the ``available_time > now`` check, this record would be
    treated as already-known data and leak future information into the
    replay — three minutes of foresight is enough to erase a 0.2% strategy
    edge, and here the leak is 50 (simulated) seconds.
    """
    post_created_at = 100.0
    collector_receipt_time = 250.0
    replay_now = 200.0  # after creation, strictly before receipt
    assert post_created_at < replay_now < collector_receipt_time

    record = FakeSocialRecord(available_time=collector_receipt_time, **{attr: value})
    result, reason = func(record, now=replay_now, sentiment_enabled=True)
    assert result is None, (
        f"{func.__name__} leaked a record collected in the future "
        f"(available_time={collector_receipt_time}) to a replay at now={replay_now}"
    )
    assert "point-in-time violation" in reason


@pytest.mark.parametrize("func,attr,value", ALL_FUNCS)
def test_visible_once_receipt_time_reached(func, attr, value) -> None:
    """The same record becomes visible exactly when ``now`` reaches
    ``available_time`` (boundary is inclusive).

    Guard: proves the previous test's ``None`` is genuinely about timing and
    not some unrelated defect that makes the function always return ``None``.
    """
    record = FakeSocialRecord(available_time=250.0, **{attr: value})
    result, reason = func(record, now=250.0, sentiment_enabled=True)
    assert result is not None
    assert reason == ""


@given(
    now=st.floats(min_value=0, max_value=1e9, allow_nan=False, allow_infinity=False),
    delay=st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_available_time_strictly_after_now_always_invisible(
    now: float, delay: float
) -> None:
    """Property: for ANY ``now`` and ANY strictly-future ``available_time``,
    the feature is invisible.

    Generalises the fixed receipt-time-leak test above across the full float
    range rather than one hand-picked pair of timestamps — a fixed example
    could pass by coincidence of the chosen numbers; this cannot.
    """
    record = FakeSocialRecord(available_time=now + delay, mention_velocity_1h=99.0)
    result, reason = attention.mention_velocity_1h(record, now=now, sentiment_enabled=True)
    assert result is None
    assert "point-in-time violation" in reason


# ---------------------------------------------------------------------------
# Missing field on an otherwise-visible record: None, never 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func,attr,value", ALL_FUNCS)
def test_none_field_on_record_returns_none(func, attr, value) -> None:
    """A visible record whose specific field is unset is None, never 0.0.

    Guard: §0 is explicit — None means "could not find out", 0 means "looked,
    and it is quiet". A collector that has not populated this field yet must
    not be reported as a confident zero reading.
    """
    record = FakeSocialRecord(available_time=100.0)  # attr left at default None
    result, reason = func(record, now=100.0, sentiment_enabled=True)
    assert result is None
    assert "is None on the record" in reason


# ---------------------------------------------------------------------------
# Known-value tests (hand-computed, not re-deriving the implementation)
# ---------------------------------------------------------------------------


def test_mention_velocity_1h_known_value() -> None:
    record = FakeSocialRecord(available_time=100.0, mention_velocity_1h=17.5)
    value, reason = attention.mention_velocity_1h(record, now=100.0, sentiment_enabled=True)
    assert value == pytest.approx(17.5)
    assert reason == ""


def test_mention_velocity_24h_known_value() -> None:
    record = FakeSocialRecord(available_time=100.0, mention_velocity_24h=3.25)
    value, reason = attention.mention_velocity_24h(
        record, now=100.0, sentiment_enabled=True
    )
    assert value == pytest.approx(3.25)
    assert reason == ""


def test_mention_zscore_7d_known_value() -> None:
    record = FakeSocialRecord(available_time=100.0, mention_zscore_7d=-2.4)
    value, reason = attention.mention_zscore_7d(record, now=100.0, sentiment_enabled=True)
    assert value == pytest.approx(-2.4)
    assert reason == ""


def test_unique_contributors_24h_known_value_casts_to_float() -> None:
    """An integer count on the record is returned as a float, unmodified."""
    record = FakeSocialRecord(available_time=100.0, unique_contributors_24h=7)
    value, reason = attention.unique_contributors_24h(
        record, now=100.0, sentiment_enabled=True
    )
    assert value == pytest.approx(7.0)
    assert isinstance(value, float)
    assert reason == ""


def test_contributor_to_post_ratio_known_value() -> None:
    record = FakeSocialRecord(available_time=100.0, contributor_to_post_ratio=0.8)
    value, reason = attention.contributor_to_post_ratio(
        record, now=100.0, sentiment_enabled=True
    )
    assert value == pytest.approx(0.8)
    assert reason == ""


# ---------------------------------------------------------------------------
# Non-finite values must degrade to None, never crash or leak NaN/inf
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_mention_velocity_1h_nonfinite_returns_none(bad_value: float) -> None:
    record = FakeSocialRecord(available_time=100.0, mention_velocity_1h=bad_value)
    value, reason = attention.mention_velocity_1h(record, now=100.0, sentiment_enabled=True)
    assert value is None
    assert "non-finite" in reason


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_unique_contributors_24h_nonfinite_value_does_not_crash(bad_value: float) -> None:
    """Non-finite ``unique_contributors_24h`` must degrade to None, not raise.

    BUG (found and fixed as part of this test): unlike every sibling function
    in this module, ``unique_contributors_24h`` cast straight to ``int(value)``
    without a ``math.isfinite`` guard first. ``int(float("nan"))`` raises
    ``ValueError`` and ``int(float("inf"))`` raises ``OverflowError``, so one
    malformed record would crash the whole feature computation instead of
    degrading to ``None`` the way every other function in this module does.
    Fixed in ``src/memetrader/features/attention.py::unique_contributors_24h``
    by adding the same ``math.isfinite`` check the other four functions
    already had.
    """
    record = FakeSocialRecord(available_time=100.0, unique_contributors_24h=bad_value)
    value, reason = attention.unique_contributors_24h(
        record, now=100.0, sentiment_enabled=True
    )
    assert value is None
    assert "non-finite" in reason


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_record_same_now_is_deterministic() -> None:
    """Calling the same function twice with the same inputs must be
    bit-identical — no hidden global or time-of-call dependence.
    """
    record = FakeSocialRecord(available_time=100.0, mention_velocity_1h=17.5)
    first = attention.mention_velocity_1h(record, now=100.0, sentiment_enabled=True)
    second = attention.mention_velocity_1h(record, now=100.0, sentiment_enabled=True)
    assert first == second
