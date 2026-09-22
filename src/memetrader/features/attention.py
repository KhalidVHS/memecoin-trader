"""Social and attention features.

Two constraints govern every feature in this module:

**1. Available time is collector receipt time, not post creation time.**

A post created at 10:00 and collected at 10:03 was not in the dataset at 10:00.
A replay that treats it as available at 10:00 grants three minutes of foresight
— smaller than it sounds only until you realise the strategy's threshold is
0.2%. The ``available_time`` field on ``SocialEventRecord`` is set to receipt
time, not event time, precisely because of this; these functions check it before
using any record.

This is not a theoretical concern. Arctic Shift's index runs behind live Reddit
by a variable amount; the most recent window is structurally incomplete. The
``observed_through`` field on ``SocialEventRecord`` records the latest post
timestamp that is fully indexed, so a replay can know whether the window it is
reading is complete or trailing.

**2. Sentiment is currently disabled: ``[sentiment] enabled = false``.**

No sentiment data has been collected. Every function in this module returns
``None`` with a recorded reason. The code is implemented against the
``SocialEventRecord`` schema so it computes correctly when collection is
enabled, but the degradation path is the live path right now.

The ``sentiment_enabled`` parameter on each function defaults to ``False``
matching the config state. Passing ``True`` with ``record=None`` is a valid
TIER_0 state (collection enabled but no data yet); passing ``True`` with a
record activates the computation.
"""

from __future__ import annotations

import math

# ``SocialEventRecord`` is referenced in type comments only — the schema is
# defined in ``histdata.schemas`` which is being written concurrently. These
# functions receive ``object | None`` so they compile without a hard import.

_REASON_DISABLED = "sentiment is disabled ([sentiment] enabled = false)"
_REASON_NO_RECORD = "no social event record available at this time"
_REASON_STALE = "record is stale — observed_through is before the current window"


def mention_velocity_1h(
    record: object | None,
    *,
    now: float,
    sentiment_enabled: bool = False,
) -> tuple[float | None, str]:
    """Mentions per hour computed over the trailing 1h window.

    Uses ``SocialEventRecord.mention_velocity_1h`` directly — the computation
    is done by the collector and stored; this function validates the record is
    within the time constraints and extracts the value.

    Returns ``(None, reason)`` when:
    * Sentiment is disabled (default state).
    * No record is available.
    * The record's ``available_time > now`` (point-in-time violation — this
      should never happen if the ``PointInTimeState`` is correct, but the check
      is here as a defence-in-depth guard that fails loudly rather than silently
      using future data).
    * The record's ``observed_through`` indicates the collection window is
      incomplete and the available data should not be treated as confident.
    """
    if not sentiment_enabled:
        return None, _REASON_DISABLED
    if record is None:
        return None, _REASON_NO_RECORD

    available_time = getattr(record, "available_time", None)
    if available_time is None or float(available_time) > now:
        # A record not yet available at ``now`` is a point-in-time violation.
        # The PointInTimeState protocol should have filtered this out; this
        # guard catches the case where a feature function is called with
        # raw records bypassing the replay state.
        return None, (
            f"record available_time {available_time} > now {now}: "
            "point-in-time violation — feature uses receipt time, not event time"
        )

    value = getattr(record, "mention_velocity_1h", None)
    if value is None:
        return None, "mention_velocity_1h is None on the record"
    v = float(value)
    if not math.isfinite(v):
        return None, "mention_velocity_1h is non-finite"
    return v, ""


def mention_velocity_24h(
    record: object | None,
    *,
    now: float,
    sentiment_enabled: bool = False,
) -> tuple[float | None, str]:
    """Mentions per hour averaged over the trailing 24h window.

    Same time-constraint rules as ``mention_velocity_1h``. The 24h window
    is less susceptible to the "incomplete most-recent-hour" problem because
    the missing hour is a smaller fraction of the total window, but it is still
    subject to the collector receipt time constraint.
    """
    if not sentiment_enabled:
        return None, _REASON_DISABLED
    if record is None:
        return None, _REASON_NO_RECORD

    available_time = getattr(record, "available_time", None)
    if available_time is None or float(available_time) > now:
        return None, (
            f"record available_time {available_time} > now {now}: "
            "point-in-time violation"
        )

    value = getattr(record, "mention_velocity_24h", None)
    if value is None:
        return None, "mention_velocity_24h is None on the record"
    v = float(value)
    if not math.isfinite(v):
        return None, "mention_velocity_24h is non-finite"
    return v, ""


def mention_zscore_7d(
    record: object | None,
    *,
    now: float,
    sentiment_enabled: bool = False,
) -> tuple[float | None, str]:
    """Z-score of current mention velocity against the 7-day non-overlapping baseline.

    The baseline must be computed over strictly non-overlapping prior buckets
    (see ``SocialEventRecord.mention_zscore_7d`` docstring). Including the
    current window in its own baseline biases the z-score toward zero —
    shrinking the very anomaly the z-score exists to detect — so the score
    stored on the record was computed excluding the current window.

    Returns ``None`` when sentiment is disabled, no record, or the record does
    not carry a pre-computed z-score (the collector handles the baseline window
    calculation, not this function).
    """
    if not sentiment_enabled:
        return None, _REASON_DISABLED
    if record is None:
        return None, _REASON_NO_RECORD

    available_time = getattr(record, "available_time", None)
    if available_time is None or float(available_time) > now:
        return None, (
            f"record available_time {available_time} > now {now}: "
            "point-in-time violation"
        )

    value = getattr(record, "mention_zscore_7d", None)
    if value is None:
        return None, "mention_zscore_7d is None on the record"
    v = float(value)
    if not math.isfinite(v):
        return None, "mention_zscore_7d is non-finite"
    return v, ""


def unique_contributors_24h(
    record: object | None,
    *,
    now: float,
    sentiment_enabled: bool = False,
) -> tuple[float | None, str]:
    """Number of distinct accounts that posted in the trailing 24h.

    High unique contributor count relative to post count suggests organic
    activity; low ratio (few accounts, many posts) is a bot-post pattern.
    Returns the raw count here; the ratio is ``contributor_to_post_ratio``.

    Returns an ``int`` value cast to ``float`` for consistency with the other
    feature functions.
    """
    if not sentiment_enabled:
        return None, _REASON_DISABLED
    if record is None:
        return None, _REASON_NO_RECORD

    available_time = getattr(record, "available_time", None)
    if available_time is None or float(available_time) > now:
        return None, (
            f"record available_time {available_time} > now {now}: "
            "point-in-time violation"
        )

    value = getattr(record, "unique_contributors_24h", None)
    if value is None:
        return None, "unique_contributors_24h is None on the record"
    count = int(value)
    return float(count), ""


def contributor_to_post_ratio(
    record: object | None,
    *,
    now: float,
    sentiment_enabled: bool = False,
) -> tuple[float | None, str]:
    """Unique contributors divided by total posts in 24h.

    Values near 1.0 mean every poster posted once (organic spread); values near
    0 mean a few accounts posted many times (coordinated or bot activity). The
    ratio is bounded above by 1.0 — more contributors than posts is impossible
    — but not bounded below, and very small values are the concerning ones.
    """
    if not sentiment_enabled:
        return None, _REASON_DISABLED
    if record is None:
        return None, _REASON_NO_RECORD

    available_time = getattr(record, "available_time", None)
    if available_time is None or float(available_time) > now:
        return None, (
            f"record available_time {available_time} > now {now}: "
            "point-in-time violation"
        )

    value = getattr(record, "contributor_to_post_ratio", None)
    if value is None:
        return None, "contributor_to_post_ratio is None on the record"
    v = float(value)
    if not math.isfinite(v):
        return None, "contributor_to_post_ratio is non-finite"
    return v, ""


__all__ = [
    "contributor_to_post_ratio",
    "mention_velocity_1h",
    "mention_velocity_24h",
    "mention_zscore_7d",
    "unique_contributors_24h",
]
