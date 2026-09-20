"""Immutable identifiers for every object in the decision-to-fill chain.

The audit's C11 and the `prompts._decision_line` finding share one root cause:
nothing in the old system had an identity. Fills were matched to actions by
symbol, so a stop-loss and a model SELL on the same coin in the same tick were
indistinguishable in the journal; and a crash between the trade append and the
state save left two files with no way to tell which rows belonged together.

Every record now carries an ID minted here. Two properties matter:

* **Ordering.** IDs are prefixed with a zero-padded microsecond timestamp, so
  lexical sort equals chronological sort. That makes a JSONL ledger sortable
  without parsing it.
* **Uniqueness without coordination.** A short random suffix means two
  processes that should never have been running at once still cannot silently
  produce colliding IDs — the duplicate shows up as two distinct rows rather
  than as one row overwriting another.

They are deliberately *not* content hashes. A content hash of an intent would
collide whenever the same order is legitimately placed twice, which is exactly
the case idempotency has to be able to distinguish. Content hashing is used
separately, by :func:`quote_fingerprint`, where equality *is* the question.
"""

from __future__ import annotations

import hashlib
import os
import time

__all__ = [
    "new_action_id",
    "new_decision_id",
    "new_fill_id",
    "new_intent_id",
    "new_order_id",
    "new_run_id",
    "quote_fingerprint",
]


def _mint(prefix: str) -> str:
    # Microseconds, not seconds: a slow tick can emit several intents inside one
    # second and they must still sort in the order they were created.
    micros = int(time.time() * 1_000_000)
    suffix = os.urandom(4).hex()
    return f"{prefix}-{micros:018d}-{suffix}"


def new_run_id() -> str:
    """One per process start. Every record made by that process carries it, so a
    duplicate-process incident is visible as two run IDs interleaved in one
    ledger rather than as inexplicable state."""
    return _mint("run")


def new_decision_id() -> str:
    """One per slow tick that reached a strategy."""
    return _mint("dec")


def new_action_id() -> str:
    """One per per-symbol action inside a decision."""
    return _mint("act")


def new_intent_id() -> str:
    """One per order the execution layer intends to place.

    This is the idempotency key. It is minted *before* anything is quoted or
    submitted and persisted before any side effect, so a crash mid-submission
    leaves a row that recovery can find and reconcile rather than an orphan.
    """
    return _mint("int")


def new_order_id() -> str:
    """One per quote-bound order derived from an intent."""
    return _mint("ord")


def new_fill_id() -> str:
    """One per settled (or failed) execution attempt."""
    return _mint("fil")


def quote_fingerprint(
    *,
    side: str,
    input_mint: str,
    output_mint: str,
    in_amount_atomic: int,
    out_amount_atomic: int,
    slot: int | None,
) -> str:
    """A content hash binding a quote to the exact swap it describes.

    C3 in the audit: risk could shrink an order after it was quoted, and the
    broker then executed the reduced notional against the original-size quote.
    Binding is the fix — an order carries the fingerprint of the quote it was
    built from, and the broker recomputes it from the quote it was handed. If
    they differ, the order is refused rather than filled at a price that was
    never offered for that size.

    Every economically load-bearing field of the swap is in the digest. The
    amounts are the atomic integers, not dollars, because dollars are a derived
    presentation of the swap and two different swaps can round to the same one.
    """
    payload = "|".join(
        (
            side,
            input_mint,
            output_mint,
            str(in_amount_atomic),
            str(out_amount_atomic),
            "none" if slot is None else str(slot),
        )
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()
