"""Latency model for the backtest execution engine.

The invariant this module enforces: **a decision generated from a bar close
must not fill at that close.** The clock must advance past the bar boundary
before the order meets a market state. This is enforced in two ways:

1. ``total_seconds`` has a structural minimum: even the fastest configured
   distribution cannot return zero. The default minimum is ``_MIN_LATENCY_S``
   (currently 0.1 s — one tick on a 100ms slot chain). Returning zero is only
   possible by explicitly passing a zero floor, which requires deliberate action.

2. ``ready_at(submitted_at)`` always returns ``submitted_at + total_seconds``,
   where ``total_seconds > 0``. The ``OrderReceipt.ready_at`` invariant in
   ``types.py`` then ensures ``ready_at > submitted_at``, which the engine's
   event queue reads before allowing settlement.

Stage decomposition
-------------------
The pipeline has six stages, each drawing from its own distribution:

``quote_request_s``    Time from intent to dispatching the quote request.
``quote_response_s``   Time for the venue to respond with a route and price.
``risk_confirm_s``     Time for the risk layer to re-evaluate the quote.
``signing_s``          Time to sign and serialise the transaction.
``submission_s``       Time to submit to the RPC endpoint.
``landing_s``          Time for the validator network to include the tx.

Total wall-clock latency is the sum. Each stage draws independently from a
configurable distribution. The RNG is always explicit and seeded — never the
global ``random`` module — so the same seed produces identical draws regardless
of what other code ran before.

Landing probability vs. landing delay
--------------------------------------
A transaction that lands late is a different failure mode from one that never
lands. The strategy's P&L differs:

* A **dropped transaction** (``landing_probability < 1``) means the order is
  terminal-FAILED with gas charged. The strategy must re-quote.
* A **late-landing transaction** (``landing_delay_s`` > expected) means the
  order lands but at a market state that has moved. The fill is LANDED but the
  price may be different from what was expected.

These are separate quantities on purpose. Mixing them would mean either
understating the failure cost (dropped becomes delayed) or overstating it
(delayed becomes dropped). The caller (the venue simulator) receives both
quantities and decides independently.

p50 / p90 / p99 presets
------------------------
Three presets calibrate the robustness suite. The numbers are documented
assumptions, not measurements; the module docstring says so. A TIER_3
calibrated model would replace them with observed distributions.

* ``PRESET_P50``  — median conditions.
* ``PRESET_P90``  — 90th-percentile congestion.
* ``PRESET_P99``  — tail congestion (should still produce occasional fills).

Each preset is a :class:`LatencyConfig` and can be passed directly to
:class:`LatencyModel`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

from ..types import ValidationError, finite, non_negative, positive

# ---------------------------------------------------------------------------
# Structural minimum latency
# ---------------------------------------------------------------------------

#: The smallest total latency the model will return without an explicit
#: override. One tenth of a second is fast on Solana (slots are ~400ms) but
#: makes the "no zero" invariant testable without producing unrealistically
#: slow numbers in the happy path. A caller that wants zero latency for a
#: specific test must explicitly set ``min_total_s=0.0`` — that is the
#: deliberate friction that prevents accidental look-ahead.
_MIN_LATENCY_S: float = 0.1


# ---------------------------------------------------------------------------
# Distribution configuration
# ---------------------------------------------------------------------------

DistributionKind = Literal["constant", "uniform", "lognormal"]
"""Supported per-stage distribution shapes.

``constant`` — always the stated value; useful for deterministic tests.
``uniform``  — Uniform(low, high); captures a wide range cheaply.
``lognormal``— LogNormal(mu, sigma); right-skewed, matching observed latencies.
"""


@dataclass(frozen=True, slots=True)
class StageConfig:
    """Configuration for one latency stage.

    ``kind`` selects the distribution family. ``p50_s`` is the median in
    seconds (used as ``mu`` for lognormal after log-transform, as ``(low+high)/2``
    for uniform, or directly as the constant). ``spread_s`` is the half-width
    (uniform) or standard deviation in log-space (lognormal); ignored for
    ``constant``.

    All times are in seconds. Negative is refused at construction.
    """

    kind: DistributionKind = "lognormal"
    p50_s: float = 0.05
    spread_s: float = 0.5  # log-space sigma for lognormal; half-range for uniform

    def __post_init__(self) -> None:
        non_negative(self.p50_s, "p50_s")
        non_negative(self.spread_s, "spread_s")

    def sample(self, rng: random.Random) -> float:
        """Draw one sample from this stage's distribution.

        Always returns a non-negative value. The lognormal can never be
        negative by definition; uniform is bounded [low, high] with
        low = max(0, p50_s - spread_s).
        """
        if self.kind == "constant":
            return self.p50_s
        if self.kind == "uniform":
            low = max(0.0, self.p50_s - self.spread_s)
            high = self.p50_s + self.spread_s
            return rng.uniform(low, high)
        if self.kind == "lognormal":
            # mu = log(median), so exp(mu) == median. sigma is the log-space
            # standard deviation. The resulting distribution has the stated
            # median and is right-skewed, matching observed network latencies.
            import math

            if self.p50_s <= 0:
                return 0.0
            mu = math.log(self.p50_s)
            return rng.lognormvariate(mu, self.spread_s)
        raise ValidationError(f"unknown distribution kind {self.kind!r}")  # type: ignore[unreachable]


# ---------------------------------------------------------------------------
# Full latency config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LatencyConfig:
    """Per-stage latency distributions and landing model.

    ``min_total_s`` enforces the no-zero invariant. A total below this floor
    is silently raised to the floor. Callers that want zero latency (e.g., for
    a test that explicitly disables latency) must pass ``min_total_s=0.0``.
    The default is ``_MIN_LATENCY_S``.

    ``landing_probability`` is the probability a submitted transaction is
    eventually included by the validator network. ``1.0`` means it always
    lands; ``0.94`` is roughly the observed failure rate on Solana mainnet
    during normal conditions (``broker.py`` documents 6% as the config
    default). This is independent from landing delay.

    ``extra_landing_spread_s`` is added to the landing stage's draw when the
    network is congested — modelling the difference between "a transaction that
    takes longer to land" and "a transaction that never lands". Set to 0.0 to
    make landing delay identical to the configured landing stage.
    """

    quote_request: StageConfig = field(default_factory=lambda: StageConfig(p50_s=0.01))
    quote_response: StageConfig = field(
        default_factory=lambda: StageConfig(p50_s=0.15, spread_s=0.4)
    )
    risk_confirm: StageConfig = field(
        default_factory=lambda: StageConfig(kind="constant", p50_s=0.005)
    )
    signing: StageConfig = field(
        default_factory=lambda: StageConfig(kind="constant", p50_s=0.002)
    )
    submission: StageConfig = field(
        default_factory=lambda: StageConfig(p50_s=0.05, spread_s=0.3)
    )
    landing: StageConfig = field(
        default_factory=lambda: StageConfig(p50_s=0.4, spread_s=0.5)
    )

    landing_probability: float = 0.94
    extra_landing_spread_s: float = 0.0
    min_total_s: float = _MIN_LATENCY_S

    def __post_init__(self) -> None:
        if not 0.0 <= self.landing_probability <= 1.0:
            raise ValidationError(
                f"landing_probability {self.landing_probability} outside [0, 1]"
            )
        non_negative(self.extra_landing_spread_s, "extra_landing_spread_s")
        non_negative(self.min_total_s, "min_total_s")


# ---------------------------------------------------------------------------
# Named presets for the robustness suite
# ---------------------------------------------------------------------------

#: Median-conditions preset. The numbers are documented assumptions calibrated
#: against public Solana RPC benchmarks; a TIER_3 model replaces them with
#: observed distributions. At p50 most transactions land within ~650ms.
PRESET_P50 = LatencyConfig(
    quote_request=StageConfig(kind="lognormal", p50_s=0.010, spread_s=0.3),
    quote_response=StageConfig(kind="lognormal", p50_s=0.120, spread_s=0.4),
    risk_confirm=StageConfig(kind="constant", p50_s=0.005),
    signing=StageConfig(kind="constant", p50_s=0.002),
    submission=StageConfig(kind="lognormal", p50_s=0.040, spread_s=0.3),
    landing=StageConfig(kind="lognormal", p50_s=0.400, spread_s=0.4),
    landing_probability=0.94,
    extra_landing_spread_s=0.0,
    min_total_s=_MIN_LATENCY_S,
)

#: 90th-percentile congestion. Quote and landing stages are meaningfully
#: slower; landing probability drops to reflect elevated competition for block
#: space. Most transactions eventually land but take a few seconds longer.
PRESET_P90 = LatencyConfig(
    quote_request=StageConfig(kind="lognormal", p50_s=0.030, spread_s=0.5),
    quote_response=StageConfig(kind="lognormal", p50_s=0.400, spread_s=0.6),
    risk_confirm=StageConfig(kind="constant", p50_s=0.005),
    signing=StageConfig(kind="constant", p50_s=0.002),
    submission=StageConfig(kind="lognormal", p50_s=0.120, spread_s=0.5),
    landing=StageConfig(kind="lognormal", p50_s=1.200, spread_s=0.7),
    landing_probability=0.88,
    extra_landing_spread_s=0.5,
    min_total_s=_MIN_LATENCY_S,
)

#: 99th-percentile tail congestion. The slot auction is highly competitive;
#: many transactions expire before landing. Still produces *some* fills — a
#: scenario where nothing lands is not a robustness test, it is a network
#: outage. Landing probability at 0.60 matches observed tail-congestion events.
PRESET_P99 = LatencyConfig(
    quote_request=StageConfig(kind="lognormal", p50_s=0.100, spread_s=0.8),
    quote_response=StageConfig(kind="lognormal", p50_s=1.000, spread_s=0.9),
    risk_confirm=StageConfig(kind="constant", p50_s=0.005),
    signing=StageConfig(kind="constant", p50_s=0.002),
    submission=StageConfig(kind="lognormal", p50_s=0.500, spread_s=0.8),
    landing=StageConfig(kind="lognormal", p50_s=3.000, spread_s=1.0),
    landing_probability=0.60,
    extra_landing_spread_s=2.0,
    min_total_s=_MIN_LATENCY_S,
)


# ---------------------------------------------------------------------------
# Draw result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LatencyDraw:
    """One draw from the latency model, with stage breakdown.

    ``total_s`` is the sum of all six stage samples, floored at
    ``config.min_total_s``. It is always > 0 unless ``min_total_s=0.0`` is
    explicitly configured (which is only valid for tests that deliberately
    disable the bar-close invariant).

    ``will_land`` is drawn independently from ``total_s`` and reflects the
    probability that the submitted transaction is eventually included. A
    transaction that does not land still incurs ``total_s`` of delay — the
    strategy must wait before discovering the failure — but produces a
    FAILED fill rather than a LANDED one.

    ``landing_delay_s`` is the additional time the landing stage contributes
    on top of the pre-landing stages. The venue simulator uses this to
    schedule the settlement event: an order lands at
    ``submitted_at + total_s + landing_delay_s``. A late-landing order is
    priced against the market state at that later time, not at submission time.
    """

    quote_request_s: float
    quote_response_s: float
    risk_confirm_s: float
    signing_s: float
    submission_s: float
    landing_s: float
    total_s: float
    will_land: bool
    landing_delay_s: float  # extra delay before settlement; >= 0

    def __post_init__(self) -> None:
        for name in (
            "quote_request_s",
            "quote_response_s",
            "risk_confirm_s",
            "signing_s",
            "submission_s",
            "landing_s",
            "total_s",
            "landing_delay_s",
        ):
            non_negative(getattr(self, name), name)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class LatencyModel:
    """Draws latency samples from a configured per-stage distribution.

    The RNG is always an explicit ``random.Random`` instance, never the global
    module-level RNG. This makes draws deterministic and isolated: two models
    with the same seed produce identical draws regardless of what other code
    runs between them, which is the reproducibility guarantee the backtest
    depends on.

    Usage::

        model = LatencyModel(PRESET_P50, seed=42)
        draw = model.draw()
        receipt_ready_at = model.ready_at(submitted_at=now)

    The ``ready_at`` method is the primary integration point for the venue
    simulator: it draws latency and returns the earliest time the order may
    be matched against a market state.
    """

    def __init__(
        self,
        config: LatencyConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        """Construct a latency model.

        Parameters
        ----------
        config:
            Stage distributions and landing model. Defaults to
            ``PRESET_P50``.
        seed:
            RNG seed. ``None`` seeds from the OS (non-reproducible, appropriate
            for a live run). Pass an explicit integer for any deterministic
            context (tests, backtests, robustness suite).
        """
        self._config = config if config is not None else PRESET_P50
        # Use a local Random instance, never the global one. Two models in one
        # process (e.g., a baseline and a stressed variant running in the same
        # sweep) must not share state, and the global RNG is shared by
        # definition.
        self._rng = random.Random(seed)

    @property
    def config(self) -> LatencyConfig:
        return self._config

    def draw(self) -> LatencyDraw:
        """Draw one complete latency sample from all six stages.

        The draw is fully determined by the RNG state at call time, so two
        calls with the same seed produce the same sequence of draws. The
        robustness suite relies on this to compare baseline vs. stressed runs
        over identical market scenarios.

        The ``total_s`` field is floored at ``config.min_total_s`` after
        summing the stages. This is the structural enforcement of the "no zero
        latency" invariant — even if every stage happens to draw near zero,
        the floor keeps the total above the minimum. The only way to get zero
        total is to set ``min_total_s=0.0`` explicitly.

        ``will_land`` is drawn from a Bernoulli(landing_probability) and is
        independent of the stage draws. A high-latency draw can still land;
        a low-latency draw can still drop.

        ``landing_delay_s`` is drawn from the landing stage config *plus* the
        ``extra_landing_spread_s``, which models the difference between a
        transaction that takes a bit longer vs. one that never lands.
        """
        cfg = self._config
        qr = cfg.quote_request.sample(self._rng)
        qrs = cfg.quote_response.sample(self._rng)
        rc = cfg.risk_confirm.sample(self._rng)
        sig = cfg.signing.sample(self._rng)
        sub = cfg.submission.sample(self._rng)
        land = cfg.landing.sample(self._rng)

        raw_total = qr + qrs + rc + sig + sub + land
        total = max(raw_total, cfg.min_total_s)

        will_land = self._rng.random() < cfg.landing_probability

        # Landing delay: an additional draw representing congestion-driven
        # extra wait *after* the normal landing time. Kept separate so the
        # venue can schedule settlement at submitted_at + total + landing_delay.
        landing_delay = (
            cfg.extra_landing_spread_s * self._rng.random()
            if cfg.extra_landing_spread_s > 0.0
            else 0.0
        )

        return LatencyDraw(
            quote_request_s=qr,
            quote_response_s=qrs,
            risk_confirm_s=rc,
            signing_s=sig,
            submission_s=sub,
            landing_s=land,
            total_s=total,
            will_land=will_land,
            landing_delay_s=landing_delay,
        )

    def ready_at(self, submitted_at: float) -> float:
        """Return the earliest time an order submitted at ``submitted_at`` may settle.

        Draws latency once and returns ``submitted_at + total_s``. The returned
        value is always strictly greater than ``submitted_at`` when
        ``config.min_total_s > 0`` (the default). This satisfies the
        ``OrderReceipt.ready_at >= submitted_at`` invariant in ``types.py``
        and enforces the bar-close invariant: an order submitted at bar-close
        time ``t`` cannot be ready until at least ``t + min_total_s``.

        Callers that need both ``total_s`` and ``will_land`` should call
        :meth:`draw` directly. ``ready_at`` is a convenience for the common
        case where only the ready time is needed.
        """
        finite(submitted_at, "submitted_at")
        draw = self.draw()
        return submitted_at + draw.total_s


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "PRESET_P50",
    "PRESET_P90",
    "PRESET_P99",
    "DistributionKind",
    "LatencyConfig",
    "LatencyDraw",
    "LatencyModel",
    "StageConfig",
]
