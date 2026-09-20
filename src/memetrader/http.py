"""One place that builds HTTP clients, and one place that decides how hard to
try.

**Why this module exists.** httpx verifies TLS against the `certifi` bundle,
which contains only public roots. On a corporate network doing TLS inspection —
this machine sits behind Zscaler — every certificate is re-signed by a private
CA that `certifi` has never heard of, so *every* httpx call fails with
``CERTIFICATE_VERIFY_FAILED``. The fix is to verify against
``ssl.create_default_context()`` instead, which loads the operating system's
own trust store, where the proxy's CA actually lives.

This is not a weakening of verification: certificates are still fully verified,
just against the OS roots rather than a bundled copy. It is also strictly more
portable — on Linux and macOS the same call resolves to the normal OpenSSL
paths. Every module obtains its client from :func:`make_client` for exactly
this reason; a bare ``httpx.Client`` constructed anywhere in the tree is a bug
that only shows up on this network.

**What the audit added.** §15 on this file: *"Shared client is useful but
insufficient for resilience. Per-source timeout/retry budgets, exponential
backoff+jitter, rate-limit headers, circuit breaker, metrics, and request IDs.
Never retry order submission without idempotency."* Each of those is a separate
failure mode observed against a free public API on a 60-second fast tick:

* **Split timeouts.** One scalar timeout is what lets a hung read stall a tick.
  A connect that never completes and a body that trickles are different faults
  with different sensible budgets, so :class:`Timeouts` carries connect, read,
  write and pool separately. httpx's own default is 5s across the board; ours
  is deliberately tighter on connect than on read, because a connect that has
  not completed in a couple of seconds behind the proxy is not going to.

* **A retry budget, not a retry count.** Bounding attempts alone does not bound
  time: three attempts against a 15s read timeout is a 45-second stall inside a
  60-second tick, and the tick is then late for reasons no log line explains.
  :class:`RetryPolicy` bounds *both* — ``max_attempts`` and
  ``total_budget_seconds`` — and the budget is checked before every sleep and
  before every attempt, so the call returns inside the budget rather than one
  timeout past it.

* **Full jitter.** Nine coins polled on the same tick fail on the same tick and
  would retry on the same millisecond. That is a self-inflicted thundering herd
  against a free endpoint that is already struggling, and it converts one bad
  second into a rate-limit ban. The sleep is drawn uniformly from
  ``[0, capped_backoff]`` (AWS "full jitter"), which is the variant that
  actually decorrelates callers rather than merely blurring them.

* **A per-host circuit breaker.** After ``failure_threshold`` consecutive
  failures a host is opened for a cooldown and subsequent calls fail *fast*
  rather than spending the tick budget rediscovering that it is down. The
  breaker's state is exported (:meth:`CircuitBreaker.snapshot`) because an open
  breaker is a **data-health signal**, not just a latency problem:
  ``RiskState.data_health_ok`` is what it feeds, and the audit's §11 failure
  table lists "Data vendor/API rate limit → **No data-dependent trade**".

* **Idempotency.** Only ``GET``/``HEAD``/``OPTIONS`` are retried by default.
  The audit is explicit — *"Never retry order submission without
  idempotency"* — and a retried ``POST`` to a swap endpoint is a second order.
  A caller that has an idempotency key may pass ``idempotent=True`` and take
  responsibility for it.

* **Request IDs and structured logs.** Every outbound call gets a short ID that
  appears on every attempt line, so "the 429 at 14:03" can be tied to a
  specific attempt of a specific call rather than to a host.

**Secrets never reach a log line.** The environment on this machine holds a
live ``Ocp-Apim-Subscription-Key`` and an internal gateway hostname. Header
redaction in :func:`redact_headers` is unconditional and by *name*, not by a
caller remembering to ask, and query strings are scrubbed by
:func:`redact_url`. :class:`LogPolicy` additionally lets a caller declare a
hostname secret, in which case it is logged as a stable digest — enough to
correlate two log lines, not enough to name the host.

**Clocks.** Durations here (retry budget, breaker cooldown) are measured on
``time.monotonic``. That is the one deliberate exception to this codebase's
"epoch seconds as float" rule: a wall clock can step backwards under NTP, and a
retry budget that can go negative is a retry budget that does not bound
anything. Wall-clock timestamps still belong on *observations*; these are
elapsed times.

Sync only. ``httpx.AsyncClient`` appears nowhere in this project.
"""

from __future__ import annotations

import hashlib
import logging
import random
import ssl
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

__all__ = [
    "BROWSER_UA",
    "DEFAULT_HEADERS",
    "DEFAULT_LOG_POLICY",
    "DEFAULT_RETRY",
    "DEFAULT_TIMEOUTS",
    "IDEMPOTENT_METHODS",
    "NON_RETRYABLE_STATUSES",
    "RETRYABLE_STATUSES",
    "Attempt",
    "BreakerState",
    "BreakerStatus",
    "CircuitBreaker",
    "CircuitOpen",
    "HttpError",
    "LogPolicy",
    "RequestFailed",
    "RequestOutcome",
    "RetryPolicy",
    "Timeouts",
    "execute",
    "is_retryable_status",
    "make_client",
    "new_request_id",
    "redact_headers",
    "redact_url",
    "request",
    "retry_after_seconds",
    "ssl_context",
]

log = logging.getLogger(__name__)

# Cloudflare fronts DexScreener and answers the stock httpx user-agent with a
# 403 interstitial whenever it feels like it. A plain browser UA is enough;
# nothing else about the request needs to lie.
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

DEFAULT_HEADERS = {"User-Agent": BROWSER_UA, "Accept": "application/json"}


@lru_cache(maxsize=1)
def ssl_context() -> ssl.SSLContext:
    """The OS trust store. Cached — building a context reads the whole root
    store from disk, and doing that per request is measurably slow."""
    return ssl.create_default_context()


# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Timeouts:
    """Four separate budgets, because they fail for four different reasons.

    A single scalar is what lets a hung *read* stall a trading tick: the socket
    connected fine, the server accepted the request, and then sent one byte
    every few seconds until the whole 60-second tick was gone. Splitting them
    means the call that is actually stuck is the one that is cut off.

    Defaults, and why each is what it is:

    * ``connect=5.0`` — behind the TLS-inspecting proxy a connection either
      completes in well under a second or it is not going to. Five seconds is
      generous; it exists to absorb a DNS hiccup, not a dead host.
    * ``read=12.0`` — the Arctic Shift 422 investigation measured server-side
      query timeouts at 3.1-3.4s and successes at 0.1-1.6s, and GeckoTerminal
      OHLCV pages are the slowest thing we fetch. Twelve seconds is comfortably
      past the slowest healthy response and comfortably inside a 60s tick.
    * ``write=10.0`` — request bodies here are small; this only binds if the
      upstream socket has stopped draining.
    * ``pool=5.0`` — waiting for a free connection from the pool. An unbounded
      pool wait is indistinguishable from a hung server in a log, and this is
      a sync single-threaded client, so contention here means a leak.

    The whole point is that these are per-call-site configurable: Jupiter on
    the order path and Reddit on a disabled experiment stream should not share
    a budget.
    """

    connect: float = 5.0
    read: float = 12.0
    write: float = 10.0
    pool: float = 5.0

    @property
    def worst_case_seconds(self) -> float:
        """Upper bound on one attempt, used to size a retry budget honestly.

        Connect and read are sequential phases of the same attempt, so they
        add. This is what makes "three attempts fits in a 60s tick" a claim you
        can check rather than a hope.
        """
        return self.connect + self.read + self.write + self.pool

    def as_httpx(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self.connect, read=self.read, write=self.write, pool=self.pool
        )

    @classmethod
    def coerce(cls, value: float | Timeouts | httpx.Timeout | None) -> Timeouts:
        """Accept the legacy scalar without silently changing its meaning.

        ``make_client(15.0)`` is called from five modules that predate this
        type. A bare number used to mean "15 seconds for everything", and it
        still does — it is spread across all four phases rather than
        reinterpreted, so nobody's effective budget shrinks under them.
        """
        if value is None:
            return DEFAULT_TIMEOUTS
        if isinstance(value, Timeouts):
            return value
        if isinstance(value, httpx.Timeout):
            return cls(
                connect=value.connect if value.connect is not None else 5.0,
                read=value.read if value.read is not None else 12.0,
                write=value.write if value.write is not None else 10.0,
                pool=value.pool if value.pool is not None else 5.0,
            )
        seconds = float(value)
        return cls(connect=seconds, read=seconds, write=seconds, pool=seconds)


DEFAULT_TIMEOUTS = Timeouts()


# ---------------------------------------------------------------------------
# Retry classification
# ---------------------------------------------------------------------------

# Statuses where trying the identical request again is a reasonable thing to
# believe might work. Each is here because it says so, not because it is a 5xx:
#   408 Request Timeout      — the server gave up waiting on us.
#   425 Too Early            — replay explicitly invited.
#   429 Too Many Requests    — real rate limiting; honour Retry-After.
#   500/502/503/504          — transient upstream/gateway faults.
RETRYABLE_STATUSES: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})

# Named explicitly so the exclusion is a documented decision rather than an
# absence somebody later "fixes".
#
# **422 is the important one, and it is NOT rate limiting.** Arctic Shift (the
# Reddit archive host) answers some requests with
# ``422 {"data": null, "error": "Timeout. Maybe slow down a bit"}``. The message
# invites exactly the wrong response. 2026-09-19 was spent establishing that it
# is not a rate limit:
#
#   * 12 back-to-back comment pages at 0.6s spacing returned 12/12 x 200;
#   * a failing request reproduces on the *first cold request of a fresh
#     process*, with no prior traffic to be limited for;
#   * 0.6s, 1.0s, 2.0s and 2.5s spacing all failed identically, as did a 105s
#     backoff before the retry;
#   * 11 different ``fields`` combinations x 3 tries each = 33 consecutive 200s,
#     so the field list is not the trigger.
#
# It is a property of the specific ``(after, before)`` range: failures take
# 3.1-3.4s and successes 0.1-1.6s, the shape of a ~3s server-side query timeout
# on a range holding too many rows. Retrying the identical range cannot succeed,
# and backing off cannot help — it burns 3.3 seconds of tick budget per attempt
# to re-learn a permanent fact about that range. DexScreener's 422 on a
# malformed address is the same class: the request is wrong, not early.
#
# So: 422 is terminal. ``sentiment.py`` does not engineer around it either — it
# stops the walk and reports the shortfall in ``degraded_reason`` so a low
# comment count cannot be mistaken for a quiet subreddit.
NON_RETRYABLE_STATUSES: frozenset[int] = frozenset({400, 401, 403, 404, 409, 410, 422})

# Methods whose repetition is safe by definition (RFC 9110 §9.2.2). A retried
# POST to a swap endpoint is a second order, which is audit C11's duplicate
# submission in one line of code.
IDEMPOTENT_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS"})


def is_retryable_status(status: int, *, policy: RetryPolicy | None = None) -> bool:
    """Whether ``status`` may be retried.

    Allow-list, not deny-list. An unknown status is *not* retried, because the
    failure mode of guessing wrong in that direction is burning a tick budget
    on something that will never succeed — which is precisely what a 422 retry
    loop does. See :data:`NON_RETRYABLE_STATUSES` for the measurement.
    """
    allowed = RETRYABLE_STATUSES if policy is None else policy.retry_statuses
    return status in allowed and status not in NON_RETRYABLE_STATUSES


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """A bounded attempt count **and** a bounded total time.

    Both bounds are load-bearing and neither implies the other. Three attempts
    against a 12-second read timeout is 36 seconds of a 60-second tick; a
    30-second budget with no attempt cap is an unbounded number of fast
    failures against a host answering 503 in 5ms. The loop stops at whichever
    binds first, and :class:`RequestOutcome` records which one did.

    ``total_budget_seconds`` covers *everything*: every attempt's own latency
    plus every backoff sleep. A budget that only counted sleeps would be
    satisfied while the call ran for a minute.

    Backoff is exponential with **full jitter**: the sleep is drawn uniformly
    from ``[0, min(cap, base * mult ** (attempt - 1))]``. Nine coins polled on
    one tick fail together, and equal sleeps would have them retry together —
    a herd we would be aiming at a free public API. Full jitter is the variant
    that actually decorrelates them; halving the jitter range to "look less
    random" reintroduces the correlation it exists to remove.
    """

    max_attempts: int = 3
    total_budget_seconds: float = 20.0
    backoff_base_seconds: float = 0.25
    backoff_multiplier: float = 3.0
    backoff_max_seconds: float = 4.0
    retry_statuses: frozenset[int] = RETRYABLE_STATUSES
    # 429 responses carry Retry-After often enough to be worth honouring, and
    # ignoring a server's own number is how a soft limit becomes a ban. It is
    # capped because some hosts answer with minutes, and a tick cannot wait
    # minutes — past the cap we stop rather than sleep, and say so.
    respect_retry_after: bool = True
    max_retry_after_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.total_budget_seconds <= 0:
            raise ValueError("total_budget_seconds must be > 0")

    def backoff_ceiling(self, attempt: int) -> float:
        """Un-jittered ceiling after ``attempt`` failed attempts (1-based)."""
        raw = self.backoff_base_seconds * (self.backoff_multiplier ** (attempt - 1))
        return min(raw, self.backoff_max_seconds)

    def backoff(self, attempt: int, rng: random.Random) -> float:
        return rng.uniform(0.0, self.backoff_ceiling(attempt))


DEFAULT_RETRY = RetryPolicy()

#: Jitter source used when a caller does not supply one. An explicit instance
#: rather than the bare ``random`` module: the module's top-level functions are
#: methods of a hidden instance, so passing the module where a ``random.Random``
#: is declared only type-checks by accident. Retry jitter is not
#: security-sensitive and is never replayed, so a process-local instance is
#: enough; tests that need determinism pass their own ``rng``.
_JITTER_RNG = random.Random()


def retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse ``Retry-After``. ``None`` means the header was absent or unparsable.

    Missing is never zero here either: ``None`` is "the server did not tell us"
    and leads to our own backoff, while ``0.0`` would be "the server said retry
    immediately" — a different and much rarer instruction.

    Only the delta-seconds form is parsed. The HTTP-date form is legal and
    essentially never sent by these APIs, and parsing it correctly requires
    trusting our clock against theirs, which is a worse bet than falling back
    to our own jittered backoff.
    """
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        value = float(raw.strip())
    except ValueError:
        return None
    if value < 0 or value != value:  # negative or NaN
        return None
    return value


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

_REDACTED = "<redacted>"

# Matched case-insensitively against the header name. ``Ocp-Apim-Subscription-Key``
# is in this list because it is live in this machine's environment; the rest are
# here so that adding a source does not require remembering to extend it.
SECRET_HEADER_NAMES: frozenset[str] = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "api-key",
        "apikey",
        "x-auth-token",
        "x-goog-api-key",
        "anthropic-api-key",
        "ocp-apim-subscription-key",
    }
)

# Query-string parameter names that carry credentials. Jupiter's keyed base
# takes its key in a header, but a URL is the easiest place for a secret to end
# up by accident and the cost of over-redacting a query param in a log is zero.
SECRET_QUERY_KEYS: frozenset[str] = frozenset(
    {"key", "api_key", "apikey", "token", "access_token", "secret", "password", "sig"}
)


def _digest(value: str) -> str:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=4).hexdigest()


def redact_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    """Replace every credential-bearing header value with ``<redacted>``.

    Unconditional and by name. The alternative — a caller passing
    ``log_headers=False`` at each site — fails the first time somebody adds a
    source and does not know the rule, and the failure is a live subscription
    key in a file that gets pasted into a bug report.
    """
    if not headers:
        return {}
    return {
        name: (_REDACTED if name.lower() in SECRET_HEADER_NAMES else value)
        for name, value in headers.items()
    }


def redact_url(url: str | httpx.URL, *, policy: LogPolicy | None = None) -> str:
    """A URL safe to log: credentials stripped, secret hosts reduced to a digest.

    The internal gateway hostname is itself sensitive — it names infrastructure
    that is not supposed to be discoverable — so a caller can declare it in
    :class:`LogPolicy` and it is logged as ``host-<8 hex>``. That is still
    enough to tell two hosts apart across log lines, which is all a latency
    investigation needs.
    """
    pol = policy or DEFAULT_LOG_POLICY
    parts = urlsplit(str(url))
    host = parts.hostname or ""
    netloc = pol.label_host(host)
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    query = urlencode(
        [
            (k, _REDACTED if k.lower() in SECRET_QUERY_KEYS else v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
        ]
    )
    return urlunsplit((parts.scheme, netloc, parts.path, query, ""))


@dataclass(frozen=True, slots=True)
class LogPolicy:
    """Which hostnames must never appear in a log line verbatim."""

    secret_hosts: frozenset[str] = frozenset()

    def label_host(self, host: str) -> str:
        if not host:
            return ""
        lowered = host.lower()
        if lowered in self.secret_hosts:
            return f"host-{_digest(lowered)}"
        return host


DEFAULT_LOG_POLICY = LogPolicy()


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class BreakerState(StrEnum):
    """Three states, because two are not enough to recover safely.

    ``HALF_OPEN`` is what stops a recovering host being hit by the full tick's
    worth of traffic the instant the cooldown expires: exactly one probe is
    allowed through, and the host is only closed if it succeeds.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True, slots=True)
class BreakerStatus:
    """A snapshot of one host's health, for risk and for the status command.

    This is read by the data-health check, not only by a latency dashboard. An
    open breaker on DexScreener means marks are stale, which under the audit's
    §11 failure table means "no data-dependent trade" — so the type carries
    enough to explain the refusal to an operator (``last_error``,
    ``cooldown_remaining_seconds``) rather than just a boolean.
    """

    host: str
    state: BreakerState
    consecutive_failures: int
    cooldown_remaining_seconds: float | None
    last_error: str | None
    opened_count: int

    @property
    def healthy(self) -> bool:
        return self.state is BreakerState.CLOSED


class CircuitBreaker:
    """Per-host fail-fast after ``failure_threshold`` consecutive failures.

    The measurement that motivates it: when GeckoTerminal is down, a nine-coin
    sweep makes nine calls, each of which burns the full retry budget
    (attempts x read timeout + sleeps) before giving up. At the defaults that
    is nine x ~20 seconds = three minutes inside a 60-second tick, and every
    one of those calls already knew the answer after the first. Opening the
    host converts the second through ninth calls into an immediate
    :class:`CircuitOpen`, which the tick can record as a data-health incident
    in milliseconds.

    Keyed on host rather than on URL deliberately. DexScreener's pairs endpoint
    and its tokens endpoint fail together because the host is what is down; per
    path, the breaker would need nine times the evidence to trip.

    *Consecutive* failures, not a rate. A rate needs a window and a window
    needs tuning; a free API that answers one request in four is not usable for
    trading anyway, and the recovery path (half-open probe) is what actually
    governs how quickly we come back.

    Not thread-safe by design — this is a single-threaded synchronous process,
    and a lock here would be a claim about concurrency that nothing else in the
    codebase supports.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 4,
        cooldown_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}
        self._last_error: dict[str, str] = {}
        self._opened_count: dict[str, int] = {}
        self._half_open: set[str] = set()

    # -- state ----------------------------------------------------------

    def _state(self, host: str) -> BreakerState:
        opened_at = self._opened_at.get(host)
        if opened_at is None:
            return BreakerState.CLOSED
        if self._clock() - opened_at >= self.cooldown_seconds:
            return BreakerState.HALF_OPEN
        return BreakerState.OPEN

    def status(self, host: str) -> BreakerStatus:
        state = self._state(host)
        remaining: float | None = None
        if state is BreakerState.OPEN:
            remaining = max(
                0.0, self.cooldown_seconds - (self._clock() - self._opened_at[host])
            )
        return BreakerStatus(
            host=host,
            state=state,
            consecutive_failures=self._failures.get(host, 0),
            cooldown_remaining_seconds=remaining,
            last_error=self._last_error.get(host),
            opened_count=self._opened_count.get(host, 0),
        )

    def snapshot(self) -> dict[str, BreakerStatus]:
        """Every host this breaker has an opinion about.

        Risk reads this. A host that has never failed is absent rather than
        present-and-healthy, because "we have not called it this run" and "we
        called it and it was fine" are different facts and the second one is
        the only one that should let an entry through.
        """
        hosts = set(self._failures) | set(self._opened_at)
        return {host: self.status(host) for host in sorted(hosts)}

    @property
    def open_hosts(self) -> tuple[str, ...]:
        return tuple(
            host
            for host, status in self.snapshot().items()
            if status.state is BreakerState.OPEN
        )

    # -- transitions ----------------------------------------------------

    def allow(self, host: str) -> bool:
        """May a call to ``host`` proceed? Consumes the half-open probe slot."""
        state = self._state(host)
        if state is BreakerState.CLOSED:
            return True
        if state is BreakerState.OPEN:
            return False
        # HALF_OPEN: exactly one probe. The second caller in the same tick is
        # refused, which is the whole point — a recovering host must not be
        # handed the backlog that knocked it over.
        if host in self._half_open:
            return False
        self._half_open.add(host)
        return True

    def record_success(self, host: str) -> None:
        self._failures.pop(host, None)
        self._opened_at.pop(host, None)
        self._last_error.pop(host, None)
        self._half_open.discard(host)

    def record_failure(self, host: str, error: str) -> None:
        self._half_open.discard(host)
        self._last_error[host] = error
        if host in self._opened_at:
            # A failed half-open probe restarts the cooldown rather than
            # letting the next caller probe again immediately.
            self._opened_at[host] = self._clock()
            self._failures[host] = self._failures.get(host, 0) + 1
            return
        count = self._failures.get(host, 0) + 1
        self._failures[host] = count
        if count >= self.failure_threshold:
            self._opened_at[host] = self._clock()
            self._opened_count[host] = self._opened_count.get(host, 0) + 1
            log.warning(
                "circuit opened host=%s consecutive_failures=%d cooldown=%.1fs last_error=%s",
                host,
                count,
                self.cooldown_seconds,
                error,
            )

    def reset(self, host: str | None = None) -> None:
        """Clear state. ``None`` clears every host. Operator action only."""
        if host is None:
            self._failures.clear()
            self._opened_at.clear()
            self._last_error.clear()
            self._opened_count.clear()
            self._half_open.clear()
        else:
            self.record_success(host)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class HttpError(httpx.HTTPError):
    """Base for this module's own failures.

    Subclasses ``httpx.HTTPError`` on purpose: five modules already write
    ``except httpx.HTTPError`` around their fetches, and a new exception type
    that slipped past those handlers would turn a vendor outage into a crashed
    tick. Inheriting means the existing fail-soft paths keep working unchanged.
    """


class CircuitOpen(HttpError):
    """The host's breaker is open; no request was made.

    Distinct from a timeout because the *response* is different: a timeout is a
    reason to try the next source, an open breaker is a reason to mark the
    whole source unhealthy and refuse entries.
    """

    def __init__(self, host: str, status: BreakerStatus) -> None:
        remaining = status.cooldown_remaining_seconds
        detail = "" if remaining is None else f", {remaining:.1f}s of cooldown remaining"
        super().__init__(
            f"circuit open for {host}: {status.consecutive_failures} "
            f"consecutive failures{detail}"
        )
        self.host = host
        self.status = status


class RequestFailed(HttpError):
    """Every permitted attempt failed at the transport layer."""

    def __init__(self, message: str, outcome: RequestOutcome) -> None:
        super().__init__(message)
        self.outcome = outcome


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


def new_request_id() -> str:
    """A short ID stamped on every attempt of one logical call.

    Eight hex characters, not a UUID: it exists to correlate three log lines
    inside one tick, and a 36-character UUID on every attempt line makes the
    log harder to read for no additional discrimination at this volume. It is
    deliberately *not* time-prefixed like ``ids.py`` — a request is not a
    ledger record and must not be mistaken for one.
    """
    return f"req-{random.getrandbits(32):08x}"


@dataclass(frozen=True, slots=True)
class Attempt:
    """One physical HTTP attempt. Latency is measured even when it failed —
    especially when it failed, since a 3.3-second failure and a 0.2-second
    failure are different diagnoses (see the 422 note above)."""

    number: int
    latency_seconds: float
    status: int | None
    error: str | None
    retryable: bool
    backoff_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class RequestOutcome:
    """Everything one logical call did, including the attempts that failed.

    Returned rather than logged-and-discarded because the audit wants metrics,
    and because "we got a price" and "we got a price on the third attempt after
    eleven seconds" are different facts about data health.
    """

    request_id: str
    method: str
    host: str
    url: str
    attempts: tuple[Attempt, ...]
    elapsed_seconds: float
    response: httpx.Response | None = None
    error: Exception | None = None
    breaker_state: BreakerState = BreakerState.CLOSED
    # Which bound stopped the loop: "ok", "attempts", "budget", "terminal",
    # "circuit_open", "retry_after_too_long". Recorded because "it failed" and
    # "it ran out of tick budget" call for different operator responses.
    stopped_by: str = "ok"

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def succeeded(self) -> bool:
        return self.response is not None and self.response.status_code < 400


# ---------------------------------------------------------------------------
# The factory
# ---------------------------------------------------------------------------


def make_client(
    timeout: float | Timeouts | httpx.Timeout | None = 15.0,
    headers: dict[str, str] | None = None,
    *,
    limits: httpx.Limits | None = None,
    transport: httpx.BaseTransport | None = None,
    follow_redirects: bool = True,
) -> httpx.Client:
    """Build the one kind of client this project is allowed to use.

    Signature is backward compatible: ``make_client(15.0)`` and
    ``make_client(cfg.data.http_timeout_seconds, _HEADERS)`` — the two forms
    every existing caller uses — behave exactly as before. The first parameter
    now *also* accepts a :class:`Timeouts`, which is how a call site splits its
    budgets; a bare float is still spread across all four phases so nobody's
    effective timeout changes under them.

    ``verify=ssl_context()`` is the Zscaler fix and is not optional: there is
    no parameter to turn it off, because the only reason anyone would reach for
    one is to pass ``verify=False``, and a paper trader that has learned to
    ignore certificate errors is a live trader that will.

    ``transport`` exists for tests. Every test in this repo mocks the transport
    with ``httpx.MockTransport`` — there are no live network calls in the suite,
    and there must not be: a test that hits DexScreener fails in CI, fails on a
    plane, and occasionally fails because DexScreener is having a bad minute.
    """
    resolved = Timeouts.coerce(timeout)
    kwargs: dict[str, Any] = {
        "timeout": resolved.as_httpx(),
        "headers": {**DEFAULT_HEADERS, **(headers or {})},
        "follow_redirects": follow_redirects,
    }
    if limits is not None:
        kwargs["limits"] = limits
    if transport is not None:
        # httpx rejects `verify` alongside an explicit transport in some
        # versions, and it would be meaningless anyway — a mock transport does
        # no TLS.
        kwargs["transport"] = transport
    else:
        kwargs["verify"] = ssl_context()
    return httpx.Client(**kwargs)


# ---------------------------------------------------------------------------
# The request loop
# ---------------------------------------------------------------------------


def _host_of(url: str | httpx.URL) -> str:
    return (urlsplit(str(url)).hostname or "").lower()


def _transport_error_is_retryable(exc: Exception) -> bool:
    """Connect/read/pool faults are worth another go; a bad URL is not.

    ``httpx.TransportError`` covers timeouts, connection resets and proxy
    failures — all genuinely transient. ``httpx.InvalidURL`` and
    ``httpx.UnsupportedProtocol`` are programming errors and retrying them
    burns budget to re-raise the same exception.
    """
    if isinstance(exc, (httpx.InvalidURL, httpx.UnsupportedProtocol)):
        return False
    return isinstance(exc, httpx.TransportError)


def execute(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    params: Any = None,
    json: Any = None,
    content: Any = None,
    headers: Mapping[str, str] | None = None,
    timeouts: Timeouts | float | None = None,
    retry: RetryPolicy | None = None,
    breaker: CircuitBreaker | None = None,
    idempotent: bool | None = None,
    request_id: str | None = None,
    log_policy: LogPolicy | None = None,
    logger: logging.Logger | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> RequestOutcome:
    """Perform one logical call and return what happened. Never raises for HTTP.

    This is the metrics-bearing entry point: it returns a
    :class:`RequestOutcome` even when every attempt failed, so a caller can
    record *why* a source was unavailable instead of catching an exception that
    has already thrown the attempt history away. :func:`request` is the thin
    raising wrapper for callers that just want a response.

    **The ordering of the checks is the design.** Budget is checked before each
    attempt *and* before each sleep, so the call cannot return one timeout past
    its budget; the breaker is consulted before the first attempt, so an open
    host costs microseconds; and the retry decision consults
    :func:`is_retryable_status` rather than ``status >= 400``, which is what
    keeps a 422 from being retried as if it were a rate limit.

    ``idempotent`` defaults to "is the method in
    :data:`IDEMPOTENT_METHODS`". Passing ``True`` for a POST is a deliberate
    statement that the endpoint carries an idempotency key; the audit's C11
    duplicate-submission row is what that statement is measured against.
    """
    policy = retry or DEFAULT_RETRY
    log_pol = log_policy or DEFAULT_LOG_POLICY
    lg = logger or log
    random_source: random.Random = rng or _JITTER_RNG
    method = method.upper()
    rid = request_id or new_request_id()
    host = _host_of(url)
    safe_url = redact_url(url, policy=log_pol)
    safe_host = log_pol.label_host(host)
    may_retry = (method in IDEMPOTENT_METHODS) if idempotent is None else idempotent
    per_attempt = Timeouts.coerce(timeouts) if timeouts is not None else None

    started = clock()
    attempts: list[Attempt] = []

    def finish(
        *,
        response: httpx.Response | None,
        error: Exception | None,
        stopped_by: str,
    ) -> RequestOutcome:
        return RequestOutcome(
            request_id=rid,
            method=method,
            host=host,
            url=safe_url,
            attempts=tuple(attempts),
            elapsed_seconds=clock() - started,
            response=response,
            error=error,
            breaker_state=(
                breaker.status(host).state if breaker is not None else BreakerState.CLOSED
            ),
            stopped_by=stopped_by,
        )

    if breaker is not None and not breaker.allow(host):
        status = breaker.status(host)
        lg.warning(
            "http circuit_open rid=%s method=%s host=%s url=%s cooldown_remaining=%s",
            rid,
            method,
            safe_host,
            safe_url,
            status.cooldown_remaining_seconds,
        )
        return finish(
            response=None, error=CircuitOpen(host, status), stopped_by="circuit_open"
        )

    last_response: httpx.Response | None = None
    last_error: Exception | None = None

    for attempt_no in range(1, policy.max_attempts + 1):
        remaining = policy.total_budget_seconds - (clock() - started)
        if attempt_no > 1 and remaining <= 0:
            lg.warning(
                "http budget_exhausted rid=%s method=%s host=%s attempts=%d elapsed=%.3fs",
                rid,
                method,
                safe_host,
                len(attempts),
                clock() - started,
            )
            return finish(response=last_response, error=last_error, stopped_by="budget")

        # The budget is enforced *inside* the attempt, not only between
        # attempts. Checking only between them bounds the number of stalls, not
        # their total length: three attempts against a 12-second read timeout
        # overruns a 20-second budget by seven seconds no matter how carefully
        # the gaps are policed. Clamping each phase to what is left of the
        # budget is what makes "this call returns within
        # total_budget_seconds" a guarantee rather than an intention.
        base = per_attempt if per_attempt is not None else Timeouts.coerce(client.timeout)
        effective = Timeouts(
            connect=min(base.connect, remaining),
            read=min(base.read, remaining),
            write=min(base.write, remaining),
            pool=min(base.pool, remaining),
        )

        attempt_started = clock()
        status_code: int | None = None
        error_text: str | None = None
        try:
            response = client.request(
                method,
                url,
                params=params,
                json=json,
                content=content,
                headers=dict(headers) if headers else None,
                timeout=effective.as_httpx(),
            )
        except httpx.HTTPError as exc:
            latency = clock() - attempt_started
            last_error = exc
            last_response = None
            error_text = f"{type(exc).__name__}: {exc}"
            retryable = may_retry and _transport_error_is_retryable(exc)
            attempts.append(
                Attempt(
                    number=attempt_no,
                    latency_seconds=latency,
                    status=None,
                    error=error_text,
                    retryable=retryable,
                )
            )
            lg.warning(
                "http attempt rid=%s n=%d method=%s host=%s url=%s status=none "
                "latency=%.3fs error=%s retryable=%s headers=%s",
                rid,
                attempt_no,
                method,
                safe_host,
                safe_url,
                latency,
                error_text,
                retryable,
                redact_headers(headers),
            )
            if breaker is not None:
                breaker.record_failure(host, error_text)
            if not retryable:
                return finish(response=None, error=exc, stopped_by="terminal")
        else:
            latency = clock() - attempt_started
            status_code = response.status_code
            last_response = response
            last_error = None
            retryable = may_retry and is_retryable_status(status_code, policy=policy)
            attempts.append(
                Attempt(
                    number=attempt_no,
                    latency_seconds=latency,
                    status=status_code,
                    error=None,
                    retryable=retryable,
                )
            )
            lg.info(
                "http attempt rid=%s n=%d method=%s host=%s url=%s status=%d "
                "latency=%.3fs retryable=%s headers=%s",
                rid,
                attempt_no,
                method,
                safe_host,
                safe_url,
                status_code,
                latency,
                retryable,
                redact_headers(headers),
            )
            if status_code < 500 and status_code != 429:
                # A 4xx that is not a rate limit is a *successful* conversation
                # with the host: the server is up and answered. Counting it as
                # a breaker failure would open DexScreener because one mint is
                # unknown, which is the 422 mistake in a different costume.
                if breaker is not None:
                    breaker.record_success(host)
                if not retryable:
                    return finish(response=response, error=None, stopped_by="ok")
            else:
                if breaker is not None:
                    breaker.record_failure(host, f"HTTP {status_code}")
                if not retryable:
                    return finish(response=response, error=None, stopped_by="terminal")

        if attempt_no == policy.max_attempts:
            lg.warning(
                "http attempts_exhausted rid=%s method=%s host=%s attempts=%d elapsed=%.3fs",
                rid,
                method,
                safe_host,
                len(attempts),
                clock() - started,
            )
            return finish(response=last_response, error=last_error, stopped_by="attempts")

        delay = policy.backoff(attempt_no, random_source)
        if (
            policy.respect_retry_after
            and last_response is not None
            and status_code == 429
            and (hinted := retry_after_seconds(last_response)) is not None
        ):
            if hinted > policy.max_retry_after_seconds:
                lg.warning(
                    "http retry_after_too_long rid=%s host=%s retry_after=%.1fs cap=%.1fs",
                    rid,
                    safe_host,
                    hinted,
                    policy.max_retry_after_seconds,
                )
                return finish(
                    response=last_response, error=None, stopped_by="retry_after_too_long"
                )
            # Honour the server's number, then add our own jitter on top so the
            # nine coins that were all told "wait 2s" do not all wake at 2.000s.
            delay = hinted + random_source.uniform(0.0, policy.backoff_ceiling(attempt_no))

        remaining = policy.total_budget_seconds - (clock() - started)
        if delay >= remaining:
            lg.warning(
                "http budget_exhausted rid=%s host=%s backoff=%.3fs remaining=%.3fs",
                rid,
                safe_host,
                delay,
                remaining,
            )
            return finish(response=last_response, error=last_error, stopped_by="budget")

        attempts[-1] = Attempt(
            number=attempts[-1].number,
            latency_seconds=attempts[-1].latency_seconds,
            status=attempts[-1].status,
            error=attempts[-1].error,
            retryable=attempts[-1].retryable,
            backoff_seconds=delay,
        )
        sleep(delay)

    # Unreachable: the loop returns on its final iteration. Kept so a future
    # edit to the bounds cannot fall off the end into an implicit None.
    return finish(response=last_response, error=last_error, stopped_by="attempts")


def request(
    client: httpx.Client,
    method: str,
    url: str,
    **kwargs: Any,
) -> httpx.Response:
    """:func:`execute`, raising instead of returning a failed outcome.

    Raises :class:`CircuitOpen` when the host is fenced off and
    :class:`RequestFailed` when every permitted attempt failed at the transport
    layer. An HTTP *status* — including a 422 or a final 503 — is returned as a
    response, because whether a 404 is an error is the caller's question:
    ``market.py`` treats a 404 on a mint as "no pair", which is data, not a
    failure.
    """
    outcome = execute(client, method, url, **kwargs)
    if outcome.response is not None:
        return outcome.response
    if isinstance(outcome.error, CircuitOpen):
        raise outcome.error
    raise RequestFailed(
        f"{method} {outcome.url} failed after {outcome.attempt_count} attempt(s) "
        f"in {outcome.elapsed_seconds:.2f}s ({outcome.stopped_by})",
        outcome,
    ) from outcome.error
