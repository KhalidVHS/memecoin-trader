"""Resilience tests for the shared HTTP client.

**No test in this file touches the network.** Every one drives an
``httpx.MockTransport``, and that is a hard rule rather than a preference: a
test that hits DexScreener fails in CI, fails on a plane, and occasionally
fails because DexScreener is having a bad minute — at which point the suite
stops being evidence about our code.

Time is injected too. A retry-budget test that actually slept would take twenty
seconds and would still be measuring the machine's scheduler rather than the
policy, so ``clock`` and ``sleep`` are fakes that advance a counter. That makes
the assertions exact instead of approximate: "the budget stopped it at 20.0s"
rather than "it took roughly twenty seconds, probably".
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import httpx
import pytest

from memetrader import http

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeClock:
    """A monotonic clock that only moves when the test says so.

    ``advance_per_request`` is what makes an *attempt* cost time, which is the
    half of the retry budget that a sleep-counting fake would miss entirely.
    """

    now: float = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeSleep:
    """Records every sleep instead of taking one."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.clock.advance(seconds)


def responder(
    statuses: list[int],
    *,
    clock: FakeClock | None = None,
    latency: float = 0.0,
    headers: dict[str, str] | None = None,
    seen: list[httpx.Request] | None = None,
):
    """A transport that serves ``statuses`` in order, repeating the last one."""
    remaining = list(statuses)

    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if clock is not None and latency:
            # A faithful fake honours the read timeout it was handed. That is
            # what makes the budget test meaningful: the loop clamps each
            # attempt's timeout to what is left of the budget, and a transport
            # that ignored it would let the fake overrun in a way a real socket
            # could not.
            read_budget = (request.extensions.get("timeout") or {}).get("read")
            if read_budget is not None and latency > read_budget:
                clock.advance(read_budget)
                raise httpx.ReadTimeout("read timed out", request=request)
            clock.advance(latency)
        status = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return httpx.Response(status, json={"ok": status < 400}, headers=headers or {})

    return httpx.MockTransport(handle)


def raiser(exc: Exception, *, clock: FakeClock | None = None, latency: float = 0.0):
    def handle(request: httpx.Request) -> httpx.Response:
        if clock is not None and latency:
            clock.advance(latency)
        raise exc

    return httpx.MockTransport(handle)


URL = "https://api.dexscreener.com/latest/dex/tokens/BONK"


# ---------------------------------------------------------------------------
# make_client — the Zscaler contract, and backward compatibility
# ---------------------------------------------------------------------------


def test_make_client_still_takes_a_bare_float_and_a_headers_dict() -> None:
    """Five modules call ``make_client(cfg.data.http_timeout_seconds, _HEADERS)``
    positionally. Changing that signature would break all of them at once, and
    the breakage would be an import-time TypeError in the middle of a tick."""
    with http.make_client(7.5, {"X-Trace": "1"}) as client:
        assert client.headers["X-Trace"] == "1"
        assert client.headers["User-Agent"] == http.BROWSER_UA
        assert client.timeout.connect == 7.5
        assert client.timeout.read == 7.5


def test_a_bare_float_is_spread_across_all_four_phases_not_reinterpreted() -> None:
    """The legacy scalar meant "15 seconds for everything". It still does — a
    caller's effective budget must not shrink because this module grew a richer
    type underneath them."""
    t = http.Timeouts.coerce(15.0)
    assert (t.connect, t.read, t.write, t.pool) == (15.0, 15.0, 15.0, 15.0)


def test_timeouts_can_be_split_per_call_site() -> None:
    with http.make_client(http.Timeouts(connect=1.0, read=30.0)) as client:
        assert client.timeout.connect == 1.0
        assert client.timeout.read == 30.0


def test_the_client_verifies_against_the_os_trust_store() -> None:
    """The Zscaler finding: certifi's bundle has no entry for the proxy's
    private CA, so every call fails with CERTIFICATE_VERIFY_FAILED. The context
    must be the OS one, and it must be the *cached* one — building it reads the
    whole root store from disk."""
    assert http.ssl_context() is http.ssl_context()
    assert http.ssl_context().verify_mode is not None


# ---------------------------------------------------------------------------
# Retry classification — the 422
# ---------------------------------------------------------------------------


def test_a_422_is_not_retried_as_if_it_were_rate_limiting() -> None:
    """The recorded investigation (2026-09-19): Arctic Shift answers some
    requests with ``422 {"error": "Timeout. Maybe slow down a bit"}``, and the
    message invites exactly the wrong response. It reproduces on the first cold
    request of a fresh process; 0.6s, 1.0s, 2.0s and 2.5s spacing all failed
    identically, as did a 105-second backoff. It is a ~3s server-side query
    timeout on a range holding too many rows, so the identical request can
    never succeed and every retry costs 3.3 seconds of tick budget."""
    assert http.is_retryable_status(422) is False
    assert 422 in http.NON_RETRYABLE_STATUSES

    clock = FakeClock()
    sleep = FakeSleep(clock)
    seen: list[httpx.Request] = []
    with http.make_client(transport=responder([422], seen=seen)) as client:
        outcome = http.execute(
            client, "GET", URL, clock=clock, sleep=sleep, retry=http.RetryPolicy()
        )

    assert len(seen) == 1, "a 422 was retried; see the 2026-09-19 investigation"
    assert outcome.attempt_count == 1
    assert outcome.response is not None
    assert outcome.response.status_code == 422
    assert sleep.calls == []


def test_a_429_is_retried_because_that_one_really_is_rate_limiting() -> None:
    clock = FakeClock()
    sleep = FakeSleep(clock)
    seen: list[httpx.Request] = []
    with http.make_client(transport=responder([429, 200], seen=seen)) as client:
        outcome = http.execute(client, "GET", URL, clock=clock, sleep=sleep)

    assert len(seen) == 2
    assert outcome.succeeded


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
def test_the_retryable_set_is_an_allow_list(status: int) -> None:
    assert http.is_retryable_status(status) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 410, 422, 451, 418])
def test_everything_else_is_terminal_including_statuses_we_have_never_seen(
    status: int,
) -> None:
    """Allow-list, not deny-list. Guessing wrong in the other direction burns a
    tick budget on something that will never succeed, which is the 422 loop."""
    assert http.is_retryable_status(status) is False


def test_a_post_is_not_retried_by_default() -> None:
    """Audit C11's duplicate-submission row in one line: a retried POST to a
    swap endpoint is a second order."""
    seen: list[httpx.Request] = []
    clock = FakeClock()
    with http.make_client(transport=responder([503], seen=seen)) as client:
        http.execute(client, "POST", URL, clock=clock, sleep=FakeSleep(clock))

    assert len(seen) == 1


def test_a_post_may_opt_in_when_the_caller_owns_an_idempotency_key() -> None:
    seen: list[httpx.Request] = []
    clock = FakeClock()
    with http.make_client(transport=responder([503, 200], seen=seen)) as client:
        outcome = http.execute(
            client,
            "POST",
            URL,
            idempotent=True,
            clock=clock,
            sleep=FakeSleep(clock),
        )

    assert len(seen) == 2
    assert outcome.succeeded


# ---------------------------------------------------------------------------
# Retry budget — both bounds
# ---------------------------------------------------------------------------


def test_the_attempt_count_is_capped() -> None:
    seen: list[httpx.Request] = []
    clock = FakeClock()
    policy = http.RetryPolicy(max_attempts=3, total_budget_seconds=1000.0)
    with http.make_client(transport=responder([503], seen=seen)) as client:
        outcome = http.execute(
            client, "GET", URL, retry=policy, clock=clock, sleep=FakeSleep(clock)
        )

    assert len(seen) == 3
    assert outcome.attempt_count == 3
    assert outcome.stopped_by == "attempts"


def test_the_total_elapsed_time_is_capped_independently_of_the_attempt_count() -> None:
    """The bound that a max_attempts alone does not give you. Each attempt here
    burns 9 seconds of *its own* latency — a slow 503, not a fast one — so a
    generous attempt cap would still stall a 60-second tick. The budget has to
    count attempt latency, not just sleeps."""
    clock = FakeClock()
    sleep = FakeSleep(clock)
    seen: list[httpx.Request] = []
    policy = http.RetryPolicy(max_attempts=50, total_budget_seconds=20.0)
    transport = responder([503], clock=clock, latency=9.0, seen=seen)

    with http.make_client(transport=transport) as client:
        outcome = http.execute(client, "GET", URL, retry=policy, clock=clock, sleep=sleep)

    assert outcome.stopped_by == "budget"
    assert outcome.elapsed_seconds <= policy.total_budget_seconds
    assert len(seen) < policy.max_attempts
    assert outcome.attempt_count >= 2


class MaxJitter:
    """An RNG that always draws the top of the range.

    Used where the property under test is the *bound*, not the randomness.
    Seeding ``random.Random`` would still leave the draw free to land under the
    remaining budget, which makes the test pass for the wrong reason roughly
    four times in five.
    """

    def uniform(self, low: float, high: float) -> float:
        return high


def test_the_budget_stops_the_loop_before_a_sleep_that_would_overrun_it() -> None:
    """Checked *before* the sleep rather than after. Sleeping past the budget
    and then noticing is how a 60-second tick becomes a 75-second tick."""
    clock = FakeClock()
    sleep = FakeSleep(clock)
    policy = http.RetryPolicy(
        max_attempts=10,
        total_budget_seconds=1.0,
        backoff_base_seconds=5.0,
        backoff_max_seconds=5.0,
    )
    with http.make_client(transport=responder([503])) as client:
        outcome = http.execute(
            client,
            "GET",
            URL,
            retry=policy,
            clock=clock,
            sleep=sleep,
            rng=MaxJitter(),  # type: ignore[arg-type]
        )

    assert sleep.calls == []
    assert outcome.stopped_by == "budget"
    assert clock.now <= 1.0


def test_a_transport_error_is_retried_and_then_raised_by_request() -> None:
    clock = FakeClock()
    transport = raiser(httpx.ConnectTimeout("connect timed out"))
    with (
        http.make_client(transport=transport) as client,
        pytest.raises(http.RequestFailed) as excinfo,
    ):
        http.request(
            client,
            "GET",
            URL,
            retry=http.RetryPolicy(max_attempts=2),
            clock=clock,
            sleep=FakeSleep(clock),
        )

    assert excinfo.value.outcome.attempt_count == 2


def test_a_malformed_url_is_not_retried() -> None:
    """``InvalidURL`` is a programming error. Retrying it burns budget to
    re-raise the identical exception."""
    clock = FakeClock()
    transport = raiser(httpx.UnsupportedProtocol("no scheme"))
    with http.make_client(transport=transport) as client:
        outcome = http.execute(
            client,
            "GET",
            URL,
            retry=http.RetryPolicy(max_attempts=5),
            clock=clock,
            sleep=FakeSleep(clock),
        )

    assert outcome.attempt_count == 1
    assert outcome.stopped_by == "terminal"


def test_a_retry_after_longer_than_the_cap_stops_rather_than_sleeping() -> None:
    """Some hosts answer 429 with minutes. A tick cannot wait minutes, and
    sleeping "just this once" is how one slow source eats the whole schedule."""
    clock = FakeClock()
    sleep = FakeSleep(clock)
    transport = responder([429], headers={"Retry-After": "600"})
    with http.make_client(transport=transport) as client:
        outcome = http.execute(client, "GET", URL, clock=clock, sleep=sleep)

    assert outcome.stopped_by == "retry_after_too_long"
    assert sleep.calls == []


def test_a_missing_retry_after_is_none_and_not_zero() -> None:
    """ "The server did not tell us" and "the server said retry immediately" are
    different instructions. Missing is never zero."""
    assert http.retry_after_seconds(httpx.Response(429)) is None
    assert (
        http.retry_after_seconds(httpx.Response(429, headers={"Retry-After": "0"})) == 0.0
    )
    assert (
        http.retry_after_seconds(httpx.Response(429, headers={"Retry-After": "soon"}))
        is None
    )


# ---------------------------------------------------------------------------
# Backoff and jitter
# ---------------------------------------------------------------------------


def test_backoff_grows_exponentially_up_to_a_cap() -> None:
    policy = http.RetryPolicy(
        backoff_base_seconds=0.25, backoff_multiplier=3.0, backoff_max_seconds=4.0
    )
    assert policy.backoff_ceiling(1) == pytest.approx(0.25)
    assert policy.backoff_ceiling(2) == pytest.approx(0.75)
    assert policy.backoff_ceiling(3) == pytest.approx(2.25)
    assert policy.backoff_ceiling(4) == pytest.approx(4.0)  # capped
    assert policy.backoff_ceiling(9) == pytest.approx(4.0)


def test_the_backoff_actually_carries_jitter() -> None:
    """Nine coins polled on one tick fail on one tick. Identical sleeps would
    have them retry on the same millisecond — a herd we would be aiming at a
    free public API, converting one bad second into a rate-limit ban.

    The assertion is on *spread*, not on a specific draw: the point is that two
    callers do not agree, not that any particular number comes out."""
    policy = http.RetryPolicy(backoff_base_seconds=1.0, backoff_max_seconds=1.0)
    draws = {policy.backoff(1, random.Random(seed)) for seed in range(64)}

    assert len(draws) > 32, "backoff is deterministic; the herd is not decorrelated"
    assert all(0.0 <= d <= 1.0 for d in draws)


def test_jitter_spans_the_whole_interval_rather_than_blurring_the_middle() -> None:
    """Full jitter, uniform over ``[0, ceiling]``. Halving the range to look
    less random reintroduces the correlation the jitter exists to remove."""
    policy = http.RetryPolicy(backoff_base_seconds=1.0, backoff_max_seconds=1.0)
    draws = [policy.backoff(1, random.Random(seed)) for seed in range(400)]

    assert min(draws) < 0.05
    assert max(draws) > 0.95


def test_the_recorded_backoff_is_what_was_actually_slept() -> None:
    clock = FakeClock()
    sleep = FakeSleep(clock)
    policy = http.RetryPolicy(max_attempts=2, total_budget_seconds=100.0)
    with http.make_client(transport=responder([503, 200])) as client:
        outcome = http.execute(client, "GET", URL, retry=policy, clock=clock, sleep=sleep)

    assert len(sleep.calls) == 1
    assert outcome.attempts[0].backoff_seconds == pytest.approx(sleep.calls[0])


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


def test_the_breaker_opens_after_n_consecutive_failures() -> None:
    clock = FakeClock()
    breaker = http.CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0, clock=clock)

    for _ in range(2):
        breaker.record_failure("api.example.com", "HTTP 503")
    assert breaker.status("api.example.com").state is http.BreakerState.CLOSED

    breaker.record_failure("api.example.com", "HTTP 503")
    assert breaker.status("api.example.com").state is http.BreakerState.OPEN
    assert breaker.open_hosts == ("api.example.com",)


def test_an_open_breaker_fails_fast_without_making_a_request() -> None:
    """The measurement: when a host is down, a nine-coin sweep makes nine calls
    that each burn the full retry budget to rediscover the same fact. At the
    defaults that is minutes inside a 60-second tick."""
    clock = FakeClock()
    sleep = FakeSleep(clock)
    seen: list[httpx.Request] = []
    breaker = http.CircuitBreaker(failure_threshold=2, cooldown_seconds=60.0, clock=clock)

    with http.make_client(transport=responder([503], seen=seen)) as client:
        http.execute(
            client,
            "GET",
            URL,
            breaker=breaker,
            retry=http.RetryPolicy(max_attempts=2),
            clock=clock,
            sleep=sleep,
        )
        calls_before = len(seen)
        outcome = http.execute(
            client, "GET", URL, breaker=breaker, clock=clock, sleep=sleep
        )

    assert len(seen) == calls_before, "a request was made through an open breaker"
    assert isinstance(outcome.error, http.CircuitOpen)
    assert outcome.stopped_by == "circuit_open"
    assert outcome.elapsed_seconds == 0.0


def test_request_raises_circuit_open_so_callers_can_tell_it_from_a_timeout() -> None:
    """A timeout is a reason to try the next source. An open breaker is a reason
    to mark the whole source unhealthy and refuse entries — audit §11,
    "Data vendor/API rate limit → no data-dependent trade"."""
    clock = FakeClock()
    breaker = http.CircuitBreaker(failure_threshold=1, clock=clock)
    breaker.record_failure("api.dexscreener.com", "HTTP 503")

    with (
        http.make_client(transport=responder([200])) as client,
        pytest.raises(http.CircuitOpen),
    ):
        http.request(client, "GET", URL, breaker=breaker, clock=clock)


def test_the_breaker_closes_after_the_cooldown_when_the_probe_succeeds() -> None:
    clock = FakeClock()
    breaker = http.CircuitBreaker(failure_threshold=1, cooldown_seconds=30.0, clock=clock)
    breaker.record_failure("api.dexscreener.com", "HTTP 503")
    assert breaker.status("api.dexscreener.com").state is http.BreakerState.OPEN

    clock.advance(30.0)
    assert breaker.status("api.dexscreener.com").state is http.BreakerState.HALF_OPEN

    with http.make_client(transport=responder([200])) as client:
        outcome = http.execute(
            client, "GET", URL, breaker=breaker, clock=clock, sleep=FakeSleep(clock)
        )

    assert outcome.succeeded
    assert breaker.status("api.dexscreener.com").state is http.BreakerState.CLOSED


def test_half_open_admits_exactly_one_probe() -> None:
    """A recovering host must not be handed the backlog that knocked it over."""
    clock = FakeClock()
    breaker = http.CircuitBreaker(failure_threshold=1, cooldown_seconds=10.0, clock=clock)
    breaker.record_failure("h", "boom")
    clock.advance(10.0)

    assert breaker.allow("h") is True
    assert breaker.allow("h") is False


def test_a_failed_half_open_probe_restarts_the_cooldown() -> None:
    clock = FakeClock()
    breaker = http.CircuitBreaker(failure_threshold=1, cooldown_seconds=10.0, clock=clock)
    breaker.record_failure("h", "boom")
    clock.advance(10.0)
    breaker.allow("h")
    breaker.record_failure("h", "still down")

    assert breaker.status("h").state is http.BreakerState.OPEN
    assert breaker.status("h").cooldown_remaining_seconds == pytest.approx(10.0)


def test_a_success_clears_the_consecutive_failure_count() -> None:
    """*Consecutive*, not cumulative. A host that fails once an hour is not a
    host that should ever be fenced off."""
    breaker = http.CircuitBreaker(failure_threshold=3, clock=FakeClock())
    breaker.record_failure("h", "x")
    breaker.record_failure("h", "x")
    breaker.record_success("h")
    breaker.record_failure("h", "x")

    assert breaker.status("h").state is http.BreakerState.CLOSED
    assert breaker.status("h").consecutive_failures == 1


def test_a_404_does_not_count_against_the_breaker() -> None:
    """A 4xx that is not a rate limit is a *successful* conversation: the server
    is up and answered. Counting it as a failure would open DexScreener because
    one mint is unknown — the 422 mistake in a different costume."""
    clock = FakeClock()
    breaker = http.CircuitBreaker(failure_threshold=2, clock=clock)
    with http.make_client(transport=responder([404])) as client:
        for _ in range(5):
            http.execute(
                client, "GET", URL, breaker=breaker, clock=clock, sleep=FakeSleep(clock)
            )

    assert breaker.status("api.dexscreener.com").state is http.BreakerState.CLOSED


def test_the_breaker_state_is_readable_for_a_data_health_check() -> None:
    """Risk reads this. A host that has never been called is *absent* rather
    than present-and-healthy: "we did not call it" and "we called it and it was
    fine" are different facts and only the second may let an entry through."""
    clock = FakeClock()
    breaker = http.CircuitBreaker(failure_threshold=1, clock=clock)
    breaker.record_failure("api.geckoterminal.com", "HTTP 503")

    snapshot = breaker.snapshot()
    assert set(snapshot) == {"api.geckoterminal.com"}
    assert snapshot["api.geckoterminal.com"].healthy is False
    assert snapshot["api.geckoterminal.com"].last_error == "HTTP 503"
    assert "lite-api.jup.ag" not in snapshot


def test_the_breaker_is_per_host_not_global() -> None:
    clock = FakeClock()
    breaker = http.CircuitBreaker(failure_threshold=1, clock=clock)
    breaker.record_failure("api.dexscreener.com", "HTTP 503")

    assert breaker.allow("api.dexscreener.com") is False
    assert breaker.allow("lite-api.jup.ag") is True


# ---------------------------------------------------------------------------
# Request IDs and logging
# ---------------------------------------------------------------------------


def test_every_attempt_of_one_call_carries_the_same_request_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock()
    caplog.set_level(logging.INFO)
    with http.make_client(transport=responder([503, 200])) as client:
        outcome = http.execute(
            client,
            "GET",
            URL,
            retry=http.RetryPolicy(max_attempts=2),
            clock=clock,
            sleep=FakeSleep(clock),
        )

    lines = [r.getMessage() for r in caplog.records if "http attempt" in r.getMessage()]
    assert len(lines) == 2
    assert all(f"rid={outcome.request_id}" in line for line in lines)
    assert "n=1" in lines[0]
    assert "n=2" in lines[1]


def test_two_calls_get_different_request_ids() -> None:
    assert http.new_request_id() != http.new_request_id()


def test_the_log_line_carries_status_and_latency(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock()
    caplog.set_level(logging.INFO)
    transport = responder([200], clock=clock, latency=0.25)
    with http.make_client(transport=transport) as client:
        http.execute(client, "GET", URL, clock=clock, sleep=FakeSleep(clock))

    (line,) = [r.getMessage() for r in caplog.records if "http attempt" in r.getMessage()]
    assert "status=200" in line
    assert "latency=0.250s" in line


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def test_auth_headers_are_redacted_unconditionally() -> None:
    """The environment on this machine holds a live ``Ocp-Apim-Subscription-Key``.
    Redaction is by header *name* and takes no opt-in flag, because a per-site
    flag fails the first time somebody adds a source without knowing the rule —
    and that failure is a live key in a file pasted into a bug report."""
    redacted = http.redact_headers(
        {
            "Authorization": "Bearer sk-live-do-not-log",
            "Ocp-Apim-Subscription-Key": "0123456789abcdef",
            "X-Api-Key": "jup-secret",
            "Cookie": "session=abc",
            "User-Agent": http.BROWSER_UA,
        }
    )

    assert redacted["Authorization"] == "<redacted>"
    assert redacted["Ocp-Apim-Subscription-Key"] == "<redacted>"
    assert redacted["X-Api-Key"] == "<redacted>"
    assert redacted["Cookie"] == "<redacted>"
    assert redacted["User-Agent"] == http.BROWSER_UA
    assert "sk-live-do-not-log" not in str(redacted)
    assert "0123456789abcdef" not in str(redacted)


def test_header_matching_is_case_insensitive() -> None:
    """httpx normalizes header case; a redaction list that did not would leak
    the first time a source spelled it ``AUTHORIZATION``."""
    assert http.redact_headers({"aUtHoRiZaTiOn": "secret"})["aUtHoRiZaTiOn"] == "<redacted>"


def test_a_secret_header_never_reaches_a_log_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock()
    caplog.set_level(logging.INFO)
    with http.make_client(transport=responder([200])) as client:
        http.execute(
            client,
            "GET",
            URL,
            headers={"Ocp-Apim-Subscription-Key": "LIVE-KEY-MATERIAL"},
            clock=clock,
            sleep=FakeSleep(clock),
        )

    assert "LIVE-KEY-MATERIAL" not in caplog.text
    assert "<redacted>" in caplog.text


def test_a_credential_in_a_query_string_is_scrubbed() -> None:
    """A URL is the easiest place for a secret to end up by accident, and the
    cost of over-redacting a query parameter in a log is zero."""
    safe = http.redact_url("https://api.jup.ag/swap/v1/quote?api_key=SECRET&amount=1000")

    assert "SECRET" not in safe
    assert "amount=1000" in safe


def test_a_declared_secret_hostname_is_logged_as_a_stable_digest() -> None:
    """The internal gateway names infrastructure that is not meant to be
    discoverable. A digest still distinguishes two hosts across log lines,
    which is all a latency investigation needs."""
    policy = http.LogPolicy(secret_hosts=frozenset({"gateway.internal.example"}))
    one = http.redact_url("https://gateway.internal.example/v1/messages", policy=policy)
    two = http.redact_url("https://gateway.internal.example/v1/models", policy=policy)

    assert "gateway.internal.example" not in one
    assert one.split("/v1")[0] == two.split("/v1")[0]
    assert http.redact_url(URL, policy=policy).startswith("https://api.dexscreener.com")


def test_a_failing_call_logs_the_error_without_the_secret(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock()
    caplog.set_level(logging.INFO)
    transport = raiser(httpx.ReadTimeout("read timed out"))
    with http.make_client(transport=transport) as client:
        http.execute(
            client,
            "GET",
            URL,
            headers={"Authorization": "Bearer live-token"},
            retry=http.RetryPolicy(max_attempts=1),
            clock=clock,
            sleep=FakeSleep(clock),
        )

    assert "live-token" not in caplog.text
    assert "ReadTimeout" in caplog.text


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


def test_the_outcome_records_every_attempt_including_the_failed_ones() -> None:
    """ "We got a price" and "we got a price on the third attempt after eleven
    seconds" are different facts about data health, and only the second one
    explains a late tick."""
    clock = FakeClock()
    transport = responder([503, 503, 200], clock=clock, latency=0.5)
    with http.make_client(transport=transport) as client:
        outcome = http.execute(
            client,
            "GET",
            URL,
            retry=http.RetryPolicy(max_attempts=3, total_budget_seconds=100.0),
            clock=clock,
            sleep=FakeSleep(clock),
        )

    assert outcome.succeeded
    assert [a.status for a in outcome.attempts] == [503, 503, 200]
    assert all(a.latency_seconds == pytest.approx(0.5) for a in outcome.attempts)
    assert outcome.stopped_by == "ok"


def test_a_404_is_returned_rather_than_raised() -> None:
    """Whether a 404 is an error is the caller's question: ``market.py`` reads
    it as "no pair for this mint", which is data, not a failure."""
    clock = FakeClock()
    with http.make_client(transport=responder([404])) as client:
        response = http.request(client, "GET", URL, clock=clock, sleep=FakeSleep(clock))

    assert response.status_code == 404
