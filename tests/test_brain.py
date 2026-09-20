"""Tests for the advisory call and, mostly, for what it refuses to believe.

The central assertion in this file is negative: ``_validate`` never repairs
anything. Audit C6 named the old ``_normalize`` as a critical finding, and one
case shows why a repair function is worse than no function at all — it tested
``size < 0.0``, and every comparison against NaN is false, so a NaN size was not
negative, not positive, not out of range. It passed the repair, passed
``Field(ge=0.0)``, passed every risk comparison for the same reason, and arrived
at the broker as an order size.

So each of the six defects below discards the **whole** decision. There is no
partial acceptance and no inference about what the model meant, and
``TestNoRepair`` asserts the repair function is gone rather than merely unused.

Removed from the previous version of this file, with reasons:

* Every test of ``_normalize``'s corrections (drop-unknown, keep-first-duplicate,
  fill-missing-with-HOLD, clamp-negative-to-zero) — those behaviours are the
  audit finding. Each has a mirror-image test here asserting a rejection.
* ``decide()`` tests — renamed ``advise()`` and returning ``AdvisoryDecision``
  (audit C6: the model advises, it does not decide).

Kept and adapted: the thinking-block tests, the SDK-shape tests (which read the
installed ``anthropic`` package rather than calling it), and the cost and
cache-rate arithmetic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import anthropic
import httpx
import pytest

from memetrader import brain
from memetrader.brain import (
    BrainError,
    ModelCallError,
    ModelOutputError,
    Usage,
    advise,
)
from memetrader.types import AdvisoryAction, AdvisoryDecision, ValidationError
from test_prompts import (  # reuse the evidence fixtures; same types, same shapes
    SYMBOLS,
    FakeCadence,
    FakeRisk,
    _evidence,
    _portfolio,
)


@dataclass(frozen=True)
class FakeModel:
    name: str = "claude-opus-5"
    effort: str = "high"
    max_tokens: int = 4096

    def cost_usd(self, inp: int, out: int, cache_read: int, cache_write: int) -> float:
        # $1 / $5 / $0.10 / $1.25 per Mtok, chosen so the arithmetic is readable.
        return (inp * 1.0 + out * 5.0 + cache_read * 0.10 + cache_write * 1.25) / 1_000_000


def action(symbol: str, act: str = "HOLD", size: float = 0.0, **kw) -> AdvisoryAction:
    return AdvisoryAction(
        action=act,  # type: ignore[arg-type]
        symbol=symbol,
        size_usd=size,
        confidence=kw.get("confidence", 0.5),
        reasoning=kw.get("reasoning", "RSI14 at 58.2 and nothing else moved."),
    )


def decision(*actions: AdvisoryAction, read: str = "Quiet tape.") -> AdvisoryDecision:
    return AdvisoryDecision(market_read=read, actions=list(actions))


def good() -> AdvisoryDecision:
    return decision(action("BONK", "BUY", 50.0), action("WIF"))


class _Block:
    def __init__(self, kind: str, **kw) -> None:
        self.type = kind
        for k, v in kw.items():
            setattr(self, k, v)


class _Usage:
    def __init__(self, **kw) -> None:
        self.input_tokens = kw.get("input_tokens", 100)
        self.output_tokens = kw.get("output_tokens", 200)
        self.cache_read_input_tokens = kw.get("cache_read_input_tokens", 900)
        self.cache_creation_input_tokens = kw.get("cache_creation_input_tokens", 0)


_DEFAULT = object()


class _Response:
    def __init__(
        self,
        parsed: AdvisoryDecision | None = None,
        *,
        content: list[Any] | None = None,
        stop_reason: str = "end_turn",
        usage: Any = _DEFAULT,
        stop_details: Any | None = None,
    ) -> None:
        self.parsed_output = parsed
        self.content = content or []
        self.stop_reason = stop_reason
        self.usage = _Usage() if usage is _DEFAULT else usage
        self.stop_details = stop_details


class FakeClient:
    """Stands in for ``anthropic.Anthropic``. Records the call it was given."""

    def __init__(self, response: Any = None, raises: Exception | None = None) -> None:
        self._response = response if response is not None else _Response(good())
        self._raises = raises
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return self._response


def run(client: FakeClient, **kw):
    params = {
        "symbols": SYMBOLS,
        "model": FakeModel(),
        "risk": FakeRisk(),
        "cadence": FakeCadence(),
        "starting_cash_usd": 1000.0,
        "client": client,
    }
    params.update(kw)
    return advise(_evidence(), _portfolio(), [], [], **params)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_a_valid_output_comes_back_intact(self):
        result, usage = run(FakeClient())
        assert [a.symbol for a in result.actions] == list(SYMBOLS)
        assert result.actions[0].action == "BUY"
        assert usage.output_tokens == 200

    def test_actions_are_reordered_to_the_configured_universe(self):
        out_of_order = decision(action("WIF"), action("BONK", "SELL", 10.0))
        result, _ = run(FakeClient(_Response(out_of_order)))
        assert [a.symbol for a in result.actions] == ["BONK", "WIF"]

    def test_a_lowercase_symbol_is_normalized_not_rejected(self):
        # Case is a formatting difference, not a claim about a different coin.
        mixed = decision(action("bonk", "BUY", 5.0), action("WIF"))
        result, _ = run(FakeClient(_Response(mixed)))
        assert result.actions[0].symbol == "BONK"

    def test_the_prompt_fingerprint_rides_on_usage(self):
        _, usage = run(FakeClient())
        assert len(usage.prompt_fingerprint) == 16


class TestWholeOutputDiscarded:
    """Audit C6: "reject whole invalid output. No model repair may create an order.\""""

    def _reject(self, bad: AdvisoryDecision) -> str:
        with pytest.raises(ModelOutputError) as exc:
            run(FakeClient(_Response(bad)))
        return str(exc.value)

    def test_a_hallucinated_symbol_discards_everything(self):
        bad = decision(
            action("BONK", "BUY", 50.0), action("WIF"), action("SOLANAMOON", "BUY", 900.0)
        )
        message = self._reject(bad)
        assert "SOLANAMOON" in message
        assert "discarding the whole output" in message

    def test_a_duplicate_symbol_discards_everything(self):
        bad = decision(
            action("BONK", "BUY", 50.0), action("BONK", "SELL", 50.0), action("WIF")
        )
        assert "two actions for BONK" in self._reject(bad)

    def test_a_missing_symbol_discards_everything(self):
        # The old code filled this with HOLD, converting an incomplete answer
        # into a confident flat one — and a HOLD on a position that should have
        # been exited is not a null action.
        assert "WIF" in self._reject(decision(action("BONK", "BUY", 50.0)))

    def test_a_negative_size_discards_everything(self):
        bad = decision(action("BONK", "BUY", 0.0), action("WIF"))
        bad.actions[0] = AdvisoryAction.model_construct(
            action="BUY", symbol="BONK", size_usd=-25.0, confidence=0.5, reasoning="x"
        )
        assert "negative size_usd" in self._reject(bad)

    def test_a_hold_with_a_non_zero_size_discards_everything(self):
        bad = decision(action("BONK", "HOLD", 42.0), action("WIF"))
        assert "contradict each other" in self._reject(bad)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_size_discards_everything(self, value: float):
        """The case the old repair could not see.

        ``nan < 0.0`` is false, ``nan > max_notional`` is false, and
        ``Field(ge=0.0)`` does not reject it either. A NaN walks the whole
        system precisely because nothing about it is ever true, so the check has
        to be ``math.isfinite`` and not a comparison.
        """
        bad = decision(action("BONK", "BUY", 0.0), action("WIF"))
        bad.actions[0] = AdvisoryAction.model_construct(
            action="BUY", symbol="BONK", size_usd=value, confidence=0.5, reasoning="x"
        )
        message = self._reject(bad)
        assert "unusable size_usd" in message or "negative size_usd" in message

    def test_the_schema_itself_also_rejects_nan(self):
        # Defense in depth: the field validator on AdvisoryAction is the first
        # gate, _validate is the second. Neither is sufficient alone — the
        # schema can be bypassed by model_construct, and the validator only runs
        # on data that reached it.
        with pytest.raises((ValidationError, ValueError)):
            AdvisoryAction(
                action="BUY",
                symbol="BONK",
                size_usd=math.nan,
                confidence=0.5,
                reasoning="x",
            )

    def test_a_discard_is_a_brain_error_so_callers_fall_back(self):
        bad = decision(action("BONK", "HOLD", 1.0), action("WIF"))
        with pytest.raises(BrainError):
            run(FakeClient(_Response(bad)))


class TestNoRepair:
    def test_the_repair_function_is_gone(self):
        assert not hasattr(brain, "_normalize")

    def test_nothing_is_exported_that_produces_an_order(self):
        # advise() returns advice. The order types are not even imported here.
        assert "OrderIntent" not in dir(brain)
        assert brain.__all__ == [
            "BrainError",
            "ModelCallError",
            "ModelOutputError",
            "Usage",
            "advise",
        ]


class TestApiFailures:
    def _raise(self, exc: Exception) -> ModelCallError:
        with pytest.raises(ModelCallError) as caught:
            run(FakeClient(raises=exc))
        return caught.value

    def test_a_rate_limit_is_retryable(self):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(429, request=request)
        error = self._raise(
            anthropic.RateLimitError("slow down", response=response, body=None)
        )
        assert error.retryable is True

    def test_a_500_is_retryable_and_a_400_is_not(self):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        for status, retryable in ((500, True), (400, False)):
            response = httpx.Response(status, request=request)
            error = self._raise(
                anthropic.APIStatusError("boom", response=response, body=None)
            )
            assert error.retryable is retryable, status

    def test_a_connection_failure_is_retryable(self):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        error = self._raise(anthropic.APIConnectionError(request=request))
        assert error.retryable is True

    def test_an_api_failure_never_returns_a_hold(self):
        """A skipped tick and a considered HOLD are different events.

        A decision log that renders them identically cannot be used to debug a
        bad run, which is why this raises instead of returning something
        plausible.
        """
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(503, request=request)
        with pytest.raises(ModelCallError):
            run(
                FakeClient(
                    raises=anthropic.APIStatusError("down", response=response, body=None)
                )
            )

    def test_a_refusal_is_an_output_error(self):
        response = _Response(
            None, stop_reason="refusal", stop_details=_Block("refusal", category="policy")
        )
        with pytest.raises(ModelOutputError) as exc:
            run(FakeClient(response))
        assert "refused" in str(exc.value)

    def test_empty_structured_output_is_an_output_error(self):
        with pytest.raises(ModelOutputError):
            run(FakeClient(_Response(None, stop_reason="max_tokens")))


class TestThinking:
    def test_thinking_blocks_are_concatenated(self):
        response = _Response(
            good(),
            content=[
                _Block("thinking", thinking="first thought"),
                _Block("text", text="ignored"),
                _Block("thinking", thinking="second thought"),
            ],
        )
        _, usage = run(FakeClient(response))
        assert usage.thinking == "first thought\n\nsecond thought"

    def test_a_redacted_block_leaves_a_marker_not_ciphertext(self):
        # "The model did not think" and "the model thought and we may not see
        # it" are different events, and the ciphertext costs disk for bytes
        # nobody can read.
        response = _Response(good(), content=[_Block("redacted_thinking", data="x" * 5000)])
        _, usage = run(FakeClient(response))
        assert usage.thinking == brain._REDACTED_THINKING
        assert "xxxx" not in (usage.thinking or "")

    def test_no_thinking_blocks_means_none(self):
        _, usage = run(FakeClient(_Response(good(), content=[_Block("text", text="hi")])))
        assert usage.thinking is None


class TestUsage:
    def test_total_input_includes_both_cache_counts(self):
        usage = Usage(
            input_tokens=100,
            cache_read_input_tokens=900,
            cache_creation_input_tokens=50,
        )
        assert usage.total_input_tokens == 1050
        assert usage.cache_hit_rate == pytest.approx(900 / 1050)

    def test_cost_prices_cache_creation_too(self):
        """It did not, once, and that understated exactly the call a cache
        regression makes you pay over and over."""
        usage = Usage(
            input_tokens=1_000_000,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=1_000_000,
        )
        assert usage.cost_usd(FakeModel()) == pytest.approx(1.0 + 1.25)

    def test_a_response_without_usage_does_not_crash(self):
        usage = Usage.from_response(_Response(good(), usage=None))
        assert usage.total_input_tokens == 0
        assert usage.cache_hit_rate == 0.0

    def test_a_cache_miss_is_logged_as_a_warning(self, caplog):
        response = _Response(good(), usage=_Usage(cache_read_input_tokens=0))
        with caplog.at_level("WARNING"):
            run(FakeClient(response))
        assert "prompt cache miss" in caplog.text


class TestSdkShape:
    """Read from the installed ``anthropic`` 1.7.0. No network, no billing."""

    def test_the_call_passes_adaptive_thinking_with_no_token_budget(self):
        client = FakeClient()
        run(client)
        thinking = client.calls[0]["thinking"]
        assert thinking == {"type": "adaptive"}
        # budget_tokens is a key of the *enabled* variant only; depth here is
        # controlled by effort.
        assert "budget_tokens" not in thinking

    def test_adaptive_is_a_real_thinking_variant(self):
        from anthropic.types import thinking_config_param as tcp

        names = {n for n in dir(tcp) if "Adaptive" in n}
        assert names, dir(tcp)

    def test_effort_is_a_key_of_output_config(self):
        from anthropic.types.output_config_param import OutputConfigParam

        assert "effort" in OutputConfigParam.__annotations__

    def test_the_call_passes_effort_and_the_advisory_schema_together(self):
        client = FakeClient()
        run(client)
        call = client.calls[0]
        assert call["output_config"] == {"effort": "high"}
        assert call["output_format"] is AdvisoryDecision

    def test_the_system_prompt_is_a_block_list_with_cache_breakpoints(self):
        client = FakeClient()
        run(client)
        system = client.calls[0]["system"]
        assert isinstance(system, list)
        assert all(b["cache_control"] == {"type": "ephemeral"} for b in system)

    def test_the_frozen_prefix_is_identical_across_two_calls(self):
        client = FakeClient()
        run(client)
        run(client)
        assert client.calls[0]["system"] == client.calls[1]["system"]
        # ...and the volatile half is what carries the per-tick state.
        assert "CURRENT TIME" in client.calls[0]["messages"][0]["content"]
