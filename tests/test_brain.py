"""Thinking capture, the request shape, and the cost arithmetic in ``brain.py``.

Nothing here touches the network. The test that would prove the request shape
end to end is ``tests/test_prompts.py::test_live_call_hits_the_prompt_cache``,
and on 2026-09-19 it could not be run: no ``ANTHROPIC_API_KEY`` was reachable
from the environment or from a ``.env``. So the request-shape tests below are
written against the installed SDK's *own* type definitions rather than against
anybody's memory of the API — they cannot prove the server accepts the call,
but they will fail the moment an SDK upgrade moves ``effort`` out of
``output_config`` or drops the adaptive thinking variant, which is the drift
that would otherwise be discovered by a 400 in production.

Response stubs follow the style of the ``Boom`` client in test_prompts.py:
plain objects carrying exactly the attributes ``brain`` reads, so a stub can
also express shapes a real response never would.
"""

from __future__ import annotations

import inspect
import typing
from pathlib import Path
from typing import Any

import pytest

from memetrader import config as config_mod
from memetrader.brain import Usage, decide
from memetrader.types import Action, PortfolioState, TradeDecision

REPO_ROOT = Path(__file__).resolve().parents[1]

NOW = 1_764_000_000.0


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _Block:
    """One content block. ``type`` is the discriminator ``brain`` switches on."""

    def __init__(self, type: str, **fields: Any) -> None:
        self.type = type
        for key, value in fields.items():
            setattr(self, key, value)


class _SdkUsage:
    def __init__(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens


class _Response:
    def __init__(
        self,
        *,
        content: tuple[Any, ...] = (),
        usage: Any = None,
        parsed_output: Any = None,
        stop_reason: str | None = "end_turn",
    ) -> None:
        self.content = list(content)
        self.usage = usage
        self.parsed_output = parsed_output
        self.stop_reason = stop_reason


class _StubMessages:
    def __init__(self, client: _StubClient) -> None:
        self._client = client

    def parse(self, **kwargs: Any) -> Any:
        self._client.kwargs = kwargs
        return self._client.response


class _StubClient:
    """Just enough SDK surface for ``decide`` to reach ``messages.parse``,
    keeping the kwargs so a test can assert on the request we actually send."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}
        self.messages = _StubMessages(self)


def _thinking_block(text: str) -> _Block:
    return _Block("thinking", thinking=text, signature="sig")


def _text_block(text: str = "{}") -> _Block:
    return _Block("text", text=text)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cfg():
    return config_mod.load(REPO_ROOT / "config.toml")


@pytest.fixture
def empty_portfolio(cfg) -> PortfolioState:
    return PortfolioState(
        ts=NOW,
        cash_usd=cfg.starting_cash_usd,
        positions={},
        marks={},
        position_values_usd={},
        unrealized_pnl_usd=0.0,
        realized_pnl_usd=0.0,
        total_value_usd=cfg.starting_cash_usd,
        starting_cash_usd=cfg.starting_cash_usd,
    )


def _all_hold(cfg) -> TradeDecision:
    return TradeDecision(
        market_read="nothing is happening",
        actions=[
            Action(
                action="HOLD",
                symbol=symbol,
                size_usd=0.0,
                confidence=0.4,
                reasoning="flow flat",
            )
            for symbol in cfg.symbols
        ],
    )


# ---------------------------------------------------------------------------
# Thinking extraction
# ---------------------------------------------------------------------------


def test_thinking_is_read_from_the_content_blocks():
    response = _Response(
        content=(_thinking_block("BONK flow is thinning"), _text_block()),
        usage=_SdkUsage(input_tokens=10),
    )
    assert Usage.from_response(response).thinking == "BONK flow is thinning"


def test_multiple_thinking_blocks_are_joined_in_order():
    """Adaptive thinking may emit more than one block; indexing [0] would
    silently keep a prefix of the reasoning and drop the conclusion."""
    response = _Response(
        content=(
            _thinking_block("first, the liquidity trend"),
            _text_block(),
            _thinking_block("then, the sentiment z-score"),
        )
    )
    assert Usage.from_response(response).thinking == (
        "first, the liquidity trend\n\nthen, the sentiment z-score"
    )


def test_no_thinking_block_is_none():
    response = _Response(content=(_text_block(),), usage=_SdkUsage(output_tokens=5))
    assert Usage.from_response(response).thinking is None


def test_whitespace_only_thinking_is_none_not_empty_string():
    """A blank string would render as an empty 'thinking:' line in `report`,
    which reads as "the model thought nothing" rather than "no trace"."""
    response = _Response(content=(_thinking_block("   \n\t "),))
    assert Usage.from_response(response).thinking is None


def test_redacted_thinking_is_marked_rather_than_dropped():
    """The ciphertext is useless in a log, but "did not think" and "thought
    where we cannot read it" must not render identically."""
    response = _Response(content=(_Block("redacted_thinking", data="AAAA=="), _text_block()))
    thinking = Usage.from_response(response).thinking
    assert thinking is not None
    assert "redacted" in thinking
    assert "AAAA==" not in thinking, "the encrypted payload must not reach the log"


def test_redacted_and_readable_blocks_both_survive():
    response = _Response(
        content=(
            _thinking_block("the readable part"),
            _Block("redacted_thinking", data="AAAA=="),
        )
    )
    thinking = Usage.from_response(response).thinking or ""
    assert "the readable part" in thinking
    assert "redacted" in thinking


def test_unknown_block_types_are_ignored():
    response = _Response(content=(_Block("tool_use", id="t1", name="x", input={}),))
    assert Usage.from_response(response).thinking is None


def test_response_missing_content_entirely_is_tolerated():
    """``from_response`` reads defensively because the stubs in the test suite,
    and any future SDK shape, may not carry every attribute."""

    class Bare:
        usage = _SdkUsage(input_tokens=7)

    usage = Usage.from_response(Bare())
    assert usage.thinking is None
    assert usage.input_tokens == 7


def test_thinking_survives_a_response_with_no_usage_block():
    usage = Usage.from_response(_Response(content=(_thinking_block("still thought"),)))
    assert usage.thinking == "still thought"
    assert usage.input_tokens == 0


def test_the_four_token_counts_are_unaffected_by_the_new_field():
    response = _Response(
        content=(_thinking_block("x"),),
        usage=_SdkUsage(11, 22, 33, 44),
    )
    usage = Usage.from_response(response)
    assert (usage.input_tokens, usage.output_tokens) == (11, 22)
    assert (usage.cache_read_input_tokens, usage.cache_creation_input_tokens) == (33, 44)
    assert usage.total_input_tokens == 88


# ---------------------------------------------------------------------------
# decide() — the contract loop.py depends on
# ---------------------------------------------------------------------------


def test_decide_returns_exactly_two_values_and_carries_the_thinking(
    cfg, empty_portfolio
):
    """``loop.py`` unpacks two values. A third would break a file this change
    is not allowed to touch, so the thinking rides inside ``Usage``."""
    client = _StubClient(
        _Response(
            content=(_thinking_block("liquidity is draining on POPCAT"), _text_block()),
            usage=_SdkUsage(120, 400, 2_000, 0),
            parsed_output=_all_hold(cfg),
        )
    )
    returned = decide(cfg, {}, empty_portfolio, [], [], client=client)

    assert len(returned) == 2
    decision, usage = returned
    assert isinstance(usage, Usage)
    assert usage.thinking == "liquidity is draining on POPCAT"
    assert [a.symbol for a in decision.actions] == list(cfg.symbols)


def test_decide_reports_no_thinking_as_none(cfg, empty_portfolio):
    client = _StubClient(
        _Response(
            content=(_text_block(),),
            usage=_SdkUsage(120, 400, 2_000, 0),
            parsed_output=_all_hold(cfg),
        )
    )
    _, usage = decide(cfg, {}, empty_portfolio, [], [], client=client)
    assert usage.thinking is None


def test_decide_asks_for_adaptive_thinking_and_effort_in_output_config(
    cfg, empty_portfolio
):
    client = _StubClient(
        _Response(content=(_text_block(),), parsed_output=_all_hold(cfg))
    )
    decide(cfg, {}, empty_portfolio, [], [], client=client)

    sent = client.kwargs
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"] == {"effort": cfg.model.effort}
    assert sent["output_format"] is TradeDecision
    # `budget_tokens` is the thing that is believed to 400 on this model family.
    assert "budget_tokens" not in sent["thinking"]
    assert "effort" not in sent, "effort must travel inside output_config"


# ---------------------------------------------------------------------------
# The request shape, checked against the installed SDK
# ---------------------------------------------------------------------------


def _allowed_literals(annotation: Any) -> set[Any]:
    """Every literal value an annotation permits, seeing through ``| None``."""
    values: set[Any] = set()
    for arg in typing.get_args(annotation) or (annotation,):
        if arg is type(None):
            continue
        values.update(typing.get_args(arg) or (arg,))
    return values


def test_every_kwarg_we_send_is_one_messages_parse_accepts(cfg, empty_portfolio):
    from anthropic.resources.messages import Messages

    client = _StubClient(
        _Response(content=(_text_block(),), parsed_output=_all_hold(cfg))
    )
    decide(cfg, {}, empty_portfolio, [], [], client=client)

    accepted = set(inspect.signature(Messages.parse).parameters) - {"self"}
    unknown = set(client.kwargs) - accepted
    assert not unknown, f"messages.parse would reject these kwargs: {sorted(unknown)}"
    for required in ("thinking", "output_config", "output_format"):
        assert required in accepted


def test_adaptive_is_a_real_thinking_variant_and_owns_no_budget_tokens():
    """Pins the claim in ``decide``'s comment to something checkable: the
    union has an adaptive variant, and ``budget_tokens`` is not part of it."""
    from anthropic.types import ThinkingConfigParam

    variants = {}
    for variant in typing.get_args(ThinkingConfigParam):
        hints = typing.get_type_hints(variant)
        for value in _allowed_literals(hints["type"]):
            variants[value] = hints

    assert "adaptive" in variants
    assert set({"type": "adaptive"}) <= set(variants["adaptive"])
    assert "budget_tokens" not in variants["adaptive"]
    # It exists on exactly one variant, which is why the two cannot be combined.
    assert [k for k, v in variants.items() if "budget_tokens" in v] == ["enabled"]


def test_effort_is_a_key_of_output_config_and_our_value_is_allowed(cfg):
    from anthropic.types.output_config_param import OutputConfigParam

    hints = typing.get_type_hints(OutputConfigParam)
    assert "effort" in hints
    assert "format" in hints, "parse merges the json_schema in here under 'format'"
    assert cfg.model.effort in _allowed_literals(hints["effort"])


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_cost_charges_for_cache_creation_tokens(cfg):
    """The defect this fixes: cache-creation billed as free. Two Usages that
    differ only in creation tokens must not cost the same."""
    without = Usage(input_tokens=100, output_tokens=50, cache_read_input_tokens=2_000)
    with_write = Usage(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=2_000,
        cache_creation_input_tokens=4_000,
    )
    assert with_write.cost_usd(cfg) > without.cost_usd(cfg)
    expected = 4_000 * cfg.model.price_cache_write_per_mtok / 1_000_000
    assert with_write.cost_usd(cfg) - without.cost_usd(cfg) == pytest.approx(expected)


def test_cache_creation_is_priced_above_plain_input(cfg):
    """It must not be folded into ``input_tokens``, which is what the two
    display call sites used to do — that prices the premium away."""
    assert cfg.model.price_cache_write_per_mtok > cfg.model.price_input_per_mtok
    as_input = Usage(input_tokens=10_000)
    as_write = Usage(cache_creation_input_tokens=10_000)
    assert as_write.cost_usd(cfg) > as_input.cost_usd(cfg)


def test_usage_cost_is_the_model_formula_with_all_four_counts(cfg):
    usage = Usage(
        input_tokens=1_234,
        output_tokens=5_678,
        cache_read_input_tokens=9_012,
        cache_creation_input_tokens=3_456,
    )
    assert usage.cost_usd(cfg) == cfg.model.cost_usd(
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_input_tokens,
        usage.cache_creation_input_tokens,
    )


def test_cache_hit_rate_denominator_includes_creation_tokens(cfg):
    """``report.print_spend`` shares this definition; the two used to disagree
    because the report divided by (input + cache_read) only."""
    usage = Usage(
        input_tokens=100,
        cache_read_input_tokens=300,
        cache_creation_input_tokens=100,
    )
    assert usage.total_input_tokens == 500
    assert usage.cache_hit_rate == pytest.approx(0.6)


def test_cache_hit_rate_is_zero_not_a_zero_division_on_an_empty_usage():
    assert Usage().cache_hit_rate == 0.0
