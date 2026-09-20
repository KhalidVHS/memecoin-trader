"""The one model call per slow tick, and the validation that follows it.

``decide()`` is deliberately a single ``client.messages.parse()`` call. There is
no retry-on-malformed-JSON loop and no hand-rolled parser: structured outputs
constrain the response to the ``TradeDecision`` schema server-side, and the SDK
returns an already-validated Pydantic object on ``response.parsed_output``.

What Pydantic guarantees is the *shape*. It does not guarantee the *semantics* —
nothing in the schema stops the model from returning two actions for BONK, none
for WIF, a HOLD with ``size_usd = 42``, or an action for a coin we do not trade.
``_normalize()`` fixes all four, deterministically, and records what it changed.

Failure policy: an API error raises ``BrainError``. It must never come back as a
HOLD. A model that looked at the evidence and chose to sit still, and a tick
where we never reached the model at all, are different events, and a decision log
that renders them identically cannot be used to debug a bad run.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import anthropic

from .prompts import build_system, render_user
from .types import (
    Action,
    DecisionRecord,
    EvidenceBundle,
    PortfolioState,
    RiskVerdict,
    TradeDecision,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import Config

log = logging.getLogger(__name__)

__all__ = [
    "BrainError",
    "ModelCallError",
    "ModelOutputError",
    "Usage",
    "decide",
]


class BrainError(RuntimeError):
    """Anything that stopped us getting a usable decision this tick.

    ``loop.py`` catches this and skips the tick — logging the skip as a skip,
    not as a HOLD.
    """


class ModelCallError(BrainError):
    """The API call itself failed: rate limit, 5xx, connection, bad request."""

    def __init__(self, message: str, *, retryable: bool, cause: Exception | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.__cause__ = cause


class ModelOutputError(BrainError):
    """The call returned, but there is no usable decision in it."""


# Thinking arrives as *content blocks*, not as a field on the response. Read
# from the installed anthropic 1.7.0 type definitions on 2026-09-19:
# ``ParsedMessage.content`` is a list discriminated on ``type``, and the
# ``thinking`` variant carries the text on ``.thinking``. Adaptive thinking may
# emit more than one such block, so they are concatenated rather than indexed.
_REDACTED_THINKING = "[redacted_thinking: encrypted by the API, unreadable here]"


def _thinking_from_response(response: Any) -> str | None:
    """The model's reasoning for this call, or ``None`` if it did not think.

    A ``redacted_thinking`` block contributes a marker rather than its ``.data``,
    which is ciphertext: writing that into every row of ``decisions.jsonl``
    would cost real disk for bytes nobody can read. It must not collapse to
    ``None`` either — "the model did not think" and "the model thought and we
    are not allowed to see it" are different events, the same distinction this
    module already draws between a skipped tick and a HOLD.
    """
    parts: list[str] = []
    for block in getattr(response, "content", None) or ():
        kind = getattr(block, "type", None)
        if kind == "thinking":
            text = str(getattr(block, "thinking", "") or "").strip()
            if text:
                parts.append(text)
        elif kind == "redacted_thinking":
            parts.append(_REDACTED_THINKING)
    return "\n\n".join(parts) or None


@dataclass(frozen=True, slots=True)
class Usage:
    """What one call cost and what the model was thinking, in the shape
    ``DecisionRecord`` wants.

    Token counts are surfaced from day one so the cost of the run is visible in
    ``decisions.jsonl`` rather than on next month's invoice — and so a cache
    regression (``cache_read_input_tokens`` collapsing to zero) shows up in the
    log instead of silently multiplying the bill.

    ``thinking`` rides here rather than in a third return value because
    ``decide()``'s ``tuple[TradeDecision, Usage]`` is unpacked by ``loop.py``
    and by the tests; widening that tuple would break both. It belongs
    alongside the counts anyway: on ``effort = "high"`` the reasoning is most of
    what the output tokens were spent on, so the field that explains a bad trade
    and the field that explains the bill are the same purchase.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    thinking: str | None = None

    @property
    def total_input_tokens(self) -> int:
        """Total prompt size. ``input_tokens`` alone is the uncached remainder."""
        return (
            self.input_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )

    @property
    def cache_hit_rate(self) -> float:
        total = self.total_input_tokens
        return self.cache_read_input_tokens / total if total else 0.0

    def cost_usd(self, cfg: Config) -> float:
        """Every token this call billed for, priced once.

        This used to pass only ``input_tokens`` and so charged nothing at all
        for cache *creation*, which bills above the base input rate — it
        understated the first call against any fresh prefix, which is exactly
        the call a cache regression makes you pay over and over.
        """
        return cfg.model.cost_usd(
            self.input_tokens,
            self.output_tokens,
            self.cache_read_input_tokens,
            self.cache_creation_input_tokens,
        )

    @classmethod
    def from_response(cls, response: Any) -> Usage:
        thinking = _thinking_from_response(response)
        u = getattr(response, "usage", None)
        if u is None:
            return cls(thinking=thinking)
        return cls(
            input_tokens=int(getattr(u, "input_tokens", 0) or 0),
            output_tokens=int(getattr(u, "output_tokens", 0) or 0),
            cache_read_input_tokens=int(
                getattr(u, "cache_read_input_tokens", 0) or 0
            ),
            cache_creation_input_tokens=int(
                getattr(u, "cache_creation_input_tokens", 0) or 0
            ),
            thinking=thinking,
        )


def _client_for(cfg: Config) -> anthropic.Anthropic:
    # An explicit key from config wins; otherwise let the SDK resolve credentials
    # from the environment (ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, an `ant auth
    # login` profile, ...). Passing api_key=None explicitly would break profile
    # resolution, so only pass it when we actually have one.
    if cfg.anthropic_api_key:
        return anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    return anthropic.Anthropic()


def _normalize(
    cfg: Config, decision: TradeDecision
) -> tuple[TradeDecision, list[str]]:
    """Force the semantic invariants the schema cannot express.

    Returns the corrected decision plus a list of human-readable notes describing
    every correction, so a model that keeps getting this wrong is visible in the
    log rather than quietly patched over.
    """
    notes: list[str] = []
    configured = set(cfg.symbols)
    by_symbol: dict[str, Action] = {}

    for raw in decision.actions:
        symbol = (raw.symbol or "").strip().upper()
        if symbol not in configured:
            notes.append(
                f"dropped action for unknown symbol {raw.symbol!r} "
                f"({raw.action}, {raw.size_usd})"
            )
            continue
        if symbol in by_symbol:
            notes.append(
                f"dropped duplicate action for {symbol} "
                f"({raw.action}, {raw.size_usd}); kept the first"
            )
            continue

        size = float(raw.size_usd)
        if raw.action == "HOLD":
            if size != 0.0:
                notes.append(f"{symbol}: HOLD with size_usd={size}, forced to 0.0")
                size = 0.0
        elif size < 0.0:
            notes.append(
                f"{symbol}: negative size_usd={size} on {raw.action}, forced to 0.0"
            )
            size = 0.0

        by_symbol[symbol] = raw.model_copy(update={"symbol": symbol, "size_usd": size})

    for symbol in cfg.symbols:
        if symbol not in by_symbol:
            notes.append(f"{symbol}: no action returned, filled with HOLD")
            by_symbol[symbol] = Action(
                action="HOLD",
                symbol=symbol,
                size_usd=0.0,
                confidence=0.0,
                reasoning=(
                    "No action was returned for this coin. Filled in as HOLD by "
                    "brain.py — this is a model output error, not a judgement."
                ),
            )

    ordered = [by_symbol[s] for s in cfg.symbols]
    market_read = decision.market_read
    if notes:
        market_read = market_read + "\n\n[brain.py corrections: " + "; ".join(notes) + "]"
    return TradeDecision(market_read=market_read, actions=ordered), notes


def decide(
    cfg: Config,
    evidence: dict[str, EvidenceBundle],
    portfolio: PortfolioState,
    history: Sequence[DecisionRecord],
    rejections: Sequence[RiskVerdict],
    *,
    client: anthropic.Anthropic | None = None,
) -> tuple[TradeDecision, Usage]:
    """One tick's decision. Raises ``BrainError`` rather than inventing a HOLD."""
    client = client or _client_for(cfg)
    system = build_system(cfg)
    user = render_user(cfg, evidence, portfolio, history, rejections)

    try:
        response = client.messages.parse(
            model=cfg.model.name,
            max_tokens=cfg.model.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            # Checked against the installed anthropic 1.7.0 on 2026-09-19 by
            # reading the SDK, *not* by a live call — no API key was reachable,
            # so nothing below is confirmed against the server yet.
            #
            # `ThinkingConfigParam` is a three-way union: enabled, disabled,
            # adaptive. `budget_tokens` is a key of the *enabled* variant only,
            # so {"type": "adaptive", "budget_tokens": N} is not expressible —
            # depth here is controlled by effort, not by a token budget. Still
            # unverified: whether enabled+budget_tokens 400s on this model
            # family. The SDK does not settle it. Its
            # MODELS_TO_WARN_WITH_THINKING_ENABLED list — the models for which
            # it warns that enabled is deprecated in favour of adaptive — holds
            # only claude-opus-4-6 and claude-mythos-preview, so for
            # claude-opus-5 the SDK neither warns nor blocks. Replace this
            # paragraph the first time a real call proves it either way.
            thinking={"type": "adaptive"},
            # Confirmed in the same pass: `effort` is a key of
            # `OutputConfigParam`, and `Messages.parse` merges the generated
            # TradeDecision schema into this very dict — literally
            # {**output_config, "format": transformed_output_format} — so
            # passing both is the supported combination, not a collision.
            # Note the SDK's effort literal is low|medium|high|xhigh|max, two
            # wider than config.py's allowlist; see the note there.
            output_config={"effort": cfg.model.effort},
            output_format=TradeDecision,
        )
    except anthropic.RateLimitError as exc:
        raise ModelCallError(
            f"rate limited by the Claude API: {exc}", retryable=True, cause=exc
        ) from exc
    except anthropic.APIStatusError as exc:
        raise ModelCallError(
            f"Claude API returned HTTP {exc.status_code}: {exc}",
            retryable=exc.status_code >= 500 or exc.status_code == 408,
            cause=exc,
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise ModelCallError(
            f"could not reach the Claude API: {exc}", retryable=True, cause=exc
        ) from exc
    except anthropic.AnthropicError as exc:  # anything else the SDK raises
        raise ModelCallError(
            f"Claude SDK error: {exc}", retryable=False, cause=exc
        ) from exc

    usage = Usage.from_response(response)

    if getattr(response, "stop_reason", None) == "refusal":
        details = getattr(response, "stop_details", None)
        raise ModelOutputError(
            f"the model refused to answer (category "
            f"{getattr(details, 'category', None)!r}); no decision this tick"
        )

    parsed = getattr(response, "parsed_output", None)
    if parsed is None:
        raise ModelOutputError(
            "structured output came back empty "
            f"(stop_reason={getattr(response, 'stop_reason', None)!r}); "
            "no decision this tick"
        )

    decision, notes = _normalize(cfg, parsed)
    for note in notes:
        log.warning("model output corrected: %s", note)
    if usage.cache_read_input_tokens == 0 and usage.total_input_tokens > 0:
        # Not fatal, but it means this tick paid full price for the whole stable
        # prefix. Two in a row means something upstream is rewriting it.
        log.warning(
            "prompt cache miss: 0 cached tokens of %d input tokens",
            usage.total_input_tokens,
        )
    return decision, usage
