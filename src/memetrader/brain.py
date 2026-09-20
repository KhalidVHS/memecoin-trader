"""The optional model call, and the validation that decides whether to believe it.

**This module produces advice, not orders (audit C6).** The audit's finding was
that an LLM held order authority — symbol selection, sizing and timing — with no
measured predictive value behind it, and that the system had never established
the counterfactual: whether the model beat a deterministic rule on the same
evidence. Until that has been measured, ``strategy.py`` runs a deterministic
baseline by default and this path is opt-in. What comes back here is an
``AdvisoryDecision``: a suggestion that a strategy may weigh, that ``risk.py``
bounds, and that cannot become an order by itself.

``advise()`` is deliberately a single ``client.messages.parse()`` call. There is
no retry-on-malformed-JSON loop and no hand-rolled parser: structured outputs
constrain the response to the ``AdvisoryDecision`` schema server-side, and the
SDK returns an already-validated Pydantic object on ``response.parsed_output``.

**Why the old ``_normalize`` is gone, and what replaced it.** Pydantic
guarantees the *shape*, not the *semantics*: nothing in a schema stops a model
returning two actions for BONK, none for WIF, a HOLD sized at $42, or an action
for a coin we do not trade. The old code *repaired* all four — clamped, dropped,
back-filled — and traded the repair.

That is the wrong instinct, and one case proves it. The repair tested
``size < 0.0``, and every comparison against NaN is false, so a NaN size was not
negative, was not positive, was not out of range. It passed the repair. Pydantic
``Field(ge=0.0)`` does not reject NaN either, so it passed the schema. It then
passed every risk comparison for the same reason — ``nan > max_notional`` is
false, so the cap does not bind — and arrived at the broker as an order size.
A single unchecked float can walk the entire length of this system precisely
because nothing about it is ever true.

So :func:`_validate` repairs nothing. A model that emits an invalid value has
malfunctioned, and the correct response to a malfunctioning component is to
discard its entire output and record the failure, not to guess what it meant and
trade the guess. Finiteness is checked with ``types.finite``, which is the one
place in the codebase that knows NaN is not a number that can be compared.

Failure policy: an API error raises ``BrainError``. It must never come back as a
HOLD. A model that looked at the evidence and chose to sit still, and a tick
where we never reached the model at all, are different events, and a decision
log that renders them identically cannot be used to debug a bad run. Discarding
a malformed output is the same kind of event and raises the same way — the
caller falls back to the deterministic baseline, which is a decision made by
code that works.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import anthropic

if TYPE_CHECKING:
    from anthropic.types import TextBlockParam

from .prompts import build_system, render_user, system_fingerprint
from .types import (
    AdvisoryDecision,
    DecisionRecord,
    EvidenceBundle,
    PortfolioState,
    RiskBounds,
    ValidationError,
    finite,
)

log = logging.getLogger(__name__)

__all__ = [
    "BrainError",
    "ModelCallError",
    "ModelOutputError",
    "Usage",
    "advise",
]


class BrainError(RuntimeError):
    """Anything that stopped us getting usable advice this tick.

    The caller catches this and proceeds without advice — logging the skip as a
    skip, not as a HOLD.
    """


class ModelCallError(BrainError):
    """The API call itself failed: rate limit, 5xx, connection, bad request."""

    def __init__(self, message: str, *, retryable: bool, cause: Exception | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.__cause__ = cause


class ModelOutputError(BrainError):
    """The call returned, but there is no usable advice in it.

    Raised for *any* semantic defect, however small and however local. There is
    no partial acceptance: see :func:`_validate`.
    """


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
    log instead of silently multiplying the bill. It did exactly that on the
    12-hour run of 2026-09-20: 3,683 cache-read tokens against 176,784
    cache-write, a 1.0% hit rate, $1.02 of a $3.48 bill spent re-processing an
    unchanged prefix. ``prompts.build_system`` is now byte-frozen for that
    reason and ``prompt_fingerprint`` below lets a log line prove it.

    ``thinking`` rides here rather than in a third return value because
    ``advise()``'s ``tuple[AdvisoryDecision, Usage]`` is unpacked by the caller
    and by the tests; widening that tuple would break both. It belongs alongside
    the counts anyway: on ``effort = "high"`` the reasoning is most of what the
    output tokens were spent on, so the field that explains a bad trade and the
    field that explains the bill are the same purchase.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    thinking: str | None = None
    #: Hash of the frozen system prefix this call was made against. Audit §8
    #: asks for the prompt to be pinned alongside the model so a change in
    #: advisory behaviour can be attributed rather than argued about.
    prompt_fingerprint: str = ""

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

    def cost_usd(self, model: Any) -> float:
        """Every token this call billed for, priced once.

        Takes the model settings object directly rather than the whole config:
        this module is given what it needs instead of reaching into a global.

        An earlier version passed only ``input_tokens`` and so charged nothing
        at all for cache *creation*, which bills above the base input rate. It
        understated the first call against any fresh prefix, which is exactly
        the call a cache regression makes you pay over and over — the error and
        the bug it was hiding reinforced each other.
        """
        return model.cost_usd(
            self.input_tokens,
            self.output_tokens,
            self.cache_read_input_tokens,
            self.cache_creation_input_tokens,
        )

    @classmethod
    def from_response(cls, response: Any, *, prompt_fingerprint: str = "") -> Usage:
        thinking = _thinking_from_response(response)
        u = getattr(response, "usage", None)
        if u is None:
            return cls(thinking=thinking, prompt_fingerprint=prompt_fingerprint)
        return cls(
            input_tokens=int(getattr(u, "input_tokens", 0) or 0),
            output_tokens=int(getattr(u, "output_tokens", 0) or 0),
            cache_read_input_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
            cache_creation_input_tokens=int(
                getattr(u, "cache_creation_input_tokens", 0) or 0
            ),
            thinking=thinking,
            prompt_fingerprint=prompt_fingerprint,
        )


def _client_for(api_key: str | None) -> anthropic.Anthropic:
    # An explicit key wins; otherwise let the SDK resolve credentials from the
    # environment (ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, an `ant auth login`
    # profile, ...). Passing api_key=None explicitly would break profile
    # resolution, so only pass it when we actually have one.
    if api_key:
        return anthropic.Anthropic(api_key=api_key)
    return anthropic.Anthropic()


def _validate(decision: AdvisoryDecision, symbols: Sequence[str]) -> AdvisoryDecision:
    """Accept the whole output or none of it. **Nothing here repairs anything.**

    Audit C6/``brain.py::_normalize``: "strict finite schema and reject whole
    invalid output. No model repair may create an order."

    The checks, and why each one is fatal rather than local:

    * **Unknown symbol.** A ticker we do not trade is a hallucination. The old
      code dropped it and kept the rest, which treats a model that invented a
      coin as reliable about the coins it did not invent. It is one sample of
      the same output; if part of it is fabricated, the part we cannot check is
      not more trustworthy for having a familiar name on it.
    * **Duplicate symbol.** Two actions for one coin means the model held two
      views at once. "Keep the first" picks one at random and calls it a
      decision.
    * **Missing symbol.** The old code back-filled HOLD, which silently converts
      an incomplete answer into a confident flat one for that coin — and a HOLD
      on a position that should have been exited is not a null action.
    * **Non-finite or negative size.** See the module docstring. ``finite``
      raises on NaN and infinity, which is the check that the comparison
      operators cannot do.
    * **HOLD with a non-zero size.** Contradictory: the action says do nothing
      and the number says trade. Either could be the intent and guessing which
      is precisely the repair this function exists to refuse.

    ``confidence`` is not re-checked here: ``AdvisoryAction`` carries a
    finite-rejecting field validator for it, so an invalid one has already
    raised inside Pydantic before the object could be constructed.
    """
    configured = list(symbols)
    known = set(configured)
    seen: dict[str, Any] = {}

    for raw in decision.actions:
        symbol = (raw.symbol or "").strip().upper()
        if symbol not in known:
            raise ModelOutputError(
                f"model returned an action for {raw.symbol!r}, which is not in the "
                f"configured universe {sorted(known)}; discarding the whole output "
                "(a hallucinated ticker is a malfunction, not a skippable line)"
            )
        if symbol in seen:
            raise ModelOutputError(
                f"model returned two actions for {symbol}; discarding the whole "
                "output rather than picking one of two contradictory views"
            )
        try:
            size = finite(float(raw.size_usd), f"{symbol}.size_usd")
        except (ValidationError, TypeError, ValueError) as exc:
            raise ModelOutputError(
                f"model returned an unusable size_usd for {symbol} ({raw.size_usd!r}): "
                f"{exc}; discarding the whole output"
            ) from exc
        if size < 0.0:
            raise ModelOutputError(
                f"model returned a negative size_usd for {symbol} ({size}); "
                "discarding the whole output"
            )
        if raw.action == "HOLD" and size != 0.0:
            raise ModelOutputError(
                f"model returned HOLD for {symbol} with size_usd={size}; the action "
                "and the number contradict each other, so the whole output is "
                "discarded rather than one of them being guessed at"
            )
        seen[symbol] = raw.model_copy(update={"symbol": symbol, "size_usd": size})

    missing = [s for s in configured if s not in seen]
    if missing:
        raise ModelOutputError(
            f"model returned no action for {', '.join(missing)}; discarding the whole "
            "output rather than inventing a HOLD it did not advise"
        )

    # Re-ordered to the configured universe so downstream code and the journal
    # see a stable order regardless of what order the model emitted.
    return AdvisoryDecision(
        market_read=decision.market_read,
        actions=[seen[s] for s in configured],
    )


def advise(
    evidence: Mapping[str, EvidenceBundle],
    portfolio: PortfolioState,
    history: Sequence[DecisionRecord] = (),
    bounds: Sequence[RiskBounds] = (),
    *,
    symbols: Sequence[str],
    model: Any,
    risk: Any,
    starting_cash_usd: float,
    cadence: Any = None,
    api_key: str | None = None,
    client: anthropic.Anthropic | None = None,
    decision_history: int = 10,
    now: float | None = None,
) -> tuple[AdvisoryDecision, Usage]:
    """One tick's advice, or a ``BrainError``. Never invents a HOLD.

    Settings arrive as explicit arguments rather than as a ``Config`` object so
    that what this function reads is visible in its signature — and so a test
    can exercise it without constructing the whole application config.

    ``symbols`` is the configured universe and is load-bearing twice over: it
    fixes the order of the returned actions and it is the allowlist
    :func:`_validate` checks hallucinated tickers against.
    """
    client = client or _client_for(api_key)
    system = build_system(
        symbols=symbols,
        risk=risk,
        cadence=cadence,
        starting_cash_usd=starting_cash_usd,
    )
    fingerprint = system_fingerprint(system)
    user = render_user(
        evidence,
        portfolio,
        history,
        bounds,
        now=now,
        decision_history=decision_history,
    )

    try:
        response = client.messages.parse(
            model=model.name,
            max_tokens=model.max_tokens,
            # `build_system` returns plain dicts on purpose: they are what
            # gets hashed into the fingerprint and what the frozen-prefix test
            # compares, and a TypedDict would make that comparison depend on
            # the SDK. The blocks are structurally `TextBlockParam` already.
            system=cast("list[TextBlockParam]", system),
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
            # AdvisoryDecision schema into this very dict — literally
            # {**output_config, "format": transformed_output_format} — so
            # passing both is the supported combination, not a collision.
            # Note the SDK's effort literal is low|medium|high|xhigh|max, two
            # wider than config.py's allowlist; see the note there.
            output_config={"effort": model.effort},
            output_format=AdvisoryDecision,
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

    usage = Usage.from_response(response, prompt_fingerprint=fingerprint)

    if getattr(response, "stop_reason", None) == "refusal":
        details = getattr(response, "stop_details", None)
        raise ModelOutputError(
            f"the model refused to answer (category "
            f"{getattr(details, 'category', None)!r}); no advice this tick"
        )

    parsed = getattr(response, "parsed_output", None)
    if parsed is None:
        raise ModelOutputError(
            "structured output came back empty "
            f"(stop_reason={getattr(response, 'stop_reason', None)!r}); "
            "no advice this tick"
        )

    if usage.cache_read_input_tokens == 0 and usage.total_input_tokens > 0:
        # Not fatal, but it means this tick paid full price for the whole stable
        # prefix. Two in a row means something upstream is rewriting it, which
        # is the failure that cost 29% of the 2026-09-20 run. The fingerprint is
        # logged so the answer to "did the prefix change?" is in the log rather
        # than in a guess.
        log.warning(
            "prompt cache miss: 0 cached tokens of %d input tokens (system prefix %s)",
            usage.total_input_tokens,
            fingerprint,
        )

    return _validate(parsed, symbols), usage
