"""Execution fill models — the layer that decides what price an order actually gets.

Three ``ExecutionModel`` implementations, one per fidelity tier this repository
can currently support:

* :class:`BarExecutionModel` (TIER_0) — OHLCV-only screening. Every report it
  produces carries ``types.NON_EXECUTABLE_NOTICE`` because a bar has no notion
  of size-specific depth, route availability, or landing risk; it estimates
  *signal*, not money. See ``FidelityTier.permits_pnl_claim``.
* :class:`QuoteReplayExecutionModel` (TIER_2) — replays a recorded
  size-specific quote ladder and models landing probability/delay
  independently of network latency.
* :class:`PoolStateExecutionModel` (TIER_2/3) — recomputes the swap from
  historical reserves via ``amm.constant_product``, distinguishing observed
  historical depth from a counterfactual self-impact simulation.

The one rule every model here obeys, restated because it is the single most
important invariant in this file: **a decision made from a bar close, a
ladder, or a pool snapshot must never fill against that same observation.**
``BarExecutionModel`` enforces this by filling at the *next* bar's open, never
the bar that produced the decision. The other two models enforce it by
re-validating (re-fetching the ladder/pool state, checking expiry) at
``fill()`` time rather than trusting what ``price()`` saw.

The second rule, audit finding C3 (``docs/BACKTEST-CONTRACTS.md`` §6): a quote
must be requoted, never rescaled, when the traded size changes. Every
``price()`` method below derives its output *only* from ``intent.in_amount_atomic``
— there is no code path that takes an existing ``Quote`` and scales its
amounts, so a risk clamp that produces a new, smaller intent structurally
produces a brand-new quote at that exact size.

The third rule: a failed quote, an expired quote, a disappeared route, or
insufficient liquidity must never produce a synthetic fill. Every failure path
below returns ``fill=None`` (or raises ``NoRoute``) rather than inventing an
amount.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import replace as _dc_replace
from typing import TYPE_CHECKING

from memetrader.config import ExecutionConfig
from memetrader.execution.amm.constant_product import amount_out as _cp_amount_out
from memetrader.execution.amm.constant_product import (
    price_impact_pct as _cp_price_impact_pct,
)
from memetrader.execution.amm.route_replay import RouteProvenance
from memetrader.execution.costs import build_cost_breakdown
from memetrader.execution.interfaces import ApprovedOrder, NoRoute
from memetrader.execution.latency import LatencyModel
from memetrader.ids import new_fill_id, new_order_id
from memetrader.ids import quote_fingerprint as _quote_fingerprint
from memetrader.types import (
    NON_EXECUTABLE_NOTICE,
    Candle,
    ExecutionReport,
    FidelityTier,
    Fill,
    OrderIntent,
    OrderState,
    Quote,
    Side,
    Timeframe,
    TokenMeta,
    ValidationError,
)

if TYPE_CHECKING:
    from memetrader.histdata.point_in_time import PointInTimeState
    from memetrader.histdata.schemas import PoolState

# An OrderIntent carries only a symbol, not a pool address — the caller wires
# symbol -> pool_id (e.g. via a universe registry) through this resolver
# rather than PoolStateExecutionModel guessing at a naming convention.
PoolIdResolver = Callable[[str], str]

__all__ = [
    "BarExecutionModel",
    "PoolStateExecutionModel",
    "QuoteReplayExecutionModel",
]

# Duplicated from histdata/point_in_time.py rather than imported: that module
# imports execution/interfaces.py's sibling types under TYPE_CHECKING only to
# avoid a runtime cycle, and this module follows the same discipline for the
# same reason (another agent owns histdata/point_in_time.py's runtime surface).
_INTERVAL_SECONDS: dict[Timeframe, float] = {
    Timeframe.M5: 300.0,
    Timeframe.H1: 3600.0,
}


# ---------------------------------------------------------------------------
# BarExecutionModel — TIER_0
# ---------------------------------------------------------------------------


class BarExecutionModel:
    """OHLCV-only screening fill model. TIER_0 — never a PnL claim.

    There is no mint registry available at this tier (an ``OrderIntent`` only
    carries a ``symbol``), so this model synthesizes a ``TokenMeta`` per
    symbol with ``verified=False`` and a caller-supplied assumed decimal
    count. That is a documented, deliberate approximation appropriate only for
    screening: a real swap must never size against an inferred decimals value
    (see ``TokenMeta`` docstring, audit §15), which is exactly why this model
    cannot be promoted past TIER_0.
    """

    def __init__(
        self,
        cfg: ExecutionConfig,
        *,
        usd_token: TokenMeta,
        token_decimals: int = 9,
        timeframe: Timeframe = Timeframe.H1,
        participation_cap_pct: float = 0.1,
        conservative_spread_bps: int = 50,
        slippage_bps: int = 100,
    ) -> None:
        if not 0.0 < participation_cap_pct <= 1.0:
            raise ValidationError(
                f"participation_cap_pct must be in (0, 1], got {participation_cap_pct!r}"
            )
        self._cfg = cfg
        self._usd_token = usd_token
        self._token_decimals = token_decimals
        self._timeframe = timeframe
        self._participation_cap_pct = participation_cap_pct
        self._conservative_spread_bps = conservative_spread_bps
        self._slippage_bps = slippage_bps

    @property
    def fidelity(self) -> FidelityTier:
        return FidelityTier.TIER_0

    # ------------------------------------------------------------------
    # ExecutionModel protocol
    # ------------------------------------------------------------------

    def price(
        self,
        *,
        intent: OrderIntent,
        state: PointInTimeState,
        now: float,
    ) -> Quote | None:
        """Reference-price a route from the last *closed* bar.

        This quote exists only to size the order and bound risk — it is never
        what ``fill()`` executes at. A conservative spread is applied against
        the trader on top of the raw close, because a screening-tier model
        that quotes the bare mid systematically overstates its own edge.
        """
        bars = state.bars(intent.symbol, self._timeframe, lookback=1)
        if not bars:
            raise NoRoute(f"no closed bar available for {intent.symbol} at t={now}")
        last = bars[-1]

        token = self._token_meta(intent.symbol)
        spread_frac = self._conservative_spread_bps / 10_000.0
        if intent.side is Side.BUY:
            ref_price = last.close * (1.0 + spread_frac)
            input_token, output_token = self._usd_token, token
        else:
            ref_price = last.close * (1.0 - spread_frac)
            input_token, output_token = token, self._usd_token

        out_amount = self._convert(
            intent.in_amount_atomic, input_token, ref_price, output_token, intent.side
        )
        if out_amount <= 0:
            raise NoRoute(f"reference price produced zero output for {intent.symbol}")
        min_out = out_amount * (10_000 - self._slippage_bps) // 10_000

        fp = _quote_fingerprint(
            side=str(intent.side),
            input_mint=input_token.mint,
            output_mint=output_token.mint,
            in_amount_atomic=intent.in_amount_atomic,
            out_amount_atomic=out_amount,
            slot=None,
        )
        interval = _INTERVAL_SECONDS[self._timeframe]
        return Quote(
            symbol=intent.symbol,
            side=intent.side,
            input_token=input_token,
            output_token=output_token,
            in_amount_atomic=intent.in_amount_atomic,
            out_amount_atomic=out_amount,
            min_out_amount_atomic=min_out,
            # Price impact is not modeled at TIER_0: a bar has no depth curve,
            # only a close price. Reporting 0.0 rather than a guess.
            price_impact_pct=0.0,
            route_labels=("bar_estimate",),
            fingerprint=fp,
            requested_at=now,
            received_at=now,
            context_slot=None,
            expires_at=now + interval,
            reference_price_usd=ref_price,
        )

    def fill(
        self,
        *,
        order: ApprovedOrder,
        state: PointInTimeState,
        now: float,
    ) -> ExecutionReport:
        """Fill at the *next* bar's open. Never the bar that produced the decision.

        This is the anti-lookahead rule this whole model exists to enforce: a
        signal computed from a bar's close cannot buy or sell at that close,
        because the close was not knowable until the bar's ``available_time``
        (publication delay aside) — and by the time it *was* knowable, the
        market had already moved on to the next bar.
        """
        intent = order.intent
        if now <= order.decided_at:
            raise ValidationError(
                f"fill() called at now={now} <= decided_at={order.decided_at}: a fill at or "
                "before the decision timestamp is look-ahead and must never be constructed"
            )

        # ">=" rather than ">": a decision made the instant bar1 becomes
        # available (decided_at == bar1.available_time) is legitimately
        # followed by the very next bar, whose ts opens at that same instant
        # in a contiguous candle series. Excluding it with a strict ">" would
        # incorrectly demand the *next* bar after that one, adding a full
        # extra interval of look-ahead-avoidance the rule never asked for.
        # What must never happen is filling against the *decision* bar
        # itself (bar1, whose ts is strictly before decided_at) — that case
        # is still excluded below.
        bars = state.bars(intent.symbol, self._timeframe, lookback=500)
        candidates = [b for b in bars if b.ts >= order.decided_at]
        if not candidates:
            # The bar that must exist for this fill to be legitimate has not
            # arrived yet from this model's point of view. This is exactly the
            # "route disappeared between price and fill" case in spirit: no
            # bar means no way to execute, so raise rather than invent a price.
            raise NoRoute(
                f"no bar opened after decided_at={order.decided_at} for {intent.symbol}"
            )
        next_bar = min(candidates, key=lambda b: b.ts)
        entry_price = next_bar.open  # THE rule: never next_bar.close, never last_bar.close.

        token = self._token_meta(intent.symbol)
        side = intent.side
        if side is Side.BUY:
            input_token, output_token = self._usd_token, token
        else:
            input_token, output_token = token, self._usd_token

        requested_in_ui = input_token.to_ui(intent.in_amount_atomic)

        # Participation cap: never assume the strategy could have traded more
        # than a conservative fraction of what the bar actually saw trade.
        # ``Candle.volume`` is the traded token's UI quantity, so a BUY's USD
        # request is converted through entry_price to compare like units.
        if side is Side.BUY:
            desired_token_ui = requested_in_ui / entry_price if entry_price > 0 else 0.0
        else:
            desired_token_ui = requested_in_ui

        cap_token_ui = next_bar.volume * self._participation_cap_pct
        filled_token_ui = min(desired_token_ui, cap_token_ui)

        if filled_token_ui <= 0.0:
            # Documented choice: participation-cap exhaustion is a full
            # reject, not a zero-size partial. A zero-size Fill row would
            # violate the "SELL quantity == quoted quantity exactly" spirit
            # of the accounting invariants for no benefit.
            return ExecutionReport(
                report_id=new_fill_id(),
                intent_id=intent.intent_id,
                order_id=None,
                state=OrderState.FAILED,
                ts=now,
                fidelity=FidelityTier.TIER_0,
                fill=None,
                costs=None,
                reason=(
                    f"participation cap exhausted (bar volume={next_bar.volume}); "
                    f"{NON_EXECUTABLE_NOTICE}"
                ),
            )

        # Documented choice: partial fill, not full reject, when the cap binds
        # but is not zero. The bar genuinely saw that much volume trade; a
        # screening model that refuses the whole order because it wanted more
        # than the bar could bear would understate how much of the signal was
        # actually actionable.
        partial = filled_token_ui < desired_token_ui - 1e-12

        if partial and side is Side.SELL:
            # ...but that rationale is an entry-side argument, and it does not
            # survive contact with contracts §6 ("SELL quantity == quoted
            # quantity, exactly", enforced on the settled Fill by
            # invariants.check_sell_quantity_matches_quote). A partially filled
            # SELL under-delivers against the quote it was bound to and
            # breaches that invariant outright — the same reasoning the
            # cap-exhausted branch above already applies, just at a non-zero
            # size.
            #
            # It also cascades, which is how this surfaced: an exit that leaves
            # a sliver of the position behind re-triggers the same stop on the
            # next decision tick, which sells the same fraction of the
            # remainder, and one thin bar becomes a chain of ever-smaller dust
            # trades that never fully closes the position.
            #
            # Rejecting is the conservative reading a screening model owes an
            # exit: record that the position could not be got out inside the
            # bar's own traded volume, rather than record an exit that did not
            # happen at the size it claims.
            return ExecutionReport(
                report_id=new_fill_id(),
                intent_id=intent.intent_id,
                order_id=None,
                state=OrderState.FAILED,
                ts=now,
                fidelity=FidelityTier.TIER_0,
                fill=None,
                costs=None,
                reason=(
                    f"participation cap binds a SELL (bar volume={next_bar.volume}, "
                    f"wanted {desired_token_ui}, cap {cap_token_ui}); an exit must "
                    f"fill its quoted quantity exactly or not at all; "
                    f"{NON_EXECUTABLE_NOTICE}"
                ),
            )

        if side is Side.BUY:
            fill_in_atomic = input_token.to_atomic(filled_token_ui * entry_price)
            fill_out_atomic = output_token.to_atomic(filled_token_ui)
        else:
            # The exact intent integer, deliberately not the float round-trip
            # ``input_token.to_atomic(filled_token_ui)``. A SELL reaching here
            # is never partial (the branch above rejects those), so
            # ``filled_token_ui`` is exactly ``to_ui(intent.in_amount_atomic)``
            # and the only thing re-deriving it can do is lose an atomic unit
            # to float rounding — which it does, and which
            # ``check_sell_quantity_matches_quote`` then reports as a 1-unit
            # breach against a quote that priced the intent's own integer.
            # Contracts §0 keeps token amounts as ``int`` atomic precisely so
            # this leg never has to survive a round trip through a float.
            fill_in_atomic = intent.in_amount_atomic
            fill_out_atomic = output_token.to_atomic(filled_token_ui * entry_price)

        token_amount_atomic = fill_out_atomic if side is Side.BUY else fill_in_atomic
        notional_usd = (
            input_token.to_ui(fill_in_atomic)
            if side is Side.BUY
            else output_token.to_ui(fill_out_atomic)
        )

        order_id = new_order_id()
        fill = Fill(
            fill_id=new_fill_id(),
            order_id=order_id,
            intent_id=intent.intent_id,
            decision_id=intent.decision_id,
            ts=now,
            symbol=intent.symbol,
            side=side,
            state=OrderState.LANDED,
            in_amount_atomic=fill_in_atomic,
            out_amount_atomic=fill_out_atomic,
            token_amount_atomic=token_amount_atomic,
            token_decimals=token.decimals,
            quote_fingerprint=order.quote.fingerprint,
            price_usd=entry_price,
            notional_usd=notional_usd,
            price_impact_pct=None,
            pool_fee_usd=0.0,
            gas_usd=self._cfg.gas_usd_per_swap,
            note="partial: participation cap" if partial else None,
        )
        costs = build_cost_breakdown(
            fill,
            self._cfg,
            quote_replayed=False,
            latency_seconds=now - order.decided_at,
            price_at_decision=order.quote.reference_price_usd,
            price_at_fill=entry_price,
        )
        return ExecutionReport(
            report_id=new_fill_id(),
            intent_id=intent.intent_id,
            order_id=order_id,
            state=OrderState.LANDED,
            ts=now,
            fidelity=FidelityTier.TIER_0,
            fill=fill,
            costs=costs,
            # Every report this TIER_0 model produces must carry the notice
            # verbatim: FidelityTier.TIER_0.permits_pnl_claim is False, and the
            # broker must not be able to read a landed report as validated PnL.
            reason=NON_EXECUTABLE_NOTICE,
        )

    # ------------------------------------------------------------------
    # Same-bar stop/target ambiguity
    # ------------------------------------------------------------------

    def resolve_same_bar_exit(
        self,
        bar: Candle,
        *,
        stop_price: float | None,
        target_price: float | None,
    ) -> tuple[float, str]:
        """Resolve which of a stop and a target executes inside one bar.

        Documented choice: pessimistic ordering. A bar's OHLC gives no
        information about the *path* the price took between open and close,
        so if both a stop and a target level fall inside ``[low, high]`` this
        model cannot know which was touched first. Assuming the friendlier
        sequence (target first) is exactly the optimism this whole layer
        exists to refuse; assuming the stop hit first is the conservative,
        auditable choice. The stop is therefore checked unconditionally
        before the target: whenever both are in range, the stop wins.

        Returns ``(fill_price, "stop" | "target")``. Raises ``ValidationError``
        if neither level falls within the bar's range — that is a caller bug
        (the exit should not have been evaluated against this bar at all).
        """
        if stop_price is not None and bar.low <= stop_price <= bar.high:
            return stop_price, "stop"
        if target_price is not None and bar.low <= target_price <= bar.high:
            return target_price, "target"
        raise ValidationError(
            f"neither stop_price={stop_price} nor target_price={target_price} falls "
            f"within bar range [{bar.low}, {bar.high}]"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _token_meta(self, symbol: str) -> TokenMeta:
        # No mint registry exists at TIER_0 — see class docstring.
        # ``verified=False`` marks this identity as synthetic so nothing can
        # mistake it for a chain-verified mint later in the pipeline.
        return TokenMeta(
            mint=symbol,
            decimals=self._token_decimals,
            source="bar_execution_model",
            verified=False,
        )

    @staticmethod
    def _convert(
        amount_atomic: int,
        input_token: TokenMeta,
        price_usd_per_token: float,
        output_token: TokenMeta,
        side: Side,
    ) -> int:
        in_ui = input_token.to_ui(amount_atomic)
        if side is Side.BUY:
            if price_usd_per_token <= 0:
                return 0
            out_ui = in_ui / price_usd_per_token
        else:
            out_ui = in_ui * price_usd_per_token
        return output_token.to_atomic(out_ui)


# ---------------------------------------------------------------------------
# QuoteReplayExecutionModel — TIER_2
# ---------------------------------------------------------------------------


class QuoteReplayExecutionModel:
    """Replays a recorded, size-specific quote ladder. TIER_2 — a PnL claim
    is permitted, but only because every number below is either an exact
    on-chain-shaped integer or a modeled probability drawn from its own named
    stream.

    Landing probability/delay and network latency are drawn from two
    independent ``random.Random`` instances, never one shared RNG. This
    mirrors ``backtest/clock.py``'s ``deterministic_seed`` rationale exactly
    (see that module's docstring): if a coin-flip and a latency draw share an
    RNG, adding one extra latency draw shifts every subsequent coin-flip,
    which destroys the ability to compare two runs "on the same seed".
    ``LatencyModel.draw()`` itself already bundles a ``will_land``/
    ``landing_delay_s`` pair drawn from *its own* internal RNG alongside the
    six stage timings — that internal draw is deliberately ignored here and
    only ``total_s`` is used, precisely because that field is not independent
    of the timing stages. This model's own ``landing_rng`` is the only source
    of the landing outcome.
    """

    def __init__(
        self,
        cfg: ExecutionConfig,
        *,
        latency_model: LatencyModel,
        landing_rng_seed: int,
        landing_probability: float = 0.94,
        max_landing_delay_s: float = 2.0,
        quote_validity_seconds: float = 5.0,
    ) -> None:
        if not 0.0 <= landing_probability <= 1.0:
            raise ValidationError(
                f"landing_probability must be in [0, 1], got {landing_probability!r}"
            )
        if max_landing_delay_s < 0.0:
            raise ValidationError(
                f"max_landing_delay_s must be >= 0, got {max_landing_delay_s!r}"
            )
        if quote_validity_seconds <= 0.0:
            raise ValidationError(
                f"quote_validity_seconds must be > 0, got {quote_validity_seconds!r}"
            )
        self._cfg = cfg
        self._latency_model = latency_model
        self._landing_rng = random.Random(landing_rng_seed)
        self._landing_probability = landing_probability
        self._max_landing_delay_s = max_landing_delay_s
        self._quote_validity_seconds = quote_validity_seconds

    @property
    def fidelity(self) -> FidelityTier:
        return FidelityTier.TIER_2

    # ------------------------------------------------------------------
    # ExecutionModel protocol
    # ------------------------------------------------------------------

    def price(
        self,
        *,
        intent: OrderIntent,
        state: PointInTimeState,
        now: float,
    ) -> Quote | None:
        """Quote the intent's exact size against the recorded ladder.

        Audit C3: this method derives the traded amount *only* from
        ``intent.in_amount_atomic``. There is no parameter, cache, or code
        path here that takes a previously obtained ``Quote`` and rescales it
        — a risk clamp that produces a smaller ``intent`` therefore always
        produces a brand-new quote at exactly that size, never a rescaled
        copy of a larger one.
        """
        ladder = state.quote_ladder(intent.symbol, intent.side)
        if ladder is None:
            raise NoRoute(f"no quote ladder for {intent.symbol}/{intent.side}")
        rung = ladder.best_rung_for(intent.in_amount_atomic)
        if rung is None:
            raise NoRoute(
                f"no ladder rung at or below {intent.in_amount_atomic} atomic for {intent.symbol}"
            )

        # The ladder only ever probed discrete sizes. Scaling the floor rung's
        # marginal rate up to the exact requested size (rather than settling
        # for the rung's own, smaller amount) is what keeps the quote bound to
        # exactly intent.in_amount_atomic, per C3 above. Integer floor scaling
        # throughout — no float touches the atomic amounts.
        out_amount = (
            intent.in_amount_atomic * rung.out_amount_atomic // rung.in_amount_atomic
        )
        if out_amount == 0:
            raise NoRoute(f"scaled ladder output is zero for {intent.symbol}")
        min_out = intent.in_amount_atomic * rung.min_out_atomic // rung.in_amount_atomic

        token = TokenMeta(
            mint=ladder.asset_id,
            decimals=self._token_decimals_for(ladder.asset_id),
            source="quote_ladder",
        )
        usd_token = self._usd_token
        if intent.side is Side.BUY:
            input_token, output_token = usd_token, token
        else:
            input_token, output_token = token, usd_token

        fp = _quote_fingerprint(
            side=str(intent.side),
            input_mint=input_token.mint,
            output_mint=output_token.mint,
            in_amount_atomic=intent.in_amount_atomic,
            out_amount_atomic=out_amount,
            slot=ladder.context_slot,
        )
        return Quote(
            symbol=intent.symbol,
            side=intent.side,
            input_token=input_token,
            output_token=output_token,
            in_amount_atomic=intent.in_amount_atomic,
            out_amount_atomic=out_amount,
            min_out_amount_atomic=min_out,
            price_impact_pct=rung.price_impact_pct,
            route_labels=rung.route_labels,
            fingerprint=fp,
            requested_at=now,
            received_at=ladder.event_time,
            context_slot=ladder.context_slot,
            # A quote is only valid for a bounded window after it was
            # obtained — a real Jupiter quote goes stale within seconds.
            expires_at=ladder.event_time + self._quote_validity_seconds,
            reference_price_usd=None,
        )

    def fill(
        self,
        *,
        order: ApprovedOrder,
        state: PointInTimeState,
        now: float,
    ) -> ExecutionReport:
        """Settle against ``order.quote``, drawing landing outcome and latency
        from two independent RNGs.

        Never manufactures a fill: an expired quote or a route that vanished
        between ``price()`` and here produces a terminal non-fill with
        ``fill=None``, and a dropped transaction produces a ``Fill`` row with
        zero amounts (the gas was still spent).
        """
        quote = order.quote
        intent = order.intent

        # Defense in depth for C3: the quote bound to this order must have
        # been obtained for exactly this intent's size. A mismatch means a
        # stale or mis-sized quote is being fed to fill(), which must never
        # settle — see class/method docstrings above.
        if quote.in_amount_atomic != intent.in_amount_atomic:
            raise NoRoute(
                f"quote size {quote.in_amount_atomic} does not match approved intent size "
                f"{intent.in_amount_atomic} — a stale or mis-sized quote must never fill"
            )

        if quote.is_expired(now):
            return ExecutionReport(
                report_id=new_fill_id(),
                intent_id=intent.intent_id,
                order_id=None,
                state=OrderState.EXPIRED,
                ts=now,
                fidelity=FidelityTier.TIER_2,
                fill=None,
                costs=None,
                reason=f"quote expired at {quote.expires_at}, fill attempted at {now}",
            )

        # Route disappearance check: the route must still be quotable now,
        # not just at price() time. A ladder that vanished (or thinned below
        # this size) between price() and fill() is exactly the "route
        # disappeared" case interfaces.py's fill() docstring calls out.
        ladder = state.quote_ladder(intent.symbol, intent.side)
        if ladder is None or ladder.best_rung_for(intent.in_amount_atomic) is None:
            raise NoRoute(
                f"route for {intent.symbol}/{intent.side} disappeared before fill"
            )

        latency_draw = self._latency_model.draw()
        will_land = self._landing_rng.random() < self._landing_probability

        order_id = new_order_id()
        gas_usd = self._cfg.gas_usd_per_swap

        if not will_land:
            fill = Fill(
                fill_id=new_fill_id(),
                order_id=order_id,
                intent_id=intent.intent_id,
                decision_id=intent.decision_id,
                ts=now,
                symbol=intent.symbol,
                side=intent.side,
                state=OrderState.FAILED,
                in_amount_atomic=0,
                out_amount_atomic=0,
                token_amount_atomic=0,
                token_decimals=quote.token_meta.decimals,
                quote_fingerprint=quote.fingerprint,
                price_usd=None,
                notional_usd=0.0,
                price_impact_pct=None,
                pool_fee_usd=0.0,
                gas_usd=gas_usd,
                note="transaction failed to land",
            )
            costs = build_cost_breakdown(
                fill, self._cfg, quote_replayed=True, latency_seconds=latency_draw.total_s
            )
            return ExecutionReport(
                report_id=new_fill_id(),
                intent_id=intent.intent_id,
                order_id=order_id,
                state=OrderState.FAILED,
                ts=now,
                fidelity=FidelityTier.TIER_2,
                fill=fill,
                costs=costs,
                reason="transaction failed to land",
            )

        # Landed. min_out is already respected by construction: Quote's own
        # validation guarantees min_out_amount_atomic <= out_amount_atomic,
        # and the ladder replay never rescales the settled amount below it.
        fill = Fill(
            fill_id=new_fill_id(),
            order_id=order_id,
            intent_id=intent.intent_id,
            decision_id=intent.decision_id,
            ts=now,
            symbol=intent.symbol,
            side=intent.side,
            state=OrderState.LANDED,
            in_amount_atomic=quote.in_amount_atomic,
            out_amount_atomic=quote.out_amount_atomic,
            token_amount_atomic=quote.token_amount_atomic,
            token_decimals=quote.token_meta.decimals,
            quote_fingerprint=quote.fingerprint,
            price_usd=quote.effective_price_usd,
            notional_usd=quote.usd_notional,
            price_impact_pct=quote.price_impact_pct,
            pool_fee_usd=0.0,  # replayed quote: outAmount is already fee-net.
            gas_usd=gas_usd,
        )
        costs = build_cost_breakdown(
            fill,
            self._cfg,
            quote_replayed=True,
            latency_seconds=latency_draw.total_s,
            price_at_decision=None,
            price_at_fill=quote.effective_price_usd,
        )
        return ExecutionReport(
            report_id=new_fill_id(),
            intent_id=intent.intent_id,
            order_id=order_id,
            state=OrderState.LANDED,
            ts=now,
            fidelity=FidelityTier.TIER_2,
            fill=fill,
            costs=costs,
            reason="",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _usd_token = TokenMeta(
        mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
        decimals=6,
        source="quote_replay_model",
    )

    @staticmethod
    def _token_decimals_for(_asset_id: str) -> int:
        # No decimals registry travels with a QuoteLadder record (only the
        # mint does). 9 is the common SPL default for memecoins; a caller with
        # a real registry should wrap this model rather than rely on the
        # default for anything beyond TIER_2 screening-grade replay.
        return 9


# ---------------------------------------------------------------------------
# PoolStateExecutionModel — TIER_2/3
# ---------------------------------------------------------------------------


class PoolStateExecutionModel:
    """Recomputes the swap from historical reserves. TIER_2 by default;
    caller may declare TIER_3 once the model has been calibrated against
    observed landed/failed transactions (``docs/BACKTEST-CONTRACTS.md`` §3).

    ``PoolState.reserve_in_atomic``/``reserve_out_atomic`` name the pool's
    *base* and *quote* reserves respectively — not "input"/"output" of a
    particular swap direction. A BUY spends the quote reserve and receives
    from the base reserve; a SELL is the reverse. Getting this backwards
    silently swaps which side eats the price impact, so every call site below
    resolves the AMM's (reserve_in, reserve_out) pair explicitly from
    ``side`` rather than reusing the schema's field names directly.
    """

    def __init__(
        self,
        cfg: ExecutionConfig,
        *,
        usd_token: TokenMeta,
        pool_id_for: PoolIdResolver | None = None,
        slippage_bps: int = 100,
        model_self_impact: bool = False,
        fidelity: FidelityTier = FidelityTier.TIER_2,
    ) -> None:
        self._cfg = cfg
        self._usd_token = usd_token
        self._pool_id_for = (
            pool_id_for if pool_id_for is not None else (lambda symbol: symbol)
        )
        self._slippage_bps = slippage_bps
        self._model_self_impact = model_self_impact
        self._fidelity = fidelity
        # Only ever populated when model_self_impact=True. Keyed by pool_id;
        # holds (reserve_in_atomic, reserve_out_atomic) *after* this model's
        # own prior fills against that pool within this run. Gated behind a
        # flag and defaulted off: layering a self-impact simulation on top of
        # reserves that already reflect every other participant's realized
        # flow double-counts impact unless the operator has deliberately
        # opted into a counterfactual "what if I had also traded here" mode.
        self._self_impact_reserves: dict[str, tuple[int, int]] = {}

    @property
    def fidelity(self) -> FidelityTier:
        return self._fidelity

    # ------------------------------------------------------------------
    # ExecutionModel protocol
    # ------------------------------------------------------------------

    def price(
        self,
        *,
        intent: OrderIntent,
        state: PointInTimeState,
        now: float,
    ) -> Quote | None:
        resolved = self._resolve_pool(intent.symbol, state)
        if resolved is None:
            raise NoRoute(f"no pool state for {intent.symbol}")
        pool_id, pool, _provenance = resolved

        token = TokenMeta(mint=pool.asset_id, decimals=9, source="pool_state_model")
        reserve_in, reserve_out = self._amm_reserves(pool, intent.side)
        out_amount = _cp_amount_out(
            reserve_in, reserve_out, intent.in_amount_atomic, pool.fee_rate_bps
        )
        if out_amount == 0:
            raise NoRoute(
                f"pool {pool_id} has insufficient depth for {intent.in_amount_atomic} atomic"
            )
        impact = _cp_price_impact_pct(
            reserve_in, reserve_out, intent.in_amount_atomic, pool.fee_rate_bps
        )
        min_out = out_amount * (10_000 - self._slippage_bps) // 10_000

        if intent.side is Side.BUY:
            input_token, output_token = self._usd_token, token
        else:
            input_token, output_token = token, self._usd_token

        fp = _quote_fingerprint(
            side=str(intent.side),
            input_mint=input_token.mint,
            output_mint=output_token.mint,
            in_amount_atomic=intent.in_amount_atomic,
            out_amount_atomic=out_amount,
            slot=pool.slot,
        )
        return Quote(
            symbol=intent.symbol,
            side=intent.side,
            input_token=input_token,
            output_token=output_token,
            in_amount_atomic=intent.in_amount_atomic,
            out_amount_atomic=out_amount,
            min_out_amount_atomic=min_out,
            price_impact_pct=impact,
            route_labels=(pool_id,),
            fingerprint=fp,
            requested_at=now,
            received_at=now,
            context_slot=pool.slot,
            expires_at=None,
            reference_price_usd=pool.price_usd,
        )

    def fill(
        self,
        *,
        order: ApprovedOrder,
        state: PointInTimeState,
        now: float,
    ) -> ExecutionReport:
        """Recompute the swap against the pool state available *at settlement
        time*, not the snapshot ``price()`` saw. Prices move during latency;
        trusting the decision-time snapshot would understate that."""
        intent = order.intent
        resolved = self._resolve_pool(intent.symbol, state)
        if resolved is None:
            raise NoRoute(f"pool state for {intent.symbol} disappeared before fill")
        pool_id, pool, provenance = resolved

        reserve_in, reserve_out = self._amm_reserves(pool, intent.side)
        in_amount = intent.in_amount_atomic
        out_amount = _cp_amount_out(reserve_in, reserve_out, in_amount, pool.fee_rate_bps)

        order_id = new_order_id()
        gas_usd = self._cfg.gas_usd_per_swap

        if out_amount == 0:
            return ExecutionReport(
                report_id=new_fill_id(),
                intent_id=intent.intent_id,
                order_id=order_id,
                state=OrderState.FAILED,
                ts=now,
                fidelity=self._fidelity,
                fill=None,
                costs=None,
                reason=f"pool {pool_id} depth exhausted at settlement",
            )
        if out_amount < order.quote.min_out_amount_atomic:
            return ExecutionReport(
                report_id=new_fill_id(),
                intent_id=intent.intent_id,
                order_id=order_id,
                state=OrderState.FAILED,
                ts=now,
                fidelity=self._fidelity,
                fill=None,
                costs=None,
                reason=(
                    f"settlement output {out_amount} below min_out "
                    f"{order.quote.min_out_amount_atomic}"
                ),
            )

        input_token = self._usd_token if intent.side is Side.BUY else order.quote.token_meta
        fee_bps = pool.fee_rate_bps
        in_amount_net = in_amount * (10_000 - fee_bps) // 10_000
        fee_atomic = in_amount - in_amount_net
        if intent.side is Side.BUY:
            pool_fee_usd = input_token.to_ui(fee_atomic)
        else:
            # The fee is charged in the traded token, not USD. Converting it
            # requires a price; the pool's own mid is the best available one
            # and, absent it, the fee is reported as 0.0 rather than guessed.
            pool_fee_usd = (
                input_token.to_ui(fee_atomic) * pool.price_usd if pool.price_usd else 0.0
            )

        token_meta = order.quote.token_meta
        token_amount_atomic = out_amount if intent.side is Side.BUY else in_amount
        price_usd = (
            out_amount / in_amount
            if intent.side is Side.SELL and in_amount
            else pool.price_usd
        )
        notional_usd = (
            input_token.to_ui(in_amount)
            if intent.side is Side.BUY
            else self._usd_token.to_ui(out_amount)
        )
        impact = _cp_price_impact_pct(reserve_in, reserve_out, in_amount, fee_bps)

        fill = Fill(
            fill_id=new_fill_id(),
            order_id=order_id,
            intent_id=intent.intent_id,
            decision_id=intent.decision_id,
            ts=now,
            symbol=intent.symbol,
            side=intent.side,
            state=OrderState.LANDED,
            in_amount_atomic=in_amount,
            out_amount_atomic=out_amount,
            token_amount_atomic=token_amount_atomic,
            token_decimals=token_meta.decimals,
            quote_fingerprint=order.quote.fingerprint,
            price_usd=price_usd,
            notional_usd=notional_usd,
            price_impact_pct=impact,
            pool_fee_usd=pool_fee_usd,
            gas_usd=gas_usd,
            note=f"route_provenance={provenance.value}",
        )
        costs = build_cost_breakdown(
            fill,
            self._cfg,
            quote_replayed=False,
            latency_seconds=now - order.decided_at,
            price_at_decision=order.quote.reference_price_usd,
            price_at_fill=pool.price_usd,
        )

        if self._model_self_impact:
            self._apply_self_impact(pool_id, pool, intent.side, in_amount, out_amount)

        return ExecutionReport(
            report_id=new_fill_id(),
            intent_id=intent.intent_id,
            order_id=order_id,
            state=OrderState.LANDED,
            ts=now,
            fidelity=self._fidelity,
            fill=fill,
            costs=costs,
            reason="",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_pool(
        self, symbol: str, state: PointInTimeState
    ) -> tuple[str, PoolState, RouteProvenance] | None:
        pool_id = self._pool_id_for(symbol)
        historical = state.pool_state(pool_id)
        if historical is None:
            return None
        if self._model_self_impact and pool_id in self._self_impact_reserves:
            reserve_in, reserve_out = self._self_impact_reserves[pool_id]
            overridden = _dc_replace(
                historical, reserve_in_atomic=reserve_in, reserve_out_atomic=reserve_out
            )
            # Once this model's own hypothesized trade has been layered on
            # top of the historical snapshot, the reserves no longer describe
            # a verified on-chain state — the route is counterfactual from
            # this point forward for this pool.
            return pool_id, overridden, RouteProvenance.COUNTERFACTUAL
        return pool_id, historical, RouteProvenance.HISTORICAL

    @staticmethod
    def _amm_reserves(pool: PoolState, side: Side) -> tuple[int, int]:
        # See class docstring: reserve_in_atomic/reserve_out_atomic name the
        # pool's base/quote reserves, not a swap direction. A BUY spends the
        # quote reserve and receives from the base reserve; a SELL reverses
        # that role.
        if side is Side.BUY:
            return pool.reserve_out_atomic, pool.reserve_in_atomic
        return pool.reserve_in_atomic, pool.reserve_out_atomic

    def _apply_self_impact(
        self, pool_id: str, pool: PoolState, side: Side, in_amount: int, out_amount: int
    ) -> None:
        if side is Side.BUY:
            new_base = pool.reserve_in_atomic - out_amount
            new_quote = pool.reserve_out_atomic + in_amount
        else:
            new_base = pool.reserve_in_atomic + in_amount
            new_quote = pool.reserve_out_atomic - out_amount
        self._self_impact_reserves[pool_id] = (new_base, new_quote)
