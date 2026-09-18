"""Code decides, the model proposes.

``check`` is the last thing between a language model's opinion and the user's
money, so it is a pure function of (proposal, book, snapshot, config, now) with
no I/O, no clock of its own and no hidden state. Every rule names itself in
``RiskVerdict.rule`` and writes a reason a trader could act on — those reasons
go into the decision log and are fed back to the model on the next tick, so
"BUY $400 rejected: would put BONK at 41% of book, limit is 30%" is worth far
more than "max_position_pct violated".

Two rules **clamp instead of rejecting**. Sizing is a negotiation, not a veto:
if the model wants $400 and only $200 is legal, trade $200 and tell it why.
Only when the clamped size falls below ``min_trade_usd`` does the clamp become
a rejection.

A ``source="stop_loss"`` proposal bypasses four rules: the two sizing rules
(``min_trade_size``, ``max_position_pct``) and the two pool-health rules
(``max_price_impact``, ``min_liquidity``, plus the ``missing_snapshot`` check
that exists only to feed the latter). The sizing bypass is obvious — a forced
exit must not be blocked by a rule about how big a trade should be. The
pool-health bypass is the one worth stating out loud: those two rules fire
hardest when a pool is collapsing, which is the exact scenario the stop exists
for, so enforcing them there traps the position instead of protecting it. Each
bypass is recorded in ``RiskVerdict.notes`` and logged as a warning, so a bad
forced fill is visible after the fact rather than silent.

Still enforced on a forced exit: ``stale_data`` (the fast tick retries in 60s,
so blocking costs a minute, not the position), ``missing_quote`` (there is
nothing to fill against) and ``no_position``.

Deliberately absent: any max-trades-per-day cap and any minimum-hold-time. Both
were considered and removed. Do not reintroduce them.
"""

from __future__ import annotations

from .broker import pool_fee_rate
from .config import Config
from .types import (
    FillQuote,
    MarketSnapshot,
    PortfolioState,
    RiskVerdict,
    Side,
    TradeProposal,
)

#: Clamped BUY sizes are shaved by this factor before being approved. The
#: broker raises ``InsufficientCash`` on a strict ``<`` comparison, and a size
#: computed as "exactly all the cash" can land a half-ulp over the line once
#: the fee is re-multiplied there. One part in 10^12 of a dollar is invisible
#: to the user and removes the failure mode entirely.
_CLAMP_SAFETY = 1.0 - 1e-12


def _usd(amount: float) -> str:
    return f"${amount:,.2f}"


def _pct(value: float) -> str:
    return f"{value:.1f}%"


def _reject(
    symbol: str, rule: str, reason: str, notes: tuple[str, ...] = ()
) -> RiskVerdict:
    """Every verdict carries its ``symbol``, approved or not.

    Rejections are replayed to the model on the next tick so it stops
    re-proposing illegal trades. With three coins in play, a bare
    "max_position_pct" line left the model to infer which coin it fired on from
    the prose; the field makes it unambiguous. The symbol stays in ``reason``
    too, because that is what a human reads.
    """
    return RiskVerdict(
        approved=False,
        approved_usd=0.0,
        symbol=symbol,
        rule=rule,
        reason=reason,
        notes=notes,
    )


def check(
    proposal: TradeProposal,
    state: PortfolioState,
    snapshot: MarketSnapshot,
    cfg: Config,
    *,
    now: float,
) -> RiskVerdict:
    """Judge one proposal. Returns an approval (possibly clamped) or a rejection.

    ``notes`` carries every clamp that was applied; ``rule`` and ``reason`` are
    populated only on rejection.
    """
    risk = cfg.risk
    symbol = proposal.symbol
    side = Side(proposal.side)
    forced = proposal.source == "stop_loss"
    notes: list[str] = []

    # -- stale_data --------------------------------------------------------
    age = snapshot.age_seconds(now)
    if age > risk.max_snapshot_age_seconds:
        return _reject(
            symbol,
            "stale_data",
            f"{side} {symbol} rejected: market data is {age:.0f}s old and the "
            f"limit is {risk.max_snapshot_age_seconds:.0f}s — refusing to "
            f"trade blind",
        )

    # -- missing_quote -----------------------------------------------------
    # Checked before everything downstream because price impact, the pool fee
    # and the cash requirement all read off the quote. A forced exit cannot
    # argue with this one: there is nothing to fill against.
    quote = proposal.quote
    if quote is None:
        return _reject(
            symbol,
            "missing_quote",
            f"{side} {symbol} rejected: no executable route was returned, so "
            f"there is no price to trade at",
        )

    # -- max_price_impact --------------------------------------------------
    # Bypassed for a forced exit, and this is the bypass that matters most. Both
    # this rule and the liquidity floor below fire hardest exactly when a pool is
    # collapsing — which is precisely the moment the stop exists for. Blocking
    # the exit there does not protect the position, it traps it, and the cost of
    # being trapped is unbounded while the cost of a bad fill is bounded by what
    # is left. So the rules become warnings on the verdict instead of a veto.
    if quote.price_impact_pct > risk.max_price_impact_pct:
        if not forced:
            return _reject(
                symbol,
                "max_price_impact",
                f"{side} {_usd(proposal.usd_notional)} of {symbol} rejected: price "
                f"impact {quote.price_impact_pct:.2f}% exceeds the "
                f"{risk.max_price_impact_pct:.2f}% limit — the pool is too thin "
                f"for this size",
            )
        notes.append(
            f"forced exit accepting {quote.price_impact_pct:.2f}% price impact, "
            f"over the {risk.max_price_impact_pct:.2f}% limit"
        )

    # -- min_liquidity -----------------------------------------------------
    coin = snapshot.coins.get(symbol)
    if coin is None:
        if not forced:
            return _reject(
                symbol,
                "missing_snapshot",
                f"{side} {symbol} rejected: no market snapshot for {symbol} this "
                f"tick, so its liquidity cannot be verified",
            )
        # A forced exit already has a routable quote; the snapshot was only ever
        # needed to check liquidity, which it is bypassing anyway.
        notes.append(f"forced exit with no {symbol} snapshot to verify liquidity against")
    elif coin.liquidity_usd < risk.min_liquidity_usd:
        if not forced:
            return _reject(
                symbol,
                "min_liquidity",
                f"{side} {symbol} rejected: pool liquidity is "
                f"{_usd(coin.liquidity_usd)}, below the "
                f"{_usd(risk.min_liquidity_usd)} floor — at this depth a 5m price "
                f"move is one swap, not a trend",
            )
        notes.append(
            f"forced exit from a draining pool: liquidity {_usd(coin.liquidity_usd)} "
            f"is below the {_usd(risk.min_liquidity_usd)} floor"
        )

    # -- min_trade_size ----------------------------------------------------
    notional = float(proposal.usd_notional)
    if not forced and notional < risk.min_trade_usd:
        return _reject(
            symbol,
            "min_trade_size",
            f"{side} {_usd(notional)} of {symbol} rejected: below the "
            f"{_usd(risk.min_trade_usd)} minimum — fees and gas would eat it",
        )

    if side is Side.BUY:
        approved = _size_buy(proposal, state, cfg, quote, notes)
    else:
        approved = _size_sell(proposal, state, notes)

    if isinstance(approved, RiskVerdict):  # a sizing rule rejected outright
        return approved

    # -- min_trade_size, again, on the clamped size ------------------------
    # A clamp that lands in dust is a rejection, not an approval of dust.
    if not forced and approved < risk.min_trade_usd:
        return _reject(
            symbol,
            "min_trade_size",
            f"{side} {symbol} rejected: the largest legal size here is "
            f"{_usd(approved)}, below the {_usd(risk.min_trade_usd)} minimum",
            notes=tuple(notes),
        )

    return RiskVerdict(
        approved=True, approved_usd=approved, symbol=symbol, notes=tuple(notes)
    )


def _size_buy(
    proposal: TradeProposal,
    state: PortfolioState,
    cfg: Config,
    quote: FillQuote,
    notes: list[str],
) -> float | RiskVerdict:
    """Apply the two BUY clamps in order, tighter one wins."""
    risk = cfg.risk
    symbol = proposal.symbol
    forced = proposal.source == "stop_loss"
    approved = float(proposal.usd_notional)

    # -- max_position_pct --------------------------------------------------
    # Bypassed for a forced exit. (A stop_loss proposal is a SELL in practice;
    # the bypass is written here anyway so the rule is stated once, in full.)
    if not forced:
        held = state.position_values_usd.get(symbol, 0.0)
        limit_usd = risk.max_position_pct * state.total_value_usd
        if held + approved > limit_usd:
            headroom = limit_usd - held
            resulting = held + approved
            share = (
                100.0 * resulting / state.total_value_usd
                if state.total_value_usd > 0
                else float("inf")
            )
            limit_share = 100.0 * risk.max_position_pct
            if headroom < risk.min_trade_usd:
                return _reject(
                    symbol,
                    "max_position_pct",
                    f"BUY {_usd(approved)} of {symbol} rejected: would put "
                    f"{symbol} at {_pct(share)} of a "
                    f"{_usd(state.total_value_usd)} book, limit is "
                    f"{_pct(limit_share)} — only {_usd(max(headroom, 0.0))} of "
                    f"headroom is left",
                    notes=tuple(notes),
                )
            notes.append(
                f"max_position_pct: BUY {_usd(approved)} clamped to "
                f"{_usd(headroom)} — full size would put {symbol} at "
                f"{_pct(share)} of a {_usd(state.total_value_usd)} book, limit "
                f"is {_pct(limit_share)}"
            )
            approved = headroom

    # -- insufficient_cash -------------------------------------------------
    # Never bypassed: a forced exit is a SELL, and no amount of urgency creates
    # money that is not there.
    gas = cfg.execution.gas_usd_per_swap
    # Must match ``broker.place_order`` exactly, or risk approves a size the
    # broker then refuses with InsufficientCash. On a real Jupiter route the
    # pool fee is already inside ``quote.price_usd`` and this is 0.0; see
    # ``broker.pool_fee_rate`` for why.
    fee_rate = pool_fee_rate(quote)
    fee_pct = fee_rate * 100.0
    needed = approved * (1.0 + fee_rate) + gas
    if needed > state.cash_usd:
        affordable = (state.cash_usd - gas) / (1.0 + fee_rate) * _CLAMP_SAFETY
        if affordable < risk.min_trade_usd:
            return _reject(
                symbol,
                "insufficient_cash",
                f"BUY {_usd(approved)} of {symbol} rejected: needs "
                f"{_usd(needed)} including the {fee_pct:.2f}% pool fee and "
                f"{_usd(gas)} gas, but cash is {_usd(state.cash_usd)} — the "
                f"most that could be bought is {_usd(max(affordable, 0.0))}, "
                f"below the {_usd(risk.min_trade_usd)} minimum",
                notes=tuple(notes),
            )
        notes.append(
            f"insufficient_cash: BUY {_usd(approved)} clamped to "
            f"{_usd(affordable)} — the full size needs {_usd(needed)} with the "
            f"{fee_pct:.2f}% pool fee and {_usd(gas)} gas, and cash is "
            f"{_usd(state.cash_usd)}"
        )
        approved = affordable

    return approved


def _size_sell(
    proposal: TradeProposal, state: PortfolioState, notes: list[str]
) -> float | RiskVerdict:
    """SELL has one rule and one clamp, and neither is ever bypassed: you
    cannot sell what you do not own, forced exit or not."""
    symbol = proposal.symbol
    approved = float(proposal.usd_notional)

    if symbol not in state.positions:
        return _reject(
            symbol,
            "no_position",
            f"SELL {_usd(approved)} of {symbol} rejected: there is no open "
            f"{symbol} position to sell",
            notes=tuple(notes),
        )

    position_value = state.position_values_usd.get(symbol, 0.0)
    if approved > position_value:
        notes.append(
            f"sell clamped: {_usd(approved)} exceeds the "
            f"{_usd(position_value)} {symbol} position, selling all of it"
        )
        approved = position_value

    return approved


__all__ = ["check"]
