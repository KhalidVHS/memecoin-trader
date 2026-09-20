"""Integration tests for the scheduler — the seam where the audit's findings lived.

These are not unit tests of the modules the loop calls; those live next door and
number in the hundreds. What is tested here is the *order of operations*, which
is where C3, C4, C11 and C12 each came from. Every one of them was a case where
two individually-correct components were wired together wrongly, so every one of
them is invisible to a test that exercises either component alone.

The fakes are deliberately dumb: a broker that records what it was asked to do,
a strategy that returns fixed targets, a quoter that counts calls. A fake that
reimplements the real component's logic cannot catch a wiring bug, because it
will happily agree with whatever wiring it is given.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from memetrader import config as config_mod
from memetrader import ids, loop, quotes, risk
from memetrader.ids import quote_fingerprint
from memetrader.types import (
    CoinSnapshot as Snap,
)
from memetrader.types import (
    DataQuality,
    EvidenceBundle,
    ExecutionMode,
    Fill,
    Mark,
    OrderIntent,
    OrderState,
    PoolRef,
    PriceLadder,
    Provenance,
    Quote,
    Side,
    StrategyDecision,
    TargetPosition,
    TechnicalBrief,
    Technicals,
    TokenMeta,
    TxnCounts,
)
from memetrader.types import (
    MarketSnapshot as MSnap,
)
from memetrader.types import (
    Position as Pos,
)

NOW = 1_800_000_000.0
BONK = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
USDC = TokenMeta(mint=quotes.USDC_MINT, decimals=6, source="test")
TOKEN = TokenMeta(mint=BONK, decimals=5, source="test")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg(tmp_path: Path):
    """The real config, redirected at a temp data dir.

    Loading the real ``config.toml`` rather than hand-building a Config is
    deliberate: it means these tests fail when the shipped config stops being
    able to drive the loop, which is a failure mode a synthetic fixture hides.
    """
    base = config_mod.load()
    return dataclasses.replace(base, data_dir=tmp_path, execution_mode=ExecutionMode.PAPER)


def _snapshot(*, price=0.00002, liquidity=5_000_000.0, ts=NOW) -> MSnap:
    prov = Provenance(source="test", receive_time=ts, event_time=ts, quality=DataQuality.OK)
    pool = PoolRef(
        pair_address="pool1",
        dex_id="raydium",
        base_mint=BONK,
        quote_mint=quotes.USDC_MINT,
        quote_symbol="USDC",
        created_at=ts - 400 * 86_400,
    )
    coin = Snap(
        symbol="BONK",
        mint=BONK,
        price_usd=price,
        liquidity_usd=liquidity,
        volume_24h_usd=9_000_000.0,
        volume_1h_usd=400_000.0,
        fdv_usd=1_500_000_000.0,
        price_change=PriceLadder(m5=0.1, h1=2.0, h6=1.0, h24=3.0),
        txns_m5=TxnCounts(buys=40, sells=30),
        txns_h1=TxnCounts(buys=500, sells=480),
        txns_h24=TxnCounts(buys=9000, sells=8800),
        pool=pool,
        provenance=prov,
    )
    return MSnap(ts=ts, coins={"BONK": coin})


def _quote(*, side: Side, in_atomic: int, out_atomic: int, ts=NOW, impact=0.2) -> Quote:
    in_tok, out_tok = (USDC, TOKEN) if side is Side.BUY else (TOKEN, USDC)
    return Quote(
        symbol="BONK",
        side=side,
        input_token=in_tok,
        output_token=out_tok,
        in_amount_atomic=in_atomic,
        out_amount_atomic=out_atomic,
        min_out_amount_atomic=int(out_atomic * 0.995),
        price_impact_pct=impact,
        route_labels=("Raydium",),
        fingerprint=quote_fingerprint(
            side=side.value,
            input_mint=in_tok.mint,
            output_mint=out_tok.mint,
            in_amount_atomic=in_atomic,
            out_amount_atomic=out_atomic,
            slot=None,
        ),
        requested_at=ts - 0.2,
        received_at=ts,
        expires_at=ts + 10.0,
        reference_price_usd=0.00002,
    )


class FakeBroker:
    """Records orders. Deliberately does not simulate a book."""

    def __init__(self, cfg, *, mode=ExecutionMode.PAPER, positions=None, cash=1000.0):
        self.cfg = cfg
        self._mode = mode
        self._positions = dict(positions or {})
        self.cash_usd = cash
        self.starting_cash_usd = 1000.0
        self.realized_pnl_usd = 0.0
        self.fees_paid_usd = 0.0
        self.gas_paid_usd = 0.0
        self.failed_gas_usd = 0.0
        self.run_id = ids.new_run_id()
        self.orders: list[tuple] = []
        self.reconcile_calls = 0
        self.open_intents: tuple = ()

    @property
    def mode(self):
        return self._mode

    def get_positions(self):
        return dict(self._positions)

    def reconcile(self):
        self.reconcile_calls += 1
        return _Recon(self.open_intents)

    def place_order(self, intent, quote, *, now):
        self.orders.append((intent, quote))
        return Fill(
            fill_id=ids.new_fill_id(),
            order_id=ids.new_order_id(),
            intent_id=intent.intent_id,
            decision_id=intent.decision_id,
            ts=now,
            symbol=intent.symbol,
            side=intent.side,
            state=OrderState.LANDED,
            in_amount_atomic=quote.in_amount_atomic,
            out_amount_atomic=quote.out_amount_atomic,
            token_amount_atomic=(
                quote.out_amount_atomic
                if intent.side is Side.BUY
                else quote.in_amount_atomic
            ),
            token_decimals=TOKEN.decimals,
            quote_fingerprint=quote.fingerprint,
            price_usd=0.00002,
            notional_usd=25.0,
            price_impact_pct=quote.price_impact_pct,
            pool_fee_usd=0.0,
            gas_usd=0.21,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _Recon:
    open_intents: tuple
    ledger_fills: tuple = ()
    journaled_intents: tuple = ()
    replayed_fill_ids: tuple = ()

    @property
    def clean(self) -> bool:
        return True


class FixedStrategy:
    """Returns the targets it was given. Counts how often it was asked."""

    def __init__(self, targets: tuple[TargetPosition, ...]):
        self.targets = targets
        self.calls = 0

    @property
    def strategy_id(self) -> str:
        return "fixed-test"

    def decide(self, evidence, book, *, now):
        self.calls += 1
        return StrategyDecision(
            decision_id=ids.new_decision_id(),
            ts=now,
            strategy_id=self.strategy_id,
            market_read="",
            targets=self.targets,
            diagnostics={},
        )


def _technicals(*, realized_vol_pct: float | None = 45.0) -> TechnicalBrief:
    """Just enough for the sizing rules to have something to divide by.

    ``realized_vol_pct`` is the field that matters: risk sizes inversely to it
    and refuses outright when it is None, so a fixture without it exercises the
    refusal path rather than the sizing path. Defaulted here so tests that are
    about something else do not have to know that.
    """
    # Every other field is explicitly None: "not enough history to compute this"
    # is a claim the type makes room for, and a fixture that invents values for
    # indicators the test does not care about would let a sizing rule quietly
    # start depending on one.
    fields = {
        f.name: None
        for f in dataclasses.fields(Technicals)
        if f.name not in {"timeframe", "candles_used", "realized_vol_pct"}
    }
    h1 = Technicals(
        timeframe="1h", candles_used=60, realized_vol_pct=realized_vol_pct, **fields
    )
    return TechnicalBrief(symbol="BONK", m5=None, h1=h1, flow=None)


def _evidence(snap: MSnap, *, realized_vol_pct: float | None = 45.0):
    return {
        "BONK": EvidenceBundle(
            symbol="BONK",
            snapshot=snap.coins["BONK"],
            technicals=_technicals(realized_vol_pct=realized_vol_pct),
            sentiment=None,
            sentiment_unavailable_reason="sentiment disabled",
        )
    }


def _trader(
    cfg,
    *,
    broker=None,
    targets=(),
    monkeypatch=None,
    snap=None,
    vol=45.0,
    patch_evidence=True,
):
    broker = broker if broker is not None else FakeBroker(cfg)
    t = loop.Trader(
        cfg,
        broker=broker,
        strategy_impl=FixedStrategy(targets),
        client=object(),  # never used; every network call is patched out
        now=lambda: NOW,
    )
    if monkeypatch is not None:
        snap = snap if snap is not None else _snapshot()
        monkeypatch.setattr(t, "snapshot", lambda **kw: snap)
        monkeypatch.setattr(t, "token", lambda symbol, mint: TOKEN)
        if patch_evidence:
            monkeypatch.setattr(t, "evidence", lambda s: _evidence(s, realized_vol_pct=vol))
    return t


# ---------------------------------------------------------------------------
# C11 — the persisted record gates trading
# ---------------------------------------------------------------------------


def test_preflight_refuses_to_trade_while_an_intent_has_no_terminal_state(cfg):
    """An order whose outcome is unknown must stop the process, not be retried.

    This is the C11 scenario end to end: place, die, restart. The successor
    cannot distinguish "the swap landed and we crashed before writing the fill"
    from "the swap never went out", and guessing either way is a real position
    error. Refusing is the only answer available from inside the process.
    """
    broker = FakeBroker(cfg)
    # An `OrderIntent`, not a journal `OpenIntent`: the broker's reconciliation
    # hands back the intents it wrote and could not match to a fill, and those
    # are the real dataclass. The ledger's `OpenIntent` summaries arrive on the
    # other branch of the same refusal. Using the wrong one here passed a `str`
    # where a `Side` was expected and only failed once preflight formatted it.
    broker.open_intents = (
        OrderIntent(
            intent_id="int-1",
            decision_id=None,
            action_id=None,
            run_id="run-0",
            ts=NOW - 60,
            symbol="BONK",
            side=Side.BUY,
            in_amount_atomic=10_000_000,
            max_in_amount_atomic=10_000_000,
            source="strategy",
        ),
    )
    t = _trader(cfg, broker=broker)
    with pytest.raises(loop.StartupRefusal) as exc:
        t.preflight()
    assert "int-1" in str(exc.value)
    assert "BONK" in str(exc.value)


def test_preflight_passes_on_a_clean_record_and_reconciles_the_broker(cfg):
    t = _trader(cfg)
    notes = t.preflight()
    assert t.broker.reconcile_calls == 1
    assert notes == ()


# ---------------------------------------------------------------------------
# C12 — read-only is a capability, not a flag
# ---------------------------------------------------------------------------


def test_read_only_mode_never_places_an_order(cfg, monkeypatch):
    """The old --dry-run checked a boolean at some call sites and not others.

    Mode is now a property of the broker, so a read-only run cannot execute even
    along a path that forgot to look.
    """
    ro = dataclasses.replace(cfg, execution_mode=ExecutionMode.READ_ONLY)
    broker = FakeBroker(ro, mode=ExecutionMode.READ_ONLY)
    t = _trader(
        ro, broker=broker, targets=(TargetPosition("BONK", 25.0),), monkeypatch=monkeypatch
    )
    monkeypatch.setattr(
        quotes,
        "quote_buy_usd",
        lambda *a, **k: _quote(
            side=Side.BUY, in_atomic=25_000_000, out_atomic=1_250_000_000
        ),
    )
    result = t.slow_tick()
    assert broker.orders == []
    assert result.fills == ()
    # The intent is still *formed* — a read-only run should show what it would
    # have done, at the size it would have done it.
    assert len(result.intents) == 1
    assert result.intents[0].in_amount_atomic == 25_000_000


def test_read_only_mode_writes_nothing_to_the_ledger(cfg, monkeypatch, tmp_path):
    ro = dataclasses.replace(cfg, execution_mode=ExecutionMode.READ_ONLY)
    broker = FakeBroker(ro, mode=ExecutionMode.READ_ONLY)
    t = _trader(
        ro, broker=broker, targets=(TargetPosition("BONK", 25.0),), monkeypatch=monkeypatch
    )
    monkeypatch.setattr(
        quotes,
        "quote_buy_usd",
        lambda *a, **k: _quote(
            side=Side.BUY, in_atomic=25_000_000, out_atomic=1_250_000_000
        ),
    )
    t.slow_tick()
    assert not (tmp_path / "ledger.jsonl").exists()


def test_read_only_mode_does_not_liquidate_through_the_stop_path(cfg, monkeypatch):
    """The specific C12 bug: the stop-loss path never consulted the dry-run flag,
    so a 'dry' run could still close a position."""
    ro = dataclasses.replace(cfg, execution_mode=ExecutionMode.READ_ONLY)
    position = Pos(
        symbol="BONK",
        mint=BONK,
        quantity_atomic=125_000_000_000,
        decimals=5,
        avg_entry_price_usd=0.00004,
        opened_at=NOW - 3600,
        cost_basis_usd=50.0,
    )
    broker = FakeBroker(ro, mode=ExecutionMode.READ_ONLY, positions={"BONK": position})
    t = _trader(ro, broker=broker, monkeypatch=monkeypatch)
    # Priced far below entry: the stop is unambiguously breached.
    monkeypatch.setattr(
        t,
        "marks",
        lambda snap, *, now: {
            "BONK": Mark(
                symbol="BONK",
                price_usd=0.00002,
                basis="route",
                provenance=Provenance(source="test", receive_time=now),
            )
        },
    )
    monkeypatch.setattr(
        quotes,
        "quote_sell_tokens",
        lambda *a, **k: _quote(
            side=Side.SELL, in_atomic=125_000_000_000, out_atomic=25_000_000
        ),
    )
    result = t.fast_tick()
    assert broker.orders == []
    assert result.fills == ()


# ---------------------------------------------------------------------------
# C3 — the quote binds the size
# ---------------------------------------------------------------------------


def test_the_order_is_quoted_at_the_risk_bounded_size_not_the_wanted_size(cfg, monkeypatch):
    """C3: risk used to clamp a size *after* the quote and the broker then filled
    the clamped notional against the unclamped quote — a price nobody offered
    for that size. The bound must come first and the quote must follow it."""
    broker = FakeBroker(cfg)
    t = _trader(
        cfg,
        broker=broker,
        targets=(TargetPosition("BONK", 500.0),),
        monkeypatch=monkeypatch,
    )
    # Risk permits far less than the target asks for.
    monkeypatch.setattr(
        type(t.risk),
        "entry_bounds",
        lambda self, symbol, **kw: loop.RiskBounds(
            symbol=symbol, side=Side.BUY, max_notional_usd=30.0, binding_rule="position_cap"
        ),
    )
    seen: list[float] = []

    def fake_quote(_cfg, *, usd_notional, **kw):
        seen.append(usd_notional)
        return _quote(
            side=Side.BUY, in_atomic=int(usd_notional * 1_000_000), out_atomic=1_500_000_000
        )

    monkeypatch.setattr(quotes, "quote_buy_usd", fake_quote)
    t.slow_tick()

    assert seen == [30.0], "the quote must be requested at the permitted size, once"
    intent, quote = broker.orders[0]
    assert intent.in_amount_atomic == quote.in_amount_atomic
    assert intent.max_in_amount_atomic == quote.in_amount_atomic


def test_a_quote_that_fails_reconfirmation_is_not_executed(cfg, monkeypatch):
    """``confirm_quote`` re-runs the quote-dependent rules against the quote that
    will actually be sent. A quote acceptable at request time can be 6% impact by
    the time it comes back, and that is a different order."""
    broker = FakeBroker(cfg)
    t = _trader(
        cfg, broker=broker, targets=(TargetPosition("BONK", 25.0),), monkeypatch=monkeypatch
    )
    monkeypatch.setattr(
        quotes,
        "quote_buy_usd",
        lambda *a, **k: _quote(
            side=Side.BUY, in_atomic=25_000_000, out_atomic=1_250_000_000
        ),
    )
    monkeypatch.setattr(
        type(t.risk),
        "confirm_quote",
        lambda self, bounds, quote, **kw: dataclasses.replace(
            bounds,
            max_notional_usd=0.0,
            vetoes=("price_impact",),
            reasons=("6.1% impact at this size",),
            binding_rule="price_impact",
        ),
    )
    result = t.slow_tick()
    assert broker.orders == []
    assert result.fills == ()


# ---------------------------------------------------------------------------
# C4 — there is no such thing as a fallback quote
# ---------------------------------------------------------------------------


def test_no_quote_means_no_trade(cfg, monkeypatch):
    """C4: the old code synthesised a 'degraded' quote from the mid price when
    Jupiter was unreachable and then executed it as though a router had offered
    it. A missing quote is a missing price."""
    broker = FakeBroker(cfg)
    t = _trader(
        cfg, broker=broker, targets=(TargetPosition("BONK", 25.0),), monkeypatch=monkeypatch
    )
    monkeypatch.setattr(quotes, "quote_buy_usd", lambda *a, **k: None)
    result = t.slow_tick()
    assert broker.orders == []
    assert result.fills == ()


def test_unknown_token_decimals_block_the_order(cfg, monkeypatch):
    """Without decimals every atomic amount is a guess, and a guess of 9 where
    the answer is 5 is a 10,000x sizing error."""
    broker = FakeBroker(cfg)
    t = _trader(cfg, broker=broker, targets=(TargetPosition("BONK", 25.0),))
    monkeypatch.setattr(t, "snapshot", lambda **kw: _snapshot())
    monkeypatch.setattr(t, "token", lambda symbol, mint: None)
    called: list[int] = []
    monkeypatch.setattr(quotes, "quote_buy_usd", lambda *a, **k: called.append(1))
    t.slow_tick()
    assert called == []
    assert broker.orders == []


# ---------------------------------------------------------------------------
# Targets, not orders
# ---------------------------------------------------------------------------


def test_a_target_already_held_produces_no_order(cfg, monkeypatch):
    """The point of targets over actions. Under the old design the model emitting
    'BUY $50 BONK' on two consecutive ticks opened two positions, because the
    model had to remember the book and did not."""
    position = Pos(
        symbol="BONK",
        mint=BONK,
        quantity_atomic=125_000_000_000,
        decimals=5,
        avg_entry_price_usd=0.00002,
        opened_at=NOW - 3600,
        cost_basis_usd=25.0,
    )
    broker = FakeBroker(cfg, positions={"BONK": position})
    t = _trader(
        cfg, broker=broker, targets=(TargetPosition("BONK", 25.0),), monkeypatch=monkeypatch
    )
    monkeypatch.setattr(
        t,
        "marks",
        lambda snap, *, now: {
            "BONK": Mark(
                symbol="BONK",
                price_usd=0.00002,
                basis="route",
                provenance=Provenance(source="test", receive_time=now),
            )
        },
    )
    result = t.slow_tick()
    assert broker.orders == []
    assert result.intents == ()


def test_a_drift_inside_the_rebalance_band_is_left_alone(cfg, monkeypatch):
    """Rebalancing a $25 position by $5 pays gas, spread and impact to move
    nothing. The band is a cost control, not a nicety."""
    position = Pos(
        symbol="BONK",
        mint=BONK,
        quantity_atomic=100_000_000_000,
        decimals=5,
        avg_entry_price_usd=0.00002,
        opened_at=NOW - 3600,
        cost_basis_usd=20.0,
    )
    broker = FakeBroker(cfg, positions={"BONK": position})
    t = _trader(
        cfg, broker=broker, targets=(TargetPosition("BONK", 25.0),), monkeypatch=monkeypatch
    )
    monkeypatch.setattr(
        t,
        "marks",
        lambda snap, *, now: {
            "BONK": Mark(
                symbol="BONK",
                price_usd=0.00002,
                basis="route",
                provenance=Provenance(source="test", receive_time=now),
            )
        },
    )
    # Held $20 against a $25 target: a $5 delta, inside the $15 band.
    assert cfg.strategy.rebalance_band_usd > 5.0
    t.slow_tick()
    assert broker.orders == []


def test_an_unmarkable_position_is_never_rebalanced_against(cfg, monkeypatch):
    """C8 in the sizing layer. A delta computed against an unknown holding value
    is arithmetic on a guess, and the resulting order is that guess with a dollar
    sign in front of it."""
    position = Pos(
        symbol="BONK",
        mint=BONK,
        quantity_atomic=125_000_000_000,
        decimals=5,
        avg_entry_price_usd=0.00002,
        opened_at=NOW - 3600,
        cost_basis_usd=25.0,
    )
    broker = FakeBroker(cfg, positions={"BONK": position})
    t = _trader(
        cfg, broker=broker, targets=(TargetPosition("BONK", 0.0),), monkeypatch=monkeypatch
    )
    monkeypatch.setattr(
        t,
        "marks",
        lambda snap, *, now: {
            "BONK": Mark(
                symbol="BONK",
                price_usd=None,
                basis="unavailable",
                provenance=Provenance(source="test", receive_time=now),
                reason="no route and no mid",
            )
        },
    )
    result = t.slow_tick()
    assert "BONK" in result.portfolio.unmarkable
    assert result.portfolio.total_value_usd is None, "an unmarkable leg must void the total"
    assert broker.orders == []


# ---------------------------------------------------------------------------
# Halt behaviour
# ---------------------------------------------------------------------------


def test_a_halted_run_does_not_call_the_strategy(cfg, monkeypatch):
    """Halted means exits only. Calling the strategy anyway would burn a model
    call to produce targets that are guaranteed to be vetoed, and would write a
    decision that never had a chance of executing."""
    broker = FakeBroker(cfg)
    strategy = FixedStrategy((TargetPosition("BONK", 25.0),))
    t = loop.Trader(
        cfg, broker=broker, strategy_impl=strategy, client=object(), now=lambda: NOW
    )
    monkeypatch.setattr(t, "snapshot", lambda **kw: _snapshot())
    monkeypatch.setattr(t, "token", lambda symbol, mint: TOKEN)
    monkeypatch.setattr(
        type(t.risk.continuous),
        "evaluate",
        lambda self, **kw: loop.RiskState(
            ts=NOW, halted=True, halt_reasons=("max drawdown 24.1% > 20%",)
        ),
    )
    result = t.slow_tick()
    assert strategy.calls == 0
    assert result.decision is None
    assert result.risk_state.halted
    assert any("halted" in n for n in result.notes)


def test_an_open_breaker_marks_data_health_bad(cfg, monkeypatch):
    """An open circuit breaker means a vendor has failed repeatedly and is being
    left alone. Trading through that is trading on a book marked from whatever
    was cached — precisely the condition under which a stop cannot fire."""
    t = _trader(cfg, monkeypatch=monkeypatch)
    assert t._data_health_ok() is True
    monkeypatch.setattr(
        type(t.breaker), "open_hosts", property(lambda self: ("api.jup.ag",))
    )
    assert t._data_health_ok() is False


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_a_strategy_exception_is_an_error_not_a_hold(cfg, monkeypatch):
    """A failed tick and a tick that decided to hold nothing must not render the
    same. Conflating them is how an outage reads as a flat market afterwards."""

    class Exploding(FixedStrategy):
        def decide(self, evidence, book, *, now):
            raise RuntimeError("boom")

    t = loop.Trader(
        cfg,
        broker=FakeBroker(cfg),
        strategy_impl=Exploding(()),
        client=object(),
        now=lambda: NOW,
    )
    monkeypatch.setattr(t, "snapshot", lambda **kw: _snapshot())
    result = t.slow_tick()
    assert result.error == "boom"
    assert result.decision is None
    # The decision baseline must NOT advance: the next successful tick should
    # measure liquidity across the whole outage, not claim 15 minutes over
    # evidence nobody read.
    assert t.decision_baseline is None


def test_a_broker_refusal_does_not_stop_the_loop(cfg, monkeypatch):
    broker = FakeBroker(cfg)

    def refuse(intent, quote, *, now):
        raise loop.BrokerError("insufficient cash")

    broker.place_order = refuse
    t = _trader(
        cfg, broker=broker, targets=(TargetPosition("BONK", 25.0),), monkeypatch=monkeypatch
    )
    monkeypatch.setattr(
        quotes,
        "quote_buy_usd",
        lambda *a, **k: _quote(
            side=Side.BUY, in_atomic=25_000_000, out_atomic=1_250_000_000
        ),
    )
    result = t.slow_tick()
    assert result.fills == ()
    assert result.error is None, "a refused order is an ordinary tick outcome"


# ---------------------------------------------------------------------------
# Risk ledger persistence
# ---------------------------------------------------------------------------


def test_the_risk_ledger_survives_a_restart(cfg):
    """Halt history in memory only means a restart clears a drawdown halt and
    un-quarantines a symbol that just stopped out — turning a circuit breaker
    into 'halt until someone restarts it', which is its exact opposite."""
    ledger = risk.RiskLedger(
        peak_value_usd=1_400.0,
        day_start_value_usd=1_200.0,
        day_start_ts=NOW - 3600,
        window_start_value_usd=1_000.0,
        window_start_ts=NOW - 86_400,
        consecutive_failures=2,
        quarantined_until={"BONK": NOW + 1800},
        last_entry_ts={"BONK": NOW - 300},
    )
    loop._save_risk_ledger(cfg, ledger)
    back = loop._load_risk_ledger(cfg)
    assert back.peak_value_usd == 1_400.0
    assert back.consecutive_failures == 2
    assert back.quarantined_until == {"BONK": NOW + 1800}
    assert back.last_entry_ts == {"BONK": NOW - 300}


def test_an_unreadable_risk_ledger_starts_halted(cfg, tmp_path):
    """NOT a silent reset to empty. An unreadable ledger means the halt history
    is gone, and resuming as though nothing ever went wrong is the failure this
    is trying to prevent."""
    (tmp_path / "risk_ledger.json").write_text("{not json", encoding="utf-8")
    back = loop._load_risk_ledger(cfg)
    assert back.manual_halt is True
    assert "unreadable" in (back.manual_halt_reason or "")


def test_a_missing_risk_ledger_is_an_empty_one_not_a_halt(cfg):
    """A first run has no history, which is a different thing from lost history."""
    assert loop._load_risk_ledger(cfg) == risk.EMPTY_LEDGER


# ---------------------------------------------------------------------------
# Stops
# ---------------------------------------------------------------------------


def test_the_stop_price_has_exactly_one_definition(cfg, monkeypatch):
    """There used to be three, disagreeing at the third decimal, so a position
    could be past its stop in the fast tick and not in the slow one. The loop
    must not recompute it — it asks ``portfolio.stop_loss_breaches``."""
    import inspect

    source = inspect.getsource(loop)
    assert "stop_loss_breaches" in source
    # No arithmetic on stop_loss_pct anywhere in the loop except the log message.
    for line in source.splitlines():
        if "stop_loss_pct" in line and "stop_loss_breaches" not in line:
            assert (
                "log." in line
                or "#" in line.strip()[:1]
                or 'f"' in line
                or "self.cfg.risk.stop_loss_pct," in line
            ), line


def test_a_degraded_stop_reference_is_reported_not_hidden(cfg, monkeypatch, caplog):
    """A stop firing against a haircut mid rather than a route quote is allowed —
    freezing stops during an outage is the more dangerous failure on an asset
    that gaps — but it is not the same event and must not read as one."""
    position = Pos(
        symbol="BONK",
        mint=BONK,
        quantity_atomic=125_000_000_000,
        decimals=5,
        avg_entry_price_usd=0.00004,
        opened_at=NOW - 3600,
        cost_basis_usd=50.0,
    )
    broker = FakeBroker(cfg, positions={"BONK": position})
    t = _trader(cfg, broker=broker, monkeypatch=monkeypatch)
    monkeypatch.setattr(
        t,
        "marks",
        lambda snap, *, now: {
            "BONK": Mark(
                symbol="BONK",
                price_usd=0.0000196,
                basis="mid",
                haircut_pct=2.0,
                provenance=Provenance(source="dexscreener", receive_time=now),
                reason="no route quote; mid haircut 2.0%",
            )
        },
    )
    monkeypatch.setattr(
        quotes,
        "quote_sell_tokens",
        lambda *a, **k: _quote(
            side=Side.SELL, in_atomic=125_000_000_000, out_atomic=24_500_000
        ),
    )
    with caplog.at_level("WARNING"):
        result = t.fast_tick()
    assert result.stop_exits == ("BONK",)
    assert any("mid reference" in r.getMessage() for r in caplog.records)


def test_a_stop_exit_sells_the_exact_atomic_quantity_held(cfg, monkeypatch):
    """Not a dollar amount, and not a rounded one. C2 was SELL quotes and ledger
    quantities disagreeing because dollars were the unit of record; a full exit
    must name the integer it is disposing of."""
    position = Pos(
        symbol="BONK",
        mint=BONK,
        quantity_atomic=123_456_789_100,
        decimals=5,
        avg_entry_price_usd=0.00004,
        opened_at=NOW - 3600,
        cost_basis_usd=50.0,
    )
    broker = FakeBroker(cfg, positions={"BONK": position})
    t = _trader(cfg, broker=broker, monkeypatch=monkeypatch)
    monkeypatch.setattr(
        t,
        "marks",
        lambda snap, *, now: {
            "BONK": Mark(
                symbol="BONK",
                price_usd=0.00002,
                basis="route",
                provenance=Provenance(source="test", receive_time=now),
            )
        },
    )
    seen: list[int] = []

    def fake_sell(_cfg, *, token_amount_atomic, **kw):
        seen.append(token_amount_atomic)
        return _quote(side=Side.SELL, in_atomic=token_amount_atomic, out_atomic=24_691_357)

    monkeypatch.setattr(quotes, "quote_sell_tokens", fake_sell)
    t.fast_tick()
    assert seen == [123_456_789_100]


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


def test_the_scheduler_uses_a_monotonic_clock(cfg):
    """An NTP correction moving wall time backwards used to stall the loop until
    it caught up; moving it forwards fired every missed tick at once."""
    import inspect

    src = inspect.getsource(loop.Trader.run)
    code = [ln for ln in src.splitlines() if not ln.strip().startswith("#")]
    assert any("time.monotonic()" in ln for ln in code)
    assert not any("time.time()" in ln for ln in code)


def test_the_cadence_is_measured_from_the_deadline_not_from_completion(cfg):
    """Scheduling the next tick relative to completion lets a run of slow ticks
    drift the decision interval out indefinitely."""
    import inspect

    src = inspect.getsource(loop.Trader.run)
    assert "max(mono, next_slow) + slow" in src


# ---------------------------------------------------------------------------
# Evidence honesty
# ---------------------------------------------------------------------------


def test_a_disabled_sentiment_stream_is_reported_as_unavailable_not_as_zero(
    cfg, monkeypatch
):
    """'Missing is never zero' at the top of the pipeline. A disabled stream must
    say it is disabled, so a strategy discounts rather than trading on a blank."""
    off = dataclasses.replace(
        cfg, sentiment=dataclasses.replace(cfg.sentiment, enabled=False)
    )
    t = _trader(off, monkeypatch=monkeypatch, patch_evidence=False)
    bundles = t.evidence(_snapshot())
    assert bundles["BONK"].sentiment is None
    assert "disabled" in (bundles["BONK"].sentiment_unavailable_reason or "")


def test_the_liquidity_window_is_the_real_gap_not_the_configured_cadence(cfg, monkeypatch):
    """A long tick, a vendor outage or a restart all move it, and the trend is
    labelled with this number. Claiming 15 minutes over a 3-hour gap would
    misdescribe the signal the system ranks highest."""
    t = _trader(cfg, monkeypatch=monkeypatch, patch_evidence=False)
    old = _snapshot(ts=NOW - 10_800, liquidity=6_000_000.0)
    t.decision_baseline = old
    seen: list[float | None] = []
    monkeypatch.setattr(
        loop.signals,
        "brief",
        lambda snap, prev, *, elapsed_seconds: seen.append(elapsed_seconds),
    )
    t.evidence(_snapshot())
    assert seen == [10_800.0]
    assert seen[0] != cfg.cadence.slow_tick_seconds


# ---------------------------------------------------------------------------
# Journalling order
# ---------------------------------------------------------------------------


def test_the_intent_is_journaled_before_the_order_is_placed(cfg, monkeypatch, tmp_path):
    """The intent ID is the idempotency key. If it is written after the swap, a
    crash in between leaves a position with nothing on disk explaining it — which
    is C11 with the two files swapped."""
    broker = FakeBroker(cfg)
    order_of_events: list[str] = []
    t = _trader(
        cfg, broker=broker, targets=(TargetPosition("BONK", 25.0),), monkeypatch=monkeypatch
    )
    real_append = t.ledger.append_intent

    def spy_intent(intent, *, state=OrderState.PROPOSED):
        order_of_events.append("intent")
        return real_append(intent, state=state)

    def spy_place(intent, quote, *, now):
        order_of_events.append("place")
        return FakeBroker.place_order(broker, intent, quote, now=now)

    monkeypatch.setattr(t.ledger, "append_intent", spy_intent)
    broker.place_order = spy_place
    monkeypatch.setattr(
        quotes,
        "quote_buy_usd",
        lambda *a, **k: _quote(
            side=Side.BUY, in_atomic=25_000_000, out_atomic=1_250_000_000
        ),
    )
    t.slow_tick()
    assert order_of_events == ["intent", "place"]


def test_a_landed_fill_is_journaled_and_the_intent_reaches_a_terminal_state(
    cfg, monkeypatch, tmp_path
):
    broker = FakeBroker(cfg)
    t = _trader(
        cfg, broker=broker, targets=(TargetPosition("BONK", 25.0),), monkeypatch=monkeypatch
    )
    monkeypatch.setattr(
        quotes,
        "quote_buy_usd",
        lambda *a, **k: _quote(
            side=Side.BUY, in_atomic=25_000_000, out_atomic=1_250_000_000
        ),
    )
    t.slow_tick()
    rows = list(loop.journal.read(tmp_path / "ledger.jsonl"))
    kinds = [r["kind"] for r in rows]
    assert "intent" in kinds and "fill" in kinds and "decision" in kinds
    # And nothing is left open.
    assert t.ledger.open_intents() == ()


# ---------------------------------------------------------------------------
# Marking
# ---------------------------------------------------------------------------


def test_gas_on_failed_swaps_is_counted_in_the_book(cfg, monkeypatch):
    """A failed swap still pays full gas. Reporting only successful-swap gas
    understates the cost of exactly the condition that produces the most
    failures."""
    broker = FakeBroker(cfg)
    broker.gas_paid_usd = 1.05
    broker.failed_gas_usd = 0.63
    t = _trader(cfg, broker=broker, monkeypatch=monkeypatch)
    book = t.book(_snapshot(), now=NOW)
    assert book.gas_paid_usd == pytest.approx(1.68)


def test_a_book_with_no_positions_is_fully_marked(cfg, monkeypatch):
    t = _trader(cfg, monkeypatch=monkeypatch)
    book = t.book(_snapshot(), now=NOW)
    assert book.fully_marked
    assert book.total_value_usd == pytest.approx(1000.0)
    assert not math.isnan(book.total_value_usd)
