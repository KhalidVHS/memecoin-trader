"""Broker arithmetic, asserted against hand-computed numbers.

Every expected value in this file was worked out on paper first and is written
as a literal, not as a re-implementation of the code under test. A test that
recomputes the production formula proves nothing; these prove the formula.

The one exception is the conservation block at the bottom, which is deliberately
a *property* test rather than a worked example: the audit's Phase 1 exit gate is
"every simulated fill exactly reconciles cash and token atomic units", and the
only honest way to assert "exactly" over an arbitrary sequence is to recompute
both ledgers independently, in integers, and demand equality — not
``pytest.approx``.

No network, no real files — every test gets its own ``tmp_path`` data dir.

Tests carried over unchanged in intent
--------------------------------------
Cold start, copy-on-``get_positions``, the hand-computed round trip (the
1000 -> BUY $400 -> SELL @ 0.000025 -> 1099.58 worked example), cost basis
includes costs, partial-sell proportionality, two-partials-equal-one-full, dust
deletion, sell-with-no-position, failed-tx gas, failed sell leaves the position,
deterministic rng, insufficient cash, scale-in averaging, and the whole
persistence block. Only the call shape changed: ``place_order`` now takes an
``OrderIntent`` and a ``Quote`` instead of a symbol, a side and a dollar amount.

Tests deleted, and why
----------------------
* ``test_a_degraded_quote_round_trips`` and ``test_partial_sell_degraded``.
  There is no degraded quote any more (audit C4); ``ValuationEstimate`` cannot
  reach ``place_order`` and there is a test asserting exactly that instead.
* ``test_the_fee_branch_is_decided_by_the_quote_not_the_config``. Both branches
  are gone: no quote pays an explicit pool fee, because ``outAmount`` is already
  net of every hop (the 45x double-count). Replaced by
  ``test_a_routed_quote_pays_no_explicit_pool_fee``.
* ``test_overselling_is_clamped_to_the_position``. Clamping filled a smaller
  size against a larger quote — audit C3 in miniature. Replaced by
  ``test_overselling_is_refused_rather_than_clamped``.
* ``test_non_positive_notional_is_rejected`` / ``..._price_...``. Both are now
  unrepresentable: ``Quote.__post_init__`` refuses a zero input amount, so there
  is no broker-level check left to test. Asserted once at the type boundary in
  ``test_a_zero_input_quote_cannot_be_constructed``.
"""

from __future__ import annotations

import inspect
import json
import random
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from memetrader import config
from memetrader.broker import (
    SCHEMA_VERSION,
    BrokerError,
    FillModel,
    InsufficientCash,
    InsufficientPosition,
    LedgerCorrupt,
    LiveModeUnsupported,
    LocalPaperBroker,
    NoPosition,
    QuoteBindingError,
    QuoteRejected,
    ReadOnlyViolation,
    assert_live_supported,
    pool_fee_micro,
)
from memetrader.ids import new_decision_id, new_intent_id, quote_fingerprint
from memetrader.quotes import USDC
from memetrader.types import (
    ExecutionMode,
    OrderIntent,
    OrderState,
    Quote,
    Side,
    TokenMeta,
    ValidationError,
    ValuationEstimate,
)

MINT = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
DECIMALS = 5
BONK = TokenMeta(mint=MINT, decimals=DECIMALS, source="test:verified", verified=True)

#: Every quote is timestamped here unless a test cares about staleness.
NOW = 1_700_000_000.0

_MICRO = 10**6
_UNIT = 10**DECIMALS


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_cfg(
    tmp_path: Path,
    *,
    starting_cash_usd: float = 1000.0,
    failed_tx_rate: float = 0.0,
    gas_usd_per_swap: float = 0.21,
) -> config.Config:
    """A real config with the data dir redirected and execution pinned."""
    base = config.load()
    execution = replace(
        base.execution,
        failed_tx_rate=failed_tx_rate,
        gas_usd_per_swap=gas_usd_per_swap,
    )
    return replace(
        base,
        data_dir=tmp_path,
        starting_cash_usd=starting_cash_usd,
        execution=execution,
    )


def usd(amount: float) -> int:
    """Dollars to micro-USDC. Only ever called on literals in this file."""
    return round(amount * _MICRO)


def tokens(ui: float) -> int:
    """UI tokens to atomic units at BONK's 5 decimals."""
    return round(ui * _UNIT)


def make_quote(
    *,
    symbol: str = "BONK",
    side: Side = Side.BUY,
    in_amount_atomic: int,
    out_amount_atomic: int,
    min_out_amount_atomic: int | None = None,
    price_impact_pct: float = 0.4,
    now: float = NOW,
    ttl_seconds: float | None = 10.0,
    slot: int | None = 448_161_273,
    token: TokenMeta = BONK,
    fingerprint: str | None = None,
) -> Quote:
    """A router quote with a *correct* fingerprint unless one is forced.

    ``min_out`` defaults to ``out``, i.e. zero assumed slippage, so that the
    hand-computed expectations stay hand-computable. The tests that care about
    the conservative fill model set it explicitly.
    """
    input_token, output_token = (USDC, token) if side is Side.BUY else (token, USDC)
    if min_out_amount_atomic is None:
        min_out_amount_atomic = out_amount_atomic
    if fingerprint is None:
        fingerprint = quote_fingerprint(
            side=str(side),
            input_mint=input_token.mint,
            output_mint=output_token.mint,
            in_amount_atomic=in_amount_atomic,
            out_amount_atomic=out_amount_atomic,
            slot=slot,
        )
    return Quote(
        symbol=symbol,
        side=side,
        input_token=input_token,
        output_token=output_token,
        in_amount_atomic=in_amount_atomic,
        out_amount_atomic=out_amount_atomic,
        min_out_amount_atomic=min_out_amount_atomic,
        price_impact_pct=price_impact_pct,
        route_labels=("AlphaQ", "Scorch"),
        fingerprint=fingerprint,
        requested_at=now,
        received_at=now,
        context_slot=slot,
        expires_at=None if ttl_seconds is None else now + ttl_seconds,
    )


def make_intent(
    quote: Quote,
    *,
    in_amount_atomic: int | None = None,
    max_in_amount_atomic: int | None = None,
    symbol: str | None = None,
    side: Side | None = None,
    intent_id: str | None = None,
    source: str = "strategy",
) -> OrderIntent:
    """The intent that authorises ``quote``, unless a test breaks it on purpose."""
    size = quote.in_amount_atomic if in_amount_atomic is None else in_amount_atomic
    return OrderIntent(
        intent_id=intent_id or new_intent_id(),
        decision_id=new_decision_id(),
        action_id=None,
        run_id="run-test",
        ts=quote.received_at,
        symbol=symbol or quote.symbol,
        side=quote.side if side is None else side,
        in_amount_atomic=size,
        max_in_amount_atomic=size if max_in_amount_atomic is None else max_in_amount_atomic,
        source=source,  # type: ignore[arg-type]
    )


def never_fails() -> random.Random:
    """An rng is still injected so the draw sequence is explicit in tests."""
    return random.Random(1234)


def buy(
    broker: LocalPaperBroker,
    *,
    usd_in: float,
    price_usd: float,
    now: float = NOW,
    **kwargs,
):
    """BUY ``usd_in`` dollars of BONK at exactly ``price_usd``."""
    in_atomic = usd(usd_in)
    out_atomic = tokens(usd_in / price_usd)
    q = make_quote(
        side=Side.BUY,
        in_amount_atomic=in_atomic,
        out_amount_atomic=out_atomic,
        now=now,
        **kwargs,
    )
    return broker.place_order(make_intent(q), q, now=now)


def sell(
    broker: LocalPaperBroker,
    *,
    token_atomic: int,
    price_usd: float,
    now: float = NOW,
    **kwargs,
):
    """SELL an exact atomic quantity at exactly ``price_usd``."""
    out_atomic = usd(token_atomic / _UNIT * price_usd)
    q = make_quote(
        side=Side.SELL,
        in_amount_atomic=token_atomic,
        out_amount_atomic=out_atomic,
        now=now,
        **kwargs,
    )
    return broker.place_order(make_intent(q), q, now=now)


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Cold start
# ---------------------------------------------------------------------------


def test_first_run_initializes_cash_to_starting_cash(tmp_path: Path) -> None:
    broker = LocalPaperBroker(
        make_cfg(tmp_path, starting_cash_usd=2500.0), rng=never_fails()
    )
    assert broker.cash_usd == 2500.0
    assert broker.cash_micro_usd == 2_500_000_000
    assert broker.get_positions() == {}
    assert broker.realized_pnl_usd == 0.0
    assert broker.reconciliation.clean


def test_get_positions_returns_a_copy(tmp_path: Path) -> None:
    """Callers mutating the returned dict must not be able to edit the book."""
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    buy(broker, usd_in=400.0, price_usd=0.00002)
    view = broker.get_positions()
    del view["BONK"]
    assert "BONK" in broker.get_positions()


def test_mode_is_part_of_the_broker(tmp_path: Path) -> None:
    """Audit C12: the old ``--dry-run`` was a flag checked at one call site in
    loop.py, and the stop-loss path bypassed it. The capability now travels with
    the object, so there is no second call site to forget."""
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    assert broker.mode is ExecutionMode.PAPER
    assert broker.mode.may_mutate
    assert not ExecutionMode.READ_ONLY.may_mutate


def test_live_mode_cannot_be_constructed(tmp_path: Path) -> None:
    with pytest.raises(LiveModeUnsupported):
        LocalPaperBroker(make_cfg(tmp_path), mode=ExecutionMode.LIVE)
    with pytest.raises(LiveModeUnsupported):
        assert_live_supported(ExecutionMode.LIVE)
    assert assert_live_supported(ExecutionMode.PAPER) is None


# ---------------------------------------------------------------------------
# The worked example
# ---------------------------------------------------------------------------


def test_round_trip_hand_computed(tmp_path: Path) -> None:
    """1000 cash. BUY $400 of BONK at 0.00002 (20,000,000 tokens), gas 0.21.

    cash = 1000 - 400 - 0.21 = 599.79, basis = 400.21.
    SELL all 20,000,000 at 0.000025 = $500 proceeds, gas 0.21.
    cash = 599.79 + 500 - 0.21 = 1099.58
    realized = 500 - 400.21 - 0.21 = 99.58
    """
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())

    fill = buy(broker, usd_in=400.0, price_usd=0.00002)
    assert fill.state is OrderState.LANDED
    assert broker.cash_micro_usd == usd(599.79)
    position = broker.get_positions()["BONK"]
    assert position.quantity_atomic == tokens(20_000_000)
    assert position.cost_basis_usd == pytest.approx(400.21)
    assert position.avg_entry_price_usd == pytest.approx(0.00002)

    exit_fill = sell(broker, token_atomic=tokens(20_000_000), price_usd=0.000025)
    assert broker.cash_micro_usd == usd(1099.58)
    assert broker.realized_pnl_usd == pytest.approx(99.58)
    assert exit_fill.realized_pnl_usd == pytest.approx(99.58)
    assert broker.get_positions() == {}


def test_cost_basis_includes_gas(tmp_path: Path) -> None:
    """Audit §11: a position's break-even is its true break-even, so a fill is
    marked at a small loss the instant it lands. The *price* must not absorb the
    gas, though — that would corrupt the entry price the strategy reasons about."""
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    buy(broker, usd_in=400.0, price_usd=0.00002)
    position = broker.get_positions()["BONK"]
    assert position.cost_basis_usd == pytest.approx(400.21)
    assert position.avg_entry_price_usd == pytest.approx(0.00002)
    assert position.cost_basis_usd > position.quantity * position.avg_entry_price_usd


def test_a_routed_quote_pays_no_explicit_pool_fee(tmp_path: Path) -> None:
    """The 45x double-count. A live BONK round trip cost 4.4 bp (buy at
    2.9636e-6, sell at 2.9623e-6) against the 200 bp that billing 0.25% on each
    of two hops implied: ``outAmount`` is already net of every hop's AMM fee.
    Only gas is additive."""
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    fill = buy(broker, usd_in=400.0, price_usd=0.00002)
    assert fill.pool_fee_usd == 0.0
    assert broker.fees_paid_usd == 0.0
    assert broker.gas_paid_usd == pytest.approx(0.21)
    assert pool_fee_micro(fill and make_quote(in_amount_atomic=1, out_amount_atomic=1)) == 0


def test_scaling_in_averages_the_entry_price(tmp_path: Path) -> None:
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    buy(broker, usd_in=100.0, price_usd=0.00002)  # 5,000,000 tokens
    buy(broker, usd_in=100.0, price_usd=0.00004)  # 2,500,000 tokens
    position = broker.get_positions()["BONK"]
    assert position.quantity_atomic == tokens(7_500_000)
    # 200 notional / 7.5M tokens, gas excluded from the price.
    assert position.avg_entry_price_usd == pytest.approx(200.0 / 7_500_000)
    assert position.cost_basis_usd == pytest.approx(200.42)


def test_scaling_in_keeps_the_original_open_time(tmp_path: Path) -> None:
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    buy(broker, usd_in=100.0, price_usd=0.00002, now=NOW)
    buy(broker, usd_in=100.0, price_usd=0.00002, now=NOW + 5.0)
    assert broker.get_positions()["BONK"].opened_at == NOW


# ---------------------------------------------------------------------------
# Partial exits
# ---------------------------------------------------------------------------


def test_partial_sell_removes_a_proportional_share_of_basis(tmp_path: Path) -> None:
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    buy(broker, usd_in=400.0, price_usd=0.00002)  # basis 400.21, 20M tokens
    sell(broker, token_atomic=tokens(5_000_000), price_usd=0.000025)  # a quarter

    position = broker.get_positions()["BONK"]
    assert position.quantity_atomic == tokens(15_000_000)
    assert position.cost_basis_usd == pytest.approx(400.21 * 0.75)
    # proceeds 125 - basis share 100.0525 - gas 0.21
    assert broker.realized_pnl_usd == pytest.approx(24.7375)


def test_two_partial_sells_equal_one_full_sell_in_tokens(tmp_path: Path) -> None:
    """The difference is exactly one extra gas charge and nothing else; in
    atomic units the two paths must land on the identical (empty) book."""
    split = LocalPaperBroker(make_cfg(tmp_path / "a"), rng=never_fails())
    whole = LocalPaperBroker(make_cfg(tmp_path / "b"), rng=never_fails())
    for broker in (split, whole):
        buy(broker, usd_in=400.0, price_usd=0.00002)

    sell(split, token_atomic=tokens(10_000_000), price_usd=0.000025)
    sell(split, token_atomic=tokens(10_000_000), price_usd=0.000025)
    sell(whole, token_atomic=tokens(20_000_000), price_usd=0.000025)

    assert split.get_positions() == whole.get_positions() == {}
    gas = usd(0.21)
    assert split.cash_micro_usd == whole.cash_micro_usd - gas
    assert split.realized_pnl_usd == pytest.approx(whole.realized_pnl_usd - 0.21)


def test_selling_the_whole_quantity_deletes_the_position_exactly(tmp_path: Path) -> None:
    """Atomic units make dust a thing that either exists or does not; the old
    float book needed a relative epsilon to stop 1e-16 of a token lingering."""
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    buy(broker, usd_in=400.0, price_usd=0.00002)
    held = broker.get_positions()["BONK"].quantity_atomic
    sell(broker, token_atomic=held - 4_000_000_000, price_usd=0.000025)
    assert broker.get_positions()["BONK"].quantity_atomic == 4_000_000_000
    sell(broker, token_atomic=4_000_000_000, price_usd=0.000025)
    assert "BONK" not in broker.get_positions()


def test_dust_worth_less_than_a_micro_dollar_cannot_be_sold(tmp_path: Path) -> None:
    """One atomic unit of a 5-decimal token at 2.5e-5 is 2.5e-10 dollars, which
    rounds to zero micro-USDC. Filling it would burn 21 cents of gas to receive
    nothing, and the fill would divide by a zero expected output."""
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    buy(broker, usd_in=400.0, price_usd=0.00002)
    with pytest.raises(QuoteRejected, match="dust"):
        sell(broker, token_atomic=1, price_usd=0.000025)


def test_overselling_is_refused_rather_than_clamped(tmp_path: Path) -> None:
    """The old broker clamped the size and filled it against the larger quote —
    C3 in miniature, since the executed size was never the priced size."""
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    buy(broker, usd_in=400.0, price_usd=0.00002)
    before = broker.cash_micro_usd
    with pytest.raises(InsufficientPosition):
        sell(broker, token_atomic=tokens(30_000_000), price_usd=0.000025)
    assert broker.cash_micro_usd == before
    assert broker.get_positions()["BONK"].quantity_atomic == tokens(20_000_000)
    assert rows(broker.ledger_path) and len(rows(broker.ledger_path)) == 1


def test_selling_without_a_position_raises(tmp_path: Path) -> None:
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    with pytest.raises(NoPosition):
        sell(broker, token_atomic=tokens(1000), price_usd=0.000025)


@pytest.mark.parametrize("usd_in", [1000.0, 999.9, 1500.0])
def test_a_buy_larger_than_cash_is_refused(tmp_path: Path, usd_in: float) -> None:
    """1000 cash cannot fund a 1000 buy: gas comes out of the same pocket."""
    broker = LocalPaperBroker(
        make_cfg(tmp_path, starting_cash_usd=1000.0), rng=never_fails()
    )
    with pytest.raises(InsufficientCash):
        buy(broker, usd_in=usd_in, price_usd=0.00002)
    assert broker.cash_micro_usd == usd(1000.0)
    assert rows(broker.ledger_path) == []
    assert rows(broker.intents_path) == []


def test_a_buy_that_exactly_fits_including_gas_is_allowed(tmp_path: Path) -> None:
    broker = LocalPaperBroker(
        make_cfg(tmp_path, starting_cash_usd=1000.0), rng=never_fails()
    )
    buy(broker, usd_in=999.79, price_usd=0.00002)
    assert broker.cash_micro_usd == 0


# ---------------------------------------------------------------------------
# Failed transactions — audit C5c
# ---------------------------------------------------------------------------


def test_a_failed_transaction_burns_gas_and_files_a_row(tmp_path: Path) -> None:
    """The old code returned ``None`` on a failed attempt, so the gas was free
    and the attempt was invisible. A FAILED Fill with zero amounts and non-zero
    gas is the whole of C5c."""
    cfg = make_cfg(tmp_path, failed_tx_rate=1.0)
    broker = LocalPaperBroker(cfg, rng=never_fails())
    fill = buy(broker, usd_in=400.0, price_usd=0.00002)

    assert fill.state is OrderState.FAILED
    assert fill.failed
    assert fill.in_amount_atomic == 0
    assert fill.out_amount_atomic == 0
    assert fill.token_amount_atomic == 0
    assert fill.gas_usd == pytest.approx(0.21)
    assert fill.price_usd is None  # nothing traded, so no price was realised
    assert broker.cash_micro_usd == usd(1000.0) - usd(0.21)
    assert broker.get_positions() == {}
    assert broker.failed_gas_usd == pytest.approx(0.21)
    assert len(rows(broker.ledger_path)) == 1
    assert rows(broker.ledger_path)[0]["state"] == str(OrderState.FAILED)


def test_a_failed_sell_leaves_the_position_untouched(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, failed_tx_rate=0.0)
    broker = LocalPaperBroker(cfg, rng=never_fails())
    buy(broker, usd_in=400.0, price_usd=0.00002)

    broker.cfg = replace(cfg, execution=replace(cfg.execution, failed_tx_rate=1.0))
    fill = sell(broker, token_atomic=tokens(20_000_000), price_usd=0.000025)

    assert fill.failed
    assert broker.get_positions()["BONK"].quantity_atomic == tokens(20_000_000)
    assert broker.realized_pnl_usd == 0.0


def test_the_failure_draw_is_deterministic_for_a_seed(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, failed_tx_rate=0.5)
    outcomes = []
    for run in range(2):
        broker = LocalPaperBroker(
            replace(cfg, data_dir=tmp_path / f"r{run}"), rng=random.Random(7)
        )
        outcomes.append(
            [buy(broker, usd_in=10.0, price_usd=0.00002).failed for _ in range(8)]
        )
    assert outcomes[0] == outcomes[1]
    assert any(outcomes[0]) and not all(outcomes[0])  # the seed exercises both branches


# ---------------------------------------------------------------------------
# The fill model — audit C5a/C5b
# ---------------------------------------------------------------------------


def test_the_default_fill_model_is_the_slippage_worst_output(tmp_path: Path) -> None:
    """MIN_OUT by default: a simulator whose whole complaint about itself is
    that it used to be optimistic should err the other way."""
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    assert broker.fill_model is FillModel.MIN_OUT

    q = make_quote(
        side=Side.BUY,
        in_amount_atomic=usd(400.0),
        out_amount_atomic=tokens(20_000_000),
        min_out_amount_atomic=tokens(19_900_000),  # 50 bps worse
    )
    fill = broker.place_order(make_intent(q), q, now=NOW)
    assert fill.token_amount_atomic == tokens(19_900_000)
    assert broker.get_positions()["BONK"].quantity_atomic == tokens(19_900_000)


def test_slippage_versus_the_quote_is_recorded_and_signed(tmp_path: Path) -> None:
    """C5b. Negative is worse than quoted, which is the only direction the
    conservative model can produce; an unsigned magnitude would hide that."""
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    q = make_quote(
        side=Side.BUY,
        in_amount_atomic=usd(400.0),
        out_amount_atomic=tokens(20_000_000),
        min_out_amount_atomic=tokens(19_900_000),
    )
    fill = broker.place_order(make_intent(q), q, now=NOW)
    assert fill.slippage_bps_vs_quote == pytest.approx(-50.0)
    assert rows(broker.ledger_path)[0]["slippage_bps_vs_quote"] == pytest.approx(-50.0)


def test_the_optimistic_model_is_available_for_measurement(tmp_path: Path) -> None:
    """EXPECTED_OUT exists so the cost of the conservatism can be measured by
    running both and diffing the P&L, rather than argued about. Never default."""
    broker = LocalPaperBroker(
        make_cfg(tmp_path), rng=never_fails(), fill_model=FillModel.EXPECTED_OUT
    )
    q = make_quote(
        side=Side.BUY,
        in_amount_atomic=usd(400.0),
        out_amount_atomic=tokens(20_000_000),
        min_out_amount_atomic=tokens(19_900_000),
    )
    fill = broker.place_order(make_intent(q), q, now=NOW)
    assert fill.token_amount_atomic == tokens(20_000_000)
    assert fill.slippage_bps_vs_quote == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Binding — audit C3
# ---------------------------------------------------------------------------


def test_a_mismatched_fingerprint_is_refused(tmp_path: Path) -> None:
    """The broker recomputes rather than trusting. A quote whose contents were
    edited after it was obtained no longer hashes to what it carries."""
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    q = make_quote(
        side=Side.BUY,
        in_amount_atomic=usd(400.0),
        out_amount_atomic=tokens(20_000_000),
        fingerprint="0" * 32,
    )
    with pytest.raises(QuoteBindingError, match="fingerprint"):
        broker.place_order(make_intent(q), q, now=NOW)
    assert rows(broker.ledger_path) == []
    assert rows(broker.intents_path) == []


def test_a_quote_for_a_different_size_cannot_fill_this_intent(tmp_path: Path) -> None:
    """C3 as the audit found it: risk shrank the order after it was quoted and
    the broker filled the reduced notional at the original-size price."""
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    q = make_quote(
        side=Side.BUY, in_amount_atomic=usd(400.0), out_amount_atomic=tokens(20_000_000)
    )
    shrunk = make_intent(q, in_amount_atomic=usd(200.0), max_in_amount_atomic=usd(400.0))
    with pytest.raises(QuoteBindingError, match="authorises"):
        broker.place_order(shrunk, q, now=NOW)
    assert broker.cash_micro_usd == usd(1000.0)


def test_an_intent_larger_than_its_own_bound_is_refused(tmp_path: Path) -> None:
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    q = make_quote(
        side=Side.BUY, in_amount_atomic=usd(400.0), out_amount_atomic=tokens(20_000_000)
    )
    over = make_intent(q, max_in_amount_atomic=usd(100.0))
    with pytest.raises(QuoteBindingError, match="bound"):
        broker.place_order(over, q, now=NOW)


@pytest.mark.parametrize("field", ["symbol", "side"])
def test_an_intent_for_a_different_order_is_refused(tmp_path: Path, field: str) -> None:
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    q = make_quote(
        side=Side.BUY, in_amount_atomic=usd(400.0), out_amount_atomic=tokens(20_000_000)
    )
    kwargs = {"symbol": "WIF"} if field == "symbol" else {"side": Side.SELL}
    with pytest.raises(QuoteBindingError):
        broker.place_order(make_intent(q, **kwargs), q, now=NOW)


# ---------------------------------------------------------------------------
# Admission — audit C4, C5d, C5e, §15
# ---------------------------------------------------------------------------


def test_a_valuation_estimate_cannot_reach_place_order(tmp_path: Path) -> None:
    """Audit C4, both ways round.

    Type level: ``place_order``'s second parameter is annotated ``Quote``, and
    ``ValuationEstimate`` is not a ``Quote`` and shares no base with one, so a
    type checker rejects the call outright — there is no ``degraded`` flag to
    forget to check.

    Runtime: passing one anyway fails before anything is journaled, applied or
    written. The old code's fallback *was* a FillQuote, so the broker could not
    tell the difference and filled a DexScreener mid plus a fixed spread.
    """
    annotation = (
        inspect.signature(LocalPaperBroker.place_order).parameters["quote"].annotation
    )
    assert annotation is Quote or annotation == "Quote"
    assert not issubclass(ValuationEstimate, Quote)
    assert set(Quote.__mro__) & set(ValuationEstimate.__mro__) == {object}

    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    estimate = ValuationEstimate(
        symbol="BONK",
        mid_price_usd=0.00002,
        haircut_pct=1.0,
        reason="jupiter unreachable",
        at=NOW,
        source="dexscreener:mid",
    )
    real = make_quote(
        side=Side.BUY, in_amount_atomic=usd(400.0), out_amount_atomic=tokens(20_000_000)
    )
    with pytest.raises((AttributeError, TypeError, BrokerError)):
        broker.place_order(make_intent(real), estimate, now=NOW)  # type: ignore[arg-type]

    assert broker.cash_micro_usd == usd(1000.0)
    assert rows(broker.ledger_path) == []
    assert rows(broker.intents_path) == []
    assert not broker.cfg.state_path.exists()


def test_an_expired_quote_is_refused(tmp_path: Path) -> None:
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    q = make_quote(
        side=Side.BUY,
        in_amount_atomic=usd(400.0),
        out_amount_atomic=tokens(20_000_000),
        ttl_seconds=10.0,
    )
    with pytest.raises(QuoteRejected, match="expired"):
        broker.place_order(make_intent(q), q, now=NOW + 10.0)


def test_a_quote_older_than_the_ceiling_is_refused(tmp_path: Path) -> None:
    """§11: "Stale price/quote -> reject stale quote; re-quote exact amount ->
    no new trade". A quote with no expiry of its own still has an age."""
    broker = LocalPaperBroker(
        make_cfg(tmp_path), rng=never_fails(), max_quote_age_seconds=15.0
    )
    q = make_quote(
        side=Side.BUY,
        in_amount_atomic=usd(400.0),
        out_amount_atomic=tokens(20_000_000),
        ttl_seconds=None,
    )
    broker_fill = broker.place_order(make_intent(q), q, now=NOW + 14.0)
    assert broker_fill.state is OrderState.LANDED
    with pytest.raises(QuoteRejected, match="old"):
        broker.place_order(make_intent(q), q, now=NOW + 16.0)


def test_a_quote_from_the_future_halts_rather_than_trades(tmp_path: Path) -> None:
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    q = make_quote(
        side=Side.BUY,
        in_amount_atomic=usd(400.0),
        out_amount_atomic=tokens(20_000_000),
        ttl_seconds=None,
    )
    with pytest.raises(QuoteRejected, match="future"):
        broker.place_order(make_intent(q), q, now=NOW - 5.0)


def test_excessive_price_impact_is_refused(tmp_path: Path) -> None:
    """C5e. The default ceiling is the risk layer's, so the broker can never be
    more permissive than the thing that is supposed to be gating it."""
    cfg = make_cfg(tmp_path)
    broker = LocalPaperBroker(cfg, rng=never_fails())
    assert cfg.risk.max_price_impact_pct == 3.0
    q = make_quote(
        side=Side.BUY,
        in_amount_atomic=usd(400.0),
        out_amount_atomic=tokens(20_000_000),
        price_impact_pct=3.5,
    )
    with pytest.raises(QuoteRejected, match="impact"):
        broker.place_order(make_intent(q), q, now=NOW)


def test_unverified_decimals_are_refused(tmp_path: Path) -> None:
    """§15. An inferred exponent is a silent factor-of-1000 error in every
    quantity downstream, and it arrives looking like a plausible number."""
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    guessed = TokenMeta(
        mint=MINT, decimals=DECIMALS, source="derived:mid-price", verified=False
    )
    q = make_quote(
        side=Side.BUY,
        in_amount_atomic=usd(400.0),
        out_amount_atomic=tokens(20_000_000),
        token=guessed,
    )
    with pytest.raises(QuoteRejected, match="unverified"):
        broker.place_order(make_intent(q), q, now=NOW)


def test_a_non_usdc_cash_leg_is_refused(tmp_path: Path) -> None:
    """Otherwise ``notional_usd`` is not dollars and the whole book is mislabelled."""
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    usdt = TokenMeta(
        mint="Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        decimals=6,
        source="t",
        verified=True,
    )
    q = make_quote(
        side=Side.BUY,
        in_amount_atomic=usd(400.0),
        out_amount_atomic=tokens(20_000_000),
        token=BONK,
    )
    q = replace(q, input_token=usdt)
    q = replace(
        q,
        fingerprint=quote_fingerprint(
            side="BUY",
            input_mint=usdt.mint,
            output_mint=MINT,
            in_amount_atomic=q.in_amount_atomic,
            out_amount_atomic=q.out_amount_atomic,
            slot=q.context_slot,
        ),
    )
    with pytest.raises(QuoteRejected, match="USDC"):
        broker.place_order(make_intent(q), q, now=NOW)


def test_a_zero_input_quote_cannot_be_constructed() -> None:
    with pytest.raises(ValidationError):
        make_quote(in_amount_atomic=0, out_amount_atomic=1)
    with pytest.raises(ValidationError):
        make_quote(in_amount_atomic=-1, out_amount_atomic=1)
    with pytest.raises(ValidationError):
        make_quote(in_amount_atomic=1, out_amount_atomic=10, min_out_amount_atomic=11)


# ---------------------------------------------------------------------------
# READ_ONLY — audit C12
# ---------------------------------------------------------------------------


def test_read_only_leaves_state_and_ledger_byte_identical(tmp_path: Path) -> None:
    """The structural half of C12. Not "the guard is checked at the call site"
    — the mode travels with the broker and every writer asserts it, so there is
    no second path (the old stop-loss exit) that can bypass it."""
    cfg = make_cfg(tmp_path)
    warm = LocalPaperBroker(cfg, rng=never_fails())
    buy(warm, usd_in=400.0, price_usd=0.00002)

    state_before = cfg.state_path.read_bytes()
    ledger_before = warm.ledger_path.read_bytes()
    intents_before = warm.intents_path.read_bytes()
    listing_before = sorted(p.name for p in tmp_path.iterdir())

    ro = LocalPaperBroker(cfg, mode=ExecutionMode.READ_ONLY, rng=never_fails())
    assert ro.cash_micro_usd == warm.cash_micro_usd
    fill = buy(ro, usd_in=100.0, price_usd=0.00002)

    assert fill.state is OrderState.LANDED
    assert fill.note and "read_only" in fill.note
    assert cfg.state_path.read_bytes() == state_before
    assert ro.ledger_path.read_bytes() == ledger_before
    assert ro.intents_path.read_bytes() == intents_before
    assert sorted(p.name for p in tmp_path.iterdir()) == listing_before
    # Not even in memory.
    assert ro.cash_micro_usd == warm.cash_micro_usd
    assert ro.get_positions()["BONK"].quantity_atomic == tokens(20_000_000)


def test_read_only_still_refuses_what_paper_would_refuse(tmp_path: Path) -> None:
    """A read-only run is a rehearsal; it has to surface binding bugs, not
    swallow them."""
    ro = LocalPaperBroker(
        make_cfg(tmp_path), mode=ExecutionMode.READ_ONLY, rng=never_fails()
    )
    q = make_quote(
        side=Side.BUY,
        in_amount_atomic=usd(400.0),
        out_amount_atomic=tokens(20_000_000),
        fingerprint="0" * 32,
    )
    with pytest.raises(QuoteBindingError):
        ro.place_order(make_intent(q), q, now=NOW)
    with pytest.raises(InsufficientCash):
        buy(ro, usd_in=5000.0, price_usd=0.00002)


def test_read_only_writers_refuse_individually(tmp_path: Path) -> None:
    ro = LocalPaperBroker(
        make_cfg(tmp_path), mode=ExecutionMode.READ_ONLY, rng=never_fails()
    )
    with pytest.raises(ReadOnlyViolation):
        ro.save()


def test_read_only_is_reproducible(tmp_path: Path) -> None:
    """The failure coin flip is deliberately not drawn: a read-only run answers
    "what would this order look like", and a random failure would make the
    answer irreproducible without making it more informative."""
    cfg = make_cfg(tmp_path, failed_tx_rate=1.0)
    ro = LocalPaperBroker(cfg, mode=ExecutionMode.READ_ONLY, rng=random.Random(0))
    for _ in range(5):
        assert not buy(ro, usd_in=100.0, price_usd=0.00002).failed


# ---------------------------------------------------------------------------
# Idempotency and recovery — audit C11
# ---------------------------------------------------------------------------


def test_resubmitting_an_intent_id_is_a_no_op(tmp_path: Path) -> None:
    """The retry-after-timeout case. Two calls, one fill, one ledger row."""
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    q = make_quote(
        side=Side.BUY, in_amount_atomic=usd(400.0), out_amount_atomic=tokens(20_000_000)
    )
    intent = make_intent(q)

    first = broker.place_order(intent, q, now=NOW)
    cash_after_first = broker.cash_micro_usd
    second = broker.place_order(intent, q, now=NOW)

    assert second is first
    assert second.fill_id == first.fill_id
    assert broker.cash_micro_usd == cash_after_first
    assert len(rows(broker.ledger_path)) == 1
    assert len(rows(broker.intents_path)) == 1
    assert broker.get_positions()["BONK"].quantity_atomic == tokens(20_000_000)


def test_idempotency_survives_a_restart(tmp_path: Path) -> None:
    """The key is rebuilt from the ledger, not held only in memory — otherwise a
    crash between the fill and the retry double-executes."""
    cfg = make_cfg(tmp_path)
    broker = LocalPaperBroker(cfg, rng=never_fails())
    q = make_quote(
        side=Side.BUY, in_amount_atomic=usd(400.0), out_amount_atomic=tokens(20_000_000)
    )
    intent = make_intent(q)
    first = broker.place_order(intent, q, now=NOW)

    restarted = LocalPaperBroker(cfg, rng=never_fails())
    again = restarted.place_order(intent, q, now=NOW)
    assert again.fill_id == first.fill_id
    assert len(rows(restarted.ledger_path)) == 1
    assert restarted.cash_micro_usd == broker.cash_micro_usd


def test_a_replayed_intent_is_returned_even_when_its_quote_went_stale(
    tmp_path: Path,
) -> None:
    """Idempotency is checked before validation on purpose: a retry is replaying
    an order that already happened, and refusing it would turn a completed trade
    into an exception."""
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    q = make_quote(
        side=Side.BUY, in_amount_atomic=usd(400.0), out_amount_atomic=tokens(20_000_000)
    )
    intent = make_intent(q)
    first = broker.place_order(intent, q, now=NOW)
    assert broker.place_order(intent, q, now=NOW + 3600.0).fill_id == first.fill_id


def test_a_crash_between_the_ledger_and_the_checkpoint_is_replayed(tmp_path: Path) -> None:
    """The crash window the whole ordering exists for.

    The ledger is appended and fsynced (step 5) before the state file is
    replaced (step 7), so a crash in between leaves a durable fill the
    checkpoint has not folded in. Startup replays everything after
    ``last_fill_id`` and lands on exactly the balances the uncrashed run had —
    not approximately, exactly, because ``_apply`` is a pure function of the
    Fill and the current book.
    """
    cfg = make_cfg(tmp_path)
    broker = LocalPaperBroker(cfg, rng=never_fails())
    buy(broker, usd_in=400.0, price_usd=0.00002)

    stale_state = tmp_path / "state.after-first.json"
    shutil.copy2(cfg.state_path, stale_state)  # the checkpoint as of fill #1

    buy(broker, usd_in=100.0, price_usd=0.00004)
    sell(broker, token_atomic=tokens(1_000_000), price_usd=0.000025)
    expected_cash = broker.cash_micro_usd
    expected_positions = broker.get_positions()
    expected_realized = broker.realized_pnl_usd

    # Crash: the ledger has all three rows, the checkpoint has one.
    shutil.copy2(stale_state, cfg.state_path)

    recovered = LocalPaperBroker(cfg, rng=never_fails())
    report = recovered.reconciliation
    assert len(report.replayed_fill_ids) == 2
    assert report.ledger_fills == 3
    assert not report.clean  # startup must be able to see that it recovered

    assert recovered.cash_micro_usd == expected_cash
    assert recovered.get_positions() == expected_positions
    assert recovered.realized_pnl_usd == pytest.approx(expected_realized)

    # And a clean restart after recovery reports nothing left to do.
    recovered.save()
    assert LocalPaperBroker(cfg, rng=never_fails()).reconciliation.clean


def test_an_intent_with_no_fill_is_reported_open(tmp_path: Path) -> None:
    """The state the old code could not name: a crash between the intent write
    and the ledger append. Here it means no money moved; on a live venue it
    would mean an order of unknown status, and §11's answer is the same either
    way — reconcile before any new order."""
    cfg = make_cfg(tmp_path)
    broker = LocalPaperBroker(cfg, rng=never_fails())
    orphan = make_intent(
        make_quote(
            side=Side.BUY, in_amount_atomic=usd(50.0), out_amount_atomic=tokens(2_500_000)
        ),
        intent_id=new_intent_id(),
    )
    broker._journal_intent(orphan)

    report = LocalPaperBroker(cfg, rng=never_fails()).reconcile()
    assert [i.intent_id for i in report.open_intents] == [orphan.intent_id]
    assert not report.clean


def test_a_checkpoint_ahead_of_the_ledger_is_fatal(tmp_path: Path) -> None:
    """Impossible by construction — the ledger is always written first — so if
    it happens a file was truncated or replaced and nothing here may guess."""
    cfg = make_cfg(tmp_path)
    broker = LocalPaperBroker(cfg, rng=never_fails())
    buy(broker, usd_in=400.0, price_usd=0.00002)
    broker.ledger_path.write_text("")
    with pytest.raises(LedgerCorrupt, match="ahead"):
        LocalPaperBroker(cfg, rng=never_fails())


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_state_round_trips_exactly(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    broker = LocalPaperBroker(cfg, rng=never_fails())
    buy(broker, usd_in=400.0, price_usd=0.00002)
    sell(broker, token_atomic=tokens(5_000_000), price_usd=0.000025)

    reloaded = LocalPaperBroker(cfg, rng=never_fails())
    assert reloaded.cash_micro_usd == broker.cash_micro_usd
    assert reloaded.get_positions() == broker.get_positions()
    assert reloaded.realized_pnl_usd == broker.realized_pnl_usd
    assert reloaded.gas_paid_usd == broker.gas_paid_usd

    # Byte-identical apart from run_id, which is per-process by design: the
    # checkpoint records which run last wrote it.
    before = json.loads(cfg.state_path.read_text())
    reloaded.save()
    after = json.loads(cfg.state_path.read_text())
    assert {k: v for k, v in after.items() if k != "run_id"} == {
        k: v for k, v in before.items() if k != "run_id"
    }


def test_state_carries_a_schema_version_and_a_run_id(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    broker = LocalPaperBroker(cfg, run_id="run-abc", rng=never_fails())
    buy(broker, usd_in=400.0, price_usd=0.00002)
    payload = json.loads(cfg.state_path.read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["run_id"] == "run-abc"
    assert payload["last_fill_id"] == rows(broker.ledger_path)[-1]["fill_id"]
    assert isinstance(payload["cash_micro_usd"], int)


def test_every_ledger_row_carries_the_run_that_made_it(tmp_path: Path) -> None:
    """A duplicate-process incident must be visible as two run IDs interleaved
    in one ledger rather than as inexplicable state."""
    cfg = make_cfg(tmp_path)
    for run in ("run-a", "run-b"):
        broker = LocalPaperBroker(cfg, run_id=run, rng=never_fails())
        buy(broker, usd_in=10.0, price_usd=0.00002)
    assert [r["run_id"] for r in rows(cfg.trades_path)] == ["run-a", "run-b"]


def test_an_unknown_schema_version_is_refused_not_repaired(tmp_path: Path) -> None:
    """Resetting to starting cash would quietly erase the entire P&L history."""
    cfg = make_cfg(tmp_path)
    broker = LocalPaperBroker(cfg, rng=never_fails())
    buy(broker, usd_in=400.0, price_usd=0.00002)
    payload = json.loads(cfg.state_path.read_text())
    payload["schema_version"] = SCHEMA_VERSION + 1
    cfg.state_path.write_text(json.dumps(payload))
    with pytest.raises(LedgerCorrupt, match="schema_version"):
        LocalPaperBroker(cfg, rng=never_fails())


def test_a_corrupt_state_file_is_refused(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    cfg.state_path.write_text("{not json")
    with pytest.raises(LedgerCorrupt):
        LocalPaperBroker(cfg, rng=never_fails())


def test_a_malformed_ledger_row_is_fatal_not_skipped(tmp_path: Path) -> None:
    """Skipping it would silently drop a trade, which is the other worst thing a
    ledger can do."""
    cfg = make_cfg(tmp_path)
    broker = LocalPaperBroker(cfg, rng=never_fails())
    buy(broker, usd_in=400.0, price_usd=0.00002)
    with broker.ledger_path.open("a", encoding="utf-8") as handle:
        handle.write("{truncated\n")
    with pytest.raises(LedgerCorrupt):
        LocalPaperBroker(cfg, rng=never_fails())


def test_saving_leaves_no_temp_debris(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    broker = LocalPaperBroker(cfg, rng=never_fails())
    buy(broker, usd_in=400.0, price_usd=0.00002)
    broker.save()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "intents.jsonl",
        "state.json",
        "trades.jsonl",
    ]


def test_one_ledger_row_per_attempt_including_failures(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, failed_tx_rate=0.5)
    broker = LocalPaperBroker(cfg, rng=random.Random(7))
    attempts = 0
    for _ in range(10):
        buy(broker, usd_in=10.0, price_usd=0.00002)
        attempts += 1
    ledger = rows(cfg.trades_path)
    assert len(ledger) == attempts
    assert {r["state"] for r in ledger} == {str(OrderState.LANDED), str(OrderState.FAILED)}
    assert len(rows(broker.intents_path)) == attempts


def test_the_ledger_row_carries_the_mint_for_a_brand_new_position(tmp_path: Path) -> None:
    """The append happens before the in-memory fold, so the mint cannot be read
    off the position — it has to be threaded through from the quote, or recovery
    rebuilds a position with an empty mint."""
    broker = LocalPaperBroker(make_cfg(tmp_path), rng=never_fails())
    buy(broker, usd_in=400.0, price_usd=0.00002)
    assert rows(broker.ledger_path)[0]["mint"] == MINT
    assert rows(broker.ledger_path)[0]["token_decimals"] == DECIMALS


# ---------------------------------------------------------------------------
# Conservation — the Phase 1 exit gate
# ---------------------------------------------------------------------------


def _drive(broker: LocalPaperBroker, seed: int, steps: int = 40) -> None:
    """A random but legal sequence of buys and partial sells.

    Sizes and prices are drawn to produce awkward, non-round atomic amounts —
    proportional basis splits with a remainder are precisely where a float book
    drifts.
    """
    rng = random.Random(seed)
    now = NOW
    for _ in range(steps):
        now += 1.0
        price = rng.uniform(1e-6, 5e-5)
        held = broker.get_positions().get("BONK")
        if held is not None and rng.random() < 0.45:
            quantity = rng.randint(1, held.quantity_atomic)
            out_atomic = max(1, round(quantity / _UNIT * price * _MICRO))
            q = make_quote(
                side=Side.SELL,
                in_amount_atomic=quantity,
                out_amount_atomic=out_atomic,
                now=now,
            )
        else:
            spend = round(
                rng.uniform(1.0, min(50.0, max(1.5, broker.cash_usd - 1.0))) * _MICRO
            )
            if spend + usd(0.21) > broker.cash_micro_usd:
                continue
            out_atomic = max(1, round(spend / _MICRO / price * _UNIT))
            q = make_quote(
                side=Side.BUY, in_amount_atomic=spend, out_amount_atomic=out_atomic, now=now
            )
        broker.place_order(make_intent(q), q, now=now)


def _assert_reconciles(broker: LocalPaperBroker) -> None:
    """Recompute both ledgers from the trade journal, in integers, and demand
    exact equality with the book. Every number here is an ``int``; there is no
    tolerance anywhere, because a tolerance is how accounting drift hides."""
    ledger = rows(broker.ledger_path)

    cash = usd(broker.starting_cash_usd)
    token_atomic = 0
    gas_total = 0
    for row in ledger:
        gas = round(row["gas_usd"] * _MICRO)
        fee = round(row["pool_fee_usd"] * _MICRO)
        gas_total += gas
        if row["state"] == str(OrderState.FAILED):
            cash -= gas
            assert row["in_amount_atomic"] == row["out_amount_atomic"] == 0
            assert gas > 0  # C5c: a failure is never free
            continue
        if row["side"] == "BUY":
            cash -= row["in_amount_atomic"] + gas + fee
            token_atomic += row["token_amount_atomic"]
        else:
            cash += row["out_amount_atomic"] - gas - fee
            token_atomic -= row["token_amount_atomic"]

    assert cash == broker.cash_micro_usd
    held = broker.get_positions().get("BONK")
    assert token_atomic == (held.quantity_atomic if held else 0)
    assert token_atomic >= 0
    assert gas_total == round(broker.gas_paid_usd * _MICRO)

    # The identity _apply exists to maintain: nothing is created or destroyed
    # except realised P&L and burnt gas.
    basis = sum(round(p.cost_basis_usd * _MICRO) for p in broker.get_positions().values())
    assert broker.cash_micro_usd + basis == (
        usd(broker.starting_cash_usd)
        + round(broker.realized_pnl_usd * _MICRO)
        - round(broker.failed_gas_usd * _MICRO)
    )

    # Realised P&L is the sum of what the rows said it was, to the micro-dollar.
    assert round(broker.realized_pnl_usd * _MICRO) == sum(
        round(row["realized_pnl_usd"] * _MICRO) for row in ledger
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 17, 404])
def test_cash_and_tokens_reconcile_exactly_over_a_random_sequence(
    tmp_path: Path, seed: int
) -> None:
    """The Phase 1 exit gate, stated as the audit states it: *every* simulated
    fill exactly reconciles cash and token atomic units. Not to a tolerance."""
    cfg = make_cfg(tmp_path, failed_tx_rate=0.1)
    broker = LocalPaperBroker(cfg, rng=random.Random(seed))
    _drive(broker, seed)
    _assert_reconciles(broker)


@pytest.mark.parametrize("seed", [5, 11, 23])
def test_a_random_sequence_survives_a_restart_at_every_point(
    tmp_path: Path, seed: int
) -> None:
    """Replaying the ledger from the checkpoint must reproduce the book exactly,
    for any sequence — that is what makes recovery a replay rather than a guess."""
    cfg = make_cfg(tmp_path, failed_tx_rate=0.1)
    broker = LocalPaperBroker(cfg, rng=random.Random(seed))
    _drive(broker, seed, steps=25)

    expected_cash = broker.cash_micro_usd
    expected_positions = broker.get_positions()

    cfg.state_path.unlink()  # worst case: no checkpoint at all, replay everything
    recovered = LocalPaperBroker(cfg, rng=random.Random(seed))
    assert recovered.cash_micro_usd == expected_cash
    assert recovered.get_positions() == expected_positions
    _assert_reconciles(recovered)


def test_the_conservative_fill_model_never_flatters_the_book(tmp_path: Path) -> None:
    """MIN_OUT must produce a book that is weakly worse than EXPECTED_OUT on the
    identical quote sequence. If conservatism ever *helped*, the sign of the
    slippage adjustment would be wrong somewhere."""
    quotes_and_intents = []
    now = NOW
    rng = random.Random(3)
    for _ in range(12):
        now += 1.0
        price = rng.uniform(1e-6, 5e-5)
        spend = round(rng.uniform(5.0, 40.0) * _MICRO)
        out_atomic = max(100, round(spend / _MICRO / price * _UNIT))
        q = make_quote(
            side=Side.BUY,
            in_amount_atomic=spend,
            out_amount_atomic=out_atomic,
            min_out_amount_atomic=out_atomic - out_atomic // 200,  # 50 bps
            now=now,
        )
        quotes_and_intents.append((make_intent(q), q, now))

    books = {}
    for model in (FillModel.MIN_OUT, FillModel.EXPECTED_OUT):
        cfg = make_cfg(tmp_path / model.value)
        broker = LocalPaperBroker(cfg, rng=never_fails(), fill_model=model)
        for intent, quote, when in quotes_and_intents:
            broker.place_order(intent, quote, now=when)
        books[model] = broker

    assert (
        books[FillModel.MIN_OUT].cash_micro_usd
        == books[FillModel.EXPECTED_OUT].cash_micro_usd
    )
    assert (
        books[FillModel.MIN_OUT].get_positions()["BONK"].quantity_atomic
        < books[FillModel.EXPECTED_OUT].get_positions()["BONK"].quantity_atomic
    )
