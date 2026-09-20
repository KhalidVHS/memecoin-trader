"""The report's arithmetic and its refusals.

``report.py`` is a display module, so the temptation is to test it by rendering
and grepping. Almost nothing here does that: the compute half returns frozen
dataclasses and these tests assert on their fields, which is the entire reason
the split exists. The four tests that *do* render are the ones where the
rendering is the claim — that an unmarkable position appears as UNMARKABLE, that
an unknown total does not print as ``$0.00``, and that the integrity section
appears above the numbers rather than below them.

What is asserted, mapped to the audit finding it closes:

* **C1** — the sample (fills, decisions, elapsed hours, round trips) is a
  first-class object; the caveat names the shortfall against the audit's
  prospective gate; the do-nothing counterfactual is computed.
* **C8** — an unmarkable position is neither dropped nor carried at cost; a
  ``None`` total propagates into every percentage and renders as unavailable;
  every valuation line carries its basis and haircut.
* **C11** — torn and corrupt ledger lines, open intents, failed fills and their
  gas, and decisions whose intents never settled all surface, and they render
  first.
* **Costs** — gas on failed swaps is counted, slippage is ``None`` rather than
  0 on an empty sample, and the $0.00 pool fee is labelled as embedded rather
  than absent.

Fixtures are built from the real types (``Fill``, ``Position``, ``Mark``,
``PortfolioState`` via ``portfolio.mark_book``) and the real ledger writer
(``journal.Ledger``). Nothing in the type layer is mocked: a test that passes
against a mock of ``Fill`` would keep passing after ``Fill`` changed shape,
which is exactly the drift this suite exists to catch.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from rich.console import Console

from memetrader import config, journal, report
from memetrader.http import BreakerState, BreakerStatus
from memetrader.portfolio import mark_book
from memetrader.types import (
    DecisionRecord,
    ExecutionMode,
    Fill,
    Mark,
    OrderIntent,
    OrderState,
    PortfolioState,
    Position,
    Provenance,
    Side,
)

NOW = 1_800_000_000.0
MINT = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_cfg(tmp_path: Path) -> config.Config:
    """A real config with the data dir redirected. No network, no shared state."""
    return replace(config.load(), data_dir=tmp_path)


def make_fill(
    *,
    fill_id: str = "f1",
    intent_id: str = "i1",
    decision_id: str | None = "d1",
    symbol: str = "BONK",
    side: Side = Side.BUY,
    state: OrderState = OrderState.LANDED,
    ts: float = NOW - 3600.0,
    notional_usd: float = 400.0,
    price_usd: float | None = 0.000_02,
    gas_usd: float = 0.21,
    pool_fee_usd: float = 0.0,
    realized_pnl_usd: float = 0.0,
    slippage_bps_vs_quote: float | None = None,
    price_impact_pct: float | None = 0.4,
    note: str | None = None,
) -> Fill:
    return Fill(
        fill_id=fill_id,
        order_id=f"o-{fill_id}",
        intent_id=intent_id,
        decision_id=decision_id,
        ts=ts,
        symbol=symbol,
        side=side,
        state=state,
        in_amount_atomic=400_000_000 if side is Side.BUY else 2_000_000_000,
        out_amount_atomic=2_000_000_000 if side is Side.BUY else 400_000_000,
        token_amount_atomic=2_000_000_000,
        token_decimals=5,
        quote_fingerprint=f"fp-{fill_id}",
        price_usd=price_usd,
        notional_usd=notional_usd,
        price_impact_pct=price_impact_pct,
        pool_fee_usd=pool_fee_usd,
        gas_usd=gas_usd,
        realized_pnl_usd=realized_pnl_usd,
        slippage_bps_vs_quote=slippage_bps_vs_quote,
        note=note,
    )


def failed_fill(**kwargs) -> Fill:
    """A failed swap: zero amounts moved, full gas paid. See ``Fill``'s docstring."""
    kwargs.setdefault("state", OrderState.FAILED)
    kwargs.setdefault("notional_usd", 0.0)
    kwargs.setdefault("note", "simulated tx failure")
    return make_fill(**kwargs)


def make_position(
    *,
    symbol: str = "BONK",
    quantity_atomic: int = 2_000_000_000,
    cost_basis_usd: float = 400.21,
) -> Position:
    return Position(
        symbol=symbol,
        mint=MINT,
        quantity_atomic=quantity_atomic,
        decimals=5,
        avg_entry_price_usd=0.00002,
        opened_at=NOW - 7200.0,
        cost_basis_usd=cost_basis_usd,
    )


def make_mark(
    symbol: str = "BONK",
    *,
    price_usd: float | None = 0.000_021,
    basis: str = "route",
    haircut_pct: float = 0.0,
    reason: str | None = None,
) -> Mark:
    return Mark(
        symbol=symbol,
        price_usd=price_usd,
        basis=basis,  # type: ignore[arg-type]
        provenance=(
            None if price_usd is None else Provenance(source="test", receive_time=NOW)
        ),
        haircut_pct=haircut_pct,
        reason=reason,
    )


def make_state(
    *,
    positions: dict[str, Position] | None = None,
    marks: dict[str, Mark] | None = None,
    cash_usd: float = 600.0,
    realized_pnl_usd: float = 0.0,
    starting_cash_usd: float = 1000.0,
    gas_paid_usd: float = 0.0,
) -> PortfolioState:
    """Marked through the real ``mark_book`` so the C8 invariants are the real ones."""
    return mark_book(
        cash_usd=cash_usd,
        positions=positions or {},
        marks=marks or {},
        realized_pnl_usd=realized_pnl_usd,
        starting_cash_usd=starting_cash_usd,
        gas_paid_usd=gas_paid_usd,
        now=NOW,
    )


def summary_for(state: PortfolioState, fills: tuple[Fill, ...] = ()) -> report.RunSummary:
    return report.build_run_summary(
        state=state,
        sample=report.build_sample(fills=fills, decisions=len(fills), now=NOW),
        stop_loss_pct=0.15,
        mode=ExecutionMode.PAPER,
    )


def rendered(fn, *args, **kwargs) -> str:
    console = Console(record=True, width=200, no_color=True)
    fn(console, *args, **kwargs)
    return console.export_text()


def decision_record(
    *,
    decision_id: str = "d1",
    ts: float = NOW - 3600.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
) -> DecisionRecord:
    return DecisionRecord(
        decision_id=decision_id,
        run_id="run-1",
        ts=ts,
        strategy_id="baseline-v1",
        market_read="nothing is happening",
        targets=(),
        bounds=(),
        intents=(),
        fills=(),
        mode=ExecutionMode.PAPER,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
        model="claude-opus-5",
        effort="high",
    )


def make_intent(
    *,
    intent_id: str = "i1",
    decision_id: str | None = "d1",
    symbol: str = "BONK",
    ts: float = NOW - 3600.0,
) -> OrderIntent:
    return OrderIntent(
        intent_id=intent_id,
        decision_id=decision_id,
        action_id=None,
        run_id="run-1",
        ts=ts,
        symbol=symbol,
        side=Side.BUY,
        in_amount_atomic=400_000_000,
        max_in_amount_atomic=400_000_000,
        source="strategy",
    )


# ---------------------------------------------------------------------------
# C8 — an unknown total is unavailable, never zero
# ---------------------------------------------------------------------------


def test_an_unmarkable_position_makes_the_total_unavailable_not_zero():
    state = make_state(positions={"BONK": make_position()}, marks={})
    s = summary_for(state)

    assert s.total_value_usd is None
    assert s.total_return_pct is None
    assert s.value_available is False
    # The trap: cash is known and the position is not, so a naive sum would
    # produce a confident $600.00 that is missing the whole position.
    assert s.cash_usd == 600.0
    assert s.unmarkable == ("BONK",)


def test_an_unknown_total_makes_the_counterfactual_comparison_unavailable():
    s = summary_for(make_state(positions={"BONK": make_position()}, marks={}))
    # The do-nothing book is always knowable; the *comparison* is not.
    assert s.cash_counterfactual_usd == 1000.0
    assert s.excess_vs_cash_usd is None


def test_an_unmarkable_position_is_neither_dropped_nor_marked_at_cost():
    state = make_state(
        positions={"BONK": make_position(cost_basis_usd=400.21)},
        marks={"BONK": make_mark(price_usd=None, basis="unavailable")},
    )
    (line,) = summary_for(state).positions

    assert line.symbol == "BONK"  # not dropped
    assert line.markable is False
    assert line.mark_price_usd is None
    assert line.value_usd is None
    assert line.unrealized_pnl_usd is None
    assert line.unrealized_pnl_pct is None
    # The C8 defect in one assertion: the cost basis must not have become the
    # value, because that shows exactly zero loss during a rug.
    assert line.value_usd != pytest.approx(line.cost_basis_usd)


def test_an_unmarkable_position_renders_as_unmarkable_and_the_total_as_unavailable():
    state = make_state(positions={"BONK": make_position()}, marks={})
    out = rendered(report.render_book, summary_for(state))

    assert "UNMARKABLE" in out
    assert "unavailable" in out
    # If a sum had leaked out, this is what it would have printed.
    assert "$600.00" not in out.split("cash")[-1].split("total")[-1]


def test_every_valuation_line_carries_its_basis_and_haircut():
    state = make_state(
        positions={"BONK": make_position()},
        marks={
            "BONK": make_mark(
                price_usd=0.0000205, basis="mid", haircut_pct=2.0, reason="pool mid"
            )
        },
    )
    (line,) = summary_for(state).positions

    assert line.basis == "mid"
    assert line.haircut_pct == 2.0
    assert line.executable_basis is False
    assert "mid" in rendered(report.render_book, summary_for(state))


def test_only_a_route_basis_counts_as_executable():
    state = make_state(
        positions={"BONK": make_position()}, marks={"BONK": make_mark(basis="route")}
    )
    (line,) = summary_for(state).positions
    assert line.executable_basis is True
    assert line.haircut_pct == 0.0


def test_a_marked_book_reports_a_total_and_an_excess_over_cash():
    state = make_state(
        positions={"BONK": make_position(cost_basis_usd=400.0)},
        marks={"BONK": make_mark(price_usd=0.02)},
        cash_usd=600.0,
        starting_cash_usd=1000.0,
    )
    s = summary_for(state)
    # 2_000_000_000 atomic at 5 decimals is 20,000 UI tokens; at $0.02 that is
    # $400.00, plus $600 cash. The atomic/decimals conversion is the point.
    assert s.total_value_usd == pytest.approx(1000.0)
    assert s.excess_vs_cash_usd == pytest.approx(0.0)
    assert s.total_return_pct == pytest.approx(0.0)


def test_the_stop_price_is_the_portfolio_modules_definition():
    from memetrader.portfolio import stop_price_usd

    position = make_position(cost_basis_usd=400.21)
    state = make_state(positions={"BONK": position}, marks={"BONK": make_mark()})
    (line,) = summary_for(state).positions
    assert line.stop_price_usd == stop_price_usd(position, 0.15)


# ---------------------------------------------------------------------------
# C1 — the sample, and the refusal to claim alpha
# ---------------------------------------------------------------------------


def test_the_low_sample_caveat_names_the_shortfall_and_refuses_an_alpha_claim():
    fills = (
        make_fill(fill_id="f1", ts=NOW - 7200.0),
        make_fill(fill_id="f2", side=Side.SELL, ts=NOW - 3600.0, realized_pnl_usd=12.0),
    )
    sample = report.build_sample(fills=fills, decisions=3, now=NOW)

    assert sample.adequate is False
    assert sample.closed_round_trips == 1
    assert sample.landed_fills == 2
    assert sample.decisions == 3
    assert sample.elapsed_hours == pytest.approx(2.0)
    assert "NO ALPHA CLAIM IS SUPPORTABLE" in sample.caveat
    assert str(report.ADEQUATE_ROUND_TRIPS) in sample.caveat
    assert sample.round_trip_shortfall == pytest.approx(report.ADEQUATE_ROUND_TRIPS)


def test_the_caveat_is_rendered_with_the_headline_not_in_a_footnote():
    out = rendered(report.render_summary, summary_for(make_state()))
    assert "NO ALPHA CLAIM IS SUPPORTABLE" in out
    assert out.index("Sample") < out.index("book value")


def test_an_empty_record_reports_no_elapsed_time_rather_than_zero_hours():
    sample = report.build_sample(fills=(), decisions=0, now=NOW)
    assert sample.elapsed_hours is None
    assert sample.round_trip_shortfall is None
    assert "no elapsed time" in sample.caveat


def test_the_buy_and_hold_benchmark_is_declared_uncomputable_rather_than_guessed():
    note = summary_for(make_state()).benchmark_note
    assert "unavailable" in note
    assert "t0" in note


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------


def test_gas_on_failed_swaps_is_counted_and_broken_out():
    fills = (
        make_fill(fill_id="f1", gas_usd=0.21),
        failed_fill(fill_id="f2", intent_id="i2", gas_usd=0.21),
        failed_fill(fill_id="f3", intent_id="i3", gas_usd=0.30),
    )
    costs = report.build_cost_breakdown(fills)

    assert costs.gas_paid_usd == pytest.approx(0.72)
    assert costs.gas_on_failed_usd == pytest.approx(0.51)
    assert "FAILED" in rendered(report.render_costs, costs)


def test_a_failed_fill_is_not_a_landed_fill_and_not_a_round_trip():
    fills = (
        failed_fill(fill_id="f1", side=Side.SELL),
        make_fill(fill_id="f2", side=Side.SELL, realized_pnl_usd=5.0),
    )
    stats = report.build_trade_stats(fills)

    assert stats.fills == 2
    assert stats.failed == 1
    assert stats.landed == 1
    assert stats.closed_round_trips == 1


def test_a_zero_pool_fee_is_labelled_embedded_rather_than_free():
    costs = report.build_cost_breakdown((make_fill(pool_fee_usd=0.0),))
    assert costs.pool_fees_usd == 0.0
    assert costs.pool_fee_decomposed is False
    assert "embedded" in costs.pool_fee_note
    assert "NOT because trading was free" in costs.pool_fee_note


def test_slippage_with_no_samples_is_none_not_zero():
    costs = report.build_cost_breakdown((make_fill(slippage_bps_vs_quote=None),))
    assert costs.slippage_samples == 0
    assert costs.mean_slippage_bps is None
    assert costs.worst_slippage_bps is None
    assert "n/a" in rendered(report.render_costs, costs)


def test_worst_slippage_is_the_largest_magnitude_not_the_largest_signed_value():
    fills = (
        make_fill(fill_id="f1", slippage_bps_vs_quote=12.0),
        make_fill(fill_id="f2", slippage_bps_vs_quote=-40.0),
    )
    costs = report.build_cost_breakdown(fills)
    assert costs.mean_slippage_bps == pytest.approx(-14.0)
    assert costs.worst_slippage_bps == pytest.approx(-40.0)


def test_cost_ratio_is_none_when_nothing_was_traded():
    costs = report.build_cost_breakdown((failed_fill(),))
    assert costs.notional_traded_usd == 0.0
    assert costs.cost_bps_of_notional is None


# ---------------------------------------------------------------------------
# Trade statistics
# ---------------------------------------------------------------------------


def test_trade_stats_over_an_empty_record_are_none_not_zero():
    stats = report.build_trade_stats(())
    assert stats.win_rate_pct is None
    assert stats.avg_win_usd is None
    assert stats.avg_loss_usd is None
    assert stats.expectancy_usd is None
    assert stats.profit_factor is None
    assert stats.largest_win_share_pct is None


def test_trade_stats_worked_example():
    fills = (
        make_fill(fill_id="b1", side=Side.BUY),
        make_fill(fill_id="s1", side=Side.SELL, realized_pnl_usd=10.0),
        make_fill(fill_id="s2", side=Side.SELL, realized_pnl_usd=-4.0),
    )
    stats = report.build_trade_stats(fills)

    assert stats.closed_round_trips == 2  # the BUY is not an outcome
    assert stats.win_rate_pct == pytest.approx(50.0)
    assert stats.avg_win_usd == pytest.approx(10.0)
    assert stats.avg_loss_usd == pytest.approx(-4.0)
    assert stats.expectancy_usd == pytest.approx(3.0)
    assert stats.profit_factor == pytest.approx(2.5)
    assert stats.largest_win_share_pct == pytest.approx(100.0)


def test_profit_factor_is_none_rather_than_infinite_with_no_losses():
    stats = report.build_trade_stats((make_fill(side=Side.SELL, realized_pnl_usd=10.0),))
    assert stats.profit_factor is None
    assert "n/a" in rendered(report.render_trades, stats)


# ---------------------------------------------------------------------------
# Model spend
# ---------------------------------------------------------------------------


def test_model_spend_uses_the_configs_cost_formula_including_cache_writes(tmp_path):
    cfg = make_cfg(tmp_path)
    decisions = (
        report.summarize_decision(
            decision_record(input_tokens=1000, output_tokens=200, cache_read=5000)
        ),
        report.summarize_decision(
            decision_record(decision_id="d2", input_tokens=500, cache_write=2000)
        ),
    )
    spend = report.build_model_spend(decisions, model=cfg.model, strategy_kind="advisory")

    assert spend.calls == 2
    assert spend.cost_usd == pytest.approx(cfg.model.cost_usd(1500, 200, 5000, 2000))
    # Independent of the formula: a cache write must cost more than nothing, so
    # dropping the bucket would change the answer.
    assert spend.cost_usd > cfg.model.cost_usd(1500, 200, 5000, 0)


def test_model_spend_is_labelled_irrelevant_when_the_strategy_makes_no_calls(tmp_path):
    cfg = make_cfg(tmp_path)
    spend = report.build_model_spend(
        (report.summarize_decision(decision_record(input_tokens=1000)),),
        model=cfg.model,
        strategy_kind="momentum",
    )
    assert spend.relevant is False
    assert spend.projected_daily_usd(900.0) is None
    out = rendered(report.render_spend, spend, slow_tick_seconds=900.0)
    assert "not applicable" in out


def test_decisions_with_no_token_usage_are_not_counted_as_calls(tmp_path):
    cfg = make_cfg(tmp_path)
    decisions = (
        report.summarize_decision(decision_record(decision_id="d1")),
        report.summarize_decision(
            decision_record(decision_id="d2", input_tokens=100, output_tokens=10)
        ),
    )
    spend = report.build_model_spend(decisions, model=cfg.model, strategy_kind="advisory")
    # Two decision rows, one billed call. Counting both would halve the reported
    # cost per call for every deterministic tick in the record.
    assert spend.calls == 1
    assert spend.cost_per_call_usd == pytest.approx(spend.cost_usd)


def test_cost_per_call_is_none_rather_than_a_division_by_zero(tmp_path):
    cfg = make_cfg(tmp_path)
    spend = report.build_model_spend((), model=cfg.model, strategy_kind="advisory")
    assert spend.calls == 0
    assert spend.cost_per_call_usd is None
    assert spend.cache_hit_rate is None


# ---------------------------------------------------------------------------
# C11 — integrity
# ---------------------------------------------------------------------------


def test_a_torn_final_ledger_line_surfaces(tmp_path):
    cfg = make_cfg(tmp_path)
    ledger = journal.Ledger(cfg.ledger_path, run_id="run-1", fsync=False)
    ledger.append_fill(make_fill())
    # Exactly what a process killed mid-append leaves: a partial line with no
    # trailing newline.
    with cfg.ledger_path.open("a", encoding="utf-8") as fh:
        fh.write('{"kind": "fill", "row_id": "f2", "pay')

    facts = report.collect(cfg)
    integrity = report.build_integrity(
        scans=facts.scans, fills=facts.fills, intents=facts.intents
    )

    assert len(facts.fills) == 1
    assert integrity.torn_lines
    assert integrity.ok is False
    assert "torn ledger line" in rendered(report.render_integrity, integrity)


def test_a_corrupt_mid_file_line_surfaces_and_the_rows_after_it_survive(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    ledger = journal.Ledger(cfg.ledger_path, run_id="run-1", fsync=False)
    ledger.append_fill(make_fill(fill_id="f1"))
    with cfg.ledger_path.open("a", encoding="utf-8") as fh:
        fh.write("{ not json at all\n")
    ledger = journal.Ledger(cfg.ledger_path, run_id="run-1", fsync=False)
    ledger.append_fill(make_fill(fill_id="f2", intent_id="i2"))

    facts = report.collect(cfg)
    integrity = report.build_integrity(scans=facts.scans, fills=facts.fills)

    assert len(facts.fills) == 2  # the rows after the bad one are a true record
    assert integrity.corrupt_lines
    assert "CORRUPT" in rendered(report.render_integrity, integrity)


def test_an_open_intent_surfaces(tmp_path):
    cfg = make_cfg(tmp_path)
    ledger = journal.Ledger(cfg.ledger_path, run_id="run-1", fsync=False)
    ledger.append_intent(make_intent(intent_id="i1"))
    ledger.append_state(intent_id="i1", state=OrderState.SUBMITTED, ts=NOW - 60.0)

    facts = report.collect(cfg)
    integrity = report.build_integrity(
        scans=facts.scans, fills=facts.fills, open_intents=facts.open_intents
    )

    (open_intent,) = integrity.open_intents
    assert open_intent.intent_id == "i1"
    assert open_intent.was_submitted is True
    assert integrity.ok is False
    assert "OPEN INTENT" in rendered(report.render_integrity, integrity)


def test_an_intent_with_a_terminal_fill_is_not_open(tmp_path):
    cfg = make_cfg(tmp_path)
    ledger = journal.Ledger(cfg.ledger_path, run_id="run-1", fsync=False)
    ledger.append_intent(make_intent(intent_id="i1"))
    ledger.append_fill(make_fill(fill_id="f1", intent_id="i1"))

    facts = report.collect(cfg)
    assert facts.open_intents == ()


def test_a_decision_whose_intents_never_settled_surfaces(tmp_path):
    cfg = make_cfg(tmp_path)
    ledger = journal.Ledger(cfg.ledger_path, run_id="run-1", fsync=False)
    ledger.append_decision(decision_record(decision_id="d1"))
    ledger.append_intent(make_intent(intent_id="i1", decision_id="d1"))
    ledger.append_intent(make_intent(intent_id="i2", decision_id="d1", symbol="WIF"))
    # Only one of the two intents reached a terminal state.
    ledger.append_fill(make_fill(fill_id="f1", intent_id="i1", decision_id="d1"))

    facts = report.collect(cfg)
    integrity = report.build_integrity(
        scans=facts.scans,
        fills=facts.fills,
        intents=facts.intents,
        decisions=facts.decisions,
    )

    (gap,) = integrity.decision_gaps
    assert gap.decision_id == "d1"
    assert gap.intent_ids == ("i2",)
    assert gap.symbols == ("WIF",)
    assert "NO TERMINAL FILL" in rendered(report.render_integrity, integrity)


def test_failed_fills_and_their_gas_surface_in_the_integrity_section():
    integrity = report.build_integrity(
        fills=(make_fill(fill_id="f1"), failed_fill(fill_id="f2", gas_usd=0.21))
    )
    assert len(integrity.failed_fills) == 1
    assert integrity.gas_burned_on_failures_usd == pytest.approx(0.21)
    out = rendered(report.render_integrity, integrity)
    assert "FAILED FILLS" in out
    assert "$0.21" in out


def test_an_open_circuit_breaker_host_surfaces_when_one_is_passed_in():
    breakers = (
        BreakerStatus(
            host="api.dexscreener.com",
            state=BreakerState.OPEN,
            consecutive_failures=4,
            cooldown_remaining_seconds=42.0,
            last_error="timeout",
            opened_count=1,
        ),
        BreakerStatus(
            host="quote-api.jup.ag",
            state=BreakerState.HALF_OPEN,
            consecutive_failures=4,
            cooldown_remaining_seconds=None,
            last_error="timeout",
            opened_count=1,
        ),
    )
    integrity = report.build_integrity(breakers=breakers)

    # HALF_OPEN is mid-recovery and is deliberately not reported as open.
    assert integrity.open_breaker_hosts == ("api.dexscreener.com",)
    assert "CIRCUIT OPEN" in rendered(report.render_integrity, integrity)


def test_breaker_status_is_optional_and_a_clean_record_says_so():
    integrity = report.build_integrity()
    assert integrity.ok is True
    assert integrity.breakers == ()
    assert "integrity ok" in rendered(report.render_integrity, integrity)


def test_an_unreadable_fill_row_is_reported_rather_than_aborting_the_report(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    # Valid JSON, invalid fill: the broker would raise LedgerCorrupt here, which
    # is right for trading and wrong for the tool you investigate it with.
    cfg.trades_path.write_text('{"fill_id": "f1", "symbol": "BONK"}\n', encoding="utf-8")

    facts = report.collect(cfg)
    assert facts.fills == ()
    assert facts.unreadable_rows
    integrity = report.build_integrity(
        scans=facts.scans, unreadable_rows=facts.unreadable_rows
    )
    assert "unreadable row" in rendered(report.render_integrity, integrity)


def test_a_fill_in_both_the_ledger_and_the_trades_file_is_counted_once(tmp_path):
    cfg = make_cfg(tmp_path)
    fill = make_fill(fill_id="f1", gas_usd=0.21)
    journal.Ledger(cfg.ledger_path, run_id="run-1", fsync=False).append_fill(fill)
    journal.append(cfg.trades_path, _flat_fill_row(fill), fsync=False)

    facts = report.collect(cfg)
    assert len(facts.fills) == 1
    assert report.build_cost_breakdown(facts.fills).gas_paid_usd == pytest.approx(0.21)


def _flat_fill_row(fill: Fill) -> dict:
    """The broker's flat ``trades.jsonl`` shape, which ``collect`` also reads."""
    row = {f: getattr(fill, f) for f in Fill.__slots__}
    row["side"] = str(fill.side)
    row["state"] = str(fill.state)
    row["mint"] = MINT
    return row


# ---------------------------------------------------------------------------
# The whole report
# ---------------------------------------------------------------------------


def test_integrity_is_rendered_above_the_numbers(tmp_path):
    cfg = make_cfg(tmp_path)
    ledger = journal.Ledger(cfg.ledger_path, run_id="run-1", fsync=False)
    ledger.append_fill(failed_fill(fill_id="f1"))

    rep = report.load_report(
        cfg,
        state=make_state(positions={"BONK": make_position()}, marks={}),
        now=NOW,
    )
    out = rendered(report.render_report, rep)

    assert out.index("record integrity") < out.index("Sample")
    assert out.index("Sample") < out.index("book value")
    assert "FAILED FILLS" in out
    assert "UNMARKABLE" in out


def test_load_report_over_an_empty_data_dir_reports_nothing_rather_than_failing(tmp_path):
    cfg = make_cfg(tmp_path)
    rep = report.load_report(cfg, state=make_state(), now=NOW)

    assert rep.integrity.ok is True
    assert rep.trades.fills == 0
    assert rep.costs.gas_paid_usd == 0.0
    assert rep.summary.sample.elapsed_hours is None
    # It still renders, and it still refuses to claim anything.
    assert "NO ALPHA CLAIM IS SUPPORTABLE" in rendered(report.render_report, rep)


def test_load_report_makes_no_network_call(tmp_path, monkeypatch):
    """``report`` must work offline, after the fact. Audit-adjacent, but the
    reason is operational: the one time you need the ledger explained is the
    time the data source is down."""
    import httpx

    def explode(*args, **kwargs):
        raise AssertionError("report.py must not make a network call")

    monkeypatch.setattr(httpx.Client, "request", explode)
    monkeypatch.setattr(httpx.Client, "send", explode)

    cfg = make_cfg(tmp_path)
    journal.Ledger(cfg.ledger_path, run_id="run-1", fsync=False).append_fill(make_fill())
    rep = report.load_report(cfg, state=make_state(), now=NOW)
    rendered(report.render_report, rep)
    assert rep.trades.fills == 1
