"""The durable ledger: crash recovery, idempotency, joins by ID, and encoding.

Audit C11 is the thing under test. The old shape appended a trade row and then
saved a state file, and a kill between the two left two files that disagreed
with no ID to reconcile them by. Each section below pins one of the properties
that replaced it, and every test states the failure it is standing guard over
rather than restating the function name.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from memetrader import ids, journal
from memetrader.types import Fill, OrderIntent, OrderState, Side, TxnCounts

NOW = 1_700_000_000.0
RUN = "run-000001700000000000000-abcd1234"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def intent(**kw: Any) -> OrderIntent:
    base: dict[str, Any] = {
        "intent_id": ids.new_intent_id(),
        "decision_id": None,
        "action_id": None,
        "run_id": RUN,
        "ts": NOW,
        "symbol": "BONK",
        "side": Side.BUY,
        "in_amount_atomic": 100_000_000,
        "max_in_amount_atomic": 100_000_000,
        "source": "strategy",
    }
    return OrderIntent(**{**base, **kw})


def fill(**kw: Any) -> Fill:
    base: dict[str, Any] = {
        "fill_id": ids.new_fill_id(),
        "order_id": ids.new_order_id(),
        "intent_id": ids.new_intent_id(),
        "decision_id": None,
        "ts": NOW,
        "symbol": "BONK",
        "side": Side.BUY,
        "state": OrderState.LANDED,
        "in_amount_atomic": 100_000_000,
        "out_amount_atomic": 5_000_000_000_000,
        "token_amount_atomic": 5_000_000_000_000,
        "token_decimals": 5,
        "quote_fingerprint": "deadbeef",
        "price_usd": 0.00002,
        "notional_usd": 100.0,
        "price_impact_pct": 0.4,
        "pool_fee_usd": 0.25,
        "gas_usd": 0.21,
    }
    return Fill(**{**base, **kw})


def ledger(tmp_path: Path) -> journal.Ledger:
    return journal.Ledger(tmp_path / "ledger.jsonl", run_id=RUN)


def lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


# ---------------------------------------------------------------------------
# Torn final line — a kill mid-append
# ---------------------------------------------------------------------------


def test_a_torn_final_line_is_discarded_and_reported(tmp_path: Path) -> None:
    """The only corruption an append-only file can produce by itself: the
    process died between the ``write`` and the ``\\n``.

    Discarding it is right — a half-written JSON object is not partially true,
    because you cannot tell whether the missing half held a field that changes
    its meaning. Doing it *silently* is what was wrong before: "the ledger has
    two rows" and "the ledger has two rows and one that could not be read" are
    different facts, and only the second explains a missing trade."""
    path = tmp_path / "ledger.jsonl"
    path.write_text(
        '{"row_id": "int-1", "kind": "intent"}\n'
        '{"row_id": "int-2", "kind": "intent"}\n'
        '{"row_id": "int-3", "kind": "int',
        encoding="utf-8",
    )

    result = journal.scan(path)

    assert [r["row_id"] for r in result.rows] == ["int-1", "int-2"]
    assert result.torn_final_line is True
    assert result.torn_final_bytes > 0
    assert result.corrupt_line_numbers == ()
    assert result.clean is False
    assert result.discarded == 1


def test_a_complete_file_reports_no_tear(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    journal.append(path, {"row_id": "a", "kind": "note"})

    result = journal.scan(path)
    assert result.torn_final_line is False
    assert result.clean is True


def test_mid_file_corruption_is_reported_separately_from_a_tear(
    tmp_path: Path,
) -> None:
    """Different causes, different responses. A torn *final* line is an expected
    consequence of a kill; a bad line in the middle means something rewrote the
    file, which nothing in this system is allowed to do."""
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"row_id": "a"}\nnot json at all\n{"row_id": "b"}\n', encoding="utf-8")

    result = journal.scan(path)

    assert [r["row_id"] for r in result.rows] == ["a", "b"]
    assert result.torn_final_line is False
    assert result.corrupt_line_numbers == (2,)


def test_a_bad_final_line_that_ends_in_a_newline_is_corruption_not_a_tear(
    tmp_path: Path,
) -> None:
    """Detected structurally — failed to parse *and* the file does not end in a
    newline — rather than by guessing from the content. A genuinely malformed
    row that was written completely must not be excused as a crash."""
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"row_id": "a"}\n{"broken": \n', encoding="utf-8")

    result = journal.scan(path)
    assert result.torn_final_line is False
    assert result.corrupt_line_numbers == (2,)


def test_rows_before_the_tear_survive_byte_for_byte(tmp_path: Path) -> None:
    lg = ledger(tmp_path)
    good = intent(symbol="WIF")
    lg.append_intent(good)
    before = lg.path.read_text(encoding="utf-8")

    with lg.path.open("a", encoding="utf-8") as fh:
        fh.write('{"row_id": "int-tor')

    assert lg.path.read_text(encoding="utf-8").startswith(before)
    result = journal.scan(lg.path)
    assert result.torn_final_line is True
    assert [r["intent_id"] for r in result.rows] == [good.intent_id]


def test_a_ledger_opened_over_a_torn_file_reports_it_at_construction(
    tmp_path: Path,
) -> None:
    """Recovery is the first thing startup does, so the tear has to be visible
    from the object it opens rather than only from a log line it may not be
    configured to emit."""
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"row_id": "a", "kind": "note"}\n{"row_id": "b', encoding="utf-8")

    lg = journal.Ledger(path, run_id=RUN)
    assert lg.opened_with.torn_final_line is True


def test_a_torn_line_is_logged_rather_than_vanishing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"row_id": "a"}\n{"row', encoding="utf-8")

    with caplog.at_level("WARNING"):
        journal.scan(path)

    assert "torn final line" in caplog.text


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_appending_the_same_intent_twice_writes_one_row(tmp_path: Path) -> None:
    """``intent_id`` is the idempotency key: a retry reuses it, so a duplicated
    attempt has to be detectable rather than becoming a second order. This is
    the "unique constraint" the audit says does not exist, in the only form a
    JSONL file can offer one."""
    lg = ledger(tmp_path)
    order = intent()

    assert lg.append_intent(order) is True
    assert lg.append_intent(order) is False

    assert len(lines(lg.path)) == 1
    assert len(lg.rows()) == 1


def test_appending_the_same_fill_twice_writes_one_row(tmp_path: Path) -> None:
    lg = ledger(tmp_path)
    settled = fill()

    assert lg.append_fill(settled) is True
    assert lg.append_fill(settled) is False
    assert len(lg.rows()) == 1


def test_idempotency_survives_a_process_restart(tmp_path: Path) -> None:
    """The index is rebuilt from the file at construction, which is what makes
    recovery safe to run repeatedly — the case that matters, since a crashed
    process is exactly the one that will be restarted."""
    order = intent()
    first = ledger(tmp_path)
    first.append_intent(order)

    second = journal.Ledger(tmp_path / "ledger.jsonl", run_id=RUN)
    assert second.append_intent(order) is False
    assert len(second.rows()) == 1


def test_two_distinct_intents_are_both_written(tmp_path: Path) -> None:
    """The mirror of the test above: deduplication must not be so eager that it
    collapses two genuine orders. Content hashing would have done exactly that,
    which is why ``ids.py`` mints random-suffixed IDs instead."""
    lg = ledger(tmp_path)
    assert lg.append_intent(intent()) is True
    assert lg.append_intent(intent()) is True
    assert len(lg.rows()) == 2


def test_the_same_state_transition_recorded_twice_is_one_row(tmp_path: Path) -> None:
    """Keyed on ``(intent_id, state)``, not on ``intent_id``: an order passes
    through several states and each is a distinct row, but re-running recovery
    must not double them."""
    lg = ledger(tmp_path)
    order = intent()
    lg.append_intent(order)

    assert lg.append_state(intent_id=order.intent_id, state=OrderState.SUBMITTED, ts=NOW)
    assert not lg.append_state(
        intent_id=order.intent_id, state=OrderState.SUBMITTED, ts=NOW
    )
    assert lg.append_state(intent_id=order.intent_id, state=OrderState.LANDED, ts=NOW + 1)

    assert len(lg.rows()) == 3


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def test_an_intent_with_no_terminal_state_is_returned_by_reconciliation(
    tmp_path: Path,
) -> None:
    """What a crash leaves behind, and what startup must resolve before the
    symbol may be traded again. The audit's failure table for this row reads
    "Reconcile before any new order"."""
    lg = ledger(tmp_path)
    order = intent(symbol="WIF")
    lg.append_intent(order)
    lg.append_state(intent_id=order.intent_id, state=OrderState.SUBMITTED, ts=NOW + 1)

    (open_one,) = lg.open_intents()
    assert open_one.intent_id == order.intent_id
    assert open_one.symbol == "WIF"
    assert open_one.last_state is OrderState.SUBMITTED
    assert open_one.was_submitted is True


def test_an_intent_with_a_landed_fill_is_not_returned(tmp_path: Path) -> None:
    lg = ledger(tmp_path)
    order = intent()
    lg.append_intent(order)
    lg.append_fill(fill(intent_id=order.intent_id, state=OrderState.LANDED))

    assert lg.open_intents() == ()


@pytest.mark.parametrize(
    "state",
    [OrderState.LANDED, OrderState.FAILED, OrderState.EXPIRED, OrderState.RECONCILED],
)
def test_every_terminal_state_closes_an_intent(tmp_path: Path, state: OrderState) -> None:
    """A *failed* attempt closes it too. Leaving a failed order open forever
    would halt the symbol on a fault that has already been fully accounted
    for — including its gas, which the fill row still carries."""
    lg = ledger(tmp_path)
    order = intent()
    lg.append_intent(order)
    lg.append_fill(fill(intent_id=order.intent_id, state=state, notional_usd=0.0))

    assert lg.open_intents() == ()


@pytest.mark.parametrize(
    "state",
    [
        OrderState.PROPOSED,
        OrderState.RISK_APPROVED,
        OrderState.QUOTE_BOUND,
        OrderState.SUBMITTED,
    ],
)
def test_no_non_terminal_state_closes_an_intent(tmp_path: Path, state: OrderState) -> None:
    lg = ledger(tmp_path)
    order = intent()
    lg.append_intent(order)
    lg.append_state(intent_id=order.intent_id, state=state, ts=NOW + 1)

    assert len(lg.open_intents()) == 1


def test_only_the_open_intent_is_returned_when_others_settled(tmp_path: Path) -> None:
    lg = ledger(tmp_path)
    settled, stranded = intent(symbol="BONK"), intent(symbol="WIF")
    lg.append_intent(settled)
    lg.append_intent(stranded)
    lg.append_fill(fill(intent_id=settled.intent_id, symbol="BONK"))
    lg.append_state(intent_id=stranded.intent_id, state=OrderState.SUBMITTED, ts=NOW + 2)

    assert [o.symbol for o in lg.open_intents()] == ["WIF"]


def test_an_unrecognised_state_string_does_not_close_an_intent(tmp_path: Path) -> None:
    """Treating an unreadable state as terminal would silently close an order
    whose outcome we cannot read, which is the opposite of what recovery is
    for."""
    path = tmp_path / "ledger.jsonl"
    order = intent()
    journal.Ledger(path, run_id=RUN).append_intent(order)
    journal.append(
        path,
        {
            "schema_version": journal.SCHEMA_VERSION,
            "row_id": "x",
            "kind": "order_state",
            "ts": NOW,
            "intent_id": order.intent_id,
            "state": "teleported",
        },
    )

    assert len(journal.Ledger(path, run_id=RUN).open_intents()) == 1


def test_open_intents_come_back_oldest_first(tmp_path: Path) -> None:
    """Relies on the property ``ids.py`` promises: the IDs carry a zero-padded
    18-digit microsecond prefix, so **lexical sort is chronological sort**. The
    oldest open intent is the one most likely to have actually settled at the
    venue, so it is the one to resolve first."""
    lg = ledger(tmp_path)
    first, second, third = intent(), intent(), intent()
    for order in (third, first, second):  # written out of order on purpose
        lg.append_intent(order)

    got = [o.intent_id for o in lg.open_intents()]
    assert got == sorted(got)
    assert got == sorted([first.intent_id, second.intent_id, third.intent_id])


# ---------------------------------------------------------------------------
# Joining by ID, never by symbol
# ---------------------------------------------------------------------------


def test_a_stop_loss_and_a_strategy_fill_on_the_same_symbol_are_distinguishable(
    tmp_path: Path,
) -> None:
    """The audit's ``prompts._decision_line`` finding, reproduced and fixed.

    Same coin, same tick, two orders: a stop-loss exit and a strategy SELL. The
    old code matched fills to actions by *symbol*, so both landed on one key
    and the report attributed them to whichever it looked at first — in the one
    case where you urgently need to know which policy sold. A symbol is a
    property of an order; an ``intent_id`` *is* the order."""
    lg = ledger(tmp_path)
    stop = intent(symbol="BONK", side=Side.SELL, source="stop_loss", reason="-15%")
    strategy = intent(symbol="BONK", side=Side.SELL, source="strategy", reason="target")
    lg.append_intent(stop)
    lg.append_intent(strategy)

    stop_fill = fill(
        intent_id=stop.intent_id, symbol="BONK", side=Side.SELL, realized_pnl_usd=-15.0
    )
    strategy_fill = fill(
        intent_id=strategy.intent_id, symbol="BONK", side=Side.SELL, realized_pnl_usd=4.0
    )
    lg.append_fill(stop_fill)
    lg.append_fill(strategy_fill)

    grouped = journal.fills_by_intent(lg.rows())

    assert set(grouped) == {stop.intent_id, strategy.intent_id}
    assert len(grouped[stop.intent_id]) == 1
    assert len(grouped[strategy.intent_id]) == 1
    assert grouped[stop.intent_id][0]["fill_id"] == stop_fill.fill_id
    assert grouped[stop.intent_id][0]["payload"]["realized_pnl_usd"] == -15.0
    assert grouped[strategy.intent_id][0]["payload"]["realized_pnl_usd"] == 4.0

    # And the *cause* of each is recoverable, which is the point of asking.
    by_intent = {r["intent_id"]: r for r in lg.rows() if r["kind"] == "intent"}
    assert by_intent[stop.intent_id]["payload"]["source"] == "stop_loss"
    assert by_intent[strategy.intent_id]["payload"]["source"] == "strategy"


def test_rows_are_gathered_by_decision_id_not_by_a_time_window(
    tmp_path: Path,
) -> None:
    """Two ticks that overlap — a slow tick still running when the next fires —
    are inseparable under a time window and trivially separable by ID."""
    lg = ledger(tmp_path)
    first_decision, second_decision = ids.new_decision_id(), ids.new_decision_id()
    a = intent(symbol="BONK", decision_id=first_decision)
    b = intent(symbol="BONK", decision_id=second_decision)
    lg.append_intent(a)
    lg.append_intent(b)
    lg.append_fill(fill(intent_id=a.intent_id, decision_id=first_decision))
    lg.append_fill(fill(intent_id=b.intent_id, decision_id=second_decision))

    rows = journal.rows_for_decision(lg.rows(), first_decision)
    assert {r["intent_id"] for r in rows} == {a.intent_id}


def test_every_row_carries_the_run_id_and_the_schema_version(tmp_path: Path) -> None:
    """``run_id`` on every row is what makes a duplicate-process incident
    *visible* afterwards — two run IDs interleaved in one file — rather than
    inexplicable state."""
    lg = ledger(tmp_path)
    order = intent()
    lg.append_intent(order)
    lg.append_state(intent_id=order.intent_id, state=OrderState.SUBMITTED, ts=NOW)
    lg.append_fill(fill(intent_id=order.intent_id))

    for row in lg.rows():
        assert row["run_id"] == RUN
        assert row["schema_version"] == journal.SCHEMA_VERSION
        assert isinstance(row["row_id"], str)


def test_the_file_sorts_chronologically_by_row_id_without_parsing(
    tmp_path: Path,
) -> None:
    """The ids.py promise, relied on here: ``sort`` on the raw file is a correct
    chronological sort. That is what makes a 400-megabyte ledger inspectable
    from a shell."""
    lg = ledger(tmp_path)
    for _ in range(5):
        lg.append_intent(intent())

    row_ids = [r["row_id"] for r in lg.rows()]
    assert row_ids == sorted(row_ids)


# ---------------------------------------------------------------------------
# Durability
# ---------------------------------------------------------------------------


def test_an_append_is_fsynced_before_it_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``flush`` moves the bytes to the OS; only ``fsync`` moves them to the
    device. Without it, a power loss can lose a row the caller was told was
    persisted — and by construction the row you lose is the most recent one,
    which is the order that was in flight."""
    synced: list[int] = []
    real_fsync = journal.os.fsync

    def spy(fd: int) -> None:
        synced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(journal.os, "fsync", spy)

    path = tmp_path / "ledger.jsonl"
    journal.append(path, {"row_id": "a", "kind": "note"})

    assert len(synced) == 1


def test_the_ledger_fsyncs_every_kind_of_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real_fsync = journal.os.fsync
    monkeypatch.setattr(
        journal.os, "fsync", lambda fd: (calls.append(fd), real_fsync(fd))[1]
    )

    lg = ledger(tmp_path)
    order = intent()
    lg.append_intent(order)
    lg.append_state(intent_id=order.intent_id, state=OrderState.SUBMITTED, ts=NOW)
    lg.append_fill(fill(intent_id=order.intent_id))

    assert len(calls) == 3


def test_a_suppressed_duplicate_does_not_touch_the_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An fsync is a few hundred microseconds to a few milliseconds. Paying it
    to write nothing would make a retry storm expensive as well as pointless."""
    lg = ledger(tmp_path)
    order = intent()
    lg.append_intent(order)

    calls: list[int] = []
    monkeypatch.setattr(journal.os, "fsync", lambda fd: calls.append(fd))
    assert lg.append_intent(order) is False
    assert calls == []


def test_a_whole_file_rewrite_goes_through_a_temp_file_and_a_replace(
    tmp_path: Path,
) -> None:
    """A reader sees the old file or the new one, never a half-written one.
    ``Path.replace`` is ``os.replace``; on Windows it is ``MoveFileEx`` with
    ``REPLACE_EXISTING`` and it is atomic."""
    path = tmp_path / "snapshot.json"
    journal.atomic_write_text(path, '{"cash_usd": 1000.0}')
    journal.atomic_write_text(path, '{"cash_usd": 900.0}')

    assert json.loads(path.read_text(encoding="utf-8"))["cash_usd"] == 900.0
    assert [p.name for p in tmp_path.iterdir()] == ["snapshot.json"]


def test_a_failed_rewrite_leaves_no_temp_file_and_no_damage(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    journal.atomic_write_text(path, "original")

    # A non-string payload fails inside the temp file, before the replace. The
    # original must be untouched and no ``.tmp`` may be left behind for the
    # next reader to trip over.
    with pytest.raises(TypeError):
        journal.atomic_write_text(path, object())  # type: ignore[arg-type]

    assert path.read_text(encoding="utf-8") == "original"
    assert [p.name for p in tmp_path.iterdir()] == ["snapshot.json"]


def test_append_creates_parent_directories_that_do_not_exist(tmp_path: Path) -> None:
    """First run of a fresh checkout: ``data/`` is gitignored, so it is never
    there, and the first fill must not be the thing that discovers that."""
    path = tmp_path / "data" / "nested" / "ledger.jsonl"
    journal.append(path, {"row_id": "a"})
    assert path.is_file()


def test_every_row_is_one_physical_line_even_with_a_newline_in_the_text(
    tmp_path: Path,
) -> None:
    """The trailing newline is what makes the *next* append a clean row rather
    than a corruption of this one."""
    lg = ledger(tmp_path)
    lg.append_note(note_id="n1", ts=NOW, text="a reason\nwith a newline in it")

    assert len(lines(lg.path)) == 1
    assert lg.path.read_text(encoding="utf-8").endswith("\n")
    assert lg.rows()[0]["payload"]["text"] == "a reason\nwith a newline in it"


# ---------------------------------------------------------------------------
# UTF-8
# ---------------------------------------------------------------------------


def test_non_ascii_round_trips_through_the_ledger_as_itself(tmp_path: Path) -> None:
    """Windows defaults to cp1252 for files and for redirected stdio, and this
    codebase has already been bitten: ``memetrader status | tail`` died with
    UnicodeEncodeError on a ``↓`` while the same command in a terminal was
    fine. Every open here names ``encoding="utf-8"``, and ``ensure_ascii=False``
    keeps the characters readable rather than turning them into ``\\u2193``."""
    lg = ledger(tmp_path)
    text = "liquidity ↓ 42% — 買い圧力 strong, price €0.000021 ✓"
    lg.append_note(note_id="n1", ts=NOW, text=text)

    (line,) = lines(lg.path)
    assert "↓" in line
    assert "—" in line
    assert "\\u2193" not in line
    assert lg.rows()[0]["payload"]["text"] == text

    # And the file really is UTF-8 on disk, not the platform default.
    raw = lg.path.read_bytes()
    assert "↓".encode() in raw
    with pytest.raises(UnicodeDecodeError):
        raw.decode("ascii")


def test_a_non_ascii_symbol_survives_an_intent_round_trip(tmp_path: Path) -> None:
    lg = ledger(tmp_path)
    order = intent(symbol="ĐOGE↑", reason="momentum ≥ 2 sigma")
    lg.append_intent(order)

    (row,) = lg.rows()
    assert row["payload"]["symbol"] == "ĐOGE↑"
    assert row["payload"]["reason"] == "momentum ≥ 2 sigma"


# ---------------------------------------------------------------------------
# Serialization — the decisions that regress silently
# ---------------------------------------------------------------------------


def test_a_side_reaches_disk_as_buy_and_not_as_side_dot_buy(tmp_path: Path) -> None:
    """``Side.BUY`` compares equal to ``"BUY"`` either way, so only the written
    bytes can tell you whether ``json.dumps`` used the value or the repr. A
    plain ``Enum`` here would make ``dumps`` raise — and the ledger would stop
    being written at the exact moment a trade happened."""
    lg = ledger(tmp_path)
    lg.append_intent(intent(side=Side.SELL))

    (line,) = lines(lg.path)
    assert '"side": "SELL"' in line
    assert "Side.SELL" not in line


def test_an_order_state_reaches_disk_as_its_value(tmp_path: Path) -> None:
    lg = ledger(tmp_path)
    lg.append_fill(fill(state=OrderState.FAILED))

    (row,) = lg.rows()
    assert row["state"] == "failed"
    assert row["payload"]["state"] == "failed"


def test_infinity_becomes_the_string_inf_rather_than_null(tmp_path: Path) -> None:
    """``TxnCounts.ratio`` returns ``inf`` for a pool with buys and zero sells —
    the strongest buy-pressure reading the feed can produce, and exactly the
    number you want to find six hours later. JSON has no literal for it, so the
    choice is a visible ``"inf"`` or a silent ``null``, and a ``null`` would be
    indistinguishable from "we did not measure"."""
    assert journal.to_jsonable(float("inf")) == "inf"
    assert journal.to_jsonable(float("-inf")) == "-inf"
    assert journal.to_jsonable(float("nan")) is None
    assert math.isinf(TxnCounts(buys=41, sells=0).ratio)

    lg = ledger(tmp_path)
    lg.append_note(note_id="n", ts=NOW, text="x")
    journal.append(lg.path, {"ratio": TxnCounts(buys=41, sells=0).ratio})

    line = lines(lg.path)[-1]
    assert '"ratio": "inf"' in line
    assert "Infinity" not in line
    assert "NaN" not in line


def test_no_ledger_line_ever_contains_a_non_standard_json_constant(
    tmp_path: Path,
) -> None:
    """``json.dumps`` emits bare ``NaN`` and ``Infinity`` quite happily, and
    they are not JSON. Dropping the float handling would keep the ledger
    writable and stop it being readable by anything that is not Python — the
    kind of breakage you discover long after the trades it describes."""

    def reject(token: str) -> object:
        raise AssertionError(f"non-standard JSON constant in the ledger: {token}")

    path = tmp_path / "ledger.jsonl"
    journal.append(
        path, {"ratio": float("inf"), "drawdown": float("nan"), "edge": float("-inf")}
    )

    (line,) = lines(path)
    assert json.loads(line, parse_constant=reject) == {
        "ratio": "inf",
        "drawdown": None,
        "edge": "-inf",
    }


# ---------------------------------------------------------------------------
# read / tail — the legacy surface report.py still uses
# ---------------------------------------------------------------------------


def test_read_of_a_missing_file_yields_nothing(tmp_path: Path) -> None:
    """`report` and the prompt history both call this before any trade has ever
    been made, so "no file yet" is a normal state and not an error."""
    assert list(journal.read(tmp_path / "never" / "written.jsonl")) == []


def test_read_of_a_directory_yields_nothing(tmp_path: Path) -> None:
    assert list(journal.read(tmp_path)) == []


def test_read_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "l.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert [r["a"] for r in journal.read(path)] == [1, 2]


def test_tail_returns_the_last_n_rows_oldest_first(tmp_path: Path) -> None:
    path = tmp_path / "l.jsonl"
    for i in range(10):
        journal.append(path, {"i": i})
    assert [r["i"] for r in journal.tail(path, 3)] == [7, 8, 9]


def test_tail_of_zero_returns_nothing(tmp_path: Path) -> None:
    """``rows[-0:]`` is the whole list, so the guard is the only thing standing
    between "show me no history" and "show me all of it". Config can set
    ``prompt.decision_history = 0``, and that has to mean zero."""
    path = tmp_path / "l.jsonl"
    journal.append(path, {"i": 0})
    assert journal.tail(path, 0) == []


def test_tail_ignores_a_torn_trailing_line(tmp_path: Path) -> None:
    path = tmp_path / "l.jsonl"
    for i in range(4):
        journal.append(path, {"i": i})
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"i": 4, "sym')

    assert [r["i"] for r in journal.tail(path, 3)] == [1, 2, 3]


def test_a_pre_audit_row_without_ids_is_still_readable(tmp_path: Path) -> None:
    """Rows written before this file existed carry no ``schema_version``, which
    makes them version 1 by definition. They are readable and deliberately not
    deduplicated — inventing an identity for them would make two genuinely
    distinct old rows collide."""
    path = tmp_path / "trades.jsonl"
    path.write_text('{"symbol": "BONK", "filled_usd": 100.0}\n', encoding="utf-8")

    (row,) = journal.read(path)
    assert row["symbol"] == "BONK"
    assert "schema_version" not in row
    assert journal.Ledger(path, run_id=RUN).open_intents() == ()
