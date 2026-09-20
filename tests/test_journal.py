"""Serialization, and the two append-only ledgers.

``journal.py`` is the only place that knows how to turn this project's
dataclasses into a line of JSON. Every other module hands it a ``Fill`` or a
``DecisionRecord`` and never thinks about encoding again — which makes this the
only file where the encoding decisions can be pinned at all.

Two of them are load-bearing and both regress silently:

* ``Side`` has to land in the log as ``"BUY"``, not ``"Side.BUY"``. That works
  only because ``Side`` is a ``StrEnum``; a plain ``Enum`` would make
  ``json.dumps`` raise, and the ledger would stop being written at the exact
  moment a trade happened.
* ``float('inf')`` has to land as the string ``"inf"``. ``TxnCounts.ratio``
  returns it for a pool with buys and no sells, JSON has no literal for it, and
  the alternative encodings are all worse.

The rest is crash behaviour. Both logs are append-only precisely so a losing
trade can be explained after the fact, which means ``read`` has to survive a
process that died halfway through a write, and ``append`` has to work on the
first run of a fresh checkout where ``data/`` does not exist yet.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest

from memetrader import journal
from memetrader.types import Action, Fill, RiskVerdict, Side, TradeDecision, TxnCounts

NOW = 1_700_000_000.0


# ---------------------------------------------------------------------------
# Shapes
#
# The structural tests below use dataclasses declared here rather than borrowed
# from types.py. What is under test is how ``to_jsonable`` treats a *shape* —
# frozen, slotted, nested — and pinning that to whatever fields a domain type
# happens to carry this week would make these tests fail for unrelated reasons.
# The domain types appear further down, where the domain is the point.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Leg:
    venue: str
    fee_pct: float


@dataclasses.dataclass(frozen=True, slots=True)
class Route:
    symbol: str
    leg: Leg
    labels: tuple[str, ...]


def fill(**kw: object) -> Fill:
    base: dict[str, object] = {
        "ts": NOW,
        "symbol": "BONK",
        "side": Side.BUY,
        "requested_usd": 100.0,
        "filled_usd": 100.0,
        "price_usd": 0.00002,
        "quantity": 5_000_000.0,
        "price_impact_pct": 0.4,
        "pool_fee_usd": 0.25,
        "gas_usd": 0.21,
    }
    return Fill(**{**base, **kw})  # type: ignore[arg-type]


def written(path: Path) -> list[str]:
    """The file exactly as it sits on disk, one entry per physical line."""
    return path.read_text(encoding="utf-8").splitlines()


def _reject_constant(token: str) -> object:
    """``json.loads`` accepts ``NaN`` and ``Infinity`` by default; most other
    parsers do not, and neither does the JSON spec. Passing this as
    ``parse_constant`` turns "we wrote a token no one else can read" from a
    silent fact about the file into a failing test."""
    raise AssertionError(f"non-standard JSON constant in the ledger: {token}")


def strict_loads(line: str) -> dict:
    return json.loads(line, parse_constant=_reject_constant)


# ---------------------------------------------------------------------------
# to_jsonable — dataclasses
# ---------------------------------------------------------------------------


def test_a_frozen_slotted_dataclass_becomes_a_plain_dict() -> None:
    assert journal.to_jsonable(Leg(venue="Raydium", fee_pct=0.25)) == {
        "venue": "Raydium",
        "fee_pct": 0.25,
    }


def test_a_dataclass_nested_in_another_is_converted_all_the_way_down() -> None:
    """``asdict`` would do this too, but it also deep-copies and chokes on the
    pydantic models that ``DecisionRecord`` carries — hence the hand-rolled
    recursion, which has to be checked to actually recurse."""
    route = Route(symbol="BONK", leg=Leg(venue="Orca", fee_pct=0.3), labels=("Orca",))

    assert journal.to_jsonable(route) == {
        "symbol": "BONK",
        "leg": {"venue": "Orca", "fee_pct": 0.3},
        "labels": ["Orca"],
    }


def test_a_dataclass_class_object_is_left_alone() -> None:
    """``is_dataclass`` answers True for the class as well as for instances, so
    the ``not isinstance(obj, type)`` guard is what stops a stray class
    reference in a record from being rendered as an empty-ish dict."""
    assert journal.to_jsonable(Leg) is Leg


def test_a_real_fill_round_trips_through_the_trades_log(tmp_path: Path) -> None:
    """One row of ``trades.jsonl``, end to end — the actual job of this module."""
    path = tmp_path / "trades.jsonl"
    journal.append(path, fill(realized_pnl_usd=-12.5, note="stop-loss"))

    (row,) = journal.read(path)
    assert row["symbol"] == "BONK"
    assert row["side"] == "BUY"
    assert row["price_usd"] == pytest.approx(0.00002)
    assert row["realized_pnl_usd"] == pytest.approx(-12.5)
    assert row["failed"] is False
    assert row["note"] == "stop-loss"


# ---------------------------------------------------------------------------
# to_jsonable — pydantic
# ---------------------------------------------------------------------------


def test_a_pydantic_model_is_dumped_by_pydantic() -> None:
    action = Action(
        action="BUY", symbol="WIF", size_usd=250.0, confidence=0.8, reasoning="flow"
    )
    assert journal.to_jsonable(action) == {
        "action": "BUY",
        "symbol": "WIF",
        "size_usd": 250.0,
        "confidence": 0.8,
        "reasoning": "flow",
    }


def test_a_pydantic_model_nests_its_own_children() -> None:
    """``model_dump`` is called and then trusted — the recursion does not walk
    back into a model's contents, so a model holding models has to come out
    whole on pydantic's own terms."""
    decision = TradeDecision(
        market_read="chop",
        actions=[
            Action(
                action="HOLD", symbol="BONK", size_usd=0.0, confidence=0.4, reasoning="x"
            )
        ],
    )
    dumped = journal.to_jsonable(decision)

    assert dumped["market_read"] == "chop"
    assert dumped["actions"] == [
        {
            "action": "HOLD",
            "symbol": "BONK",
            "size_usd": 0.0,
            "confidence": 0.4,
            "reasoning": "x",
        }
    ]


def test_pydantic_models_inside_a_dataclass_are_reached(tmp_path: Path) -> None:
    """The ``DecisionRecord`` shape: a dataclass whose fields are tuples of
    pydantic models. Both conversions have to fire, in that order."""

    @dataclasses.dataclass(frozen=True, slots=True)
    class Record:
        actions: tuple[Action, ...]
        verdicts: tuple[RiskVerdict, ...]

    record = Record(
        actions=(
            Action(
                action="SELL", symbol="WIF", size_usd=50.0, confidence=0.6, reasoning="y"
            ),
        ),
        verdicts=(
            RiskVerdict(
                approved=False,
                approved_usd=0.0,
                symbol="WIF",
                rule="min_liquidity",
                reason="pool is draining",
                notes=("clamped once",),
            ),
        ),
    )

    path = tmp_path / "decisions.jsonl"
    journal.append(path, record)
    (row,) = journal.read(path)

    assert row["actions"] == [
        {
            "action": "SELL",
            "symbol": "WIF",
            "size_usd": 50.0,
            "confidence": 0.6,
            "reasoning": "y",
        }
    ]
    assert row["verdicts"][0]["rule"] == "min_liquidity"
    assert row["verdicts"][0]["notes"] == ["clamped once"]


# ---------------------------------------------------------------------------
# to_jsonable — StrEnum
#
# The decision log is read back by `loop.recent_decisions` and fed to the model,
# and it is read by a human explaining a bad trade. Both want "BUY".
# ---------------------------------------------------------------------------


def test_a_side_serializes_as_its_plain_value() -> None:
    assert journal.to_jsonable(Side.BUY) == "BUY"
    assert journal.to_jsonable(Side.SELL) == "SELL"


def test_a_side_reaches_disk_as_buy_and_not_as_side_dot_buy(tmp_path: Path) -> None:
    """The assertion that matters is on the bytes, not on the Python object.
    ``Side.BUY`` compares equal to ``"BUY"`` either way, so only the written
    line can tell you whether ``json.dumps`` used the value or the repr."""
    path = tmp_path / "trades.jsonl"
    journal.append(path, fill(side=Side.SELL))

    (line,) = written(path)
    assert '"side": "SELL"' in line
    assert "Side.SELL" not in line
    assert strict_loads(line)["side"] == "SELL"


# ---------------------------------------------------------------------------
# to_jsonable — containers
# ---------------------------------------------------------------------------


def test_tuples_become_json_lists() -> None:
    assert journal.to_jsonable(("Raydium", "Orca")) == ["Raydium", "Orca"]


def test_sets_become_json_lists() -> None:
    """JSON has no set. Order is whatever the set felt like, so only membership
    is a real claim to make here."""
    out = journal.to_jsonable({"Raydium", "Orca"})
    assert isinstance(out, list)
    assert sorted(out) == ["Orca", "Raydium"]


def test_containers_are_converted_element_by_element() -> None:
    assert journal.to_jsonable((Leg(venue="Meteora", fee_pct=0.2),)) == [
        {"venue": "Meteora", "fee_pct": 0.2}
    ]


def test_dict_keys_are_coerced_to_strings() -> None:
    """JSON object keys are strings. ``marks`` and ``position_values_usd`` are
    keyed by symbol so this rarely bites, but a numeric key would otherwise
    round-trip back as a string and quietly stop matching its lookups."""
    assert journal.to_jsonable({1: "a", "b": 2}) == {"1": "a", "b": 2}


# ---------------------------------------------------------------------------
# to_jsonable — the floats JSON cannot hold
# ---------------------------------------------------------------------------


def test_nan_becomes_null() -> None:
    """NaN means the arithmetic did not produce an answer, and ``null`` is the
    honest JSON for that."""
    assert journal.to_jsonable(float("nan")) is None


def test_infinity_becomes_the_string_inf_rather_than_null() -> None:
    """This is the case the module comment is about, and it is a deliberate
    inconsistency with NaN above.

    ``TxnCounts.ratio`` returns ``float('inf')`` for a pool with buys and zero
    sells. That is not a glitch or a divide-by-zero to be papered over — it is
    the strongest buy-pressure reading the feed can produce, and it is exactly
    the number you want to find in the ledger six hours later when you are
    working out why the model bought. JSON has no literal for it, so the choice
    is between a visible ``"inf"`` and a silent ``null`` — and a ``null`` here
    would be indistinguishable from "we did not measure", which is the same
    mistake the sentiment brief refuses to make with ``mention_velocity_1h``.
    """
    assert journal.to_jsonable(float("inf")) == "inf"
    assert journal.to_jsonable(float("-inf")) == "-inf"


def test_a_pool_with_no_sells_reaches_the_ledger_as_inf(tmp_path: Path) -> None:
    """The same thing again, but arrived at the way it actually happens."""
    counts = TxnCounts(buys=41, sells=0)
    assert counts.ratio == float("inf")

    path = tmp_path / "trades.jsonl"
    journal.append(path, {"symbol": "BONK", "buy_sell_ratio_m5": counts.ratio})

    (line,) = written(path)
    assert '"buy_sell_ratio_m5": "inf"' in line
    assert strict_loads(line)["buy_sell_ratio_m5"] == "inf"


def test_a_ledger_line_never_contains_a_non_standard_json_constant(
    tmp_path: Path,
) -> None:
    """``json.dumps`` emits bare ``NaN`` and ``Infinity`` tokens quite happily,
    and they are not JSON. If the float handling in ``to_jsonable`` is ever
    dropped, the ledger keeps being written and only stops being readable by
    anything that is not Python — which is the kind of breakage you discover
    long after the trades it describes."""
    path = tmp_path / "trades.jsonl"
    journal.append(
        path,
        {"ratio": float("inf"), "drawdown": float("nan"), "edge": float("-inf")},
    )

    (line,) = written(path)
    assert "NaN" not in line
    assert "Infinity" not in line
    assert strict_loads(line) == {"ratio": "inf", "drawdown": None, "edge": "-inf"}


def test_ordinary_numbers_are_left_exactly_as_they_are() -> None:
    assert journal.to_jsonable(0.00002) == 0.00002
    assert journal.to_jsonable(-12.5) == -12.5
    assert journal.to_jsonable(0.0) == 0.0
    assert journal.to_jsonable(7) == 7
    assert journal.to_jsonable(True) is True
    assert journal.to_jsonable(None) is None
    assert journal.to_jsonable("BONK") == "BONK"


def test_nan_and_inf_are_caught_inside_a_nested_record(tmp_path: Path) -> None:
    """The float branch is the last one checked, so it only ever runs on values
    the recursion has already reached. A ratio sitting two levels down in a
    record is the realistic case."""
    path = tmp_path / "decisions.jsonl"
    journal.append(path, {"coins": {"BONK": {"ratio": TxnCounts(buys=9, sells=0).ratio}}})

    (row,) = journal.read(path)
    assert row["coins"]["BONK"]["ratio"] == "inf"


# ---------------------------------------------------------------------------
# append
# ---------------------------------------------------------------------------


def test_append_creates_parent_directories_that_do_not_exist(tmp_path: Path) -> None:
    """First run of a fresh checkout: ``data/`` is gitignored, so it is never
    there, and the first fill must not be the thing that discovers that."""
    path = tmp_path / "data" / "nested" / "trades.jsonl"
    assert not path.parent.exists()

    journal.append(path, fill())

    assert path.is_file()
    assert len(written(path)) == 1


def test_append_leaves_every_earlier_row_byte_for_byte_alone(tmp_path: Path) -> None:
    """Append-only is the whole premise of the ledger: a log you can rewrite is
    a log you cannot learn from."""
    path = tmp_path / "trades.jsonl"
    journal.append(path, fill(symbol="BONK"))
    first = path.read_text(encoding="utf-8")

    journal.append(path, fill(symbol="WIF"))
    journal.append(path, fill(symbol="POPCAT"))

    after = path.read_text(encoding="utf-8")
    assert after.startswith(first)
    assert [r["symbol"] for r in journal.read(path)] == ["BONK", "WIF", "POPCAT"]


def test_every_row_is_one_line_and_the_file_ends_with_a_newline(
    tmp_path: Path,
) -> None:
    """The trailing newline is what makes the *next* append a clean row rather
    than a corruption of this one."""
    path = tmp_path / "trades.jsonl"
    journal.append(path, fill(note="a reason\nwith a newline in it"))

    assert path.read_text(encoding="utf-8").endswith("\n")
    assert len(written(path)) == 1


def test_non_ascii_is_written_as_itself_rather_than_an_escape(tmp_path: Path) -> None:
    """``ensure_ascii=False``. The model's reasoning is full of em dashes and
    arrows, and a log you read with your eyes should not be full of ``\\u2014``."""
    path = tmp_path / "decisions.jsonl"
    journal.append(path, {"reasoning": "flow is one-sided — buys only"})

    (line,) = written(path)
    assert "—" in line
    assert "\\u2014" not in line
    assert strict_loads(line)["reasoning"] == "flow is one-sided — buys only"


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


def test_read_skips_a_corrupt_trailing_line_and_still_yields_every_good_row(
    tmp_path: Path,
) -> None:
    """The crash-mid-write case, which is the only corruption an append-only log
    can actually produce: the process died with a partial line on disk and no
    trailing newline. Everything before it is still a true record of what
    happened and must survive."""
    path = tmp_path / "trades.jsonl"
    path.write_text(
        '{"symbol": "BONK", "filled_usd": 100.0}\n'
        '{"symbol": "WIF", "filled_usd": 50.0}\n'
        '{"symbol": "POPCAT", "filled_u',
        encoding="utf-8",
    )

    rows = list(journal.read(path))
    assert [r["symbol"] for r in rows] == ["BONK", "WIF"]


def test_read_skips_a_corrupt_line_in_the_middle_too(tmp_path: Path) -> None:
    """Rarer, but the handler is per-line rather than "stop at the first bad
    one", so a row after the damage is still returned."""
    path = tmp_path / "trades.jsonl"
    path.write_text(
        '{"symbol": "BONK"}\nnot json at all\n{"symbol": "WIF"}\n', encoding="utf-8"
    )

    assert [r["symbol"] for r in journal.read(path)] == ["BONK", "WIF"]


def test_read_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "trades.jsonl"
    path.write_text('{"symbol": "BONK"}\n\n   \n{"symbol": "WIF"}\n', encoding="utf-8")

    assert [r["symbol"] for r in journal.read(path)] == ["BONK", "WIF"]


def test_read_of_a_missing_file_yields_nothing(tmp_path: Path) -> None:
    """`report` and the prompt's decision history both call this before any
    trade has ever been made, so "no file yet" is a normal state and not an
    error."""
    assert list(journal.read(tmp_path / "never" / "written.jsonl")) == []


def test_read_of_an_empty_file_yields_nothing(tmp_path: Path) -> None:
    path = tmp_path / "trades.jsonl"
    path.write_text("", encoding="utf-8")
    assert list(journal.read(path)) == []


def test_read_of_a_directory_yields_nothing(tmp_path: Path) -> None:
    """The guard is ``is_file``, not ``exists``, so a path that resolves to a
    directory is treated as "nothing to read" rather than raising IsADirectory
    from inside a generator."""
    assert list(journal.read(tmp_path)) == []


def test_read_is_lazy(tmp_path: Path) -> None:
    """It is a generator, so a missing file must not raise until it is iterated
    — ``tail`` relies on being able to build one unconditionally."""
    rows = journal.read(tmp_path / "absent.jsonl")
    assert list(rows) == []


# ---------------------------------------------------------------------------
# tail
# ---------------------------------------------------------------------------


def ledger(tmp_path: Path, count: int) -> Path:
    path = tmp_path / "trades.jsonl"
    for i in range(count):
        journal.append(path, {"i": i})
    return path


def test_tail_returns_the_last_n_rows_in_order(tmp_path: Path) -> None:
    path = ledger(tmp_path, 10)
    assert [r["i"] for r in journal.tail(path, 3)] == [7, 8, 9]


def test_tail_of_zero_returns_nothing(tmp_path: Path) -> None:
    """``rows[-0:]`` is the whole list, so the ``n > 0`` guard is the only thing
    standing between "show me no history" and "show me all of it". Config can
    set ``prompt.decision_history = 0``, and that has to mean zero."""
    path = ledger(tmp_path, 10)
    assert journal.tail(path, 0) == []


def test_tail_larger_than_the_file_returns_everything(tmp_path: Path) -> None:
    path = ledger(tmp_path, 3)
    assert [r["i"] for r in journal.tail(path, 100)] == [0, 1, 2]


def test_tail_of_exactly_the_file_length_returns_everything(tmp_path: Path) -> None:
    path = ledger(tmp_path, 3)
    assert [r["i"] for r in journal.tail(path, 3)] == [0, 1, 2]


def test_tail_of_one_returns_the_most_recent_row(tmp_path: Path) -> None:
    path = ledger(tmp_path, 5)
    assert [r["i"] for r in journal.tail(path, 1)] == [4]


def test_tail_of_a_missing_file_returns_an_empty_list(tmp_path: Path) -> None:
    assert journal.tail(tmp_path / "absent.jsonl", 5) == []


def test_tail_ignores_a_corrupt_trailing_line(tmp_path: Path) -> None:
    """``tail`` is what feeds the model its own recent history, so a crash
    mid-write must cost the last decision, not the memory of all of them."""
    path = ledger(tmp_path, 4)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"i": 4, "sym')

    assert [r["i"] for r in journal.tail(path, 3)] == [1, 2, 3]


def test_tail_returns_a_list_not_a_generator(tmp_path: Path) -> None:
    """Callers index it and take its length; ``read`` is the lazy one."""
    path = ledger(tmp_path, 2)
    rows = journal.tail(path, 2)
    assert isinstance(rows, list)
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# The round trip, on the numbers that motivated all of this
# ---------------------------------------------------------------------------


def test_a_record_full_of_unrepresentable_floats_survives_the_whole_trip(
    tmp_path: Path,
) -> None:
    path = tmp_path / "decisions.jsonl"
    journal.append(
        path,
        {
            "ratios": (
                TxnCounts(buys=9, sells=0).ratio,
                TxnCounts(buys=0, sells=0).ratio,
                TxnCounts(buys=8, sells=4).ratio,
            ),
            "velocity": float("nan"),
        },
    )

    (row,) = journal.tail(path, 1)
    assert row["ratios"] == ["inf", 1.0, 2.0]
    assert row["velocity"] is None
    # And the source values really were the things JSON cannot hold.
    assert math.isinf(TxnCounts(buys=9, sells=0).ratio)
