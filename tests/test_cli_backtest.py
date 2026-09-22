"""Tests for ``memetrader.cli_backtest`` — the ``run``/``resume`` CLI.

All fixtures are synthetic and written under ``tmp_path``: real bar files
built with ``backfill.write_series``/``RawBar`` (mirroring
``tests/histdata/test_catalog.py``'s pattern), a hand-written minimal
universe TOML, and a hand-written minimal backtest config TOML. Nothing here
reads or writes ``history/``, ``universe/``, or ``runs/`` in the repo itself —
every path is rooted at ``tmp_path``, per the "never write runtime state into
the repo" rule.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from memetrader.backfill import RawBar, write_series
from memetrader.cli_backtest import app
from memetrader.types import NON_EXECUTABLE_NOTICE

runner = CliRunner()

POOL = "Poo1AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
MINT = "MintAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
SYMBOL = "TEST"

# Bar grid: hourly/5m bars around this epoch second (a fixed, arbitrary
# Monday — the actual date is irrelevant, only the spacing matters).
BASE_TS = 1_717_200_000.0
HOUR = 3600.0
FIVE_MIN = 300.0


def _make_hourly_bars(start_ts: float, count: int) -> list[RawBar]:
    return [
        RawBar(
            ts=start_ts + i * HOUR,
            open=1.0 + i * 0.001,
            high=1.01 + i * 0.001,
            low=0.99 + i * 0.001,
            close=1.0 + i * 0.001,
            volume=1_000.0,
        )
        for i in range(count)
    ]


def _make_five_min_bars(start_ts: float, count: int) -> list[RawBar]:
    return [
        RawBar(
            ts=start_ts + i * FIVE_MIN,
            open=1.0 + i * 0.0002,
            high=1.005 + i * 0.0002,
            low=0.998 + i * 0.0002,
            close=1.0 + i * 0.0002,
            volume=200.0,
        )
        for i in range(count)
    ]


def _write_history(root: Path, *, hours: int) -> None:
    # Bars start exactly at BASE_TS (== config.start_ts for every fixture
    # built from this): BacktestConfig.start_ts's own contract ("the first
    # event yielded by any loader must have available_time >= start_ts")
    # means any bar dated before start_ts would simply be dropped by
    # _bounded — there is no such thing as "warm-up" data a contract-abiding
    # stream can hand the replay before its own start. (cli_backtest.py's
    # _decision_ticks starting one tick after start_ts, not at it, is what
    # gives the strategy a real bar to look at on its first opportunity.)
    hourly_count = hours + 2
    five_min_count = (hours + 2) * 12
    for tf, bars in (
        ("1h", _make_hourly_bars(BASE_TS, hourly_count)),
        ("5m", _make_five_min_bars(BASE_TS, five_min_count)),
    ):
        path = root / POOL / f"{tf}.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_series(path, bars)


def _write_universe(path: Path, *, pool_created: str = "2024-01-01") -> None:
    # eligible_from = pool_created + 14 days (UniverseCatalog's default
    # min_pool_age_days), which is comfortably before BASE_TS for this date,
    # so the single coin is eligible for the whole replay window.
    path.write_text(
        f"""
[meta]
resolved_at = "2024-06-01T00:00:00Z"

[[coins]]
symbol = "{SYMBOL}"
mint = "{MINT}"
pool = "{POOL}"
dex = "raydium"
quote = "USDC"
pool_created = "{pool_created}"
liquidity_usd = 500000.0
fdv_usd = 2000000.0
liquidity_fdv_ratio = 0.25
""",
        encoding="utf-8",
    )


def _write_config(
    path: Path,
    *,
    start_ts: float,
    end_ts: float,
    label: str = "cli_test",
) -> None:
    path.write_text(
        f"""
[run]
label = "{label}"
run_id_prefix = "clitest"

[date_range]
start_ts = {start_ts}
end_ts   = {end_ts}

[data]
universe_version = "v1"
timeframes = ["5m", "1h"]

[cadence]
# fast_tick_seconds is 450, not the "natural" 300 (== the 5m bar interval):
# runner._engine_config sets EngineConfig.settlement_latency_seconds to
# max(1.0, fast_tick_seconds) with no publication-delay term added
# (runner.py:135 at the time of writing — see the final report). Since
# BarExecutionModel.fill() (execution/fill_models.py) needs a bar with
# ts >= decided_at to have actually been *ingested* — i.e. its
# available_time = ts + interval + publication_delay_seconds must be <=
# settlement time — a settlement latency of exactly one bar interval
# (300s) is 60s short of the 360s (interval + 60s publication delay) a
# fill actually needs, and every entry fails with "route disappeared...
# no bar opened". 450 clears that 360s floor with margin, purely so this
# fixture exercises a real fill instead of a permanent, config-shape-
# independent failure.
fast_tick_seconds     = 450
decision_tick_seconds = 900

[portfolio]
starting_cash_usd = 1000.0

[execution]
publication_delay_seconds = 60.0
execution_model  = "ohlcv_midpoint"
execution_mode   = "paper"
fidelity_tier    = "tier_0_ohlcv"

[execution.latency]
# mu/sigma here have no effect on fill timing: EngineConfig.settlement_
# latency_seconds (runner._engine_config) is derived from
# [cadence] fast_tick_seconds alone, not from this lognormal draw (see
# EngineConfig's own docstring: settlement latency is "a deterministic
# constant rather than a drawn ... sample"). See fast_tick_seconds' comment
# above for what actually controls whether a fill can succeed.
mu              = -1.5
sigma           = 0.4
failed_tx_rate  = 0.0

[folds]
train_seconds   = 7_776_000
val_seconds     = 2_592_000
test_seconds    = 2_592_000
roll_seconds    = 2_592_000
embargo_seconds = 172_800
n_outer_folds   = 3
holdout_seconds  = 3_888_000
holdout_min_pct  = 0.20

[holdout]
opened = false

[seeds]
latency_draw  = 42
failed_tx     = 43
strategy_rng  = 44

[cost_stress]
multipliers = [1]
""",
        encoding="utf-8",
    )


def _build_fixture(tmp_path: Path, *, hours: int = 72) -> dict[str, Path]:
    """A complete, self-contained set of inputs for one CLI invocation."""
    history_root = tmp_path / "history"
    universe_path = tmp_path / "universe.toml"
    config_path = tmp_path / "config.toml"
    output_dir = tmp_path / "out"

    start_ts = BASE_TS
    end_ts = BASE_TS + hours * HOUR

    _write_history(history_root, hours=hours)
    _write_universe(universe_path)
    _write_config(config_path, start_ts=start_ts, end_ts=end_ts)

    return {
        "history_root": history_root,
        "universe_path": universe_path,
        "config_path": config_path,
        "output_dir": output_dir,
    }


def _run_args(fixture: dict[str, Path], *extra: str) -> list[str]:
    return [
        "run",
        str(fixture["config_path"]),
        "--history-root",
        str(fixture["history_root"]),
        "--universe",
        str(fixture["universe_path"]),
        "--output-dir",
        str(fixture["output_dir"]),
        *extra,
    ]


# ---------------------------------------------------------------------------
# `run`
# ---------------------------------------------------------------------------


def test_run_produces_manifest_ledger_metrics(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    result = runner.invoke(app, _run_args(fixture))

    assert result.exit_code == 0, result.output
    assert fixture["output_dir"].exists()

    # The three artifact paths printed by the CLI must actually exist and be
    # well-formed JSON — a CLI that prints a path to a file it never wrote
    # would still pass a weaker "exit code 0" assertion, so check the files.
    manifest_files = list(fixture["output_dir"].rglob("manifest.json"))
    ledger_files = list(fixture["output_dir"].rglob("ledger.json"))
    metrics_files = list(fixture["output_dir"].rglob("metrics.json"))
    assert len(manifest_files) == 1, result.output
    assert len(ledger_files) == 1, result.output
    assert len(metrics_files) == 1, result.output

    manifest = json.loads(manifest_files[0].read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "cli_test"
    ledger = json.loads(ledger_files[0].read_text(encoding="utf-8"))
    assert ledger  # non-empty structure, not a stub
    metrics = json.loads(metrics_files[0].read_text(encoding="utf-8"))
    assert metrics


def test_run_prints_non_executable_notice_for_tier0(tmp_path: Path) -> None:
    # baseline_price_volume.toml (and this fixture) are both fidelity_tier =
    # "tier_0_ohlcv" — a tier that does not permit a PnL claim (contracts §3),
    # so the CLI must print the exact notice text, not a paraphrase.
    fixture = _build_fixture(tmp_path)
    result = runner.invoke(app, _run_args(fixture))

    assert result.exit_code == 0, result.output
    assert NON_EXECUTABLE_NOTICE in result.output


def test_run_reports_decision_ticks_were_processed(tmp_path: Path) -> None:
    # decision_tick_seconds=900 over a 72h window is 288 ticks; the run must
    # actually process ticks, not just wire streams together and stop dry.
    fixture = _build_fixture(tmp_path, hours=72)
    result = runner.invoke(app, _run_args(fixture))

    assert result.exit_code == 0, result.output
    metrics = json.loads(
        next((fixture["output_dir"]).rglob("metrics.json")).read_text(encoding="utf-8")
    )
    # n_periods is the length of the recorded equity curve — one point per
    # tick the engine actually ran. A run that silently processed zero ticks
    # (e.g. an empty/misfiltered stream) would still produce a metrics file —
    # this is the check that would fail if _decision_ticks or the
    # [start_ts, end_ts) bound were wrong.
    assert metrics["n_periods"] > 0, metrics


def test_run_missing_config_file_is_a_clean_error(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    missing = tmp_path / "does_not_exist.toml"
    result = runner.invoke(
        app,
        [
            "run",
            str(missing),
            "--history-root",
            str(fixture["history_root"]),
            "--universe",
            str(fixture["universe_path"]),
            "--output-dir",
            str(fixture["output_dir"]),
        ],
    )
    assert result.exit_code != 0
    assert not fixture["output_dir"].exists()


def test_run_missing_universe_file_is_a_clean_error(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    missing_universe = tmp_path / "no_universe.toml"
    result = runner.invoke(
        app,
        [
            "run",
            str(fixture["config_path"]),
            "--history-root",
            str(fixture["history_root"]),
            "--universe",
            str(missing_universe),
            "--output-dir",
            str(fixture["output_dir"]),
        ],
    )
    assert result.exit_code != 0
    assert "universe" in result.output.lower()


def test_run_invalid_config_toml_is_a_clean_error(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    bad_config = tmp_path / "bad_config.toml"
    bad_config.write_text("this is not valid backtest config = [[[", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "run",
            str(bad_config),
            "--history-root",
            str(fixture["history_root"]),
            "--universe",
            str(fixture["universe_path"]),
            "--output-dir",
            str(fixture["output_dir"]),
        ],
    )
    assert result.exit_code != 0
    assert "config error" in result.output.lower()


# ---------------------------------------------------------------------------
# `resume`
# ---------------------------------------------------------------------------


def test_resume_from_final_snapshot_replays_almost_nothing_new(tmp_path: Path) -> None:
    """Resuming from the *last* snapshot of an already-completed run is the
    one case a synthetic fixture can reach without a way to interrupt
    ``runner.run`` mid-flight (its public API always runs to completion; the
    snapshot file is simply overwritten at every interval, so by the time
    ``run`` returns, ``--snapshot-to`` always holds the final tick's state —
    see ``runner.run``'s own docstring). It is still a meaningful check: a
    ``resume`` that ignored the snapshot's ``clock_now`` and rebuilt streams
    from ``config.start_ts`` would reprocess the entire history and end up
    with an ``n_periods`` close to the *first* run's — this is exactly what
    ``_build_runner_inputs``'s ``stream_start_ts`` bound exists to prevent.
    """
    fixture = _build_fixture(tmp_path, hours=72)
    snapshot_path = tmp_path / "snap.json"

    first = runner.invoke(
        app,
        _run_args(
            fixture,
            "--snapshot-to",
            str(snapshot_path),
            "--run-id",
            "resume-test-run",
        ),
    )
    assert first.exit_code == 0, first.output
    assert snapshot_path.exists()

    first_metrics = json.loads(
        next((fixture["output_dir"]).rglob("metrics.json")).read_text(encoding="utf-8")
    )
    first_periods = first_metrics["n_periods"]
    assert first_periods > 10  # sanity: the fixture actually ran many ticks

    resumed_output_dir = tmp_path / "out_resumed"
    second = runner.invoke(
        app,
        [
            "resume",
            str(fixture["config_path"]),
            str(snapshot_path),
            "--history-root",
            str(fixture["history_root"]),
            "--universe",
            str(fixture["universe_path"]),
            "--output-dir",
            str(resumed_output_dir),
            "--run-id",
            "resume-test-run",
        ],
    )
    assert second.exit_code == 0, second.output

    second_metrics = json.loads(
        next(resumed_output_dir.rglob("metrics.json")).read_text(encoding="utf-8")
    )
    # Resuming from the final snapshot must replay at most the handful of
    # events sharing the exact last available_time, never the whole history —
    # a resume bug that rebuilt streams from config.start_ts would blow this
    # assertion wide open (n_periods would be close to first_periods).
    assert second_metrics["n_periods"] < first_periods / 2


def test_resume_restores_ledger_cash_rather_than_starting_fresh(tmp_path: Path) -> None:
    """The resumed ledger's cash must be the *restored* cash from the
    snapshot's ledger_state, not the config's starting_cash_usd. A resume that
    silently built a brand-new ledger (ignoring ``resume_from`` entirely)
    would start from the untouched starting cash instead — this is the
    assertion that would catch that.
    """
    fixture = _build_fixture(tmp_path, hours=72)
    snapshot_path = tmp_path / "snap.json"

    first = runner.invoke(
        app,
        _run_args(
            fixture, "--snapshot-to", str(snapshot_path), "--run-id", "cash-test-run"
        ),
    )
    assert first.exit_code == 0, first.output
    first_ledger = json.loads(
        next((fixture["output_dir"]).rglob("ledger.json")).read_text(encoding="utf-8")
    )
    starting_cash_micro_usd = first_ledger["starting_cash_micro_usd"]
    first_cash = first_ledger["cash_micro_usd"]
    # BuyAndHoldStrategy spends cash buying the single eligible coin, so a
    # real run must have moved cash away from the untouched starting amount —
    # otherwise this fixture would not be exercising resume in a meaningful
    # way at all.
    assert first_cash != starting_cash_micro_usd

    resumed_output_dir = tmp_path / "out_resumed"
    second = runner.invoke(
        app,
        [
            "resume",
            str(fixture["config_path"]),
            str(snapshot_path),
            "--history-root",
            str(fixture["history_root"]),
            "--universe",
            str(fixture["universe_path"]),
            "--output-dir",
            str(resumed_output_dir),
            "--run-id",
            "cash-test-run",
        ],
    )
    assert second.exit_code == 0, second.output
    second_ledger = json.loads(
        next(resumed_output_dir.rglob("ledger.json")).read_text(encoding="utf-8")
    )
    assert second_ledger["starting_cash_micro_usd"] == starting_cash_micro_usd
    assert second_ledger["cash_micro_usd"] == first_cash


def test_resume_missing_snapshot_is_a_clean_error(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    missing_snapshot = tmp_path / "no_such_snapshot.json"
    result = runner.invoke(
        app,
        [
            "resume",
            str(fixture["config_path"]),
            str(missing_snapshot),
            "--history-root",
            str(fixture["history_root"]),
            "--universe",
            str(fixture["universe_path"]),
            "--output-dir",
            str(fixture["output_dir"]),
        ],
    )
    assert result.exit_code != 0
    assert "resume failed" in result.output.lower()
