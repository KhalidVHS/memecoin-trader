"""Every validation branch in ``config.py``, and the computed accessors.

The point of this file is the *error* paths. ``load()`` is the one place that
gets to say "no" — the module docstring's promise is that a bad config fails at
startup rather than surfacing as a weird fill three hours into a run, and an
unasserted validation branch is a promise nobody has checked.

Configs are written under ``tmp_path`` and kept minimal: each one carries the
smallest valid skeleton plus the single thing under test, so a failure names
the branch rather than a wall of TOML.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from memetrader.config import (
    ConfigError,
    DataConfig,
    ExecutionConfig,
    ModelConfig,
    SentimentConfig,
    find_project_root,
    load,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# 43 and 44 base58 characters — the two real mints from config.toml.
BONK_MINT = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
WIF_MINT = "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm"

ONE_COIN = f"""
    [[coins]]
    symbol = "BONK"
    mint = "{BONK_MINT}"
"""


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch):
    """``load()`` reads credentials from the environment, so a developer's real
    keys would otherwise decide what these tests assert."""
    for name in (
        "ANTHROPIC_API_KEY",
        "JUPITER_API_KEY",
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "REDDIT_USER_AGENT",
    ):
        monkeypatch.delenv(name, raising=False)


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")
    return path


def load_body(tmp_path: Path, body: str):
    return load(write_config(tmp_path, body))


# ---------------------------------------------------------------------------
# [[coins]]
# ---------------------------------------------------------------------------


def test_no_coins_at_all_is_fatal(tmp_path):
    with pytest.raises(ConfigError, match="at least one"):
        load_body(tmp_path, "[portfolio]\nstarting_cash_usd = 1000.0\n")


def test_empty_coins_array_is_fatal(tmp_path):
    with pytest.raises(ConfigError, match="at least one"):
        load_body(tmp_path, "coins = []\n")


def test_duplicate_symbol_is_fatal(tmp_path):
    with pytest.raises(ConfigError, match="duplicate coin symbol 'BONK'"):
        load_body(
            tmp_path,
            f"""
            [[coins]]
            symbol = "BONK"
            mint = "{BONK_MINT}"

            [[coins]]
            symbol = "BONK"
            mint = "{WIF_MINT}"
            """,
        )


def test_duplicate_is_detected_after_upper_casing(tmp_path):
    """Symbols are upper-cased on the way in, so 'bonk' and 'BONK' are the same
    coin — two entries for it would silently give the model two votes."""
    with pytest.raises(ConfigError, match="duplicate coin symbol 'BONK'"):
        load_body(
            tmp_path,
            f"""
            [[coins]]
            symbol = "bonk"
            mint = "{BONK_MINT}"

            [[coins]]
            symbol = "BONK"
            mint = "{WIF_MINT}"
            """,
        )


def test_symbol_is_upper_cased(tmp_path):
    cfg = load_body(tmp_path, f'[[coins]]\nsymbol = "wif"\nmint = "{WIF_MINT}"\n')
    assert cfg.symbols == ("WIF",)


def test_missing_symbol_or_mint_names_the_key(tmp_path):
    with pytest.raises(ConfigError, match=r"missing \[coins\] mint"):
        load_body(tmp_path, '[[coins]]\nsymbol = "BONK"\n')
    with pytest.raises(ConfigError, match=r"missing \[coins\] symbol"):
        load_body(tmp_path, f'[[coins]]\nmint = "{BONK_MINT}"\n')


@pytest.mark.parametrize("length", [0, 1, 31, 45, 64])
def test_implausible_mint_length_is_fatal(tmp_path, length):
    with pytest.raises(ConfigError, match="not a plausible Solana address"):
        load_body(tmp_path, f'[[coins]]\nsymbol = "BONK"\nmint = "{"A" * length}"\n')


@pytest.mark.parametrize("length", [32, 43, 44])
def test_mint_length_boundaries_are_inclusive(tmp_path, length):
    cfg = load_body(tmp_path, f'[[coins]]\nsymbol = "BONK"\nmint = "{"A" * length}"\n')
    assert len(cfg.coin("BONK").mint) == length


def test_mint_check_is_length_only_and_does_not_claim_to_be_base58(tmp_path):
    """Documenting the limit of a deliberately cheap check: it catches the
    common paste error before any HTTP call, and market.py verifying the mint
    live at startup is what catches the rest."""
    cfg = load_body(tmp_path, f'[[coins]]\nsymbol = "BONK"\nmint = "{"0" * 43}"\n')
    assert cfg.coin("BONK").mint == "0" * 43


def test_aliases_default_to_the_symbol(tmp_path):
    cfg = load_body(tmp_path, ONE_COIN)
    assert cfg.coin("BONK").aliases == ("BONK",)


# ---------------------------------------------------------------------------
# [model]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("effort", ["", "medium-high", "HIGHEST", "1", "extra-high"])
def test_invalid_effort_is_fatal(tmp_path, effort):
    with pytest.raises(ConfigError, match=r"effort must be one of"):
        load_body(tmp_path, f'{ONE_COIN}\n[model]\neffort = "{effort}"\n')


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_valid_effort_is_accepted_case_insensitively(tmp_path, effort):
    """The allowlist is exactly what anthropic 1.7.0's OutputConfigParam takes.
    A value the API would accept must not be rejected here as a typo."""
    cfg = load_body(tmp_path, f'{ONE_COIN}\n[model]\neffort = "{effort.upper()}"\n')
    assert cfg.model.effort == effort


def test_model_defaults_apply_when_the_table_is_absent(tmp_path):
    model = load_body(tmp_path, ONE_COIN).model
    assert model.name == "claude-opus-5"
    assert model.effort == "high"
    assert model.max_tokens == 8000
    assert (model.price_input_per_mtok, model.price_output_per_mtok) == (5.0, 25.0)
    assert model.price_cache_read_per_mtok == 0.5
    assert model.price_cache_write_per_mtok == 6.25


def test_cache_write_default_is_the_documented_multiple_of_input(tmp_path):
    """Cache writes bill at 1.25x base input. Keeping the two defaults in that
    ratio is the whole reason the number is 6.25 and not a round one."""
    model = load_body(tmp_path, ONE_COIN).model
    assert model.price_cache_write_per_mtok == pytest.approx(
        1.25 * model.price_input_per_mtok
    )


def test_prices_are_read_from_the_file(tmp_path):
    cfg = load_body(
        tmp_path,
        f"""
        {ONE_COIN}
        [model]
        price_input_per_mtok = 1.0
        price_output_per_mtok = 2.0
        price_cache_read_per_mtok = 0.25
        price_cache_write_per_mtok = 1.25
        """,
    )
    assert cfg.model.price_cache_write_per_mtok == 1.25


# ---------------------------------------------------------------------------
# ModelConfig.cost_usd — the one cost formula
# ---------------------------------------------------------------------------


def _model(**overrides) -> ModelConfig:
    fields = {
        "name": "claude-opus-5",
        "effort": "high",
        "max_tokens": 8000,
        "price_input_per_mtok": 5.0,
        "price_output_per_mtok": 25.0,
        "price_cache_read_per_mtok": 0.5,
        "price_cache_write_per_mtok": 6.25,
    }
    fields.update(overrides)
    return ModelConfig(**fields)


def test_cost_prices_all_four_token_classes():
    cost = _model().cost_usd(1_000_000, 1_000_000, 1_000_000, 1_000_000)
    assert cost == pytest.approx(5.0 + 25.0 + 0.5 + 6.25)


def test_cost_charges_cache_writes_above_plain_input():
    """The defect: cache-creation tokens had no price at all, and the display
    call sites folded them into ``input_tokens`` and so billed them at 1x."""
    model = _model()
    as_input = model.cost_usd(10_000, 0, 0, 0)
    as_write = model.cost_usd(0, 0, 0, 10_000)
    assert as_write > as_input
    assert as_write == pytest.approx(1.25 * as_input)


def test_cost_of_a_realistic_cached_tick():
    # A tick reading a warm ~2k-token prefix: the cache read is nearly free and
    # the output dominates, which is the shape the README's estimate assumes.
    cost = _model().cost_usd(900, 1_400, 2_100, 0)
    expected = (900 * 5.0 + 1_400 * 25.0 + 2_100 * 0.5) / 1_000_000
    assert cost == pytest.approx(expected)


def test_cost_of_nothing_is_zero():
    assert _model().cost_usd(0, 0, 0, 0) == 0.0


def test_cost_requires_the_cache_write_count():
    """Not defaulted to 0 on purpose — a caller forgetting that cache-creation
    tokens exist is precisely how this was wrong three separate ways."""
    with pytest.raises(TypeError):
        _model().cost_usd(1, 2, 3)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# [cadence]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seconds", [0, 1, 4, -60])
def test_fast_tick_below_five_seconds_is_fatal(tmp_path, seconds):
    with pytest.raises(ConfigError, match="hammer the APIs"):
        load_body(tmp_path, f"{ONE_COIN}\n[cadence]\nfast_tick_seconds = {seconds}\n")


def test_fast_tick_of_exactly_five_seconds_is_allowed(tmp_path):
    cfg = load_body(
        tmp_path, f"{ONE_COIN}\n[cadence]\nfast_tick_seconds = 5\nslow_tick_seconds = 5\n"
    )
    assert cfg.cadence.fast_tick_seconds == 5


def test_slow_tick_faster_than_fast_tick_is_fatal(tmp_path):
    with pytest.raises(ConfigError, match="slow_tick_seconds must be >= fast_tick"):
        load_body(
            tmp_path,
            f"{ONE_COIN}\n[cadence]\nfast_tick_seconds = 120\nslow_tick_seconds = 60\n",
        )


def test_equal_cadences_are_allowed(tmp_path):
    cfg = load_body(
        tmp_path,
        f"{ONE_COIN}\n[cadence]\nfast_tick_seconds = 60\nslow_tick_seconds = 60\n",
    )
    assert cfg.cadence.slow_tick_seconds == 60


# ---------------------------------------------------------------------------
# [risk]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pct", [0.0, -0.1, 1.01, 30.0])
def test_max_position_pct_outside_zero_to_one_is_fatal(tmp_path, pct):
    """A fraction, not a percent: `30` here would authorise a 3,000% position."""
    with pytest.raises(ConfigError, match=r"max_position_pct must be a fraction"):
        load_body(tmp_path, f"{ONE_COIN}\n[risk]\nmax_position_pct = {pct}\n")


def test_max_position_pct_of_one_is_allowed(tmp_path):
    cfg = load_body(tmp_path, f"{ONE_COIN}\n[risk]\nmax_position_pct = 1.0\n")
    assert cfg.risk.max_position_pct == 1.0


@pytest.mark.parametrize("pct", [0.0, -0.05, 1.0, 15.0])
def test_stop_loss_pct_outside_the_open_interval_is_fatal(tmp_path, pct):
    """1.0 is excluded too: a stop at -100% is not a stop."""
    with pytest.raises(ConfigError, match=r"stop_loss_pct must be a fraction"):
        load_body(tmp_path, f"{ONE_COIN}\n[risk]\nstop_loss_pct = {pct}\n")


def test_stop_loss_pct_just_inside_the_interval_is_allowed(tmp_path):
    cfg = load_body(tmp_path, f"{ONE_COIN}\n[risk]\nstop_loss_pct = 0.99\n")
    assert cfg.risk.stop_loss_pct == 0.99


# ---------------------------------------------------------------------------
# [execution]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rate", [-0.01, 1.0, 1.5, 6.0])
def test_failed_tx_rate_outside_zero_to_one_is_fatal(tmp_path, rate):
    """1.0 is excluded: every swap failing is a config typo, not a scenario.
    And `6` for "6%" is the mistake this branch is really here to catch."""
    with pytest.raises(ConfigError, match=r"failed_tx_rate must be in"):
        load_body(tmp_path, f"{ONE_COIN}\n[execution]\nfailed_tx_rate = {rate}\n")


def test_failed_tx_rate_of_zero_is_allowed(tmp_path):
    cfg = load_body(tmp_path, f"{ONE_COIN}\n[execution]\nfailed_tx_rate = 0.0\n")
    assert cfg.execution.failed_tx_rate == 0.0


def _execution(**overrides) -> ExecutionConfig:
    fields = {
        "slippage_bps_fallback": 50.0,
        "gas_usd_per_swap": 0.21,
        "failed_tx_rate": 0.06,
        "default_pool_fee_pct": 0.25,
        "pool_fee_pct": {"Raydium": 0.25, "Orca": 0.30},
    }
    fields.update(overrides)
    return ExecutionConfig(**fields)


def test_fee_for_a_single_known_venue():
    assert _execution().fee_pct_for(("Orca",)) == pytest.approx(0.30)


def test_two_hop_route_owes_two_fees():
    """The whole reason this sums rather than looks up: a route through two
    pools pays both, and charging one is how a simulator flatters itself."""
    assert _execution().fee_pct_for(("Raydium", "Orca")) == pytest.approx(0.55)


def test_three_hop_route_owes_three_fees():
    assert _execution().fee_pct_for(("Orca", "Orca", "Orca")) == pytest.approx(0.90)


def test_unknown_venue_falls_back_to_the_default_not_to_free():
    assert _execution().fee_pct_for(("SomeNewAmm",)) == pytest.approx(0.25)


def test_unknown_venues_fall_back_per_hop():
    assert _execution().fee_pct_for(("SomeNewAmm", "Another")) == pytest.approx(0.50)


def test_mixed_known_and_unknown_hops():
    assert _execution().fee_pct_for(("Orca", "SomeNewAmm")) == pytest.approx(0.55)


def test_no_route_labels_still_charges_one_default_fee():
    """An empty routePlan means Jupiter told us nothing, not that the swap was
    free."""
    assert _execution().fee_pct_for(()) == pytest.approx(0.25)
    assert _execution().fee_pct_for([]) == pytest.approx(0.25)


def test_fee_accepts_a_list_as_well_as_a_tuple():
    assert _execution().fee_pct_for(["Raydium", "Raydium"]) == pytest.approx(0.50)


def test_pool_fee_table_is_read_from_the_file(tmp_path):
    cfg = load_body(
        tmp_path,
        f"""
        {ONE_COIN}
        [execution]
        default_pool_fee_pct = 0.40

        [execution.pool_fee_pct]
        Raydium = 0.05
        "Meteora DLMM" = 0.10
        """,
    )
    assert cfg.execution.fee_pct_for(("Raydium", "Meteora DLMM")) == pytest.approx(0.15)
    assert cfg.execution.fee_pct_for(("Unknown",)) == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# [data] — jupiter_url_base
# ---------------------------------------------------------------------------


def _data(**overrides) -> DataConfig:
    fields = {
        "dexscreener_base": "https://api.dexscreener.com",
        "geckoterminal_base": "https://api.geckoterminal.com/api/v2",
        "jupiter_base": "https://lite-api.jup.ag",
        "jupiter_base_keyed": "https://api.jup.ag",
        "http_timeout_seconds": 15.0,
        "candles_5m": 100,
        "candles_1h": 100,
        "jupiter_api_key": None,
    }
    fields.update(overrides)
    return DataConfig(**fields)


def test_jupiter_uses_the_lite_host_without_a_key():
    assert _data().jupiter_url_base == "https://lite-api.jup.ag"


def test_jupiter_switches_to_the_keyed_host_with_a_key():
    assert _data(jupiter_api_key="k").jupiter_url_base == "https://api.jup.ag"


def test_jupiter_key_comes_from_the_environment_not_the_file(tmp_path, monkeypatch):
    """Keys live in the environment; config.toml is committed."""
    assert load_body(tmp_path, ONE_COIN).data.jupiter_api_key is None
    monkeypatch.setenv("JUPITER_API_KEY", "secret")
    cfg = load_body(tmp_path, ONE_COIN)
    assert cfg.data.jupiter_api_key == "secret"
    assert cfg.data.jupiter_url_base == cfg.data.jupiter_base_keyed


def test_empty_jupiter_key_is_treated_as_absent(tmp_path, monkeypatch):
    """An exported-but-blank variable is the classic .env mistake; it must not
    route us to the keyed host, which would then 401 every quote."""
    monkeypatch.setenv("JUPITER_API_KEY", "")
    cfg = load_body(tmp_path, ONE_COIN)
    assert cfg.data.jupiter_api_key is None
    assert cfg.data.jupiter_url_base == cfg.data.jupiter_base


# ---------------------------------------------------------------------------
# [sentiment] — credentials
# ---------------------------------------------------------------------------


def _sentiment(**overrides) -> SentimentConfig:
    fields = {
        "enabled": True,
        "cache_ttl_seconds": 600,
        "lookback_hours": 24,
        "baseline_days": 7,
        "subreddits": ("solana",),
        "reddit_client_id": None,
        "reddit_client_secret": None,
        "reddit_user_agent": "memetrader/0.1",
    }
    fields.update(overrides)
    return SentimentConfig(**fields)


@pytest.mark.parametrize(
    "client_id, secret, expected",
    [
        ("id", "secret", True),
        ("id", None, False),
        (None, "secret", False),
        (None, None, False),
        ("", "secret", False),
        ("id", "", False),
    ],
)
def test_has_reddit_credentials_requires_both(client_id, secret, expected):
    """Half a credential pair is no credential: PRAW would fail at OAuth and
    the run would lose its freshest sentiment source mid-flight."""
    cfg = _sentiment(reddit_client_id=client_id, reddit_client_secret=secret)
    assert cfg.has_reddit_credentials is expected


def test_reddit_credentials_are_read_from_the_environment(tmp_path, monkeypatch):
    assert load_body(tmp_path, ONE_COIN).sentiment.has_reddit_credentials is False
    monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
    assert load_body(tmp_path, ONE_COIN).sentiment.has_reddit_credentials is True


def test_reddit_user_agent_falls_back_to_a_specific_string(tmp_path):
    """Reddit rejects generic user agents, so the default cannot be blank."""
    agent = load_body(tmp_path, ONE_COIN).sentiment.reddit_user_agent
    assert "memetrader" in agent


def test_ttl_stays_below_the_decision_cadence_in_the_shipped_config():
    """A cache hit returns a brief with empty ``top_posts`` — titles are never
    persisted — so a TTL above the cadence would silently strip the only
    qualitative evidence in the stream from every decision. See the comment
    beside the value in config.toml."""
    cfg = load(REPO_ROOT / "config.toml")
    assert cfg.sentiment.cache_ttl_seconds < cfg.cadence.slow_tick_seconds


# ---------------------------------------------------------------------------
# find_project_root
# ---------------------------------------------------------------------------


def test_find_project_root_finds_config_in_the_directory_itself(tmp_path):
    write_config(tmp_path, ONE_COIN)
    assert find_project_root(tmp_path) == tmp_path.resolve()


def test_find_project_root_walks_up_from_a_nested_directory(tmp_path):
    write_config(tmp_path, ONE_COIN)
    nested = tmp_path / "src" / "memetrader" / "deep"
    nested.mkdir(parents=True)
    assert find_project_root(nested) == tmp_path.resolve()


def test_find_project_root_prefers_the_nearest_config(tmp_path):
    write_config(tmp_path, ONE_COIN)
    inner = tmp_path / "inner"
    inner.mkdir()
    write_config(inner, ONE_COIN)
    assert find_project_root(inner) == inner.resolve()


def test_find_project_root_falls_back_to_the_installed_package_root(tmp_path):
    """With no config.toml anywhere above the start directory, the documented
    fallback is the repo the package was installed from — which is how
    ``memetrader`` works when run from an unrelated cwd."""
    stray = tmp_path / "nowhere"
    stray.mkdir()
    assert find_project_root(stray) == REPO_ROOT


# ---------------------------------------------------------------------------
# Whole-file behaviour
# ---------------------------------------------------------------------------


def test_load_creates_the_data_directory(tmp_path):
    cfg = load_body(tmp_path, ONE_COIN)
    assert cfg.data_dir == tmp_path / "data"
    assert cfg.data_dir.is_dir()
    assert cfg.state_path.name == "state.json"
    assert cfg.trades_path.name == "trades.jsonl"
    assert cfg.decisions_path.name == "decisions.jsonl"


def test_coin_lookup_raises_for_an_unconfigured_symbol(tmp_path):
    cfg = load_body(tmp_path, ONE_COIN)
    with pytest.raises(KeyError):
        cfg.coin("DOGE")


def test_the_shipped_config_loads(tmp_path):
    """The one config that actually matters. Every other test here builds a
    synthetic file, so nothing else would notice a typo in the real one."""
    cfg = load(REPO_ROOT / "config.toml")
    assert len(cfg.coins) >= 1
    assert cfg.model.price_cache_write_per_mtok > 0
