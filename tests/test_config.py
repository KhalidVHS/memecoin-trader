"""What ``config.py`` promises: the shipped file loads, absent keys fall through
to the enforcing module, and everything else fails loudly at startup.

Three things this file is really testing, in order of how much they matter.

1. **The real ``config.toml`` loads and says what it says.** Every other test
   here builds a synthetic file, so nothing else would notice a typo in the one
   config that actually runs — or, worse, a key the loader silently ignores
   while the operator reads it and believes it.
2. **A key absent from TOML takes the enforcing module's default.** That is the
   whole point of the rewrite: a limit used to exist twice, as a field in
   config.py and as the code enforcing it in risk.py, and two copies of a
   number are two numbers that will eventually disagree. The test for this is
   an equality against ``RiskParams()`` itself, not against a literal — a
   literal here would be the third copy.
3. **The error paths.** ``load()`` is the one place that gets to say "no". An
   unasserted validation branch is a promise nobody has checked.

Configs are written under ``tmp_path`` and kept minimal: each carries the
smallest valid skeleton plus the single thing under test, so a failure names
the branch rather than a wall of TOML. Nothing here touches the network and
nothing writes into the repo.
"""

from __future__ import annotations

import textwrap
import tomllib
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
from memetrader.http import RetryPolicy
from memetrader.market import MarketParams, SafetyScreenParams
from memetrader.portfolio import MarkParams
from memetrader.risk import RiskParams
from memetrader.strategy import StrategySettings
from memetrader.types import ExecutionMode

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = REPO_ROOT / "config.toml"

# 43 and 44 base58 characters — two of the real mints from config.toml.
BONK_MINT = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
WIF_MINT = "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm"

ONE_COIN = f"""
    [[coins]]
    symbol = "BONK"
    mint = "{BONK_MINT}"
"""


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch):
    """``load()`` reads every credential from the environment, so a developer's
    real keys would otherwise decide what these tests assert.

    Note that this only governs the ``tmp_path`` configs: loading the *shipped*
    config runs ``load_dotenv(REPO_ROOT / ".env")``, which will put a
    developer's real keys straight back. No test below asserts anything about a
    secret while loading the shipped file, and that is deliberate.
    """
    for name in (
        "ANTHROPIC_API_KEY",
        "JUPITER_API_KEY",
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "REDDIT_USER_AGENT",
        "MEMETRADER_AUTHOR_HASH_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")
    return path


def load_body(tmp_path: Path, body: str):
    return load(write_config(tmp_path, body))


@pytest.fixture(scope="session")
def shipped():
    """The config the system actually runs on. Read-only in every test here."""
    return load(SHIPPED_CONFIG)


# ---------------------------------------------------------------------------
# The shipped config.toml
# ---------------------------------------------------------------------------


#: TOML table -> the attribute on ``Config`` it is supposed to populate. Used to
#: sweep every scalar key in the real file and prove it lands somewhere. A key
#: the loader quietly drops is the failure mode this catches: the operator reads
#: the file, believes the number, and the code obeys a different one.
_TABLE_TO_ATTR = {
    "model": "model",
    "cadence": "cadence",
    "strategy": "strategy",
    "risk": "risk",
    "execution": "execution",
    "data": "data",
    "market": "market",
    "safety_screen": "safety",
    "sentiment": "sentiment",
    "prompt": "prompt",
}


def _shipped_scalar_keys():
    raw = tomllib.loads(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    params = []
    for table, attr in _TABLE_TO_ATTR.items():
        for key, value in raw.get(table, {}).items():
            # `kind` lands on Config.strategy_kind rather than on StrategySettings;
            # lists and sub-tables (universe, subreddits, [risk.mark],
            # [execution.pool_fee_pct]) are asserted by hand below.
            if key == "kind" or isinstance(value, list | dict):
                continue
            params.append(pytest.param(attr, key, value, id=f"{table}.{key}"))
    return params


@pytest.mark.parametrize(("attr", "key", "value"), _shipped_scalar_keys())
def test_every_scalar_key_in_the_shipped_config_reaches_the_object(
    shipped, attr, key, value
):
    """A key present in config.toml must arrive, unaltered, on the object that
    enforces it. Silently ignoring one is the same defect as restating it: what
    the operator reads stops being what the code obeys."""
    target = getattr(shipped, attr)
    assert hasattr(target, key), f"[{attr}] has no field {key!r} — config.toml states it"
    assert getattr(target, key) == value


def test_the_shipped_config_loads_and_names_the_three_coins(shipped):
    assert shipped.symbols == ("BONK", "WIF", "POPCAT")
    assert shipped.coin("BONK").mint == BONK_MINT
    assert shipped.coin("WIF").mint == WIF_MINT
    assert shipped.coin("WIF").aliases == ("WIF", "dogwifhat", "dogwifcoin")
    assert shipped.starting_cash_usd == 1000.0


def test_the_shipped_config_ships_the_post_audit_safety_posture(shipped):
    """The three settings the audit changed, pinned in the file that ships.

    A default is authority — it is what runs when nobody chose — so each of
    these is a finding, not a preference: C6 (the model does not hold order
    authority), C7 (an unmeasured input is not on) and C12 (this is a paper
    trader)."""
    assert shipped.strategy_kind == "baseline"
    assert shipped.sentiment.enabled is False
    assert shipped.execution_mode is ExecutionMode.PAPER


def test_the_shipped_config_carries_the_correlated_sleeve(shipped):
    """C10: BONK, WIF and POPCAT are one factor with three tickers. A sleeve
    that omitted one of them would let a 30% per-symbol cap become 90% of the
    book in the same trade."""
    assert shipped.risk.correlated_sleeve == frozenset({"BONK", "WIF", "POPCAT"})
    assert shipped.risk.universe == frozenset({"BONK", "WIF", "POPCAT"})


def test_the_shipped_config_maps_the_split_http_timeouts(shipped):
    """``[http] *_timeout_seconds`` do not share a name with ``Timeouts``'
    fields, so the sweep above cannot see them."""
    assert shipped.http.timeouts.connect == 5.0
    assert shipped.http.timeouts.read == 12.0
    assert shipped.http.timeouts.write == 10.0
    assert shipped.http.timeouts.pool == 5.0
    # retry_budget_seconds is the TOML spelling of total_budget_seconds.
    assert shipped.http.retry.total_budget_seconds == 20.0
    assert shipped.http.retry.max_attempts == 3
    assert shipped.http.breaker_failure_threshold == 4


def test_the_shipped_config_maps_the_mark_and_pool_fee_subtables(shipped):
    assert shipped.mark.mid_haircut_pct == 2.0
    assert shipped.mark.estimate_min_haircut_pct == 5.0
    assert shipped.mark.max_mark_age_seconds == 120.0
    assert shipped.mark.allow_mid_stop_reference is True
    assert shipped.execution.fee_pct_for(("Raydium",)) == pytest.approx(0.25)
    assert shipped.execution.fee_pct_for(("Meteora DLMM", "Orca")) == pytest.approx(0.55)


def test_the_shipped_config_charges_no_pool_fee_on_a_fill(shipped):
    """C2's 45x double-count: the router's ``outAmount`` is already net of the
    pool fee, so ``[risk] assumed_pool_fee_pct`` is 0.0 on purpose. A non-zero
    value here would re-introduce the finding."""
    assert shipped.risk.assumed_pool_fee_pct == 0.0


def test_the_sentiment_ttl_stays_below_the_decision_cadence(shipped):
    """The TTL is a debounce for interactive runs, not a cost control for the
    loop: above the cadence, a decision tick would read a cache instead of
    re-fetching. See the comment beside the value in config.toml."""
    assert shipped.sentiment.cache_ttl_seconds < shipped.cadence.slow_tick_seconds


# ---------------------------------------------------------------------------
# Absent keys take the enforcing module's default
# ---------------------------------------------------------------------------


def test_an_absent_risk_table_yields_exactly_the_module_defaults(tmp_path):
    """The central property of the rewrite, stated as an equality against
    ``RiskParams()`` itself rather than against literals — a literal here would
    be the second copy of the number that this design exists to remove.

    Only ``universe`` and ``correlated_sleeve`` differ, and both are argued for
    in ``_load_risk``'s docstring: the library default of "nothing is
    permitted" is right for a library and is a silent total refusal to trade at
    startup."""
    cfg = load_body(tmp_path, ONE_COIN)
    assert cfg.risk == RiskParams(
        universe=frozenset({"BONK"}), correlated_sleeve=frozenset({"BONK"})
    )


def test_an_absent_key_takes_the_module_default_not_a_second_copy(tmp_path):
    """Spelled out per field as well, because the equality above would still
    pass if config.py hardcoded every default to the same value it happens to
    have today. These assertions read the module, so they track it."""
    cfg = load_body(tmp_path, ONE_COIN)
    defaults = RiskParams()
    assert cfg.risk.max_drawdown_pct == defaults.max_drawdown_pct
    assert cfg.risk.min_liquidity_usd == defaults.min_liquidity_usd
    assert cfg.risk.max_consecutive_failures == defaults.max_consecutive_failures
    assert cfg.risk.require_known_pool_age is defaults.require_known_pool_age


def test_absent_strategy_mark_market_and_safety_tables_take_module_defaults(tmp_path):
    cfg = load_body(tmp_path, ONE_COIN)
    assert cfg.strategy == StrategySettings()
    assert cfg.mark == MarkParams()
    assert cfg.market == MarketParams()
    assert cfg.safety == SafetyScreenParams()
    assert cfg.http.retry == RetryPolicy()


def test_the_read_timeout_still_follows_the_legacy_data_timeout(tmp_path):
    """One deliberate exception to "absent means module default".

    ``[data] http_timeout_seconds`` predates the split timeouts and still sets
    the *read* budget, so with both tables absent the read timeout is
    ``DataConfig``'s 15.0 rather than ``Timeouts``' own 12.0. Pinned because it
    is a surprise, not because it is ideal."""
    cfg = load_body(tmp_path, ONE_COIN)
    assert cfg.http.timeouts.read == 15.0
    assert cfg.http.timeouts.connect == 5.0


def test_a_present_key_overrides_the_module_default(tmp_path):
    """The other half: falling through to the default must not mean ignoring
    the file."""
    cfg = load_body(tmp_path, f"{ONE_COIN}\n[risk]\nmax_drawdown_pct = 7.5\n")
    assert cfg.risk.max_drawdown_pct == 7.5
    assert cfg.risk.max_drawdown_pct != RiskParams().max_drawdown_pct


def test_defaults_that_are_safety_decisions_rather_than_taste(tmp_path):
    """Audit C6 and C7, at the level of what runs when nobody chose."""
    cfg = load_body(tmp_path, ONE_COIN)
    assert cfg.strategy_kind == "baseline"  # not "advisory": the LLM has no authority
    assert cfg.sentiment.enabled is False  # unmeasured input, off until ablated
    assert cfg.execution_mode is ExecutionMode.PAPER
    assert cfg.prompt.decision_history == 10


# ---------------------------------------------------------------------------
# execution_mode
# ---------------------------------------------------------------------------


def _with_mode(mode: str) -> str:
    """Top-level keys must precede the first ``[[coins]]`` table, or TOML reads
    them as fields *of* that coin — which the loader would then ignore in
    silence. Worth a helper rather than a comment: the mistake is invisible."""
    return f'execution_mode = "{mode}"\n{ONE_COIN}'


@pytest.mark.parametrize("mode", ["read_only", "paper", "live"])
def test_every_execution_mode_is_accepted_case_insensitively(tmp_path, mode):
    cfg = load_body(tmp_path, _with_mode(mode.upper()))
    assert cfg.execution_mode is ExecutionMode(mode)


def test_live_parses_without_anything_live_materialising(tmp_path):
    """Config's job is to parse the enum, not to authorise it.

    ``LIVE`` has to be a value the type admits so every capability check is
    written against the real three-way distinction — but nothing in a ``Config``
    can submit a transaction. The refusal lives at broker construction, where
    the wallet would have to exist; loading is inert."""
    cfg = load_body(tmp_path, _with_mode("live"))
    assert cfg.execution_mode is ExecutionMode.LIVE
    assert cfg.anthropic_api_key is None


@pytest.mark.parametrize("mode", ["", "dry_run", "real", "PAPER_TRADING", "livee"])
def test_an_unknown_execution_mode_is_fatal_and_lists_the_alternatives(tmp_path, mode):
    with pytest.raises(ConfigError, match="execution_mode must be one of"):
        load_body(tmp_path, _with_mode(mode))


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
    coin — two entries for it would double every per-symbol cap."""
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
# [strategy]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["cash", "baseline", "advisory"])
def test_every_known_strategy_is_accepted_case_insensitively(tmp_path, kind):
    cfg = load_body(tmp_path, f'{ONE_COIN}\n[strategy]\nkind = "{kind.upper()}"\n')
    assert cfg.strategy_kind == kind


@pytest.mark.parametrize("kind", ["", "llm", "momentum", "advisor", "base_line"])
def test_an_unknown_strategy_kind_is_fatal(tmp_path, kind):
    """A typo must be a startup failure, not a silent fall-through to whatever
    ``build_strategy``'s match statement does last."""
    with pytest.raises(ConfigError, match=r"\[strategy\] kind must be one of"):
        load_body(tmp_path, f'{ONE_COIN}\n[strategy]\nkind = "{kind}"\n')


def test_strategy_size_above_the_risk_entry_cap_is_fatal(tmp_path):
    """A limit stated in one place and enforced in another is the §15 defect
    this file exists to prevent: risk would veto every entry the strategy
    proposed, and the log would blame the market."""
    with pytest.raises(ConfigError, match=r"flat_size_usd 700\.0 exceeds"):
        load_body(tmp_path, f"{ONE_COIN}\n[strategy]\nflat_size_usd = 700.0\n")


def test_strategy_size_equal_to_the_risk_entry_cap_is_allowed(tmp_path):
    cfg = load_body(tmp_path, f"{ONE_COIN}\n[strategy]\nflat_size_usd = 100.0\n")
    assert cfg.strategy.flat_size_usd == 100.0


def test_a_strategy_min_trade_below_the_risk_min_trade_is_fatal(tmp_path):
    """The mirror image: the strategy would propose orders risk is guaranteed
    to veto, which costs a tick and produces no trade."""
    with pytest.raises(ConfigError, match=r"\[strategy\] min_trade_usd 5.0 is below"):
        load_body(tmp_path, f"{ONE_COIN}\n[strategy]\nmin_trade_usd = 5.0\n")


def test_a_strategy_min_trade_above_the_risk_min_trade_is_allowed(tmp_path):
    cfg = load_body(tmp_path, f"{ONE_COIN}\n[strategy]\nmin_trade_usd = 20.0\n")
    assert cfg.strategy.min_trade_usd == 20.0


def test_a_non_numeric_strategy_value_names_the_key(tmp_path):
    with pytest.raises(ConfigError, match="flat_size_usd must be a number"):
        load_body(tmp_path, f'{ONE_COIN}\n[strategy]\nflat_size_usd = "big"\n')


# ---------------------------------------------------------------------------
# [model]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("effort", ["", "medium-high", "HIGHEST", "1", "extra-high"])
def test_invalid_effort_is_fatal(tmp_path, effort):
    with pytest.raises(ConfigError, match=r"\[model\] effort must be one of"):
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
    """Regression, wrong formula #1 and #3: this method had no cache-write
    price at all, and the two display call sites folded cache-creation into
    ``input_tokens`` and so billed it at 1x. Both understated the bill."""
    model = _model()
    as_input = model.cost_usd(10_000, 0, 0, 0)
    as_write = model.cost_usd(0, 0, 0, 10_000)
    assert as_write > as_input
    assert as_write == pytest.approx(1.25 * as_input)


def test_cost_counts_output_and_cache_reads_not_input_alone():
    """Regression, wrong formula #2: ``brain.Usage.cost_usd`` passed only
    ``input_tokens``, so an expensive thinking-heavy tick reported as cheap."""
    model = _model()
    input_only = model.cost_usd(900, 0, 0, 0)
    whole_bill = model.cost_usd(900, 1_400, 2_100, 0)
    assert whole_bill > input_only
    assert whole_bill == pytest.approx((900 * 5.0 + 1_400 * 25.0 + 2_100 * 0.5) / 1e6)


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
    with pytest.raises(ConfigError, match="fast_tick_seconds below 5"):
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
# [risk] — the fraction/percent unit trap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pct", [0.0, -0.1, 1.01, 30.0])
def test_max_position_pct_outside_zero_to_one_is_fatal(tmp_path, pct):
    """A FRACTION, not a percent. `30` meaning "30%" would read as 3,000% of
    book, which is the one typo in this file that can lose the whole book in a
    single order."""
    with pytest.raises(ConfigError, match="max_position_pct"):
        load_body(tmp_path, f"{ONE_COIN}\n[risk]\nmax_position_pct = {pct}\n")


def test_the_percent_typo_error_says_which_unit_was_expected(tmp_path):
    """An error naming the key is not enough here: the operator typed a number
    that *looks* right to them, so the message has to say which unit it is in
    and show the intended spelling."""
    with pytest.raises(ConfigError) as exc:
        load_body(tmp_path, f"{ONE_COIN}\n[risk]\nmax_position_pct = 30\n")
    message = str(exc.value)
    assert "max_position_pct" in message
    assert "FRACTION" in message or "fraction" in message
    assert "0.30" in message or "(0, 1]" in message
    assert "30" in message  # the offending value, so it is recognisable


@pytest.mark.parametrize("pct", [0.0, -0.05, 15.0])
def test_stop_loss_pct_outside_the_fraction_range_is_fatal(tmp_path, pct):
    with pytest.raises(ConfigError, match="stop_loss_pct"):
        load_body(tmp_path, f"{ONE_COIN}\n[risk]\nstop_loss_pct = {pct}\n")


def test_a_stop_loss_of_exactly_one_is_fatal(tmp_path):
    """1.0 is excluded where max_position_pct's 1.0 is not: a stop that fires
    at -100% is not a stop, it is a total loss already taken. This is the one
    bound config.py tightens beyond ``RiskParams``' own."""
    with pytest.raises(ConfigError, match=r"stop_loss_pct must be a fraction in \(0, 1\)"):
        load_body(tmp_path, f"{ONE_COIN}\n[risk]\nstop_loss_pct = 1.0\n")


def test_fractions_at_the_top_of_their_range_are_allowed(tmp_path):
    cfg = load_body(
        tmp_path,
        f"{ONE_COIN}\n[risk]\nmax_position_pct = 1.0\nstop_loss_pct = 0.99\n",
    )
    assert cfg.risk.max_position_pct == 1.0
    assert cfg.risk.stop_loss_pct == 0.99


@pytest.mark.parametrize("pct", [3.0, 0.5, 25.0, 99.9])
def test_whole_percent_fields_are_not_range_checked_as_fractions(tmp_path, pct):
    """The other half of the unit convention, and the reason the two checks
    cannot be unified: ``max_price_impact_pct = 3.0`` means 3%, and rejecting
    it for exceeding 1.0 would be the same bug pointed the other way."""
    cfg = load_body(tmp_path, f"{ONE_COIN}\n[risk]\nmax_price_impact_pct = {pct}\n")
    assert cfg.risk.max_price_impact_pct == pct


def test_a_whole_percent_at_or_above_one_hundred_is_still_fatal(tmp_path):
    """Not unchecked, just checked against the right unit."""
    with pytest.raises(ConfigError, match="max_price_impact_pct"):
        load_body(tmp_path, f"{ONE_COIN}\n[risk]\nmax_price_impact_pct = 100.0\n")


# ---------------------------------------------------------------------------
# [risk] — universe, sleeve and the sizing bounds
# ---------------------------------------------------------------------------


def test_universe_and_sleeve_default_to_the_configured_coins(tmp_path):
    """``RiskParams`` defaults both to the empty set, which is right for a
    library — an unstated universe should permit nothing — and would be a
    silent, total refusal to trade if it survived startup."""
    cfg = load_body(
        tmp_path,
        f"""
        {ONE_COIN}

        [[coins]]
        symbol = "WIF"
        mint = "{WIF_MINT}"
        """,
    )
    assert cfg.risk.universe == frozenset({"BONK", "WIF"})
    assert cfg.risk.correlated_sleeve == frozenset({"BONK", "WIF"})
    assert RiskParams().universe == frozenset()  # the library default it overrides


def test_an_explicit_universe_may_be_a_subset_and_is_upper_cased(tmp_path):
    cfg = load_body(
        tmp_path,
        f"""
        {ONE_COIN}

        [[coins]]
        symbol = "WIF"
        mint = "{WIF_MINT}"

        [risk]
        universe = ["bonk"]
        """,
    )
    assert cfg.risk.universe == frozenset({"BONK"})
    # The sleeve still defaults to every configured coin, not to the universe.
    assert cfg.risk.correlated_sleeve == frozenset({"BONK", "WIF"})


def test_a_universe_naming_an_unconfigured_coin_is_fatal(tmp_path):
    """Otherwise the typo is invisible: the coin simply never trades, and
    nothing in the logs says why."""
    with pytest.raises(ConfigError, match=r"\[risk\] universe names \['DOGE'\]"):
        load_body(tmp_path, f'{ONE_COIN}\n[risk]\nuniverse = ["BONK", "DOGE"]\n')


def test_default_entry_above_max_entry_is_fatal(tmp_path):
    with pytest.raises(ConfigError, match="default_entry_usd exceeds max_entry_usd"):
        load_body(tmp_path, f"{ONE_COIN}\n[risk]\ndefault_entry_usd = 700.0\n")


def test_default_entry_equal_to_max_entry_is_allowed(tmp_path):
    cfg = load_body(
        tmp_path,
        f"{ONE_COIN}\n[risk]\ndefault_entry_usd = 100.0\nmax_entry_usd = 100.0\n",
    )
    assert cfg.risk.default_entry_usd == 100.0


def test_a_non_numeric_risk_value_names_the_key(tmp_path):
    with pytest.raises(ConfigError, match="min_liquidity_usd must be a number"):
        load_body(tmp_path, f'{ONE_COIN}\n[risk]\nmin_liquidity_usd = "lots"\n')


# ---------------------------------------------------------------------------
# [http]
# ---------------------------------------------------------------------------


def test_a_retry_budget_longer_than_a_fast_tick_is_fatal(tmp_path):
    """A retry that returns after its own tick has been missed is a stall, not
    resilience."""
    with pytest.raises(ConfigError, match=r"retry_budget_seconds 60\.0 is long enough"):
        load_body(tmp_path, f"{ONE_COIN}\n[http]\nretry_budget_seconds = 60.0\n")


def test_a_retry_budget_of_exactly_forty_five_seconds_is_allowed(tmp_path):
    cfg = load_body(tmp_path, f"{ONE_COIN}\n[http]\nretry_budget_seconds = 45.0\n")
    assert cfg.http.retry.total_budget_seconds == 45.0


def test_the_budget_ceiling_also_covers_the_module_spelling(tmp_path):
    """``total_budget_seconds`` is the field name; both spellings must hit the
    same ceiling, or the check is bypassable by writing the other one."""
    with pytest.raises(ConfigError, match="is long enough"):
        load_body(tmp_path, f"{ONE_COIN}\n[http]\ntotal_budget_seconds = 90.0\n")


def test_split_timeouts_are_read_per_phase(tmp_path):
    cfg = load_body(
        tmp_path,
        f"""
        {ONE_COIN}
        [http]
        connect_timeout_seconds = 1.5
        read_timeout_seconds = 2.5
        write_timeout_seconds = 3.5
        pool_timeout_seconds = 4.5
        """,
    )
    assert (cfg.http.timeouts.connect, cfg.http.timeouts.read) == (1.5, 2.5)
    assert (cfg.http.timeouts.write, cfg.http.timeouts.pool) == (3.5, 4.5)


def test_an_invalid_retry_policy_is_reported_as_a_config_error(tmp_path):
    """``RetryPolicy`` raises ``ValueError``; the operator needs a ``ConfigError``
    naming the table, not a traceback out of http.py."""
    with pytest.raises(ConfigError, match=r"\[http\] max_attempts"):
        load_body(tmp_path, f"{ONE_COIN}\n[http]\nmax_attempts = 0\n")


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
# Secrets: environment only, never the file
# ---------------------------------------------------------------------------


SECRETS_IN_TOML = f"""
anthropic_api_key = "sk-ant-from-the-file"
{ONE_COIN}

[data]
jupiter_api_key = "jup-from-the-file"

[sentiment]
reddit_client_id = "id-from-the-file"
reddit_client_secret = "secret-from-the-file"
author_hash_key = "salt-from-the-file"
"""


def test_a_secret_written_into_the_toml_does_not_populate_anything(tmp_path):
    """config.toml is committed, so a key in it is a published key. The loader
    reads every credential from the environment and ignores the file — the
    author hash key most pointedly of all: a salt committed next to the hashes
    it salts is not a salt."""
    cfg = load_body(tmp_path, SECRETS_IN_TOML)
    assert cfg.anthropic_api_key is None
    assert cfg.data.jupiter_api_key is None
    assert cfg.sentiment.reddit_client_id is None
    assert cfg.sentiment.reddit_client_secret is None
    assert cfg.sentiment.author_hash_key is None


def test_secrets_are_taken_from_the_environment_even_when_the_toml_states_them(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-the-env")
    monkeypatch.setenv("JUPITER_API_KEY", "jup-from-the-env")
    monkeypatch.setenv("MEMETRADER_AUTHOR_HASH_KEY", "salt-from-the-env")
    cfg = load_body(tmp_path, SECRETS_IN_TOML)
    assert cfg.anthropic_api_key == "sk-ant-from-the-env"
    assert cfg.data.jupiter_api_key == "jup-from-the-env"
    assert cfg.sentiment.author_hash_key == "salt-from-the-env"


def test_an_absent_author_hash_key_is_none_not_a_committed_default(tmp_path):
    """None means "fresh random key per process": hashes stay unlinkable across
    runs, which is the safe failure. A default string here would be a shared,
    public salt."""
    assert load_body(tmp_path, ONE_COIN).sentiment.author_hash_key is None


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
    ("client_id", "secret", "expected"),
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
    """Half a credential pair is no credential: PRAW would fail at OAuth and the
    run would lose the stream mid-flight rather than at startup."""
    cfg = _sentiment(reddit_client_id=client_id, reddit_client_secret=secret)
    assert cfg.has_reddit_credentials is expected


def test_reddit_credentials_are_read_from_the_environment(tmp_path, monkeypatch):
    assert load_body(tmp_path, ONE_COIN).sentiment.has_reddit_credentials is False
    monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
    assert load_body(tmp_path, ONE_COIN).sentiment.has_reddit_credentials is True


def test_reddit_user_agent_falls_back_to_a_specific_string(tmp_path):
    """Reddit rejects generic user agents, so the default cannot be blank."""
    assert "memetrader" in load_body(tmp_path, ONE_COIN).sentiment.reddit_user_agent


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


def test_find_project_root_raises_when_there_is_no_config_anywhere(tmp_path, monkeypatch):
    """The genuinely-not-found path. It needs the package's own root moved out
    of the way, because the installed repo always has a config.toml and would
    otherwise satisfy the fallback branch above."""
    from memetrader import config as config_module

    fake_pkg = tmp_path / "elsewhere" / "src" / "memetrader"
    fake_pkg.mkdir(parents=True)
    monkeypatch.setattr(config_module, "__file__", str(fake_pkg / "config.py"))
    stray = tmp_path / "nowhere"
    stray.mkdir()
    with pytest.raises(ConfigError, match=r"could not find config\.toml"):
        find_project_root(stray)


# ---------------------------------------------------------------------------
# Whole-file behaviour
# ---------------------------------------------------------------------------


def test_load_creates_the_data_directory_and_puts_every_path_inside_it(tmp_path):
    """All five artefacts live under one directory, so a run's whole record can
    be archived or discarded as a unit."""
    cfg = load_body(tmp_path, ONE_COIN)
    assert cfg.root == tmp_path.resolve()
    assert cfg.data_dir == tmp_path / "data"
    assert cfg.data_dir.is_dir()
    paths = (
        cfg.state_path,
        cfg.trades_path,
        cfg.intents_path,
        cfg.decisions_path,
        cfg.ledger_path,
    )
    assert [p.name for p in paths] == [
        "state.json",
        "trades.jsonl",
        "intents.jsonl",
        "decisions.jsonl",
        "ledger.jsonl",
    ]
    assert all(p.parent == cfg.data_dir for p in paths)
    # The ledger is not the trades file: C11's recovery story needs one ordered
    # stream joining decisions, intents and fills, which a fills-only file
    # cannot provide.
    assert cfg.ledger_path != cfg.trades_path


def test_symbols_preserves_the_order_the_file_states(tmp_path):
    cfg = load_body(
        tmp_path,
        f"""
        [[coins]]
        symbol = "WIF"
        mint = "{WIF_MINT}"

        [[coins]]
        symbol = "BONK"
        mint = "{BONK_MINT}"
        """,
    )
    assert cfg.symbols == ("WIF", "BONK")
    assert cfg.coin("BONK").mint == BONK_MINT


def test_coin_lookup_raises_for_an_unconfigured_symbol(tmp_path):
    cfg = load_body(tmp_path, ONE_COIN)
    with pytest.raises(KeyError, match="DOGE"):
        cfg.coin("DOGE")


def test_starting_cash_defaults_and_is_read_from_the_file(tmp_path):
    assert load_body(tmp_path, ONE_COIN).starting_cash_usd == 1000.0
    cfg = load_body(tmp_path, f"{ONE_COIN}\n[portfolio]\nstarting_cash_usd = 250.0\n")
    assert cfg.starting_cash_usd == 250.0
