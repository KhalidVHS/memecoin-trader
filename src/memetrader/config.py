"""Parse ``config.toml`` plus the environment into a frozen, validated object.

Everything that could be a magic number lives here. Validation happens once at
startup and fails loudly — a bad config should never surface as a weird fill
three hours into a run.

The adversarial audit changed this file's job. It used to *own* every tuning
constant as a bespoke dataclass, which meant a limit existed in two places: as
a field here and as the code that enforced it over in ``risk.py``. §15's
complaint about ``prompts._limits_block`` is the same defect one layer up —
when a number is written down twice, the two copies drift, and the one the
operator reads is not the one the code obeys.

So the owning module now defines its own frozen parameter object
(:class:`risk.RiskParams`, :class:`portfolio.MarkParams`,
:class:`market.MarketParams`, :class:`market.SafetyScreenParams`,
:class:`strategy.StrategySettings`, :class:`http.Timeouts`,
:class:`http.RetryPolicy`) next to the code that enforces it, and this file's
only remaining job is to *populate* those objects from TOML. There is exactly
one definition of every limit, and the prompt renders it from the same object
risk enforces it from.

Two consequences worth stating:

* Defaults live in the module, not here. A key absent from ``config.toml``
  takes the enforcing module's default, so the file stays readable and a new
  knob cannot silently become ``0.0`` because someone forgot a line.
* The parameter objects are constructible without this file at all, which is
  what lets a test or a sweep evaluate two parameterisations in one process.

Unit convention, which has bitten before and is worth repeating: config values
named ``_pct`` under ``[risk]`` are **fractions** where they are used as
multipliers (``max_position_pct = 0.30``, ``stop_loss_pct = 0.15``) and **whole
percents** everywhere else (``max_price_impact_pct = 3.0`` means 3%). The
enforcing dataclasses document which is which per field; the validation below
range-checks the fractions so a 30 typed where 0.30 was meant fails at startup
rather than authorising a 3,000% position.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .http import RetryPolicy, Timeouts
from .market import MarketParams, SafetyScreenParams
from .portfolio import MarkParams
from .risk import RiskParams
from .strategy import StrategySettings
from .types import ExecutionMode


class ConfigError(ValueError):
    """Raised at startup when config.toml is wrong. Always fatal."""


#: Ordered cheapest to most expensive, which is also least to most thinking.
#: Kept as a tuple rather than a set so the error message lists them in an
#: order that tells the reader which direction costs money.
_EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

#: The strategies ``strategy.build_strategy`` knows. Validated here so a typo is
#: a startup failure rather than a silent fall-through to whatever the match
#: statement's default happens to be.
_STRATEGIES: tuple[str, ...] = ("cash", "baseline", "advisory")


@dataclass(frozen=True, slots=True)
class CoinConfig:
    symbol: str
    mint: str
    aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    effort: str
    max_tokens: int
    price_input_per_mtok: float
    price_output_per_mtok: float
    price_cache_read_per_mtok: float
    price_cache_write_per_mtok: float

    def cost_usd(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read: int,
        cache_write: int,
    ) -> float:
        """The whole bill for one call. The only cost formula in the codebase.

        There were three of these before 2026-09-19 and none was right: this
        one had no cache-write price at all, ``brain.Usage.cost_usd`` passed
        only ``input_tokens``, and the two display call sites folded
        cache-creation into ``input_tokens`` and so billed it at 1x. Every one
        of them understated the bill in a different direction.

        ``cache_write`` is deliberately required rather than defaulted to 0.
        The single failure this method has actually suffered is a caller
        forgetting that cache-creation tokens exist, and a default is how that
        happens silently.
        """
        return (
            input_tokens * self.price_input_per_mtok
            + output_tokens * self.price_output_per_mtok
            + cache_read * self.price_cache_read_per_mtok
            + cache_write * self.price_cache_write_per_mtok
        ) / 1_000_000


@dataclass(frozen=True, slots=True)
class CadenceConfig:
    fast_tick_seconds: int
    slow_tick_seconds: int


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    slippage_bps_fallback: float
    gas_usd_per_swap: float
    failed_tx_rate: float
    default_pool_fee_pct: float
    pool_fee_pct: dict[str, float]

    def fee_pct_for(self, route_labels: tuple[str, ...] | list[str]) -> float:
        """Sum the pool fee across every hop in the route.

        A two-hop route pays two pool fees. Unknown venues fall back to the
        default rather than being treated as free.

        Retained for reporting and for estimating a cost hurdle. It is NOT used
        to charge a fill any more: the audit's C2 investigation found the pool
        fee was being deducted on top of a Jupiter ``outAmount`` that was
        already net of it, a 45x double-count on a 0.25% pool. The router's
        output is the fee-inclusive truth; only gas is additive.
        """
        if not route_labels:
            return self.default_pool_fee_pct
        return sum(
            self.pool_fee_pct.get(label, self.default_pool_fee_pct)
            for label in route_labels
        )


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Endpoints and credentials. Tuning knobs moved to :class:`MarketParams`."""

    dexscreener_base: str
    geckoterminal_base: str
    jupiter_base: str
    jupiter_base_keyed: str
    http_timeout_seconds: float
    candles_5m: int
    candles_1h: int
    jupiter_api_key: str | None

    @property
    def jupiter_url_base(self) -> str:
        """The keyed host when a key exists, the rate-decaying lite host if not."""
        return self.jupiter_base_keyed if self.jupiter_api_key else self.jupiter_base


@dataclass(frozen=True, slots=True)
class HttpConfig:
    timeouts: Timeouts
    retry: RetryPolicy
    breaker_failure_threshold: int
    breaker_cooldown_seconds: float


@dataclass(frozen=True, slots=True)
class SentimentConfig:
    enabled: bool
    cache_ttl_seconds: int
    lookback_hours: int
    baseline_days: int
    subreddits: tuple[str, ...]
    reddit_client_id: str | None
    reddit_client_secret: str | None
    reddit_user_agent: str
    author_hash_key: str | None = None
    include_comments: bool = True

    @property
    def has_reddit_credentials(self) -> bool:
        return bool(self.reddit_client_id and self.reddit_client_secret)


@dataclass(frozen=True, slots=True)
class PromptConfig:
    decision_history: int


@dataclass(frozen=True, slots=True)
class Config:
    root: Path
    data_dir: Path
    starting_cash_usd: float
    coins: tuple[CoinConfig, ...]
    execution_mode: ExecutionMode
    model: ModelConfig
    cadence: CadenceConfig
    strategy_kind: str
    strategy: StrategySettings
    risk: RiskParams
    mark: MarkParams
    market: MarketParams
    safety: SafetyScreenParams
    execution: ExecutionConfig
    data: DataConfig
    http: HttpConfig
    sentiment: SentimentConfig
    prompt: PromptConfig
    anthropic_api_key: str | None

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(c.symbol for c in self.coins)

    def coin(self, symbol: str) -> CoinConfig:
        for c in self.coins:
            if c.symbol == symbol:
                return c
        raise KeyError(f"{symbol!r} is not a configured coin")

    @property
    def state_path(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def trades_path(self) -> Path:
        return self.data_dir / "trades.jsonl"

    @property
    def intents_path(self) -> Path:
        return self.data_dir / "intents.jsonl"

    @property
    def decisions_path(self) -> Path:
        return self.data_dir / "decisions.jsonl"

    @property
    def ledger_path(self) -> Path:
        """The append-only journal of decisions, intents, state changes and fills.

        Separate from ``trades_path``, which the broker owns and which holds
        fills alone. The audit's C11 recovery story needs a single ordered
        stream in which a decision, the intents it produced and the fills those
        settled into can be joined by ID — a fills-only file cannot answer
        "which intent was in flight when the process died".
        """
        return self.data_dir / "ledger.jsonl"


def _require(table: dict, key: str, where: str) -> Any:
    if key not in table:
        raise ConfigError(f"config.toml is missing [{where}] {key}")
    return table[key]


def _floats(table: dict, keys: Sequence[str]) -> dict[str, float]:
    """Pull only the keys that are present, as floats.

    Absent keys are omitted rather than defaulted, so the enforcing module's
    own default applies. That is the whole point of not restating defaults
    here: a number written down twice is a number that will disagree with
    itself.
    """
    out: dict[str, float] = {}
    for key in keys:
        if key in table:
            try:
                out[key] = float(table[key])
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{key} must be a number, got {table[key]!r}") from exc
    return out


def _ints(table: dict, keys: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in keys:
        if key in table:
            try:
                out[key] = int(table[key])
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{key} must be an integer, got {table[key]!r}") from exc
    return out


def _bools(table: dict, keys: Sequence[str]) -> dict[str, bool]:
    return {key: bool(table[key]) for key in keys if key in table}


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` looking for config.toml."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "config.toml").is_file():
            return candidate
    # Fall back to the package's own repo root (src/memetrader/config.py -> ../..)
    pkg_root = Path(__file__).resolve().parents[2]
    if (pkg_root / "config.toml").is_file():
        return pkg_root
    raise ConfigError(f"could not find config.toml starting from {here}")


def load(path: Path | None = None) -> Config:
    """Load and validate configuration. Raises ``ConfigError`` on anything wrong."""
    root = path.parent.resolve() if path else find_project_root()
    config_path = path or (root / "config.toml")
    load_dotenv(root / ".env")

    with config_path.open("rb") as fh:
        raw = tomllib.load(fh)

    coins = _load_coins(raw)
    symbols = frozenset(c.symbol for c in coins)

    mode_raw = str(raw.get("execution_mode", "paper")).lower()
    try:
        execution_mode = ExecutionMode(mode_raw)
    except ValueError as exc:
        allowed = "|".join(m.value for m in ExecutionMode)
        raise ConfigError(
            f"execution_mode must be one of {allowed}, got {mode_raw!r}"
        ) from exc

    model = _load_model(raw.get("model", {}))
    cadence = _load_cadence(raw.get("cadence", {}))
    strategy_kind, strategy = _load_strategy(raw.get("strategy", {}))
    risk = _load_risk(raw.get("risk", {}), symbols)
    mark = _load_mark(raw.get("risk", {}).get("mark", {}))
    execution = _load_execution(raw.get("execution", {}))
    data = _load_data(raw.get("data", {}))
    market = _load_market(raw.get("market", {}), data)
    safety = _load_safety(raw.get("safety_screen", {}))
    http = _load_http(raw.get("http", {}), data.http_timeout_seconds)
    sentiment = _load_sentiment(raw.get("sentiment", {}))
    prompt = PromptConfig(
        decision_history=int(raw.get("prompt", {}).get("decision_history", 10))
    )

    # A risk limit stated in the prompt but not enforced is the §15 defect this
    # file exists to prevent, so the two sizing bounds must actually agree.
    if strategy.flat_size_usd > risk.max_entry_usd:
        raise ConfigError(
            f"[strategy] flat_size_usd {strategy.flat_size_usd} exceeds "
            f"[risk] max_entry_usd {risk.max_entry_usd} — risk would silently "
            f"refuse every entry the strategy proposes"
        )
    if strategy.min_trade_usd < risk.min_trade_usd:
        raise ConfigError(
            f"[strategy] min_trade_usd {strategy.min_trade_usd} is below "
            f"[risk] min_trade_usd {risk.min_trade_usd} — the strategy would "
            f"propose orders risk is guaranteed to veto"
        )

    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        root=root,
        data_dir=data_dir,
        starting_cash_usd=float(raw.get("portfolio", {}).get("starting_cash_usd", 1000.0)),
        coins=coins,
        execution_mode=execution_mode,
        model=model,
        cadence=cadence,
        strategy_kind=strategy_kind,
        strategy=strategy,
        risk=risk,
        mark=mark,
        market=market,
        safety=safety,
        execution=execution,
        data=data,
        http=http,
        sentiment=sentiment,
        prompt=prompt,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
    )


def _load_coins(raw: dict) -> tuple[CoinConfig, ...]:
    coins_raw = raw.get("coins") or []
    if len(coins_raw) < 1:
        raise ConfigError("config.toml must define at least one [[coins]] entry")
    coins: list[CoinConfig] = []
    seen: set[str] = set()
    for entry in coins_raw:
        symbol = str(_require(entry, "symbol", "coins")).upper()
        if symbol in seen:
            raise ConfigError(f"duplicate coin symbol {symbol!r} in config.toml")
        seen.add(symbol)
        mint = str(_require(entry, "mint", "coins"))
        # Solana mints are base58, 32-44 chars. A cheap check that catches the
        # common paste error long before any HTTP call.
        if not (32 <= len(mint) <= 44):
            raise ConfigError(
                f"{symbol}: mint {mint!r} is not a plausible Solana address "
                f"(expected 32-44 base58 characters, got {len(mint)})"
            )
        aliases = tuple(str(a) for a in entry.get("aliases", [symbol])) or (symbol,)
        coins.append(CoinConfig(symbol=symbol, mint=mint, aliases=aliases))
    return tuple(coins)


def _load_model(m: dict) -> ModelConfig:
    effort = str(m.get("effort", "high")).lower()
    # The full set anthropic 1.7.0's OutputConfigParam accepts, read from the
    # installed package on 2026-09-19 rather than from docs. Deliberately not
    # narrower: a value the API would accept should not fail here as though it
    # were a typo. The cost consequence is real and is stated in the README —
    # the $180-250/month figure was calibrated at "high", and effort is the
    # single largest lever on it, because it is thinking tokens that move.
    if effort not in _EFFORT_LEVELS:
        allowed = "|".join(_EFFORT_LEVELS)
        raise ConfigError(f"[model] effort must be one of {allowed}, got {effort!r}")
    return ModelConfig(
        name=str(m.get("name", "claude-opus-5")),
        effort=effort,
        max_tokens=int(m.get("max_tokens", 8000)),
        price_input_per_mtok=float(m.get("price_input_per_mtok", 5.0)),
        price_output_per_mtok=float(m.get("price_output_per_mtok", 25.0)),
        price_cache_read_per_mtok=float(m.get("price_cache_read_per_mtok", 0.5)),
        price_cache_write_per_mtok=float(m.get("price_cache_write_per_mtok", 6.25)),
    )


def _load_cadence(c: dict) -> CadenceConfig:
    cadence = CadenceConfig(
        fast_tick_seconds=int(c.get("fast_tick_seconds", 60)),
        slow_tick_seconds=int(c.get("slow_tick_seconds", 900)),
    )
    if cadence.fast_tick_seconds < 5:
        raise ConfigError("[cadence] fast_tick_seconds below 5 will hammer the APIs")
    if cadence.slow_tick_seconds < cadence.fast_tick_seconds:
        raise ConfigError("[cadence] slow_tick_seconds must be >= fast_tick_seconds")
    return cadence


def _load_strategy(s: dict) -> tuple[str, StrategySettings]:
    """Which strategy runs, and on what terms.

    The default is ``baseline``, not ``advisory``. Audit C6's finding is that
    the language model held order authority without measured predictive value,
    and a default *is* authority — it is what runs when nobody chose. Enabling
    the model is now an explicit line in this file, which is also what makes an
    honest A/B against the baseline possible.
    """
    kind = str(s.get("kind", "baseline")).lower()
    if kind not in _STRATEGIES:
        raise ConfigError(
            f"[strategy] kind must be one of {'|'.join(_STRATEGIES)}, got {kind!r}"
        )
    fields: dict[str, Any] = {
        **_floats(
            s,
            (
                "horizon_seconds",
                "entry_hurdle_pct",
                "flat_size_usd",
                "min_trade_usd",
                "rebalance_band_usd",
                "shrinkage",
                "interval_vol_multiple",
                "cash_fraction_per_entry",
            ),
        ),
        **_ints(s, ("max_positions",)),
    }
    try:
        settings = StrategySettings(**fields)
    except ValueError as exc:
        raise ConfigError(f"[strategy] {exc}") from exc
    return kind, settings


def _load_risk(r: dict, symbols: frozenset[str]) -> RiskParams:
    """Populate :class:`risk.RiskParams`.

    Two defaults here are not the dataclass's, and both are deliberate.

    ``universe`` defaults to the configured coins. ``RiskParams`` itself
    defaults to the empty set — correct for a library, since an unstated
    universe should permit nothing — but as a *startup* default it would mean a
    silent, total refusal to trade, which is a worse failure than a loud one.

    ``correlated_sleeve`` also defaults to the configured coins, because the
    audit's C10 finding is precisely that they are not three bets. BONK, WIF and
    POPCAT are one factor with three tickers; treating them as independent is
    how a 30% position cap becomes 90% of the book in the same trade.
    """
    fields: dict[str, Any] = {
        **_floats(
            r,
            (
                "max_snapshot_age_seconds",
                "min_liquidity_usd",
                "min_pool_age_seconds",
                "post_stop_quarantine_seconds",
                "min_seconds_between_entries",
                "max_position_pct",
                "stop_loss_pct",
                "max_gross_exposure_pct",
                "max_net_exposure_pct",
                "max_sleeve_exposure_pct",
                "min_cash_floor_pct",
                "max_cash_fraction_pct",
                "target_volatility_pct",
                "default_entry_usd",
                "max_entry_usd",
                "min_trade_usd",
                "max_price_impact_pct",
                "max_quote_age_seconds",
                "max_depth_participation_pct",
                "gas_usd_per_swap",
                "assumed_pool_fee_pct",
                "max_daily_loss_pct",
                "max_window_loss_pct",
                "loss_window_seconds",
                "max_drawdown_pct",
            ),
        ),
        **_ints(r, ("max_consecutive_failures",)),
        **_bools(r, ("require_known_pool_age", "require_volatility_estimate")),
    }
    fields["universe"] = frozenset(str(s).upper() for s in r.get("universe", symbols))
    fields["correlated_sleeve"] = frozenset(
        str(s).upper() for s in r.get("correlated_sleeve", symbols)
    )

    unknown = fields["universe"] - symbols
    if unknown:
        raise ConfigError(
            f"[risk] universe names {sorted(unknown)}, which are not configured [[coins]]"
        )

    # The same check on the sleeve, for the same reason. A typo here does not
    # fail loudly — it just puts a symbol that will never be held into the
    # correlated group, which quietly weakens the C10 sleeve cap with nothing
    # in the logs to say so.
    stray = fields["correlated_sleeve"] - symbols
    if stray:
        raise ConfigError(
            f"[risk] correlated_sleeve names {sorted(stray)}, which are not "
            f"configured [[coins]]"
        )

    try:
        params = RiskParams(**fields)
    except ValueError as exc:
        raise ConfigError(f"[risk] {exc}") from exc

    # The two fraction-valued fields, range-checked so a 30 typed where 0.30
    # was meant fails here rather than authorising a 3,000% position.
    if not 0 < params.max_position_pct <= 1.0:
        raise ConfigError("[risk] max_position_pct must be a fraction in (0, 1]")
    if not 0 < params.stop_loss_pct < 1.0:
        raise ConfigError("[risk] stop_loss_pct must be a fraction in (0, 1)")
    if params.default_entry_usd > params.max_entry_usd:
        raise ConfigError("[risk] default_entry_usd exceeds max_entry_usd")
    return params


def _load_mark(m: dict) -> MarkParams:
    fields: dict[str, Any] = {
        **_floats(
            m, ("mid_haircut_pct", "estimate_min_haircut_pct", "max_mark_age_seconds")
        ),
        **_bools(m, ("allow_mid_stop_reference",)),
    }
    try:
        return MarkParams(**fields)
    except ValueError as exc:
        raise ConfigError(f"[risk.mark] {exc}") from exc


def _load_execution(e: dict) -> ExecutionConfig:
    execution = ExecutionConfig(
        slippage_bps_fallback=float(e.get("slippage_bps_fallback", 50.0)),
        gas_usd_per_swap=float(e.get("gas_usd_per_swap", 0.21)),
        failed_tx_rate=float(e.get("failed_tx_rate", 0.06)),
        default_pool_fee_pct=float(e.get("default_pool_fee_pct", 0.25)),
        pool_fee_pct={str(k): float(v) for k, v in (e.get("pool_fee_pct") or {}).items()},
    )
    if not 0.0 <= execution.failed_tx_rate < 1.0:
        raise ConfigError("[execution] failed_tx_rate must be in [0, 1)")
    return execution


def _load_data(d: dict) -> DataConfig:
    return DataConfig(
        dexscreener_base=str(d.get("dexscreener_base", "https://api.dexscreener.com")),
        geckoterminal_base=str(
            d.get("geckoterminal_base", "https://api.geckoterminal.com/api/v2")
        ),
        jupiter_base=str(d.get("jupiter_base", "https://lite-api.jup.ag")),
        jupiter_base_keyed=str(d.get("jupiter_base_keyed", "https://api.jup.ag")),
        http_timeout_seconds=float(d.get("http_timeout_seconds", 15.0)),
        candles_5m=int(d.get("candles_5m", 100)),
        candles_1h=int(d.get("candles_1h", 100)),
        jupiter_api_key=os.environ.get("JUPITER_API_KEY") or None,
    )


def _load_market(m: dict, data: DataConfig) -> MarketParams:
    """Endpoints come from ``[data]``; tuning from ``[market]``.

    The split keeps a URL — which is an address, not a parameter — out of the
    object that gets varied in a sweep.
    """
    fields: dict[str, Any] = {
        "dexscreener_base": data.dexscreener_base,
        "geckoterminal_base": data.geckoterminal_base,
        "http_timeout_seconds": data.http_timeout_seconds,
        "candles_5m": data.candles_5m,
        "candles_1h": data.candles_1h,
    }
    fields.update(_floats(m, ("http_timeout_seconds",)))
    fields.update(_ints(m, ("candles_5m", "candles_1h", "max_mints_per_request")))
    try:
        return MarketParams(**fields)
    except ValueError as exc:
        raise ConfigError(f"[market] {exc}") from exc


def _load_safety(s: dict) -> SafetyScreenParams:
    fields: dict[str, Any] = {
        **_floats(
            s,
            (
                "min_liquidity_usd",
                "min_pool_age_seconds",
                "min_volume_24h_usd",
                "min_liquidity_to_fdv",
                "max_snapshot_age_seconds",
            ),
        ),
        **_bools(s, ("require_trusted_quote",)),
    }
    try:
        return SafetyScreenParams(**fields)
    except ValueError as exc:
        raise ConfigError(f"[safety_screen] {exc}") from exc


def _load_http(h: dict, legacy_timeout: float) -> HttpConfig:
    """Retry, backoff and breaker settings.

    ``[data] http_timeout_seconds`` still exists and still works; it now sets
    the *read* timeout only. A single scalar covering connect, read, write and
    pool is what lets one hung read consume a whole 60-second tick, which is
    why the four are separable.
    """
    timeout_fields = {
        "connect": 5.0,
        "read": legacy_timeout,
        "write": 10.0,
        "pool": 5.0,
    }
    for name in timeout_fields:
        key = f"{name}_timeout_seconds"
        if key in h:
            timeout_fields[name] = float(h[key])
    try:
        timeouts = Timeouts(**timeout_fields)
        retry_fields: dict[str, Any] = {
            **_ints(h, ("max_attempts",)),
            **_floats(
                h,
                (
                    "total_budget_seconds",
                    "backoff_base_seconds",
                    "backoff_multiplier",
                    "backoff_max_seconds",
                    "max_retry_after_seconds",
                ),
            ),
            **(
                {"total_budget_seconds": float(h["retry_budget_seconds"])}
                if "retry_budget_seconds" in h
                else {}
            ),
        }
        retry = RetryPolicy(**retry_fields)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"[http] {exc}") from exc

    # A retry budget longer than the tick it runs inside is a stall, not
    # resilience: the tick it belongs to has already been missed by the time
    # the last attempt returns.
    if retry.total_budget_seconds > 45.0:
        raise ConfigError(
            f"[http] retry_budget_seconds {retry.total_budget_seconds} is long enough "
            f"to consume a whole fast tick; lower it or raise [cadence] fast_tick_seconds"
        )

    return HttpConfig(
        timeouts=timeouts,
        retry=retry,
        breaker_failure_threshold=int(h.get("breaker_failure_threshold", 4)),
        breaker_cooldown_seconds=float(h.get("breaker_cooldown_seconds", 60.0)),
    )


def _load_sentiment(s: dict) -> SentimentConfig:
    """Reddit attention. Off by default after the audit.

    C7 found public Reddit text being interpolated into a prompt with order
    authority. The text path is gone, but the stream as a whole stays disabled
    until the ablation the audit specifies has actually been run — an input
    with no measured contribution should not be on by default merely because it
    was on yesterday.

    ``author_hash_key`` comes from the environment, never from this file. A
    salt committed alongside the hashes it salts is not a salt; a bare digest
    of a short Reddit username is reversible by enumeration in seconds.
    """
    return SentimentConfig(
        enabled=bool(s.get("enabled", False)),
        cache_ttl_seconds=int(s.get("cache_ttl_seconds", 600)),
        lookback_hours=int(s.get("lookback_hours", 24)),
        baseline_days=int(s.get("baseline_days", 7)),
        subreddits=tuple(str(x) for x in s.get("subreddits", [])),
        include_comments=bool(s.get("include_comments", True)),
        reddit_client_id=os.environ.get("REDDIT_CLIENT_ID") or None,
        reddit_client_secret=os.environ.get("REDDIT_CLIENT_SECRET") or None,
        reddit_user_agent=os.environ.get("REDDIT_USER_AGENT")
        or "memetrader/0.1 (paper trading research)",
        author_hash_key=os.environ.get("MEMETRADER_AUTHOR_HASH_KEY") or None,
    )
