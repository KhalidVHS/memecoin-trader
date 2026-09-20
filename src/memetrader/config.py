"""Parse ``config.toml`` plus the environment into a frozen, validated object.

Everything that could be a magic number lives here. Validation happens once at
startup and fails loudly — a bad config should never surface as a weird fill
three hours into a run.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(ValueError):
    """Raised at startup when config.toml is wrong. Always fatal."""


#: Ordered cheapest to most expensive, which is also least to most thinking.
#: Kept as a tuple rather than a set so the error message lists them in an
#: order that tells the reader which direction costs money.
_EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")


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
class RiskConfig:
    max_position_pct: float  # fraction, 0.30 = 30% of book
    stop_loss_pct: float  # fraction, 0.15 = -15% from entry
    min_trade_usd: float
    max_price_impact_pct: float  # whole percent
    max_snapshot_age_seconds: float
    min_liquidity_usd: float


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
        """
        if not route_labels:
            return self.default_pool_fee_pct
        return sum(
            self.pool_fee_pct.get(label, self.default_pool_fee_pct)
            for label in route_labels
        )


@dataclass(frozen=True, slots=True)
class DataConfig:
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
class SentimentConfig:
    enabled: bool
    cache_ttl_seconds: int
    lookback_hours: int
    baseline_days: int
    subreddits: tuple[str, ...]
    reddit_client_id: str | None
    reddit_client_secret: str | None
    reddit_user_agent: str
    # Submissions alone measured almost nothing: across the five subreddits
    # below, the 90 days ending 2026-09-19 held 23 BONK submissions, newest 14.7
    # days old. Comments outrun submissions 7.4:1 on the same subreddits
    # (136 vs >=1008 over the 24h ending 2026-09-20 00:13 UTC), and a comment
    # sample contained a BONK mention no submission did. What that buys is a
    # credible zero rather than more mentions — the same window matched 0/0/0
    # coins in submissions and 1/0/0 in comments. It roughly doubles the
    # request count against a host that rate-limits, hence the switch.
    #
    # Defaulted, unlike its neighbours, because a required field here would
    # break every positional ``SentimentConfig(...)`` in the tests — which is
    # exactly how the ``ModelConfig`` cache-write field broke them once already.
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
    model: ModelConfig
    cadence: CadenceConfig
    risk: RiskConfig
    execution: ExecutionConfig
    data: DataConfig
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
    def decisions_path(self) -> Path:
        return self.data_dir / "decisions.jsonl"


def _require(table: dict, key: str, where: str):
    if key not in table:
        raise ConfigError(f"config.toml is missing [{where}] {key}")
    return table[key]


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

    coins_raw = raw.get("coins") or []
    if len(coins_raw) < 1:
        raise ConfigError("config.toml must define at least one [[coins]] entry")
    coins = []
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

    m = raw.get("model", {})
    effort = str(m.get("effort", "high")).lower()
    # The full set anthropic 1.7.0's OutputConfigParam accepts, read from the
    # installed package on 2026-09-19 rather than from docs. Deliberately not
    # narrower: a value the API would accept should not fail here as though it
    # were a typo. The cost consequence is real and is stated in the README —
    # the $180-250/month figure was calibrated at "high", and effort is the
    # single largest lever on it, because it is thinking tokens that move.
    if effort not in _EFFORT_LEVELS:
        raise ConfigError(
            f"[model] effort must be one of {'|'.join(_EFFORT_LEVELS)}, got {effort!r}"
        )
    model = ModelConfig(
        name=str(m.get("name", "claude-opus-5")),
        effort=effort,
        max_tokens=int(m.get("max_tokens", 8000)),
        price_input_per_mtok=float(m.get("price_input_per_mtok", 5.0)),
        price_output_per_mtok=float(m.get("price_output_per_mtok", 25.0)),
        price_cache_read_per_mtok=float(m.get("price_cache_read_per_mtok", 0.5)),
        price_cache_write_per_mtok=float(m.get("price_cache_write_per_mtok", 6.25)),
    )

    c = raw.get("cadence", {})
    cadence = CadenceConfig(
        fast_tick_seconds=int(c.get("fast_tick_seconds", 60)),
        slow_tick_seconds=int(c.get("slow_tick_seconds", 900)),
    )
    if cadence.fast_tick_seconds < 5:
        raise ConfigError("[cadence] fast_tick_seconds below 5 will hammer the APIs")
    if cadence.slow_tick_seconds < cadence.fast_tick_seconds:
        raise ConfigError("[cadence] slow_tick_seconds must be >= fast_tick_seconds")

    r = raw.get("risk", {})
    risk = RiskConfig(
        max_position_pct=float(r.get("max_position_pct", 0.30)),
        stop_loss_pct=float(r.get("stop_loss_pct", 0.15)),
        min_trade_usd=float(r.get("min_trade_usd", 10.0)),
        max_price_impact_pct=float(r.get("max_price_impact_pct", 3.0)),
        max_snapshot_age_seconds=float(r.get("max_snapshot_age_seconds", 90)),
        min_liquidity_usd=float(r.get("min_liquidity_usd", 50_000.0)),
    )
    if not 0 < risk.max_position_pct <= 1.0:
        raise ConfigError("[risk] max_position_pct must be a fraction in (0, 1]")
    if not 0 < risk.stop_loss_pct < 1.0:
        raise ConfigError("[risk] stop_loss_pct must be a fraction in (0, 1)")

    e = raw.get("execution", {})
    execution = ExecutionConfig(
        slippage_bps_fallback=float(e.get("slippage_bps_fallback", 50.0)),
        gas_usd_per_swap=float(e.get("gas_usd_per_swap", 0.21)),
        failed_tx_rate=float(e.get("failed_tx_rate", 0.06)),
        default_pool_fee_pct=float(e.get("default_pool_fee_pct", 0.25)),
        pool_fee_pct={
            str(k): float(v) for k, v in (e.get("pool_fee_pct") or {}).items()
        },
    )
    if not 0.0 <= execution.failed_tx_rate < 1.0:
        raise ConfigError("[execution] failed_tx_rate must be in [0, 1)")

    d = raw.get("data", {})
    data = DataConfig(
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

    s = raw.get("sentiment", {})
    sentiment = SentimentConfig(
        enabled=bool(s.get("enabled", True)),
        cache_ttl_seconds=int(s.get("cache_ttl_seconds", 600)),
        lookback_hours=int(s.get("lookback_hours", 24)),
        baseline_days=int(s.get("baseline_days", 7)),
        subreddits=tuple(str(x) for x in s.get("subreddits", [])),
        include_comments=bool(s.get("include_comments", True)),
        reddit_client_id=os.environ.get("REDDIT_CLIENT_ID") or None,
        reddit_client_secret=os.environ.get("REDDIT_CLIENT_SECRET") or None,
        reddit_user_agent=os.environ.get("REDDIT_USER_AGENT")
        or "memetrader/0.1 (paper trading research)",
    )

    p = raw.get("prompt", {})
    prompt = PromptConfig(decision_history=int(p.get("decision_history", 10)))

    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        root=root,
        data_dir=data_dir,
        starting_cash_usd=float(
            raw.get("portfolio", {}).get("starting_cash_usd", 1000.0)
        ),
        coins=tuple(coins),
        model=model,
        cadence=cadence,
        risk=risk,
        execution=execution,
        data=data,
        sentiment=sentiment,
        prompt=prompt,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
    )
