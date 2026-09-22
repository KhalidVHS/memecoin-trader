"""Backtest run configuration, loaded from TOML.

Why TOML rather than YAML?  PyYAML is not a project dependency, and adding
one for a config format is not justified when the stdlib ``tomllib`` already
exists and the rest of this codebase (``config.py``) uses TOML.  The file
extension is ``.toml``; the ``configs/backtests/`` directory holds the starters.

Why a frozen dataclass rather than Pydantic?  The live ``Config`` uses frozen
dataclasses and the same pattern of module-local parameter objects.  Consistency
matters more than concision here: a reader who knows the live config can
understand this one without switching mental models.

``config_hash`` is a stable identifier for a specific configuration.  It is
used in the run manifest to detect config drift between two runs that claim to
be the same experiment, and to detect when a resumed run's config has changed
since it was started.  Two requirements:

1. Same config → same hash, always.  This rules out using Python's built-in
   ``hash()``, which is randomised across process restarts by PYTHONHASHSEED.
2. Different config → different hash (collision resistance), which rules out
   a checksum of the raw file bytes — two files with different whitespace but
   identical semantics should hash the same, and two files with different
   ``starting_cash_micro_usd`` but otherwise identical content should hash
   differently.

The approach: serialise the config to a dict in a canonical order (sorted keys
at every level), dump it as JSON (whose spec mandates key uniqueness), and
SHA-256 the UTF-8 bytes.  The first 16 hex characters are enough to distinguish
runs in a manifest without being unwieldy.

Latency model parameters live here rather than in ``execution/latency.py``
because they are part of the *run specification* — they are what a researcher
tunes to match observed shadow data — while the latency module itself is
infrastructure.  The same separation exists in the live config between
``[execution]`` (the numbers) and ``broker.py`` (the enforcement).

Cost-stress multipliers are 1x / 2x / 3x rather than a free float because the
purpose is a specific robustness test: "does the signal survive 3x costs?".  A
continuous parameter would require a different kind of sweep; the discrete
values are what the spec's ``execution_stress.toml`` documents.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from memetrader.types import ExecutionMode, FidelityTier


class BacktestConfigError(ValueError):
    """Raised when a backtest config file is invalid.  Always fatal at load time."""


# ---------------------------------------------------------------------------
# Latency model parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LatencyParams:
    """Parameters for the simulated submission latency distribution.

    The latency model (``execution/latency.py``) draws from a log-normal
    distribution whose shape is set by these two parameters.  They are
    expressed in log-space (``mu`` and ``sigma`` of the underlying normal)
    so that a researcher who has fitted the shadow data can paste the MLE
    estimates in directly without transformation.

    ``failed_tx_rate`` is the per-order probability of a transaction that is
    submitted but never lands.  On Solana, this rate is non-trivial at peak
    congestion; the default (0.06) matches the live ``[execution]`` config.
    """

    mu: float
    """Mean of the log-normal's underlying normal, in log-seconds."""

    sigma: float
    """Standard deviation of the log-normal's underlying normal, in log-seconds."""

    failed_tx_rate: float
    """Probability in [0, 1) that a submitted transaction does not land."""

    def __post_init__(self) -> None:
        import math

        for name, val in (("mu", self.mu), ("sigma", self.sigma)):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise BacktestConfigError(f"latency.{name} must be a number")
            if math.isnan(float(val)) or math.isinf(float(val)):
                raise BacktestConfigError(f"latency.{name} must be finite, got {val}")
        if self.sigma <= 0:
            raise BacktestConfigError(f"latency.sigma must be > 0, got {self.sigma}")
        if not 0.0 <= self.failed_tx_rate < 1.0:
            raise BacktestConfigError(
                f"latency.failed_tx_rate must be in [0, 1), got {self.failed_tx_rate}"
            )


# ---------------------------------------------------------------------------
# Walk-forward / fold parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FoldParams:
    """Time-interval-aware cross-validation parameters.

    All durations are in seconds.  They are intervals, not row counts, because
    assets in this universe have wildly different bar densities (BONK misses
    0.5% of 5 m bars; SLERF misses 86.5%).  Row-based splits would assign
    SLERF more wall-clock training history than BONK for the same fold index.

    ``embargo_seconds`` covers max holding horizon + publication delay +
    rolling-state carryover, as the spec requires.  It is the minimum gap
    between the last training label and the first validation observation, so
    that a feature computed from a training bar cannot appear in validation.

    ``holdout_seconds`` and ``holdout_min_pct`` implement the spec's
    ``max(45 days, 20%)`` policy.  The default of 45 days is below the plan's
    90-day recommendation because the full history is only ~209 days of 1h
    data; the shortfall is stated in the warning below and in every run's
    manifest — it is not hidden.
    """

    train_seconds: float
    val_seconds: float
    test_seconds: float
    roll_seconds: float
    embargo_seconds: float
    n_outer_folds: int
    holdout_seconds: float = 45 * 86_400
    """
    45 days is the spec's minimum.  The plan recommends 90 days but that is
    43% of the 209-day record.  When the configured value is below 90 days,
    the runner emits a loud warning: the shortfall is stated, not hidden.
    """
    holdout_min_pct: float = 0.20
    """Holdout must be at least this fraction of the total record length."""

    def __post_init__(self) -> None:
        import math

        for name in (
            "train_seconds",
            "val_seconds",
            "test_seconds",
            "roll_seconds",
            "embargo_seconds",
            "holdout_seconds",
            "holdout_min_pct",
        ):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise BacktestConfigError(f"folds.{name} must be a number")
            if math.isnan(float(v)) or math.isinf(float(v)):
                raise BacktestConfigError(f"folds.{name} must be finite, got {v}")
            if v < 0:
                raise BacktestConfigError(f"folds.{name} must be >= 0, got {v}")
        if not isinstance(self.n_outer_folds, int) or isinstance(
            self.n_outer_folds, bool
        ):
            raise BacktestConfigError("folds.n_outer_folds must be an int")
        if self.n_outer_folds < 1:
            raise BacktestConfigError(
                f"folds.n_outer_folds must be >= 1, got {self.n_outer_folds}"
            )
        if not 0 < self.holdout_min_pct < 1:
            raise BacktestConfigError(
                f"folds.holdout_min_pct must be in (0, 1), got {self.holdout_min_pct}"
            )


# ---------------------------------------------------------------------------
# BacktestConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Complete specification for one backtest run.

    Fields are grouped into logical sections that mirror the TOML structure.
    Every field has a type annotation; no field is ``Any``.

    ``label`` is the human-readable name for the run.  It appears in the
    manifest and in report filenames.  It does not need to be unique (the
    run ID is unique); it should be memorable.

    ``universe_version`` identifies which snapshot of the asset universe the
    run uses.  Point-in-time correctness depends on this: a universe that
    includes tokens that were not yet listed at the run's start date is
    survivorship bias by another name.

    ``timeframes`` is the set of bar resolutions the run loads.  The default
    is both ``5m`` and ``1h``, matching the live system.

    ``fast_tick_seconds`` and ``decision_tick_seconds`` mirror the live
    system's cadence (``CadenceConfig``).  The decision tick is when the
    strategy runs; the fast tick is when execution and quote refreshes happen.
    They must be equal or the decision tick must be a multiple of the fast
    tick — the engine steps the fast tick N times per decision tick.

    ``starting_cash_micro_usd`` is in micro-USD (1 USD = 1_000_000 µUSD)
    matching ``LocalPaperBroker.cash_micro_usd``.  The default corresponds
    to $1,000.

    ``publication_delay_seconds`` is subtracted from every bar's
    ``available_time`` by the loader.  It models the delay between a candle
    closing at the exchange and GeckoTerminal publishing it in their API.
    Default is 0 (conservative: assume publication is instantaneous).

    ``seeds`` is a mapping of RNG stream name to integer seed.  Any stream
    not listed uses ``SimulatedClock.deterministic_seed(stream_name)``.

    ``cost_stress_multipliers`` are applied one run at a time by the runner.
    The canonical set is (1, 2, 3) matching the spec's "1x/2x/3x" test.
    """

    # ------------------------------------------------------------------ #
    # Identity                                                             #
    # ------------------------------------------------------------------ #
    label: str
    run_id_prefix: str
    """Short prefix prepended to the generated run ID, e.g. ``"baseline"``."""

    # ------------------------------------------------------------------ #
    # Date range                                                           #
    # ------------------------------------------------------------------ #
    start_ts: float
    """Replay start as epoch seconds.  The first event yielded by any loader
    must have ``available_time >= start_ts``."""
    end_ts: float
    """Replay end as epoch seconds (exclusive).  Events after this are dropped."""

    # ------------------------------------------------------------------ #
    # Data                                                                 #
    # ------------------------------------------------------------------ #
    universe_version: str
    timeframes: tuple[str, ...]

    # ------------------------------------------------------------------ #
    # Cadence (mirrors live CadenceConfig)                                 #
    # ------------------------------------------------------------------ #
    fast_tick_seconds: int
    """How often the engine steps the clock.  Matches live fast_tick_seconds."""
    decision_tick_seconds: int
    """How often the strategy runs.  Must be a multiple of fast_tick_seconds."""

    # ------------------------------------------------------------------ #
    # Portfolio                                                            #
    # ------------------------------------------------------------------ #
    starting_cash_micro_usd: int
    """Starting portfolio cash in micro-USD.  1 USD = 1_000_000."""

    # ------------------------------------------------------------------ #
    # Point-in-time correctness                                            #
    # ------------------------------------------------------------------ #
    publication_delay_seconds: float
    """Added to bar_close_ts + interval to derive available_time.  0 = no delay."""

    # ------------------------------------------------------------------ #
    # Execution model                                                      #
    # ------------------------------------------------------------------ #
    execution_model: str
    """Which ExecutionModel implementation to load.  E.g. ``"ohlcv_midpoint"``."""
    execution_mode: ExecutionMode
    fidelity_tier: FidelityTier
    latency: LatencyParams

    # ------------------------------------------------------------------ #
    # Folds / splits                                                       #
    # ------------------------------------------------------------------ #
    folds: FoldParams

    # ------------------------------------------------------------------ #
    # Holdout policy                                                       #
    # ------------------------------------------------------------------ #
    holdout_opened: bool
    """Whether the holdout set has been inspected.  Once True, the experiment
    is locked — revising results after seeing the holdout invalidates it.
    The registry enforces this; this flag is the on-disk record."""

    # ------------------------------------------------------------------ #
    # Reproducibility                                                      #
    # ------------------------------------------------------------------ #
    seeds: dict[str, int]
    """Per-stream seeds.  Streams absent from this dict are seeded from the
    run ID via ``SimulatedClock.deterministic_seed``."""

    # ------------------------------------------------------------------ #
    # Cost stress                                                          #
    # ------------------------------------------------------------------ #
    cost_stress_multipliers: tuple[int, ...]
    """The multipliers to apply in successive stress runs.  Canonical: (1, 2, 3)."""

    # ------------------------------------------------------------------ #
    # Derived                                                              #
    # ------------------------------------------------------------------ #

    @property
    def config_hash(self) -> str:
        """Stable 32-hex-char SHA-256 prefix of the canonicalised config.

        "Stable" means: the same ``BacktestConfig`` always produces the same
        hash, regardless of Python version, PYTHONHASHSEED, or process restart.
        We achieve this by serialising through ``json.dumps`` with
        ``sort_keys=True``, which is deterministic for the types we use
        (str, int, float, bool, list, dict).  ``float`` is represented in
        JSON with full precision via ``repr``-equivalent formatting.

        The hash changes when *any* field changes, including ``label`` — two
        configs that differ only in their human-readable label are different
        configs, and manifests that share a hash must be identical in all ways
        that affect the economics.
        """
        canonical = json.dumps(
            _config_to_canonical_dict(self),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            # allow_nan=False is the default; we never store NaN in config.
        )
        digest = hashlib.sha256(canonical.encode("ascii")).hexdigest()
        return digest[:32]

    def validate(self) -> None:
        """Cross-field validation that cannot be expressed per-field.

        Called automatically by ``load()``.  Callers constructing
        ``BacktestConfig`` directly (e.g. in tests) should call this
        explicitly if they want the same guarantees.
        """
        import math

        for name, v in (("start_ts", self.start_ts), ("end_ts", self.end_ts)):
            if math.isnan(v) or math.isinf(v):
                raise BacktestConfigError(f"{name} must be finite")
        if self.end_ts <= self.start_ts:
            raise BacktestConfigError(
                f"end_ts {self.end_ts} must be > start_ts {self.start_ts}"
            )
        if self.fast_tick_seconds < 5:
            raise BacktestConfigError(
                f"fast_tick_seconds {self.fast_tick_seconds} below minimum 5"
            )
        if self.decision_tick_seconds < self.fast_tick_seconds:
            raise BacktestConfigError(
                "decision_tick_seconds must be >= fast_tick_seconds"
            )
        if self.decision_tick_seconds % self.fast_tick_seconds != 0:
            raise BacktestConfigError(
                f"decision_tick_seconds {self.decision_tick_seconds} must be a "
                f"multiple of fast_tick_seconds {self.fast_tick_seconds}"
            )
        if self.starting_cash_micro_usd <= 0:
            raise BacktestConfigError(
                f"starting_cash_micro_usd must be > 0, got {self.starting_cash_micro_usd}"
            )
        if self.publication_delay_seconds < 0:
            raise BacktestConfigError(
                f"publication_delay_seconds must be >= 0, "
                f"got {self.publication_delay_seconds}"
            )
        if not self.timeframes:
            raise BacktestConfigError("timeframes must list at least one resolution")
        if not self.cost_stress_multipliers:
            raise BacktestConfigError(
                "cost_stress_multipliers must contain at least one value"
            )
        for m in self.cost_stress_multipliers:
            if not isinstance(m, int) or m < 1:
                raise BacktestConfigError(
                    f"cost_stress_multipliers must be positive ints, got {m!r}"
                )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load(path: Path) -> BacktestConfig:
    """Load and validate a backtest config from a TOML file.

    Raises ``BacktestConfigError`` on any structural or semantic error.
    The error message names the offending field so the caller does not need to
    inspect a traceback to fix a typo.
    """
    if not path.exists():
        raise BacktestConfigError(f"backtest config not found: {path}")

    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    try:
        cfg = _parse(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise BacktestConfigError(
            f"backtest config {path} is invalid: {exc}"
        ) from exc

    cfg.validate()
    return cfg


def _require(table: dict, key: str, section: str) -> object:
    if key not in table:
        raise BacktestConfigError(
            f"backtest config is missing [{section}] {key!r}"
        )
    return table[key]


def _parse(raw: dict) -> BacktestConfig:  # noqa: PLR0912, PLR0914, PLR0915
    """Convert a raw TOML dict to a ``BacktestConfig``.

    This function is intentionally explicit rather than generic.  A generic
    approach (``dacite``, Pydantic) would hide where each field comes from,
    making it harder to confirm that the TOML structure matches the dataclass
    structure.  The verbosity is the documentation.
    """
    # ------------------------------------------------------------------ #
    # [run]                                                                #
    # ------------------------------------------------------------------ #
    run = raw.get("run", {})
    label = str(_require(run, "label", "run"))
    run_id_prefix = str(run.get("run_id_prefix", label[:16].replace(" ", "_")))

    # ------------------------------------------------------------------ #
    # [date_range]                                                         #
    # ------------------------------------------------------------------ #
    dr = raw.get("date_range", {})
    start_ts = float(_require(dr, "start_ts", "date_range"))
    end_ts = float(_require(dr, "end_ts", "date_range"))

    # ------------------------------------------------------------------ #
    # [data]                                                               #
    # ------------------------------------------------------------------ #
    data = raw.get("data", {})
    universe_version = str(_require(data, "universe_version", "data"))
    timeframes = tuple(str(t) for t in data.get("timeframes", ["5m", "1h"]))

    # ------------------------------------------------------------------ #
    # [cadence]                                                            #
    # ------------------------------------------------------------------ #
    cad = raw.get("cadence", {})
    fast_tick_seconds = int(cad.get("fast_tick_seconds", 300))
    decision_tick_seconds = int(cad.get("decision_tick_seconds", 900))

    # ------------------------------------------------------------------ #
    # [portfolio]                                                          #
    # ------------------------------------------------------------------ #
    port = raw.get("portfolio", {})
    # Accept either micro_usd directly or a USD float (converted here).
    if "starting_cash_micro_usd" in port:
        starting_cash_micro_usd = int(port["starting_cash_micro_usd"])
    elif "starting_cash_usd" in port:
        starting_cash_micro_usd = int(float(port["starting_cash_usd"]) * 1_000_000)
    else:
        starting_cash_micro_usd = 1_000 * 1_000_000  # $1,000 default

    # ------------------------------------------------------------------ #
    # [execution]                                                          #
    # ------------------------------------------------------------------ #
    exc = raw.get("execution", {})
    publication_delay_seconds = float(exc.get("publication_delay_seconds", 0.0))
    execution_model = str(exc.get("execution_model", "ohlcv_midpoint"))

    mode_raw = str(exc.get("execution_mode", "paper")).lower()
    try:
        execution_mode = ExecutionMode(mode_raw)
    except ValueError as err:
        allowed = "|".join(m.value for m in ExecutionMode)
        raise BacktestConfigError(
            f"[execution] execution_mode must be one of {allowed}, got {mode_raw!r}"
        ) from err

    tier_raw = str(exc.get("fidelity_tier", "tier_0_ohlcv"))
    try:
        fidelity_tier = FidelityTier(tier_raw)
    except ValueError as err:
        allowed = "|".join(t.value for t in FidelityTier)
        raise BacktestConfigError(
            f"[execution] fidelity_tier must be one of {allowed}, got {tier_raw!r}"
        ) from err

    lat = exc.get("latency", {})
    latency = LatencyParams(
        mu=float(lat.get("mu", -1.5)),
        sigma=float(lat.get("sigma", 0.4)),
        failed_tx_rate=float(lat.get("failed_tx_rate", 0.06)),
    )

    # ------------------------------------------------------------------ #
    # [folds]                                                              #
    # ------------------------------------------------------------------ #
    fol = raw.get("folds", {})
    folds = FoldParams(
        train_seconds=float(fol.get("train_seconds", 90 * 86_400)),
        val_seconds=float(fol.get("val_seconds", 30 * 86_400)),
        test_seconds=float(fol.get("test_seconds", 30 * 86_400)),
        roll_seconds=float(fol.get("roll_seconds", 30 * 86_400)),
        embargo_seconds=float(fol.get("embargo_seconds", 2 * 86_400)),
        n_outer_folds=int(fol.get("n_outer_folds", 3)),
        holdout_seconds=float(fol.get("holdout_seconds", 45 * 86_400)),
        holdout_min_pct=float(fol.get("holdout_min_pct", 0.20)),
    )

    # ------------------------------------------------------------------ #
    # [holdout]                                                            #
    # ------------------------------------------------------------------ #
    holdout = raw.get("holdout", {})
    holdout_opened = bool(holdout.get("opened", False))

    # ------------------------------------------------------------------ #
    # [seeds]                                                              #
    # ------------------------------------------------------------------ #
    seeds_raw = raw.get("seeds", {})
    seeds: dict[str, int] = {}
    for stream_name, seed_val in seeds_raw.items():
        try:
            seeds[str(stream_name)] = int(seed_val)
        except (TypeError, ValueError) as err:
            raise BacktestConfigError(
                f"[seeds] {stream_name!r} must be an integer, got {seed_val!r}"
            ) from err

    # ------------------------------------------------------------------ #
    # [cost_stress]                                                        #
    # ------------------------------------------------------------------ #
    cs = raw.get("cost_stress", {})
    multipliers_raw = cs.get("multipliers", [1, 2, 3])
    cost_stress_multipliers = tuple(int(m) for m in multipliers_raw)

    return BacktestConfig(
        label=label,
        run_id_prefix=run_id_prefix,
        start_ts=start_ts,
        end_ts=end_ts,
        universe_version=universe_version,
        timeframes=timeframes,
        fast_tick_seconds=fast_tick_seconds,
        decision_tick_seconds=decision_tick_seconds,
        starting_cash_micro_usd=starting_cash_micro_usd,
        publication_delay_seconds=publication_delay_seconds,
        execution_model=execution_model,
        execution_mode=execution_mode,
        fidelity_tier=fidelity_tier,
        latency=latency,
        folds=folds,
        holdout_opened=holdout_opened,
        seeds=seeds,
        cost_stress_multipliers=cost_stress_multipliers,
    )


# ---------------------------------------------------------------------------
# Canonical dict for hashing
# ---------------------------------------------------------------------------


def _config_to_canonical_dict(cfg: BacktestConfig) -> dict:
    """Return a JSON-serialisable dict with no non-deterministic types.

    ``asdict`` from ``dataclasses`` handles nested dataclasses recursively.
    The result contains only Python primitives (str, int, float, bool, list,
    dict) which ``json.dumps(sort_keys=True)`` serialises deterministically.

    We convert tuples to lists (JSON has no tuple) and enums to their string
    value.  The conversion must be the same every time, which it is because the
    traversal order of a dict is insertion order (Python 3.7+) and
    ``json.dumps(sort_keys=True)`` always sorts.
    """
    raw = asdict(cfg)
    return _normalise(raw)


def _normalise(obj: object) -> object:
    """Recursively convert enums, tuples, and non-JSON-native types."""
    if isinstance(obj, dict):
        return {str(k): _normalise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalise(v) for v in obj]
    # StrEnum values are strings; asdict preserves the enum value attribute.
    # dataclasses.asdict already calls value on enum members via
    # the default copy logic, but let's be explicit.
    if hasattr(obj, "value") and isinstance(obj.value, str):
        return obj.value
    return obj
