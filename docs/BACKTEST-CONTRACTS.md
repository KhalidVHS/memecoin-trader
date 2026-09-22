# Backtest engine — frozen contracts

Every module in the backtest tree is written against this document. If an
implementation needs a signature that is not here, the signature is wrong or
this document is incomplete — resolve it here first, do not invent a local
variant. Interface drift between independently written modules is the specific
failure this file exists to prevent.

Read alongside `docs/CANNOT-REPLAY.md`, which records what the data cannot do.

---

## 0. Non-negotiable conventions (inherited from `types.py`)

| Convention | Rule |
|---|---|
| Time | epoch **seconds** as `float`, always. Never ms, never `datetime` in a record. |
| Percentages | whole numbers. `-4.2` means −4.2%. Exception: config fields documented as fractions (`max_position_pct`, `stop_loss_pct`). |
| Token amounts | `int`, atomic units. A float quantity is a rounding error waiting to be called a fill. |
| Cash | `int` **micro-USD** inside the ledger (matches `LocalPaperBroker.cash_micro_usd`). Floats are presentation only. |
| Missing | `None` = "could not find out". `0` = "looked, and it is quiet". Never collapse them. |
| Validation | every numeric field validated finite at construction, via `types.finite` / `finite_or_none` / `positive` / `non_negative` / `atomic`. |
| Modules | `from __future__ import annotations` at the top of every file. `X | None`, not `Optional[X]`. |

---

## 1. The point-in-time invariant

> At simulated time `t`, every component may access only records whose
> `available_time <= t`.

`Provenance` now carries `available_time: float | None` and an `available_at`
property that falls back to `receive_time`. `HistoricalEvent.sort_key` orders
the replay queue and **deliberately does not contain `event_time`** — sorting a
replay by when things happened rather than by when they were knowable is the
most common way a backtest grants itself foresight, and the resulting equity
curve looks entirely plausible.

Publication-delay rules every loader must apply:

- **A candle is unavailable until `ts + interval + publication_delay`.** Never
  at `ts`. The default delay is a config value, not a magic number.
- **Social data is available at collector receipt time**, not post creation
  time.
- **Token metadata changes are timestamped**; a rename is not retroactive.
- **Universe membership is point-in-time**, including dead and delisted tokens.
- **Labels carry `label_start_ts` and `label_end_ts`**, so purging can see the
  full interval an observation depends on.

---

## 2. Layout, and why it deviates from the source plan

The plan was written without the repository tree. Three deviations, each
forced by something real:

1. **No `domain/` package.** `types.py` already owns `Forecast`,
   `TargetPosition`, `OrderIntent`, `Fill`, `Quote`, `Position`,
   `PortfolioState`, `Provenance`, `Observed[T]`, `OrderState`. A second domain
   layer *is* the backtest/live divergence the plan warns against, so the new
   records (`HistoricalEvent`, `CostBreakdown`, `OrderReceipt`,
   `ExecutionReport`, `FidelityTier`, `EventKind`) were added **to `types.py`**.

2. **`histdata/`, not `data/`.** `.gitignore` line 2 was an unanchored
   `data/`, which matches a directory of that name at any depth and silently
   swallowed `src/memetrader/data/`. The pattern is now anchored to `/data/`,
   but the package is still named `histdata` because `Config.data_dir` already
   means `root/data` and two things called "data" in one namespace is a bug
   waiting to happen.

3. **`strategies/` (plural) and `construction.py`.** `strategy.py` and
   `portfolio.py` already exist as modules and already own the `Strategy`
   protocol and `mark_book`. They stay. New baselines go in `strategies/`,
   portfolio construction in `construction.py`.

```text
src/memetrader/
  types.py          [MODIFIED] contracts live here
  strategy.py       [UNCHANGED] owns Strategy protocol + BaselineStrategy
  portfolio.py      [UNCHANGED] owns mark_book, stop_loss_breaches
  risk.py           [UNCHANGED] owns RiskEngine
  broker.py         [UNCHANGED] owns LocalPaperBroker
  journal.py        [UNCHANGED] owns Ledger
  histdata/   schemas catalog quality point_in_time universe loaders/*
  features/   registry pipeline market microstructure onchain attention
  strategies/ baselines
  construction.py
  execution/  interfaces costs latency fill_models amm/*
  backtest/   config clock event_queue engine broker ledger snapshots invariants runner
  validation/ splits walk_forward leakage recursive bootstrap multiple_testing ablation promotion
  metrics/    performance attribution benchmarks capacity
  experiments/ manifest registry
  shadow.py         prospective quote-ladder + pool-state collector
  cli_backtest.py   typer sub-app, registered into cli.app
```

### Dependency decisions (the plan offered options; these are the picks)

| Option | Decision | Reason |
|---|---|---|
| NautilusTrader as engine | **No** — borrow architecture only | LGPL-3.0, no Jupiter/Solana adapter, v2 is pre-production |
| vectorbt for research tier | **No** | Apache-2.0 **with Commons Clause** — commercial-use restriction |
| Freqtrade modules | **No** — reimplement lookahead/recursive *concepts* | GPL-3.0 |
| skfolio for splits | **No** | its purge/embargo are observation counts; we need interval-aware across irregular multi-asset data. The plan says so itself. |
| `arch` for SPA/bootstrap | **No** — numpy-native | pulls scipy + statsmodels; SPA via stationary bootstrap is ~80 lines and more testable |
| DuckDB catalog | **No** | `pyarrow.dataset` already gives partition discovery and predicate pushdown; SQL convenience is not worth the dep |
| Parquet artifacts | **Yes** — `pyarrow` added | columnar is genuinely right for fold/fill/equity analysis |
| Property tests | **Yes** — `hypothesis` added (dev) | required for accounting and AMM math |

**Storage split:** append-only streams (ledger, execution reports) are gzipped
JSONL — streaming, crash-safe, matches `journal.py`. Immutable analysis
artifacts written once at run end are Parquet.

---

## 3. Fidelity tiers — and what we actually have

`FidelityTier` in `types.py`. **This repository is at TIER_0**: OHLCV only, 24
coins, 1h + 5m, bounded by the vendor's 180-day public horizon.

- `FidelityTier.permits_pnl_claim` is `True` only for TIER_2/TIER_3.
- `validation.promotion` **hard-fails** below TIER_2.
- Any report from a sub-TIER_2 run must print `types.NON_EXECUTABLE_NOTICE`
  verbatim. It is a module constant so no report can quietly soften it.

This is why `shadow.py` matters more than anything else in the tree: every day
we do not collect quote ladders is a day of executable history that cannot be
recovered later. Jupiter is a live routing service, not an archive.

---

## 4. Frozen interfaces

Reuse the **existing** live engines. Do not write backtest copies.

```python
# EXISTING — src/memetrader/risk.py
RiskEngine(params: RiskParams = DEFAULT_PARAMS)
  .entry_bounds(symbol, *, book, risk_state, snapshot, quote=None, valuation=None,
                forecast=None, volatility_pct=None, ledger=EMPTY_LEDGER, now) -> RiskBounds
  .exit_bounds(symbol, *, book, risk_state, snapshot=None, forced=False, now) -> RiskBounds
  .confirm_quote(bounds, quote, *, notional_usd, now) -> RiskBounds
ContinuousRisk.evaluate(*, book, ledger=EMPTY_LEDGER, data_quality_ok=True, now) -> RiskState
update_ledger(ledger, *, book, fills=(), risk_events=(), params=DEFAULT_PARAMS, now) -> RiskLedger

# EXISTING — src/memetrader/portfolio.py
mark_book(*, cash_usd, positions, marks, realized_pnl_usd, starting_cash_usd,
          fees_paid_usd=0.0, gas_paid_usd=0.0, now) -> PortfolioState
build_mark(symbol, *, route=None, mid_price_usd=None, mid_provenance=None,
           estimate=None, params=DEFAULT_MARK_PARAMS, now) -> Mark
stop_loss_breaches(state, stop_loss_pct, *, params=DEFAULT_MARK_PARAMS) -> tuple[StopBreach, ...]

# EXISTING — src/memetrader/strategy.py
class Strategy(Protocol):
    strategy_id: str
    def decide(self, evidence: dict[str, EvidenceBundle], portfolio: PortfolioState,
               *, now: float) -> StrategyDecision: ...

# EXISTING — src/memetrader/ids.py
new_run_id/new_decision_id/new_action_id/new_intent_id/new_order_id/new_fill_id() -> str
quote_fingerprint(*, side, input_mint, output_mint, in_amount_atomic,
                  out_amount_atomic, slot) -> str
```

### New protocols

```python
# histdata/point_in_time.py — the only legal way to read history during a replay
class PointInTimeState(Protocol):
    now: float
    def bars(self, asset_id: str, timeframe: Timeframe, *, lookback: int
             ) -> tuple[Candle, ...]: ...
    def snapshot(self, asset_id: str) -> CoinSnapshot | None: ...
    def universe(self) -> frozenset[str]: ...
    def pool_state(self, pool_id: str) -> PoolState | None: ...
    def quote_ladder(self, asset_id: str, side: Side) -> QuoteLadder | None: ...
```
Every method returns **only** records with `available_at <= self.now`, and
every `Candle` returned satisfies `closed is True`. A forming bar must never
escape this boundary.

```python
# execution/interfaces.py
@dataclass(frozen=True, slots=True)
class ApprovedOrder:
    intent: OrderIntent
    bounds: RiskBounds
    quote: Quote
    decided_at: float

class ExecutionModel(Protocol):
    """NOTE: named ExecutionModel, not FillModel — broker.py already has a
    `FillModel` enum (MIN_OUT / EXPECTED_OUT) and the names must not collide."""
    @property
    def fidelity(self) -> FidelityTier: ...
    def price(self, *, intent: OrderIntent, state: PointInTimeState,
              now: float) -> Quote | None: ...
    def fill(self, *, order: ApprovedOrder, state: PointInTimeState,
             now: float) -> ExecutionReport: ...

class ExecutionVenue(Protocol):
    def submit(self, order: ApprovedOrder, now: float) -> OrderReceipt: ...
    def process_event(self, event: HistoricalEvent) -> list[ExecutionReport]: ...
```

```python
# construction.py
class PortfolioConstructor(Protocol):
    def build_targets(self, forecasts: Sequence[Forecast], portfolio: PortfolioState,
                      market: PointInTimeState) -> tuple[TargetPosition, ...]: ...
```

---

## 5. The tick order the engine must mirror

Taken from `loop.Trader.slow_tick` — the backtest must reproduce this exact
sequence or it is not testing the deployed system:

1. Snapshot (point-in-time).
2. `portfolio.mark_book` → `PortfolioState`.
3. `risk.update_ledger` → `ContinuousRisk.evaluate` → `RiskState`.
4. `portfolio.stop_loss_breaches` → `risk.exit_bounds(forced=True)` → execute.
   **Stops run before strategy**, every tick.
5. If stops fired: re-mark and re-evaluate risk.
6. Build `EvidenceBundle` per symbol.
7. If `risk_state.halted`: return — exits only, no entries.
8. `strategy.decide(...)` → `StrategyDecision`.
9. Diff targets vs inventory, filtered by `rebalance_band_usd`; **SELLs before
   BUYs**; per symbol: `entry_bounds`/`exit_bounds` → price → `confirm_quote`
   → append intent → submit → report → append fill + state.
10. Append `DecisionRecord`.

**A decision from a bar close must not fill at that close.** Enforced twice:
`EVENT_PRIORITY` puts `DECISION_TICK` (70) before `ORDER_READY` (80) and
`EXECUTION` (90) at equal timestamps, *and* `execution/latency.py` advances the
clock before the order meets a market state.

---

## 6. Accounting invariants (`backtest/invariants.py`)

Checked continuously, not at the end. Any breach raises — a backtest that
silently violates conservation is worse than no backtest.

- Cash never negative (no leverage modelled).
- No sale exceeding settled inventory.
- Cash and tokens change **only** through ledger entries.
- Fees charged exactly once.
- Every fill belongs to one order and one position lot.
- Same inputs + same hashes → **byte-identical** economic output.
- A dry run mutates nothing.
- An execution failure can never become a synthetic fill.
- Risk resizing triggers a **new quote at the approved size** (audit C3).
- SELL quantity == quoted quantity, exactly.
- Duplicate `ExecutionReport.report_id` is a no-op.

---

## 7. Validation rules

- **Splits are interval-aware**, not row-count-based: assets have wildly
  different sampling density (BONK misses 0.5% of 5m bars; SLERF misses 86.5%).
- Purge any training observation whose `[label_start_ts, label_end_ts]`
  overlaps validation/test.
- Embargo in **wall-clock**, covering max holding horizon + publication delay +
  rolling-state carryover.
- All assets split on the **same wall-clock boundaries**. Randomly assigning
  coin-observations to folds lets the model see one market event in both train
  and test.
- Outer folds: train 180d / val 30d / test 30d / roll 30d are *starting values*
  to be pre-registered, not truths.
- **Holdout:** the plan says 90 days or ~20%. We have ~209 days of 1h. 90 days
  is 43% of the record. Default is `max(45 days, 20%)` with a loud warning that
  this is below the plan's recommendation *because the horizon is 209 days* —
  the shortfall is stated, not hidden.
- Opening the holdout is recorded in the experiment registry. Revising after
  seeing it invalidates it; the registry enforces this.

---

## 8. Testing requirements

`pytest`, offline, no network. `hypothesis` for accounting and AMM math.
Every module ships tests under `tests/<area>/`.

Mandatory tests (plan §13), each of which must **fail** if the guard is removed:

- *Prefix equivalence*: feature at `t` from the full dataset == feature at `t`
  from data truncated at `t`.
- *Future sentinel*: inject an extreme future value; no earlier feature or
  trade may change.
- *Timestamp delay*: increase `available_time`; signals must move later.
- *Universe survivorship*: remove future metadata; historical eligibility
  unchanged.
- *Fold audit*: no training label interval overlaps validation/test.
- *Signal-at-close cannot fill at that close.*
- *Risk clamp forces an exact-size requote.*
- *Expired quote cannot fill. Failed route produces no position change.*
- *Duplicate execution report is idempotent.*
- *Cash/token conservation; fee reconciliation; multi-lot exits.*

**Gate (must stay green):**
```
uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run mypy
```
Baseline to preserve: **942 passed, 1 skipped**; ruff clean; mypy 0 errors in
`src/` (48 pre-existing in `tests/` — do not add to that count).

Ruff config that bites: `line-length = 92`, `max-line-length = 100` for E501,
selected rules include `PTH` (pathlib everywhere), `DTZ`, `PERF`, `RUF`, `B`,
`BLE` (no bare `except Exception` outside the three exempted modules).

---

## 9. Run artifacts

`runs/backtests/<run_id>/` — gitignored (generated), same as live runs.

```text
manifest.json  config.yaml  data_manifest.json  folds.parquet  forecasts.parquet
orders.parquet execution_reports.parquet fills.parquet positions.parquet
equity.parquet attribution.parquet metrics.json leakage_report.json
promotion_report.json report.md
```

`manifest.json` carries: git commit, dirty-worktree hash, data partition
hashes, universe version, feature definitions, model artifact hash, config
hash, dependency lock hash, seeds, trial number, parent experiment, the
`FidelityTier`, and **whether any human or LLM has seen prior holdout
results**. A run without provenance is not evidence.
