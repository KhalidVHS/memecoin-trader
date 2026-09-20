# memetrader — Complete Project Overview

*A Solana memecoin paper trader. A deterministic strategy — optionally advised
by Claude Opus 5 — trades BONK, WIF and POPCAT against a $1,000 paper book on a
dual cadence, with every decision, every piece of evidence and every fill
written to disk. No wallet, no private keys, no real money.*

This document explains the entire project: what it is, what it deliberately is
not, how every module works, why each non-obvious decision was made, what the
tests cover, how to operate it, what it costs, and what a real 12-hour run
produced.

---

## Table of contents

1. [What this is](#1-what-this-is)
2. [What this is not](#2-what-this-is-not)
3. [Architecture at a glance](#3-architecture-at-a-glance)
4. [The governing ideas](#4-the-governing-ideas)
5. [The tick cycle, end to end](#5-the-tick-cycle-end-to-end)
6. [Module reference](#6-module-reference)
7. [The prompt and the cache boundary](#7-the-prompt-and-the-cache-boundary)
8. [Risk layer reference](#8-risk-layer-reference)
9. [Fill realism](#9-fill-realism)
10. [Data, ledgers and file layout](#10-data-ledgers-and-file-layout)
11. [Configuration reference](#11-configuration-reference)
12. [CLI reference](#12-cli-reference)
13. [Tooling: the documented run harness](#13-tooling-the-documented-run-harness)
14. [Results of the 12-hour run](#14-results-of-the-12-hour-run)
15. [Cost model](#15-cost-model)
16. [Tests](#16-tests)
17. [Engineering conventions](#17-engineering-conventions)
18. [Deliberate omissions](#18-deliberate-omissions)
19. [Known issues and next steps](#19-known-issues-and-next-steps)
20. [Glossary](#20-glossary)

---

## 1. What this is

`memetrader` is a **paper trading system** for three Solana memecoins. Every 15
minutes it assembles a fresh evidence bundle for each coin, hands it to a
**strategy**, and receives back a set of **target positions** — a dollar target
per symbol. The loop diffs those targets against the inventory it actually
holds, asks the risk engine for bounds on each resulting trade, quotes the
permitted size against live Solana routing data, and simulates the fill.

The default strategy is **deterministic**: shrunk momentum with an explicit
round-trip cost hurdle and an uncertainty band. A language model is available as
a named strategy kind (`advisory`) and is off unless someone turns it on. That
is a change from the original design, in which the model chose and sized every
trade. The adversarial audit's finding (C6) was that an LLM held order authority
with no measured predictive value behind it and no established counterfactual
against a deterministic rule on the same evidence — and that a *default* is
authority, because it is what runs when nobody chose. Making the model opt-in is
also what makes an honest A/B against `baseline` possible.

Between decisions, a cheap 60-second tick keeps the book marked to market and
enforces stop-losses without calling any strategy at all.

The goal is **not** to make money. The goal is to produce a high-quality,
fully-documented record of *(market state → policy → rationale → outcome)*
tuples that can be studied afterwards — the system was built so that after a run
you can ask "why did it think that was a good idea?" and get a real answer, down
to the intent ID of every order and, when the advisory path ran, the verbatim
prompt the model saw at that moment.

**Scale:** ~16,900 lines of source across 19 modules, ~12,400 lines of tests
(900 test cases), plus a 1,097-line documentation harness.

| Layer | Lines | Files |
| --- | ---: | ---: |
| `src/memetrader/` | 16,908 | 19 |
| `tests/` | 12,429 | 15 |
| `tools/` | 1,108 | 2 |
| config + docs | ~1,800 | 3 |

---

## 2. What this is not

**There is no real money anywhere in this project.** There is no wallet, no
private key, no RPC signer, no transaction submission. Prices, liquidity,
routing and slippage are all read from live public APIs, but the fills are
simulated locally against a JSON ledger.

The seam where a real venue *would* attach is the `Broker` **Protocol** in
`types.py`. `LocalPaperBroker` implements it. A live broker would implement the
same protocol, and nothing else in the system would need to change — the risk
layer, the loop, the prompt and the journal all speak to the protocol, not the
implementation. That is a deliberate reversibility seam, not an invitation.

The absence is now expressed in the type system rather than in a comment.
`ExecutionMode` has three values, and `ExecutionMode.LIVE` is refused by
`assert_live_supported()` with a `NotImplementedError`. There is no stub, no
half-written signer and no flag that would become dangerous if someone set it.

Other things it is not:

- Not a backtester. It only trades forward, against live data.
- Not a high-frequency system. The fastest thing it does is mark a book once a
  minute.
- Not asynchronous. It is a single thread, synchronous `httpx`, start to finish.
  Concurrency would buy a handful of seconds per 15-minute tick and cost every
  invariant in the codebase.
- Not a sentiment-polarity engine. See §4.

---

## 3. Architecture at a glance

```
                  ┌──────────────────────────────────────────────┐
                  │              cli.py (typer)                  │
                  │   check · status · once · run · report ·     │
                  │        reset      (--mode on once/run)       │
                  └───────────────────┬──────────────────────────┘
                                      │
                  ┌───────────────────▼──────────────────────────┐
                  │        loop.py — Trader (preflight first)    │
                  │   fast_tick (60s)      slow_tick (900s)      │
                  └───┬──────────────┬──────────────┬────────────┘
                      │              │              │
   ┌──────────────────▼───┐  ┌───────▼────────┐  ┌──▼──────────────────┐
   │  EVIDENCE            │  │  STRATEGY      │  │  EXECUTION          │
   │  market.py  (prices) │  │  strategy.py   │  │  risk.py   (bounds) │
   │  signals.py (tech)   │─▶│   baseline     │─▶│  quotes.py (route)  │
   │  sentiment.py(off by │  │   advisory ──▶ │  │  broker.py (fill)   │
   │              default)│  │   prompts.py + │  │  ids.py (identity)  │
   │                      │  │   brain.py     │  │                     │
   └──────────────────────┘  └────────────────┘  └──┬──────────────────┘
                                targets                │ bounded intents
                  ┌──────────────────────────────────▼───────────┐
                  │  portfolio.py (mark, stops) · journal.py      │
                  │  data/ledger.jsonl · state.json ·             │
                  │  trades.jsonl · intents.jsonl ·               │
                  │  decisions.jsonl · risk_ledger.json           │
                  └──────────────────────────────────────────────┘
                  report.py renders all of it. http.py makes every client.
```

**Data flows one way.** Evidence is read, turned into targets, bounded,
quoted, executed, journaled. Nothing downstream writes back into the evidence
layer, and nothing upstream of `broker.py` can place an order: a strategy emits
targets, the loop turns targets into intents, and risk returns *bounds* on
those intents rather than permission to act.

---

## 4. The governing ideas

Everything in the codebase follows from a handful of decisions. If you read
nothing else, read this section.

### 4.1 Dual cadence

A **fast tick** runs every 60 seconds: refresh prices, mark the book, enforce
stop-losses. No model call, so it is nearly free.

A **slow tick** runs every 900 seconds (15 minutes): build the full evidence
bundle, ask the strategy for targets, bound every resulting trade, execute what
survives.

The reason is blunt: *a -15% stop that only checks every 15 minutes is not a
stop.* Memecoins can travel that far in a fraction of a decision interval. The
fast tick exists so that getting out is never gated on an expensive decision.

A consequence that took a bug to learn: because the two cadences interleave,
anything measured *between* reads has to name which cadence it belongs to. The
liquidity trend used for a decision is measured against `decision_baseline` —
the snapshot the last decision was made on — not against `previous`, the
freshest read of any kind. When both shared `previous`, a pool that shed 10% of
its depth between decisions was reported as **-0.7%** and read as noise.

### 4.2 Evidence streams, ranked by trust

In descending order of how much weight the system gives them:

1. **Price and on-chain flow** (DexScreener) — the most trustworthy. Actual
   trades, actual liquidity, buy/sell transaction counts over m5/h1/h6/h24.
2. **Technicals** (pandas over GeckoTerminal 5m and 1h candles) — descriptive,
   not predictive. Two indicator families only, computed over **closed bars
   only**.
3. **Reddit attention** — the least trustworthy, a measure of *attention* rather
   than sentiment, and **disabled by default**.

Each stream degrades independently. A stream that fails is reported as
explicitly unavailable, with a reason, rather than silently contributing a
blank.

The technicals carry a warning that belongs here rather than in a docstring:
RSI, the two EMAs and their distances, MACD line/signal/histogram, %B and
bandwidth are all deterministic transforms of *one* close series. When they
"agree", what has happened is that one price path was described five times.
There is therefore **no agreement score, confluence count or "N of M signals
bullish" number anywhere in the codebase**, and none may be added; the
indicators are kept as a labelled benchmark feature set with no demonstrated
alpha, so that a future ablation has something to ablate against.

### 4.3 Attention, not polarity — and off by default

The sentiment module does not try to tell you whether Reddit *likes* a coin. It
measures how much people are talking about it, how fast that is changing against
its own 7-day baseline, and how many distinct accounts are doing the talking.
The contributor-to-post ratio is the point: **few accounts posting a lot is a
shill-farm signature**, and that is a real, measurable thing. "Is the sentiment
positive?" is not — polarity on memecoins is manufactured on demand for a few
dollars, which is why the old `keyword_polarity` field and its twelve-line
lexicon were deleted rather than improved.

Two audit findings then shaped the module as it now stands.

**The stream is disabled by default** (`[sentiment] enabled = false`, audit
§7/§11). Nobody has measured whether it improves net out-of-sample results at
the intended capacity after its latency and cost, so it is an experiment source
rather than a production input. When disabled it reports as `UNAVAILABLE` —
never as a zero. "Nobody is talking about this" and "we did not look" are
opposite trades.

**No untrusted text reaches a model, by construction** (audit C7). The old
module collected Reddit submission and comment text and the prompt renderer
interpolated it verbatim into a prompt that had order authority — indirect
prompt injection in a capital-allocation loop, writable by any member of the
public. The fix was deletion, not escaping: `SentimentBrief` no longer has an
excerpt field, `TopPost` no longer exists, and the *collection* path is gone.
Post text now exists only inside a single tick, only as input to the alias
matcher, and `Post` redacts it from its own `repr` so a stray log line cannot
leak it either. The defence is the absent field.

### 4.4 Missing is never zero

**This is the single most pervasive invariant in the codebase.**

Every optional field is typed `X | None`. `None` means *"we could not find
this out."* `0` means *"we looked, and it is quiet."* These are different
claims and they are never allowed to collapse into each other.

- `types.py` types every optional measurement as `X | None`.
- `signals.py` returns `None` rather than a default when a window is too short.
- `sentiment.py` forces **every** rate to `None` when a sweep returned nothing,
  because an all-empty sweep leaves the failure list empty and would otherwise
  read as a confident, maximally bearish "attention is flat."
- `prompts.py` renders every `None` as `n/a`, with inline annotations like
  `<- n/a = not yet indexed by the source, NOT zero attention`.
- `market.py` keeps *absent* (`None`), *malformed* (quarantine the observation
  with a reason) and *zero* apart, instead of coercing all three to `0`.
- `report.py` prints `n/a`, never `0.0`, for the same reason.

**A position that cannot be priced is not priced.** `portfolio.mark()` produces
a `Mark` with a stated `basis` — `route`, `mid`, `estimate` or `unavailable` —
and **never** cost basis. This is audit C8, and it is worth stating why the
older, friendlier behaviour was a defect rather than a conservative default:
carrying an unmarkable position at what you paid for it displays exactly zero
unrealised loss during the only two events that matter, a rug and a data
outage, and because the stop measured drawdown against that same fabricated
value, the stop was guaranteed not to fire at precisely the moment it existed
for. `PortfolioState.total_value_usd` is therefore `None` when any held
position is unmarkable, and that `None` propagates — every downstream
percentage would otherwise be computed against a figure partly made of cost
basis standing in for a price nobody could obtain. `PortfolioState.unmarkable`
names the offenders, and an unmarkable position is treated as a **risk
incident**: data health fails, entries are blocked, exits stay open.

### 4.5 Numbers are integers where it counts

Token amounts are **atomic integers** — the exact `inAmount`/`outAmount`
Jupiter quoted — and cash is **integer micro-USD**. Dollars are a *rendering*,
computed at the display edge and never fed back into arithmetic.

This is audit C2, and the worked example is the argument: the old code set
`filled_usd` to the *requested* dollars and back-derived a quantity as
`filled_usd / price_usd`. Request $100 at a $1 mid, have Jupiter quote 100
tokens for $95, and the book records $100 of proceeds and 105.263 tokens.
Neither number describes a swap that was ever offered. Nothing in the codebase
re-derives a quantity from dollars any more.

`finite()` and `finite_or_none()` reject NaN and infinities at construction.
A NaN is the one value that passes every comparison by being false against all
of them: it is not negative, not positive, not above a cap — so a single
unchecked float could walk the entire length of the old system precisely
because nothing about it was ever true.

### 4.6 The strategy proposes targets; the code decides

A strategy returns a `StrategyDecision` carrying `TargetPosition`s — a dollar
target per symbol — plus optional `Forecast`s and a written market read. It
cannot place an order, size an order, choose a venue or see a quote.

`loop._diff_targets` subtracts what is held from what is targeted and keeps the
difference only if it clears both `rebalance_band_usd` and `min_trade_usd`. The
band is not a nicety: every round trip pays gas plus spread plus price impact,
and rebalancing a $50 position by $3 pays all of that to move nothing. A symbol
that is held but unmarkable is skipped and said so out loud, because sizing a
delta against an unknown is arithmetic on a guess.

Risk then returns **`RiskBounds`** — a maximum notional, a list of vetoes with
the rule that fired, the binding rule, and any rules bypassed — not an approval
and not a mutated order. The distinction is structural: an approval is a
permission that later code can act on, whereas a bound is a ceiling that later
code must respect. And a bound never silently shrinks an order that has already
been quoted; a smaller size is re-quoted, and the new quote's fingerprint is
re-confirmed before anything can fill.

### 4.7 Quotes are bound, and failures are refusals

Every `Quote` carries a fingerprint over its side, both mints, both atomic
amounts and the context slot. The broker recomputes that fingerprint from the
quote it was handed and refuses on mismatch (audit C3/C5). Binding, not trust.

A route that could not be obtained is **not a worse route, it is no route**
(audit C4). The old code synthesised a quote from a DexScreener mid plus a fixed
slippage assumption, flagged it `degraded`, and filled against it — so a Jupiter
outage, precisely when a mid-price fiction is least credible, *manufactured
trades*. The executable quote functions now return `Quote | None`, and `None`
means there is nothing to execute. `Quote` and `ValuationEstimate` are separate
types so that a number good enough to display can never be mistaken by the type
checker for a price someone offered.

The safety screen follows the same rule and **fails closed**: a `SafetyVerdict`
keeps `unknowns` separate from `vetoes`, and names in `deferred` the on-chain
checks (mint authority, freeze authority, LP locks) that are not implemented. A
check that could not be run is not a check that passed.

---

## 5. The tick cycle, end to end

### Before any tick: preflight

`Trader.preflight()` runs once at startup and asks three questions in the order
that matters:

1. **Does the broker's ledger replay into the state it claims?** `load()` /
   `reconcile()` answer this, rebuilding state from fills if it does not.
2. **Is the journal intact?** A torn final line is the signature of a crash
   mid-append, and it means the last thing that happened is exactly the thing we
   cannot read. It is discarded, and the discard is reported as a note.
3. **Is any order intent still open?** An intent with no terminal state is an
   order whose outcome is unknown. Re-running the loop would either place it
   twice or abandon it, and there is no third option available from inside this
   process.

The third question can **refuse the run**. `StartupRefusal` names the offending
intents and points at `data/ledger.jsonl` and `memetrader reset`:

> `2 order intent(s) have no terminal state: … Their outcome is unknown, so
> placing new orders risks duplicating them. Resolve them in
> data/ledger.jsonl (or `memetrader reset` if this is a paper book you are
> willing to discard) before trading.`

This is audit C11. The old shape appended a trade row and then wrote state —
two files, two operations, no transaction and no IDs — so a kill between them
left a ledger ahead of state with nothing to say which rows belonged to which
decision. Refusing to start is the honest response to an unknown execution
state; trading on top of it is not.

### Fast tick (every 60 seconds)

1. `market.snapshot(with_candles=False)` — prices and liquidity only.
2. `Trader.book()` — mark every position, compute unrealized P&L.
3. `ContinuousRisk.evaluate()` — drawdown, loss budgets, failure streak, data
   health. Evaluated even on ticks with no order in them, because those
   conditions arrive *between* orders.
4. `portfolio.stop_loss_breaches()` — worst drawdown first.
5. For each breach: `Trader._force_exit()` → `risk.exit_bounds(forced=True)` →
   quote → `confirm_quote` → journal intent → `broker.place_order()`.
6. Re-mark if anything filled, so the caller sees the book *after* the exits.
7. Store the snapshot as `previous`. **Do not** touch `decision_baseline`.

Candles are deliberately skipped. Fetching OHLCV the fast tick never reads was
spending 6 of GeckoTerminal's ~30 keyless requests per minute plus ~7.5 seconds
of inter-request spacing on every single 60-second tick — it earned HTTP 429s
and degraded the *slow* tick's technicals, which are the only place candles are
actually used.

### Slow tick (every 900 seconds)

1. `market.snapshot()` — full, with candles.
2. Mark the book; evaluate continuous risk.
3. **Stops first.** A position past its stop should not survive long enough for
   a strategy to have an opinion about it.
4. `Trader.evidence()` — assemble every enabled stream per coin into an
   `EvidenceBundle`, computing the real elapsed gap since `decision_baseline`
   (not the configured cadence: a long tick, an outage or a restart all move it,
   and the liquidity trend is printed with the window it was measured over).
5. **If risk is halted, stop here.** Halted means exits only, and the stops
   above already ran. Calling the strategy anyway would burn a model call to
   produce targets guaranteed to be vetoed, and would journal a decision that
   never had a chance of executing.
6. `strategy.decide(evidence, book, now)` → `StrategyDecision` with targets.
   Under `advisory` this is where `prompts.build_system` / `render_user` and one
   `brain.advise()` call happen; under `baseline` and `cash` there is no network
   call at all.
7. `_diff_targets` → the list of (symbol, delta) trades that clear the
   rebalance band.
8. For each trade: `risk.entry_bounds` / `exit_bounds` → quote at the permitted
   size → `confirm_quote` → journal the intent → `broker.place_order()` →
   journal the outcome, re-marking between fills so each successive sizing sees
   the updated book.
9. Write one `DecisionRecord` — targets, bounds, intents, fills, risk state,
   token usage — to the ledger. Skipped entirely unless the mode `may_mutate`.
10. Advance **both** `previous` and `decision_baseline`.

**If the strategy raises, the tick returns early with `error` set and
`decision is None`.** `decision_baseline` is deliberately left alone: a tick
that died before the strategy ran is not a decision, so the next successful tick
measures liquidity across the whole outage and prints that longer window, rather
than claiming 15 minutes over evidence nobody ever read.

**A failure must never come back as a hold.** A tick that failed and a tick that
decided to hold what it has are different events, and the CLI renders them
differently.

### The four-step ordering inside `_execute`

The order is the whole point:

1. **Quote the size risk permitted**, not the size that was wanted. A quote
   describes one specific swap; it is not a price curve you may evaluate at
   another point.
2. **`confirm_quote`** re-runs the quote-dependent rules — price impact, depth
   participation, quote age, identity — against the quote that will actually be
   sent. A quote that was fine at $25 may be 6% impact at the same $25 thirty
   seconds later.
3. **Journal the intent before the side effect.** The intent ID is the
   idempotency key; if the process dies after this line and before the fill,
   `preflight` finds an open intent and refuses to trade rather than placing it
   a second time.
4. **Place, then journal the outcome.** The broker writes its own fill row
   inside `place_order` before it persists state, so a fill can never exist in
   the book without a row explaining it.

Every refusal returns `None`, never an exception: a refused order is an ordinary
outcome of a tick and the loop must keep running.

---

## 6. Module reference

### `types.py` — 1,364 lines. The shared shapes.

Frozen dataclasses and two Pydantic models, plus the validators (`finite`,
`finite_or_none`, `positive`, `non_negative`, `atomic`) every constructor calls.
No logic beyond derived properties.

| Type | Purpose / notable detail |
| --- | --- |
| `Side`, `Timeframe` | `StrEnum`s. |
| **`ExecutionMode`** | `READ_ONLY` / `PAPER` / `LIVE`. A capability, not a flag: `may_mutate` gates every write, and `assert_live_supported()` raises `NotImplementedError` because there is no wallet. |
| `OrderState` | The state machine an intent moves through. Terminal states are what `preflight` looks for. |
| `DataQuality` | How trustworthy an observation is. |
| `Provenance` | Event time, source and receive time for a value — with `oldest_age_seconds`, so "how old is the oldest thing in this claim" is answerable. |
| `Observed[T]` | A value plus its provenance. |
| `Candle`, `CandleSeries` | OHLCV bars; the series knows which bars are closed. |
| `PriceLadder` | `m5`/`h1`/`h6`/`h24`, **all `float \| None`** — DexScreener omits `m5` on roughly 13 of 30 live pairs, including BONK's best pool. |
| `TxnCounts` | Buys/sells; `.ratio` returns `inf` when sells == 0, which is meaningful, not an error. |
| `PoolRef` | The chosen pool, its quote token and its age. |
| `CoinSnapshot` | Price, liquidity, volume, the ladders, quality and reason. |
| `MarketSnapshot` | All coins + `ts`, with `age_seconds`. |
| `Technicals` | Every field `X \| None`. |
| `FlowBrief` | Carries both `liquidity_trend_pct` **and** `liquidity_trend_seconds`, so a percentage can never be printed without the window it was measured over. |
| `TechnicalBrief` | 5m + 1h `Technicals` and the `FlowBrief`. |
| `SentimentBrief` | Counts, velocities, z-score, contributors, ratio. **No text field** (audit C7). |
| `TokenMeta` | Mint and decimals — the only place decimals live. |
| **`Quote`** | An executable route: integer `in_amount_atomic`/`out_amount_atomic`, slot, expiry and a `fingerprint`. |
| **`ValuationEstimate`** | A number good enough to display and *not* good enough to trade. A separate type so the two can never be confused (audit C4). |
| `Fill` | An executed or failed attempt, in atomic units and micro-USD. A failure is a `Fill` with `state=FAILED`, zero amounts and non-zero gas. |
| **`OrderIntent`** | The idempotency key. Journaled before the side effect. |
| `Position` | `quantity_atomic`, cost basis, `age_seconds`. |
| **`Broker`** | **Protocol.** The reversibility seam. |
| `Mark` | A price with a `basis` (`route`/`mid`/`estimate`/`unavailable`) and provenance. Never cost basis. |
| `PortfolioState` | Cash, positions, marks, `unmarkable`, `fully_marked`, and a `total_value_usd` that is `None` when anything is unmarkable. |
| **`RiskBounds`** | `max_notional_usd`, `vetoes`, `reasons`, `binding_rule`, `bypassed_rules`. A ceiling, not an approval. |
| `RiskState` | Halt flags and reasons, exposure, drawdown, failure streak, quarantines, data health. |
| `Forecast` | An expected return with an uncertainty band and a `calibration_id` that is `None` until someone measures one. |
| **`TargetPosition`** | A dollar target for one symbol. The entire strategy output surface. |
| `StrategyDecision` | Targets + forecasts + market read + `strategy_id` + `decision_id`. |
| `AdvisoryAction`, `AdvisoryDecision` | **Pydantic.** The structured-output schema for the opt-in advisory path. Advice, not orders. |
| `DecisionRecord` | One journaled slow tick: read, targets, bounds, intents, fills, risk state, mode, tokens, model, effort, thinking, `advisory_used`. |
| `EvidenceBundle` | Per-coin: snapshot + technicals + sentiment + unavailability reason. |

### `ids.py` — 119 lines. Identity for everything.

Every object in the decision-to-fill chain carries an ID minted here, and two
properties matter. **Ordering:** IDs are prefixed with a zero-padded 18-digit
microsecond timestamp, so lexical sort equals chronological sort and a JSONL
ledger can be sorted without parsing it — microseconds rather than seconds
because a slow tick emits several intents inside one second and they must still
order. **Uniqueness without coordination:** a short random suffix means two
processes that should never have been running at once still cannot silently
produce colliding IDs; the duplicate shows up as two rows rather than one
overwriting the other.

They are deliberately *not* content hashes. A content hash of an intent would
collide whenever the same order is legitimately placed twice, which is exactly
the case idempotency has to distinguish. Content hashing lives in
`quote_fingerprint`, where equality *is* the question.

### `strategy.py` — 519 lines. Who decides what to hold.

Three implementations of one `Strategy` Protocol, selected by
`build_strategy(cfg)` from `[strategy] kind`:

- **`CashStrategy`** — targets zero everywhere. The null hypothesis, and the
  control arm any claim of alpha has to beat.
- **`BaselineStrategy`** (`strategy_id = "baseline-momentum-v1"`) — the default.
  Deterministic momentum, shrunk toward zero by a fixed factor because raw
  short-horizon momentum on memecoins is mostly noise; an expected round-trip
  cost is subtracted; and an entry is taken only when the **lower** quantile of
  the expected net return clears `entry_hurdle_pct`. A wide, uncertain +5% is
  not a trade; a tight +1.2% is.
- **`AdvisoryStrategy`** — the baseline, plus a model that may modify the
  targets it produced. Opt-in.

Sizing is deliberately **flat** (`flat_size_usd`). Sizing proportional to
conviction requires a calibration that has never been measured here, and an
uncalibrated conviction number scaling position size is how a bad model loses
money faster than a coin flip. Until `Forecast.calibration_id` is non-`None`,
every entry is the same size and the only decision is in or out.

### `config.py` — 737 lines. Load and validate.

Parses `config.toml`, reads secrets from the environment via `load_dotenv`,
and validates hard.

- `find_project_root` walks upward looking for `config.toml`, so the CLI works
  from any subdirectory.
- Validation: mint addresses must be 32–44 characters; symbols must be unique;
  cadences must be positive with fast < slow; every `_pct` fraction must be in
  range; effort must be one of `low | medium | high | xhigh | max`;
  `execution_mode` must be one of `read_only | paper | live`; `[strategy] kind`
  must be one of `cash | baseline | advisory`.
- **Cross-table validation.** The strategy's `min_trade_usd` must not be below
  the risk layer's, and its `flat_size_usd` must not exceed `max_entry_usd` —
  two tables that disagree produce a strategy whose every target is vetoed, and
  finding that out at load time costs a second instead of a tick.
- `Config.ledger_path` is the one place `data/ledger.jsonl` is named.
- **`ModelConfig.cost_usd(input, output, cache_read, cache_write)` is the only
  cost formula in the codebase.** Its docstring records that three earlier
  versions each understated the bill in a different way. `cache_write` is
  deliberately a *required* argument, not a defaulted one, so a new call site
  cannot silently price cache creation at 1x instead of 1.25x.
- `ExecutionConfig.fee_pct_for` sums pool fees across every hop of a route.
- `DataConfig.jupiter_url_base` picks the keyed host when an API key exists and
  the free lite host otherwise.
- `SentimentConfig.include_comments` is *defaulted* rather than required —
  specifically because making the `ModelConfig` cache-write field required once
  broke every positionally-constructed test object, and that lesson was applied
  here in advance.
- Secrets read from env: `ANTHROPIC_API_KEY`, `JUPITER_API_KEY`,
  `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` / `REDDIT_USER_AGENT`.

### `http.py` — 1,126 lines. One client factory, one retry policy.

Load-bearing for two separate reasons.

**TLS.** `httpx` verifies against `certifi`, which contains only public roots.
**This machine sits behind a corporate TLS-inspecting proxy**, so every outbound
call fails `CERTIFICATE_VERIFY_FAILED` unless verification uses
`ssl.create_default_context()` — the OS trust store, the only store that has the
inspection CA in it. This is not weaker verification, just verification against
a different set of roots, and it is strictly more portable. A bare
`httpx.Client` constructed anywhere in the tree is a bug that only shows up on
this network.

**Resilience** (audit §15). Each of these is a failure mode observed against a
free public API on a 60-second tick:

- **Split timeouts.** One scalar timeout is what lets a hung read stall a tick.
  `Timeouts` carries connect, read, write and pool separately, and is tighter on
  connect than on read, because a connect that has not completed in a couple of
  seconds behind the proxy is not going to.
- **A retry *budget*, not a retry count.** Bounding attempts alone does not
  bound time: three attempts against a 15s read timeout is a 45-second stall
  inside a 60-second tick. `RetryPolicy` bounds both `max_attempts` and
  `total_budget_seconds`.
- **Exponential backoff with jitter**, an honoured but capped `Retry-After`, and
  a **circuit breaker** so a source that is down stops being asked every tick.

### `market.py` — 1,193 lines. DexScreener, GeckoTerminal, and the safety screen.

Four hazards are handled at this boundary, all of them discovered live:

**Unit skew.** `pairCreatedAt` is in milliseconds. `_ms_to_seconds` is
defensive rather than a blind `/1000`: it returns `None` for anything outside
2020–2100 rather than emitting a plausible-looking wrong timestamp.

**Stringly-typed numerics.** Every numeric field in these APIs may arrive as a
string, a number, or be absent. Coercion happens once, here.

**Missingness.** The old `_as_float`/`_as_int` returned `0` for a value that was
absent *and* for one that was garbage — turning "DexScreener omitted the m5
block for this quiet pair" into a confident "the price was exactly flat over
five minutes", and the string `"n/a"` into a real-looking number. On a live
30-pair sample, 13 pairs had no `m5` block at all, including BONK's best pool,
so this was not a rare path. `_opt_float` and `_opt_int` now keep three cases
apart: **absent** → `None`; **malformed** → `MalformedField`, and the coin's
observation is quarantined with a reason rather than silently repaired;
**present and valid** → the value.

**The safety screen.** `market.screen` returns a `SafetyVerdict` per coin and
fails closed: `vetoes` are measured failures, `unknowns` are screens that could
not be *evaluated*, and unknowns veto too, because a screen that passes on
missing data is not a screen. They are reported separately so an operator can
tell "this pool is too small" from "we never learned how big this pool is".
`deferred` names the C9 checks this screen does **not** perform — mint
authority, freeze authority, LP ownership and lock, holder/dev concentration,
Token-2022 extensions, a sellability probe — and it is part of the return value
on purpose, so the gap is reported rather than implied.

**Pair selection — the important one.** `_best_pair` has two rules: *never take
`pairs[0]`*, and *only consider pools quoted in SOL, USDC or USDT.* Observed
live: the highest-`liquidity.usd` BONK pool was a Meteora DLMM quoted against a
pump.fun token, reporting $2.4M of liquidity and a BONK price of $0.01434 —
about **4,900x the real price**. A naive "pick the deepest pool" would have
traded the entire book against a fiction.

Also here: a browser User-Agent defeats Cloudflare's 403 interstitial, and
`_check` gives 403 its own do-not-retry message. `_fetch_candles` **reverses
GeckoTerminal's `ohlcv_list`**, which arrives newest-first — un-reversed, every
EMA, MACD and ATR computes over time-reversed data and produces confident
nonsense. `_GECKO_DELAY_SECONDS = 1.5` paces the keyless endpoint.

### `quotes.py` — 671 lines. Jupiter routing. No fallback quote.

A DexScreener mid is what the last trade printed at; it is not what you would
get. For an asset class where a $500 order can move the pool, that gap is the
entire difference between a backtest that looks good and one that is true, so
every trade is priced by asking Jupiter for a real route at the exact size.

- **The atomic amounts are the fact.** `inAmount` and `outAmount` are preserved
  verbatim onto `Quote` and consumed as integers by the broker (audit C2).
- **A quote is bound to the size it was obtained for.** Every quote carries a
  `quote_fingerprint` over (side, both mints, both atomic amounts, context
  slot); the broker recomputes it and refuses on mismatch (audit C3).
- **There is no `_fallback` any more** (audit C4). The executable functions
  return `Quote | None`. Failure has exactly one other place to go:
  `mark_route` may return a `ValuationEstimate`, which is a display number and
  is not executable by type.
- `_derive_decimals` recovers a token's decimals from a single base/UI amount
  pair. Decimals are an integer power of ten, so being off by 3x moves `log10`
  by 0.48 — short of the 0.5 needed to round to the wrong integer.
- `_token_decimals` tries, in order: the process cache → Jupiter's
  `/tokens/v2/search` (the v1 route now 404s) → a throwaway probe.
- Jupiter's `priceImpactPct` is a **fraction string**, and it is multiplied by
  100 at the boundary so that everything downstream is in whole percent.

### `signals.py` — 615 lines. Technicals, pandas only.

Two rules govern this file.

**Deliberately few indicators.** The predecessor project ran eight signals and
finished **-16.62%**. Cutting to two finished **+5.39%**. The list is closed:
EMA(9/21) with MACD, and RSI(14) with Bollinger %B and ATR as context. Adding
a ninth signal is not an enhancement; it is the mistake that was already made.

**Missing is never zero.** Every function returns `None` rather than a default
when the window is too short or the data is absent.

**Only closed bars may feed a feature.** The audit's look-ahead finding: every
indicator and the volume ratio were computed over the *partial current bar*,
which leaks the in-progress period into a number presented as history. Every
computation here now runs over closed bars only.

**Agreement is not confirmation.** See §4.2 — these are all transforms of one
close series, there is no confluence score, and none may be added.

pandas and numpy only — **never TA-Lib**, which needs a C toolchain on Windows.

Implementation details that each fixed a real defect:

- `_sma_seeded_ema` seeds the EMA with the SMA of the first `span` values, not
  pandas' first-observation seed. On a 100-bar window the difference can flip a
  MACD cross.
- `_wilder_smoothed` (alpha = 1/period) is used for RSI and ATR — *not*
  `ewm(span=)`, which is a different smoother and produces different numbers
  from every charting package on earth.
- `_rsi` returns `50.0` for a flat window (0/0) and `100.0` for gain with no
  loss.
- `_macd_cross` returns `("none", None)` — the age of a cross that never
  happened is not a number.
- `_bollinger` uses population sigma and leaves %B **unclamped**. Excursions
  outside [0, 1] are the signal.
- `_atr` uses true range *including both gaps*. Memecoins gap constantly;
  high-minus-low alone understates volatility and makes a -15% stop look safer
  than it is.
- `flow_brief` takes `previous` and `elapsed_seconds` **from the caller**. It
  used to be handed the 60-second-ago snapshot while the prompt said "vs last
  tick" — see §4.1.

### `sentiment.py` — 1,720 lines. Reddit attention, disabled by default.

Almost all of it is care. `SentimentSettings.enabled` is wired from
`[sentiment] enabled`, which ships `false`; a disabled stream reports as
`UNAVAILABLE`, never as zero. Nothing collected here is text (§4.3).

**Two providers behind one shape:**

- `ArcticShiftProvider` — keyless archive mirror. `index_lags = True`.
- `PrawProvider` — the official Reddit API. `index_lags = False`. It sweeps
  `/new` and `/r/<sub>/comments/` listings rather than calling `search()`,
  because Reddit's search index is populated asynchronously and a search-based
  sweep systematically under-reports the most recent — i.e. the most
  interesting — hour.

**`brief()` never raises.** Total failure returns `None`, and the loop reports
that as an explicit unavailability reason.

**`build_brief()` is the single arithmetic implementation** shared by both
providers, so the two can never drift. `sweep_size == 0` forces every rate to
`None` (see §4.4). `observed_through` bounds zero-seeding so that an hour the
source has not indexed yet never becomes a fabricated zero bucket.

**Alias matching.** `alias_pattern` uses explicit lookarounds rather than `\b`,
so `$WIF`, `#WIF` and `WIF_army` match while *wife*, *swift*, *wifi* and
*midwife* do not. The docstring honestly lists what still leaks: homonyms (a
real captured hit was *"Imagine yourself being an ALIEN WIF HAT"*), negative
attention, deliberate obfuscation, and mentions that live only in link bodies.

**The 422 investigation.** `_ARCTIC_SPACING_S = 0.6` carries a long finding:
the archive host's HTTP 422 *"Timeout. Maybe slow down a bit"* is **not rate
limiting**. That was established across 12 back-to-back 200s, a cold-process
reproduction, four different spacing values, a 105-second backoff, and 33
consecutive 200s across 11 field combinations. It is a ~3-second server-side
query timeout on a dense `(after, before)` range. The consequence: **no page
budget completes a 24-hour comment walk of a busy subreddit**, and the module
*reports the shortfall* rather than engineering around it and pretending the
window was covered.

**Privacy, Reddit API terms and audit C7 all shape the cache.** It stores
derived counts, timestamps and salted author hashes **only** — never post
bodies, never titles, never plaintext usernames. `_author_hash` is blake2b with
`digest_size=8`.

**No text extractor is implemented.** §8 of the audit permits an isolated,
schema-constrained extractor whose output is structured values only, but it also
requires an adversarial corpus and a locked ablation before such a feature may
influence anything. Neither exists, so the honest implementation is none.

Other constants: `_MAX_INDEX_LAG_SECONDS = 900.0`; `MIN_BASELINE_HOURS = 48`
(the 7-day z-score needs two days of buckets before it means anything);
`CACHE_VERSION = 2`, bumped when comments joined the sweep, with a per-symbol
`unit` field so that flipping `include_comments` invalidates only the affected
symbols instead of the whole cache.

### `prompts.py` — 717 lines. The cache boundary.

See §7 — it gets its own section. Nothing rendered here can place an order, and
the prompt says so to the model rather than implying authority it no longer has.

### `brain.py` — 438 lines. The optional model call.

One `client.messages.parse()` per slow tick, and only under
`[strategy] kind = "advisory"`. No retry loop, no hand-rolled JSON parser — the
SDK's structured-output path does both. What comes back is an
`AdvisoryDecision`: a suggestion a strategy may weigh, that `risk.py` bounds,
and that cannot become an order by itself.

- **`_validate` repairs nothing.** The old `_normalize` repaired four things
  Pydantic cannot express — an unknown symbol, a duplicated symbol, a hold with
  non-zero size, a negative size — and traded the repair. One case proves that
  instinct wrong: the repair tested `size < 0.0`, and every comparison against
  NaN is false, so a NaN size was not negative, not positive, not out of range.
  It passed the repair, passed `Field(ge=0.0)`, and then passed every risk
  comparison for the same reason. A model that emits an invalid value has
  malfunctioned, and the correct response to a malfunctioning component is to
  discard its entire output and record the failure.
- Errors: `BrainError` → `ModelCallError` (carries `retryable`) and
  `ModelOutputError`.
- `_thinking_from_response` concatenates `thinking` blocks and renders
  `redacted_thinking` as a marker. Storing the ciphertext would cost disk for
  bytes nobody can read; collapsing it to `None` would erase the distinction
  between "did not think" and "thought, redacted."
- `Usage` carries the four token counts plus the thinking text, and derives
  `total_input_tokens`, `cache_hit_rate` and `cost_usd` — the last by
  delegating to `ModelConfig.cost_usd`, never by re-deriving.
- The call passes `thinking={"type": "adaptive"}` and
  `output_config={"effort": cfg.model.effort}` alongside an
  `output_format` of `AdvisoryDecision`. A comment records that this shape was
  read out of the installed `anthropic` type stubs rather than from
  documentation, and flags precisely which parts remain unverified.
- Logs `prompt cache miss: 0 cached tokens of %d input tokens` — a warning that
  turned out to fire 48 times out of 49 in the real run. See §14.

### `risk.py` — 1,489 lines. The four layers.

See §8 — it gets its own section.

### `broker.py` — 1,209 lines. The paper book.

`LocalPaperBroker` implements the `Broker` protocol. **`SCHEMA_VERSION = 2`** —
the schema changed incompatibly when amounts became atomic integers and cash
became integer micro-USD, and a pre-audit `data/` directory therefore cannot be
loaded. Move it aside (`mv data data-v1`); `data-v*/` is gitignored for exactly
this.

- **Cash is integer micro-USD** (`_MICRO = 10**USDC_DECIMALS`) and quantities
  are atomic integers. There is no float accumulator anywhere in the book, so
  there is no epsilon to tune and no drift to reconcile.
- **The mode is enforced here.** The broker holds the `ExecutionMode`, and a
  mode that cannot mutate cannot write — that check lives in one place rather
  than at every call site that might remember to ask.
- **`pool_fee_micro()` returns 0, unconditionally, and that is not a bug.**
  Jupiter's `outAmount` is the number of tokens the pools actually send you, so
  every hop's AMM fee — along with slippage and price impact — is already
  deducted inside the route. Live evidence: BONK quoted at 2.9636e-6 to buy and
  2.9623e-6 to sell, a **4.4 basis point** round trip, against the **200 bp** a
  two-hop 0.25% charge would imply — **45x**. The old "degraded" path that
  justified charging a fee was deleted with the object it filled against (C4),
  so the function now exists only to say zero in one testable place.
- **Gas is additive and is charged on every attempt, including failed ones.** A
  failed Solana swap still pays the validator, so a failure is recorded as a
  `Fill` with `state=FAILED`, zero amounts and non-zero gas (audit C5c).
  Dropping the row would hide the cost.
- **The failure model is a configured iid coin flip, and the audit is right
  that this is not realism.** §15 asks for failures conditioned on congestion,
  quote age, route, priority fee and program error, and for observed priority
  fees rather than a constant `gas_usd_per_swap`. Neither is implemented; both
  are labelled assumptions rather than measurements so nobody cites them as
  evidence.
- **Cost basis includes fees and gas.** A position's break-even is its true
  break-even.
- `load()` refuses to overwrite a corrupt or future-versioned state file rather
  than silently resetting the book to $1,000, and `reconcile()` replays fills
  when state and ledger disagree.
- `save()` goes through `_atomic_write_text`: `mkstemp` in the same directory,
  write, `fsync`, `os.replace`, and unlink the temp file on **`BaseException`**
  — which includes `KeyboardInterrupt`, the single most likely way this process
  ever dies.
- The fill row is appended **before** state is saved, so a fill can never exist
  in the book without a row explaining it.

### `portfolio.py` — 565 lines. Marking and stops.

Rewritten against audit **C8**, which named it the single most dangerous file in
the project.

- `mark()` returns a `Mark` with a stated `basis` or an explicit absence, and
  **never** cost basis (§4.4). The old behaviour even had a passing test
  asserting it — `test_missing_mark_does_not_crash_and_carries_at_cost` — which
  is how a bug survives for months.
- Marks are ranked by how close they are to a number someone would actually
  receive. `"route"` — the output of a *sell* route quoted at the position's own
  size — is the only basis that estimates liquidation value, so it alone carries
  no haircut and is the only one risk will treat as executable. `"mid"` takes a
  haircut, an `"estimate"` a larger one, and beyond `max_mark_age_seconds` the
  position is `"unavailable"`.
- `stop_loss_breaches()` converts the config fraction to whole percent in
  **exactly one place**, uses `_PCT_EPSILON = 1e-9` (because
  `0.15 * 100 == 15.000000000000002` and a position exactly at the stop must
  fire), skips unmarkable positions, and returns **worst drawdown first** — if
  multiple stops fire on the same tick, the bleeding one goes first.

### `journal.py` — 833 lines. The durable ledger.

Audit C11 is the whole reason this file is more than forty lines. The old shape
appended a trade row and then wrote state: two files, two operations, no
transaction, no IDs. The replacement is an **event-sourced append-only ledger**
at `data/ledger.jsonl`, owned by `journal.Ledger`, carrying decisions, order
intents, state transitions and fills in one stream. Positions are derived from
immutable rows rather than mirrored in a second file, so there is no second file
to disagree with. Four properties make that safe:

1. **Every row carries identity** — `schema_version`, `run_id`, and whichever of
   `decision_id` / `action_id` / `intent_id` / `order_id` / `fill_id` apply. That
   is what lets `fills_by_intent` join a fill to the intent that caused it *by
   ID*. The old code matched fills to actions by symbol, so a stop-loss and a
   strategy SELL on the same coin in the same tick were indistinguishable, and
   the attribution in the report was a coin flip in exactly the situation you
   most want to understand.
2. **Rows sort themselves.** IDs are prefixed with a zero-padded 18-digit
   microsecond timestamp, so `sorted(rows, key=itemgetter("row_id"))` — or
   `sort` from a shell — is a correct chronological sort with no timestamp
   parsing.
3. **Appends are durable before they return.** Write, `flush`, `os.fsync`, then
   return. Without the fsync, `write` has only reached the OS page cache, and
   the one row you lose to a hard kill is the order that was in flight.
4. **Corruption is survivable and visible.** `scan()` reports a torn final line
   and any unparseable rows instead of failing, and `preflight` surfaces both.

`to_jsonable` recursively converts dataclasses, Pydantic models, enums and
tuples. It encodes `inf` as a **string** (because `TxnCounts.ratio` is
meaningfully infinite when a pool has zero sells) and `NaN` as `None`.

### `report.py` — 1,797 lines. Rendering.

Nothing here computes anything a trading decision depends on. It reads the
ledgers and the marked book and formats them. Keeping arithmetic out of the
display layer is why the numbers in `report` can be trusted to match the ledger
— the one exception, the spend summary, sums rows into a `Usage` object
specifically so that it shares one cost formula and one cache-hit definition
with `once`. (Both used to differ: the cost folded cache-creation into input and
billed it at 1x, and the hit rate divided by `input + cache_read` only, so the
same run reported a flattering rate in one place and a lower one in another.)

`collect()` reads the four record files and builds a `Report` of typed parts —
`RunSummary`, `PositionLine`, `TradeStats`, `DecisionSummary`, `CostBreakdown`,
`ModelSpend`, `IntegrityReport`, `Sample`. Three of those exist to keep the
reader honest:

- **`ModelSpend.relevant`** is false when no advisory call was made, and the
  renderer then prints **"Model spend not applicable"** rather than a `$0.00`
  that looks like a measurement of something.
- **`IntegrityReport`** carries what the ledger scan found — torn lines,
  unparseable rows, intents with no terminal state — so a report over a damaged
  record says so on its face.
- **`Sample`**, with `ADEQUATE_ROUND_TRIPS` and `ADEQUATE_RUN_HOURS`, states
  whether there is enough data for any of the numbers above to mean anything.
  Most short runs are labelled inadequate, on purpose.

**Deliberately not computed here: Sharpe, Sortino, Calmar and maximum
drawdown**, along with their confidence intervals. Every one of those needs an
equity curve sampled at fixed intervals, and the record contains fills, not a
curve — the fast tick marks the book but does not persist the mark. Deriving a
Sharpe from five fills produces a number with no standard error that somebody
would then quote. The report says this on its face rather than leaving a gap
where a reader assumes an oversight. The fix is to persist a periodic equity
series, at which point all four become computable and comparable at once.

Details: `_utf8()` forces UTF-8 on stdout/stderr because on Windows Python
picks the ANSI code page for a *redirected* stream — `memetrader status | tail`
died with `UnicodeEncodeError` on a `↓` while the same command in a terminal
was fine. `_price()` switches to scientific notation below $0.01, because
memecoin prices span nine orders of magnitude and a fixed precision either
prints `$0.00` for BONK or a wall of zeros for WIF. Thinking traces are clipped
to 280 characters in the decision list — the full text is never discarded, it
stays in `decisions.jsonl`.

### `loop.py` — 1,251 lines. The scheduler.

`Trader` owns preflight, the two cadences, target diffing, the four-step
execution ordering (§5) and the persisted risk ledger. Two details are worth
stating here rather than in §5:

- **The risk ledger is persisted to `data/risk_ledger.json`**, because drawdown
  peaks, loss windows, quarantines and failure streaks are state that must
  survive a restart — otherwise every crash resets the breakers that exist to
  notice a crash-shaped problem. If the file exists and **cannot be read, the
  run starts halted** rather than starting from an empty risk history, because
  an unreadable risk record and a clean one are not the same thing.
- **A halted run permits exits.** Halting a system that holds inventory it
  cannot sell is not safety, it is a trap.

### `cli.py` — 511 lines. Typer entry point.

See §12. `--dry-run` is gone; `--mode` replaced it.

---

## 7. The prompt and the cache boundary

This section describes the **advisory** path only. Under the default `baseline`
strategy nothing in `prompts.py` or `brain.py` runs at all.

**Prompt caching is a byte-exact prefix match.** The cached prefix ends at the
first byte that differs, and everything after it is re-processed and re-billed
at the write rate. One volatile character near the top therefore invalidates the
entire prompt, not the line it sits on. That single fact determines the entire
structure of `prompts.py`, and §14 is what it costs when you get it wrong.

The prompt also states plainly that the model is advising, not ordering:
targets it proposes are bounded by the risk layer whatever it says, and
"advising past a limit does not raise it; it only produces a bounded or vetoed
intent and wastes the tick."

### `build_system(...)` → two blocks, both `cache_control: ephemeral`

**Byte-frozen for the life of a run.** It depends only on the configured
universe and the risk and cadence settings — values that cannot change without
a restart. `tests/test_prompts.py` asserts that two builds at different
wall-clock times are byte-identical, and pins `system_fingerprint`, a 16-hex
digest that is cheap enough for a log line to carry and is recorded alongside
the model name on every decision, so a change in advisory behaviour can be
attributed to a prompt edit rather than argued about.

Block 1 is the role, the universe and the evidence rules; block 2 is the limits,
the lessons and the output rules. The split is placed after the evidence rules
because Opus's **minimum cacheable prefix is 512 tokens**, so a terse system
prompt would not be cacheable at all, and the second marker sits at the end of
the block, which is what the per-tick user turn actually matches against.

The frozen half contains:

- **The rules of the world.** The exact limits the risk layer enforces —
  rendered by `_limits_block` from config — *and* which limits deliberately do
  not exist (no max trades per day, no minimum hold time).
- **How to read each stream, in descending order of trust** (§4.2).
- **"What the last project taught us"** — four empirical findings carried
  forward from the predecessor system:
  1. Trailing stops produced **269 exits at roughly -2.26% each**. They
     converted noise into realized losses.
  2. Cutting from **eight signals to one or two** turned **-16.62%** into
     **+5.39%**.
  3. **Never DCA a loser.**
  4. **Nearly every added layer of sophistication hurt.**

### `render_user(...)` → everything volatile

Every number that changes tick to tick lives here, after the cache boundary: the
clock, the book, the evidence bundles, the recent decision history and the risk
bounds from the previous tick.

**No untrusted text is interpolated anywhere in it, by construction.** The only
external strings are symbols from local config and reason strings generated
inside this codebase. The old `_sentiment_lines` rendered Reddit titles and
bodies verbatim; the defence now is the absent field, not an absent line (§4.3).

Rendering rules:

- **Every helper prints `None` as `n/a`**, with explicit inline annotation
  where a reader might misinterpret it — e.g.
  `<- n/a = not yet indexed by the source, NOT zero attention`.
- `_level()` handles prices spanning ten orders of magnitude.
- `_liquidity_trend` prints the measurement window alongside the percentage, so
  the number can never be silently read as the cadence.
- The prompt closes with:
  `Return one action per coin in the universe. Name the number that drove each
  one, and return HOLD at 0.0 where you have no read.`

`tests/test_prompts.py` asserts **byte-identity** of the system blocks across
completely different evidence bundles. If someone accidentally interpolates a
price into the system prompt, that test fails.

---

## 8. Risk layer reference

Risk does not approve and it does not mutate orders. It returns **`RiskBounds`**
— a maximum notional, the vetoes that fired with their rule names and reasons,
the binding rule, and any rules bypassed. The difference matters: an approval is
a permission that downstream code acts on, whereas a bound is a ceiling that
downstream code must respect, and a ceiling cannot be widened by being passed
around.

`RiskEngine` composes four layers and exposes `entry_bounds`, `exit_bounds` and
`confirm_quote`. All of it is pure — no I/O, no clock of its own — and the
persisted `RiskLedger` is passed in rather than read.

### Layer 1 — `EligibilityRisk`: may we touch this symbol at all?

Vetoes, in order: `not_in_universe`, `quarantined`, `entry_cooldown`,
`missing_snapshot`, `degraded_snapshot`, `untrusted_quote_token`, `no_price`,
`no_liquidity_observation`, `min_liquidity`, `stale_data`, `unknown_pool_age`,
`new_pool`.

Two of these are absences rather than failures, and they veto anyway.
`no_liquidity_observation` is not `min_liquidity` with a zero — one says the
pool is too thin, the other says we never learned how thick it is.
`unknown_pool_age` vetoes because `require_known_pool_age = true`: an unknown
age is not a pass.

### Layer 2 — `PortfolioRisk`: how much of the book may this become?

Returns **caps**, each named: `max_position_pct`, `max_gross_exposure_pct`,
`max_net_exposure_pct`, `max_sleeve_exposure_pct`, `min_cash_floor_pct`,
`volatility_target`, `uncalibrated_forecast`.

`max_sleeve_exposure_pct` is audit **C10** and is the one people delete by
accident: BONK, WIF and POPCAT are not three bets, they are one factor with
three tickers. Treating them as independent is how a 30% per-symbol cap becomes
90% of the book in the same trade, so the correlated sleeve defaults to the
configured universe.

`uncalibrated_forecast` is the counterpart to flat sizing (§6): a forecast with
no `calibration_id` may not enlarge a position, however confident it sounds.

When `total_value_usd` is `None` — a book with an unmarkable position — this
layer **vetoes entries** rather than dividing by a number partly made of cost
basis (audit C8).

### Layer 3 — `PreTradeRisk`: is this specific order sane?

Caps: `max_entry_usd`, `insufficient_cash`, `max_depth_participation_pct`.
`check_quote` vetoes: `quote_mismatch`, `quote_expired`, `quote_age`,
`max_price_impact`.

`check_quote` is deliberately **one method shared by sizing and binding**, so
the checks run when the order is bound are *literally* the checks run when it
was sized. The audit's "quote changes after risk approval" row exists because
those were two pieces of code.

### Layer 4 — `ContinuousRisk`: should we be trading at all right now?

Evaluated every tick, including ticks with no order in them, because a
drawdown, a failure streak or a data outage arrives *between* orders and the old
system could only notice at the moment it was about to trade anyway.

Halt reasons: the manual kill switch, `max_drawdown_pct` from peak,
`max_daily_loss_pct`, `max_window_loss_pct` over `loss_window_seconds`, and
`max_consecutive_failures`. The kill switch is a plain boolean with no
conditions on it, because §11 of the audit is explicit that it is the only
control that reliably works when something unanticipated is happening.

An unmarkable position does **not** engage the kill switch — that would be
disproportionate to one missing price — but it does fail data health, which
blocks every entry while leaving every exit open, and quarantines the symbol.

**`halted` permits exits.** Halting a system that holds inventory it cannot sell
is not safety, it is a trap.

### The forced-exit asymmetry

`exit_bounds(..., forced=True)` differs from `entry_bounds` on purpose:

- `halted` does not block an exit. Once halted, exits are the only thing
  permitted.
- A degraded snapshot or a `ValuationEstimate` does not block an exit. C4's
  veto is an entry veto.
- A quarantine does not block an exit. It exists to stop a rebuy.
- **`min_liquidity` and `max_price_impact` are bypassed on a forced exit.** Both
  fire hardest when a pool is collapsing, which is the exact scenario the stop
  exists for. Enforcing them there does not protect the position, it traps it —
  and the cost of being trapped is unbounded while the cost of a bad fill is
  bounded by what is left.
- **`stale_data` is still enforced**, forced or not. This is the one rule a stop
  does not get to argue with, and the reason is measured rather than aesthetic:
  the fast tick retries in 60 seconds, so blocking costs a minute, whereas
  exiting on a price we cannot vouch for costs the fill.

Every bypass is recorded in `RiskBounds.bypassed_rules`, so a bad forced fill is
visible after the fact rather than silent.

When a position is **unmarkable**, the exit bound falls back to cost basis, noted
loudly. That is a ceiling on permission, not a valuation: execution is bounded by
inventory in atomic units, so a ceiling that is too high cannot cause an
over-sell, while a ceiling that is too low would trap exactly the position that
most needs to leave. It is the only place in the system where cost basis is
still allowed near a price, and only for that reason.

---

## 9. Fill realism

A paper fill that ignores costs teaches a strategy to churn. Each cost modelled
here is real:

1. **Routing.** Every fill is priced from a live **Jupiter Quote API** route for
   the actual notional, and the integer `inAmount`/`outAmount` it returns *are*
   the fill. Real multi-hop routing and real depth, not a mid-price fantasy.
2. **Price impact.** Jupiter's `priceImpactPct`, converted from fraction to
   whole percent at the boundary, enforced against a 3% ceiling at sizing time
   **and again** against the quote that will actually be sent.
3. **Pool fees.** `$0.00`, unconditionally — and that is correct, because the
   route already has them deducted. See `pool_fee_micro` in §6.
4. **Gas.** `$0.21` per swap attempt, charged **even when the transaction
   fails**, and recorded as a `FAILED` fill rather than dropped.
5. **Failed transactions.** `failed_tx_rate = 0.06`, drawn once per attempt
   after validation so a seeded RNG replays a run exactly.

**No quote, no fill.** There is no degraded path: if Jupiter does not answer,
the trade does not happen. That is a deliberate loss of activity in exchange for
never manufacturing a trade out of an outage.

**What this model is not.** The failure rate is an assumption, not a
measurement, and so is the flat gas figure. Audit §15 asks for failures
conditioned on congestion, quote age, route and program error, and for observed
priority fees; neither is implemented, and both are labelled as assumptions in
the code so nobody cites them as evidence.

The consequence is quantified and it matters: **on a $1,000 book at $0.21 per
swap, a round trip needs roughly +0.28% gross just to break even on gas.** §14
shows a directionally-correct trade that lost money anyway because of exactly
this.

---

## 10. Data, ledgers and file layout

```
memecoin-trader/
├── config.toml               # all tunables, heavily commented
├── pyproject.toml            # deps, scripts, ruff, mypy
├── uv.lock
├── .env                      # secrets — GITIGNORED, never committed
├── .env.example              # the template
├── README.md
├── OVERVIEW.md               # this file
├── src/memetrader/
│   ├── __init__.py
│   ├── types.py              # shared shapes + the Broker protocol
│   ├── ids.py                # identity for every record
│   ├── config.py             # load + validate
│   ├── http.py               # client factory + retry/breaker policy
│   ├── market.py             # DexScreener + GeckoTerminal + safety screen
│   ├── quotes.py             # Jupiter routing
│   ├── signals.py            # pandas technicals
│   ├── sentiment.py          # Reddit attention (off by default)
│   ├── strategy.py           # cash | baseline | advisory
│   ├── prompts.py            # system + user prompt (advisory only)
│   ├── brain.py              # the optional model call
│   ├── risk.py               # the four risk layers
│   ├── broker.py             # the paper book
│   ├── portfolio.py          # marking + stops
│   ├── journal.py            # the event-sourced ledger
│   ├── loop.py               # preflight + the dual-cadence scheduler
│   ├── report.py             # rendering
│   └── cli.py                # typer commands
├── tests/                    # 15 files, 900 cases
├── tools/
│   ├── document_run.py       # the documented-run harness
│   └── run_12h.cmd           # Task Scheduler wrapper
├── data/                     # GITIGNORED — the live record
│   ├── ledger.jsonl          # decisions, intents, transitions, fills
│   ├── state.json            # the book: cash, positions, realized P&L (v2)
│   ├── trades.jsonl          # one row per fill
│   ├── intents.jsonl         # one row per order intent, pre-submission
│   ├── decisions.jsonl       # legacy; read by report, no longer written
│   ├── risk_ledger.json      # drawdown peak, loss windows, quarantines
│   └── sentiment_cache.json  # derived counts + salted hashes only
├── data-v1/                  # GITIGNORED — a pre-schema-v2 data/, moved aside
└── runs/                     # GITIGNORED — documented run output
    └── <UTC timestamp>/
        ├── SUMMARY.md        # live status, rewritten every tick
        ├── SYSTEM-PROMPT.md  # the cached system prompt, verbatim
        ├── TRADES.md
        ├── FAST-TICKS.md     # minute-by-minute equity curve
        ├── EVENTS.md         # timeline + anything abnormal
        ├── RESULTS.md        # post-run analysis
        ├── console.log
        └── ticks/tick-NNNN-*.md
```

**`data/`, `data-v*/` and `runs/` are gitignored on purpose.** They are
generated, not source; `data/` is a trading record; and a run's console log can
carry environment details that do not belong in a repository. `.env` is
gitignored because it holds live credentials.

### Who owns what

`journal.Ledger` owns `ledger.jsonl`, which is where a decision row is now
written. The broker owns `state.json`, `trades.jsonl` and `intents.jsonl`. The
loop owns `risk_ledger.json`. `decisions.jsonl` is legacy: nothing writes it any
more, and `report.collect` still reads it so that a report over a pre-rewrite
run still works. One writer per file, always — two writers is how two files come
to disagree, which is the failure C11 was about.

### The schema change

`state.json` is **schema v2**. A pre-audit v1 directory cannot be loaded — the
representation changed from floats to atomic integers and micro-USD, and there
is no migration that could invent the atomic amounts the old rows never
recorded. `load()` refuses rather than resetting the book, so the operator's
move is explicit:

```bash
mv data data-v1
```

`data-v*/` exists in `.gitignore` for exactly this reason.

---

## 11. Configuration reference

All of it lives in `config.toml` (300 lines, extensively commented). Secrets
live in `.env`.

### Top level
| Key | Value | Meaning |
| --- | --- | --- |
| `execution_mode` | `"paper"` | `read_only` \| `paper` \| `live`. `live` is refused. `--mode` overrides this for one invocation. |

### `[strategy]`
| Key | Value | Meaning |
| --- | --- | --- |
| `kind` | `"baseline"` | `cash` \| `baseline` \| `advisory`. The default is not the model. |
| `horizon_seconds` | `3600.0` | The window a forecast is a forecast *of*. |
| `entry_hurdle_pct` | `1.0` | The **lower** quantile of expected net return must clear this. |
| `flat_size_usd` | `25.0` | Flat sizing, on purpose — see §6. |
| `min_trade_usd` | `10.0` | Must be ≥ `[risk] min_trade_usd`; validated at load. |
| `rebalance_band_usd` | `15.0` | Drift smaller than this is left alone. |
| `max_positions` | `2` | |

### `[portfolio]`
| Key | Value | Meaning |
| --- | --- | --- |
| `starting_cash_usd` | `1000.0` | The paper book. |

### `[[coins]]` — three entries
| Symbol | Mint | Aliases |
| --- | --- | --- |
| BONK | `DezXAZ8z…` | bonk, $bonk |
| WIF | `EKpQGSJt…` | wif, dogwifhat, $wif |
| POPCAT | `7GCihgDB…` | popcat, $popcat |

### `[model]` — used only when `[strategy] kind = "advisory"`
| Key | Value |
| --- | --- |
| `name` | `claude-opus-5` |
| `effort` | `xhigh` |
| `max_tokens` | `8000` |
| `price_input_per_mtok` | `5.00` |
| `price_output_per_mtok` | `25.00` |
| `price_cache_read_per_mtok` | `0.50` |
| `price_cache_write_per_mtok` | `6.25` |

### `[cadence]`
| Key | Value |
| --- | --- |
| `fast_tick_seconds` | `60` |
| `slow_tick_seconds` | `900` |

### `[risk]` — eligibility
| Key | Value | Meaning |
| --- | --- | --- |
| `max_snapshot_age_seconds` | `90` | Stale data → no trade, even on a forced exit. |
| `min_liquidity_usd` | `50_000.0` | Thinner than this, you are the exit liquidity. |
| `min_pool_age_seconds` | `86_400` | A pool younger than a day has no history. |
| `require_known_pool_age` | `true` | An unknown age is a veto, not a pass. |
| `post_stop_quarantine_seconds` | `3600` | No re-entry into what just stopped out. |
| `min_seconds_between_entries` | `900` | One entry per symbol per slow tick. |

### `[risk]` — portfolio
| Key | Value | Meaning |
| --- | --- | --- |
| `max_position_pct` | `0.30` | **Fraction** of book value, per symbol. |
| `stop_loss_pct` | `0.15` | **Fraction**. Forced exit at -15% from entry, on the fast tick. |
| `max_gross_exposure_pct` | `60.0` | Whole percent. |
| `max_net_exposure_pct` | `60.0` | |
| `min_cash_floor_pct` | `10.0` | Always keep enough to pay gas and exit. |
| `correlated_sleeve` | `["BONK","WIF","POPCAT"]` | Audit C10: one factor, three tickers. |
| `max_sleeve_exposure_pct` | `60.0` | The cap that makes the sleeve mean something. |
| `target_volatility_pct` | `8.0` | |
| `require_volatility_estimate` | `true` | No estimate, no entry. |

### `[risk]` — pre-trade and continuous
| Key | Value | Meaning |
| --- | --- | --- |
| `default_entry_usd` | `25.0` | |
| `max_entry_usd` | `100.0` | Must be ≥ `[strategy] flat_size_usd`. |
| `min_trade_usd` | `10.0` | Below this, costs dominate. |
| `max_price_impact_pct` | `3.0` | Whole percent, checked twice. |
| `max_quote_age_seconds` | `10.0` | Older than this is not a price. |
| `max_depth_participation_pct` | `1.0` | Never take more than 1% of pool depth. |
| `gas_usd_per_swap` | `0.21` | |
| `assumed_pool_fee_pct` | `0.0` | Zero, because the route already charged it. |
| `max_daily_loss_pct` | `5.0` | |
| `max_window_loss_pct` | `10.0` over `loss_window_seconds = 604_800` | 7 days. |
| `max_drawdown_pct` | `20.0` | From peak book value; **halts the run**. |
| `max_consecutive_failures` | `3` | Failed swaps in a row → halt, not retry. |

### `[risk.mark]`
| Key | Value | Meaning |
| --- | --- | --- |
| `mid_haircut_pct` | `2.0` | A mid is a reference, so discount it. |
| `estimate_min_haircut_pct` | `5.0` | An estimate is worse than a mid. |
| `max_mark_age_seconds` | `120.0` | Beyond this the position is **unmarkable**. |
| `allow_mid_stop_reference` | `true` | A mid may trigger a stop even though it may not size an entry. |

The file carries an explicit note: *"No max-trades-per-day and no minimum hold
time. Deliberate."*

### `[execution]`
| Key | Value |
| --- | --- |
| `slippage_bps_fallback` | `50.0` — used only when Jupiter is unreachable |
| `gas_usd_per_swap` | `0.21` |
| `failed_tx_rate` | `0.06` |
| `default_pool_fee_pct` | `0.25` |
| `[execution.pool_fee_pct]` | per-venue table (Raydium, Orca, Meteora, …) |

### `[data]` and `[market]`
API base URLs for DexScreener, GeckoTerminal and Jupiter (lite and keyed),
`http_timeout_seconds` (now the *read* timeout — see `[http]`), candle counts,
and `max_mints_per_request = 30`, DexScreener's documented batch ceiling.

### `[safety_screen]`
| Key | Value | Meaning |
| --- | --- | --- |
| `min_liquidity_usd` | `50_000.0` | |
| `min_pool_age_seconds` | `604_800` | One week. |
| `min_liquidity_to_fdv` | `0.005` | A $1bn FDV on a $200k pool is not a market. |
| `max_snapshot_age_seconds` | `900.0` | |
| `require_trusted_quote` | `true` | Only price against SOL/USDC/USDT pairs. |

The table's comment names the five checks this screen cannot perform without an
on-chain RPC — mint and freeze authority, LP ownership and lock, holder and dev
concentration, Token-2022 extensions, a sellability probe — and every
`SafetyVerdict` repeats them in `deferred`, so the gap is visible in the record
rather than assumed away.

### `[http]`
| Key | Value | Meaning |
| --- | --- | --- |
| `connect_timeout_seconds` | `5.0` | Four phases, separately, because one scalar timeout is how a hung read eats a tick. |
| `read_timeout_seconds` | `12.0` | |
| `write_timeout_seconds` | `10.0` | |
| `pool_timeout_seconds` | `5.0` | |
| `max_attempts` | `3` | |
| `retry_budget_seconds` | `20.0` | Must fit inside a fast tick; load rejects anything over 45s. |
| `backoff_base_seconds` / `_multiplier` / `_max_seconds` | `0.25` / `3.0` / `4.0` | |
| `max_retry_after_seconds` | `10.0` | Cap on an honoured `Retry-After`. |
| `breaker_failure_threshold` | `4` | Then the host is cut off for the cooldown. |
| `breaker_cooldown_seconds` | `60.0` | Open hosts become a data-health signal. |

### `[sentiment]`
| Key | Value | Note |
| --- | --- | --- |
| `enabled` | **`false`** | Changed from `true` by the audit. Off until the ablation it specifies has been run. |
| `cache_ttl_seconds` | `600` | **Deliberately below the slow tick.** It is a debounce for interactive `status`/`once`, not a cost control — a cache that outlived the cadence would feed a decision stale attention data. |
| `lookback_hours` | `24` | |
| `baseline_days` | `7` | The z-score baseline. |
| `include_comments` | `true` | Flipping this bumps the per-symbol cache `unit`. |
| `subreddits` | CryptoCurrency, CryptoMoonShots, SatoshiStreetBets, solana, pumpfun | |

### `[prompt]`
| Key | Value |
| --- | --- |
| `decision_history` | `10` |

### `.env`
```
ANTHROPIC_API_KEY=...        # only for [strategy] kind = "advisory"
JUPITER_API_KEY=...          # optional — keyed host instead of lite-api
REDDIT_CLIENT_ID=...         # optional — official API instead of Arctic Shift
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=...
```

---

## 12. CLI reference

Installed as `memetrader` (`[project.scripts]` → `memetrader.cli:app`).

The app's own help line: *"A deterministic strategy (optionally advised by
Claude) trading three Solana memecoins against a paper book."*

| Command | Options | What it does |
| --- | --- | --- |
| `memetrader check` | — | Validate config, confirm every data source answers, screen every token. **Run this first.** |
| `memetrader status` | `-v/--verbose` | Every evidence stream for every coin, plus the book. No strategy call. |
| `memetrader once` | `--mode`, `-v/--verbose` | One decision cycle: evidence → strategy → risk → quote → execute. |
| `memetrader run` | `--max-ticks`, `--mode`, `-v/--verbose` | The dual-cadence loop. Ctrl+C is safe — state is saved after every change. |
| `memetrader report` | `--trades` (20), `--decisions` (5), `--live-marks / --no-live-marks` (on) | P&L, trade history, decision history, integrity and spend to date. |
| `memetrader reset` | `-y/--yes` | Wipe `data/` back to the starting cash. Irreversible. |

### `--mode`, verbatim from the help text

> `read_only` = decide and show everything, mutate nothing. `paper` = trade the
> simulated book. `live` = refused; there is no wallet. Defaults to
> `config.toml`'s `execution_mode`.

**`--dry-run` is gone.** It was a boolean, and a boolean cannot express the
difference between *do not trade* and *do not write anything*: the old flag
suppressed journaling in some places and execution in others, and which of the
two it meant depended on the call site that checked it. The replacement is a
three-valued capability held by the **broker**, so a mode that cannot mutate
cannot write, regardless of who forgot to ask.

`--live-marks / --no-live-marks` on `report` decides whether current marks are
fetched. `--no-live-marks` makes the report fully offline; it does not
substitute cost basis for a price, it reports the absence.

Two operational notes baked into the code: `report` fetches prices **without
candles**, so it does not compete for GeckoTerminal's rate limit with a `run`
loop in another terminal; and `run` installs handlers for SIGINT, SIGTERM and
SIGBREAK, sleeping in short slices so Ctrl+C is responsive rather than waiting
out a full 60-second nap.

**Exit codes.** `1` is a bad input — a config error, an unrecognised `--mode`.
`2` is a refusal: `--mode live` (the broker raises `LiveModeUnsupported` in its
own constructor, so live cannot get as far as building a book) and
`StartupRefusal` from preflight. Both print a message an operator can read
rather than a traceback that looks like the system fell over, and the refusal
names `memetrader reset`, the only in-tree way to discard a paper book you have
decided not to reconcile.

---

## 13. Tooling: the documented run harness

### `tools/document_run.py` — 1,097 lines

Wraps `loop.Trader` and writes Markdown, because the point of a long run is not
the P&L — it is having enough on disk afterwards to ask *why* each trade looked
like a good idea at the time.

Three design decisions are worth understanding before changing it:

**There is not always a prompt to capture.** Audit C6 took the model off the
decision path, so under the default `baseline` strategy there is no prompt, no
token count and no spend, and the harness says so rather than rendering empty
sections that imply a model ran.

**When there is one, it captures the prompt rather than reconstructing it.**
`brain` looks up `build_system` and `render_user` in its own module namespace,
so the harness rebinds those two names and records what they actually returned.
Re-rendering afterwards from the `TickResult` would produce a *near-miss*: by
the time `slow_tick` returns, the bounds have been replaced with this tick's and
the ledger has grown by a row, so both the history and bounds sections would
differ from what the model was really shown. A near-miss is worse than nothing
when the whole file claims to be the input to a specific decision.

**The supervisor restarts; it does not resume.** If `Trader.run` dies for any
reason, a fresh `Trader` is constructed and the run continues to the deadline.
That is safe because the book lives in `data/state.json` and is written
atomically after every mutation. The only things lost are in-memory —
`decision_baseline` (so the next liquidity trend spans the outage *and says
so*) and `previous` — and both are reported in `EVENTS.md` rather than papered
over.

**Nothing here influences a decision.** It is a pure observer. It adds
no risk rule, no spend cap and no retry that the live loop would not have done
on its own, so the decisions it documents are exactly the decisions
`memetrader run` would have made unattended. Recorder failures are caught and
logged, so a documentation bug can never take down trading.

A small but real detail: Markdown fences use **tildes**, not backticks, because
the captured prompt is free text the harness does not control and a stray
triple-backtick inside it would end the fence early and scramble the file.

### `tools/run_12h.cmd` — Windows Task Scheduler wrapper

Launched by Task Scheduler so a long run is independent of any terminal,
console window or job object. Closing the terminal that created the task does
not touch it.

Credentials come from `.env`. This is load-bearing: Claude Code injects
`ANTHROPIC_API_KEY` and `ANTHROPIC_BASE_URL` into *its own child shells only*,
so a run launched any other way saw no credentials and every model call failed
with *"Could not resolve authentication method."*

Operational lessons from actually doing this:

- Set `MultipleInstances: IgnoreNew` — otherwise two traders can end up running
  against one `state.json`.
- `schtasks /End` **orphans descendant processes.** It reports SUCCESS and
  leaves the Python process running. Verify by listing processes, and kill the
  tree with `taskkill /F /T /PID`.
- `StopIfGoingOnBatteries` defaults to true and will kill a long run the moment
  the laptop is unplugged. Fix it with `Set-ScheduledTask`.

---

## 14. Results of the 12-hour run

**This run predates the adversarial audit, and it is the evidence that motivated
most of it.** It was made by the *old* system: the model chose and sized every
trade, Reddit sentiment was on, marks fell back to cost basis, quotes had a
degraded fallback, and amounts were floats. Nothing here describes how the
current code behaves, and none of these numbers is a measurement of the
`baseline` strategy — they are kept because the failures they exposed are the
reason the code changed, and deleting the evidence would leave the reasoning
unsupported. Read it as a post-mortem, not as a performance record.

Window: 2026-09-20 02:49 → 14:49 UTC. Model `claude-opus-5`, effort `xhigh`,
adaptive thinking. $1,000 paper book. Full analysis in
`runs/20260920T024841Z/RESULTS.md`.

### Headline

| | |
| --- | --- |
| Final book value | **$996.81** |
| Total return | **-0.32%** (-$3.19) |
| Realized P&L | -$0.25 |
| Unrealized P&L (2 open) | -$2.73 |
| Gas paid | $1.05 |
| Pool fees | $0.00 *(correctly)* |
| **LLM spend** | **$3.48** |

**The model cost 11x more than the trading lost.** For a 12-hour window on a
$1,000 book that is the dominant economic fact of the run, and it is one of the
reasons the model is no longer the default: a component that costs $7/day has to
demonstrate that it beats a deterministic rule on the same evidence, and nothing
here demonstrates that either way.

### Execution health — clean

49 slow ticks + 671 fast ticks = **720 minutes exactly**. No gaps. Zero failed
model calls, zero supervisor restarts, zero recorder failures, zero stop-losses
fired.

### What it did

147 decisions (49 ticks × 3 coins):

| Action | Count | Share | Mean confidence |
| --- | ---: | ---: | ---: |
| HOLD | 142 | 96.6% | 0.658 |
| BUY | 4 | 2.7% | 0.500 |
| SELL | 1 | 0.7% | 0.580 |

**The model was measurably less confident when it traded than when it held.**
The most confident buy in twelve hours was 0.58. Whatever policy it is running,
it treats action as the uncertain choice and inaction as the safe one.

WIF was never traded — held all 49 ticks.

### The one closed trade — the most instructive event in the run

**The BONK round trip was directionally correct and still lost money.**

- Bought 03:34 at `2.9705e-06`
- Sold 07:49 at `2.9739e-06` — *higher*
- Rebought 09:04 at `2.9590e-06` — *cheaper than it sold*

The model called the top of a small range, exited, and re-entered lower. The
sell reasoning cited specific numbers: h1 buy/sell 0.52 against h24 0.95, both
timeframes' EMAs bearish, RSI falling on both, volume at 0.08x its 20-period
mean — *"Exit flat rather than wait for the -15% stop to be the
decision-maker."*

It was right, and it booked **-$0.25** anyway. The gross move was about +0.11%
on $150 ≈ $0.17, against $0.42 of gas across two legs plus $0.21 more for a
failed retry. **Transaction costs consumed a correct call.**

### Equity curve

Peak $1,001.43, trough $995.20, close $996.81. Max drawdown **-0.62%**. Total
range across twelve hours: $6.23. The stop-loss machinery ran 671 times and
correctly did nothing.

### Two things that look like bugs and are not

- **$0.00 pool fees on every fill** — correct. Jupiter's quoted price already
  has the pool fee in it (§6, §9).
- **One failed transaction in five swaps** — correct. 20% against a configured
  6% rate is high, but entirely ordinary at n=5.

### The bug the run exposed: the prompt cache never worked

| | |
| --- | --- |
| Cache read tokens | 3,683 |
| Cache write tokens | 176,784 |
| **Effective hit rate** | **1.0%** |

48 of 49 ticks missed. The run paid the 1.25x cache-*write* premium on nearly
every tick and collected the 0.1x read discount exactly once.

**Cause.** The first diagnosis was the default **5-minute cache TTL** against
`slow_tick_seconds = 900` — the cache expiring before the next tick arrives. The
audit found a second cause underneath it, and `prompts.py` now records that one
as the real one: **volatile content inside the system block** — limits rendered
from live values, a clock, a portfolio summary. Caching is a byte-exact prefix
match, so one changing character near the top re-bills the entire prompt at the
write rate no matter what the TTL is. The TTL mismatch would have capped the hit
rate; the volatile prefix guaranteed it was near zero.

The fix was structural: `build_system` is now byte-frozen for the life of a run
and everything volatile moved into `render_user` (§7), with a test asserting
byte-identity across builds.

**Cost:** 176,784 write tokens at $6.25/Mtok = **$1.10**. As cache reads at
$0.50/Mtok they would have cost **$0.09**. **$1.02 wasted out of $3.48 — 29%
of the run.**

`brain.py` logged `prompt cache miss` 48 times and was right every time.

**What remains after the structural fix:** the TTL question. Either request the
1-hour cache TTL on the system blocks, or drop `slow_tick_seconds` below 300, or
accept the miss and stop paying the write premium by removing the breakpoints.
The configuration this run used was the worst of the three — it paid for caching
and received none.

### Using this data

- **Do not train on the P&L.** One closed trade is one data point. -0.32% over
  12 hours is indistinguishable from noise.
- **The 142 holds are the dataset.** Each carries a full justification written
  against the live evidence bundle that produced it, and each tick file
  contains the verbatim prompt. That is 142 labelled (state → policy →
  rationale) tuples. The policy lives in the holds.
- **Price transaction costs explicitly.** Gas was $1.05 against $0.25 of
  realized loss — 4x the trading outcome. Any reward signal that ignores gas
  will learn to churn.
- **Treat sentiment as partially blind.** `mention_zscore_7d` was `null` for
  the entire window: it needs 48 hourly buckets and the `CACHE_VERSION` 1→2
  bump reset that history right before the run. All 147 decisions were made
  without the cross-coin attention-anomaly signal.

---

## 15. Cost model

**Under the default `baseline` strategy there is no model call and therefore no
model spend.** `memetrader report` prints **"Model spend not applicable"** in
that case — `ModelSpend.relevant` is false — rather than a `$0.00` that reads
like a measurement of a cheap model instead of the absence of one. The running
costs that remain are gas and the opportunity cost of being wrong.

Everything below applies only to `[strategy] kind = "advisory"`.

`ModelConfig.cost_usd` is the only formula. Per million tokens at the configured
prices: input $5.00, output $25.00, cache read $0.50, cache write $6.25.
`cache_write` is deliberately a *required* argument, so a new call site cannot
silently price cache creation at 1x instead of 1.25x.

- **Measured, pre-audit:** $3.48 per 12 hours at `xhigh` ≈ **$7/day ≈
  $210/month** at a 15-minute cadence — and 29% of that was the cache defect, so
  the comparable figure after the prompt was frozen is nearer $150/month.
- **`effort` is the largest single lever**, because it is the thinking tokens
  that move. `low | medium | high | xhigh | max` are accepted, exactly what the
  API accepts, so nothing valid is rejected as a typo. Every decision row
  records the model and effort that produced it, so an A/B in the ledger is a
  one-line config change.
- **Running in bursts is the cheapest way to keep this in the tens of dollars**
  and costs nothing in code: the loop resumes cleanly from the persisted book.

`memetrader report` prints spend to date, per-call cost, a projected daily run
rate at the current cadence, and the cache hit rate. It emits an explicit
warning when there are zero cache reads across multiple calls:

> `! zero cache reads across multiple calls — the stable prompt prefix is being
> invalidated, and you are paying several times over.`

---

## 16. Tests

**900 test cases across 15 files, ~12,400 lines** — more test code than source
code.

| File | Cases | Focus |
| --- | ---: | --- |
| `test_config.py` | 206 | Every validation path, the cost formula, fee-per-hop summation, mode and strategy-kind parsing, the strategy↔risk cross-checks |
| `test_risk.py` | 107 | All four layers, every cap and veto name, the forced-exit bypass set, halt conditions, `confirm_quote` |
| `test_sentiment.py` | 68 | Both providers, `build_brief` arithmetic, alias regex (including the *wife*/*swift*/*midwife* negatives), cache versioning, zero-sweep → all-`None`, disabled → UNAVAILABLE, no text field anywhere |
| `test_broker.py` | 67 | Atomic/micro-USD accounting, fingerprint mismatch refusal, gas on failures, mode enforcement, schema-v2 load refusal, reconciliation |
| `test_http.py` | 56 | Split timeouts, retry budget, backoff and jitter, `Retry-After` cap, the circuit breaker |
| `test_portfolio.py` | 49 | Mark bases and haircuts, **unmarkable is never cost basis**, `None` propagation, stop epsilon, worst-first ordering |
| `test_journal.py` | 49 | Row identity, lexical-equals-chronological ordering, fsync durability, torn-line scan, `inf`/NaN encoding |
| `test_signals.py` | 48 | Seeded EMA, Wilder smoothing, RSI edge cases, unclamped %B, true-range ATR, closed-bars-only, `None` propagation |
| `test_market.py` | 45 | `_best_pair` rules, ms→s bounds, candle reversal, absent/malformed/zero kept apart, the safety screen's fail-closed behaviour |
| `test_prompts.py` | 40 | **Byte-identity of the system blocks across different evidence**, the fingerprint, `n/a` rendering, no untrusted text |
| `test_report.py` | 38 | The typed report parts, spend-not-applicable, integrity surfacing, sample adequacy |
| `test_quotes.py` | 38 | Decimal derivation, both-legs-to-UI conversion, fingerprinting, `None` on failure (no fallback) |
| `test_brain.py` | 35 | `_validate` discards rather than repairs, NaN rejection, thinking extraction, usage arithmetic, errors are not holds |
| `test_loop.py` | 30 | Cadence, `decision_baseline` vs `previous`, preflight refusal, target diffing and the rebalance band, the four-step execution order |
| `test_strategy.py` | 24 | Baseline hurdle and shrinkage, flat sizing, cash targets zero, advisory cannot exceed bounds, closed-candle history |

**One caveat when running the suite:** `tests/test_prompts.py` contains a live
Anthropic call gated behind `MEMETRADER_LIVE_TESTS=1`. It is skipped by default;
setting that variable bills a real call.

---

## 17. Engineering conventions

**Units, applied everywhere without exception:**

- All timestamps are **epoch seconds as `float`**. Anything arriving in
  milliseconds is converted at the module boundary that receives it.
- All percentages are **whole numbers**: `-4.2` means -4.2%.
- The sole exception is config values named `_pct`, which are documented
  fractions: `max_position_pct = 0.30`, `stop_loss_pct = 0.15`. Each is
  converted to whole percent in exactly one place.
- **Token amounts are atomic integers** and **cash is integer micro-USD**.
  Floats appear at the display edge and nowhere else in the book.
- Every uncertain value is `X | None`, and NaN or infinity is rejected at
  construction by `finite` / `finite_or_none` rather than propagated.

**Style:**

- `ruff`, `line-length = 92` — measured from where the author's hand-wrapping
  naturally falls — with `max-line-length = 100` for E501, so the formatter
  targets 92 but an occasional 97-character line is not an error.
- A large `select` list, `ignore = ["N818", "N812"]`.
- `pyproject.toml` contains a prose block naming **every deliberately-absent
  rule group with its reason**: PL, TRY/EM, S, ANN/D, COM/ISC, ARG/FBT/PT/SLF/
  ERA/TC. The absences are documented decisions, not oversights.
- `per-file-ignores` for BLE001 on `cli.py`, `loop.py` and `sentiment.py` —
  the three places a broad `except` is the correct behaviour. `quotes.py` uses
  an inline `# noqa` instead, because there it is one specific line, not a
  file-wide policy.
- `mypy` **strict** on `src`. Relaxed `operator`/`union-attr` for `tests.*`;
  `ignore_missing_imports` for praw, prawcore and pandas.

**Comments.** The codebase's distinguishing feature is that nearly every
non-obvious decision is recorded *with the measurement that motivated it* —
the 4,900x fake-liquidity pool, the 45x fee double-count, the 422-is-not-rate-
limiting investigation, the eight-signals-to-two result, the 269 trailing-stop
exits. Those comments are the most valuable content in the repository. When
changing this code, read them first; they exist because someone already tried
the obvious thing.

---

## 18. Deliberate omissions

Things this project does **not** have, each on purpose:

| Absent | Why |
| --- | --- |
| **TA-Lib** | Needs a C toolchain on Windows. pandas and numpy do everything needed. |
| **More than two indicator families** | Eight signals → -16.62%. Two → +5.39%. The list is closed. |
| **Trailing stops** | 269 exits at ~-2.26% each in the predecessor project. Noise converted to realized losses. |
| **Max trades per day** | The model is told this limit does not exist, so it must justify each trade on its own merits rather than budget them. |
| **Minimum hold time** | Same reasoning. |
| **A daily spend cap** | Explicitly requested to remain absent. Spend is reported, not enforced. |
| **Async / threads** | Would buy seconds per 15-minute tick and cost every invariant in the codebase. |
| **A real-money path** | No wallet, no keys, no signer. `ExecutionMode.LIVE` raises rather than stubs. The `Broker` protocol is where one *would* attach. |
| **A retry loop around the model call** | A failed tick is a visible failed tick. Retrying hides an outage and risks duplicate decisions. |
| **A separate rejections file** | Every bound is already in the ledger. A second file would duplicate facts on disk and add a way for the two to disagree. |
| **An LLM on the decision path by default** | Audit C6. No measured predictive value, no counterfactual against a deterministic rule, and a default *is* authority. Available as `kind = "advisory"`. |
| **A degraded/fallback quote** | Audit C4. A route we could not obtain is not a worse route; manufacturing one turns a vendor outage into a trade. |
| **A confluence or agreement score** | The indicators are transforms of one close series. Such a score is a correlation artefact wearing a confidence interval. |
| **Sentiment on by default** | Audit §7. No ablation has shown it contributes anything a price series does not. Off, and reported as UNAVAILABLE rather than zero. |
| **Raw social text in any prompt** | Audit C7. The fix is deletion, not escaping — there is no escaping scheme that separates data from instructions inside one natural-language context. |
| **Sharpe, Sortino, Calmar, max drawdown** | They need an equity curve at fixed intervals and the record stores fills. §6. |
| **Approvals and clamps in the risk layer** | Replaced by `RiskBounds`. An approval can be acted on; a ceiling can only be respected. |
| **Conviction-scaled position sizing** | Requires a calibration nobody has measured. Flat sizing until `Forecast.calibration_id` exists. |

---

## 19. Known issues and next steps

1. **Nothing has been measured since the rewrite.** There is no run of the
   `baseline` strategy long enough to say anything about it, and the only
   numbers in §14 describe a system that no longer exists. The first thing this
   project needs is a `cash`-versus-`baseline` comparison over a window long
   enough to have a standard error.
2. **No equity curve is persisted**, so Sharpe, Sortino, Calmar and max drawdown
   remain uncomputable (§6). Recording a periodic mark is the smallest change
   that unlocks all four.
3. **`Forecast.calibration_id` is always `None`.** Until a calibration exists,
   sizing is flat and conviction is decoration. Measuring it is the prerequisite
   for anything more sophisticated than in-or-out.
4. **The execution failure model is an assumption**, not a measurement: an iid
   6% coin flip and a constant gas figure, where audit §15 asks for failures
   conditioned on congestion, quote age, route and program error.
5. **Five token-safety checks are deferred** for want of an on-chain RPC — mint
   and freeze authority, LP ownership and lock, holder and dev concentration,
   Token-2022 extensions, a sellability probe. They are named in every
   `SafetyVerdict`, which makes the gap visible but does not close it.
6. **The prompt cache TTL question is open** (§14), for the advisory path only:
   the 1-hour TTL, a sub-300-second cadence, or no breakpoints at all.
7. **`mention_zscore_7d` needs 48 hours of warm-up** after any `CACHE_VERSION`
   bump — relevant whenever sentiment is turned on for an experiment.
8. **Arctic Shift cannot complete a 24-hour comment walk** of a busy subreddit
   within its ~3-second server-side query timeout. The module reports the
   shortfall honestly; closing the gap needs Reddit API credentials.
9. **DexScreener omits `m5` on many pairs**, including BONK's best pool. Handled
   correctly as `None`, but it means the shortest-horizon price change is often
   simply unavailable.

---

## 20. Glossary

| Term | Meaning here |
| --- | --- |
| **Fast tick** | The 60-second cycle: mark the book, enforce stops, no strategy call. |
| **Slow tick** | The 900-second cycle: evidence → strategy → targets → bounds → quote → execution. |
| **Evidence bundle** | Per-coin `CoinSnapshot` + `Technicals` + `SentimentBrief` + unavailability reason. |
| **Execution mode** | `read_only` (decide, mutate nothing), `paper` (trade the simulated book), `live` (refused). A capability the broker enforces. |
| **Target position** | A dollar target for one symbol. The whole of what a strategy may say. |
| **Rebalance band** | The drift below which a target difference is left alone, because the round trip costs more than the gap. |
| **Bounds** | `RiskBounds`: a maximum notional plus named vetoes. A ceiling, not permission. |
| **Intent** | An `OrderIntent` — the idempotency key, journaled before the order is placed. An intent with no terminal state stops the next run. |
| **Quote fingerprint** | A hash over a quote's side, mints, atomic amounts and slot. The broker recomputes it and refuses on mismatch. |
| **Unmarkable** | A held position no basis could price. A risk incident, not a number, and never carried at cost. |
| **`decision_baseline`** | The snapshot the last decision was made on. What liquidity trend is measured against. |
| **`previous`** | The freshest snapshot of any kind. Overwritten every 60 seconds. |
| **Deferred check** | A safety check this system does not perform, named in every verdict so its absence is in the record. |
| **Attention** | Volume and velocity of mentions — explicitly *not* sentiment polarity. |
| **Shill-farm signature** | A contributor-to-post ratio below 0.5: few accounts posting a lot. |
| **Missing is never zero** | `None` = "could not find out." `0` = "looked, it is quiet." Never conflated. |
| **The seam** | The `Broker` protocol in `types.py`, where a real venue would attach. |

---

*Paper trading only. No real funds have ever been at risk in this project.*
