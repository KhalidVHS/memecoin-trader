# memetrader

A deterministic strategy — optionally advised by Claude — trading three Solana
memecoins against a $1,000 **paper** book. No real money, no wallet, no private
keys, and no live-execution path anywhere in the tree. Real prices and real fill
mechanics, so the resulting P&L means something.

## Setup

```bash
uv sync
cp .env.example .env               # optional — see below
uv run memetrader check            # config, mints, sources, token screen
uv run memetrader status           # live evidence for every coin, no strategy call
uv run memetrader once --mode read_only
uv run memetrader run
```

`ANTHROPIC_API_KEY` is only needed if you switch `[strategy] kind` to
`"advisory"`. The default strategy is deterministic and makes no model call, so
a fresh clone runs with no credentials at all.

## Commands

| Command | What it does |
|---|---|
| `memetrader check` | Validates `config.toml`, resolves every configured mint to its live DexScreener pool, confirms each data source answers, and runs the safety screen over every token. Run this first — a mint that is wrong, or whose pool has since migrated, fails here in a second rather than as a strange fill three hours into a run. |
| `memetrader status` | Every evidence stream for every coin, plus the book. No strategy call, no cost. |
| `memetrader once --mode read_only` | One full decision cycle printed end to end — evidence, targets, risk bounds, quotes, what *would* execute — while mutating nothing and writing nothing. The main development loop. |
| `memetrader once` | The same cycle, applied to the paper book. |
| `memetrader run` | The dual-cadence loop. `--max-ticks` stops after N decision cycles. Ctrl+C is safe — state is saved after every change. |
| `memetrader report` | P&L, trade history, decision history, ledger integrity and spend to date. `--no-live-marks` makes it fully offline. |
| `memetrader reset` | Wipe `data/` back to the starting cash. Irreversible. |

### Execution mode

`--dry-run` no longer exists. It was a boolean, and a boolean cannot express the
difference between *do not trade* and *do not write anything*. In its place is a
three-valued capability, `--mode`, on both `once` and `run`:

- `read_only` — decide and print everything; mutate nothing, write nothing.
- `paper` — trade the simulated book. The default.
- `live` — refused. There is no wallet, and the code says so rather than
  pretending the path exists.

The default comes from `execution_mode` in `config.toml`; `--mode` overrides it
for one invocation. The mode is a capability the *broker* enforces, not a flag
each call site is trusted to check.

## How it works

Two cadences. A **fast tick every 60 seconds** refreshes prices, marks the book
and enforces stop-losses — cheap, no strategy call, because a −15% stop that
only checks every 15 minutes is not a stop. A **slow tick every 15 minutes**
builds the evidence bundle, asks the strategy for target positions, and
reconciles the book to those targets.

Evidence comes from two streams that are on by default:

- **Price and on-chain flow** — DexScreener. Buy/sell ratios and liquidity trend
  are actual money moving, which makes this the most trustworthy stream. A
  draining pool is the single most important thing that can happen to a memecoin
  position, and price alone will not tell you in time.
- **Technicals** — hand-computed in pandas on closed 5m and 1h candles from
  GeckoTerminal. They are a labelled benchmark feature set, not independent
  evidence: they are all transforms of one close series, so their "agreement" is
  one price path described several times. There is deliberately no confluence
  score.

**Reddit sentiment is off by default** (`[sentiment] enabled = false`). Nothing
has measured whether it improves out-of-sample results after its latency and
cost, so it is an experiment source rather than a production input. When
disabled it reports as `UNAVAILABLE` — never as zero. It also carries no post
text at all: raw social text used to be interpolated into a prompt that had
order authority, which is prompt injection in a capital-allocation loop, so the
collection path was deleted rather than escaped. What survives is counts and
rates.

### The strategy proposes targets; the code decides

A strategy emits **target positions** — a dollar target per symbol — and nothing
else. `loop.py` diffs those targets against the inventory it actually holds and
trades the gap, ignoring drifts smaller than `rebalance_band_usd` so the book
does not churn for less than the fees. A strategy cannot place an order, size an
order, or name a venue.

Three kinds, set by `[strategy] kind`:

- `cash` — targets zero everywhere. The null hypothesis, and the control arm any
  claim of alpha has to beat.
- `baseline` — deterministic shrunk momentum with an explicit round-trip cost
  hurdle and an uncertainty band. Entries require the *lower* quantile of
  expected net return to clear the hurdle, not the mean. **The default.**
- `advisory` — the baseline plus a model that may modify targets. Opt-in, one
  visible line in `config.toml`.

The default is not the model, on purpose. A default is authority — it is what
runs when nobody chose — and the model's edge over a deterministic rule on the
same evidence has never been measured. Making it a named strategy kind is also
what makes an honest A/B against `baseline` possible.

Every target then passes through `risk.py`, which returns **bounds** — a maximum
notional, plus vetoes with the rule name that fired. It does not approve, and it
does not silently shrink an order that has already been quoted; a shrunk order
is re-quoted and the quote's fingerprint re-checked before it can fill.

### Fill realism

Fills are priced off the **Jupiter Quote API** — the actual routed output amount
for that exact trade size against the real AMM curve, no wallet and no signing.
Jupiter's integer `inAmount`/`outAmount` are carried through verbatim: token
amounts are atomic integers and cash is integer micro-USD, so nothing anywhere
back-derives a quantity from dollars.

A quote we could not obtain is not a worse quote, it is no quote. There is no
synthetic fallback fill — a Jupiter outage means no trade, not a trade priced
off a mid-price fiction.

On top of the routed price the paper broker charges gas on every attempt,
including failed ones, and a **6% failed-transaction rate**. The pool fee is
zero and that is not an oversight: Jupiter's `outAmount` already has every hop's
AMM fee, slippage and price impact deducted inside the route, so charging a pool
fee again bills you twice. Observed round trip on BONK was 4.4 bp where a
per-hop fee model implied 200 bp — 45x.

## Core invariants

- **Missing is never zero.** Every uncertain field is `X | None`. `None` means
  we could not find out; `0` means we looked and it is quiet. They are opposite
  trades, so they are never the same value.
- **Atomic integers and micro-USD.** Token amounts are integer atomic units;
  cash is integer micro-USD. Floats appear only at the display edge.
- **Quotes are bound.** Every quote is fingerprinted over its side, mints,
  atomic amounts and slot, and the broker recomputes the fingerprint from the
  quote it was handed and refuses on mismatch.
- **Risk returns bounds, not approvals.** `RiskBounds` is a quantity ceiling
  plus vetoes, never a mutable permission that later code can widen.
- **The safety screen fails closed, and keeps `unknowns` separate from
  `vetoes`.** A check that could not be run is not a check that passed.
- **A book whose value is unknown reports `None`.** An unmarkable position is a
  risk incident, not a number, and it is never carried at cost basis.

## Configuration

`config.toml` is the only file you need to edit: the execution mode, the three
coins, the strategy kind, the budget, the four risk layers, the cadence, the
model and its effort level, the HTTP budgets and the safety screen. Every number
in the fill and cost model is a config value, not a constant.

### Selected risk limits

| Rule | Default |
|---|---|
| Max position size | 30% of book value, per symbol |
| Stop-loss | −15% from entry, enforced by code |
| Max gross / net exposure | 60% |
| Min cash floor | 10% of book |
| Max sleeve exposure (BONK+WIF+POPCAT) | 60% |
| Min trade size | $10 |
| Default / max entry | $25 / $100 |
| Max price impact | 3% |
| Max depth participation | 1% of pool depth |
| Max snapshot age | 90s |
| Max quote age | 10s — older than that is not a price |
| Min pool liquidity | $50,000 |
| Min pool age | 24h, and an unknown age is a veto, not a pass |
| Post-stop quarantine | 1h — no re-entry into what just stopped out |
| Max daily / 7-day loss | 5% / 10% |
| Max drawdown | 20% from peak — halts the run |
| Max consecutive failed swaps | 3 — halt, not retry |
| Max trades per day | **none, deliberately** |
| Minimum hold time | **none, deliberately** |

## Cost

Under the default `baseline` strategy there is no model call, so there is no
model spend. `memetrader report` says **"Model spend not applicable"** rather
than printing a $0.00 that looks like a measurement.

With `kind = "advisory"`, spend depends mostly on `effort`, which is the largest
single lever on the bill because it is the thinking tokens that move. The
accepted levels are `low`, `medium`, `high`, `xhigh` and `max` — exactly what
the API accepts, so nothing valid is rejected as a typo. Every decision row
records the model, the effort and the input/output/cache tokens that produced
it, so comparing two settings in the ledger is a one-line config change.

## Optional credentials

**Reddit** (free, ~2 minutes) only matters if you turn `[sentiment] enabled` on.
Create a "script" app at
[reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) — the redirect URI
can be `http://localhost:8080`, it is never used — and set three variables in
`.env`:

```
REDDIT_CLIENT_ID=<the string under the app name>
REDDIT_CLIENT_SECRET=<the "secret" field>
REDDIT_USER_AGENT=memetrader/0.1 by u/<your-username>
```

Reddit rejects generic user agents, so the third is not optional padding.
Without credentials the sweep falls back to the keyless Arctic Shift mirror,
which is an archive and measurably trails live Reddit. The code will not paper
over that: when a source's index has not reached the current hour,
`mention_velocity_1h` is `n/a` rather than `0.0`.

**Jupiter** (free) moves quotes from `lite-api.jup.ag`, whose rate limit
deliberately decays, to `api.jup.ag`. Set `JUPITER_API_KEY` in `.env`.

**Anthropic** is required only for `kind = "advisory"`. `ANTHROPIC_API_KEY` is
read from the environment; no key is stored in this repository.

## Data

`data/` is gitignored and holds the whole record:

- `ledger.jsonl` — the joined stream: decisions, order intents, state
  transitions and fills, every row carrying IDs that sort chronologically by
  lexical sort. Owned by `journal.Ledger`.
- `state.json` — the book, written atomically after every mutation.
- `trades.jsonl` — append-only fill log.
- `intents.jsonl` — the broker's pre-submission intent log, written before any
  side effect.
- `decisions.jsonl` — the legacy decision log. Nothing writes it any more;
  `report` still reads it so a report over an older run still works.
- `risk_ledger.json` — persisted risk state (drawdown peak, loss windows,
  quarantines, consecutive failures). If this file exists and cannot be read,
  the run starts **halted** rather than starting from an empty risk history.

### Startup can refuse

Before anything else runs, the process checks the ledger for order intents with
no terminal state. If it finds any, it refuses to start, names the intents, and
points at `data/ledger.jsonl` and `memetrader reset`. An unresolved intent means
a previous process died mid-order, and the honest response to an unknown
execution state is to stop rather than to trade on top of it.

### The state schema changed

`state.json` is schema **v2** and cannot load a pre-audit v1 `data/` directory.
Move the old one aside:

```bash
mv data data-v1
```

`data-v*/` is gitignored for exactly this.

## Out of scope

Real money, wallets, private keys, and any live-execution path. These are
deliberate omissions, and `--mode live` refuses rather than stubs. The `Broker`
protocol in `types.py` is the seam where a real venue would attach if that ever
changed.
