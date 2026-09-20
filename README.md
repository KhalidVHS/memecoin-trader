# memetrader

Claude Opus 5 actively trading three Solana memecoins against a $1,000 paper
book. No real money, no wallet, no private keys — but real prices and real fill
mechanics, so the resulting P&L means something.

## Setup

```bash
uv sync
cp .env.example .env          # optional — see below
uv run memetrader check       # config, mints and credentials — no model call
uv run memetrader status      # live prices + technicals + sentiment, no model call
uv run memetrader once --dry-run
uv run memetrader run
```

`ANTHROPIC_API_KEY` is read from your environment. Nothing else is required —
Reddit and Jupiter credentials are optional and the bot degrades cleanly without
them.

## Commands

| Command | What it does |
|---|---|
| `memetrader check` | Validates `config.toml`, resolves every configured mint to its live DexScreener pool, and reports whether the Anthropic, Reddit and Jupiter credentials are set. Run this first — a mint that is wrong, or whose pool has since migrated, fails here in a second rather than as a strange fill three hours into a run. |
| `memetrader status` | All three evidence streams for every coin. No model call, no cost. |
| `memetrader once --dry-run` | One full decision cycle printed end to end — evidence, the model's reasoning, proposed actions, risk verdicts — **without mutating state**. The main development loop. |
| `memetrader once` | The same, applied. |
| `memetrader run` | The dual-cadence loop. Ctrl+C is safe. |
| `memetrader report` | P&L, trade history, decision history, spend to date. |
| `memetrader reset` | Wipe `data/` back to the starting cash. |

## How it works

Two cadences. A **fast tick every 60 seconds** refreshes prices, marks the book
and enforces stop-losses — cheap, no model call, because a −15% stop that only
checks every 15 minutes is not a stop. A **slow tick every 15 minutes** builds
the full evidence bundle, calls the model once, runs every proposed action
through the risk layer, and executes what survives.

The model's decision rests on three independent evidence streams:

- **Price and on-chain flow** — DexScreener. Buy/sell ratios and liquidity
  trend are actual money moving, which makes this the most trustworthy stream. A
  draining pool is the single most important thing that can happen to a memecoin
  position, and price alone will not tell you in time.
- **Technicals** — hand-computed in pandas on 5m and 1h candles from
  GeckoTerminal. The 5m/1h agreement or disagreement is the point.
- **Social sentiment** — Reddit, measured as *attention* rather than polarity.
  Velocity, breadth and novelty survive scrutiny; polarity is manufactured by
  shill farms and is flagged low-trust in the prompt.

Each stream has its own cache and its own failure mode, and the prompt labels
which stream every number came from — so the model can weigh them, and so you
can tell from the decision log which stream drove a bad trade.

**Code decides, the model proposes.** Every action passes through `risk.py`
before it can reach the broker. Rejections are logged with the rule that fired
and fed back to the model on the next tick, so it learns the boundaries rather
than re-proposing illegal trades.

### Fill realism

Fills are priced off the **Jupiter Quote API** — the actual routed output amount
for that exact trade size against the real AMM curve, no wallet and no signing.
This is better execution realism than any broker demo, because broker demos fill
at the quoted mid and quietly hide the slippage that actually kills memecoin
strategies.

On top of the routed price the paper broker charges the pool fee for the venue
the route actually used, gas on every attempt, and a **6% failed-transaction
rate** — a failed Solana swap still pays full gas, and charging it is the thing
naive simulators get wrong.

## Configuration

`config.toml` is the only file you need to edit: the three coins, the budget,
the risk limits, the cadence, the model and its effort level. Every number in
the fill and cost model is a config value, not a constant.

### Risk limits

| Rule | Default |
|---|---|
| Max position size | 30% of book |
| Stop-loss | −15% from entry, enforced by code |
| Min trade size | $10 |
| Max price impact | 3% |
| Max snapshot age | 90s — refuses to trade on stale data |
| Min pool liquidity | $50,000 |
| Max trades per day | **none, deliberately** |
| Minimum hold time | **none, deliberately** |

## Cost

At a 15-minute cadence, `claude-opus-5` with `effort = "high"` is roughly
**$180–250/month** run continuously, most of it thinking tokens. The shipped
default is `xhigh`, which buys more reasoning across three disagreeing evidence
streams and costs correspondingly more — effort is the largest single lever on
the bill, precisely because it is the thinking tokens that move. Every decision
row records input, output and cache-read tokens, and `memetrader report` sums
spend to date — you will know what it costs from the first day rather than from
the invoice.

Running in bursts is the cheapest way to keep this in the tens of dollars, and
costs nothing in code: the loop resumes cleanly from `state.json`.

`model` and `effort` are config values and every decision row records which
produced it, so comparing `high` against `xhigh` in the decision log is a
one-line change. The accepted levels are `low`, `medium`, `high`, `xhigh` and
`max` — exactly what the API accepts, so nothing valid is rejected as a typo.

## Optional credentials

**Reddit** (free, ~2 minutes) is what makes the sentiment brief *current*, and
it is the one credential worth bothering with. Create a "script" app at
[reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) — the redirect URI
can be `http://localhost:8080`, it is never used — and set three variables in
`.env`:

```
REDDIT_CLIENT_ID=<the string under the app name>
REDDIT_CLIENT_SECRET=<the "secret" field>
REDDIT_USER_AGENT=memetrader/0.1 by u/<your-username>
```

Reddit rejects generic user agents, so the third is not optional padding.

Without credentials the bot still runs: sentiment falls back to the keyless
Arctic Shift mirror plus the on-chain flow numbers. **The difference is
freshness.** Arctic Shift is an archive, and its index measurably trails live
Reddit — ~10 hours behind on a check on 2026-09-18. The code will not paper over
that: when a source's index has not reached the current hour, `mention_velocity_1h`
is reported as `n/a` rather than `0.0`, and the prompt says so explicitly, because
"nobody is talking about this" and "we have not looked yet" are opposite trades.
Run `memetrader status` and look for the `vel1h n/a` line to see which one you are on.

With credentials, the sweep reads each subreddit's `/new` listing directly — live,
with no search index in between — so the current hour is always observed.

**Jupiter** (free) moves quotes from `lite-api.jup.ag`, whose rate limit
deliberately decays, to `api.jup.ag`. Set `JUPITER_API_KEY` in `.env`.

## Data

`data/` is gitignored and holds the whole ledger:

- `state.json` — the book, written atomically after every mutation
- `trades.jsonl` — append-only fill log
- `decisions.jsonl` — append-only decision log, including rejected proposals and
  the token spend for each call

## Out of scope

Real money, wallets, private keys, and any live-execution path. These are
deliberate omissions. The `Broker` protocol in `types.py` is the seam where a
real venue would attach if that ever changes — and the LLM layer would never
learn the difference.
