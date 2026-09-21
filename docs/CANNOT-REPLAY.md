# What this dataset cannot tell you

Every number here was measured, not estimated. Where a limit was discovered by
hitting it, the vendor's own words are quoted.

This document lives in `docs/` rather than beside the data in `history/`
because `history/` is gitignored — the data is regenerable output, but the
account of its limits is analysis and has to survive in the repository. The
per-series provenance (`history/manifest.json`) is written by the backfill and
travels with the bars it describes.

**Read this before trusting a backtest result.** Three of the limits below are
large enough to change a conclusion.

---

## 1. The hard ceiling: 180 days, and it moves

The plan assumed free-tier `before_timestamp` paging was unbounded, on the
evidence of a probe that walked BONK back to 2026-02-24. That probe stopped
exactly at the boundary and the boundary went unnoticed. Paging deeper returns:

```
HTTP 401  "You can only access data from the past 180 days with Public API."
```

Measured depth sweep (WIF 1h, 6s spacing, 2026-09-21):

| `before_timestamp` | Result |
|---|---|
| now − 0d | 200, 1000 rows |
| now − 30d | 200, 1000 rows |
| now − 90d | 200, 1000 rows |
| now − 180d | 200, 1000 rows |
| now − 270d | **401** |
| now − 365d | **401** |
| now − 500d | **401** |

**Consequences, in order of how much they matter:**

1. **"1h as far back as the free tier allows" and "5m for six months" are the
   same window.** The scope decision that separated them no longer separates
   anything. Both timeframes cover roughly the same ~7 months.
2. **No multi-year regime can be tested.** There is no 2024 memecoin mania and
   no 2025 drawdown in this dataset. Every coin is observed over one market
   period, so any parameter fitted here is fitted to that period.
3. **The window slides forward daily.** A re-run next month reaches a different
   start date, so a result is only reproducible against a pinned copy of the
   data, not against a re-download. `manifest.json` records `fetched_at` and
   `horizon_reached` per series for exactly this reason.

The realised depth is slightly better than the headline, because the cap
applies to the *requested* timestamp and not to the rows returned: a request at
the 180-day mark still yields the 1000 bars ending there. Actual 1h coverage is
**2026-02-24 → present, ~209 days, 4,995 bars** for a liquid coin, and the
deepest bar in the whole 1h set is 2026-02-03.

A paid key removes this limit. Nothing else in this document changes if you buy
one.

## 2. What was actually downloaded

24 coins, pool-keyed, `universe/solana_memecoins.toml`.

| | Series | Rows | Missing | Window | Hit the 180-day horizon |
|---|---:|---:|---:|---|---:|
| **1h** | 24 | 114,258 | 6,101 (5.1%) | 2026-02-03 → 2026-09-21 | 24 / 24 |
| **5m** | 24 | 687,699 | 552,092 (44.5%) | 2026-03-09 → 2026-09-21 | 22 / 24 |

**801,957 rows, 50 MB gzipped. 48 series written, 0 failed.**

Structurally verified (1h): strictly increasing timestamps, no duplicates,
every gap an exact multiple of the interval, `low ≤ open/close ≤ high`, all
positive, and every bar survives `types.Candle` construction.

**The two series that did not reach the horizon are the ones to watch.** They
stopped early because the vendor has no deeper 5m history for that pool, not
because of the 180-day cap — so their windows are much shorter than the rest:

| Coin | 5m starts | Coverage |
|---|---|---|
| ACT | 2026-08-01 | **51 days** |
| LOCKIN | 2026-05-11 | 133 days |
| everything else | ~2026-03-18 | ~187 days |

ACT has a full 1h series but only seven weeks of 5m. Any 5m-based result that
includes it is drawn from a quarter of the sample the other coins provide.

## 3. Missing bars are real information, and they are not evenly spread

The vendor **omits** bars in which nothing traded rather than emitting a
zero-volume bar. Confirmed: zero zero-volume bars across the sample. So a
missing bar means *no trades printed* — which is knowable, and different from
*the fetch failed*. Nothing is forward-filled or zero-filled; gaps are recorded
in the manifest.

Missing 1h bars over the ~209-day window, out of ~4,995. Note that a coin with
*fewer rows* is not better covered — SLERF's 3,693 rows plus 1,568 gaps still
only spans the same window:

| Coin | Rows | Missing | Missing % |
|---|---:|---:|---:|
| BONK, GIGA | 4,995 | 0 | 0.0% |
| POPCAT, PNUT, GOAT, CHILLGUY | 4,995 | 1 | 0.02% |
| BOME, MOODENG | 4,995 | 2 | 0.04% |
| MEW, FWOG | ~4,993 | 4 | 0.08% |
| AURA | 4,993 | 7 | 0.14% |
| ACT | 4,992 | 11 | 0.22% |
| PONKE | 4,993 | 13 | 0.26% |
| WIF | 4,991 | 19 | 0.38% |
| GME | 4,990 | 37 | 0.74% |
| RETARDIO | 4,991 | 49 | 0.98% |
| DADDY | 4,992 | 77 | 1.5% |
| LOCKIN | 4,991 | 149 | 3.0% |
| BILLY | 4,991 | 518 | 9.4% |
| SC | 3,993 | 540 | 11.9% |
| MICHI | 3,994 | 668 | 14.3% |
| MOTHER | 3,997 | 1,016 | 20.3% |
| BODEN | 3,710 | 1,413 | 27.6% |
| SLERF | 3,693 | 1,568 | 29.8% |

### At 5m the same split is far more severe

Aggregate 5m missing is **44.5%**, but that single number is meaningless
because the spread is enormous — memecoins simply do not trade every five
minutes, and how often they do is exactly what separates a live coin from a
dead one:

| Coin | Rows | Missing |
|---|---:|---:|
| BONK | 52,916 | **0.5%** |
| GIGA | 47,915 | 10.9% |
| CHILLGUY, BOME | ~42,000 | ~21% |
| MOODENG, GOAT, PNUT, AURA, MEW, POPCAT | ~37,000 | 28–32% |
| WIF | 34,944 | 35.5% |
| FWOG, PONKE, RETARDIO, GME | 28–33,000 | 39–48% |
| DADDY, LOCKIN | 16–25,000 | 53–58% |
| BILLY, SC | ~13,000 | 74–75% |
| MICHI, BODEN, MOTHER | ~10,500 | 81% |
| SLERF | 7,276 | **86.5%** |

A coin missing 86.5% of its 5m bars has, on average, one print every 37
minutes. Calling the resulting series "5-minute data" is a category error: for
SLERF, BODEN, MOTHER and MICHI the 5m file is a trade log with timestamps
rounded to 5 minutes, and `realized_vol_pct` over it is not a 5-minute
volatility. **Use 1h for the thin tail, or exclude it.**

**This is the survivorship-bias correction working as intended.** The thin tail
is thin *because those coins died*, and a universe scraped from today's
leaderboard would not contain them. They are the failures a backtest must see.

**But it breaks a feature.** `realized_vol_pct` is the sample sigma of log
returns over the last 21 closes. Across a gap, one "bar" of log return spans
several hours of real time, which inflates the number. At 0.02% missing this is
noise; at 29.8% it is not a volatility estimate at all. Since
`half_width = interval_vol_multiple × realized_vol_pct` is subtracted from the
forecast, an inflated vol makes the strategy *refuse* trades — so the bias for
thin coins is toward false negatives, not false positives.

**Recommendation:** exclude a coin/period where missing exceeds a stated
threshold, and state the threshold in the result. Do not silently include
SLERF, BODEN, MOTHER, MICHI, SC or BILLY in an aggregate without saying so.

SLERF deserves a specific note: `universe/solana_memecoins.toml` records
`liquidity_usd = $19,000,029` against `fdv_usd = $10,932,060`, a ratio of 1.738.
That was verified as a genuine burned-LP position across 16 pools, not an
impostor token — but 29.8% missing hourly bars over seven months shows that
deep locked liquidity and an active market are different things. Liquidity is
not tradability.

## 4. Fidelity: the backfilled bars are the bars the live system traded on

This is the one strong positive claim in this document.

`signals._realized_vol_pct` is a pure function of the last 21 closes. The 90
documented live ticks embedded the complete evidence bundle per coin, including
the `realized_vol_pct` the strategy actually read. Recomputing it from
backfilled bars at each recorded `receive_time`, with an independent
reimplementation (`tools/verify_backfill.py`, deliberately not importing the
original so a change to either side shows up):

| | 1h | 5m |
|---|---|---|
| Comparisons | 79 | 117 |
| Median absolute error | 1.3 × 10⁻¹⁵ pp | — |
| **Maximum** absolute error | **2.1 × 10⁻¹⁴ pp** | **2.6 × 10⁻¹⁴ pp** |
| Within 10⁻⁶ pp tolerance | 79 / 79 | 117 / 117 |

That is floating-point noise. Agreement across 21 closes at 10⁻¹⁴ cannot occur
by coincidence: **the downloaded series are the series the live system was
served**, on both timeframes.

Only ~79 of 270 records could be compared — the rest had no recorded
`realized_vol_pct` because the live fetch had failed. That is the POPCAT 429
defect (below), showing up honestly as absent data rather than as agreement.

### Cross-timeframe: the two pulls corroborate each other exactly

The 1h and 5m series were downloaded independently, in separate runs, paging
backwards through different numbers of pages. Resampling 5m to hourly (an
hourly close is the close of the last 5m bar inside the hour) and comparing
against the fetched 1h close:

| Pool | Hours compared | Exact matches | Max deviation |
|---|---:|---:|---:|
| BONK `5zpyut…` | 4,428 | 4,428 | 0.0% |
| WIF `EP2ib…` | 4,498 | 4,498 | 0.0% |
| POPCAT `FRhB8…` | 4,513 | 4,513 | 0.0% |

**13,439 hourly closes, zero deviations.** Two independent downloads agreeing
bit-for-bit is the strongest available evidence that neither pull is
misaligned, mis-keyed or silently revised.

## 5. What has no historical record at all

Every row is a live input that cannot be replayed. The modelling choice is the
backtest engine's decision; the bias column is what it costs.

| Input | Historical record? | Modelling choice | Bias introduced |
|---|---|---|---|
| Jupiter route quote | **None** — no timestamp/slot param, 10s TTL | Synthesize from bar close + `slippage_bps_fallback = 50` | Ignores real route depth; understates cost in stress |
| Price impact at size | **None** | Constant, or scaled from `reserve_in_usd` | Optimistic at size |
| `price_change.h1` | **Not retrievable** — rolling window computed vendor-side at request time | Reconstruct from bars | Measured below; comparable to the hurdle |
| Liquidity / txns / FDV | **None** — snapshot only | Pin at today's value, or disable those screens | Survivorship: today's liquidity applied to the past |
| `snapshot.quality` freshness | N/A offline | Always `OK` | **Removes the degradation that made POPCAT untradeable live** |
| Pool age screen | Derivable from `pool_created_at` | Enforce honestly per bar | — |
| `failed_tx_rate` (6% live) | **None** | Seeded `random.Random`, reproducible | Assumes iid; real failures cluster |
| Sentiment | Not collected (`[sentiment] enabled = false`) | Disabled | — |

**Do not add pool fees on top of modelled slippage.** Jupiter's `outAmount` is
already net of every hop's fee (`quotes.py:55-63`), `[risk]
assumed_pool_fee_pct = 0.0`, and `broker.pool_fee_micro()` is structurally 0.
Double-counting is the easiest way to build a backtest that rejects every trade.

## 6. The h1 reconstruction error is the size of the decision threshold

`BaselineStrategy` reads `snapshot.price_change.h1` and the whole decision is:

```
gross      = price_change.h1 × 0.6          # shrinkage
net        = gross − 0.8                    # ROUND_TRIP_COST_PCT, strategy.py:236
half_width = 0.5 × realized_vol_pct         # interval_vol_multiple
eligible if (net − half_width) ≥ 0.2        # entry_hurdle_pct
```

DexScreener's `h1` is a *rolling trailing* window; an hourly bar is *aligned*.
Reconstructing from bars therefore cannot be exact. Measured against the 116
live values (corrupt snapshot excluded, see §7):

| Reconstruction | n | Median | p90 | Max |
|---|---:|---:|---:|---:|
| From **5m** bars (`close[t]/close[t−12]`) | 115 | **0.349 pp** | 0.930 pp | 1.575 pp |
| From **1h** bars (`close[t]/close[t−1]`) | 115 | 0.815 pp | 2.338 pp | 3.982 pp |

The maxima are the load-bearing numbers here. With the corrupt snapshot
included they were 73.4 pp and 73.5 pp; excluding that single record drops them
to 1.6 pp and 4.0 pp. So the reconstruction has **no other large errors at all**
— the error is tightly bounded and the one extreme value was the vendor's
mistake, not the method's.

Pre-measured per coin against the live candle arrays, before this pull. The
pooled 5m figure reproduces the per-coin pre-measurement **exactly** (0.349 pp),
which is a second, independent confirmation that the downloaded bars are the
live bars:

| | BONK | WIF | POPCAT |
|---|---|---|---|
| From 5m bars | 0.349 pp | 0.338 pp | 0.442 pp |
| From 1h bars | 1.042 pp | 0.897 pp | 2.681 pp |

**Two consequences, and they shape how results must be read:**

1. **5m reconstruction is ~2.3× more accurate than 1h** (0.349 pp vs 0.815 pp
   median, and 1.6 pp vs 4.0 pp worst case), because it approximates a rolling
   window with a rolling window. This is why the 5m pull is necessary rather
   than merely nice to have.
2. **The residual error is the same size as the threshold.** The measured
   0.349 pp median error in `h1` becomes **0.209 pp** in `gross` after ×0.6
   shrinkage — larger than the entire `entry_hurdle_pct` of 0.2. For a strategy
   that trades on a marginal threshold
   crossing, the backtest will disagree with live at exactly the margin where it
   matters.

**Therefore: read results as distributional, not trade-for-trade.** Any
conclusion that turns on a hurdle within ~0.2 pp of a boundary is noise. Report
sensitivity across a range of hurdle values, never a single point estimate.

## 7. One live snapshot was corrupt, and the system had no defence

Tick 31 of run `20260920T224616Z` recorded WIF at `price_usd = 0.05636` with
`price_change.h1 = −72.51` and `price_change.m5 = −72.56`, flagged
`quality: "ok"`. Neighbouring ticks show ~0.2047 before and after.

The backfilled bars settle it. In that hour WIF traded:

```
09-21 01:00  o=0.20577  h=0.20792  l=0.20141  c=0.20237
```

**The low was 0.20141. A price of 0.05636 never existed.** The vendor served a
corrupt snapshot for one tick and marked it `ok`.

Two things follow:

- **The 73.5 pp outlier is not reconstruction error.** It is the backfill being
  right and the live snapshot being wrong. It must be excluded from the §6
  measurement rather than clipped into it — including it would slander the
  reconstruction for the vendor's mistake. Prevalence: **1 of 270 records
  (0.37%)**, isolated, detected by `|h1| > 40 ∧ |m5| > 40`.
- **This is a live defect, not a data defect.** Nothing in the live path
  sanity-checks a snapshot price against recent bars, and `quality` was `ok`.
  A −72% move with `shrinkage × 0.6` would not have triggered a buy, so this
  instance was harmless; an equally corrupt *positive* print would not have
  been. Out of scope here, but it should be fixed.

## 8. Known defects this dataset does not fix

- **The POPCAT 429 bug (live).** 38 of 39 hourly-OHLCV fetches returned HTTP
  429 during the documented run. Each slow tick fires 6 GeckoTerminal calls in
  a burst and POPCAT is last in line, so it absorbed every rejection: the
  "3-coin" run was really a 2-coin run. Visible here as 79 of 270 usable
  fidelity records. `market.py` does not route through `http.execute`, so it
  gets no retry or backoff — `backfill.py` deliberately does.
- **48 pre-existing mypy errors, all in `tests/`.** `src/` is clean. Verified
  as a pre-existing baseline, unrelated to this work.
- **The live BONK pool is not the deepest one.** Config trades a $293k pool;
  `universe/solana_memecoins.toml` resolved the same pool for consistency with
  the live record, but deeper pools exist. Affects live price impact, not this
  dataset.

## 9. Reproducing this

```
uv run memetrader backfill --since 2026-02-01 --timeframes 1h --out history --pace 3.5
uv run memetrader backfill --since 2026-03-01 --timeframes 5m --out history --pace 3.5
uv run python tools/extract_ticks.py --out ticks.jsonl
uv run python tools/verify_backfill.py --ticks ticks.jsonl --history history
```

`--since` is deliberately set earlier than the reachable horizon. The walk is
stopped by the vendor's 401, which is recorded per series as
`horizon_reached: true`, so the start date is a measured fact rather than a
guess that has to be revised as the window slides.

`--resume` skips series already complete in the manifest, so an interrupted
multi-hour pull is cheap to restart.
