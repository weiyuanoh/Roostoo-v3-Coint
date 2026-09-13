# Synchronized research-data layer

Checkpoint 4 supplies reproducible inputs for later statistical work. It does not perform ADF,
KPSS, Johansen, VAR/VECM, ECM, half-life, Kalman, PCA, causality, scoring, signal generation, or
trade selection.

## Universe and common quote

The canonical modelling universe is fixed, in order, to:

```text
BTC, ETH, SOL, XRP, ADA
```

`RESEARCH_QUOTE_CURRENCY` is configurable and defaults to `USDT`, matching the Binance public spot
markets already used by the project. One `ResearchUniverse` has exactly one quote; mixed-quote
panels are rejected by construction.

| Canonical asset | Normalized research pair | Binance symbol |
| --- | --- | --- |
| BTC | `BTC/USDT` | `BTCUSDT` |
| ETH | `ETH/USDT` | `ETHUSDT` |
| SOL | `SOL/USDT` | `SOLUSDT` |
| XRP | `XRP/USDT` | `XRPUSDT` |
| ADA | `ADA/USDT` | `ADAUSDT` |

The legacy Roostoo-facing `/USD` aliases remain unchanged. Research code addresses canonical assets;
the universe/Binance adapter owns pair translation.

## Canonical raw data

Binance public spot klines are retrieved and retained at `5m`. Each `Candle` records symbol,
interval, UTC open/close timestamps, OHLCV, source, and retrieval timestamp. Historical requests are
start-open-inclusive and end-completed-close-inclusive. The requested start/end are configurable,
so six, twelve, or more months can use the same core path.

Only rows whose source close timestamp is no later than the explicit `as_of` boundary are canonical.
Forming observations are diagnosed and excluded. The existing `Candle` constructor enforces finite
positive prices, valid OHLC relationships, non-negative finite volume, and aware timestamps. Existing
closure, interval, continuity, ordering, and staleness inspection is reused.

Repeated retrieval is idempotent when market content is identical. Retrieval timestamp differences
do not make an otherwise identical observation conflict. If OHLCV, interval, symbol, or bar bounds
disagree for the same symbol/open timestamp, `RawDataConflictError` is raised; neither row silently
wins.

Raw diagnostics retain exact duplicates, missing intervals, out-of-order/non-monotonic input,
forming bars, stale series, unexpected symbols/intervals, invalid values, and large continuity gaps.
No interpolation or automatic repair is performed.

## UTC 15-minute aggregation and observability

Quarter-hour model intervals are explicit UTC left-closed/right-open buckets:

```text
[00:00, 00:15), [00:15, 00:30), [00:30, 00:45), [00:45, 01:00), ...
```

The model bar is labelled with `open_time` at the left boundary and indexed/observed at the right
boundary. Thus raw bars opened at 10:00, 10:05, and 10:10 create one model bar opened at 10:00,
closed/indexed/observable at 10:15.

Exactly the three aligned raw opens are required. Aggregation is:

```text
open   = first raw open
high   = max(raw highs)
low    = min(raw lows)
close  = third raw close
volume = sum(raw volumes)
```

Two rows, a missing middle row, a misaligned row, an invalid constituent duration, or a bucket not
complete by `as_of` produces no model bar. Such buckets are reported, never partially aggregated.
Alignment is implemented directly rather than delegated to library resampling defaults.

## Five-asset synchronization

The panel index is the model-bar observable close timestamp. A row is admitted only when all five
normalized pairs have one valid bar for the exact same `[open, close)` interval. Missing SOL, for
example, drops the entire timestamp; SOL from the prior interval is never forwarded into it.

Panel diagnostics record forming keys, unexpected symbols/intervals, exact duplicates,
per-symbol out-of-order/non-monotonic input, fully absent quarter-hours, missing assets, and
cross-symbol interval mismatch. Rows are immutable and strictly chronological.

The panel exposes three unmodified representations in canonical asset order:

```text
P_t       = close price
x_t       = log(P_t)
Delta x_t = x_t - x_(t-1)
```

Levels are not standardized or demeaned. Returns span consecutive valid panel rows; window gap
validation makes a dropped interval explicit before future modelling.

## Bounded research windows

`ResearchPanel.window` constructs an immutable subset using either:

- a requested observation count ending at `t_k`; or
- a calendar start/lookback and end boundary.

Only panel rows with timestamps at or before the requested end are considered. A valid window ends
exactly at `t_k`. Validation reports, rather than hiding, an unavailable end, insufficient valid
observations, invalid chronology, incomplete assets, non-finite values, and internal gaps beyond a
caller-selected maximum.

Calendar duration and actual valid observation count remain distinct. At uninterrupted 15-minute
frequency, 14/30/60 days are approximately 1,344/2,880/5,760 rows, but the implementation never
substitutes those counts for timestamp/gap validation.

## Provenance and future model chronology

Canonical raw market content, universe, quote, and frequency produce a deterministic SHA-256 source
fingerprint. Retrieval time is excluded so retrieving identical source rows again does not create a
new data version. The synchronized panel also has a deterministic fingerprint.
`source_fingerprint_through(t_k)` derives fit provenance from only raw rows available through
`t_k`, so later observations cannot change the identifier of an earlier research input.

`ModelFitMetadata` is provenance only. It records model name/version/fit ID, fit timestamp, data
start/end, interval, universe, quote, observation count, and source fingerprint. It enforces:

```text
data_start <= data_end <= fit_timestamp
```

It can provide the `model_version` and `model_fit_end_timestamp` fields already enforced by
Checkpoint 3 Orders. No fitted parameters or statistical implementation exists here.

## Storage, bootstrap, and incremental refresh

The existing atomic `CandleStore` is reused for both candle layers, under distinct roots:

```text
data/research/raw/USDT/BTC_USDT_5m.csv
data/research/derived/USDT/BTC_USDT_15m.csv
data/research/manifests/USDT/<source-fingerprint>.json
```

Equivalent paths exist for the other assets. Raw 5-minute files remain the source of truth. Derived
files may be deleted and rebuilt. Atomic JSON manifests record bounds, mappings, counts, source and
panel fingerprints, and affected aggregation buckets. Future model outputs belong in a separate
model-output namespace and are not created in this checkpoint.

Historical bootstrap accepts arbitrary UTC start/end bounds. Incremental retrieval overlaps each
latest stored open deliberately, merges exact duplicates, and rebuilds affected quarter-hour
buckets. The synchronized panel is reconstructed from the final derived set. Tests require the
incremental and clean-full outputs/fingerprints to match.

## Read-only research CLI

The commands make public data/local state inspectable but never place an exchange order:

```bash
poetry run stat-arb-bot research-fetch \
  --start 2026-01-01T00:00:00Z --end 2026-07-01T00:00:00Z

poetry run stat-arb-bot research-fetch \
  --start 2026-01-01T00:00:00Z --end 2026-07-02T00:00:00Z --incremental

poetry run stat-arb-bot research-build --as-of 2026-07-02T00:00:00Z
poetry run stat-arb-bot research-coverage --as-of 2026-07-02T00:00:00Z
```

Coverage reports include per-asset raw/model counts, earliest/latest raw times, missing intervals,
excluded model bars, forming count and staleness, plus common panel bounds, synchronized count,
dropped timestamp count, and source fingerprint. Commands refuse an `as_of`/end that would truncate
newer stored canonical raw history.

## Relationship to the backtest information clock

The central chronology is:

```text
three raw 5m bars complete
-> the UTC 15m right boundary t_k is observable
-> synchronized panel row k becomes available
-> research/model fit may use data through t_k
-> signal may be generated at t_k
-> the next-bar open price is the earliest execution price
```

Adjacent left-closed model bars share a wall-clock boundary: bar `k` closes and bar `k+1` opens at
the same UTC label. Checkpoint 3 maintains strict event ordering by assigning the eligible execution
event one microsecond after the signal when those labels coincide. The execution price remains the
`k+1` open. This is event sequencing, not extra market information or a one-bar delay.

Permanent integration tests assert that the panel row time equals `StrategyContext`'s observable
time and that its resulting Fill is strictly later and uses the following bar's open.
