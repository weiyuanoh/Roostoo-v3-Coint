# Roostoo V3 statistical-arbitrage bot

This repository is the independent successor to the V2 bot. It currently contains
strategy-neutral infrastructure through the Checkpoint 6 ECM-driven state-space foundation:

- validated, explicit environment configuration and structured logging;
- a Roostoo v3 client with signing, safe GET retries, typed errors, and account/order queries;
- typed support for the locally documented v6 short-open, short-close, and short-position APIs;
- Binance public-spot candle retrieval with endpoint fallback, pagination, UTC normalization,
  forming-candle exclusion, and continuity/staleness checks;
- deterministic local CSV candle storage; and
- a read-only operations CLI;
- exchange-neutral `Decimal` fills with optional trade-group metadata;
- signed long/short positions, segregated short collateral, fees, financing, and marked PnL;
- multi-asset portfolio snapshots with NAV, GMV, NMV, exposures, and leverage; and
- atomic, versioned portfolio-state persistence plus Roostoo reconciliation mappings;
- synchronized completed-bar replay with a bounded strategy history;
- explicit close-observed/next-open information clocks and anti-lookahead enforcement;
- market/limit simulated execution, configurable costs, financing, and liquidity policies; and
- non-atomic grouped multi-leg results with visible partial and failed legs;
- canonical BTC/ETH/SOL/XRP/ADA five-minute Binance history under one configurable quote;
- exact UTC-aligned 15-minute aggregation and conservative five-asset synchronization;
- bounded close/log-price/log-return research windows with structured history validation; and
- deterministic source/panel fingerprints, atomic manifests, and incremental reconstruction;
- structured level/difference ADF and KPSS integration diagnostics;
- explicit VAR(1–8) AIC/BIC/HQIC selection with a tested VECM lag mapping;
- Johansen trace/max-eigen rank diagnostics without forcing rank one;
- inspectable rank-one beta, alpha, Gamma, residual covariance, and ECM decomposition;
- empirical and deterministic VECM convergence diagnostics; and
- immutable daily-fit scheduling, cross-window stability comparison, and safe JSON persistence;
- an exact augmented-state representation for every configured VECM lag order;
- residual-covariance innovation loading and explicit configurable observation noise;
- rank-gated ECM-driven Kalman filtering with audited structural-refit handoffs;
- observed/filtered ECM, separate alpha and Gamma contributions, and covariance-aware forecasts;
- random-walk KF and plain-VECM forecast baselines; and
- deterministic filter/regime persistence and chronological forecast diagnostics.

There is intentionally no production strategy, corrector/anchor selection, trade construction,
position sizing, risk layer, or live execution coordinator. The package does not import the V2
repository at runtime. The v6 short mutation methods exist as an API boundary, but the CLI exposes no
commands that open, close, place, or cancel orders.

The accounting identity, Roostoo mappings, V2 reuse decisions, and unresolved API questions are
documented in [`docs/accounting.md`](docs/accounting.md). Historical timing, fill assumptions,
costs, grouping, and limitations are defined in [`docs/backtesting.md`](docs/backtesting.md).
The five-asset mappings, aggregation, synchronization, research windows, and reproducibility
contract are defined in [`docs/research-data.md`](docs/research-data.md).
The structural assumptions, chronology, parameter interpretation, convergence, and stability
diagnostics are defined in [`docs/modelling.md`](docs/modelling.md).
The fixed-beta VECM state-space derivation, filter clock, rank transitions, forecasts, baselines, and
restart contract are defined in [`docs/state-space.md`](docs/state-space.md).

The universal replay rule is:

```text
information through close[k] -> signal at k -> earliest execution at open[k+1]
```

## Requirements and setup

The project targets Python 3.11.15 and uses Poetry 2.x.

```bash
poetry env use 3.11.15
poetry install
cp .env.example .env
```

For signed Roostoo queries, set `ROOSTOO_API_KEY` and `ROOSTOO_API_SECRET` in the process
environment or an explicitly selected dotenv file. Never commit `.env`.

## Read-only commands

```bash
poetry run stat-arb-bot smoke
poetry run stat-arb-bot exchange-info
poetry run stat-arb-bot ticker --pair BTC/USD
poetry run stat-arb-bot --env-file .env balance
poetry run stat-arb-bot --env-file .env short-positions
poetry run stat-arb-bot collect BTC/USD ETH/USD --interval 1h --limit 1000
poetry run stat-arb-bot validate-candles BTC/USD ETH/USD --interval 1h
poetry run stat-arb-bot research-fetch \
  --start 2026-01-01T00:00:00Z --end 2026-07-01T00:00:00Z
poetry run stat-arb-bot research-build --as-of 2026-07-01T00:00:00Z
poetry run stat-arb-bot research-coverage --as-of 2026-07-01T00:00:00Z
```

Historical boundaries accept either 13-digit epoch milliseconds or timezone-aware ISO-8601:

```bash
poetry run stat-arb-bot collect BTC/USD --interval 1h \
  --start-time 2026-01-01T00:00:00Z --end-time 2026-02-01T00:00:00Z
```

By default, logs are written to `logs/bot.jsonl`, operational candles to `data/candles/`, and
canonical/derived research data to `data/research/`. These paths are ignored by Git and configurable
through `.env.example`.

## Verification

```bash
poetry run ruff check .
poetry run pytest
```

`API.md` is the local source of truth for the new v6 short endpoints. Anything it does not specify
(including financing/funding behavior, liquidation mechanics, and broader margin semantics) remains
unimplemented and must be confirmed before live long-short execution is enabled.
