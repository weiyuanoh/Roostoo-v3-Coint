# Roostoo V3 statistical-arbitrage bot

This repository is the independent successor to the V2 bot. It currently contains
strategy-neutral infrastructure through the Checkpoint 3 historical-replay foundation:

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
- non-atomic grouped multi-leg results with visible partial and failed legs.

There is intentionally no VECM, Johansen, Kalman, cointegration, production strategy, risk, or live
execution coordinator. The package does not import the V2 repository at runtime. The v6 short
mutation methods exist as an API boundary, but the CLI exposes no commands that open, close, place,
or cancel orders.

The accounting identity, Roostoo mappings, V2 reuse decisions, and unresolved API questions are
documented in [`docs/accounting.md`](docs/accounting.md). Historical timing, fill assumptions,
costs, grouping, and limitations are defined in [`docs/backtesting.md`](docs/backtesting.md).

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
```

Historical boundaries accept either 13-digit epoch milliseconds or timezone-aware ISO-8601:

```bash
poetry run stat-arb-bot collect BTC/USD --interval 1h \
  --start-time 2026-01-01T00:00:00Z --end-time 2026-02-01T00:00:00Z
```

By default, logs are written to `logs/bot.jsonl` and candles to `data/candles/`; both are ignored
by Git and configurable through `.env.example`.

## Verification

```bash
poetry run ruff check .
poetry run pytest
```

`API.md` is the local source of truth for the new v6 short endpoints. Anything it does not specify
(including financing/funding behavior, liquidation mechanics, and broader margin semantics) remains
unimplemented and must be confirmed before live long-short execution is enabled.
