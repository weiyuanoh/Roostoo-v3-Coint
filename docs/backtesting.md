# Historical replay and simulated execution

This document defines the Checkpoint 3 backtesting contract. It is infrastructure only: no alpha,
Johansen, VECM, ECM, Kalman, half-life, pair selection, sizing, or risk rule is implemented.

> **Global anti-lookahead rule:** information through `close[k]` produces a signal at `k`; the
> earliest execution is in `k+1`. A newly created order can never trade at the open, high, low, or
> close of its signal bar.

## V2 inspection and migration decisions

V2 remains read-only and TARGET has no runtime import from it.

| V2 source | Useful behavior | Checkpoint 3 decision |
| --- | --- | --- |
| `bot/backtest/portfolio.py::Portfolio` | Cash, holdings, fill fees, equity series, and trade records | Do not copy. It is float-based, long-only, stores one entry per symbol, sells a whole position, and conflates some fee/PnL reporting. TARGET's Checkpoint 2 signed Decimal Portfolio is the accounting authority. |
| `bot/backtest/portfolio.py::compute_equity_metrics` | Total return, drawdown, Sharpe, and Sortino summary | Defer. It assumes hourly samples by taking every 24th point and annualizing daily returns, so it is not a safe generic metric for arbitrary model bars. TARGET result snapshots retain everything needed for an interval-aware metric layer. |
| `bot/backtest/simulated_executor.py` | Separate simulated-execution surface, precision/minimum-order checks, slippage, stable order-like results | Preserve the separation, configurable slippage idea, and deterministic IDs. Reimplement against normalized Orders/Fills and signed Portfolio. V2 fills immediately at the caller's current price, ignores bid/ask and `use_limit`, has no partial fills, and cannot short. |
| `bot/backtest/ridge_score_portfolio.py::_simulate_scored_window` | Chronological grouping by timestamp, mark/equity rows, executor event collection | Preserve chronological replay and structured state rows. Replace the loop: V2 generates intents from a bar and executes at that same bar's close, which violates this checkpoint's timing invariant. Final same-price liquidation is also not migrated. |
| `bot/backtest/ridge_score_portfolio.py::run_portfolio_backtest` | Walk-forward folds with train/test boundaries | Keep as a future research convention, not as generic engine code. Its ridge/regime/cluster gates are strategy-specific. |
| `bot/backtest/ridge_score.py` and experiment modules | Scoring, entries/exits, regime diagnostics, refit experiments | Do not migrate in Checkpoint 3; these are V2 alpha and research workflows. |
| `bot/telemetry.py` | Stable IDs, order/fill fields, slippage and closed-trade metadata | Preserve the structured metadata concepts. Strategy/live JSONL behavior is not copied. Deterministic replay IDs and model chronology fields live in the new domain results. |

V2 has useful walk-forward boundaries, but its ordinary portfolio simulation does not enforce an
explicit information clock. The TARGET timing guarantees therefore come from the new implementation,
not an assertion that V2's same-bar replay was already lookahead-safe.

## Information clock and event ordering

Every synchronized frame has four explicit UTC timestamps:

- `bar_open`: first time represented by the bar;
- `bar_close`: end of its OHLCV interval;
- `observable_at`: no earlier than `bar_close` and currently equal to it;
- `first_executable_at`: the next synchronized frame's open, strictly after `observable_at`.

For a completed frame `k`, replay follows this order:

1. Orders that existed before frame `k` and are eligible execute against frame `k` assumptions.
2. Their normalized Fills are applied to the signed Portfolio.
3. Positions are marked at `close[k]`.
4. Configured financing timestamps newly known by `close[k]` are posted once.
5. A bounded history ending at `k` and an immutable Portfolio snapshot are passed to the strategy.
6. Returned OrderIntents become Orders timestamped at `observable_at[k]`.
7. Those Orders receive `first_executable_at = open[k+1]`; no field from frame `k` is considered for
   their fills.
8. The marked state is retained as a result snapshot.

The engine processes fills at a future bar's open before it exposes that future bar's close to the
strategy. Thus a historical loop may possess the complete input data, but the strategy never does.

## Strategy information boundary and synchronization

`MarketHistoryView` contains an immutable prefix of synchronized `HistoricalFrame` objects. It has
no reference to frames after the current index. `window(symbol, W)` returns at most the last `W`
completed observations and always ends at the current frame. `StrategyContext` rejects a history
whose maximum observation time exceeds its information clock.

`HistoricalPanel` requires all configured symbols to have bars with identical open and close times.
A timestamp with a missing asset or mismatched close is skipped and reported; it is never
forward-filled. Bars whose close is later than the run's `as_of` time are excluded and counted.
Duplicate symbol/timestamp bars and mixed intervals are errors. `as_of` is mandatory so the
forming-bar boundary is an explicit, reproducible input rather than a hidden wall-clock dependency.

Skipping an unsynchronized timestamp makes the next fully synchronized frame the next decision and
execution opportunity. There is no claim that a missing physical bar was synthesized.

## Intent, order, fill, and Portfolio boundary

```text
StrategyContext -> OrderIntent -> Order -> SimulatedExecutor -> Fill -> Portfolio
```

- `OrderIntent` states economic BUY/SELL direction, quantity, order type, optional collateral,
  trade group, phase, and model metadata.
- `Order` adds deterministic ID plus signal-information, signal, submission, and first-eligible
  timestamps. Construction rejects future model fits and non-causal execution eligibility.
- `SimulatedExecutor` produces normalized Checkpoint 2 `Fill` objects and a separate cost breakdown.
- Only a `Fill` mutates Portfolio. BUY increases signed inventory; SELL decreases it.

Roostoo-specific `SHORT_OPEN` and `SHORT_CLOSE` are not generic order sides. Under `API.md`, a filled
short open maps economically to normalized SELL plus restricted collateral, and a reduce-only short
close maps to normalized BUY. Future live execution must map these to the v6 actions rather than
send an ordinary spot SELL. Roostoo's reduce-only close cannot reverse into a long; a live reversal
would require separate close and open actions even though Portfolio can account for a logical
signed-inventory reversal.

The simulator's LIMIT is an exchange-neutral conventional buy/sell limit. It must not be translated
blindly into Roostoo's collateral-sized `SHORT_OPEN` limit: `API.md` documents exchange-specific
trigger behavior for that action. A future Roostoo execution adapter owns that translation.

An intent's `short_collateral` is the total restricted collateral assigned to its intended
short-opening order quantity. Partial pure-short fills allocate it pro rata. Orders crossing through
zero should be split into close and open intents, matching the exchange boundary and avoiding an
ambiguous collateral allocation across partial reversal fills.

## Fill conventions

Market orders use `open[k+1]` as their reference. A BUY adds half the configured full spread and
adverse slippage; a SELL subtracts them. Slippage may contain a basis-point amount and a fixed
per-unit price amount. The price effect is an implicit economic cost, while spread and slippage
notionals remain separately recorded.

A future-bar buy limit fills only when that bar's low is at or below the limit. A sell limit fills
only when its high is at or above the limit. It always fills at the limit, including a favorable
opening gap; the simulator does not grant the more favorable open. A gap-through fill is timestamped
at the bar open. A fill supported only by the future bar's high/low is conservatively timestamped at
that bar's close, because OHLC cannot reveal its actual touch time. With OHLC alone, the ordering of
high and low, queue priority, and available liquidity at the limit are unknowable.

Full-fill mode provides unlimited simulated bar capacity. Capacity-limited mode shares an explicit
fraction of base-asset bar volume among same-symbol orders in deterministic submission order. An
unfilled limit or capacity-constrained remainder stays pending for later eligible frames. A
configured rejection policy surfaces a failed leg; it never manufactures an offsetting fill.

## Costs and financing

The cost model exposes independently:

- maker fee;
- taker fee;
- full bid/ask spread assumption;
- basis-point slippage;
- fixed per-unit slippage;
- optional short-open fee override;
- optional short-close fee override; and
- symbol-specific parameter overrides.

Fees become `Fill.fee` and Portfolio debits them exactly once. Execution-price spread/slippage is
not added again as a fee. The simulator has no hard-coded Roostoo rates.

`ScheduledShortFinancingModel` accepts explicit timestamps and signed configuration rates. A
positive rate creates a debit based on the current marked absolute short notional; a negative rate
creates a credit. It emits the generic Checkpoint 2 `FinancingEntry`, which Portfolio posts once.
The zero model is the default. `API.md` supplies no borrow/funding schedule or rate, so no value is
inferred for Roostoo.

## Multi-leg groups and result state

Orders and fills may retain a `trade_group_id`; inventory remains per instrument. OPEN, CLOSE, and
ADJUST phases allow a result to derive combined entry/exit timestamps when every corresponding leg
fills. A group reports intended/submitted/filled/failed/partial legs, first eligibility, fees,
spread, slippage, explicitly attributed financing, and realized gross PnL. Financing is assigned
to a group only when its generic metadata carries that group ID; it is never guessed from a symbol
that multiple trades may share. Status is:

- `COMPLETE` when every submitted instruction fully fills;
- `PARTIAL` when any actual fill exists but at least one instruction is incomplete or failed;
- `FAILED` when no instruction fills and a terminal failure/expiry exists;
- `PENDING` only for a non-terminal intermediate view.

One leg filling, a hedge failure, a partial quantity, and one-sided close failure all leave the
actual Portfolio exposure visible. There is no synthetic pair position, atomic-fill fiction, or
automatic hedge recovery.

`BacktestResult` retains immutable Portfolio snapshots, final OrderRecords, every normalized Fill
with costs and its PositionTransition, applied financing entries, group summaries, and the panel
quality report. Orders and Fill metadata carry optional model version and fit-end timestamp for
later replay audits.

## Decimal boundary and determinism

Market candles remain floats. `execution_decimal` is the single conversion boundary and delegates
to the checked string-based Decimal normalization used by accounting. Execution prices, quantities,
fees, collateral, financing, and Portfolio values then remain Decimal.

Every run creates a fresh Portfolio and fresh order/group state, resets its strategy, restarts IDs at
`sim-00000001`, and uses stable chronological/submission ordering. No pending order, fill capacity,
financing cursor, or Portfolio value survives a run.

## Deliberate limitations

- No strategy, risk constraints, order sizing, cancellation API, time-in-force, latency, queue
  model, complex market impact, or automatic failed-hedge recovery exists.
- Volume capacity is a simple documented ceiling, not evidence of executable liquidity.
- OHLC bars cannot determine intrabar path; limit fills use the conservative rule above.
- Orders submitted on the final loaded frame expire because no eligible future frame is present.
- Exchange precision, minimum-notional validation, and Roostoo live-action coordination remain for
  production execution work.
- Generic performance summaries are deferred until their sampling/annualization contract is
  explicit; raw time-series state is retained now.
