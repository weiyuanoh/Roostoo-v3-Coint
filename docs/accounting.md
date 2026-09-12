# Long-short accounting foundation

This document records the Checkpoint 2 accounting decisions. `API.md` is the primary local
specification for Roostoo short behavior; this document does not add exchange guarantees that are
absent there.

## Roostoo behavior explicitly documented in `API.md`

- `POST /v6/short_open` has no side or quantity request field. It commits at least 1 USD of
  collateral and Roostoo calculates quantity as `collateral / EntryPrice`.
- A market short opens immediately at the best bid. A limit short remains `PENDING` until filled.
- Repeated short opens on the same pair merge into one short position. Returned `ShortQty` and
  `Collateral` are totals, `EntryPrice` is the quantity-weighted average for the merged position,
  and `OpenFee` applies only to the latest request.
- Free USD must cover collateral plus the open fee. Canceling a pending limit releases its locked
  collateral and open fee.
- `POST /v6/short_close` is a best-ask, reduce-only close. It accepts an absolute `close_qty` or a
  `close_pct`; quantity takes precedence, and omitting both closes the whole position.
- An over-sized close is capped at the open quantity and cannot reverse short inventory into long
  inventory. A small remainder may be fully closed automatically.
- A partial close leaves average entry unchanged and reduces quantity and collateral
  proportionally. A full close releases the position.
- The close response reports `RealizedPNL`, `CloseFee`, `ClosedQty`, `FullyClosed`, and
  `ReturnAmount = released collateral + RealizedPNL - CloseFee`. Partial closes also report
  remaining quantity and collateral. A realized loss is capped at the backing collateral.
- `/v6/short_positions` returns only open positions. `ShortQty` is a positive magnitude; the
  endpoint also reports weighted average entry, collateral, current price, pre-close-fee
  unrealized PnL, position value, creation time, and `OPEN` status.
- Short history uses Roostoo order sides `SHORT_OPEN` and `SHORT_CLOSE`.

## Undocumented or unresolved exchange behavior

`API.md` does not specify:

- whether a spot long and a short can coexist independently in the same pair;
- whether the account is formally one-way or hedge mode across spot and short facilities;
- whether short-sale proceeds ever appear in free, locked, or total wallet balances;
- a complete account-equity/NAV identity or the exact v3 balance fields for collateral;
- leverage selection, maintenance margin, liquidation, borrow, funding, financing, or interest;
- partial fills for limit short opens;
- a single-action long-to-short or short-to-long reversal;
- a close timestamp or close order/position ID in the v6 close response;
- rounding/precision guarantees for reconstructing the incremental price of a merged short open;
- how the collateral loss cap should be represented in a deterministic internal gross-PnL ledger
  if an uncapped mark-to-market loss exceeds collateral.

The numeric partial-close example is not exactly self-reconciling from its displayed values. Its
visible `ClosedQty * (EntryPrice - ClosePrice)` differs from `RealizedPNL`, and
`previous Collateral - RemainingCollateral + RealizedPNL - CloseFee` differs from
`ReturnAmount`. This may reflect hidden precision or an example defect; `API.md` does not say.
`reconcile_short_close` reports both differences rather than silently overwriting internal gross
PnL or wallet settlement expectations.

These points must be confirmed before live execution/reconciliation is enabled. The generic
financing ledger exists specifically so absence of a current API field is not treated as proof that
financing is zero.

## V2 components inspected

| V2 source | Finding | Checkpoint 2 treatment |
| --- | --- | --- |
| `bot/backtest/portfolio.py` | Simple cash plus long market value; buy/sell fees and long realized PnL. Its trade PnL includes the exit fee but omits the already-debited entry fee. | Generalized conceptually. The float, one-position-entry, full-sale-only implementation was not copied; gross PnL and all fees are now separate. |
| `bot/backtest/portfolio.py::compute_equity_metrics` | Backtest return/drawdown/Sharpe/Sortino calculations | Not migrated; performance metrics belong with Checkpoint 3 backtesting. |
| `bot/live_state.py` | Small JSON state file written through a temporary replacement; filters out non-positive inventory | Reused the atomic replacement pattern. State schema and all signed accounting fields were reimplemented. |
| `bot/executor.py` | Roostoo order fields and commission extraction; live order/pending logic | Inspected only. Execution is out of scope and was not migrated. |
| `bot/backtest/simulated_executor.py` | Fill-price/fee handoff to the old Portfolio | Inspected only. Simulation and backtesting are deferred to Checkpoint 3. |
| `bot/telemetry.py` | Trade IDs, fill metadata, gross PnL logging | Metadata needs informed the normalized Fill. Strategy/live telemetry was not migrated. |
| `bot/monitoring.py` | Wallet-versus-local quantity comparisons, positive-only holdings | Not migrated; it assumes long-only wallet inventory and strategy-specific state. |

The TARGET contains no imports from the V2 `bot` package.

The Checkpoint 1 Roostoo response dataclasses parse API numeric values as floats. The accounting
adapter converts their decimal text into `Decimal`, preventing further binary-float arithmetic but
cannot recover lexical precision already discarded at the client boundary. Changing those response
types would alter verified Checkpoint 1 behavior, so that possible client improvement is deferred
and exchange comparisons support an explicit tolerance.

## Exchange-to-domain mapping

| Roostoo concept | Internal representation |
| --- | --- |
| Filled `SHORT_OPEN` | `Fill(side=SELL)` because it decreases signed inventory |
| Filled `SHORT_CLOSE` | `Fill(side=BUY)` because it increases signed inventory toward zero |
| Positive `ShortQty` | Negative `Position.quantity` |
| Open/add collateral | `Fill.short_collateral`, then `Position.short_collateral` |
| Partial/full collateral return | `Fill.released_short_collateral`; proportional internal fallback when no explicit value is supplied |
| `OpenFee` / `CloseFee` | Positive `Fill.fee`, debited from cash once |
| Roostoo action names and IDs | Fill metadata for audit/reconciliation; never transition rules |
| `/v6/short_positions` item | Read-only `PositionObservation`, not authoritative internal PnL |
| Exchange unrealized PnL | Reconciliation value compared with deterministic signed-inventory PnL |
| Close `RealizedPNL` and `ReturnAmount` | Settlement observations compared with the deterministic cover application |
| `PENDING` short open | Not a Fill and rejected by the mapper |

A merged short-open response reports total quantity and weighted average, not the incremental fill.
The mapper therefore requires the previous exchange position and reconstructs the incremental price
algebraically. This reconstruction is tagged in metadata and remains subject to API rounding.

## Internal accounting convention

Inventory is net and signed:

- positive quantity is long;
- negative quantity is short;
- zero is flat.

BUY increases signed quantity and SELL decreases it. Whether a fill opens, adds, closes, covers, or
reverses is derived from prior inventory. A reversal is supported as an accounting operation even
though Roostoo's short-close endpoint is explicitly reduce-only and live execution would need to
split the actions.

Execution quantities, prices, fees, collateral, cash, and PnL use `Decimal`. Normalized fills retain
optional order IDs, a trade-group ID, original exchange action, and exchange metadata.

The current Portfolio uses segregated short collateral. Opening/increasing a short debits available
cash by its supplied collateral and credits restricted collateral by the same amount; it does not
credit short-sale proceeds to available cash. This is a declared internal convention consistent
with the documented collateral/return flow, not a claim about undocumented wallet fields.

The marked equity identity is:

```text
equity = available cash
       + long market value
       + restricted short collateral
       + short unrealized PnL
```

Exposure identities are:

```text
position market value = signed quantity * mark price
NMV = sum(signed position market values)
long market value = sum(positive market values)
short market value = sum(negative market values)
short exposure = abs(short market value)
GMV = long market value + short exposure
gross leverage = GMV / equity, when equity > 0
```

PnL is separately observable:

```text
unrealized PnL = signed quantity * (mark - average entry)
net PnL = realized gross PnL + unrealized PnL - fees + signed financing
```

Negative financing entries are charges and positive entries are credits. Fees and financing are
already reflected in cash, so they are not subtracted from the cash-based equity identity again.

## Persistence and reconciliation

`PortfolioStateStore` writes versioned JSON atomically using a temporary file, `fsync`, and replace.
Decimals are serialized as strings. Cash, signed positions, average entries, realized PnL, fees,
financing, marks, collateral, timestamps, fill history, financing history, and grouping metadata all
round-trip.

Roostoo position data maps to a `PositionObservation`. Reconciliation returns explicit
observed-minus-internal differences for quantity, average entry, mark, collateral, and unrealized
PnL with a caller-selected tolerance. Exchange values remain observations and do not overwrite the
deterministic Position model.

Short-close reconciliation likewise compares reported realized PnL and wallet return with the
internal cover transition. This is important because the displayed `API.md` sample does not
reconcile exactly. A live reconciliation policy for genuine differences is intentionally deferred
until the API precision and collateral-cap behavior are confirmed.
