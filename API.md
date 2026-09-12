# **Roostoo Shorting API — Quick Brief**

**Open short position (Trade)**

```
POST /v6/short_open
Auth RCL_TopLevelCheck
```

Open a new short position, or add to the one you already hold on that pair.

Note these short endpoints are served under `/v6` instead of `/v3`. They use the same host, the same `API_KEY` / `SECRET_KEY` and the same `RCL_TopLevelCheck` signing rules as the endpoints above.

**Parameters**

| Name | Type | Mandatory | Description |
| --- | --- | --- | --- |
| pair | STRING | YES | Used with `BTC/USD`, etc... |
| collateral | STRING | YES | The USD amount to lock as collateral. Minimum `1`. |
| timestamp | STRING | YES | Used with 13-digits millsecomd timestamp |
| order_type | STRING | NO | Used with `LIMIT`. If not sent, the order is a market order. |
| price | DECIMAL | NO |  |

Additional mandatory parameters based on `order_type`:

| Type | Additional mandatory parameters |
| --- | --- |
| `LIMIT` | `price` |

Other info:

- There is no `side` or `quantity` parameter. A short is sized by the `collateral` you commit, and the quantity is calculated from it as `collateral / EntryPrice`.
- A market order fills immediately at the current best bid, charged at the taker fee rate.
- A `LIMIT` order does not fill when placed. It stays pending until the market reaches your price, and is charged at the maker fee rate. If `price` is at or above the current market it fills when the price rises, if it is below the current market it fills when the price drops.
- Opening again on a pair you are already short will merge into the existing position, using a quantity weighted average entry price. No second position is created.
- The pending order created by a `LIMIT` order can be canceled with `POST /v3/cancel_order` using its `order_id`. Canceling releases both the locked collateral and the open fee.
- Your free `USD` balance must cover `collateral` + fee, otherwise you will get a `ErrMsg` in response.

**Response when it's a market order:**

```json
{
  "Success": true,
  "ID": 412,
  "Pair": "BTC/USD",
  "OrderType": "MARKET",
  "EntryPrice": 61840.25,
  "ShortQty": 0.01617,
  "Collateral": 999.95,
  "OpenFee": 0.119994,
  "Status": "OPEN",
  "CreateTimestamp": 1757980800000
}
```

**Response when it's a limit order:**

```json
{
  "Success": true,
  "ID": 90271,
  "Pair": "BTC/USD",
  "OrderType": "LIMIT",
  "EntryPrice": 63000,
  "ShortQty": 0.015873,
  "Collateral": 999.99,
  "OpenFee": 0.079999,
  "Status": "PENDING",
  "CreateTimestamp": 1757980800000
}
```

**Return Explain**

| Name | Type | Description |
| --- | --- | --- |
| ID | INT | When `Status` is `OPEN` this is the short position id, the same id returned by `/v6/short_positions`. When `Status` is `PENDING` this is the order id, used to cancel the order. |
| Pair | STRING | The pair of this position. |
| OrderType | STRING | `MARKET` or `LIMIT`. |
| EntryPrice | FLOAT | The filled price for a market order, or your requested price for a limit order. After a merge it is the weighted average entry price of the whole position. |
| ShortQty | FLOAT | The quantity shorted. After a merge it is the total quantity of the whole position. |
| Collateral | FLOAT | The USD locked against the position. After a merge it is the total collateral of the whole position. |
| OpenFee | FLOAT | The commission charged for this request only. It is not the cumulative fee of a merged position. |
| Status | STRING | `OPEN` means filled and the position is live. `PENDING` means the limit order is waiting to be filled. |
| CreateTimestamp | INT | The 13-digits millsecomd timestamp of this request. |

## **Close short position (Trade)**

```
POST /v6/short_close
Auth RCL_TopLevelCheck
```

Close all or part of an open short position. It always fills immediately at the current best ask.

**Parameters**

| Name | Type | Mandatory | Description |
| --- | --- | --- | --- |
| pair | STRING | YES | Used with `BTC/USD`, etc... |
| timestamp | STRING | YES | Used with 13-digits millsecomd timestamp |
| close_qty | STRING | NO | The absolute quantity to close. |
| close_pct | STRING | NO | The percentage to close, greater than `0` and at most `100`. |

Other info:

- if `close_qty` is sent, it takes precedence over `close_pct`.
- if none of `close_qty` and `close_pct` is sent, system will close the whole position.
- every close is reduce only. A `close_qty` bigger than the open quantity is reduced to it, so a short can never be over closed or turned into a long.
- a partial close keeps the entry price unchanged and reduces the quantity and the collateral in proportion. The remaining collateral stays locked until the position is fully closed.
- a short can never lose more than the collateral backing it, so `RealizedPNL` is capped at that loss.
- if the requested part would leave a remainder too small to keep, system will close the whole position instead and `FullyClosed` will be `true`.

**Response when it's a partial close:**

```json
{
  "Success": true,
  "ClosePrice": 60500.1,
  "RealizedPNL": 10.8258,
  "CloseFee": 0.0484,
  "ReturnAmount": 510.7774,
  "ClosedQty": 0.008085,
  "FullyClosed": false,
  "RemainingQty": 0.008085,
  "RemainingCollateral": 499.98
}
```

**Response when it's a full close:**

```json
{
  "Success": true,
  "ClosePrice": 60500.1,
  "RealizedPNL": 21.6516,
  "CloseFee": 0.0968,
  "ReturnAmount": 1021.5048,
  "ClosedQty": 0.01617,
  "FullyClosed": true
}
```

**Return Explain**

| Name | Type | Description |
| --- | --- | --- |
| ClosePrice | FLOAT | The price this close was filled at. |
| RealizedPNL | FLOAT | The settled profit or loss on the closed part of the position. |
| CloseFee | FLOAT | The commission charged on the closed part. |
| ReturnAmount | FLOAT | The USD returned to your wallet, `closed collateral + RealizedPNL - CloseFee`. It can be negative when the loss reaches the full collateral. |
| ClosedQty | FLOAT | The quantity actually closed. |
| FullyClosed | BOOL | `true` if the position is now fully closed, `false` if a part of it is still open. |
| RemainingQty | FLOAT | The quantity still open. Not returned when it's a full close. |
| RemainingCollateral | FLOAT | The collateral still locked. Not returned when it's a full close. |

## **Get short positions**

```
GET /v6/short_positions
Auth RCL_TopLevelCheck
```

Get all your currently open short positions, with live profit and loss.

**Parameters**

| Name | Type | Mandatory | Description |
| --- | --- | --- | --- |
| timestamp | STRING | YES | Used with 13-digits millsecomd timestamp |

Other info:

- only open positions are returned. Closed positions can be found in your order history with `POST /v3/query_order`, where opening and closing a short appear as orders with `Side` = `SHORT_OPEN` and `Side` = `SHORT_CLOSE`.
- `Positions` is always a list. It is `[]` when no position is open, never `null`.
- `UnrealizedPNL` is the value a close would realize before the close fee is charged.

**Response when at least one position is open:**

```json
{
  "Success": true,
  "Positions": [
    {
      "ID": 412,
      "Pair": "BTC/USD",
      "EntryPrice": 61840.25,
      "ShortQty": 0.01617,
      "Collateral": 999.95,
      "CurrentPrice": 60500.1,
      "UnrealizedPNL": 21.670225,
      "UnrealizedPNLPct": 0.021671,
      "PositionValue": 1021.620225,
      "CreateTimestamp": 1757980800000,
      "PositionStatus": "OPEN"
    }
  ]
}
```

**Response when no position is open:**

```json
{
  "Success": true,
  "Positions": []
}
```

**Return Explain**

| Name | Type | Description |
| --- | --- | --- |
| ID | INT | The short position id, the same id returned by `/v6/short_open` when `Status` is `OPEN`. |
| Pair | STRING | The pair of this position. |
| EntryPrice | FLOAT | The weighted average entry price of the position. |
| ShortQty | FLOAT | The quantity shorted. |
| Collateral | FLOAT | The USD locked against the position. |
| CurrentPrice | FLOAT | The current market price a close would be filled at. |
| UnrealizedPNL | FLOAT | The profit or loss if the position were closed now, before the close fee. |
| UnrealizedPNLPct | FLOAT | `UnrealizedPNL` against the collateral, like `0.0217` you can see it as `2.17%` profit, or `-0.0107` as `1.07%` loss. |
| PositionValue | FLOAT | `Collateral + UnrealizedPNL`, what a full close is worth before the fee. |
| CreateTimestamp | INT | The 13-digits millsecomd timestamp when the position was opened. |
| PositionStatus | STRING | Always `OPEN` on this endpoint. |

**Response when a short request fails:**

```json
{
  "Success": false,
  "ErrMsg": "insufficient balance"
}
```