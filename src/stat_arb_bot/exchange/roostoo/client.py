"""Roostoo v3 client plus the locally documented v6 short endpoints."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from typing import Any

import requests

from stat_arb_bot.exchange.roostoo.auth import sign_params
from stat_arb_bot.exchange.roostoo.errors import (
    RoostooAuthenticationError,
    RoostooRateLimitError,
    RoostooRejectedError,
    RoostooResponseError,
    RoostooTransportError,
)
from stat_arb_bot.exchange.roostoo.models import (
    ShortCloseResult,
    ShortOpenResult,
    ShortPosition,
)
from stat_arb_bot.observability.logging import get_logger

log = get_logger("exchange.roostoo")


def _decimal_string(value: str | int | float | Decimal, *, name: str) -> str:
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be finite")
    rendered = format(parsed, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


class RoostooClient:
    """Synchronous Roostoo REST client with safe GET retries.

    Mutating POST requests are deliberately never retried because the API does
    not document idempotency semantics.
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = "https://mock-api.roostoo.com",
        timeout: float = 10.0,
        max_attempts: int = 3,
        backoff_seconds: float = 0.25,
        session: requests.Session | None = None,
        clock_ms: Callable[[], int] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds must be non-negative")
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.max_attempts = int(max_attempts)
        self.backoff_seconds = float(backoff_seconds)
        self.session = session or requests.Session()
        self.clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self.sleeper = sleeper
        self.server_time_offset_ms = 0

    def _timestamp(self) -> str:
        return str(self.clock_ms() + self.server_time_offset_ms)

    def _require_credentials(self) -> None:
        if not self.api_key or not self.api_secret:
            raise RoostooAuthenticationError("ROOSTOO_API_KEY and ROOSTOO_API_SECRET are required")

    def _signed_request_parts(
        self, params: Mapping[str, Any]
    ) -> tuple[dict[str, str], str, dict[str, Any]]:
        self._require_credentials()
        headers, encoded, signed = sign_params(
            params,
            self.api_secret,
            now_ms=self._timestamp(),
        )
        headers["RST-API-KEY"] = self.api_key
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        return headers, encoded, signed

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        retryable: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        method = method.upper()
        may_retry = method == "GET" if retryable is None else retryable
        attempts = self.max_attempts if may_retry else 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                response = self.session.request(
                    method,
                    f"{self.base_url}{path}",
                    timeout=self.timeout,
                    **kwargs,
                )
                status = int(getattr(response, "status_code", 200))
                if status in {401, 403}:
                    raise RoostooAuthenticationError(f"{method} {path} returned HTTP {status}")
                if status == 429:
                    if attempt + 1 < attempts:
                        self.sleeper(self._retry_delay(response, attempt))
                        continue
                    raise RoostooRateLimitError(f"{method} {path} returned HTTP 429")
                response.raise_for_status()
                try:
                    payload = response.json()
                except (TypeError, ValueError) as exc:
                    raise RoostooResponseError(f"{method} {path} returned invalid JSON") from exc
                if not isinstance(payload, dict):
                    raise RoostooResponseError(
                        f"{method} {path} returned {type(payload).__name__}, expected object"
                    )
                return payload
            except (RoostooAuthenticationError, RoostooRateLimitError, RoostooResponseError):
                raise
            except requests.RequestException as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    self.sleeper(self.backoff_seconds * (2**attempt))
                    continue
                break
            except (TypeError, ValueError) as exc:
                raise RoostooResponseError(
                    f"{method} {path} returned invalid response data"
                ) from exc

        raise RoostooTransportError(f"{method} {path} failed: {last_error}") from last_error

    def _retry_delay(self, response: Any, attempt: int) -> float:
        headers = getattr(response, "headers", {}) or {}
        value = headers.get("Retry-After")
        if value:
            try:
                return max(0.0, float(value))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(value).timestamp()
                    return max(0.0, retry_at - time.time())
                except (TypeError, ValueError, OverflowError):
                    pass
        return self.backoff_seconds * (2**attempt)

    @staticmethod
    def _require_success(payload: dict[str, Any], *, operation: str) -> dict[str, Any]:
        if payload.get("Success") is False:
            message = str(payload.get("ErrMsg") or f"{operation} was rejected")
            raise RoostooRejectedError(message, payload=payload)
        if payload.get("Success") is not True:
            raise RoostooResponseError(f"{operation} response is missing Success=true")
        return payload

    # Public v3 endpoints

    def server_time(self) -> int:
        payload = self._request_json("GET", "/v3/serverTime")
        if "ServerTime" not in payload:
            raise RoostooResponseError("server-time response is missing ServerTime")
        try:
            return int(payload["ServerTime"])
        except (TypeError, ValueError) as exc:
            raise RoostooResponseError(
                "server-time response ServerTime must be an integer"
            ) from exc

    def sync_server_time(self) -> int:
        server_time = self.server_time()
        self.server_time_offset_ms = server_time - self.clock_ms()
        return self.server_time_offset_ms

    def exchange_info(self) -> dict[str, Any]:
        return self._request_json("GET", "/v3/exchangeInfo")

    def ticker(self, pair: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"timestamp": self._timestamp()}
        if pair:
            params["pair"] = pair.upper()
        payload = self._require_success(
            self._request_json("GET", "/v3/ticker", params=params),
            operation="ticker",
        )
        data = payload.get("Data", {})
        if not isinstance(data, dict):
            raise RoostooResponseError("ticker response Data must be an object")
        return data

    # Signed v3 endpoints

    def balance(self) -> dict[str, Any]:
        headers, _, params = self._signed_request_parts({})
        payload = self._require_success(
            self._request_json("GET", "/v3/balance", headers=headers, params=params),
            operation="balance",
        )
        wallet = payload.get("SpotWallet", payload.get("Wallet", {}))
        if not isinstance(wallet, dict):
            raise RoostooResponseError("balance response wallet must be an object")
        return wallet

    def pending_count(self) -> dict[str, Any]:
        headers, _, params = self._signed_request_parts({})
        return self._request_json("GET", "/v3/pending_count", headers=headers, params=params)

    def place_order(
        self,
        pair: str,
        side: str,
        quantity: str | int | float | Decimal,
        order_type: str = "MARKET",
        price: str | int | float | Decimal | None = None,
    ) -> dict[str, Any]:
        normalized_type = order_type.upper()
        if normalized_type not in {"MARKET", "LIMIT"}:
            raise ValueError("order_type must be MARKET or LIMIT")
        normalized_side = side.upper()
        if normalized_side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        payload: dict[str, Any] = {
            "pair": pair.upper(),
            "side": normalized_side,
            "type": normalized_type,
            "quantity": _decimal_string(quantity, name="quantity"),
        }
        if Decimal(payload["quantity"]) <= 0:
            raise ValueError("quantity must be positive")
        if normalized_type == "LIMIT":
            if price is None:
                raise ValueError("price is required for LIMIT orders")
            payload["price"] = _decimal_string(price, name="price")
            if Decimal(payload["price"]) <= 0:
                raise ValueError("price must be positive")
        headers, body, _ = self._signed_request_parts(payload)
        return self._require_success(
            self._request_json(
                "POST", "/v3/place_order", headers=headers, data=body, retryable=False
            ),
            operation="place order",
        )

    def query_order(
        self,
        *,
        order_id: int | None = None,
        pair: str | None = None,
        pending_only: bool | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {}
        if order_id is not None:
            payload["order_id"] = str(order_id)
        else:
            if pair:
                payload["pair"] = pair.upper()
            if pending_only is not None:
                payload["pending_only"] = "TRUE" if pending_only else "FALSE"
            if limit is not None:
                payload["limit"] = str(limit)
        headers, body, _ = self._signed_request_parts(payload)
        response = self._request_json(
            "POST", "/v3/query_order", headers=headers, data=body, retryable=True
        )
        if response.get("Success") is False:
            if "no order" in str(response.get("ErrMsg", "")).lower():
                return []
        self._require_success(response, operation="query order")
        orders = response.get("OrderMatched", [])
        if not isinstance(orders, list):
            raise RoostooResponseError("query-order response OrderMatched must be a list")
        if not all(isinstance(order, dict) for order in orders):
            raise RoostooResponseError("query-order response contains a non-object order")
        return orders

    def cancel_order(
        self,
        *,
        order_id: int | None = None,
        pair: str | None = None,
    ) -> list[Any]:
        if order_id is not None and pair is not None:
            raise ValueError("provide at most one of order_id and pair")
        payload: dict[str, Any] = {}
        if order_id is not None:
            payload["order_id"] = str(order_id)
        elif pair:
            payload["pair"] = pair.upper()
        headers, body, _ = self._signed_request_parts(payload)
        response = self._require_success(
            self._request_json(
                "POST", "/v3/cancel_order", headers=headers, data=body, retryable=False
            ),
            operation="cancel order",
        )
        canceled = response.get("CanceledList", [])
        if not isinstance(canceled, list):
            raise RoostooResponseError("cancel-order response CanceledList must be a list")
        return canceled

    # Signed v6 short endpoints documented in TARGET/API.md

    def open_short(
        self,
        pair: str,
        collateral: str | int | float | Decimal,
        *,
        order_type: str = "MARKET",
        price: str | int | float | Decimal | None = None,
    ) -> ShortOpenResult:
        normalized_type = order_type.upper()
        if normalized_type not in {"MARKET", "LIMIT"}:
            raise ValueError("order_type must be MARKET or LIMIT")
        collateral_text = _decimal_string(collateral, name="collateral")
        if Decimal(collateral_text) < 1:
            raise ValueError("collateral must be at least 1 USD")
        payload: dict[str, Any] = {"pair": pair.upper(), "collateral": collateral_text}
        if normalized_type == "LIMIT":
            if price is None:
                raise ValueError("price is required for LIMIT short opens")
            price_text = _decimal_string(price, name="price")
            if Decimal(price_text) <= 0:
                raise ValueError("price must be positive")
            payload.update({"order_type": "LIMIT", "price": price_text})
        headers, body, _ = self._signed_request_parts(payload)
        response = self._require_success(
            self._request_json(
                "POST", "/v6/short_open", headers=headers, data=body, retryable=False
            ),
            operation="open short",
        )
        return ShortOpenResult.from_payload(response)

    def close_short(
        self,
        pair: str,
        *,
        close_quantity: str | int | float | Decimal | None = None,
        close_percent: str | int | float | Decimal | None = None,
    ) -> ShortCloseResult:
        payload: dict[str, Any] = {"pair": pair.upper()}
        if close_quantity is not None:
            quantity_text = _decimal_string(close_quantity, name="close_quantity")
            if Decimal(quantity_text) <= 0:
                raise ValueError("close_quantity must be positive")
            payload["close_qty"] = quantity_text
        elif close_percent is not None:
            percent_text = _decimal_string(close_percent, name="close_percent")
            percent = Decimal(percent_text)
            if not 0 < percent <= 100:
                raise ValueError("close_percent must be greater than 0 and at most 100")
            payload["close_pct"] = percent_text
        headers, body, _ = self._signed_request_parts(payload)
        response = self._require_success(
            self._request_json(
                "POST", "/v6/short_close", headers=headers, data=body, retryable=False
            ),
            operation="close short",
        )
        return ShortCloseResult.from_payload(response)

    def short_positions(self) -> list[ShortPosition]:
        headers, _, params = self._signed_request_parts({})
        response = self._require_success(
            self._request_json("GET", "/v6/short_positions", headers=headers, params=params),
            operation="short positions",
        )
        positions = response.get("Positions")
        if not isinstance(positions, list):
            raise RoostooResponseError("short-positions response Positions must be a list")
        if not all(isinstance(position, dict) for position in positions):
            raise RoostooResponseError("short-positions response contains a non-object position")
        return [ShortPosition.from_payload(position) for position in positions]
