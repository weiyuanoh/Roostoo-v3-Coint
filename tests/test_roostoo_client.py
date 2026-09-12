from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import requests

from stat_arb_bot.exchange.roostoo.client import RoostooClient
from stat_arb_bot.exchange.roostoo.errors import (
    RoostooAuthenticationError,
    RoostooRejectedError,
    RoostooResponseError,
    RoostooTransportError,
)
from stat_arb_bot.exchange.roostoo.models import ShortPosition


class FakeResponse:
    def __init__(
        self,
        payload: Any,
        *,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.headers = dict(headers or {})

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self.payload


class FakeSession:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((method, url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def client(session: FakeSession, **kwargs: Any) -> RoostooClient:
    return RoostooClient(
        api_key="key",
        api_secret="secret",
        session=session,  # type: ignore[arg-type]
        clock_ms=lambda: 123456,
        sleeper=lambda _seconds: None,
        **kwargs,
    )


def test_open_short_market_uses_documented_v6_shape() -> None:
    session = FakeSession(
        [
            FakeResponse(
                {
                    "Success": True,
                    "ID": 412,
                    "Pair": "BTC/USD",
                    "OrderType": "MARKET",
                    "EntryPrice": 61840.25,
                    "ShortQty": 0.01617,
                    "Collateral": 999.95,
                    "OpenFee": 0.119994,
                    "Status": "OPEN",
                    "CreateTimestamp": 1757980800000,
                }
            )
        ]
    )

    result = client(session).open_short("btc/usd", 1000)

    method, url, kwargs = session.calls[0]
    assert (method, url) == ("POST", "https://mock-api.roostoo.com/v6/short_open")
    assert kwargs["data"] == "collateral=1000&pair=BTC/USD&timestamp=123456"
    assert kwargs["headers"]["RST-API-KEY"] == "key"
    assert "order_type" not in kwargs["data"]
    assert result.id == 412
    assert result.status == "OPEN"


def test_open_short_limit_includes_limit_fields() -> None:
    session = FakeSession(
        [
            FakeResponse(
                {
                    "Success": True,
                    "ID": 90271,
                    "Pair": "BTC/USD",
                    "OrderType": "LIMIT",
                    "EntryPrice": 63000,
                    "ShortQty": 0.015873,
                    "Collateral": 999.99,
                    "OpenFee": 0.079999,
                    "Status": "PENDING",
                    "CreateTimestamp": 1757980800000,
                }
            )
        ]
    )

    client(session).open_short("BTC/USD", "1000.00", order_type="limit", price="63000.0")

    body = session.calls[0][2]["data"]
    assert body == ("collateral=1000&order_type=LIMIT&pair=BTC/USD&price=63000&timestamp=123456")


def test_close_short_quantity_takes_precedence_and_parses_partial_close() -> None:
    session = FakeSession(
        [
            FakeResponse(
                {
                    "Success": True,
                    "ClosePrice": 60500.1,
                    "RealizedPNL": 10.8258,
                    "CloseFee": 0.0484,
                    "ReturnAmount": 510.7774,
                    "ClosedQty": 0.008085,
                    "FullyClosed": False,
                    "RemainingQty": 0.008085,
                    "RemainingCollateral": 499.98,
                }
            )
        ]
    )

    result = client(session).close_short("BTC/USD", close_quantity="0.008085", close_percent=50)

    body = session.calls[0][2]["data"]
    assert "close_qty=0.008085" in body
    assert "close_pct" not in body
    assert result.fully_closed is False
    assert result.remaining_quantity == pytest.approx(0.008085)


def test_short_positions_returns_typed_positions() -> None:
    session = FakeSession(
        [
            FakeResponse(
                {
                    "Success": True,
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
                            "PositionStatus": "OPEN",
                        }
                    ],
                }
            )
        ]
    )

    positions = client(session).short_positions()

    assert positions == [
        ShortPosition(
            id=412,
            pair="BTC/USD",
            entry_price=61840.25,
            short_quantity=0.01617,
            collateral=999.95,
            current_price=60500.1,
            unrealized_pnl=21.670225,
            unrealized_pnl_pct=0.021671,
            position_value=1021.620225,
            create_timestamp=1757980800000,
            status="OPEN",
        )
    ]


def test_signed_endpoint_requires_credentials_before_http() -> None:
    session = FakeSession([])
    unauthenticated = RoostooClient(api_key="", api_secret="", session=session)  # type: ignore[arg-type]

    with pytest.raises(RoostooAuthenticationError):
        unauthenticated.balance()
    assert session.calls == []


def test_v3_ticker_normalizes_pair_and_returns_data() -> None:
    session = FakeSession(
        [FakeResponse({"Success": True, "Data": {"BTC/USD": {"LastPrice": 100}}})]
    )

    result = client(session).ticker("btc/usd")

    assert result["BTC/USD"]["LastPrice"] == 100
    assert session.calls[0][2]["params"] == {"timestamp": "123456", "pair": "BTC/USD"}


def test_v3_balance_uses_signed_get_query() -> None:
    session = FakeSession([FakeResponse({"Success": True, "SpotWallet": {"USD": {"Free": 1000}}})])

    wallet = client(session).balance()

    method, url, kwargs = session.calls[0]
    assert (method, url) == ("GET", "https://mock-api.roostoo.com/v3/balance")
    assert kwargs["params"] == {"timestamp": "123456"}
    assert kwargs["headers"]["RST-API-KEY"] == "key"
    assert wallet["USD"]["Free"] == 1000


def test_v3_place_order_preserves_contract_and_does_not_retry() -> None:
    session = FakeSession([FakeResponse({"Success": True, "OrderDetail": {"OrderID": 7}})])

    result = client(session, max_attempts=3).place_order(
        "btc/usd", "buy", "0.01", order_type="limit", price="60000.00"
    )

    assert result["OrderDetail"]["OrderID"] == 7
    assert session.calls[0][2]["data"] == (
        "pair=BTC/USD&price=60000&quantity=0.01&side=BUY&timestamp=123456&type=LIMIT"
    )
    assert len(session.calls) == 1


def test_v3_cancel_returns_undocumented_list_without_inference() -> None:
    canceled = [{"OrderID": 90271, "Pair": "BTC/USD"}]
    session = FakeSession([FakeResponse({"Success": True, "CanceledList": canceled})])

    assert client(session).cancel_order(order_id=90271) == canceled
    assert session.calls[0][2]["data"] == "order_id=90271&timestamp=123456"


def test_get_retries_transport_error() -> None:
    session = FakeSession([requests.ConnectionError("temporary"), FakeResponse({"ServerTime": 10})])

    assert client(session, max_attempts=2).server_time() == 10
    assert len(session.calls) == 2


def test_read_only_query_post_can_retry() -> None:
    session = FakeSession(
        [
            requests.ConnectionError("temporary"),
            FakeResponse({"Success": True, "OrderMatched": []}),
        ]
    )

    assert client(session, max_attempts=2).query_order(pair="BTC/USD") == []
    assert len(session.calls) == 2


def test_mutating_post_is_never_retried() -> None:
    session = FakeSession([requests.ConnectionError("uncertain outcome")])

    with pytest.raises(RoostooTransportError):
        client(session, max_attempts=3).open_short("BTC/USD", 10)
    assert len(session.calls) == 1


def test_api_rejection_is_typed() -> None:
    session = FakeSession([FakeResponse({"Success": False, "ErrMsg": "insufficient balance"})])

    with pytest.raises(RoostooRejectedError, match="insufficient balance"):
        client(session).open_short("BTC/USD", 10)


def test_success_marker_is_required() -> None:
    session = FakeSession([FakeResponse({"Positions": []})])

    with pytest.raises(RoostooResponseError, match="Success=true"):
        client(session).short_positions()


def test_short_response_field_types_are_checked() -> None:
    session = FakeSession(
        [
            FakeResponse(
                {
                    "Success": True,
                    "ClosePrice": 10,
                    "RealizedPNL": 1,
                    "CloseFee": 0.1,
                    "ReturnAmount": 10.9,
                    "ClosedQty": 1,
                    "FullyClosed": "false",
                }
            )
        ]
    )

    with pytest.raises(RoostooResponseError, match="must be a boolean"):
        client(session).close_short("BTC/USD")


@pytest.mark.parametrize("collateral", [0, "0.99", -1])
def test_short_collateral_minimum_is_enforced(collateral: Any) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        client(FakeSession([])).open_short("BTC/USD", collateral)


@pytest.mark.parametrize("percent", [0, -1, 100.01])
def test_close_percent_range_is_enforced(percent: Any) -> None:
    with pytest.raises(ValueError, match="at most 100"):
        client(FakeSession([])).close_short("BTC/USD", close_percent=percent)
