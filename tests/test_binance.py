from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
import requests

from stat_arb_bot.market_data.binance import BinanceData, BinanceDataError, UnknownSymbolError


def kline(open_ms: int, close_ms: int, close: str = "101") -> list[Any]:
    return [open_ms, "100", "102", "99", close, "10", close_ms]


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self.payload


class FakeSession:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def client(session: FakeSession, **kwargs: Any) -> BinanceData:
    clock = kwargs.pop("clock", lambda: datetime.fromtimestamp(7_300 / 1000, tz=timezone.utc))
    return BinanceData(
        base_urls=("https://one.test", "https://two.test"),
        max_attempts=1,
        session=session,  # type: ignore[arg-type]
        sleeper=lambda _seconds: None,
        clock=clock,
        **kwargs,
    )


def test_fetch_normalizes_and_excludes_forming_candle() -> None:
    session = FakeSession([FakeResponse([kline(0, 3_599), kline(3_600, 7_599)])])

    candles = client(session).fetch_klines("btc/usd", interval="1h")

    assert len(candles) == 1
    assert candles[0].symbol == "BTC/USD"
    assert candles[0].open_time.tzinfo == timezone.utc
    assert session.calls[0][1]["params"]["symbol"] == "BTCUSDT"


def test_endpoint_fallback_is_used() -> None:
    session = FakeSession([FakeResponse({}, status_code=451), FakeResponse([kline(0, 3_599)])])

    candles = client(session).fetch_klines("BTC/USD")

    assert len(candles) == 1
    assert [call[0] for call in session.calls] == [
        "https://one.test/api/v3/klines",
        "https://two.test/api/v3/klines",
    ]


def test_paginated_fetch_deduplicates_boundary_rows() -> None:
    session = FakeSession(
        [
            FakeResponse([kline(0, 3_599), kline(3_600, 7_199)]),
            FakeResponse([kline(3_600, 7_199), kline(7_200, 10_799)]),
            FakeResponse([]),
        ]
    )
    data = client(
        session,
        clock=lambda: datetime.fromtimestamp(20_000 / 1000, tz=timezone.utc),
    )

    candles = data.fetch_klines_paginated(
        "BTC/USD", start_time=0, end_time=10_800, limit=2, sleep_seconds=0
    )

    assert [int(candle.open_time.timestamp() * 1000) for candle in candles] == [0, 3_600, 7_200]


def test_unknown_symbol_is_explicit() -> None:
    with pytest.raises(UnknownSymbolError, match="mapping"):
        client(FakeSession([])).fetch_klines("NOPE/USD")


def test_explicit_research_quote_maps_to_matching_binance_symbol() -> None:
    assert BinanceData.binance_symbol("btc/usdt") == "BTCUSDT"
    assert BinanceData.binance_symbol("eth/usdc") == "ETHUSDC"


def test_invalid_payload_is_rejected() -> None:
    with pytest.raises(BinanceDataError, match="non-list"):
        client(FakeSession([FakeResponse({"code": -1}), FakeResponse({"code": -1})])).fetch_klines(
            "BTC/USD"
        )
