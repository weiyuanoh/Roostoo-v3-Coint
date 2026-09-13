"""Binance public spot-kline client."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import requests

from stat_arb_bot.domain.market import Candle, epoch_ms_to_utc
from stat_arb_bot.market_data.symbols import binance_symbol_for_pair
from stat_arb_bot.observability.logging import get_logger

log = get_logger("market_data.binance")


class BinanceDataError(RuntimeError):
    """Raised when Binance data cannot be fetched or parsed."""


class UnknownSymbolError(BinanceDataError):
    """Raised when no Binance symbol mapping exists for a Roostoo pair."""


class BinanceData:
    """Fetch Binance public spot candles with fallback endpoints and safe retries."""

    def __init__(
        self,
        *,
        base_urls: list[str] | tuple[str, ...] | None = None,
        timeout: float = 10.0,
        max_attempts: int = 3,
        backoff_seconds: float = 0.25,
        session: requests.Session | None = None,
        clock: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds must be non-negative")
        urls = base_urls or (
            "https://data-api.binance.vision",
            "https://api.binance.com",
        )
        self.base_urls = tuple(dict.fromkeys(url.rstrip("/") for url in urls if url.strip()))
        if not self.base_urls:
            raise ValueError("at least one Binance base URL is required")
        self.timeout = float(timeout)
        self.max_attempts = int(max_attempts)
        self.backoff_seconds = float(backoff_seconds)
        self.session = session or requests.Session()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.sleeper = sleeper
        self.candles: dict[str, list[Candle]] = {}

    @staticmethod
    def binance_symbol(pair: str) -> str:
        normalized = pair.upper()
        try:
            return binance_symbol_for_pair(normalized)
        except KeyError as exc:
            raise UnknownSymbolError(f"no Binance symbol mapping for {normalized}") from exc

    def fetch_klines(
        self,
        pair: str,
        interval: str = "1h",
        limit: int = 1000,
        start_time: int | None = None,
        end_time: int | None = None,
        *,
        closed_only: bool = True,
    ) -> list[Candle]:
        """Fetch normalized candles for a Roostoo pair such as ``BTC/USD``."""

        if limit <= 0:
            raise ValueError("limit must be positive")
        if start_time is not None and end_time is not None and start_time >= end_time:
            return []
        normalized_pair = pair.upper()
        params: dict[str, Any] = {
            "symbol": self.binance_symbol(normalized_pair),
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)

        payload = self._fetch_payload(params, pair=normalized_pair)
        retrieved_at = self.clock()
        if retrieved_at.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        retrieved_at = retrieved_at.astimezone(timezone.utc)
        candles = [
            self._parse_kline(row, normalized_pair, interval, retrieved_at=retrieved_at)
            for row in payload
        ]
        if closed_only:
            candles = [candle for candle in candles if candle.is_closed(retrieved_at)]
        return candles

    def _fetch_payload(self, params: dict[str, Any], *, pair: str) -> list[list[Any]]:
        last_error: Exception | None = None
        for base_url in self.base_urls:
            for attempt in range(self.max_attempts):
                try:
                    response = self.session.get(
                        f"{base_url}/api/v3/klines",
                        params=params,
                        timeout=self.timeout,
                    )
                    status = int(getattr(response, "status_code", 200))
                    if status == 429 and attempt + 1 < self.max_attempts:
                        self.sleeper(self.backoff_seconds * (2**attempt))
                        continue
                    response.raise_for_status()
                    try:
                        payload = response.json()
                    except (TypeError, ValueError) as exc:
                        raise BinanceDataError(
                            f"Binance klines for {pair} returned invalid JSON"
                        ) from exc
                    if not isinstance(payload, list):
                        raise BinanceDataError(
                            f"Binance klines for {pair} returned a non-list payload"
                        )
                    return payload
                except BinanceDataError as exc:
                    last_error = exc
                    break
                except requests.RequestException as exc:
                    last_error = exc
                    if attempt + 1 < self.max_attempts:
                        self.sleeper(self.backoff_seconds * (2**attempt))
            log.warning("Binance klines failed via %s for %s: %s", base_url, pair, last_error)
        raise BinanceDataError(f"Binance klines failed for {pair}: {last_error}") from last_error

    @staticmethod
    def _parse_kline(
        row: Any,
        pair: str,
        interval: str,
        *,
        retrieved_at: datetime | None = None,
    ) -> Candle:
        if not isinstance(row, list | tuple) or len(row) < 7:
            raise BinanceDataError(f"invalid Binance kline row for {pair}: {row!r}")
        try:
            return Candle(
                open_time=epoch_ms_to_utc(row[0]),
                symbol=pair,
                interval=interval,
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                close_time=epoch_ms_to_utc(row[6]),
                source="BINANCE",
                retrieved_at=retrieved_at,
            )
        except (TypeError, ValueError) as exc:
            raise BinanceDataError(f"invalid Binance kline row for {pair}: {row!r}") from exc

    def fetch_klines_paginated(
        self,
        pair: str,
        interval: str = "1h",
        *,
        start_time: int,
        end_time: int,
        limit: int = 1000,
        sleep_seconds: float = 0.1,
        closed_only: bool = True,
    ) -> list[Candle]:
        if start_time >= end_time:
            return []
        if sleep_seconds < 0:
            raise ValueError("sleep_seconds must be non-negative")
        cursor = int(start_time)
        by_open_time: dict[datetime, Candle] = {}
        while cursor < end_time:
            page = self.fetch_klines(
                pair,
                interval=interval,
                limit=limit,
                start_time=cursor,
                end_time=end_time,
                closed_only=closed_only,
            )
            if not page:
                break
            for candle in page:
                if int(candle.open_time.timestamp() * 1000) < end_time:
                    by_open_time[candle.open_time] = candle
            next_cursor = int(page[-1].open_time.timestamp() * 1000) + 1
            if next_cursor <= cursor:
                raise BinanceDataError("Binance pagination did not advance")
            cursor = next_cursor
            if len(page) < min(limit, 1000):
                break
            if sleep_seconds:
                self.sleeper(sleep_seconds)
        return [by_open_time[key] for key in sorted(by_open_time)]

    def load_history(
        self,
        pairs: list[str] | tuple[str, ...],
        *,
        interval: str = "1h",
        limit: int = 1000,
    ) -> int:
        loaded = 0
        for pair in pairs:
            try:
                candles = self.fetch_klines(pair, interval=interval, limit=limit)
            except BinanceDataError:
                log.exception("Failed to load Binance history for %s", pair)
                continue
            if candles:
                self.candles[pair.upper()] = candles
                loaded += 1
        return loaded

    def update_latest(self, pairs: list[str] | tuple[str, ...], *, interval: str = "1h") -> None:
        for pair in pairs:
            normalized = pair.upper()
            latest = self.fetch_klines(normalized, interval=interval, limit=3)
            existing = {candle.open_time: candle for candle in self.candles.get(normalized, [])}
            existing.update({candle.open_time: candle for candle in latest})
            self.candles[normalized] = [existing[key] for key in sorted(existing)][-2000:]
