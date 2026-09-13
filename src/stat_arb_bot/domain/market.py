"""Normalized market-data types."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Any


def epoch_ms_to_utc(value: int | str) -> datetime:
    return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)


def utc_to_epoch_ms(value: datetime) -> int:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return int(value.astimezone(timezone.utc).timestamp() * 1000)


@dataclass(frozen=True, order=True)
class Candle:
    """One closed or forming OHLCV bar with timezone-aware UTC timestamps."""

    open_time: datetime
    symbol: str
    interval: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: datetime
    source: str | None = None
    retrieved_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.open_time.tzinfo is None or self.close_time.tzinfo is None:
            raise ValueError("candle timestamps must be timezone-aware")
        if self.close_time < self.open_time:
            raise ValueError("close_time must not precede open_time")
        prices = (self.open, self.high, self.low, self.close)
        if not all(isfinite(value) and value > 0 for value in prices):
            raise ValueError("OHLC prices must be positive and finite")
        if not isfinite(self.volume) or self.volume < 0:
            raise ValueError("volume must be non-negative and finite")
        if self.high < max(self.open, self.low, self.close):
            raise ValueError("high is inconsistent with OHLC values")
        if self.low > min(self.open, self.high, self.close):
            raise ValueError("low is inconsistent with OHLC values")
        source = self.source.strip().upper() if self.source else None
        retrieved_at = self.retrieved_at
        if retrieved_at is not None:
            if retrieved_at.tzinfo is None:
                raise ValueError("retrieved_at must be timezone-aware")
            retrieved_at = retrieved_at.astimezone(timezone.utc)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "retrieved_at", retrieved_at)

    def is_closed(self, as_of: datetime | None = None) -> bool:
        now = as_of or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        return self.close_time <= now.astimezone(timezone.utc)

    def as_record(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "open_time": self.open_time.astimezone(timezone.utc).isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "close_time": self.close_time.astimezone(timezone.utc).isoformat(),
            "source": self.source,
            "retrieved_at": self.retrieved_at.isoformat() if self.retrieved_at else None,
        }
