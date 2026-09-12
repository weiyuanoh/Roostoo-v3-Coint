from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from stat_arb_bot.domain import Candle


@pytest.fixture
def candle_factory():
    def make(
        offset_hours: int = 0,
        *,
        symbol: str = "BTC/USD",
        interval: str = "1h",
        base: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc),
    ) -> Candle:
        open_time = base + timedelta(hours=offset_hours)
        return Candle(
            open_time=open_time,
            symbol=symbol,
            interval=interval,
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
            volume=10.0,
            close_time=open_time + timedelta(hours=1) - timedelta(milliseconds=1),
        )

    return make
