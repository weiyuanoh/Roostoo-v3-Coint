"""Bounded Binance bootstrap and overlap-safe incremental retrieval."""

from __future__ import annotations

from datetime import datetime
from typing import Mapping, Sequence

from stat_arb_bot.domain import Candle, utc_to_epoch_ms
from stat_arb_bot.domain.execution import utc_datetime
from stat_arb_bot.market_data import BinanceData
from stat_arb_bot.research_data.universe import ResearchUniverse


class ResearchHistoryLoader:
    def __init__(self, client: BinanceData) -> None:
        self.client = client

    def bootstrap(
        self,
        universe: ResearchUniverse,
        *,
        start: datetime,
        end: datetime,
        page_limit: int = 1000,
        page_delay: float = 0.1,
    ) -> dict[str, tuple[Candle, ...]]:
        """Fetch a start-inclusive, completed-close-inclusive interval."""

        normalized_start = utc_datetime(start, name="start")
        normalized_end = utc_datetime(end, name="end")
        if normalized_start >= normalized_end:
            raise ValueError("history start must precede end")
        result: dict[str, tuple[Candle, ...]] = {}
        for asset in universe.assets:
            pair = universe.pair(asset)
            fetched = self.client.fetch_klines_paginated(
                pair,
                interval="5m",
                start_time=utc_to_epoch_ms(normalized_start),
                end_time=utc_to_epoch_ms(normalized_end),
                limit=page_limit,
                sleep_seconds=page_delay,
                closed_only=True,
            )
            result[asset] = tuple(
                candle
                for candle in fetched
                if candle.open_time >= normalized_start and candle.close_time <= normalized_end
            )
        return result

    def incremental_refresh(
        self,
        universe: ResearchUniverse,
        existing: Mapping[str, Sequence[Candle]],
        *,
        end: datetime,
        start_if_empty: datetime,
        page_limit: int = 1000,
        page_delay: float = 0.1,
    ) -> dict[str, tuple[Candle, ...]]:
        """Fetch from each latest stored open, deliberately overlapping one row."""

        normalized_end = utc_datetime(end, name="end")
        empty_start = utc_datetime(start_if_empty, name="start_if_empty")
        result: dict[str, tuple[Candle, ...]] = {}
        for asset in universe.assets:
            rows = tuple(existing.get(asset, ()))
            start = max((candle.open_time for candle in rows), default=empty_start)
            pair = universe.pair(asset)
            fetched = self.client.fetch_klines_paginated(
                pair,
                interval="5m",
                start_time=utc_to_epoch_ms(start),
                end_time=utc_to_epoch_ms(normalized_end),
                limit=page_limit,
                sleep_seconds=page_delay,
                closed_only=True,
            )
            result[asset] = tuple(
                candle
                for candle in fetched
                if candle.open_time >= start and candle.close_time <= normalized_end
            )
        return result
