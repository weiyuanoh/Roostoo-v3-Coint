"""Explicit UTC-aligned 5-minute to 15-minute aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from stat_arb_bot.domain import Candle
from stat_arb_bot.domain.execution import utc_datetime
from stat_arb_bot.research_data.quality import RawQualityDiagnostics, prepare_raw_series

RAW_INTERVAL = "5m"
MODEL_INTERVAL = "15m"
RAW_STEP = timedelta(minutes=5)
MODEL_STEP = timedelta(minutes=15)


@dataclass(frozen=True, slots=True)
class AggregationDiagnostics:
    symbol: str
    raw_quality: RawQualityDiagnostics
    valid_model_bars: int
    incomplete_bucket_opens: tuple[datetime, ...]
    misaligned_raw_opens: tuple[datetime, ...]

    @property
    def excluded_model_bars(self) -> int:
        return len(set(self.incomplete_bucket_opens))


@dataclass(frozen=True, slots=True)
class AggregationResult:
    symbol: str
    bars: tuple[Candle, ...]
    diagnostics: AggregationDiagnostics


def model_bucket_open(value: datetime) -> datetime:
    normalized = utc_datetime(value, name="bar open")
    return normalized.replace(minute=(normalized.minute // 15) * 15, second=0, microsecond=0)


def _is_five_minute_aligned(value: datetime) -> bool:
    normalized = value.astimezone(timezone.utc)
    return normalized.minute % 5 == 0 and normalized.second == 0 and normalized.microsecond == 0


def aggregate_five_to_fifteen(
    candles: Iterable[Candle],
    *,
    symbol: str,
    as_of: datetime,
    max_age_intervals: int | None = None,
) -> AggregationResult:
    """Build only complete `[open, close)` UTC quarter-hour buckets."""

    normalized_as_of = utc_datetime(as_of, name="as_of")
    raw = prepare_raw_series(
        candles,
        symbol=symbol,
        interval=RAW_INTERVAL,
        as_of=normalized_as_of,
        max_age_intervals=max_age_intervals,
    )
    buckets: dict[datetime, dict[datetime, Candle]] = {}
    misaligned: list[datetime] = []
    for candle in raw.candles:
        open_time = candle.open_time.astimezone(timezone.utc)
        if not _is_five_minute_aligned(open_time):
            misaligned.append(open_time)
            continue
        bucket_open = model_bucket_open(open_time)
        buckets.setdefault(bucket_open, {})[open_time] = candle

    result: list[Candle] = []
    incomplete: list[datetime] = []
    for bucket_open, by_open in sorted(buckets.items()):
        bucket_close = bucket_open + MODEL_STEP
        expected_opens = tuple(bucket_open + index * RAW_STEP for index in range(3))
        constituents = tuple(by_open.get(open_time) for open_time in expected_opens)
        if (
            any(candle is None for candle in constituents)
            or set(by_open) != set(expected_opens)
            or bucket_close > normalized_as_of
        ):
            incomplete.append(bucket_open)
            continue
        complete = tuple(candle for candle in constituents if candle is not None)
        if any(
            candle.close_time > candle.open_time + RAW_STEP or candle.close_time <= candle.open_time
            for candle in complete
        ):
            incomplete.append(bucket_open)
            continue
        retrieved = [candle.retrieved_at for candle in complete if candle.retrieved_at is not None]
        result.append(
            Candle(
                open_time=bucket_open,
                close_time=bucket_close,
                symbol=symbol.strip().upper(),
                interval=MODEL_INTERVAL,
                open=complete[0].open,
                high=max(candle.high for candle in complete),
                low=min(candle.low for candle in complete),
                close=complete[-1].close,
                volume=sum(candle.volume for candle in complete),
                source="DERIVED_BINANCE_5M",
                retrieved_at=max(retrieved) if retrieved else None,
            )
        )
    diagnostics = AggregationDiagnostics(
        symbol=symbol.strip().upper(),
        raw_quality=raw.diagnostics,
        valid_model_bars=len(result),
        incomplete_bucket_opens=tuple(incomplete),
        misaligned_raw_opens=tuple(sorted(misaligned)),
    )
    return AggregationResult(symbol.strip().upper(), tuple(result), diagnostics)
