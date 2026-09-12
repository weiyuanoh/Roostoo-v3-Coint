"""Candle continuity, closure, ordering, and staleness checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from collections import Counter

from stat_arb_bot.domain.market import Candle


_FIXED_INTERVALS = {
    "1s": timedelta(seconds=1),
    "1m": timedelta(minutes=1),
    "3m": timedelta(minutes=3),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "2h": timedelta(hours=2),
    "4h": timedelta(hours=4),
    "6h": timedelta(hours=6),
    "8h": timedelta(hours=8),
    "12h": timedelta(hours=12),
    "1d": timedelta(days=1),
    "3d": timedelta(days=3),
    "1w": timedelta(weeks=1),
}


class CandleQualityError(ValueError):
    """Raised when a candle series violates required quality constraints."""


@dataclass(frozen=True)
class CandleQualityReport:
    candles: int
    duplicate_open_times: tuple[datetime, ...]
    missing_open_times: tuple[datetime, ...]
    out_of_order: bool
    forming_open_times: tuple[datetime, ...]
    stale: bool
    unexpected_symbols: tuple[str, ...]
    unexpected_intervals: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return self.candles > 0 and not (
            self.duplicate_open_times
            or self.missing_open_times
            or self.out_of_order
            or self.forming_open_times
            or self.stale
            or self.unexpected_symbols
            or self.unexpected_intervals
        )


def interval_delta(interval: str) -> timedelta:
    """Return the fixed duration of a supported Binance interval."""

    try:
        return _FIXED_INTERVALS[interval]
    except KeyError as exc:
        if interval == "1M":
            raise CandleQualityError("calendar-month interval 1M is not fixed-duration") from exc
        raise CandleQualityError(
            f"unsupported Binance interval for validation: {interval}"
        ) from exc


def inspect_candles(
    candles: list[Candle] | tuple[Candle, ...],
    *,
    interval: str,
    expected_symbol: str | None = None,
    as_of: datetime | None = None,
    max_age_intervals: int | None = None,
) -> CandleQualityReport:
    """Inspect a single-symbol candle series without mutating it."""

    now = as_of or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    now = now.astimezone(timezone.utc)
    step = interval_delta(interval)
    if max_age_intervals is not None and max_age_intervals < 0:
        raise ValueError("max_age_intervals must be non-negative")
    open_times = [candle.open_time for candle in candles]
    counts = Counter(open_times)
    duplicates = tuple(sorted(value for value, count in counts.items() if count > 1))
    out_of_order = any(right <= left for left, right in zip(open_times, open_times[1:]))
    unique_sorted = sorted(set(open_times))
    missing: list[datetime] = []
    for left, right in zip(unique_sorted, unique_sorted[1:]):
        expected = left + step
        while expected < right:
            missing.append(expected)
            expected += step
    forming = tuple(candle.open_time for candle in candles if not candle.is_closed(now))
    stale = False
    if candles and max_age_intervals is not None:
        latest_close = max(candle.close_time for candle in candles)
        stale = latest_close < now - step * max_age_intervals
    unexpected_symbols = tuple(
        sorted(
            {
                candle.symbol
                for candle in candles
                if expected_symbol is not None and candle.symbol != expected_symbol.upper()
            }
        )
    )
    unexpected_intervals = tuple(
        sorted({candle.interval for candle in candles if candle.interval != interval})
    )
    return CandleQualityReport(
        candles=len(candles),
        duplicate_open_times=duplicates,
        missing_open_times=tuple(missing),
        out_of_order=out_of_order,
        forming_open_times=forming,
        stale=stale,
        unexpected_symbols=unexpected_symbols,
        unexpected_intervals=unexpected_intervals,
    )


def require_candle_quality(
    candles: list[Candle] | tuple[Candle, ...],
    *,
    interval: str,
    expected_symbol: str | None = None,
    as_of: datetime | None = None,
    max_age_intervals: int | None = None,
) -> CandleQualityReport:
    report = inspect_candles(
        candles,
        interval=interval,
        expected_symbol=expected_symbol,
        as_of=as_of,
        max_age_intervals=max_age_intervals,
    )
    problems = []
    if report.candles == 0:
        problems.append("series is empty")
    if report.duplicate_open_times:
        problems.append(f"{len(report.duplicate_open_times)} duplicate timestamps")
    if report.missing_open_times:
        problems.append(f"{len(report.missing_open_times)} missing bars")
    if report.out_of_order:
        problems.append("bars are out of order")
    if report.forming_open_times:
        problems.append(f"{len(report.forming_open_times)} bars are still forming")
    if report.stale:
        problems.append("latest bar is stale")
    if report.unexpected_symbols:
        problems.append("series contains unexpected symbols")
    if report.unexpected_intervals:
        problems.append("series contains unexpected intervals")
    if problems:
        raise CandleQualityError("; ".join(problems))
    return report
