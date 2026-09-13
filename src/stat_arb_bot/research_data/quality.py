"""Raw five-minute quality diagnostics and deterministic overlap handling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite
from typing import Iterable

from stat_arb_bot.domain import Candle
from stat_arb_bot.domain.execution import utc_datetime
from stat_arb_bot.market_data.validation import inspect_candles, interval_delta


class RawDataConflictError(ValueError):
    """Raised when overlapping observations disagree on market data."""


@dataclass(frozen=True, slots=True)
class ContinuityGap:
    previous_open: datetime
    next_open: datetime
    missing_intervals: int


@dataclass(frozen=True, slots=True)
class RawQualityDiagnostics:
    symbol: str
    received_rows: int
    accepted_rows: int
    exact_duplicate_open_times: tuple[datetime, ...]
    missing_open_times: tuple[datetime, ...]
    out_of_order: bool
    non_monotonic: bool
    forming_open_times: tuple[datetime, ...]
    stale: bool
    unexpected_symbols: tuple[str, ...]
    unexpected_intervals: tuple[str, ...]
    invalid_open_times: tuple[datetime, ...]
    large_gaps: tuple[ContinuityGap, ...]

    @property
    def valid_for_storage(self) -> bool:
        return not (
            self.forming_open_times
            or self.unexpected_symbols
            or self.unexpected_intervals
            or self.invalid_open_times
        )


@dataclass(frozen=True, slots=True)
class RawSeries:
    symbol: str
    interval: str
    candles: tuple[Candle, ...]
    diagnostics: RawQualityDiagnostics


def _market_payload(candle: Candle) -> tuple[object, ...]:
    return (
        candle.symbol.strip().upper(),
        candle.interval,
        candle.open_time.astimezone(timezone.utc),
        candle.close_time.astimezone(timezone.utc),
        candle.open,
        candle.high,
        candle.low,
        candle.close,
        candle.volume,
    )


def _canonical_duplicate(candles: list[Candle]) -> Candle:
    far_future = datetime.max.replace(tzinfo=timezone.utc)
    return min(
        candles,
        key=lambda candle: (
            candle.source is None,
            candle.retrieved_at or far_future,
            candle.source or "",
        ),
    )


def prepare_raw_series(
    candles: Iterable[Candle],
    *,
    symbol: str,
    as_of: datetime,
    interval: str = "5m",
    max_age_intervals: int | None = None,
    large_gap_threshold_intervals: int = 3,
) -> RawSeries:
    """Exclude unusable observations and retain explicit diagnostics.

    Exact repeated observations are idempotent. A same-symbol/open-time row
    with different market content raises instead of choosing one silently.
    """

    expected_symbol = symbol.strip().upper()
    if not expected_symbol:
        raise ValueError("symbol must not be empty")
    if large_gap_threshold_intervals < 1:
        raise ValueError("large_gap_threshold_intervals must be positive")
    normalized_as_of = utc_datetime(as_of, name="as_of")
    rows = tuple(candles)
    expected_rows = [row for row in rows if row.symbol.strip().upper() == expected_symbol]
    open_times = [row.open_time.astimezone(timezone.utc) for row in expected_rows]
    out_of_order = any(right < left for left, right in zip(open_times, open_times[1:]))
    non_monotonic = any(right <= left for left, right in zip(open_times, open_times[1:]))
    unexpected_symbols = tuple(
        sorted({row.symbol for row in rows if row.symbol.strip().upper() != expected_symbol})
    )
    unexpected_intervals = tuple(
        sorted({row.interval for row in expected_rows if row.interval != interval})
    )
    candidates = [row for row in expected_rows if row.interval == interval]
    step = interval_delta(interval)
    invalid = tuple(
        sorted(
            row.open_time
            for row in candidates
            if not (
                all(
                    isfinite(value) and value > 0
                    for value in (row.open, row.high, row.low, row.close)
                )
                and isfinite(row.volume)
                and row.volume >= 0
                and row.high >= max(row.open, row.low, row.close)
                and row.low <= min(row.open, row.high, row.close)
                and row.close_time
                in {
                    row.open_time + step,
                    row.open_time + step - timedelta(milliseconds=1),
                }
            )
        )
    )
    invalid_set = set(invalid)
    grouped: dict[datetime, list[Candle]] = {}
    for row in candidates:
        if row.open_time in invalid_set:
            continue
        grouped.setdefault(row.open_time.astimezone(timezone.utc), []).append(row)

    exact_duplicates: list[datetime] = []
    deduplicated: list[Candle] = []
    for open_time, group in sorted(grouped.items()):
        payloads = {_market_payload(row) for row in group}
        if len(payloads) > 1:
            raise RawDataConflictError(
                f"conflicting observations for {expected_symbol} at {open_time.isoformat()}"
            )
        if len(group) > 1:
            exact_duplicates.append(open_time)
        deduplicated.append(_canonical_duplicate(group))

    forming = tuple(row.open_time for row in deduplicated if not row.is_closed(normalized_as_of))
    forming_set = set(forming)
    accepted = tuple(row for row in deduplicated if row.open_time not in forming_set)
    base_report = inspect_candles(
        accepted,
        interval=interval,
        expected_symbol=expected_symbol,
        as_of=normalized_as_of,
        max_age_intervals=max_age_intervals,
    )
    large_gaps: list[ContinuityGap] = []
    for left, right in zip(accepted, accepted[1:]):
        missing_count = int((right.open_time - left.open_time) / step) - 1
        if missing_count >= large_gap_threshold_intervals:
            large_gaps.append(
                ContinuityGap(
                    previous_open=left.open_time,
                    next_open=right.open_time,
                    missing_intervals=missing_count,
                )
            )
    diagnostics = RawQualityDiagnostics(
        symbol=expected_symbol,
        received_rows=len(rows),
        accepted_rows=len(accepted),
        exact_duplicate_open_times=tuple(exact_duplicates),
        missing_open_times=base_report.missing_open_times,
        out_of_order=out_of_order,
        non_monotonic=non_monotonic,
        forming_open_times=forming,
        stale=base_report.stale,
        unexpected_symbols=unexpected_symbols,
        unexpected_intervals=unexpected_intervals,
        invalid_open_times=invalid,
        large_gaps=tuple(large_gaps),
    )
    return RawSeries(expected_symbol, interval, accepted, diagnostics)


def merge_raw_series(
    existing: Iterable[Candle],
    incoming: Iterable[Candle],
    *,
    symbol: str,
    as_of: datetime,
    interval: str = "5m",
    max_age_intervals: int | None = None,
) -> RawSeries:
    """Merge an overlapping refresh under the canonical conflict policy."""

    return prepare_raw_series(
        (*tuple(existing), *tuple(incoming)),
        symbol=symbol,
        interval=interval,
        as_of=as_of,
        max_age_intervals=max_age_intervals,
    )
