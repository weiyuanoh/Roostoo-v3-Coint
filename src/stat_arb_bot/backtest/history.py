"""Synchronized, bounded historical market-data views."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from stat_arb_bot.backtest.clock import BarClock
from stat_arb_bot.domain.execution import utc_datetime
from stat_arb_bot.domain.market import Candle


@dataclass(frozen=True, slots=True)
class HistoricalFrame:
    """One synchronized set of completed bars and its information clock."""

    clock: BarClock
    bars: Mapping[str, Candle]

    def __post_init__(self) -> None:
        bars = {symbol.strip().upper(): bar for symbol, bar in self.bars.items()}
        if not bars:
            raise ValueError("historical frame must contain at least one bar")
        if any(
            bar.open_time.astimezone(timezone.utc) != self.clock.bar_open for bar in bars.values()
        ):
            raise ValueError("all frame bars must share the clock's open timestamp")
        if any(
            bar.close_time.astimezone(timezone.utc) != self.clock.bar_close for bar in bars.values()
        ):
            raise ValueError("all frame bars must share the clock's close timestamp")
        object.__setattr__(self, "bars", MappingProxyType(bars))


@dataclass(frozen=True, slots=True)
class PanelBuildReport:
    excluded_forming_bars: int
    skipped_unsynchronized_timestamps: tuple[datetime, ...]


class MarketHistoryView:
    """Read-only history whose final frame is the complete information set."""

    __slots__ = ("_frames", "_required_symbols")

    def __init__(
        self,
        frames: Sequence[HistoricalFrame],
        required_symbols: Sequence[str],
    ) -> None:
        if not frames:
            raise ValueError("history must contain at least one frame")
        self._frames = tuple(frames)
        self._required_symbols = tuple(symbol.strip().upper() for symbol in required_symbols)

    @property
    def frames(self) -> tuple[HistoricalFrame, ...]:
        return self._frames

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._required_symbols

    @property
    def information_timestamp(self) -> datetime:
        return self._frames[-1].clock.observable_at

    @property
    def max_observation_timestamp(self) -> datetime:
        return max(bar.close_time for bar in self._frames[-1].bars.values())

    @property
    def latest_frame(self) -> HistoricalFrame:
        return self._frames[-1]

    def latest(self, symbol: str) -> Candle:
        return self._frames[-1].bars[symbol.strip().upper()]

    def window(self, symbol: str, length: int) -> tuple[Candle, ...]:
        if length <= 0:
            raise ValueError("window length must be positive")
        normalized = symbol.strip().upper()
        return tuple(frame.bars[normalized] for frame in self._frames[-length:])


class HistoricalPanel:
    """Build synchronized frames without forward-filling missing assets."""

    def __init__(
        self,
        candles: Iterable[Candle],
        *,
        required_symbols: Sequence[str],
        as_of: datetime,
    ) -> None:
        required = tuple(dict.fromkeys(symbol.strip().upper() for symbol in required_symbols))
        if not required or any(not symbol for symbol in required):
            raise ValueError("required_symbols must contain non-empty symbols")
        normalized_as_of = utc_datetime(as_of, name="as_of")
        grouped: dict[datetime, dict[str, Candle]] = {}
        excluded_forming = 0
        intervals: set[str] = set()
        for candle in candles:
            symbol = candle.symbol.strip().upper()
            if symbol not in required:
                continue
            if not candle.is_closed(normalized_as_of):
                excluded_forming += 1
                continue
            open_time = candle.open_time.astimezone(timezone.utc)
            bucket = grouped.setdefault(open_time, {})
            if symbol in bucket:
                raise ValueError(f"duplicate bar for {symbol} at {open_time.isoformat()}")
            bucket[symbol] = candle
            intervals.add(candle.interval)
        if len(intervals) > 1:
            raise ValueError("all historical panel bars must use one interval")

        complete: list[tuple[datetime, datetime, Mapping[str, Candle]]] = []
        skipped: list[datetime] = []
        for open_time, bars in sorted(grouped.items()):
            if set(bars) != set(required):
                skipped.append(open_time)
                continue
            close_times = {bar.close_time.astimezone(timezone.utc) for bar in bars.values()}
            if len(close_times) != 1:
                skipped.append(open_time)
                continue
            complete.append((open_time, close_times.pop(), bars))

        frames: list[HistoricalFrame] = []
        for index, (open_time, close_time, bars) in enumerate(complete):
            if index + 1 < len(complete):
                next_open = complete[index + 1][0]
                first_executable_at = (
                    next_open if next_open > close_time else close_time + timedelta(microseconds=1)
                )
            else:
                first_executable_at = close_time + timedelta(microseconds=1)
            clock = BarClock(
                bar_open=open_time,
                bar_close=close_time,
                observable_at=close_time,
                first_executable_at=first_executable_at,
            )
            frames.append(HistoricalFrame(clock=clock, bars=bars))

        self.required_symbols = required
        self.frames = tuple(frames)
        self.report = PanelBuildReport(
            excluded_forming_bars=excluded_forming,
            skipped_unsynchronized_timestamps=tuple(skipped),
        )

    def history_through(self, index: int) -> MarketHistoryView:
        if index < 0 or index >= len(self.frames):
            raise IndexError("history index out of range")
        return MarketHistoryView(self.frames[: index + 1], self.required_symbols)
