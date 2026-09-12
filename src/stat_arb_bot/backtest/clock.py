"""Explicit information and execution clocks for historical bars."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from stat_arb_bot.domain.execution import utc_datetime


@dataclass(frozen=True, slots=True)
class BarClock:
    """The four distinct times attached to a completed model bar.

    ``observable_at`` is when all OHLCV values may enter the strategy's
    information set. Orders produced then cannot execute before
    ``first_executable_at``.
    """

    bar_open: datetime
    bar_close: datetime
    observable_at: datetime
    first_executable_at: datetime

    def __post_init__(self) -> None:
        bar_open = utc_datetime(self.bar_open, name="bar_open")
        bar_close = utc_datetime(self.bar_close, name="bar_close")
        observable_at = utc_datetime(self.observable_at, name="observable_at")
        first_executable_at = utc_datetime(
            self.first_executable_at,
            name="first_executable_at",
        )
        if bar_close < bar_open:
            raise ValueError("bar_close must not precede bar_open")
        if observable_at < bar_close:
            raise ValueError("a bar cannot be observable before it closes")
        if first_executable_at <= observable_at:
            raise ValueError("first executable time must be after the information time")
        object.__setattr__(self, "bar_open", bar_open)
        object.__setattr__(self, "bar_close", bar_close)
        object.__setattr__(self, "observable_at", observable_at)
        object.__setattr__(self, "first_executable_at", first_executable_at)
