"""Minimal strategy boundary for historical infrastructure tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from stat_arb_bot.accounting import PortfolioSnapshot
from stat_arb_bot.backtest.clock import BarClock
from stat_arb_bot.backtest.history import MarketHistoryView
from stat_arb_bot.backtest.orders import OrderIntent


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """The bounded information set available after the current bar closes."""

    clock: BarClock
    history: MarketHistoryView
    portfolio: PortfolioSnapshot

    def __post_init__(self) -> None:
        if self.history.information_timestamp != self.clock.observable_at:
            raise ValueError("strategy history and information clock disagree")
        if self.history.max_observation_timestamp > self.clock.observable_at:
            raise ValueError("strategy history contains future information")


class HistoricalStrategy(Protocol):
    def reset(self) -> None: ...

    def on_bar(self, context: StrategyContext) -> Sequence[OrderIntent]: ...
