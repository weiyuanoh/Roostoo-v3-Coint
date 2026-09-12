"""Configurable scheduled short-financing models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence

from stat_arb_bot.accounting import PortfolioSnapshot
from stat_arb_bot.domain.execution import (
    ZERO,
    DecimalLike,
    FinancingEntry,
    decimal_value,
    utc_datetime,
)


class FinancingModel(Protocol):
    def entries_between(
        self,
        previous_information_time: datetime | None,
        current_information_time: datetime,
        portfolio: PortfolioSnapshot,
    ) -> Sequence[FinancingEntry]: ...


@dataclass(frozen=True, slots=True)
class ZeroFinancingModel:
    def entries_between(
        self,
        previous_information_time: datetime | None,
        current_information_time: datetime,
        portfolio: PortfolioSnapshot,
    ) -> tuple[FinancingEntry, ...]:
        del previous_information_time, current_information_time, portfolio
        return ()


@dataclass(frozen=True, slots=True)
class FinancingRateEvent:
    """A rate applied once to marked absolute short notional.

    Positive rates are costs and produce negative FinancingEntry amounts;
    negative rates are credits. Rates are configuration, not Roostoo defaults.
    """

    timestamp: datetime
    rate: DecimalLike
    symbol: str | None = None
    category: str = "short_financing"
    reference: str | None = None
    trade_group_id: str | None = None

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper() if self.symbol else None
        category = self.category.strip()
        if not category:
            raise ValueError("category must not be empty")
        object.__setattr__(self, "timestamp", utc_datetime(self.timestamp))
        object.__setattr__(self, "rate", decimal_value(self.rate, name="financing rate"))
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "category", category)


class ScheduledShortFinancingModel:
    """Stateless replay of configured periodic short-notional charges."""

    def __init__(self, events: Sequence[FinancingRateEvent]) -> None:
        self.events = tuple(sorted(events, key=lambda event: event.timestamp))

    def entries_between(
        self,
        previous_information_time: datetime | None,
        current_information_time: datetime,
        portfolio: PortfolioSnapshot,
    ) -> tuple[FinancingEntry, ...]:
        current = utc_datetime(current_information_time, name="current_information_time")
        previous = (
            utc_datetime(previous_information_time, name="previous_information_time")
            if previous_information_time is not None
            else None
        )
        if previous is not None and current <= previous:
            raise ValueError("financing window must move forward")
        positions = portfolio.positions_by_symbol
        entries: list[FinancingEntry] = []
        for event in self.events:
            if event.timestamp > current or (previous is not None and event.timestamp <= previous):
                continue
            candidates = (
                (positions[event.symbol],)
                if event.symbol is not None and event.symbol in positions
                else portfolio.positions
                if event.symbol is None
                else ()
            )
            for position in candidates:
                if position.quantity >= ZERO:
                    continue
                notional = abs(position.market_value)
                amount = -(notional * event.rate)
                if amount == ZERO:
                    continue
                entries.append(
                    FinancingEntry(
                        amount=amount,
                        timestamp=event.timestamp,
                        category=event.category,
                        symbol=position.symbol,
                        exchange_reference=event.reference,
                        exchange_metadata={
                            "simulation": True,
                            "rate": str(event.rate),
                            "marked_short_notional": str(notional),
                            "information_timestamp": current.isoformat(),
                            "trade_group_id": event.trade_group_id,
                        },
                    )
                )
        return tuple(entries)
