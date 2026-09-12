"""Immutable accounting views and transition records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from stat_arb_bot.domain.execution import ZERO


class PositionDirection(str, Enum):
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    symbol: str
    quantity: Decimal
    average_entry_price: Decimal
    realized_gross_pnl: Decimal
    cumulative_fees: Decimal
    financing: Decimal
    latest_mark_price: Decimal | None
    short_collateral: Decimal
    first_fill_at: datetime | None
    opened_at: datetime | None
    updated_at: datetime | None

    @property
    def direction(self) -> PositionDirection:
        if self.quantity > ZERO:
            return PositionDirection.LONG
        if self.quantity < ZERO:
            return PositionDirection.SHORT
        return PositionDirection.FLAT

    @property
    def market_value(self) -> Decimal:
        if self.quantity == ZERO:
            return ZERO
        if self.latest_mark_price is None:
            raise ValueError(f"active position {self.symbol} has no mark price")
        return self.quantity * self.latest_mark_price

    @property
    def unrealized_pnl(self) -> Decimal:
        if self.quantity == ZERO:
            return ZERO
        if self.latest_mark_price is None:
            raise ValueError(f"active position {self.symbol} has no mark price")
        return self.quantity * (self.latest_mark_price - self.average_entry_price)

    @property
    def net_pnl(self) -> Decimal:
        return self.realized_gross_pnl + self.unrealized_pnl - self.cumulative_fees + self.financing


@dataclass(frozen=True, slots=True)
class PositionTransition:
    symbol: str
    previous_quantity: Decimal
    new_quantity: Decimal
    previous_average_entry_price: Decimal
    new_average_entry_price: Decimal
    closed_long_quantity: Decimal
    covered_short_quantity: Decimal
    opened_long_quantity: Decimal
    opened_short_quantity: Decimal
    realized_gross_pnl: Decimal
    added_short_collateral: Decimal
    released_short_collateral: Decimal

    @property
    def reversed(self) -> bool:
        return (
            self.previous_quantity > ZERO > self.new_quantity
            or self.previous_quantity < ZERO < self.new_quantity
        )


@dataclass(frozen=True, slots=True)
class FillApplication:
    transition: PositionTransition
    cash_before: Decimal
    cash_change: Decimal
    cash_after: Decimal


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    timestamp: datetime
    base_currency: str
    initial_cash: Decimal
    cash: Decimal
    available_cash: Decimal
    restricted_short_collateral: Decimal
    equity: Decimal
    gross_market_value: Decimal
    net_market_value: Decimal
    long_market_value: Decimal
    short_market_value: Decimal
    short_exposure: Decimal
    gross_leverage: Decimal | None
    realized_gross_pnl: Decimal
    unrealized_pnl: Decimal
    cumulative_fees: Decimal
    financing: Decimal
    net_pnl: Decimal
    positions: tuple[PositionSnapshot, ...]

    @property
    def nav(self) -> Decimal:
        return self.equity

    @property
    def positions_by_symbol(self) -> Mapping[str, PositionSnapshot]:
        return MappingProxyType({position.symbol: position for position in self.positions})

    @property
    def active_positions(self) -> tuple[PositionSnapshot, ...]:
        return tuple(position for position in self.positions if position.quantity != ZERO)
