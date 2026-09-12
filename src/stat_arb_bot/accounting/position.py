"""Signed-inventory position accounting."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from stat_arb_bot.accounting.models import PositionSnapshot, PositionTransition
from stat_arb_bot.domain.execution import (
    ZERO,
    DecimalLike,
    Fill,
    FinancingEntry,
    decimal_value,
    utc_datetime,
)


def _optional_time(value: str | datetime | None, *, name: str) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    return utc_datetime(parsed, name=name)


class Position:
    """One instrument's signed quantity and cumulative accounting state."""

    __slots__ = (
        "_average_entry_price",
        "_cumulative_fees",
        "_financing",
        "_first_fill_at",
        "_latest_mark_price",
        "_opened_at",
        "_quantity",
        "_realized_gross_pnl",
        "_short_collateral",
        "_updated_at",
        "symbol",
    )

    def __init__(
        self,
        symbol: str,
        *,
        quantity: DecimalLike = ZERO,
        average_entry_price: DecimalLike = ZERO,
        realized_gross_pnl: DecimalLike = ZERO,
        cumulative_fees: DecimalLike = ZERO,
        financing: DecimalLike = ZERO,
        latest_mark_price: DecimalLike | None = None,
        short_collateral: DecimalLike = ZERO,
        first_fill_at: str | datetime | None = None,
        opened_at: str | datetime | None = None,
        updated_at: str | datetime | None = None,
    ) -> None:
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol must not be empty")
        self.symbol = normalized_symbol
        self._quantity = decimal_value(quantity, name="quantity")
        self._average_entry_price = decimal_value(average_entry_price, name="average_entry_price")
        self._realized_gross_pnl = decimal_value(realized_gross_pnl, name="realized_gross_pnl")
        self._cumulative_fees = decimal_value(cumulative_fees, name="cumulative_fees")
        self._financing = decimal_value(financing, name="financing")
        self._latest_mark_price = (
            decimal_value(latest_mark_price, name="latest_mark_price")
            if latest_mark_price is not None
            else None
        )
        self._short_collateral = decimal_value(short_collateral, name="short_collateral")
        self._first_fill_at = _optional_time(first_fill_at, name="first_fill_at")
        self._opened_at = _optional_time(opened_at, name="opened_at")
        self._updated_at = _optional_time(updated_at, name="updated_at")
        self._validate_state()

    @property
    def quantity(self) -> Decimal:
        return self._quantity

    @property
    def average_entry_price(self) -> Decimal:
        return self._average_entry_price

    @property
    def realized_gross_pnl(self) -> Decimal:
        return self._realized_gross_pnl

    @property
    def cumulative_fees(self) -> Decimal:
        return self._cumulative_fees

    @property
    def financing(self) -> Decimal:
        return self._financing

    @property
    def latest_mark_price(self) -> Decimal | None:
        return self._latest_mark_price

    @property
    def short_collateral(self) -> Decimal:
        return self._short_collateral

    @property
    def first_fill_at(self) -> datetime | None:
        return self._first_fill_at

    @property
    def opened_at(self) -> datetime | None:
        return self._opened_at

    @property
    def updated_at(self) -> datetime | None:
        return self._updated_at

    @property
    def market_value(self) -> Decimal:
        return self.snapshot().market_value

    @property
    def unrealized_pnl(self) -> Decimal:
        return self.snapshot().unrealized_pnl

    def _validate_state(self) -> None:
        if self._average_entry_price < ZERO:
            raise ValueError("average_entry_price must be non-negative")
        if self._cumulative_fees < ZERO:
            raise ValueError("cumulative_fees must be non-negative")
        if self._short_collateral < ZERO:
            raise ValueError("short_collateral must be non-negative")
        if self._latest_mark_price is not None and self._latest_mark_price <= ZERO:
            raise ValueError("latest_mark_price must be positive")
        if self._quantity == ZERO:
            if self._average_entry_price != ZERO:
                raise ValueError("flat positions must have zero average entry price")
            if self._short_collateral != ZERO:
                raise ValueError("flat positions must have zero short collateral")
        else:
            if self._average_entry_price <= ZERO:
                raise ValueError("active positions must have a positive average entry price")
            if self._latest_mark_price is None:
                raise ValueError("active positions must have a mark price")
        if self._quantity > ZERO and self._short_collateral != ZERO:
            raise ValueError("long positions cannot carry short collateral")
        if self._quantity < ZERO and self._short_collateral <= ZERO:
            raise ValueError("short positions require positive collateral")

    def mark(self, price: DecimalLike, timestamp: datetime) -> None:
        normalized_price = decimal_value(price, name="mark price")
        if normalized_price <= ZERO:
            raise ValueError("mark price must be positive")
        self._latest_mark_price = normalized_price
        self._updated_at = utc_datetime(timestamp)

    def apply_fill(self, fill: Fill) -> PositionTransition:
        """Apply one fill and return its decomposed signed-position transition."""

        if fill.symbol != self.symbol:
            raise ValueError(f"fill symbol {fill.symbol} does not match {self.symbol}")

        old_quantity = self._quantity
        old_average = self._average_entry_price
        delta = fill.signed_quantity
        new_quantity = old_quantity + delta

        closed_long = (
            min(old_quantity, fill.quantity) if old_quantity > ZERO and delta < ZERO else ZERO
        )
        covered_short = (
            min(abs(old_quantity), fill.quantity) if old_quantity < ZERO and delta > ZERO else ZERO
        )
        opened_long = fill.quantity - covered_short if delta > ZERO else ZERO
        opened_short = fill.quantity - closed_long if delta < ZERO else ZERO

        if closed_long > ZERO:
            realized = closed_long * (fill.price - old_average)
        elif covered_short > ZERO:
            realized = covered_short * (old_average - fill.price)
        else:
            realized = ZERO

        if opened_short > ZERO:
            if fill.short_collateral <= ZERO:
                raise ValueError("a fill opening short inventory requires short_collateral")
            added_collateral = fill.short_collateral
        else:
            if fill.short_collateral != ZERO:
                raise ValueError("short_collateral is only valid when opening short inventory")
            added_collateral = ZERO

        if covered_short > ZERO:
            proportional_release = (
                self._short_collateral
                if covered_short == abs(old_quantity)
                else self._short_collateral * covered_short / abs(old_quantity)
            )
            released_collateral = (
                fill.released_short_collateral
                if fill.released_short_collateral is not None
                else proportional_release
            )
            if released_collateral > self._short_collateral:
                raise ValueError("released short collateral exceeds the position collateral")
            if new_quantity >= ZERO and released_collateral != self._short_collateral:
                raise ValueError("fully covered short inventory must release all collateral")
        else:
            if fill.released_short_collateral not in {None, ZERO}:
                raise ValueError(
                    "released_short_collateral is only valid when covering short inventory"
                )
            released_collateral = ZERO

        if old_quantity == ZERO:
            new_average = fill.price
        elif old_quantity * delta > ZERO:
            new_average = (abs(old_quantity) * old_average + fill.quantity * fill.price) / abs(
                new_quantity
            )
        elif new_quantity == ZERO:
            new_average = ZERO
        elif old_quantity * new_quantity > ZERO:
            new_average = old_average
        else:
            new_average = fill.price

        new_collateral = self._short_collateral - released_collateral + added_collateral
        if new_quantity >= ZERO and new_collateral != ZERO:
            raise ValueError("non-short inventory cannot retain short collateral")
        if new_quantity < ZERO and new_collateral <= ZERO:
            raise ValueError("short inventory requires positive remaining collateral")

        previous_opened_at = self._opened_at
        reversed_position = old_quantity * new_quantity < ZERO
        self._quantity = new_quantity
        self._average_entry_price = new_average
        self._realized_gross_pnl += realized
        self._cumulative_fees += fill.fee
        self._short_collateral = new_collateral
        self._latest_mark_price = fill.price
        self._first_fill_at = self._first_fill_at or fill.timestamp
        if new_quantity == ZERO:
            self._opened_at = None
        elif old_quantity == ZERO or reversed_position:
            self._opened_at = fill.timestamp
        else:
            self._opened_at = previous_opened_at
        self._updated_at = fill.timestamp
        self._validate_state()

        return PositionTransition(
            symbol=self.symbol,
            previous_quantity=old_quantity,
            new_quantity=new_quantity,
            previous_average_entry_price=old_average,
            new_average_entry_price=new_average,
            closed_long_quantity=closed_long,
            covered_short_quantity=covered_short,
            opened_long_quantity=opened_long,
            opened_short_quantity=opened_short,
            realized_gross_pnl=realized,
            added_short_collateral=added_collateral,
            released_short_collateral=released_collateral,
        )

    def apply_financing(self, entry: FinancingEntry) -> None:
        if entry.symbol != self.symbol:
            raise ValueError(f"financing symbol {entry.symbol} does not match {self.symbol}")
        self._financing += entry.amount
        self._updated_at = entry.timestamp

    def snapshot(self) -> PositionSnapshot:
        return PositionSnapshot(
            symbol=self.symbol,
            quantity=self._quantity,
            average_entry_price=self._average_entry_price,
            realized_gross_pnl=self._realized_gross_pnl,
            cumulative_fees=self._cumulative_fees,
            financing=self._financing,
            latest_mark_price=self._latest_mark_price,
            short_collateral=self._short_collateral,
            first_fill_at=self._first_fill_at,
            opened_at=self._opened_at,
            updated_at=self._updated_at,
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": str(self._quantity),
            "average_entry_price": str(self._average_entry_price),
            "realized_gross_pnl": str(self._realized_gross_pnl),
            "cumulative_fees": str(self._cumulative_fees),
            "financing": str(self._financing),
            "latest_mark_price": (
                str(self._latest_mark_price) if self._latest_mark_price is not None else None
            ),
            "short_collateral": str(self._short_collateral),
            "first_fill_at": self._first_fill_at.isoformat() if self._first_fill_at else None,
            "opened_at": self._opened_at.isoformat() if self._opened_at else None,
            "updated_at": self._updated_at.isoformat() if self._updated_at else None,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Position:
        return cls(
            symbol=str(state["symbol"]),
            quantity=str(state.get("quantity", "0")),
            average_entry_price=str(state.get("average_entry_price", "0")),
            realized_gross_pnl=str(state.get("realized_gross_pnl", "0")),
            cumulative_fees=str(state.get("cumulative_fees", "0")),
            financing=str(state.get("financing", "0")),
            latest_mark_price=(
                str(state["latest_mark_price"])
                if state.get("latest_mark_price") is not None
                else None
            ),
            short_collateral=str(state.get("short_collateral", "0")),
            first_fill_at=state.get("first_fill_at"),
            opened_at=state.get("opened_at"),
            updated_at=state.get("updated_at"),
        )
