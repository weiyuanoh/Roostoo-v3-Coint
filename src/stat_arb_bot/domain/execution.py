"""Exchange-neutral execution and financing events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, TypeAlias

DecimalLike: TypeAlias = Decimal | str | int | float
ZERO = Decimal("0")


def decimal_value(value: DecimalLike, *, name: str) -> Decimal:
    """Convert a value through its decimal text, rejecting NaN and infinity."""

    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def utc_datetime(value: datetime, *, name: str = "timestamp") -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


class Side(str, Enum):
    """Normalized inventory direction, independent of exchange order actions."""

    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True, slots=True)
class Fill:
    """One normalized execution applied to signed inventory.

    ``side`` describes inventory direction only: BUY increases signed quantity
    and SELL decreases it. ``short_collateral`` is restricted cash added for
    any portion that opens/increases a short. ``released_short_collateral`` may
    carry an exchange-reported release on a cover; when absent the accounting
    engine releases collateral proportionally.
    """

    symbol: str
    side: Side | str
    quantity: DecimalLike
    price: DecimalLike
    timestamp: datetime
    fee: DecimalLike = ZERO
    fee_currency: str | None = None
    exchange_order_id: str | None = None
    client_order_id: str | None = None
    trade_group_id: str | None = None
    original_action: str | None = None
    short_collateral: DecimalLike = ZERO
    released_short_collateral: DecimalLike | None = None
    exchange_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        if not symbol:
            raise ValueError("symbol must not be empty")
        try:
            side = self.side if isinstance(self.side, Side) else Side(self.side.upper())
        except (AttributeError, ValueError) as exc:
            raise ValueError("side must be BUY or SELL") from exc
        quantity = decimal_value(self.quantity, name="quantity")
        price = decimal_value(self.price, name="price")
        fee = decimal_value(self.fee, name="fee")
        collateral = decimal_value(self.short_collateral, name="short_collateral")
        released = (
            decimal_value(
                self.released_short_collateral,
                name="released_short_collateral",
            )
            if self.released_short_collateral is not None
            else None
        )
        if quantity <= ZERO:
            raise ValueError("quantity must be positive")
        if price <= ZERO:
            raise ValueError("price must be positive")
        if fee < ZERO:
            raise ValueError("fee must be non-negative")
        if collateral < ZERO:
            raise ValueError("short_collateral must be non-negative")
        if released is not None and released < ZERO:
            raise ValueError("released_short_collateral must be non-negative")
        fee_currency = self.fee_currency.strip().upper() if self.fee_currency else None
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "timestamp", utc_datetime(self.timestamp))
        object.__setattr__(self, "fee", fee)
        object.__setattr__(self, "fee_currency", fee_currency)
        object.__setattr__(self, "short_collateral", collateral)
        object.__setattr__(self, "released_short_collateral", released)
        object.__setattr__(self, "exchange_metadata", dict(self.exchange_metadata))

    @property
    def signed_quantity(self) -> Decimal:
        direction = Decimal("1") if self.side is Side.BUY else Decimal("-1")
        return direction * self.quantity

    def to_state(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": str(self.quantity),
            "price": str(self.price),
            "timestamp": self.timestamp.isoformat(),
            "fee": str(self.fee),
            "fee_currency": self.fee_currency,
            "exchange_order_id": self.exchange_order_id,
            "client_order_id": self.client_order_id,
            "trade_group_id": self.trade_group_id,
            "original_action": self.original_action,
            "short_collateral": str(self.short_collateral),
            "released_short_collateral": (
                str(self.released_short_collateral)
                if self.released_short_collateral is not None
                else None
            ),
            "exchange_metadata": self.exchange_metadata,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Fill:
        return cls(
            symbol=str(state["symbol"]),
            side=str(state["side"]),
            quantity=str(state["quantity"]),
            price=str(state["price"]),
            timestamp=datetime.fromisoformat(str(state["timestamp"])),
            fee=str(state.get("fee", "0")),
            fee_currency=state.get("fee_currency"),
            exchange_order_id=state.get("exchange_order_id"),
            client_order_id=state.get("client_order_id"),
            trade_group_id=state.get("trade_group_id"),
            original_action=state.get("original_action"),
            short_collateral=str(state.get("short_collateral", "0")),
            released_short_collateral=(
                str(state["released_short_collateral"])
                if state.get("released_short_collateral") is not None
                else None
            ),
            exchange_metadata=dict(state.get("exchange_metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class FinancingEntry:
    """A signed financing/funding cash event.

    Negative ``amount`` values are charges; positive values are credits. The
    amount is applied to cash and equity exactly once.
    """

    amount: DecimalLike
    timestamp: datetime
    category: str
    symbol: str | None = None
    exchange_reference: str | None = None
    exchange_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        category = self.category.strip()
        if not category:
            raise ValueError("category must not be empty")
        symbol = self.symbol.strip().upper() if self.symbol else None
        object.__setattr__(self, "amount", decimal_value(self.amount, name="amount"))
        object.__setattr__(self, "timestamp", utc_datetime(self.timestamp))
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "exchange_metadata", dict(self.exchange_metadata))

    def to_state(self) -> dict[str, Any]:
        return {
            "amount": str(self.amount),
            "timestamp": self.timestamp.isoformat(),
            "category": self.category,
            "symbol": self.symbol,
            "exchange_reference": self.exchange_reference,
            "exchange_metadata": self.exchange_metadata,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> FinancingEntry:
        return cls(
            amount=str(state["amount"]),
            timestamp=datetime.fromisoformat(str(state["timestamp"])),
            category=str(state["category"]),
            symbol=state.get("symbol"),
            exchange_reference=state.get("exchange_reference"),
            exchange_metadata=dict(state.get("exchange_metadata", {})),
        )
