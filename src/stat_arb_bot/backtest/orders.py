"""Strategy intents and executable historical orders."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from stat_arb_bot.domain.execution import (
    ZERO,
    DecimalLike,
    Side,
    decimal_value,
    utc_datetime,
)


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class TradePhase(str, Enum):
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    ADJUST = "ADJUST"


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """An economic instruction emitted by a strategy after a completed bar.

    BUY/SELL denotes signed inventory direction. It does not encode a
    Roostoo endpoint action. A collateral-backed short intent supplies the
    total restricted collateral for the intended quantity.
    """

    symbol: str
    side: Side | str
    quantity: DecimalLike
    order_type: OrderType | str = OrderType.MARKET
    limit_price: DecimalLike | None = None
    short_collateral: DecimalLike = ZERO
    trade_group_id: str | None = None
    phase: TradePhase | str = TradePhase.ADJUST
    client_order_id: str | None = None
    model_version: str | None = None
    model_fit_end_timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        if not symbol:
            raise ValueError("symbol must not be empty")
        try:
            side = self.side if isinstance(self.side, Side) else Side(self.side.upper())
        except (AttributeError, ValueError) as exc:
            raise ValueError("side must be BUY or SELL") from exc
        try:
            order_type = (
                self.order_type
                if isinstance(self.order_type, OrderType)
                else OrderType(self.order_type.upper())
            )
        except (AttributeError, ValueError) as exc:
            raise ValueError("unsupported order type") from exc
        try:
            phase = (
                self.phase if isinstance(self.phase, TradePhase) else TradePhase(self.phase.upper())
            )
        except (AttributeError, ValueError) as exc:
            raise ValueError("unsupported trade phase") from exc
        quantity = decimal_value(self.quantity, name="quantity")
        collateral = decimal_value(self.short_collateral, name="short_collateral")
        limit_price = (
            decimal_value(self.limit_price, name="limit_price")
            if self.limit_price is not None
            else None
        )
        if quantity <= ZERO:
            raise ValueError("quantity must be positive")
        if collateral < ZERO:
            raise ValueError("short_collateral must be non-negative")
        if order_type is OrderType.LIMIT and limit_price is None:
            raise ValueError("limit orders require limit_price")
        if order_type is OrderType.MARKET and limit_price is not None:
            raise ValueError("market orders cannot have limit_price")
        if limit_price is not None and limit_price <= ZERO:
            raise ValueError("limit_price must be positive")
        fit_time = (
            utc_datetime(self.model_fit_end_timestamp, name="model_fit_end_timestamp")
            if self.model_fit_end_timestamp is not None
            else None
        )
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "order_type", order_type)
        object.__setattr__(self, "limit_price", limit_price)
        object.__setattr__(self, "short_collateral", collateral)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "model_fit_end_timestamp", fit_time)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class Order:
    """A normalized executable instruction with enforced chronology."""

    order_id: str
    symbol: str
    side: Side
    quantity: Decimal
    order_type: OrderType
    signal_information_timestamp: datetime
    signal_timestamp: datetime
    submission_timestamp: datetime
    first_eligible_execution_timestamp: datetime
    limit_price: Decimal | None = None
    short_collateral: Decimal = ZERO
    trade_group_id: str | None = None
    phase: TradePhase = TradePhase.ADJUST
    client_order_id: str | None = None
    model_version: str | None = None
    model_fit_end_timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        order_id = self.order_id.strip()
        symbol = self.symbol.strip().upper()
        if not order_id or not symbol:
            raise ValueError("order_id and symbol must not be empty")
        try:
            side = self.side if isinstance(self.side, Side) else Side(str(self.side).upper())
        except ValueError as exc:
            raise ValueError("side must be BUY or SELL") from exc
        try:
            order_type = (
                self.order_type
                if isinstance(self.order_type, OrderType)
                else OrderType(str(self.order_type).upper())
            )
        except ValueError as exc:
            raise ValueError("unsupported order type") from exc
        try:
            phase = (
                self.phase
                if isinstance(self.phase, TradePhase)
                else TradePhase(str(self.phase).upper())
            )
        except ValueError as exc:
            raise ValueError("unsupported trade phase") from exc
        quantity = decimal_value(self.quantity, name="quantity")
        collateral = decimal_value(self.short_collateral, name="short_collateral")
        limit_price = (
            decimal_value(self.limit_price, name="limit_price")
            if self.limit_price is not None
            else None
        )
        if quantity <= ZERO:
            raise ValueError("quantity must be positive")
        if collateral < ZERO:
            raise ValueError("short_collateral must be non-negative")
        if order_type is OrderType.LIMIT and limit_price is None:
            raise ValueError("limit orders require limit_price")
        if order_type is OrderType.MARKET and limit_price is not None:
            raise ValueError("market orders cannot have limit_price")
        if limit_price is not None and limit_price <= ZERO:
            raise ValueError("limit_price must be positive")
        info_time = utc_datetime(
            self.signal_information_timestamp,
            name="signal_information_timestamp",
        )
        signal_time = utc_datetime(self.signal_timestamp, name="signal_timestamp")
        submission_time = utc_datetime(self.submission_timestamp, name="submission_timestamp")
        eligible_time = utc_datetime(
            self.first_eligible_execution_timestamp,
            name="first_eligible_execution_timestamp",
        )
        if info_time > signal_time:
            raise ValueError("signal information cannot postdate the signal")
        if signal_time > submission_time:
            raise ValueError("signal cannot postdate order submission")
        if eligible_time <= signal_time:
            raise ValueError("execution eligibility must be strictly after the signal")
        if self.model_fit_end_timestamp is not None:
            fit_time = utc_datetime(
                self.model_fit_end_timestamp,
                name="model_fit_end_timestamp",
            )
            if fit_time > info_time:
                raise ValueError("model fit end cannot postdate the signal information set")
            object.__setattr__(self, "model_fit_end_timestamp", fit_time)
        object.__setattr__(self, "order_id", order_id)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "order_type", order_type)
        object.__setattr__(self, "limit_price", limit_price)
        object.__setattr__(self, "short_collateral", collateral)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "signal_information_timestamp", info_time)
        object.__setattr__(self, "signal_timestamp", signal_time)
        object.__setattr__(self, "submission_timestamp", submission_time)
        object.__setattr__(self, "first_eligible_execution_timestamp", eligible_time)
        object.__setattr__(self, "metadata", dict(self.metadata))
