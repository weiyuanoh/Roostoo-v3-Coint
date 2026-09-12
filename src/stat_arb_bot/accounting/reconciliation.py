"""Exchange-observation reconciliation without exchange-specific dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from stat_arb_bot.accounting.models import PositionSnapshot
from stat_arb_bot.domain.execution import (
    ZERO,
    DecimalLike,
    decimal_value,
    utc_datetime,
)


@dataclass(frozen=True, slots=True)
class PositionObservation:
    """Normalized external view used only to compare with internal state."""

    symbol: str
    signed_quantity: DecimalLike
    average_entry_price: DecimalLike
    mark_price: DecimalLike
    short_collateral: DecimalLike
    reported_unrealized_pnl: DecimalLike
    observed_at: datetime
    external_position_id: str | None = None
    status: str | None = None
    exchange_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        if not symbol:
            raise ValueError("symbol must not be empty")
        quantity = decimal_value(self.signed_quantity, name="signed_quantity")
        average = decimal_value(self.average_entry_price, name="average_entry_price")
        mark = decimal_value(self.mark_price, name="mark_price")
        collateral = decimal_value(self.short_collateral, name="short_collateral")
        unrealized = decimal_value(
            self.reported_unrealized_pnl,
            name="reported_unrealized_pnl",
        )
        if quantity >= ZERO:
            raise ValueError("a short-position observation must have negative signed quantity")
        if average <= ZERO or mark <= ZERO:
            raise ValueError("observed prices must be positive")
        if collateral <= ZERO:
            raise ValueError("observed short collateral must be positive")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "signed_quantity", quantity)
        object.__setattr__(self, "average_entry_price", average)
        object.__setattr__(self, "mark_price", mark)
        object.__setattr__(self, "short_collateral", collateral)
        object.__setattr__(self, "reported_unrealized_pnl", unrealized)
        object.__setattr__(self, "observed_at", utc_datetime(self.observed_at))
        object.__setattr__(self, "exchange_metadata", dict(self.exchange_metadata))


@dataclass(frozen=True, slots=True)
class PositionReconciliation:
    """Observed-minus-internal differences for one position."""

    symbol: str
    quantity_delta: Decimal
    average_entry_price_delta: Decimal
    mark_price_delta: Decimal
    short_collateral_delta: Decimal
    unrealized_pnl_delta: Decimal
    tolerance: Decimal

    @property
    def matches(self) -> bool:
        return all(
            abs(value) <= self.tolerance
            for value in (
                self.quantity_delta,
                self.average_entry_price_delta,
                self.mark_price_delta,
                self.short_collateral_delta,
                self.unrealized_pnl_delta,
            )
        )


def reconcile_position(
    internal: PositionSnapshot,
    observed: PositionObservation,
    *,
    tolerance: DecimalLike = ZERO,
) -> PositionReconciliation:
    if internal.symbol != observed.symbol:
        raise ValueError(
            f"internal symbol {internal.symbol} does not match observation {observed.symbol}"
        )
    if internal.latest_mark_price is None:
        raise ValueError("internal position has no mark price")
    normalized_tolerance = decimal_value(tolerance, name="tolerance")
    if normalized_tolerance < ZERO:
        raise ValueError("tolerance must be non-negative")
    return PositionReconciliation(
        symbol=internal.symbol,
        quantity_delta=observed.signed_quantity - internal.quantity,
        average_entry_price_delta=(observed.average_entry_price - internal.average_entry_price),
        mark_price_delta=observed.mark_price - internal.latest_mark_price,
        short_collateral_delta=observed.short_collateral - internal.short_collateral,
        unrealized_pnl_delta=(observed.reported_unrealized_pnl - internal.unrealized_pnl),
        tolerance=normalized_tolerance,
    )
