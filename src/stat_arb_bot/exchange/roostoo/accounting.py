"""Map documented Roostoo short responses into exchange-neutral accounting objects."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from stat_arb_bot.accounting import FillApplication, PositionObservation
from stat_arb_bot.domain import Fill, Side, epoch_ms_to_utc
from stat_arb_bot.domain.execution import DecimalLike, decimal_value
from stat_arb_bot.exchange.roostoo.models import (
    ShortCloseResult,
    ShortOpenResult,
    ShortPosition as RoostooShortPosition,
)


class RoostooAccountingMappingError(ValueError):
    """Raised when a Roostoo response cannot be normalized without guessing."""


@dataclass(frozen=True, slots=True)
class RoostooShortCloseReconciliation:
    """Reported-minus-internal differences for a v6 short close."""

    realized_pnl_delta: Decimal
    return_amount_delta: Decimal
    tolerance: Decimal

    @property
    def matches(self) -> bool:
        return (
            abs(self.realized_pnl_delta) <= self.tolerance
            and abs(self.return_amount_delta) <= self.tolerance
        )


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def short_open_to_fill(
    result: ShortOpenResult,
    *,
    previous: RoostooShortPosition | None,
    client_order_id: str | None = None,
    trade_group_id: str | None = None,
) -> Fill:
    """Normalize a filled v6 short-open response.

    A merged response reports totals rather than this request's fill quantity
    and price. The caller must explicitly pass either the prior exchange
    position or ``None`` when it knows the pair was previously flat. A merged
    incremental fill is reconstructed from the documented weighted average.
    """

    if result.status.upper() != "OPEN":
        raise RoostooAccountingMappingError(
            "a pending short-open response is an order acknowledgement, not a fill"
        )
    if previous is not None and previous.pair.upper() != result.pair.upper():
        raise RoostooAccountingMappingError("previous and new short positions have different pairs")

    new_quantity = _decimal(result.short_quantity)
    new_average = _decimal(result.entry_price)
    new_collateral = _decimal(result.collateral)
    old_quantity = _decimal(previous.short_quantity) if previous else Decimal("0")
    old_average = _decimal(previous.entry_price) if previous else Decimal("0")
    old_collateral = _decimal(previous.collateral) if previous else Decimal("0")
    fill_quantity = new_quantity - old_quantity
    collateral_added = new_collateral - old_collateral
    if fill_quantity <= 0:
        raise RoostooAccountingMappingError("short-open response did not increase short quantity")
    if collateral_added <= 0:
        raise RoostooAccountingMappingError("short-open response did not increase collateral")
    if previous is None:
        fill_price = new_average
        reconstructed = False
    else:
        fill_price = (new_average * new_quantity - old_average * old_quantity) / fill_quantity
        reconstructed = True
    if fill_price <= 0:
        raise RoostooAccountingMappingError("reconstructed short-open fill price is not positive")

    return Fill(
        symbol=result.pair,
        side=Side.SELL,
        quantity=fill_quantity,
        price=fill_price,
        timestamp=epoch_ms_to_utc(result.create_timestamp),
        fee=_decimal(result.open_fee),
        fee_currency="USD",
        client_order_id=client_order_id,
        trade_group_id=trade_group_id,
        original_action="SHORT_OPEN",
        short_collateral=collateral_added,
        exchange_metadata={
            "roostoo_position_id": result.id,
            "order_type": result.order_type,
            "status": result.status,
            "reported_total_short_quantity": str(new_quantity),
            "reported_total_collateral": str(new_collateral),
            "reported_weighted_average_entry_price": str(new_average),
            "fill_price_reconstructed": reconstructed,
        },
    )


def short_close_to_fill(
    result: ShortCloseResult,
    *,
    previous: RoostooShortPosition,
    timestamp: datetime,
    exchange_order_id: str | None = None,
    client_order_id: str | None = None,
    trade_group_id: str | None = None,
) -> Fill:
    """Normalize a v6 short-close settlement into a BUY inventory fill."""

    closed_quantity = _decimal(result.closed_quantity)
    previous_quantity = _decimal(previous.short_quantity)
    previous_collateral = _decimal(previous.collateral)
    if closed_quantity <= 0 or closed_quantity > previous_quantity:
        raise RoostooAccountingMappingError(
            "short-close quantity is incompatible with the previous exchange position"
        )
    if result.fully_closed:
        released_collateral = previous_collateral
    else:
        if result.remaining_quantity is None or result.remaining_collateral is None:
            raise RoostooAccountingMappingError(
                "partial short-close response is missing remaining position fields"
            )
        released_collateral = previous_collateral - _decimal(result.remaining_collateral)
    if released_collateral <= 0:
        raise RoostooAccountingMappingError(
            "short-close response did not release positive collateral"
        )

    return Fill(
        symbol=previous.pair,
        side=Side.BUY,
        quantity=closed_quantity,
        price=_decimal(result.close_price),
        timestamp=timestamp,
        fee=_decimal(result.close_fee),
        fee_currency="USD",
        exchange_order_id=exchange_order_id,
        client_order_id=client_order_id,
        trade_group_id=trade_group_id,
        original_action="SHORT_CLOSE",
        released_short_collateral=released_collateral,
        exchange_metadata={
            "roostoo_position_id": previous.id,
            "reported_realized_pnl": str(_decimal(result.realized_pnl)),
            "reported_return_amount": str(_decimal(result.return_amount)),
            "fully_closed": result.fully_closed,
            "reported_remaining_quantity": (
                str(_decimal(result.remaining_quantity))
                if result.remaining_quantity is not None
                else None
            ),
            "reported_remaining_collateral": (
                str(_decimal(result.remaining_collateral))
                if result.remaining_collateral is not None
                else None
            ),
        },
    )


def short_position_to_observation(
    position: RoostooShortPosition,
    *,
    observed_at: datetime,
) -> PositionObservation:
    """Map positive Roostoo ``ShortQty`` into negative signed inventory."""

    return PositionObservation(
        symbol=position.pair,
        signed_quantity=-_decimal(position.short_quantity),
        average_entry_price=_decimal(position.entry_price),
        mark_price=_decimal(position.current_price),
        short_collateral=_decimal(position.collateral),
        reported_unrealized_pnl=_decimal(position.unrealized_pnl),
        observed_at=observed_at,
        external_position_id=str(position.id),
        status=position.status,
        exchange_metadata={
            "reported_unrealized_pnl_pct": str(_decimal(position.unrealized_pnl_pct)),
            "reported_position_value": str(_decimal(position.position_value)),
            "create_timestamp": position.create_timestamp,
        },
    )


def reconcile_short_close(
    application: FillApplication,
    result: ShortCloseResult,
    *,
    tolerance: DecimalLike = Decimal("0"),
) -> RoostooShortCloseReconciliation:
    """Compare deterministic cover accounting with Roostoo settlement fields."""

    if application.transition.covered_short_quantity <= 0:
        raise ValueError("fill application did not cover short inventory")
    normalized_tolerance = decimal_value(tolerance, name="tolerance")
    if normalized_tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    return RoostooShortCloseReconciliation(
        realized_pnl_delta=(
            _decimal(result.realized_pnl) - application.transition.realized_gross_pnl
        ),
        return_amount_delta=_decimal(result.return_amount) - application.cash_change,
        tolerance=normalized_tolerance,
    )
