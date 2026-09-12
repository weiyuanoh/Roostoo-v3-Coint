"""Deterministic next-bar simulated execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Callable, Protocol

from stat_arb_bot.backtest.costs import (
    ExecutionCostBreakdown,
    ExecutionCostModel,
    execution_decimal,
)
from stat_arb_bot.backtest.orders import Order, OrderStatus, OrderType
from stat_arb_bot.domain.execution import ZERO, Fill, Side
from stat_arb_bot.domain.market import Candle


class FillPolicy(Protocol):
    """Maximum simulated base-asset quantity available in one bar."""

    def capacity(self, candle: Candle) -> Decimal | None: ...


@dataclass(frozen=True, slots=True)
class FullFillPolicy:
    def capacity(self, candle: Candle) -> None:
        del candle
        return None


@dataclass(frozen=True, slots=True)
class VolumeCapacityFillPolicy:
    fraction_of_bar_volume: Decimal

    def __init__(self, fraction_of_bar_volume: Decimal | str | int | float) -> None:
        fraction = execution_decimal(
            fraction_of_bar_volume,
            name="fraction_of_bar_volume",
        )
        if fraction < ZERO or fraction > Decimal("1"):
            raise ValueError("fraction_of_bar_volume must be between zero and one")
        object.__setattr__(self, "fraction_of_bar_volume", fraction)

    def capacity(self, candle: Candle) -> Decimal:
        return execution_decimal(candle.volume, name=f"{candle.symbol} volume") * (
            self.fraction_of_bar_volume
        )


RejectionPolicy = Callable[[Order, Candle], str | None]


@dataclass(frozen=True, slots=True)
class ExecutionAttempt:
    order_id: str
    status: OrderStatus
    fill: Fill | None = None
    costs: ExecutionCostBreakdown | None = None
    reason: str | None = None


class SimulatedExecutor:
    """Execute eligible instructions without owning portfolio state."""

    def __init__(
        self,
        *,
        cost_model: ExecutionCostModel | None = None,
        fill_policy: FillPolicy | None = None,
        rejection_policy: RejectionPolicy | None = None,
        base_currency: str = "USD",
    ) -> None:
        self.cost_model = cost_model or ExecutionCostModel()
        self.fill_policy = fill_policy or FullFillPolicy()
        self.rejection_policy = rejection_policy
        self.base_currency = base_currency.strip().upper()
        if not self.base_currency:
            raise ValueError("base_currency must not be empty")

    def capacity(self, candle: Candle) -> Decimal | None:
        return self.fill_policy.capacity(candle)

    def candidate_execution_time(self, order: Order, candle: Candle) -> datetime | None:
        """Return the conservative time supported by this bar, without filling."""

        if candle.open_time < order.first_eligible_execution_timestamp:
            return None
        eligible = self._eligible_execution(order, candle)
        return eligible[1] if eligible is not None else None

    def execute(
        self,
        order: Order,
        candle: Candle,
        *,
        remaining_quantity: Decimal,
        available_capacity: Decimal | None,
        current_position_quantity: Decimal,
    ) -> ExecutionAttempt:
        """Attempt one fill at the future bar open/range.

        The caller owns cumulative bar capacity and order state so reset and
        partial-fill behaviour remain deterministic.
        """

        if order.symbol != candle.symbol.strip().upper():
            raise ValueError("order and candle symbols differ")
        if candle.open_time < order.first_eligible_execution_timestamp:
            return ExecutionAttempt(order.order_id, OrderStatus.PENDING, reason="not_eligible")
        if self.rejection_policy is not None:
            reason = self.rejection_policy(order, candle)
            if reason:
                return ExecutionAttempt(order.order_id, OrderStatus.REJECTED, reason=reason)

        eligible_execution = self._eligible_execution(order, candle)
        if eligible_execution is None:
            return ExecutionAttempt(order.order_id, OrderStatus.PENDING, reason="limit_not_touched")
        base_price, execution_time = eligible_execution
        if execution_time <= order.signal_timestamp:
            raise ValueError("fill timestamp must be strictly after signal timestamp")
        fill_quantity = remaining_quantity
        if available_capacity is not None:
            fill_quantity = min(fill_quantity, available_capacity)
        if fill_quantity <= ZERO:
            return ExecutionAttempt(order.order_id, OrderStatus.PENDING, reason="no_capacity")

        costs = self.cost_model.quote(
            order,
            base_price=base_price,
            quantity=fill_quantity,
            current_position_quantity=current_position_quantity,
        )
        signed_fill = fill_quantity if order.side is Side.BUY else -fill_quantity
        new_quantity = current_position_quantity + signed_fill
        opened_short = (
            max(ZERO, -new_quantity) - max(ZERO, -current_position_quantity)
            if order.side is Side.SELL
            else ZERO
        )
        collateral = ZERO
        if opened_short > ZERO:
            if order.short_collateral <= ZERO:
                return ExecutionAttempt(
                    order.order_id,
                    OrderStatus.REJECTED,
                    reason="short_open_requires_collateral",
                )
            collateral = order.short_collateral * fill_quantity / order.quantity

        fill = Fill(
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=costs.execution_price,
            timestamp=execution_time,
            fee=costs.total_fee,
            fee_currency=self.base_currency,
            exchange_order_id=order.order_id,
            client_order_id=order.client_order_id,
            trade_group_id=order.trade_group_id,
            original_action="SIMULATED_BUY" if order.side is Side.BUY else "SIMULATED_SELL",
            short_collateral=collateral,
            exchange_metadata={
                "strategy_metadata": order.metadata,
                "simulation": True,
                "base_price": str(costs.base_price),
                "spread_cost": str(costs.spread_cost),
                "slippage_cost": str(costs.slippage_cost),
                "trading_fee": str(costs.trading_fee),
                "short_open_fee": str(costs.short_open_fee),
                "short_close_fee": str(costs.short_close_fee),
                "model_version": order.model_version,
                "model_fit_end_timestamp": (
                    order.model_fit_end_timestamp.isoformat()
                    if order.model_fit_end_timestamp is not None
                    else None
                ),
            },
        )
        status = (
            OrderStatus.FILLED
            if fill_quantity == remaining_quantity
            else OrderStatus.PARTIALLY_FILLED
        )
        return ExecutionAttempt(order.order_id, status, fill=fill, costs=costs)

    @staticmethod
    def _eligible_execution(
        order: Order,
        candle: Candle,
    ) -> tuple[Decimal, datetime] | None:
        if order.order_type is OrderType.MARKET:
            return execution_decimal(candle.open, name=f"{candle.symbol} open"), candle.open_time
        limit = order.limit_price
        if limit is None:
            raise ValueError("limit order is missing a limit price")
        low = execution_decimal(candle.low, name=f"{candle.symbol} low")
        high = execution_decimal(candle.high, name=f"{candle.symbol} high")
        open_price = execution_decimal(candle.open, name=f"{candle.symbol} open")
        if order.side is Side.BUY and open_price <= limit:
            return limit, candle.open_time
        if order.side is Side.SELL and open_price >= limit:
            return limit, candle.open_time
        if order.side is Side.BUY and low <= limit:
            return limit, candle.close_time
        if order.side is Side.SELL and high >= limit:
            return limit, candle.close_time
        return None
