"""Explicit execution-cost models and the float-to-Decimal boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType
from typing import Mapping

from stat_arb_bot.backtest.orders import Order, OrderType
from stat_arb_bot.domain.execution import ZERO, DecimalLike, Side, decimal_value

BPS_DENOMINATOR = Decimal("10000")


def execution_decimal(value: DecimalLike, *, name: str) -> Decimal:
    """Cross market-data values into exact accounting decimals in one place."""

    return decimal_value(value, name=name)


@dataclass(frozen=True, slots=True)
class CostParameters:
    maker_fee_bps: DecimalLike = ZERO
    taker_fee_bps: DecimalLike = ZERO
    spread_bps: DecimalLike = ZERO
    slippage_bps: DecimalLike = ZERO
    fixed_slippage: DecimalLike = ZERO
    short_open_fee_bps: DecimalLike | None = None
    short_close_fee_bps: DecimalLike | None = None

    def __post_init__(self) -> None:
        names = (
            "maker_fee_bps",
            "taker_fee_bps",
            "spread_bps",
            "slippage_bps",
            "fixed_slippage",
        )
        for name in names:
            value = execution_decimal(getattr(self, name), name=name)
            if value < ZERO:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in ("short_open_fee_bps", "short_close_fee_bps"):
            raw = getattr(self, name)
            if raw is None:
                continue
            value = execution_decimal(raw, name=name)
            if value < ZERO:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class ExecutionCostBreakdown:
    base_price: Decimal
    execution_price: Decimal
    spread_cost: Decimal
    slippage_cost: Decimal
    trading_fee: Decimal
    short_open_fee: Decimal
    short_close_fee: Decimal

    @property
    def total_fee(self) -> Decimal:
        return self.trading_fee + self.short_open_fee + self.short_close_fee

    @property
    def total_implicit_cost(self) -> Decimal:
        return self.spread_cost + self.slippage_cost


@dataclass(frozen=True, slots=True)
class ExecutionCostModel:
    """Maker/taker fees plus separately reported spread and slippage."""

    default: CostParameters = field(default_factory=CostParameters)
    symbol_overrides: Mapping[str, CostParameters] = field(default_factory=dict)

    def __post_init__(self) -> None:
        overrides = {
            symbol.strip().upper(): parameters
            for symbol, parameters in self.symbol_overrides.items()
        }
        if any(not symbol for symbol in overrides):
            raise ValueError("cost override symbol must not be empty")
        object.__setattr__(self, "symbol_overrides", MappingProxyType(overrides))

    def parameters_for(self, symbol: str) -> CostParameters:
        return self.symbol_overrides.get(symbol.strip().upper(), self.default)

    def quote(
        self,
        order: Order,
        *,
        base_price: DecimalLike,
        quantity: DecimalLike,
        current_position_quantity: DecimalLike,
    ) -> ExecutionCostBreakdown:
        """Return execution price and observable fee components for one fill."""

        base = execution_decimal(base_price, name="base_price")
        fill_quantity = execution_decimal(quantity, name="quantity")
        current = execution_decimal(current_position_quantity, name="current_position_quantity")
        if base <= ZERO or fill_quantity <= ZERO:
            raise ValueError("base_price and quantity must be positive")
        parameters = self.parameters_for(order.symbol)

        spread_per_unit = ZERO
        slippage_per_unit = ZERO
        if order.order_type is OrderType.MARKET:
            spread_per_unit = base * parameters.spread_bps / (BPS_DENOMINATOR * 2)
            slippage_per_unit = (
                base * parameters.slippage_bps / BPS_DENOMINATOR + parameters.fixed_slippage
            )
        adjustment = spread_per_unit + slippage_per_unit
        execution_price = base + adjustment if order.side is Side.BUY else base - adjustment
        if execution_price <= ZERO:
            raise ValueError("configured costs produce a non-positive execution price")

        signed_fill = fill_quantity if order.side is Side.BUY else -fill_quantity
        new_quantity = current + signed_fill
        opened_short = ZERO
        covered_short = ZERO
        if order.side is Side.SELL:
            opened_short = max(ZERO, -new_quantity) - max(ZERO, -current)
        else:
            covered_short = min(fill_quantity, abs(min(current, ZERO)))
        ordinary_quantity = fill_quantity - opened_short - covered_short

        base_fee_bps = (
            parameters.maker_fee_bps
            if order.order_type is OrderType.LIMIT
            else parameters.taker_fee_bps
        )
        trading_fee = ordinary_quantity * execution_price * base_fee_bps / BPS_DENOMINATOR
        short_open_rate = (
            parameters.short_open_fee_bps
            if parameters.short_open_fee_bps is not None
            else base_fee_bps
        )
        short_close_rate = (
            parameters.short_close_fee_bps
            if parameters.short_close_fee_bps is not None
            else base_fee_bps
        )
        short_open_fee = opened_short * execution_price * short_open_rate / BPS_DENOMINATOR
        short_close_fee = covered_short * execution_price * short_close_rate / BPS_DENOMINATOR
        return ExecutionCostBreakdown(
            base_price=base,
            execution_price=execution_price,
            spread_cost=spread_per_unit * fill_quantity,
            slippage_cost=slippage_per_unit * fill_quantity,
            trading_fee=trading_fee,
            short_open_fee=short_open_fee,
            short_close_fee=short_close_fee,
        )
