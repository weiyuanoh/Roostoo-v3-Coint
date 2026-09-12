"""Immutable research results from a historical replay."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from stat_arb_bot.accounting import PortfolioSnapshot, PositionTransition
from stat_arb_bot.backtest.costs import ExecutionCostBreakdown
from stat_arb_bot.backtest.history import PanelBuildReport
from stat_arb_bot.backtest.orders import Order, OrderStatus
from stat_arb_bot.domain import Fill, FinancingEntry


@dataclass(frozen=True, slots=True)
class OrderRecord:
    order: Order
    status: OrderStatus
    filled_quantity: Decimal
    fill_timestamps: tuple[datetime, ...]
    failure_reason: str | None = None

    @property
    def execution_timestamp(self) -> datetime | None:
        return self.fill_timestamps[-1] if self.fill_timestamps else None


@dataclass(frozen=True, slots=True)
class AppliedFillRecord:
    fill: Fill
    order_id: str
    costs: ExecutionCostBreakdown
    transition: PositionTransition


class TradeGroupStatus(str, Enum):
    PENDING = "PENDING"
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class TradeGroupResult:
    trade_group_id: str
    status: TradeGroupStatus
    order_ids: tuple[str, ...]
    intended_legs: int
    submitted_legs: int
    filled_legs: int
    failed_legs: int
    partial_legs: int
    first_eligible_execution_timestamp: datetime
    combined_entry_timestamp: datetime | None
    combined_exit_timestamp: datetime | None
    total_fees: Decimal
    total_spread_cost: Decimal
    total_slippage_cost: Decimal
    attributed_financing: Decimal
    aggregate_realized_gross_pnl: Decimal


@dataclass(frozen=True, slots=True)
class BacktestResult:
    snapshots: tuple[PortfolioSnapshot, ...]
    orders: tuple[OrderRecord, ...]
    fills: tuple[AppliedFillRecord, ...]
    financing_entries: tuple[FinancingEntry, ...]
    trade_groups: tuple[TradeGroupResult, ...]
    data_report: PanelBuildReport

    @property
    def final_snapshot(self) -> PortfolioSnapshot:
        if not self.snapshots:
            raise ValueError("backtest has no snapshots")
        return self.snapshots[-1]
