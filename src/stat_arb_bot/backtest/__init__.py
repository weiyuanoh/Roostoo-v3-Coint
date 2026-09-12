"""Reusable deterministic historical backtesting infrastructure."""

from stat_arb_bot.backtest.clock import BarClock
from stat_arb_bot.backtest.costs import (
    CostParameters,
    ExecutionCostBreakdown,
    ExecutionCostModel,
    execution_decimal,
)
from stat_arb_bot.backtest.engine import BacktestEngine
from stat_arb_bot.backtest.execution import (
    ExecutionAttempt,
    FullFillPolicy,
    SimulatedExecutor,
    VolumeCapacityFillPolicy,
)
from stat_arb_bot.backtest.financing import (
    FinancingRateEvent,
    ScheduledShortFinancingModel,
    ZeroFinancingModel,
)
from stat_arb_bot.backtest.history import (
    HistoricalFrame,
    HistoricalPanel,
    MarketHistoryView,
    PanelBuildReport,
)
from stat_arb_bot.backtest.orders import (
    Order,
    OrderIntent,
    OrderStatus,
    OrderType,
    TradePhase,
)
from stat_arb_bot.backtest.results import (
    AppliedFillRecord,
    BacktestResult,
    OrderRecord,
    TradeGroupResult,
    TradeGroupStatus,
)
from stat_arb_bot.backtest.strategy import HistoricalStrategy, StrategyContext

__all__ = [
    "AppliedFillRecord",
    "BacktestEngine",
    "BacktestResult",
    "BarClock",
    "CostParameters",
    "ExecutionAttempt",
    "ExecutionCostBreakdown",
    "ExecutionCostModel",
    "FinancingRateEvent",
    "FullFillPolicy",
    "HistoricalFrame",
    "HistoricalPanel",
    "HistoricalStrategy",
    "MarketHistoryView",
    "Order",
    "OrderIntent",
    "OrderRecord",
    "OrderStatus",
    "OrderType",
    "PanelBuildReport",
    "ScheduledShortFinancingModel",
    "SimulatedExecutor",
    "StrategyContext",
    "TradeGroupResult",
    "TradeGroupStatus",
    "TradePhase",
    "VolumeCapacityFillPolicy",
    "ZeroFinancingModel",
    "execution_decimal",
]
