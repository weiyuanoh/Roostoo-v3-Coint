"""Deterministic historical replay with enforced next-bar execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Iterable, Sequence

from stat_arb_bot.accounting import Portfolio
from stat_arb_bot.backtest.execution import SimulatedExecutor
from stat_arb_bot.backtest.financing import FinancingModel, ZeroFinancingModel
from stat_arb_bot.backtest.history import HistoricalPanel
from stat_arb_bot.backtest.orders import Order, OrderIntent, OrderStatus, TradePhase
from stat_arb_bot.backtest.results import (
    AppliedFillRecord,
    BacktestResult,
    OrderRecord,
    TradeGroupResult,
    TradeGroupStatus,
)
from stat_arb_bot.backtest.strategy import HistoricalStrategy, StrategyContext
from stat_arb_bot.domain import Candle, FinancingEntry
from stat_arb_bot.domain.execution import ZERO, DecimalLike


@dataclass(slots=True)
class _OrderState:
    order: Order
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = ZERO
    fill_timestamps: list[datetime] = field(default_factory=list)
    failure_reason: str | None = None

    @property
    def remaining_quantity(self) -> Decimal:
        return self.order.quantity - self.filled_quantity

    @property
    def active(self) -> bool:
        return self.status in {OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED}

    def record(self) -> OrderRecord:
        return OrderRecord(
            order=self.order,
            status=self.status,
            filled_quantity=self.filled_quantity,
            fill_timestamps=tuple(self.fill_timestamps),
            failure_reason=self.failure_reason,
        )


class BacktestEngine:
    """Replay synchronized completed bars through execution and Portfolio.

    Each call to :meth:`run` creates fresh portfolio, pending-order, group, and
    financing state. The same inputs therefore produce the same result.
    """

    def __init__(
        self,
        *,
        initial_cash: DecimalLike,
        required_symbols: Sequence[str],
        executor: SimulatedExecutor | None = None,
        financing_model: FinancingModel | None = None,
        base_currency: str = "USD",
    ) -> None:
        self.initial_cash = initial_cash
        self.required_symbols = tuple(
            dict.fromkeys(symbol.strip().upper() for symbol in required_symbols)
        )
        if not self.required_symbols or any(not symbol for symbol in self.required_symbols):
            raise ValueError("required_symbols must contain non-empty symbols")
        self.executor = executor or SimulatedExecutor(base_currency=base_currency)
        self.financing_model = financing_model or ZeroFinancingModel()
        self.base_currency = base_currency

    def run(
        self,
        strategy: HistoricalStrategy,
        candles: Iterable[Candle],
        *,
        as_of: datetime,
    ) -> BacktestResult:
        panel = HistoricalPanel(
            candles,
            required_symbols=self.required_symbols,
            as_of=as_of,
        )
        if not panel.frames:
            raise ValueError("no synchronized completed historical frames are available")

        strategy.reset()
        portfolio = Portfolio(self.initial_cash, base_currency=self.base_currency)
        order_states: list[_OrderState] = []
        fill_records: list[AppliedFillRecord] = []
        snapshots = []
        financing_entries = []
        previous_information_time: datetime | None = None
        next_order_number = 1

        for index, frame in enumerate(panel.frames):
            capacities = {
                symbol: self.executor.capacity(candle) for symbol, candle in frame.bars.items()
            }
            active_states = [
                state for state in order_states if state.active and state.order.symbol in frame.bars
            ]
            active_states.sort(
                key=lambda state: (
                    self.executor.candidate_execution_time(
                        state.order,
                        frame.bars[state.order.symbol],
                    )
                    or frame.clock.bar_close + timedelta(microseconds=1)
                )
            )
            for state in active_states:
                candle = frame.bars[state.order.symbol]
                current_position = portfolio.position(state.order.symbol)
                current_quantity = current_position.quantity if current_position else ZERO
                attempt = self.executor.execute(
                    state.order,
                    candle,
                    remaining_quantity=state.remaining_quantity,
                    available_capacity=capacities[state.order.symbol],
                    current_position_quantity=current_quantity,
                )
                if attempt.status is OrderStatus.REJECTED:
                    state.status = OrderStatus.REJECTED
                    state.failure_reason = attempt.reason
                    continue
                if attempt.fill is None:
                    continue
                if attempt.costs is None:
                    raise RuntimeError("simulated fill is missing its cost breakdown")
                application = portfolio.apply_fill(attempt.fill)
                state.filled_quantity += attempt.fill.quantity
                state.fill_timestamps.append(attempt.fill.timestamp)
                state.status = attempt.status
                if capacities[state.order.symbol] is not None:
                    capacities[state.order.symbol] -= attempt.fill.quantity
                fill_records.append(
                    AppliedFillRecord(
                        fill=attempt.fill,
                        order_id=state.order.order_id,
                        costs=attempt.costs,
                        transition=application.transition,
                    )
                )

            marks = {
                symbol: frame.bars[symbol].close
                for symbol in portfolio.positions
                if symbol in frame.bars
            }
            if marks:
                portfolio.mark_many(marks, frame.clock.bar_close)

            marked_snapshot = portfolio.snapshot(frame.clock.observable_at)
            due_entries = self.financing_model.entries_between(
                previous_information_time,
                frame.clock.observable_at,
                marked_snapshot,
            )
            for entry in due_entries:
                if entry.timestamp > frame.clock.observable_at:
                    raise ValueError("financing entry cannot postdate the information clock")
                portfolio.apply_financing(entry)
                financing_entries.append(entry)

            snapshot = portfolio.snapshot(frame.clock.observable_at)
            history = panel.history_through(index)
            context = StrategyContext(
                clock=frame.clock,
                history=history,
                portfolio=snapshot,
            )
            intents = tuple(strategy.on_bar(context))
            for intent in intents:
                if not isinstance(intent, OrderIntent):
                    raise TypeError("strategy must return OrderIntent values")
                if intent.symbol not in self.required_symbols:
                    raise ValueError(f"strategy submitted unknown symbol {intent.symbol}")
                order = self._make_order(
                    intent,
                    order_id=f"sim-{next_order_number:08d}",
                    information_time=frame.clock.observable_at,
                    first_eligible_time=frame.clock.first_executable_at,
                )
                order_states.append(_OrderState(order=order))
                next_order_number += 1
            snapshots.append(snapshot)
            previous_information_time = frame.clock.observable_at

        for state in order_states:
            if state.active:
                state.status = OrderStatus.EXPIRED
                state.failure_reason = state.failure_reason or "end_of_replay"

        order_records = tuple(state.record() for state in order_states)
        return BacktestResult(
            snapshots=tuple(snapshots),
            orders=order_records,
            fills=tuple(fill_records),
            financing_entries=tuple(financing_entries),
            trade_groups=self._group_results(order_records, fill_records, financing_entries),
            data_report=panel.report,
        )

    @staticmethod
    def _make_order(
        intent: OrderIntent,
        *,
        order_id: str,
        information_time: datetime,
        first_eligible_time: datetime,
    ) -> Order:
        return Order(
            order_id=order_id,
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
            order_type=intent.order_type,
            signal_information_timestamp=information_time,
            signal_timestamp=information_time,
            submission_timestamp=information_time,
            first_eligible_execution_timestamp=first_eligible_time,
            limit_price=intent.limit_price,
            short_collateral=intent.short_collateral,
            trade_group_id=intent.trade_group_id,
            phase=intent.phase,
            client_order_id=intent.client_order_id,
            model_version=intent.model_version,
            model_fit_end_timestamp=intent.model_fit_end_timestamp,
            metadata=intent.metadata,
        )

    @staticmethod
    def _group_results(
        orders: tuple[OrderRecord, ...],
        fills: list[AppliedFillRecord],
        financing_entries: list[FinancingEntry],
    ) -> tuple[TradeGroupResult, ...]:
        grouped: dict[str, list[OrderRecord]] = {}
        for record in orders:
            group_id = record.order.trade_group_id
            if group_id is not None:
                grouped.setdefault(group_id, []).append(record)
        fills_by_order: dict[str, list[AppliedFillRecord]] = {}
        for record in fills:
            fills_by_order.setdefault(record.order_id, []).append(record)

        results: list[TradeGroupResult] = []
        for group_id, records in sorted(grouped.items()):
            all_fills = [
                fill for record in records for fill in fills_by_order.get(record.order.order_id, [])
            ]
            filled_legs = sum(record.status is OrderStatus.FILLED for record in records)
            failed_legs = sum(
                record.status in {OrderStatus.REJECTED, OrderStatus.EXPIRED} for record in records
            )
            partial_legs = sum(
                record.filled_quantity > ZERO and record.status is not OrderStatus.FILLED
                for record in records
            )
            if filled_legs == len(records):
                status = TradeGroupStatus.COMPLETE
            elif not all_fills and failed_legs:
                status = TradeGroupStatus.FAILED
            elif all_fills:
                status = TradeGroupStatus.PARTIAL
            else:
                status = TradeGroupStatus.PENDING
            entry_timestamp = BacktestEngine._phase_completion_timestamp(records, TradePhase.OPEN)
            exit_timestamp = BacktestEngine._phase_completion_timestamp(records, TradePhase.CLOSE)
            results.append(
                TradeGroupResult(
                    trade_group_id=group_id,
                    status=status,
                    order_ids=tuple(record.order.order_id for record in records),
                    intended_legs=len(records),
                    submitted_legs=len(records),
                    filled_legs=filled_legs,
                    failed_legs=failed_legs,
                    partial_legs=partial_legs,
                    first_eligible_execution_timestamp=min(
                        record.order.first_eligible_execution_timestamp for record in records
                    ),
                    combined_entry_timestamp=entry_timestamp,
                    combined_exit_timestamp=exit_timestamp,
                    total_fees=sum((fill.fill.fee for fill in all_fills), ZERO),
                    total_spread_cost=sum((fill.costs.spread_cost for fill in all_fills), ZERO),
                    total_slippage_cost=sum((fill.costs.slippage_cost for fill in all_fills), ZERO),
                    attributed_financing=sum(
                        (
                            entry.amount
                            for entry in financing_entries
                            if entry.exchange_metadata.get("trade_group_id") == group_id
                        ),
                        ZERO,
                    ),
                    aggregate_realized_gross_pnl=sum(
                        (fill.transition.realized_gross_pnl for fill in all_fills),
                        ZERO,
                    ),
                )
            )
        return tuple(results)

    @staticmethod
    def _phase_completion_timestamp(
        records: list[OrderRecord],
        phase: TradePhase,
    ) -> datetime | None:
        phase_records = [record for record in records if record.order.phase is phase]
        if not phase_records or any(
            record.status is not OrderStatus.FILLED for record in phase_records
        ):
            return None
        timestamps = [record.execution_timestamp for record in phase_records]
        if any(timestamp is None for timestamp in timestamps):
            return None
        return max(timestamp for timestamp in timestamps if timestamp is not None)
