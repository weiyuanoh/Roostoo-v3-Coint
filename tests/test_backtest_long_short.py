from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from stat_arb_bot.backtest import (
    BacktestEngine,
    CostParameters,
    ExecutionCostModel,
    FinancingRateEvent,
    OrderIntent,
    OrderStatus,
    OrderType,
    ScheduledShortFinancingModel,
    SimulatedExecutor,
    StrategyContext,
    TradeGroupStatus,
    TradePhase,
    VolumeCapacityFillPolicy,
)
from stat_arb_bot.domain import Candle, Side

BASE = datetime(2026, 2, 1, tzinfo=timezone.utc)
BTC = "BTC/USD"
ETH = "ETH/USD"


def candle(
    index: int,
    symbol: str,
    price_open: float,
    price_close: float | None = None,
    *,
    volume: float = 100.0,
) -> Candle:
    close = price_open if price_close is None else price_close
    open_time = BASE + timedelta(hours=index)
    return Candle(
        open_time=open_time,
        symbol=symbol,
        interval="1h",
        open=price_open,
        high=max(price_open, close),
        low=min(price_open, close),
        close=close,
        volume=volume,
        close_time=open_time + timedelta(hours=1) - timedelta(milliseconds=1),
    )


def paired_bars(prices: list[tuple[float, float, float, float]]) -> list[Candle]:
    result: list[Candle] = []
    for index, (btc_open, btc_close, eth_open, eth_close) in enumerate(prices):
        result.extend(
            [
                candle(index, BTC, btc_open, btc_close),
                candle(index, ETH, eth_open, eth_close),
            ]
        )
    return result


class OpenAndClosePair:
    def reset(self) -> None:
        pass

    def on_bar(self, context: StrategyContext) -> tuple[OrderIntent, ...]:
        frame_count = len(context.history.frames)
        if frame_count == 1:
            return (
                OrderIntent(BTC, Side.BUY, "1", trade_group_id="pair-1", phase=TradePhase.OPEN),
                OrderIntent(
                    ETH,
                    Side.SELL,
                    "10",
                    short_collateral="30000",
                    trade_group_id="pair-1",
                    phase=TradePhase.OPEN,
                ),
            )
        if frame_count == 3:
            return (
                OrderIntent(BTC, Side.SELL, "1", trade_group_id="pair-1", phase=TradePhase.CLOSE),
                OrderIntent(ETH, Side.BUY, "10", trade_group_id="pair-1", phase=TradePhase.CLOSE),
            )
        return ()


def test_long_short_group_end_to_end_with_costs_financing_and_delayed_closes() -> None:
    bars = paired_bars(
        [
            (49000, 49500, 3100, 3050),
            (50000, 50000, 3000, 3000),
            (51000, 52000, 2900, 2800),
            (52000, 52000, 2800, 2800),
        ]
    )
    event_time = bars[4].close_time
    financing = ScheduledShortFinancingModel(
        [
            FinancingRateEvent(
                timestamp=event_time,
                rate="0.001",
                reference="configured-test",
                trade_group_id="pair-1",
            )
        ]
    )
    executor = SimulatedExecutor(
        cost_model=ExecutionCostModel(default=CostParameters(taker_fee_bps="10"))
    )
    result = BacktestEngine(
        initial_cash="100000",
        required_symbols=(BTC, ETH),
        executor=executor,
        financing_model=financing,
    ).run(OpenAndClosePair(), bars, as_of=BASE + timedelta(days=1))

    entry_orders = result.orders[:2]
    assert all(order.order.signal_timestamp == bars[0].close_time for order in entry_orders)
    assert {record.fill.timestamp for record in result.fills[:2]} == {BASE + timedelta(hours=1)}
    assert result.fills[0].fill.price == Decimal("50000")
    assert result.fills[1].fill.price == Decimal("3000")

    marked = result.snapshots[2]
    assert marked.positions_by_symbol[BTC].quantity == Decimal("1")
    assert marked.positions_by_symbol[ETH].quantity == Decimal("-10")
    assert marked.positions_by_symbol[BTC].unrealized_pnl == Decimal("2000")
    assert marked.positions_by_symbol[ETH].unrealized_pnl == Decimal("2000")
    assert marked.positions_by_symbol[ETH].short_collateral == Decimal("30000")
    assert marked.long_market_value == Decimal("52000")
    assert marked.short_market_value == Decimal("-28000")
    assert marked.gross_market_value == Decimal("80000")
    assert marked.net_market_value == Decimal("24000")
    assert marked.cumulative_fees == Decimal("80")
    assert marked.financing == Decimal("-28.000")
    assert marked.equity == Decimal("103892.000")

    close_orders = result.orders[2:]
    assert all(order.order.signal_timestamp == bars[4].close_time for order in close_orders)
    assert {record.fill.timestamp for record in result.fills[2:]} == {BASE + timedelta(hours=3)}
    final = result.final_snapshot
    assert all(position.quantity == 0 for position in final.positions)
    assert final.realized_gross_pnl == Decimal("4000")
    assert final.unrealized_pnl == 0
    assert final.restricted_short_collateral == 0
    assert final.cumulative_fees == Decimal("160")
    assert final.financing == Decimal("-28.000")
    assert final.equity == Decimal("103812.000")

    group = result.trade_groups[0]
    assert group.status is TradeGroupStatus.COMPLETE
    assert group.intended_legs == group.submitted_legs == group.filled_legs == 4
    assert group.failed_legs == group.partial_legs == 0
    assert group.combined_entry_timestamp == BASE + timedelta(hours=1)
    assert group.combined_exit_timestamp == BASE + timedelta(hours=3)
    assert group.total_fees == Decimal("160")
    assert group.attributed_financing == Decimal("-28.000")
    assert group.aggregate_realized_gross_pnl == Decimal("4000")

    assert len(result.financing_entries) == 1
    assert result.financing_entries[0].amount == Decimal("-28.000")
    for fill_record in result.fills:
        order = next(
            item.order for item in result.orders if item.order.order_id == fill_record.order_id
        )
        assert (
            order.signal_information_timestamp
            <= order.signal_timestamp
            < fill_record.fill.timestamp
        )


class OpenPairOnly:
    def reset(self) -> None:
        pass

    def on_bar(self, context: StrategyContext) -> tuple[OrderIntent, ...]:
        if len(context.history.frames) != 1:
            return ()
        return (
            OrderIntent(BTC, Side.BUY, "1", trade_group_id="pair", phase=TradePhase.OPEN),
            OrderIntent(
                ETH,
                Side.SELL,
                "10",
                short_collateral="30000",
                trade_group_id="pair",
                phase=TradePhase.OPEN,
            ),
        )


def run_failed_open(rejected_symbol: str):
    def reject(order, _candle):
        return "configured_leg_failure" if order.symbol == rejected_symbol else None

    bars = paired_bars([(49000, 49000, 3100, 3100), (50000, 50000, 3000, 3000)])
    return BacktestEngine(
        initial_cash="100000",
        required_symbols=(BTC, ETH),
        executor=SimulatedExecutor(rejection_policy=reject),
    ).run(OpenPairOnly(), bars, as_of=BASE + timedelta(days=1))


def test_btc_long_fills_while_eth_short_fails() -> None:
    result = run_failed_open(ETH)

    assert result.final_snapshot.positions_by_symbol[BTC].quantity == Decimal("1")
    assert ETH not in result.final_snapshot.positions_by_symbol
    assert result.trade_groups[0].status is TradeGroupStatus.PARTIAL
    assert result.trade_groups[0].filled_legs == 1
    assert result.trade_groups[0].failed_legs == 1


def test_eth_short_fills_while_btc_long_fails() -> None:
    result = run_failed_open(BTC)

    assert BTC not in result.final_snapshot.positions_by_symbol
    assert result.final_snapshot.positions_by_symbol[ETH].quantity == Decimal("-10")
    assert result.final_snapshot.positions_by_symbol[ETH].short_collateral == Decimal("30000")
    assert result.trade_groups[0].status is TradeGroupStatus.PARTIAL


def test_capacity_limited_partial_fill_is_not_synthetically_completed() -> None:
    class SingleLeg:
        def reset(self) -> None:
            pass

        def on_bar(self, context: StrategyContext) -> tuple[OrderIntent, ...]:
            if len(context.history.frames) == 1:
                return (OrderIntent(BTC, Side.BUY, "1", trade_group_id="partial"),)
            return ()

    bars = [candle(0, BTC, 100, volume=1), candle(1, BTC, 100, volume=1)]
    result = BacktestEngine(
        initial_cash="1000",
        required_symbols=(BTC,),
        executor=SimulatedExecutor(
            fill_policy=VolumeCapacityFillPolicy(fraction_of_bar_volume="0.5")
        ),
    ).run(SingleLeg(), bars, as_of=BASE + timedelta(days=1))

    assert result.final_snapshot.positions_by_symbol[BTC].quantity == Decimal("0.5")
    assert result.orders[0].status is OrderStatus.EXPIRED
    assert result.orders[0].filled_quantity == Decimal("0.5")
    assert result.trade_groups[0].status is TradeGroupStatus.PARTIAL
    assert result.trade_groups[0].partial_legs == 1


def test_one_close_leg_failure_leaves_real_exposure_visible() -> None:
    def reject_eth_close(order, _candle):
        if order.symbol == ETH and order.phase is TradePhase.CLOSE:
            return "configured_close_failure"
        return None

    bars = paired_bars(
        [
            (49000, 49000, 3100, 3100),
            (50000, 50000, 3000, 3000),
            (52000, 52000, 2800, 2800),
            (52000, 52000, 2800, 2800),
        ]
    )
    result = BacktestEngine(
        initial_cash="100000",
        required_symbols=(BTC, ETH),
        executor=SimulatedExecutor(rejection_policy=reject_eth_close),
    ).run(OpenAndClosePair(), bars, as_of=BASE + timedelta(days=1))

    final = result.final_snapshot
    assert final.positions_by_symbol[BTC].quantity == 0
    assert final.positions_by_symbol[ETH].quantity == Decimal("-10")
    assert final.positions_by_symbol[ETH].unrealized_pnl == Decimal("2000")
    assert result.trade_groups[0].status is TradeGroupStatus.PARTIAL
    assert result.trade_groups[0].failed_legs == 1


def test_spread_slippage_and_fee_components_are_separately_observable() -> None:
    executor = SimulatedExecutor(
        cost_model=ExecutionCostModel(
            default=CostParameters(
                taker_fee_bps="10",
                spread_bps="20",
                slippage_bps="10",
                fixed_slippage="0.05",
            )
        )
    )
    bars = [candle(0, BTC, 90), candle(1, BTC, 100)]
    result = BacktestEngine(
        initial_cash="1000",
        required_symbols=(BTC,),
        executor=executor,
    ).run(
        _SingleMarketBuy(),
        bars,
        as_of=BASE + timedelta(days=1),
    )

    record = result.fills[0]
    assert record.fill.price == Decimal("100.25")
    assert record.costs.spread_cost == Decimal("0.100")
    assert record.costs.slippage_cost == Decimal("0.150")
    assert record.costs.trading_fee == Decimal("0.10025")
    assert record.fill.fee == Decimal("0.10025")


def test_maker_and_symbol_specific_fee_override_apply_to_limit_fill() -> None:
    class LimitBuy:
        def reset(self) -> None:
            pass

        def on_bar(self, context: StrategyContext) -> tuple[OrderIntent, ...]:
            if len(context.history.frames) == 1:
                return (
                    OrderIntent(
                        BTC,
                        Side.BUY,
                        "1",
                        order_type=OrderType.LIMIT,
                        limit_price="100",
                    ),
                )
            return ()

    first = candle(0, BTC, 90)
    second = Candle(
        open_time=BASE + timedelta(hours=1),
        symbol=BTC,
        interval="1h",
        open=105,
        high=106,
        low=99,
        close=101,
        volume=100,
        close_time=BASE + timedelta(hours=2) - timedelta(milliseconds=1),
    )
    executor = SimulatedExecutor(
        cost_model=ExecutionCostModel(
            default=CostParameters(maker_fee_bps="10"),
            symbol_overrides={BTC: CostParameters(maker_fee_bps="50")},
        )
    )
    result = BacktestEngine(
        initial_cash="1000",
        required_symbols=(BTC,),
        executor=executor,
    ).run(LimitBuy(), [first, second], as_of=BASE + timedelta(days=1))

    assert result.fills[0].fill.timestamp == second.close_time
    assert result.fills[0].fill.price == Decimal("100")
    assert result.fills[0].costs.trading_fee == Decimal("0.5")
    assert result.fills[0].fill.fee == Decimal("0.5")


def test_short_fee_overrides_are_not_collapsed_into_ordinary_fee() -> None:
    class ShortOnly:
        def reset(self) -> None:
            pass

        def on_bar(self, context: StrategyContext) -> tuple[OrderIntent, ...]:
            if len(context.history.frames) == 1:
                return (OrderIntent(ETH, Side.SELL, "10", short_collateral="30000"),)
            if len(context.history.frames) == 2:
                return (OrderIntent(ETH, Side.BUY, "10"),)
            return ()

    executor = SimulatedExecutor(
        cost_model=ExecutionCostModel(
            default=CostParameters(
                taker_fee_bps="10",
                short_open_fee_bps="20",
                short_close_fee_bps="30",
            )
        )
    )
    bars = [candle(0, ETH, 3100), candle(1, ETH, 3000), candle(2, ETH, 2900)]
    result = BacktestEngine(
        initial_cash="100000",
        required_symbols=(ETH,),
        executor=executor,
    ).run(ShortOnly(), bars, as_of=BASE + timedelta(days=1))

    assert result.fills[0].costs.trading_fee == 0
    assert result.fills[0].costs.short_open_fee == Decimal("60")
    assert result.fills[0].fill.fee == Decimal("60")
    assert result.fills[1].costs.trading_fee == 0
    assert result.fills[1].costs.short_close_fee == Decimal("87")
    assert result.fills[1].fill.fee == Decimal("87")


class _SingleMarketBuy:
    def reset(self) -> None:
        pass

    def on_bar(self, context: StrategyContext) -> tuple[OrderIntent, ...]:
        if len(context.history.frames) == 1:
            return (OrderIntent(BTC, Side.BUY, "1"),)
        return ()
