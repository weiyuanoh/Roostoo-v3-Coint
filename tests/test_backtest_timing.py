from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from stat_arb_bot.backtest import (
    BacktestEngine,
    HistoricalPanel,
    OrderIntent,
    OrderStatus,
    OrderType,
    StrategyContext,
    execution_decimal,
)
from stat_arb_bot.domain import Candle, Side

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
SYMBOLS = ("BTC/USD", "ETH/USD")


def bar(
    index: int,
    symbol: str,
    *,
    open_price: float,
    close_price: float | None = None,
    low: float | None = None,
    high: float | None = None,
    complete: bool = True,
) -> Candle:
    close = open_price if close_price is None else close_price
    bar_low = min(open_price, close) if low is None else low
    bar_high = max(open_price, close) if high is None else high
    open_time = BASE + timedelta(hours=index)
    close_time = open_time + timedelta(hours=1) - timedelta(milliseconds=1)
    if not complete:
        close_time = open_time + timedelta(hours=2)
    return Candle(
        open_time=open_time,
        symbol=symbol,
        interval="1h",
        open=open_price,
        high=bar_high,
        low=bar_low,
        close=close,
        volume=100.0,
        close_time=close_time,
    )


class BuyAfterFirstBar:
    def reset(self) -> None:
        self.contexts: list[StrategyContext] = []

    def on_bar(self, context: StrategyContext) -> tuple[OrderIntent, ...]:
        self.contexts.append(context)
        if len(context.history.frames) == 1:
            return (OrderIntent("BTC/USD", Side.BUY, "1"),)
        return ()


def run_btc(strategy, candles: list[Candle]):
    return BacktestEngine(initial_cash="100000", required_symbols=("BTC/USD",)).run(
        strategy,
        candles,
        as_of=BASE + timedelta(days=1),
    )


def test_signal_bar_cannot_fill_itself_and_market_uses_next_open() -> None:
    strategy = BuyAfterFirstBar()
    candles = [
        bar(0, "BTC/USD", open_price=90, close_price=100),
        bar(1, "BTC/USD", open_price=110, close_price=111),
    ]

    result = run_btc(strategy, candles)

    assert len(result.fills) == 1
    fill = result.fills[0].fill
    order = result.orders[0].order
    assert fill.price == Decimal("110")
    assert fill.timestamp == candles[1].open_time
    assert fill.timestamp != candles[0].close_time
    assert order.signal_information_timestamp == candles[0].close_time
    assert order.first_eligible_execution_timestamp == candles[1].open_time
    assert order.signal_timestamp < fill.timestamp


def test_strategy_history_is_bounded_and_rolling_window_ends_at_current_bar() -> None:
    class WindowStrategy:
        def reset(self) -> None:
            self.windows: list[tuple[Candle, ...]] = []

        def on_bar(self, context: StrategyContext) -> tuple[()]:
            window = context.history.window("BTC/USD", 2)
            self.windows.append(window)
            assert max(item.close_time for item in window) == context.clock.observable_at
            assert context.history.max_observation_timestamp == context.clock.observable_at
            return ()

    strategy = WindowStrategy()
    candles = [bar(i, "BTC/USD", open_price=100 + i) for i in range(4)]
    run_btc(strategy, candles)

    assert [len(window) for window in strategy.windows] == [1, 2, 2, 2]
    assert [item.open_time for item in strategy.windows[-1]] == [
        candles[2].open_time,
        candles[3].open_time,
    ]


def test_future_data_changes_cannot_contaminate_signal_at_k() -> None:
    class StateStrategy:
        def reset(self) -> None:
            self.states: list[Decimal] = []

        def on_bar(self, context: StrategyContext) -> tuple[OrderIntent, ...]:
            state = sum(
                (
                    execution_decimal(item.close, name="close")
                    for item in context.history.window("BTC/USD", 2)
                ),
                Decimal("0"),
            )
            self.states.append(state)
            if len(context.history.frames) == 2:
                return (
                    OrderIntent(
                        "BTC/USD",
                        Side.BUY,
                        state / Decimal("100"),
                        model_version="dummy-v1",
                        model_fit_end_timestamp=context.clock.observable_at,
                    ),
                )
            return ()

    common = [
        bar(0, "BTC/USD", open_price=99, close_price=100),
        bar(1, "BTC/USD", open_price=100, close_price=102),
    ]
    data_a = common + [bar(2, "BTC/USD", open_price=103, close_price=104)]
    data_b = common + [bar(2, "BTC/USD", open_price=900, close_price=950)]
    strategy_a = StateStrategy()
    strategy_b = StateStrategy()

    result_a = run_btc(strategy_a, data_a)
    result_b = run_btc(strategy_b, data_b)

    assert strategy_a.states[:2] == strategy_b.states[:2]
    order_a = result_a.orders[0].order
    order_b = result_b.orders[0].order
    assert order_a.signal_information_timestamp == order_b.signal_information_timestamp
    assert order_a.quantity == order_b.quantity
    assert order_a.metadata == order_b.metadata


def test_synchronization_skips_missing_asset_timestamp_without_forward_fill() -> None:
    candles = [
        bar(0, "BTC/USD", open_price=100),
        bar(0, "ETH/USD", open_price=10),
        bar(1, "BTC/USD", open_price=101),
        bar(2, "BTC/USD", open_price=102),
        bar(2, "ETH/USD", open_price=12),
    ]
    panel = HistoricalPanel(
        candles,
        required_symbols=SYMBOLS,
        as_of=BASE + timedelta(days=1),
    )

    assert [frame.clock.bar_open for frame in panel.frames] == [
        BASE,
        BASE + timedelta(hours=2),
    ]
    assert panel.report.skipped_unsynchronized_timestamps == (BASE + timedelta(hours=1),)
    assert all(set(frame.bars) == set(SYMBOLS) for frame in panel.frames)


def test_forming_bar_is_excluded_from_strategy_information_set() -> None:
    strategy = BuyAfterFirstBar()
    completed = bar(0, "BTC/USD", open_price=100)
    forming = bar(1, "BTC/USD", open_price=101, complete=False)

    result = BacktestEngine(initial_cash="1000", required_symbols=("BTC/USD",)).run(
        strategy,
        [completed, forming],
        as_of=BASE + timedelta(hours=2),
    )

    assert len(strategy.contexts) == 1
    assert strategy.contexts[0].history.latest("BTC/USD") == completed
    assert result.data_report.excluded_forming_bars == 1
    assert result.orders[0].status is OrderStatus.EXPIRED


def test_model_metadata_chronology_is_enforced() -> None:
    class BadModelClock:
        def reset(self) -> None:
            pass

        def on_bar(self, context: StrategyContext) -> tuple[OrderIntent, ...]:
            return (
                OrderIntent(
                    "BTC/USD",
                    Side.BUY,
                    "1",
                    model_version="future-fit",
                    model_fit_end_timestamp=context.clock.observable_at + timedelta(seconds=1),
                ),
            )

    with pytest.raises(ValueError, match="model fit end"):
        run_btc(BadModelClock(), [bar(0, "BTC/USD", open_price=100)])


def test_limit_order_cannot_use_signal_bar_range() -> None:
    class LimitStrategy:
        def reset(self) -> None:
            pass

        def on_bar(self, context: StrategyContext) -> tuple[OrderIntent, ...]:
            if len(context.history.frames) == 1:
                return (
                    OrderIntent(
                        "BTC/USD",
                        Side.BUY,
                        "1",
                        order_type=OrderType.LIMIT,
                        limit_price="90",
                    ),
                )
            return ()

    candles = [
        bar(0, "BTC/USD", open_price=100, low=80, high=120),
        bar(1, "BTC/USD", open_price=100, low=95, high=110),
        bar(2, "BTC/USD", open_price=95, low=85, high=100),
    ]
    result = run_btc(LimitStrategy(), candles)

    assert len(result.fills) == 1
    assert result.fills[0].fill.timestamp == candles[2].close_time
    assert result.fills[0].fill.price == Decimal("90")


def test_decimal_conversion_boundary_uses_text_not_binary_float_expansion() -> None:
    assert execution_decimal(0.1, name="test price") == Decimal("0.1")


def test_repeated_run_resets_all_engine_and_strategy_state() -> None:
    engine = BacktestEngine(initial_cash="100000", required_symbols=("BTC/USD",))
    strategy = BuyAfterFirstBar()
    candles = [bar(0, "BTC/USD", open_price=100), bar(1, "BTC/USD", open_price=110)]

    first = engine.run(strategy, candles, as_of=BASE + timedelta(days=1))
    second = engine.run(strategy, candles, as_of=BASE + timedelta(days=1))

    assert first == second
    assert first.orders[0].order.order_id == "sim-00000001"
