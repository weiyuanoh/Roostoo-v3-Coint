from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from stat_arb_bot.backtest import BacktestEngine, OrderIntent, StrategyContext
from stat_arb_bot.domain import Candle, Side
from stat_arb_bot.research_data import (
    RawDataConflictError,
    ResearchDataBuilder,
    ResearchDataStore,
    ResearchHistoryLoader,
    ResearchUniverse,
    summarize_coverage,
)

BASE = datetime(2026, 5, 1, tzinfo=timezone.utc)
UNIVERSE = ResearchUniverse()


def raw_bar(asset: str, index: int, *, price_shift: float = 0) -> Candle:
    asset_offset = UNIVERSE.assets.index(asset) * 10
    price = 100.0 + asset_offset + index + price_shift
    open_time = BASE + timedelta(minutes=5 * index)
    return Candle(
        open_time=open_time,
        close_time=open_time + timedelta(minutes=5) - timedelta(milliseconds=1),
        symbol=UNIVERSE.pair(asset),
        interval="5m",
        open=price,
        high=price + 2,
        low=price - 1,
        close=price + 1,
        volume=10 + index,
        source="BINANCE",
        retrieved_at=BASE + timedelta(days=1),
    )


def raw_dataset(
    count: int, *, future_shift_after: int | None = None
) -> dict[str, tuple[Candle, ...]]:
    return {
        asset: tuple(
            raw_bar(
                asset,
                index,
                price_shift=500
                if future_shift_after is not None and index >= future_shift_after
                else 0,
            )
            for index in range(count)
        )
        for asset in UNIVERSE.assets
    }


def test_future_contamination_does_not_change_panel_window_or_transforms_through_k() -> None:
    data_a = raw_dataset(9)
    data_b = raw_dataset(9, future_shift_after=6)
    builder = ResearchDataBuilder(UNIVERSE)
    as_of = BASE + timedelta(minutes=45)
    dataset_a = builder.build(data_a, as_of=as_of, requested_start=BASE)
    dataset_b = builder.build(data_b, as_of=as_of, requested_start=BASE)
    end_k = BASE + timedelta(minutes=30)

    panel_a = dataset_a.panel.through(end_k)
    panel_b = dataset_b.panel.through(end_k)
    window_a = dataset_a.panel.window(end=end_k, observations=2, minimum_observations=2)
    window_b = dataset_b.panel.window(end=end_k, observations=2, minimum_observations=2)

    assert panel_a.close_prices() == panel_b.close_prices()
    assert panel_a.log_prices() == panel_b.log_prices()
    assert panel_a.log_returns() == panel_b.log_returns()
    assert window_a.close_prices() == window_b.close_prices()
    assert window_a.log_prices() == window_b.log_prices()
    assert window_a.log_returns() == window_b.log_returns()
    assert dataset_a.source_fingerprint_through(end_k) == dataset_b.source_fingerprint_through(
        end_k
    )
    assert panel_a.diagnostics == panel_b.diagnostics
    assert all(timestamp <= end_k for timestamp in window_a.timestamps)


def test_incremental_overlap_matches_clean_full_reconstruction() -> None:
    complete = raw_dataset(6)
    bootstrap = {asset: rows[:4] for asset, rows in complete.items()}
    incoming = {asset: rows[3:] for asset, rows in complete.items()}
    builder = ResearchDataBuilder(UNIVERSE)
    initial = builder.build(
        bootstrap,
        as_of=BASE + timedelta(minutes=20),
        requested_start=BASE,
    )

    incremental = builder.incremental_update(
        initial,
        incoming,
        as_of=BASE + timedelta(minutes=30),
    )
    full = builder.build(
        complete,
        as_of=BASE + timedelta(minutes=30),
        requested_start=BASE,
    )

    assert incremental.source_fingerprint == full.source_fingerprint
    assert incremental.panel.fingerprint == full.panel.fingerprint
    assert incremental.panel.close_prices() == full.panel.close_prices()
    for asset in UNIVERSE.assets:
        assert incremental.raw_by_asset[asset] == full.raw_by_asset[asset]
        assert incremental.aggregation_by_asset[asset].bars == full.aggregation_by_asset[asset].bars
        assert incremental.affected_model_buckets[asset] == (BASE + timedelta(minutes=15),)


def test_atomic_research_storage_keeps_raw_derived_and_manifest_separate(tmp_path: Path) -> None:
    dataset = ResearchDataBuilder(UNIVERSE).build(
        raw_dataset(6),
        as_of=BASE + timedelta(minutes=30),
        requested_start=BASE,
    )
    store = ResearchDataStore(tmp_path)

    paths = store.write_dataset(dataset)
    loaded = store.load_raw(UNIVERSE)
    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))

    assert all("/raw/USDT/" in str(path) for path in paths.raw_paths)
    assert all("/derived/USDT/" in str(path) for path in paths.derived_paths)
    assert manifest["source_fingerprint"] == dataset.source_fingerprint
    assert manifest["panel_fingerprint"] == dataset.panel.fingerprint
    assert manifest["binance_symbols"]["BTC"] == "BTCUSDT"
    assert loaded == dict(dataset.raw_by_asset)
    rebuilt = ResearchDataBuilder(UNIVERSE).build(
        loaded,
        as_of=BASE + timedelta(minutes=30),
        requested_start=BASE,
    )
    assert rebuilt.source_fingerprint == dataset.source_fingerprint
    assert rebuilt.panel.fingerprint == dataset.panel.fingerprint


def test_store_append_is_idempotent_and_surfaces_conflicting_overlap(tmp_path: Path) -> None:
    store = ResearchDataStore(tmp_path)
    original = raw_bar("BTC", 0)
    first = store.append_raw(UNIVERSE, "BTC", [original], as_of=BASE + timedelta(hours=1))
    second = store.append_raw(
        UNIVERSE,
        "BTC",
        [replace(original, retrieved_at=BASE + timedelta(days=2))],
        as_of=BASE + timedelta(hours=1),
    )

    assert len(first.candles) == len(second.candles) == 1
    assert second.diagnostics.exact_duplicate_open_times == (BASE,)
    with pytest.raises(RawDataConflictError):
        store.append_raw(
            UNIVERSE,
            "BTC",
            [replace(original, close=original.close + 0.5)],
            as_of=BASE + timedelta(hours=1),
        )


def test_coverage_report_exposes_per_asset_and_common_panel_counts() -> None:
    dataset = ResearchDataBuilder(UNIVERSE).build(
        raw_dataset(6),
        as_of=BASE + timedelta(minutes=30),
    )

    coverage = summarize_coverage(dataset)

    assert coverage.quote_currency == "USDT"
    assert coverage.synchronized_rows == 2
    assert coverage.common_start == BASE + timedelta(minutes=15)
    assert coverage.common_end == BASE + timedelta(minutes=30)
    assert all(item.raw_5m_rows == 6 for item in coverage.assets)
    assert all(item.valid_15m_rows == 2 for item in coverage.assets)


def test_builder_preserves_raw_forming_and_stale_diagnostics_after_exclusion() -> None:
    dataset = ResearchDataBuilder(UNIVERSE).build(
        raw_dataset(3),
        as_of=BASE + timedelta(minutes=10),
        max_age_intervals=0,
    )

    coverage = summarize_coverage(dataset)
    assert all(item.raw_5m_rows == 2 for item in coverage.assets)
    assert all(item.forming_5m_rows == 1 for item in coverage.assets)
    assert all(item.valid_15m_rows == 0 for item in coverage.assets)


def test_common_quote_universe_has_explicit_binance_mapping_and_rejects_mixing() -> None:
    assert UNIVERSE.pairs == (
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "XRP/USDT",
        "ADA/USDT",
    )
    assert tuple(UNIVERSE.binance_symbols.values()) == (
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "ADAUSDT",
    )
    with pytest.raises(ValueError, match="common-quote"):
        UNIVERSE.asset_for_pair("ETH/USDC")


class OneSignal:
    def reset(self) -> None:
        self.information_times: list[datetime] = []

    def on_bar(self, context: StrategyContext) -> tuple[OrderIntent, ...]:
        self.information_times.append(context.clock.observable_at)
        if len(context.history.frames) == 1:
            return (OrderIntent("BTC/USDT", Side.BUY, "1"),)
        return ()


def test_research_timestamp_and_backtest_information_clock_have_no_bar_offset() -> None:
    dataset = ResearchDataBuilder(UNIVERSE).build(
        raw_dataset(6),
        as_of=BASE + timedelta(minutes=30),
    )
    model_bars = [
        bar for asset in UNIVERSE.assets for bar in dataset.aggregation_by_asset[asset].bars
    ]
    strategy = OneSignal()

    result = BacktestEngine(
        initial_cash="100000",
        required_symbols=UNIVERSE.pairs,
    ).run(strategy, model_bars, as_of=BASE + timedelta(minutes=30))

    first_panel_time = dataset.panel.timestamps[0]
    assert strategy.information_times[0] == first_panel_time
    assert result.orders[0].order.signal_information_timestamp == first_panel_time
    assert result.orders[0].order.first_eligible_execution_timestamp == (
        first_panel_time + timedelta(microseconds=1)
    )
    assert result.fills[0].fill.timestamp > first_panel_time
    assert result.fills[0].fill.price == dataset.aggregation_by_asset["BTC"].bars[1].open


def test_bounded_history_loader_uses_common_quote_pairs_and_filters_interval() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.pairs: list[str] = []

        def fetch_klines_paginated(self, pair: str, **_kwargs):
            self.pairs.append(pair)
            asset = pair.split("/")[0]
            return [raw_bar(asset, 0), raw_bar(asset, 1)]

    client = FakeClient()
    loader = ResearchHistoryLoader(client)  # type: ignore[arg-type]

    loaded = loader.bootstrap(
        UNIVERSE,
        start=BASE,
        end=BASE + timedelta(minutes=10),
        page_delay=0,
    )

    assert client.pairs == list(UNIVERSE.pairs)
    assert all(len(rows) == 2 for rows in loaded.values())
