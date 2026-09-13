from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from math import log

import pytest

from stat_arb_bot.domain import Candle
from stat_arb_bot.research_data import (
    ModelFitMetadata,
    ResearchUniverse,
    WindowIssue,
    build_synchronized_panel,
)

BASE = datetime(2026, 4, 1, tzinfo=timezone.utc)
UNIVERSE = ResearchUniverse("USDT")


def model_bar(asset: str, index: int, close: float | None = None) -> Candle:
    open_time = BASE + timedelta(minutes=15 * index)
    price = float(100 + UNIVERSE.assets.index(asset) * 10 + index)
    final = price if close is None else close
    return Candle(
        open_time=open_time,
        close_time=open_time + timedelta(minutes=15),
        symbol=UNIVERSE.pair(asset),
        interval="15m",
        open=price,
        high=max(price, final) + 1,
        low=min(price, final) - 1,
        close=final,
        volume=100,
        source="DERIVED_BINANCE_5M",
        retrieved_at=BASE + timedelta(days=1),
    )


def model_set(indices: range | tuple[int, ...]) -> list[Candle]:
    return [model_bar(asset, index) for index in indices for asset in UNIVERSE.assets]


def test_complete_five_asset_timestamp_is_included_in_asset_order() -> None:
    panel = build_synchronized_panel(
        reversed(model_set((0,))),
        universe=UNIVERSE,
        as_of=BASE + timedelta(minutes=15),
    )

    assert panel.timestamps == (BASE + timedelta(minutes=15),)
    assert tuple(panel.rows[0].bars) == UNIVERSE.assets
    assert panel.rows[0].observable_at == BASE + timedelta(minutes=15)


def test_missing_sol_drops_timestamp_and_never_forward_fills() -> None:
    bars = model_set((0, 1))
    bars = [bar for bar in bars if not (bar.symbol == "SOL/USDT" and bar.open_time > BASE)]

    panel = build_synchronized_panel(
        bars,
        universe=UNIVERSE,
        as_of=BASE + timedelta(minutes=30),
    )

    assert panel.timestamps == (BASE + timedelta(minutes=15),)
    dropped = panel.diagnostics.dropped_timestamps[0]
    assert dropped.timestamp == BASE + timedelta(minutes=30)
    assert dropped.missing_assets == ("SOL",)
    assert all(row.bars["SOL"].open_time == row.open_time for row in panel.rows)


def test_future_model_rows_are_excluded_by_as_of_and_order_is_chronological() -> None:
    panel = build_synchronized_panel(
        reversed(model_set((0, 1))),
        universe=UNIVERSE,
        as_of=BASE + timedelta(minutes=15),
    )

    assert panel.timestamps == (BASE + timedelta(minutes=15),)
    assert len(panel.diagnostics.forming_keys) == 5
    assert all(key[1] == BASE + timedelta(minutes=30) for key in panel.diagnostics.forming_keys)
    assert panel.diagnostics.out_of_order is True
    assert panel.diagnostics.non_monotonic is True


def test_cross_symbol_interval_mismatch_is_dropped_explicitly() -> None:
    bars = model_set((0,))
    sol_index = next(index for index, bar in enumerate(bars) if bar.symbol == "SOL/USDT")
    bars[sol_index] = replace(bars[sol_index], open_time=BASE + timedelta(minutes=1))

    panel = build_synchronized_panel(
        bars,
        universe=UNIVERSE,
        as_of=BASE + timedelta(minutes=15),
    )

    assert panel.rows == ()
    assert panel.diagnostics.dropped_timestamps[0].reason == "cross_symbol_interval_mismatch"


def test_raw_log_price_and_log_return_representations_are_deterministic() -> None:
    bars = model_set((0, 1))
    panel = build_synchronized_panel(
        bars,
        universe=UNIVERSE,
        as_of=BASE + timedelta(minutes=30),
    )

    prices = panel.close_prices()
    levels = panel.log_prices()
    returns = panel.log_returns()

    assert prices.assets == UNIVERSE.assets
    assert prices.values[0][0] == 100
    assert levels.values[0][0] == log(100)
    assert returns.timestamps == (BASE + timedelta(minutes=30),)
    assert returns.values[0][0] == pytest.approx(log(101) - log(100))


def test_observation_window_ends_exactly_at_k_and_hides_future_rows() -> None:
    panel = build_synchronized_panel(
        model_set(range(5)),
        universe=UNIVERSE,
        as_of=BASE + timedelta(minutes=75),
    )
    end = BASE + timedelta(minutes=60)

    window = panel.window(end=end, observations=3, minimum_observations=3)

    assert window.validation.valid
    assert window.actual_observations == 3
    assert window.timestamps[-1] == end
    assert all(timestamp <= end for timestamp in window.timestamps)
    assert BASE + timedelta(minutes=75) not in window.timestamps


def test_calendar_window_reports_requested_interval_and_actual_count() -> None:
    panel = build_synchronized_panel(
        model_set(range(4)),
        universe=UNIVERSE,
        as_of=BASE + timedelta(minutes=60),
    )
    end = BASE + timedelta(minutes=60)

    window = panel.window(end=end, lookback=timedelta(minutes=30), minimum_observations=2)

    assert window.requested_start == BASE + timedelta(minutes=30)
    assert window.requested_end == end
    assert window.timestamps == (
        BASE + timedelta(minutes=45),
        BASE + timedelta(minutes=60),
    )
    assert window.actual_observations == 2


def test_insufficient_history_and_internal_gap_are_structured_issues() -> None:
    sparse = build_synchronized_panel(
        model_set((0, 2)),
        universe=UNIVERSE,
        as_of=BASE + timedelta(minutes=45),
    )

    window = sparse.window(
        end=BASE + timedelta(minutes=45),
        observations=3,
        minimum_observations=3,
        maximum_internal_gap=timedelta(minutes=15),
    )

    assert window.validation.valid is False
    assert WindowIssue.INSUFFICIENT_OBSERVATIONS in window.validation.issues
    assert WindowIssue.INTERNAL_GAP_TOO_LARGE in window.validation.issues
    assert window.validation.actual_observations == 2
    assert sparse.diagnostics.missing_panel_timestamps == (BASE + timedelta(minutes=30),)


def test_model_fit_metadata_preserves_data_chronology_and_provenance() -> None:
    panel = build_synchronized_panel(
        model_set(range(3)),
        universe=UNIVERSE,
        as_of=BASE + timedelta(minutes=45),
    )
    window = panel.window(
        end=BASE + timedelta(minutes=45),
        observations=3,
        minimum_observations=3,
    )

    metadata = ModelFitMetadata.from_window(
        window,
        model_name="future-structural-model",
        model_version="spec-v1",
        fit_id="fit-0001",
        fit_timestamp=BASE + timedelta(minutes=45),
        source_data_fingerprint="a" * 64,
    )

    assert metadata.data_end_timestamp == BASE + timedelta(minutes=45)
    assert metadata.data_end_timestamp <= metadata.fit_timestamp
    assert metadata.observation_count == 3
    assert metadata.universe == UNIVERSE.assets
    assert metadata.order_chronology_fields()["model_fit_end_timestamp"] == window.data_end

    with pytest.raises(ValueError, match="cannot precede"):
        ModelFitMetadata(
            model_name="bad",
            model_version="v1",
            fit_id="bad-fit",
            fit_timestamp=BASE,
            data_start_timestamp=BASE,
            data_end_timestamp=BASE + timedelta(minutes=15),
            bar_interval="15m",
            universe=UNIVERSE.assets,
            quote_currency="USDT",
            observation_count=1,
            source_data_fingerprint="b" * 64,
        )
