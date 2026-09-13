from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from stat_arb_bot.domain import Candle
from stat_arb_bot.research_data import (
    RawDataConflictError,
    aggregate_five_to_fifteen,
    prepare_raw_series,
)

BASE = datetime(2026, 3, 1, tzinfo=timezone.utc)
PAIR = "BTC/USDT"


def raw_bar(
    index: int,
    *,
    open_price: float = 100,
    high: float = 102,
    low: float = 99,
    close: float = 101,
    volume: float = 10,
    offset: timedelta = timedelta(0),
    retrieved_offset: int = 1,
) -> Candle:
    open_time = BASE + timedelta(minutes=5 * index) + offset
    return Candle(
        open_time=open_time,
        close_time=open_time + timedelta(minutes=5) - timedelta(milliseconds=1),
        symbol=PAIR,
        interval="5m",
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        source="BINANCE",
        retrieved_at=BASE + timedelta(hours=retrieved_offset),
    )


def test_three_raw_bars_aggregate_exact_ohlcv_and_utc_boundaries() -> None:
    bars = [
        raw_bar(0, open_price=100, high=103, low=99, close=102, volume=2),
        raw_bar(1, open_price=102, high=106, low=101, close=104, volume=3),
        raw_bar(2, open_price=104, high=105, low=98, close=100, volume=5),
    ]

    result = aggregate_five_to_fifteen(
        bars,
        symbol=PAIR,
        as_of=BASE + timedelta(minutes=15),
    )

    assert len(result.bars) == 1
    model = result.bars[0]
    assert model.open_time == BASE
    assert model.close_time == BASE + timedelta(minutes=15)
    assert model.open == 100
    assert model.high == 106
    assert model.low == 98
    assert model.close == 100
    assert model.volume == 10
    assert model.interval == "15m"
    assert model.source == "DERIVED_BINANCE_5M"


def test_missing_middle_constituent_excludes_entire_model_bar() -> None:
    result = aggregate_five_to_fifteen(
        [raw_bar(0), raw_bar(2)],
        symbol=PAIR,
        as_of=BASE + timedelta(minutes=15),
    )

    assert result.bars == ()
    assert result.diagnostics.incomplete_bucket_opens == (BASE,)
    assert result.diagnostics.raw_quality.missing_open_times == (BASE + timedelta(minutes=5),)


def test_two_completed_constituents_do_not_create_partial_trailing_bar() -> None:
    result = aggregate_five_to_fifteen(
        [raw_bar(0), raw_bar(1)],
        symbol=PAIR,
        as_of=BASE + timedelta(minutes=15),
    )

    assert result.bars == ()
    assert result.diagnostics.excluded_model_bars == 1


def test_misaligned_raw_observation_is_reported_and_not_relabelled() -> None:
    misaligned = raw_bar(0, offset=timedelta(minutes=1))
    result = aggregate_five_to_fifteen(
        [misaligned, raw_bar(1), raw_bar(2)],
        symbol=PAIR,
        as_of=BASE + timedelta(minutes=15),
    )

    assert result.bars == ()
    assert result.diagnostics.misaligned_raw_opens == (BASE + timedelta(minutes=1),)


def test_exact_duplicate_is_idempotent_but_conflicting_overlap_raises() -> None:
    bars = [raw_bar(0), raw_bar(1), raw_bar(2)]
    duplicate = replace(bars[1], retrieved_at=BASE + timedelta(hours=2))

    result = aggregate_five_to_fifteen(
        [bars[0], bars[1], duplicate, bars[2]],
        symbol=PAIR,
        as_of=BASE + timedelta(minutes=15),
    )
    assert len(result.bars) == 1
    assert result.diagnostics.raw_quality.exact_duplicate_open_times == (
        BASE + timedelta(minutes=5),
    )

    conflict = replace(bars[1], close=101.5)
    with pytest.raises(RawDataConflictError, match="conflicting observations"):
        aggregate_five_to_fifteen(
            [*bars, conflict],
            symbol=PAIR,
            as_of=BASE + timedelta(minutes=15),
        )


def test_forming_rows_are_excluded_and_quality_reports_order_gap_and_staleness() -> None:
    rows = [raw_bar(3), raw_bar(0), raw_bar(1)]
    result = prepare_raw_series(
        rows,
        symbol=PAIR,
        as_of=BASE + timedelta(minutes=14),
        max_age_intervals=1,
        large_gap_threshold_intervals=2,
    )

    assert result.diagnostics.out_of_order is True
    assert result.diagnostics.non_monotonic is True
    assert result.diagnostics.forming_open_times == (BASE + timedelta(minutes=15),)
    assert result.diagnostics.stale is False
    assert [bar.open_time for bar in result.candles] == [BASE, BASE + timedelta(minutes=5)]


def test_large_continuity_gap_and_stale_series_are_explicit() -> None:
    result = prepare_raw_series(
        [raw_bar(0), raw_bar(4)],
        symbol=PAIR,
        as_of=BASE + timedelta(hours=1),
        max_age_intervals=2,
        large_gap_threshold_intervals=2,
    )

    assert result.diagnostics.stale is True
    assert result.diagnostics.large_gaps[0].missing_intervals == 3
    assert len(result.diagnostics.missing_open_times) == 3


def test_malformed_raw_interval_duration_is_excluded_explicitly() -> None:
    malformed = replace(raw_bar(0), close_time=BASE + timedelta(minutes=4))

    result = prepare_raw_series(
        [malformed],
        symbol=PAIR,
        as_of=BASE + timedelta(minutes=10),
    )

    assert result.candles == ()
    assert result.diagnostics.invalid_open_times == (BASE,)
