from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from stat_arb_bot.market_data.store import CandleStore
from stat_arb_bot.market_data.validation import (
    CandleQualityError,
    inspect_candles,
    require_candle_quality,
)


def test_quality_report_detects_gap_duplicate_and_order(candle_factory) -> None:
    candles = [candle_factory(0), candle_factory(2), candle_factory(2)]

    report = inspect_candles(
        candles,
        interval="1h",
        as_of=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    assert report.valid is False
    assert report.out_of_order is True
    assert len(report.duplicate_open_times) == 1
    assert report.missing_open_times == (datetime(2026, 1, 1, 1, tzinfo=timezone.utc),)


def test_quality_report_detects_forming_and_stale(candle_factory) -> None:
    forming = inspect_candles(
        [candle_factory(0)],
        interval="1h",
        as_of=datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc),
    )
    stale = inspect_candles(
        [candle_factory(0)],
        interval="1h",
        as_of=datetime(2026, 1, 1, 4, tzinfo=timezone.utc),
        max_age_intervals=1,
    )

    assert forming.forming_open_times
    assert stale.stale is True


def test_require_quality_raises_actionable_error(candle_factory) -> None:
    with pytest.raises(CandleQualityError, match="missing bars"):
        require_candle_quality(
            [candle_factory(0), candle_factory(2)],
            interval="1h",
            as_of=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )


def test_empty_series_is_invalid() -> None:
    report = inspect_candles([], interval="1h")

    assert report.valid is False
    with pytest.raises(CandleQualityError, match="series is empty"):
        require_candle_quality([], interval="1h")


def test_store_round_trip_sorts_and_deduplicates(tmp_path: Path, candle_factory) -> None:
    store = CandleStore(tmp_path)
    changed = replace(candle_factory(0), close=100.5)

    path = store.write_csv("BTC/USD", "1h", [candle_factory(1), candle_factory(0), changed])
    loaded = store.read_csv("BTC/USD", "1h")

    assert path == tmp_path / "BTC_USD_1h.csv"
    assert [candle.open_time.hour for candle in loaded] == [0, 1]
    assert loaded[0].close == 100.5


def test_store_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid pair"):
        CandleStore(tmp_path).path_for("../../secret", "1h")
