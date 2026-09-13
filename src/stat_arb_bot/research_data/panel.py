"""Conservative synchronized five-asset model panels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite, log
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Iterable, Mapping

from stat_arb_bot.domain import Candle
from stat_arb_bot.domain.execution import utc_datetime
from stat_arb_bot.research_data.aggregation import MODEL_INTERVAL, MODEL_STEP
from stat_arb_bot.research_data.quality import RawDataConflictError
from stat_arb_bot.research_data.universe import ResearchUniverse

if TYPE_CHECKING:
    from stat_arb_bot.research_data.windows import ResearchWindow


@dataclass(frozen=True, slots=True)
class DroppedPanelTimestamp:
    timestamp: datetime
    missing_assets: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class PanelDiagnostics:
    received_rows: int
    synchronized_rows: int
    exact_duplicate_keys: tuple[tuple[str, datetime], ...]
    out_of_order: bool
    non_monotonic: bool
    forming_keys: tuple[tuple[str, datetime], ...]
    unexpected_symbols: tuple[str, ...]
    unexpected_intervals: tuple[str, ...]
    dropped_timestamps: tuple[DroppedPanelTimestamp, ...]
    missing_panel_timestamps: tuple[datetime, ...]

    @property
    def dropped_timestamp_count(self) -> int:
        return len(
            {item.timestamp for item in self.dropped_timestamps}
            | set(self.missing_panel_timestamps)
        )


@dataclass(frozen=True, slots=True)
class ResearchPanelRow:
    """One five-asset observation indexed by its observable close boundary."""

    timestamp: datetime
    open_time: datetime
    bars: Mapping[str, Candle]

    def __post_init__(self) -> None:
        timestamp = utc_datetime(self.timestamp)
        open_time = utc_datetime(self.open_time, name="open_time")
        if timestamp <= open_time:
            raise ValueError("panel timestamp must follow its open time")
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "open_time", open_time)
        object.__setattr__(self, "bars", MappingProxyType(dict(self.bars)))

    @property
    def observable_at(self) -> datetime:
        return self.timestamp


@dataclass(frozen=True, slots=True)
class ResearchMatrix:
    representation: str
    timestamps: tuple[datetime, ...]
    assets: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if len(self.timestamps) != len(self.values):
            raise ValueError("matrix timestamps and rows differ in length")
        if any(len(row) != len(self.assets) for row in self.values):
            raise ValueError("matrix row width differs from the universe")
        if any(not isfinite(value) for row in self.values for value in row):
            raise ValueError("matrix values must be finite")


class ResearchPanel:
    """An immutable chronological intersection of all five model-bar series."""

    __slots__ = ("diagnostics", "rows", "universe")

    def __init__(
        self,
        universe: ResearchUniverse,
        rows: Iterable[ResearchPanelRow],
        diagnostics: PanelDiagnostics,
    ) -> None:
        normalized_rows = tuple(rows)
        if any(
            right.timestamp <= left.timestamp
            for left, right in zip(normalized_rows, normalized_rows[1:])
        ):
            raise ValueError("panel rows must be strictly chronological")
        expected = set(universe.assets)
        if any(set(row.bars) != expected for row in normalized_rows):
            raise ValueError("each panel row must contain the complete five-asset universe")
        self.universe = universe
        self.rows = normalized_rows
        self.diagnostics = diagnostics

    @property
    def timestamps(self) -> tuple[datetime, ...]:
        return tuple(row.timestamp for row in self.rows)

    @property
    def start(self) -> datetime | None:
        return self.rows[0].timestamp if self.rows else None

    @property
    def end(self) -> datetime | None:
        return self.rows[-1].timestamp if self.rows else None

    def through(self, end: datetime) -> ResearchPanel:
        boundary = utc_datetime(end, name="end")
        rows = tuple(row for row in self.rows if row.timestamp <= boundary)
        diagnostics = PanelDiagnostics(
            received_rows=len(rows) * len(self.universe.assets),
            synchronized_rows=len(rows),
            exact_duplicate_keys=tuple(
                key for key in self.diagnostics.exact_duplicate_keys if key[1] <= boundary
            ),
            out_of_order=False,
            non_monotonic=False,
            forming_keys=(),
            unexpected_symbols=(),
            unexpected_intervals=(),
            dropped_timestamps=tuple(
                item for item in self.diagnostics.dropped_timestamps if item.timestamp <= boundary
            ),
            missing_panel_timestamps=tuple(
                timestamp
                for timestamp in self.diagnostics.missing_panel_timestamps
                if timestamp <= boundary
            ),
        )
        return ResearchPanel(
            self.universe,
            rows,
            diagnostics,
        )

    def close_prices(self) -> ResearchMatrix:
        return self._matrix("close", lambda candle: candle.close)

    def log_prices(self) -> ResearchMatrix:
        return self._matrix("log_price", lambda candle: log(candle.close))

    def log_returns(self) -> ResearchMatrix:
        levels = self.log_prices()
        return ResearchMatrix(
            representation="log_return",
            timestamps=levels.timestamps[1:],
            assets=levels.assets,
            values=tuple(
                tuple(current - previous for current, previous in zip(row, prior))
                for prior, row in zip(levels.values, levels.values[1:])
            ),
        )

    def _matrix(
        self,
        representation: str,
        transform: Callable[[Candle], float],
    ) -> ResearchMatrix:
        assets = self.universe.assets
        return ResearchMatrix(
            representation=representation,
            timestamps=self.timestamps,
            assets=assets,
            values=tuple(
                tuple(transform(row.bars[asset]) for asset in assets) for row in self.rows
            ),
        )

    @property
    def fingerprint(self) -> str:
        payload = {
            "assets": self.universe.assets,
            "quote_currency": self.universe.quote_currency,
            "interval": MODEL_INTERVAL,
            "rows": [
                {
                    "timestamp": row.timestamp.isoformat(),
                    "values": [
                        float(row.bars[asset].close).hex() for asset in self.universe.assets
                    ],
                }
                for row in self.rows
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def window(
        self,
        *,
        end: datetime,
        observations: int | None = None,
        start: datetime | None = None,
        lookback: timedelta | None = None,
        minimum_observations: int = 1,
        maximum_internal_gap: timedelta | None = MODEL_STEP,
    ) -> ResearchWindow:
        from stat_arb_bot.research_data.windows import build_research_window

        return build_research_window(
            self,
            end=end,
            observations=observations,
            start=start,
            lookback=lookback,
            minimum_observations=minimum_observations,
            maximum_internal_gap=maximum_internal_gap,
        )


def _payload(candle: Candle) -> tuple[object, ...]:
    return (
        candle.open_time.astimezone(timezone.utc),
        candle.close_time.astimezone(timezone.utc),
        candle.open,
        candle.high,
        candle.low,
        candle.close,
        candle.volume,
    )


def build_synchronized_panel(
    model_bars: Iterable[Candle],
    *,
    universe: ResearchUniverse,
    as_of: datetime,
) -> ResearchPanel:
    """Intersect exact 15-minute close timestamps without forward filling."""

    boundary = utc_datetime(as_of, name="as_of")
    received = tuple(model_bars)
    expected_pairs = set(universe.pairs)
    unexpected_symbols = tuple(
        sorted({bar.symbol for bar in received if bar.symbol.strip().upper() not in expected_pairs})
    )
    expected_rows = [bar for bar in received if bar.symbol.strip().upper() in expected_pairs]
    input_times_by_symbol: dict[str, list[datetime]] = {}
    for bar in expected_rows:
        input_times_by_symbol.setdefault(bar.symbol.strip().upper(), []).append(
            bar.close_time.astimezone(timezone.utc)
        )
    out_of_order = any(
        right < left
        for times in input_times_by_symbol.values()
        for left, right in zip(times, times[1:])
    )
    non_monotonic = any(
        right <= left
        for times in input_times_by_symbol.values()
        for left, right in zip(times, times[1:])
    )
    unexpected_intervals = tuple(
        sorted({bar.interval for bar in expected_rows if bar.interval != MODEL_INTERVAL})
    )
    candidates = [bar for bar in expected_rows if bar.interval == MODEL_INTERVAL]
    grouped: dict[tuple[str, datetime], list[Candle]] = {}
    for bar in candidates:
        key = (bar.symbol.strip().upper(), bar.close_time.astimezone(timezone.utc))
        grouped.setdefault(key, []).append(bar)

    exact_duplicates: list[tuple[str, datetime]] = []
    forming: list[tuple[str, datetime]] = []
    canonical: dict[tuple[str, datetime], Candle] = {}
    for key, group in sorted(grouped.items()):
        if len({_payload(bar) for bar in group}) > 1:
            raise RawDataConflictError(
                f"conflicting model bars for {key[0]} at {key[1].isoformat()}"
            )
        if len(group) > 1:
            exact_duplicates.append(key)
        selected = group[0]
        if selected.close_time > boundary:
            forming.append(key)
        else:
            canonical[key] = selected

    timestamps = sorted({timestamp for _, timestamp in canonical})
    rows: list[ResearchPanelRow] = []
    dropped: list[DroppedPanelTimestamp] = []
    for timestamp in timestamps:
        by_asset = {
            asset: canonical[(universe.pair(asset), timestamp)]
            for asset in universe.assets
            if (universe.pair(asset), timestamp) in canonical
        }
        missing = tuple(asset for asset in universe.assets if asset not in by_asset)
        if missing:
            dropped.append(DroppedPanelTimestamp(timestamp, missing, "missing_asset_bar"))
            continue
        open_times = {bar.open_time.astimezone(timezone.utc) for bar in by_asset.values()}
        expected_open = timestamp - MODEL_STEP
        if open_times != {expected_open}:
            dropped.append(DroppedPanelTimestamp(timestamp, (), "cross_symbol_interval_mismatch"))
            continue
        rows.append(ResearchPanelRow(timestamp, expected_open, by_asset))

    missing_panel_times: list[datetime] = []
    for left, right in zip(rows, rows[1:]):
        expected = left.timestamp + MODEL_STEP
        while expected < right.timestamp:
            missing_panel_times.append(expected)
            expected += MODEL_STEP

    diagnostics = PanelDiagnostics(
        received_rows=len(received),
        synchronized_rows=len(rows),
        exact_duplicate_keys=tuple(exact_duplicates),
        out_of_order=out_of_order,
        non_monotonic=non_monotonic,
        forming_keys=tuple(forming),
        unexpected_symbols=unexpected_symbols,
        unexpected_intervals=unexpected_intervals,
        dropped_timestamps=tuple(dropped),
        missing_panel_timestamps=tuple(missing_panel_times),
    )
    return ResearchPanel(universe, rows, diagnostics)
