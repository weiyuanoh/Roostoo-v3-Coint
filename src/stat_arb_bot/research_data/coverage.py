"""Structured raw/model/panel coverage diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from stat_arb_bot.research_data.pipeline import ResearchDataset


@dataclass(frozen=True, slots=True)
class AssetCoverage:
    asset: str
    pair: str
    earliest_raw_open: datetime | None
    latest_raw_close: datetime | None
    raw_5m_rows: int
    valid_15m_rows: int
    missing_5m_intervals: int
    excluded_15m_bars: int
    forming_5m_rows: int
    stale: bool


@dataclass(frozen=True, slots=True)
class CoverageReport:
    quote_currency: str
    assets: tuple[AssetCoverage, ...]
    common_start: datetime | None
    common_end: datetime | None
    synchronized_rows: int
    dropped_timestamps: int
    source_fingerprint: str


def summarize_coverage(dataset: ResearchDataset) -> CoverageReport:
    assets: list[AssetCoverage] = []
    for asset in dataset.universe.assets:
        raw = dataset.raw_by_asset[asset]
        aggregation = dataset.aggregation_by_asset[asset]
        diagnostics = aggregation.diagnostics
        assets.append(
            AssetCoverage(
                asset=asset,
                pair=dataset.universe.pair(asset),
                earliest_raw_open=raw[0].open_time if raw else None,
                latest_raw_close=raw[-1].close_time if raw else None,
                raw_5m_rows=len(raw),
                valid_15m_rows=len(aggregation.bars),
                missing_5m_intervals=len(diagnostics.raw_quality.missing_open_times),
                excluded_15m_bars=diagnostics.excluded_model_bars,
                forming_5m_rows=len(diagnostics.raw_quality.forming_open_times),
                stale=diagnostics.raw_quality.stale,
            )
        )
    return CoverageReport(
        quote_currency=dataset.universe.quote_currency,
        assets=tuple(assets),
        common_start=dataset.panel.start,
        common_end=dataset.panel.end,
        synchronized_rows=len(dataset.panel.rows),
        dropped_timestamps=dataset.panel.diagnostics.dropped_timestamp_count,
        source_fingerprint=dataset.source_fingerprint,
    )
