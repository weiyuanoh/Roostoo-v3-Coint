"""Full and incremental reconstruction of canonical research datasets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from types import MappingProxyType
from typing import Iterable, Mapping

from stat_arb_bot.domain import Candle
from stat_arb_bot.domain.execution import utc_datetime
from stat_arb_bot.research_data.aggregation import (
    AggregationResult,
    aggregate_five_to_fifteen,
    model_bucket_open,
)
from stat_arb_bot.research_data.metadata import fingerprint_raw_candles
from stat_arb_bot.research_data.panel import ResearchPanel, build_synchronized_panel
from stat_arb_bot.research_data.quality import merge_raw_series, prepare_raw_series
from stat_arb_bot.research_data.universe import ResearchUniverse


@dataclass(frozen=True, slots=True)
class ResearchDataset:
    universe: ResearchUniverse
    raw_by_asset: Mapping[str, tuple[Candle, ...]]
    aggregation_by_asset: Mapping[str, AggregationResult]
    panel: ResearchPanel
    source_fingerprint: str
    requested_start: datetime | None
    requested_end: datetime
    affected_model_buckets: Mapping[str, tuple[datetime, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_by_asset", MappingProxyType(dict(self.raw_by_asset)))
        object.__setattr__(
            self,
            "aggregation_by_asset",
            MappingProxyType(dict(self.aggregation_by_asset)),
        )
        object.__setattr__(
            self,
            "affected_model_buckets",
            MappingProxyType(dict(self.affected_model_buckets)),
        )

    def source_fingerprint_through(self, end: datetime) -> str:
        """Fingerprint only raw observations available through a model time."""

        boundary = utc_datetime(end, name="end")
        bounded = {
            asset: tuple(
                candle for candle in self.raw_by_asset[asset] if candle.close_time <= boundary
            )
            for asset in self.universe.assets
        }
        return fingerprint_raw_candles(bounded, self.universe)


class ResearchDataBuilder:
    def __init__(self, universe: ResearchUniverse | None = None) -> None:
        self.universe = universe or ResearchUniverse()

    def build(
        self,
        raw_by_asset: Mapping[str, Iterable[Candle]],
        *,
        as_of: datetime,
        requested_start: datetime | None = None,
        max_age_intervals: int | None = None,
    ) -> ResearchDataset:
        boundary = utc_datetime(as_of, name="as_of")
        start = (
            utc_datetime(requested_start, name="requested_start")
            if requested_start is not None
            else None
        )
        if start is not None and start >= boundary:
            raise ValueError("requested_start must precede as_of")
        canonical: dict[str, tuple[Candle, ...]] = {}
        aggregations: dict[str, AggregationResult] = {}
        affected: dict[str, tuple[datetime, ...]] = {}
        for asset in self.universe.assets:
            pair = self.universe.pair(asset)
            raw = prepare_raw_series(
                raw_by_asset.get(asset, ()),
                symbol=pair,
                as_of=boundary,
                max_age_intervals=max_age_intervals,
            )
            canonical[asset] = raw.candles
            aggregation = aggregate_five_to_fifteen(
                raw.candles,
                symbol=pair,
                as_of=boundary,
                max_age_intervals=max_age_intervals,
            )
            aggregation = replace(
                aggregation,
                diagnostics=replace(aggregation.diagnostics, raw_quality=raw.diagnostics),
            )
            aggregations[asset] = aggregation
            affected[asset] = tuple(bar.open_time for bar in aggregation.bars)
        panel = build_synchronized_panel(
            (bar for asset in self.universe.assets for bar in aggregations[asset].bars),
            universe=self.universe,
            as_of=boundary,
        )
        return ResearchDataset(
            universe=self.universe,
            raw_by_asset=canonical,
            aggregation_by_asset=aggregations,
            panel=panel,
            source_fingerprint=fingerprint_raw_candles(canonical, self.universe),
            requested_start=start,
            requested_end=boundary,
            affected_model_buckets=affected,
        )

    def incremental_update(
        self,
        existing: ResearchDataset,
        incoming_by_asset: Mapping[str, Iterable[Candle]],
        *,
        as_of: datetime,
        max_age_intervals: int | None = None,
    ) -> ResearchDataset:
        if existing.universe != self.universe:
            raise ValueError("existing dataset belongs to a different universe")
        boundary = utc_datetime(as_of, name="as_of")
        canonical: dict[str, tuple[Candle, ...]] = {}
        aggregations: dict[str, AggregationResult] = {}
        affected: dict[str, tuple[datetime, ...]] = {}
        for asset in self.universe.assets:
            incoming = tuple(incoming_by_asset.get(asset, ()))
            pair = self.universe.pair(asset)
            merged = merge_raw_series(
                existing.raw_by_asset.get(asset, ()),
                incoming,
                symbol=pair,
                as_of=boundary,
                max_age_intervals=max_age_intervals,
            )
            canonical[asset] = merged.candles
            rebuilt = aggregate_five_to_fifteen(
                merged.candles,
                symbol=pair,
                as_of=boundary,
                max_age_intervals=max_age_intervals,
            )
            rebuilt = replace(
                rebuilt,
                diagnostics=replace(rebuilt.diagnostics, raw_quality=merged.diagnostics),
            )
            affected_opens = tuple(
                sorted(
                    {
                        model_bucket_open(candle.open_time)
                        for candle in incoming
                        if candle.symbol.strip().upper() == pair
                    }
                )
            )
            old_by_open = {bar.open_time: bar for bar in existing.aggregation_by_asset[asset].bars}
            new_by_open = {bar.open_time: bar for bar in rebuilt.bars}
            for bucket_open in affected_opens:
                old_by_open.pop(bucket_open, None)
                if bucket_open in new_by_open:
                    old_by_open[bucket_open] = new_by_open[bucket_open]
            combined_bars = tuple(old_by_open[key] for key in sorted(old_by_open))
            aggregations[asset] = AggregationResult(
                pair,
                combined_bars,
                rebuilt.diagnostics,
            )
            affected[asset] = affected_opens
        panel = build_synchronized_panel(
            (bar for asset in self.universe.assets for bar in aggregations[asset].bars),
            universe=self.universe,
            as_of=boundary,
        )
        return ResearchDataset(
            universe=self.universe,
            raw_by_asset=canonical,
            aggregation_by_asset=aggregations,
            panel=panel,
            source_fingerprint=fingerprint_raw_candles(canonical, self.universe),
            requested_start=existing.requested_start,
            requested_end=boundary,
            affected_model_buckets=affected,
        )
