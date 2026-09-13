"""Chronological, non-mutating structural-fit scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from stat_arb_bot.domain.execution import utc_datetime
from stat_arb_bot.models.config import StructuralModelConfig
from stat_arb_bot.models.structural import StructuralEstimator, StructuralFitResult
from stat_arb_bot.research_data.pipeline import ResearchDataset


@dataclass(frozen=True, slots=True)
class RollingFitResult:
    fits: tuple[StructuralFitResult, ...]
    skipped_timestamps: tuple[datetime, ...]


class StructuralFitScheduler:
    """Fit no more often than cadence using the latest completed panel row."""

    def __init__(self, config: StructuralModelConfig | None = None) -> None:
        self.config = config or StructuralModelConfig()
        self.estimator = StructuralEstimator(self.config)

    def scheduled_data_ends(
        self,
        dataset: ResearchDataset,
        *,
        start: datetime,
        end: datetime,
    ) -> tuple[datetime, ...]:
        lower = utc_datetime(start, name="start")
        upper = utc_datetime(end, name="end")
        if lower > upper:
            raise ValueError("rolling-fit start must not follow end")
        eligible = tuple(
            row.timestamp for row in dataset.panel.rows if lower <= row.timestamp <= upper
        )
        selected: list[datetime] = []
        for timestamp in eligible:
            if not selected or timestamp >= selected[-1] + self.config.fit_cadence:
                selected.append(timestamp)
        return tuple(selected)

    def fit_range(
        self,
        dataset: ResearchDataset,
        *,
        start: datetime,
        end: datetime,
    ) -> RollingFitResult:
        fits: list[StructuralFitResult] = []
        skipped: list[datetime] = []
        for data_end in self.scheduled_data_ends(dataset, start=start, end=end):
            window = dataset.panel.window(
                end=data_end,
                lookback=self.config.primary_window,
                minimum_observations=self.config.minimum_observations,
            )
            if not window.validation.valid:
                skipped.append(data_end)
                continue
            fits.append(
                self.estimator.fit(
                    window,
                    fit_timestamp=data_end,
                    source_fingerprint=dataset.source_fingerprint_through(data_end),
                )
            )
        return RollingFitResult(tuple(fits), tuple(skipped))
