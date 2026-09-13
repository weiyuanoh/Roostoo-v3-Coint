"""Chronological, non-trading comparison across three sequential forecast models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence

import numpy as np

from stat_arb_bot.models.numeric import readonly_array
from stat_arb_bot.models.state_space.baselines import PlainVECMForecaster, RandomWalkKalmanFilter
from stat_arb_bot.models.state_space.config import KalmanConfig
from stat_arb_bot.models.state_space.diagnostics import (
    ChronologicalForecastRecord,
    ForecastMetrics,
    InnovationDiagnostics,
    evaluate_forecasts,
    summarize_innovations,
)
from stat_arb_bot.models.state_space.filter import FilterStepResult
from stat_arb_bot.models.state_space.regime import RegimeTransition, StructuralRegimeManager
from stat_arb_bot.models.structural import StructuralFitResult
from stat_arb_bot.research_data.panel import ResearchPanel


@dataclass(frozen=True, slots=True)
class SequentialDiagnosticStudyResult:
    active_timestamps: tuple[datetime, ...]
    inactive_timestamps: tuple[datetime, ...]
    active_structural_fit_ids: tuple[str, ...]
    transitions: tuple[RegimeTransition, ...]
    filter_steps: tuple[FilterStepResult, ...]
    forecasts: tuple[ChronologicalForecastRecord, ...]
    metrics: tuple[ForecastMetrics, ...]
    innovations: InnovationDiagnostics


def _record(
    *,
    model: str,
    assets: tuple[str, ...],
    fit: StructuralFitResult,
    origin_timestamp: datetime,
    horizon: int,
    origin_observation: np.ndarray,
    predicted: np.ndarray,
    bar_minutes: int,
) -> ChronologicalForecastRecord:
    vecm = fit.rank_one
    if vecm is None:
        raise ValueError("forecast records require a rank-one fit")
    return ChronologicalForecastRecord(
        model=model,
        assets=assets,
        structural_fit_id=fit.metadata.fit_id,
        origin_timestamp=origin_timestamp,
        target_timestamp=origin_timestamp + timedelta(minutes=horizon * bar_minutes),
        horizon_steps=horizon,
        origin_observation=readonly_array(origin_observation, dimensions=1),
        predicted_log_prices=readonly_array(predicted, dimensions=1),
        native_beta=vecm.native_beta,
        cointegration_constant=vecm.native_cointegration_constant,
    )


def run_sequential_diagnostic_study(
    panel: ResearchPanel,
    structural_fits: Sequence[StructuralFitResult],
    *,
    config: KalmanConfig | None = None,
    forecast_origin_stride: int = 4,
) -> SequentialDiagnosticStudyResult:
    """Replay fits and observations chronologically; never generates an order."""

    if forecast_origin_stride <= 0:
        raise ValueError("forecast_origin_stride must be positive")
    specification = config or KalmanConfig()
    fits = tuple(sorted(structural_fits, key=lambda fit: fit.metadata.fit_timestamp))
    if not fits:
        raise ValueError("sequential study requires at least one structural fit")
    if any(
        right.metadata.fit_timestamp <= left.metadata.fit_timestamp
        for left, right in zip(fits, fits[1:])
    ):
        raise ValueError("structural fits must have distinct chronological timestamps")
    matrix = panel.log_prices()
    timestamps = matrix.timestamps
    values = np.asarray(matrix.values, dtype=np.float64)
    index_by_timestamp = {timestamp: index for index, timestamp in enumerate(timestamps)}
    actual = {
        timestamp: readonly_array(row, dimensions=1) for timestamp, row in zip(timestamps, values)
    }
    manager = StructuralRegimeManager(specification)
    steps: list[FilterStepResult] = []
    active_times: list[datetime] = []
    inactive_times: list[datetime] = []
    active_ids: list[str] = []
    records: list[ChronologicalForecastRecord] = []

    for fit_index, fit in enumerate(fits):
        activation = fit.metadata.fit_timestamp
        if activation not in index_by_timestamp:
            raise ValueError("structural fit timestamp is not a completed panel observation")
        start_index = index_by_timestamp[activation]
        if fit.metadata.data_end_timestamp > activation:
            raise ValueError("structural fit has invalid future chronology")

        # If a prior rank-one filter exists, consume activation y_t under the
        # old fit before changing structural regimes. The new fit may itself
        # use y_t, but no future observation is introduced.
        if manager.active_filter is not None and manager.active_filter.timestamp < activation:
            old_step = manager.active_filter.step(
                timestamp=activation,
                observation=values[start_index],
            )
            steps.append(old_step)
            active_times.append(activation)

        history_start = max(0, start_index - 8)
        manager.apply_fit(
            fit,
            timestamp=activation,
            history_timestamps=timestamps[history_start : start_index + 1],
            level_history=values[history_start : start_index + 1],
        )
        if manager.active_filter is not None:
            active_ids.append(fit.metadata.fit_id)
            model = manager.active_filter.model
            random_walk = RandomWalkKalmanFilter(
                timestamp=activation,
                observation=values[start_index],
                process_covariance=model.vecm.residual_covariance,
                measurement_noise=model.measurement_noise,
                initial_covariance_multiplier=specification.initial_covariance_multiplier,
                psd_tolerance=specification.psd_tolerance,
                symmetry_tolerance=specification.symmetry_tolerance,
                bar_minutes=specification.bar_minutes,
            )
            plain = PlainVECMForecaster(model)
        else:
            random_walk = None
            plain = None
        next_activation = (
            fits[fit_index + 1].metadata.fit_timestamp
            if fit_index + 1 < len(fits)
            else timestamps[-1] + timedelta(minutes=specification.bar_minutes)
        )
        interval_indices = [
            index
            for index in range(start_index + 1, len(timestamps))
            if timestamps[index] < next_activation
        ]
        for offset, index in enumerate(interval_indices):
            timestamp = timestamps[index]
            if manager.active_filter is None:
                inactive_times.append(timestamp)
                continue
            step = manager.active_filter.step(timestamp=timestamp, observation=values[index])
            steps.append(step)
            active_times.append(timestamp)
            assert random_walk is not None and plain is not None
            random_walk.step(timestamp=timestamp, observation=values[index])
            if offset % forecast_origin_stride:
                continue
            horizons = manager.active_filter.diagnostic_horizons()
            history_start = max(0, index - model.layout.lagged_difference_count)
            observed_history = values[history_start : index + 1]
            for horizon in horizons:
                ecm_forecast = manager.active_filter.forecast(horizon)
                random_forecast = random_walk.forecast(
                    horizon,
                    structural_fit_id=fit.metadata.fit_id,
                )
                plain_forecast = plain.forecast(
                    timestamp=timestamp,
                    level_history=observed_history,
                    horizon_steps=horizon,
                )
                records.extend(
                    (
                        _record(
                            model="ecm_driven_kf",
                            assets=matrix.assets,
                            fit=fit,
                            origin_timestamp=timestamp,
                            horizon=horizon,
                            origin_observation=values[index],
                            predicted=ecm_forecast.price_mean,
                            bar_minutes=specification.bar_minutes,
                        ),
                        _record(
                            model="random_walk_kf",
                            assets=matrix.assets,
                            fit=fit,
                            origin_timestamp=timestamp,
                            horizon=horizon,
                            origin_observation=values[index],
                            predicted=random_forecast.price_mean,
                            bar_minutes=specification.bar_minutes,
                        ),
                        _record(
                            model="plain_vecm",
                            assets=matrix.assets,
                            fit=fit,
                            origin_timestamp=timestamp,
                            horizon=horizon,
                            origin_observation=values[index],
                            predicted=plain_forecast.price_mean,
                            bar_minutes=specification.bar_minutes,
                        ),
                    )
                )
    metrics = evaluate_forecasts(records, actual, bar_minutes=specification.bar_minutes)
    return SequentialDiagnosticStudyResult(
        active_timestamps=tuple(active_times),
        inactive_timestamps=tuple(inactive_times),
        active_structural_fit_ids=tuple(dict.fromkeys(active_ids)),
        transitions=tuple(manager.transitions),
        filter_steps=tuple(steps),
        forecasts=tuple(records),
        metrics=metrics,
        innovations=summarize_innovations(steps, assets=matrix.assets),
    )
