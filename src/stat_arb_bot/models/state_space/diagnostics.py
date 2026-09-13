"""Innovation and strictly out-of-sample forecast diagnostics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from statsmodels.tsa.stattools import acf

from stat_arb_bot.models.numeric import readonly_array
from stat_arb_bot.models.state_space.filter import FilterStepResult


@dataclass(frozen=True, slots=True)
class InnovationDiagnostics:
    assets: tuple[str, ...]
    sample_count: int
    start: datetime | None
    end: datetime | None
    mean: NDArray
    mean_absolute_error: NDArray
    root_mean_square_error: NDArray
    standardized_mean: NDArray
    autocorrelation_by_lag: Mapping[int, NDArray]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "autocorrelation_by_lag",
            MappingProxyType(dict(self.autocorrelation_by_lag)),
        )


def summarize_innovations(
    steps: Sequence[FilterStepResult],
    *,
    assets: tuple[str, ...],
    lags: tuple[int, ...] = (1, 4, 16),
) -> InnovationDiagnostics:
    if not steps:
        zeros = readonly_array(np.zeros(len(assets)), dimensions=1)
        return InnovationDiagnostics(assets, 0, None, None, zeros, zeros, zeros, zeros, {})
    values = readonly_array([step.innovation for step in steps], dimensions=2)
    standardized = np.asarray(
        [
            [np.nan if item is None else item for item in step.standardized_innovation]
            for step in steps
        ],
        dtype=np.float64,
    )
    correlations: dict[int, NDArray] = {}
    for lag in lags:
        if lag >= len(values):
            continue
        by_asset = [acf(values[:, index], nlags=lag, fft=True)[lag] for index in range(len(assets))]
        correlations[lag] = readonly_array(by_asset, dimensions=1)
    return InnovationDiagnostics(
        assets=assets,
        sample_count=len(values),
        start=steps[0].timestamp,
        end=steps[-1].timestamp,
        mean=readonly_array(np.mean(values, axis=0), dimensions=1),
        mean_absolute_error=readonly_array(np.mean(np.abs(values), axis=0), dimensions=1),
        root_mean_square_error=readonly_array(
            np.sqrt(np.mean(np.square(values), axis=0)),
            dimensions=1,
        ),
        standardized_mean=readonly_array(np.nanmean(standardized, axis=0), dimensions=1),
        autocorrelation_by_lag=correlations,
    )


@dataclass(frozen=True, slots=True)
class ChronologicalForecastRecord:
    model: str
    assets: tuple[str, ...]
    structural_fit_id: str
    origin_timestamp: datetime
    target_timestamp: datetime
    horizon_steps: int
    origin_observation: NDArray
    predicted_log_prices: NDArray
    native_beta: NDArray
    cointegration_constant: float


@dataclass(frozen=True, slots=True)
class ForecastMetrics:
    model: str
    horizon_steps: int
    elapsed: timedelta
    sample_count: int
    rmse: float
    mae: float
    rmse_by_asset: NDArray
    mae_by_asset: NDArray
    directional_accuracy: float
    ecm_rmse: float
    relative_return_rmse: float


def evaluate_forecasts(
    records: Sequence[ChronologicalForecastRecord],
    actual_by_timestamp: Mapping[datetime, NDArray],
    *,
    bar_minutes: int = 15,
) -> tuple[ForecastMetrics, ...]:
    grouped: dict[tuple[str, int], list[tuple[ChronologicalForecastRecord, NDArray]]] = defaultdict(
        list
    )
    for record in records:
        actual = actual_by_timestamp.get(record.target_timestamp)
        if actual is not None:
            grouped[(record.model, record.horizon_steps)].append((record, actual))
    results: list[ForecastMetrics] = []
    for (model, horizon), observations in sorted(grouped.items()):
        predicted = np.asarray([record.predicted_log_prices for record, _ in observations])
        actual = np.asarray([value for _, value in observations])
        origin = np.asarray([record.origin_observation for record, _ in observations])
        errors = predicted - actual
        predicted_change = predicted - origin
        actual_change = actual - origin
        rmse_by_asset = np.sqrt(np.mean(np.square(errors), axis=0))
        mae_by_asset = np.mean(np.abs(errors), axis=0)
        nonzero = np.abs(actual_change) > np.finfo(float).eps
        direction_hits = np.sign(predicted_change) == np.sign(actual_change)
        directional = float(np.mean(direction_hits[nonzero])) if np.any(nonzero) else 0.0
        predicted_ecm = np.asarray(
            [
                float(prediction @ record.native_beta) + record.cointegration_constant
                for prediction, (record, _) in zip(predicted, observations)
            ]
        )
        actual_ecm = np.asarray(
            [
                float(realized @ record.native_beta) + record.cointegration_constant
                for realized, (record, _) in zip(actual, observations)
            ]
        )
        predicted_relative = predicted_change - np.mean(predicted_change, axis=1, keepdims=True)
        actual_relative = actual_change - np.mean(actual_change, axis=1, keepdims=True)
        results.append(
            ForecastMetrics(
                model=model,
                horizon_steps=horizon,
                elapsed=timedelta(minutes=horizon * bar_minutes),
                sample_count=len(observations),
                rmse=float(np.sqrt(np.mean(np.square(errors)))),
                mae=float(np.mean(np.abs(errors))),
                rmse_by_asset=readonly_array(rmse_by_asset, dimensions=1),
                mae_by_asset=readonly_array(mae_by_asset, dimensions=1),
                directional_accuracy=directional,
                ecm_rmse=float(np.sqrt(np.mean(np.square(predicted_ecm - actual_ecm)))),
                relative_return_rmse=float(
                    np.sqrt(np.mean(np.square(predicted_relative - actual_relative)))
                ),
            )
        )
    return tuple(results)
