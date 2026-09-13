"""Non-trading random-walk KF and observed-history VECM forecast baselines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stat_arb_bot.domain.execution import utc_datetime
from stat_arb_bot.models.numeric import readonly_array
from stat_arb_bot.models.state_space.numeric import validated_covariance
from stat_arb_bot.models.state_space.representation import ResolvedMeasurementNoise, VECMStateSpace


@dataclass(frozen=True, slots=True)
class BaselineForecast:
    model: str
    origin_timestamp: datetime
    structural_fit_id: str
    horizon_steps: int
    elapsed: timedelta
    price_mean: NDArray
    price_covariance: NDArray
    forecast_change: NDArray
    expected_ecm_path: NDArray | None


@dataclass(frozen=True, slots=True)
class RandomWalkStepResult:
    timestamp: datetime
    predicted_mean: NDArray
    filtered_mean: NDArray
    predicted_covariance: NDArray
    filtered_covariance: NDArray
    innovation: NDArray
    innovation_covariance: NDArray
    kalman_gain: NDArray


class RandomWalkKalmanFilter:
    """A simple five-price random walk retained solely as a forecast baseline."""

    def __init__(
        self,
        *,
        timestamp: datetime,
        observation: ArrayLike,
        process_covariance: ArrayLike,
        measurement_noise: ResolvedMeasurementNoise,
        initial_covariance_multiplier: float = 1.0,
        psd_tolerance: float = 1e-10,
        symmetry_tolerance: float = 1e-10,
        bar_minutes: int = 15,
    ) -> None:
        self.timestamp = utc_datetime(timestamp)
        self.mean = readonly_array(observation, dimensions=1)
        self.dimension = len(self.mean)
        self.psd_tolerance = psd_tolerance
        self.symmetry_tolerance = symmetry_tolerance
        self.bar_minutes = bar_minutes
        self.process_covariance = validated_covariance(
            process_covariance,
            dimension=self.dimension,
            name="random-walk process covariance",
            psd_tolerance=psd_tolerance,
            symmetry_tolerance=symmetry_tolerance,
            require_positive_diagonal=True,
        )
        self.measurement_noise = measurement_noise
        self.covariance = readonly_array(
            self.process_covariance * initial_covariance_multiplier,
            dimensions=2,
        )

    def step(self, *, timestamp: datetime, observation: ArrayLike) -> RandomWalkStepResult:
        current = utc_datetime(timestamp)
        if current <= self.timestamp:
            raise ValueError("random-walk observations must be strictly chronological")
        observed = readonly_array(observation, dimensions=1)
        if observed.shape != (self.dimension,):
            raise ValueError("random-walk observation has an incompatible dimension")
        predicted_mean = self.mean
        predicted_covariance = validated_covariance(
            self.covariance + self.process_covariance,
            dimension=self.dimension,
            name="random-walk predicted covariance",
            psd_tolerance=self.psd_tolerance,
            symmetry_tolerance=self.symmetry_tolerance,
        )
        innovation = readonly_array(observed - predicted_mean, dimensions=1)
        innovation_covariance = validated_covariance(
            predicted_covariance + self.measurement_noise.covariance,
            dimension=self.dimension,
            name="random-walk innovation covariance",
            psd_tolerance=self.psd_tolerance,
            symmetry_tolerance=self.symmetry_tolerance,
            require_positive_diagonal=True,
        )
        gain = readonly_array(
            np.linalg.solve(innovation_covariance, predicted_covariance.T).T,
            dimensions=2,
        )
        filtered_mean = readonly_array(predicted_mean + gain @ innovation, dimensions=1)
        identity = np.eye(self.dimension)
        residual = identity - gain
        filtered_covariance = validated_covariance(
            residual @ predicted_covariance @ residual.T
            + gain @ self.measurement_noise.covariance @ gain.T,
            dimension=self.dimension,
            name="random-walk filtered covariance",
            psd_tolerance=self.psd_tolerance,
            symmetry_tolerance=self.symmetry_tolerance,
        )
        result = RandomWalkStepResult(
            current,
            predicted_mean,
            filtered_mean,
            predicted_covariance,
            filtered_covariance,
            innovation,
            innovation_covariance,
            gain,
        )
        self.timestamp = current
        self.mean = filtered_mean
        self.covariance = filtered_covariance
        return result

    def forecast(self, horizon_steps: int, *, structural_fit_id: str) -> BaselineForecast:
        if horizon_steps <= 0:
            raise ValueError("forecast horizon must be positive")
        covariance = validated_covariance(
            self.covariance + horizon_steps * self.process_covariance,
            dimension=self.dimension,
            name="random-walk forecast covariance",
            psd_tolerance=self.psd_tolerance,
            symmetry_tolerance=self.symmetry_tolerance,
        )
        return BaselineForecast(
            model="random_walk_kf",
            origin_timestamp=self.timestamp,
            structural_fit_id=structural_fit_id,
            horizon_steps=horizon_steps,
            elapsed=timedelta(minutes=horizon_steps * self.bar_minutes),
            price_mean=self.mean,
            price_covariance=covariance,
            forecast_change=readonly_array(np.zeros(self.dimension), dimensions=1),
            expected_ecm_path=None,
        )


class PlainVECMForecaster:
    """Direct VECM propagation initialized from observed history, with no KF update."""

    def __init__(self, model: VECMStateSpace) -> None:
        self.model = model

    def forecast(
        self,
        *,
        timestamp: datetime,
        level_history: ArrayLike,
        horizon_steps: int,
    ) -> BaselineForecast:
        if horizon_steps <= 0 or horizon_steps > self.model.config.maximum_forecast_steps:
            raise ValueError("plain VECM horizon is outside configured limits")
        current_time = utc_datetime(timestamp)
        if self.model.structural_fit.metadata.data_end_timestamp > current_time:
            raise ValueError("plain VECM cannot use a future structural fit")
        history = readonly_array(level_history, dimensions=2)
        state = np.asarray(self.model.state_from_history(history), dtype=np.float64).copy()
        covariance = np.zeros((self.model.layout.dimension, self.model.layout.dimension))
        ecm_path = [
            float(state[self.model.layout.price_slice] @ self.model.vecm.native_beta)
            + self.model.vecm.native_cointegration_constant
        ]
        for _ in range(horizon_steps):
            state = np.asarray(self.model.deterministic_transition(state), dtype=np.float64)
            covariance = self.model.transition_matrix @ covariance @ self.model.transition_matrix.T
            covariance += self.model.process_covariance
            ecm_path.append(
                float(state[self.model.layout.price_slice] @ self.model.vecm.native_beta)
                + self.model.vecm.native_cointegration_constant
            )
        prices = readonly_array(state[self.model.layout.price_slice], dimensions=1)
        price_covariance = validated_covariance(
            covariance[self.model.layout.price_slice, self.model.layout.price_slice],
            dimension=self.model.layout.asset_count,
            name="plain VECM price covariance",
            psd_tolerance=self.model.config.psd_tolerance,
            symmetry_tolerance=self.model.config.symmetry_tolerance,
        )
        current_prices = history[-1]
        return BaselineForecast(
            model="plain_vecm",
            origin_timestamp=current_time,
            structural_fit_id=self.model.structural_fit.metadata.fit_id,
            horizon_steps=horizon_steps,
            elapsed=timedelta(minutes=horizon_steps * self.model.config.bar_minutes),
            price_mean=prices,
            price_covariance=price_covariance,
            forecast_change=readonly_array(prices - current_prices, dimensions=1),
            expected_ecm_path=readonly_array(ecm_path, dimensions=1),
        )
