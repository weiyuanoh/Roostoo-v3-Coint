"""Chronology-safe ECM-driven Kalman predict/update and forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stat_arb_bot.domain.execution import utc_datetime
from stat_arb_bot.models.numeric import readonly_array
from stat_arb_bot.models.state_space.numeric import (
    StateSpaceNumericalError,
    ensure_finite_state,
    validated_covariance,
)
from stat_arb_bot.models.state_space.config import KalmanConfig, MeasurementNoiseConfig
from stat_arb_bot.models.state_space.representation import VECMStateSpace


FILTER_MODEL_NAME = "ecm_driven_vecm_kalman"
FILTER_MODEL_VERSION = "1"


class FilterStatus(str, Enum):
    ACTIVE = "ACTIVE"
    RESTORED = "RESTORED"


@dataclass(frozen=True, slots=True)
class PredictedState:
    mean: NDArray
    covariance: NDArray
    predicted_observation: NDArray


@dataclass(frozen=True, slots=True)
class ECMObservation:
    observed: float
    filtered: float
    observed_z_score: float | None
    filtered_z_score: float | None


@dataclass(frozen=True, slots=True)
class FilterStepResult:
    timestamp: datetime
    structural_fit_id: str
    structural_fit_data_end: datetime
    rank: int
    filter_model: str
    filter_version: str
    state_layout_version: str
    status: FilterStatus
    filter_config: KalmanConfig
    predicted_state_mean: NDArray
    filtered_state_mean: NDArray
    predicted_covariance: NDArray
    filtered_covariance: NDArray
    observed_log_prices: NDArray
    predicted_observation: NDArray
    innovation: NDArray
    standardized_innovation: tuple[float | None, ...]
    innovation_covariance: NDArray
    kalman_gain: NDArray
    measurement_noise_covariance: NDArray
    measurement_noise_config: MeasurementNoiseConfig
    measurement_noise_interpretation: str
    process_noise_covariance: NDArray
    process_innovation_loading: NDArray
    structural_residual_covariance: NDArray
    process_noise_interpretation: str
    ecm: ECMObservation
    observed_error_correction_contribution: NDArray
    filtered_error_correction_contribution: NDArray
    next_error_correction_contribution: NDArray
    next_short_run_contribution: NDArray
    next_deterministic_contribution: NDArray
    next_expected_change: NDArray

    @property
    def one_step_forecast_error(self) -> NDArray:
        return self.innovation

    def chronology_fields(self) -> dict[str, object]:
        return {
            "model_fit_end_timestamp": self.structural_fit_data_end,
            "filter_timestamp": self.timestamp,
            "model_version": self.filter_version,
            "metadata": {
                "structural_fit_id": self.structural_fit_id,
                "state_layout_version": self.state_layout_version,
            },
        }


@dataclass(frozen=True, slots=True)
class StateForecast:
    origin_timestamp: datetime
    structural_fit_id: str
    horizon_steps: int
    elapsed: timedelta
    elapsed_hours: float
    state_mean: NDArray
    state_covariance: NDArray
    price_mean: NDArray
    price_covariance: NDArray
    latent_state_change: NDArray
    observed_market_relative_change: NDArray
    expected_ecm_path: NDArray


@dataclass(frozen=True, slots=True)
class FilterCheckpoint:
    timestamp: datetime
    structural_fit_id: str
    structural_fit_data_end: datetime
    rank: int
    filter_model: str
    filter_version: str
    state_layout_version: str
    status: FilterStatus
    state_mean: NDArray
    state_covariance: NDArray
    last_observation: NDArray
    initialization_method: str
    measurement_noise_covariance: NDArray
    config: KalmanConfig


class ECMDrivenKalmanFilter:
    """Sequential filter whose transition remains fixed for one structural fit."""

    def __init__(
        self,
        model: VECMStateSpace,
        *,
        timestamp: datetime,
        state_mean: ArrayLike,
        state_covariance: ArrayLike,
        last_observation: ArrayLike,
        status: FilterStatus = FilterStatus.ACTIVE,
    ) -> None:
        self.model = model
        self.timestamp = utc_datetime(timestamp, name="filter timestamp")
        if model.structural_fit.metadata.data_end_timestamp > self.timestamp:
            raise ValueError("structural fit data_end cannot follow filter initialization")
        self.state_mean = ensure_finite_state(
            state_mean,
            dimension=model.layout.dimension,
            maximum_norm=model.config.maximum_state_norm,
            name="filter state",
        )
        self.state_covariance = validated_covariance(
            state_covariance,
            dimension=model.layout.dimension,
            name="filter state covariance",
            psd_tolerance=model.config.psd_tolerance,
            symmetry_tolerance=model.config.symmetry_tolerance,
        )
        observation = readonly_array(last_observation, dimensions=1)
        if observation.shape != (model.layout.asset_count,):
            raise ValueError("last observation has an incompatible dimension")
        self.last_observation = observation
        self.status = status

    @classmethod
    def initialize(
        cls,
        model: VECMStateSpace,
        *,
        initialization_timestamp: datetime,
        history_timestamps: Sequence[datetime],
        level_history: ArrayLike,
        carried_price_mean: ArrayLike | None = None,
        carried_price_covariance: ArrayLike | None = None,
    ) -> ECMDrivenKalmanFilter:
        timestamp = utc_datetime(initialization_timestamp, name="initialization_timestamp")
        times = tuple(utc_datetime(item, name="history timestamp") for item in history_timestamps)
        levels = readonly_array(level_history, dimensions=2)
        if len(times) != len(levels):
            raise ValueError("history timestamps and levels differ in length")
        if not times or times[-1] != timestamp:
            raise ValueError("bounded initialization history must end exactly at initialization")
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("initialization history must be strictly chronological")
        if any(item > timestamp for item in times):
            raise ValueError("initialization cannot use future observations")
        state = model.state_from_history(
            levels,
            current_price_override=carried_price_mean,
        )
        covariance = np.zeros((model.layout.dimension, model.layout.dimension), dtype=np.float64)
        base = model.vecm.residual_covariance * model.config.initial_covariance_multiplier
        for block in range(model.layout.var_order):
            start = block * model.layout.asset_count
            covariance[
                start : start + model.layout.asset_count, start : start + model.layout.asset_count
            ] = base
        if carried_price_covariance is not None:
            carried = validated_covariance(
                carried_price_covariance,
                dimension=model.layout.asset_count,
                name="carried price covariance",
                psd_tolerance=model.config.psd_tolerance,
                symmetry_tolerance=model.config.symmetry_tolerance,
            )
            covariance[model.layout.price_slice, model.layout.price_slice] = carried
        return cls(
            model,
            timestamp=timestamp,
            state_mean=state,
            state_covariance=covariance,
            last_observation=levels[-1],
        )

    @classmethod
    def from_checkpoint(
        cls,
        model: VECMStateSpace,
        checkpoint: FilterCheckpoint,
    ) -> ECMDrivenKalmanFilter:
        if checkpoint.structural_fit_id != model.structural_fit.metadata.fit_id:
            raise ValueError("checkpoint belongs to a different structural fit")
        if checkpoint.state_layout_version != model.layout.version:
            raise ValueError("checkpoint state layout version is incompatible")
        if checkpoint.config != model.config:
            raise ValueError("checkpoint filter configuration is incompatible")
        if not np.array_equal(
            checkpoint.measurement_noise_covariance,
            model.measurement_noise.covariance,
        ):
            raise ValueError("checkpoint measurement-noise configuration is incompatible")
        return cls(
            model,
            timestamp=checkpoint.timestamp,
            state_mean=checkpoint.state_mean,
            state_covariance=checkpoint.state_covariance,
            last_observation=checkpoint.last_observation,
            status=FilterStatus.RESTORED,
        )

    def predict_prior(self) -> PredictedState:
        transition = self.model.transition_matrix
        predicted_mean = self.model.deterministic_transition(self.state_mean)
        covariance = transition @ self.state_covariance @ transition.T
        covariance += self.model.process_covariance
        predicted_covariance = validated_covariance(
            covariance,
            dimension=self.model.layout.dimension,
            name="predicted covariance",
            psd_tolerance=self.model.config.psd_tolerance,
            symmetry_tolerance=self.model.config.symmetry_tolerance,
        )
        predicted_observation = readonly_array(
            self.model.measurement_matrix @ predicted_mean,
            dimensions=1,
        )
        return PredictedState(predicted_mean, predicted_covariance, predicted_observation)

    def step(self, *, timestamp: datetime, observation: ArrayLike) -> FilterStepResult:
        current_time = utc_datetime(timestamp, name="observation timestamp")
        if current_time <= self.timestamp:
            raise ValueError("filter observations must be strictly chronological")
        if self.model.structural_fit.metadata.data_end_timestamp > current_time:
            raise ValueError("filter cannot use a structural fit from the future")
        # This prediction is computed entirely from the posterior through k-1.
        prior = self.predict_prior()
        observed = readonly_array(observation, dimensions=1)
        n = self.model.layout.asset_count
        if observed.shape != (n,):
            raise ValueError("observation has an incompatible dimension")
        measurement = self.model.measurement_matrix
        innovation = readonly_array(observed - prior.predicted_observation, dimensions=1)
        innovation_covariance = measurement @ prior.covariance @ measurement.T
        innovation_covariance += self.model.measurement_noise.covariance
        innovation_covariance = validated_covariance(
            innovation_covariance,
            dimension=n,
            name="innovation covariance",
            psd_tolerance=self.model.config.psd_tolerance,
            symmetry_tolerance=self.model.config.symmetry_tolerance,
            require_positive_diagonal=True,
        )
        condition = float(np.linalg.cond(innovation_covariance))
        if (
            not np.isfinite(condition)
            or condition > self.model.config.maximum_innovation_condition_number
        ):
            raise StateSpaceNumericalError(
                f"innovation covariance is singular or ill-conditioned ({condition})"
            )
        projected = prior.covariance @ measurement.T
        try:
            gain = np.linalg.solve(innovation_covariance, projected.T).T
        except np.linalg.LinAlgError as exc:
            raise StateSpaceNumericalError("innovation covariance is singular") from exc
        gain = readonly_array(gain, dimensions=2)
        filtered_mean = ensure_finite_state(
            prior.mean + gain @ innovation,
            dimension=self.model.layout.dimension,
            maximum_norm=self.model.config.maximum_state_norm,
            name="filtered state",
        )
        identity = np.eye(self.model.layout.dimension)
        residual_operator = identity - gain @ measurement
        joseph = residual_operator @ prior.covariance @ residual_operator.T
        joseph += gain @ self.model.measurement_noise.covariance @ gain.T
        filtered_covariance = validated_covariance(
            joseph,
            dimension=self.model.layout.dimension,
            name="filtered covariance",
            psd_tolerance=self.model.config.psd_tolerance,
            symmetry_tolerance=self.model.config.symmetry_tolerance,
        )
        standard_deviation = np.sqrt(np.diag(innovation_covariance))
        standardized = tuple(
            float(value / scale) if scale > 0 else None
            for value, scale in zip(innovation, standard_deviation)
        )
        vecm = self.model.vecm
        observed_ecm = float(observed @ vecm.native_beta) + vecm.native_cointegration_constant
        filtered_prices = filtered_mean[self.model.layout.price_slice]
        filtered_ecm = (
            float(filtered_prices @ vecm.native_beta) + vecm.native_cointegration_constant
        )
        spread = vecm.ecm.standard_deviation
        observed_z = (observed_ecm - vecm.ecm.mean) / spread if spread > 0 else None
        filtered_z = (filtered_ecm - vecm.ecm.mean) / spread if spread > 0 else None
        decomposition = self.model.decompose_state(filtered_mean)
        result = FilterStepResult(
            timestamp=current_time,
            structural_fit_id=self.model.structural_fit.metadata.fit_id,
            structural_fit_data_end=self.model.structural_fit.metadata.data_end_timestamp,
            rank=1,
            filter_model=FILTER_MODEL_NAME,
            filter_version=FILTER_MODEL_VERSION,
            state_layout_version=self.model.layout.version,
            status=self.status,
            filter_config=self.model.config,
            predicted_state_mean=prior.mean,
            filtered_state_mean=filtered_mean,
            predicted_covariance=prior.covariance,
            filtered_covariance=filtered_covariance,
            observed_log_prices=observed,
            predicted_observation=prior.predicted_observation,
            innovation=innovation,
            standardized_innovation=standardized,
            innovation_covariance=innovation_covariance,
            kalman_gain=gain,
            measurement_noise_covariance=self.model.measurement_noise.covariance,
            measurement_noise_config=self.model.measurement_noise.config,
            measurement_noise_interpretation=self.model.measurement_noise.interpretation,
            process_noise_covariance=self.model.process_covariance,
            process_innovation_loading=self.model.innovation_loading,
            structural_residual_covariance=self.model.vecm.residual_covariance,
            process_noise_interpretation=self.model.process_noise.source,
            ecm=ECMObservation(observed_ecm, filtered_ecm, observed_z, filtered_z),
            observed_error_correction_contribution=readonly_array(
                vecm.native_alpha * observed_ecm,
                dimensions=1,
            ),
            filtered_error_correction_contribution=readonly_array(
                vecm.native_alpha * filtered_ecm,
                dimensions=1,
            ),
            next_error_correction_contribution=decomposition.error_correction,
            next_short_run_contribution=decomposition.short_run,
            next_deterministic_contribution=decomposition.deterministic,
            next_expected_change=decomposition.conditional_mean_change,
        )
        self.timestamp = current_time
        self.state_mean = filtered_mean
        self.state_covariance = filtered_covariance
        self.last_observation = observed
        self.status = FilterStatus.ACTIVE
        return result

    def forecast(self, horizon_steps: int) -> StateForecast:
        if horizon_steps <= 0 or horizon_steps > self.model.config.maximum_forecast_steps:
            raise ValueError("forecast horizon is outside configured limits")
        mean = np.asarray(self.state_mean, dtype=np.float64).copy()
        covariance = np.asarray(self.state_covariance, dtype=np.float64).copy()
        ecm_path = [
            float(mean[self.model.layout.price_slice] @ self.model.vecm.native_beta)
            + self.model.vecm.native_cointegration_constant
        ]
        for _ in range(horizon_steps):
            mean = np.asarray(self.model.deterministic_transition(mean), dtype=np.float64)
            covariance = self.model.transition_matrix @ covariance @ self.model.transition_matrix.T
            covariance += self.model.process_covariance
            ecm_path.append(
                float(mean[self.model.layout.price_slice] @ self.model.vecm.native_beta)
                + self.model.vecm.native_cointegration_constant
            )
        state_mean = ensure_finite_state(
            mean,
            dimension=self.model.layout.dimension,
            maximum_norm=self.model.config.maximum_state_norm,
            name="forecast state",
        )
        state_covariance = validated_covariance(
            covariance,
            dimension=self.model.layout.dimension,
            name="forecast state covariance",
            psd_tolerance=self.model.config.psd_tolerance,
            symmetry_tolerance=self.model.config.symmetry_tolerance,
        )
        price_mean = readonly_array(state_mean[self.model.layout.price_slice], dimensions=1)
        price_covariance = validated_covariance(
            state_covariance[self.model.layout.price_slice, self.model.layout.price_slice],
            dimension=self.model.layout.asset_count,
            name="forecast price covariance",
            psd_tolerance=self.model.config.psd_tolerance,
            symmetry_tolerance=self.model.config.symmetry_tolerance,
        )
        current_prices = self.state_mean[self.model.layout.price_slice]
        return StateForecast(
            origin_timestamp=self.timestamp,
            structural_fit_id=self.model.structural_fit.metadata.fit_id,
            horizon_steps=horizon_steps,
            elapsed=timedelta(minutes=horizon_steps * self.model.config.bar_minutes),
            elapsed_hours=horizon_steps * self.model.config.bar_minutes / 60.0,
            state_mean=state_mean,
            state_covariance=state_covariance,
            price_mean=price_mean,
            price_covariance=price_covariance,
            latent_state_change=readonly_array(price_mean - current_prices, dimensions=1),
            observed_market_relative_change=readonly_array(
                price_mean - self.last_observation,
                dimensions=1,
            ),
            expected_ecm_path=readonly_array(ecm_path, dimensions=1),
        )

    def diagnostic_horizons(self) -> tuple[int, ...]:
        horizons = set(self.model.config.standard_forecast_steps)
        half_life = self.model.vecm.ecm.persistence.half_life_observations
        if half_life is not None:
            dynamic = min(
                max(1, int(round(half_life))),
                self.model.config.maximum_forecast_steps,
            )
            horizons.add(dynamic)
        return tuple(sorted(horizons))

    def checkpoint(self) -> FilterCheckpoint:
        return FilterCheckpoint(
            timestamp=self.timestamp,
            structural_fit_id=self.model.structural_fit.metadata.fit_id,
            structural_fit_data_end=self.model.structural_fit.metadata.data_end_timestamp,
            rank=1,
            filter_model=FILTER_MODEL_NAME,
            filter_version=FILTER_MODEL_VERSION,
            state_layout_version=self.model.layout.version,
            status=self.status,
            state_mean=self.state_mean,
            state_covariance=self.state_covariance,
            last_observation=self.last_observation,
            initialization_method=self.model.config.initialization_method,
            measurement_noise_covariance=self.model.measurement_noise.covariance,
            config=self.model.config,
        )
