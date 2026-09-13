"""Explicit configuration for VECM and baseline Kalman filters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MeasurementNoiseMode(str, Enum):
    RESIDUAL_DIAGONAL_FRACTION = "RESIDUAL_DIAGONAL_FRACTION"
    EXPLICIT_DIAGONAL = "EXPLICIT_DIAGONAL"
    EXPLICIT_FULL = "EXPLICIT_FULL"


@dataclass(frozen=True, slots=True)
class MeasurementNoiseConfig:
    """Observation noise around the slower VECM-consistent latent state."""

    mode: MeasurementNoiseMode = MeasurementNoiseMode.RESIDUAL_DIAGONAL_FRACTION
    residual_variance_fraction: float = 0.01
    diagonal_variances: tuple[float, ...] | None = None
    covariance: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.residual_variance_fraction < 0:
            raise ValueError("residual_variance_fraction must be non-negative")
        if self.mode is MeasurementNoiseMode.RESIDUAL_DIAGONAL_FRACTION:
            if self.diagonal_variances is not None or self.covariance is not None:
                raise ValueError(
                    "residual-fraction measurement noise cannot include explicit values"
                )
        elif self.mode is MeasurementNoiseMode.EXPLICIT_DIAGONAL:
            if self.diagonal_variances is None or self.covariance is not None:
                raise ValueError("explicit diagonal mode requires only diagonal_variances")
            if not self.diagonal_variances or any(value < 0 for value in self.diagonal_variances):
                raise ValueError("measurement diagonal variances must be non-negative")
        elif self.mode is MeasurementNoiseMode.EXPLICIT_FULL:
            if self.covariance is None or self.diagonal_variances is not None:
                raise ValueError("explicit full mode requires only covariance")


@dataclass(frozen=True, slots=True)
class KalmanConfig:
    measurement_noise: MeasurementNoiseConfig = MeasurementNoiseConfig()
    initialization_method: str = "latest_observation_and_bounded_differences"
    initial_covariance_multiplier: float = 1.0
    maximum_forecast_steps: int = 192
    standard_forecast_steps: tuple[int, ...] = (4, 16, 32, 64, 96)
    bar_minutes: int = 15
    psd_tolerance: float = 1e-10
    symmetry_tolerance: float = 1e-10
    maximum_innovation_condition_number: float = 1e14
    maximum_state_norm: float = 1e6

    def __post_init__(self) -> None:
        if not self.initialization_method.strip():
            raise ValueError("initialization_method must not be empty")
        if self.initial_covariance_multiplier <= 0:
            raise ValueError("initial_covariance_multiplier must be positive")
        if self.maximum_forecast_steps <= 0 or self.bar_minutes <= 0:
            raise ValueError("forecast limit and bar duration must be positive")
        if not self.standard_forecast_steps or any(
            step <= 0 or step > self.maximum_forecast_steps for step in self.standard_forecast_steps
        ):
            raise ValueError("standard forecast steps must be within the configured limit")
        if self.psd_tolerance <= 0 or self.symmetry_tolerance <= 0:
            raise ValueError("matrix tolerances must be positive")
        if self.maximum_innovation_condition_number <= 1:
            raise ValueError("innovation condition-number limit must exceed one")
        if self.maximum_state_norm <= 0:
            raise ValueError("maximum_state_norm must be positive")
