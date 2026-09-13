"""Explicit, immutable configuration for structural estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class LagCriterion(str, Enum):
    AIC = "aic"
    BIC = "bic"
    HQIC = "hqic"


@dataclass(frozen=True, slots=True)
class StructuralModelConfig:
    """V1 structural specification; none of these fields are trading rules."""

    primary_window: timedelta = timedelta(days=30)
    comparison_windows: tuple[timedelta, ...] = (
        timedelta(days=14),
        timedelta(days=30),
        timedelta(days=60),
    )
    fit_cadence: timedelta = timedelta(hours=24)
    minimum_observations: int = 200
    minimum_var_lag: int = 1
    maximum_var_lag: int = 8
    lag_criterion: LagCriterion = LagCriterion.BIC
    significance_level: float = 0.05
    johansen_deterministic_order: int = 0
    vecm_deterministic: str = "ci"
    adf_regression: str = "c"
    kpss_regression: str = "c"
    near_constant_variance: float = 1e-12
    convergence_steps: int = 256
    autocorrelation_lags: tuple[int, ...] = (1, 4, 16)

    def __post_init__(self) -> None:
        if self.primary_window <= timedelta(0):
            raise ValueError("primary_window must be positive")
        if not self.comparison_windows or any(
            window <= timedelta(0) for window in self.comparison_windows
        ):
            raise ValueError("comparison_windows must contain positive durations")
        if self.fit_cadence <= timedelta(0):
            raise ValueError("fit_cadence must be positive")
        if self.minimum_observations <= 5:
            raise ValueError("minimum_observations must exceed the five-variable universe")
        if self.minimum_var_lag < 1 or self.maximum_var_lag < self.minimum_var_lag:
            raise ValueError("VAR lag range must satisfy 1 <= minimum <= maximum")
        if self.significance_level not in {0.01, 0.05, 0.10}:
            raise ValueError("Johansen significance_level must be 0.01, 0.05, or 0.10")
        if self.johansen_deterministic_order not in {-1, 0, 1}:
            raise ValueError("Johansen deterministic order must be -1, 0, or 1")
        if self.vecm_deterministic != "ci":
            raise ValueError("V1 supports only a constant inside the cointegration relation ('ci')")
        if self.adf_regression not in {"n", "c", "ct", "ctt"}:
            raise ValueError("unsupported ADF deterministic specification")
        if self.kpss_regression not in {"c", "ct"}:
            raise ValueError("unsupported KPSS deterministic specification")
        if self.near_constant_variance <= 0 or self.convergence_steps <= 0:
            raise ValueError("numeric diagnostic thresholds must be positive")
        if any(lag <= 0 for lag in self.autocorrelation_lags):
            raise ValueError("autocorrelation lags must be positive")

    def as_mapping(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["primary_window_seconds"] = self.primary_window.total_seconds()
        payload["comparison_window_seconds"] = tuple(
            item.total_seconds() for item in self.comparison_windows
        )
        payload["fit_cadence_seconds"] = self.fit_cadence.total_seconds()
        payload["lag_criterion"] = self.lag_criterion.value
        del payload["primary_window"]
        del payload["comparison_windows"]
        del payload["fit_cadence"]
        return MappingProxyType(payload)


def vecm_lagged_differences(var_order: int) -> int:
    """Map a levels VAR(p) order to the VECM's p-1 lagged differences."""

    if var_order < 1:
        raise ValueError("VAR order must be at least one")
    return var_order - 1
