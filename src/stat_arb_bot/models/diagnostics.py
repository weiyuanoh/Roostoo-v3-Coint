"""Structured integration and univariate persistence diagnostics."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike
from statsmodels.tsa.stattools import acf, adfuller, kpss

from stat_arb_bot.models.config import StructuralModelConfig
from stat_arb_bot.models.numeric import readonly_array
from stat_arb_bot.research_data.windows import ResearchWindow


@dataclass(frozen=True, slots=True)
class UnitRootDiagnostic:
    test: str
    deterministic: str
    sample_count: int
    statistic: float | None
    p_value: float | None
    critical_values: Mapping[str, float]
    warnings: tuple[str, ...]
    failure: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "critical_values", MappingProxyType(dict(self.critical_values)))


@dataclass(frozen=True, slots=True)
class SeriesIntegrationDiagnostic:
    asset: str
    level_count: int
    difference_count: int
    level_variance: float
    difference_variance: float
    finite: bool
    level_near_constant: bool
    difference_near_constant: bool
    level_adf: UnitRootDiagnostic
    level_kpss: UnitRootDiagnostic
    difference_adf: UnitRootDiagnostic
    difference_kpss: UnitRootDiagnostic
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IntegrationDiagnostics:
    assets: tuple[str, ...]
    series: tuple[SeriesIntegrationDiagnostic, ...]
    sample_count: int

    def for_asset(self, asset: str) -> SeriesIntegrationDiagnostic:
        normalized = asset.strip().upper()
        return next(item for item in self.series if item.asset == normalized)


def _failure(test: str, deterministic: str, count: int, message: str) -> UnitRootDiagnostic:
    return UnitRootDiagnostic(test, deterministic, count, None, None, {}, (), message)


def adf_diagnostic(values: ArrayLike, *, regression: str = "c") -> UnitRootDiagnostic:
    series = readonly_array(values, dimensions=1)
    if len(series) < 4:
        return _failure("ADF", regression, len(series), "at least four observations are required")
    caught: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            statistic, p_value, _lags, _nobs, critical, *_ = adfuller(
                series,
                regression=regression,
                autolag="BIC",
            )
        caught.extend(str(item.message) for item in records)
        return UnitRootDiagnostic(
            "ADF",
            regression,
            len(series),
            float(statistic),
            float(p_value),
            {str(key): float(value) for key, value in critical.items()},
            tuple(caught),
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _failure("ADF", regression, len(series), str(exc))


def kpss_diagnostic(values: ArrayLike, *, regression: str = "c") -> UnitRootDiagnostic:
    series = readonly_array(values, dimensions=1)
    if len(series) < 4:
        return _failure("KPSS", regression, len(series), "at least four observations are required")
    caught: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            statistic, p_value, _lags, critical = kpss(
                series,
                regression=regression,
                nlags="auto",
            )
        caught.extend(str(item.message) for item in records)
        return UnitRootDiagnostic(
            "KPSS",
            regression,
            len(series),
            float(statistic),
            float(p_value),
            {str(key): float(value) for key, value in critical.items()},
            tuple(caught),
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _failure("KPSS", regression, len(series), str(exc))


def diagnose_integration(
    window: ResearchWindow,
    config: StructuralModelConfig,
) -> IntegrationDiagnostics:
    if not window.validation.valid:
        raise ValueError("integration diagnostics require a valid bounded research window")
    matrix = window.log_prices()
    levels = readonly_array(matrix.values, dimensions=2)
    results: list[SeriesIntegrationDiagnostic] = []
    for index, asset in enumerate(matrix.assets):
        level = levels[:, index]
        difference = np.diff(level)
        level_variance = float(np.var(level, ddof=1))
        difference_variance = float(np.var(difference, ddof=1))
        finite = bool(np.all(np.isfinite(level)) and np.all(np.isfinite(difference)))
        level_near = level_variance <= config.near_constant_variance
        difference_near = difference_variance <= config.near_constant_variance
        messages = []
        if not finite:
            messages.append("non-finite level or difference value")
        if level_near:
            messages.append("log-price level is near constant")
        if difference_near:
            messages.append("log-price difference is near constant")
        results.append(
            SeriesIntegrationDiagnostic(
                asset=asset,
                level_count=len(level),
                difference_count=len(difference),
                level_variance=level_variance,
                difference_variance=difference_variance,
                finite=finite,
                level_near_constant=level_near,
                difference_near_constant=difference_near,
                level_adf=adf_diagnostic(level, regression=config.adf_regression),
                level_kpss=kpss_diagnostic(level, regression=config.kpss_regression),
                difference_adf=adf_diagnostic(difference, regression=config.adf_regression),
                difference_kpss=kpss_diagnostic(
                    difference,
                    regression=config.kpss_regression,
                ),
                warnings=tuple(messages),
            )
        )
    return IntegrationDiagnostics(matrix.assets, tuple(results), len(levels))


@dataclass(frozen=True, slots=True)
class PersistenceDiagnostic:
    sample_count: int
    variance: float
    autocorrelations: Mapping[int, float]
    ar1_intercept: float | None
    ar1_phi: float | None
    half_life_observations: float | None
    adf: UnitRootDiagnostic
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "autocorrelations", MappingProxyType(dict(self.autocorrelations)))


def persistence_diagnostic(
    values: ArrayLike,
    *,
    config: StructuralModelConfig,
) -> PersistenceDiagnostic:
    series = readonly_array(values, dimensions=1)
    variance = float(np.var(series, ddof=1)) if len(series) > 1 else 0.0
    requested = tuple(lag for lag in config.autocorrelation_lags if lag < len(series))
    correlations: dict[int, float] = {}
    if requested:
        calculated = acf(series, nlags=max(requested), fft=True)
        correlations = {lag: float(calculated[lag]) for lag in requested}
    messages: list[str] = []
    intercept: float | None = None
    phi: float | None = None
    half_life: float | None = None
    if len(series) >= 3 and variance > config.near_constant_variance:
        design = np.column_stack((np.ones(len(series) - 1), series[:-1]))
        coefficients, *_ = np.linalg.lstsq(design, series[1:], rcond=None)
        intercept, phi = (float(coefficients[0]), float(coefficients[1]))
        magnitude = abs(phi)
        if magnitude >= 1:
            messages.append("AR(1) persistence is not finite mean reverting: |phi| >= 1")
        elif magnitude <= np.finfo(float).eps:
            half_life = 0.0
            messages.append("AR(1) persistence is near zero; decay is effectively immediate")
        else:
            half_life = float(-np.log(2.0) / np.log(magnitude))
    else:
        messages.append("ECM is too short or near constant for an AR(1) estimate")
    return PersistenceDiagnostic(
        sample_count=len(series),
        variance=variance,
        autocorrelations=correlations,
        ar1_intercept=intercept,
        ar1_phi=phi,
        half_life_observations=half_life,
        adf=adf_diagnostic(series, regression=config.adf_regression),
        warnings=tuple(messages),
    )
