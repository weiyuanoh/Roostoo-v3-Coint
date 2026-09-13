"""Inspectable rank-one VECM estimates and deterministic dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from statsmodels.tsa.vector_ar.vecm import VECM

from stat_arb_bot.models.config import StructuralModelConfig, vecm_lagged_differences
from stat_arb_bot.models.diagnostics import PersistenceDiagnostic, persistence_diagnostic
from stat_arb_bot.models.numeric import readonly_array, sign_aligned_unit_vector


@dataclass(frozen=True, slots=True)
class ECMStatistics:
    values: NDArray
    current: float
    mean: float
    standard_deviation: float
    median: float
    median_absolute_deviation: float
    robust_spread: float
    z_score: float | None
    persistence: PersistenceDiagnostic


@dataclass(frozen=True, slots=True)
class ContributionDecomposition:
    error_correction: NDArray
    short_run: NDArray
    deterministic: NDArray
    conditional_mean_change: NDArray


@dataclass(frozen=True, slots=True)
class ConvergenceDiagnostic:
    simplified_persistence: float
    simulated_ecm_path: NDArray
    simulated_half_life_observations: float | None
    empirical_half_life_observations: float | None
    material_disagreement: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RankOneVECMResult:
    assets: tuple[str, ...]
    var_order: int
    lagged_difference_count: int
    deterministic: str
    sample_count: int
    native_beta: NDArray
    native_cointegration_constant: float
    native_alpha: NDArray
    native_pi: NDArray
    reporting_beta: NDArray
    reporting_cointegration_constant: float
    reporting_alpha: NDArray
    reporting_scale: float
    alpha_standard_errors: NDArray
    beta_standard_errors: NDArray
    gamma_matrices: tuple[NDArray, ...]
    gamma_standard_errors: tuple[NDArray, ...]
    deterministic_outside: NDArray
    residuals: NDArray
    fitted_values: NDArray
    residual_covariance: NDArray
    ecm: ECMStatistics
    current_error_correction_contribution: NDArray
    current_adjustment_signs: tuple[int, ...]
    convergence: ConvergenceDiagnostic
    history_tail: NDArray

    def decompose(self, level_history: ArrayLike) -> ContributionDecomposition:
        """Decompose the next expected change using only supplied history."""

        levels = readonly_array(level_history, dimensions=2)
        if levels.shape[1] != len(self.assets):
            raise ValueError("level history width differs from fitted universe")
        if len(levels) < self.lagged_difference_count + 1:
            raise ValueError("insufficient history for fitted Gamma dynamics")
        equilibrium = float(levels[-1] @ self.native_beta) + self.native_cointegration_constant
        correction = readonly_array(self.native_alpha * equilibrium, dimensions=1)
        changes = np.diff(levels, axis=0)
        short_run = np.zeros(len(self.assets), dtype=np.float64)
        for offset, gamma in enumerate(self.gamma_matrices, start=1):
            short_run += gamma @ changes[-offset]
        deterministic = readonly_array(self.deterministic_outside, dimensions=1)
        short_run = readonly_array(short_run, dimensions=1)
        conditional = readonly_array(correction + short_run + deterministic, dimensions=1)
        return ContributionDecomposition(correction, short_run, deterministic, conditional)

    def simulate_zero_shock(
        self,
        *,
        steps: int,
        level_history: ArrayLike | None = None,
    ) -> NDArray:
        if steps <= 0:
            raise ValueError("simulation steps must be positive")
        history = readonly_array(
            self.history_tail if level_history is None else level_history,
            dimensions=2,
        )
        if len(history) < self.lagged_difference_count + 1:
            raise ValueError("insufficient starting history for deterministic simulation")
        evolving = [row.copy() for row in history]
        path = [float(evolving[-1] @ self.native_beta) + self.native_cointegration_constant]
        for _ in range(steps):
            state = np.asarray(evolving, dtype=np.float64)
            change = self.decompose(state).conditional_mean_change
            evolving.append(evolving[-1] + change)
            keep = max(self.lagged_difference_count + 1, 2)
            evolving = evolving[-keep:]
            path.append(float(evolving[-1] @ self.native_beta) + self.native_cointegration_constant)
        return readonly_array(path, dimensions=1)

    @property
    def alpha_by_asset(self) -> Mapping[str, float]:
        return MappingProxyType(
            {asset: float(value) for asset, value in zip(self.assets, self.native_alpha)}
        )

    @property
    def current_correction_by_asset(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                asset: float(value)
                for asset, value in zip(
                    self.assets,
                    self.current_error_correction_contribution,
                )
            }
        )


def _ecm_statistics(
    values: NDArray,
    config: StructuralModelConfig,
) -> ECMStatistics:
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust = 1.4826 * mad
    current = float(values[-1])
    z_score = (
        (current - mean) / standard_deviation
        if standard_deviation > config.near_constant_variance**0.5
        else None
    )
    return ECMStatistics(
        values=readonly_array(values, dimensions=1),
        current=current,
        mean=mean,
        standard_deviation=standard_deviation,
        median=median,
        median_absolute_deviation=mad,
        robust_spread=robust,
        z_score=z_score,
        persistence=persistence_diagnostic(values, config=config),
    )


def _half_decay(path: NDArray, tolerance: float) -> tuple[float | None, str | None]:
    initial = abs(float(path[0]))
    if initial <= tolerance:
        return None, "initial ECM is too close to equilibrium to measure deterministic decay"
    target = initial / 2.0
    for index in range(1, len(path)):
        if abs(float(path[index])) <= target:
            prior = abs(float(path[index - 1]))
            current = abs(float(path[index]))
            if prior == current:
                return float(index), None
            fraction = (prior - target) / (prior - current)
            return float(index - 1 + min(max(fraction, 0.0), 1.0)), None
    return None, "deterministic ECM path did not halve within the configured horizon"


def _validate_covariance(covariance: NDArray, width: int) -> None:
    if covariance.shape != (width, width):
        raise ValueError(f"residual covariance must be {width}x{width}")
    if not np.allclose(covariance, covariance.T, rtol=1e-8, atol=1e-10):
        raise ValueError("residual covariance must be symmetric")
    if np.any(np.diag(covariance) <= 0):
        raise ValueError("residual covariance diagonal must be positive")


def fit_rank_one_vecm(
    levels: ArrayLike,
    *,
    assets: tuple[str, ...],
    var_order: int,
    config: StructuralModelConfig,
) -> RankOneVECMResult:
    values = readonly_array(levels, dimensions=2)
    if values.shape[1] != len(assets):
        raise ValueError("level data width differs from asset universe")
    lagged_differences = vecm_lagged_differences(var_order)
    fitted = VECM(
        values,
        k_ar_diff=lagged_differences,
        coint_rank=1,
        deterministic=config.vecm_deterministic,
    ).fit()
    native_beta = readonly_array(fitted.beta[:, 0], dimensions=1)
    native_alpha = readonly_array(fitted.alpha[:, 0], dimensions=1)
    cointegration_constant = (
        float(fitted.det_coef_coint[0, 0]) if fitted.det_coef_coint.size else 0.0
    )
    reporting_beta, reporting_scale = sign_aligned_unit_vector(native_beta)
    reporting_constant = cointegration_constant * reporting_scale
    reporting_alpha = readonly_array(native_alpha / reporting_scale, dimensions=1)
    native_pi = readonly_array(np.outer(native_alpha, native_beta), dimensions=2)
    gamma = readonly_array(fitted.gamma, dimensions=2)
    gamma_se = readonly_array(fitted.stderr_gamma, dimensions=2)
    gamma_matrices = tuple(
        readonly_array(gamma[:, index * len(assets) : (index + 1) * len(assets)], dimensions=2)
        for index in range(lagged_differences)
    )
    gamma_standard_errors = tuple(
        readonly_array(
            gamma_se[:, index * len(assets) : (index + 1) * len(assets)],
            dimensions=2,
        )
        for index in range(lagged_differences)
    )
    deterministic_outside = np.zeros(len(assets), dtype=np.float64)
    if fitted.det_coef.size:
        # V1's ``ci`` specification has no outside term. Fail explicitly if a
        # future configuration introduces one without defining its time basis.
        raise ValueError("outside deterministic coefficients need an explicit time basis")
    ecm_values = readonly_array(
        values @ native_beta + cointegration_constant,
        dimensions=1,
    )
    ecm = _ecm_statistics(ecm_values, config)
    correction = readonly_array(native_alpha * ecm.current, dimensions=1)
    signs = tuple(int(np.sign(value)) for value in correction)
    covariance = readonly_array(fitted.sigma_u, dimensions=2)
    _validate_covariance(covariance, len(assets))
    keep = max(lagged_differences + 1, 2)
    history_tail = readonly_array(values[-keep:], dimensions=2)

    # Build an intermediate result so the shared decomposition drives the
    # deterministic simulation; no parallel forecast equation is maintained.
    placeholder = ConvergenceDiagnostic(
        simplified_persistence=float(1.0 + native_beta @ native_alpha),
        simulated_ecm_path=readonly_array([ecm.current], dimensions=1),
        simulated_half_life_observations=None,
        empirical_half_life_observations=ecm.persistence.half_life_observations,
        material_disagreement=False,
        warnings=(),
    )
    result = RankOneVECMResult(
        assets=assets,
        var_order=var_order,
        lagged_difference_count=lagged_differences,
        deterministic=config.vecm_deterministic,
        sample_count=len(values),
        native_beta=native_beta,
        native_cointegration_constant=cointegration_constant,
        native_alpha=native_alpha,
        native_pi=native_pi,
        reporting_beta=reporting_beta,
        reporting_cointegration_constant=reporting_constant,
        reporting_alpha=reporting_alpha,
        reporting_scale=reporting_scale,
        alpha_standard_errors=readonly_array(fitted.stderr_alpha[:, 0], dimensions=1),
        beta_standard_errors=readonly_array(fitted.stderr_beta[:, 0], dimensions=1),
        gamma_matrices=gamma_matrices,
        gamma_standard_errors=gamma_standard_errors,
        deterministic_outside=readonly_array(deterministic_outside, dimensions=1),
        residuals=readonly_array(fitted.resid, dimensions=2),
        fitted_values=readonly_array(fitted.fittedvalues, dimensions=2),
        residual_covariance=covariance,
        ecm=ecm,
        current_error_correction_contribution=correction,
        current_adjustment_signs=signs,
        convergence=placeholder,
        history_tail=history_tail,
    )
    path = result.simulate_zero_shock(steps=config.convergence_steps)
    simulated_half_life, warning = _half_decay(
        path,
        config.near_constant_variance**0.5,
    )
    empirical = ecm.persistence.half_life_observations
    disagreement = bool(
        simulated_half_life is not None
        and empirical is not None
        and min(simulated_half_life, empirical) > 0
        and max(simulated_half_life, empirical) / min(simulated_half_life, empirical) >= 2.0
    )
    messages = ([warning] if warning else []) + (
        ["empirical and deterministic half-life estimates differ by at least 2x"]
        if disagreement
        else []
    )
    convergence = ConvergenceDiagnostic(
        simplified_persistence=float(1.0 + native_beta @ native_alpha),
        simulated_ecm_path=path,
        simulated_half_life_observations=simulated_half_life,
        empirical_half_life_observations=empirical,
        material_disagreement=disagreement,
        warnings=tuple(messages),
    )
    return RankOneVECMResult(
        assets=result.assets,
        var_order=result.var_order,
        lagged_difference_count=result.lagged_difference_count,
        deterministic=result.deterministic,
        sample_count=result.sample_count,
        native_beta=result.native_beta,
        native_cointegration_constant=result.native_cointegration_constant,
        native_alpha=result.native_alpha,
        native_pi=result.native_pi,
        reporting_beta=result.reporting_beta,
        reporting_cointegration_constant=result.reporting_cointegration_constant,
        reporting_alpha=result.reporting_alpha,
        reporting_scale=result.reporting_scale,
        alpha_standard_errors=result.alpha_standard_errors,
        beta_standard_errors=result.beta_standard_errors,
        gamma_matrices=result.gamma_matrices,
        gamma_standard_errors=result.gamma_standard_errors,
        deterministic_outside=result.deterministic_outside,
        residuals=result.residuals,
        fitted_values=result.fitted_values,
        residual_covariance=result.residual_covariance,
        ecm=result.ecm,
        current_error_correction_contribution=result.current_error_correction_contribution,
        current_adjustment_signs=result.current_adjustment_signs,
        convergence=convergence,
        history_tail=result.history_tail,
    )
