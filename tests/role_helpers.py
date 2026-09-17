from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from functools import lru_cache

import numpy as np

from stat_arb_bot.models.numeric import readonly_array, sign_aligned_unit_vector
from stat_arb_bot.models.state_space import ECMDrivenKalmanFilter, build_vecm_state_space
from stat_arb_bot.models.structural import StructuralFitResult
from stat_arb_bot.models.vecm import ECMStatistics

from model_helpers import simulate_rank_one_vecm
from state_space_helpers import rank_one_fit


CONTROLLED_BETA = np.asarray([1.0, -1.0, 0.5, -0.3, -0.2])


@lru_cache(maxsize=1)
def _base() -> tuple[StructuralFitResult, np.ndarray]:
    levels, _, _ = simulate_rank_one_vecm(
        1000,
        beta=CONTROLLED_BETA,
        alpha=np.asarray([-0.20, 0.0, 0.0, 0.0, 0.0]),
        gamma=0.0,
        seed=707,
    )
    return rank_one_fit(levels, var_order=2, fit_id="controlled-role-fit")


def controlled_fit(
    *,
    alpha: np.ndarray | None = None,
    alpha_standard_errors: np.ndarray | None = None,
    gamma_edges: dict[tuple[int, int], float] | None = None,
) -> tuple[StructuralFitResult, np.ndarray]:
    base, levels = _base()
    vecm = base.rank_one
    assert vecm is not None
    adjustment = np.asarray(
        alpha if alpha is not None else [-0.20, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    standard_errors = np.asarray(
        alpha_standard_errors if alpha_standard_errors is not None else np.full(5, 0.02),
        dtype=np.float64,
    )
    gamma = np.zeros((5, 5), dtype=np.float64)
    for (target, source), coefficient in (gamma_edges or {(0, 1): 0.40}).items():
        gamma[target, source] = coefficient
    gamma_standard_error = np.full((5, 5), 0.02, dtype=np.float64)
    gamma_covariance = np.eye(25, dtype=np.float64) * 0.02**2
    reporting_beta, scale = sign_aligned_unit_vector(CONTROLLED_BETA)
    ecm_values = levels @ CONTROLLED_BETA
    ecm = ECMStatistics(
        values=readonly_array(ecm_values, dimensions=1),
        current=float(ecm_values[-1]),
        mean=float(np.mean(ecm_values)),
        standard_deviation=float(np.std(ecm_values, ddof=1)),
        median=float(np.median(ecm_values)),
        median_absolute_deviation=float(np.median(np.abs(ecm_values - np.median(ecm_values)))),
        robust_spread=float(1.4826 * np.median(np.abs(ecm_values - np.median(ecm_values)))),
        z_score=None,
        persistence=vecm.ecm.persistence,
    )
    replacement = replace(
        vecm,
        native_beta=readonly_array(CONTROLLED_BETA, dimensions=1),
        native_cointegration_constant=0.0,
        native_alpha=readonly_array(adjustment, dimensions=1),
        native_pi=readonly_array(np.outer(adjustment, CONTROLLED_BETA), dimensions=2),
        reporting_beta=reporting_beta,
        reporting_cointegration_constant=0.0,
        reporting_alpha=readonly_array(adjustment / scale, dimensions=1),
        reporting_scale=scale,
        alpha_standard_errors=readonly_array(standard_errors, dimensions=1),
        gamma_matrices=(readonly_array(gamma, dimensions=2),),
        gamma_standard_errors=(readonly_array(gamma_standard_error, dimensions=2),),
        gamma_parameter_covariance=readonly_array(gamma_covariance, dimensions=2),
        ecm=ecm,
        current_error_correction_contribution=readonly_array(
            adjustment * ecm.current,
            dimensions=1,
        ),
        current_adjustment_signs=tuple(int(np.sign(value)) for value in adjustment * ecm.current),
    )
    return replace(base, rank_one=replacement), levels.copy()


def controlled_filter(
    fit: StructuralFitResult,
    levels: np.ndarray,
    *,
    current_ecm: float = 0.10,
    previous_change: np.ndarray | None = None,
) -> ECMDrivenKalmanFilter:
    availability = build_vecm_state_space(fit)
    assert availability.model is not None
    vecm = fit.rank_one
    assert vecm is not None
    current = levels[-1].copy()
    current[0] += (current_ecm - float(current @ vecm.native_beta)) / vecm.native_beta[0]
    change = np.zeros(5) if previous_change is None else np.asarray(previous_change, dtype=float)
    timestamp = fit.metadata.fit_timestamp
    return ECMDrivenKalmanFilter.initialize(
        availability.model,
        initialization_timestamp=timestamp,
        history_timestamps=(timestamp - timedelta(minutes=15), timestamp),
        level_history=np.vstack((current - change, current)),
    )
