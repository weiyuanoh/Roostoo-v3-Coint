from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from stat_arb_bot.models import (
    RankScope,
    StructuralEstimator,
    StructuralModelConfig,
    estimate_johansen_rank,
    fit_rank_one_vecm,
    select_var_lag,
)

from model_helpers import (
    MODEL_UNIVERSE,
    independent_random_walks,
    one_common_trend_system,
    research_window_from_levels,
    simulate_rank_one_vecm,
)

CONFIG = StructuralModelConfig(minimum_observations=200, maximum_var_lag=4, convergence_steps=128)


def test_independent_random_walk_system_has_no_forced_rank_one_result() -> None:
    levels = independent_random_walks(1400)
    lag = select_var_lag(levels, CONFIG)

    result = estimate_johansen_rank(levels, var_order=lag.selected_var_order, config=CONFIG)

    assert result.inferred_rank == 0
    assert result.trace_statistics.shape == (5,)
    assert result.trace_critical_values.shape == (5, 3)
    assert result.maximum_eigenvalue_statistics.shape == (5,)
    assert result.eigenvectors.shape == (5, 5)


def test_one_common_trend_exposes_multiple_relations_without_arbitrary_reduction() -> None:
    levels = one_common_trend_system()
    window = research_window_from_levels(levels)

    result = StructuralEstimator(CONFIG).fit(
        window,
        fit_timestamp=window.data_end,
        source_fingerprint="f" * 64,
    )

    assert result.johansen.inferred_rank == 4
    assert result.rank_scope is RankScope.MULTIPLE_RELATIONS_OUTSIDE_V1
    assert result.rank_one is None
    assert result.johansen.eigenvectors.shape == (5, 5)


def test_known_rank_one_system_recovers_beta_subspace_and_vecm_shapes() -> None:
    levels, known_beta, _known_alpha = simulate_rank_one_vecm()
    lag = select_var_lag(levels, CONFIG)
    johansen = estimate_johansen_rank(levels, var_order=lag.selected_var_order, config=CONFIG)

    assert johansen.inferred_rank == 1
    fitted = fit_rank_one_vecm(
        levels,
        assets=MODEL_UNIVERSE.assets,
        var_order=lag.selected_var_order,
        config=CONFIG,
    )
    expected = known_beta / np.linalg.norm(known_beta)
    recovered = fitted.reporting_beta
    similarity = abs(float(expected @ recovered))
    assert similarity > 0.94
    anchor = int(np.argmax(np.abs(fitted.reporting_beta)))
    assert fitted.reporting_beta[anchor] > 0
    assert np.linalg.norm(fitted.reporting_beta) == pytest.approx(1.0)
    assert fitted.native_alpha.shape == (5,)
    assert fitted.alpha_standard_errors.shape == (5,)
    assert fitted.beta_standard_errors.shape == (5,)
    assert fitted.native_pi.shape == (5, 5)
    assert len(fitted.gamma_matrices) == lag.selected_var_order - 1
    assert all(matrix.shape == (5, 5) for matrix in fitted.gamma_matrices)
    gamma_standard_errors = (
        np.column_stack(fitted.gamma_standard_errors).reshape(-1, order="F")
        if fitted.gamma_standard_errors
        else np.empty(0)
    )
    assert fitted.gamma_parameter_covariance.shape == (
        len(gamma_standard_errors),
        len(gamma_standard_errors),
    )
    assert np.allclose(
        np.sqrt(np.diag(fitted.gamma_parameter_covariance)),
        gamma_standard_errors,
    )
    assert fitted.residual_covariance.shape == (5, 5)
    assert np.allclose(fitted.residual_covariance, fitted.residual_covariance.T)
    assert np.all(np.diag(fitted.residual_covariance) > 0)
    assert fitted.ecm.persistence.adf.p_value < 0.05


def test_known_adjuster_alpha_and_current_correction_are_interpretable() -> None:
    beta = np.asarray([1.0, -1.0, 0.0, 0.0, 0.0])
    alpha = np.asarray([-0.22, 0.0, 0.0, 0.0, 0.0])
    levels, _, _ = simulate_rank_one_vecm(beta=beta, alpha=alpha, seed=991)
    fitted = fit_rank_one_vecm(
        levels,
        assets=MODEL_UNIVERSE.assets,
        var_order=2,
        config=CONFIG,
    )

    assert fitted.native_alpha[0] < -0.1
    assert abs(fitted.native_alpha[0]) > 3 * max(abs(fitted.native_alpha[1:]))
    assert fitted.current_error_correction_contribution[0] == pytest.approx(
        fitted.native_alpha[0] * fitted.ecm.current
    )
    assert fitted.alpha_by_asset["BTC"] == fitted.native_alpha[0]


def test_decomposition_keeps_error_correction_and_gamma_dynamics_separate() -> None:
    levels, _, _ = simulate_rank_one_vecm(800)
    fitted = fit_rank_one_vecm(
        levels,
        assets=MODEL_UNIVERSE.assets,
        var_order=2,
        config=CONFIG,
    )

    decomposition = fitted.decompose(levels[-3:])

    assert np.allclose(
        decomposition.error_correction,
        fitted.native_alpha
        * (float(levels[-1] @ fitted.native_beta) + fitted.native_cointegration_constant),
    )
    assert np.allclose(
        decomposition.conditional_mean_change,
        decomposition.error_correction + decomposition.short_run + decomposition.deterministic,
    )


def test_beta_scale_and_sign_leave_pi_and_conditional_economics_unchanged() -> None:
    levels, _, _ = simulate_rank_one_vecm(800)
    fitted = fit_rank_one_vecm(
        levels,
        assets=MODEL_UNIVERSE.assets,
        var_order=2,
        config=CONFIG,
    )

    assert np.allclose(
        np.outer(fitted.native_alpha, fitted.native_beta),
        np.outer(fitted.reporting_alpha, fitted.reporting_beta),
    )
    assert fitted.reporting_cointegration_constant == pytest.approx(
        fitted.native_cointegration_constant * fitted.reporting_scale
    )
    native_ecm = float(levels[-1] @ fitted.native_beta) + fitted.native_cointegration_constant
    reporting_ecm = (
        float(levels[-1] @ fitted.reporting_beta) + fitted.reporting_cointegration_constant
    )
    assert reporting_ecm == pytest.approx(native_ecm * fitted.reporting_scale)
    assert np.allclose(
        fitted.native_alpha * native_ecm,
        fitted.reporting_alpha * reporting_ecm,
    )
    assert np.allclose(fitted.native_pi, np.outer(fitted.reporting_alpha, fitted.reporting_beta))


def test_fit_result_is_rank_scoped_and_numpy_parameters_are_immutable() -> None:
    levels, _, _ = simulate_rank_one_vecm(900)
    window = research_window_from_levels(levels)

    result = StructuralEstimator(CONFIG).fit(
        window,
        fit_timestamp=window.data_end,
        source_fingerprint="a" * 64,
    )

    assert result.rank_scope is RankScope.V1_RANK_ONE
    assert result.rank_one is not None
    with pytest.raises(ValueError, match="read-only"):
        result.rank_one.native_beta[0] = 2
    with pytest.raises(Exception):
        replace(result, rank_one=None)


def test_empirical_and_vecm_implied_convergence_are_both_exposed() -> None:
    levels, _, _ = simulate_rank_one_vecm(900)
    fitted = fit_rank_one_vecm(
        levels,
        assets=MODEL_UNIVERSE.assets,
        var_order=2,
        config=CONFIG,
    )

    assert fitted.convergence.simplified_persistence == pytest.approx(
        1.0 + fitted.native_beta @ fitted.native_alpha
    )
    assert len(fitted.convergence.simulated_ecm_path) == CONFIG.convergence_steps + 1
    assert fitted.ecm.persistence.half_life_observations is not None
