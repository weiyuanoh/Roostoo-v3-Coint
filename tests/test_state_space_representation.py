from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from stat_arb_bot.models import StructuralEstimator
from stat_arb_bot.models.state_space.config import (
    KalmanConfig,
    MeasurementNoiseConfig,
    MeasurementNoiseMode,
)
from stat_arb_bot.models.state_space.filter import ECMDrivenKalmanFilter
from stat_arb_bot.models.state_space.numeric import StateSpaceNumericalError
from stat_arb_bot.models.state_space.representation import (
    StateSpaceStatus,
    build_vecm_state_space,
)

from model_helpers import (
    independent_random_walks,
    one_common_trend_system,
    research_window_from_levels,
)
from state_space_helpers import STATE_CONFIG, rank_one_fit


@pytest.mark.parametrize("var_order", [1, 2, 4])
def test_augmented_transition_exactly_reproduces_direct_vecm_recursion(var_order: int) -> None:
    fit, levels = rank_one_fit(var_order=var_order)
    availability = build_vecm_state_space(fit)
    assert availability.model is not None
    model = availability.model
    state = model.state_from_history(levels)

    transitioned = model.deterministic_transition(state)
    direct = fit.rank_one.decompose(levels).conditional_mean_change
    expected_price = levels[-1] + direct

    assert model.layout.dimension == 5 * var_order
    assert np.allclose(transitioned[model.layout.price_slice], expected_price, atol=1e-12)
    if var_order > 1:
        assert np.allclose(transitioned[model.layout.difference_slice(1)], direct, atol=1e-12)
    for lag in range(2, var_order):
        assert np.allclose(
            transitioned[model.layout.difference_slice(lag)],
            state[model.layout.difference_slice(lag - 1)],
            atol=1e-12,
        )


def test_error_correction_gamma_and_deterministic_terms_remain_inspectable() -> None:
    fit, levels = rank_one_fit(var_order=3)
    model = build_vecm_state_space(fit).model
    assert model is not None
    state = model.state_from_history(levels)

    state_parts = model.decompose_state(state)
    history_parts = fit.rank_one.decompose(levels)

    assert np.allclose(state_parts.error_correction, history_parts.error_correction)
    assert np.allclose(state_parts.short_run, history_parts.short_run)
    assert np.allclose(state_parts.deterministic, history_parts.deterministic)
    assert np.allclose(state_parts.conditional_mean_change, history_parts.conditional_mean_change)


def test_process_noise_loading_preserves_price_difference_cross_covariance() -> None:
    fit, levels = rank_one_fit(var_order=3)
    model = build_vecm_state_space(fit).model
    assert model is not None
    n = model.layout.asset_count
    sigma = fit.rank_one.residual_covariance
    q = model.process_covariance

    assert model.innovation_loading.shape == (15, 5)
    assert np.allclose(q[:n, :n], sigma)
    assert np.allclose(q[:n, n : 2 * n], sigma)
    assert np.allclose(q[n : 2 * n, :n], sigma)
    assert np.allclose(q[n : 2 * n, n : 2 * n], sigma)
    assert np.allclose(q[2 * n :, :], 0)
    assert np.min(np.linalg.eigvalsh(q)) >= -1e-10


def test_measurement_matrix_and_all_explicit_noise_modes() -> None:
    fit, _ = rank_one_fit(var_order=2)
    fraction = build_vecm_state_space(fit).model
    diagonal = build_vecm_state_space(
        fit,
        KalmanConfig(
            measurement_noise=MeasurementNoiseConfig(
                mode=MeasurementNoiseMode.EXPLICIT_DIAGONAL,
                diagonal_variances=(1e-7,) * 5,
            )
        ),
    ).model
    full_covariance = tuple(
        tuple(1e-7 if left == right else 2e-8 for right in range(5)) for left in range(5)
    )
    full = build_vecm_state_space(
        fit,
        KalmanConfig(
            measurement_noise=MeasurementNoiseConfig(
                mode=MeasurementNoiseMode.EXPLICIT_FULL,
                covariance=full_covariance,
            )
        ),
    ).model

    assert fraction is not None and diagonal is not None and full is not None
    assert np.array_equal(fraction.measurement_matrix[:, :5], np.eye(5))
    assert np.array_equal(fraction.measurement_matrix[:, 5:], np.zeros((5, 5)))
    assert np.allclose(
        np.diag(fraction.measurement_noise.covariance),
        np.diag(fit.rank_one.residual_covariance) * 0.01,
    )
    assert np.allclose(np.diag(diagonal.measurement_noise.covariance), 1e-7)
    assert np.allclose(full.measurement_noise.covariance, full_covariance)


def test_invalid_measurement_covariance_is_rejected_without_diagonal_replacement() -> None:
    fit, _ = rank_one_fit()
    invalid = tuple(tuple(-1.0 if i == j == 0 else 0.0 for j in range(5)) for i in range(5))

    with pytest.raises(StateSpaceNumericalError, match="negative diagonal"):
        build_vecm_state_space(
            fit,
            KalmanConfig(
                measurement_noise=MeasurementNoiseConfig(
                    mode=MeasurementNoiseMode.EXPLICIT_FULL,
                    covariance=invalid,
                )
            ),
        )


def test_non_psd_covariance_and_exploding_state_fail_loudly() -> None:
    fit, levels = rank_one_fit(var_order=2)
    model = build_vecm_state_space(fit).model
    assert model is not None
    invalid_covariance = np.eye(model.layout.dimension)
    invalid_covariance[0, 0] = -1

    with pytest.raises(StateSpaceNumericalError, match="negative diagonal"):
        ECMDrivenKalmanFilter(
            model,
            timestamp=fit.metadata.data_end_timestamp,
            state_mean=model.state_from_history(levels),
            state_covariance=invalid_covariance,
            last_observation=levels[-1],
        )
    with pytest.raises(StateSpaceNumericalError, match="exploding"):
        model.deterministic_transition(np.full(model.layout.dimension, 1e7))


def test_rank_zero_and_multiple_rank_return_structured_inactive_states() -> None:
    rank_zero_levels = independent_random_walks(1400)
    rank_zero_window = research_window_from_levels(rank_zero_levels)
    rank_zero = StructuralEstimator(STATE_CONFIG).fit(
        rank_zero_window,
        fit_timestamp=rank_zero_window.data_end,
        source_fingerprint="b" * 64,
    )
    rank_many_levels = one_common_trend_system()
    rank_many_window = research_window_from_levels(rank_many_levels)
    rank_many = StructuralEstimator(STATE_CONFIG).fit(
        rank_many_window,
        fit_timestamp=rank_many_window.data_end,
        source_fingerprint="c" * 64,
    )

    unavailable_zero = build_vecm_state_space(rank_zero)
    unavailable_many = build_vecm_state_space(rank_many)

    assert unavailable_zero.status is StateSpaceStatus.INACTIVE_RANK_ZERO
    assert unavailable_zero.model is None
    assert unavailable_many.status is StateSpaceStatus.INACTIVE_MULTIPLE_RANK
    assert unavailable_many.model is None


def test_beta_scale_and_sign_do_not_change_transition_dynamics() -> None:
    fit, levels = rank_one_fit(var_order=3)
    scale = -3.7
    scaled_vecm = replace(
        fit.rank_one,
        native_beta=fit.rank_one.native_beta * scale,
        native_cointegration_constant=fit.rank_one.native_cointegration_constant * scale,
        native_alpha=fit.rank_one.native_alpha / scale,
    )
    scaled_fit = replace(fit, rank_one=scaled_vecm)

    original = build_vecm_state_space(fit).model
    scaled = build_vecm_state_space(scaled_fit).model

    assert original is not None and scaled is not None
    assert np.allclose(original.transition_matrix, scaled.transition_matrix)
    assert np.allclose(original.transition_intercept, scaled.transition_intercept)
    assert np.allclose(original.process_covariance, scaled.process_covariance)

    timestamps = tuple(
        fit.metadata.data_end_timestamp - timedelta(minutes=15 * index)
        for index in reversed(range(len(levels)))
    )
    original_filter = ECMDrivenKalmanFilter.initialize(
        original,
        initialization_timestamp=fit.metadata.data_end_timestamp,
        history_timestamps=timestamps,
        level_history=levels,
    )
    scaled_filter = ECMDrivenKalmanFilter.initialize(
        scaled,
        initialization_timestamp=fit.metadata.data_end_timestamp,
        history_timestamps=timestamps,
        level_history=levels,
    )
    original_forecast = original_filter.forecast(16)
    scaled_forecast = scaled_filter.forecast(16)
    assert np.allclose(original_forecast.state_mean, scaled_forecast.state_mean)
    assert np.allclose(original_forecast.latent_state_change, scaled_forecast.latent_state_change)
    assert np.allclose(
        scaled_forecast.expected_ecm_path,
        original_forecast.expected_ecm_path * scale,
    )
