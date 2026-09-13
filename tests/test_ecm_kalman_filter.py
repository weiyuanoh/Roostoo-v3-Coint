from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import numpy as np
import pytest

from stat_arb_bot.backtest.orders import Order, OrderType, TradePhase
from stat_arb_bot.domain import Side
from stat_arb_bot.models.state_space.baselines import (
    PlainVECMForecaster,
    RandomWalkKalmanFilter,
)
from stat_arb_bot.models.state_space.config import KalmanConfig, MeasurementNoiseConfig
from stat_arb_bot.models.state_space.filter import ECMDrivenKalmanFilter
from stat_arb_bot.models.state_space.representation import build_vecm_state_space

from model_helpers import MODEL_BASE, simulate_rank_one_vecm
from state_space_helpers import rank_one_fit


def _timestamps(count: int) -> tuple:
    return tuple(MODEL_BASE + timedelta(minutes=15 * (index + 1)) for index in range(count))


def _initialized_filter(
    *,
    var_order: int = 2,
    measurement_fraction: float = 0.01,
) -> tuple[ECMDrivenKalmanFilter, np.ndarray, object]:
    levels, _, _ = simulate_rank_one_vecm(1100)
    fit, _ = rank_one_fit(levels[:900], var_order=var_order)
    availability = build_vecm_state_space(
        fit,
        KalmanConfig(
            measurement_noise=MeasurementNoiseConfig(
                residual_variance_fraction=measurement_fraction
            )
        ),
    )
    assert availability.model is not None
    model = availability.model
    filter_ = ECMDrivenKalmanFilter.initialize(
        model,
        initialization_timestamp=_timestamps(900)[-1],
        history_timestamps=_timestamps(900),
        level_history=levels[:900],
    )
    return filter_, levels, fit


def test_prediction_occurs_before_current_observation_and_update_uses_only_y_k() -> None:
    first, levels, _ = _initialized_filter()
    second, _, _ = _initialized_filter()
    prior_first = first.predict_prior()
    prior_second = second.predict_prior()
    timestamp = _timestamps(901)[-1]
    observation_a = levels[900]
    observation_b = levels[900] + np.asarray([0.05, 0, 0, 0, 0])

    result_a = first.step(timestamp=timestamp, observation=observation_a)
    result_b = second.step(timestamp=timestamp, observation=observation_b)

    assert np.array_equal(prior_first.mean, prior_second.mean)
    assert np.array_equal(result_a.predicted_state_mean, prior_first.mean)
    assert np.array_equal(result_b.predicted_state_mean, prior_second.mean)
    assert not np.array_equal(result_a.filtered_state_mean, result_b.filtered_state_mean)
    assert np.allclose(result_a.innovation, observation_a - result_a.predicted_observation)


def test_future_observations_cannot_change_filter_result_at_k() -> None:
    filter_a, levels, _ = _initialized_filter()
    filter_b, _, _ = _initialized_filter()
    at_k = _timestamps(901)[-1]

    result_a = filter_a.step(timestamp=at_k, observation=levels[900])
    result_b = filter_b.step(timestamp=at_k, observation=levels[900])
    _unused_future_a = levels[901:]
    _unused_future_b = levels[901:] + 100.0

    assert np.array_equal(result_a.predicted_state_mean, result_b.predicted_state_mean)
    assert np.array_equal(result_a.filtered_state_mean, result_b.filtered_state_mean)
    assert np.array_equal(result_a.innovation, result_b.innovation)
    assert np.array_equal(result_a.next_expected_change, result_b.next_expected_change)


def test_step_exposes_ecm_alpha_gamma_and_valid_covariances() -> None:
    filter_, levels, fit = _initialized_filter(var_order=3)
    result = filter_.step(timestamp=_timestamps(901)[-1], observation=levels[900])

    assert result.structural_fit_id == fit.metadata.fit_id
    assert result.structural_fit_data_end <= result.timestamp
    assert result.ecm.observed == pytest.approx(
        levels[900] @ fit.rank_one.native_beta + fit.rank_one.native_cointegration_constant
    )
    assert np.allclose(
        result.observed_error_correction_contribution,
        fit.rank_one.native_alpha * result.ecm.observed,
    )
    assert np.allclose(
        result.filtered_error_correction_contribution,
        fit.rank_one.native_alpha * result.ecm.filtered,
    )
    assert np.allclose(
        result.next_expected_change,
        result.next_error_correction_contribution
        + result.next_short_run_contribution
        + result.next_deterministic_contribution,
    )
    assert np.min(np.linalg.eigvalsh(result.filtered_covariance)) >= -1e-10
    assert np.min(np.linalg.eigvalsh(result.innovation_covariance)) > 0
    assert result.chronology_fields()["filter_timestamp"] == result.timestamp

    order = Order(
        order_id="chronology-only",
        symbol="BTC/USDT",
        side=Side.BUY,
        quantity=Decimal("1"),
        order_type=OrderType.MARKET,
        signal_information_timestamp=result.timestamp,
        signal_timestamp=result.timestamp,
        submission_timestamp=result.timestamp,
        first_eligible_execution_timestamp=result.timestamp + timedelta(microseconds=1),
        trade_group_id=None,
        phase=TradePhase.ADJUST,
        model_version=result.filter_version,
        model_fit_end_timestamp=result.structural_fit_data_end,
    )
    assert (
        order.model_fit_end_timestamp <= result.timestamp < order.first_eligible_execution_timestamp
    )


def test_one_and_multi_step_forecasts_have_distinct_change_definitions_and_valid_covariance() -> (
    None
):
    filter_, levels, _ = _initialized_filter(var_order=3)
    result = filter_.step(timestamp=_timestamps(901)[-1], observation=levels[900] + 0.001)
    forecast = filter_.forecast(16)

    assert forecast.horizon_steps == 16
    assert forecast.elapsed == timedelta(hours=4)
    assert forecast.elapsed_hours == 4
    assert forecast.price_mean.shape == (5,)
    assert forecast.price_covariance.shape == (5, 5)
    assert np.allclose(forecast.price_covariance, forecast.price_covariance.T)
    assert np.all(np.diag(forecast.price_covariance) >= 0)
    assert np.min(np.linalg.eigvalsh(forecast.price_covariance)) >= -1e-10
    current_latent = result.filtered_state_mean[:5]
    assert np.allclose(forecast.latent_state_change, forecast.price_mean - current_latent)
    assert np.allclose(
        forecast.observed_market_relative_change,
        forecast.price_mean - result.observed_log_prices,
    )
    assert len(forecast.expected_ecm_path) == 17
    assert {4, 16, 32, 64, 96}.issubset(filter_.diagnostic_horizons())


def test_state_space_forecast_matches_checkpoint5_deterministic_vecm_path() -> None:
    filter_, levels, fit = _initialized_filter(var_order=4)
    horizon = 32

    forecast = filter_.forecast(horizon)
    direct = fit.rank_one.simulate_zero_shock(
        steps=horizon,
        level_history=levels[:900],
    )
    plain = PlainVECMForecaster(filter_.model).forecast(
        timestamp=filter_.timestamp,
        level_history=levels[:900],
        horizon_steps=horizon,
    )

    assert np.allclose(forecast.expected_ecm_path, direct, atol=1e-11)
    assert np.allclose(forecast.price_mean, plain.price_mean, atol=1e-11)
    assert np.allclose(plain.expected_ecm_path, direct, atol=1e-11)


def test_known_restoring_force_changes_vecm_forecast_while_random_walk_stays_flat() -> None:
    beta = np.asarray([1.0, -1.0, 0.0, 0.0, 0.0])
    alpha = np.asarray([-0.3, 0.0, 0.0, 0.0, 0.0])
    levels, _, _ = simulate_rank_one_vecm(1000, beta=beta, alpha=alpha, gamma=0.0, seed=411)
    fit, _ = rank_one_fit(levels[:900], var_order=1)
    model = build_vecm_state_space(fit).model
    assert model is not None
    filter_ = ECMDrivenKalmanFilter.initialize(
        model,
        initialization_timestamp=_timestamps(900)[-1],
        history_timestamps=_timestamps(900),
        level_history=levels[:900],
    )
    displaced = levels[900].copy()
    displaced[0] += 0.08
    result = filter_.step(timestamp=_timestamps(901)[-1], observation=displaced)
    vecm_forecast = filter_.forecast(1)
    random_walk = RandomWalkKalmanFilter(
        timestamp=filter_.timestamp,
        observation=result.filtered_state_mean[:5],
        process_covariance=fit.rank_one.residual_covariance,
        measurement_noise=model.measurement_noise,
    )
    random_forecast = random_walk.forecast(1, structural_fit_id=fit.metadata.fit_id)

    assert result.ecm.filtered > 0
    assert fit.rank_one.native_beta @ result.next_error_correction_contribution < 0
    assert abs(vecm_forecast.expected_ecm_path[-1]) < abs(vecm_forecast.expected_ecm_path[0])
    assert np.allclose(random_forecast.forecast_change, 0)
    assert not np.allclose(vecm_forecast.latent_state_change, 0)


def test_filter_tracks_noisy_latent_system_and_preserves_dominant_adjustment_order() -> None:
    beta = np.asarray([1.0, -1.0, 0.0, 0.0, 0.0])
    alpha = np.asarray([-0.24, 0.0, 0.0, 0.0, 0.0])
    latent, _, _ = simulate_rank_one_vecm(1150, beta=beta, alpha=alpha, seed=831)
    fit, _ = rank_one_fit(latent[:900], var_order=2)
    config = KalmanConfig(measurement_noise=MeasurementNoiseConfig(residual_variance_fraction=0.5))
    model = build_vecm_state_space(fit, config).model
    assert model is not None
    rng = np.random.default_rng(832)
    measurement_scale = np.sqrt(np.diag(model.measurement_noise.covariance))
    observed = latent + rng.normal(0, measurement_scale, size=latent.shape)
    filter_ = ECMDrivenKalmanFilter.initialize(
        model,
        initialization_timestamp=_timestamps(900)[-1],
        history_timestamps=_timestamps(900),
        level_history=observed[:900],
    )
    filtered = []
    results = []
    for index in range(900, len(latent)):
        result = filter_.step(timestamp=_timestamps(index + 1)[-1], observation=observed[index])
        results.append(result)
        filtered.append(result.filtered_state_mean[:5])
    filtered_values = np.asarray(filtered)

    observed_rmse = float(np.sqrt(np.mean((observed[900:] - latent[900:]) ** 2)))
    filtered_rmse = float(np.sqrt(np.mean((filtered_values - latent[900:]) ** 2)))
    assert filtered_rmse < observed_rmse
    last_contribution = np.abs(results[-1].filtered_error_correction_contribution)
    assert last_contribution[0] > 3 * max(last_contribution[1:])
    assert all(np.all(np.isfinite(result.filtered_state_mean)) for result in results)


def test_initialization_rejects_future_or_insufficient_lag_history() -> None:
    fit, levels = rank_one_fit(var_order=4)
    model = build_vecm_state_space(fit).model
    assert model is not None
    timestamp = _timestamps(len(levels))[-1]

    with pytest.raises(ValueError, match="end exactly"):
        ECMDrivenKalmanFilter.initialize(
            model,
            initialization_timestamp=timestamp,
            history_timestamps=(*_timestamps(len(levels)), timestamp + timedelta(minutes=15)),
            level_history=np.vstack((levels, levels[-1])),
        )
    with pytest.raises(ValueError, match="insufficient lag history"):
        ECMDrivenKalmanFilter.initialize(
            model,
            initialization_timestamp=timestamp,
            history_timestamps=(timestamp,),
            level_history=levels[-1:],
        )
