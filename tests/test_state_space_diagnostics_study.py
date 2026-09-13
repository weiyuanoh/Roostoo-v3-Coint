from __future__ import annotations

from datetime import timedelta

import numpy as np

from stat_arb_bot.models import StructuralEstimator
from stat_arb_bot.models.state_space.config import KalmanConfig
from stat_arb_bot.models.state_space.diagnostics import summarize_innovations
from stat_arb_bot.models.state_space.filter import ECMDrivenKalmanFilter
from stat_arb_bot.models.state_space.representation import build_vecm_state_space
from stat_arb_bot.models.state_space.study import run_sequential_diagnostic_study

from model_helpers import (
    MODEL_BASE,
    independent_random_walks,
    research_dataset_from_levels,
    research_window_from_levels,
    simulate_rank_one_vecm,
)
from state_space_helpers import STATE_CONFIG, rank_one_fit


def _timestamps(count: int) -> tuple:
    return tuple(MODEL_BASE + timedelta(minutes=15 * (index + 1)) for index in range(count))


def test_innovation_diagnostics_surface_structural_break_instead_of_changing_beta() -> None:
    levels, _, _ = simulate_rank_one_vecm(1100)
    fit, _ = rank_one_fit(levels[:800], var_order=2)
    model = build_vecm_state_space(fit).model
    assert model is not None
    observations = levels.copy()
    observations[950:, 0] += np.arange(1, len(observations) - 949) * 0.025
    filter_ = ECMDrivenKalmanFilter.initialize(
        model,
        initialization_timestamp=_timestamps(800)[-1],
        history_timestamps=_timestamps(800),
        level_history=observations[:800],
    )
    beta_before = model.vecm.native_beta.copy()
    pre_break = []
    post_break = []
    for index in range(800, len(observations)):
        result = filter_.step(
            timestamp=_timestamps(index + 1)[-1],
            observation=observations[index],
        )
        if 850 <= index < 950:
            pre_break.append(result)
        elif index >= 950:
            post_break.append(result)

    normal = summarize_innovations(pre_break, assets=fit.rank_one.assets)
    broken = summarize_innovations(post_break, assets=fit.rank_one.assets)

    assert np.linalg.norm(broken.mean_absolute_error) > 2 * np.linalg.norm(
        normal.mean_absolute_error
    )
    assert np.linalg.norm(broken.mean) > 2 * np.linalg.norm(normal.mean)
    assert np.array_equal(model.vecm.native_beta, beta_before)
    assert 1 in broken.autocorrelation_by_lag


def test_chronological_study_gates_rank_and_compares_three_non_trading_forecasts() -> None:
    levels, _, _ = simulate_rank_one_vecm(850)
    dataset = research_dataset_from_levels(levels)
    first, _ = rank_one_fit(levels[:500], var_order=2, fit_id="active-one")
    zero_levels = independent_random_walks(600)
    zero_window = research_window_from_levels(zero_levels)
    inactive = StructuralEstimator(STATE_CONFIG).fit(
        zero_window,
        fit_timestamp=zero_window.data_end,
        source_fingerprint="6" * 64,
        fit_id="inactive-zero",
    )
    second, _ = rank_one_fit(levels[:700], var_order=2, fit_id="active-two")

    result = run_sequential_diagnostic_study(
        dataset.panel,
        (first, inactive, second),
        config=KalmanConfig(maximum_forecast_steps=96),
        forecast_origin_stride=16,
    )

    assert result.active_structural_fit_ids == ("active-one", "active-two")
    assert result.active_timestamps
    assert result.inactive_timestamps
    assert all(
        first.metadata.fit_timestamp < timestamp < second.metadata.fit_timestamp
        for timestamp in result.inactive_timestamps
    )
    assert {record.model for record in result.forecasts} == {
        "ecm_driven_kf",
        "plain_vecm",
        "random_walk_kf",
    }
    assert all(record.target_timestamp > record.origin_timestamp for record in result.forecasts)
    assert {metric.model for metric in result.metrics} == {
        "ecm_driven_kf",
        "plain_vecm",
        "random_walk_kf",
    }
    assert {4, 16, 32, 64, 96}.issubset({metric.horizon_steps for metric in result.metrics})
    assert result.innovations.sample_count == len(result.filter_steps)
    assert all(step.structural_fit_data_end <= step.timestamp for step in result.filter_steps)
