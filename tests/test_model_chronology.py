from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from stat_arb_bot.models import StructuralEstimator, StructuralFitScheduler, StructuralModelConfig

from model_helpers import (
    MODEL_BASE,
    independent_random_walks,
    research_dataset_from_levels,
    research_window_from_levels,
    simulate_rank_one_vecm,
)


def _config() -> StructuralModelConfig:
    return StructuralModelConfig(
        primary_window=timedelta(days=2),
        comparison_windows=(timedelta(days=1), timedelta(days=2), timedelta(days=3)),
        fit_cadence=timedelta(hours=24),
        minimum_observations=100,
        maximum_var_lag=3,
        convergence_steps=64,
    )


def test_fit_at_k_cannot_see_or_be_changed_by_observations_after_k() -> None:
    common, _, _ = simulate_rank_one_vecm(700)
    altered = common.copy()
    rng = np.random.default_rng(55)
    altered[500:] += np.cumsum(rng.normal(0, 0.1, size=(200, 5)), axis=0)
    window_a = research_window_from_levels(common, end_index=499)
    window_b = research_window_from_levels(altered, end_index=499)
    estimator = StructuralEstimator(_config())

    fit_a = estimator.fit(
        window_a,
        fit_timestamp=window_a.data_end,
        source_fingerprint="1" * 64,
    )
    fit_b = estimator.fit(
        window_b,
        fit_timestamp=window_b.data_end,
        source_fingerprint="1" * 64,
    )

    assert window_a.log_prices() == window_b.log_prices()
    assert fit_a.metadata == fit_b.metadata
    assert fit_a.lag_selection == fit_b.lag_selection
    assert fit_a.johansen.inferred_rank == fit_b.johansen.inferred_rank
    assert np.array_equal(fit_a.johansen.trace_statistics, fit_b.johansen.trace_statistics)
    assert np.array_equal(fit_a.johansen.eigenvectors, fit_b.johansen.eigenvectors)
    assert fit_a.integration == fit_b.integration
    assert (fit_a.rank_one is None) == (fit_b.rank_one is None)
    if fit_a.rank_one is not None and fit_b.rank_one is not None:
        assert np.array_equal(fit_a.rank_one.native_beta, fit_b.rank_one.native_beta)
        assert np.array_equal(fit_a.rank_one.ecm.values, fit_b.rank_one.ecm.values)


def test_fit_chronology_rejects_future_data() -> None:
    levels = independent_random_walks(300)
    window = research_window_from_levels(levels)

    with pytest.raises(ValueError, match="no later"):
        StructuralEstimator(_config()).fit(
            window,
            fit_timestamp=window.data_end - timedelta(minutes=15),
            source_fingerprint="2" * 64,
        )


def test_rolling_scheduler_is_daily_and_fingerprints_only_through_each_data_end() -> None:
    levels, _, _ = simulate_rank_one_vecm(400)
    dataset = research_dataset_from_levels(levels)
    config = replace(
        _config(),
        primary_window=timedelta(days=1),
        minimum_observations=90,
    )
    scheduler = StructuralFitScheduler(config)
    start = MODEL_BASE + timedelta(days=1)
    end = MODEL_BASE + timedelta(days=3)

    rolling = scheduler.fit_range(dataset, start=start, end=end)

    assert len(rolling.fits) == 3
    assert all(
        right.metadata.fit_timestamp - left.metadata.fit_timestamp >= timedelta(hours=24)
        for left, right in zip(rolling.fits, rolling.fits[1:])
    )
    for fit in rolling.fits:
        assert fit.metadata.data_end_timestamp <= fit.metadata.fit_timestamp
        assert fit.metadata.source_data_fingerprint == dataset.source_fingerprint_through(
            fit.metadata.data_end_timestamp
        )
        assert fit.metadata.observation_count == 96


def test_default_structural_cadence_is_not_every_model_bar() -> None:
    config = StructuralModelConfig()

    assert config.fit_cadence == timedelta(hours=24)
    assert config.primary_window == timedelta(days=30)
    assert config.comparison_windows == (
        timedelta(days=14),
        timedelta(days=30),
        timedelta(days=60),
    )
