from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from stat_arb_bot.models import StructuralEstimator
from stat_arb_bot.models.state_space.filter import ECMDrivenKalmanFilter
from stat_arb_bot.models.state_space.persistence import FilterStateStore, RegimeStateStore
from stat_arb_bot.models.state_space.regime import (
    RegimeTransitionAction,
    StructuralRegimeManager,
)
from stat_arb_bot.models.state_space.representation import build_vecm_state_space

from model_helpers import (
    MODEL_BASE,
    independent_random_walks,
    one_common_trend_system,
    research_window_from_levels,
    simulate_rank_one_vecm,
)
from state_space_helpers import STATE_CONFIG, rank_one_fit


def _timestamps(count: int) -> tuple:
    return tuple(MODEL_BASE + timedelta(minutes=15 * (index + 1)) for index in range(count))


def test_rank_and_refit_handoffs_rebuild_layout_and_never_keep_stale_dynamics() -> None:
    levels, _, _ = simulate_rank_one_vecm(1500)
    first_fit, _ = rank_one_fit(levels[:1300], var_order=2, fit_id="rank-one-p2")
    changed_fit, _ = rank_one_fit(levels[:1300], var_order=4, fit_id="rank-one-p4")
    zero_levels = independent_random_walks(1400)
    zero_window = research_window_from_levels(zero_levels)
    zero_fit = StructuralEstimator(STATE_CONFIG).fit(
        zero_window,
        fit_timestamp=zero_window.data_end,
        source_fingerprint="0" * 64,
        fit_id="rank-zero",
    )
    many_levels = one_common_trend_system()
    many_window = research_window_from_levels(many_levels)
    many_fit = StructuralEstimator(STATE_CONFIG).fit(
        many_window,
        fit_timestamp=many_window.data_end,
        source_fingerprint="9" * 64,
        fit_id="rank-many",
    )
    manager = StructuralRegimeManager()
    initial_time = first_fit.metadata.data_end_timestamp

    initial = manager.apply_fit(
        first_fit,
        timestamp=initial_time,
        history_timestamps=_timestamps(1300),
        level_history=levels[:1300],
    )
    assert initial.action is RegimeTransitionAction.ACTIVATE
    assert manager.active_filter is not None
    manager.active_filter.step(timestamp=_timestamps(1301)[-1], observation=levels[1300])
    carried_price = manager.active_filter.state_mean[:5].copy()
    changed = manager.apply_fit(
        changed_fit,
        timestamp=_timestamps(1301)[-1],
        history_timestamps=_timestamps(1301),
        level_history=levels[:1301],
    )

    assert changed.action is RegimeTransitionAction.REFIT_CARRY_PRICE
    assert changed.previous_dimension == 10
    assert changed.new_dimension == 20
    assert changed.carried_filtered_price
    assert manager.active_filter is not None
    assert np.array_equal(manager.active_filter.state_mean[:5], carried_price)

    left_rank_one = manager.apply_fit(
        zero_fit,
        timestamp=_timestamps(1400)[-1],
        history_timestamps=_timestamps(1400),
        level_history=levels[:1400],
    )
    assert left_rank_one.action is RegimeTransitionAction.DEACTIVATE_RANK_ZERO
    assert manager.active_filter is None
    assert manager.inactive_state is not None
    assert manager.inactive_state.final_active_checkpoint is not None

    remained_inactive = manager.apply_fit(
        many_fit,
        timestamp=_timestamps(1401)[-1],
        history_timestamps=_timestamps(1401),
        level_history=levels[:1401],
    )
    assert remained_inactive.action is RegimeTransitionAction.REMAIN_INACTIVE
    assert manager.active_filter is None

    returned = manager.apply_fit(
        first_fit,
        timestamp=_timestamps(1402)[-1],
        history_timestamps=_timestamps(1402),
        level_history=levels[:1402],
    )
    assert returned.action is RegimeTransitionAction.ACTIVATE
    assert not returned.carried_filtered_price
    assert manager.active_filter is not None
    assert np.allclose(manager.active_filter.state_mean[:5], levels[1401])

    multiple_manager = StructuralRegimeManager()
    multiple_manager.apply_fit(
        first_fit,
        timestamp=initial_time,
        history_timestamps=_timestamps(1300),
        level_history=levels[:1300],
    )
    to_multiple = multiple_manager.apply_fit(
        many_fit,
        timestamp=_timestamps(1400)[-1],
        history_timestamps=_timestamps(1400),
        level_history=levels[:1400],
    )
    assert to_multiple.action is RegimeTransitionAction.DEACTIVATE_MULTIPLE_RANK
    assert multiple_manager.active_filter is None

    zero_manager = StructuralRegimeManager()
    zero_manager.apply_fit(
        zero_fit,
        timestamp=_timestamps(1400)[-1],
        history_timestamps=_timestamps(1400),
        level_history=levels[:1400],
    )
    from_zero = zero_manager.apply_fit(
        first_fit,
        timestamp=_timestamps(1401)[-1],
        history_timestamps=_timestamps(1401),
        level_history=levels[:1401],
    )
    assert from_zero.action is RegimeTransitionAction.ACTIVATE
    assert zero_manager.active_filter is not None


def test_inactive_rank_regime_status_round_trips(tmp_path: Path) -> None:
    zero_levels = independent_random_walks(1400)
    zero_window = research_window_from_levels(zero_levels)
    zero_fit = StructuralEstimator(STATE_CONFIG).fit(
        zero_window,
        fit_timestamp=zero_window.data_end,
        source_fingerprint="7" * 64,
        fit_id="persisted-rank-zero",
    )
    manager = StructuralRegimeManager()
    manager.apply_fit(
        zero_fit,
        timestamp=zero_window.data_end,
        history_timestamps=zero_window.timestamps,
        level_history=zero_window.log_prices().values,
    )
    store = RegimeStateStore(tmp_path)

    path = store.save(manager.checkpoint())
    loaded = store.load(zero_fit.metadata.fit_id)
    restored = StructuralRegimeManager.from_checkpoint(loaded, zero_fit)

    assert path.suffix == ".json"
    assert restored.active_filter is None
    assert restored.inactive_state is not None
    assert restored.inactive_state.rank == 0
    assert restored.inactive_state.structural_fit_id == zero_fit.metadata.fit_id
    assert loaded.config == restored.config


def test_persist_restart_continue_matches_uninterrupted_filter(tmp_path: Path) -> None:
    levels, _, _ = simulate_rank_one_vecm(1000)
    fit, _ = rank_one_fit(levels[:900], var_order=3)
    model = build_vecm_state_space(fit).model
    assert model is not None
    continuous = ECMDrivenKalmanFilter.initialize(
        model,
        initialization_timestamp=_timestamps(900)[-1],
        history_timestamps=_timestamps(900),
        level_history=levels[:900],
    )
    continuous.step(timestamp=_timestamps(901)[-1], observation=levels[900])
    store = FilterStateStore(tmp_path)
    path = store.save(continuous.checkpoint())
    loaded = store.load(fit.metadata.fit_id)
    restarted = ECMDrivenKalmanFilter.from_checkpoint(model, loaded)

    uninterrupted_result = continuous.step(
        timestamp=_timestamps(902)[-1],
        observation=levels[901],
    )
    restarted_result = restarted.step(
        timestamp=_timestamps(902)[-1],
        observation=levels[901],
    )

    assert path.suffix == ".json"
    assert loaded.config == model.config
    assert np.array_equal(
        uninterrupted_result.predicted_state_mean, restarted_result.predicted_state_mean
    )
    assert np.array_equal(
        uninterrupted_result.filtered_state_mean, restarted_result.filtered_state_mean
    )
    assert np.array_equal(
        uninterrupted_result.filtered_covariance, restarted_result.filtered_covariance
    )
    assert np.array_equal(uninterrupted_result.innovation, restarted_result.innovation)
    assert not list(tmp_path.rglob("*.pkl"))


def test_restart_rejects_different_fit_or_filter_configuration(tmp_path: Path) -> None:
    levels, _, _ = simulate_rank_one_vecm(950)
    fit, _ = rank_one_fit(levels[:900], var_order=2)
    model = build_vecm_state_space(fit).model
    assert model is not None
    filter_ = ECMDrivenKalmanFilter.initialize(
        model,
        initialization_timestamp=_timestamps(900)[-1],
        history_timestamps=_timestamps(900),
        level_history=levels[:900],
    )
    store = FilterStateStore(tmp_path)
    store.save(filter_.checkpoint())
    checkpoint = store.load(fit.metadata.fit_id)
    different_fit, _ = rank_one_fit(levels[:900], var_order=2, fit_id="different-fit")
    different_model = build_vecm_state_space(different_fit).model
    assert different_model is not None

    with pytest.raises(ValueError, match="different structural fit"):
        ECMDrivenKalmanFilter.from_checkpoint(different_model, checkpoint)
