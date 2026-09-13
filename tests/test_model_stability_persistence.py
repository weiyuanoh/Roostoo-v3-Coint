from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np

from stat_arb_bot.models import (
    RankScope,
    StructuralEstimator,
    StructuralModelConfig,
    StructuralModelStore,
    compare_structural_fits,
    compare_windows,
)

from model_helpers import (
    MODEL_BASE,
    research_dataset_from_levels,
    research_window_from_levels,
    simulate_rank_one_vecm,
)


def _config() -> StructuralModelConfig:
    return StructuralModelConfig(
        primary_window=timedelta(days=2),
        comparison_windows=(timedelta(days=1), timedelta(days=2), timedelta(days=3)),
        minimum_observations=80,
        maximum_var_lag=3,
        convergence_steps=64,
    )


def _direct_relation_system(observations: int, *, second_relation: bool) -> np.ndarray:
    rng = np.random.default_rng(7823 if second_relation else 7822)
    trends = np.cumsum(rng.normal(0, 0.006, size=(observations, 4)), axis=0)
    error = np.empty(observations)
    error[0] = 0
    shocks = rng.normal(0, 0.003, size=observations)
    for index in range(1, observations):
        error[index] = 0.65 * error[index - 1] + shocks[index]
    bases = np.log([50000.0, 3000.0, 120.0, 0.6])
    first_four = trends + bases
    if second_relation:
        fifth = first_four[:, 0] + first_four[:, 1] + error
    else:
        fifth = first_four[:, 0] - first_four[:, 1] + error
    return np.column_stack((first_four, fifth))


def test_structural_break_produces_material_beta_instability() -> None:
    config = _config()
    first = _direct_relation_system(700, second_relation=False)
    second = _direct_relation_system(700, second_relation=True)
    fit_a = StructuralEstimator(config).fit(
        research_window_from_levels(first),
        fit_timestamp=MODEL_BASE + timedelta(minutes=15 * len(first)),
        source_fingerprint="3" * 64,
        fit_id="before-break",
    )
    fit_b = StructuralEstimator(config).fit(
        research_window_from_levels(second),
        fit_timestamp=MODEL_BASE + timedelta(minutes=15 * len(second)),
        source_fingerprint="4" * 64,
        fit_id="after-break",
    )

    assert fit_a.rank_scope is RankScope.V1_RANK_ONE
    assert fit_b.rank_scope is RankScope.V1_RANK_ONE
    comparison = compare_structural_fits(fit_a, fit_b)
    assert comparison.rank_stable
    assert comparison.beta_cosine_similarity is not None
    assert comparison.beta_cosine_similarity < 0.75


def test_cross_window_fits_remain_separate_and_identify_primary() -> None:
    levels, _, _ = simulate_rank_one_vecm(400)
    dataset = research_dataset_from_levels(levels)
    config = _config()

    comparison = compare_windows(
        dataset,
        fit_timestamp=dataset.panel.end,
        config=config,
    )

    assert comparison.primary_window == timedelta(days=2)
    assert comparison.primary is not None
    assert tuple(item.actual_observations for item in comparison.fits) == (96, 192, 288)
    assert all(
        item.fit is not comparison.primary
        for item in comparison.fits
        if item.window != timedelta(days=2)
    )


def test_versioned_json_persistence_is_safe_and_inspectable(tmp_path: Path) -> None:
    levels, _, _ = simulate_rank_one_vecm(1400)
    window = research_window_from_levels(levels)
    fit = StructuralEstimator(_config()).fit(
        window,
        fit_timestamp=window.data_end,
        source_fingerprint="5" * 64,
    )
    store = StructuralModelStore(tmp_path)

    path = store.save(fit)
    payload = store.load_payload(fit.metadata.fit_id)

    assert path.suffix == ".json"
    assert payload["schema_version"] == 1
    assert payload["model_name"] == "five_asset_vecm"
    assert payload["fit"]["metadata"]["fit_id"] == fit.metadata.fit_id
    assert payload["fit"]["johansen"]["inferred_rank"] == fit.johansen.inferred_rank
    assert fit.rank_one is not None
    assert payload["fit"]["rank_one"]["native_beta"] == fit.rank_one.native_beta.tolist()
    assert not list(tmp_path.rglob("*.pkl"))
