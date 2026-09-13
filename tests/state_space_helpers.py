from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from stat_arb_bot.models import (
    RankScope,
    StructuralEstimator,
    StructuralModelConfig,
    fit_rank_one_vecm,
)
from stat_arb_bot.models.config import vecm_lagged_differences
from stat_arb_bot.models.structural import StructuralFitResult

from model_helpers import research_window_from_levels, simulate_rank_one_vecm


STATE_CONFIG = StructuralModelConfig(
    minimum_observations=200,
    maximum_var_lag=4,
    convergence_steps=128,
)


def rank_one_fit(
    levels: NDArray[np.float64] | None = None,
    *,
    var_order: int = 2,
    fit_id: str | None = None,
) -> tuple[StructuralFitResult, NDArray[np.float64]]:
    source = levels
    if source is None:
        source, _, _ = simulate_rank_one_vecm(1400)
    window = research_window_from_levels(source)
    base = StructuralEstimator(STATE_CONFIG).fit(
        window,
        fit_timestamp=window.data_end,
        source_fingerprint="a" * 64,
    )
    custom_vecm = fit_rank_one_vecm(
        source,
        assets=window.universe_assets,
        var_order=var_order,
        config=STATE_CONFIG,
    )
    identifier = fit_id or f"synthetic-rank-one-p{var_order}"
    fit = replace(
        base,
        metadata=replace(base.metadata, fit_id=identifier),
        lag_selection=replace(
            base.lag_selection,
            selected_var_order=var_order,
            vecm_lagged_differences=vecm_lagged_differences(var_order),
        ),
        johansen=replace(
            base.johansen,
            inferred_rank=1,
            trace_rank=1,
            var_order=var_order,
            vecm_lagged_differences=vecm_lagged_differences(var_order),
        ),
        rank_scope=RankScope.V1_RANK_ONE,
        rank_one=custom_vecm,
    )
    return fit, source
