from __future__ import annotations

import numpy as np
import pytest

from stat_arb_bot.models import (
    LagCriterion,
    StructuralModelConfig,
    adf_diagnostic,
    diagnose_integration,
    kpss_diagnostic,
    PersistenceDiagnostic,
    select_var_lag,
    vecm_lagged_differences,
)
from stat_arb_bot.models.diagnostics import persistence_diagnostic

from model_helpers import independent_random_walks, research_window_from_levels


def test_level_and_difference_diagnostics_are_structured_for_all_assets() -> None:
    levels = independent_random_walks(500)
    window = research_window_from_levels(levels)
    config = StructuralModelConfig(minimum_observations=100, maximum_var_lag=3)

    diagnostics = diagnose_integration(window, config)

    assert diagnostics.assets == window.universe_assets
    assert diagnostics.sample_count == 500
    assert len(diagnostics.series) == 5
    for item in diagnostics.series:
        assert item.level_count == 500
        assert item.difference_count == 499
        assert item.finite
        assert item.level_adf.test == "ADF"
        assert item.level_kpss.test == "KPSS"
        assert item.difference_adf.p_value is not None
        assert item.difference_kpss.critical_values


def test_unit_root_diagnostics_return_failures_instead_of_opaque_library_errors() -> None:
    adf = adf_diagnostic(np.ones(20))
    kpss = kpss_diagnostic(np.ones(20))

    assert adf.failure is not None
    assert kpss.failure is not None


@pytest.mark.parametrize(("var_order", "expected"), [(1, 0), (2, 1), (8, 7)])
def test_var_order_to_vecm_difference_lag_mapping(var_order: int, expected: int) -> None:
    assert vecm_lagged_differences(var_order) == expected


def test_var_lag_selection_records_every_criterion_and_uses_configured_primary() -> None:
    levels = independent_random_walks(600)
    config = StructuralModelConfig(
        minimum_observations=100,
        maximum_var_lag=4,
        lag_criterion=LagCriterion.BIC,
    )

    result = select_var_lag(levels, config)

    assert tuple(row.var_order for row in result.rows) == (1, 2, 3, 4)
    assert all(
        row.aic is not None and row.bic is not None and row.hqic is not None for row in result.rows
    )
    assert result.selected_var_order == min(result.rows, key=lambda row: row.bic).var_order
    assert result.vecm_lagged_differences == result.selected_var_order - 1
    assert result.sample_count == 600


def test_explosive_ecm_does_not_get_a_clipped_mean_reverting_half_life() -> None:
    result: PersistenceDiagnostic = persistence_diagnostic(
        np.asarray([2.0**index for index in range(20)]),
        config=StructuralModelConfig(minimum_observations=20),
    )

    assert result.ar1_phi is not None and abs(result.ar1_phi) >= 1
    assert result.half_life_observations is None
    assert any("not finite mean reverting" in warning for warning in result.warnings)
