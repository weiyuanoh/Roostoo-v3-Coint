from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from stat_arb_bot.models.roles import (
    ExpressionType,
    LegDirection,
    RoleSelectionConfig,
    RoleStatus,
    classify_roles,
    run_role_history_study,
)
from model_helpers import research_dataset_from_levels
from role_helpers import controlled_filter, controlled_fit


def test_current_pair_result_separates_structural_role_ec_move_and_forecast() -> None:
    fit, levels = controlled_fit()
    filter_ = controlled_filter(fit, levels, current_ecm=0.10)

    result = classify_roles(fit, filter_=filter_)

    assert result.status is RoleStatus.PAIR_CANDIDATE
    assert result.expression_type is ExpressionType.PAIR
    assert result.dominant_corrector == "BTC"
    assert result.leader_anchor == "ETH"
    assert result.filtered_ecm == pytest.approx(0.10)
    assert result.filtered_error_correction_contributions[0] == pytest.approx(-0.02)
    assert result.expected_ecm_at_horizon is not None
    assert abs(result.expected_ecm_at_horizon) < abs(result.filtered_ecm)
    assert tuple((leg.asset, leg.direction) for leg in result.intended_legs) == (
        ("BTC", LegDirection.SHORT),
        ("ETH", LegDirection.LONG),
    )
    forecasts = {item.asset: item for item in result.selected_asset_forecasts}
    assert result.relative_corrector_anchor_forecast == pytest.approx(
        forecasts["BTC"].latent_state_change - forecasts["ETH"].latent_state_change
    )


def test_full_beta_basket_is_an_explicit_separate_benchmark() -> None:
    fit, levels = controlled_fit()
    result = classify_roles(fit, filter_=controlled_filter(fit, levels, current_ecm=0.10))

    benchmark = result.full_cointegrating_basket
    assert benchmark is not None
    assert benchmark.assets == ("BTC", "ETH", "SOL", "XRP", "ADA")
    assert np.allclose(benchmark.native_beta, fit.rank_one.native_beta)
    assert "not the default" in benchmark.description
    assert tuple(leg.asset for leg in result.intended_legs) == ("BTC", "ETH")


def test_destabilizing_asset_is_not_forced_into_a_corrector_role() -> None:
    fit, levels = controlled_fit(alpha=np.asarray([0.20, 0.0, 0.0, 0.0, 0.0]))
    result = classify_roles(
        fit,
        filter_=controlled_filter(fit, levels, current_ecm=0.10),
    )

    assert result.status is RoleStatus.NO_CREDIBLE_CORRECTOR
    assert result.expression_type is ExpressionType.NONE


def test_no_anchor_status_retains_current_ecm_and_corrector_forecast_diagnostics() -> None:
    fit, levels = controlled_fit(
        alpha=np.asarray([-0.20, 0.10, -0.16, 0.10, 0.10]),
        gamma_edges={},
    )
    result = classify_roles(
        fit,
        filter_=controlled_filter(fit, levels, current_ecm=0.10),
    )

    assert result.status is RoleStatus.NO_CREDIBLE_ANCHOR
    assert result.filtered_ecm == pytest.approx(0.10)
    assert result.filtered_error_correction_contributions is not None
    assert result.forecast_horizon_steps is not None
    assert result.expected_ecm_at_horizon is not None
    assert result.selected_asset_forecasts[0].asset == result.dominant_corrector
    assert result.expression_type is ExpressionType.NONE


def test_nonconverging_current_forecast_returns_explicit_status_without_losing_roles() -> None:
    fit, levels = controlled_fit(gamma_edges={(0, 1): 0.40, (0, 0): 1.50})
    filter_ = controlled_filter(
        fit,
        levels,
        current_ecm=0.10,
        previous_change=np.asarray([0.10, 0.0, 0.0, 0.0, 0.0]),
    )

    result = classify_roles(fit, filter_=filter_)

    assert result.status is RoleStatus.NO_EXPECTED_CONVERGENCE
    assert result.dominant_corrector == "BTC"
    assert result.leader_anchor == "ETH"
    assert result.expression_type is ExpressionType.PAIR


def test_direction_conflict_is_reported_after_structural_roles_are_fixed() -> None:
    fit, levels = controlled_fit(
        gamma_edges={(0, 1): 0.40, (1, 1): 0.50},
    )
    filter_ = controlled_filter(
        fit,
        levels,
        current_ecm=0.10,
        previous_change=np.asarray([0.0, 0.10, 0.0, 0.0, 0.0]),
    )

    result = classify_roles(fit, filter_=filter_)

    assert result.status is RoleStatus.CORRECTOR_DIRECTION_CONFLICT
    assert result.filtered_error_correction_contributions[0] < 0
    assert result.selected_asset_forecasts[0].latent_state_change > 0
    assert abs(result.expected_ecm_at_horizon) < abs(result.filtered_ecm)


def test_future_arrays_cannot_contaminate_role_result_at_k() -> None:
    fit, levels = controlled_fit()
    filter_a = controlled_filter(fit, levels, current_ecm=-0.08)
    filter_b = controlled_filter(fit, levels, current_ecm=-0.08)
    future_a = levels.copy()
    future_b = levels.copy()
    future_b[-20:] += 100.0

    result_a = classify_roles(fit, filter_=filter_a)
    result_b = classify_roles(fit, filter_=filter_b)

    assert not np.array_equal(future_a[-20:], future_b[-20:])
    assert result_a.status == result_b.status
    assert result_a.dominant_corrector == result_b.dominant_corrector
    assert result_a.leader_anchor == result_b.leader_anchor
    assert result_a.profile.correction_shares == result_b.profile.correction_shares
    assert result_a.filtered_ecm == result_b.filtered_ecm
    assert result_a.expected_ecm_at_horizon == result_b.expected_ecm_at_horizon
    assert result_a.intended_legs == result_b.intended_legs


def test_filter_fit_mismatch_is_rejected_instead_of_using_stale_roles() -> None:
    fit, levels = controlled_fit()
    other = replace(fit, metadata=replace(fit.metadata, fit_id="new-rank-one-regime"))
    filter_ = controlled_filter(fit, levels)

    with pytest.raises(ValueError, match="different fit IDs"):
        classify_roles(other, filter_=filter_)


def test_role_history_study_is_chronological_and_contains_no_pnl_or_orders() -> None:
    fit, levels = controlled_fit()
    dataset = research_dataset_from_levels(levels)
    # Reuse the immutable fit at its exact bounded timestamp, which is also a
    # completed synchronized panel timestamp.
    study = run_role_history_study(
        dataset.panel,
        [fit],
        role_config=RoleSelectionConfig(asymmetry_dominant_share_threshold=0.40),
    )

    assert study.fit_count == 1
    assert study.rank_one_count == 1
    assert study.dominant_corrector_counts == {"BTC": 1}
    assert study.anchor_counts == {"ETH": 1}
    assert study.median_dominant_share == pytest.approx(1.0)
    assert study.asymmetric_correction_supported
    assert study.pair_candidate_count == 1
    assert study.minimal_basket_candidate_count == 0
    assert study.no_valid_candidate_count == 0
    assert not hasattr(study, "pnl")
    assert not hasattr(study.results[0], "orders")


def test_role_history_rejects_fit_timestamp_absent_from_completed_panel() -> None:
    fit, levels = controlled_fit()
    dataset = research_dataset_from_levels(levels)
    shifted = replace(
        fit,
        metadata=replace(
            fit.metadata,
            fit_timestamp=fit.metadata.fit_timestamp.replace(microsecond=1),
        ),
    )

    with pytest.raises(ValueError, match="completed panel observation"):
        run_role_history_study(dataset.panel, [shifted])
