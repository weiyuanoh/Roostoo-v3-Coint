from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from stat_arb_bot.models import RankScope
from stat_arb_bot.models import StructuralEstimator
from stat_arb_bot.models.numeric import readonly_array
from stat_arb_bot.models.roles import (
    AdjustmentClassification,
    ExpressionType,
    GammaTestStatus,
    RoleSelectionConfig,
    RoleStatus,
    build_structural_role_profile,
    classify_roles,
)

from model_helpers import research_window_from_levels, simulate_rank_one_vecm
from role_helpers import CONTROLLED_BETA, controlled_fit
from state_space_helpers import STATE_CONFIG


def test_rank_gate_returns_explicit_inactive_states_without_stale_roles() -> None:
    fit, _ = controlled_fit()
    rank_zero = replace(
        fit,
        johansen=replace(fit.johansen, inferred_rank=0, trace_rank=0),
        rank_scope=RankScope.NO_COINTEGRATION,
        rank_one=None,
    )
    multiple = replace(
        fit,
        johansen=replace(fit.johansen, inferred_rank=2, trace_rank=2),
        rank_scope=RankScope.MULTIPLE_RELATIONS_OUTSIDE_V1,
        rank_one=None,
    )

    zero_result = classify_roles(rank_zero)
    multiple_result = classify_roles(multiple)

    assert zero_result.status is RoleStatus.INACTIVE_RANK_ZERO
    assert multiple_result.status is RoleStatus.INACTIVE_MULTIPLE_RANK
    assert zero_result.dominant_corrector is None
    assert multiple_result.leader_anchor is None
    assert zero_result.profile.asset_diagnostics == ()


def test_kappa_correction_shares_and_alpha_credibility_are_explicit() -> None:
    fit, _ = controlled_fit()

    profile = build_structural_role_profile(fit)
    by_asset = {item.asset: item for item in profile.asset_diagnostics}

    assert np.allclose(profile.kappa, fit.rank_one.native_beta * fit.rank_one.native_alpha)
    assert profile.beta_alpha_total == pytest.approx(float(np.sum(profile.kappa)))
    assert by_asset["BTC"].kappa == pytest.approx(-0.20)
    assert by_asset["BTC"].correction_share == pytest.approx(1.0)
    assert by_asset["BTC"].adjustment_classification is (
        AdjustmentClassification.CREDIBLE_RESTORING
    )
    assert by_asset["ETH"].weak_adjustment_evidence
    assert by_asset["ETH"].correction_share == 0.0
    assert profile.dominant_corrector == "BTC"


def test_gamma_wald_test_recovers_anchor_to_corrector_relationship() -> None:
    fit, _ = controlled_fit(gamma_edges={(0, 1): 0.40, (2, 3): -0.10})

    profile = build_structural_role_profile(fit)
    relationship = next(
        item
        for item in profile.gamma_diagnostics
        if item.source_asset == "ETH" and item.target_asset == "BTC"
    )
    btc_summary = next(item for item in profile.lead_lag_summaries if item.asset == "BTC")
    eth_summary = next(item for item in profile.lead_lag_summaries if item.asset == "ETH")

    assert relationship.status is GammaTestStatus.TESTED
    assert relationship.lag_set == (1,)
    assert relationship.coefficients == pytest.approx((0.40,))
    assert relationship.significant
    assert relationship.p_value is not None and relationship.p_value < 1e-20
    assert relationship in btc_summary.significant_incoming
    assert relationship in eth_summary.significant_outgoing
    assert profile.leader_anchor == "ETH"


def test_gamma_test_jointly_restricts_every_fitted_lag() -> None:
    fit, _ = controlled_fit()
    vecm = fit.rank_one
    assert vecm is not None
    gamma_one = np.zeros((5, 5))
    gamma_two = np.zeros((5, 5))
    gamma_one[0, 1] = 0.30
    gamma_two[0, 1] = -0.20
    standard_errors = np.full((5, 5), 0.02)
    two_lag_vecm = replace(
        vecm,
        var_order=3,
        lagged_difference_count=2,
        gamma_matrices=(
            readonly_array(gamma_one, dimensions=2),
            readonly_array(gamma_two, dimensions=2),
        ),
        gamma_standard_errors=(
            readonly_array(standard_errors, dimensions=2),
            readonly_array(standard_errors, dimensions=2),
        ),
        gamma_parameter_covariance=readonly_array(np.eye(50) * 0.02**2, dimensions=2),
    )

    profile = build_structural_role_profile(replace(fit, rank_one=two_lag_vecm))
    relationship = next(
        item
        for item in profile.gamma_diagnostics
        if item.source_asset == "ETH" and item.target_asset == "BTC"
    )

    assert relationship.lag_set == (1, 2)
    assert relationship.degrees_of_freedom == 2
    assert relationship.coefficients == pytest.approx((0.30, -0.20))
    assert relationship.significant


def test_no_significant_alpha_returns_no_credible_corrector() -> None:
    fit, _ = controlled_fit(
        alpha=np.asarray([-0.02, 0.0, 0.0, 0.0, 0.0]),
        alpha_standard_errors=np.full(5, 0.10),
    )

    result = classify_roles(fit)

    assert result.status is RoleStatus.NO_CREDIBLE_CORRECTOR
    assert result.dominant_corrector is None
    btc = result.profile.asset_diagnostics[0]
    assert btc.restoring
    assert btc.adjustment_classification is AdjustmentClassification.RESTORING_BUT_UNCERTAIN


def test_material_adjustment_by_all_assets_does_not_force_an_anchor() -> None:
    fit, _ = controlled_fit(
        alpha=np.asarray([-0.20, 0.10, -0.16, 0.10, 0.10]),
        gamma_edges={},
    )

    result = classify_roles(fit)

    assert result.dominant_corrector == "BTC"
    assert result.profile.weak_adjustment_candidates == ()
    assert result.leader_anchor is None
    assert result.status is RoleStatus.NO_CREDIBLE_ANCHOR


def test_secondary_corrector_rule_produces_only_a_minimal_three_asset_membership() -> None:
    fit, _ = controlled_fit(alpha=np.asarray([-0.20, 0.0, -0.24, 0.0, 0.0]))

    result = classify_roles(fit)

    assert result.dominant_corrector == "BTC"
    assert result.secondary_corrector == "SOL"
    assert result.leader_anchor == "ETH"
    assert result.expression_type is ExpressionType.MINIMAL_CORRECTING_BASKET
    assert tuple(leg.asset for leg in result.intended_legs) == ("BTC", "SOL", "ETH")
    assert len(result.intended_legs) == 3


def test_secondary_below_configured_ratio_is_not_admitted() -> None:
    fit, _ = controlled_fit(alpha=np.asarray([-0.20, 0.0, -0.12, 0.0, 0.0]))

    profile = build_structural_role_profile(
        fit,
        RoleSelectionConfig(secondary_share_ratio=0.50),
    )

    assert profile.dominant_corrector == "BTC"
    assert profile.secondary_corrector is None


def test_role_membership_is_invariant_to_compatible_beta_alpha_scale_and_sign() -> None:
    fit, _ = controlled_fit(alpha=np.asarray([-0.20, 0.0, -0.24, 0.0, 0.0]))
    vecm = fit.rank_one
    assert vecm is not None
    base = build_structural_role_profile(fit)
    scale = -3.5
    equivalent_vecm = replace(
        vecm,
        native_beta=readonly_array(vecm.native_beta * scale, dimensions=1),
        native_alpha=readonly_array(vecm.native_alpha / scale, dimensions=1),
        native_cointegration_constant=vecm.native_cointegration_constant * scale,
        alpha_standard_errors=readonly_array(
            vecm.alpha_standard_errors / abs(scale),
            dimensions=1,
        ),
    )

    equivalent = build_structural_role_profile(replace(fit, rank_one=equivalent_vecm))

    assert np.allclose(base.kappa, equivalent.kappa)
    assert base.correction_shares == equivalent.correction_shares
    assert base.dominant_corrector == equivalent.dominant_corrector
    assert base.secondary_corrector == equivalent.secondary_corrector
    assert base.leader_anchor == equivalent.leader_anchor


def test_future_data_changes_cannot_contaminate_fit_level_roles_at_k() -> None:
    levels_a, _, _ = simulate_rank_one_vecm(1100, seed=9182)
    levels_b = levels_a.copy()
    levels_b[900:] += np.asarray([2.0, -3.0, 4.0, -5.0, 6.0])
    window_a = research_window_from_levels(levels_a, end_index=899)
    window_b = research_window_from_levels(levels_b, end_index=899)
    fit_a = StructuralEstimator(STATE_CONFIG).fit(
        window_a,
        fit_timestamp=window_a.data_end,
        source_fingerprint="8" * 64,
    )
    fit_b = StructuralEstimator(STATE_CONFIG).fit(
        window_b,
        fit_timestamp=window_b.data_end,
        source_fingerprint="8" * 64,
    )

    profile_a = build_structural_role_profile(fit_a)
    profile_b = build_structural_role_profile(fit_b)

    assert not np.array_equal(levels_a[900:], levels_b[900:])
    assert profile_a.rank == profile_b.rank
    assert profile_a.dominant_corrector == profile_b.dominant_corrector
    assert profile_a.secondary_corrector == profile_b.secondary_corrector
    assert profile_a.leader_anchor == profile_b.leader_anchor
    assert profile_a.correction_shares == profile_b.correction_shares
    assert profile_a.gamma_diagnostics == profile_b.gamma_diagnostics


def test_fitted_gamma_joint_covariance_matches_reported_marginal_standard_errors() -> None:
    fit, _ = controlled_fit()
    vecm = fit.rank_one
    assert vecm is not None
    flattened_standard_errors = np.sqrt(np.diag(vecm.gamma_parameter_covariance))
    reconstructed = np.column_stack(vecm.gamma_standard_errors).reshape(-1, order="F")

    assert vecm.gamma_parameter_covariance.shape == (25, 25)
    assert np.allclose(flattened_standard_errors, reconstructed)
    assert np.allclose(vecm.native_beta, CONTROLLED_BETA)
