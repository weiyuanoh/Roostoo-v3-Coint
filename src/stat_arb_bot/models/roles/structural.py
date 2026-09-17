"""Scale-invariant structural corrector and leader/anchor diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from stat_arb_bot.models.numeric import readonly_array
from stat_arb_bot.models.roles.config import RoleSelectionConfig
from stat_arb_bot.models.roles.gamma import gamma_lead_lag_diagnostics, summarize_lead_lag
from stat_arb_bot.models.roles.results import (
    AdjustmentClassification,
    AssetRoleDiagnostic,
    StructuralRoleProfile,
)
from stat_arb_bot.models.structural import StructuralFitResult


def _alpha_uncertainty(alpha: float, standard_error: float) -> tuple[float | None, float | None]:
    if not np.isfinite(standard_error) or standard_error <= 0:
        return None, None
    statistic = float(alpha / standard_error)
    p_value = float(2.0 * norm.sf(abs(statistic)))
    return statistic, p_value


def build_structural_role_profile(
    fit: StructuralFitResult,
    config: RoleSelectionConfig | None = None,
) -> StructuralRoleProfile:
    """Classify immutable fit-level roles without any current-price information."""

    specification = config or RoleSelectionConfig()
    rank = fit.johansen.inferred_rank
    metadata = fit.metadata
    if rank != 1:
        return StructuralRoleProfile(
            timestamp=metadata.fit_timestamp,
            structural_fit_id=metadata.fit_id,
            structural_fit_data_end=metadata.data_end_timestamp,
            rank=rank,
            assets=metadata.universe,
            beta=None,
            native_cointegration_constant=None,
            reporting_beta=None,
            reporting_cointegration_constant=None,
            alpha=None,
            kappa=None,
            beta_alpha_total=None,
            asset_diagnostics=(),
            dominant_corrector=None,
            secondary_corrector=None,
            weak_adjustment_candidates=(),
            gamma_diagnostics=(),
            lead_lag_summaries=(),
            leader_anchor=None,
        )

    vecm = fit.rank_one
    if vecm is None:
        raise ValueError("rank-one fit is missing VECM parameters")
    if metadata.data_end_timestamp > metadata.fit_timestamp:
        raise ValueError("role profile cannot use a structural fit from the future")
    beta = vecm.native_beta
    alpha = vecm.native_alpha
    kappa = np.asarray(beta * alpha, dtype=np.float64)
    strengths = np.maximum(-kappa, 0.0)
    total_strength = float(np.sum(strengths))
    shares = strengths / total_strength if total_strength > 0 else np.zeros_like(strengths)
    diagnostics: list[AssetRoleDiagnostic] = []
    credible_indices: list[int] = []
    weak_candidates: list[str] = []
    tolerance = specification.numerical_zero_tolerance
    for index, asset in enumerate(vecm.assets):
        standard_error = float(vecm.alpha_standard_errors[index])
        t_statistic, p_value = _alpha_uncertainty(float(alpha[index]), standard_error)
        significant = p_value is not None and p_value < specification.alpha_significance_level
        weak = p_value is not None and not significant
        restoring = bool(kappa[index] < -tolerance)
        if restoring and significant:
            classification = AdjustmentClassification.CREDIBLE_RESTORING
            credible_indices.append(index)
        elif restoring:
            classification = AdjustmentClassification.RESTORING_BUT_UNCERTAIN
        elif kappa[index] > tolerance:
            classification = AdjustmentClassification.NON_RESTORING
        else:
            classification = AdjustmentClassification.NEUTRAL
        if weak:
            weak_candidates.append(asset)
        diagnostics.append(
            AssetRoleDiagnostic(
                asset=asset,
                beta=float(beta[index]),
                alpha=float(alpha[index]),
                alpha_standard_error=(
                    standard_error if np.isfinite(standard_error) and standard_error > 0 else None
                ),
                alpha_t_statistic=t_statistic,
                alpha_p_value=p_value,
                alpha_significant=significant,
                weak_adjustment_evidence=weak,
                kappa=float(kappa[index]),
                restoring=restoring,
                restoring_strength=float(strengths[index]),
                correction_share=float(shares[index]),
                adjustment_classification=classification,
            )
        )

    dominant_index = (
        min(credible_indices, key=lambda index: (-float(shares[index]), index))
        if credible_indices
        else None
    )
    dominant = vecm.assets[dominant_index] if dominant_index is not None else None
    secondary_index: int | None = None
    if dominant_index is not None:
        threshold = specification.secondary_share_ratio * float(shares[dominant_index])
        eligible = [
            index
            for index in credible_indices
            if index != dominant_index
            and float(shares[index]) >= threshold
            # For every non-zero ECM this is equivalent to requiring the two
            # expected alpha*ECM price moves to have the same direction.
            and np.sign(alpha[index]) == np.sign(alpha[dominant_index])
        ]
        if eligible:
            secondary_index = min(eligible, key=lambda index: (-float(shares[index]), index))
    secondary = vecm.assets[secondary_index] if secondary_index is not None else None

    gamma = gamma_lead_lag_diagnostics(
        vecm,
        significance_level=specification.gamma_significance_level,
    )
    summaries = summarize_lead_lag(vecm.assets, gamma)
    excluded = {item for item in (dominant, secondary) if item is not None}
    anchor_candidates: list[tuple[float, float, int, str]] = []
    if dominant is not None:
        by_asset = {item.asset: item for item in diagnostics}
        for index, asset in enumerate(vecm.assets):
            asset_diagnostic = by_asset[asset]
            if asset in excluded:
                continue
            if not asset_diagnostic.weak_adjustment_evidence:
                continue
            if asset_diagnostic.correction_share > specification.maximum_anchor_correction_share:
                continue
            relationship = next(
                (
                    item
                    for item in gamma
                    if item.source_asset == asset and item.target_asset == dominant
                ),
                None,
            )
            if relationship is None or not relationship.significant or relationship.p_value is None:
                continue
            anchor_candidates.append(
                (
                    relationship.p_value,
                    asset_diagnostic.correction_share,
                    index,
                    asset,
                )
            )
    leader_anchor = min(anchor_candidates)[3] if anchor_candidates else None
    return StructuralRoleProfile(
        timestamp=metadata.fit_timestamp,
        structural_fit_id=metadata.fit_id,
        structural_fit_data_end=metadata.data_end_timestamp,
        rank=rank,
        assets=vecm.assets,
        beta=beta,
        native_cointegration_constant=vecm.native_cointegration_constant,
        reporting_beta=vecm.reporting_beta,
        reporting_cointegration_constant=vecm.reporting_cointegration_constant,
        alpha=alpha,
        kappa=readonly_array(kappa, dimensions=1),
        beta_alpha_total=float(beta @ alpha),
        asset_diagnostics=tuple(diagnostics),
        dominant_corrector=dominant,
        secondary_corrector=secondary,
        weak_adjustment_candidates=tuple(weak_candidates),
        gamma_diagnostics=gamma,
        lead_lag_summaries=summaries,
        leader_anchor=leader_anchor,
    )
