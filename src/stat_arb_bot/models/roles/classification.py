"""Current, membership-only corrector/anchor classification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from stat_arb_bot.models.numeric import readonly_array
from stat_arb_bot.models.roles.config import RoleSelectionConfig
from stat_arb_bot.models.roles.results import (
    AssetForecastDiagnostic,
    ExpressionType,
    FullCointegratingBasket,
    IntendedLeg,
    LegDirection,
    RoleClassificationResult,
    RoleStatus,
    StructuralRoleProfile,
)
from stat_arb_bot.models.roles.structural import build_structural_role_profile
from stat_arb_bot.models.state_space.filter import ECMDrivenKalmanFilter, StateForecast
from stat_arb_bot.models.structural import StructuralFitResult


def _direction(value: float, tolerance: float) -> LegDirection:
    if value > tolerance:
        return LegDirection.LONG
    if value < -tolerance:
        return LegDirection.SHORT
    return LegDirection.FLAT


def _opposite(direction: LegDirection) -> LegDirection:
    if direction is LegDirection.LONG:
        return LegDirection.SHORT
    if direction is LegDirection.SHORT:
        return LegDirection.LONG
    return LegDirection.FLAT


def _expression_type(profile: StructuralRoleProfile) -> ExpressionType:
    if profile.dominant_corrector is None or profile.leader_anchor is None:
        return ExpressionType.NONE
    if profile.secondary_corrector is not None:
        return ExpressionType.MINIMAL_CORRECTING_BASKET
    return ExpressionType.PAIR


def _basket(
    profile: StructuralRoleProfile,
    *,
    ecm: float | None,
    tolerance: float,
) -> FullCointegratingBasket | None:
    if profile.beta is None:
        return None
    beta = profile.beta
    if ecm is None or abs(ecm) <= tolerance:
        directions = tuple(LegDirection.FLAT for _ in profile.assets)
    else:
        # To express convergence, positive ECM implies selling the beta
        # portfolio; negative ECM implies buying it. These are benchmark
        # directions only and intentionally contain no quantities.
        multiplier = -float(np.sign(ecm))
        directions = tuple(_direction(multiplier * value, tolerance) for value in beta)
    if (
        profile.reporting_beta is None
        or profile.native_cointegration_constant is None
        or profile.reporting_cointegration_constant is None
    ):
        raise ValueError("rank-one benchmark is missing its normalized equilibrium")
    return FullCointegratingBasket(
        assets=profile.assets,
        native_beta=beta,
        native_cointegration_constant=profile.native_cointegration_constant,
        reporting_beta=profile.reporting_beta,
        reporting_cointegration_constant=profile.reporting_cointegration_constant,
        directions=directions,
        description=(
            "FULL_COINTEGRATING_BASKET is the beta-defined benchmark; it is not the "
            "default corrector/anchor expression or a sizing instruction"
        ),
    )


def _base_result(
    profile: StructuralRoleProfile,
    *,
    status: RoleStatus,
    reason: str,
) -> RoleClassificationResult:
    return RoleClassificationResult(
        timestamp=profile.timestamp,
        structural_fit_id=profile.structural_fit_id,
        structural_fit_data_end=profile.structural_fit_data_end,
        rank=profile.rank,
        status=status,
        reason=reason,
        profile=profile,
        observed_ecm=None,
        filtered_ecm=None,
        empirical_half_life_observations=None,
        forecast_horizon_steps=None,
        expected_ecm_at_horizon=None,
        observed_error_correction_contributions=None,
        filtered_error_correction_contributions=None,
        selected_asset_forecasts=(),
        relative_corrector_anchor_forecast=None,
        expression_type=ExpressionType.NONE,
        intended_legs=(),
        full_cointegrating_basket=None,
    )


@dataclass(frozen=True, slots=True)
class _CurrentDiagnostics:
    timestamp: datetime
    observed_ecm: float
    filtered_ecm: float | None
    empirical_half_life: float | None
    horizon: int | None
    forecast: StateForecast | None
    observed_contributions: np.ndarray
    filtered_contributions: np.ndarray | None


def _current_diagnostics(
    fit: StructuralFitResult,
    profile: StructuralRoleProfile,
    filter_: ECMDrivenKalmanFilter | None,
) -> _CurrentDiagnostics:
    vecm = fit.rank_one
    if vecm is None:
        raise ValueError("current diagnostics require a rank-one VECM")
    half_life = vecm.ecm.persistence.half_life_observations
    if filter_ is None:
        return _CurrentDiagnostics(
            timestamp=profile.timestamp,
            observed_ecm=vecm.ecm.current,
            filtered_ecm=None,
            empirical_half_life=half_life,
            horizon=None,
            forecast=None,
            observed_contributions=vecm.current_error_correction_contribution,
            filtered_contributions=None,
        )
    if filter_.model.structural_fit.metadata.fit_id != fit.metadata.fit_id:
        raise ValueError("Kalman state and structural role profile use different fit IDs")
    timestamp = filter_.timestamp
    if fit.metadata.data_end_timestamp > timestamp:
        raise ValueError("role classification cannot use a future structural fit")
    layout = filter_.model.layout
    filtered_prices = filter_.state_mean[layout.price_slice]
    observed_ecm = (
        float(filter_.last_observation @ vecm.native_beta) + vecm.native_cointegration_constant
    )
    filtered_ecm = float(filtered_prices @ vecm.native_beta) + vecm.native_cointegration_constant
    diagnostic_half_life = (
        half_life if half_life is not None else vecm.convergence.simulated_half_life_observations
    )
    horizon = (
        min(
            max(1, int(round(diagnostic_half_life))),
            filter_.model.config.maximum_forecast_steps,
        )
        if diagnostic_half_life is not None
        else None
    )
    return _CurrentDiagnostics(
        timestamp=timestamp,
        observed_ecm=observed_ecm,
        filtered_ecm=filtered_ecm,
        empirical_half_life=half_life,
        horizon=horizon,
        forecast=filter_.forecast(horizon) if horizon is not None else None,
        observed_contributions=readonly_array(vecm.native_alpha * observed_ecm, dimensions=1),
        filtered_contributions=readonly_array(vecm.native_alpha * filtered_ecm, dimensions=1),
    )


def _forecasts_for(
    profile: StructuralRoleProfile,
    current: _CurrentDiagnostics,
    assets: tuple[str, ...],
) -> tuple[AssetForecastDiagnostic, ...]:
    if current.forecast is None:
        return ()
    asset_index = {asset: index for index, asset in enumerate(profile.assets)}
    return tuple(
        AssetForecastDiagnostic(
            asset=asset,
            latent_state_change=float(current.forecast.latent_state_change[asset_index[asset]]),
            observed_market_relative_change=float(
                current.forecast.observed_market_relative_change[asset_index[asset]]
            ),
        )
        for asset in assets
    )


def _unavailable_expression_result(
    profile: StructuralRoleProfile,
    current: _CurrentDiagnostics,
    *,
    status: RoleStatus,
    reason: str,
    selected_assets: tuple[str, ...],
    tolerance: float,
) -> RoleClassificationResult:
    return RoleClassificationResult(
        timestamp=current.timestamp,
        structural_fit_id=profile.structural_fit_id,
        structural_fit_data_end=profile.structural_fit_data_end,
        rank=1,
        status=status,
        reason=reason,
        profile=profile,
        observed_ecm=current.observed_ecm,
        filtered_ecm=current.filtered_ecm,
        empirical_half_life_observations=current.empirical_half_life,
        forecast_horizon_steps=current.horizon,
        expected_ecm_at_horizon=(
            float(current.forecast.expected_ecm_path[-1]) if current.forecast is not None else None
        ),
        observed_error_correction_contributions=current.observed_contributions,
        filtered_error_correction_contributions=current.filtered_contributions,
        selected_asset_forecasts=_forecasts_for(profile, current, selected_assets),
        relative_corrector_anchor_forecast=None,
        expression_type=ExpressionType.NONE,
        intended_legs=(),
        full_cointegrating_basket=_basket(
            profile,
            ecm=(
                current.filtered_ecm if current.filtered_ecm is not None else current.observed_ecm
            ),
            tolerance=tolerance,
        ),
    )


def classify_roles(
    fit: StructuralFitResult,
    *,
    filter_: ECMDrivenKalmanFilter | None = None,
    config: RoleSelectionConfig | None = None,
) -> RoleClassificationResult:
    """Return structural roles and, when supplied, chronology-safe current diagnostics.

    This function never creates an order and never accepts future returns or PnL.
    """

    specification = config or RoleSelectionConfig()
    profile = build_structural_role_profile(fit, specification)
    if profile.rank == 0:
        return _base_result(
            profile,
            status=RoleStatus.INACTIVE_RANK_ZERO,
            reason="rank zero has no active V1 equilibrium role classification",
        )
    if profile.rank != 1:
        return _base_result(
            profile,
            status=RoleStatus.INACTIVE_MULTIPLE_RANK,
            reason="multiple cointegrating relations are outside V1 role scope",
        )
    vecm = fit.rank_one
    if vecm is None:
        raise ValueError("rank-one fit is missing VECM parameters")
    current = _current_diagnostics(fit, profile, filter_)
    if profile.dominant_corrector is None:
        return _unavailable_expression_result(
            profile,
            current,
            status=RoleStatus.NO_CREDIBLE_CORRECTOR,
            reason="no restoring asset has statistically credible alpha adjustment",
            selected_assets=(),
            tolerance=specification.numerical_zero_tolerance,
        )
    expression_type = _expression_type(profile)
    if profile.leader_anchor is None:
        correctors_without_anchor = tuple(
            item
            for item in (profile.dominant_corrector, profile.secondary_corrector)
            if item is not None
        )
        return _unavailable_expression_result(
            profile,
            current,
            status=RoleStatus.NO_CREDIBLE_ANCHOR,
            reason=(
                "no non-corrector combines weak long-run adjustment with significant "
                "Gamma leadership toward the dominant corrector"
            ),
            selected_assets=correctors_without_anchor,
            tolerance=specification.numerical_zero_tolerance,
        )

    correctors = tuple(
        item
        for item in (profile.dominant_corrector, profile.secondary_corrector)
        if item is not None
    )
    if filter_ is None:
        legs = tuple(
            [IntendedLeg(asset, "CORRECTOR", LegDirection.FLAT) for asset in correctors]
            + [IntendedLeg(profile.leader_anchor, "LEADER_ANCHOR", LegDirection.FLAT)]
        )
        return RoleClassificationResult(
            timestamp=current.timestamp,
            structural_fit_id=profile.structural_fit_id,
            structural_fit_data_end=profile.structural_fit_data_end,
            rank=1,
            status=RoleStatus.STRUCTURAL_DIAGNOSTIC_ONLY,
            reason="structural membership is available; no current Kalman state was supplied",
            profile=profile,
            observed_ecm=current.observed_ecm,
            filtered_ecm=None,
            empirical_half_life_observations=current.empirical_half_life,
            forecast_horizon_steps=None,
            expected_ecm_at_horizon=None,
            observed_error_correction_contributions=current.observed_contributions,
            filtered_error_correction_contributions=None,
            selected_asset_forecasts=(),
            relative_corrector_anchor_forecast=None,
            expression_type=expression_type,
            intended_legs=legs,
            full_cointegrating_basket=_basket(
                profile,
                ecm=current.observed_ecm,
                tolerance=specification.numerical_zero_tolerance,
            ),
        )

    assert current.filtered_ecm is not None
    assert current.filtered_contributions is not None
    filtered_ecm = current.filtered_ecm
    filtered_contributions = current.filtered_contributions
    forecast = current.forecast
    asset_index = {asset: index for index, asset in enumerate(profile.assets)}
    selected_assets = (*correctors, profile.leader_anchor)
    selected_forecasts = _forecasts_for(profile, current, selected_assets)
    relative = None
    expected_ecm = None
    if forecast is not None:
        corrector_change = float(
            forecast.latent_state_change[asset_index[profile.dominant_corrector]]
        )
        anchor_change = float(forecast.latent_state_change[asset_index[profile.leader_anchor]])
        relative = corrector_change - anchor_change
        expected_ecm = float(forecast.expected_ecm_path[-1])
    dominant_index = asset_index[profile.dominant_corrector]
    dominant_move = float(filtered_contributions[dominant_index])
    corrector_direction = _direction(dominant_move, specification.numerical_zero_tolerance)
    legs = tuple(
        [
            IntendedLeg(
                asset,
                "DOMINANT_CORRECTOR" if offset == 0 else "SECONDARY_CORRECTOR",
                _direction(
                    float(filtered_contributions[asset_index[asset]]),
                    specification.numerical_zero_tolerance,
                ),
            )
            for offset, asset in enumerate(correctors)
        ]
        + [IntendedLeg(profile.leader_anchor, "LEADER_ANCHOR", _opposite(corrector_direction))]
    )
    status = (
        RoleStatus.PAIR_CANDIDATE
        if expression_type is ExpressionType.PAIR
        else RoleStatus.MINIMAL_BASKET_CANDIDATE
    )
    reason = "structural roles and current convergence diagnostics agree"
    if expected_ecm is None or not abs(expected_ecm) < abs(filtered_ecm):
        status = RoleStatus.NO_EXPECTED_CONVERGENCE
        reason = "expected ECM magnitude does not shrink at the active half-life horizon"
    elif selected_forecasts:
        forecast_move = selected_forecasts[0].latent_state_change
        if (
            abs(dominant_move) > specification.numerical_zero_tolerance
            and abs(forecast_move) > specification.numerical_zero_tolerance
            and np.sign(dominant_move) != np.sign(forecast_move)
        ):
            status = RoleStatus.CORRECTOR_DIRECTION_CONFLICT
            reason = "filtered alpha*ECM direction conflicts with the h-step latent forecast"
    return RoleClassificationResult(
        timestamp=current.timestamp,
        structural_fit_id=profile.structural_fit_id,
        structural_fit_data_end=profile.structural_fit_data_end,
        rank=1,
        status=status,
        reason=reason,
        profile=profile,
        observed_ecm=current.observed_ecm,
        filtered_ecm=filtered_ecm,
        empirical_half_life_observations=current.empirical_half_life,
        forecast_horizon_steps=current.horizon,
        expected_ecm_at_horizon=expected_ecm,
        observed_error_correction_contributions=current.observed_contributions,
        filtered_error_correction_contributions=filtered_contributions,
        selected_asset_forecasts=selected_forecasts,
        relative_corrector_anchor_forecast=relative,
        expression_type=expression_type,
        intended_legs=legs,
        full_cointegrating_basket=_basket(
            profile,
            ecm=filtered_ecm,
            tolerance=specification.numerical_zero_tolerance,
        ),
    )
