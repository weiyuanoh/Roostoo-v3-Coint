"""Immutable, auditable results for corrector/leader role identification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from numpy.typing import NDArray


class RoleStatus(str, Enum):
    INACTIVE_RANK_ZERO = "INACTIVE_RANK_ZERO"
    INACTIVE_MULTIPLE_RANK = "INACTIVE_MULTIPLE_RANK"
    NO_CREDIBLE_CORRECTOR = "NO_CREDIBLE_CORRECTOR"
    NO_CREDIBLE_ANCHOR = "NO_CREDIBLE_ANCHOR"
    NO_EXPECTED_CONVERGENCE = "NO_EXPECTED_CONVERGENCE"
    CORRECTOR_DIRECTION_CONFLICT = "CORRECTOR_DIRECTION_CONFLICT"
    STRUCTURAL_DIAGNOSTIC_ONLY = "STRUCTURAL_DIAGNOSTIC_ONLY"
    PAIR_CANDIDATE = "PAIR_CANDIDATE"
    MINIMAL_BASKET_CANDIDATE = "MINIMAL_BASKET_CANDIDATE"


class AdjustmentClassification(str, Enum):
    CREDIBLE_RESTORING = "CREDIBLE_RESTORING"
    RESTORING_BUT_UNCERTAIN = "RESTORING_BUT_UNCERTAIN"
    NON_RESTORING = "NON_RESTORING"
    NEUTRAL = "NEUTRAL"


class GammaTestStatus(str, Enum):
    TESTED = "TESTED"
    NO_LAGGED_DIFFERENCES = "NO_LAGGED_DIFFERENCES"
    INVALID_COVARIANCE = "INVALID_COVARIANCE"


class ExpressionType(str, Enum):
    NONE = "NONE"
    PAIR = "PAIR"
    MINIMAL_CORRECTING_BASKET = "MINIMAL_CORRECTING_BASKET"


class LegDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass(frozen=True, slots=True)
class AssetRoleDiagnostic:
    asset: str
    beta: float
    alpha: float
    alpha_standard_error: float | None
    alpha_t_statistic: float | None
    alpha_p_value: float | None
    alpha_significant: bool
    weak_adjustment_evidence: bool
    kappa: float
    restoring: bool
    restoring_strength: float
    correction_share: float
    adjustment_classification: AdjustmentClassification

    @property
    def contribution_to_beta_alpha(self) -> float:
        return self.kappa


@dataclass(frozen=True, slots=True)
class GammaLeadLagDiagnostic:
    source_asset: str
    target_asset: str
    lag_set: tuple[int, ...]
    coefficients: tuple[float, ...]
    coefficient_standard_errors: tuple[float, ...]
    coefficient_signs: tuple[int, ...]
    coefficient_sum: float
    wald_statistic: float | None
    degrees_of_freedom: int
    p_value: float | None
    significant: bool
    status: GammaTestStatus
    reason: str | None


@dataclass(frozen=True, slots=True)
class AssetLeadLagSummary:
    asset: str
    outgoing: tuple[GammaLeadLagDiagnostic, ...]
    incoming: tuple[GammaLeadLagDiagnostic, ...]
    significant_outgoing: tuple[GammaLeadLagDiagnostic, ...]
    significant_incoming: tuple[GammaLeadLagDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class FullCointegratingBasket:
    assets: tuple[str, ...]
    native_beta: NDArray
    native_cointegration_constant: float
    reporting_beta: NDArray
    reporting_cointegration_constant: float
    directions: tuple[LegDirection, ...]
    description: str


@dataclass(frozen=True, slots=True)
class StructuralRoleProfile:
    timestamp: datetime
    structural_fit_id: str
    structural_fit_data_end: datetime
    rank: int
    assets: tuple[str, ...]
    beta: NDArray | None
    native_cointegration_constant: float | None
    reporting_beta: NDArray | None
    reporting_cointegration_constant: float | None
    alpha: NDArray | None
    kappa: NDArray | None
    beta_alpha_total: float | None
    asset_diagnostics: tuple[AssetRoleDiagnostic, ...]
    dominant_corrector: str | None
    secondary_corrector: str | None
    weak_adjustment_candidates: tuple[str, ...]
    gamma_diagnostics: tuple[GammaLeadLagDiagnostic, ...]
    lead_lag_summaries: tuple[AssetLeadLagSummary, ...]
    leader_anchor: str | None

    @property
    def correction_shares(self) -> Mapping[str, float]:
        return MappingProxyType(
            {item.asset: item.correction_share for item in self.asset_diagnostics}
        )


@dataclass(frozen=True, slots=True)
class AssetForecastDiagnostic:
    asset: str
    latent_state_change: float
    observed_market_relative_change: float


@dataclass(frozen=True, slots=True)
class IntendedLeg:
    asset: str
    role: str
    direction: LegDirection


@dataclass(frozen=True, slots=True)
class RoleClassificationResult:
    timestamp: datetime
    structural_fit_id: str
    structural_fit_data_end: datetime
    rank: int
    status: RoleStatus
    reason: str
    profile: StructuralRoleProfile
    observed_ecm: float | None
    filtered_ecm: float | None
    empirical_half_life_observations: float | None
    forecast_horizon_steps: int | None
    expected_ecm_at_horizon: float | None
    observed_error_correction_contributions: NDArray | None
    filtered_error_correction_contributions: NDArray | None
    selected_asset_forecasts: tuple[AssetForecastDiagnostic, ...]
    relative_corrector_anchor_forecast: float | None
    expression_type: ExpressionType
    intended_legs: tuple[IntendedLeg, ...]
    full_cointegrating_basket: FullCointegratingBasket | None

    @property
    def dominant_corrector(self) -> str | None:
        return self.profile.dominant_corrector

    @property
    def secondary_corrector(self) -> str | None:
        return self.profile.secondary_corrector

    @property
    def leader_anchor(self) -> str | None:
        return self.profile.leader_anchor

    @property
    def beta(self) -> NDArray | None:
        return self.profile.beta

    @property
    def alpha(self) -> NDArray | None:
        return self.profile.alpha

    @property
    def kappa(self) -> NDArray | None:
        return self.profile.kappa

    @property
    def correction_shares(self) -> Mapping[str, float]:
        return self.profile.correction_shares

    @property
    def alpha_significance(self) -> Mapping[str, tuple[float | None, bool]]:
        return MappingProxyType(
            {
                item.asset: (item.alpha_p_value, item.alpha_significant)
                for item in self.profile.asset_diagnostics
            }
        )

    @property
    def gamma_diagnostics(self) -> tuple[GammaLeadLagDiagnostic, ...]:
        return self.profile.gamma_diagnostics
