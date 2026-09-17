"""Corrector and leader/anchor diagnostics; no trading or sizing logic."""

from stat_arb_bot.models.roles.classification import classify_roles
from stat_arb_bot.models.roles.config import RoleSelectionConfig
from stat_arb_bot.models.roles.gamma import gamma_lead_lag_diagnostics, summarize_lead_lag
from stat_arb_bot.models.roles.history import RoleHistoryStudy, run_role_history_study
from stat_arb_bot.models.roles.results import (
    AdjustmentClassification,
    AssetForecastDiagnostic,
    AssetLeadLagSummary,
    AssetRoleDiagnostic,
    ExpressionType,
    FullCointegratingBasket,
    GammaLeadLagDiagnostic,
    GammaTestStatus,
    IntendedLeg,
    LegDirection,
    RoleClassificationResult,
    RoleStatus,
    StructuralRoleProfile,
)
from stat_arb_bot.models.roles.structural import build_structural_role_profile

__all__ = [
    "AdjustmentClassification",
    "AssetForecastDiagnostic",
    "AssetLeadLagSummary",
    "AssetRoleDiagnostic",
    "ExpressionType",
    "FullCointegratingBasket",
    "GammaLeadLagDiagnostic",
    "GammaTestStatus",
    "IntendedLeg",
    "LegDirection",
    "RoleClassificationResult",
    "RoleHistoryStudy",
    "RoleSelectionConfig",
    "RoleStatus",
    "StructuralRoleProfile",
    "build_structural_role_profile",
    "classify_roles",
    "gamma_lead_lag_diagnostics",
    "run_role_history_study",
    "summarize_lead_lag",
]
