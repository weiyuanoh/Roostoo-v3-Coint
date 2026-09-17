"""Exchange-neutral structural statistics over bounded research data."""

from stat_arb_bot.models.config import LagCriterion, StructuralModelConfig, vecm_lagged_differences
from stat_arb_bot.models.diagnostics import (
    IntegrationDiagnostics,
    PersistenceDiagnostic,
    SeriesIntegrationDiagnostic,
    UnitRootDiagnostic,
    adf_diagnostic,
    diagnose_integration,
    kpss_diagnostic,
)
from stat_arb_bot.models.johansen import JohansenResult, estimate_johansen_rank
from stat_arb_bot.models.lag_selection import LagSelectionResult, select_var_lag
from stat_arb_bot.models.persistence import StructuralModelStore
from stat_arb_bot.models.roles import (
    ExpressionType,
    RoleClassificationResult,
    RoleHistoryStudy,
    RoleSelectionConfig,
    RoleStatus,
    build_structural_role_profile,
    classify_roles,
    run_role_history_study,
)
from stat_arb_bot.models.scheduler import RollingFitResult, StructuralFitScheduler
from stat_arb_bot.models.stability import (
    CrossWindowComparison,
    StructuralStabilityComparison,
    compare_structural_fits,
    compare_windows,
)
from stat_arb_bot.models.state_space import (
    ECMDrivenKalmanFilter,
    KalmanConfig,
    MeasurementNoiseConfig,
    MeasurementNoiseMode,
    PlainVECMForecaster,
    RandomWalkKalmanFilter,
    StructuralRegimeManager,
    build_vecm_state_space,
)
from stat_arb_bot.models.structural import RankScope, StructuralEstimator, StructuralFitResult
from stat_arb_bot.models.vecm import (
    ContributionDecomposition,
    ConvergenceDiagnostic,
    ECMStatistics,
    RankOneVECMResult,
    fit_rank_one_vecm,
)

__all__ = [
    "ContributionDecomposition",
    "ConvergenceDiagnostic",
    "CrossWindowComparison",
    "ECMStatistics",
    "ECMDrivenKalmanFilter",
    "ExpressionType",
    "IntegrationDiagnostics",
    "JohansenResult",
    "LagCriterion",
    "LagSelectionResult",
    "KalmanConfig",
    "MeasurementNoiseConfig",
    "MeasurementNoiseMode",
    "PersistenceDiagnostic",
    "PlainVECMForecaster",
    "RandomWalkKalmanFilter",
    "RankOneVECMResult",
    "RankScope",
    "RoleClassificationResult",
    "RoleHistoryStudy",
    "RoleSelectionConfig",
    "RoleStatus",
    "RollingFitResult",
    "SeriesIntegrationDiagnostic",
    "StructuralEstimator",
    "StructuralFitResult",
    "StructuralFitScheduler",
    "StructuralModelConfig",
    "StructuralModelStore",
    "StructuralRegimeManager",
    "StructuralStabilityComparison",
    "UnitRootDiagnostic",
    "adf_diagnostic",
    "build_vecm_state_space",
    "build_structural_role_profile",
    "classify_roles",
    "compare_structural_fits",
    "compare_windows",
    "diagnose_integration",
    "estimate_johansen_rank",
    "fit_rank_one_vecm",
    "kpss_diagnostic",
    "run_role_history_study",
    "select_var_lag",
    "vecm_lagged_differences",
]
