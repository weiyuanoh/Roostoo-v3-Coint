"""ECM-driven state estimation and non-trading forecast diagnostics."""

from stat_arb_bot.models.state_space.baselines import (
    BaselineForecast,
    PlainVECMForecaster,
    RandomWalkKalmanFilter,
    RandomWalkStepResult,
)
from stat_arb_bot.models.state_space.config import (
    KalmanConfig,
    MeasurementNoiseConfig,
    MeasurementNoiseMode,
)
from stat_arb_bot.models.state_space.diagnostics import (
    ChronologicalForecastRecord,
    ForecastMetrics,
    InnovationDiagnostics,
    evaluate_forecasts,
    summarize_innovations,
)
from stat_arb_bot.models.state_space.filter import (
    ECMDrivenKalmanFilter,
    ECMObservation,
    FilterCheckpoint,
    FilterStatus,
    FilterStepResult,
    PredictedState,
    StateForecast,
)
from stat_arb_bot.models.state_space.numeric import (
    StateSpaceNumericalError,
    validated_covariance,
)
from stat_arb_bot.models.state_space.persistence import FilterStateStore, RegimeStateStore
from stat_arb_bot.models.state_space.regime import (
    InactiveFilterState,
    RegimeCheckpoint,
    RegimeTransition,
    RegimeTransitionAction,
    StructuralRegimeManager,
)
from stat_arb_bot.models.state_space.representation import (
    ProcessNoiseSpecification,
    ResolvedMeasurementNoise,
    StateLayout,
    StateSpaceAvailability,
    StateSpaceStatus,
    VECMStateSpace,
    build_vecm_state_space,
)
from stat_arb_bot.models.state_space.study import (
    SequentialDiagnosticStudyResult,
    run_sequential_diagnostic_study,
)

__all__ = [
    "BaselineForecast",
    "ChronologicalForecastRecord",
    "ECMDrivenKalmanFilter",
    "ECMObservation",
    "FilterCheckpoint",
    "FilterStateStore",
    "FilterStatus",
    "FilterStepResult",
    "ForecastMetrics",
    "InactiveFilterState",
    "InnovationDiagnostics",
    "KalmanConfig",
    "MeasurementNoiseConfig",
    "MeasurementNoiseMode",
    "PlainVECMForecaster",
    "PredictedState",
    "ProcessNoiseSpecification",
    "RandomWalkKalmanFilter",
    "RandomWalkStepResult",
    "RegimeCheckpoint",
    "RegimeStateStore",
    "RegimeTransition",
    "RegimeTransitionAction",
    "ResolvedMeasurementNoise",
    "SequentialDiagnosticStudyResult",
    "StateForecast",
    "StateLayout",
    "StateSpaceAvailability",
    "StateSpaceNumericalError",
    "StateSpaceStatus",
    "StructuralRegimeManager",
    "VECMStateSpace",
    "build_vecm_state_space",
    "evaluate_forecasts",
    "run_sequential_diagnostic_study",
    "summarize_innovations",
    "validated_covariance",
]
