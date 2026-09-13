"""Bounded-window orchestration for immutable structural fits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from stat_arb_bot.domain.execution import utc_datetime
from stat_arb_bot.models.config import StructuralModelConfig
from stat_arb_bot.models.diagnostics import IntegrationDiagnostics, diagnose_integration
from stat_arb_bot.models.johansen import JohansenResult, estimate_johansen_rank
from stat_arb_bot.models.lag_selection import LagSelectionResult, select_var_lag
from stat_arb_bot.models.numeric import readonly_array
from stat_arb_bot.models.vecm import RankOneVECMResult, fit_rank_one_vecm
from stat_arb_bot.research_data.metadata import ModelFitMetadata
from stat_arb_bot.research_data.windows import ResearchWindow


MODEL_NAME = "five_asset_vecm"
MODEL_SCHEMA_VERSION = "1"


class RankScope(str, Enum):
    NO_COINTEGRATION = "NO_COINTEGRATION"
    V1_RANK_ONE = "V1_RANK_ONE"
    MULTIPLE_RELATIONS_OUTSIDE_V1 = "MULTIPLE_RELATIONS_OUTSIDE_V1"


@dataclass(frozen=True, slots=True)
class StructuralFitResult:
    metadata: ModelFitMetadata
    config: StructuralModelConfig
    integration: IntegrationDiagnostics
    lag_selection: LagSelectionResult
    johansen: JohansenResult
    rank_scope: RankScope
    rank_one: RankOneVECMResult | None

    def __post_init__(self) -> None:
        if (self.johansen.inferred_rank == 1) != (self.rank_one is not None):
            raise ValueError("a rank-one VECM result is required if and only if rank equals one")


def deterministic_fit_id(
    window: ResearchWindow,
    *,
    fit_timestamp: datetime,
    source_fingerprint: str,
    config: StructuralModelConfig,
) -> str:
    payload = {
        "model": MODEL_NAME,
        "schema": MODEL_SCHEMA_VERSION,
        "fit_timestamp": utc_datetime(fit_timestamp).isoformat(),
        "data_start": window.data_start.isoformat() if window.data_start else None,
        "data_end": window.data_end.isoformat() if window.data_end else None,
        "source": source_fingerprint,
        "config": dict(config.as_mapping()),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class StructuralEstimator:
    """Pure estimator: accepts bounded data and performs no I/O or trading."""

    def __init__(self, config: StructuralModelConfig | None = None) -> None:
        self.config = config or StructuralModelConfig()

    def fit(
        self,
        window: ResearchWindow,
        *,
        fit_timestamp: datetime,
        source_fingerprint: str,
        fit_id: str | None = None,
    ) -> StructuralFitResult:
        timestamp = utc_datetime(fit_timestamp, name="fit_timestamp")
        if not window.validation.valid:
            raise ValueError(f"cannot fit an invalid research window: {window.validation.issues}")
        if window.data_end is None or window.data_end > timestamp:
            raise ValueError("structural fit data must end no later than fit_timestamp")
        levels = readonly_array(window.log_prices().values, dimensions=2)
        if len(levels) < self.config.minimum_observations:
            raise ValueError("structural fit has fewer than configured minimum observations")
        integration = diagnose_integration(window, self.config)
        lag_selection = select_var_lag(levels, self.config)
        johansen = estimate_johansen_rank(
            levels,
            var_order=lag_selection.selected_var_order,
            config=self.config,
        )
        if johansen.inferred_rank == 0:
            scope = RankScope.NO_COINTEGRATION
            rank_one = None
        elif johansen.inferred_rank == 1:
            scope = RankScope.V1_RANK_ONE
            rank_one = fit_rank_one_vecm(
                levels,
                assets=window.universe_assets,
                var_order=lag_selection.selected_var_order,
                config=self.config,
            )
        else:
            scope = RankScope.MULTIPLE_RELATIONS_OUTSIDE_V1
            rank_one = None
        identifier = fit_id or deterministic_fit_id(
            window,
            fit_timestamp=timestamp,
            source_fingerprint=source_fingerprint,
            config=self.config,
        )
        metadata = ModelFitMetadata.from_window(
            window,
            model_name=MODEL_NAME,
            model_version=MODEL_SCHEMA_VERSION,
            fit_id=identifier,
            fit_timestamp=timestamp,
            source_data_fingerprint=source_fingerprint,
        )
        return StructuralFitResult(
            metadata=metadata,
            config=self.config,
            integration=integration,
            lag_selection=lag_selection,
            johansen=johansen,
            rank_scope=scope,
            rank_one=rank_one,
        )
