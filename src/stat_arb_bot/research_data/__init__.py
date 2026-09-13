"""Synchronized, model-free five-asset research-data infrastructure."""

from stat_arb_bot.research_data.aggregation import (
    MODEL_INTERVAL,
    RAW_INTERVAL,
    AggregationDiagnostics,
    AggregationResult,
    aggregate_five_to_fifteen,
    model_bucket_open,
)
from stat_arb_bot.research_data.coverage import (
    AssetCoverage,
    CoverageReport,
    summarize_coverage,
)
from stat_arb_bot.research_data.history import ResearchHistoryLoader
from stat_arb_bot.research_data.metadata import ModelFitMetadata, fingerprint_raw_candles
from stat_arb_bot.research_data.panel import (
    DroppedPanelTimestamp,
    PanelDiagnostics,
    ResearchMatrix,
    ResearchPanel,
    ResearchPanelRow,
    build_synchronized_panel,
)
from stat_arb_bot.research_data.pipeline import ResearchDataBuilder, ResearchDataset
from stat_arb_bot.research_data.quality import (
    ContinuityGap,
    RawDataConflictError,
    RawQualityDiagnostics,
    RawSeries,
    merge_raw_series,
    prepare_raw_series,
)
from stat_arb_bot.research_data.storage import DatasetPaths, ResearchDataStore
from stat_arb_bot.research_data.universe import ResearchUniverse
from stat_arb_bot.research_data.windows import (
    ResearchWindow,
    WindowIssue,
    WindowValidationResult,
    build_research_window,
)

__all__ = [
    "AggregationDiagnostics",
    "AggregationResult",
    "AssetCoverage",
    "ContinuityGap",
    "CoverageReport",
    "DatasetPaths",
    "DroppedPanelTimestamp",
    "MODEL_INTERVAL",
    "ModelFitMetadata",
    "PanelDiagnostics",
    "RAW_INTERVAL",
    "RawDataConflictError",
    "RawQualityDiagnostics",
    "RawSeries",
    "ResearchDataBuilder",
    "ResearchDataStore",
    "ResearchDataset",
    "ResearchHistoryLoader",
    "ResearchMatrix",
    "ResearchPanel",
    "ResearchPanelRow",
    "ResearchUniverse",
    "ResearchWindow",
    "WindowIssue",
    "WindowValidationResult",
    "aggregate_five_to_fifteen",
    "build_research_window",
    "build_synchronized_panel",
    "fingerprint_raw_candles",
    "merge_raw_series",
    "model_bucket_open",
    "prepare_raw_series",
    "summarize_coverage",
]
