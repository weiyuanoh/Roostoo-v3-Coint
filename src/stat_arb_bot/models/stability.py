"""Scale-aware comparisons across structural refits and window lengths."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from stat_arb_bot.domain.execution import utc_datetime
from stat_arb_bot.models.config import StructuralModelConfig
from stat_arb_bot.models.structural import StructuralEstimator, StructuralFitResult
from stat_arb_bot.research_data.pipeline import ResearchDataset


@dataclass(frozen=True, slots=True)
class StructuralStabilityComparison:
    previous_fit_id: str
    current_fit_id: str
    previous_rank: int
    current_rank: int
    rank_stable: bool
    previous_var_order: int
    current_var_order: int
    lag_order_stable: bool
    beta_cosine_similarity: float | None
    alpha_cosine_similarity: float | None
    alpha_l2_change: float | None
    empirical_half_life_change: float | None
    deterministic_half_life_change: float | None


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        return None
    return float(np.clip((left @ right) / denominator, -1.0, 1.0))


def _difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else right - left


def compare_structural_fits(
    previous: StructuralFitResult,
    current: StructuralFitResult,
) -> StructuralStabilityComparison:
    previous_rank = previous.johansen.inferred_rank
    current_rank = current.johansen.inferred_rank
    beta_similarity = None
    alpha_similarity = None
    alpha_change = None
    empirical_change = None
    deterministic_change = None
    if previous.rank_one is not None and current.rank_one is not None:
        # Reporting normalization is sign-aligned and alpha is inversely
        # rescaled, so both comparisons are invariant to native beta scaling.
        beta_similarity = _cosine(
            previous.rank_one.reporting_beta,
            current.rank_one.reporting_beta,
        )
        alpha_similarity = _cosine(
            previous.rank_one.reporting_alpha,
            current.rank_one.reporting_alpha,
        )
        alpha_change = float(
            np.linalg.norm(current.rank_one.reporting_alpha - previous.rank_one.reporting_alpha)
        )
        empirical_change = _difference(
            previous.rank_one.ecm.persistence.half_life_observations,
            current.rank_one.ecm.persistence.half_life_observations,
        )
        deterministic_change = _difference(
            previous.rank_one.convergence.simulated_half_life_observations,
            current.rank_one.convergence.simulated_half_life_observations,
        )
    return StructuralStabilityComparison(
        previous_fit_id=previous.metadata.fit_id,
        current_fit_id=current.metadata.fit_id,
        previous_rank=previous_rank,
        current_rank=current_rank,
        rank_stable=previous_rank == current_rank,
        previous_var_order=previous.lag_selection.selected_var_order,
        current_var_order=current.lag_selection.selected_var_order,
        lag_order_stable=(
            previous.lag_selection.selected_var_order == current.lag_selection.selected_var_order
        ),
        beta_cosine_similarity=beta_similarity,
        alpha_cosine_similarity=alpha_similarity,
        alpha_l2_change=alpha_change,
        empirical_half_life_change=empirical_change,
        deterministic_half_life_change=deterministic_change,
    )


@dataclass(frozen=True, slots=True)
class WindowFitSummary:
    window: timedelta
    actual_observations: int
    fit: StructuralFitResult | None
    failure: str | None

    @property
    def rank(self) -> int | None:
        return self.fit.johansen.inferred_rank if self.fit is not None else None


@dataclass(frozen=True, slots=True)
class CrossWindowComparison:
    fit_timestamp: datetime
    primary_window: timedelta
    fits: tuple[WindowFitSummary, ...]

    @property
    def primary(self) -> StructuralFitResult | None:
        return next(
            (item.fit for item in self.fits if item.window == self.primary_window),
            None,
        )


def compare_windows(
    dataset: ResearchDataset,
    *,
    fit_timestamp: datetime,
    config: StructuralModelConfig | None = None,
) -> CrossWindowComparison:
    specification = config or StructuralModelConfig()
    timestamp = utc_datetime(fit_timestamp, name="fit_timestamp")
    eligible = tuple(row for row in dataset.panel.rows if row.timestamp <= timestamp)
    if not eligible:
        raise ValueError("no completed model bar is available at fit_timestamp")
    data_end = eligible[-1].timestamp
    fingerprint = dataset.source_fingerprint_through(data_end)
    summaries: list[WindowFitSummary] = []
    for lookback in specification.comparison_windows:
        window = dataset.panel.window(
            end=data_end,
            lookback=lookback,
            minimum_observations=specification.minimum_observations,
        )
        try:
            fit = StructuralEstimator(specification).fit(
                window,
                fit_timestamp=timestamp,
                source_fingerprint=fingerprint,
            )
            summaries.append(WindowFitSummary(lookback, len(window.rows), fit, None))
        except (ValueError, np.linalg.LinAlgError) as exc:
            summaries.append(WindowFitSummary(lookback, len(window.rows), None, str(exc)))
    return CrossWindowComparison(timestamp, specification.primary_window, tuple(summaries))
