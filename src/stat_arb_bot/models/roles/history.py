"""Chronological descriptive study of immutable structural role profiles."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from stat_arb_bot.models.roles.classification import classify_roles
from stat_arb_bot.models.roles.config import RoleSelectionConfig
from stat_arb_bot.models.roles.results import RoleClassificationResult, RoleStatus
from stat_arb_bot.models.state_space.config import KalmanConfig
from stat_arb_bot.models.state_space.filter import ECMDrivenKalmanFilter
from stat_arb_bot.models.state_space.representation import build_vecm_state_space
from stat_arb_bot.models.structural import StructuralFitResult
from stat_arb_bot.research_data.panel import ResearchPanel


@dataclass(frozen=True, slots=True)
class RoleHistoryStudy:
    results: tuple[RoleClassificationResult, ...]
    fit_count: int
    rank_one_count: int
    dominant_corrector_counts: Mapping[str, int]
    anchor_counts: Mapping[str, int]
    status_counts: Mapping[str, int]
    secondary_corrector_count: int
    dominant_corrector_persistence: float | None
    anchor_persistence: float | None
    mean_correction_share: Mapping[str, float]
    correction_share_standard_deviation: Mapping[str, float]
    median_dominant_share: float | None
    asymmetric_fit_fraction: float | None
    asymmetric_correction_supported: bool
    leader_anchor_evidence_fraction: float | None
    pair_candidate_count: int
    minimal_basket_candidate_count: int
    no_valid_candidate_count: int

    def __post_init__(self) -> None:
        for name in (
            "dominant_corrector_counts",
            "anchor_counts",
            "status_counts",
            "mean_correction_share",
            "correction_share_standard_deviation",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


def _persistence(identities: Sequence[str | None]) -> float | None:
    comparable = [
        left == right
        for left, right in zip(identities, identities[1:])
        if left is not None and right is not None
    ]
    return float(np.mean(comparable)) if comparable else None


def run_role_history_study(
    panel: ResearchPanel,
    structural_fits: Sequence[StructuralFitResult],
    *,
    role_config: RoleSelectionConfig | None = None,
    kalman_config: KalmanConfig | None = None,
) -> RoleHistoryStudy:
    """Classify each fit at activation using only panel observations through that time."""

    specification = role_config or RoleSelectionConfig()
    fits = tuple(sorted(structural_fits, key=lambda item: item.metadata.fit_timestamp))
    if any(
        right.metadata.fit_timestamp <= left.metadata.fit_timestamp
        for left, right in zip(fits, fits[1:])
    ):
        raise ValueError("structural fits must have distinct chronological timestamps")
    matrix = panel.log_prices()
    timestamps = matrix.timestamps
    values = np.asarray(matrix.values, dtype=np.float64)
    index_by_time = {timestamp: index for index, timestamp in enumerate(timestamps)}
    results: list[RoleClassificationResult] = []
    for fit in fits:
        activation = fit.metadata.fit_timestamp
        if fit.metadata.data_end_timestamp > activation:
            raise ValueError("role history cannot use a future structural fit")
        if fit.johansen.inferred_rank != 1:
            results.append(classify_roles(fit, config=specification))
            continue
        if activation not in index_by_time:
            raise ValueError("rank-one fit timestamp is not a completed panel observation")
        availability = build_vecm_state_space(fit, kalman_config)
        if availability.model is None:
            raise ValueError("rank-one fit unexpectedly has no state-space representation")
        index = index_by_time[activation]
        lag_history = availability.model.layout.lagged_difference_count + 1
        start = index - lag_history + 1
        if start < 0:
            raise ValueError(
                "insufficient bounded panel history for rank-one filter initialization"
            )
        filter_ = ECMDrivenKalmanFilter.initialize(
            availability.model,
            initialization_timestamp=activation,
            history_timestamps=timestamps[start : index + 1],
            level_history=values[start : index + 1],
        )
        results.append(classify_roles(fit, filter_=filter_, config=specification))

    rank_one = [result for result in results if result.rank == 1]
    dominants = [result.dominant_corrector for result in rank_one]
    anchors = [result.leader_anchor for result in rank_one]
    dominant_counts = Counter(item for item in dominants if item is not None)
    anchor_counts = Counter(item for item in anchors if item is not None)
    status_counts = Counter(result.status.value for result in results)
    assets = panel.universe.assets
    share_rows = np.asarray(
        [
            [result.profile.correction_shares.get(asset, 0.0) for asset in assets]
            for result in rank_one
        ],
        dtype=np.float64,
    )
    if len(share_rows):
        means = {asset: float(value) for asset, value in zip(assets, np.mean(share_rows, axis=0))}
        standard_deviations = {
            asset: float(value) for asset, value in zip(assets, np.std(share_rows, axis=0))
        }
    else:
        means = {asset: 0.0 for asset in assets}
        standard_deviations = {asset: 0.0 for asset in assets}
    dominant_shares = [
        result.profile.correction_shares[result.dominant_corrector]
        for result in rank_one
        if result.dominant_corrector is not None
    ]
    median_dominant = float(np.median(dominant_shares)) if dominant_shares else None
    asymmetric_fraction = (
        float(
            np.mean(np.asarray(dominant_shares) >= specification.asymmetry_dominant_share_threshold)
        )
        if dominant_shares
        else None
    )
    pair_count = sum(result.status is RoleStatus.PAIR_CANDIDATE for result in rank_one)
    basket_count = sum(result.status is RoleStatus.MINIMAL_BASKET_CANDIDATE for result in rank_one)
    return RoleHistoryStudy(
        results=tuple(results),
        fit_count=len(results),
        rank_one_count=len(rank_one),
        dominant_corrector_counts=dominant_counts,
        anchor_counts=anchor_counts,
        status_counts=status_counts,
        secondary_corrector_count=sum(
            result.secondary_corrector is not None for result in rank_one
        ),
        dominant_corrector_persistence=_persistence(dominants),
        anchor_persistence=_persistence(anchors),
        mean_correction_share=means,
        correction_share_standard_deviation=standard_deviations,
        median_dominant_share=median_dominant,
        asymmetric_fit_fraction=asymmetric_fraction,
        asymmetric_correction_supported=bool(
            median_dominant is not None
            and median_dominant >= specification.asymmetry_dominant_share_threshold
        ),
        leader_anchor_evidence_fraction=(
            sum(item is not None for item in anchors) / len(rank_one) if rank_one else None
        ),
        pair_candidate_count=pair_count,
        minimal_basket_candidate_count=basket_count,
        no_valid_candidate_count=len(rank_one) - pair_count - basket_count,
    )
