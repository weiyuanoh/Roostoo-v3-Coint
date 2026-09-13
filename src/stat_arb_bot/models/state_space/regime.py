"""Explicit structural-fit and rank-regime handoffs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Sequence

from numpy.typing import ArrayLike

from stat_arb_bot.domain.execution import utc_datetime
from stat_arb_bot.models.state_space.config import KalmanConfig
from stat_arb_bot.models.state_space.filter import ECMDrivenKalmanFilter, FilterCheckpoint
from stat_arb_bot.models.state_space.representation import (
    StateSpaceAvailability,
    StateSpaceStatus,
    build_vecm_state_space,
)
from stat_arb_bot.models.structural import StructuralFitResult


class RegimeTransitionAction(str, Enum):
    ACTIVATE = "ACTIVATE"
    REFIT_CARRY_PRICE = "REFIT_CARRY_PRICE"
    DEACTIVATE_RANK_ZERO = "DEACTIVATE_RANK_ZERO"
    DEACTIVATE_MULTIPLE_RANK = "DEACTIVATE_MULTIPLE_RANK"
    REMAIN_INACTIVE = "REMAIN_INACTIVE"


@dataclass(frozen=True, slots=True)
class InactiveFilterState:
    timestamp: datetime
    structural_fit_id: str
    rank: int
    status: StateSpaceStatus
    reason: str
    final_active_checkpoint: FilterCheckpoint | None


@dataclass(frozen=True, slots=True)
class RegimeTransition:
    timestamp: datetime
    previous_fit_id: str | None
    new_fit_id: str
    previous_rank: int | None
    new_rank: int
    action: RegimeTransitionAction
    previous_layout_version: str | None
    new_layout_version: str | None
    previous_dimension: int | None
    new_dimension: int | None
    carried_filtered_price: bool
    covariance_mapping: str


@dataclass(frozen=True, slots=True)
class RegimeCheckpoint:
    timestamp: datetime
    structural_fit_id: str
    rank: int
    status: StateSpaceStatus
    reason: str | None
    active_filter_checkpoint: FilterCheckpoint | None
    final_active_checkpoint: FilterCheckpoint | None
    config: KalmanConfig


class StructuralRegimeManager:
    """Own one active filter and make every refit/rank transition auditable."""

    def __init__(self, config: KalmanConfig | None = None) -> None:
        self.config = config or KalmanConfig()
        self.active_filter: ECMDrivenKalmanFilter | None = None
        self.inactive_state: InactiveFilterState | None = None
        self.last_fit: StructuralFitResult | None = None
        self.final_active_checkpoint: FilterCheckpoint | None = None
        self.transitions: list[RegimeTransition] = []
        self.last_regime_timestamp: datetime | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: RegimeCheckpoint,
        structural_fit: StructuralFitResult,
        config: KalmanConfig | None = None,
    ) -> StructuralRegimeManager:
        if checkpoint.structural_fit_id != structural_fit.metadata.fit_id:
            raise ValueError("regime checkpoint belongs to a different structural fit")
        if checkpoint.rank != structural_fit.johansen.inferred_rank:
            raise ValueError("regime checkpoint rank disagrees with structural fit")
        if config is not None and config != checkpoint.config:
            raise ValueError("regime checkpoint filter configuration is incompatible")
        manager = cls(config or checkpoint.config)
        manager.last_fit = structural_fit
        manager.last_regime_timestamp = checkpoint.timestamp
        manager.final_active_checkpoint = checkpoint.final_active_checkpoint
        availability = build_vecm_state_space(structural_fit, manager.config)
        if checkpoint.status is StateSpaceStatus.ACTIVE:
            if availability.model is None or checkpoint.active_filter_checkpoint is None:
                raise ValueError("active regime checkpoint lacks a compatible rank-one filter")
            manager.active_filter = ECMDrivenKalmanFilter.from_checkpoint(
                availability.model,
                checkpoint.active_filter_checkpoint,
            )
        else:
            if availability.status is not checkpoint.status:
                raise ValueError("inactive checkpoint status disagrees with structural fit")
            manager.inactive_state = InactiveFilterState(
                checkpoint.timestamp,
                checkpoint.structural_fit_id,
                checkpoint.rank,
                checkpoint.status,
                checkpoint.reason or "inactive structural rank",
                checkpoint.final_active_checkpoint,
            )
        return manager

    def apply_fit(
        self,
        structural_fit: StructuralFitResult,
        *,
        timestamp: datetime,
        history_timestamps: Sequence[datetime],
        level_history: ArrayLike,
    ) -> RegimeTransition:
        current = utc_datetime(timestamp, name="regime transition timestamp")
        if self.last_regime_timestamp is not None and current <= self.last_regime_timestamp:
            raise ValueError("structural regime transitions must be strictly chronological")
        if structural_fit.metadata.data_end_timestamp > current:
            raise ValueError("cannot activate a structural fit before its data_end")
        availability = build_vecm_state_space(structural_fit, self.config)
        previous_filter = self.active_filter
        previous_fit = self.last_fit
        previous_checkpoint = previous_filter.checkpoint() if previous_filter else None
        if previous_checkpoint is not None:
            self.final_active_checkpoint = previous_checkpoint
        if availability.model is None:
            self.active_filter = None
            action = (
                RegimeTransitionAction.DEACTIVATE_RANK_ZERO
                if availability.status is StateSpaceStatus.INACTIVE_RANK_ZERO
                else RegimeTransitionAction.DEACTIVATE_MULTIPLE_RANK
            )
            if previous_filter is None:
                action = RegimeTransitionAction.REMAIN_INACTIVE
            self.inactive_state = InactiveFilterState(
                timestamp=current,
                structural_fit_id=structural_fit.metadata.fit_id,
                rank=availability.rank,
                status=availability.status,
                reason=availability.reason or "inactive structural rank",
                final_active_checkpoint=self.final_active_checkpoint,
            )
            new_layout = None
            carried = False
            covariance_mapping = "inactive; no state covariance"
        else:
            carried_price = None
            carried_covariance = None
            carried = previous_filter is not None
            if carried:
                carried_price = previous_filter.state_mean[previous_filter.model.layout.price_slice]
                carried_covariance = previous_filter.state_covariance[
                    previous_filter.model.layout.price_slice,
                    previous_filter.model.layout.price_slice,
                ]
            self.active_filter = ECMDrivenKalmanFilter.initialize(
                availability.model,
                initialization_timestamp=current,
                history_timestamps=history_timestamps,
                level_history=level_history,
                carried_price_mean=carried_price,
                carried_price_covariance=carried_covariance,
            )
            self.inactive_state = None
            new_layout = availability.model.layout
            action = (
                RegimeTransitionAction.REFIT_CARRY_PRICE
                if carried
                else RegimeTransitionAction.ACTIVATE
            )
            covariance_mapping = (
                "carried filtered 5x5 price block; rebuilt lag blocks from fitted residual covariance"
                if carried
                else "block-diagonal initialization from fitted residual covariance"
            )
        transition = RegimeTransition(
            timestamp=current,
            previous_fit_id=previous_fit.metadata.fit_id if previous_fit else None,
            new_fit_id=structural_fit.metadata.fit_id,
            previous_rank=previous_fit.johansen.inferred_rank if previous_fit else None,
            new_rank=structural_fit.johansen.inferred_rank,
            action=action,
            previous_layout_version=(
                previous_filter.model.layout.version if previous_filter else None
            ),
            new_layout_version=new_layout.version if new_layout else None,
            previous_dimension=(
                previous_filter.model.layout.dimension if previous_filter else None
            ),
            new_dimension=new_layout.dimension if new_layout else None,
            carried_filtered_price=carried,
            covariance_mapping=covariance_mapping,
        )
        self.last_fit = structural_fit
        self.last_regime_timestamp = current
        self.transitions.append(transition)
        return transition

    @property
    def availability(self) -> StateSpaceAvailability:
        if self.last_fit is None:
            raise ValueError("no structural fit has been applied")
        return build_vecm_state_space(self.last_fit, self.config)

    def checkpoint(self) -> RegimeCheckpoint:
        if self.last_fit is None:
            raise ValueError("no structural regime exists to checkpoint")
        if self.active_filter is not None:
            active = self.active_filter.checkpoint()
            return RegimeCheckpoint(
                timestamp=active.timestamp,
                structural_fit_id=self.last_fit.metadata.fit_id,
                rank=1,
                status=StateSpaceStatus.ACTIVE,
                reason=None,
                active_filter_checkpoint=active,
                final_active_checkpoint=self.final_active_checkpoint,
                config=self.config,
            )
        if self.inactive_state is None:
            raise ValueError("regime manager has neither active nor inactive state")
        return RegimeCheckpoint(
            timestamp=self.inactive_state.timestamp,
            structural_fit_id=self.inactive_state.structural_fit_id,
            rank=self.inactive_state.rank,
            status=self.inactive_state.status,
            reason=self.inactive_state.reason,
            active_filter_checkpoint=None,
            final_active_checkpoint=self.inactive_state.final_active_checkpoint,
            config=self.config,
        )
