"""Exact first-order state-space representation of an immutable rank-one VECM."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stat_arb_bot.models.numeric import readonly_array
from stat_arb_bot.models.state_space.config import (
    KalmanConfig,
    MeasurementNoiseConfig,
    MeasurementNoiseMode,
)
from stat_arb_bot.models.state_space.numeric import (
    StateSpaceNumericalError,
    ensure_finite_state,
    validated_covariance,
)
from stat_arb_bot.models.structural import StructuralFitResult
from stat_arb_bot.models.vecm import ContributionDecomposition, RankOneVECMResult


STATE_LAYOUT_VERSION = "vecm-augmented-v1"


class StateSpaceStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE_RANK_ZERO = "INACTIVE_RANK_ZERO"
    INACTIVE_MULTIPLE_RANK = "INACTIVE_MULTIPLE_RANK"


@dataclass(frozen=True, slots=True)
class StateLayout:
    assets: tuple[str, ...]
    var_order: int
    lagged_difference_count: int
    dimension: int
    version: str = STATE_LAYOUT_VERSION

    @classmethod
    def for_vecm(cls, vecm: RankOneVECMResult) -> StateLayout:
        asset_count = len(vecm.assets)
        return cls(
            assets=vecm.assets,
            var_order=vecm.var_order,
            lagged_difference_count=vecm.lagged_difference_count,
            dimension=asset_count * vecm.var_order,
        )

    @property
    def asset_count(self) -> int:
        return len(self.assets)

    @property
    def price_slice(self) -> slice:
        return slice(0, self.asset_count)

    def difference_slice(self, lag: int) -> slice:
        """Return delta x_(t-lag+1), where lag one is current delta x_t."""

        if lag < 1 or lag > self.lagged_difference_count:
            raise ValueError("difference lag is outside this state layout")
        start = lag * self.asset_count
        return slice(start, start + self.asset_count)

    @property
    def labels(self) -> tuple[str, ...]:
        labels = [f"x[{asset}]" for asset in self.assets]
        for lag in range(1, self.lagged_difference_count + 1):
            time_label = "t" if lag == 1 else f"t-{lag - 1}"
            labels.extend(f"delta_x_{time_label}[{asset}]" for asset in self.assets)
        return tuple(labels)


@dataclass(frozen=True, slots=True)
class ResolvedMeasurementNoise:
    config: MeasurementNoiseConfig
    covariance: NDArray
    interpretation: str


@dataclass(frozen=True, slots=True)
class ProcessNoiseSpecification:
    source: str
    structural_fit_id: str
    residual_covariance: NDArray
    innovation_loading: NDArray
    state_covariance: NDArray


@dataclass(frozen=True, slots=True)
class VECMStateSpace:
    structural_fit: StructuralFitResult
    vecm: RankOneVECMResult
    layout: StateLayout
    transition_matrix: NDArray
    transition_intercept: NDArray
    innovation_loading: NDArray
    process_covariance: NDArray
    measurement_matrix: NDArray
    measurement_noise: ResolvedMeasurementNoise
    process_noise: ProcessNoiseSpecification
    config: KalmanConfig

    def state_from_history(
        self,
        level_history: ArrayLike,
        *,
        current_price_override: ArrayLike | None = None,
    ) -> NDArray:
        levels = readonly_array(level_history, dimensions=2)
        n = self.layout.asset_count
        required = self.layout.lagged_difference_count + 1
        if levels.shape[1] != n:
            raise ValueError("initialization history width differs from the state universe")
        if len(levels) < required:
            raise ValueError(
                f"insufficient lag history: need {required} levels, received {len(levels)}"
            )
        state = np.zeros(self.layout.dimension, dtype=np.float64)
        current = (
            levels[-1]
            if current_price_override is None
            else readonly_array(
                current_price_override,
                dimensions=1,
            )
        )
        if current.shape != (n,):
            raise ValueError("current price override has an incompatible dimension")
        state[self.layout.price_slice] = current
        changes = np.diff(levels, axis=0)
        for lag in range(1, self.layout.lagged_difference_count + 1):
            state[self.layout.difference_slice(lag)] = changes[-lag]
        return ensure_finite_state(
            state,
            dimension=self.layout.dimension,
            maximum_norm=self.config.maximum_state_norm,
            name="initialized state",
        )

    def decompose_state(self, state: ArrayLike) -> ContributionDecomposition:
        current = ensure_finite_state(
            state,
            dimension=self.layout.dimension,
            maximum_norm=self.config.maximum_state_norm,
            name="state",
        )
        prices = current[self.layout.price_slice]
        ecm = float(prices @ self.vecm.native_beta) + self.vecm.native_cointegration_constant
        correction = readonly_array(self.vecm.native_alpha * ecm, dimensions=1)
        short_run = np.zeros(self.layout.asset_count, dtype=np.float64)
        for lag, gamma in enumerate(self.vecm.gamma_matrices, start=1):
            short_run += gamma @ current[self.layout.difference_slice(lag)]
        short_run = readonly_array(short_run, dimensions=1)
        deterministic = readonly_array(self.vecm.deterministic_outside, dimensions=1)
        total = readonly_array(correction + short_run + deterministic, dimensions=1)
        return ContributionDecomposition(correction, short_run, deterministic, total)

    def deterministic_transition(self, state: ArrayLike) -> NDArray:
        current = ensure_finite_state(
            state,
            dimension=self.layout.dimension,
            maximum_norm=self.config.maximum_state_norm,
            name="prior state",
        )
        result = self.transition_matrix @ current + self.transition_intercept
        return ensure_finite_state(
            result,
            dimension=self.layout.dimension,
            maximum_norm=self.config.maximum_state_norm,
            name="predicted state",
        )

    def transition(self, state: ArrayLike, innovation: ArrayLike) -> NDArray:
        shock = readonly_array(innovation, dimensions=1)
        if shock.shape != (self.layout.asset_count,):
            raise ValueError("structural innovation has an incompatible dimension")
        result = self.deterministic_transition(state) + self.innovation_loading @ shock
        return ensure_finite_state(
            result,
            dimension=self.layout.dimension,
            maximum_norm=self.config.maximum_state_norm,
            name="innovated state",
        )


@dataclass(frozen=True, slots=True)
class StateSpaceAvailability:
    status: StateSpaceStatus
    structural_fit_id: str
    rank: int
    model: VECMStateSpace | None
    reason: str | None


def _measurement_noise(
    config: MeasurementNoiseConfig,
    residual_covariance: NDArray,
    *,
    kalman_config: KalmanConfig,
) -> ResolvedMeasurementNoise:
    n = len(residual_covariance)
    if config.mode is MeasurementNoiseMode.RESIDUAL_DIAGONAL_FRACTION:
        covariance = np.diag(np.diag(residual_covariance) * config.residual_variance_fraction)
        interpretation = (
            f"diagonal observation variance at {config.residual_variance_fraction:g} "
            "times fitted VECM residual variance"
        )
    elif config.mode is MeasurementNoiseMode.EXPLICIT_DIAGONAL:
        if config.diagonal_variances is None or len(config.diagonal_variances) != n:
            raise StateSpaceNumericalError(f"explicit measurement diagonal must contain {n} values")
        covariance = np.diag(np.asarray(config.diagonal_variances, dtype=np.float64))
        interpretation = "explicit diagonal observation covariance"
    else:
        if config.covariance is None:
            raise StateSpaceNumericalError("explicit full measurement covariance is missing")
        covariance = np.asarray(config.covariance, dtype=np.float64)
        interpretation = "explicit full observation covariance"
    resolved = validated_covariance(
        covariance,
        dimension=n,
        name="measurement covariance R",
        psd_tolerance=kalman_config.psd_tolerance,
        symmetry_tolerance=kalman_config.symmetry_tolerance,
    )
    return ResolvedMeasurementNoise(config, resolved, interpretation)


def build_vecm_state_space(
    structural_fit: StructuralFitResult,
    config: KalmanConfig | None = None,
) -> StateSpaceAvailability:
    specification = config or KalmanConfig()
    rank = structural_fit.johansen.inferred_rank
    if rank == 0:
        return StateSpaceAvailability(
            StateSpaceStatus.INACTIVE_RANK_ZERO,
            structural_fit.metadata.fit_id,
            rank,
            None,
            "V1 ECM state-space model requires structural rank one",
        )
    if rank != 1:
        return StateSpaceAvailability(
            StateSpaceStatus.INACTIVE_MULTIPLE_RANK,
            structural_fit.metadata.fit_id,
            rank,
            None,
            "multiple cointegrating relations are outside V1 scope",
        )
    vecm = structural_fit.rank_one
    if vecm is None:
        raise ValueError("rank-one structural fit is missing its VECM parameters")
    layout = StateLayout.for_vecm(vecm)
    n = layout.asset_count
    dimension = layout.dimension
    pi = np.outer(vecm.native_alpha, vecm.native_beta)
    affine = vecm.native_alpha * vecm.native_cointegration_constant + vecm.deterministic_outside
    transition = np.zeros((dimension, dimension), dtype=np.float64)
    intercept = np.zeros(dimension, dtype=np.float64)
    transition[layout.price_slice, layout.price_slice] = np.eye(n) + pi
    intercept[layout.price_slice] = affine
    for lag, gamma in enumerate(vecm.gamma_matrices, start=1):
        transition[layout.price_slice, layout.difference_slice(lag)] = gamma
    if layout.lagged_difference_count:
        newest_difference = layout.difference_slice(1)
        transition[newest_difference, layout.price_slice] = pi
        intercept[newest_difference] = affine
        for lag, gamma in enumerate(vecm.gamma_matrices, start=1):
            transition[newest_difference, layout.difference_slice(lag)] = gamma
        for lag in range(2, layout.lagged_difference_count + 1):
            transition[layout.difference_slice(lag), layout.difference_slice(lag - 1)] = np.eye(n)
    loading = np.zeros((dimension, n), dtype=np.float64)
    loading[layout.price_slice] = np.eye(n)
    if layout.lagged_difference_count:
        loading[layout.difference_slice(1)] = np.eye(n)
    residual_covariance = validated_covariance(
        vecm.residual_covariance,
        dimension=n,
        name="VECM residual covariance",
        psd_tolerance=specification.psd_tolerance,
        symmetry_tolerance=specification.symmetry_tolerance,
        require_positive_diagonal=True,
    )
    process_covariance = validated_covariance(
        loading @ residual_covariance @ loading.T,
        dimension=dimension,
        name="augmented process covariance Q",
        psd_tolerance=specification.psd_tolerance,
        symmetry_tolerance=specification.symmetry_tolerance,
    )
    measurement = np.zeros((n, dimension), dtype=np.float64)
    measurement[:, layout.price_slice] = np.eye(n)
    resolved_noise = _measurement_noise(
        specification.measurement_noise,
        residual_covariance,
        kalman_config=specification,
    )
    transition = readonly_array(transition, dimensions=2)
    intercept = readonly_array(intercept, dimensions=1)
    loading = readonly_array(loading, dimensions=2)
    measurement = readonly_array(measurement, dimensions=2)
    process = ProcessNoiseSpecification(
        source="Q_state = G Sigma_epsilon G'",
        structural_fit_id=structural_fit.metadata.fit_id,
        residual_covariance=residual_covariance,
        innovation_loading=loading,
        state_covariance=process_covariance,
    )
    model = VECMStateSpace(
        structural_fit=structural_fit,
        vecm=vecm,
        layout=layout,
        transition_matrix=transition,
        transition_intercept=intercept,
        innovation_loading=loading,
        process_covariance=process_covariance,
        measurement_matrix=measurement,
        measurement_noise=resolved_noise,
        process_noise=process,
        config=specification,
    )
    return StateSpaceAvailability(
        StateSpaceStatus.ACTIVE,
        structural_fit.metadata.fit_id,
        rank,
        model,
        None,
    )
