"""Covariance validation for auditable state-space calculations."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stat_arb_bot.models.numeric import readonly_array


class StateSpaceNumericalError(ValueError):
    """Raised instead of silently repairing structurally invalid numerics."""


def validated_covariance(
    value: ArrayLike,
    *,
    dimension: int,
    name: str,
    psd_tolerance: float,
    symmetry_tolerance: float,
    require_positive_diagonal: bool = False,
) -> NDArray[np.float64]:
    covariance = readonly_array(value, dimensions=2)
    if covariance.shape != (dimension, dimension):
        raise StateSpaceNumericalError(
            f"{name} must have shape {(dimension, dimension)}, got {covariance.shape}"
        )
    if not np.allclose(
        covariance,
        covariance.T,
        rtol=0.0,
        atol=symmetry_tolerance,
    ):
        raise StateSpaceNumericalError(f"{name} is materially asymmetric")
    symmetric = readonly_array((covariance + covariance.T) / 2.0, dimensions=2)
    diagonal = np.diag(symmetric)
    if np.any(diagonal < -psd_tolerance):
        raise StateSpaceNumericalError(f"{name} has a negative diagonal variance")
    if require_positive_diagonal and np.any(diagonal <= 0):
        raise StateSpaceNumericalError(f"{name} diagonal variances must be positive")
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(symmetric)))
    if minimum_eigenvalue < -psd_tolerance:
        raise StateSpaceNumericalError(
            f"{name} is materially non-PSD (minimum eigenvalue {minimum_eigenvalue})"
        )
    return symmetric


def ensure_finite_state(
    value: ArrayLike,
    *,
    dimension: int,
    maximum_norm: float,
    name: str,
) -> NDArray[np.float64]:
    state = readonly_array(value, dimensions=1)
    if state.shape != (dimension,):
        raise StateSpaceNumericalError(f"{name} must have shape {(dimension,)}, got {state.shape}")
    norm = float(np.linalg.norm(state))
    if norm > maximum_norm:
        raise StateSpaceNumericalError(f"{name} is exploding (norm {norm})")
    return state
