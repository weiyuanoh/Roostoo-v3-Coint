"""Small numerical safeguards shared by model result objects."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def readonly_array(value: ArrayLike, *, dimensions: int | None = None) -> NDArray[np.float64]:
    array = np.array(value, dtype=np.float64, copy=True)
    if dimensions is not None and array.ndim != dimensions:
        raise ValueError(f"expected a {dimensions}-dimensional array, got {array.ndim}")
    if not np.all(np.isfinite(array)):
        raise ValueError("model arrays must contain only finite values")
    array.setflags(write=False)
    return array


def sign_aligned_unit_vector(beta: ArrayLike) -> tuple[NDArray[np.float64], float]:
    """Return L2-normalized beta and the exact scalar applied to native beta."""

    vector = readonly_array(beta, dimensions=1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("beta must have a positive finite L2 norm")
    anchor = int(np.argmax(np.abs(vector)))
    sign = 1.0 if vector[anchor] >= 0 else -1.0
    scale = sign / norm
    normalized = readonly_array(vector * scale, dimensions=1)
    return normalized, scale


def finite_tuple(values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not all(np.isfinite(result)):
        raise ValueError("values must be finite")
    return result
