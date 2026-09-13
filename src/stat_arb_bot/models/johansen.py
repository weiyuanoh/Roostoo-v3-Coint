"""Johansen rank diagnostics without forcing a tradable rank."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import ArrayLike, NDArray
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from stat_arb_bot.models.config import StructuralModelConfig
from stat_arb_bot.models.numeric import readonly_array


@dataclass(frozen=True, slots=True)
class JohansenResult:
    inferred_rank: int
    trace_rank: int
    maximum_eigenvalue_rank: int
    significance_level: float
    deterministic_order: int
    var_order: int
    vecm_lagged_differences: int
    sample_count: int
    trace_statistics: NDArray
    trace_critical_values: NDArray
    maximum_eigenvalue_statistics: NDArray
    maximum_eigenvalue_critical_values: NDArray
    eigenvalues: NDArray
    eigenvectors: NDArray


def _sequential_rank(statistics: NDArray, critical: NDArray) -> int:
    for rank, (statistic, threshold) in enumerate(zip(statistics, critical)):
        if statistic <= threshold:
            return rank
    return len(statistics)


def estimate_johansen_rank(
    levels: ArrayLike,
    *,
    var_order: int,
    config: StructuralModelConfig,
) -> JohansenResult:
    values = readonly_array(levels, dimensions=2)
    lagged_differences = var_order - 1
    raw = coint_johansen(
        values,
        det_order=config.johansen_deterministic_order,
        k_ar_diff=lagged_differences,
    )
    column = {0.10: 0, 0.05: 1, 0.01: 2}[config.significance_level]
    trace = readonly_array(raw.lr1, dimensions=1)
    trace_critical = readonly_array(raw.cvt, dimensions=2)
    max_eigen = readonly_array(raw.lr2, dimensions=1)
    max_eigen_critical = readonly_array(raw.cvm, dimensions=2)
    trace_rank = _sequential_rank(trace, trace_critical[:, column])
    maximum_rank = _sequential_rank(max_eigen, max_eigen_critical[:, column])
    return JohansenResult(
        inferred_rank=trace_rank,
        trace_rank=trace_rank,
        maximum_eigenvalue_rank=maximum_rank,
        significance_level=config.significance_level,
        deterministic_order=config.johansen_deterministic_order,
        var_order=var_order,
        vecm_lagged_differences=lagged_differences,
        sample_count=len(values),
        trace_statistics=trace,
        trace_critical_values=trace_critical,
        maximum_eigenvalue_statistics=max_eigen,
        maximum_eigenvalue_critical_values=max_eigen_critical,
        eigenvalues=readonly_array(raw.eig, dimensions=1),
        eigenvectors=readonly_array(raw.evec, dimensions=2),
    )
