"""Joint Gamma restrictions for transparent short-run lead/lag evidence."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from stat_arb_bot.models.roles.results import (
    AssetLeadLagSummary,
    GammaLeadLagDiagnostic,
    GammaTestStatus,
)
from stat_arb_bot.models.vecm import RankOneVECMResult


def _relationship(
    vecm: RankOneVECMResult,
    *,
    source_index: int,
    target_index: int,
    significance_level: float,
) -> GammaLeadLagDiagnostic:
    source = vecm.assets[source_index]
    target = vecm.assets[target_index]
    lag_count = vecm.lagged_difference_count
    if lag_count == 0:
        return GammaLeadLagDiagnostic(
            source,
            target,
            (),
            (),
            (),
            (),
            0.0,
            None,
            0,
            None,
            False,
            GammaTestStatus.NO_LAGGED_DIFFERENCES,
            "VAR(1) has no lagged-difference Gamma coefficients to test",
        )

    coefficients = np.asarray(
        [matrix[target_index, source_index] for matrix in vecm.gamma_matrices],
        dtype=np.float64,
    )
    standard_errors = np.asarray(
        [matrix[target_index, source_index] for matrix in vecm.gamma_standard_errors],
        dtype=np.float64,
    )
    asset_count = len(vecm.assets)
    # vec(Gamma) follows Fortran order: within each parameter column, target
    # equations vary fastest. Gamma columns are [lag_1 assets, lag_2 assets, ...].
    indices = np.asarray(
        [
            ((lag * asset_count + source_index) * asset_count) + target_index
            for lag in range(lag_count)
        ],
        dtype=np.int64,
    )
    full_covariance = np.asarray(vecm.gamma_parameter_covariance, dtype=np.float64)
    expected = asset_count * asset_count * lag_count
    if full_covariance.shape != (expected, expected):
        return _invalid(
            source,
            target,
            coefficients,
            standard_errors,
            "Gamma parameter covariance has an incompatible shape",
        )
    restriction_covariance = full_covariance[np.ix_(indices, indices)]
    if not np.all(np.isfinite(restriction_covariance)) or not np.allclose(
        restriction_covariance,
        restriction_covariance.T,
        rtol=1e-8,
        atol=1e-12,
    ):
        return _invalid(
            source,
            target,
            coefficients,
            standard_errors,
            "Gamma restriction covariance is non-finite or asymmetric",
        )
    eigenvalues = np.linalg.eigvalsh(restriction_covariance)
    tolerance = max(1e-14, float(np.max(np.abs(eigenvalues))) * 1e-10)
    if np.min(eigenvalues) <= tolerance:
        return _invalid(
            source,
            target,
            coefficients,
            standard_errors,
            "Gamma restriction covariance is singular or not positive definite",
        )
    try:
        statistic = float(coefficients @ np.linalg.solve(restriction_covariance, coefficients))
    except np.linalg.LinAlgError:
        return _invalid(
            source,
            target,
            coefficients,
            standard_errors,
            "Gamma restriction covariance could not be solved",
        )
    statistic = max(statistic, 0.0)
    p_value = float(chi2.sf(statistic, lag_count))
    return GammaLeadLagDiagnostic(
        source_asset=source,
        target_asset=target,
        lag_set=tuple(range(1, lag_count + 1)),
        coefficients=tuple(float(value) for value in coefficients),
        coefficient_standard_errors=tuple(float(value) for value in standard_errors),
        coefficient_signs=tuple(int(np.sign(value)) for value in coefficients),
        coefficient_sum=float(np.sum(coefficients)),
        wald_statistic=statistic,
        degrees_of_freedom=lag_count,
        p_value=p_value,
        significant=p_value < significance_level,
        status=GammaTestStatus.TESTED,
        reason=None,
    )


def _invalid(
    source: str,
    target: str,
    coefficients: np.ndarray,
    standard_errors: np.ndarray,
    reason: str,
) -> GammaLeadLagDiagnostic:
    return GammaLeadLagDiagnostic(
        source_asset=source,
        target_asset=target,
        lag_set=tuple(range(1, len(coefficients) + 1)),
        coefficients=tuple(float(value) for value in coefficients),
        coefficient_standard_errors=tuple(float(value) for value in standard_errors),
        coefficient_signs=tuple(int(np.sign(value)) for value in coefficients),
        coefficient_sum=float(np.sum(coefficients)),
        wald_statistic=None,
        degrees_of_freedom=len(coefficients),
        p_value=None,
        significant=False,
        status=GammaTestStatus.INVALID_COVARIANCE,
        reason=reason,
    )


def gamma_lead_lag_diagnostics(
    vecm: RankOneVECMResult,
    *,
    significance_level: float,
) -> tuple[GammaLeadLagDiagnostic, ...]:
    """Test every directed cross-asset Gamma relationship jointly over all lags."""

    results = []
    for source_index in range(len(vecm.assets)):
        for target_index in range(len(vecm.assets)):
            if source_index == target_index:
                continue
            results.append(
                _relationship(
                    vecm,
                    source_index=source_index,
                    target_index=target_index,
                    significance_level=significance_level,
                )
            )
    return tuple(results)


def summarize_lead_lag(
    assets: tuple[str, ...],
    diagnostics: tuple[GammaLeadLagDiagnostic, ...],
) -> tuple[AssetLeadLagSummary, ...]:
    """Retain incoming and outgoing evidence as separate, inspectable collections."""

    summaries = []
    for asset in assets:
        outgoing = tuple(item for item in diagnostics if item.source_asset == asset)
        incoming = tuple(item for item in diagnostics if item.target_asset == asset)
        summaries.append(
            AssetLeadLagSummary(
                asset=asset,
                outgoing=outgoing,
                incoming=incoming,
                significant_outgoing=tuple(item for item in outgoing if item.significant),
                significant_incoming=tuple(item for item in incoming if item.significant),
            )
        )
    return tuple(summaries)
