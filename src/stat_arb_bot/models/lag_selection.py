"""Parsimonious levels-VAR lag selection with explicit VECM mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from statsmodels.tsa.api import VAR

from stat_arb_bot.models.config import LagCriterion, StructuralModelConfig, vecm_lagged_differences
from stat_arb_bot.models.numeric import readonly_array


@dataclass(frozen=True, slots=True)
class LagCriterionRow:
    var_order: int
    aic: float | None
    bic: float | None
    hqic: float | None
    failure: str | None = None


@dataclass(frozen=True, slots=True)
class LagSelectionResult:
    criterion: LagCriterion
    selected_var_order: int
    vecm_lagged_differences: int
    rows: tuple[LagCriterionRow, ...]
    sample_count: int


def select_var_lag(
    levels: ArrayLike,
    config: StructuralModelConfig,
) -> LagSelectionResult:
    values = readonly_array(levels, dimensions=2)
    rows: list[LagCriterionRow] = []
    for order in range(config.minimum_var_lag, config.maximum_var_lag + 1):
        try:
            fitted = VAR(values).fit(order, trend="c")
            criteria = (float(fitted.aic), float(fitted.bic), float(fitted.hqic))
            if not all(np.isfinite(criteria)):
                raise ValueError("one or more information criteria are non-finite")
            rows.append(LagCriterionRow(order, *criteria))
        except (ValueError, np.linalg.LinAlgError) as exc:
            rows.append(LagCriterionRow(order, None, None, None, str(exc)))
    field = config.lag_criterion.value
    eligible = [row for row in rows if getattr(row, field) is not None]
    if not eligible:
        raise ValueError("VAR lag selection failed for every configured order")
    selected = min(eligible, key=lambda row: (getattr(row, field), row.var_order))
    return LagSelectionResult(
        criterion=config.lag_criterion,
        selected_var_order=selected.var_order,
        vecm_lagged_differences=vecm_lagged_differences(selected.var_order),
        rows=tuple(rows),
        sample_count=len(values),
    )
