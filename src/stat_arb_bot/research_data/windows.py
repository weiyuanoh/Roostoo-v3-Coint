"""Bounded research windows with structured minimum-history validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from math import isfinite

from stat_arb_bot.domain.execution import utc_datetime
from stat_arb_bot.research_data.aggregation import MODEL_STEP
from stat_arb_bot.research_data.panel import ResearchMatrix, ResearchPanel, ResearchPanelRow


class WindowIssue(str, Enum):
    INVALID_CHRONOLOGY = "INVALID_CHRONOLOGY"
    INVALID_OBSERVATION_REQUEST = "INVALID_OBSERVATION_REQUEST"
    END_NOT_AVAILABLE = "END_NOT_AVAILABLE"
    INSUFFICIENT_OBSERVATIONS = "INSUFFICIENT_OBSERVATIONS"
    INTERNAL_GAP_TOO_LARGE = "INTERNAL_GAP_TOO_LARGE"
    INCOMPLETE_UNIVERSE = "INCOMPLETE_UNIVERSE"
    NON_FINITE_VALUE = "NON_FINITE_VALUE"


@dataclass(frozen=True, slots=True)
class WindowValidationResult:
    valid: bool
    issues: tuple[WindowIssue, ...]
    requested_minimum_observations: int
    actual_observations: int
    maximum_observed_gap: timedelta | None


@dataclass(frozen=True, slots=True)
class ResearchWindow:
    universe_assets: tuple[str, ...]
    quote_currency: str
    rows: tuple[ResearchPanelRow, ...]
    requested_end: datetime
    requested_start: datetime | None
    requested_observations: int | None
    validation: WindowValidationResult

    @property
    def actual_observations(self) -> int:
        return len(self.rows)

    @property
    def timestamps(self) -> tuple[datetime, ...]:
        return tuple(row.timestamp for row in self.rows)

    @property
    def data_start(self) -> datetime | None:
        return self.rows[0].timestamp if self.rows else None

    @property
    def data_end(self) -> datetime | None:
        return self.rows[-1].timestamp if self.rows else None

    def close_prices(self) -> ResearchMatrix:
        return self._panel().close_prices()

    def log_prices(self) -> ResearchMatrix:
        return self._panel().log_prices()

    def log_returns(self) -> ResearchMatrix:
        return self._panel().log_returns()

    def _panel(self) -> ResearchPanel:
        from stat_arb_bot.research_data.panel import PanelDiagnostics
        from stat_arb_bot.research_data.universe import ResearchUniverse

        diagnostics = PanelDiagnostics(
            received_rows=0,
            synchronized_rows=len(self.rows),
            exact_duplicate_keys=(),
            out_of_order=False,
            non_monotonic=False,
            forming_keys=(),
            unexpected_symbols=(),
            unexpected_intervals=(),
            dropped_timestamps=(),
            missing_panel_timestamps=(),
        )
        return ResearchPanel(ResearchUniverse(self.quote_currency), self.rows, diagnostics)


def build_research_window(
    panel: ResearchPanel,
    *,
    end: datetime,
    observations: int | None = None,
    start: datetime | None = None,
    lookback: timedelta | None = None,
    minimum_observations: int = 1,
    maximum_internal_gap: timedelta | None = MODEL_STEP,
) -> ResearchWindow:
    """Return an immutable prefix ending no later than the requested timestamp."""

    boundary = utc_datetime(end, name="end")
    normalized_start = utc_datetime(start, name="start") if start is not None else None
    issues: list[WindowIssue] = []
    selectors = sum(value is not None for value in (observations, normalized_start, lookback))
    if selectors > 1 or observations is not None and observations <= 0:
        issues.append(WindowIssue.INVALID_OBSERVATION_REQUEST)
    if minimum_observations <= 0:
        issues.append(WindowIssue.INVALID_OBSERVATION_REQUEST)
    if maximum_internal_gap is not None and maximum_internal_gap <= timedelta(0):
        issues.append(WindowIssue.INVALID_OBSERVATION_REQUEST)
    if lookback is not None:
        if lookback <= timedelta(0):
            issues.append(WindowIssue.INVALID_CHRONOLOGY)
        normalized_start = boundary - lookback
    if normalized_start is not None and normalized_start >= boundary:
        issues.append(WindowIssue.INVALID_CHRONOLOGY)

    eligible = [row for row in panel.rows if row.timestamp <= boundary]
    if normalized_start is not None:
        if lookback is not None:
            eligible = [row for row in eligible if row.timestamp > normalized_start]
        else:
            eligible = [row for row in eligible if row.timestamp >= normalized_start]
    if observations is not None and observations > 0:
        eligible = eligible[-observations:]
    rows = tuple(eligible)
    if not rows or rows[-1].timestamp != boundary:
        issues.append(WindowIssue.END_NOT_AVAILABLE)
    required_count = max(minimum_observations, observations or 0)
    if len(rows) < required_count:
        issues.append(WindowIssue.INSUFFICIENT_OBSERVATIONS)

    expected_assets = set(panel.universe.assets)
    if any(set(row.bars) != expected_assets for row in rows):
        issues.append(WindowIssue.INCOMPLETE_UNIVERSE)
    if any(not isfinite(row.bars[asset].close) for row in rows for asset in panel.universe.assets):
        issues.append(WindowIssue.NON_FINITE_VALUE)
    gaps = tuple(right.timestamp - left.timestamp for left, right in zip(rows, rows[1:]))
    maximum_gap = max(gaps) if gaps else None
    if (
        maximum_internal_gap is not None
        and maximum_gap is not None
        and maximum_gap > maximum_internal_gap
    ):
        issues.append(WindowIssue.INTERNAL_GAP_TOO_LARGE)

    unique_issues = tuple(dict.fromkeys(issues))
    validation = WindowValidationResult(
        valid=not unique_issues,
        issues=unique_issues,
        requested_minimum_observations=minimum_observations,
        actual_observations=len(rows),
        maximum_observed_gap=maximum_gap,
    )
    return ResearchWindow(
        universe_assets=panel.universe.assets,
        quote_currency=panel.universe.quote_currency,
        rows=rows,
        requested_end=boundary,
        requested_start=normalized_start,
        requested_observations=observations,
        validation=validation,
    )
