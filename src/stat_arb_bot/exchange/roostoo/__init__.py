"""Roostoo REST API adapter."""

from stat_arb_bot.exchange.roostoo.accounting import (
    RoostooAccountingMappingError,
    RoostooShortCloseReconciliation,
    reconcile_short_close,
    short_close_to_fill,
    short_open_to_fill,
    short_position_to_observation,
)
from stat_arb_bot.exchange.roostoo.client import RoostooClient

__all__ = [
    "RoostooAccountingMappingError",
    "RoostooClient",
    "RoostooShortCloseReconciliation",
    "reconcile_short_close",
    "short_close_to_fill",
    "short_open_to_fill",
    "short_position_to_observation",
]
