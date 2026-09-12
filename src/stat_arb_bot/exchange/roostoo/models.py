"""Typed response models for the documented Roostoo v6 short API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stat_arb_bot.exchange.roostoo.errors import RoostooResponseError


def _required(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise RoostooResponseError(f"short API response is missing {key}")
    return payload[key]


def _int(payload: dict[str, Any], key: str) -> int:
    try:
        return int(_required(payload, key))
    except (TypeError, ValueError) as exc:
        raise RoostooResponseError(f"short API response field {key} must be an integer") from exc


def _float(payload: dict[str, Any], key: str) -> float:
    try:
        return float(_required(payload, key))
    except (TypeError, ValueError) as exc:
        raise RoostooResponseError(f"short API response field {key} must be numeric") from exc


def _optional_float(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RoostooResponseError(f"short API response field {key} must be numeric") from exc


def _bool(payload: dict[str, Any], key: str) -> bool:
    value = _required(payload, key)
    if not isinstance(value, bool):
        raise RoostooResponseError(f"short API response field {key} must be a boolean")
    return value


@dataclass(frozen=True)
class ShortOpenResult:
    id: int
    pair: str
    order_type: str
    entry_price: float
    short_quantity: float
    collateral: float
    open_fee: float
    status: str
    create_timestamp: int

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ShortOpenResult":
        return cls(
            id=_int(payload, "ID"),
            pair=str(_required(payload, "Pair")),
            order_type=str(_required(payload, "OrderType")).upper(),
            entry_price=_float(payload, "EntryPrice"),
            short_quantity=_float(payload, "ShortQty"),
            collateral=_float(payload, "Collateral"),
            open_fee=_float(payload, "OpenFee"),
            status=str(_required(payload, "Status")).upper(),
            create_timestamp=_int(payload, "CreateTimestamp"),
        )


@dataclass(frozen=True)
class ShortCloseResult:
    close_price: float
    realized_pnl: float
    close_fee: float
    return_amount: float
    closed_quantity: float
    fully_closed: bool
    remaining_quantity: float | None = None
    remaining_collateral: float | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ShortCloseResult":
        return cls(
            close_price=_float(payload, "ClosePrice"),
            realized_pnl=_float(payload, "RealizedPNL"),
            close_fee=_float(payload, "CloseFee"),
            return_amount=_float(payload, "ReturnAmount"),
            closed_quantity=_float(payload, "ClosedQty"),
            fully_closed=_bool(payload, "FullyClosed"),
            remaining_quantity=_optional_float(payload, "RemainingQty"),
            remaining_collateral=_optional_float(payload, "RemainingCollateral"),
        )


@dataclass(frozen=True)
class ShortPosition:
    id: int
    pair: str
    entry_price: float
    short_quantity: float
    collateral: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    position_value: float
    create_timestamp: int
    status: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ShortPosition":
        return cls(
            id=_int(payload, "ID"),
            pair=str(_required(payload, "Pair")),
            entry_price=_float(payload, "EntryPrice"),
            short_quantity=_float(payload, "ShortQty"),
            collateral=_float(payload, "Collateral"),
            current_price=_float(payload, "CurrentPrice"),
            unrealized_pnl=_float(payload, "UnrealizedPNL"),
            unrealized_pnl_pct=_float(payload, "UnrealizedPNLPct"),
            position_value=_float(payload, "PositionValue"),
            create_timestamp=_int(payload, "CreateTimestamp"),
            status=str(_required(payload, "PositionStatus")).upper(),
        )
