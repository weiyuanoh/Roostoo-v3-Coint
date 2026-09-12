"""Multi-asset portfolio accounting under a segregated short-collateral convention."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Mapping

from stat_arb_bot.accounting.models import (
    FillApplication,
    PortfolioSnapshot,
    PositionSnapshot,
)
from stat_arb_bot.accounting.position import Position
from stat_arb_bot.domain.execution import (
    ZERO,
    DecimalLike,
    Fill,
    FinancingEntry,
    decimal_value,
    utc_datetime,
)

STATE_VERSION = 1


class Portfolio:
    """Cash, restricted short collateral, and signed instrument positions.

    Equity is defined as::

        available cash
        + long market value
        + restricted short collateral
        + short unrealized PnL

    Short-sale proceeds are not treated as available cash. The transition
    engine depends only on normalized fills, never exchange actions or endpoint
    names.
    """

    def __init__(self, initial_cash: DecimalLike, *, base_currency: str = "USD") -> None:
        normalized_cash = decimal_value(initial_cash, name="initial_cash")
        if normalized_cash <= ZERO:
            raise ValueError("initial_cash must be positive")
        normalized_currency = base_currency.strip().upper()
        if not normalized_currency:
            raise ValueError("base_currency must not be empty")
        self.initial_cash = normalized_cash
        self.base_currency = normalized_currency
        self._cash = normalized_cash
        self._positions: dict[str, Position] = {}
        self._fills: list[Fill] = []
        self._financing_entries: list[FinancingEntry] = []
        self._unattributed_financing = ZERO

    @property
    def cash(self) -> Decimal:
        return self._cash

    @property
    def available_cash(self) -> Decimal:
        return self._cash

    @property
    def fills(self) -> tuple[Fill, ...]:
        return tuple(self._fills)

    @property
    def financing_entries(self) -> tuple[FinancingEntry, ...]:
        return tuple(self._financing_entries)

    @property
    def positions(self) -> Mapping[str, PositionSnapshot]:
        return MappingProxyType(
            {symbol: position.snapshot() for symbol, position in sorted(self._positions.items())}
        )

    def position(self, symbol: str) -> PositionSnapshot | None:
        position = self._positions.get(symbol.strip().upper())
        return position.snapshot() if position else None

    def apply_fill(self, fill: Fill) -> FillApplication:
        """Apply a normalized fill exactly once and update cash atomically."""

        fee_currency = fill.fee_currency or self.base_currency
        if fill.fee != ZERO and fee_currency != self.base_currency:
            raise ValueError(
                f"fee currency {fee_currency} cannot be accounted in {self.base_currency} "
                "without an explicit conversion"
            )

        position = self._positions.get(fill.symbol)
        is_new = position is None
        if position is None:
            position = Position(fill.symbol)
        transition = position.apply_fill(fill)

        short_realized = (
            transition.realized_gross_pnl if transition.covered_short_quantity > ZERO else ZERO
        )
        cash_change = (
            transition.closed_long_quantity * fill.price
            - transition.opened_long_quantity * fill.price
            - transition.added_short_collateral
            + transition.released_short_collateral
            + short_realized
            - fill.fee
        )
        cash_before = self._cash
        self._cash += cash_change
        if is_new:
            self._positions[fill.symbol] = position
        self._fills.append(fill)
        return FillApplication(
            transition=transition,
            cash_before=cash_before,
            cash_change=cash_change,
            cash_after=self._cash,
        )

    def apply_financing(self, entry: FinancingEntry) -> None:
        """Apply a signed charge/credit to cash and its attribution ledger."""

        self._cash += entry.amount
        if entry.symbol is None:
            self._unattributed_financing += entry.amount
        else:
            position = self._positions.get(entry.symbol)
            if position is None:
                position = Position(entry.symbol)
                self._positions[entry.symbol] = position
            position.apply_financing(entry)
        self._financing_entries.append(entry)

    def mark(self, symbol: str, price: DecimalLike, timestamp: datetime) -> None:
        normalized_symbol = symbol.strip().upper()
        try:
            position = self._positions[normalized_symbol]
        except KeyError as exc:
            raise KeyError(f"no position state exists for {normalized_symbol}") from exc
        position.mark(price, timestamp)

    def mark_many(
        self,
        marks: Mapping[str, DecimalLike],
        timestamp: datetime,
    ) -> None:
        normalized_time = utc_datetime(timestamp)
        normalized_marks = {
            symbol.strip().upper(): decimal_value(price, name=f"mark price for {symbol}")
            for symbol, price in marks.items()
        }
        unknown = sorted(set(normalized_marks) - set(self._positions))
        if unknown:
            raise KeyError(f"no position state exists for: {', '.join(unknown)}")
        invalid = [symbol for symbol, price in normalized_marks.items() if price <= ZERO]
        if invalid:
            raise ValueError(f"mark prices must be positive for: {', '.join(sorted(invalid))}")
        for symbol, price in normalized_marks.items():
            self._positions[symbol].mark(price, normalized_time)

    def snapshot(self, timestamp: datetime | None = None) -> PortfolioSnapshot:
        snapshot_time = utc_datetime(timestamp or datetime.now(timezone.utc))
        positions = tuple(position.snapshot() for _, position in sorted(self._positions.items()))
        long_market_value = sum(
            (position.market_value for position in positions if position.quantity > ZERO),
            ZERO,
        )
        short_market_value = sum(
            (position.market_value for position in positions if position.quantity < ZERO),
            ZERO,
        )
        short_exposure = abs(short_market_value)
        gross_market_value = long_market_value + short_exposure
        net_market_value = long_market_value + short_market_value
        restricted_collateral = sum((position.short_collateral for position in positions), ZERO)
        short_unrealized = sum(
            (position.unrealized_pnl for position in positions if position.quantity < ZERO),
            ZERO,
        )
        unrealized = sum((position.unrealized_pnl for position in positions), ZERO)
        realized = sum((position.realized_gross_pnl for position in positions), ZERO)
        fees = sum((position.cumulative_fees for position in positions), ZERO)
        financing = self._unattributed_financing + sum(
            (position.financing for position in positions), ZERO
        )
        equity = self._cash + long_market_value + restricted_collateral + short_unrealized
        net_pnl = realized + unrealized - fees + financing
        gross_leverage = gross_market_value / equity if equity > ZERO else None
        return PortfolioSnapshot(
            timestamp=snapshot_time,
            base_currency=self.base_currency,
            initial_cash=self.initial_cash,
            cash=self._cash,
            available_cash=self._cash,
            restricted_short_collateral=restricted_collateral,
            equity=equity,
            gross_market_value=gross_market_value,
            net_market_value=net_market_value,
            long_market_value=long_market_value,
            short_market_value=short_market_value,
            short_exposure=short_exposure,
            gross_leverage=gross_leverage,
            realized_gross_pnl=realized,
            unrealized_pnl=unrealized,
            cumulative_fees=fees,
            financing=financing,
            net_pnl=net_pnl,
            positions=positions,
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "base_currency": self.base_currency,
            "initial_cash": str(self.initial_cash),
            "cash": str(self._cash),
            "unattributed_financing": str(self._unattributed_financing),
            "positions": [position.to_state() for _, position in sorted(self._positions.items())],
            "fills": [fill.to_state() for fill in self._fills],
            "financing_entries": [entry.to_state() for entry in self._financing_entries],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Portfolio:
        version = int(state.get("version", 0))
        if version != STATE_VERSION:
            raise ValueError(f"unsupported portfolio state version: {version}")
        portfolio = cls(
            str(state["initial_cash"]),
            base_currency=str(state.get("base_currency", "USD")),
        )
        portfolio._cash = decimal_value(state["cash"], name="cash")
        portfolio._unattributed_financing = decimal_value(
            state.get("unattributed_financing", "0"),
            name="unattributed_financing",
        )
        positions = [Position.from_state(item) for item in state.get("positions", [])]
        if len({position.symbol for position in positions}) != len(positions):
            raise ValueError("portfolio state contains duplicate position symbols")
        portfolio._positions = {position.symbol: position for position in positions}
        portfolio._fills = [Fill.from_state(item) for item in state.get("fills", [])]
        portfolio._financing_entries = [
            FinancingEntry.from_state(item) for item in state.get("financing_entries", [])
        ]
        return portfolio
