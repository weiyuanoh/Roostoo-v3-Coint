from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from stat_arb_bot.accounting import Portfolio
from stat_arb_bot.domain import Fill, FinancingEntry, Side
from stat_arb_bot.persistence import PortfolioStateError, PortfolioStateStore

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_portfolio_state_round_trip_preserves_signed_accounting(tmp_path: Path) -> None:
    portfolio = Portfolio("100000")
    portfolio.apply_fill(
        Fill(
            "BTC/USD",
            Side.BUY,
            "1",
            "50000",
            T0,
            fee="5",
            fee_currency="USD",
            exchange_order_id="long-1",
            trade_group_id="pair-1",
        )
    )
    portfolio.apply_fill(
        Fill(
            "ETH/USD",
            Side.SELL,
            "10",
            "3000",
            T0,
            fee="3",
            fee_currency="USD",
            short_collateral="30000",
            original_action="SHORT_OPEN",
            trade_group_id="pair-1",
            exchange_metadata={"position_id": 42},
        )
    )
    portfolio.mark_many({"BTC/USD": "52000", "ETH/USD": "2800"}, T0)
    portfolio.apply_financing(
        FinancingEntry(
            "-1.25",
            T0 + timedelta(hours=1),
            "borrow",
            symbol="ETH/USD",
            exchange_reference="funding-1",
        )
    )
    store = PortfolioStateStore(tmp_path / "portfolio.json")

    path = store.save(portfolio)
    restored = store.load()

    assert path == tmp_path / "portfolio.json"
    assert restored.snapshot(T0 + timedelta(hours=1)) == portfolio.snapshot(T0 + timedelta(hours=1))
    assert restored.fills == portfolio.fills
    assert restored.financing_entries == portfolio.financing_entries
    assert restored.position("ETH/USD").quantity == Decimal("-10")  # type: ignore[union-attr]
    assert not list(tmp_path.glob(".portfolio.json.*"))


def test_snapshot_and_position_map_are_read_only() -> None:
    portfolio = Portfolio("1000")
    portfolio.apply_fill(Fill("BTC/USD", Side.BUY, "1", "100", T0))
    snapshot = portfolio.snapshot(T0)

    with pytest.raises(FrozenInstanceError):
        snapshot.cash = Decimal("0")  # type: ignore[misc]
    with pytest.raises(TypeError):
        snapshot.positions_by_symbol["ETH/USD"] = snapshot.positions[0]  # type: ignore[index]


def test_state_store_rejects_missing_or_invalid_state(tmp_path: Path) -> None:
    store = PortfolioStateStore(tmp_path / "missing.json")
    with pytest.raises(PortfolioStateError, match="does not exist"):
        store.load()

    store.path.write_text('{"version": 999}', encoding="utf-8")
    with pytest.raises(PortfolioStateError, match="invalid portfolio state"):
        store.load()
