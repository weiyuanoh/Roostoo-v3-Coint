from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from stat_arb_bot.accounting import Position, PositionDirection
from stat_arb_bot.domain import Fill, Side

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def fill(
    side: Side,
    quantity: str,
    price: str,
    *,
    hours: int = 0,
    collateral: str = "0",
) -> Fill:
    return Fill(
        symbol="BTC/USD",
        side=side,
        quantity=quantity,
        price=price,
        timestamp=T0 + timedelta(hours=hours),
        short_collateral=collateral,
    )


def test_flat_to_long() -> None:
    position = Position("BTC/USD")

    transition = position.apply_fill(fill(Side.BUY, "10", "100"))

    assert position.quantity == Decimal("10")
    assert position.average_entry_price == Decimal("100")
    assert position.unrealized_pnl == Decimal("0")
    assert transition.opened_long_quantity == Decimal("10")
    assert position.snapshot().direction is PositionDirection.LONG


def test_add_to_long_uses_quantity_weighted_average_without_realized_pnl() -> None:
    position = Position("BTC/USD")
    position.apply_fill(fill(Side.BUY, "10", "100"))

    transition = position.apply_fill(fill(Side.BUY, "10", "120", hours=1))

    assert position.quantity == Decimal("20")
    assert position.average_entry_price == Decimal("110")
    assert transition.realized_gross_pnl == Decimal("0")


def test_partially_close_long_realizes_pnl_and_preserves_average() -> None:
    position = Position("BTC/USD")
    position.apply_fill(fill(Side.BUY, "10", "100"))

    transition = position.apply_fill(fill(Side.SELL, "6", "120", hours=1))

    assert position.quantity == Decimal("4")
    assert position.average_entry_price == Decimal("100")
    assert position.realized_gross_pnl == Decimal("120")
    assert transition.closed_long_quantity == Decimal("6")


def test_fully_close_long_resets_average_and_unrealized() -> None:
    position = Position("BTC/USD")
    position.apply_fill(fill(Side.BUY, "10", "100"))

    position.apply_fill(fill(Side.SELL, "10", "90", hours=1))

    assert position.quantity == Decimal("0")
    assert position.average_entry_price == Decimal("0")
    assert position.unrealized_pnl == Decimal("0")
    assert position.realized_gross_pnl == Decimal("-100")
    assert position.opened_at is None


def test_flat_to_short_uses_negative_signed_inventory() -> None:
    position = Position("BTC/USD")

    transition = position.apply_fill(fill(Side.SELL, "10", "100", collateral="1000"))

    assert position.quantity == Decimal("-10")
    assert position.average_entry_price == Decimal("100")
    assert position.short_collateral == Decimal("1000")
    assert transition.opened_short_quantity == Decimal("10")
    assert position.snapshot().direction is PositionDirection.SHORT


def test_add_to_short_uses_quantity_weighted_average_without_realized_pnl() -> None:
    position = Position("BTC/USD")
    position.apply_fill(fill(Side.SELL, "10", "100", collateral="1000"))

    transition = position.apply_fill(fill(Side.SELL, "10", "120", hours=1, collateral="1200"))

    assert position.quantity == Decimal("-20")
    assert position.average_entry_price == Decimal("110")
    assert position.short_collateral == Decimal("2200")
    assert transition.realized_gross_pnl == Decimal("0")


def test_partially_cover_short_realizes_pnl_and_preserves_average() -> None:
    position = Position("BTC/USD")
    position.apply_fill(fill(Side.SELL, "10", "100", collateral="1000"))

    transition = position.apply_fill(fill(Side.BUY, "6", "80", hours=1))

    assert position.quantity == Decimal("-4")
    assert position.average_entry_price == Decimal("100")
    assert position.realized_gross_pnl == Decimal("120")
    assert position.short_collateral == Decimal("400")
    assert transition.covered_short_quantity == Decimal("6")
    assert transition.released_short_collateral == Decimal("600")


def test_fully_cover_short_resets_average_collateral_and_unrealized() -> None:
    position = Position("BTC/USD")
    position.apply_fill(fill(Side.SELL, "10", "100", collateral="1000"))

    position.apply_fill(fill(Side.BUY, "10", "110", hours=1))

    assert position.quantity == Decimal("0")
    assert position.average_entry_price == Decimal("0")
    assert position.short_collateral == Decimal("0")
    assert position.unrealized_pnl == Decimal("0")
    assert position.realized_gross_pnl == Decimal("-100")


def test_long_to_short_reversal_equals_close_then_open() -> None:
    reversed_position = Position("BTC/USD")
    reversed_position.apply_fill(fill(Side.BUY, "10", "100"))

    transition = reversed_position.apply_fill(
        fill(Side.SELL, "15", "110", hours=1, collateral="550")
    )

    split_position = Position("BTC/USD")
    split_position.apply_fill(fill(Side.BUY, "10", "100"))
    split_position.apply_fill(fill(Side.SELL, "10", "110", hours=1))
    split_position.apply_fill(fill(Side.SELL, "5", "110", hours=1, collateral="550"))

    assert transition.reversed is True
    assert transition.closed_long_quantity == Decimal("10")
    assert transition.opened_short_quantity == Decimal("5")
    assert reversed_position.snapshot() == split_position.snapshot()
    assert reversed_position.quantity == Decimal("-5")
    assert reversed_position.average_entry_price == Decimal("110")


def test_short_to_long_reversal_equals_cover_then_open() -> None:
    reversed_position = Position("BTC/USD")
    reversed_position.apply_fill(fill(Side.SELL, "10", "100", collateral="1000"))

    transition = reversed_position.apply_fill(fill(Side.BUY, "15", "90", hours=1))

    split_position = Position("BTC/USD")
    split_position.apply_fill(fill(Side.SELL, "10", "100", collateral="1000"))
    split_position.apply_fill(fill(Side.BUY, "10", "90", hours=1))
    split_position.apply_fill(fill(Side.BUY, "5", "90", hours=1))

    assert transition.reversed is True
    assert transition.covered_short_quantity == Decimal("10")
    assert transition.opened_long_quantity == Decimal("5")
    assert reversed_position.snapshot() == split_position.snapshot()
    assert reversed_position.quantity == Decimal("5")
    assert reversed_position.average_entry_price == Decimal("90")
