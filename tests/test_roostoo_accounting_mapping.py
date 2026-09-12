from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from stat_arb_bot.accounting import Portfolio, reconcile_position
from stat_arb_bot.domain import Side
from stat_arb_bot.exchange.roostoo import (
    RoostooAccountingMappingError,
    reconcile_short_close,
    short_close_to_fill,
    short_open_to_fill,
    short_position_to_observation,
)
from stat_arb_bot.exchange.roostoo.models import (
    ShortCloseResult,
    ShortOpenResult,
    ShortPosition,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def api_position(
    *,
    quantity: float = 10,
    entry: float = 100,
    collateral: float = 1000,
    current: float = 90,
    unrealized: float = 100,
) -> ShortPosition:
    return ShortPosition(
        id=42,
        pair="BTC/USD",
        entry_price=entry,
        short_quantity=quantity,
        collateral=collateral,
        current_price=current,
        unrealized_pnl=unrealized,
        unrealized_pnl_pct=unrealized / collateral,
        position_value=collateral + unrealized,
        create_timestamp=1767225600000,
        status="OPEN",
    )


def test_first_short_open_maps_positive_api_quantity_to_sell_fill() -> None:
    result = ShortOpenResult(
        id=42,
        pair="BTC/USD",
        order_type="MARKET",
        entry_price=100,
        short_quantity=10,
        collateral=1000,
        open_fee=1,
        status="OPEN",
        create_timestamp=1767225600000,
    )

    event = short_open_to_fill(result, previous=None, trade_group_id="pair-1")

    assert event.side is Side.SELL
    assert event.signed_quantity == Decimal("-10")
    assert event.short_collateral == Decimal("1000")
    assert event.original_action == "SHORT_OPEN"
    assert event.trade_group_id == "pair-1"
    assert event.exchange_order_id is None
    assert event.exchange_metadata["roostoo_position_id"] == 42


def test_merged_short_open_reconstructs_incremental_fill_from_reported_totals() -> None:
    previous = api_position(current=100, unrealized=0)
    merged = ShortOpenResult(
        id=42,
        pair="BTC/USD",
        order_type="MARKET",
        entry_price=110,
        short_quantity=20,
        collateral=2200,
        open_fee=1.2,
        status="OPEN",
        create_timestamp=1767229200000,
    )

    event = short_open_to_fill(merged, previous=previous)

    assert event.quantity == Decimal("10")
    assert event.price == Decimal("120")
    assert event.short_collateral == Decimal("1200")
    assert event.exchange_metadata["fill_price_reconstructed"] is True


def test_pending_short_open_is_not_misclassified_as_fill() -> None:
    pending = ShortOpenResult(
        id=99,
        pair="BTC/USD",
        order_type="LIMIT",
        entry_price=100,
        short_quantity=10,
        collateral=1000,
        open_fee=1,
        status="PENDING",
        create_timestamp=1767225600000,
    )

    with pytest.raises(RoostooAccountingMappingError, match="not a fill"):
        short_open_to_fill(pending, previous=None)


def test_partial_short_close_maps_to_buy_and_reported_collateral_release() -> None:
    result = ShortCloseResult(
        close_price=90,
        realized_pnl=40,
        close_fee=0.5,
        return_amount=439.5,
        closed_quantity=4,
        fully_closed=False,
        remaining_quantity=6,
        remaining_collateral=600,
    )

    event = short_close_to_fill(result, previous=api_position(), timestamp=NOW)

    assert event.side is Side.BUY
    assert event.quantity == Decimal("4")
    assert event.released_short_collateral == Decimal("400")
    assert event.original_action == "SHORT_CLOSE"
    assert event.exchange_metadata["reported_realized_pnl"] == "40"


def test_exchange_short_observation_reconciles_with_internal_signed_position() -> None:
    exchange_position = api_position()
    opening = ShortOpenResult(
        id=42,
        pair="BTC/USD",
        order_type="MARKET",
        entry_price=100,
        short_quantity=10,
        collateral=1000,
        open_fee=0,
        status="OPEN",
        create_timestamp=1767225600000,
    )
    portfolio = Portfolio("10000")
    portfolio.apply_fill(short_open_to_fill(opening, previous=None))
    portfolio.mark("BTC/USD", "90", NOW)

    observation = short_position_to_observation(exchange_position, observed_at=NOW)
    reconciliation = reconcile_position(portfolio.position("BTC/USD"), observation)  # type: ignore[arg-type]

    assert observation.signed_quantity == Decimal("-10")
    assert reconciliation.matches is True


def test_close_mapping_preserves_exchange_settlement_for_reconciliation() -> None:
    previous = api_position()
    result = ShortCloseResult(
        close_price=90,
        realized_pnl=40,
        close_fee=0.5,
        return_amount=439.5,
        closed_quantity=4,
        fully_closed=False,
        remaining_quantity=6,
        remaining_collateral=600,
    )
    portfolio = Portfolio("10000")
    portfolio.apply_fill(
        short_open_to_fill(
            ShortOpenResult(
                id=42,
                pair="BTC/USD",
                order_type="MARKET",
                entry_price=100,
                short_quantity=10,
                collateral=1000,
                open_fee=0,
                status="OPEN",
                create_timestamp=1767225600000,
            ),
            previous=None,
        )
    )

    application = portfolio.apply_fill(
        short_close_to_fill(result, previous=previous, timestamp=NOW)
    )
    reconciliation = reconcile_short_close(application, result)

    assert application.transition.covered_short_quantity == Decimal("4")
    assert application.transition.realized_gross_pnl == Decimal("40")
    assert portfolio.position("BTC/USD").quantity == Decimal("-6")  # type: ignore[union-attr]
    assert portfolio.position("BTC/USD").short_collateral == Decimal("600")  # type: ignore[union-attr]
    assert reconciliation.matches is True


def test_documented_sample_close_differences_are_surfaced_not_hidden() -> None:
    previous = api_position(
        quantity=0.01617,
        entry=61840.25,
        collateral=999.95,
        current=60500.1,
        unrealized=21.670225,
    )
    result = ShortCloseResult(
        close_price=60500.1,
        realized_pnl=10.8258,
        close_fee=0.0484,
        return_amount=510.7774,
        closed_quantity=0.008085,
        fully_closed=False,
        remaining_quantity=0.008085,
        remaining_collateral=499.98,
    )
    portfolio = Portfolio("10000")
    portfolio.apply_fill(
        short_open_to_fill(
            ShortOpenResult(
                id=42,
                pair="BTC/USD",
                order_type="MARKET",
                entry_price=61840.25,
                short_quantity=0.01617,
                collateral=999.95,
                open_fee=0,
                status="OPEN",
                create_timestamp=1767225600000,
            ),
            previous=None,
        )
    )

    application = portfolio.apply_fill(
        short_close_to_fill(result, previous=previous, timestamp=NOW)
    )
    reconciliation = reconcile_short_close(application, result)

    assert reconciliation.realized_pnl_delta != Decimal("0")
    assert reconciliation.return_amount_delta != Decimal("0")
    assert reconciliation.matches is False
