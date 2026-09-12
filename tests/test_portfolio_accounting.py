from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from stat_arb_bot.accounting import Portfolio
from stat_arb_bot.domain import Fill, FinancingEntry, Side

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def fill(
    symbol: str,
    side: Side,
    quantity: str,
    price: str,
    *,
    fee: str = "0",
    collateral: str = "0",
    group: str | None = None,
    hours: int = 0,
) -> Fill:
    return Fill(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        timestamp=T0 + timedelta(hours=hours),
        fee=fee,
        fee_currency="USD",
        short_collateral=collateral,
        trade_group_id=group,
    )


def assert_equity_identity(portfolio: Portfolio) -> None:
    snapshot = portfolio.snapshot(T0)
    short_unrealized = sum(
        (position.unrealized_pnl for position in snapshot.positions if position.quantity < 0),
        Decimal("0"),
    )
    assert snapshot.equity == (
        snapshot.cash
        + snapshot.long_market_value
        + snapshot.restricted_short_collateral
        + short_unrealized
    )
    assert snapshot.equity == snapshot.initial_cash + snapshot.net_pnl


def test_opening_long_or_short_at_mark_creates_no_economic_pnl() -> None:
    long = Portfolio("10000")
    long.apply_fill(fill("BTC/USD", Side.BUY, "10", "100"))
    short = Portfolio("10000")
    short.apply_fill(fill("BTC/USD", Side.SELL, "10", "100", collateral="1000"))

    assert long.snapshot(T0).equity == Decimal("10000")
    assert short.snapshot(T0).equity == Decimal("10000")
    assert long.snapshot(T0).unrealized_pnl == Decimal("0")
    assert short.snapshot(T0).unrealized_pnl == Decimal("0")
    assert_equity_identity(long)
    assert_equity_identity(short)


def test_long_and_short_pnl_are_symmetric_for_equal_notional() -> None:
    portfolio = Portfolio("10000")
    portfolio.apply_fill(fill("LONG/USD", Side.BUY, "10", "100"))
    portfolio.apply_fill(fill("SHORT/USD", Side.SELL, "10", "100", collateral="1000"))
    portfolio.mark_many({"LONG/USD": "110", "SHORT/USD": "110"}, T0)

    positions = portfolio.positions
    assert positions["LONG/USD"].unrealized_pnl == Decimal("100")
    assert positions["SHORT/USD"].unrealized_pnl == Decimal("-100")
    assert portfolio.snapshot(T0).unrealized_pnl == Decimal("0")
    assert_equity_identity(portfolio)


def test_full_closes_move_pnl_to_realized_and_preserve_equity() -> None:
    portfolio = Portfolio("10000")
    portfolio.apply_fill(fill("BTC/USD", Side.BUY, "10", "100"))
    portfolio.mark("BTC/USD", "120", T0)
    before = portfolio.snapshot(T0)

    portfolio.apply_fill(fill("BTC/USD", Side.SELL, "10", "120", hours=1))
    after = portfolio.snapshot(T0 + timedelta(hours=1))

    assert before.equity == after.equity == Decimal("10200")
    assert after.positions_by_symbol["BTC/USD"].quantity == Decimal("0")
    assert after.unrealized_pnl == Decimal("0")
    assert after.realized_gross_pnl == Decimal("200")
    assert_equity_identity(portfolio)


def test_full_short_cover_moves_pnl_to_realized_and_releases_collateral() -> None:
    portfolio = Portfolio("10000")
    portfolio.apply_fill(fill("BTC/USD", Side.SELL, "10", "100", collateral="1000"))
    portfolio.mark("BTC/USD", "80", T0)
    before = portfolio.snapshot(T0)

    portfolio.apply_fill(fill("BTC/USD", Side.BUY, "10", "80", hours=1))
    after = portfolio.snapshot(T0 + timedelta(hours=1))

    assert before.equity == after.equity == Decimal("10200")
    assert after.positions_by_symbol["BTC/USD"].quantity == Decimal("0")
    assert after.restricted_short_collateral == Decimal("0")
    assert after.unrealized_pnl == Decimal("0")
    assert after.realized_gross_pnl == Decimal("200")
    assert_equity_identity(portfolio)


def test_fees_reduce_equity_exactly_once() -> None:
    portfolio = Portfolio("10000")
    portfolio.apply_fill(fill("BTC/USD", Side.BUY, "10", "100", fee="2.50"))
    opened = portfolio.snapshot(T0)

    portfolio.apply_fill(fill("BTC/USD", Side.SELL, "10", "100", fee="1.50", hours=1))
    closed = portfolio.snapshot(T0 + timedelta(hours=1))

    assert opened.equity == Decimal("9997.50")
    assert closed.equity == Decimal("9996.00")
    assert closed.cumulative_fees == Decimal("4.00")
    assert closed.net_pnl == Decimal("-4.00")
    assert_equity_identity(portfolio)


def test_financing_charge_and_credit_affect_equity_exactly_once() -> None:
    portfolio = Portfolio("10000")
    portfolio.apply_fill(fill("ETH/USD", Side.SELL, "10", "100", collateral="1000"))
    portfolio.apply_financing(
        FinancingEntry("-5", T0 + timedelta(hours=1), "short_borrow", symbol="ETH/USD")
    )
    charged = portfolio.snapshot(T0 + timedelta(hours=1))
    portfolio.apply_financing(
        FinancingEntry("2", T0 + timedelta(hours=2), "funding_credit", symbol="ETH/USD")
    )
    credited = portfolio.snapshot(T0 + timedelta(hours=2))

    assert charged.equity == Decimal("9995")
    assert credited.equity == Decimal("9997")
    assert credited.financing == Decimal("-3")
    assert credited.net_pnl == Decimal("-3")
    assert_equity_identity(portfolio)


def test_concrete_long_btc_short_eth_scenario_and_partial_closes() -> None:
    portfolio = Portfolio("100000")
    portfolio.apply_fill(fill("BTC/USD", Side.BUY, "1", "50000", group="pair-1"))
    portfolio.apply_fill(
        fill(
            "ETH/USD",
            Side.SELL,
            "10",
            "3000",
            collateral="30000",
            group="pair-1",
        )
    )
    portfolio.mark_many({"BTC/USD": "52000", "ETH/USD": "2800"}, T0)

    marked = portfolio.snapshot(T0)
    positions = marked.positions_by_symbol
    assert positions["BTC/USD"].unrealized_pnl == Decimal("2000")
    assert positions["ETH/USD"].unrealized_pnl == Decimal("2000")
    assert marked.long_market_value == Decimal("52000")
    assert marked.short_market_value == Decimal("-28000")
    assert marked.short_exposure == Decimal("28000")
    assert marked.gross_market_value == Decimal("80000")
    assert marked.net_market_value == Decimal("24000")
    assert marked.equity == Decimal("104000")
    assert marked.nav == marked.equity
    assert marked.gross_leverage == Decimal("80000") / Decimal("104000")
    assert {event.trade_group_id for event in portfolio.fills} == {"pair-1"}
    assert_equity_identity(portfolio)

    portfolio.apply_fill(fill("BTC/USD", Side.SELL, "0.5", "52000", group="pair-1", hours=1))
    portfolio.apply_fill(fill("ETH/USD", Side.BUY, "4", "2800", group="pair-1", hours=1))
    partial = portfolio.snapshot(T0 + timedelta(hours=1))

    assert partial.positions_by_symbol["BTC/USD"].quantity == Decimal("0.5")
    assert partial.positions_by_symbol["BTC/USD"].average_entry_price == Decimal("50000")
    assert partial.positions_by_symbol["ETH/USD"].quantity == Decimal("-6")
    assert partial.positions_by_symbol["ETH/USD"].average_entry_price == Decimal("3000")
    assert partial.realized_gross_pnl == Decimal("1800")
    assert partial.unrealized_pnl == Decimal("2200.0")
    assert partial.equity == Decimal("104000.0")
    assert partial.long_market_value == Decimal("26000.0")
    assert partial.short_market_value == Decimal("-16800")
    assert partial.gross_market_value == Decimal("42800.0")
    assert partial.net_market_value == Decimal("9200.0")
    assert_equity_identity(portfolio)


def test_reversal_cash_accounting_matches_close_then_reopen() -> None:
    one_fill = Portfolio("10000")
    one_fill.apply_fill(fill("BTC/USD", Side.BUY, "10", "100"))
    one_fill.apply_fill(fill("BTC/USD", Side.SELL, "15", "110", collateral="550", hours=1))

    split = Portfolio("10000")
    split.apply_fill(fill("BTC/USD", Side.BUY, "10", "100"))
    split.apply_fill(fill("BTC/USD", Side.SELL, "10", "110", hours=1))
    split.apply_fill(fill("BTC/USD", Side.SELL, "5", "110", collateral="550", hours=1))

    assert one_fill.cash == split.cash
    assert (
        one_fill.snapshot(T0 + timedelta(hours=1)).equity
        == split.snapshot(T0 + timedelta(hours=1)).equity
    )
    assert one_fill.position("BTC/USD") == split.position("BTC/USD")
    assert_equity_identity(one_fill)


def test_short_to_long_reversal_cash_matches_cover_then_reopen() -> None:
    one_fill = Portfolio("10000")
    one_fill.apply_fill(fill("BTC/USD", Side.SELL, "10", "100", collateral="1000"))
    one_fill.apply_fill(fill("BTC/USD", Side.BUY, "15", "90", hours=1))

    split = Portfolio("10000")
    split.apply_fill(fill("BTC/USD", Side.SELL, "10", "100", collateral="1000"))
    split.apply_fill(fill("BTC/USD", Side.BUY, "10", "90", hours=1))
    split.apply_fill(fill("BTC/USD", Side.BUY, "5", "90", hours=1))

    assert one_fill.cash == split.cash == Decimal("9650")
    assert (
        one_fill.snapshot(T0 + timedelta(hours=1)).equity
        == split.snapshot(T0 + timedelta(hours=1)).equity
    )
    assert one_fill.position("BTC/USD") == split.position("BTC/USD")
    assert_equity_identity(one_fill)


def test_non_base_currency_fee_requires_explicit_conversion() -> None:
    portfolio = Portfolio("10000")
    event = Fill(
        "BTC/USD",
        Side.BUY,
        "1",
        "100",
        T0,
        fee="0.01",
        fee_currency="BTC",
    )

    with pytest.raises(ValueError, match="explicit conversion"):
        portfolio.apply_fill(event)
    assert portfolio.positions == {}
