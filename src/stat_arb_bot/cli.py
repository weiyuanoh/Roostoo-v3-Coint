"""Read-only operational commands for infrastructure verification and data collection."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stat_arb_bot.config import Settings, SettingsError, load_settings
from stat_arb_bot.domain import utc_to_epoch_ms
from stat_arb_bot.exchange.roostoo import RoostooClient
from stat_arb_bot.exchange.roostoo.errors import RoostooError
from stat_arb_bot.market_data import BinanceData, CandleStore
from stat_arb_bot.market_data.binance import BinanceDataError
from stat_arb_bot.market_data.validation import (
    CandleQualityError,
    inspect_candles,
    require_candle_quality,
)
from stat_arb_bot.observability.logging import configure_logging, get_logger

log = get_logger("cli")


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _print_json(value: Any) -> None:
    print(json.dumps(value, default=_json_default, indent=2, sort_keys=True))


def _epoch_ms(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "timestamp must be epoch milliseconds or an ISO-8601 datetime"
        ) from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("ISO-8601 timestamps must include a timezone")
    return utc_to_epoch_ms(parsed)


def _roostoo(settings: Settings) -> RoostooClient:
    return RoostooClient(
        api_key=settings.roostoo_api_key,
        api_secret=settings.roostoo_api_secret,
        base_url=settings.roostoo_base_url,
        timeout=settings.request_timeout_seconds,
        max_attempts=settings.http_max_attempts,
        backoff_seconds=settings.http_backoff_seconds,
    )


def _binance(settings: Settings) -> BinanceData:
    return BinanceData(
        base_urls=settings.binance_base_urls,
        timeout=settings.request_timeout_seconds,
        max_attempts=settings.http_max_attempts,
        backoff_seconds=settings.http_backoff_seconds,
    )


def _cmd_server_time(settings: Settings, _args: argparse.Namespace) -> None:
    _print_json({"server_time": _roostoo(settings).server_time()})


def _cmd_exchange_info(settings: Settings, _args: argparse.Namespace) -> None:
    _print_json(_roostoo(settings).exchange_info())


def _cmd_ticker(settings: Settings, args: argparse.Namespace) -> None:
    _print_json(_roostoo(settings).ticker(args.pair))


def _cmd_balance(settings: Settings, _args: argparse.Namespace) -> None:
    settings.require_roostoo_credentials()
    _print_json(_roostoo(settings).balance())


def _cmd_pending_count(settings: Settings, _args: argparse.Namespace) -> None:
    settings.require_roostoo_credentials()
    _print_json(_roostoo(settings).pending_count())


def _cmd_orders(settings: Settings, args: argparse.Namespace) -> None:
    settings.require_roostoo_credentials()
    _print_json(
        _roostoo(settings).query_order(
            order_id=args.order_id,
            pair=args.pair,
            pending_only=args.pending_only,
            limit=args.limit,
        )
    )


def _cmd_short_positions(settings: Settings, _args: argparse.Namespace) -> None:
    settings.require_roostoo_credentials()
    _print_json(_roostoo(settings).short_positions())


def _cmd_collect(settings: Settings, args: argparse.Namespace) -> None:
    client = _binance(settings)
    store = CandleStore(settings.data_dir)
    results: list[dict[str, Any]] = []
    for pair in args.pairs:
        if args.start_time is not None and args.end_time is not None:
            fetched = client.fetch_klines_paginated(
                pair,
                interval=args.interval,
                start_time=args.start_time,
                end_time=args.end_time,
                limit=args.limit,
                sleep_seconds=args.page_delay,
            )
        else:
            fetched = client.fetch_klines(
                pair,
                interval=args.interval,
                limit=args.limit,
                start_time=args.start_time,
                end_time=args.end_time,
            )
        if not fetched:
            raise BinanceDataError(f"Binance returned no closed candles for {pair.upper()}")
        existing = store.read_csv(pair, args.interval)
        merged = {candle.open_time: candle for candle in [*existing, *fetched]}
        candles = [merged[open_time] for open_time in sorted(merged)]
        report = require_candle_quality(
            candles, interval=args.interval, expected_symbol=pair.upper()
        )
        path = store.write_csv(pair, args.interval, candles)
        results.append(
            {
                "pair": pair.upper(),
                "fetched": len(fetched),
                "stored": report.candles,
                "path": path,
            }
        )
    _print_json(results)


def _cmd_validate(settings: Settings, args: argparse.Namespace) -> None:
    store = CandleStore(settings.data_dir)
    results: list[dict[str, Any]] = []
    failed = False
    for pair in args.pairs:
        candles = store.read_csv(pair, args.interval)
        report = inspect_candles(
            candles,
            interval=args.interval,
            expected_symbol=pair.upper(),
            max_age_intervals=args.max_age_intervals,
        )
        results.append({"pair": pair.upper(), **asdict(report)})
        failed = failed or not report.valid
    _print_json(results)
    if failed:
        raise CandleQualityError("one or more stored candle series failed validation")


def _cmd_smoke(settings: Settings, args: argparse.Namespace) -> None:
    roostoo = _roostoo(settings)
    exchange_info = roostoo.exchange_info()
    candles = _binance(settings).fetch_klines(args.pair, interval=args.interval, limit=3)
    if not candles:
        raise BinanceDataError(f"Binance returned no closed candles for {args.pair.upper()}")
    _print_json(
        {
            "roostoo": {
                "server_time": roostoo.server_time(),
                "exchange_info_received": bool(exchange_info),
            },
            "binance": {
                "pair": args.pair.upper(),
                "interval": args.interval,
                "closed_candles": len(candles),
                "latest_close_time": candles[-1].close_time,
            },
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stat-arb-bot",
        description=(
            "Read-only Roostoo inspection and Binance candle collection. "
            "This checkpoint exposes no trade mutation commands."
        ),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="optional dotenv file; process environment variables take precedence",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    server_time = subparsers.add_parser("server-time", help="query Roostoo server time")
    server_time.set_defaults(handler=_cmd_server_time)

    exchange_info = subparsers.add_parser("exchange-info", help="query Roostoo exchange metadata")
    exchange_info.set_defaults(handler=_cmd_exchange_info)

    ticker = subparsers.add_parser("ticker", help="query Roostoo public ticker data")
    ticker.add_argument("--pair", help="optional pair such as BTC/USD")
    ticker.set_defaults(handler=_cmd_ticker)

    balance = subparsers.add_parser("balance", help="query the signed Roostoo wallet")
    balance.set_defaults(handler=_cmd_balance)

    pending_count = subparsers.add_parser(
        "pending-count", help="query the signed Roostoo pending-order count"
    )
    pending_count.set_defaults(handler=_cmd_pending_count)

    orders = subparsers.add_parser("orders", help="query Roostoo order history")
    orders.add_argument("--order-id", type=int)
    orders.add_argument("--pair")
    orders.add_argument("--pending-only", action=argparse.BooleanOptionalAction, default=None)
    orders.add_argument("--limit", type=int)
    orders.set_defaults(handler=_cmd_orders)

    short_positions = subparsers.add_parser(
        "short-positions", help="query documented v6 open short positions"
    )
    short_positions.set_defaults(handler=_cmd_short_positions)

    collect = subparsers.add_parser(
        "collect", help="download, validate, and persist closed Binance candles"
    )
    collect.add_argument("pairs", nargs="+", help="Roostoo pairs such as BTC/USD ETH/USD")
    collect.add_argument("--interval", default="1h")
    collect.add_argument("--limit", type=int, default=1000)
    collect.add_argument("--start-time", type=_epoch_ms)
    collect.add_argument("--end-time", type=_epoch_ms)
    collect.add_argument("--page-delay", type=float, default=0.1)
    collect.set_defaults(handler=_cmd_collect)

    validate = subparsers.add_parser("validate-candles", help="validate stored candle CSVs")
    validate.add_argument("pairs", nargs="+")
    validate.add_argument("--interval", default="1h")
    validate.add_argument("--max-age-intervals", type=int)
    validate.set_defaults(handler=_cmd_validate)

    smoke = subparsers.add_parser(
        "smoke", help="exercise public Roostoo and Binance connectivity without credentials"
    )
    smoke.add_argument("--pair", default="BTC/USD")
    smoke.add_argument("--interval", default="1h")
    smoke.set_defaults(handler=_cmd_smoke)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = load_settings(env_file=args.env_file)
        configure_logging(
            level=settings.log_level,
            console=settings.log_console,
            log_dir=settings.log_dir if settings.log_file else None,
        )
        args.handler(settings, args)
        return 0
    except (SettingsError, RoostooError, BinanceDataError, CandleQualityError, ValueError) as exc:
        log.error("%s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
