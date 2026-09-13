"""Deterministic CSV persistence for normalized candles."""

from __future__ import annotations

import csv
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

from stat_arb_bot.domain.market import Candle


_FIELDS = (
    "symbol",
    "interval",
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "source",
    "retrieved_at",
)


class CandleStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, pair: str, interval: str) -> Path:
        normalized_pair = pair.upper()
        if re.fullmatch(r"[A-Z0-9]+/[A-Z0-9]+", normalized_pair) is None:
            raise ValueError(f"invalid pair: {pair!r}")
        if re.fullmatch(r"[A-Za-z0-9]+", interval) is None:
            raise ValueError(f"invalid interval: {interval!r}")
        return self.root / f"{normalized_pair.replace('/', '_')}_{interval}.csv"

    def write_csv(self, pair: str, interval: str, candles: list[Candle]) -> Path:
        path = self.path_for(pair, interval)
        mismatched = [
            candle
            for candle in candles
            if candle.symbol != pair.upper() or candle.interval != interval
        ]
        if mismatched:
            raise ValueError("all candles must match the requested pair and interval")
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized = {candle.open_time: candle for candle in candles}
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                writer = csv.DictWriter(handle, fieldnames=_FIELDS)
                writer.writeheader()
                for open_time in sorted(normalized):
                    writer.writerow(normalized[open_time].as_record())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return path

    def append_csv(self, pair: str, interval: str, candles: list[Candle]) -> Path:
        existing = self.read_csv(pair, interval)
        return self.write_csv(pair, interval, [*existing, *candles])

    def read_csv(self, pair: str, interval: str) -> list[Candle]:
        path = self.path_for(pair, interval)
        if not path.exists():
            return []
        candles: list[Candle] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                candles.append(
                    Candle(
                        symbol=str(row["symbol"]),
                        interval=str(row["interval"]),
                        open_time=datetime.fromisoformat(row["open_time"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                        close_time=datetime.fromisoformat(row["close_time"]),
                        source=row.get("source") or None,
                        retrieved_at=(
                            datetime.fromisoformat(row["retrieved_at"])
                            if row.get("retrieved_at")
                            else None
                        ),
                    )
                )
        return sorted(candles, key=lambda candle: candle.open_time)

    def read_many(self, pairs: list[str] | tuple[str, ...], interval: str) -> list[Candle]:
        candles = [candle for pair in pairs for candle in self.read_csv(pair, interval)]
        return sorted(candles, key=lambda candle: (candle.open_time, candle.symbol))
