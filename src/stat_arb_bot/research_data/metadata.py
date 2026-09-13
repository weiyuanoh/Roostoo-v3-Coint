"""Research dataset fingerprints and future model-fit chronology metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

from stat_arb_bot.domain import Candle
from stat_arb_bot.domain.execution import utc_datetime
from stat_arb_bot.research_data.universe import ResearchUniverse
from stat_arb_bot.research_data.windows import ResearchWindow


def fingerprint_raw_candles(
    raw_by_asset: Mapping[str, Sequence[Candle]],
    universe: ResearchUniverse,
) -> str:
    """Hash canonical market content, intentionally excluding retrieval time."""

    rows = []
    for asset in universe.assets:
        for candle in sorted(raw_by_asset.get(asset, ()), key=lambda item: item.open_time):
            rows.append(
                (
                    asset,
                    candle.symbol,
                    candle.interval,
                    candle.open_time.isoformat(),
                    candle.close_time.isoformat(),
                    float(candle.open).hex(),
                    float(candle.high).hex(),
                    float(candle.low).hex(),
                    float(candle.close).hex(),
                    float(candle.volume).hex(),
                    candle.source,
                )
            )
    payload = {
        "assets": universe.assets,
        "quote_currency": universe.quote_currency,
        "raw_interval": "5m",
        "rows": rows,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelFitMetadata:
    """Chronology and provenance for a future fitted model, not a model itself."""

    model_name: str
    model_version: str
    fit_id: str
    fit_timestamp: datetime
    data_start_timestamp: datetime
    data_end_timestamp: datetime
    bar_interval: str
    universe: tuple[str, ...]
    quote_currency: str
    observation_count: int
    source_data_fingerprint: str

    def __post_init__(self) -> None:
        names = ("model_name", "model_version", "fit_id", "bar_interval")
        for name in names:
            value = getattr(self, name).strip()
            if not value:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, value)
        fit_time = utc_datetime(self.fit_timestamp, name="fit_timestamp")
        start = utc_datetime(self.data_start_timestamp, name="data_start_timestamp")
        end = utc_datetime(self.data_end_timestamp, name="data_end_timestamp")
        if start > end:
            raise ValueError("model data start must not follow data end")
        if end > fit_time:
            raise ValueError("model fit cannot precede the end of its input data")
        if self.observation_count <= 0:
            raise ValueError("observation_count must be positive")
        quote = self.quote_currency.strip().upper()
        if not quote:
            raise ValueError("quote_currency must not be empty")
        normalized_universe = tuple(asset.strip().upper() for asset in self.universe)
        if len(set(normalized_universe)) != len(normalized_universe):
            raise ValueError("model universe contains duplicate assets")
        fingerprint = self.source_data_fingerprint.strip().lower()
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise ValueError("source_data_fingerprint must be a SHA-256 hex digest")
        object.__setattr__(self, "fit_timestamp", fit_time)
        object.__setattr__(self, "data_start_timestamp", start)
        object.__setattr__(self, "data_end_timestamp", end)
        object.__setattr__(self, "quote_currency", quote)
        object.__setattr__(self, "universe", normalized_universe)
        object.__setattr__(self, "source_data_fingerprint", fingerprint)

    @classmethod
    def from_window(
        cls,
        window: ResearchWindow,
        *,
        model_name: str,
        model_version: str,
        fit_id: str,
        fit_timestamp: datetime,
        source_data_fingerprint: str,
        bar_interval: str = "15m",
    ) -> ModelFitMetadata:
        if not window.validation.valid or window.data_start is None or window.data_end is None:
            raise ValueError("cannot describe a model fit from an invalid research window")
        return cls(
            model_name=model_name,
            model_version=model_version,
            fit_id=fit_id,
            fit_timestamp=fit_timestamp,
            data_start_timestamp=window.data_start,
            data_end_timestamp=window.data_end,
            bar_interval=bar_interval,
            universe=window.universe_assets,
            quote_currency=window.quote_currency,
            observation_count=window.actual_observations,
            source_data_fingerprint=source_data_fingerprint,
        )

    def order_chronology_fields(self) -> dict[str, object]:
        return {
            "model_version": self.model_version,
            "model_fit_end_timestamp": self.data_end_timestamp,
            "metadata": {
                "model_name": self.model_name,
                "model_fit_id": self.fit_id,
                "model_fit_timestamp": self.fit_timestamp.isoformat(),
                "source_data_fingerprint": self.source_data_fingerprint,
            },
        }
