"""Atomic raw/derived CSV storage and lightweight dataset manifests."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from stat_arb_bot.domain import Candle
from stat_arb_bot.market_data import CandleStore
from stat_arb_bot.research_data.pipeline import ResearchDataset
from stat_arb_bot.research_data.quality import RawSeries, merge_raw_series
from stat_arb_bot.research_data.universe import ResearchUniverse


@dataclass(frozen=True, slots=True)
class DatasetPaths:
    raw_paths: tuple[Path, ...]
    derived_paths: tuple[Path, ...]
    manifest_path: Path


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


class ResearchDataStore:
    """Keep canonical 5m source data distinct from reproducible 15m output."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _raw_store(self, universe: ResearchUniverse) -> CandleStore:
        return CandleStore(self.root / "raw" / universe.quote_currency)

    def _derived_store(self, universe: ResearchUniverse) -> CandleStore:
        return CandleStore(self.root / "derived" / universe.quote_currency)

    def load_raw(self, universe: ResearchUniverse) -> dict[str, tuple[Candle, ...]]:
        store = self._raw_store(universe)
        return {
            asset: tuple(store.read_csv(universe.pair(asset), "5m")) for asset in universe.assets
        }

    def append_raw(
        self,
        universe: ResearchUniverse,
        asset: str,
        candles: Iterable[Candle],
        *,
        as_of: datetime,
    ) -> RawSeries:
        pair = universe.pair(asset)
        store = self._raw_store(universe)
        merged = merge_raw_series(
            store.read_csv(pair, "5m"),
            candles,
            symbol=pair,
            as_of=as_of,
        )
        store.write_csv(pair, "5m", list(merged.candles))
        return merged

    def write_dataset(self, dataset: ResearchDataset) -> DatasetPaths:
        raw_store = self._raw_store(dataset.universe)
        derived_store = self._derived_store(dataset.universe)
        raw_paths: list[Path] = []
        derived_paths: list[Path] = []
        for asset in dataset.universe.assets:
            pair = dataset.universe.pair(asset)
            raw_paths.append(raw_store.write_csv(pair, "5m", list(dataset.raw_by_asset[asset])))
            derived_paths.append(
                derived_store.write_csv(
                    pair,
                    "15m",
                    list(dataset.aggregation_by_asset[asset].bars),
                )
            )
        manifest = {
            "schema_version": 1,
            "source": "BINANCE",
            "assets": dataset.universe.assets,
            "pairs": dataset.universe.pairs,
            "binance_symbols": dict(dataset.universe.binance_symbols),
            "quote_currency": dataset.universe.quote_currency,
            "raw_interval": "5m",
            "model_interval": "15m",
            "requested_start": dataset.requested_start,
            "requested_end": dataset.requested_end,
            "source_fingerprint": dataset.source_fingerprint,
            "panel_fingerprint": dataset.panel.fingerprint,
            "raw_rows": {
                asset: len(dataset.raw_by_asset[asset]) for asset in dataset.universe.assets
            },
            "model_rows": {
                asset: len(dataset.aggregation_by_asset[asset].bars)
                for asset in dataset.universe.assets
            },
            "synchronized_rows": len(dataset.panel.rows),
            "panel_start": dataset.panel.start,
            "panel_end": dataset.panel.end,
            "affected_model_buckets": {
                asset: list(dataset.affected_model_buckets[asset])
                for asset in dataset.universe.assets
            },
        }
        manifest_path = self._write_manifest(dataset, manifest)
        return DatasetPaths(tuple(raw_paths), tuple(derived_paths), manifest_path)

    def _write_manifest(self, dataset: ResearchDataset, manifest: dict[str, Any]) -> Path:
        directory = self.root / "manifests" / dataset.universe.quote_currency
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{dataset.source_fingerprint}.json"
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=directory,
                prefix=f".{path.name}.",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(
                    manifest,
                    handle,
                    default=_json_value,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return path
