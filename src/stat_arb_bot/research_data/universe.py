"""Canonical five-asset research universe and exchange-symbol boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from stat_arb_bot.market_data.symbols import RESEARCH_ASSETS, binance_symbol_for_pair


@dataclass(frozen=True, slots=True)
class ResearchUniverse:
    """The fixed initial asset universe under one configurable quote."""

    quote_currency: str = "USDT"
    assets: tuple[str, ...] = field(default=RESEARCH_ASSETS, init=False)

    def __post_init__(self) -> None:
        quote = self.quote_currency.strip().upper()
        if not quote or not quote.isalnum():
            raise ValueError("quote_currency must be alphanumeric")
        object.__setattr__(self, "quote_currency", quote)

    def pair(self, asset: str) -> str:
        normalized = asset.strip().upper()
        if normalized not in self.assets:
            raise ValueError(f"asset {normalized!r} is outside the research universe")
        return f"{normalized}/{self.quote_currency}"

    @property
    def pairs(self) -> tuple[str, ...]:
        return tuple(self.pair(asset) for asset in self.assets)

    @property
    def binance_symbols(self) -> Mapping[str, str]:
        return MappingProxyType(
            {asset: binance_symbol_for_pair(self.pair(asset)) for asset in self.assets}
        )

    def asset_for_pair(self, pair: str) -> str:
        normalized = pair.strip().upper()
        try:
            asset, quote = normalized.split("/", maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"invalid research pair {pair!r}") from exc
        if quote != self.quote_currency or asset not in self.assets:
            raise ValueError(f"pair {normalized} does not belong to this common-quote universe")
        return asset
