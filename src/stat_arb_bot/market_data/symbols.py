"""Roostoo pair to Binance public-spot symbol mapping.

The mapping is infrastructure inventory, not the V2 deployment universe.
"""

BINANCE_SYMBOL_MAP: dict[str, str] = {
    "BTC/USD": "BTCUSDT",
    "ETH/USD": "ETHUSDT",
    "SOL/USD": "SOLUSDT",
    "BNB/USD": "BNBUSDT",
    "XRP/USD": "XRPUSDT",
    "DOGE/USD": "DOGEUSDT",
    "ADA/USD": "ADAUSDT",
    "AVAX/USD": "AVAXUSDT",
    "LINK/USD": "LINKUSDT",
    "DOT/USD": "DOTUSDT",
    "SUI/USD": "SUIUSDT",
    "NEAR/USD": "NEARUSDT",
    "LTC/USD": "LTCUSDT",
    "TON/USD": "TONUSDT",
    "UNI/USD": "UNIUSDT",
    "FET/USD": "FETUSDT",
    "HBAR/USD": "HBARUSDT",
    "XLM/USD": "XLMUSDT",
    "FIL/USD": "FILUSDT",
    "APT/USD": "APTUSDT",
    "ARB/USD": "ARBUSDT",
    "SEI/USD": "SEIUSDT",
    "PEPE/USD": "PEPEUSDT",
    "SHIB/USD": "SHIBUSDT",
    "FLOKI/USD": "FLOKIUSDT",
    "WIF/USD": "WIFUSDT",
    "BONK/USD": "BONKUSDT",
    "TRX/USD": "TRXUSDT",
    "ICP/USD": "ICPUSDT",
    "AAVE/USD": "AAVEUSDT",
    "WLD/USD": "WLDUSDT",
    "ONDO/USD": "ONDOUSDT",
    "CRV/USD": "CRVUSDT",
    "PENDLE/USD": "PENDLEUSDT",
    "ENA/USD": "ENAUSDT",
    "TAO/USD": "TAOUSDT",
    "POL/USD": "POLUSDT",
    "ZEC/USD": "ZECUSDT",
    "TRUMP/USD": "TRUMPUSDT",
    "EIGEN/USD": "EIGENUSDT",
    "VIRTUAL/USD": "VIRTUALUSDT",
    "CAKE/USD": "CAKEUSDT",
    "PAXG/USD": "PAXGUSDT",
}

RESEARCH_ASSETS: tuple[str, ...] = ("BTC", "ETH", "SOL", "XRP", "ADA")


def binance_symbol_for_pair(pair: str) -> str:
    """Map a normalized pair to a Binance spot symbol.

    The legacy Roostoo `/USD` aliases remain stable. Research pairs use their
    explicit configured quote, with the five-asset universe kept fixed.
    """

    normalized = pair.strip().upper()
    if normalized in BINANCE_SYMBOL_MAP:
        return BINANCE_SYMBOL_MAP[normalized]
    parts = normalized.split("/")
    if len(parts) == 2 and parts[0] in RESEARCH_ASSETS and parts[1].isalnum():
        return "".join(parts)
    raise KeyError(normalized)
