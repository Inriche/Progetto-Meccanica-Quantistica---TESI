import requests
from typing import List, Dict, Any

BINANCE_REST = "https://api.binance.com/api/v3/klines"

def fetch_klines(symbol: str, interval: str, limit: int = 500) -> List[Dict[str, Any]]:
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = requests.get(BINANCE_REST, params=params, timeout=15)
    r.raise_for_status()
    raw = r.json()

    out = []
    for k in raw:
        out.append({
            "t": k[0],   # open time
            "T": k[6],   # close time
            "o": k[1],
            "h": k[2],
            "l": k[3],
            "c": k[4],
            "v": k[5],
            "x": True,   # closed
            "i": interval
        })
    return out