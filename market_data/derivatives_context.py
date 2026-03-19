import requests
from typing import Dict, Any, Optional


BINANCE_FAPI_BASE = "https://fapi.binance.com"


def _safe_get(url: str, params: dict) -> Optional[Any]:
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def fetch_current_open_interest(symbol: str = "BTCUSDT") -> Optional[float]:
    data = _safe_get(
        f"{BINANCE_FAPI_BASE}/fapi/v1/openInterest",
        {"symbol": symbol.upper()}
    )
    if not data:
        return None

    try:
        return float(data["openInterest"])
    except Exception:
        return None


def fetch_open_interest_stats(symbol: str = "BTCUSDT", period: str = "15m", limit: int = 2) -> list[dict]:
    data = _safe_get(
        f"{BINANCE_FAPI_BASE}/futures/data/openInterestHist",
        {
            "symbol": symbol.upper(),
            "period": period,
            "limit": limit,
        }
    )
    if not data or not isinstance(data, list):
        return []

    out = []
    for item in data:
        try:
            out.append({
                "sumOpenInterest": float(item["sumOpenInterest"]),
                "sumOpenInterestValue": float(item["sumOpenInterestValue"]),
                "timestamp": int(item["timestamp"]),
            })
        except Exception:
            continue
    return out


def fetch_latest_funding(symbol: str = "BTCUSDT") -> Optional[float]:
    data = _safe_get(
        f"{BINANCE_FAPI_BASE}/fapi/v1/fundingRate",
        {
            "symbol": symbol.upper(),
            "limit": 1,
        }
    )
    if not data or not isinstance(data, list):
        return None

    try:
        return float(data[-1]["fundingRate"])
    except Exception:
        return None


def fetch_mark_price_and_funding(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    data = _safe_get(
        f"{BINANCE_FAPI_BASE}/fapi/v1/premiumIndex",
        {"symbol": symbol.upper()}
    )
    if not data or not isinstance(data, dict):
        return {}

    out = {}
    try:
        out["markPrice"] = float(data["markPrice"])
    except Exception:
        pass

    try:
        out["lastFundingRate"] = float(data["lastFundingRate"])
    except Exception:
        pass

    try:
        out["nextFundingTime"] = int(data["nextFundingTime"])
    except Exception:
        pass

    return out


def build_derivatives_context(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Free derivatives context from Binance:
    - current open interest
    - recent OI change
    - current/last funding
    - simple crowding label
    """
    oi_now = fetch_current_open_interest(symbol)
    oi_hist = fetch_open_interest_stats(symbol, period="15m", limit=2)
    funding = fetch_latest_funding(symbol)
    mark_info = fetch_mark_price_and_funding(symbol)

    oi_change_pct = None
    if len(oi_hist) >= 2:
        prev_oi = oi_hist[-2]["sumOpenInterest"]
        last_oi = oi_hist[-1]["sumOpenInterest"]
        if prev_oi > 0:
            oi_change_pct = (last_oi - prev_oi) / prev_oi

    crowding = "neutral"
    if funding is not None and oi_change_pct is not None:
        # heuristic:
        # funding positive + OI rising = crowded longs
        # funding negative + OI rising = crowded shorts
        if funding > 0 and oi_change_pct > 0.01:
            crowding = "crowded_longs"
        elif funding < 0 and oi_change_pct > 0.01:
            crowding = "crowded_shorts"
        elif abs(funding) < 0.00005 and abs(oi_change_pct) < 0.005:
            crowding = "neutral"

    return {
        "open_interest_now": oi_now,
        "open_interest_change_pct_15m": oi_change_pct,
        "funding_rate": funding if funding is not None else mark_info.get("lastFundingRate"),
        "mark_price": mark_info.get("markPrice"),
        "next_funding_time": mark_info.get("nextFundingTime"),
        "crowding": crowding,
        "source": "binance_free"
    }