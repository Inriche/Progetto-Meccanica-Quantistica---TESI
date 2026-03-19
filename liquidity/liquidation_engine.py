import requests
from typing import Dict, Any, List, Optional

from config import CONFIG


LEGACY_URL = "https://open-api.coinglass.com/public/v2/liqMap"
V3_URL = "https://open-api-v3.coinglass.com/api/futures/liquidation/map"
V4_URL = "https://open-api-v4.coinglass.com/api/futures/liquidation/map"


def _normalize_clusters(raw_items: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    clusters = []

    for item in raw_items:
        price = (
            item.get("price")
            or item.get("p")
            or item.get("level")
            or item.get("liqPrice")
        )
        size = (
            item.get("liquidationSize")
            or item.get("size")
            or item.get("value")
            or item.get("amount")
            or 0
        )

        try:
            price_f = float(price)
            size_f = float(size)
        except (TypeError, ValueError):
            continue

        if price_f > 0:
            clusters.append({
                "price": price_f,
                "size": size_f
            })

    clusters.sort(key=lambda x: x["size"], reverse=True)
    return clusters


def _try_legacy(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(
            LEGACY_URL,
            params={"symbol": symbol},
            timeout=10
        )
        print(f"[LIQ] legacy status={r.status_code}")

        if r.status_code != 200:
            return None

        data = r.json()
        print(f"[LIQ] legacy payload keys={list(data.keys()) if isinstance(data, dict) else type(data)}")

        raw_items = data.get("data", [])
        clusters = _normalize_clusters(raw_items)

        if not clusters:
            print("[LIQ] legacy no clusters after normalize")
            return None

        return {"clusters": clusters[:10], "source": "legacy_v2"}
    except Exception as e:
        print(f"[LIQ] legacy error: {e}")
        return None


def _try_v3_or_v4(url: str, symbol: str, api_key: str) -> Optional[Dict[str, Any]]:
    try:
        headers = {"CG-API-KEY": api_key}

        r = requests.get(
            url,
            params={"symbol": symbol},
            headers=headers,
            timeout=10
        )

        print(f"[LIQ] {url} status={r.status_code}")

        if r.status_code != 200:
            try:
                print(f"[LIQ] {url} body={r.text[:300]}")
            except Exception:
                pass
            return None

        data = r.json()
        print(f"[LIQ] {url} payload keys={list(data.keys()) if isinstance(data, dict) else type(data)}")

        raw_items = data.get("data", [])
        if isinstance(raw_items, dict):
            raw_items = (
                raw_items.get("items")
                or raw_items.get("clusters")
                or raw_items.get("levels")
                or []
            )

        clusters = _normalize_clusters(raw_items)
        if not clusters:
            print(f"[LIQ] {url} no clusters after normalize")
            return None

        return {"clusters": clusters[:10], "source": url}
    except Exception as e:
        print(f"[LIQ] {url} error: {e}")
        return None


def fetch_liquidation_map(symbol: str = "BTC") -> Dict[str, Any]:
    api_key = getattr(CONFIG, "coinglass_api_key", "") or ""

    if api_key:
        res = _try_v3_or_v4(V4_URL, symbol, api_key)
        if res:
            return res

        res = _try_v3_or_v4(V3_URL, symbol, api_key)
        if res:
            return res

    return {}