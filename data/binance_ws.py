import asyncio
import json
from typing import Callable, Awaitable, Dict, Any, Optional, List

import websockets


def build_stream_url(symbol: str) -> str:
    sym = symbol.lower()
    streams = [
        f"{sym}@kline_15m",
        f"{sym}@kline_1h",
        f"{sym}@kline_4h",
        f"{sym}@depth20@100ms",   # partial book (often uses bids/asks, no 'e')
    ]
    return "wss://stream.binance.com:9443/stream?streams=" + "/".join(streams)


TimeframeMap = {
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
}


async def run_ws(
    symbol: str,
    on_closed_kline: Callable[[str, Dict[str, Any]], Awaitable[None]],
    on_depth: Optional[Callable[[List[List[str]], List[List[str]]], Awaitable[None]]] = None,
) -> None:
    """
    Binance combined stream listener.
    - Calls on_closed_kline(tf, k) ONLY when a kline is closed (x=True).
    - Calls on_depth(bids, asks) when a depth snapshot/update is received.
      Supports BOTH formats:
        - partial book: {bids: [...], asks: [...]}
        - diff depth: {e: depthUpdate, b: [...], a: [...]}
    """
    url = build_stream_url(symbol)

    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                print(f"[WS] connected: {url}")

                async for msg in ws:
                    payload = json.loads(msg)
                    data = payload.get("data", {})

                    # --- DEPTH (partial book stream) ---
                    # depth20@100ms commonly sends: {"lastUpdateId":..., "bids":[...], "asks":[...]}
                    if on_depth is not None:
                        if "bids" in data and "asks" in data:
                            await on_depth(data.get("bids", []), data.get("asks", []))
                            continue

                        # diff depth format (other streams): {"e":"depthUpdate","b":[...],"a":[...]}
                        if data.get("e") == "depthUpdate":
                            await on_depth(data.get("b", []), data.get("a", []))
                            continue

                    # --- KLINE ---
                    k = data.get("k")
                    if not k:
                        continue

                    interval = k.get("i")
                    is_closed = bool(k.get("x", False))
                    if not is_closed:
                        continue

                    tf = TimeframeMap.get(interval)
                    if tf is None:
                        continue

                    await on_closed_kline(tf, k)

        except Exception as e:
            print(f"[WS] error: {e} -> reconnecting in 3s")
            await asyncio.sleep(3)