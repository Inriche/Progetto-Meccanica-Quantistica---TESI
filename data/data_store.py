from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Any, Optional

import pandas as pd

@dataclass
class Candle:
    open_time: int
    close_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @staticmethod
    def from_binance_kline(k: Dict[str, Any]) -> "Candle":
        return Candle(
            open_time=int(k["t"]),
            close_time=int(k["T"]),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
        )

class DataStore:
    def __init__(self, maxlen_m15: int, maxlen_h1: int, maxlen_h4: int) -> None:
        self.buffers: Dict[str, Deque[Candle]] = {
            "15m": deque(maxlen=maxlen_m15),
            "1h": deque(maxlen=maxlen_h1),
            "4h": deque(maxlen=maxlen_h4),
        }

    def add_candle(self, timeframe: str, candle: Candle) -> None:
        buf = self.buffers[timeframe]
        # avoid duplicates by open_time
        if len(buf) > 0 and buf[-1].open_time == candle.open_time:
            buf[-1] = candle
        else:
            buf.append(candle)

    def to_df(self, timeframe: str) -> pd.DataFrame:
        buf = self.buffers[timeframe]
        if not buf:
            return pd.DataFrame(columns=["open_time","close_time","open","high","low","close","volume"])
        return pd.DataFrame([c.__dict__ for c in buf]).copy()

    def latest_close(self, timeframe: str) -> Optional[float]:
        buf = self.buffers[timeframe]
        return buf[-1].close if buf else None