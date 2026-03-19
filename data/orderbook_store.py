from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque
import time


@dataclass
class OrderBookState:
    ts_ms: int
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]


class OrderBookStore:
    def __init__(self) -> None:
        self.state: Optional[OrderBookState] = None
        self.imbalance_history = deque(maxlen=20)

    def update_from_binance_depth(self, bids, asks) -> None:
        b = [(float(p), float(q)) for p, q in bids]
        a = [(float(p), float(q)) for p, q in asks]

        self.state = OrderBookState(
            ts_ms=int(time.time() * 1000),
            bids=b,
            asks=a
        )

        imb = self._compute_raw_imbalance(top_n=10)
        if imb is not None:
            self.imbalance_history.append(imb)

    def _compute_raw_imbalance(self, top_n: int = 10) -> Optional[float]:
        if self.state is None:
            return None

        bids = self.state.bids[:top_n]
        asks = self.state.asks[:top_n]

        bid_notional = sum(p * q for p, q in bids)
        ask_notional = sum(p * q for p, q in asks)

        denom = bid_notional + ask_notional
        if denom <= 0:
            return None

        return (bid_notional - ask_notional) / denom

    def imbalance(self, top_n: int = 10) -> Optional[float]:
        if not self.imbalance_history:
            return None
        return sum(self.imbalance_history) / len(self.imbalance_history)

    def raw_imbalance(self, top_n: int = 10) -> Optional[float]:
        return self._compute_raw_imbalance(top_n=top_n)

    def age_ms(self) -> Optional[int]:
        if self.state is None:
            return None
        return int(time.time() * 1000) - self.state.ts_ms