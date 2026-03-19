import os
from datetime import datetime
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

def save_snapshot(
    df_m15: pd.DataFrame,
    snapshot_dir: str,
    symbol: str,
    ts: datetime,
    entry: Optional[float],
    sl: Optional[float],
    tp1: Optional[float],
    tp2: Optional[float],
) -> str:
    os.makedirs(snapshot_dir, exist_ok=True)

    # Take last N candles for display
    view = df_m15.tail(120).copy()
    if view.empty:
        return ""

    # Create a simple candle-like plot (line close + high/low wicks)
    x = range(len(view))
    closes = view["close"].values
    highs = view["high"].values
    lows = view["low"].values

    plt.figure(figsize=(12, 6))
    plt.plot(x, closes)

    # draw wicks
    for i in range(len(view)):
        plt.vlines(i, lows[i], highs[i], linewidth=0.5)

    # horizontal levels
    for lvl, label in [(entry, "ENTRY"), (sl, "SL"), (tp1, "TP1"), (tp2, "TP2")]:
        if lvl is not None:
            plt.axhline(lvl, linestyle="--", linewidth=1)
            plt.text(len(view) - 1, lvl, f" {label}", va="center")

    plt.title(f"{symbol.upper()} M15 Snapshot @ {ts.strftime('%Y-%m-%d %H:%M:%S')}")
    plt.tight_layout()

    filename = f"{symbol.upper()}_{ts.strftime('%Y%m%d_%H%M%S')}.png"
    path = os.path.join(snapshot_dir, filename)
    plt.savefig(path, dpi=140)
    plt.close()
    return path