import os
import sqlite3
from typing import Optional, Dict, Any

import pandas as pd


DB_PATH = "out/assistant.db"


def get_conn():
    return sqlite3.connect(DB_PATH)


def load_future_candles_after_timestamp(
    event_timestamp_iso: str,
    timeframe: str = "15m",
    limit: int = 100,
) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()

    try:
        event_ms = int(pd.Timestamp(event_timestamp_iso).timestamp() * 1000)
    except Exception:
        return pd.DataFrame()

    conn = get_conn()
    query = """
        SELECT open_time, close_time, open, high, low, close
        FROM candles
        WHERE timeframe = ?
          AND open_time >= ?
        ORDER BY open_time ASC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(timeframe, event_ms, limit))
    conn.close()

    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


def simulate_outcome_from_candles(
    decision: str,
    entry: Optional[float],
    sl: Optional[float],
    tp1: Optional[float],
    tp2: Optional[float],
    future_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Returns:
    {
        "status": "tp1_hit" | "tp2_hit" | "sl_hit" | "open" | "no_data",
        "hit_price": ...,
        "hit_time": ...,
        "bars_checked": ...
    }
    """
    if (
        future_df is None
        or future_df.empty
        or entry is None
        or sl is None
        or tp1 is None
        or decision not in ("BUY", "SELL")
    ):
        return {
            "status": "no_data",
            "hit_price": None,
            "hit_time": None,
            "bars_checked": 0,
        }

    for _, row in future_df.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        ts = str(row["time"])

        if decision == "BUY":
            # worst-case conservative ordering inside same candle:
            # stop before tp if both touched
            if low <= sl:
                return {
                    "status": "sl_hit",
                    "hit_price": sl,
                    "hit_time": ts,
                    "bars_checked": len(future_df),
                }

            if tp2 is not None and high >= tp2:
                return {
                    "status": "tp2_hit",
                    "hit_price": tp2,
                    "hit_time": ts,
                    "bars_checked": len(future_df),
                }

            if high >= tp1:
                return {
                    "status": "tp1_hit",
                    "hit_price": tp1,
                    "hit_time": ts,
                    "bars_checked": len(future_df),
                }

        elif decision == "SELL":
            if high >= sl:
                return {
                    "status": "sl_hit",
                    "hit_price": sl,
                    "hit_time": ts,
                    "bars_checked": len(future_df),
                }

            if tp2 is not None and low <= tp2:
                return {
                    "status": "tp2_hit",
                    "hit_price": tp2,
                    "hit_time": ts,
                    "bars_checked": len(future_df),
                }

            if low <= tp1:
                return {
                    "status": "tp1_hit",
                    "hit_price": tp1,
                    "hit_time": ts,
                    "bars_checked": len(future_df),
                }

    return {
        "status": "open",
        "hit_price": None,
        "hit_time": None,
        "bars_checked": len(future_df),
    }