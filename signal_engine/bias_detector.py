import pandas as pd
from features.indicators import ema

def detect_bias_h1(df_h1: pd.DataFrame) -> str:
    if len(df_h1) < 250:
        return "neutral"
    e200 = ema(df_h1["close"], 200).iloc[-1]
    price = df_h1["close"].iloc[-1]
    if price > e200:
        return "bullish"
    if price < e200:
        return "bearish"
    return "neutral"

def detect_bias_h4(df_h4: pd.DataFrame) -> str:
    # simpler: use EMA50 on H4 as coarse bias
    if len(df_h4) < 80:
        return "neutral"
    e50 = ema(df_h4["close"], 50).iloc[-1]
    price = df_h4["close"].iloc[-1]
    if price > e50:
        return "bullish"
    if price < e50:
        return "bearish"
    return "neutral"

def combined_bias(b1: str, b4: str) -> str:
    if b1 == b4 and b1 in ("bullish", "bearish"):
        return b1
    return "neutral"