import pandas as pd
from .indicators import atr

def volatility_regime(df_m15: pd.DataFrame) -> str:
    if len(df_m15) < 50:
        return "unknown"
    a = atr(df_m15, 14).iloc[-1]
    if pd.isna(a):
        return "unknown"
    # crude regime classification
    if a < df_m15["close"].iloc[-1] * 0.002:  # <0.2%
        return "low"
    if a > df_m15["close"].iloc[-1] * 0.01:   # >1%
        return "high"
    return "normal"