import pandas as pd
from features.indicators import ema, atr
from signal_engine.bias_detector import detect_bias_h1, detect_bias_h4, combined_bias


def classify_market_context(df_m15: pd.DataFrame, df_h1: pd.DataFrame, df_h4: pd.DataFrame) -> str:
    if len(df_m15) < 150 or len(df_h1) < 100 or len(df_h4) < 60:
        return "warming_up"

    b_h1 = detect_bias_h1(df_h1)
    b_h4 = detect_bias_h4(df_h4)
    b_comb = combined_bias(b_h1, b_h4)

    if b_comb == "neutral":
        return "transition"

    e50 = ema(df_m15["close"], 50).iloc[-1]
    e200 = ema(df_m15["close"], 200).iloc[-1]
    last_close = df_m15["close"].iloc[-1]
    a14 = atr(df_m15, 14).iloc[-1]

    if pd.isna(a14):
        return "transition"

    # distanza dal valore medio
    dist_e50 = abs(last_close - e50) / last_close
    dist_e200 = abs(last_close - e200) / last_close

    recent_range = float(df_m15["high"].tail(20).max() - df_m15["low"].tail(20).min())
    normalized_range = recent_range / last_close

    # chop: medie vicine + range piccolo
    if abs(e50 - e200) / last_close < 0.002 and normalized_range < 0.006:
        return "chop"

    # trend troppo esteso: prezzo molto lontano da EMA50
    if dist_e50 > 0.01:
        return "trend_extended"

    # trend pulito: bias allineato + prezzo ordinato rispetto alle EMA
    if b_comb == "bullish" and last_close > e50 > e200:
        return "trend_clean"

    if b_comb == "bearish" and last_close < e50 < e200:
        return "trend_clean"

    return "transition"