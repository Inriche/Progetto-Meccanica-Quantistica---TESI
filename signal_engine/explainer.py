import pandas as pd
from features.indicators import ema, atr
from signal_engine.bias_detector import detect_bias_h1, detect_bias_h4, combined_bias


def explain_no_setup(df_m15: pd.DataFrame, df_h1: pd.DataFrame, df_h4: pd.DataFrame) -> str:
    if len(df_m15) < 150 or len(df_h1) < 100 or len(df_h4) < 60:
        return "warming up"

    b_h1 = detect_bias_h1(df_h1)
    b_h4 = detect_bias_h4(df_h4)
    b_comb = combined_bias(b_h1, b_h4)

    reasons = []

    # -------------------------
    # 1) Pullback in value zone
    # -------------------------
    e50 = ema(df_m15["close"], 50).iloc[-1]
    e200 = ema(df_m15["close"], 200).iloc[-1]
    last = df_m15.iloc[-1]

    lo = min(e50, e200)
    hi = max(e50, e200)

    in_zone = (last["low"] <= hi and last["high"] >= lo)

    if not in_zone:
        reasons.append("no pullback in value zone")

    # -------------------------
    # 2) Sweep reclaim check
    # -------------------------
    prior_high = float(df_m15["high"].iloc[-20:-1].max())
    prior_low = float(df_m15["low"].iloc[-20:-1].min())

    swept_up = last["high"] > prior_high
    reclaimed_down = last["close"] < prior_high

    swept_down = last["low"] < prior_low
    reclaimed_up = last["close"] > prior_low

    sweep_ok = (swept_up and reclaimed_down) or (swept_down and reclaimed_up)

    if not sweep_ok:
        reasons.append("no sweep reclaim")

    # -------------------------
    # 3) Breakout confirmation
    # -------------------------
    if len(df_m15) >= 20:
        prev = df_m15.iloc[-2]
        range_high = float(df_m15["high"].iloc[-12:-1].max())
        range_low = float(df_m15["low"].iloc[-12:-1].min())

        a14 = atr(df_m15, 14).iloc[-1]
        candle_range = float(last["high"] - last["low"])

        atr_ok = pd.notna(a14) and candle_range <= a14 * 1.8

        broke_up = last["close"] > range_high and prev["close"] <= range_high and last["close"] > last["open"]
        broke_down = last["close"] < range_low and prev["close"] >= range_low and last["close"] < last["open"]

        breakout_ok = atr_ok and (
            (b_comb == "bullish" and broke_up) or
            (b_comb == "bearish" and broke_down)
        )

        if not breakout_ok:
            reasons.append("no breakout confirmation")

    # -------------------------
    # 4) Bias info
    # -------------------------
    if b_comb == "neutral":
        reasons.append("multi-timeframe bias not aligned")

    if not reasons:
        return "setup exists but may be filtered by RR/score/risk"

    return "; ".join(reasons)