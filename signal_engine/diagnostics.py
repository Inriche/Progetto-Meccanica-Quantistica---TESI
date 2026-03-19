import pandas as pd
from features.indicators import ema, atr, rr
from signal_engine.bias_detector import detect_bias_h1, detect_bias_h4, combined_bias
from signal_engine.setups import trend_pullback, sweep_reclaim, breakout_confirmation


def run_diagnostics(df_m15: pd.DataFrame, df_h1: pd.DataFrame, df_h4: pd.DataFrame) -> dict:
    out = {
        "bias_h1": "neutral",
        "bias_h4": "neutral",
        "combined_bias": "neutral",
        "value_zone": False,
        "sweep_reclaim": False,
        "breakout_confirmation": False,
        "trend_pullback_rr": None,
        "sweep_reclaim_rr": None,
        "breakout_rr": None,
        "notes": [],
    }

    if len(df_m15) < 150 or len(df_h1) < 100 or len(df_h4) < 60:
        out["notes"].append("not enough history")
        return out

    b_h1 = detect_bias_h1(df_h1)
    b_h4 = detect_bias_h4(df_h4)
    b_comb = combined_bias(b_h1, b_h4)

    out["bias_h1"] = b_h1
    out["bias_h4"] = b_h4
    out["combined_bias"] = b_comb

    # value zone check
    e50 = ema(df_m15["close"], 50).iloc[-1]
    e200 = ema(df_m15["close"], 200).iloc[-1]
    last = df_m15.iloc[-1]

    lo = min(e50, e200)
    hi = max(e50, e200)
    in_zone = (last["low"] <= hi and last["high"] >= lo)
    out["value_zone"] = bool(in_zone)

    # sweep reclaim check
    prior_high = float(df_m15["high"].iloc[-20:-1].max())
    prior_low = float(df_m15["low"].iloc[-20:-1].min())

    swept_up = last["high"] > prior_high
    reclaimed_down = last["close"] < prior_high
    swept_down = last["low"] < prior_low
    reclaimed_up = last["close"] > prior_low

    sweep_ok = (swept_up and reclaimed_down) or (swept_down and reclaimed_up)
    out["sweep_reclaim"] = bool(sweep_ok)

    # breakout check
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
    out["breakout_confirmation"] = bool(breakout_ok)

    # try real setup functions
    if b_comb in ("bullish", "bearish"):
        tp = trend_pullback(df_m15, df_h1, b_comb)
        if tp is not None:
            out["trend_pullback_rr"] = round(rr(tp.entry, tp.sl, tp.tp1), 2)

        sr = sweep_reclaim(df_m15, b_comb)
        if sr is not None:
            out["sweep_reclaim_rr"] = round(rr(sr.entry, sr.sl, sr.tp1), 2)

        bo = breakout_confirmation(df_m15, df_h1, b_comb)
        if bo is not None:
            out["breakout_rr"] = round(rr(bo.entry, bo.sl, bo.tp1), 2)

    if not out["value_zone"]:
        out["notes"].append("no pullback in value zone")
    if not out["sweep_reclaim"]:
        out["notes"].append("no sweep reclaim")
    if not out["breakout_confirmation"]:
        out["notes"].append("no breakout confirmation")

    return out