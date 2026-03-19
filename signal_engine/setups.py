from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from features.indicators import ema, atr


@dataclass
class SetupResult:
    setup: str
    decision: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    reasons: list[str]


def _last_swing_high(df: pd.DataFrame, lookback: int = 20) -> float:
    return float(df["high"].tail(lookback).max())


def _last_swing_low(df: pd.DataFrame, lookback: int = 20) -> float:
    return float(df["low"].tail(lookback).min())


def trend_pullback(df_m15: pd.DataFrame, df_h1: pd.DataFrame, bias: str) -> Optional[SetupResult]:
    if len(df_m15) < 250 or len(df_h1) < 100:
        return None

    e50 = ema(df_m15["close"], 50).iloc[-1]
    e200 = ema(df_m15["close"], 200).iloc[-1]
    a14 = atr(df_m15, 14).iloc[-1]

    if pd.isna(a14):
        return None

    lo = min(e50, e200)
    hi = max(e50, e200)

    last = df_m15.iloc[-1]
    in_zone = (last["low"] <= hi and last["high"] >= lo)

    if bias == "bullish" and in_zone and last["close"] > last["open"]:
        swing_low_m15 = _last_swing_low(df_m15, 25)
        entry = float(_last_swing_high(df_m15.tail(3), 3))
        sl = float(swing_low_m15 - 0.3 * a14)

        tp1 = _last_swing_high(df_h1, 30)
        risk = entry - sl
        tp2 = float(tp1 + risk)

        if tp1 <= entry:
            return None

        return SetupResult(
            setup="TREND_PULLBACK",
            decision="BUY",
            entry=entry,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            reasons=[
                "M15 pullback into EMA value zone",
                "Bullish close confirmation",
                "Bullish H1/H4 bias",
                "TP1 based on H1 swing high",
            ],
        )

    if bias == "bearish" and in_zone and last["close"] < last["open"]:
        swing_high_m15 = _last_swing_high(df_m15, 25)
        entry = float(_last_swing_low(df_m15.tail(3), 3))
        sl = float(swing_high_m15 + 0.3 * a14)

        tp1 = _last_swing_low(df_h1, 30)
        risk = sl - entry
        tp2 = float(tp1 - risk)

        if tp1 >= entry:
            return None

        return SetupResult(
            setup="TREND_PULLBACK",
            decision="SELL",
            entry=entry,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            reasons=[
                "M15 pullback into EMA value zone",
                "Bearish close confirmation",
                "Bearish H1/H4 bias",
                "TP1 based on H1 swing low",
            ],
        )

    return None


def sweep_reclaim(df_m15: pd.DataFrame, bias: str) -> Optional[SetupResult]:
    if len(df_m15) < 120:
        return None

    a14 = atr(df_m15, 14).iloc[-1]
    if pd.isna(a14):
        return None

    prior_high = float(df_m15["high"].iloc[-20:-1].max())
    prior_low = float(df_m15["low"].iloc[-20:-1].min())

    last = df_m15.iloc[-1]

    swept_up = last["high"] > prior_high
    reclaimed_down = last["close"] < prior_high

    if swept_up and reclaimed_down and bias in ("bearish", "neutral"):
        entry = float(prior_high)
        sl = float(last["high"] + 0.25 * a14)
        tp1 = float(entry - 1.5 * (sl - entry))
        tp2 = float(entry - 2.5 * (sl - entry))
        return SetupResult(
            setup="SWEEP_RECLAIM",
            decision="SELL",
            entry=entry,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            reasons=[
                "Liquidity sweep above local high",
                "Reclaim below the swept level",
                "Sell aligned with bearish/neutral context",
            ],
        )

    swept_down = last["low"] < prior_low
    reclaimed_up = last["close"] > prior_low

    if swept_down and reclaimed_up and bias in ("bullish", "neutral"):
        entry = float(prior_low)
        sl = float(last["low"] - 0.25 * a14)
        tp1 = float(entry + 1.5 * (entry - sl))
        tp2 = float(entry + 2.5 * (entry - sl))
        return SetupResult(
            setup="SWEEP_RECLAIM",
            decision="BUY",
            entry=entry,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            reasons=[
                "Liquidity sweep below local low",
                "Reclaim above the swept level",
                "Buy aligned with bullish/neutral context",
            ],
        )

    return None


def breakout_confirmation(df_m15: pd.DataFrame, df_h1: pd.DataFrame, bias: str) -> Optional[SetupResult]:
    if len(df_m15) < 120 or len(df_h1) < 80:
        return None

    a14 = atr(df_m15, 14).iloc[-1]
    if pd.isna(a14):
        return None

    last = df_m15.iloc[-1]
    prev = df_m15.iloc[-2]

    range_high = float(df_m15["high"].iloc[-12:-1].max())
    range_low = float(df_m15["low"].iloc[-12:-1].min())

    candle_range = float(last["high"] - last["low"])
    atr_ok = candle_range <= a14 * 1.8

    if bias == "bullish":
        broke_up = last["close"] > range_high
        strong_close = last["close"] > last["open"]
        prev_not_already_broken = prev["close"] <= range_high

        if broke_up and strong_close and prev_not_already_broken and atr_ok:
            local_swing_low = _last_swing_low(df_m15, 12)
            entry = float(last["close"])
            sl = float(local_swing_low - 0.25 * a14)

            risk = entry - sl
            if risk <= 0:
                return None

            h1_high = _last_swing_high(df_h1, 40)
            tp1 = float(h1_high)

            # se il target H1 è inutilizzabile o troppo vicino, fallback serio
            rr1 = (tp1 - entry) / risk if tp1 > entry else -1

            if rr1 < 1.2:
                tp1 = float(entry + 1.8 * risk)
                tp2 = float(entry + 2.8 * risk)
            else:
                tp2 = float(tp1 + risk)

            rr1 = (tp1 - entry) / risk
            if rr1 < 1.2:
                return None

            return SetupResult(
                setup="BREAKOUT_CONFIRMATION",
                decision="BUY",
                entry=entry,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
                reasons=[
                    "Bullish H1/H4 bias",
                    "M15 breakout above local range high",
                    "Bullish close confirmation on breakout",
                    "RR validated after breakout filter",
                ],
            )

    if bias == "bearish":
        broke_down = last["close"] < range_low
        strong_close = last["close"] < last["open"]
        prev_not_already_broken = prev["close"] >= range_low

        if broke_down and strong_close and prev_not_already_broken and atr_ok:
            local_swing_high = _last_swing_high(df_m15, 12)
            entry = float(last["close"])
            sl = float(local_swing_high + 0.25 * a14)

            risk = sl - entry
            if risk <= 0:
                return None

            h1_low = _last_swing_low(df_h1, 40)
            tp1 = float(h1_low)

            rr1 = (entry - tp1) / risk if tp1 < entry else -1

            if rr1 < 1.2:
                tp1 = float(entry - 1.8 * risk)
                tp2 = float(entry - 2.8 * risk)
            else:
                tp2 = float(tp1 - risk)

            rr1 = (entry - tp1) / risk
            if rr1 < 1.2:
                return None

            return SetupResult(
                setup="BREAKOUT_CONFIRMATION",
                decision="SELL",
                entry=entry,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
                reasons=[
                    "Bearish H1/H4 bias",
                    "M15 breakout below local range low",
                    "Bearish close confirmation on breakout",
                    "RR validated after breakout filter",
                ],
            )

    return None