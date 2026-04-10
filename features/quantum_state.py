from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from features.indicators import atr


@dataclass
class QuantumState:
    state: str
    coherence: float
    phase_bias: float
    interference: float
    tunneling_probability: float
    amplitude: float
    state_confidence: float


def _safe_close_returns(df: pd.DataFrame, lookback: int) -> pd.Series:
    if "close" not in df.columns or len(df) < lookback + 2:
        return pd.Series(dtype=float)

    return (
        df["close"]
        .astype(float)
        .pct_change()
        .dropna()
        .tail(lookback)
    )


def _phase_strength(returns: pd.Series) -> tuple[float, float]:
    if returns.empty:
        return 0.0, 0.0

    drift = float(returns.mean())
    vol = float(returns.std(ddof=0))
    if vol <= 1e-9:
        phase_bias = 0.0
    else:
        phase_bias = float(np.tanh((drift / vol) * 1.8))

    amplitude = float(np.clip(vol * np.sqrt(len(returns)) * 12.0, 0.0, 1.0))
    return phase_bias, amplitude


def build_quantum_state(
    df_m15: pd.DataFrame,
    df_h1: pd.DataFrame,
    df_h4: pd.DataFrame,
) -> QuantumState:
    if len(df_m15) < 80 or len(df_h1) < 40 or len(df_h4) < 20:
        return QuantumState(
            state="WARMING_UP",
            coherence=0.0,
            phase_bias=0.0,
            interference=0.0,
            tunneling_probability=0.0,
            amplitude=0.0,
            state_confidence=0.0,
        )

    m15_returns = _safe_close_returns(df_m15, 32)
    h1_returns = _safe_close_returns(df_h1, 20)
    h4_returns = _safe_close_returns(df_h4, 12)

    phase_m15, amp_m15 = _phase_strength(m15_returns)
    phase_h1, amp_h1 = _phase_strength(h1_returns)
    phase_h4, amp_h4 = _phase_strength(h4_returns)

    phases = np.array([phase_m15, phase_h1, phase_h4], dtype=float)
    amplitudes = np.array([amp_m15, amp_h1, amp_h4], dtype=float)

    # Coherence grows when directional phase is aligned across timeframes.
    coherence = float(np.clip(1.0 - np.std(phases), 0.0, 1.0))
    phase_bias = float(np.clip(np.mean(phases), -1.0, 1.0))

    higher_phase = float(np.mean([phase_h1, phase_h4]))
    interference = float(np.clip(phase_m15 * higher_phase, -1.0, 1.0))
    amplitude = float(np.clip(np.mean(amplitudes), 0.0, 1.0))

    last_close = float(df_m15["close"].iloc[-1])
    recent_range = float(df_m15["high"].tail(16).max() - df_m15["low"].tail(16).min())
    normalized_range = (recent_range / last_close) if last_close > 0 else 0.0

    a14 = atr(df_m15, 14).iloc[-1]
    if pd.isna(a14) or last_close <= 0:
        compression = 0.0
    else:
        compression = float(np.clip(1.0 - ((a14 / last_close) * 90.0), 0.0, 1.0))

    tunneling_probability = float(
        np.clip(
            0.45 * coherence
            + 0.35 * abs(phase_bias)
            + 0.20 * compression
            - min(normalized_range * 35.0, 0.35),
            0.0,
            1.0,
        )
    )

    directional_confidence = coherence * max(abs(phase_bias), 0.15)
    state_confidence = float(np.clip(0.55 * coherence + 0.45 * amplitude, 0.0, 1.0))

    if coherence < 0.42:
        state = "DECOHERENT"
    elif tunneling_probability >= 0.72 and phase_bias > 0.18:
        state = "BULLISH_TUNNEL"
    elif tunneling_probability >= 0.72 and phase_bias < -0.18:
        state = "BEARISH_TUNNEL"
    elif directional_confidence >= 0.46 and phase_bias > 0:
        state = "COHERENT_BULLISH"
    elif directional_confidence >= 0.46 and phase_bias < 0:
        state = "COHERENT_BEARISH"
    elif amplitude < 0.18:
        state = "LOW_ENERGY"
    else:
        state = "TRANSITIONAL"

    return QuantumState(
        state=state,
        coherence=round(coherence, 4),
        phase_bias=round(phase_bias, 4),
        interference=round(interference, 4),
        tunneling_probability=round(tunneling_probability, 4),
        amplitude=round(amplitude, 4),
        state_confidence=round(state_confidence, 4),
    )
