from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from features.indicators import atr


@dataclass
class QuantumState:
    """Quantum-inspired state built from classical time-series statistics."""

    state: str
    # Cross-timeframe directional alignment and persistence proxy in [0, 1].
    coherence: float
    # Signed directional bias in [-1, 1], derived from standardized drift.
    phase_bias: float
    # Short-vs-higher timeframe directional interaction in [-1, 1].
    interference: float
    # Standardized breakout/deviation likelihood proxy in [0, 1].
    tunneling_probability: float
    # Relative volatility intensity proxy in [0, 1].
    amplitude: float
    # Aggregate confidence from coherence and amplitude in [0, 1].
    state_confidence: float


def _safe_close_returns(df: pd.DataFrame, lookback: int) -> pd.Series:
    if "close" not in df.columns or len(df) < lookback + 2:
        return pd.Series(dtype=float)

    return df["close"].astype(float).pct_change().dropna().tail(lookback)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _hurst_proxy(returns: pd.Series) -> float:
    """Estimate Hurst exponent proxy with variance scaling on cumulative returns."""
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 16:
        return 0.5

    path = r.cumsum().to_numpy(dtype=float)
    scales = (2, 4, 8, 16)
    log_scales: list[float] = []
    log_vars: list[float] = []

    for k in scales:
        if len(path) <= k + 2:
            continue
        diffs = path[k:] - path[:-k]
        var_k = float(np.var(diffs, ddof=1))
        if not math.isfinite(var_k) or var_k <= 0:
            continue
        log_scales.append(math.log(float(k)))
        log_vars.append(math.log(var_k))

    if len(log_scales) < 2:
        return 0.5

    slope, _ = np.polyfit(log_scales, log_vars, deg=1)
    return float(np.clip(0.5 * float(slope), 0.0, 1.0))


def _phase_vol_hurst(returns: pd.Series) -> tuple[float, float, float]:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    n = len(r)
    if n < 2:
        return 0.0, 0.0, 0.5

    drift = float(r.mean())
    vol = float(r.std(ddof=1))
    if not math.isfinite(vol) or vol <= 0:
        return 0.0, 0.0, 0.5

    # t-statistic of mean return as directional proxy.
    t_stat = drift / (vol / math.sqrt(n))
    phase_bias = float(math.tanh(t_stat))
    realized_vol = float(vol * math.sqrt(n))
    return phase_bias, realized_vol, _hurst_proxy(r)


def _pairwise_alignment(phases: np.ndarray) -> float:
    if phases.size < 2:
        return 0.0
    products: list[float] = []
    for i in range(phases.size):
        for j in range(i + 1, phases.size):
            products.append(float(phases[i] * phases[j]))
    if not products:
        return 0.0
    arr = np.array(products, dtype=float)
    return _clip01(float(((arr + 1.0) / 2.0).mean()))


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _latest_price_zscore(close: pd.Series, window: int = 32) -> float:
    prices = pd.to_numeric(close, errors="coerce").dropna()
    if len(prices) < max(3, window):
        return 0.0

    sample = prices.tail(window)
    mean = float(sample.mean())
    std = float(sample.std(ddof=1))
    last = float(sample.iloc[-1])
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0 or not math.isfinite(last):
        return 0.0
    return float((last - mean) / std)


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

    phase_m15, vol_m15, hurst_m15 = _phase_vol_hurst(m15_returns)
    phase_h1, vol_h1, hurst_h1 = _phase_vol_hurst(h1_returns)
    phase_h4, vol_h4, hurst_h4 = _phase_vol_hurst(h4_returns)

    phases = np.array([phase_m15, phase_h1, phase_h4], dtype=float)
    vols = np.array([vol_m15, vol_h1, vol_h4], dtype=float)
    hursts = np.array([hurst_m15, hurst_h1, hurst_h4], dtype=float)

    # Coherence: directional alignment + persistence away from random walk.
    alignment = _pairwise_alignment(phases)
    persistence = float(np.mean(np.abs(hursts - 0.5) * 2.0))
    coherence = _clip01((alignment + persistence) / 2.0)
    phase_bias = float(np.clip(np.mean(phases), -1.0, 1.0))

    higher_phase = float(np.mean([phase_h1, phase_h4]))
    interference = float(np.clip(phase_m15 * higher_phase, -1.0, 1.0))

    positive_vols = vols[np.isfinite(vols) & (vols > 0.0)]
    if positive_vols.size == 0:
        amplitude = 0.0
    else:
        vol_baseline = float(np.median(positive_vols))
        normalized_vols = positive_vols / (positive_vols + vol_baseline)
        amplitude = _clip01(float(np.mean(normalized_vols)))

    close_series = pd.to_numeric(df_m15["close"], errors="coerce")
    atr_pct = (atr(df_m15, 14) / close_series.replace(0.0, pd.NA)).replace([np.inf, -np.inf], pd.NA).dropna()
    if atr_pct.empty:
        compression = 0.5
    else:
        atr_tail = pd.to_numeric(atr_pct.tail(96), errors="coerce").dropna()
        if atr_tail.empty:
            compression = 0.5
        else:
            current_atr = float(atr_tail.iloc[-1])
            baseline_atr = float(atr_tail.median())
            denom = baseline_atr + current_atr
            if not math.isfinite(current_atr) or not math.isfinite(baseline_atr) or denom <= 0:
                compression = 0.5
            else:
                compression = _clip01(baseline_atr / denom)

    # P(tunneling) uses standardized displacement, compression, and coherence.
    zscore = _latest_price_zscore(close_series, window=32)
    zscore_probability = _clip01((2.0 * _normal_cdf(abs(zscore))) - 1.0)
    tunneling_probability = _clip01(float(np.mean([zscore_probability, compression, coherence])))

    directional_confidence = coherence * abs(phase_bias)
    state_confidence = _clip01(float(np.mean([coherence, amplitude])))

    if coherence < 0.35:
        state = "DECOHERENT"
    elif tunneling_probability >= 0.70 and phase_bias > 0.15:
        state = "BULLISH_TUNNEL"
    elif tunneling_probability >= 0.70 and phase_bias < -0.15:
        state = "BEARISH_TUNNEL"
    elif directional_confidence >= 0.28 and phase_bias > 0:
        state = "COHERENT_BULLISH"
    elif directional_confidence >= 0.28 and phase_bias < 0:
        state = "COHERENT_BEARISH"
    elif amplitude < 0.20:
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
