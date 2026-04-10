from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

EPS = 1e-12
HURST_MIN_SAMPLES = 100


@dataclass
class QuantumState:
    """Quantum-inspired summary computed from classical stochastic features."""

    state: str
    # [0, 1]: cross-timeframe alignment + persistence strength.
    coherence: float
    # [-1, 1]: normalized directional drift (positive bullish, negative bearish).
    phase_bias: float
    # [-1, 1]: alignment/disalignment between compatible horizons.
    interference: float
    # [0, 1]: probability proxy from rolling z-score transformed with normal CDF.
    tunneling_probability: float
    # [0, 1]: relative volatility intensity across timeframes.
    amplitude: float
    # [0, 1]: aggregate confidence from coherence and amplitude.
    state_confidence: float


def _safe_close_returns(df: pd.DataFrame, lookback: int) -> pd.Series:
    if "close" not in df.columns or len(df) < lookback + 2:
        return pd.Series(dtype=float)
    return pd.to_numeric(df["close"], errors="coerce").pct_change().dropna().tail(lookback)


def _safe_all_close_returns(df: pd.DataFrame) -> pd.Series:
    if "close" not in df.columns or len(df) < 3:
        return pd.Series(dtype=float)
    return pd.to_numeric(df["close"], errors="coerce").pct_change().dropna()


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _hurst_exponent(returns: pd.Series) -> float | None:
    """
    Hurst estimate via variance scaling on cumulative returns.
    Used only when the sample is sufficiently long.
    """
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < HURST_MIN_SAMPLES:
        return None

    path = r.cumsum().to_numpy(dtype=float)
    scales = (2, 4, 8, 16, 32)
    log_scales: list[float] = []
    log_vars: list[float] = []

    for scale in scales:
        if len(path) <= scale + 2:
            continue
        diffs = path[scale:] - path[:-scale]
        var_scale = float(np.var(diffs, ddof=1))
        if not math.isfinite(var_scale) or var_scale <= 0:
            continue
        log_scales.append(math.log(float(scale)))
        log_vars.append(math.log(var_scale))

    if len(log_scales) < 2:
        return None

    slope, _ = np.polyfit(log_scales, log_vars, deg=1)
    hurst = float(np.clip(0.5 * float(slope), 0.0, 1.0))
    return hurst if math.isfinite(hurst) else None


def _persistence_fallback_short_sample(returns: pd.Series) -> float:
    """
    Robust fallback for short samples:
    trend persistence proxy = |mean| / std, normalized to [0, 1].
    """
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 3:
        return 0.0
    drift = float(r.mean())
    vol = float(r.std(ddof=1))
    if not math.isfinite(drift) or not math.isfinite(vol) or vol <= EPS:
        return 0.0
    signal_to_noise = abs(drift) / (vol + EPS)
    return _clip01(signal_to_noise / (1.0 + signal_to_noise))


def _persistence_score(returns: pd.Series) -> float:
    """
    Persistence score in [0, 1].
    - Long samples: Hurst-based distance from random walk (0.5).
    - Short samples: robust standardized-drift fallback.
    """
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 3:
        return 0.0

    hurst = _hurst_exponent(r)
    if hurst is not None:
        return _clip01(abs(hurst - 0.5) * 2.0)
    return _persistence_fallback_short_sample(r)


def _directional_bias(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 2:
        return 0.0
    drift = float(r.mean())
    vol = float(r.std(ddof=1))
    if not math.isfinite(drift) or not math.isfinite(vol) or vol <= EPS:
        return 0.0
    z = drift / (vol + EPS)
    return float(np.clip(z / (1.0 + abs(z)), -1.0, 1.0))


def _phase_vol_persistence(returns: pd.Series) -> tuple[float, float, float]:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    n = len(r)
    if n < 2:
        return 0.0, 0.0, 0.0

    vol = float(r.std(ddof=1))
    if not math.isfinite(vol) or vol <= EPS:
        return 0.0, 0.0, _persistence_score(r)

    phase_bias = _directional_bias(r)
    realized_vol = float(vol * math.sqrt(n))
    return phase_bias, realized_vol, _persistence_score(r)


def _pairwise_alignment(phases: np.ndarray) -> float:
    if phases.size < 2:
        return 0.0
    pair_products: list[float] = []
    for i in range(phases.size):
        for j in range(i + 1, phases.size):
            pair_products.append(float(phases[i] * phases[j]))
    if not pair_products:
        return 0.0
    arr = np.array(pair_products, dtype=float)
    return _clip01(float(((arr + 1.0) / 2.0).mean()))


def _safe_correlation(a: pd.Series, b: pd.Series, min_points: int = 12) -> float:
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < min_points:
        return 0.0

    std_a = float(aligned.iloc[:, 0].std(ddof=1))
    std_b = float(aligned.iloc[:, 1].std(ddof=1))
    if (
        not math.isfinite(std_a)
        or not math.isfinite(std_b)
        or std_a <= EPS
        or std_b <= EPS
    ):
        return 0.0

    corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
    if not math.isfinite(corr):
        return 0.0
    return float(np.clip(corr, -1.0, 1.0))


def _rolling_zscore_probability(close: pd.Series, window: int = 32, min_periods: int = 16) -> float:
    prices = pd.to_numeric(close, errors="coerce").dropna()
    if len(prices) < max(3, min_periods):
        return 0.5

    rolling_mean = prices.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = prices.rolling(window=window, min_periods=min_periods).std(ddof=1)
    mu = float(rolling_mean.iloc[-1]) if not rolling_mean.empty else float("nan")
    sigma = float(rolling_std.iloc[-1]) if not rolling_std.empty else float("nan")
    x = float(prices.iloc[-1])
    if not math.isfinite(mu) or not math.isfinite(sigma) or not math.isfinite(x):
        return 0.5

    z = (x - mu) / (sigma + EPS)
    return _clip01((2.0 * _normal_cdf(abs(z))) - 1.0)


def _compute_interference_from_m15(df_m15: pd.DataFrame) -> float:
    if "close" not in df_m15.columns:
        return 0.0

    close = pd.to_numeric(df_m15["close"], errors="coerce")
    r_15m = close.pct_change()
    r_1h_proxy = close.pct_change(4)
    r_4h_proxy = close.pct_change(16)

    corr_1h = _safe_correlation(r_15m, r_1h_proxy)
    corr_4h = _safe_correlation(r_15m, r_4h_proxy)
    return float(np.clip(np.mean([corr_1h, corr_4h]), -1.0, 1.0))


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
    if "close" not in df_m15.columns or "close" not in df_h1.columns or "close" not in df_h4.columns:
        return QuantumState(
            state="WARMING_UP",
            coherence=0.0,
            phase_bias=0.0,
            interference=0.0,
            tunneling_probability=0.0,
            amplitude=0.0,
            state_confidence=0.0,
        )

    m15_returns_recent = _safe_close_returns(df_m15, 32)
    h1_returns_recent = _safe_close_returns(df_h1, 20)
    h4_returns_recent = _safe_close_returns(df_h4, 12)
    # Structural persistence uses long history; directional bias remains recent.
    m15_returns_structural = _safe_all_close_returns(df_m15)
    h1_returns_structural = _safe_all_close_returns(df_h1)
    h4_returns_structural = _safe_all_close_returns(df_h4)

    phase_m15, vol_m15, _ = _phase_vol_persistence(m15_returns_recent)
    phase_h1, vol_h1, _ = _phase_vol_persistence(h1_returns_recent)
    phase_h4, vol_h4, _ = _phase_vol_persistence(h4_returns_recent)
    pers_m15 = _persistence_score(m15_returns_structural)
    pers_h1 = _persistence_score(h1_returns_structural)
    pers_h4 = _persistence_score(h4_returns_structural)

    phases = np.array([phase_m15, phase_h1, phase_h4], dtype=float)
    vols = np.array([vol_m15, vol_h1, vol_h4], dtype=float)
    persistences = np.array([pers_m15, pers_h1, pers_h4], dtype=float)

    coherence = _clip01((_pairwise_alignment(phases) + float(np.mean(persistences))) / 2.0)
    phase_bias = float(np.clip(np.mean(phases), -1.0, 1.0))
    interference = _compute_interference_from_m15(df_m15)

    positive_vols = vols[np.isfinite(vols) & (vols > 0.0)]
    if positive_vols.size == 0:
        amplitude = 0.0
    else:
        vol_baseline = float(np.median(positive_vols))
        normalized_vols = positive_vols / (positive_vols + vol_baseline + EPS)
        amplitude = _clip01(float(np.mean(normalized_vols)))

    tunneling_probability = _rolling_zscore_probability(
        pd.to_numeric(df_m15["close"], errors="coerce"),
        window=32,
        min_periods=16,
    )

    directional_confidence = coherence * abs(phase_bias)
    state_confidence = _clip01((coherence + amplitude) / 2.0)

    if coherence < (1.0 / 3.0):
        state = "DECOHERENT"
    elif tunneling_probability >= 0.80 and phase_bias > 0.0:
        state = "BULLISH_TUNNEL"
    elif tunneling_probability >= 0.80 and phase_bias < 0.0:
        state = "BEARISH_TUNNEL"
    elif directional_confidence >= 0.25 and phase_bias > 0.0:
        state = "COHERENT_BULLISH"
    elif directional_confidence >= 0.25 and phase_bias < 0.0:
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
