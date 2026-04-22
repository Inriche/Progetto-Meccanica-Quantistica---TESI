from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

EPS = 1e-12
HURST_MIN_SAMPLES = 100
TIMEFRAME_NAMES = ("m15", "h1", "h4")
TIMEFRAME_LOOKBACKS = {
    "m15": 32,
    "h1": 20,
    "h4": 12,
}


@dataclass
class QuantumState:
    """Classical quantum-inspired summary of a three-timeframe market state."""

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
    # [0, 1]: aggregate confidence from coherence, amplitude, and stability.
    state_confidence: float
    # [0, 1]: normalized effective energy from a classical Hamiltonian proxy.
    energy: float = 0.0
    # [0, 1]: loss of alignment across timeframes.
    decoherence_rate: float = 0.0
    # [0, 1]: regime-shift / instability likelihood.
    transition_rate: float = 0.0
    # Dominant timeframe mode, one of m15 / h1 / h4.
    dominant_mode: str = "m15"


def _resolve_runtime_value(runtime_cfg: Mapping[str, Any] | None, key: str, default: float) -> float:
    if runtime_cfg is None:
        return float(default)
    try:
        value = runtime_cfg.get(key, default)  # type: ignore[assignment]
    except Exception:
        return float(default)
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _clip_signed(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(np.clip(value, -1.0, 1.0))


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _safe_close_series(df: pd.DataFrame) -> pd.Series:
    if "close" not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df["close"], errors="coerce").dropna()


def _safe_return_series(df: pd.DataFrame, lookback: int) -> pd.Series:
    close = _safe_close_series(df)
    if len(close) < 3:
        return pd.Series(dtype=float)
    return close.pct_change().dropna().tail(max(3, int(lookback)))


def _safe_all_close_returns(df: pd.DataFrame) -> pd.Series:
    close = _safe_close_series(df)
    if len(close) < 3:
        return pd.Series(dtype=float)
    return close.pct_change().dropna()


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


def _directional_bias(returns: pd.Series, phase_sensitivity: float = 1.0) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 2:
        return 0.0
    drift = float(r.mean())
    vol = float(r.std(ddof=1))
    if not math.isfinite(drift) or not math.isfinite(vol) or vol <= EPS:
        return 0.0
    z = (drift / (vol + EPS)) * max(0.1, float(phase_sensitivity))
    return float(np.clip(np.tanh(z / 2.0), -1.0, 1.0))


def _phase_vol_persistence(
    returns: pd.Series,
    *,
    phase_sensitivity: float,
) -> tuple[float, float, float]:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    n = len(r)
    if n < 2:
        return 0.0, 0.0, 0.0

    vol = float(r.std(ddof=1))
    if not math.isfinite(vol) or vol <= EPS:
        return 0.0, 0.0, _persistence_score(r)

    phase_bias = _directional_bias(r, phase_sensitivity=phase_sensitivity)
    realized_vol = float(vol * math.sqrt(n))
    return phase_bias, realized_vol, _persistence_score(r)


def _phase_angle_from_bias(phase_bias: float) -> float:
    return float(np.clip(phase_bias, -1.0, 1.0) * math.pi)


def _pairwise_alignment(phases: np.ndarray) -> float:
    if phases.size < 2:
        return 0.0
    pair_products: list[float] = []
    for i in range(phases.size):
        for j in range(i + 1, phases.size):
            pair_products.append(float(math.cos(phases[i] - phases[j])))
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


def _estimate_timeframe_metrics(
    df: pd.DataFrame,
    *,
    lookback: int,
    phase_sensitivity: float,
) -> dict[str, float]:
    recent_returns = _safe_return_series(df, lookback)
    structural_returns = _safe_all_close_returns(df)

    recent_phase, recent_vol, recent_persistence = _phase_vol_persistence(
        recent_returns,
        phase_sensitivity=phase_sensitivity,
    )
    structural_phase, structural_vol, structural_persistence = _phase_vol_persistence(
        structural_returns,
        phase_sensitivity=phase_sensitivity,
    )

    if len(recent_returns) < 3:
        return {
            "amplitude": 0.0,
            "phase": 0.0,
            "potential": 1.0,
            "volatility": 0.0,
            "persistence": 0.0,
            "phase_gap": 0.0,
            "vol_shock": 0.0,
        }

    recent_vol_norm = recent_vol / (recent_vol + structural_vol + EPS) if math.isfinite(recent_vol) else 0.0
    directional_intensity = abs(recent_phase)
    raw_amplitude = (
        0.45 * _clip01(recent_vol_norm)
        + 0.35 * _clip01(recent_persistence)
        + 0.20 * _clip01(directional_intensity)
    )

    phase_gap = abs(recent_phase - structural_phase) / 2.0
    vol_shock = abs(recent_vol - structural_vol) / (recent_vol + structural_vol + EPS)
    effective_potential = _clip01(
        0.40 * (1.0 - _clip01(raw_amplitude))
        + 0.35 * (1.0 - _clip01(recent_persistence))
        + 0.25 * (1.0 - abs(recent_phase))
    )

    return {
        "amplitude": _clip01(raw_amplitude),
        "phase": _clip_signed(recent_phase),
        "potential": effective_potential,
        "volatility": float(recent_vol),
        "persistence": _clip01(recent_persistence),
        "phase_gap": _clip01(phase_gap),
        "vol_shock": _clip01(vol_shock),
        "structural_phase": _clip_signed(structural_phase),
        "structural_persistence": _clip01(structural_persistence),
    }


def _build_state_vector(amplitudes: np.ndarray, phases: np.ndarray) -> np.ndarray:
    amps = np.asarray(amplitudes, dtype=float)
    amps = np.where(np.isfinite(amps) & (amps > 0.0), amps, 0.0)
    if amps.size == 0:
        return np.zeros(0, dtype=np.complex128)

    norm = float(np.linalg.norm(amps))
    if not math.isfinite(norm) or norm <= EPS:
        amps = np.full(amps.shape, 1.0 / math.sqrt(float(amps.size)), dtype=float)
    else:
        amps = amps / norm

    phase_angles = np.asarray([_phase_angle_from_bias(float(p)) for p in phases], dtype=float)
    return amps.astype(np.complex128) * np.exp(1j * phase_angles)


def _build_density_matrix(psi: np.ndarray) -> np.ndarray:
    if psi.size == 0:
        return np.zeros((0, 0), dtype=np.complex128)
    return np.outer(psi, np.conjugate(psi))


def _estimate_coupling_strength(
    left_returns: pd.Series,
    right_returns: pd.Series,
    left_phase: float,
    right_phase: float,
    *,
    phase_sensitivity: float,
    base_coupling: float,
) -> float:
    corr = _safe_correlation(left_returns, right_returns)
    phase_alignment = math.cos(_phase_angle_from_bias(left_phase) - _phase_angle_from_bias(right_phase))
    coupling = (
        0.50 * ((corr + 1.0) / 2.0)
        + 0.50 * ((phase_alignment + 1.0) / 2.0)
    )
    sensitivity = max(0.1, float(phase_sensitivity))
    return _clip01(float(base_coupling) * coupling * (0.75 + 0.25 * sensitivity))


def _build_effective_hamiltonian(
    amplitudes: np.ndarray,
    phases: np.ndarray,
    returns_by_mode: dict[str, pd.Series],
    potentials: np.ndarray,
    *,
    coupling_strength: float,
    phase_sensitivity: float,
) -> np.ndarray:
    n = int(amplitudes.size)
    h = np.zeros((n, n), dtype=float)
    for i in range(n):
        h[i, i] = float(np.clip(potentials[i], 0.0, 1.0))

    mode_names = list(TIMEFRAME_NAMES[:n])
    for i in range(n):
        for j in range(i + 1, n):
            left_name = mode_names[i]
            right_name = mode_names[j]
            coupling = _estimate_coupling_strength(
                returns_by_mode.get(left_name, pd.Series(dtype=float)),
                returns_by_mode.get(right_name, pd.Series(dtype=float)),
                float(phases[i]),
                float(phases[j]),
                phase_sensitivity=phase_sensitivity,
                base_coupling=coupling_strength,
            )
            h[i, j] = -coupling
            h[j, i] = -coupling
    return h


def _compute_energy(psi: np.ndarray, h: np.ndarray) -> float:
    if psi.size == 0 or h.size == 0:
        return 0.0
    raw_energy = float(np.real(np.conjugate(psi) @ h @ psi))
    return _clip01((raw_energy + 1.0) / 2.0)


def _compute_coherence(rho: np.ndarray) -> float:
    if rho.size == 0:
        return 0.0
    off_diag = rho - np.diag(np.diag(rho))
    off_diag_abs = float(np.sum(np.abs(off_diag)))
    # For a normalized 3-mode state, 2.0 is a stable upper bound.
    return _clip01(off_diag_abs / 2.0)


def _compute_interference(phases: np.ndarray, returns_by_mode: dict[str, pd.Series]) -> float:
    if phases.size < 2:
        return 0.0
    pair_terms: list[float] = []
    mode_names = list(TIMEFRAME_NAMES[: phases.size])
    for i in range(phases.size):
        for j in range(i + 1, phases.size):
            phase_alignment = math.cos(_phase_angle_from_bias(phases[i]) - _phase_angle_from_bias(phases[j]))
            corr = _safe_correlation(
                returns_by_mode.get(mode_names[i], pd.Series(dtype=float)),
                returns_by_mode.get(mode_names[j], pd.Series(dtype=float)),
            )
            pair_terms.append(0.5 * phase_alignment + 0.5 * corr)
    if not pair_terms:
        return 0.0
    return _clip_signed(float(np.mean(pair_terms)))


def _compute_decoherence_rate(
    *,
    coherence: float,
    phase_gap: float,
    vol_shock: float,
    amplitude_dispersion: float,
    return_disagreement: float,
) -> float:
    decoherence = (
        0.35 * (1.0 - coherence)
        + 0.20 * phase_gap
        + 0.20 * vol_shock
        + 0.15 * amplitude_dispersion
        + 0.10 * return_disagreement
    )
    return _clip01(decoherence)


def _compute_transition_rate(
    *,
    energy: float,
    decoherence_rate: float,
    phase_gap: float,
    vol_shock: float,
    amplitude_dispersion: float,
) -> float:
    transition = (
        0.28 * energy
        + 0.30 * decoherence_rate
        + 0.22 * phase_gap
        + 0.12 * vol_shock
        + 0.08 * amplitude_dispersion
    )
    return _clip01(transition)


def _select_dominant_mode(amplitudes: np.ndarray) -> str:
    if amplitudes.size == 0:
        return "m15"
    idx = int(np.argmax(amplitudes))
    return TIMEFRAME_NAMES[min(idx, len(TIMEFRAME_NAMES) - 1)]


def _classify_state(
    *,
    coherence: float,
    phase_bias: float,
    energy: float,
    decoherence_rate: float,
    transition_rate: float,
    tunneling_probability: float,
    amplitude: float,
    energy_threshold: float,
    transition_threshold: float,
) -> str:
    if coherence < 0.32 or decoherence_rate >= 0.68:
        return "DECOHERENT"

    if tunneling_probability >= 0.80 and transition_rate >= transition_threshold:
        if phase_bias > 0.08:
            return "BULLISH_TUNNEL"
        if phase_bias < -0.08:
            return "BEARISH_TUNNEL"

    if energy <= energy_threshold and amplitude <= 0.40:
        return "LOW_ENERGY"

    if transition_rate >= transition_threshold or (abs(phase_bias) < 0.10 and coherence < 0.70):
        return "TRANSITIONAL"

    if coherence >= 0.66 and phase_bias >= 0.16 and energy <= 0.68:
        return "COHERENT_BULLISH"
    if coherence >= 0.66 and phase_bias <= -0.16 and energy <= 0.68:
        return "COHERENT_BEARISH"

    if amplitude < 0.20:
        return "LOW_ENERGY"

    return "TRANSITIONAL"


def build_quantum_state(
    df_m15: pd.DataFrame,
    df_h1: pd.DataFrame,
    df_h4: pd.DataFrame,
    runtime_cfg: Mapping[str, Any] | None = None,
) -> QuantumState:
    """Build a classical, quantum-inspired 3-mode state from m15/h1/h4 data."""
    if len(df_m15) < 80 or len(df_h1) < 40 or len(df_h4) < 20:
        return QuantumState(
            state="WARMING_UP",
            coherence=0.0,
            phase_bias=0.0,
            interference=0.0,
            tunneling_probability=0.0,
            amplitude=0.0,
            state_confidence=0.0,
            energy=0.0,
            decoherence_rate=0.0,
            transition_rate=0.0,
            dominant_mode="m15",
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
            energy=0.0,
            decoherence_rate=0.0,
            transition_rate=0.0,
            dominant_mode="m15",
        )

    phase_sensitivity = _resolve_runtime_value(runtime_cfg, "quantum_phase_sensitivity", 1.0)
    coupling_strength = _resolve_runtime_value(runtime_cfg, "quantum_coupling_strength", 0.65)
    energy_threshold = _resolve_runtime_value(runtime_cfg, "quantum_energy_threshold", 0.45)
    transition_threshold = _resolve_runtime_value(runtime_cfg, "quantum_transition_threshold", 0.55)

    mode_frames = {
        "m15": df_m15,
        "h1": df_h1,
        "h4": df_h4,
    }
    mode_metrics: dict[str, dict[str, float]] = {}
    returns_by_mode: dict[str, pd.Series] = {}

    for mode_name in TIMEFRAME_NAMES:
        frame = mode_frames[mode_name]
        lookback = TIMEFRAME_LOOKBACKS[mode_name]
        returns_by_mode[mode_name] = _safe_return_series(frame, lookback)
        mode_metrics[mode_name] = _estimate_timeframe_metrics(
            frame,
            lookback=lookback,
            phase_sensitivity=phase_sensitivity,
        )

    amplitudes = np.array([mode_metrics[m]["amplitude"] for m in TIMEFRAME_NAMES], dtype=float)
    phases = np.array([mode_metrics[m]["phase"] for m in TIMEFRAME_NAMES], dtype=float)
    potentials = np.array([mode_metrics[m]["potential"] for m in TIMEFRAME_NAMES], dtype=float)

    psi = _build_state_vector(amplitudes, phases)
    rho = _build_density_matrix(psi)
    h = _build_effective_hamiltonian(
        amplitudes,
        phases,
        returns_by_mode,
        potentials,
        coupling_strength=coupling_strength,
        phase_sensitivity=phase_sensitivity,
    )

    coherence = _compute_coherence(rho)
    phase_bias = float(np.clip(np.average(phases, weights=np.clip(amplitudes, EPS, None)), -1.0, 1.0))
    interference = _compute_interference(phases, returns_by_mode)
    energy = _compute_energy(psi, h)
    dominant_mode = _select_dominant_mode(amplitudes)

    amp_mean = float(np.mean(amplitudes)) if amplitudes.size else 0.0
    amp_dispersion = float(np.std(amplitudes)) if amplitudes.size else 0.0
    amp_dispersion = _clip01(amp_dispersion / max(amp_mean, 0.33))

    phase_gap = float(np.mean([mode_metrics[m]["phase_gap"] for m in TIMEFRAME_NAMES]))
    vol_shock = float(np.mean([mode_metrics[m]["vol_shock"] for m in TIMEFRAME_NAMES]))
    recent_vs_structural = []
    return_disagreement_terms = []
    for mode_name in TIMEFRAME_NAMES:
        recent = returns_by_mode[mode_name]
        if len(recent) < 3:
            continue
        structural = _safe_all_close_returns(mode_frames[mode_name])
        corr = _safe_correlation(recent, structural)
        return_disagreement_terms.append((1.0 - ((corr + 1.0) / 2.0)))
        recent_vs_structural.append(abs(mode_metrics[mode_name]["phase"] - mode_metrics[mode_name]["structural_phase"]) / 2.0)
    return_disagreement = float(np.mean(return_disagreement_terms)) if return_disagreement_terms else 0.0
    phase_gap = float(np.mean(recent_vs_structural)) if recent_vs_structural else phase_gap

    decoherence_rate = _compute_decoherence_rate(
        coherence=coherence,
        phase_gap=_clip01(phase_gap),
        vol_shock=_clip01(vol_shock),
        amplitude_dispersion=_clip01(amp_dispersion),
        return_disagreement=_clip01(return_disagreement),
    )
    transition_rate = _compute_transition_rate(
        energy=energy,
        decoherence_rate=decoherence_rate,
        phase_gap=_clip01(phase_gap),
        vol_shock=_clip01(vol_shock),
        amplitude_dispersion=_clip01(amp_dispersion),
    )

    tunneling_probability = _rolling_zscore_probability(
        _safe_close_series(df_m15),
        window=32,
        min_periods=16,
    )

    amplitude = _clip01(amp_mean)
    state_confidence = _clip01(
        0.35 * coherence
        + 0.25 * amplitude
        + 0.20 * (1.0 - decoherence_rate)
        + 0.20 * (1.0 - transition_rate)
    )

    state = _classify_state(
        coherence=coherence,
        phase_bias=phase_bias,
        energy=energy,
        decoherence_rate=decoherence_rate,
        transition_rate=transition_rate,
        tunneling_probability=tunneling_probability,
        amplitude=amplitude,
        energy_threshold=energy_threshold,
        transition_threshold=transition_threshold,
    )

    return QuantumState(
        state=state,
        coherence=round(coherence, 4),
        phase_bias=round(phase_bias, 4),
        interference=round(interference, 4),
        tunneling_probability=round(tunneling_probability, 4),
        amplitude=round(amplitude, 4),
        state_confidence=round(state_confidence, 4),
        energy=round(energy, 4),
        decoherence_rate=round(decoherence_rate, 4),
        transition_rate=round(transition_rate, 4),
        dominant_mode=dominant_mode,
    )
