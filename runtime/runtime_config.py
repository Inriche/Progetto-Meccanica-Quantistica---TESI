import json
import os
from typing import Any, Dict

from config import CONFIG


RUNTIME_CONFIG_PATH = "out/config_runtime.json"


DEFAULT_RUNTIME_CONFIG = {
    "rr_min": float(CONFIG.rr_min),
    "min_score_for_signal": int(CONFIG.min_score_for_signal),
    "max_signals_per_day": int(CONFIG.max_signals_per_day),
    "cooldown_minutes": int(CONFIG.cooldown_minutes),

    "orderbook_neutral_threshold": 0.10,
    "orderbook_full_score_threshold": 0.60,

    "squeeze_risk_high_pct": 0.002,
    "squeeze_risk_medium_pct": 0.005,

    "derivatives_extreme_oi_pct": 0.02,
    "derivatives_mild_oi_pct": 0.01,

    "quantum_coherence_threshold": 0.62,
    "quantum_tunneling_threshold": 0.70,

    "strategy_mode": "BALANCED",
    "news_enabled": True,
    "news_cache_minutes": 15,
    "news_headline_limit": 6,
}


PRESET_CONFIGS = {
    "Conservative": {
        "rr_min": 1.8,
        "min_score_for_signal": 78,
        "max_signals_per_day": 2,
        "cooldown_minutes": 90,
        "orderbook_neutral_threshold": 0.12,
        "orderbook_full_score_threshold": 0.65,
        "squeeze_risk_high_pct": 0.0020,
        "squeeze_risk_medium_pct": 0.0045,
        "derivatives_extreme_oi_pct": 0.018,
        "derivatives_mild_oi_pct": 0.009,
        "quantum_coherence_threshold": 0.68,
        "quantum_tunneling_threshold": 0.74,
        "strategy_mode": "TREND_FOLLOWER",
        "news_enabled": True,
        "news_cache_minutes": 15,
        "news_headline_limit": 6,
    },
    "Balanced": {
        "rr_min": 1.5,
        "min_score_for_signal": 70,
        "max_signals_per_day": 2,
        "cooldown_minutes": 60,
        "orderbook_neutral_threshold": 0.10,
        "orderbook_full_score_threshold": 0.60,
        "squeeze_risk_high_pct": 0.0020,
        "squeeze_risk_medium_pct": 0.0050,
        "derivatives_extreme_oi_pct": 0.020,
        "derivatives_mild_oi_pct": 0.010,
        "quantum_coherence_threshold": 0.62,
        "quantum_tunneling_threshold": 0.70,
        "strategy_mode": "BALANCED",
        "news_enabled": True,
        "news_cache_minutes": 15,
        "news_headline_limit": 6,
    },
    "Aggressive": {
        "rr_min": 1.3,
        "min_score_for_signal": 62,
        "max_signals_per_day": 4,
        "cooldown_minutes": 30,
        "orderbook_neutral_threshold": 0.08,
        "orderbook_full_score_threshold": 0.50,
        "squeeze_risk_high_pct": 0.0015,
        "squeeze_risk_medium_pct": 0.0040,
        "derivatives_extreme_oi_pct": 0.025,
        "derivatives_mild_oi_pct": 0.012,
        "quantum_coherence_threshold": 0.56,
        "quantum_tunneling_threshold": 0.64,
        "strategy_mode": "BREAKOUT_SURFER",
        "news_enabled": True,
        "news_cache_minutes": 10,
        "news_headline_limit": 8,
    },
}


def ensure_runtime_config_file() -> None:
    os.makedirs("out", exist_ok=True)
    if not os.path.exists(RUNTIME_CONFIG_PATH):
        with open(RUNTIME_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_RUNTIME_CONFIG, f, indent=2)


def load_runtime_config() -> Dict[str, Any]:
    ensure_runtime_config_file()

    try:
        with open(RUNTIME_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        merged = DEFAULT_RUNTIME_CONFIG.copy()
        merged.update(data)
        return merged
    except Exception:
        return DEFAULT_RUNTIME_CONFIG.copy()


def save_runtime_config(data: Dict[str, Any]) -> None:
    os.makedirs("out", exist_ok=True)

    clean = DEFAULT_RUNTIME_CONFIG.copy()
    clean.update(data)

    with open(RUNTIME_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)


def reset_runtime_config() -> None:
    save_runtime_config(DEFAULT_RUNTIME_CONFIG.copy())


def apply_preset(name: str) -> Dict[str, Any]:
    preset = PRESET_CONFIGS.get(name)
    if preset is None:
        return load_runtime_config()

    cfg = DEFAULT_RUNTIME_CONFIG.copy()
    cfg.update(preset)
    save_runtime_config(cfg)
    return cfg


def describe_runtime_profile(cfg: Dict[str, Any]) -> Dict[str, str]:
    rr_min = float(cfg["rr_min"])
    min_score = int(cfg["min_score_for_signal"])
    ob_neutral = float(cfg["orderbook_neutral_threshold"])
    squeeze_high = float(cfg["squeeze_risk_high_pct"])
    quantum_coherence = float(cfg["quantum_coherence_threshold"])

    strictness_score = 0

    if rr_min >= 1.8:
        strictness_score += 2
    elif rr_min >= 1.5:
        strictness_score += 1

    if min_score >= 78:
        strictness_score += 2
    elif min_score >= 70:
        strictness_score += 1

    if ob_neutral >= 0.12:
        strictness_score += 1

    if squeeze_high <= 0.002:
        strictness_score += 1

    if quantum_coherence >= 0.66:
        strictness_score += 1

    if strictness_score >= 5:
        profile = "Strict"
        description = "The engine is currently selective and conservative. Expect fewer but cleaner candidates."
    elif strictness_score >= 3:
        profile = "Balanced"
        description = "The engine is moderately selective. Good compromise between quality and frequency."
    else:
        profile = "Permissive"
        description = "The engine is currently permissive. Expect more candidates, more noise, and more blocked setups."

    return {
        "profile": profile,
        "description": description,
    }
