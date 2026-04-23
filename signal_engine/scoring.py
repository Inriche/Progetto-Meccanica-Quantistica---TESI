from dataclasses import dataclass
from dataclasses import dataclass
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from runtime.runtime_config import load_runtime_config

MODEL_PATH = "out/model.joblib"
# Scoring mode:
# - "heuristic": legacy rules only
# - "hybrid": ML-weighted calibrated blend of heuristic and ML probability
# - "ml": ML-only when model is available, otherwise safe heuristic fallback
SCORING_MODE = os.getenv("SCORING_MODE", "hybrid").strip().lower()
_SCORING_MODES = {"heuristic", "hybrid", "ml"}

_ML_ARTIFACT_CACHE: Optional[Dict[str, Any]] = None
_ML_ARTIFACT_MTIME: Optional[float] = None
_ML_ARTIFACT_LAST_ATTEMPT: float = 0.0
_ML_ARTIFACT_LAST_WARNING: Optional[str] = None

logger = logging.getLogger(__name__)


@dataclass
class ScoreBreakdown:
    score: int
    grade: str
    components: List[Tuple[str, int]]
    heuristic_score: int = 0
    raw_hybrid_score: Optional[float] = None
    calibrated_hybrid_score: Optional[float] = None
    scoring_mode: str = "hybrid"


def _normalize_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    return mode if mode in _SCORING_MODES else "hybrid"


def _resolve_scoring_mode() -> str:
    env_mode = os.getenv("SCORING_MODE", "").strip().lower()
    if env_mode in _SCORING_MODES:
        return env_mode

    try:
        cfg = load_runtime_config()
        cfg_mode = _normalize_mode(cfg.get("scoring_mode"))
        if cfg_mode in _SCORING_MODES:
            return cfg_mode
    except Exception:
        pass

    return _normalize_mode(SCORING_MODE)


def _log_ml_warning(message: str) -> None:
    global _ML_ARTIFACT_LAST_WARNING
    if message != _ML_ARTIFACT_LAST_WARNING:
        logger.warning(message)
        _ML_ARTIFACT_LAST_WARNING = message


def _load_ml_artifact() -> Optional[Dict[str, Any]]:
    global _ML_ARTIFACT_CACHE, _ML_ARTIFACT_MTIME, _ML_ARTIFACT_LAST_ATTEMPT

    if not os.path.exists(MODEL_PATH):
        if _ML_ARTIFACT_CACHE is None:
            _log_ml_warning(f"[scoring] ML artifact missing at {MODEL_PATH}; hybrid fields unavailable until the model is trained.")
        return _ML_ARTIFACT_CACHE

    try:
        current_mtime = os.path.getmtime(MODEL_PATH)
    except OSError:
        current_mtime = None

    if _ML_ARTIFACT_CACHE is not None and current_mtime is not None and _ML_ARTIFACT_MTIME == current_mtime:
        return _ML_ARTIFACT_CACHE

    now = time.monotonic()
    if _ML_ARTIFACT_CACHE is None and (now - _ML_ARTIFACT_LAST_ATTEMPT) < 2.0:
        return None
    _ML_ARTIFACT_LAST_ATTEMPT = now

    try:
        import joblib

        artifact = joblib.load(MODEL_PATH)
        if isinstance(artifact, dict) and "pipeline" in artifact:
            _ML_ARTIFACT_CACHE = artifact
            _ML_ARTIFACT_MTIME = current_mtime
            return artifact
        _log_ml_warning(
            f"[scoring] Invalid ML artifact at {MODEL_PATH}; expected a dict with a pipeline."
        )
    except Exception as exc:
        _log_ml_warning(f"[scoring] Failed to load ML artifact at {MODEL_PATH}: {exc!r}")

    return _ML_ARTIFACT_CACHE


def _predict_ml_probability(feature_row: Dict[str, Any]) -> Optional[float]:
    artifact = _load_ml_artifact()
    if artifact is None:
        _log_ml_warning("[scoring] ML probability unavailable because the artifact could not be loaded.")
        return None

    pipeline = artifact.get("pipeline")
    feature_columns = artifact.get("feature_columns", [])
    if pipeline is None or not feature_columns:
        _log_ml_warning("[scoring] ML probability unavailable because the artifact is missing a pipeline or feature columns.")
        return None

    try:
        data = {col: feature_row.get(col) for col in feature_columns}
        X = pd.DataFrame([data])
        proba = pipeline.predict_proba(X)
        if proba is None or len(proba) == 0:
            return None

        classes = getattr(pipeline, "classes_", None)
        if classes is None:
            classifier = getattr(pipeline, "named_steps", {}).get("classifier")
            classes = getattr(classifier, "classes_", None)
        if classes is None:
            _log_ml_warning("[scoring] ML probability unavailable because classifier classes could not be resolved.")
            return None

        classes_list = list(classes)
        if 1 not in classes_list:
            _log_ml_warning("[scoring] ML probability unavailable because class 1 is not present in the classifier outputs.")
            return None
        positive_idx = classes_list.index(1)
        if positive_idx >= len(proba[0]):
            _log_ml_warning("[scoring] ML probability unavailable because the positive class index is outside the predict_proba output.")
            return None
        return float(proba[0][positive_idx])
    except Exception as exc:
        _log_ml_warning(f"[scoring] ML probability unavailable because predict_proba failed: {exc!r}")
        return None


def _bounded_score(value: float) -> int:
    if not np.isfinite(value):
        return 0
    return max(0, min(100, int(round(float(value)))))


def compute_score(
    bias_h1: str,
    bias_h4: str,
    combined: str,
    rr_est: float,
    volatility: str,
    setup_name: str,
    context: str = "transition",
    decision: str = "UNKNOWN",
    ml_features: Optional[Dict[str, Any]] = None,
    extra_heuristic_points: int = 0,
) -> ScoreBreakdown:
    score = 0
    comps: List[Tuple[str, int]] = []

    # Bias alignment
    if combined in ("bullish", "bearish"):
        score += 20
        comps.append(("bias_alignment", 20))
    else:
        score -= 10
        comps.append(("bias_alignment", -10))

    # Volatility regime
    if volatility == "normal":
        score += 10
        comps.append(("volatility_ok", 10))
    elif volatility == "low":
        score += 5
        comps.append(("volatility_low", 5))
    elif volatility == "high":
        score -= 10
        comps.append(("volatility_high_penalty", -10))

    # RR quality
    if rr_est >= 2.0:
        score += 15
        comps.append(("rr_ok", 15))
    elif rr_est >= 1.5:
        score += 8
        comps.append(("rr_ok_mid", 8))
    else:
        score -= 20
        comps.append(("rr_fail_penalty", -20))

    # Setup quality
    if setup_name == "SWEEP_RECLAIM":
        score += 25
        comps.append(("setup_quality", 25))
    elif setup_name == "TREND_PULLBACK":
        score += 20
        comps.append(("setup_quality", 20))
    elif setup_name == "BREAKOUT_CONFIRMATION":
        score += 18
        comps.append(("setup_quality", 18))

    # Market context
    if context == "trend_clean":
        score += 10
        comps.append(("context_trend_clean", 10))
    elif context == "trend_extended":
        score -= 8
        comps.append(("context_trend_extended", -8))
    elif context == "chop":
        score -= 15
        comps.append(("context_chop", -15))
    elif context == "transition":
        score -= 5
        comps.append(("context_transition", -5))

    if extra_heuristic_points != 0:
        score += int(extra_heuristic_points)
        comps.append(("contextual_adjustments", int(extra_heuristic_points)))

    heuristic_score = max(0, min(100, score))
    # Keep this score available for "full" feature-sets without circular dependence.
    ml_payload: Dict[str, Any] = {
        "rr_estimated": rr_est,
        "setup": setup_name,
        "context": context,
        "decision": decision,
        "score": heuristic_score,
        "action": None,
        "ob_imbalance": None,
        "ob_raw": None,
        "ob_age_ms": None,
        "funding_rate": None,
        "oi_now": None,
        "oi_change_pct": None,
        "crowding": None,
        "strategy_mode": None,
        "strategy_score": None,
        "news_bias": None,
        "news_sentiment": None,
        "news_impact": None,
        "news_score": None,
        "quantum_state": None,
        "quantum_coherence": None,
        "quantum_phase_bias": None,
        "quantum_interference": None,
        "quantum_tunneling": None,
        "quantum_energy": None,
        "quantum_decoherence_rate": None,
        "quantum_transition_rate": None,
        "quantum_dominant_mode": None,
        "quantum_score": None,
    }
    if ml_features:
        ml_payload.update(ml_features)
        if ml_payload.get("score") is None:
            ml_payload["score"] = heuristic_score

    mode = _resolve_scoring_mode()
    raw_hybrid_score: Optional[float] = None
    calibrated_hybrid_score: Optional[float] = None
    ml_probability = _predict_ml_probability(ml_payload)
    effective_mode = mode
    if ml_probability is not None:
        heuristic_norm = float(np.clip(heuristic_score / 100.0, 0.0, 1.0))
        ml_prob = float(np.clip(ml_probability, 0.0, 1.0))
        legacy_hybrid_score = (0.65 * heuristic_score) + (0.35 * (ml_prob * 100.0))
        raw_hybrid_score = float(legacy_hybrid_score)
        raw_hybrid_probability = (0.20 * heuristic_norm) + (0.80 * ml_prob)
        calibrated_hybrid_score = raw_hybrid_probability * 100.0
    elif mode in ("hybrid", "ml"):
        effective_mode = "heuristic_fallback"

    if mode == "heuristic":
        score = heuristic_score
    elif ml_probability is None:
        score = heuristic_score
    elif mode == "ml":
        score = _bounded_score(float(ml_probability) * 100.0)
        comps.append(("ml_adjustment", int(score - heuristic_score)))
    else:
        # Keep the legacy blend as a reference score, but use a more ML-driven
        # calibrated score for thresholding because it tracks outcomes better.
        score = _bounded_score(calibrated_hybrid_score)
        comps.append(("ml_adjustment", int(score - heuristic_score)))

    grade = "A" if score >= 80 else ("B" if score >= 70 else "C")

    return ScoreBreakdown(
        score=score,
        grade=grade,
        components=comps,
        heuristic_score=heuristic_score,
        raw_hybrid_score=raw_hybrid_score,
        calibrated_hybrid_score=calibrated_hybrid_score,
        scoring_mode=effective_mode,
    )
