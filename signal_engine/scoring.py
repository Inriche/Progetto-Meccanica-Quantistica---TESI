from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

MODEL_PATH = "out/model.joblib"
# Scoring mode:
# - "heuristic": legacy rules only
# - "hybrid": blend heuristic + ML probability
# - "ml": ML-only when model is available, otherwise safe heuristic fallback
SCORING_MODE = os.getenv("SCORING_MODE", "hybrid").strip().lower()

_ML_ARTIFACT_CACHE: Optional[Dict[str, Any]] = None
_ML_ARTIFACT_LOAD_ATTEMPTED = False


@dataclass
class ScoreBreakdown:
    score: int
    grade: str
    components: List[Tuple[str, int]]


def _load_ml_artifact() -> Optional[Dict[str, Any]]:
    global _ML_ARTIFACT_CACHE, _ML_ARTIFACT_LOAD_ATTEMPTED
    if _ML_ARTIFACT_LOAD_ATTEMPTED:
        return _ML_ARTIFACT_CACHE

    _ML_ARTIFACT_LOAD_ATTEMPTED = True
    if not os.path.exists(MODEL_PATH):
        _ML_ARTIFACT_CACHE = None
        return None

    try:
        import joblib

        artifact = joblib.load(MODEL_PATH)
        if isinstance(artifact, dict) and "pipeline" in artifact:
            _ML_ARTIFACT_CACHE = artifact
            return artifact
    except Exception:
        _ML_ARTIFACT_CACHE = None
        return None

    _ML_ARTIFACT_CACHE = None
    return None


def _predict_ml_probability(feature_row: Dict[str, Any]) -> Optional[float]:
    artifact = _load_ml_artifact()
    if artifact is None:
        return None

    pipeline = artifact.get("pipeline")
    feature_columns = artifact.get("feature_columns", [])
    if pipeline is None or not feature_columns:
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
            return None

        classes_list = list(classes)
        if 1 not in classes_list:
            return None
        positive_idx = classes_list.index(1)
        if positive_idx >= len(proba[0]):
            return None
        return float(proba[0][positive_idx])
    except Exception:
        return None


def compute_score(
    bias_h1: str,
    bias_h4: str,
    combined: str,
    rr_est: float,
    volatility: str,
    setup_name: str,
    context: str = "transition",
    decision: str = "UNKNOWN",
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

    heuristic_score = max(0, min(100, score))

    ml_probability = _predict_ml_probability(
        {
            "rr_estimated": rr_est,
            "setup": setup_name,
            "context": context,
            "decision": decision,
        }
    )

    mode = SCORING_MODE if SCORING_MODE in {"heuristic", "hybrid", "ml"} else "hybrid"
    if mode == "heuristic":
        score = heuristic_score
    elif ml_probability is None:
        score = heuristic_score
    elif mode == "ml":
        score = max(0, min(100, int(round(float(ml_probability) * 100.0))))
        comps.append(("ml_adjustment", int(score - heuristic_score)))
    else:
        ml_score = max(0, min(100, int(round(float(ml_probability) * 100.0))))
        final_score = int(round((0.65 * heuristic_score) + (0.35 * ml_score)))
        comps.append(("ml_adjustment", int(final_score - heuristic_score)))
        score = max(0, min(100, final_score))

    grade = "A" if score >= 80 else ("B" if score >= 70 else "C")

    return ScoreBreakdown(score=score, grade=grade, components=comps)
