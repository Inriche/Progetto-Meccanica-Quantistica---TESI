from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ScoreBreakdown:
    score: int
    grade: str
    components: List[Tuple[str, int]]


def compute_score(
    bias_h1: str,
    bias_h4: str,
    combined: str,
    rr_est: float,
    volatility: str,
    setup_name: str,
    context: str = "transition",
) -> ScoreBreakdown:
    score = 0
    comps = []

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

    score = max(0, min(100, score))
    grade = "A" if score >= 80 else ("B" if score >= 70 else "C")

    return ScoreBreakdown(score=score, grade=grade, components=comps)