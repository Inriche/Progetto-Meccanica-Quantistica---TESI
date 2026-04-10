from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class StrategyProfile:
    code: str
    label: str
    description: str
    setup_priority: Tuple[str, ...]
    preferred_setups: Tuple[str, ...]
    news_weight: float
    quantum_weight: float
    liquidity_weight: float


STRATEGY_PROFILES: Dict[str, StrategyProfile] = {
    "BALANCED": StrategyProfile(
        code="BALANCED",
        label="Balanced",
        description="Blends pullback, liquidity sweep and breakout logic with moderate sensitivity to quantum and news inputs.",
        setup_priority=("TREND_PULLBACK", "SWEEP_RECLAIM", "BREAKOUT_CONFIRMATION"),
        preferred_setups=("TREND_PULLBACK", "SWEEP_RECLAIM"),
        news_weight=1.0,
        quantum_weight=1.0,
        liquidity_weight=1.0,
    ),
    "LIQUIDITY_HUNTER": StrategyProfile(
        code="LIQUIDITY_HUNTER",
        label="Liquidity Hunter",
        description="Prioritizes sweep/reclaim behavior and nearby liquidation clusters before trend continuation setups.",
        setup_priority=("SWEEP_RECLAIM", "TREND_PULLBACK", "BREAKOUT_CONFIRMATION"),
        preferred_setups=("SWEEP_RECLAIM",),
        news_weight=0.8,
        quantum_weight=0.9,
        liquidity_weight=1.4,
    ),
    "TREND_FOLLOWER": StrategyProfile(
        code="TREND_FOLLOWER",
        label="Trend Follower",
        description="Prefers orderly trend continuation, cleaner structure and confirmation over reactive sweeps.",
        setup_priority=("TREND_PULLBACK", "BREAKOUT_CONFIRMATION", "SWEEP_RECLAIM"),
        preferred_setups=("TREND_PULLBACK", "BREAKOUT_CONFIRMATION"),
        news_weight=1.0,
        quantum_weight=1.0,
        liquidity_weight=0.8,
    ),
    "BREAKOUT_SURFER": StrategyProfile(
        code="BREAKOUT_SURFER",
        label="Breakout Surfer",
        description="Looks for expansion moves with tunneling-like acceleration and breakout confirmation leading the stack.",
        setup_priority=("BREAKOUT_CONFIRMATION", "TREND_PULLBACK", "SWEEP_RECLAIM"),
        preferred_setups=("BREAKOUT_CONFIRMATION",),
        news_weight=1.1,
        quantum_weight=1.1,
        liquidity_weight=0.8,
    ),
    "QUANTUM_HYBRID": StrategyProfile(
        code="QUANTUM_HYBRID",
        label="Quantum Hybrid",
        description="Weights coherence, phase alignment and news catalysts more heavily before allowing a directional idea.",
        setup_priority=("TREND_PULLBACK", "BREAKOUT_CONFIRMATION", "SWEEP_RECLAIM"),
        preferred_setups=("TREND_PULLBACK", "BREAKOUT_CONFIRMATION"),
        news_weight=1.2,
        quantum_weight=1.4,
        liquidity_weight=0.9,
    ),
}


def get_strategy_profile(code: str) -> StrategyProfile:
    if code in STRATEGY_PROFILES:
        return STRATEGY_PROFILES[code]
    return STRATEGY_PROFILES["BALANCED"]


def list_strategy_codes() -> List[str]:
    return list(STRATEGY_PROFILES.keys())


def strategy_points(
    profile: StrategyProfile,
    setup_name: str,
    decision: str,
    context: str,
    squeeze_risk: str,
    latest_price,
    liquidation_cluster,
    quantum_coherence: float,
    quantum_tunneling: float,
    quantum_phase_bias: float,
    news_sentiment: float,
    news_impact: float,
) -> tuple[int, list[str]]:
    pts = 0.0
    reasons: list[str] = []

    if setup_name in profile.preferred_setups:
        pts += 4.0
        reasons.append(f"strategy prefers {setup_name}")

    if profile.code == "LIQUIDITY_HUNTER":
        if setup_name == "SWEEP_RECLAIM":
            pts += 5.0
        elif setup_name == "BREAKOUT_CONFIRMATION":
            pts -= 3.0

        if latest_price and liquidation_cluster and latest_price > 0:
            dist_pct = abs(float(liquidation_cluster) - float(latest_price)) / float(latest_price)
            if dist_pct < 0.004:
                pts += 4.0 * profile.liquidity_weight
                reasons.append("liquidity cluster is nearby")

    elif profile.code == "TREND_FOLLOWER":
        if context == "trend_clean":
            pts += 5.0
        elif context in ("transition", "chop"):
            pts -= 5.0

        if setup_name == "TREND_PULLBACK":
            pts += 4.0
        elif setup_name == "SWEEP_RECLAIM":
            pts -= 4.0

    elif profile.code == "BREAKOUT_SURFER":
        if setup_name == "BREAKOUT_CONFIRMATION":
            pts += 6.0
        elif setup_name == "SWEEP_RECLAIM":
            pts -= 3.0

        if quantum_tunneling >= 0.68:
            pts += 4.0
            reasons.append("high tunneling probability supports breakout logic")

    elif profile.code == "QUANTUM_HYBRID":
        if quantum_coherence >= 0.65:
            pts += 4.0 * profile.quantum_weight
        if decision == "BUY" and quantum_phase_bias > 0.20:
            pts += 4.0 * profile.quantum_weight
        elif decision == "SELL" and quantum_phase_bias < -0.20:
            pts += 4.0 * profile.quantum_weight
        elif abs(quantum_phase_bias) < 0.08:
            pts -= 3.0

    if squeeze_risk == "SQUEEZE RISK HIGH" and profile.code != "LIQUIDITY_HUNTER":
        pts -= 3.0
    elif squeeze_risk == "SQUEEZE RISK MEDIUM":
        pts -= 1.0

    signed_news = float(news_sentiment) if decision == "BUY" else -float(news_sentiment)
    if news_impact >= 0.65 and signed_news > 0.20:
        pts += 3.0 * profile.news_weight
        reasons.append("news backdrop aligns with the direction")
    elif news_impact >= 0.65 and signed_news < -0.20:
        pts -= 4.0 * profile.news_weight
        reasons.append("news backdrop works against the direction")

    return int(round(max(-12.0, min(12.0, pts)))), reasons
