from typing import Optional, Tuple


def squeeze_risk_label(
    price: Optional[float],
    liquidation_cluster: Optional[float],
    high_pct: float = 0.002,
    medium_pct: float = 0.005,
) -> Tuple[str, str]:
    """
    Returns (label, icon) based on distance between price and nearest liquidation cluster.
    """

    if price is None or liquidation_cluster is None or price <= 0:
        return "N/A", "⚪"

    dist_pct = abs(liquidation_cluster - price) / price

    if dist_pct < high_pct:
        return "SQUEEZE RISK HIGH", "🔴"

    if dist_pct < medium_pct:
        return "SQUEEZE RISK MEDIUM", "🟠"

    return "SQUEEZE RISK LOW", "🟢"


def funding_bias_label(funding_rate):
    if funding_rate is None:
        return "N/A", "⚪"
    if funding_rate > 0.0001:
        return "Bullish / longs paying", "🟢"
    if funding_rate < -0.0001:
        return "Bearish / shorts paying", "🔴"
    return "Flat funding", "⚪"


def oi_momentum_label(oi_change_pct):
    if oi_change_pct is None:
        return "N/A", "⚪"
    if oi_change_pct > 0.01:
        return "OI rising fast", "🟠"
    if oi_change_pct > 0.002:
        return "OI rising", "🟡"
    if oi_change_pct < -0.01:
        return "OI dropping fast", "🔵"
    if oi_change_pct < -0.002:
        return "OI dropping", "🔹"
    return "OI flat", "⚪"


def structure_read_label(context: str) -> tuple[str, str]:
    if context == "trend_clean":
        return "Clean trend", "🟢"
    if context == "trend_extended":
        return "Extended trend", "🟠"
    if context == "transition":
        return "Transition", "🔵"
    if context == "chop":
        return "Chop / noisy", "🔴"
    return "Unknown", "⚪"


def orderbook_read_label(imbalance) -> tuple[str, str]:
    if imbalance is None:
        return "No data", "⚪"
    try:
        x = float(imbalance)
    except Exception:
        return "No data", "⚪"

    if x > 0.35:
        return "Buy pressure", "🟢"
    if x < -0.35:
        return "Sell pressure", "🔴"
    return "Balanced", "⚪"


def derivatives_read_label(crowding: str, funding_rate, oi_change_pct) -> tuple[str, str]:
    # simple readable summary
    if crowding == "crowded_longs":
        return "Crowded longs", "🟠"
    if crowding == "crowded_shorts":
        return "Crowded shorts", "🟠"

    if oi_change_pct is not None:
        try:
            oi = float(oi_change_pct)
            if oi < -0.002:
                return "OI dropping", "🔹"
            if oi > 0.002:
                return "OI rising", "🟡"
        except Exception:
            pass

    if funding_rate is not None:
        try:
            fr = float(funding_rate)
            if fr > 0.0001:
                return "Positive funding", "🟢"
            if fr < -0.0001:
                return "Negative funding", "🔴"
        except Exception:
            pass

    return "Neutral derivatives", "⚪"


def quantum_read_label(quantum_state: str, quantum_coherence, quantum_phase_bias) -> tuple[str, str]:
    if not quantum_state or quantum_state == "N/A":
        return "Quantum state unavailable", "⚪"

    if quantum_state in ("COHERENT_BULLISH", "BULLISH_TUNNEL"):
        return "Quantum bullish coherence", "🟢"
    if quantum_state in ("COHERENT_BEARISH", "BEARISH_TUNNEL"):
        return "Quantum bearish coherence", "🔴"
    if quantum_state == "DECOHERENT":
        return "Quantum decoherence", "🟠"
    if quantum_state == "LOW_ENERGY":
        return "Quantum low energy", "🔵"

    try:
        coherence = float(quantum_coherence)
        phase = float(quantum_phase_bias)
    except Exception:
        return str(quantum_state).replace("_", " ").title(), "⚪"

    if coherence >= 0.60:
        if phase > 0:
            return "Quantum aligned up", "🟢"
        if phase < 0:
            return "Quantum aligned down", "🔴"

    return str(quantum_state).replace("_", " ").title(), "⚪"


def grade_badge(score) -> tuple[str, str]:
    if score is None:
        return "N/A", "⚪"
    try:
        s = float(score)
    except Exception:
        return "N/A", "⚪"

    if s >= 80:
        return "A", "🟢"
    if s >= 70:
        return "B", "🟡"
    return "C", "🔴"


def rr_quality_label(rr_estimated) -> tuple[str, str]:
    if rr_estimated is None:
        return "N/A", "⚪"
    try:
        rr = float(rr_estimated)
    except Exception:
        return "N/A", "⚪"

    if rr >= 2.0:
        return "High RR", "🟢"
    if rr >= 1.5:
        return "Acceptable RR", "🟡"
    return "Low RR", "🔴"


def setup_state_label(decision: str, setup: str) -> tuple[str, str]:
    if decision == "BUY" or decision == "SELL":
        return "Signal active", "🟢"
    if setup == "BLOCKED":
        return "Candidate blocked", "🟠"
    if setup == "NONE":
        return "No setup", "⚪"
    return "Monitoring", "🔵"


def final_verdict_text(
    context: str,
    action: str,
    decision: str,
    setup: str,
    score,
    rr_estimated,
    crowding: str,
    rr_min_required: float = 1.5,
    min_score_for_signal: int = 70,
) -> tuple[str, str]:
    """
    Returns (title, body) for a final dashboard verdict.
    """

    try:
        score_val = float(score) if score is not None else None
    except Exception:
        score_val = None

    try:
        rr_val = float(rr_estimated) if rr_estimated is not None else None
    except Exception:
        rr_val = None

    if decision in ("BUY", "SELL"):
        return (
            "Signal active",
            f"{decision} setup live. Context={context}. Action={action}. Score={score_val}, RR={rr_val}.",
        )

    if setup == "BLOCKED":
        if rr_val is not None and rr_val < rr_min_required:
            return (
                "Candidate blocked",
                (
                    f"Setup exists but RR is too low ({rr_val:.2f} < min {rr_min_required:.2f}). "
                    f"Context={context}. Action={action}."
                ),
            )
        if score_val is not None and score_val < min_score_for_signal:
            return (
                "Candidate blocked",
                (
                    f"Setup exists but score is too low ({int(score_val)} < min {int(min_score_for_signal)}). "
                    f"Context={context}. Action={action}."
                ),
            )
        return (
            "Candidate blocked",
            f"Setup detected but filters blocked execution. Context={context}. Action={action}.",
        )

    if setup == "NONE":
        if context == "trend_clean":
            return (
                "No trigger yet",
                f"Structure is clean but there is no valid trigger yet. Action={action}.",
            )
        if context == "trend_extended":
            return (
                "Wait reset",
                f"Trend is extended and no clean setup is active. Action={action}.",
            )
        if context == "transition":
            return (
                "Wait structure",
                f"Market is in transition. No reliable trigger yet. Action={action}.",
            )
        if context == "chop":
            return (
                "Avoid noise",
                f"Market is choppy and unreliable. Action={action}.",
            )

    if crowding in ("crowded_longs", "crowded_shorts"):
        return (
            "Crowded derivatives",
            f"Derivatives positioning looks crowded ({crowding}). Action={action}.",
        )

    return (
        "Stand by",
        f"Current read: context={context}, action={action}, setup={setup}.",
    )
