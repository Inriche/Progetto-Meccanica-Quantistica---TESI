from features.quantum_state import QuantumState


def quantum_points(
    quantum: QuantumState,
    decision: str,
    coherence_threshold: float = 0.62,
    tunneling_threshold: float = 0.70,
    energy_threshold: float = 0.45,
    decoherence_penalty: float = 4.0,
    transition_threshold: float = 0.55,
    phase_sensitivity: float = 1.0,
    coupling_strength: float = 0.65,
) -> int:
    if decision not in ("BUY", "SELL"):
        return 0

    if quantum.state == "WARMING_UP":
        return 0

    pts = 0
    directional_bias = float(quantum.phase_bias)
    signed_bias = directional_bias if decision == "BUY" else -directional_bias
    signed_interference = float(quantum.interference) if decision == "BUY" else -float(quantum.interference)
    coherence = float(quantum.coherence)
    energy = float(getattr(quantum, "energy", 0.0))
    decoherence_rate = float(getattr(quantum, "decoherence_rate", 0.0))
    transition_rate = float(getattr(quantum, "transition_rate", 0.0))
    dominant_mode = str(getattr(quantum, "dominant_mode", "m15") or "m15").lower()
    dominant_mode_weight = {
        "m15": 1.0,
        "h1": 0.7,
        "h4": 0.5,
    }.get(dominant_mode, 0.7)
    phase_scale = max(0.5, float(phase_sensitivity))
    coupling_scale = max(0.25, float(coupling_strength))

    if coherence >= coherence_threshold:
        pts += 4
    elif coherence < 0.42:
        pts -= 6

    if signed_bias > 0.35:
        pts += 5
    elif signed_bias > 0.15:
        pts += 3
    elif signed_bias < -0.25:
        pts -= 6

    if signed_interference > 0.10:
        pts += 2
    elif signed_interference < -0.10:
        pts -= 3

    if energy <= energy_threshold and coherence >= 0.58 and abs(signed_bias) > 0.08:
        pts += 2
    elif energy >= 0.72:
        pts -= 2

    if decoherence_rate >= 0.65:
        pts -= int(round(max(3.0, decoherence_penalty)))
    elif decoherence_rate >= 0.45:
        pts -= int(round(max(1.5, 0.5 * float(decoherence_penalty))))

    if transition_rate >= transition_threshold:
        if quantum.state in ("BULLISH_TUNNEL", "BEARISH_TUNNEL"):
            pts += 3
        elif quantum.state == "TRANSITIONAL":
            pts += 1
        else:
            pts -= 1
    elif transition_rate < 0.22 and energy > 0.65:
        pts -= 1

    if quantum.tunneling_probability >= tunneling_threshold:
        pts += 2

    if decision == "BUY" and quantum.state in ("COHERENT_BULLISH", "BULLISH_TUNNEL"):
        pts += 3
    elif decision == "SELL" and quantum.state in ("COHERENT_BEARISH", "BEARISH_TUNNEL"):
        pts += 3
    elif quantum.state == "DECOHERENT":
        pts -= 4

    if dominant_mode == "m15" and abs(signed_bias) > 0.15:
        pts += int(round(1.5 * dominant_mode_weight))
    elif dominant_mode == "h4" and coherence >= 0.68:
        pts += int(round(1.0 * dominant_mode_weight))
    elif dominant_mode == "h4" and transition_rate >= transition_threshold and energy <= 0.60:
        pts += int(round(2.0 * dominant_mode_weight))

    if energy <= energy_threshold and transition_rate <= transition_threshold and coherence >= 0.60:
        pts += 1

    # Keep the additive score interpretable and avoid noisy overreaction.
    pts = int(round(pts * (0.95 + 0.03 * phase_scale + 0.02 * coupling_scale)))

    return max(-12, min(12, int(round(pts))))
