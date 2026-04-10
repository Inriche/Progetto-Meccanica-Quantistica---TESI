from features.quantum_state import QuantumState


def quantum_points(
    quantum: QuantumState,
    decision: str,
    coherence_threshold: float = 0.62,
    tunneling_threshold: float = 0.70,
) -> int:
    if decision not in ("BUY", "SELL"):
        return 0

    if quantum.state == "WARMING_UP":
        return 0

    pts = 0
    directional_bias = float(quantum.phase_bias)
    signed_bias = directional_bias if decision == "BUY" else -directional_bias
    signed_interference = float(quantum.interference) if decision == "BUY" else -float(quantum.interference)

    if quantum.coherence >= coherence_threshold:
        pts += 4
    elif quantum.coherence < 0.42:
        pts -= 6

    if signed_bias > 0.35:
        pts += 6
    elif signed_bias > 0.15:
        pts += 3
    elif signed_bias < -0.25:
        pts -= 7

    if signed_interference > 0.10:
        pts += 3
    elif signed_interference < -0.10:
        pts -= 4

    if quantum.tunneling_probability >= tunneling_threshold:
        pts += 3

    if decision == "BUY" and quantum.state in ("COHERENT_BULLISH", "BULLISH_TUNNEL"):
        pts += 4
    elif decision == "SELL" and quantum.state in ("COHERENT_BEARISH", "BEARISH_TUNNEL"):
        pts += 4
    elif quantum.state == "DECOHERENT":
        pts -= 4

    return max(-12, min(12, int(round(pts))))
