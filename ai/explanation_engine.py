def generate_explanation(
    price: float,
    bias_h1: str,
    bias_h4: str,
    combined_bias: str,
    context: str,
    volatility: str,
    setup_info: str,
    action: str,
    why: str,
    ob_avg,
    ob_raw,
    quantum_state: str = "N/A",
    quantum_coherence=None,
    quantum_phase_bias=None,
    quantum_tunneling=None,
    quantum_energy=None,
    quantum_decoherence_rate=None,
    quantum_transition_rate=None,
    quantum_dominant_mode=None,
) -> str:
    parts = []

    parts.append(
        f"Market status: price={price:.2f}, H1={bias_h1}, H4={bias_h4}, combined bias={combined_bias}, context={context}, volatility={volatility}."
    )

    if ob_avg is not None and ob_raw is not None:
        parts.append(
            f"Order book: average imbalance={ob_avg:.3f}, raw imbalance={ob_raw:.3f}."
        )
    else:
        parts.append("Order book: not available.")

    quantum_bits = [f"state={quantum_state}"]
    if quantum_coherence is not None:
        quantum_bits.append(f"coherence={quantum_coherence:.2f}")
    if quantum_phase_bias is not None:
        quantum_bits.append(f"phase_bias={quantum_phase_bias:.2f}")
    if quantum_tunneling is not None:
        quantum_bits.append(f"tunneling={quantum_tunneling:.2f}")
    if quantum_energy is not None:
        quantum_bits.append(f"energy={quantum_energy:.2f}")
    if quantum_decoherence_rate is not None:
        quantum_bits.append(f"decoherence={quantum_decoherence_rate:.2f}")
    if quantum_transition_rate is not None:
        quantum_bits.append(f"transition={quantum_transition_rate:.2f}")
    if quantum_dominant_mode is not None:
        quantum_bits.append(f"dominant_mode={quantum_dominant_mode}")
    parts.append("Quantum layer: " + ", ".join(quantum_bits) + ".")

    if setup_info == "no_setup":
        parts.append("No valid setup is currently active.")
    else:
        parts.append(f"Current setup status: {setup_info}.")

    parts.append(f"Suggested action: {action}.")
    parts.append(f"Reason: {why}.")

    if context == "trend_extended":
        parts.append(
            "Interpretation: the trend is still directional, but already extended, so chasing price is discouraged."
        )
    elif context == "trend_clean":
        parts.append(
            "Interpretation: the trend structure is clean, so valid pullbacks or breakouts deserve attention."
        )
    elif context == "chop":
        parts.append(
            "Interpretation: the market is choppy, so signals are less reliable and patience is preferred."
        )
    elif context == "transition":
        parts.append(
            "Interpretation: the market is transitioning, so structure confirmation is needed before acting."
        )

    return " ".join(parts)
