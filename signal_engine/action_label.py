def suggest_action(
    context: str,
    setup_info: str,
    why: str,
    combined_bias: str,
    squeeze_risk: str = "N/A",
) -> str:
    """
    Returns a compact action label for the operator.
    """

    base_action = "STAND_BY"

    if setup_info.startswith("BREAKOUT_CONFIRMATION") or setup_info.startswith("TREND_PULLBACK") or setup_info.startswith("SWEEP_RECLAIM"):
        base_action = "READY_IF_TRIGGER"

    elif context == "chop":
        base_action = "AVOID_CHOP"

    elif context == "trend_extended":
        if "no pullback in value zone" in why:
            base_action = "WAIT_PULLBACK"
        elif "no breakout confirmation" in why:
            base_action = "WAIT_BREAKOUT"
        else:
            base_action = "WAIT_RESET"

    elif context == "trend_clean":
        if "no breakout confirmation" in why:
            base_action = "WAIT_BREAKOUT"
        elif "no pullback in value zone" in why:
            base_action = "WAIT_PULLBACK"
        else:
            base_action = "WATCH_CLOSELY"

    elif context == "transition":
        base_action = "WAIT_STRUCTURE"

    elif combined_bias == "neutral":
        base_action = "NO_BIAS"

    # Overlay caution if liquidation risk is high
    if squeeze_risk == "SQUEEZE RISK HIGH":
        if base_action in ("READY_IF_TRIGGER", "WAIT_BREAKOUT", "WAIT_PULLBACK", "WATCH_CLOSELY"):
            return base_action + "_CAUTION"

    if squeeze_risk == "SQUEEZE RISK MEDIUM":
        if base_action == "READY_IF_TRIGGER":
            return "READY_IF_TRIGGER_CAUTION"

    return base_action