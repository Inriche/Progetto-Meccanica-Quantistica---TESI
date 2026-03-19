def derivatives_points(
    funding_rate,
    oi_change_pct,
    crowding: str,
    decision: str,
    extreme_oi_pct: float = 0.02,
    mild_oi_pct: float = 0.01,
) -> int:
    pts = 0

    if funding_rate is None or oi_change_pct is None or crowding is None:
        return 0

    if decision == "SELL":
        if crowding == "crowded_shorts":
            pts -= 6
        elif crowding == "crowded_longs":
            pts += 4

    elif decision == "BUY":
        if crowding == "crowded_longs":
            pts -= 6
        elif crowding == "crowded_shorts":
            pts += 4

    if abs(oi_change_pct) > extreme_oi_pct:
        pts -= 2
    elif abs(oi_change_pct) > mild_oi_pct:
        pts -= 1

    if decision == "SELL" and funding_rate > 0.0001:
        pts += 2
    elif decision == "BUY" and funding_rate < -0.0001:
        pts += 2

    return pts