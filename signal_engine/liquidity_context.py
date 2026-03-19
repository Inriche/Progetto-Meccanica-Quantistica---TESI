from typing import Optional


def squeeze_risk_label_from_prices(
    price: Optional[float],
    liquidation_cluster: Optional[float],
    high_pct: float = 0.002,
    medium_pct: float = 0.005,
) -> str:
    if price is None or liquidation_cluster is None or price <= 0:
        return "N/A"

    dist_pct = abs(liquidation_cluster - price) / price

    if dist_pct < high_pct:
        return "SQUEEZE RISK HIGH"

    if dist_pct < medium_pct:
        return "SQUEEZE RISK MEDIUM"

    return "SQUEEZE RISK LOW"