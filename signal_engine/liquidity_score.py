from typing import Optional


def liquidation_distance_points(
    entry: Optional[float],
    decision: str,
    liquidation_cluster: Optional[float],
) -> int:
    """
    Score based on distance between entry and nearest liquidation cluster.

    Logic:
    - if cluster is too close in the direction against the trade -> penalty
    - if cluster is reasonably far -> small bonus
    - if no data -> 0
    """

    if entry is None or liquidation_cluster is None or entry <= 0:
        return 0

    dist_pct = abs(liquidation_cluster - entry) / entry

    # Very close cluster = danger zone
    if dist_pct < 0.002:   # < 0.20%
        return -10

    # Moderately close = caution
    if dist_pct < 0.004:   # < 0.40%
        return -5

    # Good space
    if dist_pct > 0.012:   # > 1.2%
        return 4

    # Neutral
    return 0