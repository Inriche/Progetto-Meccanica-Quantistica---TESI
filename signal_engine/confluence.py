import pandas as pd


def nearest_h1_levels(df_h1: pd.DataFrame):
    recent_high = float(df_h1["high"].tail(30).max())
    recent_low = float(df_h1["low"].tail(30).min())
    return recent_high, recent_low


def confluence_points(entry: float, decision: str, df_h1: pd.DataFrame) -> int:
    """
    Score based on proximity of entry to useful H1 structure.
    """
    if len(df_h1) < 30:
        return 0

    recent_high, recent_low = nearest_h1_levels(df_h1)
    price = entry

    # distanza percentuale dal livello
    dist_high = abs(price - recent_high) / price
    dist_low = abs(price - recent_low) / price

    if decision == "BUY":
        # buy preferito se non troppo vicino a un high già fatto
        if dist_high < 0.002:
            return -8
        if dist_low < 0.004:
            return 6
        return 2

    if decision == "SELL":
        # sell preferito se non troppo vicino a un low già fatto
        if dist_low < 0.002:
            return -8
        if dist_high < 0.004:
            return 6
        return 2

    return 0