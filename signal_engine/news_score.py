from market_data.news_context import NewsContext


def news_points(
    news: NewsContext,
    decision: str,
    enabled: bool = True,
) -> int:
    if not enabled or decision not in ("BUY", "SELL"):
        return 0

    if news.bias == "disabled":
        return 0

    sentiment = float(news.sentiment_score)
    impact = float(news.impact_score)

    signed_sentiment = sentiment if decision == "BUY" else -sentiment
    pts = 0.0

    if signed_sentiment > 0.28:
        pts += 5.0
    elif signed_sentiment > 0.12:
        pts += 2.0
    elif signed_sentiment < -0.28:
        pts -= 6.0
    elif signed_sentiment < -0.12:
        pts -= 3.0

    if impact >= 0.72 and signed_sentiment < -0.10:
        pts -= 2.0
    elif impact >= 0.72 and signed_sentiment > 0.10:
        pts += 1.0

    if news.stale:
        pts *= 0.5

    return int(round(max(-8.0, min(8.0, pts))))
