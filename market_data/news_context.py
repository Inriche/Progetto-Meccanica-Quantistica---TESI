from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import List
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

import requests


NEWS_CACHE_PATH = "out/news_cache.json"

POSITIVE_KEYWORDS = {
    "approval",
    "approved",
    "adoption",
    "inflow",
    "surge",
    "rally",
    "breakout",
    "institutional",
    "buying",
    "partnership",
    "record high",
}

NEGATIVE_KEYWORDS = {
    "hack",
    "lawsuit",
    "ban",
    "liquidation",
    "sell-off",
    "outflow",
    "crash",
    "fear",
    "investigation",
    "fraud",
    "exploit",
    "breach",
}

HIGH_IMPACT_KEYWORDS = {
    "fed",
    "cpi",
    "sec",
    "etf",
    "tariff",
    "inflation",
    "rates",
    "fomc",
    "hack",
    "lawsuit",
    "liquidation",
    "policy",
}

TOPIC_MAP = {
    "macro": ("fed", "cpi", "fomc", "inflation", "rates", "policy"),
    "regulation": ("sec", "lawsuit", "ban", "investigation"),
    "flows": ("etf", "inflow", "outflow", "institutional"),
    "risk": ("hack", "exploit", "breach", "fraud", "liquidation"),
}


@dataclass
class NewsHeadline:
    title: str
    link: str
    published_at: str
    sentiment: float
    impact: float
    topic: str


@dataclass
class NewsContext:
    bias: str
    sentiment_score: float
    impact_score: float
    headline_count: int
    dominant_topic: str
    stale: bool
    headlines: List[NewsHeadline]
    source: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["headlines"] = [asdict(h) for h in self.headlines]
        return payload


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _query_urls(symbol: str) -> List[str]:
    base_asset = symbol.upper().replace("USDT", "").replace("USD", "")
    query = quote_plus(f"{base_asset} OR Bitcoin crypto market")
    return [
        f"https://news.google.com/rss/search?q={query}+when:1d&hl=en-US&gl=US&ceid=US:en",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
    ]


def _keyword_topic(title: str) -> str:
    lower = title.lower()
    for topic, keywords in TOPIC_MAP.items():
        if any(k in lower for k in keywords):
            return topic
    return "general"


def _score_headline(title: str) -> tuple[float, float, str]:
    lower = title.lower()

    sentiment = 0.0
    impact = 0.15

    for kw in POSITIVE_KEYWORDS:
        if kw in lower:
            sentiment += 0.18
            impact += 0.06

    for kw in NEGATIVE_KEYWORDS:
        if kw in lower:
            sentiment -= 0.22
            impact += 0.08

    for kw in HIGH_IMPACT_KEYWORDS:
        if kw in lower:
            impact += 0.12

    topic = _keyword_topic(title)
    return max(-1.0, min(1.0, sentiment)), max(0.0, min(1.0, impact)), topic


def _parse_rss(xml_text: str, limit: int) -> List[NewsHeadline]:
    root = ET.fromstring(xml_text)
    items = root.findall(".//item")
    headlines: List[NewsHeadline] = []
    seen_titles: set[str] = set()

    for item in items:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        published_at = (item.findtext("pubDate") or "").strip()

        if not title or title in seen_titles:
            continue

        sentiment, impact, topic = _score_headline(title)
        headlines.append(
            NewsHeadline(
                title=title,
                link=link,
                published_at=published_at,
                sentiment=round(sentiment, 4),
                impact=round(impact, 4),
                topic=topic,
            )
        )
        seen_titles.add(title)

        if len(headlines) >= limit:
            break

    return headlines


def _save_cache(context: NewsContext) -> None:
    _ensure_parent_dir(NEWS_CACHE_PATH)
    payload = {
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "context": context.to_dict(),
    }
    with open(NEWS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_cache(max_age_minutes: int) -> NewsContext | None:
    if not os.path.exists(NEWS_CACHE_PATH):
        return None

    try:
        with open(NEWS_CACHE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)

        cached_at = datetime.fromisoformat(payload["cached_at"])
        age_minutes = (datetime.now(timezone.utc) - cached_at).total_seconds() / 60.0
        if age_minutes > max_age_minutes:
            return None

        return _context_from_dict(payload["context"])
    except Exception:
        return None


def _load_cache_any() -> NewsContext | None:
    if not os.path.exists(NEWS_CACHE_PATH):
        return None

    try:
        with open(NEWS_CACHE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        context = _context_from_dict(payload["context"])
        context.stale = True
        return context
    except Exception:
        return None


def _context_from_dict(data: dict) -> NewsContext:
    headlines = [
        NewsHeadline(
            title=str(item.get("title", "")),
            link=str(item.get("link", "")),
            published_at=str(item.get("published_at", "")),
            sentiment=float(item.get("sentiment", 0.0)),
            impact=float(item.get("impact", 0.0)),
            topic=str(item.get("topic", "general")),
        )
        for item in data.get("headlines", [])
    ]
    return NewsContext(
        bias=str(data.get("bias", "neutral")),
        sentiment_score=float(data.get("sentiment_score", 0.0)),
        impact_score=float(data.get("impact_score", 0.0)),
        headline_count=int(data.get("headline_count", len(headlines))),
        dominant_topic=str(data.get("dominant_topic", "general")),
        stale=bool(data.get("stale", False)),
        headlines=headlines,
        source=str(data.get("source", "cache")),
    )


def build_news_context(
    symbol: str = "BTCUSDT",
    limit: int = 6,
    cache_minutes: int = 15,
    enabled: bool = True,
) -> NewsContext:
    if not enabled:
        return NewsContext(
            bias="disabled",
            sentiment_score=0.0,
            impact_score=0.0,
            headline_count=0,
            dominant_topic="none",
            stale=False,
            headlines=[],
            source="disabled",
        )

    cached = _load_cache(cache_minutes)
    if cached is not None:
        return cached

    headlines: List[NewsHeadline] = []
    source = "rss"

    for url in _query_urls(symbol):
        try:
            response = requests.get(
                url,
                timeout=8,
                headers={"User-Agent": "Mozilla/5.0 ProjectXXX Trading Assistant"},
            )
            if response.status_code != 200:
                continue
            headlines.extend(_parse_rss(response.text, limit=limit))
            if len(headlines) >= limit:
                break
        except Exception:
            continue

    unique_by_title: dict[str, NewsHeadline] = {}
    for item in headlines:
        unique_by_title[item.title] = item

    deduped = list(unique_by_title.values())[:limit]

    if not deduped:
        fallback = _load_cache_any()
        if fallback is not None:
            return fallback
        return NewsContext(
            bias="neutral",
            sentiment_score=0.0,
            impact_score=0.0,
            headline_count=0,
            dominant_topic="general",
            stale=True,
            headlines=[],
            source="unavailable",
        )

    weighted_sentiment = sum(h.sentiment * max(h.impact, 0.1) for h in deduped)
    weight_total = sum(max(h.impact, 0.1) for h in deduped)
    sentiment_score = weighted_sentiment / weight_total if weight_total > 0 else 0.0
    impact_score = sum(h.impact for h in deduped) / len(deduped)

    topic_counts: dict[str, int] = {}
    for item in deduped:
        topic_counts[item.topic] = topic_counts.get(item.topic, 0) + 1
    dominant_topic = max(topic_counts, key=topic_counts.get) if topic_counts else "general"

    if sentiment_score >= 0.18:
        bias = "bullish"
    elif sentiment_score <= -0.18:
        bias = "bearish"
    elif impact_score >= 0.65:
        bias = "high_alert"
    else:
        bias = "neutral"

    context = NewsContext(
        bias=bias,
        sentiment_score=round(float(sentiment_score), 4),
        impact_score=round(float(impact_score), 4),
        headline_count=len(deduped),
        dominant_topic=dominant_topic,
        stale=False,
        headlines=deduped,
        source=source,
    )
    _save_cache(context)
    return context
