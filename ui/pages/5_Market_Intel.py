import json
from pathlib import Path

import streamlit as st

from runtime.runtime_config import load_runtime_config
from signal_engine.strategy_profile import STRATEGY_PROFILES, get_strategy_profile


TICKETS_DIR = Path("out/tickets")


def load_latest_ticket():
    if not TICKETS_DIR.exists():
        return None

    files = sorted(TICKETS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None

    with open(files[0], "r", encoding="utf-8") as f:
        return json.load(f)


st.set_page_config(
    page_title="Trading Assistant - Market Intel",
    layout="wide",
)

st.title("Trading Assistant - Market Intel")

cfg = load_runtime_config()
strategy = get_strategy_profile(str(cfg.get("strategy_mode", "BALANCED")))
ticket = load_latest_ticket()

st.subheader("Active Strategy")

s1, s2, s3 = st.columns(3)

with s1:
    st.metric("Strategy Code", strategy.code)

with s2:
    st.metric("News Engine", "ON" if cfg.get("news_enabled", True) else "OFF")

with s3:
    st.metric("News Headlines Limit", int(cfg.get("news_headline_limit", 6)))

st.info(strategy.description)

st.markdown(
    f"""
    **Setup priority:** {", ".join(strategy.setup_priority)}  
    **Preferred setups:** {", ".join(strategy.preferred_setups)}  
    **News weight:** {strategy.news_weight}  
    **Quantum weight:** {strategy.quantum_weight}  
    **Liquidity weight:** {strategy.liquidity_weight}
    """
)

st.divider()

st.subheader("Latest News Snapshot")

if ticket is None:
    st.info("No ticket available yet.")
else:
    news = ticket.get("news", {}) or ticket.get("event_snapshot", {}).get("news", {})
    strategy_snapshot = ticket.get("strategy", {}) or ticket.get("event_snapshot", {}).get("strategy", {})

    if strategy_snapshot:
        st.markdown("**Latest Ticket Strategy Context**")
        st.json(strategy_snapshot)

    if not news:
        st.info("No news snapshot inside the latest ticket.")
    else:
        n1, n2, n3, n4 = st.columns(4)

        with n1:
            st.metric("Bias", news.get("bias", "N/A"))

        with n2:
            st.metric("Sentiment", news.get("sentiment_score", "N/A"))

        with n3:
            st.metric("Impact", news.get("impact_score", "N/A"))

        with n4:
            st.metric("Topic", news.get("dominant_topic", "N/A"))

        headlines = news.get("headlines", [])
        if headlines:
            st.markdown("**Headlines**")
            for item in headlines:
                title = str(item.get("title", ""))
                link = str(item.get("link", "#"))
                st.write(
                    f"- [{title}]({link}) | topic={item.get('topic')} | "
                    f"impact={item.get('impact')} | sentiment={item.get('sentiment')}"
                )
        else:
            st.info("No headlines in the latest news snapshot.")

st.divider()

st.subheader("Available Strategy Modes")

for code, profile in STRATEGY_PROFILES.items():
    st.markdown(
        f"""
        **{profile.label} ({code})**  
        {profile.description}  
        setup priority: {", ".join(profile.setup_priority)}
        """
    )

st.caption("Strategy and news intelligence page")
