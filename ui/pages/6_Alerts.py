import pandas as pd
import streamlit as st

from runtime.alert_engine import load_alerts, load_latest_alert


st.set_page_config(
    page_title="Trading Assistant - Alerts",
    layout="wide",
)

st.title("Trading Assistant - Alerts")
st.caption("Operational alert stream for live signals, blocked candidates, and high-impact news regimes.")

alerts = load_alerts(limit=250)
latest_alert = load_latest_alert()

if st.button("Refresh"):
    st.rerun()

if not alerts:
    st.info("No alerts generated yet.")
    st.stop()

df = pd.DataFrame(alerts)
metadata_df = pd.json_normalize(df["metadata"]).add_prefix("meta_") if "metadata" in df.columns else pd.DataFrame()
if not metadata_df.empty:
    df = pd.concat([df.drop(columns=["metadata"]), metadata_df], axis=1)

severity_filter = st.selectbox(
    "Severity Filter",
    options=["ALL"] + sorted(df["severity"].dropna().astype(str).unique().tolist()),
)

type_filter = st.selectbox(
    "Type Filter",
    options=["ALL"] + sorted(df["type"].dropna().astype(str).unique().tolist()),
)

filtered = df.copy()
if severity_filter != "ALL":
    filtered = filtered[filtered["severity"].astype(str) == severity_filter]
if type_filter != "ALL":
    filtered = filtered[filtered["type"].astype(str) == type_filter]

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric("Total Alerts", len(df))
with m2:
    st.metric("Live Signals", int((df["type"] == "signal_live").sum()))
with m3:
    st.metric("Blocked Candidates", int((df["type"] == "blocked_candidate").sum()))
with m4:
    st.metric("News Alerts", int((df["type"] == "news_regime").sum()))
with m5:
    st.metric("Critical", int((df["severity"] == "critical").sum()))

st.divider()

st.subheader("Latest Alert")
if latest_alert:
    st.markdown(
        f"""
        **Timestamp:** {latest_alert.get("timestamp", "N/A")}  
        **Type:** {latest_alert.get("type", "N/A")}  
        **Severity:** {latest_alert.get("severity", "N/A")}  
        **Title:** {latest_alert.get("title", "N/A")}  
        **Body:** {latest_alert.get("body", "N/A")}
        """
    )
    if latest_alert.get("metadata"):
        st.json(latest_alert["metadata"])

st.divider()

st.subheader("Alert Stream")
view_columns = [
    "timestamp",
    "type",
    "severity",
    "title",
    "body",
    "signal_id",
    "meta_symbol",
    "meta_decision",
    "meta_setup",
    "meta_score",
    "meta_strategy_mode",
    "meta_news_bias",
    "meta_quantum_state",
]
available_columns = [col for col in view_columns if col in filtered.columns]
st.dataframe(filtered[available_columns], width="stretch")

st.caption("Alerts page")
