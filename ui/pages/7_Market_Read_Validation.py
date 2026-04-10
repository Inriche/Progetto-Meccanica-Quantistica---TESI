import pandas as pd
import streamlit as st

from runtime.runtime_config import load_runtime_config
from validation.market_read import (
    load_market_read_df,
    summarize_market_read,
    summarize_market_read_by,
)


st.set_page_config(
    page_title="Trading Assistant - Market Read Validation",
    layout="wide",
)

st.title("Trading Assistant - Market Read Validation")

cfg = load_runtime_config()
horizon_bars = int(cfg.get("validation_horizon_bars", 16))
min_follow_through_pct = float(cfg.get("validation_min_follow_through_pct", 0.0035))
max_adverse_pct = float(cfg.get("validation_max_adverse_pct", 0.0025))

st.caption(
    "Automatic validation checks whether the market moved in the expected direction after each signal. "
    "It measures follow-through, adverse excursion, and TP/SL interaction on future M15 candles."
)

if st.button("Refresh"):
    st.rerun()

df = load_market_read_df(
    limit=250,
    horizon_bars=horizon_bars,
    min_follow_through_pct=min_follow_through_pct,
    max_adverse_pct=max_adverse_pct,
)

if df.empty:
    st.info("No signal validations available yet.")
    st.stop()

summary = summarize_market_read(df)

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric("Signals Checked", summary["total"])
with m2:
    st.metric("Completed", summary["completed"])
with m3:
    st.metric("Validated Rate", "N/A" if summary["validation_rate"] is None else f"{summary['validation_rate']:.2f}%")
with m4:
    st.metric("Avg Read Score", "N/A" if summary["avg_read_score"] is None else summary["avg_read_score"])
with m5:
    st.metric("Pending", summary["pending"])

st.divider()

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.metric("Validated", summary["validated"])
with s2:
    st.metric("Invalidated", summary["invalidated"])
with s3:
    st.metric("Mixed", summary["mixed"])
with s4:
    st.metric(
        "Avg Favorable / Adverse",
        (
            "N/A"
            if summary["avg_favorable_excursion_pct"] is None or summary["avg_adverse_excursion_pct"] is None
            else f"{summary['avg_favorable_excursion_pct']:.3f}% / {summary['avg_adverse_excursion_pct']:.3f}%"
        ),
    )

st.divider()

status_counts = df["validation_status"].fillna("N/A").value_counts().reset_index()
status_counts.columns = ["validation_status", "count"]

left, right = st.columns(2)

with left:
    st.subheader("Validation Status")
    st.dataframe(status_counts, width="stretch")

with right:
    st.subheader("Runtime Thresholds")
    st.json(
        {
            "validation_horizon_bars": horizon_bars,
            "validation_min_follow_through_pct": min_follow_through_pct,
            "validation_max_adverse_pct": max_adverse_pct,
        }
    )

st.divider()

st.subheader("Validation By Strategy")
strategy_summary = summarize_market_read_by(df, "strategy_mode")
if strategy_summary.empty:
    st.info("No strategy validation summary available.")
else:
    st.dataframe(strategy_summary, width="stretch")

st.divider()

st.subheader("Validation By Setup")
setup_summary = summarize_market_read_by(df, "setup")
if setup_summary.empty:
    st.info("No setup validation summary available.")
else:
    st.dataframe(setup_summary, width="stretch")

st.divider()

st.subheader("Validation By Context")
context_summary = summarize_market_read_by(df, "context")
if context_summary.empty:
    st.info("No context validation summary available.")
else:
    st.dataframe(context_summary, width="stretch")

st.divider()

st.subheader("Recent Signal Reads")

view_cols = [
    "timestamp",
    "signal_id",
    "decision",
    "setup",
    "context",
    "action",
    "score",
    "strategy_mode",
    "validation_status",
    "outcome_status",
    "read_score",
    "favorable_excursion_pct",
    "adverse_excursion_pct",
    "close_move_pct",
    "validation_note",
]

table_df = df[view_cols].copy()
for pct_col in ["favorable_excursion_pct", "adverse_excursion_pct", "close_move_pct"]:
    if pct_col in table_df.columns:
        table_df[pct_col] = pd.to_numeric(table_df[pct_col], errors="coerce").apply(
            lambda x: None if pd.isna(x) else round(float(x) * 100.0, 3)
        )

st.dataframe(table_df, width="stretch")

st.caption("Automatic market-read validation page")
