import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.stats_helpers import load_events, count_by, safe_mean, filter_trade_candidates


st.set_page_config(
    page_title="Trading Assistant - Stats",
    layout="wide"
)

st.title("Trading Assistant - Stats")

events = load_events()

if events.empty:
    st.warning("No events found yet.")
    st.stop()

trade_df = filter_trade_candidates(events)

top1, top2, top3, top4 = st.columns(4)

with top1:
    st.metric("Total Events", len(events))

with top2:
    st.metric("Signal Events", int((events["event_type"] == "signal").sum()))

with top3:
    st.metric("Status Events", int((events["event_type"] == "status").sum()))

with top4:
    blocked_count = int(((events["event_type"] == "signal") & (events["setup"] == "BLOCKED")).sum())
    st.metric("Blocked Candidates", blocked_count)

st.divider()

m1, m2, m3, m4 = st.columns(4)

avg_score = safe_mean(trade_df["score"]) if not trade_df.empty else None
avg_rr = safe_mean(trade_df["rr_estimated"]) if not trade_df.empty else None
avg_oi_change = safe_mean(events["oi_change_pct"])
avg_funding = safe_mean(events["funding_rate"])

with m1:
    st.metric("Avg Score", "N/A" if avg_score is None else round(avg_score, 2))

with m2:
    st.metric("Avg RR", "N/A" if avg_rr is None else round(avg_rr, 2))

with m3:
    st.metric("Avg OI Change", "N/A" if avg_oi_change is None else f"{avg_oi_change*100:.3f}%")

with m4:
    st.metric("Avg Funding", "N/A" if avg_funding is None else round(avg_funding, 6))

st.divider()

c1, c2 = st.columns(2)

with c1:
    st.subheader("Context Distribution")
    ctx_df = count_by(events, "context")
    if not ctx_df.empty:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=ctx_df["context"],
                    y=ctx_df["count"],
                    name="Context"
                )
            ]
        )
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No context data.")

with c2:
    st.subheader("Action Distribution")
    act_df = count_by(events, "action")
    if not act_df.empty:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=act_df["action"],
                    y=act_df["count"],
                    name="Action"
                )
            ]
        )
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No action data.")

st.divider()

c3, c4 = st.columns(2)

with c3:
    st.subheader("Setup Distribution")
    setup_df = count_by(events, "setup")
    if not setup_df.empty:
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=setup_df["setup"],
                    values=setup_df["count"],
                    hole=0.4
                )
            ]
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No setup data.")

with c4:
    st.subheader("Crowding Distribution")
    crowd_df = count_by(events, "crowding")
    if not crowd_df.empty:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=crowd_df["crowding"],
                    y=crowd_df["count"],
                    name="Crowding"
                )
            ]
        )
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No crowding data.")

st.divider()

st.subheader("Latest Trade Candidates")

if trade_df.empty:
    st.info("No signal events yet.")
else:
    view = trade_df[
        [
            "timestamp",
            "decision",
            "setup",
            "context",
            "action",
            "score",
            "rr_estimated",
            "funding_rate",
            "oi_change_pct",
            "crowding",
            "ticket_path",
        ]
    ].copy()

    st.dataframe(view, width="stretch")

st.divider()

st.subheader("Filtered Event Explorer")

col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    event_filter = st.selectbox(
        "Event Type",
        options=["ALL"] + sorted(events["event_type"].dropna().astype(str).unique().tolist())
    )

with col_f2:
    context_filter = st.selectbox(
        "Context",
        options=["ALL"] + sorted(events["context"].dropna().astype(str).unique().tolist())
    )

with col_f3:
    action_filter = st.selectbox(
        "Action",
        options=["ALL"] + sorted(events["action"].dropna().astype(str).unique().tolist())
    )

filtered = events.copy()

if event_filter != "ALL":
    filtered = filtered[filtered["event_type"].astype(str) == event_filter]

if context_filter != "ALL":
    filtered = filtered[filtered["context"].astype(str) == context_filter]

if action_filter != "ALL":
    filtered = filtered[filtered["action"].astype(str) == action_filter]

st.dataframe(filtered, width="stretch")

st.caption("Stats page for the Trading Assistant")