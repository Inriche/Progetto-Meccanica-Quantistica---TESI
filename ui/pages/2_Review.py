import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from execution.execution_store import get_execution_note, set_execution_note
from execution.outcome_simulator import (
    load_future_candles_after_timestamp,
    simulate_outcome_from_candles,
)

from ui.review_helpers import (
    load_events_basic,
    load_event_by_id,
    load_ticket_from_path,
    load_snapshot_from_ticket,
    load_candles_around_event,
    split_why_reasons,
    extract_ticket_key_reasons,
    render_reason_badge,
    get_signal_id_from_ticket,
)

st.set_page_config(
    page_title="Trading Assistant - Review",
    layout="wide"
)

st.title("Trading Assistant - Review")

events = load_events_basic()

if events.empty:
    st.warning("No events found yet.")
    st.stop()

events["label"] = (
    events["id"].astype(str)
    + " | "
    + events["timestamp"].astype(str)
    + " | "
    + events["event_type"].astype(str)
    + " | "
    + events["setup"].astype(str)
    + " | "
    + events["action"].astype(str)
)

selected_label = st.selectbox(
    "Select event",
    options=events["label"].tolist()
)

selected_id = int(selected_label.split(" | ")[0])
event = load_event_by_id(selected_id)

if event is None:
    st.error("Event not found.")
    st.stop()

ticket = load_ticket_from_path(event.get("ticket_path"))
snapshot_path = load_snapshot_from_ticket(ticket)

signal_id = get_signal_id_from_ticket(ticket)
execution_note = get_execution_note(signal_id) if signal_id else {}

future_df = load_future_candles_after_timestamp(
    event_timestamp_iso=str(event["timestamp"]),
    timeframe="15m",
    limit=100,
)

sim_result = simulate_outcome_from_candles(
    decision=str(event.get("decision")),
    entry=event.get("entry"),
    sl=event.get("sl"),
    tp1=event.get("tp1"),
    tp2=event.get("tp2"),
    future_df=future_df,
)

candles_before = st.slider("Candles before event", 20, 200, 80, 10)
candles_after = st.slider("Candles after event", 0, 100, 30, 5)

df_candles = load_candles_around_event(
    event_timestamp_iso=str(event["timestamp"]),
    timeframe="15m",
    candles_before=candles_before,
    candles_after=candles_after,
)

st.subheader("Event Summary")

st.divider()

st.subheader("Execution Review")

default_status = execution_note.get("status", "unreviewed")
default_result = execution_note.get("result", "none")
default_note = execution_note.get("note", "")

with st.form("execution_review_form"):
    review_status = st.selectbox(
        "Operator Decision",
        options=["unreviewed", "taken", "skipped"],
        index=["unreviewed", "taken", "skipped"].index(default_status if default_status in ["unreviewed", "taken", "skipped"] else "unreviewed"),
    )

    review_result = st.selectbox(
        "Outcome",
        options=["none", "won", "lost", "breakeven"],
        index=["none", "won", "lost", "breakeven"].index(default_result if default_result in ["none", "won", "lost", "breakeven"] else "none"),
    )

    review_note = st.text_area("Notes", value=default_note, height=120)

    submitted = st.form_submit_button("Save Execution Review")

    if submitted:
        if signal_id:
            set_execution_note(
                signal_id,
                {
                    "status": review_status,
                    "result": review_result,
                    "note": review_note.strip(),
                    "event_id": int(event["id"]),
                    "timestamp": str(event["timestamp"]),
                },
            )
            st.success("Execution review saved.")
            st.rerun()
        else:
            st.warning("This event has no ticket/signal id to attach a review.")
note_after = get_execution_note(signal_id) if signal_id else {}

r1, r2, r3, r4 = st.columns(4)
with r1:
    st.metric("Review Status", note_after.get("status", "unreviewed"))
with r2:
    st.metric("Manual Outcome", note_after.get("result", "none"))
with r3:
    st.metric("Auto Outcome", sim_result.get("status", "no_data"))
with r4:
    st.metric("Has Notes", "yes" if note_after.get("note") else "no")

st.markdown(
    f"""
    **Auto outcome details:**  
    hit_price = {sim_result.get('hit_price')}  
    hit_time = {sim_result.get('hit_time')}  
    bars_checked = {sim_result.get('bars_checked')}
    """
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Decision", event.get("decision", "N/A"))

with c2:
    st.metric("Setup", event.get("setup", "N/A"))

with c3:
    st.metric("Context", event.get("context", "N/A"))

with c4:
    st.metric("Action", event.get("action", "N/A"))

st.write(f"**Timestamp:** {event.get('timestamp', 'N/A')}")
st.write(f"**Event Type:** {event.get('event_type', 'N/A')}")
st.write(f"**Why:** {event.get('why', 'N/A')}")

st.divider()

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Score", "N/A" if pd.isna(event.get("score")) else int(event.get("score")))

with m2:
    rr_val = event.get("rr_estimated")
    st.metric("RR", "N/A" if pd.isna(rr_val) else round(float(rr_val), 2))

with m3:
    fr = event.get("funding_rate")
    st.metric("Funding", "N/A" if pd.isna(fr) else round(float(fr), 6))

with m4:
    oi_ch = event.get("oi_change_pct")
    st.metric("OI Change 15m", "N/A" if pd.isna(oi_ch) else f"{float(oi_ch)*100:.2f}%")

st.divider()

left, right = st.columns([1.35, 1])

with left:
    st.subheader("Review Chart (Historical M15 Around Event)")

    if not df_candles.empty:
        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=df_candles["time"],
                open=df_candles["open"],
                high=df_candles["high"],
                low=df_candles["low"],
                close=df_candles["close"],
                name="BTCUSDT"
            )
        )

        entry = event.get("entry")
        sl = event.get("sl")
        tp1 = event.get("tp1")
        tp2 = event.get("tp2")

        liq_cluster = None
        if ticket is not None:
            liq_cluster = (
                ticket.get("liquidity", {})
                .get("nearest_liquidation_cluster")
            )

        event_time = pd.Timestamp(event["timestamp"]).to_pydatetime()

        fig.add_shape(
            type="line",
            x0=event_time,
            x1=event_time,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(dash="dash")
        )

        fig.add_annotation(
            x=event_time,
            y=1,
            xref="x",
            yref="paper",
            text="EVENT",
            showarrow=False,
            yanchor="bottom"
        )

        if entry is not None and not pd.isna(entry):
            fig.add_hline(y=float(entry), line_dash="solid", annotation_text="ENTRY")

        if sl is not None and not pd.isna(sl):
            fig.add_hline(y=float(sl), line_dash="dot", annotation_text="SL")

        if tp1 is not None and not pd.isna(tp1):
            fig.add_hline(y=float(tp1), line_dash="dash", annotation_text="TP1")

        if tp2 is not None and not pd.isna(tp2):
            fig.add_hline(y=float(tp2), line_dash="dash", annotation_text="TP2")

        if liq_cluster is not None:
            fig.add_hline(y=float(liq_cluster), line_dash="dot", annotation_text="LIQ")

        fig.update_layout(
            height=560,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )

        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No historical candle window available for this event.")

with right:
    st.subheader("Snapshot")
    if snapshot_path:
        st.image(snapshot_path, width="stretch")
    else:
        st.info("No snapshot available for this event.")

st.divider()

r1, r2 = st.columns([1, 1])

with r1:
    st.subheader("Event Raw Data")
    st.json(event)

with r2:
    st.subheader("Ticket JSON")
    if ticket is not None:
        st.json(ticket)
    else:
        st.info("No ticket for this event.")

st.divider()

st.subheader("Event Snapshot From Ticket")

if ticket is not None:
    event_snapshot = ticket.get("event_snapshot", {})
    if event_snapshot:
        st.json(event_snapshot)
    else:
        st.info("No event snapshot inside ticket.")
else:
    st.info("No ticket available.")

st.divider()

st.subheader("Readable Breakdown")

setup = str(event.get("setup", "N/A"))
action = str(event.get("action", "N/A"))
context = str(event.get("context", "N/A"))
crowding = str(event.get("crowding", "N/A"))
score = event.get("score")
rr_val = event.get("rr_estimated")

summary_lines = [
    f"- Context: {context}",
    f"- Setup: {setup}",
    f"- Action: {action}",
    f"- Score: {score}",
    f"- RR: {rr_val}",
    f"- Crowding: {crowding}",
]

for line in summary_lines:
    st.write(line)

st.divider()

st.subheader("Parsed Reasons")

parsed_why = split_why_reasons(event.get("why"))
ticket_reasons = extract_ticket_key_reasons(ticket)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**From event `why`**")
    if parsed_why:
        for reason in parsed_why:
            st.markdown(render_reason_badge(reason), unsafe_allow_html=True)
    else:
        st.info("No parsed reasons found in event.")

with col_b:
    st.markdown("**From ticket `key_reasons`**")
    if ticket_reasons:
        for reason in ticket_reasons:
            st.markdown(render_reason_badge(reason), unsafe_allow_html=True)
    else:
        st.info("No key reasons found in ticket.")

st.caption("Review page for single-event analysis")