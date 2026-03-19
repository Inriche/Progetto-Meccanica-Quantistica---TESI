import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.ui_helpers import (
    squeeze_risk_label,
    funding_bias_label,
    oi_momentum_label,
    structure_read_label,
    orderbook_read_label,
    derivatives_read_label,
    grade_badge,
    rr_quality_label,
    setup_state_label,
    final_verdict_text,
)

DB_PATH = "out/assistant.db"
TICKETS_DIR = Path("out/tickets")
SNAPSHOTS_DIR = Path("out/snapshots")
COMMANDS_FILE = Path("out/commands.txt")


st.set_page_config(
    page_title="Trading Assistant Dashboard",
    layout="wide"
)


def get_conn():
    return sqlite3.connect(DB_PATH)


def load_recent_events(limit: int = 20) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()

    conn = get_conn()
    query = f"""
        SELECT
            timestamp,
            event_type,
            symbol,
            decision,
            setup,
            context,
            action,
            why,
            entry,
            sl,
            tp1,
            tp2,
            rr_estimated,
            score,
            ob_imbalance,
            ob_raw,
            ob_age_ms,
            funding_rate,
            oi_now,
            oi_change_pct,
            crowding,
            ticket_path
        FROM signals
        ORDER BY id DESC
        LIMIT {limit}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_latest_event():
    df = load_recent_events(limit=1)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def load_latest_ticket():
    if not TICKETS_DIR.exists():
        return None

    files = sorted(TICKETS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None

    latest = files[0]
    with open(latest, "r", encoding="utf-8") as f:
        return json.load(f)


def load_latest_snapshot():
    if not SNAPSHOTS_DIR.exists():
        return None

    files = sorted(SNAPSHOTS_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None

    return files[0]


def load_recent_candles(limit=200):
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()

    conn = get_conn()

    query = f"""
        SELECT open_time, open, high, low, close
        FROM candles
        WHERE timeframe='15m'
        ORDER BY open_time DESC
        LIMIT {limit}
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return df

    df = df.sort_values("open_time")
    df["time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


def send_command(cmd: str):
    COMMANDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(COMMANDS_FILE, "a", encoding="utf-8") as f:
        f.write(cmd.strip().lower() + "\n")


def color_badge(label: str, value: str):
    palette = {
        "READY_IF_TRIGGER": "#16a34a",
        "READY_IF_TRIGGER_CAUTION": "#f97316",
        "WAIT_PULLBACK": "#eab308",
        "WAIT_PULLBACK_CAUTION": "#f97316",
        "WAIT_BREAKOUT": "#38bdf8",
        "WAIT_BREAKOUT_CAUTION": "#f97316",
        "WATCH_CLOSELY": "#a78bfa",
        "WATCH_CLOSELY_CAUTION": "#f97316",
        "WAIT_RESET": "#f97316",
        "WAIT_STRUCTURE": "#0ea5e9",
        "AVOID_CHOP": "#ef4444",
        "STAND_BY": "#64748b",
        "NO_BIAS": "#475569",
        "trend_clean": "#16a34a",
        "trend_extended": "#f97316",
        "transition": "#38bdf8",
        "chop": "#ef4444",
        "FLAT": "#64748b",
        "BUY": "#16a34a",
        "SELL": "#ef4444",
        "NONE": "#64748b",
        "BLOCKED": "#f97316",
    }
    color = palette.get(str(value), "#334155")
    st.markdown(
        f"""
        <div style="
            background:{color};
            color:white;
            padding:10px 14px;
            border-radius:12px;
            font-weight:600;
            text-align:center;
            margin-top:4px;
        ">
            {label}: {value}
        </div>
        """,
        unsafe_allow_html=True,
    )


def pressure_label(imb):
    if imb is None or pd.isna(imb):
        return "N/A", "⚪"
    if imb > 0.35:
        return "BUY PRESSURE", "🟢"
    if imb < -0.35:
        return "SELL PRESSURE", "🔴"
    return "NEUTRAL", "⚪"


st.title("Trading Assistant Dashboard")

top1, top2, top3 = st.columns([1, 1, 1])

with top1:
    if st.button("Refresh"):
        st.rerun()

with top2:
    if st.button("Analyze now"):
        send_command("a")
        st.success("Analyze command queued.")

with top3:
    if st.button("Status snapshot"):
        send_command("s")
        st.success("Status command queued.")

latest_event = load_latest_event()
latest_ticket = load_latest_ticket()
latest_snapshot = load_latest_snapshot()
recent_df = load_recent_events(limit=15)

if latest_event is None:
    st.warning("No events found yet.")
    st.stop()

df_candles = load_recent_candles()
latest_price = None
if not df_candles.empty:
    latest_price = float(df_candles["close"].iloc[-1])

liq_cluster = None
if latest_ticket is not None:
    liq_cluster = (
        latest_ticket.get("liquidity", {})
        .get("nearest_liquidation_cluster")
    )

squeeze_label, squeeze_icon = squeeze_risk_label(latest_price, liq_cluster)

st.subheader("Live Status")

c1, c2, c3, c4 = st.columns(4)

with c1:
    color_badge("Decision", str(latest_event.get("decision", "N/A")))

with c2:
    color_badge("Setup", str(latest_event.get("setup", "N/A")))

with c3:
    color_badge("Context", str(latest_event.get("context", "N/A")))

with c4:
    color_badge("Action", str(latest_event.get("action", "N/A")))

st.write(f"**Timestamp:** {latest_event.get('timestamp', 'N/A')}")
st.write(f"**Why:** {latest_event.get('why', 'N/A')}")

st.divider()

st.subheader("Summary Panel")

context_val = str(latest_event.get("context", "N/A"))
action_val = str(latest_event.get("action", "N/A"))
imb_summary = latest_event.get("ob_imbalance")
crowding_val = latest_event.get("crowding", "N/A")
funding_val = latest_event.get("funding_rate")
oi_change_val = latest_event.get("oi_change_pct")

structure_label, structure_icon = structure_read_label(context_val)
orderbook_label, orderbook_icon = orderbook_read_label(imb_summary)
deriv_label, deriv_icon = derivatives_read_label(crowding_val, funding_val, oi_change_val)

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.markdown(f"### {structure_icon} Structure")
    st.write(structure_label)

with s2:
    st.markdown(f"### {orderbook_icon} Order Book")
    st.write(orderbook_label)

with s3:
    st.markdown(f"### {deriv_icon} Derivatives")
    st.write(deriv_label)

with s4:
    st.markdown("### 🎯 Final Read")
    st.write(action_val)

st.divider()

st.subheader("Confidence Panel")

score_val = latest_event.get("score")
rr_val = latest_event.get("rr_estimated")
decision_val = str(latest_event.get("decision", "N/A"))
setup_val = str(latest_event.get("setup", "N/A"))

grade_label, grade_icon = grade_badge(score_val)
rr_label, rr_icon = rr_quality_label(rr_val)
setup_state, setup_state_icon = setup_state_label(decision_val, setup_val)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("### 📊 Score")
    st.write("N/A" if score_val is None or pd.isna(score_val) else int(score_val))

with c2:
    st.markdown(f"### {grade_icon} Grade")
    st.write(grade_label)

with c3:
    st.markdown(f"### {rr_icon} RR Quality")
    st.write(rr_label)

with c4:
    st.markdown(f"### {setup_state_icon} Setup State")
    st.write(setup_state)

st.divider()

st.subheader("Final Verdict")

verdict_title, verdict_body = final_verdict_text(
    context=str(latest_event.get("context", "N/A")),
    action=str(latest_event.get("action", "N/A")),
    decision=str(latest_event.get("decision", "N/A")),
    setup=str(latest_event.get("setup", "N/A")),
    score=latest_event.get("score"),
    rr_estimated=latest_event.get("rr_estimated"),
    crowding=str(latest_event.get("crowding", "N/A")),
)

st.markdown(
    f"""
    <div style="
        background:#0f172a;
        color:white;
        padding:16px;
        border-radius:14px;
        border:1px solid #334155;
        margin-bottom:10px;
    ">
        <div style="font-size:20px; font-weight:700; margin-bottom:8px;">
            {verdict_title}
        </div>
        <div style="font-size:15px; color:#cbd5e1;">
            {verdict_body}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

st.subheader("Market Microstructure")

imb = latest_event.get("ob_imbalance")
raw = latest_event.get("ob_raw")
age = latest_event.get("ob_age_ms")

m1, m2, m3 = st.columns(3)

with m1:
    val = "N/A" if pd.isna(imb) or imb is None else round(float(imb), 3)
    st.metric("OrderBook Imbalance Avg", val)

with m2:
    val = "N/A" if pd.isna(raw) or raw is None else round(float(raw), 3)
    st.metric("OrderBook Raw", val)

with m3:
    val = "N/A" if pd.isna(age) or age is None else int(age)
    st.metric("Data Age (ms)", val)

pressure, icon = pressure_label(imb)
st.markdown(f"### Pressure: {icon} {pressure}")

st.divider()

st.subheader("Derivatives Context")


d1, d2, d3, d4 = st.columns(4)

with d1:
    fr = latest_event.get("funding_rate")
    st.metric("Funding Rate", "N/A" if fr is None or pd.isna(fr) else round(float(fr), 6))

with d2:
    oi_now = latest_event.get("oi_now")
    st.metric("Open Interest Now", "N/A" if oi_now is None or pd.isna(oi_now) else round(float(oi_now), 2))

with d3:
    oi_ch = latest_event.get("oi_change_pct")
    st.metric("OI Change 15m", "N/A" if oi_ch is None or pd.isna(oi_ch) else f"{float(oi_ch)*100:.2f}%")

with d4:
    crowding = latest_event.get("crowding", "N/A")
    st.metric("Crowding", crowding)

funding_label, funding_icon = funding_bias_label(fr)
oi_label, oi_icon = oi_momentum_label(oi_ch)

st.markdown(f"**Funding Bias:** {funding_icon} {funding_label}")
st.markdown(f"**OI Momentum:** {oi_icon} {oi_label}")

st.divider()

st.subheader("Liquidity Risk")

l1, l2 = st.columns(2)

with l1:
    st.metric(
        "Nearest Liquidation Cluster",
        "N/A" if liq_cluster is None else round(float(liq_cluster), 2)
    )

with l2:
    st.markdown(f"### {squeeze_icon} {squeeze_label}")

st.divider()

left, right = st.columns([1.25, 1])

with left:
    st.subheader("BTCUSDT Chart (M15)")

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

        entry = latest_event.get("entry")
        sl = latest_event.get("sl")
        tp1 = latest_event.get("tp1")
        tp2 = latest_event.get("tp2")

        if entry is not None and not pd.isna(entry):
            fig.add_hline(
                y=float(entry),
                line_dash="solid",
                line_color="blue",
                annotation_text="ENTRY"
            )

        if sl is not None and not pd.isna(sl):
            fig.add_hline(
                y=float(sl),
                line_dash="dot",
                line_color="red",
                annotation_text="SL"
            )

        if tp1 is not None and not pd.isna(tp1):
            fig.add_hline(
                y=float(tp1),
                line_dash="dash",
                line_color="green",
                annotation_text="TP1"
            )

        if tp2 is not None and not pd.isna(tp2):
            fig.add_hline(
                y=float(tp2),
                line_dash="dash",
                line_color="green",
                annotation_text="TP2"
            )

        if liq_cluster is not None:
            fig.add_hline(
                y=float(liq_cluster),
                line_dash="dot",
                line_color="orange",
                annotation_text="LIQ CLUSTER"
            )

        fig.update_layout(
            height=520,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )

        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No candle data yet.")

with right:
    st.subheader("Latest Event Details")
    st.write(f"**Event Type:** {latest_event.get('event_type', 'N/A')}")
    st.write(f"**Symbol:** {latest_event.get('symbol', 'N/A')}")
    st.write(f"**Score:** {latest_event.get('score', 'N/A')}")
    st.write(f"**RR Estimated:** {latest_event.get('rr_estimated', 'N/A')}")
    st.write(f"**Entry:** {latest_event.get('entry', 'N/A')}")
    st.write(f"**SL:** {latest_event.get('sl', 'N/A')}")
    st.write(f"**TP1:** {latest_event.get('tp1', 'N/A')}")
    st.write(f"**TP2:** {latest_event.get('tp2', 'N/A')}")

    st.subheader("Latest Snapshot")
    if latest_snapshot is not None:
        st.image(str(latest_snapshot), width="stretch")
    else:
        st.info("No snapshot available yet.")

st.divider()

col_json, col_info = st.columns([1, 1])

with col_json:
    st.subheader("Latest Ticket JSON")
    if latest_ticket is not None:
        st.json(latest_ticket)
    else:
        st.info("No ticket JSON found yet.")

with col_info:
    st.subheader("Quick Read")

    context = str(latest_event.get("context", "N/A"))
    action = str(latest_event.get("action", "N/A"))
    why = str(latest_event.get("why", "N/A"))
    crowding = latest_event.get("crowding", "N/A")

    st.markdown(
        f"""
        **Context:** {context}  
        **Action:** {action}  
        **Reason:** {why}  
        **Nearest Liquidation Cluster:** {liq_cluster if liq_cluster is not None else "N/A"}  
        **Squeeze Risk:** {squeeze_icon} {squeeze_label}  
        **Crowding:** {crowding}  
        **Funding Bias:** {funding_icon} {funding_label}  
        **OI Momentum:** {oi_icon} {oi_label}
        """
    )

st.divider()

st.subheader("Recent Journal Events")
if recent_df.empty:
    st.info("No recent events.")
else:
    st.dataframe(recent_df, width="stretch")

st.divider()
st.caption("Trading Assistant MVP dashboard")