import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from runtime.alert_engine import load_latest_alert
from runtime.runtime_config import load_runtime_config, save_runtime_config
from ui.ui_helpers import (
    derivatives_read_label,
    final_verdict_text,
    funding_bias_label,
    grade_badge,
    oi_momentum_label,
    orderbook_read_label,
    quantum_read_label,
    rr_quality_label,
    setup_state_label,
    squeeze_risk_label,
    structure_read_label,
)
from validation.market_read import load_market_read_df, summarize_market_read

DB_PATH = "out/assistant.db"
TICKETS_DIR = Path("out/tickets")
SNAPSHOTS_DIR = Path("out/snapshots")
COMMANDS_FILE = Path("out/commands.txt")
SCORING_MODES = ("heuristic", "hybrid", "ml")


st.set_page_config(
    page_title="Trading Assistant Dashboard",
    layout="wide",
)


def get_conn():
    return sqlite3.connect(DB_PATH)


def get_table_columns(table_name: str) -> list[str]:
    if not os.path.exists(DB_PATH):
        return []

    conn = get_conn()
    try:
        cur = conn.execute(f"PRAGMA table_info({table_name})")
        return [row[1] for row in cur.fetchall()]
    finally:
        conn.close()


def format_value(value, *, precision: int | None = None):
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, str):
        return value
    if precision is not None:
        try:
            return round(float(value), precision)
        except Exception:
            return str(value)
    if isinstance(value, (int, float)):
        return value
    return str(value)


def load_recent_events(limit: int = 20) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()

    conn = get_conn()
    desired_cols = [
        "timestamp",
        "event_type",
        "symbol",
        "decision",
        "setup",
        "context",
        "action",
        "why",
        "entry",
        "sl",
        "tp1",
        "tp2",
        "rr_estimated",
        "heuristic_score",
        "score",
        "ob_imbalance",
        "ob_raw",
        "ob_age_ms",
        "funding_rate",
        "oi_now",
        "oi_change_pct",
        "crowding",
        "strategy_mode",
        "strategy_score",
        "scoring_mode",
        "news_bias",
        "news_sentiment",
        "news_impact",
        "news_score",
        "quantum_state",
        "quantum_coherence",
        "quantum_phase_bias",
        "quantum_interference",
        "quantum_tunneling",
        "quantum_energy",
        "quantum_decoherence_rate",
        "quantum_transition_rate",
        "quantum_dominant_mode",
        "quantum_score",
        "ticket_path",
    ]
    available_cols = set(get_table_columns("signals"))
    select_cols = [c for c in desired_cols if c in available_cols]
    if not select_cols:
        conn.close()
        return pd.DataFrame()

    query = f"""
        SELECT {", ".join(select_cols)}
        FROM signals
        ORDER BY id DESC
        LIMIT {limit}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    for col in desired_cols:
        if col not in df.columns:
            df[col] = None
    return df[desired_cols]


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

    df = df.sort_values("open_time").copy()
    df.loc[:, "time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


def send_command(cmd: str):
    COMMANDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(COMMANDS_FILE, "a", encoding="utf-8") as f:
        f.write(cmd.strip().lower() + "\n")


def normalize_scoring_mode(value) -> str:
    mode = str(value or "").strip().lower()
    return mode if mode in SCORING_MODES else "hybrid"


def save_scoring_mode(mode: str) -> bool:
    cfg = load_runtime_config()
    target_mode = normalize_scoring_mode(mode)
    current_mode = normalize_scoring_mode(cfg.get("scoring_mode"))
    if current_mode == target_mode:
        return False
    cfg["scoring_mode"] = target_mode
    save_runtime_config(cfg)
    return True


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
        "COHERENT_BULLISH": "#16a34a",
        "COHERENT_BEARISH": "#ef4444",
        "BULLISH_TUNNEL": "#22c55e",
        "BEARISH_TUNNEL": "#f43f5e",
        "DECOHERENT": "#f97316",
        "TRANSITIONAL": "#64748b",
        "LOW_ENERGY": "#0ea5e9",
        "WARMING_UP": "#64748b",
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

runtime_cfg = load_runtime_config()
active_scoring_mode = normalize_scoring_mode(runtime_cfg.get("scoring_mode"))

sm1, sm2 = st.columns([1, 2])
with sm1:
    selected_scoring_mode = st.selectbox(
        "Scoring Approach",
        options=list(SCORING_MODES),
        index=list(SCORING_MODES).index(active_scoring_mode),
        key="scoring_mode_selector",
    )
with sm2:
    st.markdown(f"**Active Scoring Mode:** `{active_scoring_mode}`")
    st.caption("La modifica viene salvata in `out/config_runtime.json` e applicata nei cicli successivi.")

if selected_scoring_mode != active_scoring_mode:
    if save_scoring_mode(selected_scoring_mode):
        st.success(f"Scoring mode aggiornato a `{selected_scoring_mode}`.")
        st.rerun()

latest_event = load_latest_event()
latest_ticket = load_latest_ticket()
latest_snapshot = load_latest_snapshot()
recent_df = load_recent_events(limit=15)
latest_alert = load_latest_alert()
market_read_df = load_market_read_df(
    limit=40,
    horizon_bars=int(runtime_cfg.get("validation_horizon_bars", 16)),
    min_follow_through_pct=float(runtime_cfg.get("validation_min_follow_through_pct", 0.0035)),
    max_adverse_pct=float(runtime_cfg.get("validation_max_adverse_pct", 0.0025)),
)
market_read_summary = summarize_market_read(market_read_df)

if latest_event is None:
    st.warning("No events found yet.")
    st.stop()

df_candles = load_recent_candles()
latest_price = None
if not df_candles.empty:
    latest_price = float(df_candles["close"].iloc[-1])

liq_cluster = None
if latest_ticket is not None:
    liq_cluster = latest_ticket.get("liquidity", {}).get("nearest_liquidation_cluster")

squeeze_label, squeeze_icon = squeeze_risk_label(
    latest_price,
    liq_cluster,
    high_pct=float(runtime_cfg["squeeze_risk_high_pct"]),
    medium_pct=float(runtime_cfg["squeeze_risk_medium_pct"]),
)

st.subheader("Live Status")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    color_badge("Decision", str(latest_event.get("decision", "N/A")))

with c2:
    color_badge("Setup", str(latest_event.get("setup", "N/A")))

with c3:
    color_badge("Context", str(latest_event.get("context", "N/A")))

with c4:
    color_badge("Action", str(latest_event.get("action", "N/A")))

with c5:
    color_badge("Quantum", str(latest_event.get("quantum_state", "N/A")))

st.write(f"**Timestamp:** {latest_event.get('timestamp', 'N/A')}")
st.write(f"**Why:** {latest_event.get('why', 'N/A')}")

meta1, meta2, meta3 = st.columns(3)
with meta1:
    st.metric("Strategy Mode", str(latest_event.get("strategy_mode", "N/A")))
with meta2:
    st.metric("News Bias", str(latest_event.get("news_bias", "N/A")))
with meta3:
    news_score_val = latest_event.get("news_score")
    st.metric("News Score", "N/A" if news_score_val is None or pd.isna(news_score_val) else int(news_score_val))

st.divider()

st.subheader("Alerts And Validation")

av1, av2, av3, av4 = st.columns(4)

with av1:
    st.metric(
        "Validated Rate",
        "N/A" if market_read_summary["validation_rate"] is None else f"{market_read_summary['validation_rate']:.2f}%",
    )

with av2:
    st.metric(
        "Avg Read Score",
        "N/A" if market_read_summary["avg_read_score"] is None else market_read_summary["avg_read_score"],
    )

with av3:
    st.metric(
        "Validated / Completed",
        f"{market_read_summary['validated']}/{market_read_summary['completed']}",
    )

with av4:
    if latest_alert is None:
        st.metric("Latest Alert", "none")
    else:
        st.metric("Latest Alert", str(latest_alert.get("type", "N/A")))

if latest_alert is not None:
    st.markdown(
        f"""
        **Alert Severity:** {latest_alert.get("severity", "N/A")}  
        **Alert Title:** {latest_alert.get("title", "N/A")}  
        **Alert Body:** {latest_alert.get("body", "N/A")}
        """
    )

completed_reads = (
    market_read_df[market_read_df["validation_status"].isin(["validated", "invalidated", "mixed"])].copy()
    if not market_read_df.empty
    else pd.DataFrame()
)
if not completed_reads.empty:
    latest_read = completed_reads.iloc[0]
    st.markdown(
        f"""
        **Latest Completed Read:** {latest_read.get("validation_status", "N/A")}  
        **Read Score:** {latest_read.get("read_score", "N/A")}  
        **Validation Note:** {latest_read.get("validation_note", "N/A")}
        """
    )

st.divider()

st.subheader("Summary Panel")

context_val = str(latest_event.get("context", "N/A"))
action_val = str(latest_event.get("action", "N/A"))
imb_summary = latest_event.get("ob_imbalance")
crowding_val = latest_event.get("crowding", "N/A")
funding_val = latest_event.get("funding_rate")
oi_change_val = latest_event.get("oi_change_pct")
quantum_state_val = str(latest_event.get("quantum_state", "N/A"))
quantum_coherence_val = latest_event.get("quantum_coherence")
quantum_phase_val = latest_event.get("quantum_phase_bias")
strategy_mode_val = str(latest_event.get("strategy_mode", "N/A"))
news_bias_val = str(latest_event.get("news_bias", "N/A"))
news_sentiment_val = latest_event.get("news_sentiment")
news_impact_val = latest_event.get("news_impact")

structure_label, structure_icon = structure_read_label(context_val)
quantum_label, quantum_icon = quantum_read_label(
    quantum_state_val,
    quantum_coherence_val,
    quantum_phase_val,
)
orderbook_label, orderbook_icon = orderbook_read_label(imb_summary)
deriv_label, deriv_icon = derivatives_read_label(crowding_val, funding_val, oi_change_val)

s1, s2, s3, s4, s5 = st.columns(5)

with s1:
    st.markdown(f"### {structure_icon} Structure")
    st.write(structure_label)

with s2:
    st.markdown(f"### {quantum_icon} Quantum")
    st.write(quantum_label)

with s3:
    st.markdown(f"### {orderbook_icon} Order Book")
    st.write(orderbook_label)

with s4:
    st.markdown(f"### {deriv_icon} Derivatives")
    st.write(deriv_label)

with s5:
    st.markdown("### Final Read")
    st.write(action_val)

st.divider()

st.subheader("Strategy And News")

sn1, sn2, sn3, sn4, sn5 = st.columns(5)

with sn1:
    st.metric("Strategy", strategy_mode_val)

with sn2:
    strategy_score_val = latest_event.get("strategy_score")
    st.metric("Strategy Score", "N/A" if strategy_score_val is None or pd.isna(strategy_score_val) else int(strategy_score_val))

with sn3:
    st.metric("News Bias", news_bias_val)

with sn4:
    st.metric("News Sentiment", "N/A" if news_sentiment_val is None or pd.isna(news_sentiment_val) else round(float(news_sentiment_val), 3))

with sn5:
    st.metric("News Impact", "N/A" if news_impact_val is None or pd.isna(news_impact_val) else round(float(news_impact_val), 3))

if latest_ticket is not None:
    ticket_news = latest_ticket.get("news", {}) or latest_ticket.get("event_snapshot", {}).get("news", {})
    headlines = ticket_news.get("headlines", [])
    if headlines:
        st.markdown("**Top News Headlines**")
        for item in headlines[:5]:
            title = str(item.get("title", ""))
            topic = str(item.get("topic", "general"))
            impact = item.get("impact")
            sentiment = item.get("sentiment")
            st.write(f"- [{title}]({item.get('link', '#')}) | topic={topic} | impact={impact} | sentiment={sentiment}")

st.divider()

st.subheader("Confidence Panel")

score_val = latest_event.get("score")
rr_val = latest_event.get("rr_estimated")
decision_val = str(latest_event.get("decision", "N/A"))
setup_val = str(latest_event.get("setup", "N/A"))
quantum_score_val = latest_event.get("quantum_score")

grade_label, grade_icon = grade_badge(score_val)
rr_label, rr_icon = rr_quality_label(rr_val)
setup_state, setup_state_icon = setup_state_label(decision_val, setup_val)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown("### Score")
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

with c5:
    st.markdown("### Quantum Score")
    st.write("N/A" if quantum_score_val is None or pd.isna(quantum_score_val) else int(quantum_score_val))

st.divider()

st.subheader("Quantum Layer")

q1, q2, q3, q4, q5 = st.columns(5)

with q1:
    st.metric("Quantum State", quantum_state_val)

with q2:
    st.metric(
        "Coherence",
        "N/A" if quantum_coherence_val is None or pd.isna(quantum_coherence_val) else round(float(quantum_coherence_val), 3),
    )

with q3:
    st.metric(
        "Phase Bias",
        "N/A" if quantum_phase_val is None or pd.isna(quantum_phase_val) else round(float(quantum_phase_val), 3),
    )

with q4:
    qv = latest_event.get("quantum_interference")
    st.metric(
        "Interference",
        "N/A" if qv is None or pd.isna(qv) else round(float(qv), 3),
    )

with q5:
    qv = latest_event.get("quantum_tunneling")
    st.metric(
        "Tunneling Prob.",
        "N/A" if qv is None or pd.isna(qv) else round(float(qv), 3),
    )

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
    rr_min_required=float(runtime_cfg["rr_min"]),
    min_score_for_signal=int(runtime_cfg["min_score_for_signal"]),
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
    st.metric("OI Change 15m", "N/A" if oi_ch is None or pd.isna(oi_ch) else f"{float(oi_ch) * 100:.2f}%")

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
        "N/A" if liq_cluster is None else round(float(liq_cluster), 2),
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
                name="BTCUSDT",
            )
        )

        entry = latest_event.get("entry")
        sl = latest_event.get("sl")
        tp1 = latest_event.get("tp1")
        tp2 = latest_event.get("tp2")

        if entry is not None and not pd.isna(entry):
            fig.add_hline(y=float(entry), line_dash="solid", line_color="blue", annotation_text="ENTRY")

        if sl is not None and not pd.isna(sl):
            fig.add_hline(y=float(sl), line_dash="dot", line_color="red", annotation_text="SL")

        if tp1 is not None and not pd.isna(tp1):
            fig.add_hline(y=float(tp1), line_dash="dash", line_color="green", annotation_text="TP1")

        if tp2 is not None and not pd.isna(tp2):
            fig.add_hline(y=float(tp2), line_dash="dash", line_color="green", annotation_text="TP2")

        if liq_cluster is not None:
            fig.add_hline(
                y=float(liq_cluster),
                line_dash="dot",
                line_color="orange",
                annotation_text="LIQ CLUSTER",
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
    st.write(f"**Active Score:** {format_value(latest_event.get('score'))}")
    st.write(f"**Legacy / Heuristic Score:** {format_value(latest_event.get('heuristic_score'))}")
    st.write(f"**Strategy Score:** {format_value(latest_event.get('strategy_score'))}")
    st.write(f"**Raw Hybrid Score:** {format_value(latest_event.get('raw_hybrid_score'), precision=2)}")
    st.write(f"**Calibrated Hybrid Score:** {format_value(latest_event.get('calibrated_hybrid_score'), precision=2)}")
    st.write(f"**Scoring Mode:** {format_value(latest_event.get('scoring_mode'))}")
    st.write(f"**RR Estimated:** {latest_event.get('rr_estimated', 'N/A')}")
    st.write(f"**Quantum Score:** {latest_event.get('quantum_score', 'N/A')}")
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
        **Strategy:** {strategy_mode_val}  
        **Reason:** {why}  
        **News Bias:** {news_bias_val}  
        **News Sentiment:** {news_sentiment_val}  
        **News Impact:** {news_impact_val}  
        **Quantum State:** {quantum_state_val}  
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
    st.dataframe(recent_df.fillna("—"), width="stretch")

st.divider()
st.caption("Trading Assistant MVP dashboard")
