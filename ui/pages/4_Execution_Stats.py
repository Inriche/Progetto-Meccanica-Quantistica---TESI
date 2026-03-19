import os
import sqlite3

import pandas as pd
import streamlit as st

from execution.execution_store import load_execution_store


DB_PATH = "out/assistant.db"


def get_conn():
    return sqlite3.connect(DB_PATH)


def load_signals_df() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()

    conn = get_conn()
    query = """
        SELECT
            signal_id,
            id as event_id,
            timestamp,
            event_type,
            decision,
            setup,
            context,
            action,
            score,
            rr_estimated,
            why
        FROM signals
        WHERE event_type = 'signal'
        ORDER BY id DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def safe_win_rate(df: pd.DataFrame) -> str:
    if df.empty:
        return "N/A"

    decided = df[df["result"].isin(["won", "lost"])].copy()
    if decided.empty:
        return "N/A"

    wins = int((decided["result"] == "won").sum())
    total = len(decided)
    return f"{(wins / total) * 100:.2f}%"


def summarize_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame(columns=[column, "count", "wins", "losses", "breakeven", "win_rate"])

    rows = []
    for value, grp in df.groupby(column, dropna=False):
        val = "N/A" if pd.isna(value) else str(value)

        wins = int((grp["result"] == "won").sum())
        losses = int((grp["result"] == "lost").sum())
        breakeven = int((grp["result"] == "breakeven").sum())
        count = len(grp)

        decided = wins + losses
        win_rate = f"{(wins / decided) * 100:.2f}%" if decided > 0 else "N/A"

        rows.append({
            column: val,
            "count": count,
            "wins": wins,
            "losses": losses,
            "breakeven": breakeven,
            "win_rate": win_rate,
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["count", "wins"], ascending=[False, False]).reset_index(drop=True)


st.set_page_config(
    page_title="Trading Assistant - Execution Stats",
    layout="wide"
)

st.title("Trading Assistant - Execution Stats")

store = load_execution_store()
signals_df = load_signals_df()

if not store:
    st.info("No execution reviews saved yet.")
    st.stop()

rows = []
for signal_id, note in store.items():
    if isinstance(note, dict):
        rows.append({
            "signal_id": signal_id,
            "status": note.get("status"),
            "result": note.get("result"),
            "note": note.get("note"),
            "event_id": note.get("event_id"),
            "timestamp": note.get("timestamp"),
        })

reviews_df = pd.DataFrame(rows)

merged = reviews_df.merge(
    signals_df,
    how="left",
    on="signal_id",
    suffixes=("_review", "_signal")
)

top1, top2, top3, top4 = st.columns(4)

with top1:
    st.metric("Reviewed Signals", len(reviews_df))

with top2:
    st.metric("Taken", int((reviews_df["status"] == "taken").sum()))

with top3:
    st.metric("Skipped", int((reviews_df["status"] == "skipped").sum()))

with top4:
    wins = int((reviews_df["result"] == "won").sum())
    losses = int((reviews_df["result"] == "lost").sum())
    st.metric("Win/Loss", f"{wins}/{losses}")

st.divider()

valid = merged[merged["result"].isin(["won", "lost", "breakeven"])].copy()
decided = merged[merged["result"].isin(["won", "lost"])].copy()

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Resolved Outcomes", len(valid))

with m2:
    st.metric("Win Rate", safe_win_rate(merged))

with m3:
    avg_score = pd.to_numeric(decided["score"], errors="coerce").dropna()
    st.metric("Avg Score (decided)", "N/A" if avg_score.empty else round(float(avg_score.mean()), 2))

with m4:
    avg_rr = pd.to_numeric(decided["rr_estimated"], errors="coerce").dropna()
    st.metric("Avg RR (decided)", "N/A" if avg_rr.empty else round(float(avg_rr.mean()), 2))

st.divider()

c1, c2 = st.columns(2)

with c1:
    st.subheader("Status Distribution")
    status_counts = reviews_df["status"].fillna("N/A").value_counts().reset_index()
    status_counts.columns = ["status", "count"]
    st.dataframe(status_counts, width="stretch")

with c2:
    st.subheader("Outcome Distribution")
    result_counts = reviews_df["result"].fillna("N/A").value_counts().reset_index()
    result_counts.columns = ["result", "count"]
    st.dataframe(result_counts, width="stretch")

st.divider()

st.subheader("Win Rate by Context")
ctx_summary = summarize_by(valid, "context")
if ctx_summary.empty:
    st.info("No context analytics yet.")
else:
    st.dataframe(ctx_summary, width="stretch")

st.divider()

st.subheader("Win Rate by Setup")
setup_summary = summarize_by(valid, "setup")
if setup_summary.empty:
    st.info("No setup analytics yet.")
else:
    st.dataframe(setup_summary, width="stretch")

st.divider()

st.subheader("Win Rate by Action")
action_summary = summarize_by(valid, "action")
if action_summary.empty:
    st.info("No action analytics yet.")
else:
    st.dataframe(action_summary, width="stretch")

st.divider()

st.subheader("Reviewed Signals Table")

view_cols = [
    "signal_id",
    "status",
    "result",
    "context",
    "setup",
    "action",
    "decision",
    "score",
    "rr_estimated",
    "note",
    "timestamp_review",
]

available_cols = [c for c in view_cols if c in merged.columns]
st.dataframe(merged[available_cols].sort_values("timestamp_review", ascending=False), width="stretch")

st.caption("Execution analytics page")