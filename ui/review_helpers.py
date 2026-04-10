import json
import os
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd


DB_PATH = "out/assistant.db"


def get_conn():
    return sqlite3.connect(DB_PATH)


def load_events_basic() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()

    conn = get_conn()
    query = """
        SELECT
            id,
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
            strategy_mode,
            strategy_score,
            news_bias,
            news_sentiment,
            news_impact,
            news_score,
            quantum_state,
            quantum_coherence,
            quantum_phase_bias,
            quantum_interference,
            quantum_tunneling,
            quantum_score,
            ticket_path
        FROM signals
        ORDER BY id DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_event_by_id(event_id: int) -> Optional[dict]:
    if not os.path.exists(DB_PATH):
        return None

    conn = get_conn()
    query = """
        SELECT
            id,
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
            strategy_mode,
            strategy_score,
            news_bias,
            news_sentiment,
            news_impact,
            news_score,
            quantum_state,
            quantum_coherence,
            quantum_phase_bias,
            quantum_interference,
            quantum_tunneling,
            quantum_score,
            ticket_path
        FROM signals
        WHERE id = ?
        LIMIT 1
    """
    df = pd.read_sql_query(query, conn, params=(event_id,))
    conn.close()

    if df.empty:
        return None
    return df.iloc[0].to_dict()


def load_ticket_from_path(ticket_path: Optional[str]) -> Optional[dict]:
    if not ticket_path:
        return None

    path = Path(ticket_path)
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_snapshot_from_ticket(ticket: Optional[dict]) -> Optional[str]:
    if not ticket:
        return None

    try:
        snap = ticket.get("evidence", {}).get("snapshot_path")
        if snap and Path(snap).exists():
            return snap
    except Exception:
        pass

    return None


def iso_to_ms(ts: str) -> Optional[int]:
    try:
        return int(pd.Timestamp(ts).timestamp() * 1000)
    except Exception:
        return None


def load_candles_around_event(
    event_timestamp_iso: str,
    timeframe: str = "15m",
    candles_before: int = 80,
    candles_after: int = 30,
) -> pd.DataFrame:
    """
    Load historical candles around the event timestamp, not the latest candles.
    """
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()

    event_ms = iso_to_ms(event_timestamp_iso)
    if event_ms is None:
        return pd.DataFrame()

    # 15m candle = 900_000 ms
    tf_ms_map = {
        "15m": 15 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
    }
    tf_ms = tf_ms_map.get(timeframe, 15 * 60 * 1000)

    start_ms = event_ms - candles_before * tf_ms
    end_ms = event_ms + candles_after * tf_ms

    conn = get_conn()
    query = """
        SELECT open_time, open, high, low, close
        FROM candles
        WHERE timeframe = ?
          AND open_time BETWEEN ? AND ?
        ORDER BY open_time ASC
    """
    df = pd.read_sql_query(query, conn, params=(timeframe, start_ms, end_ms))
    conn.close()

    if df.empty:
        return df

    df = df.copy()
    df.loc[:, "time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df

def split_why_reasons(why_text: Optional[str]) -> list[str]:
    if not why_text:
        return []

    parts = [p.strip() for p in str(why_text).split(";")]
    return [p for p in parts if p]


def extract_ticket_key_reasons(ticket: Optional[dict]) -> list[str]:
    if not ticket:
        return []

    try:
        reasons = ticket.get("evidence", {}).get("key_reasons", [])
        if isinstance(reasons, list):
            return [str(r).strip() for r in reasons if str(r).strip()]
    except Exception:
        pass

    return []

def classify_reason(reason: str) -> tuple[str, str]:
    """
    Returns (label, color) for a parsed reason.
    Colors are simple semantic tags for UI rendering.
    """
    r = str(reason).lower()

    if "blocked by filters" in r:
        return "Blocked", "#f97316"

    if r.startswith("rr_est=") or " rr_est=" in r:
        return "RR", "#ef4444" if "min" in r else "#eab308"

    if r.startswith("score=") or " score=" in r:
        return "Score", "#ef4444"

    if "crowding=" in r:
        if "crowded_longs" in r or "crowded_shorts" in r:
            return "Crowding", "#f97316"
        return "Crowding", "#64748b"

    if "strategy_mode=" in r or "strategy_note=" in r:
        return "Strategy", "#14b8a6"

    if "news_bias=" in r or "news " in r or "news_topic=" in r:
        return "News", "#8b5cf6"

    if "squeeze_risk=" in r:
        if "high" in r:
            return "Squeeze Risk", "#ef4444"
        if "medium" in r:
            return "Squeeze Risk", "#f97316"
        return "Squeeze Risk", "#64748b"

    if "quantum_state=" in r or "quantum " in r:
        return "Quantum", "#06b6d4"

    if "orderbook" in r:
        return "Order Book", "#0ea5e9"

    if "funding_rate=" in r:
        return "Funding", "#a78bfa"

    if "oi_now=" in r or "oi_change" in r:
        return "Open Interest", "#38bdf8"

    if "action=" in r:
        return "Action", "#16a34a"

    if "trigger=" in r:
        return "Trigger", "#64748b"

    if "context=" in r or "combined_bias=" in r or "vol=" in r:
        return "Context", "#22c55e"

    return "Info", "#334155"


def render_reason_badge(reason: str) -> str:
    label, color = classify_reason(reason)
    return f"""
    <div style="
        background:{color};
        color:white;
        padding:8px 12px;
        border-radius:10px;
        margin-bottom:8px;
        font-size:14px;
        line-height:1.35;
    ">
        <strong>{label}</strong><br>{reason}
    </div>
    """
from execution.execution_store import get_execution_note


def get_signal_id_from_ticket(ticket: Optional[dict]) -> Optional[str]:
    if not ticket:
        return None
    try:
        sig_id = ticket.get("id")
        return str(sig_id) if sig_id else None
    except Exception:
        return None
