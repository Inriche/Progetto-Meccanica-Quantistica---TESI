import os
import sqlite3
from typing import Optional

import pandas as pd


DB_PATH = "out/assistant.db"


def get_conn():
    return sqlite3.connect(DB_PATH)


def load_events(limit: Optional[int] = None) -> pd.DataFrame:
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

    if limit is not None:
        query += f" LIMIT {int(limit)}"

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def safe_mean(series: pd.Series):
    if series is None or series.empty:
        return None
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


def count_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame(columns=[column, "count"])
    out = (
        df[column]
        .fillna("N/A")
        .astype(str)
        .value_counts()
        .rename_axis(column)
        .reset_index(name="count")
    )
    return out


def filter_trade_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df["event_type"].astype(str).eq("signal")
    return df.loc[mask].copy()
