from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict

import pandas as pd

from execution.outcome_simulator import (
    load_future_candles_after_timestamp,
    simulate_outcome_from_candles,
)


DB_PATH = "out/assistant.db"
COMPLETED_STATUSES = {"validated", "invalidated", "mixed"}


def get_conn():
    return sqlite3.connect(DB_PATH)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def load_signal_candidates(limit: int = 200) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()

    conn = get_conn()
    cols_df = pd.read_sql_query("PRAGMA table_info(signals)", conn)
    available = set(cols_df["name"].astype(str).tolist()) if not cols_df.empty else set()
    desired_cols = [
        "signal_id",
        "timestamp",
        "symbol",
        "decision",
        "setup",
        "context",
        "action",
        "entry",
        "sl",
        "tp1",
        "tp2",
        "score",
        "rr_estimated",
        "strategy_mode",
        "news_bias",
        "news_impact",
        "quantum_state",
        "quantum_coherence",
        "quantum_phase_bias",
        "quantum_interference",
        "quantum_tunneling",
        "quantum_energy",
        "quantum_decoherence_rate",
        "quantum_transition_rate",
        "quantum_dominant_mode",
        "raw_hybrid_score",
        "calibrated_hybrid_score",
    ]
    select_parts = [col if col in available else f"NULL AS {col}" for col in desired_cols]
    query = f"""
        SELECT
            {", ".join(select_parts)}
        FROM signals
        WHERE event_type = 'signal'
          AND decision IN ('BUY', 'SELL')
        ORDER BY id DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df


def evaluate_market_read(
    signal_row: Dict[str, Any],
    *,
    horizon_bars: int = 16,
    min_follow_through_pct: float = 0.0035,
    max_adverse_pct: float = 0.0025,
) -> Dict[str, Any]:
    decision = str(signal_row.get("decision", "")).upper()
    entry = _safe_float(signal_row.get("entry"))
    sl = _safe_float(signal_row.get("sl"))
    tp1 = _safe_float(signal_row.get("tp1"))
    tp2 = _safe_float(signal_row.get("tp2"))

    base = {
        "signal_id": signal_row.get("signal_id"),
        "timestamp": signal_row.get("timestamp"),
        "symbol": signal_row.get("symbol"),
        "decision": decision,
        "setup": signal_row.get("setup"),
        "context": signal_row.get("context"),
        "action": signal_row.get("action"),
        "score": signal_row.get("score"),
        "rr_estimated": signal_row.get("rr_estimated"),
        "strategy_mode": signal_row.get("strategy_mode"),
        "news_bias": signal_row.get("news_bias"),
        "news_impact": signal_row.get("news_impact"),
        "quantum_state": signal_row.get("quantum_state"),
        "quantum_coherence": signal_row.get("quantum_coherence"),
        "quantum_phase_bias": signal_row.get("quantum_phase_bias"),
        "quantum_interference": signal_row.get("quantum_interference"),
        "quantum_tunneling": signal_row.get("quantum_tunneling"),
        "quantum_energy": signal_row.get("quantum_energy"),
        "quantum_decoherence_rate": signal_row.get("quantum_decoherence_rate"),
        "quantum_transition_rate": signal_row.get("quantum_transition_rate"),
        "quantum_dominant_mode": signal_row.get("quantum_dominant_mode"),
        "raw_hybrid_score": signal_row.get("raw_hybrid_score"),
        "calibrated_hybrid_score": signal_row.get("calibrated_hybrid_score"),
        "validation_status": "no_data",
        "outcome_status": "no_data",
        "read_score": None,
        "directional_alignment": None,
        "favorable_excursion_pct": None,
        "adverse_excursion_pct": None,
        "close_move_pct": None,
        "bars_loaded": 0,
        "enough_bars": False,
        "validation_note": "No validation data available.",
    }

    if decision not in ("BUY", "SELL") or entry is None or entry <= 0:
        base["validation_note"] = "Signal is missing direction or entry."
        return base

    future_df = load_future_candles_after_timestamp(
        str(signal_row.get("timestamp", "")),
        timeframe="15m",
        limit=horizon_bars,
    )

    if future_df.empty:
        base["validation_note"] = "Future candles are not available yet."
        return base

    highest = _safe_float(future_df["high"].max())
    lowest = _safe_float(future_df["low"].min())
    last_close = _safe_float(future_df["close"].iloc[-1])

    if highest is None or lowest is None or last_close is None:
        base["validation_note"] = "Future candles are incomplete."
        return base

    if decision == "BUY":
        favorable = max(0.0, (highest - entry) / entry)
        adverse = max(0.0, (entry - lowest) / entry)
        close_move = (last_close - entry) / entry
    else:
        favorable = max(0.0, (entry - lowest) / entry)
        adverse = max(0.0, (highest - entry) / entry)
        close_move = (entry - last_close) / entry

    outcome = simulate_outcome_from_candles(
        decision=decision,
        entry=entry,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        future_df=future_df,
    )
    outcome_status = str(outcome.get("status", "no_data"))
    enough_bars = len(future_df) >= horizon_bars

    if outcome_status in ("tp1_hit", "tp2_hit"):
        validation_status = "validated"
        note = f"Market moved in favor of the signal and hit {outcome_status}."
    elif outcome_status == "sl_hit":
        validation_status = "invalidated"
        note = "Market invalidated the read and hit stop loss."
    elif not enough_bars:
        validation_status = "pending"
        note = "Validation window is still open."
    elif favorable >= min_follow_through_pct and adverse <= max_adverse_pct:
        validation_status = "validated"
        note = "Directional follow-through exceeded the validation threshold."
    elif favorable < (min_follow_through_pct * 0.5) and adverse >= max_adverse_pct:
        validation_status = "invalidated"
        note = "Adverse excursion dominated the move."
    else:
        validation_status = "mixed"
        note = "Signal had partial follow-through but no clean confirmation."

    alignment = favorable - adverse
    scale = max(min_follow_through_pct, 0.0001)
    outcome_bonus = {
        "tp2_hit": 24,
        "tp1_hit": 14,
        "sl_hit": -24,
    }.get(outcome_status, 0)
    read_score = max(
        0.0,
        min(
            100.0,
            50.0 + (alignment / scale) * 22.0 + (close_move / scale) * 8.0 + outcome_bonus,
        ),
    )

    base.update(
        {
            "validation_status": validation_status,
            "outcome_status": outcome_status,
            "read_score": round(read_score, 2),
            "directional_alignment": round(alignment, 6),
            "favorable_excursion_pct": round(favorable, 6),
            "adverse_excursion_pct": round(adverse, 6),
            "close_move_pct": round(close_move, 6),
            "bars_loaded": int(len(future_df)),
            "enough_bars": bool(enough_bars),
            "validation_note": note,
        }
    )
    return base


def load_market_read_df(
    *,
    limit: int = 200,
    horizon_bars: int = 16,
    min_follow_through_pct: float = 0.0035,
    max_adverse_pct: float = 0.0025,
) -> pd.DataFrame:
    signals_df = load_signal_candidates(limit=limit)
    if signals_df.empty:
        return pd.DataFrame()

    rows = []
    for row in signals_df.to_dict("records"):
        rows.append(
            evaluate_market_read(
                row,
                horizon_bars=horizon_bars,
                min_follow_through_pct=min_follow_through_pct,
                max_adverse_pct=max_adverse_pct,
            )
        )

    return pd.DataFrame(rows)


def summarize_market_read(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "total": 0,
            "completed": 0,
            "validated": 0,
            "invalidated": 0,
            "mixed": 0,
            "pending": 0,
            "no_data": 0,
            "validation_rate": None,
            "avg_read_score": None,
            "avg_favorable_excursion_pct": None,
            "avg_adverse_excursion_pct": None,
        }

    completed = df[df["validation_status"].isin(COMPLETED_STATUSES)].copy()
    score_series = pd.to_numeric(completed["read_score"], errors="coerce").dropna()
    favorable_series = pd.to_numeric(completed["favorable_excursion_pct"], errors="coerce").dropna()
    adverse_series = pd.to_numeric(completed["adverse_excursion_pct"], errors="coerce").dropna()

    validated = int((df["validation_status"] == "validated").sum())
    invalidated = int((df["validation_status"] == "invalidated").sum())
    mixed = int((df["validation_status"] == "mixed").sum())

    return {
        "total": int(len(df)),
        "completed": int(len(completed)),
        "validated": validated,
        "invalidated": invalidated,
        "mixed": mixed,
        "pending": int((df["validation_status"] == "pending").sum()),
        "no_data": int((df["validation_status"] == "no_data").sum()),
        "validation_rate": None if completed.empty else round((validated / len(completed)) * 100.0, 2),
        "avg_read_score": None if score_series.empty else round(float(score_series.mean()), 2),
        "avg_favorable_excursion_pct": None if favorable_series.empty else round(float(favorable_series.mean()) * 100.0, 3),
        "avg_adverse_excursion_pct": None if adverse_series.empty else round(float(adverse_series.mean()) * 100.0, 3),
    }


def summarize_market_read_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame()

    rows = []
    for value, grp in df.groupby(column, dropna=False):
        label = "N/A" if pd.isna(value) else str(value)
        completed = grp[grp["validation_status"].isin(COMPLETED_STATUSES)].copy()
        score_series = pd.to_numeric(completed["read_score"], errors="coerce").dropna()

        rows.append(
            {
                column: label,
                "count": int(len(grp)),
                "validated": int((grp["validation_status"] == "validated").sum()),
                "invalidated": int((grp["validation_status"] == "invalidated").sum()),
                "mixed": int((grp["validation_status"] == "mixed").sum()),
                "pending": int((grp["validation_status"] == "pending").sum()),
                "validation_rate": None if completed.empty else round(((completed["validation_status"] == "validated").sum() / len(completed)) * 100.0, 2),
                "avg_read_score": None if score_series.empty else round(float(score_series.mean()), 2),
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["count", "validated"], ascending=[False, False]).reset_index(drop=True)
