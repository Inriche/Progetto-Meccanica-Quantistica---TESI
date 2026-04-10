from __future__ import annotations

import argparse
import os
import sqlite3
from typing import Any, Optional

import pandas as pd

from config import CONFIG
from execution import outcome_simulator
from validation import market_read


DEFAULT_OUTPUT_CSV = "out/training_dataset.csv"


def _configure_validation_db_path(db_path: str) -> None:
    market_read.DB_PATH = db_path
    outcome_simulator.DB_PATH = db_path


def _load_signal_feature_rows(db_path: str, limit: Optional[int] = None) -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame()

    query = """
        SELECT
            s.signal_id,
            s.timestamp,
            s.symbol,
            dl.trigger,
            s.decision,
            s.setup,
            s.context,
            s.action,
            s.score,
            s.rr_estimated,
            s.entry,
            s.sl,
            s.tp1,
            s.tp2,
            s.ob_imbalance,
            s.ob_raw,
            s.ob_age_ms,
            s.funding_rate,
            s.oi_now,
            s.oi_change_pct,
            s.crowding,
            s.strategy_mode,
            s.strategy_score,
            s.news_bias,
            s.news_sentiment,
            s.news_impact,
            s.news_score,
            s.quantum_state,
            s.quantum_coherence,
            s.quantum_phase_bias,
            s.quantum_interference,
            s.quantum_tunneling,
            s.quantum_score,
            s.ticket_path
        FROM signals s
        LEFT JOIN decision_logs dl
            ON dl.ticket_path = s.ticket_path
           AND dl.decision = s.decision
        WHERE s.event_type = 'signal'
          AND s.decision IN ('BUY', 'SELL')
        ORDER BY s.id ASC
    """

    params: tuple[Any, ...] = ()
    if limit is not None and limit > 0:
        query += " LIMIT ?"
        params = (int(limit),)

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

    if df.empty:
        return df

    # Keep one row per signal in case of duplicated decision-log joins.
    return df.drop_duplicates(subset=["signal_id"], keep="first").reset_index(drop=True)


def _derive_label(validation_status: str, outcome_status: str) -> Optional[int]:
    """
    Supervised label built from existing market validation logic:
    1 = validated directional signal, 0 = invalidated signal, None = uncertain.
    """
    v = str(validation_status).lower()
    o = str(outcome_status).lower()
    if v == "validated" or o in {"tp1_hit", "tp2_hit"}:
        return 1
    if v == "invalidated" or o == "sl_hit":
        return 0
    return None


def _label_rows(
    raw_df: pd.DataFrame,
    *,
    horizon_bars: int,
    min_follow_through_pct: float,
    max_adverse_pct: float,
) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df.copy()

    labeled_rows: list[dict[str, Any]] = []

    for row in raw_df.to_dict("records"):
        validation = market_read.evaluate_market_read(
            row,
            horizon_bars=horizon_bars,
            min_follow_through_pct=min_follow_through_pct,
            max_adverse_pct=max_adverse_pct,
        )
        validation_status = str(validation.get("validation_status", "no_data"))
        outcome_status = str(validation.get("outcome_status", "no_data"))
        label = _derive_label(validation_status, outcome_status)

        row["label"] = label
        row["label_status"] = "labeled" if label is not None else "unlabeled"
        row["validation_status"] = validation_status
        row["outcome_status"] = outcome_status
        labeled_rows.append(row)

    return pd.DataFrame(labeled_rows)


def build_training_dataset(
    *,
    db_path: str,
    output_csv_path: str,
    export_parquet: bool = True,
    include_unlabeled: bool = False,
    limit: Optional[int] = None,
    horizon_bars: int = 16,
    min_follow_through_pct: float = 0.0035,
    max_adverse_pct: float = 0.0025,
) -> pd.DataFrame:
    _configure_validation_db_path(db_path)
    raw_df = _load_signal_feature_rows(db_path=db_path, limit=limit)
    labeled_df = _label_rows(
        raw_df,
        horizon_bars=horizon_bars,
        min_follow_through_pct=min_follow_through_pct,
        max_adverse_pct=max_adverse_pct,
    )

    if labeled_df.empty:
        os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
        labeled_df.to_csv(output_csv_path, index=False)
        return labeled_df

    if not include_unlabeled:
        labeled_df = labeled_df[labeled_df["label"].notna()].copy()

    # Keep an explicit feature/label split in the exported table.
    ordered_columns = [
        "timestamp",
        "symbol",
        "trigger",
        "decision",
        "setup",
        "context",
        "action",
        "score",
        "rr_estimated",
        "ob_imbalance",
        "ob_raw",
        "ob_age_ms",
        "funding_rate",
        "oi_now",
        "oi_change_pct",
        "crowding",
        "strategy_mode",
        "strategy_score",
        "news_bias",
        "news_sentiment",
        "news_impact",
        "news_score",
        "quantum_state",
        "quantum_coherence",
        "quantum_phase_bias",
        "quantum_interference",
        "quantum_tunneling",
        "quantum_score",
        "signal_id",
        "ticket_path",
        "label",
        "label_status",
        "validation_status",
        "outcome_status",
    ]
    existing_columns = [c for c in ordered_columns if c in labeled_df.columns]
    dataset_df = labeled_df[existing_columns].copy()

    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    dataset_df.to_csv(output_csv_path, index=False)

    if export_parquet:
        parquet_path = os.path.splitext(output_csv_path)[0] + ".parquet"
        try:
            dataset_df.to_parquet(parquet_path, index=False)
        except Exception:
            pass

    return dataset_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build supervised dataset from trading logs.")
    parser.add_argument("--db-path", default=CONFIG.db_path, help="Path to sqlite database.")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="Output CSV path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of source signals.")
    parser.add_argument("--horizon-bars", type=int, default=16, help="Validation horizon in 15m bars.")
    parser.add_argument("--min-follow-through-pct", type=float, default=0.0035)
    parser.add_argument("--max-adverse-pct", type=float, default=0.0025)
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Keep rows without deterministic label.",
    )
    parser.add_argument(
        "--no-parquet",
        action="store_true",
        help="Disable parquet export.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = build_training_dataset(
        db_path=str(args.db_path),
        output_csv_path=str(args.output_csv),
        export_parquet=not bool(args.no_parquet),
        include_unlabeled=bool(args.include_unlabeled),
        limit=args.limit,
        horizon_bars=int(args.horizon_bars),
        min_follow_through_pct=float(args.min_follow_through_pct),
        max_adverse_pct=float(args.max_adverse_pct),
    )
    print(
        f"[dataset_builder] rows={len(df)} "
        f"output_csv={args.output_csv}"
    )


if __name__ == "__main__":
    main()
