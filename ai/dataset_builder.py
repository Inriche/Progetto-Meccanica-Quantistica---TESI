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
MICRO_COLS = (
    "ob_imbalance",
    "ob_raw",
    "ob_age_ms",
    "funding_rate",
    "oi_now",
    "oi_change_pct",
)
MICRO_MIN_COVERAGE_WARN_PCT = 1.0


def _configure_validation_db_path(db_path: str) -> None:
    market_read.DB_PATH = db_path
    outcome_simulator.DB_PATH = db_path


def _log_microstructure_coverage(stage: str, df: pd.DataFrame) -> None:
    if df.empty:
        print(f"[dataset_builder] micro_coverage stage={stage} rows=0")
        return
    total = len(df)
    parts = [f"rows={total}"]
    for col in MICRO_COLS:
        if col in df.columns:
            nn = int(df[col].notna().sum())
            pct = (100.0 * nn / total) if total > 0 else 0.0
            parts.append(f"{col}_nn={nn} ({pct:.1f}%)")
    print(f"[dataset_builder] micro_coverage stage={stage} " + " ".join(parts))
    if stage == "raw_signals":
        present_cols = [c for c in MICRO_COLS if c in df.columns]
        if present_cols:
            mean_cov = float(sum((100.0 * float(df[c].notna().sum()) / total) for c in present_cols) / len(present_cols))
            if mean_cov < float(MICRO_MIN_COVERAGE_WARN_PCT):
                print(
                    "[dataset_builder][warning] historical microstructure coverage is very low. "
                    "Feature-set microstructure/full may be weak unless signals are regenerated with supported sources."
                )


def _load_signal_feature_rows(db_path: str, limit: Optional[int] = None) -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame()

    desired_cols = [
        "signal_id",
        "timestamp",
        "symbol",
        "decision",
        "setup",
        "context",
        "action",
        "score",
        "rr_estimated",
        "entry",
        "sl",
        "tp1",
        "tp2",
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
        "quantum_energy",
        "quantum_decoherence_rate",
        "quantum_transition_rate",
        "quantum_dominant_mode",
        "quantum_score",
        "ticket_path",
    ]

    params: tuple[Any, ...] = ()
    conn = sqlite3.connect(db_path)
    try:
        cols_df = pd.read_sql_query("PRAGMA table_info(signals)", conn)
        available = set(cols_df["name"].astype(str).tolist()) if not cols_df.empty else set()
        select_parts = []
        for col in desired_cols:
            if col in available:
                select_parts.append(f"s.{col}")
            else:
                select_parts.append(f"NULL AS {col}")
        query = f"""
            SELECT
                {", ".join(select_parts)},
                NULL AS trigger
            FROM signals s
            WHERE s.event_type = 'signal'
            ORDER BY s.id ASC
        """
        if limit is not None and limit > 0:
            query += " LIMIT ?"
            params = (int(limit),)
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

    if df.empty:
        return df

    # signals is the source of truth for point-in-time features.
    return df.drop_duplicates(subset=["signal_id"], keep="first").reset_index(drop=True)


def _infer_decision(row: dict[str, Any]) -> Optional[str]:
    decision = str(row.get("decision", "")).upper().strip()
    if decision in ("BUY", "SELL"):
        return decision
    try:
        entry = float(row.get("entry"))
        tp1 = float(row.get("tp1"))
    except Exception:
        return None
    if entry <= 0:
        return None
    if tp1 > entry:
        return "BUY"
    if tp1 < entry:
        return "SELL"
    return None


def _derive_label(validation_status: str, outcome_status: str) -> Optional[int]:
    """
    Supervised label built from existing market validation logic:
    1 = validated directional signal, 0 = non-desirable outcome, None = uncertain.
    Mixed/no clear follow-through cases are treated as non-desirable for supervised training.
    """
    v = str(validation_status).lower()
    o = str(outcome_status).lower()
    if v == "validated" or o in {"tp1_hit", "tp2_hit"}:
        return 1
    if v in {"invalidated", "mixed"} or o == "sl_hit":
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
    unlabeled_reasons: dict[str, int] = {}

    for row in raw_df.to_dict("records"):
        row["decision"] = _infer_decision(row)
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
        if label is None:
            reason_key = f"{validation_status}|{outcome_status}"
            unlabeled_reasons[reason_key] = unlabeled_reasons.get(reason_key, 0) + 1
        labeled_rows.append(row)

    out_df = pd.DataFrame(labeled_rows)
    out_df.attrs["unlabeled_reasons"] = unlabeled_reasons
    return out_df


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
    print(f"[dataset_builder] loaded_signals={len(raw_df)}")
    _log_microstructure_coverage("raw_signals", raw_df)

    labeled_df = _label_rows(
        raw_df,
        horizon_bars=horizon_bars,
        min_follow_through_pct=min_follow_through_pct,
        max_adverse_pct=max_adverse_pct,
    )
    unlabeled_reasons = labeled_df.attrs.get("unlabeled_reasons", {})
    print(
        f"[dataset_builder] labeled_candidates={len(labeled_df)} "
        f"deterministic_labels={int(pd.to_numeric(labeled_df.get('label'), errors='coerce').notna().sum()) if not labeled_df.empty else 0} "
        f"discarded={int(pd.to_numeric(labeled_df.get('label'), errors='coerce').isna().sum()) if not labeled_df.empty else 0}"
    )
    if unlabeled_reasons:
        top_reason, top_count = max(unlabeled_reasons.items(), key=lambda x: x[1])
        print(f"[dataset_builder] top_discard_reason={top_reason} count={top_count}")

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
        "quantum_energy",
        "quantum_decoherence_rate",
        "quantum_transition_rate",
        "quantum_dominant_mode",
        "quantum_score",
        "signal_id",
        "ticket_path",
        "label",
        "label_status",
        "validation_status",
        "outcome_status",
    ]
    unique_ordered_columns = list(dict.fromkeys(ordered_columns))
    existing_columns = [c for c in unique_ordered_columns if c in labeled_df.columns]
    dataset_df = labeled_df[existing_columns].copy()
    _log_microstructure_coverage("final_dataset", dataset_df)

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
