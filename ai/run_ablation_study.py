from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from ai.compare_scoring_modes import compare_scoring_modes
from ai.model_trainer import FEATURE_SET_DEFINITIONS, train_first_model


DEFAULT_FEATURE_SETS = [
    "base",
    "base_quantum",
    "base_microstructure",
    "full",
]
DEFAULT_OUTPUT_CSV = "out/ablation_study_results.csv"
DEFAULT_OUTPUT_JSON = "out/ablation_study_results.json"
DEFAULT_MODELS_DIR = "out/models"
DEFAULT_COMPARE_DIR = "out/ablation_details"


def _parse_list_arg(value: str) -> list[str]:
    values = [v.strip().lower() for v in str(value).split(",") if v.strip()]
    return values


def _parse_min_scores(value: str | None, fallback_min_score: float | None) -> list[float | None] | None:
    if value is None:
        return None
    scores: list[float | None] = []
    for raw in str(value).split(","):
        token = raw.strip()
        if not token:
            continue
        scores.append(float(token))
    if not scores and fallback_min_score is not None:
        return [float(fallback_min_score)]
    return scores if scores else None


def run_ablation_study(
    *,
    feature_sets: list[str],
    dataset_path: str,
    db_path: str,
    test_start: str,
    test_end: str,
    min_score: float | None,
    min_scores: list[float | None] | None,
    model_output_dir: str,
    compare_output_dir: str,
    output_csv: str,
    output_json: str,
    symbol: str | None,
    train_end: str | None,
    timeframe: str,
    horizon_bars: int,
    initial_capital: float,
    risk_per_trade: float,
    max_notional_fraction: float,
    fee_rate: float,
    slippage_bps: float,
    random_state: int,
) -> pd.DataFrame:
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(compare_output_dir, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    for feature_set in feature_sets:
        if feature_set not in FEATURE_SET_DEFINITIONS:
            raise ValueError(
                f"Unknown feature_set={feature_set}. Allowed: {', '.join(FEATURE_SET_DEFINITIONS.keys())}"
            )

        model_path = str(Path(model_output_dir) / f"model_{feature_set}.joblib")
        artifact = train_first_model(
            dataset_path=dataset_path,
            model_path=model_path,
            feature_set=feature_set,
            random_state=random_state,
        )

        compare_csv = str(Path(compare_output_dir) / f"compare_{feature_set}.csv")
        compare_json = str(Path(compare_output_dir) / f"compare_{feature_set}.json")
        compare_df = compare_scoring_modes(
            db_path=db_path,
            symbol=symbol,
            test_start=test_start,
            test_end=test_end,
            train_end=train_end,
            model_path=model_path,
            output_csv=compare_csv,
            output_json=compare_json,
            min_score=min_score,
            min_scores=min_scores,
            timeframe=timeframe,
            horizon_bars=int(horizon_bars),
            initial_capital=float(initial_capital),
            risk_per_trade=float(risk_per_trade),
            max_notional_fraction=float(max_notional_fraction),
            fee_rate=float(fee_rate),
            slippage_bps=float(slippage_bps),
        )

        if compare_df.empty:
            continue

        compare_df = compare_df.copy()
        compare_df.loc[:, "feature_set"] = feature_set
        compare_df.loc[:, "model_path"] = model_path
        compare_df.loc[:, "holdout_accuracy"] = artifact.get("holdout_accuracy")
        compare_df.loc[:, "rows_trained"] = artifact.get("rows_trained")
        all_rows.append(compare_df)

    final_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if not final_df.empty:
        ordered = [
            "feature_set",
            "model_path",
            "holdout_accuracy",
            "rows_trained",
        ]
        cols = ordered + [c for c in final_df.columns if c not in ordered]
        final_df = final_df[cols].copy()

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_df.to_dict("records"), f, indent=2)
    return final_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML feature-set ablation study (train + scoring comparison).")
    parser.add_argument("--feature-sets", default=",".join(DEFAULT_FEATURE_SETS))
    parser.add_argument("--dataset-path", default="out/training_dataset.csv")
    parser.add_argument("--db-path", default="out/assistant.db")
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--min-score", type=float, default=70.0)
    parser.add_argument("--min-scores", default=None, help="Comma-separated thresholds, e.g. 40,50,60,70")
    parser.add_argument("--model-output-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--compare-output-dir", default=DEFAULT_COMPARE_DIR)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--horizon-bars", type=int, default=24)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-notional-fraction", type=float, default=1.0)
    parser.add_argument("--fee-rate", type=float, default=0.0004)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    feature_sets = _parse_list_arg(args.feature_sets)
    min_scores = _parse_min_scores(args.min_scores, args.min_score)
    out_df = run_ablation_study(
        feature_sets=feature_sets,
        dataset_path=str(args.dataset_path),
        db_path=str(args.db_path),
        test_start=str(args.test_start),
        test_end=str(args.test_end),
        min_score=None if args.min_score is None else float(args.min_score),
        min_scores=min_scores,
        model_output_dir=str(args.model_output_dir),
        compare_output_dir=str(args.compare_output_dir),
        output_csv=str(args.output_csv),
        output_json=str(args.output_json),
        symbol=None if args.symbol in (None, "", "None") else str(args.symbol),
        train_end=None if args.train_end in (None, "", "None") else str(args.train_end),
        timeframe=str(args.timeframe),
        horizon_bars=int(args.horizon_bars),
        initial_capital=float(args.initial_capital),
        risk_per_trade=float(args.risk_per_trade),
        max_notional_fraction=float(args.max_notional_fraction),
        fee_rate=float(args.fee_rate),
        slippage_bps=float(args.slippage_bps),
        random_state=int(args.random_state),
    )
    print(out_df.to_string(index=False))
    print(f"[run_ablation_study] csv={args.output_csv} json={args.output_json}")


if __name__ == "__main__":
    main()
