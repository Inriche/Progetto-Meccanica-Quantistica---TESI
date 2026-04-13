from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ai.compare_scoring_modes import MODES, compare_scoring_modes
from ai.model_trainer import FEATURE_SET_DEFINITIONS, train_first_model


DEFAULT_OUTPUT_CSV = "out/candidate_validation_results.csv"
DEFAULT_OUTPUT_JSON = "out/candidate_validation_results.json"
DEFAULT_MODELS_DIR = "out/models"
DEFAULT_DETAILS_DIR = "out/candidate_validation_details"


@dataclass(frozen=True)
class CandidateConfig:
    feature_set: str
    mode: str
    min_score: float


def _parse_candidate(raw: str) -> CandidateConfig:
    items = [p.strip() for p in str(raw).split(",") if p.strip()]
    kv: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        kv[key.strip().lower()] = value.strip()

    feature_set = str(kv.get("feature_set", "")).lower()
    mode = str(kv.get("mode", "")).lower()
    min_score_raw = kv.get("min_score")
    if not feature_set or not mode or min_score_raw is None:
        raise ValueError(f"Invalid candidate: {raw}. Expected feature_set=...,mode=...,min_score=...")
    if feature_set not in FEATURE_SET_DEFINITIONS:
        raise ValueError(f"Unknown feature_set={feature_set}.")
    if mode not in MODES:
        raise ValueError(f"Unknown mode={mode}.")
    return CandidateConfig(
        feature_set=feature_set,
        mode=mode,
        min_score=float(min_score_raw),
    )


def _parse_candidates(values: list[str]) -> list[CandidateConfig]:
    parsed: list[CandidateConfig] = []
    seen: set[tuple[str, str, float]] = set()
    for raw in values:
        cfg = _parse_candidate(raw)
        key = (cfg.feature_set, cfg.mode, float(cfg.min_score))
        if key in seen:
            continue
        seen.add(key)
        parsed.append(cfg)
    return parsed


def run_candidate_validation(
    *,
    candidates: list[CandidateConfig],
    dataset_path: str,
    db_path: str,
    test_start: str,
    test_end: str,
    output_csv: str,
    output_json: str,
    model_output_dir: str,
    details_output_dir: str,
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
    os.makedirs(details_output_dir, exist_ok=True)

    model_cache: dict[str, dict[str, Any]] = {}
    rows: list[pd.DataFrame] = []
    for idx, candidate in enumerate(candidates, start=1):
        feature_set = candidate.feature_set
        model_info = model_cache.get(feature_set)
        if model_info is None:
            model_path = str(Path(model_output_dir) / f"model_{feature_set}.joblib")
            artifact = train_first_model(
                dataset_path=dataset_path,
                model_path=model_path,
                feature_set=feature_set,
                random_state=random_state,
            )
            model_info = {
                "model_path": model_path,
                "holdout_accuracy": artifact.get("holdout_accuracy"),
                "rows_trained": artifact.get("rows_trained"),
            }
            model_cache[feature_set] = model_info

        safe_tag = f"{idx}_{candidate.feature_set}_{candidate.mode}_{int(round(candidate.min_score * 10))}"
        compare_csv = str(Path(details_output_dir) / f"candidate_{safe_tag}.csv")
        compare_json = str(Path(details_output_dir) / f"candidate_{safe_tag}.json")
        result_df = compare_scoring_modes(
            db_path=db_path,
            symbol=symbol,
            test_start=test_start,
            test_end=test_end,
            train_end=train_end,
            model_path=str(model_info["model_path"]),
            output_csv=compare_csv,
            output_json=compare_json,
            min_score=float(candidate.min_score),
            min_scores=[float(candidate.min_score)],
            timeframe=timeframe,
            horizon_bars=int(horizon_bars),
            initial_capital=float(initial_capital),
            risk_per_trade=float(risk_per_trade),
            max_notional_fraction=float(max_notional_fraction),
            fee_rate=float(fee_rate),
            slippage_bps=float(slippage_bps),
            modes=[candidate.mode],
        )

        if result_df.empty:
            continue

        result_df = result_df.copy()
        result_df.loc[:, "feature_set"] = candidate.feature_set
        result_df.loc[:, "model_path"] = model_info["model_path"]
        result_df.loc[:, "holdout_accuracy"] = model_info["holdout_accuracy"]
        result_df.loc[:, "rows_trained"] = model_info["rows_trained"]
        result_df.loc[:, "candidate_spec"] = (
            f"feature_set={candidate.feature_set},mode={candidate.mode},min_score={candidate.min_score}"
        )
        rows.append(result_df)

    final_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not final_df.empty:
        ordered = [
            "candidate_spec",
            "feature_set",
            "mode",
            "min_score",
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
    parser = argparse.ArgumentParser(description="Run validation round only on selected candidate configs.")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate config: feature_set=...,mode=...,min_score=... (repeatable).",
    )
    parser.add_argument("--dataset-path", default="out/training_dataset.csv")
    parser.add_argument("--db-path", default="out/assistant.db")
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--model-output-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--details-output-dir", default=DEFAULT_DETAILS_DIR)
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
    candidates = _parse_candidates(list(args.candidate or []))
    out_df = run_candidate_validation(
        candidates=candidates,
        dataset_path=str(args.dataset_path),
        db_path=str(args.db_path),
        test_start=str(args.test_start),
        test_end=str(args.test_end),
        output_csv=str(args.output_csv),
        output_json=str(args.output_json),
        model_output_dir=str(args.model_output_dir),
        details_output_dir=str(args.details_output_dir),
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
    print(f"[run_candidate_validation] csv={args.output_csv} json={args.output_json}")


if __name__ == "__main__":
    main()
