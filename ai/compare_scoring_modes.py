from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional

import pandas as pd

from backtest.engine import BacktestConfig, load_historical_inputs, run_backtest
from backtest.portfolio import PortfolioConfig


DEFAULT_MODEL_PATH = "out/model.joblib"
DEFAULT_OUTPUT_CSV = "out/scoring_modes_comparison.csv"
DEFAULT_OUTPUT_JSON = "out/scoring_modes_comparison.json"

MODES = ("heuristic", "hybrid", "ml")


def _parse_timestamp(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp: {value}")
    return ts


def _load_model_artifact(model_path: str) -> Optional[dict[str, Any]]:
    if not os.path.exists(model_path):
        return None
    try:
        import joblib

        artifact = joblib.load(model_path)
        if isinstance(artifact, dict) and "pipeline" in artifact:
            return artifact
    except Exception:
        return None
    return None


def _predict_positive_probability(df: pd.DataFrame, artifact: Optional[dict[str, Any]]) -> pd.Series:
    if artifact is None or df.empty:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")

    pipeline = artifact.get("pipeline")
    feature_columns = artifact.get("feature_columns", [])
    if pipeline is None or not feature_columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")

    x = pd.DataFrame({col: df[col] if col in df.columns else pd.NA for col in feature_columns})
    try:
        proba = pipeline.predict_proba(x)
        classes = getattr(pipeline, "classes_", None)
        if classes is None:
            classifier = getattr(pipeline, "named_steps", {}).get("classifier")
            classes = getattr(classifier, "classes_", None)
        if classes is None:
            return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
        classes_list = list(classes)
        if 1 not in classes_list:
            return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
        idx = classes_list.index(1)
        if proba is None or len(proba) != len(df):
            return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
        out = []
        for row in proba:
            if idx >= len(row):
                out.append(pd.NA)
            else:
                out.append(float(row[idx]))
        return pd.Series(out, index=df.index, dtype="object")
    except Exception:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")


def _apply_mode_scores(
    base_df: pd.DataFrame,
    *,
    mode: str,
    artifact: Optional[dict[str, Any]],
) -> pd.DataFrame:
    out = base_df.copy()
    out.loc[:, "score"] = pd.to_numeric(out.get("score"), errors="coerce").fillna(0.0)
    out.loc[:, "ml_probability"] = _predict_positive_probability(out, artifact)
    out.loc[:, "ml_score"] = pd.to_numeric(out["ml_probability"], errors="coerce") * 100.0

    if mode == "heuristic":
        return out

    if mode == "ml":
        out.loc[:, "score"] = out["ml_score"].where(out["ml_score"].notna(), out["score"])
        return out

    # hybrid: blend heuristic score with ML score when available.
    hybrid_score = (0.65 * out["score"]) + (0.35 * out["ml_score"])
    out.loc[:, "score"] = hybrid_score.where(out["ml_score"].notna(), out["score"])
    return out


def _apply_score_threshold(df: pd.DataFrame, min_score: Optional[float]) -> pd.DataFrame:
    if df.empty or min_score is None or "score" not in df.columns:
        return df.copy()
    out = df.copy()
    out = out[pd.to_numeric(out["score"], errors="coerce") >= float(min_score)].copy()
    return out


def _filter_test_period(df: pd.DataFrame, *, test_start: pd.Timestamp, test_end: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.assign(_ts=ts)
    out = out[out["_ts"].notna()].copy()
    out = out[(out["_ts"] >= test_start) & (out["_ts"] <= test_end)].copy()
    return out.drop(columns=["_ts"])


def compare_scoring_modes(
    *,
    db_path: str,
    symbol: Optional[str],
    test_start: str,
    test_end: str,
    train_end: Optional[str],
    model_path: str,
    output_csv: str,
    output_json: str,
    min_score: Optional[float],
    timeframe: str,
    horizon_bars: int,
    initial_capital: float,
    risk_per_trade: float,
    max_notional_fraction: float,
    fee_rate: float,
    slippage_bps: float,
) -> pd.DataFrame:
    start_ts = _parse_timestamp(test_start)
    end_ts = _parse_timestamp(test_end)
    if end_ts < start_ts:
        raise ValueError("test_end must be >= test_start")

    if train_end:
        train_end_ts = _parse_timestamp(train_end)
        if start_ts <= train_end_ts:
            raise ValueError("test_start must be strictly after train_end to avoid train/test overlap")

    shared_portfolio = PortfolioConfig(
        initial_capital=float(initial_capital),
        risk_per_trade=float(risk_per_trade),
        max_notional_fraction=float(max_notional_fraction),
        fee_rate=float(fee_rate),
        slippage_bps=float(slippage_bps),
    )
    base_config = BacktestConfig(
        db_path=db_path,
        symbol=symbol,
        input_mode="signals",
        min_score=None,
        timeframe=timeframe,
        horizon_bars=int(horizon_bars),
        close_open_positions_at_horizon=True,
        portfolio=shared_portfolio,
    )

    base_inputs = load_historical_inputs(base_config)
    base_inputs = _filter_test_period(base_inputs, test_start=start_ts, test_end=end_ts)

    artifact = _load_model_artifact(model_path)
    rows: list[dict[str, Any]] = []

    for mode in MODES:
        mode_inputs = _apply_mode_scores(base_inputs, mode=mode, artifact=artifact)
        mode_inputs = _apply_score_threshold(mode_inputs, min_score=min_score)
        result = run_backtest(
            base_config,
            inputs_df=mode_inputs,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        m = result.metrics
        rows.append(
            {
                "mode": mode,
                "test_start": start_ts.isoformat(),
                "test_end": end_ts.isoformat(),
                "symbol": symbol,
                "initial_capital": float(shared_portfolio.initial_capital),
                "risk_per_trade": float(shared_portfolio.risk_per_trade),
                "max_notional_fraction": float(shared_portfolio.max_notional_fraction),
                "fee_rate": float(shared_portfolio.fee_rate),
                "slippage_bps": float(shared_portfolio.slippage_bps),
                "horizon_bars": int(horizon_bars),
                "timeframe": timeframe,
                "min_score": min_score,
                "signals_considered": int(len(mode_inputs)),
                "trades": int(m.trades),
                "total_return": float(m.total_return),
                "win_rate": float(m.win_rate),
                "profit_factor": float(m.profit_factor),
                "expectancy": float(m.expectancy),
                "max_drawdown": float(m.max_drawdown),
                "sharpe_ratio": float(m.sharpe_ratio),
                "sortino_ratio": float(m.sortino_ratio),
                "p_value": m.p_value,
                "buy_hold_return": m.buy_hold_return,
                "outperformance_vs_benchmark": m.outperformance_vs_benchmark,
            }
        )

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return out_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare heuristic/hybrid/ml scoring modes on the same test window.")
    parser.add_argument("--db-path", default="out/assistant.db")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--test-start", required=True, help="ISO timestamp, e.g. 2026-01-01T00:00:00Z")
    parser.add_argument("--test-end", required=True, help="ISO timestamp, e.g. 2026-03-01T00:00:00Z")
    parser.add_argument("--train-end", default=None, help="Optional ISO timestamp to enforce train/test separation.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--min-score", type=float, default=70.0)
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--horizon-bars", type=int, default=24)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-notional-fraction", type=float, default=1.0)
    parser.add_argument("--fee-rate", type=float, default=0.0004)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_df = compare_scoring_modes(
        db_path=str(args.db_path),
        symbol=None if args.symbol in (None, "", "None") else str(args.symbol),
        test_start=str(args.test_start),
        test_end=str(args.test_end),
        train_end=None if args.train_end in (None, "", "None") else str(args.train_end),
        model_path=str(args.model_path),
        output_csv=str(args.output_csv),
        output_json=str(args.output_json),
        min_score=None if args.min_score is None else float(args.min_score),
        timeframe=str(args.timeframe),
        horizon_bars=int(args.horizon_bars),
        initial_capital=float(args.initial_capital),
        risk_per_trade=float(args.risk_per_trade),
        max_notional_fraction=float(args.max_notional_fraction),
        fee_rate=float(args.fee_rate),
        slippage_bps=float(args.slippage_bps),
    )
    print(out_df.to_string(index=False))
    print(f"[compare_scoring_modes] csv={args.output_csv} json={args.output_json}")


if __name__ == "__main__":
    main()
