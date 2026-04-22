from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, load_historical_inputs, run_backtest
from backtest.portfolio import PortfolioConfig


DEFAULT_MODEL_PATH = "out/model.joblib"
DEFAULT_OUTPUT_CSV = "out/scoring_modes_comparison.csv"
DEFAULT_OUTPUT_JSON = "out/scoring_modes_comparison.json"

MODES = ("heuristic", "hybrid", "ml")
NUMERIC_FEATURES = {
    "rr_estimated",
    "score",
    "ob_imbalance",
    "ob_raw",
    "ob_age_ms",
    "funding_rate",
    "oi_now",
    "oi_change_pct",
    "strategy_score",
    "news_sentiment",
    "news_impact",
    "news_score",
    "quantum_coherence",
    "quantum_phase_bias",
    "quantum_interference",
    "quantum_tunneling",
    "quantum_energy",
    "quantum_decoherence_rate",
    "quantum_transition_rate",
    "quantum_score",
}


LOGGER = logging.getLogger("ai.compare_scoring_modes")


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
            artifact.setdefault("__model_path", model_path)
            return artifact
    except Exception:
        return None
    return None


def _predict_positive_probability(df: pd.DataFrame, artifact: Optional[dict[str, Any]]) -> pd.Series:
    if artifact is None or df.empty:
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")

    model_path = str(artifact.get("__model_path", "unknown"))
    pipeline = artifact.get("pipeline")
    feature_columns = artifact.get("feature_columns", [])
    if pipeline is None or not feature_columns:
        LOGGER.warning(
            "[compare_scoring_modes] ml_fallback: invalid artifact model_path=%s feature_columns=%s runtime_columns=%s",
            model_path,
            feature_columns,
            list(df.columns),
        )
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")

    x = pd.DataFrame(index=df.index)
    for col in feature_columns:
        if col in df.columns:
            x.loc[:, col] = df[col]
        else:
            x.loc[:, col] = np.nan
        if col in NUMERIC_FEATURES:
            col_numeric = pd.to_numeric(x[col], errors="coerce").astype("float64")
            x.loc[:, col] = col_numeric.replace([np.inf, -np.inf], np.nan)
        else:
            col_cat = x[col].astype("object")
            x.loc[:, col] = col_cat.where(pd.notna(col_cat), np.nan)

    try:
        proba = pipeline.predict_proba(x)
        classes = getattr(pipeline, "classes_", None)
        if classes is None:
            classifier = getattr(pipeline, "named_steps", {}).get("classifier")
            classes = getattr(classifier, "classes_", None)
        if classes is None:
            LOGGER.warning(
                "[compare_scoring_modes] ml_fallback: missing classes model_path=%s feature_columns=%s runtime_columns=%s",
                model_path,
                feature_columns,
                list(df.columns),
            )
            return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
        classes_list = list(classes)
        if 1 not in classes_list:
            LOGGER.warning(
                "[compare_scoring_modes] ml_fallback: class 1 missing model_path=%s classes=%s feature_columns=%s runtime_columns=%s",
                model_path,
                classes_list,
                feature_columns,
                list(df.columns),
            )
            return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
        idx = classes_list.index(1)
        if proba is None or len(proba) != len(df):
            LOGGER.warning(
                "[compare_scoring_modes] ml_fallback: invalid predict_proba output model_path=%s feature_columns=%s runtime_columns=%s",
                model_path,
                feature_columns,
                list(df.columns),
            )
            return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
        out = []
        for row in proba:
            if idx >= len(row):
                out.append(np.nan)
            else:
                out.append(float(row[idx]))
        return pd.Series(out, index=df.index, dtype="float64")
    except Exception as ex:
        LOGGER.warning(
            "[compare_scoring_modes] ml_fallback: predict_proba exception model_path=%s error=%r feature_columns=%s runtime_columns=%s",
            model_path,
            ex,
            feature_columns,
            list(df.columns),
        )
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")


def _apply_mode_scores(
    base_df: pd.DataFrame,
    *,
    mode: str,
    artifact: Optional[dict[str, Any]],
    hybrid_score_variant: str = "calibrated",
) -> pd.DataFrame:
    out = base_df.copy()
    base_score = out["score"] if "score" in out.columns else pd.Series(0.0, index=out.index, dtype="float64")
    out.loc[:, "score"] = pd.to_numeric(base_score, errors="coerce").fillna(0.0).astype("float64")
    out = out.astype({"score": "float64"}, copy=False)
    out.loc[:, "ml_probability"] = _predict_positive_probability(out, artifact)
    out.loc[:, "ml_score"] = pd.to_numeric(out["ml_probability"], errors="coerce").astype(float) * 100.0

    if mode == "heuristic":
        return out

    if mode == "ml":
        ml_score = pd.to_numeric(out["ml_score"], errors="coerce").astype("float64")
        out.loc[:, "score"] = ml_score.where(ml_score.notna(), out["score"]).astype("float64")
        return out

    # hybrid: prefer the calibrated score when available, but preserve access
    # to the legacy/raw blend for compatibility and diagnostics.
    raw_hybrid = _safe_series(out, "raw_hybrid_score")
    calibrated_hybrid = _safe_series(out, "calibrated_hybrid_score")
    if hybrid_score_variant == "raw":
        hybrid_score = raw_hybrid.where(raw_hybrid.notna(), out["score"])
    elif hybrid_score_variant == "legacy":
        hybrid_score = ((0.65 * out["score"]) + (0.35 * out["ml_score"])).astype("float64")
    else:
        hybrid_score = calibrated_hybrid.where(calibrated_hybrid.notna(), raw_hybrid)
        hybrid_score = hybrid_score.where(hybrid_score.notna(), out["score"])
    out.loc[:, "raw_hybrid_score"] = raw_hybrid
    out.loc[:, "calibrated_hybrid_score"] = calibrated_hybrid
    out.loc[:, "score"] = hybrid_score.where(out["ml_score"].notna(), out["score"]).astype("float64")
    return out


def _safe_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").astype("float64")
    return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")


def _log_hybrid_variant_health(df: pd.DataFrame, *, variant: str, mode: str) -> None:
    if mode != "hybrid" or df.empty:
        return

    raw = _safe_series(df, "raw_hybrid_score")
    calibrated = _safe_series(df, "calibrated_hybrid_score")
    raw_nn = int(raw.notna().sum())
    cal_nn = int(calibrated.notna().sum())
    if raw_nn == 0 and cal_nn == 0:
        LOGGER.warning(
            "[compare_scoring_modes] hybrid_variant_missing mode=%s variant=%s raw_nn=0 cal_nn=0; "
            "falling back to stored score for this window",
            mode,
            variant,
        )
        return

    shared_mask = raw.notna() & calibrated.notna()
    shared_count = int(shared_mask.sum())
    if shared_count == 0:
        return

    equal_ratio = float((raw[shared_mask].round(8) == calibrated[shared_mask].round(8)).mean())
    if equal_ratio >= 0.95:
        LOGGER.warning(
            "[compare_scoring_modes] hybrid_variant_collapse mode=%s variant=%s equal_ratio=%.3f shared_rows=%s raw_nn=%s cal_nn=%s",
            mode,
            variant,
            equal_ratio,
            shared_count,
            raw_nn,
            cal_nn,
        )
    else:
        LOGGER.info(
            "[compare_scoring_modes] hybrid_variant_health mode=%s variant=%s equal_ratio=%.3f shared_rows=%s raw_nn=%s cal_nn=%s",
            mode,
            variant,
            equal_ratio,
            shared_count,
            raw_nn,
            cal_nn,
        )


def _score_summary(series: pd.Series) -> dict[str, Any]:
    cleaned = pd.to_numeric(series, errors="coerce").dropna().astype("float64")
    if cleaned.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "q05": None,
            "q10": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q90": None,
            "q95": None,
        }
    q = cleaned.quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    return {
        "count": int(len(cleaned)),
        "mean": float(cleaned.mean()),
        "std": float(cleaned.std(ddof=1)) if len(cleaned) > 1 else 0.0,
        "min": float(cleaned.min()),
        "max": float(cleaned.max()),
        "q05": float(q.loc[0.05]),
        "q10": float(q.loc[0.10]),
        "q25": float(q.loc[0.25]),
        "q50": float(q.loc[0.50]),
        "q75": float(q.loc[0.75]),
        "q90": float(q.loc[0.90]),
        "q95": float(q.loc[0.95]),
    }


def _threshold_table(series: pd.Series, thresholds: list[float]) -> list[dict[str, Any]]:
    cleaned = pd.to_numeric(series, errors="coerce").dropna().astype("float64")
    total = len(cleaned)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        count = int((cleaned >= float(threshold)).sum()) if total else 0
        rows.append(
            {
                "threshold": float(threshold),
                "count": count,
                "pct": None if total == 0 else round((count / total) * 100.0, 3),
            }
        )
    return rows


def _histogram_table(series: pd.Series, bins: list[float]) -> list[dict[str, Any]]:
    cleaned = pd.to_numeric(series, errors="coerce").dropna().astype("float64")
    if cleaned.empty:
        return []
    cuts = pd.cut(cleaned, bins=bins, include_lowest=True)
    counts = cuts.value_counts().sort_index()
    rows: list[dict[str, Any]] = []
    for interval, count in counts.items():
        rows.append({"bin": str(interval), "count": int(count)})
    return rows


def _binary_outcome_series(df: pd.DataFrame, label_column: str = "label") -> Optional[pd.Series]:
    if label_column in df.columns:
        labels = pd.to_numeric(df[label_column], errors="coerce")
        labels = labels.where(labels.isin([0, 1]), np.nan)
        if labels.notna().any():
            return labels.astype("float64")

    if "validation_status" in df.columns:
        mapped = df["validation_status"].map({"validated": 1.0, "invalidated": 0.0})
        if mapped.notna().any():
            return mapped.astype("float64")

    if "outcome_status" in df.columns:
        mapped = df["outcome_status"].map({"tp1_hit": 1.0, "tp2_hit": 1.0, "sl_hit": 0.0})
        if mapped.notna().any():
            return mapped.astype("float64")

    return None


def _decile_table(df: pd.DataFrame, *, score_column: str, label_column: str = "label") -> list[dict[str, Any]]:
    if score_column not in df.columns:
        return []
    out = df.copy()
    out.loc[:, score_column] = pd.to_numeric(out[score_column], errors="coerce")
    out = out[out[score_column].notna()].copy()
    if out.empty:
        return []

    out.loc[:, "_decile"] = pd.qcut(out[score_column].rank(method="first"), 10, labels=False, duplicates="drop")
    rows: list[dict[str, Any]] = []
    binary_outcomes = _binary_outcome_series(out, label_column=label_column)
    for decile, grp in out.groupby("_decile", dropna=True):
        row: dict[str, Any] = {
            "decile": int(decile),
            "count": int(len(grp)),
            "score_mean": float(pd.to_numeric(grp[score_column], errors="coerce").mean()),
            "score_median": float(pd.to_numeric(grp[score_column], errors="coerce").median()),
        }
        if binary_outcomes is not None:
            labels = binary_outcomes.loc[grp.index].dropna()
            if not labels.empty:
                row["label_rate"] = float(labels.mean())
                row["win_rate"] = float((labels >= 1).mean())
        if "validation_status" in grp.columns:
            row["validated_rate"] = float((grp["validation_status"] == "validated").mean())
        if "outcome_status" in grp.columns:
            row["tp_hit_rate"] = float(grp["outcome_status"].isin(["tp1_hit", "tp2_hit"]).mean())
        rows.append(row)
    return rows


def build_score_diagnostics(
    df: pd.DataFrame,
    *,
    score_column: str,
    mode: str,
    label_column: str = "label",
    thresholds: Optional[list[float]] = None,
    bins: Optional[list[float]] = None,
    include_deciles: bool = True,
) -> dict[str, Any]:
    thresholds = thresholds or [40, 45, 50, 55, 60, 65, 70]
    bins = bins or [0, 20, 30, 40, 45, 50, 55, 60, 65, 70, 80, 100]
    out = {
        "mode": mode,
        "score_column": score_column,
        "summary": _score_summary(_safe_series(df, score_column)),
        "thresholds": _threshold_table(_safe_series(df, score_column), thresholds),
        "histogram": _histogram_table(_safe_series(df, score_column), bins),
        "deciles": _decile_table(df, score_column=score_column, label_column=label_column) if include_deciles else [],
    }
    return out


def _apply_score_threshold(df: pd.DataFrame, min_score: Optional[float]) -> pd.DataFrame:
    if df.empty or min_score is None or "score" not in df.columns:
        return df.copy()
    out = df.copy()
    out = out[pd.to_numeric(out["score"], errors="coerce") >= float(min_score)].copy()
    return out


def _apply_confirmation_filter(
    df: pd.DataFrame,
    *,
    calibrated_min_score: float = 65.0,
    setup: str = "BLOCKED",
    quantum_state: str = "LOW_ENERGY",
    decision: str = "BUY",
    context: str = "trend_clean",
) -> pd.DataFrame:
    """
    Experimental regime-specific confirmation gate.

    This is intentionally narrow and should be treated as an analysis-only filter:
    it keeps the primary ranking signal unchanged while requiring the
    calibrated-hybrid top-tail subgroup that showed promise in April.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    score = pd.to_numeric(out.get("calibrated_hybrid_score"), errors="coerce")
    mask = score >= float(calibrated_min_score)
    if "setup" in out.columns:
        mask &= out["setup"].astype(str) == str(setup)
    if "quantum_state" in out.columns:
        mask &= out["quantum_state"].astype(str) == str(quantum_state)
    if "decision" in out.columns:
        mask &= out["decision"].astype(str).str.upper() == str(decision).upper()
    if "context" in out.columns:
        mask &= out["context"].astype(str) == str(context)
    return out[mask].copy()


def _parse_min_scores_arg(value: Optional[str]) -> list[Optional[float]]:
    if value is None:
        return []
    parts = [p.strip() for p in str(value).split(",")]
    scores: list[Optional[float]] = []
    for p in parts:
        if not p:
            continue
        try:
            scores.append(float(p))
        except Exception:
            continue
    return scores


def _parse_modes_arg(value: Optional[str]) -> list[str]:
    if value is None:
        return []
    parts = [p.strip().lower() for p in str(value).split(",")]
    return [p for p in parts if p in MODES]


def _parse_float_list_arg(value: Optional[str]) -> list[float]:
    if value is None:
        return []
    out: list[float] = []
    for part in str(value).split(","):
        token = part.strip()
        if not token:
            continue
        try:
            out.append(float(token))
        except Exception:
            continue
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
    min_scores: Optional[list[Optional[float]]] = None,
    timeframe: str,
    horizon_bars: int,
    initial_capital: float,
    risk_per_trade: float,
    max_notional_fraction: float,
    fee_rate: float,
    slippage_bps: float,
    modes: Optional[list[str]] = None,
    hybrid_score_variant: str = "calibrated",
    experimental_confirmation_filter: bool = False,
    confirmation_calibrated_min_score: float = 65.0,
    confirmation_setup: str = "BLOCKED",
    confirmation_quantum_state: str = "LOW_ENERGY",
    confirmation_decision: str = "BUY",
    confirmation_context: str = "trend_clean",
    diagnostics_csv: Optional[str] = None,
    diagnostics_json: Optional[str] = None,
    diagnostic_thresholds: Optional[list[float]] = None,
    diagnostic_bins: Optional[list[float]] = None,
    include_deciles: bool = True,
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
    print(f"[compare_scoring_modes] loaded_historical_inputs={len(base_inputs)}")
    base_inputs = _filter_test_period(base_inputs, test_start=start_ts, test_end=end_ts)
    print(f"[compare_scoring_modes] after_test_period_filter={len(base_inputs)}")

    artifact = _load_model_artifact(model_path)
    rows: list[dict[str, Any]] = []
    diagnostics_payload_by_mode: dict[str, dict[str, Any]] = {}
    thresholds = [s for s in (min_scores or [min_score])]
    if not thresholds:
        thresholds = [None]
    selected_modes = [m for m in (modes or list(MODES)) if m in MODES]
    if not selected_modes:
        selected_modes = list(MODES)

    for threshold in thresholds:
        for mode in selected_modes:
            mode_inputs = _apply_mode_scores(
                base_inputs,
                mode=mode,
                artifact=artifact,
                hybrid_score_variant=hybrid_score_variant,
            )
            print(f"[compare_scoring_modes] threshold={threshold} mode={mode} after_mode_score={len(mode_inputs)}")
            if experimental_confirmation_filter:
                before_filter = len(mode_inputs)
                mode_inputs = _apply_confirmation_filter(
                    mode_inputs,
                    calibrated_min_score=confirmation_calibrated_min_score,
                    setup=confirmation_setup,
                    quantum_state=confirmation_quantum_state,
                    decision=confirmation_decision,
                    context=confirmation_context,
                )
                LOGGER.info(
                    "[compare_scoring_modes] experimental_confirmation_filter applied mode=%s variant=%s "
                    "before=%s after=%s rule=calibrated>=%s setup=%s quantum_state=%s decision=%s context=%s",
                    mode,
                    hybrid_score_variant,
                    before_filter,
                    len(mode_inputs),
                    confirmation_calibrated_min_score,
                    confirmation_setup,
                    confirmation_quantum_state,
                    confirmation_decision,
                    confirmation_context,
                )
                if len(mode_inputs) < 10:
                    LOGGER.warning(
                        "[compare_scoring_modes] confirmation-filtered sample is very small mode=%s variant=%s rows=%s; conclusions are noisy",
                        mode,
                        hybrid_score_variant,
                        len(mode_inputs),
                    )
            diag_score_col = "score"
            if mode == "hybrid":
                if hybrid_score_variant == "raw" and "raw_hybrid_score" in mode_inputs.columns:
                    diag_score_col = "raw_hybrid_score"
                elif hybrid_score_variant == "legacy":
                    diag_score_col = "score"
                elif "calibrated_hybrid_score" in mode_inputs.columns:
                    diag_score_col = "calibrated_hybrid_score"

            _log_hybrid_variant_health(mode_inputs, variant=hybrid_score_variant, mode=mode)
            LOGGER.info(
                "[compare_scoring_modes] selected_score_source mode=%s variant=%s diag_score_col=%s",
                mode,
                hybrid_score_variant,
                diag_score_col,
            )

            if mode not in diagnostics_payload_by_mode:
                diagnostics_payload_by_mode[mode] = build_score_diagnostics(
                    mode_inputs,
                    score_column=diag_score_col,
                    mode=mode,
                    label_column="label" if "label" in mode_inputs.columns else "validation_status",
                    thresholds=diagnostic_thresholds,
                    bins=diagnostic_bins,
                    include_deciles=include_deciles,
                )
            mode_inputs = _apply_score_threshold(mode_inputs, min_score=threshold)
            print(f"[compare_scoring_modes] threshold={threshold} mode={mode} after_score_threshold={len(mode_inputs)}")
            if len(mode_inputs) < 20:
                LOGGER.warning(
                    "[compare_scoring_modes] low_trade_sample mode=%s threshold=%s signals=%s; threshold conclusions will be noisy",
                    mode,
                    threshold,
                    len(mode_inputs),
                )
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
                    "min_score": threshold,
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
    if diagnostics_csv or diagnostics_json:
        flat_rows: list[dict[str, Any]] = []
        for payload in diagnostics_payload_by_mode.values():
            mode = str(payload.get("mode"))
            score_column = str(payload.get("score_column"))
            summary = payload.get("summary", {})
            for key, value in summary.items():
                flat_rows.append(
                    {
                        "section": "summary",
                        "mode": mode,
                        "score_column": score_column,
                        "metric": key,
                        "value": value,
                    }
                )
            for row in payload.get("thresholds", []):
                flat_rows.append(
                    {
                        "section": "threshold",
                        "mode": mode,
                        "score_column": score_column,
                        **row,
                    }
                )
            for row in payload.get("histogram", []):
                flat_rows.append(
                    {
                        "section": "histogram",
                        "mode": mode,
                        "score_column": score_column,
                        **row,
                    }
                )
            for row in payload.get("deciles", []):
                flat_rows.append(
                    {
                        "section": "decile",
                        "mode": mode,
                        "score_column": score_column,
                        **row,
                    }
                )

        diag_df = pd.DataFrame(flat_rows)
        if diagnostics_csv:
            os.makedirs(os.path.dirname(diagnostics_csv) or ".", exist_ok=True)
            diag_df.to_csv(diagnostics_csv, index=False)
        if diagnostics_json:
            os.makedirs(os.path.dirname(diagnostics_json) or ".", exist_ok=True)
            with open(diagnostics_json, "w", encoding="utf-8") as f:
                json.dump(list(diagnostics_payload_by_mode.values()), f, indent=2)
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
    parser.add_argument("--min-scores", default=None, help="Comma-separated score thresholds, e.g. 40,50,60,70")
    parser.add_argument("--modes", default=None, help="Comma-separated modes, e.g. heuristic,hybrid,ml")
    parser.add_argument("--hybrid-score-variant", default="calibrated", choices=["legacy", "raw", "calibrated"])
    parser.add_argument(
        "--experimental-confirmation-filter",
        action="store_true",
        help="Apply the narrow experimental calibrated top-tail confirmation gate before backtest.",
    )
    parser.add_argument("--confirmation-calibrated-min-score", type=float, default=65.0)
    parser.add_argument("--confirmation-setup", default="BLOCKED")
    parser.add_argument("--confirmation-quantum-state", default="LOW_ENERGY")
    parser.add_argument("--confirmation-decision", default="BUY")
    parser.add_argument("--confirmation-context", default="trend_clean")
    parser.add_argument("--diagnostics-csv", default=None, help="Optional CSV path for score diagnostics.")
    parser.add_argument("--diagnostics-json", default=None, help="Optional JSON path for score diagnostics.")
    parser.add_argument("--diagnostic-thresholds", default="40,45,50,55,60,65,70")
    parser.add_argument("--diagnostic-bins", default="0,20,30,40,45,50,55,60,65,70,80,100")
    parser.add_argument("--no-deciles", action="store_true", help="Disable decile-level diagnostics.")
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
    min_scores = _parse_min_scores_arg(args.min_scores)
    selected_modes = _parse_modes_arg(args.modes)
    diagnostic_thresholds = _parse_float_list_arg(args.diagnostic_thresholds)
    diagnostic_bins = _parse_float_list_arg(args.diagnostic_bins)
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
        min_scores=min_scores if min_scores else None,
        timeframe=str(args.timeframe),
        horizon_bars=int(args.horizon_bars),
        initial_capital=float(args.initial_capital),
        risk_per_trade=float(args.risk_per_trade),
        max_notional_fraction=float(args.max_notional_fraction),
        fee_rate=float(args.fee_rate),
        slippage_bps=float(args.slippage_bps),
        modes=selected_modes if selected_modes else None,
        hybrid_score_variant=str(args.hybrid_score_variant),
        experimental_confirmation_filter=bool(args.experimental_confirmation_filter),
        confirmation_calibrated_min_score=float(args.confirmation_calibrated_min_score),
        confirmation_setup=str(args.confirmation_setup),
        confirmation_quantum_state=str(args.confirmation_quantum_state),
        confirmation_decision=str(args.confirmation_decision),
        confirmation_context=str(args.confirmation_context),
        diagnostics_csv=None if args.diagnostics_csv in (None, "", "None") else str(args.diagnostics_csv),
        diagnostics_json=None if args.diagnostics_json in (None, "", "None") else str(args.diagnostics_json),
        diagnostic_thresholds=diagnostic_thresholds if diagnostic_thresholds else None,
        diagnostic_bins=diagnostic_bins if diagnostic_bins else None,
        include_deciles=not bool(args.no_deciles),
    )
    print(out_df.to_string(index=False))
    print(f"[compare_scoring_modes] csv={args.output_csv} json={args.output_json}")


if __name__ == "__main__":
    main()
