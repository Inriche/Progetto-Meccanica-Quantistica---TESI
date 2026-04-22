from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from typing import Any

import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, RobustScaler
except Exception as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "Missing ML dependencies. Install scikit-learn and joblib before training."
    ) from exc


DEFAULT_DATASET_PATH = "out/training_dataset.csv"
DEFAULT_MODEL_PATH = "out/model.joblib"

FEATURE_SET_DEFINITIONS: dict[str, list[str]] = {
    "base": [
        "rr_estimated",
        "setup",
        "context",
        "decision",
    ],
    "base_quantum": [
        "rr_estimated",
        "setup",
        "context",
        "decision",
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
    ],
    "base_microstructure": [
        "rr_estimated",
        "setup",
        "context",
        "decision",
        "ob_imbalance",
        "ob_raw",
        "ob_age_ms",
    ],
    "base_news": [
        "rr_estimated",
        "setup",
        "context",
        "decision",
        "news_bias",
        "news_sentiment",
        "news_impact",
        "news_score",
        "funding_rate",
        "oi_now",
        "oi_change_pct",
        "crowding",
        "strategy_mode",
        "strategy_score",
    ],
    "full": [
        "rr_estimated",
        "setup",
        "context",
        "decision",
        "action",
        "score",
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
    ],
}

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
MICROSTRUCTURE_FEATURES = {
    "ob_imbalance",
    "ob_raw",
    "ob_age_ms",
    "funding_rate",
    "oi_now",
    "oi_change_pct",
}
LABEL_COLUMN = "label"


def _resolve_feature_columns(raw_df: pd.DataFrame, feature_set: str) -> list[str]:
    requested = FEATURE_SET_DEFINITIONS.get(feature_set)
    if requested is None:
        raise ValueError(f"Unknown feature_set={feature_set}. Choose one of: {', '.join(FEATURE_SET_DEFINITIONS.keys())}")
    return [c for c in requested if c in raw_df.columns]


def _safe_prepare_dataset(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    for col in feature_columns + [LABEL_COLUMN]:
        if col not in out.columns:
            out.loc[:, col] = np.nan

    out.loc[:, LABEL_COLUMN] = pd.to_numeric(out[LABEL_COLUMN], errors="coerce")
    out = out[out[LABEL_COLUMN].isin([0, 1])].copy()
    if out.empty:
        return out

    for col in feature_columns:
        if col in NUMERIC_FEATURES:
            numeric_col = pd.to_numeric(out[col], errors="coerce").astype("float64")
            numeric_col = numeric_col.replace([np.inf, -np.inf], np.nan)
            out.loc[:, col] = numeric_col
        else:
            cat_col = out[col].astype("object")
            cat_col = cat_col.where(pd.notna(cat_col), np.nan)
            out.loc[:, col] = cat_col
    return out


def _microstructure_support_stats(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"non_null_ratio": 0.0, "informative_ratio": 0.0}

    cols = [c for c in MICROSTRUCTURE_FEATURES if c in df.columns]
    if not cols:
        return {"non_null_ratio": 0.0, "informative_ratio": 0.0}

    n = float(len(df))
    non_null_any = pd.Series(False, index=df.index)
    informative_any = pd.Series(False, index=df.index)

    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce")
        non_null_any = non_null_any | series.notna()
        if col == "ob_age_ms":
            informative_any = informative_any | (series.notna() & (series < 9_000_000))
        else:
            informative_any = informative_any | (series.notna() & (series.abs() > 1e-12))

    return {
        "non_null_ratio": float(non_null_any.sum()) / n,
        "informative_ratio": float(informative_any.sum()) / n,
    }


def train_first_model(
    *,
    dataset_path: str = DEFAULT_DATASET_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    feature_set: str = "base",
    random_state: int = 42,
    min_micro_informative_ratio: float = 0.01,
    allow_sparse_microstructure: bool = False,
) -> dict[str, Any]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    raw_df = pd.read_csv(dataset_path)
    feature_set_key = str(feature_set).strip().lower()
    selected_feature_columns = _resolve_feature_columns(raw_df, feature_set_key)
    if not selected_feature_columns:
        raise ValueError(f"No usable columns for feature_set={feature_set_key} in dataset.")

    df = _safe_prepare_dataset(raw_df, selected_feature_columns)
    if df.empty:
        raise ValueError("Dataset is empty or has no valid labeled rows.")

    support_stats: dict[str, float] | None = None
    if any(c in MICROSTRUCTURE_FEATURES for c in selected_feature_columns):
        support_stats = _microstructure_support_stats(df)
        informative_ratio = float(support_stats["informative_ratio"])
        non_null_ratio = float(support_stats["non_null_ratio"])
        msg = (
            f"Microstructure support is weak in dataset: "
            f"non_null_ratio={non_null_ratio:.4f}, informative_ratio={informative_ratio:.4f}, "
            f"feature_set={feature_set_key}"
        )
        if informative_ratio < float(min_micro_informative_ratio):
            if allow_sparse_microstructure:
                print(f"[model_trainer][warning] {msg} (training allowed by --allow-sparse-microstructure)")
            else:
                raise ValueError(
                    f"{msg}. Training blocked: historical microstructure not sufficiently supported."
                )

    # Prefer chronological ordering when timestamp is available.
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        if ts.notna().any():
            df = df.assign(_ts_sort=ts).sort_values("_ts_sort").drop(columns=["_ts_sort"])

    X = df[selected_feature_columns].copy()
    y = df[LABEL_COLUMN].astype(int).copy()
    if y.nunique() < 2:
        raise ValueError("Training requires at least two label classes (0 and 1).")

    numeric_features = [c for c in selected_feature_columns if c in NUMERIC_FEATURES]
    categorical_features = [c for c in selected_feature_columns if c not in NUMERIC_FEATURES]
    if numeric_features:
        for col in numeric_features:
            if col in X.columns:
                col_numeric = pd.to_numeric(X[col], errors="coerce").astype("float64")
                X.loc[:, col] = col_numeric.replace([np.inf, -np.inf], np.nan)
    if categorical_features:
        for col in categorical_features:
            if col in X.columns:
                col_cat = X[col].astype("object")
                X.loc[:, col] = col_cat.where(pd.notna(col_cat), np.nan)

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
                        ("scaler", RobustScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )
    if not transformers:
        raise ValueError("No features available to train the model.")

    preprocessor = ColumnTransformer(transformers=transformers)

    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    accuracy: float | None = None
    if len(df) >= 20:
        holdout_size = max(1, int(round(len(df) * 0.2)))
        holdout_size = min(holdout_size, len(df) - 1)
        X_train = X.iloc[:-holdout_size].copy()
        y_train = y.iloc[:-holdout_size].copy()
        X_test = X.iloc[-holdout_size:].copy()
        y_test = y.iloc[-holdout_size:].copy()

        if not X_train.empty and not X_test.empty and y_train.nunique() >= 2:
            pipeline.fit(X_train, y_train)
            accuracy = float(pipeline.score(X_test, y_test))

    pipeline.fit(X, y)

    artifact: dict[str, Any] = {
        "model_type": "logistic_regression",
        "pipeline": pipeline,
        "feature_set": feature_set_key,
        "feature_columns": selected_feature_columns,
        "label_mapping": {"0": "negative", "1": "positive"},
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_trained": int(len(df)),
        "class_balance": {
            "0": int((y == 0).sum()),
            "1": int((y == 1).sum()),
        },
        "holdout_accuracy": accuracy,
        "microstructure_support": support_stats,
    }

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(artifact, model_path)
    return artifact


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train first supervised ML model.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--feature-set", default="base", choices=list(FEATURE_SET_DEFINITIONS.keys()))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-micro-informative-ratio", type=float, default=0.01)
    parser.add_argument("--allow-sparse-microstructure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifact = train_first_model(
        dataset_path=str(args.dataset_path),
        model_path=str(args.model_path),
        feature_set=str(args.feature_set),
        random_state=int(args.random_state),
        min_micro_informative_ratio=float(args.min_micro_informative_ratio),
        allow_sparse_microstructure=bool(args.allow_sparse_microstructure),
    )
    print(
        f"[model_trainer] model_saved={args.model_path} "
        f"feature_set={artifact.get('feature_set')} "
        f"rows={artifact['rows_trained']} "
        f"holdout_accuracy={artifact.get('holdout_accuracy')}"
    )


if __name__ == "__main__":
    main()
