from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from typing import Any

import pandas as pd

try:
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except Exception as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "Missing ML dependencies. Install scikit-learn and joblib before training."
    ) from exc


DEFAULT_DATASET_PATH = "out/training_dataset.csv"
DEFAULT_MODEL_PATH = "out/model.joblib"

FEATURE_COLUMNS = ["rr_estimated", "setup", "context", "decision"]
LABEL_COLUMN = "label"


def _safe_prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    for col in FEATURE_COLUMNS + [LABEL_COLUMN]:
        if col not in out.columns:
            out.loc[:, col] = pd.NA

    out.loc[:, LABEL_COLUMN] = pd.to_numeric(out[LABEL_COLUMN], errors="coerce")
    out = out[out[LABEL_COLUMN].isin([0, 1])].copy()
    if out.empty:
        return out

    out.loc[:, "rr_estimated"] = pd.to_numeric(out["rr_estimated"], errors="coerce")
    out.loc[:, "setup"] = out["setup"].astype("string")
    out.loc[:, "context"] = out["context"].astype("string")
    out.loc[:, "decision"] = out["decision"].astype("string")
    return out


def train_first_model(
    *,
    dataset_path: str = DEFAULT_DATASET_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    random_state: int = 42,
) -> dict[str, Any]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    raw_df = pd.read_csv(dataset_path)
    df = _safe_prepare_dataset(raw_df)
    if df.empty:
        raise ValueError("Dataset is empty or has no valid labeled rows.")

    # Prefer chronological ordering when timestamp is available.
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        if ts.notna().any():
            df = df.assign(_ts_sort=ts).sort_values("_ts_sort").drop(columns=["_ts_sort"])

    X = df[FEATURE_COLUMNS].copy()
    y = df[LABEL_COLUMN].astype(int).copy()
    if y.nunique() < 2:
        raise ValueError("Training requires at least two label classes (0 and 1).")

    numeric_features = ["rr_estimated"]
    categorical_features = ["setup", "context", "decision"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

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
        "feature_columns": FEATURE_COLUMNS,
        "label_mapping": {"0": "negative", "1": "positive"},
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_trained": int(len(df)),
        "class_balance": {
            "0": int((y == 0).sum()),
            "1": int((y == 1).sum()),
        },
        "holdout_accuracy": accuracy,
    }

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(artifact, model_path)
    return artifact


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train first supervised ML model.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifact = train_first_model(
        dataset_path=str(args.dataset_path),
        model_path=str(args.model_path),
        random_state=int(args.random_state),
    )
    print(
        f"[model_trainer] model_saved={args.model_path} "
        f"rows={artifact['rows_trained']} "
        f"holdout_accuracy={artifact.get('holdout_accuracy')}"
    )


if __name__ == "__main__":
    main()
