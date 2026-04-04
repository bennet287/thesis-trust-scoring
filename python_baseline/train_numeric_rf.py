from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "results" / "models"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = RAW_DIR / "UNSW_NB15_testing-set.csv"


def detect_target_column(df: pd.DataFrame) -> str:
    candidate_targets = ["label", "Label", "target", "class", "Class"]
    for col in candidate_targets:
        if col in df.columns:
            return col
    raise ValueError(f"Could not detect target column. Columns: {list(df.columns)}")


def main() -> None:
    print("Loading UNSW-NB15...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    target_col = detect_target_column(train_df)

    drop_candidates = ["id", "attack_cat"]
    train_df = train_df.drop(columns=[c for c in drop_candidates if c in train_df.columns], errors="ignore")
    test_df = test_df.drop(columns=[c for c in drop_candidates if c in test_df.columns], errors="ignore")

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(int)

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(int)

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    X_train_num = X_train[numeric_cols].copy()
    X_test_num = X_test[numeric_cols].copy()

    print(f"Numeric feature count: {len(numeric_cols)}")

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training numeric-only Random Forest...")
    pipeline.fit(X_train_num, y_train)

    y_pred = pipeline.predict(X_test_num)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n=== Numeric RF Metrics ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    pipeline_path = MODEL_DIR / "numeric_rf_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    print(f"Saved numeric RF pipeline to: {pipeline_path}")

    metrics = {
        "model": "numeric_random_forest",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "n_numeric_features": len(numeric_cols),
    }

    metrics_path = METRICS_DIR / "numeric_rf_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
