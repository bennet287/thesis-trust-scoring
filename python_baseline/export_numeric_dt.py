from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


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


def tree_to_dict(clf: DecisionTreeClassifier) -> dict:
    tree = clf.tree_
    return {
        "children_left": tree.children_left.tolist(),
        "children_right": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.tolist(),
        "value": tree.value.tolist(),
        "node_count": int(tree.node_count),
        "max_depth": int(tree.max_depth),
        "n_classes": int(clf.n_classes_),
        "classes": clf.classes_.tolist(),
    }


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

    # Keep numeric columns only for Rust-friendly export
    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    X_train_num = X_train[numeric_cols].copy()
    X_test_num = X_test[numeric_cols].copy()

    print(f"Numeric feature count: {len(numeric_cols)}")
    print(f"Numeric features: {numeric_cols}")

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    criterion="entropy",
                    random_state=42,
                ),
            ),
        ]
    )

    print("Training numeric-only Decision Tree...")
    pipeline.fit(X_train_num, y_train)

    y_pred = pipeline.predict(X_test_num)
    y_prob = pipeline.predict_proba(X_test_num)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n=== Numeric DT Metrics ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    imputer: SimpleImputer = pipeline.named_steps["imputer"]
    scaler: MinMaxScaler = pipeline.named_steps["scaler"]
    model: DecisionTreeClassifier = pipeline.named_steps["model"]

    export = {
        "feature_names": numeric_cols,
        "imputer_statistics": imputer.statistics_.tolist(),
        "scaler_min": scaler.data_min_.tolist(),
        "scaler_max": scaler.data_max_.tolist(),
        "tree": tree_to_dict(model),
    }

    export_path = MODEL_DIR / "numeric_dt_export.json"
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(export, f)

    print(f"Saved Rust-friendly model export to: {export_path}")

    pipeline_path = MODEL_DIR / "numeric_dt_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    print(f"Saved numeric pipeline to: {pipeline_path}")

    metrics = {
        "model": "numeric_decision_tree",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "n_numeric_features": len(numeric_cols),
    }

    metrics_path = METRICS_DIR / "numeric_dt_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to: {metrics_path}")

    # Save one sample for Rust test
    sample = X_test_num.iloc[[0]].copy()
    sample_path = MODEL_DIR / "numeric_dt_sample_input.json"
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_names": numeric_cols,
                "values": sample.iloc[0].tolist(),
                "expected_attack_probability": float(y_prob[0]),
                "expected_trust_score": float(1.0 - y_prob[0]),
            },
            f,
            indent=2,
        )

    print(f"Saved sample input to: {sample_path}")


if __name__ == "__main__":
    main()