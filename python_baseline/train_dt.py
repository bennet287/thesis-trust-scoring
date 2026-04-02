from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
MODEL_DIR = PROJECT_ROOT / "results" / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_FILE = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = RAW_DIR / "UNSW_NB15_testing-set.csv"


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    return train_df, test_df


def detect_target_column(df: pd.DataFrame) -> str:
    candidate_targets = ["label", "Label", "target", "class", "Class"]
    for col in candidate_targets:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not detect target column. Available columns: {list(df.columns)}"
    )


def drop_non_feature_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    drop_candidates = ["id", "attack_cat"]
    existing = [c for c in drop_candidates if c in df.columns and c != target_col]
    return df.drop(columns=existing, errors="ignore")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    return preprocessor


def trust_to_action(trust_score: float) -> str:
    if trust_score >= 0.80:
        return "ALLOW"
    if trust_score >= 0.50:
        return "LIMIT"
    if trust_score >= 0.20:
        return "QUARANTINE"
    return "BLOCK"


def main() -> None:
    print("Loading UNSW-NB15 data...")
    train_df, test_df = load_data()

    target_col = detect_target_column(train_df)
    print(f"Detected target column: {target_col}")

    train_df = drop_non_feature_columns(train_df, target_col)
    test_df = drop_non_feature_columns(test_df, target_col)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(int)

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(int)

    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Feature columns: {len(X_train.columns)}")

    preprocessor = build_preprocessor(X_train)

    model = DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        criterion="entropy",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print("Training Decision Tree...")
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]  # probability of attack class = 1

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n=== Metrics ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # Demo on first 5 samples
    print("\n=== Demo predictions (first 5 test samples) ===")
    demo_rows = X_test.iloc[:5]
    demo_probs = pipeline.predict_proba(demo_rows)[:, 1]

    for i, attack_prob in enumerate(demo_probs, start=1):
        trust_score = 1.0 - float(attack_prob)
        action = trust_to_action(trust_score)
        print(
            f"Sample {i}: "
            f"attack_probability={attack_prob:.4f}, "
            f"trust_score={trust_score:.4f}, "
            f"action={action}"
        )

    # Save metrics
    metrics = {
        "model": "DecisionTree",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features_raw": int(len(X_train.columns)),
    }

    metrics_path = RESULTS_DIR / "dt_metrics.json"
    pd.Series(metrics).to_json(metrics_path, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")

    model_path = MODEL_DIR / "dt_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved model pipeline to: {model_path}")

    # Save a small demo output table
    demo_out = X_test.iloc[:10].copy()
    demo_out["attack_probability"] = y_prob[:10]
    demo_out["trust_score"] = 1.0 - demo_out["attack_probability"]
    demo_out["action"] = demo_out["trust_score"].apply(trust_to_action)

    demo_path = PROCESSED_DIR / "dt_demo_predictions.csv"
    demo_out.to_csv(demo_path, index=False)
    print(f"Saved demo predictions to: {demo_path}")


if __name__ == "__main__":
    main()