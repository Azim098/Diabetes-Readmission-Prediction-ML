import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "diabetic_data_clean.csv"
MODEL_PATH = BASE_DIR / "model_pipeline.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"
METADATA_PATH = BASE_DIR / "feature_metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
K_FEATURES = 30


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Copy diabetic_data_clean.csv into the data/ folder."
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Remove identifier / highly-missing columns if they exist.
    drop_columns = [c for c in ["encounter_id", "patient_nbr", "weight"] if c in df.columns]
    df = df.drop(columns=drop_columns, errors="ignore")

    target = "readmitted"
    if target not in df.columns:
        raise ValueError("Target column 'readmitted' was not found.")

    # Match the preprocessing used in the notebook.
    categorical_cols = df.drop(columns=[target]).select_dtypes(include=["object"]).columns.tolist()
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    X = df.drop(columns=[target])
    y = df[target]

    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    # Fit preprocessing separately only to determine the selected feature count.
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    variance_selector = VarianceThreshold(threshold=0)
    X_train_var = variance_selector.fit_transform(X_train_encoded)
    X_test_var = variance_selector.transform(X_test_encoded)

    feature_names_var = feature_names[variance_selector.get_support()]

    selector = SelectKBest(score_func=f_classif, k=K_FEATURES)
    X_train_selected = selector.fit_transform(X_train_var, y_train)
    X_test_selected = selector.transform(X_test_var)

    selected_features = feature_names_var[selector.get_support()].tolist()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # Best parameters obtained during the notebook's tuning stage.
    model = GradientBoostingClassifier(
        n_estimators=150,
        min_samples_split=2,
        min_samples_leaf=2,
        max_depth=3,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
    }

    # Build the actual deployment pipeline and refit its complete sequence on X_train.
    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("variance_selector", VarianceThreshold(threshold=0)),
            ("feature_selector", SelectKBest(score_func=f_classif, k=K_FEATURES)),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )

    final_pipeline.fit(X_train, y_train)

    joblib.dump(final_pipeline, MODEL_PATH)

    # Store metadata for the Flask form.
    fields = []
    for column in X.columns:
        if column in numerical_features:
            series = pd.to_numeric(X[column], errors="coerce")
            fields.append(
                {
                    "name": column,
                    "type": "number",
                    "min": int(series.min()),
                    "max": int(series.max()),
                    "default": int(round(series.median())),
                }
            )
        else:
            values = sorted(str(v) for v in X[column].dropna().unique())
            if "Unknown" in values:
                values.remove("Unknown")
                values.insert(0, "Unknown")
            # Long diagnosis-code categories are better as text fields.
            widget = "select" if len(values) <= 25 else "text"
            fields.append(
                {
                    "name": column,
                    "type": "categorical",
                    "widget": widget,
                    "options": values if widget == "select" else [],
                    "default": values[0] if values else "Unknown",
                    "placeholder": "Enter a diagnosis code" if column.startswith("diag_") else "",
                }
            )

    metadata = {
        "target": target,
        "classes": [str(c) for c in final_pipeline.named_steps["model"].classes_],
        "fields": fields,
        "selected_features": selected_features,
        "model": "GradientBoostingClassifier",
        "model_parameters": {
            "n_estimators": 150,
            "learning_rate": 0.1,
            "max_depth": 3,
            "min_samples_split": 2,
            "min_samples_leaf": 2,
        },
    }

    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Model saved to:", MODEL_PATH)
    print("Metadata saved to:", METADATA_PATH)
    print("Metrics saved to:", METRICS_PATH)
    print("Test Accuracy:", metrics["accuracy"])
    print("Test Weighted F1:", metrics["f1_weighted"])
    print("Selected features:", len(selected_features))


if __name__ == "__main__":
    main()
