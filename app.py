import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_pipeline.pkl"
METADATA_PATH = BASE_DIR / "feature_metadata.json"
METRICS_PATH = BASE_DIR / "model_metrics.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "model_pipeline.pkl not found. Run 'python train_model.py' first."
    )

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        "feature_metadata.json not found. Run 'python train_model.py' first."
    )

model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024

NUMERIC_FIELDS = {
    field["name"] for field in metadata["fields"] if field["type"] == "number"
}


def build_input(form):
    data = {}
    errors = []

    for field in metadata["fields"]:
        name = field["name"]
        value = form.get(name, "").strip()

        if value == "":
            errors.append(f"{name} is required.")
            continue

        if name in NUMERIC_FIELDS:
            try:
                data[name] = int(value)
            except ValueError:
                errors.append(f"{name} must be a whole number.")
        else:
            data[name] = value

    if errors:
        raise ValueError(" ".join(errors))

    return pd.DataFrame([data])


@app.get("/")
def home():
    return render_template(
        "index.html",
        fields=metadata["fields"],
        metrics=metrics,
        prediction=None,
        probabilities=None,
        error=None,
        form_values={},
    )


@app.post("/predict")
def predict():
    form_values = request.form.to_dict()

    try:
        input_df = build_input(request.form)
        prediction = str(model.predict(input_df)[0])

        probabilities = None
        if hasattr(model, "predict_proba"):
            raw_probabilities = model.predict_proba(input_df)[0]
            probabilities = [
                {
                    "class": str(class_name),
                    "probability": round(float(probability) * 100, 2),
                }
                for class_name, probability in zip(model.classes_, raw_probabilities)
            ]
            probabilities.sort(key=lambda x: x["probability"], reverse=True)

        return render_template(
            "index.html",
            fields=metadata["fields"],
            metrics=metrics,
            prediction=prediction,
            probabilities=probabilities,
            error=None,
            form_values=form_values,
        )

    except Exception as exc:
        return render_template(
            "index.html",
            fields=metadata["fields"],
            metrics=metrics,
            prediction=None,
            probabilities=None,
            error=str(exc),
            form_values=form_values,
        ), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
