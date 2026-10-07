"""
src/december_predict.py - Generate predictions for data/december_chart_inputs.csv.
"""

from pathlib import Path
import pandas as pd
from catboost import CatBoostRegressor

from pipeline import (
    load_preprocessing,
    prepare,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
PREDICTIONS_DIR = ROOT / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def find_december_file() -> Path:
    """Locate December chart input dataset."""
    candidates = [
        DATA_DIR / "december_chart_inputs.csv",
        DATA_DIR / "december-chart-inputs.csv",
        DATA_DIR / "december.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("December dataset not found in data/ directory.")


def main():
    model_path = MODEL_DIR / "catboost_freight_rate_final.cbm"
    preprocess_path = MODEL_DIR / "final_preprocessing_artifacts.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not preprocess_path.exists():
        raise FileNotFoundError(f"Preprocessing file not found: {preprocess_path}")

    print("Loading CatBoost model...")
    model = CatBoostRegressor()
    model.load_model(model_path)

    print("Loading preprocessing artifacts...")
    preprocessing_stats = load_preprocessing(preprocess_path)

    december_file = find_december_file()
    print(f"Loading December input data: {december_file}")
    df = pd.read_csv(december_file)

    load_ids = df["load_id"].copy() if "load_id" in df.columns else df.index
    X_raw = df.drop(columns=["posted_rate", "load_id"], errors="ignore")

    print("Transforming features...")
    X = prepare(X_raw, preprocessing_stats)

    print("Generating predictions...")
    predictions = model.predict(X)

    output = df.copy() if "predicted_rate" not in df.columns else df.drop(columns=["predicted_rate"])
    output["predicted_rate"] = predictions

    # Save outputs
    output_path = PREDICTIONS_DIR / "december-chart-inputs-scored.csv"
    output[["load_id", "predicted_rate"]].to_csv(output_path, index=False)

    # In case data/december_chart_inputs.csv needs to be updated directly in place
    direct_december_path = DATA_DIR / "december_chart_inputs.csv"
    if direct_december_path.exists():
        output.to_csv(direct_december_path, index=False)

    print(f"Predictions successfully saved to: {output_path}")


if __name__ == "__main__":
    main()
