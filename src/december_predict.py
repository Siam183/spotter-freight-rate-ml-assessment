"""
december_predict.py - Generate predictions for the December 2025 assessment evaluation dataset.
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
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_december_file() -> Path:
    """Locate December dataset."""
    candidates = [
        DATA_DIR / "december.csv",
        DATA_DIR / "december-chart-inputs.csv",
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

    print("Loading final CatBoost model...")
    model = CatBoostRegressor()
    model.load_model(model_path)

    print("Loading preprocessing artifacts...")
    preprocessing_stats = load_preprocessing(preprocess_path)

    december_file = find_december_file()
    print(f"Loading December data: {december_file}")
    df = pd.read_csv(december_file)
    print(f"December rows loaded: {len(df):,}")

    load_ids = df["load_id"].copy() if "load_id" in df.columns else df.index
    X_raw = df.drop(columns=["posted_rate", "load_id"], errors="ignore")

    print("Transforming December features...")
    X = prepare(X_raw, preprocessing_stats)

    print("Generating December predictions...")
    predictions = model.predict(X)

    output = pd.DataFrame({
        "load_id": load_ids,
        "predicted_rate": predictions
    })

    output_path = OUTPUT_DIR / "december_predictions.csv"
    output.to_csv(output_path, index=False)

    print(f"\nPredictions saved to: {output_path}")
    print(f"Prediction rows: {len(output):,}")
    print("\nFirst 10 predictions:")
    print(output.head(10))


if __name__ == "__main__":
    main()
