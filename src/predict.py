"""
predict.py - Model inference and validation prediction generator.
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


def find_validation_file() -> Path:
    """Locate validation dataset."""
    path = DATA_DIR / "validation.csv"
    if path.exists():
        return path
    raise FileNotFoundError("Validation file not found. Expected: data/validation.csv")


def main():
    validation_path = find_validation_file()
    model_path = MODEL_DIR / "catboost_freight_rate_final.cbm"
    preprocessing_path = MODEL_DIR / "final_preprocessing_artifacts.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not preprocessing_path.exists():
        raise FileNotFoundError(f"Preprocessing file not found: {preprocessing_path}")

    print(f"Loading validation data: {validation_path}")
    df = pd.read_csv(validation_path)
    print(f"Validation rows: {len(df):,}")

    load_ids = df["load_id"].copy()
    X_raw = df.drop(columns=["load_id"], errors="ignore")

    print("Loading preprocessing statistics...")
    preprocessing_stats = load_preprocessing(preprocessing_path)
    X = prepare(X_raw, preprocessing_stats)

    print(f"Loading model: {model_path}")
    model = CatBoostRegressor()
    model.load_model(model_path)

    print("Generating validation predictions...")
    predictions = model.predict(X)

    output = pd.DataFrame({
        "load_id": load_ids,
        "predicted_rate": predictions,
    })[["load_id", "predicted_rate"]]

    # Validation Checks
    if output["load_id"].duplicated().any():
        raise ValueError("Duplicate load_id values detected.")
    if output["predicted_rate"].isna().any():
        raise ValueError("NaN predictions detected.")
    if not output["predicted_rate"].map(lambda x: pd.notna(x) and pd.isfinite(x)).all():
        raise ValueError("Non-finite predictions detected.")

    # Save Output
    root_output = ROOT / "validation_predictions.csv"
    prediction_output = PREDICTIONS_DIR / "validation_predictions.csv"

    output.to_csv(root_output, index=False)
    output.to_csv(prediction_output, index=False)

    print("\nPrediction complete.")
    print(f"Rows: {len(output):,}")
    print(f"Minimum predicted rate: ${output['predicted_rate'].min():,.2f}")
    print(f"Maximum predicted rate: ${output['predicted_rate'].max():,.2f}")
    print(f"Mean predicted rate: ${output['predicted_rate'].mean():,.2f}")
    print(f"\nSaved submission file: {root_output}")


if __name__ == "__main__":
    main()
