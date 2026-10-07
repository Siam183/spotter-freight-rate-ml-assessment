"""
train.py - Script to train and persist the CatBoost model and preprocessing statistics.
"""

from pathlib import Path
import json
import pandas as pd
from catboost import CatBoostRegressor

from pipeline import (
    fit_imputation_values,
    prepare,
    save_preprocessing,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def find_training_file() -> Path:
    """Locate training dataset dynamically."""
    candidates = [
        DATA_DIR / "train_test.csv",
        DATA_DIR / "train-test.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Training file not found. Expected data/train_test.csv or data/train-test.csv"
    )


def main():
    # 1. Load Data
    train_path = find_training_file()
    print(f"Loading training data: {train_path}")
    df = pd.read_csv(train_path)
    print(f"Training rows: {len(df):,}")

    # 2. Extract Target & Features
    y = df["posted_rate"]
    X_raw = df.drop(columns=["posted_rate", "load_id"], errors="ignore")

    # 3. Fit Statistics
    print("Fitting preprocessing statistics...")
    preprocessing_stats = fit_imputation_values(X_raw)

    # 4. Transform Pipeline
    print("Creating engineered features...")
    X = prepare(X_raw, preprocessing_stats)
    print(f"Final feature count: {X.shape[1]}")

    # 5. Fit CatBoost Model
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=3,
        random_strength=1,
        bagging_temperature=0,
        loss_function="RMSE",
        eval_metric="MAE",
        random_seed=42,
        verbose=100,
    )

    categorical_features = [col for col in ["pickup", "delivery", "equipment"] if col in X.columns]

    print("Training final CatBoost model...")
    model.fit(X, y, cat_features=categorical_features)

    # 6. Save Model
    model_path = MODEL_DIR / "catboost_freight_rate_final.cbm"
    model.save_model(model_path)
    print(f"Model saved to: {model_path}")

    # 7. Save Preprocessing Artifacts
    preprocessing_path = MODEL_DIR / "final_preprocessing_artifacts.pkl"
    save_preprocessing(preprocessing_path, preprocessing_stats)
    print(f"Preprocessing saved to: {preprocessing_path}")

    # 8. Save Metadata & Config
    parameters = {
        "model": "CatBoostRegressor",
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 7,
        "l2_leaf_reg": 3,
        "random_strength": 1,
        "bagging_temperature": 0,
        "loss_function": "RMSE",
        "eval_metric": "MAE",
        "random_seed": 42,
        "categorical_features": categorical_features,
        "training_rows": int(len(df)),
        "target": "posted_rate",
        "feature_columns": list(X.columns),
    }

    parameters_path = MODEL_DIR / "catboost_final_parameters.json"
    with open(parameters_path, "w", encoding="utf-8") as f:
        json.dump(parameters, f, indent=4)

    print(f"Parameters saved to: {parameters_path}")
    print("\nTraining complete successfully.")


if __name__ == "__main__":
    main()
