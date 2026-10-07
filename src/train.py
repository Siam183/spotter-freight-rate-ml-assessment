from pathlib import Path
import json

import pandas as pd
from catboost import CatBoostRegressor

from pipeline import (
    fit_imputation_values,
    prepare,
    save_preprocessing,
)


# Project root
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)


def find_training_file():
    """
    Support both the official assessment filename and
    the filename used in the local working directory.
    """

    candidates = [
        DATA_DIR / "train_test.csv",
        DATA_DIR / "train-test.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Training file not found. Expected either:\n"
        "data/train_test.csv\n"
        "or\n"
        "data/train-test.csv"
    )


def main():

    # --------------------------------------------------
    # 1. Load training data
    # --------------------------------------------------

    train_path = find_training_file()

    print(f"Loading training data: {train_path}")

    df = pd.read_csv(train_path)

    print(f"Training rows: {len(df):,}")


    # --------------------------------------------------
    # 2. Prepare target and features
    # --------------------------------------------------

    df["date"] = pd.to_datetime(df["date"])

    y = df["posted_rate"]

    # load_id is an identifier, not a predictive feature.
    X_raw = df.drop(
        columns=["posted_rate", "load_id"]
    )


    # --------------------------------------------------
    # 3. Fit preprocessing statistics
    # --------------------------------------------------

    print("Fitting preprocessing statistics...")

    preprocessing_stats = fit_imputation_values(
        X_raw
    )


    # --------------------------------------------------
    # 4. Apply preprocessing + feature engineering
    # --------------------------------------------------

    print("Creating engineered features...")

    X = prepare(
        X_raw,
        preprocessing_stats
    )

    print(f"Final feature count: {X.shape[1]}")


    # --------------------------------------------------
    # 5. Final CatBoost model
    # --------------------------------------------------

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


    # Categorical route/equipment features
    categorical_features = [
        "pickup",
        "delivery",
        "equipment",
    ]


    print("Training final CatBoost model...")

    model.fit(
        X,
        y,
        cat_features=categorical_features,
    )


    # --------------------------------------------------
    # 6. Save final model
    # --------------------------------------------------

    model_path = (
        MODEL_DIR /
        "catboost_freight_rate_final.cbm"
    )

    model.save_model(
        model_path
    )

    print(f"Model saved to: {model_path}")


    # --------------------------------------------------
    # 7. Save preprocessing statistics
    # --------------------------------------------------

    preprocessing_path = (
        MODEL_DIR /
        "final_preprocessing_artifacts.pkl"
    )

    save_preprocessing(
        preprocessing_path,
        preprocessing_stats
    )

    print(
        f"Preprocessing saved to: "
        f"{preprocessing_path}"
    )


    # --------------------------------------------------
    # 8. Save model parameters
    # --------------------------------------------------

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
    }

    parameters_path = (
        MODEL_DIR /
        "catboost_final_parameters.json"
    )

    with open(
        parameters_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            parameters,
            f,
            indent=4,
        )

    print(
        f"Parameters saved to: "
        f"{parameters_path}"
    )


    print("\nTraining complete.")


if __name__ == "__main__":
    main()
