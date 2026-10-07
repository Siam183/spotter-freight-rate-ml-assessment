from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

from pipeline import (
    load_preprocessing,
    prepare,
)


# Project root
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
PREDICTIONS_DIR = ROOT / "predictions"

PREDICTIONS_DIR.mkdir(exist_ok=True)


def find_validation_file():
    """
    Support the official validation filename.
    """

    candidates = [
        DATA_DIR / "validation.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Validation file not found.\n"
        "Expected: data/validation.csv"
    )


def main():

    # --------------------------------------------------
    # 1. Locate files
    # --------------------------------------------------

    validation_path = find_validation_file()

    model_path = (
        MODEL_DIR /
        "catboost_freight_rate_final.cbm"
    )

    preprocessing_path = (
        MODEL_DIR /
        "final_preprocessing_artifacts.pkl"
    )


    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if not preprocessing_path.exists():
        raise FileNotFoundError(
            f"Preprocessing file not found: "
            f"{preprocessing_path}"
        )


    # --------------------------------------------------
    # 2. Load validation data
    # --------------------------------------------------

    print(
        f"Loading validation data: "
        f"{validation_path}"
    )

    df = pd.read_csv(
        validation_path
    )

    print(
        f"Validation rows: {len(df):,}"
    )


    # Keep load IDs for the final submission.
    load_ids = df["load_id"].copy()


    # --------------------------------------------------
    # 3. Prepare validation features
    # --------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"]
    )

    X_raw = df.drop(
        columns=["load_id"]
    )


    # --------------------------------------------------
    # 4. Load training preprocessing statistics
    # --------------------------------------------------

    preprocessing_stats = load_preprocessing(
        preprocessing_path
    )


    # IMPORTANT:
    # These statistics were fitted on training data.
    # They are reused here without refitting.
    X = prepare(
        X_raw,
        preprocessing_stats
    )


    # --------------------------------------------------
    # 5. Load final CatBoost model
    # --------------------------------------------------

    print(
        f"Loading model: {model_path}"
    )

    model = CatBoostRegressor()

    model.load_model(
        model_path
    )


    # --------------------------------------------------
    # 6. Generate predictions
    # --------------------------------------------------

    print("Generating validation predictions...")

    predictions = model.predict(
        X
    )


    # --------------------------------------------------
    # 7. Build required submission file
    # --------------------------------------------------

    output = pd.DataFrame({
        "load_id": load_ids,
        "predicted_rate": predictions,
    })


    # Make sure the required column order is exact.
    output = output[
        [
            "load_id",
            "predicted_rate",
        ]
    ]


    # --------------------------------------------------
    # 8. Basic validation checks
    # --------------------------------------------------

    if len(output) != 12000:
        raise ValueError(
            f"Expected 12,000 predictions, "
            f"but generated {len(output):,}."
        )

    if output["load_id"].duplicated().any():
        raise ValueError(
            "Duplicate load_id values detected."
        )

    if output["predicted_rate"].isna().any():
        raise ValueError(
            "NaN predictions detected."
        )

    if not output["predicted_rate"].map(
        lambda x: pd.notna(x) and pd.isfinite(x)
    ).all():
        raise ValueError(
            "Non-finite predictions detected."
        )


    # --------------------------------------------------
    # 9. Save prediction files
    # --------------------------------------------------

    root_output = (
        ROOT /
        "validation_predictions.csv"
    )

    prediction_output = (
        PREDICTIONS_DIR /
        "validation_predictions.csv"
    )


    output.to_csv(
        root_output,
        index=False,
    )

    output.to_csv(
        prediction_output,
        index=False,
    )


    # --------------------------------------------------
    # 10. Print summary
    # --------------------------------------------------

    print("\nPrediction complete.")

    print(
        f"Rows: {len(output):,}"
    )

    print(
        f"Minimum predicted rate: "
        f"${output['predicted_rate'].min():,.2f}"
    )

    print(
        f"Maximum predicted rate: "
        f"${output['predicted_rate'].max():,.2f}"
    )

    print(
        f"Mean predicted rate: "
        f"${output['predicted_rate'].mean():,.2f}"
    )

    print(
        f"\nSaved submission file:"
        f"\n{root_output}"
    )

    print(
        f"\nSaved copy:"
        f"\n{prediction_output}"
    )


if __name__ == "__main__":
    main()
