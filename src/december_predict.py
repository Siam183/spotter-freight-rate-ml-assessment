"""
december_predict.py - Generate final December freight-rate predictions.
"""

from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

from pipeline import add_features


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
PREDICTIONS_DIR = ROOT / "predictions"

PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DECEMBER LOCATION COORDINATES
# ============================================================

DECEMBER_COORDS = {
    "Lexington": (36.99152, -84.99876),
    "Fort Wayne": (41.31561, -85.36206),
}


# ============================================================
# FIND DECEMBER FILE
# ============================================================

def find_december_file() -> Path:
    candidates = [
        DATA_DIR / "december-chart-inputs.csv",
        DATA_DIR / "december_chart_inputs.csv",
        DATA_DIR / "december.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "December dataset not found in data/ directory."
    )


# ============================================================
# FIND HISTORICAL TRAINING FILE
# ============================================================

def find_training_file() -> Path:
    candidates = [
        DATA_DIR / "train-test.csv",
        DATA_DIR / "train_test.csv",
        DATA_DIR / "train-test.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Historical training dataset not found."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    model_path = MODEL_DIR / "catboost_freight_rate_final.cbm"

    december_path = find_december_file()
    training_path = find_training_file()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    print("=" * 60)
    print("DECEMBER PREDICTION PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Load December data
    # --------------------------------------------------------

    print("\nLoading December data:")
    print(december_path)

    december_df = pd.read_csv(december_path)

    print(
        f"December dataset loaded: "
        f"{december_df.shape[0]} rows, "
        f"{december_df.shape[1]} columns"
    )

    # --------------------------------------------------------
    # Load historical training data
    # --------------------------------------------------------

    print("\nLoading historical training data:")
    print(training_path)

    historical_train = pd.read_csv(training_path)

    print(
        f"Historical data loaded: "
        f"{historical_train.shape[0]} rows, "
        f"{historical_train.shape[1]} columns"
    )

    # --------------------------------------------------------
    # Prepare December dataframe
    # --------------------------------------------------------

    dec = december_df.copy()

    dec["date"] = pd.to_datetime(dec["date"])

    # Same weight cleaning used during training
    dec["weight"] = dec["weight"].abs()

    # --------------------------------------------------------
    # Add geographic coordinates
    # --------------------------------------------------------

    print("\nAdding geographic coordinates...")

    unknown_pickups = (
        set(dec["pickup"].dropna())
        - set(DECEMBER_COORDS)
    )

    unknown_deliveries = (
        set(dec["delivery"].dropna())
        - set(DECEMBER_COORDS)
    )

    if unknown_pickups:
        raise ValueError(
            "Unknown pickup locations: "
            f"{sorted(unknown_pickups)}"
        )

    if unknown_deliveries:
        raise ValueError(
            "Unknown delivery locations: "
            f"{sorted(unknown_deliveries)}"
        )

    dec["pickup_lat"] = dec["pickup"].map(
        lambda x: DECEMBER_COORDS[x][0]
    )

    dec["pickup_lon"] = dec["pickup"].map(
        lambda x: DECEMBER_COORDS[x][1]
    )

    dec["delivery_lat"] = dec["delivery"].map(
        lambda x: DECEMBER_COORDS[x][0]
    )

    dec["delivery_lon"] = dec["delivery"].map(
        lambda x: DECEMBER_COORDS[x][1]
    )

    # --------------------------------------------------------
    # October historical market_index / quote_signal
    # --------------------------------------------------------

    print("\nCalculating October historical statistics...")

    historical_train["date"] = pd.to_datetime(
        historical_train["date"]
    )

    october_train = historical_train[
        historical_train["date"].dt.month == 10
    ].copy()

    if october_train.empty:
        raise ValueError(
            "No October records found in historical training data."
        )

    oct_market = (
        october_train
        .groupby("equipment")["market_index"]
        .median()
        .to_dict()
    )

    oct_quote = (
        october_train
        .groupby("equipment")["quote_signal"]
        .median()
        .to_dict()
    )

    overall_market = historical_train[
        "market_index"
    ].median()

    overall_quote = historical_train[
        "quote_signal"
    ].median()

    dec["market_index"] = (
        dec["equipment"]
        .map(oct_market)
        .fillna(overall_market)
    )

    dec["quote_signal"] = (
        dec["equipment"]
        .map(oct_quote)
        .fillna(overall_quote)
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    print("\nCreating engineered features...")

    dec_features = add_features(dec)

    print(
        f"Engineered feature shape: "
        f"{dec_features.shape}"
    )

    # --------------------------------------------------------
    # Load final CatBoost model
    # --------------------------------------------------------

    print("\nLoading final CatBoost model...")

    final_catboost_model = CatBoostRegressor()

    final_catboost_model.load_model(model_path)

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Match exact saved-model feature order
    # --------------------------------------------------------

    model_features = final_catboost_model.feature_names_

    if not model_features:
        raise ValueError(
            "The CatBoost model does not contain feature names."
        )

    # The original final model contains date.
    # pipeline.add_features() removes date, so restore it.
    if (
        "date" in model_features
        and "date" not in dec_features.columns
    ):
        dec_features["date"] = dec["date"]

    missing_features = [
        feature
        for feature in model_features
        if feature not in dec_features.columns
    ]

    if missing_features:
        raise ValueError(
            "December engineered data is missing model features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_features
            )
        )

    X_december = dec_features[model_features]

    # --------------------------------------------------------
    # Validation checks
    # --------------------------------------------------------

    missing_values = X_december.isna().sum().sum()

    print(
        f"\nDecember model input shape: "
        f"{X_december.shape}"
    )

    print(
        f"Missing values: {missing_values}"
    )

    if missing_values > 0:
        print("\nMissing values by feature:")

        print(
            X_december.isna()
            .sum()
            .loc[lambda x: x > 0]
        )

        raise ValueError(
            "December model input contains missing values."
        )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print("\nGenerating December predictions...")

    december_pred = final_catboost_model.predict(
        X_december
    )

    # --------------------------------------------------------
    # Create output
    # --------------------------------------------------------

    december_predictions = december_df.copy()

    december_predictions["predicted_rate"] = december_pred

    # --------------------------------------------------------
    # Prediction statistics
    # --------------------------------------------------------

    print(
        "\nDecember prediction range: "
        f"{december_pred.min():.2f} "
        f"to "
        f"{december_pred.max():.2f}"
    )

    print(
        "December prediction mean: "
        f"{december_pred.mean():.2f}"
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    output_path = (
        PREDICTIONS_DIR
        / "december-chart-inputs-scored.csv"
    )

    december_predictions.to_csv(
        output_path,
        index=False
    )

    print(
        "\nPredictions successfully saved to:"
    )
    print(output_path)

    # --------------------------------------------------------
    # Display predictions
    # --------------------------------------------------------

    print("\nDecember predictions:")

    display_columns = [
        col for col in [
            "load_id",
            "pickup",
            "delivery",
            "equipment",
            "distance",
            "weight",
            "predicted_rate",
        ]
        if col in december_predictions.columns
    ]

    print(
        december_predictions[
            display_columns
        ].to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("DECEMBER PREDICTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()