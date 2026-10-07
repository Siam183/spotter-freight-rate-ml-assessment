import numpy as np
import pandas as pd
import joblib


EARTH_RADIUS_MILES = 3958.761


def fit_imputation_values(df):
    """
    Fit imputation statistics using the supplied training data only.
    """

    stats = {
        "weight_overall": df["weight"].median(),
        "market_overall": df["market_index"].median(),

        "weight_by_equipment": (
            df.groupby("equipment")["weight"]
            .median()
            .to_dict()
        ),

        "market_by_date": (
            df.groupby("date")["market_index"]
            .median()
            .to_dict()
        ),

        "market_by_equipment": (
            df.groupby("equipment")["market_index"]
            .median()
            .to_dict()
        ),
    }

    return stats


def apply_imputation(df, stats):
    """
    Apply previously fitted preprocessing statistics.

    The statistics must be fitted on the training data and then
    reused for validation/test data.
    """

    out = df.copy()

    # Negative weights are treated as invalid physical measurements.
    out["weight"] = out["weight"].abs()

    # Equipment-level weight median, then overall median.
    out["weight"] = out["weight"].fillna(
        out["equipment"].map(stats["weight_by_equipment"])
    )

    out["weight"] = out["weight"].fillna(
        stats["weight_overall"]
    )

    # Market index: date median -> equipment median -> overall median.
    out["market_index"] = out["market_index"].fillna(
        out["date"].map(stats["market_by_date"])
    )

    out["market_index"] = out["market_index"].fillna(
        out["equipment"].map(stats["market_by_equipment"])
    )

    out["market_index"] = out["market_index"].fillna(
        stats["market_overall"]
    )

    return out


def haversine_miles(lat1, lon1, lat2, lon2):
    """
    Calculate straight-line geographic distance between
    two latitude/longitude points in miles.
    """

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return (
        2
        * EARTH_RADIUS_MILES
        * np.arcsin(np.sqrt(a))
    )


def add_features(df):
    """
    Add engineered features used by the final CatBoost model.
    """

    out = df.copy()

    out["date"] = pd.to_datetime(out["date"])

    # Calendar features
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["day_of_month"] = out["date"].dt.day
    out["day_of_week"] = out["date"].dt.dayofweek
    out["day_of_year"] = out["date"].dt.dayofyear
    out["week_of_year"] = (
        out["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    # Coordinate differences
    out["lat_diff"] = (
        out["delivery_lat"] - out["pickup_lat"]
    )

    out["lon_diff"] = (
        out["delivery_lon"] - out["pickup_lon"]
    )

    out["abs_lat_diff"] = out["lat_diff"].abs()
    out["abs_lon_diff"] = out["lon_diff"].abs()

    # Straight-line geographic distance
    out["geo_distance"] = haversine_miles(
        out["pickup_lat"],
        out["pickup_lon"],
        out["delivery_lat"],
        out["delivery_lon"],
    )

    # Route midpoint
    out["mid_lat"] = (
        out["pickup_lat"] + out["delivery_lat"]
    ) / 2

    out["mid_lon"] = (
        out["pickup_lon"] + out["delivery_lon"]
    ) / 2

    # Derived rate-related features
    safe_distance = out["distance"].replace(
        0,
        np.nan
    )

    out["weight_per_mile"] = (
        out["weight"] / safe_distance
    )

    out["distance_market"] = (
        out["distance"] * out["market_index"]
    )

    out["distance_quote"] = (
        out["distance"] * out["quote_signal"]
    )

    out["market_quote"] = (
        out["market_index"] * out["quote_signal"]
    )

    return out


def prepare(df, stats):
    """
    Apply the complete preprocessing and feature-engineering pipeline.
    """

    cleaned = apply_imputation(
        df,
        stats
    )

    features = add_features(
        cleaned
    )

    return features


def save_preprocessing(path, stats):
    """
    Save preprocessing statistics for reuse during prediction.
    """

    joblib.dump(
        stats,
        path
    )


def load_preprocessing(path):
    """
    Load previously saved preprocessing statistics.
    """

    return joblib.load(
        path
    )
