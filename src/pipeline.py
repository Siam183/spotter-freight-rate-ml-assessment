"""
pipeline.py - Core data cleaning, missing value imputation, and feature engineering.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

EARTH_RADIUS_MILES = 3958.761


def fit_imputation_values(df: pd.DataFrame) -> dict:
    """Fit imputation statistics using the supplied training data only."""
    df_temp = df.copy()
    
    # Standardize date format for mapping
    if "date" in df_temp.columns:
        df_temp["date_str"] = pd.to_datetime(df_temp["date"]).dt.strftime("%Y-%m-%d")
    else:
        df_temp["date_str"] = np.nan

    stats = {
        "weight_overall": float(df_temp["weight"].median()) if "weight" in df_temp.columns else 0.0,
        "market_overall": float(df_temp["market_index"].median()) if "market_index" in df_temp.columns else 1.0,
        "weight_by_equipment": (
            df_temp.groupby("equipment")["weight"].median().to_dict()
            if "equipment" in df_temp.columns and "weight" in df_temp.columns
            else {}
        ),
        "market_by_date": (
            df_temp.groupby("date_str")["market_index"].median().to_dict()
            if "market_index" in df_temp.columns
            else {}
        ),
        "market_by_equipment": (
            df_temp.groupby("equipment")["market_index"].median().to_dict()
            if "equipment" in df_temp.columns and "market_index" in df_temp.columns
            else {}
        ),
    }

    return stats


def apply_imputation(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Apply previously fitted preprocessing statistics to new datasets."""
    out = df.copy()

    # Abs weight to handle invalid negative values
    if "weight" in out.columns:
        out["weight"] = out["weight"].abs()
        out["weight"] = out["weight"].fillna(out["equipment"].map(stats.get("weight_by_equipment", {})))
        out["weight"] = out["weight"].fillna(stats.get("weight_overall", 0.0))

    # Market index imputation chain: date -> equipment -> overall
    if "market_index" in out.columns:
        if "date" in out.columns:
            date_str_series = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
            out["market_index"] = out["market_index"].fillna(date_str_series.map(stats.get("market_by_date", {})))

        if "equipment" in out.columns:
            out["market_index"] = out["market_index"].fillna(out["equipment"].map(stats.get("market_by_equipment", {})))

        out["market_index"] = out["market_index"].fillna(stats.get("market_overall", 1.0))

    return out


def haversine_miles(lat1, lon1, lat2, lon2):
    """Calculate straight-line geographic distance between two points in miles."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered domain and calendar features."""
    out = df.copy()

    # Standardize string representations for CatBoost categorical features
    for cat_col in ["pickup", "delivery", "equipment"]:
        if cat_col in out.columns:
            out[cat_col] = out[cat_col].fillna("UNKNOWN").astype(str)

    # Calendar features
    out["date"] = pd.to_datetime(out["date"])
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["day_of_month"] = out["date"].dt.day
    out["day_of_week"] = out["date"].dt.dayofweek
    out["day_of_year"] = out["date"].dt.dayofyear
    out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)

    # Geographic features
    if {"delivery_lat", "pickup_lat", "delivery_lon", "pickup_lon"}.issubset(out.columns):
        out["lat_diff"] = out["delivery_lat"] - out["pickup_lat"]
        out["lon_diff"] = out["delivery_lon"] - out["pickup_lon"]
        out["abs_lat_diff"] = out["lat_diff"].abs()
        out["abs_lon_diff"] = out["lon_diff"].abs()

        out["geo_distance"] = haversine_miles(
            out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"]
        )

        out["mid_lat"] = (out["pickup_lat"] + out["delivery_lat"]) / 2
        out["mid_lon"] = (out["pickup_lon"] + out["delivery_lon"]) / 2

    # Interaction & derived features
    if "distance" in out.columns:
        safe_distance = out["distance"].replace(0, np.nan)

        if "weight" in out.columns:
            out["weight_per_mile"] = (out["weight"] / safe_distance).fillna(0.0)

        if "market_index" in out.columns:
            out["distance_market"] = out["distance"] * out["market_index"]

        if "quote_signal" in out.columns:
            out["distance_quote"] = out["distance"] * out["quote_signal"]

    if {"market_index", "quote_signal"}.issubset(out.columns):
        out["market_quote"] = out["market_index"] * out["quote_signal"]

    # Drop non-predictive date object
    out = out.drop(columns=["date"], errors="ignore")

    return out


def prepare(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Full preprocessing and feature engineering execution pipeline."""
    cleaned = apply_imputation(df, stats)
    features = add_features(cleaned)
    return features


def save_preprocessing(path: Path, stats: dict):
    """Save preprocessing statistics object to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(stats, path)


def load_preprocessing(path: Path) -> dict:
    """Load precomputed preprocessing statistics from disk."""
    return joblib.load(path)
