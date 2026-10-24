"""Compare several baseline models on the time-based split (month 10)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder

DATA_DIR = Path("data")
# The assessment uses underscores; hyphenated names are accepted too
TRAIN_FILES = ("train_test.csv", "train-test.csv")
TARGET = "posted_rate"
FEATURES = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "distance", "weight", "equipment_cat",
    "month", "day", "dayofweek", "dayofyear", "is_weekend",
    "haversine_dist", "dist_ratio", "pickup_enc", "delivery_enc",
]


def find_data_file(*names):
    """Return the first existing file among data/<names>."""
    for name in names:
        path = DATA_DIR / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"None of these files were found in {DATA_DIR}/: {', '.join(names)}")


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(a)) * 3956


def build_features(df, encoder):
    """Add all derived columns used by the models."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    # Calendar features
    out["month"] = out["date"].dt.month
    out["day"] = out["date"].dt.day
    out["dayofweek"] = out["date"].dt.dayofweek
    out["dayofyear"] = out["date"].dt.dayofyear
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    # Equipment and distances
    out["equipment_cat"] = out["equipment"].astype("category").cat.codes
    out["haversine_dist"] = haversine(
        out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"]
    )
    out["dist_ratio"] = out["distance"] / (out["haversine_dist"] + 1e-5)

    # City encoding
    out[["pickup_enc", "delivery_enc"]] = encoder.transform(out[["pickup", "delivery"]])
    return out


def prepare_split():
    """Load data and return X_train, y_train, X_val, y_val (train: months 1-9, val: month 10)."""
    df = pd.read_csv(find_data_file(*TRAIN_FILES)).dropna(subset=[TARGET])

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(df[["pickup", "delivery"]])
    df = build_features(df, encoder)

    # Remove outliers (rate per mile outside [0.5, 10.0])
    rpm = df[TARGET] / df["distance"]
    df = df[(rpm >= 0.5) & (rpm <= 10.0)]

    train = df[df["month"] < 10]
    valid = df[df["month"] == 10]
    return train[FEATURES], train[TARGET], valid[FEATURES], valid[TARGET]


def main():
    print("Loading data for model evaluation...")
    X_train, y_train, X_val, y_val = prepare_split()

    # Baseline models do not accept NaN values
    X_train_filled, X_val_filled = X_train.fillna(0), X_val.fillna(0)

    # (name, model, log target?, use NaN-filled data?)
    experiments = [
        ("Linear Regression (Normal)", LinearRegression(), False, True),
        ("Linear Regression (Log)   ", LinearRegression(), True, True),
        (
            "Random Forest (Log)       ",
            RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1),
            True,
            True,
        ),
        ("HistGradientBoosting (Log)", HistGradientBoostingRegressor(random_state=42), True, False),
    ]

    print("Running baseline models:")
    for name, model, use_log, fill_nan in experiments:
        Xtr, Xva = (X_train_filled, X_val_filled) if fill_nan else (X_train, X_val)

        model.fit(Xtr, np.log(y_train) if use_log else y_train)
        preds = model.predict(Xva)
        if use_log:
            preds = np.exp(preds)

        print(
            f"  {name} -> R2: {r2_score(y_val, preds):.4f}, "
            f"MAE: {mean_absolute_error(y_val, preds):.2f}"
        )
    print()


if __name__ == "__main__":
    main()