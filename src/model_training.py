"""Train the final model and generate predictions (validation + December)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

DATA_DIR = Path("data")
# The assessment uses underscores; hyphenated names are accepted too
TRAIN_FILES = ("train_test.csv", "train-test.csv")
VALIDATION_FILES = ("validation.csv",)
DECEMBER_FILES = ("december_chart_inputs.csv", "december-chart-inputs.csv")
VALIDATION_OUTPUT = "validation_predictions.csv"

TARGET = "posted_rate"
FEATURES = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "distance", "weight", "equipment_cat",
    "month", "day", "dayofweek", "dayofyear", "is_weekend",
    "haversine_dist", "dist_ratio", "pickup_enc", "delivery_enc",
]
MODEL_PARAMS = dict(max_iter=400, learning_rate=0.03, early_stopping=False, random_state=42)


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
    """Add all derived columns used by the model."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    # Calendar features
    out["month"] = out["date"].dt.month
    out["day"] = out["date"].dt.day
    out["dayofweek"] = out["date"].dt.dayofweek
    out["dayofyear"] = out["date"].dt.dayofyear
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    # Equipment (codes computed on the current dataframe, as in the original code)
    out["equipment_cat"] = out["equipment"].astype("category").cat.codes

    # Distances
    out["haversine_dist"] = haversine(
        out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"]
    )
    out["dist_ratio"] = out["distance"] / (out["haversine_dist"] + 1e-5)

    # City encoding
    out[["pickup_enc", "delivery_enc"]] = encoder.transform(out[["pickup", "delivery"]])
    return out


def train_model(train_raw, encoder):
    """Train the HGB on log(posted_rate) after outlier filtering."""
    train = build_features(train_raw, encoder)

    # Remove outliers (rate per mile outside [0.5, 10.0])
    rpm = train[TARGET] / train["distance"]
    clean = train[(rpm >= 0.5) & (rpm <= 10.0)]
    print(f"Removed {len(train) - len(clean)} outliers from training set.")

    model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(clean[FEATURES], np.log(clean[TARGET]))
    return model


def predict_rates(model, df_raw, encoder):
    """Predict rates (original scale) for a raw dataframe."""
    return np.exp(model.predict(build_features(df_raw, encoder)[FEATURES]))


def fill_december_coordinates(december_raw, train_raw):
    """Fill Lexington -> Fort Wayne coordinates from the training data."""
    lex = train_raw.loc[train_raw["pickup"] == "Lexington", ["pickup_lat", "pickup_lon"]].iloc[0]
    fw = train_raw.loc[train_raw["delivery"] == "Fort Wayne", ["delivery_lat", "delivery_lon"]].iloc[0]

    filled = december_raw.copy()
    filled["pickup_lat"], filled["pickup_lon"] = lex["pickup_lat"], lex["pickup_lon"]
    filled["delivery_lat"], filled["delivery_lon"] = fw["delivery_lat"], fw["delivery_lon"]
    return filled


def main():
    print("Loading data...")
    train_raw = pd.read_csv(find_data_file(*TRAIN_FILES)).dropna(subset=[TARGET])
    validation_raw = pd.read_csv(find_data_file(*VALIDATION_FILES))
    december_path = find_data_file(*DECEMBER_FILES)
    december_raw = pd.read_csv(december_path)

    print("Training HistGradientBoostingRegressor model...")
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(train_raw[["pickup", "delivery"]])
    model = train_model(train_raw, encoder)
    print("Model training complete.")

    # --- Validation ---
    print("Predicting validation set rates...")
    pd.DataFrame(
        {
            "load_id": validation_raw["load_id"],
            "predicted_rate": predict_rates(model, validation_raw, encoder),
        }
    ).to_csv(VALIDATION_OUTPUT, index=False)
    print(f"Saved validation predictions to {VALIDATION_OUTPUT}")

    # --- December ---
    print("Predicting December inputs...")
    december_filled = fill_december_coordinates(december_raw, train_raw)
    december_raw["predicted_rate"] = predict_rates(model, december_filled, encoder)
    december_raw.to_csv(december_path, index=False)
    print(f"Completed and saved December inputs to {december_path}")


if __name__ == "__main__":
    main()