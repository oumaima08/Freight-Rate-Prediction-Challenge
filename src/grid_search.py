"""Grid search for HistGradientBoosting (time-based split: months 1-9 / month 10)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid

# Same folder (src/), so this import works when running `python src/grid_search.py`
from compare_models import prepare_split

SCORER_DIR = Path("scorer_results")

PARAM_GRID = {
    "target_log": [True, False],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_iter": [50, 100, 200, 400, 800],
}


def evaluate(cfg, X_train, y_train, X_val, y_val) -> dict:
    """Train one configuration and return its metrics."""
    model = HistGradientBoostingRegressor(
        learning_rate=cfg["learning_rate"],
        max_iter=cfg["max_iter"],
        early_stopping=False,
        random_state=42,
    )

    if cfg["target_log"]:
        model.fit(X_train, np.log(y_train))
        preds = np.exp(model.predict(X_val))
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

    return {
        **cfg,
        "r2": r2_score(y_val, preds),
        "mae": mean_absolute_error(y_val, preds),
        "rmse": np.sqrt(mean_squared_error(y_val, preds)),
    }


def main():
    print("Loading data for grid search CV...")
    X_train, y_train, X_val, y_val = prepare_split()

    configs = list(ParameterGrid(PARAM_GRID))
    print(f"Running large grid search CV over {len(configs)} configurations...")

    results = []
    for cfg in configs:
        row = evaluate(cfg, X_train, y_train, X_val, y_val)
        results.append(row)
        print(
            f"  TargetLog={row['target_log']} lr={row['learning_rate']} "
            f"iter={row['max_iter']} -> R2: {row['r2']:.4f}, "
            f"MAE: {row['mae']:.2f}, RMSE: {row['rmse']:.2f}"
        )

    columns = ["target_log", "learning_rate", "max_iter", "r2", "mae", "rmse"]
    df_results = pd.DataFrame(results)[columns]

    SCORER_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCORER_DIR / "tuning_results.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\nTuning complete. Results stored in {out_path}")

    best = df_results.loc[df_results["mae"].idxmin()]
    print(
        f"Best config by MAE: TargetLog={best['target_log']} "
        f"lr={best['learning_rate']} iter={best['max_iter']} -> MAE: {best['mae']:.2f}"
    )


if __name__ == "__main__":
    main()