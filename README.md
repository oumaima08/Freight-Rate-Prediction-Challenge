# Machine Learning Challenge — Freight Rate Prediction

A machine learning pipeline that predicts freight shipping rates using historical load data. The final model is a `HistGradientBoostingRegressor` trained on log-transformed rates, selected after a systematic baseline comparison and hyperparameter grid search.

---

## Project Structure

```
Machine_Learning_Challenge/
├── data/
│   ├── train-test.csv                      # Historical load data (training & testing)
│   ├── validation.csv                      # Validation set (no target column)
│   ├── validation-predictions-template.csv # Template for the submission file
│   └── december-chart-inputs.csv           # Fixed inputs for December 2025 forecast
├── src/
│   ├── benchmark_models.py   # Baseline model comparison (months 1–9 train / month 10 eval)
│   ├── model_optimization.py # Hyperparameter grid search for HistGradientBoosting
│   ├── model_training.py     # Final model training + validation & December predictions
│   └── score.py              # Output validator and December chart generator
├── scorer_results/
│   ├── hyperparameter_search_results.csv  # Grid search results for all configurations
│   └── candidate_december.png             # December 2025 predicted rate chart
├── EDA.ipynb              # Exploratory data analysis notebook
├── pipeline_runner.py     # Single entry point to run the full pipeline
└── requirements.txt       # Python dependencies
```

---

## Pipeline Overview

The full pipeline runs in three sequential steps, all orchestrated by `pipeline_runner.py`:

```
benchmark_models.py  →  model_optimization.py  →  model_training.py
```

1. **`benchmark_models.py`** — Evaluates four baseline models on a time-based split (months 1–9 for training, month 10 for validation): Linear Regression (normal and log target), Random Forest, and HistGradientBoosting.

2. **`model_optimization.py`** — Runs a 40-configuration grid search over `HistGradientBoostingRegressor` hyperparameters (`learning_rate`, `max_iter`, `target_log`). Results are saved to `scorer_results/hyperparameter_search_results.csv`.

3. **`model_training.py`** — Trains the final model on the full dataset using the best hyperparameters, then generates:
   - `validation_predictions.csv` — predictions for the validation set
   - Updated `december-chart-inputs.csv` — daily rate predictions for December 2025

---

## Features

The model uses 16 features derived from the raw data:

| Feature | Description |
|---|---|
| `pickup_lat/lon`, `delivery_lat/lon` | Geographic coordinates |
| `distance` | Stated route distance (miles) |
| `weight` | Shipment weight (lbs) |
| `equipment_cat` | Encoded equipment type |
| `month`, `day`, `dayofweek`, `dayofyear` | Calendar features |
| `is_weekend` | Binary weekend flag |
| `haversine_dist` | Great-circle distance (miles) |
| `dist_ratio` | Ratio of stated to haversine distance |
| `pickup_enc`, `delivery_enc` | Ordinal-encoded city names |

Outliers with a rate-per-mile outside `[0.5, 10.0]` are removed before training.

---

## Setup

**Requirements:** Python 3.9+

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Run the full pipeline

From the project root:

```bash
python pipeline_runner.py
```

This runs `benchmark_models.py`, then `model_optimization.py`, then `model_training.py` in sequence.

### Run individual steps

```bash
python src/benchmark_models.py    # Baseline model comparison
python src/model_optimization.py  # Hyperparameter tuning
python src/model_training.py      # Final training & prediction
```

### Validate outputs and generate chart

```bash
python src/score.py \
    --predictions validation_predictions.csv \
    --december-predictions data/december-chart-inputs.csv \
    --output-dir scorer_results
```

The scorer checks that predictions contain exactly 12,000 rows with valid IDs and positive rates, validates the 31 December rows against fixed constraints (Lexington → Fort Wayne, 360 miles, Dry Van, 32,000 lb), and saves the December chart to `scorer_results/candidate_december.png`.

---

## Model Details

| Parameter | Value |
|---|---|
| Algorithm | `HistGradientBoostingRegressor` |
| Target transform | `log(posted_rate)` |
| `max_iter` | 400 |
| `learning_rate` | 0.03 |
| `early_stopping` | False |
| `random_state` | 42 |

Predictions are back-transformed with `exp()` to return to the original dollar scale.

---

## Dependencies

| Package | Version |
|---|---|
| `matplotlib` | >=3.8, <4 |
| `numpy` | >=1.26, <3 |
| `pandas` | >=2.0, <3 |
| `scikit-learn` | >=1.0.0 |
