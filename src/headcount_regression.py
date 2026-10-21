"""
Predict HOW MANY people are in the room (headcount), not just whether it is
occupied, using Decision Tree Regression.

The classification models in ``train_model.py`` and ``optimisation.py`` answer
a yes/no question. This script extends the project to a regression problem on
the same data, split and folds, so that the number of occupants can be
estimated from the same environmental readings and time features.

Models compared (all fitted on the training portion only):
  1. Mean baseline: always predict the average training headcount. Any useful
     model must beat this.
  2. Linear Regression: the simplest regression model, a weighted sum of the
     13 inputs, as a reference point.
  3. Decision Tree Regression: splits the inputs into regions with a series of
     yes/no questions (e.g. "is CO2 above 520 ppm?") and predicts the mean
     headcount of the training intervals in each region. Its maximum depth is
     chosen by 5-fold time-series cross-validation on the training data (lowest
     mean absolute error), never by looking at the test period.

Validation: the same chronological 80/20 split and the same 5 TimeSeriesSplit
folds as the classification models. Metrics: mean absolute error (MAE, in
people), root mean squared error (RMSE, in people) and R^2.

Outputs:
    reports/regression_depth_selection.csv   CV MAE for each candidate depth
    reports/regression_results.csv           test-period metrics per model
    reports/regression_cv.csv                CV MAE per fold per model
    reports/regression_test_predictions.csv  actual vs predicted headcount
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "room_dataset.csv"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

FEATURES = [
    "Carbon dioxide", "Temperature", "Humidity", "Formaldehyde",
    "Light Volatile Organic Compounds", "Pressure",
    "Particulate matter 1", "Particulate matter 2.5",
    "Particulate matter 4", "Particulate matter 10",
    "hour", "dayofweek", "is_weekend",
]
TARGET = "headcount"
TRAIN_FRACTION = 0.8  # must match train_model.py
N_SPLITS = 5  # must match train_model.py
CANDIDATE_DEPTHS: list[int | None] = [2, 3, 4, 5, 6, 8, 10, None]
MIN_SAMPLES_LEAF = 10  # stops leaves being fitted to a handful of intervals
RANDOM_STATE = 42


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict:
    """Score a set of headcount predictions.

    Args:
        y_true: Actual headcounts.
        y_pred: Predicted headcounts, same length as ``y_true``.

    Returns:
        Dict with ``mae`` (mean absolute error, people), ``rmse`` (root mean
        squared error, people; penalises large misses more than MAE) and
        ``r2`` (fraction of the variance in ``y_true`` explained; 1 is perfect,
        0 is no better than predicting the mean of ``y_true``, and it can be
        negative).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def cv_mae(make_model: Callable[[], object], X: pd.DataFrame, y: np.ndarray,
           n_splits: int = N_SPLITS) -> list[float]:
    """Mean absolute error on each time-ordered cross-validation fold.

    Args:
        make_model: Function returning a fresh, unfitted scikit-learn regressor.
        X: Training inputs, in time order.
        y: Training headcounts, in time order.
        n_splits: Number of TimeSeriesSplit folds.

    Returns:
        One MAE per fold. Each fold trains on all earlier data and validates on
        the next block, so no fold ever sees the future.
    """
    scores = []
    for train_idx, val_idx in TimeSeriesSplit(n_splits).split(X):
        model = make_model()
        model.fit(X.iloc[train_idx], y[train_idx])
        scores.append(mean_absolute_error(y[val_idx], model.predict(X.iloc[val_idx])))
    return scores


def select_tree_depth(X: pd.DataFrame, y: np.ndarray,
                      depths: Sequence[int | None] = CANDIDATE_DEPTHS,
                      n_splits: int = N_SPLITS) -> tuple[int | None, pd.DataFrame]:
    """Choose the Decision Tree's maximum depth by time-series cross-validation.

    Args:
        X: Training inputs, in time order.
        y: Training headcounts, in time order.
        depths: Candidate maximum depths; ``None`` means grow until the leaf
            size limit stops it.
        n_splits: Number of TimeSeriesSplit folds.

    Returns:
        ``(best_depth, table)`` where ``best_depth`` has the lowest mean CV MAE
        (ties go to the earlier, shallower candidate) and ``table`` has one row
        per candidate with columns ``max_depth``, ``mean_cv_mae`` and
        ``std_cv_mae``.
    """
    rows = []
    for depth in depths:
        scores = cv_mae(
            lambda d=depth: DecisionTreeRegressor(
                max_depth=d, min_samples_leaf=MIN_SAMPLES_LEAF, random_state=RANDOM_STATE
            ),
            X, y, n_splits,
        )
        rows.append({"max_depth": depth, "mean_cv_mae": np.mean(scores), "std_cv_mae": np.std(scores)})
    table = pd.DataFrame(rows)
    # Keep depths readable (6, not 6.0) and show "no limit" explicitly.
    table["max_depth"] = pd.Series(["None" if d is None else d for d in depths], dtype=object)
    best_pos = int(np.argmin(table["mean_cv_mae"].to_numpy()))  # first minimum = shallowest tie
    return depths[best_pos], table


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp")
    df = df.reset_index(drop=True)
    split_idx = int(len(df) * TRAIN_FRACTION)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]
    X_train, y_train = train[FEATURES], train[TARGET].to_numpy(dtype=float)
    X_test, y_test = test[FEATURES], test[TARGET].to_numpy(dtype=float)
    print(f"Train headcount: mean {y_train.mean():.2f}, max {y_train.max():.0f}; "
          f"test headcount: mean {y_test.mean():.2f}, max {y_test.max():.0f}")

    best_depth, depth_table = select_tree_depth(X_train, y_train)
    print("\nDecision Tree depth selection (5-fold time-series CV, training data only):")
    print(depth_table.round(3).to_string(index=False))
    print(f"Selected max_depth = {best_depth}")

    makers = {
        "Mean baseline": lambda: DummyRegressor(strategy="mean"),
        "Linear Regression": lambda: LinearRegression(),
        "Decision Tree Regression": lambda: DecisionTreeRegressor(
            max_depth=best_depth, min_samples_leaf=MIN_SAMPLES_LEAF, random_state=RANDOM_STATE
        ),
    }

    results, cv_rows, predictions = [], [], {}
    for name, make in makers.items():
        model = make().fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        m = regression_metrics(y_test, pred)
        m["within_1_person"] = float(np.mean(np.abs(pred - y_test) <= 1))
        folds = cv_mae(make, X_train, y_train)
        m["mean_cv_mae"], m["std_cv_mae"] = np.mean(folds), np.std(folds)
        m["model"] = name
        results.append(m)
        for fold, score in enumerate(folds):
            cv_rows.append({"model": name, "fold": fold, "mae": score})

    results_df = pd.DataFrame(results).set_index("model")
    print("\nTest period and cross-validation results:")
    print(results_df.round(3).to_string())

    REPORTS_DIR.mkdir(exist_ok=True)
    depth_table.to_csv(REPORTS_DIR / "regression_depth_selection.csv", index=False)
    results_df.to_csv(REPORTS_DIR / "regression_results.csv")
    pd.DataFrame(cv_rows).to_csv(REPORTS_DIR / "regression_cv.csv", index=False)
    pd.DataFrame({
        "timestamp": test["timestamp"].to_numpy(),
        "actual": y_test,
        "decision_tree": predictions["Decision Tree Regression"],
        "linear_regression": predictions["Linear Regression"],
    }).to_csv(REPORTS_DIR / "regression_test_predictions.csv", index=False)
    print(f"\nSaved regression outputs to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
