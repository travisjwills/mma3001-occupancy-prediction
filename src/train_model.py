"""
Train and validate occupancy-classification models on the single-room
dataset (data/processed/room_dataset.csv).

Task: predict `occupied` (1 = headcount > 0) from environmental readings.

Models compared (per brief requirement #4, "Alternative solutions" — at
least one baseline vs an improved approach):
  1. Baseline: a simple CO2-threshold rule (the classic engineering
     heuristic for CO2-based occupancy detection). Threshold is chosen on
     the TRAIN split only, by scanning candidate values and picking the one
     that maximises train-set F1 — i.e. the threshold itself is "fit", but
     using only a single feature and a single parameter, unlike the
     multivariate ML model below.
  2. Improved: a Random Forest classifier using all environmental
     variables + time-of-day/day-of-week features.

Validation (brief requirement #5):
  - Chronological train/test split (80/20) — NOT a random split, because
    adjacent 15-min samples are highly autocorrelated; a random split would
    leak near-duplicate information between train and test and overstate
    accuracy. This also mimics the realistic deployment scenario: train on
    past data, predict on future data.
  - TimeSeriesSplit cross-validation (5 folds) on the training portion only,
    to check how stable each model's performance is over time, not just a
    single lucky split.
  - Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.

Outputs: reports/model_comparison.csv, reports/confusion_matrices.txt,
reports/feature_importance.csv, and printed console summary.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "room_dataset.csv"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

FEATURES = [
    "Carbon dioxide",
    "Temperature",
    "Humidity",
    "Formaldehyde",
    "Light Volatile Organic Compounds",
    "Pressure",
    "Particulate matter 1",
    "Particulate matter 2.5",
    "Particulate matter 4",
    "Particulate matter 10",
    "hour",
    "dayofweek",
    "is_weekend",
]
TARGET = "occupied"


def best_co2_threshold(co2: np.ndarray, y: np.ndarray) -> float:
    """Scan candidate CO2 thresholds and return the one maximising F1 on
    the given (training) data. This is the full "fitting" procedure for the
    baseline model — a single scalar parameter, versus hundreds of split
    decisions in the Random Forest."""
    candidates = np.percentile(co2, np.arange(5, 100, 2))
    best_t, best_f1 = candidates[0], -1.0
    for t in candidates:
        pred = (co2 > t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def evaluate(y_true, y_pred, y_score=None) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        out["roc_auc"] = np.nan
    return out


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df)} rows, occupied rate = {df[TARGET].mean():.3f}")

    split_idx = int(len(df) * 0.8)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]
    print(f"Train: {len(train)} rows ({train['timestamp'].min()} -> {train['timestamp'].max()})")
    print(f"Test:  {len(test)} rows ({test['timestamp'].min()} -> {test['timestamp'].max()})")

    X_train, y_train = train[FEATURES], train[TARGET].to_numpy()
    X_test, y_test = test[FEATURES], test[TARGET].to_numpy()

    results = []
    confusion_report = []

    # --- Baseline: CO2 threshold rule ---
    thresh = best_co2_threshold(train["Carbon dioxide"].to_numpy(), y_train)
    base_pred_test = (test["Carbon dioxide"] > thresh).astype(int).to_numpy()
    base_score_test = test["Carbon dioxide"].to_numpy()  # higher CO2 = more "occupied"-like
    base_metrics = evaluate(y_test, base_pred_test, base_score_test)
    base_metrics["model"] = f"Baseline: CO2 > {thresh:.0f} ppm"
    results.append(base_metrics)
    confusion_report.append(
        ("Baseline (CO2 threshold)", confusion_matrix(y_test, base_pred_test))
    )

    # --- Improved: Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred_test = rf.predict(X_test)
    rf_score_test = rf.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate(y_test, rf_pred_test, rf_score_test)
    rf_metrics["model"] = "Random Forest (all features)"
    results.append(rf_metrics)
    confusion_report.append(("Random Forest", confusion_matrix(y_test, rf_pred_test)))

    # --- TimeSeriesSplit CV on training data (stability check) ---
    print("\nTimeSeriesSplit CV (5 folds) on training data:")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_f1_baseline, cv_f1_rf = [], []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        t = best_co2_threshold(train["Carbon dioxide"].to_numpy()[tr_idx], y_tr)
        pred_b = (train["Carbon dioxide"].to_numpy()[val_idx] > t).astype(int)
        f1_b = f1_score(y_val, pred_b, zero_division=0)
        cv_f1_baseline.append(f1_b)

        rf_cv = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1
        )
        rf_cv.fit(X_tr, y_tr)
        pred_rf = rf_cv.predict(X_val)
        f1_rf = f1_score(y_val, pred_rf, zero_division=0)
        cv_f1_rf.append(f1_rf)

        print(f"  fold {fold}: baseline F1={f1_b:.3f}  RF F1={f1_rf:.3f}  (val n={len(val_idx)})")

    print(f"CV mean F1 -- baseline: {np.mean(cv_f1_baseline):.3f} +/- {np.std(cv_f1_baseline):.3f}")
    print(f"CV mean F1 -- RF:       {np.mean(cv_f1_rf):.3f} +/- {np.std(cv_f1_rf):.3f}")

    # --- Save outputs ---
    REPORTS_DIR.mkdir(exist_ok=True)
    results_df = pd.DataFrame(results).set_index("model")
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv")
    print("\nHeld-out test set performance:")
    print(results_df.round(3))

    with open(REPORTS_DIR / "confusion_matrices.txt", "w") as f:
        for name, cm in confusion_report:
            f.write(f"{name}\n{cm}\n\n")

    importance = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    importance.to_csv(REPORTS_DIR / "feature_importance.csv", header=["importance"])
    print("\nRandom Forest feature importances:")
    print(importance.round(3))

    cv_df = pd.DataFrame({"fold": range(5), "baseline_f1": cv_f1_baseline, "rf_f1": cv_f1_rf})
    cv_df.to_csv(REPORTS_DIR / "cv_results.csv", index=False)


if __name__ == "__main__":
    main()
