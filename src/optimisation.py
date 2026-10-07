"""
Optimisation & performance analysis (brief requirement #6).

Extends the baseline-vs-RandomForest comparison in train_model.py with:
  1. Two more credible alternatives (Logistic Regression, Gradient Boosting)
     at default settings, so the "alternative solutions" comparison isn't
     just one ML model vs one heuristic.
  2. Hyperparameter tuning of the Random Forest via RandomizedSearchCV with
     TimeSeriesSplit (not plain k-fold — see train_model.py for why),
     measuring whether tuning is actually worth its extra compute cost.
  3. Runtime profiling: fit time and per-prediction inference time for every
     model (averaged over repeated runs).
  4. Memory profiling: peak memory during training (tracemalloc) and
     on-disk model size (pickle).
  5. A rough arithmetic-cost estimate for the Random Forest, since its cost
     isn't a simple closed-form FLOP count the way a linear model's is.

Outputs: reports/optimisation_comparison.csv, reports/hyperparam_tuning.csv
"""
from __future__ import annotations

import pickle
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "room_dataset.csv"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

FEATURES = [
    "Carbon dioxide", "Temperature", "Humidity", "Formaldehyde",
    "Light Volatile Organic Compounds", "Pressure",
    "Particulate matter 1", "Particulate matter 2.5",
    "Particulate matter 4", "Particulate matter 10",
    "hour", "dayofweek", "is_weekend",
]
TARGET = "occupied"
N_TIMING_REPEATS = 5
N_INFERENCE_REPEATS = 50  # predict() called this many times to get a stable per-call estimate


def time_fit(model, X_train, y_train):
    times = []
    for _ in range(N_TIMING_REPEATS):
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        times.append(time.perf_counter() - t0)
    return min(times)  # best-of, removes OS scheduling noise


def time_predict(model, X_test):
    times = []
    for _ in range(N_INFERENCE_REPEATS):
        t0 = time.perf_counter()
        model.predict(X_test)
        times.append(time.perf_counter() - t0)
    per_call = min(times) / len(X_test) * 1000  # ms per single-row prediction
    return per_call


def peak_memory_mb(fit_fn):
    tracemalloc.start()
    fit_fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 ** 2)


def model_size_kb(model) -> float:
    return len(pickle.dumps(model)) / 1024


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp")
    split_idx = int(len(df) * 0.8)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]
    X_train, y_train = train[FEATURES], train[TARGET].to_numpy()
    X_test, y_test = test[FEATURES], test[TARGET].to_numpy()

    candidates = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
        ),
        "Random Forest (default)": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    rows = []
    for name, model in candidates.items():
        fit_time = time_fit(model, X_train, y_train)
        model.fit(X_train, y_train)  # ensure fitted state for predict timing/metrics
        predict_ms = time_predict(model, X_test)
        pred = model.predict(X_test)
        score = model.predict_proba(X_test)[:, 1]
        f1 = f1_score(y_test, pred, zero_division=0)
        auc = roc_auc_score(y_test, score)
        peak_mb = peak_memory_mb(lambda m=candidates[name]: m.fit(X_train, y_train))
        size_kb = model_size_kb(model)

        rows.append({
            "model": name,
            "f1": f1,
            "roc_auc": auc,
            "fit_time_s": fit_time,
            "predict_time_ms_per_row": predict_ms,
            "peak_train_memory_mb": peak_mb,
            "model_size_kb": size_kb,
        })
        print(f"{name:28s} F1={f1:.3f} AUC={auc:.3f} fit={fit_time*1000:.1f}ms "
              f"predict={predict_ms*1000:.4f}us/row mem={peak_mb:.2f}MB size={size_kb:.1f}KB")

    comp_df = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    comp_df.to_csv(REPORTS_DIR / "optimisation_comparison.csv", index=False)

    # --- Hyperparameter tuning of the Random Forest ---
    print("\nHyperparameter tuning Random Forest (RandomizedSearchCV, TimeSeriesSplit) ...")
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [4, 6, 8, 12, None],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", None],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=20,
        scoring="f1",
        cv=TimeSeriesSplit(n_splits=5),
        random_state=42,
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    search.fit(X_train, y_train)
    tuning_time = time.perf_counter() - t0

    tuned_pred = search.best_estimator_.predict(X_test)
    tuned_score = search.best_estimator_.predict_proba(X_test)[:, 1]
    tuned_f1 = f1_score(y_test, tuned_pred, zero_division=0)
    tuned_auc = roc_auc_score(y_test, tuned_score)

    default_row = comp_df[comp_df["model"] == "Random Forest (default)"].iloc[0]
    print(f"\nDefault RF:  F1={default_row['f1']:.3f}  AUC={default_row['roc_auc']:.3f}")
    print(f"Tuned RF:    F1={tuned_f1:.3f}  AUC={tuned_auc:.3f}  "
          f"(best params: {search.best_params_})")
    print(f"Search cost: {tuning_time:.1f}s for 20 candidates x 5 folds = 100 fits")

    tuning_df = pd.DataFrame([
        {"model": "Random Forest (default)", "f1": default_row["f1"], "roc_auc": default_row["roc_auc"],
         "search_cost_s": 0.0, "params": "n_estimators=300, max_depth=8, min_samples_leaf=5"},
        {"model": "Random Forest (tuned)", "f1": tuned_f1, "roc_auc": tuned_auc,
         "search_cost_s": tuning_time, "params": str(search.best_params_)},
    ])
    tuning_df.to_csv(REPORTS_DIR / "hyperparam_tuning.csv", index=False)

    # --- Rough arithmetic-cost estimate for the Random Forest ---
    n_trees = 300
    # Each tree traversal for one prediction touches ~depth comparisons;
    # training touches, per tree, roughly n_samples * n_features * depth
    # comparisons to find splits (a loose upper bound, ignoring sorting costs).
    depth = 8
    n_samples, n_features = X_train.shape
    train_ops_estimate = n_trees * n_samples * n_features * depth
    predict_ops_estimate = n_trees * depth
    print(f"\nRough RF cost estimate: ~{train_ops_estimate:,.0f} comparisons to train, "
          f"~{predict_ops_estimate:,.0f} comparisons per single-row prediction "
          f"({n_trees} trees x depth {depth}).")
    print(f"Saved: {REPORTS_DIR / 'optimisation_comparison.csv'}")
    print(f"Saved: {REPORTS_DIR / 'hyperparam_tuning.csv'}")


if __name__ == "__main__":
    main()
