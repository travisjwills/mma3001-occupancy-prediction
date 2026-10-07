"""
Sensitivity check: does the Random Forest actually need the environmental
variables, or is it mostly riding on time-of-day?

Refits the same model with `hour`/`dayofweek`/`is_weekend` removed, using the
identical chronological split as train_model.py, and compares F1/ROC-AUC.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "room_dataset.csv"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

ENV_ONLY_FEATURES = [
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
]
ALL_FEATURES = ENV_ONLY_FEATURES + ["hour", "dayofweek", "is_weekend"]
TARGET = "occupied"


def fit_eval(features, train, test):
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf.fit(train[features], train[TARGET])
    pred = rf.predict(test[features])
    score = rf.predict_proba(test[features])[:, 1]
    return {
        "f1": f1_score(test[TARGET], pred, zero_division=0),
        "roc_auc": roc_auc_score(test[TARGET], score),
    }


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp")
    split_idx = int(len(df) * 0.8)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    full = fit_eval(ALL_FEATURES, train, test)
    env_only = fit_eval(ENV_ONLY_FEATURES, train, test)

    print("Full model (env + time features):", full)
    print("Env-only model (no hour/day):     ", env_only)

    out = pd.DataFrame([
        {"model": "Full (env + time)", **full},
        {"model": "Env-only (no time features)", **env_only},
    ])
    out.to_csv(REPORTS_DIR / "sensitivity_env_only.csv", index=False)
    print(f"\nSaved reports/sensitivity_env_only.csv")


if __name__ == "__main__":
    main()
