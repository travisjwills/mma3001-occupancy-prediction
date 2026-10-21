"""
Diagnose why cross-validation performance varies so much between folds.

``train_model.py`` reports one F1 score per TimeSeriesSplit fold, and one fold
(fold 3) is much worse than the others for both the baseline and the Random
Forest. This script reproduces the same chronological train/test split and the
same five folds, then reports, for each fold, the calendar dates it covers
(Melbourne time) and the share of intervals in which the room was occupied. It
also writes the occupied share for every calendar day, so an unusual week can
be spotted directly.

Outputs:
    reports/cv_fold_dates.csv     one row per fold: dates, size, occupied share
    reports/daily_occupancy.csv   one row per Melbourne calendar day
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "room_dataset.csv"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
LOCAL_TZ = "Australia/Melbourne"
TRAIN_FRACTION = 0.8  # must match train_model.py
N_SPLITS = 5  # must match train_model.py


def fold_summary(train: pd.DataFrame, n_splits: int = N_SPLITS) -> pd.DataFrame:
    """Summarise the validation block of each TimeSeriesSplit fold.

    Args:
        train: Training portion of the dataset, sorted by ``timestamp``, with
            columns ``timestamp`` (tz-aware) and ``occupied`` (0/1).
        n_splits: Number of TimeSeriesSplit folds.

    Returns:
        DataFrame with one row per fold and columns ``fold``, ``n_train``,
        ``n_validation``, ``validation_start``, ``validation_end`` (Melbourne
        local time) and ``occupied_share`` (fraction of validation intervals
        with the room occupied).
    """
    rows = []
    for fold, (train_idx, val_idx) in enumerate(TimeSeriesSplit(n_splits).split(train)):
        val = train.iloc[val_idx]
        local = val["timestamp"].dt.tz_convert(LOCAL_TZ)
        rows.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_validation": len(val_idx),
            "validation_start": local.min(),
            "validation_end": local.max(),
            "occupied_share": val["occupied"].mean(),
        })
    return pd.DataFrame(rows)


def daily_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    """Occupied share and interval count per Melbourne calendar day.

    Args:
        df: Dataset with ``timestamp`` (tz-aware) and ``occupied`` (0/1).

    Returns:
        DataFrame indexed by local date with columns ``occupied_share`` and
        ``n_intervals``.
    """
    local_date = df["timestamp"].dt.tz_convert(LOCAL_TZ).dt.date.rename("date")
    out = df.groupby(local_date)["occupied"].agg(occupied_share="mean", n_intervals="count")
    return out


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp")
    df = df.reset_index(drop=True)
    train = df.iloc[: int(len(df) * TRAIN_FRACTION)]

    folds = fold_summary(train)
    daily = daily_occupancy(df)

    REPORTS_DIR.mkdir(exist_ok=True)
    folds.to_csv(REPORTS_DIR / "cv_fold_dates.csv", index=False)
    daily.to_csv(REPORTS_DIR / "daily_occupancy.csv")

    print("Cross-validation folds (validation block, Melbourne time):")
    print(folds.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {REPORTS_DIR / 'cv_fold_dates.csv'}")
    print(f"Wrote {REPORTS_DIR / 'daily_occupancy.csv'}")


if __name__ == "__main__":
    main()
