"""Tests for the headcount regression helpers in src/headcount_regression.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from headcount_regression import regression_metrics, select_tree_depth  # noqa: E402


def test_regression_metrics_known_values():
    """Perfect predictions score MAE 0 and R^2 1; predicting the mean scores R^2 0."""
    y = [0.0, 1.0, 2.0]
    perfect = regression_metrics(y, y)
    assert perfect["mae"] == 0 and perfect["rmse"] == 0 and perfect["r2"] == 1

    flat = regression_metrics(y, [1.0, 1.0, 1.0])  # always predict the mean (1.0)
    assert flat["mae"] == pytest.approx(2 / 3)
    assert flat["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert flat["r2"] == pytest.approx(0.0)


def test_select_tree_depth_picks_a_candidate_and_reports_every_depth():
    """On a simple step pattern, a shallow tree is enough, and every candidate is scored."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 300)
    y = np.where(x > 0.5, 5.0, 0.0) + rng.normal(0, 0.1, 300)  # one split explains it
    X = pd.DataFrame({"x": x})
    depths = [1, 3, None]

    best, table = select_tree_depth(X, y, depths=depths, n_splits=3)

    assert best in depths
    assert list(table["max_depth"]) == [1, 3, "None"]
    assert table["mean_cv_mae"].notna().all()
    # A single split captures the step, so depth 1 should be (close to) the best.
    assert table.loc[0, "mean_cv_mae"] <= table["mean_cv_mae"].min() + 0.05
