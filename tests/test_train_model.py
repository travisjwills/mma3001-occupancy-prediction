"""Tests for the baseline-model and evaluation helpers in src/train_model.py."""
import numpy as np

from train_model import best_co2_threshold, evaluate


def test_best_co2_threshold_recovers_known_separating_value():
    rng = np.random.default_rng(0)
    # Occupied readings cluster high, unoccupied readings cluster low, with
    # a clean gap around 600 ppm.
    co2_occupied = rng.normal(800, 20, size=200)
    co2_empty = rng.normal(450, 20, size=200)
    co2 = np.concatenate([co2_occupied, co2_empty])
    y = np.concatenate([np.ones(200), np.zeros(200)])

    threshold = best_co2_threshold(co2, y)

    # The chosen threshold should fall somewhere in the gap between the two
    # cluster means (450 and 800), clearly past the empty cluster's mean.
    assert 460 < threshold < 750
    pred = (co2 > threshold).astype(int)
    accuracy = (pred == y).mean()
    assert accuracy > 0.95


def test_evaluate_returns_expected_metrics_for_perfect_prediction():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])

    metrics = evaluate(y_true, y_pred, y_score)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_evaluate_handles_single_class_without_crashing():
    """When a fold happens to contain only one class, ROC-AUC is undefined;
    evaluate() must report NaN rather than raising, since this situation is
    expected to occur occasionally with the small time-series CV folds used
    in this project (see docs/model_results.md)."""
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.3])

    metrics = evaluate(y_true, y_pred, y_score)

    assert np.isnan(metrics["roc_auc"])
    assert metrics["accuracy"] == 1.0
