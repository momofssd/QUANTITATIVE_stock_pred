"""Chronological calibration and held-out metrics for short-term forecasts."""

from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, mean_absolute_error, mean_squared_error, r2_score

from ..utils import clean_float


@dataclass
class CalibratedDirectionModel:
    """A fitted base classifier with a sigmoid fitted on a later, disjoint period."""

    estimator: object
    calibrator: object | None
    fallback_probability: float
    calibration_status: str

    @property
    def classes_(self):
        return np.array([0, 1])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.calibrator is None:
            gain = np.full(len(X), self.fallback_probability)
        else:
            gain = self.calibrator.predict_proba(_probability_logit(self.estimator, X))[:, 1]
        return np.column_stack([1 - gain, gain])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _probability_logit(estimator, X: pd.DataFrame) -> np.ndarray:
    """Align the positive class and transform its clipped probability to log odds."""
    classes = list(estimator.classes_)
    probabilities = estimator.predict_proba(X)
    positive = probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
    clipped = np.clip(positive, 1e-5, 1 - 1e-5)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def calibrate_direction_model(estimator, X: pd.DataFrame, y: pd.Series) -> CalibratedDirectionModel:
    """Calibrate on untouched later rows; fall back to a smoothed prior if unusable."""
    prior = float((y.sum() + 1) / (len(y) + 2))
    if len(estimator.classes_) < 2 or y.nunique() < 2 or y.value_counts().min() < 5:
        return CalibratedDirectionModel(estimator, None, prior, "insufficient_class_examples")
    calibrator = LogisticRegression(C=1.0, random_state=42).fit(_probability_logit(estimator, X), y)
    return CalibratedDirectionModel(estimator, calibrator, prior, "sigmoid_calibrated")


def chronological_partitions(rows: int, horizon_days: int, test_fraction: float, calibration_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/calibration/test positions with a horizon gap at both boundaries."""
    if not 0 < test_fraction < 1 or not 0 < calibration_fraction < 1 or test_fraction + calibration_fraction >= 1:
        raise ValueError("Test and calibration fractions must be positive and sum to less than one.")
    test_start = int(rows * (1 - test_fraction))
    calibration_start = int(rows * (1 - test_fraction - calibration_fraction))
    train = np.arange(max(calibration_start - horizon_days, 0))
    calibration = np.arange(calibration_start, max(test_start - horizon_days, calibration_start))
    test = np.arange(test_start, rows)
    if len(train) < 60 or len(calibration) < 20 or len(test) < 20:
        raise ValueError("Need at least 60 training, 20 calibration, and 20 test rows after horizon purging. Try an earlier start date.")
    return train, calibration, test


def calibrated_error_radius(residuals: np.ndarray, coverage: float = 0.8) -> float:
    """Finite-sample absolute-error quantile; time-series coverage is measured, not guaranteed."""
    errors = np.sort(np.abs(np.asarray(residuals)))
    if not 0 < coverage < 1 or not len(errors) or not np.isfinite(errors).all():
        raise ValueError("Interval calibration requires finite errors and coverage between zero and one.")
    rank = min(ceil((len(errors) + 1) * coverage), len(errors))
    return float(errors[rank - 1])


def heldout_metrics(y: pd.Series, returns: pd.Series, probability: np.ndarray, prediction: np.ndarray, train_returns: pd.Series, radius: float, horizon_days: int) -> tuple[dict, pd.DataFrame]:
    """Score fixed predictions on the final test period, including simple baselines.

    A separate non-overlapping sample describes historical purchase outcomes.
    Daily test accuracy still uses overlapping horizons and is labeled as such.
    """
    predicted_class = (probability >= 0.5).astype(int)
    base_rate = float((train_returns > 0).mean())
    baseline_brier = float(brier_score_loss(y, np.full(len(y), base_rate)))
    brier = float(brier_score_loss(y, probability))
    low, high = np.maximum(prediction - radius, -1), prediction + radius
    ledger = pd.DataFrame({
        "actual_return": returns,
        "predicted_return": prediction,
        "probability_of_gain": probability,
        "actual_gain": y,
        "predicted_gain": predicted_class,
        "range_low_return": low,
        "range_high_return": high,
    }, index=y.index)
    independent = ledger.iloc[::horizon_days]
    entries = independent[(independent.probability_of_gain >= 0.55) & (independent.predicted_return > 0)]
    bins = []
    for left, right in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:]):
        mask = (probability >= left) & ((probability < right) if right < 1 else (probability <= right))
        bins.append({
            "low": float(left), "high": float(right), "count": int(mask.sum()),
            "mean_predicted_probability": clean_float(probability[mask].mean()) if mask.any() else None,
            "observed_gain_rate": clean_float(y.iloc[np.flatnonzero(mask)].mean()) if mask.any() else None,
        })
    metrics = {
        "direction_accuracy": float(accuracy_score(y, predicted_class)),
        "direction_macro_f1": float(f1_score(y, predicted_class, labels=[0, 1], average="macro", zero_division=0)),
        "gain_precision": float(y.iloc[np.flatnonzero(predicted_class == 1)].mean()) if predicted_class.any() else None,
        "predicted_gain_count": int(predicted_class.sum()),
        "observed_gain_rate": float(y.mean()),
        "brier_score": brier,
        "brier_skill_score": 1 - brier / baseline_brier if baseline_brier > 0 else None,
        "return_mean_absolute_error": float(mean_absolute_error(returns, prediction)),
        "return_root_mean_squared_error": float(np.sqrt(mean_squared_error(returns, prediction))),
        "return_r2": float(r2_score(returns, prediction)) if returns.nunique() > 1 else None,
        "interval_observed_coverage": float(np.mean((returns >= low) & (returns <= high))),
        "test_rows": len(y),
        "test_has_both_classes": y.nunique() == 2,
        "overlapping_horizon_observations": True,
        "probability_reliability_bins": bins,
        "baselines": {
            "always_gain_accuracy": float(y.mean()),
            "training_majority_accuracy": float(accuracy_score(y, np.full(len(y), int(base_rate >= 0.5)))),
            "training_gain_rate": base_rate,
            "constant_probability_brier": baseline_brier,
            "zero_return_mae": float(mean_absolute_error(returns, np.zeros(len(y)))),
            "training_mean_return_mae": float(mean_absolute_error(returns, np.full(len(y), train_returns.mean()))),
        },
        "non_overlapping_purchase_check": {
            "sampling_stride_trading_rows": horizon_days,
            "observations": len(independent),
            "qualifying_entries": len(entries),
            "rule": "probability_of_gain >= 0.55 and predicted_return > 0",
            "win_rate": float(entries.actual_gain.mean()) if len(entries) else None,
            "mean_return": float(entries.actual_return.mean()) if len(entries) else None,
            "all_observations_gain_rate": float(independent.actual_gain.mean()),
            "costs_included": False,
        },
    }
    return metrics, ledger
