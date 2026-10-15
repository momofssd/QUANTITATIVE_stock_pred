import json

import numpy as np
import pandas as pd
import pytest

from src.machine_learning.five_day_forecast import generate_five_day_forecast, train_five_day_models
from src.machine_learning.validation import calibrated_error_radius


@pytest.fixture
def history():
    index = pd.date_range("2020-01-01", periods=420, freq="B")
    step = np.arange(len(index))
    prices = 100 + step * 0.03 + 5 * np.sin(step / 8)
    features = pd.DataFrame({"Adj Close": prices, "close": prices}, index=index)
    X = pd.DataFrame({
        "momentum": np.cos(step / 8), "cycle": np.sin(step / 8),
        "trend": step / len(step), "other": np.cos(step / 3), "constant": 1,
    }, index=index)
    return X, features


def test_final_test_metrics_match_frozen_predictions_and_targets_do_not_cross_gaps(history):
    X, features = history
    training = train_five_day_models(X, features)
    val, ledger = training.validation, training.validation_predictions
    assert val["train_last_target_date"] < val["calibration_start"]
    assert val["calibration_last_target_date"] < val["test_start"]
    assert val["test_last_target_date"] <= str(features.index[-1].date())
    assert val["test_used_for_training_or_calibration"] is False
    assert training.live_fit["uses_former_test_period"] is True
    assert val["direction_accuracy"] == pytest.approx((ledger.actual_gain == ledger.predicted_gain).mean())
    assert val["return_mean_absolute_error"] == pytest.approx((ledger.actual_return - ledger.predicted_return).abs().mean())
    assert val["brier_score"] == pytest.approx(((ledger.actual_gain - ledger.probability_of_gain) ** 2).mean())
    covered = (ledger.actual_return >= ledger.range_low_return) & (ledger.actual_return <= ledger.range_high_return)
    assert val["interval_observed_coverage"] == pytest.approx(covered.mean())
    assert val["non_overlapping_purchase_check"]["observations"] == len(ledger.iloc[::5])
    assert sum(item["count"] for item in val["probability_reliability_bins"]) == len(ledger)
    json.dumps(val, allow_nan=False)


def test_changing_final_test_outcomes_cannot_change_evaluation_predictions(history):
    X, features = history
    original = train_five_day_models(X, features)
    changed = features.copy()
    start = original.validation["test_start"]
    changed.loc[start:, "Adj Close"] *= np.linspace(1.1, 1.8, len(changed.loc[start:]))
    rerun = train_five_day_models(X, changed)
    columns = ["predicted_return", "probability_of_gain", "range_low_return", "range_high_return"]
    pd.testing.assert_frame_equal(original.validation_predictions[columns], rerun.validation_predictions[columns])
    assert original.validation["interval_calibration_radius_return"] == rerun.validation["interval_calibration_radius_return"]
    assert original.validation["price_distribution"]["mean_predicted_bin_probabilities"] == rerun.validation["price_distribution"]["mean_predicted_bin_probabilities"]
    assert original.validation["return_mean_absolute_error"] != rerun.validation["return_mean_absolute_error"]


def test_asof_forecast_excludes_future_rows(history):
    X, features = history
    asof = features.index[320]
    expected, _ = generate_five_day_forecast(X.loc[:asof], features.loc[:asof], asof)
    changed = features.copy()
    changed.loc[features.index[321]:, "Adj Close"] *= 10
    actual, _ = generate_five_day_forecast(X, changed, asof)
    assert actual == expected


def test_flat_outcome_is_not_a_gain_and_calibration_fallback_is_disclosed(history):
    X, features = history
    features[:] = 100.0
    forecast, training = generate_five_day_forecast(X, features, features.index[-1])
    assert training.validation_predictions.actual_gain.sum() == 0
    assert forecast["confidence_basis"] == "insufficient_class_examples"
    assert forecast["confidence_label"] == "LOW"
    assert forecast["purchase_evidence"]["supports_purchase"] is False
    assert forecast["validation"]["return_r2"] is None
    assert forecast["warnings"]
    json.dumps(forecast, allow_nan=False)


@pytest.mark.parametrize("horizon", [0, -1, 1.5])
def test_invalid_horizon_rejected(history, horizon):
    X, features = history
    with pytest.raises(ValueError, match="positive integer"):
        train_five_day_models(X, features, horizon_days=horizon)


def test_invalid_index_and_split_rejected(history):
    X, features = history
    with pytest.raises(ValueError, match="chronological"):
        train_five_day_models(X.iloc[::-1], features.iloc[::-1])
    with pytest.raises(ValueError, match="fractions"):
        train_five_day_models(X, features, calibration_fraction=0.8)


def test_interval_quantile_uses_finite_sample_rank():
    assert calibrated_error_radius(np.arange(1, 11), coverage=0.8) == 9
