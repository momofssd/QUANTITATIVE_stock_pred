import json

import numpy as np
import pandas as pd
import pytest

from src.machine_learning.price_distribution import describe_distribution, distribution_validation, return_samples
from src.machine_learning.five_day_forecast import generate_five_day_forecast
from src.data_modeling.price_actions import build_price_action_plan
from tests.test_short_term_validation import history


def test_distribution_includes_tails_and_assigns_boundaries_once():
    residuals = np.array([-1.5, -0.03, -0.01, 0, 0.01, 0.03, 2.0])
    distribution = describe_distribution(100, 0, residuals)
    assert [item["probability"] for item in distribution["bins"]] == pytest.approx(np.array([1, 1, 2, 1, 2]) / 7)
    assert sum(item["probability"] for item in distribution["bins"]) == pytest.approx(1)
    assert distribution["bins"][0]["low_price"] == 0
    assert distribution["bins"][-1]["high_price"] is None
    assert distribution["probability_of_gain"] == pytest.approx(3 / 7)
    assert list(distribution["price_quantiles"].values()) == sorted(distribution["price_quantiles"].values())
    json.dumps(distribution, allow_nan=False)


def test_distribution_score_measures_probabilities_against_heldout_outcomes():
    metrics = distribution_validation(np.array([0.0, 0.02]), np.array([0, -0.05]), np.zeros(5))
    assert metrics["multiclass_brier_score"] == 1.0  # one correct, one wrong deterministic forecast
    assert metrics["test_used_for_distribution_calibration"] is False
    with pytest.raises(ValueError):
        return_samples(0, [])


def test_all_five_sessions_have_separate_targets_ranges_and_validation(history):
    X, features = history
    forecast, training = generate_five_day_forecast(X, features, features.index[-1])
    assert [item["session"] for item in forecast["session_forecasts"]] == [1, 2, 3, 4, 5]
    assert forecast["price_distribution"] == forecast["session_forecasts"][-1]["price_distribution"]
    for item in forecast["session_forecasts"]:
        assert 0 <= item["probability_of_gain"] <= 1
        assert sum(b["probability"] for b in item["price_distribution"]["bins"]) == pytest.approx(1)
        assert item["validation"]["purged_rows"] == item["session"]
        assert item["validation"]["train_last_target_date"] < item["validation"]["calibration_start"]
        assert item["validation"]["calibration_last_target_date"] < item["validation"]["test_start"]
        assert item["calibrated_price_range"]["low"] <= item["expected_close"] <= item["calibrated_price_range"]["high"]
    assert set(training.session_models) == {1, 2, 3, 4}
    json.dumps(forecast, allow_nan=False)


@pytest.mark.parametrize("reference", [0.05, 100.0])
def test_price_action_levels_are_ordered_even_when_forecast_is_bearish(reference):
    features = pd.DataFrame({"close": np.full(25, reference), "atr_5_pct": np.nan})
    forecast = {"purchase_evidence": {"supports_purchase": False}, "price_distribution": {"price_quantiles": {"p90": reference * 0.9}}}
    plan = build_price_action_plan(features, forecast)
    levels = plan["levels"]
    assert 0 < levels["invalidation"] < levels["support"] < levels["bounce_watch_ceiling"] < reference
    assert reference < levels["resistance"] < levels["breakout_confirmation"] < levels["take_profit_1"] < levels["take_profit_2"]
    assert not plan["new_entry_supported_by_validation"]
    assert plan["rules"][0]["action"] == "REDUCE_OR_EXIT"
    assert "cannot confirm a bounce" in plan["rules"][1]["guidance"]
    json.dumps(plan, allow_nan=False)
