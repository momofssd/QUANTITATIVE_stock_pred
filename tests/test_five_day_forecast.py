import numpy as np
import pandas as pd

from src.five_day_forecast import generate_five_day_forecast
from src.labels import create_direction_labels, create_forward_return_target


def test_five_day_targets_use_fifth_future_trading_row():
    index = pd.date_range("2026-01-05", periods=8, freq="B")
    features = pd.DataFrame({"Adj Close": [100, 105, 103, 100, 99, 110, 90, 111]}, index=index)

    labels = create_direction_labels(features)
    returns = create_forward_return_target(features)

    assert labels.iloc[0] == 1
    assert labels.iloc[1] == 0
    assert np.isclose(returns.iloc[0], 0.1)
    assert pd.isna(labels.iloc[-1])


def test_five_day_forecast_contains_direction_confidence_and_price():
    index = pd.date_range("2022-01-03", periods=260, freq="B")
    steps = np.arange(len(index))
    prices = 100 + (steps * 0.03) + (np.sin(steps / 4) * 4)
    features = pd.DataFrame({"Adj Close": prices, "close": prices}, index=index)
    X = pd.DataFrame(
        {
            "return_5_proxy": pd.Series(prices, index=index).pct_change(5),
            "cycle": np.sin(steps / 4),
            "momentum": np.cos(steps / 7),
            "trend": steps / len(steps),
            "constant": 1,
        },
        index=index,
    )

    forecast, training = generate_five_day_forecast(X, features, index[-1])

    assert forecast["horizon_trading_days"] == 5
    assert forecast["predicted_direction"] in {"BULLISH", "BEARISH"}
    assert 0.5 <= forecast["confidence_score"] <= 1
    assert forecast["possible_price_in_5_trading_days"] > 0
    assert forecast["price_estimate_basis"] == "direction_conditioned_forward_return_regression"
    assert forecast["unconditional_expected_price_in_5_trading_days"] > 0
    assert isinstance(forecast["direction_price_models_agree"], bool)
    assert forecast["direction_price_models_agree"] is True
    assert "possible price" in forecast["interpretation_note"].lower() or "uncertain" in forecast["interpretation_note"]
    assert forecast["validation"]["purged_rows"] == 5
    assert training.training_rows > 120
