import numpy as np
import pandas as pd

from src.decision_engine import ALLOWED_SIGNALS, SIGNAL_DEFINITIONS, calculate_decision


def _feature_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=260, freq="D")
    close = pd.Series(np.linspace(100, 140, 260), index=index)
    return pd.DataFrame(
        {
            "close": close,
            "ma50": close.rolling(50).mean(),
            "ma200": close.rolling(200).mean(),
            "close_above_200ma": 1,
            "ma50_above_ma200": 1,
            "ma200_slope": 0.0002,
            "close_to_50ma_pct": 0.02,
            "close_to_200ma_pct": 0.10,
            "rs_20d_change": 0.03,
            "vix_20d_change": -0.05,
            "atr_14": 3.0,
            "atr_pct": 0.025,
            "rsi_14": 58,
            "prior_120d_low": close.rolling(120, min_periods=1).min(),
            "prior_120d_high": close.rolling(120, min_periods=1).max(),
            "prior_252d_high": close.rolling(252, min_periods=1).max(),
        },
        index=index,
    ).dropna()


def test_decision_engine_zone_and_signal_invariants():
    result = calculate_decision(_feature_frame(), "BULL_TREND", {"bullish": 0.65, "sideways": 0.25, "bearish": 0.10})
    buy = result.price_zones["preferred_long_term_buy_zone"]
    short_buy = result.price_zones["short_term_tactical_buy_zone"]
    trim = result.price_zones["trim_or_sell_zone"]
    near_term = result.near_term_forecast
    invalidation = result.price_zones["invalidation_zone"]["price"]

    assert result.decision["final_signal"] in ALLOWED_SIGNALS
    assert 0 <= result.decision["confidence_score"] <= 1
    assert buy["low"] < buy["high"]
    assert short_buy["low"] < short_buy["high"]
    assert trim["low"] < trim["high"]
    assert "bullish_scenario" in near_term
    assert "base_case_scenario" in near_term
    assert "bearish_scenario" in near_term
    assert invalidation < _feature_frame()["close"].iloc[-1]
    assert ALLOWED_SIGNALS == set(SIGNAL_DEFINITIONS)
