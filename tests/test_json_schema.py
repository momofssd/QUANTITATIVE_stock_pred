from tests.test_decision_engine import _feature_frame

from src.decision_engine import calculate_decision
from src.json_exporter import build_signal_json
from src.short_term_trade import SHORT_TERM_ACTIONS, build_short_term_trade_plan


def test_json_output_contains_required_fields():
    features = _feature_frame()
    decision = calculate_decision(features, "BULL_TREND", {"bullish": 0.60, "sideways": 0.25, "bearish": 0.15})
    five_day_forecast = {
        "horizon_trading_days": 5,
        "predicted_direction": "BULLISH",
        "confidence_score": 0.62,
        "direction_probabilities": {"bullish": 0.62, "bearish": 0.38},
        "possible_price_in_5_trading_days": 143.0,
        "possible_price_range": {"low": 139.0, "high": 146.0},
        "direction_price_models_agree": True,
    }
    short_term_trade = build_short_term_trade_plan(
        features,
        "BULL_TREND",
        five_day_forecast,
        decision.price_zones,
    )
    signal = build_signal_json(
        ticker="SMH",
        as_of_date="2026-05-19",
        benchmark_primary="QQQ",
        benchmark_secondary="SPY",
        treasury_yield_ticker="^TNX",
        oil_ticker="CL=F",
        features=features,
        regime="BULL_TREND",
        probabilities={"bullish": 0.60, "sideways": 0.25, "bearish": 0.15},
        decision_result=decision,
        model_metadata={"model_type": "RandomForestClassifier", "feature_count": 38},
        five_day_forecast=five_day_forecast,
        short_term_trade=short_term_trade,
    )

    for key in [
        "schema_version",
        "ticker",
        "current_market",
        "decision",
        "available_signals",
        "price_zones",
        "five_day_forecast",
        "short_term_trade",
        "near_term_forecast",
        "chart_data",
        "technical_indicators",
        "relative_strength",
        "risk_management",
        "model_metadata",
    ]:
        assert key in signal
    assert signal["schema_version"] == "2.2"
    assert signal["decision"]["confidence_label"] in {"LOW", "MEDIUM", "HIGH", "VERY_HIGH"}
    assert signal["decision"]["final_signal"] in signal["available_signals"]
    assert signal["available_signals"]["ACCUMULATE_ON_PULLBACK"].startswith("Trend is constructive")
    assert signal["chart_data"]["price_basis"] == "adjusted_ohlc"
    assert signal["chart_data"]["points"][-1]["close"] == signal["current_market"]["current_price"]
    assert signal["short_term_trade"]["action"] in SHORT_TERM_ACTIONS
    assert isinstance(signal["short_term_trade"]["is_good_short_term_buy_now"], bool)
