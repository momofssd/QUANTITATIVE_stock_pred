from tests.test_decision_engine import _feature_frame

from src.short_term_trade import SHORT_TERM_ACTIONS, build_short_term_trade_plan


def _zones() -> dict:
    return {
        "short_term_tactical_buy_zone": {
            "low": 138.0,
            "high": 141.0,
            "confirmation_rules": ["price holds support"],
        }
    }


def test_bullish_forecast_inside_entry_zone_supports_short_term_buy_now():
    forecast = {
        "horizon_trading_days": 5,
        "predicted_direction": "BULLISH",
        "confidence_score": 0.68,
        "direction_probabilities": {"bullish": 0.68, "bearish": 0.32},
        "possible_price_in_5_trading_days": 145.0,
        "possible_price_range": {"low": 142.0, "high": 148.0},
        "direction_price_models_agree": True,
    }

    plan = build_short_term_trade_plan(_feature_frame(), "BULL_TREND", forecast, _zones())

    assert plan["action"] == "BUY_NOW_AT_SUPPORT"
    assert plan["is_good_short_term_buy_now"] is True
    assert plan["entry_points"]["pullback_entry"]["status"] == "CURRENT_PRICE_IN_ZONE"
    assert plan["trade_management"]["stop_loss_price"] < plan["entry_points"]["pullback_entry"]["low"]
    assert plan["trade_management"]["estimated_reward_to_risk"] > 0
    assert plan["action"] in SHORT_TERM_ACTIONS


def test_bearish_five_session_forecast_does_not_support_short_term_buy():
    forecast = {
        "horizon_trading_days": 5,
        "predicted_direction": "BEARISH",
        "confidence_score": 0.62,
        "direction_probabilities": {"bullish": 0.38, "bearish": 0.62},
        "possible_price_in_5_trading_days": 135.0,
        "possible_price_range": {"low": 132.0, "high": 140.0},
        "direction_price_models_agree": True,
    }

    plan = build_short_term_trade_plan(_feature_frame(), "BULL_TREND", forecast, _zones())

    assert plan["action"] == "AVOID_SHORT_TERM_BUY"
    assert plan["is_good_short_term_buy_now"] is False
    assert plan["has_conditional_buy_setup"] is False
    assert plan["trade_management"]["stop_loss_price"] is None
    assert plan["trade_management"]["estimated_reward_to_risk"] is None
