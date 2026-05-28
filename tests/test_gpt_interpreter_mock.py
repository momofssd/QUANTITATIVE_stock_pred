from src import gpt_interpreter


def test_gemini_interpreter_mock(monkeypatch):
    expected = {
        "ticker": "TSLA",
        "final_view": "WAIT_FOR_PULLBACK",
        "is_current_price_long_term_buy": False,
        "is_current_price_long_term_sell": False,
        "preferred_buy_zone": "165-174",
        "short_term_tactical_buy_zone": "180-185",
        "breakout_buy_trigger": "195",
        "trim_or_sell_zone": "215-230",
        "invalidation_level": "154",
        "confidence_score": 0.68,
        "confidence_label": "HIGH",
        "plain_english_summary": "Wait for a better entry.",
        "near_term_price_forecast": {
            "selected_scenario": "base_case_scenario",
            "estimated_timeframe": "4-8 weeks",
            "target_price_zone": "185-195",
            "downside_or_failure_zone": "170-175",
            "confidence_label": "MEDIUM",
            "probability_comment": "Base case is most consistent with current conditions.",
            "conditions_needed": ["Hold above short-term support"],
            "what_would_invalidate_forecast": ["Close below tactical zone"],
        },
        "technical_data_used": {
            "trend": ["Price is above the 200-day average."],
            "momentum": ["RSI is neutral."],
            "volatility": ["ATR risk is manageable."],
            "volume": ["Volume is near average."],
            "relative_strength": ["Relative strength is improving."],
            "macro": ["VIX is not spiking."],
        },
        "technical_assessment": {
            "trend_health": "Constructive",
            "momentum_condition": "Neutral",
            "volatility_condition": "Normal",
            "relative_strength_condition": "Improving",
            "risk_reward_condition": "Better on pullback",
        },
        "key_reasons": ["Trend is constructive"],
        "main_risks": ["Market regime could weaken"],
        "not_financial_advice": True,
    }

    monkeypatch.setattr(gpt_interpreter, "interpret_signal_with_gemini", lambda signal_json, model=None: expected)
    assert gpt_interpreter.interpret_signal_with_gemini({"ticker": "TSLA"}) == expected
