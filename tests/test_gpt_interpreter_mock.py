from src import gpt_interpreter


def test_gemini_interpreter_mock(monkeypatch):
    expected = {
        "ticker": "TSLA",
        "short_term_view": "APPROACHING_ENTRY",
        "long_term_view": "HOLD_WAIT",
        "plain_english_summary": "Price is about one ATR above the short-term buy zone.",
        "short_term_plan": {
            "entry_zone": "180.10-184.60",
            "current_position": "NEAR, 1.9% above the zone",
            "bounce_assessment": "No washout yet; RSI(14) is 44.",
            "stop_loss": "177.80",
            "targets": "191.40, then 198.20",
            "historical_evidence": "58% of 61 touches bounced versus a 49% any-day baseline.",
            "what_confirms_bounce": ["Close above the prior session high after touching the zone"],
            "what_invalidates_setup": ["Close below the stop"],
        },
        "long_term_plan": {
            "buy_zone": "150.00-171.00",
            "trim_zone": "240.00-262.00",
            "current_position": "Between zones",
            "historical_evidence": "Value-zone visits were higher 60 sessions later 64% of the time.",
        },
        "key_reasons": ["Buy zone has beaten the baseline"],
        "main_risks": ["Price is below its 200-day average"],
        "data_cited": ["RSI(14) 44.0", "ATR 4.52"],
        "not_financial_advice": True,
    }

    monkeypatch.setattr(gpt_interpreter, "interpret_signal_with_gemini", lambda signal_json, model=None: expected)
    assert gpt_interpreter.interpret_signal_with_gemini({"ticker": "TSLA"}) == expected


def test_interpreter_prompt_omits_chart_rows():
    report = {"ticker": "TSLA", "chart_data": {"points": [{"close": 1}]}, "short_term": {}}
    assert gpt_interpreter._without_chart(report) == {"ticker": "TSLA", "short_term": {}}
