from src import gpt_interpreter


def test_gemini_interpreter_mock(monkeypatch):
    expected = {
        "ticker": "TSLA",
        "short_term_view": "APPROACHING_ENTRY",
        "long_term_view": "HOLD_WAIT",
        "headline": "TSLA is about one ATR above its short-term buy zone.",
        "summary_markdown": "- Price is **1.9% above** the buy zone.",
        "short_term_markdown": "#### Where price is\nNear the zone.",
        "historical_replay_markdown": "| Setup | Worked |\n|---|---|\n| Buy at first touch | 58% |",
        "trend_outlook_markdown": "- 10 sessions: 61% bullish (base rate).",
        "long_term_markdown": "Between the value and trim zones.",
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


def test_env_file_key_and_model_win_over_process_environment(monkeypatch):
    file_key = "AIza" + "f" * 35
    monkeypatch.setattr(gpt_interpreter, "read_env_file", lambda: {"gemini": file_key, "GEMINI_MODEL": "gemini-4-argon"})
    monkeypatch.setenv("GEMINI_API_KEY", "stale-terminal-key-000000000000000")
    monkeypatch.setenv("GEMINI_MODEL", "some-other-model")
    assert gpt_interpreter._api_key_with_source() == (file_key, ".env gemini")
    assert gpt_interpreter._model_name(None) == "gemini-4-argon"


def test_process_environment_key_used_when_env_file_has_none(monkeypatch):
    monkeypatch.setattr(gpt_interpreter, "read_env_file", lambda: {})
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "e" * 35)
    assert gpt_interpreter._api_key_with_source()[1] == "environment GEMINI_API_KEY"


def test_schema_requires_every_markdown_section():
    required = set(gpt_interpreter.INTERPRETATION_SCHEMA["required"])
    assert set(gpt_interpreter.MARKDOWN_SECTIONS) <= required
    assert set(required) == set(gpt_interpreter.INTERPRETATION_SCHEMA["properties"])
