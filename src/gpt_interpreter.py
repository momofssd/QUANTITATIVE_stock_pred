from __future__ import annotations

import json
import os

from .utils import PROJECT_ROOT, load_env_file


DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"


INTERPRETATION_SCHEMA = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string"},
        "final_view": {"type": "string"},
        "is_current_price_long_term_buy": {"type": "boolean"},
        "is_current_price_long_term_sell": {"type": "boolean"},
        "preferred_buy_zone": {"type": "string"},
        "short_term_tactical_buy_zone": {"type": "string"},
        "breakout_buy_trigger": {"type": "string"},
        "trim_or_sell_zone": {"type": "string"},
        "invalidation_level": {"type": "string"},
        "confidence_score": {"type": "number"},
        "confidence_label": {"type": "string"},
        "plain_english_summary": {"type": "string"},
        "near_term_price_forecast": {
            "type": "object",
            "properties": {
                "selected_scenario": {"type": "string"},
                "estimated_timeframe": {"type": "string"},
                "target_price_zone": {"type": "string"},
                "downside_or_failure_zone": {"type": "string"},
                "confidence_label": {"type": "string"},
                "probability_comment": {"type": "string"},
                "conditions_needed": {"type": "array", "items": {"type": "string"}},
                "what_would_invalidate_forecast": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "selected_scenario",
                "estimated_timeframe",
                "target_price_zone",
                "downside_or_failure_zone",
                "confidence_label",
                "probability_comment",
                "conditions_needed",
                "what_would_invalidate_forecast",
            ],
        },
        "technical_data_used": {
            "type": "object",
            "properties": {
                "trend": {"type": "array", "items": {"type": "string"}},
                "momentum": {"type": "array", "items": {"type": "string"}},
                "volatility": {"type": "array", "items": {"type": "string"}},
                "volume": {"type": "array", "items": {"type": "string"}},
                "relative_strength": {"type": "array", "items": {"type": "string"}},
                "macro": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["trend", "momentum", "volatility", "volume", "relative_strength", "macro"],
        },
        "technical_assessment": {
            "type": "object",
            "properties": {
                "trend_health": {"type": "string"},
                "momentum_condition": {"type": "string"},
                "volatility_condition": {"type": "string"},
                "relative_strength_condition": {"type": "string"},
                "risk_reward_condition": {"type": "string"},
            },
            "required": [
                "trend_health",
                "momentum_condition",
                "volatility_condition",
                "relative_strength_condition",
                "risk_reward_condition",
            ],
        },
        "key_reasons": {"type": "array", "items": {"type": "string"}},
        "main_risks": {"type": "array", "items": {"type": "string"}},
        "not_financial_advice": {"type": "boolean"},
    },
    "required": [
        "ticker",
        "final_view",
        "is_current_price_long_term_buy",
        "is_current_price_long_term_sell",
        "preferred_buy_zone",
        "short_term_tactical_buy_zone",
        "breakout_buy_trigger",
        "trim_or_sell_zone",
        "invalidation_level",
        "confidence_score",
        "confidence_label",
        "plain_english_summary",
        "near_term_price_forecast",
        "technical_data_used",
        "technical_assessment",
        "key_reasons",
        "main_risks",
        "not_financial_advice",
    ],
}


def _api_key() -> str | None:
    load_env_file()
    candidates = [
        os.getenv("GEMINI_API_KEY"),
        os.getenv("gemini"),
        os.getenv("GOOGLE_API_KEY"),
    ]
    for candidate in candidates:
        if candidate and len(candidate.strip()) >= 20:
            return candidate.strip()
    return None


def _skill_instructions() -> str:
    skill_path = PROJECT_ROOT / "skills" / "long_term_trade_interpreter" / "SKILL.md"
    if not skill_path.exists():
        return ""
    return skill_path.read_text(encoding="utf-8")


def interpret_signal_with_gemini(signal_json: dict, model: str | None = None) -> dict:
    """
    Sends quant JSON to Gemini and returns structured JSON.

    The default model uses Gemini 3.1 Pro Preview. Override with GEMINI_MODEL
    or --gemini-model if your account exposes a different model identifier.
    """
    key = _api_key()
    if not key:
        raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY, GOOGLE_API_KEY, or gemini in .env.")
    model_name = model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError("google-genai is required for Gemini interpretation. Install dependencies first.") from exc

    skill_instructions = _skill_instructions()
    prompt = (
        "Use these local skill instructions as the interpretation policy:\n\n"
        f"{skill_instructions}\n\n"
        "Interpret this quantitative long-term swing trading JSON. "
        "Do not merely repeat the headline decision. Analyze the actual technical data in "
        "technical_indicators, relative_strength, macro_context, risk_management, current_market, "
        "price_zones, five_day_forecast, short_term_trade, near_term_forecast, and reasoning. "
        "When five_day_forecast is present, use its predicted_direction, confidence, and possible price as the "
        "explicit five-trading-day model forecast; the possible price is conditioned on the predicted direction, "
        "while any unconditional expected price is supplemental, and do not replace it with an invented level. "
        "When short_term_trade is present, preserve its short-term action, buy-now assessment, supplied entry "
        "points, and active trade-management levels; do not invent a different short-term entry decision. "
        "For near_term_price_forecast, select the most likely scenario from near_term_forecast, determine "
        "how soon it could happen from the supplied horizon_candidates and current technical condition, and "
        "use only the target/pullback zones already provided in near_term_forecast. "
        "In technical_data_used, cite concrete indicator values from the JSON, such as moving averages, "
        "RSI, ADX, ATR/ATR%, Bollinger width, returns, volume ratio, relative strength changes, VIX, "
        "10-year Treasury yield, yield changes, oil price, oil changes, risk/reward, invalidation, "
        "and price zones when present. "
        "Treasury yield changes are percentage-point changes unless the JSON says otherwise; do not convert "
        "a 0.375 yield change into 37.5%. Oil and VIX changes are decimal returns unless stated otherwise. "
        "If price_zones.short_term_tactical_buy_zone is present, include it exactly in "
        "short_term_tactical_buy_zone and discuss whether it is a tactical pullback area, not a long-term zone. "
        "Do not invent price levels or indicator values. Use only the values supplied in the JSON, especially "
        "for near-term forecast target zones. "
        "Return JSON only using the requested schema. This is not financial advice.\n\n"
        f"{json.dumps(signal_json, indent=2)}"
    )
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=INTERPRETATION_SCHEMA,
        ),
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return json.loads(text)


def interpret_signal_with_gpt(signal_json: dict) -> dict:
    """Compatibility alias for the original spec name; this project uses Gemini."""
    return interpret_signal_with_gemini(signal_json)
