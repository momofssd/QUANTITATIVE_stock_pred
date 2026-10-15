from __future__ import annotations

import json
import os

from ..utils import PROJECT_ROOT, load_env_file


DEFAULT_GPT_MODEL = "gpt-6-astra"


INTERPRETATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
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
        "five_session_feedback": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "outlook": {"type": "string"},
                "range_probabilities": {"type": "array", "items": {"type": "string"}},
                "price_actions": {"type": "array", "items": {"type": "string"}},
                "uncertainty": {"type": "string"},
            },
            "required": ["outlook", "range_probabilities", "price_actions", "uncertainty"],
        },
        "near_term_price_forecast": {
            "type": "object",
            "additionalProperties": False,
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
            "additionalProperties": False,
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
            "additionalProperties": False,
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
        "five_session_feedback",
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
        os.getenv("OPENAI_API_KEY"),
        os.getenv("GPT_API_KEY"),
        os.getenv("GPT"),
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


def interpret_signal_with_gpt(signal_json: dict, model: str | None = None) -> dict:
    """
    Send quant JSON to OpenAI and return structured JSON.

    The default model is GPT-6 Astra. Override it with OPENAI_MODEL, GPT_MODEL,
    or the model argument.
    """
    key = _api_key()
    if not key:
        raise RuntimeError("Missing OpenAI API key. Set OPENAI_API_KEY, GPT_API_KEY, or GPT in .env.")
    model_name = model or os.getenv("OPENAI_MODEL") or os.getenv("GPT_MODEL") or DEFAULT_GPT_MODEL

    try:
        from openai import OpenAI, OpenAIError
    except ImportError as exc:
        raise ImportError("openai is required for GPT interpretation. Install dependencies first.") from exc

    skill_instructions = _skill_instructions()
    instructions = (
        "Use these local skill instructions as the interpretation policy:\n\n"
        f"{skill_instructions}\n\n"
        "Interpret this quantitative long-term swing trading JSON. "
        "Do not merely repeat the headline decision. Analyze the actual technical data in "
        "technical_indicators, relative_strength, macro_context, risk_management, current_market, "
        "price_zones, five_day_forecast, short_term_trade, near_term_forecast, and reasoning. "
        "Explain supplied five_day_forecast.session_forecasts, price_distribution bins and price_action_plan in your summary and conditions. "
        "Session bins describe closing-price probabilities, never intraday touch or bounce probabilities. "
        "Populate five_session_feedback with supplied session probabilities, conditional levels and uncertainty; if absent, state unavailable and return empty lists. "
        "Keep empirical bin probabilities distinct from the separately calibrated gain classifier; flag disagreement. "
        "Use only supplied action levels; a support touch does not confirm a bounce. Profit taking assumes an existing long above its cost basis after costs. "
        "When five_day_forecast is present, use its predicted_direction, confidence, and possible price as the "
        "explicit five-trading-day model forecast. Respect price_estimate_basis: the current version uses "
        "unconditional regression across gains and losses. Do not replace the estimate with an invented level. "
        "Explain probability_of_gain separately from final held-out direction_accuracy and regression errors. "
        "The entry reference is a daily close, not a live executable quote, and gains exclude costs. "
        "Report purchase_evidence and warnings; do not promote a bullish estimate into a buy when validation "
        "does not support it. Nominal interval coverage is not a guarantee; compare it with observed test coverage. "
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
        "Return JSON only using the requested schema. This is not financial advice."
    )
    client = OpenAI(api_key=key)
    try:
        response = client.responses.create(
            model=model_name,
            instructions=instructions,
            input=json.dumps(signal_json, indent=2),
            store=False,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "trade_interpretation",
                    "strict": True,
                    "schema": INTERPRETATION_SCHEMA,
                }
            },
        )
    except OpenAIError as exc:
        raise RuntimeError(f"OpenAI interpretation failed: {exc}") from exc

    text = getattr(response, "output_text", None)
    if not text:
        raise RuntimeError("GPT-6 Astra returned an empty response.")
    return json.loads(text)


def interpret_signal_with_gemini(signal_json: dict, model: str | None = None) -> dict:
    """Deprecated compatibility alias; interpretation now uses OpenAI."""
    return interpret_signal_with_gpt(signal_json, model=model)
