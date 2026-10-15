from __future__ import annotations

import json
import os

from .utils import PROJECT_ROOT, load_env_file


DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


_STRING_LIST = {"type": "array", "items": {"type": "string"}}

INTERPRETATION_SCHEMA = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string"},
        "short_term_view": {"type": "string"},
        "long_term_view": {"type": "string"},
        "plain_english_summary": {"type": "string"},
        "short_term_plan": {
            "type": "object",
            "properties": {
                "entry_zone": {"type": "string"},
                "current_position": {"type": "string"},
                "bounce_assessment": {"type": "string"},
                "stop_loss": {"type": "string"},
                "targets": {"type": "string"},
                "historical_evidence": {"type": "string"},
                "what_confirms_bounce": _STRING_LIST,
                "what_invalidates_setup": _STRING_LIST,
            },
            "required": [
                "entry_zone",
                "current_position",
                "bounce_assessment",
                "stop_loss",
                "targets",
                "historical_evidence",
                "what_confirms_bounce",
                "what_invalidates_setup",
            ],
        },
        "long_term_plan": {
            "type": "object",
            "properties": {
                "buy_zone": {"type": "string"},
                "trim_zone": {"type": "string"},
                "current_position": {"type": "string"},
                "historical_evidence": {"type": "string"},
            },
            "required": ["buy_zone", "trim_zone", "current_position", "historical_evidence"],
        },
        "key_reasons": _STRING_LIST,
        "main_risks": _STRING_LIST,
        "data_cited": _STRING_LIST,
        "not_financial_advice": {"type": "boolean"},
    },
    "required": [
        "ticker",
        "short_term_view",
        "long_term_view",
        "plain_english_summary",
        "short_term_plan",
        "long_term_plan",
        "key_reasons",
        "main_risks",
        "data_cited",
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
    skill_path = PROJECT_ROOT / "skills" / "entry_zone_interpreter" / "SKILL.md"
    if not skill_path.exists():
        return ""
    return skill_path.read_text(encoding="utf-8")


def _without_chart(signal_json: dict) -> dict:
    """Drop per-session chart rows; they add tokens without adding interpretable evidence."""
    return {key: value for key, value in signal_json.items() if key != "chart_data"}


def interpret_signal_with_gemini(signal_json: dict, model: str | None = None) -> dict:
    """
    Sends the price-zone report to Gemini and returns structured JSON.

    The default model uses Gemini 3.8 Flash. Override with GEMINI_MODEL
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
        "Interpret this price-zone report for a short-term swing trader and a long-term investor. "
        "The report is not a forecast: explain where price sits relative to the supplied zones and what the "
        "historical replays show, always comparing success_rate with baseline_success_rate. "
        "Keep the report's action codes as the short_term_view and long_term_view. "
        "Copy zones, stop, and targets exactly as supplied; do not invent price levels or probabilities. "
        "In data_cited, list the concrete values you relied on. "
        "Return JSON only using the requested schema. This is not financial advice.\n\n"
        f"{json.dumps(_without_chart(signal_json), indent=2)}"
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
