from __future__ import annotations

import json
import os

from .utils import PROJECT_ROOT, read_env_file


DEFAULT_GEMINI_MODEL = "gemini-4-argon"


_STRING_LIST = {"type": "array", "items": {"type": "string"}}
_MARKDOWN = {"type": "string"}

# Prose sections are markdown; the dashboard renders each one under its own heading.
MARKDOWN_SECTIONS = (
    "summary_markdown",
    "short_term_markdown",
    "historical_replay_markdown",
    "trend_outlook_markdown",
)

INTERPRETATION_SCHEMA = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string"},
        "short_term_view": {"type": "string"},
        "headline": {"type": "string"},
        **{section: _MARKDOWN for section in MARKDOWN_SECTIONS},
        "key_reasons": _STRING_LIST,
        "main_risks": _STRING_LIST,
        "data_cited": _STRING_LIST,
        "not_financial_advice": {"type": "boolean"},
    },
    "required": [
        "ticker",
        "short_term_view",
        "headline",
        *MARKDOWN_SECTIONS,
        "key_reasons",
        "main_risks",
        "data_cited",
        "not_financial_advice",
    ],
    "propertyOrdering": [
        "ticker",
        "short_term_view",
        "headline",
        *MARKDOWN_SECTIONS,
        "key_reasons",
        "main_risks",
        "data_cited",
        "not_financial_advice",
    ],
}


KEY_NAMES = ("GEMINI_API_KEY", "gemini", "GOOGLE_API_KEY")


def _api_key_with_source() -> tuple[str | None, str | None]:
    """Return the Gemini key and where it came from.

    Values in .env win over the process environment, so a stale key exported
    in a terminal cannot silently replace the project's key.
    """
    file_values = {key.lower(): value for key, value in read_env_file().items()}
    sources = [(f".env {name}", file_values.get(name.lower())) for name in KEY_NAMES]
    sources += [(f"environment {name}", os.environ.get(name)) for name in KEY_NAMES]
    for source, candidate in sources:
        if candidate and len(candidate.strip()) >= 20:
            return candidate.strip(), source
    return None, None


def _api_key() -> str | None:
    return _api_key_with_source()[0]


def _model_name(model: str | None) -> str:
    file_values = {key.lower(): value for key, value in read_env_file().items()}
    return model or file_values.get("gemini_model") or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL


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

    The default model is Gemini 4 Argon. Override with GEMINI_MODEL
    or --gemini-model if your account exposes a different model identifier.
    """
    key, key_source = _api_key_with_source()
    if not key:
        raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY, GOOGLE_API_KEY, or gemini in .env.")
    model_name = _model_name(model)

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError("google-genai is required for Gemini interpretation. Install dependencies first.") from exc

    skill_instructions = _skill_instructions()
    prompt = (
        "You are writing the AI explanation panel of a stock dashboard. Follow this interpretation and "
        "formatting guide exactly:\n\n"
        f"{skill_instructions}\n\n"
        "Return JSON only, using the requested schema. Every *_markdown field is GitHub-flavored markdown "
        "and must follow the section template in the guide. This is not financial advice.\n\n"
        "Price-zone report JSON:\n"
        f"{json.dumps(_without_chart(signal_json), indent=2)}"
    )

    from google.genai import errors

    client = genai.Client(api_key=key)
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=INTERPRETATION_SCHEMA,
            ),
        )
    except errors.APIError as exc:
        # Name the key's source (never the key) so a wrong key is easy to trace.
        raise RuntimeError(f"Gemini request failed using the key from {key_source} with model {model_name}: {exc.message or exc}") from exc
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return json.loads(text)


def interpret_signal_with_gpt(signal_json: dict) -> dict:
    """Compatibility alias for the original spec name; this project uses Gemini."""
    return interpret_signal_with_gemini(signal_json)
