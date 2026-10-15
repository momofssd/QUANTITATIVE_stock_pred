"""Optional GPT-6 Astra interpretation of an already computed quantitative signal."""

from .gemini import interpret_signal_with_gemini, interpret_signal_with_gpt

__all__ = ["interpret_signal_with_gpt", "interpret_signal_with_gemini"]
